#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor

import numpy as np
import math

from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple, Union

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Upsample(SimNN.Module):
    def __init__(self, name, size=None, scale_factor=2, mode='nearest'):
        super().__init__()
        self.name = name
        assert scale_factor == 2, f"Err upsample.scale_factor({scale_factor}) != 2"
        assert mode == 'nearest', f"Err upsample.mode({mode}) = nearest"
        self.resize = F.Resize(name + '.upsample', scale_factor=scale_factor, mode=mode,
                               nearest_mode='floor',
                               coordinate_transformation_mode='asymmetric')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, x):
        return self.resize(x)

class SiLU(SimNN.Module):
    def __init__(self, name):
        super().__init__()
        self.name    = name
        self.sigmoid = F.Sigmoid(self.name + '.sigmoid')
        self.mul     = F.Mul(self.name + '.mul')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, x):
        y = self.sigmoid(x)
        z = self.mul(x, y)
        return z

class Conv(SimNN.Module):
    def __init__(self, name, c1, c2, k=1, s=1, p=None, g=1, act='SiLU'):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.name         = name
        self.in_channels  = c1
        self.out_channels = c2
        self.kernel_size  = k
        self.stride       = s
        self.padding      = p
        self.group        = g
        assert act in ['SiLU', 'LeakyReLU(0.1)', 'Identity'], f"Illegal activation=({act})!!"

        self.conv = F.Conv2d(self.name + '.conv', c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), groups=g, bias=False)
        self.bn   = F.BatchNorm2d(self.name + '.bn', c2)
        self.act  = SiLU(self.name + '.silu') if act == 'SiLU' else (F.LeakyReLU(self.name + '.leakyRelu', alpha=0.1) \
                                                                       if act == 'LeakyReLU(0.1)' \
                                                                       else F.Identity(self.name + '.identity'))

        super().link_op2module()

    def __call__(self, x):
        y = self.conv(x)
        z = self.bn(y)
        o = self.act(z)
        return o

    def analytical_param_count(self, lvl):
        #this assumes bias=False, because we use BatchNorm2d
        conv_params = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size // self.group
        bn_params = 2 * self.out_channels #(weight, bias)

        return conv_params + bn_params

class Concat(SimNN.Module):
    def __init__(self, name, dimension=1):
        super(Concat, self).__init__()
        self.name   = name
        self.axis   = dimension
        self.concat = F.ConcatX(self.name, axis=dimension)
        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, *x):
        if len(x) == 1 and isinstance(x[0], list):
            x_tmp = x[0]
            return self.concat(*x_tmp)
        else:
            return self.concat(*x)

class Bottleneck(SimNN.Module):
    def __init__(self, name, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.name = name
        self.shortcut = shortcut and c1 == c2
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(name + '.cv1', c1, c_, 3, 1)
        self.cv2 = Conv(name + '.cv2', c_, c2, 3, 1, g=g)
        self.add = F.Add(name + '.add')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return self.cv1.analytical_param_count(lvl+1) + self.cv2.analytical_param_count(lvl+1)

    def __call__(self, x):
        y = self.cv2(self.cv1(x))
        if self.shortcut:
            y = self.add(x, y)
        return y

class C2f(SimNN.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, name, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.name = name
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(name + '.cv1', c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(name + '.cv2', 2 * self.c + n * self.c, c2, 1, 1)
        
        mlist = []
        for i in range(n):
            mlist.append(Bottleneck(f"{name}.m.{i}", self.c, self.c, shortcut, g, e=1.0))
        self.m = SimNN.ModuleList(mlist)
        
        self.split = F.SplitOpHandle(name + '.split', axis=1, count=2)
        self.concat = Concat(name + '.concat')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return self.cv1.analytical_param_count(lvl+1) + self.cv2.analytical_param_count(lvl+1) + \
               sum(m.analytical_param_count(lvl+1) for m in self.m)

    def __call__(self, x):
        y_cv1 = self.cv1(x)
        y1, y2 = self.split(y_cv1)  # y1 has self.c, y2 has self.c

        list_for_concat = [y1] 
        list_for_concat.append(y2)
        
        current_bottleneck_input = y2
        for m_module in self.m: # self.m is SimNN.ModuleList of Bottleneck instances
            bottleneck_output = m_module(current_bottleneck_input)
            list_for_concat.append(bottleneck_output)
            current_bottleneck_input = bottleneck_output # Output of current bottleneck is input to next

        return self.cv2(self.concat(*list_for_concat))

class SPPF(SimNN.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer
    def __init__(self, name, c1, c2, k=5):
        super().__init__()
        self.name = name
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(name + '.cv1', c1, c_, 1, 1)
        self.cv2 = Conv(name + '.cv2', c_ * 4, c2, 1, 1)
        # Define three distinct MaxPool2d operations
        self.m1 = F.MaxPool2d(name + '.m1', kernel_size=k, stride=1, padding=k // 2)
        self.m2 = F.MaxPool2d(name + '.m2', kernel_size=k, stride=1, padding=k // 2)
        self.m3 = F.MaxPool2d(name + '.m3', kernel_size=k, stride=1, padding=k // 2)
        self.concat = Concat(name + '.concat')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return self.cv1.analytical_param_count(lvl+1) + self.cv2.analytical_param_count(lvl+1)

    def __call__(self, x):
        x = self.cv1(x)  # Output of cv1, input to pooling and concatenation
        y1 = self.m1(x)
        y2 = self.m2(y1)
        y3 = self.m3(y2)
        return self.cv2(self.concat(x, y1, y2, y3))

class DFL(SimNN.Module):
    # Distribution Focal Loss (DFL) module
    def __init__(self, name, c1=16): # c1 is in_channels for the internal conv
        super().__init__()
        self.name = name
        self.in_channels = c1
        self.out_channels = 1  # DFL's conv outputs 1 channel
        self.kernel_size = 1   # DFL's conv uses 1x1 kernel
        self.stride = 1        # DFL's conv uses stride 1
        self.groups = 1        # Assuming default groups=1 for the conv
        self.bias = False      # DFL's conv is specified with bias=False

        self.conv = F.Conv2d(
            f"{name}.conv",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            bias=self.bias,
            groups=self.groups
        )
        
        super().link_op2module()

    def analytical_param_count(self, lvl):
        # Calculate weight parameters for the Conv2d layer
        # Formula: out_channels * (in_channels / groups) * kernel_h * kernel_w
        weight_params = self.out_channels * (self.in_channels // self.groups) * self.kernel_size * self.kernel_size
        
        # Bias parameters would be self.out_channels if self.bias were True.
        # Since self.bias is False, bias_params is 0.
        
        return weight_params

    def __call__(self, x):
        return self.conv(x)

class DetectV8(SimNN.Module):
    def __init__(self, name, nc=80, ch=(), bs=1):
        super().__init__()
        self.name = name
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.bs = bs
        
        self.cv2_layer_sets = []
        self.cv3_layer_sets = []

        # Initialize the box prediction layers
        for i, c_in in enumerate(ch):
            cv2_0 = Conv(f"{name}.cv2.{i}.0", c_in, 64, 3, 1)
            cv2_1 = Conv(f"{name}.cv2.{i}.1", 64, 64, 3, 1)
            cv2_2 = F.Conv2d(f"{name}.cv2.{i}.2", 64, 64, kernel_size=1, stride=1, bias=True)  # Default bias=True
            
            setattr(self, f"cv2_{i}_0", cv2_0)
            setattr(self, f"cv2_{i}_1", cv2_1)
            setattr(self, f"cv2_{i}_2", cv2_2)
            self.cv2_layer_sets.append((cv2_0, cv2_1, cv2_2))

        # Initialize the class prediction layers
        for i, c_in in enumerate(ch):
            cv3_0 = Conv(f"{name}.cv3.{i}.0", c_in, 128, 3, 1)
            cv3_1 = Conv(f"{name}.cv3.{i}.1", 128, 128, 3, 1)
            cv3_2 = F.Conv2d(f"{name}.cv3.{i}.2", 128, self.nc, kernel_size=1, stride=1, bias=True) # Default bias=True

            setattr(self, f"cv3_{i}_0", cv3_0)
            setattr(self, f"cv3_{i}_1", cv3_1)
            setattr(self, f"cv3_{i}_2", cv3_2)
            self.cv3_layer_sets.append((cv3_0, cv3_1, cv3_2))
        
        self.dfl = DFL(name + '.dfl')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        total = self.dfl.analytical_param_count(lvl + 1) # DFL was fixed previously
        
        # Parameters for F.Conv2d ops in cv2_layer_sets
        # cv2_2_op is F.Conv2d(name, 64, 64, kernel_size=1, stride=1, bias=True)
        # It's created with in_channels=64, out_channels=64, kernel_size=1, groups=1 (default), bias=True.
        cv2_2_op_in_channels = 64
        cv2_2_op_out_channels = 64
        cv2_2_op_kernel_size = 1 
        cv2_2_op_groups = 1
        
        # Calculate parameters for one such F.Conv2d op
        cv2_2_weight_params = cv2_2_op_out_channels * (cv2_2_op_in_channels // cv2_2_op_groups) * cv2_2_op_kernel_size * cv2_2_op_kernel_size
        cv2_2_bias_params = cv2_2_op_out_channels # bias=True was used in its creation
        
        for cv2_0, cv2_1, _ in self.cv2_layer_sets: # cv2_2_op itself is not used from the loop for param count
            total += cv2_0.analytical_param_count(lvl + 1)
            total += cv2_1.analytical_param_count(lvl + 1)
            total += cv2_2_weight_params + cv2_2_bias_params

        # Parameters for F.Conv2d ops in cv3_layer_sets
        # cv3_2_op is F.Conv2d(name, 128, self.nc, kernel_size=1, stride=1, bias=True)
        # It's created with in_channels=128, out_channels=self.nc, kernel_size=1, groups=1 (default), bias=True.
        cv3_2_op_in_channels = 128
        cv3_2_op_out_channels = self.nc # self.nc is an instance attribute
        cv3_2_op_kernel_size = 1
        cv3_2_op_groups = 1
        
        # Calculate parameters for one such F.Conv2d op
        cv3_2_weight_params = cv3_2_op_out_channels * (cv3_2_op_in_channels // cv3_2_op_groups) * cv3_2_op_kernel_size * cv3_2_op_kernel_size
        cv3_2_bias_params = cv3_2_op_out_channels # bias=True was used in its creation
                
        for cv3_0, cv3_1, _ in self.cv3_layer_sets: # cv3_2_op itself is not used from the loop for param count
            total += cv3_0.analytical_param_count(lvl + 1)
            total += cv3_1.analytical_param_count(lvl + 1)
            total += cv3_2_weight_params + cv3_2_bias_params
                
        return total

    def __call__(self, x):
        # Process predictions
        box_outputs = []
        cls_outputs = []
        
        for i in range(self.nl):
            x_head = x[i]
            
            # Box prediction branch
            cv2_0, cv2_1, cv2_2_op = self.cv2_layer_sets[i]
            out_cv2 = cv2_0(x_head)
            out_cv2 = cv2_1(out_cv2)
            out_cv2 = cv2_2_op(out_cv2)
            box_outputs.append(out_cv2)
            
            # Class prediction branch
            cv3_0, cv3_1, cv3_2_op = self.cv3_layer_sets[i]
            out_cv3 = cv3_0(x_head) # Input to cv3 is also x_head (original x[i])
            out_cv3 = cv3_1(out_cv3)
            out_cv3 = cv3_2_op(out_cv3)
            cls_outputs.append(out_cv3)
        
        # Apply DFL in a real implementation would go here
        # For simplicity in simulation, we'll just return the outputs
        return box_outputs, cls_outputs

def parse_model(d, ch):
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    
    layers: List[SimNN.Module] = []
    save  : List[int]          = []
    c2    : List[int]          = ch[-1]
    
    # Define module mappings
    module_dict = {
        'Conv': Conv,
        'C2f': C2f,
        'SPPF': SPPF,
        'nn.Upsample': Upsample,
        'Concat': Concat,
        'Detect': DetectV8
    }
    
    # Build model
    for i, (f, n, mtype, args) in enumerate(d['backbone'] + d['head']):
        mname = f"{mtype}_{i}"  # module name
        
        # Evaluate module arguments
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        
        if mtype in ['Conv', 'C2f', 'SPPF']:
            c1, c2 = ch[f], args[0] if len(args) > 0 else ch[f]
            c2 = make_divisible(c2 * gw, 8)
            if mtype == 'Conv':
                args = [c1, c2] + args[1:] if len(args) > 0 else [c1, c2]
            elif mtype == 'C2f':
                shortcut = args[1] if len(args) > 1 else False
                args = [c1, c2, n, shortcut]
            elif mtype == 'SPPF':
                k = args[1] if len(args) > 1 else 5
                args = [c1, c2, k]
            
            m = module_dict[mtype](mname, *args)
        elif mtype == 'nn.Upsample':
            args = [mname] + args if args else [mname]
            m = module_dict[mtype](*args)
        elif mtype == 'Concat':
            c2 = sum(ch[x] for x in f)
            args = [mname] + args if args else [mname]
            m = module_dict[mtype](*args)
        elif mtype == 'Detect':
            args = [mname, nc, [ch[x] for x in f]]
            m = module_dict[mtype](*args)
        else:
            raise ValueError(f"Unknown module type: {mtype}")
            
        m.i, m.f, m.type, m.np = i, f, mname, m.analytical_param_count(0)
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m)
        if i == 0:
            ch = []
        ch.append(c2)
        
    return layers, sorted(save)

def run_model(model_layers, save_list, model_input):
    y : List[SimTensor|None] = []
    x = model_input
    for m in model_layers:
        if hasattr(m, 'f') and m.f != -1:  # if not from prev layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x)  # run
        y.append(x if m.i in save_list else None)  # save output
    return x

class YOLO8S(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.bs = cfg['bs']
        self.in_channels = cfg.get('in_channels', 3)
        self.in_resolution = cfg.get('in_resolution', 640)  # 2x min stride
        
        # Get config path based on cfg or use default
        model_variant = name.split('-')[0] if '-' in name else name
        if 'yaml_cfg_path' in cfg:
            yaml_cfg_path = cfg['yaml_cfg_path']
        else:
            # For variants like yolov8s, yolov8n, etc.
            default_cfg_dir = 'config/yolov8_cfgs'
            yaml_cfg_path = os.path.join(default_cfg_dir, f"{model_variant}.yaml")
            if not os.path.exists(yaml_cfg_path):
                yaml_cfg_path = os.path.join(default_cfg_dir, "yolov8s.yaml")  # Fallback to yolov8s
        
        # Parse the YAML config
        yaml_cfg = parse_yaml(yaml_cfg_path)
        _l, _s = parse_model(yaml_cfg, ch=[self.in_channels])
        self.layers = _l  # layers
        self.save = _s  # savelist
        
        m = self.layers[-1]
        assert isinstance(m, DetectV8), f"Last OP should be DetectV8!! {m.name}"
        
        # Record submodules
        for LL in self.layers:
            self._submodules[LL.name] = LL
        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
            'yolo_input': F._from_shape('yolo_input',
                                        [self.bs, self.in_channels, self.in_resolution, self.in_resolution])
        }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self, lvl=0):
        return sum([m.analytical_param_count(lvl+1) for m in self.layers])

    def __call__(self):
        model_input = self.input_tensors['yolo_input']
        model_output = run_model(self.layers, self.save, model_input)
        return model_output

if __name__ == '__main__':
    cfg_dir = './config/yolov8_cfgs'
    cfgs = [
        'yolov8n.yaml',
        'yolov8s.yaml',
        'yolov8m.yaml',
        'yolov8l.yaml',
        'yolov8x.yaml',
        'yolov8x6.yaml',
    ]
    ch = 3  # input channels for all variants = 3
    for cfg_file in cfgs:
        print(f"Processing {cfg_file}....")
        cfg_path = os.path.join(cfg_dir, cfg_file)
        yolo_obj = YOLO8S(cfg_file.replace('.yaml', ''), {'bs': 1, 'yaml_cfg_path': cfg_path, 'in_channels': ch})
        param_count = yolo_obj.analytical_param_count()
        print(f"    #params= {param_count/1e6:.2f}M")
        yolo_obj.create_input_tensors()
        print("    input shape=", yolo_obj.input_tensors['yolo_input'].shape)
        yolo_out = yolo_obj()
        box_outputs, cls_outputs = yolo_out
        print("    box output shapes=", [y.shape for y in box_outputs])
        print("    class output shapes=", [y.shape for y in cls_outputs])
        gg = yolo_obj.get_forward_graph()
        out_onnx_file = cfg_file.replace('.yaml', '.onnx')
        print("    exporting to onnx:", out_onnx_file)
        gg.graph2onnx(out_onnx_file)
        print()