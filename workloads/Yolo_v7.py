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
from typing import List

def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Upsample(SimNN.Module):

    def __init__(self, name, size, scale_factor, mode):
        super().__init__()
        self.name   = name
        '''
            nn.Upsample == onnx.resize
            for scale_factor = 2, mode = nearest
            equivalent onnx.resize
            inputs: X = input-ND-tensor
            roi   : []
            scales: [1.0, 1.0, ...., 2.0, 2.0] : last 2 dims in N-D are 2.0
            attribs:
                mode: nearest
                coordinate_transformation_mode: asymmetric
                nearest_mode: floor
        '''
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

        self.conv         = F.Conv2d(self.name + '.conv', c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), groups=g, bias=False)
        self.bn           = F.BatchNorm2d(self.name + '.bn', c2)
        self.act = SiLU(self.name + '.silu') if act == 'SiLU' else (F.LeakyReLU(self.name + '.leakyRelu', alpha=0.1) \
                                                                        if act == 'LeakyReLU(0.1)' \
                                                                        else F.Identity(self.name + '.identity'))


        super().link_op2module()

    def __call__(self, x):
        #return self.act(self.bn(self.conv(x)))
        y = self.conv(x)
        z = self.bn(y)
        o = self.act(z)
        return o

    def analytical_param_count(self, lvl):
        #this assumes bias=False, because we use BatchNorm2d
        conv_params = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size
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

class MP(SimNN.Module):
    def __init__(self, name, k=2):
        super().__init__()
        self.name = name
        self.m    = F.MaxPool2d(self.name, kernel_size=k, stride=k)
        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, x):
        return self.m(x)

class SP(SimNN.Module):
    def __init__(self, name, k=3, s=1):
        super().__init__()
        self.name = name
        self.m    = F.MaxPool2d(name, kernel_size=k, stride=s, padding=k // 2)
        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, x):
        return self.m(x)

class SPPCSPC(SimNN.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, name, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        self.name = name
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(name + '.cv1', c1, c_, 1, 1)
        self.cv2 = Conv(name + '.cv2', c1, c_, 1, 1)
        self.cv3 = Conv(name + '.cv3', c_, c_, 3, 1)
        self.cv4 = Conv(name + '.cv4', c_, c_, 1, 1)
        self.m   = F.SimOpHandleList([F.MaxPool2d(name + f'.mpool_{ii}', kernel_size=x, stride=1, padding=x // 2) for ii, x in enumerate(k)])
        self.cv5 = Conv(name + '.cv5', 4 * c_, c_, 1, 1)
        self.cv6 = Conv(name + '.cv6', c_, c_, 3, 1)
        self.cv7 = Conv(name + '.cv7', 2 * c_, c2, 1, 1)

        self.concat1 = F.ConcatX(name + 'concat1', axis=1)
        self.concat2 = F.ConcatX(name + 'concat2', axis=1)

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return self.cv1.analytical_param_count(lvl+1) + \
               self.cv2.analytical_param_count(lvl+1) + \
               self.cv3.analytical_param_count(lvl+1) + \
               self.cv4.analytical_param_count(lvl+1) + \
               self.cv5.analytical_param_count(lvl+1) + \
               self.cv6.analytical_param_count(lvl+1) + \
               self.cv7.analytical_param_count(lvl+1)

    def __call__(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        p1 = [m(x1) for m in self.m]
        inlist = [x1] + p1
        x2 = self.concat1(*inlist)
        y1 = self.cv6(self.cv5(x2))
        y2 = self.cv2(x)
        z1 = self.concat2(y1, y2)
        return self.cv7(z1)

class RepConv(SimNN.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, name, c1, c2, k=3, s=1, p=None, g=1, act='SiLU', deploy=False):
        super().__init__()
        self.name = name

        self.deploy       = deploy
        self.groups       = g
        self.in_channels  = c1
        self.out_channels = c2
        self.kernel_size  = k

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        assert act in ['SiLU', 'LeakyReLU(0.1)', 'Identity'], f"Illegal activation=({act})!!"
        self.act = SiLU(self.name + '.silu') if act == 'SiLU' else (F.LeakyReLU(self.name + '.leakyRelu', alpha=0.1) \
                                                                        if act == 'LeakyReLU(0.1)' \
                                                                        else F.Identity(self.name + '.identity'))


        if deploy:
            self.rbr_reparam = F.Conv2d(name + '.rbr_reparam', c1, c2, kernel_size=k, stride=s,
                                        padding=autopad(k, p), groups=g, bias=True)

        else:
            #self.rbr_identity = (F.BatchNorm2d(name + '.rbr_identity', num_features=c1) if c2 == c1 and s == 1 else None)
            self.rbr_identity = (F.BatchNorm2d(name + '.rbr_identity', c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = F.SimOpHandleList([
                F.Conv2d(name + '.rbr_dense.conv', c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), groups=g, bias=False),
                #F.BatchNorm2d(name + '.rbr_dense.bn', num_features=c2),
                F.BatchNorm2d(name + '.rbr_dense.bn', c2),
                ])

            self.rbr_1x1 = F.SimOpHandleList([
                F.Conv2d(name + '.rbr_1x1.conv', c1, c2, kernel_size=1, stride=s, padding=padding_11, groups=g, bias=False),
                #F.BatchNorm2d(name + '.rbr_1x1.bn', num_features=c2),
                F.BatchNorm2d(name + '.rbr_1x1.bn', c2),
                ])
            self.add1 = F.Add(name + '.rbr_add1')
            self.add2 = F.Add(name + '.rbr_add2')

        super().link_op2module()

    def analytical_param_count(self, lvl):
        rbr_reparam_C  = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size + self.out_channels
        rbr_identity_C = 2 * self.in_channels
        rbr_dense_C    = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size + 2 * self.out_channels
        rbr_1x1_C      = self.in_channels * self.out_channels + 2 * self.out_channels

        if hasattr(self, "rbr_reparam"):
            return rbr_reparam_C
        else:
            if self.rbr_identity is None:
                return rbr_dense_C + rbr_1x1_C
            else:
                return rbr_dense_C + rbr_1x1_C + rbr_identity_C

    def __call__(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            z2 = self.add1(self.rbr_dense(inputs), self.rbr_1x1(inputs))
        else:
            z1 = self.add1(self.rbr_dense(inputs), self.rbr_1x1(inputs))
            z2 = self.add2(z1, self.rbr_identity(inputs))

        return self.act(z2)

#### from yolo.py ###
class Detect(SimNN.Module):
    stride      = None   # strides computed during build
    export      = False  # onnx export
    end2end     = False
    include_nms = False
    concat      = False

    def __init__(self, name, nc=80, anchors=(), ch=(), bs=1):  # detection layer
        super().__init__()
        self.name = name
        self.nc   = nc      # number of classes
        self.no   = nc + 5  # number of outputs per anchor
        self.nl   = len(anchors)  # number of detection layers
        self.na   = len(anchors[0]) // 2  # number of anchors
        self.ch   = ch
        self.bs   = bs

        #number of detection layers should be equal to input ch list
        assert self.nl == len(ch), f"#detection-layers{self.nl} != #ch{len(ch)}"

        self.grid = [0] * self.nl  # init grid


        oplist = []
        for ii, x in enumerate(ch):
            oplist.append(F.Conv2d(name + f".conv_{ii}", x, self.no * self.na, 1))
            oplist.append(F.Reshape(name + f".reshape_{ii}"))
            oplist.append(F.Transpose(name + f".transpose_{ii}", perm=[0,1,3,4,2]))
        self.m = F.SimOpHandleList(oplist)

        self.training = True #By default, self.training is True in PyTorch

        super().link_op2module()

    def analytical_param_count(self, lvl):
        #just get the params for Conv2d, +1 for Bias=True
        return sum([(x + 1) * self.no * self.na for x in self.ch])

    def __call__(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[3*i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            _tmp_x = F._from_data(f'tmp_x_{i}', is_const=True,
                                  data=np.array([bs, self.na, self.no, ny, nx],
                                                dtype=np.int64))
            _tmp_x.op_in.append(self.m[3*i+1].name)
            self._tensors[_tmp_x.name] = _tmp_x
            x[i] = self.m[3*i+1](x[i], _tmp_x)
            x[i] = self.m[3*i+2](x[i])

            if not self.training:  # inference
                #TODO: implement this!!
                pass
                '''
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
                '''

        # TODO: implement this
        #if self.training:
        #    out = x
        #elif self.end2end:
        #    out = torch.cat(z, 1)
        #elif self.include_nms:
        #    z = self.convert(z)
        #    out = (z, )
        #elif self.concat:
        #    out = torch.cat(z, 1)
        #else:
        #    out = (torch.cat(z, 1), x)
        out = x
        return out

    '''
    TODO: Implement this!!
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)
    '''

def parse_model(d, ch):
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers: List[SimNN.Module] = []
    save  : List[int]          = []
    c2    : List[int]          = ch[-1]
    argstbl = {'None': None, 'nn.LeakyReLU(0.1)': 'LeakyReLU(0.1)', 'nc': nc, 'anchors': anchors}
    optbl = {
            'Conv'        : Conv,
            'Concat'      : Concat,
            'MP'          : MP,
            'SP'          : SP,
            'SPPCSPC'     : SPPCSPC,
            'RepConv'     : RepConv,
            'nn.Upsample' : Upsample,
            'nn.Conv2d'   : F.Conv2d,
            'Detect'      : Detect,
            }
    for bb in ['backbone', 'head']:
        for jj, (f, n, mname, args) in enumerate(d[bb]):  # from, number, module, args
            i = len(d['backbone']) + jj if bb == 'head' else jj
            assert isinstance(mname, str), f"KAPPA: {mname}"
            m = optbl[mname] #type: ignore
            for j, a in enumerate(args):
                try:
                    args[j] = argstbl[a] if isinstance(a, str) else a  # eval strings
                except:
                    pass

            n = max(round(n * gd), 1) if n > 1 else n  # depth gain
            if m in [ F.Conv2d, Conv, SPPCSPC, RepConv]: #'REPCONV']:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
                if m in [SPPCSPC,]:
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is F.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum([ch[x] for x in f])
            elif m in [Detect]: #, IDetect, IAuxDetect, IBin, IKeypoint]:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]

            assert n == 1, f"BATA: {n}"
            m_ = m(f"{bb}.{mname}_{jj}", *args) #type: ignore
            param_count = m_.analytical_param_count(0)

            #t = str(m)[8:-2].replace('__main__.', '')  # module type
            #np = sum([x.numel() for x in m_.parameters()])  # number params
            #m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
            m_.i, m_.f, m_.type, m_.np = i, f, m_.name, param_count  # attach index, 'from' index, type, number params
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0: ch = []
            ch.append(c2)

    return layers, sorted(save)

def run_model(model_layers, save_list, model_input):
    y : List[SimTensor|None] = []
    x = model_input
    for m in model_layers:
        if m.name == 'head.dummy_detect':
            continue

        if m.f != -1: # if not from prev layer
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
        x = m(x) # run
        y.append(x if m.i in save_list else None) # save output
    return x

'''
def _initialize_biases(model, cf=None):  # initialize biases into Detect(), cf is class frequency
    # https://arxiv.org/abs/1708.02002 section 3.3
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
    m = model[-1]  # Detect() module
    for mi, s in zip(m.m, m.stride):  # from
        b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
        b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
        b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
        mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
'''

class YOLO7(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name          = name
        self.bs            = cfg['bs']
        self.yaml_cfg_path = cfg['yaml_cfg_path']
        self.in_channels   = cfg.get('in_channels', 3)
        self.in_resolution = cfg.get('in_resolution', 640)   #2x min stride

        yaml_cfg    = parse_yaml(self.yaml_cfg_path)
        _l, _s      = parse_model(yaml_cfg, ch=[self.in_channels])
        self.layers = _l #layers
        self.save   = _s #savelist

        m = self.layers[-1]
        assert isinstance(m, Detect), f"Last OP should be Detect!! {m.name}"

        #We create submodules dynamically in this constructor via parse_model
        # so setattr is unable to register these. Below code is to explicitly
        # record these submodules
        for LL in self.layers: self._submodules[LL.name] = LL
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
        model_input  = self.input_tensors['yolo_input']
        model_output = run_model(self.layers, self.save, model_input)
        return model_output

if __name__ == '__main__':
    cfg_dir = './config/yolov7_cfgs/deploy'
    cfgs = [
            'yolov7-tiny.yaml',
            'yolov7-tiny-silu.yaml',
            'yolov7.yaml',
            'yolov7x.yaml',
            #thee variants require additional custom classes like ReOrg etc.
            #'yolov7-d6.yaml',
            #'yolov7-e6.yaml',
            #'yolov7-w6.yaml',
            #'yolov7-e6e.yaml',
            ]
    ch = 3 #input channels for all variants = 3
    for cfg_file in cfgs:
        print(f"Processing {cfg_file}....")
        cfg_path = os.path.join(cfg_dir, cfg_file)
        yolo_obj = YOLO7('yolov7', {'bs': 1, 'yaml_cfg_path': cfg_path, 'in_channels': ch})
        param_count = yolo_obj.analytical_param_count()
        print(f"    #params= {param_count/1e6:.2f}M")
        yolo_obj.create_input_tensors()
        print("    input shape=", yolo_obj.input_tensors['yolo_input'].shape)
        yolo_out = yolo_obj()
        print("    output shapes=", [y.shape for y in yolo_out])
        gg = yolo_obj.get_forward_graph()
        out_onnx_file = cfg_file.replace('.yaml', '.onnx')
        print("    exporting to onnx:", out_onnx_file)
        gg.graph2onnx(out_onnx_file)
        print()
