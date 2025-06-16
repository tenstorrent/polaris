#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import logging

class BasicBlock(SimNN.Module):
    expansion = 1
    def __init__(self, name, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.name         = name
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride

        assert i_downsample is None, f"downsample in BasicBlock!!"

        self.op_blk = F.SimOpHandleList([
                F.Conv2d(name + '.conv1', in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                F.BatchNorm2d(name + '.bn1', out_channels),
                F.Conv2d(name + '.conv2', out_channels, out_channels, kernel_size=3, padding=1, stride=stride),
                F.BatchNorm2d(name + '.bn2', out_channels),
                F.Relu(name + '.relu1')
                ])
        self.relu = F.Relu(name + f'.relu2')

        super().link_op2module()

    def __call__(self, x):
        y = x + self.op_blk(x)
        z = self.relu(y)
        return z

class Bottleneck(SimNN.Module):
    expansion = 4
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.in_channels  = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.stride       = cfg.get('stride', 1)
        self.downsample   = cfg.get('downsample', None)

        conv_dims = [
                #IC, OC, K, P, S
                (self.in_channels,  self.out_channels, 1, 0, 1),
                (self.out_channels, self.out_channels, 3, 1, self.stride),
                (self.out_channels, self.out_channels*Bottleneck.expansion, 1, 0, 1),
                ]
        oplist = []
        for i, (ic, oc, k, p, s) in enumerate(conv_dims):
            conv = F.Conv2d(self.name + f'.conv{i}', ic, oc, kernel_size=k, padding=p, stride=s)
            bn   = F.BatchNorm2d(self.name + f'.bn{i}', oc)
            oplist += [conv, bn]

        self.op_blk  = F.SimOpHandleList(oplist)
        self.relu    = F.Relu(self.name + f'.relu')

        if self.downsample is not None:
            xi = self.downsample['in_channels']
            xo = self.downsample['out_channels']
            xs = self.downsample['stride']
            self.conv_ds = F.Conv2d(self.name + '.conv_ds', xi, xo, kernel_size=1, padding=0, stride=xs)
            self.bn_ds   = F.BatchNorm2d(self.name + '.bn_ds', xo)

        super().link_op2module()

    def __call__(self, x):
        y = self.op_blk(x)
        if self.downsample is None:
            z = x + y
        else:
            x = self.conv_ds(x)
            x = self.bn_ds(x)
            z = x + y
        w = self.relu(z)
        return w

class ResNet(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name           = name
        self.in_channels    = cfg.get('in_channels',   64)
        self.bs             = cfg.get('bs',             1)
        self.num_channels   = cfg.get('num_channels',   3)
        self.init_stride    = cfg.get('init_stride',    2)
        self.img_height     = cfg.get('img_height',   224)
        self.img_width      = cfg.get('img_width',    224)
        self.num_classes    = cfg.get('num_classes', 1000)
        self.layers         = cfg.get('layers', [3,4,6,3])
        self.out_indices    = cfg.get('out_indices', [0,1,2,3])

        layers_msg = f"ResNet.layers should be [int, int, int, int] with each member > 0: {self.layers}"
        assert isinstance(self.layers, list) and \
               len(self.layers) == 4 and \
               all([isinstance(x, int) and x > 0 for x in self.layers]), layers_msg

        #ops
        self.conv0    = F.Conv2d(self.name +'.conv0', self.num_channels, self.in_channels,
                                 kernel_size=7, stride=2, padding=3)
        self.bn0      = F.BatchNorm2d(self.name + '.bn0', self.in_channels)
        self.relu0    = F.Relu(self.name + '.relu0')
        self.maxpool0 = F.MaxPool2d(self.name + '.maxpool0', kernel_size=3, stride=2, padding=1)

        layer_ops_planes  = [64, 128, 256, 512]
        layer_ops_strides = [1, 2, 2, 2]
        self.layer_ops    = [ SimNN.ModuleList( self._make_layer(i,
                                                                 self.layers[i],
                                                                 planes=layer_ops_planes[i],
                                                                 stride=layer_ops_strides[i])
                                               ) for i in range(4) ]

        #self.layer1   = SimNN.ModuleList(self._make_layer(0, self.layers[0], planes= 64, stride=1))
        #self.layer2   = SimNN.ModuleList(self._make_layer(1, self.layers[1], planes=128, stride=2))
        #self.layer3   = SimNN.ModuleList(self._make_layer(2, self.layers[2], planes=256, stride=2))
        #self.layer4   = SimNN.ModuleList(self._make_layer(3, self.layers[3], planes=512, stride=2))

        #because I am creating SimNN.ModuleList dynamically as self.layer_ops,
        # SimNN.Module __setattr__ think I am just creating a list; therefore,
        # I need to register all the modules for proper graph construction
        for ML in self.layer_ops:
            for m in ML:
                self._submodules[m.name] = m
        super().link_op2module()

    def __call__(self, x):
        batch, img_chnl, img_width, img_height = x.shape

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)

        result = []
        for i in range(4):
            for blk in self.layer_ops[i]:
                x = blk(x)
            result.append(x)

        #y = [result[i] for i in self.out_indices]
        y = result
        return y

    def _make_layer(self, lyr_num, blocks, planes, stride=1):
        downsample_cfg = None
        block_list     = []

        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample_cfg = {
                    'in_channels': self.in_channels,
                    'out_channels': planes * Bottleneck.expansion,
                    'stride': stride
                    }
        bb = Bottleneck(f'layer{lyr_num}.bb0', {'in_channels': self.in_channels,
                                'out_channels': planes,
                                'downsample': downsample_cfg,
                                'stride': stride})
        block_list.append(bb)
        self.in_channels = planes * Bottleneck.expansion

        for i in range(blocks-1):
            block_list.append(Bottleneck(f'layer{lyr_num}.bb{i+1}', {
                'in_channels': self.in_channels,
                'out_channels': planes
                }))

        return block_list

if __name__ == '__main__':
    import numpy as np
    logging_format = "%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s"
    logging.basicConfig(level=logging.WARNING, format=logging_format)

    #X = F._from_shape('X', [1, 3, 224, 224], np_dtype=np.float32) #N, C, H, W
    #resnet_cfg = {'layers': [3,4,6,3],  'num_classes': 1000, 'num_channels': 3}
    #rn_model = ResNet('rn50',resnet_cfg)
    #Y = rn_model(X)

    X = F._from_shape('X', [5, 160, 128, 128])
    resnet_cfg = {'layers': [2,2,2,2],  'num_classes': 2, 'num_channels': 3, 'out_indices': [0,1,2],
                  'num_channels': 160}
    rn_model = ResNet('rn18',resnet_cfg)
    Y = rn_model(X)
    for y in Y:
        print(y)
    gg = rn_model._get_forward_graph({'X': X})
    gg.graph2onnx(f'bevdepth_backbone_rn50.onnx', do_model_check=True)
