#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np
import logging

class Bottleneck(SimNN.Module):
    expansion = 4
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.in_channels  = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.stride       = cfg.get('stride', 1)
        self.downsample   = cfg.get('downsample', None)

        conv_dims = [ #IC, OC, K, P, S
                (self.in_channels,  self.out_channels, 1, 0, 1),
                (self.out_channels, self.out_channels, 3, 1, self.stride),
                (self.out_channels, self.out_channels*Bottleneck.expansion, 1, 0, 1),
                ]
        oplist = []
        for i, (ic, oc, k, p, s) in enumerate(conv_dims):
            conv = F.Conv2d(self.name + f'.conv{i}', ic, oc, kernel_size=k, padding=p, stride=s)
            bn   = F.BatchNorm2d(self.name + f'.bn{i}', oc)
            oplist += [conv, bn]
            # oplist += [conv]

        self.op_blk  = F.SimOpHandleList(oplist)
        self.add     = F.Add(self.name + f'.add')
        self.relu    = F.Relu(self.name + f'.relu')

        #self.conv_ds = None
        #self.bn_ds   = None

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
            z = self.add(y, x)
        else:
            x1 = self.conv_ds(x)
            #logging.debug('DBG conv-ds %s = %s', x, x1)
            z  = self.add(y, x1)
            x2 = self.bn_ds(x1)
            #z  = self.add(y, x2)
        w = self.relu(z)
        return w


class ResNet(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.bs = cfg.get('batch_size', 1)
        self.num_channels = cfg.get('num_channels', 3)
        # Support for high resolution images (up to 8M pixels = ~2828x2828)
        self.img_height = cfg.get('img_height', 224)
        self.img_width = cfg.get('img_width', 224)
        self.use_adaptive_pool = cfg.get('use_adaptive_pool', False)
        
        # Initial conv layer with larger stride for high-res images
        init_stride = 4 if max(self.img_height, self.img_width) > 1000 else 2
        self.conv0 = F.Conv2d(self.name + '.conv0', self.num_channels, 64, kernel_size=7, stride=init_stride, padding=3)
        self.bn0 = F.BatchNorm2d(self.name + '.bn0', 64)
        self.relu0 = F.Relu(self.name + '.relu0')
        self.maxpool0 = F.MaxPool2d(self.name + '.maxpool0', kernel_size=3, stride=2, padding=1)

        self.in_channels = 64
        self.layers = cfg.get('layers', [3,4,6,3])
        self.num_classes = cfg.get('num_classes', 1000)

        layers_msg = f"ResNet.layers should be [int, int, int, int] with each member > 0: {self.layers}"
        assert isinstance(self.layers, list) and \
               len(self.layers) == 4 and \
               all([isinstance(x, int) and x > 0 for x in self.layers]), layers_msg

        self.layer1 = SimNN.ModuleList(self._make_layer(0, self.layers[0], planes=64, stride=1))
        self.layer2 = SimNN.ModuleList(self._make_layer(1, self.layers[1], planes=128, stride=2))
        self.layer3 = SimNN.ModuleList(self._make_layer(2, self.layers[2], planes=256, stride=2))
        self.layer4 = SimNN.ModuleList(self._make_layer(3, self.layers[3], planes=512, stride=2))

        # For high-resolution images, use adaptive pooling to ensure correct output dimensions
        if self.use_adaptive_pool:
            self.avgpool = F.AdaptiveAvgPool2d(self.name + '.avgpool', (1, 1))
        else:
            # Use fixed kernel size pooling (standard ResNet)
            self.avgpool = F.AveragePool2d(self.name + '.avgpool', (7, 7))
            
        # Create reshape operation
        self.reshape = F.ReshapeFixed(self.name + '.reshape', [self.bs, -1])
        
        # Register avgpool output tensor that will be reshape input
        self._tensors[self.name + '.avgpool.out'] = F._from_shape(self.name + '.avgpool.out', 
                                                                [self.bs, 512*Bottleneck.expansion, 1, 1])
        # Register reshape output tensor
        self._tensors[self.name + '.reshape.out'] = F._from_shape(self.name + '.reshape.out', 
                                                                [self.bs, 512*Bottleneck.expansion])
        
        self.fc = F.Linear(self.name + '.fc0', 512*Bottleneck.expansion, self.num_classes)

        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
                'x_in': F._from_shape('x_in', [self.bs, self.num_channels, self.img_height, self.img_width],
                                   is_param=False, np_dtype=np.float32),
                }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self):
        return 0

    def __call__(self):
        assert len(self.input_tensors) == 1, f"input_tensors missing!! Need create_input_tensors() before __call__: {self.input_tensors}"
        x = self.input_tensors['x_in']
        assert len(x.shape) == 4, f"Input to ResNet should be a tensor: [N,C,H,W] : {x.shape}!!"

        # Initial feature extraction with increased stride
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)

        # Process through ResNet blocks
        for blk in self.layer1:
            x = blk(x)
        for blk in self.layer2:
            x = blk(x)
        for blk in self.layer3:
            x = blk(x)
        for blk in self.layer4:
            x = blk(x)

        # Fix: Pass kernel_shape parameter name explicitly
        x = self.avgpool(x)
        x = self.reshape(x)
        x = self.fc(x)

        return x

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

def run_standalone(outdir: str ='.')->None:
    import numpy as np
    shape    = [1, 3, 224, 224] #N, C, H, W
    dtype = np.float32
    _data = np.random.randn(*shape).astype(dtype)
    x = SimTensor({'name': 'x', 'shape': shape, 'data': _data, 'dtype': np.dtype(dtype)})
    #logging.debug(x)

    #bn_model = Bottleneck('bneck', {'in_channels': 64, 'out_channels': 64})
    #y = bn_model(x)
    ##logging.debug(y)

    #gg = bn_model.get_forward_graph(x)
    #gg.graph2onnx('xyxy.onnx')


    resnet_cfgs = {
            'resnet_50' : {'layers': [3,4,6,3],  'num_classes': 1000, 'num_channels': 3},
            # 'resnet_101': {'layers': [3,4,23,3], 'num_classes': 1000, 'num_channels': 3},
            # 'resnet_152': {'layers': [3,8,36,3], 'num_classes': 1000, 'num_channels': 3},
            }

    for k,v in resnet_cfgs.items():
        rn_model = ResNet(k,v)
        rn_model.create_input_tensors()
        y = rn_model()
        #logging.debug(y)
        gg = rn_model.get_forward_graph()
        gg.graph2onnx(f'{outdir}/{k}.onnx', do_model_check=True)


if __name__ == '__main__':
    logging_format = "%(levelname)s:%(name)s:%(filename)s:%(lineno)d:%(message)s"
    logging.basicConfig(level=logging.WARNING, format=logging_format)
    run_standalone()
