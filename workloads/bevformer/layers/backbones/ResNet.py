#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

def conv3x3(name, in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return F.Conv2d(name, in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def conv1x1(name, in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return F.Conv2d(name, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(SimNN.Module):
    expansion = 1

    def __init__(self, name, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.name = name

        self.conv1 = conv3x3(name + '.conv1', inplanes, planes, stride)
        self.bn1 = F.BatchNorm2d(name + '.bn1', planes)
        self.relu1 = F.Relu(name + '.relu1')
        self.conv2 = conv3x3(name + '.conv2', planes, planes)
        self.bn2 = F.BatchNorm2d(name + '.bn2', planes)
        self.relu2 = F.Relu(name + '.relu2')
        self.add  = F.Add(name + '.add')
        self.downsample = downsample
        self.stride = stride
        super().link_op2module()

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # no relu here before addition in basic block

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            identity = self.downsample.bn(identity)

        out = self.add(out, identity)
        out = self.relu2(out)

        return out

class Bottleneck(SimNN.Module):
    expansion = 4

    def __init__(self, name, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.name = name

        self.conv1 = conv1x1(name + '.conv1', inplanes, planes)
        self.bn1 = F.BatchNorm2d(name + '.bn1', planes)
        self.conv2 = conv3x3(name + '.conv2', planes, planes, stride)
        self.bn2 = F.BatchNorm2d(name + '.bn2', planes)
        self.conv3 = conv1x1(name + '.conv3', planes, planes * self.expansion)
        self.bn3 = F.BatchNorm2d(name + '.bn3', planes * self.expansion)
        self.relu1 = F.Relu(name + '.relu1')
        self.relu2 = F.Relu(name + '.relu2')
        self.relu3 = F.Relu(name + '.relu3')
        self.add  = F.Add(name + '.add')
        self.downsample = downsample
        self.stride = stride
        super().link_op2module()

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample.conv(x)
            identity = self.downsample.bn(identity)

        out = self.add(out, identity)
        out = self.relu3(out)

        return out

class ResNet(SimNN.Module):

    def __init__(self, name, cfg):
        super(ResNet, self).__init__()
        self.name = name

        layers = cfg['layers']
        out_indices = cfg.get('out_indices', [0, 1, 2, 3])
        num_classes = cfg.get('num_classes', 1000)
        in_channels = cfg.get('in_channels', 64)
        num_channels = cfg.get('num_channels', 3)  # input channels

        self.inplanes = in_channels
        self.out_indices = out_indices

        self.conv1 = F.Conv2d(name + '.conv1', num_channels, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = F.BatchNorm2d(name + '.bn1', in_channels)
        self.relu = F.Relu(name + '.relu')
        self.maxpool = F.MaxPool2d(name + '.maxpool', kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(name + '.layer1', in_channels, layers[0])
        self.layer2 = self._make_layer(name + '.layer2', in_channels*2, layers[1], stride=2)
        self.layer3 = self._make_layer(name + '.layer3', in_channels*4, layers[2], stride=2)
        self.layer4 = self._make_layer(name + '.layer4', in_channels*8, layers[3], stride=2)

        # Skip avgpool and fc for BEV detection (not needed)
        self.avgpool = None
        self.fc = None

        super().link_op2module()

    def _make_layer(self, name, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            # Create downsample as a Module with the two operations
            downsample = SimNN.Module()
            downsample.conv = conv1x1(name + '.downsample.0', self.inplanes, planes * Bottleneck.expansion, stride)
            downsample.bn = F.BatchNorm2d(name + '.downsample.1', planes * Bottleneck.expansion)

        layer_modules = []
        layer_modules.append(Bottleneck(name + '.0', self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layer_modules.append(Bottleneck(name + f'.{i}', self.inplanes, planes))

        return SimNN.ModuleList(layer_modules)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        x = self._run_layer(self.layer1, x)
        if 0 in self.out_indices:
            outs.append(x)

        x = self._run_layer(self.layer2, x)
        if 1 in self.out_indices:
            outs.append(x)

        x = self._run_layer(self.layer3, x)
        if 2 in self.out_indices:
            outs.append(x)

        x = self._run_layer(self.layer4, x)
        if 3 in self.out_indices:
            outs.append(x)

        return outs

    def _run_layer(self, layer_list, x):
        # ttsim ModuleList is not directly callable; iterate items sequentially
        for i in range(len(layer_list)):
            x = layer_list[i](x)
        return x
