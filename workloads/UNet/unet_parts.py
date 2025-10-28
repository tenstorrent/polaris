#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

""" Parts of the U-Net model """
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class DoubleConv(SimNN.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, objname, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.name = objname
        self.double_conv = F.SimOpHandleList(
            [F.Conv2d(f'{self.name}_conv1', in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            F.BatchNorm2d(f'{self.name}_bn1', mid_channels),
            F.Relu(f'{self.name}_relu1'),
            F.Conv2d(f'{self.name}_conv2', mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            F.BatchNorm2d(f'{self.name}_bn2', out_channels),
            F.Relu(f'{self.name}_relu2')],
        )
        super().link_op2module()

    def __call__(self, x):
        return self.double_conv(x)


class Down(SimNN.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, objname, in_channels, out_channels):
        super().__init__()
        self.name = objname
        self.maxpool = F.MaxPool2d(f'{self.name}_maxpool', 2)
        self.conv = DoubleConv(f'{self.name}_conv', in_channels, out_channels)
        super().link_op2module()

    def __call__(self, x):
        x = self.maxpool(x)
        return self.conv(x)


class Up(SimNN.Module):
    """Upscaling then double conv"""

    def __init__(self, objname, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.name = objname

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = F.Upsample(f'{self.name}_upsample', scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(f'{self.name}_conv', in_channels, out_channels, in_channels // 2)
        else:
            self.up = F.ConvTranspose2d(f'{self.name}_convtranspose', in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(f'{self.name}_conv', in_channels, out_channels)

        self.padop = F.Pad(f'{self.name}_pad')
        self.concatxop = F.ConcatX(f'{self.name}_concat', axis=1)
        super().link_op2module()

    def __call__(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        _, _, x2Y, x2X = x2.shape
        _, _, x1Y, x1X = x1.shape
        diffY = x2Y - x1Y
        diffX = x2X - x1X

        import numpy as np
        pad_tensor = F._from_data(f'{self.name}_pad_tensor', np.array([diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]))
        self._tensors[pad_tensor.name] = pad_tensor
        x1 = self.padop(x1, pad_tensor)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        x = self.concatxop(x2, x1)
        return self.conv(x)


class OutConv(SimNN.Module):
    def __init__(self, objname, in_channels, out_channels):
        super().__init__()
        self.name = objname
        self.conv = F.Conv2d(f'{self.name}_conv', in_channels, out_channels, kernel_size=1)
        super().link_op2module()

    def __call__(self, x):
        return self.conv(x)
