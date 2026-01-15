#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt.common import TtConv2D
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr

class TtConvModule():
    def __init__(self, conv_args, conv_pth, device=None):
        self.device = device
        self.conv = TtConv2D(conv_args, DictAsAttr(conv_pth['conv']), device=self.device, dealloc_act=True)

    def __call__(self, x):
        x = self.conv(x)
        return x #[0]


class TtFPN():
    def __init__(self, conv_args, conv_pth, device):
        self.device = device
        self.lateral_convs = TtConvModule(conv_args.lateral_convs, conv_pth['lateral_convs'], device=device)
        self.fpn_convs = TtConvModule(conv_args.fpn_convs, conv_pth['fpn_convs'], device=device)

    def __call__(self, inputs):
        # Build laterals
        laterals = self.lateral_convs(inputs[0])
        # Apply FPN convs
        outs = self.fpn_convs(laterals)
        ttnn.deallocate(laterals)
        
        return outs # tuple(outs)

class ConvParams:
    class conv:
        def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                    bias=True, dilation=[1,1], groups=1, weight_dtype=ttnn.bfloat16,
                    input_height=200, input_width=176, batch_size=1):
            self.in_channels=in_channels
            self.out_channels=out_channels
            self.kernel_size=kernel_size
            self.padding=padding
            self.bias=bias
            self.dilation=dilation
            self.groups=groups
            self.weight_dtype=weight_dtype
            self.input_height=input_height
            self.input_width=input_width
            self.batch_size=batch_size

    def __init__(self):
        self.lateral_convs = ConvParams.conv(
            in_channels=2048,
            out_channels=256,
            kernel_size=[1,1],
            padding=[0,0],
            bias=True,
            dilation=[1,1],
            groups=1,
            weight_dtype=ttnn.bfloat16,
            input_height=200,
            input_width=176,
            batch_size=1,
        )
        self.fpn_convs = ConvParams.conv(
            in_channels=256,
            out_channels=256,
            kernel_size=[3,3],
            padding=[1,1],
            bias=True,
            dilation=[3,3],
            groups=1,
            weight_dtype=ttnn.bfloat16,
            input_height=200,
            input_width=176,
            batch_size=1,
        )

conv_args = ConvParams()
