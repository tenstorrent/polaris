#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

class FPN(SimNN.Module):
    def __init__(self, name, in_channels, out_channels, start_level=0, add_extra_convs='on_output',
                 relu_before_extra_convs=True, num_outs=5):
        super(FPN, self).__init__()
        self.name = name

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False

        if add_extra_convs == 'on_input':
            self.add_extra_convs = 'on_input'
        elif add_extra_convs == 'on_output':
            self.add_extra_convs = 'on_output'
        elif add_extra_convs == 'on_lateral':
            self.add_extra_convs = 'on_lateral'
        else:
            raise ValueError('add_extra_convs must be one of on_input, on_output, on_lateral')

        self.start_level = start_level
        if self.start_level < 0:
            raise ValueError('start_level must be >= 0')

        self.lateral_convs = []
        self.fpn_convs = []
        # Pre-register all lateral and fpn conv ops as attributes so graph capture includes them
        for i in range(self.start_level, self.num_ins):
            idx = i - self.start_level
            l_conv_name = f'{name}.lateral_convs.{idx}'
            fpn_conv_name = f'{name}.fpn_convs.{idx}'

            l_conv = F.Conv2d(l_conv_name, in_channels[i], out_channels, 1)
            fpn_conv = F.Conv2d(fpn_conv_name, out_channels, out_channels, 3, padding=1)

            setattr(self, f'lateral_conv_{idx}', l_conv)
            setattr(self, f'fpn_conv_{idx}', fpn_conv)
            self.lateral_convs.append(getattr(self, f'lateral_conv_{idx}'))
            self.fpn_convs.append(getattr(self, f'fpn_conv_{idx}'))

        # Pre-create Add and Resize ops for top-down merges
        self.add_ops = []
        self.resize_ops = []
        used_levels = len(self.in_channels) - self.start_level
        for i in range(1, used_levels):
            add_op = F.Add(f'{self.name}.add_{i}')
            setattr(self, f'add_{i}', add_op)
            self.add_ops.append(getattr(self, f'add_{i}'))

            # In standard FPN, each top-down step upsamples by 2x spatially
            resize_op = F.Resize(f'{self.name}.upsample_{i}', scale_factor=[2.0, 2.0])
            setattr(self, f'upsample_{i}', resize_op)
            self.resize_ops.append(getattr(self, f'upsample_{i}'))

        if self.add_extra_convs == 'on_input':
            self.extra_convs = []
            for i in range(self.start_level, self.num_outs - self.num_ins):
                extra_conv_name = f'{name}.extra_convs.{i - self.start_level}'
                extra_conv = F.Conv2d(extra_conv_name, in_channels[-1], out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)

        super().link_op2module()

    def __call__(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            laterals.append(lateral_conv(inputs[i + self.start_level]))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Use pre-created nearest-neighbor Resize with fixed 2x scale
            upsample_op = getattr(self, f'upsample_{i}')
            upsample_op.set_module(self)
            upsampled = upsample_op(laterals[i])
            # Merge with lateral using precreated add op (index i-1)
            laterals[i - 1] = self.add_ops[i - 1](laterals[i - 1], upsampled)

        # build outputs
        outs = []
        for i, lateral in enumerate(laterals):
            outs.append(self.fpn_convs[i](lateral))

        if self.add_extra_convs == 'on_input':
            for i in range(self.num_outs - used_backbone_levels):
                if i < len(self.extra_convs):
                    outs.append(self.extra_convs[i](inputs[-1]))
                else:
                    outs.append(outs[-1])
        elif self.add_extra_convs == 'on_lateral':
            for i in range(used_backbone_levels, self.num_outs):
                if self.relu_before_extra_convs:
                    outs.append(F.Relu(f'{self.name}.extra_relu_{i}')(outs[-1]))
                outs.append(F.Conv2d(f'{self.name}.extra_conv_{i}', self.out_channels, self.out_channels, 3, stride=2, padding=1)(outs[-1]))

        return tuple(outs)
