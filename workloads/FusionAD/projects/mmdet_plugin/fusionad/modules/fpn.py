# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FPN (Feature Pyramid Network) neck for FusionAD — TTSim implementation.

Takes multi-scale backbone features and produces uniform-channel outputs.

FusionAD config:
  in_channels  = [512, 1024, 2048]  (from ResNet-101 stages 1,2,3)
  out_channels = 256
  num_outs     = 4  (3 from laterals + 1 extra conv on highest level)

"""

import os
import sys

polaris_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class FPN(SimNN.Module):
    """Feature Pyramid Network: uniform-channel multi-scale features.

    Args:
        name: Module name prefix.
        cfg: Dict with keys:
            - in_channels: list of int (per-level input channels)
            - out_channels: int (uniform output channels)
            - num_outs: int (number of output levels, default 4)
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        in_channels = cfg['in_channels']
        out_channels = cfg['out_channels']
        num_outs = cfg.get('num_outs', 4)
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        # Lateral 1x1 convolutions (reduce channels to out_channels)
        self.lateral_convs = []
        for i, ic in enumerate(in_channels):
            c = F.Conv2d(f'{name}.lat{i}', ic, out_channels, kernel_size=1)
            self.lateral_convs.append(c)
            setattr(self, f'lat{i}', c)

        # FPN 3x3 smoothing convolutions
        self.fpn_convs = []
        for i in range(self.num_ins):
            c = F.Conv2d(f'{name}.fpn{i}', out_channels, out_channels,
                         kernel_size=3, padding=1)
            self.fpn_convs.append(c)
            setattr(self, f'fpn{i}', c)

        # Extra convolutions for additional output levels
        # add_extra_convs="on_output": extra convs take last FPN output
        # relu_before_extra_convs=True: ReLU applied before each extra conv
        self.extra_convs = []
        self.extra_relus = []
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                c = F.Conv2d(f'{name}.extra{i}', out_channels, out_channels,
                             kernel_size=3, stride=2, padding=1)
                self.extra_convs.append(c)
                setattr(self, f'extra{i}', c)
                r = F.Relu(f'{name}.extra_relu{i}')
                self.extra_relus.append(r)
                setattr(self, f'extra_relu{i}', r)

        super().link_op2module()

    def __call__(self, features):
        assert len(features) == self.num_ins
        laterals = [self.lateral_convs[i](features[i])
                     for i in range(self.num_ins)]

        # Top-down pathway with upsampling
        for i in range(self.num_ins - 2, -1, -1):
            src_h = laterals[i + 1].shape[-2]
            src_w = laterals[i + 1].shape[-1]
            tgt_h = laterals[i].shape[-2]
            tgt_w = laterals[i].shape[-1]
            scale_h = tgt_h / src_h
            scale_w = tgt_w / src_w
            up = laterals[i + 1].interpolate(scale_factor=(scale_h, scale_w))
            laterals[i] = laterals[i] + up

        # Smoothing convolutions
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # Extra levels (stride-2 convs on last FPN output, with ReLU)
        if self.num_outs > self.num_ins:
            x_extra = outs[-1]
            for j, conv in enumerate(self.extra_convs):
                x_extra = self.extra_relus[j](x_extra)
                x_extra = conv(x_extra)
                outs.append(x_extra)

        return outs

    def analytical_param_count(self):
        return 0  # Counted via graph analysis
