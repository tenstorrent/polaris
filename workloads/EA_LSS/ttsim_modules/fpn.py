#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of FPN (Feature Pyramid Network) neck.

Original file: mmdet3d/models/necks/fpn.py

FPN builds a multi-scale feature pyramid from a backbone by:
  1. Applying 1×1 lateral convolutions to each backbone level.
  2. Adding top-down upsampled (×2) features to lower-level laterals.
  3. Applying 3×3 FPN convolutions to produce final pyramidal features.

Optional BatchNorm2d and ReLU after each conv are controlled by
``with_norm`` and ``with_act`` constructor flags.

For EALSS, FPN is applied to the camera image backbone outputs with
no norm and no activation (norm_cfg=None, act_cfg=None).

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data


class FPN(SimNN.Module):
    """
    TTSim Feature Pyramid Network neck.

    Takes outputs from multiple backbone stages (sorted from fine to coarse)
    and produces a set of feature maps at uniform channel width via lateral
    convolutions, top-down merging, and FPN convolutions.

    Call signature:
        outs = fpn(f0, f1, f2, f3, ...)
            fi: [B, in_channels[i], H_i, W_i]  (fine to coarse)
        Returns list of SimTensors [out0, out1, ..., out_{num_outs-1}]

    Args:
        name (str): Unique module name prefix.
        in_channels (list[int]): Input channels per backbone level.
        out_channels (int): Uniform output channels for all FPN levels.
        num_outs (int): Number of output levels.  If num_outs > num_ins,
            extra levels are produced by max-pooling the last output.
        start_level (int): First backbone level to include. Default: 0.
        end_level (int): Last backbone level (exclusive). -1 = all levels.
        with_norm (bool): Add BatchNorm2d after each conv. Default: False.
        with_act (bool): Add ReLU after each conv. Default: False.
        eps (float): BatchNorm epsilon (used only if with_norm=True).
    """

    def __init__(
        self,
        name: str,
        in_channels: list,
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        with_norm: bool = False,
        with_act: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.with_norm = with_norm
        self.with_act = with_act
        self.num_ins = len(in_channels)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level

        self.num_backbone_levels = self.backbone_end_level - start_level

        # Build lateral convs (1×1) and FPN convs (3×3)
        for i in range(self.num_backbone_levels):
            in_ch = in_channels[start_level + i]

            # Lateral conv: 1×1, no bias when norm follows
            lat_bias = not with_norm
            setattr(
                self,
                f"lat_conv{i}",
                F.Conv2d(f"{name}.lat_conv{i}", in_ch, out_channels,
                         kernel_size=1, padding=0, bias=lat_bias),
            )
            if with_norm:
                setattr(
                    self,
                    f"lat_bn{i}",
                    F.BatchNorm2d(f"{name}.lat_bn{i}", out_channels, epsilon=eps),
                )

            # FPN conv: 3×3
            fpn_bias = not with_norm
            setattr(
                self,
                f"fpn_conv{i}",
                F.Conv2d(f"{name}.fpn_conv{i}", out_channels, out_channels,
                         kernel_size=3, padding=1, bias=fpn_bias),
            )
            if with_norm:
                setattr(
                    self,
                    f"fpn_bn{i}",
                    F.BatchNorm2d(f"{name}.fpn_bn{i}", out_channels, epsilon=eps),
                )

        # Top-down upsample ops (one per level pair, fine←coarse)
        # Use F.Resize (not F.Upsample) because Resize satisfies the 2-input
        # ONNX arity requirement via its built-in params (roi, scales).
        for i in range(self.num_backbone_levels - 1):
            setattr(
                self,
                f"ups{i}",
                F.Resize(
                    f"{name}.ups{i}",
                    scale_factor=2.0,
                    mode="nearest",
                    nearest_mode="floor",
                    coordinate_transformation_mode="asymmetric",
                ),
            )
            # Add op for merging lateral + upsampled
            setattr(self, f"add{i}", F.Add(f"{name}.add{i}"))

        # Extra max-pool levels (if num_outs > backbone_levels)
        for i in range(abs(num_outs - self.num_backbone_levels)):
            setattr(
                self,
                f"maxpool_extra{i}",
                F.MaxPool2d(f"{name}.maxpool_extra{i}", kernel_size=1, stride=2),
            )

        super().link_op2module()

    def _apply_lateral(self, i, x):
        lat_conv = getattr(self, f"lat_conv{i}")
        out = lat_conv(x)
        if self.with_norm:
            out = getattr(self, f"lat_bn{i}")(out)
        if self.with_act:
            out = F.Relu(self.name + f".lat_relu{i}")(out)
        return out

    def _apply_fpn(self, i, x):
        fpn_conv = getattr(self, f"fpn_conv{i}")
        out = fpn_conv(x)
        if self.with_norm:
            out = getattr(self, f"fpn_bn{i}")(out)
        if self.with_act:
            out = F.Relu(self.name + f".fpn_relu{i}")(out)
        return out

    def __call__(self, *inputs):
        """
        Forward pass.

        Args:
            *inputs: One SimTensor per backbone level (fine→coarse order).

        Returns:
            list[SimTensor]: Output feature maps for each FPN level.
        """
        assert len(inputs) == self.num_ins, (
            f"FPN expected {self.num_ins} inputs, got {len(inputs)}"
        )

        used = list(inputs)[self.start_level:self.backbone_end_level]

        # 1. Lateral convolutions
        laterals = [self._apply_lateral(i, feat) for i, feat in enumerate(used)]

        # 2. Top-down path (from coarsest to finest)
        for i in range(self.num_backbone_levels - 1, 0, -1):
            ups_op = getattr(self, f"ups{i - 1}")
            add_op = getattr(self, f"add{i - 1}")
            upsampled = ups_op(laterals[i])
            laterals[i - 1] = add_op(laterals[i - 1], upsampled)

        # 3. FPN convs on merged laterals
        outs = [self._apply_fpn(i, laterals[i]) for i in range(self.num_backbone_levels)]

        # 4. Extra levels via max-pooling
        if self.num_outs > len(outs):
            extra_count = self.num_outs - self.num_backbone_levels
            for i in range(extra_count):
                mp_op = getattr(self, f"maxpool_extra{i}")
                outs.append(mp_op(outs[-1]))

        return outs

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Lateral conv (1×1): in_ch * out_ch [+ out_ch if no BN]
        FPN conv (3×3): out_ch * out_ch * 9 [+ out_ch if no BN]
        BN (if with_norm): 2 * out_ch per conv
        """
        lat_bias = not self.with_norm
        fpn_bias = not self.with_norm
        total = 0
        for i in range(self.num_backbone_levels):
            in_ch = self.in_channels[self.start_level + i]
            # Lateral 1×1
            total += in_ch * self.out_channels
            if lat_bias:
                total += self.out_channels
            if self.with_norm:
                total += 2 * self.out_channels  # BN scale + bias
            # FPN 3×3
            total += self.out_channels * self.out_channels * 9
            if fpn_bias:
                total += self.out_channels
            if self.with_norm:
                total += 2 * self.out_channels
        return total
