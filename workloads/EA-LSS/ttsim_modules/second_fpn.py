#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of SECONDFPN neck.

Original file: mmdet3d/models/necks/second_fpn.py

SECONDFPN is a Feature Pyramid Network variant used by SECOND/PointPillars/
PartA2/MVXNet.  It upsamples multi-scale LiDAR features using transposed
convolutions, applies BatchNorm2d + ReLU, then concatenates all up-sampled
feature maps along the channel dimension.

Default configuration:
    in_channels=[128, 128, 256]
    out_channels=[256, 256, 256]
    upsample_strides=[1, 2, 4]
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)
    upsample_cfg=dict(type='deconv', bias=False)

For each deblock i:
    ConvTranspose2d(in_channels[i], out_channels[i],
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
    + BatchNorm2d(out_channels[i])  + ReLU

Output: ConcatX of all deblock outputs along channel dimension.

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


class SECONDFPN(SimNN.Module):
    """
    TTSim SECONDFPN — multi-scale feature deconvolution and fusion.

    Each scale applies a transposed convolution (deconv) with kernel_size
    equal to the upsample stride.  After BN + ReLU, all outputs are
    concatenated along the channel axis.

    Call signature:
        out = secondfpn(x0, x1, x2, ...)   one tensor per level

    Args:
        name (str): Unique module name prefix.
        in_channels (list[int]): Input channels per backbone level.
            Default: [128, 128, 256].
        out_channels (list[int]): Output channels per deblock.
            Default: [256, 256, 256].
        upsample_strides (list[int]): Upsample stride per deblock.
            Default: [1, 2, 4].  Also used as kernel_size for deconv.
        eps (float): BatchNorm epsilon. Default: 1e-3.

    Shape (default config, input from SECOND( in=128 )):
        in0: [B, 128, H/2,  W/2]   out0: [B, 256, H/2, W/2]
        in1: [B, 128, H/4,  W/4]   out1: [B, 256, H/2, W/2]  (×2)
        in2: [B, 256, H/8,  W/8]   out2: [B, 256, H/2, W/2]  (×4)
        Concat → [B, 768, H/2, W/2]
    """

    def __init__(
        self,
        name: str,
        in_channels: list = None,
        out_channels: list = None,
        upsample_strides: list = None,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.name = name

        if in_channels is None:
            in_channels = [128, 128, 256]
        if out_channels is None:
            out_channels = [256, 256, 256]
        if upsample_strides is None:
            upsample_strides = [1, 2, 4]

        assert len(in_channels) == len(out_channels) == len(upsample_strides)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides
        self.num_deblocks = len(in_channels)

        for i in range(self.num_deblocks):
            stride = upsample_strides[i]
            # Use kernel_size == stride (mmdet3d deconv convention)
            kernel_size = max(stride, 1)

            # F.ConvTranspose2d signature: (name, in_ch, out_ch, kernel_size, stride)
            # Weight shape: [in_channels, out_channels, kH, kW]  (ONNX ConvTranspose)
            setattr(
                self,
                f"deconv{i}",
                F.ConvTranspose2d(
                    f"{name}.deconv{i}",
                    in_channels[i],
                    out_channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            )
            setattr(
                self,
                f"bn{i}",
                F.BatchNorm2d(f"{name}.bn{i}", out_channels[i], epsilon=eps),
            )

        # Register relu and concat ops so link_op2module tracks them and their
        # output tensors are auto-stored in _tensors when fired — required for
        # polaris graph construction.
        for i in range(self.num_deblocks):
            setattr(self, f"relu{i}", F.Relu(f"{name}.relu{i}"))
        if self.num_deblocks > 1:
            self.cat = F.ConcatX(f"{name}.cat", axis=1)

        super().link_op2module()

    def __call__(self, *inputs):
        """
        Forward pass.

        Args:
            *inputs: One SimTensor per backbone level.
                     len(inputs) must equal len(in_channels).

        Returns:
            SimTensor: Concatenated deblock outputs [B, sum(out_channels), H, W].
        """
        assert len(inputs) == self.num_deblocks, (
            f"Expected {self.num_deblocks} inputs, got {len(inputs)}"
        )

        ups = []
        for i, x_i in enumerate(inputs):
            deconv_i = getattr(self, f"deconv{i}")
            bn_i     = getattr(self, f"bn{i}")
            x_up = getattr(self, f"relu{i}")(bn_i(deconv_i(x_i)))
            ups.append(x_up)

        if len(ups) == 1:
            return ups[0]

        # Concatenate along channel dimension (axis=1)
        out = self.cat(*ups)
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        ConvTranspose2d (no bias): in_ch * out_ch * kH * kW
        BatchNorm2d:               2 * out_ch  (scale + bias)
        """
        total = 0
        for i in range(self.num_deblocks):
            k = max(self.upsample_strides[i], 1)
            total += self.in_channels[i] * self.out_channels[i] * k * k
            total += 2 * self.out_channels[i]
        return total
