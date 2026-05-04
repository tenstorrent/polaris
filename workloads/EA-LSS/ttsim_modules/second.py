#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of SECOND backbone.

Original file: mmdet3d/models/backbones/second.py

The SECOND backbone consists of multiple 2-D convolutional stages.
Each stage begins with a strided Conv2d (to downsample), followed by
``layer_num`` repeated Conv2d blocks, each preceded by BatchNorm2d and
followed by ReLU.

Default configuration (from mmdet3d):
    in_channels=128
    out_channels=[128, 128, 256]
    layer_nums=[3, 5, 5]
    layer_strides=[2, 2, 2]
    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)
    conv_cfg=dict(type='Conv2d', bias=False)

All Conv2d operators use kernel_size=3.  No bias on Conv2d (bias=False).

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


# ---------------------------------------------------------------------------
# Single SECOND stage (one block element of self.blocks in the original)
# ---------------------------------------------------------------------------

class SECONDStage(SimNN.Module):
    """
    One stage of the SECOND backbone.

    Graph per stage:
        x [B, C_in, H, W]
          → Conv2d(C_in, C_out, 3, stride=stride, pad=1, bias=False) → BN → ReLU
          → [Conv2d(C_out, C_out, 3, pad=1, bias=False) → BN → ReLU] × layer_num
          → [B, C_out, H', W']   where H' = H // stride

    Args:
        name (str): Unique module name prefix.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        stride (int): Stride for the initial (downsampling) conv.
        layer_num (int): Number of repeated conv blocks after the initial one.
        eps (float): BatchNorm epsilon. Default: 1e-3.
        bias (bool): Whether conv layers have bias. Default: False.
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        stride: int,
        layer_num: int,
        eps: float = 1e-3,
        bias: bool = False,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.layer_num = layer_num
        self.eps = eps
        self.bias = bias

        # Initial strided conv
        setattr(
            self,
            "conv0",
            F.Conv2d(
                f"{name}.conv0",
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=bias,
            ),
        )
        setattr(self, "bn0", F.BatchNorm2d(f"{name}.bn0", out_channels, epsilon=eps))

        # Repeated conv blocks
        for j in range(layer_num):
            setattr(
                self,
                f"conv{j + 1}",
                F.Conv2d(
                    f"{name}.conv{j + 1}",
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                ),
            )
            setattr(
                self,
                f"bn{j + 1}",
                F.BatchNorm2d(f"{name}.bn{j + 1}", out_channels, epsilon=eps),
            )

        # Register relu ops so link_op2module tracks them and their output
        # tensors are auto-stored in _tensors when fired — required for polaris
        # graph construction.
        setattr(self, "relu0", F.Relu(f"{name}.relu0"))
        for j in range(layer_num):
            setattr(self, f"relu{j + 1}", F.Relu(f"{name}.relu{j + 1}"))

        super().link_op2module()

    def __call__(self, x):
        conv0 = getattr(self, "conv0")
        bn0   = getattr(self, "bn0")
        x = getattr(self, "relu0")(bn0(conv0(x)))

        for j in range(self.layer_num):
            conv_j = getattr(self, f"conv{j + 1}")
            bn_j   = getattr(self, f"bn{j + 1}")
            x = getattr(self, f"relu{j + 1}")(bn_j(conv_j(x)))

        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Initial conv (no bias): 3*3 * in_ch * out_ch
        Initial BN:             2 * out_ch
        Each repeated conv:     3*3 * out_ch^2
        Each repeated BN:       2 * out_ch
        """
        bias_extra = self.out_channels if self.bias else 0
        # Initial conv + BN
        params = 9 * self.in_channels * self.out_channels + bias_extra
        params += 2 * self.out_channels
        # Repeated blocks
        for _ in range(self.layer_num):
            params += 9 * self.out_channels * self.out_channels + bias_extra
            params += 2 * self.out_channels
        return params


# ---------------------------------------------------------------------------
# SECOND backbone
# ---------------------------------------------------------------------------

class SECOND(SimNN.Module):
    """
    TTSim SECOND backbone.

    Produces a tuple of multi-scale feature maps, one per stage.  All
    stages apply strided 3×3 convolutions for downsampling, then several
    isotropic 3×3 convolutions to increase receptive field.

    Call signature:
        outs = second(x)         where x is [B, in_channels, H, W]
        outs[i] is [B, out_channels[i], H // prod(strides[:i+1]), ...]

    Args:
        name (str): Unique module name prefix.
        in_channels (int): Number of input channels. Default: 128.
        out_channels (list[int]): Output channels per stage.
            Default: [128, 128, 256].
        layer_nums (list[int]): Repeated conv blocks per stage.
            Default: [3, 5, 5].
        layer_strides (list[int]): Stride of initial conv per stage.
            Default: [2, 2, 2].
        eps (float): BatchNorm epsilon. Default: 1e-3.
        bias (bool): Conv bias. Default: False (mmdet3d default).
    """

    def __init__(
        self,
        name: str,
        in_channels: int = 128,
        out_channels: list = None,
        layer_nums: list = None,
        layer_strides: list = None,
        eps: float = 1e-3,
        bias: bool = False,
    ):
        super().__init__()
        self.name = name

        if out_channels is None:
            out_channels = [128, 128, 256]
        if layer_nums is None:
            layer_nums = [3, 5, 5]
        if layer_strides is None:
            layer_strides = [2, 2, 2]

        assert len(out_channels) == len(layer_nums) == len(layer_strides)

        self.out_channels = out_channels
        self.layer_nums = layer_nums
        self.layer_strides = layer_strides
        self.num_stages = len(out_channels)

        in_filters = [in_channels] + list(out_channels[:-1])

        for i in range(self.num_stages):
            stage = SECONDStage(
                name=f"{name}.stage{i}",
                in_channels=in_filters[i],
                out_channels=out_channels[i],
                stride=layer_strides[i],
                layer_num=layer_nums[i],
                eps=eps,
                bias=bias,
            )
            setattr(self, f"stage{i}", stage)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (SimTensor): Input [B, in_channels, H, W].

        Returns:
            list[SimTensor]: One feature map per stage.
                stage i → [B, out_channels[i], H // prod(strides), W // prod(strides)]
        """
        outs = []
        for i in range(self.num_stages):
            stage = getattr(self, f"stage{i}")
            x = stage(x)
            outs.append(x)
        return outs

    def analytical_param_count(self, lvl: int = 0) -> int:
        total = 0
        for i in range(self.num_stages):
            stage = getattr(self, f"stage{i}")
            total += stage.analytical_param_count(lvl + 1)
        return total
