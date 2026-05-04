#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of SE_Block (Squeeze-and-Excitation Block).

Original file: mmdet3d/models/detectors/ealss.py (lines 24-31)

Graph:
    x [B, C, H, W]
      → AdaptiveAvgPool2d(1)  → [B, C, 1, 1]   (global average pool)
      → Conv2d(C, C, k=1)     → [B, C, 1, 1]   (channel weighting)
      → Sigmoid               → [B, C, 1, 1]   (gate in [0,1])
      → Mul(x, att)           → [B, C, H, W]   (scale input channels)

Parameters:
    Conv2d(C, C, k=1, bias=True): C*C + C

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


class SE_Block(SimNN.Module):
    """
    TTSim Squeeze-and-Excitation Block.

    Squeezes spatial information via global average pooling, then learns
    per-channel scaling factors through a small Conv2d + Sigmoid.  The
    result is multiplied element-wise with the original input, allowing
    the model to recalibrate channel-wise feature responses.

    Args:
        name (str): Unique module name prefix.
        channels (int): Number of input (and output) channels C.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)

    Params:
        Conv2d(C, C, k=1, bias=True) → C*C + C
    """

    def __init__(self, name: str, channels: int):
        super().__init__()
        self.name = name
        self.channels = channels

        # nn.AdaptiveAvgPool2d(1) → output_size=1 means [B, C, 1, 1]
        self.gap = F.AdaptiveAvgPool2d(name + ".gap", output_size=1)

        # nn.Conv2d(c, c, kernel_size=1, stride=1) — bias=True by default
        self.conv = F.Conv2d(
            name + ".conv",
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (SimTensor): Input of shape (B, C, H, W).

        Returns:
            SimTensor: Channel-recalibrated output of shape (B, C, H, W).
        """
        att = self.gap(x)                                   # [B, C, 1, 1]
        att = self.conv(att)                                # [B, C, 1, 1]
        att = F.Sigmoid(self.name + ".sigmoid")(att)        # [B, C, 1, 1]
        out = F.Mul(self.name + ".mul")(x, att)             # [B, C, H, W]
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Learnable parameters:
            Conv2d(C, C, k=1, bias=True): C*C (weight) + C (bias) = C^2 + C
        """
        C = self.channels
        return C * C + C
