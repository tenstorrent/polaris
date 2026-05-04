#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of PositionEmbeddingLearned.

Original file: mmdet3d/models/dense_heads/transfusion_head.py (class at ~line 120)

Computes learned absolute position embeddings from 3-D (or 6-D) query
coordinates via a small 1-D convolutional network:

    xyz [B, P, input_channel]
      → Transpose         → [B, input_channel, P]
      → Conv1d(input_channel → num_pos_feats, k=1) + BN1d + ReLU
      → Conv1d(num_pos_feats → num_pos_feats, k=1)
      → output [B, num_pos_feats, P]

The output is used to enrich transformer queries / keys with spatial
position information in TransFusionHead.

Parameters (default input_channel=3, num_pos_feats=288):
    Conv1d layer-0: input_channel * num_pos_feats + num_pos_feats (bias)
    BN1d:           2 * num_pos_feats
    Conv1d layer-1: num_pos_feats * num_pos_feats + num_pos_feats (bias)
    Total: in_ch * F + F + 2*F + F*F + F  where F = num_pos_feats

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

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.mlp import BatchNorm1d, ConvModule1d


class PositionEmbeddingLearned(SimNN.Module):
    """
    TTSim PositionEmbeddingLearned.

    Applies a two-layer 1-D conv head to 3-D / 6-D position coordinates
    to produce per-point position embeddings.

    Args:
        name (str): Unique module name prefix.
        input_channel (int): Coordinate dimension (e.g. 3 for XYZ).
            Default: 3.
        num_pos_feats (int): Output embedding dimension.
            Default: 288.

    Shape:
        - Input:  (B, P, input_channel)
        - Output: (B, num_pos_feats, P)
    """

    def __init__(
        self,
        name: str,
        input_channel: int = 3,
        num_pos_feats: int = 288,
    ):
        super().__init__()
        self.name = name
        self.input_channel = input_channel
        self.num_pos_feats = num_pos_feats

        # Layer 0: Conv1d(input_channel → num_pos_feats, k=1) + BN1d + ReLU
        # Using ConvModule1d from mlp.py (with_bn=False; BN created separately)
        self.conv0 = ConvModule1d(
            name + ".conv0",
            in_channels=input_channel,
            out_channels=num_pos_feats,
            kernel_size=1,
            with_bn=False,
            with_relu=True,       # ReLU applied inside ConvModule1d
        )
        self._submodules[self.conv0.name] = self.conv0

        # BatchNorm1d after conv0
        self.bn0 = BatchNorm1d(name + ".bn0", num_pos_feats)
        self._submodules[self.bn0.name] = self.bn0

        # Layer 1: Conv1d(num_pos_feats → num_pos_feats, k=1) — no BN, no ReLU
        self.conv1 = ConvModule1d(
            name + ".conv1",
            in_channels=num_pos_feats,
            out_channels=num_pos_feats,
            kernel_size=1,
            with_bn=False,
            with_relu=False,
        )
        self._submodules[self.conv1.name] = self.conv1

        # Register tr_in so its output tensor is tracked for polaris graph.
        self.tr_in = F.Transpose(name + ".tr_in", perm=[0, 2, 1])

        super().link_op2module()

    def __call__(self, xyz):
        """
        Forward pass.

        Args:
            xyz (SimTensor): Input coordinates of shape (B, P, input_channel).

        Returns:
            SimTensor: Position embeddings of shape (B, num_pos_feats, P).
        """
        # xyz: [B, P, input_channel] — transpose to [B, input_channel, P]
        xyz_t = self.tr_in(xyz)

        # Two-layer conv head
        out = self.conv0(xyz_t)     # [B, num_pos_feats, P]  (conv+relu)
        out = self.bn0(out)         # [B, num_pos_feats, P]  (BN)
        out = self.conv1(out)       # [B, num_pos_feats, P]  (final conv)
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        conv0: input_channel * num_pos_feats + num_pos_feats (bias=True)
        bn0:   2 * num_pos_feats
        conv1: num_pos_feats * num_pos_feats + num_pos_feats (bias=True)
        """
        F_dim = self.num_pos_feats
        params = (
            self.input_channel * F_dim + F_dim   # conv0 + bias
            + 2 * F_dim                          # BN0 (scale + bias)
            + F_dim * F_dim + F_dim              # conv1 + bias
        )
        return params
