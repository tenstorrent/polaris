#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MyResBlock - Residual block for temporal feature processing
Converted from MapTracker PyTorch implementation to ttsim
"""

# -------------------------------PyTorch--------------------------------

# import torch
# import torch.nn as nn
# from typing import Optional, Sequence, Tuple, Union
# from mmdet.models import NECKS
# from mmcv.cnn.utils import kaiming_init, constant_init
# from mmcv.cnn.resnet import conv3x3
# from torch import Tensor
#
# from einops import rearrange
#
#
# class MyResBlock(nn.Module):
#     def __init__(self,
#                  inplanes: int,
#                  planes: int,
#                  stride: int = 1,
#                  dilation: int = 1,
#                  style: str = 'pytorch',
#                  with_cp: bool = False):
#         super().__init__()
#         assert style in ['pytorch', 'caffe']
#         self.conv1 = conv3x3(inplanes, planes, stride, dilation)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.stride = stride
#         self.dilation = dilation
#         assert not with_cp
#
#     def forward(self, x: Tensor) -> Tensor:
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# @NECKS.register_module()
# class TemporalNet(nn.Module):
#     def __init__(self, history_steps, hidden_dims, num_blocks):
#         super(TemporalNet, self).__init__()
#         self.history_steps = history_steps
#         self.hidden_dims = hidden_dims
#         self.num_blocks = num_blocks
#
#         layers = []
#
#         in_dims = (history_steps+1) * hidden_dims
#         self.conv_in = conv3x3(in_dims, hidden_dims, 1, 1)
#         self.bn = nn.BatchNorm2d(hidden_dims)
#         self.relu = nn.ReLU(inplace=True)
#
#         for _ in range(self.num_blocks):
#             layers.append(MyResBlock(hidden_dims, hidden_dims))
#         self.res_layer = nn.Sequential(*layers)
#
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#
#
#     def forward(self, history_feats, curr_feat):
#         input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)
#         input_feats = rearrange(input_feats, 'b t c h w -> b (t c) h w')
#
#         out = self.conv_in(input_feats)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.res_layer(out)
#         if curr_feat.dim() == 3:
#             out = out.squeeze(0)
#
#         return out

# -------------------------------TTSIM-----------------------------------

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.sim_nn import ModuleList

import numpy as np


class MyResBlock(SimNN.Module):
    """Residual block: Conv3x3 + BN + ReLU + Conv3x3 + BN + Add + ReLU"""

    def __init__(self, name, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        self.name = name
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation

        # Conv3x3: kernel=3, padding=dilation (to maintain spatial size)
        padding = dilation

        # First conv block: Conv3x3 + BN + ReLU
        self.conv1 = F.Conv2d(
            name + ".conv1",
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = F.BatchNorm2d(name + ".bn1", planes)
        self.relu1 = F.Relu(name + ".relu1")

        # Second conv block: Conv3x3 + BN
        self.conv2 = F.Conv2d(
            name + ".conv2",
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = F.BatchNorm2d(name + ".bn2", planes)

        # Residual addition
        self.add = F.Add(name + ".add")

        # Final ReLU
        self.relu2 = F.Relu(name + ".relu2")

        super().link_op2module()

    def __call__(self, x):
        # Save residual
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual
        out = self.add(residual, out)

        # Final activation
        out = self.relu2(out)

        return out

    def analytical_param_count(self, lvl):
        # Conv1: inplanes * planes * 3 * 3
        conv1_params = self.inplanes * self.planes * 3 * 3
        # BN1: 2 * planes (weight + bias)
        bn1_params = 2 * self.planes
        # Conv2: planes * planes * 3 * 3
        conv2_params = self.planes * self.planes * 3 * 3
        # BN2: 2 * planes (weight + bias)
        bn2_params = 2 * self.planes

        return conv1_params + bn1_params + conv2_params + bn2_params


class TemporalNet(SimNN.Module):
    """
    Temporal feature fusion network

    Args:
        name: Module name
        history_steps: Number of history frames
        hidden_dims: Feature dimension
        num_blocks: Number of residual blocks

    Architecture:
        1. Concatenate history features with current features
        2. Rearrange (b,t,c,h,w) -> (b,t*c,h,w)
        3. Conv3x3 + BN + ReLU
        4. Stack of MyResBlocks
    """

    def __init__(self, name, history_steps, hidden_dims, num_blocks):
        super().__init__()
        self.name = name
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks

        # Calculate input dimension after time-to-channel conversion
        in_dims = (history_steps + 1) * hidden_dims

        # Pre-create all operators
        # Concatenation along time dimension
        self.concat_time = F.ConcatX(name + ".concat_time", axis=1)

        # Reshape to merge time into channels: (b,t,c,h,w) -> (b,t*c,h,w)
        self.reshape = F.Reshape(name + ".reshape")

        # Initial conv block: Conv3x3 + BN + ReLU
        self.conv_in = F.Conv2d(
            name + ".conv_in",
            in_dims,
            hidden_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
        )
        self.bn = F.BatchNorm2d(name + ".bn", hidden_dims)
        self.relu = F.Relu(name + ".relu")

        # Create residual blocks
        _res_blocks = []
        for i in range(num_blocks):
            block = MyResBlock(
                name=f"{name}.res_block_{i}", inplanes=hidden_dims, planes=hidden_dims
            )
            _res_blocks.append(block)
        self.res_blocks = ModuleList(_res_blocks)

        super().link_op2module()

    def __call__(self, history_feats, curr_feat):
        """
        Forward pass

        Args:
            history_feats: [batch, time_steps, channels, height, width]
            curr_feat: [batch, channels, height, width]

        Returns:
            out: [batch, channels, height, width]
        """
        # Store original curr_feat dimensions for conditional squeeze later
        curr_feat_ndim = len(curr_feat.shape)

        # 1. Add time dimension to current feature using Reshape: [b,c,h,w] -> [b,1,c,h,w]
        b, c, h, w = curr_feat.shape
        unsqueeze_shape = F._from_data(
            self.name + ".unsqueeze_shape",
            np.array([b, 1, c, h, w], dtype=np.int64),
            is_const=True,
        )
        setattr(self, unsqueeze_shape.name, unsqueeze_shape)
        curr_feat_unsqueezed = self.reshape(curr_feat, unsqueeze_shape)
        setattr(self, curr_feat_unsqueezed.name, curr_feat_unsqueezed)

        # 2. Concatenate along time dimension: [b,t,c,h,w] + [b,1,c,h,w] -> [b,t+1,c,h,w]
        input_feats = self.concat_time(history_feats, curr_feat_unsqueezed)
        setattr(self, input_feats.name, input_feats)

        # 3. Reshape to merge time into channels: [b,t+1,c,h,w] -> [b,(t+1)*c,h,w]
        # einops: 'b t c h w -> b (t c) h w'
        b, t, c, h, w = input_feats.shape
        merge_shape = F._from_data(
            self.name + ".merge_shape",
            np.array([b, t * c, h, w], dtype=np.int64),
            is_const=True,
        )
        setattr(self, merge_shape.name, merge_shape)
        input_feats_reshaped = self.reshape(input_feats, merge_shape)
        setattr(self, input_feats_reshaped.name, input_feats_reshaped)

        # 4. Initial conv block
        out = self.conv_in(input_feats_reshaped)
        setattr(self, out.name, out)

        out = self.bn(out)
        setattr(self, out.name, out)

        out = self.relu(out)
        setattr(self, out.name, out)

        # 5. Apply residual blocks
        for i, block in enumerate(self.res_blocks):
            out = block(out)
            setattr(self, out.name, out)

        # 6. Conditional squeeze if original curr_feat was 3D
        # Use Reshape to remove batch dimension: [1,c,h,w] -> [c,h,w]
        if curr_feat_ndim == 3:
            _, c, h, w = out.shape
            squeeze_shape = F._from_data(
                self.name + ".squeeze_shape",
                np.array([c, h, w], dtype=np.int64),
                is_const=True,
            )
            setattr(self, squeeze_shape.name, squeeze_shape)
            out = self.reshape(out, squeeze_shape)
            setattr(self, out.name, out)

        return out

    def analytical_param_count(self, lvl):
        """Calculate parameter count"""
        in_dims = (self.history_steps + 1) * self.hidden_dims

        # Conv_in: weight only (no bias)
        conv_in_params = in_dims * self.hidden_dims * 3 * 3
        # BN: weight + bias
        bn_params = 2 * self.hidden_dims

        # Residual blocks
        res_blocks_params = sum(
            block.analytical_param_count(lvl)  # type: ignore[attr-defined]
            for block in self.res_blocks
        )

        total = conv_in_params + bn_params + res_blocks_params
        return total
