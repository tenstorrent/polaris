#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ConvGRU - Convolutional GRU for temporal feature fusion (ttsim version)
Converted from MapTracker's PyTorch implementation
"""

# -------------------------------PyTorch--------------------------------

# import torch
# import torch.nn as nn
# from mmdet.models import NECKS
# from mmcv.cnn.utils import kaiming_init, constant_init
#
#
# @NECKS.register_module()
# class ConvGRU(nn.Module):
#     def __init__(self, out_channels):
#         super(ConvGRU, self).__init__()
#         kernel_size = 1
#         padding = kernel_size // 2
#         self.convz = nn.Conv2d(2*out_channels,
#             out_channels, kernel_size=kernel_size, padding=padding, bias=False)
#         self.convr = nn.Conv2d(2*out_channels,
#             out_channels, kernel_size=kernel_size, padding=padding, bias=False)
#         self.convq = nn.Conv2d(2*out_channels,
#             out_channels, kernel_size=kernel_size, padding=padding, bias=False)
#         self.ln = nn.LayerNorm(out_channels)
#         self.zero_out = nn.Conv2d(out_channels, out_channels, 1, 1, bias=True)
#
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#         nn.init.zeros_(self.zero_out.weight)
#         nn.init.zeros_(self.zero_out.bias)
#
#     def forward(self, h, x):
#         if len(h.shape) == 3:
#             h = h.unsqueeze(0)
#         if len(x.shape) == 3:
#             x = x.unsqueeze(0)
#
#         hx = torch.cat([h, x], dim=1) # [1, 2c, h, w]
#         z = torch.sigmoid(self.convz(hx))
#         r = torch.sigmoid(self.convr(hx))
#         new_x = torch.cat([r * h, x], dim=1) # [1, 2c, h, w]
#         q = self.convq(new_x)
#
#         out = ((1 - z) * h + z * q) # (1, C, H, W)
#         out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
#         out = self.zero_out(out)
#         out = out + x
#         out = out.squeeze(0)
#
#         return out

# -------------------------------TTSIM-----------------------------------


import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class ConvGRU(SimNN.Module):
    """
    Convolutional GRU for temporal feature fusion

    Performs GRU-style gating to fuse hidden state h with new input x:
        z = sigmoid(conv([h, x]))  # update gate
        r = sigmoid(conv([h, x]))  # reset gate
        q = conv([r*h, x])          # candidate
        out = (1-z)*h + z*q         # updated state
        out = LayerNorm(out)
        out = zero_initialized_conv(out) + x  # residual

    Args:
        out_channels: Number of channels
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        kernel_size = 1
        padding = kernel_size // 2

        # GRU gates: z (update), r (reset), q (candidate)
        self.convz_weight = F.Conv2d(
            "convz_weight",
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.convr_weight = F.Conv2d(
            "convr_weight",
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.convq_weight = F.Conv2d(
            "convq_weight",
            2 * out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Concat operators (created once, called later)
        self.concat_hx = F.ConcatX("concat_hx", axis=1)
        self.concat_rh_x = F.ConcatX("concat_rh_x", axis=1)

        # LayerNorm (creates weight and bias internally)
        self.layer_norm = F.LayerNorm("layer_norm", out_channels)

        # Zero-initialized output conv
        self.zero_out_weight = F.Conv2d(
            "zero_out_weight",
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.zero_out_bias_param = F._from_shape("zero_out_bias", [out_channels])

        # Sigmoid activations
        self.z_sigmoid = F.Sigmoid("z_sigmoid")
        self.r_sigmoid = F.Sigmoid("r_sigmoid")

        # Math operators
        self.r_mul_h = F.Mul("r_mul_h")
        self.one_minus_z_sub = F.Sub("one_minus_z")
        self.term1_mul = F.Mul("term1")
        self.term2_mul = F.Mul("term2")
        self.gru_out_add = F.Add("gru_out")
        self.zero_out_add_bias = F.Add("zero_out_add_bias")
        self.residual_add = F.Add("residual")

        # Transpose and Reshape operators
        self.out_perm_for_ln = F.Transpose("out_perm_for_ln", perm=[0, 2, 3, 1])
        self.out_perm_back = F.Transpose("out_perm_back", perm=[0, 3, 1, 2])
        self.zero_out_bias_reshape = F.Reshape("zero_out_bias_reshape")
        self.out_squeeze = F.Reshape("out_squeeze")
        self.h_unsqueeze = F.Reshape("h_unsqueeze")
        self.x_unsqueeze = F.Reshape("x_unsqueeze")

    def __call__(self, h, x):
        """
        Forward pass

        Args:
            h: Hidden state [B, C, H, W] or [C, H, W]
            x: New input [B, C, H, W] or [C, H, W]

        Returns:
            out: Updated state [B, C, H, W] or [C, H, W]
        """
        # Handle 3D inputs by adding batch dimension
        h_is_3d = False
        if len(h.shape) == 3:
            h_shape = F._from_data(
                "h_shape",
                np.array(
                    [1, self.out_channels, h.shape[-2], h.shape[-1]], dtype=np.int64
                ),
            )
            h = self.h_unsqueeze(h, h_shape)
            h_is_3d = True

        x_is_3d = False
        if len(x.shape) == 3:
            x_shape = F._from_data(
                "x_shape",
                np.array(
                    [1, self.out_channels, x.shape[-2], x.shape[-1]], dtype=np.int64
                ),
            )
            x = self.x_unsqueeze(x, x_shape)
            x_is_3d = True

        # Concatenate h and x: [B, 2*C, H, W]
        hx = self.concat_hx(h, x)

        # Update gate: z = sigmoid(conv(hx))
        z = self.convz_weight(hx)
        z = self.z_sigmoid(z)

        # Reset gate: r = sigmoid(conv(hx))
        r = self.convr_weight(hx)
        r = self.r_sigmoid(r)

        # Apply reset to hidden: r * h
        rh = self.r_mul_h(r, h)

        # Candidate: q = conv([r*h, x])
        new_x = self.concat_rh_x(rh, x)
        q = self.convq_weight(new_x)

        # Update: out = (1-z)*h + z*q
        # Create ones tensor with same shape as z
        ones = F._from_data("ones", np.ones(z.shape, dtype=np.float32), is_const=True)
        one_minus_z = self.one_minus_z_sub(ones, z)
        term1 = self.term1_mul(one_minus_z, h)
        term2 = self.term2_mul(z, q)
        out = self.gru_out_add(term1, term2)

        # LayerNorm: normalize over channel dimension
        # out shape: [B, C, H, W] -> [B, H, W, C] for LayerNorm
        out_permuted = self.out_perm_for_ln(out)
        out_ln = self.layer_norm(out_permuted)
        # Permute back: [B, H, W, C] -> [B, C, H, W]
        out = self.out_perm_back(out_ln)

        # Zero-initialized conv
        out = self.zero_out_weight(out)
        # Add bias [C] -> [1, C, 1, 1] for broadcasting
        bias_shape = F._from_data(
            "bias_shape", np.array([1, self.out_channels, 1, 1], dtype=np.int64)
        )
        bias_reshaped = self.zero_out_bias_reshape(self.zero_out_bias_param, bias_shape)
        out = self.zero_out_add_bias(out, bias_reshaped)

        # Residual connection
        out = self.residual_add(out, x)

        # Remove batch dimension if input was 3D
        if h_is_3d or x_is_3d:
            out_shape = F._from_data(
                "out_shape",
                np.array(
                    [self.out_channels, out.shape[-2], out.shape[-1]], dtype=np.int64
                ),
            )
            out = self.out_squeeze(out, out_shape)

        return out

    def analytical_param_count(self, lvl):
        """
        Calculate total number of trainable parameters

        Returns:
            int: Total parameter count
        """
        # Conv gates (z, r, q): each has weight only (no bias)
        # Shape: [out_channels, 2*out_channels, kernel_size, kernel_size]
        conv_params_per_gate = self.out_channels * (2 * self.out_channels) * 1 * 1
        conv_total = 3 * conv_params_per_gate  # convz, convr, convq

        # LayerNorm: weight + bias = 2 * out_channels
        ln_params = 2 * self.out_channels

        # Zero-out conv: weight + bias
        # Shape: [out_channels, out_channels, 1, 1]
        zero_out_weight = self.out_channels * self.out_channels * 1 * 1
        zero_out_bias = self.out_channels
        zero_out_total = zero_out_weight + zero_out_bias

        total = conv_total + ln_params + zero_out_total
        return total
