#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of FPNC neck (+ gapcontext helper).

Original file: mmdet3d/models/necks/fpnc.py

gapcontext: Global Average Pool context module
    x → Conv2d(in,in,1) → AdaptiveAvgPool2d(1) → Resize to H×W → Add(x) → Conv2d(in,out,1)

FPNC: Extends FPN (already converted).
    Calls FPN forward, then resizes all output levels to target_size,
    concatenates along channel dim, and applies a reduction Conv2d.

    Optional use_adp: per-level adaptive pooling/upsampling + Conv2d(C,C,1).

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
from ttsim_modules.fpn import FPN


# ---------------------------------------------------------------------------
# gapcontext
# ---------------------------------------------------------------------------

class gapcontext(SimNN.Module):
    """
    Global Average Pool context module.

    Graph:
        x [B, in_ch, H, W]
          → Conv2d(in_ch, in_ch, 1)         → [B, in_ch, H, W]
          → AdaptiveAvgPool2d(1)             → [B, in_ch, 1, 1]
          → Resize to (H, W)                → [B, in_ch, H, W]
          → Add(x, resized)                 → [B, in_ch, H, W]
          → Conv2d(in_ch, out_ch, 1)        → [B, out_ch, H, W]

    Parameters (no norm, bias=True):
        conv_gap : in_ch * in_ch + in_ch
        conv_out : in_ch * out_ch + out_ch

    Args:
        name (str): Module name.
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        with_norm (bool): Add BN2d after each conv. Default: False.
        eps (float): BN epsilon (if with_norm). Default: 1e-5.
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        with_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_norm = with_norm

        # gap: Conv2d(in, in, 1) [+BN] + AdaptiveAvgPool2d(1)
        self.gap_conv = F.Conv2d(
            name + ".gap_conv",
            in_channels, in_channels, 1,
            bias=(not with_norm),
        )
        if with_norm:
            self.gap_bn = F.BatchNorm2d(name + ".gap_bn", in_channels, epsilon=eps)
        self.gap_pool = F.AdaptiveAvgPool2d(name + ".gap_pool", output_size=1)

        # resize: Resize back to input spatial size (will be set at call-time)
        # add: implicit, handled with SimOpHandle Add
        self.add_op = SimOpHandle(name + ".add", "Add", params=[], ipos=[0, 1])

        # output conv: Conv2d(in, out, 1) [+BN]
        self.out_conv = F.Conv2d(
            name + ".out_conv",
            in_channels, out_channels, 1,
            bias=(not with_norm),
        )
        if with_norm:
            self.out_bn = F.BatchNorm2d(name + ".out_bn", out_channels, epsilon=eps)

        super().link_op2module()

    def __call__(self, x):
        B, C, H, W = x.shape

        # gap branch
        y = self.gap_conv(x)
        if self.with_norm:
            y = self.gap_bn(y)
        y = self.gap_pool(y)   # [B, C, 1, 1]

        # resize gap to input spatial
        # GAP output is [B, C, 1, 1]; scale by [H, W] to restore to [B, C, H, W]
        y_resize = F.Resize(
            self.name + ".resize",
            scale_factor=[float(H), float(W)],
            mode="nearest",
            nearest_mode="floor",
            coordinate_transformation_mode="asymmetric",
        )(y)

        # add
        y_add = self.add_op(x, y_resize)  # [B, C, H, W]

        # output conv
        out = self.out_conv(y_add)
        if self.with_norm:
            out = self.out_bn(out)
        return out

    def analytical_param_count(self, lvl: int = 0) -> int:
        bias_mul = 0 if self.with_norm else 1
        p = (self.in_channels * self.in_channels + self.in_channels * bias_mul)   # gap_conv
        if self.with_norm:
            p += 2 * self.in_channels   # gap_bn
        p += (self.in_channels * self.out_channels + self.out_channels * bias_mul)  # out_conv
        if self.with_norm:
            p += 2 * self.out_channels  # out_bn
        return p


# ---------------------------------------------------------------------------
# FPNC
# ---------------------------------------------------------------------------

class FPNC(SimNN.Module):
    """
    TTSim Extended Feature Pyramid Network for camera stream (FPNC).

    Builds on the existing FPN ttsim_module, then:
      1. Resizes (or adaptively pools/upsamples) all FPN output levels to
         ``target_size = (final_dim[0]//downsample, final_dim[1]//downsample)``.
      2. Concatenates all resized levels along the channel dimension.
      3. Applies a 3×3 reduction convolution to output ``outC`` channels.

    The result is a single feature map at a fixed spatial resolution.

    Args:
        name (str): Module name prefix.
        in_channels (list[int]): Input channels per backbone level (→ FPN).
        out_channels (int): Uniform FPN output channel width.
        num_outs (int): Number of FPN output levels.
        start_level (int): First backbone level (→ FPN). Default: 0.
        end_level (int): Last backbone level (→ FPN). Default: -1.
        with_norm (bool): BN2d in FPN and adp/reduc convs. Default: False.
        with_act (bool): ReLU in FPN convs. Default: False.
        final_dim (tuple): Actual image size (H, W). Default: (900, 1600).
        downsample (int): Stride for target spatial size. Default: 4.
        use_adp (bool): Per-level adaptive pool/upsample + Conv1×1. Default False.
        outC (int): Output channels from reduc_conv. Default: 256.
        reduc_with_norm (bool): BN2d in reduc_conv. Default: False.
        eps (float): BN epsilon. Default: 1e-5.

    Call:
        outs = fpnc(f0, f1, ...)   # same inputs as FPN
        Returns: list[SimTensor] of length 1, shape [B, outC, H//ds, W//ds]
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
        final_dim: tuple = (900, 1600),
        downsample: int = 4,
        use_adp: bool = False,
        outC: int = 256,
        reduc_with_norm: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name = name
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.use_adp = use_adp
        self.outC = outC
        self.reduc_with_norm = reduc_with_norm
        self.target_size = (final_dim[0] // downsample, final_dim[1] // downsample)

        # Inner FPN
        self.fpn = FPN(
            name + ".fpn",
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            with_norm=with_norm,
            with_act=with_act,
            eps=eps,
        )

        # Adaptive convs (optional)
        if use_adp:
            for i in range(num_outs):
                adp_conv = F.Conv2d(
                    f"{name}.adp_conv{i}",
                    out_channels, out_channels, 1,
                    bias=(not reduc_with_norm),
                )
                setattr(self, f"adp_conv{i}", adp_conv)
                if reduc_with_norm:
                    setattr(self, f"adp_bn{i}",
                            F.BatchNorm2d(f"{name}.adp_bn{i}", out_channels, epsilon=eps))

        # Reduction conv: (out_channels * num_outs → outC, 3×3)
        self.reduc_conv = F.Conv2d(
            name + ".reduc_conv",
            out_channels * num_outs, outC, 3,
            padding=1,
            bias=(not reduc_with_norm),
        )
        if reduc_with_norm:
            self.reduc_bn = F.BatchNorm2d(name + ".reduc_bn", outC, epsilon=eps)

        super().link_op2module()

    def __call__(self, *inputs):
        tH, tW = self.target_size

        # Call FPN
        fpn_outs = self.fpn(*inputs)  # list[SimTensor]

        if len(fpn_outs) == 1:
            return [self.reduc_conv(fpn_outs[0])]

        # Resize all FPN outputs to target_size
        resized = []
        for i, feat in enumerate(fpn_outs):
            _, _, H, W = feat.shape
            if self.use_adp:
                adp_conv = getattr(self, f"adp_conv{i}")
                if i == 0:
                    # AdaptiveAvgPool2d → target_size
                    pooled = F.AdaptiveAvgPool2d(
                        f"{self.name}.adp_pool{i}", output_size=self.target_size
                    )(feat)
                    feat_r = adp_conv(pooled)
                else:
                    # Upsample → target_size
                    feat_r_pre = F.Resize(
                        f"{self.name}.adp_resize{i}",
                        scale_factor=[tH / H, tW / W],
                        mode="nearest",
                        nearest_mode="floor",
                        coordinate_transformation_mode="asymmetric",
                    )(feat)
                    feat_r = adp_conv(feat_r_pre)
                if self.reduc_with_norm:
                    feat_r = getattr(self, f"adp_bn{i}")(feat_r)
                resized.append(feat_r)
            else:
                if (H, W) != (tH, tW):
                    feat = F.Resize(
                        f"{self.name}.resize{i}",
                        scale_factor=[tH / H, tW / W],
                        mode="nearest",
                        nearest_mode="floor",
                        coordinate_transformation_mode="asymmetric",
                    )(feat)
                resized.append(feat)

        # Concatenate all levels
        cat = F.ConcatX(self.name + ".cat", axis=1)(*resized)

        # Reduce
        out = self.reduc_conv(cat)
        if self.reduc_with_norm:
            out = self.reduc_bn(out)
        return [out]

    def analytical_param_count(self, lvl: int = 0) -> int:
        p = self.fpn.analytical_param_count(lvl + 1)

        if self.use_adp:
            bias_mul = 0 if self.reduc_with_norm else 1
            for i in range(self.num_outs):
                p += self.out_channels * self.out_channels + self.out_channels * bias_mul
                if self.reduc_with_norm:
                    p += 2 * self.out_channels

        bias_mul = 0 if self.reduc_with_norm else 1
        p += (self.out_channels * self.num_outs) * self.outC * 9 + self.outC * bias_mul
        if self.reduc_with_norm:
            p += 2 * self.outC
        return p
