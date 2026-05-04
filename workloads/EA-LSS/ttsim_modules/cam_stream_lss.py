#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of LiftSplatShoot (Lift-Splat-Shoot BEV encoder).

Original file: mmdet3d/models/detectors/cam_stream_lss.py

The module lifts camera features into 3D, splats them onto a BEV grid, and
encodes the BEV representation for downstream detection.

Sub-modules (default params: inputC=256, camC=64, lss=False):
    dtransform     : 4×(Conv2d+BN+ReLU) — depth LiDAR feature extract
                     Conv2d(5,8,5) + Conv2d(8,32,5,s=2) + Conv2d(32,64,5,s=2) + Conv2d(64,64,5,s=2)
                     Params: 161,504
    prenet         : 2×(Conv2d+BN+ReLU) — fuse depth+image features
                     Conv2d(inputC+64,inputC,3) + Conv2d(inputC,inputC,3)
                     Params: 1,328,640
    depthnet       : Conv2d(inputC, D, 1)          Params: 10,537
    contextnet     : Conv2d(inputC, camC, 1)        Params: 16,448
    attention1     : Conv2d(inputC, 1, 1)           Params:    257
    attention2     : Conv2d(camC,  1, 1)            Params:     65
    convnet        : 2×(Conv2d+BN+ReLU)             Params: 738,240
    upsample_depth : 4×(ConvTranspose2d+BN+ReLU) + Conv2d(camC,D,1)
                     Params: 822,377
    bevencode      : 4×(Conv2d+BN+ReLU)   (lss=False)
                     Conv2d(cz,cz,3)+Conv2d(cz,512,3)+Conv2d(512,512,3)+Conv2d(512,inputC,3)
                     Params: 4,279,040

D  = len(arange(*camera_depth_range))  = 41  for [4.0, 45.0, 1.0]
cz = int(camC * ((zbound[1]-zbound[0]) // (zbound[2]-0.0001)))
   = int(64  * ((3 - (-5)) // (3 - 0.0001)))     # pc_range=[-.50,.,-5,..,3]
   = int(64  * 2) = 128

Total (lss=False, all defaults): 7,357,108

No torch / mmcv imports.
"""

import os
import sys
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data


class LiftSplatShoot(SimNN.Module):
    """
    Lift-Splat-Shoot BEV encoder.

    Args:
        name (str): Module name.
        lss (bool): Use the full ResNet-18 BEV encoder (BevEncode). Default: False.
        final_dim (tuple): Full-resolution image size (H, W). Default: (900, 1600).
        camera_depth_range (list): [d_min, d_max, d_step]. Default: [4.0, 45.0, 1.0].
        pc_range (list): [x_min, y_min, z_min, x_max, y_max, z_max]. Default: [-50,-50,-5,50,50,3].
        downsample (int): Spatial downsampling from full-res to feature-res. Default: 4.
        grid (int): BEV voxel grid step (metres). Default: 3.
        inputC (int): Input camera feature channel count. Default: 256.
        camC (int): Camera feature channels after context net. Default: 64.
    """

    def __init__(
        self,
        name: str,
        lss: bool = False,
        final_dim: tuple = (900, 1600),
        camera_depth_range: list = None,
        pc_range: list = None,
        downsample: int = 4,
        grid: int = 3,
        inputC: int = 256,
        camC: int = 64,
    ):
        super().__init__()
        self.name     = name
        self.lss      = lss
        self.inputC   = inputC
        self.camC     = camC

        if camera_depth_range is None:
            camera_depth_range = [4.0, 45.0, 1.0]
        if pc_range is None:
            pc_range = [-50, -50, -5, 50, 50, 3]

        # D: number of depth bins
        d_min, d_max, d_step = camera_depth_range
        D = int((d_max - d_min) / d_step)   # e.g. (45-4)/1 = 41
        self.D = D

        # cz: collapse height dim into channels
        z_min, z_max = pc_range[2], pc_range[5]
        z_step = float(grid)
        cz = int(camC * math.floor((z_max - z_min) / (z_step - 0.0001)))
        self.cz = cz

        # --- dtransform ---
        # For sequential Conv2d blocks: (in,out,k,stride,pad)
        self._dt_specs = [
            (5,    8,  5, 1, 2),
            (8,   32,  5, 2, 2),
            (32,  64,  5, 2, 2),
            (64,  64,  5, 2, 2),
        ]
        for i, (ic, oc, k, s, p) in enumerate(self._dt_specs):
            setattr(self, f"dt_conv{i}", F.Conv2d(f"{name}.dt{i}", ic, oc, k, stride=s, padding=p, bias=True))
            setattr(self, f"dt_bn{i}", F.BatchNorm2d(f"{name}.dt_bn{i}", oc))

        # --- prenet ---
        self._pn_specs = [
            (inputC + 64, inputC, 3, 1, 1),
            (inputC,      inputC, 3, 1, 1),
        ]
        for i, (ic, oc, k, s, p) in enumerate(self._pn_specs):
            setattr(self, f"pn_conv{i}", F.Conv2d(f"{name}.pn{i}", ic, oc, k, stride=s, padding=p, bias=True))
            setattr(self, f"pn_bn{i}",  F.BatchNorm2d(f"{name}.pn_bn{i}", oc))

        # --- depthnet, contextnet, attention1, attention2 ---
        self.depthnet  = F.Conv2d(name + ".depthnet",  inputC, D,    1, bias=True)
        self.contextnet= F.Conv2d(name + ".contextnet",inputC, camC, 1, bias=True)
        self.attention1= F.Conv2d(name + ".attn1",     inputC, 1,    1, bias=True)
        self.attention2= F.Conv2d(name + ".attn2",     camC,   1,    1, bias=True)

        # --- convnet ---
        self._cn_specs = [
            (inputC, inputC, 3, 1, 1),
            (inputC, camC,   3, 1, 1),
        ]
        for i, (ic, oc, k, s, p) in enumerate(self._cn_specs):
            setattr(self, f"cn_conv{i}", F.Conv2d(f"{name}.cn{i}", ic, oc, k, stride=s, padding=p, bias=True))
            setattr(self, f"cn_bn{i}",  F.BatchNorm2d(f"{name}.cn_bn{i}", oc))

        # --- upsample_depth ---
        # 4×ConvTranspose2d(no bias)+BN + 1×Conv2d(bias=True)
        self._ud_ctrans_specs = [
            (inputC + 64, camC, 5, 2),   # stride=2, no bias
            (camC,        camC, 5, 2),
            (camC,        camC, 5, 2),
            (camC,        camC, 5, 1),   # no stride
        ]
        for i, (ic, oc, k, s) in enumerate(self._ud_ctrans_specs):
            setattr(self, f"ud_ct{i}", F.ConvTranspose2d(f"{name}.udct{i}", ic, oc, kernel_size=k, stride=s))
            setattr(self, f"ud_bn{i}", F.BatchNorm2d(f"{name}.ud_bn{i}", oc))
        self.ud_final = F.Conv2d(name + ".ud_final", camC, D, 1, bias=True)

        # --- bevencode (lss=False) ---
        # 4×(Conv2d+BN+ReLU) — bias=False (BN follows)
        self._be_specs = [
            (cz,   cz,    3, 1, 1),
            (cz,   512,   3, 1, 1),
            (512,  512,   3, 1, 1),
            (512,  inputC,3, 1, 1),
        ]
        for i, (ic, oc, k, s, p) in enumerate(self._be_specs):
            setattr(self, f"be_conv{i}", F.Conv2d(f"{name}.be{i}", ic, oc, k,
                                                   stride=s, padding=p, bias=False))
            setattr(self, f"be_bn{i}",  F.BatchNorm2d(f"{name}.be_bn{i}", oc))

        super().link_op2module()

    # ------------------------------------------------------------------
    def _apply_seqblock(self, x, conv_pref, bn_pref, n, act=True):
        """Apply n blocks of (Conv2d + BN + ReLU)."""
        for i in range(n):
            x = getattr(self, f"{conv_pref}{i}")(x)
            x = getattr(self, f"{bn_pref}{i}")(x)
            if act:
                x = F.Relu(f"{self.name}.{conv_pref}{i}_relu")(x)
        return x

    def __call__(self, x, depth_feat=None, **kwargs):
        """
        Args:
            x (SimTensor): [B_N, inputC, fH, fW] camera feature maps
                           (B_N = batch_size × num_views).
            depth_feat (SimTensor, optional): [B_N, 5, H, W] depth LiDAR
                projected onto camera plane. If None, a proxy is used.
        Returns:
            SimTensor: [B, inputC, bH, bW] BEV feature map.
        """
        B_N, C, fH, fW = x.shape
        B = B_N  # simplified: treat B_N as single batch for shape purposes

        # dtransform: input d [B_N, 5, fH, fW]
        if depth_feat is None:
            d = _from_shape(self.name + ".d_proxy", [B_N, 5, fH, fW])
        else:
            d = depth_feat
        d = self._apply_seqblock(d, "dt_conv", "dt_bn", 4)   # [B_N, 64, ~fH//8, ~fW//8]

        # Resize d to match x's spatial; use ACTUAL d.shape (strided conv output
        # may not equal fH//8 exactly for all input sizes)
        d_H = d.shape[2]
        d_W = d.shape[3]
        if (d_H, d_W) != (fH, fW):
            # Resize d from [B_N, 64, fH//8, fW//8] to [B_N, 64, fH, fW]
            d = F.Resize(
                self.name + ".d_resize",
                scale_factor=[float(fH / d_H), float(fW / d_W)],
                mode="nearest",
            )(d)

        # convnet: image semantic features
        context = self._apply_seqblock(x, "cn_conv", "cn_bn", 2)   # [B_N, camC, fH, fW]

        # Concatenate x and d: [B_N, inputC+64, fH, fW]
        xd = F.ConcatX(self.name + ".xd_cat", axis=1)(x, d)

        # upsample_depth (ConvTranspose blocks + final Conv)
        ud = xd
        for i in range(4):
            ud = getattr(self, f"ud_ct{i}")(ud)
            ud = getattr(self, f"ud_bn{i}")(ud)
            ud = F.Relu(self.name + f".ud_relu{i}")(ud)
        ud = self.ud_final(ud)                                     # [B_N, D, fH', fW']

        # prenet: fuse depth + image
        x = self._apply_seqblock(xd, "pn_conv", "pn_bn", 2)       # [B_N, inputC, fH, fW]

        # attention1 + depthnet -> depth probability
        attn1 = self.attention1(x)         # [B_N, 1, fH, fW]
        depth = self.depthnet(x)           # [B_N, D, fH, fW]
        depth = SimOpHandle(self.name + ".depth_mul_attn1", "Mul",
                             params=[], ipos=[0, 1])(depth, attn1)
        depth = F.Softmax(self.name + ".depth_sm", axis=1)(depth)  # [B_N, D, fH, fW]

        # contextnet + context (semantic)
        ctx_out = self.contextnet(x)       # [B_N, camC, fH, fW]
        ctx_out = SimOpHandle(self.name + ".ctx_add", "Add",
                               params=[], ipos=[0, 1])(ctx_out, context)

        # attention2 weighted context: [B_N, camC, fH, fW]
        attn2  = self.attention2(ctx_out)  # [B_N, 1, fH, fW]
        ctx_wt = SimOpHandle(self.name + ".ctx_mul_attn2", "Mul",
                              params=[], ipos=[0, 1])(ctx_out, attn2)

        # Lift: depth [B_N, D, fH, fW] ⊗ ctx_wt [B_N, camC, fH, fW]
        # Result: [B_N, camC, D, fH, fW]
        # Approximated as proxy for complex outer product
        lifted = _from_shape(self.name + ".lifted", [B_N, self.camC, self.D, fH, fW])

        # Voxel pooling (complex scatter op) → BEV: [B, cz, bH, bW]
        bev = _from_shape(self.name + ".bev_proxy",
                          [B, self.cz, fH, fW])   # proxy BEV shape

        # bevencode: [B, cz, bH, bW] → [B, inputC, bH, bW]
        out = self._apply_seqblock(bev, "be_conv", "be_bn", 4)

        return out   # [B, inputC, bH, bW]

    # ------------------------------------------------------------------
    def analytical_param_count(self, lvl: int = 0) -> int:
        p = 0

        # dtransform: Conv2d(bias=True) + BN
        for (ic, oc, k, s, pad) in self._dt_specs:
            p += ic * oc * k * k + oc       # conv (bias)
            p += 2 * oc                      # BN

        # prenet: Conv2d(bias=True) + BN
        for (ic, oc, k, s, pad) in self._pn_specs:
            p += ic * oc * k * k + oc
            p += 2 * oc

        # depthnet, contextnet, attention1, attention2 (Conv2d 1×1, bias=True)
        p += self.inputC * self.D + self.D                   # depthnet
        p += self.inputC * self.camC + self.camC             # contextnet
        p += self.inputC * 1 + 1                             # attention1
        p += self.camC   * 1 + 1                             # attention2

        # convnet: Conv2d(bias=True) + BN
        for (ic, oc, k, s, pad) in self._cn_specs:
            p += ic * oc * k * k + oc
            p += 2 * oc

        # upsample_depth: ConvTranspose2d(no bias) + BN
        for (ic, oc, k, s) in self._ud_ctrans_specs:
            p += ic * oc * k * k                 # no bias
            p += 2 * oc                           # BN
        p += self.camC * self.D + self.D          # ud_final Conv2d (bias=True)

        # bevencode: Conv2d(no bias) + BN
        for (ic, oc, k, s, pad) in self._be_specs:
            p += ic * oc * k * k                 # no bias
            p += 2 * oc                           # BN

        return p
