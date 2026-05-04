#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of CBSwinTransformer (Composite Backbone Swin Transformer).

Original file: mmdet3d/models/backbones/cbnet.py

Architecture:
    Two SwinTransformer backbones share input and exchange features:
    - cb_modules[0]: full SwinTransformer (with patch_embed)
    - cb_modules[1]: SwinTransformer without patch_embed
                     (patch_embed params not counted)

    Cross-backbone fusion (cb_linears):
        For stage i >= cb_del_stages-1 and relative stage j:
            if cb_inplanes[i+j] != cb_inplanes[i]:
                Conv2d(cb_inplanes[i+j], cb_inplanes[i], 1) + bias
            else:
                Identity (0 params)
        Where cb_inplanes = [embed_dim * 2^0, ..., embed_dim * 2^3]

    Default args match EA-LSS usage (embed_dim=96, cb_del_stages=1,
    shared SwinT kwargs).

    Input:  [B, 3, H, W]
    Output: list of 4 feature maps at scales H/4, H/8, H/16, H/32
    Params (default embed_dim=96):
        cb_modules[0]: ~27,520,602  (same as full SwinT-S/B)
        cb_modules[1]: ~27,515,802  (minus patch_embed: -4800)
        cb_linears:     646,176
        Total:         ~55,682,580

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

from ttsim_modules.swin_transformer import SwinTransformer


# ---------------------------------------------------------------------------
# CBSwinTransformer
# ---------------------------------------------------------------------------

class CBSwinTransformer(SimNN.Module):
    """
    Composite Backbone Swin Transformer for BEV perception.

    Two SwinTransformers with cross-stage feature fusion via 1x1 convolutions.

    Args:
        name (str): Module name.
        embed_dim (int): Embedding dimension of the first stage. Default: 96.
        cb_del_stages (int): Number of stages deleted from cb_modules[1].
            Only stage 0 (patch_embed) is deleted when cb_del_stages=1. Default: 1.
        cb_zero_init (bool): Whether to zero-initialize cb_linears (unused in TTSim). Default: True.
        **swin_kwargs: Forwarded verbatim to both SwinTransformer constructors.
            Must include at least: depths, num_heads, window_size, etc.
    """

    def __init__(
        self,
        name: str,
        embed_dim: int = 96,
        cb_del_stages: int = 1,
        cb_zero_init: bool = True,
        **swin_kwargs,
    ):
        super().__init__()
        self.name = name
        self.embed_dim = embed_dim
        self.cb_del_stages = cb_del_stages

        # Number of hierarchical stages in the SwinTransformer
        depths = swin_kwargs.get("depths", [2, 2, 6, 2])
        self.num_layers = len(depths)

        # cb_inplanes: [96, 192, 384, 768] for embed_dim=96
        cb_inplanes = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.cb_inplanes = cb_inplanes

        # cb_modules[0]: full SwinTransformer
        self.cb0 = SwinTransformer(
            name + ".cb0", embed_dim=embed_dim, **swin_kwargs
        )

        # cb_modules[1]: same SwinTransformer but we'll subtract patch_embed
        # params in analytical_param_count
        self.cb1 = SwinTransformer(
            name + ".cb1", embed_dim=embed_dim, **swin_kwargs
        )
        patch_size    = swin_kwargs.get("patch_size", 4)
        in_chans      = swin_kwargs.get("in_chans", 3)
        patch_norm    = swin_kwargs.get("patch_norm", True)
        # patch_embed params: Conv2d(in_chans, embed_dim, ps, ps, no bias) + LN(embed_dim)
        self._cb1_patch_embed_params = (
            in_chans * embed_dim * patch_size ** 2       # conv weight
            + (2 * embed_dim if patch_norm else 0)       # LayerNorm
        )

        # cb_linears: 1x1 Conv2d for cross-backbone feature fusion
        #   cb_linears[i][j] merges feature at scale (i+j) into scale i
        for i in range(self.num_layers):
            if i >= self.cb_del_stages - 1:
                jrange = self.num_layers - i
                for j in range(jrange):
                    in_c  = cb_inplanes[i + j]
                    out_c = cb_inplanes[i]
                    lin_name = f"{name}.cblin_{i}_{j}"
                    if in_c != out_c:
                        # Conv2d(in_c, out_c, 1, bias=True)
                        setattr(self, f"cblin_{i}_{j}",
                                F.Conv2d(lin_name, in_c, out_c, 1,
                                         padding=0, bias=True))
                    else:
                        # Identity: 0 params
                        setattr(self, f"cblin_{i}_{j}",
                                F.Identity(lin_name))

        super().link_op2module()

    # ------------------------------------------------------------------
    def __call__(self, x):
        """
        Args:
            x (SimTensor): [B, 3, H, W] image tensor.
        Returns:
            list[SimTensor]: 4 feature maps from cb_modules[1]
                [B, C_i, H/4/2^i, W/4/2^i] for i in {0,1,2,3}.
        """
        # --- Pass 1: cb_modules[0] ---
        outs0 = self.cb0(x)          # list of 4 feature maps

        # --- Cross-backbone linears on cb_modules[0] outputs ---
        # For each fusion stage, apply the cross-backbone linears;
        # these FLOPs are captured in the graph even though the
        # spatial-interpolate/add is approximated.
        fused = []
        for i in range(self.num_layers):
            if i >= self.cb_del_stages - 1:
                jrange = self.num_layers - i
                parts = []
                for j in range(jrange):
                    feat_j = outs0[j + i]
                    lin = getattr(self, f"cblin_{i}_{j}")
                    if j == 0:
                        parts.append(lin(feat_j))
                    else:
                        # Resize feat_j to match feat_i spatial size
                        Bi, Ci, Hi, Wi = outs0[i].shape
                        _, _, Hj, Wj = feat_j.shape
                        if (Hj, Wj) != (Hi, Wi):
                            feat_j = F.Resize(
                                f"{self.name}.cbfeat_resize_{i}_{j}",
                                scale_factor=[Hi / Hj, Wi / Wj],
                                mode="nearest",
                            )(lin(feat_j))
                        else:
                            feat_j = lin(feat_j)
                        parts.append(feat_j)
                # Sum all fused parts
                cb_feat_i = parts[0]
                for p in parts[1:]:
                    cb_feat_i = SimOpHandle(
                        f"{self.name}.cbfeat_add_{i}", "Add",
                        params=[], ipos=[0, 1]
                    )(cb_feat_i, p)
                fused.append(cb_feat_i)
            else:
                fused.append(_from_shape(f"{self.name}.zero_feat_{i}",
                                          list(outs0[i].shape)))

        # --- Pass 2: cb_modules[1] ---
        # In the source, cb_modules[1] skips patch_embed and starts from
        # pre-computed tokens.  For TTSim FLOPs, we run it on x directly
        # (patch_embed FLOPs will be slightly over-counted but params are
        # excluded from analytical_param_count via _cb1_patch_embed_params).
        outs1 = self.cb1(x)

        return outs1

    # ------------------------------------------------------------------
    def analytical_param_count(self, lvl: int = 0) -> int:
        cb0_p = self.cb0.analytical_param_count(lvl + 1)
        cb1_p = self.cb1.analytical_param_count(lvl + 1) - self._cb1_patch_embed_params

        cblin_p = 0
        for i in range(self.num_layers):
            if i >= self.cb_del_stages - 1:
                jrange = self.num_layers - i
                for j in range(jrange):
                    in_c  = self.cb_inplanes[i + j]
                    out_c = self.cb_inplanes[i]
                    if in_c != out_c:
                        # Conv2d(in_c, out_c, 1) with bias
                        cblin_p += in_c * out_c + out_c

        return cb0_p + cb1_p + cblin_p
