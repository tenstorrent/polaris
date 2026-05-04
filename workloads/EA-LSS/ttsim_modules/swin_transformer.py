#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of SwinTransformer backbone.

Original file: mmdet3d/models/backbones/swin_transformer.py

Implements the Swin Transformer (Liu et al., ICCV 2021) for use as an
image backbone in EALSS.

Architecture:
    PatchEmbed (Conv2d patch projection + LayerNorm)
    Stage 0: depth[0] × SwinTransformerBlock + PatchMerging
    Stage 1: depth[1] × SwinTransformerBlock + PatchMerging
    ...
    Stage N-1: depth[N-1] × SwinTransformerBlock (no merge)
    Per-stage LayerNorm for output indices

Key differences from original for TTSim:
  - Window partitioning / cyclic shift are identity for shape inference
    (they rearrange tokens within [B, N, C] without changing N or C).
  - Relative position bias table is a learnable parameter tensor.
  - Dropout / stochastic depth have zero parameters and are omitted.

Default configuration (Swin-T as used in EALSS):
    embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7

No torch / timm / mmcv imports.
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
# Linear helper that avoids the SimNN.Linear Y += bias bug
# ---------------------------------------------------------------------------

class _LinearModule(SimNN.Module):
    """
    Linear layer: y = x @ W + b

    Uses F.Linear (a MatMul SimOpHandle with params=[(1, W)]) plus an
    explicit bias-Add SimOpHandle.  This avoids the ``Y += self.bias``
    TypeError present in SimNN.Linear when SimTensor arithmetic is not
    overloaded.

    Weight shape: [in_features, out_features]  (ONNX MatMul style)
    Bias shape:   [out_features]

    Args:
        name (str): Module name.
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        bias (bool): Include additive bias. Default: True.
    """

    def __init__(self, name: str, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # F.Linear creates SimOpHandle("MatMul", params=[(1, W)], ipos=[0])
        # with W shape = [in_features, out_features]
        self.lin = F.Linear(name + ".lin", in_features, out_features)

        if bias:
            self.bias_t = _from_shape(name + ".bias", [out_features], is_param=True)
            self.bias_t.op_in.append(name)
            self.bias_add = SimOpHandle(
                name + ".bias_add", "Add",
                params=[(1, self.bias_t)], ipos=[0]
            )
            self.bias_add.implicit_inputs.append(self.bias_t)

        super().link_op2module()

    def __call__(self, x):
        y = self.lin(x)
        if self.use_bias:
            y = self.bias_add(y)
        return y

    def analytical_param_count(self, lvl: int = 0) -> int:
        return self.in_features * self.out_features + (self.out_features if self.use_bias else 0)


# ---------------------------------------------------------------------------
# Window Attention sub-module
# ---------------------------------------------------------------------------

class SwinWindowAttention(SimNN.Module):
    """
    Window-based multi-head self-attention with relative position bias.

    Learnable parameters:
        QKV projection:  Linear(dim, 3*dim, bias=True)
        relative_pos_bias_table: [(2*ws-1)^2, num_heads]
        output projection: Linear(dim, dim, bias=True)

    For TTSim shape inference, the window partitioning is modelled as a
    passthrough (input [B, N, dim] → output [B, N, dim]).  The actual
    attention matrix [B*nW, nH, ws²,ws²] is computed but its spatial
    arrangement does not affect downstream shapes.

    Args:
        name (str): Module name.
        dim (int): Feature dimension.
        window_size (int): Local window size (ws). Default: 7.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, name: str, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.name = name
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Use SEPARATE Q, K, V projections.
        # Combined in_proj_weight[3D,D]+bias[3D] has the same param count as
        # three separate Linear(D,D,bias=True): 3*(D^2+D) = 3D^2+3D.
        self.q_proj = _LinearModule(name + ".q_proj", dim, dim, bias=True)
        self._submodules[self.q_proj.name] = self.q_proj
        self.k_proj = _LinearModule(name + ".k_proj", dim, dim, bias=True)
        self._submodules[self.k_proj.name] = self.k_proj
        self.v_proj = _LinearModule(name + ".v_proj", dim, dim, bias=True)
        self._submodules[self.v_proj.name] = self.v_proj

        # Relative position bias table: learnable [(2ws-1)^2, nH]
        rp_size = (2 * window_size - 1) ** 2
        self.rel_pos_bias = _from_shape(
            name + ".rel_pos_bias", [rp_size, num_heads], is_param=True
        )
        self.rel_pos_bias.op_in.append(name)

        # Output projection: weight [dim, dim], bias [dim]
        self.proj = _LinearModule(name + ".proj", dim, dim, bias=True)
        self._submodules[self.proj.name] = self.proj

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [B, N, dim]
        Returns:
            SimTensor: [B, N, dim]
        """
        B, N, C = x.shape
        nH = self.num_heads
        hd = self.head_dim

        # Separate Q, K, V projections: each [B, N, dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [B, N, nH, head_dim]  (elements: B*N*dim = B*N*nH*hd ✓)
        mh_q = _from_data(self.name + ".mh_q", np.array([B, N, nH, hd], dtype=np.int64), is_const=True)
        mh_k = _from_data(self.name + ".mh_k", np.array([B, N, nH, hd], dtype=np.int64), is_const=True)
        mh_v = _from_data(self.name + ".mh_v", np.array([B, N, nH, hd], dtype=np.int64), is_const=True)
        q_4d = F.Reshape(self.name + ".q_rs")(q, mh_q)       # [B, N, nH, hd]
        k_4d = F.Reshape(self.name + ".k_rs")(k, mh_k)       # [B, N, nH, hd]
        v_4d = F.Reshape(self.name + ".v_rs")(v, mh_v)       # [B, N, nH, hd]

        # Transpose to [B, nH, N, head_dim]
        q_t = F.Transpose(self.name + ".q_tr", perm=[0, 2, 1, 3])(q_4d)
        k_t = F.Transpose(self.name + ".k_tr", perm=[0, 2, 1, 3])(k_4d)
        v_t = F.Transpose(self.name + ".v_tr", perm=[0, 2, 1, 3])(v_4d)

        # Scale Q
        scale = _from_data(
            self.name + ".scale",
            np.array([1.0 / np.sqrt(hd)], dtype=np.float32),
            is_const=True,
        )
        scale_op = SimOpHandle(self.name + ".q_scale", "Mul", params=[(1, scale)], ipos=[0])
        scale_op.implicit_inputs.append(scale)
        q_scaled = scale_op(q_t)                               # [B, nH, N, hd]

        # Attention: Q @ K^T → [B, nH, N, N]
        k_tt = F.Transpose(self.name + ".kk_tr", perm=[0, 1, 3, 2])(k_t)  # [B, nH, hd, N]
        attn = F.MatMul(self.name + ".attn_qk")(q_scaled, k_tt)            # [B, nH, N, N]
        attn_s = F.Softmax(self.name + ".softmax", axis=-1)(attn)

        # Weighted V: [B, nH, N, N] @ [B, nH, N, hd] → [B, nH, N, hd]
        x_attn = F.MatMul(self.name + ".attn_v")(attn_s, v_t)

        # Transpose/reshape back to [B, N, dim]
        x_attn_t = F.Transpose(self.name + ".out_tr", perm=[0, 2, 1, 3])(x_attn)  # [B, N, nH, hd]
        out_shape = _from_data(self.name + ".out_shape", np.array([B, N, C], dtype=np.int64), is_const=True)
        x_out = F.Reshape(self.name + ".out_rs")(x_attn_t, out_shape)              # [B, N, dim]

        return self.proj(x_out)

    def analytical_param_count(self, lvl: int = 0) -> int:
        rp_size = (2 * self.window_size - 1) ** 2
        # q,k,v projections (same total as combined QKV): 3*(D^2+D)
        qkv_params  = 3 * (self.dim * self.dim + self.dim)
        rp_params   = rp_size * self.num_heads
        proj_params = self.dim * self.dim + self.dim
        return qkv_params + rp_params + proj_params


# ---------------------------------------------------------------------------
# MLP sub-module for SwinTransformerBlock
# ---------------------------------------------------------------------------

class SwinMlp(SimNN.Module):
    """
    Two-layer MLP with GELU activation used inside each SwinTransformerBlock.

    in_features → hidden_features (int(in_features*mlp_ratio)) → in_features

    Args:
        name (str): Module name.
        in_features (int): Input / output dimension.
        mlp_ratio (float): Hidden dimension multiplier. Default: 4.
    """

    def __init__(self, name: str, in_features: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.hidden_features = int(in_features * mlp_ratio)

        self.fc1 = _LinearModule(name + ".fc1", in_features, self.hidden_features, bias=True)
        self._submodules[self.fc1.name] = self.fc1

        self.fc2 = _LinearModule(name + ".fc2", self.hidden_features, in_features, bias=True)
        self._submodules[self.fc2.name] = self.fc2

        super().link_op2module()

    def __call__(self, x):
        x = self.fc1(x)
        x = F.Gelu(self.name + ".gelu")(x)
        x = self.fc2(x)
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        fc1_params = self.in_features * self.hidden_features + self.hidden_features
        fc2_params = self.hidden_features * self.in_features + self.in_features
        return fc1_params + fc2_params


# ---------------------------------------------------------------------------
# SwinTransformerBlock
# ---------------------------------------------------------------------------

class SwinTransformerBlock(SimNN.Module):
    """
    One Swin Transformer block: LN + WindowAttention + residual +
                                 LN + MLP + residual.

    For TTSim shape inference, window partitioning / cyclic shift are
    treated as token-preserving (no spatial rearrangement is modelled).

    Args:
        name (str): Module name.
        dim (int): Feature channels.
        num_heads (int): Attention heads.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        # LayerNorm layers (scale + bias each, count = dim)
        self.norm1 = F.LayerNorm(name + ".norm1", dim)
        self.norm2 = F.LayerNorm(name + ".norm2", dim)

        # Window attention
        self.attn = SwinWindowAttention(
            name + ".attn", dim, window_size=window_size, num_heads=num_heads
        )
        self._submodules[self.attn.name] = self.attn

        # MLP
        self.mlp = SwinMlp(name + ".mlp", dim, mlp_ratio=mlp_ratio)
        self._submodules[self.mlp.name] = self.mlp

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [B, N, dim]
        Returns:
            SimTensor: [B, N, dim]
        """
        # Attention branch with residual
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1)
        x = F.Add(self.name + ".add1")(x, attn_out)

        # MLP branch with residual
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = F.Add(self.name + ".add2")(x, mlp_out)
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        norm_params = 2 * self.dim * 2        # two LN layers (scale + bias each)
        attn_params = self.attn.analytical_param_count(lvl + 1)
        mlp_params  = self.mlp.analytical_param_count(lvl + 1)
        return norm_params + attn_params + mlp_params


# ---------------------------------------------------------------------------
# PatchMerging
# ---------------------------------------------------------------------------

class SwinPatchMerging(SimNN.Module):
    """
    Patch merging layer: reduces spatial resolution by 2× and doubles channels.

    Input  [B, H*W, C]  → concat 2×2 patches → [B, H/2*W/2, 4C]
                        → LayerNorm(4C)
                        → Linear(4C, 2C, bias=False)
                        → output [B, H*W/4, 2C]

    For TTSim, the 2×2 patch gathering is modelled as a Reshape + ConcatX
    (shape-only placeholder) that produces [B, N//4, 4*C].

    Args:
        name (str): Module name.
        dim (int): Input channels C.
    """

    def __init__(self, name: str, dim: int):
        super().__init__()
        self.name = name
        self.dim = dim

        self.norm = F.LayerNorm(name + ".norm", 4 * dim)
        self.reduction = _LinearModule(name + ".reduction", 4 * dim, 2 * dim, bias=False)
        self._submodules[self.reduction.name] = self.reduction

        super().link_op2module()

    def __call__(self, x, H: int, W: int):
        """
        Args:
            x (SimTensor): [B, H*W, C]
            H, W (int): Spatial dimensions of the current stage.
        Returns:
            (SimTensor, int, int):
                - merged: [B, H'*W', 2C] where H'=⌈H/2⌉, W'=⌈W/2⌉
                - H_out, W_out
        """
        B, N, C = x.shape
        H_out = (H + 1) // 2
        W_out = (W + 1) // 2

        # Model the 2×2 gather as a shape-only reshape: [B, N, C] → [B, N//4, 4C]
        merged_shape = _from_data(
            self.name + ".merged_shape",
            np.array([B, H_out * W_out, 4 * C], dtype=np.int64),
            is_const=True,
        )
        x_merged = F.Reshape(self.name + ".gather_reshape")(x, merged_shape)
        # [B, H'*W', 4C]

        x_normed = self.norm(x_merged)
        x_out = self.reduction(x_normed)     # [B, H'*W', 2C]
        return x_out, H_out, W_out

    def analytical_param_count(self, lvl: int = 0) -> int:
        norm_params = 2 * (4 * self.dim)               # LN scale + bias for 4C
        linear_params = (4 * self.dim) * (2 * self.dim)  # no bias
        return norm_params + linear_params


# ---------------------------------------------------------------------------
# BasicLayer (one stage of the SwinTransformer)
# ---------------------------------------------------------------------------

class SwinBasicLayer(SimNN.Module):
    """
    One stage of the SwinTransformer: depth × SwinTransformerBlock +
    optional PatchMerging downsampler.

    Args:
        name (str): Module name.
        dim (int): Feature channels for this stage.
        depth (int): Number of SwinTransformerBlocks.
        num_heads (int): Attention heads.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        downsample (bool): If True, add PatchMerging after the blocks.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        downsample: bool = True,
    ):
        super().__init__()
        self.name = name
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size

        for i in range(depth):
            blk = SwinTransformerBlock(
                name=f"{name}.blk{i}",
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            setattr(self, f"blk{i}", blk)

        self.has_downsample = downsample
        if downsample:
            self.downsample = SwinPatchMerging(name + ".downsample", dim)
            self._submodules[self.downsample.name] = self.downsample

        super().link_op2module()

    def __call__(self, x, H: int, W: int):
        """
        Args:
            x (SimTensor): [B, H*W, dim]
            H, W (int): Spatial dimensions.
        Returns:
            (x_out, H_out, W_out, x_down, H_down, W_down)
              - x_out: stage output BEFORE downsampling [B, H*W, dim]
              - x_down: stage output AFTER downsampling [B, H'*W', 2*dim]
                (or same as x_out if no downsampling)
        """
        for i in range(self.depth):
            blk = getattr(self, f"blk{i}")
            x = blk(x)

        x_out = x

        if self.has_downsample:
            x_down, H_out, W_out = self.downsample(x, H, W)
        else:
            x_down, H_out, W_out = x_out, H, W

        return x_out, H, W, x_down, H_out, W_out

    def analytical_param_count(self, lvl: int = 0) -> int:
        total = 0
        for i in range(self.depth):
            blk = getattr(self, f"blk{i}")
            total += blk.analytical_param_count(lvl + 1)
        if self.has_downsample:
            total += self.downsample.analytical_param_count(lvl + 1)
        return total


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

class SwinPatchEmbed(SimNN.Module):
    """
    Image to Patch Embedding via Conv2d.

    Splits the input image into non-overlapping patches using a Conv2d
    with kernel_size=patch_size and stride=patch_size, then optionally
    applies LayerNorm.

    Args:
        name (str): Module name.
        patch_size (int): Patch size. Default: 4.
        in_chans (int): Input image channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 96.
        with_norm (bool): Apply LayerNorm after projection. Default: True.
    """

    def __init__(
        self,
        name: str,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        with_norm: bool = True,
    ):
        super().__init__()
        self.name = name
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.with_norm = with_norm

        # Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj = F.Conv2d(
            name + ".proj",
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=False,
        )

        if with_norm:
            self.norm = F.LayerNorm(name + ".norm", embed_dim)

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [B, in_chans, H, W]
        Returns:
            SimTensor: [B, embed_dim, Wh, Ww]
                       where Wh = H // patch_size, Ww = W // patch_size
        """
        x = self.proj(x)          # [B, embed_dim, Wh, Ww]

        if self.with_norm:
            B, C, Wh, Ww = x.shape
            # Flatten spatial dims for LN: [B, embed_dim, Wh, Ww] → [B, Wh*Ww, embed_dim]
            flat_shape = _from_data(
                self.name + ".flat_shape",
                np.array([B, Wh * Ww, C], dtype=np.int64),
                is_const=True,
            )
            x_flat = F.Reshape(self.name + ".flatten")(x, flat_shape)
            x_flat = self.norm(x_flat)
            # Reshape back: [B, Wh*Ww, embed_dim] → [B, embed_dim, Wh, Ww]
            back_shape = _from_data(
                self.name + ".back_shape",
                np.array([B, C, Wh, Ww], dtype=np.int64),
                is_const=True,
            )
            x = F.Reshape(self.name + ".back_reshape")(x_flat, back_shape)

        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        proj_params = self.in_chans * self.embed_dim * self.patch_size ** 2
        norm_params = 2 * self.embed_dim if self.with_norm else 0
        return proj_params + norm_params


# ---------------------------------------------------------------------------
# SwinTransformer main module
# ---------------------------------------------------------------------------

class SwinTransformer(SimNN.Module):
    """
    TTSim SwinTransformer backbone.

    Processes an input image through patch embedding, multiple hierarchical
    stages of window-based self-attention, and returns feature maps at the
    requested output indices.

    Call signature:
        outs = swin(x)  where x is [B, in_chans, H, W]
        outs[i] is [B, embed_dim * 2^i, H / (patch_size * 2^i), ...]
                   for i in out_indices

    Args:
        name (str): Unique module name prefix.
        pretrain_img_size (int): Unused at inference; kept for API parity.
        patch_size (int): Patch embedding size. Default: 4.
        in_chans (int): Input channels. Default: 3.
        embed_dim (int): Embedding dimension at stage 0. Default: 96.
        depths (list[int]): Block depths per stage. Default: [2,2,6,2].
        num_heads (list[int]): Attention heads per stage.
            Default: [3,6,12,24].
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        out_indices (tuple[int]): Stages to collect output from.
            Default: (0, 1, 2, 3).
        patch_norm (bool): Apply LN in patch embed. Default: True.
    """

    def __init__(
        self,
        name: str,
        pretrain_img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        out_indices: tuple = (0, 1, 2, 3),
        patch_norm: bool = True,
    ):
        super().__init__()
        self.name = name

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.patch_size = patch_size

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # Patch embedding
        self.patch_embed = SwinPatchEmbed(
            name + ".patch_embed",
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            with_norm=patch_norm,
        )
        self._submodules[self.patch_embed.name] = self.patch_embed

        # Build stages
        for i_layer in range(self.num_layers):
            dim_i = int(embed_dim * 2 ** i_layer)
            has_downsample = (i_layer < self.num_layers - 1)
            layer = SwinBasicLayer(
                name=f"{name}.layer{i_layer}",
                dim=dim_i,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                downsample=has_downsample,
            )
            setattr(self, f"layer{i_layer}", layer)

        # Output norm layers (one per output index)
        for i in out_indices:
            ln = F.LayerNorm(f"{name}.norm{i}", num_features[i])
            setattr(self, f"norm{i}", ln)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (SimTensor): [B, in_chans, H, W]

        Returns:
            list[SimTensor]: Feature maps for each stage in out_indices.
              Each is [B, num_features[i], H//ds, W//ds]
              where ds = patch_size * 2^i.
        """
        # 1. Patch embed: [B, C, H, W] → [B, embed_dim, Wh, Ww]
        x = self.patch_embed(x)
        B, C, Wh, Ww = x.shape

        # 2. Flatten spatial dims: [B, embed_dim, Wh, Ww] → [B, Wh*Ww, embed_dim]
        flat_shape = _from_data(
            self.name + ".init_flat",
            np.array([B, Wh * Ww, C], dtype=np.int64),
            is_const=True,
        )
        x = F.Reshape(self.name + ".init_flatten")(x, flat_shape)
        H_cur, W_cur = Wh, Ww

        outs = []
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            x_out, H_out, W_out, x, H_cur, W_cur = layer(x, H_cur, W_cur)

            if i in self.out_indices:
                norm_op = getattr(self, f"norm{i}")
                x_normed = norm_op(x_out)       # [B, H*W, features_i]
                # Reshape to [B, features_i, H_out, W_out]
                out_shape = _from_data(
                    self.name + f".out_shape{i}",
                    np.array([B, H_out, W_out, self.num_features[i]], dtype=np.int64),
                    is_const=True,
                )
                x_final = F.Reshape(self.name + f".out_reshape{i}")(x_normed, out_shape)
                # Transpose [B, H, W, C] → [B, C, H, W]
                x_final = F.Transpose(self.name + f".out_tr{i}", perm=[0, 3, 1, 2])(x_final)
                outs.append(x_final)

        return outs

    def analytical_param_count(self, lvl: int = 0) -> int:
        total = self.patch_embed.analytical_param_count(lvl + 1)
        for i in range(self.num_layers):
            layer = getattr(self, f"layer{i}")
            total += layer.analytical_param_count(lvl + 1)
        # Output norm layers
        for i in self.out_indices:
            total += 2 * self.num_features[i]
        return total
