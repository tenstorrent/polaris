
# =============================================================================
# ORIGINAL TORCH CODE (from FusionAD)
# Source: FusionAD/projects/mmdet3d_plugin/fusionad/dense_heads/seg_head_plugin/seg_mask_head.py
# =============================================================================
# """
# Copy-paste from torch.nn.Transformer, timm, with modifications:
# """
# import copy
# from typing import Optional, List
#
# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from functools import partial
# from mmdet.models.utils.builder import TRANSFORMER
# import math
# from mmcv.runner import force_fp32
#
# count = 0
#
#
# class Mlp(nn.Module):
#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act_layer=nn.GELU,
#                  drop=0.):
#         super().__init__()
#         self.fp16_enabled = False
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     @force_fp32(apply_to=('x', ))
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class SelfAttention(nn.Module):
#     def __init__(self,
#                  cfg,
#                  dim,
#                  num_heads=2,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.fp16_enabled = False
#         self.scale = qk_scale or head_dim**-0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     @force_fp32(apply_to=('x', ))
#     def forward(self, x):
#         B, N, C = x.shape
#
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                   C // self.num_heads).permute(2, 0, 3, 1,
#                                                                4).contiguous()
#         q, k, v = qkv[0], qkv[1], qkv[
#             2]  # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x
#
#
# class Attention(nn.Module):
#     def __init__(self,
#                  cfg,
#                  dim,
#                  num_heads=2,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.fp16_enabled = False
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.linear_l1 = nn.Sequential(
#             nn.Linear(self.num_heads, self.num_heads),
#             nn.ReLU(),
#         )
#         self.linear = nn.Sequential(
#             nn.Linear(self.num_heads, 1),
#             nn.ReLU(),
#         )
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     @force_fp32(apply_to=('query', 'key', 'value'))
#     def forward(self, query, key, value, key_padding_mask, hw_lvl):
#         B, N, C = query.shape
#         _, L, _ = key.shape
#         #logging.info('query, key, value', query.shape, value.shape, key.shape)
#         q = self.q(query).reshape(B, N,
#                                   self.num_heads, C // self.num_heads).permute(
#                                       0, 2, 1,
#                                       3).contiguous()  #.permute(2, 0, 3, 1, 4)
#         k = self.k(key).reshape(B, L,
#                                 self.num_heads, C // self.num_heads).permute(
#                                     0, 2, 1,
#                                     3).contiguous()  #.permute(2, 0, 3, 1, 4)
#
#         v = self.v(value).reshape(B, L,
#                                   self.num_heads, C // self.num_heads).permute(
#                                       0, 2, 1,
#                                       3).contiguous()  #.permute(2, 0, 3, 1, 4)
#
#         attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
#
#         attn = attn.permute(0, 2, 3, 1)
#
#         new_feats = self.linear_l1(attn)
#         mask = self.linear(new_feats)
#
#         attn = attn.permute(0, 3, 1, 2)
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x, mask
#
# # AttentionTail is a cheap implementation that can make mask decoder 1 layer deeper.
# class AttentionTail(nn.Module): 
#     def __init__(self,
#                  cfg,
#                  dim,
#                  num_heads=2,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop=0.,
#                  proj_drop=0.):
#         super().__init__()
#         self.fp16_enabled = False
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim**-0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#
#         self.linear_l1 = nn.Sequential(
#             nn.Linear(self.num_heads, self.num_heads),
#             nn.ReLU(),
#         )
#
#         self.linear = nn.Sequential(
#             nn.Linear(self.num_heads, 1),
#             nn.ReLU(),
#         )
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     @force_fp32(apply_to=('query', 'key'))
#     def forward(self, query, key, key_padding_mask, hw_lvl=None):
#         B, N, C = query.shape
#         _, L, _ = key.shape
#         #logging.info('query, key, value', query.shape, value.shape, key.shape)
#         q = self.q(query).reshape(B, N,
#                                   self.num_heads, C // self.num_heads).permute(
#                                       0, 2, 1,
#                                       3).contiguous()  #.permute(2, 0, 3, 1, 4)
#         k = self.k(key).reshape(B, L,
#                                 self.num_heads, C // self.num_heads).permute(
#                                     0, 2, 1,
#                                     3).contiguous()  #.permute(2, 0, 3, 1, 4)
#         attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
#
#         attn = attn.permute(0, 2, 3, 1)
#
#         new_feats = self.linear_l1(attn)
#         mask = self.linear(new_feats)
#
#         return mask
#
#
# class Block(nn.Module):
#     def __init__(self,
#                  cfg,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm,
#                  self_attn=False):
#         super().__init__()
#         self.fp16_enabled = False
#         self.head_norm1 = norm_layer(dim)
#         self.self_attn = self_attn
#         self.attn = Attention(cfg,
#                               dim,
#                               num_heads=num_heads,
#                               qkv_bias=qkv_bias,
#                               qk_scale=qk_scale,
#                               attn_drop=attn_drop,
#                               proj_drop=drop)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#
#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()
#         self.head_norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim,
#                        hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer,
#                        drop=drop)
#         if self.self_attn:
#             self.self_attention = SelfAttention(cfg,
#                                                 dim,
#                                                 num_heads=num_heads,
#                                                 qkv_bias=qkv_bias,
#                                                 qk_scale=qk_scale,
#                                                 attn_drop=attn_drop,
#                                                 proj_drop=drop)
#             self.norm3 = norm_layer(dim)
#
#     @force_fp32(apply_to=('query', 'key', 'value'))
#     def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
#         if self.self_attn:
#             query = query + self.drop_path(self.self_attention(query))
#             query = self.norm3(query)
#         x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
#         query = query + self.drop_path(x)
#         query = self.head_norm1(query)
#
#         query = query + self.drop_path(self.mlp(query))
#         query = self.head_norm2(query)
#         return query, mask
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-53296self.num_heads956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0], ) + (1, ) * (
#         x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(
#         shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     @force_fp32(apply_to=('x', ))
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# @TRANSFORMER.register_module()
# class SegMaskHead(nn.Module):
#     def __init__(self,
#                  cfg=None,
#                  d_model=16,
#                  nhead=2,
#                  num_encoder_layers=6,
#                  num_decoder_layers=1,
#                  dim_feedforward=64,
#                  dropout=0.1,
#                  activation="relu",
#                  normalize_before=False,
#                  return_intermediate_dec=False,
#                  self_attn=False):
#         super().__init__()
#
#         self.fp16_enabled = False
#         mlp_ratio = 4
#         qkv_bias = True
#         qk_scale = None
#         drop_rate = 0
#         attn_drop_rate = 0
#
#         norm_layer = None
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = None
#         act_layer = act_layer or nn.GELU
#         block = Block(cfg,
#                       dim=d_model,
#                       num_heads=nhead,
#                       mlp_ratio=mlp_ratio,
#                       qkv_bias=qkv_bias,
#                       qk_scale=qk_scale,
#                       drop=drop_rate,
#                       attn_drop=attn_drop_rate,
#                       drop_path=0,
#                       norm_layer=norm_layer,
#                       act_layer=act_layer,
#                       self_attn=self_attn)
#         self.blocks = _get_clones(block, num_decoder_layers)
#         self.attnen = AttentionTail(cfg,
#                                     d_model,
#                                     num_heads=nhead,
#                                     qkv_bias=qkv_bias,
#                                     qk_scale=qk_scale,
#                                     attn_drop=attn_drop_rate,
#                                     proj_drop=0)
#
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def with_pos_embed(self, tensor, pos: Optional[Tensor]):
#         if pos is None:
#             return tensor
#         else:
#             return tensor + pos
#         #return tensor if pos is None else tensor + pos
#     @force_fp32(apply_to=('memory', 'mask_memory', 'pos_memory', 'query_embed',
#                           'mask_query', 'pos_query'))
#     def forward(self, memory, mask_memory, pos_memory, query_embed, mask_query,
#                 pos_query, hw_lvl):
#         if mask_memory is not None and isinstance(mask_memory, torch.Tensor):
#             mask_memory = mask_memory.to(torch.bool)
#         masks = []
#         inter_query = []
#         for i, block in enumerate(self.blocks):
#             query_embed, mask = block(self.with_pos_embed(
#                 query_embed, pos_query),
#                                       self.with_pos_embed(memory, pos_memory),
#                                       memory,
#                                       key_padding_mask=mask_memory,
#                                       hw_lvl=hw_lvl)
#             masks.append(mask)
#             inter_query.append(query_embed)
#             #if i == 1:
#             #    return mask, masks, inter_query
#         attn = self.attnen(self.with_pos_embed(query_embed, pos_query),
#                            self.with_pos_embed(memory, pos_memory),
#                            key_padding_mask=mask_memory,
#                            hw_lvl=hw_lvl)
#         return attn, masks, inter_query
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of SegMaskHead for FusionAD panoptic segmentation.

Inference-only conversion.  DropPath / Dropout are identity at inference.

Original:
  projects/mmdet3d_plugin/fusionad/dense_heads/seg_head_plugin/seg_mask_head.py

Classes:
  - Mlp              : FC1 → GELU → FC2
  - SelfAttention     : Combined QKV projection, multi-head self-attention
  - Attention         : Separate Q/K/V projection + mask prediction branch
  - AttentionTail     : Q/K projection → mask prediction (no value path)
  - Block             : Optional SelfAttention + Attention + Mlp with residuals
  - SegMaskHead       : N Block layers + AttentionTail

Weight mapping notes (PyTorch → TTSim):
  Mlp:
    fc1                → fc1  (SimNN.Linear)
    fc2                → fc2  (SimNN.Linear)
  SelfAttention:
    qkv (combined)     → q_proj, k_proj, v_proj  (split weights)
    proj               → proj
  Attention:
    q, k, v            → q_proj, k_proj, v_proj
    proj               → proj
    linear_l1.0        → mask_linear1  (SimNN.Linear on last dim=nhead)
    linear.0           → mask_linear2  (SimNN.Linear on last dim=nhead→1)
  AttentionTail:
    q, k               → q_proj, k_proj
    linear_l1.0        → mask_linear1
    linear.0           → mask_linear2
  Block:
    head_norm1         → norm1   (LayerNorm)
    head_norm2         → norm2   (LayerNorm)
    attn               → attn    (Attention)
    mlp                → mlp     (Mlp)
    self_attention     → self_attention  (SelfAttention, optional)
    norm3              → norm3   (LayerNorm, optional)
  SegMaskHead:
    blocks[i]          → block_{i}
    attnen             → attn_tail
"""

import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

dense_heads_dir = os.path.abspath(os.path.join(current_dir, '..'))
if dense_heads_dir not in sys.path:
    sys.path.insert(0, dense_heads_dir)

fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ====================================================================
# Mlp  (FC1 → GELU → FC2, no dropout at inference)
# ====================================================================
class Mlp(SimNN.Module):
    """Two-layer MLP with GELU activation."""

    def __init__(self, name, in_features, hidden_features=None,
                 out_features=None):
        super().__init__()
        self.name = name
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = SimNN.Linear(f'{name}.fc1', in_features, hidden_features)
        self.gelu = F.Gelu(f'{name}.gelu')
        self.fc2 = SimNN.Linear(f'{name}.fc2', hidden_features, out_features)

        super().link_op2module()

    def __call__(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


# ====================================================================
# SelfAttention
# ====================================================================
class SelfAttention(SimNN.Module):
    """Multi-head self-attention with combined QKV projection.

    For weight copy, split PyTorch qkv.weight [3D, D] into q/k/v.
    """

    def __init__(self, name, dim, num_heads=2, qkv_bias=True):
        super().__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_val = self.head_dim ** -0.5

        # Separate Q/K/V projections (PyTorch uses combined qkv linear)
        self.q_proj = SimNN.Linear(f'{name}.q_proj', dim, dim, bias=qkv_bias)
        self.k_proj = SimNN.Linear(f'{name}.k_proj', dim, dim, bias=qkv_bias)
        self.v_proj = SimNN.Linear(f'{name}.v_proj', dim, dim, bias=qkv_bias)
        self.proj = SimNN.Linear(f'{name}.proj', dim, dim)

        # Reshape [B,N,D] → [B,N,nhead,hd] → Transpose [B,nhead,N,hd]
        self.q_reshape = F.Reshape(f'{name}.q_rs')
        self.q_perm = F.Transpose(f'{name}.q_pm', perm=[0, 2, 1, 3])
        self.k_reshape = F.Reshape(f'{name}.k_rs')
        self.k_perm = F.Transpose(f'{name}.k_pm', perm=[0, 2, 1, 3])
        self.v_reshape = F.Reshape(f'{name}.v_rs')
        self.v_perm = F.Transpose(f'{name}.v_pm', perm=[0, 2, 1, 3])

        # k^T and QK^T
        self.k_t = F.Transpose(f'{name}.k_t', perm=[0, 1, 3, 2])
        self.qk_mm = F.MatMul(f'{name}.qk_mm')
        self.scale_mul = F.Mul(f'{name}.scale')
        self.softmax = F.Softmax(f'{name}.softmax', axis=-1)

        # attn @ V
        self.av_mm = F.MatMul(f'{name}.av_mm')
        self.out_perm = F.Transpose(f'{name}.out_pm', perm=[0, 2, 1, 3])
        self.out_reshape = F.Reshape(f'{name}.out_rs')

        # Scale constant
        self.scale_const = F._from_data(
            f'{name}.scale_c',
            np.float32(self.scale_val), is_const=True)

        super().link_op2module()

    def __call__(self, x):
        B, N, C = x.shape
        hd = self.head_dim
        nh = self.num_heads

        shape4 = F._from_data(f'{self.name}._s4q',
                               np.array([B, N, nh, hd], dtype=np.int64),
                               is_const=True)
        setattr(self, shape4.name, shape4)
        shape3 = F._from_data(f'{self.name}._s3',
                               np.array([B, N, C], dtype=np.int64),
                               is_const=True)
        setattr(self, shape3.name, shape3)

        q = self.q_perm(self.q_reshape(self.q_proj(x), shape4))
        k = self.k_perm(self.k_reshape(self.k_proj(x), shape4))
        v = self.v_perm(self.v_reshape(self.v_proj(x), shape4))

        scores = self.scale_mul(self.qk_mm(q, self.k_t(k)), self.scale_const)
        attn = self.softmax(scores)
        out = self.out_reshape(self.out_perm(self.av_mm(attn, v)), shape3)
        out = self.proj(out)
        return out


# ====================================================================
# Attention  (cross-attention with mask prediction branch)
# ====================================================================
class Attention(SimNN.Module):
    """Cross-attention with separate Q/K/V and a mask prediction branch.

    The mask branch:
      attn scores → permute [B,N,L,nhead] → linear_l1(ReLU) → linear(ReLU)
      → mask [B,N,L,1]

    Returns: (x, mask)
    """

    def __init__(self, name, dim, num_heads=2, qkv_bias=True):
        super().__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_val = self.head_dim ** -0.5

        # Q / K / V projections
        self.q_proj = SimNN.Linear(f'{name}.q', dim, dim, bias=qkv_bias)
        self.k_proj = SimNN.Linear(f'{name}.k', dim, dim, bias=qkv_bias)
        self.v_proj = SimNN.Linear(f'{name}.v', dim, dim, bias=qkv_bias)
        self.proj = SimNN.Linear(f'{name}.proj', dim, dim)

        # Mask branch:  linear_l1 = Linear(nhead→nhead)+ReLU,
        #               linear   = Linear(nhead→1)+ReLU
        self.mask_linear1 = SimNN.Linear(
            f'{name}.mask_l1', num_heads, num_heads)
        self.mask_relu1 = F.Relu(f'{name}.mask_r1')
        self.mask_linear2 = SimNN.Linear(
            f'{name}.mask_l2', num_heads, 1)
        self.mask_relu2 = F.Relu(f'{name}.mask_r2')

        # Reshape / permute ops
        self.q_reshape = F.Reshape(f'{name}.q_rs')
        self.q_perm = F.Transpose(f'{name}.q_pm', perm=[0, 2, 1, 3])
        self.k_reshape = F.Reshape(f'{name}.k_rs')
        self.k_perm = F.Transpose(f'{name}.k_pm', perm=[0, 2, 1, 3])
        self.v_reshape = F.Reshape(f'{name}.v_rs')
        self.v_perm = F.Transpose(f'{name}.v_pm', perm=[0, 2, 1, 3])

        self.k_t = F.Transpose(f'{name}.k_t', perm=[0, 1, 3, 2])
        self.qk_mm = F.MatMul(f'{name}.qk_mm')
        self.scale_mul = F.Mul(f'{name}.scale')

        # Mask branch permute: [B,nhead,N,L] → [B,N,L,nhead]
        self.attn_to_mask_perm = F.Transpose(
            f'{name}.attn2mask', perm=[0, 2, 3, 1])
        # Back: [B,N,L,nhead] → [B,nhead,N,L]
        self.mask_to_attn_perm = F.Transpose(
            f'{name}.mask2attn', perm=[0, 3, 1, 2])

        self.softmax = F.Softmax(f'{name}.softmax', axis=-1)
        self.av_mm = F.MatMul(f'{name}.av_mm')
        self.out_perm = F.Transpose(f'{name}.out_pm', perm=[0, 2, 1, 3])
        self.out_reshape = F.Reshape(f'{name}.out_rs')

        self.scale_const = F._from_data(
            f'{name}.scale_c',
            np.float32(self.scale_val), is_const=True)

        super().link_op2module()

    def __call__(self, query, key, value):
        B, N, C = query.shape
        _, L, _ = key.shape
        hd = self.head_dim
        nh = self.num_heads

        shape_q = F._from_data(f'{self.name}._sq',
                                np.array([B, N, nh, hd], dtype=np.int64),
                                is_const=True)
        setattr(self, shape_q.name, shape_q)
        shape_k = F._from_data(f'{self.name}._sk',
                                np.array([B, L, nh, hd], dtype=np.int64),
                                is_const=True)
        setattr(self, shape_k.name, shape_k)
        shape_out = F._from_data(f'{self.name}._so',
                                  np.array([B, N, C], dtype=np.int64),
                                  is_const=True)
        setattr(self, shape_out.name, shape_out)

        q = self.q_perm(self.q_reshape(self.q_proj(query), shape_q))
        k = self.k_perm(self.k_reshape(self.k_proj(key), shape_k))
        v = self.v_perm(self.v_reshape(self.v_proj(value), shape_k))

        # [B, nhead, N, L]
        scores = self.scale_mul(
            self.qk_mm(q, self.k_t(k)), self.scale_const)

        # ── Mask branch ──
        # [B, nhead, N, L] → [B, N, L, nhead]
        attn_for_mask = self.attn_to_mask_perm(scores)
        mask = self.mask_relu1(self.mask_linear1(attn_for_mask))
        mask = self.mask_relu2(self.mask_linear2(mask))  # [B,N,L,1]

        # ── Attention value path ──
        # [B, N, L, nhead] → [B, nhead, N, L]
        attn = self.mask_to_attn_perm(attn_for_mask)
        attn = self.softmax(attn)
        out = self.out_reshape(
            self.out_perm(self.av_mm(attn, v)), shape_out)
        out = self.proj(out)

        return out, mask


# ====================================================================
# AttentionTail  (Q/K only → mask prediction, no value path)
# ====================================================================
class AttentionTail(SimNN.Module):
    """Lightweight attention that only produces a mask (no value projection)."""

    def __init__(self, name, dim, num_heads=2, qkv_bias=True):
        super().__init__()
        self.name = name
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale_val = self.head_dim ** -0.5

        self.q_proj = SimNN.Linear(f'{name}.q', dim, dim, bias=qkv_bias)
        self.k_proj = SimNN.Linear(f'{name}.k', dim, dim, bias=qkv_bias)

        self.mask_linear1 = SimNN.Linear(
            f'{name}.mask_l1', num_heads, num_heads)
        self.mask_relu1 = F.Relu(f'{name}.mask_r1')
        self.mask_linear2 = SimNN.Linear(
            f'{name}.mask_l2', num_heads, 1)
        self.mask_relu2 = F.Relu(f'{name}.mask_r2')

        self.q_reshape = F.Reshape(f'{name}.q_rs')
        self.q_perm = F.Transpose(f'{name}.q_pm', perm=[0, 2, 1, 3])
        self.k_reshape = F.Reshape(f'{name}.k_rs')
        self.k_perm = F.Transpose(f'{name}.k_pm', perm=[0, 2, 1, 3])

        self.k_t = F.Transpose(f'{name}.k_t', perm=[0, 1, 3, 2])
        self.qk_mm = F.MatMul(f'{name}.qk_mm')
        self.scale_mul = F.Mul(f'{name}.scale')

        self.attn_to_mask_perm = F.Transpose(
            f'{name}.attn2mask', perm=[0, 2, 3, 1])

        self.scale_const = F._from_data(
            f'{name}.scale_c',
            np.float32(self.scale_val), is_const=True)

        super().link_op2module()

    def __call__(self, query, key):
        B, N, C = query.shape
        _, L, _ = key.shape
        hd = self.head_dim
        nh = self.num_heads

        shape_q = F._from_data(f'{self.name}._sq',
                                np.array([B, N, nh, hd], dtype=np.int64),
                                is_const=True)
        setattr(self, shape_q.name, shape_q)
        shape_k = F._from_data(f'{self.name}._sk',
                                np.array([B, L, nh, hd], dtype=np.int64),
                                is_const=True)
        setattr(self, shape_k.name, shape_k)

        q = self.q_perm(self.q_reshape(self.q_proj(query), shape_q))
        k = self.k_perm(self.k_reshape(self.k_proj(key), shape_k))

        scores = self.scale_mul(
            self.qk_mm(q, self.k_t(k)), self.scale_const)

        # [B, nhead, N, L] → [B, N, L, nhead]
        attn = self.attn_to_mask_perm(scores)
        mask = self.mask_relu1(self.mask_linear1(attn))
        mask = self.mask_relu2(self.mask_linear2(mask))  # [B,N,L,1]
        return mask


# ====================================================================
# Block  (optional SelfAttention + Attention + Mlp with residuals)
# ====================================================================
class Block(SimNN.Module):
    """Single decoder block: cross-attention + MLP with residual & LayerNorm.

    DropPath with drop_prob=0 is identity at inference.
    """

    def __init__(self, name, dim, num_heads, mlp_ratio=4,
                 qkv_bias=True, use_self_attn=False):
        super().__init__()
        self.name = name
        self.use_self_attn = use_self_attn

        mlp_hidden = int(dim * mlp_ratio)

        # Cross-attention
        self.attn = Attention(f'{name}.attn', dim, num_heads, qkv_bias)
        self.norm1 = F.LayerNorm(f'{name}.norm1', dim, epsilon=1e-6)
        self.mlp = Mlp(f'{name}.mlp', dim, mlp_hidden)
        self.norm2 = F.LayerNorm(f'{name}.norm2', dim, epsilon=1e-6)

        # Residual adds
        self.res_attn = F.Add(f'{name}.res_attn')
        self.res_mlp = F.Add(f'{name}.res_mlp')

        # Optional self-attention
        if use_self_attn:
            self.self_attention = SelfAttention(
                f'{name}.sa', dim, num_heads, qkv_bias)
            self.norm3 = F.LayerNorm(f'{name}.norm3', dim, epsilon=1e-6)
            self.res_sa = F.Add(f'{name}.res_sa')

        super().link_op2module()

    def __call__(self, query, key, value):
        """Returns (query, mask)."""
        if self.use_self_attn:
            sa_out = self.self_attention(query)
            query = self.norm3(self.res_sa(query, sa_out))

        x, mask = self.attn(query, key, value)
        query = self.norm1(self.res_attn(query, x))
        mlp_out = self.mlp(query)
        query = self.norm2(self.res_mlp(query, mlp_out))
        return query, mask


# ====================================================================
# SegMaskHead
# ====================================================================
class SegMaskHead(SimNN.Module):
    """Mask decoder head for panoptic segmentation.

    Consists of N Block decoder layers followed by an AttentionTail.

    forward(memory, mask_memory, pos_memory, query_embed, mask_query, pos_query, hw_lvl):
        Returns (attn_mask, masks_list, inter_query_list)
    """

    def __init__(self, name, d_model=256, nhead=8,
                 num_decoder_layers=6, use_self_attn=False):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers

        # Decoder blocks
        self.blocks = []
        for i in range(num_decoder_layers):
            blk = Block(f'{name}.block_{i}', d_model, nhead,
                        mlp_ratio=4, qkv_bias=True,
                        use_self_attn=use_self_attn)
            setattr(self, f'block_{i}', blk)
            self.blocks.append(blk)

        # Final attention tail
        self.attn_tail = AttentionTail(
            f'{name}.attn_tail', d_model, nhead, qkv_bias=True)

        # Pos-embedding add for memory (used once, before the loop)
        self.mem_pos_add = F.Add(f'{name}.m_pos_add')
        # query_pos_add is created per-iteration in __call__ to avoid name collision

        super().link_op2module()

    def __call__(self, memory, mask_memory=None, pos_memory=None,
                 query_embed=None, mask_query=None, pos_query=None,
                 hw_lvl=None):
        """
        Args:
            memory:       [B, L, D]  encoder memory features.
            mask_memory:  [B, L]    padding mask for memory (ignored).
            pos_memory:   [B, L, D] positional encoding for memory (or None).
            query_embed:  [B, N, D] query embeddings.
            mask_query:   [B, N]    padding mask for queries (ignored).
            pos_query:    [B, N, D] positional encoding for queries (or None).
            hw_lvl:       list of (H, W) tuples per feature level (ignored).

        Returns:
            attn_mask:   SimTensor [B, N, L, 1] from AttentionTail.
            masks:       list of SimTensor [B, N, L, 1] from each Block.
            inter_query: list of SimTensor [B, N, D] query states.
        """
        masks = []
        inter_query = []

        # Precompute key with pos_memory (constant across blocks)
        if pos_memory is not None:
            key_with_pos = self.mem_pos_add(memory, pos_memory)
        else:
            key_with_pos = memory

        for i in range(self.num_decoder_layers):
            # query + pos for attention (pos only affects q/k, not residual)
            if pos_query is not None:
                _qpa = F.Add(f'{self.name}.q_pos_add_{i}')
                setattr(self, _qpa.name, _qpa)
                _qpa.set_module(self)
                q_in = _qpa(query_embed, pos_query)
            else:
                q_in = query_embed

            query_embed, mask = self.blocks[i](
                q_in, key_with_pos, memory)
            masks.append(mask)
            inter_query.append(query_embed)

        # Final AttentionTail
        if pos_query is not None:
            _qpa_final = F.Add(f'{self.name}.q_pos_add_final')
            setattr(self, _qpa_final.name, _qpa_final)
            _qpa_final.set_module(self)
            q_final = _qpa_final(query_embed, pos_query)
        else:
            q_final = query_embed
        attn_mask = self.attn_tail(q_final, key_with_pos)

        return attn_mask, masks, inter_query

    def analytical_param_count(self):
        total = 0
        for blk in self.blocks:
            total += blk.attn.q_proj.analytical_param_count(0)
            total += blk.attn.k_proj.analytical_param_count(0)
            total += blk.attn.v_proj.analytical_param_count(0)
            total += blk.attn.proj.analytical_param_count(0)
            total += blk.attn.mask_linear1.analytical_param_count(0)
            total += blk.attn.mask_linear2.analytical_param_count(0)
            total += blk.mlp.fc1.analytical_param_count(0)
            total += blk.mlp.fc2.analytical_param_count(0)
            if blk.use_self_attn:
                total += blk.self_attention.q_proj.analytical_param_count(0)
                total += blk.self_attention.k_proj.analytical_param_count(0)
                total += blk.self_attention.v_proj.analytical_param_count(0)
                total += blk.self_attention.proj.analytical_param_count(0)
        total += self.attn_tail.q_proj.analytical_param_count(0)
        total += self.attn_tail.k_proj.analytical_param_count(0)
        total += self.attn_tail.mask_linear1.analytical_param_count(0)
        total += self.attn_tail.mask_linear2.analytical_param_count(0)
        return total


# ====================================================================
# Self-test
# ====================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SegMaskHead TTSim module — self-test")
    logger.info("=" * 70)

    passed = 0
    total = 0

    # ── TEST 1: Mlp construction + forward ──
    total += 1
    try:
        mlp = Mlp('test_mlp', 64, 256)
        x = F._from_data('t1.x', np.random.randn(1, 10, 64).astype(np.float32))
        out = mlp(x)
        assert list(out.shape) == [1, 10, 64], f"Expected [1,10,64], got {out.shape}"
        logger.debug(f"[OK] TEST 1: Mlp forward shape {list(out.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 1: {e}")

    # ── TEST 2: SelfAttention construction + forward ──
    total += 1
    try:
        sa = SelfAttention('test_sa', 64, num_heads=8)
        x = F._from_data('t2.x', np.random.randn(1, 10, 64).astype(np.float32))
        out = sa(x)
        assert list(out.shape) == [1, 10, 64], f"Expected [1,10,64], got {out.shape}"
        logger.debug(f"[OK] TEST 2: SelfAttention forward shape {list(out.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 2: {e}")

    # ── TEST 3: Attention construction + forward ──
    total += 1
    try:
        attn = Attention('test_attn', 64, num_heads=8)
        q = F._from_data('t3.q', np.random.randn(1, 5, 64).astype(np.float32))
        k = F._from_data('t3.k', np.random.randn(1, 20, 64).astype(np.float32))
        v = F._from_data('t3.v', np.random.randn(1, 20, 64).astype(np.float32))
        out, mask = attn(q, k, v)
        assert list(out.shape) == [1, 5, 64], f"Expected [1,5,64], got {out.shape}"
        assert list(mask.shape) == [1, 5, 20, 1], \
            f"Expected [1,5,20,1], got {mask.shape}"
        logger.debug(f"[OK] TEST 3: Attention out={list(out.shape)} mask={list(mask.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 3: {e}")

    # ── TEST 4: AttentionTail construction + forward ──
    total += 1
    try:
        at = AttentionTail('test_at', 64, num_heads=8)
        q = F._from_data('t4.q', np.random.randn(1, 5, 64).astype(np.float32))
        k = F._from_data('t4.k', np.random.randn(1, 20, 64).astype(np.float32))
        mask = at(q, k)
        assert list(mask.shape) == [1, 5, 20, 1], \
            f"Expected [1,5,20,1], got {mask.shape}"
        logger.debug(f"[OK] TEST 4: AttentionTail mask shape {list(mask.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 4: {e}")

    # ── TEST 5: Block construction + forward ──
    total += 1
    try:
        blk = Block('test_blk', 64, num_heads=8)
        q = F._from_data('t5.q', np.random.randn(1, 5, 64).astype(np.float32))
        k = F._from_data('t5.k', np.random.randn(1, 20, 64).astype(np.float32))
        v = F._from_data('t5.v', np.random.randn(1, 20, 64).astype(np.float32))
        out, mask = blk(q, k, v)
        assert list(out.shape) == [1, 5, 64], f"Expected [1,5,64], got {out.shape}"
        assert list(mask.shape) == [1, 5, 20, 1], \
            f"Expected [1,5,20,1], got {mask.shape}"
        logger.debug(f"[OK] TEST 5: Block out={list(out.shape)} mask={list(mask.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 5: {e}")

    # ── TEST 6: Block with self_attn ──
    total += 1
    try:
        blk2 = Block('test_blk2', 64, num_heads=8, use_self_attn=True)
        q = F._from_data('t6.q', np.random.randn(1, 5, 64).astype(np.float32))
        k = F._from_data('t6.k', np.random.randn(1, 20, 64).astype(np.float32))
        v = F._from_data('t6.v', np.random.randn(1, 20, 64).astype(np.float32))
        out, mask = blk2(q, k, v)
        assert list(out.shape) == [1, 5, 64]
        logger.debug(f"[OK] TEST 6: Block(self_attn) out={list(out.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 6: {e}")

    # ── TEST 7: SegMaskHead full forward (no pos) ──
    total += 1
    try:
        head = SegMaskHead('test_smh', d_model=64, nhead=8,
                           num_decoder_layers=2)
        mem = F._from_data('t7.m', np.random.randn(1, 100, 64).astype(np.float32))
        qry = F._from_data('t7.q', np.random.randn(1, 10, 64).astype(np.float32))
        attn_mask, masks, iq = head(mem, None, None, qry)
        assert list(attn_mask.shape) == [1, 10, 100, 1]
        assert len(masks) == 2
        assert len(iq) == 2
        assert list(iq[-1].shape) == [1, 10, 64]
        logger.debug(
            f"[OK] TEST 7: SegMaskHead attn={list(attn_mask.shape)} "
            f"blocks={len(masks)} iq={list(iq[-1].shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 7: {e}")

    # ── TEST 8: SegMaskHead with pos_query ──
    total += 1
    try:
        head2 = SegMaskHead('test_smh2', d_model=64, nhead=8,
                            num_decoder_layers=1)
        mem = F._from_data('t8.m', np.random.randn(1, 50, 64).astype(np.float32))
        qry = F._from_data('t8.q', np.random.randn(1, 5, 64).astype(np.float32))
        pos = F._from_data('t8.p', np.random.randn(1, 5, 64).astype(np.float32))
        attn_mask, masks, iq = head2(mem, None, None, qry, None, pos)
        assert list(attn_mask.shape) == [1, 5, 50, 1]
        logger.debug(f"[OK] TEST 8: SegMaskHead(pos_query) attn={list(attn_mask.shape)}")
        passed += 1
    except Exception as e:
        logger.debug(f"[FAIL] TEST 8: {e}")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"RESULTS: {passed}/{total} tests passed.")
    if passed == total:
        logger.info("ALL TESTS PASSED.")
    else:
        logger.info("SOME TESTS FAILED.")
