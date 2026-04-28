#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comparison test: SegMaskHead (TTSim vs PyTorch).

Tests all sub-modules individually (Mlp, SelfAttention, Attention,
AttentionTail, Block) and the full SegMaskHead, comparing outputs
with shared weights.

Config from fusion_base_e2e.py:
  thing_transformer_head: d_model=256, nhead=8, num_decoder_layers=4
  stuff_transformer_head: d_model=256, nhead=8, num_decoder_layers=6, self_attn=True

For speed we test at reduced dim (d_model=64, nhead=8, layers=2).
"""

import copy
import os
import sys
import traceback
from typing import Optional
from functools import partial

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..','..')
sys.path.insert(0, polaris_path)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch import Tensor

np.random.seed(42)
torch.manual_seed(42)

import ttsim.front.functional.op as F

# TTSim source
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.seg_head_plugin.seg_mask_head import (
    Mlp as TT_Mlp,
    SelfAttention as TT_SelfAttention,
    Attention as TT_Attention,
    AttentionTail as TT_AttentionTail,
    Block as TT_Block,
    SegMaskHead as TT_SegMaskHead,
)


# ====================================================================
# PyTorch reference classes (inline, no mmcv dependency)
# ====================================================================

class PT_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PT_SelfAttention(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PT_Attention(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads), nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1), nn.ReLU())
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = self.v(value).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.permute(0, 2, 3, 1)
        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)
        attn = attn.permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask


class PT_AttentionTail(nn.Module):
    def __init__(self, cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads), nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1), nn.ReLU())
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, key_padding_mask=None, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.permute(0, 2, 3, 1)
        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)
        return mask


class PT_Block(nn.Module):
    def __init__(self, cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=None, self_attn=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.head_norm1 = norm_layer(dim)
        self.self_attn = self_attn
        self.attn = PT_Attention(cfg, dim, num_heads=num_heads,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # drop_path=0 at inference
        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = PT_Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                          act_layer=act_layer, drop=drop)
        if self.self_attn:
            self.self_attention = PT_SelfAttention(
                cfg, dim, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.norm3 = norm_layer(dim)

    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn:
            query = query + self.drop_path(self.self_attention(query))
            query = self.norm3(query)
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        query = query + self.drop_path(x)
        query = self.head_norm1(query)
        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)
        return query, mask


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class PT_SegMaskHead(nn.Module):
    def __init__(self, cfg=None, d_model=16, nhead=2, num_decoder_layers=1,
                 self_attn=False):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU
        block = PT_Block(cfg, dim=d_model, num_heads=nhead, mlp_ratio=4,
                         qkv_bias=True, qk_scale=None, drop=0, attn_drop=0,
                         drop_path=0, norm_layer=norm_layer,
                         act_layer=act_layer, self_attn=self_attn)
        self.blocks = _get_clones(block, num_decoder_layers)
        self.attnen = PT_AttentionTail(cfg, d_model, num_heads=nhead,
                                       qkv_bias=True, qk_scale=None,
                                       attn_drop=0, proj_drop=0)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, mask_memory, pos_memory, query_embed,
                mask_query, pos_query, hw_lvl):
        if mask_memory is not None and isinstance(mask_memory, torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)
        masks = []
        inter_query = []
        for i, block in enumerate(self.blocks):
            query_embed, mask = block(
                self.with_pos_embed(query_embed, pos_query),
                self.with_pos_embed(memory, pos_memory),
                memory, key_padding_mask=mask_memory, hw_lvl=hw_lvl)
            masks.append(mask)
            inter_query.append(query_embed)
        attn = self.attnen(
            self.with_pos_embed(query_embed, pos_query),
            self.with_pos_embed(memory, pos_memory),
            key_padding_mask=mask_memory, hw_lvl=hw_lvl)
        return attn, masks, inter_query


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear(pt_linear, tt_linear):
    """Copy nn.Linear weights to SimNN.Linear."""
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_layernorm(pt_ln, tt_ln):
    """Copy nn.LayerNorm weights to F.LayerNorm."""
    # TTSim LayerNorm stores scale and bias as params[0] and params[1]
    scale_tensor = tt_ln.params[0][1]
    bias_tensor = tt_ln.params[1][1]
    scale_tensor.data = pt_ln.weight.data.detach().numpy().astype(np.float32)
    bias_tensor.data = pt_ln.bias.data.detach().numpy().astype(np.float32)


def copy_mlp(pt_mlp, tt_mlp):
    """Copy Mlp weights."""
    copy_linear(pt_mlp.fc1, tt_mlp.fc1)
    copy_linear(pt_mlp.fc2, tt_mlp.fc2)


def copy_self_attention(pt_sa, tt_sa):
    """Copy SelfAttention weights. PyTorch has combined qkv; TTSim has split."""
    dim = pt_sa.qkv.in_features
    # Split combined qkv weight [3*dim, dim] into q, k, v
    w = pt_sa.qkv.weight.data.detach().numpy()  # [3D, D]
    b = pt_sa.qkv.bias.data.detach().numpy()     # [3D]
    wq, wk, wv = w[:dim], w[dim:2*dim], w[2*dim:]
    bq, bk, bv = b[:dim], b[dim:2*dim], b[2*dim:]

    tt_sa.q_proj.param.data = wq.astype(np.float32)
    tt_sa.q_proj.bias.data = bq.astype(np.float32)
    tt_sa.k_proj.param.data = wk.astype(np.float32)
    tt_sa.k_proj.bias.data = bk.astype(np.float32)
    tt_sa.v_proj.param.data = wv.astype(np.float32)
    tt_sa.v_proj.bias.data = bv.astype(np.float32)

    copy_linear(pt_sa.proj, tt_sa.proj)


def copy_attention(pt_attn, tt_attn):
    """Copy Attention weights (separate q/k/v + mask branch)."""
    copy_linear(pt_attn.q, tt_attn.q_proj)
    copy_linear(pt_attn.k, tt_attn.k_proj)
    copy_linear(pt_attn.v, tt_attn.v_proj)
    copy_linear(pt_attn.proj, tt_attn.proj)
    # Mask branch: linear_l1[0] and linear[0] are the nn.Linear layers
    copy_linear(pt_attn.linear_l1[0], tt_attn.mask_linear1)
    copy_linear(pt_attn.linear[0], tt_attn.mask_linear2)


def copy_attention_tail(pt_at, tt_at):
    """Copy AttentionTail weights."""
    copy_linear(pt_at.q, tt_at.q_proj)
    copy_linear(pt_at.k, tt_at.k_proj)
    copy_linear(pt_at.linear_l1[0], tt_at.mask_linear1)
    copy_linear(pt_at.linear[0], tt_at.mask_linear2)


def copy_block(pt_block, tt_block):
    """Copy Block weights (attention + mlp + norms + optional self_attn)."""
    copy_attention(pt_block.attn, tt_block.attn)
    copy_mlp(pt_block.mlp, tt_block.mlp)
    copy_layernorm(pt_block.head_norm1, tt_block.norm1)
    copy_layernorm(pt_block.head_norm2, tt_block.norm2)
    if pt_block.self_attn:
        copy_self_attention(pt_block.self_attention, tt_block.self_attention)
        copy_layernorm(pt_block.norm3, tt_block.norm3)


def copy_seg_mask_head(pt_head, tt_head):
    """Copy full SegMaskHead weights."""
    for i, (pt_blk, tt_blk) in enumerate(
            zip(pt_head.blocks, tt_head.blocks)):
        copy_block(pt_blk, tt_blk)
    copy_attention_tail(pt_head.attnen, tt_head.attn_tail)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-5):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
    print(f"  {name}:")
    print(f"    PT shape: {pt_np.shape}  TT shape: {tt_np.shape}")
    if pt_np.shape != tt_np.shape:
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np - tt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if max_diff < atol:
        print(f"    [OK] Match (atol={atol})")
        return True
    print(f"    [FAIL] Exceeds tolerance (atol={atol})")
    return False


# ====================================================================
# Test dimensions
# ====================================================================
DIM = 64
NHEAD = 8
B = 1
N_QUERY = 5
N_MEM = 20


# ====================================================================
# TEST 1: Mlp
# ====================================================================
def test_mlp():
    print("\n" + "=" * 60)
    print("TEST 1: Mlp")
    print("=" * 60)

    pt = PT_Mlp(DIM, DIM * 4)
    pt.eval()

    tt = TT_Mlp('t1_mlp', DIM, DIM * 4)
    copy_mlp(pt, tt)

    x_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    pt_out = pt(torch.from_numpy(x_np))
    tt_out = tt(F._from_data('t1.x', x_np))

    return compare(pt_out, tt_out, "Mlp output")


# ====================================================================
# TEST 2: SelfAttention
# ====================================================================
def test_self_attention():
    print("\n" + "=" * 60)
    print("TEST 2: SelfAttention")
    print("=" * 60)

    pt = PT_SelfAttention(None, DIM, num_heads=NHEAD, qkv_bias=True)
    pt.eval()

    tt = TT_SelfAttention('t2_sa', DIM, num_heads=NHEAD, qkv_bias=True)
    copy_self_attention(pt, tt)

    x_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    pt_out = pt(torch.from_numpy(x_np))
    tt_out = tt(F._from_data('t2.x', x_np))

    return compare(pt_out, tt_out, "SelfAttention output")


# ====================================================================
# TEST 3: Attention
# ====================================================================
def test_attention():
    print("\n" + "=" * 60)
    print("TEST 3: Attention")
    print("=" * 60)

    pt = PT_Attention(None, DIM, num_heads=NHEAD, qkv_bias=True)
    pt.eval()

    tt = TT_Attention('t3_attn', DIM, num_heads=NHEAD, qkv_bias=True)
    copy_attention(pt, tt)

    q_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    k_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)
    v_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)

    pt_out, pt_mask = pt(torch.from_numpy(q_np), torch.from_numpy(k_np),
                         torch.from_numpy(v_np), None, None)
    tt_out, tt_mask = tt(F._from_data('t3.q', q_np),
                         F._from_data('t3.k', k_np),
                         F._from_data('t3.v', v_np))

    ok1 = compare(pt_out, tt_out, "Attention output")
    ok2 = compare(pt_mask, tt_mask, "Attention mask")
    return ok1 and ok2


# ====================================================================
# TEST 4: AttentionTail
# ====================================================================
def test_attention_tail():
    print("\n" + "=" * 60)
    print("TEST 4: AttentionTail")
    print("=" * 60)

    pt = PT_AttentionTail(None, DIM, num_heads=NHEAD, qkv_bias=True)
    pt.eval()

    tt = TT_AttentionTail('t4_at', DIM, num_heads=NHEAD, qkv_bias=True)
    copy_attention_tail(pt, tt)

    q_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    k_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)

    pt_mask = pt(torch.from_numpy(q_np), torch.from_numpy(k_np), None)
    tt_mask = tt(F._from_data('t4.q', q_np), F._from_data('t4.k', k_np))

    return compare(pt_mask, tt_mask, "AttentionTail mask")


# ====================================================================
# TEST 5: Block (no self_attn)
# ====================================================================
def test_block():
    print("\n" + "=" * 60)
    print("TEST 5: Block (no self_attn)")
    print("=" * 60)

    pt = PT_Block(None, DIM, num_heads=NHEAD, qkv_bias=True, self_attn=False)
    pt.eval()

    tt = TT_Block('t5_blk', DIM, num_heads=NHEAD, qkv_bias=True,
                   use_self_attn=False)
    copy_block(pt, tt)

    q_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    k_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)
    v_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)

    pt_out, pt_mask = pt(torch.from_numpy(q_np), torch.from_numpy(k_np),
                         torch.from_numpy(v_np))
    tt_out, tt_mask = tt(F._from_data('t5.q', q_np),
                         F._from_data('t5.k', k_np),
                         F._from_data('t5.v', v_np))

    ok1 = compare(pt_out, tt_out, "Block query output", atol=1e-4)
    ok2 = compare(pt_mask, tt_mask, "Block mask output")
    return ok1 and ok2


# ====================================================================
# TEST 6: Block (with self_attn)
# ====================================================================
def test_block_self_attn():
    print("\n" + "=" * 60)
    print("TEST 6: Block (with self_attn)")
    print("=" * 60)

    pt = PT_Block(None, DIM, num_heads=NHEAD, qkv_bias=True, self_attn=True)
    pt.eval()

    tt = TT_Block('t6_blk', DIM, num_heads=NHEAD, qkv_bias=True,
                   use_self_attn=True)
    copy_block(pt, tt)

    q_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    k_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)
    v_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)

    pt_out, pt_mask = pt(torch.from_numpy(q_np), torch.from_numpy(k_np),
                         torch.from_numpy(v_np))
    tt_out, tt_mask = tt(F._from_data('t6.q', q_np),
                         F._from_data('t6.k', k_np),
                         F._from_data('t6.v', v_np))

    ok1 = compare(pt_out, tt_out, "Block(sa) query output", atol=1e-4)
    ok2 = compare(pt_mask, tt_mask, "Block(sa) mask output")
    return ok1 and ok2


# ====================================================================
# TEST 7: SegMaskHead (thing config, no pos, no self_attn)
# ====================================================================
def test_seg_mask_head_thing():
    print("\n" + "=" * 60)
    print("TEST 7: SegMaskHead (thing-like, no self_attn, no pos)")
    print("=" * 60)

    pt = PT_SegMaskHead(d_model=DIM, nhead=NHEAD, num_decoder_layers=2,
                        self_attn=False)
    pt.eval()

    tt = TT_SegMaskHead('t7_smh', d_model=DIM, nhead=NHEAD,
                         num_decoder_layers=2, use_self_attn=False)
    copy_seg_mask_head(pt, tt)

    mem_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)
    qry_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)

    # PyTorch call: forward(memory, mask_memory, pos_memory, query_embed,
    #                       mask_query, pos_query, hw_lvl)
    pt_attn, pt_masks, pt_iq = pt(
        torch.from_numpy(mem_np), None, None,
        torch.from_numpy(qry_np), None, None,
        hw_lvl=None)

    # TTSim call: __call__(memory, mask_memory, pos_memory, query_embed, ...)
    tt_attn, tt_masks, tt_iq = tt(
        F._from_data('t7.m', mem_np),
        None, None,
        F._from_data('t7.q', qry_np))

    ok = True
    ok &= compare(pt_attn, tt_attn, "SegMaskHead attn_mask", atol=1e-4)
    for i in range(len(pt_masks)):
        ok &= compare(pt_masks[i], tt_masks[i],
                       f"SegMaskHead mask[{i}]", atol=1e-4)
    for i in range(len(pt_iq)):
        ok &= compare(pt_iq[i], tt_iq[i],
                       f"SegMaskHead inter_query[{i}]", atol=1e-4)
    return ok


# ====================================================================
# TEST 8: SegMaskHead (stuff config, with self_attn + pos_query)
# ====================================================================
def test_seg_mask_head_stuff():
    print("\n" + "=" * 60)
    print("TEST 8: SegMaskHead (stuff-like, self_attn, with pos_query)")
    print("=" * 60)

    pt = PT_SegMaskHead(d_model=DIM, nhead=NHEAD, num_decoder_layers=2,
                        self_attn=True)
    pt.eval()

    tt = TT_SegMaskHead('t8_smh', d_model=DIM, nhead=NHEAD,
                         num_decoder_layers=2, use_self_attn=True)
    copy_seg_mask_head(pt, tt)

    mem_np = np.random.randn(B, N_MEM, DIM).astype(np.float32)
    qry_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)
    pos_np = np.random.randn(B, N_QUERY, DIM).astype(np.float32)

    # PyTorch: pos_memory=None, mask_query=None, mask_memory=None
    pt_attn, pt_masks, pt_iq = pt(
        torch.from_numpy(mem_np), None, None,
        torch.from_numpy(qry_np), None,
        torch.from_numpy(pos_np),
        hw_lvl=None)

    tt_attn, tt_masks, tt_iq = tt(
        F._from_data('t8.m', mem_np),
        None, None,
        F._from_data('t8.q', qry_np),
        None,
        F._from_data('t8.p', pos_np))

    ok = True
    ok &= compare(pt_attn, tt_attn, "SegMaskHead(stuff) attn_mask", atol=1e-4)
    for i in range(len(pt_masks)):
        ok &= compare(pt_masks[i], tt_masks[i],
                       f"SegMaskHead(stuff) mask[{i}]", atol=1e-4)
    for i in range(len(pt_iq)):
        ok &= compare(pt_iq[i], tt_iq[i],
                       f"SegMaskHead(stuff) inter_query[{i}]", atol=1e-4)
    return ok


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SegMaskHead Comparison Test: PyTorch vs TTSim")
    print("=" * 60)

    tests = [
        ("TEST 1: Mlp", test_mlp),
        ("TEST 2: SelfAttention", test_self_attention),
        ("TEST 3: Attention", test_attention),
        ("TEST 4: AttentionTail", test_attention_tail),
        ("TEST 5: Block (no self_attn)", test_block),
        ("TEST 6: Block (with self_attn)", test_block_self_attn),
        ("TEST 7: SegMaskHead (thing)", test_seg_mask_head_thing),
        ("TEST 8: SegMaskHead (stuff)", test_seg_mask_head_stuff),
    ]

    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"\n  [EXCEPTION] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok in results:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(results)} tests passed.")
    if passed == len(results):
        print("  ALL TESTS PASSED.")
    else:
        print("  SOME TESTS FAILED.")
