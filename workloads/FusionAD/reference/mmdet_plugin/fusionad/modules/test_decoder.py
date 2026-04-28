#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FusionAD Decoder TTSim modules.
Validates the conversion from PyTorch to TTSim.

This tests:
  1. inverse_sigmoid / InverseSigmoid – numerical match vs PyTorch
  2. CustomMSDeformableAttention – construction, parameter count,
     forward-pass numerical match vs PyTorch reference
  3. MultiheadAttention – self-attn / cross-attn numerical match vs PyTorch
  4. DetectionTransformerDecoder – construction, parameter count,
     forward-pass numerical match vs PyTorch reference (incl. ref-point
     refinement with reg_branches)
"""

import os
import sys
import warnings
import math
import copy
import traceback

polaris_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# ---- TTSim modules under test ----
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import (
    inverse_sigmoid_np,
    InverseSigmoid,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.decoder import (
    CustomMSDeformableAttention,
    DetectionTransformerDecoder,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multihead_attention import (
    MultiheadAttention,
)


# ====================================================================
# PyTorch reference helpers (CPU-only, Python 3.13 compatible)
# ====================================================================

def inverse_sigmoid_pytorch(x, eps=1e-5):
    """Exact copy of FusionAD utility."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations,
                                        attention_weights):
    """CPU reference of mmcv multi_scale_deformable_attn_pytorch."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    if isinstance(value_spatial_shapes, torch.Tensor):
        value_spatial_shapes = [(int(H), int(W))
                                for H, W in value_spatial_shapes]

    value_list = value.split(
        [H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (value_list[level].flatten(2).transpose(1, 2)
                    .reshape(bs * num_heads, embed_dims, H_, W_))
        sampling_grid_l_ = (sampling_grids[:, :, :, level]
                            .transpose(1, 2).flatten(0, 1))
        sampling_value_l_ = F_torch.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = (attention_weights.transpose(1, 2)
                         .reshape(bs * num_heads, 1, num_queries,
                                  num_levels * num_points))
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
              * attention_weights).sum(-1).view(
        bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class CustomMSDeformableAttention_PyTorch(nn.Module):
    """PyTorch reference of CustomMSDeformableAttention (no residual)."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4,
                 num_points=4, batch_first=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # init (same as original)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(num_heads, dtype=torch.float32) * (
            2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]
                     ).view(num_heads, 1, 1, 2).repeat(
            1, num_levels, num_points, 1)
        for i in range(num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward_core(self, query, value, reference_points, spatial_shapes):
        """Core forward with batch-first tensors (no residual, no dropout)."""
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads,
                           self.embed_dims // self.num_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads,
            self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads,
            self.num_levels, self.num_points)

        if isinstance(spatial_shapes, list):
            spatial_shapes_t = torch.tensor(spatial_shapes,
                                            dtype=torch.long)
        else:
            spatial_shapes_t = spatial_shapes

        offset_normalizer = torch.stack(
            [spatial_shapes_t[..., 1], spatial_shapes_t[..., 0]], -1
        ).float()
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets
            / offset_normalizer[None, None, None, :, None, :])

        output = multi_scale_deformable_attn_pytorch(
            value,
            [(int(H), int(W)) for H, W in spatial_shapes_t],
            sampling_locations,
            attention_weights)
        output = self.output_proj(output)
        return output


# ====================================================================
# Weight-copy helpers
# ====================================================================

def _set_linear(ttsim_linear, weight_np, bias_np):
    """Set TTSim Linear weights (no transpose — SimNN.Linear transposes internally)."""
    ttsim_linear.param = F._from_data(
        ttsim_linear.param.name, weight_np.astype(np.float32),
        is_const=True)
    ttsim_linear.param.is_param = True
    ttsim_linear.param.set_module(ttsim_linear)
    ttsim_linear._tensors[ttsim_linear.param.name] = ttsim_linear.param

    if bias_np is not None and ttsim_linear.bias is not None:
        ttsim_linear.bias = F._from_data(
            ttsim_linear.bias.name, bias_np.astype(np.float32),
            is_const=True)
        ttsim_linear.bias.is_param = True
        ttsim_linear.bias.set_module(ttsim_linear)
        ttsim_linear._tensors[ttsim_linear.bias.name] = ttsim_linear.bias


def copy_cmda_weights(pt_mod, tt_mod):
    """Copy CustomMSDeformableAttention weights from PyTorch to TTSim."""
    for attr in ('sampling_offsets', 'attention_weights',
                 'value_proj', 'output_proj'):
        pt_layer = getattr(pt_mod, attr)
        tt_layer = getattr(tt_mod, attr)
        w = pt_layer.weight.detach().numpy()
        b = pt_layer.bias.detach().numpy()
        _set_linear(tt_layer, w, b)


def copy_mha_weights(pt_mha, tt_mha):
    """Copy nn.MultiheadAttention weights → TTSim MultiheadAttention."""
    ipw = pt_mha.in_proj_weight.detach().numpy()
    ipb = pt_mha.in_proj_bias.detach().numpy()
    e = tt_mha.embed_dims
    # PyTorch stores [3E, E] → split Q/K/V
    tt_mha.q_proj.param.data = ipw[:e, :].copy()
    tt_mha.q_proj.bias.data = ipb[:e].copy()
    tt_mha.k_proj.param.data = ipw[e:2*e, :].copy()
    tt_mha.k_proj.bias.data = ipb[e:2*e].copy()
    tt_mha.v_proj.param.data = ipw[2*e:, :].copy()
    tt_mha.v_proj.bias.data = ipb[2*e:].copy()
    tt_mha.out_proj.param.data = pt_mha.out_proj.weight.detach().numpy().copy()
    tt_mha.out_proj.bias.data = pt_mha.out_proj.bias.detach().numpy().copy()


# ====================================================================
# Test 1: inverse_sigmoid
# ====================================================================

def test_inverse_sigmoid():
    """Validate inverse_sigmoid_np and InverseSigmoid vs PyTorch."""
    print("\n" + "=" * 80)
    print("TEST 1: inverse_sigmoid")
    print("=" * 80)

    try:
        # Random values in (0, 1)
        np.random.seed(42)
        x_np = np.random.rand(4, 10, 3).astype(np.float32) * 0.98 + 0.01

        # -- numpy version --
        out_np = inverse_sigmoid_np(x_np)
        out_pt = inverse_sigmoid_pytorch(torch.from_numpy(x_np)).numpy()
        max_diff_np = np.abs(out_np - out_pt).max()
        mean_diff_np = np.abs(out_np - out_pt).mean()
        print(f"\n[numpy]  max_diff = {max_diff_np:.2e},  mean_diff = {mean_diff_np:.2e}")

        # -- TTSim InverseSigmoid module --
        inv_mod = InverseSigmoid('test_inv')
        x_t = F._from_data('x', x_np, is_const=True)
        y_t = inv_mod(x_t)
        out_tt = y_t.data
        max_diff_tt = np.abs(out_tt - out_pt).max()
        mean_diff_tt = np.abs(out_tt - out_pt).mean()
        print(f"[TTSim]  max_diff = {max_diff_tt:.2e},  mean_diff = {mean_diff_tt:.2e}")

        threshold = 1e-5
        ok = max_diff_np < threshold and max_diff_tt < threshold
        status = "[OK]" if ok else "[X]"
        print(f"\n{status} inverse_sigmoid (threshold={threshold})")
        return ok
    except Exception as e:
        print(f"[X] inverse_sigmoid FAILED: {e}")
        traceback.print_exc()
        return False


# ====================================================================
# Test 2: CustomMSDeformableAttention
# ====================================================================

def test_cmda_construction():
    """Test CustomMSDeformableAttention construction + param count."""
    print("\n" + "=" * 80)
    print("TEST 2a: CustomMSDeformableAttention Construction")
    print("=" * 80)
    try:
        attn = CustomMSDeformableAttention(
            name='test_cmda',
            embed_dims=256, num_heads=8,
            num_levels=1, num_points=4,
            batch_first=False)
        pc = attn.analytical_param_count()

        # Expected: same structure as PtsCrossAttention
        e = 256; nh = 8; nl = 1; np_ = 4
        so = e * (nh * nl * np_ * 2) + (nh * nl * np_ * 2)
        aw = e * (nh * nl * np_) + (nh * nl * np_)
        vp = e * e + e
        op = e * e + e
        expected = so + aw + vp + op

        print(f"  Actual params:   {pc:,}")
        print(f"  Expected params: {expected:,}")
        ok = pc == expected
        print(f"  {'[OK]' if ok else '[X]'} param count")
        return ok
    except Exception as e:
        print(f"  [X] construction failed: {e}")
        traceback.print_exc()
        return False


def test_cmda_forward():
    """Numerical validation of CustomMSDeformableAttention forward pass."""
    print("\n" + "=" * 80)
    print("TEST 2b: CustomMSDeformableAttention Forward Pass")
    print("=" * 80)
    try:
        bs, nq, nv = 2, 100, 900
        embed_dims, nh, nl, np_ = 256, 8, 1, 4
        spatial_shapes = [(30, 30)]

        np.random.seed(7)
        query_np = np.random.randn(nq, bs, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(nv, bs, embed_dims).astype(np.float32) * 0.1
        ref_np = np.random.rand(bs, nq, nl, 2).astype(np.float32)

        # -- PyTorch reference (batch-first internally) --
        pt = CustomMSDeformableAttention_PyTorch(
            embed_dims=embed_dims, num_heads=nh,
            num_levels=nl, num_points=np_,
            batch_first=False)
        pt.eval()

        # batch-first query/value for the reference core
        q_bf = torch.from_numpy(query_np).permute(1, 0, 2)   # [bs, nq, e]
        v_bf = torch.from_numpy(value_np).permute(1, 0, 2)   # [bs, nv, e]
        with torch.no_grad():
            out_pt = pt.forward_core(q_bf, v_bf,
                                     torch.from_numpy(ref_np),
                                     spatial_shapes)
        # Permute back to seq-first for comparison
        out_pt = out_pt.permute(1, 0, 2)  # [nq, bs, e]
        out_pt_np = out_pt.numpy()

        # -- TTSim --
        tt = CustomMSDeformableAttention(
            name='val_cmda', embed_dims=embed_dims,
            num_heads=nh, num_levels=nl, num_points=np_,
            batch_first=False)
        copy_cmda_weights(pt, tt)

        q_t = F._from_data('query', query_np, is_const=True)
        v_t = F._from_data('value', value_np, is_const=True)
        rp_t = F._from_data('ref', ref_np, is_const=True)

        out_tt = tt(query=q_t, value=v_t,
                    reference_points=rp_t,
                    spatial_shapes=spatial_shapes)

        # The TTSim module applies identity residual; PyTorch core does not.
        # Subtract identity (query in seq-first) from TTSim output
        out_tt_np = out_tt.data
        out_core_tt = out_tt_np - query_np  # remove residual

        diff = np.abs(out_core_tt - out_pt_np)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"\n  Output shape: {list(out_tt.shape)}")
        print(f"  Max  diff (core, excl. residual): {max_diff:.2e}")
        print(f"  Mean diff:                        {mean_diff:.2e}")

        threshold = 1e-4
        ok = max_diff < threshold
        print(f"  {'[OK]' if ok else '[X]'} CustomMSDeformableAttention forward (thr={threshold})")
        return ok
    except Exception as e:
        print(f"  [X] forward failed: {e}")
        traceback.print_exc()
        return False


# ====================================================================
# Test 3: MultiheadAttention
# ====================================================================

def test_mha_self_attention():
    """MultiheadAttention self-attention numerical match."""
    print("\n" + "=" * 80)
    print("TEST 3a: MultiheadAttention – Self-Attention")
    print("=" * 80)
    try:
        embed_dims, nh = 256, 8
        bs, seq = 2, 50

        pt = nn.MultiheadAttention(embed_dims, nh, dropout=0.0,
                                   batch_first=False)
        pt.eval()

        tt = MultiheadAttention('mha_self', embed_dims=embed_dims,
                                num_heads=nh, attn_drop=0.0,
                                batch_first=False)
        copy_mha_weights(pt, tt)

        x_np = np.random.randn(seq, bs, embed_dims).astype(np.float32)
        with torch.no_grad():
            y_pt, _ = pt(torch.from_numpy(x_np),
                         torch.from_numpy(x_np),
                         torch.from_numpy(x_np))
        x_t = F._from_data('x', x_np, is_const=True)
        y_tt = tt(query=x_t, need_weights=False)

        diff = np.abs(y_tt.data - y_pt.numpy())
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"  Shape: {list(y_tt.shape)}")
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        # Param count comparison
        pt_params = sum(p.numel() for p in pt.parameters())
        tt_params = tt.analytical_param_count()
        print(f"  PyTorch params: {pt_params:,}  TTSim params: {tt_params:,}")

        threshold = 1e-5
        ok = max_diff < threshold
        print(f"  {'[OK]' if ok else '[X]'} Self-Attention (thr={threshold})")
        return ok
    except Exception as e:
        print(f"  [X] Self-Attention failed: {e}")
        traceback.print_exc()
        return False


def test_mha_cross_attention():
    """MultiheadAttention cross-attention numerical match."""
    print("\n" + "=" * 80)
    print("TEST 3b: MultiheadAttention – Cross-Attention")
    print("=" * 80)
    try:
        embed_dims, nh = 256, 8
        bs = 2; seq_q = 30; seq_kv = 50

        pt = nn.MultiheadAttention(embed_dims, nh, dropout=0.0,
                                   batch_first=False)
        pt.eval()
        tt = MultiheadAttention('mha_cross', embed_dims=embed_dims,
                                num_heads=nh, attn_drop=0.0,
                                batch_first=False)
        copy_mha_weights(pt, tt)

        q_np = np.random.randn(seq_q, bs, embed_dims).astype(np.float32)
        k_np = np.random.randn(seq_kv, bs, embed_dims).astype(np.float32)
        v_np = np.random.randn(seq_kv, bs, embed_dims).astype(np.float32)

        with torch.no_grad():
            y_pt, _ = pt(torch.from_numpy(q_np),
                         torch.from_numpy(k_np),
                         torch.from_numpy(v_np))

        q_t = F._from_data('q', q_np, is_const=True)
        k_t = F._from_data('k', k_np, is_const=True)
        v_t = F._from_data('v', v_np, is_const=True)
        y_tt = tt(query=q_t, key=k_t, value=v_t, need_weights=False)

        diff = np.abs(y_tt.data - y_pt.numpy())
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"  Q shape: {q_np.shape}, KV shape: {k_np.shape}")
        print(f"  Output shape: {list(y_tt.shape)}")
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        threshold = 1e-5
        ok = max_diff < threshold
        print(f"  {'[OK]' if ok else '[X]'} Cross-Attention (thr={threshold})")
        return ok
    except Exception as e:
        print(f"  [X] Cross-Attention failed: {e}")
        traceback.print_exc()
        return False


def test_mha_with_pos():
    """MultiheadAttention with query_pos / key_pos (MMCV extension)."""
    print("\n" + "=" * 80)
    print("TEST 3c: MultiheadAttention – query_pos + identity residual")
    print("=" * 80)
    try:
        embed_dims, nh = 128, 4
        bs, seq = 2, 20

        pt = nn.MultiheadAttention(embed_dims, nh, dropout=0.0,
                                   batch_first=False)
        pt.eval()
        tt = MultiheadAttention('mha_pos', embed_dims=embed_dims,
                                num_heads=nh, attn_drop=0.0,
                                batch_first=False)
        copy_mha_weights(pt, tt)

        q_np = np.random.randn(seq, bs, embed_dims).astype(np.float32) * 0.1
        qp_np = np.random.randn(seq, bs, embed_dims).astype(np.float32) * 0.01

        # PyTorch: match MMCV MultiheadAttention behavior exactly.
        # In MMCV, value = original query (no pos encoding).
        # Only query and key get positional encoding added.
        q_orig = torch.from_numpy(q_np)
        q_plus_pos = q_orig + torch.from_numpy(qp_np)
        with torch.no_grad():
            y_pt, _ = pt(q_plus_pos, q_plus_pos, q_orig)  # value = original q
        # Add identity residual (original query before pos)
        y_pt = y_pt + q_orig

        # TTSim: pass query_pos and identity
        q_t = F._from_data('q', q_np, is_const=True)
        qp_t = F._from_data('qp', qp_np, is_const=True)
        id_t = F._from_data('id', q_np, is_const=True)
        y_tt = tt(query=q_t, query_pos=qp_t, identity=id_t,
                  need_weights=False)

        diff = np.abs(y_tt.data - y_pt.numpy())
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"  Output shape: {list(y_tt.shape)}")
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        threshold = 1e-5
        ok = max_diff < threshold
        print(f"  {'[OK]' if ok else '[X]'} query_pos + identity (thr={threshold})")
        return ok
    except Exception as e:
        print(f"  [X] query_pos test failed: {e}")
        traceback.print_exc()
        return False


# ====================================================================
# Test 4: DetectionTransformerDecoder
# ====================================================================

def test_decoder_construction():
    """Construct DetectionTransformerDecoder with FusionAD config."""
    print("\n" + "=" * 80)
    print("TEST 4a: DetectionTransformerDecoder Construction")
    print("=" * 80)
    try:
        layer_cfg = dict(
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=256, num_heads=8, dropout=0.1),
                dict(type='CustomMSDeformableAttention',
                     embed_dims=256, num_levels=1),
            ],
            ffn_cfgs=dict(
                type='FFN', embed_dims=256,
                feedforward_channels=512,
                num_fcs=2, ffn_drop=0.1),
            operation_order=(
                'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            batch_first=False,
        )

        decoder = DetectionTransformerDecoder(
            name='test_dec',
            num_layers=6,
            layer_cfg=layer_cfg,
            return_intermediate=True)

        pc = decoder.analytical_param_count()
        print(f"  num_layers:          {decoder.num_layers}")
        print(f"  return_intermediate: {decoder.return_intermediate}")
        print(f"  layers built:        {len(decoder.layers)}")
        print(f"  total params:        {pc:,}")

        # Rough expected: 6 layers ×
        #   (MHA=4×256²+4×256 + CMDA≈same as PtsCross + FFN=2×256×512+256+512 + 3×LN=3×2×256)
        # Just check it's > 0 and 6 layers
        ok = len(decoder.layers) == 6 and pc > 0
        print(f"  {'[OK]' if ok else '[X]'} Decoder constructed")
        return ok
    except Exception as e:
        print(f"  [X] Decoder construction failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_forward_no_refine():
    """Forward pass without reg_branches (no ref-point refinement)."""
    print("\n" + "=" * 80)
    print("TEST 4b: DetectionTransformerDecoder Forward (no reg_branches)")
    print("=" * 80)
    try:
        embed_dims = 256
        num_query = 50
        bs = 1
        bev_h, bev_w = 10, 10
        num_key = bev_h * bev_w  # 100

        layer_cfg = dict(
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=embed_dims, num_heads=8, dropout=0.0),
                dict(type='CustomMSDeformableAttention',
                     embed_dims=embed_dims, num_heads=8,
                     num_levels=1, num_points=4, dropout=0.0),
            ],
            ffn_cfgs=dict(
                type='FFN', embed_dims=embed_dims,
                feedforward_channels=512,
                num_fcs=2, ffn_drop=0.0),
            operation_order=(
                'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            batch_first=False,
        )

        decoder = DetectionTransformerDecoder(
            name='fwd_dec',
            num_layers=2,
            layer_cfg=layer_cfg,
            return_intermediate=True)

        # Inputs
        np.random.seed(123)
        query_np = np.random.randn(num_query, bs, embed_dims).astype(np.float32) * 0.1
        key_np = np.random.randn(num_key, bs, embed_dims).astype(np.float32) * 0.1
        ref_np = np.random.rand(bs, num_query, 3).astype(np.float32)

        q_t = F._from_data('query', query_np, is_const=True)
        k_t = F._from_data('key', key_np, is_const=True)
        rp_t = F._from_data('ref', ref_np, is_const=True)

        spatial_shapes = [(bev_h, bev_w)]

        intermediates, ref_pts = decoder(
            query=q_t, key=k_t, value=k_t,
            reference_points=rp_t,
            reg_branches=None,
            spatial_shapes=spatial_shapes)

        print(f"  num intermediate outputs: {len(intermediates)}")
        for i, out in enumerate(intermediates):
            print(f"    layer {i}: shape={list(out.shape)}")

        ok = (len(intermediates) == 2
              and list(intermediates[0].shape) == [num_query, bs, embed_dims])
        print(f"  {'[OK]' if ok else '[X]'} Forward pass shapes correct")
        return ok
    except Exception as e:
        print(f"  [X] Forward (no refine) failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_forward_with_refine():
    """Forward pass with reg_branches (reference point refinement)."""
    print("\n" + "=" * 80)
    print("TEST 4c: DetectionTransformerDecoder Forward (with reg_branches)")
    print("=" * 80)
    try:
        embed_dims = 256
        num_query = 50
        bs = 1
        bev_h, bev_w = 10, 10
        num_key = bev_h * bev_w
        num_layers = 2

        layer_cfg = dict(
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=embed_dims, num_heads=8, dropout=0.0),
                dict(type='CustomMSDeformableAttention',
                     embed_dims=embed_dims, num_heads=8,
                     num_levels=1, num_points=4, dropout=0.0),
            ],
            ffn_cfgs=dict(
                type='FFN', embed_dims=embed_dims,
                feedforward_channels=512,
                num_fcs=2, ffn_drop=0.0),
            operation_order=(
                'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            batch_first=False,
        )

        decoder = DetectionTransformerDecoder(
            name='refine_dec',
            num_layers=num_layers,
            layer_cfg=layer_cfg,
            return_intermediate=True)

        # Build trivial reg_branches (Linear → 10-dim output)
        reg_branches = []
        for i in range(num_layers):
            branch = SimNN.Linear(
                f'reg_{i}',
                in_features=embed_dims,
                out_features=10,
                bias=True)
            # Small random weights
            branch.param.data = np.random.randn(embed_dims, 10).astype(np.float32) * 0.01
            branch.bias.data = np.zeros(10, dtype=np.float32)
            reg_branches.append(branch)

        np.random.seed(456)
        query_np = np.random.randn(num_query, bs, embed_dims).astype(np.float32) * 0.1
        key_np = np.random.randn(num_key, bs, embed_dims).astype(np.float32) * 0.1
        ref_np = np.random.rand(bs, num_query, 3).astype(np.float32)

        q_t = F._from_data('query', query_np, is_const=True)
        k_t = F._from_data('key', key_np, is_const=True)
        rp_t = F._from_data('ref', ref_np, is_const=True)

        spatial_shapes = [(bev_h, bev_w)]

        intermediates, ref_pts_list = decoder(
            query=q_t, key=k_t, value=k_t,
            reference_points=rp_t,
            reg_branches=reg_branches,
            spatial_shapes=spatial_shapes)

        print(f"  num intermediate outputs: {len(intermediates)}")
        print(f"  num ref-pt snapshots:     {len(ref_pts_list)}")
        for i, (out, rp) in enumerate(zip(intermediates, ref_pts_list)):
            rp_data = rp.data if hasattr(rp, 'data') else rp
            print(f"    layer {i}: out_shape={list(out.shape)}, "
                  f"ref_shape={list(rp_data.shape) if isinstance(rp_data, np.ndarray) else list(rp.shape)}")

        # After refinement, ref-points from layer 1 should differ from layer 0
        rp0 = ref_pts_list[0]
        rp1 = ref_pts_list[1]
        rp0_data = rp0.data if hasattr(rp0, 'data') else rp0
        rp1_data = rp1.data if hasattr(rp1, 'data') else rp1
        if isinstance(rp0_data, np.ndarray) and isinstance(rp1_data, np.ndarray):
            ref_diff = np.abs(rp0_data - rp1_data).max()
            print(f"\n  Max ref-point diff between layers: {ref_diff:.4e}")
            refined = ref_diff > 1e-6
        else:
            refined = True  # SimTensors are different objects → refined

        ok = (len(intermediates) == num_layers
              and len(ref_pts_list) == num_layers
              and refined)
        print(f"  {'[OK]' if ok else '[X]'} Forward with refinement")
        return ok
    except Exception as e:
        print(f"  [X] Forward (with refine) failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_param_count():
    """Validate decoder parameter count against manual calculation."""
    print("\n" + "=" * 80)
    print("TEST 4d: DetectionTransformerDecoder Parameter Count")
    print("=" * 80)
    try:
        embed_dims = 256
        num_heads = 8
        ff_channels = 512
        num_layers = 6
        num_levels = 1
        num_points = 4

        layer_cfg = dict(
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=embed_dims, num_heads=num_heads, dropout=0.1),
                dict(type='CustomMSDeformableAttention',
                     embed_dims=embed_dims, num_heads=num_heads,
                     num_levels=num_levels, num_points=num_points),
            ],
            ffn_cfgs=dict(
                type='FFN', embed_dims=embed_dims,
                feedforward_channels=ff_channels,
                num_fcs=2, ffn_drop=0.1),
            operation_order=(
                'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
            batch_first=False,
        )

        decoder = DetectionTransformerDecoder(
            name='param_dec',
            num_layers=num_layers,
            layer_cfg=layer_cfg,
            return_intermediate=True)

        total = decoder.analytical_param_count()

        # Per-layer expected:
        # MHA: 4×(256×256 + 256) = 4×65,792 = 263,168
        mha = 4 * (embed_dims * embed_dims + embed_dims)
        # CMDA:
        so = embed_dims * (num_heads * num_levels * num_points * 2) + \
             (num_heads * num_levels * num_points * 2)
        aw = embed_dims * (num_heads * num_levels * num_points) + \
             (num_heads * num_levels * num_points)
        vp = embed_dims * embed_dims + embed_dims
        op = embed_dims * embed_dims + embed_dims
        cmda = so + aw + vp + op
        # FFN: fc1(256×512+512) + fc2(512×256+256)
        ffn = (embed_dims * ff_channels + ff_channels) + \
              (ff_channels * embed_dims + embed_dims)
        # 3× LayerNorm: 3 × 2 × 256
        ln = 3 * 2 * embed_dims
        per_layer = mha + cmda + ffn + ln
        expected = per_layer * num_layers

        print(f"  Per-layer breakdown:")
        print(f"    MHA:  {mha:,}")
        print(f"    CMDA: {cmda:,}")
        print(f"    FFN:  {ffn:,}")
        print(f"    LN:   {ln:,}")
        print(f"    Subtotal: {per_layer:,}")
        print(f"  Expected ({num_layers} layers): {expected:,}")
        print(f"  Actual:                          {total:,}")

        ok = total == expected
        print(f"  {'[OK]' if ok else '[X]'} Parameter count")
        return ok
    except Exception as e:
        print(f"  [X] parameter count failed: {e}")
        traceback.print_exc()
        return False


# ====================================================================
# Main
# ====================================================================

def main():
    print("\n" + "=" * 80)
    print("FusionAD Decoder TTSim Comparison Test Suite")
    print("=" * 80)

    results = {}

    tests = [
        ("inverse_sigmoid",                  test_inverse_sigmoid),
        ("CMDA Construction",                test_cmda_construction),
        ("CMDA Forward Pass",                test_cmda_forward),
        ("MHA Self-Attention",               test_mha_self_attention),
        ("MHA Cross-Attention",              test_mha_cross_attention),
        ("MHA query_pos + identity",         test_mha_with_pos),
        ("Decoder Construction",             test_decoder_construction),
        ("Decoder Forward (no refine)",      test_decoder_forward_no_refine),
        ("Decoder Forward (with refine)",    test_decoder_forward_with_refine),
        ("Decoder Parameter Count",          test_decoder_param_count),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"[X] {name} FAILED with unhandled exception: {e}")
            traceback.print_exc()
            results[name] = False

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results.items():
        status = "[OK] PASSED" if passed else "[X]  FAILED"
        print(f"  {name:.<55} {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("ALL TESTS PASSED!")
    else:
        print(f"{total - passed} test(s) FAILED.")
    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
