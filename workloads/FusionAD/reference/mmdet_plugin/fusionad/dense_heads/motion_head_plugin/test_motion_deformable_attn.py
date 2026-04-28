#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for motion_deformable_attn.py (TTSim vs PyTorch).

Validates that MotionDeformableAttention, CustomModeMultiheadAttention,
and MotionTransformerAttentionLayer produce identical results to
manually-built PyTorch equivalents.

No mmcv dependency — all PyTorch references are built from plain torch.nn.
"""

import os
import sys
import copy
import traceback
import math
import warnings

polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# TTSim modules
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.motion_deformable_attn import (
    MotionDeformableAttention,
    CustomModeMultiheadAttention,
    MotionTransformerAttentionLayer,
)


# ====================================================================
# Pure-PyTorch reference implementations (no mmcv)
# ====================================================================

def multi_scale_deformable_attn_pytorch(value, spatial_shapes,
                                        sampling_locations,
                                        attention_weights):
    """Pure-PyTorch multi-scale deformable attention.

    Args:
        value: (bs, num_keys, num_heads, embed_dims_per_head)
        spatial_shapes: list of (H, W) tuples
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2)
                            values in [0, 1]
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        (bs, num_queries, num_heads * embed_dims_per_head)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    split_sizes = [H * W for H, W in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1,1]

    sampling_value_list = []
    for level, (H_, W_) in enumerate(spatial_shapes):
        # (bs, H*W, num_heads, D) -> (bs*num_heads, D, H, W)
        value_l_ = (value_list[level].flatten(2).transpose(1, 2)
                     .reshape(bs * num_heads, embed_dims, H_, W_))
        # (bs, nq, nh, np, 2) -> (bs*nh, nq, np, 2)
        sampling_grid_l_ = (sampling_grids[:, :, :, level]
                            .transpose(1, 2).flatten(0, 1))
        sampling_value_l_ = F_torch.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # attention_weights: (bs, nq, nh, nl, np) -> (bs*nh, 1, nq, nl*np)
    attn = (attention_weights.transpose(1, 2)
            .reshape(bs * num_heads, 1, num_queries, num_levels * num_points))
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attn
              ).sum(-1).view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class PT_CustomModeMultiheadAttention(nn.Module):
    """Pure-PyTorch equivalent of CustomModeMultiheadAttention."""

    def __init__(self, embed_dims, num_heads):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dims, num_heads,
                                          dropout=0.0, batch_first=False)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None):
        if query_pos is not None:
            query_pos = query_pos.unsqueeze(2)  # (B,A,D)->(B,A,1,D)
        if key_pos is not None:
            key_pos = key_pos.unsqueeze(2)

        bs, n_agent, n_query, D = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Flatten (B,A,P,D)->(B*A,P,D), transpose to (P,B*A,D)
        q = query.flatten(0, 1).transpose(0, 1)
        k = key.flatten(0, 1).transpose(0, 1)
        v = value.flatten(0, 1).transpose(0, 1)
        ident = identity.flatten(0, 1)

        out = self.attn(q, k, v)[0]  # (P, B*A, D)
        out = out.transpose(0, 1)     # (B*A, P, D)
        out = ident + out              # residual
        return out.view(bs, n_agent, n_query, D)


class PT_MotionDeformableAttention(nn.Module):
    """Pure-PyTorch equivalent of MotionDeformableAttention (no mmcv)."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1,
                 num_points=4, num_steps=1, sample_index=-1,
                 bev_range=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index
        self.bev_range = bev_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_steps * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_steps * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(num_steps * embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True))

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, spatial_shapes=None, level_start_index=None,
                bbox_results=None, reference_trajs=None, **kwargs):
        bs, num_agent, num_mode, _ = query.shape
        num_query = num_agent * num_mode

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        query = torch.flatten(query, start_dim=1, end_dim=2)

        # value: (num_bev, B, D) -> (B, num_bev, D)
        value = value.permute(1, 0, 2)
        bs_, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs_, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs_, num_query, self.num_heads, self.num_steps,
            self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs_, num_query, self.num_heads, self.num_steps,
            self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs_, num_query, self.num_heads, self.num_steps,
            self.num_levels, self.num_points)

        # Reference trajectory processing
        reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]
        reference_trajs_ego = self._agent_to_ego(
            copy.deepcopy(reference_trajs), bbox_results).detach()
        reference_trajs_ego = torch.flatten(reference_trajs_ego, 1, 2)
        # (B, nq, 1, L, 2) -> (B, nq, 1, 1, L, 1, 2)
        reference_trajs_ego = reference_trajs_ego[:, :, None, :, :, None, :]
        bev = self.bev_range
        reference_trajs_ego[..., 0] -= bev[0]
        reference_trajs_ego[..., 1] -= bev[1]
        reference_trajs_ego[..., 0] /= (bev[3] - bev[0])
        reference_trajs_ego[..., 1] /= (bev[4] - bev[1])

        if isinstance(spatial_shapes, np.ndarray):
            spatial_shapes_t = torch.from_numpy(spatial_shapes).long()
        else:
            spatial_shapes_t = spatial_shapes
        offset_normalizer = torch.stack(
            [spatial_shapes_t[..., 1], spatial_shapes_t[..., 0]], -1).float()
        sampling_locations = (reference_trajs_ego
                              + sampling_offsets
                              / offset_normalizer[None, None, None, None, :, None, :])

        # Rearrange: (bs nq nh ns nl np c) -> (bs nq ns nh nl np c)
        sampling_locations = sampling_locations.permute(0, 1, 3, 2, 4, 5, 6)
        attention_weights = attention_weights.permute(0, 1, 3, 2, 4, 5)
        sampling_locations = sampling_locations.reshape(
            bs_, num_query * self.num_steps, self.num_heads,
            self.num_levels, self.num_points, 2)
        attention_weights = attention_weights.reshape(
            bs_, num_query * self.num_steps, self.num_heads,
            self.num_levels, self.num_points)

        spatial_shapes_list = [(int(spatial_shapes_t[i, 0]),
                                int(spatial_shapes_t[i, 1]))
                               for i in range(self.num_levels)]
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes_list, sampling_locations, attention_weights)

        output = output.view(bs_, num_query, self.num_steps, -1)
        output = torch.flatten(output, 2, 3)
        output = self.output_proj(output)
        output = output.view(bs_, num_agent, num_mode, -1)
        return output + identity  # dropout omitted

    def _agent_to_ego(self, reference_trajs, bbox_results):
        batch_size = reference_trajs.shape[0]
        ref_ego_list = []
        for i in range(batch_size):
            boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
            det_centers = boxes_3d.gravity_center.to(reference_trajs.device)
            batch_ref = reference_trajs[i]
            batch_ref += det_centers[:, None, None, None, :2]
            ref_ego_list.append(batch_ref)
        return torch.stack(ref_ego_list)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear_pt_to_tt(pt_linear, tt_linear):
    """Copy PyTorch nn.Linear -> TTSim SimNN.Linear."""
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_mha_weights(pt_mha, tt_mha):
    """Copy PyTorch nn.MultiheadAttention -> TTSim MultiheadAttention."""
    D = pt_mha.embed_dim
    w = pt_mha.in_proj_weight.data.detach().numpy()
    b = pt_mha.in_proj_bias.data.detach().numpy()
    wq, wk, wv = w[:D], w[D:2*D], w[2*D:]
    bq, bk, bv = b[:D], b[D:2*D], b[2*D:]

    tt_mha.q_proj.param.data = wq.astype(np.float32)
    tt_mha.q_proj.bias.data = bq.astype(np.float32)
    tt_mha.k_proj.param.data = wk.astype(np.float32)
    tt_mha.k_proj.bias.data = bk.astype(np.float32)
    tt_mha.v_proj.param.data = wv.astype(np.float32)
    tt_mha.v_proj.bias.data = bv.astype(np.float32)

    tt_mha.out_proj.param.data = pt_mha.out_proj.weight.data.detach().numpy().astype(np.float32)
    tt_mha.out_proj.bias.data = pt_mha.out_proj.bias.data.detach().numpy().astype(np.float32)


def copy_cmma_weights(pt_cmma, tt_cmma):
    """Copy PT_CustomModeMultiheadAttention -> TTSim CustomModeMultiheadAttention."""
    copy_mha_weights(pt_cmma.attn, tt_cmma.attn)


def copy_mda_weights(pt_mda, tt_mda):
    """Copy PT_MotionDeformableAttention -> TTSim MotionDeformableAttention."""
    copy_linear_pt_to_tt(pt_mda.sampling_offsets, tt_mda.sampling_offsets)
    copy_linear_pt_to_tt(pt_mda.attention_weights, tt_mda.attention_weights)
    copy_linear_pt_to_tt(pt_mda.value_proj, tt_mda.value_proj)
    # output_proj: Sequential(Linear, LayerNorm, ReLU)
    copy_linear_pt_to_tt(pt_mda.output_proj[0], tt_mda.output_proj_linear)
    # Neutralize PT LayerNorm (TTSim LN has no affine params)
    pt_mda.output_proj[1].weight.data.fill_(1.0)
    pt_mda.output_proj[1].bias.data.fill_(0.0)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-4):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') and not isinstance(tt_out, np.ndarray) else tt_out
    print(f"\n  {name}:")
    print(f"    PyTorch shape: {pt_np.shape}")
    print(f"    TTSim   shape: {tuple(tt_np.shape)}")
    if pt_np.shape != tuple(tt_np.shape):
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np - tt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if np.allclose(pt_np, tt_np, atol=atol):
        print(f"    [OK] Match (atol={atol})")
        return True
    print(f"    [FAIL] Exceeds tolerance")
    return False


# ====================================================================
# Fake bbox_results for agent->ego conversion
# ====================================================================

class FakeLiDARInstance3DBoxes:
    """Minimal stand-in for LiDARInstance3DBoxes providing gravity_center."""
    def __init__(self, centers):
        self._centers = torch.from_numpy(centers.astype(np.float32))

    @property
    def gravity_center(self):
        return self._centers


def make_fake_bbox_results(batch_size, num_agents, seed=123):
    """Create fake bbox_results with random agent centers."""
    rng = np.random.RandomState(seed)
    results = []
    for _ in range(batch_size):
        centers = rng.randn(num_agents, 3).astype(np.float32) * 20
        boxes = FakeLiDARInstance3DBoxes(centers)
        scores = torch.ones(num_agents)
        labels = torch.zeros(num_agents, dtype=torch.long)
        bbox_index = torch.arange(num_agents)
        mask = torch.ones(num_agents, dtype=torch.bool)
        results.append((boxes, scores, labels, bbox_index, mask))
    return results


# ====================================================================
# Config
# ====================================================================

np.random.seed(42)
torch.manual_seed(42)

D = 64
H = 8
BS = 1
A = 8
P = 6
NUM_LEVELS = 1
NUM_POINTS = 4
NUM_STEPS = 12
BEV_H, BEV_W = 50, 50
BEV_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
SAMPLE_INDEX = -1

passed = 0
failed = 0


# ====================================================================
# TEST 1: CustomModeMultiheadAttention — PyTorch vs TTSim
# ====================================================================

print("=" * 80)
print("TEST 1: CustomModeMultiheadAttention — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_cmma = PT_CustomModeMultiheadAttention(embed_dims=D, num_heads=H)
    tt_cmma = CustomModeMultiheadAttention('t1_cmma', embed_dims=D, num_heads=H)
    copy_cmma_weights(pt_cmma, tt_cmma)

    q_np = np.random.randn(BS, A, P, D).astype(np.float32)
    qpos_np = np.random.randn(BS, A, D).astype(np.float32)
    kpos_np = np.random.randn(BS, A, D).astype(np.float32)

    q_pt = torch.from_numpy(q_np)
    qpos_pt = torch.from_numpy(qpos_np)
    kpos_pt = torch.from_numpy(kpos_np)
    q_tt = F._from_data('t1_q', q_np, is_const=True)
    qpos_tt = F._from_data('t1_qpos', qpos_np, is_const=True)
    kpos_tt = F._from_data('t1_kpos', kpos_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_cmma(q_pt, query_pos=qpos_pt, key_pos=kpos_pt)

    tt_out = tt_cmma(q_tt, query_pos=qpos_tt, key_pos=kpos_tt)

    ok = compare(pt_out, tt_out, "CustomModeMultiheadAttention")
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 1")

except Exception as e:
    print(f"  [FAIL] TEST 1 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 2: CustomModeMultiheadAttention — param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: CustomModeMultiheadAttention — param count")
print("=" * 80)

try:
    expected = 4 * (D * D + D)
    actual = tt_cmma.analytical_param_count()
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")
    ok = expected == actual
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 2")

except Exception as e:
    print(f"  [FAIL] TEST 2 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 3: CustomModeMultiheadAttention — no positional encodings
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: CustomModeMultiheadAttention — no positional encodings")
print("=" * 80)

try:
    with torch.no_grad():
        pt_out_nopos = pt_cmma(q_pt)

    tt_out_nopos = tt_cmma(q_tt)

    ok = compare(pt_out_nopos, tt_out_nopos, "CMMA (no pos)")
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 3")

except Exception as e:
    print(f"  [FAIL] TEST 3 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 4: CustomModeMultiheadAttention — varying A, P
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: CustomModeMultiheadAttention — varying A, P")
print("=" * 80)

try:
    all_ok = True
    for test_A, test_P in [(4, 3), (12, 6), (8, 10)]:
        pt_cmma2 = PT_CustomModeMultiheadAttention(embed_dims=D, num_heads=H)
        tt_cmma2 = CustomModeMultiheadAttention(f't4_cmma_{test_A}_{test_P}',
                                                 embed_dims=D, num_heads=H)
        copy_cmma_weights(pt_cmma2, tt_cmma2)

        x_np = np.random.randn(BS, test_A, test_P, D).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_tt = F._from_data(f't4_x_{test_A}_{test_P}', x_np, is_const=True)
        qp_np = np.random.randn(BS, test_A, D).astype(np.float32)
        kp_np = np.random.randn(BS, test_A, D).astype(np.float32)

        with torch.no_grad():
            pt_o = pt_cmma2(x_pt, query_pos=torch.from_numpy(qp_np),
                            key_pos=torch.from_numpy(kp_np))
        tt_o = tt_cmma2(x_tt,
                        query_pos=F._from_data(f't4_qp_{test_A}', qp_np, is_const=True),
                        key_pos=F._from_data(f't4_kp_{test_A}', kp_np, is_const=True))

        sub_ok = compare(pt_o, tt_o, f"CMMA A={test_A} P={test_P}")
        if not sub_ok:
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 4")

except Exception as e:
    print(f"  [FAIL] TEST 4 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 5: MotionDeformableAttention — construction & param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: MotionDeformableAttention — construction & param count")
print("=" * 80)

try:
    tt_mda = MotionDeformableAttention(
        't5_mda', embed_dims=D, num_heads=H,
        num_levels=NUM_LEVELS, num_points=NUM_POINTS,
        num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
        bev_range=BEV_RANGE)

    off_out = H * NUM_STEPS * NUM_LEVELS * NUM_POINTS * 2
    aw_out = H * NUM_STEPS * NUM_LEVELS * NUM_POINTS
    expected = (D * off_out + off_out +
                D * aw_out + aw_out +
                D * D + D +
                NUM_STEPS * D * D + D)
    actual = tt_mda.analytical_param_count()
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")
    ok = expected == actual
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 5")

except Exception as e:
    print(f"  [FAIL] TEST 5 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 6: MotionDeformableAttention — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: MotionDeformableAttention — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_mda = PT_MotionDeformableAttention(
        embed_dims=D, num_heads=H,
        num_levels=NUM_LEVELS, num_points=NUM_POINTS,
        num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
        bev_range=BEV_RANGE)

    tt_mda6 = MotionDeformableAttention(
        't6_mda', embed_dims=D, num_heads=H,
        num_levels=NUM_LEVELS, num_points=NUM_POINTS,
        num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
        bev_range=BEV_RANGE)

    copy_mda_weights(pt_mda, tt_mda6)

    num_bev = BEV_H * BEV_W
    bev_np = np.random.randn(num_bev, BS, D).astype(np.float32)
    q6_np = np.random.randn(BS, A, P, D).astype(np.float32)
    T_full = NUM_STEPS
    ref_np = np.random.randn(BS, A, P, T_full, NUM_LEVELS, 2).astype(np.float32) * 5.0

    spatial_shapes_np = np.array([[BEV_H, BEV_W]], dtype=np.int64)
    bbox_results = make_fake_bbox_results(BS, A, seed=42)

    with torch.no_grad():
        pt_out = pt_mda(
            query=torch.from_numpy(q6_np), value=torch.from_numpy(bev_np),
            spatial_shapes=spatial_shapes_np,
            bbox_results=bbox_results,
            reference_trajs=torch.from_numpy(ref_np))

    tt_out = tt_mda6(
        query=F._from_data('t6_q', q6_np, is_const=True),
        value=F._from_data('t6_bev', bev_np, is_const=True),
        spatial_shapes=spatial_shapes_np,
        level_start_index=np.array([0]),
        bbox_results=bbox_results,
        reference_trajs=F._from_data('t6_ref', ref_np, is_const=True))

    ok = compare(pt_out, tt_out, "MotionDeformableAttention", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 6")

except Exception as e:
    print(f"  [FAIL] TEST 6 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 7: MotionDeformableAttention — with query_pos
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: MotionDeformableAttention — with query_pos")
print("=" * 80)

try:
    qpos7_np = np.random.randn(BS, A, P, D).astype(np.float32)
    ref7_np = np.random.randn(BS, A, P, T_full, NUM_LEVELS, 2).astype(np.float32) * 5.0

    with torch.no_grad():
        pt_out7 = pt_mda(
            query=torch.from_numpy(q6_np), value=torch.from_numpy(bev_np),
            query_pos=torch.from_numpy(qpos7_np),
            spatial_shapes=spatial_shapes_np,
            bbox_results=bbox_results,
            reference_trajs=torch.from_numpy(ref7_np))

    tt_out7 = tt_mda6(
        query=F._from_data('t7_q', q6_np, is_const=True),
        value=F._from_data('t7_bev', bev_np, is_const=True),
        query_pos=F._from_data('t7_qpos', qpos7_np, is_const=True),
        spatial_shapes=spatial_shapes_np,
        level_start_index=np.array([0]),
        bbox_results=bbox_results,
        reference_trajs=F._from_data('t7_ref', ref7_np, is_const=True))

    ok = compare(pt_out7, tt_out7, "MDA with query_pos", atol=1e-3)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 7")

except Exception as e:
    print(f"  [FAIL] TEST 7 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 8: MotionTransformerAttentionLayer — construction & param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: MotionTransformerAttentionLayer — construction & param count")
print("=" * 80)

try:
    tt_mtal = MotionTransformerAttentionLayer(
        't8_mtal',
        embed_dims=D,
        attn_cfgs=[dict(
            type='MotionDeformableAttention',
            num_steps=NUM_STEPS, embed_dims=D,
            num_levels=NUM_LEVELS, num_heads=H,
            num_points=NUM_POINTS, sample_index=SAMPLE_INDEX,
            bev_range=BEV_RANGE)],
        ffn_cfgs=dict(
            type='FFN', embed_dims=D, feedforward_channels=D * 2,
            num_fcs=2, ffn_drop=0.0,
            act_cfg=dict(type='ReLU', inplace=True)),
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))

    off_out = H * NUM_STEPS * NUM_LEVELS * NUM_POINTS * 2
    aw_out = H * NUM_STEPS * NUM_LEVELS * NUM_POINTS
    mda_params = (D * off_out + off_out + D * aw_out + aw_out +
                  D * D + D + NUM_STEPS * D * D + D)
    ffn_params = D * (D * 2) + (D * 2) + (D * 2) * D + D
    expected = mda_params + ffn_params
    actual = tt_mtal.analytical_param_count()
    print(f"  MDA params:  {mda_params:,}")
    print(f"  FFN params:  {ffn_params:,}")
    print(f"  Expected:    {expected:,}")
    print(f"  Actual:      {actual:,}")
    ok = expected == actual
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 8")

except Exception as e:
    print(f"  [FAIL] TEST 8 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 9: MotionTransformerAttentionLayer — CustomModeMultiheadAttention cfg
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: MotionTransformerAttentionLayer — CustomModeMultiheadAttention cfg")
print("=" * 80)

try:
    tt_mtal_mha = MotionTransformerAttentionLayer(
        't9_mtal',
        embed_dims=D,
        attn_cfgs=[dict(
            type='CustomModeMultiheadAttention',
            embed_dims=D, num_heads=H)],
        ffn_cfgs=dict(
            type='FFN', embed_dims=D, feedforward_channels=D * 2,
            num_fcs=2, ffn_drop=0.0,
            act_cfg=dict(type='ReLU', inplace=True)),
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))

    mha_params = 4 * (D * D + D)
    ffn_params = D * (D * 2) + (D * 2) + (D * 2) * D + D
    expected = mha_params + ffn_params
    actual = tt_mtal_mha.analytical_param_count()
    print(f"  MHA params:  {mha_params:,}")
    print(f"  FFN params:  {ffn_params:,}")
    print(f"  Expected:    {expected:,}")
    print(f"  Actual:      {actual:,}")
    ok = expected == actual
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 9")

except Exception as e:
    print(f"  [FAIL] TEST 9 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 10: MotionDeformableAttention — weight shapes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: MotionDeformableAttention — weight shapes")
print("=" * 80)

try:
    mda_chk = MotionDeformableAttention(
        't10_mda', embed_dims=D, num_heads=H,
        num_levels=NUM_LEVELS, num_points=NUM_POINTS,
        num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX)

    all_ok = True
    checks = [
        ('sampling_offsets', list(mda_chk.sampling_offsets.param.shape),
         [H * NUM_STEPS * NUM_LEVELS * NUM_POINTS * 2, D]),
        ('attention_weights', list(mda_chk.attention_weights.param.shape),
         [H * NUM_STEPS * NUM_LEVELS * NUM_POINTS, D]),
        ('value_proj', list(mda_chk.value_proj.param.shape), [D, D]),
        ('output_proj_linear', list(mda_chk.output_proj_linear.param.shape),
         [D, NUM_STEPS * D]),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            print(f"  [FAIL] {name}: {actual} != {expected}")
            all_ok = False
        else:
            print(f"  {name} param: {actual} [OK]")

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 10")

except Exception as e:
    print(f"  [FAIL] TEST 10 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 11: MotionDeformableAttention — varying BEV sizes
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: MotionDeformableAttention — varying BEV size")
print("=" * 80)

try:
    all_ok = True
    for bev_h, bev_w in [(30, 30), (50, 100)]:
        pt_mda_v = PT_MotionDeformableAttention(
            embed_dims=D, num_heads=H,
            num_levels=NUM_LEVELS, num_points=NUM_POINTS,
            num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
            bev_range=BEV_RANGE)
        tt_mda_v = MotionDeformableAttention(
            f't11_mda_{bev_h}x{bev_w}', embed_dims=D, num_heads=H,
            num_levels=NUM_LEVELS, num_points=NUM_POINTS,
            num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
            bev_range=BEV_RANGE)
        copy_mda_weights(pt_mda_v, tt_mda_v)

        n_bev = bev_h * bev_w
        bv_np = np.random.randn(n_bev, BS, D).astype(np.float32)
        q_v_np = np.random.randn(BS, A, P, D).astype(np.float32)
        ref_v_np = np.random.randn(BS, A, P, T_full, NUM_LEVELS, 2).astype(np.float32) * 5.0
        ss_np = np.array([[bev_h, bev_w]], dtype=np.int64)
        bb = make_fake_bbox_results(BS, A, seed=99)

        with torch.no_grad():
            pt_o = pt_mda_v(
                query=torch.from_numpy(q_v_np), value=torch.from_numpy(bv_np),
                spatial_shapes=ss_np,
                bbox_results=bb, reference_trajs=torch.from_numpy(ref_v_np))

        tt_o = tt_mda_v(
            query=F._from_data(f't11_q_{bev_h}', q_v_np, is_const=True),
            value=F._from_data(f't11_bev_{bev_h}', bv_np, is_const=True),
            spatial_shapes=ss_np,
            level_start_index=np.array([0]),
            bbox_results=bb,
            reference_trajs=F._from_data(f't11_ref_{bev_h}', ref_v_np, is_const=True))

        sub_ok = compare(pt_o, tt_o, f"MDA BEV={bev_h}x{bev_w}", atol=1e-3)
        if not sub_ok:
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 11")

except Exception as e:
    print(f"  [FAIL] TEST 11 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 12: MotionTransformerAttentionLayer — full forward (MDA + FFN + norms)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: MotionTransformerAttentionLayer — full forward (MDA cross_attn)")
print("=" * 80)

try:
    # Build PyTorch equivalent manually:
    # operation_order = ('cross_attn', 'norm', 'ffn', 'norm')
    # cross_attn = MotionDeformableAttention
    # norm = LayerNorm (neutralized)
    # ffn = Linear(D, 2D) + ReLU + Linear(2D, D) + residual
    pt_mda12 = PT_MotionDeformableAttention(
        embed_dims=D, num_heads=H,
        num_levels=NUM_LEVELS, num_points=NUM_POINTS,
        num_steps=NUM_STEPS, sample_index=SAMPLE_INDEX,
        bev_range=BEV_RANGE)
    pt_norm0 = nn.LayerNorm(D)
    pt_ffn_fc1 = nn.Linear(D, D * 2)
    pt_ffn_relu = nn.ReLU(inplace=True)
    pt_ffn_fc2 = nn.Linear(D * 2, D)
    pt_norm1 = nn.LayerNorm(D)
    # Neutralize PT LayerNorm (TTSim LN has no affine params)
    pt_norm0.weight.data.fill_(1.0); pt_norm0.bias.data.fill_(0.0)
    pt_norm1.weight.data.fill_(1.0); pt_norm1.bias.data.fill_(0.0)

    # TTSim
    tt_mtal12 = MotionTransformerAttentionLayer(
        't12_mtal', embed_dims=D,
        attn_cfgs=[dict(
            type='MotionDeformableAttention',
            num_steps=NUM_STEPS, embed_dims=D,
            num_levels=NUM_LEVELS, num_heads=H,
            num_points=NUM_POINTS, sample_index=SAMPLE_INDEX,
            bev_range=BEV_RANGE)],
        ffn_cfgs=dict(
            type='FFN', embed_dims=D, feedforward_channels=D * 2,
            num_fcs=2, ffn_drop=0.0,
            act_cfg=dict(type='ReLU', inplace=True)),
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))

    # Copy MDA weights
    copy_mda_weights(pt_mda12, tt_mtal12.attentions[0])

    # Copy FFN weights
    tt_ffn12 = tt_mtal12.ffns[0]
    copy_linear_pt_to_tt(pt_ffn_fc1, tt_ffn12.layers[0])
    copy_linear_pt_to_tt(pt_ffn_fc2, tt_ffn12.layers[1])

    # Inputs
    q12_np = np.random.randn(BS, A, P, D).astype(np.float32)
    bev12_np = np.random.randn(BEV_H * BEV_W, BS, D).astype(np.float32)
    ref12_np = np.random.randn(BS, A, P, T_full, NUM_LEVELS, 2).astype(np.float32) * 5.0
    ss12_np = np.array([[BEV_H, BEV_W]], dtype=np.int64)
    bb12 = make_fake_bbox_results(BS, A, seed=77)

    # PyTorch manual forward: cross_attn -> norm -> ffn (with residual) -> norm
    with torch.no_grad():
        q_pt12 = torch.from_numpy(q12_np)
        bev_pt12 = torch.from_numpy(bev12_np)
        ref_pt12 = torch.from_numpy(ref12_np)

        # cross_attn (post-norm: identity=None -> defaults to input query)
        x = pt_mda12(query=q_pt12, value=bev_pt12,
                      spatial_shapes=ss12_np,
                      bbox_results=bb12,
                      reference_trajs=ref_pt12)
        # norm
        x = pt_norm0(x)
        # ffn with residual add
        identity_ffn = x
        x = pt_ffn_fc1(x)
        x = pt_ffn_relu(x)
        x = pt_ffn_fc2(x)
        x = x + identity_ffn  # residual
        # norm
        pt_out12 = pt_norm1(x)

    # TTSim forward
    tt_out12 = tt_mtal12(
        query=F._from_data('t12_q', q12_np, is_const=True),
        value=F._from_data('t12_bev', bev12_np, is_const=True),
        spatial_shapes=ss12_np,
        level_start_index=np.array([0]),
        bbox_results=bb12,
        reference_trajs=F._from_data('t12_ref', ref12_np, is_const=True))

    ok = compare(pt_out12, tt_out12, "MTAL full forward (MDA)", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 12")

except Exception as e:
    print(f"  [FAIL] TEST 12 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# SUMMARY
# ====================================================================

total = passed + failed
print("\n" + "=" * 80)
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 80)
if failed == 0:
    print("[OK] All tests passed!")
else:
    print("[FAIL] Some tests failed")
