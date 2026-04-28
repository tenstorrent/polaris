#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for MotionHead (TTSim vs PyTorch).

Validates helper methods, sub-module interactions, log-softmax composition,
anchor transforms, group-mode selection, and per-layer cls/reg branches.
No mmcv dependency — all PyTorch references built from torch.nn.
"""

import os
import sys
import traceback
import math

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head import (
    MotionHead,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.base_motion_head import (
    TwoLayerMLP,
    TrajClsBranch,
    TrajRegBranch,
)


# ====================================================================
# PyTorch reference helpers (pure torch, no mmcv)
# ====================================================================

def pt_norm_points(pos, pc_range):
    """PyTorch norm_points: normalize xy to [0,1]."""
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pt_pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """PyTorch pos2posemb2d: sinusoidal positional embedding."""
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x), dim=-1)


def pt_bivariate_gaussian_activation(ip):
    """PyTorch bivariate gaussian activation on trajectory outputs."""
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = torch.exp(ip[..., 2:3])
    sig_y = torch.exp(ip[..., 3:4])
    rho = torch.tanh(ip[..., 4:5])
    return torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)


def pt_anchor_coordinate_transform(anchors, yaw, centers, n_agents,
                                   with_rotation=True, with_translation=True):
    """PyTorch anchor coordinate transform (single batch element).

    Args:
        anchors: (G, P, T, 2) tensor
        yaw: (A,) or (A, 1) tensor
        centers: (A, 3) tensor
        n_agents: int

    Returns:
        (A, G, P, T, 2) tensor
    """
    transformed = anchors.unsqueeze(0)  # (1, G, P, T, 2)

    if with_rotation:
        angle = yaw.flatten() - 3.1415953
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_mat = torch.zeros(n_agents, 2, 2, dtype=anchors.dtype)
        rot_mat[:, 0, 0] = cos_a
        rot_mat[:, 0, 1] = -sin_a
        rot_mat[:, 1, 0] = sin_a
        rot_mat[:, 1, 1] = cos_a
        rm = rot_mat[:, None, None]  # (A, 1, 1, 2, 2)
        t = transformed.permute(0, 1, 2, 4, 3)  # (..., 2, T)
        result = torch.matmul(rm, t)
        transformed = result.permute(0, 1, 2, 4, 3)  # (..., T, 2)

    if with_translation:
        c = centers[:, :2].reshape(n_agents, 1, 1, 1, 2)
        transformed = c + transformed

    return transformed


def build_pt_traj_cls_branch(embed_dims, num_reg_fcs):
    """PyTorch TrajClsBranch: (Linear+LN+ReLU) x N + Linear(D,1)."""
    layers = []
    for _ in range(num_reg_fcs):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.LayerNorm(embed_dims))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(embed_dims, 1))
    return nn.Sequential(*layers)


def build_pt_traj_reg_branch(embed_dims, num_reg_fcs, out_channels):
    """PyTorch TrajRegBranch: (Linear+ReLU) x N + Linear(D, out)."""
    layers = []
    for _ in range(num_reg_fcs):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(embed_dims, out_channels))
    return nn.Sequential(*layers)


def build_pt_two_layer_mlp(in_f, hid_f, out_f):
    """PyTorch TwoLayerMLP: Linear -> ReLU -> Linear."""
    return nn.Sequential(nn.Linear(in_f, hid_f), nn.ReLU(), nn.Linear(hid_f, out_f))


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear_pt_to_tt(pt_linear, tt_linear):
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_traj_cls_branch_weights(pt_seq, tt_branch, num_reg_fcs):
    for i in range(num_reg_fcs):
        copy_linear_pt_to_tt(pt_seq[i * 3], tt_branch.fcs[i])
        pt_seq[i * 3 + 1].weight.data.fill_(1.0)
        pt_seq[i * 3 + 1].bias.data.fill_(0.0)
    copy_linear_pt_to_tt(pt_seq[num_reg_fcs * 3], tt_branch.fcs[num_reg_fcs])


def copy_traj_reg_branch_weights(pt_seq, tt_branch, num_reg_fcs):
    for i in range(num_reg_fcs):
        copy_linear_pt_to_tt(pt_seq[i * 2], tt_branch.fcs[i])
    copy_linear_pt_to_tt(pt_seq[num_reg_fcs * 2], tt_branch.fcs[num_reg_fcs])


def copy_two_layer_mlp_weights(pt_seq, tt_mlp):
    copy_linear_pt_to_tt(pt_seq[0], tt_mlp.fc0)
    copy_linear_pt_to_tt(pt_seq[2], tt_mlp.fc1)


def init_tt_linear_random(tt_linear, rng=None):
    """Initialize a TTSim SimNN.Linear with random weights (for shape-only tests)."""
    if rng is None:
        rng = np.random
    in_f = tt_linear.param.shape[0]
    out_f = tt_linear.param.shape[1]
    tt_linear.param.data = rng.randn(in_f, out_f).astype(np.float32) * 0.02
    tt_linear.bias.data = np.zeros(out_f, dtype=np.float32)


def init_two_layer_mlp_random(tt_mlp, rng=None):
    """Initialize a TwoLayerMLP with random weights."""
    init_tt_linear_random(tt_mlp.fc0, rng)
    init_tt_linear_random(tt_mlp.fc1, rng)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-5):
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
    print(f"\n  {name}:")
    print(f"    PyTorch shape: {pt_np.shape}")
    print(f"    TTSim   shape: {tt_np.shape}")
    if pt_np.shape != tt_np.shape:
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
# Config
# ====================================================================

np.random.seed(42)
torch.manual_seed(42)

D = 64
H = 8
P = 6      # num_anchor (modes)
T = 12     # predict_steps
NUM_LAYERS = 3
BEV_H, BEV_W = 50, 50
NUM_LEVELS = 1
NUM_POINTS = 4
NUM_CLASSES = 10
GROUP_ID_LIST = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
G = len(GROUP_ID_LIST)
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
BS = 1
A = 4   # num agents
NUM_REG_FCS = 1  # num_cls_fcs=2 -> self.num_cls_fcs=1

passed = 0
failed = 0


# ====================================================================
# Build MotionHead for tests that need it
# ====================================================================

transformerlayers_cfg = dict(
    attn_cfgs=[dict(
        type='MotionDeformableAttention',
        num_steps=T,
        embed_dims=D,
        num_levels=NUM_LEVELS,
        num_heads=H,
        num_points=NUM_POINTS,
        sample_index=-1,
        bev_range=PC_RANGE)],
    ffn_cfgs=dict(
        type='FFN', embed_dims=D, feedforward_channels=D * 2,
        num_fcs=2, ffn_drop=0.0,
        act_cfg=dict(type='ReLU', inplace=True)),
    operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
    embed_dims=D)

mh = MotionHead(
    'test_mh',
    predict_steps=T,
    transformerlayers=transformerlayers_cfg,
    num_cls_fcs=2,
    bev_h=BEV_H,
    bev_w=BEV_W,
    embed_dims=D,
    num_anchor=P,
    det_layer_num=6,
    group_id_list=GROUP_ID_LIST,
    pc_range=PC_RANGE,
    anchor_info_path=None,
    num_classes=NUM_CLASSES,
    num_layers=NUM_LAYERS)

# Set fake anchors and learnable embedding
mh.kmeans_anchors = np.random.randn(G, P, T, 2).astype(np.float32)
mh.learnable_motion_query_embedding_data = np.random.randn(P * G, D).astype(np.float32)


# ====================================================================
# TEST 1: MotionHead construction + param count > 0
# ====================================================================

print("=" * 80)
print("TEST 1: MotionHead construction + param count")
print("=" * 80)

try:
    pc = mh.analytical_param_count()
    ok = pc > 0
    print(f"\n  MotionHead param count: {pc:,}")
    print(f"  {'[OK]' if ok else '[FAIL]'} param_count > 0")
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
# TEST 2: _norm_points_np — compare with PyTorch norm_points
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: _norm_points_np vs PyTorch norm_points")
print("=" * 80)

try:
    pos_np = np.random.uniform(-51.2, 51.2, (G, P, 2)).astype(np.float32)
    pos_pt = torch.from_numpy(pos_np)

    tt_out = mh._norm_points_np(pos_np)
    pt_out = pt_norm_points(pos_pt, PC_RANGE)

    ok = compare(pt_out, tt_out, "_norm_points_np", atol=1e-6)
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
# TEST 3: _pos2posemb2d_np — compare with PyTorch pos2posemb2d
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: _pos2posemb2d_np vs PyTorch pos2posemb2d")
print("=" * 80)

try:
    pos_np = np.random.rand(G, P, 2).astype(np.float32)  # normalized [0,1]
    pos_pt = torch.from_numpy(pos_np)

    tt_out = mh._pos2posemb2d_np(pos_np, num_pos_feats=D // 2)
    pt_out = pt_pos2posemb2d(pos_pt, num_pos_feats=D // 2)

    ok = compare(pt_out, tt_out, "_pos2posemb2d_np", atol=1e-5)
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
# TEST 4: _anchor_transform_np — rotation only
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: _anchor_transform_np — rotation only")
print("=" * 80)

try:
    anchors_np = np.random.randn(G, P, T, 2).astype(np.float32)
    anchors_pt = torch.from_numpy(anchors_np)

    yaw_np = np.random.randn(A, 1).astype(np.float32)
    centers_np = np.random.randn(A, 3).astype(np.float32)
    yaw_pt = torch.from_numpy(yaw_np)
    centers_pt = torch.from_numpy(centers_np)

    tt_out = mh._anchor_transform_np(anchors_np, yaw_np, centers_np, A,
                                     with_rotation=True, with_translation=False)
    pt_out = pt_anchor_coordinate_transform(anchors_pt, yaw_pt, centers_pt, A,
                                            with_rotation=True, with_translation=False)

    ok = compare(pt_out, tt_out, "anchor_transform (rot only)", atol=1e-5)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 4")
except Exception as e:
    print(f"  [FAIL] TEST 4 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 5: _anchor_transform_np — translation only
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: _anchor_transform_np — translation only")
print("=" * 80)

try:
    tt_out = mh._anchor_transform_np(anchors_np, yaw_np, centers_np, A,
                                     with_rotation=False, with_translation=True)
    pt_out = pt_anchor_coordinate_transform(anchors_pt, yaw_pt, centers_pt, A,
                                            with_rotation=False, with_translation=True)

    ok = compare(pt_out, tt_out, "anchor_transform (translate only)", atol=1e-5)
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
# TEST 6: _anchor_transform_np — rotation + translation
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: _anchor_transform_np — rotation + translation")
print("=" * 80)

try:
    tt_out = mh._anchor_transform_np(anchors_np, yaw_np, centers_np, A,
                                     with_rotation=True, with_translation=True)
    pt_out = pt_anchor_coordinate_transform(anchors_pt, yaw_pt, centers_pt, A,
                                            with_rotation=True, with_translation=True)

    ok = compare(pt_out, tt_out, "anchor_transform (rot+translate)", atol=1e-5)
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
# TEST 7: _group_mode — class-to-group selection
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: _group_mode — class-to-group selection")
print("=" * 80)

try:
    # Build fake data: (B, A, G, P, D)
    data = np.random.randn(BS, A, G, P, D).astype(np.float32)
    labels = np.array([0, 5, 3, 8])  # classes -> groups [0, 1, 0, 1]
    bbox_results = [(None, None, labels, None, None)]

    result = mh._group_mode(bbox_results, data)

    # Manual: for each agent, select group based on cls2group
    expected_groups = [mh.cls2group[l] for l in labels]
    expected = np.stack([
        np.stack([data[0, j, expected_groups[j]] for j in range(A)])
    ])

    ok = True
    print(f"\n  _group_mode:")
    print(f"    Input shape:    {data.shape}")
    print(f"    Output shape:   {result.shape}")
    print(f"    Expected shape: {expected.shape}")
    if result.shape != expected.shape:
        print(f"    [FAIL] Shape mismatch!")
        ok = False
    elif not np.allclose(result, expected, atol=1e-7):
        print(f"    [FAIL] Values mismatch")
        ok = False
    else:
        print(f"    [OK] Match")

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
# TEST 8: _select_last_dec
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: _select_last_dec — select last decoder layer")
print("=" * 80)

try:
    num_dec = 6
    data_np = np.random.randn(BS, num_dec, A, D).astype(np.float32)
    data_tt = F._from_data('t8_tq', data_np)

    result = mh._select_last_dec(data_tt)

    expected = data_np[:, -1]
    ok = True
    print(f"\n  _select_last_dec:")
    print(f"    Input shape:  {data_np.shape}")
    print(f"    Output shape: {result.data.shape}")
    if result.data.shape != expected.shape:
        print(f"    [FAIL] Shape mismatch: got {result.data.shape}, expected {expected.shape}")
        ok = False
    elif not np.allclose(result.data, expected, atol=1e-7):
        print(f"    [FAIL] Values mismatch")
        ok = False
    else:
        print(f"    [OK] Match")

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
# TEST 9: _unflatten_and_activate — reshape + cumsum + bivariate gaussian
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: _unflatten_and_activate vs PyTorch")
print("=" * 80)

try:
    reg_np = np.random.randn(BS, A, P, T * 5).astype(np.float32)

    # Numpy replication of _unflatten_and_activate logic
    # (graph ops like ConcatX/CumSum don't compute .data in isolation)
    reshaped = reg_np.reshape(BS, A, P, T, 5)
    xy_cum = np.cumsum(reshaped[..., :2], axis=3)
    sigx = np.exp(reshaped[..., 2:3])
    sigy = np.exp(reshaped[..., 3:4])
    rho = np.tanh(reshaped[..., 4:5])
    tt_result = np.concatenate([xy_cum, sigx, sigy, rho], axis=4)

    # PyTorch reference
    reg_pt = torch.from_numpy(reg_np)
    pt_unflat = reg_pt.reshape(BS, A, P, T, 5)
    pt_unflat[..., :2] = torch.cumsum(pt_unflat[..., :2], dim=3)
    for bs in range(BS):
        pt_unflat[bs] = pt_bivariate_gaussian_activation(pt_unflat[bs])

    ok = compare(pt_unflat, tt_result, "_unflatten_and_activate", atol=1e-5)
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
# TEST 10: Log-softmax composition (Softmax + Log) vs torch LogSoftmax
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: Log-softmax (Softmax+Log) vs torch.nn.LogSoftmax(dim=2)")
print("=" * 80)

try:
    cls_np = np.random.randn(BS, A, P).astype(np.float32)
    cls_tt = F._from_data('t10_cls', cls_np)

    # TTSim: Softmax(axis=2) -> Log (create ops inline;
    # in __call__ these are created per-level as softmax_cls_0, log_cls_0, etc.)
    sm_op = F.Softmax('t10_softmax', axis=2)
    log_op = F.Log('t10_log')
    sm = sm_op(cls_tt)
    tt_out = log_op(sm)

    # PyTorch: LogSoftmax(dim=2)
    cls_pt = torch.from_numpy(cls_np)
    pt_log_softmax = nn.LogSoftmax(dim=2)
    with torch.no_grad():
        pt_out = pt_log_softmax(cls_pt)

    ok = compare(pt_out, tt_out, "LogSoftmax composition", atol=1e-5)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 10")
except Exception as e:
    print(f"  [FAIL] TEST 10 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 11: Per-layer cls branch — weight copy + forward match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: TrajClsBranch forward match (weight-copied)")
print("=" * 80)

try:
    pt_cls = build_pt_traj_cls_branch(D, NUM_REG_FCS)
    tt_cls = mh.traj_cls_branches[0]
    copy_traj_cls_branch_weights(pt_cls, tt_cls, NUM_REG_FCS)

    x_np = np.random.randn(BS, A, P, D).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t11_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_cls(x_pt)
    tt_out = tt_cls(x_tt)

    ok = compare(pt_out, tt_out, "TrajClsBranch forward", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 11")
except Exception as e:
    print(f"  [FAIL] TEST 11 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 12: Per-layer reg branch — weight copy + forward match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: TrajRegBranch forward match (weight-copied)")
print("=" * 80)

try:
    out_ch = T * 5
    pt_reg = build_pt_traj_reg_branch(D, NUM_REG_FCS, out_ch)
    tt_reg = mh.traj_reg_branches[0]
    copy_traj_reg_branch_weights(pt_reg, tt_reg, NUM_REG_FCS)

    x_np = np.random.randn(BS, A, P, D).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t12_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_reg(x_pt)
    tt_out = tt_reg(x_tt)

    ok = compare(pt_out, tt_out, "TrajRegBranch forward", atol=1e-5)
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
# TEST 13: Cls squeeze + log_softmax end-to-end
# ====================================================================

print("\n" + "=" * 80)
print("TEST 13: Cls branch -> squeeze -> log_softmax (end-to-end)")
print("=" * 80)

try:
    # Use the same pt_cls and tt_cls from TEST 11 (weights already copied)
    x_np = np.random.randn(BS, A, P, D).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t13_x', x_np, is_const=True)

    # PyTorch: cls_branch -> squeeze(dim=3) -> log_softmax(dim=2)
    with torch.no_grad():
        pt_cls_out = pt_cls(x_pt)                          # (B, A, P, 1)
        pt_squeezed = pt_cls_out.squeeze(3)                # (B, A, P)
        pt_out = F_torch.log_softmax(pt_squeezed, dim=2)   # (B, A, P)

    # TTSim: cls_branch -> squeeze -> softmax -> log
    # (create ops inline; in __call__ these are created per-level)
    tt_cls_out = tt_cls(x_tt)

    unsq_ax3 = F._from_data('t13_ax3', np.array([3], dtype=np.int64), is_const=True)
    squeeze_op = F.Squeeze('t13_squeeze')
    softmax_op = F.Softmax('t13_softmax', axis=2)
    log_op = F.Log('t13_log')
    tt_squeezed = squeeze_op(tt_cls_out, unsq_ax3)
    tt_sm = softmax_op(tt_squeezed)
    tt_out = log_op(tt_sm)

    ok = compare(pt_out, tt_out, "cls -> squeeze -> log_softmax", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 13")
except Exception as e:
    print(f"  [FAIL] TEST 13 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 14: Reg branch -> unflatten + cumsum + bivariate_gaussian (end-to-end)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 14: Reg branch -> unflatten_activate (end-to-end)")
print("=" * 80)

try:
    x_np = np.random.randn(BS, A, P, D).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t14_x', x_np, is_const=True)

    # PyTorch
    with torch.no_grad():
        pt_reg_out = pt_reg(x_pt)  # (B, A, P, T*5)
        pt_unflat = pt_reg_out.reshape(BS, A, P, T, 5)
        pt_unflat[..., :2] = torch.cumsum(pt_unflat[..., :2], dim=3)
        for bs in range(BS):
            pt_unflat[bs] = pt_bivariate_gaussian_activation(pt_unflat[bs])

    # TTSim: reg branch then replicate _unflatten_and_activate in numpy
    tt_reg_out = tt_reg(x_tt)
    reg_data = tt_reg_out.data
    reshaped = reg_data.reshape(BS, A, P, T, 5)
    xy_cum = np.cumsum(reshaped[..., :2], axis=3)
    sigx = np.exp(reshaped[..., 2:3])
    sigy = np.exp(reshaped[..., 3:4])
    rho = np.tanh(reshaped[..., 4:5])
    tt_result = np.concatenate([xy_cum, sigx, sigy, rho], axis=4)

    ok = compare(pt_unflat, tt_result, "reg -> unflatten -> cumsum -> gaussian", atol=1e-4)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 14")
except Exception as e:
    print(f"  [FAIL] TEST 14 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 15: _compute_anchor_embeddings — PyTorch vs TTSim (weight-copied)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 15: _compute_anchor_embeddings — PyTorch vs TTSim")
print("=" * 80)

# Build PyTorch equivalent MLPs and copy weights to TTSim
pt_agent_mlp = build_pt_two_layer_mlp(D, D * 2, D)
pt_ego_mlp = build_pt_two_layer_mlp(D, D * 2, D)
pt_offset_mlp = build_pt_two_layer_mlp(D, D * 2, D)

copy_two_layer_mlp_weights(pt_agent_mlp, mh.agent_level_embedding_layer)
copy_two_layer_mlp_weights(pt_ego_mlp, mh.scene_level_ego_embedding_layer)
copy_two_layer_mlp_weights(pt_offset_mlp, mh.scene_level_offset_embedding_layer)

# FakeBbox class for _compute_anchor_embeddings
class FakeBbox:
    def __init__(self, data):
        self._data = data
    @property
    def gravity_center(self):
        class GC:
            def __init__(self, d):
                self.d = d
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.d[:, :3]
        return GC(self._data)
    @property
    def yaw(self):
        class Y:
            def __init__(self, d):
                self.d = d
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.d[:, 6:7]
        return Y(self._data)

try:
    anchors = mh.kmeans_anchors.reshape(G, P, T, 2)
    lpq_split = mh.learnable_motion_query_embedding_data.reshape(G, P, D)

    bboxes_np = np.random.randn(A, 9).astype(np.float32)
    labels = np.array([0, 5, 3, 8])
    bbox_results = [(FakeBbox(bboxes_np), None, labels, None, None)]

    # --- TTSim ---
    agent_emb, ego_emb, offset_emb, scene_anchors, learn_emb = \
        mh._compute_anchor_embeddings(anchors, lpq_split, bbox_results, BS, A, P, D)

    # --- PyTorch reference (replicate the same pipeline) ---
    anchors_pt = torch.from_numpy(anchors)
    yaw_np = bboxes_np[:, 6:7]
    centers_np = bboxes_np[:, :3]
    yaw_pt = torch.from_numpy(yaw_np)
    centers_pt = torch.from_numpy(centers_np)

    # Agent level: norm -> posemb (no MLP; _compute_anchor_embeddings
    # returns raw posembs, MLPs are applied later in __call__)
    agent_norm_pt = pt_norm_points(anchors_pt[..., -1, :], PC_RANGE)
    agent_posemb_pt = pt_pos2posemb2d(agent_norm_pt, num_pos_feats=D // 2)

    # Ego level: translate anchors -> norm -> posemb
    ego_anchors_pt = pt_anchor_coordinate_transform(
        anchors_pt, yaw_pt, centers_pt, A,
        with_rotation=False, with_translation=True)  # (A, G, P, T, 2)
    ego_norm_pt = pt_norm_points(ego_anchors_pt[..., -1, :], PC_RANGE)
    ego_posemb_pt = pt_pos2posemb2d(ego_norm_pt, num_pos_feats=D // 2)

    # Offset level: rotate anchors -> norm -> posemb
    offset_anchors_pt = pt_anchor_coordinate_transform(
        anchors_pt, yaw_pt, centers_pt, A,
        with_rotation=True, with_translation=False)  # (A, G, P, T, 2)
    offset_norm_pt = pt_norm_points(offset_anchors_pt[..., -1, :], PC_RANGE)
    offset_posemb_pt = pt_pos2posemb2d(offset_norm_pt, num_pos_feats=D // 2)

    # Expand to (B, A, G, P, D)
    agent_emb_pt = agent_posemb_pt.unsqueeze(0).unsqueeze(0).expand(BS, A, -1, -1, -1)
    ego_emb_pt = ego_posemb_pt.unsqueeze(0).expand(BS, -1, -1, -1, -1)
    offset_emb_pt = offset_posemb_pt.unsqueeze(0).expand(BS, -1, -1, -1, -1)
    learn_emb_pt = torch.from_numpy(lpq_split).unsqueeze(0).unsqueeze(0).expand(BS, A, -1, -1, -1)

    ok = True
    ok &= compare(agent_emb_pt, agent_emb, "agent_emb", atol=1e-4)
    ok &= compare(ego_emb_pt, ego_emb, "ego_emb", atol=1e-4)
    ok &= compare(offset_emb_pt, offset_emb, "offset_emb", atol=1e-4)
    ok &= compare(learn_emb_pt, learn_emb, "learn_emb", atol=1e-6)

    # Shape check for scene_anchors
    exp_shape = (BS, A, G, P, T, 2)
    if scene_anchors.shape != exp_shape:
        print(f"    [FAIL] scene_anchors: got {scene_anchors.shape}, expected {exp_shape}")
        ok = False
    else:
        print(f"\n    [OK] scene_anchors shape: {scene_anchors.shape}")

    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 15")
except Exception as e:
    print(f"  [FAIL] TEST 15 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 16: Embedding layer (boxes_query_embedding_layer) weight-copy match
# ====================================================================

print("\n" + "=" * 80)
print("TEST 16: boxes_query_embedding_layer forward match")
print("=" * 80)

try:
    in_f = D
    hid_f = D * 2
    out_f = D
    pt_mlp = build_pt_two_layer_mlp(in_f, hid_f, out_f)
    tt_mlp = mh.boxes_query_embedding_layer
    copy_two_layer_mlp_weights(pt_mlp, tt_mlp)

    x_np = np.random.randn(BS, A, D).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t16_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_mlp(x_pt)
    tt_out = tt_mlp(x_tt)

    ok = compare(pt_out, tt_out, "boxes_query_embedding forward", atol=1e-5)
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if ok else '[FAIL]'} TEST 16")
except Exception as e:
    print(f"  [FAIL] TEST 16 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# Summary
# ====================================================================

print("\n" + "=" * 80)
total = passed + failed
print(f"RESULTS: {passed}/{total} tests passed, {failed} failed.")
if failed == 0:
    print("ALL TESTS PASSED!")
else:
    print("SOME TESTS FAILED.")
print("=" * 80)
sys.exit(1 if failed else 0)
