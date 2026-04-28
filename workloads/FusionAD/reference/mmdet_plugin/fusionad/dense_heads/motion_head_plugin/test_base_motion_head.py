#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for BaseMotionHead sub-modules (TTSim vs PyTorch).

Validates that TwoLayerMLP, TrackQueryFuser, TrajClsBranch, TrajRegBranch,
and _extract_tracking_centers in TTSim produce identical results to
equivalent PyTorch modules.
"""

import os
import sys
import traceback

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..','..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.base_motion_head import (
    TwoLayerMLP,
    TrackQueryFuser,
    TrajClsBranch,
    TrajRegBranch,
    BaseMotionHead,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import LayerNorm


# ====================================================================
# PyTorch reference modules
# ====================================================================

def build_pt_two_layer_mlp(in_features, hidden_features, out_features):
    """PyTorch equivalent: Linear -> ReLU -> Linear."""
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    )


def build_pt_track_query_fuser(in_features, out_features):
    """PyTorch equivalent: Linear -> LayerNorm -> ReLU."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
    )


def build_pt_traj_cls_branch(embed_dims, num_reg_fcs):
    """PyTorch equivalent: (Linear+LN+ReLU) x N + Linear(D,1)."""
    layers = []
    layers.append(nn.Linear(embed_dims, embed_dims))
    layers.append(nn.LayerNorm(embed_dims))
    layers.append(nn.ReLU(inplace=True))
    for _ in range(num_reg_fcs - 1):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.LayerNorm(embed_dims))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(embed_dims, 1))
    return nn.Sequential(*layers)


def build_pt_traj_reg_branch(embed_dims, num_reg_fcs, out_channels):
    """PyTorch equivalent: (Linear+ReLU) x N + Linear(D, out)."""
    layers = []
    layers.append(nn.Linear(embed_dims, embed_dims))
    layers.append(nn.ReLU())
    for _ in range(num_reg_fcs - 1):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(embed_dims, out_channels))
    return nn.Sequential(*layers)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear_pt_to_tt(pt_linear, tt_linear):
    """Copy weights from PyTorch nn.Linear to TTSim SimNN.Linear."""
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_two_layer_mlp_weights(pt_seq, tt_mlp):
    """Copy PyTorch Sequential(Linear, ReLU, Linear) -> TTSim TwoLayerMLP."""
    copy_linear_pt_to_tt(pt_seq[0], tt_mlp.fc0)
    copy_linear_pt_to_tt(pt_seq[2], tt_mlp.fc1)


def copy_track_query_fuser_weights(pt_seq, tt_fuser):
    """Copy PyTorch Sequential(Linear, LN, ReLU) -> TTSim TrackQueryFuser."""
    copy_linear_pt_to_tt(pt_seq[0], tt_fuser.fc)
    # Neutralize PyTorch LN affine (TTSim LN has no affine)
    pt_seq[1].weight.data.fill_(1.0)
    pt_seq[1].bias.data.fill_(0.0)


def copy_traj_cls_branch_weights(pt_seq, tt_branch, num_reg_fcs):
    """Copy PyTorch Sequential(Linear+LN+ReLU)xN+Linear -> TTSim TrajClsBranch."""
    for i in range(num_reg_fcs):
        pt_fc = pt_seq[i * 3]       # Linear
        pt_ln = pt_seq[i * 3 + 1]   # LayerNorm
        tt_fc = tt_branch.fcs[i]
        copy_linear_pt_to_tt(pt_fc, tt_fc)
        # Neutralize PyTorch LN affine
        pt_ln.weight.data.fill_(1.0)
        pt_ln.bias.data.fill_(0.0)
    # Final Linear
    pt_fc = pt_seq[num_reg_fcs * 3]
    tt_fc = tt_branch.fcs[num_reg_fcs]
    copy_linear_pt_to_tt(pt_fc, tt_fc)


def copy_traj_reg_branch_weights(pt_seq, tt_branch, num_reg_fcs):
    """Copy PyTorch Sequential(Linear+ReLU)xN+Linear -> TTSim TrajRegBranch."""
    for i in range(num_reg_fcs):
        pt_fc = pt_seq[i * 2]
        tt_fc = tt_branch.fcs[i]
        copy_linear_pt_to_tt(pt_fc, tt_fc)
    # Final Linear
    pt_fc = pt_seq[num_reg_fcs * 2]
    tt_fc = tt_branch.fcs[num_reg_fcs]
    copy_linear_pt_to_tt(pt_fc, tt_fc)


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

EMBED_DIMS = 64
NUM_REG_FCS = 2
PREDICT_STEPS = 12
BS = 1
NUM_AGENTS = 5
NUM_MODES = 6
DET_LAYER_NUM = 6

passed = 0
failed = 0


# ====================================================================
# TEST 1: TwoLayerMLP (D -> 2D -> D)
# ====================================================================

print("=" * 80)
print("TEST 1: TwoLayerMLP — PyTorch vs TTSim")
print("=" * 80)

try:
    in_f = EMBED_DIMS
    hid_f = EMBED_DIMS * 2
    out_f = EMBED_DIMS
    pt_mlp = build_pt_two_layer_mlp(in_f, hid_f, out_f)
    tt_mlp = TwoLayerMLP('t1_mlp', in_f, hid_f, out_f)
    copy_two_layer_mlp_weights(pt_mlp, tt_mlp)

    x_np = np.random.randn(BS, NUM_AGENTS, in_f).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t1_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_mlp(x_pt)
    tt_out = tt_mlp(x_tt)

    ok = compare(pt_out, tt_out, "TwoLayerMLP output", atol=1e-5)
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
# TEST 2: TwoLayerMLP with different dims (D -> 2D -> D, 4D input)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: TwoLayerMLP — 4D input (B, A, P, D)")
print("=" * 80)

try:
    in_f = EMBED_DIMS
    hid_f = EMBED_DIMS * 2
    out_f = EMBED_DIMS
    pt_mlp = build_pt_two_layer_mlp(in_f, hid_f, out_f)
    tt_mlp = TwoLayerMLP('t2_mlp', in_f, hid_f, out_f)
    copy_two_layer_mlp_weights(pt_mlp, tt_mlp)

    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, in_f).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t2_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_mlp(x_pt)
    tt_out = tt_mlp(x_tt)

    ok = compare(pt_out, tt_out, "TwoLayerMLP 4D output", atol=1e-5)
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
# TEST 3: TrackQueryFuser (D*det_layer_num -> D)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: TrackQueryFuser — PyTorch vs TTSim (no LN affine)")
print("=" * 80)

try:
    in_f = EMBED_DIMS * DET_LAYER_NUM
    out_f = EMBED_DIMS
    pt_fuser = build_pt_track_query_fuser(in_f, out_f)
    tt_fuser = TrackQueryFuser('t3_fuser', in_f, out_f)
    copy_track_query_fuser_weights(pt_fuser, tt_fuser)

    x_np = np.random.randn(BS, NUM_AGENTS, in_f).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t3_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_fuser(x_pt)
    tt_out = tt_fuser(x_tt)

    ok = compare(pt_out, tt_out, "TrackQueryFuser output", atol=1e-4)
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
# TEST 4: TrajClsBranch (num_reg_fcs=1)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: TrajClsBranch — num_reg_fcs=1")
print("=" * 80)

try:
    nrfc = 1
    pt_cls = build_pt_traj_cls_branch(EMBED_DIMS, nrfc)
    tt_cls = TrajClsBranch('t4_cls', EMBED_DIMS, nrfc)
    copy_traj_cls_branch_weights(pt_cls, tt_cls, nrfc)

    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t4_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_cls(x_pt)
    tt_out = tt_cls(x_tt)

    ok = compare(pt_out, tt_out, "TrajClsBranch output (nrfc=1)", atol=1e-4)
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
# TEST 5: TrajClsBranch (num_reg_fcs=2)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: TrajClsBranch — num_reg_fcs=2")
print("=" * 80)

try:
    nrfc = 2
    pt_cls = build_pt_traj_cls_branch(EMBED_DIMS, nrfc)
    tt_cls = TrajClsBranch('t5_cls', EMBED_DIMS, nrfc)
    copy_traj_cls_branch_weights(pt_cls, tt_cls, nrfc)

    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t5_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_cls(x_pt)
    tt_out = tt_cls(x_tt)

    ok = compare(pt_out, tt_out, "TrajClsBranch output (nrfc=2)", atol=1e-4)
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
# TEST 6: TrajRegBranch (predict_steps*5 output)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: TrajRegBranch — traj_reg (out=predict_steps*5)")
print("=" * 80)

try:
    nrfc = NUM_REG_FCS
    out_ch = PREDICT_STEPS * 5
    pt_reg = build_pt_traj_reg_branch(EMBED_DIMS, nrfc, out_ch)
    tt_reg = TrajRegBranch('t6_reg', EMBED_DIMS, nrfc, out_ch)
    copy_traj_reg_branch_weights(pt_reg, tt_reg, nrfc)

    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t6_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_reg(x_pt)
    tt_out = tt_reg(x_tt)

    ok = compare(pt_out, tt_out, "TrajRegBranch output (reg)", atol=1e-5)
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
# TEST 7: TrajRegBranch (predict_steps*2 output — refine branch)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: TrajRegBranch — traj_refine (out=predict_steps*2)")
print("=" * 80)

try:
    nrfc = NUM_REG_FCS
    out_ch = PREDICT_STEPS * 2
    pt_refine = build_pt_traj_reg_branch(EMBED_DIMS, nrfc, out_ch)
    tt_refine = TrajRegBranch('t7_refine', EMBED_DIMS, nrfc, out_ch)
    copy_traj_reg_branch_weights(pt_refine, tt_refine, nrfc)

    x_np = np.random.randn(BS, NUM_AGENTS, NUM_MODES, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t7_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_refine(x_pt)
    tt_out = tt_refine(x_tt)

    ok = compare(pt_out, tt_out, "TrajRegBranch output (refine)", atol=1e-5)
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
# TEST 8: _extract_tracking_centers — numpy arrays
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: _extract_tracking_centers — numpy arrays")
print("=" * 80)

try:
    class DummyHead(BaseMotionHead):
        def __init__(self):
            super().__init__()
            self.name = 'test_extract'

    head = DummyHead()
    bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # 3 agents
    bboxes = np.array([
        [10.0, 20.0, 0, 0, 0, 0, 0, 0, 0],
        [-30.0, -10.0, 0, 0, 0, 0, 0, 0, 0],
        [0.0, 0.0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    results = [(bboxes, None, None, None, None)]

    centers = head._extract_tracking_centers(results, bev_range)

    # Manually compute expected: (x - x_min) / (x_max - x_min)
    x_norm = (bboxes[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
    y_norm = (bboxes[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
    expected = np.stack([x_norm, y_norm], axis=-1)[None, ...]  # [1, 3, 2]

    ok = True
    print(f"\n  _extract_tracking_centers:")
    print(f"    Output shape: {centers.shape}, expected: {expected.shape}")
    if centers.shape != expected.shape:
        print(f"    [FAIL] Shape mismatch!")
        ok = False
    else:
        diff = np.abs(centers - expected)
        max_diff = diff.max()
        print(f"    Max diff: {max_diff:.6e}")
        if np.allclose(centers, expected, atol=1e-6):
            print(f"    [OK] Match")
        else:
            print(f"    [FAIL] Values mismatch!")
            ok = False

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
# TEST 9: _extract_tracking_centers — torch tensors
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: _extract_tracking_centers — torch tensors via .tensor attr")
print("=" * 80)

try:
    class DummyHead2(BaseMotionHead):
        def __init__(self):
            super().__init__()
            self.name = 'test_extract2'

    class BoxWithTensor:
        """Mimics LiDARInstance3DBoxes with .tensor attribute."""
        def __init__(self, data):
            self.tensor = torch.tensor(data, dtype=torch.float32)

    head = DummyHead2()
    bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    boxes_data = np.array([
        [25.0, -15.0, 0, 0, 0, 0, 0, 0, 0],
        [0.0, 51.2, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    box_obj = BoxWithTensor(boxes_data)
    results = [(box_obj, None, None, None, None)]

    centers = head._extract_tracking_centers(results, bev_range)

    x_norm = (boxes_data[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
    y_norm = (boxes_data[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
    expected = np.stack([x_norm, y_norm], axis=-1)[None, ...]

    ok = True
    print(f"\n  _extract_tracking_centers (torch):")
    print(f"    Output shape: {centers.shape}, expected: {expected.shape}")
    if not np.allclose(centers, expected, atol=1e-6):
        print(f"    [FAIL] Values mismatch")
        ok = False
    else:
        print(f"    [OK] Match")

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
# TEST 10: _extract_tracking_centers — batch_size=2
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: _extract_tracking_centers — batch_size=2")
print("=" * 80)

try:
    class DummyHead3(BaseMotionHead):
        def __init__(self):
            super().__init__()
            self.name = 'test_extract3'

    head = DummyHead3()
    bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    bboxes1 = np.array([[10.0, 20.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    bboxes2 = np.array([[-20.0, -30.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    results = [
        (bboxes1, None, None, None, None),
        (bboxes2, None, None, None, None),
    ]

    centers = head._extract_tracking_centers(results, bev_range)

    ok = True
    print(f"\n  _extract_tracking_centers (batch=2):")
    print(f"    Output shape: {centers.shape}")
    if centers.shape != (2, 1, 2):
        print(f"    [FAIL] Expected shape (2, 1, 2)")
        ok = False
    else:
        # Check each batch independently
        for b, bb in enumerate([bboxes1, bboxes2]):
            x_n = (bb[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_n = (bb[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            exp = np.stack([x_n, y_n], axis=-1)
            if not np.allclose(centers[b], exp, atol=1e-6):
                print(f"    [FAIL] Batch {b} mismatch")
                ok = False
                break
        if ok:
            print(f"    [OK] All batches match")

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
# TEST 11: Analytical param count — TwoLayerMLP
# ====================================================================

print("\n" + "=" * 80)
print("TEST 11: Analytical param count — TwoLayerMLP")
print("=" * 80)

try:
    D = 256
    H = 512
    tt_mlp = TwoLayerMLP('t11', D, H, D)
    pt_mlp = build_pt_two_layer_mlp(D, H, D)

    tt_count = tt_mlp.analytical_param_count()
    pt_count = sum(p.numel() for p in pt_mlp.parameters())

    ok = tt_count == pt_count
    print(f"\n  TwoLayerMLP param count:")
    print(f"    TTSim:   {tt_count}")
    print(f"    PyTorch: {pt_count}")
    print(f"    {'[OK]' if ok else '[FAIL]'} Match: {ok}")

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
# TEST 12: Analytical param count — TrajRegBranch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 12: Analytical param count — TrajRegBranch")
print("=" * 80)

try:
    D = 256
    nrfc = 2
    out_ch = 60
    tt_reg = TrajRegBranch('t12', D, nrfc, out_ch)
    pt_reg = build_pt_traj_reg_branch(D, nrfc, out_ch)

    tt_count = tt_reg.analytical_param_count()
    pt_count = sum(p.numel() for p in pt_reg.parameters())

    ok = tt_count == pt_count
    print(f"\n  TrajRegBranch param count:")
    print(f"    TTSim:   {tt_count}")
    print(f"    PyTorch: {pt_count}")
    print(f"    {'[OK]' if ok else '[FAIL]'} Match: {ok}")

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
