#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for BEVFormerTrackHead branches (TTSim vs PyTorch).

Validates that ClsBranch / RegBranch in TTSim produce identical results
to the equivalent PyTorch nn.Sequential branches, and that the
post-processing (reference-point refinement + pc_range scaling) matches.
"""

import os
import sys
import traceback
import copy

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.track_head import (
    ClsBranch,
    RegBranch,
    BEVFormerTrackHead,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.decoder import (
    inverse_sigmoid_np,
)


# ====================================================================
# PyTorch reference branches (exact architecture from track_head.py)
# ====================================================================

def build_pt_cls_branch(embed_dims, num_reg_fcs, cls_out_channels):
    layers = []
    for _ in range(num_reg_fcs):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.LayerNorm(embed_dims))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(embed_dims, cls_out_channels))
    return nn.Sequential(*layers)


def build_pt_reg_branch(embed_dims, num_reg_fcs, out_channels):
    layers = []
    for _ in range(num_reg_fcs):
        layers.append(nn.Linear(embed_dims, embed_dims))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(embed_dims, out_channels))
    return nn.Sequential(*layers)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_cls_weights_pt_to_tt(pt_branch, tt_branch, num_reg_fcs):
    """Copy cls branch weights: PyTorch Sequential -> TTSim ClsBranch (no transpose)."""
    for i in range(num_reg_fcs):
        pt_fc = pt_branch[i * 3]      # nn.Linear
        tt_fc = tt_branch.fcs[i]
        tt_fc.param.data = pt_fc.weight.data.detach().numpy().astype(np.float32)
        tt_fc.bias.data = pt_fc.bias.data.detach().numpy().astype(np.float32)
    # Final FC
    pt_fc = pt_branch[num_reg_fcs * 3]
    tt_fc = tt_branch.fcs[num_reg_fcs]
    tt_fc.param.data = pt_fc.weight.data.detach().numpy().astype(np.float32)
    tt_fc.bias.data = pt_fc.bias.data.detach().numpy().astype(np.float32)


def copy_reg_weights_pt_to_tt(pt_branch, tt_branch, num_reg_fcs):
    """Copy reg branch weights: PyTorch Sequential -> TTSim RegBranch (no transpose)."""
    for i in range(num_reg_fcs):
        pt_fc = pt_branch[i * 2]
        tt_fc = tt_branch.fcs[i]
        tt_fc.param.data = pt_fc.weight.data.detach().numpy().astype(np.float32)
        tt_fc.bias.data = pt_fc.bias.data.detach().numpy().astype(np.float32)
    # Final FC
    pt_fc = pt_branch[num_reg_fcs * 2]
    tt_fc = tt_branch.fcs[num_reg_fcs]
    tt_fc.param.data = pt_fc.weight.data.detach().numpy().astype(np.float32)
    tt_fc.bias.data = pt_fc.bias.data.detach().numpy().astype(np.float32)


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
CLS_OUT_CHANNELS = 10
CODE_SIZE = 10
PAST_STEPS = 4
FUT_STEPS = 4
TRAJ_OUT = (PAST_STEPS + FUT_STEPS) * 2
BS = 2
NQ = 8
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

passed = 0
failed = 0


# ====================================================================
# TEST 1: ClsBranch (no LayerNorm affine — expected small diff)
# ====================================================================

print("=" * 80)
print("TEST 1: ClsBranch — PyTorch vs TTSim (no LN affine)")
print("=" * 80)

try:
    pt_cls = build_pt_cls_branch(EMBED_DIMS, NUM_REG_FCS, CLS_OUT_CHANNELS)
    # Disable LN affine to match TTSim (TTSim LN has no affine params)
    for i in range(NUM_REG_FCS):
        ln = pt_cls[i * 3 + 1]
        ln.weight.data.fill_(1.0)
        ln.bias.data.fill_(0.0)

    tt_cls = ClsBranch('t1_cls', EMBED_DIMS, NUM_REG_FCS, CLS_OUT_CHANNELS)
    copy_cls_weights_pt_to_tt(pt_cls, tt_cls, NUM_REG_FCS)

    x_np = np.random.randn(BS, NQ, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t1_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_cls(x_pt)

    tt_out = tt_cls(x_tt)

    ok = compare(pt_out, tt_out, "ClsBranch output", atol=1e-4)
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
# TEST 2: RegBranch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: RegBranch — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_reg = build_pt_reg_branch(EMBED_DIMS, NUM_REG_FCS, CODE_SIZE)
    tt_reg = RegBranch('t2_reg', EMBED_DIMS, NUM_REG_FCS, CODE_SIZE)
    copy_reg_weights_pt_to_tt(pt_reg, tt_reg, NUM_REG_FCS)

    x_np = np.random.randn(BS, NQ, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t2_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_reg(x_pt)

    tt_out = tt_reg(x_tt)

    ok = compare(pt_out, tt_out, "RegBranch output", atol=1e-5)
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
# TEST 3: TrajRegBranch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: TrajRegBranch — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_traj = build_pt_reg_branch(EMBED_DIMS, NUM_REG_FCS, TRAJ_OUT)
    tt_traj = RegBranch('t3_traj', EMBED_DIMS, NUM_REG_FCS, TRAJ_OUT)
    copy_reg_weights_pt_to_tt(pt_traj, tt_traj, NUM_REG_FCS)

    x_np = np.random.randn(BS, NQ, EMBED_DIMS).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t3_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_traj(x_pt)

    tt_out = tt_traj(x_tt)

    ok = compare(pt_out, tt_out, "TrajRegBranch output", atol=1e-5)
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
# TEST 4: Reference-point refinement (post-processing math)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: Reference-point refinement — PyTorch vs TTSim numpy")
print("=" * 80)

try:
    def inverse_sigmoid_pt(x, eps=1e-5):
        x = torch.clamp(x, 0, 1)
        x1 = torch.clamp(x, min=eps)
        x2 = torch.clamp(1 - x, min=eps)
        return torch.log(x1 / x2)

    # Random reg output and reference points
    reg_np = np.random.randn(BS, NQ, CODE_SIZE).astype(np.float32) * 0.1
    ref_np = np.random.rand(BS, NQ, 3).astype(np.float32) * 0.8 + 0.1

    # --- PyTorch path ---
    reg_pt = torch.from_numpy(reg_np.copy())
    ref_pt = torch.from_numpy(ref_np.copy())

    reference_pt = inverse_sigmoid_pt(ref_pt)
    tmp_pt = reg_pt.clone()
    tmp_pt[..., 0:2] += reference_pt[..., 0:2]
    tmp_pt[..., 0:2] = tmp_pt[..., 0:2].sigmoid()
    tmp_pt[..., 4:5] += reference_pt[..., 2:3]
    tmp_pt[..., 4:5] = tmp_pt[..., 4:5].sigmoid()

    last_ref_pt = torch.cat([tmp_pt[..., 0:2], tmp_pt[..., 4:5]], dim=-1)

    tmp_pt[..., 0:1] = tmp_pt[..., 0:1] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    tmp_pt[..., 1:2] = tmp_pt[..., 1:2] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    tmp_pt[..., 4:5] = tmp_pt[..., 4:5] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]

    last_ref_logits_pt = inverse_sigmoid_pt(last_ref_pt)

    # --- TTSim numpy path (mirrors get_detections) ---
    tmp_np = reg_np.copy()
    ref_logits_np = inverse_sigmoid_np(ref_np)

    tmp_np[..., 0:2] += ref_logits_np[..., 0:2]
    tmp_np[..., 0:2] = 1.0 / (1.0 + np.exp(
        -tmp_np[..., 0:2].astype(np.float64))).astype(np.float32)
    tmp_np[..., 4:5] += ref_logits_np[..., 2:3]
    tmp_np[..., 4:5] = 1.0 / (1.0 + np.exp(
        -tmp_np[..., 4:5].astype(np.float64))).astype(np.float32)

    last_ref_np = np.concatenate(
        [tmp_np[..., 0:2], tmp_np[..., 4:5]], axis=-1)

    tmp_np[..., 0:1] = tmp_np[..., 0:1] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    tmp_np[..., 1:2] = tmp_np[..., 1:2] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    tmp_np[..., 4:5] = tmp_np[..., 4:5] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]

    last_ref_logits_np = inverse_sigmoid_np(last_ref_np)

    ok1 = compare(tmp_pt, tmp_np, "Coord output after pc_range scaling", atol=1e-4)
    ok2 = compare(last_ref_logits_pt, last_ref_logits_np,
                  "last_ref_points (inverse sigmoid)", atol=1e-4)

    ok = ok1 and ok2
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
# TEST 5: BEVFormerTrackHead construction — shape / param checks
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: BEVFormerTrackHead construction — with_box_refine=True")
print("=" * 80)

try:
    NUM_PRED = 3
    head = BEVFormerTrackHead(
        name='t5_head',
        embed_dims=EMBED_DIMS,
        num_reg_fcs=NUM_REG_FCS,
        cls_out_channels=CLS_OUT_CHANNELS,
        code_size=CODE_SIZE,
        bev_h=10, bev_w=10,
        past_steps=PAST_STEPS,
        fut_steps=FUT_STEPS,
        pc_range=PC_RANGE,
        num_pred=NUM_PRED,
        num_query=NQ,
        with_box_refine=True,
        transformer=None,
    )
    ok = True

    # Check branch count
    assert len(head.cls_branches) == NUM_PRED, \
        f"Expected {NUM_PRED} cls branches, got {len(head.cls_branches)}"
    assert len(head.reg_branches) == NUM_PRED, \
        f"Expected {NUM_PRED} reg branches, got {len(head.reg_branches)}"
    assert len(head.traj_branches) == NUM_PRED, \
        f"Expected {NUM_PRED} traj branches, got {len(head.traj_branches)}"

    # Check pc_range constants
    assert np.isclose(head.range_x_const.data[0], PC_RANGE[3] - PC_RANGE[0])
    assert np.isclose(head.offset_x_const.data[0], PC_RANGE[0])
    assert np.isclose(head.range_z_const.data[0], PC_RANGE[5] - PC_RANGE[2])
    assert np.isclose(head.offset_z_const.data[0], PC_RANGE[2])

    # Check analytical_param_count is positive
    total = head.analytical_param_count()
    assert total > 0, f"Expected positive param count, got {total}"
    print(f"\n  Branch counts: cls={len(head.cls_branches)}, "
          f"reg={len(head.reg_branches)}, traj={len(head.traj_branches)}")
    print(f"  Analytical param count: {total:,}")

    if ok:
        passed += 1
    print(f"\n[OK] TEST 5")

except Exception as e:
    print(f"  [FAIL] TEST 5 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 6: BEVFormerTrackHead construction — with_box_refine=False
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: BEVFormerTrackHead construction — with_box_refine=False")
print("=" * 80)

try:
    NUM_PRED = 4
    head_shared = BEVFormerTrackHead(
        name='t6_head',
        embed_dims=EMBED_DIMS,
        num_reg_fcs=NUM_REG_FCS,
        cls_out_channels=CLS_OUT_CHANNELS,
        code_size=CODE_SIZE,
        bev_h=10, bev_w=10,
        past_steps=PAST_STEPS,
        fut_steps=FUT_STEPS,
        pc_range=PC_RANGE,
        num_pred=NUM_PRED,
        num_query=NQ,
        with_box_refine=False,
        transformer=None,
    )

    # All layers should return the same shared branch
    for lvl in range(NUM_PRED):
        assert head_shared._get_cls_branch(lvl) is head_shared._get_cls_branch(0)
        assert head_shared._get_reg_branch(lvl) is head_shared._get_reg_branch(0)
        assert head_shared._get_traj_branch(lvl) is head_shared._get_traj_branch(0)

    print("  All layers share the same branch instance.")
    passed += 1
    print(f"\n[OK] TEST 6")

except Exception as e:
    print(f"  [FAIL] TEST 6 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 7: Full branch forward pass with multi-layer (with_box_refine)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: Multi-layer branch forward — PyTorch vs TTSim")
print("=" * 80)

try:
    NUM_PRED = 3
    pt_cls_list = []
    pt_reg_list = []
    pt_traj_list = []

    head_ref = BEVFormerTrackHead(
        name='t7_head',
        embed_dims=EMBED_DIMS,
        num_reg_fcs=NUM_REG_FCS,
        cls_out_channels=CLS_OUT_CHANNELS,
        code_size=CODE_SIZE,
        bev_h=10, bev_w=10,
        past_steps=PAST_STEPS,
        fut_steps=FUT_STEPS,
        pc_range=PC_RANGE,
        num_pred=NUM_PRED,
        num_query=NQ,
        with_box_refine=True,
        transformer=None,
    )

    all_ok = True
    for lvl in range(NUM_PRED):
        pt_cls = build_pt_cls_branch(EMBED_DIMS, NUM_REG_FCS, CLS_OUT_CHANNELS)
        pt_reg = build_pt_reg_branch(EMBED_DIMS, NUM_REG_FCS, CODE_SIZE)
        pt_traj = build_pt_reg_branch(EMBED_DIMS, NUM_REG_FCS, TRAJ_OUT)

        # Neutralize LN affine
        for j in range(NUM_REG_FCS):
            pt_cls[j * 3 + 1].weight.data.fill_(1.0)
            pt_cls[j * 3 + 1].bias.data.fill_(0.0)

        tt_cls_b = head_ref._get_cls_branch(lvl)
        tt_reg_b = head_ref._get_reg_branch(lvl)
        tt_traj_b = head_ref._get_traj_branch(lvl)

        copy_cls_weights_pt_to_tt(pt_cls, tt_cls_b, NUM_REG_FCS)
        copy_reg_weights_pt_to_tt(pt_reg, tt_reg_b, NUM_REG_FCS)
        copy_reg_weights_pt_to_tt(pt_traj, tt_traj_b, NUM_REG_FCS)

        x_np = np.random.randn(BS, NQ, EMBED_DIMS).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        x_tt = F._from_data(f't7_x_L{lvl}', x_np, is_const=True)

        with torch.no_grad():
            pt_cls_out = pt_cls(x_pt)
            pt_reg_out = pt_reg(x_pt)
            pt_traj_out = pt_traj(x_pt)

        tt_cls_out = tt_cls_b(x_tt)
        tt_reg_out = tt_reg_b(x_tt)
        tt_traj_out = tt_traj_b(x_tt)

        ok_c = compare(pt_cls_out, tt_cls_out,
                       f"ClsBranch L{lvl}", atol=1e-4)
        ok_r = compare(pt_reg_out, tt_reg_out,
                       f"RegBranch L{lvl}", atol=1e-5)
        ok_t = compare(pt_traj_out, tt_traj_out,
                       f"TrajBranch L{lvl}", atol=1e-5)

        if not (ok_c and ok_r and ok_t):
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 7")

except Exception as e:
    print(f"  [FAIL] TEST 7 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 8: ClsBranch output shape with various batch dims
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: ClsBranch shape — various batch dimensions")
print("=" * 80)

try:
    tt_cls = ClsBranch('t8_cls', EMBED_DIMS, NUM_REG_FCS, CLS_OUT_CHANNELS)
    # Init weights — param shape is [out_features, in_features]
    for i in range(NUM_REG_FCS + 1):
        fc = tt_cls.fcs[i]
        fc.param.data = np.random.randn(
            fc.out_features, fc.in_features).astype(np.float32) * 0.02
        fc.bias.data = np.zeros(fc.out_features, dtype=np.float32)

    shapes = [
        (1, NQ, EMBED_DIMS),
        (4, NQ, EMBED_DIMS),
    ]

    all_ok = True
    for shape in shapes:
        x_np = np.random.randn(*shape).astype(np.float32)
        x_tt = F._from_data(f't8_x_{shape}', x_np, is_const=True)
        out = tt_cls(x_tt)
        expected = shape[:-1] + (CLS_OUT_CHANNELS,)
        actual = tuple(out.data.shape)
        ok = actual == expected
        print(f"  input={shape} -> output={actual}, expected={expected} "
              f"{'[OK]' if ok else '[FAIL]'}")
        if not ok:
            all_ok = False

    if all_ok:
        passed += 1
    else:
        failed += 1
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 8")

except Exception as e:
    print(f"  [FAIL] TEST 8 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# Summary
# ====================================================================

print("\n" + "=" * 80)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
print("=" * 80)

if failed > 0:
    sys.exit(1)
else:
    print("[OK] All tests passed!")
    sys.exit(0)
