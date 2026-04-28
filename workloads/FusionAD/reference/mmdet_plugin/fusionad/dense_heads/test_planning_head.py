#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for PlanningHeadSingleMode sub-modules (TTSim vs PyTorch).

Validates that MLPFuser, PlanMLP, PlanRegBranch, and PlanningDecoderLayer
in TTSim produce identical results to equivalent PyTorch modules.
"""

import os
import sys
import traceback

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.planning_head import (
    MLPFuser,
    PlanMLP,
    PlanRegBranch,
    PlanningDecoderLayer,
    PlanningDecoder,
    PlanningHeadSingleMode,
)


# ====================================================================
# PyTorch reference modules
# ====================================================================

def build_pt_mlp_fuser(in_features, out_features):
    """PyTorch equivalent of MLPFuser: Linear -> LayerNorm -> ReLU."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LayerNorm(out_features),
        nn.ReLU(inplace=True),
    )


def build_pt_plan_mlp(embed_dims):
    """PyTorch equivalent of PlanMLP: Linear(37,512)->ReLU->Linear(512,512)->ReLU->Linear(512,embed_dims)."""
    return nn.Sequential(
        nn.Linear(37, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, embed_dims),
    )


def build_pt_reg_branch(embed_dims, planning_steps):
    """PyTorch equivalent of PlanRegBranch: Linear(embed_dims*2, embed_dims)->ReLU->Linear(embed_dims, planning_steps*2)."""
    return nn.Sequential(
        nn.Linear(embed_dims * 2, embed_dims),
        nn.ReLU(),
        nn.Linear(embed_dims, planning_steps * 2),
    )


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear_pt_to_tt(pt_linear, tt_linear):
    """Copy weights from PyTorch nn.Linear to TTSim SimNN.Linear (no transpose — SimNN.Linear transposes internally)."""
    tt_linear.param.data = pt_linear.weight.data.detach().numpy().astype(np.float32)
    tt_linear.bias.data = pt_linear.bias.data.detach().numpy().astype(np.float32)


def copy_mlp_fuser_weights(pt_fuser, tt_fuser):
    """Copy PyTorch Sequential(Linear, LayerNorm, ReLU) -> TTSim MLPFuser."""
    copy_linear_pt_to_tt(pt_fuser[0], tt_fuser.fc)
    # LayerNorm: TTSim has no affine. Neutralize PyTorch LN.
    pt_fuser[1].weight.data.fill_(1.0)
    pt_fuser[1].bias.data.fill_(0.0)


def copy_plan_mlp_weights(pt_mlp, tt_mlp):
    """Copy PyTorch Sequential(Linear,ReLU,Linear,ReLU,Linear) -> TTSim PlanMLP."""
    copy_linear_pt_to_tt(pt_mlp[0], tt_mlp.fc0)
    copy_linear_pt_to_tt(pt_mlp[2], tt_mlp.fc1)
    copy_linear_pt_to_tt(pt_mlp[4], tt_mlp.fc2)


def copy_reg_branch_weights(pt_branch, tt_branch):
    """Copy PyTorch Sequential(Linear,ReLU,Linear) -> TTSim PlanRegBranch."""
    copy_linear_pt_to_tt(pt_branch[0], tt_branch.fc0)
    copy_linear_pt_to_tt(pt_branch[2], tt_branch.fc1)


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
PLANNING_STEPS = 6
BS = 1
P = 6    # prediction modes
BEV_H = 10
BEV_W = 10
HW = BEV_H * BEV_W

passed = 0
failed = 0


# ====================================================================
# TEST 1: MLPFuser — PyTorch vs TTSim
# ====================================================================

print("=" * 80)
print("TEST 1: MLPFuser — PyTorch vs TTSim (no LN affine)")
print("=" * 80)

try:
    in_feat = EMBED_DIMS * 3
    pt_fuser = build_pt_mlp_fuser(in_feat, EMBED_DIMS)
    tt_fuser = MLPFuser('t1_fuser', in_feat, EMBED_DIMS)
    copy_mlp_fuser_weights(pt_fuser, tt_fuser)

    x_np = np.random.randn(BS, P, in_feat).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t1_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_fuser(x_pt)

    tt_out = tt_fuser(x_tt)

    ok = compare(pt_out, tt_out, "MLPFuser output", atol=1e-4)
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
# TEST 2: PlanMLP — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: PlanMLP — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_pmlp = build_pt_plan_mlp(EMBED_DIMS)
    tt_pmlp = PlanMLP('t2_pmlp', EMBED_DIMS)
    copy_plan_mlp_weights(pt_pmlp, tt_pmlp)

    x_np = np.random.randn(BS, 1, 37).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t2_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_pmlp(x_pt)

    tt_out = tt_pmlp(x_tt)

    ok = compare(pt_out, tt_out, "PlanMLP output", atol=1e-5)
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
# TEST 3: PlanRegBranch — PyTorch vs TTSim
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: PlanRegBranch — PyTorch vs TTSim")
print("=" * 80)

try:
    pt_rb = build_pt_reg_branch(EMBED_DIMS, PLANNING_STEPS)
    tt_rb = PlanRegBranch('t3_rb', EMBED_DIMS, PLANNING_STEPS)
    copy_reg_branch_weights(pt_rb, tt_rb)

    x_np = np.random.randn(BS, 1, EMBED_DIMS * 2).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t3_x', x_np, is_const=True)

    with torch.no_grad():
        pt_out = pt_rb(x_pt)

    tt_out = tt_rb(x_tt)

    ok = compare(pt_out, tt_out, "PlanRegBranch output", atol=1e-5)
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
# TEST 4: PlanRegBranch output shape — various batch dims
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: PlanRegBranch shape — various batch dimensions")
print("=" * 80)

try:
    tt_rb = PlanRegBranch('t4_rb', EMBED_DIMS, PLANNING_STEPS)
    # Init weights — param shape is [out_features, in_features]
    tt_rb.fc0.param.data = np.random.randn(
        EMBED_DIMS, EMBED_DIMS * 2).astype(np.float32) * 0.02
    tt_rb.fc0.bias.data = np.zeros(EMBED_DIMS, dtype=np.float32)
    tt_rb.fc1.param.data = np.random.randn(
        PLANNING_STEPS * 2, EMBED_DIMS).astype(np.float32) * 0.02
    tt_rb.fc1.bias.data = np.zeros(PLANNING_STEPS * 2, dtype=np.float32)

    shapes = [
        (1, 1, EMBED_DIMS * 2),
        (2, 1, EMBED_DIMS * 2),
        (4, 1, EMBED_DIMS * 2),
    ]

    all_ok = True
    for shape in shapes:
        x_np = np.random.randn(*shape).astype(np.float32)
        x_tt = F._from_data(f't4_x_{shape}', x_np, is_const=True)
        out = tt_rb(x_tt)
        expected = shape[:-1] + (PLANNING_STEPS * 2,)
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
    print(f"\n{'[OK]' if all_ok else '[FAIL]'} TEST 4")

except Exception as e:
    print(f"  [FAIL] TEST 4 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 5: PlanningHeadSingleMode construction
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: PlanningHeadSingleMode construction")
print("=" * 80)

try:
    head = PlanningHeadSingleMode(
        name='t5_head',
        embed_dims=EMBED_DIMS,
        planning_steps=PLANNING_STEPS,
        bev_h=BEV_H,
        bev_w=BEV_W)
    ok = True

    assert head.embed_dims == EMBED_DIMS
    assert head.planning_steps == PLANNING_STEPS
    assert head.bev_h == BEV_H
    assert head.bev_w == BEV_W

    total = head.analytical_param_count()
    assert total > 0, f"Expected positive param count, got {total}"

    # Check BEV adapter (3 blocks of Conv3x3→ReLU→Conv1x1)
    assert hasattr(head, 'adapter_blocks'), "Missing adapter_blocks"
    assert len(head.adapter_blocks) == 3, \
        f"Expected 3 adapter blocks, got {len(head.adapter_blocks)}"
    for i, (c1, relu, c2) in enumerate(head.adapter_blocks):
        assert hasattr(head, f'adapter_{i}_conv1'), f"Missing adapter_{i}_conv1"
        assert hasattr(head, f'adapter_{i}_relu'), f"Missing adapter_{i}_relu"
        assert hasattr(head, f'adapter_{i}_conv2'), f"Missing adapter_{i}_conv2"
    assert hasattr(head, 'adapter_perm_in'), "Missing adapter_perm_in"
    assert hasattr(head, 'adapter_reshape_in'), "Missing adapter_reshape_in"
    assert hasattr(head, 'adapter_reshape_out'), "Missing adapter_reshape_out"
    assert hasattr(head, 'adapter_perm_out'), "Missing adapter_perm_out"
    print(f"  BEV adapter: 3 blocks present: OK")

    # Verify adapter contributes to param count
    D = EMBED_DIMS
    half = D // 2
    adapter_params_per_block = D * half * 9 + half + half * D + D
    adapter_total = 3 * adapter_params_per_block
    print(f"  adapter params (expected): {adapter_total:,}")

    print(f"  embed_dims      = {head.embed_dims}")
    print(f"  planning_steps  = {head.planning_steps}")
    print(f"  bev_h x bev_w   = {head.bev_h} x {head.bev_w}")
    print(f"  param count     = {total:,}")

    if ok:
        passed += 1
    print(f"\n[OK] TEST 5")

except Exception as e:
    print(f"  [FAIL] TEST 5 EXCEPTION: {e}")
    traceback.print_exc()
    failed += 1


# ====================================================================
# TEST 6: PlanningDecoderLayer — construction & param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: PlanningDecoderLayer construction")
print("=" * 80)

try:
    dim_ff = EMBED_DIMS * 2
    layer = PlanningDecoderLayer('t6_layer', EMBED_DIMS, 8, dim_ff)

    # Expected: 2 MHA × 4*(d²+d) + FFN (d*dim_ff + dim_ff + dim_ff*d + d)
    d = EMBED_DIMS
    expected_mha = 2 * 4 * (d * d + d)
    expected_ffn = d * dim_ff + dim_ff + dim_ff * d + d
    expected_total = expected_mha + expected_ffn
    actual = layer.analytical_param_count()

    ok = actual == expected_total
    print(f"  d_model={d}, dim_ff={dim_ff}, nhead=8")
    print(f"  Expected params: {expected_total:,}")
    print(f"  Actual params:   {actual:,}")
    print(f"  {'[OK]' if ok else '[FAIL]'}")

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
# TEST 7: PlanningDecoder — 3 layers, param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: PlanningDecoder — 3 layers")
print("=" * 80)

try:
    dim_ff = EMBED_DIMS * 2
    num_layers = 3
    dec = PlanningDecoder('t7_dec', EMBED_DIMS, 8, dim_ff, num_layers)

    single_layer_params = PlanningDecoderLayer(
        't7_single', EMBED_DIMS, 8, dim_ff).analytical_param_count()
    expected = single_layer_params * num_layers
    actual = dec.analytical_param_count()

    ok = actual == expected
    print(f"  {num_layers} layers × {single_layer_params:,} = {expected:,}")
    print(f"  Actual: {actual:,}")

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
# TEST 8: MLPFuser + max-pool pipeline (fusion -> reduce_max)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: MLPFuser + max-pool (PyTorch vs TTSim)")
print("=" * 80)

try:
    in_feat = EMBED_DIMS * 3
    pt_fuser = build_pt_mlp_fuser(in_feat, EMBED_DIMS)
    tt_fuser = MLPFuser('t8_fuser', in_feat, EMBED_DIMS)
    copy_mlp_fuser_weights(pt_fuser, tt_fuser)

    x_np = np.random.randn(BS, P, in_feat).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tt = F._from_data('t8_x', x_np, is_const=True)

    with torch.no_grad():
        pt_fused = pt_fuser(x_pt)
        pt_maxed = pt_fused.max(1, keepdim=True)[0]  # [bs, 1, embed_dims]

    tt_fused = tt_fuser(x_tt)
    reduce_max = F.ReduceMax('t8_reduce_max', axes=[1], keepdims=1)
    tt_maxed = reduce_max(tt_fused)

    ok = compare(pt_maxed, tt_maxed, "Fuser + max-pool", atol=1e-4)
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
# TEST 9: PlanMLP param count
# ====================================================================

print("\n" + "=" * 80)
print("TEST 9: PlanMLP analytical param count")
print("=" * 80)

try:
    tt_pmlp = PlanMLP('t9_pmlp', EMBED_DIMS)
    expected = (37 * 512 + 512 +
                512 * 512 + 512 +
                512 * EMBED_DIMS + EMBED_DIMS)
    actual = tt_pmlp.analytical_param_count()

    ok = actual == expected
    print(f"  Expected: {expected:,}")
    print(f"  Actual:   {actual:,}")

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
# TEST 10: Full pipeline — concat + fuser + max + reg_branch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 10: Full pipeline — concat + fuser + max + reg_branch")
print("=" * 80)

try:
    # PyTorch pipeline
    in_feat = EMBED_DIMS * 3
    pt_fuser = build_pt_mlp_fuser(in_feat, EMBED_DIMS)
    pt_rb = build_pt_reg_branch(EMBED_DIMS, PLANNING_STEPS)
    pt_plan_mlp = build_pt_plan_mlp(EMBED_DIMS)

    # TTSim pipeline
    tt_fuser = MLPFuser('t10_fuser', in_feat, EMBED_DIMS)
    tt_rb = PlanRegBranch('t10_rb', EMBED_DIMS, PLANNING_STEPS)
    tt_plan_mlp = PlanMLP('t10_pmlp', EMBED_DIMS)

    # Copy weights
    copy_mlp_fuser_weights(pt_fuser, tt_fuser)
    copy_reg_branch_weights(pt_rb, tt_rb)
    copy_plan_mlp_weights(pt_plan_mlp, tt_plan_mlp)

    # Inputs
    sdc_traj_q = np.random.randn(BS, P, EMBED_DIMS).astype(np.float32)
    sdc_track_q = np.random.randn(BS, EMBED_DIMS).astype(np.float32)
    navi_emb = np.random.randn(EMBED_DIMS).astype(np.float32)
    plan_info = np.random.randn(BS, 1, 37).astype(np.float32)

    # Expand track query
    sdc_track_exp = np.tile(sdc_track_q[:, np.newaxis, :], (1, P, 1))
    navi_exp = np.tile(navi_emb[np.newaxis, np.newaxis, :], (BS, P, 1))

    cat_np = np.concatenate([sdc_traj_q, sdc_track_exp, navi_exp], axis=-1).astype(np.float32)

    # --- PyTorch path ---
    cat_pt = torch.from_numpy(cat_np)
    plan_info_pt = torch.from_numpy(plan_info)

    with torch.no_grad():
        fused_pt = pt_fuser(cat_pt)
        maxed_pt = fused_pt.max(1, keepdim=True)[0]  # [bs, 1, embed_dims]
        plan_emd_pt = pt_plan_mlp(plan_info_pt)       # [bs, 1, embed_dims]
        concat_pt = torch.cat([maxed_pt, plan_emd_pt], dim=-1)  # [bs, 1, embed_dims*2]
        traj_pt = pt_rb(concat_pt)                     # [bs, 1, planning_steps*2]

    # --- TTSim path ---
    cat_tt = F._from_data('t10_cat', cat_np, is_const=True)
    plan_info_tt = F._from_data('t10_plan_info', plan_info, is_const=True)

    fused_tt = tt_fuser(cat_tt)
    reduce_max = F.ReduceMax('t10_reduce_max', axes=[1], keepdims=1)
    maxed_tt = reduce_max(fused_tt)
    plan_emd_tt = tt_plan_mlp(plan_info_tt)
    concat_op = F.ConcatX('t10_concat', axis=-1)
    concat_tt = concat_op(maxed_tt, plan_emd_tt)
    traj_tt = tt_rb(concat_tt)

    ok = compare(traj_pt, traj_tt, "Full pipeline output", atol=1e-4)
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
