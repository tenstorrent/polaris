#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FusionAD bbox utilities (TTSim vs PyTorch).

Validates that the TTSim SimNN.Module implementations of NormalizeBbox
and DenormalizeBbox produce identical results to the PyTorch originals.
Also validates the numpy convenience functions.
"""

import os
import sys
import traceback

polaris_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch

import ttsim.front.functional.op as F

# ---- TTSim modules under test ----
from workloads.FusionAD.projects.mmdet_plugin.core.bbox.util import (
    NormalizeBbox,
    DenormalizeBbox,
    normalize_bbox_np,
    denormalize_bbox_np,
)


# ====================================================================
# PyTorch reference (exact copy from original)
# ====================================================================

def normalize_bbox_pytorch(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes

def denormalize_bbox_pytorch(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]
    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


# ====================================================================
# Helper
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
# TEST 1: NormalizeBbox TTSim module – 9-dim (with velocity)
# ====================================================================

print("=" * 80)
print("TEST 1: NormalizeBbox TTSim Module – 9-dim (with velocity)")
print("=" * 80)

try:
    np.random.seed(42)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    bboxes_np = np.random.rand(3, 9).astype(np.float32) + 0.1
    bboxes_np[:, 6] = np.random.randn(3) * 0.5
    bboxes_np[:, 7:9] = np.random.randn(3, 2)

    # PyTorch reference
    pt_out = normalize_bbox_pytorch(torch.from_numpy(bboxes_np), pc_range)

    # TTSim module
    norm_mod = NormalizeBbox('t1_norm', has_velocity=True)
    bboxes_t = F._from_data('t1_bboxes', bboxes_np, is_const=True)
    tt_out = norm_mod(bboxes_t)

    ok = compare(pt_out, tt_out, "NormalizeBbox (9-dim)")
    if not ok:
        sys.exit(1)
    print("\n[OK] TEST 1 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 1 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 2: NormalizeBbox TTSim module – 7-dim (no velocity)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 2: NormalizeBbox TTSim Module – 7-dim (no velocity)")
print("=" * 80)

try:
    bboxes7_np = bboxes_np[:, :7].copy()

    pt_out7 = normalize_bbox_pytorch(torch.from_numpy(bboxes7_np), pc_range)

    norm_mod7 = NormalizeBbox('t2_norm', has_velocity=False)
    bboxes7_t = F._from_data('t2_bboxes', bboxes7_np, is_const=True)
    tt_out7 = norm_mod7(bboxes7_t)

    ok = compare(pt_out7, tt_out7, "NormalizeBbox (7-dim)")
    if not ok:
        sys.exit(1)
    print("\n[OK] TEST 2 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 2 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 3: DenormalizeBbox TTSim module – 10-dim (with velocity)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 3: DenormalizeBbox TTSim Module – 10-dim (with velocity)")
print("=" * 80)

try:
    # Normalize first to get valid input for denormalize
    norm_pt = normalize_bbox_pytorch(torch.from_numpy(bboxes_np), pc_range)
    norm_np = norm_pt.detach().numpy().copy()

    # PyTorch reference
    pt_denorm = denormalize_bbox_pytorch(norm_pt, pc_range)

    # TTSim module
    denorm_mod = DenormalizeBbox('t3_denorm', has_velocity=True)
    norm_t = F._from_data('t3_norm', norm_np, is_const=True)
    tt_denorm = denorm_mod(norm_t)

    ok = compare(pt_denorm, tt_denorm, "DenormalizeBbox (10-dim)")
    if not ok:
        sys.exit(1)
    print("\n[OK] TEST 3 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 3 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 4: DenormalizeBbox TTSim module – 8-dim (no velocity)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 4: DenormalizeBbox TTSim Module – 8-dim (no velocity)")
print("=" * 80)

try:
    norm_pt8 = normalize_bbox_pytorch(torch.from_numpy(bboxes7_np), pc_range)
    norm_np8 = norm_pt8.detach().numpy().copy()

    pt_denorm8 = denormalize_bbox_pytorch(norm_pt8, pc_range)

    denorm_mod8 = DenormalizeBbox('t4_denorm', has_velocity=False)
    norm_t8 = F._from_data('t4_norm', norm_np8, is_const=True)
    tt_denorm8 = denorm_mod8(norm_t8)

    ok = compare(pt_denorm8, tt_denorm8, "DenormalizeBbox (8-dim)")
    if not ok:
        sys.exit(1)
    print("\n[OK] TEST 4 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 4 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 5: Round-trip (NormalizeBbox → DenormalizeBbox, TTSim graph)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 5: Round-trip (TTSim NormalizeBbox → DenormalizeBbox)")
print("=" * 80)

try:
    # 9-dim
    norm_mod_rt = NormalizeBbox('t5_norm', has_velocity=True)
    denorm_mod_rt = DenormalizeBbox('t5_denorm', has_velocity=True)
    bboxes_rt = F._from_data('t5_bboxes', bboxes_np, is_const=True)
    normalized = norm_mod_rt(bboxes_rt)
    recovered = denorm_mod_rt(normalized)

    ok = compare(torch.from_numpy(bboxes_np), recovered,
                 "Round-trip 9-dim (TTSim)", atol=1e-5)
    if not ok:
        sys.exit(1)

    # 7-dim
    norm_mod_rt7 = NormalizeBbox('t5_norm7', has_velocity=False)
    denorm_mod_rt7 = DenormalizeBbox('t5_denorm7', has_velocity=False)
    bboxes_rt7 = F._from_data('t5_bboxes7', bboxes7_np, is_const=True)
    normalized7 = norm_mod_rt7(bboxes_rt7)
    recovered7 = denorm_mod_rt7(normalized7)

    ok7 = compare(torch.from_numpy(bboxes7_np), recovered7,
                  "Round-trip 7-dim (TTSim)", atol=1e-5)
    if not ok7:
        sys.exit(1)

    print("\n[OK] TEST 5 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 5 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 6: Numpy convenience functions vs PyTorch
# ====================================================================

print("\n" + "=" * 80)
print("TEST 6: Numpy Convenience Functions vs PyTorch")
print("=" * 80)

try:
    pt_norm = normalize_bbox_pytorch(torch.from_numpy(bboxes_np), pc_range)
    np_norm = normalize_bbox_np(bboxes_np, pc_range)
    ok_n = compare(pt_norm, np_norm, "normalize_bbox_np")

    pt_denorm = denormalize_bbox_pytorch(pt_norm, pc_range)
    np_denorm = denormalize_bbox_np(np_norm, pc_range)
    ok_d = compare(pt_denorm, np_denorm, "denormalize_bbox_np")

    if not (ok_n and ok_d):
        sys.exit(1)
    print("\n[OK] TEST 6 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 6 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 7: Batch dimensions (TTSim modules)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 7: Batch Dimensions (TTSim modules)")
print("=" * 80)

try:
    for label, shape in [("(4, 9)", (4, 9)),
                          ("(2, 5, 9)", (2, 5, 9))]:
        b_np = np.random.rand(*shape).astype(np.float32) + 0.1
        b_np[..., 6] = np.random.randn(*shape[:-1]) * 0.5
        b_np[..., 7:9] = np.random.randn(*shape[:-1], 2)

        pt_n = normalize_bbox_pytorch(torch.from_numpy(b_np), pc_range)

        nm = NormalizeBbox(f't7_norm_{label}', has_velocity=True)
        b_t = F._from_data(f't7_b_{label}', b_np, is_const=True)
        tt_n = nm(b_t)
        ok = compare(pt_n, tt_n, f"NormalizeBbox {label}")
        if not ok:
            sys.exit(1)

        pt_d = denormalize_bbox_pytorch(pt_n, pc_range)
        dm = DenormalizeBbox(f't7_denorm_{label}', has_velocity=True)
        tt_d = dm(tt_n)
        ok = compare(pt_d, tt_d, f"DenormalizeBbox {label}")
        if not ok:
            sys.exit(1)

    print("\n[OK] TEST 7 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 7 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# TEST 8: atan2 edge cases (quadrants)
# ====================================================================

print("\n" + "=" * 80)
print("TEST 8: atan2 Edge Cases (all quadrants)")
print("=" * 80)

try:
    # Test rotations in all four quadrants
    angles = np.array([0.3, 1.2, 2.5, -0.5, -1.8, -2.9, 0.0, np.pi/2],
                       dtype=np.float32)
    bboxes_angles = np.zeros((len(angles), 9), dtype=np.float32)
    bboxes_angles[:, 0] = 1.0   # cx
    bboxes_angles[:, 1] = 2.0   # cy
    bboxes_angles[:, 2] = 3.0   # cz
    bboxes_angles[:, 3] = 0.5   # w (positive for log)
    bboxes_angles[:, 4] = 1.2   # l
    bboxes_angles[:, 5] = 0.8   # h
    bboxes_angles[:, 6] = angles  # rot
    bboxes_angles[:, 7] = 0.1   # vx
    bboxes_angles[:, 8] = -0.2  # vy

    pt_norm_a = normalize_bbox_pytorch(torch.from_numpy(bboxes_angles), pc_range)
    pt_denorm_a = denormalize_bbox_pytorch(pt_norm_a, pc_range)

    nm_a = NormalizeBbox('t8_norm', has_velocity=True)
    dm_a = DenormalizeBbox('t8_denorm', has_velocity=True)
    b_t_a = F._from_data('t8_bboxes', bboxes_angles, is_const=True)
    tt_norm_a = nm_a(b_t_a)
    tt_denorm_a = dm_a(tt_norm_a)

    ok = compare(pt_denorm_a, tt_denorm_a, "atan2 quadrant test", atol=1e-4)
    if not ok:
        # Show per-angle comparison for diagnostics
        pt_np = pt_denorm_a.detach().numpy()
        tt_np = tt_denorm_a.data
        for i, angle in enumerate(angles):
            pt_rot = pt_np[i, 6]
            tt_rot = tt_np[i, 6]
            print(f"    angle={angle:+.4f} => PT rot={pt_rot:+.6f}, TT rot={tt_rot:+.6f}, "
                  f"diff={abs(pt_rot-tt_rot):.2e}")
        sys.exit(1)
    print("\n[OK] TEST 8 PASSED")

except Exception as e:
    print(f"  [FAIL] TEST 8 FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ====================================================================
# Test Summary
# ====================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

tests = [
    "NormalizeBbox TTSim – 9-dim (with velocity)",
    "NormalizeBbox TTSim – 7-dim (no velocity)",
    "DenormalizeBbox TTSim – 10-dim (with velocity)",
    "DenormalizeBbox TTSim – 8-dim (no velocity)",
    "Round-trip (TTSim NormalizeBbox → DenormalizeBbox)",
    "Numpy Convenience Functions vs PyTorch",
    "Batch Dimensions (TTSim modules)",
    "atan2 Edge Cases (all quadrants)",
]

for i, test in enumerate(tests, 1):
    print(f"  TEST {i}: {test:.<55s} [OK] PASSED")

print(f"\nTotal: {len(tests)}/{len(tests)} tests passed")
print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
