#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for bbox/util.py: normalize_bbox and denormalize_bbox.
Tests shape and numerical equivalence between PyTorch and TTSIM versions.

Both versions implement the same logic:
  normalize_bbox  : [cx,cy,cz,w,l,h,rot(,vx,vy)] → [cx,cy,log(w),log(l),cz,log(h),sin,cos(,vx,vy)]
  denormalize_bbox: inverse of the above
"""

import numpy as np
import torch
import os, sys

_POLARIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, _POLARIS_DIR)

import ttsim.front.functional.op as F


# ===========================================================================
# PyTorch versions (inlined from original util.py)
# ===========================================================================

def normalize_bbox_torch(bboxes, pc_range):
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
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox_torch(normalized_bboxes, pc_range):
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
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


# ===========================================================================
# TTSIM versions (inlined from converted util.py)
# ===========================================================================

_atan2_counter = 0


def atan2_ttsim(y, x):
    """Compute torch.atan2(y, x) for TTSim tensors using numpy on .data."""
    global _atan2_counter
    _atan2_counter += 1
    sfx = str(_atan2_counter)

    y_data = y.data
    x_data = x.data

    if y_data is not None and x_data is not None:
        angle_data = np.arctan2(y_data, x_data).astype(np.float32)
        return F._from_data(f'atan2_result_{sfx}', angle_data, is_const=False)
    else:
        return F._from_shape(f'atan2_result_{sfx}', list(y.shape), np_dtype=np.float32)


def normalize_bbox_ttsim(bboxes, pc_range):
    """TTSIM normalize_bbox — same signature as torch version."""
    out_shape = list(bboxes.shape)
    out_shape[-1] = 1

    axes = F._from_data('norm_axes', np.array([-1], dtype=np.int64), is_const=True)
    steps = F._from_data('norm_steps', np.array([1], dtype=np.int64), is_const=True)

    def _s(v):
        return F._from_data(f'norm_s{v}', np.array([v], dtype=np.int64), is_const=True)

    def _e(v):
        return F._from_data(f'norm_e{v}', np.array([v], dtype=np.int64), is_const=True)

    cx  = F.SliceF('norm_cx',  out_shape=out_shape)(bboxes, _s(0), _e(1), axes, steps)
    cy  = F.SliceF('norm_cy',  out_shape=out_shape)(bboxes, _s(1), _e(2), axes, steps)
    cz  = F.SliceF('norm_cz',  out_shape=out_shape)(bboxes, _s(2), _e(3), axes, steps)
    w   = F.SliceF('norm_w',   out_shape=out_shape)(bboxes, _s(3), _e(4), axes, steps)
    l   = F.SliceF('norm_l',   out_shape=out_shape)(bboxes, _s(4), _e(5), axes, steps)
    h   = F.SliceF('norm_h',   out_shape=out_shape)(bboxes, _s(5), _e(6), axes, steps)
    rot = F.SliceF('norm_rot', out_shape=out_shape)(bboxes, _s(6), _e(7), axes, steps)

    w_log = F.Log('norm_log_w')(w)
    l_log = F.Log('norm_log_l')(l)
    h_log = F.Log('norm_log_h')(h)

    rot_sin = F.Sin('norm_sin_rot')(rot)
    rot_cos = F.Cos('norm_cos_rot')(rot)

    if bboxes.shape[-1] > 7:
        vx = F.SliceF('norm_vx', out_shape=out_shape)(bboxes, _s(7), _e(8), axes, steps)
        vy = F.SliceF('norm_vy', out_shape=out_shape)(bboxes, _s(8), _e(9), axes, steps)
        normalized = F.ConcatX('norm_concat', axis=-1)(
            cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy
        )
    else:
        normalized = F.ConcatX('norm_concat', axis=-1)(
            cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos
        )
    return normalized


def denormalize_bbox_ttsim(normalized_bboxes, pc_range):
    """TTSIM denormalize_bbox — same signature as torch version."""
    out_shape = list(normalized_bboxes.shape)
    out_shape[-1] = 1

    axes = F._from_data('denorm_axes', np.array([-1], dtype=np.int64), is_const=True)
    steps = F._from_data('denorm_steps', np.array([1], dtype=np.int64), is_const=True)

    def _s(v):
        return F._from_data(f'denorm_s{v}', np.array([v], dtype=np.int64), is_const=True)

    def _e(v):
        return F._from_data(f'denorm_e{v}', np.array([v], dtype=np.int64), is_const=True)

    rot_sine   = F.SliceF('denorm_rot_sin', out_shape=out_shape)(
        normalized_bboxes, _s(6), _e(7), axes, steps)
    rot_cosine = F.SliceF('denorm_rot_cos', out_shape=out_shape)(
        normalized_bboxes, _s(7), _e(8), axes, steps)
    rot = atan2_ttsim(rot_sine, rot_cosine)

    cx = F.SliceF('denorm_cx', out_shape=out_shape)(normalized_bboxes, _s(0), _e(1), axes, steps)
    cy = F.SliceF('denorm_cy', out_shape=out_shape)(normalized_bboxes, _s(1), _e(2), axes, steps)
    cz = F.SliceF('denorm_cz', out_shape=out_shape)(normalized_bboxes, _s(4), _e(5), axes, steps)

    w_log = F.SliceF('denorm_w_log', out_shape=out_shape)(normalized_bboxes, _s(2), _e(3), axes, steps)
    l_log = F.SliceF('denorm_l_log', out_shape=out_shape)(normalized_bboxes, _s(3), _e(4), axes, steps)
    h_log = F.SliceF('denorm_h_log', out_shape=out_shape)(normalized_bboxes, _s(5), _e(6), axes, steps)

    w = F.Exp('denorm_exp_w')(w_log)
    l = F.Exp('denorm_exp_l')(l_log)
    h = F.Exp('denorm_exp_h')(h_log)

    if normalized_bboxes.shape[-1] > 8:
        vx = F.SliceF('denorm_vx', out_shape=out_shape)(normalized_bboxes, _s(8), _e(9), axes, steps)
        vy = F.SliceF('denorm_vy', out_shape=out_shape)(normalized_bboxes, _s(9), _e(10), axes, steps)
        denormalized = F.ConcatX('denorm_concat', axis=-1)(cx, cy, cz, w, l, h, rot, vx, vy)
    else:
        denormalized = F.ConcatX('denorm_concat', axis=-1)(cx, cy, cz, w, l, h, rot)
    return denormalized


# ===========================================================================
# Test helpers
# ===========================================================================

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def make_bboxes_np(batch, dim, seed=42):
    """Generate deterministic bbox data (positive dims to keep log valid)."""
    rng = np.random.RandomState(seed)
    data = rng.randn(batch, dim).astype(np.float32)
    # dims (w,l,h) at indices 3,4,5 must be positive for log
    data[:, 3:6] = np.abs(data[:, 3:6]) + 0.1
    return data


# ===========================================================================
# Tests
# ===========================================================================

def test_normalize_shape_no_velocity():
    """normalize_bbox with 7-dim input → 8-dim output."""
    print("\n" + "=" * 70)
    print("TEST 1: normalize_bbox shape — no velocity (7→8)")
    print("=" * 70)

    data = make_bboxes_np(5, 7)
    pc_range = None

    # PyTorch
    pt_out = normalize_bbox_torch(torch.from_numpy(data), pc_range)

    # TTSIM
    tt_in = F._from_data('bbox_in', data)
    tt_out = normalize_bbox_ttsim(tt_in, pc_range)

    check("PyTorch shape", list(pt_out.shape) == [5, 8], f"got {list(pt_out.shape)}")
    check("TTSIM  shape", list(tt_out.shape) == [5, 8], f"got {list(tt_out.shape)}")


def test_normalize_shape_with_velocity():
    """normalize_bbox with 9-dim input → 10-dim output."""
    print("\n" + "=" * 70)
    print("TEST 2: normalize_bbox shape — with velocity (9→10)")
    print("=" * 70)

    data = make_bboxes_np(4, 9, seed=99)
    pc_range = None

    pt_out = normalize_bbox_torch(torch.from_numpy(data), pc_range)

    tt_in = F._from_data('bbox_in_v', data)
    tt_out = normalize_bbox_ttsim(tt_in, pc_range)

    check("PyTorch shape", list(pt_out.shape) == [4, 10], f"got {list(pt_out.shape)}")
    check("TTSIM  shape", list(tt_out.shape) == [4, 10], f"got {list(tt_out.shape)}")


def test_denormalize_shape_no_velocity():
    """denormalize_bbox with 8-dim input → 7-dim output."""
    print("\n" + "=" * 70)
    print("TEST 3: denormalize_bbox shape — no velocity (8→7)")
    print("=" * 70)

    data = make_bboxes_np(5, 7, seed=10)
    pc_range = None

    # First normalize to get valid 8-dim input
    norm_pt = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    norm_np = norm_pt.detach().numpy()

    pt_out = denormalize_bbox_torch(norm_pt, pc_range)

    tt_in = F._from_data('norm_in', norm_np)
    tt_out = denormalize_bbox_ttsim(tt_in, pc_range)

    check("PyTorch shape", list(pt_out.shape) == [5, 7], f"got {list(pt_out.shape)}")
    check("TTSIM  shape", list(tt_out.shape) == [5, 7], f"got {list(tt_out.shape)}")


def test_denormalize_shape_with_velocity():
    """denormalize_bbox with 10-dim input → 9-dim output."""
    print("\n" + "=" * 70)
    print("TEST 4: denormalize_bbox shape — with velocity (10→9)")
    print("=" * 70)

    data = make_bboxes_np(3, 9, seed=20)
    pc_range = None

    norm_pt = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    norm_np = norm_pt.detach().numpy()

    pt_out = denormalize_bbox_torch(norm_pt, pc_range)

    tt_in = F._from_data('norm_in_v', norm_np)
    tt_out = denormalize_bbox_ttsim(tt_in, pc_range)

    check("PyTorch shape", list(pt_out.shape) == [3, 9], f"got {list(pt_out.shape)}")
    check("TTSIM  shape", list(tt_out.shape) == [3, 9], f"got {list(tt_out.shape)}")


def test_normalize_numerical_no_velocity():
    """Numerical comparison of normalize_bbox (7-dim)."""
    print("\n" + "=" * 70)
    print("TEST 5: normalize_bbox numerical — no velocity")
    print("=" * 70)

    data = make_bboxes_np(6, 7, seed=55)
    pc_range = None

    pt_out = normalize_bbox_torch(torch.from_numpy(data), pc_range).detach().numpy()

    tt_in = F._from_data('bbox_num', data)
    tt_out = normalize_bbox_ttsim(tt_in, pc_range)
    tt_data = tt_out.data

    if tt_data is None:
        check("Data propagation", False, "TTSIM output .data is None (shape-only mode)")
        return

    check("Arrays close (atol=1e-5)", np.allclose(pt_out, tt_data, atol=1e-5),
          f"max diff = {np.max(np.abs(pt_out - tt_data)):.2e}")

    # Per-component checks
    # cx(0), cy(1) pass-through
    check("cx pass-through", np.allclose(pt_out[:, 0], tt_data[:, 0], atol=1e-6))
    check("cy pass-through", np.allclose(pt_out[:, 1], tt_data[:, 1], atol=1e-6))
    # log(w)(2), log(l)(3)
    check("log(w) match", np.allclose(pt_out[:, 2], tt_data[:, 2], atol=1e-5))
    check("log(l) match", np.allclose(pt_out[:, 3], tt_data[:, 3], atol=1e-5))
    # cz(4)
    check("cz pass-through", np.allclose(pt_out[:, 4], tt_data[:, 4], atol=1e-6))
    # log(h)(5)
    check("log(h) match", np.allclose(pt_out[:, 5], tt_data[:, 5], atol=1e-5))
    # sin(rot)(6), cos(rot)(7)
    check("sin(rot) match", np.allclose(pt_out[:, 6], tt_data[:, 6], atol=1e-5))
    check("cos(rot) match", np.allclose(pt_out[:, 7], tt_data[:, 7], atol=1e-5))


def test_normalize_numerical_with_velocity():
    """Numerical comparison of normalize_bbox (9-dim, with velocity)."""
    print("\n" + "=" * 70)
    print("TEST 6: normalize_bbox numerical — with velocity")
    print("=" * 70)

    data = make_bboxes_np(4, 9, seed=77)
    pc_range = None

    pt_out = normalize_bbox_torch(torch.from_numpy(data), pc_range).detach().numpy()

    tt_in = F._from_data('bbox_numv', data)
    tt_out = normalize_bbox_ttsim(tt_in, pc_range)
    tt_data = tt_out.data

    if tt_data is None:
        check("Data propagation", False, "TTSIM output .data is None")
        return

    check("Arrays close (atol=1e-5)", np.allclose(pt_out, tt_data, atol=1e-5),
          f"max diff = {np.max(np.abs(pt_out - tt_data)):.2e}")
    check("vx pass-through", np.allclose(pt_out[:, 8], tt_data[:, 8], atol=1e-6))
    check("vy pass-through", np.allclose(pt_out[:, 9], tt_data[:, 9], atol=1e-6))


def test_denormalize_numerical_no_velocity():
    """Numerical comparison of denormalize_bbox (8→7)."""
    print("\n" + "=" * 70)
    print("TEST 7: denormalize_bbox numerical — no velocity")
    print("=" * 70)

    data = make_bboxes_np(5, 7, seed=33)
    pc_range = None

    norm_pt = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    norm_np = norm_pt.detach().numpy()

    pt_out = denormalize_bbox_torch(norm_pt, pc_range).detach().numpy()

    tt_in = F._from_data('denorm_num', norm_np)
    tt_out = denormalize_bbox_ttsim(tt_in, pc_range)
    tt_data = tt_out.data

    if tt_data is None:
        check("Data propagation", False, "TTSIM output .data is None")
        return

    check("Arrays close (atol=1e-4)", np.allclose(pt_out, tt_data, atol=1e-4),
          f"max diff = {np.max(np.abs(pt_out - tt_data)):.2e}")
    check("cx match", np.allclose(pt_out[:, 0], tt_data[:, 0], atol=1e-5))
    check("cy match", np.allclose(pt_out[:, 1], tt_data[:, 1], atol=1e-5))
    check("cz match", np.allclose(pt_out[:, 2], tt_data[:, 2], atol=1e-5))
    check("w (exp) match", np.allclose(pt_out[:, 3], tt_data[:, 3], atol=1e-4))
    check("l (exp) match", np.allclose(pt_out[:, 4], tt_data[:, 4], atol=1e-4))
    check("h (exp) match", np.allclose(pt_out[:, 5], tt_data[:, 5], atol=1e-4))
    check("rot (atan2) match", np.allclose(pt_out[:, 6], tt_data[:, 6], atol=1e-4),
          f"max diff = {np.max(np.abs(pt_out[:, 6] - tt_data[:, 6])):.2e}")


def test_denormalize_numerical_with_velocity():
    """Numerical comparison of denormalize_bbox (10→9)."""
    print("\n" + "=" * 70)
    print("TEST 8: denormalize_bbox numerical — with velocity")
    print("=" * 70)

    data = make_bboxes_np(3, 9, seed=44)
    pc_range = None

    norm_pt = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    norm_np = norm_pt.detach().numpy()

    pt_out = denormalize_bbox_torch(norm_pt, pc_range).detach().numpy()

    tt_in = F._from_data('denorm_numv', norm_np)
    tt_out = denormalize_bbox_ttsim(tt_in, pc_range)
    tt_data = tt_out.data

    if tt_data is None:
        check("Data propagation", False, "TTSIM output .data is None")
        return

    check("Arrays close (atol=1e-4)", np.allclose(pt_out, tt_data, atol=1e-4),
          f"max diff = {np.max(np.abs(pt_out - tt_data)):.2e}")
    check("vx match", np.allclose(pt_out[:, 7], tt_data[:, 7], atol=1e-6))
    check("vy match", np.allclose(pt_out[:, 8], tt_data[:, 8], atol=1e-6))


def test_round_trip_no_velocity():
    """normalize→denormalize should recover original bbox (7-dim)."""
    print("\n" + "=" * 70)
    print("TEST 9: Round-trip consistency — no velocity")
    print("=" * 70)

    data = make_bboxes_np(8, 7, seed=60)
    pc_range = None

    # PyTorch round-trip
    pt_norm = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    pt_round = denormalize_bbox_torch(pt_norm, pc_range).detach().numpy()

    check("PT round-trip (atol=1e-5)", np.allclose(data, pt_round, atol=1e-5),
          f"max diff = {np.max(np.abs(data - pt_round)):.2e}")

    # TTSIM round-trip (shape only — data propagation may not chain through all ops)
    tt_in = F._from_data('rt_in', data)
    tt_norm = normalize_bbox_ttsim(tt_in, pc_range)
    check("TTSIM normalize shape", list(tt_norm.shape) == [8, 8])

    # Use numpy data from torch-normalized for denorm to verify denorm alone
    tt_norm2 = F._from_data('rt_norm', pt_norm.detach().numpy())
    tt_round = denormalize_bbox_ttsim(tt_norm2, pc_range)
    check("TTSIM denormalize shape", list(tt_round.shape) == [8, 7])

    if tt_round.data is not None:
        check("TTSIM round-trip (atol=1e-4)", np.allclose(data, tt_round.data, atol=1e-4),
              f"max diff = {np.max(np.abs(data - tt_round.data)):.2e}")


def test_round_trip_with_velocity():
    """normalize→denormalize should recover original bbox (9-dim)."""
    print("\n" + "=" * 70)
    print("TEST 10: Round-trip consistency — with velocity")
    print("=" * 70)

    data = make_bboxes_np(4, 9, seed=70)
    pc_range = None

    pt_norm = normalize_bbox_torch(torch.from_numpy(data), pc_range)
    pt_round = denormalize_bbox_torch(pt_norm, pc_range).detach().numpy()

    check("PT round-trip (atol=1e-5)", np.allclose(data, pt_round, atol=1e-5),
          f"max diff = {np.max(np.abs(data - pt_round)):.2e}")

    tt_norm2 = F._from_data('rt_normv', pt_norm.detach().numpy())
    tt_round = denormalize_bbox_ttsim(tt_norm2, pc_range)
    check("TTSIM denormalize shape", list(tt_round.shape) == [4, 9])

    if tt_round.data is not None:
        check("TTSIM round-trip (atol=1e-4)", np.allclose(data, tt_round.data, atol=1e-4),
              f"max diff = {np.max(np.abs(data - tt_round.data)):.2e}")


def test_batch_sizes():
    """Verify shapes across different batch sizes."""
    print("\n" + "=" * 70)
    print("TEST 11: Various batch sizes")
    print("=" * 70)

    for bs in [1, 16, 64]:
        data = make_bboxes_np(bs, 7, seed=bs)
        tt_in = F._from_data(f'bs{bs}', data)
        tt_out = normalize_bbox_ttsim(tt_in, None)
        check(f"batch={bs}  shape [{bs},7]→[{bs},8]", list(tt_out.shape) == [bs, 8])


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("bbox/util.py — PyTorch vs TTSIM Validation")
    print("=" * 70)

    test_normalize_shape_no_velocity()
    test_normalize_shape_with_velocity()
    test_denormalize_shape_no_velocity()
    test_denormalize_shape_with_velocity()
    test_normalize_numerical_no_velocity()
    test_normalize_numerical_with_velocity()
    test_denormalize_numerical_no_velocity()
    test_denormalize_numerical_with_velocity()
    test_round_trip_no_velocity()
    test_round_trip_with_velocity()
    test_batch_sizes()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
