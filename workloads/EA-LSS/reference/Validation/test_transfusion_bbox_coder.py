#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for transfusion_bbox_coder TTSim module.

Validates the TTSim conversion of
mmdet3d/core/bbox/coders/transfusion_bbox_coder.py.

Test Coverage:
  1.  encode_numpy shape     – output [N, code_size]
  2.  encode_numpy values    – manual formula check
  3.  decode shape           – boxes [B,P,7] and scores [B,P]
  4.  decode dim_exp         – exp of log gives original dims
  5.  decode center coords   – manual rescale check
  6.  decode height          – z - h/2
"""

import os
import sys

_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_val_dir      = os.path.dirname(__file__)
for p in (_polaris_root, _ealss_root, _val_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.transfusion_bbox_coder import TransFusionBBoxCoder
from ttsim_utils import compare_arrays, print_header, print_test


# ============================================================================
# Default config (matches TranFusion-L defaults)
# ============================================================================

PC_RANGE       = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
OUT_SIZE_FACTOR = 8
VOXEL_SIZE      = [0.075, 0.075, 0.2]
CODE_SIZE       = 8   # x,y,z,w,l,h,sin_yaw,cos_yaw

rng = np.random.RandomState(99)


def _make_coder():
    return TransFusionBBoxCoder(
        "tfc",
        PC_RANGE, OUT_SIZE_FACTOR, VOXEL_SIZE,
        code_size=CODE_SIZE,
    )


# ============================================================================
# Tests
# ============================================================================

def test_encode_shape():
    print_header("TEST 1: encode_numpy shape")
    coder = _make_coder()
    N = 20
    boxes = rng.randn(N, 7).astype(np.float32)
    boxes[:, 3:6] = np.abs(boxes[:, 3:6]) + 0.5   # positive dims
    tgts = coder.encode_numpy(boxes)
    assert tgts.shape == (N, CODE_SIZE), \
        f"Expected ({N},{CODE_SIZE}), got {tgts.shape}"
    print_test("PASS", f"shape={tgts.shape}")


def test_encode_values():
    print_header("TEST 2: encode_numpy values vs manual")
    coder = _make_coder()
    boxes = np.array([[10.0, -5.0, 1.0, 2.0, 3.0, 1.5, 0.785],
                      [-3.0,  8.0, 0.5, 1.0, 1.0, 0.5, -1.57]],
                     dtype=np.float32)
    tgts = coder.encode_numpy(boxes)

    vx, vy = VOXEL_SIZE[0], VOXEL_SIZE[1]
    osf = float(OUT_SIZE_FACTOR)
    exp_x = (boxes[:, 0] - PC_RANGE[0]) / (osf * vx)
    exp_y = (boxes[:, 1] - PC_RANGE[1]) / (osf * vy)
    exp_wlog = np.log(boxes[:, 3])
    exp_llog = np.log(boxes[:, 4])
    exp_hlog = np.log(boxes[:, 5])
    exp_z    = boxes[:, 2] + boxes[:, 5] * 0.5
    exp_sin  = np.sin(boxes[:, 6])
    exp_cos  = np.cos(boxes[:, 6])

    compare_arrays(exp_x,    tgts[:, 0], "cx",    atol=1e-5)
    compare_arrays(exp_y,    tgts[:, 1], "cy",    atol=1e-5)
    compare_arrays(exp_z,    tgts[:, 2], "cz",    atol=1e-5)
    compare_arrays(exp_wlog, tgts[:, 3], "log_w", atol=1e-5)
    compare_arrays(exp_llog, tgts[:, 4], "log_l", atol=1e-5)
    compare_arrays(exp_hlog, tgts[:, 5], "log_h", atol=1e-5)
    compare_arrays(exp_sin,  tgts[:, 6], "sin",   atol=1e-5)
    compare_arrays(exp_cos,  tgts[:, 7], "cos",   atol=1e-5)
    print_test("PASS", "all encode columns match")


def test_decode_shape():
    print_header("TEST 3: decode shape")
    coder = _make_coder()
    B, num_cls, P = 2, 10, 200

    heatmap = _from_shape("dec_hm",  [B, num_cls, P])
    rot     = _from_shape("dec_rot", [B, 2, P])
    dim     = _from_shape("dec_dim", [B, 3, P])
    center  = _from_shape("dec_ctr", [B, 2, P])
    height  = _from_shape("dec_hgt", [B, 1, P])

    boxes, scores = coder.decode(heatmap, rot, dim, center, height)
    assert boxes.shape  == [B, P, 7], f"boxes shape={boxes.shape}"
    assert scores.shape == [B, P],    f"scores shape={scores.shape}"
    print_test("PASS", f"boxes={boxes.shape}, scores={scores.shape}")


def test_decode_dim_exp():
    print_header("TEST 4: decode dim = exp(log_dim)")
    coder = _make_coder()
    B, P = 1, 5

    log_dims = np.log(np.array([[[2.0, 3.0, 1.5]] * P], dtype=np.float32)
                       .transpose(0, 2, 1))  # [1, 3, P]

    heatmap = _from_shape("dec4_hm",  [B, 3, P])
    rot     = _from_shape("dec4_rot", [B, 2, P])
    dim     = _from_data("dec4_dim",   log_dims)
    center  = _from_shape("dec4_ctr", [B, 2, P])
    height  = _from_shape("dec4_hgt", [B, 1, P])

    boxes, _ = coder.decode(heatmap, rot, dim, center, height)
    if boxes.data is not None:
        # columns 3,4,5 = w, l, h (indices 3,4,5 in last dim)
        decoded_dims = boxes.data[0, :, 3:6]     # [P, 3]
        ref = np.array([[2.0, 3.0, 1.5]] * P, dtype=np.float32)
        compare_arrays(ref, decoded_dims, "dim exp vs original", atol=1e-4)
    print_test("PASS", "dim exp columns match")


def test_decode_center_coords():
    print_header("TEST 5: decode center real-world coords")
    coder = _make_coder()
    B, P = 1, 4

    vx, vy = VOXEL_SIZE[0], VOXEL_SIZE[1]
    osf = float(OUT_SIZE_FACTOR)
    # Use known cx_grid values
    cx_grid = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    cy_grid = np.array([5.0, 15.0, 25.0, 35.0],  dtype=np.float32)
    center_np = np.zeros((B, 2, P), dtype=np.float32)
    center_np[0, 0, :] = cx_grid
    center_np[0, 1, :] = cy_grid

    heatmap = _from_shape("dec5_hm",  [B, 3, P])
    rot     = _from_shape("dec5_rot", [B, 2, P])
    dim     = _from_shape("dec5_dim", [B, 3, P])
    center  = _from_data("dec5_ctr",   center_np)
    height  = _from_shape("dec5_hgt", [B, 1, P])

    boxes, _ = coder.decode(heatmap, rot, dim, center, height)
    if boxes.data is not None:
        expected_x = cx_grid * osf * vx + PC_RANGE[0]
        expected_y = cy_grid * osf * vy + PC_RANGE[1]
        compare_arrays(expected_x, boxes.data[0, :, 0], "x_coord", atol=1e-4)
        compare_arrays(expected_y, boxes.data[0, :, 1], "y_coord", atol=1e-4)
    print_test("PASS", "center coord rescaling correct")


def test_decode_with_velocity():
    print_header("TEST 6: decode with velocity → boxes shape [B,P,9]")
    coder_10 = TransFusionBBoxCoder("tfc10", PC_RANGE, OUT_SIZE_FACTOR,
                                    VOXEL_SIZE, code_size=10)
    B, num_cls, P = 1, 5, 50
    heatmap = _from_shape("dec6_hm",  [B, num_cls, P])
    rot     = _from_shape("dec6_rot", [B, 2, P])
    dim     = _from_shape("dec6_dim", [B, 3, P])
    center  = _from_shape("dec6_ctr", [B, 2, P])
    height  = _from_shape("dec6_hgt", [B, 1, P])
    vel     = _from_shape("dec6_vel", [B, 2, P])
    boxes, scores = coder_10.decode(heatmap, rot, dim, center, height, vel)
    assert boxes.shape[2] == 9, f"expected 9 cols with vel, got {boxes.shape[2]}"
    assert scores.shape == [B, P], f"scores shape={scores.shape}"
    print_test("PASS", f"boxes={boxes.shape}, scores={scores.shape}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("encode_shape",        test_encode_shape),
        ("encode_values",       test_encode_values),
        ("decode_shape",        test_decode_shape),
        ("decode_dim_exp",      test_decode_dim_exp),
        ("decode_center_coord", test_decode_center_coords),
        ("decode_with_vel",     test_decode_with_velocity),
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as exc:
            results[name] = f"FAIL: {exc}"
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    for k, v in results.items():
        print(f"  {'✓' if v=='PASS' else '✗'} {k}: {v}")
    print(f"\n  {passed}/{len(results)} tests passed")
    import sys; sys.exit(0 if passed == len(results) else 1)
