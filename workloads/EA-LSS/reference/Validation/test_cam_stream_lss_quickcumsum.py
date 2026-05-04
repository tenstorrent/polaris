#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for cam_stream_lss_quickcumsum TTSim module.

Validates the TTSim conversion of QuickCumsum in
mmdet3d/models/detectors/cam_stream_lss.py (lines 86-130).

Test Coverage:
  1.  cumsum_trick_numpy shape  – output [V, C] and [V, D]
  2.  cumsum_trick_numpy values – manual per-voxel sum check
  3.  all-unique voxels         – V == N, no aggregation
  4.  all-same voxel            – V == 1, sum of all rows
  5.  QuickCumsum shape         – worst-case [N, C] and [N, D]
  6.  QuickCumsum data          – matches cumsum_trick_numpy
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
from ttsim_modules.cam_stream_lss_quickcumsum import (
    cumsum_trick_numpy, QuickCumsum
)
from ttsim_utils import compare_arrays, print_header, print_test


# ============================================================================
# Tests
# ============================================================================

def test_cumsum_numpy_shape():
    print_header("TEST 1: cumsum_trick_numpy shape")
    N, C, D = 30, 8, 4
    rng = np.random.RandomState(3)
    x         = rng.randn(N, C).astype(np.float32)
    geom      = rng.randn(N, D).astype(np.float32)
    # 10 unique voxels, 3 points each
    ranks     = np.repeat(np.arange(10), 3)
    xv, gv = cumsum_trick_numpy(x, geom, ranks)
    assert xv.shape == (10, C), f"Expected (10,{C}), got {xv.shape}"
    assert gv.shape == (10, D), f"Expected (10,{D}), got {gv.shape}"
    print_test("PASS", f"xv={xv.shape}, gv={gv.shape}")


def test_cumsum_numpy_values():
    print_header("TEST 2: cumsum_trick_numpy values")
    # 4 points in 2 voxels: voxel 0 has rows 0,1 | voxel 1 has rows 2,3
    x = np.array([[1.0, 0.0],
                  [2.0, 3.0],
                  [0.5, 0.5],
                  [1.5, 2.5]], dtype=np.float32)
    geom  = np.arange(8, dtype=np.float32).reshape(4, 2)
    ranks = np.array([0, 0, 1, 1])
    xv, gv = cumsum_trick_numpy(x, geom, ranks)
    assert xv.shape == (2, 2), f"Expected (2,2), got {xv.shape}"
    expected_v0 = x[:2].sum(axis=0)
    expected_v1 = x[2:].sum(axis=0)
    compare_arrays(expected_v0, xv[0], "voxel 0 sum", atol=1e-6)
    compare_arrays(expected_v1, xv[1], "voxel 1 sum", atol=1e-6)
    # geom should be last row of each voxel
    compare_arrays(geom[1], gv[0], "geom voxel 0", atol=1e-6)
    compare_arrays(geom[3], gv[1], "geom voxel 1", atol=1e-6)
    print_test("PASS", "values match manual sum")


def test_cumsum_all_unique():
    print_header("TEST 3: all-unique voxels (V == N)")
    N, C, D = 20, 4, 3
    rng = np.random.RandomState(77)
    x    = rng.randn(N, C).astype(np.float32)
    geom = rng.randn(N, D).astype(np.float32)
    ranks = np.arange(N)   # each point is its own voxel
    xv, gv = cumsum_trick_numpy(x, geom, ranks)
    assert xv.shape == (N, C), f"Expected ({N},{C}), got {xv.shape}"
    compare_arrays(x, xv, "unique voxel features preserved", atol=1e-5)
    print_test("PASS", f"shape={xv.shape}")


def test_cumsum_all_same_voxel():
    print_header("TEST 4: all-same voxel (V == 1)")
    N, C, D = 15, 6, 2
    rng = np.random.RandomState(55)
    x    = rng.randn(N, C).astype(np.float32)
    geom = rng.randn(N, D).astype(np.float32)
    ranks = np.zeros(N, dtype=np.int64)
    xv, gv = cumsum_trick_numpy(x, geom, ranks)
    assert xv.shape == (1, C), f"Expected (1,{C}), got {xv.shape}"
    compare_arrays(x.sum(axis=0), xv[0], "total sum", atol=1e-4)
    compare_arrays(geom[-1], gv[0], "geom last row", atol=1e-5)
    print_test("PASS", f"shape={xv.shape}")


def test_quickcumsum_worst_case_shape():
    print_header("TEST 5: QuickCumsum worst-case shape")
    N, C, D = 100, 64, 4
    qcs = QuickCumsum("qcs_s")
    x_t    = _from_shape("qcs_x_s",    [N, C])
    geom_t = _from_shape("qcs_geom_s", [N, D])
    x_out, g_out = qcs(x_t, geom_t)
    # Shape inference returns worst-case [N, C] and [N, D]
    assert list(x_out.shape) == [N, C], f"Expected [{N},{C}], got {x_out.shape}"
    assert list(g_out.shape) == [N, D], f"Expected [{N},{D}], got {g_out.shape}"
    print_test("PASS", f"x_out={x_out.shape}, g_out={g_out.shape}")


def test_quickcumsum_data():
    print_header("TEST 6: QuickCumsum data vs cumsum_trick_numpy")
    N, C, D = 30, 8, 4
    rng = np.random.RandomState(17)
    x_np    = rng.randn(N, C).astype(np.float32)
    geom_np = rng.randn(N, D).astype(np.float32)
    ranks_np = np.repeat(np.arange(10), 3)

    qcs = QuickCumsum("qcs_d")
    x_t    = _from_data("qcs_x_d",    x_np)
    geom_t = _from_data("qcs_geom_d", geom_np)
    ranks_t = _from_data("qcs_ranks_d", ranks_np)

    x_out, g_out = qcs(x_t, geom_t, ranks_t)
    assert x_out.data is not None, "QuickCumsum x_out.data is None"
    assert g_out.data is not None, "QuickCumsum g_out.data is None"

    xv_ref, gv_ref = cumsum_trick_numpy(x_np, geom_np, ranks_np)
    compare_arrays(xv_ref, x_out.data, "x_voxel vs ref", atol=1e-5)
    compare_arrays(gv_ref, g_out.data, "geom_out vs ref", atol=1e-5)
    print_test("PASS", f"actual shape x={x_out.data.shape}, g={g_out.data.shape}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("cumsum_shape",             test_cumsum_numpy_shape),
        ("cumsum_values",            test_cumsum_numpy_values),
        ("all_unique",               test_cumsum_all_unique),
        ("all_same",                 test_cumsum_all_same_voxel),
        ("quickcumsum_shape",        test_quickcumsum_worst_case_shape),
        ("quickcumsum_data",         test_quickcumsum_data),
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
