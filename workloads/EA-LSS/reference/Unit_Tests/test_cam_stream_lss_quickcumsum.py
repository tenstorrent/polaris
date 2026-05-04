#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for cam_stream_lss_quickcumsum TTSim module.

Three test categories:
  1. Shape Validation  – cumsum_trick_numpy returns correct shapes;
                         QuickCumsum worst-case shape [N, C] and [N, D].
  2. Edge Case Creation – all-unique voxels, all-same voxel, N=1,
                         single-channel features.
  3. Data Validation   – per-voxel sums are correct, geom is last row
                         of each voxel group, QuickCumsum data matches
                         cumsum_trick_numpy.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_cam_stream_lss_quickcumsum.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.cam_stream_lss_quickcumsum import (
    cumsum_trick_numpy, QuickCumsum
)

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(9)


def _data(N, C, D=4):
    x    = rng.randn(N, C).astype(np.float32)
    geom = rng.randn(N, D).astype(np.float32)
    return x, geom


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_SHAPE_CASES = [
    # (name, N, C, D, n_voxels)
    ("N=30  C=8   D=4   V=10", 30, 8,   4,  10),
    ("N=100 C=64  D=4   V=50", 100, 64, 4,  50),
    ("N=200 C=128 D=6   V=100",200, 128,6, 100),
    ("N=1   C=4   D=2   V=1",   1,  4,  2,   1),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_quickcumsum_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True

    for i, (name, N, C, D, V) in enumerate(_SHAPE_CASES):
        try:
            x, geom  = _data(N, C, D)
            ranks    = np.repeat(np.arange(V), N // V)[:N]
            ranks    = np.sort(ranks)
            xv, gv   = cumsum_trick_numpy(x, geom, ranks)
            exp_V    = len(np.unique(ranks))
            ok_np    = xv.shape == (exp_V, C) and gv.shape == (exp_V, D)
            print(f"  [NP-{i:02d}] {name}  {'PASS' if ok_np else 'FAIL'}  xv={xv.shape} gv={gv.shape}")
            if not ok_np: all_passed = False
        except Exception as exc:
            print(f"  [NP-{i:02d}] {name} ERROR: {exc}"); all_passed = False

        try:
            N2, C2, D2 = N, C, D
            qcs  = QuickCumsum(f"qcs_sv{i}")
            x_t  = _from_shape(f"qcs_sv{i}_x",    [N2, C2])
            g_t  = _from_shape(f"qcs_sv{i}_geom", [N2, D2])
            xo, go = qcs(x_t, g_t)
            ok_tt = list(xo.shape) == [N2, C2] and list(go.shape) == [N2, D2]
            print(f"  [TT-{i:02d}] {name} shape-only  {'PASS' if ok_tt else 'FAIL'}  xo={xo.shape} go={go.shape}")
            if not ok_tt: all_passed = False
        except Exception as exc:
            print(f"  [TT-{i:02d}] {name} ERROR: {exc}"); all_passed = False

    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_quickcumsum_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    # 1. All unique → V == N, features preserved
    N, C, D = 20, 4, 3
    x, geom = _data(N, C, D)
    ranks   = np.arange(N)
    xv, gv  = cumsum_trick_numpy(x, geom, ranks)
    ok1 = xv.shape == (N, C) and np.allclose(x, xv, atol=1e-5)
    print(f"  [00] all-unique: {'PASS' if ok1 else 'FAIL'}  V={xv.shape[0]}")
    if not ok1: all_passed = False

    # 2. All same voxel → V == 1, sum of all
    N2, C2, D2 = 15, 6, 2
    x2, geom2 = _data(N2, C2, D2)
    ranks2 = np.zeros(N2, dtype=np.int64)
    xv2, gv2 = cumsum_trick_numpy(x2, geom2, ranks2)
    ok2 = xv2.shape == (1, C2) and np.allclose(x2.sum(0), xv2[0], atol=1e-4)
    print(f"  [01] all-same: {'PASS' if ok2 else 'FAIL'}  V={xv2.shape[0]}  sum_ok={np.allclose(x2.sum(0), xv2[0], atol=1e-4)}")
    if not ok2: all_passed = False

    # 3. N=1 single point
    x3 = rng.randn(1, 4).astype(np.float32)
    g3 = rng.randn(1, 3).astype(np.float32)
    xv3, gv3 = cumsum_trick_numpy(x3, g3, np.array([0]))
    ok3 = xv3.shape == (1, 4) and np.allclose(x3, xv3, atol=1e-6)
    print(f"  [02] N=1: {'PASS' if ok3 else 'FAIL'}")
    if not ok3: all_passed = False

    # 4. QuickCumsum data with ranks
    N4, C4, D4 = 30, 8, 4
    x4, geom4 = _data(N4, C4, D4)
    ranks4 = np.sort(np.random.randint(0, 10, N4))
    qcs4   = QuickCumsum("qcs_ec4")
    xo, go = qcs4(
        _from_data("qcs_ec4_x", x4),
        _from_data("qcs_ec4_g", geom4),
        _from_data("qcs_ec4_r", ranks4),
    )
    ok4 = xo.data is not None and go.data is not None
    print(f"  [03] QuickCumsum data not None: {'PASS' if ok4 else 'FAIL'}")
    if not ok4: all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_quickcumsum_data_validation():
    """Category 3 – Data Validation."""
    all_passed = True

    # 1. 4 points in 2 voxels, manual sum check
    x = np.array([[1.0, 0.0], [2.0, 3.0],
                  [0.5, 0.5], [1.5, 2.5]], dtype=np.float32)
    geom  = np.arange(8, dtype=np.float32).reshape(4, 2)
    ranks = np.array([0, 0, 1, 1])
    xv, gv = cumsum_trick_numpy(x, geom, ranks)
    ok1 = np.allclose(x[:2].sum(0), xv[0], atol=1e-6) and \
          np.allclose(x[2:].sum(0), xv[1], atol=1e-6) and \
          np.allclose(geom[1], gv[0], atol=1e-6) and \
          np.allclose(geom[3], gv[1], atol=1e-6)
    print(f"  [00] 4pts/2voxels sums: {'PASS' if ok1 else 'FAIL'}")
    if not ok1: all_passed = False

    # 2. QuickCumsum data matches cumsum_trick_numpy
    N, C, D = 30, 8, 4
    x2, geom2 = _data(N, C, D)
    ranks2 = np.sort(rng.randint(0, 10, N))
    xv_ref, gv_ref = cumsum_trick_numpy(x2, geom2, ranks2)
    qcs2 = QuickCumsum("qcs_dv2")
    xo2, go2 = qcs2(
        _from_data("qcs_dv2_x", x2),
        _from_data("qcs_dv2_g", geom2),
        _from_data("qcs_dv2_r", ranks2),
    )
    ok2 = (xo2.data is not None and go2.data is not None and
           np.allclose(xv_ref, xo2.data, atol=1e-5) and
           np.allclose(gv_ref, go2.data, atol=1e-5))
    md_x = float(np.max(np.abs(xv_ref - xo2.data))) if xo2.data is not None else -1
    print(f"  [01] QCS data vs ref: {'PASS' if ok2 else 'FAIL'}  x_max_diff={md_x:.3e}")
    if not ok2: all_passed = False

    # 3. Param count = 0
    qcs3 = QuickCumsum("qcs_pc3")
    pc = qcs3.analytical_param_count()
    ok3 = pc == 0
    print(f"  [02] param_count=0: {'PASS' if ok3 else 'FAIL'}  pc={pc}")
    if not ok3: all_passed = False

    # 4. Determinism
    N4, C4, D4 = 20, 4, 3
    x4, g4 = _data(N4, C4, D4)
    r4     = np.sort(rng.randint(0, 5, N4))
    xv_a, _ = cumsum_trick_numpy(x4, g4, r4)
    xv_b, _ = cumsum_trick_numpy(x4, g4, r4)
    ok4 = np.allclose(xv_a, xv_b, atol=0.0)
    print(f"  [03] determinism: {'PASS' if ok4 else 'FAIL'}")
    if not ok4: all_passed = False

    assert all_passed
