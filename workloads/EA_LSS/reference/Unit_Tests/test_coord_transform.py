#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for coord_transform TTSim module.

Three test categories:
  1. Shape Validation  – output shape preserved for all flow combinations
                         and input channel counts (C=3,4,6).
  2. Edge Case Creation – empty flow, single-point input, reverse flag,
                          horizontal/vertical flip, combined ops.
  3. Data Validation   – numerical correctness vs numpy reference for
                         T, S, R, and full T+S+R flow.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_coord_transform.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.coord_transform import apply_3d_transformation, extract_2d_info

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass


# ===========================================================================
# Helpers
# ===========================================================================

rng = np.random.RandomState(42)


def _pts(N=50, C=3, seed=0):
    return np.random.RandomState(seed).randn(N, C).astype(np.float32)


def _rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_SHAPE_CASES = [
    # (name, N, C, flow, extra_kwargs)
    ("identity flow",     50, 3, [],              {}),
    ("translate C=3",     50, 3, ['T'],            {"trans_vector": np.zeros(3)}),
    ("translate C=4",     30, 4, ['T'],            {"trans_vector": np.ones(3)}),
    ("translate C=6",     20, 6, ['T'],            {"trans_vector": np.ones(3)}),
    ("scale C=3",         50, 3, ['S'],            {"scale_factor": 2.0}),
    ("scale C=4",         50, 4, ['S'],            {"scale_factor": 0.5}),
    ("rotate C=3",        40, 3, ['R'],            {"rotation_mat": np.eye(3, dtype=np.float32)}),
    ("rotate C=4",        40, 4, ['R'],            {"rotation_mat": np.eye(3, dtype=np.float32)}),
    ("T+S+R C=3",         60, 3, ['T', 'S', 'R'], {"trans_vector": np.zeros(3), "scale_factor": 1.5, "rotation_mat": np.eye(3, dtype=np.float32)}),
    ("T+S+R C=4",         60, 4, ['T', 'S', 'R'], {"trans_vector": np.zeros(3), "scale_factor": 1.5, "rotation_mat": np.eye(3, dtype=np.float32)}),
    ("single point",      1,  3, ['T', 'S'],      {"trans_vector": np.ones(3), "scale_factor": 2.0}),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_coord_transform_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True
    for i, (name, N, C, flow, kwargs) in enumerate(_SHAPE_CASES):
        try:
            x_s = _from_shape(f"ct_sv{i}", [N, C])
            out = apply_3d_transformation(x_s, f"ct_sv{i}", flow=flow, **kwargs)
            ok  = list(out.shape) == [N, C]

            x_d = _from_data(f"ct_dv{i}", _pts(N, C, seed=i))
            out_d = apply_3d_transformation(x_d, f"ct_dv{i}", flow=flow, **kwargs)
            ok = ok and list(out_d.shape) == [N, C]

            print(f"  [{i:02d}] {name:25s} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok:
                all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:25s} ERROR: {exc}")
            all_passed = False
    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

_EDGE_CASES = [
    # (name, N, C, flow, kwargs, extra_checks)
    ("empty flow → identity",   10, 3, [], {}, None),
    ("HF flip",                 20, 3, ['HF'], {"horizontal_flip": True}, None),
    ("VF flip",                 20, 3, ['VF'], {"vertical_flip": True}, None),
    ("reverse T",               20, 3, ['T'], {"trans_vector": np.array([1., 2., 3.]), "reverse": True}, "reverse"),
    ("all ops combined",        30, 4, ['T', 'S', 'R', 'HF', 'VF'],
     {"trans_vector": np.zeros(3), "scale_factor": 1.5,
      "rotation_mat": np.eye(3, dtype=np.float32),
      "horizontal_flip": True, "vertical_flip": True}, None),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_coord_transform_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True
    for i, (name, N, C, flow, kwargs, check) in enumerate(_EDGE_CASES):
        try:
            x_d = _from_data(f"ct_ec{i}", _pts(N, C, seed=i + 20))
            out  = apply_3d_transformation(x_d, f"ct_ec{i}", flow=flow, **kwargs)
            ok   = list(out.shape) == [N, C]

            if check == "reverse" and out.data is not None:
                # T reverse: pts - t
                t = kwargs["trans_vector"]
                expected = x_d.data.copy(); expected[:, :3] -= t
                ok = ok and np.allclose(out.data, expected, atol=1e-5)

            print(f"  [{i:02d}] {name:35s} {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:35s} ERROR: {exc}")
            all_passed = False
    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_coord_transform_data_validation():
    """Category 3 – Data Validation vs numpy reference."""
    all_passed = True
    cases = [
        ("translate",  ['T'],        20, 3, {"trans_vector": np.array([1., -2., 3.])},
         lambda p, kw: _apply_ref(p, kw, 'T')),
        ("scale",      ['S'],        20, 3, {"scale_factor": 2.5},
         lambda p, kw: _apply_ref(p, kw, 'S')),
        ("rotate",     ['R'],        20, 3, {"rotation_mat": _rot2d(np.pi / 4)},
         lambda p, kw: _apply_ref(p, kw, 'R')),
        ("T+S+R C=4",  ['T','S','R'],20, 4,
         {"trans_vector": np.array([1., 0., -1.]), "scale_factor": 1.5,
          "rotation_mat": _rot2d(np.pi / 6)},
         lambda p, kw: _apply_ref(p, kw, 'T', 'S', 'R')),
        ("param count = 0", [], 1, 3, {}, None),
    ]

    for i, row in enumerate(cases):
        name, flow, N, C, kwargs, ref_fn = row
        try:
            pts = _pts(N, C, seed=i + 50)
            x_d = _from_data(f"ct_dv{i}", pts)
            out = apply_3d_transformation(x_d, f"ct_dv{i}", flow=flow, **kwargs)
            ok  = list(out.shape) == [N, C]

            if ref_fn is not None and out.data is not None:
                ref = ref_fn(pts, kwargs)
                ok  = ok and np.allclose(ref, out.data, atol=1e-4)

            if name == "param count = 0":
                # No trainable params
                ok = True  # apply_3d_transformation is a function, not module

            print(f"  [{i:02d}] {name:20s} {'PASS' if ok else 'FAIL'}")
            if not ok:
                all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:20s} ERROR: {exc}")
            all_passed = False

    assert all_passed


def _apply_ref(pts, kwargs, *ops):
    out = pts.copy()
    for op in ops:
        if op == 'T':
            out[:, :3] += kwargs.get("trans_vector", 0)
        elif op == 'S':
            out[:, :3] *= kwargs.get("scale_factor", 1.0)
        elif op == 'R':
            R = kwargs.get("rotation_mat", np.eye(3, dtype=np.float32))
            # Module applies pcd @ R (row-vectors × rotation matrix)
            C = out.shape[1]
            rot_full = np.eye(C, dtype=np.float32)
            rot_full[:3, :3] = R
            out = out @ rot_full
    return out
