#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for HardSimpleVFE and HardSimpleVFE_ATT TTSim modules.

Three test categories:
  1. Shape Validation  – correct output shapes for various N/M/F combos.
  2. Edge Case Creation – minimal N, large batch, different num_features.
  3. Data Validation   – exact param count (4322) and _from_data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_voxel_encoder.py -v
"""

import os, sys, logging
import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.voxel_encoder import HardSimpleVFE, HardSimpleVFE_ATT

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(13)


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_shape_validation():
    """Category 1 – output shapes are correct for HardSimpleVFE and HardSimpleVFE_ATT."""
    all_passed = True

    # HardSimpleVFE: [N, M, F] → [N, F]
    vfe_cases = [
        ("N=100,M=10,F=4",  100, 10,  4),
        ("N=50,M=20,F=9",    50, 20,  9),
        ("N=200,M=5,F=4",   200,  5,  4),
        ("N=1,M=1,F=4",       1,  1,  4),
    ]
    for name, N, M, F in vfe_cases:
        try:
            vfe = HardSimpleVFE(f"vfe_sv_{N}_{M}_{F}", num_features=F)
            x   = _from_shape(f"vfe_sv_{N}_{M}_{F}_in", [N, M, F])
            o   = vfe(x)
            ok  = list(o.shape) == [N, F]
            print(f"  HardSimpleVFE {name}: got={list(o.shape)} exp=[{N},{F}]  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  HardSimpleVFE {name}: ERROR {exc}")
            all_passed = False

    # HardSimpleVFE_ATT: [N, M=10, 5] → [N, 32]
    # M is fixed at 10 (matches hardcoded dim_pa=10 inside VoxelFeature_TA)
    att_cases = [
        ("N=100,M=10", 100, 10),
        ("N=50,M=10",   50, 10),
        ("N=1,M=10",     1, 10),
    ]
    for name, N, M in att_cases:
        try:
            vfe_att = HardSimpleVFE_ATT(f"vfe_att_sv_{N}_{M}")
            x       = _from_shape(f"vfe_att_sv_{N}_{M}_in", [N, M, 5])
            o       = vfe_att(x)
            ok      = list(o.shape) == [N, 32]
            print(f"  HardSimpleVFE_ATT {name}: got={list(o.shape)} exp=[{N},32]  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  HardSimpleVFE_ATT {name}: ERROR {exc}")
            all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_edge_cases():
    """Category 2 – edge cases: N=1, large N, minimal features."""
    all_passed = True

    # Edge 1: HardSimpleVFE with different num_features
    for F in [4, 5, 9, 16]:
        try:
            vfe = HardSimpleVFE(f"vfe_ec_F{F}", num_features=F)
            x   = _from_shape(f"vfe_ec_F{F}_in", [10, 5, F])
            o   = vfe(x)
            p   = vfe.analytical_param_count()
            ok  = list(o.shape) == [10, F] and p == 0
            print(f"  HardSimpleVFE(F={F}): shape={list(o.shape)} params={p}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  HardSimpleVFE F={F} ERROR: {exc}")
            all_passed = False

    # Edge 2: HardSimpleVFE_ATT with N=1 (minimal batch)
    try:
        vfe_att = HardSimpleVFE_ATT("vfe_att_ec_N1")
        x       = _from_shape("vfe_att_ec_N1_in", [1, 10, 5])
        o       = vfe_att(x)
        ok      = list(o.shape) == [1, 32]
        print(f"  HardSimpleVFE_ATT N=1: shape={list(o.shape)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  HardSimpleVFE_ATT N=1 ERROR: {exc}")
        all_passed = False

    # Edge 3: HardSimpleVFE_ATT with large N
    try:
        vfe_att = HardSimpleVFE_ATT("vfe_att_ec_N1000")
        x       = _from_shape("vfe_att_ec_N1000_in", [1000, 10, 5])
        o       = vfe_att(x)
        ok      = list(o.shape) == [1000, 32]
        print(f"  HardSimpleVFE_ATT N=1000: shape={list(o.shape)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  HardSimpleVFE_ATT N=1000 ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_voxel_encoder_data_validation():
    """Category 3 – exact param counts and _from_data inputs."""
    all_passed = True

    # HardSimpleVFE: always 0 params
    for F in [4, 9, 16]:
        vfe = HardSimpleVFE(f"vfe_dv_F{F}", num_features=F)
        p   = vfe.analytical_param_count()
        ok  = p == 0
        print(f"  HardSimpleVFE(F={F}) params: got={p} exp=0  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # HardSimpleVFE_ATT: exact 4322 params
    vfe_att  = HardSimpleVFE_ATT("vfe_att_dv")
    got      = vfe_att.analytical_param_count()
    expected = 4322
    ok       = got == expected
    print(f"  HardSimpleVFE_ATT params: got={got} exp={expected}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_data input
    x_np    = rng.randn(50, 10, 5).astype(np.float32)
    vfe_att = HardSimpleVFE_ATT("vfe_att_dv_data")
    o       = vfe_att(_from_data("vfe_att_dv_x", x_np))
    ok      = list(o.shape) == [50, 32]
    print(f"  HardSimpleVFE_ATT _from_data: shape={list(o.shape)}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    assert all_passed
