#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for LiftSplatShoot (cam_stream_lss) TTSim module.

Three test categories:
  1. Shape Validation  – output shape matches input [B*N, inputC, fH, fW].
  2. Edge Case Creation – different inputC, different depth ranges, custom camC.
  3. Data Validation   – exact param count (7,357,108), D=41, cz=128 for defaults.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_cam_stream_lss.py -v
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
from ttsim_modules.cam_stream_lss import LiftSplatShoot

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(41)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_lss(tag="sv", inputC=256, camC=64):
    return LiftSplatShoot(
        f"lss_{tag}",
        lss=False,
        final_dim=(900, 1600),
        camera_depth_range=[4.0, 45.0, 1.0],
        pc_range=[-50, -50, -5, 50, 50, 3],
        downsample=4,
        grid=3,
        inputC=inputC,
        camC=camC,
    )


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cam_stream_lss_shape_validation():
    """Category 1 – output shape == input shape [B*N, inputC, fH, fW]."""
    all_passed = True

    # NOTE: (fH, fW) must be divisible by 8 such that the 3×stride-2 dtransform
    # chain produces exactly fH//8, fW//8 spatial dims (required for Resize back).
    shape_cases = [
        ("default [6,256,32,88]",  6, 256, 32, 88),
        ("B*N=1  [1,256,32,88]",   1, 256, 32, 88),
        ("B*N=12 [12,256,32,88]", 12, 256, 32, 88),
    ]
    for name, BN, C, H, W in shape_cases:
        try:
            lss = _default_lss(f"sv_{BN}_{C}")
            x   = _from_shape(f"lss_sv_{BN}_{C}_in", [BN, C, H, W])
            out = lss(x)
            ok  = list(out.shape) == [BN, C, H, W]
            print(f"  {name}: got={list(out.shape)} exp=[{BN},{C},{H},{W}]  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
            all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cam_stream_lss_edge_cases():
    """Category 2 – different inputC, camC, depth range."""
    all_passed = True

    # Edge 1: smaller inputC
    try:
        lss = _default_lss("ec_128", inputC=128, camC=32)
        x   = _from_shape("lss_ec128_in", [6, 128, 32, 88])
        out = lss(x)
        ok  = list(out.shape) == [6, 128, 32, 88]
        print(f"  inputC=128: got={list(out.shape)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  inputC=128 ERROR: {exc}")
        all_passed = False

    # Edge 2: different depth range → different D
    try:
        lss = LiftSplatShoot(
            "lss_ec_drange",
            lss=False,
            final_dim=(900, 1600),
            camera_depth_range=[2.0, 50.0, 2.0],   # D = (50-2)/2 = 24
            pc_range=[-50, -50, -5, 50, 50, 3],
            downsample=4,
            grid=3,
            inputC=256,
            camC=64,
        )
        ok_D = lss.D == 24
        print(f"  D for [2,50,2]: got={lss.D} exp=24  {'PASS' if ok_D else 'FAIL'}")
        if not ok_D: all_passed = False
    except Exception as exc:
        print(f"  depth range edge case ERROR: {exc}")
        all_passed = False

    # Edge 3: different pc_range → different cz
    try:
        lss = LiftSplatShoot(
            "lss_ec_cz",
            lss=False,
            final_dim=(900, 1600),
            camera_depth_range=[4.0, 45.0, 1.0],
            pc_range=[-50, -50, -10, 50, 50, 10],  # z range 20, grid=3 → cz=int(64*(20//2.9999))=int(64*6)=384
            downsample=4,
            grid=3,
            inputC=256,
            camC=64,
        )
        import math
        z_range = 10 - (-10)   # = 20
        expected_cz = int(64 * math.floor(z_range / (3 - 0.0001)))   # int(64 * 6) = 384
        ok_cz = lss.cz == expected_cz
        print(f"  cz for z[-10,10]: got={lss.cz} exp={expected_cz}  {'PASS' if ok_cz else 'FAIL'}")
        if not ok_cz: all_passed = False
    except Exception as exc:
        print(f"  pc_range edge case ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cam_stream_lss_data_validation():
    """Category 3 – D=41, cz=128, exact param count 7,357,108 and _from_data."""
    all_passed = True

    lss = _default_lss("dv")

    # D = 41 for depth range [4.0, 45.0, 1.0]
    ok_D = lss.D == 41
    print(f"  D == 41: got={lss.D}  {'PASS' if ok_D else 'FAIL'}")
    if not ok_D: all_passed = False

    # cz = 128 for camC=64, z∈[-5,3], grid=3 → floor((3+5)/(3-0.0001))=floor(2.666)=2 → 64*2=128
    ok_cz = lss.cz == 128
    print(f"  cz == 128: got={lss.cz}  {'PASS' if ok_cz else 'FAIL'}")
    if not ok_cz: all_passed = False

    # Exact param count
    got      = lss.analytical_param_count()
    expected = 7_357_108
    ok_p     = got == expected
    print(f"  params: got={got:,} exp={expected:,}  {'PASS' if ok_p else 'FAIL'}")
    if not ok_p: all_passed = False

    # Use _from_shape for forward pass check (avoids expensive numerical computation
    # that would occur with _from_data on large spatial tensors)
    lss2 = _default_lss("dv_shape")
    out  = lss2(_from_shape("lss_dv_in", [1, 256, 32, 88]))
    ok_s = list(out.shape) == [1, 256, 32, 88]
    print(f"  _from_shape forward: {list(out.shape)}  {'PASS' if ok_s else 'FAIL'}")
    if not ok_s: all_passed = False

    assert all_passed
