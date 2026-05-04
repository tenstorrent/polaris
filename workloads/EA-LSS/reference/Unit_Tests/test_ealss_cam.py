#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for EALSS_CAM TTSim module.

Three test categories:
  1. Shape Validation  – camera-only forward output is prediction dict.
  2. Edge Case Creation – imc variants, lc_fusion=True.
  3. Data Validation   – analytical_param_count ≈ 72.8M.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_ealss_cam.py -v
"""

import os, sys, logging
import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for p in [polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.ealss_cam import EALSS_CAM

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(55)

# Expected params for default EALSS_CAM (lc_fusion=False, camera_stream=True, imc=512)
EXPECTED_PARAMS = 72_768_614


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_cam_shape_validation():
    """Category 1 – EALSS_CAM forward output is prediction dict with canonical keys."""
    all_passed = True

    cases = [
        ("B=1,N=6,256x704",  [6,  3, 256, 704]),
        ("B=1,N=6,128x352",  [6,  3, 128, 352]),
    ]
    for name, img_shape in cases:
        try:
            m   = EALSS_CAM(f"ecam_sv_{name[:6]}")
            img = _from_shape(f"ecam_sv_img_{name[:6]}", img_shape)
            out = m(img)
            ok  = isinstance(out, dict) and "heatmap" in out
            print(f"  {name}: keys={sorted(out.keys())} {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {name}: ERROR {exc}"); all_passed = False

    assert all_passed, "Shape validation failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_cam_edge_cases():
    """Category 2 – imc=256, lc_fusion=True, both affect param count."""
    all_passed = True

    # imc=512 > imc=256
    try:
        m_256 = EALSS_CAM("ecam_ec_imc256", imc=256)
        m_512 = EALSS_CAM("ecam_ec_imc512", imc=512)
        ok    = m_512.analytical_param_count() > m_256.analytical_param_count()
        print(f"  imc=512 > imc=256 params: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  imc comparison: ERROR {exc}"); all_passed = False

    # lc_fusion=True adds LiDAR modules
    try:
        m_nofus = EALSS_CAM("ecam_ec_nofus", lc_fusion=False)
        m_fused = EALSS_CAM("ecam_ec_fused", lc_fusion=True)
        ok      = m_fused.analytical_param_count() > m_nofus.analytical_param_count()
        print(f"  lc_fusion=True adds params: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  lc_fusion: ERROR {exc}"); all_passed = False

    # lc_fusion=True still forwards correctly
    try:
        m   = EALSS_CAM("ecam_ec_fwd_fus", lc_fusion=True)
        img = _from_shape("ecam_ec_fus_img", [6, 3, 256, 704])
        out = m(img)
        ok  = isinstance(out, dict) and "heatmap" in out
        print(f"  lc_fusion=True forward dict: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  lc_fusion=True forward: ERROR {exc}"); all_passed = False

    assert all_passed, "Edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_ealss_cam_data_validation():
    """Category 3 – analytical_param_count == exact expected value."""
    all_passed = True

    m  = EALSS_CAM("ecam_dv")
    p  = m.analytical_param_count()
    ok = p == EXPECTED_PARAMS
    print(f"  params={p:,} expected={EXPECTED_PARAMS:,}: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_shape forward pass (fast)
    try:
        m2  = EALSS_CAM("ecam_dv2")
        img = _from_shape("ecam_dv_img", [6, 3, 256, 704])
        out = m2(img)
        ok2 = isinstance(out, dict)
        print(f"  _from_shape forward → dict: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_shape forward: ERROR {exc}"); all_passed = False

    assert all_passed, "Data validation failed"
