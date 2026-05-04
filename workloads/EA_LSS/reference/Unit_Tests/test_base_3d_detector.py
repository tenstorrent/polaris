#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Base3DDetector TTSim module.

Three test categories:
  1. Shape Validation  – passthrough output shape.
  2. Edge Case Creation – various input dimensions.
  3. Data Validation   – analytical_param_count == 0.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_base_3d_detector.py -v
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
from ttsim_modules.base_3d_detector import Base3DDetector

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_base_3d_detector_shape_validation():
    """Category 1 – Base3DDetector passthrough output shapes."""
    all_passed = True

    cases = [
        ("BEV [1,3,64,64]",   [1, 3, 64, 64]),
        ("BEV [2,256,32,32]", [2, 256, 32, 32]),
        ("1D  [1,128,200]",   [1, 128, 200]),
    ]
    for name, shape in cases:
        try:
            m   = Base3DDetector(f"b3d_sv_{name[:6]}")
            x   = _from_shape(f"b3d_x_{name[:6]}", shape)
            out = m(x)
            ok  = list(out.shape) == shape
            print(f"  {name}: out={list(out.shape)} {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
            all_passed = False

    assert all_passed, "One or more shape validation tests failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_base_3d_detector_edge_cases():
    """Category 2 – Edge case inputs (batch=1, single pixel, 5D)."""
    all_passed = True

    cases = [
        ("scalar proxy [1,1,1,1]", [1, 1, 1, 1]),
        ("large batch B=4",        [4, 64, 32, 32]),
        ("channels only [1,512]",  [1, 512]),
    ]
    for name, shape in cases:
        try:
            m   = Base3DDetector(f"b3d_ec_{name[:9]}")
            x   = _from_shape(f"b3d_ec_x_{name[:6]}", shape)
            out = m(x)
            ok  = list(out.shape) == shape
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {name}: ERROR {exc}")
            all_passed = False

    assert all_passed, "One or more edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_base_3d_detector_data_validation():
    """Category 3 – analytical_param_count == 0 and _from_data passthrough."""
    all_passed = True

    # Param count
    m  = Base3DDetector("b3d_dv_params")
    p  = m.analytical_param_count()
    ok = p == 0
    print(f"  params==0: got={p} {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_data passthrough
    try:
        arr = rng.randn(1, 32, 32).astype(np.float32)
        m2  = Base3DDetector("b3d_dv_data")
        x   = _from_data("b3d_x_data", arr)
        out = m2(x)
        ok2 = list(out.shape) == [1, 32, 32]
        print(f"  _from_data passthrough: shape={list(out.shape)} {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_data passthrough: ERROR {exc}")
        all_passed = False

    assert all_passed, "One or more data validation tests failed"
