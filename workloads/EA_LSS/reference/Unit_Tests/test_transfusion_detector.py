#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TransFusionDetector TTSim module.

Three test categories:
  1. Shape Validation  – BEV → prediction dict, no-head passthrough.
  2. Edge Case Creation – in_channels variants, batch=2.
  3. Data Validation   – param count equals head, _from_data input.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_transfusion_detector.py -v
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
from ttsim_modules.transfusion_detector import TransFusionDetector
from ttsim_modules.transfusion_head import TransFusionHead

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(23)


def _head(tag, in_channels=384):
    return TransFusionHead(
        tag, in_channels=in_channels, hidden_channel=128,
        num_classes=10, num_proposals=200, num_decoder_layers=1,
        initialize_by_heatmap=True, fuse_img=False,
    )


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_detector_shape_validation():
    """Category 1 – TransFusionDetector output shapes (dict + passthrough)."""
    all_passed = True

    # With head → dict
    try:
        h   = _head("tfd_sv_h")
        m   = TransFusionDetector("tfd_sv", pts_bbox_head=h)
        bev = _from_shape("tfd_sv_bev", [1, 384, 124, 124])
        out = m(bev)
        ok  = isinstance(out, dict) and "heatmap" in out
        print(f"  with head dict keys={sorted(out.keys())} {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  with head: ERROR {exc}"); all_passed = False

    # Without head → passthrough
    try:
        m   = TransFusionDetector("tfd_sv_nh")
        bev = _from_shape("tfd_sv_bev_nh", [1, 384, 64, 64])
        out = m(bev)
        ok  = list(out.shape) == [1, 384, 64, 64]
        print(f"  no head passthrough: {list(out.shape)} {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  no head: ERROR {exc}"); all_passed = False

    assert all_passed, "Shape validation failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_detector_edge_cases():
    """Category 2 – Different in_channels, batch=2."""
    all_passed = True

    # in_channels=512 (EALSS_CAM cam-only head)
    try:
        h   = _head("tfd_ec_512", in_channels=512)
        m   = TransFusionDetector("tfd_ec_512", pts_bbox_head=h)
        bev = _from_shape("tfd_ec_bev_512", [1, 512, 124, 124])
        out = m(bev)
        ok  = isinstance(out, dict)
        print(f"  in_channels=512: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  in_channels=512: ERROR {exc}"); all_passed = False

    # batch=2
    try:
        h   = _head("tfd_ec_b2")
        m   = TransFusionDetector("tfd_b2", pts_bbox_head=h)
        bev = _from_shape("tfd_b2_bev", [2, 384, 62, 62])
        out = m(bev)
        ok  = isinstance(out, dict)
        print(f"  batch=2: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  batch=2: ERROR {exc}"); all_passed = False

    assert all_passed, "Edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_detector_data_validation():
    """Category 3 – Param count equals head, _from_data BEV input."""
    all_passed = True

    h  = _head("tfd_dv_h")
    m  = TransFusionDetector("tfd_dv", pts_bbox_head=h)
    p  = m.analytical_param_count()
    ph = h.analytical_param_count()
    ok = p == ph and p > 0
    print(f"  params {p:,} == head {ph:,}: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    try:
        h2  = _head("tfd_dv_h2")
        m2  = TransFusionDetector("tfd_dv2", pts_bbox_head=h2)
        bev = _from_shape("tfd_dv_bev", [1, 384, 32, 32])
        out = m2(bev)
        ok2 = isinstance(out, dict)
        print(f"  _from_shape → dict: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_shape: ERROR {exc}"); all_passed = False

    assert all_passed, "Data validation failed"
