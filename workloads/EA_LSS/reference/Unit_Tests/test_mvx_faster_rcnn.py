#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MVXFasterRCNN and DynamicMVXFasterRCNN TTSim modules.

Three test categories:
  1. Shape Validation  – both classes forward to prediction dict.
  2. Edge Case Creation – minimal head, batch=2.
  3. Data Validation   – params delegated to attached head.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_mvx_faster_rcnn.py -v
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
from ttsim_modules.mvx_faster_rcnn import MVXFasterRCNN, DynamicMVXFasterRCNN
from ttsim_modules.transfusion_head import TransFusionHead

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(17)


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
def test_mvx_faster_rcnn_shape_validation():
    """Category 1 – MVXFasterRCNN and DynamicMVXFasterRCNN output shapes."""
    all_passed = True

    for cls, cname in [(MVXFasterRCNN, "MVXFasterRCNN"),
                        (DynamicMVXFasterRCNN, "DynamicMVXFasterRCNN")]:
        try:
            h   = _head(f"mfr_sv_{cname[:5]}")
            m   = cls(f"mfr_sv_{cname[:5]}", pts_bbox_head=h)
            bev = _from_shape(f"mfr_bev_{cname[:5]}", [1, 384, 124, 124])
            out = m(bev)
            ok  = isinstance(out, dict) and "heatmap" in out
            print(f"  {cname} dict keys={sorted(out.keys())} {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {cname}: ERROR {exc}"); all_passed = False

    assert all_passed, "Shape validation failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_mvx_faster_rcnn_edge_cases():
    """Category 2 – No head (passthrough) and batch=2."""
    all_passed = True

    # No head → passthrough
    for cls, cname in [(MVXFasterRCNN, "MFRCNN"), (DynamicMVXFasterRCNN, "DMFRCNN")]:
        try:
            m   = cls(f"mfr_ec_nh_{cname}")
            bev = _from_shape(f"mfr_ec_bev_nh_{cname}", [1, 384, 64, 64])
            out = m(bev)
            ok  = list(out.shape) == [1, 384, 64, 64]
            print(f"  {cname} no-head passthrough: {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {cname} no-head: ERROR {exc}"); all_passed = False

    # Batch=2
    try:
        h   = _head("mfr_ec_b2")
        m   = MVXFasterRCNN("mfr_ec_b2", pts_bbox_head=h)
        bev = _from_shape("mfr_ec_bev_b2", [2, 384, 62, 62])
        out = m(bev)
        ok  = isinstance(out, dict)
        print(f"  B=2 dict: {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  B=2: ERROR {exc}"); all_passed = False

    assert all_passed, "Edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_mvx_faster_rcnn_data_validation():
    """Category 3 – Param count equals head, _from_data BEV input."""
    all_passed = True

    h  = _head("mfr_dv_h")
    m  = MVXFasterRCNN("mfr_dv", pts_bbox_head=h)
    p  = m.analytical_param_count()
    ph = h.analytical_param_count()
    ok = p == ph and p > 0
    print(f"  params {p:,} == head {ph:,}: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    try:
        h2  = _head("mfr_dv_h2")
        m2  = MVXFasterRCNN("mfr_dv2", pts_bbox_head=h2)
        bev = _from_shape("mfr_dv_bev", [1, 384, 32, 32])
        out = m2(bev)
        ok2 = isinstance(out, dict)
        print(f"  _from_shape → dict: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_shape: ERROR {exc}"); all_passed = False

    assert all_passed, "Data validation failed"
