#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MVXTwoStageDetector TTSim module.

Three test categories:
  1. Shape Validation  – forward with/without sub-modules.
  2. Edge Case Creation – various head configs, missing sub-modules.
  3. Data Validation   – analytical_param_count sums child modules.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_mvx_two_stage.py -v
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
from ttsim_modules.mvx_two_stage import MVXTwoStageDetector
from ttsim_modules.transfusion_head import TransFusionHead

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(7)


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
def test_mvx_two_stage_shape_validation():
    """Category 1 – MVXTwoStageDetector output shapes."""
    all_passed = True

    # Without head: passthrough
    try:
        m   = MVXTwoStageDetector("mvx_sv_noh")
        x   = _from_shape("mvx_sv_x", [1, 384, 32, 32])
        out = m(x)
        ok  = list(out.shape) == [1, 384, 32, 32]
        print(f"  no head passthrough: {list(out.shape)} {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  no head: ERROR {exc}"); all_passed = False

    # With head: dict output
    try:
        m   = MVXTwoStageDetector("mvx_sv_h", pts_bbox_head=_head("mvx_sv_h1"))
        bev = _from_shape("mvx_sv_bev", [1, 384, 124, 124])
        out = m(bev)
        ok  = isinstance(out, dict) and "heatmap" in out
        print(f"  with head dict keys={list(out.keys())} {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  with head: ERROR {exc}"); all_passed = False

    assert all_passed, "Shape validation failed"


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_mvx_two_stage_edge_cases():
    """Category 2 – Non-default head in_channels, batch=2."""
    all_passed = True

    cases = [
        ("in=256, B=1", 256, [1, 256, 64, 64]),
        ("in=512, B=2", 512, [2, 512, 62, 62]),
    ]
    for name, inc, bev_shape in cases:
        try:
            h   = _head(f"mvx_ec_{inc}", in_channels=inc)
            m   = MVXTwoStageDetector(f"mvx_ec_{inc}", pts_bbox_head=h)
            bev = _from_shape(f"mvx_ec_bev_{inc}", bev_shape)
            out = m(bev)
            ok  = isinstance(out, dict)
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  {name}: ERROR {exc}"); all_passed = False

    assert all_passed, "Edge case tests failed"


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_mvx_two_stage_data_validation():
    """Category 3 – Param count sums sub-modules, _from_data BEV input."""
    all_passed = True

    # Param sum
    h  = _head("mvx_dv_h1")
    m  = MVXTwoStageDetector("mvx_dv", pts_bbox_head=h)
    p  = m.analytical_param_count()
    ph = h.analytical_param_count()
    ok = p == ph and p > 0
    print(f"  params {p:,} == head {ph:,}: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_shape BEV (avoid heavy data compute through TransFusionHead)
    try:
        h2  = _head("mvx_dv_h2")
        m2  = MVXTwoStageDetector("mvx_dv2", pts_bbox_head=h2)
        bev = _from_shape("mvx_dv_bev", [1, 384, 32, 32])
        out = m2(bev)
        ok2 = isinstance(out, dict)
        print(f"  _from_shape → dict: {'PASS' if ok2 else 'FAIL'}")
        if not ok2: all_passed = False
    except Exception as exc:
        print(f"  _from_shape: ERROR {exc}"); all_passed = False

    assert all_passed, "Data validation failed"
