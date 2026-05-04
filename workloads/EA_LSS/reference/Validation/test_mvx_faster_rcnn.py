#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for MVXFasterRCNN and DynamicMVXFasterRCNN TTSim modules.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_mvx_faster_rcnn.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.mvx_faster_rcnn import MVXFasterRCNN, DynamicMVXFasterRCNN
from ttsim_modules.transfusion_head import TransFusionHead
from reference.Validation.ttsim_utils import print_header, print_test


def _build_head(tag, in_channels=384):
    return TransFusionHead(
        tag, in_channels=in_channels, hidden_channel=128, num_classes=10,
        num_proposals=200, num_decoder_layers=1, initialize_by_heatmap=True, fuse_img=False,
    )


def test_mvxfrcnn_forward():
    print_header("Test 1: MVXFasterRCNN forward shape (dict output)")
    head = _build_head("mfr_h1")
    m    = MVXFasterRCNN("mfr", pts_bbox_head=head)
    bev  = _from_shape("mfr_bev", [1, 384, 124, 124])
    out  = m(bev)
    ok   = isinstance(out, dict) and "heatmap" in out
    print_test("MVXFasterRCNN output is prediction dict", f"keys={list(out.keys())}", ok)
    return ok


def test_dynamic_mvxfrcnn():
    print_header("Test 2: DynamicMVXFasterRCNN forward shape")
    head = _build_head("dmfr_h1")
    m    = DynamicMVXFasterRCNN("dmfr", pts_bbox_head=head)
    bev  = _from_shape("dmfr_bev", [2, 384, 62, 62])
    out  = m(bev)
    ok   = isinstance(out, dict) and "heatmap" in out
    print_test("DynamicMVXFasterRCNN output is prediction dict",
               f"B=2 keys={list(out.keys())}", ok)
    return ok


def test_params_match_head():
    print_header("Test 3: MVXFasterRCNN params == head params")
    head = _build_head("mfr_h2")
    m    = MVXFasterRCNN("mfr2", pts_bbox_head=head)
    ok   = m.analytical_param_count() == head.analytical_param_count()
    print_test("param_count delegated to head",
               f"model={m.analytical_param_count()}, head={head.analytical_param_count()}", ok)
    return ok


if __name__ == "__main__":
    results = [test_mvxfrcnn_forward(), test_dynamic_mvxfrcnn(), test_params_match_head()]
    n_pass  = sum(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{len(results)} tests passed")
    if n_pass < len(results):
        sys.exit(1)
