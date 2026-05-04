#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for MVXTwoStageDetector TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_mvx_two_stage.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.mvx_two_stage import MVXTwoStageDetector
from ttsim_modules.transfusion_head import TransFusionHead
from reference.Validation.ttsim_utils import print_header, print_test


def test_empty_mvx():
    print_header("Test 1: MVXTwoStageDetector with no sub-modules")
    m   = MVXTwoStageDetector("mvx_empty")
    p   = m.analytical_param_count()
    x   = _from_shape("mvx_x", [1, 384, 32, 32])
    out = m(x)
    ok_params  = p == 0
    ok_shape   = list(out.shape) == [1, 384, 32, 32]
    ok = ok_params and ok_shape
    print_test("No sub-modules: params==0, passthrough OK", f"p={p}, out={list(out.shape)}", ok)
    return ok


def test_mvx_with_head():
    print_header("Test 2: MVXTwoStageDetector with pts_bbox_head only")
    head = TransFusionHead(
        "mvx2_head",
        in_channels=384, hidden_channel=128,
        num_classes=10, num_proposals=200,
        num_decoder_layers=1, initialize_by_heatmap=True, fuse_img=False,
    )
    m    = MVXTwoStageDetector("mvx2", pts_bbox_head=head)
    p    = m.analytical_param_count()
    bev  = _from_shape("mvx2_bev", [1, 384, 124, 124])
    out  = m(bev)
    ok_params = p == head.analytical_param_count()
    ok_keys   = isinstance(out, dict) and "heatmap" in out
    ok = ok_params and ok_keys
    print_test("MVX delegates __call__ to head, params correct",
               f"p={p}, keys={list(out.keys())}", ok)
    return ok


def test_mvx_param_sum():
    print_header("Test 3: MVXTwoStageDetector param sum across head variants")
    head_small = TransFusionHead("mvx3_h1", in_channels=256, hidden_channel=64,
                                  num_classes=5, num_proposals=100, num_decoder_layers=1,
                                  initialize_by_heatmap=True, fuse_img=False)
    m   = MVXTwoStageDetector("mvx3", pts_bbox_head=head_small)
    p   = m.analytical_param_count()
    ok  = p == head_small.analytical_param_count() and p > 0
    print_test("Param sum matches child", f"p={p}", ok)
    return ok


if __name__ == "__main__":
    results = [test_empty_mvx(), test_mvx_with_head(), test_mvx_param_sum()]
    n_pass  = sum(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{len(results)} tests passed")
    if n_pass < len(results):
        sys.exit(1)
