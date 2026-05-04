#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for TransFusionDetector TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_transfusion_detector.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.transfusion_detector import TransFusionDetector
from ttsim_modules.transfusion_head import TransFusionHead
from reference.Validation.ttsim_utils import print_header, print_test


def test_tfd_forward():
    print_header("Test 1: TransFusionDetector forward (BEV → prediction dict)")
    head = TransFusionHead(
        "tfd_h",
        in_channels=384, hidden_channel=128, num_classes=10, num_proposals=200,
        num_decoder_layers=1, initialize_by_heatmap=True, fuse_img=False,
    )
    m    = TransFusionDetector("tfd", pts_bbox_head=head)
    bev  = _from_shape("tfd_bev", [1, 384, 124, 124])
    out  = m(bev)
    ok   = isinstance(out, dict) and "heatmap" in out
    print_test("TransFusionDetector output is prediction dict", f"keys={list(out.keys())}", ok)
    return ok


def test_tfd_no_head():
    print_header("Test 2: TransFusionDetector — no head: passthrough")
    m   = TransFusionDetector("tfd_nh")
    bev = _from_shape("tfd_nh_bev", [1, 384, 124, 124])
    out = m(bev)
    ok  = list(out.shape) == [1, 384, 124, 124]
    print_test("Passthrough when no head attached", f"shape={list(out.shape)}", ok)
    return ok


def test_tfd_params():
    print_header("Test 3: TransFusionDetector params == head params")
    head = TransFusionHead(
        "tfd_h2",
        in_channels=512, hidden_channel=128, num_classes=10, num_proposals=200,
        num_decoder_layers=1, initialize_by_heatmap=True, fuse_img=False,
    )
    m  = TransFusionDetector("tfd2", pts_bbox_head=head)
    p  = m.analytical_param_count()
    ok = p == head.analytical_param_count() and p > 0
    print_test("Param count delegates to head", f"p={p}", ok)
    return ok


if __name__ == "__main__":
    results = [test_tfd_forward(), test_tfd_no_head(), test_tfd_params()]
    n_pass  = sum(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{len(results)} tests passed")
    if n_pass < len(results):
        sys.exit(1)
