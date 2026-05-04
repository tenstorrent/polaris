#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for box3d_nms TTSim module.

Validates the TTSim conversion of
mmdet3d/core/post_processing/box3d_nms.py.

Test Coverage:
  1.  circle_nms ordering     – highest-score box always kept
  2.  circle_nms threshold    – distant boxes both kept
  3.  aligned_3d_nms IoU=1    – identical boxes suppress all but one
  4.  box3d_multiclass_nms shape – output tuple structure
  5.  box3d_multiclass_nms score threshold
  6.  box3d_multiclass_nms max_num
"""

import os
import sys

_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_val_dir      = os.path.dirname(__file__)
for p in (_polaris_root, _ealss_root, _val_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ttsim_modules.box3d_nms import (
    circle_nms, aligned_3d_nms, box3d_multiclass_nms
)
from ttsim_utils import print_header, print_test


# ============================================================================
# Tests
# ============================================================================

def test_circle_nms_keeps_highest_score():
    print_header("TEST 1: circle_nms keeps highest-score box")
    # Three overlapping points; highest score should be index 1
    dets = np.array([
        [0.0, 0.0, 0.5],   # score 0.5
        [0.1, 0.1, 0.9],   # score 0.9
        [0.2, 0.2, 0.7],   # score 0.7
    ], dtype=np.float32)
    kept = circle_nms(dets, thresh=2.0, post_max_size=100)
    assert 1 in kept, f"Highest score (idx 1) should be kept, got {kept}"
    print_test("PASS", f"kept indices={kept}")


def test_circle_nms_distant_boxes_both_kept():
    print_header("TEST 2: circle_nms keeps distant boxes")
    dets = np.array([
        [0.0,  0.0,  0.8],
        [100.0, 100.0, 0.7],
    ], dtype=np.float32)
    kept = circle_nms(dets, thresh=1.0, post_max_size=10)
    assert len(kept) == 2, f"Both boxes should be kept, got {kept}"
    print_test("PASS", f"kept={kept}")


def test_aligned_3d_nms_identical_boxes():
    print_header("TEST 3: aligned_3d_nms suppresses duplicates")
    # 5 identical boxes → only 1 should survive
    box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float32)
    boxes   = np.tile(box, (5, 1))
    scores  = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    classes = np.zeros(5, dtype=np.int32)
    keep = aligned_3d_nms(boxes, scores, classes, thresh=0.3)
    assert len(keep) == 1, f"Expected 1 kept box, got {len(keep)}: {keep}"
    assert scores[keep[0]] == 0.9, f"Best score should be kept: {scores[keep[0]]}"
    print_test("PASS", f"kept={keep}")


def test_box3d_multiclass_nms_output_structure():
    print_header("TEST 4: box3d_multiclass_nms output structure")
    rng = np.random.RandomState(42)
    N = 50
    num_cls = 3
    mlvl_bboxes = rng.randn(N, 7).astype(np.float32)
    mlvl_bboxes_for_nms = mlvl_bboxes[:, :6].astype(np.float32)  # first 6 cols
    # scores: [N, num_cls+1] — last column is background
    mlvl_scores = np.zeros((N, num_cls + 1), dtype=np.float32)
    mlvl_scores[:, :num_cls] = rng.rand(N, num_cls).astype(np.float32) + 0.6

    result = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                                  score_thr=0.0, max_num=100, nms_thr=0.5)
    assert isinstance(result, tuple), "Expected tuple"
    assert len(result) == 4, f"Expected 4-tuple (boxes,scores,labels,dir_scores), got {len(result)}"
    bboxes_out, scores_out, labels_out, dir_out = result
    assert bboxes_out.ndim  == 2 and bboxes_out.shape[1] == 7, f"boxes shape={bboxes_out.shape}"
    assert scores_out.ndim == 1, f"scores ndim={scores_out.ndim}"
    assert labels_out.ndim == 1, f"labels ndim={labels_out.ndim}"
    assert len(bboxes_out) == len(scores_out) == len(labels_out)
    print_test("PASS", f"out shape=({len(bboxes_out)}, 7)")


def test_box3d_multiclass_nms_score_threshold():
    print_header("TEST 5: box3d_multiclass_nms score threshold")
    rng = np.random.RandomState(7)
    N = 30
    num_cls = 2
    mlvl_bboxes = rng.randn(N, 7).astype(np.float32)
    mlvl_bboxes_for_nms = mlvl_bboxes[:, :6].astype(np.float32)
    mlvl_scores = np.zeros((N, num_cls + 1), dtype=np.float32)
    # Set varying scores for class 0
    mlvl_scores[:, 0] = np.linspace(0.1, 0.9, N, dtype=np.float32)
    threshold = 0.6
    result = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                                  score_thr=threshold, max_num=100, nms_thr=0.5)
    _, scores_out, _, _ = result
    if len(scores_out) > 0:
        assert scores_out.min() >= threshold - 1e-5, \
            f"All scores should be >= {threshold}, min={scores_out.min()}"
    print_test("PASS", f"kept {len(scores_out)} boxes above score={threshold}")


def test_box3d_multiclass_nms_max_num():
    print_header("TEST 6: box3d_multiclass_nms max_num cap")
    rng = np.random.RandomState(11)
    N = 200
    num_cls = 1
    # Spread boxes far apart so NMS keeps all → only cap via max_num
    mlvl_bboxes = rng.randn(N, 7).astype(np.float32)
    mlvl_bboxes[:, :3] *= 1000.0           # very spread out → no IoU suppression
    mlvl_bboxes_for_nms = mlvl_bboxes[:, :6].astype(np.float32)
    mlvl_scores = np.zeros((N, num_cls + 1), dtype=np.float32)
    mlvl_scores[:, 0] = rng.rand(N).astype(np.float32) + 0.5
    max_num = 10
    result = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_scores,
                                  score_thr=0.0, max_num=max_num, nms_thr=0.5)
    assert len(result[0]) <= max_num, \
        f"Output should be <= {max_num}, got {len(result[0])}"
    print_test("PASS", f"kept {len(result[0])} <= max_num={max_num}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("circle_nms_highest_score",  test_circle_nms_keeps_highest_score),
        ("circle_nms_distant",        test_circle_nms_distant_boxes_both_kept),
        ("aligned_3d_nms_identical",  test_aligned_3d_nms_identical_boxes),
        ("multiclass_structure",      test_box3d_multiclass_nms_output_structure),
        ("multiclass_score_thresh",   test_box3d_multiclass_nms_score_threshold),
        ("multiclass_max_num",        test_box3d_multiclass_nms_max_num),
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as exc:
            results[name] = f"FAIL: {exc}"
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    for k, v in results.items():
        print(f"  {'✓' if v=='PASS' else '✗'} {k}: {v}")
    print(f"\n  {passed}/{len(results)} tests passed")
    import sys; sys.exit(0 if passed == len(results) else 1)
