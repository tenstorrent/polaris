#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for box3d_nms TTSim module.

Three test categories:
  1. Shape Validation  – circle_nms output is a list; aligned_3d_nms
                         output length <= N; box3d_multiclass_nms returns
                         correct tuple structure.
  2. Edge Case Creation – N=1, all same score, empty result when threshold
                         too high, large num_boxes.
  3. Data Validation   – circle_nms suppresses close boxes, keeps distant;
                         aligned_3d_nms suppresses identical boxes;
                         box3d_multiclass_nms score threshold and max_num.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_box3d_nms.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.box3d_nms import circle_nms, aligned_3d_nms, box3d_multiclass_nms

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(55)


def _spread_boxes(N, M=7):
    b = rng.randn(N, M).astype(np.float32)
    b[:, :3] *= 1000.0   # spread far apart
    return b


def _mlvl(N, nc, all_high=True, spread=True):
    bboxes = _spread_boxes(N) if spread else rng.randn(N, 7).astype(np.float32)
    bfn    = bboxes[:, :6]
    scores = np.zeros((N, nc + 1), dtype=np.float32)
    if all_high:
        scores[:, :nc] = rng.rand(N, nc).astype(np.float32) + 0.5
    else:
        scores[:, 0] = np.linspace(0.1, 0.9, N, dtype=np.float32)
    return bboxes, bfn, scores


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_box3d_nms_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True

    # circle_nms → list, all indices valid
    for i, N in enumerate([1, 5, 20]):
        try:
            dets = np.column_stack([rng.randn(N, 2) * 10,
                                    rng.rand(N).astype(np.float32) + 0.5
                                   ]).astype(np.float32)
            kept = circle_nms(dets, thresh=3.0, post_max_size=N)
            ok   = isinstance(kept, list) and all(0 <= k < N for k in kept)
            print(f"  [CN-{i:02d}] N={N:3d}  len(kept)={len(kept):3d}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [CN-{i:02d}] ERROR: {exc}"); all_passed = False

    # aligned_3d_nms → list len <= N
    for i, N in enumerate([1, 10, 50]):
        try:
            boxes   = rng.randn(N, 7).astype(np.float32)
            scores  = rng.rand(N).astype(np.float32)
            classes = np.zeros(N, dtype=np.int64)
            kept = aligned_3d_nms(boxes, scores, classes, thresh=0.3)
            ok   = isinstance(kept, (list, np.ndarray)) and len(kept) <= N
            print(f"  [AN-{i:02d}] N={N:3d}  len(kept)={len(kept):3d}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [AN-{i:02d}] ERROR: {exc}"); all_passed = False

    # box3d_multiclass_nms → 4-tuple with matching lengths
    for i, N in enumerate([5, 20]):
        try:
            bbs, bfn, sc = _mlvl(N, nc=3)
            res = box3d_multiclass_nms(bbs, bfn, sc, score_thr=0.0, max_num=100)
            ok  = len(res) == 4 and len(res[0]) == len(res[1]) == len(res[2])
            print(f"  [MN-{i:02d}] N={N:3d}  out={len(res[0])}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [MN-{i:02d}] ERROR: {exc}"); all_passed = False

    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_box3d_nms_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    # 1. Single detection → always kept
    dets1 = np.array([[5.0, 5.0, 0.9]], dtype=np.float32)
    kept1 = circle_nms(dets1, thresh=1.0)
    ok1 = len(kept1) == 1
    print(f"  [00] circle_nms single det kept: {'PASS' if ok1 else 'FAIL'}")
    if not ok1: all_passed = False

    # 2. All scores below threshold → empty result
    N = 10
    bbs, bfn, sc = _mlvl(N, nc=2, all_high=False, spread=True)
    sc[:] = 0.0   # zero scores
    res2 = box3d_multiclass_nms(bbs, bfn, sc, score_thr=0.5, max_num=100)
    ok2 = len(res2[0]) == 0
    print(f"  [01] all scores=0 → empty result: {'PASS' if ok2 else 'FAIL'}")
    if not ok2: all_passed = False

    # 3. max_num=1 → at most 1 box
    N = 50
    bbs3, bfn3, sc3 = _mlvl(N, nc=2, spread=True)
    res3 = box3d_multiclass_nms(bbs3, bfn3, sc3, score_thr=0.0, max_num=1)
    ok3 = len(res3[0]) <= 1
    print(f"  [02] max_num=1 → at most 1 box: {'PASS' if ok3 else 'FAIL'}  got {len(res3[0])}")
    if not ok3: all_passed = False

    # 4. circle_nms post_max_size limit
    N = 30
    dets4 = np.column_stack([rng.randn(N, 2) * 100,
                              rng.rand(N).astype(np.float32)]).astype(np.float32)
    kept4 = circle_nms(dets4, thresh=0.01, post_max_size=5)
    ok4 = len(kept4) <= 5
    print(f"  [03] circle_nms post_max_size=5: {'PASS' if ok4 else 'FAIL'}  kept={len(kept4)}")
    if not ok4: all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_box3d_nms_data_validation():
    """Category 3 – Data Validation."""
    all_passed = True

    # 1. circle_nms: highest-score box is always kept
    dets = np.array([[0.0, 0.0, 0.5],
                     [0.1, 0.1, 0.9],
                     [0.2, 0.2, 0.7]], dtype=np.float32)
    kept = circle_nms(dets, thresh=2.0, post_max_size=10)
    ok1  = 1 in kept   # idx 1 has highest score 0.9
    print(f"  [00] circle_nms top-score kept (idx 1): {'PASS' if ok1 else 'FAIL'}  kept={kept}")
    if not ok1: all_passed = False

    # 2. circle_nms: two distant points both kept
    dets2 = np.array([[0.0, 0.0, 0.8],
                      [100.0, 100.0, 0.7]], dtype=np.float32)
    kept2 = circle_nms(dets2, thresh=1.0, post_max_size=10)
    ok2 = len(kept2) == 2
    print(f"  [01] circle_nms distant both kept: {'PASS' if ok2 else 'FAIL'}  kept={kept2}")
    if not ok2: all_passed = False

    # 3. aligned_3d_nms: 5 identical boxes → 1 kept
    box = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0], dtype=np.float32)
    boxes3  = np.tile(box, (5, 1))
    scores3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    keep3   = aligned_3d_nms(boxes3, scores3, np.zeros(5, dtype=np.int64), thresh=0.3)
    ok3 = len(keep3) == 1
    print(f"  [02] aligned_3d_nms 5 identical → 1: {'PASS' if ok3 else 'FAIL'}  kept={keep3}")
    if not ok3: all_passed = False

    # 4. box3d_multiclass_nms scores >= threshold
    N = 30
    bbs, bfn, sc = _mlvl(N, nc=2, spread=True)
    threshold = 0.7
    res = box3d_multiclass_nms(bbs, bfn, sc, score_thr=threshold, max_num=100)
    if len(res[1]) > 0:
        ok4 = res[1].min() >= threshold - 1e-5
    else:
        ok4 = True  # empty is valid if nothing passes threshold
    print(f"  [03] multiclass scores >= {threshold}: {'PASS' if ok4 else 'FAIL'}  min_score={res[1].min() if len(res[1]) else 'N/A'}")
    if not ok4: all_passed = False

    assert all_passed
