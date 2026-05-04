#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of 3-D NMS post-processing utilities.

Original file: mmdet3d/core/post_processing/box3d_nms.py

Provides pure-numpy reference implementations of:
  - circle_nms          : circular-distance NMS (no IoU; BEV distance).
  - aligned_3d_nms      : IoU-based 3-D NMS for axis-aligned boxes.
  - box3d_multiclass_nms: per-class NMS with score threshold + top-K.

These functions run **after** the detection head and are not part of the
differentiable forward graph.  They are therefore **not** expressed as
SimTensor graph ops.  Instead they are pure-numpy functions suitable for
validation scripts and post-processing pipelines.

The original code depends on:
  - mmdet3d.ops.iou3d.iou3d_utils.nms_gpu  (CUDA-only, not available in TTSim)
  - numba.jit  (optional; replaced with pure-numpy equivalent here)

All CUDA dependencies are replaced with CPU-only numpy implementations.

No torch / mmcv / numba imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np


# ---------------------------------------------------------------------------
# circle_nms  (pure numpy)
# ---------------------------------------------------------------------------

def circle_nms(dets: np.ndarray, thresh: float, post_max_size: int = 83) -> list:
    """
    Circular NMS — suppresses detections whose BEV centre-distance is
    within ``thresh`` of a higher-scoring detection.

    Args:
        dets (np.ndarray): [N, 3] array; columns are (x, y, score).
        thresh (float): Squared distance threshold (pass radius**2).
        post_max_size (int): Maximum number of detections to return.

    Returns:
        list[int]: Indices of kept detections (up to post_max_size).
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]

    order = np.argsort(scores)[::-1].astype(np.int32)
    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype=np.int32)
    keep = []

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(int(i))
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2
            if dist <= thresh:
                suppressed[j] = 1

    return keep[:post_max_size]


# ---------------------------------------------------------------------------
# _iou_aligned_3d  (internal helper)
# ---------------------------------------------------------------------------

def _iou_aligned_3d(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    """
    IoU between two axis-aligned 3-D boxes.

    Args:
        boxes_a (np.ndarray): [N, 6] — (x1,y1,z1,x2,y2,z2).
        boxes_b (np.ndarray): [M, 6].

    Returns:
        np.ndarray: [N, M] pairwise IoU.
    """
    area_a = np.prod(np.maximum(boxes_a[:, 3:6] - boxes_a[:, 0:3], 0.0), axis=1)  # [N]
    area_b = np.prod(np.maximum(boxes_b[:, 3:6] - boxes_b[:, 0:3], 0.0), axis=1)  # [M]

    inter_min = np.maximum(boxes_a[:, np.newaxis, 0:3],
                           boxes_b[np.newaxis, :, 0:3])  # [N,M,3]
    inter_max = np.minimum(boxes_a[:, np.newaxis, 3:6],
                           boxes_b[np.newaxis, :, 3:6])  # [N,M,3]
    inter_dims = np.maximum(inter_max - inter_min, 0.0)  # [N,M,3]
    inter_vol = np.prod(inter_dims, axis=-1)              # [N,M]

    union_vol = (area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter_vol)
    iou = inter_vol / np.maximum(union_vol, 1e-8)
    return iou


# ---------------------------------------------------------------------------
# aligned_3d_nms  (pure numpy)
# ---------------------------------------------------------------------------

def aligned_3d_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    thresh: float,
) -> np.ndarray:
    """
    IoU-based NMS for axis-aligned 3-D bounding boxes.

    Boxes from different classes are never suppressed against each other
    (class-aware NMS).

    Args:
        boxes   (np.ndarray): [N, 6] — (x1,y1,z1,x2,y2,z2).
        scores  (np.ndarray): [N] confidence scores.
        classes (np.ndarray): [N] integer class labels.
        thresh  (float): IoU suppression threshold.

    Returns:
        np.ndarray: 1-D int64 array of kept box indices.
    """
    order = np.argsort(scores)[::-1].astype(np.int64)
    pick = []

    while order.size > 0:
        i = order[0]
        pick.append(i)
        if order.size == 1:
            break
        rest = order[1:]

        box_i = boxes[i: i + 1]   # [1, 6]
        box_r = boxes[rest]        # [R, 6]
        iou = _iou_aligned_3d(box_i, box_r)[0]  # [R]

        # Suppress same-class boxes with iou > thresh
        same_cls = classes[rest] == classes[i]
        suppress = same_cls & (iou > thresh)
        order = rest[~suppress]

    return np.array(pick, dtype=np.int64)


# ---------------------------------------------------------------------------
# box3d_multiclass_nms  (pure numpy)
# ---------------------------------------------------------------------------

def box3d_multiclass_nms(
    mlvl_bboxes: np.ndarray,
    mlvl_bboxes_for_nms: np.ndarray,
    mlvl_scores: np.ndarray,
    score_thr: float,
    max_num: int,
    use_rotate_nms: bool = False,
    nms_thr: float = 0.2,
    mlvl_dir_scores: np.ndarray = None,
):
    """
    Multi-class NMS for 3-D bounding boxes.

    Replaces ``mmdet3d.core.post_processing.box3d_multiclass_nms``.
    Uses ``aligned_3d_nms`` (numpy IoU) instead of ``nms_gpu`` (CUDA).
    Rotate-NMS is approximated by the same aligned NMS (BEV IoU available
    only via CUDA; this provides a shape-correct CPU fallback).

    Args:
        mlvl_bboxes        : [N, M]  final boxes (all attributes).
        mlvl_bboxes_for_nms: [N, 4] or [N, 6]  boxes used for IoU.
        mlvl_scores        : [N, num_cls+1]  class scores + background.
        score_thr          : float  minimum score to keep.
        max_num            : int    maximum output detections.
        use_rotate_nms     : bool   (not used in CPU fallback).
        nms_thr            : float  IoU suppression threshold.
        mlvl_dir_scores    : [N] optional direction scores.

    Returns:
        tuple:
            bboxes     np.ndarray [K, M]
            scores     np.ndarray [K]
            labels     np.ndarray [K] int64
            dir_scores np.ndarray [K] or empty
    """
    num_classes = mlvl_scores.shape[1] - 1
    all_bboxes = []
    all_scores = []
    all_labels = []
    all_dir_scores = []

    for cls_id in range(num_classes):
        cls_mask = mlvl_scores[:, cls_id] > score_thr
        if not cls_mask.any():
            continue

        _scores = mlvl_scores[cls_mask, cls_id]       # [K]
        _bboxes = mlvl_bboxes[cls_mask]               # [K, M]
        _nms_bboxes = mlvl_bboxes_for_nms[cls_mask]   # [K, 4 or 6]

        # Pad to 6 columns if only 4 are given (BEV → add dummy z extent)
        if _nms_bboxes.shape[1] == 4:
            z_min = np.zeros((_nms_bboxes.shape[0], 1), dtype=np.float32)
            z_max = np.ones((_nms_bboxes.shape[0], 1), dtype=np.float32)
            _nms_bboxes = np.concatenate(
                [_nms_bboxes[:, :2], z_min,
                 _nms_bboxes[:, 2:4], z_max], axis=1
            )

        _classes = np.full(len(_scores), cls_id, dtype=np.int64)
        selected = aligned_3d_nms(_nms_bboxes, _scores, _classes, nms_thr)

        all_bboxes.append(_bboxes[selected])
        all_scores.append(_scores[selected])
        all_labels.append(np.full(len(selected), cls_id, dtype=np.int64))

        if mlvl_dir_scores is not None:
            all_dir_scores.append(mlvl_dir_scores[cls_mask][selected])

    if all_bboxes:
        bboxes = np.concatenate(all_bboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        if len(bboxes) > max_num:
            top_idx = np.argsort(scores)[::-1][:max_num]
            bboxes = bboxes[top_idx]
            scores = scores[top_idx]
            labels = labels[top_idx]

        if mlvl_dir_scores is not None:
            dir_scores = np.concatenate(all_dir_scores, axis=0)
            if len(dir_scores) > max_num:
                dir_scores = dir_scores[top_idx]
        else:
            dir_scores = np.zeros(0, dtype=np.float32)
    else:
        bboxes     = np.zeros((0, mlvl_bboxes.shape[-1]), dtype=np.float32)
        scores     = np.zeros(0, dtype=np.float32)
        labels     = np.zeros(0, dtype=np.int64)
        dir_scores = np.zeros(0, dtype=np.float32)

    return bboxes, scores, labels, dir_scores
