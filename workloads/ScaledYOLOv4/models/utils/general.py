#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""General utility functions - TTSim compatible version"""

import glob
import math
import os
import random
import numpy as np
from loguru import logger

def init_seeds(seed=0):
    """Initialize random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def check_file(file):
    """Searches for file if not found locally"""
    if os.path.isfile(file) or file == "":
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)
        assert len(files), f"File Not Found: {file}"
        return files[0]


def make_divisible(x, divisor):
    """Returns x evenly divisible by divisor"""
    return math.ceil(x / divisor) * divisor


def check_anchor_order(m):
    """Check anchor order against stride order for YOLO Detect() module m

    Note: This is a simplified version for ttsim that doesn't modify anchors.
    In ttsim, anchors are numpy arrays stored as constants.
    """
    # Calculate anchor areas
    if hasattr(m, "anchor_grid_np"):
        a = (m.anchor_grid_np[..., 0] * m.anchor_grid_np[..., 1]).flatten()
        da = a[-1] - a[0]  # delta a

        if m.stride is not None:
            ds = m.stride[-1] - m.stride[0]  # delta s
            if (da > 0) != (ds > 0):  # opposite signs
                logger.debug(
                    "WARNING: Anchor order should be reversed but this is not supported in ttsim"
                )
                logger.debug("Please fix anchor order in YAML config file")


def xyxy2xywh(x):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]

    where xy1=top-left, xy2=bottom-right
    """
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]

    where xy1=top-left, xy2=bottom-right
    """
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape"""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    """Clip bounding xyxy bounding boxes to image shape (height, width)"""
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def coco80_to_coco91_class():
    """Converts 80-index (val2014) to 91-index (paper)"""
    x = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    return x


def labels_to_class_weights(labels, nc=80):
    """Get class weights (inverse frequency) from training labels"""
    if labels[0] is None:
        return np.array([])

    labels = np.concatenate(labels, 0)
    classes = labels[:, 0].astype(np.int32)
    weights = np.bincount(classes, minlength=nc)

    weights[weights == 0] = 1
    normalized_weights = 1.0 / weights
    normalized_weights /= normalized_weights.sum()
    return normalized_weights


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    """Produces image weights based on class mAPs"""
    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(np.int32), minlength=nc) for i in range(n)]
    )
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    return image_weights


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves"""
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [min(recall[-1] + 1e-3, 1.0)]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def ap_per_class(tp, conf, pred_cls, target_cls):
    """Compute the average precision, given the recall and precision curves

    Args:
        tp: True positives (nparray, nx1 or nx10)
        conf: Objectness value from 0-1 (nparray)
        pred_cls: Predicted object classes (nparray)
        target_cls: True object classes (nparray)

    Returns:
        The average precision as computed in py-faster-rcnn
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1
    s = [unique_classes.shape[0], tp.shape[1]]
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])

            # Precision
            precision = tpc / (tpc + fpc)
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")
