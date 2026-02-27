#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim conversion of util/box_ops.py

Bounding box utilities for object detection:
  - Coordinate transforms: cxcywh ↔ xyxy
  - Box area computation
  - IoU (Intersection over Union)
  - GIoU (Generalized IoU)
  - Mask to box extraction

Design: Utility functions that accept and return SimTensor objects.
Uses NumPy for numerical computation with proper SimTensor wrapping.
"""

import os, sys
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from ttsim.ops.tensor import SimTensor

# ══════════════════════════════════════════════════════════════════════════════
# COORDINATE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════


def box_cxcywh_to_xyxy(x):
    """
    Convert boxes from (center_x, center_y, width, height) to (x0, y0, x1, y1).

    Mirrors PyTorch's box_cxcywh_to_xyxy.

    Args:
        x: SimTensor or ndarray [..., 4]

    Returns:
        SimTensor [..., 4] in xyxy format
    """
    # Extract shape and data
    if isinstance(x, SimTensor):
        x_shape = x.shape
        x_data = x.data
        dtype = x.dtype if hasattr(x, "dtype") else np.float32
    else:
        x_data = x if isinstance(x, np.ndarray) else np.asarray(x)
        x_shape = list(x_data.shape) if hasattr(x_data, "shape") else None
        dtype = x_data.dtype if hasattr(x_data, "dtype") else np.float32

    # Shape inference: output shape same as input
    output_shape = x_shape

    # Numerical computation if data available
    if x_data is not None:
        x_c = x_data[..., 0]
        y_c = x_data[..., 1]
        w = x_data[..., 2]
        h = x_data[..., 3]

        x0 = x_c - 0.5 * w
        y0 = y_c - 0.5 * h
        x1 = x_c + 0.5 * w
        y1 = y_c + 0.5 * h

        out = np.stack([x0, y0, x1, y1], axis=-1).astype(np.float32)
        return SimTensor(
            {
                "name": "box_xyxy",
                "shape": list(out.shape),
                "data": out,
                "dtype": np.float32,
            }
        )
    else:
        # Shape inference only
        return SimTensor(
            {"name": "box_xyxy", "shape": output_shape, "data": None, "dtype": dtype}
        )


def box_xyxy_to_cxcywh(x):
    """
    Convert boxes from (x0, y0, x1, y1) to (center_x, center_y, width, height).

    Mirrors PyTorch's box_xyxy_to_cxcywh.

    Args:
        x: SimTensor or ndarray [..., 4]

    Returns:
        SimTensor [..., 4] in cxcywh format
    """
    # Extract shape and data
    if isinstance(x, SimTensor):
        x_shape = x.shape
        x_data = x.data
        dtype = x.dtype if hasattr(x, "dtype") else np.float32
    else:
        x_data = x if isinstance(x, np.ndarray) else np.asarray(x)
        x_shape = list(x_data.shape) if hasattr(x_data, "shape") else None
        dtype = x_data.dtype if hasattr(x_data, "dtype") else np.float32

    # Shape inference: output shape same as input
    output_shape = x_shape

    # Numerical computation if data available
    if x_data is not None:
        x0 = x_data[..., 0]
        y0 = x_data[..., 1]
        x1 = x_data[..., 2]
        y1 = x_data[..., 3]

        x_c = (x0 + x1) / 2.0
        y_c = (y0 + y1) / 2.0
        w = x1 - x0
        h = y1 - y0

        out = np.stack([x_c, y_c, w, h], axis=-1).astype(np.float32)
        return SimTensor(
            {
                "name": "box_cxcywh",
                "shape": list(out.shape),
                "data": out,
                "dtype": np.float32,
            }
        )
    else:
        # Shape inference only
        return SimTensor(
            {"name": "box_cxcywh", "shape": output_shape, "data": None, "dtype": dtype}
        )


def masks_to_boxes(masks):
    """
    Compute bounding boxes around binary masks.

    Mirrors PyTorch masks_to_boxes:
        - Creates coordinate grids
        - Applies mask to filter coordinates
        - Finds min/max for each mask

    Design: Shape inference always performed, numerical computation when data available.

    Args:
        masks : SimTensor or ndarray  [N, H, W]

    Returns:
        SimTensor [N, 4] in (x_min, y_min, x_max, y_max)
    """
    # Extract shape and data
    if isinstance(masks, SimTensor):
        masks_shape = masks.shape
        masks_data = masks.data
        dtype = masks.dtype if hasattr(masks, "dtype") else np.float32
    else:
        masks_shape = list(masks.shape) if hasattr(masks, "shape") else None
        masks_data = masks if isinstance(masks, np.ndarray) else np.asarray(masks)
        dtype = masks_data.dtype if hasattr(masks_data, "dtype") else np.float32

    # Shape inference: [N, H, W] → [N, 4]
    N = masks_shape[0] if masks_shape else 0
    output_shape = [N, 4]

    # Numerical computation if data available
    if masks_data is not None:
        # Check if empty (matching PyTorch's masks.numel() == 0)
        num_elements = (
            masks.numel() if isinstance(masks, SimTensor) else masks_data.size
        )
        if num_elements == 0:
            out = np.zeros((0, 4), dtype=np.float32)
            return SimTensor(
                {
                    "name": "boxes",
                    "shape": list(out.shape),
                    "data": out,
                    "dtype": np.float32,
                }
            )

        N, h, w = masks_data.shape

        # Create coordinate grids
        y = np.arange(h, dtype=np.float32)
        x = np.arange(w, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")

        # Broadcast masks to match coordinate grids
        x_mask = masks_data * xx[np.newaxis, :, :]
        y_mask = masks_data * yy[np.newaxis, :, :]

        # Find min/max coordinates
        x_max = x_mask.reshape(N, -1).max(axis=1)
        x_min = np.where(masks_data, x_mask, 1e8).reshape(N, -1).min(axis=1)
        y_max = y_mask.reshape(N, -1).max(axis=1)
        y_min = np.where(masks_data, y_mask, 1e8).reshape(N, -1).min(axis=1)

        out = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)
        return SimTensor(
            {
                "name": "boxes",
                "shape": list(out.shape),
                "data": out,
                "dtype": np.float32,
            }
        )
    else:
        # Shape inference only
        return SimTensor(
            {"name": "boxes", "shape": output_shape, "data": None, "dtype": dtype}
        )


# ══════════════════════════════════════════════════════════════════════════════
# BOX AREA AND IOU OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════


def box_area(boxes):
    """
    Compute area of boxes in (x0, y0, x1, y1) format.

    Mirrors torchvision.ops.boxes.box_area:
        area = (x1 - x0) * (y1 - y0)

    Design: Shape inference always performed, numerical computation when data available.

    Args:
        boxes : SimTensor or ndarray  [N, 4]

    Returns:
        SimTensor [N] with box areas
    """
    # Extract shape and data
    if isinstance(boxes, SimTensor):
        boxes_shape = boxes.shape
        boxes_data = boxes.data
        dtype = boxes.dtype if hasattr(boxes, "dtype") else np.float32
    else:
        boxes_shape = list(boxes.shape) if hasattr(boxes, "shape") else None
        boxes_data = boxes if isinstance(boxes, np.ndarray) else np.asarray(boxes)
        dtype = boxes_data.dtype if hasattr(boxes_data, "dtype") else np.float32

    # Shape inference: [N, 4] → [N]
    N = boxes_shape[0] if boxes_shape else 0
    output_shape = [N]

    # Numerical computation if data available
    if boxes_data is not None:
        # area = (x1 - x0) * (y1 - y0)
        out = (boxes_data[:, 2] - boxes_data[:, 0]) * (
            boxes_data[:, 3] - boxes_data[:, 1]
        )
        return SimTensor(
            {
                "name": "box_area",
                "shape": list(out.shape),
                "data": out.astype(np.float32),
                "dtype": np.float32,
            }
        )
    else:
        # Shape inference only
        return SimTensor(
            {"name": "box_area", "shape": output_shape, "data": None, "dtype": dtype}
        )


def box_iou(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes (both in xyxy format).

    Mirrors PyTorch box_iou:
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        union = area1[:, None] + area2 - inter
        iou = inter / union

    Design: Shape inference always performed, numerical computation when data available.

    Args:
        boxes1 : SimTensor or ndarray  [N, 4]
        boxes2 : SimTensor or ndarray  [M, 4]

    Returns:
        (iou, union) both SimTensor [N, M]
    """
    # Extract shapes and data
    if isinstance(boxes1, SimTensor):
        b1_shape = boxes1.shape
        b1_data = boxes1.data
        dtype = boxes1.dtype if hasattr(boxes1, "dtype") else np.float32
    else:
        b1_shape = list(boxes1.shape) if hasattr(boxes1, "shape") else None
        b1_data = boxes1 if isinstance(boxes1, np.ndarray) else np.asarray(boxes1)
        dtype = b1_data.dtype if hasattr(b1_data, "dtype") else np.float32

    if isinstance(boxes2, SimTensor):
        b2_shape = boxes2.shape
        b2_data = boxes2.data
    else:
        b2_shape = list(boxes2.shape) if hasattr(boxes2, "shape") else None
        b2_data = boxes2 if isinstance(boxes2, np.ndarray) else np.asarray(boxes2)

    # Shape inference: [N, 4] × [M, 4] → [N, M]
    N = b1_shape[0] if b1_shape else 0
    M = b2_shape[0] if b2_shape else 0
    output_shape = [N, M]

    # Numerical computation if data available
    if b1_data is not None and b2_data is not None:
        # Compute areas
        area1 = (b1_data[:, 2] - b1_data[:, 0]) * (b1_data[:, 3] - b1_data[:, 1])  # [N]
        area2 = (b2_data[:, 2] - b2_data[:, 0]) * (b2_data[:, 3] - b2_data[:, 1])  # [M]

        # Compute intersection
        lt = np.maximum(b1_data[:, None, :2], b2_data[:, :2])  # [N, M, 2]
        rb = np.minimum(b1_data[:, None, 2:], b2_data[:, 2:])  # [N, M, 2]

        wh = np.maximum(rb - lt, 0)  # [N, M, 2] clamp to 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # Compute union
        union = area1[:, None] + area2 - inter  # [N, M]

        # Compute IoU
        iou_data = inter / union  # [N, M]

        iou_tensor = SimTensor(
            {
                "name": "iou",
                "shape": list(iou_data.shape),
                "data": iou_data.astype(np.float32),
                "dtype": np.float32,
            }
        )
        union_tensor = SimTensor(
            {
                "name": "union",
                "shape": list(union.shape),
                "data": union.astype(np.float32),
                "dtype": np.float32,
            }
        )
        return iou_tensor, union_tensor
    else:
        # Shape inference only
        iou_tensor = SimTensor(
            {"name": "iou", "shape": output_shape, "data": None, "dtype": dtype}
        )
        union_tensor = SimTensor(
            {"name": "union", "shape": output_shape, "data": None, "dtype": dtype}
        )
        return iou_tensor, union_tensor


def generalized_box_iou(boxes1, boxes2):
    """
    Compute Generalized IoU (GIoU) between two sets of boxes (both in xyxy format).

    Mirrors PyTorch generalized_box_iou:
        iou, union = box_iou(boxes1, boxes2)
        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        area = wh[:, :, 0] * wh[:, :, 1]
        return iou - (area - union) / area

    Design: Shape inference always performed, numerical computation when data available.

    Args:
        boxes1 : SimTensor or ndarray  [N, 4]
        boxes2 : SimTensor or ndarray  [M, 4]

    Returns:
        SimTensor [N, M] with GIoU values
    """
    # Extract shapes and data
    if isinstance(boxes1, SimTensor):
        b1_shape = boxes1.shape
        b1_data = boxes1.data
        dtype = boxes1.dtype if hasattr(boxes1, "dtype") else np.float32
    else:
        b1_shape = list(boxes1.shape) if hasattr(boxes1, "shape") else None
        b1_data = boxes1 if isinstance(boxes1, np.ndarray) else np.asarray(boxes1)
        dtype = b1_data.dtype if hasattr(b1_data, "dtype") else np.float32

    if isinstance(boxes2, SimTensor):
        b2_shape = boxes2.shape
        b2_data = boxes2.data
    else:
        b2_shape = list(boxes2.shape) if hasattr(boxes2, "shape") else None
        b2_data = boxes2 if isinstance(boxes2, np.ndarray) else np.asarray(boxes2)

    # Shape inference: [N, 4] × [M, 4] → [N, M]
    N = b1_shape[0] if b1_shape else 0
    M = b2_shape[0] if b2_shape else 0
    output_shape = [N, M]

    # Numerical computation if data available
    if b1_data is not None and b2_data is not None:
        # Compute IoU and union
        iou_tensor, union_tensor = box_iou(
            SimTensor(
                {"name": "b1", "shape": b1_shape, "data": b1_data, "dtype": np.float32}
            ),
            SimTensor(
                {"name": "b2", "shape": b2_shape, "data": b2_data, "dtype": np.float32}
            ),
        )
        iou_data = iou_tensor.data
        union_data = union_tensor.data

        # Compute enclosing box area
        lt = np.minimum(b1_data[:, None, :2], b2_data[:, :2])  # [N, M, 2]
        rb = np.maximum(b1_data[:, None, 2:], b2_data[:, 2:])  # [N, M, 2]

        wh = np.maximum(rb - lt, 0)  # [N, M, 2]
        area = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

        # GIoU formula: IoU - (area_enclosing - union) / area_enclosing
        giou_data = iou_data - (area - union_data) / area  # [N, M]

        return SimTensor(
            {
                "name": "giou",
                "shape": list(giou_data.shape),
                "data": giou_data.astype(np.float32),
                "dtype": np.float32,
            }
        )
    else:
        # Shape inference only
        return SimTensor(
            {"name": "giou", "shape": output_shape, "data": None, "dtype": dtype}
        )


# import os, sys
# import numpy as np
# from types import SimpleNamespace

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# from ttsim.ops.desc.data_compute import (
#     compute_box_cxcywh_to_xyxy,
#     compute_box_xyxy_to_cxcywh,
#     compute_box_area,
#     compute_box_iou,
#     compute_generalized_box_iou,
#     compute_masks_to_boxes,
# )


# def box_cxcywh_to_xyxy(x):
#     """
#     Convert boxes from (center_x, center_y, width, height) to (x0, y0, x1, y1).

#     Args:
#         x : SimpleNamespace or ndarray  [..., 4]

#     Returns:
#         SimpleNamespace(shape, data, dtype)
#     """
#     x_data = x.data if isinstance(x, SimpleNamespace) else np.asarray(x)
#     out    = compute_box_cxcywh_to_xyxy(x_data)
#     return SimpleNamespace(shape=list(out.shape), data=out, dtype=out.dtype)


# def box_xyxy_to_cxcywh(x):
#     """
#     Convert boxes from (x0, y0, x1, y1) to (center_x, center_y, width, height).

#     Args:
#         x : SimpleNamespace or ndarray  [..., 4]

#     Returns:
#         SimpleNamespace(shape, data, dtype)
#     """
#     x_data = x.data if isinstance(x, SimpleNamespace) else np.asarray(x)
#     out    = compute_box_xyxy_to_cxcywh(x_data)
#     return SimpleNamespace(shape=list(out.shape), data=out, dtype=out.dtype)


# def box_area(boxes):
#     """
#     Compute area of boxes in (x0, y0, x1, y1) format.

#     Args:
#         boxes : SimpleNamespace or ndarray  [N, 4]

#     Returns:
#         SimpleNamespace(shape, data, dtype)  [N]
#     """
#     boxes_data = boxes.data if isinstance(boxes, SimpleNamespace) else np.asarray(boxes)
#     out        = compute_box_area(boxes_data)
#     return SimpleNamespace(shape=list(out.shape), data=out, dtype=out.dtype)


# def box_iou(boxes1, boxes2):
#     """
#     Compute pairwise IoU between two sets of boxes (both in xyxy format).

#     Args:
#         boxes1 : SimpleNamespace or ndarray  [N, 4]
#         boxes2 : SimpleNamespace or ndarray  [M, 4]

#     Returns:
#         (iou, union)  both SimpleNamespace  [N, M]
#     """
#     b1 = boxes1.data if isinstance(boxes1, SimpleNamespace) else np.asarray(boxes1)
#     b2 = boxes2.data if isinstance(boxes2, SimpleNamespace) else np.asarray(boxes2)

#     iou_data, union_data = compute_box_iou(b1, b2)

#     iou   = SimpleNamespace(shape=list(iou_data.shape),   data=iou_data,   dtype=iou_data.dtype)
#     union = SimpleNamespace(shape=list(union_data.shape), data=union_data, dtype=union_data.dtype)
#     return iou, union


# def generalized_box_iou(boxes1, boxes2):
#     """
#     Compute Generalized IoU (GIoU) between two sets of boxes (both in xyxy format).

#     Args:
#         boxes1 : SimpleNamespace or ndarray  [N, 4]
#         boxes2 : SimpleNamespace or ndarray  [M, 4]

#     Returns:
#         SimpleNamespace(shape, data, dtype)  [N, M]
#     """
#     b1 = boxes1.data if isinstance(boxes1, SimpleNamespace) else np.asarray(boxes1)
#     b2 = boxes2.data if isinstance(boxes2, SimpleNamespace) else np.asarray(boxes2)

#     out = compute_generalized_box_iou(b1, b2)
#     return SimpleNamespace(shape=list(out.shape), data=out, dtype=out.dtype)


# def masks_to_boxes(masks):
#     """
#     Compute bounding boxes around binary masks.

#     Args:
#         masks : SimpleNamespace or ndarray  [N, H, W]

#     Returns:
#         SimpleNamespace(shape, data, dtype)  [N, 4]
#     """
#     masks_data = masks.data if isinstance(masks, SimpleNamespace) else np.asarray(masks)
#     out        = compute_masks_to_boxes(masks_data)
#     return SimpleNamespace(shape=list(out.shape), data=out, dtype=out.dtype)
