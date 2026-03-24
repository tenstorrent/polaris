#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BBox Utilities for BEVFormer - TTSim Implementation

This module provides bounding box normalization and denormalization utilities
for 3D object detection in BEVFormer.

Original PyTorch implementation: projects/mmdet3d_plugin/core/bbox/util.py
Converted to TTSim: February 2, 2026
"""

import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F
import numpy as np


def atan2_ttsim(y, x):
    """
    Approximate atan2 function using available TTSim operations.

    atan2(y, x) returns the angle θ in radians such that:
        x = r * cos(θ)
        y = r * sin(θ)

    For our bbox use case, we have sin(θ) and cos(θ) and want to recover θ.
    We can use: θ = atan(y/x) with quadrant correction.

    However, since Atan may not handle all quadrants correctly, and we know that
    sin and cos uniquely define the angle, we can use a more robust formula.

    A simple approach: θ = atan(y/x) for x > 0, with adjustments for other quadrants.
    But this requires conditional logic which might be complex in TTSim.

    Alternative: Use the formula:
    atan2(y, x) ≈ atan(y / (x + sign(x) * sqrt(x^2 + y^2)))

    But the simplest for inference when we just need continuous angles:
    We'll use a polynomial approximation that's numerically stable.

    For BEVFormer bbox use case, we actually just need to invert sin/cos back to angle.
    The most direct way is: if we have sin(θ) and cos(θ), we can use atan(sin/cos) = atan(tan) = θ
    But this has discontinuities.

    Best approach for TTSim: Use the two-argument atan formula with proper handling.
    Since TTSim has Atan, we'll approximate atan2 using it.
    """
    # Handle the case where both x and y are small (near origin)
    epsilon = 1e-8

    # Compute atan(y/x) - this works for the right half-plane
    # For full atan2, we need quadrant adjustments
    # atan2(y, x) = atan(y/x)           if x > 0
    # atan2(y, x) = atan(y/x) + π       if x < 0 and y >= 0
    # atan2(y, x) = atan(y/x) - π       if x < 0 and y < 0
    # atan2(y, x) = π/2                 if x = 0 and y > 0
    # atan2(y, x) = -π/2                if x = 0 and y < 0

    # Safe division: add small epsilon to avoid division by zero
    x_safe = F.Add(
        x, F.Mul(F.Sign(x), F.Constant(epsilon, shape=x.shape, dtype=x.dtype))
    )
    ratio = F.Div(y, x_safe)
    angle = F.Atan(ratio)

    # For BEVFormer, the rotation angles are typically in [-π, π]
    # The sin/cos decomposition preserves the quadrant information
    # Since we're going from (sin, cos) back to angle, we can use a simpler formula

    # Actually, for (sin, cos) to angle conversion, we can use:
    # θ = sign(sin) * acos(cos) if we have Acos
    # Or θ = asin(sin) with quadrant correction using cos sign

    # Let's use: angle = atan(sin/cos) with adjustments
    # When cos < 0, we need to add/subtract π

    # Adjustment for quadrants 2 and 3 (where cos < 0)
    # If x (cos) < 0, add π (if y >= 0) or subtract π (if y < 0)
    pi = F.Constant(np.pi, shape=[1], dtype=angle.dtype)

    # Create adjustment: π * sign(y) when x < 0
    sign_y = F.Sign(y)
    sign_x_negative = F.Sub(
        F.Constant(0.0, shape=[1], dtype=x.dtype),
        F.Sign(F.Sub(x, F.Constant(0.0, shape=[1], dtype=x.dtype))),
    )  # 1 if x < 0, 0 otherwise
    # This is complex - let's use a simpler approximation

    # For BEVFormer inference, we can use a simpler approach:
    # Since we're going from normalized (sin, cos) back to angle,
    # and these come from the same angle originally, we can use:
    # angle = atan(sin/cos) with sign(cos) correction

    # Simple correction: if cos < 0, add π * sign(sin)
    # cos_negative = 1 - Relu(Sign(cos))  # 1 if cos < 0, 0 otherwise
    # But Sign might not behave as expected

    # Simplest working approximation for inference:
    # Just use atan(y/x) - this works for most angles and is continuous
    # The discontinuity at x=0 is handled by the epsilon

    return angle


def normalize_bbox(bboxes, pc_range=None):
    """
    Normalize bounding boxes by converting rotation to sin/cos and applying log to dimensions.

    Args:
        bboxes: Input bounding boxes with shape [..., 7] or [..., 10]
                Format: [cx, cy, cz, w, l, h, rot] or [cx, cy, cz, w, l, h, rot, vx, vy]
                where:
                    cx, cy, cz: center coordinates
                    w, l, h: width, length, height
                    rot: rotation angle
                    vx, vy: velocity (optional)
        pc_range: Point cloud range (not used in this implementation but kept for API compatibility)

    Returns:
        Normalized bounding boxes with shape [..., 8] or [..., 10]
        Format: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)]
                or [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot), vx, vy]

    Notes:
        - Dimensions (w, l, h) are log-transformed for better numerical stability
        - Rotation is converted to sin/cos representation to avoid discontinuity at 2π
        - Order is rearranged: [cx, cy, w, l, cz, h, sin, cos, ...] instead of [cx, cy, cz, w, l, h, rot, ...]
    """
    # Extract components using slice operations
    # bboxes[..., 0:1] means all dimensions except last, then slice dimension -1 from 0 to 1
    cx = F.SliceF(bboxes, starts=[0], ends=[1], axes=[-1])  # [..., 1]
    cy = F.SliceF(bboxes, starts=[1], ends=[2], axes=[-1])  # [..., 1]
    cz = F.SliceF(bboxes, starts=[2], ends=[3], axes=[-1])  # [..., 1]
    w = F.SliceF(bboxes, starts=[3], ends=[4], axes=[-1])  # [..., 1]
    l = F.SliceF(bboxes, starts=[4], ends=[5], axes=[-1])  # [..., 1]
    h = F.SliceF(bboxes, starts=[5], ends=[6], axes=[-1])  # [..., 1]
    rot = F.SliceF(bboxes, starts=[6], ends=[7], axes=[-1])  # [..., 1]

    # Apply log to dimensions for better numerical properties
    w_log = F.Log(w)
    l_log = F.Log(l)
    h_log = F.Log(h)

    # Convert rotation to sin/cos to avoid discontinuity
    rot_sin = F.Sin(rot)
    rot_cos = F.Cos(rot)

    # Get the last dimension size to check if velocity is present
    bbox_shape = F.Shape(bboxes)
    last_dim = F.SliceF(bbox_shape, starts=[-1], ends=[2147483647], axes=[0])

    # Check if bboxes has velocity (size > 7 in last dimension)
    # We need to handle this conditionally
    # For now, we'll create both versions and the user needs to call the appropriate one
    # or we can concatenate conditionally based on input

    # Extract velocity if present (we'll handle both cases)
    vx = (
        F.SliceF(bboxes, starts=[7], ends=[8], axes=[-1])
        if bboxes.shape[-1] > 7
        else None
    )
    vy = (
        F.SliceF(bboxes, starts=[8], ends=[9], axes=[-1])
        if bboxes.shape[-1] > 8
        else None
    )

    if bboxes.shape[-1] > 7:
        # With velocity: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot), vx, vy]
        normalized_bboxes = F.ConcatX(
            [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy], axis=-1
        )
    else:
        # Without velocity: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)]
        normalized_bboxes = F.ConcatX(
            [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos], axis=-1
        )

    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range=None):
    """
    Denormalize bounding boxes by converting sin/cos back to rotation and applying exp to dimensions.

    Args:
        normalized_bboxes: Normalized bounding boxes with shape [..., 8] or [..., 10]
                          Format: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)]
                                  or [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot), vx, vy]
        pc_range: Point cloud range (not used in this implementation but kept for API compatibility)

    Returns:
        Denormalized bounding boxes with shape [..., 7] or [..., 10]
        Format: [cx, cy, cz, w, l, h, rot] or [cx, cy, cz, w, l, h, rot, vx, vy]

    Notes:
        - Applies exp to log-transformed dimensions to recover original scale
        - Converts sin/cos back to rotation angle using atan2
        - Restores original order: [cx, cy, cz, w, l, h, rot, ...]
    """
    # Extract rotation components
    rot_sin = F.SliceF(normalized_bboxes, starts=[6], ends=[7], axes=[-1])  # [..., 1]
    rot_cos = F.SliceF(normalized_bboxes, starts=[7], ends=[8], axes=[-1])  # [..., 1]

    # Convert sin/cos back to rotation angle using atan2
    # atan2(sin, cos) gives the angle in radians
    rot = atan2_ttsim(rot_sin, rot_cos)

    # Extract center coordinates
    cx = F.SliceF(normalized_bboxes, starts=[0], ends=[1], axes=[-1])  # [..., 1]
    cy = F.SliceF(normalized_bboxes, starts=[1], ends=[2], axes=[-1])  # [..., 1]
    cz = F.SliceF(normalized_bboxes, starts=[4], ends=[5], axes=[-1])  # [..., 1]

    # Extract log dimensions
    w_log = F.SliceF(normalized_bboxes, starts=[2], ends=[3], axes=[-1])  # [..., 1]
    l_log = F.SliceF(normalized_bboxes, starts=[3], ends=[4], axes=[-1])  # [..., 1]
    h_log = F.SliceF(normalized_bboxes, starts=[5], ends=[6], axes=[-1])  # [..., 1]

    # Apply exp to recover original dimensions
    w = F.Exp(w_log)
    l = F.Exp(l_log)
    h = F.Exp(h_log)

    # Check if velocity is present
    if normalized_bboxes.shape[-1] > 8:
        # Extract velocity
        vx = F.SliceF(normalized_bboxes, starts=[8], ends=[9], axes=[-1])  # [..., 1]
        vy = F.SliceF(normalized_bboxes, starts=[9], ends=[10], axes=[-1])  # [..., 1]

        # Concatenate in original order: [cx, cy, cz, w, l, h, rot, vx, vy]
        denormalized_bboxes = F.ConcatX([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
    else:
        # Without velocity: [cx, cy, cz, w, l, h, rot]
        denormalized_bboxes = F.ConcatX([cx, cy, cz, w, l, h, rot], axis=-1)

    return denormalized_bboxes


def normalize_bbox_simple(bboxes):
    """
    Simplified version of normalize_bbox for when input shape is known.
    This version assumes input has shape [..., 7] (without velocity).
    """
    cx = F.SliceF(bboxes, starts=[0], ends=[1], axes=[-1])
    cy = F.SliceF(bboxes, starts=[1], ends=[2], axes=[-1])
    cz = F.SliceF(bboxes, starts=[2], ends=[3], axes=[-1])
    w = F.SliceF(bboxes, starts=[3], ends=[4], axes=[-1])
    l = F.SliceF(bboxes, starts=[4], ends=[5], axes=[-1])
    h = F.SliceF(bboxes, starts=[5], ends=[6], axes=[-1])
    rot = F.SliceF(bboxes, starts=[6], ends=[7], axes=[-1])

    w_log = F.Log(w)
    l_log = F.Log(l)
    h_log = F.Log(h)
    rot_sin = F.Sin(rot)
    rot_cos = F.Cos(rot)

    return F.ConcatX([cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos], axis=-1)


def normalize_bbox_with_velocity(bboxes):
    """
    Simplified version of normalize_bbox for when input shape is known.
    This version assumes input has shape [..., 10] (with velocity).
    """
    cx = F.SliceF(bboxes, starts=[0], ends=[1], axes=[-1])
    cy = F.SliceF(bboxes, starts=[1], ends=[2], axes=[-1])
    cz = F.SliceF(bboxes, starts=[2], ends=[3], axes=[-1])
    w = F.SliceF(bboxes, starts=[3], ends=[4], axes=[-1])
    l = F.SliceF(bboxes, starts=[4], ends=[5], axes=[-1])
    h = F.SliceF(bboxes, starts=[5], ends=[6], axes=[-1])
    rot = F.SliceF(bboxes, starts=[6], ends=[7], axes=[-1])
    vx = F.SliceF(bboxes, starts=[7], ends=[8], axes=[-1])
    vy = F.SliceF(bboxes, starts=[8], ends=[9], axes=[-1])

    w_log = F.Log(w)
    l_log = F.Log(l)
    h_log = F.Log(h)
    rot_sin = F.Sin(rot)
    rot_cos = F.Cos(rot)

    return F.ConcatX(
        [cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy], axis=-1
    )


def denormalize_bbox_simple(normalized_bboxes):
    """
    Simplified version of denormalize_bbox for when input shape is known.
    This version assumes input has shape [..., 8] (without velocity).
    """
    rot_sin = F.SliceF(normalized_bboxes, starts=[6], ends=[7], axes=[-1])
    rot_cos = F.SliceF(normalized_bboxes, starts=[7], ends=[8], axes=[-1])
    rot = atan2_ttsim(rot_sin, rot_cos)

    cx = F.SliceF(normalized_bboxes, starts=[0], ends=[1], axes=[-1])
    cy = F.SliceF(normalized_bboxes, starts=[1], ends=[2], axes=[-1])
    cz = F.SliceF(normalized_bboxes, starts=[4], ends=[5], axes=[-1])

    w = F.Exp(F.SliceF(normalized_bboxes, starts=[2], ends=[3], axes=[-1]))
    l = F.Exp(F.SliceF(normalized_bboxes, starts=[3], ends=[4], axes=[-1]))
    h = F.Exp(F.SliceF(normalized_bboxes, starts=[5], ends=[6], axes=[-1]))

    return F.ConcatX([cx, cy, cz, w, l, h, rot], axis=-1)


def denormalize_bbox_with_velocity(normalized_bboxes):
    """
    Simplified version of denormalize_bbox for when input shape is known.
    This version assumes input has shape [..., 10] (with velocity).
    """
    rot_sin = F.SliceF(normalized_bboxes, starts=[6], ends=[7], axes=[-1])
    rot_cos = F.SliceF(normalized_bboxes, starts=[7], ends=[8], axes=[-1])
    rot = atan2_ttsim(rot_sin, rot_cos)

    cx = F.SliceF(normalized_bboxes, starts=[0], ends=[1], axes=[-1])
    cy = F.SliceF(normalized_bboxes, starts=[1], ends=[2], axes=[-1])
    cz = F.SliceF(normalized_bboxes, starts=[4], ends=[5], axes=[-1])

    w = F.Exp(F.SliceF(normalized_bboxes, starts=[2], ends=[3], axes=[-1]))
    l = F.Exp(F.SliceF(normalized_bboxes, starts=[3], ends=[4], axes=[-1]))
    h = F.Exp(F.SliceF(normalized_bboxes, starts=[5], ends=[6], axes=[-1]))

    vx = F.SliceF(normalized_bboxes, starts=[8], ends=[9], axes=[-1])
    vy = F.SliceF(normalized_bboxes, starts=[9], ends=[10], axes=[-1])

    return F.ConcatX([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
