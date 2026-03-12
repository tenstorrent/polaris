#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for BBox Utils (normalize_bbox and denormalize_bbox)

This test suite validates the TTSim implementation of bounding box normalization
and denormalization utilities against the PyTorch reference implementation.

Test Coverage:
1. Module construction and basic functionality
2. Shape inference for various input dimensions
3. Normalization without velocity (7D input)
4. Normalization with velocity (10D input)
5. Round-trip consistency (normalize + denormalize = identity)
6. Numerical accuracy against PyTorch
7. Batch processing
8. Edge cases (small dimensions, various rotation angles)
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import time
import torch
import numpy as np
import ttsim.front.functional.op as F
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

# Import PyTorch reference implementation
import_path = os.path.join(os.path.dirname(__file__), "../PyTorch Scripts")
sys.path.insert(0, import_path)
from util import (
    normalize_bbox as normalize_bbox_torch,
    denormalize_bbox as denormalize_bbox_torch,
)

# Get polaris root for device config
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


# Import TTSim bbox utilities (we'll call the functions directly)
# Since these are functional (not modules), we need to use them differently
# Let's directly import the TTSim functional operations we need


def normalize_bbox_ttsim(bboxes_tensor):
    """TTSim version of normalize_bbox"""
    # Extract components using slice operations
    # SliceF requires: data, starts, ends, axes, steps as separate tensor inputs
    # And needs out_shape attribute specifying the output shape

    # Calculate output shape for slice (all dimensions same except last which becomes 1)
    out_shape = list(bboxes_tensor.shape)
    out_shape[-1] = 1

    starts_0 = F._from_data("starts_0", np.array([0], dtype=np.int64))
    starts_1 = F._from_data("starts_1", np.array([1], dtype=np.int64))
    starts_2 = F._from_data("starts_2", np.array([2], dtype=np.int64))
    starts_3 = F._from_data("starts_3", np.array([3], dtype=np.int64))
    starts_4 = F._from_data("starts_4", np.array([4], dtype=np.int64))
    starts_5 = F._from_data("starts_5", np.array([5], dtype=np.int64))
    starts_6 = F._from_data("starts_6", np.array([6], dtype=np.int64))
    starts_7 = F._from_data("starts_7", np.array([7], dtype=np.int64))
    starts_8 = F._from_data("starts_8", np.array([8], dtype=np.int64))

    ends_1 = F._from_data("ends_1", np.array([1], dtype=np.int64))
    ends_2 = F._from_data("ends_2", np.array([2], dtype=np.int64))
    ends_3 = F._from_data("ends_3", np.array([3], dtype=np.int64))
    ends_4 = F._from_data("ends_4", np.array([4], dtype=np.int64))
    ends_5 = F._from_data("ends_5", np.array([5], dtype=np.int64))
    ends_6 = F._from_data("ends_6", np.array([6], dtype=np.int64))
    ends_7 = F._from_data("ends_7", np.array([7], dtype=np.int64))
    ends_8 = F._from_data("ends_8", np.array([8], dtype=np.int64))
    ends_9 = F._from_data("ends_9", np.array([9], dtype=np.int64))

    axes_neg1 = F._from_data("axes_neg1", np.array([-1], dtype=np.int64))
    steps_1 = F._from_data("steps_1", np.array([1], dtype=np.int64))

    cx = F.SliceF("slice_cx", out_shape=out_shape)(
        bboxes_tensor, starts_0, ends_1, axes_neg1, steps_1
    )
    cy = F.SliceF("slice_cy", out_shape=out_shape)(
        bboxes_tensor, starts_1, ends_2, axes_neg1, steps_1
    )
    cz = F.SliceF("slice_cz", out_shape=out_shape)(
        bboxes_tensor, starts_2, ends_3, axes_neg1, steps_1
    )
    w = F.SliceF("slice_w", out_shape=out_shape)(
        bboxes_tensor, starts_3, ends_4, axes_neg1, steps_1
    )
    l = F.SliceF("slice_l", out_shape=out_shape)(
        bboxes_tensor, starts_4, ends_5, axes_neg1, steps_1
    )
    h = F.SliceF("slice_h", out_shape=out_shape)(
        bboxes_tensor, starts_5, ends_6, axes_neg1, steps_1
    )
    rot = F.SliceF("slice_rot", out_shape=out_shape)(
        bboxes_tensor, starts_6, ends_7, axes_neg1, steps_1
    )

    # Apply log to dimensions
    w_log = F.Log("log_w")(w)
    l_log = F.Log("log_l")(l)
    h_log = F.Log("log_h")(h)

    # Convert rotation to sin/cos
    rot_sin = F.Sin("sin_rot")(rot)
    rot_cos = F.Cos("cos_rot")(rot)

    # Check if velocity present
    if bboxes_tensor.shape[-1] > 7:
        vx = F.SliceF("slice_vx", out_shape=out_shape)(
            bboxes_tensor, starts_7, ends_8, axes_neg1, steps_1
        )
        vy = F.SliceF("slice_vy", out_shape=out_shape)(
            bboxes_tensor, starts_8, ends_9, axes_neg1, steps_1
        )
        result = F.ConcatX("concat_result", axis=-1)(
            cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos, vx, vy
        )
    else:
        result = F.ConcatX("concat_result", axis=-1)(
            cx, cy, w_log, l_log, cz, h_log, rot_sin, rot_cos
        )

    return result


def atan2_simple(y, x):
    """Improved atan2 using TTSim operations with quadrant handling"""
    # Generate unique names for each operation to avoid conflicts
    import time

    suffix = str(int(time.time() * 1000000) % 1000000)  # Unique timestamp-based suffix

    # Safe division: add small epsilon to avoid division by zero
    epsilon = 1e-8
    epsilon_const = F._from_data(
        f"epsilon_{suffix}", np.array([epsilon], dtype=np.float32), is_const=True
    )

    # Basic atan(y/x) computation
    sign_x = F.Sign(f"sign_x_{suffix}")(x)
    eps_scaled = F.Mul(f"eps_scaled_{suffix}")(sign_x, epsilon_const)
    x_safe = F.Add(f"x_safe_{suffix}")(x, eps_scaled)
    ratio = F.Div(f"ratio_{suffix}")(y, x_safe)
    basic_angle = F.Atan(f"atan_{suffix}")(ratio)

    # For proper atan2, we need to adjust based on the sign of x:
    # - If x < 0 and y >= 0: add π
    # - If x < 0 and y < 0: subtract π
    # This can be simplified to: if x < 0: add sign(y) * π

    # Create constants
    pi = F._from_data(
        f"pi_{suffix}", np.array([np.pi], dtype=np.float32), is_const=True
    )
    zero = F._from_data(
        f"zero_{suffix}", np.array([0.0], dtype=np.float32), is_const=True
    )
    one = F._from_data(
        f"one_{suffix}", np.array([1.0], dtype=np.float32), is_const=True
    )

    # Check if x is negative: sign(x) < 0 means x < 0
    # sign(x) will be -1 if x < 0, so (1 - sign(x))/2 = 1 if x < 0, else 0
    one_minus_sign_x = F.Sub(f"one_minus_sign_x_{suffix}")(one, sign_x)
    two = F._from_data(
        f"two_{suffix}", np.array([2.0], dtype=np.float32), is_const=True
    )
    x_negative_mask = F.Div(f"x_negative_mask_{suffix}")(
        one_minus_sign_x, two
    )  # 1 if x<0, else 0

    # Get sign of y
    sign_y = F.Sign(f"sign_y_{suffix}")(y)

    # Compute adjustment: sign(y) * π * x_negative_mask
    pi_times_sign_y = F.Mul(f"pi_times_sign_y_{suffix}")(pi, sign_y)
    adjustment = F.Mul(f"adjustment_{suffix}")(pi_times_sign_y, x_negative_mask)

    # Final angle = basic_angle + adjustment
    angle = F.Add(f"final_angle_{suffix}")(basic_angle, adjustment)

    return angle


def denormalize_bbox_ttsim(normalized_bboxes_tensor):
    """TTSim version of denormalize_bbox"""
    # Create index tensors
    # Calculate output shape for slice (all dimensions same except last which becomes 1)
    out_shape = list(normalized_bboxes_tensor.shape)
    out_shape[-1] = 1

    starts_0 = F._from_data("starts_0_d", np.array([0], dtype=np.int64))
    starts_1 = F._from_data("starts_1_d", np.array([1], dtype=np.int64))
    starts_2 = F._from_data("starts_2_d", np.array([2], dtype=np.int64))
    starts_3 = F._from_data("starts_3_d", np.array([3], dtype=np.int64))
    starts_4 = F._from_data("starts_4_d", np.array([4], dtype=np.int64))
    starts_5 = F._from_data("starts_5_d", np.array([5], dtype=np.int64))
    starts_6 = F._from_data("starts_6_d", np.array([6], dtype=np.int64))
    starts_7 = F._from_data("starts_7_d", np.array([7], dtype=np.int64))
    starts_8 = F._from_data("starts_8_d", np.array([8], dtype=np.int64))
    starts_9 = F._from_data("starts_9_d", np.array([9], dtype=np.int64))

    ends_1 = F._from_data("ends_1_d", np.array([1], dtype=np.int64))
    ends_2 = F._from_data("ends_2_d", np.array([2], dtype=np.int64))
    ends_3 = F._from_data("ends_3_d", np.array([3], dtype=np.int64))
    ends_4 = F._from_data("ends_4_d", np.array([4], dtype=np.int64))
    ends_5 = F._from_data("ends_5_d", np.array([5], dtype=np.int64))
    ends_6 = F._from_data("ends_6_d", np.array([6], dtype=np.int64))
    ends_7 = F._from_data("ends_7_d", np.array([7], dtype=np.int64))
    ends_8 = F._from_data("ends_8_d", np.array([8], dtype=np.int64))
    ends_9 = F._from_data("ends_9_d", np.array([9], dtype=np.int64))
    ends_10 = F._from_data("ends_10_d", np.array([10], dtype=np.int64))

    axes_neg1 = F._from_data("axes_neg1_d", np.array([-1], dtype=np.int64))
    steps_1 = F._from_data("steps_1_d", np.array([1], dtype=np.int64))

    # Extract rotation components
    rot_sin = F.SliceF("slice_rot_sin", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_6, ends_7, axes_neg1, steps_1
    )
    rot_cos = F.SliceF("slice_rot_cos", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_7, ends_8, axes_neg1, steps_1
    )
    rot = atan2_simple(rot_sin, rot_cos)

    # Extract center coordinates
    cx = F.SliceF("slice_cx_denorm", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_0, ends_1, axes_neg1, steps_1
    )
    cy = F.SliceF("slice_cy_denorm", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_1, ends_2, axes_neg1, steps_1
    )
    cz = F.SliceF("slice_cz_denorm", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_4, ends_5, axes_neg1, steps_1
    )

    # Extract and exp dimensions
    w_log = F.SliceF("slice_w_log", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_2, ends_3, axes_neg1, steps_1
    )
    l_log = F.SliceF("slice_l_log", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_3, ends_4, axes_neg1, steps_1
    )
    h_log = F.SliceF("slice_h_log", out_shape=out_shape)(
        normalized_bboxes_tensor, starts_5, ends_6, axes_neg1, steps_1
    )

    w = F.Exp("exp_w")(w_log)
    l = F.Exp("exp_l")(l_log)
    h = F.Exp("exp_h")(h_log)

    # Check if velocity present
    if normalized_bboxes_tensor.shape[-1] > 8:
        vx = F.SliceF("slice_vx_denorm", out_shape=out_shape)(
            normalized_bboxes_tensor, starts_8, ends_9, axes_neg1, steps_1
        )
        vy = F.SliceF("slice_vy_denorm", out_shape=out_shape)(
            normalized_bboxes_tensor, starts_9, ends_10, axes_neg1, steps_1
        )
        result = F.ConcatX("concat_denorm_result", axis=-1)(
            cx, cy, cz, w, l, h, rot, vx, vy
        )
    else:
        result = F.ConcatX("concat_denorm_result", axis=-1)(cx, cy, cz, w, l, h, rot)

    return result


def print_header(text, char="="):
    """Print a formatted header"""
    print("\n" + char * 80)
    print(text)
    print(char * 80 + "\n")


def print_test(text):
    """Print a test description"""
    print(f"\n{text}")
    print("-" * 80)


def test_normalize_without_velocity():
    """Test normalization for bboxes without velocity (7D)"""
    print_test("TEST 1: Normalize BBoxes (Without Velocity)")

    # Create test data: [cx, cy, cz, w, l, h, rot]
    # Use realistic values for 3D object detection
    bboxes_np = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],  # Box at origin, no rotation
            [10.0, 5.0, -1.0, 1.8, 4.2, 1.6, np.pi / 4],  # Rotated 45 degrees
            [-5.0, -3.0, 0.5, 2.2, 4.5, 1.7, -np.pi / 3],  # Negative rotation
        ],
        dtype=np.float32,
    )

    print(f"Input shape: {bboxes_np.shape}")
    print(f"Input bboxes:\n{bboxes_np}")

    # PyTorch reference
    bboxes_torch = torch.from_numpy(bboxes_np)
    normalized_torch = normalize_bbox_torch(bboxes_torch, None)
    normalized_torch_np = normalized_torch.numpy()

    print(f"\nPyTorch output shape: {normalized_torch_np.shape}")
    print(f"PyTorch normalized:\n{normalized_torch_np}")

    # TTSim implementation (using F operations directly)
    bboxes_ttsim = F._from_data("bboxes", bboxes_np)
    normalized_ttsim = normalize_bbox_ttsim(bboxes_ttsim)

    # Get the data from the tensor
    normalized_ttsim_np = normalized_ttsim.data

    print(f"\nTTSim output shape: {normalized_ttsim_np.shape}")
    print(f"TTSim normalized:\n{normalized_ttsim_np}")

    # Compare
    print("\nComparison:")
    print(f"Expected output shape: {normalized_torch_np.shape}")
    print(f"Actual output shape: {normalized_ttsim_np.shape}")

    max_diff = np.max(np.abs(normalized_torch_np - normalized_ttsim_np))
    mean_diff = np.mean(np.abs(normalized_torch_np - normalized_ttsim_np))

    print(f"\nNumerical accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    tolerance = 1e-5
    passed = (
        max_diff < tolerance and normalized_torch_np.shape == normalized_ttsim_np.shape
    )

    if passed:
        print(f"✓ Test PASSED (within tolerance {tolerance})")
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")

    return passed


def test_normalize_with_velocity():
    """Test normalization for bboxes with velocity (10D)"""
    print_test("TEST 2: Normalize BBoxes (With Velocity)")

    # Create test data: [cx, cy, cz, w, l, h, rot, vx, vy]
    bboxes_np = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0, 1.0, 0.5],
            [10.0, 5.0, -1.0, 1.8, 4.2, 1.6, np.pi / 4, -0.5, 1.2],
            [-5.0, -3.0, 0.5, 2.2, 4.5, 1.7, -np.pi / 3, 0.8, -0.3],
        ],
        dtype=np.float32,
    )

    print(f"Input shape: {bboxes_np.shape}")
    print(f"Input bboxes (with velocity):\n{bboxes_np}")

    # PyTorch reference
    bboxes_torch = torch.from_numpy(bboxes_np)
    normalized_torch = normalize_bbox_torch(bboxes_torch, None)
    normalized_torch_np = normalized_torch.numpy()

    print(f"\nPyTorch output shape: {normalized_torch_np.shape}")

    # TTSim implementation
    bboxes_ttsim = F._from_data("bboxes_vel", bboxes_np)
    normalized_ttsim = normalize_bbox_ttsim(bboxes_ttsim)
    normalized_ttsim_np = normalized_ttsim.data

    print(f"TTSim output shape: {normalized_ttsim_np.shape}")

    # Compare
    max_diff = np.max(np.abs(normalized_torch_np - normalized_ttsim_np))
    mean_diff = np.mean(np.abs(normalized_torch_np - normalized_ttsim_np))

    print(f"\nNumerical accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    tolerance = 1e-5
    if max_diff < tolerance and normalized_torch_np.shape == normalized_ttsim_np.shape:
        print(f"✓ Test PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")
        return False


def test_denormalize_without_velocity():
    """Test denormalization for bboxes without velocity (8D)"""
    print_test("TEST 3: Denormalize BBoxes (Without Velocity)")

    # Create normalized test data: [cx, cy, log(w), log(l), cz, log(h), sin(rot), cos(rot)]
    normalized_np = np.array(
        [
            [0.0, 0.0, np.log(2.0), np.log(4.0), 0.0, np.log(1.5), 0.0, 1.0],
            [
                10.0,
                5.0,
                np.log(1.8),
                np.log(4.2),
                -1.0,
                np.log(1.6),
                np.sin(np.pi / 4),
                np.cos(np.pi / 4),
            ],
            [
                -5.0,
                -3.0,
                np.log(2.2),
                np.log(4.5),
                0.5,
                np.log(1.7),
                np.sin(-np.pi / 3),
                np.cos(-np.pi / 3),
            ],
        ],
        dtype=np.float32,
    )

    print(f"Input shape: {normalized_np.shape}")
    print(f"Normalized bboxes:\n{normalized_np}")

    # PyTorch reference
    normalized_torch = torch.from_numpy(normalized_np)
    denormalized_torch = denormalize_bbox_torch(normalized_torch, None)
    denormalized_torch_np = denormalized_torch.numpy()

    print(f"\nPyTorch output shape: {denormalized_torch_np.shape}")
    print(f"PyTorch denormalized:\n{denormalized_torch_np}")

    # TTSim implementation
    normalized_ttsim = F._from_data("normalized_bboxes", normalized_np)
    denormalized_ttsim = denormalize_bbox_ttsim(normalized_ttsim)
    denormalized_ttsim_np = denormalized_ttsim.data

    print(f"\nTTSim output shape: {denormalized_ttsim_np.shape}")
    print(f"TTSim denormalized:\n{denormalized_ttsim_np}")

    # Compare results
    max_diff = np.max(np.abs(denormalized_torch_np - denormalized_ttsim_np))
    mean_diff = np.mean(np.abs(denormalized_torch_np - denormalized_ttsim_np))

    print(f"\nNumerical accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    tolerance = 1e-5
    if (
        max_diff < tolerance
        and denormalized_torch_np.shape == denormalized_ttsim_np.shape
    ):
        print(f"✓ Test PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")
        return False


def test_round_trip_consistency():
    """Test that normalize -> denormalize recovers original values"""
    print_test("TEST 4: Round-Trip Consistency (Normalize -> Denormalize)")

    # Create original bboxes
    original_np = np.array(
        [
            [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [10.0, 5.0, -1.0, 1.8, 4.2, 1.6, np.pi / 4],
            [-5.0, -3.0, 0.5, 2.2, 4.5, 1.7, -np.pi / 3],
            [20.0, -10.0, 2.0, 3.5, 5.0, 2.0, np.pi / 2],
        ],
        dtype=np.float32,
    )

    print(f"Original bboxes shape: {original_np.shape}")
    print(f"Original values:\n{original_np}")

    # Normalize
    bboxes_ttsim = F._from_data("bboxes_orig", original_np)
    normalized_ttsim = normalize_bbox_ttsim(bboxes_ttsim)
    normalized_np = normalized_ttsim.data

    print(f"\nNormalized shape: {normalized_np.shape}")

    # Denormalize
    normalized_ttsim2 = F._from_data("normalized", normalized_np)
    recovered_ttsim = denormalize_bbox_ttsim(normalized_ttsim2)
    recovered_np = recovered_ttsim.data

    print(f"Recovered shape: {recovered_np.shape}")
    print(f"Recovered values:\n{recovered_np}")

    # Compare with original
    max_diff = np.max(np.abs(original_np - recovered_np))
    mean_diff = np.mean(np.abs(original_np - recovered_np))

    print(f"\nRound-trip accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    # Check individual components
    print(f"\nComponent-wise differences:")
    print(
        f"  Center (cx, cy, cz): {np.max(np.abs(original_np[:, :3] - recovered_np[:, :3])):.10f}"
    )
    print(
        f"  Dimensions (w, l, h): {np.max(np.abs(original_np[:, 3:6] - recovered_np[:, 3:6])):.10f}"
    )
    print(f"  Rotation: {np.max(np.abs(original_np[:, 6] - recovered_np[:, 6])):.10f}")

    tolerance = 1e-4  # Slightly higher tolerance for round-trip due to sin/cos/atan approximations
    if max_diff < tolerance:
        print(f"✓ Test PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")
        return False


def test_batch_processing():
    """Test with batch dimensions"""
    print_test("TEST 5: Batch Processing")

    # Create batched data: [B, N, 7]
    batch_np = np.array(
        [
            # Batch 1
            [
                [0.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0],
                [10.0, 5.0, -1.0, 1.8, 4.2, 1.6, np.pi / 4],
            ],
            # Batch 2
            [
                [-5.0, -3.0, 0.5, 2.2, 4.5, 1.7, -np.pi / 3],
                [20.0, -10.0, 2.0, 3.5, 5.0, 2.0, np.pi / 2],
            ],
        ],
        dtype=np.float32,
    )

    print(f"Input shape (batched): {batch_np.shape}")

    # PyTorch reference
    batch_torch = torch.from_numpy(batch_np)
    normalized_torch = normalize_bbox_torch(batch_torch, None)
    normalized_torch_np = normalized_torch.numpy()

    print(f"PyTorch output shape: {normalized_torch_np.shape}")

    # TTSim implementation
    batch_ttsim = F._from_data("batch", batch_np)
    normalized_ttsim = normalize_bbox_ttsim(batch_ttsim)
    normalized_ttsim_np = normalized_ttsim.data

    print(f"TTSim output shape: {normalized_ttsim_np.shape}")

    # Compare
    max_diff = np.max(np.abs(normalized_torch_np - normalized_ttsim_np))
    mean_diff = np.mean(np.abs(normalized_torch_np - normalized_ttsim_np))

    print(f"\nNumerical accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    tolerance = 1e-5
    if max_diff < tolerance and normalized_torch_np.shape == normalized_ttsim_np.shape:
        print(f"✓ Test PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")
        return False


def test_edge_cases():
    """Test edge cases like very small dimensions, various angles"""
    print_test("TEST 6: Edge Cases")

    # Test various rotation angles including boundary cases
    test_angles = [
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        np.pi,
        -np.pi / 4,
        -np.pi / 2,
        -np.pi,
    ]

    edge_cases_np = np.array(
        [
            [
                0.0,
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                angle,
            ]  # Small dimensions with various angles
            for angle in test_angles
        ],
        dtype=np.float32,
    )

    print(f"Testing {len(test_angles)} different rotation angles")
    print(f"Input shape: {edge_cases_np.shape}")

    # PyTorch reference
    edge_torch = torch.from_numpy(edge_cases_np)
    normalized_torch = normalize_bbox_torch(edge_torch, None)
    normalized_torch_np = normalized_torch.numpy()

    # TTSim implementation
    edge_ttsim = F._from_data("edge_cases", edge_cases_np)
    normalized_ttsim = normalize_bbox_ttsim(edge_ttsim)
    normalized_ttsim_np = normalized_ttsim.data

    # Compare
    max_diff = np.max(np.abs(normalized_torch_np - normalized_ttsim_np))
    mean_diff = np.mean(np.abs(normalized_torch_np - normalized_ttsim_np))

    print(f"\nNumerical accuracy:")
    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    tolerance = 1e-5
    if max_diff < tolerance and normalized_torch_np.shape == normalized_ttsim_np.shape:
        print(f"✓ Test PASSED (within tolerance {tolerance})")
        return True
    else:
        print(f"✗ Test FAILED (exceeds tolerance {tolerance})")
        print(f"\nAngle-wise comparison (original → normalized):")
        for i, angle in enumerate(test_angles):
            print(
                f"  Angle {angle:7.4f}: sin={normalized_torch_np[i, 6]:.6f}, cos={normalized_torch_np[i, 7]:.6f}"
            )
        return False


def run_all_tests():
    """Run all validation tests"""
    print_header("BBox Utils TTSim Module Test Suite")

    tests = [
        ("Normalize Without Velocity", test_normalize_without_velocity),
        ("Normalize With Velocity", test_normalize_with_velocity),
        ("Denormalize Without Velocity", test_denormalize_without_velocity),
        ("Round-Trip Consistency", test_round_trip_consistency),
        ("Batch Processing", test_batch_processing),
        ("Edge Cases", test_edge_cases),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print_header("TEST SUMMARY")

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        dots = "." * (60 - len(test_name))
        print(f"{test_name}{dots} {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n All tests passed! The module is working correctly.")
    else:
        print(
            f"\n  {total_count - passed_count} test(s) failed. Please review the output above."
        )

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
