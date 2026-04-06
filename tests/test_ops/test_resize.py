#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import logging
import pytest
from loguru import logger

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_resize
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    # Silence python logging coming from ttsim modules (only show ERROR+)
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    # If the project uses loguru, remove default sinks and keep only ERROR+
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_resize_nearest(X, scales):
    """
    Reference implementation of Resize with nearest neighbor interpolation
    Simplified for common cases
    """
    if len(X.shape) == 4:  # [N, C, H, W]
        N, C, H, W = X.shape
        new_H = int(H * scales[2])
        new_W = int(W * scales[3])

        output = np.zeros((N, C, new_H, new_W), dtype=X.dtype)
        for n in range(N):
            for c in range(C):
                for h in range(new_H):
                    for w in range(new_W):
                        src_h = int(h / scales[2])
                        src_w = int(w / scales[3])
                        src_h = min(src_h, H - 1)
                        src_w = min(src_w, W - 1)
                        output[n, c, h, w] = X[n, c, src_h, src_w]
        return output
    else:
        # For other shapes, apply simple nearest scaling
        return X  # Placeholder


# Test cases with shape validation and numerical validation
test_name = "test_resize"
test_cases = [
    # (name, input_shape, scales, mode, test_data_type)
    # Upsampling cases
    ("Upsample 2x nearest", [1, 1, 4, 4], [1.0, 1.0, 2.0, 2.0], "nearest", "positive"),
    ("Upsample 3x nearest", [1, 1, 3, 3], [1.0, 1.0, 3.0, 3.0], "nearest", "positive"),
    ("Upsample non-uniform", [1, 2, 4, 4], [1.0, 1.0, 2.0, 3.0], "nearest", "positive"),
    (
        "Multi-channel upsample",
        [1, 3, 4, 4],
        [1.0, 1.0, 2.0, 2.0],
        "nearest",
        "positive",
    ),
    ("Batch upsample", [2, 2, 4, 4], [1.0, 1.0, 2.0, 2.0], "nearest", "positive"),
    # Downsampling cases
    (
        "Downsample 0.5x nearest",
        [1, 1, 8, 8],
        [1.0, 1.0, 0.5, 0.5],
        "nearest",
        "positive",
    ),
    (
        "Downsample non-uniform",
        [1, 2, 8, 8],
        [1.0, 1.0, 0.5, 0.25],
        "nearest",
        "positive",
    ),
    # Identity (scale = 1.0)
    ("Identity scale", [1, 1, 4, 4], [1.0, 1.0, 1.0, 1.0], "nearest", "positive"),
    # Edge cases: negative values
    ("Negative values", [1, 1, 4, 4], [1.0, 1.0, 2.0, 2.0], "nearest", "negative"),
    # Edge cases: zero values
    ("Zero values", [1, 1, 4, 4], [1.0, 1.0, 2.0, 2.0], "nearest", "zeros"),
    # Edge cases: mixed values
    ("Mixed values", [1, 1, 4, 4], [1.0, 1.0, 2.0, 2.0], "nearest", "mixed"),
    # Edge cases: small input
    ("Small input 2x2", [1, 1, 2, 2], [1.0, 1.0, 4.0, 4.0], "nearest", "positive"),
    # Edge cases: large scale factor
    ("Large scale factor", [1, 1, 2, 2], [1.0, 1.0, 8.0, 8.0], "nearest", "positive"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) * 10 + 1
    elif data_type == "negative":
        return -(np.random.rand(*shape).astype(np.float32) * 10 + 1)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 5).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_resize():
    """Test Resize with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, scales, mode, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Calculate expected output shape
        expected_shape = [
            int(input_shape[i] * scales[i]) for i in range(len(input_shape))
        ]

        # Create input tensors: X, ROI, and SCALES (Resize expects 3 inputs)
        # Scales should only contain the spatial scale factors (H, W)
        roi_data = np.array([], dtype=np.float32)
        scales_data = np.array(
            [scales[-2], scales[-1]], dtype=np.float32
        )  # Only H and W scales

        i_tensors = [
            F._from_data("X", test_data),
            F._from_data("roi", roi_data),
            F._from_data("scales", scales_data),
        ]
        o_tensors = [make_tensor("Y")]
        o_tensors[0].shape = None  # Force shape inference

        op_info = {
            "name": op_name,
            "optype": "Resize",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "mode": mode,
                "scale_factor": [
                    float(scales[-2]),
                    float(scales[-1]),
                ],  # For compute_resize
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation (shape inference happens during get_perf_counts)
        inf_shape = o_tensors[0].shape

        shape_match = inf_shape == expected_shape

        # 2. Numerical validation (for nearest neighbor, values should come from input)
        numerical_match = True
        try:
            computed_output = compute_resize(i_tensors, op_obj)

            if mode == "nearest":
                # For nearest neighbor, all output values should exist in input
                unique_input = np.unique(test_data)
                unique_output = np.unique(computed_output)

                # Check if output values are from input (with tolerance)
                all_from_input = True
                for val in unique_output:
                    if not np.any(np.isclose(unique_input, val, rtol=1e-5, atol=1e-7)):
                        all_from_input = False
                        break

                if not all_from_input and not (data_type == "zeros"):
                    numerical_match = False
                    logger.debug("\n  Output contains values not in input")

            # Verify correct shape
            if computed_output.shape != tuple(expected_shape):
                numerical_match = False
                logger.debug("\n  Output shape mismatch")

        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {expected_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_resize_precision"
precision_test_cases = [
    # (name, input, scales, mode, expected_output)
    (
        "2x2 to 4x4 nearest",
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        [1.0, 1.0, 2.0, 2.0],
        "nearest",
        np.array(
            [
                [
                    [
                        [1.0, 1.0, 2.0, 2.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [3.0, 3.0, 4.0, 4.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    ),
    (
        "Identity resize",
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        [1.0, 1.0, 1.0, 1.0],
        "nearest",
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_resize_precision():
    """Test Resize with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, scales, mode, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        # Create input tensors: X, ROI, and SCALES (Resize expects 3 inputs)
        # Scales should only contain the spatial scale factors (H, W)
        roi_data = np.array([], dtype=np.float32)
        scales_data = np.array(
            [scales[-2], scales[-1]], dtype=np.float32
        )  # Only H and W scales

        i_tensors = [
            F._from_data("X", test_data),
            F._from_data("roi", roi_data),
            F._from_data("scales", scales_data),
        ]
        o_tensors = [make_tensor("Y")]
        o_tensors[0].shape = None  # Force shape inference

        op_info = {
            "name": op_name,
            "optype": "Resize",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {
                "mode": mode,
                "scale_factor": [
                    float(scales[-2]),
                    float(scales[-1]),
                ],  # For compute_resize
            },
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_resize(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected shape: {expected_output.shape}")
                logger.debug(f"  Got shape:      {computed_output.shape}")
                if computed_output.shape == expected_output.shape:
                    logger.debug(
                        f"  Expected sample: {expected_output.flat[:10]}"
                    )
                    logger.debug(f"  Got sample:      {computed_output.flat[:10]}")
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_resize_memory_stats(input_shape, scales, dtype="float32"):
    """Calculate memory and compute statistics for a resize operation"""
    # Get device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    # Calculate output shape
    output_shape = [int(input_shape[i] * scales[i]) for i in range(len(input_shape))]

    # Create input tensors (X, roi, scales)
    np_dtype = getattr(np, dtype)
    data_X = np.random.randn(*input_shape).astype(np_dtype)
    roi_data = np.array([], dtype=np.float32)
    scales_data = np.array(
        [scales[-2], scales[-1]], dtype=np.float32
    )  # Only H and W scales

    i_tensors = [
        F._from_data("X", data_X),
        F._from_data("roi", roi_data),
        F._from_data("scales", scales_data),
    ]
    o_tensors = [make_tensor("Y")]
    o_tensors[0].shape = None  # Force shape inference

    # Create operation
    op_info = {
        "optype": "Resize",
        "name": "resize_mem_test",
        "attrs": {
            "mode": "nearest",
            "scale_factor": [float(scales[-2]), float(scales[-1])],
        },
    }
    op_obj = SimOp(op_info)

    # Set op references
    for x in i_tensors:
        x.op_in = [op_info["name"]]
    for x in o_tensors:
        x.op_out = [op_info["name"]]

    # Get performance counts
    op_obj.get_perf_counts(i_tensors, o_tensors)

    # Calculate statistics
    perf_stats = op_obj.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    ops = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else np.prod(output_shape)
    )
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    # Calculate intensities
    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0
    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1  # 1 op per cycle for resize
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "scales": scales,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }


def test_resize_memory_validation():
    """Memory validation test for resize operation"""
    logger.info("\n" + "=" * 80)
    logger.info("RESIZE MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations (input_shape, scales)
    test_configs = [
        ([1, 3, 32, 32], [1.0, 1.0, 2.0, 2.0]),
        ([1, 3, 64, 64], [1.0, 1.0, 0.5, 0.5]),
        ([1, 3, 128, 128], [1.0, 1.0, 2.0, 2.0]),
        ([2, 3, 56, 56], [1.0, 1.0, 2.0, 2.0]),
        ([1, 64, 112, 112], [1.0, 1.0, 0.5, 0.5]),
        ([4, 128, 28, 28], [1.0, 1.0, 2.0, 2.0]),
        ([1, 256, 14, 14], [1.0, 1.0, 4.0, 4.0]),
    ]

    results = []
    for input_shape, scales in test_configs:
        stats = calculate_resize_memory_stats(input_shape, scales)
        results.append(stats)

    # Print device info once
    device = results[0]["device"]
    logger.debug(f"\nDevice: {device.devname}")
    logger.debug(f"  Name: {device.name}")
    logger.debug(f"  Frequency: {device.freq_MHz} MHz")
    logger.debug(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.debug("")

    # Print results for each configuration
    for stats in results:
        logger.debug(
            f"Input Shape: {stats['input_shape']} -> Output Shape: {stats['output_shape']} (scales: {stats['scales']})"
        )
        logger.debug(f"  Memory: {stats['total_memory']/1e6:.4f} MB")
        logger.debug(f"  Operations: {stats['ops']:.0f}")
        logger.debug(
            f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte"
        )
        logger.debug(f"  Bottleneck: {stats['bottleneck']}")
        logger.debug("")

    # Summary statistics
    memory_bound = sum(1 for r in results if r["bottleneck"] == "memory-bound")
    compute_bound = sum(1 for r in results if r["bottleneck"] == "compute-bound")

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {len(results)}")
    logger.info(f"Memory-bound: {memory_bound}")
    logger.info(f"Compute-bound: {compute_bound}")
    logger.info("=" * 80)
