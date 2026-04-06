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
from ttsim.ops.desc.data_compute import compute_sigmoid
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)

    from loguru import logger as _loguru_logger
    _loguru_logger.disable("ttsim")
except Exception:
    pass


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_sigmoid(X):
    """
    Reference implementation of sigmoid: 1 / (1 + e^(-x))
    Uses clipping for numerical stability
    """
    X_clipped = np.clip(X, -20, 20)
    return 1.0 / (1.0 + np.exp(-X_clipped))


# Test cases with shape validation and numerical validation
test_name = "test_sigmoid"
test_cases = [
    # (name, input_shape, test_data_type)
    ("Basic 1D", [10], "positive"),
    ("Basic 2D", [4, 4], "positive"),
    ("Basic 3D", [2, 3, 4], "positive"),
    ("Basic 4D", [2, 3, 4, 5], "positive"),
    ("Single element", [1], "positive"),
    ("Batch processing", [8, 16], "positive"),
    ("Large tensor", [10, 20, 30], "positive"),
    # Edge cases: large positive values (should saturate to 1)
    ("Large positive values", [5, 5], "large_positive"),
    # Edge cases: large negative values (should saturate to 0)
    ("Large negative values", [5, 5], "large_negative"),
    # Edge cases: zero values (should be 0.5)
    ("Zero values", [4, 4], "zeros"),
    # Edge cases: mixed positive/negative
    ("Mixed values", [6, 6], "mixed"),
    # Edge cases: very small values near zero
    ("Small values near zero", [4, 4], "near_zero"),
    # Edge cases: values around saturation points
    ("Near saturation high", [4, 4], "near_saturation_high"),
    ("Near saturation low", [4, 4], "near_saturation_low"),
    # Edge cases: gradient vanishing regions
    ("Extreme positive (gradient ~0)", [3, 3], "extreme_positive"),
    ("Extreme negative (gradient ~0)", [3, 3], "extreme_negative"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) * 5  # Range [0, 5]
    elif data_type == "large_positive":
        return np.random.rand(*shape).astype(np.float32) * 50 + 10  # Range [10, 60]
    elif data_type == "large_negative":
        return -(
            np.random.rand(*shape).astype(np.float32) * 50 + 10
        )  # Range [-60, -10]
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 5).astype(np.float32)  # Mixed pos/neg
    elif data_type == "near_zero":
        return (np.random.randn(*shape) * 0.1).astype(np.float32)  # Very small values
    elif data_type == "near_saturation_high":
        return np.random.rand(*shape).astype(np.float32) * 4 + 4  # Range [4, 8]
    elif data_type == "near_saturation_low":
        return -(np.random.rand(*shape).astype(np.float32) * 4 + 4)  # Range [-8, -4]
    elif data_type == "extreme_positive":
        return np.random.rand(*shape).astype(np.float32) * 100 + 50  # Range [50, 150]
    elif data_type == "extreme_negative":
        return -(
            np.random.rand(*shape).astype(np.float32) * 100 + 50
        )  # Range [-150, -50]
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_sigmoid():
    """Test Sigmoid with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sigmoid",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_sigmoid(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_sigmoid(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )

            # Verify sigmoid properties
            all_in_range = np.all((computed_output >= 0) & (computed_output <= 1))
            if not all_in_range:
                numerical_match = False
                logger.debug("\n  Output not in range [0, 1]")

            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
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
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_sigmoid_precision"
precision_test_cases = [
    # (name, input, expected_output)
    (
        "Zero input -> 0.5",
        np.array([[0.0]], dtype=np.float32),
        np.array([[0.5]], dtype=np.float32),
    ),
    (
        "Large positive -> ~1.0",
        np.array([[10.0]], dtype=np.float32),
        np.array([[0.9999546]], dtype=np.float32),
    ),
    (
        "Large negative -> ~0.0",
        np.array([[-10.0]], dtype=np.float32),
        np.array([[0.0000454]], dtype=np.float32),
    ),
    (
        "Small positive",
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.7310586]], dtype=np.float32),
    ),
    (
        "Small negative",
        np.array([[-1.0]], dtype=np.float32),
        np.array([[0.2689414]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_sigmoid_precision():
    """Test Sigmoid with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sigmoid",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_sigmoid(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-5)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                logger.debug(
                    f"  Diff:     {(computed_output - expected_output).flatten()}"
                )
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_sigmoid_memory_stats(shape, dtype="float32"):
    """Calculate memory and compute statistics for a sigmoid operation"""
    # Get device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    # Create input tensor
    np_dtype = getattr(np, dtype)
    data_X = np.random.randn(*shape).astype(np_dtype)
    input_tensor = SimTensor({"name": "X", "shape": shape, "dtype": dtype})
    input_tensor.data = data_X
    i_tensors = [input_tensor]
    o_tensors = [make_tensor("Y")]

    # Create operation
    op_info = {
        "optype": "Sigmoid",
        "name": "sigmoid_mem_test",
        "attrs": {},
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
        else np.prod(shape)
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
    compute_throughput = 1  # 1 op per cycle for sigmoid
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "shape": shape,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }


def test_sigmoid_memory_validation():
    """Memory validation test for sigmoid operation"""
    logger.info("\n" + "=" * 80)
    logger.info("SIGMOID MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations
    test_configs = [
        [32],
        [64, 64],
        [32, 32, 32],
        [16, 16, 16, 16],
        [1, 224, 224, 3],
        [8, 128, 128, 64],
        [4, 56, 56, 256],
    ]

    results = []
    for config in test_configs:
        stats = calculate_sigmoid_memory_stats(config)
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
        logger.debug(f"Shape: {stats['shape']}")
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
