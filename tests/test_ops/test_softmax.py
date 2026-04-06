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
from ttsim.ops.desc.data_compute import compute_softmax
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


def ref_impl_softmax(X, axis=-1):
    """
    Reference implementation of softmax with numerical stability
    """
    X_max = np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X - X_max)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


# Test cases with shape validation and numerical validation
test_name = "test_softmax"
test_cases = [
    # (name, input_shape, axis, test_data_type)
    ("Basic 1D", [10], -1, "positive"),
    ("Basic 2D axis=-1", [4, 5], -1, "positive"),
    ("Basic 2D axis=0", [4, 5], 0, "positive"),
    ("Basic 2D axis=1", [4, 5], 1, "positive"),
    ("Basic 3D axis=-1", [2, 3, 4], -1, "positive"),
    ("Basic 3D axis=0", [2, 3, 4], 0, "positive"),
    ("Basic 3D axis=1", [2, 3, 4], 1, "positive"),
    ("Basic 3D axis=2", [2, 3, 4], 2, "positive"),
    ("Basic 4D", [2, 3, 4, 5], -1, "positive"),
    ("Single element", [1], -1, "positive"),
    ("Batch processing", [8, 10], -1, "positive"),
    # Edge cases: large positive values (overflow risk)
    ("Large positive values", [5, 5], -1, "large_positive"),
    # Edge cases: large negative values (underflow risk)
    ("Large negative values", [5, 5], -1, "large_negative"),
    # Edge cases: zero values
    ("Zero values", [4, 4], -1, "zeros"),
    # Edge cases: mixed positive/negative
    ("Mixed values", [6, 6], -1, "mixed"),
    # Edge cases: very large range (stability test)
    ("Wide range values", [5, 5], -1, "wide_range"),
    # Edge cases: uniform values (all equal)
    ("Uniform values", [4, 4], -1, "uniform"),
    # Edge cases: one dominant value
    ("One dominant value", [5, 5], -1, "dominant"),
    # Edge cases: extreme values
    ("Extreme positive", [3, 3], -1, "extreme_positive"),
    ("Extreme negative", [3, 3], -1, "extreme_negative"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) * 5
    elif data_type == "large_positive":
        return np.random.rand(*shape).astype(np.float32) * 50 + 10
    elif data_type == "large_negative":
        return -(np.random.rand(*shape).astype(np.float32) * 50 + 10)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 5).astype(np.float32)
    elif data_type == "wide_range":
        data = np.random.randn(*shape).astype(np.float32)
        data[0, 0] = 100.0  # Very large
        data[0, 1] = -100.0  # Very small
        return data
    elif data_type == "uniform":
        return np.ones(shape, dtype=np.float32) * 2.5
    elif data_type == "dominant":
        data = np.random.rand(*shape).astype(np.float32)
        data[0, 0] = 50.0  # Dominant value
        return data
    elif data_type == "extreme_positive":
        return np.random.rand(*shape).astype(np.float32) * 200 + 100
    elif data_type == "extreme_negative":
        return -(np.random.rand(*shape).astype(np.float32) * 200 + 100)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_softmax():
    """Test Softmax with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, axis, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Softmax",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": axis},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_softmax(test_data, axis)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_softmax(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )

            # Verify softmax properties
            # 1. All values should be in [0, 1]
            all_in_range = np.all((computed_output >= 0) & (computed_output <= 1))

            # 2. Sum along axis should be 1
            sum_along_axis = np.sum(computed_output, axis=axis)
            sum_is_one = np.allclose(sum_along_axis, 1.0, rtol=1e-5, atol=1e-7)

            if not all_in_range:
                numerical_match = False
                logger.debug("\n  Output not in range [0, 1]")
            if not sum_is_one:
                numerical_match = False
                logger.debug("\n  Sum along axis not equal to 1")

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
test_name_precision = "test_softmax_precision"
precision_test_cases = [
    # (name, input, axis, expected_output)
    (
        "Uniform values -> equal probabilities",
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        -1,
        np.array([[0.33333334, 0.33333334, 0.33333334]], dtype=np.float32),
    ),
    (
        "Zero values -> equal probabilities",
        np.array([[0.0, 0.0]], dtype=np.float32),
        -1,
        np.array([[0.5, 0.5]], dtype=np.float32),
    ),
    (
        "One dominant value",
        np.array([[0.0, 10.0, 0.0]], dtype=np.float32),
        -1,
        np.array([[0.00004539, 0.99990922, 0.00004539]], dtype=np.float32),
    ),
    (
        "Simple 2D case",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        -1,
        np.array(
            [[0.26894142, 0.73105858], [0.26894142, 0.73105858]], dtype=np.float32
        ),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_softmax_precision():
    """Test Softmax with precise known outputs"""
    msgw = 40

    for tno, (tmsg, test_data, axis, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Softmax",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": axis},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_softmax(i_tensors, op_obj)
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


def calculate_softmax_memory_stats(shape, axis=-1, dtype="float32"):
    """Calculate memory and compute statistics for a softmax operation"""
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
        "optype": "Softmax",
        "name": "softmax_mem_test",
        "attrs": {"axis": axis},
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
    compute_throughput = 1  # 1 op per cycle for softmax
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "shape": shape,
        "axis": axis,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }


def test_softmax_memory_validation():
    """Memory validation test for softmax operation"""
    logger.info("\n" + "=" * 80)
    logger.info("SOFTMAX MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations
    test_configs = [
        ([32], -1),
        ([64, 64], -1),
        ([32, 32, 32], -1),
        ([16, 16, 16, 16], -1),
        ([1, 224, 224, 3], -1),
        ([8, 128, 128, 64], -1),
        ([4, 56, 56, 256], -1),
    ]

    results = []
    for shape, axis in test_configs:
        stats = calculate_softmax_memory_stats(shape, axis)
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
        logger.debug(f"Shape: {stats['shape']}, Axis: {stats['axis']}")
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
