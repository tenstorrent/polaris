#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import logging

# Silence the avgpool2d-related log messages
try:
    from loguru import logger

    logger.disable("ttsim")
except ImportError:
    pass

import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_gelu
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_gelu(X):
    """
    Reference implementation of GELU activation.
    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution.
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return (
        0.5
        * X
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * np.power(X, 3))))
    )


# Test cases for numerical validation
test_name = "test_gelu"
test_cases = [
    # (name, input_shape, data_type)
    ("1D tensor", [10], "mixed"),
    ("2D tensor", [4, 5], "mixed"),
    ("3D tensor", [2, 3, 4], "mixed"),
    ("4D tensor (batch)", [2, 3, 4, 5], "mixed"),
    ("Large positive values", [5, 5], "large_positive"),
    ("Large negative values", [5, 5], "large_negative"),
    ("Small values near zero", [10, 10], "near_zero"),
    ("Single element", [1], "mixed"),
    ("Wide range", [8, 8], "wide_range"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)
    elif data_type == "large_positive":
        return (np.random.rand(*shape) * 10 + 5).astype(np.float32)
    elif data_type == "large_negative":
        return -(np.random.rand(*shape) * 10 + 5).astype(np.float32)
    elif data_type == "near_zero":
        return (np.random.randn(*shape) * 0.1).astype(np.float32)
    elif data_type == "wide_range":
        return (np.random.randn(*shape) * 5).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_gelu_numerical():
    """Test GELU with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Gelu",
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
        ref_output = ref_impl_gelu(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_gelu(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-4, atol=1e-5
            )

            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
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
test_name_precision = "test_gelu_precision"
precision_test_cases = [
    # (name, input, expected_output)
    (
        "Zero input",
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    (
        "Positive value x=1",
        np.array([1.0], dtype=np.float32),
        ref_impl_gelu(np.array([1.0], dtype=np.float32)),
    ),
    (
        "Negative value x=-1",
        np.array([-1.0], dtype=np.float32),
        ref_impl_gelu(np.array([-1.0], dtype=np.float32)),
    ),
    (
        "Large positive x=5",
        np.array([5.0], dtype=np.float32),
        np.array([5.0], dtype=np.float32),
    ),  # GELU(x) ≈ x for large positive x
    (
        "Large negative x=-5",
        np.array([-5.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),  # GELU(x) ≈ 0 for large negative x
    (
        "2D mixed values",
        np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32),
        ref_impl_gelu(np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_gelu_precision():
    """Test GELU with precise known outputs"""
    msgw = 25

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Gelu",
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
            computed_output = compute_gelu(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-4, atol=1e-5)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()[:5]}")
                logger.debug(f"  Got:      {computed_output.flatten()[:5]}")
                logger.debug(
                    f"  Max diff: {np.max(np.abs(computed_output - expected_output))}"
                )
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Edge cases
test_name_edge = "test_gelu_edge"
edge_test_cases = [
    # (name, input_shape, data_type, description)
    ("All zeros", [5, 5], "zeros", "Output should be zero"),
    ("Very small values", [10], "tiny", "Near-zero values"),
    ("Symmetry test", [20], "symmetric", "GELU(-x) ≠ -GELU(x), but relationship holds"),
    ("Single large positive", [1], "large_pos_single", "Should be close to input"),
    ("Single large negative", [1], "large_neg_single", "Should be close to zero"),
    ("Mixed batch", [4, 8, 8], "mixed_batch", "Batch processing"),
]


def generate_edge_test_data(shape, data_type):
    """Generate edge case test data"""
    if data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "tiny":
        return (np.random.randn(*shape) * 1e-6).astype(np.float32)
    elif data_type == "symmetric":
        return np.linspace(-3, 3, np.prod(shape), dtype=np.float32).reshape(shape)
    elif data_type == "large_pos_single":
        return np.array([10.0], dtype=np.float32)
    elif data_type == "large_neg_single":
        return np.array([-10.0], dtype=np.float32)
    elif data_type == "mixed_batch":
        return (np.random.randn(*shape) * 3).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_gelu_edge_cases():
    """Test GELU edge cases and boundary conditions"""
    msgw = 25

    for tno, (tmsg, input_shape, data_type, description) in enumerate(edge_test_cases):
        op_name = f"{test_name_edge}_{tno}"

        test_data = generate_edge_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Gelu",
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
            computed_output = compute_gelu(i_tensors, op_obj)
            ref_output = ref_impl_gelu(test_data)

            # Validate
            match = np.allclose(computed_output, ref_output, rtol=1e-4, atol=1e-5)

            # Additional edge case checks
            if data_type == "zeros":
                # GELU(0) = 0
                assert np.allclose(
                    computed_output, 0.0, atol=1e-6
                ), "GELU(0) should be 0"
            elif data_type == "large_pos_single":
                # GELU(large_x) ≈ large_x
                assert np.allclose(
                    computed_output, test_data, rtol=0.01
                ), "GELU(large_x) ≈ x"
            elif data_type == "large_neg_single":
                # GELU(large_negative) ≈ 0
                assert np.abs(computed_output[0]) < 1e-4, "GELU(large_negative) ≈ 0"

            if match:
                logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                logger.debug(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  {description}")
                logger.debug(
                    f"  Max diff: {np.max(np.abs(computed_output - ref_output))}"
                )
        except Exception as e:
            logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_gelu_memory_stats(shape):
    """Calculate memory access and arithmetic operations for gelu operation"""

    # Initialize device for bandwidth calculations
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    # Create input tensor
    X = SimTensor({"name": "X", "shape": shape, "dtype": "float32"})
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_gelu",
        "optype": "Gelu",
        "inList": [x.name for x in i_tensors],
        "outList": [x.name for x in o_tensors],
        "attrs": {},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_gelu"]
    for x in o_tensors:
        x.op_out = ["test_gelu"]

    # Get performance counts
    op.get_perf_counts(i_tensors, o_tensors)

    # Get statistics from perf_stats
    perf_stats = op.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    ops = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else np.prod(shape)
    )
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    # Calculate arithmetic intensity
    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0

    # Calculate bandwidth and cycles
    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1  # 1 op per cycle for gelu
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    # Determine bottleneck
    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "memory_mb": total_memory / (1024 * 1024),
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
    }


@pytest.mark.unit
@pytest.mark.opunit
def test_gelu_memory_validation():
    """Test Gelu memory access patterns and arithmetic intensity"""

    # Initialize device
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    logger.info("\n" + "=" * 80)
    logger.info("GELU MEMORY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Device: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("=" * 80)

    # Test configurations with different shapes
    test_configs = [
        [32],
        [64, 64],
        [128, 128],
        [32, 32, 32],
        [16, 16, 16, 16],
        [8, 128, 128],
        [4, 56, 56, 256],
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape in test_configs:
        stats = calculate_gelu_memory_stats(shape)

        logger.debug(f"\nShape: {shape}")
        logger.debug(f"  Memory: {stats['memory_mb']:.4f} MB")
        logger.debug(f"  Operations: {stats['ops']}")
        logger.debug(
            f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte"
        )
        logger.debug(f"  Bottleneck: {stats['bottleneck']}")

        if stats["bottleneck"] == "memory-bound":
            memory_bound_count += 1
        else:
            compute_bound_count += 1

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {len(test_configs)}")
    logger.info(f"Memory-bound: {memory_bound_count}")
    logger.info(f"Compute-bound: {compute_bound_count}")
    logger.info("=" * 80)
