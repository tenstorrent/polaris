#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os, logging
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_sign
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device
from loguru import logger

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_sign(X):
    """Reference implementation of sign function"""
    return np.sign(X)


# Test cases with shape validation and numerical validation
test_name = "test_sign"
test_cases = [
    # (name, input_shape, test_data_type)
    ("Basic 1D", [10], "positive"),
    ("Basic 2D", [4, 4], "positive"),
    ("Basic 3D", [2, 3, 4], "positive"),
    ("Basic 4D", [2, 3, 4, 5], "positive"),
    ("Single element", [1], "positive"),
    ("Batch processing", [8, 16], "positive"),
    ("Large tensor", [10, 20, 30], "positive"),
    # Edge cases: negative values (should be -1)
    ("Negative values", [5, 5], "negative"),
    # Edge cases: zero values (should be 0)
    ("Zero values", [4, 4], "zeros"),
    # Edge cases: mixed positive/negative/zero
    ("Mixed values", [6, 6], "mixed"),
    # Edge cases: very small positive values (should be +1)
    ("Small positive", [4, 4], "small_positive"),
    # Edge cases: very small negative values (should be -1)
    ("Small negative", [4, 4], "small_negative"),
    # Edge cases: large positive values (should be +1)
    ("Large positive", [3, 3], "large_positive"),
    # Edge cases: large negative values (should be -1)
    ("Large negative", [3, 3], "large_negative"),
    # Edge cases: mixed with zeros
    ("Mixed with zeros", [5, 5], "mixed_with_zeros"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) * 10 + 0.1
    elif data_type == "negative":
        return -(np.random.rand(*shape).astype(np.float32) * 10 + 0.1)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 5).astype(np.float32)
    elif data_type == "small_positive":
        return np.random.rand(*shape).astype(np.float32) * 1e-6
    elif data_type == "small_negative":
        return -(np.random.rand(*shape).astype(np.float32) * 1e-6)
    elif data_type == "large_positive":
        return np.random.rand(*shape).astype(np.float32) * 1e6
    elif data_type == "large_negative":
        return -(np.random.rand(*shape).astype(np.float32) * 1e6)
    elif data_type == "mixed_with_zeros":
        data = (np.random.randn(*shape) * 5).astype(np.float32)
        # Set some random elements to zero
        mask = np.random.rand(*shape) < 0.2
        data[mask] = 0
        return data
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_sign():
    """Test Sign with shape validation, edge cases, and numerical validation"""
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
            "optype": "Sign",
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
        ref_output = ref_impl_sign(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_sign(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )

            # Verify sign properties: output should be -1, 0, or +1
            all_valid = np.all(np.isin(computed_output, [-1.0, 0.0, 1.0]))
            if not all_valid:
                numerical_match = False
                assert False, f"\n  Output contains invalid values (not -1, 0, or 1)"

            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                assert False, f"\n  Max difference: {max_diff}"
        except Exception as e:
            numerical_match = f"Error: {e}"
            assert False, f"\n  Numerical validation error: {e}"

        # Report results
        if shape_match and numerical_match == True:
            logger.info(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            logger.info(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.error(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.error(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.error(f"  Numerical match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_sign_precision"
precision_test_cases = [
    # (name, input, expected_output)
    (
        "Positive values -> +1",
        np.array([[1.0, 5.0, 100.0]], dtype=np.float32),
        np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
    ),
    (
        "Negative values -> -1",
        np.array([[-1.0, -5.0, -100.0]], dtype=np.float32),
        np.array([[-1.0, -1.0, -1.0]], dtype=np.float32),
    ),
    (
        "Zero -> 0",
        np.array([[0.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    ),
    (
        "Mixed values",
        np.array([[1.0, -1.0, 0.0, 5.0, -5.0]], dtype=np.float32),
        np.array([[1.0, -1.0, 0.0, 1.0, -1.0]], dtype=np.float32),
    ),
    (
        "Small values",
        np.array([[1e-10, -1e-10, 0.0]], dtype=np.float32),
        np.array([[1.0, -1.0, 0.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_sign_precision():
    """Test Sign with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Sign",
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
            computed_output = compute_sign(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-7, atol=1e-7)
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


def calculate_sign_memory_stats(shape):
    """Calculate memory access and arithmetic operations for sign operation"""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    X = SimTensor({"name": "X", "shape": shape, "dtype": "float32"})
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_sign",
        "optype": "Sign",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_sign"]
    for x in o_tensors:
        x.op_out = ["test_sign"]

    op.get_perf_counts(i_tensors, o_tensors)

    perf_stats = op.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    instr_sum = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else 0
    )
    ops = instr_sum if instr_sum > 0 else perf_stats.get("inElems", np.prod(shape))
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0

    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "memory_mb": total_memory / (1024 * 1024),
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
    }


@pytest.mark.performance
@pytest.mark.slow
def test_sign_memory_validation():
    """Test Sign memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    logger.info("\n" + "=" * 80)
    logger.info("SIGN MEMORY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Device: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("=" * 80)

    # Typical activation shapes [N, C, H, W]
    test_configs = [
        [1, 64, 56, 56],
        [1, 128, 28, 28],
        [1, 256, 14, 14],
        [4, 64, 56, 56],
        [4, 128, 28, 28],
        [8, 256, 14, 14],
        [16, 64, 32, 32],
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape in test_configs:
        stats = calculate_sign_memory_stats(shape)

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
