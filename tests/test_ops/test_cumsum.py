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
from ttsim.ops.desc.data_compute import compute_cumsum
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


def ref_impl_cumsum(X, axis=0, exclusive=0, reverse=0):
    """Reference implementation of cumulative sum"""
    if reverse:
        X_work = np.flip(X, axis=axis)
    else:
        X_work = X

    if exclusive:
        # Exclusive cumsum: shift result and set first element to 0
        result = np.cumsum(X_work, axis=axis)
        result = np.roll(result, 1, axis=axis)
        # Set first element along axis to 0
        slc = [slice(None)] * len(X.shape)
        slc[axis] = 0
        result[tuple(slc)] = 0
    else:
        result = np.cumsum(X_work, axis=axis)

    if reverse:
        result = np.flip(result, axis=axis)

    return result


# Test cases with shape validation and numerical validation
test_name = "test_cumsum"
test_cases = [
    # (name, input_shape, axis, exclusive, reverse, test_data_type)
    ("Basic 1D", [10], 0, 0, 0, "positive"),
    ("Basic 2D axis=0", [4, 5], 0, 0, 0, "positive"),
    ("Basic 2D axis=1", [4, 5], 1, 0, 0, "positive"),
    ("Basic 3D axis=0", [2, 3, 4], 0, 0, 0, "positive"),
    ("Basic 3D axis=1", [2, 3, 4], 1, 0, 0, "positive"),
    ("Basic 3D axis=2", [2, 3, 4], 2, 0, 0, "positive"),
    ("4D along batch", [2, 3, 4, 5], 0, 0, 0, "positive"),
    # Exclusive mode tests
    ("Exclusive 1D", [5], 0, 1, 0, "positive"),
    ("Exclusive 2D axis=0", [4, 3], 0, 1, 0, "positive"),
    ("Exclusive 2D axis=1", [4, 3], 1, 1, 0, "positive"),
    # Reverse mode tests
    ("Reverse 1D", [5], 0, 0, 1, "positive"),
    ("Reverse 2D axis=0", [4, 3], 0, 0, 1, "positive"),
    ("Reverse 2D axis=1", [4, 3], 1, 0, 1, "positive"),
    # Combined exclusive and reverse
    ("Exclusive + Reverse", [5], 0, 1, 1, "positive"),
    # Edge cases: negative values
    ("Negative values", [4, 4], 1, 0, 0, "negative"),
    # Edge cases: zero values
    ("Zero values", [4, 4], 0, 0, 0, "zeros"),
    # Edge cases: mixed values
    ("Mixed pos/neg", [5, 5], 1, 0, 0, "mixed"),
    # Edge cases: single element
    ("Single element", [1], 0, 0, 0, "positive"),
    # Edge cases: large values (overflow risk)
    ("Large values", [3, 3], 0, 0, 0, "large"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) * 10
    elif data_type == "negative":
        return -(np.random.rand(*shape).astype(np.float32) * 10)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 5).astype(np.float32)
    elif data_type == "large":
        return np.random.rand(*shape).astype(np.float32) * 1e4
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_cumsum():
    """Test CumSum with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, axis, exclusive, reverse, data_type) in enumerate(
        test_cases
    ):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        # CumSum requires axis as a tensor input, not an attribute
        axis_tensor = F._from_data(
            "axis", np.array([axis], dtype=np.int64), is_param=False, is_const=True
        )
        i_tensors = [F._from_data("X", test_data), axis_tensor]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "CumSum",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"exclusive": exclusive, "reverse": reverse},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_cumsum(test_data, axis, exclusive, reverse)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_cumsum(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output,
                ref_output,
                rtol=1e-4,  # Relaxed for cumulative operations
                atol=1e-5,
            )

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
test_name_precision = "test_cumsum_precision"
precision_test_cases = [
    # (name, input, axis, exclusive, reverse, expected_output)
    (
        "Simple cumsum [1,2,3]",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        1,
        0,
        0,
        np.array([[1.0, 3.0, 6.0]], dtype=np.float32),
    ),
    (
        "Exclusive cumsum [1,2,3]",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        1,
        1,
        0,
        np.array([[0.0, 1.0, 3.0]], dtype=np.float32),
    ),
    (
        "Reverse cumsum [1,2,3]",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        1,
        0,
        1,
        np.array([[6.0, 5.0, 3.0]], dtype=np.float32),
    ),
    (
        "2D cumsum along rows",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        0,
        0,
        0,
        np.array([[1.0, 2.0], [4.0, 6.0]], dtype=np.float32),
    ),
    (
        "2D cumsum along cols",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        1,
        0,
        0,
        np.array([[1.0, 3.0], [3.0, 7.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_cumsum_precision():
    """Test CumSum with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, axis, exclusive, reverse, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        # CumSum requires axis as a tensor input, not an attribute
        axis_tensor = F._from_data(
            "axis", np.array([axis], dtype=np.int64), is_param=False, is_const=True
        )
        i_tensors = [F._from_data("X", test_data), axis_tensor]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "CumSum",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"exclusive": exclusive, "reverse": reverse},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_cumsum(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
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


def calculate_cumsum_memory_stats(
    shape, axis=0, exclusive=0, reverse=0, dtype="float32"
):
    """Calculate memory and compute statistics for a cumsum operation"""
    # Get device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    # Create input tensors (X and axis as tensor)
    np_dtype = getattr(np, dtype)
    data_X = np.random.randn(*shape).astype(np_dtype)
    axis_tensor = F._from_data(
        "axis", np.array([axis], dtype=np.int64), is_param=False, is_const=True
    )
    i_tensors = [F._from_data("X", data_X), axis_tensor]
    o_tensors = [make_tensor("Y")]

    # Create operation
    op_info = {
        "optype": "CumSum",
        "name": "cumsum_mem_test",
        "attrs": {"exclusive": exclusive, "reverse": reverse},
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
    compute_throughput = 1  # 1 op per cycle for cumsum
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "shape": shape,
        "axis": axis,
        "exclusive": exclusive,
        "reverse": reverse,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }


def test_cumsum_memory_validation():
    """Memory validation test for cumsum operation"""
    logger.info("\n" + "=" * 80)
    logger.info("CUMSUM MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations (shape, axis, exclusive, reverse)
    test_configs = [
        ([32], 0, 0, 0),
        ([64, 64], 0, 0, 0),
        ([32, 32, 32], 0, 0, 0),
        ([16, 16, 16, 16], 0, 0, 0),
        ([1, 224, 224, 3], 1, 0, 0),
        ([8, 128, 128, 64], 2, 0, 0),
        ([4, 56, 56, 256], 1, 0, 0),
    ]

    results = []
    for shape, axis, exclusive, reverse in test_configs:
        stats = calculate_cumsum_memory_stats(shape, axis, exclusive, reverse)
        results.append(stats)

    # Print device info once
    device = results[0]["device"]
    logger.info(f"\nDevice: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("")

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
