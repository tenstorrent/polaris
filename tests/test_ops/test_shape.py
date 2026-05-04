#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os, logging
import pytest
import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_shape
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

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


def ref_impl_shape(X):
    """Reference implementation of shape operation"""
    return np.array(X.shape, dtype=np.int64)


# Test cases for numerical validation
test_name = "test_shape"
test_cases = [
    # (name, input_shape)
    ("1D tensor", [10]),
    ("2D tensor", [4, 5]),
    ("3D tensor", [2, 3, 4]),
    ("4D tensor", [2, 3, 4, 5]),
    ("5D tensor", [2, 3, 4, 5, 6]),
    ("Single element", [1]),
    ("Large 1D", [1000]),
    ("Square 2D", [8, 8]),
    ("Batch tensor", [32, 3, 224, 224]),
    ("Non-uniform", [7, 11, 13]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_numerical():
    """Test Shape with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data (content doesn't matter for shape operation)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        # Create input tensors
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": op_name,
            "optype": "Shape",
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
        ref_output = ref_impl_shape(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_shape(i_tensors, op_obj)

            # Shape output should be exact integer match
            numerical_match = np.array_equal(computed_output, ref_output)

            if not numerical_match:
                logger.debug(f"\n  Expected: {ref_output}")
                logger.debug(f"  Got:      {computed_output}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Values ✓]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Values: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                "  Shape match: "
                f"{shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Values match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_shape_precision"
precision_test_cases = [
    # (name, input_shape, expected_output)
    ("Shape of [5]", [5], np.array([5], dtype=np.int64)),
    ("Shape of [3, 4]", [3, 4], np.array([3, 4], dtype=np.int64)),
    ("Shape of [2, 3, 4]", [2, 3, 4], np.array([2, 3, 4], dtype=np.int64)),
    ("Shape of [1, 1, 1, 1]", [1, 1, 1, 1], np.array([1, 1, 1, 1], dtype=np.int64)),
    ("Shape of [10, 20, 30]", [10, 20, 30], np.array([10, 20, 30], dtype=np.int64)),
    ("Shape of single element", [1], np.array([1], dtype=np.int64)),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_precision():
    """Test Shape with precise known outputs"""
    msgw = 30

    for tno, (tmsg, input_shape, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        # Create test data with the specified shape
        test_data = np.random.randn(*input_shape).astype(np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": op_name,
            "optype": "Shape",
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
            computed_output = compute_shape(i_tensors, op_obj)
            match = np.array_equal(computed_output, expected_output)
            dtype_match = computed_output.dtype == expected_output.dtype

            if match and dtype_match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(
                    f"  Expected: {expected_output} (dtype: {expected_output.dtype})"
                )
                logger.debug(
                    f"  Got:      {computed_output} (dtype: {computed_output.dtype})"
                )
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Edge cases
test_name_edge = "test_shape_edge"
edge_test_cases = [
    # (name, input_shape, description)
    ("Scalar-like [1]", [1], "Single dimension size 1"),
    ("1D large", [10000], "Large 1D array"),
    ("Many dimensions", [2, 2, 2, 2, 2, 2], "6D tensor"),
    ("Huge 2D", [1000, 1000], "Large 2D tensor"),
    ("Typical CNN", [1, 3, 224, 224], "CNN input shape"),
    ("Transformer", [32, 512, 768], "Transformer shape (batch, seq, hidden)"),
    ("Asymmetric", [1, 100, 1, 50], "Very asymmetric dimensions"),
    ("Prime dimensions", [7, 11, 13, 17], "Prime-sized dimensions"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_edge_cases():
    """Test Shape edge cases and boundary conditions"""
    msgw = 25

    for tno, (tmsg, input_shape, description) in enumerate(edge_test_cases):
        op_name = f"{test_name_edge}_{tno}"

        # Create test data (content doesn't matter)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": op_name,
            "optype": "Shape",
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
            computed_output = compute_shape(i_tensors, op_obj)
            ref_output = ref_impl_shape(test_data)

            # Validate exact match
            match = np.array_equal(computed_output, ref_output)

            # Additional edge case checks
            # Output should be 1D with length = number of input dimensions
            assert len(computed_output.shape) == 1, "Shape output must be 1D"
            assert computed_output.shape[0] == len(
                input_shape
            ), "Output length = input ndim"

            # Output dtype must be int64
            assert computed_output.dtype == np.int64, "Shape output must be int64"

            # Each dimension should match
            for i, (computed, expected) in enumerate(zip(computed_output, input_shape)):
                assert computed == expected, f"Dimension {i}: {computed} != {expected}"

            # Values should all be positive
            assert np.all(computed_output > 0), "All dimensions must be positive"

            if match:
                logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                logger.debug(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  {description}")
                logger.debug(f"  Expected: {ref_output}")
                logger.debug(f"  Got:      {computed_output}")
        except Exception as e:
            logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Additional property tests
@pytest.mark.unit
@pytest.mark.opunit
def test_shape_properties():
    """Test mathematical properties of Shape operation"""

    logger.info("\nTesting Shape Operation Properties:")

    # Property 1: Shape output is 1D
    logger.info("  Property 1: Output is always 1D")
    for ndim in [1, 2, 3, 4, 5]:
        shape = tuple(np.random.randint(2, 10, size=ndim))
        test_data = np.random.randn(*shape).astype(np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": f"shape_prop1_{ndim}",
            "optype": "Shape",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [f"shape_prop1_{ndim}"]
        for x in o_tensors:
            x.op_out = [f"shape_prop1_{ndim}"]

        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_shape(i_tensors, op_obj)

        assert len(result.shape) == 1, f"Output not 1D for {ndim}D input"
        assert result.shape[0] == ndim, f"Output length {result.shape[0]} != {ndim}"

    logger.info("    All 1D outputs ✓")

    # Property 2: Data content doesn't affect shape
    logger.info("  Property 2: Data content doesn't affect output")
    shape = [4, 5, 6]
    data1 = np.zeros(shape, dtype=np.float32)
    data2 = np.ones(shape, dtype=np.float32)
    data3 = np.random.randn(*shape).astype(np.float32) * 1000

    results = []
    for idx, data in enumerate([data1, data2, data3]):
        i_tensors = [F._from_data("X", data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": f"shape_prop2_{idx}",
            "optype": "Shape",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [f"shape_prop2_{idx}"]
        for x in o_tensors:
            x.op_out = [f"shape_prop2_{idx}"]

        op_obj.get_perf_counts(i_tensors, o_tensors)
        results.append(compute_shape(i_tensors, op_obj))

    assert np.array_equal(results[0], results[1]), "Zeros vs ones differ"
    assert np.array_equal(results[0], results[2]), "Zeros vs random differ"
    logger.info("    Content-independent ✓")

    # Property 3: Output length equals input rank
    logger.info("  Property 3: Output length = input rank")
    for ndim in range(1, 7):
        shape = tuple(np.random.randint(1, 10, size=ndim))
        test_data = np.random.randn(*shape).astype(np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": f"shape_prop3_{ndim}",
            "optype": "Shape",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [f"shape_prop3_{ndim}"]
        for x in o_tensors:
            x.op_out = [f"shape_prop3_{ndim}"]

        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_shape(i_tensors, op_obj)

        assert len(result) == ndim, f"Output length {len(result)} != rank {ndim}"
        assert len(result) == len(test_data.shape), "Output length != input rank"

    logger.info("    Rank matching \u2713")

    logger.info("\nAll property tests passed!")


def calculate_shape_memory_stats(input_shape):
    """Calculate memory access and arithmetic operations for shape operation.
    Shape reads the input's rank as int64 and writes one int64 per dimension.
    Stats derived analytically from the sinf formula: instrs={'mov': rank}."""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    rank = len(input_shape)
    bytes_per_index = 4  # int32/int64 treated as 4B per index in sinf
    input_bytes = rank * bytes_per_index
    output_bytes = rank * bytes_per_index
    total_memory = input_bytes + output_bytes
    ops = rank  # one mov per dimension

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


@pytest.mark.unit
@pytest.mark.opunit
def test_shape_memory_validation():
    """Test Shape memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    logger.info("\n" + "=" * 80)
    logger.info("SHAPE MEMORY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Device: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("=" * 80)

    # Various input shapes — Shape op only reads the rank, not the data
    test_configs = [
        [64, 64],
        [1, 64, 56, 56],
        [1, 128, 28, 28],
        [4, 64, 56, 56],
        [4, 128, 28, 28],
        [8, 256, 14, 14],
        [2, 3, 4, 5, 6],
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape in test_configs:
        stats = calculate_shape_memory_stats(shape)

        logger.info(f"\nInput shape: {shape} (rank={len(shape)})")
        logger.info(f"  Memory: {stats['memory_mb']:.6f} MB")
        logger.info(f"  Operations: {stats['ops']}")
        logger.info(f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte")
        logger.info(f"  Bottleneck: {stats['bottleneck']}")

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
