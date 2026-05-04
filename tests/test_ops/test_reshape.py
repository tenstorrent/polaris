#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
import logging
from functools import lru_cache
from loguru import logger

# Silence the ttsim-related log messages
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("ttsim")
except ImportError:
    pass

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F

try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    HAS_DEVICE_BACKEND = True
    DEVICE_BACKEND_IMPORT_ERROR = None
except ImportError as e:
    get_arspec_from_yaml = None  # type: ignore[assignment]
    Device = None  # type: ignore[assignment]
    HAS_DEVICE_BACKEND = False
    DEVICE_BACKEND_IMPORT_ERROR = e

RESHAPE_MEMORY_SKIP = pytest.mark.skipif(
    not HAS_DEVICE_BACKEND,
    reason="ttsim config/device backend not available for reshape memory validation",
)

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
from ttsim.ops.desc.data_compute import compute_reshape


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl(original_shape, target_shape, allowzero: int = 0):
    data = np.random.random_sample(original_shape).astype(np.float32)
    shape = np.array(target_shape, dtype=np.int64)
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped.shape


def _run_reshape(data, target_shape, op_name, attrs=None):
    """Build tensors + op, run shape inference + compute_reshape."""
    shape_arr = np.array(target_shape, dtype=np.int64)

    i_tensors = [F._from_data("X", data), F._from_data("S", shape_arr, is_const=True)]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": op_name,
        "optype": "Reshape",
        "inList": ["X", "S"],
        "outList": ["Y"],
        "attrs": attrs or {},
    }
    op_obj = SimOp(op_info)
    for x in i_tensors:
        x.op_in = [op_name]
    for x in o_tensors:
        x.op_out = [op_name]

    op_obj.get_perf_counts(i_tensors, o_tensors)
    computed = compute_reshape(i_tensors, op_obj)
    return computed, o_tensors


# --------------------------------------------------------------------------
# Shape-only test cases  (original tests)
# --------------------------------------------------------------------------
test_name = "test_reshape"
test_cases = [
    ("Basic reshape with -1, allowzero=0", [2, 3, 4], [2, -1], 0),
    ("Copy dim with 0, allowzero=0", [2, 3, 4], [2, 0, -1], 0),
    ("All specified dims, allowzero=0", [2, 3, 4], [4, 3, 2], 0),
    ("reordered_all_dims, allowzero=0", [2, 3, 4], [4, 2, 3], 0),
    ("reordered_last_dims, allowzero=0", [2, 3, 4], [2, 4, 3], 0),
    ("reduced_dims, allowzero=0", [2, 3, 4], [2, 12], 0),
    ("extended_dims, allowzero=0", [2, 3, 4], [2, 3, 2, 2], 0),
    ("one_dim, allowzero=0", [2, 3, 4], [24], 0),
    ("negative_dim, allowzero=0", [2, 3, 4], [2, -1, 2], 0),
    ("negative_extended_dims, allowzero=0", [2, 3, 4], [-1, 2, 3, 4], 0),
    ("zero_dim, allowzero=0", [2, 3, 4], [2, 0, 4, 1], 0),
    ("zero_and_negative_dim, allowzero=0", [2, 3, 4], [2, 0, 1, -1], 0),
    ("Basic reshape with -1, allowzero=1", [2, 3, 4], [2, -1], 1),
    ("All specified dims, allowzero=1", [2, 3, 4], [4, 3, 2], 1),
    ("reordered_all_dims, allowzero=1", [2, 3, 4], [4, 2, 3], 1),
    ("reordered_last_dims, allowzero=1", [2, 3, 4], [2, 4, 3], 1),
    ("reduced_dims, allowzero=1", [2, 3, 4], [2, 12], 1),
    ("extended_dims, allowzero=1", [2, 3, 4], [2, 3, 2, 2], 1),
    ("one_dim, allowzero=1", [2, 3, 4], [24], 1),
    ("negative_dim, allowzero=1", [2, 3, 4], [2, -1, 2], 1),
    ("negative_extended_dims, allowzero=1", [2, 3, 4], [-1, 2, 3, 4], 1),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape, target_shape, allowzero) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"
        i_tensors = [
            F._from_shape("X", input_shape, np_dtype=np.float32),
            F._from_data("S", np.array(target_shape), is_const=True),
        ]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "Reshape",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shape, target_shape, allowzero)
        ref_shape = list(ref_shape)

        if inf_shape == ref_shape:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug("\t", x)
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug("\t", x)
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# ==========================================================================
# Numerical validation tests
# ==========================================================================

num_test_cases = [
    # Flatten
    ("Flatten 2D to 1D", [3, 4], [12]),
    ("Flatten 3D to 1D", [2, 3, 4], [24]),
    ("Flatten 4D to 1D", [2, 3, 4, 5], [120]),
    # Expand dims
    ("1D to 2D", [12], [3, 4]),
    ("1D to 3D", [24], [2, 3, 4]),
    ("1D to 4D", [24], [1, 2, 3, 4]),
    # Same rank reshape
    ("2D to different 2D", [3, 4], [4, 3]),
    ("2D to different 2D (2)", [6, 4], [8, 3]),
    ("3D to different 3D", [2, 3, 4], [4, 3, 2]),
    # With -1 (infer)
    ("Infer last dim", [3, 4], [2, -1]),
    ("Infer first dim", [3, 4], [-1, 4]),
    ("Infer middle dim", [2, 3, 4], [2, -1, 4]),
    ("Flatten with -1", [2, 3, 4], [-1]),
    ("4D infer batch", [2, 3, 4, 5], [-1, 3, 4, 5]),
    # Identity reshape
    ("Same shape 1D", [12], [12]),
    ("Same shape 2D", [3, 4], [3, 4]),
    ("Same shape 4D", [2, 3, 4, 5], [2, 3, 4, 5]),
    # Single element
    ("Single element expand", [1], [1, 1, 1]),
    ("Single element collapse", [1, 1, 1], [1]),
    # Large
    ("Large flatten", [8, 16, 4], [512]),
    ("Large expand", [512], [8, 16, 4]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_numerical():
    """Numerical validation of compute_reshape across shapes"""

    msgw = get_max_test_msg_len(num_test_cases)

    for tno, (tmsg, in_shape, target_shape) in enumerate(num_test_cases):
        op_name = f"test_reshape_num_{tno}"

        data = np.array(np.random.randn(*in_shape), dtype=np.float32)

        # Resolve -1 for NumPy reference
        expected = np.reshape(data, target_shape)

        computed, o_tensors = _run_reshape(data, target_shape, op_name)

        # Shape check
        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(expected.shape)}"

        # Numerical check
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] numerical mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

edge_cases = [
    ("All zeros", [3, 4], [12], "zeros"),
    ("All ones", [2, 3], [6], "ones"),
    ("Negative values", [3, 4], [4, 3], "negative"),
    ("Large values", [3, 4], [2, 6], "large"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_edge_cases():
    """Edge cases for Reshape"""

    msgw = get_max_test_msg_len(edge_cases)

    for tno, (tmsg, in_shape, target_shape, data_gen) in enumerate(edge_cases):
        op_name = f"test_reshape_edge_{tno}"

        if data_gen == "zeros":
            data = np.zeros(in_shape, dtype=np.float32)
        elif data_gen == "ones":
            data = np.ones(in_shape, dtype=np.float32)
        elif data_gen == "negative":
            data = np.array(-np.random.rand(*in_shape) - 1, dtype=np.float32)
        elif data_gen == "large":
            data = np.array(
                np.random.uniform(1e6, 1e8, size=in_shape), dtype=np.float32
            )
        else:
            data = np.array(np.random.randn(*in_shape), dtype=np.float32)

        expected = np.reshape(data, target_shape)
        computed, _ = _run_reshape(data, target_shape, op_name)

        np.testing.assert_allclose(
            computed, expected, rtol=1e-5, atol=1e-7, err_msg=f"[{tmsg}] mismatch"
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Precision tests with known outputs
# --------------------------------------------------------------------------

precision_test_cases = [
    (
        "Flatten sequential",
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [6],
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
    ),
    (
        "Expand sequential",
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        [2, 3],
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    ),
    (
        "Reshape 2x3 to 3x2",
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        [3, 2],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    ),
    (
        "Identity reshape",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [2, 2],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
    (
        "Infer with -1",
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32),
        [2, -1],
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    ),
    (
        "Single element to 3D",
        np.array([42.0], dtype=np.float32),
        [1, 1, 1],
        np.array([[[42.0]]], dtype=np.float32),
    ),
    (
        "3D to 1D",
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        [24],
        np.arange(24, dtype=np.float32),
    ),
    (
        "Add batch dim",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        [1, 2, 2],
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_precision():
    """Precision tests with known expected outputs"""

    msgw = get_max_test_msg_len(precision_test_cases)

    for tno, (tmsg, data, target_shape, expected) in enumerate(precision_test_cases):
        op_name = f"test_reshape_prec_{tno}"

        computed, o_tensors = _run_reshape(data, target_shape, op_name)

        # Shape check
        assert o_tensors[0].shape == list(
            expected.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(expected.shape)}"

        # Value check
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{tmsg}] precision mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_preserves_total_elements():
    """Element count is unchanged after reshape"""

    configs = [
        ([3, 4], [12]),
        ([2, 3, 4], [4, 6]),
        ([2, 3, 4, 5], [6, 20]),
        ([24], [2, 3, 4]),
    ]

    for in_shape, target_shape in configs:
        data = np.array(np.random.randn(*in_shape), dtype=np.float32)
        computed, _ = _run_reshape(data, target_shape, "test_elems")

        assert (
            computed.size == data.size
        ), f"Element count changed: {computed.size} vs {data.size}"

    logger.debug("  Total elements preserved -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_preserves_data_order():
    """Reshape preserves C-order (row-major) data layout"""

    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    for target in [[24], [4, 6], [6, 4], [2, 12], [12, 2], [1, 24]]:
        computed, _ = _run_reshape(data, target, "test_order")

        np.testing.assert_array_equal(
            computed.flatten(),
            data.flatten(),
            err_msg=f"Data order changed for target {target}",
        )

    logger.debug("  Data order preserved (C-order) -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_identity():
    """Reshaping to the same shape returns identical data"""

    for shape in [[4], [3, 4], [2, 3, 4], [1, 2, 3, 4]]:
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        computed, o_tensors = _run_reshape(data, shape, "test_identity")

        np.testing.assert_array_equal(
            computed, data, err_msg=f"Identity reshape failed for shape {shape}"
        )
        assert o_tensors[0].shape == list(data.shape)

    logger.debug("  Identity reshape = no-op -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_roundtrip():
    """Reshape A->B->A recovers original data"""

    configs = [
        ([3, 4], [12]),
        ([2, 3, 4], [6, 4]),
        ([2, 3, 4], [24]),
        ([24], [2, 3, 4]),
    ]

    for shape_a, shape_b in configs:
        data = np.array(np.random.randn(*shape_a), dtype=np.float32)

        # A -> B
        first, _ = _run_reshape(data, shape_b, "test_rt_fwd")
        # B -> A
        second, _ = _run_reshape(first, shape_a, "test_rt_back")

        np.testing.assert_array_equal(
            second,
            data,
            err_msg=f"Roundtrip failed: {shape_a} -> {shape_b} -> {shape_a}",
        )

    logger.debug("  Reshape roundtrip = identity -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_inferred_dim():
    """Reshape with -1 correctly infers the missing dimension"""

    configs = [
        # (in_shape, target_with_minus1, expected_resolved_shape)
        ([12], [-1, 4], [3, 4]),
        ([12], [3, -1], [3, 4]),
        ([24], [2, -1, 4], [2, 3, 4]),
        ([2, 3, 4], [-1], [24]),
        ([2, 3, 4], [6, -1], [6, 4]),
    ]

    for in_shape, target, expected_shape in configs:
        data = np.array(np.random.randn(*in_shape), dtype=np.float32)
        _, o_tensors = _run_reshape(data, target, "test_infer")

        assert (
            o_tensors[0].shape == expected_shape
        ), f"Inferred shape {o_tensors[0].shape} != expected {expected_shape} for target {target}"

    logger.debug("  -1 dim inference correct -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_reshape_constant_input():
    """Reshaping a constant tensor gives a constant tensor with new shape"""

    val = 3.14
    data = np.full([2, 3, 4], val, dtype=np.float32)
    computed, o_tensors = _run_reshape(data, [6, 4], "test_const")

    assert np.all(computed == val), "Constant input should stay constant"
    assert o_tensors[0].shape == [
        6,
        4,
    ], f"Shape should be [6,4], got {o_tensors[0].shape}"

    logger.debug("  Constant input -> constant output -- OK")


@lru_cache(maxsize=1)
def _get_reshape_test_device():
    """Load and cache the device used by reshape memory tests."""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    return Device(packages["n150"])

def calculate_reshape_memory_stats(input_shape, target_shape, dtype="float32"):
    """Calculate memory and compute statistics for a reshape operation"""
    device = _get_reshape_test_device()

    # Create input tensors (X and shape)
    np_dtype = getattr(np, dtype)
    data_X = np.random.randn(*input_shape).astype(np_dtype)
    shape_arr = np.array(target_shape, dtype=np.int64)
    i_tensors = [F._from_data("X", data_X), F._from_data("shape", shape_arr)]

    o_tensors = [make_tensor("Y")]

    # Create operation
    op_info = {
        "optype": "Reshape",
        "name": "reshape_mem_test",
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
        else np.prod(input_shape)
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
    compute_throughput = 1  # 1 op per cycle for reshape
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "input_shape": input_shape,
        "target_shape": target_shape,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_memory": total_memory,
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "device": device,
    }

@RESHAPE_MEMORY_SKIP
def test_reshape_memory_validation():
    """Memory validation test for reshape operation"""
    logger.info("\n" + "=" * 80)
    logger.info("RESHAPE MEMORY VALIDATION TEST")
    logger.info("=" * 80)

    # Test configurations (input_shape, target_shape)
    test_configs = [
        ([32], [8, 4]),
        ([64, 64], [4096]),
        ([32, 32, 32], [1024, 32]),
        ([16, 16, 16, 16], [256, 256]),
        ([1, 224, 224, 3], [1, 150528]),
        ([8, 128, 128, 64], [8, 16384, 64]),
        ([4, 56, 56, 256], [4, 3136, 256]),
    ]

    results = []
    for input_shape, target_shape in test_configs:
        stats = calculate_reshape_memory_stats(input_shape, target_shape)
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
            f"Input Shape: {stats['input_shape']} -> Target Shape: {stats['target_shape']}"
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
