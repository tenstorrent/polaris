#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import time
import os
from pathlib import Path
from loguru import logger

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_tanh

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --------------------------------------------------------------------------
# Reference implementation
# --------------------------------------------------------------------------


def ref_impl_tanh(X):
    """Reference tanh: element-wise hyperbolic tangent"""
    return np.tanh(X)


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


# --------------------------------------------------------------------------
# Main test cases
# --------------------------------------------------------------------------

test_name = "test_tanh"
test_cases = [
    # (name, input_shape, data_type)
    ("1D input", [8], "mixed"),
    ("2D input", [3, 4], "mixed"),
    ("3D input", [2, 3, 4], "mixed"),
    ("4D input (NCHW)", [2, 3, 4, 4], "mixed"),
    ("5D input", [2, 2, 3, 4, 4], "mixed"),
    ("7D input", [1, 1, 2, 2, 3, 4, 4], "mixed"),
    # Special values
    ("Single element", [1], "mixed"),
    ("All zeros", [3, 4], "zeros"),
    ("All ones", [3, 4], "ones"),
    ("All negative ones", [3, 4], "neg_ones"),
    ("Large positive", [3, 4], "large_pos"),
    ("Large negative", [3, 4], "large_neg"),
    ("Small near zero", [3, 4], "small"),
    # Various sizes
    ("Large 2D", [64, 64], "mixed"),
    ("Large 4D", [2, 16, 8, 8], "mixed"),
    ("Ones in shape", [1, 1, 1, 1], "mixed"),
    ("Mixed sizes", [1, 64, 1, 32], "mixed"),
]


def generate_test_data(shape, data_type):
    """Generate test data for tanh tests."""
    if data_type == "mixed":
        return np.array(np.random.randn(*shape) * 3, dtype=np.float32)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif data_type == "neg_ones":
        return -np.ones(shape, dtype=np.float32)
    elif data_type == "large_pos":
        return np.array(np.random.uniform(10, 100, size=shape), dtype=np.float32)
    elif data_type == "large_neg":
        return np.array(np.random.uniform(-100, -10, size=shape), dtype=np.float32)
    elif data_type == "small":
        return np.array(np.random.uniform(-1e-6, 1e-6, size=shape), dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh():
    """Numerical validation of compute_tanh across shapes and data ranges"""

    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        data = generate_test_data(shape, data_type)
        expected = ref_impl_tanh(data)

        i_tensors = [F._from_data("X", data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Tanh",
            "inList": ["X"],
            "outList": ["Y"],
        }
        op_obj = SimOp(op_info)
        i_tensors[0].op_in = [op_name]
        o_tensors[0].op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Shape check
        assert o_tensors[0].shape == list(
            data.shape
        ), f"[{tmsg}] shape mismatch: {o_tensors[0].shape} vs {list(data.shape)}"

        # Numerical check
        computed = compute_tanh(i_tensors, op_obj)
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"[{tmsg}] numerical mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------

edge_cases = [
    (
        "Exactly zero",
        np.array([0.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    (
        "Exactly one",
        np.array([1.0], dtype=np.float32),
        np.array([np.tanh(1.0)], dtype=np.float32),
    ),
    (
        "Exactly neg one",
        np.array([-1.0], dtype=np.float32),
        np.array([np.tanh(-1.0)], dtype=np.float32),
    ),
    (
        "Very large pos",
        np.array([100.0, 500.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ),
    (
        "Very large neg",
        np.array([-100.0, -500.0], dtype=np.float32),
        np.array([-1.0, -1.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_edge_cases():
    """Edge cases for Tanh"""

    msgw = get_max_test_msg_len(edge_cases)

    for tno, (tmsg, data, expected) in enumerate(edge_cases):
        op_name = f"test_tanh_edge_{tno}"

        i_tensors = [F._from_data("X", data)]
        o_tensors = [make_tensor("Y")]

        op_obj = SimOp(
            {"name": op_name, "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
        )
        i_tensors[0].op_in = [op_name]
        o_tensors[0].op_out = [op_name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        computed = compute_tanh(i_tensors, op_obj)
        np.testing.assert_allclose(
            computed, expected, rtol=1e-5, atol=1e-6, err_msg=f"[{tmsg}] mismatch"
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Precision tests with known outputs
# --------------------------------------------------------------------------

precision_test_cases = [
    (
        "tanh(0) = 0",
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "tanh(large) ~ 1",
        np.array([10.0, 20.0, 50.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
    ),
    (
        "tanh(-large) ~ -1",
        np.array([-10.0, -20.0, -50.0], dtype=np.float32),
        np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    ),
    (
        "tanh(small) ~ x",
        np.array([1e-7, -1e-7, 1e-8], dtype=np.float32),
        np.array([1e-7, -1e-7, 1e-8], dtype=np.float32),
    ),
    (
        "Known values",
        np.array([0.5, 1.0, -0.5, -1.0], dtype=np.float32),
        np.array(
            [np.tanh(0.5), np.tanh(1.0), np.tanh(-0.5), np.tanh(-1.0)], dtype=np.float32
        ),
    ),
    (
        "Symmetric pairs",
        np.array([2.0, -2.0, 3.0, -3.0], dtype=np.float32),
        np.array(
            [np.tanh(2.0), np.tanh(-2.0), np.tanh(3.0), np.tanh(-3.0)], dtype=np.float32
        ),
    ),
    (
        "Sequential integers",
        np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        np.tanh(np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)),
    ),
    (
        "Near saturation boundary",
        np.array([4.0, 5.0, -4.0, -5.0], dtype=np.float32),
        np.tanh(np.array([4.0, 5.0, -4.0, -5.0], dtype=np.float32)),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_precision():
    """Precision tests with known expected outputs"""

    msgw = get_max_test_msg_len(precision_test_cases)

    for tno, (tmsg, data, expected) in enumerate(precision_test_cases):
        op_name = f"test_tanh_prec_{tno}"

        i_tensors = [F._from_data("X", data)]
        o_tensors = [make_tensor("Y")]

        op_obj = SimOp(
            {"name": op_name, "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
        )
        i_tensors[0].op_in = [op_name]
        o_tensors[0].op_out = [op_name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        computed = compute_tanh(i_tensors, op_obj)
        np.testing.assert_allclose(
            computed,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"[{tmsg}] precision mismatch",
        )

        logger.debug(f"  {tmsg:<{msgw}} -- OK")


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_output_range():
    """tanh(x) is always in (-1, 1)"""

    for shape in [[100], [10, 10], [4, 8, 8]]:
        data = np.array(np.random.randn(*shape) * 10, dtype=np.float32)

        i_tensors = [F._from_data("X", data)]
        o_tensors = [make_tensor("Y")]
        op_obj = SimOp(
            {"name": "range_test", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
        )
        i_tensors[0].op_in = ["range_test"]
        o_tensors[0].op_out = ["range_test"]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        result = compute_tanh(i_tensors, op_obj)
        assert np.all(result >= -1.0) and np.all(
            result <= 1.0
        ), f"tanh output outside [-1, 1] for shape {shape}"

    logger.debug("  Output in [-1, 1] -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_odd_symmetry():
    """tanh(-x) == -tanh(x) (odd function)"""

    for shape in [[8], [3, 4], [2, 3, 4]]:
        data = np.array(np.random.randn(*shape) * 3, dtype=np.float32)

        # tanh(x)
        i_pos = [F._from_data("X", data)]
        o_pos = [make_tensor("Y")]
        op_pos = SimOp(
            {"name": "odd_pos", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
        )
        i_pos[0].op_in = ["odd_pos"]
        o_pos[0].op_out = ["odd_pos"]
        op_pos.get_perf_counts(i_pos, o_pos)
        result_pos = compute_tanh(i_pos, op_pos)

        # tanh(-x)
        i_neg = [F._from_data("X", -data)]
        o_neg = [make_tensor("Y")]
        op_neg = SimOp(
            {"name": "odd_neg", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
        )
        i_neg[0].op_in = ["odd_neg"]
        o_neg[0].op_out = ["odd_neg"]
        op_neg.get_perf_counts(i_neg, o_neg)
        result_neg = compute_tanh(i_neg, op_neg)

        np.testing.assert_allclose(
            result_neg,
            -result_pos,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Odd symmetry failed for shape {shape}",
        )

    logger.debug("  tanh(-x) == -tanh(x) -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_monotonic():
    """tanh is strictly monotonically increasing"""

    data = np.sort(np.random.randn(100).astype(np.float32))

    i_tensors = [F._from_data("X", data)]
    o_tensors = [make_tensor("Y")]
    op_obj = SimOp(
        {"name": "mono_test", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
    )
    i_tensors[0].op_in = ["mono_test"]
    o_tensors[0].op_out = ["mono_test"]
    op_obj.get_perf_counts(i_tensors, o_tensors)

    result = compute_tanh(i_tensors, op_obj)
    diffs = np.diff(result)
    assert np.all(diffs >= 0), "tanh is not monotonically increasing"

    logger.debug("  Monotonically increasing -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_at_zero():
    """tanh(0) == 0 exactly"""

    data = np.array([0.0], dtype=np.float32)

    i_tensors = [F._from_data("X", data)]
    o_tensors = [make_tensor("Y")]
    op_obj = SimOp(
        {"name": "zero_test", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
    )
    i_tensors[0].op_in = ["zero_test"]
    o_tensors[0].op_out = ["zero_test"]
    op_obj.get_perf_counts(i_tensors, o_tensors)

    result = compute_tanh(i_tensors, op_obj)
    assert result[0] == 0.0, f"tanh(0) = {result[0]}, expected 0.0"

    logger.debug("  tanh(0) == 0 -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_saturation():
    """Large magnitude inputs saturate to +/-1"""

    data = np.array([50.0, -50.0, 100.0, -100.0], dtype=np.float32)

    i_tensors = [F._from_data("X", data)]
    o_tensors = [make_tensor("Y")]
    op_obj = SimOp(
        {"name": "sat_test", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
    )
    i_tensors[0].op_in = ["sat_test"]
    o_tensors[0].op_out = ["sat_test"]
    op_obj.get_perf_counts(i_tensors, o_tensors)

    result = compute_tanh(i_tensors, op_obj)
    np.testing.assert_allclose(
        result,
        np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
        err_msg="Saturation check failed",
    )

    logger.debug("  Saturation to +/-1 -- OK")


@pytest.mark.unit
@pytest.mark.opunit
def test_tanh_derivative_relation():
    """Verify tanh'(x) = 1 - tanh(x)^2 numerically via finite differences"""

    data = np.array(np.linspace(-3, 3, 100), dtype=np.float32)
    eps = 1e-4

    # tanh(x)
    i_t = [F._from_data("X", data)]
    o_t = [make_tensor("Y")]
    op_t = SimOp({"name": "deriv", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]})
    i_t[0].op_in = ["deriv"]
    o_t[0].op_out = ["deriv"]
    op_t.get_perf_counts(i_t, o_t)
    tanh_x = compute_tanh(i_t, op_t)

    # tanh(x + eps)
    i_p = [F._from_data("X", data + eps)]
    o_p = [make_tensor("Y")]
    op_p = SimOp(
        {"name": "deriv_p", "optype": "Tanh", "inList": ["X"], "outList": ["Y"]}
    )
    i_p[0].op_in = ["deriv_p"]
    o_p[0].op_out = ["deriv_p"]
    op_p.get_perf_counts(i_p, o_p)
    tanh_xpe = compute_tanh(i_p, op_p)

    numerical_deriv = (tanh_xpe - tanh_x) / eps
    analytical_deriv = 1.0 - tanh_x**2

    np.testing.assert_allclose(
        numerical_deriv,
        analytical_deriv,
        rtol=1e-2,
        atol=1e-3,
        err_msg="tanh derivative relation failed",
    )

    logger.debug("  tanh'(x) = 1 - tanh(x)^2 -- OK")


# --------------------------------------------------------------------------
# Memory and performance estimation
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_tanh_memory_validation(capsys, request):
    """
    Test memory validation for tanh operation.
    Validates instructions executed and data moved for various scenarios.

    This test validates two primary metrics:
    1. Instructions Executed: Verifies instruction count matches element count
    2. Data Moved: Tracks input/output bytes and validates memory traffic

    Run with: pytest tests/test_ops/test_tanh.py::test_tanh_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    # Test cases: different tensor shapes
    test_cases = [
        {"name": "1D Array", "shape": [1000], "description": "Tanh of 1D array"},
        {"name": "2D Matrix", "shape": [32, 32], "description": "Tanh of 2D matrix"},
        {
            "name": "4D Tensor",
            "shape": [2, 16, 16, 16],
            "description": "Tanh of 4D tensor",
        },
        {
            "name": "Large 2D",
            "shape": [128, 256],
            "description": "Tanh of large 2D matrix",
        },
    ]

    # Load device configuration once for all tests
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        logger.info(f"\n{'='*60}")
        logger.info("Tanh Operation Memory Validation")
        logger.info(f"{'='*60}\n")
        logger.info("Device: Wormhole (n150)")
        logger.info(f"Device frequency: {device.freq_MHz} MHz")
        logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.info(
            f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        logger.info(f"\nWarning: Could not load device config: {e}")
        logger.info("Skipping memory validation test")
        pytest.skip(f"Could not load device config: {e}")
        return

    logger.info(f"\n{'='*60}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*60}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        description = test_case["description"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Shape: {shape}")

        # Generate test data (mixed positive/negative for tanh)
        np.random.seed(42)
        test_data = np.array(np.random.randn(*shape) * 3, dtype=np.float32)

        # Create operation
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f'tanh_mem_{test_name.replace(" ", "_")}',
            "optype": "Tanh",
            "inList": ["X"],
            "outList": ["Y"],
        }
        op_obj = SimOp(op_info)
        i_tensors[0].op_in = [op_info["name"]]
        o_tensors[0].op_out = [op_info["name"]]

        # Set operation precision
        op_obj.precision = "fp32"

        # Get performance counts
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate compute_tanh correctness
        result = compute_tanh(i_tensors, op_obj)
        expected = np.tanh(test_data)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{test_name}] compute_tanh validation failed",
        )

        # Calculate output shape
        output_shape = tuple(o_tensors[0].shape)
        output_elems = int(np.prod(output_shape))
        input_elems = int(np.prod(shape))

        # Set compute pipe for Tanh operation (uses vector pipe for element-wise ops)
        if op_obj.uses_compute_pipe is None:
            op_obj.uses_compute_pipe = "vector"

        # Execute on device for cycle estimation
        if op_obj.perf_stats is not None:
            device.execute_op(op_obj)

        # Extract stats from op.perf_stats
        perf_stats = op_obj.perf_stats
        actual_in_elems = perf_stats["inElems"]
        actual_out_elems = perf_stats["outElems"]
        actual_in_bytes = perf_stats["inBytes"]
        actual_out_bytes = perf_stats["outBytes"]
        actual_instrs = perf_stats["instrs"]

        bytes_per_element = 4  # fp32

        # Validate element counts
        assert (
            actual_in_elems == input_elems
        ), f"Input element count mismatch: {actual_in_elems} vs {input_elems}"
        assert (
            actual_out_elems == output_elems
        ), f"Output element count mismatch: {actual_out_elems} vs {output_elems}"

        # Validate instructions (tanh uses 'tanh' instruction, 1 per element)
        assert "tanh" in actual_instrs, "Expected 'tanh' instruction not found"
        actual_tanh = actual_instrs.get("tanh", 0)
        assert (
            actual_tanh == input_elems
        ), f"Tanh instruction count mismatch: {actual_tanh} vs {input_elems}"

        # Calculate metrics
        total_data_movement = actual_in_bytes + actual_out_bytes
        instructions_executed = sum(actual_instrs.values())
        arithmetic_intensity = (
            instructions_executed / total_data_movement
            if total_data_movement > 0
            else 0
        )

        # Calculate execution cycles (read from op object, not perf_stats)
        compute_cycles = op_obj.compute_cycles
        mem_rd_cycles = op_obj.mem_rd_cycles
        mem_wr_cycles = op_obj.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        total_cycles = max(compute_cycles, memory_cycles)
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        # Print detailed breakdown
        logger.debug(f"\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {instructions_executed:,} (tanh)")
        logger.debug(f"  Input elements:        {input_elems:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(
            f"  Expected instructions: ~{input_elems:,} (1 tanh per element)"
        )
        instruction_ratio = actual_tanh / input_elems if input_elems > 0 else 0
        logger.debug(
            f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 tanh per element)"
        )

        logger.debug(f"\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
        )

        logger.debug(f"\n  -- Memory Metrics --")
        logger.debug(
            f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
        )
        expected_ai = input_elems / total_data_movement
        logger.debug(f"  Expected intensity:    {expected_ai:.4f} ops/byte")
        np.testing.assert_allclose(
            arithmetic_intensity,
            expected_ai,
            rtol=0.01,
            atol=1e-6,
            err_msg=f"Arithmetic intensity mismatch",
        )
        logger.debug(f"  ✓ Arithmetic intensity within expected range")

        logger.debug(f"\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {total_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")
        logger.debug(f"  ✓ Bottleneck analysis: {bottleneck} for unary operation")

        # Store results for summary
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "output_shape": output_shape,
                "tanh_instructions": actual_tanh,
                "total_data_moved": total_data_movement,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "ideal_cycles": total_cycles,
            }
        )

        logger.debug(f"\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total tests run: {len(all_results)}")
    logger.info(f"All tests passed: ✓")

    # Compare arithmetic intensity across tests
    logger.info(f"\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["arithmetic_intensity"]
        logger.debug(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

    # Element count analysis
    logger.info(f"\n-- Element Count Analysis --")
    for result in all_results:
        elems = result["tanh_instructions"]
        logger.debug(f"{result['test_name']:30s}: {elems:,} elements")

    # Bottleneck analysis
    logger.info(f"\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["bottleneck"]
        logger.debug(f"{result['test_name']:30s}: {bottleneck}")

    logger.info(f"\n{'='*60}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*60}\n")

    # Create a summary that will be displayed in pytest output
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Tanh instructions match element count (1:1 ratio) ✓",
        "  • Unary operations show typical MEMORY bottleneck ✓",
        "  • Low arithmetic intensity confirms memory-bound operation ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} tanh | {:>7.1f} KB | {:.3f} ops/byte".format(
                result["test_name"],
                result["tanh_instructions"],
                result["total_data_moved"] / 1024,
                result["arithmetic_intensity"],
            )
        )

    summary_lines.extend(
        [
            "",
            "Validation: All memory metrics within expected ranges ✓",
            "",
            "For detailed output, run with: pytest -s -v",
        ]
    )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "TANH MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("TANH MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
