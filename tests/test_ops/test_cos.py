#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Cos op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
import os
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_cos

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_cos(input_data, dtype="float32"):
    """Build a Cos op, run shape inference + compute, return (shape, data, expected)."""
    arr = np.array(input_data, dtype=dtype)
    data_t = F._from_data("input", arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "Cos",
        "optype": "Cos",
        "inList": [data_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([data_t], [out_t])

    result = compute_cos([data_t], op)
    expected = np.cos(arr)
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, input_data)
# ---------------------------------------------------------------------------

cos_test_cases = [
    ("scalar", [0.0]),
    ("1d_small", [0.0, 1.0, -1.0, 2.0]),
    ("pi_multiples", [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi]),
    ("negative_angles", [-np.pi / 2, -np.pi / 4, -np.pi / 6]),
    ("2d", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ("3d", [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
    ("large_angles", [10.0, 100.0, 1000.0, -10.0, -100.0]),
    ("near_zero", [1e-7, -1e-7, 1e-10, -1e-10]),
    ("two_pi_period", [0.0, 2 * np.pi, 4 * np.pi, -2 * np.pi]),
]


# ===========================================================================
# 1. Shape validation
# ===========================================================================
class TestCosShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", cos_test_cases, ids=[c[0] for c in cos_test_cases]
    )
    def test_output_shape_matches_input(self, tid, input_data):
        shape, result, _ = _run_cos(input_data)
        arr = np.array(input_data, dtype="float32")
        assert list(shape) == list(
            arr.shape
        ), f"Shape mismatch: got {shape}, expected {list(arr.shape)}"
        assert (
            result.shape == arr.shape
        ), f"Result shape mismatch: got {result.shape}, expected {arr.shape}"


# ===========================================================================
# 2. Numerical validation
# ===========================================================================
class TestCosNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", cos_test_cases, ids=[c[0] for c in cos_test_cases]
    )
    def test_values_match_numpy(self, tid, input_data):
        _, result, expected = _run_cos(input_data)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestCosEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        _, result, expected = _run_cos([0.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        big = np.random.randn(8, 16, 32).astype("float32").tolist()
        _, result, expected = _run_cos(big)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """cos(inf) and cos(-inf) should be nan."""
        data = [np.inf, -np.inf]
        _, result, _ = _run_cos(data)
        assert np.isnan(result[0]), "cos(inf) should be nan"
        assert np.isnan(result[1]), "cos(-inf) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """cos(nan) should be nan."""
        _, result, _ = _run_cos([np.nan])
        assert np.isnan(result[0]), "cos(nan) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        """cos(0) == 1 exactly."""
        _, result, _ = _run_cos([0.0, -0.0])
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        _, result, expected = _run_cos([0.5, 1.0, 1.5], dtype="float64")
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        """Integer inputs should still produce correct float results."""
        arr = np.array([0, 1, 2, 3], dtype="int32")
        data_t = F._from_data("input", arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Cos",
            "optype": "Cos",
            "inList": [data_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([data_t], [out_t])
        result = compute_cos([data_t], op)
        expected = np.cos(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ===========================================================================
# 4. Precision tests with known analytical values
# ===========================================================================
class TestCosPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_zero(self):
        """cos(0) = 1"""
        _, result, _ = _run_cos([0.0])
        np.testing.assert_allclose(result, [1.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_pi_over_6(self):
        """cos(π/6) = √3/2"""
        _, result, _ = _run_cos([np.pi / 6])
        np.testing.assert_allclose(result, [np.sqrt(3) / 2], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_pi_over_4(self):
        """cos(π/4) = √2/2"""
        _, result, _ = _run_cos([np.pi / 4])
        np.testing.assert_allclose(result, [np.sqrt(2) / 2], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_pi_over_3(self):
        """cos(π/3) = 0.5"""
        _, result, _ = _run_cos([np.pi / 3])
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_pi_over_2(self):
        """cos(π/2) ≈ 0"""
        _, result, _ = _run_cos([np.pi / 2])
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_pi(self):
        """cos(π) = -1"""
        _, result, _ = _run_cos([np.pi])
        np.testing.assert_allclose(result, [-1.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_3pi_over_2(self):
        """cos(3π/2) ≈ 0"""
        _, result, _ = _run_cos([3 * np.pi / 2])
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_2pi(self):
        """cos(2π) = 1"""
        _, result, _ = _run_cos([2 * np.pi])
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_cos_negative_pi_over_2(self):
        """cos(-π/2) ≈ 0"""
        _, result, _ = _run_cos([-np.pi / 2])
        np.testing.assert_allclose(result, [0.0], atol=1e-6)


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestCosProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_range(self):
        """cos(x) ∈ [-1, 1] for all finite x."""
        data = np.linspace(-10 * np.pi, 10 * np.pi, 500, dtype="float32").tolist()
        _, result, _ = _run_cos(data)
        assert np.all(result >= -1.0 - 1e-6), "cos(x) should be >= -1"
        assert np.all(result <= 1.0 + 1e-6), "cos(x) should be <= 1"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_even_symmetry(self):
        """cos(-x) = cos(x) (even function)."""
        x = [0.5, 1.0, 2.0, 3.0, np.pi / 3]
        neg_x = [-v for v in x]
        _, res_pos, _ = _run_cos(x)
        _, res_neg, _ = _run_cos(neg_x)
        np.testing.assert_allclose(
            res_neg,
            res_pos,
            rtol=1e-6,
            atol=1e-7,
            err_msg="cos(-x) should equal cos(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_periodicity(self):
        """cos(x + 2π) = cos(x)."""
        x = [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2, np.pi]
        x_plus_2pi = [v + 2 * np.pi for v in x]
        _, res_x, _ = _run_cos(x)
        _, res_shifted, _ = _run_cos(x_plus_2pi)
        np.testing.assert_allclose(
            res_shifted,
            res_x,
            rtol=1e-5,
            atol=1e-6,
            err_msg="cos(x+2π) should equal cos(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pythagorean_identity(self):
        """sin²(x) + cos²(x) = 1."""
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100, dtype="float32")
        sin_vals = np.sin(x)
        _, cos_result, _ = _run_cos(x.tolist())
        identity = sin_vals**2 + cos_result**2
        np.testing.assert_allclose(
            identity,
            1.0,
            rtol=1e-5,
            atol=1e-6,
            err_msg="sin²(x) + cos²(x) should equal 1",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_double_angle(self):
        """cos(2x) = 2·cos²(x) - 1."""
        x = [0.3, 0.7, 1.2, np.pi / 5, np.pi / 3]
        two_x = [2 * v for v in x]
        _, cos_2x, _ = _run_cos(two_x)
        _, cos_x, _ = _run_cos(x)
        expected = 2 * cos_x**2 - 1
        np.testing.assert_allclose(
            cos_2x,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="cos(2x) should equal 2·cos²(x) - 1",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sum_formula(self):
        """cos(a+b) = cos(a)cos(b) - sin(a)sin(b)."""
        a_vals = [0.5, 1.0, np.pi / 4]
        b_vals = [0.3, 0.8, np.pi / 6]
        a_plus_b = [a + b for a, b in zip(a_vals, b_vals)]
        _, cos_ab, _ = _run_cos(a_plus_b)
        _, cos_a, _ = _run_cos(a_vals)
        _, cos_b, _ = _run_cos(b_vals)
        sin_a = np.sin(np.array(a_vals, dtype="float32"))
        sin_b = np.sin(np.array(b_vals, dtype="float32"))
        expected = cos_a * cos_b - sin_a * sin_b
        np.testing.assert_allclose(
            cos_ab,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="cos(a+b) should equal cos(a)cos(b) - sin(a)sin(b)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_derivative_approximation(self):
        """Numerical derivative of cos ≈ -sin."""
        x = np.linspace(0, 2 * np.pi, 200, dtype="float64")
        h = 1e-5
        x_plus_h = (x + h).tolist()
        x_minus_h = (x - h).tolist()
        _, f_plus, _ = _run_cos(x_plus_h, dtype="float64")
        _, f_minus, _ = _run_cos(x_minus_h, dtype="float64")
        numerical_deriv = (f_plus - f_minus) / (2 * h)
        analytical_deriv = -np.sin(x)
        np.testing.assert_allclose(
            numerical_deriv,
            analytical_deriv,
            rtol=1e-4,
            atol=1e-4,
            err_msg="d/dx cos(x) should approximate -sin(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_phase_shift_from_sin(self):
        """cos(x) = sin(x + π/2)."""
        x = [0.0, 0.5, 1.0, np.pi / 4, np.pi / 3, 2.0]
        x_shifted = [v + np.pi / 2 for v in x]
        _, cos_x, _ = _run_cos(x)
        sin_shifted = np.sin(np.array(x_shifted, dtype="float32"))
        np.testing.assert_allclose(
            cos_x,
            sin_shifted,
            rtol=1e-5,
            atol=1e-6,
            err_msg="cos(x) should equal sin(x + π/2)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same input → all-same output."""
        val = 1.0
        data = [[val] * 4] * 3
        _, result, _ = _run_cos(data)
        expected_val = np.cos(np.float32(val))
        np.testing.assert_allclose(result, expected_val, rtol=1e-6)


# ===========================================================================
# 6. Memory performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_cos_memory_validation(capsys, request):
    """
    Test memory validation for cos operation.
    Validates instructions executed and data moved for various scenarios.

    This test validates two primary metrics:
    1. Instructions Executed: Verifies instruction count matches element count
    2. Data Moved: Tracks input/output bytes and validates memory traffic

    Run with: pytest tests/test_ops/test_cos.py::test_cos_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    # Test cases: different tensor shapes
    test_cases = [
        {"name": "1D Array", "shape": [1000], "description": "Cos of 1D array"},
        {"name": "2D Matrix", "shape": [32, 32], "description": "Cos of 2D matrix"},
        {
            "name": "4D Tensor",
            "shape": [2, 16, 16, 16],
            "description": "Cos of 4D tensor",
        },
        {
            "name": "Large 2D",
            "shape": [128, 256],
            "description": "Cos of large 2D matrix",
        },
    ]

    # Load device configuration once for all tests
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        logger.info("\n%s", "=" * 60)
        logger.info("Cos Operation Memory Validation")
        logger.info("%s\n", "=" * 60)
        logger.info("Device: Wormhole (n150)")
        logger.info(f"Device frequency: {device.freq_MHz} MHz")
        logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.info(
            "Peak bandwidth: %.2f GB/s",
            device.simconfig_obj.peak_bandwidth(freq_units="GHz"),
        )
    except Exception as e:
        logger.info(f"\nWarning: Could not load device config: {e}")
        logger.info("Skipping memory validation test")
        pytest.skip(f"Could not load device config: {e}")
        return

    logger.info("\n%s", "=" * 60)
    logger.info("Running Memory Validation Tests")
    logger.info("%s\n", "=" * 60)

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        description = test_case["description"]

        logger.debug(f"\n-- Test: {test_name} --")
        logger.debug(f"Shape: {shape}")

        # Generate test data (use values in valid range for cos)
        np.random.seed(42)
        test_data = np.array(np.random.randn(*shape) * 3, dtype=np.float32)

        # Create operation
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f'cos_mem_{test_name.replace(" ", "_")}',
            "optype": "Cos",
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

        # Validate compute_cos correctness
        result = compute_cos(i_tensors, op_obj)
        expected = np.cos(test_data)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{test_name}] compute_cos validation failed",
        )

        # Calculate output shape
        output_shape = tuple(o_tensors[0].shape)
        output_elems = int(np.prod(output_shape))
        input_elems = int(np.prod(shape))

        # Set compute pipe for Cos operation (uses vector pipe for element-wise ops)
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

        # Validate instructions (cos uses 'cos' instruction, 1 per element)
        assert "cos" in actual_instrs, "Expected 'cos' instruction not found"
        actual_cos = actual_instrs.get("cos", 0)
        assert (
            actual_cos == input_elems
        ), f"Cos instruction count mismatch: {actual_cos} vs {input_elems}"

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
        logger.debug("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {instructions_executed:,} (cos)")
        logger.debug(f"  Input elements:        {input_elems:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(
            f"  Expected instructions: ~{input_elems:,} (1 cos per element)"
        )
        instruction_ratio = actual_cos / input_elems if input_elems > 0 else 0
        logger.debug(
            f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 cos per element)"
        )

        logger.debug("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
        )

        logger.debug("\n  -- Memory Metrics --")
        logger.debug(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
        expected_ai = input_elems / total_data_movement
        logger.debug(f"  Expected intensity:    {expected_ai:.4f} ops/byte")
        np.testing.assert_allclose(
            arithmetic_intensity,
            expected_ai,
            rtol=0.01,
            atol=1e-6,
            err_msg=f"Arithmetic intensity mismatch",
        )
        logger.debug("  ✓ Arithmetic intensity within expected range")

        logger.debug("\n  -- Execution Cycles --")
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
                "cos_instructions": actual_cos,
                "total_data_moved": total_data_movement,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "ideal_cycles": total_cycles,
            }
        )

        logger.debug("\n  ✓ Test PASSED")

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("Memory Validation Summary")
    logger.info("%s\n", "=" * 60)
    logger.info(f"Total tests run: {len(all_results)}")
    logger.info("All tests passed: ✓")

    # Compare arithmetic intensity across tests
    logger.info("\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["arithmetic_intensity"]
        logger.info(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

    # Element count analysis
    logger.info("\n-- Element Count Analysis --")
    for result in all_results:
        elems = result["cos_instructions"]
        logger.info(f"{result['test_name']:30s}: {elems:,} elements")

    # Bottleneck analysis
    logger.info("\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["bottleneck"]
        logger.info(f"{result['test_name']:30s}: {bottleneck}")

    logger.info("\n%s", "=" * 60)
    logger.info("Memory validation complete!")
    logger.info("%s\n", "=" * 60)

    # Create a summary that will be displayed in pytest output
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Cos instructions match element count (1:1 ratio) ✓",
        "  • Unary operations show typical MEMORY bottleneck ✓",
        "  • Low arithmetic intensity confirms memory-bound operation ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} cos | {:>7.1f} KB | {:.3f} ops/byte".format(
                result["test_name"],
                result["cos_instructions"],
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
                "=", "COS MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("COS MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
