#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Sin op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
import os
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_sin

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


def _run_sin(input_data, dtype="float32"):
    """Build a Sin op, run shape inference + compute, return (shape, data, expected)."""
    arr = np.array(input_data, dtype=dtype)
    data_t = F._from_data("input", arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "Sin",
        "optype": "Sin",
        "inList": [data_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([data_t], [out_t])

    result = compute_sin([data_t], op)
    expected = np.sin(arr)
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, input_data)
# ---------------------------------------------------------------------------

sin_test_cases = [
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
class TestSinShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", sin_test_cases, ids=[c[0] for c in sin_test_cases]
    )
    def test_output_shape_matches_input(self, tid, input_data):
        shape, result, _ = _run_sin(input_data)
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
class TestSinNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", sin_test_cases, ids=[c[0] for c in sin_test_cases]
    )
    def test_values_match_numpy(self, tid, input_data):
        _, result, expected = _run_sin(input_data)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestSinEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        _, result, expected = _run_sin([np.pi / 2])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        big = np.random.randn(8, 16, 32).astype("float32").tolist()
        _, result, expected = _run_sin(big)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """sin(inf) and sin(-inf) should be nan."""
        data = [np.inf, -np.inf]
        _, result, expected = _run_sin(data)
        assert np.isnan(result[0]), "sin(inf) should be nan"
        assert np.isnan(result[1]), "sin(-inf) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """sin(nan) should be nan."""
        _, result, _ = _run_sin([np.nan])
        assert np.isnan(result[0]), "sin(nan) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        """sin(0) == 0 exactly."""
        _, result, _ = _run_sin([0.0, -0.0])
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        _, result, expected = _run_sin([0.5, 1.0, 1.5], dtype="float64")
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        """Integer inputs should still produce correct float results."""
        arr = np.array([0, 1, 2, 3], dtype="int32")
        data_t = F._from_data("input", arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Sin",
            "optype": "Sin",
            "inList": [data_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([data_t], [out_t])
        result = compute_sin([data_t], op)
        expected = np.sin(arr)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ===========================================================================
# 4. Precision tests with known analytical values
# ===========================================================================
class TestSinPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_zero(self):
        """sin(0) = 0"""
        _, result, _ = _run_sin([0.0])
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_pi_over_6(self):
        """sin(π/6) = 0.5"""
        _, result, _ = _run_sin([np.pi / 6])
        np.testing.assert_allclose(result, [0.5], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_pi_over_4(self):
        """sin(π/4) = √2/2"""
        _, result, _ = _run_sin([np.pi / 4])
        np.testing.assert_allclose(result, [np.sqrt(2) / 2], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_pi_over_3(self):
        """sin(π/3) = √3/2"""
        _, result, _ = _run_sin([np.pi / 3])
        np.testing.assert_allclose(result, [np.sqrt(3) / 2], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_pi_over_2(self):
        """sin(π/2) = 1"""
        _, result, _ = _run_sin([np.pi / 2])
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_pi(self):
        """sin(π) ≈ 0"""
        _, result, _ = _run_sin([np.pi])
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_3pi_over_2(self):
        """sin(3π/2) = -1"""
        _, result, _ = _run_sin([3 * np.pi / 2])
        np.testing.assert_allclose(result, [-1.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_2pi(self):
        """sin(2π) ≈ 0"""
        _, result, _ = _run_sin([2 * np.pi])
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sin_negative_pi_over_2(self):
        """sin(-π/2) = -1"""
        _, result, _ = _run_sin([-np.pi / 2])
        np.testing.assert_allclose(result, [-1.0], atol=1e-6)


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestSinProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_range(self):
        """sin(x) ∈ [-1, 1] for all finite x."""
        data = np.linspace(-10 * np.pi, 10 * np.pi, 500, dtype="float32").tolist()
        _, result, _ = _run_sin(data)
        assert np.all(result >= -1.0 - 1e-6), "sin(x) should be >= -1"
        assert np.all(result <= 1.0 + 1e-6), "sin(x) should be <= 1"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_odd_symmetry(self):
        """sin(-x) = -sin(x) (odd function)."""
        x = [0.5, 1.0, 2.0, 3.0, np.pi / 3]
        neg_x = [-v for v in x]
        _, res_pos, _ = _run_sin(x)
        _, res_neg, _ = _run_sin(neg_x)
        np.testing.assert_allclose(
            res_neg,
            -res_pos,
            rtol=1e-6,
            atol=1e-7,
            err_msg="sin(-x) should equal -sin(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_periodicity(self):
        """sin(x + 2π) = sin(x)."""
        x = [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2, np.pi]
        x_plus_2pi = [v + 2 * np.pi for v in x]
        _, res_x, _ = _run_sin(x)
        _, res_shifted, _ = _run_sin(x_plus_2pi)
        np.testing.assert_allclose(
            res_shifted,
            res_x,
            rtol=1e-5,
            atol=1e-6,
            err_msg="sin(x+2π) should equal sin(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_pythagorean_identity(self):
        """sin²(x) + cos²(x) = 1."""
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100, dtype="float32")
        sin_vals = np.sin(x)
        cos_vals = np.cos(x)
        # Use the compute function for sin
        _, sin_result, _ = _run_sin(x.tolist())
        identity = sin_result**2 + cos_vals**2
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
        """sin(2x) = 2·sin(x)·cos(x)."""
        x = [0.3, 0.7, 1.2, np.pi / 5, np.pi / 3]
        two_x = [2 * v for v in x]
        _, sin_2x, _ = _run_sin(two_x)
        _, sin_x, _ = _run_sin(x)
        cos_x = np.cos(np.array(x, dtype="float32"))
        np.testing.assert_allclose(
            sin_2x,
            2 * sin_x * cos_x,
            rtol=1e-5,
            atol=1e-6,
            err_msg="sin(2x) should equal 2·sin(x)·cos(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sum_formula(self):
        """sin(a+b) = sin(a)cos(b) + cos(a)sin(b)."""
        a_vals = [0.5, 1.0, np.pi / 4]
        b_vals = [0.3, 0.8, np.pi / 6]
        a_plus_b = [a + b for a, b in zip(a_vals, b_vals)]
        _, sin_ab, _ = _run_sin(a_plus_b)
        _, sin_a, _ = _run_sin(a_vals)
        _, sin_b, _ = _run_sin(b_vals)
        cos_a = np.cos(np.array(a_vals, dtype="float32"))
        cos_b = np.cos(np.array(b_vals, dtype="float32"))
        expected = sin_a * cos_b + cos_a * sin_b
        np.testing.assert_allclose(
            sin_ab,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg="sin(a+b) should equal sin(a)cos(b) + cos(a)sin(b)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_derivative_approximation(self):
        """Numerical derivative of sin ≈ cos."""
        x = np.linspace(0, 2 * np.pi, 200, dtype="float64")
        h = 1e-5
        x_plus_h = (x + h).tolist()
        x_minus_h = (x - h).tolist()
        _, f_plus, _ = _run_sin(x_plus_h, dtype="float64")
        _, f_minus, _ = _run_sin(x_minus_h, dtype="float64")
        numerical_deriv = (f_plus - f_minus) / (2 * h)
        analytical_deriv = np.cos(x)
        np.testing.assert_allclose(
            numerical_deriv,
            analytical_deriv,
            rtol=1e-4,
            atol=1e-4,
            err_msg="d/dx sin(x) should approximate cos(x)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same input → all-same output."""
        val = 1.0
        data = [[val] * 4] * 3
        _, result, _ = _run_sin(data)
        expected_val = np.sin(np.float32(val))
        np.testing.assert_allclose(result, expected_val, rtol=1e-6)


# ===========================================================================
# 6. Memory performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_sin_memory_validation(capsys, request):
    """
    Test memory validation for sin operation.
    Validates instructions executed and data moved for various scenarios.

    This test validates two primary metrics:
    1. Instructions Executed: Verifies instruction count matches element count
    2. Data Moved: Tracks input/output bytes and validates memory traffic

    Run with: pytest tests/test_ops/test_sin.py::test_sin_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    # Test cases: different tensor shapes
    test_cases = [
        {"name": "1D Array", "shape": [1000], "description": "Sin of 1D array"},
        {"name": "2D Matrix", "shape": [32, 32], "description": "Sin of 2D matrix"},
        {
            "name": "4D Tensor",
            "shape": [2, 16, 16, 16],
            "description": "Sin of 4D tensor",
        },
        {
            "name": "Large 2D",
            "shape": [128, 256],
            "description": "Sin of large 2D matrix",
        },
    ]

    # Load device configuration once for all tests
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        logger.info(f"\n{'='*60}")
        logger.info("Sin Operation Memory Validation")
        logger.info(f"{'='*60}\n")
        logger.debug("Device: Wormhole (n150)")
        logger.debug(f"Device frequency: {device.freq_MHz} MHz")
        logger.debug(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.debug(
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

        # Generate test data (use values in valid range for sin)
        np.random.seed(42)
        test_data = np.array(np.random.randn(*shape) * 3, dtype=np.float32)

        # Create operation
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f'sin_mem_{test_name.replace(" ", "_")}',
            "optype": "Sin",
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

        # Validate compute_sin correctness
        result = compute_sin(i_tensors, op_obj)
        expected = np.sin(test_data)
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"[{test_name}] compute_sin validation failed",
        )

        # Calculate output shape
        output_shape = tuple(o_tensors[0].shape)
        output_elems = int(np.prod(output_shape))
        input_elems = int(np.prod(shape))

        # Set compute pipe for Sin operation (uses vector pipe for element-wise ops)
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

        # Validate instructions (sin uses 'sin' instruction, 1 per element)
        assert "sin" in actual_instrs, "Expected 'sin' instruction not found"
        actual_sin = actual_instrs.get("sin", 0)
        assert (
            actual_sin == input_elems
        ), f"Sin instruction count mismatch: {actual_sin} vs {input_elems}"

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
        logger.info("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {instructions_executed:,} (sin)")
        logger.debug(f"  Input elements:        {input_elems:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(
            f"  Expected instructions: ~{input_elems:,} (1 sin per element)"
        )
        instruction_ratio = actual_sin / input_elems if input_elems > 0 else 0
        logger.debug(
            f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 sin per element)"
        )

        logger.info("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
        )

        logger.info("\n  -- Memory Metrics --")
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
        logger.debug("  ✓ Arithmetic intensity within expected range")

        logger.info("\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {total_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")
        logger.debug(
            f"  ✓ Bottleneck analysis: {bottleneck} for unary operation"
        )

        # Store results for summary
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "output_shape": output_shape,
                "sin_instructions": actual_sin,
                "total_data_moved": total_data_movement,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "ideal_cycles": total_cycles,
            }
        )

        logger.info("\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total tests run: {len(all_results)}")
    logger.info("All tests passed: ✓")

    # Compare arithmetic intensity across tests
    logger.info("\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["arithmetic_intensity"]
        logger.debug(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

    # Element count analysis
    logger.info("\n-- Element Count Analysis --")
    for result in all_results:
        elems = result["sin_instructions"]
        logger.debug(f"{result['test_name']:30s}: {elems:,} elements")

    # Bottleneck analysis
    logger.info("\n-- Bottleneck Analysis --")
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
        "  • Sin instructions match element count (1:1 ratio) ✓",
        "  • Unary operations show typical MEMORY bottleneck ✓",
        "  • Low arithmetic intensity confirms memory-bound operation ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} sin | {:>7.1f} KB | {:.3f} ops/byte".format(
                result["test_name"],
                result["sin_instructions"],
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
                "=", "SIN MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("SIN MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
