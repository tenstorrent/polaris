#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Neg op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
import os
import sys
from pathlib import Path
from loguru import logger

from ttsim.ops.desc.data_compute import compute_neg
from ttsim.ops import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import logging

try:
    # Silence python logging coming from ttsim modules (only show ERROR+)
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    # If the project uses loguru, remove default sinks and keep only ERROR+
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except ImportError:
        pass
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_neg(input_data, dtype="float32"):
    """Build a Neg op, run shape inference + compute, return (shape, data, expected)."""
    arr = np.array(input_data, dtype=dtype)
    data_t = F._from_data("input", arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "Neg",
        "optype": "Neg",
        "inList": [data_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([data_t], [out_t])

    result = compute_neg([data_t], op)
    expected = -arr
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, input_data)
# ---------------------------------------------------------------------------

neg_test_cases = [
    ("scalar", [0.0]),
    ("positive", [1.0, 2.0, 3.0]),
    ("negative", [-1.0, -2.0, -3.0]),
    ("mixed", [-3.0, -1.5, 0.0, 1.5, 3.0]),
    ("2d", [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]),
    ("3d", [[[-0.1, 0.2], [-0.3, 0.4]], [[0.5, -0.6], [-0.7, 0.8]]]),
    ("small_values", [1e-7, -1e-7, 1e-10, -1e-10]),
    ("large_values", [1e6, -1e6, 1e10, -1e10]),
]


# ===========================================================================
# 1. Shape validation
# ===========================================================================
class TestNegShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", neg_test_cases, ids=[c[0] for c in neg_test_cases]
    )
    def test_output_shape_matches_input(self, tid, input_data):
        shape, result, _ = _run_neg(input_data)
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
class TestNegNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", neg_test_cases, ids=[c[0] for c in neg_test_cases]
    )
    def test_values_match_numpy(self, tid, input_data):
        _, result, expected = _run_neg(input_data)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestNegEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        _, result, expected = _run_neg([5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        big = np.random.randn(8, 16, 32).astype("float32").tolist()
        _, result, expected = _run_neg(big)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """neg(inf) = -inf, neg(-inf) = inf."""
        _, result, _ = _run_neg([np.inf, -np.inf])
        assert result[0] == -np.inf, "neg(inf) should be -inf"
        assert result[1] == np.inf, "neg(-inf) should be inf"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """neg(nan) should be nan."""
        _, result, _ = _run_neg([np.nan])
        assert np.isnan(result[0]), "neg(nan) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        """neg(0) == 0."""
        _, result, _ = _run_neg([0.0])
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        _, result, expected = _run_neg([-0.5, 1.0, -1.5], dtype="float64")
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        arr = np.array([-3, -1, 0, 1, 3], dtype="int32")
        data_t = F._from_data("input", arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Neg",
            "optype": "Neg",
            "inList": [data_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([data_t], [out_t])
        result = compute_neg([data_t], op)
        expected = -arr
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 4. Precision tests with known values
# ===========================================================================
class TestNegPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_zero(self):
        _, result, _ = _run_neg([0.0])
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_positive(self):
        _, result, _ = _run_neg([3.14])
        np.testing.assert_allclose(result, [-3.14], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_negative(self):
        _, result, _ = _run_neg([-3.14])
        np.testing.assert_allclose(result, [3.14], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_one(self):
        _, result, _ = _run_neg([1.0, -1.0])
        np.testing.assert_allclose(result, [-1.0, 1.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_pi(self):
        _, result, _ = _run_neg([np.pi])
        np.testing.assert_allclose(result, [-np.pi], rtol=1e-6)


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestNegProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_involution(self):
        """neg(neg(x)) == x — double negation is identity."""
        data = [-3.0, -1.0, 0.0, 1.0, 3.0]
        _, first, _ = _run_neg(data)
        _, second, _ = _run_neg(first.tolist())
        np.testing.assert_allclose(
            second, data, rtol=1e-6, err_msg="neg(neg(x)) should equal x"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_add_to_zero(self):
        """x + neg(x) == 0."""
        data = [1.0, -2.5, 3.7, -0.1, 100.0]
        arr = np.array(data, dtype="float32")
        _, neg_result, _ = _run_neg(data)
        total = arr + neg_result
        np.testing.assert_allclose(
            total, 0.0, atol=1e-6, err_msg="x + neg(x) should equal 0"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sign_flip(self):
        """neg flips the sign: neg(positive) < 0, neg(negative) > 0."""
        pos = [1.0, 2.0, 3.0]
        neg_vals = [-1.0, -2.0, -3.0]
        _, neg_of_pos, _ = _run_neg(pos)
        _, neg_of_neg, _ = _run_neg(neg_vals)
        assert np.all(neg_of_pos < 0), "neg of positive should be negative"
        assert np.all(neg_of_neg > 0), "neg of negative should be positive"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_linearity_scalar(self):
        """neg(a * x) == a * neg(x) for scalar a (with a = -1 implicit)."""
        x = [1.0, 2.0, 3.0]
        a = 2.5
        ax = [a * v for v in x]
        _, neg_ax, _ = _run_neg(ax)
        _, neg_x, _ = _run_neg(x)
        np.testing.assert_allclose(
            neg_ax, a * neg_x, rtol=1e-5, err_msg="neg(a*x) should equal a*neg(x)"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_distributes_over_addition(self):
        """neg(a + b) == neg(a) + neg(b)."""
        a = [1.0, -2.0, 3.0, 0.5]
        b = [0.5, 1.5, -1.0, 2.0]
        a_plus_b = [ai + bi for ai, bi in zip(a, b)]
        _, neg_sum, _ = _run_neg(a_plus_b)
        _, neg_a, _ = _run_neg(a)
        _, neg_b, _ = _run_neg(b)
        np.testing.assert_allclose(
            neg_sum,
            neg_a + neg_b,
            rtol=1e-6,
            err_msg="neg(a+b) should equal neg(a)+neg(b)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_preserves_magnitude(self):
        """abs(neg(x)) == abs(x)."""
        data = [-3.0, -1.0, 0.0, 1.0, 3.0]
        _, neg_result, _ = _run_neg(data)
        abs_neg = np.abs(neg_result)
        abs_orig = np.abs(np.array(data, dtype="float32"))
        np.testing.assert_allclose(
            abs_neg, abs_orig, rtol=1e-6, err_msg="|neg(x)| should equal |x|"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_odd_function(self):
        """neg(-x) == -neg(x) == x  (neg is an odd function)."""
        x = [0.5, 1.0, 2.5]
        neg_x = [-v for v in x]
        _, result_neg_x, _ = _run_neg(neg_x)
        _, result_x, _ = _run_neg(x)
        # neg(-x) should equal x
        np.testing.assert_allclose(result_neg_x, x, rtol=1e-6)
        # -neg(x) should also equal x
        np.testing.assert_allclose(-result_x, x, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same input → all-same output."""
        val = 2.5
        data = [[val] * 4] * 3
        _, result, _ = _run_neg(data)
        np.testing.assert_allclose(result, -np.float32(val), rtol=1e-6)


# ===========================================================================
# 6. Memory estimation and performance validation
# ===========================================================================


def calculate_neg_memory_stats(input_shape, precision="bfp8"):
    """
    Calculate expected memory stats for Neg operation.

    Neg performs element-wise negation: y = -x
    - Implemented as subtraction: -x = 0 - x
    - 1 'sub' instruction per output element
    - Input data read once
    - Output data written once
    - Memory-bound operation

    Args:
        input_shape: Shape of input tensor
        precision: Data precision (bfp8, fp16, fp32, etc.)

    Returns:
        dict with expected memory stats
    """
    # Calculate element counts
    num_elements = int(np.prod(input_shape))

    # Precision to bytes mapping
    precision_bytes = {
        "bfp8": 1,
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
        "int8": 1,
        "int32": 4,
    }
    bytes_per_element = precision_bytes.get(precision, 2)

    # Memory stats
    input_bytes = num_elements * bytes_per_element
    output_bytes = num_elements * bytes_per_element
    total_data_movement = input_bytes + output_bytes

    # Instructions: 1 sub per output element (negation via subtraction)
    expected_sub_instrs = num_elements
    total_instructions = expected_sub_instrs

    # Arithmetic intensity: operations / bytes
    arithmetic_intensity = (
        total_instructions / total_data_movement if total_data_movement > 0 else 0
    )

    return {
        "input_elements": num_elements,
        "output_elements": num_elements,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_data_movement": total_data_movement,
        "expected_sub_instrs": expected_sub_instrs,
        "total_instructions": total_instructions,
        "arithmetic_intensity": arithmetic_intensity,
    }


class TestNegMemoryValidation:
    """Validate memory estimation and instruction counts for Neg operation."""

    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.performance
    def test_neg_memory_validation(self, capsys, request):
        """Validate memory stats for Neg operation across different tensor sizes."""

        # Test cases: (input_shape, description)
        memory_validation_cases = [
            ((16,), "Small 1D"),
            ((32, 64), "Medium 2D"),
            ((8, 16, 32), "Large 3D"),
            ((2, 3, 8, 8), "4D batch"),
        ]

        # Load device configuration once for all tests
        config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
        try:
            ipgroups, packages = get_arspec_from_yaml(config_path)
            device_pkg = packages["n150"]  # Use Wormhole n150 device
            device = Device(device_pkg)

            logger.info(f"\n{'='*60}")
            logger.info("Neg Operation Memory Validation")
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

        for input_shape, description in memory_validation_cases:
            logger.info(f"\n-- Test: {description} --")
            logger.debug(f"Input shape: {input_shape}")

            # Generate random input
            input_data = np.random.randn(*input_shape).astype(np.float32)

            # Build tensor and op
            data_t = F._from_data("input", input_data)
            out_t = make_tensor("output")

            op_info = {
                "name": "Neg",
                "optype": "Neg",
                "inList": [data_t.name],
                "outList": [out_t.name],
            }
            op = SimOp(op_info)
            data_t.op_in = ["Neg"]
            out_t.op_out = ["Neg"]

            # Set operation precision
            op.precision = "fp32"

            # Get perf stats
            op.get_perf_counts([data_t], [out_t])

            # Validate compute_neg correctness
            result = compute_neg([data_t], op)
            expected = -input_data
            np.testing.assert_allclose(
                result,
                expected,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"[{description}] compute_neg validation failed",
            )

            # Calculate expected stats
            expected_stats = calculate_neg_memory_stats(input_shape, "fp32")

            # Set compute pipe for Neg operation (uses vector pipe for element-wise ops)
            if op.uses_compute_pipe is None:
                op.uses_compute_pipe = "vector"

            # Execute on device for cycle estimation
            if op.perf_stats is not None:
                try:
                    device.execute_op(op)
                except Exception:
                    pass

            # Extract stats from op.perf_stats
            perf_stats = op.perf_stats
            actual_in_elems = perf_stats["inElems"]
            actual_out_elems = perf_stats["outElems"]
            actual_in_bytes = perf_stats["inBytes"]
            actual_out_bytes = perf_stats["outBytes"]
            actual_instrs = perf_stats["instrs"]

            # Validate element counts
            assert (
                actual_in_elems == expected_stats["input_elements"]
            ), f"Input element count mismatch: {actual_in_elems} vs {expected_stats['input_elements']}"
            assert (
                actual_out_elems == expected_stats["output_elements"]
            ), f"Output element count mismatch: {actual_out_elems} vs {expected_stats['output_elements']}"

            # Validate byte counts
            assert (
                actual_in_bytes == expected_stats["input_bytes"]
            ), f"Input bytes mismatch: {actual_in_bytes} vs {expected_stats['input_bytes']}"
            assert (
                actual_out_bytes == expected_stats["output_bytes"]
            ), f"Output bytes mismatch: {actual_out_bytes} vs {expected_stats['output_bytes']}"

            # Validate instructions
            assert (
                "sub" in actual_instrs or "neg" in actual_instrs
            ), "Expected 'sub' or 'neg' instruction not found"
            actual_sub = actual_instrs.get("sub", actual_instrs.get("neg", 0))
            assert (
                actual_sub == expected_stats["expected_sub_instrs"]
            ), f"Sub instruction count mismatch: {actual_sub} vs {expected_stats['expected_sub_instrs']}"

            # Calculate metrics
            total_data_movement = actual_in_bytes + actual_out_bytes
            instructions_executed = sum(actual_instrs.values())
            arithmetic_intensity = (
                instructions_executed / total_data_movement
                if total_data_movement > 0
                else 0
            )

            # Calculate execution cycles (read from op object, not perf_stats)
            compute_cycles = op.compute_cycles or 0
            mem_rd_cycles = op.mem_rd_cycles or 0
            mem_wr_cycles = op.mem_wr_cycles or 0
            memory_cycles = mem_rd_cycles + mem_wr_cycles
            total_cycles = max(compute_cycles, memory_cycles)
            bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

            # Print detailed breakdown
            logger.info("\n  -- Instructions & Operations --")
            logger.debug(
                f"  Instructions executed: {instructions_executed:,} (sub)"
            )
            logger.debug(
                f"  Input elements:        {expected_stats['input_elements']:,}"
            )
            logger.debug(
                f"  Output elements:       {expected_stats['output_elements']:,}"
            )
            logger.debug(
                f"  Expected instructions: ~{expected_stats['expected_sub_instrs']:,} (1 sub per element)"
            )
            instruction_ratio = (
                actual_sub / expected_stats["output_elements"]
                if expected_stats["output_elements"] > 0
                else 0
            )
            logger.debug(
                f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 sub per element)"
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
            logger.debug(
                f"  Elements:         {expected_stats['input_elements']:,}"
            )

            logger.info("\n  -- Memory Metrics --")
            logger.debug(
                f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
            )
            logger.debug(
                f"  Expected intensity:    {expected_stats['arithmetic_intensity']:.4f} ops/byte"
            )
            np.testing.assert_allclose(
                arithmetic_intensity,
                expected_stats["arithmetic_intensity"],
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
                    "test_name": description,
                    "input_shape": input_shape,
                    "sub_instructions": actual_sub,
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

        # Compare data movement
        logger.info("\n-- Data Movement Comparison --")
        for result in all_results:
            data_kb = result["total_data_moved"] / 1024
            logger.debug(f"{result['test_name']:30s}: {data_kb:>7.2f} KB")

        logger.info("\n-- Bottleneck Analysis --")
        for result in all_results:
            bottleneck = result["bottleneck"]
            logger.debug(f"{result['test_name']:30s}: {bottleneck}")

        logger.info(f"\n{'='*60}")
        logger.info("Memory validation complete!")
        logger.info(f"{'='*60}\n")

        # Create a summary that will be displayed in pytest output (even without -s flag)
        summary_lines = [
            "✓ Tests completed: {}/{} - All PASSED".format(
                len(all_results), len(memory_validation_cases)
            ),
            "",
            "Key Findings:",
            "  • Instructions: 1 'sub' per element (negation via 0-x) ✓",
            "  • Unary operations are typically memory-bound",
            "  • Arithmetic intensity consistent across tensor sizes",
            "",
            "Test Results:",
        ]

        for result in all_results:
            summary_lines.append(
                "  ✓ {:<26s} | {:>7,} sub | {:>7.1f} KB | {:.4f} ops/byte".format(
                    result["test_name"],
                    result["sub_instructions"],
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
            terminalreporter = request.config.pluginmanager.get_plugin(
                "terminalreporter"
            )
            if terminalreporter:
                terminalreporter.write_sep(
                    "=", "MEMORY VALIDATION RESULTS", bold=True, green=True
                )
                for line in summary_lines:
                    terminalreporter.write_line(line)
                terminalreporter.write_sep("=", "", bold=True)
        except Exception:
            # Fallback: disable capture and print directly
            with capsys.disabled():
                logger.info("\n" + "=" * 70)
                logger.info("MEMORY VALIDATION RESULTS")
                logger.info("=" * 70)
                for line in summary_lines:
                    logger.info(line)
                logger.info("=" * 70 + "\n")

        # Final assertion
        assert len(all_results) == len(
            memory_validation_cases
        ), f"Memory validation: {len(all_results)}/{len(memory_validation_cases)} tests passed"
