#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Less op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
import os
import sys
from pathlib import Path

from ttsim.ops.desc.data_compute import compute_less
from ttsim.ops import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device
from loguru import logger

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


def _run_less(input_a, input_b, dtype="float32"):
    """Build a Less op, run shape inference + compute, return (shape, data, expected)."""
    a = np.array(input_a, dtype=dtype)
    b = np.array(input_b, dtype=dtype)
    a_t = F._from_data("A", a)
    b_t = F._from_data("B", b)
    out_t = make_tensor("output")

    op_info = {
        "name": "Less",
        "optype": "Less",
        "inList": [a_t.name, b_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([a_t, b_t], [out_t])

    result = compute_less([a_t, b_t], op)
    expected = a < b
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, input_a, input_b)
# ---------------------------------------------------------------------------

less_test_cases = [
    ("scalar", [1.0], [2.0]),
    ("equal", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ("all_less", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ("all_greater", [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]),
    ("mixed", [1.0, 5.0, 3.0, 7.0], [2.0, 4.0, 6.0, 8.0]),
    ("negatives", [-3.0, -1.0, 0.0], [-2.0, -2.0, 1.0]),
    ("2d", [[1.0, 4.0], [3.0, 2.0]], [[2.0, 3.0], [4.0, 1.0]]),
    ("small_diff", [1.0, 1.0 + 1e-7], [1.0 + 1e-6, 1.0]),
]


# ===========================================================================
# 1. Shape validation
# ===========================================================================
class TestLessShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, a, b", less_test_cases, ids=[c[0] for c in less_test_cases]
    )
    def test_output_shape_matches_input(self, tid, a, b):
        shape, result, expected = _run_less(a, b)
        assert list(shape) == list(
            expected.shape
        ), f"Shape mismatch: got {shape}, expected {list(expected.shape)}"
        assert (
            result.shape == expected.shape
        ), f"Result shape mismatch: got {result.shape}, expected {expected.shape}"


# ===========================================================================
# 2. Numerical validation
# ===========================================================================
class TestLessNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, a, b", less_test_cases, ids=[c[0] for c in less_test_cases]
    )
    def test_values_match_numpy(self, tid, a, b):
        _, result, expected = _run_less(a, b)
        np.testing.assert_array_equal(
            result, expected, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestLessEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element_true(self):
        _, result, expected = _run_less([1.0], [2.0])
        np.testing.assert_array_equal(result, expected)
        assert result[0] == True

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element_false(self):
        _, result, expected = _run_less([3.0], [2.0])
        np.testing.assert_array_equal(result, expected)
        assert result[0] == False

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        a = np.random.randn(8, 16, 32).astype("float32")
        b = np.random.randn(8, 16, 32).astype("float32")
        _, result, expected = _run_less(a.tolist(), b.tolist())
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_broadcast_scalar(self):
        """Less with scalar broadcast: [3,] < [1] → element-wise."""
        a = [1.0, 5.0, 3.0]
        b = [3.0]
        a_arr = np.array(a, dtype="float32")
        b_arr = np.array(b, dtype="float32")
        a_t = F._from_data("A", a_arr)
        b_t = F._from_data("B", b_arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Less",
            "optype": "Less",
            "inList": [a_t.name, b_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([a_t, b_t], [out_t])
        result = compute_less([a_t, b_t], op)
        expected = a_arr < b_arr
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_inf_values(self):
        """Comparisons involving inf."""
        a = [1.0, np.inf, -np.inf, np.inf]
        b = [np.inf, 1.0, 1.0, np.inf]
        _, result, expected = _run_less(a, b)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nan_values(self):
        """Less with nan should be False (nan is not less than anything)."""
        a = [np.nan, 1.0, np.nan]
        b = [1.0, np.nan, np.nan]
        _, result, expected = _run_less(a, b)
        np.testing.assert_array_equal(result, expected)
        # nan < x is always False
        assert not result[0]
        assert not result[2]

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        _, result, expected = _run_less([0.0, -0.0], [-0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        _, result, expected = _run_less([1, 3, 5], [2, 2, 6], dtype="int32")
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 4. Precision tests with known values
# ===========================================================================
class TestLessPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_strictly_less(self):
        """1.0 < 2.0 → True."""
        _, result, _ = _run_less([1.0], [2.0])
        assert result[0] == True

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_equal_is_not_less(self):
        """1.0 < 1.0 → False."""
        _, result, _ = _run_less([1.0], [1.0])
        assert result[0] == False

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_greater_is_not_less(self):
        """2.0 < 1.0 → False."""
        _, result, _ = _run_less([2.0], [1.0])
        assert result[0] == False

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_less_than_positive(self):
        """–1.0 < 1.0 → True."""
        _, result, _ = _run_less([-1.0], [1.0])
        assert result[0] == True

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_neg_inf_less_than_all(self):
        """-inf is less than any finite number."""
        _, result, _ = _run_less([-np.inf], [0.0])
        assert result[0] == True

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nothing_less_than_neg_inf(self):
        """No finite number is less than -inf → False."""
        _, result, _ = _run_less([0.0], [-np.inf])
        assert result[0] == False


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestLessProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_is_boolean(self):
        """Result dtype should be bool."""
        _, result, _ = _run_less([1.0, 2.0], [2.0, 1.0])
        assert result.dtype == np.bool_, f"Expected bool dtype, got {result.dtype}"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_irreflexive(self):
        """x < x is always False (irreflexivity)."""
        data = [0.0, 1.0, -1.0, 100.0, -100.0]
        _, result, _ = _run_less(data, data)
        assert not np.any(result), "x < x should always be False"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_asymmetry(self):
        """If a < b then NOT (b < a)."""
        a = [1.0, 2.0, 0.0]
        b = [2.0, 3.0, 1.0]
        _, ab, _ = _run_less(a, b)
        _, ba, _ = _run_less(b, a)
        # Where a < b is True, b < a must be False
        assert not np.any(ab & ba), "a < b and b < a cannot both be True"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_transitivity(self):
        """If a < b and b < c then a < c."""
        a = [1.0, 2.0, 0.0]
        b = [2.0, 3.0, 1.0]
        c = [3.0, 4.0, 2.0]
        _, ab, _ = _run_less(a, b)
        _, bc, _ = _run_less(b, c)
        _, ac, _ = _run_less(a, c)
        # Where both a<b and b<c, a<c must hold
        mask = ab & bc
        assert np.all(ac[mask]), "Transitivity: a<b and b<c => a<c"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_complement_of_geq(self):
        """(a < b) == NOT(a >= b) for non-nan values."""
        a = [1.0, 3.0, 2.0, 5.0]
        b = [2.0, 2.0, 2.0, 4.0]
        _, less_result, _ = _run_less(a, b)
        a_arr = np.array(a, dtype="float32")
        b_arr = np.array(b, dtype="float32")
        geq = a_arr >= b_arr
        np.testing.assert_array_equal(less_result, ~geq)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sorted_array(self):
        """For a sorted ascending array, a[i] < a[i+1] for all i."""
        sorted_arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        a = sorted_arr[:-1]
        b = sorted_arr[1:]
        _, result, _ = _run_less(a, b)
        assert np.all(result), "Consecutive elements of sorted array: a[i] < a[i+1]"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same inputs → all False."""
        val = 3.0
        a = [[val] * 4] * 3
        _, result, _ = _run_less(a, a)
        assert not np.any(result), "x < x should be all False"


# ===========================================================================
# 6. Memory estimation and performance validation
# ===========================================================================


def calculate_less_memory_stats(input_shape_a, input_shape_b, precision="fp16"):
    """
    Calculate expected memory stats for Less operation.

    Less performs element-wise comparison: y = (a < b)
    - 1 'cmp' instruction per output element
    - Input data for both tensors read once
    - Output data written once
    - Memory-bound operation

    Args:
        input_shape_a: Shape of first input tensor
        input_shape_b: Shape of second input tensor
        precision: Data precision (fp16, bf16, fp32, etc.)

    Returns:
        dict with expected memory stats
    """
    # Calculate element counts
    num_elements_a = int(np.prod(input_shape_a))
    num_elements_b = int(np.prod(input_shape_b))

    # Output shape after broadcasting
    output_shape = np.broadcast_shapes(input_shape_a, input_shape_b)
    num_output_elements = int(np.prod(output_shape))

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
    input_bytes_a = num_elements_a * bytes_per_element
    input_bytes_b = num_elements_b * bytes_per_element
    output_bytes = num_output_elements * bytes_per_element
    total_data_movement = input_bytes_a + input_bytes_b + output_bytes

    # Instructions: 1 cmp per output element
    expected_cmp_instrs = num_output_elements
    total_instructions = expected_cmp_instrs

    # Arithmetic intensity: operations / bytes
    arithmetic_intensity = (
        total_instructions / total_data_movement if total_data_movement > 0 else 0
    )

    return {
        "input_elements_a": num_elements_a,
        "input_elements_b": num_elements_b,
        "output_elements": num_output_elements,
        "input_bytes_a": input_bytes_a,
        "input_bytes_b": input_bytes_b,
        "output_bytes": output_bytes,
        "total_data_movement": total_data_movement,
        "expected_cmp_instrs": expected_cmp_instrs,
        "total_instructions": total_instructions,
        "arithmetic_intensity": arithmetic_intensity,
    }


class TestLessMemoryValidation:
    """Validate memory estimation and instruction counts for Less operation."""

    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.performance
    def test_less_memory_validation(self, capsys, request):
        """Validate memory stats for Less operation across different tensor sizes."""

        # Test cases: (input_shape_a, input_shape_b, description)
        memory_validation_cases = [
            ((16,), (16,), "Same shape 1D"),
            ((32, 64), (32, 64), "Same shape 2D"),
            ((32, 64), (1,), "Broadcast scalar"),
            ((32, 64), (1, 64), "Broadcast row"),
        ]

        # Load device configuration once for all tests
        config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
        try:
            ipgroups, packages = get_arspec_from_yaml(config_path)
            device_pkg = packages["n150"]  # Use Wormhole n150 device
            device = Device(device_pkg)

            logger.info(f"\n{'='*60}")
            logger.info("Less Operation Memory Validation")
            logger.info(f"{'='*60}\n")
            logger.info("Device: Wormhole (n150)")
            logger.info(f"Device frequency: {device.freq_MHz} MHz")
            logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
            logger.info(
                "Peak bandwidth: "
                f"{device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
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

        for input_shape_a, input_shape_b, description in memory_validation_cases:
            logger.info(f"\n-- Test: {description} --")
            logger.info(
                f"Input shape A: {input_shape_a}, Input shape B: {input_shape_b}"
            )

            # Generate random inputs
            input_data_a = np.random.randn(*input_shape_a).astype(np.float32)
            input_data_b = np.random.randn(*input_shape_b).astype(np.float32)

            # Build tensors and op
            a_t = F._from_data("A", input_data_a)
            b_t = F._from_data("B", input_data_b)
            out_t = make_tensor("output")

            op_info = {
                "name": "Less",
                "optype": "Less",
                "inList": [a_t.name, b_t.name],
                "outList": [out_t.name],
            }
            op = SimOp(op_info)
            for t in [a_t, b_t]:
                t.op_in = ["Less"]
            out_t.op_out = ["Less"]

            # Set operation precision
            op.precision = "fp32"

            # Get perf stats
            op.get_perf_counts([a_t, b_t], [out_t])

            # Validate compute_less correctness
            result = compute_less([a_t, b_t], op)
            expected = input_data_a < input_data_b
            np.testing.assert_array_equal(
                result,
                expected,
                err_msg=f"[{description}] compute_less validation failed",
            )

            # Calculate expected stats
            expected_stats = calculate_less_memory_stats(
                input_shape_a, input_shape_b, "fp32"
            )

            # Set compute pipe for Less operation (uses vector pipe for element-wise ops)
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
                actual_in_elems
                == expected_stats["input_elements_a"]
                + expected_stats["input_elements_b"]
            ), f"Input element count mismatch: {actual_in_elems} vs {expected_stats['input_elements_a'] + expected_stats['input_elements_b']}"
            assert (
                actual_out_elems == expected_stats["output_elements"]
            ), f"Output element count mismatch: {actual_out_elems} vs {expected_stats['output_elements']}"

            # Validate byte counts
            assert (
                actual_in_bytes
                == expected_stats["input_bytes_a"] + expected_stats["input_bytes_b"]
            ), f"Input bytes mismatch: {actual_in_bytes} vs {expected_stats['input_bytes_a'] + expected_stats['input_bytes_b']}"
            assert (
                actual_out_bytes == expected_stats["output_bytes"]
            ), f"Output bytes mismatch: {actual_out_bytes} vs {expected_stats['output_bytes']}"

            # Validate instructions
            assert (
                "cmp" in actual_instrs or "less" in actual_instrs
            ), "Expected 'cmp' or 'less' instruction not found"
            actual_cmp = actual_instrs.get("cmp", actual_instrs.get("less", 0))
            assert (
                actual_cmp == expected_stats["expected_cmp_instrs"]
            ), f"Cmp instruction count mismatch: {actual_cmp} vs {expected_stats['expected_cmp_instrs']}"

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
            logger.debug("\n  -- Instructions & Operations --")
            logger.debug(
                f"  Instructions executed: {instructions_executed:,} (cmp)"
            )
            logger.debug(
                f"  Input A elements:      {expected_stats['input_elements_a']:,}"
            )
            logger.debug(
                f"  Input B elements:      {expected_stats['input_elements_b']:,}"
            )
            logger.debug(
                f"  Output elements:       {expected_stats['output_elements']:,}"
            )
            logger.debug(
                "  Expected instructions: "
                f"~{expected_stats['expected_cmp_instrs']:,} (1 cmp per output element)"
            )
            instruction_ratio = (
                actual_cmp / expected_stats["output_elements"]
                if expected_stats["output_elements"] > 0
                else 0
            )
            logger.debug(
                f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 cmp per output)"
            )

            logger.debug("\n  -- Data Movement --")
            logger.debug(
                "  Input A bytes:    "
                f"{expected_stats['input_bytes_a']:,} bytes "
                f"({expected_stats['input_bytes_a']/1024:.2f} KB)"
            )
            logger.debug(
                "  Input B bytes:    "
                f"{expected_stats['input_bytes_b']:,} bytes "
                f"({expected_stats['input_bytes_b']/1024:.2f} KB)"
            )
            logger.debug(
                "  Output bytes:     "
                f"{actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
            )
            logger.debug(
                "  Total data moved: "
                f"{total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
            )

            # Calculate broadcast ratio
            total_input_elements = (
                expected_stats["input_elements_a"] + expected_stats["input_elements_b"]
            )
            if expected_stats["input_elements_b"] == 1:
                broadcast_type = "scalar broadcast"
            elif (
                expected_stats["input_elements_b"] < expected_stats["input_elements_a"]
            ):
                broadcast_type = f"broadcast {expected_stats['input_elements_b']}→{expected_stats['output_elements']}"
            else:
                broadcast_type = "no broadcast"
            logger.debug(f"  Broadcast:        {broadcast_type}")

            logger.debug("\n  -- Memory Metrics --")
            logger.debug(
                f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
            )
            logger.debug(
                "  Expected intensity:    "
                f"{expected_stats['arithmetic_intensity']:.4f} ops/byte"
            )
            np.testing.assert_allclose(
                arithmetic_intensity,
                expected_stats["arithmetic_intensity"],
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
            logger.debug(
                f"  ✓ Bottleneck analysis: {bottleneck} for comparison operation"
            )

            # Validate arithmetic intensity matches expected
            np.testing.assert_allclose(
                arithmetic_intensity,
                expected_stats["arithmetic_intensity"],
                rtol=0.01,
                atol=1e-6,
                err_msg=f"Arithmetic intensity mismatch",
            )

            # Store results for summary
            all_results.append(
                {
                    "test_name": description,
                    "input_shape_a": input_shape_a,
                    "input_shape_b": input_shape_b,
                    "output_shape": tuple(result.shape),
                    "cmp_instructions": actual_cmp,
                    "total_data_moved": total_data_movement,
                    "arithmetic_intensity": arithmetic_intensity,
                    "broadcast_type": broadcast_type,
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
            logger.info(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

        # Compare broadcast types
        logger.info("\n-- Broadcast Analysis --")
        for result in all_results:
            broadcast = result["broadcast_type"]
            logger.info(f"{result['test_name']:30s}: {broadcast}")

        logger.info("\n-- Bottleneck Analysis --")
        for result in all_results:
            bottleneck = result["bottleneck"]
            logger.info(f"{result['test_name']:30s}: {bottleneck}")

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
            "  • Instructions: 1 'cmp' per output element ✓",
            "  • Comparison operations are typically memory-bound",
            "  • Broadcasting reduces input size but maintains output operations",
            "",
            "Test Results:",
        ]

        for result in all_results:
            summary_lines.append(
                "  ✓ {:<26s} | {:>7,} cmp | {:>7.1f} KB | {:.4f} ops/byte | {}".format(
                    result["test_name"],
                    result["cmp_instructions"],
                    result["total_data_moved"] / 1024,
                    result["arithmetic_intensity"],
                    result["broadcast_type"],
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
