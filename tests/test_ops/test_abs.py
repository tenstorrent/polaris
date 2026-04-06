#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Abs op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_abs

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_abs(input_data, dtype="float32"):
    """Build an Abs op, run shape inference + compute, return (shape, data, expected)."""
    arr = np.array(input_data, dtype=dtype)
    data_t = F._from_data("input", arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "Abs",
        "optype": "Abs",
        "inList": [data_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([data_t], [out_t])

    result = compute_abs([data_t], op)
    expected = np.abs(arr)
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, input_data)
# ---------------------------------------------------------------------------

abs_test_cases = [
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
class TestAbsShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", abs_test_cases, ids=[c[0] for c in abs_test_cases]
    )
    def test_output_shape_matches_input(self, tid, input_data):
        shape, result, _ = _run_abs(input_data)
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
class TestAbsNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, input_data", abs_test_cases, ids=[c[0] for c in abs_test_cases]
    )
    def test_values_match_numpy(self, tid, input_data):
        _, result, expected = _run_abs(input_data)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestAbsEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        _, result, expected = _run_abs([-5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        big = np.random.randn(8, 16, 32).astype("float32").tolist()
        _, result, expected = _run_abs(big)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """abs(inf) = inf, abs(-inf) = inf."""
        _, result, _ = _run_abs([np.inf, -np.inf])
        assert result[0] == np.inf, "abs(inf) should be inf"
        assert result[1] == np.inf, "abs(-inf) should be inf"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """abs(nan) should be nan."""
        _, result, _ = _run_abs([np.nan])
        assert np.isnan(result[0]), "abs(nan) should be nan"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        """abs(0) == 0, abs(-0) == 0."""
        _, result, _ = _run_abs([0.0, -0.0])
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        _, result, expected = _run_abs([-0.5, 1.0, -1.5], dtype="float64")
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        arr = np.array([-3, -1, 0, 1, 3], dtype="int32")
        data_t = F._from_data("input", arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Abs",
            "optype": "Abs",
            "inList": [data_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([data_t], [out_t])
        result = compute_abs([data_t], op)
        expected = np.abs(arr)
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 4. Precision tests with known values
# ===========================================================================
class TestAbsPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_abs_zero(self):
        _, result, _ = _run_abs([0.0])
        np.testing.assert_allclose(result, [0.0], atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_abs_positive(self):
        _, result, _ = _run_abs([3.14])
        np.testing.assert_allclose(result, [3.14], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_abs_negative(self):
        _, result, _ = _run_abs([-3.14])
        np.testing.assert_allclose(result, [3.14], atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_abs_symmetric_pair(self):
        """abs(x) == abs(-x) for specific values."""
        _, res_pos, _ = _run_abs([1.23456])
        _, res_neg, _ = _run_abs([-1.23456])
        np.testing.assert_allclose(res_pos, res_neg, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_abs_one(self):
        _, result, _ = _run_abs([1.0, -1.0])
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-7)


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestAbsProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_non_negative(self):
        """abs(x) >= 0 for all x."""
        data = np.random.randn(100).astype("float32").tolist()
        _, result, _ = _run_abs(data)
        assert np.all(result >= 0), "abs(x) should always be >= 0"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_even_symmetry(self):
        """abs(-x) == abs(x)."""
        x = [0.5, 1.0, 2.5, 3.7, np.pi]
        neg_x = [-v for v in x]
        _, res_pos, _ = _run_abs(x)
        _, res_neg, _ = _run_abs(neg_x)
        np.testing.assert_allclose(
            res_neg, res_pos, rtol=1e-6, err_msg="abs(-x) should equal abs(x)"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_idempotent(self):
        """abs(abs(x)) == abs(x)."""
        data = [-3.0, -1.0, 0.0, 1.0, 3.0]
        _, first, _ = _run_abs(data)
        _, second, _ = _run_abs(first.tolist())
        np.testing.assert_allclose(
            second, first, rtol=1e-6, err_msg="abs(abs(x)) should equal abs(x)"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_positive_fixed_point(self):
        """abs(x) == x when x >= 0."""
        data = [0.0, 0.5, 1.0, 2.0, 100.0]
        _, result, _ = _run_abs(data)
        np.testing.assert_allclose(
            result, data, rtol=1e-6, err_msg="abs(x) should equal x for x >= 0"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_negate(self):
        """abs(x) == -x when x < 0."""
        data = [-0.5, -1.0, -2.0, -100.0]
        _, result, _ = _run_abs(data)
        expected = [-v for v in data]
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, err_msg="abs(x) should equal -x for x < 0"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_triangle_inequality(self):
        """|a + b| <= |a| + |b|."""
        a = np.random.randn(50).astype("float32")
        b = np.random.randn(50).astype("float32")
        _, abs_sum, _ = _run_abs((a + b).tolist())
        _, abs_a, _ = _run_abs(a.tolist())
        _, abs_b, _ = _run_abs(b.tolist())
        assert np.all(abs_sum <= abs_a + abs_b + 1e-6), "|a+b| should be <= |a| + |b|"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_multiplicative(self):
        """|a * b| == |a| * |b|."""
        a = [-2.0, 3.0, -0.5, 1.5]
        b = [3.0, -2.0, 4.0, -1.0]
        product = [ai * bi for ai, bi in zip(a, b)]
        _, abs_product, _ = _run_abs(product)
        _, abs_a, _ = _run_abs(a)
        _, abs_b, _ = _run_abs(b)
        np.testing.assert_allclose(
            abs_product, abs_a * abs_b, rtol=1e-5, err_msg="|a*b| should equal |a|*|b|"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same input → all-same output."""
        val = -2.5
        data = [[val] * 4] * 3
        _, result, _ = _run_abs(data)
        np.testing.assert_allclose(result, np.abs(np.float32(val)), rtol=1e-6)


# ===========================================================================
# 6. Memory and performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_abs_memory_validation(capsys, request):
    """
    Test memory validation for Abs operation.
    Validates 'abs' instructions and data movement for element-wise absolute value.

    This test validates:
    1. Instructions: 'abs' instruction count matches output elements (1 per element)
    2. Data Movement: Reads input, writes output (same size)
    3. Element-wise Operation: Each element processed independently

    Run with: pytest tests/test_ops/test_abs.py::test_abs_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 80)
    logger.info("Abs Operation Memory Validation")
    logger.info("=" * 80)

    # Load device configuration once
    polaris_root = Path(__file__).parent.parent.parent
    config_path = polaris_root / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        logger.info(f"\nDevice: {device.devname} ({device.name})")
        logger.info(f"Frequency: {device.freq_MHz} MHz")
        logger.info(
            f"Peak Bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        pytest.skip(f"Could not load device config: {e}")

    # Test cases: different shapes and value ranges
    test_cases = [
        {
            "name": "1D Vector",
            "shape": (256,),
            "description": "Simple 1D vector absolute value",
        },
        {
            "name": "2D Matrix",
            "shape": (64, 64),
            "description": "2D matrix with mixed positive/negative",
        },
        {
            "name": "3D Tensor",
            "shape": (16, 32, 32),
            "description": "3D tensor absolute value",
        },
        {
            "name": "4D Batch",
            "shape": (8, 16, 32, 32),
            "description": "4D batch tensor (typical CNN feature map)",
        },
        {"name": "Large 2D", "shape": (128, 256), "description": "Large 2D matrix"},
    ]

    logger.info(f"\n{'='*80}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Shape: {shape}")

        # Generate test data with mixed positive/negative values
        np.random.seed(42)
        input_data = np.random.randn(*shape).astype(np.float32)

        # Create operation with fp32 precision for consistency
        data_t = F._from_data("input", input_data)
        out_t = make_tensor("output")

        op_info = {
            "name": f"abs_mem_{test_name.replace(' ', '_')}",
            "optype": "Abs",
            "inList": [data_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.precision = "fp32"
        op.uses_compute_pipe = "vector"

        # Get performance counts
        op.get_perf_counts([data_t], [out_t])

        # Execute using compute function (Abs doesn't support device execution)
        result = compute_abs([data_t], op)
        out_t.data = result

        # Verify correctness
        expected_output = np.abs(input_data)
        actual_output = out_t.data
        np.testing.assert_allclose(
            actual_output,
            expected_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Abs output mismatch for {test_name}",
        )

        # Extract performance stats directly
        perf_stats = op.perf_stats
        num_elements = int(np.prod(shape))

        # Validate output shape
        assert out_t.shape == list(
            shape
        ), f"Output shape {out_t.shape} != expected {list(shape)}"
        logger.debug(f"Output shape: {out_t.shape}")

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate 'abs' instruction is present
        assert (
            "abs" in actual_instrs
        ), f"Expected 'abs' instruction for Abs, got {list(actual_instrs.keys())}"

        # Get memory metrics
        input_bytes = perf_stats.get("inBytes", 0)
        output_bytes = perf_stats.get("outBytes", 0)
        total_data_moved = input_bytes + output_bytes

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Calculate estimated cycles from device model
        # Memory bandwidth (bytes per cycle)
        mem_clock_mhz = device.memfreq_MHz
        peak_bw_gb_s = device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        bytes_per_mem_cycle = (
            (peak_bw_gb_s * 1e9) / (mem_clock_mhz * 1e6) if mem_clock_mhz > 0 else 0
        )

        # Compute throughput (vector pipe)
        compute_clock_mhz = device.freq_MHz
        vector_ops_per_cycle = 512  # Vector pipe operations per cycle

        # Estimated cycles
        memory_cycles = (
            total_data_moved / bytes_per_mem_cycle if bytes_per_mem_cycle > 0 else 0
        )
        compute_cycles = (
            total_instructions / vector_ops_per_cycle if vector_ops_per_cycle > 0 else 0
        )
        ideal_cycles = max(compute_cycles, memory_cycles)

        # Bottleneck
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        logger.debug("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {total_instructions:,}")
        logger.debug(f"  Instruction types:     {dict(actual_instrs)}")
        logger.debug(f"  Input elements:        {num_elements:,}")
        logger.debug(f"  Output elements:       {num_elements:,}")

        # Validate: 'abs' instructions should match output elements (1 per element)
        assert (
            abs(total_instructions - num_elements) <= num_elements * 0.1
        ), f"Instruction mismatch: {total_instructions} vs expected ~{num_elements}"
        logger.debug("  ✓ Instruction count validates (1 'abs' per element)")

        logger.debug("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {input_bytes:,} ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {output_bytes:,} ({output_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_moved:,} ({total_data_moved/1024:.2f} KB)"
        )

        # Validate: output bytes should equal input bytes (same shape)
        assert (
            abs(output_bytes - input_bytes) <= 1
        ), f"Input/Output bytes should be equal for Abs"
        logger.debug("  ✓ Input/Output bytes equal (element-wise operation)")

        logger.debug("\n  -- Memory Metrics --")
        logger.debug(
            f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
        )
        logger.debug(
            f"  Bytes per element:     {output_bytes/num_elements if num_elements > 0 else 0:.1f}"
        )

        # Abs is typically memory-bound (simple compute)
        assert (
            arithmetic_intensity < 1.0
        ), f"Arithmetic intensity too high for memory-bound Abs: {arithmetic_intensity}"
        logger.debug("  ✓ Low arithmetic intensity (memory-bound operation)")

        logger.debug("\n  -- Execution Cycles (Estimated) --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,.0f}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,.0f}")
        logger.debug(f"  Ideal cycles:     {ideal_cycles:,.0f}")
        logger.debug(f"  Bottleneck:       {bottleneck}")

        # Validate: Abs should be memory-bound for typical cases
        if num_elements > 1000:
            assert (
                bottleneck == "MEMORY"
            ), f"Expected MEMORY bottleneck, got {bottleneck}"
            logger.debug("  ✓ Memory-bound as expected")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "num_elements": num_elements,
                "instructions": total_instructions,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "total_data_moved": total_data_moved,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
            }
        )

        logger.debug("\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*80}\n")
    logger.info(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    logger.info(f"\n-- Arithmetic Intensity Comparison --")
    logger.info(f"{'Test Name':<30s} {'Ops/Byte':<12s} {'Data Moved':<15s}")
    logger.info("-" * 60)
    for result in all_results:
        logger.debug(
            f"{result['test_name']:<30s} {result['arithmetic_intensity']:<12.4f} {result['total_data_moved']/1024:>10.1f} KB"
        )

    # Shape Analysis
    logger.info(f"\n-- Shape & Element Count --")
    logger.info(f"{'Test Name':<30s} {'Shape':<20s} {'Elements':<15s}")
    logger.info("-" * 70)
    for result in all_results:
        shape_str = "x".join(map(str, result["shape"]))
        logger.debug(
            f"{result['test_name']:<30s} {shape_str:<20s} {result['num_elements']:>12,}"
        )

    # Bottleneck Analysis
    logger.info(f"\n-- Bottleneck Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    logger.info("-" * 80)
    for result in all_results:
        logger.debug(
            f"{result['test_name']:<30s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*80}\n")

    # Create pytest summary
    summary_lines = [
        "✓ Abs Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • 'abs' instructions match output elements (1:1 ratio) ✓",
        "  • All operations are MEMORY-bound ✓",
        "  • Input/Output bytes equal (element-wise operation) ✓",
        "  • Low arithmetic intensity (simple unary operation) ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        shape_str = "x".join(map(str, result["shape"]))
        summary_lines.append(
            "  ✓ {:<28s} | {:>7,} abs | {:>8.1f} KB | {}".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                shape_str,
            )
        )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "ABS MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("ABS MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
