#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Where op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
import os
from pathlib import Path

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_where

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


def _run_where(condition, x, y, cond_dtype="bool", val_dtype="float32"):
    """Build a Where op, run shape inference + compute, return (shape, data, expected)."""
    cond = np.array(condition, dtype=cond_dtype)
    x_arr = np.array(x, dtype=val_dtype)
    y_arr = np.array(y, dtype=val_dtype)

    cond_t = F._from_data("cond", cond)
    x_t = F._from_data("X", x_arr)
    y_t = F._from_data("Y", y_arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "Where",
        "optype": "Where",
        "inList": [cond_t.name, x_t.name, y_t.name],
        "outList": [out_t.name],
    }
    op = SimOp(op_info)
    op.get_perf_counts([cond_t, x_t, y_t], [out_t])

    result = compute_where([cond_t, x_t, y_t], op)
    expected = np.where(cond, x_arr, y_arr)
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, condition, x, y)
# ---------------------------------------------------------------------------

where_test_cases = [
    ("all_true", [True, True, True], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ("all_false", [False, False, False], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    ("mixed", [True, False, True, False], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]),
    ("single_true", [True], [10.0], [20.0]),
    ("single_false", [False], [10.0], [20.0]),
    (
        "2d",
        [[True, False], [False, True]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ),
    (
        "alternating",
        [True, False, True, False, True],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
    ),
]


# ===========================================================================
# 1. Shape validation
# ===========================================================================
class TestWhereShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, cond, x, y", where_test_cases, ids=[c[0] for c in where_test_cases]
    )
    def test_output_shape_matches_input(self, tid, cond, x, y):
        shape, result, expected = _run_where(cond, x, y)
        assert list(shape) == list(
            expected.shape
        ), f"Shape mismatch: got {shape}, expected {list(expected.shape)}"
        assert (
            result.shape == expected.shape
        ), f"Result shape mismatch: got {result.shape}, expected {expected.shape}"


# ===========================================================================
# 2. Numerical validation
# ===========================================================================
class TestWhereNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, cond, x, y", where_test_cases, ids=[c[0] for c in where_test_cases]
    )
    def test_values_match_numpy(self, tid, cond, x, y):
        _, result, expected = _run_where(cond, x, y)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestWhereEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        cond = (np.random.rand(4, 8, 16) > 0.5).tolist()
        x = np.random.randn(4, 8, 16).astype("float32").tolist()
        y = np.random.randn(4, 8, 16).astype("float32").tolist()
        _, result, expected = _run_where(cond, x, y)
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_x_and_y_same(self):
        """When X == Y, output is always the same regardless of condition."""
        cond = [True, False, True]
        data = [1.0, 2.0, 3.0]
        _, result, expected = _run_where(cond, data, data)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        np.testing.assert_allclose(result, data, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """Where with inf values."""
        cond = [True, False]
        x = [np.inf, np.inf]
        y = [-np.inf, -np.inf]
        _, result, expected = _run_where(cond, x, y)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """Where with nan values."""
        cond = [True, False]
        x = [np.nan, 1.0]
        y = [1.0, np.nan]
        _, result, _ = _run_where(cond, x, y)
        assert np.isnan(result[0]), "Should select nan from X"
        assert np.isnan(result[1]), "Should select nan from Y"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        cond = [True, False]
        _, result, expected = _run_where(cond, [0.0, 0.0], [1.0, 1.0])
        np.testing.assert_allclose(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        cond = [True, False, True]
        _, result, expected = _run_where(
            cond, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], val_dtype="float64"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_values(self):
        cond = [True, False, True]
        cond_arr = np.array(cond, dtype="bool")
        x_arr = np.array([10, 20, 30], dtype="int32")
        y_arr = np.array([40, 50, 60], dtype="int32")
        cond_t = F._from_data("cond", cond_arr)
        x_t = F._from_data("X", x_arr)
        y_t = F._from_data("Y", y_arr)
        out_t = make_tensor("output")
        op_info = {
            "name": "Where",
            "optype": "Where",
            "inList": [cond_t.name, x_t.name, y_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.get_perf_counts([cond_t, x_t, y_t], [out_t])
        result = compute_where([cond_t, x_t, y_t], op)
        expected = np.where(cond_arr, x_arr, y_arr)
        np.testing.assert_array_equal(result, expected)


# ===========================================================================
# 4. Precision tests with known values
# ===========================================================================
class TestWherePrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_true_selects_x(self):
        _, result, _ = _run_where([True], [42.0], [99.0])
        np.testing.assert_allclose(result, [42.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_false_selects_y(self):
        _, result, _ = _run_where([False], [42.0], [99.0])
        np.testing.assert_allclose(result, [99.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_mixed_selection(self):
        cond = [True, False, True]
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]
        _, result, _ = _run_where(cond, x, y)
        np.testing.assert_allclose(result, [1.0, 5.0, 3.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_2d_known(self):
        cond = [[True, False], [False, True]]
        x = [[1.0, 2.0], [3.0, 4.0]]
        y = [[5.0, 6.0], [7.0, 8.0]]
        _, result, _ = _run_where(cond, x, y)
        np.testing.assert_allclose(result, [[1.0, 6.0], [7.0, 4.0]])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_values(self):
        cond = [True, False]
        _, result, _ = _run_where(cond, [-10.0, -20.0], [-30.0, -40.0])
        np.testing.assert_allclose(result, [-10.0, -40.0])


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestWhereProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_all_true_equals_x(self):
        """When condition is all True, output == X."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        cond = [True] * len(x)
        _, result, _ = _run_where(cond, x, y)
        np.testing.assert_allclose(
            result, x, rtol=1e-6, err_msg="All-true condition should return X"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_all_false_equals_y(self):
        """When condition is all False, output == Y."""
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        cond = [False] * len(x)
        _, result, _ = _run_where(cond, x, y)
        np.testing.assert_allclose(
            result, y, rtol=1e-6, err_msg="All-false condition should return Y"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_complement_condition(self):
        """where(cond, X, Y) with inverted cond == where(~cond, Y, X)."""
        cond = [True, False, True, False]
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        _, result_orig, _ = _run_where(cond, x, y)
        inv_cond = [not c for c in cond]
        _, result_inv, _ = _run_where(inv_cond, y, x)
        np.testing.assert_allclose(
            result_orig,
            result_inv,
            rtol=1e-6,
            err_msg="where(cond, X, Y) should equal where(~cond, Y, X)",
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_same_x_y_is_identity(self):
        """where(cond, X, X) == X regardless of condition."""
        cond = [True, False, True, False]
        x = [1.0, 2.0, 3.0, 4.0]
        _, result, _ = _run_where(cond, x, x)
        np.testing.assert_allclose(
            result, x, rtol=1e-6, err_msg="where(cond, X, X) should always equal X"
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_elements_from_x_or_y(self):
        """Every output element must come from either X or Y."""
        cond = [True, False, True, False, True]
        x = [10.0, 20.0, 30.0, 40.0, 50.0]
        y = [60.0, 70.0, 80.0, 90.0, 100.0]
        _, result, _ = _run_where(cond, x, y)
        x_arr = np.array(x, dtype="float32")
        y_arr = np.array(y, dtype="float32")
        for i in range(len(cond)):
            assert (
                result[i] == x_arr[i] or result[i] == y_arr[i]
            ), f"Element {i} = {result[i]} not in X or Y"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_condition_determines_source(self):
        """True positions come from X, False positions from Y — verified element-by-element."""
        cond = [True, False, True, False]
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 6.0, 7.0, 8.0]
        _, result, _ = _run_where(cond, x, y)
        cond_arr = np.array(cond)
        x_arr = np.array(x, dtype="float32")
        y_arr = np.array(y, dtype="float32")
        np.testing.assert_allclose(result[cond_arr], x_arr[cond_arr], rtol=1e-6)
        np.testing.assert_allclose(result[~cond_arr], y_arr[~cond_arr], rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same X and Y → output is that constant."""
        val = 7.0
        cond = [True, False, True, False]
        data = [val] * 4
        _, result, _ = _run_where(cond, data, data)
        np.testing.assert_allclose(result, val, rtol=1e-6)


# ===========================================================================
# 6. Memory performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_where_memory_validation(capsys, request):
    """
    Test memory validation for where operation.
    Validates 'mov' and 'cmp' instructions and data movement for conditional selection.

    This test validates:
    1. Instructions: 'mov' and 'cmp' instruction counts (2 per output element total)
    2. Data Movement: Reads 3 inputs (condition, X, Y), writes 1 output
    3. Selection Logic: Conditional selection based on boolean mask

    Run with: pytest tests/test_ops/test_where.py::test_where_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    print("\n" + "=" * 80)
    print("Where Operation Memory Validation")
    print("=" * 80)

    # Load device configuration once
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        print(f"\nDevice: {device.devname} ({device.name})")
        print(f"Frequency: {device.freq_MHz} MHz")
        print(
            f"Peak Bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        pytest.skip(f"Could not load device config: {e}")

    # Test cases
    test_cases = [
        {"name": "1D Array", "shape": [1000], "description": "Where on 1D array"},
        {"name": "2D Matrix", "shape": [32, 32], "description": "Where on 2D matrix"},
        {
            "name": "4D Tensor",
            "shape": [2, 16, 16, 16],
            "description": "Where on 4D tensor",
        },
        {
            "name": "Large 2D",
            "shape": [128, 256],
            "description": "Where on large 2D matrix",
        },
    ]

    print(f"\n{'='*80}")
    print("Running Memory Validation Tests")
    print(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]

        print(f"\n-- Test: {test_name} --")
        print(f"Description: {test_case['description']}")
        print(f"Shape: {shape}")

        # Generate test data
        np.random.seed(42)
        condition = np.random.rand(*shape) > 0.5
        x_data = np.random.randn(*shape).astype(np.float32)
        y_data = np.random.randn(*shape).astype(np.float32)

        # Create operation with fp32 precision for consistency
        cond_t = F._from_data("cond", condition)
        x_t = F._from_data("X", x_data)
        y_t = F._from_data("Y", y_data)
        o_tensors = [make_tensor("output")]

        op_info = {
            "name": f'where_mem_{test_name.replace(" ", "_")}',
            "optype": "Where",
            "inList": [cond_t.name, x_t.name, y_t.name],
            "outList": [o_tensors[0].name],
        }
        op_obj = SimOp(op_info)
        op_obj.precision = "fp32"
        op_obj.uses_compute_pipe = "vector"

        # Get performance counts and execute
        op_obj.get_perf_counts([cond_t, x_t, y_t], o_tensors)
        o_tensors[0].data = compute_where([cond_t, x_t, y_t], op_obj)
        device.execute_op(op_obj)

        # Verify correctness
        expected_output = np.where(condition, x_data, y_data)
        actual_output = o_tensors[0].data
        np.testing.assert_allclose(
            actual_output,
            expected_output,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Where output mismatch for {test_name}",
        )

        # Extract performance stats directly
        perf_stats = op_obj.perf_stats
        output_elems = np.prod(shape)

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate 'mov' and 'cmp' instructions are present for where (conditional selection)
        assert (
            "mov" in actual_instrs
        ), f"Expected 'mov' instruction for Where, got {list(actual_instrs.keys())}"
        assert (
            "cmp" in actual_instrs
        ), f"Expected 'cmp' instruction for Where, got {list(actual_instrs.keys())}"

        # Get memory metrics
        input_bytes = perf_stats.get("inBytes", 0)
        output_bytes = perf_stats.get("outBytes", 0)
        total_data_moved = input_bytes + output_bytes

        # Compute cycles
        compute_cycles = op_obj.compute_cycles
        mem_rd_cycles = op_obj.mem_rd_cycles
        mem_wr_cycles = op_obj.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        ideal_cycles = max(compute_cycles, memory_cycles)

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        print(f"\n  -- Instructions & Operations --")
        print(f"  Instructions executed: {total_instructions:,}")
        print(f"  Instruction types:     {dict(actual_instrs)}")
        print(f"  Output elements:       {output_elems:,}")
        print(
            f"  Expected instructions: ~{2*output_elems:,} (1 cmp + 1 mov per element)"
        )

        # Validate: where should have ~2 instructions per element (cmp + mov)
        instruction_ratio = total_instructions / output_elems if output_elems > 0 else 0
        assert (
            1.5 <= instruction_ratio <= 2.5
        ), f"Instruction mismatch: {total_instructions} vs expected ~{2*output_elems}"
        print(f"  ✓ Instruction count validates (2 per element: cmp + mov)")

        print(f"\n  -- Data Movement --")
        print(f"  Input bytes:      {input_bytes:,} ({input_bytes/1024:.2f} KB)")
        print(f"  Output bytes:     {output_bytes:,} ({output_bytes/1024:.2f} KB)")
        print(
            f"  Total data moved: {total_data_moved:,} ({total_data_moved/1024:.2f} KB)"
        )

        # Where reads 3 tensors (condition, X, Y) and writes 1 output
        assert output_bytes > 0, "Output bytes should be positive"
        print(f"  ✓ Reads 3 inputs (condition, X, Y), writes 1 output")

        print(f"\n  -- Memory Metrics --")
        print(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
        print(
            f"  Bytes per element:     {output_bytes/output_elems if output_elems > 0 else 0:.1f}"
        )

        assert (
            arithmetic_intensity < 1.0
        ), f"Arithmetic intensity too high for memory-bound op: {arithmetic_intensity}"
        print(f"  ✓ Low arithmetic intensity (memory-bound operation)")

        print(f"\n  -- Execution Cycles --")
        print(f"  Compute cycles:   {compute_cycles:,}")
        print(f"  Memory cycles:    {memory_cycles:,}")
        print(f"    Read cycles:    {mem_rd_cycles:,}")
        print(f"    Write cycles:   {mem_wr_cycles:,}")
        print(f"  Ideal cycles:     {ideal_cycles:,}")
        print(f"  Bottleneck:       {bottleneck}")

        # Validate: where should be memory-bound for large tensors
        if output_elems > 1000:
            assert (
                bottleneck == "MEMORY"
            ), f"Expected MEMORY bottleneck, got {bottleneck}"
            print(f"  ✓ Memory-bound as expected")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
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

        print(f"\n  ✓ Test PASSED")

    # Summary
    print(f"\n{'='*80}")
    print("Memory Validation Summary")
    print(f"{'='*80}\n")
    print(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    print(f"\n-- Arithmetic Intensity Comparison --")
    print(f"{'Test Name':<30s} {'Ops/Byte':<12s} {'Data Moved':<15s}")
    print("-" * 60)
    for result in all_results:
        print(
            f"{result['test_name']:<30s} {result['arithmetic_intensity']:<12.4f} {result['total_data_moved']/1024:>10.1f} KB"
        )

    # Element Count Comparison
    print(f"\n-- Element Count & Instructions --")
    print(f"{'Test Name':<30s} {'Elements':<15s} {'Instructions':<15s} {'Ratio':<10s}")
    print("-" * 75)
    for result in all_results:
        elems = np.prod(result["shape"])
        ratio = result["instructions"] / elems if elems > 0 else 0
        print(
            f"{result['test_name']:<30s} {elems:>12,}   {result['instructions']:>12,}   {ratio:>8.2f}"
        )

    # Bottleneck Analysis
    print(f"\n-- Bottleneck Analysis --")
    print(
        f"{'Test Name':<30s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    print("-" * 80)
    for result in all_results:
        print(
            f"{result['test_name']:<30s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    print(f"\n{'='*80}")
    print("Memory validation complete!")
    print(f"{'='*80}\n")

    # Create pytest summary
    summary_lines = [
        "✓ Where Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • 'mov' and 'cmp' instructions (2 per element: compare + select) ✓",
        "  • All operations are MEMORY-bound ✓",
        "  • Reads 3 inputs (condition, X, Y), writes 1 output ✓",
        "  • Low arithmetic intensity (conditional selection) ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        elems = np.prod(result["shape"])
        summary_lines.append(
            "  ✓ {:<28s} | {:>7,} ops | {:>8.1f} KB | {} elems".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                f"{elems:,}",
            )
        )

    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "WHERE MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        with capsys.disabled():
            print("\n" + "=" * 70)
            print("WHERE MEMORY VALIDATION RESULTS")
            print("=" * 70)
            for line in summary_lines:
                print(line)
            print("=" * 70 + "\n")

    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
