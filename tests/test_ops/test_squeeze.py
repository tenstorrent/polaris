#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the Squeeze op."""

import pytest
import os
from pathlib import Path
import numpy as np

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_squeeze

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
def _run_squeeze(data_np, axes_np, tag="squeeze"):
    """Run Squeeze through SimOp and return (actual_output, expected_output)."""
    axes_np = np.asarray(axes_np, dtype=np.int64)

    i_tensors = [
        F._from_data("data", data_np),
        F._from_data("axes", axes_np),
    ]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "Squeeze",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)
    o_tensors[0].data = compute_squeeze(i_tensors, op)

    actual = o_tensors[0].data

    # Also test compute_squeeze directly to validate compute function
    expected_from_compute = compute_squeeze(i_tensors, op)
    expected = np.squeeze(data_np, axis=tuple(int(a) for a in axes_np))

    # Verify compute_squeeze matches numpy reference
    np.testing.assert_array_equal(
        expected_from_compute, expected, err_msg=f"compute_squeeze mismatch for {tag}"
    )

    return actual, expected, o_tensors[0]


# ===================================================================
# 1. Shape validation tests
# ===================================================================
shape_test_cases = [
    # (data_shape, axes, expected_out_shape, id)
    ([1, 3, 4], [0], [3, 4], "remove_first"),
    ([3, 1, 4], [1], [3, 4], "remove_middle"),
    ([3, 4, 1], [2], [3, 4], "remove_last"),
    ([1, 1, 3, 4], [0, 1], [3, 4], "remove_two_leading"),
    ([1, 3, 1, 4], [0, 2], [3, 4], "remove_non_adjacent"),
    ([1, 1, 1], [0, 1, 2], [], "squeeze_all_to_scalar"),
    ([1, 3, 1, 4, 1], [0, 2, 4], [3, 4], "remove_three"),
    ([2, 3, 4], [], [2, 3, 4], "no_axes_no_change"),  # empty axes, no size-1 dims
    ([1], [0], [], "single_elem_to_scalar"),
    ([1, 1], [0], [1], "remove_one_of_two"),
]


class TestSqueezeShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("data_shape,axes,expected_shape,tid", shape_test_cases)
    def test_output_shape(self, data_shape, axes, expected_shape, tid):
        data = (
            np.random.randn(*data_shape).astype(np.float32)
            if data_shape
            else np.float32(1.0)
        )
        if not data_shape:
            data = np.array(1.0, dtype=np.float32).reshape([])
        _, _, oT = _run_squeeze(data, axes, tag=f"shape_{tid}")
        assert (
            list(oT.shape) == expected_shape
        ), f"{tid}: {oT.shape} != {expected_shape}"


# ===================================================================
# 2. Numerical validation tests
# ===================================================================
numerical_cases = [
    # (data_shape, axes, id)
    ([1, 4, 5], [0], "3d_remove_batch"),
    ([2, 1, 3], [1], "3d_remove_mid"),
    ([2, 3, 1], [2], "3d_remove_last"),
    ([1, 1, 2, 3], [0, 1], "4d_remove_two"),
    ([1, 2, 1, 3, 1], [0, 2, 4], "5d_remove_three"),
    ([1], [0], "scalar"),
]


class TestSqueezeNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("data_shape,axes,tid", numerical_cases)
    def test_values(self, data_shape, axes, tid):
        np.random.seed(42)
        data = np.random.randn(*data_shape).astype(np.float32)
        actual, expected, _ = _run_squeeze(data, axes, tag=f"num_{tid}")
        np.testing.assert_array_equal(actual, expected, err_msg=f"{tid} mismatch")


# ===================================================================
# 3. Edge-case tests
# ===================================================================
class TestSqueezeEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_axis(self):
        """Negative axis indices should work."""
        data = np.random.randn(3, 1, 4).astype(np.float32)
        actual, expected, _ = _run_squeeze(data, [-2], tag="neg_axis")
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_last_axis(self):
        """axes=[-1] on trailing size-1 dim."""
        data = np.random.randn(3, 4, 1).astype(np.float32)
        actual, expected, _ = _run_squeeze(data, [-1], tag="neg_last")
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == [3, 4]

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        """Squeeze on a large tensor preserves data."""
        data = np.random.randn(1, 64, 128, 1).astype(np.float32)
        actual, expected, _ = _run_squeeze(data, [0, 3], tag="large")
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == [64, 128]

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64(self):
        """Works with float64 dtype."""
        data = np.random.randn(1, 3, 1).astype(np.float64)
        actual, expected, _ = _run_squeeze(data, [0, 2], tag="f64")
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.float64

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int32(self):
        """Works with integer dtype."""
        data = np.array([[[10], [20], [30]]], dtype=np.int32)  # shape (1,3,1)
        actual, expected, _ = _run_squeeze(data, [0, 2], tag="i32")
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == [3]

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_inf(self):
        """Inf values survive squeeze."""
        data = np.array([[[np.inf, -np.inf, 0.0]]], dtype=np.float32)  # (1,1,3)
        actual, expected, _ = _run_squeeze(data, [0, 1], tag="inf")
        np.testing.assert_array_equal(actual, expected)
        assert actual[0] == np.inf
        assert actual[1] == -np.inf

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_nan(self):
        """NaN values survive squeeze."""
        data = np.array([[[np.nan, 1.0]]], dtype=np.float32)  # (1,1,2)
        actual, expected, _ = _run_squeeze(data, [0, 1], tag="nan")
        assert np.isnan(actual[0])
        assert actual[1] == 1.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_empty_axes(self):
        """Empty axes array on tensor with no size-1 dims → no change."""
        data = np.random.randn(2, 3, 4).astype(np.float32)
        actual, expected, _ = _run_squeeze(data, [], tag="empty_axes")
        np.testing.assert_array_equal(actual, expected)
        assert list(actual.shape) == [2, 3, 4]


# ===================================================================
# 4. Precision tests with known values
# ===================================================================
class TestSqueezePrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sequential_values(self):
        """Sequential data is preserved exactly."""
        data = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        actual, _, _ = _run_squeeze(data, [0], tag="seq")
        expected = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_identity_matrix(self):
        """Identity matrix passes through correctly."""
        data = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
        actual, _, _ = _run_squeeze(data, [0], tag="eye")
        np.testing.assert_array_equal(actual, np.eye(4, dtype=np.float32))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros(self):
        """All zeros preserved."""
        data = np.zeros((1, 1, 5), dtype=np.float32)
        actual, _, _ = _run_squeeze(data, [0, 1], tag="zeros")
        np.testing.assert_array_equal(actual, np.zeros(5, dtype=np.float32))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_ones(self):
        """All ones preserved."""
        data = np.ones((3, 1, 4, 1), dtype=np.float32)
        actual, _, _ = _run_squeeze(data, [1, 3], tag="ones")
        np.testing.assert_array_equal(actual, np.ones((3, 4), dtype=np.float32))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        """Single element tensor squeezed to scalar."""
        data = np.array([[[42.0]]], dtype=np.float32)  # (1,1,1)
        actual, _, _ = _run_squeeze(data, [0, 1, 2], tag="single")
        assert float(actual) == 42.0
        assert actual.ndim == 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_small_values(self):
        """Very small values preserved."""
        vals = np.array([1e-38, 1e-30, 1e-20], dtype=np.float32).reshape(1, 3, 1)
        actual, _, _ = _run_squeeze(vals, [0, 2], tag="small")
        np.testing.assert_array_equal(actual, vals.squeeze(axis=(0, 2)))


# ===================================================================
# 5. Mathematical property tests
# ===================================================================
class TestSqueezeProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_inverse_of_unsqueeze(self):
        """Squeeze is the inverse of unsqueeze: squeeze(unsqueeze(x, ax), ax) == x."""
        original = np.random.randn(3, 4).astype(np.float32)
        unsqueezed = np.expand_dims(original, axis=1)  # (3,1,4)
        actual, _, _ = _run_squeeze(unsqueezed, [1], tag="inv_unsq")
        np.testing.assert_array_equal(actual, original)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_element_count_preserved(self):
        """Total number of elements unchanged after squeeze."""
        data = np.random.randn(1, 5, 1, 7, 1).astype(np.float32)
        actual, _, _ = _run_squeeze(data, [0, 2, 4], tag="numel")
        assert actual.size == data.size

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_data_contiguity(self):
        """Flat data ordering identical before and after squeeze."""
        data = np.random.randn(1, 3, 1, 4).astype(np.float32)
        actual, _, _ = _run_squeeze(data, [0, 2], tag="flat")
        np.testing.assert_array_equal(actual.ravel(), data.ravel())

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_squeeze_commutative(self):
        """Order of axes in the squeeze call doesn't matter."""
        data = np.random.randn(1, 3, 1, 4, 1).astype(np.float32)
        actual_a, _, _ = _run_squeeze(data, [0, 2, 4], tag="comm_a")
        actual_b, _, _ = _run_squeeze(data, [4, 0, 2], tag="comm_b")
        np.testing.assert_array_equal(actual_a, actual_b)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_rank_reduction(self):
        """Output rank == input rank - len(axes)."""
        data = np.random.randn(1, 3, 1, 4, 1).astype(np.float32)
        axes = [0, 2, 4]
        actual, _, oT = _run_squeeze(data, axes, tag="rank")
        assert len(oT.shape) == data.ndim - len(axes)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_squeeze_preserves_statistics(self):
        """Mean and std are identical after squeeze (same data)."""
        data = np.random.randn(1, 10, 1, 20).astype(np.float32)
        actual, _, _ = _run_squeeze(data, [0, 2], tag="stats")
        np.testing.assert_allclose(actual.mean(), data.mean(), atol=1e-7)
        np.testing.assert_allclose(actual.std(), data.std(), atol=1e-7)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_squeeze_dtype_preserved(self):
        """Output dtype matches input dtype."""
        for dt in [np.float32, np.float64, np.int32, np.int64]:
            data = np.ones((1, 4), dtype=dt)
            actual, _, _ = _run_squeeze(data, [0], tag=f"dtype_{dt.__name__}")
            assert actual.dtype == dt


# ===================================================================
# 6. Memory validation
# ===================================================================
def calculate_squeeze_memory_stats(op, device, input_shape, axes, precision="fp16"):
    """
    Calculate memory performance metrics for a single squeeze operation.

    Args:
        op: SimOp representing the squeeze operation
        device: Device instance for execution
        input_shape: Shape of input data tensor
        axes: Axes to squeeze
        precision: Data precision (default: 'fp16')

    Returns:
        Dictionary containing memory statistics
    """

    # Normalize precision to string format
    def normalize_precision(prec):
        if prec is None:
            return "fp16"
        if hasattr(prec, "name"):
            prec = prec.name
        prec_str = str(prec).lower()
        dtype_map = {
            "float32": "fp16",
            "float16": "fp16",
            "bfloat16": "bf16",
            "int8": "int8",
            "int32": "int32",
        }
        return dtype_map.get(prec_str, "fp16")

    # Set operation configuration
    op.precision = normalize_precision(precision)
    if op.uses_compute_pipe is None:
        op.uses_compute_pipe = "vector"

    # Execute the operation to get performance stats
    if op.perf_stats is not None:
        device.execute_op(op)

        # Extract basic metrics
        total_instructions = 0
        if "instrs" in op.perf_stats:
            for instr, count in op.perf_stats["instrs"].items():
                total_instructions += count

        input_bytes = op.perf_stats.get("inBytes", 0)
        output_bytes = op.perf_stats.get("outBytes", 0)
        total_data_moved = input_bytes + output_bytes

        # Compute cycles
        compute_cycles = op.compute_cycles
        mem_rd_cycles = op.mem_rd_cycles
        mem_wr_cycles = op.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck determination
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        # Execution time
        ideal_cycles = max(compute_cycles, memory_cycles)
        execution_time_ms = ideal_cycles / device.freq_MHz / 1e3

        # Additional metrics
        read_write_ratio = input_bytes / output_bytes if output_bytes > 0 else 0
        bytes_per_cycle = total_data_moved / memory_cycles if memory_cycles > 0 else 0

        return {
            "instructions_executed": total_instructions,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "total_data_moved": total_data_moved,
            "arithmetic_intensity": arithmetic_intensity,
            "compute_cycles": compute_cycles,
            "mem_rd_cycles": mem_rd_cycles,
            "mem_wr_cycles": mem_wr_cycles,
            "memory_cycles": memory_cycles,
            "ideal_cycles": ideal_cycles,
            "bottleneck": bottleneck,
            "execution_time_ms": execution_time_ms,
            "read_write_ratio": read_write_ratio,
            "bytes_per_cycle": bytes_per_cycle,
            "precision": op.precision,
        }
    else:
        return None


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_squeeze_memory_validation(capsys, request):
    """
    Test memory validation for squeeze operation.
    Validates instructions executed and data moved for various scenarios.

    It performs data movement (mov instructions) but no arithmetic computation.

    Run with: pytest tests/test_ops/test_squeeze.py::test_squeeze_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    print("\n" + "=" * 60)
    print("Squeeze Operation Memory Validation")
    print("=" * 60)

    # Load device configuration
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        print(f"\nDevice: {device.devname} ({device.name})")
        print(f"Device frequency: {device.freq_MHz} MHz")
        print(f"Memory frequency: {device.memfreq_MHz} MHz")
        print(
            f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        print(f"\nWarning: Could not load device config: {e}")
        pytest.skip(f"Could not load device config: {e}")
        return

    # Test cases
    test_cases = [
        {
            "name": "Remove Leading Dim",
            "shape": [1, 64, 32],
            "axes": [0],
            "description": "Remove batch dimension",
        },
        {
            "name": "Remove Middle Dim",
            "shape": [8, 1, 64, 32],
            "axes": [1],
            "description": "Remove middle singleton",
        },
        {
            "name": "Remove Two Dims",
            "shape": [1, 64, 1, 32],
            "axes": [0, 2],
            "description": "Remove two singletons",
        },
        {
            "name": "Remove Multiple Dims",
            "shape": [1, 32, 1, 64, 1],
            "axes": [0, 2, 4],
            "description": "Remove three singletons",
        },
    ]

    print(f"\n{'='*60}")
    print("Running Memory Validation Tests")
    print(f"{'='*60}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        axes = test_case["axes"]
        description = test_case["description"]

        print(f"\n-- Test: {test_name} --")
        print(f"Description: {description}")
        print(f"Input shape: {shape}, Squeeze axes: {axes}")

        # Generate test data
        np.random.seed(42)
        data_arr = np.random.randn(*shape).astype(np.float32)
        axes_arr = np.array(axes, dtype=np.int64)

        # Create operation
        data_t = F._from_data("data", data_arr)
        axes_t = F._from_data("axes", axes_arr)
        out_t = make_tensor("Y")

        op_info = {
            "name": f'squeeze_mem_{test_name.replace(" ", "_")}',
            "optype": "Squeeze",
            "inList": [data_t.name, axes_t.name],
            "outList": [out_t.name],
        }
        op_obj = SimOp(op_info)
        data_t.op_in = [op_info["name"]]
        axes_t.op_in = [op_info["name"]]
        out_t.op_out = [op_info["name"]]

        # Set operation precision
        op_obj.precision = "fp32"

        # Get performance counts
        op_obj.get_perf_counts([data_t, axes_t], [out_t])

        # Validate compute_squeeze correctness
        result = compute_squeeze([data_t, axes_t], op_obj)
        expected = np.squeeze(data_arr, axis=tuple(int(a) for a in axes_arr))
        np.testing.assert_array_equal(
            result, expected, err_msg=f"[{test_name}] compute_squeeze validation failed"
        )

        # Calculate output shape
        output_shape = list(out_t.shape)
        output_elems = int(np.prod(output_shape)) if output_shape else 1
        input_elems = int(np.prod(shape))
        dims_removed = len(axes)

        # Set compute pipe for Squeeze operation (uses vector pipe for data movement)
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

        # Validate element count preservation (squeeze doesn't change data, only shape)
        assert (
            input_elems == output_elems
        ), f"Element count should be preserved: {input_elems} != {output_elems}"

        # Validate element counts
        assert (
            actual_out_elems == output_elems
        ), f"Output element count mismatch: {actual_out_elems} vs {output_elems}"

        # Validate instructions (squeeze uses 'mov' for data movement)
        assert "mov" in actual_instrs, "Expected 'mov' instruction not found"
        actual_mov = actual_instrs.get("mov", 0)

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
        print(f"\n  -- Instructions & Operations --")
        print(f"  Instructions executed: {instructions_executed:,} (mov)")
        print(f"  Input elements:        {input_elems:,}")
        print(f"  Output elements:       {output_elems:,}")
        print(f"  Expected instructions: ~{output_elems:,} (mov for data movement)")
        instruction_ratio = actual_mov / output_elems if output_elems > 0 else 0
        print(f"  Instruction ratio:     {instruction_ratio:.2f} (✓ mov per output)")
        print(f"  Dimensions removed:    {dims_removed}")

        print(f"\n  -- Data Movement --")
        print(
            f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
        )
        print(
            f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
        )
        print(
            f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
        )
        print(
            f"  Elements preserved: {input_elems:,} → {output_elems:,} (shape transformation only)"
        )
        print(f"  Output shape:     {' × '.join(map(str, output_shape))}")

        print(f"\n  -- Memory Metrics --")
        print(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
        # For reshape operations, arithmetic intensity should be very low (< 0.5)
        assert (
            arithmetic_intensity < 0.5
        ), f"Arithmetic intensity too high for shape transformation: {arithmetic_intensity}"
        print(f"  ✓ Low AI expected for shape transformation (no arithmetic)")

        print(f"\n  -- Execution Cycles --")
        print(f"  Compute cycles:   {compute_cycles:,}")
        print(f"  Memory cycles:    {memory_cycles:,}")
        print(f"    Read cycles:    {mem_rd_cycles:,}")
        print(f"    Write cycles:   {mem_wr_cycles:,}")
        print(f"  Ideal cycles:     {total_cycles:,}")
        print(f"  Bottleneck:       {bottleneck}")
        # Validate: squeeze should always be memory-bound (pure data movement)
        assert (
            bottleneck == "MEMORY"
        ), f"Expected MEMORY bottleneck for squeeze operation, got {bottleneck}"
        print(f"  ✓ Bottleneck analysis: {bottleneck} for shape transformation")

        # Store results for summary
        all_results.append(
            {
                "test_name": test_name,
                "input_shape": shape,
                "output_shape": output_shape,
                "axes": axes,
                "dims_removed": dims_removed,
                "mov_instructions": actual_mov,
                "total_data_moved": total_data_movement,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "ideal_cycles": total_cycles,
            }
        )

        print(f"\n  ✓ Test PASSED")

    # Summary
    print(f"\n{'='*60}")
    print("Memory Validation Summary")
    print(f"{'='*60}\n")
    print(f"Total tests run: {len(all_results)}")
    print(f"All tests passed: ✓")

    print(f"\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["arithmetic_intensity"]
        print(
            f"{result['test_name']:30s}: {ai:.4f} ops/byte (dims removed: {result['dims_removed']})"
        )

    print(f"\n-- Shape Transformation Analysis --")
    for result in all_results:
        input_shape_str = "×".join(map(str, result["input_shape"]))
        output_shape_str = "×".join(map(str, result["output_shape"]))
        print(f"{result['test_name']:30s}: {input_shape_str} → {output_shape_str}")

    print(f"\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["bottleneck"]
        print(f"{result['test_name']:30s}: {bottleneck}")

    print(f"\n{'='*60}")
    print("Memory validation complete!")
    print(f"{'='*60}\n")

    # Summary for pytest output (visible even without -s flag)
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Instructions: 'mov' for data movement ✓",
        "  • Element count preserved (shape transformation) ✓",
        "  • All operations are MEMORY-bound ✓",
        "  • Arithmetic Intensity: < 0.5 ops/byte (no arithmetic) ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} mov | {:>7.1f} KB | {:.4f} ops/byte | dims_removed={}".format(
                result["test_name"],
                result["mov_instructions"],
                result["total_data_moved"] / 1024,
                result["arithmetic_intensity"],
                result["dims_removed"],
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
                "=", "MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            print("\n" + "=" * 70)
            print("MEMORY VALIDATION RESULTS")
            print("=" * 70)
            for line in summary_lines:
                print(line)
            print("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
