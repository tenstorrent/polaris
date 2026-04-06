#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for the ArgMax op.

ArgMax returns the indices of the maximum values along a given axis.
Registration: ARITY_1->1, r0_func shape inference, compute_argmax.
Attrs: axis, keepdims, select_last_index.
Output dtype: int64.
"""

import pytest
import os
from pathlib import Path
import numpy as np
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_argmax

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
def _run_argmax(data, axis=0, keepdims=1, select_last_index=0, tag="argmax"):
    """Run ArgMax through SimOp and return (actual_data, expected_data, oT)."""
    i_tensors = [F._from_data("X", data)]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "ArgMax",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {
            "axis": axis,
            "keepdims": keepdims,
            "select_last_index": select_last_index,
        },
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)
    o_tensors[0].data = compute_argmax(i_tensors, op)

    actual = o_tensors[0].data

    # Reference
    X = data
    ax = axis
    if select_last_index:
        X_rev = np.flip(X, axis=ax)
        idx_rev = np.argmax(X_rev, axis=ax, keepdims=bool(keepdims))
        expected = (X.shape[ax] - 1 - idx_rev).astype(np.int64)
    else:
        expected = np.argmax(X, axis=ax, keepdims=bool(keepdims)).astype(np.int64)

    return actual, expected, o_tensors[0]


# ===================================================================
# 1. Shape validation
# ===================================================================
shape_cases = [
    # (input_shape, axis, keepdims, expected_shape, id)
    ((3, 4), 0, 1, (1, 4), "2d_ax0_kd"),
    ((3, 4), 1, 1, (3, 1), "2d_ax1_kd"),
    ((3, 4), 0, 0, (4,), "2d_ax0_nokd"),
    ((3, 4), 1, 0, (3,), "2d_ax1_nokd"),
    ((2, 3, 4), 0, 1, (1, 3, 4), "3d_ax0_kd"),
    ((2, 3, 4), 1, 1, (2, 1, 4), "3d_ax1_kd"),
    ((2, 3, 4), 2, 1, (2, 3, 1), "3d_ax2_kd"),
    ((2, 3, 4), 2, 0, (2, 3), "3d_ax2_nokd"),
    ((2, 3, 4, 5), 0, 1, (1, 3, 4, 5), "4d_ax0_kd"),
    ((2, 3, 4, 5), 3, 0, (2, 3, 4), "4d_ax3_nokd"),
    ((5,), 0, 1, (1,), "1d_kd"),
    ((5,), 0, 0, (), "1d_nokd"),
    ((2, 3, 4, 5), -1, 1, (2, 3, 4, 1), "neg_axis_kd"),
    ((2, 3, 4, 5), -2, 0, (2, 3, 5), "neg_axis_nokd"),
]


class TestArgMaxShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape,axis,keepdims,expected_shape,tid", shape_cases)
    def test_output_shape(self, shape, axis, keepdims, expected_shape, tid):
        data = np.random.randn(*shape).astype(np.float32)
        _, _, oT = _run_argmax(data, axis=axis, keepdims=keepdims, tag=f"shape_{tid}")
        assert (
            tuple(oT.shape) == expected_shape
        ), f"{tid}: {tuple(oT.shape)} != {expected_shape}"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_dtype_int64(self):
        data = np.random.randn(3, 4).astype(np.float32)
        actual, _, oT = _run_argmax(data, axis=0, tag="dtype_check")
        assert oT.dtype == np.int64
        assert actual.dtype == np.int64


# ===================================================================
# 2. Numerical validation
# ===================================================================
numerical_cases = [
    # (input_shape, axis, keepdims, id)
    ((3, 4), 0, 1, "2d_ax0"),
    ((3, 4), 1, 1, "2d_ax1"),
    ((3, 4), 1, 0, "2d_ax1_nokd"),
    ((2, 3, 4), 0, 1, "3d_ax0"),
    ((2, 3, 4), 2, 0, "3d_ax2_nokd"),
    ((2, 3, 4, 5), 1, 1, "4d_ax1"),
    ((5,), 0, 1, "1d"),
    ((10,), 0, 0, "1d_nokd"),
    ((2, 3, 4, 5), -1, 1, "neg_axis"),
]


class TestArgMaxNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape,axis,keepdims,tid", numerical_cases)
    def test_values_match_numpy(self, shape, axis, keepdims, tid):
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float64)
        actual, expected, _ = _run_argmax(
            data, axis=axis, keepdims=keepdims, tag=f"num_{tid}"
        )
        np.testing.assert_array_equal(actual, expected)


# ===================================================================
# 3. Edge cases
# ===================================================================
class TestArgMaxEdgeCases:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        """Single element tensor → index is 0."""
        data = np.array([42.0])
        actual, expected, _ = _run_argmax(data, axis=0, keepdims=0, tag="single")
        assert actual == 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_all_same_values(self):
        """All values equal → first index (0) is returned."""
        data = np.full((4, 4), 5.0)
        actual, expected, _ = _run_argmax(data, axis=1, keepdims=0, tag="same_vals")
        np.testing.assert_array_equal(actual, np.zeros(4, dtype=np.int64))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_axis(self):
        """Negative axis works correctly."""
        data = np.random.randn(2, 3, 4).astype(np.float64)
        actual_neg, _, _ = _run_argmax(data, axis=-1, keepdims=1, tag="neg_ax")
        actual_pos, _, _ = _run_argmax(data, axis=2, keepdims=1, tag="pos_ax")
        np.testing.assert_array_equal(actual_neg, actual_pos)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_at_beginning(self):
        """Max is the first element."""
        data = np.array([[10.0, 1.0, 2.0, 3.0]])
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="max_begin")
        assert actual[0] == 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_at_end(self):
        """Max is the last element."""
        data = np.array([[1.0, 2.0, 3.0, 10.0]])
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="max_end")
        assert actual[0] == 3

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_values(self):
        """All negative values — argmax still finds least negative."""
        data = np.array([[-5.0, -1.0, -3.0, -2.0]])
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="neg_vals")
        assert actual[0] == 1  # -1.0 is the max

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_inf_is_max(self):
        """Positive infinity is always the argmax."""
        data = np.array([[1.0, np.inf, 100.0, 999.0]])
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="inf_max")
        assert actual[0] == 1

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        """Works on larger tensor."""
        np.random.seed(123)
        data = np.random.randn(8, 16, 32).astype(np.float64)
        actual, expected, _ = _run_argmax(data, axis=2, keepdims=0, tag="large")
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_select_last_index(self):
        """select_last_index=1 returns index of last occurrence of max."""
        data = np.array([[5.0, 3.0, 5.0, 2.0]])
        actual, _, _ = _run_argmax(
            data, axis=1, keepdims=0, select_last_index=1, tag="last_idx"
        )
        assert actual[0] == 2  # last occurrence of 5.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_select_first_index_default(self):
        """select_last_index=0 (default) returns index of first occurrence."""
        data = np.array([[5.0, 3.0, 5.0, 2.0]])
        actual, _, _ = _run_argmax(
            data, axis=1, keepdims=0, select_last_index=0, tag="first_idx"
        )
        assert actual[0] == 0  # first occurrence of 5.0


# ===================================================================
# 4. Precision tests with known values
# ===================================================================
class TestArgMaxPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_identity_matrix_rows(self):
        """Identity matrix — argmax along axis=1 returns diagonal indices."""
        data = np.eye(4, dtype=np.float64)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="eye_rows")
        np.testing.assert_array_equal(actual, np.arange(4, dtype=np.int64))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_identity_matrix_cols(self):
        """Identity matrix — argmax along axis=0 returns diagonal indices."""
        data = np.eye(4, dtype=np.float64)
        actual, _, _ = _run_argmax(data, axis=0, keepdims=0, tag="eye_cols")
        np.testing.assert_array_equal(actual, np.arange(4, dtype=np.int64))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_ascending_sequence(self):
        """Ascending sequence → argmax is last element."""
        data = np.arange(10, dtype=np.float64).reshape(1, 10)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="asc")
        assert actual[0] == 9

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_descending_sequence(self):
        """Descending sequence → argmax is first element."""
        data = np.arange(10, 0, -1, dtype=np.float64).reshape(1, 10)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="desc")
        assert actual[0] == 0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_one_hot_rows(self):
        """One-hot rows — argmax gives the hot index."""
        data = np.zeros((3, 5), dtype=np.float64)
        data[0, 2] = 1.0
        data[1, 4] = 1.0
        data[2, 0] = 1.0
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="onehot")
        np.testing.assert_array_equal(actual, np.array([2, 4, 0], dtype=np.int64))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_keepdims_shape_matches(self):
        """keepdims=1 preserves rank, keepdims=0 reduces rank."""
        data = np.random.randn(3, 4, 5).astype(np.float64)
        _, _, oT_kd = _run_argmax(data, axis=1, keepdims=1, tag="kd_yes")
        _, _, oT_nokd = _run_argmax(data, axis=1, keepdims=0, tag="kd_no")
        assert len(oT_kd.shape) == 3  # rank preserved
        assert len(oT_nokd.shape) == 2  # rank reduced


# ===================================================================
# 5. Mathematical properties
# ===================================================================
class TestArgMaxProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_index_in_valid_range(self):
        """All returned indices lie in [0, dim_size)."""
        np.random.seed(7)
        data = np.random.randn(4, 5, 6).astype(np.float64)
        for axis in range(3):
            actual, _, _ = _run_argmax(
                data, axis=axis, keepdims=0, tag=f"range_ax{axis}"
            )
            assert actual.min() >= 0
            assert actual.max() < data.shape[axis]

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_argmax_selects_maximum(self):
        """Value at the returned index equals the actual max along axis."""
        np.random.seed(11)
        data = np.random.randn(3, 4, 5).astype(np.float64)
        axis = 2
        actual, _, _ = _run_argmax(data, axis=axis, keepdims=0, tag="selects_max")
        # Gather values at returned indices
        gathered = np.take_along_axis(
            data, np.expand_dims(actual, axis=axis), axis=axis
        )
        gathered = np.squeeze(gathered, axis=axis)
        true_max = np.max(data, axis=axis)
        np.testing.assert_allclose(gathered, true_max)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_invariant_to_offset(self):
        """argmax(X + c) == argmax(X) for any constant c."""
        np.random.seed(13)
        data = np.random.randn(3, 8).astype(np.float64)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="offset_base")
        shifted, _, _ = _run_argmax(
            data + 1000.0, axis=1, keepdims=0, tag="offset_shift"
        )
        np.testing.assert_array_equal(actual, shifted)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_invariant_to_positive_scale(self):
        """argmax(c * X) == argmax(X) for c > 0."""
        np.random.seed(17)
        data = np.random.randn(4, 6).astype(np.float64)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="scale_base")
        scaled, _, _ = _run_argmax(data * 7.5, axis=1, keepdims=0, tag="scale_pos")
        np.testing.assert_array_equal(actual, scaled)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_flips_with_negative_scale(self):
        """argmax(-X) == argmin(X)."""
        np.random.seed(19)
        data = np.random.randn(3, 10).astype(np.float64)
        am_neg, _, _ = _run_argmax(-data, axis=1, keepdims=0, tag="neg_scale")
        argmin_ref = np.argmin(data, axis=1).astype(np.int64)
        np.testing.assert_array_equal(am_neg, argmin_ref)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_unique_max_deterministic(self):
        """With all unique values, argmax is deterministic regardless of select_last_index."""
        np.random.seed(23)
        data = np.random.randn(3, 10).astype(np.float64)
        first, _, _ = _run_argmax(
            data, axis=1, keepdims=0, select_last_index=0, tag="uniq_first"
        )
        last, _, _ = _run_argmax(
            data, axis=1, keepdims=0, select_last_index=1, tag="uniq_last"
        )
        np.testing.assert_array_equal(first, last)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_batch_independence(self):
        """ArgMax on each batch element matches individual computation."""
        np.random.seed(29)
        data = np.random.randn(4, 8).astype(np.float64)
        actual, _, _ = _run_argmax(data, axis=1, keepdims=0, tag="batch_ind")
        for i in range(4):
            row_am, _, _ = _run_argmax(
                data[i : i + 1], axis=1, keepdims=0, tag=f"row_{i}"
            )
            assert actual[i] == row_am[0]


# ===================================================================
# 6. Memory validation
# ===================================================================
def calculate_argmax_memory_stats(
    op, device, input_shape, axis, keepdims, precision="fp16"
):
    """
    Calculate memory performance metrics for a single argmax operation.

    Args:
        op: SimOp representing the argmax operation
        device: Device instance for execution
        input_shape: Shape of input data tensor
        axis: ArgMax axis
        keepdims: Whether to keep reduced dimensions
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
def test_argmax_memory_validation(capsys, request):
    """
    Test memory validation for argmax operation.
    Validates instructions executed and data moved for various scenarios.

    Run with: pytest tests/test_ops/test_argmax.py::test_argmax_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 60)
    logger.info("ArgMax Operation Memory Validation")
    logger.info("=" * 60)

    # Load device configuration
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        logger.info(f"\nDevice: {device.devname} ({device.name})")
        logger.info(f"Device frequency: {device.freq_MHz} MHz")
        logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.info(
            f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        logger.info(f"\nWarning: Could not load device config: {e}")
        pytest.skip(f"Could not load device config: {e}")
        return

    # Test cases
    test_cases = [
        {
            "name": "2D Row ArgMax",
            "shape": [64, 32],
            "axis": 0,
            "keepdims": 1,
            "description": "Find max across rows",
        },
        {
            "name": "2D Col ArgMax",
            "shape": [32, 64],
            "axis": 1,
            "keepdims": 0,
            "description": "Find max across columns",
        },
        {
            "name": "3D Channel ArgMax",
            "shape": [2, 64, 16],
            "axis": 1,
            "keepdims": 1,
            "description": "Find max across channels",
        },
        {
            "name": "4D Batch ArgMax",
            "shape": [8, 16, 16, 32],
            "axis": 3,
            "keepdims": 0,
            "description": "Find max in last dimension",
        },
    ]

    logger.info(f"\n{'='*60}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*60}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        axis = test_case["axis"]
        keepdims = test_case["keepdims"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Input shape: {shape}")
        logger.debug(f"ArgMax axis: {axis}")
        logger.debug(f"Keep dims: {keepdims}")

        # Generate test data
        np.random.seed(42)
        data_arr = np.random.randn(*shape).astype(np.float32)

        # Create operation
        data_t = F._from_data("X", data_arr)
        out_t = make_tensor("Y")

        op_info = {
            "name": f'argmax_mem_{test_name.replace(" ", "_")}',
            "optype": "ArgMax",
            "inList": [data_t.name],
            "outList": [out_t.name],
            "attrs": {
                "axis": axis,
                "keepdims": keepdims,
                "select_last_index": 0,
            },
        }
        op_obj = SimOp(op_info)
        op_obj.get_perf_counts([data_t], [out_t])

        # Calculate output shape
        output_shape = out_t.shape

        mem_stats = calculate_argmax_memory_stats(
            op_obj, device, shape, axis, keepdims, precision="fp16"
        )

        if mem_stats:
            output_elems = np.prod(output_shape)
            input_elems = np.prod(shape)
            reduction_size = shape[axis]

            logger.debug(f"Output shape: {output_shape}")
            logger.debug(f"Reduction size: {reduction_size}")

            logger.debug("\n  -- Instructions & Operations --")
            logger.debug(
                f"  Instructions executed: {mem_stats['instructions_executed']:,}"
            )
            logger.debug(f"  Input elements:        {input_elems:,}")
            logger.debug(f"  Output elements:       {output_elems:,}")
            logger.debug(
                f"  Expected instructions: ~{input_elems:,} (cmp for all input elements)"
            )

            instruction_ratio = (
                mem_stats["instructions_executed"] / input_elems
                if input_elems > 0
                else 0
            )
            assert (
                0.8 <= instruction_ratio <= 1.5
            ), f"Instruction count mismatch: {mem_stats['instructions_executed']} vs expected ~{input_elems}"
            logger.debug(
                f"  Instruction ratio:     {instruction_ratio:.2f} (instructions per input elem, ✓ within expected range)"
            )

            ops_per_output = (
                mem_stats["instructions_executed"] / output_elems
                if output_elems > 0
                else 0
            )
            logger.debug(
                f"  Ops per output elem:   {ops_per_output:.1f} (equals reduction size: {reduction_size})"
            )

            logger.debug("\n  -- Data Movement --")
            logger.debug(
                f"  Input bytes:      {mem_stats['input_bytes']:,} bytes ({mem_stats['input_bytes']/1024:.2f} KB)"
            )
            logger.debug(
                f"  Output bytes:     {mem_stats['output_bytes']:,} bytes ({mem_stats['output_bytes']/1024:.2f} KB)"
            )
            logger.debug(
                f"  Total data moved: {mem_stats['total_data_moved']:,} bytes ({mem_stats['total_data_moved']/1024:.2f} KB)"
            )

            assert mem_stats["output_bytes"] > 0, "Output bytes should be positive"
            actual_bytes_per_elem = (
                mem_stats["output_bytes"] / output_elems if output_elems > 0 else 0
            )
            # ArgMax output is always int64 (8 bytes per element)
            precision_map = {1: "int8", 2: "int16", 4: "int32", 8: "int64"}
            detected_precision = precision_map.get(
                int(actual_bytes_per_elem), f"{actual_bytes_per_elem} bytes/elem"
            )
            assert (
                actual_bytes_per_elem == 8
            ), f"ArgMax output should be int64 (8 bytes), got {actual_bytes_per_elem}"
            logger.debug(
                f"  Output precision: {detected_precision} ({int(actual_bytes_per_elem)} bytes/element, ✓ correct for ArgMax)"
            )

            logger.debug("\n  -- Memory Metrics --")
            logger.debug(
                f"  Arithmetic intensity:  {mem_stats['arithmetic_intensity']:.4f} ops/byte"
            )
            logger.debug(
                f"  Read/Write ratio:      {mem_stats['read_write_ratio']:.2f}"
            )
            logger.debug(
                f"  Bytes per cycle:       {mem_stats['bytes_per_cycle']:.2f}"
            )

            # Note: Arithmetic intensity increases with reduction size
            if reduction_size > 16:
                logger.debug(
                    f"  Note: Large reduction size ({reduction_size}) increases arithmetic intensity"
                )

            logger.debug("\n  -- Execution Cycles --")
            logger.debug(f"  Compute cycles:   {mem_stats['compute_cycles']:,}")
            logger.debug(f"  Memory cycles:    {mem_stats['memory_cycles']:,}")
            logger.debug(f"    Read cycles:    {mem_stats['mem_rd_cycles']:,}")
            logger.debug(f"    Write cycles:   {mem_stats['mem_wr_cycles']:,}")
            logger.debug(f"  Ideal cycles:     {mem_stats['ideal_cycles']:,}")
            logger.debug(f"  Bottleneck:       {mem_stats['bottleneck']}")

            # Note: Bottleneck depends on device characteristics and reduction size
            # Large reductions typically have higher arithmetic intensity but may still be memory-bound
            if mem_stats["bottleneck"] == "COMPUTE":
                logger.debug(
                    f"  ✓ Compute-bound operation (reduction size={reduction_size})"
                )
            else:
                logger.debug(
                    f"  ✓ Memory-bound operation (reduction size={reduction_size})"
                )

            all_results.append(
                {
                    "test_name": test_name,
                    "input_shape": shape,
                    "output_shape": output_shape,
                    "axis": axis,
                    "reduction_size": reduction_size,
                    "keepdims": keepdims,
                    "stats": mem_stats,
                }
            )
            logger.debug("\n  ✓ Test PASSED")
        else:
            logger.info("\n  ✗ Test FAILED: Could not calculate memory stats")
            assert False, "Failed to calculate memory stats"

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total tests run: {len(all_results)}")
    logger.info("All tests passed: ✓")

    logger.info("\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["stats"]["arithmetic_intensity"]
        logger.debug(
            f"{result['test_name']:30s}: {ai:.4f} ops/byte (reduction size: {result['reduction_size']})"
        )

    logger.info("\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["stats"]["bottleneck"]
        logger.debug(
            f"{result['test_name']:30s}: {bottleneck:10s} (axis={result['axis']}, reduction={result['reduction_size']})"
        )

    logger.info(f"\n{'='*60}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*60}\n")

    # Summary for pytest output
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Instructions match input elements (1 cmp per input) ✓",
        "  • Ops per output = reduction size (expected behavior) ✓",
        "  • Output always int64 (8 bytes per element) ✓",
        "  • Bottleneck depends on reduction size and device characteristics ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        mem_stats = result["stats"]
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} ops | {:>7.1f} KB | {:.3f} ops/byte | reduction={}".format(
                result["test_name"],
                mem_stats["instructions_executed"],
                mem_stats["total_data_moved"] / 1024,
                mem_stats["arithmetic_intensity"],
                result["reduction_size"],
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

    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "ARGMAX MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("ARGMAX MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
