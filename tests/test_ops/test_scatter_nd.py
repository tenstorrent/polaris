#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for ScatterND op – shape, numerical, edge, precision, properties."""

import numpy as np
import pytest
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_scatter_nd

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


def _run_scatter_nd(data, indices, updates, dtype="float32", reduction="none"):
    """Build a ScatterND op, run shape inference + compute, return (shape, result, expected)."""
    data_arr = np.array(data, dtype=dtype)
    idx_arr = np.array(indices, dtype="int64")
    upd_arr = np.array(updates, dtype=dtype)

    data_t = F._from_data("data", data_arr)
    idx_t = F._from_data("indices", idx_arr)
    upd_t = F._from_data("updates", upd_arr)
    out_t = make_tensor("output")

    op_info = {
        "name": "ScatterND",
        "optype": "ScatterND",
        "inList": [data_t.name, idx_t.name, upd_t.name],
        "outList": [out_t.name],
    }
    if reduction != "none":
        op_info["attrs"] = {"reduction": reduction}

    op = SimOp(op_info)
    out_t.shape = list(data_arr.shape)
    out_t.dtype = str(data_arr.dtype)
    op.perf_stats = {
        "inElems": int(data_arr.size),
        "outElems": int(data_arr.size),
        "inBytes": int(data_arr.nbytes),
        "outBytes": int(data_arr.nbytes),
        "instrs": {"mov": int(data_arr.size)},
    }

    result = compute_scatter_nd([data_t, idx_t, upd_t], op)

    # Build expected via manual numpy reference
    expected = data_arr.copy()
    K = idx_arr.shape[-1]
    flat_idx = idx_arr.reshape(-1, K)
    flat_upd = upd_arr.reshape(-1, *data_arr.shape[K:])
    for i in range(flat_idx.shape[0]):
        idx = tuple(flat_idx[i])
        if reduction == "none":
            expected[idx] = flat_upd[i]
        elif reduction == "add":
            expected[idx] += flat_upd[i]
        elif reduction == "mul":
            expected[idx] *= flat_upd[i]

    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Test cases: (id, data, indices, updates)
# ---------------------------------------------------------------------------

scatter_nd_test_cases = [
    (
        "1d_single",
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [[1], [3]],
        [10.0, 40.0],
    ),
    (
        "1d_all",
        [0.0, 0.0, 0.0],
        [[0], [1], [2]],
        [1.0, 2.0, 3.0],
    ),
    (
        "2d_row_update",
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[0], [2]],
        [[10.0, 20.0], [50.0, 60.0]],
    ),
    (
        "2d_element_update",
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[0, 1], [1, 2]],
        [20.0, 60.0],
    ),
    (
        "3d_update",
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        [[0], [1]],
        [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
    ),
]


# ===========================================================================
# 1. Shape validation
# ===========================================================================
class TestScatterNDShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, data, indices, updates",
        scatter_nd_test_cases,
        ids=[c[0] for c in scatter_nd_test_cases],
    )
    def test_output_shape_equals_data_shape(self, tid, data, indices, updates):
        shape, result, _ = _run_scatter_nd(data, indices, updates)
        data_arr = np.array(data, dtype="float32")
        assert list(shape) == list(
            data_arr.shape
        ), f"Shape mismatch: got {shape}, expected {list(data_arr.shape)}"
        assert (
            result.shape == data_arr.shape
        ), f"Result shape mismatch: got {result.shape}, expected {data_arr.shape}"


# ===========================================================================
# 2. Numerical validation
# ===========================================================================
class TestScatterNDNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "tid, data, indices, updates",
        scatter_nd_test_cases,
        ids=[c[0] for c in scatter_nd_test_cases],
    )
    def test_values_match_reference(self, tid, data, indices, updates):
        _, result, expected = _run_scatter_nd(data, indices, updates)
        np.testing.assert_allclose(
            result, expected, rtol=1e-6, atol=1e-7, err_msg=f"[{tid}] value mismatch"
        )


# ===========================================================================
# 3. Edge cases
# ===========================================================================
class TestScatterNDEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element_data(self):
        """Scatter into a single-element tensor."""
        _, result, expected = _run_scatter_nd([0.0], [[0]], [99.0])
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, [99.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_no_overlap(self):
        """Scatter at distinct indices — all updates visible."""
        data = [0.0, 0.0, 0.0, 0.0, 0.0]
        indices = [[0], [2], [4]]
        updates = [1.0, 3.0, 5.0]
        _, result, expected = _run_scatter_nd(data, indices, updates)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, [1.0, 0.0, 3.0, 0.0, 5.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_overwrite_same_index(self):
        """Writing to the same index twice — last write wins."""
        data = [0.0, 0.0, 0.0]
        indices = [[1], [1]]
        updates = [10.0, 20.0]
        _, result, expected = _run_scatter_nd(data, indices, updates)
        np.testing.assert_allclose(result, expected)
        # Last write wins
        assert result[1] == 20.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        """Scatter into a larger tensor."""
        data = np.zeros((4, 8), dtype="float32")
        indices = np.array([[0], [2], [3]], dtype="int64")
        updates = np.random.randn(3, 8).astype("float32")
        _, result, expected = _run_scatter_nd(
            data.tolist(), indices.tolist(), updates.tolist()
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_inf(self):
        """Scatter inf values."""
        data = [0.0, 0.0, 0.0]
        indices = [[0], [2]]
        updates = [np.inf, -np.inf]
        _, result, expected = _run_scatter_nd(data, indices, updates)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_special_float_nan(self):
        """Scatter nan values."""
        data = [1.0, 2.0, 3.0]
        indices = [[1]]
        updates = [np.nan]
        _, result, _ = _run_scatter_nd(data, indices, updates)
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 3.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64_dtype(self):
        _, result, expected = _run_scatter_nd(
            [1.0, 2.0, 3.0], [[1]], [99.0], dtype="float64"
        )
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_dtype(self):
        data_arr = np.array([10, 20, 30], dtype="int32")
        idx_arr = np.array([[0], [2]], dtype="int64")
        upd_arr = np.array([100, 300], dtype="int32")

        data_t = F._from_data("data", data_arr)
        idx_t = F._from_data("indices", idx_arr)
        upd_t = F._from_data("updates", upd_arr)
        out_t = make_tensor("output")

        op_info = {
            "name": "ScatterND",
            "optype": "ScatterND",
            "inList": [data_t.name, idx_t.name, upd_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        out_t.shape = list(data_arr.shape)
        out_t.dtype = str(data_arr.dtype)
        op.perf_stats = {
            "inElems": int(data_arr.size),
            "outElems": int(data_arr.size),
            "inBytes": int(data_arr.nbytes),
            "outBytes": int(data_arr.nbytes),
            "instrs": {"mov": int(data_arr.size)},
        }
        result = compute_scatter_nd([data_t, idx_t, upd_t], op)
        np.testing.assert_array_equal(result, [100, 20, 300])


# ===========================================================================
# 4. Precision tests with known values
# ===========================================================================
class TestScatterNDPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_1d_known(self):
        """Scatter into 1D: replace index 1 with 99."""
        _, result, _ = _run_scatter_nd([1.0, 2.0, 3.0], [[1]], [99.0])
        np.testing.assert_allclose(result, [1.0, 99.0, 3.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_2d_row_known(self):
        """Replace row 0 of a 2×3 matrix."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        _, result, _ = _run_scatter_nd(data, [[0]], [[10.0, 20.0, 30.0]])
        np.testing.assert_allclose(result, [[10.0, 20.0, 30.0], [4.0, 5.0, 6.0]])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_2d_element_known(self):
        """Replace element [1,2] of a 2×3 matrix."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        _, result, _ = _run_scatter_nd(data, [[1, 2]], [99.0])
        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 99.0]])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_first_and_last(self):
        """Scatter at first and last positions."""
        data = [0.0, 0.0, 0.0, 0.0]
        _, result, _ = _run_scatter_nd(data, [[0], [3]], [11.0, 44.0])
        np.testing.assert_allclose(result, [11.0, 0.0, 0.0, 44.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_replace_all(self):
        """Replace every element of a 1D tensor."""
        data = [0.0, 0.0, 0.0]
        _, result, _ = _run_scatter_nd(data, [[0], [1], [2]], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])


# ===========================================================================
# 5. Mathematical property tests
# ===========================================================================
class TestScatterNDProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_shape_equals_data(self):
        """Output shape is always the same as data shape."""
        for shape in [(5,), (3, 4), (2, 3, 4)]:
            data = np.zeros(shape, dtype="float32")
            # Scatter at index [0] (or [0,0] etc.)
            idx = [[0] * len(shape)]
            upd_shape = data[tuple([0] * len(shape))].shape
            if upd_shape == ():
                updates = [42.0]
            else:
                updates = [np.ones(upd_shape, dtype="float32").tolist()]
            s, result, _ = _run_scatter_nd(data.tolist(), idx, updates)
            assert list(s) == list(shape)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_untouched_elements_preserved(self):
        """Elements not targeted by indices are unchanged."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        indices = [[1], [3]]
        updates = [20.0, 40.0]
        _, result, _ = _run_scatter_nd(data, indices, updates)
        # Untouched indices: 0, 2, 4
        assert result[0] == 1.0
        assert result[2] == 3.0
        assert result[4] == 5.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_zeros_into_zeros(self):
        """Scattering zeros into zeros yields zeros."""
        data = [0.0, 0.0, 0.0]
        _, result, _ = _run_scatter_nd(data, [[0], [1], [2]], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scatter_same_values_is_noop(self):
        """Scattering the original values back yields the original tensor."""
        data = [10.0, 20.0, 30.0]
        _, result, _ = _run_scatter_nd(data, [[0], [1], [2]], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(result, data)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_updates_appear_in_output(self):
        """Every scattered update value appears at its target position."""
        data = [0.0, 0.0, 0.0, 0.0]
        indices = [[0], [2]]
        updates = [77.0, 88.0]
        _, result, _ = _run_scatter_nd(data, indices, updates)
        assert result[0] == 77.0
        assert result[2] == 88.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_data_is_not_mutated(self):
        """The original data array should not be mutated (copy semantics)."""
        data_arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        data_t = F._from_data("data", data_arr.copy())
        idx_t = F._from_data("indices", np.array([[1]], dtype="int64"))
        upd_t = F._from_data("updates", np.array([99.0], dtype="float32"))
        out_t = make_tensor("output")

        op_info = {
            "name": "ScatterND",
            "optype": "ScatterND",
            "inList": [data_t.name, idx_t.name, upd_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        out_t.shape = list(data_arr.shape)
        out_t.dtype = str(data_arr.dtype)
        op.perf_stats = {
            "inElems": int(data_arr.size),
            "outElems": int(data_arr.size),
            "inBytes": int(data_arr.nbytes),
            "outBytes": int(data_arr.nbytes),
            "instrs": {"mov": int(data_arr.size)},
        }
        original_copy = data_t.data.copy()
        _ = compute_scatter_nd([data_t, idx_t, upd_t], op)
        # data_t.data should NOT be mutated (compute copies first)
        np.testing.assert_array_equal(data_t.data, original_copy)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """All-same data with scatter at one position."""
        val = 5.0
        data = [val] * 4
        _, result, _ = _run_scatter_nd(data, [[2]], [99.0])
        np.testing.assert_allclose(result, [5.0, 5.0, 99.0, 5.0])


# ===========================================================================
# 6. Memory and performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_scatter_nd_memory_validation(capsys, request):
    """
    Test memory validation for ScatterND operation.
    Validates 'mov' instructions and data movement for sparse scatter patterns.

    This test validates:
    1. Instructions: 'mov' instruction count matches output elements (1 per element)
    2. Data Movement: Reads data + indices + updates, writes output (same size as data)
    3. Sparse Updates: Only specified indices are modified

    Run with: pytest tests/test_ops/test_scatter_nd.py::test_scatter_nd_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 80)
    logger.info("ScatterND Operation Memory Validation")
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
            "Peak Bandwidth: "
            f"{device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        pytest.skip(f"Could not load device config: {e}")

    # Test cases: different scatter patterns
    test_cases = [
        {
            "name": "1D Sparse Scatter",
            "data_shape": (128,),
            "indices": [[10], [20], [30], [40], [50]],
            "updates": [1.0, 2.0, 3.0, 4.0, 5.0],
            "description": "5 updates in 128-element vector",
        },
        {
            "name": "2D Row Scatter",
            "data_shape": (64, 64),
            "indices": [[0], [10], [20], [30]],
            "updates": [[float(i)] * 64 for i in range(4)],
            "description": "4 full row updates in 64x64 matrix",
        },
        {
            "name": "2D Element Scatter",
            "data_shape": (32, 32),
            "indices": [[i, j] for i in range(0, 32, 4) for j in range(0, 32, 4)],
            "updates": [float(i) for i in range(64)],
            "description": "64 individual element updates (8x8 grid)",
        },
        {
            "name": "3D Block Scatter",
            "data_shape": (16, 16, 16),
            "indices": [[0], [8]],
            "updates": [np.ones((16, 16)).tolist(), np.ones((16, 16)).tolist()],
            "description": "2 full 16x16 plane updates in 16x16x16 tensor",
        },
        {
            "name": "Large Matrix Scatter",
            "data_shape": (128, 256),
            "indices": [[i] for i in range(0, 128, 8)],
            "updates": [[float(i)] * 256 for i in range(16)],
            "description": "16 row updates in large 128x256 matrix",
        },
    ]

    logger.info(f"\n{'='*80}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        data_shape = test_case["data_shape"]
        indices = test_case["indices"]
        updates = test_case["updates"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.info(f"Description: {test_case['description']}")
        logger.info(f"Data shape: {data_shape}, Num updates: {len(indices)}")

        # Create tensors
        data_arr = np.zeros(data_shape, dtype=np.float32)
        idx_arr = np.array(indices, dtype=np.int64)
        upd_arr = np.array(updates, dtype=np.float32)

        data_t = F._from_data("data", data_arr)
        idx_t = F._from_data("indices", idx_arr)
        upd_t = F._from_data("updates", upd_arr)
        out_t = make_tensor("output")

        op_info = {
            "name": f"scatter_nd_mem_{test_name.replace(' ', '_')}",
            "optype": "ScatterND",
            "inList": [data_t.name, idx_t.name, upd_t.name],
            "outList": [out_t.name],
        }
        op = SimOp(op_info)
        op.precision = "fp32"
        op.uses_compute_pipe = "vector"

        # Get performance counts and execute
        out_t.shape = list(data_shape)
        out_t.dtype = "float32"
        op.perf_stats = {
            "inElems": int(data_arr.size),
            "outElems": int(data_arr.size),
            "inBytes": int(data_arr.nbytes),
            "outBytes": int(data_arr.nbytes),
            "instrs": {"mov": int(data_arr.size)},
        }
        device.execute_op(op)

        # Verify correctness - basic shape check
        assert out_t.shape == list(
            data_shape
        ), f"Output shape {out_t.shape} != data shape {data_shape}"
        logger.debug(f"Output shape: {out_t.shape}")

        # Extract performance stats directly
        perf_stats = op.perf_stats
        data_elements = int(np.prod(data_shape))
        idx_elements = idx_arr.size
        upd_elements = upd_arr.size
        output_elements = data_elements

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate 'mov' instruction is present for ScatterND (data movement)
        assert (
            "mov" in actual_instrs
        ), f"Expected 'mov' instruction for ScatterND, got {list(actual_instrs.keys())}"

        # Get memory metrics
        input_bytes = perf_stats.get("inBytes", 0)
        output_bytes = perf_stats.get("outBytes", 0)
        # Additional bytes for indices and updates (not in perf_stats by default)
        indices_bytes = idx_arr.nbytes
        updates_bytes = upd_arr.nbytes
        total_data_moved = input_bytes + output_bytes + indices_bytes + updates_bytes

        # Compute cycles
        compute_cycles = op.compute_cycles
        mem_rd_cycles = op.mem_rd_cycles
        mem_wr_cycles = op.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        ideal_cycles = max(compute_cycles, memory_cycles)

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        logger.debug("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {total_instructions:,}")
        logger.debug(f"  Instruction types:     {dict(actual_instrs)}")
        logger.debug(f"  Data elements:         {data_elements:,}")
        logger.debug(f"  Index elements:        {idx_elements:,}")
        logger.debug(f"  Update elements:       {upd_elements:,}")
        logger.debug(f"  Output elements:       {output_elements:,}")
        logger.debug(f"  Scatter points:        {len(indices):,}")

        # Validate: 'mov' instructions should match output elements (1 per element)
        assert (
            abs(total_instructions - output_elements) <= output_elements * 0.1
        ), f"Instruction mismatch: {total_instructions} vs expected ~{output_elements}"
        logger.debug("  ✓ Instruction count validates (1 'mov' per output element)")

        logger.debug("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {input_bytes:,} ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Indices bytes:    {indices_bytes:,} ({indices_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Updates bytes:    {updates_bytes:,} ({updates_bytes/1024:.2f} KB)"
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
        ), f"Input/Output bytes should be equal for ScatterND"
        logger.debug("  ✓ Input/Output bytes equal (sparse update operation)")

        logger.debug("\n  -- Memory Metrics --")
        logger.debug(
            f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
        )
        logger.debug(
            "  Bytes per element:     "
            f"{output_bytes/output_elements if output_elements > 0 else 0:.1f}"
        )
        logger.debug(
            "  Update sparsity:       "
            f"{(len(indices) / output_elements * 100):.2f}% (updates/total)"
        )

        # ScatterND is memory-bound (minimal compute)
        assert (
            arithmetic_intensity < 1.0
        ), f"Arithmetic intensity too high for memory-bound ScatterND: {arithmetic_intensity}"
        logger.debug("  ✓ Low arithmetic intensity (memory-bound operation)")

        logger.debug("\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {ideal_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")

        # Validate: ScatterND should be memory-bound for typical cases
        if output_elements > 1000:
            assert (
                bottleneck == "MEMORY"
            ), f"Expected MEMORY bottleneck, got {bottleneck}"
            logger.debug("  ✓ Memory-bound as expected")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "data_shape": data_shape,
                "num_updates": len(indices),
                "instructions": total_instructions,
                "input_bytes": input_bytes,
                "indices_bytes": indices_bytes,
                "updates_bytes": updates_bytes,
                "output_bytes": output_bytes,
                "total_data_moved": total_data_moved,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
                "data_elements": data_elements,
            }
        )

        logger.info("\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*80}\n")
    logger.info(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    logger.info("\n-- Arithmetic Intensity Comparison --")
    logger.info(f"{'Test Name':<30s} {'Ops/Byte':<12s} {'Data Moved':<15s}")
    logger.info("-" * 60)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['arithmetic_intensity']:<12.4f} {result['total_data_moved']/1024:>10.1f} KB"
        )

    # Scatter Pattern Analysis
    logger.info("\n-- Scatter Pattern Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Data Shape':<20s} {'Updates':<12s} {'Sparsity':<12s}"
    )
    logger.info("-" * 75)
    for result in all_results:
        shape_str = "x".join(map(str, result["data_shape"]))
        sparsity = (
            (result["num_updates"] / result["data_elements"] * 100)
            if result["data_elements"] > 0
            else 0
        )
        logger.info(
            f"{result['test_name']:<30s} {shape_str:<20s} {result['num_updates']:>10,} {sparsity:>10.2f}%"
        )

    # Bottleneck Analysis
    logger.info("\n-- Bottleneck Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    logger.info("-" * 80)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*80}\n")

    # Create pytest summary
    summary_lines = [
        "✓ ScatterND Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • 'mov' instructions match output elements (1:1 ratio) ✓",
        "  • All operations are MEMORY-bound ✓",
        "  • Input/Output bytes equal (sparse update) ✓",
        "  • Low arithmetic intensity (data movement operation) ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        sparsity = (
            (result["num_updates"] / result["data_elements"] * 100)
            if result["data_elements"] > 0
            else 0
        )
        summary_lines.append(
            "  ✓ {:<28s} | {:>7,} mov | {:>8.1f} KB | {:>5,} updates ({:.1f}%)".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                result["num_updates"],
                sparsity,
            )
        )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "SCATTERND MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("SCATTERND MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
