#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the ReduceMax op."""

import pytest
import warnings
import numpy as np
import os
import sys
from pathlib import Path

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_reducemax
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
def _make_reducemax_tensors(data, axes=None):
    """Build input tensor list: [X] or [X, axes_tensor]."""
    tensors = [F._from_data("X", data)]
    if axes is not None:
        tensors.append(F._from_data("axes", np.array(axes, dtype=np.int64)))
    return tensors


def _run_reducemax(data, axes=None, keepdims=1, noop=0, tag="reducemax"):
    """Run ReduceMax through SimOp and return (computed, expected, oT)."""
    i_tensors = _make_reducemax_tensors(data, axes)
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "ReduceMax",
        "inList": [t.name for t in i_tensors],
        "outList": ["Y"],
        "attrs": {"keepdims": keepdims, "noop_with_empty_axes": noop},
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)

    computed = compute_reducemax(i_tensors, op)

    # Reference
    if axes is None:
        if noop:
            expected = data.copy()
        else:
            expected = np.max(data, axis=None, keepdims=bool(keepdims))
    else:
        expected = np.max(
            data, axis=tuple(int(a) for a in axes), keepdims=bool(keepdims)
        )

    return computed, expected, o_tensors[0]


# ===================================================================
# 1. Shape validation tests
# ===================================================================
shape_test_cases = [
    # (data_shape, axes, keepdims, noop, expected_shape, id)
    ([8], None, 1, 0, [1], "1d_all_kd"),
    ([3, 4], None, 1, 0, [1, 1], "2d_all_kd"),
    ([2, 3, 4], None, 1, 0, [1, 1, 1], "3d_all_kd"),
    ([3, 4], None, 0, 0, [], "2d_all_nokd"),
    ([2, 3, 4], None, 0, 0, [], "3d_all_nokd"),
    ([3, 4], None, 1, 1, [3, 4], "2d_noop"),
    ([3, 4], [0], 1, 0, [1, 4], "2d_ax0_kd"),
    ([3, 4], [1], 1, 0, [3, 1], "2d_ax1_kd"),
    ([3, 4], [0], 0, 0, [4], "2d_ax0_nokd"),
    ([2, 3, 4], [0], 1, 0, [1, 3, 4], "3d_ax0_kd"),
    ([2, 3, 4], [1], 1, 0, [2, 1, 4], "3d_ax1_kd"),
    ([2, 3, 4], [2], 1, 0, [2, 3, 1], "3d_ax2_kd"),
    ([2, 3, 4], [2], 0, 0, [2, 3], "3d_ax2_nokd"),
    ([2, 3, 4], [0, 2], 1, 0, [1, 3, 1], "3d_ax02_kd"),
    ([2, 3, 4], [0, 2], 0, 0, [3], "3d_ax02_nokd"),
    ([2, 3, 4, 5], [2, 3], 1, 0, [2, 3, 1, 1], "4d_ax23_kd"),
    ([2, 3, 4], [-1], 1, 0, [2, 3, 1], "3d_neg_ax_kd"),
    ([2, 3, 4], [-1, -2], 1, 0, [2, 1, 1], "3d_neg_multi_kd"),
]


class TestReduceMaxShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize(
        "shape,axes,keepdims,noop,expected_shape,tid", shape_test_cases
    )
    def test_output_shape(self, shape, axes, keepdims, noop, expected_shape, tid):
        data = np.random.randn(*shape).astype(np.float32)
        _, _, oT = _run_reducemax(data, axes, keepdims, noop, tag=f"shape_{tid}")
        assert (
            list(oT.shape) == expected_shape
        ), f"{tid}: {oT.shape} != {expected_shape}"


# ===================================================================
# 2. Numerical validation tests
# ===================================================================
numerical_cases = [
    # (data_shape, axes, keepdims, noop, id)
    ([8], None, 1, 0, "1d_all"),
    ([3, 4], None, 1, 0, "2d_all"),
    ([2, 3, 4], None, 1, 0, "3d_all"),
    ([2, 3, 4, 4], None, 1, 0, "4d_all"),
    ([3, 4], None, 0, 0, "2d_all_nokd"),
    ([3, 4], None, 1, 1, "2d_noop"),
    ([3, 4], [0], 1, 0, "2d_ax0"),
    ([3, 4], [1], 1, 0, "2d_ax1"),
    ([2, 3, 4], [0], 1, 0, "3d_ax0"),
    ([2, 3, 4], [1], 1, 0, "3d_ax1"),
    ([2, 3, 4], [2], 1, 0, "3d_ax2"),
    ([2, 3, 4], [2], 0, 0, "3d_ax2_nokd"),
    ([2, 3, 4], [0, 2], 1, 0, "3d_ax02"),
    ([2, 3, 4, 4], [2, 3], 1, 0, "4d_ax23"),
    ([2, 3, 4, 4], [2, 3], 0, 0, "4d_ax23_nokd"),
    ([2, 3, 4], [-1], 1, 0, "3d_neg"),
    ([2, 3, 4], [-1, -2], 1, 0, "3d_neg_multi"),
    ([2, 2, 3, 4, 4], [3, 4], 1, 0, "5d_ax34"),
    ([1], [0], 1, 0, "single"),
    ([1, 1, 1], [0, 1, 2], 1, 0, "ones_shape"),
]


class TestReduceMaxNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape,axes,keepdims,noop,tid", numerical_cases)
    def test_values(self, shape, axes, keepdims, noop, tid):
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float32)
        computed, expected, _ = _run_reducemax(
            data, axes, keepdims, noop, tag=f"num_{tid}"
        )
        np.testing.assert_array_equal(computed, expected, err_msg=f"{tid} mismatch")


# ===================================================================
# 3. Edge-case tests
# ===================================================================
class TestReduceMaxEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_empty_tensor(self):
        """ReduceMax over empty dim."""
        data = np.array([], dtype=np.float32).reshape(0, 4)
        i_tensors = _make_reducemax_tensors(data, [0])
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": "empty",
            "optype": "ReduceMax",
            "inList": [t.name for t in i_tensors],
            "outList": ["Y"],
            "attrs": {"keepdims": 1},
        }
        op = SimOp(op_info)
        for t in i_tensors:
            t.op_in = ["empty"]
        for t in o_tensors:
            t.op_out = ["empty"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                op.get_perf_counts(i_tensors, o_tensors)
                compute_reducemax(i_tensors, op)
        except (ValueError, AssertionError, IndexError):
            pass  # acceptable

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        """ReduceMax over a large tensor."""
        data = np.random.randn(2, 16, 32, 32).astype(np.float32)
        computed, expected, _ = _run_reducemax(data, [2, 3], 1, 0, tag="large")
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64(self):
        """Works with float64."""
        data = np.random.randn(3, 4).astype(np.float64)
        computed, expected, _ = _run_reducemax(data, [1], 1, 0, tag="f64")
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_int_input(self):
        """Works with integer input."""
        data = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int32)
        computed, expected, _ = _run_reducemax(data, [1], 1, 0, tag="int")
        np.testing.assert_array_equal(computed, expected)
        assert computed[0, 0] == 5
        assert computed[1, 0] == 6

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_inf(self):
        """Inf is selected as max."""
        data = np.array([[np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
        computed, expected, _ = _run_reducemax(data, [0], 1, 0, tag="inf")
        assert np.isinf(computed[0, 0])
        assert computed[0, 0] == np.inf

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_neg_inf(self):
        """Neg inf is not selected when other values exist."""
        data = np.array([[-np.inf, 1.0], [2.0, 3.0]], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [0], 1, 0, tag="neginf")
        assert computed[0, 0] == 2.0
        assert computed[0, 1] == 3.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_nan(self):
        """NaN propagation in max."""
        data = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
        computed, expected, _ = _run_reducemax(data, [0], 1, 0, tag="nan")
        # np.max propagates NaN
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_all_same(self):
        """Max of identical values = that value."""
        data = np.full((3, 4), 7.0, dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [1], 1, 0, tag="same")
        np.testing.assert_array_equal(computed, np.full((3, 1), 7.0, dtype=np.float32))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_values(self):
        """Max of all-negative picks the least negative."""
        data = np.array([[-5.0, -2.0, -8.0]], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [1], 1, 0, tag="negv")
        assert float(computed[0, 0]) == -2.0


# ===================================================================
# 4. Precision tests with known values
# ===================================================================
class TestReduceMaxPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_1d_max(self):
        data = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, None, 1, 0, tag="p_1d")
        assert computed.item() == 4.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_1d_max_no_keepdims(self):
        data = np.array([1.0, 4.0, 2.0, 3.0], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, None, 0, 0, tag="p_1d_nokd")
        assert float(computed) == 4.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_2d_axis0(self):
        data = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [0], 1, 0, tag="p_2d_ax0")
        np.testing.assert_array_equal(computed, np.array([[3.0, 5.0]]))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_2d_axis1(self):
        data = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [1], 1, 0, tag="p_2d_ax1")
        np.testing.assert_array_equal(computed, np.array([[5.0], [3.0]]))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_noop(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, None, 1, 1, tag="p_noop")
        np.testing.assert_array_equal(computed, data)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_3d_max_axis1(self):
        data = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[8.0, 6.0], [7.0, 5.0]]], dtype=np.float32
        )
        computed, _, _ = _run_reducemax(data, [1], 1, 0, tag="p_3d_ax1")
        np.testing.assert_array_equal(computed, np.array([[[3.0, 4.0]], [[8.0, 6.0]]]))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_element(self):
        data = np.array([42.0], dtype=np.float32)
        computed, _, _ = _run_reducemax(data, [0], 1, 0, tag="p_single")
        assert computed.item() == 42.0

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_sequential(self):
        """Max of arange(12) reshaped (3,4) along axis 1 → [3,6,9,11] doesn't apply; check."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        computed, _, _ = _run_reducemax(data, [1], 1, 0, tag="p_seq")
        np.testing.assert_array_equal(computed, np.array([[3.0], [7.0], [11.0]]))


# ===================================================================
# 5. Mathematical property tests
# ===================================================================
class TestReduceMaxProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_geq_all_elements(self):
        """max(X, axis) >= all elements along that axis."""
        np.random.seed(42)
        data = np.random.randn(3, 4, 5).astype(np.float32)
        computed, _, _ = _run_reducemax(data, [2], 1, 0, tag="prop_geq")
        # computed shape: (3, 4, 1), data shape: (3, 4, 5)
        assert np.all(computed >= data)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_is_element(self):
        """max(X, axis) is an actual element from X along that axis."""
        np.random.seed(43)
        data = np.random.randn(3, 5).astype(np.float32)
        computed, _, _ = _run_reducemax(data, [1], 0, 0, tag="prop_elem")
        for i in range(3):
            assert computed[i] in data[i], f"max {computed[i]} not found in row {i}"

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_of_constant(self):
        """Max of constant tensor = that constant."""
        val = -3.14
        data = np.full((4, 5), val, dtype=np.float32)
        computed, _, _ = _run_reducemax(data, None, 0, 0, tag="prop_const")
        assert float(computed) == pytest.approx(val)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_keepdims_consistency(self):
        """keepdims=0 and keepdims=1 give same values after squeeze."""
        np.random.seed(44)
        data = np.random.randn(2, 3, 4).astype(np.float32)
        c_kd, _, _ = _run_reducemax(data, [1], 1, 0, tag="prop_kd1")
        c_nokd, _, _ = _run_reducemax(data, [1], 0, 0, tag="prop_kd0")
        np.testing.assert_array_equal(c_kd.squeeze(axis=1), c_nokd)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_monotonic_with_offset(self):
        """max(X + c, axis) = max(X, axis) + c for scalar c."""
        np.random.seed(45)
        data = np.random.randn(3, 4).astype(np.float32)
        c = 5.0
        max_x, _, _ = _run_reducemax(data, [1], 1, 0, tag="prop_mono_x")
        max_xc, _, _ = _run_reducemax(data + c, [1], 1, 0, tag="prop_mono_xc")
        np.testing.assert_allclose(max_xc, max_x + c, rtol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_max_geq_mean(self):
        """max >= mean along any axis."""
        np.random.seed(46)
        data = np.random.randn(4, 6).astype(np.float32)
        max_val, _, _ = _run_reducemax(data, [1], 1, 0, tag="prop_geq_mean")
        mean_val = np.mean(data, axis=1, keepdims=True)
        assert np.all(max_val >= mean_val - 1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_noop_identity(self):
        """noop_with_empty_axes=1, axes=None → output == input."""
        data = np.random.randn(3, 4, 5).astype(np.float32)
        computed, _, _ = _run_reducemax(data, None, 1, 1, tag="prop_noop")
        np.testing.assert_array_equal(computed, data)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_idempotent(self):
        """max(max(X, axis_a), axis_a_reduced) still matches (no-op on size-1 dim)."""
        np.random.seed(47)
        data = np.random.randn(3, 4, 5).astype(np.float32)
        first_max, _, _ = _run_reducemax(data, [2], 1, 0, tag="prop_idem1")
        # first_max shape: (3, 4, 1) — reducing axis 2 again should be no-op
        second_max, _, _ = _run_reducemax(first_max, [2], 1, 0, tag="prop_idem2")
        np.testing.assert_array_equal(first_max, second_max)


# ===================================================================
# 6. Memory and Performance Validation
# ===================================================================


def normalize_precision(dtype_str):
    """Normalize tensor dtype to device-compatible precision string."""
    if dtype_str in ["torch.float16", "float16"]:
        return "fp16"
    elif dtype_str in ["torch.bfloat16", "bfloat16"]:
        return "bf16"
    elif dtype_str in ["torch.float32", "float32"]:
        return "fp32"
    elif dtype_str in ["torch.int8", "int8"]:
        return "int8"
    elif dtype_str in ["torch.int32", "int32"]:
        return "int32"
    return dtype_str


def calculate_reducemax_memory_stats(input_shape, axes, keepdims, precision="fp32"):
    """
    Calculate expected memory statistics for ReduceMax operation.

    ReduceMax uses 'cmp' instruction: 1 comparison per input element.
    Reduction operations compare all input elements to find maximum along axes.

    Args:
        input_shape: Shape of input tensor
        axes: Axes to reduce over (None reduces all)
        keepdims: Whether to keep reduced dimensions
        precision: Data type precision (fp32, fp16, bf16, etc.)

    Returns:
        Dictionary with expected memory statistics
    """
    # Calculate element counts
    input_elements = np.prod(input_shape)

    # Calculate output shape
    if axes is None:
        output_shape = [1] * len(input_shape) if keepdims else []
    else:
        output_shape = list(input_shape)
        for ax in sorted(axes, reverse=True):
            if ax < 0:
                ax = len(input_shape) + ax
            if keepdims:
                output_shape[ax] = 1
            else:
                output_shape.pop(ax)

    output_elements = np.prod(output_shape) if output_shape else 1

    # Calculate bytes (assuming fp32 for now, adjust if needed)
    bytes_per_elem = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int32": 4}.get(
        precision, 4
    )
    input_bytes = input_elements * bytes_per_elem
    output_bytes = output_elements * bytes_per_elem

    # ReduceMax uses 'cmp' instruction: 1 compare per input element
    # All input elements participate in finding the maximum along reduction axes
    cmp_count = input_elements

    # Calculate arithmetic intensity (ops/byte)
    total_bytes = input_bytes + output_bytes
    arithmetic_intensity = cmp_count / total_bytes if total_bytes > 0 else 0

    # Calculate reduction ratio
    reduction_ratio = input_elements / output_elements if output_elements > 0 else 0

    return {
        "input_elements": input_elements,
        "output_elements": output_elements,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "cmp_instructions": cmp_count,
        "total_instructions": cmp_count,
        "arithmetic_intensity": arithmetic_intensity,
        "reduction_ratio": reduction_ratio,
    }


memory_validation_cases = [
    # (input_shape, axes, keepdims, description)
    ([128], None, 1, "1D reduce all"),
    ([32, 64], [0], 1, "2D reduce axis 0"),
    ([32, 64], [1], 1, "2D reduce axis 1"),
    ([8, 16, 16], [1, 2], 1, "3D reduce spatial"),
    ([4, 8, 8, 8], [1, 2, 3], 1, "4D reduce channels"),
]


class TestReduceMaxMemoryValidation:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.performance
    def test_reducemax_memory_validation(self, capsys, request):
        """
        Validate memory usage and instruction counts for ReduceMax operations.

        This test validates two primary metrics:
        1. Instructions Executed: Verifies instruction count (1 'cmp' per input element)
        2. Data Moved: Tracks input/output bytes and validates memory traffic

        ReduceMax finds maximum values along specified axes using compare operations.
        Run with: pytest tests/test_ops/test_reducemax.py::TestReduceMaxMemoryValidation -v
        For detailed output: add -s flag
        """
        print("\n" + "=" * 60)
        print("ReduceMax Operation Memory Validation")
        print("=" * 60)

        # Load device configuration
        config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
        try:
            ipgroups, packages = get_arspec_from_yaml(config_path)
            device_pkg = packages["n150"]  # Use Wormhole n150 device
            device = Device(device_pkg)

            print(f"\nDevice: {device.devname} ({device.name})")
            print(f"Device frequency: {device.freq_MHz} MHz")
            print(f"Memory frequency: {device.memfreq_MHz} MHz")
            print(
                f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
            )
        except Exception as e:
            print(f"\nWarning: Could not load device config: {e}")
            print("Skipping memory validation test")
            pytest.skip(f"Could not load device config: {e}")
            return

        print(f"\n{'='*60}")
        print("Running Memory Validation Tests")
        print(f"{'='*60}\n")

        all_results = []

        for input_shape, axes, keepdims, description in memory_validation_cases:
            print(f"\n-- Test: {description} --")
            print(f"Input shape: {input_shape}")
            print(f"Axes: {axes}, KeepDims: {keepdims}")

            # Create test data
            np.random.seed(42)
            data = np.random.randn(*input_shape).astype(np.float32)

            # Calculate expected statistics
            expected_stats = calculate_reducemax_memory_stats(
                input_shape, axes, keepdims
            )

            # Build input tensors
            i_tensors = _make_reducemax_tensors(data, axes)
            o_tensors = [make_tensor("Y")]

            # Create and configure op
            op_info = {
                "name": "reducemax_mem_val",
                "optype": "ReduceMax",
                "inList": [t.name for t in i_tensors],
                "outList": ["Y"],
                "attrs": {"keepdims": keepdims, "noop_with_empty_axes": 0},
            }
            op = SimOp(op_info)

            for t in i_tensors:
                t.op_in = ["reducemax_mem_val"]
            for t in o_tensors:
                t.op_out = ["reducemax_mem_val"]

            # Get performance counts
            op.get_perf_counts(i_tensors, o_tensors)

            # Validate compute function
            computed = compute_reducemax(i_tensors, op)
            if axes is None:
                expected = np.max(data, axis=None, keepdims=bool(keepdims))
            else:
                expected = np.max(
                    data, axis=tuple(int(a) for a in axes), keepdims=bool(keepdims)
                )
            np.testing.assert_allclose(
                computed,
                expected,
                rtol=1e-5,
                err_msg="ReduceMax compute function mismatch",
            )

            # Extract stats from op.perf_stats
            perf_stats = op.perf_stats
            actual_in_elems = perf_stats["inElems"]
            actual_out_elems = perf_stats["outElems"]
            actual_in_bytes = perf_stats["inBytes"]
            actual_out_bytes = perf_stats["outBytes"]
            actual_instrs = perf_stats["instrs"]

            # Account for axes tensor in input element count if axes were provided
            expected_in_elems = expected_stats["input_elements"]
            if axes is not None:
                # axes tensor adds its element count to inElems
                axes_count = len(axes) if isinstance(axes, list) else 1
                expected_in_elems += axes_count

            # Account for axes tensor bytes if provided
            expected_in_bytes = expected_stats["input_bytes"]
            if axes is not None:
                # axes tensor is int64 (8 bytes per element)
                axes_count = len(axes) if isinstance(axes, list) else 1
                expected_in_bytes += axes_count * 8

            print(f"\n  -- Instructions & Operations --")
            print(f"  Instructions executed: {actual_instrs.get('cmp', 0):,} (cmp)")
            print(f"  Input elements:        {expected_stats['input_elements']:,}")
            print(f"  Output elements:       {expected_stats['output_elements']:,}")
            print(
                f"  Expected instructions: ~{expected_stats['cmp_instructions']:,} (1 cmp per input element)"
            )

            # Validate instructions
            assert "cmp" in actual_instrs, "Expected 'cmp' instruction in perf_stats"
            actual_cmp = actual_instrs["cmp"]
            expected_cmp = expected_stats["cmp_instructions"]

            assert (
                actual_cmp == expected_cmp
            ), f"Compare instruction count mismatch: {actual_cmp} vs {expected_cmp}"

            instruction_ratio = (
                actual_cmp / expected_stats["input_elements"]
                if expected_stats["input_elements"] > 0
                else 0
            )
            print(
                f"  Instruction ratio:     {instruction_ratio:.2f} (✓ 1 cmp per input)"
            )

            print(f"\n  -- Data Movement --")
            print(
                f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
            )
            print(
                f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
            )
            print(
                f"  Total data moved: {actual_in_bytes + actual_out_bytes:,} bytes ({(actual_in_bytes + actual_out_bytes)/1024:.2f} KB)"
            )
            print(
                f"  Reduction ratio:  {expected_stats['reduction_ratio']:.1f}x ({actual_in_elems} → {actual_out_elems})"
            )

            # Validate element counts
            assert (
                actual_in_elems == expected_in_elems
            ), f"Input element count mismatch: {actual_in_elems} vs {expected_in_elems}"
            assert (
                actual_out_elems == expected_stats["output_elements"]
            ), f"Output element count mismatch: {actual_out_elems} vs {expected_stats['output_elements']}"

            # Validate data movement
            assert (
                actual_in_bytes == expected_in_bytes
            ), f"Input bytes mismatch: {actual_in_bytes} vs {expected_in_bytes}"
            assert (
                actual_out_bytes == expected_stats["output_bytes"]
            ), f"Output bytes mismatch: {actual_out_bytes} vs {expected_stats['output_bytes']}"

            print(
                f"  Expected input:   {expected_in_bytes:,} bytes (✓ matches fp32 + axes)"
            )
            print(
                f"  Expected output:  {expected_stats['output_bytes']:,} bytes (✓ matches fp32)"
            )

            print(f"\n  -- Memory Metrics --")
            print(
                f"  Arithmetic intensity:  {expected_stats['arithmetic_intensity']:.4f} ops/byte"
            )

            # For reductions, arithmetic intensity varies based on reduction ratio
            assert (
                expected_stats["arithmetic_intensity"] < 2.0
            ), f"Arithmetic intensity too high: {expected_stats['arithmetic_intensity']}"
            print(f"  ✓ Arithmetic intensity within expected range for reduction")

            # Execute on device to get cycle estimates
            op.precision = "fp32"
            if op.uses_compute_pipe is None:
                op.uses_compute_pipe = "vector"

            device.execute_op(op)

            # Get cycle breakdown
            compute_cycles = op.compute_cycles if hasattr(op, "compute_cycles") else 0
            mem_rd_cycles = op.mem_rd_cycles if hasattr(op, "mem_rd_cycles") else 0
            mem_wr_cycles = op.mem_wr_cycles if hasattr(op, "mem_wr_cycles") else 0
            memory_cycles = mem_rd_cycles + mem_wr_cycles
            ideal_cycles = max(compute_cycles, memory_cycles)

            # Determine bottleneck
            if compute_cycles >= memory_cycles:
                bottleneck = "COMPUTE"
            else:
                bottleneck = "MEMORY"

            print(f"\n  -- Execution Cycles --")
            print(f"  Compute cycles:   {compute_cycles:,}")
            print(f"  Memory cycles:    {memory_cycles:,}")
            print(f"    Read cycles:    {mem_rd_cycles:,}")
            print(f"    Write cycles:   {mem_wr_cycles:,}")
            print(f"  Ideal cycles:     {ideal_cycles:,}")
            print(f"  Bottleneck:       {bottleneck}")

            # Validate: reductions can be compute or memory bound depending on size
            if (
                expected_stats["input_elements"] > 1000
            ):  # Check for reasonably sized tensors
                # Large reductions are typically more balanced or compute-bound
                print(f"  ✓ Bottleneck analysis: {bottleneck} for reduction operation")

            # Store results
            all_results.append(
                {
                    "test_name": description,
                    "input_shape": input_shape,
                    "axes": axes,
                    "keepdims": keepdims,
                    "output_shape": list(computed.shape),
                    "cmp_instructions": actual_cmp,
                    "total_data_moved": actual_in_bytes + actual_out_bytes,
                    "arithmetic_intensity": expected_stats["arithmetic_intensity"],
                    "reduction_ratio": expected_stats["reduction_ratio"],
                    "bottleneck": bottleneck,
                    "compute_cycles": compute_cycles,
                    "memory_cycles": memory_cycles,
                    "ideal_cycles": ideal_cycles,
                }
            )

            print(f"\n  ✓ Test PASSED")

        # Summary
        print(f"\n{'='*60}")
        print("Memory Validation Summary")
        print(f"{'='*60}\n")
        print(f"Total tests run: {len(all_results)}")
        print(f"All tests passed: ✓")

        # Compare arithmetic intensity across tests
        print(f"\n-- Arithmetic Intensity Comparison --")
        for result in all_results:
            ai = result["arithmetic_intensity"]
            print(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

        # Compare reduction ratios
        print(f"\n-- Reduction Ratio Comparison --")
        for result in all_results:
            ratio = result["reduction_ratio"]
            print(f"{result['test_name']:30s}: {ratio:.1f}x")

        print(f"\n-- Bottleneck Analysis --")
        for result in all_results:
            bottleneck = result["bottleneck"]
            print(f"{result['test_name']:30s}: {bottleneck}")

        print(f"\n{'='*60}")
        print("Memory validation complete!")
        print(f"{'='*60}\n")

        # Create a summary that will be displayed in pytest output (even without -s flag)
        summary_lines = [
            "✓ Tests completed: {}/{} - All PASSED".format(
                len(all_results), len(memory_validation_cases)
            ),
            "",
            "Key Findings:",
            "  • Instructions: 1 'cmp' per input element ✓",
            "  • Reduction operations show higher arithmetic intensity",
            "  • Bottleneck: Mix of COMPUTE and MEMORY based on reduction size",
            "",
            "Test Results:",
        ]

        for result in all_results:
            summary_lines.append(
                "  ✓ {:<26s} | {:>7,} cmp | {:>7.1f} KB | {:.3f} ops/byte | {:.0f}x reduce".format(
                    result["test_name"],
                    result["cmp_instructions"],
                    result["total_data_moved"] / 1024,
                    result["arithmetic_intensity"],
                    result["reduction_ratio"],
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
                print("\n" + "=" * 70)
                print("MEMORY VALIDATION RESULTS")
                print("=" * 70)
                for line in summary_lines:
                    print(line)
                print("=" * 70 + "\n")

        # Final assertion
        assert len(all_results) == len(
            memory_validation_cases
        ), f"Memory validation: {len(all_results)}/{len(memory_validation_cases)} tests passed"
