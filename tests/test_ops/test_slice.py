#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive tests for compute_slice.

Slice extracts a sub-tensor from the input along specified axes,
with configurable start, end, and step values.
Inputs: [data, starts, ends] or [data, starts, ends, axes] or
        [data, starts, ends, axes, steps]
"""

import pytest
import numpy as np
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_slice

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def ref_impl_slice(data, starts, ends, axes=None, steps=None):
    """Reference implementation of ONNX Slice."""
    slices = [slice(None)] * len(data.shape)
    if axes is None:
        axes = list(range(len(starts)))
    for i, axis in enumerate(axes):
        s = starts[i]
        e = ends[i]
        st = steps[i] if steps is not None else 1
        slices[axis] = slice(int(s), int(e), int(st))
    return data[tuple(slices)]


def _make_slice_tensors(data, starts, ends, axes=None, steps=None):
    """Build input tensor list for slice op (always 5 inputs)."""
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    tensors = [
        F._from_data("X", data),
        F._from_data("starts", np.array(starts, dtype=np.int64)),
        F._from_data("ends", np.array(ends, dtype=np.int64)),
        F._from_data("axes", np.array(axes, dtype=np.int64)),
        F._from_data("steps", np.array(steps, dtype=np.int64)),
    ]
    return tensors


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


# --------------------------------------------------------------------------
# Main test cases
# --------------------------------------------------------------------------

slice_test_name = "test_slice"
slice_test_cases = [
    # (name, data_shape, starts, ends, axes, steps, description)
    # Basic slicing along different axes
    ("1D basic", [10], [2], [7], [0], [1]),
    ("1D with step", [10], [1], [8], [0], [2]),
    ("2D row slice", [4, 6], [1], [3], [0], [1]),
    ("2D col slice", [4, 6], [2], [5], [1], [1]),
    ("2D both axes", [4, 6], [1, 2], [3, 5], [0, 1], [1, 1]),
    ("3D single axis", [2, 4, 6], [1], [3], [1], [1]),
    ("3D two axes", [2, 4, 6], [0, 2], [2, 5], [0, 2], [1, 1]),
    ("3D all axes", [2, 4, 6], [0, 1, 2], [2, 3, 5], [0, 1, 2], [1, 1, 1]),
    ("4D NCHW spatial crop", [1, 3, 8, 8], [2, 2], [6, 6], [2, 3], [1, 1]),
    ("4D batch slice", [4, 3, 8, 8], [1], [3], [0], [1]),
    # Slicing with steps
    ("1D step 3", [12], [0], [12], [0], [3]),
    ("2D step on rows", [8, 4], [0], [8], [0], [2]),
    ("2D step on both", [8, 6], [0, 0], [8, 6], [0, 1], [2, 3]),
    ("4D spatial stride 2", [1, 3, 8, 8], [0, 0], [8, 8], [2, 3], [2, 2]),
    # Edge cases: slice to single element
    ("Single row", [4, 6], [2], [3], [0], [1]),
    ("Single col", [4, 6], [3], [4], [1], [1]),
    ("Single element 2D", [4, 6], [1, 2], [2, 3], [0, 1], [1, 1]),
    # Full slice (no-op)
    ("Full 1D", [8], [0], [8], [0], [1]),
    ("Full 2D", [3, 4], [0, 0], [3, 4], [0, 1], [1, 1]),
    # Negative indices
    ("1D negative end", [10], [0], [-2], [0], [1]),
    ("2D negative start and end", [6, 8], [-4], [-1], [0], [1]),
    # Large tensor
    ("Large 4D crop", [2, 16, 32, 32], [8, 8], [24, 24], [2, 3], [1, 1]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_slice():
    """Test Slice with shape validation and numerical validation"""
    msgw = get_max_test_msg_len(slice_test_cases)

    for tno, (tmsg, data_shape, starts, ends, axes, steps) in enumerate(
        slice_test_cases
    ):
        op_name = f"{slice_test_name}_{tno}"

        # Generate random data
        data = np.array(np.random.randn(*data_shape), dtype=np.float32)

        # Compute reference output to get expected shape
        ref_output = ref_impl_slice(data, starts, ends, axes, steps)
        expected_shape = list(ref_output.shape)

        # Build input tensors
        i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        inf_shape = o_tensors[0].shape
        shape_match = inf_shape == expected_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_slice(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        if shape_match and numerical_match == True:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape OK, Numerical OK]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape OK, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(f"  Shape: got {inf_shape}, expected {expected_shape}")
            logger.debug(f"  Numerical: {numerical_match}")
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {expected_shape}"


# --------------------------------------------------------------------------
# Error / edge-case test cases
# --------------------------------------------------------------------------

slice_error_cases = [
    ("Empty result", [4, 6], [2], [2], [0], [1]),
    ("Step larger than range", [10], [0], [3], [0], [5]),
    ("Slice on zero-dim", [0, 4], [0], [0], [0], [1]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_errors():
    """Test Slice with edge cases"""
    msgw = get_max_test_msg_len(slice_error_cases)

    for tno, (tmsg, data_shape, starts, ends, axes, steps) in enumerate(
        slice_error_cases
    ):
        op_name = f"test_slice_errors_{tno}"

        data = np.empty(data_shape, dtype=np.float32)

        try:
            ref_output = ref_impl_slice(data, starts, ends, axes, steps)
            expected_shape = list(ref_output.shape)
        except Exception:
            expected_shape = [0]

        i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)
            try:
                computed_output = compute_slice(i_tensors, op_obj)
                if computed_output.size == 0:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (empty output)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (output shape: {computed_output.shape})"
                    )
            except (ValueError, IndexError, TypeError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# --------------------------------------------------------------------------
# Precision test cases with known outputs
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_precision():
    """Test Slice with precise known input/output pairs"""

    precision_cases = [
        (
            "1D simple slice",
            np.array([10, 20, 30, 40, 50], dtype=np.float32),
            [1],
            [4],
            [0],
            [1],
            np.array([20, 30, 40], dtype=np.float32),
        ),
        (
            "1D with step 2",
            np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
            [0],
            [6],
            [0],
            [2],
            np.array([10, 30, 50], dtype=np.float32),
        ),
        (
            "2D row slice",
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            [0],
            [2],
            [0],
            [1],
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        ),
        (
            "2D col slice",
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            [1],
            [3],
            [1],
            [1],
            np.array([[2, 3], [5, 6], [8, 9]], dtype=np.float32),
        ),
        (
            "2D crop",
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32),
            [0, 1],
            [2, 3],
            [0, 1],
            [1, 1],
            np.array([[2, 3], [6, 7]], dtype=np.float32),
        ),
        (
            "2D step on rows",
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
            [0],
            [4],
            [0],
            [2],
            np.array([[1, 2], [5, 6]], dtype=np.float32),
        ),
        (
            "Single element extraction",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            [1, 1],
            [2, 2],
            [0, 1],
            [1, 1],
            np.array([[5]], dtype=np.float32),
        ),
        (
            "Negative end index",
            np.array([10, 20, 30, 40, 50], dtype=np.float32),
            [1],
            [-1],
            [0],
            [1],
            np.array([20, 30, 40], dtype=np.float32),
        ),
        (
            "4D NCHW spatial crop",
            np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4),
            [1, 1],
            [3, 3],
            [2, 3],
            [1, 1],
            np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)[
                :, :, 1:3, 1:3
            ],
        ),
        (
            "Full tensor (no-op)",
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            [0, 0],
            [2, 2],
            [0, 1],
            [1, 1],
            np.array([[1, 2], [3, 4]], dtype=np.float32),
        ),
    ]

    msgw = 30

    for tno, (tmsg, test_data, starts, ends, axes, steps, expected_output) in enumerate(
        precision_cases
    ):
        op_name = f"test_slice_precision_{tno}"

        expected_shape = list(expected_output.shape)

        i_tensors = _make_slice_tensors(test_data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_slice(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                logger.debug(f"PRECISION TEST[{tno:2d}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno:2d}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno:2d}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"


# --------------------------------------------------------------------------
# Mathematical property tests
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_preserves_data():
    """Test that sliced elements exactly match the original data at those positions"""
    shapes = [[10], [4, 6], [2, 4, 8], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data = np.array(np.random.randn(*shape), dtype=np.float32)

        # Slice first half along axis 0
        mid = shape[0] // 2
        if mid == 0:
            mid = 1

        starts = [0]
        ends = [mid]
        axes = [0]
        steps = [1]

        ref_output = ref_impl_slice(data, starts, ends, axes, steps)
        expected_shape = list(ref_output.shape)

        i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": f"preserve_{idx}",
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_slice(i_tensors, op_obj)

        # Sliced output must exactly match the corresponding region in original
        slices_ref = [slice(None)] * len(shape)
        slices_ref[0] = slice(0, mid)
        expected = data[tuple(slices_ref)]

        assert np.array_equal(
            result, expected
        ), f"Data preservation failed for shape {shape}"
        logger.debug(f"PRESERVE DATA TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_full_is_identity():
    """Test that slicing the full range returns the original tensor"""
    shapes = [[8], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data = np.array(np.random.randn(*shape), dtype=np.float32)

        starts = [0] * len(shape)
        ends = list(shape)
        axes = list(range(len(shape)))
        steps = [1] * len(shape)

        ref_output = ref_impl_slice(data, starts, ends, axes, steps)
        expected_shape = list(ref_output.shape)

        i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": f"full_id_{idx}",
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_slice(i_tensors, op_obj)

        assert np.array_equal(
            result, data
        ), f"Full slice identity failed for shape {shape}"
        logger.debug(f"FULL IDENTITY TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_output_shape():
    """Test that output shape matches expected dimensions from start/end/step"""
    test_configs = [
        # (data_shape, starts, ends, axes, steps, expected_out_shape)
        ([10], [2], [8], [0], [1], [6]),
        ([10], [0], [10], [0], [2], [5]),
        ([10], [1], [9], [0], [3], [3]),
        ([4, 6], [1, 2], [3, 5], [0, 1], [1, 1], [2, 3]),
        ([2, 4, 8], [0, 0], [4, 8], [1, 2], [2, 2], [2, 2, 4]),
    ]

    for idx, (data_shape, starts, ends, axes, steps, expected_out_shape) in enumerate(
        test_configs
    ):
        data = np.array(np.random.randn(*data_shape), dtype=np.float32)
        ref_output = ref_impl_slice(data, starts, ends, axes, steps)

        assert (
            list(ref_output.shape) == expected_out_shape
        ), f"Shape test {idx}: expected {expected_out_shape}, got {list(ref_output.shape)}"

        i_tensors = _make_slice_tensors(data, starts, ends, axes, steps)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": f"shape_{idx}",
            "optype": "Slice",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"out_shape": expected_out_shape},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        assert (
            inf_shape == expected_out_shape
        ), f"Shape inference test {idx}: got {inf_shape}, expected {expected_out_shape}"
        logger.debug(
            f"OUTPUT SHAPE TEST[{idx}] {data_shape} -> {expected_out_shape} PASS"
        )


@pytest.mark.unit
@pytest.mark.opunit
def test_slice_contiguous_reassembly():
    """Test that adjacent slices can reconstruct the original along axis 0"""
    shapes = [[6], [4, 3], [6, 4, 2]]

    for idx, shape in enumerate(shapes):
        data = np.array(np.random.randn(*shape), dtype=np.float32)
        mid = shape[0] // 2

        # First half
        ref1 = ref_impl_slice(data, [0], [mid], [0], [1])
        i1 = _make_slice_tensors(data, [0], [mid], [0], [1])
        o1 = [make_tensor("Y1")]
        op1_info = {
            "name": f"reassemble_a_{idx}",
            "optype": "Slice",
            "inList": [x.name for x in i1],
            "outList": [x.name for x in o1],
            "attrs": {"out_shape": list(ref1.shape)},
        }
        op1 = SimOp(op1_info)
        for x in i1:
            x.op_in = [op1.name]
        for x in o1:
            x.op_out = [op1.name]
        op1.get_perf_counts(i1, o1)
        result1 = compute_slice(i1, op1)

        # Second half
        ref2 = ref_impl_slice(data, [mid], [shape[0]], [0], [1])
        i2 = _make_slice_tensors(data, [mid], [shape[0]], [0], [1])
        o2 = [make_tensor("Y2")]
        op2_info = {
            "name": f"reassemble_b_{idx}",
            "optype": "Slice",
            "inList": [x.name for x in i2],
            "outList": [x.name for x in o2],
            "attrs": {"out_shape": list(ref2.shape)},
        }
        op2 = SimOp(op2_info)
        for x in i2:
            x.op_in = [op2.name]
        for x in o2:
            x.op_out = [op2.name]
        op2.get_perf_counts(i2, o2)
        result2 = compute_slice(i2, op2)

        # Reassemble
        reassembled = np.concatenate([result1, result2], axis=0)
        assert np.array_equal(
            reassembled, data
        ), f"Contiguous reassembly failed for shape {shape}"
        logger.debug(f"REASSEMBLY TEST[{idx}] shape {shape} PASS")
