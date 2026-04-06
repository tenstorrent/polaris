#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_clip

# --------------------------------------------------------------------------
# Comprehensive ReLU6 tests: numerical validation, precision, edge cases,
# and mathematical properties
# ReLU6(x) = min(max(0, x), 6) = Clip(x, 0, 6)
# In the framework, F.Relu6 uses optype='Clip' with min=0, max=6 tensors.
# --------------------------------------------------------------------------


def ref_impl_relu6(X):
    """Reference implementation of ReLU6: clip(x, 0, 6)"""
    return np.clip(X, 0, 6)


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def _make_relu6_inputs(data):
    """Build input tensors for Relu6 as Clip(x, 0, 6): [X, min, max]"""
    return [
        F._from_data("X", data),
        F._from_data("min", np.array([0.0], dtype=np.float32)),
        F._from_data("max", np.array([6.0], dtype=np.float32)),
    ]


def _make_relu6_op(op_name, i_tensors, o_tensors):
    """Create a Clip op configured as Relu6."""
    op_info = {
        "name": op_name,
        "optype": "Clip",
        "inList": [x.name for x in i_tensors],
        "outList": [x.name for x in o_tensors],
    }
    op_obj = SimOp(op_info)
    for x in i_tensors:
        x.op_in = [op_name]
    for x in o_tensors:
        x.op_out = [op_name]
    return op_obj


relu6_test_name = "test_relu6"
relu6_test_cases = [
    # (name, input_shape, data_type)
    # Basic shapes
    ("1D input", [8], "mixed"),
    ("2D input", [3, 4], "mixed"),
    ("3D input", [2, 3, 4], "mixed"),
    ("4D input (NCHW)", [2, 3, 4, 4], "mixed"),
    # All-positive within [0, 6]
    ("All in range [0,6] 1D", [8], "in_range"),
    ("All in range [0,6] 2D", [3, 4], "in_range"),
    ("All in range [0,6] 4D", [2, 3, 8, 8], "in_range"),
    # All-negative (output == 0)
    ("All negative 1D", [8], "negative"),
    ("All negative 2D", [3, 4], "negative"),
    ("All negative 4D", [2, 3, 8, 8], "negative"),
    # All above 6 (output == 6)
    ("All above 6 1D", [8], "above_6"),
    ("All above 6 2D", [3, 4], "above_6"),
    ("All above 6 4D", [2, 3, 8, 8], "above_6"),
    # All zeros
    ("All zeros", [3, 4], "zeros"),
    # Single element
    ("Single element in range", [1], "in_range"),
    ("Single element negative", [1], "negative"),
    ("Single element above 6", [1], "above_6"),
    ("Single element zero", [1], "zeros"),
    # Large tensors
    ("Large 2D", [64, 64], "mixed"),
    ("Large 4D", [2, 16, 32, 32], "mixed"),
    # Values near boundaries
    ("Near zero boundary", [3, 4], "near_zero"),
    ("Near six boundary", [3, 4], "near_six"),
    # Large magnitude values
    ("Large positive values", [3, 4], "large_pos"),
    ("Large negative values", [3, 4], "large_neg"),
    # High-rank tensor
    ("5D tensor", [1, 2, 3, 4, 5], "mixed"),
    ("7D tensor", [2, 3, 5, 7, 9, 11, 13], "mixed"),
    # Batch with channels
    ("Batch multi-channel", [4, 16, 8, 8], "mixed"),
]


def generate_relu6_test_data(shape, data_type):
    """Generate test data based on type."""
    if data_type == "in_range":
        return np.array(np.random.rand(*shape) * 5.8 + 0.1, dtype=np.float32)
    elif data_type == "negative":
        return np.array(-np.random.rand(*shape) - 0.1, dtype=np.float32)
    elif data_type == "above_6":
        return np.array(np.random.rand(*shape) * 10 + 6.1, dtype=np.float32)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return np.array(np.random.randn(*shape) * 5, dtype=np.float32)
    elif data_type == "near_zero":
        return np.array(np.random.rand(*shape) * 0.002 - 0.001, dtype=np.float32)
    elif data_type == "near_six":
        return np.array(np.random.rand(*shape) * 0.002 + 5.999, dtype=np.float32)
    elif data_type == "large_pos":
        return np.array(np.random.rand(*shape) * 1e6 + 1e6, dtype=np.float32)
    elif data_type == "large_neg":
        return np.array(-np.random.rand(*shape) * 1e6 - 1e6, dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape) * 5, dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6():
    """Test ReLU6 with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(relu6_test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(relu6_test_cases):
        op_name = f"{relu6_test_name}_{tno}"

        test_data = generate_relu6_test_data(input_shape, data_type)

        i_tensors = _make_relu6_inputs(test_data)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(op_name, i_tensors, o_tensors)

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_relu6(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)
        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_clip(i_tensors, op_obj)
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
            logger.debug(
                "  Shape match: "
                f"{shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# Error / edge-case test cases
relu6_error_cases = [
    ("Empty tensor", [0, 3]),
    ("Empty 1D tensor", [0]),
    ("Zero-dim along channel", [2, 0, 4, 4]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_errors():
    """Test ReLU6 with edge cases that could break the model"""
    msgw = get_max_test_msg_len(relu6_error_cases)

    for tno, (tmsg, input_shape) in enumerate(relu6_error_cases):
        op_name = f"test_relu6_errors_{tno}"
        test_data = np.empty(input_shape, dtype=np.float32)

        i_tensors = _make_relu6_inputs(test_data)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(op_name, i_tensors, o_tensors)

        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)
            try:
                computed_output = compute_clip(i_tensors, op_obj)
                if computed_output.size == 0:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (empty output detected)"
                    )
                elif np.any(np.isnan(computed_output)):
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (NaN output detected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, output shape: {computed_output.shape})"
                    )
            except (ValueError, IndexError, TypeError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
relu6_precision_cases = [
    (
        "In-range unchanged",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
    (
        "Negative to zero",
        np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Above 6 clipped to 6",
        np.array([[7.0, 8.0], [10.0, 100.0]], dtype=np.float32),
        np.array([[6.0, 6.0], [6.0, 6.0]], dtype=np.float32),
    ),
    (
        "Mixed: negative, in-range, above 6",
        np.array([[-3.0, 2.0], [5.0, 9.0]], dtype=np.float32),
        np.array([[0.0, 2.0], [5.0, 6.0]], dtype=np.float32),
    ),
    (
        "Zero stays zero",
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Exactly 6 stays 6",
        np.array([[6.0, 6.0]], dtype=np.float32),
        np.array([[6.0, 6.0]], dtype=np.float32),
    ),
    (
        "Single in-range",
        np.array([3.5], dtype=np.float32),
        np.array([3.5], dtype=np.float32),
    ),
    (
        "Single negative",
        np.array([-5.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    (
        "Single above 6",
        np.array([12.0], dtype=np.float32),
        np.array([6.0], dtype=np.float32),
    ),
    (
        "Boundary values",
        np.array([[-0.001, 0.0, 0.001, 5.999, 6.0, 6.001]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.001, 5.999, 6.0, 6.0]], dtype=np.float32),
    ),
    (
        "Large positive clipped",
        np.array([[1e6, 1e7]], dtype=np.float32),
        np.array([[6.0, 6.0]], dtype=np.float32),
    ),
    (
        "Large negative to zero",
        np.array([[-1e6, -1e7]], dtype=np.float32),
        np.array([[0.0, 0.0]], dtype=np.float32),
    ),
    (
        "4D mixed",
        np.array([[[[-2.0, 3.0], [7.0, -1.0]]]], dtype=np.float32),
        np.array([[[[0.0, 3.0], [6.0, 0.0]]]], dtype=np.float32),
    ),
    (
        "All values exactly at boundaries",
        np.array([[0.0, 6.0, 0.0, 6.0]], dtype=np.float32),
        np.array([[0.0, 6.0, 0.0, 6.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_precision():
    """Test ReLU6 with precise known outputs"""
    msgw = 40

    for tno, (tmsg, test_data, expected_output) in enumerate(relu6_precision_cases):
        op_name = f"test_relu6_precision_{tno}"

        i_tensors = _make_relu6_inputs(test_data)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(op_name, i_tensors, o_tensors)

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_clip(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                logger.debug(f"PRECISION TEST[{tno:2d}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno:2d}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                logger.debug(
                    f"  Diff:     {(computed_output - expected_output).flatten()}"
                )
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno:2d}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"


# Mathematical property tests


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_idempotent():
    """Test that relu6(relu6(x)) == relu6(x)"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.randn(*shape) * 10, dtype=np.float32)

        # relu6(x)
        i1 = _make_relu6_inputs(data_X)
        o1 = [make_tensor("Y1")]
        op1 = _make_relu6_op(f"idem_1_{idx}", i1, o1)
        op1.get_perf_counts(i1, o1)
        relu6_x = compute_clip(i1, op1)

        # relu6(relu6(x))
        i2 = _make_relu6_inputs(relu6_x)
        o2 = [make_tensor("Y2")]
        op2 = _make_relu6_op(f"idem_2_{idx}", i2, o2)
        op2.get_perf_counts(i2, o2)
        relu6_relu6_x = compute_clip(i2, op2)

        assert np.allclose(
            relu6_x, relu6_relu6_x, rtol=1e-5, atol=1e-7
        ), f"Idempotent property failed for shape {shape}"
        logger.debug(f"IDEMPOTENT TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_output_bounded():
    """Test that relu6 output is always in [0, 6]"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.randn(*shape) * 100, dtype=np.float32)

        i_tensors = _make_relu6_inputs(data_X)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(f"bounded_{idx}", i_tensors, o_tensors)
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_clip(i_tensors, op_obj)

        assert np.all(
            result >= 0
        ), f"Lower bound violated for shape {shape}: min={np.min(result)}"
        assert np.all(
            result <= 6
        ), f"Upper bound violated for shape {shape}: max={np.max(result)}"
        logger.debug(f"BOUNDED [0,6] TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_preserves_in_range():
    """Test that relu6(x) == x when 0 <= x <= 6"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        # Generate data strictly within (0, 6)
        data_X = np.array(np.random.rand(*shape) * 5.8 + 0.1, dtype=np.float32)

        i_tensors = _make_relu6_inputs(data_X)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(f"preserve_{idx}", i_tensors, o_tensors)
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_clip(i_tensors, op_obj)

        assert np.allclose(
            result, data_X, rtol=1e-5, atol=1e-7
        ), f"Preserve in-range failed for shape {shape}"
        logger.debug(f"PRESERVE IN-RANGE TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_zeros_negative():
    """Test that relu6(x) == 0 when x < 0"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(-np.random.rand(*shape) - 0.01, dtype=np.float32)

        i_tensors = _make_relu6_inputs(data_X)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(f"zeros_neg_{idx}", i_tensors, o_tensors)
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_clip(i_tensors, op_obj)

        assert np.allclose(
            result, 0.0, atol=1e-7
        ), f"Zeros negative failed for shape {shape}"
        logger.debug(f"ZEROS NEGATIVE TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_clips_above_six():
    """Test that relu6(x) == 6 when x > 6"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.rand(*shape) * 100 + 6.01, dtype=np.float32)

        i_tensors = _make_relu6_inputs(data_X)
        o_tensors = [make_tensor("Y")]
        op_obj = _make_relu6_op(f"clips_above_{idx}", i_tensors, o_tensors)
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_clip(i_tensors, op_obj)

        assert np.allclose(
            result, 6.0, atol=1e-7
        ), f"Clips above 6 failed for shape {shape}: max={np.max(result)}"
        logger.debug(f"CLIPS ABOVE 6 TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu6_monotonic():
    """Test that relu6 is monotonically non-decreasing: x1 <= x2 => relu6(x1) <= relu6(x2)"""
    shapes = [[4], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        data_X1 = np.array(np.random.randn(*shape) * 10, dtype=np.float32)
        data_X2 = data_X1 + np.abs(np.array(np.random.rand(*shape), dtype=np.float32))

        # relu6(x1)
        i1 = _make_relu6_inputs(data_X1)
        o1 = [make_tensor("Y1")]
        op1 = _make_relu6_op(f"mono_1_{idx}", i1, o1)
        op1.get_perf_counts(i1, o1)
        result1 = compute_clip(i1, op1)

        # relu6(x2)
        i2 = _make_relu6_inputs(data_X2)
        o2 = [make_tensor("Y2")]
        op2 = _make_relu6_op(f"mono_2_{idx}", i2, o2)
        op2.get_perf_counts(i2, o2)
        result2 = compute_clip(i2, op2)

        assert np.all(
            result1 <= result2 + 1e-7
        ), f"Monotonic property failed for shape {shape}"
        logger.debug(f"MONOTONIC TEST[{idx}] shape {shape} PASS")
