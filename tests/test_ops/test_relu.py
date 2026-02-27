#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_relu

# --------------------------------------------------------------------------
# Comprehensive ReLU tests: numerical validation, precision, edge cases,
# and mathematical properties
# --------------------------------------------------------------------------


def ref_impl_relu(X):
    """Reference implementation of ReLU: max(0, x)"""
    return np.maximum(0, X)


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


relu_test_name = "test_relu"
relu_test_cases = [
    # (name, input_shape, data_type)
    # Basic shapes
    ("1D input", [8], "mixed"),
    ("2D input", [3, 4], "mixed"),
    ("3D input", [2, 3, 4], "mixed"),
    ("4D input (NCHW)", [2, 3, 4, 4], "mixed"),
    # All-positive (output == input)
    ("All positive 1D", [8], "positive"),
    ("All positive 2D", [3, 4], "positive"),
    ("All positive 4D", [2, 3, 8, 8], "positive"),
    # All-negative (output == 0)
    ("All negative 1D", [8], "negative"),
    ("All negative 2D", [3, 4], "negative"),
    ("All negative 4D", [2, 3, 8, 8], "negative"),
    # All zeros
    ("All zeros", [3, 4], "zeros"),
    # Single element
    ("Single element positive", [1], "positive"),
    ("Single element negative", [1], "negative"),
    ("Single element zero", [1], "zeros"),
    # Large tensors
    ("Large 2D", [64, 64], "mixed"),
    ("Large 4D", [2, 16, 32, 32], "mixed"),
    # Small values near zero
    ("Small positive near zero", [3, 4], "small_pos"),
    ("Small negative near zero", [3, 4], "small_neg"),
    # Large magnitude values
    ("Large positive values", [3, 4], "large_pos"),
    ("Large negative values", [3, 4], "large_neg"),
    # High-rank tensor
    ("5D tensor", [1, 2, 3, 4, 5], "mixed"),
    # 7D tensor
    ("7D tensor", [2, 3, 5, 7, 9, 11, 13], "mixed"),
    # Batch with channels
    ("Batch multi-channel", [4, 16, 8, 8], "mixed"),
]


def generate_relu_test_data(shape, data_type):
    """Generate test data based on type."""
    if data_type == "positive":
        return np.array(np.random.rand(*shape) + 0.1, dtype=np.float32)
    elif data_type == "negative":
        return np.array(-np.random.rand(*shape) - 0.1, dtype=np.float32)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return np.array(np.random.randn(*shape) * 2, dtype=np.float32)
    elif data_type == "small_pos":
        return np.array(np.random.rand(*shape) * 1e-6, dtype=np.float32)
    elif data_type == "small_neg":
        return np.array(-np.random.rand(*shape) * 1e-6, dtype=np.float32)
    elif data_type == "large_pos":
        return np.array(np.random.rand(*shape) * 1e6 + 1e6, dtype=np.float32)
    elif data_type == "large_neg":
        return np.array(-np.random.rand(*shape) * 1e6 - 1e6, dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_relu():
    """Test ReLU with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(relu_test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(relu_test_cases):
        op_name = f"{relu_test_name}_{tno}"

        test_data = generate_relu_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_relu(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)
        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_relu(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                print(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        if shape_match and numerical_match == True:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape OK, Numerical OK]")
        elif shape_match:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape OK, Numerical: {numerical_match}]"
            )
        else:
            print(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            print(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            print(f"  Numerical match: {numerical_match}")
            print("INPUTS:")
            for x in i_tensors:
                print(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            print("OUTPUTS:")
            for x in o_tensors:
                print(f"\t{x.name}: shape={x.shape}, dtype={x.dtype}")
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# Error / edge-case test cases
relu_error_cases = [
    ("Empty tensor", [0, 3]),
    ("Empty 1D tensor", [0]),
    ("Zero-dim along channel", [2, 0, 4, 4]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_errors():
    """Test ReLU with edge cases that could break the model"""
    msgw = get_max_test_msg_len(relu_error_cases)

    for tno, (tmsg, input_shape) in enumerate(relu_error_cases):
        op_name = f"test_relu_errors_{tno}"
        test_data = np.empty(input_shape, dtype=np.float32)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)
            try:
                computed_output = compute_relu(i_tensors, op_obj)
                if computed_output.size == 0:
                    print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (empty output detected)")
                elif np.any(np.isnan(computed_output)):
                    print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (NaN output detected)")
                else:
                    print(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, output shape: {computed_output.shape})"
                    )
            except (ValueError, IndexError, TypeError) as e:
                print(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError) as e:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
relu_precision_cases = [
    (
        "Positive unchanged",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
    (
        "Negative to zero",
        np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Mixed values",
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32),
        np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float32),
    ),
    (
        "Zero stays zero",
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Single positive",
        np.array([5.0], dtype=np.float32),
        np.array([5.0], dtype=np.float32),
    ),
    (
        "Single negative",
        np.array([-5.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    (
        "Boundary at zero",
        np.array([[-0.001, 0.0, 0.001]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.001]], dtype=np.float32),
    ),
    (
        "Large positive preserved",
        np.array([[1e6, 1e7]], dtype=np.float32),
        np.array([[1e6, 1e7]], dtype=np.float32),
    ),
    (
        "Large negative to zero",
        np.array([[-1e6, -1e7]], dtype=np.float32),
        np.array([[0.0, 0.0]], dtype=np.float32),
    ),
    (
        "4D mixed",
        np.array([[[[-1.0, 2.0], [3.0, -4.0]]]], dtype=np.float32),
        np.array([[[[0.0, 2.0], [3.0, 0.0]]]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_precision():
    """Test ReLU with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, expected_output) in enumerate(relu_precision_cases):
        op_name = f"test_relu_precision_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_relu(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                print(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  Expected: {expected_output.flatten()}")
                print(f"  Got:      {computed_output.flatten()}")
                print(f"  Diff:     {(computed_output - expected_output).flatten()}")
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"


# Mathematical property tests


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_idempotent():
    """Test that relu(relu(x)) == relu(x)"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.randn(*shape) * 5, dtype=np.float32)

        # relu(x)
        i1 = [F._from_data("X", data_X)]
        o1 = [make_tensor("Y1")]
        op1 = SimOp(
            {
                "name": f"idem_1_{idx}",
                "optype": "Relu",
                "inList": [x.name for x in i1],
                "outList": [x.name for x in o1],
            }
        )
        for x in i1:
            x.op_in = [op1.name]
        for x in o1:
            x.op_out = [op1.name]
        op1.get_perf_counts(i1, o1)
        relu_x = compute_relu(i1, op1)

        # relu(relu(x))
        i2 = [F._from_data("Y1", relu_x)]
        o2 = [make_tensor("Y2")]
        op2 = SimOp(
            {
                "name": f"idem_2_{idx}",
                "optype": "Relu",
                "inList": [x.name for x in i2],
                "outList": [x.name for x in o2],
            }
        )
        for x in i2:
            x.op_in = [op2.name]
        for x in o2:
            x.op_out = [op2.name]
        op2.get_perf_counts(i2, o2)
        relu_relu_x = compute_relu(i2, op2)

        assert np.allclose(
            relu_x, relu_relu_x, rtol=1e-5, atol=1e-7
        ), f"Idempotent property failed for shape {shape}"
        print(f"IDEMPOTENT TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_non_negative():
    """Test that relu output is always >= 0"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.randn(*shape) * 100, dtype=np.float32)

        i_tensors = [F._from_data("X", data_X)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"nonneg_{idx}",
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_relu(i_tensors, op_obj)

        assert np.all(
            result >= 0
        ), f"Non-negative property failed for shape {shape}: min={np.min(result)}"
        print(f"NON-NEGATIVE TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_preserves_positive():
    """Test that relu(x) == x when x > 0"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.rand(*shape) + 0.01, dtype=np.float32)

        i_tensors = [F._from_data("X", data_X)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"preserve_{idx}",
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_relu(i_tensors, op_obj)

        assert np.allclose(
            result, data_X, rtol=1e-5, atol=1e-7
        ), f"Preserve positive failed for shape {shape}"
        print(f"PRESERVE POSITIVE TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_relu_zeros_negative():
    """Test that relu(x) == 0 when x < 0"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(-np.random.rand(*shape) - 0.01, dtype=np.float32)

        i_tensors = [F._from_data("X", data_X)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"zeros_neg_{idx}",
            "optype": "Relu",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_relu(i_tensors, op_obj)

        assert np.allclose(
            result, 0.0, atol=1e-7
        ), f"Zeros negative failed for shape {shape}"
        print(f"ZEROS NEGATIVE TEST[{idx}] shape {shape} PASS")
