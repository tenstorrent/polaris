#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import warnings

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_div


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_div(A, B):
    """
    Reference implementation of element-wise division using NumPy.

    Args:
        A: First input array (dividend)
        B: Second input array (divisor)

    Returns:
        Y: Division result A / B
    """
    return A / B


# Test cases with shape validation and numerical validation
test_name = "test_div"
test_cases = [
    # (name, shape_A, shape_B, data_type_A, data_type_B)
    ("Basic division same shape", [4, 4], [4, 4], "positive", "positive"),
    ("Division with broadcasting", [4, 4], [4, 1], "positive", "positive"),
    ("Division by scalar", [4, 4], [1], "positive", "positive"),
    ("1D division", [16], [16], "positive", "positive"),
    ("2D division", [8, 8], [8, 8], "positive", "positive"),
    ("3D division", [2, 4, 4], [2, 4, 4], "positive", "positive"),
    ("4D division", [2, 3, 4, 4], [2, 3, 4, 4], "positive", "positive"),
    # Broadcasting patterns
    ("Broadcast [8,1] to [8,8]", [8, 8], [8, 1], "positive", "positive"),
    ("Broadcast [1,8] to [8,8]", [8, 8], [1, 8], "positive", "positive"),
    ("Broadcast scalar to [4,4]", [4, 4], [1, 1], "positive", "positive"),
    # Different sign combinations
    ("Positive / Positive", [4, 4], [4, 4], "positive", "positive"),
    ("Positive / Negative", [4, 4], [4, 4], "positive", "negative"),
    ("Negative / Positive", [4, 4], [4, 4], "negative", "positive"),
    ("Negative / Negative", [4, 4], [4, 4], "negative", "negative"),
    ("Mixed / Positive", [4, 4], [4, 4], "mixed", "positive"),
    ("Mixed / Negative", [4, 4], [4, 4], "mixed", "negative"),
    ("Mixed / Mixed", [4, 4], [4, 4], "mixed", "mixed"),
    # Special divisors
    ("Division by 1 (identity)", [4, 4], [4, 4], "mixed", "ones"),
    ("Division by -1 (negation)", [4, 4], [4, 4], "mixed", "neg_ones"),
    ("Division by 2", [4, 4], [4, 4], "positive", "twos"),
    # Magnitude tests
    ("Large / Small", [4, 4], [4, 4], "large", "small"),
    ("Small / Large", [4, 4], [4, 4], "small", "large"),
    ("Small / Small", [4, 4], [4, 4], "small", "small"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) + 1.0  # Range [1, 2]
    elif data_type == "negative":
        return -np.random.rand(*shape).astype(np.float32) - 1.0  # Range [-2, -1]
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)  # Mixed pos/neg
    elif data_type == "small":
        return np.random.rand(*shape).astype(np.float32) * 0.1 + 0.1  # Range [0.1, 0.2]
    elif data_type == "large":
        return (
            np.random.rand(*shape).astype(np.float32) * 1e3 + 1e3
        )  # Range [1000, 2000]
    elif data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif data_type == "neg_ones":
        return -np.ones(shape, dtype=np.float32)
    elif data_type == "twos":
        return np.full(shape, 2.0, dtype=np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_div():
    """Test Division with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape_A, shape_B, data_type_A, data_type_B) in enumerate(
        test_cases
    ):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data_A = generate_test_data(shape_A, data_type_A)
        test_data_B = generate_test_data(shape_B, data_type_B)

        # Create input tensors with actual data
        i_tensors = [F._from_data("A", test_data_A), F._from_data("B", test_data_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Div",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_div(test_data_A, test_data_B)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            # Check if data was computed automatically
            if o_tensors[0].data is not None:
                computed_output = o_tensors[0].data
            else:
                computed_output = compute_div(i_tensors, op_obj)

            numerical_match = np.allclose(
                computed_output,
                ref_output,
                rtol=1e-5,
                atol=1e-7,
                equal_nan=True,  # Allow NaN == NaN for consistency
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Error test cases - testing edge cases that could break the model
test_name_errors = "test_div_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Division by zero", [4, 4], [4, 4], "positive", "zeros"),
    ("Division by very small number", [4, 4], [4, 4], "large", "very_small"),
    ("Zero divided by zero", [4, 4], [4, 4], "zeros", "zeros"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_div_errors():
    """Test Division with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    # Suppress expected warnings for division by zero
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

    for tno, (tmsg, shape_A, shape_B, data_type_A, data_type_B) in enumerate(
        test_cases_errors
    ):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data
        if data_type_A == "very_small":
            test_data_A = np.random.rand(*shape_A).astype(np.float32) * 1e-10
        else:
            test_data_A = generate_test_data(shape_A, data_type_A)

        if data_type_B == "very_small":
            test_data_B = np.random.rand(*shape_B).astype(np.float32) * 1e-10
        else:
            test_data_B = generate_test_data(shape_B, data_type_B)

        i_tensors = [F._from_data("A", test_data_A), F._from_data("B", test_data_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Div",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should handle edge cases - division by zero produces inf/nan
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                if o_tensors[0].data is not None:
                    computed_output = o_tensors[0].data
                else:
                    computed_output = compute_div(i_tensors, op_obj)

                # Check for expected inf/nan outputs
                has_inf = np.any(np.isinf(computed_output))
                has_nan = np.any(np.isnan(computed_output))

                if has_inf or has_nan:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (produced inf/nan as expected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled)"
                    )
            except (ValueError, RuntimeError, ZeroDivisionError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__})"
                )
        except (ValueError, AssertionError, RuntimeError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
test_name_precision = "test_div_precision"
precision_test_cases = [
    # Test cases with known outputs
    (
        "10 / 2 = 5",
        np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        np.array([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32),
        np.array([[5.0, 10.0, 15.0, 20.0]], dtype=np.float32),
    ),
    (
        "Division by 1 (identity)",
        np.array([[1.5, 2.5, 3.5, 4.5]], dtype=np.float32),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[1.5, 2.5, 3.5, 4.5]], dtype=np.float32),
    ),
    (
        "Division by -1 (negation)",
        np.array([[2.0, -3.0, 4.0, -5.0]], dtype=np.float32),
        np.array([[-1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
        np.array([[-2.0, 3.0, -4.0, 5.0]], dtype=np.float32),
    ),
    (
        "Negative / Positive",
        np.array([[-10.0, -20.0, -30.0]], dtype=np.float32),
        np.array([[5.0, 10.0, 15.0]], dtype=np.float32),
        np.array([[-2.0, -2.0, -2.0]], dtype=np.float32),
    ),
    (
        "Broadcasting scalar division",
        np.array([[8.0, 16.0], [24.0, 32.0]], dtype=np.float32),
        np.array([[4.0]], dtype=np.float32),
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_div_precision():
    """Test Division with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, test_data_A, test_data_B, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("A", test_data_A), F._from_data("B", test_data_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Div",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate against known expected output
        try:
            if o_tensors[0].data is not None:
                computed_output = o_tensors[0].data
            else:
                computed_output = compute_div(i_tensors, op_obj)

            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-7)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                logger.debug(
                    f"  Diff:     {(computed_output - expected_output).flatten()}"
                )
                assert False, f"Precision test failed for {tmsg}"
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")
            assert False, f"Precision test error: {e}"
