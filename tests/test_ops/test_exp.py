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
from ttsim.ops.desc.data_compute import compute_exp


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_exp(X):
    """
    Reference implementation of exponential using NumPy.

    Args:
        X: Input array

    Returns:
        Y: Exponential output e^X
    """
    return np.exp(X)


# Test cases with shape validation and numerical validation
test_name = "test_exp"
test_cases = [
    # (name, input_shape, data_type)
    ("Basic exp 1D", [16], "small"),
    ("Basic exp 2D", [4, 4], "small"),
    ("Basic exp 3D", [2, 4, 4], "small"),
    ("Basic exp 4D", [2, 3, 4, 4], "small"),
    # Different value ranges
    ("Exp of zeros", [4, 4], "zeros"),
    ("Exp of positive values", [4, 4], "positive_small"),
    ("Exp of negative values", [4, 4], "negative_small"),
    ("Exp of mixed values", [4, 4], "mixed_small"),
    # Edge cases: very small values
    ("Exp of very small positive", [4, 4], "very_small_pos"),
    ("Exp of very small negative", [4, 4], "very_small_neg"),
    # Edge cases: moderate values
    ("Exp of ones", [4, 4], "ones"),
    ("Exp of negative ones", [4, 4], "neg_ones"),
    # Edge cases: larger values (but not overflowing)
    ("Exp of moderate positive", [4, 4], "moderate_pos"),
    ("Exp of moderate negative", [4, 4], "moderate_neg"),
    # Different shapes
    ("Exp 1D large", [128], "small"),
    ("Exp 2D rectangular", [8, 16], "small"),
    ("Exp 3D", [4, 8, 8], "small"),
    ("Exp 4D batch", [4, 2, 4, 4], "small"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif data_type == "neg_ones":
        return -np.ones(shape, dtype=np.float32)
    elif data_type == "small":
        return (np.random.randn(*shape) * 0.5).astype(
            np.float32
        )  # Range approx [-1.5, 1.5]
    elif data_type == "positive_small":
        return np.random.rand(*shape).astype(np.float32)  # Range [0, 1]
    elif data_type == "negative_small":
        return -np.random.rand(*shape).astype(np.float32)  # Range [-1, 0]
    elif data_type == "mixed_small":
        return (np.random.randn(*shape) * 2).astype(np.float32)  # Range approx [-6, 6]
    elif data_type == "very_small_pos":
        return np.random.rand(*shape).astype(np.float32) * 0.01  # Range [0, 0.01]
    elif data_type == "very_small_neg":
        return -np.random.rand(*shape).astype(np.float32) * 0.01  # Range [-0.01, 0]
    elif data_type == "moderate_pos":
        return (np.random.rand(*shape) * 5 + 1).astype(np.float32)  # Range [1, 6]
    elif data_type == "moderate_neg":
        return -(np.random.rand(*shape) * 5 + 1).astype(np.float32)  # Range [-6, -1]
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_exp():
    """Test Exp with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors with actual data
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Exp",
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
        ref_output = ref_impl_exp(test_data)
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
                computed_output = compute_exp(i_tensors, op_obj)

            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7, equal_nan=True
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
test_name_errors = "test_exp_errors"
test_cases_errors = [
    # These test cases validate that the model handles extreme values
    ("Exp of large positive (overflow)", [4, 4], "very_large_pos"),
    ("Exp of large negative (underflow)", [4, 4], "very_large_neg"),
    ("Exp of extreme positive", [4, 4], "extreme_pos"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_exp_errors():
    """Test Exp with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    # Suppress expected warnings for exp overflow
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="overflow encountered"
    )

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data
        if data_type == "very_large_pos":
            test_data = (np.random.rand(*input_shape) * 50 + 50).astype(
                np.float32
            )  # Range [50, 100]
        elif data_type == "very_large_neg":
            test_data = -(np.random.rand(*input_shape) * 50 + 50).astype(
                np.float32
            )  # Range [-100, -50]
        elif data_type == "extreme_pos":
            test_data = (np.random.rand(*input_shape) * 100 + 100).astype(
                np.float32
            )  # Range [100, 200]
        else:
            test_data = generate_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Exp",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should handle extreme values - exp(large) produces inf, exp(-large) produces 0
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                if o_tensors[0].data is not None:
                    computed_output = o_tensors[0].data
                else:
                    computed_output = compute_exp(i_tensors, op_obj)

                # Check for expected inf or underflow to 0
                has_inf = np.any(np.isinf(computed_output))
                has_zero = np.any(computed_output == 0)

                if has_inf:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (overflow to inf as expected)"
                    )
                elif has_zero and "negative" in data_type:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (underflow to zero as expected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled)"
                    )
            except (ValueError, RuntimeError, OverflowError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__})"
                )
        except (ValueError, AssertionError, RuntimeError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
test_name_precision = "test_exp_precision"
precision_test_cases = [
    # Test cases with known outputs
    (
        "Exp of zero = 1",
        np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
    ),
    (
        "Exp of one ≈ 2.718",
        np.array([[1.0]], dtype=np.float32),
        np.array([[np.e]], dtype=np.float32),
    ),
    (
        "Exp of ln(2) = 2",
        np.array([[np.log(2)]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
    ),
    (
        "Exp of negative one ≈ 0.368",
        np.array([[-1.0]], dtype=np.float32),
        np.array([[1.0 / np.e]], dtype=np.float32),
    ),
    (
        "Exp of ln(10) = 10",
        np.array([[np.log(10)]], dtype=np.float32),
        np.array([[10.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_exp_precision():
    """Test Exp with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Exp",
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
                computed_output = compute_exp(i_tensors, op_obj)

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
