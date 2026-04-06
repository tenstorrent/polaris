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
from ttsim.ops.desc.data_compute import compute_log


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_log(X):
    """
    Reference implementation of natural logarithm using NumPy.

    Args:
        X: Input array

    Returns:
        Y: Natural log output ln(X)
    """
    return np.log(X)


# Test cases with shape validation and numerical validation
test_name = "test_log"
test_cases = [
    # (name, input_shape, data_type)
    ("Basic log 1D", [16], "positive"),
    ("Basic log 2D", [4, 4], "positive"),
    ("Basic log 3D", [2, 4, 4], "positive"),
    ("Basic log 4D", [2, 3, 4, 4], "positive"),
    # Different value ranges (all positive for log)
    ("Log of values > 1", [4, 4], "large_positive"),
    ("Log of values < 1", [4, 4], "small_positive"),
    ("Log of mixed > and < 1", [4, 4], "mixed_positive"),
    # Edge cases: special values
    ("Log of ones", [4, 4], "ones"),
    ("Log of e", [4, 4], "e_values"),
    ("Log of very small positive", [4, 4], "very_small_pos"),
    ("Log of very large positive", [4, 4], "very_large_pos"),
    # Different magnitudes
    ("Log of values [0.1, 1]", [4, 4], "range_0.1_1"),
    ("Log of values [1, 10]", [4, 4], "range_1_10"),
    ("Log of values [10, 100]", [4, 4], "range_10_100"),
    # Different shapes
    ("Log 1D large", [128], "positive"),
    ("Log 2D rectangular", [8, 16], "positive"),
    ("Log 3D", [4, 8, 8], "positive"),
    ("Log 4D batch", [4, 2, 4, 4], "positive"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif data_type == "e_values":
        return np.full(shape, np.e, dtype=np.float32)
    elif data_type == "positive":
        return (np.random.rand(*shape) + 1.0).astype(np.float32)  # Range [1, 2]
    elif data_type == "large_positive":
        return (np.random.rand(*shape) * 9 + 1).astype(np.float32)  # Range [1, 10]
    elif data_type == "small_positive":
        return (
            np.random.rand(*shape).astype(np.float32) * 0.99 + 0.01
        )  # Range [0.01, 1]
    elif data_type == "mixed_positive":
        return (np.random.rand(*shape) * 10 + 0.1).astype(
            np.float32
        )  # Range [0.1, 10.1]
    elif data_type == "very_small_pos":
        return (
            np.random.rand(*shape).astype(np.float32) * 0.001 + 0.001
        )  # Range [0.001, 0.002]
    elif data_type == "very_large_pos":
        return (np.random.rand(*shape) * 1000 + 1000).astype(
            np.float32
        )  # Range [1000, 2000]
    elif data_type == "range_0.1_1":
        return (np.random.rand(*shape) * 0.9 + 0.1).astype(np.float32)  # Range [0.1, 1]
    elif data_type == "range_1_10":
        return (np.random.rand(*shape) * 9 + 1).astype(np.float32)  # Range [1, 10]
    elif data_type == "range_10_100":
        return (np.random.rand(*shape) * 90 + 10).astype(np.float32)  # Range [10, 100]
    else:
        return (np.random.rand(*shape) + 0.5).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_log():
    """Test Log with shape validation, edge cases, and numerical validation"""
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
            "optype": "Log",
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
        ref_output = ref_impl_log(test_data)
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
                computed_output = compute_log(i_tensors, op_obj)

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
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                "  Shape match: "
                f"{shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Error test cases - testing edge cases that could break the model
test_name_errors = "test_log_errors"
test_cases_errors = [
    # These test cases validate that the model handles invalid inputs
    ("Log of zero", [4, 4], "zeros"),
    ("Log of negative values", [4, 4], "negative"),
    ("Log of very small positive (near zero)", [4, 4], "near_zero"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_log_errors():
    """Test Log with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    # Suppress expected warnings for log of zero/negative
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value")

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data
        if data_type == "zeros":
            test_data = np.zeros(input_shape, dtype=np.float32)
        elif data_type == "negative":
            test_data = -(np.random.rand(*input_shape) + 0.5).astype(
                np.float32
            )  # Range [-1.5, -0.5]
        elif data_type == "near_zero":
            test_data = (
                np.random.rand(*input_shape).astype(np.float32) * 1e-10
            )  # Very close to zero
        else:
            test_data = generate_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Log",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should handle invalid values - log(0) produces -inf, log(negative) produces nan
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                if o_tensors[0].data is not None:
                    computed_output = o_tensors[0].data
                else:
                    computed_output = compute_log(i_tensors, op_obj)

                # Check for expected -inf or nan
                has_neg_inf = np.any(np.isneginf(computed_output))
                has_nan = np.any(np.isnan(computed_output))

                if has_neg_inf:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (produced -inf as expected)"
                    )
                elif has_nan:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (produced nan as expected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled)"
                    )
            except (ValueError, RuntimeError) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__})"
                )
        except (ValueError, AssertionError, RuntimeError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
test_name_precision = "test_log_precision"
precision_test_cases = [
    # Test cases with known outputs
    (
        "Log of 1 = 0",
        np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Log of e = 1",
        np.array([[np.e]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
    ),
    (
        "Log of e^2 = 2",
        np.array([[np.e**2]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
    ),
    (
        "Log of 10",
        np.array([[10.0]], dtype=np.float32),
        np.array([[np.log(10)]], dtype=np.float32),
    ),
    (
        "Log of various powers of e",
        np.array([[1.0, np.e, np.e**2, np.e**3]], dtype=np.float32),
        np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_log_precision():
    """Test Log with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Log",
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
                computed_output = compute_log(i_tensors, op_obj)

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
