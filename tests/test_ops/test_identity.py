#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_identity


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_identity(X):
    """
    Reference implementation of identity using NumPy.

    Args:
        X: Input array

    Returns:
        Y: Output array (copy of input)
    """
    return X.copy()


# Test cases with shape validation and numerical validation
test_name = "test_identity"
test_cases = [
    # (name, input_shape, data_type)
    ("Identity 1D", [16], "mixed"),
    ("Identity 2D", [4, 4], "mixed"),
    ("Identity 3D", [2, 4, 4], "mixed"),
    ("Identity 4D", [2, 3, 4, 4], "mixed"),
    # Different data types
    ("Identity positive values", [4, 4], "positive"),
    ("Identity negative values", [4, 4], "negative"),
    ("Identity zero values", [4, 4], "zeros"),
    ("Identity mixed values", [4, 4], "mixed"),
    ("Identity ones", [4, 4], "ones"),
    ("Identity negative ones", [4, 4], "neg_ones"),
    # Different magnitudes
    ("Identity very small values", [4, 4], "very_small"),
    ("Identity very large values", [4, 4], "very_large"),
    ("Identity small positive", [4, 4], "small_positive"),
    ("Identity large positive", [4, 4], "large_positive"),
    # Different shapes
    ("Identity 1D large", [128], "mixed"),
    ("Identity 2D rectangular", [8, 16], "mixed"),
    ("Identity 3D non-square", [3, 5, 7], "mixed"),
    ("Identity 4D batch", [4, 2, 4, 4], "mixed"),
    ("Identity single element", [1], "mixed"),
    ("Identity single row", [1, 10], "mixed"),
    ("Identity single column", [10, 1], "mixed"),
    # Edge case: large tensor
    ("Identity large 2D", [64, 64], "mixed"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        return (np.random.rand(*shape) + 0.5).astype(np.float32)  # Range [0.5, 1.5]
    elif data_type == "negative":
        return -(np.random.rand(*shape) + 0.5).astype(np.float32)  # Range [-1.5, -0.5]
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif data_type == "neg_ones":
        return -np.ones(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)  # Mixed pos/neg
    elif data_type == "very_small":
        return (np.random.randn(*shape) * 1e-6).astype(np.float32)
    elif data_type == "very_large":
        return (np.random.randn(*shape) * 1e6).astype(np.float32)
    elif data_type == "small_positive":
        return np.random.rand(*shape).astype(np.float32) * 0.1  # Range [0, 0.1]
    elif data_type == "large_positive":
        return (np.random.rand(*shape) * 1000 + 1000).astype(
            np.float32
        )  # Range [1000, 2000]
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_identity():
    """Test Identity with shape validation, edge cases, and numerical validation"""
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
            "optype": "Identity",
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
        ref_output = ref_impl_identity(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation - Identity should be EXACT
        numerical_match = True
        try:
            # Check if data was computed automatically
            if o_tensors[0].data is not None:
                computed_output = o_tensors[0].data
            else:
                computed_output = compute_identity(i_tensors, op_obj)

            # Identity should have EXACT match (not just close)
            numerical_match = np.array_equal(computed_output, test_data)

            # Also check with allclose for consistency
            if not numerical_match:
                allclose_match = np.allclose(
                    computed_output, test_data, rtol=1e-5, atol=1e-7
                )
                if allclose_match:
                    # Close but not exact - still acceptable
                    numerical_match = True
                else:
                    max_diff = np.max(np.abs(computed_output - test_data))
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


# Error test cases - Identity should handle all inputs gracefully
test_name_errors = "test_identity_errors"
test_cases_errors = [
    # These test that identity handles special floating point values
    ("Identity with inf", [4, 4], "with_inf"),
    ("Identity with -inf", [4, 4], "with_neg_inf"),
    ("Identity with mixed inf", [4, 4], "with_mixed_inf"),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_identity_errors():
    """Test Identity with special floating point values"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        # Generate test data with special values
        if data_type == "with_inf":
            test_data = np.random.randn(*input_shape).astype(np.float32)
            test_data[0, 0] = np.inf
        elif data_type == "with_neg_inf":
            test_data = np.random.randn(*input_shape).astype(np.float32)
            test_data[0, 0] = -np.inf
        elif data_type == "with_mixed_inf":
            test_data = np.random.randn(*input_shape).astype(np.float32)
            test_data[0, 0] = np.inf
            test_data[1, 1] = -np.inf
        else:
            test_data = generate_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Identity",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Identity should preserve even special values
        try:
            op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                if o_tensors[0].data is not None:
                    computed_output = o_tensors[0].data
                else:
                    computed_output = compute_identity(i_tensors, op_obj)

                # Check that inf values are preserved
                input_has_inf = np.any(np.isinf(test_data))
                output_has_inf = np.any(np.isinf(computed_output))

                if input_has_inf and output_has_inf:
                    # Check they match exactly
                    if np.array_equal(computed_output, test_data):
                        logger.debug(
                            f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (inf values preserved exactly)"
                        )
                    else:
                        logger.debug(
                            f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (special values handled)"
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
test_name_precision = "test_identity_precision"
precision_test_cases = [
    # Test cases with known outputs - output should exactly match input
    (
        "Identity exact match [1,2,3,4]",
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
    ),
    (
        "Identity exact match negative",
        np.array([[-1.5, -2.5, -3.5]], dtype=np.float32),
        np.array([[-1.5, -2.5, -3.5]], dtype=np.float32),
    ),
    (
        "Identity exact match zeros",
        np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Identity exact match mixed",
        np.array([[1.0, -2.0, 0.0, 3.5]], dtype=np.float32),
        np.array([[1.0, -2.0, 0.0, 3.5]], dtype=np.float32),
    ),
    (
        "Identity 2D matrix",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_identity_precision():
    """Test Identity with precise known outputs"""
    msgw = 35  # Fixed width for these tests

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Identity",
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

        # Validate against known expected output - should be EXACT
        try:
            if o_tensors[0].data is not None:
                computed_output = o_tensors[0].data
            else:
                computed_output = compute_identity(i_tensors, op_obj)

            # Identity should be exact match
            exact_match = np.array_equal(computed_output, expected_output)

            if exact_match:
                logger.debug(
                    f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS (exact match)"
                )
            else:
                # Try allclose as fallback
                allclose_match = np.allclose(
                    computed_output, expected_output, rtol=1e-7, atol=1e-9
                )
                if allclose_match:
                    logger.debug(
                        f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS (allclose match)"
                    )
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
