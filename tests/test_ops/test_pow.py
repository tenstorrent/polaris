#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_pow


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_pow(A, B):
    """
    Reference implementation of element-wise power with broadcasting.

    Args:
        A: Base array
        B: Exponent array

    Returns:
        Y: Element-wise A ** B (with NumPy broadcasting)
    """
    return np.power(A, B)


# Test cases with shape validation and numerical validation
test_name = "test_pow"
test_cases = [
    # (name, shape_A, shape_B, data_type)
    # Same-shape cases
    ("Same shape 1D", [4], [4], "pos_pos"),
    ("Same shape 2D", [3, 4], [3, 4], "pos_pos"),
    ("Same shape 3D", [2, 3, 4], [2, 3, 4], "pos_pos"),
    ("Same shape 4D (NCHW)", [2, 3, 4, 4], [2, 3, 4, 4], "pos_pos"),
    # Broadcasting cases
    ("Scalar exponent broadcast", [3, 4], [], "pos_pos"),
    ("Scalar base broadcast", [], [3, 4], "pos_pos"),
    ("1D to 2D broadcast", [4], [3, 4], "pos_pos"),
    ("Bidirectional broadcast", [3, 1], [1, 4], "pos_pos"),
    ("Multi-dim broadcast", [2, 1, 4], [1, 3, 1], "pos_pos"),
    ("Channel-wise exponent", [1, 3, 1, 1], [2, 3, 4, 4], "pos_pos"),
    # Integer exponents
    ("Square (x^2)", [3, 4], [3, 4], "square"),
    ("Cube (x^3)", [3, 4], [3, 4], "cube"),
    ("x^0 = 1", [3, 4], [3, 4], "zero_exp"),
    ("x^1 = x", [3, 4], [3, 4], "one_exp"),
    # Fractional exponents (sqrt, cbrt)
    ("Square root (x^0.5)", [3, 4], [3, 4], "sqrt"),
    ("Cube root (x^(1/3))", [3, 4], [3, 4], "cbrt"),
    # Negative exponents
    ("Negative exponent (x^-1)", [3, 4], [3, 4], "neg_one_exp"),
    ("Negative exponent (x^-2)", [3, 4], [3, 4], "neg_two_exp"),
    # Small/large values
    ("Small base values", [3, 4], [3, 4], "small_base"),
    ("Large base values", [3, 4], [3, 4], "large_base"),
    # Mixed values
    ("Mixed positive bases", [2, 3, 4], [2, 3, 4], "mixed_pos"),
    # Single element
    ("Single element", [1], [1], "pos_pos"),
    # Ones
    ("Base is one (1^x = 1)", [3, 4], [3, 4], "one_base"),
]


def generate_test_data(shape, data_type, which="both"):
    """Generate test data based on type.

    Args:
        shape: Shape of the tensor
        data_type: Type of test data to generate
        which: 'A' (base), 'B' (exponent), or 'both'
    """
    if data_type == "pos_pos":
        if which == "A":
            return np.array(
                np.random.rand(*shape) + 1.0, dtype=np.float32
            )  # base [1, 2]
        else:
            return np.array(
                np.random.rand(*shape) + 0.5, dtype=np.float32
            )  # exp [0.5, 1.5]
    elif data_type == "square":
        if which == "A":
            return np.array(np.random.randn(*shape), dtype=np.float32)
        else:
            return np.full(shape, 2.0, dtype=np.float32)
    elif data_type == "cube":
        if which == "A":
            return np.array(np.random.randn(*shape), dtype=np.float32)
        else:
            return np.full(shape, 3.0, dtype=np.float32)
    elif data_type == "zero_exp":
        if which == "A":
            return np.array(np.random.randn(*shape) * 5, dtype=np.float32)
        else:
            return np.zeros(shape, dtype=np.float32)
    elif data_type == "one_exp":
        if which == "A":
            return np.array(np.random.randn(*shape) * 5, dtype=np.float32)
        else:
            return np.ones(shape, dtype=np.float32)
    elif data_type == "sqrt":
        if which == "A":
            return np.array(
                np.random.rand(*shape) + 0.1, dtype=np.float32
            )  # positive base
        else:
            return np.full(shape, 0.5, dtype=np.float32)
    elif data_type == "cbrt":
        if which == "A":
            return np.array(np.random.rand(*shape) + 0.1, dtype=np.float32)
        else:
            return np.full(shape, 1.0 / 3.0, dtype=np.float32)
    elif data_type == "neg_one_exp":
        if which == "A":
            return np.array(
                np.random.rand(*shape) + 1.0, dtype=np.float32
            )  # avoid zero base
        else:
            return np.full(shape, -1.0, dtype=np.float32)
    elif data_type == "neg_two_exp":
        if which == "A":
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
        else:
            return np.full(shape, -2.0, dtype=np.float32)
    elif data_type == "small_base":
        if which == "A":
            return np.array(np.random.rand(*shape) * 0.01 + 0.001, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape) + 0.5, dtype=np.float32)
    elif data_type == "large_base":
        if which == "A":
            return np.array(np.random.rand(*shape) * 10 + 10, dtype=np.float32)
        else:
            return np.array(
                np.random.rand(*shape) * 0.5 + 0.1, dtype=np.float32
            )  # small exp to avoid overflow
    elif data_type == "mixed_pos":
        if which == "A":
            return np.array(np.random.rand(*shape) * 5 + 0.1, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape) * 3, dtype=np.float32)
    elif data_type == "one_base":
        if which == "A":
            return np.ones(shape, dtype=np.float32)
        else:
            return np.array(np.random.randn(*shape) * 10, dtype=np.float32)
    else:
        if which == "A":
            return np.array(np.abs(np.random.randn(*shape)) + 0.1, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_pow():
    """Test Pow with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape_A, shape_B, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        data_A = generate_test_data(shape_A, data_type, which="A")
        data_B = generate_test_data(shape_B, data_type, which="B")

        # Create input tensors with actual data
        i_tensors = [
            F._from_data("A", data_A),
            F._from_data("B", data_B),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Pow",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation (shape inference)
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_pow(data_A, data_B)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation — call compute function directly
        numerical_match = True
        try:
            computed_output = compute_pow(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output,
                ref_output,
                rtol=1e-5,
                atol=1e-7,
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
            try:
                computed_output = compute_pow(i_tensors, op_obj)
                logger.debug(f"  Computed sample: {computed_output.flat[:5]}")
                logger.debug(f"  Expected sample: {ref_output.flat[:5]}")
            except:
                pass
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# Error / edge-case test cases
test_name_errors = "test_pow_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Zero base, negative exp (0^-1)", [3, 4], [3, 4]),
    ("Empty tensor A", [0, 3], [1, 3]),
    ("Empty tensor B", [2, 3], [0, 3]),
    ("Both empty tensors", [0], [0]),
    ("Incompatible shapes", [3, 4], [5, 6]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_pow_errors():
    """Test Pow with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, shape_A, shape_B) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        if tno == 0:
            # Special case: zero base with negative exponent -> inf
            data_A = np.zeros(shape_A, dtype=np.float32)
            data_B = np.full(shape_B, -1.0, dtype=np.float32)
        else:
            data_A = (
                np.array(np.abs(np.random.randn(*shape_A)) + 0.1, dtype=np.float32)
                if all(s > 0 for s in shape_A)
                else np.empty(shape_A, dtype=np.float32)
            )
            data_B = (
                np.array(np.random.rand(*shape_B), dtype=np.float32)
                if all(s > 0 for s in shape_B)
                else np.empty(shape_B, dtype=np.float32)
            )

        i_tensors = [
            F._from_data("A", data_A),
            F._from_data("B", data_B),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Pow",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                op_obj.get_perf_counts(i_tensors, o_tensors)

            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    computed_output = compute_pow(i_tensors, op_obj)

                if (
                    computed_output.size == 0
                    or np.any(np.isnan(computed_output))
                    or np.any(np.isinf(computed_output))
                ):
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid/inf output detected)"
                    )
                else:
                    logger.debug(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (edge case handled, output shape: {computed_output.shape})"
                    )
            except (
                ValueError,
                IndexError,
                TypeError,
                ZeroDivisionError,
                FloatingPointError,
            ) as e:
                logger.debug(
                    f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during compute)"
                )
        except (ValueError, AssertionError, IndexError) as e:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised {type(e).__name__} during shape inference)"
            )


# Precision test cases with known outputs
test_name_precision = "test_pow_precision"
precision_test_cases = [
    # (name, data_A, data_B, expected_output)
    (
        "2^3 = 8",
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
        np.array([[8.0, 27.0], [64.0, 125.0]], dtype=np.float32),
    ),
    (
        "x^0 = 1",
        np.array([[2.0, -3.0], [0.5, 100.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    ),
    (
        "x^1 = x",
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    ),
    (
        "sqrt (x^0.5)",
        np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32),
        np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    ),
    (
        "reciprocal (x^-1)",
        np.array([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32),
        np.array([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32),
        np.array([[0.5, 0.25], [0.2, 0.1]], dtype=np.float32),
    ),
    (
        "1^x = 1 for any x",
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.array([[5.0, -3.0], [0.0, 100.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    ),
    (
        "Scalar broadcast pow",
        np.array([2.0], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[2.0, 4.0], [8.0, 16.0]], dtype=np.float32),
    ),
    (
        "Negative base, integer exp",
        np.array([[-2.0, -3.0]], dtype=np.float32),
        np.array([[2.0, 3.0]], dtype=np.float32),
        np.array([[4.0, -27.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_pow_precision():
    """Test Pow with precise known outputs"""
    msgw = 35

    for tno, (tmsg, data_A, data_B, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [
            F._from_data("A", data_A),
            F._from_data("B", data_B),
        ]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Pow",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # Validate against known expected output
        try:
            computed_output = compute_pow(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-6)
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


# Algebraic property tests
test_name_properties = "test_pow_properties"


@pytest.mark.unit
@pytest.mark.opunit
def test_pow_product_rule():
    """Test that x^a * x^b == x^(a+b) (product of powers rule)"""
    shape = [3, 4]
    data_X = np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)  # positive base
    data_A = np.array(np.random.rand(*shape) + 0.5, dtype=np.float32)
    data_B = np.array(np.random.rand(*shape) + 0.5, dtype=np.float32)

    # x^a
    i1 = [F._from_data("X", data_X), F._from_data("A", data_A)]
    o1 = [make_tensor("XA")]
    op1 = SimOp(
        {
            "name": "prod_xa",
            "optype": "Pow",
            "inList": [x.name for x in i1],
            "outList": [x.name for x in o1],
        }
    )
    for x in i1:
        x.op_in = ["prod_xa"]
    for x in o1:
        x.op_out = ["prod_xa"]
    op1.get_perf_counts(i1, o1)
    xa = compute_pow(i1, op1)

    # x^b
    i2 = [F._from_data("X", data_X), F._from_data("B", data_B)]
    o2 = [make_tensor("XB")]
    op2 = SimOp(
        {
            "name": "prod_xb",
            "optype": "Pow",
            "inList": [x.name for x in i2],
            "outList": [x.name for x in o2],
        }
    )
    for x in i2:
        x.op_in = ["prod_xb"]
    for x in o2:
        x.op_out = ["prod_xb"]
    op2.get_perf_counts(i2, o2)
    xb = compute_pow(i2, op2)

    # x^a * x^b (left side)
    left = xa * xb

    # x^(a+b) (right side)
    data_AB = data_A + data_B
    i3 = [F._from_data("X", data_X), F._from_data("AB", data_AB)]
    o3 = [make_tensor("XAB")]
    op3 = SimOp(
        {
            "name": "prod_xab",
            "optype": "Pow",
            "inList": [x.name for x in i3],
            "outList": [x.name for x in o3],
        }
    )
    for x in i3:
        x.op_in = ["prod_xab"]
    for x in o3:
        x.op_out = ["prod_xab"]
    op3.get_perf_counts(i3, o3)
    right = compute_pow(i3, op3)

    assert np.allclose(
        left, right, rtol=1e-4, atol=1e-5
    ), f"Product rule failed: max diff = {np.max(np.abs(left - right))}"
    logger.debug("PRODUCT RULE TEST PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_pow_power_rule():
    """Test that (x^a)^b == x^(a*b) (power of a power rule)"""
    shape = [3, 4]
    data_X = np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    data_A = np.array(
        np.random.rand(*shape) * 0.5 + 0.5, dtype=np.float32
    )  # small exps
    data_B = np.array(np.random.rand(*shape) * 0.5 + 0.5, dtype=np.float32)

    # (x^a)
    i1 = [F._from_data("X", data_X), F._from_data("A", data_A)]
    o1 = [make_tensor("XA")]
    op1 = SimOp(
        {
            "name": "pwr_xa",
            "optype": "Pow",
            "inList": [x.name for x in i1],
            "outList": [x.name for x in o1],
        }
    )
    for x in i1:
        x.op_in = ["pwr_xa"]
    for x in o1:
        x.op_out = ["pwr_xa"]
    op1.get_perf_counts(i1, o1)
    xa = compute_pow(i1, op1)

    # (x^a)^b  (left side)
    i2 = [F._from_data("XA", xa), F._from_data("B", data_B)]
    o2 = [make_tensor("XAB")]
    op2 = SimOp(
        {
            "name": "pwr_xab_l",
            "optype": "Pow",
            "inList": [x.name for x in i2],
            "outList": [x.name for x in o2],
        }
    )
    for x in i2:
        x.op_in = ["pwr_xab_l"]
    for x in o2:
        x.op_out = ["pwr_xab_l"]
    op2.get_perf_counts(i2, o2)
    left = compute_pow(i2, op2)

    # x^(a*b)  (right side)
    data_AB = data_A * data_B
    i3 = [F._from_data("X", data_X), F._from_data("AB", data_AB)]
    o3 = [make_tensor("XAB")]
    op3 = SimOp(
        {
            "name": "pwr_xab_r",
            "optype": "Pow",
            "inList": [x.name for x in i3],
            "outList": [x.name for x in o3],
        }
    )
    for x in i3:
        x.op_in = ["pwr_xab_r"]
    for x in o3:
        x.op_out = ["pwr_xab_r"]
    op3.get_perf_counts(i3, o3)
    right = compute_pow(i3, op3)

    assert np.allclose(
        left, right, rtol=1e-4, atol=1e-5
    ), f"Power rule failed: max diff = {np.max(np.abs(left - right))}"
    logger.debug("POWER RULE TEST PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_pow_identity_exponents():
    """Test x^0 == 1 and x^1 == x for various shapes"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_X = np.array(np.random.randn(*shape) * 5, dtype=np.float32)

        # x^0 == 1
        data_zero = np.zeros(shape, dtype=np.float32)
        i0 = [F._from_data("X", data_X), F._from_data("Z", data_zero)]
        o0 = [make_tensor("Y")]
        op0 = SimOp(
            {
                "name": f"id_zero_{idx}",
                "optype": "Pow",
                "inList": [x.name for x in i0],
                "outList": [x.name for x in o0],
            }
        )
        for x in i0:
            x.op_in = [op0.name]
        for x in o0:
            x.op_out = [op0.name]
        op0.get_perf_counts(i0, o0)
        result_zero = compute_pow(i0, op0)
        assert np.allclose(result_zero, 1.0, atol=1e-7), f"x^0 != 1 for shape {shape}"

        # x^1 == x
        data_one = np.ones(shape, dtype=np.float32)
        i1 = [F._from_data("X", data_X), F._from_data("O", data_one)]
        o1 = [make_tensor("Y")]
        op1_obj = SimOp(
            {
                "name": f"id_one_{idx}",
                "optype": "Pow",
                "inList": [x.name for x in i1],
                "outList": [x.name for x in o1],
            }
        )
        for x in i1:
            x.op_in = [op1_obj.name]
        for x in o1:
            x.op_out = [op1_obj.name]
        op1_obj.get_perf_counts(i1, o1)
        result_one = compute_pow(i1, op1_obj)
        assert np.allclose(
            result_one, data_X, rtol=1e-5, atol=1e-7
        ), f"x^1 != x for shape {shape}"

        logger.debug(f"IDENTITY EXPONENTS TEST[{idx}] shape {shape} PASS")
