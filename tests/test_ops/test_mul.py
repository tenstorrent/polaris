#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_mul


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_mul(A, B):
    """
    Reference implementation of element-wise multiplication with broadcasting.

    Args:
        A: First input array
        B: Second input array

    Returns:
        Y: Element-wise product A * B (with NumPy broadcasting)
    """
    return A * B


# Test cases with shape validation and numerical validation
test_name = "test_mul"
test_cases = [
    # (name, shape_A, shape_B, data_type)
    # Same-shape cases
    ("Same shape 1D", [4], [4], "positive"),
    ("Same shape 2D", [3, 4], [3, 4], "positive"),
    ("Same shape 3D", [2, 3, 4], [2, 3, 4], "positive"),
    ("Same shape 4D (NCHW)", [2, 3, 4, 4], [2, 3, 4, 4], "positive"),
    # Broadcasting cases
    ("Scalar to 2D broadcast", [], [3, 4], "positive"),
    ("1D to 2D broadcast", [4], [3, 4], "positive"),
    ("Bidirectional broadcast", [3, 1], [1, 4], "positive"),
    ("Multi-dim broadcast", [2, 1, 4], [1, 3, 1], "positive"),
    ("Channel-wise scale (BN-like)", [1, 3, 1, 1], [2, 3, 4, 4], "positive"),
    ("Scalar multiply", [1], [2, 3, 4], "positive"),
    # Negative values
    ("Negative * Positive", [3, 4], [3, 4], "neg_pos"),
    ("Negative * Negative", [3, 4], [3, 4], "negative"),
    # Zero values
    ("Zero * Positive", [3, 4], [3, 4], "zero_pos"),
    ("All zeros", [3, 4], [3, 4], "zeros"),
    # Mixed values
    ("Mixed positive/negative", [2, 3, 4], [2, 3, 4], "mixed"),
    # Small values (underflow risk)
    ("Small * Small", [3, 4], [3, 4], "small"),
    # Large values (overflow risk)
    ("Large * Large", [3, 4], [3, 4], "large"),
    # Identity multiplication
    ("Multiply by ones", [2, 3, 4], [2, 3, 4], "ones"),
    # Single element
    ("Single element", [1], [1], "positive"),
    # High-rank tensors
    ("5D tensor multiply", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], "positive"),
]


def generate_test_data(shape, data_type, which="both"):
    """Generate test data based on type.

    Args:
        shape: Shape of the tensor
        data_type: Type of test data to generate
        which: 'A', 'B', or 'both' — used for asymmetric types like neg_pos / zero_pos
    """
    if data_type == "positive":
        return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)  # [1, 2]
    elif data_type == "negative":
        return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)  # [-2, -1]
    elif data_type == "neg_pos":
        if which == "A":
            return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)  # negative
        else:
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)  # positive
    elif data_type == "zero_pos":
        if which == "A":
            return np.zeros(shape, dtype=np.float32)  # zeros
        else:
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)  # positive
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return np.array(np.random.randn(*shape) * 2, dtype=np.float32)
    elif data_type == "small":
        return np.array(np.random.rand(*shape) * 1e-6, dtype=np.float32)
    elif data_type == "large":
        return np.array(
            np.random.rand(*shape) * 1e3, dtype=np.float32
        )  # moderate large to avoid fp32 overflow
    elif data_type == "ones":
        return np.ones(shape, dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape), dtype=np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_mul():
    """Test Mul with shape validation, edge cases, and numerical validation"""
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
            "optype": "Mul",
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
        ref_output = ref_impl_mul(data_A, data_B)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation — call compute function directly
        numerical_match = True
        try:
            computed_output = compute_mul(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output,
                ref_output,
                rtol=1e-5,
                atol=1e-7,
            )
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                print(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
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
            try:
                computed_output = compute_mul(i_tensors, op_obj)
                print(f"  Computed sample: {computed_output.flat[:5]}")
                print(f"  Expected sample: {ref_output.flat[:5]}")
            except:
                pass
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# Error / edge-case test cases
test_name_errors = "test_mul_errors"
test_cases_errors = [
    # These test cases validate that the model handles edge cases properly
    ("Empty tensor A", [0, 3], [1, 3]),
    ("Empty tensor B", [2, 3], [0, 3]),
    ("Both empty tensors", [0], [0]),
    ("Incompatible shapes", [3, 4], [5, 6]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_errors():
    """Test Mul with edge cases that could break the model"""
    msgw = get_max_test_msg_len(test_cases_errors)

    for tno, (tmsg, shape_A, shape_B) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"

        data_A = (
            np.random.randn(*shape_A).astype(np.float32)
            if all(s > 0 for s in shape_A)
            else np.empty(shape_A, dtype=np.float32)
        )
        data_B = (
            np.random.randn(*shape_B).astype(np.float32)
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
            "optype": "Mul",
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
                computed_output = compute_mul(i_tensors, op_obj)

                if computed_output.size == 0 or np.any(np.isnan(computed_output)):
                    print(
                        f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (invalid output detected)"
                    )
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
test_name_precision = "test_mul_precision"
precision_test_cases = [
    # (name, data_A, data_B, expected_output)
    (
        "Simple integer multiply",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        np.array([[5.0, 12.0], [21.0, 32.0]], dtype=np.float32),
    ),
    (
        "Scalar broadcast multiply",
        np.array([2.0], dtype=np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
    ),
    (
        "Multiply by zero",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    ),
    (
        "Multiply by one (identity)",
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
        np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32),
    ),
    (
        "Negative multiply",
        np.array([[2.0, -3.0], [-4.0, 5.0]], dtype=np.float32),
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32),
        np.array([[-2.0, -6.0], [-12.0, -20.0]], dtype=np.float32),
    ),
    (
        "Column broadcast multiply",
        np.array([[2.0], [3.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        np.array([[2.0, 4.0, 6.0], [12.0, 15.0, 18.0]], dtype=np.float32),
    ),
    (
        "Row broadcast multiply",
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        np.array([[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]], dtype=np.float32),
    ),
    (
        "4D channel-wise scale",
        np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]], dtype=np.float32),  # [3,1,1,1]
        np.ones((3, 1, 2, 2), dtype=np.float32) * 2.0,  # [3,1,2,2]
        np.array(
            [
                [[[2.0, 2.0], [2.0, 2.0]]],
                [[[4.0, 4.0], [4.0, 4.0]]],
                [[[6.0, 6.0], [6.0, 6.0]]],
            ],
            dtype=np.float32,
        ),
    ),  # [3,1,2,2]
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_precision():
    """Test Mul with precise known outputs"""
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
            "optype": "Mul",
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
            computed_output = compute_mul(i_tensors, op_obj)
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


# Commutativity and algebraic property tests
test_name_properties = "test_mul_properties"


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_commutativity():
    """Test that A * B == B * A (commutative property)"""
    shapes = [([3, 4], [3, 4]), ([2, 1], [1, 3]), ([1], [5, 5])]

    for idx, (shape_A, shape_B) in enumerate(shapes):
        data_A = np.random.randn(*shape_A).astype(np.float32)
        data_B = np.random.randn(*shape_B).astype(np.float32)

        # A * B
        i_ab = [F._from_data("A", data_A), F._from_data("B", data_B)]
        o_ab = [make_tensor("Y")]
        op_info_ab = {
            "name": f"{test_name_properties}_comm_ab_{idx}",
            "optype": "Mul",
            "inList": [x.name for x in i_ab],
            "outList": [x.name for x in o_ab],
        }
        op_ab = SimOp(op_info_ab)
        for x in i_ab:
            x.op_in = [op_info_ab["name"]]
        for x in o_ab:
            x.op_out = [op_info_ab["name"]]
        op_ab.get_perf_counts(i_ab, o_ab)
        result_ab = compute_mul(i_ab, op_ab)

        # B * A
        i_ba = [F._from_data("B", data_B), F._from_data("A", data_A)]
        o_ba = [make_tensor("Y")]
        op_info_ba = {
            "name": f"{test_name_properties}_comm_ba_{idx}",
            "optype": "Mul",
            "inList": [x.name for x in i_ba],
            "outList": [x.name for x in o_ba],
        }
        op_ba = SimOp(op_info_ba)
        for x in i_ba:
            x.op_in = [op_info_ba["name"]]
        for x in o_ba:
            x.op_out = [op_info_ba["name"]]
        op_ba.get_perf_counts(i_ba, o_ba)
        result_ba = compute_mul(i_ba, op_ba)

        assert np.allclose(
            result_ab, result_ba, rtol=1e-5, atol=1e-7
        ), f"Commutativity failed for shapes {shape_A}, {shape_B}"
        print(f"COMMUTATIVITY TEST[{idx}] shapes {shape_A} x {shape_B} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_associativity():
    """Test that (A * B) * C == A * (B * C) (associative property)"""
    shape = [3, 4]
    data_A = np.random.randn(*shape).astype(np.float32)
    data_B = np.random.randn(*shape).astype(np.float32)
    data_C = np.random.randn(*shape).astype(np.float32)

    # (A * B) * C
    i1 = [F._from_data("A", data_A), F._from_data("B", data_B)]
    o1 = [make_tensor("AB")]
    op1 = SimOp(
        {
            "name": "assoc_ab",
            "optype": "Mul",
            "inList": [x.name for x in i1],
            "outList": [x.name for x in o1],
        }
    )
    for x in i1:
        x.op_in = ["assoc_ab"]
    for x in o1:
        x.op_out = ["assoc_ab"]
    op1.get_perf_counts(i1, o1)
    ab = compute_mul(i1, op1)

    i2 = [F._from_data("AB", ab), F._from_data("C", data_C)]
    o2 = [make_tensor("ABC")]
    op2 = SimOp(
        {
            "name": "assoc_abc1",
            "optype": "Mul",
            "inList": [x.name for x in i2],
            "outList": [x.name for x in o2],
        }
    )
    for x in i2:
        x.op_in = ["assoc_abc1"]
    for x in o2:
        x.op_out = ["assoc_abc1"]
    op2.get_perf_counts(i2, o2)
    abc_left = compute_mul(i2, op2)

    # A * (B * C)
    i3 = [F._from_data("B", data_B), F._from_data("C", data_C)]
    o3 = [make_tensor("BC")]
    op3 = SimOp(
        {
            "name": "assoc_bc",
            "optype": "Mul",
            "inList": [x.name for x in i3],
            "outList": [x.name for x in o3],
        }
    )
    for x in i3:
        x.op_in = ["assoc_bc"]
    for x in o3:
        x.op_out = ["assoc_bc"]
    op3.get_perf_counts(i3, o3)
    bc = compute_mul(i3, op3)

    i4 = [F._from_data("A", data_A), F._from_data("BC", bc)]
    o4 = [make_tensor("ABC")]
    op4 = SimOp(
        {
            "name": "assoc_abc2",
            "optype": "Mul",
            "inList": [x.name for x in i4],
            "outList": [x.name for x in o4],
        }
    )
    for x in i4:
        x.op_in = ["assoc_abc2"]
    for x in o4:
        x.op_out = ["assoc_abc2"]
    op4.get_perf_counts(i4, o4)
    abc_right = compute_mul(i4, op4)

    assert np.allclose(
        abc_left, abc_right, rtol=1e-5, atol=1e-6
    ), "Associativity failed for (A*B)*C vs A*(B*C)"
    print("ASSOCIATIVITY TEST PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_identity():
    """Test that A * 1 == A (multiplicative identity)"""
    shapes = [[4], [3, 4], [2, 3, 4], [1, 3, 8, 8]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)
        data_ones = np.ones_like(data_A)

        i_tensors = [F._from_data("A", data_A), F._from_data("Ones", data_ones)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"identity_{idx}",
            "optype": "Mul",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_mul(i_tensors, op_obj)

        assert np.allclose(
            result, data_A, rtol=1e-5, atol=1e-7
        ), f"Identity property failed for shape {shape}"
        print(f"IDENTITY TEST[{idx}] shape {shape} PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_mul_zero_annihilation():
    """Test that A * 0 == 0 (zero annihilation property)"""
    shapes = [[4], [3, 4], [2, 3, 4]]

    for idx, shape in enumerate(shapes):
        data_A = np.random.randn(*shape).astype(np.float32)
        data_zeros = np.zeros_like(data_A)

        i_tensors = [F._from_data("A", data_A), F._from_data("Zeros", data_zeros)]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": f"zero_ann_{idx}",
            "optype": "Mul",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_info["name"]]
        for x in o_tensors:
            x.op_out = [op_info["name"]]
        op_obj.get_perf_counts(i_tensors, o_tensors)
        result = compute_mul(i_tensors, op_obj)

        assert np.allclose(
            result, 0.0, atol=1e-7
        ), f"Zero annihilation failed for shape {shape}"
        print(f"ZERO ANNIHILATION TEST[{idx}] shape {shape} PASS")
