#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def ref_impl(shape0, shape1):
    _X0 = np.random.randn(*shape0)
    _X1 = np.random.randn(*shape1)
    _Y = np.matmul(_X0, _X1)
    return list(_Y.shape)


# Test cases
test_name = "test_matmul"
test_cases = [
    ("Standard 2D Matrix Multiplication", [3, 4], [4, 5]),
    ("Vector-Matrix Multiplication", [4], [4, 5]),
    ("Matrix-Vector Multiplication", [3, 4], [4]),
    ("Vector-Vector Multiplication", [4], [4]),
    ("Batched Matrix Multiplication", [2, 3, 4], [2, 4, 5]),
    ("Single Element Matrices", [1, 1], [1, 1]),
    ("Empty Dimension Case", [3, 0], [0, 4]),
    ("Batched Vector Operations", [2, 4], [2, 4, 5]),
    ("High-dimensional Batched", [2, 3, 4, 5], [2, 3, 5, 6]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_matmul():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, shape0, shape1) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"
        i_tensors = [
            F._from_shape("X0", shape0, np_dtype=np.float32),
            F._from_shape("X1", shape1, np_dtype=np.float32),
        ]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "MatMul",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(shape0, shape1)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print("INPUTS:")
            for x in i_tensors:
                print("\t", x)
            print("OUTPUTS:")
            for x in o_tensors:
                print("\t", x)
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


# Error test cases
test_name_errors = "test_matmul_errors"
test_cases_errors = [
    ("Incompatible matrix dimensions", [3, 4], [3, 5]),
    ("Mismatched batch dimensions", [2, 3, 4], [3, 4, 5]),
    ("Incompatible inner dimensions", [3, 4], [5, 6]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_matmul_errors():
    """Test MatMul with incompatible shapes that should raise errors"""
    msgw = max([len(x[0]) for x in test_cases_errors])
    for tno, (tmsg, shape0, shape1) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"
        i_tensors = [
            F._from_shape("X0", shape0, np_dtype=np.float32),
            F._from_shape("X1", shape1, np_dtype=np.float32),
        ]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "MatMul",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)")


# ============================================================================
# Additional tests for numerical validation, precision, and properties
# ============================================================================

from ttsim.ops.desc.data_compute import compute_matmul


def ref_impl_matmul(A, B):
    """Reference implementation using NumPy matmul"""
    return np.matmul(A, B)


@pytest.mark.unit
@pytest.mark.opunit
def test_matmul_numerical():
    """Test MatMul with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    print("\nTesting MatMul Numerical Validation:")

    test_cases_numerical = []

    # Test 1: Standard 2D matrix multiplication
    A1 = np.random.randn(3, 4).astype(np.float32)
    B1 = np.random.randn(4, 5).astype(np.float32)
    test_cases_numerical.append(("2D matrix (3x4) @ (4x5)", A1, B1))

    # Test 2: Square matrices
    A2 = np.random.randn(5, 5).astype(np.float32)
    B2 = np.random.randn(5, 5).astype(np.float32)
    test_cases_numerical.append(("Square matrices (5x5)", A2, B2))

    # Test 3: Vector-Matrix multiplication
    A3 = np.random.randn(4).astype(np.float32)
    B3 = np.random.randn(4, 5).astype(np.float32)
    test_cases_numerical.append(("Vector @ Matrix (4) @ (4x5)", A3, B3))

    # Test 4: Matrix-Vector multiplication
    A4 = np.random.randn(3, 4).astype(np.float32)
    B4 = np.random.randn(4).astype(np.float32)
    test_cases_numerical.append(("Matrix @ Vector (3x4) @ (4)", A4, B4))

    # Test 5: Vector-Vector (dot product)
    A5 = np.random.randn(5).astype(np.float32)
    B5 = np.random.randn(5).astype(np.float32)
    test_cases_numerical.append(("Vector @ Vector (5) @ (5)", A5, B5))

    # Test 6: Batched matrix multiplication
    A6 = np.random.randn(2, 3, 4).astype(np.float32)
    B6 = np.random.randn(2, 4, 5).astype(np.float32)
    test_cases_numerical.append(("Batched (2x3x4) @ (2x4x5)", A6, B6))

    # Test 7: High-dimensional batched
    A7 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    B7 = np.random.randn(2, 3, 5, 6).astype(np.float32)
    test_cases_numerical.append(("High-dim batched (2x3x4x5) @ (2x3x5x6)", A7, B7))

    # Test 8: Broadcasting - matrix @ batched
    A8 = np.random.randn(3, 4).astype(np.float32)
    B8 = np.random.randn(2, 4, 5).astype(np.float32)
    test_cases_numerical.append(("Broadcasting (3x4) @ (2x4x5)", A8, B8))

    # Test 9: Rectangular matrices
    A9 = np.random.randn(10, 3).astype(np.float32)
    B9 = np.random.randn(3, 7).astype(np.float32)
    test_cases_numerical.append(("Rectangular (10x3) @ (3x7)", A9, B9))

    # Test 10: Single row/column
    A10 = np.random.randn(1, 5).astype(np.float32)
    B10 = np.random.randn(5, 1).astype(np.float32)
    test_cases_numerical.append(("Row @ Column (1x5) @ (5x1)", A10, B10))

    passed = 0
    total = len(test_cases_numerical)

    for name, A, B in test_cases_numerical:
        # Create mock objects
        iTList = [MockTensor(A), MockTensor(B)]
        op = MockOp()

        # Compute using the function under test
        result = compute_matmul(iTList, op)

        # Compute expected result
        expected = ref_impl_matmul(A, B)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"

        # Numerical validation
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"{name}: Numerical mismatch",
        )

        print(f"  {name}: PASS [Shape ✓, Numerical ✓]")
        passed += 1

    print(f"\nMatMul Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_matmul_precision():
    """Test MatMul with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    print("\nTesting MatMul Precision (Known Outputs):")

    # Test 1: Identity matrix multiplication
    print("  Test 1: Identity matrix")
    A1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    I1 = np.eye(3, dtype=np.float32)
    iTList1 = [MockTensor(A1), MockTensor(I1)]
    op1 = MockOp()
    result1 = compute_matmul(iTList1, op1)
    expected1 = A1.copy()
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    print(f"    A @ I = A ✓")

    # Test 2: Zero matrix multiplication
    print("  Test 2: Zero matrix")
    A2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    Z2 = np.zeros((2, 3), dtype=np.float32)
    iTList2 = [MockTensor(A2), MockTensor(Z2)]
    op2 = MockOp()
    result2 = compute_matmul(iTList2, op2)
    expected2 = np.zeros((2, 3), dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    print(f"    A @ 0 = 0 ✓")

    # Test 3: Simple 2x2 matrices
    print("  Test 3: Simple 2x2 matmul")
    A3 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B3 = np.array([[2, 0], [1, 2]], dtype=np.float32)
    iTList3 = [MockTensor(A3), MockTensor(B3)]
    op3 = MockOp()
    result3 = compute_matmul(iTList3, op3)
    # [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]
    expected3 = np.array([[4, 4], [10, 8]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    print(f"    [[1,2],[3,4]] @ [[2,0],[1,2]] = [[4,4],[10,8]] ✓")

    # Test 4: Ones matrix
    print("  Test 4: Matrix of ones")
    A4 = np.ones((2, 3), dtype=np.float32)
    B4 = np.ones((3, 2), dtype=np.float32)
    iTList4 = [MockTensor(A4), MockTensor(B4)]
    op4 = MockOp()
    result4 = compute_matmul(iTList4, op4)
    # Each element is sum of 3 ones = 3
    expected4 = np.full((2, 2), 3.0, dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    print(f"    ones(2,3) @ ones(3,2) = full(2,2,3.0) ✓")

    # Test 5: Diagonal matrix
    print("  Test 5: Diagonal matrix scaling")
    A5 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    D5 = np.diag([2.0, 3.0]).astype(np.float32)
    iTList5 = [MockTensor(A5), MockTensor(D5)]
    op5 = MockOp()
    result5 = compute_matmul(iTList5, op5)
    # [[1*2, 2*3], [3*2, 4*3]] = [[2, 6], [6, 12]]
    expected5 = np.array([[2, 6], [6, 12]], dtype=np.float32)
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    print(f"    A @ diag([2,3]) scales columns ✓")

    # Test 6: Vector dot product
    print("  Test 6: Vector dot product")
    v6 = np.array([1, 2, 3], dtype=np.float32)
    w6 = np.array([4, 5, 6], dtype=np.float32)
    iTList6 = [MockTensor(v6), MockTensor(w6)]
    op6 = MockOp()
    result6 = compute_matmul(iTList6, op6)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    expected6 = np.array(32.0, dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    print(f"    [1,2,3] @ [4,5,6] = 32 ✓")

    # Test 7: Outer product (column @ row)
    print("  Test 7: Outer product")
    col7 = np.array([[1], [2], [3]], dtype=np.float32)
    row7 = np.array([[4, 5]], dtype=np.float32)
    iTList7 = [MockTensor(col7), MockTensor(row7)]
    op7 = MockOp()
    result7 = compute_matmul(iTList7, op7)
    # [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]] = [[4, 5], [8, 10], [12, 15]]
    expected7 = np.array([[4, 5], [8, 10], [12, 15]], dtype=np.float32)
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    print(f"    Outer product ✓")

    print("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_matmul_properties():
    """Test mathematical properties of matrix multiplication"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    print("\nTesting MatMul Mathematical Properties:")

    # Property 1: Non-commutativity (A @ B != B @ A in general)
    print("  Property 1: Non-commutative (A @ B != B @ A)")
    A1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B1 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    result1_AB = compute_matmul([MockTensor(A1), MockTensor(B1)], MockOp())
    result1_BA = compute_matmul([MockTensor(B1), MockTensor(A1)], MockOp())

    # Verify they're different
    assert not np.allclose(
        result1_AB, result1_BA, rtol=1e-5, atol=1e-6
    ), "MatMul should not be commutative for general matrices"
    print(f"    A @ B != B @ A (verified) ✓")

    # Property 2: Associativity (A @ B) @ C = A @ (B @ C)
    print("  Property 2: Associative ((A @ B) @ C = A @ (B @ C))")
    A2 = np.random.randn(3, 4).astype(np.float32)
    B2 = np.random.randn(4, 5).astype(np.float32)
    C2 = np.random.randn(5, 6).astype(np.float32)

    AB = compute_matmul([MockTensor(A2), MockTensor(B2)], MockOp())
    result2_left = compute_matmul([MockTensor(AB), MockTensor(C2)], MockOp())

    BC = compute_matmul([MockTensor(B2), MockTensor(C2)], MockOp())
    result2_right = compute_matmul([MockTensor(A2), MockTensor(BC)], MockOp())

    np.testing.assert_allclose(result2_left, result2_right, rtol=1e-4, atol=1e-5)
    print(f"    (A @ B) @ C = A @ (B @ C) ✓")

    # Property 3: Distributivity A @ (B + C) = A @ B + A @ C
    print("  Property 3: Distributive (A @ (B + C) = A @ B + A @ C)")
    A3 = np.random.randn(3, 4).astype(np.float32)
    B3 = np.random.randn(4, 5).astype(np.float32)
    C3 = np.random.randn(4, 5).astype(np.float32)

    BC_sum = B3 + C3
    result3_left = compute_matmul([MockTensor(A3), MockTensor(BC_sum)], MockOp())

    AB3 = compute_matmul([MockTensor(A3), MockTensor(B3)], MockOp())
    AC3 = compute_matmul([MockTensor(A3), MockTensor(C3)], MockOp())
    result3_right = AB3 + AC3

    np.testing.assert_allclose(result3_left, result3_right, rtol=1e-5, atol=1e-6)
    print(f"    A @ (B + C) = A @ B + A @ C ✓")

    # Property 4: Identity element (A @ I = A and I @ A = A)
    print("  Property 4: Identity (A @ I = A, I @ A = A)")
    A4 = np.random.randn(3, 5).astype(np.float32)
    I_right = np.eye(5, dtype=np.float32)
    I_left = np.eye(3, dtype=np.float32)

    result4_right = compute_matmul([MockTensor(A4), MockTensor(I_right)], MockOp())
    np.testing.assert_allclose(result4_right, A4, rtol=1e-6, atol=1e-7)

    result4_left = compute_matmul([MockTensor(I_left), MockTensor(A4)], MockOp())
    np.testing.assert_allclose(result4_left, A4, rtol=1e-6, atol=1e-7)
    print(f"    A @ I = I @ A = A ✓")

    # Property 5: Zero element (A @ 0 = 0 and 0 @ A = 0)
    print("  Property 5: Zero (A @ 0 = 0, 0 @ A = 0)")
    A5 = np.random.randn(3, 4).astype(np.float32)
    Z_right = np.zeros((4, 5), dtype=np.float32)
    Z_left = np.zeros((2, 3), dtype=np.float32)

    result5_right = compute_matmul([MockTensor(A5), MockTensor(Z_right)], MockOp())
    expected5_right = np.zeros((3, 5), dtype=np.float32)
    np.testing.assert_allclose(result5_right, expected5_right, rtol=1e-6, atol=1e-7)

    result5_left = compute_matmul([MockTensor(Z_left), MockTensor(A5)], MockOp())
    expected5_left = np.zeros((2, 4), dtype=np.float32)
    np.testing.assert_allclose(result5_left, expected5_left, rtol=1e-6, atol=1e-7)
    print(f"    A @ 0 = 0 @ A = 0 ✓")

    # Property 6: Transpose property (A @ B)^T = B^T @ A^T
    print("  Property 6: Transpose ((A @ B)^T = B^T @ A^T)")
    A6 = np.random.randn(3, 4).astype(np.float32)
    B6 = np.random.randn(4, 5).astype(np.float32)

    AB6 = compute_matmul([MockTensor(A6), MockTensor(B6)], MockOp())
    result6_left = AB6.T

    result6_right = compute_matmul([MockTensor(B6.T), MockTensor(A6.T)], MockOp())

    np.testing.assert_allclose(result6_left, result6_right, rtol=1e-5, atol=1e-6)
    print(f"    (A @ B)^T = B^T @ A^T ✓")

    # Property 7: Scalar multiplication (c * A) @ B = c * (A @ B)
    print("  Property 7: Scalar multiplication ((c*A) @ B = c*(A @ B))")
    A7 = np.random.randn(3, 4).astype(np.float32)
    B7 = np.random.randn(4, 5).astype(np.float32)
    c = 2.5

    result7_left = compute_matmul([MockTensor(c * A7), MockTensor(B7)], MockOp())

    AB7 = compute_matmul([MockTensor(A7), MockTensor(B7)], MockOp())
    result7_right = c * AB7

    np.testing.assert_allclose(result7_left, result7_right, rtol=1e-5, atol=1e-6)
    print(f"    (c*A) @ B = c*(A @ B) ✓")

    # Property 8: Batched matmul preserves batch dimension
    print("  Property 8: Batched matmul preserves batch structure")
    A8 = np.random.randn(2, 3, 4).astype(np.float32)
    B8 = np.random.randn(2, 4, 5).astype(np.float32)

    result8 = compute_matmul([MockTensor(A8), MockTensor(B8)], MockOp())

    # Verify batch-wise operation
    for i in range(2):
        expected8_i = np.matmul(A8[i], B8[i])
        np.testing.assert_allclose(result8[i], expected8_i, rtol=1e-5, atol=1e-6)
    print(f"    Batched matmul = element-wise matmul per batch ✓")

    print("\nAll property tests passed!")
