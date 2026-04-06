#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test element-wise addition operation with broadcasting"""

import numpy as np
import pytest
from loguru import logger

from ttsim.ops.desc.data_compute import compute_add


def ref_impl_add(A, B):
    """
    Reference implementation of element-wise addition.
    Supports NumPy broadcasting rules.

    Properties:
    - Commutative: A + B = B + A
    - Associative: (A + B) + C = A + (B + C)
    - Identity: A + 0 = A
    - Supports broadcasting for compatible shapes
    """
    return A + B


def generate_test_data():
    """Generate comprehensive test cases for element-wise addition"""
    test_cases = []

    # Basic same-shape additions
    test_cases.append(
        (
            "1d_same",
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "2d_same",
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "3d_same",
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(2, 3, 4).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "4d_same",
            np.random.randn(1, 2, 3, 4).astype(np.float32),
            np.random.randn(1, 2, 3, 4).astype(np.float32),
        )
    )

    # Broadcasting: scalar + array
    test_cases.append(
        (
            "scalar_to_1d",
            np.array([1, 2, 3], dtype=np.float32),
            np.array([5.0], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "scalar_to_2d",
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([10.0], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "scalar_to_3d",
            np.random.randn(2, 3, 4).astype(np.float32),
            np.array([5.0], dtype=np.float32),
        )
    )

    # Broadcasting: 1D array to higher dimensions
    test_cases.append(
        (
            "1d_to_2d_row",
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.array([10, 20, 30], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "1d_to_2d_col",
            np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
            np.array([[10], [20], [30]], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "1d_to_3d",
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(4).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "1d_to_4d",
            np.random.randn(1, 2, 3, 4).astype(np.float32),
            np.random.randn(4).astype(np.float32),
        )
    )

    # Broadcasting: 2D to higher dimensions
    test_cases.append(
        (
            "2d_to_3d",
            np.random.randn(2, 3, 4).astype(np.float32),
            np.random.randn(3, 4).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "2d_to_4d",
            np.random.randn(2, 3, 4, 5).astype(np.float32),
            np.random.randn(4, 5).astype(np.float32),
        )
    )

    # Broadcasting: image + bias (common in neural networks)
    test_cases.append(
        (
            "image_bias_nchw",
            np.random.randn(1, 3, 224, 224).astype(np.float32),
            np.random.randn(1, 3, 1, 1).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "batch_bias",
            np.random.randn(8, 64, 32, 32).astype(np.float32),
            np.random.randn(1, 64, 1, 1).astype(np.float32),
        )
    )

    # Broadcasting: different dimensions
    test_cases.append(
        (
            "broadcast_middle",
            np.random.randn(2, 1, 4).astype(np.float32),
            np.random.randn(2, 3, 4).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "broadcast_first",
            np.random.randn(1, 3, 4).astype(np.float32),
            np.random.randn(2, 3, 4).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "broadcast_last",
            np.random.randn(2, 3, 1).astype(np.float32),
            np.random.randn(2, 3, 4).astype(np.float32),
        )
    )

    # Zero addition (identity property)
    test_cases.append(
        (
            "zero_1d",
            np.array([1, 2, 3], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "zero_2d",
            np.random.randn(3, 4).astype(np.float32),
            np.zeros((3, 4), dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "zero_broadcast",
            np.random.randn(2, 3, 4).astype(np.float32),
            np.zeros(1, dtype=np.float32),
        )
    )

    # Negative numbers
    test_cases.append(
        (
            "negative",
            np.array([1, -2, 3], dtype=np.float32),
            np.array([-4, 5, -6], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "all_negative",
            np.array([-1, -2, -3], dtype=np.float32),
            np.array([-4, -5, -6], dtype=np.float32),
        )
    )

    # Large numbers
    test_cases.append(
        (
            "large_numbers",
            np.array([1e6, 1e7, 1e8], dtype=np.float32),
            np.array([2e6, 3e7, 4e8], dtype=np.float32),
        )
    )
    test_cases.append(
        (
            "very_large",
            np.array([1e20, 1e21], dtype=np.float32),
            np.array([2e20, 3e21], dtype=np.float32),
        )
    )

    # Small numbers
    test_cases.append(
        (
            "small_numbers",
            np.array([1e-6, 1e-7, 1e-8], dtype=np.float32),
            np.array([2e-6, 3e-7, 4e-8], dtype=np.float32),
        )
    )

    # Mixed positive and negative
    test_cases.append(
        (
            "mixed_signs",
            np.array([1, -2, 3, -4], dtype=np.float32),
            np.array([-1, 2, -3, 4], dtype=np.float32),
        )
    )

    # Batch operations (common in deep learning)
    test_cases.append(
        (
            "batch_8",
            np.random.randn(8, 10).astype(np.float32),
            np.random.randn(8, 10).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "batch_32",
            np.random.randn(32, 20).astype(np.float32),
            np.random.randn(32, 20).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "batch_broadcast",
            np.random.randn(16, 1).astype(np.float32),
            np.random.randn(16, 10).astype(np.float32),
        )
    )

    # Complex broadcasting scenarios
    test_cases.append(
        (
            "broadcast_multi_1",
            np.random.randn(1, 3, 1, 4).astype(np.float32),
            np.random.randn(2, 1, 5, 1).astype(np.float32),
        )
    )
    test_cases.append(
        (
            "broadcast_multi_2",
            np.random.randn(2, 1, 4, 1).astype(np.float32),
            np.random.randn(1, 3, 1, 5).astype(np.float32),
        )
    )

    return test_cases


@pytest.mark.unit
@pytest.mark.opunit
def test_add():
    """Test element-wise addition with shape and numerical validation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    test_cases = generate_test_data()
    passed = 0
    total = len(test_cases)

    for name, A, B in test_cases:
        # Create mock objects
        iTList = [MockTensor(A), MockTensor(B)]
        op = MockOp()

        # Compute using the function under test
        result = compute_add(iTList, op)

        # Compute expected result
        expected = ref_impl_add(A, B)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"

        # Numerical validation
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"{name}: Numerical mismatch",
        )

        logger.debug(f"  {name}: PASS [Shape ✓, Numerical ✓]")
        passed += 1

    logger.info(f"\nAdd Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_add_errors():
    """Test addition edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Add Edge Cases:")

    # Test 1: Adding zeros (identity property)
    logger.info("  Test 1: Adding zeros (identity)")
    A1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    B1 = np.zeros(5, dtype=np.float32)
    iTList1 = [MockTensor(A1), MockTensor(B1)]
    op1 = MockOp()
    result1 = compute_add(iTList1, op1)
    expected1 = ref_impl_add(A1, B1)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result1, A1, rtol=1e-5, atol=1e-7)  # Should equal A
    logger.debug("    PASS (A + 0 = A)")

    # Test 2: Adding negative of itself (should be zero)
    logger.info("  Test 2: Adding negative of itself")
    A2 = np.array([5, 10, 15, 20], dtype=np.float32)
    B2 = -A2
    iTList2 = [MockTensor(A2), MockTensor(B2)]
    op2 = MockOp()
    result2 = compute_add(iTList2, op2)
    expected2 = ref_impl_add(A2, B2)
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result2, np.zeros_like(A2), rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (A + (-A) = 0)")

    # Test 3: Very large numbers (potential overflow)
    logger.info("  Test 3: Very large numbers")
    A3 = np.array([1e30, 1e31, 1e32], dtype=np.float32)
    B3 = np.array([2e30, 3e31, 4e32], dtype=np.float32)
    iTList3 = [MockTensor(A3), MockTensor(B3)]
    op3 = MockOp()
    result3 = compute_add(iTList3, op3)
    expected3 = ref_impl_add(A3, B3)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (large numbers handled)")

    # Test 4: Very small numbers (underflow)
    logger.info("  Test 4: Very small numbers")
    A4 = np.array([1e-30, 1e-31, 1e-32], dtype=np.float32)
    B4 = np.array([2e-30, 3e-31, 4e-32], dtype=np.float32)
    iTList4 = [MockTensor(A4), MockTensor(B4)]
    op4 = MockOp()
    result4 = compute_add(iTList4, op4)
    expected4 = ref_impl_add(A4, B4)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (small numbers handled)")

    # Test 5: Mixed magnitude (large + small)
    logger.info("  Test 5: Mixed magnitude (large + small)")
    A5 = np.array([1e10, 1e10, 1e10], dtype=np.float32)
    B5 = np.array([1e-10, 1e-10, 1e-10], dtype=np.float32)
    iTList5 = [MockTensor(A5), MockTensor(B5)]
    op5 = MockOp()
    result5 = compute_add(iTList5, op5)
    expected5 = ref_impl_add(A5, B5)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (mixed magnitude)")

    # Test 6: Broadcasting with size 1 dimensions
    logger.info("  Test 6: Broadcasting with size 1 dimensions")
    A6 = np.random.randn(1, 1, 5, 5).astype(np.float32)
    B6 = np.random.randn(3, 4, 1, 1).astype(np.float32)
    iTList6 = [MockTensor(A6), MockTensor(B6)]
    op6 = MockOp()
    result6 = compute_add(iTList6, op6)
    expected6 = ref_impl_add(A6, B6)
    assert result6.shape == expected6.shape
    assert result6.shape == (3, 4, 5, 5), f"Expected (3,4,5,5), got {result6.shape}"
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (complex broadcasting)")

    # Test 7: Single element arrays
    logger.info("  Test 7: Single element arrays")
    A7 = np.array([42.0], dtype=np.float32)
    B7 = np.array([58.0], dtype=np.float32)
    iTList7 = [MockTensor(A7), MockTensor(B7)]
    op7 = MockOp()
    result7 = compute_add(iTList7, op7)
    expected7 = ref_impl_add(A7, B7)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-7)
    assert result7[0] == 100.0, f"Expected 100.0, got {result7[0]}"
    logger.debug("    PASS (single element)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_add_precision():
    """Test addition with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Add Precision (Known Outputs):")

    # Test 1: Simple integers
    logger.info("  Test 1: 2 + 3 = 5")
    A1 = np.array([2.0], dtype=np.float32)
    B1 = np.array([3.0], dtype=np.float32)
    iTList1 = [MockTensor(A1), MockTensor(B1)]
    op1 = MockOp()
    result1 = compute_add(iTList1, op1)
    expected1 = np.array([5.0], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result1[0]} ✓")

    # Test 2: Negative numbers
    logger.info("  Test 2: -5 + 3 = -2")
    A2 = np.array([-5.0], dtype=np.float32)
    B2 = np.array([3.0], dtype=np.float32)
    iTList2 = [MockTensor(A2), MockTensor(B2)]
    op2 = MockOp()
    result2 = compute_add(iTList2, op2)
    expected2 = np.array([-2.0], dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result2[0]} ✓")

    # Test 3: Zero addition
    logger.info("  Test 3: 10 + 0 = 10")
    A3 = np.array([10.0], dtype=np.float32)
    B3 = np.array([0.0], dtype=np.float32)
    iTList3 = [MockTensor(A3), MockTensor(B3)]
    op3 = MockOp()
    result3 = compute_add(iTList3, op3)
    expected3 = np.array([10.0], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result3[0]} ✓")

    # Test 4: Array addition
    logger.info("  Test 4: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]")
    A4 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    B4 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    iTList4 = [MockTensor(A4), MockTensor(B4)]
    op4 = MockOp()
    result4 = compute_add(iTList4, op4)
    expected4 = np.array([5.0, 7.0, 9.0], dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result4} ✓")

    # Test 5: Broadcast addition
    logger.info("  Test 5: [[1, 2], [3, 4]] + [10, 20] = [[11, 22], [13, 24]]")
    A5 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B5 = np.array([10.0, 20.0], dtype=np.float32)
    iTList5 = [MockTensor(A5), MockTensor(B5)]
    op5 = MockOp()
    result5 = compute_add(iTList5, op5)
    expected5 = np.array([[11.0, 22.0], [13.0, 24.0]], dtype=np.float32)
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result:\n{result5} ✓")

    # Test 6: Fractional numbers
    logger.info("  Test 6: 0.1 + 0.2 ≈ 0.3")
    A6 = np.array([0.1], dtype=np.float32)
    B6 = np.array([0.2], dtype=np.float32)
    iTList6 = [MockTensor(A6), MockTensor(B6)]
    op6 = MockOp()
    result6 = compute_add(iTList6, op6)
    expected6 = np.array([0.3], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-6)
    logger.debug(f"    Result: {result6[0]:.6f} ✓")

    # Test 7: Multiple values
    logger.info("  Test 7: Complex addition")
    A7 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B7 = np.array([[5, 6], [7, 8]], dtype=np.float32)
    iTList7 = [MockTensor(A7), MockTensor(B7)]
    op7 = MockOp()
    result7 = compute_add(iTList7, op7)
    expected7 = np.array([[6, 8], [10, 12]], dtype=np.float32)
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result:\n{result7} ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_add_properties():
    """Test mathematical properties of addition"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Add Mathematical Properties:")

    # Property 1: Commutativity (A + B = B + A)
    logger.info("  Property 1: Commutativity (A + B = B + A)")
    A1 = np.random.randn(3, 4, 5).astype(np.float32)
    B1 = np.random.randn(3, 4, 5).astype(np.float32)

    result_AB = compute_add([MockTensor(A1), MockTensor(B1)], MockOp())
    result_BA = compute_add([MockTensor(B1), MockTensor(A1)], MockOp())

    np.testing.assert_allclose(result_AB, result_BA, rtol=1e-6, atol=1e-7)
    logger.debug("    A + B = B + A ✓")

    # Property 2: Associativity ((A + B) + C = A + (B + C))
    logger.info("  Property 2: Associativity ((A + B) + C = A + (B + C))")
    A2 = np.random.randn(2, 3).astype(np.float32)
    B2 = np.random.randn(2, 3).astype(np.float32)
    C2 = np.random.randn(2, 3).astype(np.float32)

    # (A + B) + C
    AB = compute_add([MockTensor(A2), MockTensor(B2)], MockOp())
    result_left = compute_add([MockTensor(AB), MockTensor(C2)], MockOp())

    # A + (B + C)
    BC = compute_add([MockTensor(B2), MockTensor(C2)], MockOp())
    result_right = compute_add([MockTensor(A2), MockTensor(BC)], MockOp())

    np.testing.assert_allclose(result_left, result_right, rtol=1e-5, atol=1e-6)
    logger.debug("    (A + B) + C = A + (B + C) ✓")

    # Property 3: Identity element (A + 0 = A)
    logger.info("  Property 3: Identity element (A + 0 = A)")
    A3 = np.random.randn(4, 5, 6).astype(np.float32)
    zero = np.zeros_like(A3)

    result3 = compute_add([MockTensor(A3), MockTensor(zero)], MockOp())
    np.testing.assert_allclose(result3, A3, rtol=1e-6, atol=1e-7)
    logger.debug("    A + 0 = A ✓")

    # Property 4: Inverse element (A + (-A) = 0)
    logger.info("  Property 4: Inverse element (A + (-A) = 0)")
    A4 = np.random.randn(3, 4).astype(np.float32)
    neg_A4 = -A4

    result4 = compute_add([MockTensor(A4), MockTensor(neg_A4)], MockOp())
    np.testing.assert_allclose(result4, np.zeros_like(A4), rtol=1e-5, atol=1e-6)
    logger.debug("    A + (-A) = 0 ✓")

    # Property 5: Broadcasting consistency
    logger.info("  Property 5: Broadcasting consistency")
    A5 = np.random.randn(3, 1, 5).astype(np.float32)
    B5 = np.random.randn(3, 4, 1).astype(np.float32)

    result5 = compute_add([MockTensor(A5), MockTensor(B5)], MockOp())
    expected_shape = (3, 4, 5)
    assert (
        result5.shape == expected_shape
    ), f"Expected {expected_shape}, got {result5.shape}"

    # Verify by manual broadcast
    A5_broadcast = np.broadcast_to(A5, (3, 4, 5))
    B5_broadcast = np.broadcast_to(B5, (3, 4, 5))
    expected5 = A5_broadcast + B5_broadcast
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug("    Broadcasting works correctly ✓")

    # Property 6: Shape preservation (when same shape)
    logger.info("  Property 6: Shape preservation")
    test_shapes = [(5,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
    for shape in test_shapes:
        A = np.random.randn(*shape).astype(np.float32)
        B = np.random.randn(*shape).astype(np.float32)
        result = compute_add([MockTensor(A), MockTensor(B)], MockOp())
        assert result.shape == shape, f"Shape not preserved for {shape}"
    logger.debug("    All shapes preserved ✓")

    logger.info("\nAll property tests passed!")


if __name__ == "__main__":
    test_add()
    test_add_errors()
    test_add_precision()
    test_add_properties()
