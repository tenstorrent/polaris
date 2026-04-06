#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test mish activation operation"""

import numpy as np
import pytest
from loguru import logger

from ttsim.ops.desc.data_compute import compute_mish


def ref_impl_mish(X):
    """
    Reference implementation of Mish activation.
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    Properties:
    - Smooth, non-monotonic activation
    - Self-gated (output depends on input non-linearly)
    - Range: approximately [-0.31, ∞)
    - Asymptotic: approaches 0 for large negative x, approaches x for large positive x
    """
    # Clip to prevent overflow in exp
    X_clipped = np.clip(X, -20, 20)
    # softplus(x) = ln(1 + e^x) - use log1p for numerical stability
    softplus = np.log1p(np.exp(X_clipped))
    return X * np.tanh(softplus)


def generate_test_data():
    """Generate comprehensive test cases for mish activation"""
    test_cases = []

    # Basic small inputs
    test_cases.append(("small_1d", np.array([1.0, 2.0, 3.0], dtype=np.float32)))
    test_cases.append(
        ("small_2d", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    )

    # Zero input (mish(0) = 0)
    test_cases.append(("zero", np.zeros((3, 3), dtype=np.float32)))

    # Positive values
    test_cases.append(
        ("positive_small", np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32))
    )
    test_cases.append(
        ("positive_medium", np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    )
    test_cases.append(
        ("positive_large", np.array([8.0, 10.0, 15.0, 20.0], dtype=np.float32))
    )

    # Negative values
    test_cases.append(
        ("negative_small", np.array([-0.5, -1.0, -1.5, -2.0], dtype=np.float32))
    )
    test_cases.append(
        ("negative_medium", np.array([-3.0, -4.0, -5.0, -6.0], dtype=np.float32))
    )
    test_cases.append(
        ("negative_large", np.array([-8.0, -10.0, -15.0, -20.0], dtype=np.float32))
    )

    # Values near minimum point (around -1.2 where mish reaches ~-0.31)
    test_cases.append(
        ("near_minimum", np.array([-0.8, -1.0, -1.2, -1.4, -1.6], dtype=np.float32))
    )

    # Mixed positive and negative
    test_cases.append(
        ("mixed", np.array([-5.0, -2.0, 0.0, 2.0, 5.0], dtype=np.float32))
    )
    test_cases.append(
        ("mixed_2d", np.array([[-3.0, -1.0], [1.0, 3.0]], dtype=np.float32))
    )

    # Boundary values (clipping at -20, 20)
    test_cases.append(
        ("at_boundaries", np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=np.float32))
    )

    # Large positive values (should approach x, since tanh(softplus(x)) → 1)
    test_cases.append(
        ("large_positive", np.array([10.0, 12.0, 15.0, 18.0], dtype=np.float32))
    )

    # Large negative values (should approach 0, since x * e^x → 0)
    test_cases.append(
        ("large_negative", np.array([-10.0, -12.0, -15.0, -18.0], dtype=np.float32))
    )

    # Different tensor shapes
    test_cases.append(("shape_3d", np.random.randn(2, 3, 4).astype(np.float32)))
    test_cases.append(("shape_4d", np.random.randn(1, 2, 3, 4).astype(np.float32)))

    # Batch processing
    test_cases.append(("batch_16", np.random.randn(16, 10).astype(np.float32)))
    test_cases.append(("batch_32", np.random.randn(32, 20).astype(np.float32)))

    # Very small values (testing numerical stability near zero)
    test_cases.append(
        (
            "very_small",
            np.array([0.001, 0.01, 0.1, -0.001, -0.01, -0.1], dtype=np.float32),
        )
    )

    # Self-gated behavior test (around transition regions)
    test_cases.append(("transition_positive", np.linspace(0, 5, 10, dtype=np.float32)))
    test_cases.append(("transition_negative", np.linspace(-5, 0, 10, dtype=np.float32)))

    # Asymptotic behavior
    test_cases.append(
        ("asymptotic_pos", np.array([7.0, 8.0, 9.0, 10.0], dtype=np.float32))
    )
    test_cases.append(
        ("asymptotic_neg", np.array([-7.0, -8.0, -9.0, -10.0], dtype=np.float32))
    )

    return test_cases


@pytest.mark.unit
@pytest.mark.opunit
def test_mish():
    """Test mish activation with shape and numerical validation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    test_cases = generate_test_data()
    passed = 0
    total = len(test_cases)

    for name, X in test_cases:
        # Create mock objects
        iTList = [MockTensor(X)]
        op = MockOp()

        # Compute using the function under test
        result = compute_mish(iTList, op)

        # Compute expected result
        expected = ref_impl_mish(X)

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

    logger.info(f"\nMish Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_mish_errors():
    """Test mish activation edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Mish Edge Cases:")

    # Test 1: Extreme positive values (beyond clipping boundary)
    logger.info("  Test 1: Extreme positive values (overflow territory)")
    X1 = np.array([50.0, 100.0, 500.0], dtype=np.float32)
    iTList1 = [MockTensor(X1)]
    op1 = MockOp()
    result1 = compute_mish(iTList1, op1)
    expected1 = ref_impl_mish(X1)
    # Due to clipping at 20, extreme values should behave like mish(20)
    # which is approximately 20.0 (since tanh(softplus(20)) ≈ 1)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-7)
    logger.info("    PASS (returns clipped behavior)")

    # Test 2: Extreme negative values (far beyond clipping boundary)
    logger.info("  Test 2: Extreme negative values")
    X2 = np.array([-50.0, -100.0, -500.0], dtype=np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp()
    result2 = compute_mish(iTList2, op2)
    expected2 = ref_impl_mish(X2)
    # Due to clipping at -20, extreme negative values should behave like mish(-20)
    # which is approximately -20 * e^-20 ≈ 0
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-7)
    logger.info("    PASS (returns clipped behavior approaching 0)")

    # Test 3: Mixed extreme values
    logger.info("  Test 3: Mixed extreme positive and negative")
    X3 = np.array([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=np.float32)
    iTList3 = [MockTensor(X3)]
    op3 = MockOp()
    result3 = compute_mish(iTList3, op3)
    expected3 = ref_impl_mish(X3)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-7)
    logger.info("    PASS (handles mixed extreme values)")

    # Test 4: Exact clipping boundaries
    logger.info("  Test 4: Values at exact clipping boundaries (-20, 20)")
    X4 = np.array([-20.0, -19.9, 19.9, 20.0], dtype=np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp()
    result4 = compute_mish(iTList4, op4)
    expected4 = ref_impl_mish(X4)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-7)
    logger.info("    PASS (boundary values handled correctly)")

    # Test 5: Near minimum point (mish has minimum around -0.31 at x ≈ -1.2)
    logger.info("  Test 5: Behavior around minimum point (x ≈ -1.2)")
    X5 = np.array([-1.5, -1.3, -1.2, -1.1, -0.9], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp()
    result5 = compute_mish(iTList5, op5)
    expected5 = ref_impl_mish(X5)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-7)
    # Verify minimum is around -0.31
    assert (
        np.min(result5) < 0 and np.min(result5) > -0.35
    ), f"Minimum value should be around -0.31, got {np.min(result5)}"
    logger.info("    PASS (minimum point behavior correct)")

    # Test 6: Asymptotic behavior for large positive (should approach x)
    logger.info("  Test 6: Asymptotic behavior (large positive → x)")
    X6 = np.array([15.0, 18.0, 20.0], dtype=np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp()
    result6 = compute_mish(iTList6, op6)
    expected6 = ref_impl_mish(X6)
    assert result6.shape == expected6.shape
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-7)
    # For large x, mish(x) should be very close to x
    np.testing.assert_allclose(result6, X6, rtol=0.01, atol=0.01)
    logger.info("    PASS (approaches x for large positive values)")

    # Test 7: Asymptotic behavior for large negative (should approach 0)
    logger.info("  Test 7: Asymptotic behavior (large negative → 0)")
    X7 = np.array([-15.0, -18.0, -20.0], dtype=np.float32)
    iTList7 = [MockTensor(X7)]
    op7 = MockOp()
    result7 = compute_mish(iTList7, op7)
    expected7 = ref_impl_mish(X7)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-7)
    # For large negative x, mish(x) should be very close to 0
    assert np.all(
        np.abs(result7) < 1e-5
    ), f"Large negative values should approach 0, got {result7}"
    logger.info("    PASS (approaches 0 for large negative values)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_mish_precision():
    """Test mish activation with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Mish Precision (Known Outputs):")

    # Test 1: mish(0) = 0
    logger.info("  Test 1: mish(0) = 0")
    X1 = np.array([0.0], dtype=np.float32)
    iTList1 = [MockTensor(X1)]
    op1 = MockOp()
    result1 = compute_mish(iTList1, op1)
    expected1 = np.array([0.0], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.info(f"    mish(0.0) = {result1[0]:.6f} ✓")

    # Test 2: mish(1) ≈ 0.8651
    logger.info("  Test 2: mish(1) ≈ 0.8651")
    X2 = np.array([1.0], dtype=np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp()
    result2 = compute_mish(iTList2, op2)
    expected2 = np.array([0.8650983882673684], dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-6)
    logger.info(f"    mish(1.0) = {result2[0]:.6f} ✓")

    # Test 3: mish(-1) ≈ -0.3034 (near minimum)
    logger.info("  Test 3: mish(-1) ≈ -0.3034")
    X3 = np.array([-1.0], dtype=np.float32)
    iTList3 = [MockTensor(X3)]
    op3 = MockOp()
    result3 = compute_mish(iTList3, op3)
    expected3 = np.array([-0.30340147692322535], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-6)
    logger.info(f"    mish(-1.0) = {result3[0]:.6f} ✓")

    # Test 4: mish(5) ≈ 4.998 (approaching x)
    logger.info("  Test 4: mish(5) ≈ 4.998")
    X4 = np.array([5.0], dtype=np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp()
    result4 = compute_mish(iTList4, op4)
    expected4 = ref_impl_mish(X4)
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-6)
    logger.info(f"    mish(5.0) = {result4[0]:.6f} ✓")

    # Test 5: mish(-5) ≈ -0.0034 (approaching 0)
    logger.info("  Test 5: mish(-5) ≈ -0.0034")
    X5 = np.array([-5.0], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp()
    result5 = compute_mish(iTList5, op5)
    expected5 = ref_impl_mish(X5)
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-6)
    logger.info(f"    mish(-5.0) = {result5[0]:.6f} ✓")

    # Test 6: mish(2) ≈ 1.9439
    logger.info("  Test 6: mish(2) ≈ 1.9439")
    X6 = np.array([2.0], dtype=np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp()
    result6 = compute_mish(iTList6, op6)
    expected6 = ref_impl_mish(X6)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.info(f"    mish(2.0) = {result6[0]:.6f} ✓")

    # Test 7: Multiple values
    logger.info("  Test 7: Multiple known values")
    X7 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    iTList7 = [MockTensor(X7)]
    op7 = MockOp()
    result7 = compute_mish(iTList7, op7)
    expected7 = ref_impl_mish(X7)
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-6)
    logger.info(f"    mish([-2, -1, 0, 1, 2]) = {result7} ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_mish_properties():
    """Test mathematical properties of mish activation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Mish Mathematical Properties:")

    # Property 1: Smoothness (continuous everywhere)
    logger.info("  Property 1: Smoothness check (continuous)")
    X1 = np.linspace(-5, 5, 100, dtype=np.float32)
    iTList1 = [MockTensor(X1)]
    op1 = MockOp()
    result1 = compute_mish(iTList1, op1)
    # Check no sudden jumps (gradient should be bounded)
    diff = np.diff(result1)
    max_diff = np.max(np.abs(diff))
    assert max_diff < 2.0, f"Smoothness violated: max diff = {max_diff}"
    logger.info(f"    Max gradient step: {max_diff:.6f} ✓")

    # Property 2: Non-monotonic (has a minimum around x ≈ -1.2)
    logger.info("  Property 2: Non-monotonic behavior")
    X2 = np.linspace(-3, 0, 50, dtype=np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp()
    result2 = compute_mish(iTList2, op2)
    min_idx = np.argmin(result2)
    min_x = X2[min_idx]
    min_val = result2[min_idx]
    # Minimum should be around x ≈ -1.2, value ≈ -0.31
    assert -1.5 < min_x < -0.8, f"Minimum x should be around -1.2, got {min_x}"
    assert (
        -0.35 < min_val < -0.25
    ), f"Minimum value should be around -0.31, got {min_val}"
    logger.info(f"    Minimum at x={min_x:.3f}, value={min_val:.6f} ✓")

    # Property 3: Self-gated (output depends on input non-linearly)
    logger.info("  Property 3: Self-gated nature")
    X3_pos = np.array([10.0], dtype=np.float32)
    X3_neg = np.array([-10.0], dtype=np.float32)
    result3_pos = compute_mish([MockTensor(X3_pos)], MockOp())
    result3_neg = compute_mish([MockTensor(X3_neg)], MockOp())
    # For large positive: mish(x) ≈ x
    # For large negative: mish(x) ≈ 0
    assert abs(result3_pos[0] - X3_pos[0]) < 0.1, "Large positive should ≈ x"
    assert abs(result3_neg[0]) < 0.01, "Large negative should ≈ 0"
    logger.info(
        f"    mish(10)≈10: {result3_pos[0]:.3f}, mish(-10)≈0: {result3_neg[0]:.6f} ✓"
    )

    # Property 4: Shape preservation
    logger.info("  Property 4: Shape preservation")
    shapes = [(5,), (3, 4), (2, 3, 4), (1, 2, 3, 4)]
    for shape in shapes:
        X = np.random.randn(*shape).astype(np.float32)
        result = compute_mish([MockTensor(X)], MockOp())
        assert result.shape == X.shape, f"Shape not preserved for {shape}"
    logger.info("    All shapes preserved ✓")

    logger.info("\nAll property tests passed!")


if __name__ == "__main__":
    test_mish()
    test_mish_errors()
    test_mish_precision()
    test_mish_properties()
