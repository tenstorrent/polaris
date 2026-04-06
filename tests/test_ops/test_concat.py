#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def ref_impl(shapes, axis):
    """Reference implementation using NumPy concatenate"""
    arrays = [np.random.randn(*shape) for shape in shapes]
    result = np.concatenate(arrays, axis=axis)
    return list(result.shape)


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


# Test cases
test_name = "test_concat"
test_cases = [
    ("2D tensors along axis 0", [[2, 3], [4, 3]], 0, [6, 3]),
    ("2D tensors along axis 1", [[2, 3], [2, 4]], 1, [2, 7]),
    ("Multiple inputs axis 0", [[2, 3], [2, 3], [2, 3]], 0, [6, 3]),
    ("3D tensors along axis 2", [[2, 3, 4], [2, 3, 5]], 2, [2, 3, 9]),
    ("4D tensors along axis 1", [[1, 2, 3, 4], [1, 2, 3, 4]], 1, [1, 4, 3, 4]),
    ("Single element tensors", [[1, 1], [1, 1]], 0, [2, 1]),
    ("Zero-sized dimensions", [[0, 3], [2, 3]], 0, [2, 3]),
    ("Empty dimension on concat axis", [[2, 0], [2, 3]], 1, [2, 3]),
    ("Negative axis -1", [[2, 3], [2, 4]], -1, [2, 7]),
    ("Negative axis -2", [[2, 3], [4, 3]], -2, [6, 3]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_concat():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shapes, axis, expected_shape) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"
        i_tensors = [
            F._from_shape(f"X{i}", shape, np_dtype=np.float32)
            for i, shape in enumerate(input_shapes)
        ]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "Concat",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": axis},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shapes, axis)

        if inf_shape == ref_shape and inf_shape == expected_shape:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug("\t%s", x)
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug("\t%s", x)
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape} (expected {expected_shape})"


# Error test cases
test_name_errors = "test_concat_errors"
test_cases_errors = [
    ("Mismatched ranks", [[2, 3], [2, 3, 4]], 0),
    ("Mismatched non-concat dimensions", [[2, 3], [2, 4]], 0),
    ("Invalid axis", [[2, 3], [2, 3]], 5),
    ("Negative axis out of bounds", [[2, 3], [2, 3]], -5),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_errors():
    """Test Concat with incompatible shapes that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shapes, axis) in enumerate(test_cases_errors):
        op_name = f"{test_name_errors}_{tno}"
        i_tensors = [
            F._from_shape(f"X{i}", shape, np_dtype=np.float32)
            for i, shape in enumerate(input_shapes)
        ]
        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "Concat",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": axis},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
        logger.debug(
            f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)"
        )


# ============================================================================
# Additional tests for numerical validation, precision, and properties
# ============================================================================

from ttsim.ops.desc.data_compute import compute_concat


def ref_impl_concat(arrays, axis):
    """Reference implementation using NumPy concatenate with actual data"""
    return np.concatenate(arrays, axis=axis)


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_numerical():
    """Test concat with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, axis):
            self.attrs = {"axis": axis}

    logger.info("\nTesting Concat Numerical Validation:")

    test_cases_numerical = [
        (
            "2D axis 0",
            [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[5, 6], [7, 8]], dtype=np.float32),
            ],
            0,
        ),
        (
            "2D axis 1",
            [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[5, 6], [7, 8]], dtype=np.float32),
            ],
            1,
        ),
        (
            "3 inputs axis 0",
            [
                np.array([[1, 2]], dtype=np.float32),
                np.array([[3, 4]], dtype=np.float32),
                np.array([[5, 6]], dtype=np.float32),
            ],
            0,
        ),
        (
            "3D axis 2",
            [
                np.random.randn(2, 3, 4).astype(np.float32),
                np.random.randn(2, 3, 5).astype(np.float32),
            ],
            2,
        ),
        (
            "4D axis 1",
            [
                np.random.randn(1, 2, 3, 4).astype(np.float32),
                np.random.randn(1, 3, 3, 4).astype(np.float32),
            ],
            1,
        ),
        (
            "1D axis 0",
            [np.array([1, 2, 3], dtype=np.float32), np.array([4, 5], dtype=np.float32)],
            0,
        ),
        (
            "Multiple 3D",
            [
                np.random.randn(2, 3, 4).astype(np.float32),
                np.random.randn(2, 3, 4).astype(np.float32),
                np.random.randn(2, 3, 4).astype(np.float32),
            ],
            1,
        ),
        (
            "Negative axis -1",
            [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[5], [6]], dtype=np.float32),
            ],
            -1,
        ),
        (
            "Negative axis -2",
            [
                np.array([[1, 2]], dtype=np.float32),
                np.array([[3, 4]], dtype=np.float32),
            ],
            -2,
        ),
        (
            "Single value concat",
            [np.array([[1]], dtype=np.float32), np.array([[2]], dtype=np.float32)],
            0,
        ),
    ]

    passed = 0
    total = len(test_cases_numerical)

    for name, arrays, axis in test_cases_numerical:
        # Create mock objects
        iTList = [MockTensor(arr) for arr in arrays]
        op = MockOp(axis)

        # Compute using the function under test
        result = compute_concat(iTList, op)

        # Compute expected result
        expected = ref_impl_concat(arrays, axis)

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

    logger.info(f"\nConcat Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_precision():
    """Test concat with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, axis):
            self.attrs = {"axis": axis}

    logger.info("\nTesting Concat Precision (Known Outputs):")

    # Test 1: Simple 2D concatenation along axis 0
    logger.debug("  Test 1: [[1, 2], [3, 4]] + [[5, 6]] along axis 0")
    A1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B1 = np.array([[5, 6]], dtype=np.float32)
    iTList1 = [MockTensor(A1), MockTensor(B1)]
    op1 = MockOp(0)
    result1 = compute_concat(iTList1, op1)
    expected1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug("    Result:\n%s ✓", result1)

    # Test 2: Simple 2D concatenation along axis 1
    logger.debug("  Test 2: [[1, 2], [3, 4]] + [[5], [6]] along axis 1")
    A2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B2 = np.array([[5], [6]], dtype=np.float32)
    iTList2 = [MockTensor(A2), MockTensor(B2)]
    op2 = MockOp(1)
    result2 = compute_concat(iTList2, op2)
    expected2 = np.array([[1, 2, 5], [3, 4, 6]], dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug("    Result:\n%s ✓", result2)

    # Test 3: 1D concatenation
    logger.debug("  Test 3: [1, 2, 3] + [4, 5] = [1, 2, 3, 4, 5]")
    A3 = np.array([1, 2, 3], dtype=np.float32)
    B3 = np.array([4, 5], dtype=np.float32)
    iTList3 = [MockTensor(A3), MockTensor(B3)]
    op3 = MockOp(0)
    result3 = compute_concat(iTList3, op3)
    expected3 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug("    Result: %s ✓", result3)

    # Test 4: Three arrays concatenation
    logger.debug("  Test 4: [1] + [2] + [3] = [1, 2, 3]")
    A4 = np.array([1], dtype=np.float32)
    B4 = np.array([2], dtype=np.float32)
    C4 = np.array([3], dtype=np.float32)
    iTList4 = [MockTensor(A4), MockTensor(B4), MockTensor(C4)]
    op4 = MockOp(0)
    result4 = compute_concat(iTList4, op4)
    expected4 = np.array([1, 2, 3], dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug("    Result: %s ✓", result4)

    # Test 5: 3D concatenation
    logger.debug("  Test 5: 3D arrays along axis 1")
    A5 = np.ones((2, 1, 3), dtype=np.float32)
    B5 = np.ones((2, 2, 3), dtype=np.float32) * 2
    iTList5 = [MockTensor(A5), MockTensor(B5)]
    op5 = MockOp(1)
    result5 = compute_concat(iTList5, op5)
    assert result5.shape == (2, 3, 3), f"Expected (2, 3, 3), got {result5.shape}"
    # Check first channel is all 1s, next two channels are all 2s
    np.testing.assert_allclose(result5[:, 0, :], 1.0, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result5[:, 1:, :], 2.0, rtol=1e-6, atol=1e-7)
    logger.debug("    Result shape: %s ✓", result5.shape)

    # Test 6: Negative axis
    logger.debug("  Test 6: [1, 2] + [3, 4] with axis=-1 (same as axis=0)")
    A6 = np.array([1, 2], dtype=np.float32)
    B6 = np.array([3, 4], dtype=np.float32)
    iTList6 = [MockTensor(A6), MockTensor(B6)]
    op6 = MockOp(-1)
    result6 = compute_concat(iTList6, op6)
    expected6 = np.array([1, 2, 3, 4], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug("    Result: %s ✓", result6)

    # Test 7: Multiple arrays with specific values
    logger.debug("  Test 7: Concatenate identity patterns")
    A7 = np.array([[1, 0], [0, 1]], dtype=np.float32)
    B7 = np.array([[0, 1], [1, 0]], dtype=np.float32)
    iTList7 = [MockTensor(A7), MockTensor(B7)]
    op7 = MockOp(0)
    result7 = compute_concat(iTList7, op7)
    expected7 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    logger.debug("    Result:\n%s ✓", result7)

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_concat_properties():
    """Test mathematical properties of concatenation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, axis):
            self.attrs = {"axis": axis}

    logger.info("\nTesting Concat Mathematical Properties:")

    # Property 1: Concatenation preserves data order
    logger.debug("  Property 1: Data order preservation")
    A1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    B1 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    result1 = compute_concat([MockTensor(A1), MockTensor(B1)], MockOp(0))

    # First part should match A1
    np.testing.assert_allclose(result1[:2, :], A1, rtol=1e-6, atol=1e-7)
    # Second part should match B1
    np.testing.assert_allclose(result1[2:, :], B1, rtol=1e-6, atol=1e-7)
    logger.debug("    Order preserved: A then B ✓")

    # Property 2: Associativity (with same axis)
    logger.debug("  Property 2: Associativity ((A+B)+C = A+(B+C))")
    A2 = np.random.randn(2, 3).astype(np.float32)
    B2 = np.random.randn(2, 3).astype(np.float32)
    C2 = np.random.randn(2, 3).astype(np.float32)

    # (A + B) + C
    AB = compute_concat([MockTensor(A2), MockTensor(B2)], MockOp(0))
    result_left = compute_concat([MockTensor(AB), MockTensor(C2)], MockOp(0))

    # A + (B + C)
    BC = compute_concat([MockTensor(B2), MockTensor(C2)], MockOp(0))
    result_right = compute_concat([MockTensor(A2), MockTensor(BC)], MockOp(0))

    np.testing.assert_allclose(result_left, result_right, rtol=1e-5, atol=1e-6)
    logger.debug("    (A+B)+C = A+(B+C) ✓")

    # Property 3: Concatenating with empty produces original (for non-concat dims)
    logger.debug("  Property 3: Identity with appropriate empty tensor")
    A3 = np.random.randn(3, 4).astype(np.float32)
    B3 = np.zeros((0, 4), dtype=np.float32)  # Empty along concat axis

    result3 = compute_concat([MockTensor(A3), MockTensor(B3)], MockOp(0))
    np.testing.assert_allclose(result3, A3, rtol=1e-6, atol=1e-7)
    logger.debug("    A + empty = A ✓")

    # Property 4: Dimension size adds up along concat axis
    logger.debug("  Property 4: Dimension summation along concat axis")
    A4 = np.random.randn(2, 3, 4).astype(np.float32)
    B4 = np.random.randn(2, 5, 4).astype(np.float32)
    C4 = np.random.randn(2, 7, 4).astype(np.float32)

    result4 = compute_concat(
        [MockTensor(A4), MockTensor(B4), MockTensor(C4)], MockOp(1)
    )
    expected_shape = (2, 3 + 5 + 7, 4)
    assert (
        result4.shape == expected_shape
    ), f"Expected {expected_shape}, got {result4.shape}"
    logger.debug("    Concat axis size = 3+5+7 = 15 ✓")

    # Property 5: Non-concat dimensions remain unchanged
    logger.debug("  Property 5: Non-concat dimensions preserved")
    test_shapes = [
        ([(2, 3, 4), (2, 5, 4)], 1, (2, 8, 4)),
        ([(1, 2, 3), (1, 2, 3)], 0, (2, 2, 3)),
        ([(4, 5, 6, 7), (4, 5, 6, 9)], 3, (4, 5, 6, 16)),
    ]

    for shapes, axis, expected_shape in test_shapes:
        arrays = [np.random.randn(*s).astype(np.float32) for s in shapes]
        result = compute_concat([MockTensor(arr) for arr in arrays], MockOp(axis))
        assert (
            result.shape == expected_shape
        ), f"Expected {expected_shape}, got {result.shape}"

        # Verify non-concat dimensions are unchanged
        for i, orig_shape in enumerate(shapes):
            for dim_idx in range(len(orig_shape)):
                if dim_idx != axis % len(orig_shape):
                    assert result.shape[dim_idx] == orig_shape[dim_idx]

    logger.debug("    All non-concat dimensions preserved ✓")

    # Property 6: Concatenation is deterministic
    logger.debug("  Property 6: Deterministic operation")
    A6 = np.random.randn(3, 4).astype(np.float32)
    B6 = np.random.randn(3, 4).astype(np.float32)

    result6_1 = compute_concat([MockTensor(A6), MockTensor(B6)], MockOp(0))
    result6_2 = compute_concat([MockTensor(A6), MockTensor(B6)], MockOp(0))

    np.testing.assert_allclose(result6_1, result6_2, rtol=1e-6, atol=1e-7)
    logger.debug("    Same inputs produce same output ✓")

    logger.info("\nAll property tests passed!")
