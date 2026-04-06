#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test bbox_size_decode operation for YOLO-style object detection"""

import numpy as np
import pytest
import warnings
from loguru import logger

from ttsim.ops.desc.data_compute import compute_bbox_size_decode


def ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid):
    """
    Reference implementation of bbox size decoding.
    Formula: ((sigmoid(wh) * 2.0) ** 2) * anchor_grid

    This operation is used in YOLO-style object detection to convert
    predicted width/height offsets (already sigmoid-activated) into actual box dimensions.

    Args:
        wh_sigmoid: [bs, na, ny, nx, 2] - sigmoid activated wh predictions (0-1 range)
        anchor_grid: [1, na, 1, 1, 2] - anchor dimensions for this detection layer

    Returns:
        wh_decoded: [bs, na, ny, nx, 2] - decoded wh dimensions in image space
    """
    return ((wh_sigmoid * 2.0) ** 2) * anchor_grid


def generate_test_data():
    """Generate comprehensive test cases for bbox_size_decode"""
    test_cases = []

    # Test 1: YOLO small object detection (13x13 grid, large anchors)
    bs, na, ny, nx = 1, 3, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("yolo_13x13_large_anchors", wh_sigmoid, anchor_grid))

    # Test 2: YOLO medium object detection (26x26 grid, medium anchors)
    bs, na, ny, nx = 1, 3, 26, 26
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("yolo_26x26_medium_anchors", wh_sigmoid, anchor_grid))

    # Test 3: YOLO large object detection (52x52 grid, small anchors)
    bs, na, ny, nx = 1, 3, 52, 52
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("yolo_52x52_small_anchors", wh_sigmoid, anchor_grid))

    # Test 4: Very fine scale (80x80 grid, tiny anchors)
    bs, na, ny, nx = 1, 3, 80, 80
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[5.0, 7.0], [8.0, 15.0], [12.0, 10.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("yolo_80x80_tiny_anchors", wh_sigmoid, anchor_grid))

    # Test 5: Small grid (4x4)
    bs, na, ny, nx = 1, 3, 4, 4
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 100
    test_cases.append(("small_4x4", wh_sigmoid, anchor_grid))

    # Test 6: Batch processing (batch size 4)
    bs, na, ny, nx = 4, 3, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 200
    test_cases.append(("batch_4", wh_sigmoid, anchor_grid))

    # Test 7: Larger batch (batch size 8)
    bs, na, ny, nx = 8, 3, 26, 26
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 150
    test_cases.append(("batch_8", wh_sigmoid, anchor_grid))

    # Test 8: Single anchor
    bs, na, ny, nx = 1, 1, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.array([[[[[100.0, 100.0]]]]]).astype(np.float32)
    test_cases.append(("single_anchor", wh_sigmoid, anchor_grid))

    # Test 9: Many anchors (5 anchors)
    bs, na, ny, nx = 1, 5, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 200
    test_cases.append(("five_anchors", wh_sigmoid, anchor_grid))

    # Test 10: Sigmoid at 0.5 (center value)
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.full((bs, na, ny, nx, 2), 0.5, dtype=np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("center_sigmoid", wh_sigmoid, anchor_grid))

    # Test 11: Sigmoid at 0 (minimum size)
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("sigmoid_zero", wh_sigmoid, anchor_grid))

    # Test 12: Sigmoid at 1 (maximum size in this formulation)
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.ones((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("sigmoid_one", wh_sigmoid, anchor_grid))

    # Test 13: Different width and height anchors
    bs, na, ny, nx = 1, 3, 10, 10
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[200.0, 100.0], [100.0, 200.0], [150.0, 150.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("asymmetric_anchors", wh_sigmoid, anchor_grid))

    # Test 14: Very large anchors
    bs, na, ny, nx = 1, 3, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[500.0, 600.0], [800.0, 900.0], [1200.0, 1300.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("large_anchors", wh_sigmoid, anchor_grid))

    # Test 15: Small anchors
    bs, na, ny, nx = 1, 3, 52, 52
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("small_anchors", wh_sigmoid, anchor_grid))

    # Test 16: Single cell grid (1x1)
    bs, na, ny, nx = 1, 3, 1, 1
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("single_cell", wh_sigmoid, anchor_grid))

    # Test 17: Rectangular grid (non-square)
    bs, na, ny, nx = 1, 3, 10, 20
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("rectangular_10x20", wh_sigmoid, anchor_grid))

    # Test 18: Another rectangular (20x10)
    bs, na, ny, nx = 1, 3, 20, 10
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("rectangular_20x10", wh_sigmoid, anchor_grid))

    # Test 19: Mixed sigmoid values
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    wh_sigmoid[0, 0, 0, 0, :] = [0.0, 0.0]  # min
    wh_sigmoid[0, 0, 0, -1, :] = [1.0, 1.0]  # max
    wh_sigmoid[0, 0, -1, 0, :] = [0.5, 0.5]  # center
    wh_sigmoid[0, 0, -1, -1, :] = [0.25, 0.75]  # mixed
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("mixed_sigmoids", wh_sigmoid, anchor_grid))

    # Test 20: Square anchors (width == height)
    bs, na, ny, nx = 1, 3, 16, 16
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[64.0, 64.0], [128.0, 128.0], [256.0, 256.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    test_cases.append(("square_anchors", wh_sigmoid, anchor_grid))

    return test_cases


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_size_decode():
    """Test bbox_size_decode with shape and numerical validation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    test_cases = generate_test_data()
    passed = 0
    total = len(test_cases)

    for name, wh_sigmoid, anchor_grid in test_cases:
        # Create mock objects
        iTList = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
        op = MockOp()

        # Compute using the function under test
        result = compute_bbox_size_decode(iTList, op)

        # Compute expected result
        expected = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"
        assert (
            result.shape == wh_sigmoid.shape
        ), f"{name}: Output shape should match wh_sigmoid shape"

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

    logger.info(f"\nBbox Size Decode Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_size_decode_errors():
    """Test bbox_size_decode edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Size Decode Edge Cases:")

    # Suppress warnings for zero/negative anchor tests
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Test 1: Zero anchors (should produce zeros)
    logger.debug("  Test 1: Zero anchors")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.zeros((1, na, 1, 1, 2), dtype=np.float32)
    iTList1 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op1 = MockOp()
    result1 = compute_bbox_size_decode(iTList1, op1)
    expected1 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-7)
    assert np.all(result1 == 0.0), "Zero anchors should produce all zeros"
    logger.debug("    PASS (zero anchors produce zeros)")

    # Test 2: Negative anchors
    logger.debug("  Test 2: Negative anchors")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[-50.0, -60.0], [-80.0, -90.0], [-120.0, -130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    iTList2 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op2 = MockOp()
    result2 = compute_bbox_size_decode(iTList2, op2)
    expected2 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-7)
    # Negative anchors should produce negative dimensions
    logger.debug("    PASS (negative anchors handled correctly)")

    # Test 3: Very large anchors
    logger.debug("  Test 3: Very large anchors")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[10000.0, 10000.0], [20000.0, 20000.0], [30000.0, 30000.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    iTList3 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op3 = MockOp()
    result3 = compute_bbox_size_decode(iTList3, op3)
    expected3 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (very large anchors handled correctly)")

    # Test 4: All zeros (sigmoid=0, anchor=0)
    logger.debug("  Test 4: All zeros")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = np.zeros((1, na, 1, 1, 2), dtype=np.float32)
    iTList4 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op4 = MockOp()
    result4 = compute_bbox_size_decode(iTList4, op4)
    expected4 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-7)
    assert np.all(result4 == 0.0), "All zeros should produce zero output"
    logger.debug("    PASS (all zeros produces zero output)")

    # Test 5: Sigmoid = 0 with non-zero anchors (minimum box size)
    logger.debug("  Test 5: Sigmoid = 0 (minimum box size)")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = (
        np.array([[[[[100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    iTList5 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op5 = MockOp()
    result5 = compute_bbox_size_decode(iTList5, op5)
    expected5 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-7)
    # Formula: ((0 * 2.0) ** 2) * anchor = 0
    assert np.all(result5 == 0.0), "Sigmoid=0 should produce zero output"
    logger.debug("    PASS (sigmoid=0 produces minimum size)")

    # Test 6: Sigmoid = 1 (maximum size in this range)
    logger.debug("  Test 6: Sigmoid = 1 (maximum box size)")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.ones((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = (
        np.array([[[[[100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    iTList6 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op6 = MockOp()
    result6 = compute_bbox_size_decode(iTList6, op6)
    expected6 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result6.shape == expected6.shape
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-7)
    # Formula: ((1 * 2.0) ** 2) * anchor = 4 * anchor
    # result6 should be 4x the anchor values (broadcast to full shape)
    expected_max_per_anchor = 4.0 * anchor_grid
    assert np.allclose(
        result6[0, 0, 0, 0], expected_max_per_anchor[0, 0, 0, 0]
    ), "First anchor should be 4x"
    assert np.allclose(
        result6[0, 1, 0, 0], expected_max_per_anchor[0, 1, 0, 0]
    ), "Second anchor should be 4x"
    assert np.allclose(
        result6[0, 2, 0, 0], expected_max_per_anchor[0, 2, 0, 0]
    ), "Third anchor should be 4x"
    logger.debug("    PASS (sigmoid=1 produces 4x anchor size)")

    # Test 7: Single cell with extreme values
    logger.debug("  Test 7: Single cell with extreme values")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[1.0, 1.0]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[5000.0, 5000.0]]]]], dtype=np.float32)
    iTList7 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op7 = MockOp()
    result7 = compute_bbox_size_decode(iTList7, op7)
    expected7 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (single cell with extreme values)")

    # Test 8: Very small anchors (fractional)
    logger.debug("  Test 8: Very small anchors (fractional)")
    bs, na, ny, nx = 1, 3, 8, 8
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = (
        np.array([[[[[0.1, 0.1], [0.5, 0.5], [1.0, 1.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    iTList8 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op8 = MockOp()
    result8 = compute_bbox_size_decode(iTList8, op8)
    expected8 = ref_impl_bbox_size_decode(wh_sigmoid, anchor_grid)
    assert result8.shape == expected8.shape
    np.testing.assert_allclose(result8, expected8, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (small fractional anchors)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_size_decode_precision():
    """Test bbox_size_decode with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Size Decode Precision (Known Outputs):")

    # Test 1: sigmoid=0.5, anchor=100 -> ((0.5*2)^2)*100 = 1*100 = 100
    logger.debug("  Test 1: sigmoid=0.5, anchor=100")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[0.5, 0.5]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[100.0, 100.0]]]]], dtype=np.float32)
    iTList1 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op1 = MockOp()
    result1 = compute_bbox_size_decode(iTList1, op1)
    expected1 = np.array([[[[[100.0, 100.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result1[0,0,0,0]} (expected [100.0, 100.0]) ✓")

    # Test 2: sigmoid=0, anchor=100 -> ((0*2)^2)*100 = 0
    logger.debug("  Test 2: sigmoid=0, anchor=100")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[100.0, 100.0]]]]], dtype=np.float32)
    iTList2 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op2 = MockOp()
    result2 = compute_bbox_size_decode(iTList2, op2)
    expected2 = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result2[0,0,0,0]} (expected [0.0, 0.0]) ✓")

    # Test 3: sigmoid=1, anchor=100 -> ((1*2)^2)*100 = 4*100 = 400
    logger.debug("  Test 3: sigmoid=1, anchor=100")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[1.0, 1.0]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[100.0, 100.0]]]]], dtype=np.float32)
    iTList3 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op3 = MockOp()
    result3 = compute_bbox_size_decode(iTList3, op3)
    expected3 = np.array([[[[[400.0, 400.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result3[0,0,0,0]} (expected [400.0, 400.0]) ✓")

    # Test 4: sigmoid=0.25, anchor=64 -> ((0.25*2)^2)*64 = 0.25*64 = 16
    logger.debug("  Test 4: sigmoid=0.25, anchor=64")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[0.25, 0.25]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[64.0, 64.0]]]]], dtype=np.float32)
    iTList4 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op4 = MockOp()
    result4 = compute_bbox_size_decode(iTList4, op4)
    expected4 = np.array([[[[[16.0, 16.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result4[0,0,0,0]} (expected [16.0, 16.0]) ✓")

    # Test 5: sigmoid=0.75, anchor=80 -> ((0.75*2)^2)*80 = 2.25*80 = 180
    logger.debug("  Test 5: sigmoid=0.75, anchor=80")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[0.75, 0.75]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[80.0, 80.0]]]]], dtype=np.float32)
    iTList5 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op5 = MockOp()
    result5 = compute_bbox_size_decode(iTList5, op5)
    expected5 = np.array([[[[[180.0, 180.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result5[0,0,0,0]} (expected [180.0, 180.0]) ✓")

    # Test 6: Different width and height
    # w: (0.5*2)^2 * 100 = 100, h: (0.25*2)^2 * 50 = 12.5
    logger.debug("  Test 6: Different width and height")
    bs, na, ny, nx = 1, 1, 1, 1
    wh_sigmoid = np.array([[[[[0.5, 0.25]]]]], dtype=np.float32)
    anchor_grid = np.array([[[[[100.0, 50.0]]]]], dtype=np.float32)
    iTList6 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op6 = MockOp()
    result6 = compute_bbox_size_decode(iTList6, op6)
    expected6 = np.array([[[[[100.0, 12.5]]]]], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug(f"    Result: {result6[0,0,0,0]} (expected [100.0, 12.5]) ✓")

    # Test 7: Multiple anchors with different sizes
    logger.debug("  Test 7: Three anchors with known values")
    bs, na, ny, nx = 1, 3, 1, 1
    # Create sigmoid values properly shaped: [bs, na, ny, nx, 2]
    wh_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    wh_sigmoid[0, 0, 0, 0, :] = [0.5, 0.5]
    wh_sigmoid[0, 1, 0, 0, :] = [0.25, 0.75]
    wh_sigmoid[0, 2, 0, 0, :] = [1.0, 0.0]
    # Create anchor_grid properly: [1, na, 1, 1, 2]
    anchor_grid = np.zeros((1, 3, 1, 1, 2), dtype=np.float32)
    anchor_grid[0, 0, 0, 0] = [100.0, 100.0]
    anchor_grid[0, 1, 0, 0] = [64.0, 80.0]
    anchor_grid[0, 2, 0, 0] = [200.0, 50.0]
    iTList7 = [MockTensor(wh_sigmoid), MockTensor(anchor_grid)]
    op7 = MockOp()
    result7 = compute_bbox_size_decode(iTList7, op7)
    # Anchor 0: (0.5*2)^2 * [100,100] = [100,100]
    # Anchor 1: (0.25*2)^2 * 64 = 16, (0.75*2)^2 * 80 = 180 -> [16,180]
    # Anchor 2: (1*2)^2 * 200 = 800, (0*2)^2 * 50 = 0 -> [800,0]
    np.testing.assert_allclose(
        result7[0, 0, 0, 0], [100.0, 100.0], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(result7[0, 1, 0, 0], [16.0, 180.0], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result7[0, 2, 0, 0], [800.0, 0.0], rtol=1e-6, atol=1e-7)
    logger.debug(f"    Anchor 0: {result7[0,0,0,0]} ✓")
    logger.debug(f"    Anchor 1: {result7[0,1,0,0]} ✓")
    logger.debug(f"    Anchor 2: {result7[0,2,0,0]} ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_size_decode_properties():
    """Test mathematical properties of bbox_size_decode"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Size Decode Properties:")

    # Property 1: Output is always non-negative (for positive anchors)
    logger.debug("  Property 1: Non-negative output for positive anchors")
    bs, na, ny, nx = 2, 3, 13, 13
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    anchor_grid = np.abs(np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 200)
    result = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid), MockTensor(anchor_grid)], MockOp()
    )
    assert np.all(result >= 0), "Output should be non-negative for positive anchors"
    logger.debug(f"    All outputs non-negative ✓")

    # Property 2: Linear scaling with anchor size
    # Doubling anchor should double output
    logger.debug("  Property 2: Linear scaling with anchor size")
    bs, na, ny, nx = 1, 3, 10, 10
    wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)

    anchor1 = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    result1 = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid), MockTensor(anchor1)], MockOp()
    )

    anchor2 = anchor1 * 2.0
    result2 = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid), MockTensor(anchor2)], MockOp()
    )

    # result2 should be result1 * 2
    np.testing.assert_allclose(result2, result1 * 2.0, rtol=1e-5, atol=1e-7)
    logger.debug(f"    Doubling anchor doubles output ✓")

    # Property 3: Quadratic scaling with sigmoid
    # sigmoid=0.5 -> scale=1x, sigmoid=1.0 -> scale=4x
    logger.debug("  Property 3: Quadratic scaling with sigmoid")
    bs, na, ny, nx = 1, 3, 8, 8
    anchor_grid = (
        np.array([[[[[100.0, 100.0], [150.0, 150.0], [200.0, 200.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )

    wh_sigmoid1 = np.full((bs, na, ny, nx, 2), 0.5, dtype=np.float32)
    result1 = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid1), MockTensor(anchor_grid)], MockOp()
    )

    wh_sigmoid2 = np.full((bs, na, ny, nx, 2), 1.0, dtype=np.float32)
    result2 = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid2), MockTensor(anchor_grid)], MockOp()
    )

    # sigmoid=0.5: ((0.5*2)^2) = 1, sigmoid=1.0: ((1*2)^2) = 4
    # So result2 should be result1 * 4
    np.testing.assert_allclose(result2, result1 * 4.0, rtol=1e-5, atol=1e-7)
    logger.debug(f"    Quadratic relationship verified ✓")

    # Property 4: Maximum is 4x anchor (when sigmoid=1)
    logger.debug("  Property 4: Maximum output is 4x anchor")
    bs, na, ny, nx = 1, 3, 10, 10
    wh_sigmoid = np.ones((bs, na, ny, nx, 2), dtype=np.float32)
    anchor_grid = (
        np.array([[[[[50.0, 60.0], [80.0, 90.0], [120.0, 130.0]]]]])
        .T.reshape(1, 3, 1, 1, 2)
        .astype(np.float32)
    )
    result = compute_bbox_size_decode(
        [MockTensor(wh_sigmoid), MockTensor(anchor_grid)], MockOp()
    )
    # Each cell should have 4x the anchor value for its anchor index
    for a in range(na):
        expected_for_anchor = 4.0 * anchor_grid[0, a, 0, 0]
        assert np.allclose(
            result[0, a, :, :, :], expected_for_anchor
        ), f"Anchor {a} should be 4x"
    logger.debug(f"    Maximum is 4x anchor ✓")

    # Property 5: Shape preservation across batch and anchor dimensions
    logger.debug("  Property 5: Shape preservation")
    test_shapes = [
        (1, 1, 8, 8),
        (2, 3, 13, 13),
        (4, 5, 26, 26),
        (8, 3, 10, 20),
    ]
    for bs, na, ny, nx in test_shapes:
        wh_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
        anchor_grid = np.random.rand(1, na, 1, 1, 2).astype(np.float32) * 100

        result = compute_bbox_size_decode(
            [MockTensor(wh_sigmoid), MockTensor(anchor_grid)], MockOp()
        )
        assert result.shape == (
            bs,
            na,
            ny,
            nx,
            2,
        ), f"Shape mismatch for {(bs, na, ny, nx)}"
    logger.debug(f"    All shapes preserved correctly ✓")

    logger.info("\nAll property tests passed!")


if __name__ == "__main__":
    test_bbox_size_decode()
    test_bbox_size_decode_errors()
    test_bbox_size_decode_precision()
    test_bbox_size_decode_properties()
