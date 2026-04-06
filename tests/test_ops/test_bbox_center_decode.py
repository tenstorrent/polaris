#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test bbox_center_decode operation for YOLO-style object detection"""

import numpy as np
import pytest
import warnings
from loguru import logger

from ttsim.ops.desc.data_compute import compute_bbox_center_decode


def ref_impl_bbox_center_decode(xy_sigmoid, grid, stride):
    """
    Reference implementation of bbox center decoding.
    Formula: (sigmoid(xy) * 2.0 - 0.5 + grid) * stride

    This operation is used in YOLO-style object detection to convert
    predicted offsets (already sigmoid-activated) into actual image coordinates.

    Args:
        xy_sigmoid: [bs, na, ny, nx, 2] - sigmoid activated xy predictions (0-1 range)
        grid: [1, 1, ny, nx, 2] or [bs, na, ny, nx, 2] - coordinate grid
        stride: scalar - detection layer stride (downsampling factor)

    Returns:
        xy_decoded: [bs, na, ny, nx, 2] - decoded xy in image coordinates
    """
    return (xy_sigmoid * 2.0 - 0.5 + grid) * stride


def generate_test_data():
    """Generate comprehensive test cases for bbox_center_decode"""
    test_cases = []

    # Basic small detection grid (typical YOLO configurations)
    # Test 1: Single scale detection (13x13 grid, stride 32, 3 anchors, batch 1)
    bs, na, ny, nx = 1, 3, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("yolo_13x13_stride32", xy_sigmoid, grid, stride))

    # Test 2: Medium scale (26x26 grid, stride 16)
    bs, na, ny, nx = 1, 3, 26, 26
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("yolo_26x26_stride16", xy_sigmoid, grid, stride))

    # Test 3: Fine scale (52x52 grid, stride 8)
    bs, na, ny, nx = 1, 3, 52, 52
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(8.0, dtype=np.float32)
    test_cases.append(("yolo_52x52_stride8", xy_sigmoid, grid, stride))

    # Test 4: Small grid (4x4)
    bs, na, ny, nx = 1, 3, 4, 4
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("small_4x4", xy_sigmoid, grid, stride))

    # Test 5: Batch processing (batch size 4)
    bs, na, ny, nx = 4, 3, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("batch_4", xy_sigmoid, grid, stride))

    # Test 6: Larger batch (batch size 8)
    bs, na, ny, nx = 8, 3, 26, 26
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("batch_8", xy_sigmoid, grid, stride))

    # Test 7: Different number of anchors (1 anchor)
    bs, na, ny, nx = 1, 1, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("single_anchor", xy_sigmoid, grid, stride))

    # Test 8: Multiple anchors (5 anchors)
    bs, na, ny, nx = 1, 5, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("five_anchors", xy_sigmoid, grid, stride))

    # Test 9: Sigmoid values at cell center (0.5, 0.5)
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.full((bs, na, ny, nx, 2), 0.5, dtype=np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("center_sigmoid", xy_sigmoid, grid, stride))

    # Test 10: Sigmoid at 0 (left/top edge of cell)
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("sigmoid_zero", xy_sigmoid, grid, stride))

    # Test 11: Sigmoid at 1 (right/bottom beyond cell)
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.ones((bs, na, ny, nx, 2), dtype=np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("sigmoid_one", xy_sigmoid, grid, stride))

    # Test 12: Grid broadcasting test (broadcasted grid)
    bs, na, ny, nx = 2, 3, 10, 10
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("broadcast_grid", xy_sigmoid, grid, stride))

    # Test 13: Full grid shape (no broadcasting)
    bs, na, ny, nx = 2, 3, 10, 10
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("full_grid", xy_sigmoid, grid, stride))

    # Test 14: Different strides
    bs, na, ny, nx = 1, 3, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(64.0, dtype=np.float32)
    test_cases.append(("stride_64", xy_sigmoid, grid, stride))

    # Test 15: Stride 4 (very fine scale)
    bs, na, ny, nx = 1, 3, 80, 80
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(4.0, dtype=np.float32)
    test_cases.append(("stride_4_80x80", xy_sigmoid, grid, stride))

    # Test 16: Single cell grid (1x1)
    bs, na, ny, nx = 1, 3, 1, 1
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    stride = np.array(32.0, dtype=np.float32)
    test_cases.append(("single_cell", xy_sigmoid, grid, stride))

    # Test 17: Rectangular grid (non-square)
    bs, na, ny, nx = 1, 3, 10, 20
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("rectangular_10x20", xy_sigmoid, grid, stride))

    # Test 18: Another rectangular (20x10)
    bs, na, ny, nx = 1, 3, 20, 10
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("rectangular_20x10", xy_sigmoid, grid, stride))

    # Test 19: Mixed sigmoid values
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    xy_sigmoid[0, 0, 0, 0, :] = [0.0, 0.0]  # corner
    xy_sigmoid[0, 0, 0, -1, :] = [1.0, 0.0]  # corner
    xy_sigmoid[0, 0, -1, 0, :] = [0.0, 1.0]  # corner
    xy_sigmoid[0, 0, -1, -1, :] = [1.0, 1.0]  # corner
    xy_sigmoid[0, 0, 4, 4, :] = [0.5, 0.5]  # center
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(16.0, dtype=np.float32)
    test_cases.append(("mixed_sigmoids", xy_sigmoid, grid, stride))

    # Test 20: Stride 1 (no downsampling)
    bs, na, ny, nx = 1, 3, 16, 16
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(1.0, dtype=np.float32)
    test_cases.append(("stride_1", xy_sigmoid, grid, stride))

    return test_cases


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_center_decode():
    """Test bbox_center_decode with shape and numerical validation"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    test_cases = generate_test_data()
    passed = 0
    total = len(test_cases)

    for name, xy_sigmoid, grid, stride in test_cases:
        # Create mock objects
        iTList = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
        op = MockOp()

        # Compute using the function under test
        result = compute_bbox_center_decode(iTList, op)

        # Compute expected result
        expected = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"
        assert (
            result.shape == xy_sigmoid.shape
        ), f"{name}: Output shape should match xy_sigmoid shape"

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

    logger.info(f"\nBbox Center Decode Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_center_decode_errors():
    """Test bbox_center_decode edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Center Decode Edge Cases:")

    # Suppress warnings for zero/negative stride tests
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Test 1: Zero stride (should produce zeros)
    logger.info("  Test 1: Zero stride")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(0.0, dtype=np.float32)
    iTList1 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op1 = MockOp()
    result1 = compute_bbox_center_decode(iTList1, op1)
    expected1 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-7)
    assert np.all(result1 == 0.0), "Zero stride should produce all zeros"
    logger.debug("    PASS (zero stride produces zeros)")

    # Test 2: Negative stride
    logger.info("  Test 2: Negative stride")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(-16.0, dtype=np.float32)
    iTList2 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op2 = MockOp()
    result2 = compute_bbox_center_decode(iTList2, op2)
    expected2 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-7)
    # Negative stride flips the sign of the output based on intermediate values
    logger.debug("    PASS (negative stride handled correctly)")

    # Test 3: Very large stride
    logger.info("  Test 3: Very large stride")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(1000.0, dtype=np.float32)
    iTList3 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op3 = MockOp()
    result3 = compute_bbox_center_decode(iTList3, op3)
    expected3 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (very large stride handled correctly)")

    # Test 4: Zero grid (all grid coordinates at origin)
    logger.info("  Test 4: Zero grid")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    stride = np.array(16.0, dtype=np.float32)
    iTList4 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op4 = MockOp()
    result4 = compute_bbox_center_decode(iTList4, op4)
    expected4 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (zero grid handled correctly)")

    # Test 5: All zeros (sigmoid=0, grid=0, stride=0)
    logger.info("  Test 5: All zeros")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.zeros((bs, na, ny, nx, 2), dtype=np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    stride = np.array(0.0, dtype=np.float32)
    iTList5 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op5 = MockOp()
    result5 = compute_bbox_center_decode(iTList5, op5)
    expected5 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-7)
    assert np.all(result5 == 0.0), "All zeros should produce zero output"
    logger.debug("    PASS (all zeros produces zero output)")

    # Test 6: Single cell with extreme values
    logger.info("  Test 6: Single cell with extreme values")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[1.0, 1.0]]]]], dtype=np.float32)
    grid = np.array([[[[[100.0, 100.0]]]]], dtype=np.float32)
    stride = np.array(32.0, dtype=np.float32)
    iTList6 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op6 = MockOp()
    result6 = compute_bbox_center_decode(iTList6, op6)
    expected6 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result6.shape == expected6.shape
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (single cell with extreme values)")

    # Test 7: Very small stride (fractional)
    logger.info("  Test 7: Very small stride (fractional)")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(0.1, dtype=np.float32)
    iTList7 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op7 = MockOp()
    result7 = compute_bbox_center_decode(iTList7, op7)
    expected7 = ref_impl_bbox_center_decode(xy_sigmoid, grid, stride)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-7)
    logger.debug("    PASS (small fractional stride)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_center_decode_precision():
    """Test bbox_center_decode with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Center Decode Precision (Known Outputs):")

    # Test 1: Center of first cell (sigmoid=0.5, grid=[0,0], stride=16)
    # Formula: (0.5 * 2.0 - 0.5 + 0) * 16 = (1.0 - 0.5) * 16 = 0.5 * 16 = 8.0
    logger.info("  Test 1: Center of first cell")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[0.5, 0.5]]]]], dtype=np.float32)
    grid = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    stride = np.array(16.0, dtype=np.float32)
    iTList1 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op1 = MockOp()
    result1 = compute_bbox_center_decode(iTList1, op1)
    expected1 = np.array([[[[[8.0, 8.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result1[0,0,0,0]} (expected [8.0, 8.0]) ✓"
    )

    # Test 2: Left edge of cell (sigmoid=0, grid=[0,0], stride=16)
    # Formula: (0.0 * 2.0 - 0.5 + 0) * 16 = -0.5 * 16 = -8.0
    logger.info("  Test 2: Left edge of cell")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    grid = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    stride = np.array(16.0, dtype=np.float32)
    iTList2 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op2 = MockOp()
    result2 = compute_bbox_center_decode(iTList2, op2)
    expected2 = np.array([[[[[-8.0, -8.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result2[0,0,0,0]} (expected [-8.0, -8.0]) ✓"
    )

    # Test 3: Right edge beyond cell (sigmoid=1, grid=[0,0], stride=16)
    # Formula: (1.0 * 2.0 - 0.5 + 0) * 16 = 1.5 * 16 = 24.0
    logger.info("  Test 3: Right edge beyond cell")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[1.0, 1.0]]]]], dtype=np.float32)
    grid = np.array([[[[[0.0, 0.0]]]]], dtype=np.float32)
    stride = np.array(16.0, dtype=np.float32)
    iTList3 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op3 = MockOp()
    result3 = compute_bbox_center_decode(iTList3, op3)
    expected3 = np.array([[[[[24.0, 24.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result3[0,0,0,0]} (expected [24.0, 24.0]) ✓"
    )

    # Test 4: Second cell center (sigmoid=0.5, grid=[1,1], stride=16)
    # Formula: (0.5 * 2.0 - 0.5 + 1) * 16 = 1.5 * 16 = 24.0
    logger.info("  Test 4: Center of second cell")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[0.5, 0.5]]]]], dtype=np.float32)
    grid = np.array([[[[[1.0, 1.0]]]]], dtype=np.float32)
    stride = np.array(16.0, dtype=np.float32)
    iTList4 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op4 = MockOp()
    result4 = compute_bbox_center_decode(iTList4, op4)
    expected4 = np.array([[[[[24.0, 24.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result4[0,0,0,0]} (expected [24.0, 24.0]) ✓"
    )

    # Test 5: Stride 32 at cell (5, 5) with sigmoid 0.5
    # Formula: (0.5 * 2.0 - 0.5 + 5) * 32 = 5.5 * 32 = 176.0
    logger.info("  Test 5: Cell (5,5) with stride 32")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[0.5, 0.5]]]]], dtype=np.float32)
    grid = np.array([[[[[5.0, 5.0]]]]], dtype=np.float32)
    stride = np.array(32.0, dtype=np.float32)
    iTList5 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op5 = MockOp()
    result5 = compute_bbox_center_decode(iTList5, op5)
    expected5 = np.array([[[[[176.0, 176.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result5[0,0,0,0]} (expected [176.0, 176.0]) ✓"
    )

    # Test 6: Different x and y values
    # x: (0.25 * 2.0 - 0.5 + 2) * 8 = 2.0 * 8 = 16.0
    # y: (0.75 * 2.0 - 0.5 + 3) * 8 = 4.0 * 8 = 32.0
    logger.info("  Test 6: Different x and y coordinates")
    bs, na, ny, nx = 1, 1, 1, 1
    xy_sigmoid = np.array([[[[[0.25, 0.75]]]]], dtype=np.float32)
    grid = np.array([[[[[2.0, 3.0]]]]], dtype=np.float32)
    stride = np.array(8.0, dtype=np.float32)
    iTList6 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op6 = MockOp()
    result6 = compute_bbox_center_decode(iTList6, op6)
    expected6 = np.array([[[[[16.0, 32.0]]]]], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug(
        f"    Result: {result6[0,0,0,0]} (expected [16.0, 32.0]) ✓"
    )

    # Test 7: Multiple cells 2x2 grid
    logger.info("  Test 7: 2x2 grid with known values")
    bs, na, ny, nx = 1, 1, 2, 2
    xy_sigmoid = np.array(
        [[[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]]], dtype=np.float32
    )
    grid = np.array(
        [[[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]]], dtype=np.float32
    )
    stride = np.array(16.0, dtype=np.float32)
    iTList7 = [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)]
    op7 = MockOp()
    result7 = compute_bbox_center_decode(iTList7, op7)
    expected7 = np.array(
        [[[[[8.0, 8.0], [24.0, 8.0]], [[8.0, 24.0], [24.0, 24.0]]]]], dtype=np.float32
    )
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    logger.debug("    Grid centers at [8,8], [24,8], [8,24], [24,24] ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_bbox_center_decode_properties():
    """Test mathematical properties of bbox_center_decode"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self):
            self.attrs = {}

    logger.info("\nTesting Bbox Center Decode Properties:")

    # Property 1: Output range relative to stride
    # For sigmoid in [0,1], output should be in range [(grid-0.5)*stride, (grid+1.5)*stride]
    logger.info("  Property 1: Output range based on stride")
    bs, na, ny, nx = 1, 3, 13, 13
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.zeros((1, 1, ny, nx, 2), dtype=np.float32)
    for h in range(ny):
        for w in range(nx):
            grid[0, 0, h, w, 0] = w
            grid[0, 0, h, w, 1] = h
    stride = np.array(32.0, dtype=np.float32)
    result = compute_bbox_center_decode(
        [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)], MockOp()
    )

    # Check each cell's output is in expected range
    for h in range(ny):
        for w in range(nx):
            min_expected_x = (w - 0.5) * 32.0
            max_expected_x = (w + 1.5) * 32.0
            min_expected_y = (h - 0.5) * 32.0
            max_expected_y = (h + 1.5) * 32.0

            cell_x = result[:, :, h, w, 0]
            cell_y = result[:, :, h, w, 1]

            assert np.all(cell_x >= min_expected_x - 1e-5) and np.all(
                cell_x <= max_expected_x + 1e-5
            ), f"Cell ({h},{w}) x out of range"
            assert np.all(cell_y >= min_expected_y - 1e-5) and np.all(
                cell_y <= max_expected_y + 1e-5
            ), f"Cell ({h},{w}) y out of range"
    logger.debug("    All cells within expected range ✓")

    # Property 2: Linearity with stride
    # Doubling stride should double outputs
    logger.info("  Property 2: Linear scaling with stride")
    bs, na, ny, nx = 1, 3, 10, 10
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny

    stride1 = np.array(16.0, dtype=np.float32)
    result1 = compute_bbox_center_decode(
        [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride1)], MockOp()
    )

    stride2 = np.array(32.0, dtype=np.float32)
    result2 = compute_bbox_center_decode(
        [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride2)], MockOp()
    )

    # result2 should be result1 * 2
    np.testing.assert_allclose(result2, result1 * 2.0, rtol=1e-5, atol=1e-7)
    logger.debug("    Doubling stride doubles output ✓")

    # Property 3: Grid offset translation
    # Adding constant to grid should add (constant * stride) to output
    logger.info("  Property 3: Grid offset translation")
    bs, na, ny, nx = 1, 3, 8, 8
    xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
    grid1 = np.random.rand(1, 1, ny, nx, 2).astype(np.float32) * ny
    stride = np.array(16.0, dtype=np.float32)

    result1 = compute_bbox_center_decode(
        [MockTensor(xy_sigmoid), MockTensor(grid1), MockTensor(stride)], MockOp()
    )

    offset = np.array([5.0, 3.0], dtype=np.float32)
    grid2 = grid1 + offset.reshape(1, 1, 1, 1, 2)
    result2 = compute_bbox_center_decode(
        [MockTensor(xy_sigmoid), MockTensor(grid2), MockTensor(stride)], MockOp()
    )

    # result2 should be result1 + offset * stride
    expected_diff = offset * stride
    np.testing.assert_allclose(
        result2, result1 + expected_diff.reshape(1, 1, 1, 1, 2), rtol=1e-5, atol=1e-7
    )
    logger.debug("    Grid offset correctly translates output ✓")

    # Property 4: Shape preservation across batch and anchor dimensions
    logger.info("  Property 4: Shape preservation")
    test_shapes = [
        (1, 1, 8, 8),
        (2, 3, 13, 13),
        (4, 5, 26, 26),
        (8, 3, 10, 20),
    ]
    for bs, na, ny, nx in test_shapes:
        xy_sigmoid = np.random.rand(bs, na, ny, nx, 2).astype(np.float32)
        grid = np.random.rand(1, 1, ny, nx, 2).astype(np.float32)
        stride = np.array(16.0, dtype=np.float32)

        result = compute_bbox_center_decode(
            [MockTensor(xy_sigmoid), MockTensor(grid), MockTensor(stride)], MockOp()
        )
        assert result.shape == (
            bs,
            na,
            ny,
            nx,
            2,
        ), f"Shape mismatch for {(bs, na, ny, nx)}"
    logger.debug("    All shapes preserved correctly ✓")

    logger.info("\nAll property tests passed!")


if __name__ == "__main__":
    test_bbox_center_decode()
    test_bbox_center_decode_errors()
    test_bbox_center_decode_precision()
    test_bbox_center_decode_properties()
