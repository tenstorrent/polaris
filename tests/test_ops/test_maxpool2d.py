#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ttsim.front.functional.op as funcop
from ttsim.ops.tensor import SimTensor
import numpy as np
from loguru import logger
import common
import pytest
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device
from ttsim.ops.tensor import SimTensor, make_tensor
from ttsim.ops.op import SimOp

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class Maxpool2dTester(common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)

    def __setup__(self, cfgentry: dict):
        cfg: dict = cfgentry["cfg"]
        self.kernel_size: int = cfg["kernel_size"]
        self.channels: int = cfg["channels"]
        self.h: int = cfg["h"]
        self.w: int = cfg["w"]
        self.bs: int = cfg["bs"]
        unsupported_attributes: set = {k for k in cfg} - {
            "kernel_size",
            "channels",
            "bs",
            "h",
            "w",
        }
        assert (
            not unsupported_attributes
        ), f"unsupported attributes {unsupported_attributes} for testMaxpool"
        self.Maxpool1 = funcop.MaxPool2d(self.name + ".Maxpool", self.kernel_size)

    def create_input_tensors(self):
        self.input_tensors: dict[str, SimTensor] = {
            "x": funcop._from_shape(
                "x",
                [self.bs, self.channels, self.h, self.w],
                is_param=False,
                np_dtype=np.float32,
            ),
        }

    def __call__(
        self,
    ):  # Take care of all required arguments through attributes in the same object
        t1 = self.Maxpool1(self.input_tensors["x"])
        return t1


@pytest.mark.unit
@pytest.mark.opunit
def test_Maxpool(tmp_path_factory):
    testname: str = "Maxpooltest"
    configs: dict[str, dict[str, dict]] = {
        f"{testname}01": {
            "cfg": {"kernel_size": 3, "channels": 6, "bs": 4, "h": 24, "w": 24},
            "expected": {"shape": [4, 6, 8, 8]},
        },
    }
    outdir: Path = tmp_path_factory.mktemp("onnx")
    for cfgname, config in configs.items():
        btest: Maxpool2dTester = Maxpool2dTester("Maxpool", config)
        btest.create_input_tensors()
        res = btest()
        assert res.shape == config["expected"]["shape"]
        gr = btest.forward_graph()
        fname = outdir / f"{testname}_{cfgname}.onnx"
        gr.graph2onnx(fname, do_model_check=True)
        # TODO: Run this generated onnx through polaris


# ============================================================================
# Additional tests for numerical validation, precision, and properties
# ============================================================================

from ttsim.ops.desc.data_compute import compute_maxpool2d


def ref_impl_maxpool2d(X, kernel_shape, strides, pads):
    """Reference implementation using NumPy"""
    N, C, H_in, W_in = X.shape
    Kh, Kw = kernel_shape

    # Apply padding with -inf for max pooling
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(
            X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant", constant_values=-np.inf
        )
    else:
        X_padded = X

    # Calculate output size
    H_out = (H_in + pads[0] + pads[2] - Kh) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - Kw) // strides[1] + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # Perform max pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    pool_region = X_padded[
                        n, c, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c, h, w] = np.max(pool_region)

    return Y


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_numerical():
    """Test MaxPool2d with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Numerical Validation:")

    test_cases_numerical = []

    # Test 1: Standard 2x2 pooling
    X1 = np.random.randn(1, 1, 8, 8).astype(np.float32)
    test_cases_numerical.append(("2x2 pool stride 2", X1, [2, 2], [2, 2], [0, 0, 0, 0]))

    # Test 2: 3x3 pooling with stride 3
    X2 = np.random.randn(2, 3, 12, 12).astype(np.float32)
    test_cases_numerical.append(("3x3 pool stride 3", X2, [3, 3], [3, 3], [0, 0, 0, 0]))

    # Test 3: Pooling with padding
    X3 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    test_cases_numerical.append(
        ("2x2 pool with padding", X3, [2, 2], [2, 2], [1, 1, 1, 1])
    )

    # Test 4: Overlapping pooling (stride < kernel)
    X4 = np.random.randn(1, 4, 16, 16).astype(np.float32)
    test_cases_numerical.append(
        ("3x3 pool stride 2 (overlap)", X4, [3, 3], [2, 2], [0, 0, 0, 0])
    )

    # Test 5: Non-square kernel
    X5 = np.random.randn(2, 2, 8, 12).astype(np.float32)
    test_cases_numerical.append(
        ("2x3 non-square kernel", X5, [2, 3], [2, 3], [0, 0, 0, 0])
    )

    # Test 6: Large stride (non-overlapping with gaps)
    X6 = np.random.randn(1, 1, 16, 16).astype(np.float32)
    test_cases_numerical.append(("2x2 pool stride 4", X6, [2, 2], [4, 4], [0, 0, 0, 0]))

    # Test 7: Multiple channels and batch
    X7 = np.random.randn(4, 16, 14, 14).astype(np.float32)
    test_cases_numerical.append(
        ("Multi batch/channel", X7, [2, 2], [2, 2], [0, 0, 0, 0])
    )

    # Test 8: Asymmetric padding
    X8 = np.random.randn(1, 3, 7, 7).astype(np.float32)
    test_cases_numerical.append(
        ("Asymmetric padding", X8, [3, 3], [2, 2], [1, 2, 0, 1])
    )

    passed = 0
    total = len(test_cases_numerical)

    for name, X, kernel_shape, strides, pads in test_cases_numerical:
        # Create mock objects
        iTList = [MockTensor(X)]
        op = MockOp(kernel_shape, strides, pads)

        # Compute using the function under test
        result = compute_maxpool2d(iTList, op)

        # Compute expected result
        expected = ref_impl_maxpool2d(X, kernel_shape, strides, pads)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"

        # Numerical validation
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{name}: Numerical mismatch",
        )

        logger.debug(f"  {name}: PASS [Shape ✓, Numerical ✓]")
        passed += 1

    logger.info(f"\nMaxPool2d Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_errors():
    """Test MaxPool2d edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Edge Cases:")

    # Test 1: Single pixel output
    logger.info("  Test 1: Single pixel output")
    X1 = np.random.randn(1, 2, 2, 2).astype(np.float32)
    iTList1 = [MockTensor(X1)]
    op1 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result1 = compute_maxpool2d(iTList1, op1)
    expected1 = ref_impl_maxpool2d(X1, [2, 2], [2, 2], [0, 0, 0, 0])
    assert result1.shape == (1, 2, 1, 1)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (2x2 → 1x1)")

    # Test 2: All negative values
    logger.info("  Test 2: All negative values")
    X2 = np.random.randn(1, 1, 4, 4).astype(np.float32) - 10.0  # All negative
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)
    expected2 = ref_impl_maxpool2d(X2, [2, 2], [2, 2], [0, 0, 0, 0])
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    # Verify max of negatives is still the largest (least negative)
    assert np.all(result2 < 0)
    logger.debug("    PASS (negative values handled)")

    # Test 3: Large padding
    logger.info("  Test 3: Large padding")
    X3 = np.random.randn(1, 1, 4, 4).astype(np.float32)
    iTList3 = [MockTensor(X3)]
    op3 = MockOp([2, 2], [2, 2], [3, 3, 3, 3])
    result3 = compute_maxpool2d(iTList3, op3)
    expected3 = ref_impl_maxpool2d(X3, [2, 2], [2, 2], [3, 3, 3, 3])
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (large padding)")

    # Test 4: 1x1 kernel (identity-like)
    logger.info("  Test 4: 1x1 kernel")
    X4 = np.random.randn(2, 3, 5, 5).astype(np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp([1, 1], [1, 1], [0, 0, 0, 0])
    result4 = compute_maxpool2d(iTList4, op4)
    expected4 = ref_impl_maxpool2d(X4, [1, 1], [1, 1], [0, 0, 0, 0])
    # 1x1 pooling is essentially identity
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result4, X4, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (1x1 kernel = identity)")

    # Test 5: Mixed positive/negative with zeros
    logger.info("  Test 5: Mixed values with zeros")
    X5 = np.array([[[[1, -2], [0, 3]]]], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result5 = compute_maxpool2d(iTList5, op5)
    expected5 = np.array([[[[3.0]]]], dtype=np.float32)  # max of [1, -2, 0, 3]
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (max([1,-2,0,3]) = 3)")

    # Test 6: Very large stride
    logger.info("  Test 6: Large stride (sparse sampling)")
    X6 = np.random.randn(1, 2, 20, 20).astype(np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp([2, 2], [10, 10], [0, 0, 0, 0])
    result6 = compute_maxpool2d(iTList6, op6)
    expected6 = ref_impl_maxpool2d(X6, [2, 2], [10, 10], [0, 0, 0, 0])
    assert result6.shape == (1, 2, 2, 2)  # Sparse sampling
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (stride 10 on 20x20)")

    # Test 7: Padding creates -inf regions
    logger.info("  Test 7: Padding with high values")
    X7 = np.ones((1, 1, 3, 3), dtype=np.float32) * 100.0
    iTList7 = [MockTensor(X7)]
    op7 = MockOp([3, 3], [3, 3], [1, 1, 1, 1])
    result7 = compute_maxpool2d(iTList7, op7)
    expected7 = ref_impl_maxpool2d(X7, [3, 3], [3, 3], [1, 1, 1, 1])
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    # Verify that padded regions (with -inf) don't dominate
    assert np.all(result7 == 100.0)
    logger.debug("    PASS (padding doesn't dominate max)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_precision():
    """Test MaxPool2d with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Precision (Known Outputs):")

    # Test 1: Simple 2x2 max pooling
    logger.info("  Test 1: 2x2 max pooling")
    X1 = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )
    iTList1 = [MockTensor(X1)]
    op1 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result1 = compute_maxpool2d(iTList1, op1)
    # Max of [[1,2],[5,6]]=6, [[3,4],[7,8]]=8, [[9,10],[13,14]]=14, [[11,12],[15,16]]=16
    expected1 = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug("    4x4 → 2x2: max values ✓")

    # Test 2: All same values
    logger.info("  Test 2: Uniform values")
    X2 = np.full((1, 1, 4, 4), 5.0, dtype=np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)
    expected2 = np.full((1, 1, 2, 2), 5.0, dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug("    All 5.0 → All 5.0 ✓")

    # Test 3: Known max positions
    logger.info("  Test 3: Peak detection")
    X3 = np.array(
        [[[[0, 0, 0, 0], [0, 10, 0, 0], [0, 0, 0, 20], [0, 0, 0, 0]]]], dtype=np.float32
    )
    iTList3 = [MockTensor(X3)]
    op3 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result3 = compute_maxpool2d(iTList3, op3)
    expected3 = np.array([[[[10, 0], [0, 20]]]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug("    Peak values detected ✓")

    # Test 4: Negative values
    logger.info("  Test 4: Negative values")
    X4 = np.array([[[[-10, -5], [-3, -8]]]], dtype=np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result4 = compute_maxpool2d(iTList4, op4)
    expected4 = np.array([[[[-3.0]]]], dtype=np.float32)  # max(-10, -5, -3, -8) = -3
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug("    max([-10,-5,-3,-8]) = -3 ✓")

    # Test 5: 3x3 pooling
    logger.info("  Test 5: 3x3 pooling")
    X5 = np.array([[[[1, 2, 3], [4, 9, 6], [7, 8, 5]]]], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp([3, 3], [3, 3], [0, 0, 0, 0])
    result5 = compute_maxpool2d(iTList5, op5)
    expected5 = np.array([[[[9.0]]]], dtype=np.float32)  # max of all = 9
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug("    3x3 max = 9 ✓")

    # Test 6: Overlapping pooling
    logger.info("  Test 6: Overlapping 3x3 stride 1")
    X6 = np.array([[[[1, 3, 2], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp([2, 2], [1, 1], [0, 0, 0, 0])
    result6 = compute_maxpool2d(iTList6, op6)
    # [[1,3,4,5]→5, [3,2,5,6]→6], [[4,5,7,8]→8, [5,6,8,9]→9]
    expected6 = np.array([[[[5, 6], [8, 9]]]], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug("    Overlapping maxpool ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_properties():
    """Test mathematical properties of MaxPool2d"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Mathematical Properties:")

    # Property 1: Output shape formula
    logger.info("  Property 1: Output shape formula")
    test_configs = [
        ((1, 3, 32, 32), [2, 2], [2, 2], [0, 0, 0, 0], (1, 3, 16, 16)),
        ((2, 16, 28, 28), [3, 3], [2, 2], [0, 0, 0, 0], (2, 16, 13, 13)),
        ((1, 8, 14, 14), [2, 2], [1, 1], [0, 0, 0, 0], (1, 8, 13, 13)),
        ((1, 4, 8, 8), [2, 2], [2, 2], [1, 1, 1, 1], (1, 4, 5, 5)),
    ]

    for X_shape, kernel, strides, pads, expected_shape in test_configs:
        X = np.random.randn(*X_shape).astype(np.float32)
        iTList = [MockTensor(X)]
        op = MockOp(kernel, strides, pads)
        result = compute_maxpool2d(iTList, op)
        assert (
            result.shape == expected_shape
        ), f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    logger.debug("    All output shapes correct ✓")

    # Property 2: Downsampling (output max >= input values)
    logger.info("  Property 2: Max preservation (output contains max values)")
    X2 = np.random.randn(2, 4, 8, 8).astype(np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)

    # Every output value should be >= some region of input
    # Actually, every output value should exist in the input
    for n in range(2):
        for c in range(4):
            assert np.max(result2[n, c]) <= np.max(X2[n, c])
    logger.debug("    Output max ≤ input max ✓")

    # Property 3: Idempotency (maxpool of maxpool doesn't increase values further)
    logger.info("  Property 3: Monotonicity (X ≤ Y → maxpool(X) ≤ maxpool(Y))")
    X3a = np.random.randn(1, 1, 8, 8).astype(np.float32)
    X3b = X3a + 5.0  # X3b > X3a everywhere

    result3a = compute_maxpool2d(
        [MockTensor(X3a)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result3b = compute_maxpool2d(
        [MockTensor(X3b)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    # If X3b >= X3a everywhere, then maxpool(X3b) >= maxpool(X3a)
    assert np.all(result3b >= result3a)
    logger.debug("    Monotonic property holds ✓")

    # Property 4: Scale invariance property
    logger.info("  Property 4: Positive scaling (maxpool(c*X) = c*maxpool(X) for c>0)")
    X4 = np.random.randn(1, 2, 6, 6).astype(np.float32)
    c = 3.0

    result4_scaled = compute_maxpool2d(
        [MockTensor(c * X4)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result4_original = compute_maxpool2d(
        [MockTensor(X4)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    np.testing.assert_allclose(
        result4_scaled, c * result4_original, rtol=1e-6, atol=1e-7
    )
    logger.debug("    maxpool(3*X) = 3*maxpool(X) ✓")

    # Property 5: Translation property
    logger.info("  Property 5: Translation (maxpool(X + c) = maxpool(X) + c)")
    X5 = np.random.randn(1, 2, 8, 8).astype(np.float32)
    c5 = 10.0

    result5_shifted = compute_maxpool2d(
        [MockTensor(X5 + c5)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result5_original = compute_maxpool2d(
        [MockTensor(X5)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    np.testing.assert_allclose(
        result5_shifted, result5_original + c5, rtol=1e-6, atol=1e-7
    )
    logger.debug("    maxpool(X + 10) = maxpool(X) + 10 ✓")

    # Property 6: Channel independence
    logger.info("  Property 6: Channel independence")
    X6 = np.random.randn(1, 4, 8, 8).astype(np.float32)

    result6_all = compute_maxpool2d(
        [MockTensor(X6)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    # Process each channel separately and compare
    for c in range(4):
        X6_single = X6[:, c : c + 1, :, :]
        result6_single = compute_maxpool2d(
            [MockTensor(X6_single)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
        )
        np.testing.assert_allclose(
            result6_all[:, c : c + 1, :, :], result6_single, rtol=1e-6, atol=1e-7
        )
    logger.debug("    Channels processed independently ✓")

    # Property 7: 1x1 pooling is identity
    logger.info("  Property 7: 1x1 pooling = identity")
    X7 = np.random.randn(2, 3, 10, 10).astype(np.float32)
    result7 = compute_maxpool2d([MockTensor(X7)], MockOp([1, 1], [1, 1], [0, 0, 0, 0]))
    np.testing.assert_allclose(result7, X7, rtol=1e-6, atol=1e-7)
    logger.debug("    1x1 maxpool = identity ✓")

    # Property 8: Padding with -inf doesn't affect interior
    logger.info("  Property 8: Padding doesn't affect interior max")
    X8 = np.random.randn(1, 1, 6, 6).astype(np.float32)

    result8_no_pad = compute_maxpool2d(
        [MockTensor(X8)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result8_with_pad = compute_maxpool2d(
        [MockTensor(X8)], MockOp([2, 2], [2, 2], [2, 2, 2, 2])
    )

    # With padding [2,2,2,2], the 6x6 input becomes 10x10 padded → 5x5 output
    # The center 3x3 of padded output should match no-pad output
    # (H_out = (6 + 0 - 2) // 2 + 1 = 3, with pad: (6 + 4 - 2) // 2 + 1 = 5)
    np.testing.assert_allclose(
        result8_with_pad[:, :, 1:4, 1:4], result8_no_pad, rtol=1e-6, atol=1e-7
    )
    logger.debug("    Padding preserves interior ✓")

    logger.info("\nAll property tests passed!")


def calculate_maxpool2d_memory_stats(input_shape, kernel_shape, strides, pads):
    """Calculate memory access and arithmetic operations for maxpool2d operation"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    X = SimTensor({"name": "X", "shape": input_shape, "dtype": "float32"})
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_maxpool2d",
        "optype": "MaxPool",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {"kernel_shape": kernel_shape, "strides": strides, "pads": pads},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_maxpool2d"]
    for x in o_tensors:
        x.op_out = ["test_maxpool2d"]

    op.get_perf_counts(i_tensors, o_tensors)

    perf_stats = op.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    instr_sum = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else 0
    )
    ops = (
        instr_sum if instr_sum > 0 else perf_stats.get("inElems", np.prod(input_shape))
    )
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0

    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "memory_mb": total_memory / (1024 * 1024),
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
    }


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_memory_validation():
    """Test MaxPool2d memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    logger.info("\n" + "=" * 80)
    logger.info("MAXPOOL2D MEMORY VALIDATION")
    logger.info("=" * 80)
    logger.debug(f"Device: {device.devname}")
    logger.debug(f"  Name: {device.name}")
    logger.debug(f"  Frequency: {device.freq_MHz} MHz")
    logger.debug(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("=" * 80)

    # (input_shape [N,C,H,W], kernel_shape, strides, pads)
    test_configs = [
        ([1, 64, 56, 56], [3, 3], [2, 2], [0, 0, 0, 0]),
        ([1, 64, 56, 56], [2, 2], [2, 2], [0, 0, 0, 0]),
        ([1, 128, 28, 28], [3, 3], [2, 2], [0, 0, 0, 0]),
        ([1, 256, 14, 14], [3, 3], [2, 2], [0, 0, 0, 0]),
        ([4, 64, 56, 56], [3, 3], [2, 2], [0, 0, 0, 0]),
        ([4, 128, 28, 28], [3, 3], [1, 1], [1, 1, 1, 1]),
        ([8, 256, 14, 14], [3, 3], [1, 1], [1, 1, 1, 1]),
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for input_shape, kernel_shape, strides, pads in test_configs:
        stats = calculate_maxpool2d_memory_stats(
            input_shape, kernel_shape, strides, pads
        )

        logger.debug(
            f"\nInput: {input_shape}, Kernel: {kernel_shape}, Strides: {strides}"
        )
        logger.debug(f"  Memory: {stats['memory_mb']:.4f} MB")
        logger.debug(f"  Operations: {stats['ops']}")
        logger.debug(
            f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte"
        )
        logger.debug(f"  Bottleneck: {stats['bottleneck']}")

        if stats["bottleneck"] == "memory-bound":
            memory_bound_count += 1
        else:
            compute_bound_count += 1

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {len(test_configs)}")
    logger.info(f"Memory-bound: {memory_bound_count}")
    logger.info(f"Compute-bound: {compute_bound_count}")
    logger.info("=" * 80)


# ============================================================================
# Additional tests for numerical validation, precision, and properties
# ============================================================================

from ttsim.ops.desc.data_compute import compute_maxpool2d


def ref_impl_maxpool2d(X, kernel_shape, strides, pads):
    """Reference implementation using NumPy"""
    N, C, H_in, W_in = X.shape
    Kh, Kw = kernel_shape

    # Apply padding with -inf for max pooling
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(
            X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant", constant_values=-np.inf
        )
    else:
        X_padded = X

    # Calculate output size
    H_out = (H_in + pads[0] + pads[2] - Kh) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - Kw) // strides[1] + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # Perform max pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    pool_region = X_padded[
                        n, c, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c, h, w] = np.max(pool_region)

    return Y


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_numerical():
    """Test MaxPool2d with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Numerical Validation:")

    test_cases_numerical = []

    # Test 1: Standard 2x2 pooling
    X1 = np.random.randn(1, 1, 8, 8).astype(np.float32)
    test_cases_numerical.append(("2x2 pool stride 2", X1, [2, 2], [2, 2], [0, 0, 0, 0]))

    # Test 2: 3x3 pooling with stride 3
    X2 = np.random.randn(2, 3, 12, 12).astype(np.float32)
    test_cases_numerical.append(("3x3 pool stride 3", X2, [3, 3], [3, 3], [0, 0, 0, 0]))

    # Test 3: Pooling with padding
    X3 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    test_cases_numerical.append(
        ("2x2 pool with padding", X3, [2, 2], [2, 2], [1, 1, 1, 1])
    )

    # Test 4: Overlapping pooling (stride < kernel)
    X4 = np.random.randn(1, 4, 16, 16).astype(np.float32)
    test_cases_numerical.append(
        ("3x3 pool stride 2 (overlap)", X4, [3, 3], [2, 2], [0, 0, 0, 0])
    )

    # Test 5: Non-square kernel
    X5 = np.random.randn(2, 2, 8, 12).astype(np.float32)
    test_cases_numerical.append(
        ("2x3 non-square kernel", X5, [2, 3], [2, 3], [0, 0, 0, 0])
    )

    # Test 6: Large stride (non-overlapping with gaps)
    X6 = np.random.randn(1, 1, 16, 16).astype(np.float32)
    test_cases_numerical.append(("2x2 pool stride 4", X6, [2, 2], [4, 4], [0, 0, 0, 0]))

    # Test 7: Multiple channels and batch
    X7 = np.random.randn(4, 16, 14, 14).astype(np.float32)
    test_cases_numerical.append(
        ("Multi batch/channel", X7, [2, 2], [2, 2], [0, 0, 0, 0])
    )

    # Test 8: Asymmetric padding
    X8 = np.random.randn(1, 3, 7, 7).astype(np.float32)
    test_cases_numerical.append(
        ("Asymmetric padding", X8, [3, 3], [2, 2], [1, 2, 0, 1])
    )

    passed = 0
    total = len(test_cases_numerical)

    for name, X, kernel_shape, strides, pads in test_cases_numerical:
        # Create mock objects
        iTList = [MockTensor(X)]
        op = MockOp(kernel_shape, strides, pads)

        # Compute using the function under test
        result = compute_maxpool2d(iTList, op)

        # Compute expected result
        expected = ref_impl_maxpool2d(X, kernel_shape, strides, pads)

        # Shape validation
        assert (
            result.shape == expected.shape
        ), f"{name}: Shape mismatch - got {result.shape}, expected {expected.shape}"

        # Numerical validation
        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{name}: Numerical mismatch",
        )

        logger.debug(f"  {name}: PASS [Shape ✓, Numerical ✓]")
        passed += 1

    logger.info(f"\nMaxPool2d Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_errors():
    """Test MaxPool2d edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Edge Cases:")

    # Test 1: Single pixel output
    logger.info("  Test 1: Single pixel output")
    X1 = np.random.randn(1, 2, 2, 2).astype(np.float32)
    iTList1 = [MockTensor(X1)]
    op1 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result1 = compute_maxpool2d(iTList1, op1)
    expected1 = ref_impl_maxpool2d(X1, [2, 2], [2, 2], [0, 0, 0, 0])
    assert result1.shape == (1, 2, 1, 1)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (2x2 → 1x1)")

    # Test 2: All negative values
    logger.info("  Test 2: All negative values")
    X2 = np.random.randn(1, 1, 4, 4).astype(np.float32) - 10.0  # All negative
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)
    expected2 = ref_impl_maxpool2d(X2, [2, 2], [2, 2], [0, 0, 0, 0])
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    # Verify max of negatives is still the largest (least negative)
    assert np.all(result2 < 0)
    logger.debug("    PASS (negative values handled)")

    # Test 3: Large padding
    logger.info("  Test 3: Large padding")
    X3 = np.random.randn(1, 1, 4, 4).astype(np.float32)
    iTList3 = [MockTensor(X3)]
    op3 = MockOp([2, 2], [2, 2], [3, 3, 3, 3])
    result3 = compute_maxpool2d(iTList3, op3)
    expected3 = ref_impl_maxpool2d(X3, [2, 2], [2, 2], [3, 3, 3, 3])
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (large padding)")

    # Test 4: 1x1 kernel (identity-like)
    logger.info("  Test 4: 1x1 kernel")
    X4 = np.random.randn(2, 3, 5, 5).astype(np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp([1, 1], [1, 1], [0, 0, 0, 0])
    result4 = compute_maxpool2d(iTList4, op4)
    expected4 = ref_impl_maxpool2d(X4, [1, 1], [1, 1], [0, 0, 0, 0])
    # 1x1 pooling is essentially identity
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result4, X4, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (1x1 kernel = identity)")

    # Test 5: Mixed positive/negative with zeros
    logger.info("  Test 5: Mixed values with zeros")
    X5 = np.array([[[[1, -2], [0, 3]]]], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result5 = compute_maxpool2d(iTList5, op5)
    expected5 = np.array([[[[3.0]]]], dtype=np.float32)  # max of [1, -2, 0, 3]
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (max([1,-2,0,3]) = 3)")

    # Test 6: Very large stride
    logger.info("  Test 6: Large stride (sparse sampling)")
    X6 = np.random.randn(1, 2, 20, 20).astype(np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp([2, 2], [10, 10], [0, 0, 0, 0])
    result6 = compute_maxpool2d(iTList6, op6)
    expected6 = ref_impl_maxpool2d(X6, [2, 2], [10, 10], [0, 0, 0, 0])
    assert result6.shape == (1, 2, 2, 2)  # Sparse sampling
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug("    PASS (stride 10 on 20x20)")

    # Test 7: Padding creates -inf regions
    logger.info("  Test 7: Padding with high values")
    X7 = np.ones((1, 1, 3, 3), dtype=np.float32) * 100.0
    iTList7 = [MockTensor(X7)]
    op7 = MockOp([3, 3], [3, 3], [1, 1, 1, 1])
    result7 = compute_maxpool2d(iTList7, op7)
    expected7 = ref_impl_maxpool2d(X7, [3, 3], [3, 3], [1, 1, 1, 1])
    np.testing.assert_allclose(result7, expected7, rtol=1e-6, atol=1e-7)
    # Verify that padded regions (with -inf) don't dominate
    assert np.all(result7 == 100.0)
    logger.debug("    PASS (padding doesn't dominate max)")

    logger.info("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_precision():
    """Test MaxPool2d with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Precision (Known Outputs):")

    # Test 1: Simple 2x2 max pooling
    logger.info("  Test 1: 2x2 max pooling")
    X1 = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        dtype=np.float32,
    )
    iTList1 = [MockTensor(X1)]
    op1 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result1 = compute_maxpool2d(iTList1, op1)
    # Max of [[1,2],[5,6]]=6, [[3,4],[7,8]]=8, [[9,10],[13,14]]=14, [[11,12],[15,16]]=16
    expected1 = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-7)
    logger.debug("    4x4 → 2x2: max values ✓")

    # Test 2: All same values
    logger.info("  Test 2: Uniform values")
    X2 = np.full((1, 1, 4, 4), 5.0, dtype=np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)
    expected2 = np.full((1, 1, 2, 2), 5.0, dtype=np.float32)
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-7)
    logger.debug("    All 5.0 → All 5.0 ✓")

    # Test 3: Known max positions
    logger.info("  Test 3: Peak detection")
    X3 = np.array(
        [[[[0, 0, 0, 0], [0, 10, 0, 0], [0, 0, 0, 20], [0, 0, 0, 0]]]], dtype=np.float32
    )
    iTList3 = [MockTensor(X3)]
    op3 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result3 = compute_maxpool2d(iTList3, op3)
    expected3 = np.array([[[[10, 0], [0, 20]]]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-7)
    logger.debug("    Peak values detected ✓")

    # Test 4: Negative values
    logger.info("  Test 4: Negative values")
    X4 = np.array([[[[-10, -5], [-3, -8]]]], dtype=np.float32)
    iTList4 = [MockTensor(X4)]
    op4 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result4 = compute_maxpool2d(iTList4, op4)
    expected4 = np.array([[[[-3.0]]]], dtype=np.float32)  # max(-10, -5, -3, -8) = -3
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-7)
    logger.debug("    max([-10,-5,-3,-8]) = -3 ✓")

    # Test 5: 3x3 pooling
    logger.info("  Test 5: 3x3 pooling")
    X5 = np.array([[[[1, 2, 3], [4, 9, 6], [7, 8, 5]]]], dtype=np.float32)
    iTList5 = [MockTensor(X5)]
    op5 = MockOp([3, 3], [3, 3], [0, 0, 0, 0])
    result5 = compute_maxpool2d(iTList5, op5)
    expected5 = np.array([[[[9.0]]]], dtype=np.float32)  # max of all = 9
    np.testing.assert_allclose(result5, expected5, rtol=1e-6, atol=1e-7)
    logger.debug("    3x3 max = 9 ✓")

    # Test 6: Overlapping pooling
    logger.info("  Test 6: Overlapping 3x3 stride 1")
    X6 = np.array([[[[1, 3, 2], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)
    iTList6 = [MockTensor(X6)]
    op6 = MockOp([2, 2], [1, 1], [0, 0, 0, 0])
    result6 = compute_maxpool2d(iTList6, op6)
    # [[1,3,4,5]→5, [3,2,5,6]→6], [[4,5,7,8]→8, [5,6,8,9]→9]
    expected6 = np.array([[[[5, 6], [8, 9]]]], dtype=np.float32)
    np.testing.assert_allclose(result6, expected6, rtol=1e-6, atol=1e-7)
    logger.debug("    Overlapping maxpool ✓")

    logger.info("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_maxpool2d_properties():
    """Test mathematical properties of MaxPool2d"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, kernel_shape=[2, 2], strides=None, pads=[0, 0, 0, 0]):
            self.attrs = {
                "kernel_shape": kernel_shape,
                "strides": strides if strides else kernel_shape,
                "pads": pads,
            }

    logger.info("\nTesting MaxPool2d Mathematical Properties:")

    # Property 1: Output shape formula
    logger.info("  Property 1: Output shape formula")
    test_configs = [
        ((1, 3, 32, 32), [2, 2], [2, 2], [0, 0, 0, 0], (1, 3, 16, 16)),
        ((2, 16, 28, 28), [3, 3], [2, 2], [0, 0, 0, 0], (2, 16, 13, 13)),
        ((1, 8, 14, 14), [2, 2], [1, 1], [0, 0, 0, 0], (1, 8, 13, 13)),
        ((1, 4, 8, 8), [2, 2], [2, 2], [1, 1, 1, 1], (1, 4, 5, 5)),
    ]

    for X_shape, kernel, strides, pads, expected_shape in test_configs:
        X = np.random.randn(*X_shape).astype(np.float32)
        iTList = [MockTensor(X)]
        op = MockOp(kernel, strides, pads)
        result = compute_maxpool2d(iTList, op)
        assert (
            result.shape == expected_shape
        ), f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    logger.debug("    All output shapes correct ✓")

    # Property 2: Downsampling (output max >= input values)
    logger.info("  Property 2: Max preservation (output contains max values)")
    X2 = np.random.randn(2, 4, 8, 8).astype(np.float32)
    iTList2 = [MockTensor(X2)]
    op2 = MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    result2 = compute_maxpool2d(iTList2, op2)

    # Every output value should be >= some region of input
    # Actually, every output value should exist in the input
    for n in range(2):
        for c in range(4):
            assert np.max(result2[n, c]) <= np.max(X2[n, c])
    logger.debug("    Output max ≤ input max ✓")

    # Property 3: Idempotency (maxpool of maxpool doesn't increase values further)
    logger.info("  Property 3: Monotonicity (X ≤ Y → maxpool(X) ≤ maxpool(Y))")
    X3a = np.random.randn(1, 1, 8, 8).astype(np.float32)
    X3b = X3a + 5.0  # X3b > X3a everywhere

    result3a = compute_maxpool2d(
        [MockTensor(X3a)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result3b = compute_maxpool2d(
        [MockTensor(X3b)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    # If X3b >= X3a everywhere, then maxpool(X3b) >= maxpool(X3a)
    assert np.all(result3b >= result3a)
    logger.debug("    Monotonic property holds ✓")

    # Property 4: Scale invariance property
    logger.info("  Property 4: Positive scaling (maxpool(c*X) = c*maxpool(X) for c>0)")
    X4 = np.random.randn(1, 2, 6, 6).astype(np.float32)
    c = 3.0

    result4_scaled = compute_maxpool2d(
        [MockTensor(c * X4)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result4_original = compute_maxpool2d(
        [MockTensor(X4)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    np.testing.assert_allclose(
        result4_scaled, c * result4_original, rtol=1e-6, atol=1e-7
    )
    logger.debug("    maxpool(3*X) = 3*maxpool(X) ✓")

    # Property 5: Translation property
    logger.info("  Property 5: Translation (maxpool(X + c) = maxpool(X) + c)")
    X5 = np.random.randn(1, 2, 8, 8).astype(np.float32)
    c5 = 10.0

    result5_shifted = compute_maxpool2d(
        [MockTensor(X5 + c5)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result5_original = compute_maxpool2d(
        [MockTensor(X5)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    np.testing.assert_allclose(
        result5_shifted, result5_original + c5, rtol=1e-6, atol=1e-7
    )
    logger.debug("    maxpool(X + 10) = maxpool(X) + 10 ✓")

    # Property 6: Channel independence
    logger.info("  Property 6: Channel independence")
    X6 = np.random.randn(1, 4, 8, 8).astype(np.float32)

    result6_all = compute_maxpool2d(
        [MockTensor(X6)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )

    # Process each channel separately and compare
    for c in range(4):
        X6_single = X6[:, c : c + 1, :, :]
        result6_single = compute_maxpool2d(
            [MockTensor(X6_single)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
        )
        np.testing.assert_allclose(
            result6_all[:, c : c + 1, :, :], result6_single, rtol=1e-6, atol=1e-7
        )
    logger.debug("    Channels processed independently ✓")

    # Property 7: 1x1 pooling is identity
    logger.info("  Property 7: 1x1 pooling = identity")
    X7 = np.random.randn(2, 3, 10, 10).astype(np.float32)
    result7 = compute_maxpool2d([MockTensor(X7)], MockOp([1, 1], [1, 1], [0, 0, 0, 0]))
    np.testing.assert_allclose(result7, X7, rtol=1e-6, atol=1e-7)
    logger.debug("    1x1 maxpool = identity ✓")

    # Property 8: Padding with -inf doesn't affect interior
    logger.info("  Property 8: Padding doesn't affect interior max")
    X8 = np.random.randn(1, 1, 6, 6).astype(np.float32)

    result8_no_pad = compute_maxpool2d(
        [MockTensor(X8)], MockOp([2, 2], [2, 2], [0, 0, 0, 0])
    )
    result8_with_pad = compute_maxpool2d(
        [MockTensor(X8)], MockOp([2, 2], [2, 2], [2, 2, 2, 2])
    )

    # With padding [2,2,2,2], the 6x6 input becomes 10x10 padded → 5x5 output
    # The center 3x3 of padded output should match no-pad output
    # (H_out = (6 + 0 - 2) // 2 + 1 = 3, with pad: (6 + 4 - 2) // 2 + 1 = 5)
    np.testing.assert_allclose(
        result8_with_pad[:, :, 1:4, 1:4], result8_no_pad, rtol=1e-6, atol=1e-7
    )
    logger.debug("    Padding preserves interior ✓")

    logger.info("\nAll property tests passed!")
