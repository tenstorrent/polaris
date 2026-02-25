#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ttsim.front.functional.op as funcop
import numpy as np
import common
import pytest
from typing import TYPE_CHECKING


class Conv2dTester(common.SimOpTester):
    def __init__(self, name: str, cfgentry: dict):
        super().__init__(name, cfgentry)

    def __setup__(self, cfgentry: dict):
        cfg: dict = cfgentry["cfg"]
        self.in_channels: int = cfg["in_channels"]
        self.out_channels: int = cfg["out_channels"]
        self.kernel_size: int = cfg["kernel_size"]
        self.bs: int = cfg["bs"]
        unsupported_attributes: set = {k for k in cfg} - {
            "in_channels",
            "out_channels",
            "kernel_size",
            "bs",
        }
        assert (
            not unsupported_attributes
        ), f"unsupported attributes {unsupported_attributes} for testbatchnorm"
        self.conv2d1 = funcop.Conv2d(
            self.name + ".conv", self.in_channels, self.out_channels, self.kernel_size
        )

    def create_input_tensors(self):
        self.input_tensors = {
            "x": funcop._from_shape(
                "x",
                [self.bs, self.in_channels, 28, 28],
                is_param=False,
                np_dtype=np.float32,
            ),
        }

    def __call__(
        self,
    ):  # Take care of all required arguments through attributes in the same object
        if TYPE_CHECKING:
            assert self.input_tensors is not None
        t1 = self.conv2d1(self.input_tensors["x"])
        # t2 = self.batchnorm1(t1)
        return t1


@pytest.mark.unit
@pytest.mark.opunit
def test_conv(tmp_path_factory):
    testname: str = "convtest"
    configs: dict[str, dict[str, dict]] = {
        f"{testname}01": {
            "cfg": {"in_channels": 1, "out_channels": 6, "kernel_size": 5, "bs": 4},
            "expected": {"shape": [4, 6, 24, 24]},
        },
    }
    outdir: Path = tmp_path_factory.mktemp("onnx")
    for cfgname, config in configs.items():
        btest: Conv2dTester = Conv2dTester("conv", config)
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

from ttsim.ops.desc.data_compute import compute_conv2d


def ref_impl_conv2d(X, W, B, strides, pads, dilations, group):
    """Reference implementation of 2D convolution using NumPy"""
    N, C_in, H_in, W_in = X.shape
    C_out, C_per_group, Kh, Kw = W.shape

    # Apply padding
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    else:
        X_padded = X

    # Calculate output dimensions
    H_out = (H_in + pads[0] + pads[2] - dilations[0] * (Kh - 1) - 1) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - dilations[1] * (Kw - 1) - 1) // strides[1] + 1

    Y = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    if group == 1:
        # Standard convolution
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * strides[0]
                        w_start = w * strides[1]
                        conv_sum = 0.0
                        for kh in range(Kh):
                            for kw in range(Kw):
                                h_idx = h_start + kh * dilations[0]
                                w_idx = w_start + kw * dilations[1]
                                for c_in in range(C_in):
                                    conv_sum += (
                                        X_padded[n, c_in, h_idx, w_idx]
                                        * W[c_out, c_in, kh, kw]
                                    )
                        Y[n, c_out, h, w] = conv_sum
    else:
        # Grouped convolution
        C_in_per_group = C_in // group
        C_out_per_group = C_out // group

        for g in range(group):
            c_in_start = g * C_in_per_group
            c_out_start = g * C_out_per_group

            for n in range(N):
                for c_out_local in range(C_out_per_group):
                    c_out = c_out_start + c_out_local
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * strides[0]
                            w_start = w * strides[1]
                            conv_sum = 0.0
                            for kh in range(Kh):
                                for kw in range(Kw):
                                    h_idx = h_start + kh * dilations[0]
                                    w_idx = w_start + kw * dilations[1]
                                    for c_in_local in range(C_in_per_group):
                                        c_in = c_in_start + c_in_local
                                        conv_sum += (
                                            X_padded[n, c_in, h_idx, w_idx]
                                            * W[c_out, c_in_local, kh, kw]
                                        )
                            Y[n, c_out, h, w] = conv_sum

    # Add bias if present
    if B is not None:
        Y += B.reshape(1, -1, 1, 1)

    return Y


@pytest.mark.unit
@pytest.mark.opunit
def test_conv2d_numerical():
    """Test Conv2d with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(
            self, strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1
        ):
            self.attrs = {
                "strides": strides,
                "pads": pads,
                "dilations": dilations,
                "group": group,
            }

    print("\nTesting Conv2d Numerical Validation:")

    test_cases_numerical = []

    # Test 1: Simple 3x3 convolution
    X1 = np.random.randn(1, 1, 5, 5).astype(np.float32)
    W1 = np.random.randn(1, 1, 3, 3).astype(np.float32)
    B1 = np.random.randn(1).astype(np.float32)
    test_cases_numerical.append(
        ("3x3 conv no padding", X1, W1, B1, [1, 1], [0, 0, 0, 0], [1, 1], 1)
    )

    # Test 2: Convolution with padding
    X2 = np.random.randn(2, 3, 8, 8).astype(np.float32)
    W2 = np.random.randn(16, 3, 3, 3).astype(np.float32)
    B2 = np.random.randn(16).astype(np.float32)
    test_cases_numerical.append(
        ("3x3 with padding", X2, W2, B2, [1, 1], [1, 1, 1, 1], [1, 1], 1)
    )

    # Test 3: Convolution with stride
    X3 = np.random.randn(1, 3, 32, 32).astype(np.float32)
    W3 = np.random.randn(64, 3, 7, 7).astype(np.float32)
    B3 = np.random.randn(64).astype(np.float32)
    test_cases_numerical.append(
        ("7x7 stride 2", X3, W3, B3, [2, 2], [3, 3, 3, 3], [1, 1], 1)
    )

    # Test 4: Convolution without bias
    X4 = np.random.randn(2, 8, 16, 16).astype(np.float32)
    W4 = np.random.randn(16, 8, 3, 3).astype(np.float32)
    B4 = None
    test_cases_numerical.append(
        ("No bias", X4, W4, B4, [1, 1], [1, 1, 1, 1], [1, 1], 1)
    )

    # Test 5: 1x1 convolution
    X5 = np.random.randn(4, 64, 14, 14).astype(np.float32)
    W5 = np.random.randn(128, 64, 1, 1).astype(np.float32)
    B5 = np.random.randn(128).astype(np.float32)
    test_cases_numerical.append(
        ("1x1 conv", X5, W5, B5, [1, 1], [0, 0, 0, 0], [1, 1], 1)
    )

    # Test 6: Grouped convolution
    X6 = np.random.randn(1, 4, 8, 8).astype(np.float32)
    W6 = np.random.randn(8, 2, 3, 3).astype(np.float32)  # 4 input, 8 output, group=2
    B6 = np.random.randn(8).astype(np.float32)
    test_cases_numerical.append(
        ("Grouped conv", X6, W6, B6, [1, 1], [1, 1, 1, 1], [1, 1], 2)
    )

    # Test 7: Depthwise convolution (group = in_channels = out_channels)
    X7 = np.random.randn(2, 32, 16, 16).astype(np.float32)
    W7 = np.random.randn(32, 1, 3, 3).astype(np.float32)
    B7 = np.random.randn(32).astype(np.float32)
    test_cases_numerical.append(
        ("Depthwise conv", X7, W7, B7, [1, 1], [1, 1, 1, 1], [1, 1], 32)
    )

    # Test 8: Dilation
    X8 = np.random.randn(1, 3, 16, 16).astype(np.float32)
    W8 = np.random.randn(8, 3, 3, 3).astype(np.float32)
    B8 = np.random.randn(8).astype(np.float32)
    test_cases_numerical.append(
        ("Dilated conv", X8, W8, B8, [1, 1], [2, 2, 2, 2], [2, 2], 1)
    )

    passed = 0
    total = len(test_cases_numerical)

    for name, X, W, B, strides, pads, dilations, group in test_cases_numerical:
        # Create mock objects
        if B is not None:
            iTList = [MockTensor(X), MockTensor(W), MockTensor(B)]
        else:
            iTList = [MockTensor(X), MockTensor(W)]
        op = MockOp(strides, pads, dilations, group)

        # Compute using the function under test
        result = compute_conv2d(iTList, op)

        # Compute expected result
        expected = ref_impl_conv2d(X, W, B, strides, pads, dilations, group)

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

    print(f"\nConv2d Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_conv2d_errors():
    """Test Conv2d edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(
            self, strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1
        ):
            self.attrs = {
                "strides": strides,
                "pads": pads,
                "dilations": dilations,
                "group": group,
            }

    print("\nTesting Conv2d Edge Cases:")

    # Test 1: Single pixel input
    print("  Test 1: Single pixel input")
    X1 = np.random.randn(1, 3, 1, 1).astype(np.float32)
    W1 = np.random.randn(8, 3, 1, 1).astype(np.float32)
    B1 = np.random.randn(8).astype(np.float32)
    iTList1 = [MockTensor(X1), MockTensor(W1), MockTensor(B1)]
    op1 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result1 = compute_conv2d(iTList1, op1)
    expected1 = ref_impl_conv2d(X1, W1, B1, [1, 1], [0, 0, 0, 0], [1, 1], 1)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-6)
    print("    PASS (1x1 input handled)")

    # Test 2: Zero padding
    print("  Test 2: Zero padding")
    X2 = np.random.randn(1, 1, 5, 5).astype(np.float32)
    W2 = np.random.randn(1, 1, 3, 3).astype(np.float32)
    B2 = np.zeros(1, dtype=np.float32)
    iTList2 = [MockTensor(X2), MockTensor(W2), MockTensor(B2)]
    op2 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result2 = compute_conv2d(iTList2, op2)
    expected2 = ref_impl_conv2d(X2, W2, B2, [1, 1], [0, 0, 0, 0], [1, 1], 1)
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-5, atol=1e-6)
    print("    PASS (zero padding)")

    # Test 3: Large padding
    print("  Test 3: Large padding (3x3)")
    X3 = np.random.randn(1, 2, 8, 8).astype(np.float32)
    W3 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    B3 = np.random.randn(4).astype(np.float32)
    iTList3 = [MockTensor(X3), MockTensor(W3), MockTensor(B3)]
    op3 = MockOp([1, 1], [3, 3, 3, 3], [1, 1], 1)
    result3 = compute_conv2d(iTList3, op3)
    expected3 = ref_impl_conv2d(X3, W3, B3, [1, 1], [3, 3, 3, 3], [1, 1], 1)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-5, atol=1e-6)
    print("    PASS (large padding)")

    # Test 4: All zeros input
    print("  Test 4: All zeros input")
    X4 = np.zeros((2, 3, 8, 8), dtype=np.float32)
    W4 = np.random.randn(8, 3, 3, 3).astype(np.float32)
    B4 = np.random.randn(8).astype(np.float32)
    iTList4 = [MockTensor(X4), MockTensor(W4), MockTensor(B4)]
    op4 = MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1)
    result4 = compute_conv2d(iTList4, op4)
    expected4 = ref_impl_conv2d(X4, W4, B4, [1, 1], [1, 1, 1, 1], [1, 1], 1)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-6)
    # With zero input, output should equal bias (broadcast)
    for c in range(8):
        np.testing.assert_allclose(result4[0, c, :, :], B4[c], rtol=1e-5, atol=1e-6)
    print("    PASS (zero input → bias)")

    # Test 5: All zeros weights
    print("  Test 5: All zeros weights")
    X5 = np.random.randn(1, 2, 5, 5).astype(np.float32)
    W5 = np.zeros((4, 2, 3, 3), dtype=np.float32)
    B5 = np.random.randn(4).astype(np.float32)
    iTList5 = [MockTensor(X5), MockTensor(W5), MockTensor(B5)]
    op5 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result5 = compute_conv2d(iTList5, op5)
    expected5 = ref_impl_conv2d(X5, W5, B5, [1, 1], [0, 0, 0, 0], [1, 1], 1)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-6)
    # With zero weights, output should equal bias
    for c in range(4):
        np.testing.assert_allclose(result5[0, c, :, :], B5[c], rtol=1e-5, atol=1e-6)
    print("    PASS (zero weights → bias)")

    # Test 6: Large stride that reduces output to 1x1
    print("  Test 6: Large stride → 1x1 output")
    X6 = np.random.randn(1, 3, 16, 16).astype(np.float32)
    W6 = np.random.randn(8, 3, 5, 5).astype(np.float32)
    B6 = np.random.randn(8).astype(np.float32)
    iTList6 = [MockTensor(X6), MockTensor(W6), MockTensor(B6)]
    op6 = MockOp([16, 16], [2, 2, 2, 2], [1, 1], 1)
    result6 = compute_conv2d(iTList6, op6)
    expected6 = ref_impl_conv2d(X6, W6, B6, [16, 16], [2, 2, 2, 2], [1, 1], 1)
    assert result6.shape == expected6.shape
    assert result6.shape[-2:] == (
        1,
        1,
    ), f"Expected 1x1 output, got {result6.shape[-2:]}"
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-6)
    print("    PASS (large stride)")

    # Test 7: Asymmetric padding
    print("  Test 7: Asymmetric padding")
    X7 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    W7 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    B7 = np.random.randn(4).astype(np.float32)
    iTList7 = [MockTensor(X7), MockTensor(W7), MockTensor(B7)]
    op7 = MockOp([1, 1], [1, 2, 3, 4], [1, 1], 1)  # Asymmetric
    result7 = compute_conv2d(iTList7, op7)
    expected7 = ref_impl_conv2d(X7, W7, B7, [1, 1], [1, 2, 3, 4], [1, 1], 1)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-6)
    print("    PASS (asymmetric padding)")

    print("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_conv2d_precision():
    """Test Conv2d with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(
            self, strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1
        ):
            self.attrs = {
                "strides": strides,
                "pads": pads,
                "dilations": dilations,
                "group": group,
            }

    print("\nTesting Conv2d Precision (Known Outputs):")

    # Test 1: Simple 2x2 input, 2x2 kernel
    print("  Test 1: 2x2 input, 2x2 kernel")
    X1 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    W1 = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    B1 = np.array([0.0], dtype=np.float32)
    iTList1 = [MockTensor(X1), MockTensor(W1), MockTensor(B1)]
    op1 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result1 = compute_conv2d(iTList1, op1)
    # 1*1 + 2*0 + 3*0 + 4*1 = 5
    expected1 = np.array([[[[5.0]]]], dtype=np.float32)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-6)
    print(f"    Result: {result1[0, 0, 0, 0]} ✓")

    # Test 2: Identity kernel (center is 1, rest is 0)
    print("  Test 2: 3x3 identity kernel")
    X2 = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]], dtype=np.float32
    )
    W2 = np.array(
        [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]]], dtype=np.float32
    )
    B2 = np.array([0.0], dtype=np.float32)
    iTList2 = [MockTensor(X2), MockTensor(W2), MockTensor(B2)]
    op2 = MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1)
    result2 = compute_conv2d(iTList2, op2)
    # With padding and identity kernel, output should equal input
    np.testing.assert_allclose(result2[0, 0], X2[0, 0], rtol=1e-6, atol=1e-6)
    print(f"    Identity preserved ✓")

    # Test 3: Average pooling with 2x2 kernel (weights = 0.25)
    print("  Test 3: 2x2 averaging kernel")
    X3 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    W3 = np.array([[[[0.25, 0.25], [0.25, 0.25]]]], dtype=np.float32)
    B3 = np.array([0.0], dtype=np.float32)
    iTList3 = [MockTensor(X3), MockTensor(W3), MockTensor(B3)]
    op3 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result3 = compute_conv2d(iTList3, op3)
    # (1 + 2 + 3 + 4) * 0.25 = 2.5
    expected3 = np.array([[[[2.5]]]], dtype=np.float32)
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-6)
    print(f"    Average: {result3[0, 0, 0, 0]} = 2.5 ✓")

    # Test 4: Bias addition
    print("  Test 4: Bias addition")
    X4 = np.ones((1, 1, 3, 3), dtype=np.float32)
    W4 = np.zeros((1, 1, 3, 3), dtype=np.float32)
    B4 = np.array([5.0], dtype=np.float32)
    iTList4 = [MockTensor(X4), MockTensor(W4), MockTensor(B4)]
    op4 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result4 = compute_conv2d(iTList4, op4)
    # With zero weights, output = bias = 5.0
    expected4 = np.full((1, 1, 1, 1), 5.0, dtype=np.float32)
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-6)
    print(f"    Bias only: {result4[0, 0, 0, 0]} = 5.0 ✓")

    # Test 5: Two output channels with different bias
    print("  Test 5: Multi-channel with bias")
    X5 = np.zeros((1, 1, 3, 3), dtype=np.float32)
    W5 = np.zeros((2, 1, 3, 3), dtype=np.float32)
    B5 = np.array([10.0, 20.0], dtype=np.float32)
    iTList5 = [MockTensor(X5), MockTensor(W5), MockTensor(B5)]
    op5 = MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1)
    result5 = compute_conv2d(iTList5, op5)
    expected5_ch0 = np.full((1, 1), 10.0, dtype=np.float32)
    expected5_ch1 = np.full((1, 1), 20.0, dtype=np.float32)
    np.testing.assert_allclose(result5[0, 0], expected5_ch0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result5[0, 1], expected5_ch1, rtol=1e-6, atol=1e-6)
    print(f"    Channels: [10, 20] ✓")

    print("\nAll precision tests passed!")


@pytest.mark.slow
@pytest.mark.unit
@pytest.mark.opunit
def test_conv2d_properties():
    """Test mathematical properties of 2D convolution"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(
            self, strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1
        ):
            self.attrs = {
                "strides": strides,
                "pads": pads,
                "dilations": dilations,
                "group": group,
            }

    print("\nTesting Conv2d Mathematical Properties:")

    # Property 1: Output shape calculation
    print("  Property 1: Output shape formula")
    test_configs = [
        ((1, 3, 32, 32), (16, 3, 3, 3), [1, 1], [0, 0, 0, 0], (1, 16, 30, 30)),
        ((1, 3, 32, 32), (16, 3, 3, 3), [1, 1], [1, 1, 1, 1], (1, 16, 32, 32)),
        ((1, 3, 32, 32), (16, 3, 7, 7), [2, 2], [3, 3, 3, 3], (1, 16, 16, 16)),
        ((2, 64, 56, 56), (128, 64, 1, 1), [1, 1], [0, 0, 0, 0], (2, 128, 56, 56)),
    ]

    for X_shape, W_shape, strides, pads, expected_shape in test_configs:
        X = np.random.randn(*X_shape).astype(np.float32)
        W = np.random.randn(*W_shape).astype(np.float32)
        B = np.random.randn(W_shape[0]).astype(np.float32)

        iTList = [MockTensor(X), MockTensor(W), MockTensor(B)]
        op = MockOp(strides, pads, [1, 1], 1)
        result = compute_conv2d(iTList, op)

        assert (
            result.shape == expected_shape
        ), f"Shape mismatch: got {result.shape}, expected {expected_shape}"
    print(f"    All output shapes correct ✓")

    # Property 2: Linearity in input
    print("  Property 2: Linearity (conv(a*X) = a*conv(X))")
    rng = np.random.default_rng(0)
    X2 = rng.integers(-3, 4, size=(1, 3, 8, 8), dtype=np.int16).astype(np.float32)
    W2 = rng.integers(-2, 3, size=(8, 3, 3, 3), dtype=np.int16).astype(np.float32)
    B2 = np.zeros(8, dtype=np.float32)  # Zero bias for linearity
    scale = 2.0

    result2_scaled = compute_conv2d(
        [MockTensor(scale * X2), MockTensor(W2), MockTensor(B2)],
        MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1),
    )
    result2_original = compute_conv2d(
        [MockTensor(X2), MockTensor(W2), MockTensor(B2)],
        MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1),
    )

    np.testing.assert_array_equal(result2_scaled, scale * result2_original)
    print(f"    conv(2*X) = 2*conv(X) ✓")

    # Property 3: Linearity in weights
    print("  Property 3: Linearity (conv(X, a*W) = a*conv(X, W))")
    X3 = rng.integers(-3, 4, size=(1, 2, 6, 6), dtype=np.int16).astype(np.float32)
    W3 = rng.integers(-2, 3, size=(4, 2, 3, 3), dtype=np.int16).astype(np.float32)
    B3 = np.zeros(4, dtype=np.float32)
    scale = 2.0

    result3_scaled = compute_conv2d(
        [MockTensor(X3), MockTensor(scale * W3), MockTensor(B3)],
        MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1),
    )
    result3_original = compute_conv2d(
        [MockTensor(X3), MockTensor(W3), MockTensor(B3)],
        MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1),
    )

    np.testing.assert_array_equal(result3_scaled, scale * result3_original)
    print(f"    conv(X, 2*W) = 2*conv(X, W) ✓")

    # Property 4: Bias independence from input
    print("  Property 4: Bias added uniformly")
    X4 = np.random.randn(1, 2, 5, 5).astype(np.float32)
    W4 = np.random.randn(3, 2, 3, 3).astype(np.float32)
    B4_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    B4_2 = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    result4_1 = compute_conv2d(
        [MockTensor(X4), MockTensor(W4), MockTensor(B4_1)],
        MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1),
    )
    result4_2 = compute_conv2d(
        [MockTensor(X4), MockTensor(W4), MockTensor(B4_2)],
        MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1),
    )

    # Difference should equal difference in bias (broadcast to full shape)
    diff = result4_2 - result4_1
    expected_diff = (B4_2 - B4_1).reshape(1, -1, 1, 1)
    # Broadcast expected_diff to match the output shape
    expected_diff_full = np.broadcast_to(expected_diff, diff.shape)
    np.testing.assert_allclose(diff, expected_diff_full, rtol=1e-5, atol=1e-6)
    print(f"    Bias difference preserved ✓")

    # Property 5: Grouped convolution equivalence
    print("  Property 5: Grouped conv = separate convs")
    X5 = np.random.randn(1, 4, 8, 8).astype(np.float32)
    W5 = np.random.randn(4, 2, 3, 3).astype(np.float32)  # group=2
    B5 = np.random.randn(4).astype(np.float32)

    # Grouped convolution
    result5_grouped = compute_conv2d(
        [MockTensor(X5), MockTensor(W5), MockTensor(B5)],
        MockOp([1, 1], [1, 1, 1, 1], [1, 1], 2),
    )

    # Manual split and merge
    X5_g1 = X5[:, :2, :, :]
    X5_g2 = X5[:, 2:, :, :]
    W5_g1 = W5[:2, :, :, :]
    W5_g2 = W5[2:, :, :, :]
    B5_g1 = B5[:2]
    B5_g2 = B5[2:]

    result5_g1 = compute_conv2d(
        [MockTensor(X5_g1), MockTensor(W5_g1), MockTensor(B5_g1)],
        MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1),
    )
    result5_g2 = compute_conv2d(
        [MockTensor(X5_g2), MockTensor(W5_g2), MockTensor(B5_g2)],
        MockOp([1, 1], [1, 1, 1, 1], [1, 1], 1),
    )

    result5_manual = np.concatenate([result5_g1, result5_g2], axis=1)
    np.testing.assert_allclose(result5_grouped, result5_manual, rtol=1e-5, atol=1e-6)
    print(f"    Grouped conv = separate convs ✓")

    # Property 6: Padding only affects boundary
    print("  Property 6: Padding zero-fills boundaries")
    X6 = np.random.randn(1, 2, 10, 10).astype(np.float32)
    W6 = np.random.randn(4, 2, 3, 3).astype(np.float32)
    B6 = np.random.randn(4).astype(np.float32)

    # With padding
    result6_padded = compute_conv2d(
        [MockTensor(X6), MockTensor(W6), MockTensor(B6)],
        MockOp([1, 1], [2, 2, 2, 2], [1, 1], 1),
    )

    # Without padding (will produce smaller output)
    result6_no_pad = compute_conv2d(
        [MockTensor(X6), MockTensor(W6), MockTensor(B6)],
        MockOp([1, 1], [0, 0, 0, 0], [1, 1], 1),
    )

    # Center region should match
    H_no_pad, W_no_pad = result6_no_pad.shape[2:]
    center_padded = result6_padded[:, :, 2 : 2 + H_no_pad, 2 : 2 + W_no_pad]
    np.testing.assert_allclose(center_padded, result6_no_pad, rtol=1e-5, atol=1e-6)
    print(f"    Padding preserves center region ✓")

    print("\nAll property tests passed!")
