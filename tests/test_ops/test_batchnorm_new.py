#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def ref_impl_onnx(XShape, SShape, BShape, MShape, VShape, output_mean_var, **kwargs):
    """shape inference for batchnorm"""

    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, XShape),
        helper.make_tensor_value_info("S", TensorProto.FLOAT, SShape),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, BShape),
        helper.make_tensor_value_info("M", TensorProto.FLOAT, MShape),
        helper.make_tensor_value_info("V", TensorProto.FLOAT, VShape),
    ]

    # Define output tensors, Shapes to be inferred
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)]
    if output_mean_var == True:
        outputs.append(helper.make_tensor_value_info("Mean", TensorProto.FLOAT, None))
        outputs.append(helper.make_tensor_value_info("Var", TensorProto.FLOAT, None))

    # Create BatchNormalization node
    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "S", "B", "M", "V"],
        outputs=["Y"] + (["Mean", "Var"] if output_mean_var else []),
        **kwargs,
    )

    # Create graph and model
    graph = helper.make_graph([bn_node], "bn_graph", inputs, outputs)
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)

    output_shapes = {
        output.name: [dim.dim_value for dim in output.type.tensor_type.shape.dim]
        for output in inferred_model.graph.output
    }
    return output_shapes


# Test cases
test_name = "test_batchnorm_new"
test_cases = [
    {
        "x": [2, 3, 4, 5],
        "s": [3],
        "b": [3],
        "m": [3],
        "v": [3],
        "y": [2, 3, 4, 5],
        "name": "test_batchnorm_example",
    },
    {
        "x": [2, 3, 4, 5],
        "s": [3],
        "b": [3],
        "m": [3],
        "v": [3],
        "eps": 1e-2,
        "y": [2, 3, 4, 5],
        "name": "test_batchnorm_epsilon",
    },
    {
        "x": [2, 3, 4, 5],
        "s": [3],
        "b": [3],
        "m": [3],
        "v": [3],
        "t": True,
        "y": [2, 3, 4, 5],
        "output_mean_var": True,
        "name": "test_batchnorm_example_training_mode",
    },
    {
        "x": [2, 3, 4, 5],
        "s": [3],
        "b": [3],
        "m": [3],
        "v": [3],
        "t": True,
        "y": [2, 3, 4, 5],
        "output_mean_var": True,
        "name": "test_batchnorm_epsilon_training_mode",
    },
]


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_new():
    msgw = max([len(x) for x in test_cases])  # type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        test_name = trec["name"]  # type: ignore
        op_name = f"{test_name}_{tno}"

        XShape = trec["x"]  # type: ignore
        SShape = trec["s"]  # type: ignore
        BShape = trec["b"]  # type: ignore
        MShape = trec["m"]  # type: ignore
        VShape = trec["v"]  # type: ignore

        i_tensors = [
            F._from_shape(f"X", XShape),
            F._from_shape(f"S", SShape),
            F._from_shape(f"B", BShape),
            F._from_shape(f"M", MShape),
            F._from_shape(f"V", VShape),
        ]
        o_tensors = [make_tensor("Y")]

        output_mean_var = False
        if "output_mean_var" in trec:  # type: ignore
            output_mean_var = trec["output_mean_var"]  # type: ignore
            if output_mean_var == True:
                o_tensors.append(make_tensor("Mean"))
                o_tensors.append(make_tensor("Var"))

        attrs = {}
        if "eps" in trec:
            attrs["epsilon"] = trec["eps"]  # type: ignore
        if "t" in trec:
            attrs["training_mode"] = trec["t"]  # type: ignore
        op_info = {
            "name": op_name,
            "optype": "BatchNormalization",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": attrs,
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]
        op_obj.get_perf_counts(i_tensors, o_tensors)

        ref_shapes = ref_impl_onnx(
            XShape, SShape, BShape, MShape, VShape, output_mean_var, **attrs
        )
        inf_y_shape = o_tensors[0].shape
        assert inf_y_shape == ref_shapes["Y"], f"{inf_y_shape} != {ref_shapes['Y']}"
        if output_mean_var:
            inf_m_shape = o_tensors[1].shape
            assert (
                inf_m_shape == ref_shapes["Mean"]
            ), f"{inf_m_shape} != {ref_shapes['Mean']}"

            inf_v_shape = o_tensors[2].shape
            assert (
                inf_v_shape == ref_shapes["Var"]
            ), f"{inf_v_shape} != {ref_shapes['Var']}"

        print(f"TEST[{tno:3d}] {test_name:{msgw}s} PASS")


# ============================================================================
# Additional tests for numerical validation, precision, and properties
# ============================================================================

from ttsim.ops.desc.data_compute import compute_batchnorm


def ref_impl_batchnorm(X, scale, bias, mean, var, epsilon):
    """Reference implementation of batch normalization"""
    X_normalized = (X - mean.reshape(1, -1, 1, 1)) / np.sqrt(
        var.reshape(1, -1, 1, 1) + epsilon
    )
    return scale.reshape(1, -1, 1, 1) * X_normalized + bias.reshape(1, -1, 1, 1)


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_numerical():
    """Test batchnorm with numerical validation (actual data computation)"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, epsilon=1e-5):
            self.attrs = {"epsilon": epsilon}

    print("\nTesting BatchNorm Numerical Validation:")

    test_cases_numerical = []

    # Test 1: Simple 4D case
    X1 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    scale1 = np.ones(3, dtype=np.float32)
    bias1 = np.zeros(3, dtype=np.float32)
    mean1 = np.random.randn(3).astype(np.float32)
    var1 = np.random.rand(3).astype(np.float32) + 0.1
    test_cases_numerical.append(("4D standard", X1, scale1, bias1, mean1, var1, 1e-5))

    # Test 2: Different epsilon
    X2 = np.random.randn(1, 2, 8, 8).astype(np.float32)
    scale2 = np.ones(2, dtype=np.float32)
    bias2 = np.zeros(2, dtype=np.float32)
    mean2 = np.random.randn(2).astype(np.float32)
    var2 = np.random.rand(2).astype(np.float32) + 0.1
    test_cases_numerical.append(
        ("Custom epsilon", X2, scale2, bias2, mean2, var2, 1e-2)
    )

    # Test 3: Non-unit scale and bias
    X3 = np.random.randn(2, 4, 5, 5).astype(np.float32)
    scale3 = np.random.randn(4).astype(np.float32)
    bias3 = np.random.randn(4).astype(np.float32)
    mean3 = np.random.randn(4).astype(np.float32)
    var3 = np.random.rand(4).astype(np.float32) + 0.1
    test_cases_numerical.append(
        ("With scale/bias", X3, scale3, bias3, mean3, var3, 1e-5)
    )

    # Test 4: Large batch
    X4 = np.random.randn(8, 3, 32, 32).astype(np.float32)
    scale4 = np.ones(3, dtype=np.float32)
    bias4 = np.zeros(3, dtype=np.float32)
    mean4 = np.random.randn(3).astype(np.float32)
    var4 = np.random.rand(3).astype(np.float32) + 0.1
    test_cases_numerical.append(("Large batch", X4, scale4, bias4, mean4, var4, 1e-5))

    # Test 5: Single channel
    X5 = np.random.randn(2, 1, 10, 10).astype(np.float32)
    scale5 = np.array([2.0], dtype=np.float32)
    bias5 = np.array([1.0], dtype=np.float32)
    mean5 = np.array([0.0], dtype=np.float32)
    var5 = np.array([1.0], dtype=np.float32)
    test_cases_numerical.append(
        ("Single channel", X5, scale5, bias5, mean5, var5, 1e-5)
    )

    # Test 6: Many channels
    X6 = np.random.randn(1, 64, 7, 7).astype(np.float32)
    scale6 = np.ones(64, dtype=np.float32)
    bias6 = np.zeros(64, dtype=np.float32)
    mean6 = np.random.randn(64).astype(np.float32)
    var6 = np.random.rand(64).astype(np.float32) + 0.1
    test_cases_numerical.append(("Many channels", X6, scale6, bias6, mean6, var6, 1e-5))

    # Test 7: Small spatial dimensions
    X7 = np.random.randn(4, 8, 1, 1).astype(np.float32)
    scale7 = np.random.randn(8).astype(np.float32)
    bias7 = np.random.randn(8).astype(np.float32)
    mean7 = np.random.randn(8).astype(np.float32)
    var7 = np.random.rand(8).astype(np.float32) + 0.1
    test_cases_numerical.append(("1x1 spatial", X7, scale7, bias7, mean7, var7, 1e-5))

    # Test 8: Large spatial dimensions
    X8 = np.random.randn(1, 3, 224, 224).astype(np.float32)
    scale8 = np.ones(3, dtype=np.float32)
    bias8 = np.zeros(3, dtype=np.float32)
    mean8 = np.random.randn(3).astype(np.float32)
    var8 = np.random.rand(3).astype(np.float32) + 0.1
    test_cases_numerical.append(("Large 224x224", X8, scale8, bias8, mean8, var8, 1e-5))

    passed = 0
    total = len(test_cases_numerical)

    for name, X, scale, bias, mean, var, epsilon in test_cases_numerical:
        # Create mock objects
        iTList = [
            MockTensor(X),
            MockTensor(scale),
            MockTensor(bias),
            MockTensor(mean),
            MockTensor(var),
        ]
        op = MockOp(epsilon)

        # Compute using the function under test
        result = compute_batchnorm(iTList, op)

        # Compute expected result
        expected = ref_impl_batchnorm(X, scale, bias, mean, var, epsilon)

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

    print(f"\nBatchNorm Numerical Tests: {passed}/{total} passed")
    assert passed == total, f"Only {passed}/{total} tests passed"


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_errors():
    """Test batchnorm edge cases and boundary conditions"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, epsilon=1e-5):
            self.attrs = {"epsilon": epsilon}

    print("\nTesting BatchNorm Edge Cases:")

    # Test 1: Zero variance (should use epsilon)
    print("  Test 1: Zero variance channels")
    X1 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    scale1 = np.ones(3, dtype=np.float32)
    bias1 = np.zeros(3, dtype=np.float32)
    mean1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    var1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Zero variance
    iTList1 = [
        MockTensor(X1),
        MockTensor(scale1),
        MockTensor(bias1),
        MockTensor(mean1),
        MockTensor(var1),
    ]
    op1 = MockOp(1e-5)
    result1 = compute_batchnorm(iTList1, op1)
    expected1 = ref_impl_batchnorm(X1, scale1, bias1, mean1, var1, 1e-5)
    assert result1.shape == expected1.shape
    np.testing.assert_allclose(result1, expected1, rtol=1e-5, atol=1e-6)
    print("    PASS (handles zero variance with epsilon)")

    # Test 2: Very small variance (numerical stability)
    print("  Test 2: Very small variance")
    X2 = np.random.randn(2, 2, 5, 5).astype(np.float32)
    scale2 = np.ones(2, dtype=np.float32)
    bias2 = np.zeros(2, dtype=np.float32)
    mean2 = np.zeros(2, dtype=np.float32)
    var2 = np.array([1e-10, 1e-9], dtype=np.float32)
    iTList2 = [
        MockTensor(X2),
        MockTensor(scale2),
        MockTensor(bias2),
        MockTensor(mean2),
        MockTensor(var2),
    ]
    op2 = MockOp(1e-5)
    result2 = compute_batchnorm(iTList2, op2)
    expected2 = ref_impl_batchnorm(X2, scale2, bias2, mean2, var2, 1e-5)
    assert result2.shape == expected2.shape
    np.testing.assert_allclose(result2, expected2, rtol=1e-4, atol=1e-5)
    print("    PASS (numerical stability)")

    # Test 3: Very large variance
    print("  Test 3: Very large variance")
    X3 = np.random.randn(2, 3, 4, 4).astype(np.float32) * 1000
    scale3 = np.ones(3, dtype=np.float32)
    bias3 = np.zeros(3, dtype=np.float32)
    mean3 = np.random.randn(3).astype(np.float32)
    var3 = np.array([1e6, 1e7, 1e8], dtype=np.float32)
    iTList3 = [
        MockTensor(X3),
        MockTensor(scale3),
        MockTensor(bias3),
        MockTensor(mean3),
        MockTensor(var3),
    ]
    op3 = MockOp(1e-5)
    result3 = compute_batchnorm(iTList3, op3)
    expected3 = ref_impl_batchnorm(X3, scale3, bias3, mean3, var3, 1e-5)
    assert result3.shape == expected3.shape
    np.testing.assert_allclose(result3, expected3, rtol=1e-4, atol=1e-5)
    print("    PASS (large variance)")

    # Test 4: Large epsilon
    print("  Test 4: Large epsilon value")
    X4 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale4 = np.ones(2, dtype=np.float32)
    bias4 = np.zeros(2, dtype=np.float32)
    mean4 = np.zeros(2, dtype=np.float32)
    var4 = np.ones(2, dtype=np.float32)
    iTList4 = [
        MockTensor(X4),
        MockTensor(scale4),
        MockTensor(bias4),
        MockTensor(mean4),
        MockTensor(var4),
    ]
    op4 = MockOp(1.0)  # Large epsilon
    result4 = compute_batchnorm(iTList4, op4)
    expected4 = ref_impl_batchnorm(X4, scale4, bias4, mean4, var4, 1.0)
    assert result4.shape == expected4.shape
    np.testing.assert_allclose(result4, expected4, rtol=1e-5, atol=1e-6)
    print("    PASS (large epsilon)")

    # Test 5: Negative mean
    print("  Test 5: Large negative mean values")
    X5 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    scale5 = np.ones(3, dtype=np.float32)
    bias5 = np.zeros(3, dtype=np.float32)
    mean5 = np.array([-100.0, -200.0, -300.0], dtype=np.float32)
    var5 = np.ones(3, dtype=np.float32)
    iTList5 = [
        MockTensor(X5),
        MockTensor(scale5),
        MockTensor(bias5),
        MockTensor(mean5),
        MockTensor(var5),
    ]
    op5 = MockOp(1e-5)
    result5 = compute_batchnorm(iTList5, op5)
    expected5 = ref_impl_batchnorm(X5, scale5, bias5, mean5, var5, 1e-5)
    assert result5.shape == expected5.shape
    np.testing.assert_allclose(result5, expected5, rtol=1e-5, atol=1e-6)
    print("    PASS (negative means)")

    # Test 6: Zero scale (unusual but valid)
    print("  Test 6: Zero scale factor")
    X6 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale6 = np.zeros(2, dtype=np.float32)  # Zero scale
    bias6 = np.array([1.0, 2.0], dtype=np.float32)
    mean6 = np.zeros(2, dtype=np.float32)
    var6 = np.ones(2, dtype=np.float32)
    iTList6 = [
        MockTensor(X6),
        MockTensor(scale6),
        MockTensor(bias6),
        MockTensor(mean6),
        MockTensor(var6),
    ]
    op6 = MockOp(1e-5)
    result6 = compute_batchnorm(iTList6, op6)
    expected6 = ref_impl_batchnorm(X6, scale6, bias6, mean6, var6, 1e-5)
    assert result6.shape == expected6.shape
    np.testing.assert_allclose(result6, expected6, rtol=1e-5, atol=1e-6)
    # With zero scale, output should equal bias
    np.testing.assert_allclose(result6[0, 0, :, :], bias6[0], rtol=1e-5, atol=1e-6)
    print("    PASS (zero scale → bias only)")

    # Test 7: All inputs are zeros
    print("  Test 7: All-zero inputs")
    X7 = np.zeros((2, 2, 4, 4), dtype=np.float32)
    scale7 = np.ones(2, dtype=np.float32)
    bias7 = np.zeros(2, dtype=np.float32)
    mean7 = np.zeros(2, dtype=np.float32)
    var7 = np.ones(2, dtype=np.float32)
    iTList7 = [
        MockTensor(X7),
        MockTensor(scale7),
        MockTensor(bias7),
        MockTensor(mean7),
        MockTensor(var7),
    ]
    op7 = MockOp(1e-5)
    result7 = compute_batchnorm(iTList7, op7)
    expected7 = ref_impl_batchnorm(X7, scale7, bias7, mean7, var7, 1e-5)
    assert result7.shape == expected7.shape
    np.testing.assert_allclose(result7, expected7, rtol=1e-5, atol=1e-6)
    print("    PASS (all zeros)")

    print("\nAll edge case tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_precision():
    """Test batchnorm with known precise outputs"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, epsilon=1e-5):
            self.attrs = {"epsilon": epsilon}

    print("\nTesting BatchNorm Precision (Known Outputs):")

    # Test 1: Unit normalization (mean=0, var=1, scale=1, bias=0)
    print("  Test 1: Unit normalization")
    X1 = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    scale1 = np.array([1.0], dtype=np.float32)
    bias1 = np.array([0.0], dtype=np.float32)
    mean1 = np.array([2.5], dtype=np.float32)  # Mean of [1,2,3,4]
    var1 = np.array([1.25], dtype=np.float32)  # Var of [1,2,3,4]
    iTList1 = [
        MockTensor(X1),
        MockTensor(scale1),
        MockTensor(bias1),
        MockTensor(mean1),
        MockTensor(var1),
    ]
    op1 = MockOp(0.0)  # No epsilon for exact calculation
    result1 = compute_batchnorm(iTList1, op1)
    # Normalized: (x - 2.5) / sqrt(1.25) ≈ (x - 2.5) / 1.118
    expected1 = (X1 - 2.5) / np.sqrt(1.25)
    np.testing.assert_allclose(result1, expected1, rtol=1e-6, atol=1e-6)
    print(f"    Result:\n{result1[0, 0]} ✓")

    # Test 2: Simple transformation (mean=0, var=1, scale=2, bias=3)
    print("  Test 2: Scale=2, Bias=3")
    X2 = np.array([[[[0.0, 1.0], [2.0, 3.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    scale2 = np.array([2.0], dtype=np.float32)
    bias2 = np.array([3.0], dtype=np.float32)
    mean2 = np.array([0.0], dtype=np.float32)
    var2 = np.array([1.0], dtype=np.float32)
    iTList2 = [
        MockTensor(X2),
        MockTensor(scale2),
        MockTensor(bias2),
        MockTensor(mean2),
        MockTensor(var2),
    ]
    op2 = MockOp(0.0)
    result2 = compute_batchnorm(iTList2, op2)
    # Output: (x - 0) / sqrt(1) * 2 + 3 = 2x + 3
    expected2 = 2.0 * X2 + 3.0
    np.testing.assert_allclose(result2, expected2, rtol=1e-6, atol=1e-6)
    print(f"    Result:\n{result2[0, 0]} ✓")

    # Test 3: Zero mean, unit variance → identity
    print("  Test 3: Identity transformation")
    X3 = np.array([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=np.float32)  # [1, 2, 1, 2]
    scale3 = np.array([1.0, 1.0], dtype=np.float32)
    bias3 = np.array([0.0, 0.0], dtype=np.float32)
    mean3 = np.zeros(2, dtype=np.float32)
    var3 = np.ones(2, dtype=np.float32)
    iTList3 = [
        MockTensor(X3),
        MockTensor(scale3),
        MockTensor(bias3),
        MockTensor(mean3),
        MockTensor(var3),
    ]
    op3 = MockOp(0.0)
    result3 = compute_batchnorm(iTList3, op3)
    expected3 = X3  # Identity
    np.testing.assert_allclose(result3, expected3, rtol=1e-6, atol=1e-6)
    print(f"    Result equals input ✓")

    # Test 4: Center shift only
    print("  Test 4: Mean subtraction only")
    X4 = np.array([[[[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32)  # [1, 1, 2, 2]
    scale4 = np.array([1.0], dtype=np.float32)
    bias4 = np.array([0.0], dtype=np.float32)
    mean4 = np.array([5.0], dtype=np.float32)
    var4 = np.array([1.0], dtype=np.float32)
    iTList4 = [
        MockTensor(X4),
        MockTensor(scale4),
        MockTensor(bias4),
        MockTensor(mean4),
        MockTensor(var4),
    ]
    op4 = MockOp(0.0)
    result4 = compute_batchnorm(iTList4, op4)
    expected4 = X4 - 5.0  # Just mean subtraction
    np.testing.assert_allclose(result4, expected4, rtol=1e-6, atol=1e-6)
    print(f"    Result:\n{result4[0, 0]} ✓")

    # Test 5: Multiple channels with different parameters
    print("  Test 5: Multi-channel with different params")
    X5 = np.ones((1, 3, 2, 2), dtype=np.float32)  # All ones
    scale5 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bias5 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    mean5 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    var5 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    iTList5 = [
        MockTensor(X5),
        MockTensor(scale5),
        MockTensor(bias5),
        MockTensor(mean5),
        MockTensor(var5),
    ]
    op5 = MockOp(0.0)
    result5 = compute_batchnorm(iTList5, op5)
    # (1 - 1) / 1 * scale + bias = 0 + bias = bias
    expected5_ch0 = np.full((2, 2), 10.0, dtype=np.float32)
    expected5_ch1 = np.full((2, 2), 20.0, dtype=np.float32)
    expected5_ch2 = np.full((2, 2), 30.0, dtype=np.float32)
    np.testing.assert_allclose(result5[0, 0], expected5_ch0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result5[0, 1], expected5_ch1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result5[0, 2], expected5_ch2, rtol=1e-6, atol=1e-6)
    print(f"    Channels: [10, 20, 30] ✓")

    print("\nAll precision tests passed!")


@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_properties():
    """Test mathematical properties of batch normalization"""

    class MockTensor:
        def __init__(self, data):
            self.data = data

    class MockOp:
        def __init__(self, epsilon=1e-5):
            self.attrs = {"epsilon": epsilon}

    print("\nTesting BatchNorm Mathematical Properties:")

    # Property 1: Output shape equals input shape
    print("  Property 1: Shape preservation")
    test_shapes = [(1, 3, 32, 32), (2, 64, 7, 7), (4, 128, 1, 1), (8, 1, 224, 224)]
    for shape in test_shapes:
        X = np.random.randn(*shape).astype(np.float32)
        num_channels = shape[1]
        scale = np.ones(num_channels, dtype=np.float32)
        bias = np.zeros(num_channels, dtype=np.float32)
        mean = np.random.randn(num_channels).astype(np.float32)
        var = np.random.rand(num_channels).astype(np.float32) + 0.1

        iTList = [
            MockTensor(X),
            MockTensor(scale),
            MockTensor(bias),
            MockTensor(mean),
            MockTensor(var),
        ]
        op = MockOp(1e-5)
        result = compute_batchnorm(iTList, op)

        assert (
            result.shape == X.shape
        ), f"Shape not preserved: {result.shape} != {X.shape}"
    print(f"    All shapes preserved ✓")

    # Property 2: Zero scale produces bias only
    print("  Property 2: Zero scale → output = bias")
    X2 = np.random.randn(2, 4, 5, 5).astype(np.float32)
    scale2 = np.zeros(4, dtype=np.float32)
    bias2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mean2 = np.zeros(4, dtype=np.float32)
    var2 = np.ones(4, dtype=np.float32)

    iTList2 = [
        MockTensor(X2),
        MockTensor(scale2),
        MockTensor(bias2),
        MockTensor(mean2),
        MockTensor(var2),
    ]
    op2 = MockOp(1e-5)
    result2 = compute_batchnorm(iTList2, op2)

    for c in range(4):
        np.testing.assert_allclose(result2[0, c, :, :], bias2[c], rtol=1e-5, atol=1e-6)
    print(f"    Zero scale produces bias ✓")

    # Property 3: Linearity in input when mean=0, var=1
    print("  Property 3: Linearity with standard normalization")
    X3 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale3 = np.array([2.0, 3.0], dtype=np.float32)
    bias3 = np.zeros(2, dtype=np.float32)
    mean3 = np.zeros(2, dtype=np.float32)
    var3 = np.ones(2, dtype=np.float32)

    iTList3 = [
        MockTensor(X3),
        MockTensor(scale3),
        MockTensor(bias3),
        MockTensor(mean3),
        MockTensor(var3),
    ]
    op3 = MockOp(0.0)
    result3 = compute_batchnorm(iTList3, op3)

    # Should be linear: output = scale * input (when mean=0, var=1, bias=0)
    np.testing.assert_allclose(
        result3[:, 0, :, :], scale3[0] * X3[:, 0, :, :], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        result3[:, 1, :, :], scale3[1] * X3[:, 1, :, :], rtol=1e-5, atol=1e-6
    )
    print(f"    Linear with standard params ✓")

    # Property 4: Translation invariance with appropriate parameters
    print("  Property 4: Shift invariance")
    X4 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    shift = 100.0
    X4_shifted = X4 + shift

    scale4 = np.ones(3, dtype=np.float32)
    bias4 = np.zeros(3, dtype=np.float32)
    mean4 = np.zeros(3, dtype=np.float32)
    mean4_shifted = mean4 + shift
    var4 = np.ones(3, dtype=np.float32)

    result4_original = compute_batchnorm(
        [
            MockTensor(X4),
            MockTensor(scale4),
            MockTensor(bias4),
            MockTensor(mean4),
            MockTensor(var4),
        ],
        MockOp(1e-5),
    )
    result4_shifted = compute_batchnorm(
        [
            MockTensor(X4_shifted),
            MockTensor(scale4),
            MockTensor(bias4),
            MockTensor(mean4_shifted),
            MockTensor(var4),
        ],
        MockOp(1e-5),
    )

    # Output should be the same when mean is adjusted
    np.testing.assert_allclose(result4_original, result4_shifted, rtol=1e-4, atol=1e-5)
    print(f"    Shift invariance with adjusted mean ✓")

    # Property 5: Scale invariance with appropriate parameters
    print("  Property 5: Scale invariance")
    X5 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale_factor = 10.0
    X5_scaled = X5 * scale_factor

    scale5 = np.ones(2, dtype=np.float32)
    bias5 = np.zeros(2, dtype=np.float32)
    mean5 = np.zeros(2, dtype=np.float32)
    var5 = np.ones(2, dtype=np.float32)
    var5_scaled = var5 * (scale_factor**2)

    result5_original = compute_batchnorm(
        [
            MockTensor(X5),
            MockTensor(scale5),
            MockTensor(bias5),
            MockTensor(mean5),
            MockTensor(var5),
        ],
        MockOp(1e-5),
    )
    result5_scaled = compute_batchnorm(
        [
            MockTensor(X5_scaled),
            MockTensor(scale5),
            MockTensor(bias5),
            MockTensor(mean5),
            MockTensor(var5_scaled),
        ],
        MockOp(1e-5),
    )

    # Output should be the same when variance is adjusted
    np.testing.assert_allclose(result5_original, result5_scaled, rtol=1e-3, atol=1e-4)
    print(f"    Scale invariance with adjusted variance ✓")

    # Property 6: Epsilon prevents division by zero
    print("  Property 6: Epsilon prevents division by zero")
    X6 = np.random.randn(2, 2, 3, 3).astype(np.float32)
    scale6 = np.ones(2, dtype=np.float32)
    bias6 = np.zeros(2, dtype=np.float32)
    mean6 = np.zeros(2, dtype=np.float32)
    var6 = np.zeros(2, dtype=np.float32)  # Zero variance

    iTList6 = [
        MockTensor(X6),
        MockTensor(scale6),
        MockTensor(bias6),
        MockTensor(mean6),
        MockTensor(var6),
    ]
    op6 = MockOp(1e-5)

    # Should not raise division by zero error
    result6 = compute_batchnorm(iTList6, op6)
    assert not np.any(np.isnan(result6)), "NaN values detected"
    assert not np.any(np.isinf(result6)), "Inf values detected"
    print(f"    No NaN or Inf with zero variance ✓")

    print("\nAll property tests passed!")
