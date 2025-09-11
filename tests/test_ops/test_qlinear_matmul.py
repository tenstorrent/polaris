#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from typing import Any
from ttsim.ops.op import QLinearMatMulOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_qlinear_matmul_test_tensors(a_shape, b_shape, a_dtype='uint8', b_dtype='uint8', y_dtype='uint8'):
    """Helper function to create test tensors for QLinearMatMul operation"""

    # Create quantized input tensors
    a_tensor = F._from_shape('a', a_shape, np_dtype=a_dtype)
    b_tensor = F._from_shape('b', b_shape, np_dtype=b_dtype)

    # Create scale tensors (can be scalar or per-channel)
    # For input A: scale can be per-output-channel (last dimension of A)
    a_scale_shape = [a_shape[-1]] if len(a_shape) > 1 else []
    a_scale_tensor = F._from_shape('a_scale', a_scale_shape, np_dtype=np.float32)

    # For input B: scale can be per-output-channel (last dimension of B)
    b_scale_shape = [b_shape[-1]] if len(b_shape) > 1 else []
    b_scale_tensor = F._from_shape('b_scale', b_scale_shape, np_dtype=np.float32)

    # For output: scale can be scalar or per-channel
    y_scale_shape: list[int] = []  # Scalar scale for simplicity
    y_scale_tensor = F._from_shape('y_scale', y_scale_shape, np_dtype=np.float32)

    # Create zero point tensors (same shape as scales)
    a_zero_point_tensor = F._from_shape('a_zero_point', a_scale_shape, np_dtype=a_dtype)
    b_zero_point_tensor = F._from_shape('b_zero_point', b_scale_shape, np_dtype=b_dtype)
    y_zero_point_tensor = F._from_shape('y_zero_point', y_scale_shape, np_dtype=y_dtype)

    input_tensors = [
        a_tensor, a_scale_tensor, a_zero_point_tensor,
        b_tensor, b_scale_tensor, b_zero_point_tensor,
        y_scale_tensor, y_zero_point_tensor
    ]

    input_names = [
        'a', 'a_scale', 'a_zero_point',
        'b', 'b_scale', 'b_zero_point',
        'y_scale', 'y_zero_point'
    ]

    # Create output tensor
    # Compute expected output shape for matrix multiplication
    expected_output_shape = a_shape[:-2] + [a_shape[-2], b_shape[-1]]
    output = make_tensor('y')
    output_tensors = [output]
    output_names = ['y']

    return input_tensors, output_tensors, input_names, output_names, expected_output_shape


def reference_qlinear_matmul(a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point):
    """
    Reference implementation of QLinearMatMul for testing
    Follows the ONNX specification exactly
    """
    # Convert to numpy for easier manipulation
    a = np.asarray(a)
    a_scale = np.asarray(a_scale)
    a_zero_point = np.asarray(a_zero_point)
    b = np.asarray(b)
    b_scale = np.asarray(b_scale)
    b_zero_point = np.asarray(b_zero_point)
    y_scale = np.asarray(y_scale)
    y_zero_point = np.asarray(y_zero_point)

    # Dequantize inputs
    a_dequant = (a.astype(np.float32) - a_zero_point.astype(np.float32)) * a_scale.astype(np.float32)
    b_dequant = (b.astype(np.float32) - b_zero_point.astype(np.float32)) * b_scale.astype(np.float32)

    # Matrix multiplication
    result = np.matmul(a_dequant, b_dequant)

    # Quantize result
    y = np.round(result / y_scale.astype(np.float32)) + y_zero_point.astype(np.float32)

    # Apply saturation based on output zero point type
    if y_zero_point.dtype == np.uint8:
        y = np.clip(y, 0, 255)
    elif y_zero_point.dtype == np.int8:
        y = np.clip(y, -128, 127)

    return y.astype(y_zero_point.dtype)


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_basic_2d():
    """Test basic QLinearMatMul with 2D matrices"""
    print("\n=== Testing Basic QLinearMatMul (2D) ===")

    a_shape = [64, 128]  # [M, K]
    b_shape = [128, 256]  # [K, N]

    input_tensors, output_tensors, input_names, output_names, expected_output_shape = create_qlinear_matmul_test_tensors(
        a_shape, b_shape
    )

    attrs: dict[str, Any] = {}  # QLinearMatMul has no attributes

    op_info = {
        'name': 'test_qlinear_matmul_basic_2d',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute QLinearMatMul operation
    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Validate output dtype (should match y_zero_point dtype)
    assert output_tensors[0].dtype == 'uint8', \
        f"Output dtype mismatch: {output_tensors[0].dtype} != uint8"

    print(f"✓ Basic 2D QLinearMatMul test passed")
    print(f"  A shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  B shape: {input_tensors[3].shape} ({input_tensors[3].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")
    print(f"  Expected shape: {expected_output_shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_with_batch():
    """Test QLinearMatMul with batch dimensions"""
    print("\n=== Testing QLinearMatMul with Batch ===")

    a_shape = [4, 8, 16]  # [batch, M, K]
    b_shape = [4, 16, 32]  # [batch, K, N] - batch dimension will be broadcasted

    input_tensors, output_tensors, input_names, output_names, expected_output_shape = create_qlinear_matmul_test_tensors(
        a_shape, b_shape
    )

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_qlinear_matmul_with_batch',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    print(f"✓ Batched QLinearMatMul test passed")
    print(f"  A shape: {input_tensors[0].shape}")
    print(f"  B shape: {input_tensors[3].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Expected shape: {expected_output_shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_int8():
    """Test QLinearMatMul with int8 quantization"""
    print("\n=== Testing QLinearMatMul (int8) ===")

    a_shape = [32, 64]
    b_shape = [64, 16]

    input_tensors, output_tensors, input_names, output_names, expected_output_shape = create_qlinear_matmul_test_tensors(
        a_shape, b_shape, a_dtype='int8', b_dtype='int8', y_dtype='int8'
    )

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_qlinear_matmul_int8',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'int8'

    print(f"✓ int8 QLinearMatMul test passed")
    print(f"  A shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  B shape: {input_tensors[3].shape} ({input_tensors[3].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_mixed_types():
    """Test QLinearMatMul with mixed input/output types"""
    print("\n=== Testing QLinearMatMul (mixed types) ===")

    a_shape = [16, 24]
    b_shape = [24, 8]

    input_tensors, output_tensors, input_names, output_names, expected_output_shape = create_qlinear_matmul_test_tensors(
        a_shape, b_shape, a_dtype='uint8', b_dtype='int8', y_dtype='uint8'
    )

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_qlinear_matmul_mixed_types',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    print(f"✓ Mixed types QLinearMatMul test passed")
    print(f"  A: {input_tensors[0].dtype}, B: {input_tensors[3].dtype}, Y: {output_tensors[0].dtype}")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_per_channel_scales():
    """Test QLinearMatMul with per-channel scales"""
    print("\n=== Testing QLinearMatMul (per-channel scales) ===")

    a_shape = [2, 12, 16]  # [batch, M, K]
    b_shape = [2, 16, 24]  # [batch, K, N]

    # Create tensors with per-channel scales
    a_tensor = F._from_shape('a', a_shape, np_dtype='uint8')
    b_tensor = F._from_shape('b', b_shape, np_dtype='uint8')

    # Per-channel scales
    a_scale_tensor = F._from_shape('a_scale', [a_shape[-1]], np_dtype=np.float32)  # Per K dimension
    b_scale_tensor = F._from_shape('b_scale', [b_shape[-1]], np_dtype=np.float32)  # Per N dimension
    y_scale_tensor = F._from_shape('y_scale', [], np_dtype=np.float32)  # Scalar

    # Zero points matching scale shapes
    a_zero_point_tensor = F._from_shape('a_zero_point', [a_shape[-1]], np_dtype='uint8')
    b_zero_point_tensor = F._from_shape('b_zero_point', [b_shape[-1]], np_dtype='uint8')
    y_zero_point_tensor = F._from_shape('y_zero_point', [], np_dtype='uint8')

    input_tensors = [
        a_tensor, a_scale_tensor, a_zero_point_tensor,
        b_tensor, b_scale_tensor, b_zero_point_tensor,
        y_scale_tensor, y_zero_point_tensor
    ]

    input_names = [
        'a', 'a_scale', 'a_zero_point',
        'b', 'b_scale', 'b_zero_point',
        'y_scale', 'y_zero_point'
    ]

    output = make_tensor('y')
    output_tensors = [output]
    output_names = ['y']

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_qlinear_matmul_per_channel_scales',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = a_shape[:-2] + [a_shape[-2], b_shape[-1]]
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    print(f"✓ Per-channel scales QLinearMatMul test passed")
    print(f"  A scale shape: {input_tensors[1].shape} (per-channel)")
    print(f"  B scale shape: {input_tensors[4].shape} (per-channel)")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_invalid_inputs():
    """Test QLinearMatMul operation with invalid inputs"""
    print("\n=== Testing QLinearMatMul Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('input', [8], np_dtype='uint8')]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_qlinear_matmul_wrong_inputs',
            'optype': 'QLinearMatMul',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = QLinearMatMulOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Incompatible matrix dimensions
    try:
        a_tensor = F._from_shape('a', [4, 8], np_dtype='uint8')  # [M=4, K=8]
        b_tensor = F._from_shape('b', [16, 12], np_dtype='uint8')  # [K=16, N=12] - K doesn't match

        input_tensors = [
            a_tensor, F._from_shape('a_scale', [], np_dtype=np.float32),
            F._from_shape('a_zero_point', [], np_dtype='uint8'),
            b_tensor, F._from_shape('b_scale', [], np_dtype=np.float32),
            F._from_shape('b_zero_point', [], np_dtype='uint8'),
            F._from_shape('y_scale', [], np_dtype=np.float32),
            F._from_shape('y_zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_qlinear_matmul_incompatible_dims',
            'optype': 'QLinearMatMul',
            'inList': ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = QLinearMatMulOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for incompatible dimensions"
    except AssertionError as e:
        print(f"✓ Correctly caught incompatible matrix dimensions: {e}")

    # Test 3: Wrong input dtype
    try:
        input_tensors = [
            F._from_shape('a', [4, 8], np_dtype=np.float32),  # Wrong dtype
            F._from_shape('a_scale', [], np_dtype=np.float32),
            F._from_shape('a_zero_point', [], np_dtype='uint8'),
            F._from_shape('b', [8, 12], np_dtype='uint8'),
            F._from_shape('b_scale', [], np_dtype=np.float32),
            F._from_shape('b_zero_point', [], np_dtype='uint8'),
            F._from_shape('y_scale', [], np_dtype=np.float32),
            F._from_shape('y_zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_qlinear_matmul_wrong_dtype',
            'optype': 'QLinearMatMul',
            'inList': ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = QLinearMatMulOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input dtype"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input dtype: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_factory():
    """Test that QLinearMatMul operation can be created through SimOpFactory"""
    print("\n=== Testing QLinearMatMul Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    qlinear_matmul_class = SimOpFactory('QLinearMatMul')
    assert qlinear_matmul_class == QLinearMatMulOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_qlinear_matmul',
        'optype': 'QLinearMatMul',
        'inList': ['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
        'outList': ['y'],
        'attrs': {},
    }

    op_obj = qlinear_matmul_class(op_info)
    assert isinstance(op_obj, QLinearMatMulOp)
    assert op_obj.optype == 'QLinearMatMul'
    assert op_obj.name == 'factory_test_qlinear_matmul'

    print("✓ QLinearMatMul factory creation test passed")
    print(f"  Created operation: {op_obj}")


@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_matmul_transformer_scenario():
    """Test QLinearMatMul in a realistic transformer scenario"""
    print("\n=== Testing QLinearMatMul Transformer Scenario ===")

    # Typical transformer linear layer: quantized weights and activations
    batch_size = 8
    seq_length = 512
    hidden_size = 768
    ffn_size = 3072

    # Query-Key-Value projection matrices (quantized)
    qkv_shape = [hidden_size, hidden_size]  # [hidden_size, hidden_size]
    activations_shape = [batch_size, seq_length, hidden_size]  # [batch, seq, hidden]

    # Create quantized activations (input A)
    a_tensor = F._from_shape('activations', activations_shape, np_dtype='uint8')
    a_scale_tensor = F._from_shape('a_scale', [activations_shape[-1]], np_dtype=np.float32)
    a_zero_point_tensor = F._from_shape('a_zero_point', [activations_shape[-1]], np_dtype='uint8')

    # Create quantized weights (input B) - for QKV projection
    b_tensor = F._from_shape('qkv_weights', qkv_shape, np_dtype='int8')
    b_scale_tensor = F._from_shape('b_scale', [qkv_shape[-1]], np_dtype=np.float32)
    b_zero_point_tensor = F._from_shape('b_zero_point', [qkv_shape[-1]], np_dtype='int8')

    # Output quantization parameters
    y_scale_tensor = F._from_shape('y_scale', [], np_dtype=np.float32)
    y_zero_point_tensor = F._from_shape('y_zero_point', [], np_dtype='uint8')

    input_tensors = [
        a_tensor, a_scale_tensor, a_zero_point_tensor,
        b_tensor, b_scale_tensor, b_zero_point_tensor,
        y_scale_tensor, y_zero_point_tensor
    ]

    input_names = [
        'activations', 'a_scale', 'a_zero_point',
        'qkv_weights', 'b_scale', 'b_zero_point',
        'y_scale', 'y_zero_point'
    ]

    output = make_tensor('qkv_output')
    output_tensors = [output]
    output_names = ['qkv_output']

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_qlinear_matmul_transformer',
        'optype': 'QLinearMatMul',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QLinearMatMulOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = [batch_size, seq_length, hidden_size]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ Transformer QLinearMatMul test passed")
    print(f"  Transformer QKV projection: activations {activations_shape} × weights {qkv_shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")
    print(f"  Memory: {perf_stats['inBytes']} input bytes, {perf_stats['outBytes']} output bytes")


if __name__ == '__main__':
    # Run all tests manually
    test_qlinear_matmul_factory()
    test_qlinear_matmul_basic_2d()
    test_qlinear_matmul_with_batch()
    test_qlinear_matmul_int8()
    test_qlinear_matmul_mixed_types()
    test_qlinear_matmul_per_channel_scales()
    test_qlinear_matmul_invalid_inputs()
    test_qlinear_matmul_transformer_scenario()

    print("\n🎉 All QLinearMatMul operation tests completed successfully!")
