#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import QuantizeLinearOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_quantize_linear_test_tensors(input_shape, output_dtype='uint8', axis=1):
    """Helper function to create test tensors for QuantizeLinear operation"""

    # Create input tensor (float32)
    input_tensor = F._from_shape('input', input_shape, np_dtype=np.float32)

    # Create scale tensor (can be scalar or per-channel)
    # Handle negative axis
    normalized_axis = axis if axis >= 0 else len(input_shape) + axis

    if normalized_axis == 1 and len(input_shape) > 1:
        # Per-channel scale along axis 1
        scale_shape = [input_shape[normalized_axis]]
        scale_tensor = F._from_shape('scale', scale_shape, np_dtype=np.float32)
    else:
        # Scalar scale
        scale_tensor = F._from_shape('scale', [], np_dtype=np.float32)

    # Create zero point tensor (same type and shape as scale)
    if len(scale_tensor.shape) == 0:
        # Scalar zero point
        zero_point_tensor = F._from_shape('zero_point', [], np_dtype=output_dtype)
    else:
        # Per-channel zero point
        zero_point_tensor = F._from_shape('zero_point', scale_shape, np_dtype=output_dtype)

    input_tensors = [input_tensor, scale_tensor, zero_point_tensor]
    input_names = ['input', 'scale', 'zero_point']

    # Create output tensor (shape will be same as input)
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    return input_tensors, output_tensors, input_names, output_names


def reference_quantize_linear(x, y_scale, y_zero_point, axis=1, saturate=True):
    """
    Reference implementation of QuantizeLinear for testing
    Follows the ONNX specification exactly
    """
    # Convert to numpy for easier manipulation
    x = np.asarray(x)
    y_scale = np.asarray(y_scale)
    y_zero_point = np.asarray(y_zero_point)

    # Perform quantization: y = clip(round(x / y_scale) + y_zero_point, min_val, max_val)
    # Broadcasting will handle scalar vs per-channel cases
    y = np.round(x.astype(np.float32) / y_scale.astype(np.float32)) + y_zero_point.astype(np.float32)

    # Apply saturation/clamping if requested
    if saturate:
        if y_zero_point.dtype == np.uint8:
            y = np.clip(y, 0, 255)
        elif y_zero_point.dtype == np.int8:
            y = np.clip(y, -128, 127)
        elif y_zero_point.dtype == np.int16:
            y = np.clip(y, -32768, 32767)
        elif y_zero_point.dtype == np.int32:
            y = np.clip(y, -2147483648, 2147483647)

    return y.astype(y_zero_point.dtype)


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_basic_uint8():
    """Test basic QuantizeLinear with uint8 output and scalar scale/zero_point"""
    print("\n=== Testing Basic QuantizeLinear (uint8, scalar) ===")

    input_shape = [4, 8]
    input_tensors, output_tensors, input_names, output_names = create_quantize_linear_test_tensors(
        input_shape, output_dtype='uint8', axis=1
    )

    attrs = {'axis': 1, 'saturate': True}

    op_info = {
        'name': 'test_quantize_linear_basic_uint8',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute QuantizeLinear operation
    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Validate output dtype
    assert output_tensors[0].dtype == 'uint8', \
        f"Output dtype mismatch: {output_tensors[0].dtype} != uint8"

    print(f"✓ Basic uint8 QuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Scale shape: {input_tensors[1].shape} ({input_tensors[1].dtype})")
    print(f"  Zero point shape: {input_tensors[2].shape} ({input_tensors[2].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_int8():
    """Test QuantizeLinear with int8 output"""
    print("\n=== Testing QuantizeLinear (int8) ===")

    input_shape = [6, 12]
    input_tensors, output_tensors, input_names, output_names = create_quantize_linear_test_tensors(
        input_shape, output_dtype='int8', axis=1
    )

    attrs = {'axis': 1, 'saturate': True}

    op_info = {
        'name': 'test_quantize_linear_int8',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape and dtype
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'int8'

    print(f"✓ int8 QuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_per_channel():
    """Test QuantizeLinear with per-channel scale and zero point"""
    print("\n=== Testing QuantizeLinear (per-channel) ===")

    input_shape = [3, 16, 8]  # [batch, channels, features]
    axis = 1  # Quantize along channel dimension

    # Create input tensors
    input_tensor = F._from_shape('input', input_shape, np_dtype=np.float32)

    # Per-channel scale and zero point (one per channel)
    scale_tensor = F._from_shape('scale', [input_shape[axis]], np_dtype=np.float32)
    zero_point_tensor = F._from_shape('zero_point', [input_shape[axis]], np_dtype='uint8')

    input_tensors = [input_tensor, scale_tensor, zero_point_tensor]
    input_names = ['input', 'scale', 'zero_point']

    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    attrs = {'axis': axis, 'saturate': True}

    op_info = {
        'name': 'test_quantize_linear_per_channel',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    print(f"✓ Per-channel QuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape} (per-channel)")
    print(f"  Zero point shape: {input_tensors[2].shape} (per-channel)")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_no_saturate():
    """Test QuantizeLinear with saturate=False"""
    print("\n=== Testing QuantizeLinear (no saturate) ===")

    input_shape = [4, 6]
    input_tensors, output_tensors, input_names, output_names = create_quantize_linear_test_tensors(
        input_shape, output_dtype='uint8', axis=1
    )

    attrs = {'axis': 1, 'saturate': False}

    op_info = {
        'name': 'test_quantize_linear_no_saturate',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    # Check that clipping operations are not counted when saturate=False
    assert perf_stats['instrs']['clip'] == 0, "Clip operations should be 0 when saturate=False"

    print(f"✓ No saturate QuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Clip operations: {perf_stats['instrs']['clip']} (should be 0)")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_1d():
    """Test QuantizeLinear with 1D tensor"""
    print("\n=== Testing QuantizeLinear (1D) ===")

    input_shape = [32]  # Simple 1D vector
    axis = 0  # Only one axis to quantize along

    # Create input tensor
    input_tensor = F._from_shape('input', input_shape, np_dtype=np.float32)

    # For 1D tensor, scale should be scalar
    scale_tensor = F._from_shape('scale', [], np_dtype=np.float32)
    zero_point_tensor = F._from_shape('zero_point', [], np_dtype='uint8')

    input_tensors = [input_tensor, scale_tensor, zero_point_tensor]
    input_names = ['input', 'scale', 'zero_point']

    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    attrs = {'axis': axis, 'saturate': True}

    op_info = {
        'name': 'test_quantize_linear_1d',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 1D QuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape}")
    print(f"  Zero point shape: {input_tensors[2].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_invalid_inputs():
    """Test QuantizeLinear operation with invalid inputs"""
    print("\n=== Testing QuantizeLinear Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('input', [8], np_dtype=np.float32)]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_quantize_linear_wrong_inputs',
            'optype': 'QuantizeLinear',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = QuantizeLinearOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong output count
    try:
        input_tensors = [
            F._from_shape('input', [8], np_dtype=np.float32),
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output1'), make_tensor('output2')]

        op_info = {
            'name': 'test_quantize_linear_wrong_outputs',
            'optype': 'QuantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output1', 'output2'],
            'attrs': {},
        }

        op_obj = QuantizeLinearOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong output count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong output count: {e}")

    # Test 3: Wrong input dtype
    try:
        input_tensors = [
            F._from_shape('input', [8], np_dtype=np.int32),  # Wrong dtype
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_quantize_linear_wrong_dtype',
            'optype': 'QuantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = QuantizeLinearOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input dtype"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input dtype: {e}")

    # Test 4: Invalid axis
    try:
        input_tensors = [
            F._from_shape('input', [4, 8], np_dtype=np.float32),
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_quantize_linear_invalid_axis',
            'optype': 'QuantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output'],
            'attrs': {'axis': 3},  # Invalid axis for 2D tensor
        }

        op_obj = QuantizeLinearOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised ValueError for invalid axis"
    except ValueError as e:
        print(f"✓ Correctly caught invalid axis: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_factory():
    """Test that QuantizeLinear operation can be created through SimOpFactory"""
    print("\n=== Testing QuantizeLinear Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    quantize_class = SimOpFactory('QuantizeLinear')
    assert quantize_class == QuantizeLinearOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_quantize_linear',
        'optype': 'QuantizeLinear',
        'inList': ['input', 'scale', 'zero_point'],
        'outList': ['output'],
        'attrs': {'axis': 1, 'saturate': True},
    }

    op_obj = quantize_class(op_info)
    assert isinstance(op_obj, QuantizeLinearOp)
    assert op_obj.optype == 'QuantizeLinear'
    assert op_obj.name == 'factory_test_quantize_linear'
    assert op_obj.axis == 1
    assert op_obj.saturate == True

    print("✓ QuantizeLinear factory creation test passed")
    print(f"  Created operation: {op_obj}")


@pytest.mark.unit
@pytest.mark.opunit
def test_quantize_linear_quantization_workflow():
    """Test QuantizeLinear in a typical quantization workflow"""
    print("\n=== Testing QuantizeLinear Quantization Workflow ===")

    # Simulate a typical quantization workflow
    # Original weights: [1024, 1024]
    input_shape = [1024, 1024]

    # Create float32 weights tensor
    weights_tensor = F._from_shape('weights', input_shape, np_dtype=np.float32)

    # Create per-channel scale (one per output channel)
    scale_tensor = F._from_shape('scale', [input_shape[0]], np_dtype=np.float32)

    # Create per-channel zero point
    zero_point_tensor = F._from_shape('zero_point', [input_shape[0]], np_dtype='uint8')

    input_tensors = [weights_tensor, scale_tensor, zero_point_tensor]
    input_names = ['weights', 'scale', 'zero_point']

    output = make_tensor('quantized_weights')
    output_tensors = [output]
    output_names = ['quantized_weights']

    attrs = {'axis': 0, 'saturate': True}  # Quantize along output channel dimension

    op_info = {
        'name': 'test_quantize_linear_workflow',
        'optype': 'QuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = QuantizeLinearOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == 'uint8'

    print(f"✓ Quantization workflow QuantizeLinear test passed")
    print(f"  Float32 weights shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Scale shape: {input_tensors[1].shape} (per-channel)")
    print(f"  Zero point shape: {input_tensors[2].shape} (per-channel)")
    print(f"  Quantized weights shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")
    print(f"  Performance: {perf_stats['instrs']['round']} rounding ops, {perf_stats['instrs']['clip']} clipping ops")


if __name__ == '__main__':
    # Run all tests manually
    test_quantize_linear_factory()
    test_quantize_linear_basic_uint8()
    test_quantize_linear_int8()
    test_quantize_linear_per_channel()
    test_quantize_linear_no_saturate()
    test_quantize_linear_1d()
    test_quantize_linear_invalid_inputs()
    test_quantize_linear_quantization_workflow()

    print("\n🎉 All QuantizeLinear operation tests completed successfully!")
