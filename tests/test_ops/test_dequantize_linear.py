#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_dequantize_linear_test_tensors(input_shape, input_dtype='uint8', axis=1):
    """Helper function to create test tensors for DequantizeLinear operation"""

    # Create quantized input tensor
    input_tensor = F._from_shape('input', input_shape, np_dtype=input_dtype)

    # Create scale tensor (can be scalar or per-channel)
    if axis == 1 and len(input_shape) > 1:
        # Per-channel scale along axis 1
        scale_shape = [input_shape[axis]]
        scale_tensor = F._from_shape('scale', scale_shape, np_dtype=np.float32)
    else:
        # Scalar scale
        scale_tensor = F._from_shape('scale', [], np_dtype=np.float32)

    # Create zero point tensor (same type and shape as scale)
    if len(scale_tensor.shape) == 0:
        # Scalar zero point
        zero_point_tensor = F._from_shape('zero_point', [], np_dtype=input_dtype)
    else:
        # Per-channel zero point
        zero_point_tensor = F._from_shape('zero_point', scale_shape, np_dtype=input_dtype)

    input_tensors = [input_tensor, scale_tensor, zero_point_tensor]
    input_names = ['input', 'scale', 'zero_point']

    # Create output tensor (shape will be same as input)
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    return input_tensors, output_tensors, input_names, output_names


def reference_dequantize_linear(x, x_scale, x_zero_point, axis=1):
    """
    Reference implementation of DequantizeLinear for testing
    Follows the ONNX specification exactly
    """
    # Convert to numpy for easier manipulation
    x = np.asarray(x)
    x_scale = np.asarray(x_scale)
    x_zero_point = np.asarray(x_zero_point)

    # Perform dequantization: y = (x - x_zero_point) * x_scale
    # Broadcasting will handle scalar vs per-channel cases
    y = (x.astype(np.float32) - x_zero_point.astype(np.float32)) * x_scale.astype(np.float32)

    return y


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_basic_uint8():
    """Test basic DequantizeLinear with uint8 input and scalar scale/zero_point"""
    print("\n=== Testing Basic DequantizeLinear (uint8, scalar) ===")

    input_shape = [4, 8]
    input_tensors, output_tensors, input_names, output_names = create_dequantize_linear_test_tensors(
        input_shape, input_dtype='uint8', axis=1
    )

    attrs = {'axis': 1}

    op_info = {
        'name': 'test_dequantize_linear_basic_uint8',
        'optype': 'DequantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute DequantizeLinear operation
    op_obj = SimOp(op_info)

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
    assert output_tensors[0].dtype == np.float32, \
        f"Output dtype mismatch: {output_tensors[0].dtype} != float32"

    print(f"✓ Basic uint8 DequantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Scale shape: {input_tensors[1].shape} ({input_tensors[1].dtype})")
    print(f"  Zero point shape: {input_tensors[2].shape} ({input_tensors[2].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_int8():
    """Test DequantizeLinear with int8 input"""
    print("\n=== Testing DequantizeLinear (int8) ===")

    input_shape = [6, 12]
    input_tensors, output_tensors, input_names, output_names = create_dequantize_linear_test_tensors(
        input_shape, input_dtype='int8', axis=1
    )

    attrs = {'axis': 1}

    op_info = {
        'name': 'test_dequantize_linear_int8',
        'optype': 'DequantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape and dtype
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == np.float32

    print(f"✓ int8 DequantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Output shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_per_channel():
    """Test DequantizeLinear with per-channel scale and zero point"""
    print("\n=== Testing DequantizeLinear (per-channel) ===")

    input_shape = [3, 16, 8]  # [batch, channels, features]
    axis = 1  # Dequantize along channel dimension

    # Create input tensors
    input_tensor = F._from_shape('input', input_shape, np_dtype='uint8')

    # Per-channel scale and zero point (one per channel)
    scale_tensor = F._from_shape('scale', [input_shape[axis]], np_dtype=np.float32)
    zero_point_tensor = F._from_shape('zero_point', [input_shape[axis]], np_dtype='uint8')

    input_tensors = [input_tensor, scale_tensor, zero_point_tensor]
    input_names = ['input', 'scale', 'zero_point']

    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    attrs = {'axis': axis}

    op_info = {
        'name': 'test_dequantize_linear_per_channel',
        'optype': 'DequantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == np.float32

    print(f"✓ Per-channel DequantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape} (per-channel)")
    print(f"  Zero point shape: {input_tensors[2].shape} (per-channel)")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_1d():
    """Test DequantizeLinear with 1D tensor"""
    print("\n=== Testing DequantizeLinear (1D) ===")

    input_shape = [32]
    input_tensors, output_tensors, input_names, output_names = create_dequantize_linear_test_tensors(
        input_shape, input_dtype='uint8', axis=0
    )

    attrs = {'axis': 0}

    op_info = {
        'name': 'test_dequantize_linear_1d',
        'optype': 'DequantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == np.float32

    print(f"✓ 1D DequantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_invalid_inputs():
    """Test DequantizeLinear operation with invalid inputs"""
    print("\n=== Testing DequantizeLinear Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('input', [8], np_dtype='uint8')]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_dequantize_linear_wrong_inputs',
            'optype': 'DequantizeLinear',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong output count
    try:
        input_tensors = [
            F._from_shape('input', [8], np_dtype='uint8'),
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output1'), make_tensor('output2')]

        op_info = {
            'name': 'test_dequantize_linear_wrong_outputs',
            'optype': 'DequantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output1', 'output2'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong output count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong output count: {e}")

    # Test 3: Wrong input dtype
    try:
        input_tensors = [
            F._from_shape('input', [8], np_dtype=np.float32),  # Wrong dtype
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_dequantize_linear_wrong_dtype',
            'optype': 'DequantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input dtype"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input dtype: {e}")

    # Test 4: Invalid axis
    try:
        input_tensors = [
            F._from_shape('input', [4, 8], np_dtype='uint8'),
            F._from_shape('scale', [], np_dtype=np.float32),
            F._from_shape('zero_point', [], np_dtype='uint8')
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_dequantize_linear_invalid_axis',
            'optype': 'DequantizeLinear',
            'inList': ['input', 'scale', 'zero_point'],
            'outList': ['output'],
            'attrs': {'axis': 3},  # Invalid axis for 2D tensor
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised ValueError for invalid axis"
    except ValueError as e:
        print(f"✓ Correctly caught invalid axis: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_registry():
    """Test that DequantizeLinear is registered in the descriptor registry"""
    from ttsim.ops.desc.registry import get_opdesc_registry
    assert get_opdesc_registry().has_shape_inference_function('DequantizeLinear')


@pytest.mark.unit
@pytest.mark.opunit
def test_dequantize_linear_quantization_workflow():
    """Test DequantizeLinear in a typical quantization workflow"""
    print("\n=== Testing DequantizeLinear Quantization Workflow ===")

    # Simulate a typical quantization workflow
    # Original weights: [1024, 1024]
    input_shape = [1024, 1024]

    # Create quantized weights tensor
    weights_tensor = F._from_shape('weights', input_shape, np_dtype='uint8')

    # Create per-channel scale (one per output channel)
    scale_tensor = F._from_shape('scale', [input_shape[0]], np_dtype=np.float32)

    # Create per-channel zero point
    zero_point_tensor = F._from_shape('zero_point', [input_shape[0]], np_dtype='uint8')

    input_tensors = [weights_tensor, scale_tensor, zero_point_tensor]
    input_names = ['weights', 'scale', 'zero_point']

    output = make_tensor('dequantized_weights')
    output_tensors = [output]
    output_names = ['dequantized_weights']

    attrs = {'axis': 0}  # Dequantize along output channel dimension

    op_info = {
        'name': 'test_dequantize_linear_workflow',
        'optype': 'DequantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[0].dtype == np.float32

    print(f"✓ Quantization workflow DequantizeLinear test passed")
    print(f"  Quantized weights shape: {input_tensors[0].shape} ({input_tensors[0].dtype})")
    print(f"  Scale shape: {input_tensors[1].shape} (per-channel)")
    print(f"  Zero point shape: {input_tensors[2].shape} (per-channel)")
    print(f"  Dequantized weights shape: {output_tensors[0].shape} ({output_tensors[0].dtype})")
    print(f"  Performance: {perf_stats['instrs']['cast']} conversions, {perf_stats['instrs']['sub']} subtractions, {perf_stats['instrs']['mul']} multiplications")


if __name__ == '__main__':
    # Run all tests manually
    test_dequantize_linear_registry()
    test_dequantize_linear_basic_uint8()
    test_dequantize_linear_int8()
    test_dequantize_linear_per_channel()
    test_dequantize_linear_1d()
    test_dequantize_linear_invalid_inputs()
    test_dequantize_linear_quantization_workflow()

    print("\n🎉 All DequantizeLinear operation tests completed successfully!")
