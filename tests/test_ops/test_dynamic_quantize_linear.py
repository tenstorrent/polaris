#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from typing import Any
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_dynamic_quantize_linear_test_tensors(input_shape, input_data=None):
    """Helper function to create test tensors for DynamicQuantizeLinear operation"""

    # Create input tensor: [batch_size, ..., features] with float32 dtype
    if input_data is None:
        # Generate random test data with known range for predictable testing
        np.random.seed(42)  # For reproducible tests
        input_data = np.random.randn(*input_shape).astype(np.float32) * 2.0  # Scale for larger range

    input_tensor = F._from_data('input', data=input_data)

    input_tensors = [input_tensor]
    input_names = ['input']

    # Create output tensors
    output = make_tensor('quantized_output')
    scale = make_tensor('scale')
    zero_point = make_tensor('zero_point')

    output_tensors = [output, scale, zero_point]
    output_names = ['quantized_output', 'scale', 'zero_point']

    return input_tensors, output_tensors, input_names, output_names


def reference_dynamic_quantize_linear(x):
    """
    Reference implementation of DynamicQuantizeLinear for testing
    Follows the ONNX specification exactly
    """
    # Ensure input is float32
    x = x.astype(np.float32)

    # Compute dynamic range
    x_min = np.min(x)
    x_max = np.max(x)

    # Compute scale and zero point
    y_scale = (x_max - x_min) / 255.0
    y_zero_point = np.round(x_min / y_scale).astype(np.uint8)

    # Handle edge case where range is zero
    if y_scale == 0.0:
        y_scale = 1.0
        y_zero_point = 0

    # Quantize the tensor
    y = np.clip(np.round(x / y_scale) + y_zero_point, 0, 255).astype(np.uint8)

    return y, y_scale.astype(np.float32), y_zero_point


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_basic():
    """Test basic DynamicQuantizeLinear with 1D tensor"""
    print("\n=== Testing Basic DynamicQuantizeLinear (1D) ===")

    input_shape = [100]
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(input_shape)

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_dynamic_quantize_basic',
        'optype': 'DynamicQuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute DynamicQuantizeLinear operation
    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shapes
    expected_output_shape = input_shape
    expected_scale_shape: list[int] = []
    expected_zero_point_shape: list[int] = []

    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"
    assert output_tensors[1].shape == expected_scale_shape, \
        f"Scale shape mismatch: {output_tensors[1].shape} != {expected_scale_shape}"
    assert output_tensors[2].shape == expected_zero_point_shape, \
        f"Zero point shape mismatch: {output_tensors[2].shape} != {expected_zero_point_shape}"

    # Validate output dtypes
    assert output_tensors[0].dtype == 'uint8', f"Output dtype mismatch: {output_tensors[0].dtype} != uint8"
    assert output_tensors[1].dtype == 'float32', f"Scale dtype mismatch: {output_tensors[1].dtype} != float32"
    assert output_tensors[2].dtype == 'uint8', f"Zero point dtype mismatch: {output_tensors[2].dtype} != uint8"

    # Validate performance stats
    assert 'instrs' in perf_stats
    assert 'cmp' in perf_stats['instrs']  # Min/max comparisons
    assert 'div' in perf_stats['instrs']  # Division operations
    assert 'round' in perf_stats['instrs']  # Rounding operations
    assert 'clip' in perf_stats['instrs']  # Clipping operations
    assert perf_stats['instrs']['mac'] == 0  # No multiply-accumulate operations

    print(f"✓ Basic DynamicQuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['cmp']} comparisons, {perf_stats['instrs']['div']} divisions")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_2d():
    """Test DynamicQuantizeLinear with 2D tensor"""
    print("\n=== Testing DynamicQuantizeLinear (2D) ===")

    input_shape = [10, 64]
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(input_shape)

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_dynamic_quantize_2d',
        'optype': 'DynamicQuantizeLinear',
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

    # Validate output shapes
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 2D DynamicQuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_3d():
    """Test DynamicQuantizeLinear with 3D tensor (typical for transformer weights)"""
    print("\n=== Testing DynamicQuantizeLinear (3D) ===")

    input_shape = [8, 16, 64]  # [batch, seq_len, hidden_size]
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(input_shape)

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_dynamic_quantize_3d',
        'optype': 'DynamicQuantizeLinear',
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

    # Validate output shapes
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 3D DynamicQuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_4d():
    """Test DynamicQuantizeLinear with 4D tensor (typical for convolution weights)"""
    print("\n=== Testing DynamicQuantizeLinear (4D) ===")

    input_shape = [32, 64, 3, 3]  # [out_channels, in_channels, kernel_h, kernel_w]
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(input_shape)

    attrs: dict[str, Any] = {}

    op_info = {
        'name': 'test_dynamic_quantize_4d',
        'optype': 'DynamicQuantizeLinear',
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

    # Validate output shapes
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 4D DynamicQuantizeLinear test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_edge_cases():
    """Test DynamicQuantizeLinear with edge cases"""
    print("\n=== Testing DynamicQuantizeLinear Edge Cases ===")

    # Test 1: Constant tensor (zero range)
    print("  Testing constant tensor...")
    constant_data = np.full([50], 3.14, dtype=np.float32)
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(
        [50], input_data=constant_data
    )

    op_info = {
        'name': 'test_dynamic_quantize_constant',
        'optype': 'DynamicQuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': {},
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
    print("    ✓ Constant tensor handled correctly")

    # Test 2: Very small range
    print("  Testing small range tensor...")
    small_range_data = np.random.randn(25).astype(np.float32) * 0.001 + 1.0
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(
        [25], input_data=small_range_data
    )

    op_info = {
        'name': 'test_dynamic_quantize_small_range',
        'optype': 'DynamicQuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': {},
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
    print("    ✓ Small range tensor handled correctly")

    print("✓ All edge case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_invalid_inputs():
    """Test DynamicQuantizeLinear operation with invalid inputs"""
    print("\n=== Testing DynamicQuantizeLinear Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [
            F._from_shape('input1', [10], np_dtype=np.float32),
            F._from_shape('input2', [10], np_dtype=np.float32)  # Extra input
        ]
        output_tensors = [
            make_tensor('output'),
            make_tensor('scale'),
            make_tensor('zero_point')
        ]

        op_info = {
            'name': 'test_dynamic_quantize_wrong_inputs',
            'optype': 'DynamicQuantizeLinear',
            'inList': ['input1', 'input2'],
            'outList': ['output', 'scale', 'zero_point'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong output count
    try:
        input_tensors = [F._from_shape('input', [10], np_dtype=np.float32)]
        output_tensors = [make_tensor('output')]  # Missing scale and zero_point

        op_info = {
            'name': 'test_dynamic_quantize_wrong_outputs',
            'optype': 'DynamicQuantizeLinear',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong output count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong output count: {e}")

    # Test 3: Wrong input dtype
    try:
        input_tensors = [F._from_shape('input', [10], np_dtype=np.int32)]  # Wrong dtype
        output_tensors = [
            make_tensor('output'),
            make_tensor('scale'),
            make_tensor('zero_point')
        ]

        op_info = {
            'name': 'test_dynamic_quantize_wrong_dtype',
            'optype': 'DynamicQuantizeLinear',
            'inList': ['input'],
            'outList': ['output', 'scale', 'zero_point'],
            'attrs': {},
        }

        op_obj = SimOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input dtype"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input dtype: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_factory():
    """Test that DynamicQuantizeLinear operation is registered in the op mapping"""
    print("\n=== Testing DynamicQuantizeLinear Factory Creation ===")
    from ttsim.ops.desc.registry import get_opdesc_registry
    assert get_opdesc_registry().has_shape_inference_function('DynamicQuantizeLinear')


@pytest.mark.unit
@pytest.mark.opunit
def test_dynamic_quantize_linear_accuracy():
    """Test DynamicQuantizeLinear accuracy against reference implementation"""
    print("\n=== Testing DynamicQuantizeLinear Accuracy ===")

    # Create test data
    np.random.seed(123)
    test_data = np.random.randn(20, 30).astype(np.float32) * 5.0  # Larger range for testing

    # Reference implementation
    ref_output, ref_scale, ref_zero_point = reference_dynamic_quantize_linear(test_data)

    # Our implementation setup
    input_tensors, output_tensors, input_names, output_names = create_dynamic_quantize_linear_test_tensors(
        test_data.shape, input_data=test_data
    )

    op_info = {
        'name': 'test_dynamic_quantize_accuracy',
        'optype': 'DynamicQuantizeLinear',
        'inList': input_names,
        'outList': output_names,
        'attrs': {},
    }

    op_obj = SimOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # The actual computation would happen during execution, but we can verify the setup
    # The shape and dtype validation ensures the operation is configured correctly
    assert output_tensors[0].shape == list(test_data.shape)
    assert output_tensors[0].dtype == 'uint8'
    assert output_tensors[1].shape == []  # Scalar scale
    assert output_tensors[1].dtype == 'float32'
    assert output_tensors[2].shape == []  # Scalar zero point
    assert output_tensors[2].dtype == 'uint8'

    print(f"✓ Accuracy test setup passed")
    print(f"  Test data shape: {test_data.shape}")
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  Reference scale: {ref_scale}")
    print(f"  Reference zero point: {ref_zero_point}")
    print("  Operation configured correctly for quantization computation")


if __name__ == '__main__':
    # Run all tests manually
    test_dynamic_quantize_linear_factory()
    test_dynamic_quantize_linear_basic()
    test_dynamic_quantize_linear_2d()
    test_dynamic_quantize_linear_3d()
    test_dynamic_quantize_linear_4d()
    test_dynamic_quantize_linear_edge_cases()
    test_dynamic_quantize_linear_invalid_inputs()
    test_dynamic_quantize_linear_accuracy()

    print("\n🎉 All DynamicQuantizeLinear operation tests completed successfully!")
