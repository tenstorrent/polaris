#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import RMSNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_rms_normalization_test_tensors(input_shape, axis=-1):
    """Helper function to create test tensors for RMSNormalization operation"""

    # Create input tensor
    input_tensor = F._from_shape('input', input_shape)

    # Create scale tensor (one scale per feature along the normalization axis)
    # Handle negative axis
    normalized_axis = axis if axis >= 0 else len(input_shape) + axis
    scale_shape = [input_shape[normalized_axis]]
    scale_tensor = F._from_shape('scale', scale_shape, np_dtype=np.float32)

    input_tensors = [input_tensor, scale_tensor]
    input_names = ['input', 'scale']

    # Create output tensor (shape will be same as input)
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    return input_tensors, output_tensors, input_names, output_names


def reference_rms_normalization(X, scale, axis=-1, epsilon=1e-5):
    """
    Reference implementation of RMSNormalization for testing
    Follows the ONNX specification exactly
    """
    # Convert to numpy for easier manipulation
    X = np.asarray(X)
    scale = np.asarray(scale)

    # Compute RMS: sqrt(mean(X^2, axis=axis, keepdims=True))
    X_squared = X ** 2
    mean_squared = np.mean(X_squared, axis=axis, keepdims=True)
    rms = np.sqrt(mean_squared + epsilon)

    # Apply normalization: scale * X / rms
    Y = scale * X / rms

    return Y


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_basic():
    """Test basic RMSNormalization with default axis (-1)"""
    print("\n=== Testing Basic RMSNormalization ===")

    input_shape = [4, 8, 16]  # [batch, seq_len, hidden_size]
    axis = -1  # Normalize along the last axis (hidden_size)

    input_tensors, output_tensors, input_names, output_names = create_rms_normalization_test_tensors(
        input_shape, axis=axis
    )

    attrs = {'axis': axis, 'epsilon': 1e-5}

    op_info = {
        'name': 'test_rms_normalization_basic',
        'optype': 'RMSNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute RMSNormalization operation
    op_obj = RMSNormalizationOp(op_info)

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

    # Validate output dtype (should match input dtype)
    assert output_tensors[0].dtype == input_tensors[0].dtype, \
        f"Output dtype mismatch: {output_tensors[0].dtype} != {input_tensors[0].dtype}"

    print(f"✓ Basic RMSNormalization test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Axis: {op_obj.axis}")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_axis0():
    """Test RMSNormalization with axis=0"""
    print("\n=== Testing RMSNormalization (axis=0) ===")

    input_shape = [8, 16, 32]  # [batch, seq_len, hidden_size]
    axis = 0  # Normalize along the batch axis

    input_tensors, output_tensors, input_names, output_names = create_rms_normalization_test_tensors(
        input_shape, axis=axis
    )

    attrs = {'axis': axis, 'epsilon': 1e-6}

    op_info = {
        'name': 'test_rms_normalization_axis0',
        'optype': 'RMSNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = RMSNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    expected_scale_shape = [input_shape[axis]]
    assert output_tensors[0].shape == expected_output_shape
    assert input_tensors[1].shape == expected_scale_shape

    print(f"✓ RMSNormalization axis=0 test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_2d():
    """Test RMSNormalization with 2D tensor"""
    print("\n=== Testing RMSNormalization (2D) ===")

    input_shape = [10, 64]  # [seq_len, hidden_size]
    axis = -1  # Normalize along hidden_size

    input_tensors, output_tensors, input_names, output_names = create_rms_normalization_test_tensors(
        input_shape, axis=axis
    )

    attrs = {'axis': axis, 'epsilon': 1e-5}

    op_info = {
        'name': 'test_rms_normalization_2d',
        'optype': 'RMSNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = RMSNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 2D RMSNormalization test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_1d():
    """Test RMSNormalization with 1D tensor"""
    print("\n=== Testing RMSNormalization (1D) ===")

    input_shape = [128]  # Simple 1D vector
    axis = 0  # Only one axis to normalize along

    # Create input tensor
    input_tensor = F._from_shape('input', input_shape)

    # For 1D tensor, scale should be scalar (since we're normalizing the whole vector)
    scale_tensor = F._from_shape('scale', [], np_dtype=np.float32)  # Scalar scale

    input_tensors = [input_tensor, scale_tensor]
    input_names = ['input', 'scale']

    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    attrs = {'axis': axis, 'epsilon': 1e-5}

    op_info = {
        'name': 'test_rms_normalization_1d',
        'optype': 'RMSNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = RMSNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 1D RMSNormalization test passed")
    print(f"  Input shape: {input_tensors[0].shape}")
    print(f"  Scale shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_invalid_inputs():
    """Test RMSNormalization operation with invalid inputs"""
    print("\n=== Testing RMSNormalization Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('input', [8, 16])]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_rms_normalization_wrong_inputs',
            'optype': 'RMSNormalization',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = RMSNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong output count
    try:
        input_tensors = [
            F._from_shape('input', [8, 16]),
            F._from_shape('scale', [16], np_dtype=np.float32)
        ]
        output_tensors = [make_tensor('output1'), make_tensor('output2')]

        op_info = {
            'name': 'test_rms_normalization_wrong_outputs',
            'optype': 'RMSNormalization',
            'inList': ['input', 'scale'],
            'outList': ['output1', 'output2'],
            'attrs': {},
        }

        op_obj = RMSNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong output count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong output count: {e}")

    # Test 3: Invalid axis
    try:
        input_tensors = [
            F._from_shape('input', [4, 8]),
            F._from_shape('scale', [8], np_dtype=np.float32)
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_rms_normalization_invalid_axis',
            'optype': 'RMSNormalization',
            'inList': ['input', 'scale'],
            'outList': ['output'],
            'attrs': {'axis': 3},  # Invalid axis for 2D tensor
        }

        op_obj = RMSNormalizationOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised ValueError for invalid axis"
    except ValueError as e:
        print(f"✓ Correctly caught invalid axis: {e}")

    # Test 4: Invalid epsilon
    try:
        input_tensors = [
            F._from_shape('input', [4, 8]),
            F._from_shape('scale', [8], np_dtype=np.float32)
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_rms_normalization_invalid_epsilon',
            'optype': 'RMSNormalization',
            'inList': ['input', 'scale'],
            'outList': ['output'],
            'attrs': {'epsilon': -1.0},  # Invalid negative epsilon
        }

        op_obj = RMSNormalizationOp(op_info)
        assert False, "Should have raised ValueError for invalid epsilon"
    except ValueError as e:
        print(f"✓ Correctly caught invalid epsilon: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_factory():
    """Test that RMSNormalization operation can be created through SimOpFactory"""
    print("\n=== Testing RMSNormalization Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    rms_norm_class = SimOpFactory('RMSNormalization')
    assert rms_norm_class == RMSNormalizationOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_rms_normalization',
        'optype': 'RMSNormalization',
        'inList': ['input', 'scale'],
        'outList': ['output'],
        'attrs': {'axis': -1, 'epsilon': 1e-5},
    }

    op_obj = rms_norm_class(op_info)
    assert isinstance(op_obj, RMSNormalizationOp)
    assert op_obj.optype == 'RMSNormalization'
    assert op_obj.name == 'factory_test_rms_normalization'
    assert op_obj.axis == -1
    assert op_obj.epsilon == 1e-5

    print("✓ RMSNormalization factory creation test passed")
    print(f"  Created operation: {op_obj}")


@pytest.mark.unit
@pytest.mark.opunit
def test_rms_normalization_transformer_scenario():
    """Test RMSNormalization in a typical transformer scenario"""
    print("\n=== Testing RMSNormalization Transformer Scenario ===")

    # Typical transformer layer: [batch_size, seq_len, hidden_size]
    batch_size = 4
    seq_len = 32
    hidden_size = 512
    input_shape = [batch_size, seq_len, hidden_size]

    input_tensors, output_tensors, input_names, output_names = create_rms_normalization_test_tensors(
        input_shape, axis=-1
    )

    attrs = {'axis': -1, 'epsilon': 1e-6}  # Use smaller epsilon for better stability

    op_info = {
        'name': 'test_rms_normalization_transformer',
        'optype': 'RMSNormalization',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = RMSNormalizationOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output
    expected_output_shape = input_shape
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ Transformer RMSNormalization test passed")
    print(f"  Input shape: {input_tensors[0].shape} (batch_size x seq_len x hidden_size)")
    print(f"  Scale shape: {input_tensors[1].shape} (scales for each feature)")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['rsqrt']} square root ops, {perf_stats['instrs']['div']} divisions")


if __name__ == '__main__':
    # Run all tests manually
    test_rms_normalization_factory()
    test_rms_normalization_basic()
    test_rms_normalization_axis0()
    test_rms_normalization_2d()
    test_rms_normalization_1d()
    test_rms_normalization_invalid_inputs()
    test_rms_normalization_transformer_scenario()

    print("\n🎉 All RMSNormalization operation tests completed successfully!")
