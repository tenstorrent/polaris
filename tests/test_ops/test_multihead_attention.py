#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import MultiHeadAttentionOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_multihead_attention_test_tensors(batch_size=2, seq_len=8, num_heads=4, hidden_size=256,
                                           include_bias=False, include_key_padding_mask=False,
                                           include_past=False, unpacked_input=False):
    """Helper function to create test tensors for MultiHeadAttention operation"""

    # Create query tensor: [batch_size, seq_len, hidden_size]
    query_shape = [batch_size, seq_len, hidden_size]
    query = F._from_shape('query', query_shape, np_dtype=np.float32)

    input_tensors = [query]
    input_names = ['query']

    # Create key and value tensors
    if unpacked_input:
        # Format 2: unpacked - [batch_size, num_heads, seq_len, head_size]
        head_size = hidden_size // num_heads
        key_shape = [batch_size, num_heads, seq_len, head_size]
        value_shape = [batch_size, num_heads, seq_len, head_size]
    else:
        # Format 1: packed - [batch_size, seq_len, hidden_size]
        key_shape = [batch_size, seq_len, hidden_size]
        value_shape = [batch_size, seq_len, hidden_size]

    key = F._from_shape('key', key_shape, np_dtype=np.float32)
    value = F._from_shape('value', value_shape, np_dtype=np.float32)

    input_tensors.extend([key, value])
    input_names.extend(['key', 'value'])

    # Optional bias: [batch_size, seq_len, seq_len] or [1, seq_len, seq_len]
    if include_bias:
        bias_shape = [batch_size, seq_len, seq_len]
        bias = F._from_shape('bias', bias_shape, np_dtype=np.float32)
        input_tensors.append(bias)
        input_names.append('bias')

    # Optional key_padding_mask: [batch_size, seq_len]
    if include_key_padding_mask:
        mask_shape = [batch_size, seq_len]
        key_padding_mask = F._from_shape('key_padding_mask', mask_shape, np_dtype=np.bool_)
        input_tensors.append(key_padding_mask)
        input_names.append('key_padding_mask')

    # Optional past key/value states: [batch_size, past_seq_len, num_heads, head_size]
    if include_past:
        past_seq_len = seq_len // 2
        head_size = hidden_size // num_heads
        past_key_shape = [batch_size, past_seq_len, num_heads, head_size]
        past_value_shape = [batch_size, past_seq_len, num_heads, head_size]

        past_key = F._from_shape('past_key', past_key_shape, np_dtype=np.float32)
        past_value = F._from_shape('past_value', past_value_shape, np_dtype=np.float32)

        input_tensors.extend([past_key, past_value])
        input_names.extend(['past_key', 'past_value'])

    # Create output tensor: [batch_size, seq_len, hidden_size]
    output_shape = [batch_size, seq_len, hidden_size]
    output = make_tensor('output')

    output_tensors = [output]
    output_names = ['output']

    # Optional present key/value outputs
    if include_past:
        present_key = make_tensor('present_key')
        present_value = make_tensor('present_value')
        output_tensors.extend([present_key, present_value])
        output_names.extend(['present_key', 'present_value'])

    return input_tensors, output_tensors, input_names, output_names


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_basic_packed():
    """Test basic MultiHeadAttention with packed input format"""
    print("\n=== Testing Basic MultiHeadAttention (Packed) ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=False, include_key_padding_mask=False, include_past=False, unpacked_input=False
    )

    attrs = {'num_heads': num_heads}

    op_info = {
        'name': 'test_mha_basic_packed',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute MultiHeadAttention operation
    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        if tensor is not None:  # Skip None placeholders
            tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shapes
    expected_output_shape = [batch_size, seq_len, hidden_size]
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Validate performance stats
    assert 'instrs' in perf_stats
    assert 'mac' in perf_stats['instrs']
    assert perf_stats['instrs']['mac'] > 0

    print(f"✓ Basic packed MultiHeadAttention test passed")
    print(f"  Query shape: {input_tensors[0].shape}")
    print(f"  Key shape: {input_tensors[1].shape}")
    print(f"  Value shape: {input_tensors[2].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_basic_unpacked():
    """Test basic MultiHeadAttention with unpacked input format"""
    print("\n=== Testing Basic MultiHeadAttention (Unpacked) ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=False, include_key_padding_mask=False, include_past=False, unpacked_input=True
    )

    attrs = {'num_heads': num_heads}

    op_info = {
        'name': 'test_mha_basic_unpacked',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shapes
    expected_output_shape = [batch_size, seq_len, hidden_size]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ Basic unpacked MultiHeadAttention test passed")
    print(f"  Query shape: {input_tensors[0].shape}")
    print(f"  Key shape: {input_tensors[1].shape}")
    print(f"  Value shape: {input_tensors[2].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_with_bias():
    """Test MultiHeadAttention with bias tensor"""
    print("\n=== Testing MultiHeadAttention with Bias ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=True, include_key_padding_mask=False, include_past=False, unpacked_input=False
    )

    attrs = {'num_heads': num_heads}

    op_info = {
        'name': 'test_mha_with_bias',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        if tensor is not None:  # Skip None placeholders
            tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate bias shape: [batch_size, seq_len, seq_len]
    expected_bias_shape = [batch_size, seq_len, seq_len]
    bias_tensor = input_tensors[3]  # bias is at index 3
    assert bias_tensor.shape == expected_bias_shape

    print(f"✓ Bias MultiHeadAttention test passed")
    print(f"  Bias shape: {bias_tensor.shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_with_key_padding_mask():
    """Test MultiHeadAttention with key padding mask"""
    print("\n=== Testing MultiHeadAttention with Key Padding Mask ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=False, include_key_padding_mask=True, include_past=False, unpacked_input=False
    )

    attrs = {'num_heads': num_heads}

    op_info = {
        'name': 'test_mha_with_mask',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        if tensor is not None:  # Skip None placeholders
            tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Find the key_padding_mask tensor (should be at index 4 due to bias placeholder)
    mask_tensor = None
    for tensor in input_tensors:
        if tensor is not None and hasattr(tensor, 'name') and 'key_padding_mask' in tensor.name:
            mask_tensor = tensor
            break

    assert mask_tensor is not None, "key_padding_mask tensor not found"
    expected_mask_shape = [batch_size, seq_len]
    assert mask_tensor.shape == expected_mask_shape

    print(f"✓ Key padding mask MultiHeadAttention test passed")
    print(f"  Mask shape: {mask_tensor.shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_with_past_states():
    """Test MultiHeadAttention with past key/value states (incremental decoding)"""
    print("\n=== Testing MultiHeadAttention with Past States ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=False, include_key_padding_mask=False, include_past=True, unpacked_input=False
    )

    attrs = {'num_heads': num_heads}

    op_info = {
        'name': 'test_mha_with_past',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        if tensor is not None:  # Skip None placeholders
            tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shapes
    expected_output_shape = [batch_size, seq_len, hidden_size]
    expected_present_key_shape = [batch_size, seq_len + seq_len//2, num_heads, hidden_size//num_heads]  # past + current
    expected_present_value_shape = [batch_size, seq_len + seq_len//2, num_heads, hidden_size//num_heads]

    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[1].shape == expected_present_key_shape  # present_key
    assert output_tensors[2].shape == expected_present_value_shape  # present_value

    print(f"✓ Past states MultiHeadAttention test passed")
    print(f"  Past key shape: {input_tensors[3].shape}")
    print(f"  Present key shape: {output_tensors[1].shape}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_invalid_inputs():
    """Test MultiHeadAttention operation with invalid inputs"""
    print("\n=== Testing MultiHeadAttention Error Cases ===")

    # Test 1: Missing num_heads attribute
    try:
        op_info = {
            'name': 'test_mha_missing_heads',
            'optype': 'MultiHeadAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {},  # Missing num_heads
        }
        op_obj = MultiHeadAttentionOp(op_info)
        assert False, "Should have raised ValueError for missing num_heads"
    except ValueError as e:
        print(f"✓ Correctly caught missing num_heads: {e}")

    # Test 2: Invalid query shape (not 3D)
    try:
        query = F._from_shape('query', [2, 8, 4, 64], np_dtype=np.float32)  # 4D instead of 3D
        key = F._from_shape('key', [2, 8, 256], np_dtype=np.float32)
        value = F._from_shape('value', [2, 8, 256], np_dtype=np.float32)
        output = make_tensor('output')

        op_info = {
            'name': 'test_mha_invalid_query',
            'optype': 'MultiHeadAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {'num_heads': 4},
        }

        op_obj = MultiHeadAttentionOp(op_info)
        query.op_in = [op_info['name']]
        key.op_in = [op_info['name']]
        value.op_in = [op_info['name']]
        output.op_out = [op_info['name']]

        perf_stats = op_obj.get_perf_counts([query, key, value], [output])
        assert False, "Should have raised assertion error for invalid query shape"
    except AssertionError as e:
        print(f"✓ Correctly caught invalid query shape: {e}")

    # Test 3: Mismatched hidden sizes
    try:
        query = F._from_shape('query', [2, 8, 256], np_dtype=np.float32)
        key = F._from_shape('key', [2, 8, 128], np_dtype=np.float32)  # Different hidden size
        value = F._from_shape('value', [2, 8, 256], np_dtype=np.float32)
        output = make_tensor('output')

        op_info = {
            'name': 'test_mha_mismatched_hidden',
            'optype': 'MultiHeadAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {'num_heads': 4},
        }

        op_obj = MultiHeadAttentionOp(op_info)
        query.op_in = [op_info['name']]
        key.op_in = [op_info['name']]
        value.op_in = [op_info['name']]
        output.op_out = [op_info['name']]

        perf_stats = op_obj.get_perf_counts([query, key, value], [output])
        assert False, "Should have raised assertion error for mismatched hidden sizes"
    except AssertionError as e:
        print(f"✓ Correctly caught mismatched hidden sizes: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_unidirectional():
    """Test MultiHeadAttention with unidirectional attribute"""
    print("\n=== Testing MultiHeadAttention with Unidirectional ===")

    batch_size, seq_len, num_heads, hidden_size = 2, 8, 4, 256

    input_tensors, output_tensors, input_names, output_names = create_multihead_attention_test_tensors(
        batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, hidden_size=hidden_size,
        include_bias=False, include_key_padding_mask=False, include_past=False, unpacked_input=False
    )

    attrs = {'num_heads': num_heads, 'unidirectional': True}

    op_info = {
        'name': 'test_mha_unidirectional',
        'optype': 'MultiHeadAttention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = MultiHeadAttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        if tensor is not None:  # Skip None placeholders
            tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate that unidirectional attribute is properly set
    assert op_obj.unidirectional == True

    print(f"✓ Unidirectional MultiHeadAttention test passed")
    print(f"  Unidirectional: {op_obj.unidirectional}")
    print(f"  Performance: {perf_stats['instrs']['mac']} MAC operations")


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_factory():
    """Test that MultiHeadAttention operation can be created through SimOpFactory"""
    print("\n=== Testing MultiHeadAttention Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    mha_class = SimOpFactory('MultiHeadAttention')
    assert mha_class == MultiHeadAttentionOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_mha',
        'optype': 'MultiHeadAttention',
        'inList': ['query', 'key', 'value'],
        'outList': ['output'],
        'attrs': {'num_heads': 8},
    }

    op_obj = mha_class(op_info)
    assert isinstance(op_obj, MultiHeadAttentionOp)
    assert op_obj.optype == 'MultiHeadAttention'
    assert op_obj.num_heads == 8

    print("✓ MultiHeadAttention factory creation test passed")
    print(f"  Created operation: {op_obj}")


if __name__ == '__main__':
    # Run all tests manually
    test_multihead_attention_factory()
    test_multihead_attention_basic_packed()
    test_multihead_attention_basic_unpacked()
    test_multihead_attention_with_bias()
    test_multihead_attention_with_key_padding_mask()
    test_multihead_attention_with_past_states()
    test_multihead_attention_unidirectional()
    test_multihead_attention_invalid_inputs()

    print("\n🎉 All MultiHeadAttention operation tests completed successfully!")
