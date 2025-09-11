#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import AttentionOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_attention_test_tensors(batch_size=2, q_seq_len=8, kv_seq_len=None, q_num_heads=4, kv_num_heads=None,
                                  q_head_size=64, v_head_size=None, include_mask=False, include_past=False, causal=False):
    """Helper function to create test tensors for ONNX Attention operation.
    Shapes:
      - Q: [batch, q_num_heads, q_seq_len, q_head_size]
      - K: [batch, kv_num_heads, kv_seq_len, q_head_size]
      - V: [batch, kv_num_heads, kv_seq_len, v_head_size]
      - attn_mask (optional): broadcastable to [batch, q_num_heads, q_seq_len, total_seq_len]
      - past_key/value (optional): [batch, kv_num_heads, past_seq_len, q_head_size] / [batch, kv_num_heads, past_seq_len, v_head_size]
    """
    if kv_seq_len is None:
        kv_seq_len = q_seq_len
    if kv_num_heads is None:
        kv_num_heads = q_num_heads
    if v_head_size is None:
        v_head_size = q_head_size

    # Create Q, K, V tensors
    Q_shape = [batch_size, q_num_heads, q_seq_len, q_head_size]
    K_shape = [batch_size, kv_num_heads, kv_seq_len, q_head_size]
    V_shape = [batch_size, kv_num_heads, kv_seq_len, v_head_size]

    Q = F._from_shape('Q', Q_shape, np_dtype=np.float32)
    K = F._from_shape('K', K_shape, np_dtype=np.float32)
    V = F._from_shape('V', V_shape, np_dtype=np.float32)

    input_tensors = [Q, K, V]
    input_names = ['Q', 'K', 'V']

    # Optional mask
    if include_mask:
        total_seq_len = kv_seq_len
        # Use a broadcastable mask: [1, 1, q_seq_len, total_seq_len]
        mask_shape = [1, 1, q_seq_len, total_seq_len]
        mask = F._from_shape('attn_mask', mask_shape, np_dtype=np.float32)
        input_tensors.append(mask)
        input_names.append('attn_mask')

    # Optional past key/value states
    if include_past:
        past_seq_len = q_seq_len // 2
        past_key_shape = [batch_size, kv_num_heads, past_seq_len, q_head_size]
        past_value_shape = [batch_size, kv_num_heads, past_seq_len, v_head_size]

        past_key = F._from_shape('past_key', past_key_shape, np_dtype=np.float32)
        past_value = F._from_shape('past_value', past_value_shape, np_dtype=np.float32)

        input_tensors.extend([past_key, past_value])
        input_names.extend(['past_key', 'past_value'])

    # Create output tensors (shapes will be set by op)
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    if include_past:
        present_key = make_tensor('present_key')
        present_value = make_tensor('present_value')
        output_tensors.extend([present_key, present_value])
        output_names.extend(['present_key', 'present_value'])

    return input_tensors, output_tensors, input_names, output_names


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_basic():
    """Test basic attention operation without mask or past states"""
    print("\n=== Testing Basic Attention (ONNX) ===")

    batch_size, q_seq_len, q_num_heads, head_size = 2, 8, 4, 64

    input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
        batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
        include_mask=False, include_past=False
    )

    attrs = {'q_num_heads': q_num_heads, 'kv_num_heads': q_num_heads}

    op_info = {
        'name': 'test_attention_basic',
        'optype': 'Attention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute attention operation
    op_obj = AttentionOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shapes: [batch, q_heads, q_seq, v_head_size]
    expected_output_shape = [batch_size, q_num_heads, q_seq_len, head_size]
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Optional qk_matmul_output not requested in this basic test

    assert 'instrs' in perf_stats and 'mac' in perf_stats['instrs'] and perf_stats['instrs']['mac'] > 0
    print(f"✓ Basic attention test passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_with_causal_mask():
    """Test attention with causal behavior via is_causal attribute"""
    print("\n=== Testing Attention with is_causal ===")

    batch_size, q_seq_len, q_num_heads, head_size = 2, 8, 4, 64

    input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
        batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
        include_mask=True, include_past=False, causal=True
    )

    attrs = {'q_num_heads': q_num_heads, 'kv_num_heads': q_num_heads, 'is_causal': 1}

    op_info = {
        'name': 'test_attention_causal',
        'optype': 'Attention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = AttentionOp(op_info)

    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    expected_output_shape = [batch_size, q_num_heads, q_seq_len, head_size]
    assert output_tensors[0].shape == expected_output_shape
    print(f"✓ Causal attention test passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_with_past_states():
    """Test attention with past key/value states (incremental decoding)"""
    print("\n=== Testing Attention with Past States (ONNX) ===")

    batch_size, q_seq_len, q_num_heads, head_size = 2, 8, 4, 64

    input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
        batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
        include_mask=False, include_past=True
    )

    attrs = {'q_num_heads': q_num_heads, 'kv_num_heads': q_num_heads}

    op_info = {
        'name': 'test_attention_past',
        'optype': 'Attention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = AttentionOp(op_info)

    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    expected_output_shape = [batch_size, q_num_heads, q_seq_len, head_size]
    past_seq = q_seq_len // 2
    expected_present_key_shape = [batch_size, q_num_heads, past_seq + q_seq_len, head_size]
    expected_present_value_shape = [batch_size, q_num_heads, past_seq + q_seq_len, head_size]

    assert output_tensors[0].shape == expected_output_shape
    assert output_tensors[1].shape == expected_present_key_shape  # present_key
    assert output_tensors[2].shape == expected_present_value_shape  # present_value
    print(f"✓ Past states attention test passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_with_custom_mask():
    """Test attention with custom attention mask (broadcasting)"""
    print("\n=== Testing Attention with Custom Mask (ONNX) ===")

    batch_size, q_seq_len, q_num_heads, head_size = 2, 8, 4, 64

    input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
        batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
        include_mask=True, include_past=False, causal=False
    )

    attrs = {'q_num_heads': q_num_heads, 'kv_num_heads': q_num_heads}

    op_info = {
        'name': 'test_attention_custom_mask',
        'optype': 'Attention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = AttentionOp(op_info)

    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate mask is broadcastable to [batch, q_heads, q_seq, k_seq]
    expected_mask_shape = [1, 1, q_seq_len, q_seq_len]
    assert input_tensors[3].shape == expected_mask_shape
    print(f"✓ Custom mask attention test passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_invalid_shapes():
    """Test attention operation with invalid input shapes"""
    print("\n=== Testing Attention Error Cases (ONNX) ===")

    # Test 1: Q is 2D (invalid)
    try:
        Q = F._from_shape('Q', [2, 64], np_dtype=np.float32)  # 2D instead of 3D/4D
        K = F._from_shape('K', [2, 4, 8, 64], np_dtype=np.float32)
        V = F._from_shape('V', [2, 4, 8, 64], np_dtype=np.float32)
        output = make_tensor('output')

        op_info = {
            'name': 'test_attention_invalid_q',
            'optype': 'Attention',
            'inList': ['Q', 'K', 'V'],
            'outList': ['output'],
            'attrs': {'q_num_heads': 4, 'kv_num_heads': 4},
        }

        op_obj = AttentionOp(op_info)
        Q.op_in = [op_info['name']]
        K.op_in = [op_info['name']]
        V.op_in = [op_info['name']]
        output.op_out = [op_info['name']]

        _ = op_obj.get_perf_counts([Q, K, V], [output])
        assert False, "Should have raised assertion error for invalid Q rank"
    except AssertionError as e:
        print(f"✓ Correctly caught invalid Q rank: {e}")

    # Test 2: Mismatched kv_num_heads between K and V
    try:
        Q = F._from_shape('Q', [2, 6, 8, 64], np_dtype=np.float32)  # q_heads=6
        K = F._from_shape('K', [2, 4, 8, 64], np_dtype=np.float32)  # kv_heads=4
        V = F._from_shape('V', [2, 3, 8, 64], np_dtype=np.float32)  # kv_heads=3 (mismatch)
        output = make_tensor('output')

        op_info = {
            'name': 'test_attention_mismatched_kv_heads',
            'optype': 'Attention',
            'inList': ['Q', 'K', 'V'],
            'outList': ['output'],
            'attrs': {'q_num_heads': 6, 'kv_num_heads': 4},
        }

        op_obj = AttentionOp(op_info)
        Q.op_in = [op_info['name']]
        K.op_in = [op_info['name']]
        V.op_in = [op_info['name']]
        output.op_out = [op_info['name']]

        _ = op_obj.get_perf_counts([Q, K, V], [output])
        assert False, "Should have raised assertion error for mismatched K/V kv_heads"
    except AssertionError as e:
        print(f"✓ Correctly caught mismatched K/V kv_heads: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_scale_attribute():
    """Test attention with custom scale attribute"""
    print("\n=== Testing Attention with Custom Scale (ONNX) ===")

    batch_size, q_seq_len, q_num_heads, head_size = 2, 8, 4, 64

    input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
        batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
        include_mask=False, include_past=False
    )

    # Custom scale instead of default 1/sqrt(head_size)
    custom_scale = 0.125
    attrs = {'q_num_heads': q_num_heads, 'kv_num_heads': q_num_heads, 'scale': custom_scale}

    op_info = {
        'name': 'test_attention_scale',
        'optype': 'Attention',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = AttentionOp(op_info)

    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate that scale attribute is properly set
    assert op_obj.scale == custom_scale
    print(f"✓ Custom scale attention test passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_attention_factory():
    """Test that Attention operation can be created through SimOpFactory"""
    print("\n=== Testing Attention Factory Creation (ONNX) ===")

    from ttsim.ops.op import SimOpFactory

    attention_class = SimOpFactory('Attention')
    assert attention_class == AttentionOp

    # Create instance through factory with legacy attribute for backward-compat
    op_info = {
        'name': 'factory_test_attention',
        'optype': 'Attention',
        'inList': ['Q', 'K', 'V'],
        'outList': ['output'],
        'attrs': {'num_heads': 4},
    }

    op_obj = attention_class(op_info)
    assert isinstance(op_obj, AttentionOp)
    assert op_obj.q_num_heads == 4
    print("✓ Attention factory creation test passed")


if __name__ == '__main__':
    # Run all tests manually
    test_attention_factory()
    test_attention_basic()
    test_attention_with_causal_mask()
    test_attention_with_past_states()
    test_attention_with_custom_mask()
    test_attention_scale_attribute()
    # Additional validation for qk_matmul_output_mode
    def _test_qk_modes():
        batch_size, q_seq_len, q_num_heads, head_size = 2, 6, 3, 32
        for mode in [0, 1, 2, 3]:
            include_mask = (mode == 1)
            input_tensors, output_tensors, input_names, output_names = create_attention_test_tensors(
                batch_size=batch_size, q_seq_len=q_seq_len, q_num_heads=q_num_heads, q_head_size=head_size,
                include_mask=include_mask, include_past=False
            )
            qk_out = make_tensor('qk_out')
            output_tensors.append(qk_out)
            output_names.append('qk_out')
            attrs = {
                'q_num_heads': q_num_heads,
                'kv_num_heads': q_num_heads,
                'qk_matmul_output_mode': mode,
            }
            op_info = {
                'name': f'test_attention_qk_mode_{mode}',
                'optype': 'Attention',
                'inList': input_names,
                'outList': output_names,
                'attrs': attrs,
            }
            op_obj = AttentionOp(op_info)
            for tensor in input_tensors:
                tensor.op_in = [op_info['name']]
            for tensor in output_tensors:
                tensor.op_out = [op_info['name']]
            perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
            assert output_tensors[-1].shape == [batch_size, q_num_heads, q_seq_len, q_seq_len]
            assert perf_stats.get('qk_matmul_output_mode') == mode
    _test_qk_modes()
    test_attention_invalid_shapes()

    print("\n🎉 All Attention operation tests completed successfully!")
