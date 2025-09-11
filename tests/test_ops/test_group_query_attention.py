#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for GroupQueryAttention operation implementation.

This module tests the ONNX 1.20.0 GroupQueryAttention operation, which supports
grouped attention mechanisms where multiple query heads share the same key-value
heads. This is crucial for efficient inference in large language models.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, GroupQueryAttentionOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_group_query_attention_test_tensors(batch_size=2, seq_len_q=8, seq_len_k=6, num_heads=8,
                                              kv_num_heads=4, head_dim=64, use_past=False,
                                              use_rope=False, causal=False):
    """
    Helper function to create test tensors for GroupQueryAttention operation.

    Args:
        batch_size: Batch size for all tensors
        seq_len_q: Query sequence length
        seq_len_k: Key/Value sequence length
        num_heads: Number of query heads
        kv_num_heads: Number of key-value heads
        head_dim: Head dimension
        use_past: Whether to include past key/value tensors
        use_rope: Whether to include rotary position embeddings
        causal: Whether to use causal masking

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Calculate hidden sizes
    hidden_size_q = num_heads * head_dim
    hidden_size_kv = kv_num_heads * head_dim

    # Create required inputs: query, key, value
    query = F._from_shape('query', [batch_size, seq_len_q, hidden_size_q], np_dtype=np.dtype('float32'))
    key = F._from_shape('key', [batch_size, seq_len_k, hidden_size_kv], np_dtype=np.dtype('float32'))
    value = F._from_shape('value', [batch_size, seq_len_k, hidden_size_kv], np_dtype=np.dtype('float32'))

    input_tensors = [query, key, value]
    input_names = ['query', 'key', 'value']

    # Add past key/value if requested
    past_key = None
    past_value = None
    if use_past:
        past_seq_len = 4  # Some past context
        # Past key/value should have kv_num_heads, not num_heads
        past_key = F._from_shape('past_key', [batch_size, past_seq_len, kv_num_heads, head_dim], np_dtype=np.dtype('float32'))
        past_value = F._from_shape('past_value', [batch_size, past_seq_len, kv_num_heads, head_dim], np_dtype=np.dtype('float32'))
        input_tensors.extend([past_key, past_value])
        input_names.extend(['past_key', 'past_value'])

    # Add rotary embeddings if requested
    cos_cache = None
    sin_cache = None
    if use_rope:
        rope_dim = head_dim // 2  # RoPE typically uses half the head dimension
        cos_cache = F._from_shape('cos_cache', [batch_size, seq_len_q, rope_dim], np_dtype=np.dtype('float32'))
        sin_cache = F._from_shape('sin_cache', [batch_size, seq_len_q, rope_dim], np_dtype=np.dtype('float32'))
        input_tensors.extend([cos_cache, sin_cache])
        input_names.extend(['cos_cache', 'sin_cache'])

    # Create output tensors
    output_hidden_size = num_heads * head_dim
    output = F._from_shape('output', [batch_size, seq_len_q, output_hidden_size], np_dtype=np.dtype('float32'))

    # Optional outputs: present_key, present_value
    present_key = None
    present_value = None
    total_seq_len = seq_len_k
    if use_past and past_key is not None:
        total_seq_len += past_key.shape[1]

    present_key = F._from_shape('present_key', [batch_size, kv_num_heads, total_seq_len, head_dim], np_dtype=np.dtype('float32'))
    present_value = F._from_shape('present_value', [batch_size, kv_num_heads, total_seq_len, head_dim], np_dtype=np.dtype('float32'))

    output_tensors = [output, present_key, present_value]

    return input_tensors, output_tensors, input_names


class TestGroupQueryAttention:
    """Test class for GroupQueryAttention operation"""

    def test_factory_integration(self):
        """Test that GroupQueryAttention is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('GroupQueryAttention')
        assert opcls == GroupQueryAttentionOp

    def test_basic_grouped_attention(self):
        """Test basic grouped attention with 8 query heads and 4 KV heads"""
        batch_size, seq_len_q, seq_len_k = 2, 8, 6
        num_heads, kv_num_heads, head_dim = 8, 4, 64

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim
        )

        op_info = {
            'name': 'test_group_query_attention_basic',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_output_shape = [batch_size, seq_len_q, num_heads * head_dim]
        expected_present_shape = [batch_size, kv_num_heads, seq_len_k, head_dim]

        assert outT[0].shape == expected_output_shape
        assert outT[1].shape == expected_present_shape
        assert outT[2].shape == expected_present_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have attention computations

        # Group size should be 2 (8 query heads / 4 KV heads)
        assert op.group_size == 2

    def test_grouped_attention_with_past(self):
        """Test grouped attention with KV-cache (past states)"""
        batch_size, seq_len_q, seq_len_k = 2, 4, 6  # New sequence shorter than past
        num_heads, kv_num_heads, head_dim = 6, 3, 64

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim,
            use_past=True
        )

        op_info = {
            'name': 'test_group_query_attention_past',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_output_shape = [batch_size, seq_len_q, num_heads * head_dim]
        expected_present_shape = [batch_size, kv_num_heads, 10, head_dim]  # 4 (past) + 6 (current)

        assert outT[0].shape == expected_output_shape
        assert outT[1].shape == expected_present_shape
        assert outT[2].shape == expected_present_shape

    def test_grouped_attention_with_rope(self):
        """Test grouped attention with rotary position embeddings"""
        batch_size, seq_len_q, seq_len_k = 2, 8, 6
        num_heads, kv_num_heads, head_dim = 12, 6, 64

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim,
            use_rope=True
        )

        op_info = {
            'name': 'test_group_query_attention_rope',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_output_shape = [batch_size, seq_len_q, num_heads * head_dim]
        expected_present_shape = [batch_size, kv_num_heads, seq_len_k, head_dim]

        assert outT[0].shape == expected_output_shape
        assert outT[1].shape == expected_present_shape
        assert outT[2].shape == expected_present_shape

        # RoPE should add multiplication operations
        assert perf_stats['instrs']['mul'] > 0

    def test_causal_grouped_attention(self):
        """Test grouped attention with causal masking"""
        batch_size, seq_len_q, seq_len_k = 2, 8, 8  # Square attention matrix
        num_heads, kv_num_heads, head_dim = 8, 4, 64

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim
        )

        op_info = {
            'name': 'test_group_query_attention_causal',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
                'causal': True,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Causal masking should add comparison operations
        assert perf_stats['instrs']['cmp'] > 0

    def test_custom_scale_factor(self):
        """Test grouped attention with custom scale factor"""
        batch_size, seq_len_q, seq_len_k = 2, 6, 4
        num_heads, kv_num_heads, head_dim = 8, 4, 64
        scale = 0.125  # Custom scale factor

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim
        )

        op_info = {
            'name': 'test_group_query_attention_scale',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
                'scale': scale,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Custom scale should be stored correctly
        assert op.scale == scale

    def test_different_group_sizes(self):
        """Test different group sizes (query heads per KV head)"""
        test_cases = [
            (4, 4, 1),   # 1 query head per KV head
            (8, 4, 2),   # 2 query heads per KV head
            (12, 4, 3),  # 3 query heads per KV head
            (16, 4, 4),  # 4 query heads per KV head
        ]

        batch_size, seq_len_q, seq_len_k, head_dim = 2, 6, 4, 64

        for num_heads, kv_num_heads, expected_group_size in test_cases:
            inT, outT, input_names = create_group_query_attention_test_tensors(
                batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
                num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim
            )

            op_info = {
                'name': f'test_group_size_{num_heads}_{kv_num_heads}',
                'optype': 'GroupQueryAttention',
                'inList': input_names,
                'outList': ['output', 'present_key', 'present_value'],
                'attrs': {
                    'num_heads': num_heads,
                    'kv_num_heads': kv_num_heads,
                },
            }
            op = GroupQueryAttentionOp(op_info)

            # Verify group size calculation
            assert op.group_size == expected_group_size

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shapes
            expected_output_shape = [batch_size, seq_len_q, num_heads * head_dim]
            expected_present_shape = [batch_size, kv_num_heads, seq_len_k, head_dim]

            assert outT[0].shape == expected_output_shape
            assert outT[1].shape == expected_present_shape
            assert outT[2].shape == expected_present_shape

    def test_minimal_inputs(self):
        """Test with minimal required inputs (no optional inputs)"""
        batch_size, seq_len_q, seq_len_k = 1, 4, 4
        num_heads, kv_num_heads, head_dim = 4, 2, 32

        # Create minimal inputs
        query = F._from_shape('query', [batch_size, seq_len_q, num_heads * head_dim], np_dtype=np.dtype('float32'))
        key = F._from_shape('key', [batch_size, seq_len_k, kv_num_heads * head_dim], np_dtype=np.dtype('float32'))
        value = F._from_shape('value', [batch_size, seq_len_k, kv_num_heads * head_dim], np_dtype=np.dtype('float32'))

        inT = [query, key, value]

        # Create outputs
        output = F._from_shape('output', [batch_size, seq_len_q, num_heads * head_dim], np_dtype=np.dtype('float32'))
        present_key = F._from_shape('present_key', [batch_size, seq_len_k, kv_num_heads, head_dim], np_dtype=np.dtype('float32'))
        present_value = F._from_shape('present_value', [batch_size, seq_len_k, kv_num_heads, head_dim], np_dtype=np.dtype('float32'))

        outT = [output, present_key, present_value]

        op_info = {
            'name': 'test_group_query_attention_minimal',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify basic functionality
        assert perf_stats is not None
        assert perf_stats['instrs']['mac'] > 0

    def test_missing_num_heads(self):
        """Test error handling for missing num_heads attribute"""
        op_info = {
            'name': 'test_missing_num_heads',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'kv_num_heads': 4,
            },
        }

        with pytest.raises(ValueError, match="GroupQueryAttention requires 'num_heads' attribute"):
            GroupQueryAttentionOp(op_info)

    def test_missing_kv_num_heads(self):
        """Test error handling for missing kv_num_heads attribute"""
        op_info = {
            'name': 'test_missing_kv_num_heads',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
            },
        }

        with pytest.raises(ValueError, match="GroupQueryAttention requires 'kv_num_heads' attribute"):
            GroupQueryAttentionOp(op_info)

    def test_invalid_group_size(self):
        """Test error handling for invalid group size (num_heads not divisible by kv_num_heads)"""
        op_info = {
            'name': 'test_invalid_group_size',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 7,      # Not divisible by 3
                'kv_num_heads': 3,
            },
        }

        with pytest.raises(ValueError, match="num_heads .* must be divisible by kv_num_heads"):
            GroupQueryAttentionOp(op_info)

    def test_inconsistent_batch_sizes(self):
        """Test error handling for inconsistent batch sizes"""
        # Create tensors with different batch sizes
        query = F._from_shape('query', [2, 8, 512], np_dtype=np.dtype('float32'))
        key = F._from_shape('key', [3, 6, 256], np_dtype=np.dtype('float32'))  # Different batch size
        value = F._from_shape('value', [3, 6, 256], np_dtype=np.dtype('float32'))

        inT = [query, key, value]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_inconsistent_batch',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'kv_num_heads': 4,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        with pytest.raises(ValueError, match="Query, key, and value must have the same batch size"):
            op.get_perf_counts(inT, outT)

    def test_inconsistent_sequence_lengths(self):
        """Test error handling for inconsistent key/value sequence lengths"""
        # Create tensors with different sequence lengths for key/value
        query = F._from_shape('query', [2, 8, 512], np_dtype=np.dtype('float32'))
        key = F._from_shape('key', [2, 6, 256], np_dtype=np.dtype('float32'))
        value = F._from_shape('value', [2, 4, 256], np_dtype=np.dtype('float32'))  # Different seq length

        inT = [query, key, value]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_inconsistent_seq_len',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'kv_num_heads': 4,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        with pytest.raises(ValueError, match="Key and value must have the same sequence length"):
            op.get_perf_counts(inT, outT)

    def test_invalid_hidden_sizes(self):
        """Test error handling for hidden sizes not divisible by number of heads"""
        # Query hidden size not divisible by num_heads
        query = F._from_shape('query', [2, 8, 510], np_dtype=np.dtype('float32'))  # 510 % 8 != 0
        key = F._from_shape('key', [2, 6, 256], np_dtype=np.dtype('float32'))
        value = F._from_shape('value', [2, 6, 256], np_dtype=np.dtype('float32'))

        inT = [query, key, value]
        outT = [F._from_shape('output', [2, 8, 510], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_hidden_size_q',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'kv_num_heads': 4,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        with pytest.raises(ValueError, match="Query hidden size .* must be divisible by num_heads"):
            op.get_perf_counts(inT, outT)

    def test_mismatched_head_dimensions(self):
        """Test error handling for mismatched head dimensions"""
        # Create tensors with different head dimensions
        query = F._from_shape('query', [2, 8, 512], np_dtype=np.dtype('float32'))  # head_dim = 512/8 = 64
        key = F._from_shape('key', [2, 6, 252], np_dtype=np.dtype('float32'))     # head_dim = 252/4 = 63 (mismatch)
        value = F._from_shape('value', [2, 6, 252], np_dtype=np.dtype('float32'))

        inT = [query, key, value]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_mismatched_head_dims',
            'optype': 'GroupQueryAttention',
            'inList': ['query', 'key', 'value'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'kv_num_heads': 4,
            },
        }
        op = GroupQueryAttentionOp(op_info)

        with pytest.raises(ValueError, match="Head dimensions must match"):
            op.get_perf_counts(inT, outT)

    def test_large_scale_model_config(self):
        """Test configuration similar to large language models (LLaMA-style)"""
        # LLaMA-style configuration: many query heads, fewer KV heads
        batch_size, seq_len_q, seq_len_k = 4, 32, 32
        num_heads, kv_num_heads, head_dim = 32, 8, 128  # 4 queries per KV head

        inT, outT, input_names = create_group_query_attention_test_tensors(
            batch_size=batch_size, seq_len_q=seq_len_q, seq_len_k=seq_len_k,
            num_heads=num_heads, kv_num_heads=kv_num_heads, head_dim=head_dim,
            use_past=True, use_rope=True, causal=True
        )

        op_info = {
            'name': 'test_llama_style_attention',
            'optype': 'GroupQueryAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'kv_num_heads': kv_num_heads,
                'causal': True,
                'scale': 0.1,  # Custom scale for stability
            },
        }
        op = GroupQueryAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic large model configuration
        assert op.group_size == 4  # 32 query heads / 8 KV heads

        # Should have significant computational requirements
        assert perf_stats['instrs']['mac'] > 1000000  # Substantial attention computation
        assert perf_stats['instrs']['cmp'] > 0        # Causal masking
        assert perf_stats['instrs']['mul'] > 0        # RoPE operations

        # Memory requirements should be reasonable
        assert perf_stats['inBytes'] > 0
        assert perf_stats['outBytes'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
