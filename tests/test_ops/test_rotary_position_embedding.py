"""
Comprehensive test suite for RotaryPositionEmbedding operation implementation.

This module tests the ONNX 1.20.0 RotaryPositionEmbedding operation, which applies
rotary position embeddings (RoPE) to input tensors. RoPE is a position encoding
technique used in modern transformer architectures like LLaMA, GPT-J, etc.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, RotaryPositionEmbeddingOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_rotary_position_embedding_test_tensors(batch_size=2, seq_len=8, num_heads=8,
                                                  head_dim=64, rotary_embedding_dim=32,
                                                  causal=False):
    """
    Helper function to create test tensors for RotaryPositionEmbedding operation.

    Args:
        batch_size: Batch size for all tensors
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        rotary_embedding_dim: Dimension for rotary embeddings (must be <= head_dim and even)
        causal: Whether to use causal masking

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    hidden_size = num_heads * head_dim

    # Create input tensor
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create cos and sin caches
    # Cache dimension is half of rotary_embedding_dim (for sin and cos components)
    cache_dim = rotary_embedding_dim // 2
    cos_cache = F._from_shape('cos_cache', [batch_size, seq_len, cache_dim], np_dtype=np.dtype('float32'))
    sin_cache = F._from_shape('sin_cache', [batch_size, seq_len, cache_dim], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, cos_cache, sin_cache]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestRotaryPositionEmbedding:
    """Test class for RotaryPositionEmbedding operation"""

    def test_factory_integration(self):
        """Test that RotaryPositionEmbedding is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('RotaryPositionEmbedding')
        assert opcls == RotaryPositionEmbeddingOp

    def test_basic_rotary_embedding(self):
        """Test basic rotary position embedding with standard configuration"""
        batch_size, seq_len, num_heads, head_dim, rotary_embedding_dim = 2, 8, 8, 64, 32

        inT, outT = create_rotary_position_embedding_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
            head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim
        )

        op_info = {
            'name': 'test_rotary_basic',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': num_heads,
                'rotary_embedding_dim': rotary_embedding_dim,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, num_heads * head_dim]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplications for rotary operations
        assert perf_stats['instrs']['add'] > 0  # Should have additions for combining results

    def test_rotary_embedding_with_causal_masking(self):
        """Test rotary position embedding with causal masking enabled"""
        batch_size, seq_len, num_heads, head_dim, rotary_embedding_dim = 2, 8, 8, 64, 32

        inT, outT = create_rotary_position_embedding_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
            head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim,
            causal=True
        )

        op_info = {
            'name': 'test_rotary_causal',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': num_heads,
                'rotary_embedding_dim': rotary_embedding_dim,
                'causal': True,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Causal masking should add comparison operations
        assert perf_stats['instrs']['cmp'] > 0

    def test_different_rotary_dimensions(self):
        """Test different rotary embedding dimensions"""
        test_configs = [
            (16, 16),  # Small rotary dimension
            (32, 32),  # Medium rotary dimension
            (64, 64),  # Large rotary dimension (full head)
        ]

        batch_size, seq_len, num_heads = 2, 6, 8

        for rotary_embedding_dim, head_dim in test_configs:
            inT, outT = create_rotary_position_embedding_test_tensors(
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
                head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim
            )

            op_info = {
                'name': f'test_rotary_dim_{rotary_embedding_dim}',
                'optype': 'RotaryPositionEmbedding',
                'inList': ['input', 'cos_cache', 'sin_cache'],
                'outList': ['output'],
                'attrs': {
                    'num_heads': num_heads,
                    'rotary_embedding_dim': rotary_embedding_dim,
                },
            }
            op = RotaryPositionEmbeddingOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, num_heads * head_dim]
            assert outT[0].shape == expected_shape

    def test_various_head_configurations(self):
        """Test different numbers of heads and configurations"""
        test_configs = [
            (4, 32, 16),   # Few heads, small head dim
            (8, 64, 32),   # Standard configuration
            (12, 128, 64), # More heads, larger head dim
            (16, 80, 40),  # Non-power-of-2 dimensions
        ]

        batch_size, seq_len = 2, 8

        for num_heads, head_dim, rotary_embedding_dim in test_configs:
            inT, outT = create_rotary_position_embedding_test_tensors(
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
                head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim
            )

            op_info = {
                'name': f'test_heads_{num_heads}_{head_dim}',
                'optype': 'RotaryPositionEmbedding',
                'inList': ['input', 'cos_cache', 'sin_cache'],
                'outList': ['output'],
                'attrs': {
                    'num_heads': num_heads,
                    'rotary_embedding_dim': rotary_embedding_dim,
                },
            }
            op = RotaryPositionEmbeddingOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, num_heads * head_dim]
            assert outT[0].shape == expected_shape

    def test_batch_size_variations(self):
        """Test different batch sizes"""
        batch_sizes = [1, 2, 4, 8]
        seq_len, num_heads, head_dim, rotary_embedding_dim = 8, 8, 64, 32

        for batch_size in batch_sizes:
            inT, outT = create_rotary_position_embedding_test_tensors(
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
                head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim
            )

            op_info = {
                'name': f'test_batch_{batch_size}',
                'optype': 'RotaryPositionEmbedding',
                'inList': ['input', 'cos_cache', 'sin_cache'],
                'outList': ['output'],
                'attrs': {
                    'num_heads': num_heads,
                    'rotary_embedding_dim': rotary_embedding_dim,
                },
            }
            op = RotaryPositionEmbeddingOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, num_heads * head_dim]
            assert outT[0].shape == expected_shape

    def test_sequence_length_variations(self):
        """Test different sequence lengths"""
        seq_lengths = [4, 8, 16, 32, 64, 128]
        batch_size, num_heads, head_dim, rotary_embedding_dim = 2, 8, 64, 32

        for seq_len in seq_lengths:
            inT, outT = create_rotary_position_embedding_test_tensors(
                batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
                head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim
            )

            op_info = {
                'name': f'test_seq_{seq_len}',
                'optype': 'RotaryPositionEmbedding',
                'inList': ['input', 'cos_cache', 'sin_cache'],
                'outList': ['output'],
                'attrs': {
                    'num_heads': num_heads,
                    'rotary_embedding_dim': rotary_embedding_dim,
                },
            }
            op = RotaryPositionEmbeddingOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, num_heads * head_dim]
            assert outT[0].shape == expected_shape

    def test_missing_num_heads(self):
        """Test error handling for missing num_heads attribute"""
        op_info = {
            'name': 'test_missing_num_heads',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'rotary_embedding_dim': 32,
            },
        }

        with pytest.raises(ValueError, match="RotaryPositionEmbedding requires 'num_heads' attribute"):
            RotaryPositionEmbeddingOp(op_info)

    def test_missing_rotary_embedding_dim(self):
        """Test error handling for missing rotary_embedding_dim attribute"""
        op_info = {
            'name': 'test_missing_rotary_dim',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
            },
        }

        with pytest.raises(ValueError, match="RotaryPositionEmbedding requires 'rotary_embedding_dim' attribute"):
            RotaryPositionEmbeddingOp(op_info)

    def test_invalid_rotary_embedding_dim(self):
        """Test error handling for invalid rotary_embedding_dim values"""
        test_cases = [
            (0, "rotary_embedding_dim must be positive"),
            (-1, "rotary_embedding_dim must be positive"),
            (31, "rotary_embedding_dim should be even"),  # Odd number
        ]

        for rotary_dim, expected_error in test_cases:
            op_info = {
                'name': f'test_invalid_dim_{rotary_dim}',
                'optype': 'RotaryPositionEmbedding',
                'inList': ['input', 'cos_cache', 'sin_cache'],
                'outList': ['output'],
                'attrs': {
                    'num_heads': 8,
                    'rotary_embedding_dim': rotary_dim,
                },
            }

            with pytest.raises(ValueError, match=expected_error):
                RotaryPositionEmbeddingOp(op_info)

    def test_invalid_input_shapes(self):
        """Test error handling for invalid input tensor shapes"""
        # Create tensors with mismatched shapes
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        cos_cache = F._from_shape('cos_cache', [2, 10, 16], np_dtype=np.dtype('float32'))  # Wrong seq length
        sin_cache = F._from_shape('sin_cache', [2, 10, 16], np_dtype=np.dtype('float32'))

        inT = [input_tensor, cos_cache, sin_cache]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_shapes',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'rotary_embedding_dim': 32,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        with pytest.raises(ValueError, match="Cache sequence length .* must match input sequence length"):
            op.get_perf_counts(inT, outT)

    def test_hidden_size_not_divisible_by_heads(self):
        """Test error handling for hidden size not divisible by num_heads"""
        # Create input with hidden_size not divisible by num_heads
        input_tensor = F._from_shape('input', [2, 8, 510], np_dtype=np.dtype('float32'))  # 510 % 8 != 0
        cos_cache = F._from_shape('cos_cache', [2, 8, 16], np_dtype=np.dtype('float32'))
        sin_cache = F._from_shape('sin_cache', [2, 8, 16], np_dtype=np.dtype('float32'))

        inT = [input_tensor, cos_cache, sin_cache]
        outT = [F._from_shape('output', [2, 8, 510], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_hidden_size',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'rotary_embedding_dim': 32,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        with pytest.raises(ValueError, match="Hidden size .* must be divisible by num_heads"):
            op.get_perf_counts(inT, outT)

    def test_rotary_dim_exceeds_head_dim(self):
        """Test error handling for rotary_embedding_dim exceeding head dimension"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))  # head_dim = 512/8 = 64
        cos_cache = F._from_shape('cos_cache', [2, 8, 16], np_dtype=np.dtype('float32'))
        sin_cache = F._from_shape('sin_cache', [2, 8, 16], np_dtype=np.dtype('float32'))

        inT = [input_tensor, cos_cache, sin_cache]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_rotary_too_large',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'rotary_embedding_dim': 128,  # Exceeds head_dim (64)
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        with pytest.raises(ValueError, match="rotary_embedding_dim .* cannot exceed head dimension"):
            op.get_perf_counts(inT, outT)

    def test_mismatched_cache_shapes(self):
        """Test error handling for mismatched cos_cache and sin_cache shapes"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        cos_cache = F._from_shape('cos_cache', [2, 8, 16], np_dtype=np.dtype('float32'))
        sin_cache = F._from_shape('sin_cache', [2, 8, 20], np_dtype=np.dtype('float32'))  # Different dimension

        inT = [input_tensor, cos_cache, sin_cache]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_mismatched_caches',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'rotary_embedding_dim': 32,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        with pytest.raises(ValueError, match="cos_cache and sin_cache must have same shape"):
            op.get_perf_counts(inT, outT)

    def test_wrong_cache_dimensions(self):
        """Test error handling for wrong cache dimensions"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        cos_cache = F._from_shape('cos_cache', [2, 8, 20], np_dtype=np.dtype('float32'))  # Wrong dimension (should be 16)
        sin_cache = F._from_shape('sin_cache', [2, 8, 20], np_dtype=np.dtype('float32'))

        inT = [input_tensor, cos_cache, sin_cache]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_wrong_cache_dim',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'rotary_embedding_dim': 32,  # Should result in cache_dim = 16
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        with pytest.raises(ValueError, match="Cache dimension .* should be half of rotary_embedding_dim"):
            op.get_perf_counts(inT, outT)

    def test_llama_style_configuration(self):
        """Test configuration similar to LLaMA models"""
        # LLaMA-style configuration: partial rotary embeddings
        batch_size, seq_len, num_heads, head_dim = 4, 32, 32, 128
        rotary_embedding_dim = 64  # Half of head dimension

        inT, outT = create_rotary_position_embedding_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads,
            head_dim=head_dim, rotary_embedding_dim=rotary_embedding_dim,
            causal=True
        )

        op_info = {
            'name': 'test_llama_style',
            'optype': 'RotaryPositionEmbedding',
            'inList': ['input', 'cos_cache', 'sin_cache'],
            'outList': ['output'],
            'attrs': {
                'num_heads': num_heads,
                'rotary_embedding_dim': rotary_embedding_dim,
                'causal': True,
            },
        }
        op = RotaryPositionEmbeddingOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic LLaMA-style configuration
        assert op.rotary_embedding_dim == 64
        assert op.num_heads == 32

        # Should have significant computational requirements for LLaMA scale
        assert perf_stats['instrs']['mul'] > 10000  # Substantial rotary operations
        assert perf_stats['instrs']['cmp'] > 0      # Causal masking

        # Verify output shape
        expected_shape = [batch_size, seq_len, num_heads * head_dim]
        assert outT[0].shape == expected_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
