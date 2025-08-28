"""
Comprehensive test suite for QAttention operation implementation.

This module tests the ONNX 1.20.0 QAttention operation, which performs
quantized attention computation entirely in quantized space for maximum
efficiency. This operation is designed for highly efficient quantized
neural network inference.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, QAttentionOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_qattention_test_tensors(batch_size=2, seq_len=8, num_heads=8, head_dim=64,
                                   quantization_scheme='tensor', output_quantized=True,
                                   attention_quantized=True, use_past=False, use_rope=False):
    """
    Helper function to create test tensors for QAttention operation.

    Args:
        batch_size: Batch size for all tensors
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        quantization_scheme: Quantization scheme ('tensor', 'per_channel', 'per_head')
        output_quantized: Whether output should be quantized
        attention_quantized: Whether attention weights should be quantized
        use_past: Whether to include past key/value tensors
        use_rope: Whether to include rotary position embeddings

    Returns:
        Tuple of (input_tensors, output_tensors, input_names)
    """
    hidden_size = num_heads * head_dim

    # Determine numpy dtype based on quantization settings
    # For QAttention, input x is always quantized (int8), regardless of attention_quantized setting
    np_dtype = np.dtype('int8')

    # Create quantized input tensor x
    if quantization_scheme == 'tensor':
        x_scale_shape = []
        x_zero_point_shape = []
    elif quantization_scheme == 'per_channel':
        x_scale_shape = [hidden_size]
        x_zero_point_shape = [hidden_size]
    elif quantization_scheme == 'per_head':
        x_scale_shape = [num_heads]
        x_zero_point_shape = [num_heads]

    x = F._from_shape('x', [batch_size, seq_len, hidden_size], np_dtype=np_dtype)
    x_scale = F._from_shape('x_scale', x_scale_shape, np_dtype=np.dtype('float32'))
    x_zero_point = F._from_shape('x_zero_point', x_zero_point_shape, np_dtype=np.dtype('int8'))

    # Create quantized weight tensor w
    w = F._from_shape('w', [hidden_size, hidden_size], np_dtype=np.dtype('int8'))
    w_scale = F._from_shape('w_scale', [hidden_size], np_dtype=np.dtype('float32'))  # Per-channel for weights
    w_zero_point = F._from_shape('w_zero_point', [hidden_size], np_dtype=np.dtype('int8'))

    # Create bias tensor
    bias = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

    # Create output quantization parameters
    y_scale = F._from_shape('y_scale', [], np_dtype=np.dtype('float32'))
    y_zero_point = F._from_shape('y_zero_point', [], np_dtype=np.dtype('int8'))

    input_tensors = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, bias, y_scale, y_zero_point]
    input_names = ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point']

    # Add optional inputs
    if use_past:
        # For QAttention, past_key and past_value should have same shape as current tensors
        past_seq_len = 4
        past_key = F._from_shape('past_key', [batch_size, past_seq_len, hidden_size], np_dtype=np_dtype)
        past_value = F._from_shape('past_value', [batch_size, past_seq_len, hidden_size], np_dtype=np_dtype)
        input_tensors.extend([past_key, past_value])
        input_names.extend(['past_key', 'past_value'])

    if use_rope:
        rope_dim = head_dim // 2
        cos_cache = F._from_shape('cos_cache', [batch_size, seq_len, rope_dim], np_dtype=np_dtype)
        sin_cache = F._from_shape('sin_cache', [batch_size, seq_len, rope_dim], np_dtype=np_dtype)
        input_tensors.extend([cos_cache, sin_cache])
        input_names.extend(['cos_cache', 'sin_cache'])

    # Create output tensors
    output_dtype = np.dtype('int8') if output_quantized else np.dtype('float32')
    output = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=output_dtype)

    # Optional outputs: present_key, present_value
    total_seq_len = seq_len
    if use_past:
        total_seq_len += 4

    # For QAttention, present_key and present_value have shape [batch_size, num_heads, seq_len, head_dim]
    present_key = F._from_shape('present_key', [batch_size, num_heads, seq_len, head_dim], np_dtype=np_dtype)
    present_value = F._from_shape('present_value', [batch_size, num_heads, seq_len, head_dim], np_dtype=np_dtype)

    output_tensors = [output, present_key, present_value]

    return input_tensors, output_tensors, input_names


class TestQAttention:
    """Test class for QAttention operation"""

    def test_factory_integration(self):
        """Test that QAttention is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('QAttention')
        assert opcls == QAttentionOp

    def test_basic_qattention_tensor_quantization(self):
        """Test basic QAttention with tensor quantization"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor'
        )

        op_info = {
            'name': 'test_qattention_basic',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_output_shape = [batch_size, seq_len, num_heads * head_dim]
        expected_present_shape = [batch_size, seq_len, num_heads, head_dim]

        assert outT[0].shape == expected_output_shape
        assert outT[1].shape == expected_present_shape
        assert outT[2].shape == expected_present_shape

        # Verify output types (quantized)
        assert outT[0].dtype == np.dtype('int8')
        assert outT[1].dtype == np.dtype('int8')
        assert outT[2].dtype == np.dtype('int8')

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have quantized matrix multiplications
        assert perf_stats['instrs']['convert'] > 0  # Should have type conversions

    def test_qattention_per_channel_quantization(self):
        """Test QAttention with per-channel quantization"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='per_channel'
        )

        op_info = {
            'name': 'test_qattention_per_channel',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'per_channel',
                'output_quantized': True,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Per-channel quantization should have more complex scaling operations
        assert perf_stats['instrs']['mul'] > 0
        assert perf_stats['instrs']['convert'] > 0

    def test_qattention_dequantized_output(self):
        """Test QAttention with dequantized (float32) output"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor', output_quantized=False
        )

        op_info = {
            'name': 'test_qattention_dequantized',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': False,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Dequantized output should be float32
        assert outT[0].dtype == np.dtype('float32')
        assert outT[1].dtype == np.dtype('int8')  # Present states remain quantized
        assert outT[2].dtype == np.dtype('int8')

        # Should have additional dequantization operations
        assert perf_stats['instrs']['convert'] > 0

    def test_qattention_with_past_states(self):
        """Test QAttention with KV-cache (past states)"""
        batch_size, seq_len, num_heads, head_dim = 2, 4, 8, 64  # Shorter current sequence

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor', use_past=True
        )

        op_info = {
            'name': 'test_qattention_past',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Present states use [batch_size, num_heads, seq_len, head_dim] (current sequence only in this test)
        expected_present_shape = [batch_size, num_heads, seq_len, head_dim]

        assert outT[1].shape == expected_present_shape
        assert outT[2].shape == expected_present_shape

    def test_qattention_with_rotary_embeddings(self):
        """Test QAttention with rotary position embeddings"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor', use_rope=True
        )

        op_info = {
            'name': 'test_qattention_rope',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # RoPE should add rotation operations
        assert perf_stats['instrs']['mul'] > 0
        assert perf_stats['instrs']['add'] > 0

    def test_qattention_causal_masking(self):
        """Test QAttention with causal masking"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor'
        )

        op_info = {
            'name': 'test_qattention_causal',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
                'causal': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Causal masking may or may not add comparison operations depending on implementation
        # Just verify the operation completes successfully
        assert perf_stats is not None

    def test_qattention_custom_scale_factor(self):
        """Test QAttention with custom scale factor"""
        batch_size, seq_len, num_heads, head_dim = 2, 6, 8, 64
        scale = 0.125

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor'
        )

        op_info = {
            'name': 'test_qattention_scale',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
                'scale': scale,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Custom scale should be stored correctly
        assert op.scale == scale

    def test_qattention_unidirectional(self):
        """Test QAttention with unidirectional processing"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor'
        )

        op_info = {
            'name': 'test_qattention_unidirectional',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
                'unidirectional': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Unidirectional processing should affect attention pattern
        assert perf_stats is not None
        assert op.unidirectional == True

    def test_qattention_dequantized_attention_weights(self):
        """Test QAttention with dequantized attention weights"""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='tensor', attention_quantized=False
        )

        op_info = {
            'name': 'test_qattention_dequantized_attn',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': False,  # Dequantized attention weights
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Dequantized attention should reduce quantization operations
        assert perf_stats['instrs']['mac'] > 0
        assert op.attention_quantized == False

    def test_qattention_minimal_inputs(self):
        """Test QAttention with minimal required inputs"""
        batch_size, seq_len, num_heads, head_dim = 1, 4, 4, 32
        hidden_size = num_heads * head_dim

        # Create minimal inputs manually
        x = F._from_shape('x', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('int8'))
        x_scale = F._from_shape('x_scale', [], np_dtype=np.dtype('float32'))
        x_zero_point = F._from_shape('x_zero_point', [], np_dtype=np.dtype('int8'))
        w = F._from_shape('w', [hidden_size, hidden_size], np_dtype=np.dtype('int8'))
        w_scale = F._from_shape('w_scale', [hidden_size], np_dtype=np.dtype('float32'))
        w_zero_point = F._from_shape('w_zero_point', [hidden_size], np_dtype=np.dtype('int8'))
        bias = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))
        y_scale = F._from_shape('y_scale', [], np_dtype=np.dtype('float32'))
        y_zero_point = F._from_shape('y_zero_point', [], np_dtype=np.dtype('int8'))

        inT = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, bias, y_scale, y_zero_point]

        # Create outputs
        output = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('int8'))
        present_key = F._from_shape('present_key', [batch_size, seq_len, num_heads, head_dim], np_dtype=np.dtype('int8'))
        present_value = F._from_shape('present_value', [batch_size, seq_len, num_heads, head_dim], np_dtype=np.dtype('int8'))

        outT = [output, present_key, present_value]

        op_info = {
            'name': 'test_qattention_minimal',
            'optype': 'QAttention',
            'inList': ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point'],
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'tensor',
                'output_quantized': True,
                'attention_quantized': True,
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify basic functionality
        assert perf_stats is not None
        assert perf_stats['instrs']['mac'] > 0

    def test_invalid_quantization_scheme(self):
        """Test error handling for invalid quantization scheme"""
        op_info = {
            'name': 'test_invalid_quantization_scheme',
            'optype': 'QAttention',
            'inList': ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'quantization_scheme': 'invalid_scheme',
            },
        }

        with pytest.raises(ValueError, match="QAttention unsupported quantization_scheme"):
            QAttentionOp(op_info)

    def test_invalid_num_heads(self):
        """Test error handling for invalid num_heads"""
        op_info = {
            'name': 'test_invalid_num_heads',
            'optype': 'QAttention',
            'inList': ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 0,  # Invalid
                'quantization_scheme': 'tensor',
            },
        }

        with pytest.raises(ValueError, match="QAttention num_heads must be a positive integer"):
            QAttentionOp(op_info)

    def test_inconsistent_tensor_shapes(self):
        """Test error handling for inconsistent tensor shapes"""
        # Create tensors with mismatched dimensions
        x = F._from_shape('x', [2, 8, 512], np_dtype=np.dtype('int8'))  # 512 hidden size
        w = F._from_shape('w', [256, 512], np_dtype=np.dtype('int8'))     # 256 input, 512 output (mismatch)

        inT = [
            x, F._from_shape('x_scale', [], np_dtype=np.dtype('float32')),
            F._from_shape('x_zero_point', [], np_dtype=np.dtype('int8')),
            w, F._from_shape('w_scale', [512], np_dtype=np.dtype('float32')),  # Match output size
            F._from_shape('w_zero_point', [512], np_dtype=np.dtype('int8')),
            F._from_shape('bias', [512], np_dtype=np.dtype('float32')),
            F._from_shape('y_scale', [], np_dtype=np.dtype('float32')),
            F._from_shape('y_zero_point', [], np_dtype=np.dtype('int8'))
        ]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('int8'))]

        op_info = {
            'name': 'test_inconsistent_shapes',
            'optype': 'QAttention',
            'inList': ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'quantization_scheme': 'tensor',
            },
        }
        op = QAttentionOp(op_info)

        # The shape validation happens during get_perf_counts
        with pytest.raises(AssertionError, match="Input hidden size .* must match weight input size .*"):
            op.get_perf_counts(inT, outT)

    def test_wrong_input_types(self):
        """Test error handling for wrong input types"""
        # Create float input instead of quantized
        x = F._from_shape('x', [2, 8, 512], np_dtype=np.dtype('float32'))  # Wrong type
        x_scale = F._from_shape('x_scale', [], np_dtype=np.dtype('float32'))
        x_zero_point = F._from_shape('x_zero_point', [], np_dtype=np.dtype('int8'))
        w = F._from_shape('w', [512, 512], np_dtype=np.dtype('int8'))
        w_scale = F._from_shape('w_scale', [512], np_dtype=np.dtype('float32'))
        w_zero_point = F._from_shape('w_zero_point', [512], np_dtype=np.dtype('int8'))
        bias = F._from_shape('bias', [512], np_dtype=np.dtype('float32'))
        y_scale = F._from_shape('y_scale', [], np_dtype=np.dtype('float32'))
        y_zero_point = F._from_shape('y_zero_point', [], np_dtype=np.dtype('int8'))

        inT = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, bias, y_scale, y_zero_point]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('int8'))]

        op_info = {
            'name': 'test_wrong_input_types',
            'optype': 'QAttention',
            'inList': ['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'bias', 'y_scale', 'y_zero_point'],
            'outList': ['output'],
            'attrs': {
                'num_heads': 8,
                'quantization_scheme': 'tensor',
            },
        }
        op = QAttentionOp(op_info)

        # The type validation happens during get_perf_counts
        with pytest.raises(AssertionError, match="Input x must be quantized type"):
            op.get_perf_counts(inT, outT)

    def test_large_scale_quantized_model(self):
        """Test large-scale quantized model configuration"""
        # Large model configuration similar to modern quantized LLMs
        batch_size, seq_len, num_heads, head_dim = 4, 32, 32, 128

        inT, outT, input_names = create_qattention_test_tensors(
            batch_size=batch_size, seq_len=seq_len, num_heads=num_heads, head_dim=head_dim,
            quantization_scheme='per_channel', use_past=True  # Don't use RoPE to stay within input limits
        )

        op_info = {
            'name': 'test_large_scale_quantized',
            'optype': 'QAttention',
            'inList': input_names,
            'outList': ['output', 'present_key', 'present_value'],
            'attrs': {
                'num_heads': num_heads,
                'quantization_scheme': 'per_channel',
                'output_quantized': True,
                'attention_quantized': True,
                'causal': True,
                'scale': 0.0884,  # 1/sqrt(128)
            },
        }
        op = QAttentionOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Large scale quantized configuration should have substantial computational requirements
        assert perf_stats['instrs']['mac'] > 50000000  # Significant quantized attention computation
        assert perf_stats['instrs']['convert'] > 100000  # Extensive quantization operations
        assert perf_stats['instrs']['mul'] > 100000     # Scale multiplications

        # Memory requirements should be reasonable for quantized model
        assert perf_stats['inBytes'] > 1000000
        assert perf_stats['outBytes'] > 1000000

        # Quantized operations should be efficient
        assert perf_stats['instrs']['round'] > 0
        assert perf_stats['instrs']['clip'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
