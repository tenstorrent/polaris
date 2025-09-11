"""
Comprehensive test suite for SimplifiedLayerNormalization operation implementation.

This module tests the ONNX SimplifiedLayerNormalization operation, which provides
a simplified version of layer normalization commonly used in transformer models.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, SimplifiedLayerNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_simplified_layer_normalization_test_tensors(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for SimplifiedLayerNormalization operation.

    Args:
        batch_size: Batch size for input tensor
        seq_len: Sequence length
        hidden_size: Hidden dimension size (normalization happens along this dimension)

    Returns:
        Tuple of (input_tensors, output_tensors) for 2-input version (with bias)
    """
    # Create input tensor with 3D shape [batch_size, seq_len, hidden_size]
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create scale tensor with 1D shape [hidden_size]
    scale_tensor = F._from_shape('scale', [hidden_size], np_dtype=np.dtype('float32'))

    # Create bias tensor with 1D shape [hidden_size]
    bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, scale_tensor, bias_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_simplified_layer_normalization_test_tensors_no_bias(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for SimplifiedLayerNormalization operation without bias.

    Args:
        batch_size: Batch size for input tensor
        seq_len: Sequence length
        hidden_size: Hidden dimension size

    Returns:
        Tuple of (input_tensors, output_tensors) for 2-input version (without bias)
    """
    # Create input tensor with 3D shape [batch_size, seq_len, hidden_size]
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create scale tensor with 1D shape [hidden_size]
    scale_tensor = F._from_shape('scale', [hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, scale_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestSimplifiedLayerNormalization:
    """Test class for SimplifiedLayerNormalization operation"""

    def test_factory_integration(self):
        """Test that SimplifiedLayerNormalization is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('SimplifiedLayerNormalization')
        assert opcls == SimplifiedLayerNormalizationOp

    def test_basic_simplified_layer_normalization(self):
        """Test basic SimplifiedLayerNormalization with standard configuration"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_simplified_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_simplified_layernorm_basic',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have multiply-accumulate operations
        assert perf_stats['instrs']['add'] > 0  # Should have additions
        assert perf_stats['instrs']['sub'] > 0  # Should have subtractions
        assert perf_stats['instrs']['rsqrt'] > 0  # Should have reciprocal square root

    def test_simplified_layer_normalization_without_bias(self):
        """Test SimplifiedLayerNormalization without bias tensor"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_simplified_layer_normalization_test_tensors_no_bias(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_simplified_layernorm_no_bias',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify input elements count (input + scale, no bias)
        expected_input_elements = batch_size * seq_len * hidden_size + hidden_size
        assert perf_stats['inElems'] == expected_input_elements

        # Verify output elements count
        expected_output_elements = batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == expected_output_elements

    def test_simplified_layer_normalization_different_shapes(self):
        """Test SimplifiedLayerNormalization with different tensor shapes"""
        test_configs = [
            (1, 4, 128),     # Small configuration
            (4, 16, 1024),   # Large configuration
            (2, 32, 768),    # BERT-like configuration
            (8, 64, 4096),   # Large language model configuration
        ]

        for batch_size, seq_len, hidden_size in test_configs:
            inT, outT = create_simplified_layer_normalization_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
            )

            op_info = {
                'name': f'test_simplified_layernorm_{batch_size}_{seq_len}_{hidden_size}',
                'optype': 'SimplifiedLayerNormalization',
                'inList': ['input', 'scale', 'bias'],
                'outList': ['output'],
                'attrs': {},
            }
            op = SimplifiedLayerNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, hidden_size]
            assert outT[0].shape == expected_shape

            # Verify performance scales correctly
            total_elements = batch_size * seq_len * hidden_size
            assert perf_stats['inElems'] == total_elements + hidden_size + hidden_size  # input + scale + bias
            assert perf_stats['outElems'] == total_elements

    def test_simplified_layer_normalization_2d_input(self):
        """Test SimplifiedLayerNormalization with 2D input tensor"""
        batch_size, hidden_size = 4, 768

        input_tensor = F._from_shape('input', [batch_size, hidden_size], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [batch_size, hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_simplified_layernorm_2d',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [batch_size, hidden_size]
        assert perf_stats['inElems'] == batch_size * hidden_size + hidden_size + hidden_size
        assert perf_stats['outElems'] == batch_size * hidden_size

    def test_simplified_layer_normalization_1d_input(self):
        """Test SimplifiedLayerNormalization with 1D input tensor"""
        hidden_size = 512

        input_tensor = F._from_shape('input', [hidden_size], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_simplified_layernorm_1d',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size + hidden_size + hidden_size  # input + scale + bias
        assert perf_stats['outElems'] == hidden_size

    def test_simplified_layer_normalization_large_tensor(self):
        """Test SimplifiedLayerNormalization with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 4, 128, 4096  # ~2M elements

        inT, outT = create_simplified_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_simplified_layernorm_large',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = batch_size * seq_len * hidden_size
        expected_elements = 4 * 128 * 4096

        assert perf_stats['inElems'] == expected_elements + hidden_size + hidden_size  # input + scale + bias
        assert perf_stats['outElems'] == expected_elements

        # Verify performance scaling - should have significant instruction counts
        assert perf_stats['instrs']['mac'] >= expected_elements  # At least 1 MAC per element
        assert perf_stats['instrs']['add'] >= expected_elements  # At least 1 addition per element
        assert perf_stats['instrs']['sub'] >= expected_elements  # At least 1 subtraction per element
        assert perf_stats['instrs']['rsqrt'] >= expected_elements // hidden_size  # Reciprocal square root per sequence

    def test_scale_dimension_mismatch(self):
        """Test error handling for scale dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [256], np_dtype=np.dtype('float32'))  # Wrong scale dimension
        bias_tensor = F._from_shape('bias', [512], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_scale_mismatch',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Scale dimension .* must match input's last dimension"):
            op.get_perf_counts(inT, outT)

    def test_bias_dimension_mismatch(self):
        """Test error handling for bias dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [512], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [256], np_dtype=np.dtype('float32'))  # Wrong bias dimension

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_mismatch',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Bias dimension .* must match input's last dimension"):
            op.get_perf_counts(inT, outT)

    def test_scale_not_1d(self):
        """Test error handling for non-1D scale tensor"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [4, 512], np_dtype=np.dtype('float32'))  # 2D scale
        bias_tensor = F._from_shape('bias', [512], np_dtype=np.dtype('float32'))

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_scale_not_1d',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Scale tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_bias_not_1d(self):
        """Test error handling for non-1D bias tensor"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        scale_tensor = F._from_shape('scale', [512], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [4, 512], np_dtype=np.dtype('float32'))  # 2D bias

        inT = [input_tensor, scale_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_not_1d',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Bias tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_simplified_layer_normalization_memory_efficiency(self):
        """Test that SimplifiedLayerNormalization has reasonable memory requirements"""
        batch_size, seq_len, hidden_size = 4, 32, 1024
        total_elements = batch_size * seq_len * hidden_size

        inT, outT = create_simplified_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_simplified_layernorm_memory',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input + scale + bias should be larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_elements + hidden_size + hidden_size  # input + scale + bias
        assert perf_stats['outElems'] == total_elements

    def test_simplified_layer_normalization_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, seq_len, hidden_size = 2, 4, 128

        inT, outT = create_simplified_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_simplified_layernorm_backward',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('scale_grad', [hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('bias_grad', [hidden_size], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="SimplifiedLayerNormalization backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_simplified_layer_normalization_different_dtypes(self):
        """Test SimplifiedLayerNormalization with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [4, 8, 128], np_dtype=dtype)
            scale_tensor = F._from_shape('scale', [128], np_dtype=dtype)
            bias_tensor = F._from_shape('bias', [128], np_dtype=dtype)

            inT = [input_tensor, scale_tensor, bias_tensor]
            outT = [F._from_shape('output', [4, 8, 128], np_dtype=dtype)]

            op_info = {
                'name': f'test_simplified_layernorm_{dtype}',
                'optype': 'SimplifiedLayerNormalization',
                'inList': ['input', 'scale', 'bias'],
                'outList': ['output'],
                'attrs': {},
            }
            op = SimplifiedLayerNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [4, 8, 128]
            assert perf_stats['inElems'] == 4 * 8 * 128 + 128 + 128  # input + scale + bias
            assert perf_stats['outElems'] == 4 * 8 * 128

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # Input includes input, scale, and bias tensors
                # inBytes should be input + scale + bias bytes
                expected_in_bytes = (4 * 8 * 128 + 128 + 128) * 8  # total elements * 8 bytes per float64
                expected_out_bytes = 4 * 8 * 128 * 8  # output elements * 8 bytes per float64
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes

    def test_simplified_layer_normalization_transformer_pattern(self):
        """Test SimplifiedLayerNormalization with transformer layer pattern"""
        # Typical transformer layer pattern: normalize after attention and FFN
        batch_size, seq_len, hidden_size = 8, 512, 1024

        inT, outT = create_simplified_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_transformer_layernorm',
            'optype': 'SimplifiedLayerNormalization',
            'inList': ['input', 'scale', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimplifiedLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic transformer configuration
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements + hidden_size + hidden_size
        assert perf_stats['outElems'] == total_elements

        # Should have significant computational requirements for transformer scale
        assert perf_stats['instrs']['mac'] > 1000000  # Substantial MAC operations
        assert perf_stats['instrs']['add'] > 500000   # Substantial additions
        assert perf_stats['instrs']['sub'] > 250000   # Substantial subtractions
        assert perf_stats['instrs']['rsqrt'] > 4000   # Reciprocal square root operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
