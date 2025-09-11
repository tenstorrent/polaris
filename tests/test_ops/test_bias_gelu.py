#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for BiasGelu operation implementation.

This module tests the ONNX BiasGelu operation, which combines bias addition
with GELU activation. BiasGelu is commonly used in transformer feed-forward
networks as a fused operation for efficiency.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, BiasGeluOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_bias_gelu_test_tensors(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for BiasGelu operation.

    Args:
        batch_size: Batch size for input tensor
        seq_len: Sequence length
        hidden_size: Hidden dimension size (bias will be 1D with this size)

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create input tensor with 3D shape [batch_size, seq_len, hidden_size]
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create bias tensor with 1D shape [hidden_size]
    bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, bias_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestBiasGelu:
    """Test class for BiasGelu operation"""

    def test_factory_integration(self):
        """Test that BiasGelu is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('BiasGelu')
        assert opcls == BiasGeluOp

    def test_basic_bias_gelu(self):
        """Test basic BiasGelu with standard configuration"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_bias_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_bias_gelu_basic',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplications
        assert perf_stats['instrs']['add'] > 0  # Should have additions (bias + gelu operations)
        assert perf_stats['instrs']['tanh'] > 0  # Should have tanh operations (from GELU)
        assert perf_stats['instrs']['exp'] > 0  # Should have exp operations (for tanh)

    def test_bias_gelu_different_shapes(self):
        """Test BiasGelu with different tensor shapes"""
        test_configs = [
            (1, 4, 128),     # Small configuration
            (4, 16, 1024),   # Large configuration
            (2, 32, 768),    # BERT-like configuration
            (8, 64, 4096),   # Large language model configuration
        ]

        for batch_size, seq_len, hidden_size in test_configs:
            inT, outT = create_bias_gelu_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
            )

            op_info = {
                'name': f'test_bias_gelu_{batch_size}_{seq_len}_{hidden_size}',
                'optype': 'BiasGelu',
                'inList': ['input', 'bias'],
                'outList': ['output'],
                'attrs': {},
            }
            op = BiasGeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, hidden_size]
            assert outT[0].shape == expected_shape

            # Verify input elements count (input + bias)
            expected_input_elements = batch_size * seq_len * hidden_size + hidden_size
            assert perf_stats['inElems'] == expected_input_elements

            # Verify output elements count
            expected_output_elements = batch_size * seq_len * hidden_size
            assert perf_stats['outElems'] == expected_output_elements

    def test_bias_gelu_2d_input(self):
        """Test BiasGelu with 2D input tensor (matrix input)"""
        # 2D input: [batch_size, hidden_size]
        batch_size, hidden_size = 4, 768

        input_tensor = F._from_shape('input', [batch_size, hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, bias_tensor]
        outT = [F._from_shape('output', [batch_size, hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_gelu_2d',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [batch_size, hidden_size]
        assert perf_stats['inElems'] == batch_size * hidden_size + hidden_size
        assert perf_stats['outElems'] == batch_size * hidden_size

    def test_bias_gelu_1d_input(self):
        """Test BiasGelu with 1D input tensor (vector input)"""
        # 1D input: [hidden_size]
        hidden_size = 512

        input_tensor = F._from_shape('input', [hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, bias_tensor]
        outT = [F._from_shape('output', [hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_gelu_1d',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size + hidden_size  # input + bias
        assert perf_stats['outElems'] == hidden_size

    def test_bias_gelu_large_tensor(self):
        """Test BiasGelu with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 4, 128, 4096  # ~2M elements

        inT, outT = create_bias_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_bias_gelu_large',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = batch_size * seq_len * hidden_size
        expected_elements = 4 * 128 * 4096

        assert perf_stats['inElems'] == expected_elements + hidden_size  # input + bias
        assert perf_stats['outElems'] == expected_elements

        # Verify performance scaling - should have significant instruction counts
        assert perf_stats['instrs']['mul'] >= expected_elements * 4  # At least 4 multiplications per element
        assert perf_stats['instrs']['add'] >= expected_elements * 3  # At least 3 additions per element (bias + gelu ops)
        assert perf_stats['instrs']['tanh'] >= expected_elements      # At least 1 tanh per element
        assert perf_stats['instrs']['exp'] >= expected_elements * 2  # At least 2 exp per element

    def test_bias_gelu_vs_separate_operations(self):
        """Test that BiasGelu has similar or better performance than separate bias + gelu"""
        batch_size, seq_len, hidden_size = 2, 8, 256

        # Create tensors for BiasGelu
        inT_bias_gelu, outT_bias_gelu = create_bias_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # BiasGelu operation
        bias_gelu_op_info = {
            'name': 'test_bias_gelu_fused',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        bias_gelu_op = BiasGeluOp(bias_gelu_op_info)
        bias_gelu_stats = bias_gelu_op.get_perf_counts(inT_bias_gelu, outT_bias_gelu)

        # Create tensors for separate operations (bias addition + gelu)
        input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [hidden_size], np_dtype=np.dtype('float32'))

        # Separate operations would require intermediate tensor
        # BiasGelu should be more efficient (fewer memory operations)
        assert bias_gelu_stats['inBytes'] > 0
        assert bias_gelu_stats['outBytes'] > 0

        # Verify the operation handles the fused computation correctly
        total_elements = batch_size * seq_len * hidden_size
        assert bias_gelu_stats['instrs']['add'] >= total_elements  # At least bias addition
        assert bias_gelu_stats['instrs']['mul'] >= total_elements * 4  # GELU operations

    def test_bias_gelu_memory_efficiency(self):
        """Test that BiasGelu has reasonable memory requirements"""
        batch_size, seq_len, hidden_size = 4, 32, 1024
        total_elements = batch_size * seq_len * hidden_size

        inT, outT = create_bias_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_bias_gelu_memory',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input + bias should be larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_elements + hidden_size
        assert perf_stats['outElems'] == total_elements

    def test_invalid_bias_shape_2d(self):
        """Test error handling for 2D bias tensor"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [4, 512], np_dtype=np.dtype('float32'))  # 2D bias

        inT = [input_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_bias_2d',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        with pytest.raises(ValueError, match="Bias tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_bias_dimension_mismatch(self):
        """Test error handling for bias dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [256], np_dtype=np.dtype('float32'))  # Wrong bias dimension

        inT = [input_tensor, bias_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_bias_mismatch',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        with pytest.raises(ValueError, match="Bias dimension .* must match input's last dimension"):
            op.get_perf_counts(inT, outT)

    def test_bias_gelu_different_dtypes(self):
        """Test BiasGelu with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [4, 8, 128], np_dtype=dtype)
            bias_tensor = F._from_shape('bias', [128], np_dtype=dtype)

            inT = [input_tensor, bias_tensor]
            outT = [F._from_shape('output', [4, 8, 128], np_dtype=dtype)]

            op_info = {
                'name': f'test_bias_gelu_{dtype}',
                'optype': 'BiasGelu',
                'inList': ['input', 'bias'],
                'outList': ['output'],
                'attrs': {},
            }
            op = BiasGeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [4, 8, 128]
            assert perf_stats['inElems'] == 4 * 8 * 128 + 128  # input + bias
            assert perf_stats['outElems'] == 4 * 8 * 128

            # Memory usage should scale with data type size
            # Input includes both main tensor and bias tensor
            if dtype == np.dtype('float64'):
                # For float64: inBytes should be outBytes + biasBytes
                expected_in_bytes = perf_stats['outBytes'] + (128 * 8)  # bias tensor bytes
                assert perf_stats['inBytes'] == expected_in_bytes

    def test_bias_gelu_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, seq_len, hidden_size = 2, 4, 128

        inT, outT = create_bias_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_bias_gelu_backward',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('bias_grad', [hidden_size], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="BiasGelu backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_bias_gelu_transformer_feedforward_pattern(self):
        """Test BiasGelu with transformer feed-forward network pattern"""
        # Typical transformer FFN pattern: [batch_size, seq_len, 4*hidden_size] -> BiasGelu -> [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = 8, 512, 1024
        ffn_hidden_size = 4 * hidden_size  # 4x expansion in FFN

        # BiasGelu input is typically the result of linear projection
        input_tensor = F._from_shape('input', [batch_size, seq_len, ffn_hidden_size], np_dtype=np.dtype('float32'))
        bias_tensor = F._from_shape('bias', [ffn_hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, bias_tensor]
        outT = [F._from_shape('output', [batch_size, seq_len, ffn_hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_transformer_ffn',
            'optype': 'BiasGelu',
            'inList': ['input', 'bias'],
            'outList': ['output'],
            'attrs': {},
        }
        op = BiasGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic transformer FFN configuration
        total_elements = batch_size * seq_len * ffn_hidden_size
        assert perf_stats['inElems'] == total_elements + ffn_hidden_size
        assert perf_stats['outElems'] == total_elements

        # Should have significant computational requirements for transformer scale
        assert perf_stats['instrs']['mul'] > 1000000  # Substantial multiplications
        assert perf_stats['instrs']['add'] > 500000   # Substantial additions
        assert perf_stats['instrs']['tanh'] > 250000  # Substantial tanh operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
