#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for SkipLayerNormalization operation implementation.

This module tests the ONNX SkipLayerNormalization operation, which implements
the pre-layer normalization pattern commonly used in transformer architectures.
This fused operation combines skip connection addition with layer normalization
for better efficiency.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, SkipLayerNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_skip_layer_normalization_test_tensors(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for SkipLayerNormalization operation.

    Args:
        batch_size: Batch size for input tensors
        seq_len: Sequence length
        hidden_size: Hidden dimension size (normalization happens along this dimension)

    Returns:
        Tuple of (input_tensors, output_tensors) for 4-input version (with bias)
    """
    # Create input tensor with 3D shape [batch_size, seq_len, hidden_size]
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create skip tensor with same shape as input
    skip_tensor = F._from_shape('skip', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create gamma (scale) tensor with 1D shape [hidden_size]
    gamma_tensor = F._from_shape('gamma', [hidden_size], np_dtype=np.dtype('float32'))

    # Create beta (bias) tensor with 1D shape [hidden_size]
    beta_tensor = F._from_shape('beta', [hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_skip_layer_normalization_test_tensors_no_bias(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for SkipLayerNormalization operation without bias.

    Args:
        batch_size: Batch size for input tensors
        seq_len: Sequence length
        hidden_size: Hidden dimension size

    Returns:
        Tuple of (input_tensors, output_tensors) for 3-input version (without bias)
    """
    # Create input tensor with 3D shape [batch_size, seq_len, hidden_size]
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create skip tensor with same shape as input
    skip_tensor = F._from_shape('skip', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    # Create gamma (scale) tensor with 1D shape [hidden_size]
    gamma_tensor = F._from_shape('gamma', [hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor, skip_tensor, gamma_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestSkipLayerNormalization:
    """Test class for SkipLayerNormalization operation"""

    def test_factory_integration(self):
        """Test that SkipLayerNormalization is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('SkipLayerNormalization')
        assert opcls == SkipLayerNormalizationOp

    def test_basic_skip_layer_normalization(self):
        """Test basic SkipLayerNormalization with standard 3D configuration"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_skip_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_skip_layernorm_basic',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have multiply-accumulate operations
        assert perf_stats['instrs']['add'] > 0  # Should have additions (skip + bias)
        assert perf_stats['instrs']['sub'] > 0  # Should have subtractions
        assert perf_stats['instrs']['rsqrt'] > 0  # Should have reciprocal square root

    def test_skip_layer_normalization_without_bias(self):
        """Test SkipLayerNormalization without bias tensor"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_skip_layer_normalization_test_tensors_no_bias(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_skip_layernorm_no_bias',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify input elements count (input + skip + gamma, no beta)
        expected_input_elements = batch_size * seq_len * hidden_size * 2 + hidden_size
        assert perf_stats['inElems'] == expected_input_elements

        # Verify output elements count
        expected_output_elements = batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == expected_output_elements

    def test_skip_layer_normalization_different_shapes(self):
        """Test SkipLayerNormalization with different tensor shapes"""
        test_configs = [
            (1, 4, 128),     # Small configuration
            (4, 16, 1024),   # Large configuration
            (2, 32, 768),    # BERT-like configuration
            (8, 64, 4096),   # Large language model configuration
        ]

        for batch_size, seq_len, hidden_size in test_configs:
            inT, outT = create_skip_layer_normalization_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
            )

            op_info = {
                'name': f'test_skip_layernorm_{batch_size}_{seq_len}_{hidden_size}',
                'optype': 'SkipLayerNormalization',
                'inList': ['input', 'skip', 'gamma', 'beta'],
                'outList': ['output'],
                'attrs': {
                    'epsilon': 1e-5,
                },
            }
            op = SkipLayerNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, hidden_size]
            assert outT[0].shape == expected_shape

            # Verify performance scales correctly
            total_elements = batch_size * seq_len * hidden_size
            assert perf_stats['inElems'] == total_elements * 2 + hidden_size + hidden_size  # input + skip + gamma + beta
            assert perf_stats['outElems'] == total_elements

    def test_skip_layer_normalization_2d_input(self):
        """Test SkipLayerNormalization with 2D input tensor"""
        batch_size, hidden_size = 4, 768

        input_tensor = F._from_shape('input', [batch_size, hidden_size], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [batch_size, hidden_size], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [hidden_size], np_dtype=np.dtype('float32'))
        beta_tensor = F._from_shape('beta', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [batch_size, hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_skip_layernorm_2d',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [batch_size, hidden_size]
        assert perf_stats['inElems'] == batch_size * hidden_size * 2 + hidden_size + hidden_size
        assert perf_stats['outElems'] == batch_size * hidden_size

    def test_skip_layer_normalization_1d_input(self):
        """Test SkipLayerNormalization with 1D input tensor"""
        hidden_size = 512

        input_tensor = F._from_shape('input', [hidden_size], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [hidden_size], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [hidden_size], np_dtype=np.dtype('float32'))
        beta_tensor = F._from_shape('beta', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [hidden_size], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_skip_layernorm_1d',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size * 2 + hidden_size + hidden_size  # input + skip + gamma + beta
        assert perf_stats['outElems'] == hidden_size

    def test_skip_layer_normalization_large_tensor(self):
        """Test SkipLayerNormalization with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 4, 128, 4096  # ~8M elements

        inT, outT = create_skip_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_skip_layernorm_large',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = batch_size * seq_len * hidden_size
        expected_elements = 4 * 128 * 4096

        assert perf_stats['inElems'] == expected_elements * 2 + hidden_size + hidden_size  # input + skip + gamma + beta
        assert perf_stats['outElems'] == expected_elements

        # Verify performance scaling - should have substantial operations for large tensors
        assert perf_stats['instrs']['mac'] >= expected_elements  # At least 1 MAC per element
        assert perf_stats['instrs']['add'] >= expected_elements * 2  # At least 2 additions per element
        assert perf_stats['instrs']['sub'] >= expected_elements  # At least 1 subtraction per element
        assert perf_stats['instrs']['rsqrt'] >= expected_elements // hidden_size  # Reciprocal square root operations

    def test_skip_layer_normalization_memory_efficiency(self):
        """Test that SkipLayerNormalization has reasonable memory requirements"""
        batch_size, seq_len, hidden_size = 4, 32, 1024
        total_elements = batch_size * seq_len * hidden_size

        inT, outT = create_skip_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_skip_layernorm_memory',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input + skip + gamma + beta should be much larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_elements * 2 + hidden_size + hidden_size
        assert perf_stats['outElems'] == total_elements

    def test_input_skip_shape_mismatch(self):
        """Test error handling for input and skip tensor shape mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [2, 10, 512], np_dtype=np.dtype('float32'))  # Wrong seq length
        gamma_tensor = F._from_shape('gamma', [512], np_dtype=np.dtype('float32'))
        beta_tensor = F._from_shape('beta', [512], np_dtype=np.dtype('float32'))

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_input_skip_mismatch',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Input and skip tensors must have the same shape"):
            op.get_perf_counts(inT, outT)

    def test_gamma_dimension_mismatch(self):
        """Test error handling for gamma dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [2, 8, 512], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [256], np_dtype=np.dtype('float32'))  # Wrong dimension
        beta_tensor = F._from_shape('beta', [512], np_dtype=np.dtype('float32'))

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_gamma_mismatch',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Gamma dimension .* must match input's last dimension"):
            op.get_perf_counts(inT, outT)

    def test_beta_dimension_mismatch(self):
        """Test error handling for beta dimension mismatch"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [2, 8, 512], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [512], np_dtype=np.dtype('float32'))
        beta_tensor = F._from_shape('beta', [256], np_dtype=np.dtype('float32'))  # Wrong dimension

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_beta_mismatch',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Beta dimension .* must match input's last dimension"):
            op.get_perf_counts(inT, outT)

    def test_gamma_not_1d(self):
        """Test error handling for non-1D gamma tensor"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [2, 8, 512], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [4, 128], np_dtype=np.dtype('float32'))  # 2D gamma
        beta_tensor = F._from_shape('beta', [512], np_dtype=np.dtype('float32'))

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_gamma_not_1d',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Gamma tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_beta_not_1d(self):
        """Test error handling for non-1D beta tensor"""
        input_tensor = F._from_shape('input', [2, 8, 512], np_dtype=np.dtype('float32'))
        skip_tensor = F._from_shape('skip', [2, 8, 512], np_dtype=np.dtype('float32'))
        gamma_tensor = F._from_shape('gamma', [512], np_dtype=np.dtype('float32'))
        beta_tensor = F._from_shape('beta', [4, 128], np_dtype=np.dtype('float32'))  # 2D beta

        inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
        outT = [F._from_shape('output', [2, 8, 512], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_beta_not_1d',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        with pytest.raises(ValueError, match="Beta tensor must be 1D"):
            op.get_perf_counts(inT, outT)

    def test_invalid_epsilon_zero(self):
        """Test error handling for zero epsilon"""
        op_info = {
            'name': 'test_invalid_epsilon',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 0,
            },
        }

        with pytest.raises(ValueError, match="SkipLayerNormalization epsilon must be positive"):
            SkipLayerNormalizationOp(op_info)

    def test_skip_layer_normalization_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, seq_len, hidden_size = 2, 4, 128

        inT, outT = create_skip_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_skip_layernorm_backward',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-5,
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('skip_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('gamma_grad', [hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('beta_grad', [hidden_size], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="SkipLayerNormalization backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_skip_layer_normalization_different_dtypes(self):
        """Test SkipLayerNormalization with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [2, 8, 128], np_dtype=dtype)
            skip_tensor = F._from_shape('skip', [2, 8, 128], np_dtype=dtype)
            gamma_tensor = F._from_shape('gamma', [128], np_dtype=dtype)
            beta_tensor = F._from_shape('beta', [128], np_dtype=dtype)

            inT = [input_tensor, skip_tensor, gamma_tensor, beta_tensor]
            outT = [F._from_shape('output', [2, 8, 128], np_dtype=dtype)]

            op_info = {
                'name': f'test_skip_layernorm_{dtype}',
                'optype': 'SkipLayerNormalization',
                'inList': ['input', 'skip', 'gamma', 'beta'],
                'outList': ['output'],
                'attrs': {
                    'epsilon': 1e-5,
                },
            }
            op = SkipLayerNormalizationOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [2, 8, 128]
            assert perf_stats['inElems'] == 2 * 8 * 128 * 2 + 128 + 128  # input + skip + gamma + beta
            assert perf_stats['outElems'] == 2 * 8 * 128

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # Input includes input, skip, gamma, and beta tensors
                # inBytes should be input + skip + gamma + beta bytes
                expected_in_bytes = (2 * 8 * 128 * 2 + 128 + 128) * 8  # total elements * 8 bytes per float64
                expected_out_bytes = 2 * 8 * 128 * 8  # output elements * 8 bytes per float64
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes

    def test_skip_layer_normalization_transformer_block_pattern(self):
        """Test SkipLayerNormalization with transformer block pattern"""
        # Typical transformer block pattern: normalize after attention with skip connection
        batch_size, seq_len, hidden_size = 8, 512, 1024

        inT, outT = create_skip_layer_normalization_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_transformer_block_norm',
            'optype': 'SkipLayerNormalization',
            'inList': ['input', 'skip', 'gamma', 'beta'],
            'outList': ['output'],
            'attrs': {
                'epsilon': 1e-6,  # Common epsilon value in transformers
            },
        }
        op = SkipLayerNormalizationOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify this represents a realistic transformer block configuration
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements * 2 + hidden_size + hidden_size
        assert perf_stats['outElems'] == total_elements

        # Should have significant computational requirements for transformer scale
        assert perf_stats['instrs']['mac'] > 1000000  # 1M+ operations
        assert perf_stats['instrs']['add'] > 5000000   # 5M+ operations
        assert perf_stats['instrs']['sub'] > 2500000   # 2.5M+ operations
        assert perf_stats['instrs']['rsqrt'] > 4000    # Reciprocal square root operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
