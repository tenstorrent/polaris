#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for Gelu activation function implementation.

This module tests the ONNX 1.20.0 Gelu operation with extensions, which implements
the Gaussian Error Linear Unit activation function with two variants:
- Default: GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
- Tanh approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, GeluOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_gelu_test_tensors(batch_size=2, seq_len=8, hidden_size=64, dtype='float32'):
    """
    Helper function to create test tensors for Gelu operation.

    Args:
        batch_size: Batch size for the input tensor
        seq_len: Sequence length for the input tensor
        hidden_size: Hidden size dimension
        dtype: Data type for tensors

    Returns:
        Tuple of (input_tensors, output_tensors, input_names)
    """
    # Create input tensor
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype(dtype))

    input_tensors = [input_tensor]
    input_names = ['input']

    # Create output tensor
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype(dtype))

    output_tensors = [output_tensor]
    output_names = ['output']

    return input_tensors, output_tensors, input_names


def reference_gelu_tanh(x):
    """
    Reference implementation of GELU with tanh approximation.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def reference_gelu_erf(x):
    """
    Reference implementation of GELU with erf approximation (default variant).
    GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
    Using approximation: erf(x) ≈ tanh(1.41421356237 * x)
    """
    # Approximation of erf(x/sqrt(2)) using tanh
    return 0.5 * x * (1 + np.tanh(1.41421356237 * x))


class TestGelu:
    """Test class for Gelu operation"""

    def test_factory_integration(self):
        """Test that Gelu is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('Gelu')
        assert opcls == GeluOp

    def test_gelu_default_variant(self):
        """Test Gelu with default erf variant"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_gelu_default',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},  # No approximate attribute - should use default (erf)
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify output type matches input type
        assert outT[0].dtype == inT[0].dtype

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplication operations
        assert perf_stats['instrs']['add'] > 0  # Should have addition operations
        assert 'exp' in perf_stats['instrs']     # Should have exp operations for default variant (erf mapped to exp)

    def test_gelu_tanh_variant(self):
        """Test Gelu with tanh approximation variant"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_gelu_tanh',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'approximate': 'tanh'},  # Use tanh approximation
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics for tanh variant
        assert 'tanh' in perf_stats['instrs']     # Should have tanh operations
        assert perf_stats['instrs']['mul'] > 0   # Should have more multiplication operations than default

    def test_gelu_1d_tensor(self):
        """Test Gelu with 1D tensor"""
        hidden_size = 128

        # Create 1D input tensor
        input_tensor = F._from_shape('input_1d', [hidden_size], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_1d', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_gelu_1d',
            'optype': 'Gelu',
            'inList': ['input_1d'],
            'outList': ['output_1d'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 1D tensor handling
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size
        assert perf_stats['outElems'] == hidden_size

    def test_gelu_2d_tensor(self):
        """Test Gelu with 2D tensor (batch_size, features)"""
        batch_size, features = 4, 256

        # Create 2D input tensor
        input_tensor = F._from_shape('input_2d', [batch_size, features], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_2d', [batch_size, features], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_gelu_2d',
            'optype': 'Gelu',
            'inList': ['input_2d'],
            'outList': ['output_2d'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 2D tensor handling
        assert outT[0].shape == [batch_size, features]
        assert perf_stats['inElems'] == batch_size * features
        assert perf_stats['outElems'] == batch_size * features

    def test_gelu_3d_tensor(self):
        """Test Gelu with 3D tensor (batch_size, seq_len, hidden_size)"""
        batch_size, seq_len, hidden_size = 2, 16, 128

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_gelu_3d',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 3D tensor handling
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == batch_size * seq_len * hidden_size

    def test_gelu_different_dtypes(self):
        """Test Gelu with different data types"""
        dtypes = ['float16', 'float32']
        batch_size, seq_len, hidden_size = 2, 8, 64

        for dtype in dtypes:
            inT, outT, input_names = create_gelu_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size, dtype=dtype
            )

            op_info = {
                'name': f'test_gelu_{dtype}',
                'optype': 'Gelu',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = GeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify dtype preservation
            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats is not None

    def test_gelu_large_tensor(self):
        """Test Gelu with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 8, 64, 1024

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_gelu_large',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify large tensor handling
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

        # Verify instruction counts scale correctly
        assert perf_stats['instrs']['mul'] > 0
        assert perf_stats['instrs']['add'] > 0

    def test_gelu_edge_cases(self):
        """Test Gelu with edge case values"""
        # Test with various tensor sizes
        test_cases = [
            [1],           # Single element
            [100],         # Large 1D
            [4, 32],       # 2D small
            [2, 64, 32],   # 3D medium
        ]

        for shape in test_cases:
            input_tensor = F._from_shape(f'input_{shape}', shape, np_dtype=np.dtype('float32'))
            output_tensor = F._from_shape(f'output_{shape}', shape, np_dtype=np.dtype('float32'))

            inT = [input_tensor]
            outT = [output_tensor]

            op_info = {
                'name': f'test_gelu_edge_{shape}',
                'optype': 'Gelu',
                'inList': [f'input_{shape}'],
                'outList': [f'output_{shape}'],
                'attrs': {},
            }
            op = GeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shape preservation and element count
            assert outT[0].shape == shape
            expected_elements = np.prod(shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements

    def test_gelu_memory_usage(self):
        """Test Gelu memory usage calculation"""
        batch_size, seq_len, hidden_size = 4, 16, 256

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_gelu_memory',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify memory calculations
        total_elements = batch_size * seq_len * hidden_size
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_bytes
        assert perf_stats['outBytes'] == expected_bytes

    def test_gelu_no_attributes(self):
        """Test that Gelu operation correctly handles no attributes (uses default erf variant)"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # Test with empty attributes (should use default erf variant)
        op_info = {
            'name': 'test_gelu_no_attrs',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = GeluOp(op_info)
        assert op is not None
        assert op.approximate is None  # Should be None, triggering default erf variant

        # Test without attrs key
        op_info_no_attrs = {
            'name': 'test_gelu_no_attrs_key',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
        }
        op2 = GeluOp(op_info_no_attrs)
        assert op2 is not None
        assert op2.approximate is None

    def test_gelu_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with wrong number of inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'Gelu',
            'inList': ['input1', 'input2'],  # Too many inputs
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            GeluOp(op_info)

        # Test with wrong number of outputs
        op_info = {
            'name': 'test_invalid_outputs',
            'optype': 'Gelu',
            'inList': ['input'],
            'outList': ['output1', 'output2'],  # Too many outputs
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            GeluOp(op_info)

    def test_gelu_invalid_tensor_shape(self):
        """Test error handling for invalid tensor shapes"""
        # Test with 0D tensor (scalar)
        input_tensor = F._from_shape('input_scalar', [], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_scalar', [], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_shape',
            'optype': 'Gelu',
            'inList': ['input_scalar'],
            'outList': ['output_scalar'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        with pytest.raises(AssertionError, match="Gelu input must be at least 1D"):
            op.get_perf_counts(inT, outT)

    def test_gelu_performance_consistency(self):
        """Test that performance calculations are consistent"""
        batch_size, seq_len, hidden_size = 3, 12, 96

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = GeluOp(op_info)

        # Call get_perf_counts multiple times to ensure consistency
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify all expected instruction types are present
        expected_instrs = ['mul', 'add', 'exp']  # erf is mapped to exp
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_gelu_tanh_performance_consistency(self):
        """Test that tanh variant performance calculations are consistent"""
        batch_size, seq_len, hidden_size = 3, 12, 96

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_tanh_performance_consistency',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'approximate': 'tanh'},
        }
        op = GeluOp(op_info)

        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify tanh variant has correct instruction types
        expected_instrs = ['mul', 'add', 'tanh']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

        # Tanh variant should have more multiplications than default
        assert perf_stats1['instrs']['mul'] > perf_stats1['instrs']['add']

    def test_gelu_variant_differences(self):
        """Test that both variants produce different instruction counts"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # Test default (erf) variant
        op_info_default = {
            'name': 'test_default_variant',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op_default = GeluOp(op_info_default)
        perf_stats_default = op_default.get_perf_counts(inT, outT)

        # Test tanh variant
        op_info_tanh = {
            'name': 'test_tanh_variant',
            'optype': 'Gelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'approximate': 'tanh'},
        }
        op_tanh = GeluOp(op_info_tanh)
        perf_stats_tanh = op_tanh.get_perf_counts(inT, outT)

        # Variants should have different instruction counts
        assert perf_stats_default['instrs']['mul'] != perf_stats_tanh['instrs']['mul']
        assert perf_stats_default['instrs']['add'] != perf_stats_tanh['instrs']['add']

        # Default should have exp (erf mapped to exp), tanh should have tanh
        assert 'exp' in perf_stats_default['instrs']  # erf is mapped to exp
        assert 'exp' not in perf_stats_tanh['instrs']
        assert 'tanh' in perf_stats_tanh['instrs']
        assert 'tanh' not in perf_stats_default['instrs']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
