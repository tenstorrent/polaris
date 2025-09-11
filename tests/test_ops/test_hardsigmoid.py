#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for HardSigmoid activation function implementation.

This module tests the ONNX 1.20.0 HardSigmoid operation, which implements
the hardware-friendly sigmoid approximation: max(0, min(1, alpha * x + beta)).
HardSigmoid is optimized for efficient inference on resource-constrained devices.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, HardSigmoidOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_hardsigmoid_test_tensors(batch_size=2, seq_len=8, hidden_size=64, dtype='float32'):
    """
    Helper function to create test tensors for HardSigmoid operation.

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


def reference_hardsigmoid(x, alpha=0.2, beta=0.5):
    """
    Reference implementation of HardSigmoid activation function.
    HardSigmoid(x) = max(0, min(1, alpha * x + beta))
    """
    return np.maximum(0, np.minimum(1, alpha * x + beta))


class TestHardSigmoid:
    """Test class for HardSigmoid operation"""

    def test_factory_integration(self):
        """Test that HardSigmoid is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('HardSigmoid')
        assert opcls == HardSigmoidOp

    def test_basic_hardsigmoid_functionality(self):
        """Test basic HardSigmoid activation functionality"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_hardsigmoid_basic',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

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
        assert perf_stats['instrs']['cmp'] > 0  # Should have comparison operations
        assert perf_stats['instrs']['min'] > 0  # Should have min operations
        assert perf_stats['instrs']['max'] > 0  # Should have max operations

    def test_hardsigmoid_1d_tensor(self):
        """Test HardSigmoid with 1D tensor"""
        hidden_size = 128

        # Create 1D input tensor
        input_tensor = F._from_shape('input_1d', [hidden_size], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_1d', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_hardsigmoid_1d',
            'optype': 'HardSigmoid',
            'inList': ['input_1d'],
            'outList': ['output_1d'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 1D tensor handling
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size
        assert perf_stats['outElems'] == hidden_size

    def test_hardsigmoid_2d_tensor(self):
        """Test HardSigmoid with 2D tensor (batch_size, features)"""
        batch_size, features = 4, 256

        # Create 2D input tensor
        input_tensor = F._from_shape('input_2d', [batch_size, features], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_2d', [batch_size, features], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_hardsigmoid_2d',
            'optype': 'HardSigmoid',
            'inList': ['input_2d'],
            'outList': ['output_2d'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 2D tensor handling
        assert outT[0].shape == [batch_size, features]
        assert perf_stats['inElems'] == batch_size * features
        assert perf_stats['outElems'] == batch_size * features

    def test_hardsigmoid_3d_tensor(self):
        """Test HardSigmoid with 3D tensor (batch_size, seq_len, hidden_size)"""
        batch_size, seq_len, hidden_size = 2, 16, 128

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_hardsigmoid_3d',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 3D tensor handling
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == batch_size * seq_len * hidden_size

    def test_hardsigmoid_different_dtypes(self):
        """Test HardSigmoid with different data types"""
        dtypes = ['float16', 'float32']
        batch_size, seq_len, hidden_size = 2, 8, 64

        for dtype in dtypes:
            inT, outT, input_names = create_hardsigmoid_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size, dtype=dtype
            )

            op_info = {
                'name': f'test_hardsigmoid_{dtype}',
                'optype': 'HardSigmoid',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = HardSigmoidOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify dtype preservation
            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats is not None

    def test_hardsigmoid_large_tensor(self):
        """Test HardSigmoid with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 8, 64, 1024

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_hardsigmoid_large',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify large tensor handling
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

        # Verify instruction counts scale correctly
        assert perf_stats['instrs']['mul'] == total_elements
        assert perf_stats['instrs']['add'] == total_elements
        assert perf_stats['instrs']['cmp'] == 2 * total_elements  # Two comparisons per element
        assert perf_stats['instrs']['min'] == total_elements
        assert perf_stats['instrs']['max'] == total_elements

    def test_hardsigmoid_edge_cases(self):
        """Test HardSigmoid with edge case values"""
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
                'name': f'test_hardsigmoid_edge_{shape}',
                'optype': 'HardSigmoid',
                'inList': [f'input_{shape}'],
                'outList': [f'output_{shape}'],
                'attrs': {},
            }
            op = HardSigmoidOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shape preservation and element count
            assert outT[0].shape == shape
            expected_elements = np.prod(shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements

    def test_hardsigmoid_memory_usage(self):
        """Test HardSigmoid memory usage calculation"""
        batch_size, seq_len, hidden_size = 4, 16, 256

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_hardsigmoid_memory',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify memory calculations
        total_elements = batch_size * seq_len * hidden_size
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_bytes
        assert perf_stats['outBytes'] == expected_bytes

    def test_hardsigmoid_no_attributes(self):
        """Test that HardSigmoid operation correctly handles no attributes"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # Test with empty attributes
        op_info = {
            'name': 'test_hardsigmoid_no_attrs',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)
        assert op is not None

        # Test without attrs key
        op_info_no_attrs = {
            'name': 'test_hardsigmoid_no_attrs_key',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
        }
        op2 = HardSigmoidOp(op_info_no_attrs)
        assert op2 is not None

    def test_hardsigmoid_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with wrong number of inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'HardSigmoid',
            'inList': ['input1', 'input2'],  # Too many inputs
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            HardSigmoidOp(op_info)

        # Test with wrong number of outputs
        op_info = {
            'name': 'test_invalid_outputs',
            'optype': 'HardSigmoid',
            'inList': ['input'],
            'outList': ['output1', 'output2'],  # Too many outputs
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            HardSigmoidOp(op_info)

    def test_hardsigmoid_invalid_tensor_shape(self):
        """Test error handling for invalid tensor shapes"""
        # Test with 0D tensor (scalar)
        input_tensor = F._from_shape('input_scalar', [], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_scalar', [], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_shape',
            'optype': 'HardSigmoid',
            'inList': ['input_scalar'],
            'outList': ['output_scalar'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        with pytest.raises(AssertionError, match="HardSigmoid input must be at least 1D"):
            op.get_perf_counts(inT, outT)

    def test_hardsigmoid_performance_consistency(self):
        """Test that performance calculations are consistent"""
        batch_size, seq_len, hidden_size = 3, 12, 96

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        # Call get_perf_counts multiple times to ensure consistency
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify all expected instruction types are present
        expected_instrs = ['mul', 'add', 'cmp', 'min', 'max']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_hardsigmoid_hardware_efficiency(self):
        """Test that HardSigmoid exhibits hardware-efficient properties"""
        batch_size, seq_len, hidden_size = 4, 8, 64

        inT, outT, input_names = create_hardsigmoid_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_hardware_efficient',
            'optype': 'HardSigmoid',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = HardSigmoidOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify that HardSigmoid uses hardware-efficient operations
        # Should have arithmetic operations (mul, add) and comparisons (cmp, min, max)
        assert perf_stats['instrs']['mul'] > 0  # Linear scaling
        assert perf_stats['instrs']['add'] > 0  # Bias addition
        assert perf_stats['instrs']['cmp'] > 0  # Clamping bounds
        assert perf_stats['instrs']['min'] > 0  # Upper bound clamping
        assert perf_stats['instrs']['max'] > 0  # Lower bound clamping

        # Verify instruction balance - HardSigmoid should be ALU-heavy for efficiency
        total_alu_ops = perf_stats['instrs']['mul'] + perf_stats['instrs']['add'] + perf_stats['instrs']['cmp'] + perf_stats['instrs']['min'] + perf_stats['instrs']['max']
        total_elements = batch_size * seq_len * hidden_size

        # Should have exactly 6 operations per element (mul + add + 2*cmp + min + max)
        assert total_alu_ops == 6 * total_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
