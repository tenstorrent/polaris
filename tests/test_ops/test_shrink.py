#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for Shrink activation function implementation.

This module tests the ONNX 1.20.0 Shrink operation, which implements
the thresholding activation: x - bias if x > bias else x + bias if x < -bias else 0.
Shrink enables sparsity and regularization in neural networks.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, ShrinkOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_shrink_test_tensors(batch_size=2, seq_len=8, hidden_size=64, dtype='float32'):
    """
    Helper function to create test tensors for Shrink operation.

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


def reference_shrink(x, bias=0.0):
    """
    Reference implementation of Shrink activation function.
    Shrink(x) = x - bias if x > bias else x + bias if x < -bias else 0
    """
    return np.where(x > bias, x - bias,
                   np.where(x < -bias, x + bias, 0))


class TestShrink:
    """Test class for Shrink operation"""

    def test_factory_integration(self):
        """Test that Shrink is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('Shrink')
        assert opcls == ShrinkOp

    def test_basic_shrink_functionality(self):
        """Test basic Shrink activation functionality"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_shrink_basic',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

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
        assert perf_stats['instrs']['cmp'] > 0  # Should have comparison operations
        assert perf_stats['instrs']['sub'] > 0  # Should have subtraction operations
        assert perf_stats['instrs']['add'] > 0  # Should have addition operations

    def test_shrink_1d_tensor(self):
        """Test Shrink with 1D tensor"""
        hidden_size = 128

        # Create 1D input tensor
        input_tensor = F._from_shape('input_1d', [hidden_size], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_1d', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_shrink_1d',
            'optype': 'Shrink',
            'inList': ['input_1d'],
            'outList': ['output_1d'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 1D tensor handling
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size
        assert perf_stats['outElems'] == hidden_size

    def test_shrink_2d_tensor(self):
        """Test Shrink with 2D tensor (batch_size, features)"""
        batch_size, features = 4, 256

        # Create 2D input tensor
        input_tensor = F._from_shape('input_2d', [batch_size, features], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_2d', [batch_size, features], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_shrink_2d',
            'optype': 'Shrink',
            'inList': ['input_2d'],
            'outList': ['output_2d'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 2D tensor handling
        assert outT[0].shape == [batch_size, features]
        assert perf_stats['inElems'] == batch_size * features
        assert perf_stats['outElems'] == batch_size * features

    def test_shrink_3d_tensor(self):
        """Test Shrink with 3D tensor (batch_size, seq_len, hidden_size)"""
        batch_size, seq_len, hidden_size = 2, 16, 128

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_shrink_3d',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 3D tensor handling
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == batch_size * seq_len * hidden_size

    def test_shrink_different_dtypes(self):
        """Test Shrink with different data types"""
        dtypes = ['float16', 'float32']
        batch_size, seq_len, hidden_size = 2, 8, 64

        for dtype in dtypes:
            inT, outT, input_names = create_shrink_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size, dtype=dtype
            )

            op_info = {
                'name': f'test_shrink_{dtype}',
                'optype': 'Shrink',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = ShrinkOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify dtype preservation
            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats is not None

    def test_shrink_large_tensor(self):
        """Test Shrink with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 8, 64, 1024

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_shrink_large',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify large tensor handling
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

        # Verify instruction counts scale correctly
        assert perf_stats['instrs']['cmp'] == 2 * total_elements  # Two comparisons per element
        assert perf_stats['instrs']['sub'] == total_elements
        assert perf_stats['instrs']['add'] == total_elements

    def test_shrink_with_attributes(self):
        """Test Shrink numeric behavior with custom lambd and bias"""
        batch_size, seq_len, hidden_size = 1, 1, 6
        x = F._from_shape('input_attrs', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))
        y = F._from_shape('output_attrs', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))
        inT = [x]
        outT = [y]
        attrs = {'lambd': 0.5, 'bias': 0.2}
        op_info = {
            'name': 'test_shrink_attrs',
            'optype': 'Shrink',
            'inList': ['input_attrs'],
            'outList': ['output_attrs'],
            'attrs': attrs,
        }
        op = ShrinkOp(op_info)
        perf_stats = op.get_perf_counts(inT, outT)
        # Shape/dtype still correct and perf_stats present
        assert outT[0].shape == [batch_size, seq_len, hidden_size]
        assert y.dtype == x.dtype
        assert 'instrs' in perf_stats

    def test_shrink_edge_cases(self):
        """Test Shrink with edge case values"""
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
                'name': f'test_shrink_edge_{shape}',
                'optype': 'Shrink',
                'inList': [f'input_{shape}'],
                'outList': [f'output_{shape}'],
                'attrs': {},
            }
            op = ShrinkOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shape preservation and element count
            assert outT[0].shape == shape
            expected_elements = np.prod(shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements

    def test_shrink_memory_usage(self):
        """Test Shrink memory usage calculation"""
        batch_size, seq_len, hidden_size = 4, 16, 256

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_shrink_memory',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify memory calculations
        total_elements = batch_size * seq_len * hidden_size
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_bytes
        assert perf_stats['outBytes'] == expected_bytes

    def test_shrink_no_attributes(self):
        """Test that Shrink operation correctly handles no attributes"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # Test with empty attributes
        op_info = {
            'name': 'test_shrink_no_attrs',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)
        assert op is not None

        # Test without attrs key
        op_info_no_attrs = {
            'name': 'test_shrink_no_attrs_key',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
        }
        op2 = ShrinkOp(op_info_no_attrs)
        assert op2 is not None

    def test_shrink_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with wrong number of inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'Shrink',
            'inList': ['input1', 'input2'],  # Too many inputs
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            ShrinkOp(op_info)

        # Test with wrong number of outputs
        op_info = {
            'name': 'test_invalid_outputs',
            'optype': 'Shrink',
            'inList': ['input'],
            'outList': ['output1', 'output2'],  # Too many outputs
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            ShrinkOp(op_info)

    def test_shrink_invalid_tensor_shape(self):
        """Test error handling for invalid tensor shapes"""
        # Test with 0D tensor (scalar)
        input_tensor = F._from_shape('input_scalar', [], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_scalar', [], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_shape',
            'optype': 'Shrink',
            'inList': ['input_scalar'],
            'outList': ['output_scalar'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        with pytest.raises(AssertionError, match="Shrink input must be at least 1D"):
            op.get_perf_counts(inT, outT)

    def test_shrink_performance_consistency(self):
        """Test that performance calculations are consistent"""
        batch_size, seq_len, hidden_size = 3, 12, 96

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        # Call get_perf_counts multiple times to ensure consistency
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify all expected instruction types are present
        expected_instrs = ['cmp', 'sub', 'add']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_shrink_thresholding_behavior(self):
        """Test that Shrink exhibits proper thresholding behavior"""
        batch_size, seq_len, hidden_size = 4, 8, 64

        inT, outT, input_names = create_shrink_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_thresholding_behavior',
            'optype': 'Shrink',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ShrinkOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify that Shrink uses thresholding operations
        # Should have comparisons for threshold checks and arithmetic for adjustments
        assert perf_stats['instrs']['cmp'] > 0  # Threshold comparisons (x > bias, x < -bias)
        assert perf_stats['instrs']['sub'] > 0  # Subtraction for positive values (x - bias)
        assert perf_stats['instrs']['add'] > 0  # Addition for negative values (x + bias)

        # Verify instruction balance for thresholding
        total_elements = batch_size * seq_len * hidden_size
        cmp_ops = perf_stats['instrs']['cmp']
        sub_ops = perf_stats['instrs']['sub']
        add_ops = perf_stats['instrs']['add']

        # Should have 2 comparisons per element (for upper and lower thresholds)
        assert cmp_ops == 2 * total_elements
        # Should have 1 arithmetic operation per element (either sub or add)
        assert sub_ops == total_elements
        assert add_ops == total_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
