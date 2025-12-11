#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for Expand operation implementation.

This module tests the ONNX 1.20.0 Expand operation, which broadcasts
a tensor to a target shape following NumPy-style broadcasting rules.
"""

import numpy as np
import pytest
from typing import Any
from ttsim.ops.op import SimOp
from ttsim.ops.desc.registry import get_opdesc_registry
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_expand_test_tensors(input_shape, target_shape, dtype='float32'):
    """
    Helper function to create test tensors for Expand operation.

    Args:
        input_shape: Shape of the input tensor to expand
        target_shape: Target shape for expansion
        dtype: Data type for tensors

    Returns:
        Tuple of (input_tensors, output_tensors, input_names)
    """
    # Create input tensor
    input_tensor = F._from_shape('input', input_shape, np_dtype=np.dtype(dtype))

    input_tensors = [input_tensor]
    input_names = ['input']

    # Create shape tensor (int64 by default for shape specifications)
    shape_tensor = F._from_shape('shape', [len(target_shape)], np_dtype=np.dtype('int64'))
    # Set the shape data
    if hasattr(shape_tensor, 'data'):
        shape_tensor.data = np.array(target_shape, dtype=np.int64)

    input_tensors.append(shape_tensor)
    input_names.append('shape')

    # Create output tensor
    output_tensor = F._from_shape('output', target_shape, np_dtype=np.dtype(dtype))

    output_tensors = [output_tensor]
    output_names = ['output']

    return input_tensors, output_tensors, input_names


class TestExpand:
    """Test class for Expand operation"""

    def test_factory_integration(self):
        """Test that Expand is registered in the op mapping"""
        assert get_opdesc_registry().has_shape_inference_function('Expand')

    def test_basic_expand_functionality(self):
        """Test basic Expand functionality with broadcasting"""
        input_shape = [3, 1]
        target_shape = [3, 4]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_basic',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        assert outT[0].shape == target_shape
        assert outT[0].dtype == inT[0].dtype

        # Verify element counts (broadcasting increases elements)
        input_elements = np.prod(input_shape)  # 3 * 1 = 3
        output_elements = np.prod(target_shape)  # 3 * 4 = 12
        assert perf_stats['inElems'] == input_elements
        assert perf_stats['outElems'] == output_elements

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mov'] == output_elements

    def test_expand_scalar_to_tensor(self):
        """Test Expand from scalar to tensor"""
        input_shape: list[int] = []  # Scalar
        target_shape = [2, 3, 4]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_scalar',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 1  # Scalar
        assert perf_stats['outElems'] == 24  # 2 * 3 * 4

    def test_expand_1d_to_2d(self):
        """Test Expand from 1D to 2D"""
        input_shape = [5]
        target_shape = [3, 5]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_1d_to_2d',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 5
        assert perf_stats['outElems'] == 15  # 3 * 5

    def test_expand_2d_to_3d(self):
        """Test Expand from 2D to 3D"""
        input_shape = [2, 3]
        target_shape = [4, 2, 3]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_2d_to_3d',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 6
        assert perf_stats['outElems'] == 24  # 4 * 2 * 3

    def test_expand_compatible_dimensions(self):
        """Test Expand with compatible dimensions (no actual broadcasting needed)"""
        input_shape = [3, 4]
        target_shape = [3, 4]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_compatible',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 12
        assert perf_stats['outElems'] == 12  # Same size

    def test_expand_different_dtypes(self):
        """Test Expand with different data types"""
        dtypes = ['float16', 'float32', 'int8']
        input_shape = [2, 1]
        target_shape = [2, 5]

        for dtype in dtypes:
            inT, outT, input_names = create_expand_test_tensors(
                input_shape=input_shape, target_shape=target_shape, dtype=dtype
            )

            op_info = {
                'name': f'test_expand_{dtype}',
                'optype': 'Expand',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = SimOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats['inElems'] == 2
            assert perf_stats['outElems'] == 10  # 2 * 5

    def test_expand_large_tensor(self):
        """Test Expand with large tensor"""
        input_shape = [1, 256]
        target_shape = [64, 256]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_large',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 256
        assert perf_stats['outElems'] == 16384  # 64 * 256

    def test_expand_multiple_broadcast_dimensions(self):
        """Test Expand with multiple broadcasting dimensions"""
        input_shape = [1, 3, 1]
        target_shape = [4, 3, 5]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_multi_broadcast',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 3
        assert perf_stats['outElems'] == 60  # 4 * 3 * 5

    def test_expand_memory_usage(self):
        """Test Expand memory usage calculation"""
        input_shape = [1, 100]
        target_shape = [50, 100]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_expand_memory',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify memory calculations
        input_elements = np.prod(input_shape)
        output_elements = np.prod(target_shape)
        expected_input_bytes = input_elements * 4  # float32 = 4 bytes
        expected_output_bytes = output_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_input_bytes
        assert perf_stats['outBytes'] == expected_output_bytes

    def test_expand_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with insufficient inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'Expand',
            'inList': ['input'],  # Only 1 input, need 2
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)
        # Trigger shape inference with missing inputs
        with pytest.raises(Exception):
            op.get_perf_counts([], [make_tensor('output')])

    def test_expand_invalid_shape_tensor_dtype(self):
        """Test error handling for invalid shape tensor dtype"""
        input_shape = [2, 3]
        target_shape = [4, 3]

        # Create input with wrong shape tensor dtype
        input_tensor = F._from_shape('input', input_shape, np_dtype=np.dtype('float32'))
        shape_tensor = F._from_shape('shape', [2], np_dtype=np.dtype('float32'))  # Wrong dtype
        output_tensor = F._from_shape('output', target_shape, np_dtype=np.dtype('float32'))

        inT = [input_tensor, shape_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_dtype',
            'optype': 'Expand',
            'inList': ['input', 'shape'],
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)
        with pytest.raises(AssertionError, match="Shape tensor must be int32 or int64"):
            op.get_perf_counts(inT, outT)

    def test_expand_incompatible_broadcast(self):
        """Test error handling for incompatible broadcast dimensions"""
        input_shape = [3, 4]
        target_shape = [2, 4]  # Cannot broadcast 3 -> 2

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_incompatible_broadcast',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        with pytest.raises(ValueError, match="Incompatible broadcast dimensions"):
            op.get_perf_counts(inT, outT)

    def test_expand_insufficient_target_dimensions(self):
        """Test error handling for insufficient target dimensions"""
        input_shape = [2, 3, 4]  # 3D
        target_shape = [6, 4]     # Only 2D, cannot accommodate 3D input

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_insufficient_dims',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        with pytest.raises(AssertionError, match="Target shape.*must have at least as many dimensions"):
            op.get_perf_counts(inT, outT)

    def test_expand_performance_consistency(self):
        """Test that performance calculations are consistent"""
        input_shape = [1, 5]
        target_shape = [3, 5]

        inT, outT, input_names = create_expand_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'Expand',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SimOp(op_info)

        # Call get_perf_counts multiple times
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify expected instruction types (descriptor reports mov/cmp/add)
        for instr_type in ['mov', 'cmp', 'add']:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_expand_broadcasting_edge_cases(self):
        """Test Expand with various broadcasting edge cases"""
        test_cases = [
            ([1], [10]),                    # Single dimension broadcast
            ([1, 1], [5, 8]),              # 2D broadcast
            ([1, 1, 1], [2, 3, 7]),       # 3D broadcast
            ([3, 1, 2], [3, 4, 2]),       # Mixed broadcast and compatible
        ]

        for input_shape, target_shape in test_cases:
            inT, outT, input_names = create_expand_test_tensors(
                input_shape=input_shape, target_shape=target_shape
            )

            op_info = {
                'name': f'test_edge_{input_shape}_to_{target_shape}',
                'optype': 'Expand',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = SimOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].shape == target_shape
            expected_input_elements = np.prod(input_shape)
            expected_output_elements = np.prod(target_shape)
            assert perf_stats['inElems'] == expected_input_elements
            assert perf_stats['outElems'] == expected_output_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
