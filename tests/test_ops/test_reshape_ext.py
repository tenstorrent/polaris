"""
Comprehensive test suite for ReshapeExt operation implementation.

This module tests the ONNX 1.20.0 ReshapeExt operation with extensions, which provides
enhanced reshape functionality with advanced shape inference, memory optimization,
and broadcasting capabilities.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, ReshapeExtOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_reshape_ext_test_tensors(input_shape, target_shape, dtype='float32', include_mask=False):
    """
    Helper function to create test tensors for ReshapeExt operation.

    Args:
        input_shape: Shape of the input tensor to reshape
        target_shape: Target shape for reshaping
        dtype: Data type for tensors
        include_mask: Whether to include an optional mask tensor

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

    # Optional mask tensor for broadcasting
    if include_mask:
        mask_shape = [len(target_shape)]
        mask_tensor = F._from_shape('mask', mask_shape, np_dtype=np.dtype('bool'))
        input_tensors.append(mask_tensor)
        input_names.append('mask')

    # Create output tensor
    output_tensor = F._from_shape('output', target_shape, np_dtype=np.dtype(dtype))

    output_tensors = [output_tensor]
    output_names = ['output']

    return input_tensors, output_tensors, input_names


class TestReshapeExt:
    """Test class for ReshapeExt operation"""

    def test_factory_integration(self):
        """Test that ReshapeExt is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('ReshapeExt')
        assert opcls == ReshapeExtOp

    def test_basic_reshape_ext_functionality(self):
        """Test basic ReshapeExt functionality with 2D to 1D"""
        input_shape = [6, 4]
        target_shape = [24]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_basic',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes
        assert outT[0].shape == target_shape
        assert outT[0].dtype == inT[0].dtype

        # Verify element count preservation
        input_elements = np.prod(input_shape)
        output_elements = np.prod(target_shape)
        assert input_elements == output_elements

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mov'] == input_elements

    def test_reshape_ext_1d_to_2d(self):
        """Test ReshapeExt from 1D to 2D"""
        input_shape = [12]
        target_shape = [3, 4]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_1d_to_2d',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 12
        assert perf_stats['outElems'] == 12

    def test_reshape_ext_2d_to_3d(self):
        """Test ReshapeExt from 2D to 3D"""
        input_shape = [4, 6]
        target_shape = [2, 2, 6]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_2d_to_3d',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 24
        assert perf_stats['outElems'] == 24

    def test_reshape_ext_3d_to_2d(self):
        """Test ReshapeExt from 3D to 2D"""
        input_shape = [2, 3, 4]
        target_shape = [6, 4]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_3d_to_2d',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 24
        assert perf_stats['outElems'] == 24

    def test_reshape_ext_with_minus_one(self):
        """Test ReshapeExt with -1 dimension inference"""
        input_shape = [2, 3, 4]
        target_shape = [2, -1, 2]  # -1 should be inferred as 6

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_minus_one',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        expected_shape = [2, 6, 2]
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == 24
        assert perf_stats['outElems'] == 24

    def test_reshape_ext_with_zeros(self):
        """Test ReshapeExt with zero dimensions (copy from input)"""
        input_shape = [3, 4, 5]  # 60 elements
        target_shape = [0, 0, -1]  # Zeros copy from input, -1 infers the rest

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_zeros',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'allowzero': 0},  # allowzero=0 means copy from input
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        expected_shape = [3, 4, 5]  # Zeros replaced with input dimensions, -1 becomes 5
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == 60
        assert perf_stats['outElems'] == 60  # Same number of elements

    def test_reshape_ext_conservative_strategy(self):
        """Test ReshapeExt with conservative inference strategy"""
        input_shape = [8, 6]
        target_shape = [4, -1]  # -1 should be inferred as 12 (8*6/4=12)

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_conservative',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'infer_strategy': 'conservative'},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        expected_shape = [4, 12]
        assert outT[0].shape == expected_shape

    def test_reshape_ext_aggressive_strategy(self):
        """Test ReshapeExt with aggressive inference strategy"""
        input_shape = [10, 5]
        target_shape = [3, -1]  # -1 should be inferred as 8 (round(50/3)=17, but 17*3=51≠50, should fail)

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_aggressive',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'infer_strategy': 'aggressive'},
        }
        op = ReshapeExtOp(op_info)

        # Should fail because 50 / 3 = 16.666, rounded to 17, but 17 * 3 = 51 ≠ 50
        with pytest.raises(ValueError, match="Aggressive inference failed"):
            op.get_perf_counts(inT, outT)

    def test_reshape_ext_optimize_layout(self):
        """Test ReshapeExt with layout optimization"""
        input_shape = [4, 6]
        target_shape = [24]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_optimize',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'optimize_layout': True},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Layout optimization should increase mov count slightly
        expected_mov = int(24 * 1.1)  # 24 * 1.1 = 26.4, should be 26
        assert perf_stats['instrs']['mov'] == expected_mov

    def test_reshape_ext_different_dtypes(self):
        """Test ReshapeExt with different data types"""
        dtypes = ['float16', 'float32', 'int8']
        input_shape = [4, 8]
        target_shape = [32]

        for dtype in dtypes:
            inT, outT, input_names = create_reshape_ext_test_tensors(
                input_shape=input_shape, target_shape=target_shape, dtype=dtype
            )

            op_info = {
                'name': f'test_reshape_ext_{dtype}',
                'optype': 'ReshapeExt',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = ReshapeExtOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats['inElems'] == 32
            assert perf_stats['outElems'] == 32

    def test_reshape_ext_large_tensor(self):
        """Test ReshapeExt with large tensor"""
        input_shape = [16, 32, 64]
        target_shape = [32768]  # 16 * 32 * 64 = 32768

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_reshape_ext_large',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats['inElems'] == 32768
        assert perf_stats['outElems'] == 32768

    def test_reshape_ext_with_mask(self):
        """Test ReshapeExt with optional mask tensor"""
        input_shape = [6, 4]
        target_shape = [24]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape, include_mask=True
        )

        op_info = {
            'name': 'test_reshape_ext_mask',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'allow_broadcast': True},
        }
        op = ReshapeExtOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == target_shape
        assert perf_stats is not None

    def test_reshape_ext_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with insufficient inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'ReshapeExt',
            'inList': ['input'],  # Only 1 input, need at least 2
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            ReshapeExtOp(op_info)

    def test_reshape_ext_invalid_shape_tensor_dtype(self):
        """Test error handling for invalid shape tensor dtype"""
        input_shape = [4, 6]
        target_shape = [24]

        # Create input with wrong shape tensor dtype
        input_tensor = F._from_shape('input', input_shape, np_dtype=np.dtype('float32'))
        shape_tensor = F._from_shape('shape', [1], np_dtype=np.dtype('float32'))  # Wrong dtype
        output_tensor = F._from_shape('output', target_shape, np_dtype=np.dtype('float32'))

        inT = [input_tensor, shape_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_dtype',
            'optype': 'ReshapeExt',
            'inList': ['input', 'shape'],
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        with pytest.raises(AssertionError, match="Shape tensor must be int32 or int64"):
            op.get_perf_counts(inT, outT)

    def test_reshape_ext_multiple_minus_one(self):
        """Test error handling for multiple -1 dimensions"""
        input_shape = [24]
        target_shape = [-1, -1, 2]  # Multiple -1, should fail

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_multiple_minus_one',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        with pytest.raises(AssertionError, match="Only one -1 is allowed"):
            op.get_perf_counts(inT, outT)

    def test_reshape_ext_invalid_size(self):
        """Test error handling for size mismatch"""
        input_shape = [7]  # 7 elements
        target_shape = [3, 3]  # 9 elements, mismatch

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_invalid_size',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        with pytest.raises(AssertionError, match="Input size.*!= output size"):
            op.get_perf_counts(inT, outT)

    def test_reshape_ext_performance_consistency(self):
        """Test that performance calculations are consistent"""
        input_shape = [8, 6]
        target_shape = [48]

        inT, outT, input_names = create_reshape_ext_test_tensors(
            input_shape=input_shape, target_shape=target_shape
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'ReshapeExt',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = ReshapeExtOp(op_info)

        # Call get_perf_counts multiple times
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify expected instruction types
        expected_instrs = ['mov', 'cmp', 'add']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_reshape_ext_edge_cases(self):
        """Test ReshapeExt with various edge cases"""
        test_cases = [
            ([1], [1]),                    # Single element
            ([100], [100]),                # Large 1D
            ([1, 100], [100]),             # 2D to 1D
            ([2, 2, 25], [2, 2, 25]),      # Same shape
            ([4, 6], [2, 12]),             # 2D to 2D
        ]

        for input_shape, target_shape in test_cases:
            inT, outT, input_names = create_reshape_ext_test_tensors(
                input_shape=input_shape, target_shape=target_shape
            )

            op_info = {
                'name': f'test_edge_{input_shape}_to_{target_shape}',
                'optype': 'ReshapeExt',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = ReshapeExtOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].shape == target_shape
            expected_elements = np.prod(input_shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
