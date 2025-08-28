"""
Comprehensive test suite for ThresholdedReLU activation function implementation.

This module tests the ONNX 1.20.0 ThresholdedReLU operation, which implements
the thresholded rectified linear unit: x if x > alpha else 0.
ThresholdedReLU enables sparse activation and feature selection in neural networks.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, ThresholdedReluOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_thresholdedrelu_test_tensors(input_shape, dtype='float32'):
    """
    Helper function to create test tensors for ThresholdedReLU operation.

    Args:
        input_shape: Shape of the input tensor
        dtype: Data type for tensors

    Returns:
        Tuple of (input_tensors, output_tensors, input_names)
    """
    # Create input tensor
    input_tensor = F._from_shape('input', input_shape, np_dtype=np.dtype(dtype))

    input_tensors = [input_tensor]
    input_names = ['input']

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', input_shape, np_dtype=np.dtype(dtype))

    output_tensors = [output_tensor]
    output_names = ['output']

    return input_tensors, output_tensors, input_names


def reference_thresholdedrelu(x, alpha=1.0):
    """
    Reference implementation of ThresholdedReLU activation function.
    ThresholdedReLU(x) = x if x > alpha else 0
    """
    return np.where(x > alpha, x, 0)


class TestThresholdedRelu:
    """Test class for ThresholdedReLU operation"""

    def test_factory_integration(self):
        """Test that ThresholdedReLU is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('ThresholdedRelu')
        assert opcls == ThresholdedReluOp

    def test_thresholdedrelu_default_alpha(self):
        """Test ThresholdedReLU with default alpha=1.0"""
        input_shape = [4, 6]

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        op_info = {
            'name': 'test_thresholdedrelu_default',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},  # Default alpha=1.0
        }
        op = ThresholdedReluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shapes and types
        assert outT[0].shape == input_shape
        assert outT[0].dtype == inT[0].dtype

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['cmp'] > 0  # Should have comparison operations
        assert perf_stats['instrs']['mov'] > 0  # Should have move operations

    def test_thresholdedrelu_custom_alpha(self):
        """Test ThresholdedReLU with custom alpha values"""
        input_shape = [3, 5]
        alpha_values = [0.0, 0.5, 2.0, 10.0]

        for alpha in alpha_values:
            inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

            op_info = {
                'name': f'test_thresholdedrelu_alpha_{alpha}',
                'optype': 'ThresholdedRelu',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {'alpha': alpha},
            }
            op = ThresholdedReluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify alpha is set correctly
            assert op.alpha == alpha

            # Verify output properties
            assert outT[0].shape == input_shape
            assert perf_stats['inElems'] == np.prod(input_shape)

    def test_thresholdedrelu_different_shapes(self):
        """Test ThresholdedReLU with different tensor shapes"""
        test_shapes = [
            [10],           # 1D
            [4, 6],         # 2D
            [2, 3, 5],      # 3D
            [2, 3, 4, 5],   # 4D
        ]

        for shape in test_shapes:
            inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=shape)

            op_info = {
                'name': f'test_thresholdedrelu_shape_{shape}',
                'optype': 'ThresholdedRelu',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {'alpha': 0.5},
            }
            op = ThresholdedReluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].shape == shape
            expected_elements = np.prod(shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements

    def test_thresholdedrelu_different_dtypes(self):
        """Test ThresholdedReLU with different data types"""
        dtypes = ['float16', 'float32', 'float64']
        input_shape = [4, 6]

        for dtype in dtypes:
            inT, outT, input_names = create_thresholdedrelu_test_tensors(
                input_shape=input_shape, dtype=dtype
            )

            op_info = {
                'name': f'test_thresholdedrelu_dtype_{dtype}',
                'optype': 'ThresholdedRelu',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {'alpha': 1.0},
            }
            op = ThresholdedReluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats is not None

    def test_thresholdedrelu_large_tensor(self):
        """Test ThresholdedReLU with large tensor"""
        input_shape = [64, 128, 256]  # Large 3D tensor

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        op_info = {
            'name': 'test_thresholdedrelu_large',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 0.1},
        }
        op = ThresholdedReluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = 64 * 128 * 256
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

        # Verify instruction counts scale correctly
        assert perf_stats['instrs']['cmp'] == total_elements
        # For alpha != 0, we have additional move operations
        assert perf_stats['instrs']['mov'] >= total_elements

    def test_thresholdedrelu_edge_cases(self):
        """Test ThresholdedReLU with edge cases"""
        # Test with single element
        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=[1])

        op_info = {
            'name': 'test_thresholdedrelu_single',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 0.0},
        }
        op = ThresholdedReluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        assert outT[0].shape == [1]
        assert perf_stats['inElems'] == 1
        assert perf_stats['outElems'] == 1

    def test_thresholdedrelu_memory_usage(self):
        """Test ThresholdedReLU memory usage calculation"""
        input_shape = [8, 16, 32]

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        op_info = {
            'name': 'test_thresholdedrelu_memory',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 2.0},
        }
        op = ThresholdedReluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = 8 * 16 * 32
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_bytes
        assert perf_stats['outBytes'] == expected_bytes

    def test_thresholdedrelu_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with wrong number of inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'ThresholdedRelu',
            'inList': ['input1', 'input2'],  # Too many inputs
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            ThresholdedReluOp(op_info)

        # Test with wrong number of outputs
        op_info = {
            'name': 'test_invalid_outputs',
            'optype': 'ThresholdedRelu',
            'inList': ['input'],
            'outList': ['output1', 'output2'],  # Too many outputs
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            ThresholdedReluOp(op_info)

    def test_thresholdedrelu_invalid_alpha(self):
        """Test error handling for invalid alpha values"""
        input_shape = [4, 6]

        # Test negative alpha
        op_info = {
            'name': 'test_invalid_alpha_negative',
            'optype': 'ThresholdedRelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {'alpha': -1.0},
        }

        with pytest.raises(ValueError, match="alpha must be a non-negative number"):
            ThresholdedReluOp(op_info)

        # Test invalid alpha type
        op_info = {
            'name': 'test_invalid_alpha_type',
            'optype': 'ThresholdedRelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {'alpha': 'invalid'},
        }

        with pytest.raises(ValueError, match="alpha must be a non-negative number"):
            ThresholdedReluOp(op_info)

    def test_thresholdedrelu_invalid_tensor_shape(self):
        """Test error handling for invalid tensor shapes"""
        # Test with 0D tensor (scalar)
        input_tensor = F._from_shape('input_scalar', [], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_scalar', [], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_shape',
            'optype': 'ThresholdedRelu',
            'inList': ['input_scalar'],
            'outList': ['output_scalar'],
            'attrs': {},
        }
        op = ThresholdedReluOp(op_info)

        with pytest.raises(AssertionError, match="ThresholdedReLU input must be at least 1D"):
            op.get_perf_counts(inT, outT)

    def test_thresholdedrelu_performance_consistency(self):
        """Test that performance calculations are consistent"""
        input_shape = [6, 8]

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 1.5},
        }
        op = ThresholdedReluOp(op_info)

        # Call get_perf_counts multiple times
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify expected instruction types
        expected_instrs = ['cmp', 'mov']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_thresholdedrelu_sparse_activation(self):
        """Test that ThresholdedReLU exhibits proper sparse activation behavior"""
        input_shape = [10, 20]

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        # Test with alpha=0 (should behave like ReLU)
        op_info_zero = {
            'name': 'test_sparse_zero',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 0.0},
        }
        op_zero = ThresholdedReluOp(op_info_zero)
        perf_stats_zero = op_zero.get_perf_counts(inT, outT)

        # Test with high alpha (should be more sparse)
        op_info_high = {
            'name': 'test_sparse_high',
            'optype': 'ThresholdedRelu',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {'alpha': 5.0},
        }
        op_high = ThresholdedReluOp(op_info_high)
        perf_stats_high = op_high.get_perf_counts(inT, outT)

        # Both should have same element count
        assert perf_stats_zero['inElems'] == perf_stats_high['inElems']
        assert perf_stats_zero['outElems'] == perf_stats_high['outElems']

        # Both should have comparison operations
        assert perf_stats_zero['instrs']['cmp'] > 0
        assert perf_stats_high['instrs']['cmp'] > 0

    def test_thresholdedrelu_thresholding_behavior(self):
        """Test that ThresholdedReLU properly implements thresholding"""
        input_shape = [8, 12]

        inT, outT, input_names = create_thresholdedrelu_test_tensors(input_shape=input_shape)

        # Test various alpha values
        alpha_values = [0.0, 0.5, 1.0, 2.5]

        for alpha in alpha_values:
            op_info = {
                'name': f'test_thresholding_alpha_{alpha}',
                'optype': 'ThresholdedRelu',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {'alpha': alpha},
            }
            op = ThresholdedReluOp(op_info)
            perf_stats = op.get_perf_counts(inT, outT)

            # Verify operation completed successfully
            assert perf_stats is not None
            assert op.alpha == alpha

            # All thresholded operations should have comparison
            assert perf_stats['instrs']['cmp'] == np.prod(input_shape)

            # For alpha != 0, should have additional move operations
            if alpha != 0:
                assert perf_stats['instrs']['mov'] == 2 * np.prod(input_shape)
            else:
                assert perf_stats['instrs']['mov'] == np.prod(input_shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
