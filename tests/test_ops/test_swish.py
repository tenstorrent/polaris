"""
Comprehensive test suite for Swish activation function implementation.

This module tests the ONNX 1.20.0 Swish operation, which implements
the self-gated activation function: x * sigmoid(x).
Swish is known for its smooth, non-monotonic behavior and strong performance.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, SwishOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_swish_test_tensors(batch_size=2, seq_len=8, hidden_size=64, dtype='float32'):
    """
    Helper function to create test tensors for Swish operation.

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


def reference_swish(x):
    """
    Reference implementation of Swish activation function.
    Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    return x / (1 + np.exp(-x))


class TestSwish:
    """Test class for Swish operation"""

    def test_factory_integration(self):
        """Test that Swish is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('Swish')
        assert opcls == SwishOp

    def test_basic_swish_functionality(self):
        """Test basic Swish activation functionality"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_swish_basic',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

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
        assert perf_stats['instrs']['exp'] > 0  # Should have exponential operations
        assert perf_stats['instrs']['add'] > 0  # Should have addition operations
        assert perf_stats['instrs']['div'] > 0  # Should have division operations
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplication operations

    def test_swish_1d_tensor(self):
        """Test Swish with 1D tensor"""
        hidden_size = 128

        # Create 1D input tensor
        input_tensor = F._from_shape('input_1d', [hidden_size], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_1d', [hidden_size], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_swish_1d',
            'optype': 'Swish',
            'inList': ['input_1d'],
            'outList': ['output_1d'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 1D tensor handling
        assert outT[0].shape == [hidden_size]
        assert perf_stats['inElems'] == hidden_size
        assert perf_stats['outElems'] == hidden_size

    def test_swish_2d_tensor(self):
        """Test Swish with 2D tensor (batch_size, features)"""
        batch_size, features = 4, 256

        # Create 2D input tensor
        input_tensor = F._from_shape('input_2d', [batch_size, features], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_2d', [batch_size, features], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_swish_2d',
            'optype': 'Swish',
            'inList': ['input_2d'],
            'outList': ['output_2d'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 2D tensor handling
        assert outT[0].shape == [batch_size, features]
        assert perf_stats['inElems'] == batch_size * features
        assert perf_stats['outElems'] == batch_size * features

    def test_swish_3d_tensor(self):
        """Test Swish with 3D tensor (batch_size, seq_len, hidden_size)"""
        batch_size, seq_len, hidden_size = 2, 16, 128

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_swish_3d',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify 3D tensor handling
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape
        assert perf_stats['inElems'] == batch_size * seq_len * hidden_size
        assert perf_stats['outElems'] == batch_size * seq_len * hidden_size

    def test_swish_different_dtypes(self):
        """Test Swish with different data types"""
        dtypes = ['float16', 'float32']
        batch_size, seq_len, hidden_size = 2, 8, 64

        for dtype in dtypes:
            inT, outT, input_names = create_swish_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size, dtype=dtype
            )

            op_info = {
                'name': f'test_swish_{dtype}',
                'optype': 'Swish',
                'inList': input_names,
                'outList': ['output'],
                'attrs': {},
            }
            op = SwishOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify dtype preservation
            assert outT[0].dtype == np.dtype(dtype)
            assert perf_stats is not None

    def test_swish_large_tensor(self):
        """Test Swish with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 8, 64, 1024

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_swish_large',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify large tensor handling
        total_elements = batch_size * seq_len * hidden_size
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

        # Verify instruction counts scale correctly
        assert perf_stats['instrs']['exp'] == total_elements
        assert perf_stats['instrs']['add'] == total_elements
        assert perf_stats['instrs']['div'] == total_elements
        assert perf_stats['instrs']['mul'] == total_elements

    def test_swish_edge_cases(self):
        """Test Swish with edge case values"""
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
                'name': f'test_swish_edge_{shape}',
                'optype': 'Swish',
                'inList': [f'input_{shape}'],
                'outList': [f'output_{shape}'],
                'attrs': {},
            }
            op = SwishOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shape preservation and element count
            assert outT[0].shape == shape
            expected_elements = np.prod(shape)
            assert perf_stats['inElems'] == expected_elements
            assert perf_stats['outElems'] == expected_elements

    def test_swish_memory_usage(self):
        """Test Swish memory usage calculation"""
        batch_size, seq_len, hidden_size = 4, 16, 256

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_swish_memory',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify memory calculations
        total_elements = batch_size * seq_len * hidden_size
        expected_bytes = total_elements * 4  # float32 = 4 bytes

        assert perf_stats['inBytes'] == expected_bytes
        assert perf_stats['outBytes'] == expected_bytes

    def test_swish_no_attributes(self):
        """Test that Swish operation correctly handles no attributes"""
        batch_size, seq_len, hidden_size = 2, 8, 64

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        # Test with empty attributes
        op_info = {
            'name': 'test_swish_no_attrs',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)
        assert op is not None

        # Test without attrs key
        op_info_no_attrs = {
            'name': 'test_swish_no_attrs_key',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
        }
        op2 = SwishOp(op_info_no_attrs)
        assert op2 is not None

    def test_swish_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test with wrong number of inputs
        op_info = {
            'name': 'test_invalid_inputs',
            'optype': 'Swish',
            'inList': ['input1', 'input2'],  # Too many inputs
            'outList': ['output'],
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            SwishOp(op_info)

        # Test with wrong number of outputs
        op_info = {
            'name': 'test_invalid_outputs',
            'optype': 'Swish',
            'inList': ['input'],
            'outList': ['output1', 'output2'],  # Too many outputs
            'attrs': {},
        }

        with pytest.raises(AssertionError, match="should be in range"):
            SwishOp(op_info)

    def test_swish_invalid_tensor_shape(self):
        """Test error handling for invalid tensor shapes"""
        # Test with 0D tensor (scalar)
        input_tensor = F._from_shape('input_scalar', [], np_dtype=np.dtype('float32'))
        output_tensor = F._from_shape('output_scalar', [], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [output_tensor]

        op_info = {
            'name': 'test_invalid_shape',
            'optype': 'Swish',
            'inList': ['input_scalar'],
            'outList': ['output_scalar'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        with pytest.raises(AssertionError, match="Swish input must be at least 1D"):
            op.get_perf_counts(inT, outT)

    def test_swish_performance_consistency(self):
        """Test that performance calculations are consistent"""
        batch_size, seq_len, hidden_size = 3, 12, 96

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_performance_consistency',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        # Call get_perf_counts multiple times to ensure consistency
        perf_stats1 = op.get_perf_counts(inT, outT)
        perf_stats2 = op.get_perf_counts(inT, outT)

        # Results should be identical (cached)
        assert perf_stats1 == perf_stats2

        # Verify all expected instruction types are present
        expected_instrs = ['exp', 'add', 'div', 'mul']
        for instr_type in expected_instrs:
            assert instr_type in perf_stats1['instrs']
            assert perf_stats1['instrs'][instr_type] > 0

    def test_swish_self_gated_properties(self):
        """Test that Swish exhibits self-gated activation properties"""
        batch_size, seq_len, hidden_size = 4, 8, 64

        inT, outT, input_names = create_swish_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_self_gated',
            'optype': 'Swish',
            'inList': input_names,
            'outList': ['output'],
            'attrs': {},
        }
        op = SwishOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify that Swish uses self-gated operations
        # Should have exponential (for sigmoid), division, and multiplication
        assert perf_stats['instrs']['exp'] > 0  # Sigmoid computation
        assert perf_stats['instrs']['add'] > 0  # 1 + exp(-x)
        assert perf_stats['instrs']['div'] > 0  # x / (1 + exp(-x))
        assert perf_stats['instrs']['mul'] > 0  # Self-gating: x * sigmoid(x)

        # Verify instruction balance - Swish should have more FPU operations than ALU
        total_alu_ops = perf_stats['instrs']['add']
        total_fpu_ops = perf_stats['instrs']['exp'] + perf_stats['instrs']['div'] + perf_stats['instrs']['mul']

        # Swish should have more FPU operations than ALU operations due to sigmoid computation
        assert total_fpu_ops > total_alu_ops


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
