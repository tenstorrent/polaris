"""
Comprehensive test suite for FastGelu and FastGeluGrad operations implementation.

This module tests the ONNX FastGelu operations, which provide a fast approximation
of the Gaussian Error Linear Unit (GELU) activation function commonly used in
modern transformer architectures.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, FastGeluOp, FastGeluGradOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_fast_gelu_test_tensors(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for FastGelu operation.

    Args:
        batch_size: Batch size for input tensor
        seq_len: Sequence length
        hidden_size: Hidden dimension size

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create input tensor with various shapes for comprehensive testing
    input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor]

    # Create output tensor (same shape as input)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_fast_gelu_grad_test_tensors(batch_size=2, seq_len=8, hidden_size=512):
    """
    Helper function to create test tensors for FastGeluGrad operation.

    Args:
        batch_size: Batch size for input tensors
        seq_len: Sequence length
        hidden_size: Hidden dimension size

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # FastGeluGrad takes two inputs: x (original input) and dY (upstream gradient)
    x_tensor = F._from_shape('x', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))
    dY_tensor = F._from_shape('dY', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    input_tensors = [x_tensor, dY_tensor]

    # Create output tensor (same shape as inputs)
    output_tensor = F._from_shape('output', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestFastGelu:
    """Test class for FastGelu operation"""

    def test_factory_integration_fast_gelu(self):
        """Test that FastGelu is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('FastGelu')
        assert opcls == FastGeluOp

    def test_factory_integration_fast_gelu_grad(self):
        """Test that FastGeluGrad is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('FastGeluGrad')
        assert opcls == FastGeluGradOp

    def test_basic_fast_gelu(self):
        """Test basic FastGelu with standard configuration"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_fast_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_basic',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplications
        assert perf_stats['instrs']['add'] > 0  # Should have additions
        assert perf_stats['instrs']['tanh'] > 0  # Should have tanh operations
        assert perf_stats['instrs']['exp'] > 0  # Should have exp operations (for tanh)

    def test_fast_gelu_different_shapes(self):
        """Test FastGelu with different tensor shapes"""
        test_configs = [
            (1, 4, 128),     # Small configuration
            (4, 16, 1024),   # Large configuration
            (2, 32, 768),    # BERT-like configuration
            (8, 64, 4096),   # Large language model configuration
        ]

        for batch_size, seq_len, hidden_size in test_configs:
            inT, outT = create_fast_gelu_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
            )

            op_info = {
                'name': f'test_fast_gelu_{batch_size}_{seq_len}_{hidden_size}',
                'optype': 'FastGelu',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = FastGeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, hidden_size]
            assert outT[0].shape == expected_shape

            # Verify performance scales with tensor size
            total_elements = batch_size * seq_len * hidden_size
            assert perf_stats['inElems'] == total_elements
            assert perf_stats['outElems'] == total_elements
            assert perf_stats['instrs']['mul'] > total_elements  # Multiple operations per element

    def test_fast_gelu_1d_tensor(self):
        """Test FastGelu with 1D tensor (vector input)"""
        input_tensor = F._from_shape('input', [1024], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [F._from_shape('output', [1024], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_fast_gelu_1d',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [1024]
        assert perf_stats['inElems'] == 1024
        assert perf_stats['outElems'] == 1024

    def test_fast_gelu_2d_tensor(self):
        """Test FastGelu with 2D tensor (matrix input)"""
        input_tensor = F._from_shape('input', [64, 128], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [F._from_shape('output', [64, 128], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_fast_gelu_2d',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == [64, 128]
        assert perf_stats['inElems'] == 64 * 128
        assert perf_stats['outElems'] == 64 * 128

    def test_fast_gelu_large_tensor(self):
        """Test FastGelu with large tensor to verify performance scaling"""
        batch_size, seq_len, hidden_size = 4, 128, 4096  # ~2M elements

        inT, outT = create_fast_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_large',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = batch_size * seq_len * hidden_size
        expected_elements = 4 * 128 * 4096

        assert perf_stats['inElems'] == expected_elements
        assert perf_stats['outElems'] == expected_elements

        # Verify performance scaling - should have significant instruction counts
        assert perf_stats['instrs']['mul'] >= expected_elements * 4  # At least 4 multiplications per element
        assert perf_stats['instrs']['add'] >= expected_elements * 2  # At least 2 additions per element
        assert perf_stats['instrs']['tanh'] >= expected_elements      # At least 1 tanh per element
        assert perf_stats['instrs']['exp'] >= expected_elements * 2  # At least 2 exp per element (for tanh)

    def test_fast_gelu_grad_basic(self):
        """Test basic FastGeluGrad operation"""
        batch_size, seq_len, hidden_size = 2, 8, 512

        inT, outT = create_fast_gelu_grad_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_grad_basic',
            'optype': 'FastGeluGrad',
            'inList': ['x', 'dY'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluGradOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input)
        expected_shape = [batch_size, seq_len, hidden_size]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert perf_stats['inElems'] == 2 * batch_size * seq_len * hidden_size  # Two inputs
        assert perf_stats['outElems'] == batch_size * seq_len * hidden_size
        assert perf_stats['instrs']['mul'] > 0
        assert perf_stats['instrs']['add'] > 0
        assert perf_stats['instrs']['tanh'] > 0
        assert perf_stats['instrs']['exp'] > 0

    def test_fast_gelu_grad_different_shapes(self):
        """Test FastGeluGrad with different tensor shapes"""
        test_configs = [
            (1, 4, 128),     # Small configuration
            (4, 16, 1024),   # Large configuration
            (2, 32, 768),    # BERT-like configuration
        ]

        for batch_size, seq_len, hidden_size in test_configs:
            inT, outT = create_fast_gelu_grad_test_tensors(
                batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
            )

            op_info = {
                'name': f'test_fast_gelu_grad_{batch_size}_{seq_len}_{hidden_size}',
                'optype': 'FastGeluGrad',
                'inList': ['x', 'dY'],
                'outList': ['output'],
                'attrs': {},
            }
            op = FastGeluGradOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, seq_len, hidden_size]
            assert outT[0].shape == expected_shape

            # Verify input elements count (two inputs)
            expected_input_elements = 2 * batch_size * seq_len * hidden_size
            assert perf_stats['inElems'] == expected_input_elements

    def test_fast_gelu_vs_gelu_complexity(self):
        """Test that FastGelu has similar or lower complexity compared to Gelu"""
        batch_size, seq_len, hidden_size = 2, 8, 256

        # Create tensors for both operations
        input_tensor = F._from_shape('input', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))
        inT = [input_tensor]
        outT_fast = [F._from_shape('output_fast', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]
        outT_gelu = [F._from_shape('output_gelu', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        # FastGelu operation
        fast_gelu_op_info = {
            'name': 'test_fast_gelu_complexity',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output_fast'],
            'attrs': {},
        }
        fast_gelu_op = FastGeluOp(fast_gelu_op_info)
        fast_gelu_stats = fast_gelu_op.get_perf_counts(inT, outT_fast)

        # Gelu operation (approximate='tanh')
        gelu_op_info = {
            'name': 'test_gelu_complexity',
            'optype': 'Gelu',
            'inList': ['input'],
            'outList': ['output_gelu'],
            'attrs': {'approximate': 'tanh'},
        }
        gelu_op = SimOpFactory('Gelu')(gelu_op_info)
        gelu_stats = gelu_op.get_perf_counts(inT, outT_gelu)

        # FastGelu should have similar complexity to Gelu with tanh approximation
        # Allow some tolerance due to implementation differences
        tolerance_factor = 1.5

        assert fast_gelu_stats['instrs']['mul'] <= gelu_stats['instrs']['mul'] * tolerance_factor
        assert fast_gelu_stats['instrs']['add'] <= gelu_stats['instrs']['add'] * tolerance_factor
        assert fast_gelu_stats['instrs']['tanh'] <= gelu_stats['instrs']['tanh'] * tolerance_factor

        # Only compare exp if both operations have it
        if 'exp' in gelu_stats['instrs'] and 'exp' in fast_gelu_stats['instrs']:
            assert fast_gelu_stats['instrs']['exp'] <= gelu_stats['instrs']['exp'] * tolerance_factor

    def test_fast_gelu_memory_efficiency(self):
        """Test that FastGelu has reasonable memory requirements"""
        batch_size, seq_len, hidden_size = 4, 32, 1024
        total_elements = batch_size * seq_len * hidden_size

        inT, outT = create_fast_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_memory',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input + output should be at least 2x input size (in place operations possible)
        assert perf_stats['inBytes'] > 0
        assert perf_stats['outBytes'] > 0
        assert perf_stats['inElems'] == total_elements
        assert perf_stats['outElems'] == total_elements

    def test_fast_gelu_grad_memory_efficiency(self):
        """Test that FastGeluGrad has reasonable memory requirements"""
        batch_size, seq_len, hidden_size = 4, 32, 1024
        total_elements = batch_size * seq_len * hidden_size

        inT, outT = create_fast_gelu_grad_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_grad_memory',
            'optype': 'FastGeluGrad',
            'inList': ['x', 'dY'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluGradOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Should have 2 inputs and 1 output
        assert perf_stats['inElems'] == 2 * total_elements
        assert perf_stats['outElems'] == total_elements
        assert perf_stats['inBytes'] > perf_stats['outBytes']  # Input should be larger than output

    def test_fast_gelu_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, seq_len, hidden_size = 2, 4, 128

        inT, outT = create_fast_gelu_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_backward',
            'optype': 'FastGelu',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="FastGelu backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_fast_gelu_grad_backward_not_implemented(self):
        """Test that FastGeluGrad backward pass raises NotImplementedError"""
        batch_size, seq_len, hidden_size = 2, 4, 128

        inT, outT = create_fast_gelu_grad_test_tensors(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        )

        op_info = {
            'name': 'test_fast_gelu_grad_backward',
            'optype': 'FastGeluGrad',
            'inList': ['x', 'dY'],
            'outList': ['output'],
            'attrs': {},
        }
        op = FastGeluGradOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('x_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32')),
                F._from_shape('dY_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, seq_len, hidden_size], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="FastGeluGrad backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_fast_gelu_different_dtypes(self):
        """Test FastGelu with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [4, 8, 128], np_dtype=dtype)

            inT = [input_tensor]
            outT = [F._from_shape('output', [4, 8, 128], np_dtype=dtype)]

            op_info = {
                'name': f'test_fast_gelu_{dtype}',
                'optype': 'FastGelu',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = FastGeluOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [4, 8, 128]
            assert perf_stats['inElems'] == 4 * 8 * 128
            assert perf_stats['outElems'] == 4 * 8 * 128

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # For same shape input/output, bytes should be equal
                assert perf_stats['inBytes'] == perf_stats['outBytes']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
