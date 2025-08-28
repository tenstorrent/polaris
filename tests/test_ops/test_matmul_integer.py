"""
Comprehensive test suite for MatMulInteger operation implementation.

This module tests the ONNX MatMulInteger operation, which performs matrix
multiplication on integer tensors without scaling or zero-point adjustments.
This is commonly used in quantized neural networks where quantization parameters
are handled separately.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, MatMulIntegerOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_matmul_integer_test_tensors(M=4, K=8, N=6):
    """
    Helper function to create test tensors for MatMulInteger operation.

    Args:
        M: Rows in first matrix
        K: Columns in first matrix / Rows in second matrix
        N: Columns in second matrix

    Returns:
        Tuple of (input_tensors, output_tensors) for [M,K] x [K,N] -> [M,N]
    """
    # Create input tensors: A [M, K] and B [K, N]
    A_tensor = F._from_shape('A', [M, K], np_dtype=np.dtype('int32'))
    B_tensor = F._from_shape('B', [K, N], np_dtype=np.dtype('int32'))

    input_tensors = [A_tensor, B_tensor]

    # Create output tensor [M, N]
    output_tensor = F._from_shape('output', [M, N], np_dtype=np.dtype('int32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_vector_matrix_test_tensors(K=6, N=4):
    """
    Helper function to create test tensors for vector-matrix multiplication.

    Args:
        K: Vector length / Matrix rows
        N: Matrix columns

    Returns:
        Tuple of (input_tensors, output_tensors) for [K] x [K,N] -> [N]
    """
    # Create input tensors: vector [K] and matrix [K, N]
    vector_tensor = F._from_shape('vector', [K], np_dtype=np.dtype('int32'))
    matrix_tensor = F._from_shape('matrix', [K, N], np_dtype=np.dtype('int32'))

    input_tensors = [vector_tensor, matrix_tensor]

    # Create output tensor [N]
    output_tensor = F._from_shape('output', [N], np_dtype=np.dtype('int32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_matrix_vector_test_tensors(M=4, K=6):
    """
    Helper function to create test tensors for matrix-vector multiplication.

    Args:
        M: Matrix rows
        K: Matrix columns / Vector length

    Returns:
        Tuple of (input_tensors, output_tensors) for [M,K] x [K] -> [M]
    """
    # Create input tensors: matrix [M, K] and vector [K]
    matrix_tensor = F._from_shape('matrix', [M, K], np_dtype=np.dtype('int32'))
    vector_tensor = F._from_shape('vector', [K], np_dtype=np.dtype('int32'))

    input_tensors = [matrix_tensor, vector_tensor]

    # Create output tensor [M]
    output_tensor = F._from_shape('output', [M], np_dtype=np.dtype('int32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_vector_dot_product_test_tensors(K=6):
    """
    Helper function to create test tensors for vector dot product.

    Args:
        K: Vector length

    Returns:
        Tuple of (input_tensors, output_tensors) for [K] x [K] -> []
    """
    # Create input tensors: two vectors [K]
    vector_a_tensor = F._from_shape('vector_a', [K], np_dtype=np.dtype('int32'))
    vector_b_tensor = F._from_shape('vector_b', [K], np_dtype=np.dtype('int32'))

    input_tensors = [vector_a_tensor, vector_b_tensor]

    # Create output tensor [] (scalar)
    output_tensor = F._from_shape('output', [], np_dtype=np.dtype('int32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestMatMulInteger:
    """Test class for MatMulInteger operation"""

    def test_factory_integration(self):
        """Test that MatMulInteger is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('MatMulInteger')
        assert opcls == MatMulIntegerOp

    def test_basic_matrix_matrix_multiplication(self):
        """Test basic matrix-matrix multiplication [M,K] x [K,N] -> [M,N]"""
        M, K, N = 4, 6, 8

        inT, outT = create_matmul_integer_test_tensors(M=M, K=K, N=N)

        op_info = {
            'name': 'test_matmul_basic',
            'optype': 'MatMulInteger',
            'inList': ['A', 'B'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [M, N]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['mac'] > 0  # Should have multiply-accumulate operations
        assert perf_stats['instrs']['mul'] > 0  # Should have multiplications
        assert perf_stats['instrs']['add'] > 0  # Should have additions

        # Verify element counts
        expected_input_elements = M * K + K * N
        expected_output_elements = M * N
        expected_mac_operations = M * N * K

        assert perf_stats['inElems'] == expected_input_elements
        assert perf_stats['outElems'] == expected_output_elements
        assert perf_stats['instrs']['mac'] == expected_mac_operations

    def test_vector_matrix_multiplication(self):
        """Test vector-matrix multiplication [K] x [K,N] -> [N]"""
        K, N = 6, 4

        inT, outT = create_vector_matrix_test_tensors(K=K, N=N)

        op_info = {
            'name': 'test_vector_matrix',
            'optype': 'MatMulInteger',
            'inList': ['vector', 'matrix'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [N]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        expected_input_elements = K + K * N
        expected_output_elements = N
        expected_mac_operations = N * K

        assert perf_stats['inElems'] == expected_input_elements
        assert perf_stats['outElems'] == expected_output_elements
        assert perf_stats['instrs']['mac'] == expected_mac_operations

    def test_matrix_vector_multiplication(self):
        """Test matrix-vector multiplication [M,K] x [K] -> [M]"""
        M, K = 4, 6

        inT, outT = create_matrix_vector_test_tensors(M=M, K=K)

        op_info = {
            'name': 'test_matrix_vector',
            'optype': 'MatMulInteger',
            'inList': ['matrix', 'vector'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [M]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        expected_input_elements = M * K + K
        expected_output_elements = M
        expected_mac_operations = M * K

        assert perf_stats['inElems'] == expected_input_elements
        assert perf_stats['outElems'] == expected_output_elements
        assert perf_stats['instrs']['mac'] == expected_mac_operations

    def test_vector_dot_product(self):
        """Test vector dot product [K] x [K] -> []"""
        K = 6

        inT, outT = create_vector_dot_product_test_tensors(K=K)

        op_info = {
            'name': 'test_dot_product',
            'optype': 'MatMulInteger',
            'inList': ['vector_a', 'vector_b'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (scalar)
        expected_shape = []
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        expected_input_elements = K + K
        expected_output_elements = 1
        expected_mac_operations = K

        assert perf_stats['inElems'] == expected_input_elements
        assert perf_stats['outElems'] == expected_output_elements
        assert perf_stats['instrs']['mac'] == expected_mac_operations

    def test_matmul_integer_different_shapes(self):
        """Test MatMulInteger with different tensor shapes and configurations"""
        test_configs = [
            # (M, K, N, description)
            (2, 3, 4, "Small matrices"),
            (8, 12, 16, "Medium matrices"),
            (32, 64, 128, "Large matrices"),
            (1, 100, 1, "Thin matrices"),
            (100, 1, 100, "Tall matrices"),
        ]

        for M, K, N, description in test_configs:
            inT, outT = create_matmul_integer_test_tensors(M=M, K=K, N=N)

            op_info = {
                'name': f'test_matmul_{description.lower().replace(" ", "_")}',
                'optype': 'MatMulInteger',
                'inList': ['A', 'B'],
                'outList': ['output'],
                'attrs': {},
            }
            op = MatMulIntegerOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [M, N]
            assert outT[0].shape == expected_shape

            # Verify performance scales correctly
            expected_input_elements = M * K + K * N
            expected_output_elements = M * N
            expected_mac_operations = M * N * K

            assert perf_stats['inElems'] == expected_input_elements
            assert perf_stats['outElems'] == expected_output_elements
            assert perf_stats['instrs']['mac'] == expected_mac_operations

    def test_matmul_integer_large_tensor(self):
        """Test MatMulInteger with large tensor to verify performance scaling"""
        M, K, N = 64, 128, 256  # ~2M MAC operations

        inT, outT = create_matmul_integer_test_tensors(M=M, K=K, N=N)

        op_info = {
            'name': 'test_matmul_large',
            'optype': 'MatMulInteger',
            'inList': ['A', 'B'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_elements = M * K * N
        expected_elements = 64 * 128 * 256

        assert perf_stats['inElems'] == M * K + K * N
        assert perf_stats['outElems'] == M * N

        # Verify performance scaling - should have millions of operations
        expected_mac = M * N * K
        assert perf_stats['instrs']['mac'] == expected_mac
        assert perf_stats['instrs']['mac'] > 2000000  # More than 2M operations

    def test_matmul_integer_memory_efficiency(self):
        """Test that MatMulInteger has reasonable memory requirements"""
        M, K, N = 8, 16, 12

        inT, outT = create_matmul_integer_test_tensors(M=M, K=K, N=N)

        op_info = {
            'name': 'test_matmul_memory',
            'optype': 'MatMulInteger',
            'inList': ['A', 'B'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input should be larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == M * K + K * N
        assert perf_stats['outElems'] == M * N

    def test_dimension_mismatch_matrix_matrix(self):
        """Test error handling for dimension mismatch in matrix-matrix multiplication"""
        # Create tensors with mismatched inner dimensions
        A_tensor = F._from_shape('A', [4, 6], np_dtype=np.dtype('int32'))  # [4, 6]
        B_tensor = F._from_shape('B', [8, 12], np_dtype=np.dtype('int32'))  # [8, 12] - 6 != 8

        inT = [A_tensor, B_tensor]
        outT = [F._from_shape('output', [4, 12], np_dtype=np.dtype('int32'))]

        op_info = {
            'name': 'test_dimension_mismatch',
            'optype': 'MatMulInteger',
            'inList': ['A', 'B'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        with pytest.raises(ValueError, match="Matrix inner dimensions must match"):
            op.get_perf_counts(inT, outT)

    def test_dimension_mismatch_vector_matrix(self):
        """Test error handling for dimension mismatch in vector-matrix multiplication"""
        # Create tensors with mismatched dimensions
        vector_tensor = F._from_shape('vector', [6], np_dtype=np.dtype('int32'))  # [6]
        matrix_tensor = F._from_shape('matrix', [8, 12], np_dtype=np.dtype('int32'))  # [8, 12] - 6 != 8

        inT = [vector_tensor, matrix_tensor]
        outT = [F._from_shape('output', [12], np_dtype=np.dtype('int32'))]

        op_info = {
            'name': 'test_vector_matrix_mismatch',
            'optype': 'MatMulInteger',
            'inList': ['vector', 'matrix'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        with pytest.raises(ValueError, match="Vector length .* must match matrix rows"):
            op.get_perf_counts(inT, outT)

    def test_dimension_mismatch_dot_product(self):
        """Test error handling for dimension mismatch in vector dot product"""
        # Create vectors with different lengths
        vector_a_tensor = F._from_shape('vector_a', [6], np_dtype=np.dtype('int32'))  # [6]
        vector_b_tensor = F._from_shape('vector_b', [8], np_dtype=np.dtype('int32'))  # [8] - 6 != 8

        inT = [vector_a_tensor, vector_b_tensor]
        outT = [F._from_shape('output', [], np_dtype=np.dtype('int32'))]

        op_info = {
            'name': 'test_dot_product_mismatch',
            'optype': 'MatMulInteger',
            'inList': ['vector_a', 'vector_b'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        with pytest.raises(ValueError, match="Vector dimensions must match for dot product"):
            op.get_perf_counts(inT, outT)

    def test_zero_dimension_handling(self):
        """Test error handling for zero-dimension inputs"""
        # Create scalar tensors (0-dimensional)
        scalar_a_tensor = F._from_shape('scalar_a', [], np_dtype=np.dtype('int32'))
        scalar_b_tensor = F._from_shape('scalar_b', [], np_dtype=np.dtype('int32'))

        inT = [scalar_a_tensor, scalar_b_tensor]
        outT = [F._from_shape('output', [], np_dtype=np.dtype('int32'))]

        op_info = {
            'name': 'test_zero_dimension',
            'optype': 'MatMulInteger',
            'inList': ['scalar_a', 'scalar_b'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        with pytest.raises(ValueError, match="MatMulInteger inputs must have at least 1 dimension"):
            op.get_perf_counts(inT, outT)

    def test_matmul_integer_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        M, K, N = 4, 6, 8

        inT, outT = create_matmul_integer_test_tensors(M=M, K=K, N=N)

        op_info = {
            'name': 'test_matmul_backward',
            'optype': 'MatMulInteger',
            'inList': ['A', 'B'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('A_grad', [M, K], np_dtype=np.dtype('int32')),
                F._from_shape('B_grad', [K, N], np_dtype=np.dtype('int32'))]
        outGT = [F._from_shape('output_grad', [M, N], np_dtype=np.dtype('int32'))]

        with pytest.raises(NotImplementedError, match="MatMulInteger backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_matmul_integer_different_dtypes(self):
        """Test MatMulInteger with different data types"""
        dtypes = [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'), np.dtype('int64')]

        M, K, N = 4, 6, 8

        for dtype in dtypes:
            A_tensor = F._from_shape('A', [M, K], np_dtype=dtype)
            B_tensor = F._from_shape('B', [K, N], np_dtype=dtype)

            inT = [A_tensor, B_tensor]
            outT = [F._from_shape('output', [M, N], np_dtype=dtype)]

            op_info = {
                'name': f'test_matmul_{dtype}',
                'optype': 'MatMulInteger',
                'inList': ['A', 'B'],
                'outList': ['output'],
                'attrs': {},
            }
            op = MatMulIntegerOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [M, N]
            assert perf_stats['inElems'] == M * K + K * N
            assert perf_stats['outElems'] == M * N
            assert perf_stats['instrs']['mac'] == M * N * K

            # Memory usage should scale with data type size
            if dtype == np.dtype('int64'):
                # Input includes both A and B tensors
                # inBytes should be A_bytes + B_bytes
                expected_in_bytes = M * K * 8 + K * N * 8  # A + B bytes for int64
                expected_out_bytes = M * N * 8  # output bytes for int64
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes

    def test_matmul_integer_quantization_context(self):
        """Test MatMulInteger in the context of quantized neural networks"""
        # Typical quantized linear layer: input -> MatMulInteger -> bias -> activation
        batch_size, input_features, output_features = 32, 512, 1024

        inT, outT = create_matmul_integer_test_tensors(
            M=batch_size, K=input_features, N=output_features
        )

        op_info = {
            'name': 'test_quantized_linear',
            'optype': 'MatMulInteger',
            'inList': ['input', 'weight'],
            'outList': ['output'],
            'attrs': {},
        }
        op = MatMulIntegerOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # This represents a typical quantized linear layer computation
        total_operations = batch_size * input_features * output_features
        assert perf_stats['instrs']['mac'] == total_operations

        # For quantized networks, this operation is typically followed by
        # bias addition and activation - verify it produces the right output shape
        assert outT[0].shape == [batch_size, output_features]

        # Performance should be substantial for realistic model sizes
        assert perf_stats['instrs']['mac'] > 16000000  # 32 * 512 * 1024 = 16.7M operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
