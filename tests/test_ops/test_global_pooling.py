"""
Comprehensive test suite for GlobalMaxPool and GlobalAveragePool operations implementation.

This module tests the ONNX GlobalMaxPool and GlobalAveragePool operations, which are
commonly used in convolutional neural networks for classification tasks. These operations
reduce spatial dimensions by taking the maximum or average across all spatial locations.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOpFactory, GlobalMaxPoolOp, GlobalAveragePoolOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_global_pool_test_tensors(batch_size=2, channels=64, height=7, width=7):
    """
    Helper function to create test tensors for GlobalMaxPool and GlobalAveragePool operations.

    Args:
        batch_size: Batch size for input tensor
        channels: Number of channels
        height: Spatial height dimension
        width: Spatial width dimension

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create 4D input tensor [batch_size, channels, height, width]
    input_tensor = F._from_shape('input', [batch_size, channels, height, width], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor]

    # Create output tensor [batch_size, channels, 1, 1]
    output_tensor = F._from_shape('output', [batch_size, channels, 1, 1], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


def create_global_pool_3d_test_tensors(batch_size=2, channels=32, depth=8, height=7, width=7):
    """
    Helper function to create 5D test tensors for GlobalMaxPool and GlobalAveragePool operations.

    Args:
        batch_size: Batch size for input tensor
        channels: Number of channels
        depth: Spatial depth dimension
        height: Spatial height dimension
        width: Spatial width dimension

    Returns:
        Tuple of (input_tensors, output_tensors)
    """
    # Create 5D input tensor [batch_size, channels, depth, height, width]
    input_tensor = F._from_shape('input', [batch_size, channels, depth, height, width], np_dtype=np.dtype('float32'))

    input_tensors = [input_tensor]

    # Create output tensor [batch_size, channels, 1, 1, 1]
    output_tensor = F._from_shape('output', [batch_size, channels, 1, 1, 1], np_dtype=np.dtype('float32'))

    output_tensors = [output_tensor]

    return input_tensors, output_tensors


class TestGlobalMaxPool:
    """Test class for GlobalMaxPool operation"""

    def test_factory_integration_global_max_pool(self):
        """Test that GlobalMaxPool is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('GlobalMaxPool')
        assert opcls == GlobalMaxPoolOp

    def test_basic_global_max_pool(self):
        """Test basic GlobalMaxPool with standard 4D configuration"""
        batch_size, channels, height, width = 2, 64, 7, 7

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_max_pool_basic',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [batch_size, channels, 1, 1]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['cmp'] > 0  # Should have comparison operations
        assert perf_stats['instrs']['mac'] == 0  # No multiply-accumulate operations

    def test_global_max_pool_5d_tensor(self):
        """Test GlobalMaxPool with 5D tensor (3D spatial dimensions)"""
        batch_size, channels, depth, height, width = 2, 32, 8, 7, 7

        inT, outT = create_global_pool_3d_test_tensors(
            batch_size=batch_size, channels=channels, depth=depth, height=height, width=width
        )

        op_info = {
            'name': 'test_global_max_pool_5d',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [batch_size, channels, 1, 1, 1]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        total_input_elements = batch_size * channels * depth * height * width
        total_output_elements = batch_size * channels
        spatial_elements = total_input_elements // total_output_elements

        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == total_output_elements
        assert perf_stats['instrs']['cmp'] == total_output_elements * (spatial_elements - 1)

    def test_global_max_pool_different_shapes(self):
        """Test GlobalMaxPool with different tensor shapes"""
        test_configs = [
            (1, 32, 14, 14),    # Small feature maps
            (4, 128, 7, 7),     # Standard ResNet feature maps
            (2, 256, 14, 14),   # Larger feature maps
            (8, 512, 7, 7),     # Very large feature maps
        ]

        for batch_size, channels, height, width in test_configs:
            inT, outT = create_global_pool_test_tensors(
                batch_size=batch_size, channels=channels, height=height, width=width
            )

            op_info = {
                'name': f'test_global_max_pool_{batch_size}_{channels}_{height}_{width}',
                'optype': 'GlobalMaxPool',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = GlobalMaxPoolOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, channels, 1, 1]
            assert outT[0].shape == expected_shape

            # Verify element counts
            total_input_elements = batch_size * channels * height * width
            total_output_elements = batch_size * channels
            spatial_elements = height * width

            assert perf_stats['inElems'] == total_input_elements
            assert perf_stats['outElems'] == total_output_elements
            assert perf_stats['instrs']['cmp'] == total_output_elements * (spatial_elements - 1)

    def test_global_max_pool_large_tensor(self):
        """Test GlobalMaxPool with large tensor to verify performance scaling"""
        batch_size, channels, height, width = 4, 256, 14, 14  # ~400K elements

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_max_pool_large',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_input_elements = batch_size * channels * height * width
        total_output_elements = batch_size * channels
        spatial_elements = height * width

        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == total_output_elements

        # Verify performance scaling - should have many comparison operations
        expected_cmp = total_output_elements * (spatial_elements - 1)
        assert perf_stats['instrs']['cmp'] == expected_cmp
        assert perf_stats['instrs']['cmp'] > 100000  # Should be substantial for large tensors

    def test_global_max_pool_memory_efficiency(self):
        """Test that GlobalMaxPool has reasonable memory requirements"""
        batch_size, channels, height, width = 4, 128, 14, 14
        total_input_elements = batch_size * channels * height * width

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_max_pool_memory',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input should be much larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == batch_size * channels

    def test_invalid_input_rank_2d(self):
        """Test error handling for 2D input tensor"""
        input_tensor = F._from_shape('input', [4, 64], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [F._from_shape('output', [4, 64, 1, 1], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_rank_2d',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        with pytest.raises(ValueError, match="GlobalMaxPool requires input rank >= 3"):
            op.get_perf_counts(inT, outT)

    def test_global_max_pool_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, channels, height, width = 2, 32, 7, 7

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_max_pool_backward',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalMaxPoolOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, channels, height, width], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, channels, 1, 1], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="GlobalMaxPool backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_global_max_pool_different_dtypes(self):
        """Test GlobalMaxPool with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [2, 64, 7, 7], np_dtype=dtype)

            inT = [input_tensor]
            outT = [F._from_shape('output', [2, 64, 1, 1], np_dtype=dtype)]

            op_info = {
                'name': f'test_global_max_pool_{dtype}',
                'optype': 'GlobalMaxPool',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = GlobalMaxPoolOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [2, 64, 1, 1]
            assert perf_stats['inElems'] == 2 * 64 * 7 * 7
            assert perf_stats['outElems'] == 2 * 64

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # Input has more elements than output due to pooling
                # inBytes should be much larger than outBytes
                expected_in_bytes = 2 * 64 * 7 * 7 * 8  # input shape * 8 bytes per float64
                expected_out_bytes = 2 * 64 * 8  # output shape * 8 bytes per float64
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes


class TestGlobalAveragePool:
    """Test class for GlobalAveragePool operation"""

    def test_factory_integration_global_average_pool(self):
        """Test that GlobalAveragePool is properly integrated into SimOpFactory"""
        opcls = SimOpFactory('GlobalAveragePool')
        assert opcls == GlobalAveragePoolOp

    def test_basic_global_average_pool(self):
        """Test basic GlobalAveragePool with standard 4D configuration"""
        batch_size, channels, height, width = 2, 64, 7, 7

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_avg_pool_basic',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [batch_size, channels, 1, 1]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['add'] > 0  # Should have addition operations
        assert perf_stats['instrs']['div'] > 0  # Should have division operations
        assert perf_stats['instrs']['cmp'] == 0  # No comparison operations

    def test_global_average_pool_5d_tensor(self):
        """Test GlobalAveragePool with 5D tensor (3D spatial dimensions)"""
        batch_size, channels, depth, height, width = 2, 32, 8, 7, 7

        inT, outT = create_global_pool_3d_test_tensors(
            batch_size=batch_size, channels=channels, depth=depth, height=height, width=width
        )

        op_info = {
            'name': 'test_global_avg_pool_5d',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        expected_shape = [batch_size, channels, 1, 1, 1]
        assert outT[0].shape == expected_shape

        # Verify performance statistics
        total_input_elements = batch_size * channels * depth * height * width
        total_output_elements = batch_size * channels
        spatial_elements = total_input_elements // total_output_elements

        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == total_output_elements
        assert perf_stats['instrs']['add'] == total_output_elements * (spatial_elements - 1)
        assert perf_stats['instrs']['div'] == total_output_elements

    def test_global_average_pool_different_shapes(self):
        """Test GlobalAveragePool with different tensor shapes"""
        test_configs = [
            (1, 32, 14, 14),    # Small feature maps
            (4, 128, 7, 7),     # Standard ResNet feature maps
            (2, 256, 14, 14),   # Larger feature maps
            (8, 512, 7, 7),     # Very large feature maps
        ]

        for batch_size, channels, height, width in test_configs:
            inT, outT = create_global_pool_test_tensors(
                batch_size=batch_size, channels=channels, height=height, width=width
            )

            op_info = {
                'name': f'test_global_avg_pool_{batch_size}_{channels}_{height}_{width}',
                'optype': 'GlobalAveragePool',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = GlobalAveragePoolOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape
            expected_shape = [batch_size, channels, 1, 1]
            assert outT[0].shape == expected_shape

            # Verify element counts
            total_input_elements = batch_size * channels * height * width
            total_output_elements = batch_size * channels
            spatial_elements = height * width

            assert perf_stats['inElems'] == total_input_elements
            assert perf_stats['outElems'] == total_output_elements
            assert perf_stats['instrs']['add'] == total_output_elements * (spatial_elements - 1)
            assert perf_stats['instrs']['div'] == total_output_elements

    def test_global_average_pool_large_tensor(self):
        """Test GlobalAveragePool with large tensor to verify performance scaling"""
        batch_size, channels, height, width = 4, 256, 14, 14  # ~400K elements

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_avg_pool_large',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        total_input_elements = batch_size * channels * height * width
        total_output_elements = batch_size * channels
        spatial_elements = height * width

        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == total_output_elements

        # Verify performance scaling - should have many addition and division operations
        expected_add = total_output_elements * (spatial_elements - 1)
        assert perf_stats['instrs']['add'] == expected_add
        assert perf_stats['instrs']['add'] > 100000  # Should be substantial for large tensors
        assert perf_stats['instrs']['div'] == total_output_elements

    def test_global_average_pool_memory_efficiency(self):
        """Test that GlobalAveragePool has reasonable memory requirements"""
        batch_size, channels, height, width = 4, 128, 14, 14
        total_input_elements = batch_size * channels * height * width

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_avg_pool_memory',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Memory requirements should be reasonable
        # Input should be much larger than output
        assert perf_stats['inBytes'] > perf_stats['outBytes']
        assert perf_stats['inElems'] == total_input_elements
        assert perf_stats['outElems'] == batch_size * channels

    def test_invalid_input_rank_2d(self):
        """Test error handling for 2D input tensor"""
        input_tensor = F._from_shape('input', [4, 64], np_dtype=np.dtype('float32'))

        inT = [input_tensor]
        outT = [F._from_shape('output', [4, 64, 1, 1], np_dtype=np.dtype('float32'))]

        op_info = {
            'name': 'test_invalid_rank_2d',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        with pytest.raises(ValueError, match="GlobalAveragePool requires input rank >= 3"):
            op.get_perf_counts(inT, outT)

    def test_global_average_pool_backward_not_implemented(self):
        """Test that backward pass raises NotImplementedError"""
        batch_size, channels, height, width = 2, 32, 7, 7

        inT, outT = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        op_info = {
            'name': 'test_global_avg_pool_backward',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output'],
            'attrs': {},
        }
        op = GlobalAveragePoolOp(op_info)

        # Create dummy gradient tensors
        inGT = [F._from_shape('input_grad', [batch_size, channels, height, width], np_dtype=np.dtype('float32'))]
        outGT = [F._from_shape('output_grad', [batch_size, channels, 1, 1], np_dtype=np.dtype('float32'))]

        with pytest.raises(NotImplementedError, match="GlobalAveragePool backward pass not yet implemented"):
            op.backward(inT, outT, inGT, outGT)

    def test_global_average_pool_different_dtypes(self):
        """Test GlobalAveragePool with different data types"""
        dtypes = [np.dtype('float32'), np.dtype('float64')]

        for dtype in dtypes:
            input_tensor = F._from_shape('input', [2, 64, 7, 7], np_dtype=dtype)

            inT = [input_tensor]
            outT = [F._from_shape('output', [2, 64, 1, 1], np_dtype=dtype)]

            op_info = {
                'name': f'test_global_avg_pool_{dtype}',
                'optype': 'GlobalAveragePool',
                'inList': ['input'],
                'outList': ['output'],
                'attrs': {},
            }
            op = GlobalAveragePoolOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify shapes and element counts
            assert outT[0].shape == [2, 64, 1, 1]
            assert perf_stats['inElems'] == 2 * 64 * 7 * 7
            assert perf_stats['outElems'] == 2 * 64

            # Memory usage should scale with data type size
            if dtype == np.dtype('float64'):
                # Input has more elements than output due to pooling
                # inBytes should be much larger than outBytes
                expected_in_bytes = 2 * 64 * 7 * 7 * 8  # input shape * 8 bytes per float64
                expected_out_bytes = 2 * 64 * 8  # output shape * 8 bytes per float64
                assert perf_stats['inBytes'] == expected_in_bytes
                assert perf_stats['outBytes'] == expected_out_bytes

    def test_global_pooling_operations_comparison(self):
        """Test comparison between GlobalMaxPool and GlobalAveragePool"""
        batch_size, channels, height, width = 2, 64, 7, 7

        inT, outT_max = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )
        _, outT_avg = create_global_pool_test_tensors(
            batch_size=batch_size, channels=channels, height=height, width=width
        )

        # GlobalMaxPool operation
        max_pool_op_info = {
            'name': 'test_max_pool_comparison',
            'optype': 'GlobalMaxPool',
            'inList': ['input'],
            'outList': ['output_max'],
            'attrs': {},
        }
        max_pool_op = GlobalMaxPoolOp(max_pool_op_info)
        max_pool_stats = max_pool_op.get_perf_counts(inT, outT_max)

        # GlobalAveragePool operation
        avg_pool_op_info = {
            'name': 'test_avg_pool_comparison',
            'optype': 'GlobalAveragePool',
            'inList': ['input'],
            'outList': ['output_avg'],
            'attrs': {},
        }
        avg_pool_op = GlobalAveragePoolOp(avg_pool_op_info)
        avg_pool_stats = avg_pool_op.get_perf_counts(inT, outT_avg)

        # Both operations should have same input/output element counts
        assert max_pool_stats['inElems'] == avg_pool_stats['inElems']
        assert max_pool_stats['outElems'] == avg_pool_stats['outElems']

        # GlobalMaxPool should use comparisons, GlobalAveragePool should use additions and divisions
        assert max_pool_stats['instrs']['cmp'] > 0
        assert max_pool_stats['instrs']['add'] == 0
        assert max_pool_stats['instrs']['div'] == 0

        assert avg_pool_stats['instrs']['cmp'] == 0
        assert avg_pool_stats['instrs']['add'] > 0
        assert avg_pool_stats['instrs']['div'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
