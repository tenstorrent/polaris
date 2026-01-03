#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for ScatterND operation implementation.

This module tests the ONNX 1.20.0 ScatterND operation, which scatters
updates into a tensor at specified indices with support for various
reduction modes including 'none', 'add', 'mul', 'min', 'max'.
"""

import numpy as np
import pytest
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_scatter_nd_test_tensors(data_shape, indices_shape, updates_shape,
                                  data_dtype='float32', indices_dtype='int64',
                                  updates_dtype='float32', reduction='none'):
    """
    Helper function to create test tensors for ScatterND operation.

    Args:
        data_shape: Shape of input data tensor
        indices_shape: Shape of indices tensor
        updates_shape: Shape of updates tensor
        data_dtype: Data type for data tensor
        indices_dtype: Data type for indices tensor
        updates_dtype: Data type for updates tensor
        reduction: Reduction mode ('none', 'add', 'mul', 'min', 'max')

    Returns:
        Tuple of (input_tensors, output_tensor, reference_result)
    """
    # Create input tensors
    data = F._from_shape('data', data_shape, np_dtype=np.dtype(data_dtype))
    indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype(indices_dtype))
    updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype(updates_dtype))

    # Create output tensor (same shape and dtype as data)
    output = F._from_shape('output', data_shape, np_dtype=np.dtype(data_dtype))

    # Generate reference result using numpy
    reference_result = reference_scatter_nd(data, indices, updates, reduction)

    return [data, indices, updates], [output], reference_result


def reference_scatter_nd(data, indices, updates, reduction='none'):
    """
    Reference implementation of ScatterND using numpy.

    Args:
        data: Input data tensor
        indices: Index tensor
        updates: Update values
        reduction: Reduction mode

    Returns:
        Reference result tensor
    """
    # Convert to numpy arrays for computation
    data_np = np.random.rand(*data.shape).astype(np.dtype(data.dtype))

    # Generate valid indices
    index_depth = indices.shape[-1]
    indices_np = np.zeros(indices.shape, dtype=np.dtype(indices.dtype))

    # Generate valid indices for each dimension being indexed
    for i in range(index_depth):
        indices_np[..., i] = np.random.randint(0, data.shape[i], indices_np[..., i].shape)

    updates_np = np.random.rand(*updates.shape).astype(np.dtype(updates.dtype))

    # Create a copy of data for the result
    result = data_np.copy()

    # Perform scatter operation
    if reduction == 'none':
        # Direct assignment - use advanced indexing
        index_tuple = tuple(indices_np[..., i] for i in range(index_depth))
        result[index_tuple] = updates_np
    elif reduction == 'add':
        # Add to existing values
        index_tuple = tuple(indices_np[..., i] for i in range(index_depth))
        result[index_tuple] += updates_np
    elif reduction == 'mul':
        # Multiply with existing values
        index_tuple = tuple(indices_np[..., i] for i in range(index_depth))
        result[index_tuple] *= updates_np
    elif reduction == 'min':
        # Minimum with existing values
        index_tuple = tuple(indices_np[..., i] for i in range(index_depth))
        current_values = result[index_tuple]
        result[index_tuple] = np.minimum(current_values, updates_np)
    elif reduction == 'max':
        # Maximum with existing values
        index_tuple = tuple(indices_np[..., i] for i in range(index_depth))
        current_values = result[index_tuple]
        result[index_tuple] = np.maximum(current_values, updates_np)

    return result


class TestScatterND:
    """Test class for ScatterND operation"""

    def test_factory_integration(self):
        """Test that ScatterND is registered in the op mapping"""
        from ttsim.ops.desc.registry import get_opdesc_registry
        assert get_opdesc_registry().has_shape_inference_function('ScatterND')

    def test_basic_2d_scatter_none(self):
        """Test basic 2D scatter with 'none' reduction mode"""
        data_shape = [4, 6]
        indices_shape = [2, 1]  # 2 updates, 1D indices
        updates_shape = [2, 6]  # Same as indices[:-1] + data[1:]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='none'
        )

        op_info = {
            'name': 'test_scatter_nd_basic',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape (same as input data)
        assert outT[0].shape == data_shape
        assert outT[0].dtype == inT[0].dtype

        # Verify performance statistics are computed
        assert 'inBytes' in perf_stats
        assert 'outBytes' in perf_stats
        assert 'instrs' in perf_stats
        assert perf_stats['instrs']['scatter'] > 0  # Should have scatter operations

    def test_3d_scatter_with_batch_dims(self):
        """Test 3D scatter with batch dimensions"""
        data_shape = [2, 3, 4]     # Batch of 2, 3x4 matrices
        indices_shape = [2, 2, 2]  # 2 batches, 2 updates each, 2D indices
        updates_shape = [2, 2, 4]  # Same as indices[:-1] + data[2:]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='none'
        )

        op_info = {
            'name': 'test_scatter_nd_batch',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape

    def test_reduction_add(self):
        """Test scatter with 'add' reduction mode"""
        data_shape = [3, 4]
        indices_shape = [3, 1]  # Multiple indices pointing to same location
        updates_shape = [3, 4]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='add'
        )

        op_info = {
            'name': 'test_scatter_nd_add',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'add'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape
        # Should have add operations
        assert perf_stats['instrs']['add'] > 0

    def test_reduction_mul(self):
        """Test scatter with 'mul' reduction mode"""
        data_shape = [3, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='mul'
        )

        op_info = {
            'name': 'test_scatter_nd_mul',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'mul'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape
        # Should have mul operations
        assert perf_stats['instrs']['mul'] > 0

    def test_reduction_min(self):
        """Test scatter with 'min' reduction mode"""
        data_shape = [3, 4]
        indices_shape = [3, 1]
        updates_shape = [3, 4]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='min'
        )

        op_info = {
            'name': 'test_scatter_nd_min',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'min'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape
        # Should have comparison operations
        assert perf_stats['instrs']['cmp'] > 0

    def test_reduction_max(self):
        """Test scatter with 'max' reduction mode"""
        data_shape = [3, 4]
        indices_shape = [3, 1]
        updates_shape = [3, 4]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='max'
        )

        op_info = {
            'name': 'test_scatter_nd_max',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'max'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape
        # Should have comparison operations
        assert perf_stats['instrs']['cmp'] > 0

    def test_different_data_types(self):
        """Test scatter with different supported data types"""
        for dtype in ['float32', 'float64', 'int32', 'int64']:
            data_shape = [4, 4]
            indices_shape = [2, 1]
            updates_shape = [2, 4]

            inT, outT, reference = create_scatter_nd_test_tensors(
                data_shape, indices_shape, updates_shape,
                data_dtype=dtype, updates_dtype=dtype, reduction='none'
            )

            op_info = {
                'name': f'test_scatter_nd_{dtype}',
                'optype': 'ScatterND',
                'inList': ['data', 'indices', 'updates'],
                'outList': ['output'],
                'attrs': {'reduction': 'none'},
            }
            op = SimOp(op_info)

            perf_stats = op.get_perf_counts(inT, outT)

            # Verify output shape and dtype
            assert outT[0].shape == data_shape
            assert outT[0].dtype == dtype

    def test_invalid_input_count(self):
        """Test error handling for invalid input count"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        inT, outT, _ = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='none'
        )

        # Add extra input
        extra_tensor = F._from_shape('Extra', [2, 2], np_dtype=np.dtype('float32'))
        inT.append(extra_tensor)

        op_info = {
            'name': 'test_scatter_nd_invalid_inputs',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        with pytest.raises(AssertionError, match="#inputs for .* should be in range"):
            op.get_perf_counts(inT, outT)

    def test_invalid_output_count(self):
        """Test error handling for invalid output count"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        inT, outT, _ = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='none'
        )

        # Add extra output
        extra_output = F._from_shape('Extra', [4, 4], np_dtype=np.dtype('float32'))
        outT.append(extra_output)

        op_info = {
            'name': 'test_scatter_nd_invalid_outputs',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        with pytest.raises(AssertionError, match="#outputs for .* should be in range"):
            op.get_perf_counts(inT, outT)

    def test_invalid_data_dtype(self):
        """Test error handling for unsupported data types"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        # Create tensors with unsupported data type
        data = F._from_shape('data', data_shape, np_dtype=np.dtype('bool'))
        indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype('int64'))
        updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype('bool'))

        op_info = {
            'name': 'test_scatter_nd_invalid_dtype',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        output = F._from_shape('output', data_shape, np_dtype=np.dtype('bool'))

        with pytest.raises(AssertionError, match="Data tensor must be numeric type"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_invalid_indices_dtype(self):
        """Test error handling for non-integer indices"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        # Create tensors with float indices (not supported)
        data = F._from_shape('data', data_shape, np_dtype=np.dtype('float32'))
        indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype('float32'))
        updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype('float32'))

        op_info = {
            'name': 'test_scatter_nd_invalid_indices',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        output = F._from_shape('output', data_shape, np_dtype=np.dtype('float32'))

        with pytest.raises(AssertionError, match="Indices must be integer type"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_mismatched_updates_dtype(self):
        """Test error handling for mismatched updates dtype"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [2, 4]

        # Create tensors with mismatched dtypes
        data = F._from_shape('data', data_shape, np_dtype=np.dtype('float32'))
        indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype('int64'))
        updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype('float64'))

        op_info = {
            'name': 'test_scatter_nd_dtype_mismatch',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        output = F._from_shape('output', data_shape, np_dtype=np.dtype('float32'))

        with pytest.raises(AssertionError, match="Updates must have same dtype as data"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_invalid_index_depth(self):
        """Test error handling for invalid index depth"""
        data_shape = [4, 4]      # 2D data
        indices_shape = [2, 3]   # 3D indices (exceeds data dimensions)
        updates_shape = [2, 4]   # This would be invalid too

        # Create tensors manually to avoid helper validation
        data = F._from_shape('data', data_shape, np_dtype=np.dtype('float32'))
        indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype('int64'))
        updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype('float32'))

        op_info = {
            'name': 'test_scatter_nd_invalid_depth',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        output = F._from_shape('output', data_shape, np_dtype=np.dtype('float32'))

        with pytest.raises(ValueError, match="Index depth .* exceeds data tensor dimensions"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_invalid_updates_shape(self):
        """Test error handling for invalid updates shape"""
        data_shape = [4, 4]
        indices_shape = [2, 1]
        updates_shape = [3, 4]  # Wrong batch dimension

        # Create tensors manually to avoid helper validation
        data = F._from_shape('data', data_shape, np_dtype=np.dtype('float32'))
        indices = F._from_shape('indices', indices_shape, np_dtype=np.dtype('int64'))
        updates = F._from_shape('updates', updates_shape, np_dtype=np.dtype('float32'))

        op_info = {
            'name': 'test_scatter_nd_invalid_updates',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        output = F._from_shape('output', data_shape, np_dtype=np.dtype('float32'))

        with pytest.raises(ValueError, match="Updates shape .*"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_invalid_reduction_mode(self):
        """Test error handling for invalid reduction mode"""
        op_info = {
            'name': 'test_scatter_nd_invalid_reduction',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'invalid_mode'},
        }

        # Create dummy tensors for validation
        data = F._from_shape('data', [1], np_dtype=np.dtype('float32'))
        indices = F._from_shape('indices', [1, 1], np_dtype=np.dtype('int64'))
        updates = F._from_shape('updates', [1], np_dtype=np.dtype('float32'))
        output = F._from_shape('output', [1], np_dtype=np.dtype('float32'))
        
        op = SimOp(op_info)

        with pytest.raises(ValueError, match="Invalid reduction mode"):
            op.get_perf_counts([data, indices, updates], [output])

    def test_embedded_scenario(self):
        """Test ScatterND in an embedded scenario (e.g., updating embeddings)"""
        # Embedding table: vocab_size x embedding_dim
        vocab_size, embedding_dim = 1000, 128
        data_shape = [vocab_size, embedding_dim]

        # Update specific embeddings: batch_size x 1 (index dimension)
        batch_size = 4
        indices_shape = [batch_size, 1]
        updates_shape = [batch_size, embedding_dim]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='none'
        )

        op_info = {
            'name': 'test_scatter_nd_embedding',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'none'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape

        # Should have significant scatter operations for embedding updates
        assert perf_stats['instrs']['scatter'] > 0
        assert perf_stats['instrs']['index'] > 0

    def test_gradient_accumulation_scenario(self):
        """Test ScatterND for gradient accumulation (add reduction)"""
        # Parameter tensor
        param_shape = [100, 200]
        data_shape = param_shape

        # Multiple gradient contributions to same parameters
        num_gradients = 8
        indices_shape = [num_gradients, 1]
        updates_shape = [num_gradients, 200]

        inT, outT, reference = create_scatter_nd_test_tensors(
            data_shape, indices_shape, updates_shape, reduction='add'
        )

        op_info = {
            'name': 'test_scatter_nd_grad_accum',
            'optype': 'ScatterND',
            'inList': ['data', 'indices', 'updates'],
            'outList': ['output'],
            'attrs': {'reduction': 'add'},
        }
        op = SimOp(op_info)

        perf_stats = op.get_perf_counts(inT, outT)

        # Verify output shape
        assert outT[0].shape == data_shape

        # Should have add operations for gradient accumulation
        assert perf_stats['instrs']['add'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
