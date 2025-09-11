#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from ttsim.ops.op import GatherNDOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def create_gather_nd_test_tensors(data_shape, indices_data, batch_dims=0):
    """Helper function to create test tensors for GatherND operation"""

    # Create data tensor with appropriate shape
    data_tensor = F._from_shape('data', data_shape)

    # Create indices tensor from the provided data
    indices_shape = indices_data.shape
    indices_tensor = F._from_data('indices', data=indices_data)

    input_tensors = [data_tensor, indices_tensor]
    input_names = ['data', 'indices']

    # Create output tensor (shape will be computed by the operation)
    output = make_tensor('output')
    output_tensors = [output]
    output_names = ['output']

    return input_tensors, output_tensors, input_names, output_names


def reference_gather_nd(data, indices, batch_dims=0):
    """
    Reference implementation of GatherND for testing
    Follows the ONNX specification exactly
    """
    # Convert to numpy for easier manipulation
    data = np.asarray(data)
    indices = np.asarray(indices)

    # Basic validation
    assert data.ndim >= 1, "Data must have at least 1 dimension"
    assert indices.ndim >= 1, "Indices must have at least 1 dimension"
    assert batch_dims >= 0, "batch_dims must be non-negative"

    # Get dimensions
    data_shape = data.shape
    indices_shape = indices.shape
    data_rank = data.ndim
    indices_rank = indices.ndim

    # Validate batch_dims
    assert batch_dims <= min(data_rank, indices_rank), "batch_dims too large"

    # Index depth is the last dimension of indices
    index_depth = indices_shape[-1]

    # Effective data rank after removing batch dimensions
    effective_data_rank = data_rank - batch_dims
    assert index_depth <= effective_data_rank, "Index depth too large"

    # Compute output shape
    # Batch dimensions come from indices (excluding last dimension)
    if batch_dims == 0:
        batch_shape = indices_shape[:-1]
    else:
        batch_shape = indices_shape[:batch_dims]

    # Remaining dimensions come from data after indexed dimensions
    indexed_dims = index_depth + batch_dims
    remaining_shape = data_shape[indexed_dims:]

    output_shape = batch_shape + remaining_shape

    # Perform the gather operation
    # Flatten indices for easier processing
    indices_flat = indices.reshape(-1, index_depth)

    # Calculate total number of elements to gather
    total_elements = int(np.prod(indices_shape[:-1]))

    # Initialize output array
    output = np.zeros(output_shape, dtype=data.dtype)

    # Perform gathering
    for i in range(total_elements):
        # Get the coordinate for this element
        if batch_dims == 0:
            # No batch dimensions
            coord = indices_flat[i]
            output_index = np.unravel_index(i, indices_shape[:-1])
        else:
            # With batch dimensions
            batch_coord = np.unravel_index(i, indices_shape[:batch_dims])
            index_coord = indices_flat[i]
            coord = batch_coord + index_coord
            output_index = batch_coord + np.unravel_index(i, indices_shape[:-1])[batch_dims:]

        # Gather the value
        value = data[tuple(coord)]
        output[output_index] = value

    return output


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_basic_2d():
    """Test basic GatherND with 2D data and simple indexing"""
    print("\n=== Testing Basic GatherND (2D) ===")

    # Create 2D data: [3, 4]
    data_shape = [3, 4]
    # Indices to gather: [[0, 0], [1, 2], [2, 3]]
    indices_data = np.array([[0, 0], [1, 2], [2, 3]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    attrs = {'batch_dims': 0}

    op_info = {
        'name': 'test_gather_nd_basic_2d',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    # Create and execute GatherND operation
    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    # Execute get_perf_counts to validate and set shapes
    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    # indices shape is [3, 2], so output shape should be [3]
    expected_output_shape = [3]
    assert output_tensors[0].shape == expected_output_shape, \
        f"Output shape mismatch: {output_tensors[0].shape} != {expected_output_shape}"

    # Validate output dtype (should match input data dtype)
    assert output_tensors[0].dtype == input_tensors[0].dtype, \
        f"Output dtype mismatch: {output_tensors[0].dtype} != {input_tensors[0].dtype}"

    print(f"✓ Basic 2D GatherND test passed")
    print(f"  Data shape: {input_tensors[0].shape}")
    print(f"  Indices shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_3d_simple():
    """Test GatherND with 3D data and 2D indexing"""
    print("\n=== Testing GatherND (3D data, 2D indices) ===")

    # Create 3D data: [2, 3, 4]
    data_shape = [2, 3, 4]
    # Indices to gather: [[0, 1, 2], [1, 2, 3]]
    indices_data = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    attrs = {'batch_dims': 0}

    op_info = {
        'name': 'test_gather_nd_3d_simple',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    # indices shape is [2, 3], so output shape should be [2]
    expected_output_shape = [2]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 3D GatherND test passed")
    print(f"  Data shape: {input_tensors[0].shape}")
    print(f"  Indices shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_with_batch_dims():
    """Test GatherND with batch_dims attribute"""
    print("\n=== Testing GatherND with batch_dims ===")

    # Create 3D data: [2, 3, 4]
    data_shape = [2, 3, 4]
    # Indices with batch dimensions: shape [2, 2, 2]
    # This means 2 batches, each with 2 indices of depth 2
    indices_data = np.array([[[0, 1], [1, 2]], [[0, 2], [1, 3]]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=1
    )

    attrs = {'batch_dims': 1}

    op_info = {
        'name': 'test_gather_nd_batch_dims',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    # With batch_dims=1, indices shape is [2, 2, 2]
    # Output shape should be [2, 2] (batch dims from indices + remaining data dims)
    expected_output_shape = [2, 2]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ GatherND with batch_dims test passed")
    print(f"  Data shape: {input_tensors[0].shape}")
    print(f"  Indices shape: {input_tensors[1].shape}")
    print(f"  Batch dims: {op_obj.batch_dims}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_4d_with_remaining_dims():
    """Test GatherND with 4D data and remaining dimensions after indexing"""
    print("\n=== Testing GatherND (4D with remaining dims) ===")

    # Create 4D data: [2, 3, 4, 5]
    data_shape = [2, 3, 4, 5]
    # Indices to gather 2D slices: [[0, 1], [1, 2]]
    indices_data = np.array([[0, 1], [1, 2]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    attrs = {'batch_dims': 0}

    op_info = {
        'name': 'test_gather_nd_4d_remaining',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    # indices shape is [2, 2], index_depth=2
    # data shape is [2, 3, 4, 5], so indexed_dims = 2 + 0 = 2
    # remaining_dims = [4, 5]
    # output shape should be [2] + [4, 5] = [2, 4, 5]
    expected_output_shape = [2, 4, 5]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ 4D GatherND with remaining dims test passed")
    print(f"  Data shape: {input_tensors[0].shape}")
    print(f"  Indices shape: {input_tensors[1].shape}")
    print(f"  Output shape: {output_tensors[0].shape}")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_edge_cases():
    """Test GatherND with edge cases"""
    print("\n=== Testing GatherND Edge Cases ===")

    # Test 1: Single element indexing
    print("  Testing single element...")
    data_shape = [5, 3]
    indices_data = np.array([[2, 1]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    op_info = {
        'name': 'test_gather_nd_single',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': {'batch_dims': 0},
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
    assert output_tensors[0].shape == [1]
    print("    ✓ Single element indexing works")

    # Test 2: Full indexing depth
    print("  Testing full indexing depth...")
    data_shape = [2, 3, 4]
    indices_data = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    op_info = {
        'name': 'test_gather_nd_full_depth',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': {'batch_dims': 0},
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
    assert output_tensors[0].shape == [2]  # No remaining dimensions
    print("    ✓ Full indexing depth works")

    print("✓ All edge case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_invalid_inputs():
    """Test GatherND operation with invalid inputs"""
    print("\n=== Testing GatherND Error Cases ===")

    # Test 1: Wrong input count
    try:
        input_tensors = [F._from_shape('data', [3, 4])]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_gather_nd_wrong_inputs',
            'optype': 'GatherND',
            'inList': ['data'],
            'outList': ['output'],
            'attrs': {},
        }

        op_obj = GatherNDOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong input count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong input count: {e}")

    # Test 2: Wrong output count
    try:
        input_tensors = [
            F._from_shape('data', [3, 4]),
            F._from_shape('indices', [2, 2], np_dtype=np.int64)
        ]
        output_tensors = [make_tensor('output1'), make_tensor('output2')]

        op_info = {
            'name': 'test_gather_nd_wrong_outputs',
            'optype': 'GatherND',
            'inList': ['data', 'indices'],
            'outList': ['output1', 'output2'],
            'attrs': {},
        }

        op_obj = GatherNDOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for wrong output count"
    except AssertionError as e:
        print(f"✓ Correctly caught wrong output count: {e}")

    # Test 3: Invalid batch_dims
    try:
        input_tensors = [
            F._from_shape('data', [3, 4]),
            F._from_shape('indices', [2, 2], np_dtype=np.int64)
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_gather_nd_invalid_batch_dims',
            'optype': 'GatherND',
            'inList': ['data', 'indices'],
            'outList': ['output'],
            'attrs': {'batch_dims': -1},  # Invalid negative value
        }

        op_obj = GatherNDOp(op_info)
        assert False, "Should have raised ValueError for invalid batch_dims"
    except ValueError as e:
        print(f"✓ Correctly caught invalid batch_dims: {e}")

    # Test 4: Index depth too large
    try:
        input_tensors = [
            F._from_shape('data', [3, 4]),  # 2D data
            F._from_shape('indices', [2, 3], np_dtype=np.int64)  # Index depth 3 > 2
        ]
        output_tensors = [make_tensor('output')]

        op_info = {
            'name': 'test_gather_nd_index_depth_too_large',
            'optype': 'GatherND',
            'inList': ['data', 'indices'],
            'outList': ['output'],
            'attrs': {'batch_dims': 0},
        }

        op_obj = GatherNDOp(op_info)
        perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)
        assert False, "Should have raised assertion error for index depth too large"
    except AssertionError as e:
        print(f"✓ Correctly caught index depth too large: {e}")

    print("✓ All error case tests passed")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_factory():
    """Test that GatherND operation can be created through SimOpFactory"""
    print("\n=== Testing GatherND Factory Creation ===")

    from ttsim.ops.op import SimOpFactory

    # Test factory creation
    gather_nd_class = SimOpFactory('GatherND')
    assert gather_nd_class == GatherNDOp

    # Create instance through factory
    op_info = {
        'name': 'factory_test_gather_nd',
        'optype': 'GatherND',
        'inList': ['data', 'indices'],
        'outList': ['output'],
        'attrs': {'batch_dims': 0},
    }

    op_obj = gather_nd_class(op_info)
    assert isinstance(op_obj, GatherNDOp)
    assert op_obj.optype == 'GatherND'
    assert op_obj.name == 'factory_test_gather_nd'
    assert op_obj.batch_dims == 0

    print("✓ GatherND factory creation test passed")
    print(f"  Created operation: {op_obj}")


@pytest.mark.unit
@pytest.mark.opunit
def test_gather_nd_complex_scenario():
    """Test GatherND in a complex scenario similar to transformer usage"""
    print("\n=== Testing GatherND Complex Scenario ===")

    # Simulate a scenario like gathering embeddings based on token indices
    # Vocabulary embeddings: [vocab_size, embedding_dim]
    vocab_size = 1000
    embedding_dim = 128
    data_shape = [vocab_size, embedding_dim]

    # Token indices for a sequence: [batch_size, seq_len]
    batch_size = 4
    seq_len = 10
    indices_data = np.random.randint(0, vocab_size, (batch_size, seq_len, 1), dtype=np.int64)

    input_tensors, output_tensors, input_names, output_names = create_gather_nd_test_tensors(
        data_shape, indices_data, batch_dims=0
    )

    attrs = {'batch_dims': 0}

    op_info = {
        'name': 'test_gather_nd_embeddings',
        'optype': 'GatherND',
        'inList': input_names,
        'outList': output_names,
        'attrs': attrs,
    }

    op_obj = GatherNDOp(op_info)

    # Link tensors to operation
    for tensor in input_tensors:
        tensor.op_in = [op_info['name']]
    for tensor in output_tensors:
        tensor.op_out = [op_info['name']]

    perf_stats = op_obj.get_perf_counts(input_tensors, output_tensors)

    # Validate output shape
    # indices shape is [4, 10, 1], so output shape should be [4, 10, 128]
    expected_output_shape = [batch_size, seq_len, embedding_dim]
    assert output_tensors[0].shape == expected_output_shape

    print(f"✓ Complex GatherND scenario test passed")
    print(f"  Embedding table shape: {input_tensors[0].shape}")
    print(f"  Token indices shape: {input_tensors[1].shape}")
    print(f"  Output embeddings shape: {output_tensors[0].shape}")
    print(f"  Performance: {perf_stats['instrs']['gather']} gather ops, {perf_stats['instrs']['index']} index ops")


if __name__ == '__main__':
    # Run all tests manually
    test_gather_nd_factory()
    test_gather_nd_basic_2d()
    test_gather_nd_3d_simple()
    test_gather_nd_with_batch_dims()
    test_gather_nd_4d_with_remaining_dims()
    test_gather_nd_edge_cases()
    test_gather_nd_invalid_inputs()
    test_gather_nd_complex_scenario()

    print("\n🎉 All GatherND operation tests completed successfully!")
