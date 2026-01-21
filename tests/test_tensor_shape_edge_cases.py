#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

@pytest.mark.unit
@pytest.mark.opunit
def test_scalar_tensors():
    """Test scalar tensors (rank 0)"""
    # Create scalar tensor
    scalar = make_tensor('scalar')
    scalar.shape = []
    assert scalar.rank() == 0, "Scalar tensor should have rank 0"
    assert scalar.nelems() == 1, "Scalar tensor should have 1 element"
    assert scalar.check_shape(), "Scalar tensor should have valid shape"

    # Test operations on scalars
    scalar1 = F._from_shape('s1', [], np_dtype=np.float32)
    scalar2 = F._from_shape('s2', [], np_dtype=np.float32)

    # Element-wise operation on scalars
    i_tensors = [scalar1, scalar2]
    o_tensors = [make_tensor('Y')]
    op_info = {
        'name': 'scalar_add',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['scalar_add']
    for x in o_tensors: x.op_out = ['scalar_add']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [], "Scalar + Scalar should produce scalar"

    # Broadcasting with scalars
    scalar = F._from_shape('scalar', [], np_dtype=np.float32)
    tensor = F._from_shape('tensor', [3, 4, 5], np_dtype=np.float32)
    i_tensors = [scalar, tensor]
    o_tensors = [make_tensor('Y2')]
    op_info = {
        'name': 'scalar_broadcast',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['scalar_broadcast']
    for x in o_tensors: x.op_out = ['scalar_broadcast']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [3, 4, 5], "Scalar should broadcast to tensor shape"

@pytest.mark.unit
@pytest.mark.opunit
def test_zero_sized_dimensions():
    """Test zero-sized dimensions"""
    # Tensor with zero dimension
    tensor = F._from_shape('zero', [0], np_dtype=np.float32)
    assert tensor.rank() == 1, "Tensor with [0] should have rank 1"
    assert tensor.nelems() == 0, "Tensor with [0] should have 0 elements"

    # Tensor with zero in middle
    tensor = F._from_shape('zero_mid', [3, 0, 4], np_dtype=np.float32)
    assert tensor.nelems() == 0, "Tensor with zero dimension should have 0 elements"

    # MatMul with zero-sized dimensions
    i_tensors = [
        F._from_shape('X0', [3, 0], np_dtype=np.float32),
        F._from_shape('X1', [0, 4], np_dtype=np.float32),
    ]
    o_tensors = [make_tensor('Y')]
    op_info = {
        'name': 'matmul_zero',
        'optype': 'MatMul',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['matmul_zero']
    for x in o_tensors: x.op_out = ['matmul_zero']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [3, 4], "MatMul with zero dims should produce [3, 4]"

    # Element-wise with zero-sized dimensions
    # Note: In NumPy, broadcasting [0, 3] with [1, 3] results in [0, 3], not [1, 3]
    # Zero-sized dimensions remain zero-sized in the result
    i_tensors = [
        F._from_shape('X0', [0, 3], np_dtype=np.float32),
        F._from_shape('X1', [1, 3], np_dtype=np.float32),
    ]
    o_tensors = [make_tensor('Y2')]
    op_info = {
        'name': 'add_zero',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['add_zero']
    for x in o_tensors: x.op_out = ['add_zero']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [0, 3], "Broadcasting [0,3] with [1,3] should produce [0,3] (zero-sized dims remain zero)"

@pytest.mark.unit
@pytest.mark.opunit
def test_high_dimensional_tensors():
    """Test high-dimensional tensors (7D+)"""
    # 7D tensor
    shape_7d = [2, 3, 4, 5, 6, 7, 8]
    tensor = F._from_shape('tensor_7d', shape_7d, np_dtype=np.float32)
    assert tensor.rank() == 7, "7D tensor should have rank 7"
    assert tensor.check_shape(), "7D tensor should have valid shape"

    # 8D tensor
    shape_8d = [1, 1, 1, 1, 1, 1, 1, 1]
    tensor = F._from_shape('tensor_8d', shape_8d, np_dtype=np.float32)
    assert tensor.rank() == 8, "8D tensor should have rank 8"

    # Operations on high-dimensional tensors
    i_tensors = [
        F._from_shape('X0', [1, 1, 1, 1, 2, 3, 4], np_dtype=np.float32),
        F._from_shape('X1', [2, 1, 1, 1, 1, 3, 4], np_dtype=np.float32),
    ]
    o_tensors = [make_tensor('Y')]
    op_info = {
        'name': 'high_dim_add',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['high_dim_add']
    for x in o_tensors: x.op_out = ['high_dim_add']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [2, 1, 1, 1, 2, 3, 4], "High-dim broadcasting should work"

@pytest.mark.unit
@pytest.mark.opunit
def test_very_large_shapes():
    """Test very large shapes"""
    # Large dimensions
    large_shape = [1000000, 1000]
    tensor = F._from_shape('large', large_shape, np_dtype=np.float32)
    assert tensor.check_shape(), "Large shape should be valid"
    assert tensor.rank() == 2, "Large tensor should have rank 2"
    # Note: nelems() would be very large, but shape inference should work

    # Very large single dimension
    very_large = [10000000]
    tensor = F._from_shape('very_large', very_large, np_dtype=np.float32)
    assert tensor.check_shape(), "Very large dimension should be valid"

@pytest.mark.unit
@pytest.mark.opunit
def test_none_shapes():
    """Test tensors with None shapes"""
    tensor = make_tensor('none_shape')
    tensor.shape = None
    assert not tensor.check_shape(), "None shape should fail check_shape()"

    # Operations that should fail with None shapes
    tensor1 = make_tensor('t1')
    tensor1.shape = None
    tensor2 = F._from_shape('t2', [3, 4], np_dtype=np.float32)

    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [tensor1, tensor2]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'none_shape_op',
            'optype': 'Add',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['none_shape_op']
        for x in o_tensors: x.op_out = ['none_shape_op']
        op_obj.get_perf_counts(i_tensors, o_tensors)

@pytest.mark.unit
@pytest.mark.opunit
def test_special_value_handling():
    """Test special value handling"""
    # Boundary values
    tensor = F._from_shape('boundary', [1], np_dtype=np.float32)
    assert tensor.check_shape(), "Shape with 1 should be valid"

    # All ones
    tensor = F._from_shape('all_ones', [1, 1, 1, 1], np_dtype=np.float32)
    assert tensor.check_shape(), "Shape with all ones should be valid"

    # Mixed small and large
    tensor = F._from_shape('mixed', [1, 1000, 1, 100], np_dtype=np.float32)
    assert tensor.check_shape(), "Mixed shape should be valid"
