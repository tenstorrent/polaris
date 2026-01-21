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
def test_invalid_shape_operations():
    """Test operations with invalid shapes"""
    # Test with None shape
    tensor = make_tensor('invalid')
    tensor.shape = None
    assert not tensor.check_shape(), "None shape should fail check_shape()"
    
    # Test with invalid shape elements
    tensor = make_tensor('invalid2')
    tensor.shape = [1, "invalid", 3]
    assert not tensor.check_shape(), "Shape with string should fail check_shape()"
    
    # Test operation on tensor with None shape
    tensor1 = make_tensor('t1')
    tensor1.shape = None
    tensor2 = F._from_shape('t2', [3, 4], np_dtype=np.float32)
    
    # This should fail when trying to use tensor1 in an operation
    with pytest.raises((ValueError, AssertionError)):
        op_info = {
            'name': 'test_op',
            'optype': 'Add',
            'inList': [tensor1.name, tensor2.name],
            'outList': ['Y']
        }
        op_obj = SimOp(op_info)
        o_tensors = [make_tensor('Y')]
        op_obj.get_perf_counts([tensor1, tensor2], o_tensors)

@pytest.mark.unit
@pytest.mark.opunit
def test_operator_shape_errors():
    """Test operator-specific shape errors"""
    # MatMul: Incompatible matrix dimensions
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [
            F._from_shape('X0', [3, 4], np_dtype=np.float32),
            F._from_shape('X1', [3, 5], np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'matmul_error',
            'optype': 'MatMul',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['matmul_error']
        for x in o_tensors: x.op_out = ['matmul_error']
        op_obj.get_perf_counts(i_tensors, o_tensors)
    
    # Concat: Mismatched ranks
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [
            F._from_shape('X0', [2, 3], np_dtype=np.float32),
            F._from_shape('X1', [2, 3, 4], np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'concat_error',
            'optype': 'Concat',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': 0}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['concat_error']
        for x in o_tensors: x.op_out = ['concat_error']
        op_obj.get_perf_counts(i_tensors, o_tensors)
    
    # Reshape: Invalid target shape (size mismatch)
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [
            F._from_shape('X', [2, 3, 4], np_dtype=np.float32),
            F._from_data('S', np.array([2, 3, 5], dtype=np.int64), is_const=True),  # Wrong size
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'reshape_error',
            'optype': 'Reshape',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['reshape_error']
        for x in o_tensors: x.op_out = ['reshape_error']
        op_obj.get_perf_counts(i_tensors, o_tensors)
    
    # Transpose: Invalid permutation indices
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [F._from_shape('X', [2, 3, 4], np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'transpose_error',
            'optype': 'Transpose',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'perm': [0, 1, 2, 3]}  # Too many indices
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['transpose_error']
        for x in o_tensors: x.op_out = ['transpose_error']
        op_obj.get_perf_counts(i_tensors, o_tensors)
    
    # Element-wise: Non-broadcastable shapes
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [
            F._from_shape('X0', [3, 4], np_dtype=np.float32),
            F._from_shape('X1', [5, 6], np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'add_error',
            'optype': 'Add',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['add_error']
        for x in o_tensors: x.op_out = ['add_error']
        op_obj.get_perf_counts(i_tensors, o_tensors)

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_inference_function_errors():
    """Test shape inference function errors"""
    # Missing required attributes (e.g., axis for Concat)
    with pytest.raises((ValueError, AssertionError, KeyError)):
        i_tensors = [
            F._from_shape('X0', [2, 3], np_dtype=np.float32),
            F._from_shape('X1', [2, 3], np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'concat_no_axis',
            'optype': 'Concat',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            # Missing 'axis' attribute
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['concat_no_axis']
        for x in o_tensors: x.op_out = ['concat_no_axis']
        op_obj.get_perf_counts(i_tensors, o_tensors)
    
    # Invalid attribute values (negative axis out of bounds)
    with pytest.raises((ValueError, AssertionError)):
        i_tensors = [
            F._from_shape('X0', [2, 3], np_dtype=np.float32),
            F._from_shape('X1', [2, 3], np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'concat_invalid_axis',
            'optype': 'Concat',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': -5}  # Out of bounds
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['concat_invalid_axis']
        for x in o_tensors: x.op_out = ['concat_invalid_axis']
        op_obj.get_perf_counts(i_tensors, o_tensors)

@pytest.mark.unit
@pytest.mark.opunit
def test_edge_case_errors():
    """Test edge cases that should fail"""
    # Operations on empty tensors where not supported
    # (Most operations should handle empty tensors, but some might not)
    
    # Operations with zero-sized dimensions where invalid
    # (Most operations handle zero-sized dimensions, but verify)
    
    # High-dimensional tensors exceeding limits (if any)
    # Test with very high rank (if there's a limit)
    tensor = F._from_shape('high_dim', [1] * 10, np_dtype=np.float32)
    assert tensor.check_shape(), "High-dimensional tensor should be valid"
    assert tensor.rank() == 10, "Rank should be 10"
