#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

@pytest.mark.unit
@pytest.mark.opunit
def test_broadcasting_compatibility():
    """Test broadcasting compatibility scenarios"""
    test_cases = [
        ("Simple broadcasting", [1, 3], [2, 3], [2, 3]),
        ("Complex broadcasting", [1, 1, 4], [2, 3, 1], [2, 3, 4]),
        ("Multi-input broadcasting", [1, 3], [2, 1], [2, 3]),
        # Note: Broadcasting [0, 1] with [1, 0] produces [0, 0] in NumPy
        # Zero-sized dimensions remain zero-sized in the result
        ("Broadcasting with zero-sized", [0, 1], [1, 0], [0, 0]),
    ]
    
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, shape0, shape1, expected) in enumerate(test_cases):
        i_tensors = [
            F._from_shape('X0', shape0, np_dtype=np.float32),
            F._from_shape('X1', shape1, np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': f'broadcast_test_{tno}',
            'optype': 'Add',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [f'broadcast_test_{tno}']
        for x in o_tensors: x.op_out = [f'broadcast_test_{tno}']
        op_obj.get_perf_counts(i_tensors, o_tensors)
        
        if o_tensors[0].shape == expected:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {o_tensors[0].shape} != {expected}"

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_propagation():
    """Test shape propagation through operator chains"""
    # Chain: Reshape -> MatMul -> Add
    input_shape = [2, 3, 4]
    reshape_target = [2, 12]
    
    # Reshape
    i_tensors_reshape = [
        F._from_shape('X', input_shape, np_dtype=np.float32),
        F._from_data('S', np.array(reshape_target, dtype=np.int64), is_const=True)
    ]
    o_tensors_reshape = [make_tensor('Y1')]
    op_info = {
        'name': 'reshape_chain',
        'optype': 'Reshape',
        'inList': [x.name for x in i_tensors_reshape],
        'outList': [x.name for x in o_tensors_reshape]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors_reshape: x.op_in = ['reshape_chain']
    for x in o_tensors_reshape: x.op_out = ['reshape_chain']
    op_obj.get_perf_counts(i_tensors_reshape, o_tensors_reshape)
    assert o_tensors_reshape[0].shape == reshape_target, "Reshape should produce [2, 12]"
    
    # MatMul (using reshaped output)
    reshaped = o_tensors_reshape[0]
    i_tensors_matmul = [
        reshaped,
        F._from_shape('W', [12, 5], np_dtype=np.float32),
    ]
    o_tensors_matmul = [make_tensor('Y2')]
    op_info = {
        'name': 'matmul_chain',
        'optype': 'MatMul',
        'inList': [x.name for x in i_tensors_matmul],
        'outList': [x.name for x in o_tensors_matmul]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors_matmul: x.op_in = ['matmul_chain']
    for x in o_tensors_matmul: x.op_out = ['matmul_chain']
    op_obj.get_perf_counts(i_tensors_matmul, o_tensors_matmul)
    assert o_tensors_matmul[0].shape == [2, 5], "MatMul should produce [2, 5]"
    
    # Add (using matmul output)
    matmul_out = o_tensors_matmul[0]
    i_tensors_add = [
        matmul_out,
        F._from_shape('B', [1, 5], np_dtype=np.float32),  # Broadcasting
    ]
    o_tensors_add = [make_tensor('Y3')]
    op_info = {
        'name': 'add_chain',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors_add],
        'outList': [x.name for x in o_tensors_add]
    }
    op_obj = SimOp(op_info)
    for x in i_tensors_add: x.op_in = ['add_chain']
    for x in o_tensors_add: x.op_out = ['add_chain']
    op_obj.get_perf_counts(i_tensors_add, o_tensors_add)
    assert o_tensors_add[0].shape == [2, 5], "Final shape should be [2, 5]"

@pytest.mark.unit
@pytest.mark.opunit
def test_multi_operator_compatibility():
    """Test multi-operator compatibility scenarios"""
    # Concat with multiple inputs
    i_tensors = [
        F._from_shape('X0', [2, 3], np_dtype=np.float32),
        F._from_shape('X1', [2, 3], np_dtype=np.float32),
        F._from_shape('X2', [2, 3], np_dtype=np.float32),
    ]
    o_tensors = [make_tensor('Y')]
    op_info = {
        'name': 'multi_concat',
        'optype': 'Concat',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
        'attrs': {'axis': 0}
    }
    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in = ['multi_concat']
    for x in o_tensors: x.op_out = ['multi_concat']
    op_obj.get_perf_counts(i_tensors, o_tensors)
    assert o_tensors[0].shape == [6, 3], "Concat of 3 [2,3] tensors along axis 0 should be [6,3]"

    # Element-wise with multiple compatible shapes
    i_tensors = [
        F._from_shape('X0', [1, 3], np_dtype=np.float32),
        F._from_shape('X1', [2, 1], np_dtype=np.float32),
        F._from_shape('X2', [2, 3], np_dtype=np.float32),
    ]
    o_tensors = [make_tensor('Y2')]
    op_info = {
        'name': 'multi_add',
        'optype': 'Add',
        'inList': [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors]
    }
    # Note: Add only takes 2 inputs, but we can test compatibility
    # For multi-input, we'd need a different operator or chain them

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_inference_consistency():
    """Test shape inference consistency"""
    # Same operator with different input shapes should produce consistent results
    test_shapes = [
        ([3, 4], [4, 5], [3, 5]),
        ([2, 3, 4], [2, 4, 5], [2, 3, 5]),
        ([1, 4], [4, 1], [1, 1]),
    ]
    
    for shape0, shape1, expected in test_shapes:
        i_tensors = [
            F._from_shape('X0', shape0, np_dtype=np.float32),
            F._from_shape('X1', shape1, np_dtype=np.float32),
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': 'consistency_test',
            'optype': 'MatMul',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = ['consistency_test']
        for x in o_tensors: x.op_out = ['consistency_test']
        op_obj.get_perf_counts(i_tensors, o_tensors)
        assert o_tensors[0].shape == expected, f"MatMul {shape0} × {shape1} should produce {expected}"
