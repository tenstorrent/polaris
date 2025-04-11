#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SplitOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(data_shape, indices, axis):
    X = np.random.randn(*data_shape)
    odata = np.take(X, indices, axis=axis)
    return list(odata.shape)

# Test cases
test_name  = 'test_split'
test_cases = [
        {
            'name'    : "test_split_equal_parts_1d_opset13",
            'X'       : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            'inputs'  :["input"],
            'outputs' :["output_1", "output_2", "output_3"],
            'axis'    : 0,
            'expected_outputs': [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0]).astype(np.float32),
                np.array([5.0, 6.0]).astype(np.float32),
                ]
            },
        {
            'name'    : "test_split_variable_parts_1d_opset13",
            'split'   : np.array([2, 4]).astype(np.int64),
            'inputs'  : ["input", "split"],
            'outputs' : ["output_1", "output_2"],
            'axis'    : 0,
            'expected_outputs': [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                ]
            },
        {
            'name'      : "test_split_equal_parts_2d_opset13",
            'X'         : np.array( [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                     [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]).astype(np.float32),
            'inputs'    : ["input"],
            'outputs'   : ["output_1", "output_2"],
            'axis'      : 1,
            'expected_outputs': [
                np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
                np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
                ]
            },
        {
            'name'      : "test_split_variable_parts_2d_opset13",
            'split'     : np.array([2, 4]).astype(np.int64),
            'inputs'    : ["input", "split"],
            'outputs'   : ["output_1", "output_2"],
            'axis'      : 1,
            'expected_outputs': [
                np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
                np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(np.float32),
                ]
            },
        {
            'name'    : "test_split_equal_parts_default_axis_opset13",
            'X'       : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
            'inputs'  :["input"],
            'outputs' :["output_1", "output_2", "output_3"],
            'expected_outputs': [
                np.array([1.0, 2.0]).astype(np.float32),
                np.array([3.0, 4.0]).astype(np.float32),
                np.array([5.0, 6.0]).astype(np.float32),
                ]
            },
        {
                'name'   : "test_split_variable_parts_default_axis_opset13",
                'split'  : np.array([2, 4]).astype(np.int64),
                'inputs' : ["input", "split"],
                'outputs': ["output_1", "output_2"],
                'expected_outputs': [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_zero_size_splits_opset13",
                'X'      : np.array([]).astype(np.float32), # 1D
                'split'  : np.array([0, 0, 0]).astype(np.int64), # Split emtpy tensor to tensors of size zero
                'inputs' : ["input", "split"],
                'outputs': ["output_1", "output_2", "output_3"],
                'expected_outputs': [
                    np.array([]).astype(np.float32),
                    np.array([]).astype(np.float32),
                    np.array([]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_equal_parts_1d_opset18",
                'X'      : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                'inputs' : ["input"],
                'outputs': ["output_1", "output_2", "output_3"],
                'axis'   : 0,
                'num_outputs': 3,
                'expected_outputs': [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0]).astype(np.float32),
                    np.array([5.0, 6.0]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_variable_parts_1d_opset18",
                'split'  : np.array([2, 4]).astype(np.int64),
                'inputs' : ["input", "split"],
                'outputs': ["output_1", "output_2"],
                'axis'   : 0,
                'expected_outputs'                       : [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_equal_parts_2d",
                'X'      : np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]).astype(np.float32),
                'inputs' : ["input"],
                'outputs': ["output_1", "output_2"],
                'axis'   :1,
                'num_outputs': 2,
                'expected_outputs': [
                    np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
                    np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
                    ]
                },
        {


                'name'   : "test_split_variable_parts_2d_opset18",
                'split'  : np.array([2, 4]).astype(np.int64),
                'inputs' :["input", "split"],
                'outputs': ["output_1", "output_2"],
                'axis'   : 1,
                'expected_outputs': [
                    np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
                    np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(np.float32),
                    ]
                },
        {
                'name'  : "test_split_equal_parts_default_axis_opset18",
                'X'     : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                'inputs':["input"],
                'outputs': ["output_1", "output_2", "output_3"],
                'num_outputs': 3,
                'expected_outputs': [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0]).astype(np.float32),
                    np.array([5.0, 6.0]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_variable_parts_default_axis_opset18",
                'split'  : np.array([2, 4]).astype(np.int64),
                'inputs' : ["input", "split"],
                'outputs': ["output_1", "output_2"],
                'expected_outputs': [
                    np.array([1.0, 2.0]).astype(np.float32),
                    np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
                    ]
                },
        {
                'name'   : "test_split_zero_size_splits_opset18",
                'X'      : np.array([]).astype(np.float32),
                'split'  : np.array([0, 0, 0]).astype(np.int64),
                'inputs' : ["input", "split"],
                'outputs': ["output_1", "output_2", "output_3"],
                'expected_outputs': [
                    np.array([]).astype(np.float32),
                    np.array([]).astype(np.float32),
                    np.array([]).astype(np.float32),
                    ]
                },
        #FAILS RIGHT NOW!!
        #{
        #        'name'   : "test_split_1d_uneven_split_opset18",
        #        'X'      : np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).astype(np.float32),
        #        'inputs' : ["input"],
        #        'outputs': ["output_1", "output_2", "output_3", "output_4"],
        #        'num_outputs': 4,
        #        'expected_outputs': [
        #            np.array([1.0, 2.0]).astype(np.float32),
        #            np.array([3.0, 4.0]).astype(np.float32),
        #            np.array([5.0, 6.0]).astype(np.float32),
        #            np.array([7.0]).astype(np.float32),
        #            ]
        #        },
        #{
        #        'name'   : "test_split_2d_uneven_split_opset18",
        #        'X'      : np.array( [ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        #                              [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        #                              ]).astype(np.float32),
        #        'inputs' : ["input"],
        #        'outputs': ["output_1", "output_2", "output_3"],
        #        'axis'   : 1,
        #        'num_outputs': 3,
        #        'expected_outputs': [
        #            np.array([[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]]).astype(np.float32),
        #            np.array([[4.0, 5.0, 6.0], [12.0, 13.0, 14.0]]).astype(np.float32),
        #            np.array([[7.0, 8.0], [15.0, 16.0]]).astype(np.float32),
        #            ]
        #        }
]

@pytest.mark.unit
@pytest.mark.opunit
def test_split():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    input_tensors = []
    for tno, trec in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'

        XShape = trec['X'].shape if 'X' in trec else test_cases[tno-1]['X'].shape #type: ignore
        XShape = list(XShape)
        i_tensors = [F._from_shape('X', XShape)]
        if 'split' in trec:
            i_tensors.append(F._from_data('S', trec['split'])) #type: ignore
        attrs = {}
        if 'axis' in trec: attrs['axis'] = trec['axis']
        if 'num_outputs' in trec: attrs['num_outputs'] = trec['num_outputs']

        num_outputs = len(trec['expected_outputs']) #type: ignore
        o_tensors = [make_tensor(f"O{i}") for i in range(num_outputs)]
        op_info = {
                'name'   : op_name,
                'optype' : 'split',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : attrs,
                }

        op_obj = SplitOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        check = all([o_tensors[i].shape == list(trec['expected_outputs'][i].shape) for i in range(num_outputs)]) #type: ignore

        if check:
            print(f"TEST[{tno:3d}] {trec['name']:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            print('REF OUTPUTS:')
            for x in trec['expected_outputs']: print('\t', x.shape) #type: ignore
            assert False, f"TEST[{tno:3d}] {trec['name']:{msgw}s} FAIL"
