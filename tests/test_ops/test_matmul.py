#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import MatMulOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(shape0, shape1):
    _X0 = np.random.randn(*shape0)
    _X1 = np.random.randn(*shape1)
    _Y = np.matmul(_X0, _X1)
    return list(_Y.shape)

# Test cases
test_name  = 'test_matmul'
test_cases = [
    ("Standard 2D Matrix Multiplication", [3, 4],    [4, 5]   ),
    #("Vector-Matrix Multiplication",      [4],       [4, 5]   ),
    #("Matrix-Vector Multiplication",      [3, 4],    [4]      ),
    ("Batched Matrix Multiplication",     [2, 3, 4], [2, 4, 5]),
    ("Single Element Matrices",           [1, 1],    [1, 1]   ),
    ("Empty Dimension Case",              [3, 0],    [0, 4]   ),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_matmul():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, shape0, shape1) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [
                F._from_shape('X0', shape0, np_dtype=np.float32),
                F._from_shape('X1', shape1, np_dtype=np.float32),
                ]
        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'matmul',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors]
                }
        op_obj = MatMulOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(shape0, shape1)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"
