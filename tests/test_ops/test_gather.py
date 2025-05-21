#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import GatherOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(data_shape, indices, axis):
    X = np.random.randn(*data_shape)
    odata = np.take(X, indices, axis=axis)
    return list(odata.shape)

# Test cases
test_name  = 'test_gather'
test_cases = [
        ("Gather from 2D along axis 0", [3, 4],    [0, 2],           0),
        ("Gather from 2D along axis 1", [3, 4],    [1, 3],           1),
        ("Gather from 3D along axis 0", [2, 3, 4], [1],              0),
        ("Gather from 3D along axis 2", [2, 3, 4], [[0, 1], [2, 3]], 2),
        ("Gather with empty indices",   [3, 4],    [],               0),

        #from onnx.backend.test
        ("test_gather_0",                [5, 4, 3, 2], [0, 1, 3],    0),
        ("test_gather_1",                [5, 4, 3, 2], [0, 1, 3],    1),
        ("test_gather_2d_indices",       [3, 3],       [[0, 2]],     1),
        ("test_gather_negative_indices", [10],         [0, -9, -10], 0),
        ]

@pytest.mark.unit
@pytest.mark.opunit
def test_gather():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, data_shape, indices, axis) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [F._from_shape('X', data_shape, np_dtype=np.float32),
                     F._from_data('I', np.array(indices))]

        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'matmul',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : {'axis': axis},
                }
        op_obj = GatherOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(data_shape, indices, axis)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"
