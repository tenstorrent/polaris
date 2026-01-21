#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(shapes, axis):
    """Reference implementation using NumPy concatenate"""
    arrays = [np.random.randn(*shape) for shape in shapes]
    result = np.concatenate(arrays, axis=axis)
    return list(result.shape)

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases
test_name = 'test_concat'
test_cases = [
    ("2D tensors along axis 0", [[2, 3], [4, 3]], 0, [6, 3]),
    ("2D tensors along axis 1", [[2, 3], [2, 4]], 1, [2, 7]),
    ("Multiple inputs axis 0", [[2, 3], [2, 3], [2, 3]], 0, [6, 3]),
    ("3D tensors along axis 2", [[2, 3, 4], [2, 3, 5]], 2, [2, 3, 9]),
    ("4D tensors along axis 1", [[1, 2, 3, 4], [1, 2, 3, 4]], 1, [1, 4, 3, 4]),
    ("Single element tensors", [[1, 1], [1, 1]], 0, [2, 1]),
    ("Zero-sized dimensions", [[0, 3], [2, 3]], 0, [2, 3]),
    ("Empty dimension on concat axis", [[2, 0], [2, 3]], 1, [2, 3]),
    ("Negative axis -1", [[2, 3], [2, 4]], -1, [2, 7]),
    ("Negative axis -2", [[2, 3], [4, 3]], -2, [6, 3]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_concat():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shapes, axis, expected_shape) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [F._from_shape(f'X{i}', shape, np_dtype=np.float32) 
                     for i, shape in enumerate(input_shapes)]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Concat',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': axis}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shapes, axis)

        if inf_shape == ref_shape and inf_shape == expected_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape} (expected {expected_shape})"

# Error test cases
test_name_errors = 'test_concat_errors'
test_cases_errors = [
    ("Mismatched ranks", [[2, 3], [2, 3, 4]], 0),
    ("Mismatched non-concat dimensions", [[2, 3], [2, 4]], 0),
    ("Invalid axis", [[2, 3], [2, 3]], 5),
    ("Negative axis out of bounds", [[2, 3], [2, 3]], -5),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_concat_errors():
    """Test Concat with incompatible shapes that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shapes, axis) in enumerate(test_cases_errors):
        op_name = f'{test_name_errors}_{tno}'
        i_tensors = [F._from_shape(f'X{i}', shape, np_dtype=np.float32) 
                     for i, shape in enumerate(input_shapes)]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Concat',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'axis': axis}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
        print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)")
