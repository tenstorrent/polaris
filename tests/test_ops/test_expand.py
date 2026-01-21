#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases
test_name = 'test_expand'
test_cases = [
    ("1D to 2D", [3], [2, 3], [2, 3]),
    ("2D to 3D", [1, 3], [2, 1, 3], [2, 1, 3]),
    ("Broadcasting expansion", [1, 3], [4, 3], [4, 3]),
    ("No expansion needed", [2, 3], [2, 3], [2, 3]),
    ("Multiple dimension expansion", [1, 1, 3], [2, 4, 3], [2, 4, 3]),
    ("Prepend dimensions", [3], [1, 1, 3], [1, 1, 3]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_expand():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape, target_shape, expected_shape) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('shape', np.array(target_shape, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Expand',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape

        if inf_shape == expected_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {expected_shape}"

# Error test cases
test_name_errors = 'test_expand_errors'
test_cases_errors = [
    ("Incompatible expansion", [2, 3], [4, 5]),
    ("Cannot expand non-1 dimension", [2, 3], [2, 4]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_expand_errors():
    """Test Expand with invalid inputs that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shape, target_shape) in enumerate(test_cases_errors):
        op_name = f'{test_name_errors}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('shape', np.array(target_shape, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Expand',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)")
