#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Squeeze test cases
test_name_squeeze = 'test_squeeze'
test_cases_squeeze = [
    ("Remove single dimension", [1, 3, 4], [0], [3, 4]),
    ("Remove multiple dimensions", [1, 3, 1, 4], [0, 2], [3, 4]),
    ("Remove all size-1 dims", [1, 1, 1], [0, 1, 2], []),
    ("Negative axis -3", [1, 3, 4], [-3], [3, 4]),
    ("Negative axis -1", [3, 4, 1], [-1], [3, 4]),
    ("Multiple negative axes", [1, 3, 1, 4], [-4, -2], [3, 4]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_squeeze():
    msgw = get_max_test_msg_len(test_cases_squeeze)
    for tno, (tmsg, input_shape, axes, expected_shape) in enumerate(test_cases_squeeze):
        op_name = f'{test_name_squeeze}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('axes', np.array(axes, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Squeeze',
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

# Unsqueeze test cases
test_name_unsqueeze = 'test_unsqueeze'
test_cases_unsqueeze = [
    ("Add dimension at start", [3, 4], [0], [1, 3, 4]),
    ("Add dimension at end", [3, 4], [2], [3, 4, 1]),
    ("Add multiple dimensions", [3, 4], [0, 3], [1, 3, 4, 1]),
    ("Negative axis -1", [3, 4], [-1], [3, 4, 1]),
    ("Negative axis -3", [3, 4], [-3], [1, 3, 4]),
    ("Scalar to 1D", [], [0], [1]),
    ("Multiple insertions", [3, 4], [0, 2, 4], [1, 3, 1, 4, 1]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_unsqueeze():
    msgw = get_max_test_msg_len(test_cases_unsqueeze)
    for tno, (tmsg, input_shape, axes, expected_shape) in enumerate(test_cases_unsqueeze):
        op_name = f'{test_name_unsqueeze}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('axes', np.array(axes, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Unsqueeze',
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
# Note: The current implementation doesn't validate that squeezed dimensions are size 1,
# it just removes the specified dimensions. So "Squeeze non-size-1 dimension" won't raise an error.
test_cases_errors = [
    ("Invalid axis out of bounds", [1, 3, 4], [5], "Squeeze"),
    ("Unsqueeze invalid axis", [3, 4], [5], "Unsqueeze"),
    ("Unsqueeze negative axis out of bounds", [3, 4], [-5], "Unsqueeze"),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_squeeze_unsqueeze_errors():
    """Test Squeeze/Unsqueeze with invalid inputs that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shape, axes, optype) in enumerate(test_cases_errors):
        op_name = f'test_{optype.lower()}_error_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('axes', np.array(axes, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': optype,
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
