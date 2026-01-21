#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases - Note: Slice uses out_shape attribute, so we need to set it
# Note: When axes is provided, the implementation creates steps with length dataT.rank(),
# so axes must have the same length as the input rank for the test to pass.
test_name = 'test_slice'
test_cases = [
    ("1D slice", [10], [2], [7], [5]),
    ("2D slice", [5, 6], [1, 2], [4, 5], [3, 3]),
    ("3D slice", [3, 4, 5], [0, 1, 2], [2, 3, 4], [2, 2, 2]),
    # Note: Slice with partial axes is not supported by current implementation
    # because steps tensor is always created with dataT.rank() length
    ("Slice with axes", [5, 6, 7], [1, 2], [4, 5], [0, 2], [3, 6, 3]),
    ("Full slice", [10], [0], [10], [10]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_slice():
    msgw = get_max_test_msg_len(test_cases)
    for tno, test_case in enumerate(test_cases):
        if len(test_case) == 5:
            tmsg, input_shape, starts, ends, expected_shape = test_case
            axes = None
        else:
            tmsg, input_shape, starts, ends, axes, expected_shape = test_case

        op_name = f'{test_name}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('starts', np.array(starts, dtype=np.int64), is_const=True),
            F._from_data('ends', np.array(ends, dtype=np.int64), is_const=True)
        ]

        if axes is not None:
            i_tensors.append(F._from_data('axes', np.array(axes, dtype=np.int64), is_const=True))

        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Slice',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'out_shape': expected_shape}  # Slice requires out_shape
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
test_name_errors = 'test_slice_errors'
test_cases_errors = [
    ("Mismatched starts/ends lengths", [10], [2, 3], [7]),
    ("Invalid axes out of bounds", [5, 6], [1], [4], [5]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_slice_errors():
    """Test Slice with invalid inputs that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, test_case in enumerate(test_cases_errors):
        if len(test_case) == 4:
            tmsg, input_shape, starts, ends = test_case
            axes = None
        else:
            tmsg, input_shape, starts, ends, axes = test_case
        
        op_name = f'{test_name_errors}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('starts', np.array(starts, dtype=np.int64), is_const=True),
            F._from_data('ends', np.array(ends, dtype=np.int64), is_const=True)
        ]
        
        if axes is not None:
            i_tensors.append(F._from_data('axes', np.array(axes, dtype=np.int64), is_const=True))
        
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Slice',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)")
