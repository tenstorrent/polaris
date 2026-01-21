#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(input_shape, repeats):
    """Reference implementation using NumPy tile"""
    data = np.random.randn(*input_shape)
    result = np.tile(data, repeats)
    return list(result.shape)

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases
test_name = 'test_tile'
test_cases = [
    ("1D tiling", [3], [2], [6]),
    ("2D tiling", [2, 3], [2, 3], [4, 9]),
    ("3D tiling", [1, 2, 3], [2, 2, 2], [2, 4, 6]),
    ("Single dimension repeat", [2, 3], [1, 5], [2, 15]),
    ("All dimensions repeat", [2, 3], [3, 3], [6, 9]),
    ("High-dimensional tiling", [1, 1, 2, 3], [2, 2, 2, 2], [2, 2, 4, 6]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_tile():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape, repeats, expected_shape) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('repeats', np.array(repeats, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Tile',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors]
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shape, repeats)

        if inf_shape == ref_shape and inf_shape == expected_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape} (expected {expected_shape})"

# Error test cases
test_name_errors = 'test_tile_errors'
test_cases_errors = [
    ("Mismatched repeat length", [2, 3], [2]),
    ("Zero repeats", [2, 3], [0, 3]),
    ("Invalid repeat length", [2, 3], [2, 3, 4]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_tile_errors():
    """Test Tile with invalid inputs that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shape, repeats) in enumerate(test_cases_errors):
        op_name = f'{test_name_errors}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('repeats', np.array(repeats, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Tile',
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
