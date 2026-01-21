#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(input_shape, pads):
    """Reference implementation - pads only last 2 dimensions"""
    output_shape = list(input_shape)
    rank = len(input_shape)
    # Pad only last 2 dimensions - pads format: [pad_before_n-1, pad_before_n, pad_after_n-1, pad_after_n]
    pad_before = [0] * (rank - 2) + pads[:2]
    pad_after = [0] * (rank - 2) + pads[2:]
    for i in range(rank):
        output_shape[i] = input_shape[i] + pad_before[i] + pad_after[i]
    return output_shape

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases
# Note: The implementation only pads the last 2 dimensions and requires exactly 4 pad values
# Format: [pad_before_dim_n-1, pad_before_dim_n, pad_after_dim_n-1, pad_after_dim_n]
test_name = 'test_pad'
test_cases = [
    ("2D tensor basic padding", [3, 4], [1, 1, 1, 1], [5, 6]),
    ("3D tensor padding (pads last 2 dims)", [2, 3, 4], [1, 1, 1, 1], [2, 5, 6]),
    ("4D tensor padding (pads last 2 dims)", [1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 5, 6]),
    ("Zero padding", [3, 4], [0, 0, 0, 0], [3, 4]),
    ("Large padding", [1, 1], [10, 10, 10, 10], [21, 21]),
    ("Asymmetric padding", [3, 4], [1, 2, 3, 4], [7, 10]),  # [3+1+3, 4+2+4]
]

@pytest.mark.unit
@pytest.mark.opunit
def test_pad():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape, pads, expected_shape) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('pads', np.array(pads, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'mode': 'constant', 'value': 0}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shape, pads)

        if inf_shape == ref_shape and inf_shape == expected_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape} (expected {expected_shape})"

# Error test cases
test_name_errors = 'test_pad_errors'
test_cases_errors = [
    ("Invalid pad length (not 4)", [3, 4], [1, 1, 1]),
    ("Invalid pad length (5 elements)", [3, 4], [1, 1, 1, 1, 1]),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_pad_errors():
    """Test Pad with invalid inputs that should raise errors"""
    msgw = get_max_test_msg_len(test_cases_errors)
    for tno, (tmsg, input_shape, pads) in enumerate(test_cases_errors):
        op_name = f'{test_name_errors}_{tno}'
        i_tensors = [
            F._from_shape('X', input_shape, np_dtype=np.float32),
            F._from_data('pads', np.array(pads, dtype=np.int64), is_const=True)
        ]
        o_tensors = [make_tensor('Y')]
        op_info = {
            'name': op_name,
            'optype': 'Pad',
            'inList': [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs': {'mode': 'constant', 'value': 0}
        }
        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        # These should raise exceptions during shape inference
        with pytest.raises((ValueError, AssertionError)):
            op_obj.get_perf_counts(i_tensors, o_tensors)
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS (raised exception as expected)")
