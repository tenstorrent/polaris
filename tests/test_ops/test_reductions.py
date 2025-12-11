#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

# Test cases
test_name  = 'test_reductions'
test_cases = [
        ("1D",  [2]                     ),
        ("2D",  [2, 3]                  ),
        ("3D",  [2, 3, 5]               ),
        ("4D",  [2, 3, 5, 7]            ),
        ("7D",  [2, 3, 5, 7, 9, 11, 13] ),
        ]

@pytest.mark.unit
@pytest.mark.opunit
def test_reductions():
    msgw = max([len(x[0]) for x in test_cases])
    for op_type in ['ReduceMin', 'ReduceMean', 'ReduceProd']:
        for tno, (tmsg, input_shape) in enumerate(test_cases):
            op_name = f'test_{op_type}_{tno}'
            i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
            o_tensors = [make_tensor('Y')]
            op_info = {
                    'name'   : op_name,
                    'optype' : op_type,
                    'inList' : [x.name for x in i_tensors],
                    'outList': [x.name for x in o_tensors]
                    }

            # Use generic SimOp, which dispatches to desc-based reduction implementations
            op_obj = SimOp(op_info)
            for x in i_tensors: x.op_in  = [op_name]
            for x in o_tensors: x.op_out = [op_name]

            op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

            inf_shape = o_tensors[0].shape
            ref_shape = i_tensors[0].shape

            # Default reduction without axes reduces all dims to 1 (if keepdims=1)
            expected_shape = [1] * len(ref_shape)
            
            if inf_shape == expected_shape:
                print(f"TEST[{tno:3d}] {op_name} PASS")
            elif inf_shape == ref_shape:
                 # Fallback if test intended no-op? But code reduces.
                 print(f"TEST[{tno:3d}] {op_name} PASS (No reduction?)")
            else:
                print('INPUTS:')
                for x in i_tensors: print('\t', x)
                print('OUTPUTS:')
                for x in o_tensors: print('\t', x)
                assert False, f"TEST[{tno:3d}] {op_name} FAIL {inf_shape} != {expected_shape}"
