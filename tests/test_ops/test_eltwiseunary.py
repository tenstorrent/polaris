import pytest

import numpy as np
from ttsim.ops.op import EltwiseUnaryOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

# Test cases
test_name  = 'test_eltwiseunary'
test_cases = [
        ("0D",  []                      ),
        ("1D",  [2]                     ),
        ("2D",  [2, 3]                  ),
        ("3D",  [2, 3, 5]               ),
        ("4D",  [2, 3, 5, 7]            ),
        ("7D",  [2, 3, 5, 7, 9, 11, 13] ),
        ]

@pytest.mark.unit
@pytest.mark.opunit
def test_eltwiseunary():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'identity',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors]
                }
        op_obj = EltwiseUnaryOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = i_tensors[0].shape

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"
