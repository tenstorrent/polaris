import pytest

import numpy as np
from ttsim.ops.op import EltwiseBinaryOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL): return max([len(x[0]) for x in TL])

def ref_impl(shape0, shape1):
    _X0 = np.random.randn(*shape0)
    _X1 = np.random.randn(*shape1)
    _Y = np.add(_X0, _X1)
    return list(_Y.shape)

# Test cases
test_name  = 'test_eltwisebinary'
test_cases = [
        ("Scalar to 2D Broadcasting",      [],        [3, 4]    ),
        ("1D to 2D Broadcasting",          [4],       [3, 4]    ),
        ("Bidirectional Broadcasting",     [3, 1],    [1, 4]    ),
        ("Multi-dimensional Broadcasting", [2, 1, 4], [1, 3, 1] ),
        ("No Broadcasting",                [2, 3, 4], [2, 3, 4] ),
        ("Empty Dimension Broadcasting",   [1, 0],    [1]       ),
        ]

@pytest.mark.unit
@pytest.mark.opunit
def test_eltwisebinary():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, shape0, shape1) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [
                F._from_shape('X0', shape0, np_dtype=np.float32),
                F._from_shape('X1', shape1, np_dtype=np.float32),
                ]
        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'add',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors]
                }
        op_obj = EltwiseBinaryOp(op_info)
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
