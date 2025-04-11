import pytest

import numpy as np
from ttsim.ops.op import TransposeOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl(shape0, perms0):
    _X0 = np.random.randn(*shape0)
    _Y = np.transpose(_X0, perms0)
    return list(_Y.shape)

# Test cases
test_name  = 'test_transpose'
test_cases = [
    ("2D Matrix Transpose",       [3, 4],       [1, 0]      ),
    ("1D Vector",                 [5],          [0]         ),
    ("3D Tensor Transpose",       [2, 3, 4],    [1, 0, 2]   ),
    ("4D Tensor Transpose",       [2, 3, 4, 5], [3, 2, 1, 0]),
    ("Empty Dimension Transpose", [3, 0, 2],    [2, 1, 0]   ),
    ("Scalar Transpose",          [],           []          ),
]

@pytest.mark.unit
@pytest.mark.opunit
def test_transpose():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, shape, perms) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [F._from_shape('X0', shape, np_dtype=np.float32)]
        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'matmul',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : {'perm': perms},
                }
        op_obj = TransposeOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(shape, perms)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"
