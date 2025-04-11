import pytest

import numpy as np
from ttsim.ops.op import ReshapeOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])

def ref_impl(original_shape, target_shape, allowzero: int = 0):
    data = np.random.random_sample(original_shape).astype(np.float32)
    shape = np.array(target_shape, dtype=np.int64)
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped.shape

# Test cases
test_name  = 'test_reshape'
test_cases = [
        ("Basic reshape with -1, allowzero=0" , [2, 3, 4], [2, -1],       0),
        ("Copy dim with 0, allowzero=0"       , [2, 3, 4], [2, 0, -1],    0),
        ("All specified dims, allowzero=0"    , [2, 3, 4], [4, 3, 2],     0),
        ("reordered_all_dims, allowzero=0"    , [2, 3, 4], [4, 2, 3],     0),
        ("reordered_last_dims, allowzero=0"   , [2, 3, 4], [2, 4, 3],     0),
        ("reduced_dims, allowzero=0"          , [2, 3, 4], [2, 12],       0),
        ("extended_dims, allowzero=0"         , [2, 3, 4], [2, 3, 2, 2],  0),
        ("one_dim, allowzero=0"               , [2, 3, 4], [24],          0),
        ("negative_dim, allowzero=0"          , [2, 3, 4], [2, -1, 2],    0),
        ("negative_extended_dims, allowzero=0", [2, 3, 4], [-1, 2, 3, 4], 0),
        ("zero_dim, allowzero=0"              , [2, 3, 4], [2, 0, 4, 1],  0),
        ("zero_and_negative_dim, allowzero=0" , [2, 3, 4], [2, 0, 1, -1], 0),
        ("Basic reshape with -1, allowzero=1" , [2, 3, 4], [2, -1],       1),
        ("All specified dims, allowzero=1"    , [2, 3, 4], [4, 3, 2],     1),
        ("reordered_all_dims, allowzero=1"    , [2, 3, 4], [4, 2, 3],     1),
        ("reordered_last_dims, allowzero=1"   , [2, 3, 4], [2, 4, 3],     1),
        ("reduced_dims, allowzero=1"          , [2, 3, 4], [2, 12],       1),
        ("extended_dims, allowzero=1"         , [2, 3, 4], [2, 3, 2, 2],  1),
        ("one_dim, allowzero=1"               , [2, 3, 4], [24],          1),
        ("negative_dim, allowzero=1"          , [2, 3, 4], [2, -1, 2],    1),
        ("negative_extended_dims, allowzero=1", [2, 3, 4], [-1, 2, 3, 4], 1),
        ]

@pytest.mark.unit
@pytest.mark.opunit
def test_reshape():
    msgw = get_max_test_msg_len(test_cases)
    for tno, (tmsg, input_shape, target_shape, allowzero) in enumerate(test_cases):
        op_name = f'{test_name}_{tno}'
        i_tensors = [F._from_shape('X', input_shape, np_dtype=np.float32),
                     F._from_data('S', np.array(target_shape), is_const=True)]
        o_tensors = [make_tensor('Y')]
        op_info = {
                'name'   : op_name,
                'optype' : 'Reshape',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors]
                }
        op_obj = ReshapeOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(input_shape, target_shape, allowzero)
        ref_shape = list(ref_shape)

        if inf_shape == ref_shape:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            print('INPUTS:')
            for x in i_tensors: print('\t', x)
            print('OUTPUTS:')
            for x in o_tensors: print('\t', x)
            assert False, f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"
