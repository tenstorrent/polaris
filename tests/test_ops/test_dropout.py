import pytest

import numpy as np
from ttsim.ops.op import DropoutOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F


def ref_impl(XShape, drop_probability=0.5, seed=0, training_mode=False, return_mask=False):  # type: ignore
    X = np.random.randn(*XShape)
    if drop_probability == 0 or training_mode is False:
        if return_mask is True:
            return X, np.ones(X.shape, dtype=bool)
        else:
            return X

    np.random.seed(seed)
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = 1 / (1 - drop_probability)
    if return_mask:
        return mask * X * scale, mask.astype(bool)
    return mask * X * scale

# Test cases
test_name  = 'test_dropout'
test_cases = [
        {
            'name': "test_dropout_default",
            'x'   : [3, 4, 5],
            },
        {
            'name': "test_dropout_default_ratio",
            'x'   : [3, 4, 5],
            'r'   : 0.1,
            },
        {
            'name': "test_dropout_default_mask",
            'x'   : [3, 4, 5],
            'return_mask': True,
            },
        {
            'name': "test_dropout_default_mask_ratio",
            'x'   : [3, 4, 5],
            'r'   : 0.1,
            'return_mask': True,
            },
        {
            'name': "test_training_dropout_default",
            'x'   :  [3, 4, 5],
            'r'   :  0.5,
            't'   :  True,
            'num_outputs': 1,
            },
        {
            'name': "test_training_dropout_default_mask",
            'seed':  True,
            'x'   :  [3, 4, 5],
            'r'   :  0.5,
            't'   :  True,
            'return_mask': True,
            },
        {
            'name': "test_training_dropout",
            'x'   : [3, 4, 5],
            'r'   : 0.75,
            't'   : True,
            },
        {
            'name': "test_training_dropout_mask",
            'x'   : [3, 4, 5],
            'r'   : 0.75,
            't'   : True,
            'return_mask': True,
            },
        {
            'name': "test_training_dropout_zero_ratio",
            'x'   : [3, 4, 5],
            'r'   : 0.0,
            't'   : True,
            },
        {
                'name': "test_training_dropout_zero_ratio_mask",
                'x'   : [3, 4, 5],
                'r'   : 0.0,
                't'   : True,
                'return_mask': True,
                },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_dropout():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    for tno, trec in enumerate(test_cases):
        op_name     = f'{trec["name"]}_{tno}' #type: ignore
        XShape      = trec['x'] #type: ignore
        i_tensors   = [F._from_shape('X', XShape, np_dtype=np.float32)]
        o_tensors   = [make_tensor('Y')]
        KWARGS      = {} #kwargs for ref_impl
        num_outputs = 1

        if 'r' in trec: #type: ignore
            i_tensors.append(F._from_data('ratio', np.float32(trec['r']))) #type: ignore
            KWARGS['drop_probability'] = np.float32(trec['r']) #type: ignore

        if 't' in trec: #type: ignore
            i_tensors.append(F._from_data('training_mode', np.bool_(trec['t']))) #type: ignore
            KWARGS['training_mode'] = trec['t'] #type: ignore

        if 'return_mask' in trec: #type: ignore
            KWARGS['return_mask'] = trec['return_mask'] #type: ignore
            if KWARGS['return_mask'] == True:
                o_tensors.append(make_tensor('M'))
                num_outputs += 1

        op_info = {
                'name'   : op_name,
                'optype' : 'dropout',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                }
        op_obj = DropoutOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        if num_outputs == 2:
            Y, M = ref_impl(XShape, **KWARGS)
            assert o_tensors[0].shape == list(Y.shape), \
                    f"output data shape mismatch: {o_tensors[0].shape} == {Y.shape}"
            assert o_tensors[1].shape == list(M.shape), \
                    f"output mask shape mismatch: {o_tensors[1].shape} == {M.shape}"
        elif num_outputs == 1:
            Y = ref_impl(XShape, **KWARGS)
            assert o_tensors[0].shape == list(Y.shape), \
                    f"output data shape mismatch: {o_tensors[0].shape} == {Y.shape}"
        else:
            assert False, f"Dropout cannot outputs should be 1 or 2; instead it is {num_outputs}"
        print(f"TEST[{tno:3d}] {trec['name']:{msgw}s} PASS") #type: ignore
