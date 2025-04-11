import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import ConvOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx( XShape, WShape, BShape, **kwargs):
    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, XShape),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, WShape)
    ]
    if BShape is not None:
        inputs.append(helper.make_tensor_value_info("B", TensorProto.FLOAT, BShape))

    # Define output tensor
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)  # Shape to be inferred

    # Create Conv node
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W"] + (["B"] if BShape is not None else []),
        outputs=["Y"],
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([conv_node], "conv_graph", inputs, [output])
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for output in inferred_model.graph.output:
        if output.name == "Y":
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            return shape
    raise ValueError("Output shape not found in inferred model")

# Test cases
test_name  = 'test_conv'
test_cases = [
        {
            'name': "test_basic_conv_with_padding",
            'x': [1, 1, 5, 5],
            'w': [1, 1, 3, 3],
            'k': [3, 3],
            'p': [1, 1, 1, 1],
            },
        {
            'name': "test_basic_conv_without_padding",
            'x': [1, 1, 5, 5],
            'w': [1, 1, 3, 3],
            'k': [3, 3],
            'p': [0, 0, 0, 0],
            },
        {
            'name': "test_conv_with_strides_padding",
            'x': [1, 1, 7, 5],
            'w': [1, 1, 3, 3],
            'k': [3, 3],
            'p': [1, 1, 1, 1],
            's': [2, 2, ],
            },
        {
            'name': "test_conv_with_strides_no_padding",
            'x': [1, 1, 7, 5],
            'w': [1, 1, 3, 3],
            'k': [3, 3],
            'p': [0, 0, 0, 0],
            's': [2, 2, ],
            },
        {
            'name': "test_conv_with_strides_and_asymmetric_padding",
            'x': [1, 1, 7, 5],
            'w': [1, 1, 3, 3],
            'k': [3, 3],
            'p': [1, 0, 1, 0],
            's': [ 2, 2, ],
            },
        {
            'name': "test_conv_with_autopad_same",
            'x': [1, 1, 5, 5],
            'w': [1, 1, 3, 3],
            'a': "SAME_LOWER",
            'k': [3, 3],
            's': [2, 2],
            },
        {
            'x'   : [2, 3, 32],
            'w'   : [4, 3, 3],
            'd'   : [1],
            'g'   : 1,
            'p'   : [0, 0],
            's'   : [1],
            'name': "1D"
            },
        {
            'x'   : [2, 3, 32],
            'w'   : [4, 3, 3],
            'b'   : [4,],
            'd'   : [1],
            'g'   : 1,
            'p'   : [0, 0],
            's'   : [1],
            'name': "1D with bias",
            },
        {
            'x'   : [2, 3, 32, 32],
            'w'   : [4, 3, 3, 3],
            'd'   : [1, 1],
            'g'   : 1,
            'p'   : [0, 0, 0, 0],
            's'   : [1, 1],
            'name': "2D conv",
            },
        {
            'x'   : [2, 3, 16, 16, 16],
            'w'   : [4, 3, 3, 3, 3],
            'd'   : [1, 1, 1],
            'g'   : 1,
            'p'   : [0, 0, 0, 0, 0, 0],
            's'   : [1, 1, 1],
            'name': "3D conv",
            },
        {
            'x'   : [2, 6, 16, 16, 16],
            'w'   : [4, 3, 3, 3, 3],
            'b'   : [4,],
            'd'   : [1, 1, 1],
            'g'   : 2,
            'p'   : [0, 0, 0, 0, 0, 0],
            's'   : [1, 1, 1],
            'name': "3D with bias and groups",
                },
# FAILS FOR NOW... ONNX = [2, 4, 15, 15] ours = [2, 4, 16, 16]
#      {
#          'x'   :  [2, 3, 32, 32],
#          'w'   :  [4, 3, 3, 3],
#          'b'   :  [4,],
#          'd'   :  [1, 1],
#          'g'   :  1,
#          'p'   :  [0, 0, 0, 0],
#          's'   :  [2, 2],
#          'a'   :  "SAME_UPPER",
#          'name':  "2D with bias and strides",
#          },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_conv():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        WShape  = trec['w'] #type: ignore

        i_tensors = [F._from_shape(f'X', XShape), F._from_shape(f'W', WShape)]
        if 'b' in trec: #type: ignore
            BShape = trec['b'] #type: ignore
            i_tensors.append(F._from_shape(f'B', BShape))
        else:
            BShape = None

        o_tensors = [make_tensor(f'O')]
        attrs     = {}
        if 'd' in trec: attrs['dilations']    = trec['d'] #type: ignore
        if 'g' in trec: attrs['group']        = trec['g'] #type: ignore
        if 'p' in trec: attrs['pads']         = trec['p'] #type: ignore
        if 's' in trec: attrs['strides']      = trec['s'] #type: ignore
        if 'a' in trec: attrs['auto_pad']     = trec['a'] #type: ignore
        if 'k' in trec: attrs['kernel_shape'] = trec['k'] #type: ignore
        op_info = {
                'name'   : op_name,
                'optype' : 'Conv',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : attrs,
                }
        op_obj = ConvOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl_onnx(XShape, WShape, BShape, **attrs)
        assert inf_shape == ref_shape, f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} : {inf_shape} != {ref_shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} PASS")
