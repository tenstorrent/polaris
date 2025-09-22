#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import ConvTransposeOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(XShape, WShape, BShape, **kwargs):
    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, XShape),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, WShape)
    ]
    if BShape is not None:
        inputs.append(helper.make_tensor_value_info("B", TensorProto.FLOAT, BShape))

    # Define output tensor
    output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)  # Shape to be inferred

    # Create ConvTranspose node
    conv_transpose_node = helper.make_node(
        "ConvTranspose",
        inputs=["X", "W"] + (["B"] if BShape is not None else []),
        outputs=["Y"],
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([conv_transpose_node], "conv_transpose_graph", inputs, [output])
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for output in inferred_model.graph.output:
        if output.name == "Y":
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            return shape
    raise ValueError("Output shape not found in inferred model")

# Test cases
test_name  = 'test_conv_transpose'
test_cases = [
    {
        'name': "test_basic_conv_transpose",
        'x': [1, 1, 3, 3],      # Input: N=1, C_in=1, H=3, W=3
        'w': [1, 1, 3, 3],      # Weight: C_in=1, C_out/group=1, KH=3, KW=3
        'strides': [1, 1],
        'pads': [0, 0, 0, 0],
        'output_padding': [0, 0],
    },
    {
        'name': "test_conv_transpose_with_stride",
        'x': [1, 1, 2, 2],
        'w': [1, 1, 3, 3],
        'strides': [2, 2],
        'pads': [0, 0, 0, 0],
        'output_padding': [0, 0],
    },
    {
        'name': "test_conv_transpose_with_padding",
        'x': [1, 1, 3, 3],
        'w': [1, 1, 2, 2],
        'strides': [1, 1],
        'pads': [1, 1, 1, 1],
        'output_padding': [0, 0],
    },
    {
        'name': "test_conv_transpose_with_output_padding",
        'x': [1, 1, 2, 2],
        'w': [1, 1, 3, 3],
        'strides': [2, 2],
        'pads': [0, 0, 0, 0],
        'output_padding': [1, 1],
    },
    {
        'name': "test_conv_transpose_1d",
        'x': [1, 1, 4],
        'w': [1, 1, 3],
        'strides': [2],
        'pads': [0, 0],
        'output_padding': [0],
    },
    {
        'name': "test_conv_transpose_with_bias",
        'x': [1, 1, 2, 2],
        'w': [1, 1, 3, 3],
        'b': [1],
        'strides': [2, 2],
        'pads': [0, 0, 0, 0],
        'output_padding': [0, 0],
    },
    {
        'name': "test_conv_transpose_grouped",
        'x': [1, 2, 3, 3],
        'w': [1, 2, 3, 3],      # C_in/group=1, C_out/group=2, so C_in=2, C_out=4
        'strides': [1, 1],
        'pads': [0, 0, 0, 0],
        'output_padding': [0, 0],
        'group': 2,
    },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_conv_transpose():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        WShape  = trec['w'] #type: ignore

        i_tensors = [F._from_shape(f'X', XShape), F._from_shape(f'W', WShape)]
        BShape = trec.get('b') #type: ignore
        if BShape is not None:
            i_tensors.append(F._from_shape(f'B', BShape))

        o_tensors = [make_tensor(f'O')]
        attrs = {
            'strides': trec.get('strides', [1, 1]), #type: ignore
            'pads': trec.get('pads', [0, 0, 0, 0]), #type: ignore
            'output_padding': trec.get('output_padding', [0, 0]), #type: ignore
        }

        # Add optional attributes
        if 'group' in trec: #type: ignore
            attrs['group'] = trec['group'] #type: ignore
        if 'dilations' in trec: #type: ignore
            attrs['dilations'] = trec['dilations'] #type: ignore
        if 'auto_pad' in trec: #type: ignore
            attrs['auto_pad'] = trec['auto_pad'] #type: ignore

        op_info = {
            'name'   : op_name,
            'optype' : 'ConvTranspose',
            'inList' : [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs'  : attrs,
        }
        op_obj = ConvTransposeOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl_onnx(XShape, WShape, BShape, **attrs)
        assert inf_shape == ref_shape, f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} : {inf_shape} != {ref_shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} PASS")
