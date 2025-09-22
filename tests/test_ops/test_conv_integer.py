#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import ConvIntegerOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(XShape, WShape, XZeroShape, WZeroShape, **kwargs):
    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.UINT8, XShape),
        helper.make_tensor_value_info("w", TensorProto.UINT8, WShape)
    ]
    if XZeroShape is not None:
        inputs.append(helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, XZeroShape))
    if WZeroShape is not None:
        inputs.append(helper.make_tensor_value_info("w_zero_point", TensorProto.UINT8, WZeroShape))

    # Define output tensor
    output = helper.make_tensor_value_info("y", TensorProto.INT32, None)  # Shape to be inferred

    # Create ConvInteger node
    conv_integer_node = helper.make_node(
        "ConvInteger",
        inputs=["x", "w"] +
               (["x_zero_point"] if XZeroShape is not None else []) +
               (["w_zero_point"] if WZeroShape is not None else []),
        outputs=["y"],
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([conv_integer_node], "conv_integer_graph", inputs, [output])
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for output in inferred_model.graph.output:
        if output.name == "y":
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            return shape
    raise ValueError("Output shape not found in inferred model")

# Test cases
test_name  = 'test_conv_integer'
test_cases = [
    {
        'name': "test_basic_conv_integer_without_zero_points",
        'x': [1, 1, 3, 3],      # Input: N=1, C=1, H=3, W=3
        'w': [1, 1, 2, 2],      # Weight: C_out=1, C_in=1, KH=2, KW=2
        'x_zero': None,
        'w_zero': None,
    },
    {
        'name': "test_conv_integer_with_zero_points",
        'x': [1, 1, 3, 3],
        'w': [1, 1, 2, 2],
        'x_zero': [],
        'w_zero': [],
    },
    {
        'name': "test_conv_integer_with_padding",
        'x': [1, 1, 3, 3],
        'w': [1, 1, 2, 2],
        'x_zero': [],
        'w_zero': [],
        'pads': [1, 1, 1, 1],
    },
    {
        'name': "test_conv_integer_with_strides",
        'x': [1, 1, 5, 5],
        'w': [1, 1, 2, 2],
        'x_zero': [],
        'w_zero': [],
        'strides': [2, 2],
    },
    {
        'name': "test_conv_integer_grouped",
        'x': [1, 2, 3, 3],
        'w': [2, 1, 2, 2],
        'x_zero': [],
        'w_zero': [],
        'group': 2,
    },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_conv_integer():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        WShape  = trec['w'] #type: ignore
        XZeroShape = trec.get('x_zero') #type: ignore
        WZeroShape = trec.get('w_zero') #type: ignore

        i_tensors = [F._from_shape(f'X', XShape, np_dtype=np.uint8), F._from_shape(f'W', WShape, np_dtype=np.uint8)]
        if XZeroShape is not None:
            i_tensors.append(F._from_shape(f'XZ', XZeroShape, np_dtype=np.uint8))
        if WZeroShape is not None:
            i_tensors.append(F._from_shape(f'WZ', WZeroShape, np_dtype=np.uint8))

        o_tensors = [make_tensor(f'O')]
        attrs = {
            'strides': trec.get('strides', [1, 1]), #type: ignore
            'pads': trec.get('pads', [0, 0, 0, 0]), #type: ignore
        }

        # Add optional attributes
        if 'group' in trec: #type: ignore
            attrs['group'] = trec['group'] #type: ignore
        if 'dilations' in trec: #type: ignore
            attrs['dilations'] = trec['dilations'] #type: ignore

        op_info = {
            'name'   : op_name,
            'optype' : 'ConvInteger',
            'inList' : [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs'  : attrs,
        }
        op_obj = ConvIntegerOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl_onnx(XShape, WShape, XZeroShape, WZeroShape, **attrs)
        assert inf_shape == ref_shape, f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} : {inf_shape} != {ref_shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} PASS")
