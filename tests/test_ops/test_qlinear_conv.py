#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import QLinearConvOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(XShape, WShape, **kwargs):
    # QLinearConv has complex input structure with scales and zero points
    # For simplicity, we'll just validate the output shape matches ONNX expectation
    # Define input tensors for ONNX shape inference
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.UINT8, XShape),
        helper.make_tensor_value_info("x_scale", TensorProto.FLOAT, []),  # scalar
        helper.make_tensor_value_info("x_zero_point", TensorProto.UINT8, []),  # scalar
        helper.make_tensor_value_info("w", TensorProto.UINT8, WShape),
        helper.make_tensor_value_info("w_scale", TensorProto.FLOAT, []),  # scalar
        helper.make_tensor_value_info("w_zero_point", TensorProto.UINT8, []),  # scalar
        helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),  # scalar
        helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),  # scalar
    ]

    # Define output tensor
    output = helper.make_tensor_value_info("y", TensorProto.UINT8, None)  # Shape to be inferred

    # Create QLinearConv node
    qlinear_conv_node = helper.make_node(
        "QLinearConv",
        inputs=["x", "x_scale", "x_zero_point", "w", "w_scale", "w_zero_point", "y_scale", "y_zero_point"],
        outputs=["y"],
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([qlinear_conv_node], "qlinear_conv_graph", inputs, [output])
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for output in inferred_model.graph.output:
        if output.name == "y":
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            return shape
    raise ValueError("Output shape not found in inferred model")

# Test cases
test_name  = 'test_qlinear_conv'
test_cases = [
    {
        'name': "test_basic_qlinear_conv",
        'x': [1, 1, 3, 3],      # Input: N=1, C=1, H=3, W=3
        'w': [1, 1, 2, 2],      # Weight: C_out=1, C_in=1, KH=2, KW=2
        'strides': [1, 1],
        'pads': [0, 0, 0, 0],
    },
    {
        'name': "test_qlinear_conv_with_strides",
        'x': [1, 1, 5, 5],
        'w': [1, 1, 2, 2],
        'strides': [2, 2],
        'pads': [0, 0, 0, 0],
    },
    {
        'name': "test_qlinear_conv_with_padding",
        'x': [1, 1, 3, 3],
        'w': [1, 1, 2, 2],
        'strides': [1, 1],
        'pads': [1, 1, 1, 1],
    },
    {
        'name': "test_qlinear_conv_grouped",
        'x': [1, 2, 3, 3],
        'w': [2, 1, 2, 2],
        'strides': [1, 1],
        'pads': [0, 0, 0, 0],
        'group': 2,
    },
    {
        'name': "test_qlinear_conv_1d",
        'x': [1, 1, 4],
        'w': [1, 1, 3],
        'strides': [1],
        'pads': [0, 0],
    },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_qlinear_conv():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        WShape  = trec['w'] #type: ignore

        # Create input tensors with proper types
        i_tensors = [
            F._from_shape(f'X', XShape, np_dtype=np.uint8),           # input
            F._from_shape(f'XS', [], np_dtype=np.float32),            # x_scale (scalar)
            F._from_shape(f'XZ', [], np_dtype=np.uint8),              # x_zero_point (scalar)
            F._from_shape(f'W', WShape, np_dtype=np.uint8),           # weight
            F._from_shape(f'WS', [], np_dtype=np.float32),            # w_scale (scalar)
            F._from_shape(f'WZ', [], np_dtype=np.uint8),              # w_zero_point (scalar)
            F._from_shape(f'YS', [], np_dtype=np.float32),            # y_scale (scalar)
            F._from_shape(f'YZ', [], np_dtype=np.uint8),              # y_zero_point (scalar)
        ]

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
            'optype' : 'QLinearConv',
            'inList' : [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs'  : attrs,
        }
        op_obj = QLinearConvOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl_onnx(XShape, WShape, **attrs)
        assert inf_shape == ref_shape, f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} : {inf_shape} != {ref_shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} PASS")
