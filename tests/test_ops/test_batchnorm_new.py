#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import BatchNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(XShape, SShape, BShape, MShape, VShape, output_mean_var, **kwargs):
    '''shape inference for batchnorm'''

    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, XShape),
        helper.make_tensor_value_info("S", TensorProto.FLOAT, SShape),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, BShape),
        helper.make_tensor_value_info("M", TensorProto.FLOAT, MShape),
        helper.make_tensor_value_info("V", TensorProto.FLOAT, VShape),
    ]

    # Define output tensors, Shapes to be inferred
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)]
    if output_mean_var == True:
        outputs.append(helper.make_tensor_value_info("Mean", TensorProto.FLOAT, None))
        outputs.append(helper.make_tensor_value_info("Var",  TensorProto.FLOAT, None))

    # Create BatchNormalization node
    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "S", "B", "M", "V"],
        outputs=["Y"] + (["Mean", "Var"] if output_mean_var else []),
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([bn_node], "bn_graph", inputs, outputs)
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)

    output_shapes = {output.name: [dim.dim_value for dim in output.type.tensor_type.shape.dim] \
                       for output in inferred_model.graph.output }
    return output_shapes

# Test cases
test_name  = 'test_batchnorm_new'
test_cases = [
        {
            'x': [2, 3, 4, 5],
            's': [3],
            'b': [3],
            'm': [3],
            'v': [3],
            'y': [2, 3, 4, 5],
            'name': "test_batchnorm_example",
            },
        {
            'x': [2, 3, 4, 5],
            's': [3],
            'b': [3],
            'm': [3],
            'v': [3],
            'eps': 1e-2,
            'y': [2, 3, 4, 5],
            'name': "test_batchnorm_epsilon",
            },
        {
            'x': [2, 3, 4, 5],
            's': [3],
            'b': [3],
            'm': [3],
            'v': [3],
            't': True,
            'y': [2, 3, 4, 5],
            'output_mean_var': True,
            'name': "test_batchnorm_example_training_mode",
            },
        {
            'x': [2, 3, 4, 5],
            's': [3],
            'b': [3],
            'm': [3],
            'v': [3],
            't': True,
            'y': [2, 3, 4, 5],
            'output_mean_var': True,
            'name': "test_batchnorm_epsilon_training_mode",
            },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_batchnorm_new():
    msgw = max([len(x) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        test_name = trec['name'] #type: ignore
        op_name = f'{test_name}_{tno}'

        XShape = trec['x'] #type: ignore
        SShape = trec['s'] #type: ignore
        BShape = trec['b'] #type: ignore
        MShape = trec['m'] #type: ignore
        VShape = trec['v'] #type: ignore

        i_tensors = [
                F._from_shape(f'X', XShape),
                F._from_shape(f'S', SShape),
                F._from_shape(f'B', BShape),
                F._from_shape(f'M', MShape),
                F._from_shape(f'V', VShape),
                ]
        o_tensors = [make_tensor('Y')]

        output_mean_var = False
        if 'output_mean_var' in trec: #type: ignore
            output_mean_var = trec['output_mean_var'] #type: ignore
            if output_mean_var == True:
                o_tensors.append(make_tensor('Mean'))
                o_tensors.append(make_tensor('Var'))

        attrs = {}
        if 'eps' in trec: attrs['epsilon']       = trec['eps'] #type: ignore
        if 't' in trec:   attrs['training_mode'] = trec['t'] #type: ignore
        op_info = {
                'name'   : op_name,
                'optype' : 'BatchNormalization',
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : attrs,
                }
        op_obj = BatchNormalizationOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        ref_shapes  = ref_impl_onnx(XShape, SShape, BShape, MShape, VShape, output_mean_var, **attrs)
        inf_y_shape = o_tensors[0].shape
        assert inf_y_shape == ref_shapes['Y'], f"{inf_y_shape} != {ref_shapes['Y']}"
        if output_mean_var:
            inf_m_shape = o_tensors[1].shape
            assert inf_m_shape == ref_shapes['Mean'], f"{inf_m_shape} != {ref_shapes['Mean']}"

            inf_v_shape = o_tensors[2].shape
            assert inf_v_shape == ref_shapes['Var'], f"{inf_v_shape} != {ref_shapes['Var']}"

        print(f"TEST[{tno:3d}] {test_name:{msgw}s} PASS")
