#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import InstanceNormalizationOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(XShape, SShape, BShape, **kwargs):
    # Define input tensors
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, XShape),
        helper.make_tensor_value_info("scale", TensorProto.FLOAT, SShape)
    ]
    if BShape is not None:
        inputs.append(helper.make_tensor_value_info("bias", TensorProto.FLOAT, BShape))

    # Define output tensor
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)  # Shape to be inferred

    # Create InstanceNormalization node
    instance_norm_node = helper.make_node(
        "InstanceNormalization",
        inputs=["x", "scale"] + (["bias"] if BShape is not None else []),
        outputs=["y"],
        **kwargs)

    # Create graph and model
    graph = helper.make_graph([instance_norm_node], "instance_norm_graph", inputs, [output])
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for output in inferred_model.graph.output:
        if output.name == "y":
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
            return shape
    raise ValueError("Output shape not found in inferred model")

# Test cases
test_name  = 'test_instance_normalization'
test_cases = [
    {
        'name': "test_basic_instance_norm",
        'x': [2, 3, 4, 4],     # Input: N=2, C=3, H=4, W=4
        's': [3],               # Scale: C=3
        'b': None,              # No bias
        'epsilon': 1e-5,
    },
    {
        'name': "test_instance_norm_with_bias",
        'x': [1, 2, 8, 8],     # Input: N=1, C=2, H=8, W=8
        's': [2],               # Scale: C=2
        'b': [2],               # Bias: C=2
        'epsilon': 1e-5,
    },
    {
        'name': "test_instance_norm_different_epsilon",
        'x': [3, 4, 16, 16],   # Input: N=3, C=4, H=16, W=16
        's': [4],               # Scale: C=4
        'b': [4],               # Bias: C=4
        'epsilon': 1e-3,
    },
]

@pytest.mark.unit
@pytest.mark.opunit
def test_instance_normalization():
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        SShape  = trec['s'] #type: ignore
        BShape  = trec.get('b') #type: ignore

        i_tensors = [F._from_shape(f'X', XShape), F._from_shape(f'S', SShape)]
        if BShape is not None:
            i_tensors.append(F._from_shape(f'B', BShape))

        o_tensors = [make_tensor(f'O')]
        attrs = {
            'epsilon': trec.get('epsilon', 1e-5), #type: ignore
        }

        op_info = {
            'name'   : op_name,
            'optype' : 'InstanceNormalization',
            'inList' : [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
            'attrs'  : attrs,
        }
        op_obj = InstanceNormalizationOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl_onnx(XShape, SShape, BShape, **attrs)
        assert inf_shape == ref_shape, f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} : {inf_shape} != {ref_shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} PASS")
