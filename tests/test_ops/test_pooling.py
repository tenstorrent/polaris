#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest

import onnx
from onnx import helper, TensorProto

import numpy as np
from ttsim.ops.op import MaxPoolOp, AveragePoolOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F

def ref_impl_onnx(PoolType, XShape, YShape, ZShape, **kwargs):
    '''ref shape inference for pooling types'''

    # Define input tensors
    inputs = [helper.make_tensor_value_info("X", TensorProto.FLOAT, XShape)]

    # Define output tensors, shapes to be inferred
    outputs = [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)]
    if ZShape is not None and PoolType == 'MaxPool':
        outputs.append(helper.make_tensor_value_info("Z", TensorProto.FLOAT, None))

    # Create MaxPool node
    pooling_node = helper.make_node(
            PoolType,
            inputs=["X"],
            outputs=["Y"] + (["Z"] if ZShape is not None else []),
            **kwargs)

    # Create graph and model
    graph = helper.make_graph([pooling_node], "pooling_graph", inputs, outputs)
    model = helper.make_model(graph, producer_name="polaris-unit-test")

    # Infer shapes
    inferred_model = onnx.shape_inference.infer_shapes(model)

    outshapes = {output.name: [dim.dim_value for dim in output.type.tensor_type.shape.dim] \
                      for output in inferred_model.graph.output}
    if "Y" not in outshapes:
        raise ValueError("Output shape Y not found in inferred model")
    if PoolType == 'MaxPool' and ZShape is not None and "Z" not in outshapes:
        raise ValueError("Output shape Z not found in inferred model")
    return outshapes

# Test cases
test_name  = 'test_pooling'
test_cases = [
        {
            'name'        : "test_maxpool_2d_uint8",
            'x'           : [1, 1, 5, 5],
            'y'           : [1, 1, 5, 5],
            'inputs'      : ["x"],
            'outputs'     : ["y"],
            'kernel_shape': [5, 5],
            'pads'        : [2, 2, 2, 2],
            },
        {
            'name'        : "test_maxpool_2d_precomputed_pads",
            'x'           :  [1, 1, 5, 5],
            'y'           :  [1, 1, 5, 5],
            "inputs"      : ["x"],
            "outputs"     : ["y"],
            "kernel_shape": [5, 5],
            "pads"        : [2, 2, 2, 2],

            },
        {
            'name'        : "test_maxpool_with_argmax_2d_precomputed_pads",
            'x'           :  [1, 1, 5, 5],
            'y'           :  [1, 1, 5, 5],
            'z'           :  [1, 1, 5, 5],
            "inputs"      : ["x"],
            "outputs"     : ["y", "z"],
            "kernel_shape": [5, 5],
            "pads"        : [2, 2, 2, 2],

            },
        {
            'name'        : "test_maxpool_2d_precomputed_strides",
            'x'           :  [1, 1, 5, 5],
            'y'           :  [1, 1, 2, 2],
            "inputs"      : ["x"],
            "outputs"     : ["y"],
            "kernel_shape": [2, 2],
            "strides"     : [2, 2]
            },
        {
            'name'         : "test_maxpool_with_argmax_2d_precomputed_strides",
            'x'            :  [1, 1, 5, 5],
            'y'            :  [1, 1, 2, 2],
            'z'            :  [1, 1, 2, 2],
            "inputs"       : ["x"],
            "outputs"      : ["y", "z"],
            "kernel_shape" : [2, 2],
            "strides"      : [2, 2],
            "storage_order": 1,

            },
        {
                'name'        : "test_maxpool_2d_ceil",
                'x'           : [1, 1, 4, 4],
                'y'           : [1, 1, 2, 2],
                "inputs"      : ["x"],
                "outputs"     : ["y"],
                "kernel_shape": [3, 3],
                "strides"     : [2, 2],
                "ceil_mode"   : True,

                },
        {
                'name'        : "test_maxpool_2d_dilations",
                'x'           : [1, 1, 4, 4],
                'y'           : [1, 1, 2, 2],
                "inputs"      : ["x"],
                "outputs"     : ["y"],
                "kernel_shape": [2, 2],
                "strides"     : [1, 1],
                "dilations"   : [2, 2],
                },
        {
                'name'        : "test_maxpool_3d_dilations",
                'x'           : [1, 1, 4, 4, 4],
                'y'           : [1, 1, 2, 2, 2],
                "inputs"      : ["x"],
                "outputs"     : ["y"],
                "kernel_shape": [2, 2, 2],
                "strides"     : [1, 1, 1],
                "dilations"   : [2, 2, 2],
                },
#FAILS
#       {
#               'name'        : "test_maxpool_2d_ceil_output_size_reduce_by_one",
#               'x'           : [1, 1, 2, 2],
#               'y'           : [1, 1, 1, 1],
#               "inputs"      : ["x"],
#               "outputs"     : ["y"],
#               "kernel_shape": [1, 1],
#               "strides"     : [2, 2],
#               "ceil_mode"   : True,
#               },
#       {
#               'name'        : "test_maxpool_2d_precomputed_same_upper",
#               'x'           :  [1, 1, 5, 5],
#               'y'           :  [1, 1, 3, 3],
#               "inputs"      : ["x"],
#               "outputs"     : ["y"],
#               "kernel_shape": [3, 3],
#               "strides"     : [2, 2],
#               "auto_pad"    : "SAME_UPPER",
#               },
        ]

def check_pooltype(pooltype):
    msgw = max([len(x['name']) for x in test_cases]) #type: ignore
    print()
    for tno, trec in enumerate(test_cases):
        tname   = trec['name'] #type: ignore
        op_name = f'{tname}_{tno}'
        XShape  = trec['x'] #type: ignore
        YShape  = trec['y'] #type: ignore
        i_tensors = [F._from_shape(f'X', XShape)]
        o_tensors = [make_tensor('Y')]
        if pooltype == 'MaxPool' and 'z' in trec: #type: ignore
            ZShape  = trec['z'] #type: ignore
            o_tensors.append(make_tensor('Z'))
        else:
            ZShape = None

        attrs     = {}
        if 'dilations'     in trec: attrs['dilations']      = trec['dilations']     #type: ignore
        if 'pads'          in trec: attrs['pads']           = trec['pads']          #type: ignore
        if 'strides'       in trec: attrs['strides']        = trec['strides']       #type: ignore
        if 'auto_pad'      in trec: attrs['auto_pad']       = trec['auto_pad']      #type: ignore
        if 'kernel_shape'  in trec: attrs['kernel_shape']   = trec['kernel_shape']  #type: ignore
        if 'ceil_mode'     in trec: attrs['ceil_mode']      = trec['ceil_mode']     #type: ignore
        op_info = {
                'name'   : op_name,
                'optype' : pooltype,
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors],
                'attrs'  : attrs,
                }
        op_obj = MaxPoolOp(op_info) if pooltype == 'MaxPool' else AveragePoolOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]
        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)

        ref_shapes = ref_impl_onnx(pooltype, XShape, YShape, ZShape, **attrs)
        assert YShape == ref_shapes['Y'], \
                f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} {pooltype} : ONNX {YShape} != {ref_shapes['Y']}"
        assert YShape == o_tensors[0].shape, \
                f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s}  {pooltype}: TTSIM {YShape} != {o_tensors[0].shape}"
        print(f"SIMPLE TEST[{tno:3d}] {tname:{msgw}s} {pooltype} PASS")

@pytest.mark.unit
@pytest.mark.opunit
def test_pooling():
    check_pooltype('MaxPool')
    check_pooltype('AveragePool')

