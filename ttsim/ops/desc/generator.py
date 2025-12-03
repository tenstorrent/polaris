#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
import numpy as np

def ConstantOpInference(inT, outT, op, **kwargs):
    # Constant op outputs a constant tensor defined via attribute 'value'
    value = op.attrs.get('value', None)
    if value is None:
        raise ValueError(f"ConstantOpInference: Constant op {op.name} missing 'value' attribute")
    outT[0].shape = list(value.shape)
    outT[0].dtype = value.dtype
    outT[0].data = value

    # For perf counts, we assume Constant op has zero compute and memory cycles
    op.perf_stats = {
            'inBytes' : 4,
            'outBytes': outT[0].nbytes(),
            'inElems' : 1,
            'outElems': outT[0].nelems(),
            'instrs'  : {'mov': outT[0].nelems()},
            }
    return

def ConstantOfShapeOpInference(inT, outT, op, **kwargs):
    if len(inT) < 1 or inT[0].data is None:
        raise ValueError(f"ConstantOfShapeOpInference: op {op.name} missing input shape tensor")

    output_shape = [int(x) for x in inT[0].data]
    fill_value = op.attrs.get('value', np.array([0.0], dtype=np.float32))

    outT[0].shape = output_shape
    outT[0].dtype = np.asarray(fill_value).dtype

    op.perf_stats = {
        'inBytes': 4 * len(output_shape),
        'outBytes': outT[0].nbytes(),
        'inElems': len(output_shape),
        'outElems': outT[0].nelems(),
        'instrs': {'mov': outT[0].nelems()},
    }
    return

def register_generator_ops():
    _optbl = [
            ['RandomUniform',     'ARITY_0->1', 'ai.onnx',  'COMMON',  22,  22,  0,  0,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomNormal',      'ARITY_0->1', 'ai.onnx',  'COMMON',  22,  22,  0,  0,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['ConstantOfShape',   'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1,  ConstantOfShapeOpInference, True,  True,  True,  True,  True],
            ['EyeLike',           'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomUniformLike', 'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomNormalLike',  'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Multinomial',       'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Bernoulli',         'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Range',             'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'g1_func',              True,  True,  True,  True,  True],
            ['Constant',          'ARITY_0->1', 'ai.onnx',  'COMMON',  24,  21,  0,  0,  1,  1,  ConstantOpInference,  True,  True,  True,  True,  True],
            ]

    register_ops('generator', _optbl)
    return
