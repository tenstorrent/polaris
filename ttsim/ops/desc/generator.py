#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops

def register_generator_ops():
    _optbl = [
            ['RandomUniform',     'ARITY_0->1', 'ai.onnx',  'COMMON',  22,  22,  0,  0,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomNormal',      'ARITY_0->1', 'ai.onnx',  'COMMON',  22,  22,  0,  0,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['ConstantOfShape',   'ARITY_1->1', 'ai.onnx',  'COMMON',  24,  21,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['EyeLike',           'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomUniformLike', 'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['RandomNormalLike',  'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Multinomial',       'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Bernoulli',         'ARITY_1->1', 'ai.onnx',  'COMMON',  22,  22,  1,  1,  1,  1,  'g1_func', True,  True,  True,  True,  True],
            ['Range',             'ARITY_3->1', 'ai.onnx',  'COMMON',  11,  11,  3,  3,  1,  1,  'g1_func',              True,  True,  True,  True,  True],
            ['Constant',          'ARITY_0->1', 'ai.onnx',  'COMMON',  24,  21,  0,  0,  1,  1,  'ConstantOpInference',  True,  True,  True,  True,  True],
            ]

    register_ops('generator', _optbl)
    return

