#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .registry import register_ops

def register_sequence_ops():
    _optbl = [
        ['SequenceEmpty',      'ARITY_0->1',                          'ai.onnx',  'COMMON',  11, 11,  0,  0,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['SequenceLength',     'ARITY_1->1',                          'ai.onnx',  'COMMON',  11, 11,  1,  1,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['ConcatFromSequence', 'ARITY_1->1',                          'ai.onnx',  'COMMON',  11, 11,  1,  1,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['SequenceAt',         'ARITY_2->1',                          'ai.onnx',  'COMMON',  11, 11,  2,  2,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['SequenceConstruct',  'ARITY_VARIADIC[1-*]->1',              'ai.onnx',  'COMMON',  11, 11,  2147483647,  1,  1,  1,           's0_func',                       True,  True,  True,  True,  True],
        ['SequenceErase',      'ARITY_VARIADIC[1-2]->1',              'ai.onnx',  'COMMON',  11, 11,  2,  1,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['SequenceInsert',     'ARITY_VARIADIC[2-3]->1',              'ai.onnx',  'COMMON',  11, 11,  3,  2,  1,  1,                    's0_func',                       True,  True,  True,  True,  True],
        ['SplitToSequence',    'ARITY_VARIADIC[1-2]->1',              'ai.onnx',  'COMMON',  24, 11,  2,  1,  1,  1,                    's1_func',                       True,  True,  True,  True,  True],
        ['SequenceMap',        'ARITY_VARIADIC[1-*]->VARIADIC[1-*]',  'ai.onnx',  'COMMON',  17, 17,  2147483647,  1,  2147483647,  1,  'SequenceMapInferenceFunction',  True,  True,  True,  True,  True],
        ]
    register_ops('sequence', _optbl)
    return
