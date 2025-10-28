#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.desc.registry import register_ops
from ttsim.ops.desc.helpers import unary_fwd, bidir_bcast

def register_logical_ops():
    _optbl = [
            #generated code, don't edit
            ['And',            'ARITY_2->1', 'ai.onnx', 'COMMON', 7,   7,   2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['Or',             'ARITY_2->1', 'ai.onnx', 'COMMON', 7,   7,   2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['Xor',            'ARITY_2->1', 'ai.onnx', 'COMMON', 7,   7,   2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['Greater',        'ARITY_2->1', 'ai.onnx', 'COMMON', 13,  13,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['Less',           'ARITY_2->1', 'ai.onnx', 'COMMON', 13,  13,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['Equal',          'ARITY_2->1', 'ai.onnx', 'COMMON', 19,  19,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['LessOrEqual',    'ARITY_2->1', 'ai.onnx', 'COMMON', 16,  16,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['GreaterOrEqual', 'ARITY_2->1', 'ai.onnx', 'COMMON', 16,  16,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['BitwiseAnd',     'ARITY_2->1', 'ai.onnx', 'COMMON', 18,  18,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['BitwiseOr',      'ARITY_2->1', 'ai.onnx', 'COMMON', 18,  18,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['BitwiseXor',     'ARITY_2->1', 'ai.onnx', 'COMMON', 18,  18,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['BitShift',       'ARITY_2->1', 'ai.onnx', 'COMMON', 11,  11,  2,  2, 1, 1, bidir_bcast, True, True, True, True, True],
            ['BitwiseNot',     'ARITY_1->1', 'ai.onnx', 'COMMON', 18,  18,  1,  1, 1, 1, unary_fwd,   True, True, True, True, True],
            ['Not',            'ARITY_1->1', 'ai.onnx', 'COMMON', 1,   1,   1,  1, 1, 1, unary_fwd,   True, True, True, True, True],
            ]

    register_ops('logical', _optbl)
    return
