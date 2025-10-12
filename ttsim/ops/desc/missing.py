#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

def register_missing_ops():
    _optbl = [
            {'opname': 'Swish',            'group': 'math',   'inf_func': 'propagateShapeAndTypeFromFirstInput',        },
            {'opname': 'Attention',        'group': 'nn',     'inf_func': 'AttentionPropagateElemTypeFromInputToOutput',},
            {'opname': 'RMSNormalization', 'group': 'nn',     'inf_func': 'inline_lambda',                              },
            {'opname': 'RotaryEmbedding',  'group': 'nn',     'inf_func': 'inline_lambda',                              },
            {'opname': 'TensorScatter',    'group': 'tensor', 'inf_func': 'inline_lambda',                              },
            ]
    return
