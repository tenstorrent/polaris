#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import ttsim.front.ttnn as ttnn
from ttsim.utils.common import dict2obj

from workloads.ttnn.common import create_ttnn_tensor, update_param_attrs

def create_wb_params(dim1, dim2=None, has_bias=True):
    assert isinstance(dim1, int) and dim1 > 0, f"dim1 Error: {dim1}"
    odim = dim1 if dim2 is None else dim2
    pp = {}
    pp['weight'] = create_ttnn_tensor(shape=(dim1, odim))
    if has_bias:
            pp['bias'] = create_ttnn_tensor(shape=(odim,))
    return pp

def create_lyrnorm_params(emb_dim):
    return {
            'weight': create_ttnn_tensor(shape=(emb_dim,)),
            'bias'  : create_ttnn_tensor(shape=(emb_dim,))
            }

def create_bert_params(cfg, /, device, dtype):
    VS  = cfg.vocab_size
    HS  = cfg.hidden_size
    IS  = cfg.intermediate_size
    MPE = cfg.max_position_embeddings
    NL  = cfg.num_hidden_layers
    params = {
            'embeddings': {
                'position_embeddings'  : create_wb_params(VS,  HS, has_bias=False),
                'token_type_embeddings': create_wb_params(2,   HS, has_bias=False),
                'word_embeddings'      : create_wb_params(MPE, HS, has_bias=False),
                'LayerNorm'            : create_lyrnorm_params(HS)
                },
            'qa_outputs': create_wb_params(HS, 2),
            'encoder': { 'layer': [] }
            }
    for layer_num in range(NL):
        params['encoder']['layer'].append(dict2obj({
                'attention': {
                    'self'  : {
                        'query': create_wb_params(HS),
                        'key'  : create_wb_params(HS),
                        'value': create_wb_params(HS)
                        },
                    'output': {
                        'dense'    : create_wb_params(HS),
                        'LayerNorm': create_lyrnorm_params(HS)
                        }
                    },
                'intermediate': {
                    'dense': create_wb_params(HS, IS)
                    },
                'output': {
                    'dense'    : create_wb_params(IS, HS),
                    'LayerNorm': create_lyrnorm_params(HS)
                    },
                }))

    update_param_attrs(params, device=device, dtype=dtype)

    return dict2obj(params)

def create_bert_inputs(bs, seq_sz, dev):
    return {
            'input_ids'      : ttnn.Tensor(name='input_ids     ', shape=(bs, seq_sz), device=dev, dtype=ttnn.uint32),
            'token_type_ids' : ttnn.Tensor(name='token_type_ids', shape=(bs, seq_sz), device=dev, dtype=ttnn.uint32),
            'position_ids'   : ttnn.Tensor(name='position_ids  ', shape=(bs, seq_sz), device=dev, dtype=ttnn.uint32),
            'attention_mask' : None,
            }

