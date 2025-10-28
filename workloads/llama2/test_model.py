#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from workloads.llama2.model import Transformer

def test_llama2_forward():
    cfg = {'dim': 32, 'n_layers': 2, 'n_heads': 2,
            'vocab_size': 256, 'max_batch_size': 2,
            'max_seq_len': 8, 'bs': 1, 'seq_len': 1}
    model = Transformer('transformer_model', cfg)
    batch_size = cfg['bs']
    seqlen = cfg['seq_len']
     # Create random token indices in vocab range
    # tokens = F._from_shape('input_tokens', [batch_size, seqlen])
    model.create_input_tensors()
    out = model() #tokens, start_pos=0)
    print('Output shape:', out.shape)
    assert out.shape == [batch_size, seqlen, cfg['vocab_size']], 'Mismatched output shape'
    print('Test passed!')
    gg = model.get_forward_graph()
    print('Dumping ONNX...')
    gg.graph2onnx('llama2.onnx', do_model_check=False)

if __name__ == "__main__":
    test_llama2_forward()
