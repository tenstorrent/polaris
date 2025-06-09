#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from workloads.llm.transformer_model import TransformerModel

class LanguageModel(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name        = name
        self.transformer = TransformerModel(name + '.' + cfg['name'], cfg)
        self.lm_head     = SimNN.Linear(name + '.lm_head', self.transformer.dE, self.transformer.vocab_sz, bias=False)
        super().link_op2module()

    def __call__(self, input_ids):
        x      = self.transformer(input_ids) # [B, L, H]
        logits = self.lm_head(x)             # [B, L, vocab_size]
        return logits


if __name__ == '__main__':
    import numpy as np

    # Usage (GPT-J, but works for GPT2, Llama, etc.):
    gptj_config = {
            "vocab_sz": 50400, "dE": 4096, "nH": 16, "nL": 28, "nW": 2048, "drop_prob": 0.0,
            "attn_type": "causal", "use_bias": False, "norm_type": "layer",
            "positional_encoding": "learned", "use_segment_embedding": False,
            'name': 'gptj', 'bs': 1}
    gptj_model = LanguageModel('gptj_model', gptj_config)
    input_ids  = F._from_shape('input_ids',   [2, 128], np_dtype=np.int64)
    logits     = gptj_model(input_ids)
    print(logits)

