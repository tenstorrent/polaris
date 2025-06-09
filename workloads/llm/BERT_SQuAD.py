#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from workloads.llm.transformer_model import TransformerModel

class BERTSQuAD(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name        = name
        self.transformer = TransformerModel(name + '.' + cfg['name'], cfg)
        self.qa_outputs  = SimNN.Linear(name + '.qa', self.transformer.dE, 2)  # For [start, end] logits
        self.qa_split    = F.SplitOpHandle(name +'.qa_split', count=2, axis=2)
        super().link_op2module()

    def __call__(self, input_ids, segment_ids=None):
        x      = self.transformer(input_ids, segment_ids)       # [B, L, H]
        logits = self.qa_outputs(x)                             # [B, L, 2]
        start_logits, end_logits = self.qa_split(logits)        # Each [B, L, 1]
        return start_logits, end_logits
        #return start_logits.squeeze(-1), end_logits.squeeze(-1) # [B, L], [B, L]

if __name__ == '__main__':
    import numpy as np
    from workloads.llm.transformer_model import preset_cfg

    bert_cfg       = preset_cfg("bert_base_uncased")
    bert_cfg['bs'] = 1
    squad_model    = BERTSQuAD('bert_base_squad', bert_cfg)
    input_ids      = F._from_shape('input_ids',   [2, 128], np_dtype=np.int64)
    segment_ids    = F._from_shape('segment_ids', [2, 128], np_dtype=np.int64)
    start_logits, end_logits = squad_model(input_ids, segment_ids)
