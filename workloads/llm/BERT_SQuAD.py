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
    def __init__(self, name, transformer: TransformerModel):
        super().__init__()
        self.name        = name
        self.transformer = transformer
        self.qa_outputs  = F.Linear(name + '.linear', transformer.hidden_dim, 2)  # For [start, end] logits
        super().link_op2module()

    def __call__(self, input_ids, segment_ids=None):
        # Get sequence output from backbone
        x = self.transformer(input_ids, segment_ids)            # [B, L, H]
        logits = self.qa_outputs(x)                             # [B, L, 2]
        start_logits, end_logits = logits.split(1, dim=-1)      # Each [B, L, 1]
        return start_logits.squeeze(-1), end_logits.squeeze(-1) # [B, L], [B, L]

class GPTLanguageModel(SimNN.Module):
    def __init__(self, name, transformer: TransformerModel, vocab_size = None):
        super().__init__()
        self.name        = name
        self.transformer = transformer
        self.lm_head     = F.Linear(name + '.lm_head',
                transformer.hidden_dim,
                vocab_size if vocab_size is not None else transformer.vocab_size)
                #, bias=False) # Use transformer.vocab_size if not provided
        super().link_op2module()

    def __call__(self, input_ids):
        x      = self.transformer(input_ids)    # [B, L, H]
        logits = self.lm_head(x)           # [B, L, vocab_size]
        return logits


if __name__ == '__main__':
    # Usage:
    bert_base   = TransformerModel.from_preset("bert-base-uncased")
    squad_model = BERTSQuAD('bert_base_squad', bert_base)
    #start_logits, end_logits = squad_model(input_ids, segment_ids)

    # Usage (GPT-J, but works for GPT2, Llama, etc.):
    gptj_config = {
            "vocab_size": 50400, "hidden_dim": 4096, "intermediate_dim": 16384, "num_heads": 16, "num_layers": 28,
            "max_position": 2048, "dropout": 0.0, "attention_type": "causal", "use_bias": False, "norm_type": "layer",
            "positional_encoding": "learned", "use_segment_embedding": False
            }
    gptj_backbone = TransformerModel('gptj_backbone', gptj_config)
    gptj_model = GPTLanguageModel('gptj_model', gptj_backbone)
    # Forward: logits = gptj_model(input_ids)
