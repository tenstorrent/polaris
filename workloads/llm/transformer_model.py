#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from workloads.llm.attention import ConfigurableAttention

from typing import Any

class MLP(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name      = name
        self.dE        = cfg['dE']                 #hidden_dim
        self.idE       = cfg.get('idE', 3*self.dE) #intermediate_dim
        self.drop_prob = cfg.get('drop_prob', 0.01)

        #ops
        self.fc1   = SimNN.Linear(name + '.fc1', self.dE, self.idE)
        self.fc2   = SimNN.Linear(name + '.fc2', self.idE, self.dE)
        self.act   = F.Gelu(name + '.gelu')
        self.drop1 = F.Dropout(name + '.drop_mlp1', self.drop_prob)
        self.drop2 = F.Dropout(name + '.drop_mlp2', self.drop_prob)
        super().link_op2module()

    def __call__(self, x):
        print(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerBlock(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name      = name
        self.dE        = cfg['dE']
        self.drop_prob = cfg.get('drop_prob', 0.01)

        #ops
        self.attn     = ConfigurableAttention(name + '.attn', cfg)
        self.norm1    = F.LayerNorm(name + '.norm1', self.dE)
        self.mlp      = MLP(name + '.mlp', cfg)
        self.norm2    = F.LayerNorm(name + '.norm2', self.dE)
        self.dropout1 = F.Dropout(name + '.drop1', self.drop_prob)
        self.dropout2 = F.Dropout(name + '.drop2', self.drop_prob)

        super().link_op2module()

    def __call__(self, x, **kwargs):
        x              = self.norm1(x)
        attn_out, _, _ = self.attn(x, **kwargs)
        x              = x + self.dropout1(attn_out)
        x              = self.norm2(x)
        mlp_out        = self.mlp(x)
        x              = x + self.dropout2(mlp_out)
        return x

class TransformerModel(SimNN.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name     = name
        self.nL_proxy = 1

        # ---- Core Config ----
        self.vocab_sz  = config["vocab_sz"]
        self.bs        = config["bs"]
        self.dE        = config["dE"]
        self.nW        = config["nW"]
        self.nL        = config["nL"]
        #self.nH        = config["nH"]

        # ---- Embeddings ----
        self.embeddings = F.Embedding(name + '.embeddings', self.vocab_sz, self.dE)
        # Positional embedding
        if config.get("positional_encoding", "learned") == "learned":
            self.pos_embeddings    = F.Embedding(name + '.pos_embeddings', self.nW, self.dE)
            self.use_pos_embedding = True
            self.rotary_emb        = None
        elif config["positional_encoding"] == "rotary":
            if 'use_rope' in config:
                if config['use_rope'] == False:
                    config['use_rope'] = True
                    print("Overriding config.use_rope=True because positional_encoding is rotary")
            else:
                    config['use_rope'] = True
            self.pos_embeddings     = None
            self.use_pos_embedding  = False
        else:
            self.pos_embeddings    = None
            self.use_pos_embedding = False
            self.rotary_emb        = None

        # Segment embedding (for BERT)
        if config.get("use_segment_embedding", False):
            self.segment_embeddings = F.Embedding(name + '.segment_embeddings', 2, self.dE)
        else:
            self.segment_embeddings = None

        # ---- Layers ----
        self.tblocks = SimNN.ModuleList([TransformerBlock(name + f'.TransformerBlock_{i}', config) for i in range(self.nL_proxy)])
        self.norm    = F.LayerNorm(name + '.norm', self.dE)
        super().link_op2module()


    def __call__(self, x, segment_ids=None):
        assert x.rank() == 2, f"{self.name} input {x.name}.shape= {x.shape} not is [bs, seqlen] form"
        batch, qlen = x.shape

        x = self.embeddings(x)

        if self.use_pos_embedding:
            position_ids = F._from_shape(self.name + '.pos_ids', [batch, qlen])
            self._tensors[position_ids.name] = position_ids
            position_ids.set_module(self)
            x = x + self.pos_embeddings(position_ids)
        if self.segment_embeddings is not None:
            if segment_ids is None:
                segment_ids = F._from_shape(self.name + '.segment_ids', [batch, qlen])
                segment_ids.set_module(self)
            x = x + self.segment_embeddings(segment_ids)

        for tblock in self.tblocks:
            x = tblock(x)

        #Now that all sim_ops have been set, let's update the repeat counts
        for tblock in self.tblocks:
            repeated_ops: dict[str, Any] = {}
            tblock.get_ops(repeated_ops)
            for op_name,op_obj in repeated_ops.items():
                op_obj.repeat_count = self.nL

        x = self.norm(x)

        return x

def preset_cfg(name):
    cfg_tbl = {
            "bert_base_uncased": {
                "name"                  : "bert_base_uncased",
                "vocab_sz"              : 30522,
                "dE"                    : 768,
                "idE"                   : 3072,
                "nH"                    : 12,
                "nL"                    : 12,
                "nW"                    : 512,
                "drop_prob"             : 0.1,
                "attn_type"             : "bidir",
                "use_bias"              :  True,
                "norm_type"             : "layer",
                "positional_encoding"   : "learned",
                "use_segment_embedding" : True,
                },

            "bert_large_uncased": {
                "name"                  : "bert_large_uncased",
                "vocab_sz"              : 30522,
                "dE"                    : 1024,
                "idE"                   : 4096,
                "nH"                    : 16,
                "nL"                    : 24,
                "nW"                    : 512,
                "drop_prob"             : 0.1,
                "attn_type"             : "bidir",
                "use_bias"              :  True,
                "norm_type"             : "layer",
                "positional_encoding"   : "learned",
                "use_segment_embedding" : True,
                },

            "gpt2": {
                "name"                  : "gpt2",
                "vocab_sz"              : 50257,
                "dE"                    : 768,
                "idE"                   : 3072,
                "nH"                    : 12,
                "nL"                    : 12,
                "nW"                    : 1024,
                "drop_prob"             : 0.1,
                "attn_type"             : "causal",
                "use_bias"              :  True,
                "norm_type"             : "layer",
                "positional_encoding"   : "learned",
                "use_segment_embedding" : False,
                },

            "gpt2_large": {
                "name"                  : "gpt2_large",
                "vocab_sz"              : 50257,
                "dE"                    : 1280,
                "idE"                   : 5120,
                "nH"                    : 20,
                "nL"                    : 36,
                "nW"                    : 1024,
                "drop_prob"             : 0.1,
                "attn_type"             : "causal",
                "use_bias"              :  True,
                "norm_type"             : "layer",
                "positional_encoding"   : "learned",
                "use_segment_embedding" : False,
                },

            "llama2_7b": {
                    "name"                  : "llama2_7b",
                    "vocab_sz"              : 32000,
                    "dE"                    : 4096,
                    "idE"                   : 11008,
                    "nH"                    : 32,
                    "nL"                    : 32,
                    "nW"                    : 4096,
                    "drop_prob"             : 0.0,
                    "attn_type"             : "causal",
                    "use_bias"              : False,
                    "norm_type"             : "rms",
                    "positional_encoding"   : "rotary",
                    "use_segment_embedding" : False,
                    },

            "llama3_8b": {
                    "name"                  : "llama3_8b",
                    "vocab_sz"              : 128256,
                    "dE"                    : 4096,
                    "idE"                   : 14336,
                    "nH"                    : 32,
                    "nL"                    : 32,
                    "nW"                    : 8192,
                    "drop_prob"             : 0.0,
                    "attn_type"             : "causal",
                    "use_bias"              : False,
                    "norm_type"             : "rms",
                    "positional_encoding"   : "rotary",
                    "use_segment_embedding" : False,
                    },
        }
    return cfg_tbl[name]

if __name__ == '__main__':
    import numpy as np

    #test_name  = 'bert'
    test_name  = 'gpt2'
    #test_name  = 'llama2'

    if test_name == 'bert':
        # BERT Large
        bert_cfg_name = "bert_large_uncased"
        bert_cfg      = preset_cfg(bert_cfg_name)
        bert_cfg['bs']= 1
        bert_large    = TransformerModel(bert_cfg_name,bert_cfg)
        input_ids     = F._from_shape('input_ids',   [2, 128], np_dtype=np.int64)
        segment_ids   = F._from_shape('segment_ids', [2, 128], np_dtype=np.int64)
        bert_out      = bert_large(input_ids, segment_ids)
        print()
        print('-'*40)
        print(bert_out)
        print('='*40)
        gg = bert_large._get_forward_graph({'i': input_ids, 's': segment_ids})
        gg.graph2onnx(bert_cfg_name + '.onnx', do_model_check=True)
    elif test_name == 'gpt2':
        # GPT2
        gpt2_cfg_name = "gpt2"
        gpt2_cfg      = preset_cfg(gpt2_cfg_name)
        gpt2_cfg['bs']= 1
        gpt2          = TransformerModel(gpt2_cfg_name,gpt2_cfg)
        input_ids     = F._from_shape('input_ids',   [2, 64], np_dtype=np.int64)
        gpt2_out      = gpt2(input_ids)
        print()
        print('-'*40)
        print(gpt2_out)
        print('='*40)
        gg = gpt2._get_forward_graph({'i': input_ids})
        gg.graph2onnx(gpt2_cfg_name + '.onnx', do_model_check=False)
    elif test_name == 'llama2':
        pass
    else:
        print(f"ILLEGAL test_name= {test_name}")

        """
        # Llama 2-7B
        llama2 = TransformerModel.from_preset("llama2-7b")
        input_ids = torch.randint(0, 32000, (2, 128))
        out = llama2(input_ids)
        """
