#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from workloads.llm.attention import ConfigurableAttention, RotaryEmbedding

class RMSNorm(SimNN.Module):
    """Root Mean Square LayerNorm as in Llama."""
    def __init__(self, name, hidden_dim, eps=1e-6):
        super().__init__()
        self.name   = name
        self.eps    = F._from_data(name + '.eps', is_const=True, data=np.float32(eps))
        self.weight = F._from_shape(name + '.weight', [hidden_dim], is_param=True)
        super().link_op2module()

    def __call__(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms  = norm / (x.shape[-1] ** 0.5)
        return x / (rms + self.eps) * self.weight

class MLP(SimNN.Module):
    def __init__(self, name, hidden_dim, intermediate_dim, mlp_pdrop=0.1):
        super().__init__()
        self.name    = name
        self.fc1     = F.Linear(name + '.fc1', hidden_dim, intermediate_dim)
        self.act     = F.Gelu(name + '.gelu')
        self.fc2     = F.Linear(name + '.fc1', intermediate_dim, hidden_dim)
        self.dropout1 = F.Dropout(name + '.drop_mlp1', mlp_pdrop)
        self.dropout2 = F.Dropout(name + '.drop_mlp2', mlp_pdrop)
        super().link_op2module()

    def __call__(self, x):
        print(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(SimNN.Module):
    def __init__(self, name, config, attn_config, rotary_emb=None, norm_type="layer"):
        super().__init__()
        self.name    = name
        self.attn    = ConfigurableAttention(name + '.attn', attn_config, rotary_emb=rotary_emb)
        self.norm1   = F.LayerNorm(name + '.norm1', config["hidden_dim"]) \
                           if norm_type == "layer" else RMSNorm(name + '.rmsnorm1', config["hidden_dim"])
        self.mlp     = MLP(name + '.mlp', config["hidden_dim"], config["intermediate_dim"], config.get("dropout", 0.1))
        self.norm2   = F.LayerNorm(name + '.norm2', config["hidden_dim"]) \
                           if norm_type == "layer" else RMSNorm(name + '.rmsnorm2', config["hidden_dim"])
        self.dropout1 = F.Dropout(name + '.drop1', config.get("dropout", 0.1))
        self.dropout2 = F.Dropout(name + '.drop2', config.get("dropout", 0.1))
        super().link_op2module()

    def __call__(self, x, **kwargs):
        print(x)
        attn_out, _, _ = self.attn(self.norm1(x), **kwargs)
        x              = x + self.dropout1(attn_out)
        mlp_out        = self.mlp(self.norm2(x))
        x              = x + self.dropout2(mlp_out)
        return x

class TransformerModel(SimNN.Module):
    def __init__(self, name, config):
        super().__init__()
        self.name = name
        # ---- Core Config ----
        self.vocab_size       = config["vocab_size"]
        self.hidden_dim       = config["hidden_dim"]
        self.max_position     = config["max_position"]
        self.num_layers       = config["num_layers"]
        self.num_heads        = config["num_heads"]
        self.intermediate_dim = config["intermediate_dim"]

        # ---- Embeddings ----
        self.embeddings = F.Embedding(name + '.embeddings', self.vocab_size, self.hidden_dim)
        # Positional embedding
        if config.get("positional_encoding", "learned") == "learned":
            self.pos_embeddings    = F.Embedding(name + '.pos_embeddings', self.max_position, self.hidden_dim)
            self.use_pos_embedding = True
            self.rotary_emb        = None
        elif config["positional_encoding"] == "rotary":
            self.pos_embeddings     = None
            self.use_pos_embedding  = False
            self.rotary_emb         = RotaryEmbedding(name + '.rotary_emb', self.hidden_dim // self.num_heads)
        else:
            self.pos_embeddings    = None
            self.use_pos_embedding = False
            self.rotary_emb        = None

        # Segment embedding (for BERT)
        if config.get("use_segment_embedding", False):
            self.segment_embeddings = F.Embedding(name + '.segment_embeddings', 2, self.hidden_dim)
        else:
            self.segment_embeddings = None

        # ---- Layers ----
        norm_type  = config.get("norm_type", "layer")
        attn_type  = config.get("attention_type", "bidirectional")
        use_bias   = config.get("use_bias", True)
        dropout    = config.get("dropout", 0.1)
        layers = []
        for i in range(self.num_layers):
            layers.append(TransformerBlock( name + f'.TransformerBlock_{i}',
                {"hidden_dim": self.hidden_dim, "intermediate_dim": self.intermediate_dim, "dropout": dropout},
                                           #{"num_heads": self.num_heads, "hidden_dim": self.hidden_dim, "attention_type": attn_type, "dropout": dropout, "use_bias": use_bias},
                {"nH": self.num_heads, "dE": self.hidden_dim, "attn_type": attn_type, "attn_pdrop": dropout, "use_bias": use_bias},
                rotary_emb=self.rotary_emb,
                norm_type=norm_type
            ))
        self.layers = SimNN.ModuleList(layers)
        self.norm = F.LayerNorm(name + '.norm', self.hidden_dim) \
                       if norm_type == "layer" else RMSNorm(name + '.rmsnorm', self.hidden_dim)
        super().link_op2module()


    def __call__(self, input_ids, segment_ids=None):
        B, L = input_ids.shape
        x = self.embeddings(input_ids)
        if self.use_pos_embedding:
            #position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
            position_ids = F._from_shape(self.name + '.pos_ids', [B, L])
            position_ids.set_module(self)
            x = x + self.pos_embeddings(position_ids)
        if self.segment_embeddings is not None:
            print(">>>", segment_ids)
            if segment_ids is None:
                segment_ids = F._from_shape(self.name + '.segment_ids', input_ids.shape)
                segment_ids.set_module(self)
            x = x + self.segment_embeddings(segment_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

    @classmethod
    def from_preset(cls, name):
        # Add or update as you wish!
        presets = {
            "bert-base-uncased":  {"vocab_size": 30522, "hidden_dim": 768,  "intermediate_dim": 3072, "num_heads": 12, "num_layers": 12, "max_position": 512, "dropout": 0.1, "attention_type": "bidirectional", "use_bias": True, "norm_type": "layer", "positional_encoding": "learned", "use_segment_embedding": True},
            "bert-large-uncased": {"vocab_size": 30522, "hidden_dim": 1024, "intermediate_dim": 4096, "num_heads": 16, "num_layers": 24, "max_position": 512, "dropout": 0.1, "attention_type": "bidirectional", "use_bias": True, "norm_type": "layer", "positional_encoding": "learned", "use_segment_embedding": True},
            "gpt2": {"vocab_size": 50257, "hidden_dim": 768, "intermediate_dim": 3072, "num_heads": 12, "num_layers": 12, "max_position": 1024, "dropout": 0.1, "attention_type": "causal", "use_bias": True, "norm_type": "layer", "positional_encoding": "learned", "use_segment_embedding": False},
            "gpt2-large": {"vocab_size": 50257, "hidden_dim": 1280, "intermediate_dim": 5120, "num_heads": 20, "num_layers": 36, "max_position": 1024, "dropout": 0.1, "attention_type": "causal", "use_bias": True, "norm_type": "layer", "positional_encoding": "learned", "use_segment_embedding": False},
            "llama2-7b": {"vocab_size": 32000, "hidden_dim": 4096, "intermediate_dim": 11008, "num_heads": 32, "num_layers": 32, "max_position": 4096, "dropout": 0.0, "attention_type": "causal", "use_bias": False, "norm_type": "rms", "positional_encoding": "rotary", "use_segment_embedding": False},
            "llama3-8b": {"vocab_size": 128256, "hidden_dim": 4096, "intermediate_dim": 14336, "num_heads": 32, "num_layers": 32, "max_position": 8192, "dropout": 0.0, "attention_type": "causal", "use_bias": False, "norm_type": "rms", "positional_encoding": "rotary", "use_segment_embedding": False},
        }
        return cls(name, presets[name])


if __name__ == '__main__':
    import numpy as np

    # BERT Large
    bert_large  = TransformerModel.from_preset("bert-large-uncased")
    #input_ids   = torch.randint(0, 30522, (2, 128))
    #segment_ids = torch.zeros_like(input_ids)

    input_ids   = F._from_shape('input_ids',   [2, 128], np_dtype=np.int64)
    segment_ids = F._from_shape('segment_ids', [2, 128], np_dtype=np.int64)
    out         = bert_large(input_ids, segment_ids)
    print()
    print('-'*40)
    print(out)
    print('='*40)

    """
    # GPT2
    gpt2 = TransformerModel.from_preset("gpt2")
    input_ids = torch.randint(0, 50257, (2, 64))
    out = gpt2(input_ids)

    # Llama 2-7B
    llama2 = TransformerModel.from_preset("llama2-7b")
    input_ids = torch.randint(0, 32000, (2, 128))
    out = llama2(input_ids)
    """
