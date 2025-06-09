#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from ttsim.ops import SimTensor

from workloads.llm.rope import RotaryEmbedding

import math
import numpy as np
from typing import Optional, Callable, Tuple

class ConfigurableAttention(SimNN.Module):
    def __init__(self, name: str, cfg: dict, pos_enc_fn: Optional[Callable] = None):
        super().__init__()
        self.name       = name
        self.nH         = cfg['nH']
        self.dE         = cfg['dE']
        self.attn_type  = cfg.get('attn_type', 'bidir')
        self.use_bias   = cfg.get('use_bias',  True)
        self.drop_prob  = cfg.get('drop_prob', 0.0)
        self.mask_fn    = cfg.get('mask_fn',   None)
        self.use_rope   = cfg.get('use_rope',  False)
        self.use_cache  = cfg.get('use_cache', False)
        self.dH         = self.dE // self.nH
        self.pos_enc_fn = pos_enc_fn

        #ops
        self.rotary_emb = RotaryEmbedding(name + '.rope') if self.use_rope else None
        self.wqkv_proj  = SimNN.Linear(name + '.wqkv_proj', self.dE, 3*self.dE, bias=self.use_bias)
        self.wqkv_split = F.SplitOpHandle(name +'.wqkv_split', count=3, axis=2)
        self.qk_softmax = F.Softmax(name +'.qk_softmax')
        self.drop_attn  = F.Dropout(name +'.drop_attn', self.drop_prob)
        self.out_proj   = SimNN.Linear(name + '.out_proj', self.dE, self.dE, bias=self.use_bias)

        super().link_op2module()

    def _make_attention_mask(self, qlen, klen, attention_type, custom_mask=None, key_padding_mask=None):
        if attention_type == 'causal':
            # Causal (future masked): [qlen, klen]
            #create an upper triangual matrix QxK of ones...
            #mask      = torch.triu(torch.ones(qlen, klen), diagonal=1).bool()
            #create an QxK matrix filled with 0
            #attn_mask = torch.zeros((qlen, klen))
            #Put future values to -inf using the upper triangular mask
            #attn_mask.masked_fill_(mask, float('-inf'))
            attn_mask = F._from_shape('causal_attn_mask', [qlen, klen], is_const=True)
            attn_mask.set_module(self)
            self._tensors[attn_mask.name] = attn_mask
        elif attention_type == 'bidir':
            attn_mask = None
        elif attention_type == 'custom_mask' and custom_mask is not None:
            attn_mask = custom_mask  # [batch, num_heads, qlen, klen] or [qlen, klen]
        else:
            raise NotImplementedError(f"Attention type {attention_type} not implemented.")

        # Add key padding mask if provided
        if key_padding_mask is not None:
            assert False, "HANDLE KEY PADDING"
            # key_padding_mask: [batch, klen] where True = PAD
            # CODE if attn_mask is None:
            # CODE     attn_mask = torch.zeros((qlen, klen), device=device)
            # Add mask for each batch (broadcast if needed)
            # CODE attn_mask   = attn_mask.unsqueeze(0)  # [1, qlen, klen]
            # CODE mask_expand = key_padding_mask.unsqueeze(1)  # [batch, 1, klen]
            # CODE attn_mask   = attn_mask + mask_expand * float('-inf')
        return attn_mask

    def __call__(self,
                 x: SimTensor,
                 kv: Optional[SimTensor] = None,
                 mask: Optional[SimTensor] = None,
                 key_padding_mask: Optional[SimTensor] = None,
                 past_kv: Optional[Tuple[SimTensor, SimTensor]] = None,
    ) -> Tuple[SimTensor, SimTensor, Optional[Tuple[SimTensor, SimTensor]]]:
        """
        Args:
            qlen       = nW : seq-length for prefill phase
            hidden_dim = dE
            klen       = seq-length for decode/generate phase
            x: [B, qlen, hidden_dim] (queries)
            kv: [B, klen, hidden_dim] (keys/values), if cross-attention. If None, use x.
            mask: optional custom attention mask (custom_mask flavor)
            key_padding_mask: [B, klen], True=pad
            past_kv: Optional, for caching past K/V (incremental decoding)

        Returns:
            output: [B, qlen, hidden_dim]
            attn_weights: [B, num_heads, qlen, klen]
            present_kv: (K, V) for caching
        """
        assert x.rank() == 3, f"{self.name} input {x.name}.shape= {x.shape} not in [bs, nW, dE] shape"
        batch, qlen, hidden_dim = x.shape
        assert hidden_dim == self.dE, f"Input {x.name}.hidden_dim= {hidden_dim} != {self.name}.dE= {self.dE}"

        kv         = x if kv is None else kv
        _, klen, _ = kv.shape

        # Project Q, K, V
        WQKV  = self.wqkv_proj(x)
        Q,K,V = self.wqkv_split(WQKV)
        Q     = Q.reshape(batch, qlen, self.nH, self.dH).transpose(1,2)
        K     = K.reshape(batch, qlen, self.nH, self.dH).transpose(1,2)
        V     = V.reshape(batch, qlen, self.nH, self.dH).transpose(1,2)

        # Rotary embeddings (LLama/NeoX, etc.)
        if self.rotary_emb is not None:
            Q = self.rotary_emb(Q)
            K = self.rotary_emb(K)

        # Positional encoding (classic BERT/transformer-style, if any)
        if self.pos_enc_fn is not None:
            Q, K = self.pos_enc_fn(Q, K)  # User-defined function

        # Key-value cache for inference
        if past_kv is not None:
            past_K, past_V = past_kv
            K = T.cat([past_K, K], dim=2)
            V = T.cat([past_V, V], dim=2)
            klen = K.shape[2]

        # Attention scores: [B, num_heads, qlen, klen]
        attn_scores = T.matmul(Q, K.transpose(-2, -1)) / F._from_data(self.name + '.sqrt_dH',
                                                                      is_const=True,
                                                                      data=np.float32(math.sqrt(self.dH)))
        # Generate mask if needed
        if self.attn_type == 'causal':
            attn_mask = self._make_attention_mask(qlen, klen, 'causal', key_padding_mask=key_padding_mask)
        elif self.attn_type == 'bidir':
            attn_mask = self._make_attention_mask(qlen, klen, 'bidir', key_padding_mask=key_padding_mask)
        elif self.attn_type == 'custom':
            attn_mask = self._make_attention_mask(qlen, klen, 'custom', custom_mask=mask, key_padding_mask=key_padding_mask)
        else:
            raise NotImplementedError(f"Unknown attention_type {self.attn_type}")

        # Mask broadcasting: attn_mask shape could be [qlen, klen], [batch, 1, qlen, klen], etc.
        if attn_mask is not None:
            # attn_scores: [B, num_heads, qlen, klen]
            if attn_mask.rank() == 2:  # [qlen, klen]
                attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.rank() == 3:  # [B, qlen, klen]
                attn_scores = attn_scores + attn_mask.unsqueeze(1)
            elif attn_mask.rank() == 4:  # [B, num_heads, qlen, klen]
                attn_scores = attn_scores + attn_mask
            else:
                raise ValueError("attn_mask shape not supported")

        attn_weights = self.qk_softmax(attn_scores)#, dim=-1) #What does dim do in torch.softmax?
        attn_weights = self.drop_attn(attn_weights)

        # Output: [B, num_heads, qlen, head_dim]
        attn_output = T.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch, qlen, hidden_dim)

        output = self.out_proj(attn_output)

        # For generation, output cache
        present_kv = (K, V) if self.use_cache else None

        return output, attn_weights, present_kv

if __name__ == '__main__':
    #modelname = 'bert'
    modelname = 'gpt'
    if modelname == 'bert':
        bert_config = {'nH': 12, 'dE': 768, 'attn_type': 'bidir', 'drop_prob': 0.1, 'use_bias': True}
        bert_inputs = {'x': F._from_shape('bert_in', [2, 16, 768]) }
        bert_attn   = ConfigurableAttention('bert_attn', bert_config)
        out         = bert_attn(bert_inputs['x'])
        print("BERT INPUTS:")
        for k,v in bert_inputs.items(): print('    ',k,v)
        print("BERT OUTPUTS:")
        for x in out: print('    ',x)

        gg = bert_attn._get_forward_graph(bert_inputs)
        gg.graph2onnx(bert_attn.name + '.onnx', do_model_check=True)
    elif modelname == 'gpt':
        gpt_config = {'nH': 12, 'dE': 768, 'attn_type': 'causal', 'drop_prob': 0.1, 'use_bias': True,
                       'use_rope': True, 'use_cache': True}
        gpt_inputs = {'x': F._from_shape('gpt_in', [2, 20, 768])}
        gpt_attn   = ConfigurableAttention('gpt_attn', gpt_config)
        out        = gpt_attn(gpt_inputs['x'])
        print("GPT INPUTS:")
        for k,v in gpt_inputs.items(): print('    ',k,v)
        print("GPT OUTPUTS:")
        for x in out: print('    ',x)

        gg = gpt_attn._get_forward_graph(gpt_inputs)
        gg.graph2onnx(gpt_attn.name + '.onnx', do_model_check=False) #Because of Slice attr in ttsim/ops/op.py
    else:
        pass
        # Example: Cross-Attention (Encoder-Decoder)
        #
        # Cross-attention (for decoders)
        # cross_attn =  ConfigurableAttention({**bert_config, 'attn_type': 'bidir'})  # Use kv != x in forward
        # out, attn_weights, _ = cross_attn(x, kv=enc)
        #
        # Example: Longformer-style custom mask
        #
        # mask = torch.zeros(batch, 1, qlen, klen, device=device)
        # mask[..., :, klen//2:] = float('-inf')  # Only attend to first half of k
        # mask = my_custom_mask(2, 16, 16, x.device)
        # longformer_attn =  ConfigurableAttention({**bert_config, 'attn_type': 'custom', 'mask_fn': mask_fn})
        # out, attn_weights, _ = longformer_attn(x, mask=mask)
