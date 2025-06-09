#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Any
import math
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

class ATTN(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.dE           = cfg['dE']
        self.nH           = cfg['nH']
        self.idE          = cfg.get('idE', 4*self.dE)
        self.drop_prob    = cfg.get('drop_prob', 0.01)
        self.dH           = self.dE // self.nH
        self.attn_sqrt_dH = F._from_data('sqrt_dH', data=np.float32(math.sqrt(self.dH)))

        #ops
        self.wqkv_proj     = SimNN.Linear   (self.name +'.wqkv_proj', self.dE, 3*self.dE)
        self.wqkv_split    = F.SplitOpHandle(self.name +'.wqkv_split', count=3, axis=2)
        self.qk_softmax    = F.Softmax      (self.name +'.qk_softmax')
        self.drop_attn     = F.Dropout      (self.name +'.drop_attn', self.drop_prob)
        self.w0_proj       = SimNN.Linear   (self.name +'.w0_proj', self.dE, self.dE)
        self.resid_dropout = F.Dropout      (self.name +'.drop_resid', self.drop_prob)

        super().link_op2module()

    def __call__(self, x):
        batch, seqlen, hidden_dim = x.shape
        assert hidden_dim == self.dE, f"Input hidden_dim= {hidden_dim} != {self.name}.dE= {self.dE}"
        WQKV  = self.wqkv_proj(x)
        Q,K,V = self.wqkv_split(WQKV)
        Q     = Q.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2)
        K     = K.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2).transpose(2,3)
        V     = V.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2)
        QK    = T.matmul(Q,K) * self.attn_sqrt_dH
        QK    = self.drop_attn( self.qk_softmax(QK))
        QKV   = T.matmul(QK,V).transpose(1,2).reshape(batch, seqlen, hidden_dim)
        return self.resid_dropout(self.w0_proj(QKV))

    def analytical_param_count(self, lvl):
        param_count  = 0
        param_count += self.wqkv_proj.analytical_param_count(lvl+1)
        param_count += self.w0_proj.analytical_param_count(lvl+1)
        return param_count

class TransformerBlock(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name        = name
        self.dE          = cfg['dE']
        self.nH          = cfg['nH']
        self.idE         = cfg.get('idE', 4*self.dE)
        self.drop_prob   = cfg.get('drop_prob', 0.01)

        #ops
        self.ln_attn_in  = F.LayerNorm (self.name + '.ln_attn_in', self.dE)
        self.attn        = ATTN        (self.name + '.attn', cfg)
        self.lnorm       = F.LayerNorm (self.name + '.ln_mlp_in', self.dE)
        self.ff1         = SimNN.Linear(self.name + '.ff1', self.dE, self.idE)
        self.ff2         = SimNN.Linear(self.name + '.ff2', self.idE, self.dE)
        self.gelu        = F.Gelu      (self.name + '.gelu')
        self.mlp_dropout = F.Dropout   (self.name + '.drop_mlp', self.drop_prob)

        super().link_op2module()

    def __call__(self, x):
        y = x + self.attn(self.ln_attn_in(x))
        y = self.lnorm(y)
        y = self.ff1(y)
        y = self.ff2(y)
        y = self.gelu(y)
        y = self.mlp_dropout(y)
        return y

    def analytical_param_count(self, lvl):
        param_count  = 0
        param_count += self.attn.analytical_param_count(lvl+1)
        param_count += self.ff1.analytical_param_count(lvl+1)
        param_count += self.ff2.analytical_param_count(lvl+1)
        return param_count

class BasicLLM(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.vocab_sz     = cfg['vocab_sz']
        self.bs           = cfg['bs']
        self.nW           = cfg['nW']
        self.nH           = cfg['nH']
        self.dE           = cfg['dE']
        self.nL           = cfg['nL']
        self.drop_prob    = cfg.get('drop_prob',  0.01)
        #we want to simulate large LLMs with small cost...
        # so we'll only simulate for 2 layers...
        self.nL_proxy     = 1 if self.nL > 1 else self.nL

        #input tensors placeholder, populated through create_input_tensors
        #because we may override some fields after constructor like batch-size, etc.
        self.input_tensors = {}

        #ops
        self.wte       = F.Embedding ('wte', self.vocab_sz, self.dE)
        self.wpe       = F.Embedding ('wpe', self.nW, self.dE)
        self.add_emb   = F.Add       ('add_emb')
        self.drop_emb  = F.Dropout   ('drop_emb', self.drop_prob)
        self.tblocks   = SimNN.ModuleList([TransformerBlock('t' + str(i), cfg) for i in range(self.nL_proxy)])

        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
                'tokens': F._from_shape('tokens', [self.bs, self.nW], is_param=False, np_dtype=np.int64),
                'mask': F._from_shape('mask', [self.bs, self.nW], is_param=False, np_dtype=np.int64)
                }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        assert len(self.input_tensors) == 2, f"input_tensors missing!!" + \
                        f"Need create_input_tensors() before __call__: {self.input_tensors}"

        Y = self.wte(self.input_tensors['tokens'])
        Z = self.wpe(self.input_tensors['mask'])
        Y = self.add_emb(Y, Z)
        Y = self.drop_emb(Y)
        for tblock in self.tblocks:
            Y = tblock(Y)

        #Now that all sim_ops have been set, let's update the repeat counts
        for tblock in self.tblocks:
            repeated_ops: dict[str, Any] = {}
            tblock.get_ops(repeated_ops)
            for op_name,op_obj in repeated_ops.items():
                op_obj.repeat_count = self.nL

        return Y

    def analytical_param_count(self, lvl=0):
        #TOTAL PARAM COUNT
        #  For 1 TransformerBlock:
        #        12 * dE * dE
        #        2 LayerNorms with scale, bias params = 2 * dE
        #        Bias Terms: 9
        #          WQKV: 3*dE, W0: dE, FF1: 4*dE, FF2: dE
        #  Embedding Tables:
        #        WTE = vocab_sz * dE
        #        WPE = nW * dE
        #  Therefore Total Params = nL * ( 12 * dE^2 + 10 * dE ) + vocab_sz * dE + nW * dE
        #                         = nL * ( 12 * dE^2 + 10 * dE ) + dE (vocab_sz + nW )
        param_count  = 0
        param_count += sum(x.analytical_param_count(lvl+1) for x in self.tblocks) #type: ignore
        param_count += 4*self.dE #2 LayerNorms @ 2*dE each (bias, variance)
        param_count *= self.nL
        param_count += self.vocab_sz * self.dE
        param_count += self.nW * self.dE
        theoretical = self.nL * (12 * self.dE * self.dE + 13 * self.dE) + self.dE * (self.vocab_sz + self.nW)
        assert param_count == theoretical, f"BAD PARAM COUNT!! layers:{param_count} != theory:{theoretical}"
        return param_count

if __name__ == '__main__':
    nL, nH, dE, nW, vocab_sz, bs = 3, 3, 48, 32, 50257, 1
    gpt_nano_cfg = dict(nL=nL, nH=nH, dE=dE, nW=nW, vocab_sz=vocab_sz, bs=bs)
    gpt_nano = BasicLLM('llm', dict(nL=3, nH=3, dE=48, nW=32, vocab_sz=50257, bs=1))
    gpt_nano.create_input_tensors()
    y = gpt_nano()
    print(y)
    PP = gpt_nano.analytical_param_count()
    print(f"#Params={PP:,d}")
    #gg = gpt_nano.get_forward_graph()
    #gg.graph2onnx('xyxy.onnx')
