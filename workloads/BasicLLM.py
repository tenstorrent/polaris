import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Any
import math
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

class ATTN(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.bs           = cfg['bs']
        self.nW           = cfg['nW']
        self.nH           = cfg['nH']
        self.dE           = cfg['dE']
        self.attn_pdrop   = cfg['attn_pdrop']
        self.resid_pdrop  = cfg['resid_pdrop']
        self.dH           = self.dE // self.nH

        #needed for ops below
        shape0 = [self.bs, self.nW, self.nH, self.dH]
        shape1 = [self.bs, self.nW, self.dE]
        attn_sqrt_K = np.float32(math.sqrt(self.dH))

        #ops
        self.wqkv_proj     = F.Linear       (self.name +'.wqkv_proj', self.dE, 3*self.dE)
        self.wqkv_split    = F.SplitOpHandle(self.name +'.wqkv_split', count=3, axis=2)
        self.q_reshape     = F.ReshapeFixed (self.name +'.q_reshape', shape0    )
        self.q_transpose   = F.Transpose    (self.name +'.q_transpose', perm=[0,2,1,3])
        self.k_reshape     = F.ReshapeFixed (self.name +'.k_reshape', shape0   )
        self.k_transpose   = F.Transpose    (self.name +'.k_transpose', perm=[0,2,3,1])
        self.v_reshape     = F.ReshapeFixed (self.name +'.v_reshape', shape0   )
        self.v_transpose   = F.Transpose    (self.name +'.v_transpose', perm=[0,2,1,3])
        self.qk_matmul     = F.MatMul       (self.name +'.qk_matmul')
        self.qk_mul        = F.MulFixed     (self.name +'.qk_mul', 'attn_sqrt_K', attn_sqrt_K)
        self.qk_softmax    = F.Softmax      (self.name +'.qk_softmax')
        self.drop_attn     = F.Dropout      (self.name +'.drop_attn', self.attn_pdrop)
        self.qkv_matmul    = F.MatMul       (self.name +'.qkv_matmul')
        self.w0_proj_blk   = F.SimOpHandleList([
            F.Transpose    (self.name +'.qkv_transpose', perm=[0,2,1,3]),
            F.ReshapeFixed (self.name +'.qkv_reshape', shape1  ),
            F.Linear       (self.name +'.w0_proj', self.dE, self.dE),
            F.Bias         (self.name +'.w0_bias', [self.dE]),
            F.Dropout      (self.name +'.drop_resid', self.resid_pdrop)
            ])
        super().link_op2module()

    def __call__(self, x):
        WQKV = self.wqkv_proj(x)
        WQKV = self.wqkv_split(WQKV)
        Q    = self.q_transpose(self.q_reshape(WQKV[0]))
        K    = self.k_transpose(self.k_reshape(WQKV[1]))
        V    = self.v_transpose(self.v_reshape(WQKV[2]))
        QK   = self.drop_attn(
                self.qk_softmax(
                    self.qk_mul(
                        self.qk_matmul(Q,K)
                        )
                    )
                )
        QKV  = self.qkv_matmul   (QK,V)
        return self.w0_proj_blk(QKV)

class TransformerBlock(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name         = name
        self.dE           = cfg['dE']
        self.mlp_pdrop    = cfg['mlp_pdrop']

        #ops
        self.ln_attn_in    = F.LayerNorm (self.name + '.ln_attn_in', self.dE)
        self.attn          = ATTN(self.name + '.attn', cfg)
        self.mlp_in        = F.Add (self.name + '.mlp_in')
        self.mlp_blk       = F.SimOpHandleList([
            F.LayerNorm (self.name + '.ln_mlp_in', self.dE),
            F.Linear    (self.name + '.ff1', self.dE, 4*self.dE),
            F.Bias      (self.name + '.ff1_bias', [4*self.dE]),
            F.Linear    (self.name + '.ff2', 4*self.dE, self.dE),
            F.Gelu      (self.name + '.gelu'),
            F.Bias      (self.name + '.ff2_bias', [self.dE]),
            F.Dropout   (self.name + '.drop_mlp', self.mlp_pdrop)
            ])

        super().link_op2module()

    def __call__(self, x):
        y = self.attn(self.ln_attn_in(x))
        z = self.mlp_in(x, y)
        return self.mlp_blk(z)

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
        self.embd_pdrop   = cfg['embd_pdrop']
        self.attn_pdrop   = cfg['attn_pdrop']
        self.resid_pdrop  = cfg['resid_pdrop']
        self.mlp_pdrop    = cfg['mlp_pdrop']
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
        self.drop_emb  = F.Dropout   ('drop_emb', self.embd_pdrop)
        self.tblocks   = SimNN.ModuleList([TransformerBlock('t' + str(i), cfg) for i in range(self.nL_proxy)])

        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
                'tokens': F._from_shape('tokens',
                                        [self.bs, self.nW],
                                        is_param=False, np_dtype=np.int64),
                'mask': F._from_data('mask',
                                      np.array([1 for _ in range(self.bs * self.nW)], dtype=np.int64).reshape(self.bs, self.nW),
                                      is_param=False)
                }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        assert len(self.input_tensors) == 2, f"input_tensors missing!! Need create_input_tensors() before __call__: {self.input_tensors}"

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

    def analytical_param_count(self):
        #TOTAL PARAM COUNT
        #  For 1 TransformerBlock:
        #        12 * dE * dE
        #        2 LayerNorms with scale, bias params = 2 * dE
        #        FF1 bias = 4 * dE
        #        FF2 bias = dE
        #        W0 bias = dE
        #  Embedding Tables:
        #        WTE = vocab_sz * dE
        #        WPE = nW * dE
        #  Therefore Total Params = nL * ( 12 * dE^2 + 10 * dE ) + vocab_sz * dE + nW * dE 
        #                         = nL * ( 12 * dE^2 + 10 * dE ) + dE (vocab_sz + nW )
        return self.nL * (12 * self.dE * self.dE + 10 * self.dE) + self.dE * (self.vocab_sz + self.nW)
