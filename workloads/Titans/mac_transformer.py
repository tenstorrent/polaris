#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim port of titans_pytorch.mac_transformer

- SegmentedAttention   : local/segment causal attention + persistent memory K/V
- MemoryAsContextTransformer : full MAC stack
- TitansMAC            : Polaris workload entry class (mirrors BasicLLM's shape)
"""
import os, sys, math
from typing import Any
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN

from workloads.Titans.neural_memory import NeuralMemory, _RMSNorm
from workloads.Titans.memory_models import MemoryMLP

# ---------------------------------------------------------------------------
# FFN: RMSNorm -> Linear -> GEGLU -> Linear
# ---------------------------------------------------------------------------
class FeedForward(SimNN.Module):
    def __init__(self, name, dim, mult=4):
        super().__init__()
        self.name = name
        dim_inner = int(dim * mult * 2 / 3)
        self.norm  = _RMSNorm(f'{name}.norm', dim)
        self.proj_in  = SimNN.Linear(f'{name}.proj_in', dim, dim_inner * 2)
        self.split    = F.SplitOpHandle(f'{name}.split', count=2, axis=2)
        self.silu     = F.Sigmoid(f'{name}.silu_sig')   # we use sigmoid * x as silu
        self.mul_silu = F.Mul(f'{name}.mul_silu')
        self.mul_gate = F.Mul(f'{name}.mul_gate')
        self.proj_out = SimNN.Linear(f'{name}.proj_out', dim_inner, dim)
        self.dim = dim
        self.dim_inner = dim_inner
        super().link_op2module()

    def __call__(self, x):
        x = self.norm(x)
        h = self.proj_in(x)
        a, gate = self.split(h)
        gate_silu = self.mul_silu(gate, self.silu(gate))   # silu(gate)
        h = self.mul_gate(a, gate_silu)
        return self.proj_out(h)

    def analytical_param_count(self, lvl=0):
        pc  = self.proj_in.analytical_param_count(lvl + 1)
        pc += self.proj_out.analytical_param_count(lvl + 1)
        pc += self.norm.analytical_param_count(lvl + 1)
        return pc

# ---------------------------------------------------------------------------
# SegmentedAttention (local/segment causal attention + persistent K/V)
# ---------------------------------------------------------------------------
class SegmentedAttention(SimNN.Module):
    def __init__(self, name, dim, segment_len,
                 num_persist_mem_tokens=0, num_longterm_mem_tokens=0,
                 dim_head=64, heads=8):
        super().__init__()
        self.name = name
        self.dim  = dim
        self.dim_head = dim_head
        self.heads    = heads
        self.dim_inner= dim_head * heads
        self.segment_len             = segment_len
        self.num_persist_mem_tokens  = num_persist_mem_tokens
        self.num_longterm_mem_tokens = num_longterm_mem_tokens
        self.total_segment_len       = segment_len + num_longterm_mem_tokens

        self.norm   = _RMSNorm(f'{name}.norm', dim)
        self.to_qkv = SimNN.Linear(f'{name}.to_qkv', dim, self.dim_inner * 3)
        self.split  = F.SplitOpHandle(f'{name}.qkv_split', count=3, axis=2)
        self.softmax= F.Softmax(f'{name}.softmax')
        self.to_out = SimNN.Linear(f'{name}.to_out', self.dim_inner, dim)
        self.attn_scale = F._from_data(f'{name}.attn_scale',
                                       data=np.float32(1.0 / math.sqrt(dim_head)))
        self.qk_mm = F.MatMul(f'{name}.qk_mm')
        self.av_mm = F.MatMul(f'{name}.av_mm')

        # persistent memory K/V (learnable, fixed-len, shared per head)
        if num_persist_mem_tokens > 0:
            self.persistent_k = F._from_shape(f'{name}.persistent_k',
                                              shape=[1, heads, num_persist_mem_tokens, dim_head],
                                              is_param=True)
            self.persistent_v = F._from_shape(f'{name}.persistent_v',
                                              shape=[1, heads, num_persist_mem_tokens, dim_head],
                                              is_param=True)
        else:
            self.persistent_k = None  # type: ignore[assignment]
            self.persistent_v = None  # type: ignore[assignment]
        super().link_op2module()

    def __call__(self, x):
        B, N, D = x.shape
        H, Dh   = self.heads, self.dim_head
        x = self.norm(x)
        qkv = self.to_qkv(x)
        Q, K, V = self.split(qkv)
        Q = Q.reshape(B, N, H, Dh).transpose(1, 2)            # [B, H, N, Dh]
        K = K.reshape(B, N, H, Dh).transpose(1, 2)
        V = V.reshape(B, N, H, Dh).transpose(1, 2)

        # prepend persistent memory K/V
        if self.persistent_k is not None:
            pmk = self.persistent_k  
            pmv = self.persistent_v
            K = T.cat([pmk, K], dim=-2)
            V = T.cat([pmv, V], dim=-2)

        scores = self.qk_mm(Q, K.transpose(-1, -2)) * self.attn_scale
        attn   = self.softmax(scores)
        out    = self.av_mm(attn, V)                          # [B, H, N, Dh]
        out    = out.transpose(1, 2).reshape(B, N, H * Dh)
        return self.to_out(out)

    def analytical_param_count(self, lvl=0):
        pc  = self.to_qkv.analytical_param_count(lvl + 1)
        pc += self.to_out.analytical_param_count(lvl + 1)
        pc += self.norm.analytical_param_count(lvl + 1)
        if self.persistent_k is not None:
            pc += 2 * self.heads * self.num_persist_mem_tokens * self.dim_head
        return pc

# ---------------------------------------------------------------------------
# Single MAC block:  NeuralMemory -> SegmentedAttention -> FFN  (residual each)
# ---------------------------------------------------------------------------
class MACBlock(SimNN.Module):
    def __init__(self, name, dim, segment_len, dim_head, heads, ff_mult,
                 num_persist_mem_tokens, num_longterm_mem_tokens,
                 use_neural_memory, neural_memory_segment_len,
                 mem_depth, mem_expansion_factor, mem_heads, mem_dim_head,
                 mem_chunk_size,
                 max_chunks_unroll = 16):
        super().__init__()
        self.name = name
        self.use_mem = use_neural_memory
        if use_neural_memory:
            self.mem = NeuralMemory(
                name       = f'{name}.mem',
                dim        = dim,
                chunk_size = mem_chunk_size,
                dim_head   = mem_dim_head,
                heads      = mem_heads,
                depth      = mem_depth,
                expansion_factor = mem_expansion_factor,
                max_chunks_unroll = max_chunks_unroll,
            )
            self.add_mem = F.Add(f'{name}.add_mem')
        else:
            self.mem = None  # type: ignore[assignment]

        self.attn    = SegmentedAttention(
            name=f'{name}.attn', dim=dim, segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            dim_head=dim_head, heads=heads)
        self.add_attn= F.Add(f'{name}.add_attn')
        self.ff      = FeedForward(f'{name}.ff', dim, mult=ff_mult)
        self.add_ff  = F.Add(f'{name}.add_ff')
        super().link_op2module()

    def __call__(self, x, mem_state=None):
        if self.use_mem:
            retrieved, mem_state = self.mem(x, state=mem_state)
            x = self.add_mem(x, retrieved)
        x = self.add_attn(x, self.attn(x))
        x = self.add_ff  (x, self.ff(x))
        return x, mem_state

    def analytical_param_count(self, lvl=0):
        pc = 0
        if self.use_mem:
            pc += self.mem.analytical_param_count(lvl + 1)
        pc += self.attn.analytical_param_count(lvl + 1)
        pc += self.ff.analytical_param_count(lvl + 1)
        return pc

# ---------------------------------------------------------------------------
# MemoryAsContextTransformer  
# ---------------------------------------------------------------------------
class MemoryAsContextTransformer(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg  = cfg
        self.vocab_sz = cfg['vocab_sz']
        self.dim      = cfg['dE']
        self.nL       = cfg['nL']
        self.nL_proxy = 1 if self.nL > 1 else self.nL   # BasicLLM-style proxy
        self.nH       = cfg['nH']
        self.dim_head = cfg.get('dim_head', self.dim // self.nH)
        self.nW       = cfg['nW']                       # seq_len
        self.bs       = cfg['bs']
        self.ff_mult  = cfg.get('ff_mult', 4)
        self.segment_len            = cfg['segment_len']
        self.num_persist_mem_tokens = cfg.get('num_persist_mem_tokens', 0)
        self.num_longterm_mem_tokens= cfg.get('num_longterm_mem_tokens', 0)
        self.neural_memory_layers   = set(cfg.get('neural_memory_layers',
                                                   list(range(1, self.nL_proxy + 1))))
        self.neural_memory_segment_len = cfg.get('neural_memory_segment_len',
                                                  self.segment_len + self.num_longterm_mem_tokens)
        self.mem_depth        = cfg.get('mem_depth', 2)
        self.mem_expansion    = cfg.get('mem_expansion_factor', 2.0)
        self.mem_heads        = cfg.get('mem_heads', 4)
        self.mem_dim_head     = cfg.get('mem_dim_head', 64)
        self.mem_chunk_size   = cfg.get('mem_chunk_size', self.neural_memory_segment_len)
        self.max_chunks_unroll = cfg.get('max_chunks_unroll', 16)

        # absolute positional embedding (learned; replaces axial pos emb)
        self.wte = F.Embedding('wte', self.vocab_sz, self.dim)
        self.wpe = F.Embedding('wpe', self.nW, self.dim)
        self.add_emb = F.Add('add_emb')

        # longterm memory tokens (learnable)
        if self.num_longterm_mem_tokens > 0:
            self.longterm_mems = F._from_shape(
                'longterm_mems',
                shape=[1, self.num_longterm_mem_tokens, self.dim], is_param=True)
        else:
            self.longterm_mems = None  # type: ignore[assignment]

        # Two proxy blocks: one represents memory-bearing layers, the other plain
        # transformer layers. Each gets its own repeat_count equal to the actual
        # number of layers of that type in the full stack. The BasicLLM single-proxy
        # trick would silently drop memory ops because (i+1=1) is not in [2,4,6].
        mem_layer_set = set(self.neural_memory_layers)
        all_layers    = set(range(1, self.nL + 1))
        self._n_mem_layers   = len(mem_layer_set & all_layers)
        self._n_nomem_layers = self.nL - self._n_mem_layers

        self.blocks = SimNN.ModuleList([
            MACBlock(
                name='t_mem',
                dim=self.dim, segment_len=self.segment_len,
                dim_head=self.dim_head, heads=self.nH, ff_mult=self.ff_mult,
                num_persist_mem_tokens=self.num_persist_mem_tokens,
                num_longterm_mem_tokens=self.num_longterm_mem_tokens,
                use_neural_memory=True,
                neural_memory_segment_len=self.neural_memory_segment_len,
                mem_depth=self.mem_depth,
                mem_expansion_factor=self.mem_expansion,
                mem_heads=self.mem_heads,
                mem_dim_head=self.mem_dim_head,
                mem_chunk_size=self.mem_chunk_size,
                max_chunks_unroll=self.max_chunks_unroll,
            ),
            MACBlock(
                name='t_nomem',
                dim=self.dim, segment_len=self.segment_len,
                dim_head=self.dim_head, heads=self.nH, ff_mult=self.ff_mult,
                num_persist_mem_tokens=self.num_persist_mem_tokens,
                num_longterm_mem_tokens=self.num_longterm_mem_tokens,
                use_neural_memory=False,
                neural_memory_segment_len=self.neural_memory_segment_len,
                mem_depth=self.mem_depth,
                mem_expansion_factor=self.mem_expansion,
                mem_heads=self.mem_heads,
                mem_dim_head=self.mem_dim_head,
                mem_chunk_size=self.mem_chunk_size,
                max_chunks_unroll=self.max_chunks_unroll,
            ),
        ])
        self._block_repeat_counts = [self._n_mem_layers, self._n_nomem_layers]

        self.final_norm = _RMSNorm('final_norm', self.dim)
        self.to_logits  = SimNN.Linear('to_logits', self.dim, self.vocab_sz, bias=False) \
            if hasattr(SimNN.Linear, 'bias') else SimNN.Linear('to_logits', self.dim, self.vocab_sz)

        self.input_tensors = {}
        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
            'tokens': F._from_shape('tokens', [self.bs, self.nW], is_param=False, np_dtype=np.int64),
            'mask'  : F._from_shape('mask',   [self.bs, self.nW], is_param=False, np_dtype=np.int64),
        }

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def __call__(self):
        assert len(self.input_tensors) == 2, "call create_input_tensors() first"
        tok = self.input_tensors['tokens']
        pos = self.input_tensors['mask']
        x = self.add_emb(self.wte(tok), self.wpe(pos))

        if self.longterm_mems is not None:
            x = T.cat([self.longterm_mems, x], dim=1)   # rely on broadcast over batch dim

        # Run each proxy block once to build the graph. Each block carries its own
        # NeuralMemState (None on first call -- D3 will thread it across
        # MemoryAsContextTransformer.__call__ invocations for autoregressive decode).
        for blk in self.blocks:
            x, _ = blk(x, mem_state=None)

        # Set repeat_count per block to the actual count of layers of that type.
        for blk, rc in zip(self.blocks, self._block_repeat_counts):
            if rc == 0:
                continue   # nothing to count for this proxy
            repeated_ops: dict[str, Any] = {}
            blk.get_ops(repeated_ops)
            for _, op_obj in repeated_ops.items():
                op_obj.repeat_count = rc

        x = self.final_norm(x)
        return self.to_logits(x)

    def analytical_param_count(self, lvl=0):
        pc = 0
        pc += self.vocab_sz * self.dim    # wte
        pc += self.nW * self.dim          # wpe
        if self.longterm_mems is not None:
            pc += self.num_longterm_mem_tokens * self.dim
        pc += sum(b.analytical_param_count(lvl + 1) for b in self.blocks) * self.nL  # type: ignore[attr-defined]
        pc += self.dim                    # final RMSNorm
        pc += self.dim * self.vocab_sz    # logits head
        return pc


class TitansMAC(MemoryAsContextTransformer):
    """Polaris workload class. Same surface as BasicLLM:
       __init__(name, cfg) / set_batch_size / create_input_tensors /
       get_forward_graph / __call__ / analytical_param_count.
    """
    pass


if __name__ == '__main__':
    cfg = dict(
        nL=2, nH=4, dE=128, dim_head=32, nW=32, vocab_sz=256, bs=1,
        segment_len=8,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        neural_memory_layers=[2],
        neural_memory_segment_len=4,
        mem_depth=2, mem_expansion_factor=2.0, mem_heads=2, mem_dim_head=32,
        mem_chunk_size=4,
    )
    m = TitansMAC('titans_mac', cfg)
    m.create_input_tensors()
    y = m()
    print('Output shape:', y.shape)
    print(f'#Params = {m.analytical_param_count():,d}')
