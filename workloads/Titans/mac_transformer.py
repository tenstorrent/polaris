#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim port of titans_pytorch.mac_transformer -- a forward-only cost model.

Classes:
- FeedForward                 : dim -> ff_mult*dim -> dim MLP
- SegmentedAttention          : segment-local causal attention + persistent
                                memory K/V ("full" mode also available)
- MACBlock                    : attention + optional NeuralMemory + FeedForward
- MemoryAsContextTransformer  : embeddings, transformer body, output projection
- TitansMAC                   : Polaris workload entry point

The body is NOT nL physical layers. It is two proxy blocks - one with a
neural memory (t_mem) and one without (t_nomem) - whose op stats are scaled
by repeat counts derived from neural_memory_layers. Repeated layers are
cost-identical, so this is exact for cost projection while keeping the graph
O(1) in nL: nL=24 builds ~500 ops, not 24 layers' worth. Both cycles and
analytical_param_count() apply the same repeat counts.
"""
import math
from typing import Any

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops.tensor import SimTensor

from workloads.Titans.neural_memory import NeuralMemory, _RMSNorm

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
        gate_silu = self.mul_silu(gate, self.silu(gate))
        h = self.mul_gate(a, gate_silu)
        return self.proj_out(h)

    def analytical_param_count(self, lvl=0):
        pc  = self.proj_in.analytical_param_count(lvl + 1)
        pc += self.proj_out.analytical_param_count(lvl + 1)
        pc += self.norm.analytical_param_count(lvl + 1)
        return pc

class SegmentedAttention(SimNN.Module):
    def __init__(self, name, dim, segment_len,
                 num_persist_mem_tokens=0, num_longterm_mem_tokens=0,
                 dim_head=64, heads=8, attention_mode="segmented"):   # default mode:segmented
        super().__init__()
        self.name = name
        self.dim  = dim
        self.dim_head = dim_head
        self.heads    = heads
        self.attention_mode = attention_mode
        assert attention_mode in ("segmented", "full")  #Default "segmented" so nothing changes unless explicitly set to "full".
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
        self.persistent_k: SimTensor | None = None
        self.persistent_v: SimTensor | None = None

        if num_persist_mem_tokens > 0:
            self.persistent_k = F._from_shape(f'{name}.persistent_k',
                                              shape=[1, heads, num_persist_mem_tokens, dim_head],
                                              is_param=True)
            self.persistent_v = F._from_shape(f'{name}.persistent_v',
                                              shape=[1, heads, num_persist_mem_tokens, dim_head],
                                              is_param=True)
            self.persistent_k.set_module(self); self._tensors[self.persistent_k.name] = self.persistent_k
            self.persistent_v.set_module(self); self._tensors[self.persistent_v.name] = self.persistent_v
            # persistent-attention op handles (broadcast-term softmax, no .repeat)
            self.qk_mm_persist = F.MatMul(f'{name}.qk_mm_persist')
            self.av_mm_persist = F.MatMul(f'{name}.av_mm_persist')
            self.exp_seg   = F.Exp(f'{name}.exp_seg')
            self.exp_per   = F.Exp(f'{name}.exp_per')
            self.sum_seg   = F.ReduceSum(f'{name}.sum_seg', axis=-1)
            self.sum_per   = F.ReduceSum(f'{name}.sum_per', axis=-1)
            self.add_denom = F.Add(f'{name}.add_denom')
            self.div_seg   = F.Div(f'{name}.div_seg')
            self.div_per   = F.Div(f'{name}.div_per')
            self.add_out   = F.Add(f'{name}.add_out')
        else:
            self.persistent_k = None
            self.persistent_v = None
        super().link_op2module()

    # ---- fold/unfold: verified numerically exact vs reference (0.0 diff) ----
    def _to_heads(self, t, B, N, H, Dh):
        return t.reshape(B, N, H, Dh).transpose(1, 2)              # [B,H,N,Dh]

    def _segment(self, t, B, H, W, S, Dh):
        # [B,H,N,Dh] -> [B*W,H,S,Dh]   ref: 'b h (w n) d -> (b w) h n d'
        return t.reshape(B, H, W, S, Dh).transpose(1, 2).reshape(B * W, H, S, Dh)

    def _unsegment(self, t, B, H, W, S, Dh):
        # [B*W,H,S,Dh] -> [B,H,N,Dh]   (inverse)
        return t.reshape(B, W, H, S, Dh).transpose(1, 2).reshape(B, H, W * S, Dh)

    def _local_attention(self, Q, K, V):
        """
        Segment-local attention. Persistent K/V attended via broadcasting matmuls
        with a shared softmax denominator no .repeat, so no replicated-parameter
        write traffic (avoids the memory-bandwidth validator failure that physical
        per-segment replication triggers on the larger instances).
        Q, K, V: [B*W, H, S, Dh] returns [B*W, H, S, Dh]
        """
        scores_seg = self.qk_mm(Q, K.transpose(-1, -2)) * self.attn_scale   # [BW,H,S,S]

        if self.persistent_k is None:
            return self.av_mm(self.softmax(scores_seg), V)

        # persistent scores via broadcast: Q[BW,H,S,Dh] x pk^T[1,H,Dh,P] -> [BW,H,S,P]
        # transpose is patched onto SimTensor at import time
        # (ttsim/front/functional/tensor_op.py), so mypy cannot see it. The
        # ignore is required here and not on K/V above because persistent_k is
        # annotated SimTensor | None, while Q/K/V are unannotated params (Any).
        pk_t = self.persistent_k.transpose(-1, -2)   # type: ignore[attr-defined] # [1,H,Dh,P]
        scores_per = self.qk_mm_persist(Q, pk_t) * self.attn_scale          # [BW,H,S,P]

        # joint softmax over (S + P) keys, via shared denominator (no concat/split):
        e_seg = self.exp_seg(scores_seg)                                    # [BW,H,S,S]
        e_per = self.exp_per(scores_per)                                    # [BW,H,S,P]
        denom = self.add_denom(self.sum_seg(e_seg), self.sum_per(e_per))    # [BW,H,S,1]
        attn_seg = self.div_seg(e_seg, denom)                               # [BW,H,S,S]
        attn_per = self.div_per(e_per, denom)                               # [BW,H,S,P]

        # value terms: segment normal, persistent broadcast (pv [1,H,P,Dh])
        out_seg = self.av_mm(attn_seg, V)                                   # [BW,H,S,Dh]
        out_per = self.av_mm_persist(attn_per, self.persistent_v)           # [BW,H,S,Dh]
        return self.add_out(out_seg, out_per)                               # [BW,H,S,Dh]

    def _full_attention(self, Q, K, V):
        """Full O(N^2) attention baseline. Reuses _local_attention on the
        un-segmented sequence (batch=B, seq=N) -- every token attends every
        token plus persistent memory (same broadcast-term handling)."""
        return self._local_attention(Q, K, V)

    def __call__(self, x):
        B, N, D = x.shape
        H, Dh   = self.heads, self.dim_head
        x = self.norm(x)
        Q, K, V = self.split(self.to_qkv(x))
        Q = self._to_heads(Q, B, N, H, Dh)
        K = self._to_heads(K, B, N, H, Dh)
        V = self._to_heads(V, B, N, H, Dh)
        if self.attention_mode == "full":
            out = self._full_attention(Q, K, V)          # [B,H,N,Dh]
        else:
            S = self.total_segment_len
            assert N % S == 0, \
                f"N={N} must be divisible by total_segment_len={S} " \
                f"(segment_len={self.segment_len} + num_longterm_mem_tokens={self.num_longterm_mem_tokens})"
            W = N // S
            Q = self._segment(Q, B, H, W, S, Dh)
            K = self._segment(K, B, H, W, S, Dh)
            V = self._segment(V, B, H, W, S, Dh)

            out = self._local_attention(Q, K, V)           # [B*W,H,S,Dh]
            out = self._unsegment(out, B, H, W, S, Dh)     # [B,H,N,Dh]
        out = out.transpose(1, 2).reshape(B, N, H * Dh)
        return self.to_out(out)

    def analytical_param_count(self, lvl=0):
        pc  = self.to_qkv.analytical_param_count(lvl + 1)
        pc += self.to_out.analytical_param_count(lvl + 1)
        pc += self.norm.analytical_param_count(lvl + 1)
        if self.persistent_k is not None:
            pc += 2 * self.heads * self.num_persist_mem_tokens * self.dim_head
        return pc

class MACBlock(SimNN.Module):
    def __init__(self, name, dim, segment_len, dim_head, heads, ff_mult,
                 num_persist_mem_tokens, num_longterm_mem_tokens,
                 use_neural_memory,
                 mem_depth, mem_expansion_factor, mem_heads, mem_dim_head,
                 mem_chunk_size,
                 max_chunks_unroll = 16,attention_mode = "segmented"):
        super().__init__()
        self.name = name
        self.mem: NeuralMemory | None = None
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

        self.attn    = SegmentedAttention(
            name=f'{name}.attn', dim=dim, segment_len=segment_len,
            num_persist_mem_tokens=num_persist_mem_tokens,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            dim_head=dim_head, heads=heads,attention_mode=attention_mode)
        self.add_attn= F.Add(f'{name}.add_attn')
        self.ff      = FeedForward(f'{name}.ff', dim, mult=ff_mult)
        self.add_ff  = F.Add(f'{name}.add_ff')
        super().link_op2module()

    def __call__(self, x, mem_state=None):
        if self.mem is not None:
            retrieved, mem_state = self.mem(x, state=mem_state)
            x = self.add_mem(x, retrieved)
        x = self.add_attn(x, self.attn(x))
        x = self.add_ff  (x, self.ff(x))
        return x, mem_state

    def analytical_param_count(self, lvl=0):
        pc = 0
        if self.mem is not None:
            pc += self.mem.analytical_param_count(lvl + 1)
        pc += self.attn.analytical_param_count(lvl + 1)
        pc += self.ff.analytical_param_count(lvl + 1)
        return pc

class MemoryAsContextTransformer(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg  = cfg
        self.vocab_sz = cfg['vocab_sz']
        self.dim      = cfg['dE']
        self.nL       = cfg['nL']
        self.nH       = cfg['nH']
        self.dim_head = cfg.get('dim_head', self.dim // self.nH)
        self.nW       = cfg['nW']                       # seq_len
        self.bs       = cfg['bs']
        self.ff_mult  = cfg.get('ff_mult', 4)
        self.segment_len            = cfg['segment_len']
        self.num_persist_mem_tokens = cfg.get('num_persist_mem_tokens', 0)
        self.num_longterm_mem_tokens= cfg.get('num_longterm_mem_tokens', 0)
        self.neural_memory_layers   = set(cfg.get('neural_memory_layers',
                                                   list(range(1, self.nL + 1))))
        self.mem_depth        = cfg.get('mem_depth', 2)
        self.mem_expansion    = cfg.get('mem_expansion_factor', 2.0)
        self.mem_heads        = cfg.get('mem_heads', 4)
        self.mem_dim_head     = cfg.get('mem_dim_head', 64)
        # mem_chunk_size is upstream's neural_memory_segment_len (the NeuralMemory
        # chunk size); the default matches upstream's segment_len +
        # num_longterm_mem_tokens. Bounded here by max_chunks_unroll: the
        # recurrence is unrolled per chunk, so nW / mem_chunk_size must be
        # <= max_chunks_unroll
        self.mem_chunk_size   = cfg.get('mem_chunk_size',
                                        self.segment_len + self.num_longterm_mem_tokens)
        self.max_chunks_unroll = cfg.get('max_chunks_unroll', 16)
        self.attention_mode = cfg.get('attention_mode', 'segmented')

        self.wte = F.Embedding('wte', self.vocab_sz, self.dim)
        self.wpe = F.Embedding('wpe', self.nW, self.dim)
        self.add_emb = F.Add('add_emb')

        # longterm memory tokens (learnable)
        self.longterm_mems: SimTensor | None = None
        if self.num_longterm_mem_tokens > 0:
            self.longterm_mems = F._from_shape(
                'longterm_mems',
                shape=[1, self.num_longterm_mem_tokens, self.dim], is_param=True)

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
                mem_depth=self.mem_depth,
                mem_expansion_factor=self.mem_expansion,
                mem_heads=self.mem_heads,
                mem_dim_head=self.mem_dim_head,
                mem_chunk_size=self.mem_chunk_size,
                max_chunks_unroll=self.max_chunks_unroll,
                attention_mode=self.attention_mode,
            ),
            MACBlock(
                name='t_nomem',
                dim=self.dim, segment_len=self.segment_len,
                dim_head=self.dim_head, heads=self.nH, ff_mult=self.ff_mult,
                num_persist_mem_tokens=self.num_persist_mem_tokens,
                num_longterm_mem_tokens=self.num_longterm_mem_tokens,
                use_neural_memory=False,
                mem_depth=self.mem_depth,
                mem_expansion_factor=self.mem_expansion,
                mem_heads=self.mem_heads,
                mem_dim_head=self.mem_dim_head,
                mem_chunk_size=self.mem_chunk_size,
                max_chunks_unroll=self.max_chunks_unroll,
                attention_mode=self.attention_mode,
            ),
        ])
        self._block_repeat_counts = [self._n_mem_layers, self._n_nomem_layers]

        self.final_norm = _RMSNorm('final_norm', self.dim)
        self.to_logits  = SimNN.Linear('to_logits', self.dim, self.vocab_sz, bias=False)

        self.input_tensors = {}
        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
            'tokens': F._from_shape('tokens', [self.bs, self.nW], is_param=False, np_dtype=np.int64),
            'pos'  : F._from_shape('pos',   [self.bs, self.nW], is_param=False, np_dtype=np.int64),
        }

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def __call__(self):
        assert len(self.input_tensors) == 2, "call create_input_tensors() first"
        tok = self.input_tensors['tokens']
        pos = self.input_tensors['pos'] # pos = position indices [0 to N-1]:positional-embedding lookup (wpe)
        x = self.add_emb(self.wte(tok), self.wpe(pos)) # token_emb + pos_emb

        if self.longterm_mems is not None:
            x = T.cat([self.longterm_mems, x], dim=1)

        # Each proxy block stands in for rc physical layers. rc == 0 means the
        # config has no layers of that type, and such a block must not be built
        # into the graph at all - see PROXY-BLOCK COST ACCOUNTING below for why
        # emitting its ops would make cycles and params disagree.
        # Skipping is safe for the residual stream: a block maps [B, N, dim] to
        # [B, N, dim], so x keeps the same shape whether or not it runs.
        for blk, rc in zip(self.blocks, self._block_repeat_counts):
            if rc == 0:
                continue
            x, _ = blk(x, mem_state=None)

        # PROXY-BLOCK COST ACCOUNTING
        # The model is built as two proxy blocks (t_mem: memory-bearing, t_nomem:
        # plain) rather than nL physical layers. Here we set each op's repeat_count
        # to the number of layers of that block type (_block_repeat_counts), so
        # Polaris costs a stack of "rc" identical layers as per_layer_cost * rc,
        # WITHOUT materializing rc copies in the graph.
        #
        # This multiplies projected CYCLES to reflect nL layers; it does NOT create
        # nL physical layers. It is exact for cost projection because repeated
        # layers are cost-identical. Parameter counting (analytical_param_count)
        # applies the SAME _block_repeat_counts (per-block param count * rc), so
        # cycles and params are scaled consistently.
        # The rc == 0 case is load-bearing for that consistency. repeat_count
        # defaults to 1 (SimOp.__init__, ttsim/ops/op.py), so an op left
        # unassigned is billed as one full layer. Params would report the block
        # as absent while cycles reported it running once. The invariant is
        # therefore maintained upstream: a block with rc == 0 is never called,
        # so it contributes no ops here and nothing is left at the default.
        # Every op reached by this loop belongs to a block with rc >= 1.

        for blk, rc in zip(self.blocks, self._block_repeat_counts):
            if rc == 0:
                continue # not in the graph: nothing to scale
            repeated_ops: dict[str, Any] = {}
            blk.get_ops(repeated_ops)
            for _, op_obj in repeated_ops.items():
                op_obj.repeat_count = rc

        x = self.final_norm(x)
        return self.to_logits(x)

    def analytical_param_count(self, lvl=0):
        # Params scaled by _block_repeat_counts (same counts used for cycle
        # accounting in __call__), so a proxy block counts for all rc layers.
        pc = 0
        pc += self.vocab_sz * self.dim
        pc += self.nW * self.dim
        if self.longterm_mems is not None:
            pc += self.num_longterm_mem_tokens * self.dim
        pc += sum(
            # ModuleList.__iter__ is typed -> Iterator[Module], and base Module does
            # not declare analytical_param_count (every subclass implements it as an
            # informal protocol), so mypy erases the concrete MACBlock type here.
            # Hence the ignore; the same calls on self.attn / self.mem need none,
            # because those attributes keep their concrete types.
            blk.analytical_param_count(lvl + 1) * rc  # type: ignore[attr-defined]
            for blk, rc in zip(self.blocks, self._block_repeat_counts)
        )
        pc += self.dim
        pc += self.dim * self.vocab_sz
        return pc


class TitansMAC(MemoryAsContextTransformer):
    """Polaris workload class.
       __init__(name, cfg) / set_batch_size / create_input_tensors /
       get_forward_graph / __call__ / analytical_param_count.
    """
    pass
