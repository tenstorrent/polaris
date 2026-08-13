#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim port of titans_pytorch/neural_memory.NeuralMemory
  - Memory IS the weights of a small MLP (MemoryMLP).
  - store_memories chunk-by-chunk:
       (a) project seq -> keys, values, adaptive_lr, decay_factor
       (b) per-chunk: forward MemoryMLP on keys, compute (pred - v)^2 loss,
           derive grads dL/dW for each W in MemoryMLP via explicit chain-rule
           matmuls (cheap to emit, correct shapes/FLOPs for perf projection).
       (c) momentum accumulation:  M_t = a * M_{t-1} + grad_t       (one step per chunk)
       (d) decay-based weight update: W_t = (1-decay) * W_{t-1} + M_t
  - retrieve_memories:
       q = Linear(norm(seq)); out = MemoryMLP(q | W_current); merge_heads.

Forward-only. State (NeuralMemState) is a 5-tuple of SimTensors per the
namedtuple in the reference.  Backprop engineer wires gradient producers
to the op handles in store_memories / retrieve_memories.

"""
import os, sys, math
from collections import namedtuple
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN

from workloads.Titans.memory_models import MemoryMLP, ResidualNorm

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

class _MultiheadRMSNorm(SimNN.Module):
    """RMSNorm(dim) * (gamma_per_head + 1) — same as titans MultiheadRMSNorm."""
    def __init__(self, name, dim, heads):
        super().__init__()
        self.name = name
        self.dim  = dim
        self.heads = heads
        self.mulx2  = F.Mul(f'{name}.mulx2')
        self.meanop = F.Mean(f'{name}.mean', dim=-1)
        self.sqrt   = F.Sqrt(f'{name}.sqrt')
        self.recip  = F.Reciprocal(f'{name}.recip')
        self.mulnorm= F.Mul(f'{name}.mulnorm')
        self.mulgamma=F.Mul(f'{name}.mulgamma')
        self.addone = F.Add(f'{name}.addone')
        self.gamma  = F._from_shape(f'{name}.gamma', shape=[heads, 1, dim], is_param=True)
        self.one    = F._from_data(f'{name}.one', np.float32(1.0))
        self.one.set_module(self); self._tensors[self.one.name] = self.one
        super().link_op2module()

    def __call__(self, x):
        eps = F._from_shape(f'{self.name}.eps', shape=[1])
        eps.set_module(self); self._tensors[eps.name] = eps
        x2  = self.mulx2(x, x)
        mu  = self.meanop(x2).unsqueeze(-1)
        rms = self.recip(self.sqrt(mu + eps))
        normed = self.mulnorm(x, rms)
        return self.mulgamma(normed, self.addone(self.gamma, self.one))

    def analytical_param_count(self, lvl=0):
        return self.heads * self.dim

class _RMSNorm(SimNN.Module):
    """Standard RMSNorm (no learnable gamma if elementwise_affine=False)."""
    def __init__(self, name, dim, elementwise_affine=True):
        super().__init__()
        self.name = name
        self.dim  = dim
        self.eaff = elementwise_affine
        self.mulx2  = F.Mul(f'{name}.mulx2')
        self.meanop = F.Mean(f'{name}.mean', dim=-1)
        self.sqrt   = F.Sqrt(f'{name}.sqrt')
        self.recip  = F.Reciprocal(f'{name}.recip')
        self.mulnorm= F.Mul(f'{name}.mulnorm')
        if elementwise_affine:
            self.weight = F._from_shape(f'{name}.weight', shape=[dim], is_param=True)
            self.mulw   = F.Mul(f'{name}.mulw')
        super().link_op2module()

    def __call__(self, x):
        eps = F._from_shape(f'{self.name}.eps', shape=[1])
        eps.set_module(self); self._tensors[eps.name] = eps
        x2  = self.mulx2(x, x)
        mu  = self.meanop(x2).unsqueeze(-1)
        normed = self.mulnorm(x, self.recip(self.sqrt(mu + eps)))
        if self.eaff:
            return self.mulw(normed, self.weight)
        return normed

    def analytical_param_count(self, lvl=0):
        return self.dim if self.eaff else 0

class NeuralMemory(SimNN.Module):
    """
    TTSim port of titans_pytorch.neural_memory.NeuralMemory.

    Constructor mirrors the reference for the supported subset.
    """
    def __init__(
        self,
        name,
        dim,
        chunk_size = 1,
        dim_head   = None,
        heads      = 1,
        depth      = 2,
        expansion_factor = 2.0,
        mem_model_norm_add_residual = True,
        pre_rmsnorm  = True,
        post_rmsnorm = False,
        qk_rmsnorm   = False,
        attn_pool_chunks = False,
        max_chunks_unroll = 16,
    ):
        super().__init__()
        self.name = name
        self.max_chunks_unroll = max_chunks_unroll
        dim_head = dim_head if dim_head is not None else dim
        assert not (heads == 1 and dim_head != dim)

        self.dim       = dim
        self.dim_head  = dim_head
        self.heads     = heads
        self.dim_inner = dim_head * heads
        self.chunk_size = chunk_size
        self.store_chunk_size    = chunk_size
        self.retrieve_chunk_size = chunk_size

        # ----- norms -----
        self.retrieve_norm = _RMSNorm(f'{name}.retrieve_norm', dim) if pre_rmsnorm else F.Identity(f'{name}.id_rn')
        self.store_norm    = _RMSNorm(f'{name}.store_norm',    dim) if pre_rmsnorm else F.Identity(f'{name}.id_sn')
        self.multihead_rmsnorm = _MultiheadRMSNorm(f'{name}.mh_rmsnorm', dim_head, heads) if post_rmsnorm else F.Identity(f'{name}.id_mh')
        self.q_norm = _MultiheadRMSNorm(f'{name}.q_norm', dim_head, heads) if qk_rmsnorm else F.Identity(f'{name}.id_q')
        self.k_norm = _MultiheadRMSNorm(f'{name}.k_norm', dim_head, heads) if qk_rmsnorm else F.Identity(f'{name}.id_k')

        # ----- projections -----
        self.to_queries = SimNN.Linear(f'{name}.to_queries', dim, self.dim_inner)
        self.to_keys    = SimNN.Linear(f'{name}.to_keys',    dim, self.dim_inner)
        self.to_values  = SimNN.Linear(f'{name}.to_values',  dim, self.dim_inner)

        # ----- per-chunk learned hparams -----
        self.to_adaptive_step = SimNN.Linear(f'{name}.to_adaptive_step', dim, heads)
        self.to_decay_factor  = SimNN.Linear(f'{name}.to_decay_factor',  dim, heads)
        self.to_momentum      = SimNN.Linear(f'{name}.to_momentum',      dim, heads)
        self.adaptive_step_sig= F.Sigmoid(f'{name}.adaptive_step_sig')
        self.decay_sig        = F.Sigmoid(f'{name}.decay_sig')
        self.momentum_sig     = F.Sigmoid(f'{name}.momentum_sig')

        # ----- multi-head retrieve gate -----
        if heads > 1:
            self.to_retrieve_gate = SimNN.Linear(f'{name}.to_retrieve_gate', dim, heads)
            self.retrieve_gate_sig= F.Sigmoid(f'{name}.retrieve_gate_sig')
            self.combine_heads    = SimNN.Linear(f'{name}.combine_heads', self.dim_inner, dim)
        else:
            self.to_retrieve_gate = None # type: ignore[assignment]
            self.combine_heads    = F.Identity(f'{name}.id_combine')

        # ----- the memory model (its weights are the memory) -----
        inner_model = MemoryMLP(f'{name}.mem_mlp', dim_head, depth, expansion_factor)
        if mem_model_norm_add_residual:
            self.memory_model = ResidualNorm(f'{name}.mem_resnorm', dim_head, inner_model)
        else:
            self.memory_model = inner_model # type: ignore[assignment]

        self._mem_param_names = [f'W{i}' for i in range(depth)]
        self._mem_param_dims  = inner_model.dims  # length depth+1
        self._mem_inner       = inner_model
        self._mem_depth       = depth

        self.store_mm_fwd     = F.SimOpHandleList([F.MatMul(f'{name}.store_fwd_W{i}')          for i in range(depth)])
        self.store_gelu_fwd   = F.SimOpHandleList([F.Gelu  (f'{name}.store_fwd_gelu{i}')       for i in range(depth - 1)])
        self.sub_err          = F.Sub(f'{name}.sub_err')
        self.mul_scale        = F.Mul(f'{name}.mul_lr_scale')
        self.store_mm_bwd_dW  = F.SimOpHandleList([F.MatMul(f'{name}.store_bwd_dW{i}')         for i in range(depth)])
        self.store_mm_bwd_dh  = F.SimOpHandleList([F.MatMul(f'{name}.store_bwd_dh{i}')         for i in range(depth - 1)])
        self.gelu_grad        = F.SimOpHandleList([F.Mul   (f'{name}.store_bwd_geluprime{i}')  for i in range(depth - 1)])

        self.mom_mul   = F.SimOpHandleList([
            F.Mul(f'{name}.mom_mul_W{i}_c{c}')
            for i in range(depth) for c in range(self.max_chunks_unroll)
        ])
        self.mom_add   = F.SimOpHandleList([
            F.Add(f'{name}.mom_add_W{i}_c{c}')
            for i in range(depth) for c in range(self.max_chunks_unroll)
        ])
        self.decay_mul = F.SimOpHandleList([
            F.Mul(f'{name}.decay_mul_W{i}_c{c}')
            for i in range(depth) for c in range(self.max_chunks_unroll)
        ])
        self.weight_add = F.SimOpHandleList([
            F.Add(f'{name}.weight_add_W{i}_c{c}')
            for i in range(depth) for c in range(self.max_chunks_unroll)
        ])
        self.weight_sub = F.SimOpHandleList([
            F.Sub(f'{name}.weight_sub_W{i}_c{c}')
            for i in range(depth) for c in range(self.max_chunks_unroll)
        ])

        self._grads_splits = []
        for i in range(depth):
            row = []
            for k in range(1, self.max_chunks_unroll + 1):
                op = F.SplitOpHandle(f'{name}.grads_split_W{i}_n{k}', count=k, axis=1)
                setattr(self, f'grads_split_W{i}_n{k}', op)
                row.append(op)
            self._grads_splits.append(row)

        self._momentum_coef_splits = []
        self._decay_factor_splits  = []
        for k in range(1, self.max_chunks_unroll + 1):
            mop = F.SplitOpHandle(f'{name}.momentum_coef_split_n{k}', count=k, axis=1)
            dop = F.SplitOpHandle(f'{name}.decay_factor_split_n{k}',  count=k, axis=1)
            setattr(self, f'momentum_coef_split_n{k}', mop)
            setattr(self, f'decay_factor_split_n{k}',  dop)
            self._momentum_coef_splits.append(mop)
            self._decay_factor_splits.append(dop)

        self.chunk_mean = F.Mean(f'{name}.chunk_mean', dim=-2)
        self.decay_chunk_mean    = F.Mean(f'{name}.decay_chunk_mean',    dim=1)
        self.momentum_chunk_mean = F.Mean(f'{name}.momentum_chunk_mean', dim=1)

        self.cat_cache = F.Concat(f'{name}.cat_cache', axis=1)

        super().link_op2module()


    def _mc(self, i, c):
        """D1: flat-index helper for per-chunk recurrence ops.
        Maps (layer_idx i, chunk_idx c) -> position in the flat SimOpHandleList."""
        return i * self.max_chunks_unroll + c


    def init_weights(self, batch):
        """Returns dict[name -> SimTensor] mirroring lucidrains init_weights.
        Shape per param: [batch*heads, dim_in, dim_out]."""
        bh = batch * self.heads
        d  = self._mem_inner.dims
        out = {}
        for i in range(self._mem_depth):
            t = F._from_shape(
                f'{self.name}.init_W{i}',
                shape=[bh, d[i], d[i + 1]], is_param=True)
            t.set_module(self)
            self._tensors[t.name] = t
            out[f'W{i}'] = t
        return out

    def init_momentum(self, batch):
        bh = batch * self.heads
        d  = self._mem_inner.dims
        out = {}
        for i in range(self._mem_depth):
            t = F._from_shape(
                f'{self.name}.init_M{i}',
                shape=[bh, d[i], d[i + 1]], is_param=False)
            t.set_module(self)
            self._tensors[t.name] = t
            out[f'W{i}'] = t
        return out

    def retrieve_memories(self, seq, weights):
        """
        seq      : SimTensor [B, N, D]
        weights  : dict 'Wi' -> SimTensor [B*H, dim_in, dim_out]   (current memory)
        returns  : SimTensor [B, N, D]
        """
        B, N, D = seq.shape
        H, Dh   = self.heads, self.dim_head

        x = self.retrieve_norm(seq)
        q = self.to_queries(x)
        q = q.reshape(B, N, H, Dh).transpose(1, 2)
        q = self.q_norm(q)

        h = q.reshape(B * H, N, Dh)
        for i in range(self._mem_depth):
            if i > 0:
                h = self._mem_inner.gelus[i - 1](h)
            mm = self._mem_inner.matmuls[i][0]
            h  = mm(h, weights[f'W{i}'])
        out = h
        if isinstance(self.memory_model, ResidualNorm):
            out = self.memory_model.add(self.memory_model.norm(h), q.reshape(B * H, N, Dh))

        out = out.reshape(B, H, N, Dh)
        out = self.multihead_rmsnorm(out)

        if self.to_retrieve_gate is not None:
            gate = self.retrieve_gate_sig(self.to_retrieve_gate(x))
            gate = gate.reshape(B, N, H, 1).transpose(1, 2)
            out  = out * gate

        out = out.transpose(1, 2).reshape(B, N, H * Dh)
        out = self.combine_heads(out)
        return out

    def store_memories(self, seq, weights, past_state, seq_index):
        """
        Emits the per-chunk store-update graph.

        seq        : SimTensor [B, N, D]
        weights    : dict 'Wi' -> SimTensor [B*H, di, dj]
        past_state : tuple(last_update, last_momentum), both dicts as above
        seq_index  : python int (host-side bookkeeping only)

        returns: (updates_dict, NeuralMemState)
        """
        B, N, D = seq.shape
        H, Dh   = self.heads, self.dim_head
        C       = self.store_chunk_size

        n_full   = (N // C) * C
        num_ch   = n_full // C
        remainder = None

        x = self.store_norm(seq)

        chunked = self.chunk_mean(x.reshape(B, num_ch, C, D))
        adaptive_lr  = self.adaptive_step_sig(self.to_adaptive_step(x))
        decay_factor = self.decay_sig       (self.to_decay_factor (chunked))
        momentum_coef= self.momentum_sig    (self.to_momentum     (chunked))

        K = self.to_keys  (x).reshape(B, N, H, Dh).transpose(1, 2)
        V = self.to_values(x).reshape(B, N, H, Dh).transpose(1, 2)
        K = self.k_norm(K)

        acts = [K.reshape(B * H, N, Dh)]
        for i in range(self._mem_depth):
            h = acts[-1]
            if i > 0:
                h = self.store_gelu_fwd[i - 1](h)
            h = self.store_mm_fwd[i](h, weights[f'W{i}'])
            acts.append(h)
        pred = acts[-1]

        err = self.sub_err(pred, V.reshape(B * H, N, Dh))
        d_pred = self.mul_scale(err, adaptive_lr.reshape(B * H, N, 1))

        grads = {}
        d_h = d_pred
        for i in reversed(range(self._mem_depth)):
            a_in = acts[i]

            di_i = a_in.shape[-1]
            dj_i = d_h.shape[-1]
            a_in_chunked = a_in.reshape(B * H, num_ch, C, di_i)
            d_h_chunked  = d_h.reshape(B * H, num_ch, C, dj_i)

            grads[f'W{i}'] = self.store_mm_bwd_dW[i](
                a_in_chunked.transpose(-1, -2), d_h_chunked)

            if i > 0:
                d_h = self.store_mm_bwd_dh[i - 1](d_h, weights[f'W{i}'].transpose(-1, -2))
                d_h = self.gelu_grad[i - 1](d_h, a_in)

        assert num_ch <= self.max_chunks_unroll, \
            f"num_chunks={num_ch} exceeds max_chunks_unroll={self.max_chunks_unroll}. " \
            f"Either raise max_chunks_unroll or bump mem_chunk_size in the YAML."

        prev_update, prev_momentum = past_state
        new_update, new_momentum = {}, {}

        decay_factor_r  = decay_factor.reshape(B * H, num_ch, 1, 1)
        momentum_coef_r = momentum_coef.reshape(B * H, num_ch, 1, 1)

        decay_chunks    = self._decay_factor_splits[num_ch - 1](decay_factor_r)
        momentum_chunks = self._momentum_coef_splits[num_ch - 1](momentum_coef_r)


        for i in range(self._mem_depth):
            one_const = F._from_data(f'{self.name}.one_W{i}', np.float32(1.0))
            one_const.set_module(self); self._tensors[one_const.name] = one_const

            g_chunks = self._grads_splits[i][num_ch - 1](grads[f'W{i}'])

            m_prev = prev_momentum[f'W{i}']
            w_prev = prev_update[f'W{i}']

            for c in range(num_ch):
                g_c     = g_chunks[c].squeeze(1)
                eta_c   = momentum_chunks[c].squeeze(1)
                alpha_c = decay_chunks[c].squeeze(1)

                m_scaled = self.mom_mul[self._mc(i, c)](m_prev, eta_c)
                m_new    = self.mom_add[self._mc(i, c)](m_scaled, g_c)

                one_minus_alpha = self.weight_sub[self._mc(i, c)](one_const, alpha_c)
                w_decayed       = self.decay_mul[self._mc(i, c)](w_prev, one_minus_alpha)
                w_new           = self.weight_add[self._mc(i, c)](w_decayed, m_new)

                m_prev = m_new
                w_prev = w_new

            new_momentum[f'W{i}'] = m_prev
            new_update[f'W{i}']   = w_prev

        next_state = (new_update, new_momentum)
        next_mem_state = NeuralMemState(
            seq_index = seq_index + n_full,
            weights   = new_update,
            cache_store_segment = remainder,
            states    = next_state,
            updates   = new_update,
        )
        return new_update, next_mem_state

    def __call__(self, seq, state=None, store_seq=None):
        B = seq.shape[0]
        store_seq = store_seq if store_seq is not None else seq

        if state is None:
            weights    = self.init_weights(B)
            past_state = (weights, self.init_momentum(B))
            seq_index  = 0
            cache_seg  = None
        else:
            seq_index  = state.seq_index
            weights    = state.weights
            past_state = state.states
            cache_seg  = state.cache_store_segment

        if cache_seg is not None:
            store_seq = self.cat_cache(cache_seg, store_seq)

        store_seq_len = store_seq.shape[-2]

        updates, next_mem_state = self.store_memories(
            store_seq, weights, past_state, seq_index)
        weights    = next_mem_state.weights
        past_state = next_mem_state.states

        new_cache_seg = None

        next_state_final = NeuralMemState(
            seq_index           = seq_index + store_seq_len,
            weights             = weights,
            cache_store_segment = new_cache_seg,
            states              = past_state,
            updates             = updates,
        )

        retrieved = self.retrieve_memories(seq, weights)
        return retrieved, next_state_final

    def analytical_param_count(self, lvl=0):
        pc = 0
        pc += self.to_queries.analytical_param_count(lvl + 1)
        pc += self.to_keys.analytical_param_count(lvl + 1)
        pc += self.to_values.analytical_param_count(lvl + 1)
        pc += self.to_adaptive_step.analytical_param_count(lvl + 1)
        pc += self.to_decay_factor.analytical_param_count(lvl + 1)
        pc += self.to_momentum.analytical_param_count(lvl + 1)
        if self.to_retrieve_gate is not None:
            pc += self.to_retrieve_gate.analytical_param_count(lvl + 1)
            pc += self.combine_heads.analytical_param_count(lvl + 1)
        pc += self.memory_model.analytical_param_count(lvl + 1)
        # norms
        for n in ['retrieve_norm','store_norm','multihead_rmsnorm','q_norm','k_norm']:
            mod = getattr(self, n)
            if hasattr(mod, 'analytical_param_count'):
                pc += mod.analytical_param_count(lvl + 1)
        return pc
