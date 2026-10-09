#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4-Flash -- TTSIM workload (shared-KV MQA + hybrid CSA/HCA sparse
attention, hash/sqrtsoftplus MoE, manifold hyper-connections).

Architecture pinned from HF ``deepseek-ai/DeepSeek-V4-Flash`` ``config.json``
(``model_type: deepseek_v4``, ``DeepseekV4ForCausalLM``, 284B total / ~13B active):

  - vocab_size             : 129280 (untied embedding / LM head)
  - hidden_size            : 4096, num_hidden_layers: 43
  - attention (all layers) : shared-KV MQA -- 64 Q-heads x 512 with ONE shared
                             KV vector per token (k == v, dim 512), q_lora_rank
                             1024 (wq_a -> RMSNorm -> wq_b), grouped low-rank
                             o_proj (o_groups 8, o_lora_rank 1024), partial RoPE
                             (64/512, theta 1e4), sliding_window 128, per-head
                             attention sink
  - compress_ratios        : [0, 0, (4, 128) x 20, 0] -> 3 sliding-window-only
                             layers, 20 CSA layers (4x compressed KV +
                             Lightning Indexer: 64 heads x 128, top-512
                             compressed blocks), 20 HCA layers (128x compressed
                             KV, all blocks attended)
  - compressor             : learned gated pooling (wkv/wgate + APE + softmax
                             over the pool window; overlapping windows at 4x)
  - MoE (every layer)      : 256 routed + 1 shared, top-6, intermediate 2048,
                             sqrtsoftplus scoring (sqrt(softplus(x))) with
                             noaux_tc bias on routed layers, hash routing
                             (tid2eid lookup) on the first 3 layers,
                             norm_topk_prob, routed_scaling_factor 1.5,
                             swiglu_limit 10, fp4 experts in the checkpoint
  - mHC                    : manifold-constrained hyper-connections -- 4
                             residual streams, per-sublayer sinkhorn-mixed
                             hc_fn (hc_mult 4, sinkhorn iters 20)
  - 1 MTP layer            : not modeled (see README); checkpoint dtypes (fp8 /
                             fp4) come from the workload-to-arch mapping

Performance-simulation abstractions (documented; see README.md):

  1. Layer multiplicity: one proxy block per unique (gate-kind, compress-ratio)
     layer type is executed; op ``repeat_count`` scales it (BasicLLM pattern).
  2. MoE models ACTIVE compute: the top-6 expert slots + shared expert are
     applied to the routed activations; dispatch/combine not modeled.
  3. Hash routing (first 3 layers): expert *selection* is a tid2eid table
     lookup (no topk); the static graph approximates the gathered routing
     weights with a length-6 slice of the sqrtsoftplus scores (cost-equivalent;
     the router matmul itself IS real -- hash gates still score experts).
  4. Sparse attention windows: the attended width S per layer type (window-only
     / window + top-512 CSA blocks / window + all HCA blocks) is computed from
     seq_len at graph-build time; the dynamically gathered KV is approximated
     by the first S positions (the real sparse_attn also gathers selected
     positions; matmul cost is identical). Decode KV-cache reads are not
     modeled -- attention runs over the current window only (same limitation
     as the other LLM workloads).
  5. mHC sinkhorn normalization (20 iterations over tiny mixing tensors) is
     approximated by softmax normalization of the hc_fn mixes; the 4-stream
     hc_pre / hc_post residual mixing is modeled op-for-op.
  6. swiglu_limit clamps inside experts (2 eltwise ops/expert) are omitted.
  7. The Lightning-Indexer Hadamard rotation and fp4/fp8 quantization ops are
     data-format transforms, not perf-relevant matmul/eltwise work; omitted.
  8. MTP draft layer is not part of the main inference loop and is not
     simulated.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

# shared building blocks identical to the Qwen3.5 workload
from workloads.qwen3_5.qwen35_model import MoEExpert, RMSNorm


# ----------------------------------------------------------------------------
class Compressor(SimNN.Module):
    """Gated-pooling KV compressor (CSA ratio 4 with overlap, HCA ratio 128).

    kv_c  = wkv(x); score = wgate(x); per compressed block of `ratio` tokens:
    softmax(score + APE) over the pool window, weighted sum, RMSNorm.
    overlap (ratio 4) doubles the intermediate width (coff=2); the reference
    pools jointly over the 2*ratio-token overlap window -- modeled here as two
    independent ratio-pools summed (same softmax/MAC count; abstraction).
    """

    def __init__(self, name, cfg, ratio, head_dim):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 4096)
        self.ratio = ratio
        self.head_dim = head_dim
        self.overlap = ratio == 4
        self.coff = 2 if self.overlap else 1
        self.eps = cfg.get('rms_norm_eps', 1e-6)

        self.wkv = SimNN.Linear(f'{name}.wkv', self.hidden, self.coff * head_dim, bias=False)
        self.wgate = SimNN.Linear(f'{name}.wgate', self.hidden, self.coff * head_dim, bias=False)
        self.ape = F._from_shape(f'{name}.ape', shape=[self.ratio, self.coff * head_dim],
                                 is_param=True)
        self.norm = RMSNorm(f'{name}.norm', head_dim, eps=self.eps)
        super().link_op2module()

    def __call__(self, x):
        B, L, _ = x.shape
        kv = self.wkv(x)  # (B, L, coff*D)
        sc = self.wgate(x)  # (B, L, coff*D)
        C = L // self.ratio
        if C == 0:
            # decode / short window: the per-token projections still run
            # (incremental state-update path), but no complete block is emitted
            return None

        # fresh one-shot handles (pooling runs once per block)
        ape_add_op = F.Add(f'{self.name}.ape_add')
        sm_op = F.Softmax(f'{self.name}.pool_softmax')
        pool_mul_op = F.Mul(f'{self.name}.pool_mul')
        pool_sum_op = F.ReduceSum(f'{self.name}.pool_sum', axes=[2], keepdims=False)
        ops = [ape_add_op, sm_op, pool_mul_op, pool_sum_op]
        if self.overlap:
            overlap_add_op = F.Add(f'{self.name}.overlap_add')
            ops.append(overlap_add_op)
        for op in ops:
            op.set_module(self)
            self._op_hndls[op.name] = op

        keep = C * self.ratio  # drop the remainder tokens (held in state in the real model)
        if keep < L:
            kv = kv[:, 0:keep, :]
            sc = sc[:, 0:keep, :]

        kvr = kv.reshape(B, C, self.ratio, self.coff * self.head_dim)  # (B, C, r, coff*D)
        scr = sc.reshape(B, C, self.ratio, self.coff * self.head_dim)
        scr = ape_add_op(scr, self.ape)  # (B, C, r, coff*D) + (r, coff*D) APE
        # softmax over the pool window (dim 2): rotate it to the last axis
        w = sm_op(scr.permute([0, 1, 3, 2])).permute([0, 1, 3, 2])  # (B, C, r, coff*D)
        pooled = pool_sum_op(pool_mul_op(kvr, w))  # (B, C, coff*D)
        if self.overlap:
            half = self.head_dim
            pooled = ops[-1](pooled[:, :, 0:half], pooled[:, :, half:])  # (B, C, D)
        return self.norm(pooled)  # (B, C, D)

    def analytical_param_count(self, lvl=0):
        p = 2 * self.hidden * (self.coff * self.head_dim)  # wkv + wgate
        p += self.ratio * (self.coff * self.head_dim)  # APE
        p += self.head_dim  # norm
        return p


# ----------------------------------------------------------------------------
class Indexer(SimNN.Module):
    """Lightning Indexer (CSA layers): scores compressed blocks, selects top-k.

    q_idx = wq_b(q_lora) (64 heads x 128); its own 128-dim Compressor builds
    the scoring KV; scores = relu(q_idx . kv) per head, weighted by
    weights_proj(x), summed over heads, top-(min(topk, C)) selection.
    Returns the number of selected compressed blocks.
    """

    def __init__(self, name, cfg, ratio):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 4096)
        self.q_lora = cfg.get('q_lora_rank', 1024)
        self.idx_heads = cfg.get('index_n_heads', 64)
        self.idx_dim = cfg.get('index_head_dim', 128)
        self.topk = cfg.get('index_topk', 512)

        self.wq_b = SimNN.Linear(f'{name}.wq_b', self.q_lora, self.idx_heads * self.idx_dim,
                                 bias=False)
        self.weights_proj = SimNN.Linear(f'{name}.weights_proj', self.hidden, self.idx_heads,
                                         bias=False)
        self.compressor = Compressor(f'{name}.compressor', cfg, ratio, self.idx_dim)
        self.scale_t = F._from_data(f'{name}.scale',
                                    data=np.float32(self.idx_dim ** -0.5 * self.idx_heads ** -0.5),
                                    is_const=True)
        self.scale_t.set_module(self)
        super().link_op2module()

    def __call__(self, x, q_lora):
        B, L, _ = x.shape
        qi = self.wq_b(q_lora).reshape(B, L, self.idx_heads, self.idx_dim)
        qi = qi.permute([0, 2, 1, 3])  # (B, IH, L, ID)
        w = self.weights_proj(x) * self.scale_t  # (B, L, IH)

        kvc = self.compressor(x)  # (B, C, ID) or None
        C = L // self.compressor.ratio
        if kvc is None or C == 0:
            return 0  # projections ran; no complete block to score

        kvc_h = kvc.reshape(B, 1, C, self.idx_dim).repeat(1, self.idx_heads, 1, 1)  # (B, IH, C, ID)
        score = T.matmul(qi, kvc_h.transpose(2, 3))  # (B, IH, L, C)

        relu_op = F.Relu(f'{self.name}.relu')
        wmul_op = F.Mul(f'{self.name}.head_weight')
        hsum_op = F.ReduceSum(f'{self.name}.head_sum', axes=[2], keepdims=False)
        for op in (relu_op, wmul_op, hsum_op):
            op.set_module(self)
            self._op_hndls[op.name] = op
        score = relu_op(score).permute([0, 2, 1, 3])  # (B, L, IH, C)
        score = wmul_op(score, w.reshape(B, L, self.idx_heads, 1))  # per-head weights
        score = hsum_op(score)  # (B, L, C)

        sel = min(self.topk, C)
        if sel < C:
            # all blocks selected when topk >= C (topk over dim C with k == C is a no-op)
            topk_op = F.topk(f'{self.name}.topk', k=sel)
            topk_op.set_module(self)
            self._op_hndls[topk_op.name] = topk_op
            _vals, _idx = topk_op(score)  # selection (indices drive the KV gather on HW)
        return sel

    def analytical_param_count(self, lvl=0):
        p = self.q_lora * self.idx_heads * self.idx_dim  # wq_b
        p += self.hidden * self.idx_heads  # weights_proj
        p += self.compressor.analytical_param_count(lvl + 1)
        return p


# ----------------------------------------------------------------------------
class Dsv4Attention(SimNN.Module):
    """Shared-KV MQA with sliding window + optional compressed KV (CSA/HCA).

    Q path : x -> wq_a (lora 1024) -> RMSNorm -> wq_b (64 x 512)
    KV path: x -> wkv (512; ONE shared head, k == v) -> RMSNorm
    Attn   : q . kv^T over S positions -- S = min(L, 128) window plus, per layer
             type, nothing (ratio 0: pure sliding window), top-512 of the 4x
             compressed blocks (CSA) or all 128x compressed blocks (HCA);
             per-head attention sink + causal mask + softmax; ctx = probs . kv
    O path : grouped low-rank: 8 x Linear(4096 -> 1024) then Linear(8192 -> 4096)
    """

    def __init__(self, name, cfg, ratio):
        super().__init__()
        assert ratio in (0, 4, 128), f'unsupported compress ratio: {ratio}'
        self.name = name
        self.ratio = ratio
        self.hidden = cfg.get('hidden_size', 4096)
        self.n_heads = cfg.get('num_attention_heads', 64)
        self.head_dim = cfg.get('head_dim', 512)
        self.q_lora = cfg.get('q_lora_rank', 1024)
        self.o_lora = cfg.get('o_lora_rank', 1024)
        self.n_groups = cfg.get('o_groups', 8)
        self.window = cfg.get('sliding_window', 128)
        self.eps = cfg.get('rms_norm_eps', 1e-6)
        self.seq_len = cfg.get('seq_len', 1)
        self.topk = cfg.get('index_topk', 512)

        self.wq_a = SimNN.Linear(f'{name}.wq_a', self.hidden, self.q_lora, bias=False)
        self.q_norm = RMSNorm(f'{name}.q_norm', self.q_lora, eps=self.eps)
        self.wq_b = SimNN.Linear(f'{name}.wq_b', self.q_lora, self.n_heads * self.head_dim,
                                 bias=False)
        self.wkv = SimNN.Linear(f'{name}.wkv', self.hidden, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(f'{name}.kv_norm', self.head_dim, eps=self.eps)
        gwidth = (self.n_heads * self.head_dim) // self.n_groups
        self.wo_a = SimNN.ModuleList(
            [SimNN.Linear(f'{name}.wo_a.{g}', gwidth, self.o_lora, bias=False)
             for g in range(self.n_groups)]
        )
        self.wo_b = SimNN.Linear(f'{name}.wo_b', self.n_groups * self.o_lora, self.hidden,
                                  bias=False)

        # attention ops / consts
        self.attn_sink = F._from_shape(f'{name}.attn_sink', shape=[1, self.n_heads, 1, 1],
                                       is_param=True)
        self.scale_t = F._from_data(f'{name}.scale',
                                    data=np.float32(1.0 / math.sqrt(self.head_dim)), is_const=True)
        self.scale_mul_op = F.Mul(f'{name}.scale_mul')
        self.sink_add_op = F.Add(f'{name}.sink_add')
        self.mask_add_op = F.Add(f'{name}.mask_add')
        self.softmax_op = F.Softmax(f'{name}.softmax')
        # RoPE approximation constants (partial 64/512 rope approximated over the full head_dim)
        self.rope_cos = F._from_shape(f'{name}.rope_cos', shape=[1, 1, 1, self.head_dim],
                                      is_const=True)
        self.rope_sin = F._from_shape(f'{name}.rope_sin', shape=[1, 1, 1, self.head_dim],
                                      is_const=True)
        for t in (self.attn_sink, self.scale_t, self.rope_cos, self.rope_sin):
            t.set_module(self)

        # KV compressor (CSA/HCA) and Lightning Indexer (CSA only)
        self.compressor: Optional[Compressor] = (
            Compressor(f'{name}.compressor', cfg, ratio, self.head_dim) if ratio else None)
        self.indexer: Optional[Indexer] = Indexer(f'{name}.indexer', cfg, ratio) if ratio == 4 \
            else None

        # effective attended width (static; see abstraction #4) and causal mask
        self.S = self._eff_width(self.seq_len)
        if self.seq_len > 1:
            self.causal_mask = F._from_shape(f'{name}.causal_mask',
                                             shape=[1, 1, self.seq_len, self.S], is_const=True)
            self.causal_mask.set_module(self)
        super().link_op2module()

    def _eff_width(self, L):
        """Effective number of attended KV positions for a window of L tokens."""
        if self.ratio == 0:
            return min(L, self.window)  # pure sliding-window layer
        C = L // self.ratio
        sel = min(self.topk, C) if self.ratio == 4 else C  # CSA top-k / HCA all blocks
        return min(L, self.window) + sel

    def __call__(self, x):
        B, L, _ = x.shape
        assert _ == self.hidden, f'{self.name}: input hidden dim mismatch'
        assert L == self.seq_len, f'{self.name}: graph built for seq_len={self.seq_len}, got L={L}'

        # ---- Q path (low-rank)
        q_a = self.q_norm(self.wq_a(x))  # (B, L, 1024)
        q = self.wq_b(q_a).reshape(B, L, self.n_heads, self.head_dim)
        q = q.permute([0, 2, 1, 3])  # (B, H, L, D)
        q = q * self.rope_cos + q * self.rope_sin  # RoPE approx (elementwise cos/sin mix)

        # ---- KV path (shared-KV MQA: k == v)
        kv = self.kv_norm(self.wkv(x))  # (B, L, D)

        # ---- compressed KV + indexer selection
        if self.compressor is not None:
            kvc = self.compressor(x)  # (B, C, D) or None
            if self.indexer is not None:
                self.indexer(x, q_a)  # selection (drives the compressed gather on HW)
            if kvc is not None:
                kv = T.cat([kv, kvc], dim=1)  # (B, L+C, D)

        # ---- sparse window: static approximation of the dynamic KV gather
        S = self.S
        if S < kv.shape[1]:
            kv = kv[:, 0:S, :]
        kv_h = kv.reshape(B, 1, S, self.head_dim)  # (B, 1, S, D)
        kv_h = kv_h * self.rope_cos + kv_h * self.rope_sin  # RoPE approx (elementwise cos/sin mix)
        kv_h = kv_h.repeat(1, self.n_heads, 1, 1)  # (B, H, S, D) MQA head broadcast

        # ---- attention (k == v: shared KV)
        scores = self.scale_mul_op(T.matmul(q, kv_h.transpose(2, 3)), self.scale_t)  # (B, H, L, S)
        scores = self.sink_add_op(scores, self.attn_sink)  # per-head attention sink
        if L > 1:
            # SimTensor.__getitem__ is attached at runtime by tensor_op; type as Any for mypy
            mask: Any = self.causal_mask
            scores = self.mask_add_op(scores, mask)
        probs = self.softmax_op(scores)  # (B, H, L, S)
        ctx = T.matmul(probs, kv_h)  # (B, H, L, D) -- v == kv
        ctx = ctx.permute([0, 2, 1, 3])  # (B, L, H, D)

        # ---- grouped low-rank o_proj (einsum bsgd,grd->bsgr -> wo_b)
        gwidth = (self.n_heads * self.head_dim) // self.n_groups
        ctxg = ctx.reshape(B, L, self.n_groups, gwidth)  # (B, L, G, gw)
        outs = []
        for g in range(self.n_groups):
            xg = ctxg[:, :, g:g + 1, :].reshape(B, L, gwidth)
            outs.append(self.wo_a[g](xg))  # (B, L, o_lora)
        o = T.cat(outs, dim=2) if self.n_groups > 1 else outs[0]  # (B, L, G*o_lora)
        return self.wo_b(o)  # (B, L, hidden)

    def analytical_param_count(self, lvl=0):
        p = self.hidden * self.q_lora  # wq_a
        p += self.q_lora  # q_norm
        p += self.q_lora * self.n_heads * self.head_dim  # wq_b
        p += self.hidden * self.head_dim  # wkv
        p += self.head_dim  # kv_norm
        p += self.n_heads  # attn_sink
        p += self.n_groups * ((self.n_heads * self.head_dim // self.n_groups) * self.o_lora)
        p += self.n_groups * self.o_lora * self.hidden  # wo_b
        if self.compressor is not None:
            p += self.compressor.analytical_param_count(lvl + 1)
        if self.indexer is not None:
            p += self.indexer.analytical_param_count(lvl + 1)
        return p


# ----------------------------------------------------------------------------
class Dsv4MoE(SimNN.Module):
    """DeepSeek-V4 MoE: sqrtsoftplus router + top-6 (normalized, x1.5) + shared.

    scoring: s = sqrt(softplus(router(x))); routed layers add the noaux_tc
    bias for *selection* only (weights use the unbiased scores; approximated
    by the biased topk values, same as the GLM-5.3 workload). Hash layers
    (first 3) select experts via a tid2eid lookup instead of topk
    (abstraction #3).
    """

    def __init__(self, name, cfg, gate_kind):
        super().__init__()
        assert gate_kind in ('hash', 'routed'), f'bad gate kind: {gate_kind}'
        self.name = name
        self.gate_kind = gate_kind
        self.hidden = cfg.get('hidden_size', 4096)
        self.num_experts = cfg.get('n_routed_experts', 256)
        self.top_k = cfg.get('num_experts_per_tok', 6)
        self.moe_inter = cfg.get('moe_intermediate_size', 2048)
        self.n_shared = cfg.get('n_shared_experts', 1)
        self.vocab = cfg.get('vocab_size', 129280)
        self.rsf = cfg.get('routed_scaling_factor', 1.5)

        self.router = SimNN.Linear(f'{name}.gate', self.hidden, self.num_experts, bias=False)
        self.softplus_op = F.softplus(f'{name}.gate_softplus')
        self.sqrt_op = F.Sqrt(f'{name}.gate_sqrt')
        if gate_kind == 'routed':
            self.e_score_correction_bias = F._from_shape(
                f'{name}.e_score_correction_bias', shape=[self.num_experts], is_param=True)
            self.bias_add_op = F.Add(f'{name}.noaux_tc_bias')
            self.topk_op = F.topk(f'{name}.topk', k=self.top_k)
        self.wsum_op = F.Sum(f'{name}.topk_sum', dim=-1)
        self.wdiv_op = F.Div(f'{name}.topk_norm')
        self.rsf_t = F._from_data(f'{name}.routed_scaling_factor',
                                  data=np.float32(self.rsf), is_const=True)
        self.rsf_mul_op = F.Mul(f'{name}.routed_scale')

        # top_k active expert slots + 1 shared expert (active-compute modeling)
        self.experts = SimNN.ModuleList(
            [MoEExpert(f'{name}.experts.{i}', self.hidden, self.moe_inter)
             for i in range(self.top_k)]
        )
        self.shared_expert = MoEExpert(f'{name}.shared_experts.0', self.hidden, self.moe_inter)
        super().link_op2module()

    def __call__(self, x):
        s = self.sqrt_op(self.softplus_op(self.router(x)))  # sqrtsoftplus (B, L, E)
        if self.gate_kind == 'routed':
            s = self.bias_add_op(s, self.e_score_correction_bias)  # noaux_tc (selection only)
            top_vals, _top_idx = self.topk_op(s)  # (B, L, k)
        else:
            # hash routing: tid2eid lookup selects experts; approximate the
            # gathered routing weights with a length-k slice (abstraction #3)
            top_vals = s[:, :, 0:self.top_k]  # (B, L, k)
        wsum = self.wsum_op(top_vals).unsqueeze(-1)  # (B, L, 1)
        wscale = self.rsf_mul_op(self.wdiv_op(top_vals, wsum), self.rsf_t)  # norm x 1.5

        out = self.shared_expert(x) if self.n_shared > 0 else None  # (B, L, H)
        for i in range(self.top_k):
            wi = wscale[:, :, i:i + 1]  # (B, L, 1)
            ei = self.experts[i](x)  # (B, L, H)
            mulop = F.Mul(f'{self.name}.routeweight_mul{i}')
            if out is None:
                out = mulop(wi, ei)
                mulop.set_module(self)
                self._op_hndls[mulop.name] = mulop
                continue
            addop = F.Add(f'{self.name}.routecombine_add{i}')
            mulop.set_module(self)
            addop.set_module(self)
            self._op_hndls[mulop.name] = mulop
            self._op_hndls[addop.name] = addop
            out = addop(out, mulop(wi, ei))
        return out

    def analytical_param_count(self, lvl=0):
        p = self.hidden * self.num_experts  # router weight (hash gates score too)
        if self.gate_kind == 'routed':
            p += self.num_experts  # noaux_tc bias
        else:
            p += self.vocab * self.top_k  # tid2eid hash table
        p += self.num_experts * (3 * self.hidden * self.moe_inter)  # routed experts
        p += self.n_shared * (3 * self.hidden * self.moe_inter)  # shared expert
        return p


# ----------------------------------------------------------------------------
class HyperConnection(SimNN.Module):
    """Manifold-constrained hyper-connection (mHC): `hc_mult` residual streams.

    hc_pre : mixes = Linear(flatten(streams)) (hc_fn, (2+m)*m x m*H) scaled by
             rsqrt-RMS; the sinkhorn-normalized pre/post/comb mixing tensors
             are approximated by softmax slices of the mixes (abstraction #5);
             sublayer input y = sum_i pre_i * stream_i.
    hc_post: new stream i = post_i * sublayer_out + sum_j comb_ij * stream_j.
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.hc_mult = cfg.get('hc_mult', 4)
        self.hidden = cfg.get('hidden_size', 4096)
        self.eps = cfg.get('rms_norm_eps', 1e-6)
        m = self.hc_mult
        self.mix_hc = (2 + m) * m  # 24 for m=4
        self.hc_fn = SimNN.Linear(f'{name}.hc_fn', m * self.hidden, self.mix_hc, bias=False)

        self.sq_op = F.Mul(f'{name}.sq')
        self.mean_op = F.Mean(f'{name}.mean', dim=-1)
        self.add_eps_op = F.Add(f'{name}.add_eps')
        self.sqrt_op = F.Sqrt(f'{name}.sqrt')
        self.recip_op = F.Reciprocal(f'{name}.recip')
        self.mix_scale_op = F.Mul(f'{name}.mix_scale')
        self.pre_sm_op = F.Softmax(f'{name}.pre_softmax')
        self.post_sm_op = F.Softmax(f'{name}.post_softmax')
        self.comb_sm_op = F.Softmax(f'{name}.comb_softmax')
        self.eps_t = F._from_data(f'{name}.eps', data=np.float32(self.eps), is_const=True)
        super().link_op2module()

    def pre(self, streams: List[Any]):
        """Reduce hc_mult streams -> one sublayer input; return (y, post, comb)."""
        B, L, _ = streams[0].shape
        m = self.hc_mult
        flat = T.cat(streams, dim=2) if m > 1 else streams[0]  # (B, L, m*H)
        mu = self.mean_op(self.sq_op(flat, flat)).unsqueeze(-1)  # (B, L, 1)
        rstd = self.recip_op(self.sqrt_op(self.add_eps_op(mu, self.eps_t)))
        mixes = self.mix_scale_op(self.hc_fn(flat), rstd)  # (B, L, mix_hc)

        pre = self.pre_sm_op(mixes[:, :, 0:m])  # (B, L, m)
        post = self.post_sm_op(mixes[:, :, m:2 * m])  # (B, L, m)
        comb = self.comb_sm_op(mixes[:, :, 2 * m:].reshape(B, L, m, m))  # (B, L, m, m)

        y = None
        for i in range(m):
            pi = pre[:, :, i:i + 1]  # (B, L, 1)
            mulop = F.Mul(f'{self.name}.pre_mul{i}')
            mulop.set_module(self)
            self._op_hndls[mulop.name] = mulop
            term = mulop(pi, streams[i])
            if y is None:
                y = term
                continue
            addop = F.Add(f'{self.name}.pre_add{i}')
            addop.set_module(self)
            self._op_hndls[addop.name] = addop
            y = addop(y, term)
        return y, post, comb

    def post(self, sub_out, streams: List[Any], post, comb) -> List[Any]:
        """Expand sublayer output back into hc_mult new residual streams."""
        m = self.hc_mult
        B, L, _ = sub_out.shape
        new_streams = []
        for i in range(m):
            pi = post[:, :, i:i + 1]  # (B, L, 1)
            pmul = F.Mul(f'{self.name}.post_pmul{i}')
            pmul.set_module(self)
            self._op_hndls[pmul.name] = pmul
            acc = pmul(pi, sub_out)
            for j in range(m):
                cij = comb[:, :, i:i + 1, j:j + 1].reshape(B, L, 1)  # (B, L, 1)
                cmul = F.Mul(f'{self.name}.post_cmul{i}_{j}')
                cadd = F.Add(f'{self.name}.post_cadd{i}_{j}')
                cmul.set_module(self)
                cadd.set_module(self)
                self._op_hndls[cmul.name] = cmul
                self._op_hndls[cadd.name] = cadd
                acc = cadd(acc, cmul(cij, streams[j]))
            new_streams.append(acc)
        return new_streams

    def analytical_param_count(self, lvl=0):
        # hc_fn + hc_base + hc_scale
        return self.mix_hc * (self.hc_mult * self.hidden) + self.mix_hc + 3


# ----------------------------------------------------------------------------
class Dsv4FlashLayer(SimNN.Module):
    """One decoder layer with hyper-connection residuals (no plain residual add):

        streams -> hc_pre -> norm -> MQA(+compress/index) -> hc_post
                -> hc_pre -> norm -> MoE(hash|routed)      -> hc_post
    """

    def __init__(self, name, cfg, gate_kind, ratio):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 4096)
        self.eps = cfg.get('rms_norm_eps', 1e-6)
        self.attn_norm = RMSNorm(f'{name}.attn_norm', self.hidden, eps=self.eps)
        self.attn = Dsv4Attention(f'{name}.self_attn', cfg, ratio)
        self.ffn_norm = RMSNorm(f'{name}.ffn_norm', self.hidden, eps=self.eps)
        self.moe = Dsv4MoE(f'{name}.mlp', cfg, gate_kind)
        self.hc_attn = HyperConnection(f'{name}.hc_attn', cfg)
        self.hc_ffn = HyperConnection(f'{name}.hc_ffn', cfg)
        super().link_op2module()

    def __call__(self, streams: List[Any]) -> List[Any]:
        y, post, comb = self.hc_attn.pre(streams)
        a = self.attn(self.attn_norm(y))
        streams = self.hc_attn.post(a, streams, post, comb)
        y, post, comb = self.hc_ffn.pre(streams)
        m = self.moe(self.ffn_norm(y))
        return self.hc_ffn.post(m, streams, post, comb)

    def analytical_param_count(self, lvl=0):
        return (self.attn.analytical_param_count(lvl + 1)
                + self.moe.analytical_param_count(lvl + 1)
                + self.hc_attn.analytical_param_count(lvl + 1)
                + self.hc_ffn.analytical_param_count(lvl + 1)
                + 2 * self.hidden)


# ----------------------------------------------------------------------------
class Dsv4FlashMoe(SimNN.Module):
    """DeepSeek-V4-Flash top-level model (TTSIM entry class).

    Instantiated by polaris as ``Dsv4FlashMoe(instance_name, merged_cfg)``;
    all DeepSeek-V4-Flash defaults are built in, instances typically only
    override ``bs`` and ``seq_len``.
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.vocab_size = cfg.get('vocab_size', 129280)
        self.hidden = cfg.get('hidden_size', 4096)
        self.num_layers = cfg.get('num_hidden_layers', 43)
        self.n_hash = cfg.get('num_hash_layers', 3)
        self.hc_mult = cfg.get('hc_mult', 4)
        self.bs = cfg.get('bs', 1)
        self.seq_len = cfg.get('seq_len', 1)
        self.eps = cfg.get('rms_norm_eps', 1e-6)

        # per-layer compress ratios (default: the published 43-layer layout)
        ratios = cfg.get('compress_ratios')
        if ratios is None:
            assert self.num_layers >= 3 and (self.num_layers - 3) % 2 == 0, \
                'default compress_ratios needs num_hidden_layers = 3 + 2k'
            ratios = [0, 0] + [4, 128] * ((self.num_layers - 3) // 2) + [0]
        assert len(ratios) == self.num_layers, \
            f'compress_ratios length {len(ratios)} != num_hidden_layers {self.num_layers}'
        self.compress_ratios: List[int] = list(ratios)

        # proxy blocks: one per unique (gate_kind, compress_ratio) layer type,
        # in order of first appearance (repeat_count scales below)
        self.block_keys: List[Tuple[str, int]] = []
        self.block_counts: Dict[Tuple[str, int], int] = {}
        for i in range(self.num_layers):
            gk = 'hash' if i < self.n_hash else 'routed'
            key = (gk, self.compress_ratios[i])
            if key not in self.block_counts:
                self.block_keys.append(key)
                self.block_counts[key] = 0
            self.block_counts[key] += 1
        self.blocks = SimNN.ModuleList(
            [Dsv4FlashLayer(f'{name}.block_{gk}_r{ratio}', cfg, gk, ratio)
             for (gk, ratio) in self.block_keys])

        self.wte = F.Embedding(f'{name}.wte', self.vocab_size, self.hidden)
        self.final_norm = RMSNorm(f'{name}.final_norm', self.hidden, eps=self.eps)
        self.lm_head = SimNN.Linear(f'{name}.lm_head', self.hidden, self.vocab_size, bias=False)
        self.input_tensors = {}
        super().link_op2module()

    # -- layer-type census (used by tests; handy for README numbers) ---------
    @property
    def n_window_layers(self):
        return sum(1 for r in self.compress_ratios if r == 0)

    @property
    def n_csa_layers(self):
        return sum(1 for r in self.compress_ratios if r == 4)

    @property
    def n_hca_layers(self):
        return sum(1 for r in self.compress_ratios if r == 128)

    def create_input_tensors(self):
        self.input_tensors = {
            'tokens': F._from_shape('tokens', [self.bs, self.seq_len],
                                    is_param=False, np_dtype=np.int64),
        }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        assert len(self.input_tensors) == 1, \
            f'{self.name}: create_input_tensors() must be called before __call__()'

        Y = self.wte(self.input_tensors['tokens'])  # (B, L, hidden)
        streams: List[Any] = [Y] * self.hc_mult  # mHC residual streams
        for bi in range(len(self.block_keys)):
            streams = self.blocks[bi](streams)

        # scale the proxy blocks by layer multiplicity (BasicLLM repeat_count pattern)
        for bi in range(len(self.block_keys)):
            repeated_ops: dict = {}
            self.blocks[bi].get_ops(repeated_ops)
            for op_name, op_obj in repeated_ops.items():
                op_obj.repeat_count = self.block_counts[self.block_keys[bi]]

        # combine the mHC streams (hc_head sinkhorn reduction approximated by a sum)
        Yf = streams[0]
        for i in range(1, self.hc_mult):
            addop = F.Add(f'{self.name}.hc_head_add{i}')
            addop.set_module(self)
            self._op_hndls[addop.name] = addop
            Yf = addop(Yf, streams[i])
        Yf = self.final_norm(Yf)
        return self.lm_head(Yf)  # (B, L, vocab)

    def analytical_param_count(self, lvl=0):
        total = 0
        for bi in range(len(self.block_keys)):
            total += (self.block_counts[self.block_keys[bi]]
                      * self.blocks[bi].analytical_param_count(lvl + 1))
        total += self.hidden  # final norm
        total += 2 * self.vocab_size * self.hidden  # embedding + untied lm_head
        # hc_head mixing params (final sinkhorn reduction of the streams)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        total += mix_hc * (self.hc_mult * self.hidden) + mix_hc + 3
        return total


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # tiny smoke config
    tiny_cfg = {
        'vocab_size': 256, 'hidden_size': 64, 'num_hidden_layers': 6,
        'num_hash_layers': 2, 'compress_ratios': [0, 4, 128, 4, 128, 0],
        'num_attention_heads': 4, 'head_dim': 32,
        'q_lora_rank': 16, 'o_lora_rank': 16, 'o_groups': 2,
        'sliding_window': 4,
        'n_routed_experts': 8, 'num_experts_per_tok': 2,
        'moe_intermediate_size': 32, 'n_shared_experts': 1,
        'index_n_heads': 2, 'index_head_dim': 8, 'index_topk': 2,
        'hc_mult': 4, 'bs': 1, 'seq_len': 8, 'rms_norm_eps': 1e-6,
    }
    model = Dsv4FlashMoe('dsv4f_tiny', tiny_cfg)
    model.create_input_tensors()
    out = model()
    print(f'tiny forward OK, output shape: {out.shape}')
    print(f'tiny param count: {model.analytical_param_count():,}')

    # full V4-Flash decode step
    m = Dsv4FlashMoe('dsv4f_flash', {'bs': 1, 'seq_len': 1})
    m.create_input_tensors()
    out = m()
    print(f'V4-Flash decode forward OK, output shape: {out.shape}')
    print(f'V4-Flash total params: {m.analytical_param_count():,}')

    # prefill: exercises CSA top-512 / HCA all-block sparse windows
    mp = Dsv4FlashMoe('dsv4f_prefill', {'bs': 1, 'seq_len': 2048})
    mp.create_input_tensors()
    out = mp()
    print(f'V4-Flash prefill(2048) forward OK, output shape: {out.shape}')