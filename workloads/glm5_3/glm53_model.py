#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3 -- TTSIM workload (DeepSeek-style MLA + DSA sparse attention, sparse MoE).

Architecture pinned from HF ``zai-org/GLM-5.3`` ``config.json``
(``model_type: glm_moe_dsa``, ``GlmMoeDsaForCausalLM``, 753B total / ~40B active):

  - vocab_size               : 154880 (untied embedding / LM head)
  - hidden_size              : 6144
  - num_hidden_layers        : 78; ``mlp_layer_types``: first 3 dense
                               (intermediate_size 12288), remaining 75 sparse MoE
  - MoE                      : 256 routed experts + 1 shared, top-8 per token,
                               expert intermediate 2048, sigmoid scoring with
                               ``noaux_tc`` correction bias, ``norm_topk_prob``,
                               routed_scaling_factor 2.5, fp32 router
  - attention (MLA)          : 64 Q-heads / 64 KV-heads, q_lora_rank 2048,
                               kv_lora_rank 512, qk_nope_head_dim 192,
                               qk_rope_head_dim 64, v_head_dim 256,
                               rope_theta 8e6 (interleaved), attention_bias false
  - DSA indexer              : 32 index heads x 128 dims, top-2048 tokens per
                               query, ``indexer_types``: full at layers 0-3 and
                               every 4th layer after (22 full), shared elsewhere
                               (56 shared -- indices reused, no indexer compute)
  - 1 MTP (nextn) layer      : not modeled (see README)
  - fp8 (e4m3) quantized checkpoint; simulation dtypes come from the
    workload-to-arch mapping, not the checkpoint

Performance-simulation abstractions (documented; see README.md):

  1. Layer multiplicity: three proxy blocks are executed -- dense-MLP+full-indexer
     (x3), MoE+full-indexer (x19), MoE+shared-indexer (x56); op ``repeat_count``
     scales each (same pattern as ``workloads/BasicLLM.py``).
  2. MoE models ACTIVE compute: top-8 experts + shared expert applied to routed
     activations (per-token dispatch/combine not modeled).
  3. DSA top-k token selection is data-dependent; the graph is static, so the
     selected-KV window is approximated by the first ``min(L, index_topk)``
     positions when L > index_topk (cost-equivalent: the attention matmuls shrink
     from (L, L) to (L, topk) either way).
  4. Shared-indexer layers perform no indexer compute (indices reused from the
     preceding full layer); indexer weights still count toward parameters.
  5. RoPE (interleaved, 64-dim rope component of MLA) is approximated by
     elementwise cos/sin mixing; cost is negligible.
  6. MTP draft layer is not part of the main inference loop and is not simulated.
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import math
from typing import Any

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

# small shared blocks identical to the Qwen3.5 workload
from workloads.qwen3_5.qwen35_model import MoEExpert, RMSNorm


# ----------------------------------------------------------------------------
class DenseFFN(SimNN.Module):
    """Dense SwiGLU FFN (first-k layers): down( silu(gate(x)) * up(x) )"""

    def __init__(self, name, dim, inter_dim):
        super().__init__()
        self.name = name
        self.dim = dim
        self.inter_dim = inter_dim
        self.gate_proj = SimNN.Linear(f'{name}.gate_proj', dim, inter_dim, bias=False)
        self.up_proj = SimNN.Linear(f'{name}.up_proj', dim, inter_dim, bias=False)
        self.down_proj = SimNN.Linear(f'{name}.down_proj', inter_dim, dim, bias=False)
        self.silu_sig_op = F.Sigmoid(f'{name}.silu_sig')
        self.silu_mul_op = F.Mul(f'{name}.silu_mul')
        self.h_mul_op = F.Mul(f'{name}.h_mul')
        super().link_op2module()

    def __call__(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        h = self.h_mul_op(self.silu_mul_op(g, self.silu_sig_op(g)), u)
        return self.down_proj(h)

    def analytical_param_count(self, lvl=0):
        return 3 * self.dim * self.inter_dim


# ----------------------------------------------------------------------------
class GlmSparseMoE(SimNN.Module):
    """GLM-5 MoE: sigmoid router + noaux_tc bias + top-k (normalized, scaled).

    scoring: s = sigmoid(router(x)) + e_score_correction_bias
    weights: topk(s, k) -> normalize -> x routed_scaling_factor
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.hidden = cfg.get('hidden_size', 6144)
        self.num_experts = cfg.get('n_routed_experts', 256)
        self.top_k = cfg.get('num_experts_per_tok', 8)
        self.moe_inter = cfg.get('moe_intermediate_size', 2048)
        self.shared_inter = cfg.get('moe_intermediate_size', 2048)  # shared == expert width
        self.n_shared = cfg.get('n_shared_experts', 1)
        self.rsf = cfg.get('routed_scaling_factor', 2.5)

        self.router = SimNN.Linear(f'{name}.gate', self.hidden, self.num_experts, bias=False)
        self.e_score_correction_bias = F._from_shape(f'{name}.e_score_correction_bias',
                                                      shape=[self.num_experts], is_param=True)
        self.sigmoid_op = F.Sigmoid(f'{name}.router_sigmoid')
        self.bias_add_op = F.Add(f'{name}.noaux_tc_bias')
        self.topk_op = F.topk(f'{name}.topk', k=self.top_k)
        self.wsum_op = F.Sum(f'{name}.topk_sum', dim=-1)
        self.wdiv_op = F.Div(f'{name}.topk_norm')
        self.rsf_t = F._from_data(f'{name}.routed_scaling_factor',
                                  data=np.float32(self.rsf), is_const=True)
        self.rsf_mul_op = F.Mul(f'{name}.routed_scale')
        self.rsf_out_mul_op = F.Mul(f'{name}.routed_scale_out')

        self.experts = SimNN.ModuleList(
            [MoEExpert(f'{name}.experts.{i}', self.hidden, self.moe_inter) for i in range(self.top_k)]
        )
        self.shared_expert = MoEExpert(f'{name}.shared_experts.0', self.hidden, self.shared_inter)
        super().link_op2module()

    def __call__(self, x):
        s = self.sigmoid_op(self.router(x))  # (B, L, E)
        s = self.bias_add_op(s, self.e_score_correction_bias)  # noaux_tc
        top_vals, _top_idx = self.topk_op(s)  # (B, L, k)
        wsum = self.wsum_op(top_vals).unsqueeze(-1)  # (B, L, 1)
        wnorm = self.wdiv_op(top_vals, wsum)  # norm_topk_prob
        wscale = self.rsf_mul_op(wnorm, self.rsf_t)  # x routed_scaling_factor

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
        routed = self.num_experts * (3 * self.hidden * self.moe_inter)
        shared = self.n_shared * (3 * self.hidden * self.shared_inter)
        return self.hidden * self.num_experts + self.num_experts + routed + shared


# ----------------------------------------------------------------------------
class MlaDsaAttention(SimNN.Module):
    """Multi-head Latent Attention with optional DSA indexer.

    Q path : x -> q_a (lora 2048) -> RMSNorm -> q_b (64 heads x 256)
    KV path: x -> kv_a (latent 512 + rope 64) -> RMSNorm(latent) -> kv_b
             (per-head nope-k 192 + v 256); k_rope shared across heads.
    Scores : q_nope.k_nope + q_rope.k_rope, scaled, [causal mask], softmax.
    DSA    : on 'full' layers an indexer (32 heads x 128) scores all tokens and
             top-(min(topk, L-1)) positions are selected; attention matmuls run
             against min(L, index_topk) KV positions. 'shared' layers reuse the
             selection (no indexer compute).
    """

    def __init__(self, name, cfg, indexer_kind):
        super().__init__()
        assert indexer_kind in ('full', 'shared', 'none'), f'bad indexer kind: {indexer_kind}'
        self.name = name
        self.hidden = cfg.get('hidden_size', 6144)
        self.n_heads = cfg.get('num_attention_heads', 64)
        self.q_lora = cfg.get('q_lora_rank', 2048)
        self.kv_lora = cfg.get('kv_lora_rank', 512)
        self.rope_dim = cfg.get('qk_rope_head_dim', 64)
        self.nope_dim = cfg.get('qk_nope_head_dim', 192)
        self.v_dim = cfg.get('v_head_dim', 256)
        self.qk_head = self.nope_dim + self.rope_dim  # 256
        self.eps = cfg.get('rms_norm_eps', 1e-5)
        self.seq_len = cfg.get('seq_len', 1)
        self.indexer_kind = indexer_kind
        self.idx_heads = cfg.get('index_n_heads', 32)
        self.idx_dim = cfg.get('index_head_dim', 128)
        self.index_topk = cfg.get('index_topk', 2048)

        # MLA projections
        self.q_a_proj = SimNN.Linear(f'{name}.q_a_proj', self.hidden, self.q_lora, bias=False)
        self.q_a_norm = RMSNorm(f'{name}.q_a_layernorm', self.q_lora, eps=self.eps)
        self.q_b_proj = SimNN.Linear(f'{name}.q_b_proj', self.q_lora,
                                     self.n_heads * self.qk_head, bias=False)
        self.kv_a_proj = SimNN.Linear(f'{name}.kv_a_proj', self.hidden,
                                      self.kv_lora + self.rope_dim, bias=False)
        self.kv_a_norm = RMSNorm(f'{name}.kv_a_layernorm', self.kv_lora, eps=self.eps)
        self.kv_b_proj = SimNN.Linear(f'{name}.kv_b_proj', self.kv_lora,
                                      self.n_heads * (self.nope_dim + self.v_dim), bias=False)
        self.o_proj = SimNN.Linear(f'{name}.o_proj', self.n_heads * self.v_dim, self.hidden,
                                   bias=False)

        # attention ops
        self.scale_t = F._from_data(f'{name}.scale',
                                     data=np.float32(1.0 / math.sqrt(self.qk_head)), is_const=True)
        self.scale_mul_op = F.Mul(f'{name}.scale_mul')
        self.scores_add_op = F.Add(f'{name}.scores_add')
        self.softmax_op = F.Softmax(f'{name}.softmax')
        # RoPE approximation constants
        self.rope_cos = F._from_shape(f'{name}.rope_cos', shape=[1, 1, 1, self.rope_dim],
                                      is_const=True)
        self.rope_sin = F._from_shape(f'{name}.rope_sin', shape=[1, 1, 1, self.rope_dim],
                                      is_const=True)
        if self.seq_len > 1:
            self.causal_mask = F._from_shape(f'{name}.causal_mask',
                                             shape=[1, 1, self.seq_len, self.seq_len],
                                             is_const=True)

        # DSA indexer (weights exist on every layer; compute only on 'full' layers)
        self.wq_idx = SimNN.Linear(f'{name}.indexers.wq_idx', self.hidden,
                                   self.idx_heads * self.idx_dim, bias=False)
        self.wk_idx = SimNN.Linear(f'{name}.indexers.wkvw_idx', self.hidden,
                                   self.idx_heads * self.idx_dim, bias=False)
        if indexer_kind == 'full':
            self.idx_mean_op = F.ReduceMean(f'{name}.idx_head_mean', axes=[1])
        # tensors created via _from_* do not inherit link_module from Module.__setattr__;
        # set it explicitly so direct tensor-method calls (slicing, etc.) can resolve ops
        for t in (self.scale_t, self.rope_cos, self.rope_sin):
            t.set_module(self)
        if self.seq_len > 1:
            self.causal_mask.set_module(self)
        super().link_op2module()

    # -- indexer: returns effective sparse-attention width -------------------
    def _indexer_forward(self, x, B, L):
        """Run the DSA indexer; returns the effective KV window size."""
        eff = min(L, self.index_topk)
        if self.indexer_kind != 'full' or L <= 1:
            return eff  # shared/none: no compute; decode: no-op selection
        q_idx = self.wq_idx(x).reshape(B, L, self.idx_heads, self.idx_dim).permute([0, 2, 1, 3])
        k_idx = self.wk_idx(x).reshape(B, L, self.idx_heads, self.idx_dim).permute([0, 2, 1, 3])
        isc = T.matmul(q_idx, k_idx.transpose(2, 3))  # (B, IH, L, L)
        isc_avg = self.idx_mean_op(isc)  # (B, L, L) -- average over index heads
        k_sel = min(self.index_topk, L - 1)
        idx_topk_op = F.topk(f'{self.name}.idx_topk', k=k_sel)
        idx_topk_op.set_module(self)
        self._op_hndls[idx_topk_op.name] = idx_topk_op
        _vals, _idx = idx_topk_op(isc_avg)  # selection (indices drive KV gather on HW)
        return eff

    def __call__(self, x):
        B, L, _ = x.shape
        assert _ == self.hidden, f'{self.name}: input hidden dim mismatch'

        eff = self._indexer_forward(x, B, L)  # effective KV window (DSA)

        # ---- Q path
        q_a = self.q_a_norm(self.q_a_proj(x))  # (B, L, 2048)
        q_b = self.q_b_proj(q_a)  # (B, L, 16384)
        q = q_b.reshape(B, L, self.n_heads, self.qk_head).permute([0, 2, 1, 3])  # (B, H, L, 256)
        q_nope = q[:, :, :, 0:self.nope_dim]  # (B, H, L, 192)
        q_rope = q[:, :, :, self.nope_dim:]  # (B, H, L, 64)
        q_rope = q_rope * self.rope_cos + q_rope * self.rope_sin  # RoPE approx

        # ---- KV path (latent compression)
        kv_a = self.kv_a_proj(x)  # (B, L, 576)
        c_kv = self.kv_a_norm(kv_a[:, :, 0:self.kv_lora])  # (B, L, 512)
        k_rope = kv_a[:, :, self.kv_lora:]  # (B, L, 64), shared across heads
        k_rope = (k_rope.reshape(B, L, 1, self.rope_dim)
                  .repeat(1, 1, self.n_heads, 1)  # (B, L, H, 64)
                  .permute([0, 2, 1, 3]))  # (B, H, L, 64)
        k_rope = k_rope * self.rope_cos + k_rope * self.rope_sin

        kv_b = self.kv_b_proj(c_kv)  # (B, L, 28672)
        k_nope = (kv_b[:, :, 0:self.n_heads * self.nope_dim]
                  .reshape(B, L, self.n_heads, self.nope_dim).permute([0, 2, 1, 3]))  # (B, H, L, 192)
        v = (kv_b[:, :, self.n_heads * self.nope_dim:]
             .reshape(B, L, self.n_heads, self.v_dim).permute([0, 2, 1, 3]))  # (B, H, L, 256)

        # ---- DSA window: approximate dynamic top-k gather with a static slice
        if eff < L:
            k_nope = k_nope[:, :, 0:eff, :]
            k_rope = k_rope[:, :, 0:eff, :]
            v = v[:, :, 0:eff, :]

        # ---- scores
        s_nope = T.matmul(q_nope, k_nope.transpose(2, 3))  # (B, H, L, S)
        s_rope = T.matmul(q_rope, k_rope.transpose(2, 3))  # (B, H, L, S)
        scores = self.scale_mul_op(self.scores_add_op(s_nope, s_rope), self.scale_t)
        if L > 1:
            # SimTensor.__getitem__ is attached at runtime by tensor_op; type as Any for mypy
            mask: Any = self.causal_mask
            scores = scores + mask[:, :, 0:L, 0:eff]
        probs = self.softmax_op(scores)  # (B, H, L, S)

        # ---- context
        ctx = T.matmul(probs, v)  # (B, H, L, 256)
        ctx = ctx.permute([0, 2, 1, 3]).reshape(B, L, self.n_heads * self.v_dim)  # (B, L, 16384)
        return self.o_proj(ctx)  # (B, L, hidden)

    def analytical_param_count(self, lvl=0):
        p = self.hidden * self.q_lora  # q_a
        p += self.q_lora * self.n_heads * self.qk_head  # q_b
        p += self.hidden * (self.kv_lora + self.rope_dim)  # kv_a
        p += self.kv_lora * self.n_heads * (self.nope_dim + self.v_dim)  # kv_b
        p += self.n_heads * self.v_dim * self.hidden  # o_proj
        p += self.q_lora + self.kv_lora  # q_a / kv_a layernorm weights
        # DSA indexer weights (present on every layer; executed on 'full' only)
        p += 2 * self.hidden * self.idx_heads * self.idx_dim
        return p


# ----------------------------------------------------------------------------
class Glm53Layer(SimNN.Module):
    """One decoder layer: norm -> MLA(+DSA) -> residual -> norm -> FFN(dense|MoE)."""

    def __init__(self, name, cfg, mlp_kind, indexer_kind):
        super().__init__()
        assert mlp_kind in ('dense', 'sparse'), f'bad mlp kind: {mlp_kind}'
        self.name = name
        self.hidden = cfg.get('hidden_size', 6144)
        self.eps = cfg.get('rms_norm_eps', 1e-5)
        self.input_norm = RMSNorm(f'{name}.input_layernorm', self.hidden, eps=self.eps)
        self.attn = MlaDsaAttention(f'{name}.self_attn', cfg, indexer_kind)
        self.post_norm = RMSNorm(f'{name}.post_attention_layernorm', self.hidden, eps=self.eps)
        self.ffn: 'DenseFFN | GlmSparseMoE'
        if mlp_kind == 'dense':
            self.ffn = DenseFFN(f'{name}.mlp', self.hidden, cfg.get('intermediate_size', 12288))
        else:
            self.ffn = GlmSparseMoE(f'{name}.mlp', cfg)
        self.res1_op = F.Add(f'{name}.residual_add1')
        self.res2_op = F.Add(f'{name}.residual_add2')
        super().link_op2module()

    def __call__(self, x):
        h = self.res1_op(x, self.attn(self.input_norm(x)))
        return self.res2_op(h, self.ffn(self.post_norm(h)))

    def analytical_param_count(self, lvl=0):
        return (self.attn.analytical_param_count(lvl + 1)
                + self.ffn.analytical_param_count(lvl + 1)
                + 2 * self.hidden)


# ----------------------------------------------------------------------------
class Glm53Moe(SimNN.Module):
    """GLM-5.3 top-level model (TTSIM entry class).

    Instantiated by polaris as ``Glm53Moe(instance_name, merged_cfg)``; all
    GLM-5.3 defaults are built in, instances typically only override ``bs``
    and ``seq_len``.
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.vocab_size = cfg.get('vocab_size', 154880)
        self.hidden = cfg.get('hidden_size', 6144)
        self.num_layers = cfg.get('num_hidden_layers', 78)
        self.n_dense = cfg.get('first_k_dense_replace', 3)
        self.bs = cfg.get('bs', 1)
        self.seq_len = cfg.get('seq_len', 1)
        self.eps = cfg.get('rms_norm_eps', 1e-5)
        self.index_topk = cfg.get('index_topk', 2048)

        assert self.num_layers > self.n_dense, 'need at least one MoE layer'
        n_moe = self.num_layers - self.n_dense  # 75
        # indexer_types: full at layers 0..3, then every 4th layer (7, 11, ...)
        self.n_full_idx = sum(1 for i in range(self.num_layers)
                               if i < 4 or (i - 3) % 4 == 0)  # 22
        n_full_dense = min(self.n_dense, self.n_full_idx)  # 3 (layers 0-2 dense+full)
        n_full_moe = self.n_full_idx - n_full_dense  # 19 (layer 3 + every 4th)
        self.n_shared_moe = n_moe - n_full_moe  # 56
        assert n_full_dense + n_full_moe + self.n_shared_moe == self.num_layers
        # block multiplicities for the proxy pattern (used by __call__ / param count)
        self.n_dense_full = n_full_dense
        self.n_moe_full = n_full_moe

        # ops / modules
        self.wte = F.Embedding(f'{name}.wte', self.vocab_size, self.hidden)

        # proxy blocks (repeat_count scaling below)
        self.dense_full_block = Glm53Layer(f'{name}.dense_full_block', cfg, 'dense', 'full')
        self.moe_full_block = Glm53Layer(f'{name}.moe_full_block', cfg, 'sparse', 'full')
        self.moe_shared_block = Glm53Layer(f'{name}.moe_shared_block', cfg, 'sparse', 'shared')

        self.final_norm = RMSNorm(f'{name}.final_norm', self.hidden, eps=self.eps)
        self.lm_head = SimNN.Linear(f'{name}.lm_head', self.hidden, self.vocab_size, bias=False)
        self.input_tensors = {}
        super().link_op2module()

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
        Y = self.dense_full_block(Y)
        Y = self.moe_full_block(Y)
        Y = self.moe_shared_block(Y)

        # scale the proxy blocks by layer multiplicity (BasicLLM repeat_count pattern)
        for blk, rpt in ((self.dense_full_block, self.n_dense_full),
                         (self.moe_full_block, self.n_moe_full),
                         (self.moe_shared_block, self.n_shared_moe)):
            repeated_ops: dict = {}
            blk.get_ops(repeated_ops)
            for op_name, op_obj in repeated_ops.items():
                op_obj.repeat_count = rpt

        Y = self.final_norm(Y)
        Y = self.lm_head(Y)  # (B, L, vocab)
        return Y

    def analytical_param_count(self, lvl=0):
        d = self.dense_full_block.analytical_param_count(lvl + 1)
        mf = self.moe_full_block.analytical_param_count(lvl + 1)
        ms = self.moe_shared_block.analytical_param_count(lvl + 1)
        total = (self.n_dense_full * d + self.n_moe_full * mf
                 + self.n_shared_moe * ms)
        total += self.hidden  # final norm
        total += 2 * self.vocab_size * self.hidden  # embedding + untied lm_head
        return total


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    # tiny smoke config
    tiny_cfg = {
        'vocab_size': 256, 'hidden_size': 64, 'num_hidden_layers': 8,
        'first_k_dense_replace': 1, 'intermediate_size': 128,
        'num_attention_heads': 4, 'num_key_value_heads': 4,
        'q_lora_rank': 32, 'kv_lora_rank': 16,
        'qk_nope_head_dim': 8, 'qk_rope_head_dim': 4, 'v_head_dim': 16,
        'n_routed_experts': 8, 'num_experts_per_tok': 2,
        'moe_intermediate_size': 32, 'n_shared_experts': 1,
        'index_n_heads': 2, 'index_head_dim': 8, 'index_topk': 4,
        'bs': 1, 'seq_len': 8, 'rms_norm_eps': 1e-5,
    }
    model = Glm53Moe('glm53_tiny', tiny_cfg)
    model.create_input_tensors()
    out = model()
    print(f'tiny forward OK, output shape: {out.shape}')
    print(f'tiny param count: {model.analytical_param_count():,}')

    # full GLM-5.3 decode step
    m = Glm53Moe('glm53_full', {'bs': 1, 'seq_len': 1})
    m.create_input_tensors()
    out = m()
    print(f'GLM-5.3 decode forward OK, output shape: {out.shape}')
    print(f'GLM-5.3 total params: {m.analytical_param_count():,}')