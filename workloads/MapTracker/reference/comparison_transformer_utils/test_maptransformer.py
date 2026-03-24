#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MapTransformer Numerical Comparison Suite: PyTorch vs TTSim

Full numerical validation of the transformer decoder pipeline
with shared weights, exercising the actual TTSim class APIs.

Tests:
  1. Deformable attention standalone
  2. Self-attention (MHA) standalone
  3. FFN standalone
  4. Single layer through MapTransformerLayer.__call__ (self_attn + cross_attn + ffn + norms)
  5. Full 2-layer decoder through MapTransformerDecoder_new.__call__
  6. Decoder with regression branch refinement
  7. Varied inputs (edge refs, small dims)
  8. Weight transfer fidelity
"""

import os, sys, traceback, math

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.MapTracker.plugin.models.transformer_utils.MapTransformer import (
    MapTransformerDecoder_new,
    MapTransformerLayer,
)
from workloads.MapTracker.plugin.models.transformer_utils.custom_msdeformable_attention import (
    CustomMSDeformableAttention,
)
from workloads.MapTracker.plugin.models.transformer_utils.multihead_attention import (
    MultiheadAttention as MultiheadAttentionTTSim,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.builder_utils import (
    LayerNorm,
    FFN as FFNTTSim,
)

# ============================================================================
# Helpers
# ============================================================================


def _stats(label, arr):
    print(
        f"    {label}  shape={arr.shape}  "
        f"range=[{arr.min():.4e}, {arr.max():.4e}]  mean={arr.mean():.4e}"
    )


def _diff(a, b):
    d = np.abs(a - b)
    print(f"    range=[{d.min():.4e}, {d.max():.4e}]  mean={d.mean():.4e}")
    return d


def _check(name, a, b, atol=1e-5):
    d = _diff(a, b)
    ok = d.max() < atol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: max={d.max():.2e}  mean={d.mean():.2e}")
    return ok


# ============================================================================
# PyTorch Reference Implementations
# ============================================================================


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """CPU-only PyTorch deformable attention (reference)."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    if isinstance(value_spatial_shapes, torch.Tensor):
        value_spatial_shapes = [(int(H_), int(W_)) for H_, W_ in value_spatial_shapes]

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F_torch.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class CustomMSDeformableAttentionPyTorch(nn.Module):
    """PyTorch reference CustomMSDeformableAttention."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, value, reference_points, spatial_shapes, **kwargs):
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(
            bs, num_value, self.num_heads, self.embed_dims // self.num_heads
        )

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if isinstance(spatial_shapes, torch.Tensor):
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
        else:
            offset_normalizer = torch.tensor(
                [[w, h] for h, w in spatial_shapes],
                dtype=torch.float32,
                device=query.device,
            )

        ref_points = reference_points[:, :, 0, :][:, :, None, None, None, :]
        sampling_locations = ref_points + (
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)


class MapTransformerLayerPyTorch(nn.Module):
    """
    PyTorch reference: full transformer layer.

    Operation order: self_attn -> norm -> cross_attn -> norm -> ffn -> norm
    (Matches the real MapTracker decoder layer, minus memory cross-attn.)
    """

    def __init__(
        self, embed_dims=256, num_heads=8, num_levels=1, num_points=4, ffn_dim=1024
    ):
        super().__init__()
        self.embed_dims = embed_dims
        # Self-attention (standard MHA)
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0.0, batch_first=False
        )
        # Cross-attention (deformable)
        self.cross_attn = CustomMSDeformableAttentionPyTorch(
            embed_dims, num_heads, num_levels, num_points
        )
        # FFN
        self.ffn_fc1 = nn.Linear(embed_dims, ffn_dim)
        self.ffn_fc2 = nn.Linear(ffn_dim, embed_dims)
        # Norms: after self_attn, after cross_attn, after ffn
        self.norm0 = nn.LayerNorm(embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        **kwargs,
    ):
        """
        Args:
            query: [num_queries, bs, embed_dims]
            key/value: [num_value, bs, embed_dims] (BEV features)
            query_pos: [num_queries, bs, embed_dims]
            reference_points: [bs, num_queries, num_points_per_query, 2]
            spatial_shapes: list of (H, W)
        """
        # --- self-attention ---
        q_sa = query + query_pos if query_pos is not None else query
        k_sa = query + query_pos if query_pos is not None else query
        sa_out, _ = self.self_attn(q_sa, k_sa, query)
        query = query + sa_out  # residual
        query = self.norm0(query)

        # --- cross-attention (deformable) ---
        # Add query_pos before deformable attention (matches MMCV behaviour).
        # Identity (for residual) is the raw query *without* pos.
        query_with_pos = query + query_pos if query_pos is not None else query
        query_bf = query_with_pos.permute(1, 0, 2)  # [bs, nq, C]
        key_bf = key.permute(1, 0, 2)
        ca_out = self.cross_attn(query_bf, key_bf, reference_points, spatial_shapes)
        query = query + ca_out.permute(1, 0, 2)  # residual (identity = pre-pos query)
        query = self.norm1(query)

        # --- FFN ---
        ffn_out = self.ffn_fc2(torch.relu(self.ffn_fc1(query)))
        query = query + ffn_out  # residual
        query = self.norm2(query)

        return query


class MapTransformerDecoderPyTorch(nn.Module):
    """PyTorch reference: multi-layer decoder with y-axis reversal and reg branches."""

    def __init__(
        self,
        num_layers=2,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        ffn_dim=1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.layers = nn.ModuleList(
            [
                MapTransformerLayerPyTorch(
                    embed_dims, num_heads, num_levels, num_points, ffn_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query,
        key,
        value,
        query_pos,
        reference_points,
        spatial_shapes,
        reg_branches=None,
        predict_refine=False,
        **kwargs,
    ):
        """
        Args:
            query: [num_queries, bs, embed_dims]
            reference_points: [bs, nq, npts, 2]
            reg_branches: list of nn.Sequential (one per layer), or None
        """
        intermediate = []
        intermediate_ref_pts = []
        output = query
        num_queries = query.shape[0]
        bs = query.shape[1]

        for lid, layer in enumerate(self.layers):
            # y-axis reversal
            tmp_ref = reference_points.clone()
            tmp_ref[..., 1] = 1.0 - reference_points[..., 1]

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                reference_points=tmp_ref,
                spatial_shapes=spatial_shapes,
            )

            # Regression branch
            if reg_branches is not None:
                output_perm = output.permute(1, 0, 2)  # [bs, nq, C]
                reg_pts = reg_branches[lid](output_perm)  # [bs, nq, npts*2]
                num_pts = reference_points.shape[2]
                reg_pts = reg_pts.view(bs, num_queries, num_pts, 2)

                if predict_refine:
                    ref_clamped = reference_points.clamp(1e-5, 1.0 - 1e-5)
                    inv_sig = torch.log(ref_clamped / (1.0 - ref_clamped))
                    new_ref = torch.sigmoid(reg_pts + inv_sig)
                else:
                    new_ref = torch.sigmoid(reg_pts)

                reference_points = new_ref.detach()
            else:
                new_ref = reference_points

            intermediate.append(output.permute(1, 0, 2))  # [bs, nq, C]
            intermediate_ref_pts.append(new_ref)

        return intermediate, intermediate_ref_pts


# ============================================================================
# Weight Transfer Helpers
# ============================================================================


def transfer_deform_attn_weights(pt_attn, tt_attn):
    """Transfer weights: PyTorch CustomMSDeformableAttention -> TTSim."""
    tt_attn.sampling_offsets.param.data = pt_attn.sampling_offsets.weight.data.numpy()
    tt_attn.sampling_offsets.bias.data = pt_attn.sampling_offsets.bias.data.numpy()
    tt_attn.attention_weights.param.data = pt_attn.attention_weights.weight.data.numpy()
    tt_attn.attention_weights.bias.data = pt_attn.attention_weights.bias.data.numpy()
    tt_attn.value_proj.param.data = pt_attn.value_proj.weight.data.numpy()
    tt_attn.value_proj.bias.data = pt_attn.value_proj.bias.data.numpy()
    tt_attn.output_proj.param.data = pt_attn.output_proj.weight.data.numpy()
    tt_attn.output_proj.bias.data = pt_attn.output_proj.bias.data.numpy()


def transfer_mha_weights(pt_mha, tt_mha):
    """Transfer weights: PyTorch nn.MultiheadAttention -> TTSim MHA.

    PyTorch packs Q/K/V into in_proj_weight [3*E, E] and in_proj_bias [3*E].
    TTSim has separate q_proj, k_proj, v_proj.
    Both PyTorch and TTSim Linear store param as [out, in].
    """
    E = pt_mha.embed_dim
    w = pt_mha.in_proj_weight.data.numpy()
    b = pt_mha.in_proj_bias.data.numpy()
    tt_mha.q_proj.param.data = w[:E, :]
    tt_mha.q_proj.bias.data = b[:E]
    tt_mha.k_proj.param.data = w[E : 2 * E, :]
    tt_mha.k_proj.bias.data = b[E : 2 * E]
    tt_mha.v_proj.param.data = w[2 * E :, :]
    tt_mha.v_proj.bias.data = b[2 * E :]
    tt_mha.out_proj.param.data = pt_mha.out_proj.weight.data.numpy()
    tt_mha.out_proj.bias.data = pt_mha.out_proj.bias.data.numpy()


def transfer_ffn_weights(pt_fc1, pt_fc2, tt_ffn):
    """Transfer FFN weights from PyTorch Linear layers to TTSim FFN."""
    tt_ffn.layers[0].param.data = pt_fc1.weight.data.numpy()
    tt_ffn.layers[0].bias.data = pt_fc1.bias.data.numpy()
    tt_ffn.layers[1].param.data = pt_fc2.weight.data.numpy()
    tt_ffn.layers[1].bias.data = pt_fc2.bias.data.numpy()


def transfer_layer_weights(pt_layer, tt_layer, lid):
    """Transfer all weights for one transformer layer.

    TTSim has 4 norms [0..3]; norm-1 sits between the two cross_attns and is
    effectively wasted when memory_bank=None.  PyTorch has 3 norms (norm0,
    norm1, norm2).  Mapping:
        PT norm0 -> TT norm0  (after self_attn)
        TT norm1              (between cross_attns, identity affine)
        PT norm1 -> TT norm2  (after memory cross_attn / query_bev)
        PT norm2 -> TT norm3  (after FFN)
    """
    # Self-attention (attentions[0])
    transfer_mha_weights(pt_layer.self_attn, tt_layer.attentions[0])
    # Cross-attention (attentions[1])
    transfer_deform_attn_weights(pt_layer.cross_attn, tt_layer.attentions[1])
    # FFN
    transfer_ffn_weights(pt_layer.ffn_fc1, pt_layer.ffn_fc2, tt_layer.ffns[0])
    # LayerNorm affine params (TTSim LN has no affine, store externally)
    pt_norms = [pt_layer.norm0, pt_layer.norm1, pt_layer.norm2]
    tt_indices = [0, 2, 3]  # skip TTSim norm-1 (wasted)
    for pt_norm, ti in zip(pt_norms, tt_indices):
        tt_layer.norms[ti]._affine_weight = F._from_data(
            f"ln_w_l{lid}_n{ti}", pt_norm.weight.data.numpy().copy(), is_const=True
        )
        tt_layer.norms[ti]._affine_bias = F._from_data(
            f"ln_b_l{lid}_n{ti}", pt_norm.bias.data.numpy().copy(), is_const=True
        )
    # TTSim norm-1: identity affine (w=1, b=0)
    E = pt_layer.norm0.weight.shape[0]
    tt_layer.norms[1]._affine_weight = F._from_data(
        f"ln_w_l{lid}_n1", np.ones(E, dtype=np.float32), is_const=True
    )
    tt_layer.norms[1]._affine_bias = F._from_data(
        f"ln_b_l{lid}_n1", np.zeros(E, dtype=np.float32), is_const=True
    )


# ============================================================================
# TTSim Layer Builder
# ============================================================================


def build_ttsim_layer(lid, embed_dims, num_heads, num_levels, num_points, ffn_dim):
    """Build one MapTransformerLayer with full operation order (3 attentions, 4 norms).

    Operation order matches the real MapTracker decoder:
        self_attn -> norm -> cross_attn(BEV) -> norm -> cross_attn(mem) -> norm -> ffn -> norm

    The third attention (memory cross-attn) is never called when memory_bank=None;
    its presence is needed so that the second cross_attn block in __call__ can
    assign query_bev back to query.
    """
    layer = MapTransformerLayer(
        embed_dims=embed_dims,
        num_attn=3,  # self_attn + BEV cross_attn + memory cross_attn
        num_ffn=1,
        pre_norm=False,
        operation_order=[
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ],
    )

    # Self-attention (standard MHA, batch_first=False for [nq, bs, C])
    self_attn = MultiheadAttentionTTSim(
        name=f"self_attn_l{lid}",
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=False,
    )
    layer.add_attention(self_attn)

    # Cross-attention (deformable, batch_first=False — layer passes seq-first [nq, bs, C])
    cross_attn = CustomMSDeformableAttention(
        name=f"cross_attn_l{lid}",
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=False,
        dropout=0.0,
    )
    layer.add_attention(cross_attn)

    # Memory cross-attention (MHA, dummy — never called without memory_bank)
    mem_attn = MultiheadAttentionTTSim(
        name=f"mem_attn_l{lid}",
        embed_dims=embed_dims,
        num_heads=num_heads,
        batch_first=True,
    )
    layer.add_attention(mem_attn)

    # 4 norms: after self_attn, between cross_attns, after memory, after FFN
    for ni in range(4):
        norm = LayerNorm(f"norm_l{lid}_n{ni}", normalized_shape=embed_dims)
        layer.add_norm(norm)

    # FFN (2-layer with ReLU + residual)
    ffn = FFNTTSim(
        name=f"ffn_l{lid}",
        embed_dims=embed_dims,
        feedforward_channels=ffn_dim,
        num_fcs=2,
        add_identity=True,
    )
    layer.add_ffn(ffn)

    layer.finalize()
    return layer


# ============================================================================
# Monkey-patch: inject affine transform after LayerNorm normalisation
# ============================================================================

_original_norm_call = LayerNorm.__call__


def _patched_norm_call(self, x):
    """Override LayerNorm.__call__ to apply stored affine weight/bias."""
    normalized = _original_norm_call(self, x)
    if hasattr(self, "_affine_weight") and self._affine_weight is not None:
        normalized = F.Mul(self.name + ".scale")(normalized, self._affine_weight)
        normalized = F.Add(self.name + ".shift")(normalized, self._affine_bias)
    return normalized


# ============================================================================
# Tests
# ============================================================================

RESULTS = []

# Shared config
EMBED = 256
HEADS = 8
LEVELS = 1
POINTS = 4
FFN_DIM = 1024
NUM_LAYERS = 2
BS = 2
NQ = 50
NPT = 20
H, W = 16, 16
NV = H * W

# Fixed seed
np.random.seed(42)
torch.manual_seed(42)

# Shared random inputs  (seq-first: [nq, bs, C])
query_np = np.random.randn(NQ, BS, EMBED).astype(np.float32) * 0.1
query_bf_np = query_np.transpose(1, 0, 2)  # [bs, nq, C]
value_np = np.random.randn(NV, BS, EMBED).astype(np.float32) * 0.1
value_bf_np = value_np.transpose(1, 0, 2)  # [bs, nv, C]
ref_pts_np = np.random.rand(BS, NQ, NPT, 2).astype(np.float32)
query_pos_np = np.random.randn(NQ, BS, EMBED).astype(np.float32) * 0.02

spatial_shapes = [(H, W)]
spatial_np = np.array(spatial_shapes, dtype=np.int64)


def run_test(name, fn):
    print(f"\nTEST: {name}")
    print("-" * 70)
    try:
        ok = fn()
        RESULTS.append((name, ok))
        print(f"\n{'PASSED' if ok else 'FAILED'}")
    except Exception:
        traceback.print_exc()
        RESULTS.append((name, False))
        print("\nFAILED (exception)")


# ------------------------------------------------------------------
# TEST 1: Standalone Deformable Attention
# ------------------------------------------------------------------
def test_deformable_attention():
    """Compare CustomMSDeformableAttention output: PyTorch vs TTSim."""
    pt = CustomMSDeformableAttentionPyTorch(EMBED, HEADS, LEVELS, POINTS)
    pt.eval()

    tt = CustomMSDeformableAttention(
        name="t1_dattn",
        embed_dims=EMBED,
        num_heads=HEADS,
        num_levels=LEVELS,
        num_points=POINTS,
        batch_first=True,
        dropout=0.0,
    )
    transfer_deform_attn_weights(pt, tt)

    with torch.no_grad():
        out_pt = pt(
            torch.from_numpy(query_bf_np),
            torch.from_numpy(value_bf_np),
            torch.from_numpy(ref_pts_np),
            spatial_shapes,
        ).numpy()

    q_s = F._from_data("t1_q", query_bf_np, is_const=False)
    v_s = F._from_data("t1_v", value_bf_np, is_const=False)
    r_s = F._from_data("t1_r", ref_pts_np, is_const=False)
    ss = F._from_data("t1_ss", spatial_np, is_const=True)
    out_tt = tt(q_s, value=v_s, reference_points=r_s, spatial_shapes=ss).data

    _stats("Pytorch", out_pt)
    _stats("TTSim ", out_tt)
    return _check("deformable attention", out_pt, out_tt, atol=1e-5)


# ------------------------------------------------------------------
# TEST 2: Standalone Self-Attention (MHA)
# ------------------------------------------------------------------
def test_self_attention():
    """Compare nn.MultiheadAttention vs TTSim MultiheadAttention."""
    pt_mha = nn.MultiheadAttention(EMBED, HEADS, dropout=0.0, batch_first=False)
    pt_mha.eval()

    tt_mha = MultiheadAttentionTTSim(
        name="t2_mha", embed_dims=EMBED, num_heads=HEADS, batch_first=False
    )
    transfer_mha_weights(pt_mha, tt_mha)

    q_t = torch.from_numpy(query_np)
    pos_t = torch.from_numpy(query_pos_np)

    # PyTorch: add pos to q and k, then attend
    with torch.no_grad():
        q_plus = q_t + pos_t
        out_pt, _ = pt_mha(q_plus, q_plus, q_t)
        out_pt_np = out_pt.numpy()

    # TTSim: query_pos/key_pos handled internally
    q_s = F._from_data("t2_q", query_np, is_const=False)
    pos_s = F._from_data("t2_pos", query_pos_np, is_const=False)
    out_tt = tt_mha(q_s, key=q_s, value=q_s, query_pos=pos_s, key_pos=pos_s).data

    _stats("Pytorch", out_pt_np)
    _stats("TTSim ", out_tt)
    return _check("self-attention (MHA)", out_pt_np, out_tt, atol=1e-5)


# ------------------------------------------------------------------
# TEST 3: Standalone FFN
# ------------------------------------------------------------------
def test_ffn():
    """Compare PyTorch FFN (2-layer + ReLU + residual) vs TTSim FFN."""
    pt_fc1 = nn.Linear(EMBED, FFN_DIM)
    pt_fc2 = nn.Linear(FFN_DIM, EMBED)

    tt_ffn = FFNTTSim(
        name="t3_ffn",
        embed_dims=EMBED,
        feedforward_channels=FFN_DIM,
        num_fcs=2,
        add_identity=True,
    )
    transfer_ffn_weights(pt_fc1, pt_fc2, tt_ffn)

    x_np = np.random.randn(NQ, BS, EMBED).astype(np.float32) * 0.1
    x_t = torch.from_numpy(x_np)

    with torch.no_grad():
        ffn_out = pt_fc2(torch.relu(pt_fc1(x_t)))
        out_pt = (x_t + ffn_out).numpy()  # residual

    x_s = F._from_data("t3_x", x_np, is_const=False)
    out_tt = tt_ffn(x_s).data

    _stats("Pytorch", out_pt)
    _stats("TTSim ", out_tt)
    return _check("FFN (2-layer + residual)", out_pt, out_tt, atol=1e-5)


# ------------------------------------------------------------------
# TEST 4: Single Layer through MapTransformerLayer.__call__
# ------------------------------------------------------------------
def test_single_layer_via_call():
    """Full transformer layer via __call__: self_attn + cross_attn + FFN + norms."""
    # Patch LayerNorm for affine support
    LayerNorm.__call__ = _patched_norm_call

    pt_layer = MapTransformerLayerPyTorch(EMBED, HEADS, LEVELS, POINTS, FFN_DIM)
    pt_layer.eval()

    tt_layer = build_ttsim_layer(0, EMBED, HEADS, LEVELS, POINTS, FFN_DIM)
    transfer_layer_weights(pt_layer, tt_layer, 0)

    q_t = torch.from_numpy(query_np)
    v_t = torch.from_numpy(value_np)
    pos_t = torch.from_numpy(query_pos_np)
    r_t = torch.from_numpy(ref_pts_np)

    with torch.no_grad():
        out_pt = pt_layer(
            q_t,
            v_t,
            v_t,
            query_pos=pos_t,
            reference_points=r_t,
            spatial_shapes=spatial_shapes,
        ).numpy()

    q_s = F._from_data("t4_q", query_np, is_const=False)
    v_s = F._from_data("t4_v", value_np, is_const=False)
    pos_s = F._from_data("t4_pos", query_pos_np, is_const=False)
    r_s = F._from_data("t4_r", ref_pts_np, is_const=False)
    ss = F._from_data("t4_ss", spatial_np, is_const=True)

    out_tt = tt_layer(
        q_s,
        key=v_s,
        value=v_s,
        query_pos=pos_s,
        reference_points=r_s,
        spatial_shapes=ss,
    ).data

    _stats("Pytorch", out_pt)
    _stats("TTSim ", out_tt)

    # Restore original
    LayerNorm.__call__ = _original_norm_call
    return _check("MapTransformerLayer.__call__", out_pt, out_tt, atol=1e-4)


# ------------------------------------------------------------------
# TEST 5: Full 2-Layer Decoder through MapTransformerDecoder_new.__call__
# ------------------------------------------------------------------
def test_full_decoder_via_call():
    """2-layer decoder via __call__, no reg branches."""
    LayerNorm.__call__ = _patched_norm_call

    pt_decoder = MapTransformerDecoderPyTorch(
        NUM_LAYERS, EMBED, HEADS, LEVELS, POINTS, FFN_DIM
    )
    pt_decoder.eval()

    tt_decoder = MapTransformerDecoder_new(
        num_layers=NUM_LAYERS, return_intermediate=True, prop_add_stage=0
    )
    for lid in range(NUM_LAYERS):
        tt_layer = build_ttsim_layer(lid, EMBED, HEADS, LEVELS, POINTS, FFN_DIM)
        transfer_layer_weights(pt_decoder.layers[lid], tt_layer, lid)
        tt_decoder.add_layer(tt_layer)

    q_t = torch.from_numpy(query_np)
    v_t = torch.from_numpy(value_np)
    pos_t = torch.from_numpy(query_pos_np)
    r_t = torch.from_numpy(ref_pts_np)

    with torch.no_grad():
        intermediates_pt, _ = pt_decoder(
            q_t,
            v_t,
            v_t,
            pos_t,
            r_t,
            spatial_shapes,
            reg_branches=None,
            predict_refine=False,
        )

    q_s = F._from_data("t5_q", query_np, is_const=False)
    v_s = F._from_data("t5_v", value_np, is_const=False)
    pos_s = F._from_data("t5_pos", query_pos_np, is_const=False)
    r_s = F._from_data("t5_r", ref_pts_np, is_const=False)
    ss = F._from_data("t5_ss", spatial_np, is_const=True)
    lsi = F._from_data("t5_lsi", np.array([0], dtype=np.int64), is_const=True)

    intermediates_tt, _ = tt_decoder(
        q_s,
        key=v_s,
        value=v_s,
        query_pos=pos_s,
        key_padding_mask=None,
        query_key_padding_mask=None,
        reference_points=r_s,
        spatial_shapes=ss,
        level_start_index=lsi,
        reg_branches=None,
        cls_branches=None,
        predict_refine=False,
    )

    ok = True
    for lid in range(NUM_LAYERS):
        out_pt = intermediates_pt[lid].numpy()
        out_tt = intermediates_tt[lid].data
        _stats(f"  Pytorch L{lid}", out_pt)
        _stats(f"  TTSim  L{lid}", out_tt)
        ok = ok and _check(f"decoder layer {lid}", out_pt, out_tt, atol=1e-4)

    LayerNorm.__call__ = _original_norm_call
    return ok


# ------------------------------------------------------------------
# TEST 6: Decoder with regression branch refinement
# ------------------------------------------------------------------
def test_decoder_with_reg_branches():
    """Decoder + reg branches (predict_refine=False: sigmoid(offset))."""
    LayerNorm.__call__ = _patched_norm_call

    pt_decoder = MapTransformerDecoderPyTorch(
        NUM_LAYERS, EMBED, HEADS, LEVELS, POINTS, FFN_DIM
    )
    pt_decoder.eval()

    pt_reg_branches = nn.ModuleList(
        [nn.Sequential(nn.Linear(EMBED, NPT * 2)) for _ in range(NUM_LAYERS)]
    )
    pt_reg_branches.eval()

    tt_decoder = MapTransformerDecoder_new(
        num_layers=NUM_LAYERS, return_intermediate=True, prop_add_stage=0
    )
    tt_reg_branches = []
    for lid in range(NUM_LAYERS):
        tt_layer = build_ttsim_layer(lid, EMBED, HEADS, LEVELS, POINTS, FFN_DIM)
        transfer_layer_weights(pt_decoder.layers[lid], tt_layer, lid)
        tt_decoder.add_layer(tt_layer)

        # TTSim reg branch (single Linear)
        reg = SimNN.Linear(f"reg_l{lid}", in_features=EMBED, out_features=NPT * 2)
        pt_lin = pt_reg_branches[lid][0]
        reg.param.data = pt_lin.weight.data.numpy()
        reg.bias.data = pt_lin.bias.data.numpy()
        tt_reg_branches.append(reg)

    q_t = torch.from_numpy(query_np)
    v_t = torch.from_numpy(value_np)
    pos_t = torch.from_numpy(query_pos_np)
    r_t = torch.from_numpy(ref_pts_np)

    with torch.no_grad():
        intermediates_pt, ref_pts_pt = pt_decoder(
            q_t,
            v_t,
            v_t,
            pos_t,
            r_t,
            spatial_shapes,
            reg_branches=pt_reg_branches,
            predict_refine=False,
        )

    q_s = F._from_data("t6_q", query_np, is_const=False)
    v_s = F._from_data("t6_v", value_np, is_const=False)
    pos_s = F._from_data("t6_pos", query_pos_np, is_const=False)
    r_s = F._from_data("t6_r", ref_pts_np, is_const=False)
    ss = F._from_data("t6_ss", spatial_np, is_const=True)
    lsi = F._from_data("t6_lsi", np.array([0], dtype=np.int64), is_const=True)

    intermediates_tt, ref_pts_tt = tt_decoder(
        q_s,
        key=v_s,
        value=v_s,
        query_pos=pos_s,
        key_padding_mask=None,
        query_key_padding_mask=None,
        reference_points=r_s,
        spatial_shapes=ss,
        level_start_index=lsi,
        reg_branches=tt_reg_branches,
        cls_branches=None,
        predict_refine=False,
    )

    ok = True
    for lid in range(NUM_LAYERS):
        out_pt = intermediates_pt[lid].numpy()
        out_tt = intermediates_tt[lid].data
        ok = ok and _check(f"decoder+reg layer {lid} output", out_pt, out_tt, atol=1e-4)

    ref_pt_final = ref_pts_pt[-1].numpy()
    ref_tt_final = ref_pts_tt[-1].data
    _stats("  Pytorch ref_pts", ref_pt_final)
    _stats("  TTSim  ref_pts", ref_tt_final)
    ok = ok and _check("final reference points", ref_pt_final, ref_tt_final, atol=1e-5)

    LayerNorm.__call__ = _original_norm_call
    return ok


# ------------------------------------------------------------------
# TEST 7: Varied inputs (edge refs, small dims)
# ------------------------------------------------------------------
def test_varied_inputs():
    """Deformable attention with edge reference points and different dimensions."""
    np.random.seed(99)
    torch.manual_seed(99)

    bs, nq, npt = 1, 20, 10
    embed = 128
    heads = 4

    q_np_v = np.random.randn(bs, nq, embed).astype(np.float32) * 0.1
    v_np_v = np.random.randn(bs, NV, embed).astype(np.float32) * 0.1
    r_np_v = np.clip(
        np.random.randn(bs, nq, npt, 2).astype(np.float32) * 0.3 + 0.5, 0.01, 0.99
    )

    pt_attn = CustomMSDeformableAttentionPyTorch(embed, heads, LEVELS, POINTS)
    pt_attn.eval()

    tt_attn = CustomMSDeformableAttention(
        name="t7_dattn",
        embed_dims=embed,
        num_heads=heads,
        num_levels=LEVELS,
        num_points=POINTS,
        batch_first=True,
        dropout=0.0,
    )
    transfer_deform_attn_weights(pt_attn, tt_attn)

    with torch.no_grad():
        out_pt = pt_attn(
            torch.from_numpy(q_np_v),
            torch.from_numpy(v_np_v),
            torch.from_numpy(r_np_v),
            spatial_shapes,
        ).numpy()

    q_s = F._from_data("t7_q", q_np_v, is_const=False)
    v_s = F._from_data("t7_v", v_np_v, is_const=False)
    r_s = F._from_data("t7_r", r_np_v, is_const=False)
    ss = F._from_data("t7_ss", spatial_np, is_const=True)
    out_tt = tt_attn(q_s, value=v_s, reference_points=r_s, spatial_shapes=ss).data

    _stats("Pytorch", out_pt)
    _stats("TTSim ", out_tt)
    return _check("varied inputs (edge refs, small dims)", out_pt, out_tt, atol=1e-5)


# ------------------------------------------------------------------
# TEST 8: Weight transfer fidelity
# ------------------------------------------------------------------
def test_weight_transfer():
    """Verify Linear projections produce identical outputs after transfer."""
    # Deformable attention projections
    pt = CustomMSDeformableAttentionPyTorch(EMBED, HEADS, LEVELS, POINTS)
    pt.eval()

    tt = CustomMSDeformableAttention(
        name="t8_dattn",
        embed_dims=EMBED,
        num_heads=HEADS,
        num_levels=LEVELS,
        num_points=POINTS,
        batch_first=True,
    )
    transfer_deform_attn_weights(pt, tt)

    x_np = np.random.randn(BS, NQ, EMBED).astype(np.float32) * 0.1
    x_t = torch.from_numpy(x_np)
    x_s = F._from_data("t8_x", x_np, is_const=False)

    ok = True
    for proj_name in [
        "value_proj",
        "output_proj",
        "sampling_offsets",
        "attention_weights",
    ]:
        with torch.no_grad():
            pt_out = getattr(pt, proj_name)(x_t).numpy()
        tt_out = getattr(tt, proj_name)(x_s).data
        ok = ok and _check(proj_name, pt_out, tt_out, atol=1e-6)

    # MHA weight transfer
    pt_mha = nn.MultiheadAttention(EMBED, HEADS, dropout=0.0, batch_first=False)
    pt_mha.eval()
    tt_mha = MultiheadAttentionTTSim(
        name="t8_mha", embed_dims=EMBED, num_heads=HEADS, batch_first=False
    )
    transfer_mha_weights(pt_mha, tt_mha)

    x_seq_np = np.random.randn(NQ, BS, EMBED).astype(np.float32) * 0.1
    x_seq_t = torch.from_numpy(x_seq_np)
    E = EMBED

    # Compare Q projection
    with torch.no_grad():
        w_q = pt_mha.in_proj_weight[:E]
        b_q = pt_mha.in_proj_bias[:E]
        pt_q = F_torch.linear(x_seq_t, w_q, b_q).numpy()

    # TTSim MHA transposes for batch_first=False internally, so feed batch-first
    x_bf = x_seq_np.transpose(1, 0, 2)  # [bs, nq, C]
    x_bf_s = F._from_data("t8_mha_x", x_bf, is_const=False)
    tt_q_bf = tt_mha.q_proj(x_bf_s).data  # [bs, nq, C]
    tt_q = tt_q_bf.transpose(1, 0, 2)  # [nq, bs, C]
    ok = ok and _check("MHA q_proj", pt_q, tt_q, atol=1e-6)

    # FFN weight transfer
    pt_fc1 = nn.Linear(EMBED, FFN_DIM)
    pt_fc2 = nn.Linear(FFN_DIM, EMBED)
    tt_ffn = FFNTTSim(
        name="t8_ffn",
        embed_dims=EMBED,
        feedforward_channels=FFN_DIM,
        num_fcs=2,
        add_identity=False,
    )
    transfer_ffn_weights(pt_fc1, pt_fc2, tt_ffn)

    with torch.no_grad():
        pt_ffn_out = pt_fc2(torch.relu(pt_fc1(x_seq_t))).numpy()
    x_seq_s = F._from_data("t8_ffn_x", x_seq_np, is_const=False)
    tt_ffn_out = tt_ffn(x_seq_s).data
    ok = ok and _check("FFN (no residual)", pt_ffn_out, tt_ffn_out, atol=1e-5)

    return ok


# ------------------------------------------------------------------
# TEST 9: Memory Bank query_bev + query_memory Fusion
# ------------------------------------------------------------------


class MockMemoryBank:
    """Lightweight mock memory bank for testing the dual cross_attn path.

    Provides the three data dicts and valid_track_idx that
    MapTransformerLayer.__call__ reads when memory_bank is not None.
    """

    def __init__(self, bs, num_queries, embed_dims, num_tracks, mem_len=3):
        self.valid_track_idx = []
        self.batch_mem_embeds_dict = {}
        self.batch_key_padding_dict = {}
        self.batch_mem_relative_pe_dict = {}

        for b_i in range(bs):
            # Track indices: first `num_tracks` queries have memory
            track_idx = list(range(num_tracks))
            self.valid_track_idx.append(track_idx)

            # Memory embeddings: [mem_len, num_tracks, embed_dims]
            self.batch_mem_embeds_dict[b_i] = np.random.randn(
                mem_len, num_tracks, embed_dims
            ).astype(np.float32)

            # Key padding mask: [num_tracks, mem_len] (False = attend)
            self.batch_key_padding_dict[b_i] = np.zeros(
                (num_tracks, mem_len), dtype=np.float32
            )

            # Relative PE: [mem_len, num_tracks, embed_dims]
            # Note: In production, this is a graph-connected SimTensor built via
            # F.Gather(pe_table) -> F.Unsqueeze -> F.ConcatX(zeros) -> F.Transpose.
            # Here we use plain numpy for the numerical validation (the MapTransformer
            # code falls back to F._from_data when the value is not a SimTensor).
            self.batch_mem_relative_pe_dict[b_i] = (
                np.random.randn(mem_len, num_tracks, embed_dims).astype(np.float32)
                * 0.02
            )


class MapTransformerLayerPyTorchWithMemory(nn.Module):
    """PyTorch reference with the CORRECT query_bev + query_memory fusion.

    Operation order: self_attn -> norm -> cross_attn(BEV) -> norm
                      -> cross_attn(memory) -> norm -> ffn -> norm
    """

    def __init__(
        self, embed_dims=256, num_heads=8, num_levels=1, num_points=4, ffn_dim=1024
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.self_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0.0, batch_first=False
        )
        self.cross_attn = CustomMSDeformableAttentionPyTorch(
            embed_dims, num_heads, num_levels, num_points
        )
        self.mem_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=0.0, batch_first=True
        )
        self.ffn_fc1 = nn.Linear(embed_dims, ffn_dim)
        self.ffn_fc2 = nn.Linear(ffn_dim, embed_dims)
        self.norm0 = nn.LayerNorm(embed_dims)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        memory_bank=None,
        **kwargs,
    ):
        # --- self-attention ---
        q_sa = query + query_pos if query_pos is not None else query
        k_sa = q_sa
        sa_out, _ = self.self_attn(q_sa, k_sa, query)
        query = query + sa_out
        query = self.norm0(query)

        # --- BEV cross-attention (store result as query_bev) ---
        query_with_pos = query + query_pos if query_pos is not None else query
        query_bf = query_with_pos.permute(1, 0, 2)
        key_bf = key.permute(1, 0, 2)
        ca_out = self.cross_attn(query_bf, key_bf, reference_points, spatial_shapes)
        query_bev = query + ca_out.permute(1, 0, 2)  # residual (NOT normed yet)

        # norm1 is applied to query (not query_bev) — matching the actual
        # PyTorch source where the 'norm' op between cross_attns normalizes
        # the unchanged `query`, not the newly-computed `query_bev`.
        query = self.norm1(query)

        # --- memory cross-attention (uses normed query, not query_bev) ---
        if memory_bank is not None:
            bs = query.shape[1]
            query_i_list = []
            for b_i in range(bs):
                valid_idx = memory_bank.valid_track_idx[b_i]
                query_i = query[:, b_i].clone().unsqueeze(0)  # [1, nq, C]
                if len(valid_idx) != 0:
                    mem_e = torch.from_numpy(
                        memory_bank.batch_mem_embeds_dict[b_i][:, valid_idx, :]
                    )
                    mem_kp = torch.from_numpy(
                        memory_bank.batch_key_padding_dict[b_i][valid_idx]
                    )
                    mem_pe = torch.from_numpy(
                        memory_bank.batch_mem_relative_pe_dict[b_i][:, valid_idx]
                    )
                    q_track = query_i[:, valid_idx]  # [1, n_tracks, C]
                    # mem_e is [mem_len, n_tracks, C] -> transpose for batch_first
                    mem_e_bf = mem_e.permute(1, 0, 2)  # [n_tracks, mem_len, C]
                    mem_pe_bf = mem_pe.permute(1, 0, 2)  # [n_tracks, mem_len, C]
                    # Use simple per-track attention (as the real code does per-query)
                    q_track_sq = q_track.squeeze(0)  # [n_tracks, C]
                    out_track, _ = self.mem_attn(
                        q_track_sq.unsqueeze(1),  # [n_tracks, 1, C]
                        mem_e_bf + mem_pe_bf,  # [n_tracks, mem_len, C]
                        mem_e_bf,
                    )
                    query_i[:, valid_idx] = out_track.squeeze(1)
                query_i_list.append(query_i.squeeze(0))  # [nq, C]
            query_memory = torch.stack(query_i_list, dim=1)  # [nq, bs, C]
        else:
            query_memory = torch.zeros_like(query_bev)

        # CORRECT FUSION: additive combination
        query = query_memory + query_bev
        query = self.norm2(query)

        # --- FFN ---
        ffn_out = self.ffn_fc2(torch.relu(self.ffn_fc1(query)))
        query = query + ffn_out
        query = self.norm3(query)
        return query


def transfer_layer_weights_with_memory(pt_layer, tt_layer, lid):
    """Weight transfer for the memory-aware layer variant."""
    transfer_mha_weights(pt_layer.self_attn, tt_layer.attentions[0])
    transfer_deform_attn_weights(pt_layer.cross_attn, tt_layer.attentions[1])
    transfer_mha_weights(pt_layer.mem_attn, tt_layer.attentions[2])
    transfer_ffn_weights(pt_layer.ffn_fc1, pt_layer.ffn_fc2, tt_layer.ffns[0])

    pt_norms = [pt_layer.norm0, pt_layer.norm1, pt_layer.norm2, pt_layer.norm3]
    for ni, pt_norm in enumerate(pt_norms):
        tt_layer.norms[ni]._affine_weight = F._from_data(
            f"ln_w_l{lid}_n{ni}", pt_norm.weight.data.numpy().copy(), is_const=True
        )
        tt_layer.norms[ni]._affine_bias = F._from_data(
            f"ln_b_l{lid}_n{ni}", pt_norm.bias.data.numpy().copy(), is_const=True
        )


def test_memory_bank_fusion():
    """Test that query_bev + query_memory are additively combined.

    This exercises the dual cross_attn code path with a mock memory bank.
    The PyTorch reference correctly stores BEV cross-attn as query_bev and
    combines it with memory attention via addition:
        query = query_memory + query_bev

    The TTSim implementation now matches this behavior. Previously it
    overwrote query with query_memory alone, discarding the BEV
    cross-attention result on every frame after the first.

    """
    LayerNorm.__call__ = _patched_norm_call

    num_tracks = 10  # First 10 queries have temporal memory

    # Build PyTorch reference with memory
    pt_layer = MapTransformerLayerPyTorchWithMemory(
        EMBED, HEADS, LEVELS, POINTS, FFN_DIM
    )
    pt_layer.eval()

    # Build TTSim layer (same as normal, 3 attentions)
    tt_layer = build_ttsim_layer(0, EMBED, HEADS, LEVELS, POINTS, FFN_DIM)
    transfer_layer_weights_with_memory(pt_layer, tt_layer, 0)

    mock_mb = MockMemoryBank(BS, NQ, EMBED, num_tracks, mem_len=3)

    # PyTorch forward
    q_t = torch.from_numpy(query_np)
    v_t = torch.from_numpy(value_np)
    pos_t = torch.from_numpy(query_pos_np)
    r_t = torch.from_numpy(ref_pts_np)
    with torch.no_grad():
        out_pt = pt_layer(
            q_t,
            v_t,
            v_t,
            query_pos=pos_t,
            reference_points=r_t,
            spatial_shapes=spatial_shapes,
            memory_bank=mock_mb,
        ).numpy()

    # TTSim forward
    q_s = F._from_data("t9_q", query_np.copy(), is_const=False)
    v_s = F._from_data("t9_v", value_np.copy(), is_const=False)
    pos_s = F._from_data("t9_pos", query_pos_np.copy(), is_const=False)
    r_s = F._from_data("t9_r", ref_pts_np.copy(), is_const=False)
    ss = F._from_data("t9_ss", spatial_np, is_const=True)

    out_tt = tt_layer(
        q_s,
        key=v_s,
        value=v_s,
        query_pos=pos_s,
        reference_points=r_s,
        spatial_shapes=ss,
        memory_bank=mock_mb,
    ).data

    _stats("Pytorch (with memory)", out_pt)
    _stats("TTSim  (with memory)", out_tt)

    LayerNorm.__call__ = _original_norm_call

    ok = _check("memory bank query_bev+query_memory fusion", out_pt, out_tt, atol=1e-4)
    if not ok:
        # Even a rough shape/magnitude check helps diagnose
        print(
            "  NOTE: If this fails, the BEV cross-attn result is likely lost "
            "when memory_bank is active."
        )
    return ok


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MapTransformer Numerical Comparison Suite (PyTorch vs TTSim)")
    print("=" * 80)
    print(
        f"\n  embed_dims={EMBED}  num_heads={HEADS}  num_levels={LEVELS}  "
        f"num_points={POINTS}"
    )
    print(
        f"  ffn_dim={FFN_DIM}  num_layers={NUM_LAYERS}  batch_size={BS}  "
        f"num_queries={NQ}  num_pts_per_query={NPT}"
    )
    print(f"  BEV feature map: {H}x{W}")

    run_test("Deformable attention standalone", test_deformable_attention)
    run_test("Self-attention (MHA) standalone", test_self_attention)
    run_test("FFN standalone", test_ffn)
    run_test(
        "Single layer via MapTransformerLayer.__call__", test_single_layer_via_call
    )
    run_test(
        "Full 2-layer decoder via MapTransformerDecoder_new.__call__",
        test_full_decoder_via_call,
    )
    run_test("Decoder with regression branches", test_decoder_with_reg_branches)
    run_test("Varied inputs (edge refs, small dims)", test_varied_inputs)
    run_test("Weight transfer fidelity", test_weight_transfer)
    run_test("Memory bank query_bev + query_memory fusion", test_memory_bank_fusion)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, ok in RESULTS:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name}")
    print()

    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_total = len(RESULTS)
    if n_pass == n_total:
        print(f"All {n_total} tests passed.")
    else:
        print(f"{n_pass}/{n_total} tests passed.")
        sys.exit(1)
