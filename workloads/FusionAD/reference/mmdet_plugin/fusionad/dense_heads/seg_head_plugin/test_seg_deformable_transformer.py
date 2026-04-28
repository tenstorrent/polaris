#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comparison test: SegDeformableTransformer (TTSim vs PyTorch).

Tests the full SegDeformableTransformer with shared weights between
a PyTorch reference and the TTSim implementation.  The PyTorch reference
classes are defined inline (no mmcv/mmdet dependency).

Config (from fusion_base_e2e.py, reduced for speed):
  Encoder: 2 layers, MSDA self-attn, feedforward_channels=128
  Decoder: 2 layers, MHA self-attn + MSDA cross-attn, return_intermediate=True
  embed_dims=32, nhead=4, num_feature_levels=2
"""

import copy
import math
import os
import sys
import traceback

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

np.random.seed(42)
torch.manual_seed(42)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# TTSim module under test
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.seg_head_plugin.seg_deformable_transformer import (
    SegDeformableTransformer as TT_SegDeformableTransformer,
)


# ====================================================================
# Test constants (reduced for speed)
# ====================================================================
DIM = 32
NHEAD = 4
HEAD_DIM = DIM // NHEAD
NUM_LEVELS = 2
NUM_POINTS = 4
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 2
FFN_CHANNELS = 128
B = 1
NUM_QUERY = 6
# spatial shapes per level
H0, W0 = 4, 4
H1, W1 = 2, 2


# ====================================================================
# PyTorch reference: multi_scale_deformable_attn_pytorch
# ====================================================================

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations,
                                        attention_weights):
    """CPU fallback of MSDA (from mmcv)."""
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split(
        [int(H_ * W_) for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        H_, W_ = int(H_), int(W_)
        value_l_ = (value_list[level].flatten(2).transpose(1, 2)
                     .reshape(bs * num_heads, embed_dims, H_, W_))
        sampling_grid_l_ = (sampling_grids[:, :, :, level]
                            .transpose(1, 2).flatten(0, 1))
        sampling_value_l_ = TF.grid_sample(
            value_l_, sampling_grid_l_,
            mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    attention_weights = (attention_weights.transpose(1, 2)
                         .reshape(bs * num_heads, 1, num_queries,
                                  num_levels * num_points))
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2)
              * attention_weights).sum(-1).view(
                  bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


# ====================================================================
# PyTorch reference: PT_MultiScaleDeformableAttention
# ====================================================================

class PT_MultiScaleDeformableAttention(nn.Module):
    """Inline PyTorch MSDA (matches mmcv MultiScaleDeformableAttention)."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4,
                 num_points=4, batch_first=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # (S, B, D) -> (B, S, D)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads,
                           self.embed_dims // self.num_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels,
            self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # Reference points handling
        if isinstance(spatial_shapes, torch.Tensor):
            spatial_shapes_t = spatial_shapes
        else:
            spatial_shapes_t = torch.tensor(spatial_shapes, dtype=torch.long)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes_t[..., 1], spatial_shapes_t[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :])
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points
                * reference_points[:, :, None, :, None, 2:] * 0.5)

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes_t, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return identity + output


# ====================================================================
# PyTorch reference: PT_MultiheadAttention
# ====================================================================

class PT_MultiheadAttention(nn.Module):
    """Inline PyTorch MHA (matches mmcv MultiheadAttention wrapper)."""

    def __init__(self, embed_dims=256, num_heads=8, batch_first=False):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dims // num_heads

        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, key_pos=None, **kwargs):
        if key is None:
            key = query
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, nq, _ = query.shape
        _, nk, _ = key.shape

        q = self.q_proj(query).view(bs, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bs, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bs, nk, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bs, nq, self.embed_dims)
        out = self.out_proj(out)

        if not self.batch_first:
            out = out.permute(1, 0, 2)

        return identity + out


# ====================================================================
# PyTorch reference: PT_FFN
# ====================================================================

class PT_FFN(nn.Module):
    """Inline PyTorch FFN (matches mmcv FFN with num_fcs=2, ReLU)."""

    def __init__(self, embed_dims=256, feedforward_channels=1024,
                 add_identity=True):
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, feedforward_channels)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        if identity is None:
            identity = x
        out = self.fc2(torch.relu(self.fc1(x)))
        if self.add_identity:
            out = out + identity
        return out


# ====================================================================
# PyTorch reference: PT_BaseTransformerLayer
# ====================================================================

class PT_BaseTransformerLayer(nn.Module):
    """Inline PyTorch transformer layer (matches mmcv BaseTransformerLayer)."""

    def __init__(self, attentions, ffn, norms, operation_order):
        super().__init__()
        self.attentions = nn.ModuleList(attentions)
        self.ffns = nn.ModuleList([ffn])
        self.norms = nn.ModuleList(norms)
        self.operation_order = operation_order
        self.pre_norm = (len(operation_order) > 0
                         and operation_order[0] == 'norm')

    def forward(self, query, key=None, value=None, query_pos=None,
                key_pos=None, query_key_padding_mask=None,
                key_padding_mask=None, reference_points=None,
                spatial_shapes=None, level_start_index=None, **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        for op in self.operation_order:
            if op == 'self_attn':
                # self-attention: query attends to itself
                attn = self.attentions[attn_index]
                _id = identity if self.pre_norm else None
                if isinstance(attn, PT_MultiScaleDeformableAttention):
                    query = attn(
                        query, key=None, value=query, identity=_id,
                        query_pos=query_pos,
                        key_padding_mask=query_key_padding_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index)
                else:
                    query = attn(
                        query, key=query, value=query, identity=_id,
                        query_pos=query_pos, key_pos=query_pos)
                attn_index += 1
                identity = query
            elif op == 'cross_attn':
                attn = self.attentions[attn_index]
                _id = identity if self.pre_norm else None
                if isinstance(attn, PT_MultiScaleDeformableAttention):
                    query = attn(
                        query, key=None, value=value, identity=_id,
                        query_pos=query_pos,
                        key_padding_mask=key_padding_mask,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index)
                else:
                    query = attn(
                        query, key=value, value=value, identity=_id,
                        query_pos=query_pos, key_pos=None)
                attn_index += 1
                identity = query
            elif op == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif op == 'ffn':
                query = self.ffns[ffn_index](
                    query,
                    identity=identity if self.pre_norm else None)
                ffn_index += 1
                identity = query

        return query


# ====================================================================
# PyTorch reference: PT_SegDeformableTransformer
# ====================================================================

class PT_SegDeformableTransformer(nn.Module):
    """Inline PyTorch SegDeformableTransformer (as_two_stage=False,
    with_box_refine=True).
    """

    def __init__(self, encoder_layers, decoder_layers, embed_dims=256,
                 num_feature_levels=4, return_intermediate=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.return_intermediate = return_intermediate

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.encoder_post_norm = nn.LayerNorm(embed_dims)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.level_embeds = nn.Parameter(
            torch.zeros(num_feature_levels, embed_dims))
        self.reference_points_fc = nn.Linear(embed_dims, 2)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32,
                               device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32,
                               device=device),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = (reference_points[:, :, None]
                            * valid_ratios[:, None])
        return reference_points

    @staticmethod
    def inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds,
                reg_branches=None, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        spatial_shapes_t = torch.tensor(spatial_shapes, dtype=torch.long)
        level_start_index = torch.cat([
            spatial_shapes_t.new_zeros((1,)),
            spatial_shapes_t.prod(1).cumsum(0)[:-1]])
        valid_ratios = torch.ones(bs, self.num_feature_levels, 2)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device)

        # Encoder (S, B, D) format
        feat_flatten = feat_flatten.permute(1, 0, 2)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)

        memory = feat_flatten
        for layer in self.encoder_layers:
            memory = layer(
                memory, key=None, value=None,
                query_pos=lvl_pos_embed_flatten,
                query_key_padding_mask=mask_flatten,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)
        memory = self.encoder_post_norm(memory)

        # Back to (B, S, D)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        # Query split (as_two_stage=False)
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        ref_pts = torch.sigmoid(self.reference_points_fc(query_pos))
        init_reference_out = ref_pts

        # Decoder (S, B, D) format
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        output = query
        reference_points = ref_pts
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.decoder_layers):
            rp_dim = reference_points.shape[-1]
            if rp_dim == 2:
                ref_pts_input = (
                    reference_points[:, :, None, :]
                    * valid_ratios[:, None])
            else:
                ref_pts_input = (
                    reference_points[:, :, None, :]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None])

            output = layer(
                output, key=None, value=memory,
                query_pos=query_pos,
                key_padding_mask=mask_flatten,
                reference_points=ref_pts_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index)

            # reg_branches refinement
            if reg_branches is not None:
                tmp = reg_branches[lid](output.permute(1, 0, 2))
                if rp_dim == 4:
                    new_ref = torch.sigmoid(
                        tmp + self.inverse_sigmoid(reference_points))
                else:
                    new_xy = tmp[..., :2] + self.inverse_sigmoid(
                        reference_points)
                    new_ref = torch.sigmoid(
                        torch.cat([new_xy, tmp[..., 2:]], -1))
                reference_points = new_ref.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        mem_tuple = (memory, lvl_pos_embed_flatten, mask_flatten, query_pos)
        return (mem_tuple, intermediate, init_reference_out,
                intermediate_reference_points, None, None)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear(pt_linear, tt_linear):
    """Copy nn.Linear weights to SimNN.Linear."""
    tt_linear.param.data = (pt_linear.weight.data.detach().numpy()
                            .astype(np.float32))
    tt_linear.bias.data = (pt_linear.bias.data.detach().numpy()
                           .astype(np.float32))


def copy_f_layernorm(pt_ln, tt_fln):
    """Copy nn.LayerNorm weights to F.LayerNorm (which has affine params)."""
    tt_fln.params[0][1].data = (pt_ln.weight.data.detach().numpy()
                                .astype(np.float32))
    tt_fln.params[1][1].data = (pt_ln.bias.data.detach().numpy()
                                .astype(np.float32))


def copy_msda(pt_msda, tt_msda):
    """Copy MSDA weights (4 linear layers)."""
    copy_linear(pt_msda.sampling_offsets, tt_msda.sampling_offsets)
    copy_linear(pt_msda.attention_weights, tt_msda.attention_weights)
    copy_linear(pt_msda.value_proj, tt_msda.value_proj)
    copy_linear(pt_msda.output_proj, tt_msda.output_proj)


def copy_mha(pt_mha, tt_mha):
    """Copy MHA weights (4 linear layers — TTSim has separate q/k/v)."""
    copy_linear(pt_mha.q_proj, tt_mha.q_proj)
    copy_linear(pt_mha.k_proj, tt_mha.k_proj)
    copy_linear(pt_mha.v_proj, tt_mha.v_proj)
    copy_linear(pt_mha.out_proj, tt_mha.out_proj)


def copy_ffn(pt_ffn, tt_ffn):
    """Copy FFN weights (2 linear layers)."""
    copy_linear(pt_ffn.fc1, tt_ffn.layers[0])
    copy_linear(pt_ffn.fc2, tt_ffn.layers[1])


def copy_encoder_layer(pt_layer, tt_layer):
    """Copy encoder layer: 1 MSDA + 1 FFN + 2 norms.
    operation_order = ('self_attn', 'norm', 'ffn', 'norm')
    TTSim builder_utils.LayerNorm has NO affine params, so PyTorch norms
    must be at default (weight=1, bias=0) => no norm copy needed.
    """
    copy_msda(pt_layer.attentions[0], tt_layer.attentions[0])
    copy_ffn(pt_layer.ffns[0], tt_layer.ffns[0])
    # norms: builder_utils.LayerNorm has no affine, so skip


def copy_decoder_layer(pt_layer, tt_layer):
    """Copy decoder layer: 1 MHA + 1 MSDA + 1 FFN + 3 norms.
    operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    """
    copy_mha(pt_layer.attentions[0], tt_layer.attentions[0])
    copy_msda(pt_layer.attentions[1], tt_layer.attentions[1])
    copy_ffn(pt_layer.ffns[0], tt_layer.ffns[0])
    # norms: builder_utils.LayerNorm has no affine, so skip


def copy_reg_branch(pt_branch, tt_branch):
    """Copy a reg_branch: Sequential(Linear,ReLU,Linear,ReLU,Linear)."""
    # PT reg_branch: nn.Sequential with [Linear, ReLU, Linear, ReLU, Linear]
    # TT reg_branch: same structure
    pt_linears = [m for m in pt_branch.modules() if isinstance(m, nn.Linear)]
    tt_linears = [m for m in tt_branch._modules.values()
                  if isinstance(m, SimNN.Linear)]
    for pl, tl in zip(pt_linears, tt_linears):
        copy_linear(pl, tl)


def copy_transformer(pt_tr, tt_tr):
    """Copy full SegDeformableTransformer weights."""
    # Encoder layers
    for i in range(len(pt_tr.encoder_layers)):
        copy_encoder_layer(pt_tr.encoder_layers[i], tt_tr.encoder_layers[i])

    # Encoder post-norm (F.LayerNorm has affine params)
    copy_f_layernorm(pt_tr.encoder_post_norm, tt_tr.encoder_post_norm)

    # Decoder layers
    for i in range(len(pt_tr.decoder_layers)):
        copy_decoder_layer(pt_tr.decoder_layers[i], tt_tr.decoder_layers[i])

    # Level embeds: PT nn.Parameter [num_levels, C] -> TT list of F._from_data [1,1,C]
    for lvl in range(pt_tr.num_feature_levels):
        tt_tr.level_embeds[lvl].data = (
            pt_tr.level_embeds.data[lvl].detach().numpy()
            .reshape(1, 1, -1).astype(np.float32))

    # Reference points linear
    copy_linear(pt_tr.reference_points_fc, tt_tr.reference_points_fc)


# ====================================================================
# Compare helper
# ====================================================================

def compare(pt_out, tt_out, name, atol=1e-4):
    pt_np = (pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor)
             else pt_out)
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
    print(f"  {name}:")
    print(f"    PT shape: {pt_np.shape}  TT shape: {np.array(tt_np).shape}")
    if pt_np.shape != np.array(tt_np).shape:
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np - tt_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if max_diff < atol:
        print(f"    [OK] Match (atol={atol})")
        return True
    print(f"    [FAIL] Exceeds tolerance (atol={atol})")
    return False


# ====================================================================
# Build helpers
# ====================================================================

def build_pt_encoder_layer():
    """Build one PyTorch encoder layer (MSDA self-attn, FFN, 2 norms)."""
    msda = PT_MultiScaleDeformableAttention(
        embed_dims=DIM, num_heads=NHEAD, num_levels=NUM_LEVELS,
        num_points=NUM_POINTS, batch_first=False)
    ffn = PT_FFN(embed_dims=DIM, feedforward_channels=FFN_CHANNELS)
    norms = [nn.LayerNorm(DIM) for _ in range(2)]
    # Keep norms at weight=1, bias=0 (matches TTSim no-affine LayerNorm)
    for n in norms:
        nn.init.ones_(n.weight)
        nn.init.zeros_(n.bias)
    return PT_BaseTransformerLayer(
        attentions=[msda], ffn=ffn, norms=norms,
        operation_order=('self_attn', 'norm', 'ffn', 'norm'))


def build_pt_decoder_layer():
    """Build one PyTorch decoder layer (MHA self-attn, MSDA cross-attn,
    FFN, 3 norms)."""
    mha = PT_MultiheadAttention(
        embed_dims=DIM, num_heads=NHEAD, batch_first=False)
    msda = PT_MultiScaleDeformableAttention(
        embed_dims=DIM, num_heads=NHEAD, num_levels=NUM_LEVELS,
        num_points=NUM_POINTS, batch_first=False)
    ffn = PT_FFN(embed_dims=DIM, feedforward_channels=FFN_CHANNELS)
    norms = [nn.LayerNorm(DIM) for _ in range(3)]
    for n in norms:
        nn.init.ones_(n.weight)
        nn.init.zeros_(n.bias)
    return PT_BaseTransformerLayer(
        attentions=[mha, msda], ffn=ffn, norms=norms,
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm'))


def build_pt_reg_branches(num_layers, embed_dims=DIM):
    """Build reg_branches: list of Sequential(L,ReLU,L,ReLU,L)."""
    branches = nn.ModuleList()
    for _ in range(num_layers):
        branches.append(nn.Sequential(
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, 4)))
    return branches


# TTSim reg_branch builder
class TT_RegBranch(SimNN.Module):
    """Simple TTSim reg_branch: 3 linear layers with ReLU."""

    def __init__(self, name, embed_dims):
        super().__init__()
        self.name = name
        self.fc0 = SimNN.Linear(f'{name}.fc0', embed_dims, embed_dims)
        self.relu0 = F.Relu(f'{name}.relu0')
        self.fc1 = SimNN.Linear(f'{name}.fc1', embed_dims, embed_dims)
        self.relu1 = F.Relu(f'{name}.relu1')
        self.fc2 = SimNN.Linear(f'{name}.fc2', embed_dims, 4)
        super().link_op2module()

    def __call__(self, x):
        x = self.relu0(self.fc0(x))
        x = self.relu1(self.fc1(x))
        return self.fc2(x)


def copy_reg_branches(pt_branches, tt_branches):
    """Copy all reg_branch weights."""
    for pt_b, tt_b in zip(pt_branches, tt_branches):
        pt_linears = [m for m in pt_b.modules() if isinstance(m, nn.Linear)]
        tt_linears = [tt_b.fc0, tt_b.fc1, tt_b.fc2]
        for pl, tl in zip(pt_linears, tt_linears):
            copy_linear(pl, tl)


# ====================================================================
# TEST: Full SegDeformableTransformer
# ====================================================================

def test_seg_deformable_transformer():
    print("\n" + "=" * 70)
    print("TEST: SegDeformableTransformer (full forward)")
    print("=" * 70)

    # ---- Build PyTorch model ----
    enc_layers_pt = [build_pt_encoder_layer()
                     for _ in range(NUM_ENC_LAYERS)]
    dec_layers_pt = [build_pt_decoder_layer()
                     for _ in range(NUM_DEC_LAYERS)]
    pt = PT_SegDeformableTransformer(
        encoder_layers=enc_layers_pt,
        decoder_layers=dec_layers_pt,
        embed_dims=DIM,
        num_feature_levels=NUM_LEVELS,
        return_intermediate=True)
    pt.eval()

    pt_reg = build_pt_reg_branches(NUM_DEC_LAYERS, DIM)
    pt_reg.eval()

    # ---- Build TTSim model ----
    encoder_cfg = dict(
        type='DetrTransformerEncoder',
        num_layers=NUM_ENC_LAYERS,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=DIM,
                num_heads=NHEAD,
                num_levels=NUM_LEVELS,
                num_points=NUM_POINTS),
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=DIM,
                feedforward_channels=FFN_CHANNELS,
                ffn_drop=0.0),
            operation_order=('self_attn', 'norm', 'ffn', 'norm')))

    decoder_cfg = dict(
        type='DeformableDetrTransformerDecoder',
        num_layers=NUM_DEC_LAYERS,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=DIM, num_heads=NHEAD, dropout=0.0),
                dict(type='MultiScaleDeformableAttention',
                     embed_dims=DIM, num_heads=NHEAD,
                     num_levels=NUM_LEVELS,
                     num_points=NUM_POINTS),
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=DIM,
                feedforward_channels=FFN_CHANNELS,
                ffn_drop=0.0),
            operation_order=('self_attn', 'norm', 'cross_attn',
                             'norm', 'ffn', 'norm')))

    tt = TT_SegDeformableTransformer(
        name='sdt',
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        num_feature_levels=NUM_LEVELS,
        embed_dims=DIM)

    tt_reg = [TT_RegBranch(f'reg_{i}', DIM) for i in range(NUM_DEC_LAYERS)]

    # ---- Copy weights ----
    copy_transformer(pt, tt)
    copy_reg_branches(pt_reg, tt_reg)

    # ---- Create inputs ----
    feats_np = [
        np.random.randn(B, DIM, H0, W0).astype(np.float32),
        np.random.randn(B, DIM, H1, W1).astype(np.float32),
    ]
    masks_np = [
        np.zeros((B, H0, W0), dtype=np.float32),
        np.zeros((B, H1, W1), dtype=np.float32),
    ]
    pos_np = [
        np.random.randn(B, DIM, H0, W0).astype(np.float32),
        np.random.randn(B, DIM, H1, W1).astype(np.float32),
    ]
    query_embed_np = np.random.randn(NUM_QUERY, 2 * DIM).astype(np.float32)

    # PyTorch inputs
    pt_feats = [torch.from_numpy(f) for f in feats_np]
    pt_masks = [torch.from_numpy(m) for m in masks_np]
    pt_pos = [torch.from_numpy(p) for p in pos_np]
    pt_qe = torch.from_numpy(query_embed_np)

    # TTSim inputs
    tt_feats = [F._from_data(f'feat_{i}', f) for i, f in enumerate(feats_np)]
    tt_masks = [F._from_data(f'mask_{i}', m) for i, m in enumerate(masks_np)]
    tt_pos = [F._from_data(f'pos_{i}', p) for i, p in enumerate(pos_np)]
    tt_qe = F._from_data('qe', query_embed_np)

    # ---- Run PyTorch ----
    with torch.no_grad():
        pt_out = pt(pt_feats, pt_masks, pt_qe, pt_pos,
                    reg_branches=pt_reg)

    # ---- Run TTSim ----
    tt_out = tt(tt_feats, tt_masks, tt_qe, tt_pos,
                reg_branches=tt_reg)

    # ---- Compare outputs ----
    # pt_out = (mem_tuple, intermediate, init_ref, inter_refs, None, None)
    all_ok = True

    # Memory tuple [0]: (memory, lvl_pos, mask, query_pos)
    pt_mem, pt_lvl, pt_mf, pt_qpos = pt_out[0]
    tt_mem, tt_lvl, tt_mf, tt_qpos = tt_out[0]
    all_ok &= compare(pt_mem, tt_mem, "memory (S,B,D)", atol=1e-3)

    # init_reference_out [2]
    all_ok &= compare(pt_out[2], tt_out[2],
                       "init_reference_out (B,nq,2)", atol=1e-5)

    # inter_states [1] = list of decoder outputs per layer
    pt_inter = pt_out[1]
    tt_inter = tt_out[1]
    print(f"\n  Decoder intermediate states: {len(pt_inter)} layers")
    for i in range(len(pt_inter)):
        all_ok &= compare(pt_inter[i], tt_inter[i],
                           f"inter_state[{i}] (nq,B,D)", atol=1e-3)

    # inter_references [3] = list of refined references per layer
    pt_irefs = pt_out[3]
    tt_irefs = tt_out[3]
    print(f"\n  Decoder intermediate references: {len(pt_irefs)} layers")
    for i in range(len(pt_irefs)):
        all_ok &= compare(pt_irefs[i], tt_irefs[i],
                           f"inter_ref[{i}]", atol=1e-3)

    return all_ok


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SegDeformableTransformer: PyTorch vs TTSim comparison")
    print("=" * 70)
    print(f"Config: DIM={DIM}, NHEAD={NHEAD}, NUM_LEVELS={NUM_LEVELS}, "
          f"NUM_POINTS={NUM_POINTS}")
    print(f"  Encoder: {NUM_ENC_LAYERS} layers, FFN={FFN_CHANNELS}")
    print(f"  Decoder: {NUM_DEC_LAYERS} layers, FFN={FFN_CHANNELS}, "
          f"return_intermediate=True")
    print(f"  Batch={B}, Queries={NUM_QUERY}, "
          f"Spatial=[{H0}x{W0}, {H1}x{W1}]")

    try:
        ok = test_seg_deformable_transformer()
        print("\n" + "=" * 70)
        if ok:
            print("RESULT: ALL CHECKS PASSED")
        else:
            print("RESULT: SOME CHECKS FAILED")
        print("=" * 70)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
