#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comparison test: PansegformerHead (TTSim vs PyTorch).

Tests the full PansegformerHead forward pass with shared weights between
a PyTorch reference and the TTSim implementation.  The PyTorch reference
classes are defined inline (no mmcv/mmdet dependency).

Config (reduced for speed):
  embed_dims=32, nhead=4, bev_h=4, bev_w=4
  Encoder: 1 layer (MSDA self-attn, feedforward=64)
  Decoder: 2 layers (MHA self-attn + MSDA cross-attn, return_intermediate=True)
  num_query=6, num_things_classes=3, num_stuff_classes=1
  num_feature_levels=1 (single BEV level)
"""

import copy
import math
import os
import sys
import traceback

polaris_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
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

# TTSim modules under test
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.panseg_head import (
    PansegformerHead as TT_PansegformerHead,
    RegBranch as TT_RegBranch,
)


# ====================================================================
# Test constants (reduced for speed)
# ====================================================================
DIM = 32
NHEAD = 4
HEAD_DIM = DIM // NHEAD
NUM_LEVELS = 1
NUM_POINTS = 4
NUM_ENC_LAYERS = 1
NUM_DEC_LAYERS = 2
FFN_CHANNELS = 64
B = 1
NUM_QUERY = 6
BEV_H = 4
BEV_W = 4
S = BEV_H * BEV_W  # 16
NUM_THINGS = 3
NUM_STUFF = 1
NUM_DEC_THINGS = 2
NUM_DEC_STUFF = 2
POS_FEATS = DIM // 2  # 16


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
    """Inline MSDA (matches mmcv MultiScaleDeformableAttention)."""

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

        if isinstance(spatial_shapes, torch.Tensor):
            spatial_shapes_t = spatial_shapes
        else:
            spatial_shapes_t = torch.tensor(spatial_shapes, dtype=torch.long)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes_t[..., 1], spatial_shapes_t[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets
                / offset_normalizer[None, None, None, :, None, :])
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
    """Inline MHA (matches mmcv MultiheadAttention wrapper)."""

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

        q = self.q_proj(query).view(
            bs, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(
            bs, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(
            bs, nk, self.num_heads, self.head_dim).transpose(1, 2)

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
    """Inline FFN (matches mmcv FFN with num_fcs=2, ReLU)."""

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
    """Inline transformer layer (matches mmcv BaseTransformerLayer)."""

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
    """Inline SegDeformableTransformer (with_box_refine=True,
    as_two_stage=False)."""

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

        # Stack intermediate outputs for PansegformerHead
        hs = torch.stack(intermediate)          # [L, nq, B, D]
        inter_refs = torch.stack(
            intermediate_reference_points)      # [L, B, nq, ref_dim]

        mem_tuple = (memory, lvl_pos_embed_flatten, mask_flatten, query_pos)
        return (mem_tuple, hs, init_reference_out, inter_refs, None, None)


# ====================================================================
# SinePositionalEncoding (numpy — shared by both PT and TT)
# ====================================================================

def sine_pos_encoding_np(h, w, num_feats, normalize=True, offset=-0.5,
                         temperature=10000.0, eps=1e-6):
    """Compute SinePositionalEncoding as numpy array [1, 2*num_feats, H, W]."""
    scale = 2.0 * np.pi
    not_mask = np.ones((1, h, w), dtype=np.float32)
    y_embed = np.cumsum(not_mask, axis=1)
    x_embed = np.cumsum(not_mask, axis=2)

    if normalize:
        y_embed = (y_embed + offset) / (y_embed[:, -1:, :] + eps) * scale
        x_embed = (x_embed + offset) / (x_embed[:, :, -1:] + eps) * scale

    dim_t = np.arange(num_feats, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x_enc = np.zeros_like(pos_x)
    pos_x_enc[:, :, :, 0::2] = np.sin(pos_x[:, :, :, 0::2])
    pos_x_enc[:, :, :, 1::2] = np.cos(pos_x[:, :, :, 1::2])

    pos_y_enc = np.zeros_like(pos_y)
    pos_y_enc[:, :, :, 0::2] = np.sin(pos_y[:, :, :, 0::2])
    pos_y_enc[:, :, :, 1::2] = np.cos(pos_y[:, :, :, 1::2])

    pos = np.concatenate([pos_y_enc, pos_x_enc], axis=3)
    pos = pos.transpose(0, 3, 1, 2).astype(np.float32)
    return pos


# ====================================================================
# PyTorch reference: PT_PansegformerHead
# ====================================================================

class PT_PansegformerHead(nn.Module):
    """Inline PansegformerHead (as_two_stage=False, with_box_refine=True)."""

    def __init__(self, transformer, embed_dims=256, num_query=300,
                 num_things_classes=3, num_stuff_classes=1,
                 bev_h=200, bev_w=200, num_reg_fcs=2,
                 num_decoder_layers=6, num_dec_things=4, num_dec_stuff=6):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.cls_out_channels = num_things_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_decoder_layers = num_decoder_layers
        self.num_dec_things = num_dec_things
        self.num_dec_stuff = num_dec_stuff

        self.transformer = transformer

        # Query embeddings
        self.query_embedding = nn.Embedding(num_query, embed_dims * 2)
        self.stuff_query = nn.Embedding(num_stuff_classes, embed_dims * 2)

        # cls branches (one per decoder layer)
        fc_cls = nn.Linear(embed_dims, num_things_classes)
        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(num_decoder_layers)])

        # reg branches (one per decoder layer)
        def _make_reg():
            layers = []
            for _ in range(num_reg_fcs):
                layers.append(nn.Linear(embed_dims, embed_dims))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(embed_dims, 4))
            return nn.Sequential(*layers)

        self.reg_branches = nn.ModuleList(
            [_make_reg() for _ in range(num_decoder_layers)])

        # Mask decoder branches
        fc_cls_stuff = nn.Linear(embed_dims, 1)
        self.cls_thing_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(num_dec_things)])
        self.cls_stuff_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls_stuff) for _ in range(num_dec_stuff)])
        self.reg_branches2 = nn.ModuleList(
            [_make_reg() for _ in range(num_dec_things)])

    def forward(self, bev_embed):
        """
        Args:
            bev_embed: torch.Tensor [S, B, C] where S = bev_h * bev_w.
        Returns:
            dict with outputs_classes, outputs_coords, args_tuple, reference.
        """
        S, bs, C = bev_embed.shape

        # 1. Reshape [S,B,C] -> [B,C,H,W]
        bev_feat = bev_embed.permute(1, 0, 2)               # [B,S,C]
        bev_feat = bev_feat.reshape(bs, self.bev_h, self.bev_w, C)
        bev_feat = bev_feat.permute(0, 3, 1, 2)             # [B,C,H,W]

        # 2. Masks + positional encoding
        mask = torch.zeros((bs, self.bev_h, self.bev_w))
        pos_embed_np = sine_pos_encoding_np(
            self.bev_h, self.bev_w, C // 2,
            normalize=True, offset=-0.5)
        pos_embed = torch.from_numpy(pos_embed_np)
        if bs > 1:
            pos_embed = pos_embed.expand(bs, -1, -1, -1)

        # 3. Run transformer
        mlvl_feats = [bev_feat]
        mlvl_masks = [mask]
        mlvl_pos_embeds = [pos_embed]

        (memory_sf, lvl_pos_sf, mask_flat_out, query_pos_sf), \
            hs, init_reference, inter_references, _, _ = \
            self.transformer(
                mlvl_feats, mlvl_masks,
                self.query_embedding.weight,
                mlvl_pos_embeds,
                reg_branches=self.reg_branches)

        # 4. Permute to batch-first
        memory = memory_sf.permute(1, 0, 2)        # [B,S,C]
        query_pos = query_pos_sf.permute(1, 0, 2)  # [B,nq,C]
        memory_pos = lvl_pos_sf.permute(1, 0, 2)   # [B,S,C]

        # hs: [L, nq, B, C] -> [L, B, nq, C]
        hs = hs.permute(0, 2, 1, 3)
        query = hs[-1]                              # [B, nq, C]

        args_tuple = [memory, mask_flat_out, memory_pos,
                      query, None, query_pos,
                      [(self.bev_h, self.bev_w)]]

        # 5. Classification + regression per decoder layer
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = self._inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp = tmp + reference
            else:
                tmp_clone = tmp.clone()
                tmp_clone[..., :2] = tmp_clone[..., :2] + reference
                tmp = tmp_clone

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        reference = inter_references[-1]

        return {
            'bev_embed': bev_embed,
            'outputs_classes': outputs_classes,
            'outputs_coords': outputs_coords,
            'args_tuple': args_tuple,
            'reference': reference,
        }

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-5):
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)


# ====================================================================
# Weight copy helpers
# ====================================================================

def copy_linear(pt_linear, tt_linear):
    """Copy nn.Linear weights to SimNN.Linear (no transpose — SimNN.Linear transposes internally)."""
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
    """Copy MHA weights (4 linear layers)."""
    copy_linear(pt_mha.q_proj, tt_mha.q_proj)
    copy_linear(pt_mha.k_proj, tt_mha.k_proj)
    copy_linear(pt_mha.v_proj, tt_mha.v_proj)
    copy_linear(pt_mha.out_proj, tt_mha.out_proj)


def copy_ffn(pt_ffn, tt_ffn):
    """Copy FFN weights (2 linear layers)."""
    copy_linear(pt_ffn.fc1, tt_ffn.layers[0])
    copy_linear(pt_ffn.fc2, tt_ffn.layers[1])


def copy_encoder_layer(pt_layer, tt_layer):
    """Copy encoder layer: 1 MSDA + 1 FFN.
    Norms: builder_utils.LayerNorm has no affine, so skip."""
    copy_msda(pt_layer.attentions[0], tt_layer.attentions[0])
    copy_ffn(pt_layer.ffns[0], tt_layer.ffns[0])


def copy_decoder_layer(pt_layer, tt_layer):
    """Copy decoder layer: 1 MHA + 1 MSDA + 1 FFN.
    Norms: builder_utils.LayerNorm has no affine, so skip."""
    copy_mha(pt_layer.attentions[0], tt_layer.attentions[0])
    copy_msda(pt_layer.attentions[1], tt_layer.attentions[1])
    copy_ffn(pt_layer.ffns[0], tt_layer.ffns[0])


def copy_reg_branch(pt_seq, tt_rb):
    """Copy nn.Sequential(Linear,ReLU,Linear,ReLU,Linear) -> TT RegBranch."""
    pt_linears = [m for m in pt_seq.modules() if isinstance(m, nn.Linear)]
    copy_linear(pt_linears[0], tt_rb.fc0)
    copy_linear(pt_linears[1], tt_rb.fc1)
    copy_linear(pt_linears[2], tt_rb.fc2)


def copy_transformer(pt_tr, tt_tr):
    """Copy full SegDeformableTransformer weights."""
    for i in range(len(pt_tr.encoder_layers)):
        copy_encoder_layer(pt_tr.encoder_layers[i], tt_tr.encoder_layers[i])

    copy_f_layernorm(pt_tr.encoder_post_norm, tt_tr.encoder_post_norm)

    for i in range(len(pt_tr.decoder_layers)):
        copy_decoder_layer(pt_tr.decoder_layers[i], tt_tr.decoder_layers[i])

    for lvl in range(pt_tr.num_feature_levels):
        tt_tr.level_embeds[lvl].data = (
            pt_tr.level_embeds.data[lvl].detach().numpy()
            .reshape(1, 1, -1).astype(np.float32))

    copy_linear(pt_tr.reference_points_fc, tt_tr.reference_points_fc)


def copy_panseg_head(pt, tt):
    """Copy all PansegformerHead weights from PyTorch to TTSim."""
    # 1. Transformer
    copy_transformer(pt.transformer, tt.transformer)

    # 2. Query embeddings (constant tensors in TTSim)
    tt.query_embedding_weight.data = (
        pt.query_embedding.weight.data.detach().numpy().astype(np.float32))
    tt.stuff_query_weight[:] = (
        pt.stuff_query.weight.data.detach().numpy().astype(np.float32))

    # 3. cls_branches
    for i in range(pt.num_decoder_layers):
        copy_linear(pt.cls_branches[i], tt.cls_branches[i])

    # 4. reg_branches
    for i in range(pt.num_decoder_layers):
        copy_reg_branch(pt.reg_branches[i], tt.reg_branches[i])

    # 4b. dec_reg_branches (same weights as reg_branches, separate ONNX nodes)
    for i in range(pt.num_decoder_layers):
        copy_reg_branch(pt.reg_branches[i], tt.dec_reg_branches[i])

    # 5. cls_thing_branches
    for i in range(pt.num_dec_things):
        copy_linear(pt.cls_thing_branches[i], tt.cls_thing_branches[i])

    # 6. cls_stuff_branches
    for i in range(pt.num_dec_stuff):
        copy_linear(pt.cls_stuff_branches[i], tt.cls_stuff_branches[i])

    # 7. reg_branches2
    for i in range(pt.num_dec_things):
        copy_reg_branch(pt.reg_branches2[i], tt.reg_branches2[i])


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
# Builders
# ====================================================================

def build_pt_encoder_layer():
    """Build one PyTorch encoder layer (MSDA self-attn, FFN, 2 norms)."""
    msda = PT_MultiScaleDeformableAttention(
        embed_dims=DIM, num_heads=NHEAD, num_levels=NUM_LEVELS,
        num_points=NUM_POINTS, batch_first=False)
    ffn = PT_FFN(embed_dims=DIM, feedforward_channels=FFN_CHANNELS)
    norms = [nn.LayerNorm(DIM) for _ in range(2)]
    for n in norms:
        nn.init.ones_(n.weight)
        nn.init.zeros_(n.bias)
    return PT_BaseTransformerLayer(
        attentions=[msda], ffn=ffn, norms=norms,
        operation_order=('self_attn', 'norm', 'ffn', 'norm'))


def build_pt_decoder_layer():
    """Build one PyTorch decoder layer (MHA + MSDA, FFN, 3 norms)."""
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


def build_pt_transformer():
    """Build full PT_SegDeformableTransformer."""
    enc_layers = [build_pt_encoder_layer() for _ in range(NUM_ENC_LAYERS)]
    dec_layers = [build_pt_decoder_layer() for _ in range(NUM_DEC_LAYERS)]
    tr = PT_SegDeformableTransformer(
        encoder_layers=enc_layers,
        decoder_layers=dec_layers,
        embed_dims=DIM,
        num_feature_levels=NUM_LEVELS,
        return_intermediate=True)
    return tr


# ====================================================================
# TEST 1: RegBranch
# ====================================================================

def test_reg_branch():
    print("\n" + "=" * 70)
    print("TEST 1: RegBranch (PT Sequential vs TT RegBranch)")
    print("=" * 70)

    # Build PT: Sequential(Linear, ReLU, Linear, ReLU, Linear->4)
    pt = nn.Sequential(
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, DIM), nn.ReLU(),
        nn.Linear(DIM, 4))
    pt.eval()

    # Build TT
    tt = TT_RegBranch('test_rb', embed_dims=DIM)

    # Copy weights
    pt_linears = [m for m in pt.modules() if isinstance(m, nn.Linear)]
    copy_linear(pt_linears[0], tt.fc0)
    copy_linear(pt_linears[1], tt.fc1)
    copy_linear(pt_linears[2], tt.fc2)

    # Input
    x_np = np.random.randn(B, NUM_QUERY, DIM).astype(np.float32)
    pt_x = torch.from_numpy(x_np)
    tt_x = F._from_data('rb_x', x_np)

    # Run
    with torch.no_grad():
        pt_out = pt(pt_x)
    tt_out = tt(tt_x)

    return compare(pt_out, tt_out, "RegBranch output [B,nq,4]", atol=1e-5)


# ====================================================================
# TEST 2: SinePositionalEncoding
# ====================================================================

def test_sine_pos_encoding():
    print("\n" + "=" * 70)
    print("TEST 2: SinePositionalEncoding (numpy vs TTSim precomputed)")
    print("=" * 70)

    # Numpy reference
    ref = sine_pos_encoding_np(BEV_H, BEV_W, POS_FEATS,
                               normalize=True, offset=-0.5)

    # TTSim precomputed
    tt_pe = TT_PansegformerHead._build_sine_pos_encoding(
        BEV_H, BEV_W, POS_FEATS, True, -0.5, 'test_pe')

    return compare(ref, tt_pe,
                   f"SinePosEnc [1,{DIM},{BEV_H},{BEV_W}]", atol=1e-6)


# ====================================================================
# TEST 3: Full PansegformerHead forward
# ====================================================================

def test_panseg_head_forward():
    print("\n" + "=" * 70)
    print("TEST 3: PansegformerHead full forward (PT vs TT)")
    print("=" * 70)

    # ---- Build PyTorch model ----
    pt_transformer = build_pt_transformer()
    pt = PT_PansegformerHead(
        transformer=pt_transformer,
        embed_dims=DIM,
        num_query=NUM_QUERY,
        num_things_classes=NUM_THINGS,
        num_stuff_classes=NUM_STUFF,
        bev_h=BEV_H,
        bev_w=BEV_W,
        num_reg_fcs=2,
        num_decoder_layers=NUM_DEC_LAYERS,
        num_dec_things=NUM_DEC_THINGS,
        num_dec_stuff=NUM_DEC_STUFF)
    pt.eval()

    # ---- Build TTSim model ----
    tt_transformer_cfg = dict(
        embed_dims=DIM,
        num_feature_levels=NUM_LEVELS,
        encoder_cfg=dict(
            type='DetrTransformerEncoder',
            num_layers=NUM_ENC_LAYERS,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=DIM, num_heads=NHEAD,
                    num_levels=NUM_LEVELS),
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder_cfg=dict(
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
                         num_levels=NUM_LEVELS)],
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'))),
    )

    tt = TT_PansegformerHead(
        'psh',
        embed_dims=DIM,
        num_query=NUM_QUERY,
        num_things_classes=NUM_THINGS,
        num_stuff_classes=NUM_STUFF,
        bev_h=BEV_H,
        bev_w=BEV_W,
        canvas_size=(BEV_H, BEV_W),
        num_decoder_layers=NUM_DEC_LAYERS,
        num_dec_things=NUM_DEC_THINGS,
        num_dec_stuff=NUM_DEC_STUFF,
        pos_encoding_num_feats=POS_FEATS,
        transformer_cfg=tt_transformer_cfg)

    # ---- Copy weights ----
    copy_panseg_head(pt, tt)

    # ---- Create input ----
    bev_np = np.random.randn(S, B, DIM).astype(np.float32)
    pt_bev = torch.from_numpy(bev_np)
    tt_bev = F._from_data('bev_in', bev_np)

    # ---- Run ----
    with torch.no_grad():
        pt_out = pt(pt_bev)
    tt_out = tt(tt_bev)

    # ---- Compare outputs ----
    all_ok = True

    # outputs_classes per decoder layer
    print(f"\n  Decoder outputs_classes: {NUM_DEC_LAYERS} layers")
    for lvl in range(NUM_DEC_LAYERS):
        all_ok &= compare(
            pt_out['outputs_classes'][lvl],
            tt_out['outputs_classes'][lvl],
            f"outputs_classes[{lvl}] [B,nq,{NUM_THINGS}]",
            atol=1e-5)

    # outputs_coords per decoder layer
    print(f"\n  Decoder outputs_coords: {NUM_DEC_LAYERS} layers")
    for lvl in range(NUM_DEC_LAYERS):
        all_ok &= compare(
            pt_out['outputs_coords'][lvl],
            tt_out['outputs_coords'][lvl],
            f"outputs_coords[{lvl}] [B,nq,4]",
            atol=1e-5)

    # memory (args_tuple[0])
    print(f"\n  Memory and reference:")
    all_ok &= compare(
        pt_out['args_tuple'][0],
        tt_out['args_tuple'][0],
        "memory [B,S,C]",
        atol=1e-5)

    # query_pos (args_tuple[5])
    all_ok &= compare(
        pt_out['args_tuple'][5],
        tt_out['args_tuple'][5],
        "query_pos [B,nq,C]",
        atol=1e-5)

    # reference
    all_ok &= compare(
        pt_out['reference'],
        tt_out['reference'],
        "reference [B,nq,ref_dim]",
        atol=1e-5)

    return all_ok


# ====================================================================
# TEST 4: Sub-module existence
# ====================================================================

def test_submodules():
    print("\n" + "=" * 70)
    print("TEST 4: PansegformerHead sub-module existence")
    print("=" * 70)

    tt_transformer_cfg = dict(
        embed_dims=DIM,
        num_feature_levels=NUM_LEVELS,
        encoder_cfg=dict(
            type='DetrTransformerEncoder',
            num_layers=NUM_ENC_LAYERS,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=DIM, num_heads=NHEAD,
                    num_levels=NUM_LEVELS),
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder_cfg=dict(
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
                         num_levels=NUM_LEVELS)],
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'))),
    )

    tt = TT_PansegformerHead(
        'psh_sub',
        embed_dims=DIM,
        num_query=NUM_QUERY,
        num_things_classes=NUM_THINGS,
        num_stuff_classes=NUM_STUFF,
        bev_h=BEV_H,
        bev_w=BEV_W,
        canvas_size=(BEV_H, BEV_W),
        num_decoder_layers=NUM_DEC_LAYERS,
        num_dec_things=NUM_DEC_THINGS,
        num_dec_stuff=NUM_DEC_STUFF,
        pos_encoding_num_feats=POS_FEATS,
        transformer_cfg=tt_transformer_cfg)

    ok = True
    checks = [
        ('transformer', hasattr(tt, 'transformer')),
        ('things_mask_head', hasattr(tt, 'things_mask_head')),
        ('stuff_mask_head', hasattr(tt, 'stuff_mask_head')),
        ('cls_branches', len(tt.cls_branches) == NUM_DEC_LAYERS),
        ('reg_branches', len(tt.reg_branches) == NUM_DEC_LAYERS),
        ('cls_thing_branches', len(tt.cls_thing_branches) == NUM_DEC_THINGS),
        ('cls_stuff_branches', len(tt.cls_stuff_branches) == NUM_DEC_STUFF),
        ('reg_branches2', len(tt.reg_branches2) == NUM_DEC_THINGS),
        ('query_embedding_weight', hasattr(tt, 'query_embedding_weight')),
        ('stuff_query_weight', hasattr(tt, 'stuff_query_weight')),
        ('pos_embed', hasattr(tt, 'pos_embed')),
    ]
    for name, passed in checks:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        ok &= passed

    return ok


# ====================================================================
# TEST 5: args_tuple completeness
# ====================================================================

def test_args_tuple_completeness():
    print("\n" + "=" * 70)
    print("TEST 5: args_tuple completeness (memory_mask, memory_pos, query)")
    print("=" * 70)

    # ---- Build PyTorch model ----
    pt_transformer = build_pt_transformer()
    pt = PT_PansegformerHead(
        transformer=pt_transformer,
        embed_dims=DIM,
        num_query=NUM_QUERY,
        num_things_classes=NUM_THINGS,
        num_stuff_classes=NUM_STUFF,
        bev_h=BEV_H,
        bev_w=BEV_W,
        num_reg_fcs=2,
        num_decoder_layers=NUM_DEC_LAYERS,
        num_dec_things=NUM_DEC_THINGS,
        num_dec_stuff=NUM_DEC_STUFF)
    pt.eval()

    # ---- Build TTSim model ----
    tt_transformer_cfg = dict(
        embed_dims=DIM,
        num_feature_levels=NUM_LEVELS,
        encoder_cfg=dict(
            type='DetrTransformerEncoder',
            num_layers=NUM_ENC_LAYERS,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=DIM, num_heads=NHEAD,
                    num_levels=NUM_LEVELS),
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decoder_cfg=dict(
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
                         num_levels=NUM_LEVELS)],
                ffn_cfgs=dict(
                    type='FFN', embed_dims=DIM,
                    feedforward_channels=FFN_CHANNELS, ffn_drop=0.0),
                operation_order=('self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'))),
    )

    tt = TT_PansegformerHead(
        'psh5',
        embed_dims=DIM,
        num_query=NUM_QUERY,
        num_things_classes=NUM_THINGS,
        num_stuff_classes=NUM_STUFF,
        bev_h=BEV_H,
        bev_w=BEV_W,
        canvas_size=(BEV_H, BEV_W),
        num_decoder_layers=NUM_DEC_LAYERS,
        num_dec_things=NUM_DEC_THINGS,
        num_dec_stuff=NUM_DEC_STUFF,
        pos_encoding_num_feats=POS_FEATS,
        transformer_cfg=tt_transformer_cfg)

    # ---- Copy weights ----
    copy_panseg_head(pt, tt)

    # ---- Create input ----
    bev_np = np.random.randn(S, B, DIM).astype(np.float32)
    pt_bev = torch.from_numpy(bev_np)
    tt_bev = F._from_data('bev5_in', bev_np)

    # ---- Run ----
    with torch.no_grad():
        pt_out = pt(pt_bev)
    tt_out = tt(tt_bev)

    all_ok = True

    # args_tuple[1] — memory_mask [B, S]
    all_ok &= compare(
        pt_out['args_tuple'][1],
        tt_out['args_tuple'][1],
        "args_tuple[1] memory_mask [B,S]",
        atol=1e-6)

    # args_tuple[2] — memory_pos [B, S, C]
    all_ok &= compare(
        pt_out['args_tuple'][2],
        tt_out['args_tuple'][2],
        "args_tuple[2] memory_pos [B,S,C]",
        atol=1e-5)

    # args_tuple[3] — query [B, nq, C] (last decoder hidden state)
    all_ok &= compare(
        pt_out['args_tuple'][3],
        tt_out['args_tuple'][3],
        "args_tuple[3] query [B,nq,C]",
        atol=1e-5)

    return all_ok


# ====================================================================
# Main
# ====================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("PansegformerHead: PyTorch vs TTSim comparison")
    print("=" * 70)
    print(f"Config: DIM={DIM}, NHEAD={NHEAD}, NUM_LEVELS={NUM_LEVELS}, "
          f"NUM_POINTS={NUM_POINTS}")
    print(f"  Encoder: {NUM_ENC_LAYERS} layers, FFN={FFN_CHANNELS}")
    print(f"  Decoder: {NUM_DEC_LAYERS} layers, FFN={FFN_CHANNELS}, "
          f"return_intermediate=True")
    print(f"  Batch={B}, Queries={NUM_QUERY}, BEV={BEV_H}x{BEV_W}")
    print(f"  Things={NUM_THINGS}, Stuff={NUM_STUFF}")

    passed = 0
    failed = 0

    tests = [
        ("RegBranch", test_reg_branch),
        ("SinePositionalEncoding", test_sine_pos_encoding),
        ("PansegformerHead forward", test_panseg_head_forward),
        ("Sub-module existence", test_submodules),
        ("args_tuple completeness", test_args_tuple_completeness),
    ]

    for test_name, test_fn in tests:
        try:
            ok = test_fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  [ERROR] {test_name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failed} TEST(S) FAILED")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
