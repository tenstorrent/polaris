
# =============================================================================
# ORIGINAL TORCH CODE (from FusionAD)
# Source: FusionAD/projects/mmdet3d_plugin/fusionad/dense_heads/seg_head_plugin/seg_deformable_transformer.py
# =============================================================================
# from mmcv.runner.fp16_utils import force_fp32
# from mmdet.models.utils.builder import TRANSFORMER
# from mmdet.models.utils import Transformer
# import warnings
# import math
# import copy
# import torch
# import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
# from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
#                                          MultiScaleDeformableAttention,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer_sequence)
# from mmcv.runner.base_module import BaseModule
# from torch.nn.init import normal_
#
# from mmdet.models.utils.builder import TRANSFORMER
# from mmcv.cnn.bricks.registry import ATTENTION
# from torch import einsum
#
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
#
# # Copy-paste from defromable detr in mmdet.
# @TRANSFORMER.register_module()
# class SegDeformableTransformer(Transformer):
#     """Implements the DeformableDETR transformer.
#
#     Args:
#         as_two_stage (bool): Generate query from encoder features.
#             Default: False.
#         num_feature_levels (int): Number of feature maps from FPN:
#             Default: 4.
#         two_stage_num_proposals (int): Number of proposals when set
#             `as_two_stage` as True. Default: 300.
#     """
#     def __init__(self,
#                  as_two_stage=False,
#                  num_feature_levels=4,
#                  two_stage_num_proposals=300,
#                  **kwargs):
#         super(SegDeformableTransformer, self).__init__(**kwargs)
#         self.fp16_enabled = False
#         self.as_two_stage = as_two_stage
#         self.num_feature_levels = num_feature_levels
#         self.two_stage_num_proposals = two_stage_num_proposals
#         self.embed_dims = self.encoder.embed_dims
#         self.init_layers()
#
#     def init_layers(self):
#         """Initialize layers of the DeformableDetrTransformer."""
#         self.level_embeds = nn.Parameter(
#             torch.Tensor(self.num_feature_levels, self.embed_dims))
#
#         if self.as_two_stage:
#             self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
#             self.enc_output_norm = nn.LayerNorm(self.embed_dims)
#             self.pos_trans = nn.Linear(self.embed_dims * 2,
#                                        self.embed_dims * 2)
#             self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
#         else:
#             self.reference_points = nn.Linear(self.embed_dims, 2)
#
#     def init_weights(self):
#         """Initialize the transformer weights."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MultiScaleDeformableAttention):
#                 try:
#                     m.init_weight()
#                 except:
#                     m.init_weights()
#         if not self.as_two_stage:
#             xavier_init(self.reference_points, distribution='uniform', bias=0.)
#         normal_(self.level_embeds)
#
#     def gen_encoder_output_proposals(self, memory, memory_padding_mask,
#                                      spatial_shapes):
#         """Generate proposals from encoded memory.
#
#         Args:
#             memory (Tensor) : The output of encoder,
#                 has shape (bs, num_key, embed_dim).  num_key is
#                 equal the number of points on feature map from
#                 all level.
#             memory_padding_mask (Tensor): Padding mask for memory.
#                 has shape (bs, num_key).
#             spatial_shapes (Tensor): The shape of all feature maps.
#                 has shape (num_level, 2).
#
#         Returns:
#             tuple: A tuple of feature map and bbox prediction.
#
#                 - output_memory (Tensor): The input of decoder,  \
#                     has shape (bs, num_key, embed_dim).  num_key is \
#                     equal the number of points on feature map from \
#                     all levels.
#                 - output_proposals (Tensor): The normalized proposal \
#                     after a inverse sigmoid, has shape \
#                     (bs, num_keys, 4).
#         """
#
#         N, S, C = memory.shape
#         proposals = []
#         _cur = 0
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
#                 N, H, W, 1)
#             valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
#             valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
#
#             grid_y, grid_x = torch.meshgrid(
#                 torch.linspace(0,
#                                H - 1,
#                                H,
#                                dtype=torch.float32,
#                                device=memory.device),
#                 torch.linspace(0,
#                                W - 1,
#                                W,
#                                dtype=torch.float32,
#                                device=memory.device))
#             grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
#
#             scale = torch.cat([valid_W.unsqueeze(-1),
#                                valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
#             grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
#             wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
#             proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
#             proposals.append(proposal)
#             _cur += (H * W)
#         output_proposals = torch.cat(proposals, 1)
#         output_proposals_valid = ((output_proposals > 0.01) &
#                                   (output_proposals < 0.99)).all(-1,
#                                                                  keepdim=True)
#         output_proposals = torch.log(output_proposals / (1 - output_proposals))
#         output_proposals = output_proposals.masked_fill(
#             memory_padding_mask.unsqueeze(-1), float('inf'))
#         output_proposals = output_proposals.masked_fill(
#             ~output_proposals_valid, float('inf'))
#
#         output_memory = memory
#         output_memory = output_memory.masked_fill(
#             memory_padding_mask.unsqueeze(-1), float(0))
#         output_memory = output_memory.masked_fill(~output_proposals_valid,
#                                                   float(0))
#         output_memory = self.enc_output_norm(self.enc_output(output_memory))
#         return output_memory, output_proposals
#
#     @staticmethod
#     def get_reference_points(spatial_shapes, valid_ratios, device):
#         """Get the reference points used in decoder.
#
#         Args:
#             spatial_shapes (Tensor): The shape of all
#                 feature maps, has shape (num_level, 2).
#             valid_ratios (Tensor): The radios of valid
#                 points on the feature map, has shape
#                 (bs, num_levels, 2)
#             device (obj:`device`): The device where
#                 reference_points should be.
#
#         Returns:
#             Tensor: reference points used in decoder, has \
#                 shape (bs, num_keys, num_levels, 2).
#         """
#         reference_points_list = []
#         for lvl, (H, W) in enumerate(spatial_shapes):
#             #  TODO  check this 0.5
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(0.5,
#                                H - 0.5,
#                                H,
#                                dtype=torch.float32,
#                                device=device),
#                 torch.linspace(0.5,
#                                W - 0.5,
#                                W,
#                                dtype=torch.float32,
#                                device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] *
#                                                H)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] *
#                                                W)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points
#
#     def get_valid_ratio(self, mask):
#         """Get the valid radios of feature maps of all  level."""
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio
#
#     def get_proposal_pos_embed(self,
#                                proposals,
#                                num_pos_feats=128,
#                                temperature=10000):
#         """Get the position embedding of proposal."""
#         scale = 2 * math.pi
#         dim_t = torch.arange(num_pos_feats,
#                              dtype=torch.float32,
#                              device=proposals.device)
#         dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
#         # N, L, 4
#         proposals = proposals.sigmoid() * scale
#         # N, L, 4, 128
#         pos = proposals[:, :, :, None] / dim_t
#         # N, L, 4, 64, 2
#         pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
#                           dim=4).flatten(2)
#         return pos
#
#     @force_fp32(apply_to=('mlvl_feats', 'query_embed', 'mlvl_pos_embeds'))
#     def forward(self,
#                 mlvl_feats,
#                 mlvl_masks,
#                 query_embed,
#                 mlvl_pos_embeds,
#                 reg_branches=None,
#                 cls_branches=None,
#                 **kwargs):
#         """Forward function for `Transformer`.
#
#         Args:
#             mlvl_feats (list(Tensor)): Input queries from
#                 different level. Each element has shape
#                 [bs, embed_dims, h, w].
#             mlvl_masks (list(Tensor)): The key_padding_mask from
#                 different level used for encoder and decoder,
#                 each element has shape  [bs, h, w].
#             query_embed (Tensor): The query embedding for decoder,
#                 with shape [num_query, c].
#             mlvl_pos_embeds (list(Tensor)): The positional encoding
#                 of feats from different level, has the shape
#                  [bs, embed_dims, h, w].
#             reg_branches (obj:`nn.ModuleList`): Regression heads for
#                 feature maps from each decoder layer. Only would
#                 be passed when
#                 `with_box_refine` is True. Default to None.
#             cls_branches (obj:`nn.ModuleList`): Classification heads
#                 for feature maps from each decoder layer. Only would
#                  be passed when `as_two_stage`
#                  is True. Default to None.
#
#
#         Returns:
#             tuple[Tensor]: results of decoder containing the following tensor.
#
#                 - inter_states: Outputs from decoder. If
#                     return_intermediate_dec is True output has shape \
#                       (num_dec_layers, bs, num_query, embed_dims), else has \
#                       shape (1, bs, num_query, embed_dims).
#                 - init_reference_out: The initial value of reference \
#                     points, has shape (bs, num_queries, 4).
#                 - inter_references_out: The internal value of reference \
#                     points in decoder, has shape \
#                     (num_dec_layers, bs,num_query, embed_dims)
#                 - enc_outputs_class: The classification score of \
#                     proposals generated from \
#                     encoder's feature maps, has shape \
#                     (batch, h*w, num_classes). \
#                     Only would be returned when `as_two_stage` is True, \
#                     otherwise None.
#                 - enc_outputs_coord_unact: The regression results \
#                     generated from encoder's feature maps., has shape \
#                     (batch, h*w, 4). Only would \
#                     be returned when `as_two_stage` is True, \
#                     otherwise None.
#         """
#
#         assert self.as_two_stage or query_embed is not None
#         feat_flatten = []
#         mask_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (feat, mask, pos_embed) in enumerate(
#                 zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
#             bs, c, h, w = feat.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             feat = feat.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)
#             lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             feat_flatten.append(feat)
#             mask_flatten.append(mask)
#         feat_flatten = torch.cat(feat_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes,
#                                          dtype=torch.long,
#                                          device=feat_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros(
#             (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         valid_ratios = torch.stack(
#             [self.get_valid_ratio(m) for m in mlvl_masks], 1)
#
#         reference_points = \
#             self.get_reference_points(spatial_shapes,
#                                       valid_ratios,
#                                       device=feat.device)
#
#         feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
#         lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
#             1, 0, 2)  # (H*W, bs, embed_dims)
#         memory = self.encoder(query=feat_flatten,
#                               key=None,
#                               value=None,
#                               query_pos=lvl_pos_embed_flatten,
#                               query_key_padding_mask=mask_flatten,
#                               spatial_shapes=spatial_shapes,
#                               reference_points=reference_points,
#                               level_start_index=level_start_index,
#                               valid_ratios=valid_ratios,
#                               **kwargs)
#
#         memory = memory.permute(1, 0, 2)
#         bs, _, c = memory.shape
#         if self.as_two_stage:
#             output_memory, output_proposals = \
#                 self.gen_encoder_output_proposals(
#                     memory, mask_flatten, spatial_shapes)
#             enc_outputs_class = cls_branches[self.decoder.num_layers](
#                 output_memory)
#             enc_outputs_coord_unact = \
#                 reg_branches[
#                     self.decoder.num_layers](output_memory) + output_proposals
#
#             topk = self.two_stage_num_proposals
#             topk_proposals = torch.topk(enc_outputs_class[..., 0], topk,
#                                         dim=1)[1]
#             topk_coords_unact = torch.gather(
#                 enc_outputs_coord_unact, 1,
#                 topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
#             topk_coords_unact = topk_coords_unact.detach()
#             reference_points = topk_coords_unact.sigmoid()
#             init_reference_out = reference_points
#             pos_trans_out = self.pos_trans_norm(
#                 self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
#             query_pos, query = torch.split(pos_trans_out, c, dim=2)
#         else:
#             #logging.info('query_embd',query_embed.shape, c)
#             # query_embed N *(2C)
#             query_pos, query = torch.split(query_embed, c, dim=1)
#             query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
#             query = query.unsqueeze(0).expand(bs, -1, -1)
#             reference_points = self.reference_points(query_pos).sigmoid()
#             init_reference_out = reference_points
#
#         # decoder
#         query = query.permute(1, 0, 2)
#         memory = memory.permute(1, 0, 2)
#         query_pos = query_pos.permute(1, 0, 2)
#         inter_states, inter_references = self.decoder(
#             query=query,
#             key=None,
#             value=memory,
#             query_pos=query_pos,
#             key_padding_mask=mask_flatten,
#             reference_points=reference_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             valid_ratios=valid_ratios,
#             reg_branches=reg_branches,
#             **kwargs)
#         inter_references_out = inter_references
#         if self.as_two_stage:
#             return (memory,lvl_pos_embed_flatten,mask_flatten,query_pos), inter_states, init_reference_out,\
#                 inter_references_out, enc_outputs_class,\
#                 enc_outputs_coord_unact
#         return (memory,lvl_pos_embed_flatten,mask_flatten,query_pos), inter_states, init_reference_out, \
#             inter_references_out, None, None
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of SegDeformableTransformer for panoptic segmentation.

Implements the Deformable DETR transformer architecture used by
PansegformerHead for panoptic segmentation in FusionAD.

Architecture (config: as_two_stage=False, with_box_refine=True):
  Encoder: 6 layers of BaseTransformerLayer with MultiScaleDeformableAttention
           self-attention + post-norm LayerNorm
  Decoder: 6 layers of DetrTransformerDecoderLayer with MultiheadAttention
           self-attention + MultiScaleDeformableAttention cross-attention,
           with iterative reference point refinement via reg_branches

Forward flow:
  1. Flatten + concat multi-level features, masks, pos encodings
  2. Add level_embeds to positional encodings
  3. Compute spatial_shapes, level_start_index, valid_ratios, encoder ref_pts
  4. Run encoder (6 layers + post_norm) in (S, B, D) format
  5. Split query_embed → query_pos, query; expand to batch
  6. Compute decoder ref_pts via Linear(256, 2) + sigmoid
  7. Run decoder (6 layers) with iterative ref_pt refinement via reg_branches
  8. Return (memory_tuple, inter_states, init_reference, inter_references, None, None)

Original: projects/mmdet3d_plugin/fusionad/dense_heads/seg_head_plugin/
          seg_deformable_transformer.py
Converted to TTSim: March 2026

============================================================================
MMCV Import Conversions
============================================================================
1. Base Classes:
   - Transformer(BaseModule) -> SimNN.Module
   - DetrTransformerEncoder(TransformerLayerSequence) -> encoder layer list
   - DeformableDetrTransformerDecoder(TransformerLayerSequence) -> decoder loop
   - BaseTransformerLayer -> MyCustomBaseTransformerLayer (from modules/)
2. Registry:
   - @TRANSFORMER.register_module() -> removed
   - @force_fp32 -> removed
3. Operations:
   - nn.Parameter (level_embeds) -> F._from_data constants
   - nn.Linear (reference_points) -> SimNN.Linear
   - torch.cat/split/stack -> F.ConcatX / F.SliceF / numpy
   - inverse_sigmoid -> InverseSigmoid module
"""

#-------------------------------PyTorch--------------------------------

# from mmcv.runner.fp16_utils import force_fp32
# from mmdet.models.utils.builder import TRANSFORMER
# from mmdet.models.utils import Transformer
# import warnings
# import math
# import copy
# import torch
# import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
# from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
#                                          MultiScaleDeformableAttention,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer_sequence)
# from mmcv.runner.base_module import BaseModule
# from torch.nn.init import normal_
#
# from mmdet.models.utils.builder import TRANSFORMER
# from mmcv.cnn.bricks.registry import ATTENTION
# from torch import einsum
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
#
#
# @TRANSFORMER.register_module()
# class SegDeformableTransformer(Transformer):
#     def __init__(self, as_two_stage=False, num_feature_levels=4,
#                  two_stage_num_proposals=300, **kwargs):
#         super(SegDeformableTransformer, self).__init__(**kwargs)
#         self.fp16_enabled = False
#         self.as_two_stage = as_two_stage
#         self.num_feature_levels = num_feature_levels
#         self.two_stage_num_proposals = two_stage_num_proposals
#         self.embed_dims = self.encoder.embed_dims
#         self.init_layers()
#
#     def init_layers(self):
#         self.level_embeds = nn.Parameter(
#             torch.Tensor(self.num_feature_levels, self.embed_dims))
#         if self.as_two_stage:
#             ...
#         else:
#             self.reference_points = nn.Linear(self.embed_dims, 2)
#
#     @staticmethod
#     def get_reference_points(spatial_shapes, valid_ratios, device):
#         ...
#
#     def get_valid_ratio(self, mask):
#         ...
#
#     @force_fp32(apply_to=('mlvl_feats', 'query_embed', 'mlvl_pos_embeds'))
#     def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds,
#                 reg_branches=None, cls_branches=None, **kwargs):
#         ...
#         # Flatten multi-level features
#         # Run encoder
#         # Split query_embed, get reference_points
#         # Run decoder with iterative refinement
#         # Return 6-tuple

#-------------------------------TTSIM-----------------------------------

import sys
import os
import copy
import math
import warnings
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add seg_head_plugin directory for sibling imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add dense_heads directory
dense_heads_dir = os.path.abspath(os.path.join(current_dir, '..'))
if dense_heads_dir not in sys.path:
    sys.path.insert(0, dense_heads_dir)

# Add fusionad directory so "from modules.xxx" imports resolve
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import (
    build_attention,
    build_feedforward_network,
    build_norm_layer,
    LayerNorm,
    InverseSigmoid,
    inverse_sigmoid_np,
)

from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.custom_base_transformer_layer import (
    MyCustomBaseTransformerLayer,
)


# ======================================================================
# SegDeformableTransformer
# ======================================================================

class SegDeformableTransformer(SimNN.Module):
    """
    TTSim implementation of SegDeformableTransformer.

    Deformable DETR backbone for panoptic segmentation.
    Config: as_two_stage=False, with_box_refine=True.

    Encoder: N layers of BaseTransformerLayer (MSDA self-attn) + post-norm.
    Decoder: N layers of DetrTransformerDecoderLayer (MHA self-attn +
             MSDA cross-attn) with iterative reference point refinement.

    Args:
        name (str): Module name.
        encoder_cfg (dict): Encoder config with 'transformerlayers' and
            'num_layers' keys.
        decoder_cfg (dict): Decoder config with 'transformerlayers',
            'num_layers', and 'return_intermediate' keys.
        num_feature_levels (int): Number of FPN feature levels. Default: 4.
        embed_dims (int): Embedding dimension. Default: 256.
    """

    def __init__(self, name, encoder_cfg, decoder_cfg,
                 num_feature_levels=4, embed_dims=256):
        super().__init__()
        self.name = name
        self.num_feature_levels = num_feature_levels
        self.embed_dims = embed_dims

        # ---- Encoder layers ----
        enc_tl = encoder_cfg.get('transformerlayers', {})
        enc_nl = encoder_cfg.get('num_layers', 6)
        self.encoder_num_layers = enc_nl

        _enc = []
        for i in range(enc_nl):
            cfg = copy.deepcopy(enc_tl)
            cfg.pop('type', None)
            cfg.setdefault('batch_first', False)
            _enc.append(MyCustomBaseTransformerLayer(
                name=f'{name}.enc.{i}', **cfg))
        self.encoder_layers = SimNN.ModuleList(_enc)

        # Encoder post-norm (DetrTransformerEncoder defaults to LN)
        self.encoder_post_norm = F.LayerNorm(f'{name}.enc_post_norm', embed_dims)

        # ---- Decoder layers ----
        dec_tl = decoder_cfg.get('transformerlayers', {})
        dec_nl = decoder_cfg.get('num_layers', 6)
        self.return_intermediate = decoder_cfg.get('return_intermediate', False)
        self.decoder_num_layers = dec_nl

        _dec = []
        for i in range(dec_nl):
            cfg = copy.deepcopy(dec_tl)
            cfg.pop('type', None)
            cfg.setdefault('batch_first', False)
            _dec.append(MyCustomBaseTransformerLayer(
                name=f'{name}.dec.{i}', **cfg))
        self.decoder_layers = SimNN.ModuleList(_dec)

        # Per-layer inverse sigmoid for decoder refinement
        _inv_sigs = []
        for i in range(dec_nl):
            inv = InverseSigmoid(f'{name}.inv_sig_{i}')
            setattr(self, f'_inv_sig_{i}', inv)
            _inv_sigs.append(inv)
        self._inv_sigmoids = _inv_sigs

        # ---- Level embeddings ----
        # [num_feature_levels, embed_dims] split into N constants [1, 1, C]
        self.level_embeds = []
        for lvl in range(num_feature_levels):
            le = F._from_data(f'{name}.level_embed_{lvl}',
                              np.zeros((1, 1, embed_dims), dtype=np.float32),
                              is_const=True)
            setattr(self, f'level_embed_{lvl}', le)
            self.level_embeds.append(le)

        # ---- Reference points linear (as_two_stage=False) ----
        self.reference_points_fc = SimNN.Linear(
            f'{name}.reference_points',
            in_features=embed_dims, out_features=2)
        self.ref_pts_sigmoid = F.Sigmoid(f'{name}.ref_pts_sigmoid')

        super().link_op2module()

    # ------------------------------------------------------------------
    # Static helper: compute encoder reference points (numpy)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_reference_points(spatial_shapes, valid_ratios):
        """
        Compute normalised grid reference points for the encoder.

        Args:
            spatial_shapes: np.ndarray [num_levels, 2] of (H, W).
            valid_ratios: np.ndarray [bs, num_levels, 2].

        Returns:
            np.ndarray [bs, total_S, num_levels, 2] of reference points.
        """
        bs = valid_ratios.shape[0]
        reference_points_list = []
        for lvl in range(len(spatial_shapes)):
            H, W = int(spatial_shapes[lvl, 0]), int(spatial_shapes[lvl, 1])
            ref_y, ref_x = np.meshgrid(
                np.linspace(0.5, H - 0.5, H),
                np.linspace(0.5, W - 0.5, W),
                indexing='ij')
            ref_y = ref_y.reshape(-1)[np.newaxis, :]          # [1, H*W]
            ref_x = ref_x.reshape(-1)[np.newaxis, :]          # [1, H*W]
            vr_w = valid_ratios[:, lvl, 0:1]                  # [bs, 1]
            vr_h = valid_ratios[:, lvl, 1:2]                  # [bs, 1]
            ref_y = ref_y / (vr_h * H)                        # [bs, H*W]
            ref_x = ref_x / (vr_w * W)                        # [bs, H*W]
            ref = np.stack((ref_x, ref_y), axis=-1)            # [bs, H*W, 2]
            reference_points_list.append(ref)
        reference_points = np.concatenate(
            reference_points_list, axis=1)                     # [bs, total_S, 2]
        # Broadcast across levels: [bs, S, 1, 2] * [bs, 1, nl, 2]
        reference_points = (reference_points[:, :, np.newaxis, :]
                            * valid_ratios[:, np.newaxis, :, :])
        return reference_points.astype(np.float32)             # [bs, S, nl, 2]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds,
                 reg_branches=None, cls_branches=None, **kwargs):
        """
        Forward pass of SegDeformableTransformer.

        Args:
            mlvl_feats: list of SimTensors [bs, C, h_i, w_i], one per level.
            mlvl_masks: list of SimTensors [bs, h_i, w_i], padding masks.
            query_embed: SimTensor [num_query, 2*C], learnt query embeddings.
            mlvl_pos_embeds: list of SimTensors [bs, C, h_i, w_i].
            reg_branches: list of SimNN.Module (one per decoder layer) or None.
            cls_branches: unused (as_two_stage=False). Default: None.

        Returns:
            6-tuple:
              (memory_sf, lvl_pos_sf, mask_flat, query_pos_sf),
              inter_states,          # list of [nq, bs, C] per layer
              init_reference_out,    # [bs, nq, 2]
              inter_references_out,  # list of [bs, nq, 2or4] per layer
              None, None             # enc_outputs (as_two_stage=False)
        """
        n = self.name
        c = self.embed_dims

        # ==============================================================
        # 1. Flatten multi-level features
        # ==============================================================
        feat_parts, mask_parts, pos_parts = [], [], []
        spatial_shapes = []
        num_input_levels = len(mlvl_feats)

        for lvl in range(num_input_levels):
            feat = mlvl_feats[lvl]
            mask = mlvl_masks[lvl]
            pos  = mlvl_pos_embeds[lvl]
            bs = feat.shape[0]
            h, w = feat.shape[2], feat.shape[3]
            spatial_shapes.append((h, w))
            hw = h * w

            # feat: [bs,C,h,w] → [bs,C,h*w] → [bs,h*w,C]
            _fs = F._from_data(f'{n}.fs_{lvl}',
                               np.array([bs, c, hw], dtype=np.int64),
                               is_const=True)
            setattr(self, _fs.name, _fs)
            _fr = F.Reshape(f'{n}.fr_{lvl}')
            setattr(self, _fr.name, _fr)
            feat = _fr(feat, _fs)
            setattr(self, feat.name, feat)
            _fp = F.Transpose(f'{n}.fp_{lvl}', perm=[0, 2, 1])
            setattr(self, _fp.name, _fp)
            feat = _fp(feat)
            setattr(self, feat.name, feat)

            # mask: [bs,h,w] → [bs,h*w]
            _ms = F._from_data(f'{n}.ms_{lvl}',
                               np.array([bs, hw], dtype=np.int64),
                               is_const=True)
            setattr(self, _ms.name, _ms)
            _mr = F.Reshape(f'{n}.mr_{lvl}')
            setattr(self, _mr.name, _mr)
            mask = _mr(mask, _ms)
            setattr(self, mask.name, mask)

            # pos: [bs,C,h,w] → [bs,h*w,C]
            _pr = F.Reshape(f'{n}.pr_{lvl}')
            setattr(self, _pr.name, _pr)
            pos = _pr(pos, _fs)          # reuse feat shape
            setattr(self, pos.name, pos)
            _pp = F.Transpose(f'{n}.pp_{lvl}', perm=[0, 2, 1])
            setattr(self, _pp.name, _pp)
            pos = _pp(pos)
            setattr(self, pos.name, pos)

            # lvl_pos = pos + level_embed[lvl]  (broadcast [1,1,C])
            _al = F.Add(f'{n}.al_{lvl}')
            setattr(self, _al.name, _al)
            lvl_pos = _al(pos, self.level_embeds[lvl])
            setattr(self, lvl_pos.name, lvl_pos)

            feat_parts.append(feat)
            mask_parts.append(mask)
            pos_parts.append(lvl_pos)

        # Concat across levels (axis=1) — skip ConcatX for single level
        if len(feat_parts) == 1:
            feat_flatten = feat_parts[0]
            mask_flatten = mask_parts[0]
            lvl_pos_flatten = pos_parts[0]
        else:
            _cf = F.ConcatX(f'{n}.cat_f', axis=1)
            setattr(self, _cf.name, _cf)
            feat_flatten = _cf(*feat_parts)           # [bs, S, C]
            setattr(self, feat_flatten.name, feat_flatten)

            _cm = F.ConcatX(f'{n}.cat_m', axis=1)
            setattr(self, _cm.name, _cm)
            mask_flatten = _cm(*mask_parts)            # [bs, S]
            setattr(self, mask_flatten.name, mask_flatten)

            _cp = F.ConcatX(f'{n}.cat_p', axis=1)
            setattr(self, _cp.name, _cp)
            lvl_pos_flatten = _cp(*pos_parts)          # [bs, S, C]
            setattr(self, lvl_pos_flatten.name, lvl_pos_flatten)

        # ==============================================================
        # 2. Numpy constants: spatial_shapes, level_start_index,
        #    valid_ratios, encoder reference_points
        # ==============================================================
        ss_np = np.array(spatial_shapes, dtype=np.int64)
        lsi_np = np.concatenate([
            np.array([0], dtype=np.int64),
            np.cumsum(ss_np[:, 0] * ss_np[:, 1])[:-1]
        ])
        vr_np = np.ones((bs, num_input_levels, 2), dtype=np.float32)

        enc_rp_np = self._get_reference_points(ss_np, vr_np)  # [bs,S,nl,2]
        enc_rp = F._from_data(f'{n}.enc_rp', enc_rp_np, is_const=True)
        setattr(self, enc_rp.name, enc_rp)

        # ==============================================================
        # 3. Permute to (S, B, D) for encoder
        # ==============================================================
        _pf = F.Transpose(f'{n}.pf', perm=[1, 0, 2])
        setattr(self, _pf.name, _pf)
        feat_sf = _pf(feat_flatten)                # [S, bs, C]
        setattr(self, feat_sf.name, feat_sf)

        _ppf = F.Transpose(f'{n}.ppf', perm=[1, 0, 2])
        setattr(self, _ppf.name, _ppf)
        lvl_pos_sf = _ppf(lvl_pos_flatten)         # [S, bs, C]
        setattr(self, lvl_pos_sf.name, lvl_pos_sf)

        # ==============================================================
        # 4. Encoder: loop + post-norm
        # ==============================================================
        memory = feat_sf
        for i, layer in enumerate(self.encoder_layers):
            memory = layer(
                memory,
                key=None,
                value=None,
                query_pos=lvl_pos_sf,
                query_key_padding_mask=mask_flatten,
                spatial_shapes=spatial_shapes,
                reference_points=enc_rp,
                level_start_index=lsi_np,
            )

        memory = self.encoder_post_norm(memory)
        setattr(self, memory.name, memory)

        # ==============================================================
        # 5. Permute memory to (B, S, D) for query handling
        # ==============================================================
        _pm = F.Transpose(f'{n}.pm', perm=[1, 0, 2])
        setattr(self, _pm.name, _pm)
        memory_bf = _pm(memory)                    # [bs, S, C]
        setattr(self, memory_bf.name, memory_bf)

        # ==============================================================
        # 6. Query embed split + expand  (as_two_stage=False)
        # ==============================================================
        # query_embed: [nq, 2*C] → query_pos [bs,nq,C], query [bs,nq,C]
        nq = query_embed.shape[0]

        # Always use graph ops so query_embed stays connected.
        _sax = F._from_data(f'{n}.sax',
                            np.array([-1], dtype=np.int64), is_const=True)
        setattr(self, _sax.name, _sax)
        _sstp = F._from_data(f'{n}.sstp',
                             np.array([1], dtype=np.int64), is_const=True)
        setattr(self, _sstp.name, _sstp)
        _s0 = F._from_data(f'{n}.s0',
                           np.array([0], dtype=np.int64), is_const=True)
        setattr(self, _s0.name, _s0)
        _sc = F._from_data(f'{n}.sc',
                           np.array([c], dtype=np.int64), is_const=True)
        setattr(self, _sc.name, _sc)
        _s2c = F._from_data(f'{n}.s2c',
                            np.array([2 * c], dtype=np.int64),
                            is_const=True)
        setattr(self, _s2c.name, _s2c)

        _sp = F.SliceF(f'{n}.sp', out_shape=[nq, c])
        setattr(self, _sp.name, _sp)
        qp_2d = _sp(query_embed, _s0, _sc, _sax, _sstp)
        setattr(self, qp_2d.name, qp_2d)

        _sq = F.SliceF(f'{n}.sq', out_shape=[nq, c])
        setattr(self, _sq.name, _sq)
        q_2d = _sq(query_embed, _sc, _s2c, _sax, _sstp)
        setattr(self, q_2d.name, q_2d)

        # Unsqueeze + Tile → [bs, nq, C]
        _uax = F._from_data(f'{n}.uax',
                            np.array([0], dtype=np.int64), is_const=True)
        setattr(self, _uax.name, _uax)
        _tr = F._from_data(f'{n}.tr',
                           np.array([bs, 1, 1], dtype=np.int64),
                           is_const=True)
        setattr(self, _tr.name, _tr)

        _up = F.Unsqueeze(f'{n}.up')
        setattr(self, _up.name, _up)
        qp_3d = _up(qp_2d, _uax)
        setattr(self, qp_3d.name, qp_3d)
        _tp = F.Tile(f'{n}.tp')
        setattr(self, _tp.name, _tp)
        query_pos_bf = _tp(qp_3d, _tr)
        setattr(self, query_pos_bf.name, query_pos_bf)

        _uq = F.Unsqueeze(f'{n}.uq')
        setattr(self, _uq.name, _uq)
        q_3d = _uq(q_2d, _uax)
        setattr(self, q_3d.name, q_3d)
        _tq = F.Tile(f'{n}.tq')
        setattr(self, _tq.name, _tq)
        query_bf = _tq(q_3d, _tr)
        setattr(self, query_bf.name, query_bf)

        # ==============================================================
        # 7. Decoder reference points: Linear(256,2) + sigmoid
        # ==============================================================
        ref_pts = self.reference_points_fc(query_pos_bf)   # [bs, nq, 2]
        setattr(self, ref_pts.name, ref_pts)
        ref_pts = self.ref_pts_sigmoid(ref_pts)
        setattr(self, ref_pts.name, ref_pts)
        init_reference_out = ref_pts

        # ==============================================================
        # 8. Permute query / memory / query_pos to (S, B, D) for decoder
        # ==============================================================
        _pq = F.Transpose(f'{n}.pq', perm=[1, 0, 2])
        setattr(self, _pq.name, _pq)
        query_sf = _pq(query_bf)                   # [nq, bs, C]
        setattr(self, query_sf.name, query_sf)

        _pm2 = F.Transpose(f'{n}.pm2', perm=[1, 0, 2])
        setattr(self, _pm2.name, _pm2)
        memory_sf = _pm2(memory_bf)                # [S, bs, C]
        setattr(self, memory_sf.name, memory_sf)

        _pqp = F.Transpose(f'{n}.pqp', perm=[1, 0, 2])
        setattr(self, _pqp.name, _pqp)
        query_pos_sf = _pqp(query_pos_bf)          # [nq, bs, C]
        setattr(self, query_pos_sf.name, query_pos_sf)

        # ==============================================================
        # 9. Decoder loop with iterative reference point refinement
        # ==============================================================
        output = query_sf
        reference_points = ref_pts                 # [bs, nq, 2]
        intermediate = []
        intermediate_reference_points = []

        # Valid ratios constant  [bs, nl, 2]
        vr_t = F._from_data(f'{n}.vr', vr_np, is_const=True)
        setattr(self, vr_t.name, vr_t)

        for lid in range(self.decoder_num_layers):
            layer = self.decoder_layers[lid]
            rp_dim = reference_points.shape[-1]

            # ---- Compute reference_points_input ----
            if rp_dim == 2:
                # ref [bs,nq,2] → [bs,nq,1,2]
                _rua = F._from_data(f'{n}.drua_{lid}',
                                    np.array([2], dtype=np.int64),
                                    is_const=True)
                setattr(self, _rua.name, _rua)
                _ru = F.Unsqueeze(f'{n}.dru_{lid}')
                setattr(self, _ru.name, _ru)
                rp_exp = _ru(reference_points, _rua)
                setattr(self, rp_exp.name, rp_exp)

                # vr [bs,nl,2] → [bs,1,nl,2]
                _vua = F._from_data(f'{n}.dvua_{lid}',
                                    np.array([1], dtype=np.int64),
                                    is_const=True)
                setattr(self, _vua.name, _vua)
                _vu = F.Unsqueeze(f'{n}.dvu_{lid}')
                setattr(self, _vu.name, _vu)
                vr_exp = _vu(vr_t, _vua)
                setattr(self, vr_exp.name, vr_exp)

                _rm = F.Mul(f'{n}.drm_{lid}')
                setattr(self, _rm.name, _rm)
                ref_pts_input = _rm(rp_exp, vr_exp)   # [bs,nq,nl,2]
                setattr(self, ref_pts_input.name, ref_pts_input)

            else:  # rp_dim == 4
                # ref [bs,nq,4] → [bs,nq,1,4]
                _rua4 = F._from_data(f'{n}.drua4_{lid}',
                                     np.array([2], dtype=np.int64),
                                     is_const=True)
                setattr(self, _rua4.name, _rua4)
                _ru4 = F.Unsqueeze(f'{n}.dru4_{lid}')
                setattr(self, _ru4.name, _ru4)
                rp_exp = _ru4(reference_points, _rua4)
                setattr(self, rp_exp.name, rp_exp)

                # cat([vr, vr], -1) → [bs,nl,4]
                _vc = F.ConcatX(f'{n}.dvc_{lid}', axis=-1)
                setattr(self, _vc.name, _vc)
                vr4 = _vc(vr_t, vr_t)
                setattr(self, vr4.name, vr4)

                # [bs,nl,4] → [bs,1,nl,4]
                _vua4 = F._from_data(f'{n}.dvua4_{lid}',
                                     np.array([1], dtype=np.int64),
                                     is_const=True)
                setattr(self, _vua4.name, _vua4)
                _vu4 = F.Unsqueeze(f'{n}.dvu4_{lid}')
                setattr(self, _vu4.name, _vu4)
                vr4_exp = _vu4(vr4, _vua4)
                setattr(self, vr4_exp.name, vr4_exp)

                _rm4 = F.Mul(f'{n}.drm4_{lid}')
                setattr(self, _rm4.name, _rm4)
                ref_pts_input = _rm4(rp_exp, vr4_exp)  # [bs,nq,nl,4]
                setattr(self, ref_pts_input.name, ref_pts_input)

            # ---- Run decoder layer ----
            output = layer(
                output,
                key=None,
                value=memory_sf,
                query_pos=query_pos_sf,
                key_padding_mask=mask_flatten,
                reference_points=ref_pts_input,
                spatial_shapes=spatial_shapes,
                level_start_index=lsi_np,
            )

            # ---- Permute to (B, nq, D) for reg_branches ----
            _op = F.Transpose(f'{n}.dop_{lid}', perm=[1, 0, 2])
            setattr(self, _op.name, _op)
            output_bf = _op(output)                    # [bs, nq, C]
            setattr(self, output_bf.name, output_bf)

            # ---- Reference point refinement ----
            if reg_branches is not None:
                tmp = reg_branches[lid](output_bf)     # [bs, nq, 4]
                setattr(self, tmp.name, tmp)

                inv_sig = self._inv_sigmoids[lid]

                if rp_dim == 4:
                    # new_ref = sigmoid(tmp + inv_sigmoid(ref))
                    inv_ref = inv_sig(reference_points)
                    setattr(self, inv_ref.name, inv_ref)

                    _ar = F.Add(f'{n}.dar_{lid}')
                    setattr(self, _ar.name, _ar)
                    nr_pre = _ar(tmp, inv_ref)
                    setattr(self, nr_pre.name, nr_pre)

                    _sr = F.Sigmoid(f'{n}.dsr_{lid}')
                    setattr(self, _sr.name, _sr)
                    reference_points = _sr(nr_pre)
                    setattr(self, reference_points.name, reference_points)

                else:
                    # ref_pts is 2D, tmp is 4D
                    # new_ref[..., :2] = tmp[:2] + inv_sig(ref)
                    # new_ref[..., 2:] = tmp[2:]
                    # new_ref = sigmoid(new_ref)
                    inv_ref = inv_sig(reference_points)
                    setattr(self, inv_ref.name, inv_ref)

                    out_dim = tmp.shape[-1]
                    _sla = F._from_data(f'{n}.dsla_{lid}',
                                        np.array([-1], dtype=np.int64),
                                        is_const=True)
                    setattr(self, _sla.name, _sla)
                    _slstp = F._from_data(f'{n}.dslstp_{lid}',
                                          np.array([1], dtype=np.int64),
                                          is_const=True)
                    setattr(self, _slstp.name, _slstp)

                    # tmp[..., :2]
                    _sl0 = F._from_data(f'{n}.dsl0_{lid}',
                                        np.array([0], dtype=np.int64),
                                        is_const=True)
                    setattr(self, _sl0.name, _sl0)
                    _sl2 = F._from_data(f'{n}.dsl2_{lid}',
                                        np.array([2], dtype=np.int64),
                                        is_const=True)
                    setattr(self, _sl2.name, _sl2)
                    _sxy = F.SliceF(f'{n}.dsxy_{lid}',
                                    out_shape=[bs, nq, 2])
                    setattr(self, _sxy.name, _sxy)
                    tmp_xy = _sxy(tmp, _sl0, _sl2, _sla, _slstp)
                    setattr(self, tmp_xy.name, tmp_xy)

                    # tmp[..., 2:]
                    _sle = F._from_data(f'{n}.dsle_{lid}',
                                        np.array([out_dim], dtype=np.int64),
                                        is_const=True)
                    setattr(self, _sle.name, _sle)
                    _srst = F.SliceF(f'{n}.dsrst_{lid}',
                                     out_shape=[bs, nq, out_dim - 2])
                    setattr(self, _srst.name, _srst)
                    tmp_rest = _srst(tmp, _sl2, _sle, _sla, _slstp)
                    setattr(self, tmp_rest.name, tmp_rest)

                    # modified_xy = tmp_xy + inv_ref
                    _axy = F.Add(f'{n}.daxy_{lid}')
                    setattr(self, _axy.name, _axy)
                    mod_xy = _axy(tmp_xy, inv_ref)
                    setattr(self, mod_xy.name, mod_xy)

                    # concat → sigmoid
                    _cr = F.ConcatX(f'{n}.dcr_{lid}', axis=-1)
                    setattr(self, _cr.name, _cr)
                    combined = _cr(mod_xy, tmp_rest)
                    setattr(self, combined.name, combined)

                    _sr2 = F.Sigmoid(f'{n}.dsr2_{lid}')
                    setattr(self, _sr2.name, _sr2)
                    reference_points = _sr2(combined)
                    setattr(self, reference_points.name, reference_points)

            # ---- Permute back to (nq, bs, D) ----
            _ob = F.Transpose(f'{n}.dob_{lid}', perm=[1, 0, 2])
            setattr(self, _ob.name, _ob)
            output = _ob(output_bf)                    # [nq, bs, C]
            setattr(self, output.name, output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # ==============================================================
        # 10. Return
        # ==============================================================
        # memory_sf and lvl_pos_sf are in (S,B,D); mask_flatten in (B,S);
        # query_pos_sf in (nq,B,D).  Matches PyTorch return format.
        if self.return_intermediate:
            return ((memory_sf, lvl_pos_sf, mask_flatten, query_pos_sf),
                    intermediate, init_reference_out,
                    intermediate_reference_points, None, None)
        else:
            return ((memory_sf, lvl_pos_sf, mask_flatten, query_pos_sf),
                    output, init_reference_out,
                    reference_points, None, None)


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("SegDeformableTransformer TTSim Module  (self-test)")
    logger.info("=" * 70)

    # Encoder config (from fusion_base_e2e.py)
    encoder_cfg = dict(
        type='DetrTransformerEncoder',
        num_layers=6,
        transformerlayers=dict(
            type='BaseTransformerLayer',
            attn_cfgs=dict(
                type='MultiScaleDeformableAttention',
                embed_dims=256),
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'ffn', 'norm')),
    )

    # Decoder config
    decoder_cfg = dict(
        type='DeformableDetrTransformerDecoder',
        num_layers=6,
        return_intermediate=True,
        transformerlayers=dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=256, num_heads=8, dropout=0.1),
                dict(type='MultiScaleDeformableAttention',
                     embed_dims=256),
            ],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn',
                             'norm', 'ffn', 'norm'),
        ),
    )

    try:
        transformer = SegDeformableTransformer(
            name='seg_tr',
            encoder_cfg=encoder_cfg,
            decoder_cfg=decoder_cfg,
            num_feature_levels=4,
            embed_dims=256,
        )
        logger.info("\n[OK] SegDeformableTransformer constructed")
        logger.debug(f"  Encoder layers: {transformer.encoder_num_layers}")
        logger.debug(f"  Decoder layers: {transformer.decoder_num_layers}")
        logger.debug(f"  Return intermediate: {transformer.return_intermediate}")
        logger.debug(f"  Feature levels: {transformer.num_feature_levels}")
        logger.debug(f"  Embed dims: {transformer.embed_dims}")
    except Exception as e:
        logger.info(f"\n[X] Construction failed: {e}")
        import traceback
        traceback.print_exc()
