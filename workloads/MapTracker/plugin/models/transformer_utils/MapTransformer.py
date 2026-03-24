#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of MapTransformer for MapTracker.

This module implements the transformer architecture used in MapTracker for
vector map learning. It includes the decoder and custom transformer layers
for processing map queries with temporal information.

Original: maptracker/plugin/models/transformer_utils/MapTransformer.py
"""

# -------------------------------PyTorch--------------------------------

# # Copyright (c) OpenMMLab. All rights reserved.
# import math
# import warnings
# import copy
#
# import torch
# import torch.nn as nn
# from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
# from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer)
# from mmcv.runner.base_module import BaseModule, ModuleList
#
# from mmdet.models.utils.builder import TRANSFORMER
#
# from mmdet.models.utils.transformer import Transformer
#
# from .CustomMSDeformableAttention import CustomMSDeformableAttention
# from mmdet.models.utils.transformer import inverse_sigmoid
#
#
# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class MapTransformerDecoder_new(BaseModule):
#     """Implements the decoder in DETR transformer.
#     Args:
#         return_intermediate (bool): Whether to return intermediate outputs.
#         coder_norm_cfg (dict): Config of last normalization layer. Default:
#             `LN`.
#     """
#
#     def __init__(self,
#                  transformerlayers=None,
#                  num_layers=None,
#                  prop_add_stage=0,
#                  return_intermediate=True,
#                  init_cfg=None):
#
#         super().__init__(init_cfg)
#         if isinstance(transformerlayers, dict):
#             transformerlayers = [
#                 copy.deepcopy(transformerlayers) for _ in range(num_layers)
#             ]
#         else:
#             assert isinstance(transformerlayers, list) and \
#                    len(transformerlayers) == num_layers
#         self.num_layers = num_layers
#         self.layers = ModuleList()
#         for i in range(num_layers):
#             self.layers.append(build_transformer_layer(transformerlayers[i]))
#         self.embed_dims = self.layers[0].embed_dims
#         self.pre_norm = self.layers[0].pre_norm
#         self.return_intermediate = return_intermediate
#         self.prop_add_stage = prop_add_stage
#         assert prop_add_stage >= 0  and prop_add_stage < num_layers
#
#     def forward(self,
#                 query,
#                 key,
#                 value,
#                 query_pos,
#                 key_padding_mask,
#                 query_key_padding_mask,
#                 reference_points,
#                 spatial_shapes,
#                 level_start_index,
#                 reg_branches,
#                 cls_branches,
#                 predict_refine,
#                 memory_bank=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoder`.
#         Args:
#             query (Tensor): Input query with shape
#                 `(num_query, bs, embed_dims)`.
#             reference_points (Tensor): The reference
#                 points of offset. has shape (bs, num_query, num_points, 2).
#             valid_ratios (Tensor): The radios of valid
#                 points on the feature map, has shape
#                 (bs, num_levels, 2)
#             reg_branch: (obj:`nn.ModuleList`): Used for
#                 refining the regression results. Only would
#                 be passed when with_box_refine is True,
#                 otherwise would be passed a `None`.
#         Returns:
#             Tensor: Results with shape [1, num_query, bs, embed_dims] when
#                 return_intermediate is `False`, otherwise it has shape
#                 [num_layers, num_query, bs, embed_dims].
#         """
#         num_queries, bs, embed_dims = query.shape
#         output = query
#         intermediate = []
#         intermediate_reference_points = []
#
#         for lid, layer in enumerate(self.layers):
#             tmp = reference_points.clone()
#             tmp[..., 1:2] = 1.0 - reference_points[..., 1:2] # reverse y-axis
#
#             output = layer(
#                 output,
#                 key,
#                 value,
#                 query_pos=query_pos,
#                 key_padding_mask=key_padding_mask,
#                 reference_points=tmp,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 query_key_padding_mask=query_key_padding_mask,
#                 memory_bank=memory_bank,
#                 **kwargs)
#
#             reg_points = reg_branches[lid](output.permute(1, 0, 2)) # (bs, num_q, 2*num_points)
#             bs, num_queries, num_points2 = reg_points.shape
#             reg_points = reg_points.view(bs, num_queries, num_points2//2, 2) # range (0, 1)
#
#             if predict_refine:
#                 new_reference_points = reg_points + inverse_sigmoid(
#                     reference_points
#                 )
#                 new_reference_points = new_reference_points.sigmoid()
#             else:
#                 new_reference_points = reg_points.sigmoid() # (bs, num_q, num_points, 2)
#
#             reference_points = new_reference_points.clone().detach()
#
#             if self.return_intermediate:
#                 intermediate.append(output.permute(1, 0, 2)) # [(bs, num_q, embed_dims)]
#                 intermediate_reference_points.append(new_reference_points) # (bs, num_q, num_points, 2)
#
#         if self.return_intermediate:
#             return intermediate, intermediate_reference_points
#
#         return output, reference_points
#
# @TRANSFORMER_LAYER.register_module()
# class MapTransformerLayer(BaseTransformerLayer):
#     """Base `TransformerLayer` for vision transformer.
#
#     It can be built from `mmcv.ConfigDict` and support more flexible
#     customization, for example, using any number of `FFN or LN ` and
#     use different kinds of `attention` by specifying a list of `ConfigDict`
#     named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
#     when you specifying `norm` as the first element of `operation_order`.
#     More details about the `prenorm`: `On Layer Normalization in the
#     Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
#
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for `self_attention` or `cross_attention` modules,
#             The order of the configs in the list should be consistent with
#             corresponding attentions in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config. Default: None.
#         ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for FFN, The order of the configs in the list should be
#             consistent with corresponding ffn in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Support `prenorm` when you specifying first element as `norm`.
#             Default：None.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): Key, Query and Value are shape
#             of (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default to False.
#     """
#
#     def __init__(self,
#                  attn_cfgs=None,
#                  ffn_cfgs=dict(
#                      type='FFN',
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True),
#                  ),
#                  operation_order=None,
#                  norm_cfg=dict(type='LN'),
#                  init_cfg=None,
#                  batch_first=False,
#                  **kwargs):
#
#         super().__init__(
#             attn_cfgs=attn_cfgs,
#             ffn_cfgs=ffn_cfgs,
#             operation_order=operation_order,
#             norm_cfg=norm_cfg,
#             init_cfg=init_cfg,
#             batch_first=batch_first,
#             **kwargs
#         )
#
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 memory_query=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 memory_bank=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoderLayer`.
#
#         **kwargs contains some specific arguments of attentions.
#
#         Args:
#             query (Tensor): The input query with shape
#                 [num_queries, bs, embed_dims] if
#                 self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#             value (Tensor): The value tensor with same shape as `key`.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): 2D Tensor used in
#                 calculation of corresponding attention. The length of
#                 it should equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_queries]. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_keys]. Default: None.
#
#         Returns:
#             Tensor: forwarded results with shape [num_queries, bs, embed_dims].
#         """
#
#         if memory_bank is not None:
#             bs = query.shape[1]
#             all_valid_track_idx = []
#             for b_i in range(bs):
#                 all_valid_track_idx.append(memory_bank.valid_track_idx[b_i])
#
#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [
#                 copy.deepcopy(attn_masks) for _ in range(self.num_attn)
#             ]
#             warnings.warn(f'Use same attn_mask in all attentions in '
#                           f'{self.__class__.__name__} ')
#         else:
#             assert len(attn_masks) == self.num_attn, f'The length of ' \
#                         f'attn_masks {len(attn_masks)} must be equal ' \
#                         f'to the number of attention in ' \
#                         f'operation_order {self.num_attn}'
#
#         for layer in self.operation_order:
#             if layer == 'self_attn':
#                 if memory_query is None:
#                     temp_key = temp_value = query
#                 else:
#                     temp_key = temp_value = torch.cat([memory_query, query], dim=0)
#
#                 query = self.attentions[attn_index](
#                     query,
#                     temp_key,
#                     temp_value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=query_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query
#
#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1
#
#             elif layer == 'cross_attn':
#                 if attn_index == 1:
#                     query_bev = self.attentions[attn_index](
#                         query,
#                         key,
#                         value,
#                         identity if self.pre_norm else None,
#                         query_pos=query_pos,
#                         key_pos=key_pos,
#                         attn_mask=attn_masks[attn_index],
#                         key_padding_mask=key_padding_mask,
#                         **kwargs)
#                     attn_index += 1
#                 else:
#                     # Memory cross attention
#                     assert attn_index == 2
#                     if memory_bank is not None:
#                         bs = query.shape[1]
#                         query_i_list = []
#                         for b_i in range(bs):
#                             valid_track_idx = all_valid_track_idx[b_i]
#                             query_i = query[:, b_i].clone()
#                             query_i = query_i[None,:]
#                             if len(valid_track_idx) != 0:
#                                 mem_embeds = memory_bank.batch_mem_embeds_dict[b_i][:, valid_track_idx, :]
#                                 mem_key_padding_mask = memory_bank.batch_key_padding_dict[b_i][valid_track_idx]
#                                 mem_key_pos = memory_bank.batch_mem_relative_pe_dict[b_i][:, valid_track_idx]
#
#                                 query_i[:, valid_track_idx] = self.attentions[attn_index](
#                                         query_i[:,valid_track_idx],
#                                         mem_embeds,
#                                         mem_embeds,
#                                         identity=None,
#                                         query_pos=None,
#                                         key_pos=mem_key_pos,
#                                         attn_mask=None,
#                                         key_padding_mask=mem_key_padding_mask,
#                                         **kwargs)
#
#                             query_i_list.append(query_i[0])
#                         query_memory = torch.stack(query_i_list).permute(1, 0, 2)
#                     else:
#                         query_memory = torch.zeros_like(query_bev)
#
#                     query = query_memory + query_bev
#                     identity = query
#                     attn_index += 1
#
#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1
#
#         return query
#
# @TRANSFORMER.register_module()
# class MapTransformer(Transformer):
#     """Implements the DeformableDETR transformer.
#     Args:
#         as_two_stage (bool): Generate query from encoder features.
#             Default: False.
#         num_feature_levels (int): Number of feature maps from FPN:
#             Default: 4.
#         two_stage_num_proposals (int): Number of proposals when set
#             `as_two_stage` as True. Default: 300.
#     """
#
#     def __init__(self,
#                  num_feature_levels=1,
#                  num_points=20,
#                  coord_dim=2,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.num_feature_levels = num_feature_levels
#         self.embed_dims = self.encoder.embed_dims
#         self.coord_dim = coord_dim
#         self.num_points = num_points
#         self.init_layers()
#
#     def init_layers(self):
#         """Initialize layers of the DeformableDetrTransformer."""
#         # self.level_embeds = nn.Parameter(
#         #     torch.Tensor(self.num_feature_levels, self.embed_dims))
#
#     def init_weights(self):
#         """Initialize the transformer weights."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, CustomMSDeformableAttention):
#                 m.init_weights()
#
#     def forward(self,
#                 mlvl_feats,
#                 mlvl_masks,
#                 query_embed,
#                 mlvl_pos_embeds,
#                 init_reference_points,
#                 reg_branches=None,
#                 cls_branches=None,
#                 memory_query=None,
#                 memory_bank=None,
#                 **kwargs):
#         """Forward function for `Transformer`.
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
#         Returns:
#             tuple[Tensor]: results of decoder containing the following tensor.
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
#         feat_flatten = []
#         mask_flatten = []
#         # lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (feat, mask, pos_embed) in enumerate(
#                 zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
#             bs, c, h, w = feat.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             feat = feat.flatten(2).transpose(1, 2)
#             mask = mask.flatten(1)
#             feat_flatten.append(feat)
#             mask_flatten.append(mask)
#         feat_flatten = torch.cat(feat_flatten, 1)
#         mask_flatten = torch.cat(mask_flatten, 1)
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=feat_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros(
#             (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
#
#         feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
#
#         # decoder
#         query = query_embed.permute(1, 0, 2) # (num_q, bs, embed_dims)
#         if memory_query is not None:
#             memory_query = memory_query.permute(1, 0, 2)
#
#         inter_states, inter_references = self.decoder(
#             query=query,
#             key=None,
#             value=feat_flatten,
#             query_pos=None,
#             key_padding_mask=mask_flatten,
#             reference_points=init_reference_points,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             reg_branches=reg_branches,
#             cls_branches=cls_branches,
#             memory_query=memory_query,
#             memory_bank=memory_bank,
#             **kwargs)
#
#         return inter_states, init_reference_points, inter_references

# -------------------------------TTSIM-----------------------------------


import sys
import os

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import copy
import warnings
import numpy as np
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F


def inverse_sigmoid(x, name_prefix="inverse_sigmoid", module=None):
    """
    Inverse sigmoid function for coordinate refinement.

    Args:
        x: Input tensor with values in range (0, 1)
        name_prefix: Prefix for operator names
        module: Optional module instance to track operation handles

    Returns:
        Tensor with inverse sigmoid applied
    """
    # Clip to prevent log(0)
    eps = 1e-5

    # Create min and max constants
    min_val = F._from_data(
        f"{name_prefix}_min", np.array([eps], dtype=np.float32), is_const=True
    )
    max_val = F._from_data(
        f"{name_prefix}_max", np.array([1.0 - eps], dtype=np.float32), is_const=True
    )
    if module is not None:
        module._tensors[min_val.name] = min_val
        module._tensors[max_val.name] = max_val

    # Clip: max(min_val, min(max_val, x))
    clip_op = F.Clip(f"{name_prefix}_clip")
    x_clipped = clip_op(x, min_val, max_val)
    if module is not None:
        module._op_hndls[clip_op.name] = clip_op
        module._tensors[x_clipped.name] = x_clipped

    # log(x / (1 - x))
    one = F._from_data(
        f"{name_prefix}_one", np.array([1.0], dtype=np.float32), is_const=True
    )
    if module is not None:
        module._tensors[one.name] = one
    sub_op = F.Sub(f"{name_prefix}_1_minus_x")
    one_minus_x = sub_op(one, x_clipped)
    if module is not None:
        module._op_hndls[sub_op.name] = sub_op
        module._tensors[one_minus_x.name] = one_minus_x

    div_op = F.Div(f"{name_prefix}_ratio")
    ratio = div_op(x_clipped, one_minus_x)
    if module is not None:
        module._op_hndls[div_op.name] = div_op
        module._tensors[ratio.name] = ratio

    log_op = F.Log(f"{name_prefix}_log")
    result = log_op(ratio)
    if module is not None:
        module._op_hndls[log_op.name] = log_op
        module._tensors[result.name] = result

    return result


class MapTransformerDecoder_new(SimNN.Module):
    """
    TTSim implementation of MapTransformer decoder.

    Implements the decoder in DETR-style transformer with iterative refinement
    for map element predictions (lane lines, boundaries, etc.).

    Args:
        transformerlayers: Configuration for transformer layers
        num_layers: Number of decoder layers
        prop_add_stage: Stage at which to add propagated queries (default: 0)
        return_intermediate: Whether to return intermediate layer outputs
        init_cfg: Initialization config (not used in ttsim)
    """

    def __init__(
        self,
        transformerlayers=None,
        num_layers=None,
        prop_add_stage=0,
        return_intermediate=True,
        init_cfg=None,
    ):

        super().__init__()
        self.name = "map_transformer_decoder"
        self._tensors = {}  # Track all intermediate tensors

        # Note: In ttsim, we expect layers to be built externally and set later
        # This is because build_transformer_layer is an mmcv function
        self.num_layers = num_layers
        self.layers = []  # Will be populated with transformer layers
        self.embed_dims = None  # Will be set when layers are added
        self.pre_norm = None  # Will be set when layers are added
        self.return_intermediate = return_intermediate
        self.prop_add_stage = prop_add_stage

        if num_layers is not None:
            assert prop_add_stage >= 0 and prop_add_stage < num_layers

    def add_layer(self, layer):
        """
        Add a transformer layer to the decoder.

        Args:
            layer: A transformer layer module
        """
        # Finalize the layer's internal lists before adding
        layer.finalize()
        self.layers.append(layer)
        if self.embed_dims is None and hasattr(layer, "embed_dims"):
            self.embed_dims = layer.embed_dims
        if self.pre_norm is None and hasattr(layer, "pre_norm"):
            self.pre_norm = layer.pre_norm

        # Convert to ModuleList and link after all layers are added
        if len(self.layers) == self.num_layers:
            self.layers = SimNN.ModuleList(self.layers)  # type: ignore[assignment]
            super().link_op2module()

    def __call__(
        self,
        query,
        key,
        value,
        query_pos,
        key_padding_mask,
        query_key_padding_mask,
        reference_points,
        spatial_shapes,
        level_start_index,
        reg_branches,
        cls_branches,
        predict_refine,
        memory_bank=None,
        **kwargs,
    ):
        """
        Forward pass through the transformer decoder.

        Args:
            query: Input query [num_query, bs, embed_dims]
            key: Key tensor (can be None)
            value: Value tensor for cross-attention
            query_pos: Positional encoding for queries
            key_padding_mask: Padding mask for keys
            query_key_padding_mask: Padding mask for queries
            reference_points: Reference points [bs, num_query, num_points, 2]
            spatial_shapes: Spatial dimensions of feature maps
            level_start_index: Starting indices for each level
            reg_branches: Regression head modules for refinement
            cls_branches: Classification head modules
            predict_refine: Whether to refine predictions iteratively
            memory_bank: Memory bank for temporal information (optional)

        Returns:
            intermediate: List of intermediate outputs (if return_intermediate=True)
            intermediate_reference_points: List of refined reference points
        """
        # Get dimensions
        if isinstance(query.shape, tuple):
            num_queries, bs, embed_dims = query.shape
        else:
            num_queries = query.shape[0]
            bs = query.shape[1]
            embed_dims = query.shape[2]

        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Reverse y-axis for reference points (coordinate convention)
            # Get shape dynamically from reference_points
            bs = reference_points.shape[0] if len(reference_points.shape) > 0 else 1
            num_query = (
                reference_points.shape[1] if len(reference_points.shape) > 1 else 1
            )
            num_points_per_query = (
                reference_points.shape[2] if len(reference_points.shape) > 2 else 1
            )

            # Slice x coordinate [bs, num_query, num_points, 1]
            x_starts = F._from_data(
                f"{self.name}_x_starts_l{lid}",
                np.array([0, 0, 0, 0], dtype=np.int32),
                is_const=True,
            )
            setattr(self, x_starts.name, x_starts)
            x_ends = F._from_data(
                f"{self.name}_x_ends_l{lid}",
                np.array([bs, num_query, num_points_per_query, 1], dtype=np.int32),
                is_const=True,
            )
            setattr(self, x_ends.name, x_ends)
            x_axes = F._from_data(
                f"{self.name}_x_axes_l{lid}",
                np.array([0, 1, 2, 3], dtype=np.int32),
                is_const=True,
            )
            setattr(self, x_axes.name, x_axes)
            x_steps = F._from_data(
                f"{self.name}_x_steps_l{lid}",
                np.array([1, 1, 1, 1], dtype=np.int32),
                is_const=True,
            )
            setattr(self, x_steps.name, x_steps)
            x_slice_op = F.SliceF(
                f"{self.name}_x_slice_l{lid}",
                out_shape=[bs, num_query, num_points_per_query, 1],
            )
            x_coords = x_slice_op(reference_points, x_starts, x_ends, x_axes, x_steps)
            setattr(self, x_slice_op.name, x_slice_op)
            setattr(self, x_coords.name, x_coords)

            # Slice y coordinate [bs, num_query, num_points, 1]
            y_starts = F._from_data(
                f"{self.name}_y_starts_l{lid}",
                np.array([0, 0, 0, 1], dtype=np.int32),
                is_const=True,
            )
            setattr(self, y_starts.name, y_starts)
            y_ends = F._from_data(
                f"{self.name}_y_ends_l{lid}",
                np.array([bs, num_query, num_points_per_query, 2], dtype=np.int32),
                is_const=True,
            )
            setattr(self, y_ends.name, y_ends)
            y_axes = F._from_data(
                f"{self.name}_y_axes_l{lid}",
                np.array([0, 1, 2, 3], dtype=np.int32),
                is_const=True,
            )
            setattr(self, y_axes.name, y_axes)
            y_steps = F._from_data(
                f"{self.name}_y_steps_l{lid}",
                np.array([1, 1, 1, 1], dtype=np.int32),
                is_const=True,
            )
            setattr(self, y_steps.name, y_steps)
            y_slice_op = F.SliceF(
                f"{self.name}_y_slice_l{lid}",
                out_shape=[bs, num_query, num_points_per_query, 1],
            )
            y_coords = y_slice_op(reference_points, y_starts, y_ends, y_axes, y_steps)
            setattr(self, y_slice_op.name, y_slice_op)
            setattr(self, y_coords.name, y_coords)

            # Reverse y: 1.0 - y
            one = F._from_data(
                f"{self.name}_one_l{lid}",
                np.array([1.0], dtype=np.float32),
                is_const=True,
            )
            setattr(self, one.name, one)
            sub_op = F.Sub(f"{self.name}_y_rev_l{lid}")
            y_reversed = sub_op(one, y_coords)
            setattr(self, sub_op.name, sub_op)
            setattr(self, y_reversed.name, y_reversed)

            # Concatenate to form [x, y_reversed]
            concat_op = F.ConcatX(f"{self.name}_refpts_reversed_l{lid}", axis=-1)
            tmp = concat_op(x_coords, y_reversed)
            setattr(self, concat_op.name, concat_op)
            setattr(self, tmp.name, tmp)

            # Forward through transformer layer
            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=tmp,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_key_padding_mask=query_key_padding_mask,
                memory_bank=memory_bank,
                **kwargs,
            )
            setattr(self, output.name, output)

            # Apply regression branch to refine reference points
            # output shape: [num_q, bs, embed_dims]
            # Need to permute to [bs, num_q, embed_dims] for regression branch
            transpose_op = F.Transpose(f"{self.name}_out_perm_l{lid}", perm=[1, 0, 2])
            output_permuted = transpose_op(output)
            setattr(self, transpose_op.name, transpose_op)
            setattr(self, output_permuted.name, output_permuted)

            # Apply regression branch if provided
            if reg_branches is not None:
                # reg_branches[lid] is expected to be a callable that takes output and returns offsets
                # Shape: [bs, num_queries, num_points*2]
                reg_points = reg_branches[lid](output_permuted)

                # Reshape from [bs, num_q, num_points*2] to [bs, num_q, num_points, 2]
                reg_points_shape = F._from_data(
                    f"{self.name}_reg_shape_l{lid}",
                    np.array([bs, num_queries, -1, 2], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, reg_points_shape.name, reg_points_shape)
                reshape_op = F.Reshape(f"{self.name}_reg_reshape_l{lid}")
                reg_points = reshape_op(reg_points, reg_points_shape)
                setattr(self, reshape_op.name, reshape_op)
                setattr(self, reg_points.name, reg_points)

                # Refine reference points
                # Check if predict_refine is enabled (assuming it's passed in kwargs or decoder attribute)
                if hasattr(self, "predict_refine") and self.predict_refine:
                    # Iterative refinement: new = sigmoid(offset + inverse_sigmoid(old))
                    # Convert current reference points to unbounded space
                    inv_sig_ref = inverse_sigmoid(
                        reference_points, f"{self.name}_inv_sig_l{lid}", module=self
                    )
                    setattr(self, inv_sig_ref.name, inv_sig_ref)
                    # Add regression offset
                    add_op = F.Add(f"{self.name}_add_offset_l{lid}")
                    refined_unbounded = add_op(reg_points, inv_sig_ref)
                    setattr(self, add_op.name, add_op)
                    setattr(self, refined_unbounded.name, refined_unbounded)
                    # Convert back to [0,1] range
                    sig_op = F.Sigmoid(f"{self.name}_sig_l{lid}")
                    new_reference_points = sig_op(refined_unbounded)
                    setattr(self, sig_op.name, sig_op)
                    setattr(self, new_reference_points.name, new_reference_points)
                elif predict_refine:  # If passed as argument
                    inv_sig_ref = inverse_sigmoid(
                        reference_points, f"{self.name}_inv_sig_l{lid}", module=self
                    )
                    setattr(self, inv_sig_ref.name, inv_sig_ref)
                    add_op = F.Add(f"{self.name}_add_offset_l{lid}")
                    refined_unbounded = add_op(reg_points, inv_sig_ref)
                    setattr(self, add_op.name, add_op)
                    setattr(self, refined_unbounded.name, refined_unbounded)
                    sig_op = F.Sigmoid(f"{self.name}_sig_l{lid}")
                    new_reference_points = sig_op(refined_unbounded)
                    setattr(self, sig_op.name, sig_op)
                    setattr(self, new_reference_points.name, new_reference_points)
                else:
                    # Direct prediction: new = sigmoid(offset)
                    sig_op = F.Sigmoid(f"{self.name}_sig_l{lid}")
                    new_reference_points = sig_op(reg_points)
                    setattr(self, sig_op.name, sig_op)
                    setattr(self, new_reference_points.name, new_reference_points)

                # Update reference points for next layer
                reference_points = new_reference_points
            else:
                # No regression branch - keep reference points unchanged
                new_reference_points = reference_points

            # Store intermediate results
            if self.return_intermediate:
                intermediate.append(output_permuted)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return output, reference_points

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for this decoder.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            print(f"{indent}MapTransformerDecoder_new '{self.name}':")

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "analytical_param_count"):
                layer_params = layer.analytical_param_count(
                    lvl=lvl + 1 if lvl >= 2 else 0
                )
                total_params += layer_params
                if lvl >= 2:
                    print(f"{indent}  Layer {i}: {layer_params:,}")

        if lvl >= 1:
            print(f"{indent}Total MapTransformerDecoder_new params: {total_params:,}")

        return total_params


class MapTransformerLayer(SimNN.Module):
    """
    TTSim implementation of MapTransformer layer.

    Custom transformer layer with support for:
    - Self-attention among queries
    - Cross-attention to BEV features
    - Cross-attention to memory bank (for temporal tracking)
    - Feed-forward networks
    - Layer normalization

    This is a simplified version that assumes the layer components
    (attentions, norms, ffns) are built externally.
    """

    def __init__(
        self,
        embed_dims=256,
        num_attn=3,
        num_ffn=1,
        pre_norm=False,
        operation_order=None,
    ):
        super().__init__()
        self.name = "map_transformer_layer"
        self._tensors = {}  # Track all intermediate tensors
        self._call_count = 0  # Incremented each forward call for unique tensor names
        self.embed_dims = embed_dims
        self.num_attn = num_attn
        self.num_ffn = num_ffn
        self.pre_norm = pre_norm

        # Operation order: e.g., ('self_attn', 'norm', 'cross_attn', 'norm',
        #                         'cross_attn', 'norm', 'ffn', 'norm')
        self.operation_order = operation_order or [
            "self_attn",
            "norm",
            "cross_attn",
            "norm",
            "cross_attn",
            "norm",
            "ffn",
            "norm",
        ]

        # These will be populated externally
        self._attentions_list = []
        self._norms_list = []
        self._ffns_list = []

        # Don't link yet - finalize() will be called after all submodules are added

    def add_attention(self, attn_module):
        """Add an attention module."""
        self._attentions_list.append(attn_module)

    def add_norm(self, norm_module):
        """Add a normalization module."""
        self._norms_list.append(norm_module)

    def add_ffn(self, ffn_module):
        """Add a feed-forward network module."""
        self._ffns_list.append(ffn_module)

    def finalize(self):
        """Convert internal lists to ModuleList and link ops."""
        self.attentions = SimNN.ModuleList(self._attentions_list)
        self.norms = SimNN.ModuleList(self._norms_list)
        self.ffns = SimNN.ModuleList(self._ffns_list)
        super().link_op2module()

    def __call__(
        self,
        query,
        key=None,
        value=None,
        memory_query=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        memory_bank=None,
        **kwargs,
    ):
        """
        Forward pass through the transformer layer.

        Args:
            query: Input queries [num_queries, bs, embed_dims]
            key: Key tensor for cross-attention
            value: Value tensor for cross-attention
            memory_query: Memory queries for self-attention extension
            query_pos: Positional encoding for queries
            key_pos: Positional encoding for keys
            attn_masks: Attention masks (list or None)
            query_key_padding_mask: Padding mask for queries
            key_padding_mask: Padding mask for keys
            memory_bank: Memory bank for temporal tracking

        Returns:
            query: Processed query tensor [num_queries, bs, embed_dims]
        """

        # Unique prefix per call so tensor names don't collide across frames
        self._call_count += 1
        _cc = self._call_count
        pfx = f"{self.name}_c{_cc}"

        # Handle memory bank valid indices if present
        if memory_bank is not None:
            bs = query.shape[1]
            all_valid_track_idx = []
            for b_i in range(bs):
                all_valid_track_idx.append(memory_bank.valid_track_idx[b_i])

        # Initialize counters
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Process attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]

        # Process through operation sequence
        for layer_op in self.operation_order:
            if layer_op == "self_attn":
                # Self-attention (possibly extended with memory queries)
                if memory_query is None:
                    temp_key = temp_value = query
                else:
                    # Concatenate memory queries with current queries
                    _concat_op = F.ConcatX(f"{self.name}_self_attn_concat", axis=0)
                    setattr(self, _concat_op.name, _concat_op)
                    temp_key = temp_value = _concat_op(memory_query, query)
                    setattr(self, temp_key.name, temp_key)

                # Apply self-attention
                # Note: Actual attention computation handled by attention module
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else query,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                setattr(self, query.name, query)
                attn_index += 1
                identity = query

            elif layer_op == "norm":
                # Layer normalization
                query = self.norms[norm_index](query)
                setattr(self, query.name, query)
                # Set link_module so tensor indexing (e.g. query[:, b_i]) works
                query.set_module(self)
                norm_index += 1

            elif layer_op == "cross_attn":
                if attn_index == 1:
                    # First cross-attention: to BEV features
                    # NOTE: Store result in query_bev (not query) to match PyTorch.
                    # query is left unchanged so the next norm operates on the
                    # pre-BEV query (matching PyTorch flow). query_bev is fused
                    # with query_memory after the memory cross-attention.
                    query_bev = self.attentions[attn_index](
                        query,
                        key,
                        value,
                        identity if self.pre_norm else query,
                        query_pos=query_pos,
                        key_pos=key_pos,
                        attn_mask=attn_masks[attn_index],
                        key_padding_mask=key_padding_mask,
                        **kwargs,
                    )
                    setattr(self, query_bev.name, query_bev)
                    attn_index += 1
                else:
                    # Second cross-attention: to memory bank (temporal)
                    assert attn_index == 2
                    if memory_bank is not None:
                        bs = query.shape[1]
                        query_i_list = []

                        for b_i in range(bs):
                            valid_track_idx = all_valid_track_idx[b_i]
                            if query.data is not None:
                                query_np_bi = np.asarray(
                                    query.data[:, b_i : b_i + 1, :]
                                ).copy()
                            else:
                                # Graph-build phase: .data not yet available, use zeros with correct shape
                                num_q = query.shape[0]
                                embed_d = query.shape[2]
                                query_np_bi = np.zeros(
                                    (num_q, 1, embed_d), dtype=np.float32
                                )
                            query_np_bi = np.transpose(query_np_bi, (1, 0, 2))
                            query_i = F._from_data(
                                f"{pfx}_query_b{b_i}", query_np_bi, is_const=False
                            )
                            setattr(self, query_i.name, query_i)

                            if len(valid_track_idx) != 0:
                                mem_embeds_np = memory_bank.batch_mem_embeds_dict[b_i][
                                    :, valid_track_idx, :
                                ]
                                mem_key_padding_mask_np = (
                                    memory_bank.batch_key_padding_dict[b_i][
                                        valid_track_idx
                                    ]
                                )
                                pe_dict_val = memory_bank.batch_mem_relative_pe_dict[
                                    b_i
                                ]

                                # Transpose from [mem_len, n_tracks, C] to [n_tracks, mem_len, C]
                                # so MHA (batch_first=True) treats each track as a batch element
                                mem_embeds_np = np.transpose(mem_embeds_np, (1, 0, 2))

                                mem_embeds = F._from_data(
                                    f"{pfx}_mem_embeds_b{b_i}",
                                    mem_embeds_np.astype(np.float32),
                                    is_const=False,
                                )
                                setattr(self, mem_embeds.name, mem_embeds)

                                # Handle mem_key_pos: use SimTensor if available (graph-connected PE)
                                if hasattr(pe_dict_val, "op_in"):
                                    # pe_dict_val is a SimTensor [mem_len, n_tracks, C]
                                    # Transpose to [n_tracks, mem_len, C] using TTSim
                                    _pe_transpose = F.Transpose(
                                        f"{pfx}_mem_pe_transpose_b{b_i}", perm=[1, 0, 2]
                                    )
                                    setattr(self, _pe_transpose.name, _pe_transpose)
                                    mem_key_pos = _pe_transpose(pe_dict_val)
                                    setattr(self, mem_key_pos.name, mem_key_pos)
                                else:
                                    mem_key_pos_np = pe_dict_val[:, valid_track_idx]
                                    mem_key_pos_np = np.transpose(
                                        mem_key_pos_np, (1, 0, 2)
                                    )
                                    mem_key_pos = F._from_data(
                                        f"{pfx}_mem_key_pos_b{b_i}",
                                        mem_key_pos_np.astype(np.float32),
                                        is_const=False,
                                    )
                                    setattr(self, mem_key_pos.name, mem_key_pos)
                                mem_key_padding_mask = F._from_data(
                                    f"{pfx}_mem_key_mask_b{b_i}",
                                    mem_key_padding_mask_np.astype(np.float32),
                                    is_const=False,
                                )
                                setattr(
                                    self,
                                    mem_key_padding_mask.name,
                                    mem_key_padding_mask,
                                )

                                gather_idx = F._from_data(
                                    f"{pfx}_mem_gather_idx_b{b_i}",
                                    np.array(valid_track_idx, dtype=np.int64),
                                    is_const=True,
                                )
                                setattr(self, gather_idx.name, gather_idx)
                                _gather_op = F.Gather(
                                    f"{pfx}_mem_gather_b{b_i}", axis=1
                                )
                                setattr(self, _gather_op.name, _gather_op)
                                track_queries = _gather_op(query_i, gather_idx)
                                setattr(self, track_queries.name, track_queries)

                                # Transpose track_queries from [1, n_tracks, C] to [n_tracks, 1, C]
                                # so each track is a separate batch element querying its own memory
                                _tq_transpose = F.Transpose(
                                    f"{pfx}_tq_transpose_b{b_i}", perm=[1, 0, 2]
                                )
                                setattr(self, _tq_transpose.name, _tq_transpose)
                                track_queries = _tq_transpose(track_queries)
                                setattr(self, track_queries.name, track_queries)

                                attended_tracks = self.attentions[attn_index](
                                    track_queries,
                                    mem_embeds,
                                    mem_embeds,
                                    identity=None,
                                    query_pos=None,
                                    key_pos=mem_key_pos,
                                    attn_mask=None,
                                    key_padding_mask=mem_key_padding_mask,
                                    **kwargs,
                                )

                                # Transpose attended_tracks back from [n_tracks, 1, C] to [1, n_tracks, C]
                                _at_transpose = F.Transpose(
                                    f"{pfx}_at_transpose_b{b_i}", perm=[1, 0, 2]
                                )
                                setattr(self, _at_transpose.name, _at_transpose)
                                attended_tracks = _at_transpose(attended_tracks)
                                setattr(self, attended_tracks.name, attended_tracks)

                                scatter_indices_np = np.stack(
                                    [
                                        np.zeros(len(valid_track_idx), dtype=np.int64),
                                        np.array(valid_track_idx, dtype=np.int64),
                                    ],
                                    axis=-1,
                                )
                                scatter_indices = F._from_data(
                                    f"{pfx}_mem_scatter_idx_b{b_i}",
                                    scatter_indices_np,
                                    is_const=True,
                                )
                                setattr(self, scatter_indices.name, scatter_indices)
                                sq_axes_track = F._from_data(
                                    f"{pfx}_mem_sq_axes_b{b_i}",
                                    np.array([0], dtype=np.int64),
                                    is_const=True,
                                )
                                setattr(self, sq_axes_track.name, sq_axes_track)
                                _sq_op = F.Squeeze(f"{pfx}_mem_sq_b{b_i}")
                                setattr(self, _sq_op.name, _sq_op)
                                attended_squeezed = _sq_op(
                                    attended_tracks, sq_axes_track
                                )
                                setattr(self, attended_squeezed.name, attended_squeezed)
                                _scatter_op = F.ScatterND(f"{pfx}_mem_scatter_b{b_i}")
                                setattr(self, _scatter_op.name, _scatter_op)
                                query_i = _scatter_op(
                                    query_i, scatter_indices, attended_squeezed
                                )
                                setattr(self, query_i.name, query_i)

                            sq_axes = F._from_data(
                                f"{pfx}_query_b{b_i}_sq_axes",
                                np.array([0], dtype=np.int64),
                                is_const=True,
                            )
                            setattr(self, sq_axes.name, sq_axes)
                            _sq_op2 = F.Squeeze(f"{pfx}_query_b{b_i}_sq")
                            setattr(self, _sq_op2.name, _sq_op2)
                            query_i_squeezed = _sq_op2(query_i, sq_axes)
                            setattr(self, query_i_squeezed.name, query_i_squeezed)
                            unsq_axes_1 = F._from_data(
                                f"{pfx}_query_b{b_i}_unsq1_axes",
                                np.array([1], dtype=np.int64),
                                is_const=True,
                            )
                            setattr(self, unsq_axes_1.name, unsq_axes_1)
                            _unsq_op = F.Unsqueeze(f"{pfx}_query_b{b_i}_unsq1")
                            setattr(self, _unsq_op.name, _unsq_op)
                            query_i_expanded = _unsq_op(query_i_squeezed, unsq_axes_1)
                            setattr(self, query_i_expanded.name, query_i_expanded)
                            query_i_list.append(query_i_expanded)

                        # Concat along axis 1 to get [num_queries, bs, embed_dim]
                        if len(query_i_list) == 1:
                            query_memory = query_i_list[0]
                        else:
                            _concat_op = F.ConcatX(f"{pfx}_mem_concat", axis=1)
                            setattr(self, _concat_op.name, _concat_op)
                            query_memory = _concat_op(*query_i_list)
                            setattr(self, query_memory.name, query_memory)
                    else:
                        # No memory bank: skip memory attention, use current query as-is
                        query_memory = None

                    # Fuse memory attention result with BEV cross-attention
                    # PyTorch: query = query_memory + query_bev
                    if query_memory is not None:
                        _add_fusion_op = F.Add(f"{pfx}_mem_bev_fusion")
                        setattr(self, _add_fusion_op.name, _add_fusion_op)
                        query = _add_fusion_op(query_memory, query_bev)
                        setattr(self, query.name, query)
                    else:
                        # No memory bank: query_memory = zeros → query = query_bev
                        query = query_bev
                        setattr(self, query.name, query)
                    identity = query
                    attn_index += 1

            elif layer_op == "ffn":
                # Feed-forward network
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                setattr(self, query.name, query)
                ffn_index += 1

        return query

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for this layer.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            print(f"{indent}MapTransformerLayer '{self.name}':")

        # Count attention parameters
        for i, attn in enumerate(self.attentions):
            if hasattr(attn, "analytical_param_count"):
                attn_params = attn.analytical_param_count(lvl=0)
                total_params += attn_params
                if lvl >= 2:
                    print(f"{indent}  Attention {i}: {attn_params:,}")

        # Count FFN parameters
        for i, ffn in enumerate(self.ffns):
            if hasattr(ffn, "analytical_param_count"):
                ffn_params = ffn.analytical_param_count(lvl=0)
                total_params += ffn_params
                if lvl >= 2:
                    print(f"{indent}  FFN {i}: {ffn_params:,}")

        # Count norm parameters (LayerNorm has 2 * embed_dims)
        for i, norm in enumerate(self.norms):
            if hasattr(norm, "analytical_param_count"):
                norm_params = norm.analytical_param_count(lvl=0)
            else:
                norm_params = 2 * self.embed_dims
            total_params += norm_params
            if lvl >= 2:
                print(f"{indent}  Norm {i}: {norm_params:,}")

        if lvl >= 1:
            print(f"{indent}Total MapTransformerLayer params: {total_params:,}")

        return total_params


class MapTransformer(SimNN.Module):
    """
    TTSim implementation of MapTransformer.

    Main transformer architecture for MapTracker, processing BEV features
    with decoder-only structure for vectorized map learning.

    Args:
        name: Module name
        num_feature_levels: Number of feature pyramid levels (default: 1)
        num_points: Number of points per map element (default: 20)
        coord_dim: Coordinate dimensions (default: 2 for 2D)
        embed_dims: Embedding dimension (default: 256)
        num_layers: Number of decoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        ffn_dim: FFN hidden dimension (default: 1024)
        encoder: Encoder module (can be None/placeholder for map tasks)
        decoder: Decoder module (MapTransformerDecoder_new)
    """

    def __init__(
        self,
        name="map_transformer",
        num_feature_levels=1,
        num_points=20,
        coord_dim=2,
        embed_dims=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        encoder=None,
        decoder=None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self._tensors = {}  # Track all intermediate tensors
        self.num_feature_levels = num_feature_levels
        self.coord_dim = coord_dim
        self.num_points = num_points
        self.embed_dims = embed_dims

        self.encoder = encoder

        # Create decoder if not provided
        if decoder is None:
            # Import building blocks
            from ..backbones.bevformer.builder_utils import LayerNorm, FFN
            from .multihead_attention import MultiheadAttention
            from .custom_msdeformable_attention import CustomMSDeformableAttention

            decoder = MapTransformerDecoder_new(
                num_layers=num_layers, return_intermediate=True, prop_add_stage=0
            )

            # Create and add transformer layers with full components
            for i in range(num_layers):
                layer = MapTransformerLayer(
                    embed_dims=embed_dims,
                    num_attn=3,  # Self-attn + 2x cross-attn (BEV + memory)
                    num_ffn=1,
                    pre_norm=False,
                )
                layer.name = f"{self.name}.decoder_layer_{i}"

                # Add self-attention (query-to-query)
                self_attn = MultiheadAttention(
                    name=f"{layer.name}.self_attn",
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    batch_first=False,  # (seq, batch, embed)
                )
                layer.add_attention(self_attn)

                # Add cross-attention to BEV features (deformable)
                cross_attn_bev = CustomMSDeformableAttention(
                    name=f"{layer.name}.cross_attn_bev",
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    num_levels=1,
                    num_points=4,  # Sampling points
                )
                layer.add_attention(cross_attn_bev)

                # Add cross-attention to memory bank (if applicable)
                cross_attn_mem = MultiheadAttention(
                    name=f"{layer.name}.cross_attn_mem",
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    batch_first=True,  # inputs are [n_tracks, seq, C] (batch-first)
                )
                layer.add_attention(cross_attn_mem)

                # Add layer norms (8 total: before/after each attn and ffn)
                for j in range(8):
                    norm = LayerNorm(
                        name=f"{layer.name}.norm{j}", normalized_shape=embed_dims
                    )
                    layer.add_norm(norm)

                # Add FFN
                ffn = FFN(
                    name=f"{layer.name}.ffn",
                    embed_dims=embed_dims,
                    feedforward_channels=ffn_dim,
                    num_fcs=2,
                    add_identity=True,
                )
                layer.add_ffn(ffn)

                decoder.add_layer(layer)

        self.decoder = decoder

        # Update embed_dims from decoder if available
        if self.decoder is not None and hasattr(self.decoder, "embed_dims"):
            if self.decoder.embed_dims is not None:
                self.embed_dims = self.decoder.embed_dims

        super().link_op2module()

    def __call__(
        self,
        mlvl_feats,
        mlvl_masks,
        query_embed,
        mlvl_pos_embeds,
        init_reference_points,
        reg_branches=None,
        cls_branches=None,
        memory_query=None,
        memory_bank=None,
        **kwargs,
    ):
        """
        Forward pass through the transformer.

        Args:
            mlvl_feats: Multi-level features [list of [bs, embed_dims, h, w]]
            mlvl_masks: Multi-level masks [list of [bs, h, w]]
            query_embed: Query embeddings [bs, num_query, embed_dims]
            mlvl_pos_embeds: Multi-level positional embeddings (can be None)
            init_reference_points: Initial reference points [bs, num_query, num_points, 2]
            reg_branches: Regression branches for iterative refinement
            cls_branches: Classification branches
            memory_query: Memory queries from previous frames
            memory_bank: Temporal memory bank

        Returns:
            inter_states: Intermediate decoder states
            init_reference_points: Initial reference points
            inter_references: Intermediate reference points
        """

        # Flatten multi-level features
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []

        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)
        ):
            # Get spatial shape
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            # Flatten spatial dimensions: [bs, c, h, w] -> [bs, c, h*w] -> [bs, h*w, c]
            feat_shape = F._from_data(
                f"{self.name}_feat_shape_l{lvl}",
                np.array([bs, c, h * w], dtype=np.int64),
                is_const=True,
            )
            setattr(self, feat_shape.name, feat_shape)
            _flat_op = F.Reshape(f"{self.name}_feat_flat_l{lvl}")
            setattr(self, _flat_op.name, _flat_op)
            feat_flat = _flat_op(feat, feat_shape)
            setattr(self, feat_flat.name, feat_flat)
            _trans_op = F.Transpose(f"{self.name}_feat_trans_l{lvl}", perm=[0, 2, 1])
            setattr(self, _trans_op.name, _trans_op)
            feat_flat = _trans_op(feat_flat)
            setattr(self, feat_flat.name, feat_flat)

            # Flatten mask: [bs, h, w] -> [bs, h*w]
            mask_shape = F._from_data(
                f"{self.name}_mask_shape_l{lvl}",
                np.array([bs, h * w], dtype=np.int64),
                is_const=True,
            )
            setattr(self, mask_shape.name, mask_shape)
            _mflat_op = F.Reshape(f"{self.name}_mask_flat_l{lvl}")
            setattr(self, _mflat_op.name, _mflat_op)
            mask_flat = _mflat_op(mask, mask_shape)
            setattr(self, mask_flat.name, mask_flat)

            feat_flatten.append(feat_flat)
            mask_flatten.append(mask_flat)

        # Concatenate all levels (or use single level directly)
        if len(feat_flatten) > 1:
            _fcat_op = F.ConcatX(f"{self.name}_feat_concat", axis=1)
            setattr(self, _fcat_op.name, _fcat_op)
            feat_flatten = _fcat_op(*feat_flatten)
            setattr(self, feat_flatten.name, feat_flatten)  # type: ignore[attr-defined]
            _mcat_op = F.ConcatX(f"{self.name}_mask_concat", axis=1)
            setattr(self, _mcat_op.name, _mcat_op)
            mask_flatten = _mcat_op(*mask_flatten)
            setattr(self, mask_flatten.name, mask_flatten)  # type: ignore[attr-defined]
        else:
            # Single feature level - no concatenation needed
            feat_flatten = feat_flatten[0]
            mask_flatten = mask_flatten[0]

        # Create spatial_shapes tensor as SimTensor
        spatial_shapes_np = np.array(spatial_shapes, dtype=np.int64)
        spatial_shapes_tensor = F._from_data(
            f"{self.name}_spatial_shapes", spatial_shapes_np, is_const=True
        )
        setattr(self, spatial_shapes_tensor.name, spatial_shapes_tensor)

        # Calculate level_start_index
        prod_shapes = [h * w for h, w in spatial_shapes]
        level_start_index_np = np.array(
            [0] + list(np.cumsum(prod_shapes)[:-1]), dtype=np.int64
        )
        level_start_index_tensor = F._from_data(
            f"{self.name}_level_start_index", level_start_index_np, is_const=True
        )
        setattr(self, level_start_index_tensor.name, level_start_index_tensor)

        # Permute features: [bs, num_keys, embed_dims] -> [num_keys, bs, embed_dims]
        _fperm_op = F.Transpose(f"{self.name}_feat_perm", perm=[1, 0, 2])
        setattr(self, _fperm_op.name, _fperm_op)
        feat_flatten = _fperm_op(feat_flatten)
        setattr(self, feat_flatten.name, feat_flatten)  # type: ignore[attr-defined]

        # Decoder
        # query: [bs, num_q, embed_dims] -> [num_q, bs, embed_dims]
        _qperm_op = F.Transpose(f"{self.name}_query_perm", perm=[1, 0, 2])
        setattr(self, _qperm_op.name, _qperm_op)
        query = _qperm_op(query_embed)
        setattr(self, query.name, query)

        if memory_query is not None:
            _mqperm_op = F.Transpose(f"{self.name}_mem_query_perm", perm=[1, 0, 2])
            setattr(self, _mqperm_op.name, _mqperm_op)
            memory_query = _mqperm_op(memory_query)
            setattr(self, memory_query.name, memory_query)

        # Forward through decoder
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=feat_flatten,
            query_pos=None,
            key_padding_mask=mask_flatten,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index_tensor,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            memory_query=memory_query,
            memory_bank=memory_bank,
            **kwargs,
        )

        return inter_states, init_reference_points, inter_references

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for the transformer.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            print(f"{indent}MapTransformer '{self.name}':")

        # Count encoder parameters
        if self.encoder is not None and hasattr(self.encoder, "analytical_param_count"):
            encoder_params = self.encoder.analytical_param_count(
                lvl=lvl + 1 if lvl >= 2 else 0
            )
            total_params += encoder_params
            if lvl >= 2:
                print(f"{indent}  Encoder: {encoder_params:,}")

        # Count decoder parameters
        if self.decoder is not None and hasattr(self.decoder, "analytical_param_count"):
            decoder_params = self.decoder.analytical_param_count(
                lvl=lvl + 1 if lvl >= 2 else 0
            )
            total_params += decoder_params
            if lvl >= 2:
                print(f"{indent}  Decoder: {decoder_params:,}")

        if lvl >= 1:
            print(f"{indent}Total MapTransformer params: {total_params:,}")

        return total_params
