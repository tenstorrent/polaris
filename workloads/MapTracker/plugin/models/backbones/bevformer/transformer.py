#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Perception Transformer for BEVFormer.

This module implements the top-level transformer that orchestrates the
BEV encoder to generate Bird's Eye View features from multi-camera images.
The object detection decoder is separate (in MapTracker head).

Original: projects/mmdet3d_plugin/bevformer/modules/transformer.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270
"""

# -------------------------------PyTorch--------------------------------

# import numpy as np
# import torch
# import torch.nn as nn
# from mmcv.cnn import xavier_init
# from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
# from mmcv.runner.base_module import BaseModule
#
# from mmdet.models.utils.builder import TRANSFORMER
# from torch.nn.init import normal_
# from mmcv.runner.base_module import BaseModule
# from torchvision.transforms.functional import rotate
# from .temporal_self_attention import TemporalSelfAttention
# from .spatial_cross_attention import MSDeformableAttention3D
# from mmcv.runner import force_fp32, auto_fp16
#
# from einops import rearrange
#
#
# @TRANSFORMER.register_module()
# class PerceptionTransformer(BaseModule):
#     """Implements the Detr3D transformer.
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
#                  num_feature_levels=4,
#                  num_cams=6,
#                  encoder=None,
#                  embed_dims=256,
#                  use_cams_embeds=True,
#                  **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = build_transformer_layer_sequence(encoder)
#         # self.decoder = build_transformer_layer_sequence(decoder)
#         self.embed_dims = embed_dims
#         self.num_feature_levels = num_feature_levels
#         self.num_cams = num_cams
#         self.fp16_enabled = False
#
#         self.use_cams_embeds = use_cams_embeds
#
#         self.init_layers()
#
#     def init_layers(self):
#         """Initialize layers of the Detr3DTransformer."""
#         self.level_embeds = nn.Parameter(torch.Tensor(
#             self.num_feature_levels, self.embed_dims))
#         self.cams_embeds = nn.Parameter(
#             torch.Tensor(self.num_cams, self.embed_dims))
#         # self.reference_points = nn.Linear(self.embed_dims, 3)
#
#     def init_weights(self):
#         """Initialize the transformer weights."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention):
#                 try:
#                     m.init_weight()
#                 except AttributeError:
#                     m.init_weights()
#         normal_(self.level_embeds)
#         normal_(self.cams_embeds)
#         # xavier_init(self.reference_points, distribution='uniform', bias=0.)
#
#     # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
#     def get_bev_features(
#             self,
#             mlvl_feats,
#             bev_queries,
#             bev_h,
#             bev_w,
#             bev_pos=None,
#             prop_bev=None,
#             prev_bev=None,
#             warped_history_bev=None,
#             **kwargs):
#         """
#         obtain bev features.
#         """
#
#         bs = mlvl_feats[0].size(0)
#         bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
#         bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
#
#         shift = bev_queries.new_tensor((0,0))[None].repeat(bs,1)
#
#         feat_flatten = []
#         spatial_shapes = []
#
#         for lvl, feat in enumerate(mlvl_feats):
#             bs, num_cam, c, h, w = feat.shape
#             spatial_shape = (h, w)
#             feat = feat.flatten(3).permute(1, 0, 3, 2)
#             if self.use_cams_embeds:
#                 feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
#             feat = feat + self.level_embeds[None,
#                                             None, lvl:lvl + 1, :].to(feat.dtype)
#             spatial_shapes.append(spatial_shape)
#             feat_flatten.append(feat)
#
#         feat_flatten = torch.cat(feat_flatten, 2)
#
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=bev_pos.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros(
#             (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#
#         feat_flatten = feat_flatten.permute(
#             0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
#
#         # Fuse the propagated bev features from the prev step
#         if prop_bev is not None:
#             prop_bev = rearrange(prop_bev, 'b c h w -> (h w) b c')
#             valid_mask = (prop_bev.sum(-1) > 0).to(bev_queries.dtype)[..., None]
#             bev_queries = bev_queries * (1 - valid_mask) + prop_bev * valid_mask
#
#         bev_embed = self.encoder(
#             bev_queries,
#             feat_flatten,
#             feat_flatten,
#             bev_h=bev_h,
#             bev_w=bev_w,
#             bev_pos=bev_pos,
#             spatial_shapes=spatial_shapes,
#             level_start_index=level_start_index,
#             prev_bev=prev_bev,
#             shift=shift,
#             warped_history_bev=warped_history_bev,
#             **kwargs
#         )
#
#         return bev_embed
#
#     @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
#     def forward(self,
#                 mlvl_feats,
#                 bev_queries,
#                 object_query_embed,
#                 bev_h,
#                 bev_w,
#                 grid_length=[0.512, 0.512],
#                 bev_pos=None,
#                 reg_branches=None,
#                 cls_branches=None,
#                 prev_bev=None,
#                 **kwargs):
#         """Forward function for `Detr3DTransformer`.
#         Args:
#             mlvl_feats (list(Tensor)): Input queries from
#                 different level. Each element has shape
#                 [bs, num_cams, embed_dims, h, w].
#             bev_queries (Tensor): (bev_h*bev_w, c)
#             bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
#             object_query_embed (Tensor): The query embedding for decoder,
#                 with shape [num_query, c].
#             reg_branches (obj:`nn.ModuleList`): Regression heads for
#                 feature maps from each decoder layer. Only would
#                 be passed when `with_box_refine` is True. Default to None.
#         Returns:
#             tuple[Tensor]: results of decoder containing the following tensor.
#                 - bev_embed: BEV features
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
#         raise NotImplementedError

# -------------------------------TTSIM-----------------------------------

import numpy as np
import warnings
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

# TTSim imports
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.sim_nn import ModuleList

# Import converted modules
from workloads.MapTracker.plugin.models.backbones.bevformer.temporal_self_attention import (
    TemporalSelfAttention,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.spatial_cross_attention import (
    MSDeformableAttention3D,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.builder_utils import (
    LayerNorm,
)


class PerceptionTransformer(SimNN.Module):
    """Perception Transformer for BEVFormer (TTSim version).

    Orchestrates the BEV encoder to transform multi-camera image features
    into Bird's Eye View (BEV) features. Does NOT include object detection decoder.

    Args:
        encoder (Module): Pre-built BEVFormer encoder module.
        embed_dims (int): Embedding dimension. Default: 256.
        num_feature_levels (int): Number of feature pyramid levels. Default: 4.
        num_cams (int): Number of cameras. Default: 6.
        rotate_prev_bev (bool): Whether to rotate previous BEV by ego motion. Default: True.
        use_shift (bool): Whether to shift BEV by ego translation. Default: True.
        use_cams_embeds (bool): Whether to add camera embeddings. Default: True.
        rotate_center (list): Center point for BEV rotation [x, y]. Default: [100, 100].
        name (str): Module name. Default: None.
    """

    def __init__(
        self,
        encoder=None,
        embed_dims=256,
        num_feature_levels=4,
        num_cams=6,
        rotate_prev_bev=True,
        use_shift=True,
        use_cams_embeds=True,
        rotate_center=None,
        name=None,
    ):

        super().__init__()
        self.name = name if name else "perception_transformer"

        self.encoder = encoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center if rotate_center is not None else [100, 100]

        self.init_layers()

    def init_layers(self):
        """Initialize learnable parameters and sub-modules."""
        # Level embeddings for multi-scale features (num_feature_levels, embed_dims)
        # These will be loaded from pre-trained weights
        self.level_embeds = None  # Placeholder for (num_feature_levels, embed_dims)

        # Camera embeddings (num_cams, embed_dims)
        self.cams_embeds = None  # Placeholder for (num_cams, embed_dims)

    def init_weights(self):
        """Initialize weights (placeholder for TTSim inference).

        In TTSim, weights are loaded from pre-trained checkpoints.
        This method exists for API compatibility.
        """
        pass

    def get_bev_features(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=None,
        bev_pos=None,
        prop_bev=None,
        prev_bev=None,
        warped_history_bev=None,
        img_metas=None,
        **kwargs,
    ):
        """Extract BEV features from multi-camera multi-scale features.

        Args:
            mlvl_feats (list[Tensor]): Multi-level multi-camera features.
                Each element has shape [bs, num_cams, C, H, W].
            bev_queries (Tensor): BEV query embeddings with shape [bev_h*bev_w, embed_dims].
            bev_h (int): Height of BEV grid.
            bev_w (int): Width of BEV grid.
            grid_length (list): Physical size of each BEV grid cell [y, x] in meters. Default: [0.512, 0.512].
            bev_pos (Tensor): BEV positional encodings with shape [bs, embed_dims, bev_h, bev_w].
            prop_bev (Tensor): Propagated BEV features from previous step [bs, C, bev_h, bev_w]. Default: None.
            prev_bev (Tensor): Previous BEV features for temporal fusion. Default: None.
            warped_history_bev (Tensor): Warped history BEV features for memory fusion. Default: None.
            img_metas (list[dict]): Image meta information (not used in MapTracker).

        Returns:
            Tensor: BEV features with shape [bs, bev_h*bev_w, embed_dims].
        """
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Get batch size from first feature level tensor shape
        # mlvl_feats[0] shape: [bs, num_cams, C, H, W]
        bs = mlvl_feats[0].shape[0]

        # Expand bev_queries to batch: [bev_h*bev_w, embed_dims] -> [bs, bev_h*bev_w, embed_dims]
        # First unsqueeze to [1, bev_h*bev_w, embed_dims], then tile along batch dim
        _unsq_op = F.Unsqueeze(f"{self.name}_bev_queries_unsq")
        setattr(self, _unsq_op.name, _unsq_op)
        _unsq_ax = F._from_data(
            f"{self.name}_bev_queries_unsq_ax",
            np.array([0], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _unsq_ax.name, _unsq_ax)
        bev_queries_3d = _unsq_op(bev_queries, _unsq_ax)
        setattr(self, bev_queries_3d.name, bev_queries_3d)

        _tile_op = F.Tile(f"{self.name}_bev_queries_tile")
        setattr(self, _tile_op.name, _tile_op)
        _tile_repeats = F._from_data(
            f"{self.name}_bev_queries_tile_repeats",
            np.array([bs, 1, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _tile_repeats.name, _tile_repeats)
        bev_queries_reshaped = _tile_op(bev_queries_3d, _tile_repeats)
        setattr(self, bev_queries_reshaped.name, bev_queries_reshaped)

        # Reshape bev_pos: [bs, embed_dims, bev_h, bev_w] -> [bs, embed_dims, bev_h*bev_w]
        bev_pos_shape = F._from_data(
            f"{self.name}_bev_pos_shape",
            np.array([bs, self.embed_dims, bev_h * bev_w], dtype=np.int64),
            is_const=True,
        )
        setattr(self, bev_pos_shape.name, bev_pos_shape)
        _op = F.Reshape(f"{self.name}_bev_pos_reshape")
        setattr(self, _op.name, _op)
        bev_pos = _op(bev_pos, bev_pos_shape)
        setattr(self, bev_pos.name, bev_pos)
        # Permute to [bs, bev_h*bev_w, embed_dims] (batch-first, matching temporal_self_attention)
        _op = F.Transpose(f"{self.name}_bev_pos_transpose", perm=[0, 2, 1])
        setattr(self, _op.name, _op)
        bev_pos = _op(bev_pos)
        setattr(self, bev_pos.name, bev_pos)

        # Process ego motion (shift and rotation)
        shift = kwargs.get("shift", None)
        if shift is None:
            shift = F._from_data(
                f"{self.name}_zero_shift",
                np.zeros((1, 2), dtype=np.float32),
                is_const=True,
            )
            setattr(self, shift.name, shift)

        # Handle previous BEV rotation if provided
        if prev_bev is not None and self.rotate_prev_bev:
            warnings.warn(
                "BEV rotation not fully implemented in TTSim version. Using unrotated prev_bev."
            )
            pass

        # Fuse propagated BEV features from previous step (matches PyTorch source)
        if prop_bev is not None:
            # prop_bev: [bs, C, bev_h, bev_w] -> [bs, bev_h*bev_w, C]
            prop_bev_shape = F._from_data(
                f"{self.name}_prop_bev_shape",
                np.array([bs, self.embed_dims, bev_h * bev_w], dtype=np.int64),
                is_const=True,
            )
            setattr(self, prop_bev_shape.name, prop_bev_shape)
            _op = F.Reshape(f"{self.name}_prop_bev_reshape")
            setattr(self, _op.name, _op)
            prop_bev_flat = _op(prop_bev, prop_bev_shape)
            setattr(self, prop_bev_flat.name, prop_bev_flat)
            # Permute to [bs, bev_h*bev_w, C]
            _op = F.Transpose(f"{self.name}_prop_bev_transpose", perm=[0, 2, 1])
            setattr(self, _op.name, _op)
            prop_bev_flat = _op(prop_bev_flat)
            setattr(self, prop_bev_flat.name, prop_bev_flat)
            # valid_mask = (prop_bev.sum(-1) > 0)[..., None]  -- sum over C dimension
            _op_sum = F.ReduceSum(f"{self.name}_prop_bev_sum", axes=[-1], keepdims=0)
            setattr(self, _op_sum.name, _op_sum)
            prop_sum = _op_sum(prop_bev_flat)
            setattr(self, prop_sum.name, prop_sum)
            # Create binary float mask (1.0 where sum > 0, else 0.0)
            # Relu zeros negatives, scale by 1e10, Clip to [0,1] → exact 0/1.
            _op_relu = F.Relu(f"{self.name}_prop_bev_relu")
            setattr(self, _op_relu.name, _op_relu)
            relu_sum = _op_relu(prop_sum)
            setattr(self, relu_sum.name, relu_sum)
            large_scale = F._from_data(
                f"{self.name}_prop_large_scale",
                np.array(1e10, dtype=np.float32),
                is_const=True,
            )
            setattr(self, large_scale.name, large_scale)
            _op_scale = F.Mul(f"{self.name}_prop_bev_scale")
            setattr(self, _op_scale.name, _op_scale)
            scaled_sum = _op_scale(relu_sum, large_scale)
            setattr(self, scaled_sum.name, scaled_sum)
            # Clip to [0, 1] following Relu6 pattern for ONNX Clip op
            clip_min = F._from_data(
                f"{self.name}_clip_min", np.array(0.0, dtype=np.float32), is_const=True
            )
            setattr(self, clip_min.name, clip_min)
            clip_max = F._from_data(
                f"{self.name}_clip_max", np.array(1.0, dtype=np.float32), is_const=True
            )
            setattr(self, clip_max.name, clip_max)
            _op_clip = F.SimOpHandle(
                f"{self.name}_prop_bev_clip",
                "Clip",
                params=[(1, clip_min), (2, clip_max)],
                ipos=[0],
            )
            _op_clip.implicit_inputs.extend([clip_min, clip_max])
            setattr(self, _op_clip.name, _op_clip)
            valid_mask = _op_clip(scaled_sum)
            setattr(self, valid_mask.name, valid_mask)
            _op_unsq = F.Unsqueeze(f"{self.name}_prop_bev_mask_unsq")
            setattr(self, _op_unsq.name, _op_unsq)
            ax_neg1 = F._from_data(
                f"{self.name}_prop_ax_neg1",
                np.array([-1], dtype=np.int64),
                is_const=True,
            )
            setattr(self, ax_neg1.name, ax_neg1)
            valid_mask = _op_unsq(valid_mask, ax_neg1)  # [bs, bev_h*bev_w, 1]
            setattr(self, valid_mask.name, valid_mask)
            # bev_queries = bev_queries * (1 - valid_mask) + prop_bev * valid_mask
            one_const = F._from_data(
                f"{self.name}_prop_one", np.array(1.0, dtype=np.float32), is_const=True
            )
            setattr(self, one_const.name, one_const)
            _op_sub = F.Sub(f"{self.name}_prop_inv_mask")
            setattr(self, _op_sub.name, _op_sub)
            inv_mask = _op_sub(one_const, valid_mask)
            setattr(self, inv_mask.name, inv_mask)
            _op_mul1 = F.Mul(f"{self.name}_prop_bev_mul_inv")
            setattr(self, _op_mul1.name, _op_mul1)
            bev_queries_masked = _op_mul1(bev_queries_reshaped, inv_mask)
            setattr(self, bev_queries_masked.name, bev_queries_masked)
            _op_mul2 = F.Mul(f"{self.name}_prop_bev_mul_mask")
            setattr(self, _op_mul2.name, _op_mul2)
            prop_bev_masked = _op_mul2(prop_bev_flat, valid_mask)
            setattr(self, prop_bev_masked.name, prop_bev_masked)
            _op_add = F.Add(f"{self.name}_prop_bev_fuse")
            setattr(self, _op_add.name, _op_add)
            bev_queries_reshaped = _op_add(bev_queries_masked, prop_bev_masked)
            setattr(self, bev_queries_reshaped.name, bev_queries_reshaped)

        # Process multi-level features
        feat_flatten = []
        spatial_shapes = []

        # Convert embedding parameters to TTSim tensors (once, outside loop)
        cams_embeds_tensor = None
        if self.use_cams_embeds and self.cams_embeds is not None:
            cams_embeds_tensor = F._from_data(
                f"{self.name}_cams_embeds", self.cams_embeds, is_const=True
            )
            setattr(self, cams_embeds_tensor.name, cams_embeds_tensor)

        level_embeds_tensor = None
        if self.level_embeds is not None:
            level_embeds_tensor = F._from_data(
                f"{self.name}_level_embeds", self.level_embeds, is_const=True
            )
            setattr(self, level_embeds_tensor.name, level_embeds_tensor)

        for lvl, feat in enumerate(mlvl_feats):
            # feat shape: [bs, num_cams, C, H, W]
            num_cam = feat.shape[1]
            c = feat.shape[2]
            h = feat.shape[3]
            w = feat.shape[4]

            spatial_shapes.append((h, w))

            # Flatten spatial dimensions: [bs, num_cams, C, H, W] -> [bs, num_cams, C, H*W]
            hw = h * w
            feat_shape = F._from_data(
                f"{self.name}_feat_l{lvl}_flatten_shape",
                np.array([bs, num_cam, c, hw], dtype=np.int64),
                is_const=True,
            )
            setattr(self, feat_shape.name, feat_shape)
            _op = F.Reshape(f"{self.name}_feat_l{lvl}_flatten")
            setattr(self, _op.name, _op)
            feat = _op(feat, feat_shape)
            setattr(self, feat.name, feat)

            # Permute to [num_cams, bs, H*W, C]
            _op = F.Transpose(f"{self.name}_feat_l{lvl}_transpose", perm=[1, 0, 3, 2])
            setattr(self, _op.name, _op)
            feat = _op(feat)
            setattr(self, feat.name, feat)

            # Add camera embeddings if enabled
            if cams_embeds_tensor is not None:
                cams_axes = F._from_data(
                    f"{self.name}_cams_embeds_l{lvl}_axes",
                    np.array([1, 2], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, cams_axes.name, cams_axes)
                _op = F.Unsqueeze(f"{self.name}_cams_embeds_l{lvl}_unsqueeze")
                setattr(self, _op.name, _op)
                cams_embeds_expanded = _op(cams_embeds_tensor, cams_axes)
                setattr(self, cams_embeds_expanded.name, cams_embeds_expanded)
                _op = F.Add(f"{self.name}_feat_l{lvl}_add_cams")
                setattr(self, _op.name, _op)
                feat = _op(feat, cams_embeds_expanded)
                setattr(self, feat.name, feat)

            # Add level embeddings if available
            if level_embeds_tensor is not None:
                starts_tensor = F._from_data(
                    f"{self.name}_level_embed_l{lvl}_starts",
                    np.array([lvl, 0], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, starts_tensor.name, starts_tensor)
                ends_tensor = F._from_data(
                    f"{self.name}_level_embed_l{lvl}_ends",
                    np.array([lvl + 1, self.embed_dims], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, ends_tensor.name, ends_tensor)
                axes_tensor = F._from_data(
                    f"{self.name}_level_embed_l{lvl}_axes",
                    np.array([0, 1], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, axes_tensor.name, axes_tensor)
                _op = F.SliceF(
                    f"{self.name}_level_embed_l{lvl}_slice",
                    out_shape=[1, self.embed_dims],
                )
                setattr(self, _op.name, _op)
                level_embed = _op(
                    level_embeds_tensor, starts_tensor, ends_tensor, axes_tensor
                )
                setattr(self, level_embed.name, level_embed)
                level_axes = F._from_data(
                    f"{self.name}_level_embed_l{lvl}_unsqueeze_axes",
                    np.array([1, 2], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, level_axes.name, level_axes)
                _op = F.Unsqueeze(f"{self.name}_level_embed_l{lvl}_unsqueeze")
                setattr(self, _op.name, _op)
                level_embed = _op(level_embed, level_axes)
                setattr(self, level_embed.name, level_embed)
                _op = F.Add(f"{self.name}_feat_l{lvl}_add_level")
                setattr(self, _op.name, _op)
                feat = _op(feat, level_embed)
                setattr(self, feat.name, feat)

            feat_flatten.append(feat)

        # Concatenate all levels: [num_cams, bs, sum(H*W), C]
        if len(feat_flatten) > 1:
            _op = F.ConcatX(f"{self.name}_feat_flatten_concat", axis=2)
            setattr(self, _op.name, _op)
            feat_flatten = _op(*feat_flatten)
            setattr(self, feat_flatten.name, feat_flatten)  # type: ignore[attr-defined]
        else:
            feat_flatten = feat_flatten[0]

        # Prepare spatial_shapes and level_start_index
        # spatial_shapes: list of (H, W) per level – built in the loop above
        # level_start_index: cumulative start positions in the flattened token dim
        level_start_index = []
        running = 0
        for h_l, w_l in spatial_shapes:
            level_start_index.append(running)
            running += h_l * w_l

        # Permute feat_flatten to [num_cams, sum(H*W), bs, C]
        _op = F.Transpose(
            f"{self.name}_feat_flatten_final_transpose", perm=[0, 2, 1, 3]
        )
        setattr(self, _op.name, _op)
        feat_flatten = _op(feat_flatten)
        setattr(self, feat_flatten.name, feat_flatten)  # type: ignore[attr-defined]

        # Call encoder
        # Remove spatial_shapes/level_start_index from kwargs to avoid
        # duplicate keyword args (we computed them locally above).
        kwargs.pop("spatial_shapes", None)
        kwargs.pop("level_start_index", None)
        bev_embed = self.encoder(
            bev_queries_reshaped,
            feat_flatten,
            feat_flatten,  # key and value are the same
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            warped_history_bev=warped_history_bev,
            img_metas=img_metas,
            **kwargs,
        )

        return bev_embed

    def __call__(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=None,
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for PerceptionTransformer.

        Note: This method is NOT implemented in MapTracker. MapTracker only uses
        get_bev_features() to generate BEV features. The object detection decoder
        is separate and located in the MapTracker head (MapTransformerDecoder).

        Args:
            mlvl_feats (list[Tensor]): Multi-level multi-camera features.
            bev_queries (Tensor): BEV query embeddings.
            object_query_embed (Tensor): Object query embeddings.
            bev_h (int): Height of BEV grid.
            bev_w (int): Width of BEV grid.
            grid_length (list): Physical size of BEV grid cell.
            bev_pos (Tensor): BEV positional encodings.
            reg_branches: Regression heads (not used).
            cls_branches: Classification heads (not used).
            prev_bev (Tensor): Previous BEV features.

        Returns:
            NotImplementedError: This method is not used in MapTracker.
        """
        raise NotImplementedError


def rotate_bev_with_affine(
    prev_bev, rotation_angle, bev_h, bev_w, rotate_center, name="rotate_bev"
):
    """Rotate BEV features using affine transformation (TTSim implementation).

    This is a helper function to rotate BEV features for temporal alignment.
    Uses GridSample with affine transformation matrix.

    Args:
        prev_bev (Tensor): Previous BEV features with shape [bev_h*bev_w, 1, embed_dims].
        rotation_angle (float): Rotation angle in radians.
        bev_h (int): BEV grid height.
        bev_w (int): BEV grid width.
        rotate_center (list): Rotation center [cx, cy] in grid coordinates.
        name (str): Operation name prefix.

    Returns:
        Tensor: Rotated BEV features with same shape as input.
    """
    # Reshape prev_bev: [bev_h*bev_w, 1, embed_dims] -> [1, bev_h, bev_w, embed_dims]
    # Extract embed_dims from tensor shape
    embed_dims = prev_bev.shape[2]
    reshape_shape_4d = F._from_data(
        f"{name}_reshape_4d_shape",
        np.array([1, bev_h, bev_w, embed_dims], dtype=np.int64),
        is_const=True,
    )
    prev_bev = F.Reshape(f"{name}_reshape_4d")(prev_bev, reshape_shape_4d)
    prev_bev = F.Transpose(f"{name}_transpose_bchw", perm=[0, 3, 1, 2])(prev_bev)

    # Create rotation grid
    # This would require implementing an affine grid generator using rotation matrix
    # For now, return unrotated (full implementation would use GridSample with rotation grid)
    warnings.warn(f"BEV rotation not fully implemented. Returning unrotated BEV.")

    # Reshape back: [1, embed_dims, bev_h, bev_w] -> [bev_h*bev_w, 1, embed_dims]
    prev_bev = F.Transpose(f"{name}_transpose_back", perm=[0, 2, 3, 1])(prev_bev)
    reshape_shape_final = F._from_data(
        f"{name}_reshape_final_shape",
        np.array([bev_h * bev_w, 1, embed_dims], dtype=np.int64),
        is_const=True,
    )
    prev_bev = F.Reshape(f"{name}_reshape_final")(prev_bev, reshape_shape_final)

    return prev_bev


def analytical_param_count(embed_dims=256, num_feature_levels=4, num_cams=6):
    """Calculate the analytical parameter count for PerceptionTransformer.

    This does not include encoder parameters as it is passed as a pre-built module.

    Args:
        embed_dims (int): Embedding dimensions. Default: 256.
        num_feature_levels (int): Number of feature pyramid levels. Default: 4.
        num_cams (int): Number of cameras. Default: 6.

    Returns:
        int: Total number of parameters (excluding encoder).
    """
    # level_embeds: [num_feature_levels, embed_dims]
    level_embeds_params = num_feature_levels * embed_dims

    # cams_embeds: [num_cams, embed_dims]
    cams_embeds_params = num_cams * embed_dims

    total = level_embeds_params + cams_embeds_params

    return total
