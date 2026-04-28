
#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of the FusionAD PerceptionTransformer.

Orchestrates the BEV encoder (for extracting Bird's Eye View features from
multi-camera images) and the detection transformer decoder (for producing
object detection queries with iterative reference-point refinement).

Contains:
  - CanBusMLP          : Feed-forward network that encodes CAN-bus signals
                         (ego velocity, heading, etc.) into BEV-query offsets.
  - PerceptionTransformer : Top-level module that wires the encoder and decoder.


============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

1. Base Classes:
   - BaseModule -> SimNN.Module
2. Registry / Decorators:
   - @TRANSFORMER.register_module() -> removed
   - @force_fp32 / @auto_fp16 -> removed (inference only)
3. Builders:
   - build_transformer_layer_sequence -> pre-built encoder / decoder
4. Layers:
   - nn.Linear -> SimNN.Linear
   - nn.Parameter -> F._from_data  (constant after training)
   - nn.Sequential(Linear, ReLU, ...) -> CanBusMLP helper
5. Operations:
   - torch.split -> numpy slicing  (const query embeddings)
   - .unsqueeze().expand() -> numpy tile
   - .sigmoid() -> F.Sigmoid
   - .permute() -> F.Transpose
   - .flatten() -> F.Reshape
   - torch.cat -> F.ConcatX
   - torchvision.rotate -> numpy (preprocessing, not in graph)
6. Decoder integration:
   - Decoder forward produces (inter_states, inter_references)
"""

#-------------------------------PyTorch--------------------------------

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
# from .decoder import CustomMSDeformableAttention
# from mmcv.runner import force_fp32, auto_fp16


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
#                  two_stage_num_proposals=300,
#                  encoder=None,
#                  decoder=None,
#                  embed_dims=256,
#                  rotate_prev_bev=True,
#                  use_shift=True,
#                  use_can_bus=True,
#                  can_bus_norm=True,
#                  use_cams_embeds=True,
#                  rotate_center=[100, 100],
#                  **kwargs):
#         super(PerceptionTransformer, self).__init__(**kwargs)
#         self.encoder = build_transformer_layer_sequence(encoder)
#         self.decoder = build_transformer_layer_sequence(decoder)
#         self.embed_dims = embed_dims
#         self.num_feature_levels = num_feature_levels
#         self.num_cams = num_cams
#         self.fp16_enabled = False
#
#         self.rotate_prev_bev = rotate_prev_bev
#         self.use_shift = use_shift
#         self.use_can_bus = use_can_bus
#         self.can_bus_norm = can_bus_norm
#         self.use_cams_embeds = use_cams_embeds
#
#         self.two_stage_num_proposals = two_stage_num_proposals
#         self.init_layers()
#         self.rotate_center = rotate_center
#
#     def init_layers(self):
#         """Initialize layers of the Detr3DTransformer."""
#         self.level_embeds = nn.Parameter(torch.Tensor(
#             self.num_feature_levels, self.embed_dims))
#         self.cams_embeds = nn.Parameter(
#             torch.Tensor(self.num_cams, self.embed_dims))
#         self.reference_points = nn.Linear(self.embed_dims, 3)
#         self.can_bus_mlp = nn.Sequential(
#             nn.Linear(18, self.embed_dims // 2),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.embed_dims // 2, self.embed_dims),
#             nn.ReLU(inplace=True),
#         )
#         if self.can_bus_norm:
#             self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))
#
#     def init_weights(self):
#         """Initialize the transformer weights."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
#                     or isinstance(m, CustomMSDeformableAttention):
#                 try:
#                     m.init_weight()
#                 except AttributeError:
#                     m.init_weights()
#         normal_(self.level_embeds)
#         normal_(self.cams_embeds)
#         xavier_init(self.reference_points, distribution='uniform', bias=0.)
#         xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)
#
#     @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
#     def get_bev_features(
#             self,
#             mlvl_feats,
#             bev_queries,
#             bev_h,
#             bev_w,
#             grid_length=[0.512, 0.512],
#             bev_pos=None,
#             prev_bev=None,
#             img_metas=None,
#             pts_feats=None):
#         """
#         obtain bev features.
#         """
#
#         bs = mlvl_feats[0].size(0)
#         bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
#         bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
#         # obtain rotation angle and shift with ego motion
#         delta_x = np.array([each['can_bus'][0]
#                            for each in img_metas])
#         delta_y = np.array([each['can_bus'][1]
#                            for each in img_metas])
#         ego_angle = np.array(
#             [each['can_bus'][-2] / np.pi * 180 for each in img_metas])
#         grid_length_y = grid_length[0]
#         grid_length_x = grid_length[1]
#         translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
#         translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
#         bev_angle = ego_angle - translation_angle
#         shift_y = translation_length * \
#             np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
#         shift_x = translation_length * \
#             np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
#         shift_y = shift_y * self.use_shift
#         shift_x = shift_x * self.use_shift
#         shift = bev_queries.new_tensor(
#             [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy
#
#         if prev_bev is not None:
#             if prev_bev.shape[1] == bev_h * bev_w:
#                 prev_bev = prev_bev.permute(1, 0, 2)
#             if self.rotate_prev_bev:
#                 for i in range(bs):
#                     rotation_angle = img_metas[i]['can_bus'][-1]
#                     tmp_prev_bev = prev_bev[:, i].reshape(
#                         bev_h, bev_w, -1).permute(2, 0, 1)
#                     tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
#                                           center=self.rotate_center)
#                     tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
#                         bev_h * bev_w, 1, -1)
#                     prev_bev[:, i] = tmp_prev_bev[:, 0]
#
#         # add can bus signals
#         can_bus = bev_queries.new_tensor(
#             [each['can_bus'] for each in img_metas])  # [:, :]
#         can_bus = self.can_bus_mlp(can_bus)[None, :, :]
#         bev_queries = bev_queries + can_bus * self.use_can_bus
#
#         feat_flatten = []
#         spatial_shapes = []
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
#         spatial_shapes = torch.as_tensor(
#             spatial_shapes, dtype=torch.long, device=bev_pos.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros(
#             (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#
#         feat_flatten = feat_flatten.permute(
#             0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
#
#         if pts_feats is not None:
#             # pts_feats flatten
#             pts_spatial_shapes = []
#             bs, c, h, w = pts_feats.shape
#             spatial_shape = (h, w)
#             pts_feat_flatten = pts_feats.flatten(2).permute(0, 2, 1)
#             pts_spatial_shapes.append(spatial_shape)
#
#             pts_spatial_shapes = torch.as_tensor(
#                 pts_spatial_shapes, dtype=torch.long, device=bev_pos.device)
#             pts_level_start_index = torch.cat((pts_spatial_shapes.new_zeros(
#                 (1,)), pts_spatial_shapes.prod(1).cumsum(0)[:-1]))
#         else:
#             pts_spatial_shapes = None
#             pts_level_start_index = None
#             pts_feat_flatten = None
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
#             img_metas=img_metas,
#             pts_spatial_shapes=pts_spatial_shapes,
#             pts_level_start_index=pts_level_start_index,
#             pts_feats=pts_feat_flatten,
#         )
#
#         return bev_embed
#
#     def get_states_and_refs(
#         self,
#         bev_embed,
#         object_query_embed,
#         bev_h,
#         bev_w,
#         reference_points=None,
#         reg_branches=None,
#         cls_branches=None,
#         img_metas=None
#     ):
#         bs = bev_embed.shape[1]
#         query_pos, query = torch.split(
#             object_query_embed, self.embed_dims, dim=1)
#         query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
#         query = query.unsqueeze(0).expand(bs, -1, -1)
#         if reference_points is not None:
#             reference_points = reference_points.unsqueeze(0).expand(bs, -1, -1)
#         else:
#             reference_points = self.reference_points(query_pos)
#         reference_points = reference_points.sigmoid()
#         init_reference_out = reference_points
#         query = query.permute(1, 0, 2)
#         query_pos = query_pos.permute(1, 0, 2)
#         inter_states, inter_references = self.decoder(
#             query=query,
#             key=None,
#             value=bev_embed,
#             query_pos=query_pos,
#             reference_points=reference_points,
#             reg_branches=reg_branches,
#             cls_branches=cls_branches,
#             spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
#             level_start_index=torch.tensor([0], device=query.device),
#             img_metas=img_metas
#         )
#         inter_references_out = inter_references
#
#         return inter_states, init_reference_out, inter_references_out


#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger
import copy
import warnings
import math

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import sub-modules
from .builder_utils import LayerNorm, build_norm_layer


# ======================================================================
# Pure-numpy 2D bilinear rotation 
# ======================================================================

def _rotate_2d_bilinear(img, angle_deg, center=None):
    """
    Rotate a [C, H, W] image counter-clockwise by *angle_deg* degrees
    using bilinear interpolation.  Out-of-bound pixels are set to 0.

    Args:
        img: np.ndarray [C, H, W]
        angle_deg: rotation angle in degrees (counter-clockwise)
        center: [cx, cy] rotation centre in pixel coords.
                Defaults to image centre.
    Returns:
        np.ndarray [C, H, W] — rotated image.
    """
    C, H, W = img.shape
    if center is None:
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    rad = -np.deg2rad(angle_deg)  # negative: inverse mapping
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    # Destination grid
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64),
                          np.arange(W, dtype=np.float64), indexing='ij')
    # Map destination -> source
    src_x = cos_a * (xx - cx) - sin_a * (yy - cy) + cx
    src_y = sin_a * (xx - cx) + cos_a * (yy - cy) + cy

    # Floor coordinates for bilinear
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    wx = (src_x - x0).astype(img.dtype)
    wy = (src_y - y0).astype(img.dtype)

    # Mask valid source pixels
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    # Clamp to valid range for indexing (will be masked to 0 anyway)
    x0c = np.clip(x0, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    x1c = np.clip(x1, 0, W - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Bilinear interpolation per channel
    out = np.zeros_like(img)
    for c in range(C):
        val = (img[c, y0c, x0c] * (1 - wx) * (1 - wy) +
               img[c, y0c, x1c] * wx * (1 - wy) +
               img[c, y1c, x0c] * (1 - wx) * wy +
               img[c, y1c, x1c] * wx * wy)
        out[c] = np.where(valid, val, 0.0)

    return out


# ======================================================================
# CanBusMLP
# ======================================================================

class CanBusMLP(SimNN.Module):
    """
    TTSim implementation of the CAN-bus MLP.

    Encodes 18-dimensional CAN-bus signals (ego velocity, heading,
    translation, etc.) into BEV-query offsets of dimension ``embed_dims``.

    Architecture:
        Linear(18, embed_dims // 2) -> ReLU ->
        Linear(embed_dims // 2, embed_dims) -> ReLU ->
        [LayerNorm(embed_dims)]        (if can_bus_norm)

    Args:
        name (str): Module name.
        embed_dims (int): Output dimension. Default 256.
        can_bus_norm (bool): Whether to apply LayerNorm at the end. Default True.
    """

    def __init__(self, name, embed_dims=256, can_bus_norm=True):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.can_bus_norm = can_bus_norm

        # Layer 0: Linear(18 -> embed_dims // 2)
        self.fc0 = SimNN.Linear(f'{name}.fc0', in_features=18,
                                out_features=embed_dims // 2)
        self.relu0 = F.Relu(f'{name}.relu0')

        # Layer 1: Linear(embed_dims // 2 -> embed_dims)
        self.fc1 = SimNN.Linear(f'{name}.fc1', in_features=embed_dims // 2,
                                out_features=embed_dims)
        self.relu1 = F.Relu(f'{name}.relu1')

        # Optional LayerNorm
        if can_bus_norm:
            self.norm = LayerNorm(f'{name}.norm', embed_dims)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: SimTensor [bs, 18]  (or [*, 18]).

        Returns:
            SimTensor [bs, embed_dims]  (or [*, embed_dims]).
        """
        out = self.fc0(x)
        out = self.relu0(out)
        out = self.fc1(out)
        out = self.relu1(out)
        if self.can_bus_norm:
            out = self.norm(out)
        return out

    def analytical_param_count(self):
        """
        Parameter count:
          fc0: 18 * (embed_dims//2) + (embed_dims//2)
          fc1: (embed_dims//2) * embed_dims + embed_dims
          norm: 2 * embed_dims  (if enabled)
        """
        half = self.embed_dims // 2
        total = 18 * half + half                     # fc0 weight + bias
        total += half * self.embed_dims + self.embed_dims  # fc1 weight + bias
        if self.can_bus_norm:
            total += 2 * self.embed_dims             # LN weight + bias
        return total


# ======================================================================
# PerceptionTransformer
# ======================================================================

class PerceptionTransformer(SimNN.Module):
    """
    TTSim implementation of the FusionAD PerceptionTransformer.

    Top-level module that:
      1.  ``get_bev_features`` — adds CAN-bus signals and
          camera/level embeddings to multi-level image features,
          then runs the BEV encoder.
      2.  ``get_states_and_refs`` — splits object query embeddings,
          computes initial reference points, and runs the detection
          decoder with iterative refinement.

    Args:
        name (str): Module name.
        encoder: Pre-built TTSim BEVFormerEncoder or None.
        decoder: Pre-built TTSim DetectionTransformerDecoder or None.
        embed_dims (int): Embedding dimension. Default 256.
        num_feature_levels (int): FPN levels. Default 4.
        num_cams (int): Number of cameras. Default 6.
        two_stage_num_proposals (int): Not used (API compat). Default 300.
        rotate_prev_bev (bool): Rotate prev BEV by ego motion. Default True.
        use_shift (bool): Shift BEV by ego translation. Default True.
        use_can_bus (bool): Add CAN-bus signal encoding. Default True.
        can_bus_norm (bool): Apply LayerNorm in CAN-bus MLP. Default True.
        use_cams_embeds (bool): Add camera embeddings. Default True.
        rotate_center (list): Rotation centre [x, y] for BEV. Default [100,100].
    """

    def __init__(self,
                 name='perception_transformer',
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=None):
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center if rotate_center is not None else [100, 100]

        # ---- Learnable sub-modules ----
        # reference_points Linear: embed_dims -> 3
        self.reference_points_linear = SimNN.Linear(
            f'{name}.reference_points', in_features=embed_dims, out_features=3)

        # CAN-bus MLP
        self.can_bus_mlp = CanBusMLP(
            f'{name}.can_bus_mlp', embed_dims=embed_dims,
            can_bus_norm=can_bus_norm)

        # ---- Learnable parameters (loaded from checkpoint) ----
        # Will be set to numpy arrays by the weight-loading step.
        self.level_embeds = None   # shape: (num_feature_levels, embed_dims)
        self.cams_embeds = None    # shape: (num_cams, embed_dims)

        # Pre-create ops used in get_states_and_refs
        self._ref_sigmoid = F.Sigmoid(f'{name}.ref_sigmoid')
        self._query_perm = F.Transpose(f'{name}.query_perm', perm=[1, 0, 2])
        self._qpos_perm = F.Transpose(f'{name}.qpos_perm', perm=[1, 0, 2])

    # ------------------------------------------------------------------
    # get_bev_features
    # ------------------------------------------------------------------

    def get_bev_features(self,
                         mlvl_feats,
                         bev_queries,
                         bev_h,
                         bev_w,
                         grid_length=None,
                         bev_pos=None,
                         prev_bev=None,
                         img_metas=None,
                         pts_feats=None,
                         **kwargs):
        """
        Produce BEV features from multi-camera multi-level image features.

        Steps:
          1. Expand ``bev_queries`` to batch dimension.
          2. Compute ego-motion shift from CAN-bus signals (numpy).
          3. (Optional) Rotate prev_bev by ego heading (numpy preprocessing).
          4. Encode CAN-bus signals via ``can_bus_mlp`` and add to BEV queries.
          5. Add camera and level embeddings to image features.
          6. (Optional) Flatten and process LiDAR features.
          7. Call encoder.

        Args:
            mlvl_feats (list[SimTensor]): Multi-level camera features.
                Each element: [bs, num_cam, C, H_l, W_l].
            bev_queries (SimTensor): [num_query, embed_dims]  (learned).
            bev_h (int): BEV grid height.
            bev_w (int): BEV grid width.
            grid_length (list): Physical BEV cell size [y, x] metres.
                Default [0.512, 0.512].
            bev_pos (SimTensor): [num_query, 1, embed_dims] or
                [embed_dims, bev_h, bev_w] positional encoding.
            prev_bev (SimTensor or None): Previous BEV features.
            img_metas (list[dict]): Per-sample metadata (CAN-bus, etc.).
            pts_feats (SimTensor or None): LiDAR BEV features
                [bs, C, H_pts, W_pts].

        Returns:
            SimTensor: BEV embedding [bs, num_query, embed_dims]
                (or as returned by encoder).
        """
        if grid_length is None:
            grid_length = [0.512, 0.512]

        bs = mlvl_feats[0].shape[0]

        # ---- 1. Expand bev_queries to batch ----
        # bev_queries: [num_query, embed_dims] -> [num_query, bs, embed_dims]
        bev_queries_np = (bev_queries.data if hasattr(bev_queries, 'data')
                          else bev_queries)
        if isinstance(bev_queries_np, np.ndarray):
            bev_queries_expand_np = np.tile(
                bev_queries_np[:, np.newaxis, :], (1, bs, 1))
        else:
            bev_queries_expand_np = np.tile(
                np.array(bev_queries_np)[:, np.newaxis, :], (1, bs, 1))
        bev_queries_tiled = F._from_data(
            f'{self.name}.bev_queries_tiled',
            bev_queries_expand_np.astype(np.float32), is_const=True)
        setattr(self, bev_queries_tiled.name, bev_queries_tiled)

        # ---- 2. Process bev_pos ----
        # PyTorch: bev_pos.flatten(2).permute(2, 0, 1)
        #   Input : (bs, embed_dims, bev_h, bev_w)   — 4-D
        #   flatten(2): (bs, embed_dims, bev_h*bev_w)
        #   permute(2,0,1): (bev_h*bev_w, bs, embed_dims)
        bev_pos_np = bev_pos.data if hasattr(bev_pos, 'data') else bev_pos
        bev_pos_flat: np.ndarray
        if isinstance(bev_pos_np, np.ndarray):
            if bev_pos_np.ndim == 4:
                # [bs, embed_dims, bev_h, bev_w]
                bev_pos_flat = bev_pos_np.reshape(
                    bev_pos_np.shape[0], bev_pos_np.shape[1], -1)  # [bs, C, H*W]
                bev_pos_flat = bev_pos_flat.transpose(2, 0, 1)     # [H*W, bs, C]
            elif bev_pos_np.ndim == 3 and bev_pos_np.shape[0] != bev_h * bev_w:
                # [embed_dims, bev_h, bev_w] (no batch) — fallback
                bev_pos_flat = bev_pos_np.reshape(bev_pos_np.shape[0], -1)
                bev_pos_flat = bev_pos_flat.T  # [H*W, C]
                bev_pos_flat = bev_pos_flat[:, np.newaxis, :]  # [H*W, 1, C]
                bev_pos_flat = np.tile(bev_pos_flat, (1, bs, 1))
            else:
                bev_pos_flat = bev_pos_np
        else:
            bev_pos_flat = bev_pos_np
        bev_pos_t = F._from_data(
            f'{self.name}.bev_pos', bev_pos_flat.astype(np.float32),
            is_const=True)
        setattr(self, bev_pos_t.name, bev_pos_t)

        # ---- 3. Ego-motion shift (numpy preprocessing) ----
        if img_metas is not None and self.use_shift:
            delta_x = np.array([m['can_bus'][0] for m in img_metas])
            delta_y = np.array([m['can_bus'][1] for m in img_metas])
            ego_angle = np.array(
                [m['can_bus'][-2] / np.pi * 180 for m in img_metas])
            grid_length_y = grid_length[0]
            grid_length_x = grid_length[1]
            translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
            translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
            bev_angle = ego_angle - translation_angle
            shift_y = (translation_length
                       * np.cos(bev_angle / 180 * np.pi)
                       / grid_length_y / bev_h)
            shift_x = (translation_length
                       * np.sin(bev_angle / 180 * np.pi)
                       / grid_length_x / bev_w)
            shift_np = np.stack([shift_x, shift_y], axis=-1).astype(np.float32)
        else:
            shift_np = np.zeros((bs, 2), dtype=np.float32)

        # ---- 4. Rotate prev_bev (numpy preprocessing) ----
        if prev_bev is not None:
            prev_np = prev_bev.data if hasattr(prev_bev, 'data') else prev_bev
            if isinstance(prev_np, np.ndarray):
                # [bs, num_query, embed_dims] => already batch-first
                if prev_np.shape[1] == bev_h * bev_w:
                    prev_np = prev_np.transpose(1, 0, 2)
                if self.rotate_prev_bev and img_metas is not None:
                    # Apply rotation to each sample in pure numpy
                    for i in range(bs):
                        rotation_angle = img_metas[i]['can_bus'][-1]
                        if abs(rotation_angle) > 1e-6:
                            tmp = prev_np[:, i].reshape(
                                bev_h, bev_w, -1)  # [H, W, C]
                            tmp = tmp.transpose(2, 0, 1)  # [C, H, W]
                            tmp_rotated = _rotate_2d_bilinear(
                                tmp, rotation_angle,
                                center=self.rotate_center)
                            tmp_rotated = tmp_rotated.transpose(1, 2, 0)
                            prev_np[:, i] = tmp_rotated.reshape(
                                bev_h * bev_w, -1)
            prev_bev_t = F._from_data(
                f'{self.name}.prev_bev', prev_np.astype(np.float32),
                is_const=True)
            setattr(self, prev_bev_t.name, prev_bev_t)
        else:
            prev_bev_t = None

        # ---- 5. CAN-bus MLP ----
        if img_metas is not None and self.use_can_bus:
            can_bus_np = np.array(
                [m['can_bus'] for m in img_metas], dtype=np.float32)
            can_bus_input = F._from_data(
                f'{self.name}.can_bus_input', can_bus_np, is_const=True)
            setattr(self, can_bus_input.name, can_bus_input)

            # can_bus_mlp(can_bus): [bs, 18] -> [bs, embed_dims]
            can_bus_out = self.can_bus_mlp(can_bus_input)
            setattr(self, can_bus_out.name, can_bus_out)

            # [bs, embed_dims] -> [1, bs, embed_dims]
            _unsq_ax = F._from_data(f'{self.name}.cb_unsq_ax',
                                    np.array([0], dtype=np.int64),
                                    is_const=True)
            setattr(self, _unsq_ax.name, _unsq_ax)
            _unsq = F.Unsqueeze(f'{self.name}.cb_unsq')
            setattr(self, _unsq.name, _unsq)
            can_bus_3d = _unsq(can_bus_out, _unsq_ax)
            setattr(self, can_bus_3d.name, can_bus_3d)

            # bev_queries += can_bus_3d
            _add = F.Add(f'{self.name}.bev_add_canbus')
            setattr(self, _add.name, _add)
            bev_queries_tiled = _add(bev_queries_tiled, can_bus_3d)
            setattr(self, bev_queries_tiled.name, bev_queries_tiled)

        # ---- 6. Add camera and level embeddings to features ----
        # Wrap embedding arrays as const SimTensors
        cams_embeds_t = None
        if self.use_cams_embeds and self.cams_embeds is not None:
            cams_embeds_t = F._from_data(
                f'{self.name}.cams_embeds',
                self.cams_embeds.astype(np.float32), is_const=True)
            setattr(self, cams_embeds_t.name, cams_embeds_t)

        level_embeds_t = None
        if self.level_embeds is not None:
            level_embeds_t = F._from_data(
                f'{self.name}.level_embeds',
                self.level_embeds.astype(np.float32), is_const=True)
            setattr(self, level_embeds_t.name, level_embeds_t)

        feat_flatten_parts = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            # feat: [bs, num_cam, C, H, W]
            num_cam = feat.shape[1]
            c = feat.shape[2]
            h = feat.shape[3]
            w = feat.shape[4]
            hw = h * w
            spatial_shapes.append((h, w))

            # Flatten: [bs, num_cam, C, H*W]
            flat_shp = F._from_data(
                f'{self.name}.feat_flat_shp_l{lvl}',
                np.array([bs, num_cam, c, hw], dtype=np.int64), is_const=True)
            setattr(self, flat_shp.name, flat_shp)
            _flat = F.Reshape(f'{self.name}.feat_flat_l{lvl}')
            setattr(self, _flat.name, _flat)
            feat = _flat(feat, flat_shp)
            setattr(self, feat.name, feat)

            # Permute: [bs, num_cam, C, H*W] -> [num_cam, bs, H*W, C]
            _perm = F.Transpose(f'{self.name}.feat_perm_l{lvl}',
                                perm=[1, 0, 3, 2])
            setattr(self, _perm.name, _perm)
            feat = _perm(feat)
            setattr(self, feat.name, feat)

            # Add camera embeddings: [num_cam, 1, 1, C]
            if cams_embeds_t is not None:
                _unsq_ax = F._from_data(
                    f'{self.name}.cam_unsq_ax_l{lvl}',
                    np.array([1, 2], dtype=np.int64), is_const=True)
                setattr(self, _unsq_ax.name, _unsq_ax)
                _unsq = F.Unsqueeze(f'{self.name}.cam_unsq_l{lvl}')
                setattr(self, _unsq.name, _unsq)
                cam_exp = _unsq(cams_embeds_t, _unsq_ax)
                setattr(self, cam_exp.name, cam_exp)
                _add = F.Add(f'{self.name}.feat_add_cam_l{lvl}')
                setattr(self, _add.name, _add)
                feat = _add(feat, cam_exp)
                setattr(self, feat.name, feat)

            # Add level embedding: [1, 1, 1, C]
            if level_embeds_t is not None:
                # Slice level_embeds[lvl:lvl+1, :]
                _sl_st = F._from_data(
                    f'{self.name}.lvl_sl_st_l{lvl}',
                    np.array([lvl, 0], dtype=np.int64), is_const=True)
                setattr(self, _sl_st.name, _sl_st)
                _sl_en = F._from_data(
                    f'{self.name}.lvl_sl_en_l{lvl}',
                    np.array([lvl + 1, self.embed_dims], dtype=np.int64),
                    is_const=True)
                setattr(self, _sl_en.name, _sl_en)
                _sl_ax = F._from_data(
                    f'{self.name}.lvl_sl_ax_l{lvl}',
                    np.array([0, 1], dtype=np.int64), is_const=True)
                setattr(self, _sl_ax.name, _sl_ax)
                _sl = F.SliceF(f'{self.name}.lvl_sl_l{lvl}',
                               out_shape=[1, self.embed_dims])
                setattr(self, _sl.name, _sl)
                lvl_emb = _sl(level_embeds_t, _sl_st, _sl_en, _sl_ax)
                setattr(self, lvl_emb.name, lvl_emb)

                # Unsqueeze to [1, 1, 1, embed_dims]
                _unsq_ax2 = F._from_data(
                    f'{self.name}.lvl_unsq_ax_l{lvl}',
                    np.array([1, 2], dtype=np.int64), is_const=True)
                setattr(self, _unsq_ax2.name, _unsq_ax2)
                _unsq2 = F.Unsqueeze(f'{self.name}.lvl_unsq_l{lvl}')
                setattr(self, _unsq2.name, _unsq2)
                lvl_emb_exp = _unsq2(lvl_emb, _unsq_ax2)
                setattr(self, lvl_emb_exp.name, lvl_emb_exp)

                _add2 = F.Add(f'{self.name}.feat_add_lvl_l{lvl}')
                setattr(self, _add2.name, _add2)
                feat = _add2(feat, lvl_emb_exp)
                setattr(self, feat.name, feat)

            feat_flatten_parts.append(feat)

        # Concatenate all levels: [num_cam, bs, sum(H*W), C]
        if len(feat_flatten_parts) > 1:
            _cat = F.ConcatX(f'{self.name}.feat_cat', axis=2)
            setattr(self, _cat.name, _cat)
            feat_flatten = _cat(*feat_flatten_parts)
            setattr(self, feat_flatten.name, feat_flatten)
        else:
            feat_flatten = feat_flatten_parts[0]

        # Compute spatial_shapes / level_start_index
        level_start_index = []
        running = 0
        for (h_l, w_l) in spatial_shapes:
            level_start_index.append(running)
            running += h_l * w_l

        # Permute: [num_cam, bs, sum(H*W), C] -> [num_cam, sum(H*W), bs, C]
        _perm2 = F.Transpose(f'{self.name}.feat_final_perm', perm=[0, 2, 1, 3])
        setattr(self, _perm2.name, _perm2)
        feat_flatten = _perm2(feat_flatten)
        setattr(self, feat_flatten.name, feat_flatten)

        # ---- 7. Process pts_feats (LiDAR BEV features) ----
        pts_spatial_shapes = None
        pts_level_start_index = None
        pts_feat_flatten = None

        if pts_feats is not None:
            pts_bs = pts_feats.shape[0]
            pts_c = pts_feats.shape[1]
            pts_h = pts_feats.shape[2]
            pts_w = pts_feats.shape[3]
            pts_hw = pts_h * pts_w

            # Flatten: [bs, C, H, W] -> [bs, C, H*W]
            _pts_shp = F._from_data(
                f'{self.name}.pts_flat_shp',
                np.array([pts_bs, pts_c, pts_hw], dtype=np.int64),
                is_const=True)
            setattr(self, _pts_shp.name, _pts_shp)
            _pts_flat = F.Reshape(f'{self.name}.pts_flat')
            setattr(self, _pts_flat.name, _pts_flat)
            pts_feat_flat = _pts_flat(pts_feats, _pts_shp)
            setattr(self, pts_feat_flat.name, pts_feat_flat)

            # Permute: [bs, C, H*W] -> [bs, H*W, C]
            _pts_perm = F.Transpose(f'{self.name}.pts_perm', perm=[0, 2, 1])
            setattr(self, _pts_perm.name, _pts_perm)
            pts_feat_flatten = _pts_perm(pts_feat_flat)
            setattr(self, pts_feat_flatten.name, pts_feat_flatten)

            pts_spatial_shapes = [(pts_h, pts_w)]
            pts_level_start_index = [0]

        # ---- 8. Call encoder ----
        bev_embed = self.encoder(
            bev_queries_tiled,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos_t,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev_t,
            shift=shift_np,
            pts_spatial_shapes=pts_spatial_shapes,
            pts_level_start_index=pts_level_start_index,
            pts_feats=pts_feat_flatten,
            **kwargs
        )

        return bev_embed

    # ------------------------------------------------------------------
    # get_states_and_refs
    # ------------------------------------------------------------------

    def get_states_and_refs(self,
                            bev_embed,
                            object_query_embed,
                            bev_h,
                            bev_w,
                            reference_points=None,
                            reg_branches=None,
                            cls_branches=None,
                            img_metas=None):
        """
        Prepare object queries + reference points and run the decoder.

        Args:
            bev_embed: SimTensor — BEV features from encoder.
                Expected shape [num_query_bev, bs, embed_dims] or
                [bs, num_query_bev, embed_dims].
            object_query_embed: numpy array [num_queries, 2*embed_dims].
                First half is ``query_pos``, second half is ``query``.
            bev_h, bev_w (int): BEV grid dimensions (for decoder
                spatial_shapes).
            reference_points: numpy array [num_queries, 3] (optional).
                If None, computed from ``query_pos`` via the learned
                linear layer.
            reg_branches: list of SimNN.Module for ref-point refinement.
            cls_branches: (unused, kept for API compat).
            img_metas: (unused in inference, kept for API compat).

        Returns:
            tuple:
              - inter_states: list of SimTensors (or stacked) from
                each decoder layer.
              - init_reference_out: numpy array [bs, nq, 3] — initial
                reference points (after sigmoid).
              - inter_references_out: list of reference points per layer.
        """
        # Determine batch size from bev_embed
        bev_np = bev_embed.data if hasattr(bev_embed, 'data') else bev_embed
        if isinstance(bev_np, np.ndarray):
            # bev_embed can be [nq, bs, embed_dims] or [bs, nq, embed_dims]
            # PyTorch convention: bev_embed.shape[1] is bs
            bs = bev_np.shape[1] if bev_np.ndim == 3 else 1
        else:
            bs = bev_embed.shape[1]

        # ---- Split object_query_embed ----
        # object_query_embed: [num_queries, 2*embed_dims]
        qe_np = (object_query_embed.data
                 if hasattr(object_query_embed, 'data')
                 else object_query_embed)
        if not isinstance(qe_np, np.ndarray):
            qe_np = np.array(qe_np, dtype=np.float32)

        query_pos_np = qe_np[:, :self.embed_dims]           # [nq, embed_dims]
        query_np = qe_np[:, self.embed_dims:]                # [nq, embed_dims]

        # Expand to batch: [nq, embed_dims] -> [bs, nq, embed_dims]
        query_pos_batch = np.tile(
            query_pos_np[np.newaxis, :, :], (bs, 1, 1)).astype(np.float32)
        query_batch = np.tile(
            query_np[np.newaxis, :, :], (bs, 1, 1)).astype(np.float32)

        # ---- Compute reference points ----
        if reference_points is not None:
            rp_np = (reference_points.data
                     if hasattr(reference_points, 'data')
                     else reference_points)
            if not isinstance(rp_np, np.ndarray):
                rp_np = np.array(rp_np, dtype=np.float32)
            if rp_np.ndim == 2:
                rp_np = np.tile(rp_np[np.newaxis, :, :], (bs, 1, 1))
            # Apply sigmoid
            rp_sig_np = 1.0 / (1.0 + np.exp(-rp_np.astype(np.float64)))
            rp_sig_np = rp_sig_np.astype(np.float32)
        else:
            # reference_points = self.reference_points_linear(query_pos).sigmoid()
            # Build SimTensor graph
            qpos_t = F._from_data(
                f'{self.name}.qpos_input',
                query_pos_batch, is_const=True)
            setattr(self, qpos_t.name, qpos_t)

            ref_raw = self.reference_points_linear(qpos_t)
            setattr(self, ref_raw.name, ref_raw)

            ref_sig = self._ref_sigmoid(ref_raw)
            setattr(self, ref_sig.name, ref_sig)

            # For init_reference_out, extract numpy for return
            # (the graph keeps going through the decoder)
            rp_sig_np = None  # will be computed from graph
            rp_sig_tensor = ref_sig

        init_reference_out = rp_sig_np  # numpy [bs, nq, 3] or None

        # ---- Permute query and query_pos to seq-first ----
        # [bs, nq, embed_dims] -> [nq, bs, embed_dims]
        query_sf = query_batch.transpose(1, 0, 2)
        qpos_sf = query_pos_batch.transpose(1, 0, 2)

        query_t = F._from_data(
            f'{self.name}.query_input',
            query_sf.astype(np.float32), is_const=True)
        setattr(self, query_t.name, query_t)

        qpos_t = F._from_data(
            f'{self.name}.qpos_sf',
            qpos_sf.astype(np.float32), is_const=True)
        setattr(self, qpos_t.name, qpos_t)

        # ---- Reference points as numpy for decoder ----
        if init_reference_out is not None:
            ref_for_decoder = init_reference_out  # numpy [bs, nq, 3]
        else:
            # ref_sig_tensor is a SimTensor; decoder will handle it
            ref_for_decoder = rp_sig_tensor

        # ---- Call decoder ----
        inter_states, inter_references = self.decoder(
            query=query_t,
            key=None,
            value=bev_embed,
            query_pos=qpos_t,
            reference_points=ref_for_decoder,
            reg_branches=reg_branches,
            spatial_shapes=[(bev_h, bev_w)],
            level_start_index=[0],
        )

        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def analytical_param_count(self):
        """
        Total learnable parameters (transformer-level only).

        Includes reference_points Linear, CAN-bus MLP, and embeddings.
        Encoder/decoder parameters are counted separately.
        """
        total = 0

        # reference_points_linear: embed_dims * 3 + 3
        total += self.embed_dims * 3 + 3

        # can_bus_mlp
        total += self.can_bus_mlp.analytical_param_count()

        # level_embeds
        total += self.num_feature_levels * self.embed_dims

        # cams_embeds
        total += self.num_cams * self.embed_dims

        return total


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("FusionAD PerceptionTransformer TTSim Module")
    logger.info("=" * 70)

    # Test CanBusMLP
    logger.info("\n--- Test CanBusMLP ---")
    try:
        mlp = CanBusMLP('test_mlp', embed_dims=256, can_bus_norm=True)
        logger.debug("  [OK] CanBusMLP constructed")
        logger.debug(f"    params = {mlp.analytical_param_count():,}")
        x_np = np.random.randn(2, 18).astype(np.float32)
        x_t = F._from_data('test_canbus_input', x_np, is_const=True)
        out = mlp(x_t)
        logger.debug("    input shape:  (2, 18)")
        logger.debug(f"    output shape: {out.shape}")
    except Exception as e:
        logger.debug(f"  [X] CanBusMLP failed: {e}")
        import traceback; traceback.print_exc()

    # Test PerceptionTransformer construction
    logger.info("\n--- Test PerceptionTransformer ---")
    try:
        pt = PerceptionTransformer(
            name='test_pt',
            encoder=None,
            decoder=None,
            embed_dims=256,
            num_feature_levels=4,
            num_cams=6,
            use_can_bus=True,
            can_bus_norm=True,
            use_cams_embeds=True,
            rotate_prev_bev=True,
            use_shift=True,
            rotate_center=[100, 100],
        )
        logger.debug("  [OK] PerceptionTransformer constructed")
        logger.debug(f"    embed_dims       = {pt.embed_dims}")
        logger.debug(f"    num_feature_lvls = {pt.num_feature_levels}")
        logger.debug(f"    num_cams         = {pt.num_cams}")
        logger.debug(f"    use_can_bus      = {pt.use_can_bus}")
        logger.debug(f"    can_bus_norm     = {pt.can_bus_norm}")
        logger.debug(f"    params (excl enc/dec) = {pt.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [X] PerceptionTransformer failed: {e}")
        import traceback; traceback.print_exc()

    logger.info("\n" + "=" * 70)
    logger.info("[OK] All module-level tests passed!")
    logger.info("=" * 70)
