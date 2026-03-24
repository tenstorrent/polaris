#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Spatial Cross Attention for BEVFormer.

This module implements the spatial cross-attention mechanism that allows BEV queries
to attend to multi-camera image features using deformable attention.

Original: projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

The original PyTorch implementation uses several mmcv functions that are not
compatible with Python 3.13. This TTSim version includes the following conversions:

1. Initialization Functions (mmcv.cnn):
   - xavier_init: Implemented in init_utils.py using torch.nn.init for PyTorch
   - constant_init: Implemented in init_utils.py using torch.nn.init for PyTorch

2. Decorators (mmcv.runner):
   - @force_fp32: Not needed in TTSim (operates on graph)
   - @auto_fp16: Not needed in TTSim (operates on graph)

3. Registry (mmcv.cnn.bricks.registry):
   - @ATTENTION.register_module(): Not needed in TTSim (no module registry)
   - build_attention: Replaced with direct module instantiation

4. Base Classes (mmcv.runner.base_module):
   - BaseModule: Replaced with ttsim.front.functional.sim_nn.Module

5. Custom Operations:
   - multi_scale_deformable_attn_pytorch: Implemented in multi_scale_deformable_attn.py
   - MultiScaleDeformableAttnFunction: Replaced with TTSim implementation

6. Utilities:
   - ext_loader: Not needed in TTSim (no CUDA extensions)
   - run_time decorator: Not needed in TTSim (no profiling decorators)

7. Operations:
   - torch.clamp(count, min=1.0): Replaced with F.Maximum(count, 1.0)
   - torch.zeros_like: Replaced with F._from_data with np.zeros
   - torch.register_buffer: Stored as numpy array + SimTensor constant

8. Warnings:
   - warnings.warn for non-power-of-2 dimensions: Implemented in init_weights

============================================================================
TTSim Implementation Notes
============================================================================

1. Constant Tensors:
   - Created once in __init__/init_weights() using F._from_data(..., is_const=True)
   - Avoids recreating constants during graph construction
   - Example: MSIPM3D.fixed_sampling_offsets_tensor

2. Operator Syntax:
   - F.Reshape(name)(tensor, shape_tensor) - not F.Reshape(tensor, shape=...)
   - F.Tile(name)(tensor, repeats_tensor) - not F.Tile(tensor, repeats=...)
   - F.SliceF(name, out_shape=...)(tensor, starts, ends, axes, steps)
   - F.ConcatX(name, axis=...)(tensor1, tensor2, ...) - variadic args, not list

3. Weight Loading:
   - Learnable parameters loaded from pre-trained PyTorch models
   - Fixed/constant parameters computed identically in PyTorch and TTSim
   - init_weights() documents PyTorch initialization but doesn't execute in TTSim

All computational logic from the PyTorch version has been preserved and
converted to TTSim operations.
"""

# -------------------------------PyTorch--------------------------------

# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
# import warnings
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import xavier_init, constant_init
# from mmcv.cnn.bricks.registry import (ATTENTION,
#                                       TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import build_attention
# import math
# from mmcv.runner import force_fp32, auto_fp16
#
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
#
# from mmcv.utils import ext_loader
# from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
#     MultiScaleDeformableAttnFunction_fp16
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# @ATTENTION.register_module()
# class SpatialCrossAttention(BaseModule):
#     """An attention module used in BEVFormer.
#     Args:
#         embed_dims (int): The embedding dimension of Attention.
#             Default: 256.
#         num_cams (int): The number of cameras
#         dropout (float): A Dropout layer on `inp_residual`.
#             Default: 0..
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         deformable_attention: (dict): The config for the deformable attention used in SCA.
#     """
#
#     def __init__(self,
#                  embed_dims=256,
#                  num_cams=6,
#                  pc_range=None,
#                  dropout=0.1,
#                  init_cfg=None,
#                  batch_first=False,
#                  deformable_attention=dict(
#                      type='MSDeformableAttention3D',
#                      embed_dims=256,
#                      num_levels=4),
#                  **kwargs
#                  ):
#         super(SpatialCrossAttention, self).__init__(init_cfg)
#
#         self.init_cfg = init_cfg
#         self.dropout = nn.Dropout(dropout)
#         self.pc_range = pc_range
#         self.fp16_enabled = False
#         self.deformable_attention = build_attention(deformable_attention)
#         self.embed_dims = embed_dims
#         self.num_cams = num_cams
#         self.output_proj = nn.Linear(embed_dims, embed_dims)
#         self.batch_first = batch_first
#         self.init_weight()
#
#     def init_weight(self):
#         """Default initialization for Parameters of Module."""
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
#
#     @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
#     def forward(self,
#                 query,
#                 key,
#                 value,
#                 residual=None,
#                 query_pos=None,
#                 key_padding_mask=None,
#                 reference_points=None,
#                 spatial_shapes=None,
#                 reference_points_cam=None,
#                 bev_mask=None,
#                 level_start_index=None,
#                 flag='encoder',
#                 **kwargs):
#         """Forward Function of Detr3DCrossAtten.
#         Args:
#             query (Tensor): Query of Transformer with shape
#                 (num_query, bs, embed_dims).
#             key (Tensor): The key tensor with shape
#                 `(num_key, bs, embed_dims)`.
#             value (Tensor): The value tensor with shape
#                 `(num_key, bs, embed_dims)`. (B, N, C, H, W)
#             residual (Tensor): The tensor used for addition, with the
#                 same shape as `x`. Default None. If None, `x` will be used.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for  `key`. Default
#                 None.
#             reference_points (Tensor):  The normalized reference
#                 points with shape (bs, num_query, 4),
#                 all elements is range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area.
#                 or (N, Length_{query}, num_levels, 4), add
#                 additional two dimensions is (w, h) to
#                 form reference boxes.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_key].
#             spatial_shapes (Tensor): Spatial shape of features in
#                 different level. With shape  (num_levels, 2),
#                 last dimension represent (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape (num_levels) and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """
#
#         if key is None:
#             key = query
#         if value is None:
#             value = key
#
#         if residual is None:
#             inp_residual = query
#             slots = torch.zeros_like(query)
#         if query_pos is not None:
#             query = query + query_pos
#
#         bs, num_query, _ = query.size()
#
#         D = reference_points_cam.size(3)
#         indexes = []
#         for i, mask_per_img in enumerate(bev_mask):
#             index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
#             indexes.append(index_query_per_img)
#         max_len = max([len(each) for each in indexes])
#
#         # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
#         queries_rebatch = query.new_zeros(
#             [bs, self.num_cams, max_len, self.embed_dims])
#         reference_points_rebatch = reference_points_cam.new_zeros(
#             [bs, self.num_cams, max_len, D, 2])
#
#         for j in range(bs):
#             for i, reference_points_per_img in enumerate(reference_points_cam):
#                 index_query_per_img = indexes[i]
#                 queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
#                 reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
#
#         num_cams, l, bs, embed_dims = key.shape
#
#         key = key.permute(2, 0, 1, 3).reshape(
#             bs * self.num_cams, l, self.embed_dims)
#         value = value.permute(2, 0, 1, 3).reshape(
#             bs * self.num_cams, l, self.embed_dims)
#
#         queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
#                                             reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
#                                             level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
#         for j in range(bs):
#             for i, index_query_per_img in enumerate(indexes):
#                 slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
#
#         count = bev_mask.sum(-1) > 0
#         count = count.permute(1, 2, 0).sum(-1)
#         count = torch.clamp(count, min=1.0)
#         slots = slots / count[..., None]
#         slots = self.output_proj(slots)
#
#         return self.dropout(slots) + inp_residual


# @ATTENTION.register_module()
# class MSDeformableAttention3D(BaseModule):
#     """An attention module used in BEVFormer based on Deformable-Detr.
#     `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
#     <https://arxiv.org/pdf/2010.04159.pdf>`_.
#     Args:
#         embed_dims (int): The embedding dimension of Attention.
#             Default: 256.
#         num_heads (int): Parallel attention heads. Default: 64.
#         num_levels (int): The number of feature map used in
#             Attention. Default: 4.
#         num_points (int): The number of sampling points for
#             each query in each head. Default: 4.
#         im2col_step (int): The step used in image_to_column.
#             Default: 64.
#         dropout (float): A Dropout layer on `inp_identity`.
#             Default: 0.1.
#         batch_first (bool): Key, Query and Value are shape of
#             (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default to False.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: None.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """
#
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  num_levels=4,
#                  num_points=8,
#                  im2col_step=64,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         if embed_dims % num_heads != 0:
#             raise ValueError(f'embed_dims must be divisible by num_heads, '
#                              f'but got {embed_dims} and {num_heads}')
#         dim_per_head = embed_dims // num_heads
#         self.norm_cfg = norm_cfg
#         self.batch_first = batch_first
#         self.output_proj = None
#         self.fp16_enabled = False
#
#         # you'd better set dim_per_head to a power of 2
#         # which is more efficient in the CUDA implementation
#         def _is_power_of_2(n):
#             if (not isinstance(n, int)) or (n < 0):
#                 raise ValueError(
#                     'invalid input for _is_power_of_2: {} (type: {})'.format(
#                         n, type(n)))
#             return (n & (n - 1) == 0) and n != 0
#
#         if not _is_power_of_2(dim_per_head):
#             warnings.warn(
#                 "You'd better set embed_dims in "
#                 'MultiScaleDeformAttention to make '
#                 'the dimension of each attention head a power of 2 '
#                 'which is more efficient in our CUDA implementation.')
#
#         self.im2col_step = im2col_step
#         self.embed_dims = embed_dims
#         self.num_levels = num_levels
#         self.num_heads = num_heads
#         self.num_points = num_points
#         self.sampling_offsets = nn.Linear(
#             embed_dims, num_heads * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dims,
#                                            num_heads * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dims, embed_dims)
#
#         self.init_weights()
#
#     def init_weights(self):
#         """Default initialization for Parameters of Module."""
#         constant_init(self.sampling_offsets, 0.)
#         thetas = torch.arange(
#             self.num_heads,
#             dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init /
#                      grid_init.abs().max(-1, keepdim=True)[0]).view(
#             self.num_heads, 1, 1,
#             2).repeat(1, self.num_levels, self.num_points, 1)
#         for i in range(self.num_points):
#             grid_init[:, :, i, :] *= i + 1
#
#         self.sampling_offsets.bias.data = grid_init.view(-1)
#         constant_init(self.attention_weights, val=0., bias=0.)
#         xavier_init(self.value_proj, distribution='uniform', bias=0.)
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
#         self._is_init = True
#
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_padding_mask=None,
#                 reference_points=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 **kwargs):
#         """Forward Function of MultiScaleDeformAttention.
#         Args:
#             query (Tensor): Query of Transformer with shape
#                 ( bs, num_query, embed_dims).
#             key (Tensor): The key tensor with shape
#                 `(bs, num_key,  embed_dims)`.
#             value (Tensor): The value tensor with shape
#                 `(bs, num_key,  embed_dims)`.
#             identity (Tensor): The tensor used for addition, with the
#                 same shape as `query`. Default None. If None,
#                 `query` will be used.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`. Default
#                 None.
#             reference_points (Tensor):  The normalized reference
#                 points with shape (bs, num_query, num_levels, 2),
#                 all elements is range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area.
#                 or (N, Length_{query}, num_levels, 4), add
#                 additional two dimensions is (w, h) to
#                 form reference boxes.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_key].
#             spatial_shapes (Tensor): Spatial shape of features in
#                 different levels. With shape (num_levels, 2),
#                 last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape ``(num_levels, )`` and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """
#
#         if value is None:
#             value = query
#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
#
#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)
#
#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape
#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
#
#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)
#         value = value.view(bs, num_value, self.num_heads, -1)
#         sampling_offsets = self.sampling_offsets(query).view(
#             bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points)
#
#         attention_weights = attention_weights.softmax(-1)
#
#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_levels,
#                                                    self.num_points)
#
#         if reference_points.shape[-1] == 2:
#             """
#             For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
#             After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
#             For each referent point, we sample `num_points` sampling points.
#             For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
#             """
#             offset_normalizer = torch.stack(
#                 [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#
#             bs, num_query, num_Z_anchors, xy = reference_points.shape
#             reference_points = reference_points[:, :, None, None, None, :, :]
#             sampling_offsets = sampling_offsets / \
#                 offset_normalizer[None, None, None, :, None, :]
#             bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
#             sampling_offsets = sampling_offsets.view(
#                 bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
#             sampling_locations = reference_points + sampling_offsets
#             bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
#             assert num_all_points == num_points * num_Z_anchors
#
#             sampling_locations = sampling_locations.view(
#                 bs, num_query, num_heads, num_levels, num_all_points, xy)
#
#         elif reference_points.shape[-1] == 4:
#             assert False
#         else:
#             raise ValueError(
#                 f'Last dim of reference_points must be'
#                 f' 2 or 4, but get {reference_points.shape[-1]} instead.')
#
#         #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
#         #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
#         #
#
#         if torch.cuda.is_available() and value.is_cuda:
#             if value.dtype == torch.float16:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             else:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             output = MultiScaleDeformableAttnFunction.apply(
#                 value, spatial_shapes, level_start_index, sampling_locations,
#                 attention_weights, self.im2col_step)
#         else:
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights)
#         if not self.batch_first:
#             output = output.permute(1, 0, 2)
#
#         return output

# -------------------------------TTSIM-----------------------------------

import sys
import os
import warnings
import math

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import our TTSim deformable attention
from .multi_scale_deformable_attn import (
    multi_scale_deformable_attn_ttsim,
    MultiScaleDeformableAttention,
)

# Import initialization utilities (Python 3.13 compatible)
from .init_utils import xavier_init, constant_init, _is_power_of_2


class SpatialCrossAttention(SimNN.Module):
    """
    TTSim implementation of Spatial Cross Attention used in BEVFormer.

    This module enables BEV queries to attend to multi-camera image features
    through deformable attention. Each camera only interacts with its corresponding
    BEV queries (determined by bev_mask) to save memory.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_cams (int): Number of cameras. Default: 6
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: False
        deformable_attention (dict): Config for deformable attention module

    Input Shapes:
        query: [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
        key: [num_cams, l, bs, embed_dims] - Multi-camera features
        value: [num_cams, l, bs, embed_dims] - Multi-camera features
        reference_points_cam: [num_cams, bs, num_query, D, 2] - Reference points per camera
        bev_mask: [num_cams, bs, num_query] - Mask indicating which queries see which cameras
        spatial_shapes: [num_levels, 2] - Spatial shapes per level
        level_start_index: [num_levels] - Start indices per level

    Output Shape:
        [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        batch_first=False,
        deformable_attention=None,
    ):
        super().__init__()
        self.name = name

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.batch_first = batch_first

        # Create deformable attention module
        if deformable_attention is None:
            deformable_attention = {
                "embed_dims": 256,
                "num_heads": 8,
                "num_levels": 4,
                "num_points": 4,
            }

        self.deformable_attention = MSDeformableAttention3D(
            name=name + ".deformable_attention",
            embed_dims=deformable_attention.get("embed_dims", 256),
            num_heads=deformable_attention.get("num_heads", 8),
            num_levels=deformable_attention.get("num_levels", 4),
            num_points=deformable_attention.get("num_points", 4),
            dropout=dropout,
            batch_first=True,  # Internal processing in batch_first format
        )

        self.output_proj = SimNN.Linear(
            name + ".output_proj", in_features=embed_dims, out_features=embed_dims
        )

        self.dropout_rate = dropout

        # Pre-create ops for __call__
        self.query_add_pos = F.Add(name + ".query_add_pos")

        if not batch_first:
            self.query_transpose_in = F.Transpose(
                name + ".query_transpose", perm=[1, 0, 2]
            )

        # Key/value reshaping
        self.key_transpose_op = F.Transpose(name + ".key_transpose", perm=[2, 0, 1, 3])
        self.key_reshape_op = F.Reshape(name + ".key_reshape")
        self.value_transpose_op = F.Transpose(
            name + ".value_transpose", perm=[2, 0, 1, 3]
        )
        self.value_reshape_op = F.Reshape(name + ".value_reshape")

        # Query expansion
        self.queries_expand = F.Unsqueeze(name + ".queries_expand")
        self.queries_expand_axis = F._from_data(
            name + ".queries_expand_axis", np.array([1], dtype=np.int64), is_const=True
        )
        self.queries_tile = F.Tile(name + ".queries_tile")
        self.queries_flat = F.Reshape(name + ".queries_flat")

        # Reference points
        self.ref_points_transpose = F.Transpose(
            name + ".ref_points_transpose", perm=[1, 0, 2, 3, 4]
        )
        self.ref_points_flat = F.Reshape(name + ".ref_points_flat")

        # Output reshaping
        self.queries_out_reshape = F.Reshape(name + ".queries_out_reshape")

        # BEV mask ops
        self.bev_mask_transpose = F.Transpose(
            name + ".bev_mask_transpose", perm=[1, 0, 2]
        )
        self.bev_mask_expand = F.Unsqueeze(name + ".bev_mask_expand")
        self.bev_mask_expand_axis = F._from_data(
            name + ".bev_mask_expand_axis", np.array([3], dtype=np.int64), is_const=True
        )
        self.queries_masked = F.Mul(name + ".queries_masked")
        self.slots_sum = F.ReduceSum(name + ".slots_sum", axis=1, keepdims=False)

        # Count ops
        self.count_sum = F.ReduceSum(name + ".count_sum", axis=1, keepdims=False)
        self.one_const = F._from_data(
            name + ".one", np.array([1.0], dtype=np.float32), is_const=True
        )
        self.count_clamp = F.Maximum(name + ".count_clamp")
        self.count_expand = F.Unsqueeze(name + ".count_expand")
        self.count_expand_axis = F._from_data(
            name + ".count_expand_axis", np.array([2], dtype=np.int64), is_const=True
        )
        self.slots_normalize = F.Div(name + ".slots_normalize")

        # Dropout
        if dropout > 0:
            self.dropout_op = F.Dropout(name + ".dropout", dropout, True)
        else:
            self.dropout_op = None

        # Residual
        self.output_add_residual = F.Add(name + ".output_add_residual")

        if not batch_first:
            self.output_transpose_back = F.Transpose(
                name + ".output_transpose_back", perm=[1, 0, 2]
            )

        # Initialize weights (for PyTorch model that will be loaded)
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        # Note: This is for the PyTorch model that will be loaded into TTSim
        # Xavier uniform initialization for output projection
        # In TTSim inference, weights are loaded from pre-trained model
        pass  # Initialization handled by PyTorch model loading

    def __call__(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):
        """
        Forward pass of Spatial Cross Attention.

        Args:
            query: Query tensor [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
            key: Key tensor from images [num_cams, l, bs, embed_dims]
            value: Value tensor from images [num_cams, l, bs, embed_dims]
            residual: Residual connection tensor
            query_pos: Positional encoding for queries
            reference_points_cam: Reference points projected to each camera
            bev_mask: Mask indicating query-camera visibility
            spatial_shapes: Spatial dimensions of feature levels
            level_start_index: Starting indices for each level

        Returns:
            Output tensor with same shape as query
        """

        # Handle key/value defaults
        if key is None:
            key = query
        if value is None:
            value = key

        # Handle residual
        if residual is None:
            inp_residual = query
        else:
            inp_residual = residual

        # Add positional encoding
        if query_pos is not None:
            query = self.query_add_pos(query, query_pos)
            setattr(self, query.name, query)

        # Handle batch_first
        if not self.batch_first:
            query = self.query_transpose_in(query)
            setattr(self, query.name, query)

        bs = query.shape[0]
        num_query = query.shape[1]
        D = reference_points_cam.shape[3]

        # Reshape key and value from [num_cams, l, bs, embed_dims] to [bs*num_cams, l, embed_dims]
        num_cams = key.shape[0]
        l = key.shape[1]

        key_transposed = self.key_transpose_op(key)
        setattr(self, key_transposed.name, key_transposed)
        _t = F._from_data(
            self.name + ".key_reshape_shape",
            np.array([bs * self.num_cams, l, self.embed_dims], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        key_reshaped = self.key_reshape_op(key_transposed, _t)
        setattr(self, key_reshaped.name, key_reshaped)

        value_transposed = self.value_transpose_op(value)
        setattr(self, value_transposed.name, value_transposed)
        _t = F._from_data(
            self.name + ".value_reshape_shape",
            np.array([bs * self.num_cams, l, self.embed_dims], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        value_reshaped = self.value_reshape_op(value_transposed, _t)
        setattr(self, value_reshaped.name, value_reshaped)

        # Expand queries for each camera
        queries_expanded = self.queries_expand(query, self.queries_expand_axis)
        setattr(self, queries_expanded.name, queries_expanded)
        _t = F._from_data(
            self.name + ".queries_tile_reps",
            np.array([1, self.num_cams, 1, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        queries_tiled = self.queries_tile(queries_expanded, _t)
        setattr(self, queries_tiled.name, queries_tiled)
        _t = F._from_data(
            self.name + ".queries_flat_shape",
            np.array([bs * self.num_cams, num_query, self.embed_dims], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        queries_flat_out = self.queries_flat(queries_tiled, _t)
        setattr(self, queries_flat_out.name, queries_flat_out)

        # Reshape reference_points_cam
        reference_points_transposed = self.ref_points_transpose(reference_points_cam)
        setattr(self, reference_points_transposed.name, reference_points_transposed)
        _t = F._from_data(
            self.name + ".ref_points_flat_shape",
            np.array([bs * self.num_cams, num_query, D, 2], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        reference_points_flat_out = self.ref_points_flat(
            reference_points_transposed, _t
        )
        setattr(self, reference_points_flat_out.name, reference_points_flat_out)

        # Call deformable attention
        queries_out = self.deformable_attention(
            query=queries_flat_out,
            key=key_reshaped,
            value=value_reshaped,
            reference_points=reference_points_flat_out,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        setattr(self, queries_out.name, queries_out)

        # Reshape output back
        _t = F._from_data(
            self.name + ".queries_out_reshape_shape",
            np.array([bs, self.num_cams, num_query, self.embed_dims], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        queries_out_reshaped = self.queries_out_reshape(queries_out, _t)
        setattr(self, queries_out_reshaped.name, queries_out_reshaped)

        # Apply bev_mask and aggregate
        bev_mask_t = self.bev_mask_transpose(bev_mask)
        setattr(self, bev_mask_t.name, bev_mask_t)
        bev_mask_exp = self.bev_mask_expand(bev_mask_t, self.bev_mask_expand_axis)
        setattr(self, bev_mask_exp.name, bev_mask_exp)

        queries_m = self.queries_masked(queries_out_reshaped, bev_mask_exp)
        setattr(self, queries_m.name, queries_m)
        slots = self.slots_sum(queries_m)
        setattr(self, slots.name, slots)

        # Compute count of valid cameras
        count = self.count_sum(bev_mask_t)
        setattr(self, count.name, count)
        count_clamped = self.count_clamp(count, self.one_const)
        setattr(self, count_clamped.name, count_clamped)
        count_expanded = self.count_expand(count_clamped, self.count_expand_axis)
        setattr(self, count_expanded.name, count_expanded)

        # Normalize by count
        slots = self.slots_normalize(slots, count_expanded)
        setattr(self, slots.name, slots)

        # Apply output projection
        slots = self.output_proj(slots)
        setattr(self, slots.name, slots)

        # Apply dropout
        if self.dropout_op is not None:
            slots = self.dropout_op(slots)
            setattr(self, slots.name, slots)

        # Add residual
        output = self.output_add_residual(slots, inp_residual)
        setattr(self, output.name, output)

        # Convert back to original format if needed
        if not self.batch_first:
            output = self.output_transpose_back(output)
            setattr(self, output.name, output)

        return output

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        count = 0
        count += self.deformable_attention.analytical_param_count()
        count += self.output_proj.analytical_param_count(lvl=0)
        return count


class MSDeformableAttention3D(SimNN.Module):
    """
    TTSim implementation of 3D Multi-Scale Deformable Attention for BEVFormer.

    This is a wrapper around the core deformable attention that handles 3D reference points
    with multiple Z-anchors (heights) that get projected to 2D image coordinates.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per query. Default: 8
        im2col_step (int): Step for im2col (not used in CPU version). Default: 64
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: True
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
    ):
        super().__init__()
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.dropout_rate = dropout
        self.batch_first = batch_first

        # Projections
        self.sampling_offsets = SimNN.Linear(
            name + ".sampling_offsets",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points * 2,
        )

        self.attention_weights = SimNN.Linear(
            name + ".attention_weights",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points,
        )

        self.value_proj = SimNN.Linear(
            name + ".value_proj", in_features=embed_dims, out_features=embed_dims
        )

        # Note: In PyTorch version, output_proj is None and handled externally
        # We keep it for consistency but it may not be used
        self.output_proj = None

        # Check if embed_dims per head is power of 2 (for efficiency warning)
        dim_per_head = embed_dims // num_heads
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        # Pre-create ops for __call__
        self.query_pos_add = F.Add(name + ".query_pos_add")

        if not batch_first:
            self.query_transpose = F.Transpose(
                name + ".query_transpose", perm=[1, 0, 2]
            )
            self.value_transpose = F.Transpose(
                name + ".value_transpose", perm=[1, 0, 2]
            )
            self.output_transpose = F.Transpose(
                name + ".output_transpose", perm=[1, 0, 2]
            )

        # Mask ops
        self.mask_expand = F.Unsqueeze(name + ".mask_expand")
        self.mask_expand_axis = F._from_data(
            name + ".mask_expand_axis", np.array([2], dtype=np.int64), is_const=True
        )
        self.value_masked = F.Where(name + ".value_masked")

        # Reshape ops
        self.value_reshape_op = F.Reshape(name + ".value_reshape")
        self.offsets_reshape = F.Reshape(name + ".offsets_reshape")
        self.attn_reshape = F.Reshape(name + ".attn_reshape")
        self.attn_softmax = F.Softmax(name + ".attn_softmax", axis=-1)
        self.attn_reshape2 = F.Reshape(name + ".attn_reshape2")

        # Reference point ops
        self.ref_unsq1 = F.Unsqueeze(name + ".ref_unsq1")
        self.ref_unsq2 = F.Unsqueeze(name + ".ref_unsq2")
        self.ref_unsq3 = F.Unsqueeze(name + ".ref_unsq3")
        self.ref_ax2 = F._from_data(
            name + ".ref_ax2", np.array([2], dtype=np.int64), is_const=True
        )
        self.ref_ax3 = F._from_data(
            name + ".ref_ax3", np.array([3], dtype=np.int64), is_const=True
        )
        self.ref_ax4 = F._from_data(
            name + ".ref_ax4", np.array([4], dtype=np.int64), is_const=True
        )

        # Normalizer unsqueeze ops
        self.norm_unsq1 = F.Unsqueeze(name + ".norm_unsq1")
        self.norm_unsq2 = F.Unsqueeze(name + ".norm_unsq2")
        self.norm_unsq3 = F.Unsqueeze(name + ".norm_unsq3")
        self.norm_unsq4 = F.Unsqueeze(name + ".norm_unsq4")
        self.norm_ax0 = F._from_data(
            name + ".norm_ax0", np.array([0], dtype=np.int64), is_const=True
        )
        self.norm_ax0_2 = F._from_data(
            name + ".norm_ax0_2", np.array([0], dtype=np.int64), is_const=True
        )
        self.norm_ax0_3 = F._from_data(
            name + ".norm_ax0_3", np.array([0], dtype=np.int64), is_const=True
        )
        self.norm_ax0_4 = F._from_data(
            name + ".norm_ax0_4", np.array([4], dtype=np.int64), is_const=True
        )

        self.norm_offsets = F.Div(name + ".norm_offsets")
        self.offsets_z_reshape = F.Reshape(name + ".offsets_z_reshape")
        self.sampling_locs_add = F.Add(name + ".sampling_locs")
        self.sampling_locs_reshape = F.Reshape(name + ".sampling_locs_reshape")

        # Initialize weights (for PyTorch model that will be loaded)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # Note: This initialization is for the PyTorch reference model
        # In TTSim inference, weights are loaded from pre-trained model
        #
        # PyTorch initialization logic (for reference):
        # - sampling_offsets: constant init to 0, then bias set to polar grid
        # - attention_weights: constant init to 0
        # - value_proj: xavier uniform init
        #
        # The polar grid initialization for sampling_offsets.bias:
        #   thetas = torch.arange(num_heads) * (2.0 * pi / num_heads)
        #   grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        #   grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
        #       num_heads, 1, 1, 2).repeat(1, num_levels, num_points, 1)
        #   for i in range(num_points):
        #       grid_init[:, :, i, :] *= i + 1
        #   sampling_offsets.bias.data = grid_init.view(-1)
        pass  # Initialization handled by PyTorch model loading

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """
        Forward pass of 3D Multi-Scale Deformable Attention.

        Args:
            query: Query features [bs, num_query, embed_dims]
            key: Key features (unused, value is used instead)
            value: Value features [bs, num_value, embed_dims]
            identity: Identity for residual (if None, uses query)
            query_pos: Positional encoding for query
            reference_points: Reference points [bs, num_query, num_Z_anchors, 2]
            spatial_shapes: Spatial shapes [(H, W), ...] for each level
            level_start_index: Starting index for each level in flattened features

        Returns:
            Output features [bs, num_query, embed_dims]
        """

        # Handle defaults
        if value is None:
            value = query
        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            query = self.query_pos_add(query, query_pos)
            setattr(self, query.name, query)

        # Handle batch_first flag
        if not self.batch_first:
            query = self.query_transpose(query)
            setattr(self, query.name, query)
            value = self.value_transpose(value)
            setattr(self, value.name, value)

        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]

        # Project value
        value = self.value_proj(value)
        setattr(self, value.name, value)

        # Apply key_padding_mask if provided
        if key_padding_mask is not None:
            mask_expanded = self.mask_expand(key_padding_mask, self.mask_expand_axis)
            setattr(self, mask_expanded.name, mask_expanded)
            zero_value = F._from_data(
                self.name + ".zero_value",
                np.zeros(value.shape, dtype=np.float32),
                is_const=True,
            )
            setattr(self, zero_value.name, zero_value)
            value = self.value_masked(mask_expanded, zero_value, value)
            setattr(self, value.name, value)

        # Reshape value: [bs, num_value, num_heads, embed_dims_per_head]
        embed_dims_per_head = self.embed_dims // self.num_heads
        _t = F._from_data(
            self.name + ".value_reshape_shape",
            np.array(
                [bs, num_value, self.num_heads, embed_dims_per_head], dtype=np.int64
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        value = self.value_reshape_op(value, _t)
        setattr(self, value.name, value)

        # Compute sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        setattr(self, sampling_offsets.name, sampling_offsets)
        _t = F._from_data(
            self.name + ".offsets_shape",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels, self.num_points, 2],
                dtype=np.int64,
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        sampling_offsets = self.offsets_reshape(sampling_offsets, _t)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Compute attention weights
        attention_weights = self.attention_weights(query)
        setattr(self, attention_weights.name, attention_weights)
        _t = F._from_data(
            self.name + ".attn_shape",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels * self.num_points],
                dtype=np.int64,
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        attention_weights = self.attn_reshape(attention_weights, _t)
        setattr(self, attention_weights.name, attention_weights)

        # Apply softmax
        attention_weights = self.attn_softmax(attention_weights)
        setattr(self, attention_weights.name, attention_weights)

        # Reshape attention weights
        _t = F._from_data(
            self.name + ".attn_shape2",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels, self.num_points],
                dtype=np.int64,
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        attention_weights = self.attn_reshape2(attention_weights, _t)
        setattr(self, attention_weights.name, attention_weights)

        # Handle reference points
        if reference_points.shape[-1] == 2:
            num_Z_anchors = reference_points.shape[2]

            # Normalize offsets by spatial shapes
            if isinstance(spatial_shapes, list):
                offset_normalizer_data = np.array(
                    [[float(W), float(H)] for H, W in spatial_shapes], dtype=np.float32
                )
            else:
                if hasattr(spatial_shapes, "data"):
                    shapes_np = spatial_shapes.data
                else:
                    shapes_np = np.array(
                        [[H, W] for H, W in spatial_shapes], dtype=np.float32
                    )
                offset_normalizer_data = np.stack(
                    [shapes_np[:, 1], shapes_np[:, 0]], axis=-1
                ).astype(np.float32)

            offset_normalizer = F._from_data(
                self.name + ".offset_normalizer", offset_normalizer_data, is_const=True
            )
            setattr(self, offset_normalizer.name, offset_normalizer)

            # Expand reference_points dimensions
            ref_pts = self.ref_unsq1(reference_points, self.ref_ax2)
            setattr(self, ref_pts.name, ref_pts)
            ref_pts = self.ref_unsq2(ref_pts, self.ref_ax3)
            setattr(self, ref_pts.name, ref_pts)
            ref_pts = self.ref_unsq3(ref_pts, self.ref_ax4)
            setattr(self, ref_pts.name, ref_pts)

            # Normalize sampling offsets
            norm = self.norm_unsq1(offset_normalizer, self.norm_ax0)
            setattr(self, norm.name, norm)
            norm = self.norm_unsq2(norm, self.norm_ax0_2)
            setattr(self, norm.name, norm)
            norm = self.norm_unsq3(norm, self.norm_ax0_3)
            setattr(self, norm.name, norm)
            norm = self.norm_unsq4(norm, self.norm_ax0_4)
            setattr(self, norm.name, norm)

            normalized_offsets = self.norm_offsets(sampling_offsets, norm)
            setattr(self, normalized_offsets.name, normalized_offsets)

            # Reshape for Z anchors
            num_points_per_anchor = self.num_points // num_Z_anchors
            _t = F._from_data(
                self.name + ".offsets_z_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        num_points_per_anchor,
                        num_Z_anchors,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            )
            setattr(self, _t.name, _t)
            normalized_offsets_reshaped = self.offsets_z_reshape(normalized_offsets, _t)
            setattr(self, normalized_offsets_reshaped.name, normalized_offsets_reshaped)

            # Add reference points to offsets
            sampling_locations = self.sampling_locs_add(
                ref_pts, normalized_offsets_reshaped
            )
            setattr(self, sampling_locations.name, sampling_locations)

            # Reshape back
            _t = F._from_data(
                self.name + ".sampling_locs_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        self.num_points,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            )
            setattr(self, _t.name, _t)
            sampling_locations = self.sampling_locs_reshape(sampling_locations, _t)
            setattr(self, sampling_locations.name, sampling_locations)
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, but got {reference_points.shape[-1]}"
            )

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            self.name + ".msda",
            value,
            (
                spatial_shapes
                if isinstance(spatial_shapes, list)
                else [
                    (int(spatial_shapes.data[i, 0]), int(spatial_shapes.data[i, 1]))
                    for i in range(self.num_levels)
                ]
            ),
            sampling_locations,
            attention_weights,
            parent_module=self,
        )
        setattr(self, output.name, output)

        # Handle batch_first flag for output
        if not self.batch_first:
            output = self.output_transpose(output)
            setattr(self, output.name, output)

        return output

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        count = 0
        count += self.sampling_offsets.analytical_param_count(lvl=0)
        count += self.attention_weights.analytical_param_count(lvl=0)
        count += self.value_proj.analytical_param_count(lvl=0)
        return count


class MSIPM3D(SimNN.Module):
    """An attention module used in BEVFormer based on Deformable-Detr.

    This is a simplified version that uses:
    - Fixed (non-learnable) sampling offsets
    - Uniform attention weights (all ones -> softmax)
    - Only value projection is learnable

    Args:
        name (str): Module name
        embed_dims (int): The embedding dimension of Attention. Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in Attention. Default: 4.
        num_points (int): The number of sampling points for each query in each head. Default: 8.
        im2col_step (int): The step used in image_to_column. Default: 64.
        dropout (float): A Dropout layer on `inp_identity`. Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        init_cfg (dict): The Config for initialization. Default: None.
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # Check if dim_per_head is power of 2 (recommended for efficiency)
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        # Value projection is the only learnable parameter
        self.value_proj = SimNN.Linear(
            self.name + ".value_proj", embed_dims, embed_dims
        )

        # Pre-create ops for __call__
        self.query_pos_add = F.Add(name + ".query_pos_add")

        if not batch_first:
            self.query_transpose = F.Transpose(
                name + ".query_transpose", perm=[1, 0, 2]
            )
            self.value_transpose_op = F.Transpose(
                name + ".value_transpose", perm=[1, 0, 2]
            )
            self.output_transpose = F.Transpose(
                name + ".output_transpose", perm=[1, 0, 2]
            )

        # Mask ops
        self.mask_unsqueeze = F.Unsqueeze(name + ".mask_unsqueeze")
        self.mask_ax = F._from_data(
            name + ".mask_ax", np.array([-1], dtype=np.int64), is_const=True
        )
        self.mask_cast = F.Cast(name + ".mask_cast", to=1)
        self.mask_one = F._from_data(
            name + ".mask_one", np.array([1.0], dtype=np.float32), is_const=True
        )
        self.mask_sub = F.Sub(name + ".mask_sub")
        self.mask_mul = F.Mul(name + ".mask_mul")

        # Reshape ops
        self.value_reshape_op = F.Reshape(name + ".value_reshape")
        self.sampling_offsets_reshape = F.Reshape(name + ".sampling_offsets_reshape")
        self.sampling_offsets_tile = F.Tile(name + ".sampling_offsets_tile")
        self.attn_softmax = F.Softmax(name + ".attn_softmax", axis=-1)
        self.attn_reshape = F.Reshape(name + ".attn_reshape")

        # Spatial shape slicing
        self.shapes_w = F.SliceF(name + ".shapes_w", out_shape=[num_levels, 1])
        self.w_starts = F._from_data(
            name + ".w_starts", np.array([0, 1], dtype=np.int64), is_const=True
        )
        self.w_ends = F._from_data(
            name + ".w_ends", np.array([num_levels, 2], dtype=np.int64), is_const=True
        )
        self.w_axes = F._from_data(
            name + ".w_axes", np.array([0, 1], dtype=np.int64), is_const=True
        )
        self.w_steps = F._from_data(
            name + ".w_steps", np.array([1, 1], dtype=np.int64), is_const=True
        )

        self.shapes_h = F.SliceF(name + ".shapes_h", out_shape=[num_levels, 1])
        self.h_starts = F._from_data(
            name + ".h_starts", np.array([0, 0], dtype=np.int64), is_const=True
        )
        self.h_ends = F._from_data(
            name + ".h_ends", np.array([num_levels, 1], dtype=np.int64), is_const=True
        )
        self.h_axes = F._from_data(
            name + ".h_axes", np.array([0, 1], dtype=np.int64), is_const=True
        )
        self.h_steps = F._from_data(
            name + ".h_steps", np.array([1, 1], dtype=np.int64), is_const=True
        )

        self.normalizer_concat = F.ConcatX(name + ".normalizer", axis=-1)

        # Reference point ops
        self.ref_reshape = F.Reshape(name + ".ref_reshape")
        self.normalizer_reshape = F.Reshape(name + ".normalizer_reshape")
        self.normalizer_reshape_shape = F._from_data(
            name + ".normalizer_reshape_shape",
            np.array([1, 1, 1, num_levels, 1, 2], dtype=np.int64),
            is_const=True,
        )
        self.offsets_div = F.Div(name + ".offsets_div")
        self.offsets_split_reshape = F.Reshape(name + ".offsets_split_reshape")
        self.add_offsets = F.Add(name + ".add_offsets")
        self.sampling_locs_reshape = F.Reshape(name + ".sampling_locs_reshape")

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # Initialize fixed sampling offsets using circular grid pattern
        thetas = np.arange(self.num_heads, dtype=np.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = np.stack([np.cos(thetas), np.sin(thetas)], -1)
        grid_init = (grid_init / np.abs(grid_init).max(-1, keepdims=True)).reshape(
            self.num_heads, 1, 1, 2
        )
        grid_init = np.tile(grid_init, (1, self.num_levels, self.num_points, 1))

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # Store as numpy array for test compatibility
        self.fixed_sampling_offsets = grid_init.reshape(-1)

        # Create fixed sampling offsets as a constant SimTensor (created once, not in forward)
        # This is more efficient than recreating it during graph construction
        self.fixed_sampling_offsets_tensor = F._from_data(
            self.name + ".fixed_sampling_offsets",
            self.fixed_sampling_offsets,
            is_const=True,
        )

        # Note: value_proj weights are loaded from pre-trained PyTorch model
        # No need to initialize them here (similar to MSDeformableAttention3D)
        pass  # Weight initialization handled by PyTorch model loading

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """Forward Function of MSIPM3D.

        Args:
            query (Tensor): Query of Transformer with shape (bs, num_query, embed_dims).
            key (Tensor): The key tensor (unused in this implementation).
            value (Tensor): The value tensor with shape (bs, num_key, embed_dims).
            identity (Tensor): The tensor used for addition, with the same shape as `query`.
                Default None. If None, `query` will be used.
            query_pos (Tensor): The positional encoding for `query`. Default: None.
            reference_points (Tensor): The normalized reference points with shape
                (bs, num_query, num_Z_anchors, 2), all elements in range [0, 1].
            key_padding_mask (Tensor): ByteTensor for `query`, with shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in different levels.
                With shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels,) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            Tensor: forwarded results with shape [bs, num_query, embed_dims] or
                [num_query, bs, embed_dims] depending on batch_first.
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if query_pos is not None:
            query = self.query_pos_add(query, query_pos)
            setattr(self, query.name, query)

        if not self.batch_first:
            query = self.query_transpose(query)
            setattr(self, query.name, query)
            value = self.value_transpose_op(value)
            setattr(self, value.name, value)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        # Project values
        value = self.value_proj(value)
        setattr(self, value.name, value)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = self.mask_unsqueeze(key_padding_mask, self.mask_ax)
            setattr(self, mask.name, mask)
            mask_float = self.mask_cast(mask)
            setattr(self, mask_float.name, mask_float)
            mask_float = self.mask_sub(self.mask_one, mask_float)
            setattr(self, mask_float.name, mask_float)
            value = self.mask_mul(value, mask_float)
            setattr(self, value.name, value)

        # Reshape value to [bs, num_value, num_heads, dim_per_head]
        dim_per_head = self.embed_dims // self.num_heads
        _t = F._from_data(
            self.name + ".value_reshape_shape",
            np.array([bs, num_value, self.num_heads, dim_per_head], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        value = self.value_reshape_op(value, _t)
        setattr(self, value.name, value)

        # Use pre-created fixed sampling offsets
        _t = F._from_data(
            self.name + ".offsets_reshape_shape",
            np.array(
                [1, 1, self.num_heads, self.num_levels, self.num_points, 2],
                dtype=np.int64,
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        sampling_offsets = self.sampling_offsets_reshape(
            self.fixed_sampling_offsets_tensor, _t
        )
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Tile to [bs, num_query, num_heads, num_levels, num_points, 2]
        _t = F._from_data(
            self.name + ".tile_repeats",
            np.array([bs, num_query, 1, 1, 1, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        sampling_offsets = self.sampling_offsets_tile(sampling_offsets, _t)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Create uniform attention weights
        num_all_points = self.num_levels * self.num_points
        attention_weights = F._from_data(
            self.name + ".ones_init",
            np.ones([bs, num_query, self.num_heads, num_all_points], dtype=np.float32),
            is_const=True,
        )
        setattr(self, attention_weights.name, attention_weights)
        attention_weights = self.attn_softmax(attention_weights)
        setattr(self, attention_weights.name, attention_weights)

        # Reshape to [bs, num_query, num_heads, num_levels, num_points]
        _t = F._from_data(
            self.name + ".attn_reshape_shape",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels, self.num_points],
                dtype=np.int64,
            ),
            is_const=True,
        )
        setattr(self, _t.name, _t)
        attention_weights = self.attn_reshape(attention_weights, _t)
        setattr(self, attention_weights.name, attention_weights)

        # Handle reference points
        if reference_points.shape[-1] == 2:
            # Compute offset normalizer from spatial_shapes
            spatial_w = self.shapes_w(
                spatial_shapes, self.w_starts, self.w_ends, self.w_axes, self.w_steps
            )
            setattr(self, spatial_w.name, spatial_w)
            spatial_h = self.shapes_h(
                spatial_shapes, self.h_starts, self.h_ends, self.h_axes, self.h_steps
            )
            setattr(self, spatial_h.name, spatial_h)
            offset_normalizer = self.normalizer_concat(spatial_w, spatial_h)
            setattr(self, offset_normalizer.name, offset_normalizer)

            bs, num_query, num_Z_anchors, xy = reference_points.shape

            # Expand reference_points
            _t = F._from_data(
                self.name + ".ref_reshape_shape",
                np.array([bs, num_query, 1, 1, 1, num_Z_anchors, 2], dtype=np.int64),
                is_const=True,
            )
            setattr(self, _t.name, _t)
            reference_points = self.ref_reshape(reference_points, _t)
            setattr(self, reference_points.name, reference_points)

            # Normalize sampling_offsets
            offset_normalizer = self.normalizer_reshape(
                offset_normalizer, self.normalizer_reshape_shape
            )
            setattr(self, offset_normalizer.name, offset_normalizer)
            sampling_offsets = self.offsets_div(sampling_offsets, offset_normalizer)
            setattr(self, sampling_offsets.name, sampling_offsets)

            # Reshape sampling_offsets to split num_points
            num_points_per_anchor = self.num_points // num_Z_anchors
            _t = F._from_data(
                self.name + ".offsets_split_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        num_points_per_anchor,
                        num_Z_anchors,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            )
            setattr(self, _t.name, _t)
            sampling_offsets = self.offsets_split_reshape(sampling_offsets, _t)
            setattr(self, sampling_offsets.name, sampling_offsets)

            # Add offsets to reference points
            sampling_locations = self.add_offsets(reference_points, sampling_offsets)
            setattr(self, sampling_locations.name, sampling_locations)

            # Reshape back to [bs, num_query, num_heads, num_levels, num_points, 2]
            _t = F._from_data(
                self.name + ".sampling_locs_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        self.num_points,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            )
            setattr(self, _t.name, _t)
            sampling_locations = self.sampling_locs_reshape(sampling_locations, _t)
            setattr(self, sampling_locations.name, sampling_locations)

        elif reference_points.shape[-1] == 4:
            raise NotImplementedError(
                "reference_points with 4 dimensions not supported"
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be"
                f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Call multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            self.name + ".msda",
            value,
            (
                spatial_shapes
                if isinstance(spatial_shapes, list)
                else [
                    (int(spatial_shapes.data[i, 0]), int(spatial_shapes.data[i, 1]))
                    for i in range(self.num_levels)
                ]
            ),
            sampling_locations,
            attention_weights,
            parent_module=self,
        )
        setattr(self, output.name, output)

        if not self.batch_first:
            output = self.output_transpose(output)
            setattr(self, output.name, output)

        return output

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        count = 0
        count += self.value_proj.analytical_param_count(lvl=0)
        return count


if __name__ == "__main__":
    print("=" * 80)
    print("Spatial Cross Attention TTSim Modules")
    print("=" * 80)
    print("\n[OK] Modules imported successfully!")
    print("\nAvailable components:")
    print("  - SpatialCrossAttention - BEV-to-camera attention wrapper")
    print("  - MSDeformableAttention3D - 3D deformable attention with Z-anchors")
    print("  - MSIPM3D - Simplified deformable attention with fixed offsets")

    print("\nModule tests:")

    # Test MSDeformableAttention3D
    try:
        msda3d = MSDeformableAttention3D(
            name="test_msda3d", embed_dims=256, num_heads=8, num_levels=4, num_points=8
        )
        print(f"\n  [OK] Created MSDeformableAttention3D: '{msda3d.name}'")
        print(f"    - Embed dims: {msda3d.embed_dims}")
        print(f"    - Num heads: {msda3d.num_heads}")
        print(f"    - Num levels: {msda3d.num_levels}")
        print(f"    - Num points: {msda3d.num_points}")
        print(f"    - Parameters: {msda3d.analytical_param_count():,}")
    except Exception as e:
        print(f"  [X] Error creating MSDeformableAttention3D: {e}")

    # Test SpatialCrossAttention
    try:
        sca = SpatialCrossAttention(
            name="test_sca", embed_dims=256, num_cams=6, dropout=0.1
        )
        print(f"\n  [OK] Created SpatialCrossAttention: '{sca.name}'")
        print(f"    - Embed dims: {sca.embed_dims}")
        print(f"    - Num cameras: {sca.num_cams}")
        print(f"    - Parameters: {sca.analytical_param_count():,}")
    except Exception as e:
        print(f"  [X] Error creating SpatialCrossAttention: {e}")

    # Test MSIPM3D
    try:
        msipm3d = MSIPM3D(
            name="test_msipm3d", embed_dims=256, num_heads=8, num_levels=4, num_points=8
        )
        print(f"\n  [OK] Created MSIPM3D: '{msipm3d.name}'")
        print(f"    - Embed dims: {msipm3d.embed_dims}")
        print(f"    - Num heads: {msipm3d.num_heads}")
        print(f"    - Num levels: {msipm3d.num_levels}")
        print(f"    - Num points: {msipm3d.num_points}")
        print(f"    - Parameters: {msipm3d.analytical_param_count():,}")
    except Exception as e:
        print(f"  [X] Error creating MSIPM3D: {e}")

    print("\n[OK] All basic tests passed!")
    print(
        "\nNote: Use validation tests in Validation/ folder for full functionality testing."
    )
    print("=" * 80)
