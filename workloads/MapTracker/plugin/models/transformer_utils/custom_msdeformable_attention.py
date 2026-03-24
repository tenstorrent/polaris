#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of CustomMSDeformableAttention for MapTracker decoder.

This module implements the custom multi-scale deformable attention mechanism
used in the MapTracker decoder for map element detection and refinement.

Original: maptracker/plugin/models/transformer_utils/CustomMSDeformableAttention.py
Reference: Deformable DETR paper - https://arxiv.org/pdf/2010.04159.pdf

Key Differences from MSDeformableAttention3D:
1. Has output_proj Linear layer (projection after deformable attention)
2. Optional use_sampling_offsets flag (can disable learned offsets)
3. Different input format: (bs, num_query, num_points, 2) reference points
4. Used in decoder cross-attention (not BEV encoder)

============================================================================
TTSim Implementation Notes
============================================================================

1. Linear Layers:
   - sampling_offsets: [embed_dims -> num_heads * num_levels * num_points * 2]
   - attention_weights: [embed_dims -> num_heads * num_levels * num_points]
   - value_proj: [embed_dims -> embed_dims]
   - output_proj: [embed_dims -> embed_dims] (NEW - not in MSDeformableAttention3D)

2. Reference Points:
   - Input shape: (bs, num_query, num_points, 2)
   - Needs broadcasting to: (bs, num_query, num_heads, num_levels, num_points, 2)
   - Pattern: [:, :, None, None, :, :] expansion

3. Optional Sampling Offsets:
   - When use_sampling_offsets=False: Use zeros instead of learned offsets
   - Allows testing pure deformable attention without offset learning

4. Dropout:
   - Applied to output before residual addition
   - TTSim: F.Dropout operator in forward pass
"""

# -------------------------------PyTorch--------------------------------

# # ---------------------------------------------
# # Copyright (c) OpenMMLab. All rights reserved.
# # ---------------------------------------------
# #  Modified by Zhiqi Li
# # ---------------------------------------------
#
# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
# import mmcv
# import cv2 as cv
# import copy
# import warnings
# from matplotlib import pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import xavier_init, constant_init
# from mmcv.cnn.bricks.registry import (ATTENTION,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import TransformerLayerSequence
# import math
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
# from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
#                         to_2tuple)
#
# from mmcv.utils import ext_loader
# from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttnFunction,
#                                               multi_scale_deformable_attn_pytorch)
# from .fp16_dattn import MultiScaleDeformableAttnFunctionFp32
#
# @ATTENTION.register_module()
# class CustomMSDeformableAttention(BaseModule):
#     """An attention module used in Deformable-Detr.
#
#     `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
#     <https://arxiv.org/pdf/2010.04159.pdf>`_.
#
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
#                  num_points=4,
#                  im2col_step=64,
#                  dropout=0.1,
#                  use_sampling_offsets=True,
#                  batch_first=False,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         if embed_dims % num_heads != 0:
#             raise ValueError(f'embed_dims must be divisible by num_heads, '
#                              f'but got {embed_dims} and {num_heads}')
#         dim_per_head = embed_dims // num_heads
#         self.norm_cfg = norm_cfg
#         self.dropout = nn.Dropout(dropout)
#         self.batch_first = batch_first
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
#         self.use_sampling_offsets = use_sampling_offsets
#         if use_sampling_offsets:
#             self.sampling_offsets = nn.Linear(
#                 embed_dims, num_heads * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dims,
#                                            num_heads * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dims, embed_dims)
#         self.output_proj = nn.Linear(embed_dims, embed_dims)
#         self.init_weights()
#
#     def init_weights(self):
#         """Default initialization for Parameters of Module."""
#         if self.use_sampling_offsets:
#             constant_init(self.sampling_offsets, 0.)
#             thetas = torch.arange(
#                 self.num_heads,
#                 dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
#             grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#             grid_init = (grid_init /
#                         grid_init.abs().max(-1, keepdim=True)[0]).view(
#                 self.num_heads, 1, 1,
#                 2).repeat(1, self.num_levels, self.num_points, 1)
#             for i in range(self.num_points):
#                 grid_init[:, :, i, :] *= i + 1
#
#             self.sampling_offsets.bias.data = grid_init.view(-1)
#         constant_init(self.attention_weights, val=0., bias=0.)
#         xavier_init(self.value_proj, distribution='uniform', bias=0.)
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
#         self._is_init = True
#
#     @deprecated_api_warning({'residual': 'identity'},
#                             cls_name='MultiScaleDeformableAttention')
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
#                 flag='decoder',
#                 **kwargs):
#         """Forward Function of MultiScaleDeformAttention.
#
#         Args:
#             query (Tensor): Query of Transformer with shape
#                 (num_query, bs, embed_dims).
#             key (Tensor): The key tensor with shape
#                 `(num_key, bs, embed_dims)`.
#             value (Tensor): The value tensor with shape
#                 `(num_key, bs, embed_dims)`.
#             identity (Tensor): The tensor used for addition, with the
#                 same shape as `query`. Default None. If None,
#                 `query` will be used.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`. Default
#                 None.
#             reference_points (Tensor):  The normalized reference
#                 points with shape (bs, num_query, num_levels, num_points, 2),
#                 all elements is range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_key].
#             spatial_shapes (Tensor): Spatial shape of features in
#                 different levels. With shape (num_levels, 2),
#                 last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape ``(num_levels, )`` and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#
#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """
#
#         if value is None:
#             value = query
#
#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
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
#
#         if self.use_sampling_offsets:
#             sampling_offsets = self.sampling_offsets(query).view(
#                 bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
#         else:
#             sampling_offsets = query.new_zeros((bs, num_query, self.num_heads, self.num_levels, self.num_points, 2))
#
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points)
#         attention_weights = attention_weights.softmax(-1)
#
#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_levels,
#                                                    self.num_points)
#
#         # TODO: try remove sampling offsets
#         offset_normalizer = torch.stack(
#             [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1) # changed to (h, w)
#         _, _, num_points, _ = reference_points.shape
#         # (bs, num_queries, num_pts, 2) ->
#         # (bs, num_queries, num_heads, num_lvls, num_pts, 2)
#         reference_points = reference_points[:, :, None, None, :, :]
#         # reference_points[..., 1:2] = -reference_points[..., 1:2]
#         sampling_locations = reference_points + \
#             (sampling_offsets # (bs, num_queries, num_heads, num_lvls, num_pts, 2)
#             / offset_normalizer[None, None, None, :, None, :])
#         assert list(sampling_locations.shape) == [bs, num_query, self.num_heads, self.num_levels, num_points, 2]
#
#         if torch.cuda.is_available() and value.is_cuda:
#             # using fp16 deformable attention is unstable because it performs many sum operations
#             output = MultiScaleDeformableAttnFunctionFp32.apply(
#                 value, spatial_shapes, level_start_index, sampling_locations,
#                 attention_weights, self.im2col_step)
#         else:
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights)
#
#         output = self.output_proj(output)
#
#         if not self.batch_first:
#             # (num_query, bs ,embed_dims)
#             output = output.permute(1, 0, 2)
#
#         return self.dropout(output) + identity

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

# Import deformable attention implementation
from workloads.MapTracker.plugin.models.backbones.bevformer.multi_scale_deformable_attn import (
    multi_scale_deformable_attn_ttsim,
    MultiScaleDeformableAttention,
)


class CustomMSDeformableAttention(SimNN.Module):
    """
    Custom Multi-Scale Deformable Attention for MapTracker decoder.

    This attention mechanism allows decoder queries to attend to multi-scale
    BEV features using learned sampling locations and attention weights.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per query per head. Default: 4
        im2col_step (int): Step for image-to-column operation. Default: 64
        dropout (float): Dropout rate. Default: 0.1
        use_sampling_offsets (bool): Whether to use learned sampling offsets. Default: True
        batch_first (bool): If True, input is (bs, num_query, embed_dims). Default: False
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        use_sampling_offsets=True,
        batch_first=False,
    ):
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        self.name = name
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.dropout_rate = dropout
        self.use_sampling_offsets = use_sampling_offsets
        self.batch_first = batch_first

        dim_per_head = embed_dims // num_heads

        # Check if dim_per_head is power of 2 (more efficient for CUDA)
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    f"invalid input for _is_power_of_2: {n} (type: {type(n)})"
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in CustomMSDeformableAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        # Linear layers
        if use_sampling_offsets:
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

        self.output_proj = SimNN.Linear(
            name + ".output_proj", in_features=embed_dims, out_features=embed_dims
        )

        # Store dropout rate (apply conditionally in forward)
        self.dropout_rate = dropout

        # Reshape operators (pre-created)
        self.value_reshape = F.Reshape(name + ".value_reshape")
        self.offsets_reshape = F.Reshape(name + ".offsets_reshape")
        self.attn_weights_reshape1 = F.Reshape(name + ".attn_weights_reshape1")
        self.attn_weights_reshape2 = F.Reshape(name + ".attn_weights_reshape2")

        # Softmax
        self.softmax = F.Softmax(name + ".softmax", axis=-1)

        # Query position addition
        self.query_pos_add = F.Add(name + ".query_pos_add")

        # Mask operations (used when key_padding_mask is provided)
        self.mask_expand_unsq = F.Unsqueeze(name + ".mask_expand")
        self.mask_expand_axis = F._from_data(
            name + ".mask_expand_axis", np.array([2], dtype=np.int64), is_const=True
        )
        self.masked_fill_where = F.Where(name + ".masked_fill")

        # Spatial shape slicing
        self.spatial_w_op = F.SliceF(name + ".spatial_w", out_shape=[num_levels, 1])
        self.w_start = F._from_data(
            name + ".w_start", np.array([0, 1], dtype=np.int32), is_const=True
        )
        self.w_end = F._from_data(
            name + ".w_end", np.array([num_levels, 2], dtype=np.int32), is_const=True
        )
        self.w_axes = F._from_data(
            name + ".w_axes", np.array([0, 1], dtype=np.int32), is_const=True
        )
        self.w_steps = F._from_data(
            name + ".w_steps", np.array([1, 1], dtype=np.int32), is_const=True
        )
        self.spatial_h_op = F.SliceF(name + ".spatial_h", out_shape=[num_levels, 1])
        self.h_start = F._from_data(
            name + ".h_start", np.array([0, 0], dtype=np.int32), is_const=True
        )
        self.h_end = F._from_data(
            name + ".h_end", np.array([num_levels, 1], dtype=np.int32), is_const=True
        )
        self.h_axes = F._from_data(
            name + ".h_axes", np.array([0, 1], dtype=np.int32), is_const=True
        )
        self.h_steps = F._from_data(
            name + ".h_steps", np.array([1, 1], dtype=np.int32), is_const=True
        )
        self.offset_normalizer_cat = F.ConcatX(name + ".offset_normalizer", axis=1)

        # Reference point processing
        self.ref_slice_starts = F._from_data(
            name + ".ref_slice_starts", np.array([0], dtype=np.int64), is_const=True
        )
        self.ref_slice_ends = F._from_data(
            name + ".ref_slice_ends", np.array([1], dtype=np.int64), is_const=True
        )
        self.ref_slice_axes = F._from_data(
            name + ".ref_slice_axes", np.array([2], dtype=np.int64), is_const=True
        )
        self.ref_slice_steps = F._from_data(
            name + ".ref_slice_steps", np.array([1], dtype=np.int64), is_const=True
        )
        self.ref_squeeze_axes_tensor = F._from_data(
            name + ".ref_squeeze_axes", np.array([2], dtype=np.int64), is_const=True
        )
        self.ref_unsq2 = F.Unsqueeze(name + ".ref_unsq2")
        self.ref_unsq3 = F.Unsqueeze(name + ".ref_unsq3")
        self.ref_unsq4 = F.Unsqueeze(name + ".ref_unsq4")
        self.ax2_tensor = F._from_data(
            name + ".ax2", np.array([2], dtype=np.int64), is_const=True
        )
        self.ax3_tensor = F._from_data(
            name + ".ax3", np.array([3], dtype=np.int64), is_const=True
        )
        self.ax4_tensor = F._from_data(
            name + ".ax4", np.array([4], dtype=np.int64), is_const=True
        )
        self.ref_tile_op = F.Tile(name + ".ref_tile")
        self.tile_repeats_tensor = F._from_data(
            name + ".tile_repeats",
            np.array([1, 1, 1, 1, num_points, 1], dtype=np.int64),
            is_const=True,
        )

        # Normalizer and sampling
        self.normalizer_reshape_op = F.Reshape(name + ".normalizer_reshape")
        self.normalizer_shape_tensor = F._from_data(
            name + ".normalizer_shape",
            np.array([1, 1, 1, num_levels, 1, 2], dtype=np.int64),
            is_const=True,
        )
        self.normalize_offsets_div = F.Div(name + ".normalize_offsets")
        self.sampling_locations_add = F.Add(name + ".sampling_locations")

        # Dropout and residual
        if dropout > 0:
            self.dropout_op = F.Dropout(name + ".dropout", dropout, True)
        self.residual_add = F.Add(name + ".residual_add")

        # Transpose operators for batch_first handling
        if not batch_first:
            self.query_transpose_in = F.Transpose(
                name + ".query_transpose_in", perm=[1, 0, 2]
            )
            self.value_transpose_in = F.Transpose(
                name + ".value_transpose_in", perm=[1, 0, 2]
            )
            self.output_transpose_out = F.Transpose(
                name + ".output_transpose_out", perm=[1, 0, 2]
            )

    def init_weights(self):
        """
        Initialize weights for PyTorch version.

        Note: This is documentation only - TTSim loads pre-trained weights.

        PyTorch initialization:
        - sampling_offsets.bias: Grid pattern scaled by point index
        - attention_weights: Constant 0
        - value_proj: Xavier uniform
        - output_proj: Xavier uniform
        """
        if self.use_sampling_offsets:
            # Grid initialization pattern
            thetas = np.arange(self.num_heads, dtype=np.float32) * (
                2.0 * math.pi / self.num_heads
            )
            grid_init = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
            grid_init = (
                (grid_init / np.abs(grid_init).max(axis=-1, keepdims=True))
                .reshape(self.num_heads, 1, 1, 2)
                .repeat(self.num_levels, axis=1)
                .repeat(self.num_points, axis=2)
            )

            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1

            # This would be applied to sampling_offsets.bias in PyTorch
            print(f"[{self.name}] Grid init pattern shape: {grid_init.shape}")

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for this module.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count
        """
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            print(f"{indent}CustomMSDeformableAttention '{self.name}':")

        # Sampling offsets linear layer
        if self.use_sampling_offsets:
            offsets_params = self.embed_dims * (
                self.num_heads * self.num_levels * self.num_points * 2
            )
            offsets_params += (
                self.num_heads * self.num_levels * self.num_points * 2
            )  # bias
            total_params += offsets_params
            if lvl >= 2:
                print(f"{indent}  sampling_offsets: {offsets_params:,}")

        # Attention weights linear layer
        attn_weights_params = self.embed_dims * (
            self.num_heads * self.num_levels * self.num_points
        )
        attn_weights_params += (
            self.num_heads * self.num_levels * self.num_points
        )  # bias
        total_params += attn_weights_params
        if lvl >= 2:
            print(f"{indent}  attention_weights: {attn_weights_params:,}")

        # Value projection
        value_params = self.embed_dims * self.embed_dims + self.embed_dims
        total_params += value_params
        if lvl >= 2:
            print(f"{indent}  value_proj: {value_params:,}")

        # Output projection
        output_params = self.embed_dims * self.embed_dims + self.embed_dims
        total_params += output_params
        if lvl >= 2:
            print(f"{indent}  output_proj: {output_params:,}")

        if lvl >= 1:
            print(f"{indent}Total CustomMSDeformableAttention params: {total_params:,}")

        return total_params

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
        Forward pass of CustomMSDeformableAttention.

        Args:
            query: Query tensor (num_query, bs, embed_dims) or (bs, num_query, embed_dims)
            key: Not used (kept for API compatibility)
            value: Value tensor, same shape as query. If None, uses query
            identity: Residual tensor. If None, uses query
            query_pos: Positional encoding for query. Default: None
            key_padding_mask: Mask for value (bs, num_value). Default: None
            reference_points: Reference points (bs, num_query, num_points, 2)
            spatial_shapes: Spatial shapes (num_levels, 2) as [H, W]
            level_start_index: Start indices for each level (num_levels,)

        Returns:
            Output tensor with same shape as query
        """
        # Use query as value if not provided
        if value is None:
            value = query

        # Add positional encoding to query
        if query_pos is not None:
            query = self.query_pos_add(query, query_pos)

        # Handle batch_first vs sequence_first
        if not self.batch_first:
            query = self.query_transpose_in(query)
            value = self.value_transpose_in(value)

        # Get shapes
        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]

        # Value projection
        value = self.value_proj(value)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask_expanded = self.mask_expand_unsq(
                key_padding_mask, self.mask_expand_axis
            )
            zero_value_data = np.zeros(value.shape, dtype=np.float32)
            self.zero_value = F._from_data(
                self.name + ".zero_value", zero_value_data, is_const=True
            )
            value = self.masked_fill_where(mask_expanded, self.zero_value, value)

        # Reshape value: (bs, num_value, embed_dims) -> (bs, num_value, num_heads, head_dim)
        head_dim = self.embed_dims // self.num_heads
        self.value_shape_tensor = F._from_data(
            self.name + ".value_shape",
            np.array([bs, num_value, self.num_heads, head_dim], dtype=np.int64),
            is_const=False,
        )
        value = self.value_reshape(value, self.value_shape_tensor)

        # Compute sampling offsets or use zeros
        if self.use_sampling_offsets:
            sampling_offsets = self.sampling_offsets(query)
            self.offsets_shape_tensor = F._from_data(
                self.name + ".offsets_shape",
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
                is_const=False,
            )
            sampling_offsets = self.offsets_reshape(
                sampling_offsets, self.offsets_shape_tensor
            )
        else:
            self.sampling_offsets_zeros = F._from_data(
                self.name + ".sampling_offsets_zeros",
                np.zeros(
                    (
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        self.num_points,
                        2,
                    ),
                    dtype=np.float32,
                ),
                is_const=False,
            )
            sampling_offsets = self.sampling_offsets_zeros

        # Compute attention weights
        attention_weights = self.attention_weights(query)
        self.attn_shape1_tensor = F._from_data(
            self.name + ".attn_shape1",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels * self.num_points],
                dtype=np.int64,
            ),
            is_const=False,
        )
        attention_weights = self.attn_weights_reshape1(
            attention_weights, self.attn_shape1_tensor
        )

        # Softmax over points
        attention_weights = self.softmax(attention_weights)

        # Reshape to (bs, num_query, num_heads, num_levels, num_points)
        self.attn_shape2_tensor = F._from_data(
            self.name + ".attn_shape2",
            np.array(
                [bs, num_query, self.num_heads, self.num_levels, self.num_points],
                dtype=np.int64,
            ),
            is_const=False,
        )
        attention_weights = self.attn_weights_reshape2(
            attention_weights, self.attn_shape2_tensor
        )

        # Compute offset normalizer: stack [W, H] from spatial_shapes
        spatial_w = self.spatial_w_op(
            spatial_shapes, self.w_start, self.w_end, self.w_axes, self.w_steps
        )
        spatial_h = self.spatial_h_op(
            spatial_shapes, self.h_start, self.h_end, self.h_axes, self.h_steps
        )
        offset_normalizer = self.offset_normalizer_cat(spatial_w, spatial_h)

        # Broadcast reference_points to (bs, num_query, num_heads, num_levels, num_points, 2)
        # Input reference_points can be:
        # - (bs, num_query, 2): single point per query
        # - (bs, num_query, N, 2): N points per query
        #
        # PyTorch source (CustomMSDeformableAttention.forward):
        #   reference_points = reference_points[:, :, None, None, :, :]
        # This broadcasts when N == self.num_points (the normal MapTracker config).
        # When N != self.num_points, we fall back to taking the first point only
        # (same as the PyTorch test references do).
        if len(reference_points.shape) == 4:
            num_ref_pts = reference_points.shape[2]
            if num_ref_pts == self.num_points:
                # Dims match: unsqueeze at axes 2,3 (heads, levels), keep all points
                # (bs, nq, num_points, 2) -> (bs, nq, 1, 1, num_points, 2)
                reference_points_expanded = self.ref_unsq2(
                    reference_points, self.ax2_tensor
                )
                reference_points_expanded = self.ref_unsq3(
                    reference_points_expanded, self.ax3_tensor
                )
            else:
                # Dims don't match: take first reference point, then broadcast
                # (bs, nq, N, 2) -> slice [0:1] on axis 2 -> squeeze -> (bs, nq, 2)
                self.ref_slice_op = F.SliceF(
                    self.name + ".ref_slice", out_shape=[bs, num_query, 1, 2]
                )
                self.ref_slice_op.set_module(self)
                reference_points_sliced = self.ref_slice_op(
                    reference_points,
                    self.ref_slice_starts,
                    self.ref_slice_ends,
                    self.ref_slice_axes,
                    self.ref_slice_steps,
                )
                self.ref_squeeze_op = F.Squeeze(self.name + ".ref_squeeze")
                self.ref_squeeze_op.set_module(self)
                reference_points = self.ref_squeeze_op(
                    reference_points_sliced, self.ref_squeeze_axes_tensor
                )
                # Now 3D (bs, nq, 2) — fall through to unsqueeze + tile path below
                reference_points_expanded = self.ref_unsq2(
                    reference_points, self.ax2_tensor
                )
                reference_points_expanded = self.ref_unsq3(
                    reference_points_expanded, self.ax3_tensor
                )
                reference_points_expanded = self.ref_unsq4(
                    reference_points_expanded, self.ax4_tensor
                )
                reference_points_expanded = self.ref_tile_op(
                    reference_points_expanded, self.tile_repeats_tensor
                )
        else:
            # 3D input: (bs, num_query, 2)
            # Unsqueeze to add num_heads, num_levels, num_points dimensions
            reference_points_expanded = self.ref_unsq2(
                reference_points, self.ax2_tensor
            )
            reference_points_expanded = self.ref_unsq3(
                reference_points_expanded, self.ax3_tensor
            )
            reference_points_expanded = self.ref_unsq4(
                reference_points_expanded, self.ax4_tensor
            )
            # Tile along num_points dimension
            reference_points_expanded = self.ref_tile_op(
                reference_points_expanded, self.tile_repeats_tensor
            )

        # Normalize offsets by spatial shapes
        normalizer_expanded = self.normalizer_reshape_op(
            offset_normalizer, self.normalizer_shape_tensor
        )
        normalized_offsets = self.normalize_offsets_div(
            sampling_offsets, normalizer_expanded
        )

        # Compute sampling locations
        sampling_locations = self.sampling_locations_add(
            reference_points_expanded, normalized_offsets
        )

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            name=self.name + ".msda",
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            parent_module=self,
        )

        # Output projection
        output = self.output_proj(output)

        # Convert back to sequence-first if needed
        if not self.batch_first:
            output = self.output_transpose_out(output)

        # Apply dropout
        if self.dropout_rate > 0:
            output = self.dropout_op(output)

        # Apply residual connection only if identity is provided
        if identity is not None:
            output = self.residual_add(output, identity)

        return output


def analytical_param_count(
    embed_dims=256, num_heads=8, num_levels=4, num_points=4, use_sampling_offsets=True
):
    """
    Standalone function to calculate parameter count.

    Args:
        embed_dims (int): Embedding dimension
        num_heads (int): Number of attention heads
        num_levels (int): Number of feature pyramid levels
        num_points (int): Number of sampling points
        use_sampling_offsets (bool): Whether sampling offsets are used

    Returns:
        int: Total parameter count
    """
    total_params = 0

    # Sampling offsets
    if use_sampling_offsets:
        offsets_params = embed_dims * (num_heads * num_levels * num_points * 2)
        offsets_params += num_heads * num_levels * num_points * 2
        total_params += offsets_params
        print(f"sampling_offsets: {offsets_params:,}")

    # Attention weights
    attn_params = embed_dims * (num_heads * num_levels * num_points)
    attn_params += num_heads * num_levels * num_points
    total_params += attn_params
    print(f"attention_weights: {attn_params:,}")

    # Value projection
    value_params = embed_dims * embed_dims + embed_dims
    total_params += value_params
    print(f"value_proj: {value_params:,}")

    # Output projection
    output_params = embed_dims * embed_dims + embed_dims
    total_params += output_params
    print(f"output_proj: {output_params:,}")

    print(f"Total: {total_params:,}")
    return total_params


if __name__ == "__main__":
    print("CustomMSDeformableAttention TTSim Module")
    print("=" * 80)
    print()
    print("Parameter count for default configuration:")
    analytical_param_count(
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        use_sampling_offsets=True,
    )
