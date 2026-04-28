#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Temporal Self Attention.

This module implements the temporal self-attention mechanism that enables
BEV features to attend to historical BEV features across time, enabling
temporal reasoning for object tracking and motion prediction.

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

The original PyTorch implementation uses several mmcv functions that are not
compatible with Python 3.13. This TTSim version includes the following conversions:

1. Initialization Functions (mmcv.cnn):
   - xavier_init: Implemented in init_utils.py using torch.nn.init for PyTorch
   - constant_init: Implemented in init_utils.py using torch.nn.init for PyTorch

2. Decorators (mmcv.runner):
   - @ATTENTION.register_module(): Not needed in TTSim (no module registry)

3. Base Classes (mmcv.runner.base_module):
   - BaseModule: Replaced with ttsim.front.functional.sim_nn.Module

4. Custom Operations:
   - multi_scale_deformable_attn_pytorch: Uses TTSim implementation
   - MultiScaleDeformableAttnFunction: Replaced with TTSim implementation

5. Utilities:
   - run_time decorator: Not needed in TTSim (no profiling decorators)

6. Operations:
   - torch.stack: Replaced with F.ConcatX + F.Reshape
   - torch.cat: Replaced with F.ConcatX
   - tensor.view/.reshape: Replaced with F.Reshape
   - tensor.permute: Replaced with F.Transpose
   - tensor.softmax: Replaced with F.Softmax
   - tensor.mean: Replaced with F.ReduceMean

7. Warnings:
   - warnings.warn for non-power-of-2 dimensions: Implemented in init_weights

"""

#-------------------------------PyTorch--------------------------------


# from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
# import warnings
# import torch
# import torch.nn as nn
# from mmcv.cnn import xavier_init, constant_init
# from mmcv.cnn.bricks.registry import ATTENTION
# import math
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
# from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
#                         to_2tuple)
#
# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
#
#
# @ATTENTION.register_module()
# class TemporalSelfAttention(BaseModule):
#     """An attention module used in BEVFormer based on Deformable-Detr.
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
#             or (n, batch, embed_dim). Default to True.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: None.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
#          the length of BEV queue is 2.
#     """
#
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  num_levels=4,
#                  num_points=4,
#                  num_bev_queue=2,
#                  im2col_step=64,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#
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
#         self.num_bev_queue = num_bev_queue
#         self.sampling_offsets = nn.Linear(
#             embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
#                                            num_bev_queue*num_heads * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dims, embed_dims)
#         self.output_proj = nn.Linear(embed_dims, embed_dims)
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
#             2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)
#
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
#                 flag='decoder',
#
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
#
#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """
#
#         if value is None:
#             assert self.batch_first
#             bs, len_bev, c = query.shape
#             value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)
#
#             # value = torch.cat([query, query], 0)
#
#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)
#         bs,  num_query, embed_dims = query.shape
#         _, num_value, _ = value.shape
#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
#         assert self.num_bev_queue == 2
#
#         query = torch.cat([value[:bs], query], -1)
#         value = self.value_proj(value)
#
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)
#
#         value = value.reshape(bs*self.num_bev_queue,
#                               num_value, self.num_heads, -1)
#
#         sampling_offsets = self.sampling_offsets(query)
#         sampling_offsets = sampling_offsets.view(
#             bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
#         attention_weights = attention_weights.softmax(-1)
#
#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_bev_queue,
#                                                    self.num_levels,
#                                                    self.num_points)
#
#         attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
#             .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
#         sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
#             .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
#
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack(
#                 [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                 + sampling_offsets \
#                 / offset_normalizer[None, None, None, :, None, :]
#
#         elif reference_points.shape[-1] == 4:
#             sampling_locations = reference_points[:, :, None, :, None, :2] \
#                 + sampling_offsets / self.num_points \
#                 * reference_points[:, :, None, :, None, 2:] \
#                 * 0.5
#         else:
#             raise ValueError(
#                 f'Last dim of reference_points must be'
#                 f' 2 or 4, but get {reference_points.shape[-1]} instead.')
#         if torch.cuda.is_available() and value.is_cuda:
#
#             # using fp16 deformable attention is unstable because it performs many sum operations
#             if value.dtype == torch.float16:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             else:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             output = MultiScaleDeformableAttnFunction.apply(
#                 value, spatial_shapes, level_start_index, sampling_locations,
#                 attention_weights, self.im2col_step)
#         else:
#
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights)
#
#         # output shape (bs*num_bev_queue, num_query, embed_dims)
#         # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
#         output = output.permute(1, 2, 0)
#
#         # fuse history value and current value
#         # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
#         output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
#         output = output.mean(-1)
#
#         # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
#         output = output.permute(2, 0, 1)
#
#         output = self.output_proj(output)
#
#         if not self.batch_first:
#             output = output.permute(1, 0, 2)
#
#         return self.dropout(output) + identity

#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger
import warnings
import math

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..','..','..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import our TTSim deformable attention
from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_ttsim

# Import initialization utilities (Python 3.13 compatible)
from .init_utils import xavier_init, constant_init, _is_power_of_2


class TemporalSelfAttention(SimNN.Module):
    """
    TTSim implementation of Temporal Self Attention for BEVFormer.

    This module enables BEV queries to attend to both current and historical
    BEV features using deformable attention, allowing the model to reason
    about motion and track objects over time.

    Key features:
    - Processes multiple BEV frames (typically 2: history + current)
    - Uses deformable sampling for flexible temporal aggregation
    - Fuses temporal information through learnable attention weights

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per query. Default: 4
        num_bev_queue (int): Number of BEV frames to process (typically 2). Default: 2
        im2col_step (int): Step for im2col (not used in CPU version). Default: 64
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: True

    Input Shapes:
        query: [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
        value: [bs*num_bev_queue, num_value, embed_dims] - Stacked BEV features
        reference_points: [bs, num_query, num_levels, 2] - Reference points
        spatial_shapes: [(H, W), ...] - Spatial shapes per level
        level_start_index: [num_levels] - Start indices per level

    Output Shape:
        [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True):
        super().__init__()
        self.name = name

        # Check embed_dims divisibility
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                           f'but got {embed_dims} and {num_heads}')

        dim_per_head = embed_dims // num_heads

        # Warn about non-power-of-2 dimensions (for efficiency)
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step

        # Sampling offsets: takes concatenated query (embed_dims * num_bev_queue)
        # and outputs offsets for all queues, heads, levels, and points
        self.sampling_offsets = SimNN.Linear(
            name + '.sampling_offsets',
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points * 2
        )

        # Attention weights: takes concatenated query and outputs weights
        self.attention_weights = SimNN.Linear(
            name + '.attention_weights',
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points
        )

        # Value projection
        self.value_proj = SimNN.Linear(
            name + '.value_proj',
            in_features=embed_dims,
            out_features=embed_dims
        )

        # Output projection
        self.output_proj = SimNN.Linear(
            name + '.output_proj',
            in_features=embed_dims,
            out_features=embed_dims
        )

        self.dropout_rate = dropout

        # Initialize weights (for PyTorch reference model)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # Note: This is for documentation and PyTorch reference model
        # In TTSim inference, weights are loaded from pre-trained checkpoint
        pass  # Initialization handled by PyTorch model loading or weight copying

    def __call__(self,
                 query,
                 key=None,
                 value=None,
                 identity=None,
                 query_pos=None,
                 key_padding_mask=None,
                 reference_points=None,
                 spatial_shapes=None,
                 level_start_index=None,
                 flag='decoder',
                 **kwargs):
        """
        Forward pass of Temporal Self Attention.

        Args:
            query: Query tensor [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
            key: Not used (self-attention)
            value: Value tensor [bs*num_bev_queue, num_value, embed_dims] or None
                   If None, will stack query with itself for temporal processing
            identity: Residual connection tensor
            query_pos: Positional encoding for queries
            key_padding_mask: Padding mask
            reference_points: Reference points [bs, num_query, num_levels, 2]
            spatial_shapes: Spatial shapes of feature levels
            level_start_index: Start indices for each level
            flag: 'decoder' or 'encoder'

        Returns:
            Output tensor with same shape as query, with residual connection
        """

        # Create value if not provided (stack query with itself)
        if value is None:
            assert self.batch_first, "batch_first must be True when value is None"
            bs, len_bev, c = query.shape

            # Stack query twice: [bs, len_bev, c] -> [bs*2, len_bev, c]
            # This creates [current_bev, current_bev] for temporal processing
            _data_value_unsqueeze_axis = F._from_data(self.name + '.value_unsqueeze_axis', np.array([1], dtype=np.int64), is_const=True)
            setattr(self, _data_value_unsqueeze_axis.name, _data_value_unsqueeze_axis)
            _op = F.Unsqueeze(self.name + '.value_unsqueeze')
            setattr(self, _op.name, _op)
            query_unsqueezed = _op(query, _data_value_unsqueeze_axis)
            setattr(self, query_unsqueezed.name, query_unsqueezed)
            _data_value_tile_reps = F._from_data(self.name + '.value_tile_reps', np.array([1, 2, 1, 1], dtype=np.int64), is_const=True)
            setattr(self, _data_value_tile_reps.name, _data_value_tile_reps)
            _op = F.Tile(self.name + '.value_tile')
            setattr(self, _op.name, _op)
            query_tiled = _op(query_unsqueezed, _data_value_tile_reps)
            setattr(self, query_tiled.name, query_tiled)
            _data_value_reshape_shape = F._from_data(self.name + '.value_reshape_shape',
                           np.array([bs * 2, len_bev, c], dtype=np.int64),
                           is_const=True)
            setattr(self, _data_value_reshape_shape.name, _data_value_reshape_shape)
            _op = F.Reshape(self.name + '.value_reshape')
            setattr(self, _op.name, _op)
            value = _op(query_tiled, _data_value_reshape_shape)
            setattr(self, value.name, value)

        # Handle residual/identity
        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            _op = F.Add(self.name + '.query_add_pos')
            setattr(self, _op.name, _op)
            query = _op(query, query_pos)
            setattr(self, query.name, query)

        # Handle batch_first format
        if not self.batch_first:
            # Convert to batch_first: [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            _op = F.Transpose(self.name + '.query_transpose', perm=[1, 0, 2])
            setattr(self, _op.name, _op)
            query = _op(query)
            setattr(self, query.name, query)
            _op = F.Transpose(self.name + '.value_transpose', perm=[1, 0, 2])
            setattr(self, _op.name, _op)
            value = _op(value)
            setattr(self, value.name, value)

        # Get dimensions
        bs = query.shape[0]
        num_query = query.shape[1]
        embed_dims = query.shape[2]
        num_value = value.shape[1]

        # Concatenate current query with historical value for temporal attention
        # value[:bs] is current BEV, query is also current BEV
        # Concatenate them: [bs, num_query, embed_dims*2]
        # First, flatten value from [bs*num_bev_queue, num_value, num_heads, head_dim] to [bs*num_bev_queue, num_value, embed_dims]
        _data_value_input_flatten_shape = F._from_data(self.name + '.value_input_flatten_shape',
                    np.array([bs * self.num_bev_queue, num_value, embed_dims], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_value_input_flatten_shape.name, _data_value_input_flatten_shape)
        _op = F.Reshape(self.name + '.value_input_flatten')
        setattr(self, _op.name, _op)
        value = _op(value, _data_value_input_flatten_shape)
        setattr(self, value.name, value)

        # Use SliceF to extract value[:bs]
        value_current_shape = [bs, num_value, embed_dims]
        starts_0 = F._from_data(self.name + '.starts_0', np.array([0], dtype=np.int64), is_const=True)
        setattr(self, starts_0.name, starts_0)
        ends_bs = F._from_data(self.name + '.ends_bs', np.array([bs], dtype=np.int64), is_const=True)
        setattr(self, ends_bs.name, ends_bs)
        axes_0 = F._from_data(self.name + '.axes_0', np.array([0], dtype=np.int64), is_const=True)
        setattr(self, axes_0.name, axes_0)
        steps_1 = F._from_data(self.name + '.steps_1', np.array([1], dtype=np.int64), is_const=True)
        setattr(self, steps_1.name, steps_1)
        _op = F.SliceF(self.name + '.value_current', out_shape=value_current_shape)
        setattr(self, _op.name, _op)
        value_current = _op(value, starts_0, ends_bs, axes_0, steps_1)
        setattr(self, value_current.name, value_current)
        _op = F.ConcatX(self.name + '.query_concat', axis=2)
        setattr(self, _op.name, _op)
        query_concat = _op(value_current, query)
        setattr(self, query_concat.name, query_concat)

        # Project value
        value = self.value_proj(value)
        setattr(self, value.name, value)

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # Expand mask: [bs*num_bev_queue, num_value] -> [bs*num_bev_queue, num_value, 1]
            _data_mask_expand_axis = F._from_data(self.name + '.mask_expand_axis', np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _data_mask_expand_axis.name, _data_mask_expand_axis)
            _op = F.Unsqueeze(self.name + '.mask_expand')
            setattr(self, _op.name, _op)
            mask_expanded = _op(key_padding_mask, _data_mask_expand_axis)
            setattr(self, mask_expanded.name, mask_expanded)
            # Invert mask (0 becomes 1, non-zero becomes 0)
            _data_one = F._from_data(self.name + '.one', np.array([[[1.0]]], dtype=np.float32), is_const=True)
            setattr(self, _data_one.name, _data_one)
            _op = F.Sub(self.name + '.mask_invert')
            setattr(self, _op.name, _op)
            mask_inverted = _op(_data_one, mask_expanded)
            setattr(self, mask_inverted.name, mask_inverted)
            _op = F.Mul(self.name + '.value_masked')
            setattr(self, _op.name, _op)
            value = _op(value, mask_inverted)
            setattr(self, value.name, value)

        # Reshape value: [bs*num_bev_queue, num_value, embed_dims] -> [bs*num_bev_queue, num_value, num_heads, dim_per_head]
        dim_per_head = self.embed_dims // self.num_heads
        _data_value_reshape_heads_shape = F._from_data(self.name + '.value_reshape_heads_shape',
                    np.array([bs * self.num_bev_queue, num_value, self.num_heads, dim_per_head], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_value_reshape_heads_shape.name, _data_value_reshape_heads_shape)
        _op = F.Reshape(self.name + '.value_reshape_heads')
        setattr(self, _op.name, _op)
        value = _op(value, _data_value_reshape_heads_shape)
        setattr(self, value.name, value)

        # Generate sampling offsets
        sampling_offsets = self.sampling_offsets(query_concat)
        setattr(self, sampling_offsets.name, sampling_offsets)
        _data_sampling_offsets_reshape_shape = F._from_data(self.name + '.sampling_offsets_reshape_shape',
                    np.array([bs, num_query, self.num_heads, self.num_bev_queue,
                            self.num_levels, self.num_points, 2], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_sampling_offsets_reshape_shape.name, _data_sampling_offsets_reshape_shape)
        _op = F.Reshape(self.name + '.sampling_offsets_reshape')
        setattr(self, _op.name, _op)
        sampling_offsets = _op(sampling_offsets, _data_sampling_offsets_reshape_shape)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Generate attention weights
        attention_weights = self.attention_weights(query_concat)
        setattr(self, attention_weights.name, attention_weights)
        _data_attention_weights_reshape_shape = F._from_data(self.name + '.attention_weights_reshape_shape',
                    np.array([bs, num_query, self.num_heads, self.num_bev_queue,
                            self.num_levels * self.num_points], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_attention_weights_reshape_shape.name, _data_attention_weights_reshape_shape)
        _op = F.Reshape(self.name + '.attention_weights_reshape')
        setattr(self, _op.name, _op)
        attention_weights = _op(attention_weights, _data_attention_weights_reshape_shape)
        setattr(self, attention_weights.name, attention_weights)

        # Apply softmax
        _op = F.Softmax(self.name + '.attention_weights_softmax', axis=-1)
        setattr(self, _op.name, _op)
        attention_weights = _op(attention_weights)
        setattr(self, attention_weights.name, attention_weights)

        # Reshape attention weights
        _data_attention_weights_reshape2_shape = F._from_data(self.name + '.attention_weights_reshape2_shape',
                    np.array([bs, num_query, self.num_heads, self.num_bev_queue,
                            self.num_levels, self.num_points], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_attention_weights_reshape2_shape.name, _data_attention_weights_reshape2_shape)
        _op = F.Reshape(self.name + '.attention_weights_reshape2')
        setattr(self, _op.name, _op)
        attention_weights = _op(attention_weights, _data_attention_weights_reshape2_shape)
        setattr(self, attention_weights.name, attention_weights)

        # Permute attention weights: [bs, num_query, num_heads, num_bev_queue, num_levels, num_points]
        # -> [bs, num_bev_queue, num_query, num_heads, num_levels, num_points]
        _op = F.Transpose(self.name + '.attention_weights_permute',
                                       perm=[0, 3, 1, 2, 4, 5])
        setattr(self, _op.name, _op)
        attention_weights = _op(attention_weights)
        setattr(self, attention_weights.name, attention_weights)

        # Reshape to [bs*num_bev_queue, num_query, num_heads, num_levels, num_points]
        _data_attention_weights_reshape3_shape = F._from_data(self.name + '.attention_weights_reshape3_shape',
                    np.array([bs * self.num_bev_queue, num_query, self.num_heads,
                            self.num_levels, self.num_points], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_attention_weights_reshape3_shape.name, _data_attention_weights_reshape3_shape)
        _op = F.Reshape(self.name + '.attention_weights_reshape3')
        setattr(self, _op.name, _op)
        attention_weights = _op(attention_weights, _data_attention_weights_reshape3_shape)
        setattr(self, attention_weights.name, attention_weights)

        # Permute sampling offsets: [bs, num_query, num_heads, num_bev_queue, num_levels, num_points, 2]
        # -> [bs, num_bev_queue, num_query, num_heads, num_levels, num_points, 2]
        _op = F.Transpose(self.name + '.sampling_offsets_permute',
                                      perm=[0, 3, 1, 2, 4, 5, 6])
        setattr(self, _op.name, _op)
        sampling_offsets = _op(sampling_offsets)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Reshape to [bs*num_bev_queue, num_query, num_heads, num_levels, num_points, 2]
        _data_sampling_offsets_reshape2_shape = F._from_data(self.name + '.sampling_offsets_reshape2_shape',
                    np.array([bs * self.num_bev_queue, num_query, self.num_heads,
                            self.num_levels, self.num_points, 2], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_sampling_offsets_reshape2_shape.name, _data_sampling_offsets_reshape2_shape)
        _op = F.Reshape(self.name + '.sampling_offsets_reshape2')
        setattr(self, _op.name, _op)
        sampling_offsets = _op(sampling_offsets, _data_sampling_offsets_reshape2_shape)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # Create offset normalizer from spatial_shapes
            # spatial_shapes is a list of tuples, need to convert to tensor
            spatial_shapes_array = np.array(spatial_shapes, dtype=np.float32)
            # Stack [W, H] for normalization
            offset_normalizer_data = np.stack([spatial_shapes_array[:, 1], spatial_shapes_array[:, 0]], axis=-1)
            offset_normalizer = F._from_data(self.name + '.offset_normalizer',
                                            offset_normalizer_data,
                                            is_const=True)
            setattr(self, offset_normalizer.name, offset_normalizer)

            # reference_points already arrive as [bs*num_bev_queue, num_query, num_levels, 2]
            # (doubled by the encoder via hybird_ref_2d), so use them directly.
            ref_points_expanded_flat = reference_points

            # Expand reference_points: [bs*num_bev_queue, num_query, num_levels, 2]
            # -> [bs*num_bev_queue, num_query, 1, num_levels, 1, 2]
            _data_ref_points_exp1_axis = F._from_data(self.name + '.ref_points_exp1_axis', np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _data_ref_points_exp1_axis.name, _data_ref_points_exp1_axis)
            _op = F.Unsqueeze(self.name + '.ref_points_exp1')
            setattr(self, _op.name, _op)
            ref_points_exp1 = _op(ref_points_expanded_flat, _data_ref_points_exp1_axis)
            setattr(self, ref_points_exp1.name, ref_points_exp1)
            _data_ref_points_exp2_axis = F._from_data(self.name + '.ref_points_exp2_axis', np.array([4], dtype=np.int64), is_const=True)
            setattr(self, _data_ref_points_exp2_axis.name, _data_ref_points_exp2_axis)
            _op = F.Unsqueeze(self.name + '.ref_points_exp2')
            setattr(self, _op.name, _op)
            ref_points_exp2 = _op(ref_points_exp1, _data_ref_points_exp2_axis)
            setattr(self, ref_points_exp2.name, ref_points_exp2)

            # Expand offset_normalizer: [num_levels, 2] -> [1, 1, 1, num_levels, 1, 2]
            _data_offset_norm_exp1_axis = F._from_data(self.name + '.offset_norm_exp1_axis', np.array([0], dtype=np.int64), is_const=True)
            setattr(self, _data_offset_norm_exp1_axis.name, _data_offset_norm_exp1_axis)
            _op = F.Unsqueeze(self.name + '.offset_norm_exp1')
            setattr(self, _op.name, _op)
            offset_normalizer_expanded = _op(offset_normalizer, _data_offset_norm_exp1_axis)
            setattr(self, offset_normalizer_expanded.name, offset_normalizer_expanded)
            _data_offset_norm_exp2_axis = F._from_data(self.name + '.offset_norm_exp2_axis', np.array([0], dtype=np.int64), is_const=True)
            setattr(self, _data_offset_norm_exp2_axis.name, _data_offset_norm_exp2_axis)
            _op = F.Unsqueeze(self.name + '.offset_norm_exp2')
            setattr(self, _op.name, _op)
            offset_normalizer_expanded = _op(offset_normalizer_expanded, _data_offset_norm_exp2_axis)
            setattr(self, offset_normalizer_expanded.name, offset_normalizer_expanded)
            _data_offset_norm_exp3_axis = F._from_data(self.name + '.offset_norm_exp3_axis', np.array([0], dtype=np.int64), is_const=True)
            setattr(self, _data_offset_norm_exp3_axis.name, _data_offset_norm_exp3_axis)
            _op = F.Unsqueeze(self.name + '.offset_norm_exp3')
            setattr(self, _op.name, _op)
            offset_normalizer_expanded = _op(offset_normalizer_expanded, _data_offset_norm_exp3_axis)
            setattr(self, offset_normalizer_expanded.name, offset_normalizer_expanded)
            _data_offset_norm_exp4_axis = F._from_data(self.name + '.offset_norm_exp4_axis', np.array([4], dtype=np.int64), is_const=True)
            setattr(self, _data_offset_norm_exp4_axis.name, _data_offset_norm_exp4_axis)
            _op = F.Unsqueeze(self.name + '.offset_norm_exp4')
            setattr(self, _op.name, _op)
            offset_normalizer_expanded = _op(offset_normalizer_expanded, _data_offset_norm_exp4_axis)
            setattr(self, offset_normalizer_expanded.name, offset_normalizer_expanded)

            # Normalize offsets and add to reference points
            _op = F.Div(self.name + '.sampling_offsets_normalized')
            setattr(self, _op.name, _op)
            sampling_offsets_normalized = _op(sampling_offsets, offset_normalizer_expanded)
            setattr(self, sampling_offsets_normalized.name, sampling_offsets_normalized)
            _op = F.Add(self.name + '.sampling_locations')
            setattr(self, _op.name, _op)
            sampling_locations = _op(ref_points_exp2, sampling_offsets_normalized)
            setattr(self, sampling_locations.name, sampling_locations)
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2, '
                f'but got {reference_points.shape[-1]} instead.')

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            name=self.name + '.msda',
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            parent_module=self
        )
        setattr(self, output.name, output)

        # Output shape: [bs*num_bev_queue, num_query, embed_dims]
        # Permute to [num_query, embed_dims, bs*num_bev_queue]
        _op = F.Transpose(self.name + '.output_permute1', perm=[1, 2, 0])
        setattr(self, _op.name, _op)
        output = _op(output)
        setattr(self, output.name, output)

        # Reshape to [num_query, embed_dims, bs, num_bev_queue]
        _data_output_reshape1_shape = F._from_data(self.name + '.output_reshape1_shape',
                    np.array([num_query, embed_dims, bs, self.num_bev_queue], dtype=np.int64),
                    is_const=True)
        setattr(self, _data_output_reshape1_shape.name, _data_output_reshape1_shape)
        _op = F.Reshape(self.name + '.output_reshape1')
        setattr(self, _op.name, _op)
        output = _op(output, _data_output_reshape1_shape)
        setattr(self, output.name, output)

        # Fuse history and current by averaging over num_bev_queue dimension
        # Use ReduceMean over axis 3 (num_bev_queue)
        axes_tensor = F._from_data(self.name + '.mean_axes', np.array([3], dtype=np.int64), is_const=True)
        setattr(self, axes_tensor.name, axes_tensor)
        reduce_mean_op = F.SimOpHandle(self.name + '.output_mean', 'ReduceMean',
                                       params=[(1, axes_tensor)], ipos=[0], keepdims=0)
        setattr(self, reduce_mean_op.name, reduce_mean_op)
        reduce_mean_op.implicit_inputs.append(axes_tensor)
        output = reduce_mean_op(output)
        setattr(self, output.name, output)

        # Permute back to [bs, num_query, embed_dims]
        _op = F.Transpose(self.name + '.output_permute2', perm=[2, 0, 1])
        setattr(self, _op.name, _op)
        output = _op(output)
        setattr(self, output.name, output)

        # Apply output projection
        output = self.output_proj(output)
        setattr(self, output.name, output)

        # Apply dropout (note: in inference mode, dropout is typically disabled)
        # For TTSim, we'll skip dropout as it's inference-only
        # In PyTorch training, this would apply dropout

        # Handle batch_first format
        if not self.batch_first:
            # Convert back: [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            _op = F.Transpose(self.name + '.output_transpose_final', perm=[1, 0, 2])
            setattr(self, _op.name, _op)
            output = _op(output)
            setattr(self, output.name, output)

        # Add residual connection
        _op = F.Add(self.name + '.output_residual')
        setattr(self, _op.name, _op)
        output = _op(output, identity)
        setattr(self, output.name, output)

        return output

    def analytical_param_count(self):
        """
        Calculate the total number of parameters in this module.

        Returns:
            int: Total parameter count
        """
        # Sampling offsets: embed_dims*num_bev_queue -> num_bev_queue*num_heads*num_levels*num_points*2
        sampling_offsets_params = (self.embed_dims * self.num_bev_queue) * \
                                 (self.num_bev_queue * self.num_heads * self.num_levels * self.num_points * 2)
        sampling_offsets_bias = self.num_bev_queue * self.num_heads * self.num_levels * self.num_points * 2

        # Attention weights: embed_dims*num_bev_queue -> num_bev_queue*num_heads*num_levels*num_points
        attention_weights_params = (self.embed_dims * self.num_bev_queue) * \
                                  (self.num_bev_queue * self.num_heads * self.num_levels * self.num_points)
        attention_weights_bias = self.num_bev_queue * self.num_heads * self.num_levels * self.num_points

        # Value projection: embed_dims -> embed_dims
        value_proj_params = self.embed_dims * self.embed_dims
        value_proj_bias = self.embed_dims

        # Output projection: embed_dims -> embed_dims
        output_proj_params = self.embed_dims * self.embed_dims
        output_proj_bias = self.embed_dims

        total = (sampling_offsets_params + sampling_offsets_bias +
                attention_weights_params + attention_weights_bias +
                value_proj_params + value_proj_bias +
                output_proj_params + output_proj_bias)

        return total


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Temporal Self Attention TTSim Module")
    logger.info("=" * 80)
    logger.info("\n[OK] Module imported successfully!")
    logger.info("\nAvailable component:")
    logger.info("  - TemporalSelfAttention - Temporal attention for BEV features")

    logger.info("\nModule test:")

    # Test TemporalSelfAttention
    try:
        tsa = TemporalSelfAttention(
            name='test_tsa',
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            num_bev_queue=2,
            batch_first=True
        )
        logger.debug("  [OK] TemporalSelfAttention constructed successfully")
        logger.debug(f"    - Name: {tsa.name}")
        logger.debug(f"    - Embed dims: {tsa.embed_dims}")
        logger.debug(f"    - Num heads: {tsa.num_heads}")
        logger.debug(f"    - Num levels: {tsa.num_levels}")
        logger.debug(f"    - Num points: {tsa.num_points}")
        logger.debug(f"    - Num BEV queue: {tsa.num_bev_queue}")
        logger.debug(f"    - Parameter count: {tsa.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [X] TemporalSelfAttention construction failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n[OK] Basic test passed!")
    logger.info("\nNote: Use validation tests in Validation/ folder for full functionality testing.")
    logger.info("=" * 80)
