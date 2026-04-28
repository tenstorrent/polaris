#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of PtsCrossAttention for FusionAD.

This module implements standard multi-scale deformable cross-attention,
used in the decoder for cross-attending from object queries to BEV features.
It is simpler than TemporalSelfAttention (no temporal queue) and
SpatialCrossAttention (no camera projection).

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

1. Base Classes:
   - BaseModule -> SimNN.Module
2. Builders / Registry:
   - @ATTENTION.register_module() -> not needed in TTSim
   - deprecated_api_warning -> removed
3. Operations:
   - nn.Linear -> SimNN.Linear
   - nn.Dropout -> skipped (inference only)
   - torch.stack / tensor.view / .permute / .softmax -> TTSim ops
   - multi_scale_deformable_attn_pytorch -> multi_scale_deformable_attn_ttsim
   - masked_fill -> manual mask multiply
4. Initialization:
   - xavier_init / constant_init -> init_utils.py (PyTorch reference only)
"""

#-------------------------------PyTorch--------------------------------

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

# from mmcv.utils import ext_loader
# from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
#     MultiScaleDeformableAttnFunction_fp16

# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# @ATTENTION.register_module()
# class PtsCrossAttention(BaseModule):
#     """An attention module used in Deformable-Detr.

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

#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  num_levels=1,
#                  num_points=4,
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
#         self.dropout = nn.Dropout(dropout)
#         self.batch_first = batch_first
#         self.fp16_enabled = False

#         # you'd better set dim_per_head to a power of 2
#         # which is more efficient in the CUDA implementation
#         def _is_power_of_2(n):
#             if (not isinstance(n, int)) or (n < 0):
#                 raise ValueError(
#                     'invalid input for _is_power_of_2: {} (type: {})'.format(
#                         n, type(n)))
#             return (n & (n - 1) == 0) and n != 0

#         if not _is_power_of_2(dim_per_head):
#             warnings.warn(
#                 "You'd better set embed_dims in "
#                 'MultiScaleDeformAttention to make '
#                 'the dimension of each attention head a power of 2 '
#                 'which is more efficient in our CUDA implementation.')

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
#         self.output_proj = nn.Linear(embed_dims, embed_dims)
#         self.init_weights()

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

#         self.sampling_offsets.bias.data = grid_init.view(-1)
#         constant_init(self.attention_weights, val=0., bias=0.)
#         xavier_init(self.value_proj, distribution='uniform', bias=0.)
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
#         self._is_init = True

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
#                 **kwargs):
#         """Forward Function of MultiScaleDeformAttention.

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

#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """

#         if value is None:
#             value = query

#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
#         if not self.batch_first:
#             # change to (bs, num_query ,embed_dims)
#             query = query.permute(1, 0, 2)
#             value = value.permute(1, 0, 2)

#         bs, num_query, _ = query.shape
#         bs, num_value, _ = value.shape
#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)
#         value = value.view(bs, num_value, self.num_heads, -1)

#         sampling_offsets = self.sampling_offsets(query).view(
#             bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_levels * self.num_points)
#         attention_weights = attention_weights.softmax(-1)

#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_levels,
#                                                    self.num_points)
#         if reference_points.shape[-1] == 2:
#             offset_normalizer = torch.stack(
#                 [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_points[:, :, None, :, None, :] \
#                 + sampling_offsets \
#                 / offset_normalizer[None, None, None, :, None, :]
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

#             # using fp16 deformable attention is unstable because it performs many sum operations
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

#         output = self.output_proj(output)

#         if not self.batch_first:
#             # (num_query, bs ,embed_dims)
#             output = output.permute(1, 0, 2)

#         return self.dropout(output) + identity


#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger
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

# Import our TTSim deformable attention
from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_ttsim

# Import initialization utilities (Python 3.13 compatible)
from .init_utils import xavier_init, constant_init, _is_power_of_2


class PtsCrossAttention(SimNN.Module):
    """
    TTSim implementation of PtsCrossAttention (Multi-Scale Deformable Cross-Attention).

    Standard deformable cross-attention for attending from object queries to
    BEV features. Simpler than TemporalSelfAttention (no temporal queue) and
    SpatialCrossAttention (no camera projection).

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature map levels. Default: 1
        num_points (int): Number of sampling points per query per head. Default: 4
        im2col_step (int): Step for im2col (not used in CPU). Default: 64
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dim is first. Default: True

    Input Shapes:
        query:            [bs, num_query, embed_dims] (batch_first=True)
        value:            [bs, num_value, embed_dims] or None (defaults to query)
        reference_points: [bs, num_query, num_levels, 2] or [..., 4]
        spatial_shapes:   list of (H, W) tuples, one per level

    Output Shape:
        [bs, num_query, embed_dims] (batch_first=True)
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True):
        super().__init__()
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')

        dim_per_head = embed_dims // num_heads
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
        self.im2col_step = im2col_step
        self.dropout_rate = dropout

        # --- Learnable projections ---
        self.sampling_offsets = SimNN.Linear(
            name + '.sampling_offsets',
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points * 2
        )
        self.attention_weights = SimNN.Linear(
            name + '.attention_weights',
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points
        )
        self.value_proj = SimNN.Linear(
            name + '.value_proj',
            in_features=embed_dims,
            out_features=embed_dims
        )
        self.output_proj = SimNN.Linear(
            name + '.output_proj',
            in_features=embed_dims,
            out_features=embed_dims
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
                 **kwargs):
        """
        Forward pass of PtsCrossAttention.

        Args:
            query: [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
            value: same shape as query, or None (defaults to query)
            identity: residual tensor (defaults to query)
            query_pos: positional encoding added to query
            reference_points: [bs, num_query, num_levels, 2] or [..., 4]
            spatial_shapes: list of (H, W) per level
            key_padding_mask: [bs, num_value] boolean mask

        Returns:
            output + identity, same shape as query
        """

        # ---- Default value and identity ----
        if value is None:
            value = query
        if identity is None:
            identity = query

        # ---- Add positional encoding ----
        if query_pos is not None:
            _op = F.Add(self.name + '.query_add_pos')
            setattr(self, _op.name, _op)
            query = _op(query, query_pos)
            setattr(self, query.name, query)

        # ---- Handle non-batch-first format ----
        if not self.batch_first:
            _op_q = F.Transpose(self.name + '.query_to_bf', perm=[1, 0, 2])
            setattr(self, _op_q.name, _op_q)
            query = _op_q(query)
            setattr(self, query.name, query)
            _op_v = F.Transpose(self.name + '.value_to_bf', perm=[1, 0, 2])
            setattr(self, _op_v.name, _op_v)
            value = _op_v(value)
            setattr(self, value.name, value)
            # identity stays in original format; we'll transpose output back

        # ---- Dimensions ----
        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]
        dim_per_head = self.embed_dims // self.num_heads

        # ---- Value projection ----
        value = self.value_proj(value)
        setattr(self, value.name, value)

        # ---- Apply padding mask ----
        if key_padding_mask is not None:
            _ax = F._from_data(self.name + '.mask_ax', np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _ax.name, _ax)
            _unsq = F.Unsqueeze(self.name + '.mask_unsq')
            setattr(self, _unsq.name, _unsq)
            mask_3d = _unsq(key_padding_mask, _ax)
            setattr(self, mask_3d.name, mask_3d)
            _one = F._from_data(self.name + '.one', np.array([[[1.0]]], dtype=np.float32), is_const=True)
            setattr(self, _one.name, _one)
            _inv = F.Sub(self.name + '.mask_inv')
            setattr(self, _inv.name, _inv)
            mask_inv = _inv(_one, mask_3d)
            setattr(self, mask_inv.name, mask_inv)
            _mul = F.Mul(self.name + '.val_masked')
            setattr(self, _mul.name, _mul)
            value = _mul(value, mask_inv)
            setattr(self, value.name, value)

        # ---- Reshape value -> [bs, num_value, num_heads, dim_per_head] ----
        _shp = F._from_data(self.name + '.val_4d_shp',
                            np.array([bs, num_value, self.num_heads, dim_per_head], dtype=np.int64),
                            is_const=True)
        setattr(self, _shp.name, _shp)
        _resh = F.Reshape(self.name + '.val_4d')
        setattr(self, _resh.name, _resh)
        value = _resh(value, _shp)
        setattr(self, value.name, value)

        # ---- Sampling offsets ----
        sampling_offsets = self.sampling_offsets(query)
        setattr(self, sampling_offsets.name, sampling_offsets)
        _so_shp = F._from_data(self.name + '.so_shp',
                               np.array([bs, num_query, self.num_heads,
                                         self.num_levels, self.num_points, 2], dtype=np.int64),
                               is_const=True)
        setattr(self, _so_shp.name, _so_shp)
        _so_resh = F.Reshape(self.name + '.so_resh')
        setattr(self, _so_resh.name, _so_resh)
        sampling_offsets = _so_resh(sampling_offsets, _so_shp)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # ---- Attention weights -> softmax ----
        attn_w = self.attention_weights(query)
        setattr(self, attn_w.name, attn_w)
        _aw_shp = F._from_data(self.name + '.aw_shp',
                               np.array([bs, num_query, self.num_heads,
                                         self.num_levels * self.num_points], dtype=np.int64),
                               is_const=True)
        setattr(self, _aw_shp.name, _aw_shp)
        _aw_resh = F.Reshape(self.name + '.aw_resh')
        setattr(self, _aw_resh.name, _aw_resh)
        attn_w = _aw_resh(attn_w, _aw_shp)
        setattr(self, attn_w.name, attn_w)

        _sm = F.Softmax(self.name + '.aw_sm', axis=-1)
        setattr(self, _sm.name, _sm)
        attn_w = _sm(attn_w)
        setattr(self, attn_w.name, attn_w)

        # Reshape to [bs, nq, nh, nl, np]
        _aw5_shp = F._from_data(self.name + '.aw5_shp',
                                np.array([bs, num_query, self.num_heads,
                                          self.num_levels, self.num_points], dtype=np.int64),
                                is_const=True)
        setattr(self, _aw5_shp.name, _aw5_shp)
        _aw5_resh = F.Reshape(self.name + '.aw5_resh')
        setattr(self, _aw5_resh.name, _aw5_resh)
        attn_w = _aw5_resh(attn_w, _aw5_shp)
        setattr(self, attn_w.name, attn_w)

        # ---- Compute sampling locations ----
        if reference_points.shape[-1] == 2:
            # offset_normalizer = [W, H] per level (constant)
            ss_arr = np.array(spatial_shapes, dtype=np.float32)
            norm_data = np.stack([ss_arr[:, 1], ss_arr[:, 0]], axis=-1)  # [nl, 2]
            offset_norm = F._from_data(self.name + '.off_norm', norm_data, is_const=True)
            setattr(self, offset_norm.name, offset_norm)

            # Expand ref_points: [bs, nq, nl, 2] -> [bs, nq, 1, nl, 1, 2]
            _rp1_ax = F._from_data(self.name + '.rp1_ax', np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _rp1_ax.name, _rp1_ax)
            _rp1 = F.Unsqueeze(self.name + '.rp_unsq1')
            setattr(self, _rp1.name, _rp1)
            rp = _rp1(reference_points, _rp1_ax)
            setattr(self, rp.name, rp)
            _rp2_ax = F._from_data(self.name + '.rp2_ax', np.array([4], dtype=np.int64), is_const=True)
            setattr(self, _rp2_ax.name, _rp2_ax)
            _rp2 = F.Unsqueeze(self.name + '.rp_unsq2')
            setattr(self, _rp2.name, _rp2)
            rp = _rp2(rp, _rp2_ax)
            setattr(self, rp.name, rp)

            # Expand offset_normalizer: [nl, 2] -> [1, 1, 1, nl, 1, 2]
            for idx in range(4):
                _ax_t = F._from_data(f'{self.name}.on_ax{idx}', np.array([0], dtype=np.int64), is_const=True)
                setattr(self, _ax_t.name, _ax_t)
                _unsq_t = F.Unsqueeze(f'{self.name}.on_unsq{idx}')
                setattr(self, _unsq_t.name, _unsq_t)
                if idx < 3:
                    offset_norm = _unsq_t(offset_norm, _ax_t)
                else:
                    _ax_4 = F._from_data(f'{self.name}.on_ax_p', np.array([4], dtype=np.int64), is_const=True)
                    setattr(self, _ax_4.name, _ax_4)
                    offset_norm = _unsq_t(offset_norm, _ax_4)
                setattr(self, offset_norm.name, offset_norm)

            # sampling_locations = ref_points + offsets / normalizer
            _div = F.Div(self.name + '.so_div')
            setattr(self, _div.name, _div)
            so_normed = _div(sampling_offsets, offset_norm)
            setattr(self, so_normed.name, so_normed)
            _add = F.Add(self.name + '.samp_loc')
            setattr(self, _add.name, _add)
            sampling_locations = _add(rp, so_normed)
            setattr(self, sampling_locations.name, sampling_locations)

        elif reference_points.shape[-1] == 4:
            # reference_points [..., :2] + offsets / num_points * reference_points[..., 2:] * 0.5
            raise NotImplementedError(
                "reference_points with last dim == 4 is not yet implemented in TTSim. "
                "Add support when needed.")
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but got {reference_points.shape[-1]} instead.')

        # ---- Multi-scale deformable attention ----
        output = multi_scale_deformable_attn_ttsim(
            name=self.name + '.msda',
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attn_w,
            parent_module=self
        )
        setattr(self, output.name, output)

        # ---- Output projection ----
        output = self.output_proj(output)
        setattr(self, output.name, output)

        # ---- Handle non-batch-first ----
        if not self.batch_first:
            _op_out = F.Transpose(self.name + '.out_from_bf', perm=[1, 0, 2])
            setattr(self, _op_out.name, _op_out)
            output = _op_out(output)
            setattr(self, output.name, output)

        # ---- Residual connection (dropout skipped for inference) ----
        _res = F.Add(self.name + '.residual')
        setattr(self, _res.name, _res)
        output = _res(output, identity)
        setattr(self, output.name, output)

        return output

    # ------------------------------------------------------------------
    def analytical_param_count(self):
        """Total learnable parameter count."""
        # sampling_offsets:  embed_dims -> num_heads * num_levels * num_points * 2 (+bias)
        so_out = self.num_heads * self.num_levels * self.num_points * 2
        so = self.embed_dims * so_out + so_out

        # attention_weights: embed_dims -> num_heads * num_levels * num_points (+bias)
        aw_out = self.num_heads * self.num_levels * self.num_points
        aw = self.embed_dims * aw_out + aw_out

        # value_proj:  embed_dims -> embed_dims (+bias)
        vp = self.embed_dims * self.embed_dims + self.embed_dims

        # output_proj: embed_dims -> embed_dims (+bias)
        op = self.embed_dims * self.embed_dims + self.embed_dims

        return so + aw + vp + op


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("PtsCrossAttention TTSim Module")
    logger.info("=" * 70)

    try:
        pca = PtsCrossAttention(
            name='test_pca',
            embed_dims=256,
            num_heads=8,
            num_levels=1,
            num_points=4,
            batch_first=True
        )
        logger.debug("[OK] Constructed successfully")
        logger.debug(f"  embed_dims  = {pca.embed_dims}")
        logger.debug(f"  num_heads   = {pca.num_heads}")
        logger.debug(f"  num_levels  = {pca.num_levels}")
        logger.debug(f"  num_points  = {pca.num_points}")
        logger.debug(f"  params      = {pca.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X] Construction failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("=" * 70)
