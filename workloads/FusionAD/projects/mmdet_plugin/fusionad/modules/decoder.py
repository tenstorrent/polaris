
#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of the FusionAD Detection Transformer Decoder.

Contains two classes:
  - CustomMSDeformableAttention : Standard multi-scale deformable cross-attention
                                  (used by decoder layers for query-to-BEV attention)
  - DetectionTransformerDecoder : Sequence of decoder layers with iterative
                                  reference-point refinement via regression branches

Also provides:
  - inverse_sigmoid : Utility used during reference-point refinement

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

1. Base Classes:
   - TransformerLayerSequence -> SimNN.Module with layer list
   - BaseModule -> SimNN.Module
2. Registry / Decorators:
   - @TRANSFORMER_LAYER_SEQUENCE.register_module() -> removed
   - @ATTENTION.register_module() -> removed
   - @deprecated_api_warning -> removed
   - @force_fp32 / @auto_fp16 -> removed (inference only)
3. Builders:
   - build_attention -> builder_utils.build_attention
   - build_feedforward_network -> builder_utils.build_feedforward_network
   - build_norm_layer -> builder_utils.build_norm_layer
4. Operations:
   - nn.Linear -> SimNN.Linear
   - nn.Dropout -> skipped (inference only)
   - torch.stack / .view / .permute / .softmax -> TTSim F ops
   - multi_scale_deformable_attn_pytorch -> multi_scale_deformable_attn_ttsim
   - masked_fill -> manual mask multiply
   - torch.log -> F.Log
   - torch.sigmoid -> F.Sigmoid
   - torch.clamp -> F.Clip
5. Decoder-specific:
   - reg_branches[lid](output) -> reg_branches[lid](output)  (SimNN.Module list)
   - .detach() -> no-op in inference (no gradient tracking)
   - torch.zeros_like -> F._from_data with np.zeros
   - torch.stack(intermediate) -> numpy stack + F._from_data
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
#
# from mmcv.utils import ext_loader
# from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
#     MultiScaleDeformableAttnFunction_fp16
#
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# def inverse_sigmoid(x, eps=1e-5):
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1 / x2)


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class DetectionTransformerDecoder(TransformerLayerSequence):
#     def __init__(self, *args, return_intermediate=False, **kwargs):
#         super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
#         self.return_intermediate = return_intermediate
#         self.fp16_enabled = False
#
#     def forward(self, query, *args, reference_points=None,
#                 reg_branches=None, key_padding_mask=None, **kwargs):
#         output = query
#         intermediate = []
#         intermediate_reference_points = []
#         for lid, layer in enumerate(self.layers):
#             reference_points_input = reference_points[..., :2].unsqueeze(2)
#             output = layer(output, *args,
#                            reference_points=reference_points_input,
#                            key_padding_mask=key_padding_mask, **kwargs)
#             output = output.permute(1, 0, 2)
#
#             if reg_branches is not None:
#                 tmp = reg_branches[lid](output)
#                 assert reference_points.shape[-1] == 3
#                 new_reference_points = torch.zeros_like(reference_points)
#                 new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
#                 new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
#                 new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()
#
#             output = output.permute(1, 0, 2)
#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)
#
#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(intermediate_reference_points)
#         return output, reference_points


# @ATTENTION.register_module()
# class CustomMSDeformableAttention(BaseModule):
#     def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
#                  im2col_step=64, dropout=0.1, batch_first=False, norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         ...
#     def forward(self, query, key=None, value=None, identity=None,
#                 query_pos=None, key_padding_mask=None, reference_points=None,
#                 spatial_shapes=None, level_start_index=None, flag='decoder', **kwargs):
#         ...
#         return self.dropout(output) + identity

#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger
import copy
import math
import warnings

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..','..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import multi-scale deformable attention
from .multi_scale_deformable_attn_function import multi_scale_deformable_attn_ttsim

# Import initialization utilities
from .init_utils import xavier_init, constant_init, _is_power_of_2

# Import builder utilities and custom base transformer layer
from .builder_utils import (build_attention, build_feedforward_network,
                            build_norm_layer, InverseSigmoid,
                            inverse_sigmoid_np)
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer


# ======================================================================
# CustomMSDeformableAttention
# ======================================================================

class CustomMSDeformableAttention(SimNN.Module):
    """
    TTSim implementation of CustomMSDeformableAttention.

    Standard multi-scale deformable cross-attention used in the decoder
    for attending from object queries to BEV features. Functionally
    identical to PtsCrossAttention but with batch_first=False by default.

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of feature map levels. Default: 4.
        num_points (int): Number of sampling points per head. Default: 4.
        im2col_step (int): Step for im2col (unused in CPU). Default: 64.
        dropout (float): Dropout rate. Default: 0.1.
        batch_first (bool): If True, batch dim first. Default: False.

    Input Shapes:
        query:            [num_query, bs, embed_dims] (batch_first=False)
                          or [bs, num_query, embed_dims] (batch_first=True)
        value:            same as query, or None (defaults to query)
        reference_points: [bs, num_query, num_levels, 2] or [..., 4]
        spatial_shapes:   list of (H, W) tuples, one per level

    Output Shape:
        Same shape as query input.
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False):
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

        # Learnable projections
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
        Forward pass.

        Args:
            query: Query tensor.
            value: Value tensor (defaults to query if None).
            identity: Residual tensor (defaults to query if None).
            query_pos: Positional encoding added to query.
            reference_points: [bs, num_query, num_levels, 2] or [..., 4].
            spatial_shapes: list of (H, W) per level.
            key_padding_mask: [bs, num_value] boolean mask.

        Returns:
            output + identity (residual), same shape as query.
        """
        # Defaults
        if value is None:
            value = query
        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            _op = F.Add(self.name + '.query_add_pos')
            setattr(self, _op.name, _op)
            query = _op(query, query_pos)
            setattr(self, query.name, query)

        # Handle non-batch-first: transpose to [bs, seq, dim]
        if not self.batch_first:
            _op_q = F.Transpose(self.name + '.query_to_bf', perm=[1, 0, 2])
            setattr(self, _op_q.name, _op_q)
            query = _op_q(query)
            setattr(self, query.name, query)
            _op_v = F.Transpose(self.name + '.value_to_bf', perm=[1, 0, 2])
            setattr(self, _op_v.name, _op_v)
            value = _op_v(value)
            setattr(self, value.name, value)

        # Dimensions
        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]
        dim_per_head = self.embed_dims // self.num_heads

        # Value projection
        value = self.value_proj(value)
        setattr(self, value.name, value)

        # Apply padding mask
        if key_padding_mask is not None:
            _ax = F._from_data(self.name + '.mask_ax',
                               np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _ax.name, _ax)
            _unsq = F.Unsqueeze(self.name + '.mask_unsq')
            setattr(self, _unsq.name, _unsq)
            mask_3d = _unsq(key_padding_mask, _ax)
            setattr(self, mask_3d.name, mask_3d)
            _one = F._from_data(self.name + '.one',
                                np.array([[[1.0]]], dtype=np.float32),
                                is_const=True)
            setattr(self, _one.name, _one)
            _inv = F.Sub(self.name + '.mask_inv')
            setattr(self, _inv.name, _inv)
            mask_inv = _inv(_one, mask_3d)
            setattr(self, mask_inv.name, mask_inv)
            _mul = F.Mul(self.name + '.val_masked')
            setattr(self, _mul.name, _mul)
            value = _mul(value, mask_inv)
            setattr(self, value.name, value)

        # Reshape value -> [bs, num_value, num_heads, dim_per_head]
        _shp = F._from_data(self.name + '.val_4d_shp',
                            np.array([bs, num_value, self.num_heads,
                                      dim_per_head], dtype=np.int64),
                            is_const=True)
        setattr(self, _shp.name, _shp)
        _resh = F.Reshape(self.name + '.val_4d')
        setattr(self, _resh.name, _resh)
        value = _resh(value, _shp)
        setattr(self, value.name, value)

        # Sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        setattr(self, sampling_offsets.name, sampling_offsets)
        _so_shp = F._from_data(
            self.name + '.so_shp',
            np.array([bs, num_query, self.num_heads,
                      self.num_levels, self.num_points, 2], dtype=np.int64),
            is_const=True)
        setattr(self, _so_shp.name, _so_shp)
        _so_resh = F.Reshape(self.name + '.so_resh')
        setattr(self, _so_resh.name, _so_resh)
        sampling_offsets = _so_resh(sampling_offsets, _so_shp)
        setattr(self, sampling_offsets.name, sampling_offsets)

        # Attention weights -> softmax
        attn_w = self.attention_weights(query)
        setattr(self, attn_w.name, attn_w)
        _aw_shp = F._from_data(
            self.name + '.aw_shp',
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
        _aw5_shp = F._from_data(
            self.name + '.aw5_shp',
            np.array([bs, num_query, self.num_heads,
                      self.num_levels, self.num_points], dtype=np.int64),
            is_const=True)
        setattr(self, _aw5_shp.name, _aw5_shp)
        _aw5_resh = F.Reshape(self.name + '.aw5_resh')
        setattr(self, _aw5_resh.name, _aw5_resh)
        attn_w = _aw5_resh(attn_w, _aw5_shp)
        setattr(self, attn_w.name, attn_w)

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # offset_normalizer = [W, H] per level (constant)
            ss_arr = np.array(spatial_shapes, dtype=np.float32)
            norm_data = np.stack([ss_arr[:, 1], ss_arr[:, 0]], axis=-1)
            offset_norm = F._from_data(self.name + '.off_norm',
                                       norm_data, is_const=True)
            setattr(self, offset_norm.name, offset_norm)

            # Expand ref_points: [bs, nq, nl, 2] -> [bs, nq, 1, nl, 1, 2]
            _rp1_ax = F._from_data(self.name + '.rp1_ax',
                                   np.array([2], dtype=np.int64),
                                   is_const=True)
            setattr(self, _rp1_ax.name, _rp1_ax)
            _rp1 = F.Unsqueeze(self.name + '.rp_unsq1')
            setattr(self, _rp1.name, _rp1)
            rp = _rp1(reference_points, _rp1_ax)
            setattr(self, rp.name, rp)
            _rp2_ax = F._from_data(self.name + '.rp2_ax',
                                   np.array([4], dtype=np.int64),
                                   is_const=True)
            setattr(self, _rp2_ax.name, _rp2_ax)
            _rp2 = F.Unsqueeze(self.name + '.rp_unsq2')
            setattr(self, _rp2.name, _rp2)
            rp = _rp2(rp, _rp2_ax)
            setattr(self, rp.name, rp)

            # Expand normalizer: [nl, 2] -> [1, 1, 1, nl, 1, 2]
            for idx in range(4):
                _ax_t = F._from_data(f'{self.name}.on_ax{idx}',
                                     np.array([0], dtype=np.int64),
                                     is_const=True)
                setattr(self, _ax_t.name, _ax_t)
                _unsq_t = F.Unsqueeze(f'{self.name}.on_unsq{idx}')
                setattr(self, _unsq_t.name, _unsq_t)
                if idx < 3:
                    offset_norm = _unsq_t(offset_norm, _ax_t)
                else:
                    _ax_4 = F._from_data(f'{self.name}.on_ax_p',
                                         np.array([4], dtype=np.int64),
                                         is_const=True)
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
            # reference_points: [bs, nq, nl, 4] = [cx, cy, w, h]
            # sampling_locations = center + offsets / num_points * wh * 0.5

            # Slice center [bs, nq, nl, :2] and wh [bs, nq, nl, 2:]
            bs4 = reference_points.shape[0]
            nq4 = reference_points.shape[1]
            nl4 = reference_points.shape[2]

            _rp4_ax = F._from_data(self.name + '.rp4_ax',
                                   np.array([-1], dtype=np.int64), is_const=True)
            setattr(self, _rp4_ax.name, _rp4_ax)
            _rp4_stp = F._from_data(self.name + '.rp4_stp',
                                    np.array([1], dtype=np.int64), is_const=True)
            setattr(self, _rp4_stp.name, _rp4_stp)

            _rp4_st0 = F._from_data(self.name + '.rp4_st0',
                                    np.array([0], dtype=np.int64), is_const=True)
            setattr(self, _rp4_st0.name, _rp4_st0)
            _rp4_en2 = F._from_data(self.name + '.rp4_en2',
                                    np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _rp4_en2.name, _rp4_en2)
            _sl_center = F.SliceF(self.name + '.rp4_sl_center',
                                  out_shape=[bs4, nq4, nl4, 2])
            setattr(self, _sl_center.name, _sl_center)
            rp_center = _sl_center(reference_points, _rp4_st0, _rp4_en2,
                                   _rp4_ax, _rp4_stp)
            setattr(self, rp_center.name, rp_center)

            _rp4_st2 = F._from_data(self.name + '.rp4_st2',
                                    np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _rp4_st2.name, _rp4_st2)
            _rp4_en4 = F._from_data(self.name + '.rp4_en4',
                                    np.array([4], dtype=np.int64), is_const=True)
            setattr(self, _rp4_en4.name, _rp4_en4)
            _sl_wh = F.SliceF(self.name + '.rp4_sl_wh',
                              out_shape=[bs4, nq4, nl4, 2])
            setattr(self, _sl_wh.name, _sl_wh)
            rp_wh = _sl_wh(reference_points, _rp4_st2, _rp4_en4,
                           _rp4_ax, _rp4_stp)
            setattr(self, rp_wh.name, rp_wh)

            # Unsqueeze center & wh: [bs,nq,nl,2] → [bs,nq,1,nl,1,2]
            _unsq_ax2 = F._from_data(self.name + '.rp4_unsq2',
                                     np.array([2], dtype=np.int64), is_const=True)
            setattr(self, _unsq_ax2.name, _unsq_ax2)
            _unsq_ax4 = F._from_data(self.name + '.rp4_unsq4',
                                     np.array([4], dtype=np.int64), is_const=True)
            setattr(self, _unsq_ax4.name, _unsq_ax4)

            _uc1 = F.Unsqueeze(self.name + '.rp4_c_unsq1')
            setattr(self, _uc1.name, _uc1)
            rp_center = _uc1(rp_center, _unsq_ax2)
            setattr(self, rp_center.name, rp_center)
            _uc2 = F.Unsqueeze(self.name + '.rp4_c_unsq2')
            setattr(self, _uc2.name, _uc2)
            rp_center = _uc2(rp_center, _unsq_ax4)
            setattr(self, rp_center.name, rp_center)

            _uw1 = F.Unsqueeze(self.name + '.rp4_wh_unsq1')
            setattr(self, _uw1.name, _uw1)
            rp_wh = _uw1(rp_wh, _unsq_ax2)
            setattr(self, rp_wh.name, rp_wh)
            _uw2 = F.Unsqueeze(self.name + '.rp4_wh_unsq2')
            setattr(self, _uw2.name, _uw2)
            rp_wh = _uw2(rp_wh, _unsq_ax4)
            setattr(self, rp_wh.name, rp_wh)

            # offsets / num_points * wh * 0.5
            _np_c = F._from_data(self.name + '.np_const',
                                 np.array([float(self.num_points)],
                                          dtype=np.float32), is_const=True)
            setattr(self, _np_c.name, _np_c)
            _half_c = F._from_data(self.name + '.half_const',
                                   np.array([0.5], dtype=np.float32),
                                   is_const=True)
            setattr(self, _half_c.name, _half_c)

            _div4 = F.Div(self.name + '.so_div4')
            setattr(self, _div4.name, _div4)
            so_div = _div4(sampling_offsets, _np_c)
            setattr(self, so_div.name, so_div)

            _mul_wh = F.Mul(self.name + '.so_mul_wh')
            setattr(self, _mul_wh.name, _mul_wh)
            so_wh = _mul_wh(so_div, rp_wh)
            setattr(self, so_wh.name, so_wh)

            _mul_half = F.Mul(self.name + '.so_mul_half')
            setattr(self, _mul_half.name, _mul_half)
            so_scaled = _mul_half(so_wh, _half_c)
            setattr(self, so_scaled.name, so_scaled)

            _add4 = F.Add(self.name + '.samp_loc_4d')
            setattr(self, _add4.name, _add4)
            sampling_locations = _add4(rp_center, so_scaled)
            setattr(self, sampling_locations.name, sampling_locations)

        else:
            raise ValueError(
                f'Last dim of reference_points must be 2 or 4, '
                f'but got {reference_points.shape[-1]} instead.')

        # Multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            name=self.name + '.msda',
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attn_w,
            parent_module=self
        )
        setattr(self, output.name, output)

        # Output projection
        output = self.output_proj(output)
        setattr(self, output.name, output)

        # Handle non-batch-first output
        if not self.batch_first:
            _op_out = F.Transpose(self.name + '.out_from_bf', perm=[1, 0, 2])
            setattr(self, _op_out.name, _op_out)
            output = _op_out(output)
            setattr(self, output.name, output)

        # Residual connection (dropout skipped for inference)
        _res = F.Add(self.name + '.residual')
        setattr(self, _res.name, _res)
        output = _res(output, identity)
        setattr(self, output.name, output)

        return output

    def analytical_param_count(self):
        """Total learnable parameter count."""
        so_out = self.num_heads * self.num_levels * self.num_points * 2
        so = self.embed_dims * so_out + so_out

        aw_out = self.num_heads * self.num_levels * self.num_points
        aw = self.embed_dims * aw_out + aw_out

        vp = self.embed_dims * self.embed_dims + self.embed_dims
        op = self.embed_dims * self.embed_dims + self.embed_dims

        return so + aw + vp + op


# ======================================================================
# DetectionTransformerDecoder
# ======================================================================

class DetectionTransformerDecoder(SimNN.Module):
    """
    TTSim implementation of DetectionTransformerDecoder.

    Sequence of transformer decoder layers (MyCustomBaseTransformerLayer).
    Supports iterative reference-point refinement via reg_branches
    (box-refinement heads applied after each layer).

    In the FusionAD config:
      - 6 decoder layers
      - Each layer: self_attn (MultiheadAttention) -> norm ->
                    cross_attn (CustomMSDeformableAttention) -> norm ->
                    ffn -> norm
      - return_intermediate=True  (collect outputs from every layer)

    Args:
        name (str): Module name.
        num_layers (int): Number of decoder layers.
        layer_cfg (dict): Config dict for each decoder layer.
            Must contain 'attn_cfgs', 'feedforward_channels', etc.
        return_intermediate (bool): Whether to return outputs from all
            layers (for auxiliary losses). Default: False.

    Forward Args:
        query: [num_query, bs, embed_dims] (sequence-first format).
        key: BEV features [num_key, bs, embed_dims].
        value: Same as key or None.
        reference_points: [bs, num_query, 3] (x, y, z in [0,1]).
        reg_branches: list of SimNN.Module, one per layer, for
            refining reference points. Each takes [bs, nq, embed_dims]
            and outputs regression deltas.
        key_padding_mask: [bs, num_key] boolean mask.
        **kwargs: Passed through to each layer.

    Returns:
        If return_intermediate:
            (stacked_outputs, stacked_reference_points)
            shapes: [num_layers, num_query, bs, embed_dims],
                    [num_layers, bs, num_query, 3]
        Else:
            (output, reference_points)
    """

    def __init__(self,
                 name,
                 num_layers,
                 layer_cfg,
                 return_intermediate=False):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # Build decoder layers
        _layers = []
        cfg = copy.deepcopy(layer_cfg)
        cfg.pop('type', None)  # Remove 'type' key if present

        # Decoder operates in sequence-first format [nq, bs, embed_dims],
        # matching PyTorch BaseTransformerLayer default (batch_first=False).
        # Force batch_first=False so attention modules transpose correctly.
        cfg.setdefault('batch_first', False)

        for i in range(num_layers):
            _layers.append(
                MyCustomBaseTransformerLayer(
                    name=f'{name}.layer_{i}',
                    **copy.deepcopy(cfg)
                )
            )
        self.layers = SimNN.ModuleList(_layers)

        # Inverse sigmoid modules for reference point refinement
        # (one per layer to avoid ONNX name collisions)
        self._inv_sigmoid_xy = []
        self._inv_sigmoid_z = []
        for i in range(num_layers):
            inv_xy = InverseSigmoid(f'{name}.inv_sigmoid_xy_{i}')
            inv_z = InverseSigmoid(f'{name}.inv_sigmoid_z_{i}')
            setattr(self, f'_inv_sigmoid_xy_{i}', inv_xy)
            setattr(self, f'_inv_sigmoid_z_{i}', inv_z)
            self._inv_sigmoid_xy.append(inv_xy)
            self._inv_sigmoid_z.append(inv_z)

    def __call__(self,
                 query,
                 key=None,
                 value=None,
                 query_pos=None,
                 key_pos=None,
                 reference_points=None,
                 reg_branches=None,
                 key_padding_mask=None,
                 **kwargs):
        """
        Forward pass of the decoder.

        Note: query is expected in [num_query, bs, embed_dims] format
        (sequence-first, matching PyTorch BEVFormer convention).
        The layer internally handles the permutation.

        Args:
            query: [num_query, bs, embed_dims]
            key: [num_key, bs, embed_dims] - BEV features
            value: same as key, or None
            query_pos: [num_query, bs, embed_dims] or [bs, nq, embed_dims]
            key_pos: positional encoding for key
            reference_points: [bs, num_query, 3] - (x, y, z) in [0, 1]
            reg_branches: list of modules for ref-point refinement
            key_padding_mask: [bs, num_key] boolean

        Returns:
            (output, reference_points) or
            (stacked_outputs, stacked_reference_points)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Slice reference_points[..., :2] and add num_levels dim
            # reference_points: [bs, nq, 3] -> [bs, nq, 1, 2]
            ref_pts_np = reference_points.data if hasattr(reference_points, 'data') else reference_points
            if isinstance(ref_pts_np, np.ndarray):
                rp_input_np = ref_pts_np[..., :2][:, :, np.newaxis, :]
                reference_points_input = F._from_data(
                    f'{self.name}.ref_pts_input_{lid}',
                    rp_input_np.astype(np.float32),
                    is_const=True
                )
                setattr(self, reference_points_input.name,
                        reference_points_input)
            else:
                # If it's already a SimTensor, use slice ops
                _sl_ax = F._from_data(f'{self.name}.rp_sl_ax_{lid}',
                                      np.array([-1], dtype=np.int64),
                                      is_const=True)
                setattr(self, _sl_ax.name, _sl_ax)
                _sl_st = F._from_data(f'{self.name}.rp_sl_st_{lid}',
                                      np.array([0], dtype=np.int64),
                                      is_const=True)
                setattr(self, _sl_st.name, _sl_st)
                _sl_en = F._from_data(f'{self.name}.rp_sl_en_{lid}',
                                      np.array([2], dtype=np.int64),
                                      is_const=True)
                setattr(self, _sl_en.name, _sl_en)
                _sl_stp = F._from_data(f'{self.name}.rp_sl_stp_{lid}',
                                       np.array([1], dtype=np.int64),
                                       is_const=True)
                setattr(self, _sl_stp.name, _sl_stp)
                bs_rp = reference_points.shape[0]
                nq_rp = reference_points.shape[1]
                _slice = F.SliceF(f'{self.name}.rp_slice_{lid}',
                                  out_shape=[bs_rp, nq_rp, 2])
                setattr(self, _slice.name, _slice)
                rp_2d = _slice(reference_points, _sl_st, _sl_en,
                               _sl_ax, _sl_stp)
                setattr(self, rp_2d.name, rp_2d)

                _unsq_ax = F._from_data(f'{self.name}.rp_unsq_ax_{lid}',
                                        np.array([2], dtype=np.int64),
                                        is_const=True)
                setattr(self, _unsq_ax.name, _unsq_ax)
                _unsq = F.Unsqueeze(f'{self.name}.rp_unsq_{lid}')
                setattr(self, _unsq.name, _unsq)
                reference_points_input = _unsq(rp_2d, _unsq_ax)
                setattr(self, reference_points_input.name,
                        reference_points_input)

            # Run decoder layer
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs
            )

            # --- Reference point refinement ---
            if reg_branches is not None:
                # output is [num_query, bs, embed_dims] after layer
                # Need [bs, num_query, embed_dims] for reg_branches
                _perm = F.Transpose(f'{self.name}.out_perm_{lid}',
                                    perm=[1, 0, 2])
                setattr(self, _perm.name, _perm)
                output_bf = _perm(output)
                setattr(self, output_bf.name, output_bf)

                # Apply regression branch
                tmp = reg_branches[lid](output_bf)
                setattr(self, tmp.name, tmp)

                # Refinement logic (using numpy for the const ref points):
                # new_ref[..., :2] = tmp[..., :2] + inv_sig(ref[..., :2])
                # new_ref[..., 2:3] = tmp[..., 4:5] + inv_sig(ref[..., 2:3])
                # new_ref = sigmoid(new_ref)
                #
                # For purely-numpy reference_points (first iteration or
                # const ref_points), we compute in numpy:
                if isinstance(ref_pts_np, np.ndarray):
                    # This branch handles the common case where ref points
                    # come from the init embedding (numpy).
                    # After refinement they become SimTensors.
                    inv_xy = inverse_sigmoid_np(ref_pts_np[..., :2])
                    inv_z = inverse_sigmoid_np(ref_pts_np[..., 2:3])

                    # We need to add tmp[..., :2] and tmp[..., 4:5]
                    # to the inverse-sigmoid values
                    inv_xy_t = F._from_data(
                        f'{self.name}.inv_xy_{lid}',
                        inv_xy.astype(np.float32), is_const=True)
                    setattr(self, inv_xy_t.name, inv_xy_t)
                    inv_z_t = F._from_data(
                        f'{self.name}.inv_z_{lid}',
                        inv_z.astype(np.float32), is_const=True)
                    setattr(self, inv_z_t.name, inv_z_t)

                    # Slice tmp[..., :2]
                    bs_t = tmp.shape[0]
                    nq_t = tmp.shape[1]
                    out_dim = tmp.shape[2]
                    _tmp_sl = F.SliceF(
                        f'{self.name}.tmp_sl_xy_{lid}',
                        out_shape=[bs_t, nq_t, 2])
                    setattr(self, _tmp_sl.name, _tmp_sl)
                    _tmp_ax = F._from_data(
                        f'{self.name}.tmp_ax_{lid}',
                        np.array([-1], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_ax.name, _tmp_ax)
                    _tmp_st = F._from_data(
                        f'{self.name}.tmp_st_xy_{lid}',
                        np.array([0], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_st.name, _tmp_st)
                    _tmp_en = F._from_data(
                        f'{self.name}.tmp_en_xy_{lid}',
                        np.array([2], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_en.name, _tmp_en)
                    _tmp_stp = F._from_data(
                        f'{self.name}.tmp_stp_{lid}',
                        np.array([1], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_stp.name, _tmp_stp)
                    tmp_xy = _tmp_sl(tmp, _tmp_st, _tmp_en, _tmp_ax, _tmp_stp)
                    setattr(self, tmp_xy.name, tmp_xy)

                    # Slice tmp[..., 4:5]
                    _tmp_sl_z = F.SliceF(
                        f'{self.name}.tmp_sl_z_{lid}',
                        out_shape=[bs_t, nq_t, 1])
                    setattr(self, _tmp_sl_z.name, _tmp_sl_z)
                    _tmp_st_z = F._from_data(
                        f'{self.name}.tmp_st_z_{lid}',
                        np.array([4], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_st_z.name, _tmp_st_z)
                    _tmp_en_z = F._from_data(
                        f'{self.name}.tmp_en_z_{lid}',
                        np.array([5], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_en_z.name, _tmp_en_z)
                    tmp_z = _tmp_sl_z(tmp, _tmp_st_z, _tmp_en_z,
                                      _tmp_ax, _tmp_stp)
                    setattr(self, tmp_z.name, tmp_z)

                    # new_xy = sigmoid(tmp_xy + inv_sigmoid(ref_xy))
                    _add_xy = F.Add(f'{self.name}.add_xy_{lid}')
                    setattr(self, _add_xy.name, _add_xy)
                    new_xy_pre = _add_xy(tmp_xy, inv_xy_t)
                    setattr(self, new_xy_pre.name, new_xy_pre)
                    _sig_xy = F.Sigmoid(f'{self.name}.sig_xy_{lid}')
                    setattr(self, _sig_xy.name, _sig_xy)
                    new_xy = _sig_xy(new_xy_pre)
                    setattr(self, new_xy.name, new_xy)

                    # new_z = sigmoid(tmp_z + inv_sigmoid(ref_z))
                    _add_z = F.Add(f'{self.name}.add_z_{lid}')
                    setattr(self, _add_z.name, _add_z)
                    new_z_pre = _add_z(tmp_z, inv_z_t)
                    setattr(self, new_z_pre.name, new_z_pre)
                    _sig_z = F.Sigmoid(f'{self.name}.sig_z_{lid}')
                    setattr(self, _sig_z.name, _sig_z)
                    new_z = _sig_z(new_z_pre)
                    setattr(self, new_z.name, new_z)

                    # Concat [new_xy, new_z] -> [bs, nq, 3]
                    _cat = F.ConcatX(f'{self.name}.cat_ref_{lid}', axis=-1)
                    setattr(self, _cat.name, _cat)
                    reference_points = _cat(new_xy, new_z)
                    setattr(self, reference_points.name, reference_points)

                else:
                    # SimTensor ref points (from a previous layer's refinement)
                    # Use InverseSigmoid ops
                    # Slice ref[..., :2]
                    bs_r = reference_points.shape[0]
                    nq_r = reference_points.shape[1]
                    _rsl = F.SliceF(f'{self.name}.ref_sl_xy_{lid}',
                                    out_shape=[bs_r, nq_r, 2])
                    setattr(self, _rsl.name, _rsl)
                    _rsl_ax = F._from_data(f'{self.name}.ref_ax_{lid}',
                                           np.array([-1], dtype=np.int64),
                                           is_const=True)
                    setattr(self, _rsl_ax.name, _rsl_ax)
                    _rsl_st = F._from_data(f'{self.name}.ref_st_{lid}',
                                           np.array([0], dtype=np.int64),
                                           is_const=True)
                    setattr(self, _rsl_st.name, _rsl_st)
                    _rsl_en = F._from_data(f'{self.name}.ref_en_{lid}',
                                           np.array([2], dtype=np.int64),
                                           is_const=True)
                    setattr(self, _rsl_en.name, _rsl_en)
                    _rsl_stp = F._from_data(f'{self.name}.ref_stp_{lid}',
                                            np.array([1], dtype=np.int64),
                                            is_const=True)
                    setattr(self, _rsl_stp.name, _rsl_stp)
                    ref_xy = _rsl(reference_points, _rsl_st, _rsl_en,
                                  _rsl_ax, _rsl_stp)
                    setattr(self, ref_xy.name, ref_xy)

                    # Slice ref[..., 2:3]
                    _rsl_z = F.SliceF(f'{self.name}.ref_sl_z_{lid}',
                                      out_shape=[bs_r, nq_r, 1])
                    setattr(self, _rsl_z.name, _rsl_z)
                    _rsl_st_z = F._from_data(f'{self.name}.ref_st_z_{lid}',
                                             np.array([2], dtype=np.int64),
                                             is_const=True)
                    setattr(self, _rsl_st_z.name, _rsl_st_z)
                    _rsl_en_z = F._from_data(f'{self.name}.ref_en_z_{lid}',
                                             np.array([3], dtype=np.int64),
                                             is_const=True)
                    setattr(self, _rsl_en_z.name, _rsl_en_z)
                    ref_z = _rsl_z(reference_points, _rsl_st_z, _rsl_en_z,
                                   _rsl_ax, _rsl_stp)
                    setattr(self, ref_z.name, ref_z)

                    # inv_sigmoid on ref xy and z
                    inv_xy = self._inv_sigmoid_xy[lid](ref_xy)
                    setattr(self, inv_xy.name, inv_xy)
                    inv_z = self._inv_sigmoid_z[lid](ref_z)
                    setattr(self, inv_z.name, inv_z)

                    # Slice tmp[..., :2] and tmp[..., 4:5]
                    bs_t = tmp.shape[0]
                    nq_t = tmp.shape[1]
                    _tmp_sl2 = F.SliceF(f'{self.name}.tmp_sl2_xy_{lid}',
                                        out_shape=[bs_t, nq_t, 2])
                    setattr(self, _tmp_sl2.name, _tmp_sl2)
                    _tmp_ax2 = F._from_data(f'{self.name}.tmp_ax2_{lid}',
                                            np.array([-1], dtype=np.int64),
                                            is_const=True)
                    setattr(self, _tmp_ax2.name, _tmp_ax2)
                    _tmp_st2 = F._from_data(f'{self.name}.tmp_st2_xy_{lid}',
                                            np.array([0], dtype=np.int64),
                                            is_const=True)
                    setattr(self, _tmp_st2.name, _tmp_st2)
                    _tmp_en2 = F._from_data(f'{self.name}.tmp_en2_xy_{lid}',
                                            np.array([2], dtype=np.int64),
                                            is_const=True)
                    setattr(self, _tmp_en2.name, _tmp_en2)
                    _tmp_stp2 = F._from_data(f'{self.name}.tmp_stp2_{lid}',
                                             np.array([1], dtype=np.int64),
                                             is_const=True)
                    setattr(self, _tmp_stp2.name, _tmp_stp2)
                    tmp_xy = _tmp_sl2(tmp, _tmp_st2, _tmp_en2,
                                      _tmp_ax2, _tmp_stp2)
                    setattr(self, tmp_xy.name, tmp_xy)

                    _tmp_sl2_z = F.SliceF(f'{self.name}.tmp_sl2_z_{lid}',
                                          out_shape=[bs_t, nq_t, 1])
                    setattr(self, _tmp_sl2_z.name, _tmp_sl2_z)
                    _tmp_st2_z = F._from_data(
                        f'{self.name}.tmp_st2_z_{lid}',
                        np.array([4], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_st2_z.name, _tmp_st2_z)
                    _tmp_en2_z = F._from_data(
                        f'{self.name}.tmp_en2_z_{lid}',
                        np.array([5], dtype=np.int64), is_const=True)
                    setattr(self, _tmp_en2_z.name, _tmp_en2_z)
                    tmp_z = _tmp_sl2_z(tmp, _tmp_st2_z, _tmp_en2_z,
                                       _tmp_ax2, _tmp_stp2)
                    setattr(self, tmp_z.name, tmp_z)

                    # new_xy = sigmoid(tmp_xy + inv_sig_xy)
                    _add_xy = F.Add(f'{self.name}.add2_xy_{lid}')
                    setattr(self, _add_xy.name, _add_xy)
                    new_xy_pre = _add_xy(tmp_xy, inv_xy)
                    setattr(self, new_xy_pre.name, new_xy_pre)
                    _sig_xy = F.Sigmoid(f'{self.name}.sig2_xy_{lid}')
                    setattr(self, _sig_xy.name, _sig_xy)
                    new_xy = _sig_xy(new_xy_pre)
                    setattr(self, new_xy.name, new_xy)

                    # new_z = sigmoid(tmp_z + inv_sig_z)
                    _add_z = F.Add(f'{self.name}.add2_z_{lid}')
                    setattr(self, _add_z.name, _add_z)
                    new_z_pre = _add_z(tmp_z, inv_z)
                    setattr(self, new_z_pre.name, new_z_pre)
                    _sig_z = F.Sigmoid(f'{self.name}.sig2_z_{lid}')
                    setattr(self, _sig_z.name, _sig_z)
                    new_z = _sig_z(new_z_pre)
                    setattr(self, new_z.name, new_z)

                    # Concat -> [bs, nq, 3]
                    _cat = F.ConcatX(f'{self.name}.cat2_ref_{lid}', axis=-1)
                    setattr(self, _cat.name, _cat)
                    reference_points = _cat(new_xy, new_z)
                    setattr(self, reference_points.name, reference_points)

                # Permute output back to [num_query, bs, embed_dims]
                _perm_back = F.Transpose(
                    f'{self.name}.out_perm_back_{lid}', perm=[1, 0, 2])
                setattr(self, _perm_back.name, _perm_back)
                output = _perm_back(output_bf)
                setattr(self, output.name, output)

            # Collect intermediate results
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return output, reference_points

    def analytical_param_count(self):
        """Total learnable parameter count across all decoder layers."""
        total = 0
        for layer in self.layers:
            if hasattr(layer, 'analytical_param_count'):
                total += layer.analytical_param_count()
        return total


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("FusionAD Decoder TTSim Module")
    logger.info("=" * 70)

    # Test InverseSigmoid
    logger.info("\n--- Test InverseSigmoid ---")
    try:
        inv_sig = InverseSigmoid('test_inv_sig')
        x_np = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        x_t = F._from_data('test_x', x_np, is_const=True)
        result = inv_sig(x_t)
        logger.debug("  [OK] InverseSigmoid constructed")
        logger.debug(f"  Input:  {x_np}")
        # Expected: log(x / (1-x))
        expected = np.log(x_np / (1 - x_np))
        logger.debug(f"  Expected: {expected}")
    except Exception as e:
        logger.debug(f"  [X] InverseSigmoid failed: {e}")
        import traceback
        traceback.print_exc()

    # Test CustomMSDeformableAttention
    logger.info("\n--- Test CustomMSDeformableAttention ---")
    try:
        attn = CustomMSDeformableAttention(
            name='test_cmda',
            embed_dims=256,
            num_heads=8,
            num_levels=1,
            num_points=4,
            batch_first=False
        )
        logger.debug("  [OK] CustomMSDeformableAttention constructed")
        logger.debug(f"    embed_dims  = {attn.embed_dims}")
        logger.debug(f"    num_heads   = {attn.num_heads}")
        logger.debug(f"    num_levels  = {attn.num_levels}")
        logger.debug(f"    num_points  = {attn.num_points}")
        logger.debug(f"    batch_first = {attn.batch_first}")
        logger.debug(f"    params      = {attn.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [X] CustomMSDeformableAttention failed: {e}")
        import traceback
        traceback.print_exc()

    # Test DetectionTransformerDecoder construction
    logger.info("\n--- Test DetectionTransformerDecoder ---")
    try:
        layer_cfg = dict(
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                ),
                dict(
                    type='CustomMSDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                ),
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
            ),
            operation_order=(
                'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'),
        )

        decoder = DetectionTransformerDecoder(
            name='test_decoder',
            num_layers=6,
            layer_cfg=layer_cfg,
            return_intermediate=True
        )
        logger.debug("  [OK] DetectionTransformerDecoder constructed")
        logger.debug(f"    num_layers          = {decoder.num_layers}")
        logger.debug(f"    return_intermediate = {decoder.return_intermediate}")
        logger.debug(f"    num layer modules   = {len(decoder.layers)}")
        logger.debug(f"    params              = {decoder.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [X] DetectionTransformerDecoder failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "=" * 70)
    logger.info("[OK] All decoder tests passed!")
    logger.info("=" * 70)
