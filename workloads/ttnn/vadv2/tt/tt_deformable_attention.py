#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
import warnings
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr, multi_scale_deformable_attn


class TtCustomMSDeformableAttention(SimNN.Module):
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        im2col_step=192,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.name = "TtCustomMSDeformableAttention"
        self.params = params
        self.device = device
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

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
        flag="decoder",
        **kwargs,
    ):
        params = self.params
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = ttnn.permute(query, [1, 0, 2])
            value = ttnn.permute(value, [1, 0, 2])

        bs, num_query, _ = query.shape
        num_value, _, _ = value.shape
        # assert (ttnn.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1])) == num_value
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        params_value_proj = DictAsAttr(params['value_proj'])
        value = ttnn.linear(value, params_value_proj.weight, bias=params_value_proj.bias) # type: ignore[attr-defined]
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))

        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)
        params_sampling_offsets = DictAsAttr(params['sampling_offsets'])
        sampling_offsets = ttnn.linear(query, params_sampling_offsets.weight, bias=params_sampling_offsets.bias) # type: ignore[attr-defined]
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )
        params_attention_weights = DictAsAttr(params['attention_weights'])
        attention_weights = ttnn.linear(query, params_attention_weights.weight, bias=params_attention_weights.bias) # type: ignore[attr-defined]
        ttnn.deallocate(params_attention_weights.weight) # type: ignore[attr-defined]
        ttnn.deallocate(params_attention_weights.bias) # type: ignore[attr-defined]
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )
        attention_weights = ttnn.softmax(attention_weights, dim=-1)

        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            spatial_shapes.set_module(self)
            s1 = spatial_shapes[..., 1]
            s0 = spatial_shapes[..., 0]
            s1 = ttnn.Tensor(shape=s1.shape, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, data=s1.data)
            s0 = ttnn.Tensor(shape=s0.shape, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, data=s0.data)
            offset_normalizer = ttnn.stack([s1, s0], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_xy = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 0)
            sampling_offsets = ttnn.squeeze(sampling_offsets, 2)
            offset_normalizer_xy = ttnn.squeeze(offset_normalizer_xy, 0)
            offset_normalizer_xy = ttnn.squeeze(offset_normalizer_xy, 0)

            sampling_locations = ttnn.divide(sampling_offsets, offset_normalizer_xy, use_legacy=False)

            sampling_locations = ttnn.unsqueeze(sampling_locations, 2)
            sampling_locations = ttnn.unsqueeze(sampling_locations, 0)
            sampling_locations = ttnn.add(reference_xy, sampling_locations, use_legacy=False)

        elif reference_points.shape[-1] == 4:
            reference_points_reshape = ttnn.reshape(
                reference_points,
                [reference_points.shape[0], reference_points.shape[1], 1, reference_points.shape[2], 1, 2],
            )
            sampling_locations = (
                reference_points_reshape + sampling_offsets / self.num_points * reference_points_reshape * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device, self)

        params_output_proj = DictAsAttr(params['output_proj'])
        output = ttnn.linear(output, params_output_proj.weight, bias=params_output_proj.bias) # type: ignore[attr-defined]
        ttnn.deallocate(params_output_proj.weight) # type: ignore[attr-defined]
        ttnn.deallocate(params_output_proj.bias) # type: ignore[attr-defined]
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)

        return output
