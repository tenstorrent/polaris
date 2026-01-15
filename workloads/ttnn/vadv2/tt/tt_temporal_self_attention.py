#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
# import ttsim.front.ttnn as ttnnimport warnings
from workloads.ttnn.vadv2.tt.tt_utils import multi_scale_deformable_attn, DictAsAttr

class TtTemporalSelfAttention(SimNN.Module):
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims} and {num_heads}")

        self.name = "TtTemporalSelfAttention"
        self.device = device
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.params = params
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

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
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = ttnn.stack([query, query], dim=1)
            value = ttnn.reshape(value, (bs * 2, len_bev, c))

        if identity is None:
            identity = query
        if query_pos is not None:
            query = ttnn.add(query, query_pos)
        if not self.batch_first:
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert self.num_bev_queue == 2

        value.set_module(self)
        slice_value_shape = value[:bs, :, :].shape
        slice_value = ttnn._rand(slice_value_shape, dtype=ttnn.bfloat16, device=query.device)
        query = ttnn.concat(slice_value, query, axis=-1)

        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        if isinstance(params, dict):
            params_value_proj = DictAsAttr(params['value_proj'])
            value = ttnn.linear(value, params_value_proj.weight, bias=params_value_proj.bias) # type: ignore[attr-defined]
        else:
            value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias) # type: ignore[attr-defined]
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)

        value = ttnn.reshape(value, (bs * self.num_bev_queue, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        if isinstance(params, dict):
            params_sampling_offsets = DictAsAttr(params['sampling_offsets'])
            sampling_offsets = ttnn.linear(query, params_sampling_offsets.weight, bias=params_sampling_offsets.bias) # type: ignore[attr-defined]
        else:
            sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        )
        sampling_offsets = ttnn.reallocate(sampling_offsets)

        if isinstance(params, dict):
            params_attention_weights = DictAsAttr(params['attention_weights'])
            attention_weights = ttnn.linear(query, params_attention_weights.weight, bias=params_attention_weights.bias) # type: ignore[attr-defined]
            ttnn.deallocate(params_attention_weights.weight) # type: ignore[attr-defined]
            ttnn.deallocate(params_attention_weights.bias) # type: ignore[attr-defined]
        else:
            attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias) # type: ignore[attr-defined]
        ttnn.deallocate(query)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights)#, -1) ## axis=-1 is default
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        )

        attention_weights = ttnn.permute(attention_weights, (0, 3, 1, 2, 4, 5))
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        sampling_offsets = ttnn.permute(sampling_offsets, (0, 3, 1, 2, 4, 5, 6))
        sampling_offsets = ttnn.reallocate(sampling_offsets)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        if reference_points.shape[-1] == 2:
            spatial_shapes.set_module(self)
            slice_sp_s0 = spatial_shapes[..., 0].shape
            slice_sp_s1 = spatial_shapes[..., 1].shape
            sp_s1 = ttnn._rand(slice_sp_s1, dtype=ttnn.bfloat16, device=query.device)
            sp_s0 = ttnn._rand(slice_sp_s0, dtype=ttnn.bfloat16, device=query.device)
            offset_normalizer = ttnn.stack([sp_s0, sp_s1], dim=-1)
            bs_r, num_query, num_levels, _ = reference_points.shape
            reference_points_shape = reference_points.shape
            reference_points = ttnn.reshape(reference_points, (bs_r, num_query, 1, num_levels, 1, 2))
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )
            ttnn.deallocate(offset_normalizer)
            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)

            sampling_offsets_shape = sampling_offsets.shape
            sampling_offsets = ttnn.reshape(
                sampling_offsets, (sampling_offsets.shape[0], -1, sampling_offsets.shape[4], sampling_offsets.shape[5])
            )  # [2, 10000*8*1, 4, 2]
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer_xy,
                (
                    offset_normalizer_xy.shape[0],
                    offset_normalizer_xy.shape[1],
                    offset_normalizer_xy.shape[2],
                    offset_normalizer_xy.shape[-1],
                ),
            )
            sampling_locations = ttnn.div(sampling_offsets, offset_normalizer_xy)
            sampling_locations = ttnn.reshape(sampling_locations, sampling_offsets_shape)
            sampling_locations = reference_points + sampling_locations
            ttnn.deallocate(offset_normalizer_xy)
            reference_points = ttnn.reshape(reference_points, reference_points_shape)
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
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead."
            )
        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device, self)
        ttnn.deallocate(attention_weights)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(sampling_offsets)
        ttnn.deallocate(value)
        output = ttnn.permute(output, (1, 2, 0))
        output = ttnn.reshape(output, (num_query, embed_dims, bs, self.num_bev_queue))
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        output = ttnn.mean(output, dim=-1)
        output = ttnn.permute(output, (2, 0, 1))
        if isinstance(params, dict):
            params_output_proj = DictAsAttr(params['output_proj'])
            output = ttnn.linear(output, params_output_proj.weight, bias=params_output_proj.bias) # type: ignore[attr-defined]
        else:
            output = ttnn.linear(output, params.output_proj.weight, bias=params.output_proj.bias) # type: ignore[attr-defined]
            ttnn.deallocate(params.output_proj.weight) # type: ignore[attr-defined]
            ttnn.deallocate(params.output_proj.bias) # type: ignore[attr-defined]

        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        output = ttnn.add(output, identity)
        ttnn.deallocate(identity)
        return output
