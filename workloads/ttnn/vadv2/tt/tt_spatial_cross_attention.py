#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr, multi_scale_deformable_attn

class TtSpatialCrossAttention(SimNN.Module):
    def __init__(
        self,
        device,
        params,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(TtSpatialCrossAttention, self).__init__()

        self.name = "TtSpatialCrossAttention"
        self.device = device
        self.params = params
        self.init_cfg = init_cfg
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = TtMSDeformableAttention3D(device=self.device, params=params, num_levels=1)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.batch_first = batch_first

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
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = ttnn.zeros_like(query)
            slots = ttnn.to_torch(slots)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape

        D = reference_points_cam.size(3)
        indexes = []
        bev_mask.set_module(self)
        for i in range(bev_mask.size(0)):
            mask_per_img = bev_mask[i, ...]
            mask_per_img_tensor = ttnn._rand(mask_per_img.shape, dtype=ttnn.bfloat16, device=self.device)
            index_query_per_img = ttnn.sum(mask_per_img_tensor, dim=-1)
            index_query_per_img = ttnn.to_layout(index_query_per_img, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(3):  # unsqueeze 3 times
                index_query_per_img = ttnn.unsqueeze(index_query_per_img, 0)
            output_tensor = ttnn.nonzero(index_query_per_img, queue_id=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(index_query_per_img)

            output_tensor.set_module(self)
            index_query_per_img = output_tensor

            for _ in range(3):  # squeeze back
                index_query_per_img = ttnn.squeeze(index_query_per_img, 0)
            indexes.append(index_query_per_img)

        query = ttnn.to_torch(query)
        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, indexes[0].shape[-1], self.embed_dims])
        reference_points_cam.set_module(self)
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, indexes[0].shape[-1], D, 2])

        queries_rebatch = ttnn.from_torch(queries_rebatch, dtype=ttnn.bfloat16, device=self.device)
        reference_points_rebatch = ttnn.from_torch(reference_points_rebatch, dtype=ttnn.bfloat16, device=self.device)
        num_cams, l, bs, embed_dims = key.shape
        num_cams, l, bs, embed_dims = key.shape

        key = ttnn.permute(key, (2, 0, 1, 3))
        key = ttnn.reshape(key, (bs * self.num_cams, l, self.embed_dims))

        value = ttnn.permute(value, (2, 0, 1, 3))
        value = ttnn.reshape(value, (bs * self.num_cams, l, self.embed_dims))
        queries = self.deformable_attention(
            query=ttnn.reshape(queries_rebatch, (bs * self.num_cams, indexes[0].shape[-1], self.embed_dims)),
            key=key,
            value=value,
            reference_points=ttnn.reshape(reference_points_rebatch, (bs * self.num_cams, indexes[0].shape[-1], D, 2)),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        ttnn.deallocate(queries_rebatch)
        ttnn.deallocate(reference_points_rebatch)

        queries = ttnn.reshape(queries, (bs, self.num_cams, indexes[0].shape[-1], self.embed_dims))

        queries = ttnn.to_torch(queries)
        count = ttnn.sum(bev_mask, dim=-1).nelems() # > 0 ## for now return all elements
        params = self.params
        if isinstance(params, dict):
            params_output_proj = DictAsAttr(params['output_proj'])
            slots = ttnn.linear(slots, params_output_proj.weight, bias=params_output_proj.bias) # type: ignore[attr-defined]
        else:
            slots = ttnn.linear(slots, self.params.output_proj.weight, bias=self.params.output_proj.bias)
            ttnn.deallocate(self.params.output_proj.weight)
            ttnn.deallocate(self.params.output_proj.bias)

        ttnn.deallocate(count)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        output = slots + inp_residual
        ttnn.deallocate(slots)
        ttnn.deallocate(inp_residual)

        return output


class TtMSDeformableAttention3D(SimNN.Module):
    def __init__(
        self,
        device,
        params,
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
        self.name = "TtMSDeformableAttention3D"
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False
        self.device = device
        self.params = params
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
            query = ttnn.permute(query, (1, 0, 2))
            value = ttnn.permute(value, (1, 0, 2))

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)

        if isinstance(params, dict):
            params_value_proj = DictAsAttr(params['value_proj'])
            value = ttnn.linear(value, params_value_proj.weight, bias=params_value_proj.bias) # type: ignore[attr-defined]
        else:
            value = ttnn.linear(value, params.value_proj.weight, bias=params.value_proj.bias)
            ttnn.deallocate(params.value_proj.weight)
            ttnn.deallocate(params.value_proj.bias)
        if key_padding_mask is not None:
            mask = key_padding_mask[..., None]
            value = ttnn.where(mask, ttnn.zeros_like(value), value)
        value = ttnn.reshape(value, (bs, num_value, self.num_heads, -1))
        query = ttnn.to_layout(query, ttnn.TILE_LAYOUT)

        if isinstance(params, dict):
            params_sampling_offsets = DictAsAttr(params['sampling_offsets'])
            sampling_offsets = ttnn.linear(query, params_sampling_offsets.weight, bias=params_sampling_offsets.bias) # type: ignore[attr-defined]
        else:
            sampling_offsets = ttnn.linear(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
            ttnn.deallocate(params.sampling_offsets.weight)
            ttnn.deallocate(params.sampling_offsets.bias)
        sampling_offsets = ttnn.reshape(
            sampling_offsets, (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        )

        if isinstance(params, dict):
            params_attention_weights = DictAsAttr(params['attention_weights'])
            attention_weights = ttnn.linear(query, params_attention_weights.weight, bias=params_attention_weights.bias) # type: ignore[attr-defined]
            ttnn.deallocate(params_attention_weights.weight) # type: ignore[attr-defined]
            ttnn.deallocate(params_attention_weights.bias) # type: ignore[attr-defined]
        else:
            attention_weights = ttnn.linear(query, params.attention_weights.weight, bias=params.attention_weights.bias)
            ttnn.deallocate(params.attention_weights.weight)
            ttnn.deallocate(params.attention_weights.bias)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels * self.num_points)
        )

        attention_weights = ttnn.softmax(attention_weights) #, -1) ## default is -1
        attention_weights = ttnn.reallocate(attention_weights)
        attention_weights = ttnn.reshape(
            attention_weights, (bs, num_query, self.num_heads, self.num_levels, self.num_points)
        )

        if reference_points.shape[-1] == 2:
            s1_shape = spatial_shapes.shape[1]
            s0_shape = spatial_shapes.shape[0]
            offset_normalizer = ttnn.Tensor(shape=[s0_shape, s1_shape], dtype=ttnn.bfloat16, device=self.device)
            bs_r, num_query, num_Z_anchors, _ = reference_points.shape
            reference_xy = ttnn.reshape(
                reference_points, (bs_r, num_query, 1, 1, 1, reference_points.shape[-2], reference_points.shape[-1])
            )
            offset_normalizer_xy = ttnn.reshape(
                offset_normalizer, (1, 1, 1, offset_normalizer.shape[0], 1, offset_normalizer.shape[1])
            )

            sampling_offsets = ttnn.to_layout(sampling_offsets, ttnn.TILE_LAYOUT)
            offset_normalizer_xy = ttnn.to_layout(offset_normalizer_xy, ttnn.TILE_LAYOUT)

            sampling_offsets_reshaped = ttnn.reshape(
                sampling_offsets, [sampling_offsets.shape[0], -1, sampling_offsets.shape[4], sampling_offsets.shape[5]]
            )
            offset_normalizer_xy_reshaped = ttnn.reshape(
                offset_normalizer_xy,
                [offset_normalizer_xy.shape[0], -1, offset_normalizer_xy.shape[4], offset_normalizer_xy.shape[5]],
            )

            sampling_locations = ttnn.div(sampling_offsets_reshaped, offset_normalizer_xy_reshaped)
            ttnn.deallocate(sampling_offsets_reshaped)
            ttnn.deallocate(offset_normalizer_xy_reshaped)
            ttnn.deallocate(offset_normalizer_xy)

            sampling_locations = ttnn.reshape(sampling_locations, sampling_offsets.shape)

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_locations.shape
            sampling_locations = ttnn.reshape(
                sampling_locations,
                [bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy],
            )
            reference_xy_reshaped = ttnn.reshape(
                reference_xy,
                (
                    reference_xy.shape[0],
                    reference_xy.shape[1],
                    -1,
                    reference_xy.shape[4],
                    reference_xy.shape[5],
                    reference_xy.shape[6],
                ),
            )
            sampling_locations_reshaped = ttnn.reshape(
                sampling_locations,
                (
                    sampling_locations.shape[0],
                    sampling_locations.shape[1],
                    -1,
                    sampling_locations.shape[4],
                    sampling_locations.shape[5],
                    sampling_locations.shape[6],
                ),
            )

            sampling_locations_add = reference_xy_reshaped + sampling_locations_reshaped

            sampling_locations = ttnn.reshape(sampling_locations_add, sampling_locations.shape)

            ttnn.deallocate(reference_xy_reshaped)
            ttnn.deallocate(sampling_locations_reshaped)
            ttnn.deallocate(sampling_locations_add)

            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            sampling_locations = ttnn.reshape(
                sampling_locations, (bs, num_query, num_heads, num_levels, num_all_points, xy)
            )

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        output = multi_scale_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights, self.device, module=self)
        ttnn.deallocate(value)
        ttnn.deallocate(sampling_locations)
        ttnn.deallocate(attention_weights)
        if not self.batch_first:
            output = ttnn.permute(output, (1, 0, 2))

        return output
