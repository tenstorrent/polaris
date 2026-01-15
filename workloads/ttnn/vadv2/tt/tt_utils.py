#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn

class DictAsAttr:
    def __init__(self, dictionary, depth=0):
        for key, value in dictionary.items():
            if isinstance(value, dict) and depth > 0:
                setattr(self, key, DictAsAttr(value, depth - 1))
            else:
                setattr(self, key, value)

def multi_scale_deformable_attn(value, value_spatial_shapes, sampling_locations, attention_weights, device, module=None):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = []
    value_list.append(value)
    sampling_locations = ttnn.to_layout(sampling_locations, layout=ttnn.ROW_MAJOR_LAYOUT)
    sampling_grids = sampling_locations # 2 * sampling_locations - 1
    sampling_value_list = []

    value_spatial_shapes = [value_spatial_shapes.data.tolist()]
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [value_l_.shape[0], value_l_.shape[1], value_l_.shape[2] * value_l_.shape[3]])
        value_l_ = ttnn.permute(value_l_, (0, 2, 1))
        value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, H_, W_]) # int(H_.item()), int(W_.item())])

        sampling_grids.set_module(module)
        sampling_grid_l_ = sampling_grids[:, :, :, level]
        sampling_grid_l_tensor_ = ttnn._rand(sampling_grid_l_.shape, dtype=ttnn.bfloat16, device=device)
        sampling_grid_l_ = ttnn.permute(sampling_grid_l_tensor_, (0, 2, 1, 3, 4))

        sampling_grid_l_ = ttnn.reshape(
            sampling_grid_l_,
            [
                sampling_grid_l_.shape[0] * sampling_grid_l_.shape[1],
                sampling_grid_l_.shape[2],
                sampling_grid_l_.shape[3],
                sampling_grid_l_.shape[4],
            ],
        )

        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        ttnn.deallocate(value_l_)
        ttnn.deallocate(sampling_grid_l_)
        
        sampling_value_list.append(sampling_value_l_)
        
        attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))

        attention_weights = ttnn.reshape(attention_weights, [bs * num_heads, 1, num_queries, num_levels * num_points])

    output = ttnn.stack(sampling_value_list, -2)
    ttnn.deallocate(sampling_grids)
    output = ttnn.reshape(
        output, [output.shape[0], output.shape[1], output.shape[2], output.shape[3] * output.shape[4]]
    )

    output = output * attention_weights
    output = ttnn.reallocate(output)
    for val in sampling_value_list:
        ttnn.deallocate(val)
    output = ttnn.sum(output, dim=3)
    output = ttnn.reallocate(output)
    output = ttnn.reshape(output, [bs, num_heads * embed_dims, num_queries])
    output = ttnn.reallocate(output)
    output = ttnn.permute(output, (0, 2, 1))

    ttnn.deallocate(attention_weights)
    ttnn.deallocate(sampling_value_l_)
    ttnn.deallocate(value)

    return output


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device)
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=x.device
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def bbox_xyxy_to_cxcywh(bbox):
    # bbox = ttnn.unsqueeze(bbox, 0) ## bbox is already in [1, 400] shape
    bbox = ttnn.to_layout(bbox, layout=ttnn.ROW_MAJOR_LAYOUT)
    bbox_shape = bbox.shape[-1]
    # ttnn.split(bbox, (1, 1, 1, 1), 2)
    x1 = ttnn.Tensor(shape=(1, bbox_shape//4), dtype=ttnn.bfloat16, device=bbox.device)
    y1 = ttnn.Tensor(shape=(1, bbox_shape//4), dtype=ttnn.bfloat16, device=bbox.device)
    x2 = ttnn.Tensor(shape=(1, bbox_shape//4), dtype=ttnn.bfloat16, device=bbox.device)
    y2 = ttnn.Tensor(shape=(1, bbox_shape//4), dtype=ttnn.bfloat16, device=bbox.device)
    # bbox_new = [ttnn.div((x1 + x2), const_2), ttnn.div((y1 + y2), const_2), (x2 - x1), (y2 - y1)]
    # return ttnn.concat(bbox_new, dim=-1)
    const_2 = ttnn.full(shape=(1, bbox_shape//4), dtype=ttnn.bfloat16, device=bbox.device, fill_value=2.0, layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.concat(ttnn.div((x1 + x2), const_2), ttnn.div((y1 + y2), const_2), (x2 - x1), (y2 - y1), axis=-1) # type: ignore[operator]


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = ttnn.split(bbox, (1, 1, 1, 1), dim=-1)

    bbox_new = [ttnn.multiply((cx - 0.5), w), ttnn.multiply((cy - 0.5), h), ttnn.multiply((cx + 0.5), w), ttnn.multiply((cy + 0.5), h)]
    return ttnn.concat(bbox_new, dim=-1)


def tt_denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)

    bboxes_reshaped = ttnn.reshape(bboxes, (bboxes.shape[0], 2, 2))
    bboxes_even = bboxes_reshaped[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes_odd = bboxes_reshaped[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]

    bboxes_combined = ttnn.stack([bboxes_even, bboxes_odd], dim=-1)
    bboxes = ttnn.reshape(bboxes_combined, [bboxes.shape[0], -1])

    return bboxes


def tt_denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts_slice1 = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts_slice2 = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts = ttnn.concat([new_pts_slice1, new_pts_slice2], dim=-1)
    return new_pts
