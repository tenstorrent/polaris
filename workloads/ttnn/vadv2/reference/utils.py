#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn

def bbox_cxcywh_to_xyxy(bbox):
    out = ttnn.Tensor(shape=[bbox.shape[0], bbox.shape[1]//4], dtype=ttnn.float32, device=bbox.device)
    cx, cy = ttnn.split(bbox, out, num_splits=4, dim=-1)
    cx , w = ttnn.split(cx, out, num_splits=4, dim=-1)
    cy , h = ttnn.split(cy, out, num_splits=4, dim=-1)
    const_0p5 = ttnn.full(shape=out.shape, dtype=ttnn.float32, device=bbox.device, fill_value=0.5, layout=ttnn.Layout.TILE_LAYOUT)
    return ttnn.concat((cx - const_0p5 * w), (cy - const_0p5 * h), (cx + const_0p5 * w), (cy + const_0p5 * h), axis=-1)

def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    return bboxes

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    return new_pts

def denormalize_bbox(normalized_bboxes, pc_range):
    nrm_bboxes_shape = normalized_bboxes.shape
    rot_sine = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    rot_cosine = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    cx = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    cy = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    cz = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    w = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    l = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
    h = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)

    rot = ttnn.atan(ttnn.div(rot_sine, rot_cosine))

    w = ttnn.exp(w)
    l = ttnn.exp(l)
    h = ttnn.exp(h)
    if normalized_bboxes.size(-1) > 8:
        vx = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
        vy = ttnn.Tensor(shape=[nrm_bboxes_shape[0]], dtype=ttnn.float32, device=normalized_bboxes.device)
        denormalized_bboxes = ttnn.concat(cx, cy, cz, w, l, h, rot, vx, vy, axis=-1)
    else:
        denormalized_bboxes = ttnn.concat(cx, cy, cz, w, l, h, rot, axis=-1)
    return denormalized_bboxes
