#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np

class SpatialCrossAttention(SimNN.Module):
    """Spatial Cross-Attention module for BEVFormer (ttsim stub)"""

    def __init__(self, name, embed_dims=256, num_cams=6, pc_range=None, dropout=0.1,
                 num_points=8, num_levels=4):
        super(SpatialCrossAttention, self).__init__()
        self.name = name

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.dropout = F.Dropout(name + '.dropout', dropout)

        # Deformable attention path is intentionally stubbed out for ttsim

        # Output projection
        self.output_proj = F.Linear(name + '.output_proj', embed_dims, embed_dims)

        super().link_op2module()

    def __call__(self, query, key, value, reference_points=None, spatial_shapes=None, bev_pos=None):
        """
        Forward pass

        Args:
            query: BEV queries [bs, num_query, embed_dims]
            key: Multi-camera features [num_cams, bs, num_key, embed_dims]
            value: Multi-camera features [num_cams, bs, num_key, embed_dims]
            reference_points: Reference points for deformable attention
            spatial_shapes: Spatial shapes of feature maps
        """
        bs, num_query, embed_dims = query.shape

        # Build simple grid reference points if not provided
        if reference_points is None and bev_pos is not None:
            bev_h, bev_w = bev_pos.shape[2], bev_pos.shape[3]
            # [num_query, 2] normalized grid (y, x)
            yy, xx = np.meshgrid(np.arange(bev_h), np.arange(bev_w), indexing='ij')
            yy = (yy.astype(np.float32) / max(1, (bev_h - 1))).reshape([-1])
            xx = (xx.astype(np.float32) / max(1, (bev_w - 1))).reshape([-1])
            ref = np.stack([yy, xx], axis=-1)  # [num_query, 2]
            ref = F._from_data(self.name + '.reference_points_base', ref)
            ref.set_module(self)
            ref = ref.unsqueeze(0)  # [1, num_query, 2]
            # Expand to batch and levels: [bs, num_query, num_levels, 2]
            num_levels = len(spatial_shapes) if (spatial_shapes is not None and len(spatial_shapes) > 0) else 1
            repeats_ref = F._from_data(self.name + '.ref_repeats', np.array([bs, 1, 1], dtype=np.int64), is_const=True)
            repeats_ref.set_module(self)
            tile_ref_batch = F.Tile(self.name + '.tile_ref_batch')
            tile_ref_batch.set_module(self)
            ref_batched = tile_ref_batch(ref, repeats_ref)
            ref_batched = ref_batched.unsqueeze(2)
            repeats_lvl = F._from_data(self.name + '.ref_lvl_repeats', np.array([1, 1, num_levels, 1], dtype=np.int64), is_const=True)
            repeats_lvl.set_module(self)
            tile_ref_levels = F.Tile(self.name + '.tile_ref_levels')
            tile_ref_levels.set_module(self)
            reference_points = tile_ref_levels(ref_batched, repeats_lvl)

        # Stable stub path for ttsim: bypass deformable attention
        output = self.dropout(query)
        output = self.output_proj(output)
        return output
