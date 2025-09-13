#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import math
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

class TemporalSelfAttention(SimNN.Module):
    """Temporal Self-Attention for BEVFormer - handles temporal fusion of BEV features"""

    def __init__(self, name, embed_dims=256, num_heads=8, num_levels=1, num_points=4, num_bev_queue=2):
        super(TemporalSelfAttention, self).__init__()
        self.name = name

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        assert embed_dims % num_heads == 0, "embed_dims must be divisible by num_heads"

        self.head_dim = embed_dims // num_heads
        self.total_points = num_heads * num_levels * num_points

        # Linear projections for query, key, value
        self.query_proj = F.Linear(name + '.query_proj', embed_dims, embed_dims)
        self.key_proj = F.Linear(name + '.key_proj', embed_dims, embed_dims)
        self.value_proj = F.Linear(name + '.value_proj', embed_dims, embed_dims)

        # Deformable attention components
        self.sampling_offsets = F.Linear(name + '.sampling_offsets',
                                       embed_dims,
                                       num_heads * num_levels * num_points * 2)
        self.attention_weights = F.Linear(name + '.attention_weights',
                                        embed_dims,
                                        num_heads * num_levels * num_points)

        # Output projection
        self.output_proj = F.Linear(name + '.output_proj', embed_dims, embed_dims)
        self.dropout = F.Dropout(name + '.dropout', 0.1)

        # Initialize weights
        self._reset_parameters()

        super().link_op2module()

    def _reset_parameters(self):
        """Initialize parameters"""
        # Simplified initialization for Polaris compatibility
        # Initialize sampling offsets with grid pattern
        grid_init = self._get_sampling_grids()
        # Note: Polaris may handle initialization differently, so we'll skip explicit weight initialization

    def _get_sampling_grids(self):
        """Get sampling grids for temporal deformable attention"""
        grids = []
        for bev_idx in range(self.num_bev_queue):
            for level in range(self.num_levels):
                for head in range(self.num_heads):
                    for point in range(self.num_points):
                        # Create temporal sampling grid
                        # For temporal attention, we sample from different temporal positions
                        temporal_offset = (bev_idx - self.num_bev_queue // 2) * 2.0 ** level
                        spatial_offset = (point - self.num_points // 2) * 0.5

                        grids.append([temporal_offset, spatial_offset])

        return F._from_data('temporal_sampling_grid', np.array(grids, dtype=np.float32))

    def __call__(self, query, key=None, value=None, bev_pos=None, prev_bev=None, shift=None):
        """
        Forward pass of temporal self-attention

        Args:
            query: Current BEV features [bs, bev_h*bev_w, embed_dims]
            key: Previous BEV features (optional, uses query if None)
            value: Previous BEV features (optional, uses query if None)
            bev_pos: BEV positional embeddings
            prev_bev: Previous BEV features from queue
            shift: Ego-motion shift information
        """
        if key is None:
            key = query
        if value is None:
            value = query

        bs, num_query, embed_dims = query.shape

        # Concatenate current and previous BEV features for temporal attention
        if prev_bev is not None:
            # Stack current and previous BEV features
            bev_queue = F.ConcatX(self.name + '.bev_concat', axis=0)(prev_bev, query)
            bev_queue = bev_queue.reshape([self.num_bev_queue, bs, num_query, embed_dims])
        else:
            # Only current frame available
            bev_queue = query.unsqueeze(0)
            # Repeat along the new leading dimension using Tile
            repeats = F._from_data(self.name + '.repeat_vec', np.array([self.num_bev_queue, 1, 1, 1], dtype=np.int64), is_const=True)
            repeats.set_module(self)
            repeat_tile = F.Tile(self.name + '.repeat_query')
            repeat_tile.set_module(self)
            bev_queue = repeat_tile(bev_queue, repeats)

        # Flatten for linear projection
        bev_flat = bev_queue.reshape([bs * self.num_bev_queue, num_query, embed_dims])

        # Project to query, key, value
        query_proj = self.query_proj(bev_flat)
        key_proj = self.key_proj(bev_flat)
        value_proj = self.value_proj(bev_flat)

        # Reshape for multi-head attention, keep the temporal queue in batch for projections
        query_proj = query_proj.reshape([bs, self.num_bev_queue, num_query, self.num_heads, self.head_dim])
        key_proj = key_proj.reshape([bs, self.num_bev_queue, num_query, self.num_heads, self.head_dim])
        value_proj = value_proj.reshape([bs, self.num_bev_queue, num_query, self.num_heads, self.head_dim])

        # Permute for attention computation
        query_transpose = F.Transpose(self.name + '.query_transpose', perm=(0, 3, 1, 2, 4))
        key_transpose = F.Transpose(self.name + '.key_transpose', perm=(0, 3, 1, 2, 4))
        value_transpose = F.Transpose(self.name + '.value_transpose', perm=(0, 3, 1, 2, 4))
        query_transpose.set_module(self)
        key_transpose.set_module(self)
        value_transpose.set_module(self)
        query_proj = query_transpose(query_proj)  # [bs, num_heads, num_bev_queue, num_query, head_dim]
        key_proj = key_transpose(key_proj)
        value_proj = value_transpose(value_proj)

        # Get sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(bev_flat)
        attention_weights = self.attention_weights(bev_flat)

        # Reshape attention components
        sampling_offsets = sampling_offsets.reshape([bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2])
        attention_weights = attention_weights.reshape([bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels * self.num_points])
        attn_softmax = F.Softmax(self.name + '.attention_softmax', dim=-1)
        attn_softmax.set_module(self)
        attention_weights = attn_softmax(attention_weights)

        # Apply ego-motion shift if provided
        if shift is not None:
            add_shift = F.Add(self.name + '.add_shift')
            add_shift.set_module(self)
            sampling_offsets = add_shift(sampling_offsets, shift.unsqueeze(2).unsqueeze(3).unsqueeze(4))

        # Perform temporal deformable attention
        output = self._temporal_deformable_attention(
            query_proj, key_proj, value_proj, sampling_offsets, attention_weights
        )

        # Apply dropout and output projection
        output = self.dropout(output)
        output = self.output_proj(output)

        return output

    def _temporal_deformable_attention(self, query, key, value, sampling_offsets, attention_weights):
        """Perform temporal deformable attention"""
        bs, num_heads, num_bev_queue, num_query, head_dim = query.shape

        # Simplified temporal attention (would be more complex in full implementation)
        # This aggregates information across temporal frames

        # Compute attention scores
        # Merge heads for attention matmul
        key_t_op = F.Transpose(self.name + '.key_transpose_attn', perm=(0,1,3,2))
        key_t_op.set_module(self)
        key_t = key_t_op(key.reshape([bs * num_heads, num_bev_queue, num_query, head_dim]))
        matmul_scores = F.MatMul(self.name + '.temporal_attn_scores')
        matmul_scores.set_module(self)
        attn_scores = matmul_scores(query.reshape([bs * num_heads, num_bev_queue, num_query, head_dim]), key_t)

        # Scale attention scores
        scale = F._from_data(self.name + '.scale', np.float32(1.0 / math.sqrt(head_dim)), is_const=True)
        scale.set_module(self)
        scale_mul = F.Mul(self.name + '.scale_mul')
        scale_mul.set_module(self)
        attn_scores = scale_mul(attn_scores, scale)
        temp_softmax = F.Softmax(self.name + '.temporal_softmax', dim=-1)
        temp_softmax.set_module(self)
        attn_weights = temp_softmax(attn_scores)

        # Apply attention to values
        matmul_out = F.MatMul(self.name + '.temporal_attn_output')
        matmul_out.set_module(self)
        attn_output = matmul_out(attn_weights,
                              value.reshape([bs * num_heads, num_bev_queue, num_query, head_dim]))

        # Reshape back to original format
        attn_output = attn_output.reshape([bs, num_heads, num_bev_queue, num_query, head_dim])

        # Take the current frame output (last in temporal dimension) and concatenate heads
        # attn_output: [bs, num_heads, num_bev_queue, num_query, head_dim]
        # Slice temporal dim=2 at last index
        starts = F._from_data(self.name + '.take_last_starts', np.array([self.num_bev_queue - 1], dtype=np.int64), is_const=True)
        ends   = F._from_data(self.name + '.take_last_ends',   np.array([self.num_bev_queue], dtype=np.int64), is_const=True)
        axes   = F._from_data(self.name + '.take_last_axes',   np.array([2], dtype=np.int64), is_const=True)
        steps  = F._from_data(self.name + '.take_last_steps',  np.array([1], dtype=np.int64), is_const=True)
        slicer = F.SliceF(self.name + '.take_last', out_shape=[bs, num_heads, 1, num_query, head_dim])
        output = slicer(attn_output, starts, ends, axes, steps)
        output = F.Squeeze(self.name + '.squeeze_time')(output, F._from_data(self.name + '.squeeze_time_axes', np.array([2], dtype=np.int64), is_const=True))
        # Reorder to [bs, num_query, num_heads, head_dim]
        reorder_concat = F.Transpose(self.name + '.reorder_concat', perm=(0, 2, 1, 3))
        reorder_concat.set_module(self)
        output = reorder_concat(output)
        # Concatenate heads: reshape to [bs, num_query, embed_dims]
        output = output.reshape([bs, num_query, self.embed_dims])

        return output
