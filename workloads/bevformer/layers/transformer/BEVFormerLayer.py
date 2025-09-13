#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from workloads.bevformer.layers.attention.SpatialCrossAttention import SpatialCrossAttention
from workloads.bevformer.layers.attention.TemporalSelfAttention import TemporalSelfAttention

class BEVFormerLayer(SimNN.Module):
    """Single layer of BEVFormer transformer encoder"""

    def __init__(self, name, embed_dims=256, num_heads=8, feedforward_channels=1024,
                 ffn_dropout=0.1, dropout=0.1, num_bev_queue=2):
        super(BEVFormerLayer, self).__init__()
        self.name = name

        # Normalization layers
        self.norm1 = F.LayerNorm(name + '.norm1', embed_dims)
        self.norm2 = F.LayerNorm(name + '.norm2', embed_dims)

        # Attention layers
        self.temporal_self_attention = TemporalSelfAttention(
            name + '.temporal_self_attention',
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_bev_queue=num_bev_queue
        )

        self.spatial_cross_attention = SpatialCrossAttention(
            name + '.spatial_cross_attention',
            embed_dims=embed_dims,
            num_cams=6
        )
        self.add_temporal = F.Add(name + '.temporal_residual')
        self.add_spatial  = F.Add(name + '.spatial_residual')
        self.add_ffn      = F.Add(name + '.ffn_residual')

        # Feed-forward network ops
        self.ffn_linear1 = F.Linear(name + '.ffn.0', embed_dims, feedforward_channels)
        self.ffn_relu = F.Relu(name + '.ffn.1')
        self.ffn_dropout1 = F.Dropout(name + '.ffn.2', ffn_dropout)
        self.ffn_linear2 = F.Linear(name + '.ffn.3', feedforward_channels, embed_dims)
        self.ffn_dropout2 = F.Dropout(name + '.ffn.4', dropout)

        def _ffn_forward(x):
            x = self.ffn_linear1(x)
            x = self.ffn_relu(x)
            x = self.ffn_dropout1(x)
            x = self.ffn_linear2(x)
            x = self.ffn_dropout2(x)
            return x

        # bind helper for internal use
        self._ffn_forward = _ffn_forward

        super().link_op2module()

    def __call__(self, bev_queries, mlvl_feats, bev_pos, prev_bev=None, shift=None):
        """
        Forward pass of BEVFormer layer

        Args:
            bev_queries: BEV query embeddings [bs, num_query, embed_dims]
            mlvl_feats: Multi-level image features [num_levels, bs, num_cams, embed_dims, h, w]
            bev_pos: BEV positional embeddings [bs, embed_dims, bev_h, bev_w]
            prev_bev: Previous BEV features for temporal attention
            shift: Ego-motion shift information
        """
        # Temporal Self-Attention
        identity = bev_queries
        bev_queries = self.norm1(bev_queries)
        bev_queries = self.temporal_self_attention(
            bev_queries, bev_pos=bev_pos, prev_bev=prev_bev, shift=shift
        )
        bev_queries = self.add_temporal(identity, bev_queries)

        # Spatial Cross-Attention
        identity = bev_queries
        bev_queries = self.norm2(bev_queries)

        # Prepare multi-level features for spatial attention
        # Allow either pre-flattened tensor (SimTensor) or list of per-level tensors
        spatial_shapes = []
        feat_flatten = None

        if isinstance(mlvl_feats, SimTensor):
            # Already processed/flattened by transformer
            feat_flatten = mlvl_feats
        else:
            feat_list = []
            for level, feat in enumerate(mlvl_feats):
                bs, num_cams, embed_dims, h, w = feat.shape
                spatial_shapes.append([h, w])
                # Flatten to [bs, num_cams*h*w, embed_dims]
                feat_flat = feat.reshape([bs, num_cams * h * w, embed_dims])
                feat_list.append(feat_flat)
            # Concatenate features from all levels along token dim
            feat_flatten = F.ConcatX(self.name + '.feat_concat', axis=1)(*feat_list)

        bev_queries = self.spatial_cross_attention(
            bev_queries, feat_flatten, feat_flatten,
            spatial_shapes=spatial_shapes, bev_pos=bev_pos
        )
        bev_queries = self.add_spatial(identity, bev_queries)

        # Feed-Forward Network
        identity = bev_queries
        bev_queries = self._ffn_forward(bev_queries)
        bev_queries = self.add_ffn(identity, bev_queries)

        return bev_queries
