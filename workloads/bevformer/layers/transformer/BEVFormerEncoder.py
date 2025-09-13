#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from workloads.bevformer.layers.transformer.BEVFormerLayer import BEVFormerLayer

class BEVFormerEncoder(SimNN.Module):
    """BEVFormer Encoder with stacked transformer layers"""

    def __init__(self, name, num_layers=6, embed_dims=256, num_heads=8,
                 feedforward_channels=1024, ffn_dropout=0.1, dropout=0.1,
                 num_bev_queue=2):
        super(BEVFormerEncoder, self).__init__()
        self.name = name
        self.num_layers = num_layers

        # Build transformer layers
        self.layers = []
        for i in range(num_layers):
            layer = BEVFormerLayer(
                name + f'.layer_{i}',
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                dropout=dropout,
                num_bev_queue=num_bev_queue
            )
            self.layers.append(layer)

        super().link_op2module()

    def __call__(self, bev_queries, mlvl_feats, bev_pos, prev_bev=None, shift=None):
        """
        Forward pass through encoder layers

        Args:
            bev_queries: BEV query embeddings [bs, num_query, embed_dims]
            mlvl_feats: Multi-level image features [num_levels, bs, num_cams, embed_dims, h, w]
            bev_pos: BEV positional embeddings [bs, embed_dims, bev_h, bev_w]
            prev_bev: Previous BEV features for temporal attention
            shift: Ego-motion shift information

        Returns:
            Updated BEV queries after all layers
        """
        for i, layer in enumerate(self.layers):
            bev_queries = layer(
                bev_queries, mlvl_feats, bev_pos,
                prev_bev=prev_bev, shift=shift
            )

        return bev_queries
