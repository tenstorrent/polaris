#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import numpy as np
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.bevformer.layers.transformer.BEVFormerEncoder import BEVFormerEncoder

class BEVFormerTransformer(SimNN.Module):
    """Main BEVFormer Transformer combining encoder and decoder"""

    def __init__(self, name, embed_dims=256, num_cams=6, bev_h=200, bev_w=200,
                 num_layers=6, num_heads=8, feedforward_channels=1024,
                 rotate_prev_bev=True, use_shift=True, use_can_bus=True,
                 can_bus_norm=True, use_cams_embeds=True, num_bev_queue=2):
        super(BEVFormerTransformer, self).__init__()
        self.name = name

        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_bev_queue = num_bev_queue

        # Configuration flags
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        # Embeddings
        # Note: Do not reuse the same Embedding op handle multiple times in a graph path.
        # Create per-use handles where needed (level embeddings).

        # Ego-motion processing
        self.can_bus_mlp: Any = SimNN.Module()
        self.can_bus_mlp.linear1 = F.Linear(name + '.can_bus_mlp.0', 18, embed_dims // 2)
        self.can_bus_mlp.relu1 = F.Relu(name + '.can_bus_mlp.1')
        self.can_bus_mlp.linear2 = F.Linear(name + '.can_bus_mlp.2', embed_dims // 2, embed_dims)
        self.can_bus_mlp.relu2 = F.Relu(name + '.can_bus_mlp.3')

        def can_bus_forward(x):
            x = self.can_bus_mlp.linear1(x)
            x = self.can_bus_mlp.relu1(x)
            x = self.can_bus_mlp.linear2(x)
            x = self.can_bus_mlp.relu2(x)
            return x

        self.can_bus_mlp.__call__ = can_bus_forward  # type: ignore[method-assign]

        if can_bus_norm:
            self.can_bus_mlp.norm = F.LayerNorm(name + '.can_bus_norm', embed_dims)
            def can_bus_forward_with_norm(x):
                x = self.can_bus_mlp.linear1(x)
                x = self.can_bus_mlp.relu1(x)
                x = self.can_bus_mlp.linear2(x)
                x = self.can_bus_mlp.relu2(x)
                x = self.can_bus_mlp.norm(x)
                return x
            self.can_bus_mlp.__call__ = can_bus_forward_with_norm  # type: ignore[method-assign]

        # Encoder
        self.encoder = BEVFormerEncoder(
            name + '.encoder',
            num_layers=num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            num_bev_queue=num_bev_queue
        )

        # Pre-register per-level ops to ensure graph capture
        self.num_levels = 4  # typical FPN levels; adjust if config differs
        # Concat op for flattening along token dimension [bs, tokens, C]
        self.feat_concat = F.ConcatX(self.name + '.feat_concat', axis=1)
        # Level and camera add ops per level
        for i in range(self.num_levels):
            setattr(self, f'level_add_{i}', F.Add(self.name + f'.level_add_{i}'))
            setattr(self, f'cam_add_{i}',   F.Add(self.name + f'.cam_add_{i}'))
            # Embedding handles per level (to register in graph)
            setattr(self, f'level_embeds_{i}', F.Embedding(self.name + f'.level_embeds_{i}', 4, self.embed_dims))
            setattr(self, f'cams_embeds_{i}',  F.Embedding(self.name + f'.cams_embeds_{i}', self.num_cams, self.embed_dims))
            # Reshape handles per level for consistent graph capture
            setattr(self, f'level_reshape_{i}', F.ReshapeFixed(self.name + f'.level_reshape_{i}', [1, 1, self.embed_dims, 1, 1]))
            setattr(self, f'cam_reshape_{i}',   F.ReshapeFixed(self.name + f'.cam_reshape_{i}',   [1, self.num_cams, self.embed_dims, 1, 1]))

        super().link_op2module()

    def get_bev_features(self, mlvl_feats, bev_queries, bev_pos, prev_bev=None,
                        shift=None, **kwargs):
        """
        Extract BEV features using encoder

        Args:
            mlvl_feats: Multi-level image features [num_levels, bs, num_cams, embed_dims, h, w]
            bev_queries: BEV query embeddings [bs, bev_h*bev_w, embed_dims]
            bev_pos: BEV positional embeddings [bs, embed_dims, bev_h, bev_w]
            prev_bev: Previous BEV features
            shift: Ego-motion shift information
            **kwargs: Additional arguments including img_metas

        Returns:
            BEV features [bs, bev_h*bev_w, embed_dims]
        """
        bs = mlvl_feats[0].shape[0]

        # Process previous BEV features with ego-motion compensation
        if prev_bev is not None:
            prev_bev = self._process_prev_bev(prev_bev, kwargs.get('img_metas', []))

        # Add can bus signals to BEV queries
        if self.use_can_bus and 'img_metas' in kwargs:
            can_bus = self._extract_can_bus_signals(kwargs['img_metas'])
            bev_queries = bev_queries + can_bus

        # Process multi-level features
        graph_build = kwargs.get('graph_build', False)
        feat_flatten, spatial_shapes, level_start_index = self._process_mlvl_feats(mlvl_feats, graph_build=graph_build)

        # Encoder forward pass
        bev_embed = self.encoder(
            bev_queries, feat_flatten, bev_pos,
            prev_bev=prev_bev, shift=shift
        )

        return bev_embed

    def _process_prev_bev(self, prev_bev, img_metas):
        """Process previous BEV features with ego-motion compensation"""
        if prev_bev.shape[1] == self.bev_h * self.bev_w:
            # reshape to [bs, bev_h, bev_w, C]
            prev_bev = prev_bev.reshape([prev_bev.shape[0], self.bev_h, self.bev_w, self.embed_dims])
            prev_bev = F.Transpose(self.name + '.prev_bev_transpose', perm=(0, 3, 1, 2))(prev_bev)

        if self.rotate_prev_bev and img_metas:
            # Rotation is a no-op under ttsim graph; rely on shifts only
            pass

        return prev_bev

    def _extract_can_bus_signals(self, img_metas):
        """Extract and process can bus signals"""
        can_bus_signals = []
        for meta in img_metas:
            can_bus = meta.get('can_bus', [0] * 18)[:18]  # Take first 18 elements
            can_bus_signals.append(can_bus)

        can_bus_tensor = F._from_data('can_bus_tensor',
                                    np.array(can_bus_signals, dtype=np.float32))
        can_bus_embed = self.can_bus_mlp(can_bus_tensor)

        return can_bus_embed.unsqueeze(1)

    def _process_mlvl_feats(self, mlvl_feats, graph_build=False):
        """Process multi-level features for spatial attention"""
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]

        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cams, embed_dims, h, w = feat.shape
            spatial_shapes.append([h, w])

            if not graph_build:
                # Add level embeddings using pre-registered op handle
                level_embeds = getattr(self, f'level_embeds_{lvl}') if lvl < self.num_levels else F.Embedding(self.name + f'.level_embeds_{lvl}', 4, self.embed_dims)
                level_idx_tensor = F._from_data(self.name + f'.level_idx_{lvl}', np.array([lvl], dtype=np.int64), is_const=True)
                level_idx_tensor.set_module(self)
                setattr(self, f'level_idx_tensor_{lvl}', level_idx_tensor)
                level_embed = level_embeds(level_idx_tensor)
                # reshape to [1, 1, C, 1, 1] for broadcast over [B, N, C, H, W]
                level_embed = (getattr(self, f'level_reshape_{lvl}') if lvl < self.num_levels else F.ReshapeFixed(self.name + f'.level_reshape_{lvl}', [1, 1, self.embed_dims, 1, 1]))(level_embed)

                # Add camera embeddings if enabled - use pre-registered handle
                if self.use_cams_embeds:
                    cams_embeds = getattr(self, f'cams_embeds_{lvl}') if lvl < self.num_levels else F.Embedding(self.name + f'.cams_embeds_{lvl}', num_cams, self.embed_dims)
                    cam_indices = F._from_data(self.name + f'.cam_indices_{lvl}', np.arange(num_cams, dtype=np.int64), is_const=True)
                    cam_indices.set_module(self)
                    setattr(self, f'cam_indices_tensor_{lvl}', cam_indices)
                    cam_embed = cams_embeds(cam_indices)
                    # reshape to [1, N, C, 1, 1] to match [B, N, C, H, W]
                    cam_embed = (getattr(self, f'cam_reshape_{lvl}') if lvl < self.num_levels else F.ReshapeFixed(self.name + f'.cam_reshape_{lvl}', [1, num_cams, self.embed_dims, 1, 1]))(cam_embed)
                    cam_add = getattr(self, f'cam_add_{lvl}') if lvl < self.num_levels else F.Add(self.name + f'.cam_add_{lvl}')
                    feat = cam_add(feat, cam_embed)

                level_add = getattr(self, f'level_add_{lvl}') if lvl < self.num_levels else F.Add(self.name + f'.level_add_{lvl}')
                feat = level_add(feat, level_embed)

            # Flatten spatial dimensions to [bs, num_cams*h*w, C] with a fixed reshape op
            reshape_name = self.name + f'.flatten_reshape_{lvl}'
            out_shape = [bs, num_cams * h * w, embed_dims]
            reshape_flat = getattr(self, f'flatten_reshape_{lvl}', None)
            if reshape_flat is None or getattr(reshape_flat, 'out_shape', None) != out_shape:
                reshape_flat = F.ReshapeFixed(reshape_name, out_shape)
                reshape_flat.set_module(self)
                setattr(self, f'flatten_reshape_{lvl}', reshape_flat)
            feat_flat = reshape_flat(feat)
            feat_flatten.append(feat_flat)

            if lvl > 0:
                level_start_index.append(level_start_index[-1] + h * w)

        # Concatenate features from all levels along token dimension
        feat_flatten = self.feat_concat(*feat_flatten)

        return feat_flatten, spatial_shapes, level_start_index

    def __call__(self, mlvl_feats, bev_queries, bev_pos, prev_bev=None, shift=None, **kwargs):
        """Full forward pass"""
        return self.get_bev_features(mlvl_feats, bev_queries, bev_pos, prev_bev, shift, **kwargs)
