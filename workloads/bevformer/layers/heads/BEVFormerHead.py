#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import math
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

class BEVFormerHead(SimNN.Module):
    """Detection head for BEVFormer - handles 3D object detection from BEV features"""

    def __init__(self, name, bev_h=200, bev_w=200, num_query=900, num_classes=10,
                 embed_dims=256, num_reg_fcs=2, code_size=10, num_cls_fcs=2,
                 transformer=None, loss_cls=None, loss_bbox=None):
        super(BEVFormerHead, self).__init__()
        self.name = name

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.code_size = code_size
        self.num_cls_fcs = num_cls_fcs

        # Initialize BEV embedding and positional encoding (store as parameter tensors for ttsim)
        self.bev_embedding = self.create_shape_tensor('bev_embedding', [bev_h * bev_w, embed_dims], is_param=True)

        # Positional encoding for BEV
        self.positional_encoding = self._build_positional_encoding()

        # Query embeddings (object queries)
        self.query_embedding = self.create_shape_tensor('query_embedding', [num_query, embed_dims * 2], is_param=True)

        # Classification and regression branches
        self._build_cls_reg_branches()

        # Transformer for detection
        self.transformer = transformer

        super().link_op2module()

    def _build_positional_encoding(self):
        """Build positional encoding for BEV grid"""
        # Create 2D positional encoding for BEV grid
        pos_embed = F._from_data(self.name + '.pos_embed', np.zeros((self.bev_h * self.bev_w, self.embed_dims), dtype=np.float32))

        # Generate position indices
        y_positions = np.arange(self.bev_h, dtype=np.float32)
        x_positions = np.arange(self.bev_w, dtype=np.float32)

        # Create meshgrid
        y_grid, x_grid = np.meshgrid(y_positions, x_positions, indexing='ij')
        y_grid = y_grid.flatten()
        x_grid = x_grid.flatten()

        # Apply sinusoidal positional encoding
        div_term = np.exp(np.arange(0, self.embed_dims, 2) * -(np.log(10000.0) / self.embed_dims))

        pos_embed_x = np.sin(x_grid[:, None] * div_term)
        pos_embed_y = np.cos(y_grid[:, None] * div_term)

        # Interleave x and y encodings
        pos_encoding = np.zeros((self.bev_h * self.bev_w, self.embed_dims))
        pos_encoding[:, 0::2] = pos_embed_x
        pos_encoding[:, 1::2] = pos_embed_y

        return F._from_data('positional_encoding', pos_encoding)

    def _build_cls_reg_branches(self):
        """Build classification and regression branches"""
        # Classification branch as lightweight callable container
        self.cls_branch: Any = SimNN.Module()
        self.cls_branch.name = self.name + '.cls_branch'
        for i in range(self.num_cls_fcs):
            setattr(self.cls_branch, f'linear_{i}', F.Linear(self.name + f'.cls_branch.linear_{i}',
                                                          self.embed_dims, self.embed_dims))
            setattr(self.cls_branch, f'layernorm_{i}', F.LayerNorm(self.name + f'.cls_branch.layernorm_{i}',
                                                                self.embed_dims))
            setattr(self.cls_branch, f'relu_{i}', F.Relu(self.name + f'.cls_branch.relu_{i}'))

        setattr(self.cls_branch, 'final_linear', F.Linear(self.name + '.cls_branch.final_linear',
                                                       self.embed_dims, self.num_classes))

        # Regression branch as lightweight callable container
        self.reg_branch: Any = SimNN.Module()
        self.reg_branch.name = self.name + '.reg_branch'
        for i in range(self.num_reg_fcs):
            setattr(self.reg_branch, f'linear_{i}', F.Linear(self.name + f'.reg_branch.linear_{i}',
                                                          self.embed_dims, self.embed_dims))
            setattr(self.reg_branch, f'relu_{i}', F.Relu(self.name + f'.reg_branch.relu_{i}'))

        setattr(self.reg_branch, 'final_linear', F.Linear(self.name + '.reg_branch.final_linear',
                                                       self.embed_dims, self.code_size))

        # Add forward methods to branches
        def cls_forward(x):
            for i in range(self.num_cls_fcs):
                x = getattr(self.cls_branch, f'linear_{i}')(x)
                x = getattr(self.cls_branch, f'layernorm_{i}')(x)
                x = getattr(self.cls_branch, f'relu_{i}')(x)
            x = self.cls_branch.final_linear(x)
            return x

        def reg_forward(x):
            for i in range(self.num_reg_fcs):
                x = getattr(self.reg_branch, f'linear_{i}')(x)
                x = getattr(self.reg_branch, f'relu_{i}')(x)
            x = self.reg_branch.final_linear(x)
            return x

        self.cls_branch.__call__ = cls_forward  # type: ignore[method-assign]
        self.reg_branch.__call__ = reg_forward  # type: ignore[method-assign]
        # Also expose internal forwards to avoid relying on instance __call__
        self._cls_forward = cls_forward
        self._reg_forward = reg_forward

    def __call__(self, mlvl_feats, img_metas, prev_bev=None):
        """
        Forward pass of detection head

        Args:
            mlvl_feats: Multi-level image features
            img_metas: Image metadata
            prev_bev: Previous BEV features for temporal fusion

        Returns:
            Dictionary containing classification and regression outputs
        """
        bs = mlvl_feats[0].shape[0]

        # Get BEV queries and positional embeddings
        bev_queries = self.bev_embedding
        bev_queries.set_module(self)
        bev_pos = self.positional_encoding
        bev_pos.set_module(self)

        # Expand for batch size
        bev_queries = bev_queries.unsqueeze(0).reshape([1, self.bev_h * self.bev_w, self.embed_dims])
        bev_queries = bev_queries.reshape([bs, self.bev_h * self.bev_w, self.embed_dims])

        bev_pos = bev_pos.reshape([1, self.embed_dims, self.bev_h, self.bev_w])
        bev_pos = bev_pos.reshape([bs, self.embed_dims, self.bev_h, self.bev_w])

        # Get object queries
        query_embeds = self.query_embedding  # [num_query, embed_dims * 2]
        query_embeds.set_module(self)
        # Replace Split with slicing along last dim using SliceF
        # query_pos: [:, 0:embed_dims]
        qpos_starts = F._from_data(self.name + '.qpos_starts', np.array([0], dtype=np.int64), is_const=True)
        qpos_ends   = F._from_data(self.name + '.qpos_ends',   np.array([self.embed_dims], dtype=np.int64), is_const=True)
        qpos_axes   = F._from_data(self.name + '.qpos_axes',   np.array([1], dtype=np.int64), is_const=True)
        qpos_steps  = F._from_data(self.name + '.qpos_steps',  np.array([1], dtype=np.int64), is_const=True)
        qpos_slice  = F.SliceF(self.name + '.qpos_slice', out_shape=[self.num_query, self.embed_dims])
        qpos_slice.set_module(self)
        query_pos   = qpos_slice(query_embeds, qpos_starts, qpos_ends, qpos_axes, qpos_steps)

        # query: [:, embed_dims:2*embed_dims]
        q_starts = F._from_data(self.name + '.q_starts', np.array([self.embed_dims], dtype=np.int64), is_const=True)
        q_ends   = F._from_data(self.name + '.q_ends',   np.array([self.embed_dims * 2], dtype=np.int64), is_const=True)
        q_axes   = F._from_data(self.name + '.q_axes',   np.array([1], dtype=np.int64), is_const=True)
        q_steps  = F._from_data(self.name + '.q_steps',  np.array([1], dtype=np.int64), is_const=True)
        q_slice  = F.SliceF(self.name + '.q_slice', out_shape=[self.num_query, self.embed_dims])
        q_slice.set_module(self)
        query    = q_slice(query_embeds, q_starts, q_ends, q_axes, q_steps)

        # Expand queries for batch
        query_pos = query_pos.unsqueeze(0).reshape([bs, self.num_query, self.embed_dims])

        query = query.unsqueeze(0).reshape([bs, self.num_query, self.embed_dims])

        # Transformer forward pass (support encoder-only return)
        graph_build = (img_metas is None)
        t_out = self.transformer(mlvl_feats=mlvl_feats, bev_queries=bev_queries, bev_pos=bev_pos, prev_bev=prev_bev, graph_build=graph_build)
        if isinstance(t_out, tuple) and len(t_out) == 4:
            bev_embed, hs, init_reference, inter_references = t_out
        else:
            bev_embed = t_out
            hs = [query]  # fallback: use object queries as decoder features

        # When running under Polaris script (img_metas is None), return minimal outputs to finalize graph
        if img_metas is None:
            return {
                'all_cls_scores': [],
                'all_bbox_preds': [],
                'enc_outputs_class': None,
                'enc_outputs_coord_unact': None,
                'bev_embed': bev_embed
            }

        # Classification and regression predictions
        outputs_classes = []
        outputs_coords = []

        for lvl in range(len(hs)):
            # Classification
            cls_output = self._cls_forward(hs[lvl])
            outputs_classes.append(cls_output)

            # Regression
            reg_output = self._reg_forward(hs[lvl])
            outputs_coords.append(reg_output)

        # Keep as lists to avoid Stack op requirements
        # outputs_classes = F.Stack(self.name + '.cls_stack', outputs_classes, dim=0)
        # outputs_coords = F.Stack(self.name + '.reg_stack', outputs_coords, dim=0)

        return {
            'all_cls_scores': outputs_classes,  # [num_dec_layers, bs, num_query, num_classes]
            'all_bbox_preds': outputs_coords,   # [num_dec_layers, bs, num_query, code_size]
            'enc_outputs_class': None,
            'enc_outputs_coord_unact': None,
            'bev_embed': bev_embed
        }

    def get_bboxes(self, preds_dict, img_metas, rescale=False):
        """
        Get bounding boxes from predictions

        Args:
            preds_dict: Predictions from forward pass
            img_metas: Image metadata
            rescale: Whether to rescale predictions

        Returns:
            List of bounding boxes, scores, and labels
        """
        all_cls_scores = preds_dict['all_cls_scores']
        all_bbox_preds = preds_dict['all_bbox_preds']

        # Use predictions from last decoder layer (lists -> take last element)
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        bs = cls_scores.shape[0] if (img_metas is None) else len(img_metas)
        for batch_idx in range(bs):
            # Slice batch dimension using SliceF since SimTensor is not subscriptable
            starts = F._from_data(self.name + f'.bbox_starts_{batch_idx}', np.array([batch_idx, 0, 0], dtype=np.int64), is_const=True)
            ends   = F._from_data(self.name + f'.bbox_ends_{batch_idx}',   np.array([batch_idx+1, self.num_query, self.code_size], dtype=np.int64), is_const=True)
            axes   = F._from_data(self.name + f'.bbox_axes_{batch_idx}',   np.array([0, 1, 2], dtype=np.int64), is_const=True)
            steps  = F._from_data(self.name + f'.bbox_steps_{batch_idx}',  np.array([1, 1, 1], dtype=np.int64), is_const=True)
            slicer_bbox = F.SliceF(self.name + f'.bbox_slice_{batch_idx}', out_shape=[1, self.num_query, self.code_size])
            batch_bbox_preds = slicer_bbox(bbox_preds, starts, ends, axes, steps)
            batch_bbox_preds = F.Squeeze(self.name + f'.bbox_squeeze_{batch_idx}')(
                batch_bbox_preds, F._from_data(self.name + f'.bbox_squeeze_axes_{batch_idx}', np.array([0], dtype=np.int64), is_const=True))

            starts_c = F._from_data(self.name + f'.cls_starts_{batch_idx}', np.array([batch_idx, 0, 0], dtype=np.int64), is_const=True)
            ends_c   = F._from_data(self.name + f'.cls_ends_{batch_idx}',   np.array([batch_idx+1, self.num_query, self.num_classes], dtype=np.int64), is_const=True)
            axes_c   = F._from_data(self.name + f'.cls_axes_{batch_idx}',   np.array([0, 1, 2], dtype=np.int64), is_const=True)
            steps_c  = F._from_data(self.name + f'.cls_steps_{batch_idx}',  np.array([1, 1, 1], dtype=np.int64), is_const=True)
            slicer_cls = F.SliceF(self.name + f'.cls_slice_{batch_idx}', out_shape=[1, self.num_query, self.num_classes])
            batch_cls_scores = slicer_cls(cls_scores, starts_c, ends_c, axes_c, steps_c)
            batch_cls_scores = F.Squeeze(self.name + f'.cls_squeeze_{batch_idx}')(
                batch_cls_scores, F._from_data(self.name + f'.cls_squeeze_axes_{batch_idx}', np.array([0], dtype=np.int64), is_const=True))

            # Apply sigmoid to classification scores
            batch_cls_scores = F.Sigmoid(self.name + f'.cls_sigmoid_{batch_idx}')(batch_cls_scores)

            # Convert bbox predictions to 3D format (simplified)
            # In full implementation, this would involve proper bbox decoding
            bboxes_3d = batch_bbox_preds
            # Compute scores and labels; if ArgMax/ReduceMax not available, keep tensors
            # Here we leave as-is (placeholders) since ttsim may not support ArgMax on last dim conveniently
            scores = batch_cls_scores  # placeholder
            labels = batch_cls_scores  # placeholder

            result_list.append((bboxes_3d, scores, labels))

        return result_list
