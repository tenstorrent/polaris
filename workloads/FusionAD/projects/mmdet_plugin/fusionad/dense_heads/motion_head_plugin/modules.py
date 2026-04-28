#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of motion transformer decoder modules for FusionAD.

Inference-only conversion of the transformer interaction layers and the main
MotionTransformerDecoder.  Training-specific logic (dropout, gradient ops)
is omitted.  Unused layers (ModalInteraction, TopoInteraction) are omitted.

Classes:
  - IntentionInteraction       : TransformerEncoderLayer wrapper for mode interaction.
  - TrackAgentInteraction      : TransformerDecoderLayer wrapper for agent interaction.
  - MapInteraction             : TransformerDecoderLayer wrapper for map interaction.
  - MotionTransformerDecoder   : Main decoder composing all interaction + MLP layers.
"""

#----------------------------------PyTorch-----------------------------#

# import torch
# import torch.nn as nn
# from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
# from mmcv.cnn.bricks.transformer import build_transformer_layer
# from mmcv.runner.base_module import BaseModule
# from projects.mmdet3d_plugin.models.utils.functional import (
#     norm_points,
#     pos2posemb2d,
#     trajectory_coordinate_transform
# )


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class MotionTransformerDecoder(BaseModule):
#     """Implements the decoder in DETR3D transformer.
#     Args:
#         return_intermediate (bool): Whether to return intermediate outputs.
#         coder_norm_cfg (dict): Config of last normalization layer. Default：
#             `LN`.
#     """

#     def __init__(self, pc_range=None, embed_dims=256, transformerlayers=None, num_layers=3, **kwargs):
#         super(MotionTransformerDecoder, self).__init__()
#         self.pc_range = pc_range
#         self.embed_dims = embed_dims
#         self.num_layers = num_layers
#         self.intention_interaction_layers = IntentionInteraction()
#         self.track_agent_interaction_layers = nn.ModuleList(
#             [TrackAgentInteraction() for i in range(self.num_layers)])
#         self.map_interaction_layers = nn.ModuleList(
#             [MapInteraction() for i in range(self.num_layers)])
#         self.bev_interaction_layers = nn.ModuleList(
#             [build_transformer_layer(transformerlayers) for i in range(self.num_layers)])

#         self.track_agent_interaction_rl = TrackAgentInteraction()
#         self.map_interaction_rl = MapInteraction()
#         self.bev_interaction_rl = build_transformer_layer(transformerlayers)
#         self.out_mid_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims*4, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         #self.topo_interaction_layer = TopoInteraction()
#         # self.mode_interaction_layers = nn.ModuleList(
#         #     [ModalInteraction() for i in range(self.num_layers)])

#         self.traj_embed = nn.Sequential(
#             nn.Linear(2, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.static_dynamic_rl = nn.Sequential(
#             nn.Linear(self.embed_dims*2, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.dynamic_embed_rl = nn.Sequential(
#             nn.Linear(self.embed_dims*3, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.in_query_rl = nn.Sequential(
#             nn.Linear(self.embed_dims*2, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.static_dynamic_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims*2, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.dynamic_embed_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims*3, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.in_query_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims*2, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.out_query_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims*4, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )

#     def forward(self,
#                 track_query,
#                 lane_query,
#                 track_query_pos=None,
#                 lane_query_pos=None,
#                 track_bbox_results=None,
#                 bev_embed=None,
#                 reference_trajs=None,
#                 traj_reg_branches=None,
#                 traj_refine_branch=None,
#                 agent_level_embedding=None,
#                 scene_level_ego_embedding=None,
#                 scene_level_offset_embedding=None,
#                 learnable_embed=None,
#                 agent_level_embedding_layer=None,
#                 scene_level_ego_embedding_layer=None,
#                 scene_level_offset_embedding_layer=None,
#                 **kwargs):
#         """Forward function for `Detr3DTransformerDecoder`.
#         Args:
#             agent_query (B, A, D)
#             map_query (B, M, D)
#             map_query_pos (B, G, D)
#             static_intention_embed (B, A, P, D)
#             offset_query_embed (B, A, P, D)
#             global_intention_embed (B, A, P, D)
#             learnable_intention_embed (B, A, P, D)
#             det_query_pos (B, A, D)
#         Returns:
#             None
#         """
#         intermediate = []
#         intermediate_reference_trajs = []

#         B, _, P, D = agent_level_embedding.shape
#         track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)
#         track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)  # (B, A, P, D)

#         # static intention embedding, which is imutable throughout all layers
#         agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
#         static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
#         reference_trajs_input = reference_trajs.unsqueeze(4).detach()
#         #lane_query = self.topo_interaction_layer(lane_query, lane_query_pos)
#         query_embed = torch.zeros_like(static_intention_embed)
#         for lid in range(self.num_layers):
#             # fuse static and dynamic intention embedding
#             # the dynamic intention embedding is the output of the previous layer, which is initialized with anchor embedding
#             dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
#                 [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))

#             # fuse static and dynamic intention embedding
#             #query_embed_intention = self.static_dynamic_fuser(torch.cat(
#             #    [static_intention_embed, dynamic_query_embed], dim=-1))  # (B, A, P, D)

#             # fuse intention embedding with query embedding
#             query_embed = self.in_query_fuser(torch.cat([query_embed, dynamic_query_embed], dim=-1))

#             # interaction between agents
#             track_query_embed = self.track_agent_interaction_layers[lid](
#                 query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

#             # interaction between agents and map
#             map_query_embed = self.map_interaction_layers[lid](
#                 query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

#             # interaction between agents and bev, ie. interaction between agents and goals
#             # implemented with deformable transformer
#             bev_query_embed = self.bev_interaction_layers[lid](
#                 query_embed,
#                 value=bev_embed,
#                 query_pos=track_query_pos_bc,
#                 bbox_results=track_bbox_results,
#                 reference_trajs=reference_trajs_input,
#                 **kwargs)

#             # fusing the embeddings from different interaction layers
#             query_embed = [track_query_embed, map_query_embed, bev_query_embed, track_query_bc+track_query_pos_bc]
#             query_embed = torch.cat(query_embed, dim=-1)
#             query_embed = self.out_query_fuser(query_embed)
#             # query_embed = self.mode_interaction_layers[lid](query_embed)

#             if traj_reg_branches is not None:
#                 # update reference trajectory
#                 tmp = traj_reg_branches[lid](query_embed)
#                 bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
#                 tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

#                 # we predict speed of trajectory and use cumsum trick to get the trajectory
#                 tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
#                 new_reference_trajs = torch.zeros_like(reference_trajs)
#                 new_reference_trajs = tmp[..., :2]
#                 reference_trajs = new_reference_trajs.detach()
#                 reference_trajs_input = reference_trajs.unsqueeze(4)  # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2

#                 # update embedding, which is used in the next layer
#                 # only update the embedding of the last step, i.e. the goal
#                 ep_offset_embed = reference_trajs.detach()
#                 ep_ego_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
#                     2), track_bbox_results, with_translation_transform=True, with_rotation_transform=False).squeeze(2).detach()
#                 ep_agent_embed = trajectory_coordinate_transform(reference_trajs.unsqueeze(
#                     2), track_bbox_results, with_translation_transform=False, with_rotation_transform=True).squeeze(2).detach()

#                 agent_level_embedding = agent_level_embedding_layer(pos2posemb2d(
#                     norm_points(ep_agent_embed[..., -1, :], self.pc_range)))
#                 scene_level_ego_embedding = scene_level_ego_embedding_layer(pos2posemb2d(
#                     norm_points(ep_ego_embed[..., -1, :], self.pc_range)))
#                 scene_level_offset_embedding = scene_level_offset_embedding_layer(pos2posemb2d(
#                     norm_points(ep_offset_embed[..., -1, :], self.pc_range)))

#                 intermediate.append(query_embed)
#                 intermediate_reference_trajs.append(reference_trajs)

#         mid_embed = self.traj_embed(reference_trajs.detach()).max(-2)[0]
#         dynamic_query_embed = self.dynamic_embed_rl(torch.cat(
#                 [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))

#         mid_embed = self.in_query_rl(torch.cat([mid_embed, dynamic_query_embed], dim=-1))

#         track_rf_embed = self.track_agent_interaction_rl(
#             mid_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

#         map_rf_embed = self.map_interaction_rl(
#                 mid_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

#         bev_rf_embed = self.bev_interaction_rl(
#                 mid_embed,
#                 value=bev_embed,
#                 query_pos=track_query_pos_bc,
#                 bbox_results=track_bbox_results,
#                 reference_trajs=reference_trajs_input,
#                 **kwargs)

#         mid_query_embed = [track_rf_embed, map_rf_embed, bev_rf_embed, track_query_bc+track_query_pos_bc]
#         mid_query_embed = torch.cat(mid_query_embed, dim=-1)
#         mid_query_embed = self.out_mid_fuser(mid_query_embed)
#         offset = traj_refine_branch(mid_query_embed).unflatten(3, (n_steps, 2))
#         return torch.stack(intermediate), torch.stack(intermediate_reference_trajs), offset


# class TrackAgentInteraction(BaseModule):
#     """
#     Modeling the interaction between the agents
#     """
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         self.batch_first = batch_first
#         self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
#                                                                   nhead=num_heads,
#                                                                   dropout=dropout,
#                                                                   dim_feedforward=embed_dims*2,
#                                                                   batch_first=batch_first)

#     def forward(self, query, key, query_pos=None, key_pos=None):
#         '''
#         query: context query (B, A, P, D)
#         query_pos: mode pos embedding (B, A, P, D)
#         key: (B, A, D)
#         key_pos: (B, A, D)
#         '''
#         B, A, P, D = query.shape
#         if query_pos is not None:
#             query = query + query_pos
#         if key_pos is not None:
#             key = key + key_pos
#         mem = key.expand(B*A, -1, -1)
#         # N, A, P, D -> N*A, P, D
#         query = torch.flatten(query, start_dim=0, end_dim=1)
#         query = self.interaction_transformer(query, mem)
#         query = query.view(B, A, P, D)
#         return query


# class MapInteraction(BaseModule):
#     """
#     Modeling the interaction between the agent and the map
#     """
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         self.batch_first = batch_first
#         self.interaction_transformer = nn.TransformerDecoderLayer(d_model=embed_dims,
#                                                                   nhead=num_heads,
#                                                                   dropout=dropout,
#                                                                   dim_feedforward=embed_dims*2,
#                                                                   batch_first=batch_first)

#     def forward(self, query, key, query_pos=None, key_pos=None):
#         '''
#         x: context query (B, A, P, D)
#         query_pos: mode pos embedding (B, A, P, D)
#         '''
#         B, A, P, D = query.shape
#         if query_pos is not None:
#             query = query + query_pos
#         if key_pos is not None:
#             key = key + key_pos

#         # N, A, P, D -> N*A, P, D
#         query = torch.flatten(query, start_dim=0, end_dim=1)
#         mem = key.expand(B*A, -1, -1)
#         query = self.interaction_transformer(query, mem)
#         query = query.view(B, A, P, D)
#         return query


# class IntentionInteraction(BaseModule):
#     """
#     Modeling the interaction between anchors
#     """
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         self.batch_first = batch_first
#         self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
#                                                                   nhead=num_heads,
#                                                                   dropout=dropout,
#                                                                   dim_feedforward=embed_dims*2,
#                                                                   batch_first=batch_first)

#     def forward(self, query):
#         B, A, P, D = query.shape
#         # B, A, P, D -> B*A,P, D
#         rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
#         rebatch_x = self.interaction_transformer(rebatch_x)
#         out = rebatch_x.view(B, A, P, D)
#         return out

# class ModalInteraction(BaseModule):
#     """
#     Modeling the interaction between anchors
#     """
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         self.batch_first = batch_first
#         self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
#                                                                   nhead=num_heads,
#                                                                   dropout=dropout,
#                                                                   dim_feedforward=embed_dims*2,
#                                                                   batch_first=batch_first)

#     def forward(self, query):
#         B, A, P, D = query.shape
#         # B, A, P, D -> B*A,P, D
#         rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
#         rebatch_x = self.interaction_transformer(rebatch_x)
#         out = rebatch_x.view(B, A, P, D)
#         return out

# class TopoInteraction(BaseModule):
#     """
#     Modeling the interaction between anchors
#     """
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  dropout=0.1,
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)

#         self.batch_first = batch_first
#         self.interaction_transformer = nn.TransformerEncoderLayer(d_model=embed_dims,
#                                                                   nhead=num_heads,
#                                                                   dropout=dropout,
#                                                                   dim_feedforward=embed_dims*2,
#                                                                   batch_first=batch_first)

#     def forward(self, query, query_pos=None):
#         if query_pos != None:
#             query = query + query_pos
#         return self.interaction_transformer(query)

#-------------------------------TTsim-----------------------------#


import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add motion_head_plugin directory for base_motion_head imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add dense_heads directory
dense_heads_dir = os.path.abspath(os.path.join(current_dir, '..'))
if dense_heads_dir not in sys.path:
    sys.path.insert(0, dense_heads_dir)

# Add fusionad directory so "from modules.xxx" imports resolve
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import LayerNorm
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multihead_attention import MultiheadAttention
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head_plugin.base_motion_head import TwoLayerMLP


# ======================================================================
# IntentionInteraction
# ======================================================================

class IntentionInteraction(SimNN.Module):
    """
    Interaction between trajectory intention modes (anchors).

    Wraps a TransformerEncoderLayer (batch_first=True, post-norm):
        self_attn  ->  Add & Norm1  ->  FFN(fc1->ReLU->fc2)  ->  Add & Norm2

    Input (B, A, P, D) is flattened to (B*A, P, D) so each agent's P modes
    attend to each other, then reshaped back.

    PyTorch equivalent:
        nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=0.1,
                                  dim_feedforward=512, batch_first=True)

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension.  Default: 256.
        num_heads (int): Number of attention heads.  Default: 8.
        dim_feedforward (int): FFN hidden dimension.  Default: 512.
    """

    def __init__(self, name, embed_dims=256, num_heads=8, dim_feedforward=512):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.dim_feedforward = dim_feedforward

        # Self-attention
        self.self_attn = MultiheadAttention(
            f'{name}.self_attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=True, bias=True)

        # FFN
        self.ffn_fc1 = SimNN.Linear(f'{name}.ffn_fc1',
                                    in_features=embed_dims,
                                    out_features=dim_feedforward)
        self.ffn_fc2 = SimNN.Linear(f'{name}.ffn_fc2',
                                    in_features=dim_feedforward,
                                    out_features=embed_dims)
        self.ffn_relu = F.Relu(f'{name}.ffn_relu')

        # Layer norms (post-norm, no affine in TTSim)
        self.norm1 = LayerNorm(f'{name}.norm1', embed_dims)
        self.norm2 = LayerNorm(f'{name}.norm2', embed_dims)

        # Residual adds
        self.add1 = F.Add(f'{name}.add1')
        self.add2 = F.Add(f'{name}.add2')

        # Reshape ops for flatten / unflatten
        self.reshape_flat = F.Reshape(f'{name}.reshape_flat')
        self.reshape_back = F.Reshape(f'{name}.reshape_back')

        super().link_op2module()

    def __call__(self, query):
        """
        Forward pass.

        Args:
            query: SimTensor (B, A, P, D)  — per-agent mode embeddings.

        Returns:
            SimTensor (B, A, P, D) — updated mode embeddings after self-attention.
        """
        B, A, P, D = query.shape

        # Flatten (B, A, P, D) → (B*A, P, D)
        self._flat_shape = F._from_data(
            f'{self.name}._flat_shape',
            np.array([B * A, P, D], dtype=np.int64), is_const=True)
        x = self.reshape_flat(query, self._flat_shape)

        # Self-attention + residual + norm
        sa_out = self.self_attn(x)
        x = self.add1(x, sa_out)
        x = self.norm1(x)

        # FFN + residual + norm
        ffn = self.ffn_fc1(x)
        ffn = self.ffn_relu(ffn)
        ffn = self.ffn_fc2(ffn)
        x = self.add2(x, ffn)
        x = self.norm2(x)

        # Unflatten (B*A, P, D) → (B, A, P, D)
        self._back_shape = F._from_data(
            f'{self.name}._back_shape',
            np.array([B, A, P, D], dtype=np.int64), is_const=True)
        x = self.reshape_back(x, self._back_shape)

        return x

    def analytical_param_count(self):
        """Parameter count for IntentionInteraction (1 encoder layer)."""
        D = self.embed_dims
        D_ff = self.dim_feedforward
        # MHA: Q,K,V,Out projections each (D² + D)
        mha = 4 * (D * D + D)
        # FFN: fc1 (D*D_ff + D_ff) + fc2 (D_ff*D + D)
        ffn = D * D_ff + D_ff + D_ff * D + D
        return mha + ffn


# ======================================================================
# TrackAgentInteraction
# ======================================================================

class TrackAgentInteraction(SimNN.Module):
    """
    Interaction between tracked agents.

    Wraps a TransformerDecoderLayer (batch_first=True, post-norm):
        self_attn -> Add & Norm1 -> cross_attn -> Add & Norm2 -> FFN -> Add & Norm3

    Positional encodings are added to query and key *before* the decoder layer,
    matching the original PyTorch implementation.

    Input query (B, A, P, D) is flattened to (B*A, P, D).
    Input key   (B, A, D)    is tiled to     (B*A, A, D)  [requires B=1].
    Each agent's P modes attend to all A agents via cross-attention.

    PyTorch equivalent:
        nn.TransformerDecoderLayer(d_model=256, nhead=8, dropout=0.1,
                                  dim_feedforward=512, batch_first=True)

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension.  Default: 256.
        num_heads (int): Number of attention heads.  Default: 8.
        dim_feedforward (int): FFN hidden dimension.  Default: 512.
    """

    def __init__(self, name, embed_dims=256, num_heads=8, dim_feedforward=512):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.dim_feedforward = dim_feedforward

        # Positional embedding additions
        self.add_pos_q = F.Add(f'{name}.add_pos_q')
        self.add_pos_k = F.Add(f'{name}.add_pos_k')

        # Self-attention
        self.self_attn = MultiheadAttention(
            f'{name}.self_attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=True, bias=True)

        # Cross-attention
        self.cross_attn = MultiheadAttention(
            f'{name}.cross_attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=True, bias=True)

        # FFN
        self.ffn_fc1 = SimNN.Linear(f'{name}.ffn_fc1',
                                    in_features=embed_dims,
                                    out_features=dim_feedforward)
        self.ffn_fc2 = SimNN.Linear(f'{name}.ffn_fc2',
                                    in_features=dim_feedforward,
                                    out_features=embed_dims)
        self.ffn_relu = F.Relu(f'{name}.ffn_relu')

        # Layer norms (post-norm)
        self.norm1 = LayerNorm(f'{name}.norm1', embed_dims)
        self.norm2 = LayerNorm(f'{name}.norm2', embed_dims)
        self.norm3 = LayerNorm(f'{name}.norm3', embed_dims)

        # Residual adds
        self.add1 = F.Add(f'{name}.add1')
        self.add2 = F.Add(f'{name}.add2')
        self.add3 = F.Add(f'{name}.add3')

        # Reshape / tile ops
        self.reshape_flat = F.Reshape(f'{name}.reshape_flat')
        self.reshape_back = F.Reshape(f'{name}.reshape_back')
        self.tile_mem = F.Tile(f'{name}.tile_mem')

        super().link_op2module()

    def __call__(self, query, key, query_pos=None, key_pos=None):
        """
        Forward pass.

        Args:
            query:     SimTensor (B, A, P, D) — per-agent mode query.
            key:       SimTensor (B, A_k, D)  — agent / map key embeddings.
            query_pos: SimTensor (B, A, P, D) — optional positional encoding for query.
            key_pos:   SimTensor (B, A_k, D)  — optional positional encoding for key.

        Returns:
            SimTensor (B, A, P, D) — updated query after agent interaction.
        """
        B, A, P, D = query.shape

        # Add positional encodings
        if query_pos is not None:
            query = self.add_pos_q(query, query_pos)
        if key_pos is not None:
            key = self.add_pos_k(key, key_pos)

        # Tile key: (B, A_k, D) → (B*A, A_k, D)  [valid for B=1 inference]
        self._tile_repeats = F._from_data(
            f'{self.name}._tile_repeats',
            np.array([B * A, 1, 1], dtype=np.int64), is_const=True)
        mem = self.tile_mem(key, self._tile_repeats)

        # Flatten query: (B, A, P, D) → (B*A, P, D)
        self._flat_shape = F._from_data(
            f'{self.name}._flat_shape',
            np.array([B * A, P, D], dtype=np.int64), is_const=True)
        tgt = self.reshape_flat(query, self._flat_shape)

        # ---------- TransformerDecoderLayer (post-norm) ----------
        # Self-attention
        sa_out = self.self_attn(tgt)
        tgt = self.add1(tgt, sa_out)
        tgt = self.norm1(tgt)

        # Cross-attention
        ca_out = self.cross_attn(tgt, key=mem, value=mem)
        tgt = self.add2(tgt, ca_out)
        tgt = self.norm2(tgt)

        # FFN
        ffn = self.ffn_fc1(tgt)
        ffn = self.ffn_relu(ffn)
        ffn = self.ffn_fc2(ffn)
        tgt = self.add3(tgt, ffn)
        tgt = self.norm3(tgt)

        # Unflatten: (B*A, P, D) → (B, A, P, D)
        self._back_shape = F._from_data(
            f'{self.name}._back_shape',
            np.array([B, A, P, D], dtype=np.int64), is_const=True)
        tgt = self.reshape_back(tgt, self._back_shape)

        return tgt

    def analytical_param_count(self):
        """Parameter count for TrackAgentInteraction (1 decoder layer)."""
        D = self.embed_dims
        D_ff = self.dim_feedforward
        # 2 MHAs × [Q,K,V,Out projections each (D² + D)]
        mha = 2 * 4 * (D * D + D)
        # FFN: fc1 (D*D_ff + D_ff) + fc2 (D_ff*D + D)
        ffn = D * D_ff + D_ff + D_ff * D + D
        return mha + ffn


# ======================================================================
# MapInteraction
# ======================================================================

class MapInteraction(SimNN.Module):
    """
    Interaction between agents and map lanes.

    Architecture identical to TrackAgentInteraction — wraps a
    TransformerDecoderLayer (batch_first=True, post-norm).

    Input query (B, A, P, D) is flattened to (B*A, P, D).
    Input key   (B, M, D)    is tiled to     (B*A, M, D)  [requires B=1].

    PyTorch equivalent:
        nn.TransformerDecoderLayer(d_model=256, nhead=8, dropout=0.1,
                                  dim_feedforward=512, batch_first=True)

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension.  Default: 256.
        num_heads (int): Number of attention heads.  Default: 8.
        dim_feedforward (int): FFN hidden dimension.  Default: 512.
    """

    def __init__(self, name, embed_dims=256, num_heads=8, dim_feedforward=512):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.dim_feedforward = dim_feedforward

        # Positional embedding additions
        self.add_pos_q = F.Add(f'{name}.add_pos_q')
        self.add_pos_k = F.Add(f'{name}.add_pos_k')

        # Self-attention
        self.self_attn = MultiheadAttention(
            f'{name}.self_attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=True, bias=True)

        # Cross-attention
        self.cross_attn = MultiheadAttention(
            f'{name}.cross_attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=True, bias=True)

        # FFN
        self.ffn_fc1 = SimNN.Linear(f'{name}.ffn_fc1',
                                    in_features=embed_dims,
                                    out_features=dim_feedforward)
        self.ffn_fc2 = SimNN.Linear(f'{name}.ffn_fc2',
                                    in_features=dim_feedforward,
                                    out_features=embed_dims)
        self.ffn_relu = F.Relu(f'{name}.ffn_relu')

        # Layer norms (post-norm)
        self.norm1 = LayerNorm(f'{name}.norm1', embed_dims)
        self.norm2 = LayerNorm(f'{name}.norm2', embed_dims)
        self.norm3 = LayerNorm(f'{name}.norm3', embed_dims)

        # Residual adds
        self.add1 = F.Add(f'{name}.add1')
        self.add2 = F.Add(f'{name}.add2')
        self.add3 = F.Add(f'{name}.add3')

        # Reshape / tile ops
        self.reshape_flat = F.Reshape(f'{name}.reshape_flat')
        self.reshape_back = F.Reshape(f'{name}.reshape_back')
        self.tile_mem = F.Tile(f'{name}.tile_mem')

        super().link_op2module()

    def __call__(self, query, key, query_pos=None, key_pos=None):
        """
        Forward pass.

        Args:
            query:     SimTensor (B, A, P, D) — per-agent mode query.
            key:       SimTensor (B, M, D)    — map lane key embeddings.
            query_pos: SimTensor (B, A, P, D) — optional positional encoding for query.
            key_pos:   SimTensor (B, M, D)    — optional positional encoding for key.

        Returns:
            SimTensor (B, A, P, D) — updated query after map interaction.
        """
        B, A, P, D = query.shape

        # Add positional encodings
        if query_pos is not None:
            query = self.add_pos_q(query, query_pos)
        if key_pos is not None:
            key = self.add_pos_k(key, key_pos)

        # Flatten query: (B, A, P, D) → (B*A, P, D)
        self._flat_shape = F._from_data(
            f'{self.name}._flat_shape',
            np.array([B * A, P, D], dtype=np.int64), is_const=True)
        tgt = self.reshape_flat(query, self._flat_shape)

        # Tile key: (B, M, D) → (B*A, M, D)  [valid for B=1 inference]
        self._tile_repeats = F._from_data(
            f'{self.name}._tile_repeats',
            np.array([B * A, 1, 1], dtype=np.int64), is_const=True)
        mem = self.tile_mem(key, self._tile_repeats)

        # ---------- TransformerDecoderLayer (post-norm) ----------
        # Self-attention
        sa_out = self.self_attn(tgt)
        tgt = self.add1(tgt, sa_out)
        tgt = self.norm1(tgt)

        # Cross-attention
        ca_out = self.cross_attn(tgt, key=mem, value=mem)
        tgt = self.add2(tgt, ca_out)
        tgt = self.norm2(tgt)

        # FFN
        ffn = self.ffn_fc1(tgt)
        ffn = self.ffn_relu(ffn)
        ffn = self.ffn_fc2(ffn)
        tgt = self.add3(tgt, ffn)
        tgt = self.norm3(tgt)

        # Unflatten: (B*A, P, D) → (B, A, P, D)
        self._back_shape = F._from_data(
            f'{self.name}._back_shape',
            np.array([B, A, P, D], dtype=np.int64), is_const=True)
        tgt = self.reshape_back(tgt, self._back_shape)

        return tgt

    def analytical_param_count(self):
        """Parameter count for MapInteraction (1 decoder layer)."""
        D = self.embed_dims
        D_ff = self.dim_feedforward
        mha = 2 * 4 * (D * D + D)
        ffn = D * D_ff + D_ff + D_ff * D + D
        return mha + ffn


# ======================================================================
# MotionTransformerDecoder
# ======================================================================

class MotionTransformerDecoder(SimNN.Module):
    """
    Main motion transformer decoder for trajectory prediction.

    Composes intention, track-agent, map, and BEV interaction layers
    across multiple decoder iterations, with MLP fusers for feature aggregation.

    Layer structure per iteration:
        1. dynamic_embed_fuser   — fuse agent/offset/ego embeddings
        2. in_query_fuser        — fuse query + dynamic embedding
        3. TrackAgentInteraction  — agent-agent cross-attention
        4. MapInteraction         — agent-map  cross-attention
        5. BEV interaction        — deformable attention on BEV features
        6. out_query_fuser        — fuse outputs from 3, 4, 5

    After all iterations a refinement layer (rl) runs once.

    Note: bev_interaction_layers require MotionTransformerAttentionLayer
    which will be converted in motion_deformable_attn.py.  Pass them as
    pre-built layers or leave as None.

    Args:
        name (str): Module name.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        embed_dims (int): Embedding dimension.  Default: 256.
        num_layers (int): Number of decoder iterations.  Default: 3.
        bev_interaction_layers (list): Pre-built BEV interaction layers (optional).
        bev_interaction_rl (module): Pre-built BEV refinement layer (optional).
    """

    def __init__(self, name, pc_range=None, embed_dims=256, num_layers=3,
                 bev_interaction_layers=None, bev_interaction_rl=None):
        super().__init__()
        self.name = name
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_layers = num_layers
        D = embed_dims

        # ------ Interaction layers ------
        self.intention_interaction_layers = IntentionInteraction(
            f'{name}.intention_interaction')

        track_layers = [TrackAgentInteraction(f'{name}.track_agent_{i}')
                        for i in range(num_layers)]
        self.track_agent_interaction_layers = SimNN.ModuleList(track_layers)

        map_layers = [MapInteraction(f'{name}.map_{i}')
                      for i in range(num_layers)]
        self.map_interaction_layers = SimNN.ModuleList(map_layers)

        # BEV interaction layers (MotionTransformerAttentionLayer — external)
        self.bev_interaction_layers: SimNN.ModuleList | None
        if bev_interaction_layers is not None:
            self.bev_interaction_layers = SimNN.ModuleList(bev_interaction_layers)
        else:
            self.bev_interaction_layers = None

        # ------ Refinement layers ------
        self.track_agent_interaction_rl = TrackAgentInteraction(
            f'{name}.track_agent_rl')
        self.map_interaction_rl = MapInteraction(f'{name}.map_rl')
        self.bev_interaction_rl = bev_interaction_rl  # external, may be None

        # ------ MLP fusers (all TwoLayerMLP: Linear→ReLU→Linear) ------
        self.out_mid_fuser = TwoLayerMLP(
            f'{name}.out_mid_fuser', D * 4, D * 2, D)
        self.traj_embed = TwoLayerMLP(
            f'{name}.traj_embed', 2, D * 2, D)
        self.static_dynamic_rl = TwoLayerMLP(
            f'{name}.static_dynamic_rl', D * 2, D * 2, D)
        self.dynamic_embed_rl = TwoLayerMLP(
            f'{name}.dynamic_embed_rl', D * 3, D * 2, D)
        self.in_query_rl = TwoLayerMLP(
            f'{name}.in_query_rl', D * 2, D * 2, D)
        self.static_dynamic_fuser = TwoLayerMLP(
            f'{name}.static_dynamic_fuser', D * 2, D * 2, D)
        # Per-layer fusers (called once per iteration to avoid name collision)
        dyn_fuser_list = [TwoLayerMLP(
            f'{name}.dynamic_embed_fuser_{i}', D * 3, D * 2, D)
            for i in range(num_layers)]
        self.dynamic_embed_fusers = SimNN.ModuleList(dyn_fuser_list)
        inq_fuser_list = [TwoLayerMLP(
            f'{name}.in_query_fuser_{i}', D * 2, D * 2, D)
            for i in range(num_layers)]
        self.in_query_fusers = SimNN.ModuleList(inq_fuser_list)
        outq_fuser_list = [TwoLayerMLP(
            f'{name}.out_query_fuser_{i}', D * 4, D * 2, D)
            for i in range(num_layers)]
        self.out_query_fusers = SimNN.ModuleList(outq_fuser_list)

        # Per-layer embedding layers (called inside the loop to avoid name collisions
        # with the initial embedding layers called in motion_head.__call__)
        agent_emb_layers = [TwoLayerMLP(
            f'{name}.agent_emb_layer_{i}', D, D * 2, D)
            for i in range(num_layers)]
        self.agent_emb_layers = SimNN.ModuleList(agent_emb_layers)
        ego_emb_layers = [TwoLayerMLP(
            f'{name}.ego_emb_layer_{i}', D, D * 2, D)
            for i in range(num_layers)]
        self.ego_emb_layers = SimNN.ModuleList(ego_emb_layers)
        offset_emb_layers = [TwoLayerMLP(
            f'{name}.offset_emb_layer_{i}', D, D * 2, D)
            for i in range(num_layers)]
        self.offset_emb_layers = SimNN.ModuleList(offset_emb_layers)

        # ------ Concat ops for forward (refinement-only, used once) ------
        self.concat_dynamic_rl = F.ConcatX(
            f'{name}.concat_dynamic_rl', axis=-1)
        self.concat_in_query_rl = F.ConcatX(
            f'{name}.concat_in_query_rl', axis=-1)
        self.concat_out_mid = F.ConcatX(
            f'{name}.concat_out_mid', axis=-1)

        # ------ Ops for TTSim graph coverage ------
        # Broadcast track_query (B,A,D) → (B,A,P,D)
        self.unsq_tq = F.Unsqueeze(f'{name}.unsq_tq')
        self.tile_tq = F.Tile(f'{name}.tile_tq')
        self.unsq_tqp = F.Unsqueeze(f'{name}.unsq_tqp')
        self.tile_tqp = F.Tile(f'{name}.tile_tqp')

        # Refinement-only reshape (used once, outside loop)
        self.ref_flat_reshape = F.Reshape(f'{name}.ref_flat_reshape')

        super().link_op2module()

    def _norm_points_graph(self, pos_t, tag):
        """Normalize 2D points to [0,1] using pc_range. Returns SimTensor (*, 2).

        pos_t: SimTensor whose last dim is 2 (x, y).
        Uses Sub + Div ops so the computation registers in the graph.
        """
        n = self.name
        r = self._r
        pc = self.pc_range
        # Slice x and y: pos_t[..., 0:1] and pos_t[..., 1:2]
        shape = list(pos_t.shape)
        # Create constant range tensors — broadcastable scalars
        x_min = r(F._from_data(f'{n}.pc_xmin_{tag}',
                               np.array([pc[0]], dtype=np.float32), is_const=True))
        y_min = r(F._from_data(f'{n}.pc_ymin_{tag}',
                               np.array([pc[1]], dtype=np.float32), is_const=True))
        x_range = r(F._from_data(f'{n}.pc_xrange_{tag}',
                                 np.array([pc[3] - pc[0]], dtype=np.float32), is_const=True))
        y_range = r(F._from_data(f'{n}.pc_yrange_{tag}',
                                 np.array([pc[4] - pc[1]], dtype=np.float32), is_const=True))

        # Split pos into x, y using Slice on last axis
        starts_0 = r(F._from_data(f'{n}.ns0_{tag}', np.array([0], dtype=np.int64), is_const=True))
        ends_1 = r(F._from_data(f'{n}.ne1_{tag}', np.array([1], dtype=np.int64), is_const=True))
        starts_1 = r(F._from_data(f'{n}.ns1_{tag}', np.array([1], dtype=np.int64), is_const=True))
        ends_2 = r(F._from_data(f'{n}.ne2_{tag}', np.array([2], dtype=np.int64), is_const=True))
        ax_last = r(F._from_data(f'{n}.nax_{tag}',
                                 np.array([len(shape) - 1], dtype=np.int64), is_const=True))

        x_shape = shape[:-1] + [1]
        slice_x = r(F.SliceF(f'{n}.sl_x_{tag}', out_shape=x_shape))
        pos_x = slice_x(pos_t, starts_0, ends_1, ax_last)

        slice_y = r(F.SliceF(f'{n}.sl_y_{tag}', out_shape=x_shape))
        pos_y = slice_y(pos_t, starts_1, ends_2, ax_last)

        # (x - xmin) / xrange, (y - ymin) / yrange
        sub_x = r(F.Sub(f'{n}.norm_sub_x_{tag}'))
        div_x = r(F.Div(f'{n}.norm_div_x_{tag}'))
        sub_y = r(F.Sub(f'{n}.norm_sub_y_{tag}'))
        div_y = r(F.Div(f'{n}.norm_div_y_{tag}'))
        cat = r(F.ConcatX(f'{n}.norm_concat_{tag}', axis=-1))
        nx = sub_x(pos_x, x_min)
        nx = div_x(nx, x_range)
        ny = sub_y(pos_y, y_min)
        ny = div_y(ny, y_range)

        # Concat → (*, 2)
        normed = cat(nx, ny)
        return normed

    def _pos2posemb2d_graph(self, pos_t, tag):
        """Positional embedding via TTSim ops. Returns SimTensor (*, D).

        pos_t: SimTensor (*, 2) — normalized 2D coordinates.
        Applies: pos * 2π / dim_t → sin/cos → interleave → concat(y, x).
        """
        n = self.name
        r = self._r
        num_pos_feats = self.embed_dims // 2

        # Scale constant: 2*pi
        scale_c = r(F._from_data(f'{n}.pe_scale_{tag}',
                                 np.array([2.0 * np.pi], dtype=np.float32), is_const=True))
        pe_mul = r(F.Mul(f'{n}.posemb_mul_scale_{tag}'))
        pos_scaled = pe_mul(pos_t, scale_c)

        # dim_t: temperature ** (2*(i//2)/num_pos_feats) for i in range(num_pos_feats)
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = 10000.0 ** (2.0 * (dim_t // 2) / num_pos_feats)
        dim_t_c = r(F._from_data(f'{n}.pe_dimt_{tag}', dim_t, is_const=True))

        # Slice pos_scaled into x and y channels
        shape = list(pos_scaled.shape)
        starts_0 = r(F._from_data(f'{n}.ps0_{tag}', np.array([0], dtype=np.int64), is_const=True))
        ends_1 = r(F._from_data(f'{n}.pe1_{tag}', np.array([1], dtype=np.int64), is_const=True))
        starts_1 = r(F._from_data(f'{n}.ps1_{tag}', np.array([1], dtype=np.int64), is_const=True))
        ends_2 = r(F._from_data(f'{n}.pe2_{tag}', np.array([2], dtype=np.int64), is_const=True))
        ax_last = r(F._from_data(f'{n}.pax_{tag}',
                                 np.array([len(shape) - 1], dtype=np.int64), is_const=True))

        x_shape = shape[:-1] + [1]
        sl_x = r(F.SliceF(f'{n}.psl_x_{tag}', out_shape=x_shape))
        px = sl_x(pos_scaled, starts_0, ends_1, ax_last)  # (*, 1)

        sl_y = r(F.SliceF(f'{n}.psl_y_{tag}', out_shape=x_shape))
        py = sl_y(pos_scaled, starts_1, ends_2, ax_last)  # (*, 1)

        # Divide by dim_t: broadcast (*, 1) / (num_pos_feats,) → (*, num_pos_feats)
        pe_div_x = r(F.Div(f'{n}.posemb_div_x_{tag}'))
        pe_div_y = r(F.Div(f'{n}.posemb_div_y_{tag}'))
        px_d = pe_div_x(px, dim_t_c)  # (*, num_pos_feats)
        py_d = pe_div_y(py, dim_t_c)

        # Sin on x, Cos on y — total 256 trig ops matching PyTorch (128 sin + 128 cos per axis)
        pe_sin_x = r(F.Sin(f'{n}.posemb_sin_x_{tag}'))
        pe_cos_y = r(F.Cos(f'{n}.posemb_cos_y_{tag}'))
        px_emb = pe_sin_x(px_d)  # (*, num_pos_feats)
        py_emb = pe_cos_y(py_d)  # (*, num_pos_feats)

        # Concat → (*, 2 * num_pos_feats) = (*, D)
        pe_cat = r(F.ConcatX(f'{n}.posemb_cat_xy_{tag}', axis=-1))
        result = pe_cat(py_emb, px_emb)
        return result

    def _sim_data(self, t, dtype=np.float32):
        """Extract numpy data from a SimTensor, falling back to zeros if None."""
        if hasattr(t, 'op_in'):
            return t.data if t.data is not None else np.zeros(t.shape, dtype=dtype)
        return t

    def _r(self, obj):
        """Register a dynamically-created SimTensor or op into the graph.

        Uses self._tensors / self._op_hndls directly (Pattern 2,
        same as LeViT / Yolo_v7 / UNet workloads).
        For ops, also sets link_module so output tensors auto-register.
        """
        from ttsim.ops import SimTensor
        if isinstance(obj, SimTensor):
            self._tensors[obj.name] = obj
        elif isinstance(obj, (F.SimOpHandle, F.SplitOpHandle,
                              F.VariadicInputOpHandle,
                              F.MultiOutputSimOpHandle)):
            self._op_hndls[obj.name] = obj
            obj.set_module(self)
        return obj

    def __call__(self,
                 track_query,
                 lane_query,
                 track_query_pos=None,
                 lane_query_pos=None,
                 track_bbox_results=None,
                 bev_embed=None,
                 reference_trajs=None,
                 traj_reg_branches=None,
                 traj_cls_branches=None,
                 traj_refine_branch=None,
                 agent_level_embedding=None,
                 scene_level_ego_embedding=None,
                 scene_level_offset_embedding=None,
                 learnable_embed=None,
                 agent_level_embedding_layer=None,
                 scene_level_ego_embedding_layer=None,
                 scene_level_offset_embedding_layer=None,
                 **kwargs):
        """
        Forward pass for MotionTransformerDecoder.

        Iterates through num_layers of interaction (track-agent, map, BEV),
        then a refinement layer.

        Returns:
            intermediate: list of SimTensors (B, A, P, D) per layer
            intermediate_reference_trajs: list of numpy arrays (B, A, P, T, 2)
            offset: SimTensor from traj_refine_branch
        """
        n = self.name
        r = self._r  # shorthand for register
        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        A = track_query.shape[1]

        # Broadcast track_query: (B, A, D) → unsqueeze(2) → (B, A, 1, D) → tile → (B, A, P, D)
        _unsq_ax2 = r(F._from_data(f'{n}._unsq_ax2',
                                    np.array([2], dtype=np.int64), is_const=True))
        _tile_reps = r(F._from_data(f'{n}._tile_reps',
                                    np.array([1, 1, P, 1], dtype=np.int64), is_const=True))
        track_query_bc = self.unsq_tq(track_query, _unsq_ax2)
        track_query_bc = self.tile_tq(track_query_bc, _tile_reps)

        track_query_pos_bc = self.unsq_tqp(track_query_pos, _unsq_ax2)
        track_query_pos_bc = self.tile_tqp(track_query_pos_bc, _tile_reps)

        # Static intention embedding (immutable across layers)
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)

        # static_intention = agent + offset + learnable
        static_add1 = r(F.Add(f'{n}._static_add1'))
        static_add2 = r(F.Add(f'{n}._static_add2'))
        static_intention = static_add1(agent_level_embedding, scene_level_offset_embedding)
        static_intention = static_add2(static_intention, learnable_embed)

        # reference_trajs: SimTensor (B, A, P, T, 2)
        ref_trajs_t = reference_trajs  # SimTensor, updated each layer
        ref_trajs_np = self._sim_data(reference_trajs)  # numpy fallback for BEV coord transforms

        # query_embed initialized to zeros
        query_embed = r(F._from_data(f'{n}.qe_init',
                                     np.zeros((B, A, P, D), dtype=np.float32)))

        for lid in range(self.num_layers):
            # Dynamic embed: cat(agent, offset, ego) → fuser
            cat_dyn_op = r(F.ConcatX(f'{n}.concat_dynamic_fuser_{lid}', axis=-1))
            cat_dyn = cat_dyn_op(
                agent_level_embedding, scene_level_offset_embedding)
            # ConcatX takes 2 args, so chain for 3
            concat_dyn2 = r(F.ConcatX(f'{n}.cat_dyn2_{lid}', axis=-1))
            cat_dyn = concat_dyn2(cat_dyn, scene_level_ego_embedding)
            dynamic_query_embed = self.dynamic_embed_fusers[lid](cat_dyn)

            # Fuse query_embed + dynamic → in_query
            cat_in_op = r(F.ConcatX(f'{n}.concat_in_query_{lid}', axis=-1))
            cat_in = cat_in_op(query_embed, dynamic_query_embed)
            query_embed = self.in_query_fusers[lid](cat_in)

            # Track-agent interaction
            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

            # Map interaction
            map_query_embed = self.map_interaction_layers[lid](
                query_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

            # BEV interaction
            if self.bev_interaction_layers is not None:
                _unsq_ax4 = r(F._from_data(f'{n}._unsq_ax4_{lid}',
                                           np.array([4], dtype=np.int64), is_const=True))
                unsq_ref = r(F.Unsqueeze(f'{n}.unsq_ref_{lid}'))
                ref_trajs_bev = unsq_ref(ref_trajs_t, _unsq_ax4)
                bev_query_embed = self.bev_interaction_layers[lid](
                    query_embed,
                    value=bev_embed,
                    query_pos=track_query_pos_bc,
                    bbox_results=track_bbox_results,
                    reference_trajs=ref_trajs_bev,
                    **kwargs)
            else:
                bev_query_embed = query_embed

            # Fuse: cat(track, map, bev, track_query_bc+pos_bc) → out_query_fuser
            add_tp = r(F.Add(f'{n}.add_track_pos_{lid}'))
            tq_plus_pos = add_tp(track_query_bc, track_query_pos_bc)

            cat_out1 = r(F.ConcatX(f'{n}.cat_out1_{lid}', axis=-1))
            cat_out2 = r(F.ConcatX(f'{n}.cat_out2_{lid}', axis=-1))
            cat_out3 = r(F.ConcatX(f'{n}.cat_out3_{lid}', axis=-1))
            out_cat = cat_out1(track_query_embed, map_query_embed)
            out_cat = cat_out2(out_cat, bev_query_embed)
            out_cat = cat_out3(out_cat, tq_plus_pos)
            query_embed = self.out_query_fusers[lid](out_cat)

            if traj_reg_branches is not None:
                # Regression: predict trajectory update
                tmp = traj_reg_branches[lid](query_embed)
                # Reshape: (B, A, P, T*5) → (B, A, P, T, 5)
                n_steps = reference_trajs.shape[3]
                _traj_shape = r(F._from_data(f'{n}._traj_shape_{lid}',
                    np.array([B, A, P, n_steps, -1], dtype=np.int64), is_const=True))
                traj_rsh = r(F.Reshape(f'{n}.traj_reshape_{lid}'))
                tmp_r = traj_rsh(tmp, _traj_shape)

                # Slice first 2 channels: [..., :2] → new reference trajectories
                _sl_starts = r(F._from_data(f'{n}._tsl_s_{lid}',
                    np.array([0], dtype=np.int64), is_const=True))
                _sl_ends = r(F._from_data(f'{n}._tsl_e_{lid}',
                    np.array([2], dtype=np.int64), is_const=True))
                _sl_axes = r(F._from_data(f'{n}._tsl_a_{lid}',
                    np.array([4], dtype=np.int64), is_const=True))
                sl_xy = r(F.SliceF(f'{n}.sl_xy_{lid}',
                                   out_shape=[B, A, P, n_steps, 2]))
                ref_trajs_t = sl_xy(tmp_r, _sl_starts, _sl_ends, _sl_axes)

                # Keep numpy copy for BEV ref input (data-level, small)
                ref_trajs_np = self._sim_data(ref_trajs_t)

                # Slice last time step: [..., -1, :] → (B, A, P, 2)
                _ref_s = r(F._from_data(f'{n}._rsl_s_{lid}',
                    np.array([-1], dtype=np.int64), is_const=True))
                _ref_e_val = n_steps  # end = shape[3]
                _ref_e = r(F._from_data(f'{n}._rsl_e_{lid}',
                    np.array([_ref_e_val], dtype=np.int64), is_const=True))
                _ref_a = r(F._from_data(f'{n}._rsl_a_{lid}',
                    np.array([3], dtype=np.int64), is_const=True))
                sl_last = r(F.SliceF(f'{n}.sl_last_{lid}',
                                     out_shape=[B, A, P, 1, 2]))
                ref_last = sl_last(ref_trajs_t, _ref_s, _ref_e, _ref_a)
                _rfl_shape = r(F._from_data(f'{n}._rfl_shape_{lid}',
                    np.array([B, A, P, 2], dtype=np.int64), is_const=True))
                ref_rsh = r(F.Reshape(f'{n}.ref_reshape_{lid}'))
                ref_2d = ref_rsh(ref_last, _rfl_shape)  # (B, A, P, 2)

                # Update embeddings using graph ops: norm → posemb → embedding_layer
                offset_normed = self._norm_points_graph(ref_2d, f'off_{lid}')
                offset_pe = self._pos2posemb2d_graph(offset_normed, f'off_{lid}')
                scene_level_offset_embedding = self.offset_emb_layers[lid](offset_pe)

                agent_normed = self._norm_points_graph(ref_2d, f'agt_{lid}')
                agent_pe = self._pos2posemb2d_graph(agent_normed, f'agt_{lid}')
                agent_level_embedding = self.agent_emb_layers[lid](agent_pe)

                ego_normed = self._norm_points_graph(ref_2d, f'ego_{lid}')
                ego_pe = self._pos2posemb2d_graph(ego_normed, f'ego_{lid}')
                scene_level_ego_embedding = self.ego_emb_layers[lid](ego_pe)

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(ref_trajs_np.copy())

        # ---- Refinement layer ----
        # traj_embed: last-step ref → MLP → (B, A, P, D)
        # Slice ref_trajs_t last time step → (B, A, P, 1, 2) → reshape (B, A, P, 2)
        n_steps = reference_trajs.shape[3]
        _rl_s = r(F._from_data(f'{n}._rl_s', np.array([-1], dtype=np.int64), is_const=True))
        _rl_e = r(F._from_data(f'{n}._rl_e', np.array([n_steps], dtype=np.int64), is_const=True))
        _rl_a = r(F._from_data(f'{n}._rl_a', np.array([3], dtype=np.int64), is_const=True))
        sl_rl = r(F.SliceF(f'{n}.sl_rl', out_shape=[B, A, P, 1, 2]))
        ref_last_rl = sl_rl(ref_trajs_t, _rl_s, _rl_e, _rl_a)
        _rl_flat = r(F._from_data(f'{n}._rl_flat',
                                  np.array([B, A, P, 2], dtype=np.int64), is_const=True))
        ref_flat_rl = self.ref_flat_reshape(ref_last_rl, _rl_flat)
        mid_embed_t = self.traj_embed(ref_flat_rl)

        # dynamic_embed_rl
        cat_drl = self.concat_dynamic_rl(agent_level_embedding, scene_level_offset_embedding)
        concat_drl2 = r(F.ConcatX(f'{n}.cat_drl2', axis=-1))
        cat_drl = concat_drl2(cat_drl, scene_level_ego_embedding)
        dynamic_query_embed = self.dynamic_embed_rl(cat_drl)

        # in_query_rl
        cat_irl = self.concat_in_query_rl(mid_embed_t, dynamic_query_embed)
        mid_embed = self.in_query_rl(cat_irl)

        # Refinement interactions
        track_rf_embed = self.track_agent_interaction_rl(
            mid_embed, track_query, query_pos=track_query_pos_bc, key_pos=track_query_pos)

        map_rf_embed = self.map_interaction_rl(
            mid_embed, lane_query, query_pos=track_query_pos_bc, key_pos=lane_query_pos)

        if self.bev_interaction_rl is not None:
            _unsq_ax4 = r(F._from_data(f'{n}._unsq_ax4_rl',
                                       np.array([4], dtype=np.int64), is_const=True))
            unsq_rl = r(F.Unsqueeze(f'{n}.unsq_ref_rl'))
            ref_trajs_rl = unsq_rl(ref_trajs_t, _unsq_ax4)
            bev_rf_embed = self.bev_interaction_rl(
                mid_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=ref_trajs_rl,
                **kwargs)
        else:
            bev_rf_embed = mid_embed

        tq_plus_pos_rl = r(F.Add(f'{n}.add_tqp_rl'))
        tq_pp = tq_plus_pos_rl(track_query_bc, track_query_pos_bc)

        cat_mid1 = r(F.ConcatX(f'{n}.cat_mid1', axis=-1))
        cat_mid2 = r(F.ConcatX(f'{n}.cat_mid2', axis=-1))
        cat_mid3 = r(F.ConcatX(f'{n}.cat_mid3', axis=-1))
        mid_cat = cat_mid1(track_rf_embed, map_rf_embed)
        mid_cat = cat_mid2(mid_cat, bev_rf_embed)
        mid_cat = cat_mid3(mid_cat, tq_pp)
        mid_query_embed = self.out_mid_fuser(mid_cat)

        # Offset prediction
        offset = traj_refine_branch(mid_query_embed)

        return intermediate, intermediate_reference_trajs, offset

    def analytical_param_count(self):
        """
        Parameter count for MotionTransformerDecoder (excluding BEV layers).
        """
        total = 0
        D = self.embed_dims

        # IntentionInteraction (1×)
        total += self.intention_interaction_layers.analytical_param_count()

        # TrackAgentInteraction (num_layers + 1 refine)
        for layer in self.track_agent_interaction_layers:
            total += layer.analytical_param_count()  # type: ignore[attr-defined]
        total += self.track_agent_interaction_rl.analytical_param_count()

        # MapInteraction (num_layers + 1 refine)
        for layer in self.map_interaction_layers:
            total += layer.analytical_param_count()  # type: ignore[attr-defined]
        total += self.map_interaction_rl.analytical_param_count()

        # MLP fusers (single-use MLPs + per-layer MLPs)
        for mlp in [self.out_mid_fuser, self.traj_embed,
                    self.static_dynamic_rl, self.dynamic_embed_rl,
                    self.in_query_rl, self.static_dynamic_fuser]:
            total += mlp.analytical_param_count()
        for ml in [self.dynamic_embed_fusers, self.in_query_fusers,
                   self.out_query_fusers, self.agent_emb_layers,
                   self.ego_emb_layers, self.offset_emb_layers]:
            for m in ml:
                total += m.analytical_param_count()  # type: ignore[attr-defined]

        return total


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    logger.info("Motion Transformer Modules — TTSim (FusionAD)")
    logger.info("=" * 70)

    D = 256
    ok = True

    # --- IntentionInteraction ---
    try:
        ii = IntentionInteraction('test_ii', embed_dims=D)
        q = F._from_data('q', np.random.randn(1, 8, 6, D).astype(np.float32))
        out = ii(q)
        assert list(out.shape) == [1, 8, 6, D], f"Bad shape: {out.shape}"
        logger.debug(
            f"[OK] IntentionInteraction  shape={out.shape}  "
            f"params={ii.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X]  IntentionInteraction  FAILED: {e}")
        ok = False

    # --- TrackAgentInteraction ---
    try:
        tai = TrackAgentInteraction('test_tai', embed_dims=D)
        q = F._from_data('q2', np.random.randn(1, 8, 6, D).astype(np.float32))
        k = F._from_data('k2', np.random.randn(1, 8, D).astype(np.float32))
        qp = F._from_data('qp2', np.random.randn(1, 8, 6, D).astype(np.float32))
        kp = F._from_data('kp2', np.random.randn(1, 8, D).astype(np.float32))
        out = tai(q, k, query_pos=qp, key_pos=kp)
        assert list(out.shape) == [1, 8, 6, D], f"Bad shape: {out.shape}"
        logger.debug(
            f"[OK] TrackAgentInteraction shape={out.shape}  "
            f"params={tai.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X]  TrackAgentInteraction FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False

    # --- MapInteraction ---
    try:
        mi = MapInteraction('test_mi', embed_dims=D)
        q = F._from_data('q3', np.random.randn(1, 8, 6, D).astype(np.float32))
        k = F._from_data('k3', np.random.randn(1, 50, D).astype(np.float32))
        out = mi(q, k)
        assert list(out.shape) == [1, 8, 6, D], f"Bad shape: {out.shape}"
        logger.debug(
            f"[OK] MapInteraction        shape={out.shape}  "
            f"params={mi.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X]  MapInteraction        FAILED: {e}")
        ok = False

    # --- MotionTransformerDecoder (construction only) ---
    try:
        mtd = MotionTransformerDecoder('test_mtd', embed_dims=D, num_layers=3)
        p = mtd.analytical_param_count()
        logger.debug(f"[OK] MotionTransformerDecoder  params={p:,}  (excl. BEV layers)")
    except Exception as e:
        logger.debug(f"[X]  MotionTransformerDecoder  FAILED: {e}")
        ok = False

    logger.info("=" * 70)
    logger.info("ALL OK" if ok else "SOME FAILURES")
