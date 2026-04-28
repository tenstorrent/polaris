#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of PlanningHeadSingleMode for FusionAD.

Inference-only conversion of the single-mode planning head.  Training-specific
methods (forward_train, loss) and collision optimization (CasADi) are omitted.

Classes:
  - PlanningDecoderLayer   : Standard post-norm transformer decoder layer.
  - PlanningDecoder        : Stack of N PlanningDecoderLayers.
  - MLPFuser               : Linear -> LayerNorm -> ReLU fusion module.
  - PlanMLP                : 3-layer MLP for plan-info encoding (37 -> 256).
  - PlanRegBranch          : 2-layer MLP for trajectory regression.
  - PlanningHeadSingleMode : Main planning head module.
"""

# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================

# import torch
# import torch.nn as nn
# from mmdet.models.builder import HEADS, build_loss
# from einops import rearrange
# from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
# from .planning_head_plugin import CollisionNonlinearOptimizer
# import numpy as np
# import copy
#
# @HEADS.register_module()
# class PlanningHeadSingleMode(nn.Module):
#     def __init__(self,
#                  bev_h=200,
#                  bev_w=200,
#                  embed_dims=256,
#                  planning_steps=6,
#                  loss_planning=None,
#                  loss_collision=None,
#                  planning_eval=False,
#                  use_col_optim=False,
#                  col_optim_args=dict(
#                     occ_filter_range=5.0,
#                     sigma=1.0,
#                     alpha_collision=5.0,
#                  ),
#                  with_adapter=False,
#                 ):
#         """
#         Single Mode Planning Head for Autonomous Driving.
#
#         Args:
#             embed_dims (int): Embedding dimensions. Default: 256.
#             planning_steps (int): Number of steps for motion planning. Default: 6.
#             loss_planning (dict): Configuration for planning loss. Default: None.
#             loss_collision (dict): Configuration for collision loss. Default: None.
#             planning_eval (bool): Whether to use planning for evaluation. Default: False.
#             use_col_optim (bool): Whether to use collision optimization. Default: False.
#             col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
#         """
#         super(PlanningHeadSingleMode, self).__init__()
#
#         # Nuscenes
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.navi_embed = nn.Embedding(3, embed_dims)
#         self.reg_branch = nn.Sequential(
#             nn.Linear(embed_dims * 2, embed_dims),
#             nn.ReLU(),
#             nn.Linear(embed_dims, planning_steps * 2),
#         )
#         self.plan_head=nn.Sequential(
#             nn.Linear(37, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512,256)
#         )
#         self.loss_planning = build_loss(loss_planning)
#         self.planning_steps = planning_steps
#         self.planning_eval = planning_eval
#
#         #### planning head
#         fuser_dim = 3
#         attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
#         self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)
#
#         self.mlp_fuser = nn.Sequential(
#                 nn.Linear(embed_dims*fuser_dim, embed_dims),
#                 nn.LayerNorm(embed_dims),
#                 nn.ReLU(inplace=True),
#             )
#
#         self.pos_embed = nn.Embedding(1, embed_dims)
#         self.loss_collision = []
#         for cfg in loss_collision:
#             self.loss_collision.append(build_loss(cfg))
#         self.loss_collision = nn.ModuleList(self.loss_collision)
#
#         self.use_col_optim = use_col_optim
#         self.occ_filter_range = col_optim_args['occ_filter_range']
#         self.sigma = col_optim_args['sigma']
#         self.alpha_collision = col_optim_args['alpha_collision']
#
#         # TODO: reimplement it with down-scaled feature_map
#         self.with_adapter = with_adapter
#         if with_adapter:
#             bev_adapter_block = nn.Sequential(
#                 nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
#             )
#             N_Blocks = 3
#             bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
#             self.bev_adapter = nn.Sequential(*bev_adapter)
#
#     def forward_train(self,
#                       bev_embed,
#                       outs_motion={},
#                       sdc_planning=None,
#                       sdc_planning_mask=None,
#                       ego_info = None,
#                       command=None,
#                       gt_future_boxes=None,
#                       ):
#         """
#         Perform forward planning training with the given inputs.
#         Args:
#             bev_embed (torch.Tensor): The input bird's eye view feature map.
#             outs_motion (dict): A dictionary containing the motion outputs.
#             outs_occflow (dict): A dictionary containing the occupancy flow outputs.
#             sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
#             sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
#             command (torch.Tensor, optional): The driving command issued to the self-driving car.
#             gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
#             img_metas (list[dict], optional): A list of metadata information about the input images.
#
#         Returns:
#             ret_dict (dict): A dictionary containing the losses and planning outputs.
#         """
#         sdc_traj_query = outs_motion['sdc_traj_query']
#         sdc_track_query = outs_motion['sdc_track_query']
#         bev_pos = outs_motion['bev_pos']
#
#         occ_mask = None
#
#         outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, ego_info, sdc_planning['past_planning'], command)
#         loss_inputs = [sdc_planning['planning'], sdc_planning_mask, outs_planning, gt_future_boxes]
#         losses = self.loss(*loss_inputs)
#         ret_dict = dict(losses=losses, outs_motion=outs_planning)
#         return ret_dict
#
#     def forward_test(self, bev_embed, outs_motion={},outs_occflow={}, ego_info = None, past_planning = None, command=None):
#         sdc_traj_query = outs_motion['sdc_traj_query']
#         sdc_track_query = outs_motion['sdc_track_query']
#         bev_pos = outs_motion['bev_pos']
#         occ_mask = outs_occflow['seg_out']
#         outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, ego_info, past_planning, command[0])
#         # outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query, sdc_track_query, ego_info, past_planning, command)
#         return outs_planning
#
#     def forward(self,
#                 bev_embed,
#                 occ_mask,
#                 bev_pos,
#                 sdc_traj_query,
#                 sdc_track_query,
#                 ego_info,
#                 past_planning,
#                 command):
#         """
#         Forward pass for PlanningHeadSingleMode.
#
#         Args:
#             bev_embed (torch.Tensor): Bird's eye view feature embedding.
#             occ_mask (torch.Tensor): Instance mask for occupancy.
#             bev_pos (torch.Tensor): BEV position.
#             sdc_traj_query (torch.Tensor): SDC trajectory query.
#             sdc_track_query (torch.Tensor): SDC track query.
#             command (int): Driving command.
#
#         Returns:
#             dict: A dictionary containing SDC trajectory and all SDC trajectories.
#         """
#         sdc_track_query = sdc_track_query.detach()
#         sdc_traj_query = sdc_traj_query[-1]
#         P = sdc_traj_query.shape[1]
#         sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
#
#         plan_hist = past_planning.view(-1, 1, 18).to(sdc_track_query.device)
#         ego_info = ego_info.view(-1,1,18).to(sdc_track_query.device)
#         comm = command.view(-1,1,1).to(sdc_track_query.device)
#         plan_info = torch.cat([plan_hist, ego_info, comm], dim = -1)
#         plan_info = torch.tensor(plan_info,dtype=torch.float)
#
#         navi_embed = self.navi_embed.weight[command]
#         navi_embed = navi_embed[None].expand(-1,P,-1)
#         plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)
#
#         plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
#         plan_query = rearrange(plan_query, 'b p c -> p b c')
#
#         bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
#         bev_feat = bev_embed +  bev_pos
#
#         ##### Plugin adapter #####
#         if self.with_adapter:
#             bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
#             bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
#             bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')
#         ##########################
#
#         pos_embed = self.pos_embed.weight
#         plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
#
#         # plan_query: [1, 1, 256]
#         # bev_feat: [40000, 1, 256]
#         plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
#         plan_emd = self.plan_head(plan_info)
#         plan_query = torch.cat([plan_query, plan_emd],dim = -1)
#         sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
#         sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=2)
#         sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
#         if self.use_col_optim and not self.training:
#             # post process, only used when testing
#             assert occ_mask is not None
#             sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
#
#         return dict(
#             sdc_traj=sdc_traj_all,
#             sdc_traj_all=sdc_traj_all,
#         )
#
#     def collision_optimization(self, sdc_traj_all, occ_mask):
#         """
#         Optimize SDC trajectory with occupancy instance mask.
#         Args:
#             sdc_traj_all (torch.Tensor): SDC trajectory tensor.
#             occ_mask (torch.Tensor): Occupancy flow instance mask.
#         Returns:
#             torch.Tensor: Optimized SDC trajectory tensor.
#         """
#         pos_xy_t = []
#         valid_occupancy_num = 0
#
#         if occ_mask.shape[2] == 1:
#             occ_mask = occ_mask.squeeze(2)
#         occ_horizon = occ_mask.shape[1]
#         assert occ_horizon == 5
#
#         for t in range(self.planning_steps):
#             cur_t = min(t+1, occ_horizon-1)
#             pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
#             pos_xy = pos_xy[:, [1, 0]]
#             pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
#             pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25
#
#             # filter the occupancy in range
#             keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
#             pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
#             valid_occupancy_num += torch.sum(keep_index>0)
#         if valid_occupancy_num == 0:
#             return sdc_traj_all
#
#         col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
#         col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
#         sol = col_optimizer.solve()
#         sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
#         return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
#
#     def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
#         sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
#         loss_dict = dict()
#         for i in range(len(self.loss_collision)):
#             loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
#             loss_dict[f'loss_collision_{i}'] = loss_collision
#         loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
#         loss_dict.update(dict(loss_ade=loss_ade))
#         return loss_dict


# =============================================================================
# TTsim CODE
# =============================================================================
import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add fusionad directory so "from modules.xxx" imports resolve
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


from ..modules.builder_utils import LayerNorm
from ..modules.multihead_attention import MultiheadAttention


# ======================================================================
# PlanningDecoderLayer
# ======================================================================

class PlanningDecoderLayer(SimNN.Module):
    """
    Standard post-norm transformer decoder layer (inference only).

    Architecture:
        Self-Attention  ->  Add & Norm1
        Cross-Attention ->  Add & Norm2
        FFN (fc1->ReLU->fc2) -> Add & Norm3

    Mirrors PyTorch nn.TransformerDecoderLayer with batch_first=False
    and dropout=0 (inference mode).

    Args:
        name (str): Module name.
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): FFN hidden dimension.
    """

    def __init__(self, name, d_model, nhead, dim_feedforward):
        super().__init__()
        self.name = name
        self.d_model = d_model

        # Self-attention
        self.self_attn = MultiheadAttention(
            f'{name}.self_attn', embed_dims=d_model, num_heads=nhead,
            batch_first=False, bias=True)

        # Cross-attention
        self.cross_attn = MultiheadAttention(
            f'{name}.cross_attn', embed_dims=d_model, num_heads=nhead,
            batch_first=False, bias=True)

        # FFN
        self.ffn_fc1 = SimNN.Linear(f'{name}.ffn_fc1',
                              in_features=d_model, out_features=dim_feedforward)
        self.ffn_fc2 = SimNN.Linear(f'{name}.ffn_fc2',
                              in_features=dim_feedforward, out_features=d_model)
        self.ffn_relu = F.Relu(f'{name}.ffn_relu')

        # Layer norms (post-norm)
        self.norm1 = LayerNorm(f'{name}.norm1', d_model)
        self.norm2 = LayerNorm(f'{name}.norm2', d_model)
        self.norm3 = LayerNorm(f'{name}.norm3', d_model)

        # Residual adds
        self.add1 = F.Add(f'{name}.add1')
        self.add2 = F.Add(f'{name}.add2')
        self.add3 = F.Add(f'{name}.add3')

        super().link_op2module()

    def __call__(self, tgt, memory):
        """
        Forward pass.

        Args:
            tgt: SimTensor [seq_tgt, bs, d_model] — target (plan query).
            memory: SimTensor [seq_mem, bs, d_model] — encoder output (BEV features).

        Returns:
            SimTensor [seq_tgt, bs, d_model]
        """
        # Self-attention: tgt attends to itself
        sa_out = self.self_attn(tgt)
        tgt = self.add1(tgt, sa_out)
        tgt = self.norm1(tgt)

        # Cross-attention: tgt attends to memory
        ca_out = self.cross_attn(tgt, key=memory, value=memory)
        tgt = self.add2(tgt, ca_out)
        tgt = self.norm2(tgt)

        # FFN
        ffn_out = self.ffn_fc1(tgt)
        ffn_out = self.ffn_relu(ffn_out)
        ffn_out = self.ffn_fc2(ffn_out)
        tgt = self.add3(tgt, ffn_out)
        tgt = self.norm3(tgt)

        return tgt

    def analytical_param_count(self):
        """Parameter count for one decoder layer."""
        d = self.d_model
        # Each MHA: 4 projections × (d² + d)
        mha_params = 4 * (d * d + d)
        # FFN: fc1 (d × dim_ff + dim_ff) + fc2 (dim_ff × d + d)
        dim_ff = self.ffn_fc1.out_features
        ffn_params = d * dim_ff + dim_ff + dim_ff * d + d
        # LN: no affine params in TTSim
        return 2 * mha_params + ffn_params


# ======================================================================
# PlanningDecoder
# ======================================================================

class PlanningDecoder(SimNN.Module):
    """
    Stack of PlanningDecoderLayers.

    Mirrors PyTorch nn.TransformerDecoder (no final norm).

    Args:
        name (str): Module name.
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): FFN hidden dimension.
        num_layers (int): Number of decoder layers.
    """

    def __init__(self, name, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.name = name
        self.num_layers = num_layers

        layers = []
        for i in range(num_layers):
            layers.append(PlanningDecoderLayer(
                f'{name}.layer_{i}', d_model, nhead, dim_feedforward))
        self.layers = SimNN.ModuleList(layers)

        super().link_op2module()

    def __call__(self, tgt, memory):
        """
        Forward pass through all decoder layers.

        Args:
            tgt: SimTensor [seq_tgt, bs, d_model]
            memory: SimTensor [seq_mem, bs, d_model]

        Returns:
            SimTensor [seq_tgt, bs, d_model]
        """
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return output

    def analytical_param_count(self):
        total = 0
        for layer in self.layers:
            total += layer.analytical_param_count()  # type: ignore[attr-defined]
        return total


# ======================================================================
# MLPFuser
# ======================================================================

class MLPFuser(SimNN.Module):
    """
    Fusion MLP: Linear -> LayerNorm -> ReLU.

    Fuses concatenated (sdc_traj_query, sdc_track_query, navi_embed)
    into a single embedding.

    Args:
        name (str): Module name.
        in_features (int): Input dimension (typically embed_dims * 3).
        out_features (int): Output dimension (typically embed_dims).
    """

    def __init__(self, name, in_features, out_features):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features

        self.fc = SimNN.Linear(f'{name}.fc', in_features, out_features)
        self.norm = LayerNorm(f'{name}.norm', out_features)
        self.relu = F.Relu(f'{name}.relu')

        super().link_op2module()

    def __call__(self, x):
        """Forward: Linear -> LayerNorm -> ReLU."""
        return self.relu(self.norm(self.fc(x)))

    def analytical_param_count(self):
        # Linear weight + bias; LN has no affine in TTSim
        return self.in_features * self.out_features + self.out_features


# ======================================================================
# PlanMLP
# ======================================================================

class PlanMLP(SimNN.Module):
    """
    Plan-info encoder MLP.

    Architecture:
        Linear(37, 512) -> ReLU ->
        Linear(512, 512) -> ReLU ->
        Linear(512, embed_dims)

    Encodes concatenated (past_planning[18], ego_info[18], command[1]) = 37
    into embed_dims features.

    Args:
        name (str): Module name.
        embed_dims (int): Output dimension. Default 256.
    """

    def __init__(self, name, embed_dims=256):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims

        self.fc0 = SimNN.Linear(f'{name}.fc0', in_features=37, out_features=512)
        self.relu0 = F.Relu(f'{name}.relu0')
        self.fc1 = SimNN.Linear(f'{name}.fc1', in_features=512, out_features=512)
        self.relu1 = F.Relu(f'{name}.relu1')
        self.fc2 = SimNN.Linear(f'{name}.fc2', in_features=512, out_features=embed_dims)

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x: SimTensor [..., 37]

        Returns:
            SimTensor [..., embed_dims]
        """
        x = self.relu0(self.fc0(x))
        x = self.relu1(self.fc1(x))
        return self.fc2(x)

    def analytical_param_count(self):
        return (37 * 512 + 512 +
                512 * 512 + 512 +
                512 * self.embed_dims + self.embed_dims)


# ======================================================================
# PlanRegBranch
# ======================================================================

class PlanRegBranch(SimNN.Module):
    """
    Planning regression branch.

    Architecture:
        Linear(embed_dims * 2, embed_dims) -> ReLU ->
        Linear(embed_dims, planning_steps * 2)

    Outputs (x, y) displacements for each planning step.

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension.
        planning_steps (int): Number of future timesteps.
    """

    def __init__(self, name, embed_dims, planning_steps):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self.out_dim = planning_steps * 2

        self.fc0 = SimNN.Linear(f'{name}.fc0',
                          in_features=embed_dims * 2, out_features=embed_dims)
        self.relu = F.Relu(f'{name}.relu')
        self.fc1 = SimNN.Linear(f'{name}.fc1',
                          in_features=embed_dims, out_features=self.out_dim)

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x: SimTensor [..., embed_dims * 2]

        Returns:
            SimTensor [..., planning_steps * 2]
        """
        return self.fc1(self.relu(self.fc0(x)))

    def analytical_param_count(self):
        d = self.embed_dims
        return (d * 2 * d + d +
                d * self.out_dim + self.out_dim)


# ======================================================================
# PlanningHeadSingleMode
# ======================================================================

class PlanningHeadSingleMode(SimNN.Module):
    """
    TTSim implementation of the FusionAD single-mode planning head.

    Takes BEV features and motion-head outputs, produces ego-vehicle
    trajectory predictions.

    Pipeline (inference):
      1. Concatenate sdc_traj_query, sdc_track_query, navigation embedding.
      2. MLPFuser -> max-pool over prediction modes.
      3. Add positional embedding.
      4. Cross-attend plan query to BEV features (PlanningDecoder).
      5. Encode plan_info (past trajectory + ego state + command) via PlanMLP.
      6. Concatenate decoder output with plan_info embedding.
      7. Regress trajectory via PlanRegBranch.
      8. Cumulative sum for absolute positions.

    Omitted (training-only / CPU post-process):
      - Loss computation (loss_planning, loss_collision).
      - Collision optimization (CollisionNonlinearOptimizer, CasADi).
      - BEV adapter (Conv2d blocks, with_adapter=True).

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension. Default 256.
        planning_steps (int): Number of future timesteps. Default 6.
        bev_h (int): BEV grid height. Default 200.
        bev_w (int): BEV grid width. Default 200.
    """

    def __init__(self,
                 name='planning_head',
                 embed_dims=256,
                 planning_steps=6,
                 bev_h=200,
                 bev_w=200):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.planning_steps = planning_steps
        self.bev_h = bev_h
        self.bev_w = bev_w

        # ---- Embedding weights (set externally before inference) ----
        # navi_embed: nn.Embedding(3, embed_dims) — navigation command
        #   command 0 = right, 1 = left, 2 = straight (nuScenes)
        self.navi_embed_weight = None   # shape: [3, embed_dims]

        # pos_embed: nn.Embedding(1, embed_dims) — positional encoding
        self.pos_embed_weight = None    # shape: [1, embed_dims]

        # ---- Sub-modules ----
        # Fuser: concat(sdc_traj, sdc_track, navi) -> embed_dims
        fuser_dim = 3
        self.mlp_fuser = MLPFuser(
            f'{name}.mlp_fuser',
            in_features=embed_dims * fuser_dim,
            out_features=embed_dims)

        # ReduceMax over prediction modes (axis=1, keepdims=True)
        self.reduce_max = F.ReduceMax(
            f'{name}.reduce_max', axes=[1], keepdims=1)

        # Transpose [bs, 1, embed_dims] -> [1, bs, embed_dims]
        self.perm_bf_to_sf = F.Transpose(
            f'{name}.perm_bf_to_sf', perm=[1, 0, 2])

        # Add positional embedding
        self.add_pos = F.Add(f'{name}.add_pos')

        # Add BEV positional encoding to BEV features
        self.add_bev_pos = F.Add(f'{name}.add_bev_pos')

        # Planning decoder (3 layers, 8 heads, dim_ff = embed_dims * 2)
        self.attn_module = PlanningDecoder(
            f'{name}.attn_module',
            d_model=embed_dims,
            nhead=8,
            dim_feedforward=embed_dims * 2,
            num_layers=3)

        # Plan-info MLP: 37 -> 256
        self.plan_head = PlanMLP(f'{name}.plan_head', embed_dims=embed_dims)

        # Concatenate decoder output and plan_info embedding
        self.concat_plan = F.ConcatX(f'{name}.concat_plan', axis=-1)

        # Regression branch: embed_dims*2 -> planning_steps*2
        self.reg_branch = PlanRegBranch(
            f'{name}.reg_branch', embed_dims, planning_steps)

        # BEV Adapter: 3 blocks of Conv2d(D→D//2, 3×3, pad=1) + ReLU + Conv2d(D//2→D, 1×1)
        half_dims = embed_dims // 2
        self.adapter_blocks = []
        for i in range(3):
            c1 = F.Conv2d(f'{name}.adapter_{i}_conv1', embed_dims, half_dims,
                          kernel_size=3, padding=1)
            r = F.Relu(f'{name}.adapter_{i}_relu')
            c2 = F.Conv2d(f'{name}.adapter_{i}_conv2', half_dims, embed_dims,
                          kernel_size=1)
            self.adapter_blocks.append((c1, r, c2))
            setattr(self, f'adapter_{i}_conv1', c1)
            setattr(self, f'adapter_{i}_relu', r)
            setattr(self, f'adapter_{i}_conv2', c2)

        # Reshape helpers for adapter: [H*W, bs, C] <-> [bs, C, H, W]
        self.adapter_perm_in = F.Transpose(
            f'{name}.adapter_perm_in', perm=[1, 2, 0])
        self.adapter_reshape_in = F.Reshape(f'{name}.adapter_reshape_in')
        self.adapter_reshape_out = F.Reshape(f'{name}.adapter_reshape_out')
        self.adapter_perm_out = F.Transpose(
            f'{name}.adapter_perm_out', perm=[2, 0, 1])

        # Shape constants for adapter reshapes
        self.adapter_4d_shape = F._from_data(
            f'{name}.adapter_4d_shape',
            np.array([-1, embed_dims, bev_h, bev_w], dtype=np.int64),
            is_const=True)
        self.adapter_3d_shape = F._from_data(
            f'{name}.adapter_3d_shape',
            np.array([-1, embed_dims, bev_h * bev_w], dtype=np.int64),
            is_const=True)

        # Post-processing graph ops
        self.pp_reshape = F.Reshape(f'{name}.pp_reshape')
        self.pp_cumsum = F.BinaryOperator(f'{name}.pp_cumsum', optype='CumSum')
        self.pp_cumsum_ax = F._from_data(
            f'{name}.pp_cumsum_ax', np.array([1], dtype=np.int64), is_const=True)

        super().link_op2module()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, bev_embed, bev_pos,
                sdc_traj_query, sdc_track_query,
                ego_info, past_planning, command):
        """
        Inference forward pass.

        Args:
            bev_embed: BEV features [H*W, bs, embed_dims] or numpy.
            bev_pos: BEV positional encoding [bs, embed_dims, bev_h, bev_w]
                or numpy.
            sdc_traj_query: Trajectory queries from motion head.
                [num_decoder_layers, bs, P, embed_dims] or numpy.
            sdc_track_query: Track query for SDC [bs, embed_dims] or numpy.
            ego_info: Ego vehicle state [bs, 18] or numpy.
            past_planning: Past planning trajectory [bs, 18] or numpy.
            command: Navigation command (int or [bs] array).

        Returns:
            dict with:
              - 'sdc_traj': Predicted trajectory [bs, planning_steps, 2]
              - 'sdc_traj_all': Same as sdc_traj
        """
        # ---- Extract numpy data from SimTensors ----
        def _np(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            if hasattr(x, 'data'):
                return x.data
            return np.asarray(x, dtype=np.float32)

        bev_embed_np = _np(bev_embed)
        bev_pos_np = _np(bev_pos)
        sdc_traj_q_np = _np(sdc_traj_query)
        sdc_track_q_np = _np(sdc_track_query)
        ego_info_np = _np(ego_info)
        past_plan_np = _np(past_planning)

        # ---- 1. Pre-process inputs (numpy) ----

        # Take last decoder layer for trajectory query
        # sdc_traj_query: [num_layers, bs, P, embed_dims] -> [bs, P, embed_dims]
        sdc_traj_q_np = sdc_traj_q_np[-1]
        P = sdc_traj_q_np.shape[1]

        # Expand track query: [bs, embed_dims] -> [bs, P, embed_dims]
        sdc_track_q_np = np.expand_dims(sdc_track_q_np, axis=1)
        sdc_track_q_np = np.tile(sdc_track_q_np, (1, P, 1))

        # Navigation embedding lookup: weight[command] -> [embed_dims]
        # command is int or [bs]
        cmd = int(command) if np.ndim(command) == 0 else int(command[0])
        assert self.navi_embed_weight is not None
        navi_np = self.navi_embed_weight[cmd]  # [embed_dims]
        # Expand: [embed_dims] -> [1, P, embed_dims]
        navi_np = np.tile(
            navi_np[np.newaxis, np.newaxis, :],
            (sdc_traj_q_np.shape[0], P, 1)).astype(np.float32)

        # Concatenate: [bs, P, embed_dims*3]
        plan_query_np = np.concatenate(
            [sdc_traj_q_np, sdc_track_q_np, navi_np], axis=-1
        ).astype(np.float32)

        # ---- 2. Build plan_info (numpy): [bs, 1, 37] ----
        bs = plan_query_np.shape[0]
        plan_hist = past_plan_np.reshape(bs, 1, 18)
        ego = ego_info_np.reshape(bs, 1, 18)
        cmd_arr = np.full((bs, 1, 1), cmd, dtype=np.float32)
        plan_info_np = np.concatenate(
            [plan_hist, ego, cmd_arr], axis=-1).astype(np.float32)

        # ---- 3. MLPFuser -> ReduceMax (graph) ----
        plan_query_t = F._from_data(
            f'{self.name}.fuser_in', plan_query_np, is_const=False)
        setattr(self, plan_query_t.name, plan_query_t)

        fused = self.mlp_fuser(plan_query_t)          # [bs, P, embed_dims]
        fused_max = self.reduce_max(fused)             # [bs, 1, embed_dims]
        plan_q = self.perm_bf_to_sf(fused_max)         # [1, bs, embed_dims]

        # ---- 4. BEV features + positional encoding (numpy -> graph) ----
        # bev_pos: [bs, C, bev_h, bev_w] -> [H*W, bs, C]
        if bev_pos_np.ndim == 4:
            bev_pos_flat = bev_pos_np.reshape(
                bev_pos_np.shape[0], bev_pos_np.shape[1], -1)  # [bs, C, H*W]
            bev_pos_flat = bev_pos_flat.transpose(2, 0, 1)     # [H*W, bs, C]
        else:
            bev_pos_flat = bev_pos_np

        bev_embed_t = F._from_data(
            f'{self.name}.bev_embed', bev_embed_np.astype(np.float32),
            is_const=False)
        setattr(self, bev_embed_t.name, bev_embed_t)

        # ---- Apply BEV adapter (3 blocks of Conv3×3→ReLU→Conv1×1) ----
        # [H*W, bs, C] -> [bs, C, H*W] -> [bs, C, H, W]
        bev_2d = self.adapter_perm_in(bev_embed_t)
        bev_2d = self.adapter_reshape_in(bev_2d, self.adapter_4d_shape)
        for c1, relu, c2 in self.adapter_blocks:
            bev_2d = c1(bev_2d)
            bev_2d = relu(bev_2d)
            bev_2d = c2(bev_2d)
        # [bs, C, H, W] -> [bs, C, H*W] -> [H*W, bs, C]
        bev_2d = self.adapter_reshape_out(bev_2d, self.adapter_3d_shape)
        bev_embed_t = self.adapter_perm_out(bev_2d)

        bev_pos_t = F._from_data(
            f'{self.name}.bev_pos', bev_pos_flat.astype(np.float32),
            is_const=False)
        setattr(self, bev_pos_t.name, bev_pos_t)

        bev_feat = self.add_bev_pos(bev_embed_t, bev_pos_t)  # [H*W, bs, C]

        # ---- 5. Add positional embedding ----
        # pos_embed_weight: [1, embed_dims] -> [1, 1, embed_dims]
        assert self.pos_embed_weight is not None
        pos_emb_np = self.pos_embed_weight[np.newaxis, :, :].astype(np.float32)
        pos_emb_t = F._from_data(
            f'{self.name}.pos_emb', pos_emb_np, is_const=True)
        setattr(self, pos_emb_t.name, pos_emb_t)

        plan_q = self.add_pos(plan_q, pos_emb_t)  # [1, bs, embed_dims]

        # ---- 6. Planning decoder (cross-attend to BEV) ----
        plan_q = self.attn_module(plan_q, bev_feat)  # [1, bs, embed_dims]

        # ---- 7. Plan-info MLP (graph) ----
        plan_info_t = F._from_data(
            f'{self.name}.plan_info', plan_info_np, is_const=False)
        setattr(self, plan_info_t.name, plan_info_t)

        plan_emd = self.plan_head(plan_info_t)  # [bs, 1, embed_dims]

        # ---- 8. Concatenate and regress ----
        # plan_q: [1, bs, embed_dims], plan_emd: [bs, 1, embed_dims]
        # For bs=1, both are [1, 1, embed_dims]; cat on dim=-1 -> [1, 1, 2*embed_dims]
        plan_cat = self.concat_plan(plan_q, plan_emd)  # [*, *, embed_dims*2]

        sdc_traj_t = self.reg_branch(plan_cat)  # [*, *, planning_steps*2]

        # ---- 9. Post-process (TTSim graph ops) ----
        pp_shape = F._from_data(
            f'{self.name}.pp_shape',
            np.array([-1, self.planning_steps, 2], dtype=np.int64),
            is_const=True)
        setattr(self, pp_shape.name, pp_shape)
        sdc_traj = self.pp_reshape(sdc_traj_t, pp_shape)  # (bs, steps, 2)

        # Integrate per-step (dx, dy) offsets into absolute trajectory positions
        # across the planning horizon. For shape (bs, steps, 2), the temporal
        # axis is dim=1.
        sdc_traj = self.pp_cumsum(self.pp_cumsum_ax, sdc_traj)  # (bs, steps, 2)

        return dict(
            sdc_traj=sdc_traj,
            sdc_traj_all=sdc_traj,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def analytical_param_count(self):
        """Total learnable parameters."""
        total = 0

        # navi_embed: 3 × embed_dims
        total += 3 * self.embed_dims

        # pos_embed: 1 × embed_dims
        total += self.embed_dims

        # mlp_fuser
        total += self.mlp_fuser.analytical_param_count()

        # attn_module (PlanningDecoder)
        total += self.attn_module.analytical_param_count()

        # plan_head
        total += self.plan_head.analytical_param_count()

        # reg_branch
        total += self.reg_branch.analytical_param_count()

        # BEV adapter: 3 blocks × (Conv2d(D,D//2,3×3) + Conv2d(D//2,D,1×1))
        D = self.embed_dims
        half = D // 2
        for _ in range(3):
            total += D * half * 3 * 3 + half   # conv1: weight + bias
            total += half * D * 1 * 1 + D      # conv2: weight + bias

        return total


# ======================================================================
# Quick self-test
# ======================================================================

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("FusionAD PlanningHeadSingleMode TTSim Module")
    logger.info("=" * 70)

    EMBED_DIMS = 256
    PLANNING_STEPS = 6
    BEV_H = 200
    BEV_W = 200

    # --- Helper to init weights for testing ---
    def _init_weights(module):
        """Populate all Linear weights with small random values."""
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if isinstance(obj, SimNN.Linear):
                obj.param.data = np.random.randn(
                    obj.in_features, obj.out_features
                ).astype(np.float32) * 0.02
                if obj.bias is not None:
                    obj.bias.data = np.zeros(
                        obj.out_features, dtype=np.float32)
            elif isinstance(obj, SimNN.ModuleList):
                for sub in obj:
                    _init_weights(sub)
            elif isinstance(obj, SimNN.Module):
                _init_weights(obj)

    # --- Test PlanningDecoderLayer ---
    logger.info("\n--- Test PlanningDecoderLayer ---")
    try:
        layer = PlanningDecoderLayer('test_layer', EMBED_DIMS, 8, EMBED_DIMS * 2)
        logger.debug("  [OK] Constructed")
        logger.debug(f"    params = {layer.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test PlanningDecoder ---
    logger.info("\n--- Test PlanningDecoder ---")
    try:
        dec = PlanningDecoder('test_dec', EMBED_DIMS, 8, EMBED_DIMS * 2, 3)
        logger.debug("  [OK] Constructed (3 layers)")
        logger.debug(f"    params = {dec.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test MLPFuser ---
    logger.info("\n--- Test MLPFuser ---")
    try:
        fuser = MLPFuser('test_fuser', EMBED_DIMS * 3, EMBED_DIMS)
        logger.debug("  [OK] Constructed")
        logger.debug(f"    params = {fuser.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test PlanMLP ---
    logger.info("\n--- Test PlanMLP ---")
    try:
        pmlp = PlanMLP('test_pmlp', EMBED_DIMS)
        logger.debug("  [OK] Constructed")
        logger.debug(f"    params = {pmlp.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test PlanRegBranch ---
    logger.info("\n--- Test PlanRegBranch ---")
    try:
        rb = PlanRegBranch('test_rb', EMBED_DIMS, PLANNING_STEPS)
        logger.debug("  [OK] Constructed")
        logger.debug(f"    params = {rb.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test PlanningHeadSingleMode ---
    logger.info("\n--- Test PlanningHeadSingleMode ---")
    try:
        head = PlanningHeadSingleMode(
            name='test_head',
            embed_dims=EMBED_DIMS,
            planning_steps=PLANNING_STEPS,
            bev_h=BEV_H,
            bev_w=BEV_W)

        # Set embedding weights
        head.navi_embed_weight = np.random.randn(
            3, EMBED_DIMS).astype(np.float32) * 0.02
        head.pos_embed_weight = np.random.randn(
            1, EMBED_DIMS).astype(np.float32) * 0.02

        logger.debug("  [OK] Constructed")
        logger.debug(f"    embed_dims     = {head.embed_dims}")
        logger.debug(f"    planning_steps = {head.planning_steps}")
        logger.debug(f"    bev_h x bev_w  = {head.bev_h} x {head.bev_w}")
        logger.debug(f"    params         = {head.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    # --- Test forward pass with small dims ---
    logger.info("\n--- Test Forward Pass (small dims) ---")
    try:
        SMALL_BEV_H = 10
        SMALL_BEV_W = 10
        SMALL_DIM = 64
        BS = 1
        P = 6  # prediction modes / planning steps

        head_small = PlanningHeadSingleMode(
            name='test_fwd',
            embed_dims=SMALL_DIM,
            planning_steps=PLANNING_STEPS,
            bev_h=SMALL_BEV_H,
            bev_w=SMALL_BEV_W)

        head_small.navi_embed_weight = np.random.randn(
            3, SMALL_DIM).astype(np.float32) * 0.02
        head_small.pos_embed_weight = np.random.randn(
            1, SMALL_DIM).astype(np.float32) * 0.02

        _init_weights(head_small)

        # Build inputs
        HW = SMALL_BEV_H * SMALL_BEV_W
        bev_embed = np.random.randn(HW, BS, SMALL_DIM).astype(np.float32)
        bev_pos = np.random.randn(
            BS, SMALL_DIM, SMALL_BEV_H, SMALL_BEV_W).astype(np.float32)
        sdc_traj_query = np.random.randn(
            3, BS, P, SMALL_DIM).astype(np.float32)  # 3 decoder layers
        sdc_track_query = np.random.randn(
            BS, SMALL_DIM).astype(np.float32)
        ego_info = np.random.randn(BS, 18).astype(np.float32)
        past_planning = np.random.randn(BS, 18).astype(np.float32)
        command = 2  # straight

        out = head_small.forward(
            bev_embed, bev_pos,
            sdc_traj_query, sdc_track_query,
            ego_info, past_planning, command)

        logger.debug("  [OK] Forward pass succeeded")
        logger.debug(f"    sdc_traj shape: {out['sdc_traj'].shape}")
        logger.debug(f"    Expected:       ({BS}, {PLANNING_STEPS}, 2)")
        assert out['sdc_traj'].shape == (BS, PLANNING_STEPS, 2), \
            f"Shape mismatch: {out['sdc_traj'].shape}"
        logger.debug(f"    Values (first step): {out['sdc_traj'][0, 0]}")
    except Exception as e:
        logger.debug(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()

    logger.info("\n" + "=" * 70)
    logger.info("[OK] Self-test complete")
    logger.info("=" * 70)
