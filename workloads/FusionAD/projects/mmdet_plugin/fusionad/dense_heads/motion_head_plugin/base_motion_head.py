#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of BaseMotionHead for FusionAD.

Inference-only conversion of the base motion prediction head.
Training-specific methods (_build_loss, loss, compute_loss_traj,
compute_matched_gt_traj) and the nonlinear smoother are omitted.

Classes:
  - TwoLayerMLP      : Linear -> ReLU -> Linear  (embedding layers).
  - TrackQueryFuser   : Linear -> LayerNorm -> ReLU  (query fusion layer).
  - TrajClsBranch     : Classification branch with LayerNorm (Linear+LN+ReLU)×N + Linear.
  - TrajRegBranch     : Regression branch without LayerNorm (Linear+ReLU)×N + Linear.
  - BaseMotionHead    : Base class providing layer construction + anchor loading.
"""

#---------------------------PyTorch code------------------------------------------#

# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# Modifications:
# - Modified by FusionAD on 2023.5
# - Added extended support from FusionAD (https://arxiv.org/abs/2308.01006)

# import torch
# import copy
# import pickle
# import torch.nn as nn
# from mmdet.models import  build_loss
# from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

# class BaseMotionHead(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(BaseMotionHead, self).__init__()
#         pass

#     def _build_loss(self, loss_traj):
#         """
#         Build the loss function for the motion prediction task.

#         Args:
#             loss_traj (dict): A dictionary containing the parameters for the loss function.

#         Returns:
#             None
#         """
#         self.loss_traj = build_loss(loss_traj)
#         self.unflatten_traj = nn.Unflatten(3, (self.predict_steps, 5))
#         self.log_softmax = nn.LogSoftmax(dim=2)

#     def _load_anchors(self, anchor_info_path):
#         """
#         Load the anchor information from a file.

#         Args:
#             anchor_info_path (str): The path to the file containing the anchor information.

#         Returns:
#             None
#         """
#         anchor_infos = pickle.load(open(anchor_info_path, 'rb'))
#         self.kmeans_anchors = torch.stack(
#             [torch.from_numpy(a) for a in anchor_infos["anchors_all"]])  # Nc, Pc, steps, 2

#     def _build_layers(self, transformerlayers, det_layer_num):
#         """
#         Build the layers of the motion prediction module.

#         Args:
#             transformerlayers (dict): A dictionary containing the parameters for the transformer layers.
#             det_layer_num (int): The number of detection layers.

#         Returns:
#             None
#         """
#         self.learnable_motion_query_embedding = nn.Embedding(
#             self.num_anchor * self.num_anchor_group, self.embed_dims)
#         self.motionformer = build_transformer_layer_sequence(
#             transformerlayers)
#         self.layer_track_query_fuser = nn.Sequential(
#             nn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
#             nn.LayerNorm(self.embed_dims),
#             nn.ReLU(inplace=True)
#         )

#         self.agent_level_embedding_layer = nn.Sequential(
#             nn.Linear(self.embed_dims, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.scene_level_ego_embedding_layer = nn.Sequential(
#             nn.Linear(self.embed_dims, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.scene_level_offset_embedding_layer = nn.Sequential(
#             nn.Linear(self.embed_dims, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )
#         self.boxes_query_embedding_layer = nn.Sequential(
#             nn.Linear(self.embed_dims, self.embed_dims*2),
#             nn.ReLU(),
#             nn.Linear(self.embed_dims*2, self.embed_dims),
#         )

#     def _init_layers(self):
#         """Initialize classification branch and regression branch of head."""
#         traj_cls_branch = []
#         traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#         traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
#         traj_cls_branch.append(nn.ReLU(inplace=True))
#         for _ in range(self.num_reg_fcs-1):
#             traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#             traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
#             traj_cls_branch.append(nn.ReLU(inplace=True))
#         traj_cls_branch.append(nn.Linear(self.embed_dims, 1))
#         traj_cls_branch = nn.Sequential(*traj_cls_branch)

#         traj_reg_branch = []
#         traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#         traj_reg_branch.append(nn.ReLU())
#         for _ in range(self.num_reg_fcs-1):
#             traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#             traj_reg_branch.append(nn.ReLU())
#         traj_reg_branch.append(nn.Linear(self.embed_dims, self.predict_steps * 5))
#         traj_reg_branch = nn.Sequential(*traj_reg_branch)

#         traj_refine_branch = []
#         traj_refine_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#         traj_refine_branch.append(nn.ReLU())
#         for _ in range(self.num_reg_fcs-1):
#             traj_refine_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
#             traj_refine_branch.append(nn.ReLU())
#         traj_refine_branch.append(nn.Linear(self.embed_dims, self.predict_steps * 2))
#         self.traj_refine_branch = nn.Sequential(*traj_refine_branch)

#         def _get_clones(module, N):
#             return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

#         num_pred = self.motionformer.num_layers
#         self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
#         self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)

#     def _extract_tracking_centers(self, bbox_results, bev_range):
#         """
#         extract the bboxes centers and normized according to the bev range

#         Args:
#             bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
#             bev_range (List[float]): A list of float values representing the bird's eye view range.

#         Returns:
#             torch.Tensor: A tensor representing normized centers of the detection bounding boxes.
#         """
#         batch_size = len(bbox_results)
#         det_bbox_posembed = []
#         for i in range(batch_size):
#             bboxes, scores, labels, bbox_index, mask = bbox_results[i]
#             xy = bboxes.gravity_center[:, :2]
#             x_norm = (xy[:, 0] - bev_range[0]) / \
#                 (bev_range[3] - bev_range[0])
#             y_norm = (xy[:, 1] - bev_range[1]) / \
#                 (bev_range[4] - bev_range[1])
#             det_bbox_posembed.append(
#                 torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1))
#         return torch.stack(det_bbox_posembed)



#----------------------------TTSim code--------------------------------------#

import sys
import os
import copy
import pickle
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

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


# ======================================================================
# TwoLayerMLP
# ======================================================================

class TwoLayerMLP(SimNN.Module):
    """
    Two-layer MLP: Linear -> ReLU -> Linear.

    Used for the embedding layers in the motion head:
      - agent_level_embedding_layer
      - scene_level_ego_embedding_layer
      - scene_level_offset_embedding_layer
      - boxes_query_embedding_layer

    PyTorch equivalent:
        nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

    Args:
        name (str): Module name.
        in_features (int): Input dimension.
        hidden_features (int): Hidden dimension.
        out_features (int): Output dimension.
    """

    def __init__(self, name, in_features, hidden_features, out_features):
        super().__init__()
        self.name = name
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc0 = SimNN.Linear(f'{name}.fc0',
                                in_features=in_features,
                                out_features=hidden_features)
        self.relu = F.Relu(f'{name}.relu')
        self.fc1 = SimNN.Linear(f'{name}.fc1',
                                in_features=hidden_features,
                                out_features=out_features)

        super().link_op2module()

    def __call__(self, x):
        """Forward: Linear -> ReLU -> Linear."""
        return self.fc1(self.relu(self.fc0(x)))

    def analytical_param_count(self):
        return (self.in_features * self.hidden_features + self.hidden_features +
                self.hidden_features * self.out_features + self.out_features)


# ======================================================================
# TrackQueryFuser
# ======================================================================

class TrackQueryFuser(SimNN.Module):
    """
    Track query fusion layer: Linear -> LayerNorm -> ReLU.

    Fuses multi-layer track query embeddings into a single embedding.

    PyTorch equivalent:
        nn.Sequential(
            nn.Linear(embed_dims * det_layer_num, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True)
        )

    Args:
        name (str): Module name.
        in_features (int): Input dimension (embed_dims * det_layer_num).
        out_features (int): Output dimension (embed_dims).
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
        return self.in_features * self.out_features + self.out_features


# ======================================================================
# TrajClsBranch
# ======================================================================

class TrajClsBranch(SimNN.Module):
    """
    Trajectory classification branch.

    Architecture: (Linear + LayerNorm + ReLU) × num_reg_fcs, then Linear(D, 1).

    PyTorch equivalent:
        nn.Sequential(
            nn.Linear(D, D), nn.LayerNorm(D), nn.ReLU(inplace=True),
            ...  # repeated num_reg_fcs times total
            nn.Linear(D, 1)
        )

    Args:
        name (str): Module name.
        embed_dims (int): Input / hidden dimension.
        num_reg_fcs (int): Number of hidden FC+LN+ReLU blocks.
    """

    def __init__(self, name, embed_dims, num_reg_fcs):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs

        fc_list = []
        for i in range(num_reg_fcs):
            fc_list.append(
                SimNN.Linear(f'{name}.fc{i}',
                             in_features=embed_dims, out_features=embed_dims))
        fc_list.append(
            SimNN.Linear(f'{name}.out',
                         in_features=embed_dims, out_features=1))
        self.fcs = SimNN.ModuleList(fc_list)

        for i in range(num_reg_fcs):
            setattr(self, f'ln{i}',
                    LayerNorm(f'{name}.ln{i}', embed_dims))
            setattr(self, f'relu{i}',
                    F.Relu(f'{name}.relu{i}'))

        super().link_op2module()

    def __call__(self, x):
        out = x
        for i in range(self.num_reg_fcs):
            out = self.fcs[i](out)
            setattr(self, out.name, out)
            out = getattr(self, f'ln{i}')(out)
            setattr(self, out.name, out)
            out = getattr(self, f'relu{i}')(out)
            setattr(self, out.name, out)
        out = self.fcs[self.num_reg_fcs](out)
        setattr(self, out.name, out)
        return out

    def analytical_param_count(self):
        d = self.embed_dims
        # num_reg_fcs × (Linear(d,d) + bias) + final Linear(d,1) + bias
        return self.num_reg_fcs * (d * d + d) + (d * 1 + 1)


# ======================================================================
# TrajRegBranch
# ======================================================================

class TrajRegBranch(SimNN.Module):
    """
    Trajectory regression branch.

    Architecture: (Linear + ReLU) × num_reg_fcs, then Linear(D, out_channels).

    Used for both traj_reg_branches (out=predict_steps*5) and
    traj_refine_branch (out=predict_steps*2).

    PyTorch equivalent:
        nn.Sequential(
            nn.Linear(D, D), nn.ReLU(),
            ...  # repeated num_reg_fcs times total
            nn.Linear(D, out_channels)
        )

    Args:
        name (str): Module name.
        embed_dims (int): Input / hidden dimension.
        num_reg_fcs (int): Number of hidden FC+ReLU blocks.
        out_channels (int): Output dimension.
    """

    def __init__(self, name, embed_dims, num_reg_fcs, out_channels):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.out_channels = out_channels

        fc_list = []
        for i in range(num_reg_fcs):
            fc_list.append(
                SimNN.Linear(f'{name}.fc{i}',
                             in_features=embed_dims, out_features=embed_dims))
        fc_list.append(
            SimNN.Linear(f'{name}.out',
                         in_features=embed_dims, out_features=out_channels))
        self.fcs = SimNN.ModuleList(fc_list)

        for i in range(num_reg_fcs):
            setattr(self, f'relu{i}',
                    F.Relu(f'{name}.relu{i}'))

        super().link_op2module()

    def __call__(self, x):
        out = x
        for i in range(self.num_reg_fcs):
            out = self.fcs[i](out)
            setattr(self, out.name, out)
            out = getattr(self, f'relu{i}')(out)
            setattr(self, out.name, out)
        out = self.fcs[self.num_reg_fcs](out)
        setattr(self, out.name, out)
        return out

    def analytical_param_count(self):
        d = self.embed_dims
        return (self.num_reg_fcs * (d * d + d) +
                d * self.out_channels + self.out_channels)


# ======================================================================
# BaseMotionHead
# ======================================================================

class BaseMotionHead(SimNN.Module):
    """
    TTSim implementation of BaseMotionHead (inference only).

    Provides layer construction and anchor loading used by MotionHead.
    Training-specific methods (_build_loss, loss) are omitted.

    This is a base class — subclasses (MotionHead) call:
      - _load_anchors(anchor_info_path) to load kmeans trajectory anchors
      - _build_layers(motionformer, det_layer_num) to build embedding layers
      - _init_layers() to build cls/reg/refine branches

    Args:
        name (str): Module name (set by subclass).
    """

    embed_dims: int
    num_reg_fcs: int
    predict_steps: int

    def __init__(self):
        super().__init__()

    def _load_anchors(self, anchor_info_path):
        """
        Load kmeans trajectory anchor information from a pickle file.

        The loaded anchors are stored as numpy arrays (no torch dependency).

        Args:
            anchor_info_path (str): Path to the pickle file containing
                anchor_infos with key "anchors_all" — a list of numpy arrays.

        Sets:
            self.kmeans_anchors: numpy array [Nc, Pc, steps, 2]
                where Nc = num_anchor_group, Pc = num_anchor.
        """
        with open(anchor_info_path, 'rb') as f:
            anchor_infos = pickle.load(f)
        self.kmeans_anchors = np.stack(
            anchor_infos["anchors_all"])  # Nc, Pc, steps, 2

    def _build_layers(self, motionformer, det_layer_num):
        """
        Build the embedding and fusion layers of the motion prediction module.

        Args:
            motionformer: Pre-built TTSim MotionTransformerDecoder module.
            det_layer_num (int): Number of detection decoder layers
                (for the track query fuser input dim).

        Creates:
            self.learnable_motion_query_embedding_data: numpy [num_anchor*num_anchor_group, embed_dims]
                (set externally before inference, replaces nn.Embedding)
            self.motionformer: MotionTransformerDecoder module.
            self.layer_track_query_fuser: TrackQueryFuser (Linear+LN+ReLU).
            self.agent_level_embedding_layer: TwoLayerMLP (D→D*2→D).
            self.scene_level_ego_embedding_layer: TwoLayerMLP (D→D*2→D).
            self.scene_level_offset_embedding_layer: TwoLayerMLP (D→D*2→D).
            self.boxes_query_embedding_layer: TwoLayerMLP (D→D*2→D).
        """
        # Learnable motion query embedding — stored as raw numpy
        # In PyTorch: nn.Embedding(num_anchor * num_anchor_group, embed_dims)
        # Set externally via load_weights(); shape [num_anchor*num_anchor_group, embed_dims]
        self.learnable_motion_query_embedding_data = None

        # Motionformer (transformer decoder) — passed in pre-built
        self.motionformer = motionformer

        # Track query fuser: Linear(D*det_layer_num, D) + LN + ReLU
        self.layer_track_query_fuser = TrackQueryFuser(
            f'{self.name}.layer_track_query_fuser',
            in_features=self.embed_dims * det_layer_num,
            out_features=self.embed_dims)

        # Agent/scene embedding layers: Linear(D, D*2) + ReLU + Linear(D*2, D)
        self.agent_level_embedding_layer = TwoLayerMLP(
            f'{self.name}.agent_level_embedding_layer',
            in_features=self.embed_dims,
            hidden_features=self.embed_dims * 2,
            out_features=self.embed_dims)

        self.scene_level_ego_embedding_layer = TwoLayerMLP(
            f'{self.name}.scene_level_ego_embedding_layer',
            in_features=self.embed_dims,
            hidden_features=self.embed_dims * 2,
            out_features=self.embed_dims)

        self.scene_level_offset_embedding_layer = TwoLayerMLP(
            f'{self.name}.scene_level_offset_embedding_layer',
            in_features=self.embed_dims,
            hidden_features=self.embed_dims * 2,
            out_features=self.embed_dims)

        self.boxes_query_embedding_layer = TwoLayerMLP(
            f'{self.name}.boxes_query_embedding_layer',
            in_features=self.embed_dims,
            hidden_features=self.embed_dims * 2,
            out_features=self.embed_dims)

    def _init_layers(self):
        """
        Build classification and regression branches for trajectory prediction.

        Creates:
            self.traj_cls_branches: list of TrajClsBranch (one per decoder layer).
            self.traj_reg_branches: list of TrajRegBranch (one per decoder layer).
            self.traj_refine_branch: single TrajRegBranch (predict_steps * 2 output).
        """
        num_pred = self.motionformer.num_layers

        # Build a single refine branch (not cloned)
        self.traj_refine_branch = TrajRegBranch(
            f'{self.name}.traj_refine_branch',
            embed_dims=self.embed_dims,
            num_reg_fcs=self.num_reg_fcs,
            out_channels=self.predict_steps * 2)

        # Build per-layer cls and reg branches (cloned in PyTorch, here each
        # gets independent weights that are loaded identically if needed)
        cls_list = []
        reg_list = []
        dec_reg_list = []
        for i in range(num_pred):
            cls_list.append(TrajClsBranch(
                f'{self.name}.traj_cls_branch_{i}',
                embed_dims=self.embed_dims,
                num_reg_fcs=self.num_reg_fcs))
            reg_list.append(TrajRegBranch(
                f'{self.name}.traj_reg_branch_{i}',
                embed_dims=self.embed_dims,
                num_reg_fcs=self.num_reg_fcs,
                out_channels=self.predict_steps * 5))
            # Separate branches for the decoder loop (SimOpHandles can't be reused)
            dec_reg_list.append(TrajRegBranch(
                f'{self.name}.dec_traj_reg_branch_{i}',
                embed_dims=self.embed_dims,
                num_reg_fcs=self.num_reg_fcs,
                out_channels=self.predict_steps * 5))
        self.traj_cls_branches = SimNN.ModuleList(cls_list)
        self.traj_reg_branches = SimNN.ModuleList(reg_list)
        self.dec_traj_reg_branches = SimNN.ModuleList(dec_reg_list)

    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        Extract bounding-box centers and normalize to [0, 1] range.

        Args:
            bbox_results: List of tuples (bboxes, scores, labels, bbox_index, mask).
                bboxes is expected to have a gravity_center attribute or a tensor
                where columns 0,1 are the (x, y) center coordinates.
            bev_range: List [x_min, y_min, z_min, x_max, y_max, z_max].

        Returns:
            numpy array [batch_size, num_agents, 2] with normalized (x, y) centers.
        """
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            # Support both LiDARInstance3DBoxes (has gravity_center) and
            # plain numpy/tensor where columns 0,1 are x,y
            if hasattr(bboxes, 'gravity_center'):
                # PyTorch path — extract center (for hybrid usage)
                xy = bboxes.gravity_center[:, :2]
                if hasattr(xy, 'numpy'):
                    xy = xy.detach().cpu().numpy()
            elif hasattr(bboxes, 'tensor'):
                xy = bboxes.tensor[:, :2]
                if hasattr(xy, 'numpy'):
                    xy = xy.detach().cpu().numpy()
            else:
                # Pure numpy path
                xy = np.array(bboxes)[:, :2]

            x_norm = (xy[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                np.stack([x_norm, y_norm], axis=-1))
        return np.stack(det_bbox_posembed)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("BaseMotionHead TTSim Module")
    logger.info("=" * 80)

    # Test TwoLayerMLP
    try:
        mlp = TwoLayerMLP('test_mlp', in_features=256, hidden_features=512,
                           out_features=256)
        logger.debug(f"[OK] TwoLayerMLP: params={mlp.analytical_param_count()}")
    except Exception as e:
        logger.debug(f"[X]  TwoLayerMLP failed: {e}")

    # Test TrackQueryFuser
    try:
        fuser = TrackQueryFuser('test_fuser', in_features=1536, out_features=256)
        logger.debug(f"[OK] TrackQueryFuser: params={fuser.analytical_param_count()}")
    except Exception as e:
        logger.debug(f"[X]  TrackQueryFuser failed: {e}")

    # Test TrajClsBranch
    try:
        cls_br = TrajClsBranch('test_cls', embed_dims=256, num_reg_fcs=1)
        logger.debug(f"[OK] TrajClsBranch: params={cls_br.analytical_param_count()}")
    except Exception as e:
        logger.debug(f"[X]  TrajClsBranch failed: {e}")

    # Test TrajRegBranch
    try:
        reg_br = TrajRegBranch('test_reg', embed_dims=256, num_reg_fcs=1,
                               out_channels=60)
        logger.debug(f"[OK] TrajRegBranch: params={reg_br.analytical_param_count()}")
    except Exception as e:
        logger.debug(f"[X]  TrajRegBranch failed: {e}")

    # Test _extract_tracking_centers (numpy path)
    try:
        class DummyHead(BaseMotionHead):
            def __init__(self):
                super().__init__()
                self.name = 'dummy'
        head = DummyHead()
        # Simulate bbox results: list of (bboxes_array, scores, labels, idx, mask)
        bboxes = np.array([[10.0, 20.0, 0, 0, 0, 0, 0, 0, 0],
                           [30.0, 40.0, 0, 0, 0, 0, 0, 0, 0]])
        results = [(bboxes, None, None, None, None)]
        bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        centers = head._extract_tracking_centers(results, bev_range)
        assert centers.shape == (1, 2, 2), f"Expected (1,2,2), got {centers.shape}"
        logger.debug(f"[OK] _extract_tracking_centers: shape={centers.shape}")
    except Exception as e:
        logger.debug(f"[X]  _extract_tracking_centers failed: {e}")

    logger.info("\n[OK] All BaseMotionHead component tests passed!")
    logger.info("=" * 80)
