


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of MotionHead for FusionAD.

Inference-only conversion.  Training-specific methods (forward_train, loss,
compute_loss_traj, compute_matched_gt_traj, _build_loss) are omitted.

Classes:
  - MotionHead : Full motion prediction head combining trajectory prediction
                 with anchor-based multimodal motion forecasting.

Dependencies (TTSim, already converted):
  - base_motion_head.BaseMotionHead     — base class, layer builders, anchor loading
  - modules.MotionTransformerDecoder     — the motionformer decoder
  - motion_deformable_attn.MotionTransformerAttentionLayer — BEV attention layers
  - functional.py                        — Pos2PosEmb2D, NormPoints,
                                           AnchorCoordinateTransform,
                                           TrajectoryCoordinateTransform,
                                           BivariateGaussianActivation
"""

# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================

# import torch
# import copy
# from mmdet.models import HEADS
# from mmcv.runner import force_fp32, auto_fp16
# from projects.mmdet3d_plugin.models.utils.functional import (
#     bivariate_gaussian_activation,
#     norm_points,
#     pos2posemb2d,
#     anchor_coordinate_transform
# )
# from .motion_head_plugin.motion_utils import nonlinear_smoother
# from .motion_head_plugin.base_motion_head import BaseMotionHead
#
#
# @HEADS.register_module()
# class MotionHead(BaseMotionHead):
#     """
#     MotionHead module for a neural network, which predicts motion trajectories and is used in an autonomous driving task.
#
#     Args:
#         *args: Variable length argument list.
#         predict_steps (int): The number of steps to predict motion trajectories.
#         transformerlayers (dict): A dictionary defining the configuration of transformer layers.
#         bbox_coder: An instance of a bbox coder to be used for encoding/decoding boxes.
#         num_cls_fcs (int): The number of fully-connected layers in the classification branch.
#         bev_h (int): The height of the bird's-eye-view map.
#         bev_w (int): The width of the bird's-eye-view map.
#         embed_dims (int): The number of dimensions to use for the query and key vectors in transformer layers.
#         num_anchor (int): The number of anchor points.
#         det_layer_num (int): The number of layers in the transformer model.
#         group_id_list (list): A list of group IDs to use for grouping the classes.
#         pc_range: The range of the point cloud.
#         use_nonlinear_optimizer (bool): A boolean indicating whether to use a non-linear optimizer for training.
#         anchor_info_path (str): The path to the file containing the anchor information.
#         vehicle_id_list(list[int]): class id of vehicle class, used for filtering out non-vehicle objects
#     """
#     def __init__(self,
#                  *args,
#                  predict_steps=12,
#                  transformerlayers=None,
#                  bbox_coder=None,
#                  num_cls_fcs=2,
#                  bev_h=30,
#                  bev_w=30,
#                  embed_dims=256,
#                  num_anchor=6,
#                  det_layer_num=6,
#                  group_id_list=[],
#                  pc_range=None,
#                  use_nonlinear_optimizer=False,
#                  anchor_info_path=None,
#                  loss_traj=dict(),
#                  num_classes=0,
#                  vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
#                  **kwargs):
#         super(MotionHead, self).__init__()
#
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.num_cls_fcs = num_cls_fcs - 1
#         self.num_reg_fcs = num_cls_fcs - 1
#         self.embed_dims = embed_dims
#         self.num_anchor = num_anchor
#         self.num_anchor_group = len(group_id_list)
#
#         # we merge the classes into groups for anchor assignment
#         self.cls2group = [0 for i in range(num_classes)]
#         for i, grouped_ids in enumerate(group_id_list):
#             for gid in grouped_ids:
#                 self.cls2group[gid] = i
#         self.cls2group = torch.tensor(self.cls2group)
#         self.pc_range = pc_range
#         self.predict_steps = predict_steps
#         self.vehicle_id_list = vehicle_id_list
#
#         self.use_nonlinear_optimizer = use_nonlinear_optimizer
#         self._load_anchors(anchor_info_path)
#         self._build_loss(loss_traj)
#         self._build_layers(transformerlayers, det_layer_num)
#         self._init_layers()
#
#     def forward_train(self,
#                       bev_embed,
#                       gt_bboxes_3d,
#                       gt_labels_3d,
#                       gt_fut_traj=None,
#                       gt_fut_traj_mask=None,
#                       gt_sdc_fut_traj=None,
#                       gt_sdc_fut_traj_mask=None,
#                       outs_track={},
#                       outs_seg={}
#                   ):
#         """Forward function
#         Args:
#             bev_embed (Tensor): BEV feature map with the shape of [B, C, H, W].
#             gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
#                 bboxes of each sample.
#             gt_labels_3d (list[torch.Tensor]): Labels of each sample.
#             img_metas (list[dict]): Meta information of each sample.
#             gt_fut_traj (list[torch.Tensor]): Ground truth future trajectory of each sample.
#             gt_fut_traj_mask (list[torch.Tensor]): Ground truth future trajectory mask of each sample.
#             gt_sdc_fut_traj (list[torch.Tensor]): Ground truth future trajectory of each sample.
#             gt_sdc_fut_traj_mask (list[torch.Tensor]): Ground truth future trajectory mask of each sample.
#             outs_track (dict): Outputs of track head.
#             outs_seg (dict): Outputs of seg head.
#             future_states (list[torch.Tensor]): Ground truth future states of each sample.
#         Returns:
#             dict: Losses of each branch.
#         """
#         track_query = outs_track['track_query_embeddings'][None, None, ...] # num_dec, B, A_track, D
#         all_matched_idxes = [outs_track['track_query_matched_idxes']] #BxN
#         track_boxes = outs_track['track_bbox_results']
#
#         # cat sdc query/gt to the last
#         sdc_match_index = torch.zeros((1,), dtype=all_matched_idxes[0].dtype, device=all_matched_idxes[0].device)
#         sdc_match_index[0] = gt_fut_traj[0].shape[0]
#         all_matched_idxes = [torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)]
#         gt_fut_traj[0] = torch.cat([gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
#         gt_fut_traj_mask[0] = torch.cat([gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)
#         track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
#         sdc_track_boxes = outs_track['sdc_track_bbox_results']
#         track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
#         track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
#         track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
#         track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
#
#         memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
#
#         outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
#         loss_inputs = [gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, outs_motion, all_matched_idxes, track_boxes]
#         losses = self.loss(*loss_inputs)
#
#         def filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, vehicle_id_list):
#             query_label = gt_labels_3d[0][-1][all_matched_idxes[0]]
#             # select vehicle query according to vehicle_id_list
#             vehicle_mask = torch.zeros_like(query_label)
#             for veh_id in vehicle_id_list:
#                 vehicle_mask |=  query_label == veh_id
#             outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
#             outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
#             outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
#             all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask>0]
#             return outs_motion, all_matched_idxes
#
#         all_matched_idxes[0] = all_matched_idxes[0][:-1]
#         outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]         # [3, 1, 6, 256]     [n_dec, b, n_mode, d]
#         outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]          # [1, 256]           [b, d]
#         outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]  # [1, 256]           [b, d]
#         outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]            # [3, 1, 3, 6, 256]  [n_dec, b, nq, n_mode, d]
#         outs_motion['track_query'] = outs_motion['track_query'][:, :-1]             # [1, 3, 256]        [b, nq, d]
#         outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]     # [1, 3, 256]        [b, nq, d]
#
#
#         outs_motion, all_matched_idxes = filter_vehicle_query(outs_motion, all_matched_idxes, gt_labels_3d, self.vehicle_id_list)
#         outs_motion['all_matched_idxes'] = all_matched_idxes
#
#         ret_dict = dict(losses=losses, outs_motion=outs_motion, track_boxes=track_boxes)
#         return ret_dict
#
#     def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
#         """Test function"""
#         track_query = outs_track['track_query_embeddings'][None, None, ...]
#         track_boxes = outs_track['track_bbox_results']
#
#         track_query = torch.cat([track_query, outs_track['sdc_embedding'][None, None, None, :]], dim=2)
#         sdc_track_boxes = outs_track['sdc_track_bbox_results']
#
#         track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
#         track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
#         track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
#         track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
#         memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg['args_tuple']
#         outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
#         traj_results = self.get_trajs(outs_motion, track_boxes)
#         bboxes, scores, labels, bbox_index, mask = track_boxes[0]
#         outs_motion['track_scores'] = scores[None, :]
#         labels[-1] = 0
#         def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
#             if len(labels) < 1:  # No other obj query except sdc query.
#                 return None
#
#             # select vehicle query according to vehicle_id_list
#             vehicle_mask = torch.zeros_like(labels)
#             for veh_id in vehicle_id_list:
#                 vehicle_mask |=  labels == veh_id
#             outs_motion['traj_query'] = outs_motion['traj_query'][:, :, vehicle_mask>0]
#             outs_motion['track_query'] = outs_motion['track_query'][:, vehicle_mask>0]
#             outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, vehicle_mask>0]
#             outs_motion['track_scores'] = outs_motion['track_scores'][:, vehicle_mask>0]
#             return outs_motion
#
#         outs_motion = filter_vehicle_query(outs_motion, labels, self.vehicle_id_list)
#
#         # filter sdc query
#         outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
#         outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
#         outs_motion['sdc_track_query_pos'] = outs_motion['track_query_pos'][:, -1]
#         outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
#         outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
#         outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:, :-1]
#         outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]
#
#         return traj_results, outs_motion
#
#     @auto_fp16(apply_to=('bev_embed', 'track_query', 'lane_query', 'lane_query_pos', 'lane_query_embed', 'prev_bev'))
#     def forward(self,
#                 bev_embed,
#                 track_query,
#                 lane_query,
#                 lane_query_pos,
#                 track_bbox_results):
#         """
#         Applies forward pass on the model for motion prediction using bird's eye view (BEV) embedding, track query, lane query, and track bounding box results.
#
#         Args:
#         bev_embed (torch.Tensor): A tensor of shape (h*w, B, D) representing the bird's eye view embedding.
#         track_query (torch.Tensor): A tensor of shape (B, num_dec, A_track, D) representing the track query.
#         lane_query (torch.Tensor): A tensor of shape (N, M_thing, D) representing the lane query.
#         lane_query_pos (torch.Tensor): A tensor of shape (N, M_thing, D) representing the position of the lane query.
#         track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.
#
#         Returns:
#         dict: A dictionary containing the following keys and values:
#         - 'all_traj_scores': A tensor of shape (num_levels, B, A_track, num_points) with trajectory scores for each level.
#         - 'all_traj_preds': A tensor of shape (num_levels, B, A_track, num_points, num_future_steps, 2) with predicted trajectories for each level.
#         - 'valid_traj_masks': A tensor of shape (B, A_track) indicating the validity of trajectory masks.
#         - 'traj_query': A tensor containing intermediate states of the trajectory queries.
#         - 'track_query': A tensor containing the input track queries.
#         - 'track_query_pos': A tensor containing the positional embeddings of the track queries.
#         """
#
#         dtype = track_query.dtype
#         device = track_query.device
#         num_groups = self.kmeans_anchors.shape[0]
#
#         # extract the last frame of the track query
#         track_query = track_query[:, -1]
#
#         # encode the center point of the track query
#         reference_points_track = self._extract_tracking_centers(
#             track_bbox_results, self.pc_range)
#         track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))  # B, A, D
#
#         # construct the learnable query postional embedding
#         # split and stack according to groups
#         learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)  # latent anchor (P*G, D)
#         learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))
#
#         # construct the agent level/scene-level query positional embedding
#         # (num_groups, num_anchor, 12, 2)
#         # to incorporate the information of different groups and coordinates, and embed the headding and location information
#         agent_level_anchors = self.kmeans_anchors.to(dtype).to(device).view(num_groups, self.num_anchor, self.predict_steps, 2).detach()
#         scene_level_ego_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=True)  # B, A, G, P ,12 ,2
#         scene_level_offset_anchors = anchor_coordinate_transform(agent_level_anchors, track_bbox_results, with_translation_transform=False)  # B, A, G, P ,12 ,2
#
#         agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
#         scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
#         scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)
#
#         # we only use the last point of the anchor
#         agent_level_embedding = self.agent_level_embedding_layer(
#             pos2posemb2d(agent_level_norm[..., -1, :]))  # G, P, D
#         scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
#             pos2posemb2d(scene_level_ego_norm[..., -1, :]))  # B, A, G, P , D
#         scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
#             pos2posemb2d(scene_level_offset_norm[..., -1, :]))  # B, A, G, P , D
#
#         batch_size, num_agents = scene_level_ego_embedding.shape[:2]
#         agent_level_embedding = agent_level_embedding[None,None, ...].expand(batch_size, num_agents, -1, -1, -1)
#         learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)
#
#         # save for latter, anchors
#         # B, A, G, P ,12 ,2 -> B, A, P ,12 ,2
#         scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)
#
#         # select class embedding
#         # B, A, G, P , D-> B, A, P , D
#         agent_level_embedding = self.group_mode_query_pos(
#             track_bbox_results, agent_level_embedding)
#         scene_level_ego_embedding = self.group_mode_query_pos(
#             track_bbox_results, scene_level_ego_embedding)  # B, A, G, P , D-> B, A, P , D
#
#         # B, A, G, P , D -> B, A, P , D
#         scene_level_offset_embedding = self.group_mode_query_pos(
#             track_bbox_results, scene_level_offset_embedding)
#         learnable_embed = self.group_mode_query_pos(
#             track_bbox_results, learnable_embed)
#
#         init_reference = scene_level_offset_anchors.detach()
#         outputs_traj_scores = []
#         outputs_trajs = []
#
#         inter_states, inter_references, offset = self.motionformer(
#             track_query,  # B, A_track, D
#             lane_query,  # B, M, D
#             track_query_pos=track_query_pos,
#             lane_query_pos=lane_query_pos,
#             track_bbox_results=track_bbox_results,
#             bev_embed=bev_embed,
#             reference_trajs=init_reference,
#             traj_reg_branches=self.traj_reg_branches,
#             traj_cls_branches=self.traj_cls_branches,
#             traj_refine_branch=self.traj_refine_branch,
#             # anchor embeddings
#             agent_level_embedding=agent_level_embedding,
#             scene_level_ego_embedding=scene_level_ego_embedding,
#             scene_level_offset_embedding=scene_level_offset_embedding,
#             learnable_embed=learnable_embed,
#             # anchor positional embeddings layers
#             agent_level_embedding_layer=self.agent_level_embedding_layer,
#             scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
#             scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
#             spatial_shapes=torch.tensor(
#                 [[self.bev_h, self.bev_w]], device=device),
#             level_start_index=torch.tensor([0], device=device))
#
#         for lvl in range(inter_states.shape[0]):
#             outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
#             tmp = self.traj_reg_branches[lvl](inter_states[lvl])
#             tmp = self.unflatten_traj(tmp)
#
#             # we use cumsum trick here to get the trajectory
#             tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
#
#             outputs_class = self.log_softmax(outputs_class.squeeze(3))
#             outputs_traj_scores.append(outputs_class)
#
#             for bs in range(tmp.shape[0]):
#                 tmp[bs] = bivariate_gaussian_activation(tmp[bs])
#             outputs_trajs.append(tmp)
#
#         outputs_traj_scores = torch.stack(outputs_traj_scores)
#         outputs_trajs = torch.stack(outputs_trajs)
#         B, A_track, D = track_query.shape
#         valid_traj_masks = track_query.new_ones((B, A_track)) > 0
#         outs = {
#             'all_traj_scores': outputs_traj_scores,
#             'all_traj_preds': outputs_trajs,
#             'valid_traj_masks': valid_traj_masks,
#             'traj_query': inter_states,
#             'track_query': track_query,
#             'track_query_pos': track_query_pos,
#             'offset': offset
#         }
#
#         return outs
#
#     def group_mode_query_pos(self, bbox_results, mode_query_pos):
#         """
#         Group mode query positions based on the input bounding box results.
#
#         Args:
#             bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
#             mode_query_pos (torch.Tensor): A tensor of shape (B, A, G, P, D) representing the mode query positions.
#
#         Returns:
#             torch.Tensor: A tensor of shape (B, A, P, D) representing the classified mode query positions.
#         """
#         batch_size = len(bbox_results)
#         agent_num = mode_query_pos.shape[1]
#         batched_mode_query_pos = []
#         self.cls2group = self.cls2group.to(mode_query_pos.device)
#         # TODO: vectorize this
#         # group the embeddings based on the class
#         for i in range(batch_size):
#             bboxes, scores, labels, bbox_index, mask = bbox_results[i]
#             label = labels.to(mode_query_pos.device)
#             grouped_label = self.cls2group[label]
#             grouped_mode_query_pos = []
#             for j in range(agent_num):
#                 grouped_mode_query_pos.append(
#                     mode_query_pos[i, j, grouped_label[j]])
#             batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
#         return torch.stack(batched_mode_query_pos)
#
#     @force_fp32(apply_to=('preds_dicts_motion'))
#     def loss(self,
#              gt_bboxes_3d,
#              gt_fut_traj,
#              gt_fut_traj_mask,
#              preds_dicts_motion,
#              all_matched_idxes,
#              track_bbox_results):
#         """
#         Computes the loss function for the given ground truth and prediction dictionaries.
#
#         Args:
#             gt_bboxes_3d (List[torch.Tensor]): A list of tensors representing ground truth 3D bounding boxes for each image.
#             gt_fut_traj (torch.Tensor): A tensor representing the ground truth future trajectories.
#             gt_fut_traj_mask (torch.Tensor): A tensor representing the ground truth future trajectory masks.
#             preds_dicts_motion (Dict[str, torch.Tensor]): A dictionary containing motion-related prediction tensors.
#             all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched ground truth indices for each image in the batch.
#             track_bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the tracking bounding box results for each image in the batch.
#
#         Returns:
#             dict[str, torch.Tensor]: A dictionary of loss components.
#         """
#
#         # motion related predictions
#         all_traj_scores = preds_dicts_motion['all_traj_scores']
#         all_traj_preds = preds_dicts_motion['all_traj_preds']
#         traj_offset = preds_dicts_motion['offset']
#
#         num_dec_layers = len(all_traj_scores)
#
#         all_gt_fut_traj = [gt_fut_traj for _ in range(num_dec_layers)]
#         all_gt_fut_traj_mask = [
#             gt_fut_traj_mask for _ in range(num_dec_layers)]
#
#         losses_traj = []
#         gt_fut_traj_all, gt_fut_traj_mask_all = self.compute_matched_gt_traj(
#             all_gt_fut_traj[0], all_gt_fut_traj_mask[0], all_matched_idxes, track_bbox_results, gt_bboxes_3d)
#         compute_refine = False
#         for i in range(num_dec_layers):
#             compute_refine = True if i == num_dec_layers - 1 else False
#             loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr, loss_refine = self.compute_loss_traj(all_traj_scores[i], all_traj_preds[i], traj_offset,
#                                                                                          gt_fut_traj_all, gt_fut_traj_mask_all, all_matched_idxes, compute_refine)
#             losses_traj.append(
#                 (loss_traj, l_class, l_reg, l_mindae, l_minfde, l_mr))
#
#         loss_dict = dict()
#         loss_dict['loss_traj'] = losses_traj[-1][0]
#         loss_dict['l_class'] = losses_traj[-1][1]
#         loss_dict['l_reg'] = losses_traj[-1][2]
#         loss_dict['min_ade'] = losses_traj[-1][3]
#         loss_dict['min_fde'] = losses_traj[-1][4]
#         loss_dict['mr'] = losses_traj[-1][5]
#         loss_dict['loss_refine'] = loss_refine
#         # loss from other decoder layers
#         num_dec_layer = 0
#         for loss_traj_i in losses_traj[:-1]:
#             loss_dict[f'd{num_dec_layer}.loss_traj'] = loss_traj_i[0]
#             loss_dict[f'd{num_dec_layer}.l_class'] = loss_traj_i[1]
#             loss_dict[f'd{num_dec_layer}.l_reg'] = loss_traj_i[2]
#             loss_dict[f'd{num_dec_layer}.min_ade'] = loss_traj_i[3]
#             loss_dict[f'd{num_dec_layer}.min_fde'] = loss_traj_i[4]
#             loss_dict[f'd{num_dec_layer}.mr'] = loss_traj_i[5]
#             num_dec_layer += 1
#
#         return loss_dict
#
#     def compute_matched_gt_traj(self,
#                                 gt_fut_traj,
#                                 gt_fut_traj_mask,
#                                 all_matched_idxes,
#                                 track_bbox_results,
#                                 gt_bboxes_3d):
#         """
#         Computes the matched ground truth trajectories for a batch of images based on matched indexes.
#
#         Args:
#         gt_fut_traj (torch.Tensor): Ground truth future trajectories of shape (num_imgs, num_objects, num_future_steps, 2).
#         gt_fut_traj_mask (torch.Tensor): Ground truth future trajectory masks of shape (num_imgs, num_objects, num_future_steps, 2).
#         all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched indexes for each image in the batch.
#         track_bbox_results (List[torch.Tensor]): A list of tensors containing the tracking bounding box results for each image in the batch.
#         gt_bboxes_3d (List[torch.Tensor]): A list of tensors containing the ground truth 3D bounding boxes for each image in the batch.
#
#         Returns:
#         torch.Tensor: A concatenated tensor of the matched ground truth future trajectories.
#         torch.Tensor: A concatenated tensor of the matched ground truth future trajectory masks.
#         """
#         num_imgs = len(all_matched_idxes)
#         gt_fut_traj_all = []
#         gt_fut_traj_mask_all = []
#         for i in range(num_imgs):
#             matched_gt_idx = all_matched_idxes[i]
#             valid_traj_masks = matched_gt_idx >= 0
#             matched_gt_fut_traj = gt_fut_traj[i][matched_gt_idx][valid_traj_masks]
#             matched_gt_fut_traj_mask = gt_fut_traj_mask[i][matched_gt_idx][valid_traj_masks]
#             if self.use_nonlinear_optimizer:
#                 # TODO: sdc query is not supported non-linear optimizer
#                 bboxes = track_bbox_results[i][0].tensor[valid_traj_masks]
#                 matched_gt_bboxes_3d = gt_bboxes_3d[i][-1].tensor[matched_gt_idx[:-1]
#                                                                   ][valid_traj_masks[:-1]]
#                 sdc_gt_fut_traj = matched_gt_fut_traj[-1:]
#                 sdc_gt_fut_traj_mask = matched_gt_fut_traj_mask[-1:]
#                 matched_gt_fut_traj = matched_gt_fut_traj[:-1]
#                 matched_gt_fut_traj_mask = matched_gt_fut_traj_mask[:-1]
#                 bboxes = bboxes[:-1]
#                 matched_gt_fut_traj, matched_gt_fut_traj_mask = nonlinear_smoother(
#                     matched_gt_bboxes_3d, matched_gt_fut_traj, matched_gt_fut_traj_mask, bboxes)
#                 matched_gt_fut_traj = torch.cat(
#                     [matched_gt_fut_traj, sdc_gt_fut_traj], dim=0)
#                 matched_gt_fut_traj_mask = torch.cat(
#                     [matched_gt_fut_traj_mask, sdc_gt_fut_traj_mask], dim=0)
#             matched_gt_fut_traj_mask = torch.all(
#                 matched_gt_fut_traj_mask > 0, dim=-1)
#             gt_fut_traj_all.append(matched_gt_fut_traj)
#             gt_fut_traj_mask_all.append(matched_gt_fut_traj_mask)
#         gt_fut_traj_all = torch.cat(gt_fut_traj_all, dim=0)
#         gt_fut_traj_mask_all = torch.cat(gt_fut_traj_mask_all, dim=0)
#         return gt_fut_traj_all, gt_fut_traj_mask_all
#
#     def compute_loss_traj(self,
#                           traj_scores,
#                           traj_preds,
#                           offset_preds,
#                           gt_fut_traj_all,
#                           gt_fut_traj_mask_all,
#                           all_matched_idxes,
#                           need_refine):
#         """
#         Computes the trajectory loss given the predicted trajectories, ground truth trajectories, and other relevant information.
#
#         Args:
#             traj_scores (torch.Tensor): A tensor representing the trajectory scores.
#             traj_preds (torch.Tensor): A tensor representing the predicted trajectories.
#             gt_fut_traj_all (torch.Tensor): A tensor representing the ground truth future trajectories.
#             gt_fut_traj_mask_all (torch.Tensor): A tensor representing the ground truth future trajectory masks.
#             all_matched_idxes (List[torch.Tensor]): A list of tensors containing the matched ground truth indices for each image in the batch.
#
#         Returns:
#             tuple: A tuple containing the total trajectory loss, classification loss, regression loss, minimum average displacement error, minimum final displacement error, and miss rate.
#         """
#         num_imgs = traj_scores.size(0)
#         traj_prob_all = []
#         traj_preds_all = []
#         offset_preds_all = []
#         for i in range(num_imgs):
#             matched_gt_idx = all_matched_idxes[i]
#             valid_traj_masks = matched_gt_idx >= 0
#             # select valid and matched
#             batch_traj_prob = traj_scores[i, valid_traj_masks, :]
#             # (n_objs, n_modes, step, 5)
#             batch_traj_preds = traj_preds[i, valid_traj_masks, ...]
#             batch_offset_preds = offset_preds[i, valid_traj_masks, ...]
#             traj_prob_all.append(batch_traj_prob)
#             traj_preds_all.append(batch_traj_preds)
#             offset_preds_all.append(batch_offset_preds)
#         traj_prob_all = torch.cat(traj_prob_all, dim=0)
#         traj_preds_all = torch.cat(traj_preds_all, dim=0)
#         offset_preds_all = torch.cat(offset_preds_all, dim=0)
#         traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr, l_refine = self.loss_traj(
#             traj_prob_all, traj_preds_all, offset_preds_all, gt_fut_traj_all, gt_fut_traj_mask_all, need_refine)
#         return traj_loss, l_class, l_reg, l_minade, l_minfde, l_mr, l_refine
#
#     @force_fp32(apply_to=('preds_dicts'))
#     def get_trajs(self, preds_dicts, bbox_results):
#         """
#         Generates trajectories from the prediction results, bounding box results.
#
#         Args:
#             preds_dicts (tuple[list[dict]]): A tuple containing lists of dictionaries with prediction results.
#             bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
#
#         Returns:
#             List[dict]: A list of dictionaries containing decoded bounding boxes, scores, and labels after non-maximum suppression.
#         """
#         num_samples = len(bbox_results)
#         num_layers = preds_dicts['all_traj_preds'].shape[0]
#         ret_list = []
#         for i in range(num_samples):
#             preds = dict()
#             for j in range(num_layers):
#                 subfix = '_' + str(j) if j < (num_layers - 1) else ''
#                 traj = preds_dicts['all_traj_preds'][j, i]
#                 traj_scores = preds_dicts['all_traj_scores'][j, i]
#                 if j == num_layers - 1:
#                     traj[...,:2] = traj[...,:2] + preds_dicts['offset'][i]
#                 traj_scores, traj = traj_scores.cpu(), traj.cpu()
#                 preds['traj' + subfix] = traj
#                 preds['traj_scores' + subfix] = traj_scores
#             ret_list.append(preds)
#         return ret_list

# =============================================================================
# TTsim CODE
# =============================================================================

import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add dense_heads directory
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add motion_head_plugin directory
motion_plugin_dir = os.path.join(current_dir, 'motion_head_plugin')
if motion_plugin_dir not in sys.path:
    sys.path.insert(0, motion_plugin_dir)

# Add fusionad directory
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


def _sim_data(t, dtype=np.float32):
    """Extract numpy data from a SimTensor, falling back to zeros if data is None."""
    if hasattr(t, 'op_in'):
        if t.data is not None:
            return t.data
        return np.zeros(t.shape, dtype=dtype)
    return t


def _reg_tensor(module, tensor):
    """Register a dynamic SimTensor into module._tensors."""
    module._tensors[tensor.name] = tensor
    return tensor


from .motion_head_plugin.base_motion_head import BaseMotionHead
from .motion_head_plugin.modules import MotionTransformerDecoder
from .motion_head_plugin.motion_deformable_attn import MotionTransformerAttentionLayer

from workloads.FusionAD.projects.mmdet_plugin.models.utils.functional import (
    Pos2PosEmb2D,
    NormPoints,
    AnchorCoordinateTransform,
    TrajectoryCoordinateTransform,
    BivariateGaussianActivation,
)


class MotionHead(BaseMotionHead):
    """
    MotionHead for multimodal trajectory prediction.

    Predicts future trajectories for tracked agents using:
      - Anchor-based multimodal queries (kmeans anchors per class group)
      - TrackAgent / Map / BEV interactions via MotionTransformerDecoder
      - Per-layer trajectory regression + classification branches
      - Bivariate Gaussian activation on trajectory outputs

    Inference flow:
      1. Extract tracking centers → positional embedding
      2. Build per-agent anchor embeddings (agent/ego/offset level)
      3. Group anchor embeddings by class
      4. Run motionformer decoder (num_layers iterations + refinement)
      5. Per-layer: cls branch → log_softmax, reg branch → unflatten → cumsum → gaussian

    Args:
        name (str): Module name.
        predict_steps (int): Future prediction horizon.  Default: 12.
        transformerlayers (dict): Config for MotionTransformerAttentionLayer.
        num_cls_fcs (int): Num FC layers in cls/reg branches.  Default: 2.
        bev_h (int): BEV height.
        bev_w (int): BEV width.
        embed_dims (int): Embedding dimension.  Default: 256.
        num_anchor (int): Number of anchor modes per group.  Default: 6.
        det_layer_num (int): Number of detection decoder layers (for query fuser).
        group_id_list (list[list[int]]): Class-to-group mapping.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        anchor_info_path (str): Path to kmeans anchor pickle file.
        num_classes (int): Total number of object classes.
        vehicle_id_list (list[int]): Vehicle class IDs (used in test filtering).
    """

    def __init__(self, name,
                 predict_steps=12,
                 transformerlayers=None,
                 num_cls_fcs=2,
                 bev_h=30,
                 bev_w=30,
                 embed_dims=256,
                 num_anchor=6,
                 det_layer_num=6,
                 group_id_list=None,
                 pc_range=None,
                 anchor_info_path=None,
                 num_classes=0,
                 vehicle_id_list=None,
                 **kwargs):
        super().__init__()
        self.name = name
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.predict_steps = predict_steps
        self.pc_range = pc_range

        if vehicle_id_list is None:
            vehicle_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.vehicle_id_list = vehicle_id_list

        if group_id_list is None:
            group_id_list = []
        self.num_anchor_group = len(group_id_list)

        # Class-to-group mapping (numpy, no torch)
        self.cls2group = np.zeros(num_classes, dtype=np.int64)
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i

        # Load anchor trajectories
        self.kmeans_anchors: Any = None
        if anchor_info_path is not None:
            self._load_anchors(anchor_info_path)

        # Build BEV interaction layers from config
        bev_layers = self._build_bev_layers(name, transformerlayers)

        # Build motionformer (MotionTransformerDecoder)
        motionformer = MotionTransformerDecoder(
            f'{name}.motionformer',
            pc_range=pc_range,
            embed_dims=embed_dims,
            num_layers=kwargs.get('num_layers', 3),
            bev_interaction_layers=bev_layers[:-1],
            bev_interaction_rl=bev_layers[-1])

        # Build embedding + fusion layers (via BaseMotionHead)
        self._build_layers(motionformer, det_layer_num)

        # Build cls/reg/refine branches (via BaseMotionHead)
        self._init_layers()

        # Functional utility modules
        self.pos2posemb2d = Pos2PosEmb2D(
            f'{name}.pos2posemb2d', num_pos_feats=embed_dims // 2)
        self.norm_points = NormPoints(f'{name}.norm_points', pc_range)
        self.anchor_coord_transform = AnchorCoordinateTransform(
            f'{name}.anchor_coord_transform')
        self.traj_coord_transform = TrajectoryCoordinateTransform(
            f'{name}.traj_coord_transform')
        self.bivariate_gaussian = BivariateGaussianActivation(
            f'{name}.bivariate_gaussian')

        # Log-softmax / squeeze ops created dynamically per-lvl in __call__

        # Unsqueeze ops for broadcasting
        self.unsq_track_q = F.Unsqueeze(f'{name}.unsq_track_q')
        self.unsq_track_qpos = F.Unsqueeze(f'{name}.unsq_track_qpos')
        self.unsq_ref = F.Unsqueeze(f'{name}.unsq_ref')
        self.unsq_ax2 = F._from_data(f'{name}.unsq_ax2',
                                     np.array([2], dtype=np.int64), is_const=True)
        self.unsq_ax4 = F._from_data(f'{name}.unsq_ax4',
                                     np.array([4], dtype=np.int64), is_const=True)
        self.unsq_ax3 = F._from_data(f'{name}.unsq_ax3',
                                     np.array([3], dtype=np.int64), is_const=True)

        # Tile ops for broadcasting (B,A,1,D) → (B,A,P,D)
        self.tile_track_q = F.Tile(f'{name}.tile_track_q')
        self.tile_track_qpos = F.Tile(f'{name}.tile_track_qpos')
        self._tile_reps = F._from_data(
            f'{name}._tile_reps',
            np.array([1, 1, num_anchor, 1], dtype=np.int64), is_const=True)

        # Add ops
        self.add_static_intent = F.Add(f'{name}.add_static_intent')
        self.add_static_intent2 = F.Add(f'{name}.add_static_intent2')

        # --- _select_last_dec graph ops ---
        self.sld_slice = F.SliceF(f'{name}.sld_slice')
        self.sld_squeeze = F.Squeeze(f'{name}.sld_squeeze')
        self.sld_starts = F._from_data(
            f'{name}.sld_starts', np.array([-1], dtype=np.int64), is_const=True)
        self.sld_ends = F._from_data(
            f'{name}.sld_ends', np.array([2147483647], dtype=np.int64), is_const=True)
        self.sld_axes = F._from_data(
            f'{name}.sld_axes', np.array([1], dtype=np.int64), is_const=True)

        # --- _unflatten_and_activate shared constants ---
        # Ops are created per-iteration in _unflatten_and_activate;
        # only slice constants are shared here.
        self.uf_cumsum_ax = F._from_data(
            f'{name}.uf_cumsum_ax', np.array([3], dtype=np.int64), is_const=True)
        self.uf_ax4 = F._from_data(
            f'{name}.uf_ax4', np.array([4], dtype=np.int64), is_const=True)
        self.uf_s0 = F._from_data(
            f'{name}.uf_s0', np.array([0], dtype=np.int64), is_const=True)
        self.uf_e2 = F._from_data(
            f'{name}.uf_e2', np.array([2], dtype=np.int64), is_const=True)
        self.uf_s2 = self.uf_e2   # reuse [2]
        self.uf_e3 = F._from_data(
            f'{name}.uf_e3', np.array([3], dtype=np.int64), is_const=True)
        self.uf_s3 = self.uf_e3   # reuse [3]
        self.uf_e4 = F._from_data(
            f'{name}.uf_e4', np.array([4], dtype=np.int64), is_const=True)
        self.uf_s4 = self.uf_e4   # reuse [4]
        self.uf_e5 = F._from_data(
            f'{name}.uf_e5', np.array([5], dtype=np.int64), is_const=True)

        super().link_op2module()

    def _build_bev_layers(self, name, transformerlayers):
        """Build BEV interaction layers (MotionTransformerAttentionLayer) from config.

        Returns list of num_layers+1 layers (num_layers for iterations + 1 for refinement).
        """
        if transformerlayers is None:
            return [None] * 4  # 3 layers + 1 refinement default

        num_layers = 3  # default
        if 'num_layers' in transformerlayers:
            num_layers = transformerlayers.pop('num_layers')

        layers = []
        for i in range(num_layers + 1):
            suffix = f'rl' if i == num_layers else str(i)
            layer = MotionTransformerAttentionLayer(
                f'{name}.bev_interaction_{suffix}',
                attn_cfgs=transformerlayers.get('attn_cfgs'),
                ffn_cfgs=transformerlayers.get('ffn_cfgs'),
                operation_order=transformerlayers.get('operation_order',
                                                      ('cross_attn', 'norm', 'ffn', 'norm')),
                embed_dims=transformerlayers.get('embed_dims', self.embed_dims))
            layers.append(layer)
        return layers

    def __call__(self, bev_embed, track_query, lane_query, lane_query_pos,
                 track_bbox_results):
        """
        Forward pass for motion prediction (inference).

        Args:
            bev_embed: SimTensor (h*w, B, D) — BEV features (seq-first).
            track_query: SimTensor (B, num_dec, A_track, D) — track queries.
                         Only the last decoder layer is used: track_query[:, -1].
            lane_query: SimTensor (B, M, D) — lane queries.
            lane_query_pos: SimTensor (B, M, D) — lane positional encoding.
            track_bbox_results: list of bbox result tuples.

        Returns:
            dict with keys:
                'all_traj_scores': list of SimTensor (B, A, P) per level
                'all_traj_preds':  list of SimTensor (B, A, P, T, 5) per level
                'traj_query':      list of SimTensor (B, A, P, D) per level
                'track_query':     SimTensor (B, A, D)
                'track_query_pos': SimTensor (B, A, D)
                'offset':          SimTensor (B, A, P, T, 2) — refinement offset
        """
        assert self.kmeans_anchors is not None
        num_groups = self.kmeans_anchors.shape[0]

        # Use last decoder layer's track query
        # track_query: (B, num_dec, A, D) → (B, A, D) — select [:, -1]
        track_query = self._select_last_dec(track_query)

        # Positional embedding from tracking centers
        reference_points_track = self._extract_tracking_centers(
            track_bbox_results, self.pc_range)
        # reference_points_track: numpy (B, A, 2) → pos embedding (B, A, D)
        ref_track_t = _reg_tensor(self, F._from_data(f'{self.name}.ref_track',
                                   reference_points_track.astype(np.float32)))
        pos_emb = self.pos2posemb2d(ref_track_t)
        track_query_pos = self.boxes_query_embedding_layer(pos_emb)

        B = track_query.shape[0]
        A = track_query.shape[1]
        P = self.num_anchor
        D = self.embed_dims

        # Learnable query pos: (P*G, D) → split → (G, P, D) → expand
        assert self.learnable_motion_query_embedding_data is not None
        lpq = self.learnable_motion_query_embedding_data  # numpy (P*G, D)
        lpq_split = lpq.reshape(num_groups, P, D)

        # Agent-level anchors: (G, P, T, 2)
        anchors = self.kmeans_anchors.astype(np.float32).reshape(
            num_groups, P, self.predict_steps, 2)

        # Anchor coordinate transforms → positional embeddings (numpy)
        # Embedding layers are called AFTER grouping so their outputs
        # stay connected to the motionformer graph.
        agent_posemb, ego_posemb, offset_posemb, scene_offset_anchors, learnable_emb = \
            self._compute_anchor_embeddings(
                anchors, lpq_split, track_bbox_results, B, A, P, D)

        # Group by class: (B, A, G, P, ...) → (B, A, P, ...)
        agent_posemb = self._group_mode(track_bbox_results, agent_posemb)
        ego_posemb = self._group_mode(track_bbox_results, ego_posemb)
        offset_posemb = self._group_mode(track_bbox_results, offset_posemb)
        learnable_emb = self._group_mode(track_bbox_results, learnable_emb)
        init_reference = self._group_mode_anchors(
            track_bbox_results, scene_offset_anchors)  # (B, A, P, T, 2)

        # Wrap grouped posembs as SimTensors → call embedding layers
        agent_posemb_t = _reg_tensor(self, F._from_data(
            f'{self.name}.agent_posemb', agent_posemb.astype(np.float32)))
        agent_emb_t = self.agent_level_embedding_layer(agent_posemb_t)

        ego_posemb_t = _reg_tensor(self, F._from_data(
            f'{self.name}.ego_posemb', ego_posemb.astype(np.float32)))
        ego_emb_t = self.scene_level_ego_embedding_layer(ego_posemb_t)

        offset_posemb_t = _reg_tensor(self, F._from_data(
            f'{self.name}.offset_posemb', offset_posemb.astype(np.float32)))
        offset_emb_t = self.scene_level_offset_embedding_layer(offset_posemb_t)

        learnable_emb_t = _reg_tensor(self, F._from_data(
            f'{self.name}.learnable_emb', learnable_emb.astype(np.float32)))
        init_reference_t = _reg_tensor(self, F._from_data(
            f'{self.name}.init_reference', init_reference.astype(np.float32)))

        # Unsqueeze track_query: (B, A, D) → (B, A, 1, D) → tile to (B, A, P, D)
        track_query_bc = self.unsq_track_q(track_query, self.unsq_ax2)
        track_query_bc = self.tile_track_q(track_query_bc, self._tile_reps)

        track_query_pos_bc = self.unsq_track_qpos(track_query_pos, self.unsq_ax2)
        track_query_pos_bc = self.tile_track_qpos(track_query_pos_bc, self._tile_reps)

        # Spatial shapes for BEV attention
        spatial_shapes = np.array([[self.bev_h, self.bev_w]], dtype=np.int64)
        level_start_index = np.array([0], dtype=np.int64)

        # Run motionformer
        inter_states, inter_references, offset = self.motionformer(
            track_query,
            lane_query,
            track_query_pos=track_query_pos,
            lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results,
            bev_embed=bev_embed,
            reference_trajs=init_reference_t,
            traj_reg_branches=self.dec_traj_reg_branches,
            traj_cls_branches=None,
            traj_refine_branch=self.traj_refine_branch,
            agent_level_embedding=agent_emb_t,
            scene_level_ego_embedding=ego_emb_t,
            scene_level_offset_embedding=offset_emb_t,
            learnable_embed=learnable_emb_t,
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        # Post-processing per decoder layer
        outputs_traj_scores = []
        outputs_trajs = []
        num_layers = len(inter_states) if isinstance(inter_states, list) else inter_states.shape[0]
        n = self.name

        def _r(op):
            setattr(self, op.name, op)
            op.set_module(self)
            return op

        for lvl in range(num_layers):
            state = inter_states[lvl] if isinstance(inter_states, list) else self._select_level(inter_states, lvl)

            # Classification: (B, A, P, D) → (B, A, P, 1) → squeeze → log_softmax
            cls_out = self.traj_cls_branches[lvl](state)
            cls_squeezed = _r(F.Squeeze(f'{n}.squeeze_cls_{lvl}'))(cls_out, self.unsq_ax3)
            cls_score = _r(F.Log(f'{n}.log_cls_{lvl}'))(_r(F.Softmax(f'{n}.softmax_cls_{lvl}', axis=2))(cls_squeezed))
            outputs_traj_scores.append(cls_score)

            # Regression: (B, A, P, D) → (B, A, P, T*5) → reshape → cumsum → gaussian
            reg_out = self.traj_reg_branches[lvl](state)
            traj = self._unflatten_and_activate(reg_out, lvl, B, A, P)
            outputs_trajs.append(traj)

        outs = {
            'all_traj_scores': outputs_traj_scores,
            'all_traj_preds': outputs_trajs,
            'traj_query': inter_states,
            'track_query': track_query,
            'track_query_pos': track_query_pos,
            'offset': offset
        }
        return outs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_last_dec(self, track_query):
        """Select last decoder layer: track_query[:, -1] → (B, A, D).

        track_query shape: (B, num_dec, A, D).
        Uses TTSim SliceF + Squeeze to stay in graph.
        """
        # Slice [:, -1:, :, :] → (B, 1, A, D)
        sliced = self.sld_slice(track_query, self.sld_starts,
                                self.sld_ends, self.sld_axes)
        # Squeeze axis 1 → (B, A, D)
        return self.sld_squeeze(sliced, self.sld_axes)

    def _select_level(self, stacked, lvl):
        """Select level from stacked tensor: stacked[lvl]."""
        if hasattr(stacked, 'data') and stacked.data is not None:
            t = F._from_data(f'{self.name}.state_{lvl}', stacked.data[lvl])
        else:
            shape = list(stacked.shape)[1:]
            t = F._from_shape(f'{self.name}.state_{lvl}', shape)
        self._tensors[t.name] = t
        return t

    def _compute_anchor_embeddings(self, anchors, lpq_split,
                                   track_bbox_results, B, A, P, D):
        """Compute agent/ego/offset embeddings from anchors.

        All run at numpy level (data-dependent on bbox_results).

        Returns:
            agent_emb:  numpy (B, A, G, P, D)
            ego_emb:    numpy (B, A, G, P, D)
            offset_emb: numpy (B, A, G, P, D)
            scene_offset_anchors: numpy (B, A, G, P, T, 2)
            learnable_emb: numpy (B, A, G, P, D)
        """
        G = self.num_anchor_group
        T = self.predict_steps

        # Agent level: norm → pos2posemb (embedding layer called later in __call__)
        # anchors: (G, P, T, 2)
        agent_norm = self._norm_points_np(anchors[..., -1, :])  # (G, P, 2)
        agent_posemb = self._pos2posemb2d_np(agent_norm)         # (G, P, D)

        # Scene-level transforms per batch (numpy only, no graph ops)
        ego_posemb_list = []
        offset_posemb_list = []
        offset_anchors_list = []

        for i in range(B):
            bboxes, scores, labels, bbox_index, mask = track_bbox_results[i]
            if hasattr(bboxes, 'gravity_center'):
                centers = bboxes.gravity_center.detach().cpu().numpy()
                yaw = bboxes.yaw.detach().cpu().numpy()
            else:
                centers = np.array(bboxes)[:, :3]
                yaw = np.array(bboxes)[:, 6:7]

            n_agents = centers.shape[0]

            # --- scene_level_ego_anchors (translate only) ---
            ego_anchors = self._anchor_transform_np(
                anchors, yaw, centers, n_agents,
                with_rotation=False, with_translation=True)  # (A, G, P, T, 2)
            ego_norm = self._norm_points_np(ego_anchors[..., -1, :])    # (A, G, P, 2)
            ego_posemb = self._pos2posemb2d_np(ego_norm)                # (A, G, P, D)
            ego_posemb_list.append(ego_posemb)

            # --- scene_level_offset_anchors (rotate only, no translate) ---
            offset_anchors = self._anchor_transform_np(
                anchors, yaw, centers, n_agents,
                with_rotation=True, with_translation=False)  # (A, G, P, T, 2)
            offset_norm = self._norm_points_np(offset_anchors[..., -1, :])
            offset_posemb = self._pos2posemb2d_np(offset_norm)
            offset_posemb_list.append(offset_posemb)
            offset_anchors_list.append(offset_anchors)

        # Stack batch
        ego_posemb_all = np.stack(ego_posemb_list)       # (B, A, G, P, D)
        offset_posemb_all = np.stack(offset_posemb_list) # (B, A, G, P, D)
        scene_offset_anchors = np.stack(offset_anchors_list)  # (B, A, G, P, T, 2)

        # Expand agent_posemb: (G, P, D) → (1, 1, G, P, D) → broadcast to (B, A, G, P, D)
        agent_posemb_all = np.broadcast_to(
            agent_posemb[np.newaxis, np.newaxis],
            (B, A, G, P, D)).copy()

        # Learnable embed: (G, P, D) → (1, 1, G, P, D) → broadcast
        learnable_emb = np.broadcast_to(
            lpq_split[np.newaxis, np.newaxis],
            (B, A, G, P, D)).copy()

        return agent_posemb_all, ego_posemb_all, offset_posemb_all, scene_offset_anchors, learnable_emb

    def _norm_points_np(self, pos):
        """Normalize points to [0,1] using pc_range. Pure numpy."""
        x = (pos[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y = (pos[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        return np.stack([x, y], axis=-1)

    def _pos2posemb2d_np(self, pos, num_pos_feats=None, temperature=10000):
        """Pure numpy pos2posemb2d. Returns (*, 2*num_pos_feats) embedding."""
        if num_pos_feats is None:
            num_pos_feats = self.embed_dims // 2
        scale = 2.0 * np.pi
        pos_scaled = pos.astype(np.float32) * scale
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = temperature ** (2.0 * (dim_t // 2) / num_pos_feats)

        pos_x = pos_scaled[..., 0:1] / dim_t  # (*, num_pos_feats)
        pos_y = pos_scaled[..., 1:2] / dim_t

        # Interleave sin/cos
        px_sin = np.sin(pos_x[..., 0::2])
        px_cos = np.cos(pos_x[..., 1::2])
        py_sin = np.sin(pos_y[..., 0::2])
        py_cos = np.cos(pos_y[..., 1::2])

        # Stack + flatten: (*, half, 2) → (*, num_pos_feats)
        half = num_pos_feats // 2
        pos_x_emb = np.zeros(pos_x.shape[:-1] + (num_pos_feats,), dtype=np.float32)
        pos_y_emb = np.zeros_like(pos_x_emb)
        pos_x_emb[..., 0::2] = px_sin
        pos_x_emb[..., 1::2] = px_cos
        pos_y_emb[..., 0::2] = py_sin
        pos_y_emb[..., 1::2] = py_cos

        return np.concatenate([pos_y_emb, pos_x_emb], axis=-1)

    def _anchor_transform_np(self, anchors, yaw, centers, n_agents,
                             with_rotation=True, with_translation=True):
        """Pure numpy anchor coordinate transform.

        Args:
            anchors: (G, P, T, 2)
            yaw: (A, 1) angles
            centers: (A, 3) xyz
            n_agents: int

        Returns:
            (A, G, P, T, 2) transformed anchors
        """
        transformed = anchors[np.newaxis].copy()  # (1, G, P, T, 2)

        if with_rotation:
            angle = yaw.flatten() - 3.1415953
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            # rot_mat: (A, 2, 2)
            rot_mat = np.zeros((n_agents, 2, 2), dtype=np.float32)
            rot_mat[:, 0, 0] = cos_a
            rot_mat[:, 0, 1] = -sin_a
            rot_mat[:, 1, 0] = sin_a
            rot_mat[:, 1, 1] = cos_a
            # (A, 1, 1, 2, 2) @ (1, G, P, 2, T) → (A, G, P, 2, T)
            rm = rot_mat[:, np.newaxis, np.newaxis]
            t = np.transpose(transformed, (0, 1, 2, 4, 3))
            result = np.matmul(rm, t)
            transformed = np.transpose(result, (0, 1, 2, 4, 3))

        if with_translation:
            c = centers[:, :2].reshape(n_agents, 1, 1, 1, 2)
            transformed = c + transformed

        return transformed  # (A, G, P, T, 2)

    def _group_mode(self, bbox_results, data):
        """Group embeddings by class: (B, A, G, P, D) → (B, A, P, D).

        Uses cls2group to select the correct group for each agent.
        """
        batch_size = data.shape[0]
        A = data.shape[1]
        batched = []
        for i in range(batch_size):
            _, _, labels, _, _ = bbox_results[i]
            if hasattr(labels, 'numpy'):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.array(labels)
            grouped = []
            for j in range(A):
                g = self.cls2group[labels_np[j]]
                grouped.append(data[i, j, g])
            batched.append(np.stack(grouped))
        return np.stack(batched)  # (B, A, P, ...)

    def _group_mode_anchors(self, bbox_results, data):
        """Group anchors by class: (B, A, G, P, T, 2) → (B, A, P, T, 2)."""
        return self._group_mode(bbox_results, data)

    def _unflatten_and_activate(self, reg_out, lvl, B, A, P):
        """Unflatten regression output, apply cumsum and gaussian activation.

        reg_out: SimTensor (B, A, P, predict_steps*5)
        Returns: SimTensor (B, A, P, predict_steps, 5) with activated outputs.
        Creates unique ops per lvl (SimOpHandles can't be reused).
        """
        T = self.predict_steps
        n = self.name

        def r(op):
            setattr(self, op.name, op)
            op.set_module(self)
            return op

        # Reshape: (B, A, P, T*5) → (B, A, P, T, 5)
        new_shape = _reg_tensor(self, F._from_data(f'{n}.uf_shape_{lvl}',
                                 np.array([B, A, P, T, 5], dtype=np.int64),
                                 is_const=True))
        reshaped = r(F.Reshape(f'{n}.uf_reshape_{lvl}'))(reg_out, new_shape)

        # Slice xy channels: [..., :2] → (B, A, P, T, 2)
        xy = r(F.SliceF(f'{n}.uf_slice_xy_{lvl}'))(reshaped, self.uf_s0, self.uf_e2, self.uf_ax4)
        # CumSum along time axis (axis=3) for trajectory
        xy_cum = r(F.BinaryOperator(f'{n}.uf_cumsum_{lvl}', optype='CumSum'))(xy, self.uf_cumsum_ax)

        # Slice sig_x: [..., 2:3] → (B, A, P, T, 1); apply exp
        sigx = r(F.Exp(f'{n}.uf_exp_sigx_{lvl}'))(
            r(F.SliceF(f'{n}.uf_slice_sigx_{lvl}'))(reshaped, self.uf_s2, self.uf_e3, self.uf_ax4))

        # Slice sig_y: [..., 3:4] → (B, A, P, T, 1); apply exp
        sigy = r(F.Exp(f'{n}.uf_exp_sigy_{lvl}'))(
            r(F.SliceF(f'{n}.uf_slice_sigy_{lvl}'))(reshaped, self.uf_s3, self.uf_e4, self.uf_ax4))

        # Slice rho: [..., 4:5] → (B, A, P, T, 1); apply tanh
        rho = r(F.Tanh(f'{n}.uf_tanh_rho_{lvl}'))(
            r(F.SliceF(f'{n}.uf_slice_rho_{lvl}'))(reshaped, self.uf_s4, self.uf_e5, self.uf_ax4))

        # Concat: (B, A, P, T, 2+1+1+1) = (B, A, P, T, 5)
        return r(F.ConcatX(f'{n}.uf_cat_{lvl}', axis=4))(xy_cum, sigx, sigy, rho)

    def analytical_param_count(self):
        """Total parameter count for MotionHead."""
        total = 0
        D = self.embed_dims

        # Motionformer (decoder)
        total += self.motionformer.analytical_param_count()

        # Embedding layers (4 TwoLayerMLPs: D→D*2→D)
        for layer in [self.agent_level_embedding_layer,
                      self.scene_level_ego_embedding_layer,
                      self.scene_level_offset_embedding_layer,
                      self.boxes_query_embedding_layer]:
            total += layer.analytical_param_count()

        # Track query fuser
        total += self.layer_track_query_fuser.analytical_param_count()

        # Traj cls/reg branches
        for br in self.traj_cls_branches:
            total += br.analytical_param_count()  # type: ignore[attr-defined]
        for br in self.traj_reg_branches:
            total += br.analytical_param_count()  # type: ignore[attr-defined]
        total += self.traj_refine_branch.analytical_param_count()

        # Learnable embedding (like nn.Embedding)
        total += self.num_anchor * self.num_anchor_group * D

        # BEV interaction layers
        if self.motionformer.bev_interaction_layers is not None:
            for layer in self.motionformer.bev_interaction_layers:
                total += layer.analytical_param_count()  # type: ignore[attr-defined]
        if self.motionformer.bev_interaction_rl is not None:
            total += self.motionformer.bev_interaction_rl.analytical_param_count()  # type: ignore[attr-defined]

        # Functional modules (Pos2PosEmb2D, NormPoints, etc.) have 0 params
        return total


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    import traceback

    logger.info("MotionHead — TTSim (FusionAD)")
    logger.info("=" * 70)

    D = 64
    H = 8
    P = 6
    T = 12
    num_layers = 3
    bev_h, bev_w = 50, 50
    NUM_LEVELS = 1
    NUM_POINTS = 4
    num_classes = 10
    group_id_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ok = True

    # Build transformer layer config (for BEV interaction layers)
    transformerlayers_cfg = dict(
        attn_cfgs=[dict(
            type='MotionDeformableAttention',
            num_steps=T,
            embed_dims=D,
            num_levels=NUM_LEVELS,
            num_heads=H,
            num_points=NUM_POINTS,
            sample_index=-1,
            bev_range=pc_range)],
        ffn_cfgs=dict(
            type='FFN', embed_dims=D, feedforward_channels=D * 2,
            num_fcs=2, ffn_drop=0.0,
            act_cfg=dict(type='ReLU', inplace=True)),
        operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
        embed_dims=D)

    try:
        mh = MotionHead(
            'test_mh',
            predict_steps=T,
            transformerlayers=transformerlayers_cfg,
            num_cls_fcs=2,
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=D,
            num_anchor=P,
            det_layer_num=6,
            group_id_list=group_id_list,
            pc_range=pc_range,
            anchor_info_path=None,
            num_classes=num_classes,
            num_layers=num_layers)

        # Set fake anchors (normally loaded from pickle)
        G = len(group_id_list)
        mh.kmeans_anchors = np.random.randn(G, P, T, 2).astype(np.float32)

        # Set fake learnable embedding data
        mh.learnable_motion_query_embedding_data = np.random.randn(
            P * G, D).astype(np.float32)

        pc = mh.analytical_param_count()
        logger.info(f"[OK] MotionHead construction  params={pc:,}")

    except Exception as e:
        logger.info(f"[X]  MotionHead construction FAILED: {e}")
        traceback.print_exc()
        ok = False

    logger.info("=" * 70)
    logger.info("ALL OK" if ok else "SOME FAILURES")
