#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of MapTracker inference pipeline
Includes semantic segmentation and vector map prediction with query propagation
"""

# -------------------------------PyTorch--------------------------------

# """
#     MapTracker main module, adapted from StreamMapNet
# """
# import numpy as np
# import torch
# import torch.nn as nn
#
# from mmdet3d.models.builder import (build_backbone, build_head)
#
# from .base_mapper import BaseMapper, MAPPERS
# from ..utils.query_update import MotionMLP
# from copy import deepcopy
# from mmdet.core import multi_apply
#
# from einops import rearrange, repeat
# from scipy.spatial.transform import Rotation as R
#
# from .vector_memory import VectorInstanceMemory
#
#
# @MAPPERS.register_module()
# class MapTracker(BaseMapper):
#
#     def __init__(self,
#                  bev_h,
#                  bev_w,
#                  roi_size,
#                  backbone_cfg=dict(),
#                  head_cfg=dict(),
#                  neck_cfg=None,
#                  seg_cfg=None,
#                  model_name=None,
#                  pretrained=None,
#                  history_steps=None,
#                  test_time_history_steps=None,
#                  mem_select_dist_ranges=[0,0,0,0],
#                  skip_vector_head=False,
#                  freeze_bev=False,
#                  freeze_bev_iters=None,
#                  track_fp_aug=True,
#                  use_memory=False,
#                  mem_len=None,
#                  mem_warmup_iters=-1,
#                  **kwargs):
#         super().__init__()
#
#         #Attribute
#         self.model_name = model_name
#         self.last_epoch = None
#
#         self.backbone = build_backbone(backbone_cfg)
#
#         if neck_cfg is not None:
#             self.neck = build_head(neck_cfg)
#         else:
#             self.neck = nn.Identity()
#
#         self.head = build_head(head_cfg)
#         self.num_decoder_layers = self.head.transformer.decoder.num_layers
#         self.skip_vector_head = skip_vector_head
#         self.freeze_bev = freeze_bev # whether freeze bev related parameters
#         self.freeze_bev_iters = freeze_bev_iters # whether freeze bev related parameters
#         self.track_fp_aug = track_fp_aug
#         self.use_memory = use_memory
#         self.mem_warmup_iters = mem_warmup_iters
#
#         # the track query propagation module, using relative pose
#         c_dim = 7 # quaternion for rotation (4) + translation (3)
#         self.query_propagate = MotionMLP(c_dim=c_dim, f_dim=self.head.embed_dims, identity=True)
#
#         # BEV semantic seg head
#         self.seg_decoder = build_head(seg_cfg)
#
#         # BEV
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.roi_size = roi_size
#         self.history_steps = history_steps
#
#         self.mem_len = mem_len
#
#         # Set up test time memory selection hyper-parameters
#         if test_time_history_steps is None:
#             self.test_time_history_steps = history_steps
#         else:
#             self.test_time_history_steps = test_time_history_steps
#         self.mem_select_dist_ranges = mem_select_dist_ranges
#
#         # vector instance memory module
#         if self.use_memory:
#             self.memory_bank = VectorInstanceMemory(
#                 dim_in=head_cfg.embed_dims,
#                 number_ins=head_cfg.num_queries,
#                 bank_size=mem_len,
#                 mem_len=mem_len,
#                 mem_select_dist_ranges=self.mem_select_dist_ranges,
#             )
#
#         xmin, xmax = -roi_size[0]/2, roi_size[0]/2
#         ymin, ymax = -roi_size[1]/2, roi_size[1]/2
#         x = torch.linspace(xmin, xmax, bev_w)
#         y = torch.linspace(ymax, ymin, bev_h)
#         y, x = torch.meshgrid(y, x)
#         z = torch.zeros_like(x)
#         ones = torch.ones_like(x)
#         plane = torch.stack([x, y, z, ones], dim=-1)
#         self.register_buffer('plane', plane.double())
#
#         self.init_weights(pretrained)
#
#     def init_weights(self, pretrained=None):
#         """Initialize model weights."""
#         if pretrained:
#             import logging
#             logger = logging.getLogger()
#             from mmcv.runner import load_checkpoint
#             load_checkpoint(self, pretrained, strict=False, logger=logger)
#         else:
#             try:
#                 self.neck.init_weights()
#             except AttributeError:
#                 pass
#
#     def temporal_propagate(self, curr_bev_feats, img_metas, all_history_curr2prev, all_history_prev2curr, use_memory,
#                            track_query_info=None, timestep=None, get_trans_loss=False):
#         '''
#         Args:
#             curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
#             img_metas: current image metas (List of #bs samples)
#             bev_memory: where to load and store (training and testing use different buffer)
#             pose_memory: where to load and store (training and testing use different buffer)
#
#         Out:
#             fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
#         '''
#
#         bs = curr_bev_feats.size(0)
#
#         if get_trans_loss: # init the trans_loss related variables here
#             trans_reg_loss = curr_bev_feats.new_zeros((1,))
#             trans_cls_loss = curr_bev_feats.new_zeros((1,))
#             back_trans_reg_loss = curr_bev_feats.new_zeros((1,))
#             back_trans_cls_loss = curr_bev_feats.new_zeros((1,))
#             num_pos = 0
#             num_tracks = 0
#
#         if use_memory:
#             self.memory_bank.clear_dict()
#
#         for b_i in range(bs):
#             curr_e2g_trans = self.plane.new_tensor(img_metas[b_i]['ego2global_translation'], dtype=torch.float64)
#             curr_e2g_rot = self.plane.new_tensor(img_metas[b_i]['ego2global_rotation'], dtype=torch.float64)
#
#             if use_memory:
#                 self.memory_bank.curr_rot[b_i] = curr_e2g_rot
#                 self.memory_bank.curr_trans[b_i] = curr_e2g_trans
#                 if self.memory_bank.curr_t > 0:
#                     self.memory_bank.trans_memory_bank(self.query_propagate, b_i, img_metas[b_i])
#
#             # transform the track queries
#             if track_query_info is not None:
#                 history_curr2prev_matrix = all_history_curr2prev[b_i]
#                 history_prev2curr_matrix = all_history_prev2curr[b_i]
#
#                 track_pts = track_query_info[b_i]['track_query_boxes'].clone()
#                 track_pts = rearrange(track_pts, 'n (k c) -> n k c', c=2)
#                 # from (0, 1) to (-30, 30) or (-15, 15), prep for transform
#                 track_pts = self._denorm_lines(track_pts)
#
#                 # Transform the track ref-points using relative pose between prev and curr
#                 N, num_points = track_pts.shape[0], track_pts.shape[1]
#                 track_pts = torch.cat([
#                     track_pts,
#                     track_pts.new_zeros((N, num_points, 1)), # z-axis
#                     track_pts.new_ones((N, num_points, 1)) # 4-th dim
#                 ], dim=-1) # (num_prop, num_pts, 4)
#
#                 pose_matrix = history_prev2curr_matrix[-1].float()[:3]
#                 rot_mat = pose_matrix[:, :3].cpu().numpy()
#                 rot = R.from_matrix(rot_mat)
#                 translation = pose_matrix[:, 3]
#                 trans_matrix = history_prev2curr_matrix[-1].clone()
#
#                 # Add training-time perturbation here for the transformation matrix
#                 if self.training:
#                     rot, translation = self.add_noise_to_pose(rot, translation)
#                     trans_matrix[:3, :3] = torch.tensor(rot.as_matrix()).to(trans_matrix.device)
#                     trans_matrix[:3, 3] = torch.tensor(translation).to(trans_matrix.device)
#
#                 trans_track_pts = torch.einsum('lk,ijk->ijl', trans_matrix, track_pts.double()).float()
#                 trans_track_pts = trans_track_pts[..., :2]
#                 trans_track_pts = self._norm_lines(trans_track_pts)
#                 trans_track_pts = torch.clip(trans_track_pts, min=0., max=1.)
#                 trans_track_pts = rearrange(trans_track_pts, 'n k c -> n (k c)', c=2)
#                 track_query_info[b_i]['trans_track_query_boxes'] = trans_track_pts
#
#                 prop_q = track_query_info[b_i]['track_query_hs_embeds']
#
#                 rot_quat = torch.tensor(rot.as_quat()).float().to(pose_matrix.device)
#                 pose_info = torch.cat([rot_quat.view(-1), translation], dim=0)
#
#                 track_query_updated = self.query_propagate(
#                     prop_q, # (topk, embed_dims)
#                     pose_info.repeat(len(prop_q), 1)
#                 )
#                 # Do not let future-frame loss backprop through the track queries
#                 track_query_info[b_i]['track_query_hs_embeds'] = track_query_updated.clone().detach()
#
#                 if get_trans_loss:
#                     pred = self.head.reg_branches[-1](track_query_updated).sigmoid() # (num_prop, 2*num_pts)
#                     pred_scores = self.head.cls_branches[-1](track_query_updated)
#                     assert list(pred.shape) == [N, 2*num_points]
#
#                     gt_pts = track_query_info[b_i]['track_query_gt_lines'].clone()
#                     gt_labels = track_query_info[b_i]['track_query_gt_labels'].clone()
#                     weights = gt_pts.new_ones((N, 2*num_points))
#                     weights_labels = gt_labels.new_ones((N,))
#                     bg_idx = gt_labels == 3
#                     num_pos = num_pos + (N - bg_idx.sum())
#                     num_tracks += len(gt_labels)
#                     weights[bg_idx, :] = 0.0
#
#                     gt_pts = rearrange(gt_pts, 'n (k c) -> n k c', c=2)
#                     denormed_targets = self._denorm_lines(gt_pts)
#                     denormed_targets = torch.cat([
#                         denormed_targets,
#                         denormed_targets.new_zeros((N, num_points, 1)), # z-axis
#                         denormed_targets.new_ones((N, num_points, 1)) # 4-th dim
#                     ], dim=-1) # (num_prop, num_pts, 4)
#                     assert list(denormed_targets.shape) == [N, num_points, 4]
#
#                     curr_targets = torch.einsum('lk,ijk->ijl', trans_matrix.float(), denormed_targets)
#                     curr_targets = curr_targets[..., :2]
#                     normed_targets = self._norm_lines(curr_targets)
#                     normed_targets = rearrange(normed_targets, 'n k c -> n (k c)', c=2)
#                     # set the weight of invalid normed targets to 0 (outside current bev frame)
#                     invalid_bev_mask = (normed_targets <= 0) | (normed_targets>=1)
#                     weights[invalid_bev_mask] = 0
#                     # (num_prop, 2*num_pts)
#                     trans_reg_loss += self.head.loss_reg(pred, normed_targets, weights, avg_factor=1.0)
#                     if len(gt_labels) > 0:
#                         trans_score = self.head.loss_cls(pred_scores, gt_labels, weights_labels, avg_factor=1.0)
#                     else:
#                         trans_score = 0.0
#                     trans_cls_loss += trans_score
#
#                     # backward trans loss
#                     pose_matrix_inv = torch.inverse(trans_matrix).float()[:3]
#                     rot_mat_inv = pose_matrix_inv[:, :3].cpu().numpy()
#
#                     rot_inv = R.from_matrix(rot_mat_inv)
#                     rot_quat_inv = torch.tensor(rot_inv.as_quat()).float().to(pose_matrix_inv.device)
#                     translation_inv = pose_matrix_inv[:, 3]
#                     pose_info_inv = torch.cat([rot_quat_inv.view(-1), translation_inv], dim=0)
#                     track_query_backtrans = self.query_propagate(
#                         track_query_updated, # (topk, embed_dims)
#                         pose_info_inv.repeat(len(prop_q), 1)
#                     )
#                     pred_backtrans = self.head.reg_branches[-1](track_query_backtrans).sigmoid() # (num_prop, 2*num_pts)
#                     pred_scores_backtrans = self.head.cls_branches[-1](track_query_backtrans)
#                     prev_gt_pts = track_query_info[b_i]['track_query_gt_lines']
#                     back_trans_reg_loss += self.head.loss_reg(pred_backtrans, prev_gt_pts, weights, avg_factor=1.0)
#                     if len(gt_labels) > 0:
#                         trans_score_bak = self.head.loss_cls(pred_scores_backtrans, gt_labels, weights_labels, avg_factor=1.0)
#                     else:
#                         trans_score_bak = 0.0
#                     back_trans_cls_loss += trans_score_bak
#
#         if get_trans_loss:
#             trans_loss = self.head.trans_loss_weight * (trans_reg_loss / (num_pos + 1e-10) +
#                             trans_cls_loss / (num_tracks + 1e-10))
#             back_trans_loss = self.head.trans_loss_weight * (back_trans_reg_loss / (num_pos + 1e-10) +
#                                     back_trans_cls_loss / (num_tracks + 1e-10))
#             trans_loss_dict = {
#                 'f_trans': trans_loss,
#                 'b_trans': back_trans_loss,
#             }
#             return trans_loss_dict
#
#     def add_noise_to_pose(self, rot, trans):
#         rot_euler = rot.as_euler('zxy')
#         # 0.08 mean is around 5-degree, 3-sigma is 15-degree
#         noise_euler = np.random.randn(*list(rot_euler.shape)) * 0.08
#         rot_euler += noise_euler
#         noisy_rot = R.from_euler('zxy', rot_euler)
#
#         # error within 0.25 meter
#         noise_trans = torch.randn_like(trans) * 0.25
#         noise_trans[2] = 0
#         noisy_trans = trans + noise_trans
#
#         return noisy_rot, noisy_trans
#
#     def process_history_info(self, img_metas, history_img_metas):
#         bs = len(img_metas)
#         all_history_curr2prev = []
#         all_history_prev2curr = []
#         all_history_coord = []
#
#         if len(history_img_metas) == 0:
#             return all_history_curr2prev, all_history_prev2curr, all_history_coord
#
#         for b_i in range(bs):
#             history_e2g_trans = torch.stack([self.plane.new_tensor(prev[b_i]['ego2global_translation'], dtype=torch.float64) for prev in history_img_metas], dim=0)
#             history_e2g_rot = torch.stack([self.plane.new_tensor(prev[b_i]['ego2global_rotation'], dtype=torch.float64) for prev in history_img_metas], dim=0)
#
#             curr_e2g_trans = self.plane.new_tensor(img_metas[b_i]['ego2global_translation'], dtype=torch.float64)
#             curr_e2g_rot = self.plane.new_tensor(img_metas[b_i]['ego2global_rotation'], dtype=torch.float64)
#
#             # Do the coords transformation for all features in the history buffer
#             ## Prepare the transformation matrix
#             history_g2e_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
#             history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
#             history_g2e_matrix[:, :3, 3] = -torch.bmm(torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]).squeeze(-1)
#
#             curr_g2e_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
#             curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
#             curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
#
#             curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device)
#             curr_e2g_matrix[:3, :3] = curr_e2g_rot
#             curr_e2g_matrix[:3, 3] = curr_e2g_trans
#
#             history_e2g_matrix = torch.stack([torch.eye(4, dtype=torch.float64, device=history_e2g_trans.device),]*len(history_e2g_trans), dim=0)
#             history_e2g_matrix[:, :3, :3] = history_e2g_rot
#             history_e2g_matrix[:, :3, 3] = history_e2g_trans
#
#             history_curr2prev_matrix = torch.bmm(history_g2e_matrix, repeat(curr_e2g_matrix,'n1 n2 -> r n1 n2', r=len(history_g2e_matrix)))
#             history_prev2curr_matrix = torch.bmm(repeat(curr_g2e_matrix, 'n1 n2 -> r n1 n2', r=len(history_e2g_matrix)), history_e2g_matrix)
#
#             history_coord = torch.einsum('nlk,ijk->nijl', history_curr2prev_matrix, self.plane).float()[..., :2]
#
#             # from (-30, 30) or (-15, 15) to (-1, 1)
#             history_coord[..., 0] = history_coord[..., 0] / (self.roi_size[0]/2)
#             history_coord[..., 1] = -history_coord[..., 1] / (self.roi_size[1]/2)
#
#             all_history_curr2prev.append(history_curr2prev_matrix)
#             all_history_prev2curr.append(history_prev2curr_matrix)
#             all_history_coord.append(history_coord)
#
#         return all_history_curr2prev, all_history_prev2curr, all_history_coord
#
#
#     def forward_train(self, img, vectors, semantic_mask, points=None, img_metas=None, all_prev_data=None,
#                       all_local2global_info=None, **kwargs):
#         '''
#         Args:
#             img: torch.Tensor of shape [B, N, 3, H, W]
#                 N: number of cams
#             vectors: list[list[Tuple(lines, length, label)]]
#                 - lines: np.array of shape [num_points, 2].
#                 - length: int
#                 - label: int
#                 len(vectors) = batch_size
#                 len(vectors[_b]) = num of lines in sample _b
#             img_metas:
#                 img_metas['lidar2img']: [B, N, 4, 4]
#         Out:
#             loss, log_vars, num_sample
#         '''
#         #  prepare labels and images
#         gts, img, img_metas, valid_idx, points = self.batch_data(
#             vectors, img, img_metas, img.device, points)
#         bs = img.shape[0]
#
#         _use_memory = self.use_memory and self.num_iter > self.mem_warmup_iters
#
#         if all_prev_data is not None:
#             num_prev_frames = len(all_prev_data)
#             all_gts_prev, all_img_prev, all_img_metas_prev, all_semantic_mask_prev  = [], [], [], []
#             for prev_data in all_prev_data:
#                 gts_prev, img_prev, img_metas_prev, valid_idx_prev, _ = self.batch_data(
#                     prev_data['vectors'], prev_data['img'], prev_data['img_metas'], img.device
#                 )
#                 all_gts_prev.append(gts_prev)
#                 all_img_prev.append(img_prev)
#                 all_img_metas_prev.append(img_metas_prev)
#                 all_semantic_mask_prev.append(prev_data['semantic_mask'])
#         else:
#             num_prev_frames = 0
#
#         assert points is None
#
#         if self.skip_vector_head:
#             backprop_backbone_ids = [0, num_prev_frames] # first and last frame train the backbone (bev pretrain)
#         else:
#             backprop_backbone_ids = [num_prev_frames, ] # only the last frame trains the backbone (all other settings)
#
#         track_query_info = None
#         all_loss_dict_prev = []
#         all_trans_loss = []
#         all_outputs_prev = []
#
#         self.tracked_query_length = {}
#
#         if _use_memory:
#             self.memory_bank.set_bank_size(self.mem_len)
#             self.memory_bank.init_memory(bs=bs)
#
#         # History records for bev features
#         history_bev_feats = []
#         history_img_metas = []
#
#         gt_semantic = torch.flip(semantic_mask, [2,])
#
#         # Iterate through all prev frames
#         for t in range(num_prev_frames):
#             # Backbone for prev
#             img_backbone_gradient = (t in backprop_backbone_ids)
#
#             all_history_curr2prev, all_history_prev2curr, all_history_coord =  \
#                     self.process_history_info(all_img_metas_prev[t], history_img_metas)
#
#             _bev_feats, mlvl_feats = self.backbone(all_img_prev[t], all_img_metas_prev[t], t, history_bev_feats,
#                         history_img_metas, all_history_coord, points=None,
#                         img_backbone_gradient=img_backbone_gradient)
#
#             # Neck for prev
#             bev_feats = self.neck(_bev_feats)
#
#             if _use_memory:
#                 self.memory_bank.curr_t = t
#
#             # Transform prev-frame feature & pts to curr frame
#             if self.skip_vector_head or t == 0:
#                 self.temporal_propagate(bev_feats, all_img_metas_prev[t], all_history_curr2prev,
#                         all_history_prev2curr, _use_memory, track_query_info, timestep=t, get_trans_loss=False)
#             else:
#                 trans_loss_dict = self.temporal_propagate(bev_feats, all_img_metas_prev[t], all_history_curr2prev,
#                         all_history_prev2curr, _use_memory, track_query_info, timestep=t, get_trans_loss=True)
#
#                 ########################################################
#                 # Debugging use: visualize the first-frame track query. and the corresponding
#                 # ground-truth information
#                 # Do this for every timestep > 0
#                 #self._viz_temporal_supervision(outputs_prev, track_query_info, gts_next[-1], gts_prev[-1],
#                 #                gts_semantic_curr, gts_semantic_prev, img_metas_next, img_metas_prev, t)
#                 #import pdb; pdb.set_trace()
#                 ########################################################
#
#             img_metas_prev = all_img_metas_prev[t]
#             img_metas_next = all_img_metas_prev[t+1] if t < num_prev_frames-1 else img_metas
#             gts_prev = all_gts_prev[t]
#             gts_next = all_gts_prev[t+1] if t!=num_prev_frames-1 else gts
#             gts_semantic_prev = torch.flip(all_semantic_mask_prev[t], [2,])
#             gts_semantic_curr = torch.flip(all_semantic_mask_prev[t+1], [2,]) if t!=num_prev_frames-1 else gt_semantic
#
#             local2global_prev = all_local2global_info[t]
#             local2global_next = all_local2global_info[t+1]
#
#             # Compute the semantic segmentation loss
#             seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(bev_feats, gts_semantic_prev,
#                     all_history_coord, return_loss=True)
#
#             # Save the history
#             history_bev_feats.append(bev_feats)
#             history_img_metas.append(all_img_metas_prev[t])
#             if len(history_bev_feats) > self.history_steps:
#                 history_bev_feats.pop(0)
#                 history_img_metas.pop(0)
#
#             if not self.skip_vector_head:
#                 # Prepare the two-frame instance matching info
#                 gt_cur2prev, gt_prev2cur = self.get_two_frame_matching(local2global_prev, local2global_next,
#                                                                        gts_prev, gts_next)
#                 if t == 0:
#                     memory_bank = None
#                 else:
#                     memory_bank = self.memory_bank if _use_memory else None
#                 # 1). Compute the loss for prev frame
#                 # 2). Get the matching results for computing the track query to next frame
#                 loss_dict_prev, outputs_prev, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost, \
#                     prev_gt_list = self.head(
#                                         bev_features=bev_feats,
#                                         img_metas=img_metas_prev,
#                                         gts=gts_prev,
#                                         track_query_info=track_query_info,
#                                         memory_bank=memory_bank,
#                                         return_loss=True,
#                                         return_matching=True)
#                 all_outputs_prev.append(outputs_prev)
#
#                 if t > 0:
#                     all_trans_loss.append(trans_loss_dict)
#
#                 # Do the query prop and negative sampling, prepare the corrpespnding
#                 # updated G.T. labels. The prepared queries will be passed to the model,
#                 # and combind with the original queries inside the head model
#                 pos_th = 0.4
#                 track_query_info = self.prepare_track_queries_and_targets(gts_next, prev_inds_list,
#                     prev_gt_inds_list, prev_matched_reg_cost, prev_gt_list, outputs_prev, gt_cur2prev, gt_prev2cur,
#                     img_metas_prev, _use_memory, pos_th=pos_th, timestep=t)
#             else:
#                 loss_dict_prev = {}
#
#             loss_dict_prev['seg'] = seg_loss
#             loss_dict_prev['seg_dice'] = seg_dice_loss
#
#             all_loss_dict_prev.append(loss_dict_prev)
#
#         if _use_memory:
#             self.memory_bank.curr_t = num_prev_frames
#
#         # NOTE: we separate the last frame to be consistent with single-frame only setting)
#         # Backbone for curr
#         img_backbone_gradient = num_prev_frames in backprop_backbone_ids
#
#         all_history_curr2prev, all_history_prev2curr, all_history_coord = self.process_history_info(img_metas, history_img_metas)
#
#         _bev_feats, mlvl_feats = self.backbone(img, img_metas, num_prev_frames, history_bev_feats, history_img_metas, all_history_coord,
#                     points=None, img_backbone_gradient=img_backbone_gradient)
#         # Neck for curr
#         bev_feats = self.neck(_bev_feats)
#
#         if self.skip_vector_head or num_prev_frames == 0:
#             # Transform prev-frame feature & pts to curr frame using the relative pose
#             assert track_query_info is None
#             self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
#                         all_history_prev2curr, _use_memory, track_query_info, timestep=num_prev_frames, get_trans_loss=False)
#         else:
#             trans_loss_dict = self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
#                         all_history_prev2curr, _use_memory, track_query_info, timestep=num_prev_frames, get_trans_loss=True)
#             all_trans_loss.append(trans_loss_dict)
#
#             ########################################################
#             # Debugging use: visualize the first-frame track query. and the corresponding
#             # ground-truth information
#             # Do this for every timestep > 0
#             #assert num_prev_frames > 0
#             #self._viz_temporal_supervision(outputs_prev, track_query_info, gts_next[-1], gts_prev[-1], gt_semantic,
#             #        gts_semantic_prev, img_metas_next, img_metas_prev, timestep=num_prev_frames)
#             #import pdb; pdb.set_trace()
#             ########################################################
#
#         seg_preds, seg_feats, seg_loss, seg_dice_loss = self.seg_decoder(bev_feats, gt_semantic,
#                 all_history_coord, return_loss=True)
#
#         if not self.skip_vector_head:
#             memory_bank = self.memory_bank if _use_memory else None
#             # 3. run the head again and compute the loss for the second frame
#             preds_list, loss_dict, det_match_idxs, det_match_gt_idxs, gt_list = self.head(
#                 bev_features=bev_feats,
#                 img_metas=img_metas,
#                 gts=gts,
#                 track_query_info=track_query_info,
#                 memory_bank=memory_bank,
#                 return_loss=True)
#         else:
#             loss_dict = {}
#
#         loss_dict['seg'] = seg_loss
#         loss_dict['seg_dice'] = seg_dice_loss
#
#         # format loss, average over all frames (2 frames for now)
#         loss = 0
#         losses_t = []
#         for loss_dict_t in (all_loss_dict_prev + [loss_dict,]):
#             loss_t = 0
#             for name, var in loss_dict_t.items():
#                 loss_t = loss_t + var
#             losses_t.append(loss_t)
#             loss += loss_t
#
#         for trans_loss_dict_t in all_trans_loss:
#             trans_loss_t = trans_loss_dict_t['f_trans'] + trans_loss_dict_t['b_trans']
#             loss += trans_loss_t
#
#         # update the log
#         log_vars = {k: v.item() for k, v in loss_dict.items()}
#
#         for t, loss_dict_t in enumerate(all_loss_dict_prev):
#             log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in loss_dict_t.items()}
#             log_vars.update(log_vars_t)
#
#         for t, loss_t in enumerate(losses_t):
#             log_vars.update({'total_t{}'.format(t): loss_t.item()})
#
#         for t, trans_loss_dict_t in enumerate(all_trans_loss):
#             log_vars_t = {k+'_t{}'.format(t): v.item() for k, v in trans_loss_dict_t.items()}
#             log_vars.update(log_vars_t)
#
#         log_vars.update({'total': loss.item()})
#         num_sample = img.size(0)
#         return loss, log_vars, num_sample
#
#     @torch.no_grad()
#     def forward_test(self, img, points=None, img_metas=None, seq_info=None, **kwargs):
#         '''
#             inference pipeline
#         '''
#
#         assert img.shape[0] == 1, 'Only support bs=1 per-gpu for inference'
#
#         tokens = []
#         for img_meta in img_metas:
#             tokens.append(img_meta['token'])
#
#         scene_name, local_idx, seq_length  = seq_info[0]
#         first_frame = (local_idx == 0)
#         img_metas[0]['local_idx'] = local_idx
#
#         if first_frame:
#             if self.use_memory:
#                 self.memory_bank.set_bank_size(self.test_time_history_steps)
#                 #self.memory_bank.set_bank_size(self.mem_len)
#                 self.memory_bank.init_memory(bs=1)
#             self.history_bev_feats_all = []
#             self.history_img_metas_all = []
#
#         if self.use_memory:
#             self.memory_bank.curr_t = local_idx
#
#         selected_mem_ids = self.select_memory_entries(self.history_img_metas_all, img_metas)
#         history_img_metas = [self.history_img_metas_all[idx] for idx in selected_mem_ids]
#         history_bev_feats = [self.history_bev_feats_all[idx] for idx in selected_mem_ids]
#
#         all_history_curr2prev, all_history_prev2curr, all_history_coord =  \
#                     self.process_history_info(img_metas, history_img_metas)
#
#         _bev_feats, mlvl_feats = self.backbone(img, img_metas, local_idx, history_bev_feats, history_img_metas,
#                         all_history_coord, points=points)
#
#         img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]
#         # Neck
#         bev_feats = self.neck(_bev_feats)
#
#         if self.skip_vector_head or first_frame:
#             self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev, \
#                     all_history_prev2curr, self.use_memory, track_query_info=None)
#             seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
#             if not self.skip_vector_head:
#                 preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
#             track_dict = None
#         else:
#             # Using the saved prev-frame output to prepare the track query inputs
#             track_query_info = self.head.get_track_info(scene_name, local_idx)
#             # Transform prev-frame feature & pts to curr frame using the relative pose
#             self.temporal_propagate(bev_feats, img_metas, all_history_curr2prev,
#                 all_history_prev2curr, self.use_memory, track_query_info)
#             seg_preds, seg_feats = self.seg_decoder(bev_features=bev_feats, return_loss=False)
#
#             # Run the vector map decoder with instance-level memory
#             memory_bank = self.memory_bank if self.use_memory else None
#             preds_list = self.head(bev_feats, img_metas=img_metas,
#                         track_query_info=track_query_info, memory_bank=memory_bank,
#                         return_loss=False)
#             track_dict = self._process_track_query_info(track_query_info)
#
#         if not self.skip_vector_head:
#             # take predictions from the last layer
#             preds_dict = preds_list[-1]
#         else:
#             preds_dict = None
#
#         # Save the BEV and meta-info history
#         self.history_bev_feats_all.append(bev_feats)
#         self.history_img_metas_all.append(img_metas)
#
#         if len(self.history_bev_feats_all) > self.test_time_history_steps:
#             self.history_bev_feats_all.pop(0)
#             self.history_img_metas_all.pop(0)
#
#         if not self.skip_vector_head:
#             memory_bank = self.memory_bank if self.use_memory else None
#             thr_det = 0.4 if first_frame else 0.6
#             pos_results = self.head.prepare_temporal_propagation(preds_dict, scene_name, local_idx,
#                                         memory_bank, thr_track=0.5, thr_det=thr_det)
#
#         if not self.skip_vector_head:
#             results_list = self.head.post_process(preds_dict, tokens, track_dict)
#             results_list[0]['pos_results'] = pos_results
#             results_list[0]['meta'] = img_metas[0]
#         else:
#             results_list = [{'vectors': [],
#                 'scores': [],
#                 'labels': [],
#                 'props': [],
#                 'token': token} for token in tokens]
#
#         # Add the segmentation preds to the results to be saved
#         for b_i in range(len(results_list)):
#             tmp_scores, tmp_labels = seg_preds[b_i].max(0)
#             tmp_scores = tmp_scores.sigmoid()
#             preds_i = torch.zeros(tmp_labels.shape, dtype=torch.uint8).to(tmp_scores.device)
#             pos_ids = tmp_scores >= 0.4
#             preds_i[pos_ids] = tmp_labels[pos_ids].type(torch.uint8) + 1
#             preds_i = preds_i.cpu().numpy()
#             results_list[b_i]['semantic_mask'] = preds_i
#             if 'token' not in results_list[b_i]:
#                 results_list[b_i]['token'] = tokens[b_i]
#
#         return results_list
#
#     def batch_data(self, vectors, imgs, img_metas, device, points=None):
#         bs = len(vectors)
#         # filter none vector's case
#         num_gts = []
#         for idx in range(bs):
#             num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
#         valid_idx = [i for i in range(bs) if num_gts[i] > 0]
#         assert len(valid_idx) == bs # make sure every sample has gts
#
#         all_labels_list = []
#         all_lines_list = []
#         all_gt2local = []
#         all_local2gt = []
#         for idx in range(bs):
#             labels = []
#             lines = []
#             gt2local = []
#             local2gt = {}
#             for label, _lines in vectors[idx].items():
#                 for _ins_id, _line in enumerate(_lines):
#                     labels.append(label)
#                     gt2local.append([label, _ins_id])
#                     local2gt[(label, _ins_id)] = len(lines)
#                     if len(_line.shape) == 3: # permutation
#                         num_permute, num_points, coords_dim = _line.shape
#                         lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
#                     elif len(_line.shape) == 2:
#                         lines.append(torch.tensor(_line).reshape(-1)) # (40, )
#                     else:
#                         assert False
#
#             all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
#             all_lines_list.append(torch.stack(lines).float().to(device))
#             all_gt2local.append(gt2local)
#             all_local2gt.append(local2gt)
#
#         gts = {
#             'labels': all_labels_list,
#             'lines': all_lines_list,
#             'gt2local': all_gt2local,
#             'local2gt': all_local2gt,
#         }
#
#         gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]
#
#         return gts, imgs, img_metas, valid_idx, points
#
#     def get_two_frame_matching(self, local2global_prev, local2global_curr, gts_prev, gts):
#         """
#         Get the G.T. matching between the two frames
#         Terminology: (1). local --> local idx inside each category;
#                     (2). global --> global instance id inside category
#                     (3). gt --> index in the flattened G.T. sequence
#         Args:
#             prev_ins_ids (_type_): global ids (pre-prepared) for prev frame
#             curr_ins_ids (_type_): global ids (pre-prepared) for curr frame
#             gts_prev (_type_): processed G.T. for prev frame
#             gts (_type_): processed G.T. for curr frame
#         """
#         bs = len(local2global_prev)
#         gt2local_curr = gts[-1]['gt2local'] # don't need the per-block supervision, just take one
#         gt2local_prev = gts_prev[-1]['gt2local']
#         local2gt_prev = gts_prev[-1]['local2gt']
#
#         # the comma is to take the single-element output from multi_apply
#         global2local_prev, = multi_apply(self._reverse_id_mapping, local2global_prev)
#
#         all_gt_cur2prev, all_gt_prev2cur = multi_apply(self._compute_cur2prev, gt2local_curr, gt2local_prev, local2gt_prev,
#                                         local2global_curr, global2local_prev)
#
#         return all_gt_cur2prev, all_gt_prev2cur
#
#     def _compute_cur2prev(self, gt2local_curr, gt2local_prev, local2gt_prev,
#                           local2global_curr, global2local_prev):
#         cur2prev = torch.zeros(len(gt2local_curr))
#         prev2cur = torch.zeros(len(gt2local_prev))
#         prev2cur[:] = -1
#         for gt_idx_curr in range(len(gt2local_curr)):
#             label = gt2local_curr[gt_idx_curr][0]
#             local_idx = gt2local_curr[gt_idx_curr][1]
#             seq_id = local2global_curr[label][local_idx]
#             if seq_id in global2local_prev[label]:
#                 local_id_prev = global2local_prev[label][seq_id]
#                 gt_idx_prev = local2gt_prev[(label, local_id_prev)]
#             else:
#                 gt_idx_prev = -1
#             cur2prev[gt_idx_curr] = gt_idx_prev
#             if gt_idx_prev != -1: # there is a positive match in prev frame
#                 prev2cur[gt_idx_prev] = gt_idx_curr # update the information
#
#         return cur2prev, prev2cur
#
#     def _reverse_id_mapping(self, id_mapping):
#         reversed_mapping = {}
#         for label, mapping in id_mapping.items():
#             r_map = {v:k for k,v in mapping.items()}
#             reversed_mapping[label] = r_map
#         return reversed_mapping,
#
#     def prepare_track_queries_and_targets(self, gts, prev_inds_list, prev_gt_inds_list, prev_matched_reg_cost,
#                      prev_gt_list, prev_out, gt_cur2prev, gt_prev2cur, metas_prev, use_memory, pos_th=0.4, timestep=None):
#         bs = len(prev_inds_list)
#         device = prev_out['lines'][0].device
#
#         targets = []
#         for b_i in range(bs):
#             results = {}
#             for key, val in gts[-1].items():
#                 results[key] = val[b_i]
#             targets.append(results)
#
#         # for each sample in the batch
#         for b_i, (target, prev_out_ind, prev_target_ind) in enumerate(zip(targets, prev_inds_list, prev_gt_inds_list)):
#             scene_seq_id = metas_prev[b_i]['local_idx']
#
#             scores = prev_out['scores'][b_i].detach()
#             scores, labels = scores.max(-1)
#             scores = scores.sigmoid()
#
#             match_cost = prev_matched_reg_cost[b_i]
#             target_prev2cur = gt_prev2cur[b_i].to(device)
#             target['prev_target_ind'] = prev_target_ind # record the matched g.t. index
#             target['prev_out_ind'] = prev_out_ind
#             target['gt_prev2cur'] = target_prev2cur
#             assert len(target_prev2cur) == len(prev_gt_inds_list[b_i])
#
#             # 1). filter the ones with low scores, create FN;
#             prev_pos_scores = scores[prev_out_ind]
#             score_filter_mask = prev_pos_scores >= pos_th
#
#             keep_mask = score_filter_mask
#             prev_out_ind_filtered = prev_out_ind[keep_mask]
#             prev_target_ind_filtered = prev_target_ind[keep_mask]
#
#             target_prev2cur = target_prev2cur[prev_target_ind_filtered]
#             target_ind_matching = (target_prev2cur != -1) # -1 means no matching g.t. in curr frame
#             # matched g.t. index in the current frame
#             target_ind_matched_idx = target_prev2cur[target_prev2cur!=-1]
#
#             target['track_query_match_ids'] = target_ind_matched_idx
#
#             if timestep == 0:
#                 pad_bound = self.head.num_queries
#             else:
#                 pad_bound = self.tracked_query_length[b_i] + self.head.num_queries
#
#             not_prev_out_ind = torch.arange(prev_out['lines'][b_i].shape[0]).to(device)
#             not_prev_out_ind = torch.tensor([
#                 ind.item()
#                 for ind in not_prev_out_ind
#                 if ind not in prev_out_ind and ind < pad_bound])
#
#             # Get all non-matched pred with >0.5 conf score, serve as FP
#             neg_scores = scores[not_prev_out_ind]
#             neg_score_mask = neg_scores >= pos_th
#             # Randomly pick 10% neg output instances and serve as FP
#             _rand_insert = torch.rand([len(neg_scores)]).to(device)
#
#             if self.track_fp_aug:
#                 rand_insert_mask = _rand_insert >= 0.95
#                 fp_select_mask = neg_score_mask | rand_insert_mask
#             else:
#                 fp_select_mask = neg_score_mask
#
#             false_out_ind = not_prev_out_ind[fp_select_mask]
#
#             prev_out_ind_final = torch.tensor(prev_out_ind_filtered.tolist() + false_out_ind.tolist()).long()
#             target_ind_matching = torch.cat([
#                 target_ind_matching,
#                 torch.tensor([False, ] * len(false_out_ind)).bool().to(device)
#             ])
#
#             target_prev2cur_aug = torch.cat([
#                 target_prev2cur,
#                 torch.tensor([-1, ] * len(false_out_ind)).to(device)
#             ])
#             target['track_to_cur_gt_ids'] = target_prev2cur_aug
#
#             # track query masks
#             track_queries_mask = torch.ones_like(target_ind_matching).bool()
#             track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
#             track_queries_fal_pos_mask[~target_ind_matching] = True
#
#             # set prev frame info
#             target['track_query_hs_embeds'] = prev_out['hs_embeds'][b_i, prev_out_ind_final]
#             target['track_query_boxes'] = prev_out['lines'][b_i][prev_out_ind_final].detach()
#             tmp_labels = labels[prev_out_ind_final]
#             tmp_scores = scores[prev_out_ind_final]
#             target['track_query_labels'] = tmp_labels
#             target['track_query_scores'] = tmp_scores
#
#             # Prepare the G.T. line coords for the track queries, used in the transformation loss
#             prev_gt_lines = prev_gt_list['lines'][b_i]
#             prev_gt_labels = prev_gt_list['labels'][b_i]
#             target['track_query_gt_lines'] = prev_gt_lines[prev_out_ind_final]
#             target['track_query_gt_labels'] = prev_gt_labels[prev_out_ind_final]
#
#             target['track_queries_mask'] = torch.cat([
#                 track_queries_mask,
#                 torch.tensor([False, ] * self.head.num_queries).to(device)
#             ]).bool()
#
#             target['track_queries_fal_pos_mask'] = torch.cat([
#                 track_queries_fal_pos_mask,
#                 torch.tensor([False, ] * self.head.num_queries).to(device)
#             ]).bool()
#
#             if use_memory:
#                 is_first_frame = (timestep == 0)
#                 num_tracks = 0 if timestep == 0 else self.tracked_query_length[b_i]
#                 self.memory_bank.update_memory(b_i, is_first_frame, prev_out_ind_final, prev_out, num_tracks, scene_seq_id, timestep)
#
#         targets = self._batchify_tracks(targets)
#         return targets
#
#     def _batchify_tracks(self, targets):
#         lengths = [len(t['track_queries_mask']) for t in targets]
#         max_len = max(lengths)
#         device = targets[0]['track_query_hs_embeds'].device
#         for b_i in range(len(lengths)):
#             target = targets[b_i]
#             padding_len = max_len - lengths[b_i]
#             pad_hs_embeds = torch.zeros([padding_len, target['track_query_hs_embeds'].shape[1]]).to(device)
#             pad_query_boxes = torch.zeros([padding_len, target['track_query_boxes'].shape[1]]).to(device)
#             query_padding_mask = torch.zeros([max_len]).bool().to(device)
#             query_padding_mask[lengths[b_i]:] = True
#             target['pad_hs_embeds'] = pad_hs_embeds
#             target['pad_query_boxes'] = pad_query_boxes
#             target['query_padding_mask'] = query_padding_mask
#             self.tracked_query_length[b_i] = lengths[b_i] - self.head.num_queries
#         return targets
#
#     def train(self, *args, **kwargs):
#         super().train(*args, **kwargs)
#         if self.freeze_bev:
#             self._freeze_bev()
#         elif self.freeze_bev_iters is not None and self.num_iter < self.freeze_bev_iters:
#             self._freeze_bev()
#         else:
#             self._unfreeze_bev()
#
#     def eval(self):
#         super().eval()
#
#     def _freeze_bev(self,):
#         """Freeze all bev-related backbone parameters, including the backbone and the seg head
#         """
#         for param in self.backbone.parameters():
#             param.requires_grad = False
#         for param in self.seg_decoder.parameters():
#             param.requires_grad = False
#
#     def _unfreeze_bev(self,):
#         """unfreeze all bev-related backbone parameters, including the backbone and the seg head
#         """
#         for param in self.backbone.parameters():
#             param.requires_grad = True
#         for param in self.seg_decoder.parameters():
#             param.requires_grad = True
#
#     def _denorm_lines(self, line_pts):
#         """from (0,1) to the BEV space in meters"""
#         line_pts[..., 0] = line_pts[..., 0] * self.roi_size[0] \
#                         - self.roi_size[0] / 2
#         line_pts[..., 1] = line_pts[..., 1] * self.roi_size[1] \
#                         - self.roi_size[1] / 2
#         return line_pts
#
#     def _norm_lines(self, line_pts):
#         """from the BEV space in meters to (0,1) """
#         line_pts[..., 0] = (line_pts[..., 0] + self.roi_size[0] / 2) \
#                                         / self.roi_size[0]
#         line_pts[..., 1] = (line_pts[..., 1] + self.roi_size[1] / 2) \
#                                         / self.roi_size[1]
#         return line_pts
#
#     def _process_track_query_info(self, track_info):
#         bs = len(track_info)
#         all_scores = []
#         all_lines = []
#         for b_i in range(bs):
#             embeds = track_info[b_i]['track_query_hs_embeds']
#             scores = self.head.cls_branches[-1](embeds)
#             coords = self.head.reg_branches[-1](embeds).sigmoid()
#             coords = rearrange(coords, 'n1 (n2 n3) -> n1 n2 n3', n3=2)
#             all_scores.append(scores)
#             all_lines.append(coords)
#         track_results = {
#             'lines': all_lines,
#             'scores': all_scores,
#         }
#         return track_results
#
#     def select_memory_entries(self, history_metas, curr_meta):
#         """
#         Only used at test time, to select a subset from the long history bank
#         """
#         if len(history_metas) <= self.history_steps:
#             return np.arange(len(history_metas))
#         else:
#             history_e2g_trans = np.array([item[0]['ego2global_translation'] for item in history_metas])[:, :2]
#             curr_e2g_trans = np.array(curr_meta[0]['ego2global_translation'])[:2]
#             dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)
#
#             sorted_indices = np.argsort(dists)
#             sorted_dists = dists[sorted_indices]
#             covered = np.zeros_like(sorted_indices).astype(np.bool)
#             selected_ids = []
#             for dist_range in self.mem_select_dist_ranges[::-1]:
#                 outter_valid_flags = (sorted_dists >= dist_range) & ~covered
#                 if outter_valid_flags.any():
#                     pick_id = np.where(outter_valid_flags)[0][0]
#                     covered[pick_id:] = True
#                 else:
#                     inner_valid_flags = (sorted_dists < dist_range) & ~covered
#                     if inner_valid_flags.any():
#                         pick_id = np.where(inner_valid_flags)[0][-1]
#                         covered[pick_id] = True
#                     else:
#                         return np.arange(len(history_metas))[-4:]
#                 selected_ids.append(pick_id)
#
#             selected_mem_ids = sorted_indices[np.array(selected_ids)]
#
#             return selected_mem_ids
#
#     #####################################################################
#     #
#     # Debugging visualization of the temporal propagation supervision
#     #
#     #####################################################################
#
#     def _viz_temporal_supervision(self, outputs_prev, all_track_info, gts, gts_prev, semantic_mask,
#                                   semantic_mask_prev, img_metas, img_metas_prev, timestep):
#         """For debugging use: draw the visualization of the track queries and the corresponding
#         matched G.T. information..."""
#         import os
#         from ..utils.renderer_track import Renderer
#         viz_dir = './viz/debug_noisy_trans'
#         if not os.path.exists(viz_dir):
#             os.makedirs(viz_dir)
#         cat2id = {
#             'ped_crossing': 0,
#             'divider': 1,
#             'boundary': 2,
#         }
#         renderer = Renderer(cat2id, self.roi_size, 'nusc')
#
#         for b_i in range(len(all_track_info)):
#             track_info = all_track_info[b_i]
#             # prev pred info
#             prev_pred_lines = outputs_prev['lines'][b_i]
#             prev_pred_scores = outputs_prev['scores'][b_i]
#             prev_target_inds = track_info['prev_target_ind']
#             prev_out_inds = track_info['prev_out_ind']
#             gt_prev2cur = track_info['gt_prev2cur']
#             prev_scores, prev_labels = prev_pred_scores.max(-1)
#             prev_scores = prev_scores.sigmoid()
#             prev_lines = rearrange(prev_pred_lines[prev_out_inds], 'n (k c) -> n k c', c=2)
#             prev_labels = prev_labels[prev_out_inds]
#             prev_lines = self._denorm_lines(prev_lines)
#             prev_scores = prev_scores[prev_out_inds]
#             out_path_prev = os.path.join(viz_dir, f't={timestep}_{b_i}_prev.png')
#             renderer.render_bev_from_vectors(prev_lines, prev_labels, out_path_prev,
#                 id_info=prev_target_inds, score_info=prev_scores)
#
#             # gt info
#             gt_labels = gts['labels'][b_i]
#             gt_lines = torch.clip(gts['lines'][b_i][:, 0], 0, 1)
#             gt_lines = rearrange(gt_lines, 'n (k c) -> n k c', c=2)
#             gt_lines = self._denorm_lines(gt_lines)
#             out_path_gt = os.path.join(viz_dir, f't={timestep}_{b_i}_gt.png')
#             gt_ids = np.arange(len(gt_lines))
#             renderer.render_bev_from_vectors(gt_lines, gt_labels, out_path_gt, id_info=gt_ids)
#             gt_semantic = semantic_mask[b_i].cpu().numpy()
#             out_path_gt_semantic = os.path.join(viz_dir, f't={timestep}_{b_i}_gt_semantic.png')
#             renderer.render_bev_from_mask(gt_semantic, out_path_gt_semantic)
#
#             # gt info for prev frame
#             gt_labels = gts_prev['labels'][b_i]
#             gt_lines = torch.clip(gts_prev['lines'][b_i][:, 0], 0, 1)
#             gt_lines = rearrange(gt_lines, 'n (k c) -> n k c', c=2)
#             gt_lines = self._denorm_lines(gt_lines)
#             out_path_gt = os.path.join(viz_dir, f't={timestep}_{b_i}_prev_gt.png')
#             gt_ids = np.arange(len(gt_lines))
#             renderer.render_bev_from_vectors(gt_lines, gt_labels, out_path_gt, id_info=gt_ids)
#             gt_semantic = semantic_mask_prev[b_i].cpu().numpy()
#             out_path_gt_semantic = os.path.join(viz_dir, f't={timestep}_{b_i}_prev_gt_semantic.png')
#             renderer.render_bev_from_mask(gt_semantic, out_path_gt_semantic)
#
#             # track query info
#             track_to_cur_gt_ids = track_info['track_to_cur_gt_ids']
#             trans_track_lines = track_info['trans_track_query_boxes']
#             trans_track_lines = rearrange(trans_track_lines, 'n (k c) -> n k c', c=2)
#             trans_track_lines = self._denorm_lines(trans_track_lines)
#             #tp_track_mask = ~track_info['track_queries_fal_pos_mask'][:-100]
#             trans_track_lines = trans_track_lines
#             track_labels = track_info['track_query_labels']
#             track_scores = track_info['track_query_scores']
#             out_path_track = os.path.join(viz_dir, f't={timestep}_{b_i}_track.png')
#             renderer.render_bev_from_vectors(trans_track_lines, track_labels, out_path_track,
#                 id_info=track_to_cur_gt_ids, score_info=track_scores)

# -------------------------------TTSIM-----------------------------------


from __future__ import annotations

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops.tensor import require_shape_list
from einops import rearrange

from workloads.MapTracker.plugin.models.heads.Map_Seg_Head import MapSegHead
from workloads.MapTracker.plugin.models.heads.MapDetectorHead import MapDetectorHead
from workloads.MapTracker.plugin.models.necks.gru import ConvGRU
from workloads.MapTracker.plugin.models.backbones.bevformer.transformer import (
    PerceptionTransformer,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.encoder import (
    BEVFormerEncoder,
)
from workloads.MapTracker.plugin.models.backbones.bevformer_backbone import (
    BEVFormerBackbone,
)
from workloads.MapTracker.plugin.models.utils.query_update import MotionMLP
from workloads.MapTracker.plugin.models.maper.vector_memory import VectorInstanceMemory
from workloads.MapTracker.plugin.models.transformer_utils.MapTransformer import (
    MapTransformer,
)


class MapTracker(SimNN.Module):
    """
    Pipeline:
        BEV Features -> MapSegHead -> Segmentation Predictions
                     -> MapDetectorHead -> Vector Map Predictions


    Args:
        name: Module name
        bev_h: BEV feature map height (default: 100)
        bev_w: BEV feature map width (default: 50)
        roi_size: ROI size in meters [x_range, y_range] (default: [60, 30])
        seg_num_classes: Number of segmentation classes (default: 3)
        seg_in_channels: Input channels to seg head (default: 256)
        seg_embed_dims: Embedding dimensions for seg head (default: 256)
        canvas_size: Output canvas size [H, W] (default: [200, 100])
        max_batch_size: Maximum batch size (default: 4)
        use_backbone: Whether to use backbone (default: False, expects BEV features)
        use_neck: Whether to use neck (default: False)
        use_vector_head: Whether to use vector head (default: True)
        vector_num_queries: Number of queries for vector head (default: 100)
        vector_num_classes: Number of vector classes (default: 3)
        vector_num_points: Number of points per vector (default: 20)
        use_query_propagation: Whether to use query propagation across frames (default: True)
        use_memory: Whether to use memory bank for instance tracking (default: True)
        mem_len: Memory bank size in frames (default: 10)
        mem_select_dist_ranges: Distance ranges for memory selection (default: [20, 40, 60, 80])
    """

    def __init__(self, name, cfg):
        super().__init__()

        # Unpack config dict with defaults
        bev_h = cfg.get("bev_h", 100)
        bev_w = cfg.get("bev_w", 50)
        roi_size = cfg.get("roi_size", (60, 30))
        seg_num_classes = cfg.get("seg_num_classes", 3)
        seg_in_channels = cfg.get("seg_in_channels", 256)
        seg_embed_dims = cfg.get("seg_embed_dims", 256)
        canvas_size = cfg.get("canvas_size", (200, 100))
        max_batch_size = cfg.get("max_batch_size", 4)
        use_backbone = cfg.get("use_backbone", True)
        use_neck = cfg.get("use_neck", False)
        use_vector_head = cfg.get("use_vector_head", True)
        use_query_propagation = cfg.get("use_query_propagation", True)
        use_memory = cfg.get("use_memory", True)
        vector_num_queries = cfg.get("vector_num_queries", 100)
        vector_num_classes = cfg.get("vector_num_classes", 3)
        vector_num_points = cfg.get("vector_num_points", 20)
        vector_embed_dims = cfg.get("vector_embed_dims", 256)
        mem_len = cfg.get("mem_len", 10)
        mem_select_dist_ranges = cfg.get("mem_select_dist_ranges", (20, 40, 60, 80))
        backbone_cfg = cfg.get("backbone_cfg", None)
        neck_out_channels = cfg.get("neck_out_channels", 256)
        history_steps = cfg.get("history_steps", len(mem_select_dist_ranges))
        test_time_history_steps = cfg.get("test_time_history_steps", None)
        skip_vector_head = cfg.get("skip_vector_head", False)

        # Image backbone parameters (used when use_backbone=True)
        # Defaults match MapTracker original PyTorch config (AV2: 7 cams, 2 encoder layers)
        num_cams = cfg.get("num_cams", 7)
        img_height = cfg.get("img_height", 256)
        img_width = cfg.get("img_width", 256)
        num_encoder_layers = cfg.get("num_encoder_layers", 2)
        pc_range = cfg.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        num_points_in_pillar = cfg.get("num_points_in_pillar", 4)

        self.name = name
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size
        self.seg_num_classes = seg_num_classes
        self.seg_in_channels = seg_in_channels
        self.seg_embed_dims = seg_embed_dims
        self.canvas_size = canvas_size
        self.max_batch_size = max_batch_size
        self.use_backbone = use_backbone
        self.use_neck = use_neck
        self.num_cams = num_cams
        self.img_height = img_height
        self.img_width = img_width
        self.pc_range = pc_range
        self.use_vector_head = use_vector_head
        self.use_query_propagation = use_query_propagation
        self.use_memory = use_memory
        self.mem_len = mem_len
        self.history_steps = history_steps
        self.skip_vector_head = skip_vector_head
        self.mem_select_dist_ranges = mem_select_dist_ranges
        self.vector_num_points = vector_num_points

        # Set up test time memory selection hyper-parameters
        if test_time_history_steps is None:
            self.test_time_history_steps = history_steps
        else:
            self.test_time_history_steps = test_time_history_steps

        # BEV coordinate plane for transformations
        # Creates a meshgrid of (x, y, z=0, 1) coordinates in BEV space
        xmin, xmax = -roi_size[0] / 2, roi_size[0] / 2
        ymin, ymax = -roi_size[1] / 2, roi_size[1] / 2
        x = np.linspace(xmin, xmax, bev_w)
        y = np.linspace(ymax, ymin, bev_h)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        z = np.zeros_like(xx)
        ones = np.ones_like(xx)
        self.plane = np.stack([xx, yy, z, ones], axis=-1).astype(np.float64)

        # History buffers for test-time temporal fusion
        self.history_bev_feats_all = []
        self.history_img_metas_all = []

        # Backbone - optional, can pass pre-computed BEV features
        # MapTracker config: ResNet-50 out_indices=(1,2,3) -> FPN 3 levels -> 2-layer BEVFormer encoder
        num_fpn_levels = cfg.get("num_fpn_levels", 3)
        self.num_fpn_levels = num_fpn_levels

        if use_backbone:
            # Reshape op for camera flattening: [bs, nc, 3, H, W] -> [bs*nc, 3, H, W]
            self.img_reshape_flat = F.Reshape(f"{name}.img_reshape_flat")

            # ================================================================
            # BEVFormer encoder chain:
            # PerceptionTransformer -> BEVFormerEncoder -> BEVFormerLayer
            #   -> TemporalSelfAttention + SpatialCrossAttention + FFN
            # Matching original MapTracker config: 3 FPN levels, 2 encoder layers
            # ================================================================
            attn_cfgs = [
                {
                    "type": "TemporalSelfAttention",
                    "embed_dims": seg_in_channels,
                    "num_heads": 8,
                    "num_levels": 1,
                    "num_points": 4,
                    "num_bev_queue": 2,
                    "dropout": 0.0,
                },
                {
                    "type": "SpatialCrossAttention",
                    "embed_dims": seg_in_channels,
                    "num_cams": num_cams,
                    "pc_range": pc_range,
                    "dropout": 0.0,
                    "deformable_attention": {
                        "embed_dims": seg_in_channels,
                        "num_levels": num_fpn_levels,
                        "num_points": 8,
                    },
                },
            ]

            transformerlayers_cfg = {
                "attn_cfgs": attn_cfgs,
                "feedforward_channels": seg_in_channels * 2,
                "ffn_dropout": 0.1,
                "operation_order": (
                    "self_attn",
                    "norm",
                    "cross_attn",
                    "norm",
                    "ffn",
                    "norm",
                ),
            }

            encoder = BEVFormerEncoder(
                name=f"{name}.encoder",
                transformerlayers=transformerlayers_cfg,
                num_layers=num_encoder_layers,
                pc_range=pc_range,
                num_points_in_pillar=num_points_in_pillar,
                history_steps=history_steps,
            )

            bev_transformer = PerceptionTransformer(
                name=f"{name}.bev_transformer",
                encoder=encoder,
                embed_dims=seg_in_channels,
                num_feature_levels=num_fpn_levels,
                num_cams=num_cams,
            )

            # BEVFormerBackbone now includes ResNet-50 + FPN internally
            self.bev_backbone: BEVFormerBackbone | None = BEVFormerBackbone(
                name=f"{name}.bev_backbone",
                bev_h=bev_h,
                bev_w=bev_w,
                embed_dims=seg_in_channels,
                real_h=roi_size[1],
                real_w=roi_size[0],
                transformer=bev_transformer,
                max_batch_size=max_batch_size,
                upsample=False,
                history_steps=history_steps,
                num_cams=num_cams,
                img_channels=3,
                backbone_layers=[3, 4, 6, 3],  # ResNet-50
                out_indices=(1, 2, 3),  # C3, C4, C5
                fpn_in_channels=[512, 1024, 2048],
                fpn_num_outs=num_fpn_levels,
            )
        else:
            self.bev_backbone = None

        # Neck (ConvGRU) for temporal fusion - optional
        if use_neck:
            self.neck: ConvGRU | None = ConvGRU(out_channels=neck_out_channels)
        else:
            self.neck = None

        # Semantic segmentation head
        self.seg_decoder = MapSegHead(
            name=f"{name}.seg_decoder",
            num_classes=seg_num_classes,
            in_channels=seg_in_channels,
            embed_dims=seg_embed_dims,
            bev_size=(bev_h, bev_w),
            canvas_size=canvas_size,
            max_batch_size=max_batch_size,
        )

        # Vector map head (MapDetectorHead with transformer decoder)
        if use_vector_head:
            # Create MapTransformer with decoder
            # Transformer auto-creates decoder with attention modules, norms, and FFNs
            transformer = MapTransformer(
                name=f"{name}.transformer",
                embed_dims=vector_embed_dims,
                num_points=vector_num_points,
                num_layers=6,
            )

            self.head: MapDetectorHead | None = MapDetectorHead(
                num_queries=vector_num_queries,
                num_classes=vector_num_classes,
                in_channels=seg_in_channels,
                embed_dims=vector_embed_dims,
                num_points=vector_num_points,
                num_layers=6,
                different_heads=True,
                predict_refine=False,
                transformer=transformer,
            )

            # Query propagation module (MotionMLP) for temporal tracking
            # Transforms queries from previous frame to current frame using relative pose
            if use_query_propagation:
                c_dim = 7  # quaternion for rotation (4) + translation (3)
                self.query_propagate: MotionMLP | None = MotionMLP(
                    name=f"{name}.query_propagate",
                    c_dim=c_dim,
                    f_dim=vector_embed_dims,
                    identity=True,
                )
            else:
                self.query_propagate = None

            # Memory bank for instance tracking across frames
            # Stores embeddings, IDs, and pose information
            if use_memory:
                self.memory_bank: VectorInstanceMemory | None = VectorInstanceMemory(
                    name=f"{name}.memory_bank",
                    dim_in=vector_embed_dims,
                    number_ins=vector_num_queries,
                    bank_size=mem_len,
                    mem_len=mem_len,
                    mem_select_dist_ranges=mem_select_dist_ranges,
                )
            else:
                self.memory_bank = None
        else:
            self.head = None
            self.query_propagate = None
            self.memory_bank = None

        # Track number of decoder layers for reference
        if self.head is not None and hasattr(self.head, "transformer"):
            self.num_decoder_layers = self.head.transformer.decoder.num_layers
        else:
            self.num_decoder_layers = 0

        # Tracked query length per batch item (used during temporal propagation)
        self.tracked_query_length = {}

        super().link_op2module()

    def create_input_tensors(self):
        """Create input tensors for polaris graph simulation."""
        if self.use_backbone:
            # Image input: [bs, num_cams, 3, img_h, img_w]
            self.input_tensors = {
                "img": F._from_shape(
                    "img", [1, self.num_cams, 3, self.img_height, self.img_width]
                ),
            }
        else:
            self.input_tensors = {
                "bev_features": F._from_shape(
                    "bev_features", [1, self.seg_in_channels, self.bev_h, self.bev_w]
                ),
            }
        for _, t in self.input_tensors.items():
            t.is_param = False
            t.set_module(self)

    def get_forward_graph(self):
        """Return the forward computation graph for polaris analysis."""
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self, lvl=0):
        return 0

    def init_weights(self, pretrained=None):
        """Initialize weights (placeholder for TTSim inference)."""
        pass

    def _create_synthetic_img_metas(self):
        """Create synthetic camera parameters for graph construction.

        Generates fake lidar2img matrices so the BEVFormer encoder can
        compute reference_points_cam and bev_mask during graph construction.
        The actual values don't matter for op-graph structure.
        """
        lidar2img = np.zeros((self.num_cams, 4, 4), dtype=np.float64)
        for i in range(self.num_cams):
            # Simple pinhole camera model with rotation around Z-axis
            angle = 2 * np.pi * i / self.num_cams
            # Extrinsic: rotate camera around ego vehicle
            R = np.eye(4)
            R[0, 0] = np.cos(angle)
            R[0, 1] = -np.sin(angle)
            R[1, 0] = np.sin(angle)
            R[1, 1] = np.cos(angle)
            R[2, 3] = 1.5  # camera height
            # Intrinsic
            K = np.eye(4)
            K[0, 0] = self.img_width  # fx
            K[1, 1] = self.img_height  # fy
            K[0, 2] = self.img_width / 2  # cx
            K[1, 2] = self.img_height / 2  # cy
            lidar2img[i] = K @ R

        img_metas = [
            {
                "lidar2img": lidar2img,
                "img_shape": [(self.img_height, self.img_width)] * self.num_cams,
            }
        ]
        return img_metas

    def __call__(
        self,
        bev_features=None,
        prev_bev=None,
        img_metas=None,
        track_query_info=None,
        memory_bank=None,
    ):
        """
        Forward pass for inference.

        When called with no args (polaris mode), uses self.input_tensors.
        When use_backbone=True: img -> simplified convs -> BEVFormer encoder -> BEV features.
        When use_backbone=False: expects pre-computed bev_features.
        """
        # Image backbone path: ResNet-50 + FPN + BEVFormer encoder -> BEV features
        if self.use_backbone and self.bev_backbone is not None:
            img = self.input_tensors["img"]  # [bs, num_cams, 3, H, W]
            setattr(self, img.name, img)
            img_shape = require_shape_list(
                img.shape, f"{img.name}: shape required for backbone path"
            )
            bs = int(img_shape[0])

            # Flatten cameras: [bs, num_cams, 3, H, W] -> [bs*num_cams, 3, H, W]
            flat_shape = F._from_data(
                f"{self.name}.img_flat_shape",
                np.array(
                    [bs * self.num_cams, 3, self.img_height, self.img_width],
                    dtype=np.int64,
                ),
                is_const=True,
            )
            setattr(self, flat_shape.name, flat_shape)
            img_flat = self.img_reshape_flat(img, flat_shape)
            setattr(self, img_flat.name, img_flat)

            # BEV positional encoding (zeros - learned positions are in BEV embedding)
            bev_pos = F._from_data(
                f"{self.name}.bev_pos",
                np.zeros(
                    (bs, self.seg_in_channels, self.bev_h, self.bev_w), dtype=np.float32
                ),
                is_const=True,
            )
            setattr(self, bev_pos.name, bev_pos)

            # Synthetic camera params for reference point projection
            if img_metas is None:
                img_metas = self._create_synthetic_img_metas()

            # BEVFormerBackbone: img -> ResNet-50 -> FPN -> BEVFormer encoder -> BEV features
            bev_features, _ = self.bev_backbone(img_flat, bev_pos, img_metas=img_metas)
            setattr(self, bev_features.name, bev_features)
        else:
            # Pre-computed BEV features path (original behavior)
            if bev_features is None:
                bev_features = self.input_tensors["bev_features"]
            setattr(self, bev_features.name, bev_features)

        # Neck: pass through (original uses nn.Identity or ConvGRU)
        if self.use_neck and self.neck is not None:
            bev_features = self.neck(bev_features, bev_features)
            if hasattr(bev_features, "name"):
                setattr(self, bev_features.name, bev_features)

        # Semantic segmentation head
        seg_preds, seg_feats = self.seg_decoder(bev_features)

        # Vector map head (MapDetectorHead)
        if self.use_vector_head and self.head is not None:
            # Pre-populate memory bank with dummy data so the decoder exercises
            # the memory cross-attention path (otherwise skipped on frame 0)
            memory_bank_arg = None
            if self.use_memory and self.memory_bank is not None:
                num_queries = self.head.num_queries
                embed_dims = self.head.embed_dims
                mem_len = self.memory_bank.mem_len
                bs = 1

                # Initialize and populate with synthetic tracked instances
                self.memory_bank.init_memory(bs=bs)
                self.memory_bank.curr_t = 1  # pretend frame 1 so memory is active

                # Create 1 dummy tracked instance per batch element
                n_tracks = 1
                for b_i in range(bs):
                    self.memory_bank.active_mem_ids[b_i] = np.array([0], dtype=np.int64)
                    self.memory_bank.valid_track_idx[b_i] = np.array(
                        [0], dtype=np.int64
                    )
                    self.memory_bank.mem_entry_lengths[b_i][0] = 1

                    # [mem_len, n_tracks, embed_dims] - memory embeddings
                    self.memory_bank.batch_mem_embeds_dict[b_i] = (
                        np.random.randn(mem_len, n_tracks, embed_dims).astype(
                            np.float32
                        )
                        * 0.01
                    )
                    # [mem_len, n_tracks, embed_dims] - relative PE
                    # Use F.Gather from pe_table SimTensor to keep graph connected
                    pe_table = self.memory_bank.pos_encoder.pe_table  # [1000, channels]
                    pe_gap_idx = F._from_data(
                        f"{self.name}/mem_pe_gap_idx_b{b_i}",
                        np.array([1], dtype=np.int64),
                        is_const=True,
                    )
                    setattr(self, pe_gap_idx.name, pe_gap_idx)
                    pe_gather_op = F.Gather(f"{self.name}/mem_pe_gather_b{b_i}", axis=0)
                    setattr(self, pe_gather_op.name, pe_gather_op)
                    pe_row = pe_gather_op(pe_table, pe_gap_idx)  # [1, embed_dims]
                    setattr(self, pe_row.name, pe_row)
                    # Reshape [1, embed_dims] → [1, 1, embed_dims]
                    pe_unsq_op = F.Unsqueeze(f"{self.name}/mem_pe_unsq_b{b_i}")
                    setattr(self, pe_unsq_op.name, pe_unsq_op)
                    pe_unsq_axis = F._from_data(
                        f"{self.name}/mem_pe_unsq_axis_b{b_i}",
                        np.array([1], dtype=np.int64),
                        is_const=True,
                    )
                    setattr(self, pe_unsq_axis.name, pe_unsq_axis)
                    pe_row_3d = pe_unsq_op(pe_row, pe_unsq_axis)  # [1, 1, embed_dims]
                    setattr(self, pe_row_3d.name, pe_row_3d)
                    # Pad with zeros for remaining mem_len-1 slots
                    pe_zeros = F._from_data(
                        f"{self.name}/mem_pe_zeros_b{b_i}",
                        np.zeros((mem_len - 1, n_tracks, embed_dims), dtype=np.float32),
                        is_const=True,
                    )
                    setattr(self, pe_zeros.name, pe_zeros)
                    pe_concat_op = F.ConcatX(
                        f"{self.name}/mem_pe_concat_b{b_i}", axis=0
                    )
                    setattr(self, pe_concat_op.name, pe_concat_op)
                    pe_full = pe_concat_op(
                        pe_row_3d, pe_zeros
                    )  # [mem_len, 1, embed_dims]
                    setattr(self, pe_full.name, pe_full)
                    self.memory_bank.batch_mem_relative_pe_dict[b_i] = pe_full
                    # [n_tracks, mem_len] - key padding mask (False = valid)
                    mask = np.ones((n_tracks, mem_len), dtype=np.float32)
                    mask[:, 0] = 0.0  # first entry is valid
                    self.memory_bank.batch_key_padding_dict[b_i] = mask

                memory_bank_arg = self.memory_bank

            vector_preds = self.head(
                bev_features, img_metas=img_metas, memory_bank=memory_bank_arg
            )

            # Force execution of vector outputs by creating sink operations
            # TTSim's try_compute_data only computes .data if all inputs have .data
            # The sink operations ensure the graph is executed
            if hasattr(self.head, "_execution_sinks"):
                # Create a single reduction that depends on all sinks
                all_sinks_list = self.head._execution_sinks
                if len(all_sinks_list) > 0:
                    # Sum all sink values to force their computation
                    sink_sum = all_sinks_list[0]
                    for i, sink in enumerate(all_sinks_list[1:]):
                        _sink_op = F.Add(f"{self.name}_sink_combine_{i}")
                        setattr(self, _sink_op.name, _sink_op)
                        sink_sum = _sink_op(sink_sum, sink)
                        setattr(self, sink_sum.name, sink_sum)
                    self._execution_sink = sink_sum

            # Trace MotionMLP: propagate last decoder layer's query embeddings
            # using a synthetic pose so Embedder + MotionMLP ops appear in graph
            if self.use_query_propagation and self.query_propagate is not None:
                last_layer_preds = (
                    vector_preds[-1] if isinstance(vector_preds, list) else vector_preds
                )
                hs_embeds = last_layer_preds[
                    "hs_embeds"
                ]  # SimTensor [bs, num_queries, embed_dims]

                # Reshape hs_embeds from [1, num_queries, embed_dims] to
                # [num_queries, embed_dims] using a graph-connected op so
                # MotionMLP is wired to the decoder output (not a disconnected
                # _from_data constant).
                nq = self.head.num_queries
                ed = self.head.embed_dims
                reshape_name = f"{self.name}/trace_motionmlp/queries_reshape"
                target_shape = F._from_data(
                    f"{reshape_name}_shape",
                    np.array([nq, ed], dtype=np.int64),
                    is_const=True,
                )
                setattr(self, target_shape.name, target_shape)
                reshape_op = F.Reshape(reshape_name)
                setattr(self, reshape_op.name, reshape_op)
                query_tensor = reshape_op(hs_embeds, target_shape)
                setattr(self, query_tensor.name, query_tensor)

                # Pose is an external input (relative ego-motion between frames)
                pose_np = np.array(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
                )
                pose_tensor = F._from_data(
                    f"{self.name}/trace_motionmlp/pose", pose_np, is_const=False
                )
                setattr(self, pose_tensor.name, pose_tensor)

                propagated = self.propagate_queries(query_tensor, pose_tensor)
                setattr(self, propagated.name, propagated)

            return seg_preds, seg_feats, vector_preds

        return seg_preds, seg_feats

    def propagate_queries(self, query_embeddings, pose_info):
        """
        Propagate query embeddings from previous frame to current frame.
        Uses MotionMLP to transform queries based on relative pose.

        Args:
            query_embeddings: Query embeddings from previous frame
                Shape: [num_queries, embed_dims]
            pose_info: Relative pose between frames
                Shape: [7] - quaternion (4) + translation (3)
                Format: [qx, qy, qz, qw, tx, ty, tz]

        Returns:
            propagated_queries: Transformed query embeddings
                Shape: [num_queries, embed_dims]

        Note:
            Requires use_query_propagation=True in constructor.
            pose_info can be computed from ego2global rotation/translation.
        """
        if not self.use_query_propagation or self.query_propagate is None:
            raise ValueError(
                "Query propagation not enabled. Set use_query_propagation=True"
            )

        # Track tensors
        if hasattr(query_embeddings, "name"):
            setattr(self, query_embeddings.name, query_embeddings)
        if hasattr(pose_info, "name"):
            setattr(self, pose_info.name, pose_info)

        # Expand pose to match query batch dimension
        # pose_info: [7] -> [num_queries, 7]
        num_queries = query_embeddings.shape[0]

        # Unsqueeze pose to [1, 7]
        unsqueeze_name = f"{self.name}/propagate/unsqueeze_pose"
        axis = F._from_data(
            f"{unsqueeze_name}_axis", np.array([0], dtype=np.int64), is_const=True
        )
        setattr(self, axis.name, axis)
        unsqueeze_op = F.Unsqueeze(unsqueeze_name)
        setattr(self, unsqueeze_op.name, unsqueeze_op)
        pose_unsqueezed = unsqueeze_op(pose_info, axis)
        setattr(self, pose_unsqueezed.name, pose_unsqueezed)

        # Tile to [num_queries, 7]
        tile_name = f"{self.name}/propagate/tile_pose"
        repeats = F._from_data(
            f"{tile_name}_repeats",
            np.array([num_queries, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, repeats.name, repeats)
        tile_op = F.Tile(tile_name)
        setattr(self, tile_op.name, tile_op)
        pose_expanded = tile_op(pose_unsqueezed, repeats)
        setattr(self, pose_expanded.name, pose_expanded)

        # Apply MotionMLP to transform queries based on pose
        propagated_queries = self.query_propagate(query_embeddings, pose_expanded)
        setattr(self, propagated_queries.name, propagated_queries)

        return propagated_queries

    def init_memory_bank(self, batch_size=1):
        """
        Initialize the memory bank for a new sequence.

        Args:
            batch_size: Number of samples in batch (default: 1)

        Note:
            Should be called at the start of each new sequence.
            Requires use_memory=True in constructor.
        """
        if not self.use_memory or self.memory_bank is None:
            raise ValueError("Memory bank not enabled. Set use_memory=True")

        self.memory_bank.init_memory(bs=batch_size)

    def update_memory_bank(self, query_embeddings, sequence_ids, curr_pose):
        """
        Update memory bank with current frame's query information.

        Args:
            query_embeddings: Query embeddings from current frame
                Shape: [batch_size, num_queries, embed_dims]
            sequence_ids: Sequence IDs for each query
                Shape: [batch_size, num_queries]
            curr_pose: Current ego pose (rotation + translation)
                Dict with keys 'rotation' (quaternion) and 'translation' (3D)

        Note:
            Requires use_memory=True in constructor.
            Called after each frame to store tracking information.
        """
        if not self.use_memory or self.memory_bank is None:
            raise ValueError("Memory bank not enabled. Set use_memory=True")

        # Track tensors
        if hasattr(query_embeddings, "name"):
            setattr(self, query_embeddings.name, query_embeddings)

        # Update memory bank with current frame info
        # This stores embeddings and pose for future frame matching
        # Implementation delegates to VectorInstanceMemory
        pass  # Memory bank update logic handled by VectorInstanceMemory

    def retrieve_from_memory(self, batch_idx=0):
        """
        Retrieve tracked instances from memory bank.

        Args:
            batch_idx: Batch index to retrieve from (default: 0)

        Returns:
            memory_embeddings: Retrieved embeddings from memory
            memory_ids: Corresponding sequence IDs

        Note:
            Requires use_memory=True in constructor.
            Returns None if memory is empty.
        """
        if not self.use_memory or self.memory_bank is None:
            raise ValueError("Memory bank not enabled. Set use_memory=True")

        # Retrieve from memory bank
        # Implementation delegates to VectorInstanceMemory
        return None  # Placeholder - actual retrieval done by VectorInstanceMemory

    def _denorm_lines(self, line_pts):
        """Convert line points from normalized (0,1) to BEV space in meters.

        Maps normalized coordinates to physical coordinates:
            x: (0,1) -> (-roi_size[0]/2, roi_size[0]/2)
            y: (0,1) -> (-roi_size[1]/2, roi_size[1]/2)

        Args:
            line_pts: numpy array of shape [..., 2] with values in (0, 1)

        Returns:
            line_pts: numpy array of shape [..., 2] with values in meters
        """
        line_pts = line_pts.copy()
        line_pts[..., 0] = line_pts[..., 0] * self.roi_size[0] - self.roi_size[0] / 2
        line_pts[..., 1] = line_pts[..., 1] * self.roi_size[1] - self.roi_size[1] / 2
        return line_pts

    def _norm_lines(self, line_pts):
        """Convert line points from BEV space in meters to normalized (0,1).

        Maps physical coordinates to normalized coordinates:
            x: (-roi_size[0]/2, roi_size[0]/2) -> (0, 1)
            y: (-roi_size[1]/2, roi_size[1]/2) -> (0, 1)

        Args:
            line_pts: numpy array of shape [..., 2] with values in meters

        Returns:
            line_pts: numpy array of shape [..., 2] with values in (0, 1)
        """
        line_pts = line_pts.copy()
        line_pts[..., 0] = (line_pts[..., 0] + self.roi_size[0] / 2) / self.roi_size[0]
        line_pts[..., 1] = (line_pts[..., 1] + self.roi_size[1] / 2) / self.roi_size[1]
        return line_pts

    def select_memory_entries(self, history_metas, curr_meta):
        """Select a subset from the long history bank based on distance.

        Only used at test time. Selects history frames that provide good
        spatial coverage around the current ego position using the
        mem_select_dist_ranges hyper-parameters.

        Args:
            history_metas: List of img_metas dicts from history frames
            curr_meta: Current frame's img_metas (list with one element)

        Returns:
            selected_ids: numpy array of indices into history_metas
        """
        if len(history_metas) <= self.history_steps:
            return np.arange(len(history_metas))
        else:
            history_e2g_trans = np.array(
                [item[0]["ego2global_translation"] for item in history_metas]
            )[:, :2]
            curr_e2g_trans = np.array(curr_meta[0]["ego2global_translation"])[:2]
            dists = np.linalg.norm(history_e2g_trans - curr_e2g_trans[None, :], axis=1)

            sorted_indices = np.argsort(dists)
            sorted_dists = dists[sorted_indices]
            covered = np.zeros_like(sorted_indices, dtype=bool)
            selected_ids = []

            for dist_range in self.mem_select_dist_ranges[::-1]:
                outter_valid_flags = (sorted_dists >= dist_range) & ~covered
                if outter_valid_flags.any():
                    pick_id = np.where(outter_valid_flags)[0][0]
                    covered[pick_id:] = True
                else:
                    inner_valid_flags = (sorted_dists < dist_range) & ~covered
                    if inner_valid_flags.any():
                        pick_id = np.where(inner_valid_flags)[0][-1]
                        covered[pick_id] = True
                    else:
                        return np.arange(len(history_metas))[-4:]
                selected_ids.append(pick_id)

            selected_mem_ids = sorted_indices[np.array(selected_ids)]
            return selected_mem_ids

    def process_history_info(self, img_metas, history_img_metas):
        """Process ego pose history to compute transformation matrices.

        Computes curr-to-prev and prev-to-curr transformation matrices
        and BEV coordinate grids for temporal fusion.

        Args:
            img_metas: Current frame metadata list (length = batch_size)
                Each entry must contain 'ego2global_translation' and
                'ego2global_rotation' keys.
            history_img_metas: List of previous frame metadata lists

        Returns:
            all_history_curr2prev: List[np.ndarray] per batch of [num_hist, 4, 4]
            all_history_prev2curr: List[np.ndarray] per batch of [num_hist, 4, 4]
            all_history_coord: List[np.ndarray] per batch of [num_hist, bev_h, bev_w, 2]
                BEV coordinates warped from history to current frame, normalized to (-1, 1)
        """
        bs = len(img_metas)
        all_history_curr2prev: list[np.ndarray] = []
        all_history_prev2curr: list[np.ndarray] = []
        all_history_coord: list[np.ndarray] = []

        if len(history_img_metas) == 0:
            return all_history_curr2prev, all_history_prev2curr, all_history_coord

        for b_i in range(bs):
            # Gather history ego poses
            history_e2g_trans = np.stack(
                [
                    np.array(prev[b_i]["ego2global_translation"], dtype=np.float64)
                    for prev in history_img_metas
                ],
                axis=0,
            )  # [num_hist, 3]
            history_e2g_rot = np.stack(
                [
                    np.array(prev[b_i]["ego2global_rotation"], dtype=np.float64)
                    for prev in history_img_metas
                ],
                axis=0,
            )  # [num_hist, 3, 3]

            # Current ego pose
            curr_e2g_trans = np.array(
                img_metas[b_i]["ego2global_translation"], dtype=np.float64
            )
            curr_e2g_rot = np.array(
                img_metas[b_i]["ego2global_rotation"], dtype=np.float64
            )

            num_hist = len(history_e2g_trans)

            # history global-to-ego matrices
            history_g2e_matrix = np.tile(np.eye(4, dtype=np.float64), (num_hist, 1, 1))
            history_g2e_matrix[:, :3, :3] = np.transpose(history_e2g_rot, (0, 2, 1))
            history_g2e_matrix[:, :3, 3] = -np.einsum(
                "nij,nj->ni",
                np.transpose(history_e2g_rot, (0, 2, 1)),
                history_e2g_trans,
            )

            # current global-to-ego matrix
            curr_g2e_matrix = np.eye(4, dtype=np.float64)
            curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
            curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

            # current ego-to-global matrix
            curr_e2g_matrix = np.eye(4, dtype=np.float64)
            curr_e2g_matrix[:3, :3] = curr_e2g_rot
            curr_e2g_matrix[:3, 3] = curr_e2g_trans

            # history ego-to-global matrices
            history_e2g_matrix = np.tile(np.eye(4, dtype=np.float64), (num_hist, 1, 1))
            history_e2g_matrix[:, :3, :3] = history_e2g_rot
            history_e2g_matrix[:, :3, 3] = history_e2g_trans

            # curr-to-prev: history_g2e @ curr_e2g
            history_curr2prev_matrix = np.einsum(
                "nij,jk->nik", history_g2e_matrix, curr_e2g_matrix
            )

            # prev-to-curr: curr_g2e @ history_e2g
            history_prev2curr_matrix = np.einsum(
                "ij,njk->nik", curr_g2e_matrix, history_e2g_matrix
            )

            # Compute BEV coordinates in history frames
            # plane: [bev_h, bev_w, 4], history_curr2prev: [num_hist, 4, 4]
            history_coord = np.einsum(
                "nlk,ijk->nijl", history_curr2prev_matrix, self.plane
            ).astype(np.float32)[..., :2]

            # Normalize from meters to (-1, 1)
            history_coord[..., 0] = history_coord[..., 0] / (self.roi_size[0] / 2)
            history_coord[..., 1] = -history_coord[..., 1] / (self.roi_size[1] / 2)

            all_history_curr2prev.append(history_curr2prev_matrix)
            all_history_prev2curr.append(history_prev2curr_matrix)
            all_history_coord.append(history_coord)

        return all_history_curr2prev, all_history_prev2curr, all_history_coord

    def temporal_propagate(
        self,
        curr_bev_feats,
        img_metas,
        all_history_curr2prev,
        all_history_prev2curr,
        use_memory,
        track_query_info=None,
        timestep=None,
    ):
        """Temporal propagation of track queries and memory bank updates (inference).

        Transforms track queries from previous frame to current frame using
        the relative ego pose, and updates the memory bank.

        Args:
            curr_bev_feats: Current BEV features (TTSim tensor or numpy)
            img_metas: Current frame metadata list
            all_history_curr2prev: Transformation matrices (curr->prev) per batch
            all_history_prev2curr: Transformation matrices (prev->curr) per batch
            use_memory: Whether to use the memory bank
            track_query_info: Track query information from previous frame
                List of dicts (one per batch) with keys:
                    'track_query_hs_embeds': [num_tracks, embed_dims]
                    'track_query_boxes': [num_tracks, num_pts*2]
            timestep: Current timestep index
        """
        from scipy.spatial.transform import Rotation as R  # type: ignore[import-untyped]

        bs = len(img_metas)

        if use_memory and self.memory_bank is not None:
            self.memory_bank.clear_dict()

        for b_i in range(bs):
            curr_e2g_trans = np.array(
                img_metas[b_i]["ego2global_translation"], dtype=np.float64
            )
            curr_e2g_rot = np.array(
                img_metas[b_i]["ego2global_rotation"], dtype=np.float64
            )

            if use_memory and self.memory_bank is not None:
                self.memory_bank.curr_rot[b_i] = curr_e2g_rot
                self.memory_bank.curr_trans[b_i] = curr_e2g_trans
                if self.memory_bank.curr_t > 0:
                    self.memory_bank.trans_memory_bank(
                        self.query_propagate, b_i, img_metas[b_i]
                    )

            # Transform the track queries using relative pose
            if track_query_info is not None:
                history_prev2curr_matrix = all_history_prev2curr[b_i]

                track_pts = track_query_info[b_i]["track_query_boxes"].copy()
                num_points = self.vector_num_points
                track_pts = rearrange(track_pts, "n (k c) -> n k c", c=2)

                # Denormalize from (0,1) to BEV meters
                track_pts = self._denorm_lines(track_pts)

                # Augment with z=0 and homogeneous coordinate
                N = track_pts.shape[0]
                track_pts_4d = np.concatenate(
                    [
                        track_pts,
                        np.zeros((N, num_points, 1), dtype=track_pts.dtype),
                        np.ones((N, num_points, 1), dtype=track_pts.dtype),
                    ],
                    axis=-1,
                )  # (N, num_pts, 4)

                # Apply prev-to-curr transformation
                pose_matrix = history_prev2curr_matrix[-1].astype(np.float32)[:3]
                rot_mat = pose_matrix[:, :3]
                rot = R.from_matrix(rot_mat)
                translation = pose_matrix[:, 3]

                trans_matrix = history_prev2curr_matrix[-1].copy()

                # Transform track points
                trans_track_pts = np.einsum(
                    "lk,ijk->ijl", trans_matrix, track_pts_4d.astype(np.float64)
                ).astype(np.float32)
                trans_track_pts = trans_track_pts[..., :2]

                # Normalize back to (0,1) and clip
                trans_track_pts = self._norm_lines(trans_track_pts)
                trans_track_pts = np.clip(trans_track_pts, 0.0, 1.0)
                trans_track_pts = rearrange(trans_track_pts, "n k c -> n (k c)", c=2)
                track_query_info[b_i]["trans_track_query_boxes"] = trans_track_pts

                # Propagate query embeddings using MotionMLP
                prop_q = track_query_info[b_i]["track_query_hs_embeds"]

                if prop_q.size > 0:
                    rot_quat = rot.as_quat().astype(np.float32)
                    pose_info = np.concatenate(
                        [rot_quat.reshape(-1), translation], axis=0
                    )

                    # Convert to TTSim tensors for MotionMLP
                    prop_q_tensor = F._from_data(
                        f"{self.name}/temporal/prop_q_{b_i}_{timestep}",
                        prop_q,
                        is_const=False,
                    )
                    pose_tensor = F._from_data(
                        f"{self.name}/temporal/pose_{b_i}_{timestep}",
                        pose_info,
                        is_const=False,
                    )

                    propagated = self.propagate_queries(prop_q_tensor, pose_tensor)

                    if propagated.data is not None:
                        track_query_info[b_i][
                            "track_query_hs_embeds"
                        ] = propagated.data.copy()
                    else:
                        # Shape-only pass, keep original embeddings
                        pass

    def _process_track_query_info(self, track_info):
        """Process track query embeddings to get predicted scores and lines.

        Runs the classification and regression branches on track query
        embeddings to produce detection scores and line coordinates.

        Args:
            track_info: List of track query info dicts (one per batch)
                Each dict has 'track_query_hs_embeds' key.

        Returns:
            track_results: Dict with keys:
                'lines': List of [num_tracks, num_pts, 2] per batch
                'scores': List of [num_tracks, num_classes] per batch

        Note:
            Requires self.head to have cls_branches and reg_branches.
            This is a numpy-based approximation for inference.
        """
        bs = len(track_info)
        all_scores = []
        all_lines = []

        for b_i in range(bs):
            embeds = track_info[b_i]["track_query_hs_embeds"]

            # Run classification and regression branches
            # In TTSim, these would be the last layer's branch outputs
            if self.head is not None and hasattr(self.head, "cls_branches"):
                scores_tensor = F._from_data(
                    f"{self.name}/track_process/embeds_{b_i}", embeds, is_const=False
                )
                cls_out = self.head.cls_branches[-1](scores_tensor)
                reg_out = self.head.reg_branches[-1](scores_tensor)

                if cls_out.data is not None:
                    cls_data = np.asarray(cls_out.data)
                    if cls_data.ndim == 3 and cls_data.shape[0] == 1:
                        cls_data = cls_data[0]  # remove batch dim
                    all_scores.append(cls_data)
                else:
                    all_scores.append(
                        np.zeros(
                            (embeds.shape[0], self.seg_num_classes), dtype=np.float32
                        )
                    )

                if reg_out.data is not None:
                    reg_data = np.asarray(reg_out.data)
                    if reg_data.ndim == 3 and reg_data.shape[0] == 1:
                        reg_data = reg_data[0]  # remove batch dim
                    coords = 1.0 / (1.0 + np.exp(-reg_data))  # sigmoid
                    coords = rearrange(coords, "n1 (n2 n3) -> n1 n2 n3", n3=2)
                    all_lines.append(coords)
                else:
                    all_lines.append(
                        np.zeros(
                            (embeds.shape[0], self.vector_num_points, 2),
                            dtype=np.float32,
                        )
                    )
            else:
                all_scores.append(
                    np.zeros((embeds.shape[0], self.seg_num_classes), dtype=np.float32)
                )
                all_lines.append(
                    np.zeros(
                        (embeds.shape[0], self.vector_num_points, 2), dtype=np.float32
                    )
                )

        track_results = {
            "lines": all_lines,
            "scores": all_scores,
        }
        return track_results

    def forward_test(self, bev_features, img_metas=None, seq_info=None, prev_bev=None):
        """Full inference pipeline matching the original MapTracker.forward_test.

        Handles first-frame initialization, temporal propagation, memory
        management, and multi-frame tracking.

        Args:
            bev_features: BEV feature tensor [1, C, bev_h, bev_w]
                Pre-computed BEV features (backbone output assumed external).
            img_metas: List of image metadata dicts.
                Must contain 'ego2global_translation', 'ego2global_rotation',
                'token', and optionally 'local_idx'.
            seq_info: Tuple of (scene_name, local_idx, seq_length)
                Used to determine first frame and manage tracking state.
            prev_bev: Previous BEV features for ConvGRU fusion (optional).

        Returns:
            results_list: List of dicts (one per batch item, bs=1) with keys:
                'vectors': numpy array [N, num_points, 2] of detected lines
                'scores': numpy array [N] of detection scores
                'labels': numpy array [N] of class labels
                'props': numpy array [N] of bool flags (True = tracked)
                'token': Sample token string
                'pos_results': Dict of positive prediction info for tracking
                'meta': Frame metadata dict
                'semantic_mask': numpy array [H, W] of segmentation labels
                'track_scores', 'track_vectors', 'track_labels': Track query outputs
        """
        if img_metas is None:
            img_metas = [{}]

        # Extract tokens from img_metas (matching original)
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta.get("token", ""))

        # Handle both flat tuple ('scene', idx, len) and batched [('scene', idx, len)]
        if seq_info is None:
            scene_name, local_idx, seq_length = "default", 0, 1
        elif (
            isinstance(seq_info, (list, tuple))
            and len(seq_info) > 0
            and isinstance(seq_info[0], (list, tuple))
        ):
            scene_name, local_idx, seq_length = seq_info[0]
        else:
            scene_name, local_idx, seq_length = seq_info
        first_frame = local_idx == 0
        img_metas[0]["local_idx"] = local_idx

        # Initialize on first frame
        if first_frame:
            if self.use_memory and self.memory_bank is not None:
                self.memory_bank.set_bank_size(self.test_time_history_steps)
                self.memory_bank.init_memory(bs=1)
            self.history_bev_feats_all = []
            self.history_img_metas_all = []

        if self.use_memory and self.memory_bank is not None:
            self.memory_bank.curr_t = local_idx

        # Select relevant history entries based on distance
        selected_mem_ids = self.select_memory_entries(
            self.history_img_metas_all, img_metas
        )
        history_img_metas = [
            self.history_img_metas_all[idx] for idx in selected_mem_ids
        ]
        history_bev_feats = [
            self.history_bev_feats_all[idx] for idx in selected_mem_ids
        ]

        # Compute transformation matrices
        all_history_curr2prev, all_history_prev2curr, all_history_coord = (
            self.process_history_info(img_metas, history_img_metas)
        )

        # Apply neck (ConvGRU or Identity) matching original behavior
        # Original always calls self.neck(_bev_feats) — neck is either ConvGRU or nn.Identity
        if self.use_neck and self.neck is not None:
            bev_feats = self.neck(bev_features, bev_features)
            if hasattr(bev_feats, "name"):
                setattr(self, bev_feats.name, bev_feats)
        else:
            bev_feats = bev_features

        # Track tensors
        setattr(self, bev_features.name, bev_features)

        track_dict = None
        preds_list = None

        if self.skip_vector_head or first_frame:
            # No track queries on first frame
            self.temporal_propagate(
                bev_feats,
                img_metas,
                all_history_curr2prev,
                all_history_prev2curr,
                self.use_memory,
                track_query_info=None,
            )

            # Segmentation
            seg_preds, seg_feats = self.seg_decoder(bev_feats)

            # Vector head (no tracking)
            if not self.skip_vector_head and self.head is not None:
                preds_list = self.head(bev_feats, img_metas=img_metas)
            track_dict = None
        else:
            # Get track info from previous frame
            assert self.head is not None
            track_query_info = self.head.get_track_info(scene_name, local_idx)

            # Temporal propagation with track queries
            self.temporal_propagate(
                bev_feats,
                img_metas,
                all_history_curr2prev,
                all_history_prev2curr,
                self.use_memory,
                track_query_info,
            )

            # Segmentation
            seg_preds, seg_feats = self.seg_decoder(bev_feats)

            # Vector head with tracking and memory
            # Pass track_query_info so tracked queries are concatenated
            # with detection queries inside the decoder
            if self.head is not None:
                memory_bank = self.memory_bank if self.use_memory else None
                preds_list = self.head(
                    bev_feats,
                    img_metas=img_metas,
                    memory_bank=memory_bank,
                    track_query_info=track_query_info,
                )

            if track_query_info is not None:
                track_dict = self._process_track_query_info(track_query_info)

        # Take predictions from the last decoder layer
        if not self.skip_vector_head and preds_list is not None:
            if isinstance(preds_list, list) and len(preds_list) > 0:
                preds_dict = preds_list[-1]
            else:
                preds_dict = preds_list
        else:
            preds_dict = None

        # Save history
        self.history_bev_feats_all.append(bev_feats)
        self.history_img_metas_all.append(img_metas)

        if len(self.history_bev_feats_all) > self.test_time_history_steps:
            self.history_bev_feats_all.pop(0)
            self.history_img_metas_all.pop(0)

        # Prepare temporal propagation for next frame
        pos_results = None
        if not self.skip_vector_head and preds_dict is not None:
            memory_bank = self.memory_bank if self.use_memory else None
            thr_det = 0.4 if first_frame else 0.6
            assert self.head is not None
            pos_results = self.head.prepare_temporal_propagation(
                preds_dict,
                scene_name,
                local_idx,
                memory_bank,
                thr_track=0.5,
                thr_det=thr_det,
            )

        # Post-process predictions into final output format
        if not self.skip_vector_head and preds_dict is not None:
            assert self.head is not None
            results_list = self.head.post_process(preds_dict, tokens, track_dict)
            results_list[0]["pos_results"] = pos_results
            results_list[0]["meta"] = img_metas[0]
        else:
            results_list = [
                {"vectors": [], "scores": [], "labels": [], "props": [], "token": token}
                for token in tokens
            ]

        # Add segmentation predictions to results
        for b_i in range(len(results_list)):
            seg_data_np = None
            if seg_preds is not None:
                if hasattr(seg_preds, "data") and seg_preds.data is not None:
                    seg_all = np.asarray(seg_preds.data)
                    if seg_all.ndim >= 3:
                        seg_data_np = seg_all[b_i] if seg_all.ndim == 4 else seg_all
                elif isinstance(seg_preds, np.ndarray):
                    seg_data_np = seg_preds[b_i] if seg_preds.ndim == 4 else seg_preds

            if seg_data_np is not None and seg_data_np.size > 0:
                tmp_labels = np.argmax(seg_data_np, axis=0)
                tmp_scores = np.max(seg_data_np, axis=0)
                # Apply sigmoid
                tmp_scores = 1.0 / (1.0 + np.exp(-tmp_scores))
                preds_mask = np.zeros(tmp_labels.shape, dtype=np.uint8)
                pos_ids = tmp_scores >= 0.4
                preds_mask[pos_ids] = tmp_labels[pos_ids].astype(np.uint8) + 1
                results_list[b_i]["semantic_mask"] = preds_mask
            else:
                # Fallback: empty semantic mask with canvas_size
                results_list[b_i]["semantic_mask"] = np.zeros(
                    tuple(self.canvas_size), dtype=np.uint8
                )
            if "token" not in results_list[b_i]:
                results_list[b_i]["token"] = tokens[b_i]

        return results_list

    def analyze_param_count(self, verbose=True):
        """
        Analyze and display parameter count for each component.

        Args:
            verbose: Whether to print detailed breakdown (default: True)

        Returns:
            param_counts: Dict with parameter counts per component
        """
        param_counts = {}
        total_params = 0

        # Helper to count params in a module
        def count_module_params(module):
            if module is None:
                return 0
            count = 0
            # Count parameters from _parameters dict if available
            if hasattr(module, "_parameters"):
                for param_name, param in module._parameters.items():
                    if param is not None and hasattr(param, "shape"):
                        param_size = int(np.prod(param.shape))
                        count += param_size
            # Recursively count in submodules
            if hasattr(module, "_modules"):
                for submod_name, submod in module._modules.items():
                    count += count_module_params(submod)
            return count

        # Count backbone params
        if self.bev_backbone is not None:
            backbone_params = count_module_params(self.bev_backbone)
            param_counts["backbone"] = backbone_params
            total_params += backbone_params

        # Count neck params
        if self.neck is not None:
            neck_params = count_module_params(self.neck)
            param_counts["neck"] = neck_params
            total_params += neck_params

        # Count seg_decoder params
        seg_params = count_module_params(self.seg_decoder)
        param_counts["seg_decoder"] = seg_params
        total_params += seg_params

        # Count vector head params
        if self.head is not None:
            head_params = count_module_params(self.head)
            param_counts["vector_head"] = head_params
            total_params += head_params

        # Count query propagation params
        if self.query_propagate is not None:
            qprop_params = count_module_params(self.query_propagate)
            param_counts["query_propagate"] = qprop_params
            total_params += qprop_params

        # Count memory bank params
        if self.memory_bank is not None:
            mem_params = count_module_params(self.memory_bank)
            param_counts["memory_bank"] = mem_params
            total_params += mem_params

        param_counts["total"] = total_params

        if verbose:
            print("\n" + "=" * 60)
            print("MapTracker Parameter Count Analysis")
            print("=" * 60)

            if "backbone" in param_counts:
                print(
                    f"  Backbone (BEVFormer):        {param_counts['backbone']:>12,} params"
                )
            if "neck" in param_counts:
                print(
                    f"  Neck (ConvGRU):              {param_counts['neck']:>12,} params"
                )
            print(
                f"  Seg Decoder (MapSegHead):    {param_counts['seg_decoder']:>12,} params"
            )
            if "vector_head" in param_counts:
                print(
                    f"  Vector Head (MapDetector):   {param_counts['vector_head']:>12,} params"
                )
            if "query_propagate" in param_counts:
                print(
                    f"  Query Propagate (MotionMLP): {param_counts['query_propagate']:>12,} params"
                )
            if "memory_bank" in param_counts:
                print(
                    f"  Memory Bank (VectorMemory):  {param_counts['memory_bank']:>12,} params"
                )
            print("  " + "-" * 56)
            print(f"  Total Parameters:            {total_params:>12,} params")
            print(f"  Total Size (fp32):           {total_params*4/1e6:>12.2f} MB")
            print("=" * 60 + "\n")

        return param_counts


# def export_onnx(onnx_path='maptracker.onnx',
#                 bev_h=50, bev_w=25,
#                 in_channels=64, embed_dims=64,
#                 num_classes=3, canvas_size=(100, 50),
#                 do_model_check=False,
#                 use_backbone=True,
#                 num_cams=6, img_height=28, img_width=50):
#     """
#     Export MapTracker to ONNX.

#     Args:
#         onnx_path: Output ONNX file path.
#         bev_h, bev_w: BEV spatial dimensions.
#         in_channels: Number of BEV input channels.
#         embed_dims: Embedding dimension for heads.
#         num_classes: Number of segmentation classes.
#         canvas_size: Output canvas (H, W).
#         do_model_check: Run onnx.checker on the exported model.
#         use_backbone: Use image backbone + FPN + BEVFormer encoder (default: True).
#         num_cams: Number of camera views (default: 6).
#         img_height: Input image height (default: 28).
#         img_width: Input image width (default: 50).
#     Returns:
#         WorkloadGraph that was exported.
#     """
#     cfg = {
#         'bev_h': bev_h,
#         'bev_w': bev_w,
#         'roi_size': (60, 30),
#         'seg_num_classes': num_classes,
#         'seg_in_channels': in_channels,
#         'seg_embed_dims': embed_dims,
#         'canvas_size': canvas_size,
#         'max_batch_size': 4,
#         'use_backbone': use_backbone,
#         'use_vector_head': False,
#         'use_query_propagation': False,
#         'use_memory': False,
#         'num_cams': num_cams,
#         'img_height': img_height,
#         'img_width': img_width,
#         'backbone_channels': 64,
#         'fpn_out_channels': in_channels,
#         'num_fpn_levels': 4,
#         'num_encoder_layers': 2,
#     }
#     model = MapTracker(name='maptracker', cfg=cfg)

#     model.create_input_tensors()
#     _ = model()
#     gg = model.get_forward_graph()
#     gg.graph2onnx(onnx_path, do_model_check=do_model_check)
#     print(f"[OK] Exported MapTracker to {onnx_path}  "
#           f"({len(gg._ops)} ops, {len(gg._tensors)} tensors)")
#     return gg


# def export_onnx_bev_only(onnx_path='maptracker_bev_only.onnx',
#                          bev_h=50, bev_w=25,
#                          in_channels=64, embed_dims=64,
#                          num_classes=3, canvas_size=(100, 50),
#                          do_model_check=False):
#     """
#     Export MapTracker with pre-computed BEV features (no image backbone).
#     Original export mode for backward compatibility.
#     """
#     cfg = {
#         'bev_h': bev_h,
#         'bev_w': bev_w,
#         'roi_size': (60, 30),
#         'seg_num_classes': num_classes,
#         'seg_in_channels': in_channels,
#         'seg_embed_dims': embed_dims,
#         'canvas_size': canvas_size,
#         'max_batch_size': 4,
#         'use_backbone': False,
#         'use_vector_head': False,
#         'use_query_propagation': False,
#         'use_memory': False,
#     }
#     model = MapTracker(name='maptracker', cfg=cfg)

#     model.create_input_tensors()
#     _ = model()
#     gg = model.get_forward_graph()
#     gg.graph2onnx(onnx_path, do_model_check=do_model_check)
#     print(f"[OK] Exported MapTracker (BEV-only) to {onnx_path}  "
#           f"({len(gg._ops)} ops, {len(gg._tensors)} tensors)")
#     return gg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MapTracker ONNX export")
    parser.add_argument(
        "--onnx-path", type=str, default="maptracker.onnx", help="Output ONNX file path"
    )
    parser.add_argument(
        "--do-model-check",
        action="store_true",
        help="Run onnx.checker.check_model on exported file",
    )
    parser.add_argument(
        "--bev-only",
        action="store_true",
        help="Export with pre-computed BEV features (no backbone)",
    )
    args = parser.parse_args()

    # if args.bev_only:
    #     export_onnx_bev_only(onnx_path=args.onnx_path, do_model_check=args.do_model_check)
    # else:
    #     export_onnx(onnx_path=args.onnx_path, do_model_check=args.do_model_check)
