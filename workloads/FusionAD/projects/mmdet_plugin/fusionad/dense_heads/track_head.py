
#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of BEVFormerTrackHead for FusionAD.

Inference-only conversion of the detection head.  Training-specific methods
(_get_target_single, get_targets, loss_single, loss) are omitted.

Classes:
  - ClsBranch           : Classification branch (Linear -> LN -> ReLU -> ... -> Linear).
  - RegBranch           : Regression branch (Linear -> ReLU -> ... -> Linear).
  - TrajRegBranch       : Trajectory regression branch.
  - BEVFormerTrackHead  : Main detection head module.
"""

# =============================================================================
# PYTORCH CODE
# =============================================================================
# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Linear, bias_init_with_prob
# from mmcv.utils import TORCH_VERSION, digit_version
#
# from mmdet.core import (multi_apply, multi_apply, reduce_mean)
# from mmdet.models.utils.transformer import inverse_sigmoid
# from mmdet.models import HEADS
# from mmdet.models.dense_heads import DETRHead
# from mmdet3d.core.bbox.coders import build_bbox_coder
# from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# from mmcv.runner import force_fp32, auto_fp16
#
#
# @HEADS.register_module()
# class BEVFormerTrackHead(DETRHead):
#     """Head of Detr3D.
#     Args:
#         with_box_refine (bool): Whether to refine the reference points
#             in the decoder. Defaults to False.
#         as_two_stage (bool) : Whether to generate the proposal from
#             the outputs of encoder.
#         transformer (obj:`ConfigDict`): ConfigDict is used for building
#             the Encoder and Decoder.
#         bev_h, bev_w (int): spatial shape of BEV queries.
#     """
#
#     def __init__(self,
#                  *args,
#                  with_box_refine=False,
#                  as_two_stage=False,
#                  transformer=None,
#                  bbox_coder=None,
#                  num_cls_fcs=2,
#                  code_weights=None,
#                  bev_h=30,
#                  bev_w=30,
#                  past_steps=4,
#                  fut_steps=4,
#                  **kwargs):
#
#         self.bev_h = bev_h
#         self.bev_w = bev_w
#         self.fp16_enabled = False
#
#         self.with_box_refine = with_box_refine
#         self.as_two_stage = as_two_stage
#         if self.as_two_stage:
#             transformer['as_two_stage'] = self.as_two_stage
#         if 'code_size' in kwargs:
#             self.code_size = kwargs['code_size']
#         else:
#             self.code_size = 10
#         if code_weights is not None:
#             self.code_weights = code_weights
#         else:
#             self.code_weights = [1.0, 1.0, 1.0,
#                                  1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
#
#         self.bbox_coder = build_bbox_coder(bbox_coder)
#         self.pc_range = self.bbox_coder.pc_range
#         self.real_w = self.pc_range[3] - self.pc_range[0]
#         self.real_h = self.pc_range[4] - self.pc_range[1]
#         self.num_cls_fcs = num_cls_fcs - 1
#         self.past_steps = past_steps
#         self.fut_steps = fut_steps
#         super(BEVFormerTrackHead, self).__init__(
#             *args, transformer=transformer, **kwargs)
#         self.code_weights = nn.Parameter(torch.tensor(
#             self.code_weights, requires_grad=False), requires_grad=False)
#
#     def _init_layers(self):
#         """Initialize classification branch and regression branch of head."""
#         cls_branch = []
#         for _ in range(self.num_reg_fcs):
#             cls_branch.append(Linear(self.embed_dims, self.embed_dims))
#             cls_branch.append(nn.LayerNorm(self.embed_dims))
#             cls_branch.append(nn.ReLU(inplace=True))
#         cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
#         fc_cls = nn.Sequential(*cls_branch)
#
#         reg_branch = []
#         for _ in range(self.num_reg_fcs):
#             reg_branch.append(Linear(self.embed_dims, self.embed_dims))
#             reg_branch.append(nn.ReLU())
#         reg_branch.append(Linear(self.embed_dims, self.code_size))
#         reg_branch = nn.Sequential(*reg_branch)
#
#         past_traj_reg_branch = []
#         for _ in range(self.num_reg_fcs):
#             past_traj_reg_branch.append(
#                 Linear(self.embed_dims, self.embed_dims))
#             past_traj_reg_branch.append(nn.ReLU())
#         past_traj_reg_branch.append(
#             Linear(self.embed_dims, (self.past_steps + self.fut_steps)*2))
#         past_traj_reg_branch = nn.Sequential(*past_traj_reg_branch)
#
#         def _get_clones(module, N):
#             return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
#         # last reg_branch is used to generate proposal from
#         # encode feature map when as_two_stage is True.
#         num_pred = (self.transformer.decoder.num_layers + 1) if \
#             self.as_two_stage else self.transformer.decoder.num_layers
#
#         if self.with_box_refine:
#             self.cls_branches = _get_clones(fc_cls, num_pred)
#             self.reg_branches = _get_clones(reg_branch, num_pred)
#             self.past_traj_reg_branches = _get_clones(
#                 past_traj_reg_branch, num_pred)
#         else:
#             self.cls_branches = nn.ModuleList(
#                 [fc_cls for _ in range(num_pred)])
#             self.reg_branches = nn.ModuleList(
#                 [reg_branch for _ in range(num_pred)])
#             self.past_traj_reg_branches = nn.ModuleList(
#                 [past_traj_reg_branch for _ in range(num_pred)])
#         if not self.as_two_stage:
#             self.bev_embedding = nn.Embedding(
#                 self.bev_h * self.bev_w, self.embed_dims)
#             self.query_embedding = nn.Embedding(self.num_query,
#                                                 self.embed_dims * 2)
#
#     def init_weights(self):
#         """Initialize weights of the DeformDETR head."""
#         self.transformer.init_weights()
#         if self.loss_cls.use_sigmoid:
#             bias_init = bias_init_with_prob(0.01)
#             for m in self.cls_branches:
#                 nn.init.constant_(m[-1].bias, bias_init)
#
#     def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None, pts_feats=None):
#         bs, num_cam, _, _, _ = mlvl_feats[0].shape
#         dtype = mlvl_feats[0].dtype
#         bev_queries = self.bev_embedding.weight.to(dtype)
#
#         bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
#                                device=bev_queries.device).to(dtype)
#         bev_pos = self.positional_encoding(bev_mask).to(dtype)
#         bev_embed = self.transformer.get_bev_features(
#             mlvl_feats,
#             bev_queries,
#             self.bev_h,
#             self.bev_w,
#             grid_length=(self.real_h / self.bev_h,
#                          self.real_w / self.bev_w),
#             bev_pos=bev_pos,
#             prev_bev=prev_bev,
#             img_metas=img_metas,
#             pts_feats=pts_feats,
#         )
#         return bev_embed, bev_pos
#
#     def get_detections(
#         self,
#         bev_embed,
#         object_query_embeds=None,
#         ref_points=None,
#         img_metas=None,
#     ):
#         assert bev_embed.shape[0] == self.bev_h * self.bev_w
#         hs, init_reference, inter_references = self.transformer.get_states_and_refs(
#             bev_embed,
#             object_query_embeds,
#             self.bev_h,
#             self.bev_w,
#             reference_points=ref_points,
#             reg_branches=self.reg_branches if self.with_box_refine else None,
#             cls_branches=self.cls_branches if self.as_two_stage else None,
#             img_metas=img_metas,
#         )
#         hs = hs.permute(0, 2, 1, 3)
#         outputs_classes = []
#         outputs_coords = []
#         outputs_trajs = []
#         for lvl in range(hs.shape[0]):
#             if lvl == 0:
#                 # reference = init_reference
#                 reference = ref_points.sigmoid()
#             else:
#                 reference = inter_references[lvl - 1]
#                 # ref_size_base = inter_box_sizes[lvl - 1]
#             reference = inverse_sigmoid(reference)
#             outputs_class = self.cls_branches[lvl](hs[lvl])
#             tmp = self.reg_branches[lvl](hs[lvl])  # xydxdyxdz
#             outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
#                 tmp.shape[0], -1, self.past_steps + self.fut_steps, 2)
#             # TODO: check the shape of reference
#             assert reference.shape[-1] == 3
#             tmp[..., 0:2] += reference[..., 0:2]
#             tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
#             tmp[..., 4:5] += reference[..., 2:3]
#             tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
#
#             last_ref_points = torch.cat(
#                 [tmp[..., 0:2], tmp[..., 4:5]], dim=-1,
#             )
#
#             tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
#                              self.pc_range[0]) + self.pc_range[0])
#             tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
#                              self.pc_range[1]) + self.pc_range[1])
#             tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
#                              self.pc_range[2]) + self.pc_range[2])
#
#             # tmp[..., 2:4] = tmp[..., 2:4] + ref_size_basse[..., 0:2]
#             # tmp[..., 5:6] = tmp[..., 5:6] + ref_size_basse[..., 2:3]
#
#             # TODO: check if using sigmoid
#             outputs_coord = tmp
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)
#             outputs_trajs.append(outputs_past_traj)
#         outputs_classes = torch.stack(outputs_classes)
#         outputs_coords = torch.stack(outputs_coords)
#         outputs_trajs = torch.stack(outputs_trajs)
#         last_ref_points = inverse_sigmoid(last_ref_points)
#         outs = {
#             'all_cls_scores': outputs_classes,
#             'all_bbox_preds': outputs_coords,
#             'all_past_traj_preds': outputs_trajs,
#             'enc_cls_scores': None,
#             'enc_bbox_preds': None,
#             'last_ref_points': last_ref_points,
#             'query_feats': hs,
#         }
#         return outs
#
#     def _get_target_single(self,
#                            cls_score,
#                            bbox_pred,
#                            gt_labels,
#                            gt_bboxes,
#                            gt_bboxes_ignore=None):
#         """"Compute regression and classification targets for one image.
#         Outputs from a single decoder layer of a single feature level are used.
#         Args:
#             cls_score (Tensor): Box score logits from a single decoder layer
#                 for one image. Shape [num_query, cls_out_channels].
#             bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
#                 for one image, with normalized coordinate (cx, cy, w, h) and
#                 shape [num_query, 4].
#             gt_bboxes (Tensor): Ground truth bboxes for one image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (Tensor): Ground truth class indices for one image
#                 with shape (num_gts, ).
#             gt_bboxes_ignore (Tensor, optional): Bounding boxes
#                 which can be ignored. Default None.
#         Returns:
#             tuple[Tensor]: a tuple containing the following for one image.
#                 - labels (Tensor): Labels of each image.
#                 - label_weights (Tensor]): Label weights of each image.
#                 - bbox_targets (Tensor): BBox targets of each image.
#                 - bbox_weights (Tensor): BBox weights of each image.
#                 - pos_inds (Tensor): Sampled positive indices for each image.
#                 - neg_inds (Tensor): Sampled negative indices for each image.
#         """
#
#         num_bboxes = bbox_pred.size(0)
#         # assigner and sampler
#         gt_c = gt_bboxes.shape[-1]
#
#         assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
#                                              gt_labels, gt_bboxes_ignore)
#
#         sampling_result = self.sampler.sample(assign_result, bbox_pred,
#                                               gt_bboxes)
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#
#         # label targets
#         labels = gt_bboxes.new_full((num_bboxes,),
#                                     self.num_classes,
#                                     dtype=torch.long)
#         labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
#         label_weights = gt_bboxes.new_ones(num_bboxes)
#
#         # bbox targets
#         bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
#         bbox_weights = torch.zeros_like(bbox_pred)
#         bbox_weights[pos_inds] = 1.0
#
#         # DETR
#         bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
#         return (labels, label_weights, bbox_targets, bbox_weights,
#                 pos_inds, neg_inds)
#
#     def get_targets(self,
#                     cls_scores_list,
#                     bbox_preds_list,
#                     gt_bboxes_list,
#                     gt_labels_list,
#                     gt_bboxes_ignore_list=None):
#         """"Compute regression and classification targets for a batch image.
#         Outputs from a single decoder layer of a single feature level are used.
#         Args:
#             cls_scores_list (list[Tensor]): Box score logits from a single
#                 decoder layer for each image with shape [num_query,
#                 cls_out_channels].
#             bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
#                 decoder layer for each image, with normalized coordinate
#                 (cx, cy, w, h) and shape [num_query, 4].
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                 boxes which can be ignored for each image. Default None.
#         Returns:
#             tuple: a tuple containing the following targets.
#                 - labels_list (list[Tensor]): Labels for all images.
#                 - label_weights_list (list[Tensor]): Label weights for all \
#                     images.
#                 - bbox_targets_list (list[Tensor]): BBox targets for all \
#                     images.
#                 - bbox_weights_list (list[Tensor]): BBox weights for all \
#                     images.
#                 - num_total_pos (int): Number of positive samples in all \
#                     images.
#                 - num_total_neg (int): Number of negative samples in all \
#                     images.
#         """
#         assert gt_bboxes_ignore_list is None, \
#             'Only supports for gt_bboxes_ignore setting to None.'
#         num_imgs = len(cls_scores_list)
#         gt_bboxes_ignore_list = [
#             gt_bboxes_ignore_list for _ in range(num_imgs)
#         ]
#
#         (labels_list, label_weights_list, bbox_targets_list,
#          bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
#             self._get_target_single, cls_scores_list, bbox_preds_list,
#             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))
#         return (labels_list, label_weights_list, bbox_targets_list,
#                 bbox_weights_list, num_total_pos, num_total_neg)
#
#     def loss_single(self,
#                     cls_scores,
#                     bbox_preds,
#                     gt_bboxes_list,
#                     gt_labels_list,
#                     gt_bboxes_ignore_list=None):
#         """"Loss function for outputs from a single decoder layer of a single
#         feature level.
#         Args:
#             cls_scores (Tensor): Box score logits from a single decoder layer
#                 for all images. Shape [bs, num_query, cls_out_channels].
#             bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
#                 for all images, with normalized coordinate (cx, cy, w, h) and
#                 shape [bs, num_query, 4].
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                 boxes which can be ignored for each image. Default None.
#         Returns:
#             dict[str, Tensor]: A dictionary of loss components for outputs from
#                 a single decoder layer.
#         """
#         num_imgs = cls_scores.size(0)
#         cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
#         bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
#         cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
#                                            gt_bboxes_list, gt_labels_list,
#                                            gt_bboxes_ignore_list)
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          num_total_pos, num_total_neg) = cls_reg_targets
#         labels = torch.cat(labels_list, 0)
#         label_weights = torch.cat(label_weights_list, 0)
#         bbox_targets = torch.cat(bbox_targets_list, 0)
#         bbox_weights = torch.cat(bbox_weights_list, 0)
#
#         # classification loss
#         cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
#         # construct weighted avg_factor to match with the official DETR repo
#         cls_avg_factor = num_total_pos * 1.0 + \
#             num_total_neg * self.bg_cls_weight
#         if self.sync_cls_avg_factor:
#             cls_avg_factor = reduce_mean(
#                 cls_scores.new_tensor([cls_avg_factor]))
#
#         cls_avg_factor = max(cls_avg_factor, 1)
#         loss_cls = self.loss_cls(
#             cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
#
#         # Compute the average number of gt boxes accross all gpus, for
#         # normalization purposes
#         num_total_pos = loss_cls.new_tensor([num_total_pos])
#         num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
#
#         # regression L1 loss
#         bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
#         normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
#         isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
#         bbox_weights = bbox_weights * self.code_weights
#
#         loss_bbox = self.loss_bbox(
#             bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
#                                                                :10], bbox_weights[isnotnan, :10],
#             avg_factor=num_total_pos)
#         loss_cls = torch.nan_to_num(loss_cls)
#         loss_bbox = torch.nan_to_num(loss_bbox)
#         return loss_cls, loss_bbox
#
#     @force_fp32(apply_to=('preds_dicts'))
#     def loss(self,
#              gt_bboxes_list,
#              gt_labels_list,
#              preds_dicts,
#              gt_bboxes_ignore=None,
#              img_metas=None):
#         """"Loss function.
#         Args:
#
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
#                 with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels_list (list[Tensor]): Ground truth class indices for each
#                 image with shape (num_gts, ).
#             preds_dicts:
#                 all_cls_scores (Tensor): Classification score of all
#                     decoder layers, has shape
#                     [nb_dec, bs, num_query, cls_out_channels].
#                 all_bbox_preds (Tensor): Sigmoid regression
#                     outputs of all decode layers. Each is a 4D-tensor with
#                     normalized coordinate format (cx, cy, w, h) and shape
#                     [nb_dec, bs, num_query, 4].
#                 enc_cls_scores (Tensor): Classification scores of
#                     points on encode feature map , has shape
#                     (N, h*w, num_classes). Only be passed when as_two_stage is
#                     True, otherwise is None.
#                 enc_bbox_preds (Tensor): Regression results of each points
#                     on the encode feature map, has shape (N, h*w, 4). Only be
#                     passed when as_two_stage is True, otherwise is None.
#             gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
#                 which can be ignored for each image. Default None.
#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         assert gt_bboxes_ignore is None, \
#             f'{self.__class__.__name__} only supports ' \
#             f'for gt_bboxes_ignore setting to None.'
#
#         all_cls_scores = preds_dicts['all_cls_scores']
#         all_bbox_preds = preds_dicts['all_bbox_preds']
#         enc_cls_scores = preds_dicts['enc_cls_scores']
#         enc_bbox_preds = preds_dicts['enc_bbox_preds']
#
#         num_dec_layers = len(all_cls_scores)
#         device = gt_labels_list[0].device
#
#         gt_bboxes_list = [torch.cat(
#             (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
#             dim=1).to(device) for gt_bboxes in gt_bboxes_list]
#
#         all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
#         all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
#         all_gt_bboxes_ignore_list = [
#             gt_bboxes_ignore for _ in range(num_dec_layers)
#         ]
#
#         losses_cls, losses_bbox = multi_apply(
#             self.loss_single, all_cls_scores, all_bbox_preds,
#             all_gt_bboxes_list, all_gt_labels_list,
#             all_gt_bboxes_ignore_list)
#
#         loss_dict = dict()
#         # loss of proposal generated from encode feature map.
#         if enc_cls_scores is not None:
#             binary_labels_list = [
#                 torch.zeros_like(gt_labels_list[i])
#                 for i in range(len(all_gt_labels_list))
#             ]
#             enc_loss_cls, enc_losses_bbox = \
#                 self.loss_single(enc_cls_scores, enc_bbox_preds,
#                                  gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
#             loss_dict['enc_loss_cls'] = enc_loss_cls
#             loss_dict['enc_loss_bbox'] = enc_losses_bbox
#
#         # loss from the last decoder layer
#         loss_dict['loss_cls'] = losses_cls[-1]
#         loss_dict['loss_bbox'] = losses_bbox[-1]
#
#         # loss from other decoder layers
#         num_dec_layer = 0
#         for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
#                                            losses_bbox[:-1]):
#             loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
#             loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
#             num_dec_layer += 1
#         return loss_dict
#
#     @force_fp32(apply_to=('preds_dicts'))
#     def get_bboxes(self, preds_dicts, img_metas, rescale=False):
#         """Generate bboxes from bbox head predictions.
#         Args:
#             preds_dicts (tuple[list[dict]]): Prediction results.
#             img_metas (list[dict]): Point cloud and image's meta info.
#         Returns:
#             list[dict]: Decoded bbox, scores and labels after nms.
#         """
#
#         preds_dicts = self.bbox_coder.decode(preds_dicts)
#
#         num_samples = len(preds_dicts)
#         ret_list = []
#         for i in range(num_samples):
#             preds = preds_dicts[i]
#             bboxes = preds['bboxes']
#
#             bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
#
#             code_size = bboxes.shape[-1]
#             bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
#             scores = preds['scores']
#             labels = preds['labels']
#             bbox_index = preds['bbox_index']
#             mask = preds['mask']
#
#             ret_list.append([bboxes, scores, labels, bbox_index, mask])
#
#         return ret_list

# =============================================================================
# TTSim CODE
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


from ..modules.builder_utils import LayerNorm, InverseSigmoid, inverse_sigmoid_np


# ======================================================================
# ClsBranch
# ======================================================================

class ClsBranch(SimNN.Module):
    """
    Classification branch.

    Architecture: num_reg_fcs x (Linear + LayerNorm + ReLU),
    followed by a final Linear -> cls_out_channels.

    Args:
        name (str): Module name.
        embed_dims (int): Input / hidden dimension.
        num_reg_fcs (int): Number of hidden FC layers.
        cls_out_channels (int): Number of output classes.
    """

    def __init__(self, name, embed_dims, num_reg_fcs, cls_out_channels):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = cls_out_channels

        fc_list = []
        for i in range(num_reg_fcs):
            fc_list.append(
                SimNN.Linear(f'{name}.fc{i}',
                             in_features=embed_dims, out_features=embed_dims))
        fc_list.append(
            SimNN.Linear(f'{name}.out',
                         in_features=embed_dims, out_features=cls_out_channels))
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


# ======================================================================
# RegBranch
# ======================================================================

class RegBranch(SimNN.Module):
    """
    Regression branch.

    Architecture: num_reg_fcs x (Linear + ReLU),
    followed by a final Linear -> out_channels.

    Args:
        name (str): Module name.
        embed_dims (int): Input / hidden dimension.
        num_reg_fcs (int): Number of hidden FC layers.
        out_channels (int): Output dimension (e.g. code_size).
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


# ======================================================================
# BEVFormerTrackHead
# ======================================================================

class BEVFormerTrackHead(SimNN.Module):
    """
    TTSim implementation of BEVFormerTrackHead (inference only).

    Creates classification, regression, and trajectory-regression branches,
    wires them through the transformer decoder layers, and applies
    reference-point refinement + pc_range denormalization.

    Args:
        name (str): Module name.
        embed_dims (int): Query embedding dimension.
        num_reg_fcs (int): Number of hidden FC layers in each branch.
        cls_out_channels (int): Number of classification output channels.
        code_size (int): Bbox code size (default 10).
        bev_h (int): BEV grid height.
        bev_w (int): BEV grid width.
        past_steps (int): Number of past trajectory steps.
        fut_steps (int): Number of future trajectory steps.
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        num_pred (int): Number of decoder prediction layers.
        num_query (int): Number of object queries.
        with_box_refine (bool): If True, each layer has independent branch
            weights; otherwise branches share weights (only first is used).
        transformer: Pre-built TTSim PerceptionTransformer module.
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_reg_fcs=2,
                 cls_out_channels=10,
                 code_size=10,
                 bev_h=30,
                 bev_w=30,
                 past_steps=4,
                 fut_steps=4,
                 pc_range=None,
                 num_pred=6,
                 num_query=300,
                 with_box_refine=False,
                 transformer=None):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.cls_out_channels = cls_out_channels
        self.code_size = code_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.num_pred = num_pred
        self.num_query = num_query
        self.with_box_refine = with_box_refine
        self.transformer = transformer

        if pc_range is None:
            pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.pc_range = pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]

        traj_out = (past_steps + fut_steps) * 2

        # ---- Build branch modules ----
        self._build_branches(traj_out)

        # ---- Embeddings (set externally before inference) ----
        self.bev_embedding_data = None
        self.query_embedding_data = None

        # ---- Inverse-sigmoid modules (one per decoder layer + final) ----
        self.inv_sig = []
        for lvl in range(num_pred):
            inv = InverseSigmoid(f'{name}.inv_sig_L{lvl}')
            setattr(self, f'inv_sig_L{lvl}', inv)
            self.inv_sig.append(inv)
        self.inv_sig_final = InverseSigmoid(f'{name}.inv_sig_final')

        # ---- Per-layer post-processing ops ----
        for lvl in range(num_pred):
            self._create_layer_ops(lvl)

        # ---- PC-range scale / offset constants ----
        self.range_x_const = F._from_data(
            f'{name}.range_x',
            np.array([pc_range[3] - pc_range[0]], dtype=np.float32),
            is_const=True)
        self.offset_x_const = F._from_data(
            f'{name}.offset_x',
            np.array([pc_range[0]], dtype=np.float32),
            is_const=True)
        self.range_y_const = F._from_data(
            f'{name}.range_y',
            np.array([pc_range[4] - pc_range[1]], dtype=np.float32),
            is_const=True)
        self.offset_y_const = F._from_data(
            f'{name}.offset_y',
            np.array([pc_range[1]], dtype=np.float32),
            is_const=True)
        self.range_z_const = F._from_data(
            f'{name}.range_z',
            np.array([pc_range[5] - pc_range[2]], dtype=np.float32),
            is_const=True)
        self.offset_z_const = F._from_data(
            f'{name}.offset_z',
            np.array([pc_range[2]], dtype=np.float32),
            is_const=True)

        super().link_op2module()

    # ------------------------------------------------------------------
    # Branch construction
    # ------------------------------------------------------------------

    def _build_branches(self, traj_out):
        """Create cls / reg / traj branches for each prediction layer.

        Also creates a separate set of reg branches for the decoder's
        iterative reference-point refinement (``dec_reg_branches``).
        Separate modules avoid ONNX name collisions when the same branch
        architecture is evaluated on two different inputs.
        """
        cls_list = []
        reg_list = []
        traj_list = []
        dec_reg_list = []  # decoder-only reg branches (separate names)

        n = 1 if not self.with_box_refine else self.num_pred
        for i in range(n):
            cls_list.append(ClsBranch(
                f'{self.name}.cls_branch_{i}',
                self.embed_dims, self.num_reg_fcs, self.cls_out_channels))
            reg_list.append(RegBranch(
                f'{self.name}.reg_branch_{i}',
                self.embed_dims, self.num_reg_fcs, self.code_size))
            traj_list.append(RegBranch(
                f'{self.name}.traj_branch_{i}',
                self.embed_dims, self.num_reg_fcs, traj_out))
            dec_reg_list.append(RegBranch(
                f'{self.name}.dec_reg_branch_{i}',
                self.embed_dims, self.num_reg_fcs, self.code_size))

        if self.with_box_refine:
            self.cls_branches = SimNN.ModuleList(cls_list)
            self.reg_branches = SimNN.ModuleList(reg_list)
            self.traj_branches = SimNN.ModuleList(traj_list)
            self.dec_reg_branches = SimNN.ModuleList(dec_reg_list)
        else:
            self._cls_branch_shared = cls_list[0]
            self._reg_branch_shared = reg_list[0]
            self._traj_branch_shared = traj_list[0]
            self._dec_reg_branch_shared = dec_reg_list[0]

    def _get_cls_branch(self, lvl):
        if self.with_box_refine:
            return self.cls_branches[lvl]
        return self._cls_branch_shared

    def _get_reg_branch(self, lvl):
        if self.with_box_refine:
            return self.reg_branches[lvl]
        return self._reg_branch_shared

    def _get_traj_branch(self, lvl):
        if self.with_box_refine:
            return self.traj_branches[lvl]
        return self._traj_branch_shared

    def _get_dec_reg_branch(self, lvl):
        """Return decoder-specific reg branch (used for ref-point refinement)."""
        if self.with_box_refine:
            return self.dec_reg_branches[lvl]
        return self._dec_reg_branch_shared

    # ------------------------------------------------------------------
    # Per-layer ops (created once in __init__)
    # ------------------------------------------------------------------

    def _create_layer_ops(self, lvl):
        """Pre-create graph ops for one decoder layer's post-processing."""
        pfx = f'{self.name}.L{lvl}'

        setattr(self, f'sig_ref_L{lvl}', F.Sigmoid(f'{pfx}.sig_ref'))

        setattr(self, f'add_xy_L{lvl}', F.Add(f'{pfx}.add_xy'))
        setattr(self, f'sig_xy_L{lvl}', F.Sigmoid(f'{pfx}.sig_xy'))

        setattr(self, f'add_z_L{lvl}', F.Add(f'{pfx}.add_z'))
        setattr(self, f'sig_z_L{lvl}', F.Sigmoid(f'{pfx}.sig_z'))

        setattr(self, f'concat_ref_L{lvl}',
                F.ConcatX(f'{pfx}.concat_ref', axis=-1))

        setattr(self, f'mul_x_L{lvl}', F.Mul(f'{pfx}.mul_x'))
        setattr(self, f'add_ox_L{lvl}', F.Add(f'{pfx}.add_ox'))
        setattr(self, f'mul_y_L{lvl}', F.Mul(f'{pfx}.mul_y'))
        setattr(self, f'add_oy_L{lvl}', F.Add(f'{pfx}.add_oy'))
        setattr(self, f'mul_z_L{lvl}', F.Mul(f'{pfx}.mul_z'))
        setattr(self, f'add_oz_L{lvl}', F.Add(f'{pfx}.add_oz'))

        setattr(self, f'concat_coord_L{lvl}',
                F.ConcatX(f'{pfx}.concat_coord', axis=-1))

    # ------------------------------------------------------------------
    # Slice helper (mirrors util.py pattern)
    # ------------------------------------------------------------------

    def _slice(self, src, field_name, start, end):
        """
        Slice the last dimension of *src*: ``src[..., start:end]``.

        Creates the required constant tensors and SliceF op dynamically,
        registering them on *self* for graph tracking.
        """
        ndim = len(src.shape)
        starts = [0] * (ndim - 1) + [start]
        ends = [int(s) for s in src.shape[:-1]] + [end]
        axes = list(range(ndim))
        out_shape = list(src.shape[:-1]) + [end - start]

        st = F._from_data(f'{self.name}.{field_name}_st',
                          np.array(starts, dtype=np.int64), is_const=True)
        setattr(self, st.name, st)
        en = F._from_data(f'{self.name}.{field_name}_en',
                          np.array(ends, dtype=np.int64), is_const=True)
        setattr(self, en.name, en)
        ax = F._from_data(f'{self.name}.{field_name}_ax',
                          np.array(axes, dtype=np.int64), is_const=True)
        setattr(self, ax.name, ax)
        sl = F.SliceF(f'{self.name}.{field_name}_sl', out_shape=out_shape)
        setattr(self, sl.name, sl)
        result = sl(src, st, en, ax)
        setattr(self, result.name, result)
        return result

    # ------------------------------------------------------------------
    # get_bev_features
    # ------------------------------------------------------------------

    def get_bev_features(self, mlvl_feats, bev_pos, prev_bev=None,
                         img_metas=None, pts_feats=None):
        """
        Extract BEV features through the transformer encoder.

        Args:
            mlvl_feats: Multi-level image features (list of SimTensors).
            bev_pos: Pre-computed BEV positional encoding
                     (numpy array [1, embed_dims, bev_h, bev_w]).
            prev_bev: Previous frame BEV features (optional).
            img_metas: Image metadata dict.
            pts_feats: LiDAR point features (optional).

        Returns:
            tuple: (bev_embed, bev_pos)
        """
        if self.bev_embedding_data is None:
            raise RuntimeError(
                "bev_embedding_data not set — call load_weights() first.")

        bev_queries = F._from_data(
            f'{self.name}.bev_queries',
            self.bev_embedding_data.astype(np.float32), is_const=True)
        setattr(self, bev_queries.name, bev_queries)

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
            pts_feats=pts_feats,
        )
        return bev_embed, bev_pos

    # ------------------------------------------------------------------
    # get_detections
    # ------------------------------------------------------------------

    def get_detections(self, bev_embed, object_query_embeds=None,
                       ref_points=None, img_metas=None):
        """
        Run the detection decoder and apply cls/reg/traj branches.

        Args:
            bev_embed: BEV features from the encoder.
                Shape [bev_h*bev_w, bs, embed_dims].
            object_query_embeds: Query embeddings, numpy or SimTensor
                [num_query, 2*embed_dims].
            ref_points: Reference points, numpy or SimTensor
                [num_query, 3] or [bs, num_query, 3].
            img_metas: Image metadata dict (passed through to transformer).

        Returns:
            dict with keys:
              - all_cls_scores  : np.ndarray [num_pred, bs, nq, cls_out]
              - all_bbox_preds  : np.ndarray [num_pred, bs, nq, code_size]
              - all_past_traj_preds : np.ndarray [num_pred, bs, nq, steps, 2]
              - last_ref_points : np.ndarray [bs, nq, 3]
              - query_feats     : np.ndarray [num_pred, bs, nq, embed_dims]
        """
        bev_np = bev_embed.data if hasattr(bev_embed, 'data') else bev_embed
        if bev_np is None:
            bev_np = np.zeros(bev_embed.shape, dtype=np.float32)
        assert bev_np.shape[0] == self.bev_h * self.bev_w

        hs, init_reference, inter_references = \
            self.transformer.get_states_and_refs(
                bev_embed,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                reference_points=ref_points,
                reg_branches=(
                    [self._get_dec_reg_branch(i) for i in range(self.num_pred)]
                    if self.with_box_refine else None),
                cls_branches=None,
                img_metas=img_metas,
            )

        # hs is a list of SimTensors (one per layer).
        # Each has shape [nq, bs, embed_dims] (seq-first).
        # Permute to [bs, nq, embed_dims] per layer.
        # In graph-only mode, .data may be None; use synthetic zeros.
        def _to_np(t):
            """Extract numpy from SimTensor; return zeros if .data is None."""
            if hasattr(t, 'data') and t.data is not None:
                return t.data
            if hasattr(t, 'shape'):
                return np.zeros(t.shape, dtype=np.float32)
            return np.array(t, dtype=np.float32)

        hs_np_layers = []
        for layer_out in hs:
            arr = _to_np(layer_out)
            hs_np_layers.append(arr.transpose(1, 0, 2))
        hs_np = np.stack(hs_np_layers, axis=0)  # [num_pred, bs, nq, embed]

        num_pred = hs_np.shape[0]
        bs = hs_np.shape[1]
        nq = hs_np.shape[2]

        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        last_ref_points_np = None

        for lvl in range(num_pred):
            # ---- Reference for this layer ----
            if lvl == 0:
                if ref_points is not None:
                    rp_np = _to_np(ref_points)
                else:
                    # No ref_points provided — transformer computed them
                    # init_reference is numpy [bs, nq, 3] or None
                    if init_reference is not None:
                        rp_np = init_reference if isinstance(init_reference, np.ndarray) \
                            else _to_np(init_reference)
                    else:
                        rp_np = np.zeros((bs, nq, 3), dtype=np.float32)
                if not isinstance(rp_np, np.ndarray):
                    rp_np = np.array(rp_np, dtype=np.float32)
                if rp_np.ndim == 2:
                    rp_np = np.tile(rp_np[np.newaxis], (bs, 1, 1))
                reference_np = 1.0 / (1.0 + np.exp(
                    -rp_np.astype(np.float64))).astype(np.float32)
            else:
                ref_l = inter_references[lvl - 1]
                reference_np = _to_np(ref_l)

            # ---- Transpose decoder output to batch-first (graph-connected) ----
            # hs[lvl] is [nq, bs, embed_dims] (seq-first from decoder)
            # Branches expect [bs, nq, embed_dims]
            _hs_perm = F.Transpose(f'{self.name}.hs_perm_L{lvl}',
                                   perm=[1, 0, 2])
            setattr(self, _hs_perm.name, _hs_perm)
            hs_lvl_t = _hs_perm(hs[lvl])
            setattr(self, hs_lvl_t.name, hs_lvl_t)

            # ---- Apply branches (graph ops) ----
            cls_out = self._get_cls_branch(lvl)(hs_lvl_t)
            setattr(self, cls_out.name, cls_out)

            reg_out = self._get_reg_branch(lvl)(hs_lvl_t)
            setattr(self, reg_out.name, reg_out)

            traj_out = self._get_traj_branch(lvl)(hs_lvl_t)
            setattr(self, traj_out.name, traj_out)

            # ---- Extract numpy for post-processing ----
            cls_np = _to_np(cls_out)    # [bs, nq, cls_out_channels]
            tmp_np = _to_np(reg_out).copy()  # [bs, nq, code_size]

            traj_np = _to_np(traj_out).reshape(
                bs, nq, self.past_steps + self.fut_steps, 2)

            # ---- Reference-point refinement ----
            # inverse_sigmoid on reference so we can add raw offsets
            ref_logits = inverse_sigmoid_np(reference_np)
            assert ref_logits.shape[-1] == 3

            # Add xy offset and sigmoid
            tmp_np[..., 0:2] += ref_logits[..., 0:2]
            tmp_np[..., 0:2] = 1.0 / (1.0 + np.exp(
                -tmp_np[..., 0:2].astype(np.float64))).astype(np.float32)

            # Add z offset and sigmoid
            tmp_np[..., 4:5] += ref_logits[..., 2:3]
            tmp_np[..., 4:5] = 1.0 / (1.0 + np.exp(
                -tmp_np[..., 4:5].astype(np.float64))).astype(np.float32)

            # Collect refined reference for next layer
            last_ref_points_np = np.concatenate(
                [tmp_np[..., 0:2], tmp_np[..., 4:5]], axis=-1)

            # ---- Scale by pc_range ----
            tmp_np[..., 0:1] = (
                tmp_np[..., 0:1] * self.real_w + self.pc_range[0])
            tmp_np[..., 1:2] = (
                tmp_np[..., 1:2] * self.real_h + self.pc_range[1])
            tmp_np[..., 4:5] = (
                tmp_np[..., 4:5] * (self.pc_range[5] - self.pc_range[2])
                + self.pc_range[2])

            outputs_classes.append(cls_np)
            outputs_coords.append(tmp_np)
            outputs_trajs.append(traj_np)

        # ---- Also run post-processing through graph ops (last layer) ----
        # This registers the full computation on the TTSim graph for the
        # final decoder layer, enabling hardware simulation.
        self._register_postprocess_graph(
            num_pred - 1, reg_out, ref_points, inter_references,
            num_pred, bs, nq)

        all_cls = np.stack(outputs_classes, axis=0)
        all_coords = np.stack(outputs_coords, axis=0)
        all_trajs = np.stack(outputs_trajs, axis=0)
        last_ref_logits = inverse_sigmoid_np(last_ref_points_np)

        return {
            'all_cls_scores': all_cls,
            'all_bbox_preds': all_coords,
            'all_past_traj_preds': all_trajs,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'last_ref_points': last_ref_logits,
            'query_feats': hs_np,
        }

    # ------------------------------------------------------------------
    # Graph-level post-processing (registers ops on TTSim graph)
    # ------------------------------------------------------------------

    def _register_postprocess_graph(self, lvl, reg_out_t, ref_points,
                                    inter_references, num_pred, bs, nq):
        """
        Register the post-processing slice/add/sigmoid/scale chain on the
        TTSim graph for decoder layer *lvl*.

        This mirrors the numpy post-processing in get_detections but uses
        graph ops so the computation can be traced for hardware simulation.
        """
        # Get reference as SimTensor
        if lvl == 0:
            rp_np = (ref_points.data if hasattr(ref_points, 'data')
                     else ref_points)
            if not isinstance(rp_np, np.ndarray):
                rp_np = np.array(rp_np, dtype=np.float32)
            if rp_np.ndim == 2:
                rp_np = np.tile(rp_np[np.newaxis], (bs, 1, 1))
            ref_t = F._from_data(f'{self.name}.ref_for_graph_L{lvl}',
                                 rp_np.astype(np.float32), is_const=True)
            setattr(self, ref_t.name, ref_t)
            ref_sig = getattr(self, f'sig_ref_L{lvl}')(ref_t)
            setattr(self, ref_sig.name, ref_sig)
        else:
            ref_l = inter_references[lvl - 1]
            if hasattr(ref_l, 'data'):
                ref_sig = ref_l
            else:
                ref_sig = F._from_data(
                    f'{self.name}.ref_for_graph_L{lvl}',
                    np.array(ref_l, dtype=np.float32), is_const=True)
                setattr(self, ref_sig.name, ref_sig)

        # inverse_sigmoid(reference)
        inv = self.inv_sig[lvl]
        ref_logits = inv(ref_sig)
        setattr(self, ref_logits.name, ref_logits)

        # Slice reg output and reference
        tmp_xy = self._slice(reg_out_t, f'L{lvl}_tmp_xy', 0, 2)
        tmp_mid = self._slice(reg_out_t, f'L{lvl}_tmp_mid', 2, 4)
        tmp_z = self._slice(reg_out_t, f'L{lvl}_tmp_z', 4, 5)
        tmp_tail = self._slice(reg_out_t, f'L{lvl}_tmp_tail', 5,
                               self.code_size)

        ref_xy = self._slice(ref_logits, f'L{lvl}_ref_xy', 0, 2)
        ref_z = self._slice(ref_logits, f'L{lvl}_ref_z', 2, 3)

        # Add reference offsets
        add_xy_op = getattr(self, f'add_xy_L{lvl}')
        xy_added = add_xy_op(tmp_xy, ref_xy)
        setattr(self, xy_added.name, xy_added)

        sig_xy_op = getattr(self, f'sig_xy_L{lvl}')
        xy_sig = sig_xy_op(xy_added)
        setattr(self, xy_sig.name, xy_sig)

        add_z_op = getattr(self, f'add_z_L{lvl}')
        z_added = add_z_op(tmp_z, ref_z)
        setattr(self, z_added.name, z_added)

        sig_z_op = getattr(self, f'sig_z_L{lvl}')
        z_sig = sig_z_op(z_added)
        setattr(self, z_sig.name, z_sig)

        # last_ref_points = cat([xy_sig, z_sig])
        concat_ref_op = getattr(self, f'concat_ref_L{lvl}')
        last_ref = concat_ref_op(xy_sig, z_sig)
        setattr(self, last_ref.name, last_ref)

        # Scale by pc_range
        # x: xy_sig[..., 0:1] * range_x + offset_x
        x_part = self._slice(xy_sig, f'L{lvl}_x_part', 0, 1)
        y_part = self._slice(xy_sig, f'L{lvl}_y_part', 1, 2)

        mul_x_op = getattr(self, f'mul_x_L{lvl}')
        x_scaled = mul_x_op(x_part, self.range_x_const)
        setattr(self, x_scaled.name, x_scaled)

        add_ox_op = getattr(self, f'add_ox_L{lvl}')
        x_final = add_ox_op(x_scaled, self.offset_x_const)
        setattr(self, x_final.name, x_final)

        mul_y_op = getattr(self, f'mul_y_L{lvl}')
        y_scaled = mul_y_op(y_part, self.range_y_const)
        setattr(self, y_scaled.name, y_scaled)

        add_oy_op = getattr(self, f'add_oy_L{lvl}')
        y_final = add_oy_op(y_scaled, self.offset_y_const)
        setattr(self, y_final.name, y_final)

        mul_z_op = getattr(self, f'mul_z_L{lvl}')
        z_scaled = mul_z_op(z_sig, self.range_z_const)
        setattr(self, z_scaled.name, z_scaled)

        add_oz_op = getattr(self, f'add_oz_L{lvl}')
        z_final = add_oz_op(z_scaled, self.offset_z_const)
        setattr(self, z_final.name, z_final)

        # Reconstruct full coordinate tensor
        concat_coord_op = getattr(self, f'concat_coord_L{lvl}')
        coord_out = concat_coord_op(
            x_final, y_final, tmp_mid, z_final, tmp_tail)
        setattr(self, coord_out.name, coord_out)

        # inverse_sigmoid on last_ref_points
        last_ref_logits = self.inv_sig_final(last_ref)
        setattr(self, last_ref_logits.name, last_ref_logits)

        return coord_out, last_ref_logits

    # ------------------------------------------------------------------
    # Analytical parameter count
    # ------------------------------------------------------------------

    def analytical_param_count(self):
        """Total learnable parameters (head only, excludes transformer)."""
        total = 0

        # bev_embedding: bev_h * bev_w * embed_dims
        total += self.bev_h * self.bev_w * self.embed_dims

        # query_embedding: num_query * (2 * embed_dims)
        total += self.num_query * 2 * self.embed_dims

        # Branches
        n = self.num_pred if self.with_box_refine else 1
        traj_out = (self.past_steps + self.fut_steps) * 2

        for _ in range(n):
            # cls: num_reg_fcs * (embed*embed + embed) + embed*cls_out + cls_out
            for _ in range(self.num_reg_fcs):
                total += self.embed_dims * self.embed_dims + self.embed_dims
            total += self.embed_dims * self.cls_out_channels + self.cls_out_channels

            # reg: num_reg_fcs * (embed*embed + embed) + embed*code + code
            for _ in range(self.num_reg_fcs):
                total += self.embed_dims * self.embed_dims + self.embed_dims
            total += self.embed_dims * self.code_size + self.code_size

            # dec_reg (decoder-only reg branch, same structure as reg)
            for _ in range(self.num_reg_fcs):
                total += self.embed_dims * self.embed_dims + self.embed_dims
            total += self.embed_dims * self.code_size + self.code_size

            # traj: same structure as reg but different output size
            for _ in range(self.num_reg_fcs):
                total += self.embed_dims * self.embed_dims + self.embed_dims
            total += self.embed_dims * traj_out + traj_out

        return total


# ======================================================================
# Quick self-test
# ======================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("BEVFormerTrackHead TTSim Module — Self-test")
    logger.info("=" * 70)

    embed_dims = 256
    num_reg_fcs = 2
    cls_out_channels = 10
    code_size = 10
    bev_h, bev_w = 30, 30
    past_steps, fut_steps = 4, 4
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    num_pred = 6
    num_query = 300

    def _init_branch_weights(branch):
        """Fill all Linear weights with random data so forward pass works."""
        for i in range(branch.num_reg_fcs + 1):
            fc = branch.fcs[i]
            fc.param.data = np.random.randn(
                fc.in_features, fc.out_features).astype(np.float32) * 0.02
            if fc.bias is not None:
                fc.bias.data = np.zeros(fc.out_features, dtype=np.float32)

    # ---- Test ClsBranch ----
    logger.info("\n--- ClsBranch ---")
    cb = ClsBranch('test_cls', embed_dims, num_reg_fcs, cls_out_channels)
    _init_branch_weights(cb)
    x_np = np.random.randn(1, num_query, embed_dims).astype(np.float32)
    x_t = F._from_data('cls_in', x_np, is_const=True)
    cls_out = cb(x_t)
    logger.debug(f"  input:  {x_np.shape}")
    logger.debug(f"  output: {cls_out.data.shape}")
    assert cls_out.data.shape == (1, num_query, cls_out_channels)
    logger.debug("  [OK]")

    # ---- Test RegBranch ----
    logger.info("\n--- RegBranch ---")
    rb = RegBranch('test_reg', embed_dims, num_reg_fcs, code_size)
    _init_branch_weights(rb)
    reg_out = rb(x_t)
    logger.debug(f"  input:  {x_np.shape}")
    logger.debug(f"  output: {reg_out.data.shape}")
    assert reg_out.data.shape == (1, num_query, code_size)
    logger.debug("  [OK]")

    # ---- Test TrajRegBranch ----
    logger.info("\n--- TrajRegBranch ---")
    traj_out_dim = (past_steps + fut_steps) * 2
    tb = RegBranch('test_traj', embed_dims, num_reg_fcs, traj_out_dim)
    _init_branch_weights(tb)
    traj_out = tb(x_t)
    logger.debug(f"  input:  {x_np.shape}")
    logger.debug(f"  output: {traj_out.data.shape}")
    assert traj_out.data.shape == (1, num_query, traj_out_dim)
    logger.debug("  [OK]")

    # ---- Test BEVFormerTrackHead (no transformer) ----
    logger.info("\n--- BEVFormerTrackHead (standalone) ---")
    head = BEVFormerTrackHead(
        name='test_head',
        embed_dims=embed_dims,
        num_reg_fcs=num_reg_fcs,
        cls_out_channels=cls_out_channels,
        code_size=code_size,
        bev_h=bev_h,
        bev_w=bev_w,
        past_steps=past_steps,
        fut_steps=fut_steps,
        pc_range=pc_range,
        num_pred=num_pred,
        num_query=num_query,
        with_box_refine=True,
        transformer=None,
    )
    total_params = head.analytical_param_count()
    logger.debug(f"  embed_dims={embed_dims}, num_pred={num_pred}")
    logger.debug(f"  Total params: {total_params:,}")
    logger.debug("  [OK]")

    logger.info("\n" + "=" * 70)
    logger.info("[OK] All BEVFormerTrackHead self-tests passed!")
    logger.info("=" * 70)
