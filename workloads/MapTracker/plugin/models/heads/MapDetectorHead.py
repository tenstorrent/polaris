#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
TTSim implementation of MapDetectorHead for MapTracker.

This module implements the detection head that processes transformer decoder outputs
to predict map elements (lane lines, boundaries, etc.) with classification scores.

Original: maptracker/plugin/models/heads/MapDetectorHead.py
"""

import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import copy
from typing import Any
import numpy as np
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F

# -------------------------------PyTorch--------------------------------

# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
# from mmcv.runner import force_fp32
# from mmcv.cnn.bricks.transformer import build_positional_encoding
# from mmdet.models.utils import build_transformer
# from mmdet.models import build_loss
#
# from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
# from mmdet.models import HEADS
# from mmdet.models.utils.transformer import inverse_sigmoid
#
# from einops import rearrange
#
# @HEADS.register_module(force=True)
# class MapDetectorHead(nn.Module):

#     def __init__(self,
#                  num_queries,
#                  num_classes=3,
#                  in_channels=128,
#                  embed_dims=256,
#                  score_thr=0.1,
#                  num_points=20,
#                  coord_dim=2,
#                  roi_size=(60, 30),
#                  different_heads=True,
#                  predict_refine=False,
#                  bev_pos=None,
#                  sync_cls_avg_factor=True,
#                  bg_cls_weight=0.,
#                  trans_loss_weight=0.0,
#                  transformer=dict(),
#                  loss_cls=dict(),
#                  loss_reg=dict(),
#                  assigner=dict()
#                 ):
#         super().__init__()
#         self.num_queries = num_queries
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         self.embed_dims = embed_dims
#         self.different_heads = different_heads
#         self.predict_refine = predict_refine
#         self.bev_pos = bev_pos
#         self.num_points = num_points
#         self.coord_dim = coord_dim

#         self.sync_cls_avg_factor = sync_cls_avg_factor
#         self.bg_cls_weight = bg_cls_weight

#         self.trans_loss_weight = trans_loss_weight
#         # NOTE: below is a simple MLP to transform the query from prev-frame to cur-frame,
#         # we moved the propagation part outside,

#         self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
#         origin = (-roi_size[0]/2, -roi_size[1]/2)
#         self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

#         sampler_cfg = dict(type='PseudoSampler')
#         self.sampler = build_sampler(sampler_cfg, context=self)

#         self.transformer = build_transformer(transformer)

#         self.loss_cls = build_loss(loss_cls)
#         self.loss_reg = build_loss(loss_reg)
#         self.assigner = build_assigner(assigner)

#         if self.loss_cls.use_sigmoid:
#             self.cls_out_channels = num_classes
#         else:
#             self.cls_out_channels = num_classes + 1

#         self._init_embedding()
#         self._init_branch()
#         self.init_weights()


#     def init_weights(self):
#         """Initialize weights of the DeformDETR head."""

#         for p in self.input_proj.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#         xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

#         self.transformer.init_weights()

#         # init prediction branch
#         for m in self.reg_branches:
#             for param in m.parameters():
#                 if param.dim() > 1:
#                     nn.init.xavier_uniform_(param)

#         # focal loss init
#         if self.loss_cls.use_sigmoid:
#             bias_init = bias_init_with_prob(0.01)
#             if isinstance(self.cls_branches, nn.ModuleList):
#                 for m in self.cls_branches:
#                     if hasattr(m, 'bias'):
#                         nn.init.constant_(m.bias, bias_init)
#             else:
#                 m = self.cls_branches
#                 nn.init.constant_(m.bias, bias_init)

#         if hasattr(self, 'query_alpha'):
#             for m in self.query_alpha:
#                 for param in m.parameters():
#                     if param.dim() > 1:
#                         nn.init.zeros_(param)

#     def _init_embedding(self):
#         positional_encoding = dict(
#             type='SinePositionalEncoding',
#             num_feats=self.embed_dims//2,
#             normalize=True
#         )
#         self.bev_pos_embed = build_positional_encoding(positional_encoding)

#         # query_pos_embed & query_embed
#         self.query_embedding = nn.Embedding(self.num_queries,
#                                             self.embed_dims)

#         self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2)

#     def _init_branch(self,):
#         """Initialize classification branch and regression branch of head."""
#         self.input_proj = Conv2d(
#             self.in_channels, self.embed_dims, kernel_size=1)

#         cls_branch = Linear(self.embed_dims, self.cls_out_channels)

#         reg_branch = [
#             Linear(self.embed_dims, 2*self.embed_dims),
#             nn.LayerNorm(2*self.embed_dims),
#             nn.ReLU(),
#             Linear(2*self.embed_dims, 2*self.embed_dims),
#             nn.LayerNorm(2*self.embed_dims),
#             nn.ReLU(),
#             Linear(2*self.embed_dims, self.num_points * self.coord_dim),
#         ]
#         reg_branch = nn.Sequential(*reg_branch)

#         num_layers = self.transformer.decoder.num_layers
#         if self.different_heads:
#             cls_branches = nn.ModuleList(
#                 [copy.deepcopy(cls_branch) for _ in range(num_layers)])
#             reg_branches = nn.ModuleList(
#                 [copy.deepcopy(reg_branch) for _ in range(num_layers)])
#         else:
#             cls_branches = nn.ModuleList(
#                 [cls_branch for _ in range(num_layers)])
#             reg_branches = nn.ModuleList(
#                 [reg_branch for _ in range(num_layers)])

#         self.reg_branches = reg_branches
#         self.cls_branches = cls_branches

#     def _prepare_context(self, bev_features):
#         """Prepare class label and vertex context."""
#         device = bev_features.device

#         # Add 2D coordinate grid embedding
#         B, C, H, W = bev_features.shape
#         bev_mask = bev_features.new_zeros(B, H, W)
#         bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
#         bev_features = self.input_proj(bev_features) + bev_pos_embeddings # (bs, embed_dims, H, W)

#         assert list(bev_features.shape) == [B, self.embed_dims, H, W]
#         return bev_features

#     def forward_train(self, bev_features, img_metas, gts, track_query_info=None, memory_bank=None, return_matching=False):
#         '''
#         Args:
#             bev_feature (List[Tensor]): shape [B, C, H, W]
#                 feature in bev view
#         Outs:
#             preds_dict (list[dict]):
#                 lines (Tensor): Classification score of all
#                     decoder layers, has shape
#                     [bs, num_query, 2*num_points]
#                 scores (Tensor):
#                     [bs, num_query,]
#         '''

#         bev_features = self._prepare_context(bev_features)

#         bs, C, H, W = bev_features.shape
#         img_masks = bev_features.new_zeros((bs, H, W))
#         # pos_embed = self.positional_encoding(img_masks)
#         pos_embed = None

#         query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
#         input_query_num = self.num_queries

#         init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
#         init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)

#         assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
#         assert list(query_embedding.shape) == [bs, self.num_queries, self.embed_dims]

#         # Prepare the propagated track queries, concat with the original dummy queries
#         if track_query_info is not None and 'track_query_hs_embeds' in track_query_info[0]:
#             new_query_embeds = []
#             new_init_ref_pts = []
#             for b_i in range(bs):
#                 new_queries = torch.cat([track_query_info[b_i]['track_query_hs_embeds'], query_embedding[b_i],
#                            track_query_info[b_i]['pad_hs_embeds']], dim=0)
#                 new_query_embeds.append(new_queries)
#                 init_ref = rearrange(init_reference_points[b_i], 'n k c -> n (k c)', c=2)
#                 new_ref = torch.cat([track_query_info[b_i]['trans_track_query_boxes'], init_ref,
#                            track_query_info[b_i]['pad_query_boxes']], dim=0)
#                 new_ref = rearrange(new_ref, 'n (k c) -> n k c', c=2)
#                 new_init_ref_pts.append(new_ref)
#                 #print('length of track queries', track_query_info[b_i]['track_query_hs_embeds'].shape[0])


#             # concat to get the track+dummy queries
#             query_embedding = torch.stack(new_query_embeds, dim=0)
#             init_reference_points = torch.stack(new_init_ref_pts, dim=0)
#             query_kp_mask = torch.stack([t['query_padding_mask'] for t in track_query_info], dim=0)
#         else:
#             query_kp_mask = query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool)

#         # outs_dec: (num_layers, num_qs, bs, embed_dims)
#         inter_queries, init_reference, inter_references = self.transformer(
#             mlvl_feats=[bev_features,],
#             mlvl_masks=[img_masks.type(torch.bool)],
#             query_embed=query_embedding,
#             mlvl_pos_embeds=[pos_embed], # not used
#             memory_query=None,
#             init_reference_points=init_reference_points,
#             reg_branches=self.reg_branches,
#             cls_branches=self.cls_branches,
#             predict_refine=self.predict_refine,
#             query_key_padding_mask=query_kp_mask, # mask used in self-attn,
#             memory_bank=memory_bank,
#         )

#         outputs = []
#         for i, (queries) in enumerate(inter_queries):
#             reg_points = inter_references[i] # (bs, num_q, num_points, 2)
#             bs = reg_points.shape[0]
#             reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)

#             scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

#             reg_points_list = []
#             scores_list = []
#             for i in range(len(scores)):
#                 # padding queries should not be output
#                 reg_points_list.append(reg_points[i])
#                 scores_list.append(scores[i])

#             pred_dict = {
#                 'lines': reg_points_list,
#                 'scores': scores_list
#             }
#             if return_matching:
#                 pred_dict['hs_embeds'] = queries
#             outputs.append(pred_dict)

#         # Pass in the track query information to massage the cost matrix
#         loss_dict, det_match_idxs, det_match_gt_idxs, gt_info_list, matched_reg_cost = \
#                 self.loss(gts=gts, preds=outputs, track_info=track_query_info)

#         if return_matching:
#             return loss_dict, outputs[-1], det_match_idxs[-1], det_match_gt_idxs[-1], matched_reg_cost[-1], gt_info_list[-1]
#         else:
#             return outputs, loss_dict, det_match_idxs, det_match_gt_idxs, gt_info_list

#     def forward_test(self, bev_features, img_metas, track_query_info=None, memory_bank=None):
#         '''
#         Args:
#             bev_feature (List[Tensor]): shape [B, C, H, W]
#                 feature in bev view
#         Outs:
#             preds_dict (list[dict]):
#                 lines (Tensor): Classification score of all
#                     decoder layers, has shape
#                     [bs, num_query, 2*num_points]
#                 scores (Tensor):
#                     [bs, num_query,]
#         '''

#         bev_features = self._prepare_context(bev_features)

#         bs, C, H, W = bev_features.shape
#         assert bs == 1, 'Only support bs=1 per-gpu for inference'

#         img_masks = bev_features.new_zeros((bs, H, W))
#         # pos_embed = self.positional_encoding(img_masks)
#         pos_embed = None

#         query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
#         input_query_num = self.num_queries
#         # num query: self.num_query + self.topk

#         init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
#         init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)

#         assert list(init_reference_points.shape) == [bs, input_query_num, self.num_points, 2]
#         assert list(query_embedding.shape) == [bs, input_query_num, self.embed_dims]

#         # Prepare the propagated track queries, concat with the original dummy queries
#         if track_query_info is not None and 'track_query_hs_embeds' in track_query_info[0]:
#             prev_hs_embed = torch.stack([t['track_query_hs_embeds'] for t in track_query_info])
#             prev_boxes = torch.stack([t['trans_track_query_boxes'] for t in track_query_info])
#             prev_boxes = rearrange(prev_boxes, 'b n (k c) -> b n k c', c=2)

#             # concat to get the track+dummy queries
#             query_embedding = torch.cat([prev_hs_embed, query_embedding], dim=1)
#             init_reference_points = torch.cat([prev_boxes, init_reference_points], dim=1)

#         query_kp_mask = query_embedding.new_zeros((bs, query_embedding.shape[1]), dtype=torch.bool)

#         # outs_dec: (num_layers, num_qs, bs, embed_dims)
#         inter_queries, init_reference, inter_references = self.transformer(
#             mlvl_feats=[bev_features,],
#             mlvl_masks=[img_masks.type(torch.bool)],
#             query_embed=query_embedding,
#             mlvl_pos_embeds=[pos_embed], # not used
#             memory_query=None,
#             init_reference_points=init_reference_points,
#             reg_branches=self.reg_branches,
#             cls_branches=self.cls_branches,
#             predict_refine=self.predict_refine,
#             query_key_padding_mask=query_kp_mask, # mask used in self-attn,
#             memory_bank=memory_bank,
#         )

#         outputs = []
#         for i_query, (queries) in enumerate(inter_queries):
#             reg_points = inter_references[i_query] # (bs, num_q, num_points, 2)
#             bs = reg_points.shape[0]
#             reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
#             scores = self.cls_branches[i_query](queries) # (bs, num_q, num_classes)

#             reg_points_list = []
#             scores_list = []
#             for i in range(len(scores)):
#                 # padding queries should not be output
#                 reg_points_list.append(reg_points[i])
#                 scores_list.append(scores[i])

#             pred_dict = {
#                 'lines': reg_points_list,
#                 'scores': scores_list,
#                 'hs_embeds': queries,
#             }
#             outputs.append(pred_dict)

#         return outputs

#     @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
#     def _get_target_single(self,
#                            score_pred,
#                            lines_pred,
#                            gt_labels,
#                            gt_lines,
#                            track_info=None,
#                            gt_bboxes_ignore=None):
#         """
#             Compute regression and classification targets for one image.
#             Outputs from a single decoder layer of a single feature level are used.
#             Args:
#                 score_pred (Tensor): Box score logits from a single decoder layer
#                     for one image. Shape [num_query, cls_out_channels].
#                 lines_pred (Tensor):
#                     shape [num_query, 2*num_points]
#                 gt_labels (torch.LongTensor)
#                     shape [num_gt, ]
#                 gt_lines (Tensor):
#                     shape [num_gt, 2*num_points].

#             Returns:
#                 tuple[Tensor]: a tuple containing the following for one sample.
#                     - labels (LongTensor): Labels of each image.
#                         shape [num_query, 1]
#                     - label_weights (Tensor]): Label weights of each image.
#                         shape [num_query, 1]
#                     - lines_target (Tensor): Lines targets of each image.
#                         shape [num_query, num_points, 2]
#                     - lines_weights (Tensor): Lines weights of each image.
#                         shape [num_query, num_points, 2]
#                     - pos_inds (Tensor): Sampled positive indices for each image.
#                     - neg_inds (Tensor): Sampled negative indices for each image.
#         """
#         num_pred_lines = len(lines_pred)
#         # assigner and sampler

#         # We massage the matching cost here using the track info, following
#         # the 3-type supervision of TrackFormer/MOTR
#         assign_result, gt_permute_idx, matched_reg_cost = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
#                                              gts=dict(lines=gt_lines,
#                                                       labels=gt_labels, ),
#                                              track_info=track_info,
#                                              gt_bboxes_ignore=gt_bboxes_ignore)
#         sampling_result = self.sampler.sample(
#             assign_result, lines_pred, gt_lines)
#         num_gt = len(gt_lines)
#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         pos_gt_inds = sampling_result.pos_assigned_gt_inds

#         labels = gt_lines.new_full(
#                 (num_pred_lines, ), self.num_classes, dtype=torch.long) # (num_q, )
#         labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
#         label_weights = gt_lines.new_ones(num_pred_lines) # (num_q, )

#         lines_target = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
#         lines_weights = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)

#         if num_gt > 0:
#             if gt_permute_idx is not None: # using permute invariant label
#                 # gt_permute_idx: (num_q, num_gt)
#                 # pos_inds: which query is positive
#                 # pos_gt_inds: which gt each pos pred is assigned
#                 # single_matched_gt_permute_idx: which permute order is matched
#                 single_matched_gt_permute_idx = gt_permute_idx[
#                     pos_inds, pos_gt_inds
#                 ]
#                 lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
#                     lines_target.dtype) # (num_q, 2*num_pts)
#             else:
#                 lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
#                     lines_target.dtype) # (num_q, 2*num_pts)

#         lines_weights[pos_inds] = 1.0 # (num_q, 2*num_pts)

#         # normalization
#         # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
#         # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
#         # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

#         return (labels, label_weights, lines_target, lines_weights,
#                 pos_inds, neg_inds, pos_gt_inds, matched_reg_cost)

#     # @force_fp32(apply_to=('preds', 'gts'))
#     def get_targets(self, preds, gts, track_info=None, gt_bboxes_ignore_list=None):
#         """
#             Compute regression and classification targets for a batch image.
#             Outputs from a single decoder layer of a single feature level are used.
#             Args:
#                 preds (dict):
#                     - lines (Tensor): shape (bs, num_queries, 2*num_points)
#                     - scores (Tensor): shape (bs, num_queries, num_class_channels)
#                 gts (dict):
#                     - class_label (list[Tensor]): tensor shape (num_gts, )
#                     - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
#                 gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                     boxes which can be ignored for each image. Default None.
#             Returns:
#                 tuple: a tuple containing the following targets.
#                     - labels_list (list[Tensor]): Labels for all images.
#                     - label_weights_list (list[Tensor]): Label weights for all \
#                         images.
#                     - lines_targets_list (list[Tensor]): Lines targets for all \
#                         images.
#                     - lines_weight_list (list[Tensor]): Lines weights for all \
#                         images.
#                     - num_total_pos (int): Number of positive samples in all \
#                         images.
#                     - num_total_neg (int): Number of negative samples in all \
#                         images.
#         """
#         assert gt_bboxes_ignore_list is None, \
#             'Only supports for gt_bboxes_ignore setting to None.'

#         # format the inputs
#         gt_labels = gts['labels']
#         gt_lines = gts['lines']

#         lines_pred = preds['lines']

#         if track_info is None:
#             track_info = [track_info for _ in range(len(gt_labels))]

#         (labels_list, label_weights_list,
#         lines_targets_list, lines_weights_list,
#         pos_inds_list, neg_inds_list,pos_gt_inds_list, matched_reg_cost) = multi_apply(
#             self._get_target_single, preds['scores'], lines_pred,
#             gt_labels, gt_lines, track_info, gt_bboxes_ignore=gt_bboxes_ignore_list)

#         num_total_pos = sum((inds.numel() for inds in pos_inds_list))
#         num_total_neg = sum((inds.numel() for inds in neg_inds_list))

#         if track_info[0] is not None:
#             # remove the padding elements from the neg counting
#             padding_mask = torch.cat([t['query_padding_mask'] for t in track_info], dim=0)
#             num_padding = padding_mask.sum()
#             num_total_neg -= num_padding

#         new_gts = dict(
#             labels=labels_list, # list[Tensor(num_q, )], length=bs
#             label_weights=label_weights_list, # list[Tensor(num_q, )], length=bs, all ones
#             lines=lines_targets_list, # list[Tensor(num_q, 2*num_pts)], length=bs
#             lines_weights=lines_weights_list, # list[Tensor(num_q, 2*num_pts)], length=bs
#         )

#         return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, matched_reg_cost

#     # @force_fp32(apply_to=('preds', 'gts'))
#     def loss_single(self,
#                     preds,
#                     gts,
#                     track_info=None,
#                     gt_bboxes_ignore_list=None,
#                     reduction='none'):
#         """
#             Loss function for outputs from a single decoder layer of a single
#             feature level.
#             Args:
#                 preds (dict):
#                     - lines (Tensor): shape (bs, num_queries, 2*num_points)
#                     - scores (Tensor): shape (bs, num_queries, num_class_channels)
#                 gts (dict):
#                     - class_label (list[Tensor]): tensor shape (num_gts, )
#                     - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
#                 gt_bboxes_ignore_list (list[Tensor], optional): Bounding
#                     boxes which can be ignored for each image. Default None.
#             Returns:
#                 dict[str, Tensor]: A dictionary of loss components for outputs from
#                     a single decoder layer.
#         """

#         # Get target for each sample
#         new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, matched_reg_cost =\
#             self.get_targets(preds, gts, track_info, gt_bboxes_ignore_list)

#         # Batched all data
#         # for k, v in new_gts.items():
#         #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

#         # construct weighted avg_factor to match with the official DETR repo
#         cls_avg_factor = num_total_pos * 1.0 + \
#             num_total_neg * self.bg_cls_weight

#         if self.sync_cls_avg_factor:
#             cls_avg_factor = reduce_mean(
#                 preds['scores'][0].new_tensor([cls_avg_factor]))
#         cls_avg_factor = max(cls_avg_factor, 1)

#         if track_info is not None:
#             cat_padding_mask = torch.cat([t['query_padding_mask'] for t in track_info], dim=0)
#             padding_loss_mask = ~cat_padding_mask

#         # Classification loss
#         # since the inputs needs the second dim is the class dim, we permute the prediction.
#         pred_scores = torch.cat(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
#         cls_scores = pred_scores.reshape(-1, self.cls_out_channels) # (bs*num_q, cls_out_channels)
#         cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1) # (bs*num_q, )
#         cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1) # (bs*num_q, )
#         if track_info is not None:
#             cls_weights = cls_weights * padding_loss_mask.float()

#         loss_cls = self.loss_cls(
#             cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)

#         # Compute the average number of gt boxes across all gpus, for
#         # normalization purposes
#         num_total_pos = loss_cls.new_tensor([num_total_pos])
#         num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

#         pred_lines = torch.cat(preds['lines'], dim=0)
#         gt_lines = torch.cat(new_gts['lines'], dim=0)
#         line_weights = torch.cat(new_gts['lines_weights'], dim=0)
#         if track_info is not None:
#             line_weights = line_weights * padding_loss_mask[:, None].float()

#         assert len(pred_lines) == len(gt_lines)
#         assert len(gt_lines) == len(line_weights)

#         loss_reg = self.loss_reg(
#             pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

#         loss_dict = dict(
#             cls=loss_cls,
#             reg=loss_reg,
#         )

#         new_gts_info = {
#             'labels': new_gts['labels'],
#             'lines': new_gts['lines'],
#         }

#         return loss_dict, pos_inds_list, pos_gt_inds_list, matched_reg_cost, new_gts_info

#     @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
#     def loss(self,
#              gts,
#              preds,
#              gt_bboxes_ignore=None,
#              track_info=None,
#              reduction='mean',
#             ):
#         """
#             Loss Function.
#             Args:
#                 gts (list[dict]): list length: num_layers
#                     dict {
#                         'label': list[tensor(num_gts, )], list length: batchsize,
#                         'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
#                         ...
#                     }
#                 preds (list[dict]): list length: num_layers
#                     dict {
#                         'lines': tensor(bs, num_queries, 2*num_points),
#                         'scores': tensor(bs, num_queries, class_out_channels),
#                     }

#                 gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
#                     which can be ignored for each image. Default None.
#             Returns:
#                 dict[str, Tensor]: A dictionary of loss components.
#         """
#         assert gt_bboxes_ignore is None, \
#             f'{self.__class__.__name__} only supports ' \
#             f'for gt_bboxes_ignore setting to None.'

#         track_info = [track_info for _ in range(len(gts))]
#         # Since there might have multi layer
#         losses, pos_inds_lists, pos_gt_inds_lists, matched_reg_costs, gt_info_list = multi_apply(
#             self.loss_single, preds, gts, track_info, reduction=reduction)

#         # Format the losses
#         loss_dict = dict()
#         # loss from the last decoder layer
#         for k, v in losses[-1].items():
#             loss_dict[k] = v

#         # Loss from other decoder layers
#         num_dec_layer = 0
#         for loss in losses[:-1]:
#             for k, v in loss.items():
#                 loss_dict[f'd{num_dec_layer}.{k}'] = v
#             num_dec_layer += 1

#         return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_info_list, matched_reg_costs

#     def post_process(self, preds_dict, tokens, track_dict=None, thr=0.0):
#         lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
#         bs = len(lines)
#         scores = preds_dict['scores'] # (bs, num_queries, 3)

#         results = []
#         for i in range(bs):
#             tmp_vectors = lines[i]
#             # set up the prop_flags
#             tmp_prop_flags = torch.zeros(tmp_vectors.shape[0]).bool()
#             tmp_prop_flags[-100:] = 0
#             tmp_prop_flags[:-100] = 1
#             num_preds, num_points2 = tmp_vectors.shape
#             tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)

#             if self.loss_cls.use_sigmoid:
#                 tmp_scores, tmp_labels = scores[i].max(-1)
#                 tmp_scores = tmp_scores.sigmoid()
#                 pos = tmp_scores > thr
#             else:
#                 assert self.num_classes + 1 == self.cls_out_channels
#                 tmp_scores, tmp_labels = scores[i].max(-1)
#                 bg_cls = self.cls_out_channels
#                 pos = tmp_labels != bg_cls

#             tmp_vectors = tmp_vectors[pos]
#             tmp_scores = tmp_scores[pos]
#             tmp_labels = tmp_labels[pos]
#             tmp_prop_flags = tmp_prop_flags[pos]

#             if len(tmp_scores) == 0:
#                 single_result = {
#                 'vectors': [],
#                 'scores': [],
#                 'labels': [],
#                 'props': [],
#                 'token': tokens[i]
#             }
#             else:
#                 single_result = {
#                     'vectors': tmp_vectors.detach().cpu().numpy(),
#                     'scores': tmp_scores.detach().cpu().numpy(),
#                     'labels': tmp_labels.detach().cpu().numpy(),
#                     'props': tmp_prop_flags.detach().cpu().numpy(),
#                     'token': tokens[i]
#                 }

#             # also save the tracking information for analyzing
#             if track_dict is not None and len(track_dict['lines'])>0:
#                 tmp_track_scores = track_dict['scores'][i]
#                 tmp_track_vectors = track_dict['lines'][i]
#                 tmp_track_scores, tmp_track_labels = tmp_track_scores.max(-1)
#                 tmp_track_scores = tmp_track_scores.sigmoid()
#                 single_result['track_scores'] = tmp_track_scores.detach().cpu().numpy()
#                 single_result['track_vectors'] = tmp_track_vectors.detach().cpu().numpy()
#                 single_result['track_labels'] = tmp_track_labels.detach().cpu().numpy()
#             else:
#                 single_result['track_scores'] = []
#                 single_result['track_vectors'] = []
#                 single_result['track_labels'] = []

#             results.append(single_result)

#         return results

#     def prepare_temporal_propagation(self, preds_dict, scene_name, local_idx, memory_bank=None,
#                         thr_track=0.1, thr_det=0.5):
#         lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
#         queries = preds_dict['hs_embeds']
#         bs = len(lines)
#         assert bs == 1, 'now only support bs=1 for temporal-evolving inference'
#         scores = preds_dict['scores'] # (bs, num_queries, 3)

#         first_frame = local_idx == 0

#         tmp_vectors = lines[0]
#         tmp_queries = queries[0]

#         # focal loss
#         if self.loss_cls.use_sigmoid:
#             tmp_scores, tmp_labels = scores[0].max(-1)
#             tmp_scores = tmp_scores.sigmoid()
#             pos_track = tmp_scores[:-100] > thr_track
#             pos_det = tmp_scores[-100:] > thr_det
#             pos = torch.cat([pos_track, pos_det], dim=0)
#         else:
#             raise RuntimeError('The experiment uses sigmoid for cls outputs')

#         pos_vectors = tmp_vectors[pos]
#         pos_labels = tmp_labels[pos]
#         pos_queries = tmp_queries[pos]
#         pos_scores = tmp_scores[pos]

#         if first_frame:
#             global_ids = torch.arange(len(pos_vectors))
#             num_instance = len(pos_vectors)
#         else:
#             prop_ids = self.prop_info['global_ids']
#             prop_num_instance = self.prop_info['num_instance']
#             global_ids_track = prop_ids[pos_track]
#             num_newborn = int(pos_det.sum())
#             global_ids_newborn = torch.arange(num_newborn) + prop_num_instance
#             global_ids = torch.cat([global_ids_track, global_ids_newborn])
#             num_instance = prop_num_instance + num_newborn

#         self.prop_info = {
#             'vectors': pos_vectors,
#             'queries': pos_queries,
#             'scores': pos_scores,
#             'labels': pos_labels,
#             'scene_name': scene_name,
#             'local_idx': local_idx,
#             'global_ids': global_ids,
#             'num_instance': num_instance,
#         }

#         if memory_bank is not None:
#             if first_frame:
#                 num_tracks = 0
#             else:
#                 num_tracks = self.prop_active_tracks
#             pos_out_inds = torch.where(pos)[0]
#             prev_out = {
#                 'hs_embeds': queries,
#                 'scores': scores,
#             }
#             memory_bank.update_memory(0, first_frame, pos_out_inds, prev_out, num_tracks, local_idx, memory_bank.curr_t)
#             self.prop_active_tracks = len(pos_out_inds)

#         save_pos_results = {
#             'vectors': pos_vectors.cpu().numpy(),
#             'scores': pos_scores.cpu().numpy(),
#             'labels': pos_labels.cpu().numpy(),
#             'global_ids': global_ids.cpu().numpy(),
#             'scene_name': scene_name,
#             'local_idx': local_idx,
#             'num_instance': num_instance,
#         }

#         return save_pos_results

#     def get_track_info(self, scene_name, local_idx):
#         prop_info = self.prop_info
#         assert prop_info['scene_name'] == scene_name and (prop_info['local_idx']+1 == local_idx or \
#             prop_info['local_idx'] == local_idx)

#         vectors = prop_info['vectors']
#         queries = prop_info['queries']
#         device = queries.device

#         target = {}
#         target['track_query_hs_embeds'] = queries
#         target['track_query_boxes'] = vectors
#         track_info = [target, ]

#         return track_info

#     def get_self_iter_track_query(self, preds_dict):
#         num_tracks = self.prop_active_tracks

#         lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
#         queries = preds_dict['hs_embeds']
#         bs = len(lines)
#         assert bs == 1, 'now only support bs=1 for temporal-evolving inference'
#         scores = preds_dict['scores'] # (bs, num_queries, 3)

#         queries = queries[0][:num_tracks]
#         vectors = lines[0][:num_tracks]

#         target = {}
#         target['track_query_hs_embeds'] = queries
#         target['track_query_boxes'] = vectors
#         track_info = [target, ]
#         return track_info


#     def clear_temporal_cache(self):
#         self.prop_info = None

#     def train(self, *args, **kwargs):
#         super().train(*args, **kwargs)

#     def eval(self):
#         super().eval()

#     def forward(self, *args, return_loss=True, **kwargs):
#         if return_loss:
#             return self.forward_train(*args, **kwargs)
#         else:
#             return self.forward_test(*args, **kwargs)

# -------------------------------TTSIM-----------------------------------


class RegressionBranch(SimNN.Module):
    """
    TTSim implementation of regression branch for coordinate prediction.

    Architecture: Linear(embed_dims -> 2*embed_dims) -> LayerNorm -> ReLU
                  -> Linear(2*embed_dims -> 2*embed_dims) -> LayerNorm -> ReLU
                  -> Linear(2*embed_dims -> num_points*coord_dim)

    Args:
        embed_dims: Input embedding dimension (default: 256)
        num_points: Number of points per line (default: 20)
        coord_dim: Coordinate dimensions (default: 2 for 2D)
    """

    def __init__(
        self, embed_dims=256, num_points=20, coord_dim=2, name="regression_branch"
    ):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.hidden_dims = 2 * embed_dims
        self.out_dims = num_points * coord_dim

        # Create Linear layers
        self.fc1 = SimNN.Linear(f"{self.name}_fc1", self.embed_dims, self.hidden_dims)
        self.fc2 = SimNN.Linear(f"{self.name}_fc2", self.hidden_dims, self.hidden_dims)
        self.fc3 = SimNN.Linear(f"{self.name}_fc3", self.hidden_dims, self.out_dims)

        # Create LayerNorm layers (non-affine: normalize only, no weight/bias)
        # PyTorch LN affine params must be set to weight=1, bias=0 on the
        # reference side so both sides match.
        from workloads.MapTracker.plugin.models.backbones.bevformer.builder_utils import (
            LayerNorm,
        )

        self.norm1 = LayerNorm(f"{self.name}_norm1", normalized_shape=self.hidden_dims)
        self.norm2 = LayerNorm(f"{self.name}_norm2", normalized_shape=self.hidden_dims)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [bs, num_queries, embed_dims]

        Returns:
            Output tensor [bs, num_queries, num_points*coord_dim]
        """
        # Track input
        setattr(self, x.name, x)

        # Layer 1: Linear
        x = self.fc1(x)
        setattr(self, x.name, x)

        # LayerNorm 1 (non-affine)
        x = self.norm1(x)
        setattr(self, x.name, x)

        # ReLU 1
        _relu1_op = F.Relu(f"{self.name}_relu1")
        setattr(self, _relu1_op.name, _relu1_op)
        x = _relu1_op(x)
        setattr(self, x.name, x)

        # Layer 2: Linear
        x = self.fc2(x)
        setattr(self, x.name, x)

        # LayerNorm 2 (non-affine)
        x = self.norm2(x)
        setattr(self, x.name, x)

        # ReLU 2
        _relu2_op = F.Relu(f"{self.name}_relu2")
        setattr(self, _relu2_op.name, _relu2_op)
        x = _relu2_op(x)
        setattr(self, x.name, x)

        # Layer 3: Linear (output)
        x = self.fc3(x)
        setattr(self, x.name, x)

        return x

    def analytical_param_count(self, lvl=0):
        """Calculate parameter count for this branch."""
        indent = "  " * lvl

        # FC1: (embed_dims * hidden_dims) + hidden_dims
        fc1_params = self.embed_dims * self.hidden_dims + self.hidden_dims

        # FC2: (hidden_dims * hidden_dims) + hidden_dims
        fc2_params = self.hidden_dims * self.hidden_dims + self.hidden_dims

        # FC3: (hidden_dims * out_dims) + out_dims
        fc3_params = self.hidden_dims * self.out_dims + self.out_dims

        # LayerNorm is non-affine (no learnable params)
        total = fc1_params + fc2_params + fc3_params

        if lvl >= 2:
            print(f"{indent}RegressionBranch '{self.name}':")
            print(f"{indent}  FC1: {fc1_params:,}")
            print(f"{indent}  FC2: {fc2_params:,}")
            print(f"{indent}  FC3: {fc3_params:,}")

        if lvl >= 1:
            print(f"{indent}Total RegressionBranch params: {total:,}")

        return total


class ClassificationBranch(SimNN.Module):
    """
    TTSim implementation of classification branch.

    Architecture: Linear(embed_dims -> num_classes)

    Args:
        embed_dims: Input embedding dimension (default: 256)
        num_classes: Number of output classes (default: 3)
    """

    def __init__(self, embed_dims=256, num_classes=3, name="classification_branch"):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_classes = num_classes

        # Create Linear layer
        self.fc = SimNN.Linear(f"{self.name}_fc", self.embed_dims, self.num_classes)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [bs, num_queries, embed_dims]

        Returns:
            Output tensor [bs, num_queries, num_classes]
        """
        # Track input
        setattr(self, x.name, x)

        # Apply linear layer
        x = self.fc(x)
        setattr(self, x.name, x)

        return x

    def analytical_param_count(self, lvl=0):
        """Calculate parameter count for this branch."""
        indent = "  " * lvl
        total = self.embed_dims * self.num_classes + self.num_classes

        if lvl >= 1:
            print(f"{indent}ClassificationBranch params: {total:,}")

        return total


class MapDetectorHead(SimNN.Module):
    """
    TTSim implementation of MapDetectorHead.

    Detection head for MapTracker that processes BEV features through a transformer
    decoder to predict vectorized map elements with classification scores.

    Args:
        num_queries: Number of object queries (default: 100)
        num_classes: Number of map element classes (default: 3)
        in_channels: Input BEV feature channels (default: 128)
        embed_dims: Embedding dimensions (default: 256)
        num_points: Number of points per map element (default: 20)
        coord_dim: Coordinate dimensions (default: 2)
        num_layers: Number of decoder layers (default: 6)
        different_heads: Whether to use different heads per layer (default: True)
        predict_refine: Whether to use iterative refinement (default: False)
        transformer: Transformer module (optional)
    """

    def __init__(
        self,
        num_queries=100,
        num_classes=3,
        in_channels=128,
        embed_dims=256,
        num_points=20,
        coord_dim=2,
        num_layers=6,
        different_heads=True,
        predict_refine=False,
        transformer=None,
    ):
        super().__init__()
        self.name = "map_detector_head"
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_points = num_points
        self.coord_dim = coord_dim
        self.num_layers = num_layers
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.prop_info: dict[str, Any] | None = None

        # Transformer decoder
        self.transformer = transformer

        # Input projection: Conv2d(in_channels -> embed_dims, kernel=1)
        self.input_proj = F.Conv2d(
            f"{self.name}_input_proj",
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        # Conv2d bias (reshaped for broadcasting: [embed_dims] -> [1, embed_dims, 1, 1])
        self.input_proj_bias_param = F._from_shape(
            f"{self.name}_input_proj_bias_param", [embed_dims], is_param=True
        )

        # Query embedding: Embedding(num_queries, embed_dims)
        # Initialize with random weights; can be overridden via set_query_embedding_weights()
        self.query_embedding_weight = (
            np.random.randn(num_queries, embed_dims).astype(np.float32) * 0.01
        )

        # Reference points embedding: Linear(embed_dims -> num_points*2)
        self.ref_points_embed = SimNN.Linear(
            f"{self.name}_ref_points", self.embed_dims, self.num_points * 2
        )

        # BEV positional embedding (SinePositionalEncoding)
        # This will be computed dynamically based on BEV feature size

        # Classification and regression branches (one per decoder layer)
        cls_list = []
        reg_list = []

        # Initialize branches
        for i in range(num_layers):
            cls_branch = ClassificationBranch(
                embed_dims, num_classes, name=f"cls_branch_l{i}"
            )
            cls_list.append(cls_branch)

            reg_branch = RegressionBranch(
                embed_dims, num_points, coord_dim, name=f"reg_branch_l{i}"
            )
            reg_list.append(reg_branch)

        self.cls_branches = SimNN.ModuleList(cls_list)
        self.reg_branches = SimNN.ModuleList(reg_list)

        super().link_op2module()

    def set_input_proj_weights(self, weight, bias):
        """Set input projection weights."""
        # Conv2d weight: shape should be [out_channels, in_channels, kh, kw]
        # TTSim Conv2d operator stores weight as params[0][1]
        self.input_proj.params[0][1].data = weight
        # Bias parameter
        self.input_proj_bias_param.data = bias

    def set_query_embedding_weights(self, weight):
        """Set query embedding weights."""
        self.query_embedding_weight = weight

    def set_ref_points_weights(self, weight, bias):
        """Set reference points embedding weights."""
        # Linear stores param as [out_features, in_features], same as PyTorch
        self.ref_points_embed.param.data = weight
        assert self.ref_points_embed.bias is not None
        self.ref_points_embed.bias.data = bias

    def init_weights(self):
        """
        Default initialization for Parameters of Module.

        Note: In TTSim inference, weights are loaded from pre-trained PyTorch model.
        This placeholder method documents the initialization but doesn't execute it.
        For testing, use the initialization utilities in the test file.
        """
        # Initialization handled by PyTorch model loading
        pass

    def _prepare_context(self, bev_features):
        """
        Prepare BEV context with positional encoding.

        Args:
            bev_features: BEV features [bs, in_channels, H, W]

        Returns:
            Processed features [bs, embed_dims, H, W]
        """
        bs, c, h, w = bev_features.shape

        # Input projection: Conv2d 1x1
        bev_features = self.input_proj(bev_features)
        setattr(self, bev_features.name, bev_features)

        # Add bias - reshape from [embed_dims] to [1, embed_dims, 1, 1] for broadcasting
        bias_shape = F._from_data(
            f"{self.name}_bias_shape",
            np.array([1, self.embed_dims, 1, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, bias_shape.name, bias_shape)
        reshape_op = F.Reshape(f"{self.name}_bias_reshape")
        proj_bias_reshaped = reshape_op(self.input_proj_bias_param, bias_shape)
        setattr(self, reshape_op.name, reshape_op)
        setattr(self, proj_bias_reshaped.name, proj_bias_reshaped)

        add_bias_op = F.Add(f"{self.name}_add_bias")
        bev_features = add_bias_op(bev_features, proj_bias_reshaped)
        setattr(self, add_bias_op.name, add_bias_op)
        setattr(self, bev_features.name, bev_features)

        # BEV positional embedding (SinePositionalEncoding)
        # For now, we'll add a placeholder - in practice, this would use sine/cosine embeddings
        # based on spatial coordinates
        bev_pos_embed = self._get_bev_pos_embed(bs, h, w)
        setattr(self, bev_pos_embed.name, bev_pos_embed)

        # Add positional embeddings
        add_pos_op = F.Add(f"{self.name}_add_pos")
        bev_features = add_pos_op(bev_features, bev_pos_embed)
        setattr(self, add_pos_op.name, add_pos_op)
        setattr(self, bev_features.name, bev_features)

        return bev_features

    def _get_bev_pos_embed(self, bs, h, w):
        """
        Generate BEV positional embeddings using sine/cosine encoding.

        Args:
            bs: Batch size
            h: Height
            w: Width

        Returns:
            Positional embeddings [bs, embed_dims, h, w]
        """
        # SinePositionalEncoding matching mmcv (normalize=True, scale=2*pi, temperature=10000)
        # Uses cumsum to generate coordinate grids, then normalizes to [0, 2*pi]
        scale = 2.0 * np.pi
        temperature = 10000.0
        eps = 1e-6
        num_feats = self.embed_dims // 2

        not_mask = np.ones((bs, h, w), dtype=np.float32)
        y_embed = np.cumsum(not_mask, axis=1)  # [bs, h, w]: 1, 2, ..., H along rows
        x_embed = np.cumsum(not_mask, axis=2)  # [bs, h, w]: 1, 2, ..., W along cols

        # Normalize by last element and scale to [0, 2*pi]
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = np.arange(num_feats, dtype=np.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [bs, h, w, num_feats]
        pos_y = y_embed[:, :, :, None] / dim_t  # [bs, h, w, num_feats]

        # Interleave sin/cos: even indices get sin, odd indices get cos
        pos_x_enc = np.zeros_like(pos_x)
        pos_x_enc[:, :, :, 0::2] = np.sin(pos_x[:, :, :, 0::2])
        pos_x_enc[:, :, :, 1::2] = np.cos(pos_x[:, :, :, 1::2])

        pos_y_enc = np.zeros_like(pos_y)
        pos_y_enc[:, :, :, 0::2] = np.sin(pos_y[:, :, :, 0::2])
        pos_y_enc[:, :, :, 1::2] = np.cos(pos_y[:, :, :, 1::2])

        # Concatenate y and x, then permute to [bs, embed_dims, h, w]
        pos = np.concatenate([pos_y_enc, pos_x_enc], axis=3)  # [bs, h, w, embed_dims]
        pos = pos.transpose(0, 3, 1, 2).astype(np.float32)  # [bs, embed_dims, h, w]

        # Convert to SimTensor
        pos_embed = F._from_data(
            f"{self.name}_bev_pos_embed", pos.astype(np.float32), is_const=True
        )

        return pos_embed

    def __call__(
        self, bev_features, img_metas=None, memory_bank=None, track_query_info=None
    ):
        """
        Forward pass through the detection head.

        Args:
            bev_features: BEV features [bs, in_channels, H, W]
            img_metas: Image metadata list (passed through for API compatibility)
            memory_bank: Optional memory bank for temporal tracking
            track_query_info: Optional list of dicts (one per batch) with tracked
                query info from previous frame. Each dict must contain:
                - 'track_query_hs_embeds': numpy [num_tracks, embed_dims]
                - 'trans_track_query_boxes': numpy [num_tracks, num_pts*2]
                When provided, tracked queries are prepended to detection queries
                before the transformer decoder, enabling temporal tracking.

        Returns:
            outputs: List of prediction dicts (one per decoder layer)
                Each dict contains:
                - 'lines': Predicted coordinates [bs, num_queries, 2*num_points]
                - 'scores': Classification scores [bs, num_queries, num_classes]
        """
        # Prepare BEV features with positional encoding
        bev_features = self._prepare_context(bev_features)

        bs, _, h, w = bev_features.shape

        # Create image masks (all zeros = no padding)
        img_masks_np = np.zeros((bs, h, w), dtype=np.bool_)
        img_masks = F._from_data(f"{self.name}_img_masks", img_masks_np, is_const=True)
        setattr(self, img_masks.name, img_masks)

        # Initialize query embeddings
        query_embed_weight = F._from_data(
            f"{self.name}_query_embed_w", self.query_embedding_weight, is_const=True
        )
        setattr(self, query_embed_weight.name, query_embed_weight)

        # Expand to batch: [num_queries, embed_dims] -> [bs, num_queries, embed_dims]
        # First unsqueeze to add batch dimension
        unsq_axis = F._from_data(
            f"{self.name}_unsq_axis", np.array([0], dtype=np.int64), is_const=True
        )
        setattr(self, unsq_axis.name, unsq_axis)
        unsqueeze_op = F.Unsqueeze(f"{self.name}_query_unsq")
        query_embedding = unsqueeze_op(query_embed_weight, unsq_axis)
        setattr(self, unsqueeze_op.name, unsqueeze_op)
        setattr(self, query_embedding.name, query_embedding)

        # Tile to batch size
        tile_repeats = F._from_data(
            f"{self.name}_tile_reps",
            np.array([bs, 1, 1], dtype=np.int64),
            is_const=True,
        )
        setattr(self, tile_repeats.name, tile_repeats)
        tile_op = F.Tile(f"{self.name}_query_tile")
        query_embedding = tile_op(query_embedding, tile_repeats)
        setattr(self, tile_op.name, tile_op)
        setattr(self, query_embedding.name, query_embedding)

        # Initialize reference points
        # Apply linear layer to query embeddings
        init_reference_points = self.ref_points_embed(query_embedding)
        setattr(self, init_reference_points.name, init_reference_points)

        # Apply sigmoid
        sig_op = F.Sigmoid(f"{self.name}_ref_sig")
        init_reference_points = sig_op(init_reference_points)
        setattr(self, sig_op.name, sig_op)
        setattr(self, init_reference_points.name, init_reference_points)

        # Reshape to [bs, num_queries, num_points, 2]
        ref_shape = F._from_data(
            f"{self.name}_ref_shape",
            np.array([bs, self.num_queries, self.num_points, 2], dtype=np.int64),
            is_const=True,
        )
        setattr(self, ref_shape.name, ref_shape)
        reshape_op = F.Reshape(f"{self.name}_ref_reshape")
        init_reference_points = reshape_op(init_reference_points, ref_shape)
        setattr(self, reshape_op.name, reshape_op)
        setattr(self, init_reference_points.name, init_reference_points)

        # Concatenate track queries from previous frame (if available)
        # This prepends tracked queries before detection queries, matching
        # the original PyTorch MapDetectorHead.forward_test behavior.
        num_total_queries = self.num_queries
        if track_query_info is not None and len(track_query_info) > 0:
            has_tracks = (
                "track_query_hs_embeds" in track_query_info[0]
                and track_query_info[0]["track_query_hs_embeds"].size > 0
            )
            if has_tracks:
                # Stack track embeddings: [bs, num_tracks, embed_dims]
                track_embeds_list = []
                track_refs_list = []
                for b_i in range(bs):
                    track_embeds_list.append(
                        track_query_info[b_i]["track_query_hs_embeds"]
                    )
                    track_refs_list.append(
                        track_query_info[b_i]["trans_track_query_boxes"]
                    )

                track_embeds_np = np.stack(
                    track_embeds_list, axis=0
                )  # [bs, N_track, D]
                track_refs_np = np.stack(
                    track_refs_list, axis=0
                )  # [bs, N_track, num_pts*2]
                num_tracks = track_embeds_np.shape[1]
                num_total_queries = num_tracks + self.num_queries

                # Convert to SimTensors
                track_embed_tensor = F._from_data(
                    f"{self.name}_track_embeds",
                    track_embeds_np.astype(np.float32),
                    is_const=False,
                )
                setattr(self, track_embed_tensor.name, track_embed_tensor)

                # Reshape track ref points: [bs, N_track, num_pts*2] -> [bs, N_track, num_pts, 2]
                track_refs_4d = track_refs_np.reshape(
                    bs, num_tracks, self.num_points, 2
                ).astype(np.float32)
                track_ref_tensor = F._from_data(
                    f"{self.name}_track_refs", track_refs_4d, is_const=False
                )
                setattr(self, track_ref_tensor.name, track_ref_tensor)

                # Concatenate: [track_queries, detection_queries] along query dim
                concat_q_op = F.ConcatX(f"{self.name}_concat_track_det_q", axis=1)
                query_embedding = concat_q_op(track_embed_tensor, query_embedding)
                setattr(self, concat_q_op.name, concat_q_op)
                setattr(self, query_embedding.name, query_embedding)

                concat_r_op = F.ConcatX(f"{self.name}_concat_track_det_ref", axis=1)
                init_reference_points = concat_r_op(
                    track_ref_tensor, init_reference_points
                )
                setattr(self, concat_r_op.name, concat_r_op)
                setattr(self, init_reference_points.name, init_reference_points)

        # Create padding mask for queries (all False = no padding)
        query_padding_mask_np = np.zeros((bs, num_total_queries), dtype=np.bool_)
        query_padding_mask = F._from_data(
            f"{self.name}_query_mask", query_padding_mask_np, is_const=True
        )
        setattr(self, query_padding_mask.name, query_padding_mask)

        # Forward through transformer
        inter_queries, _, inter_references = self.transformer(
            mlvl_feats=[bev_features],
            mlvl_masks=[img_masks],
            query_embed=query_embedding,
            mlvl_pos_embeds=[None],
            memory_query=None,
            init_reference_points=init_reference_points,
            reg_branches=self.reg_branches,
            cls_branches=None,  # We'll apply cls branches manually
            predict_refine=self.predict_refine,
            query_key_padding_mask=query_padding_mask,
            memory_bank=memory_bank,
        )

        # Process outputs from each decoder layer
        outputs = []

        for i, queries in enumerate(inter_queries):
            # queries shape: [bs, num_queries, embed_dims]

            # Apply classification branch
            scores = self.cls_branches[i](queries)  # [bs, num_queries, num_classes]
            setattr(self, scores.name, scores)  # Track for execution

            # Get predicted reference points
            reg_points = inter_references[i]  # [bs, num_queries, num_points, 2]
            setattr(self, reg_points.name, reg_points)  # Track

            # Flatten to (bs, num_q, 2*num_points) to match original PyTorch
            if hasattr(reg_points, "data") and reg_points.data is not None:
                rp_data = np.asarray(reg_points.data).copy()
            else:
                rp_data = np.zeros(reg_points.shape, dtype=np.float32)
            bs_rp = rp_data.shape[0]
            rp_flat = rp_data.reshape(
                bs_rp, -1, 2 * self.num_points
            )  # (bs, num_q, 2*num_points)

            # Extract per-batch as lists (matching original)
            reg_points_list = [rp_flat[b] for b in range(bs_rp)]
            scores_list = []
            if hasattr(scores, "data") and scores.data is not None:
                s_data = np.asarray(scores.data).copy()
            else:
                s_data = np.zeros(scores.shape, dtype=np.float32)
            for b in range(s_data.shape[0]):
                scores_list.append(s_data[b])

            # Track query embeddings
            setattr(self, queries.name, queries)  # Track

            # Extract hs_embeds per-batch
            if hasattr(queries, "data") and queries.data is not None:
                q_data = np.asarray(queries.data).copy()
            else:
                q_data = np.zeros(queries.shape, dtype=np.float32)

            pred_dict = {
                "lines": reg_points_list,
                "scores": scores_list,
                "hs_embeds": queries,  # keep as SimTensor (original stores torch tensor with batch dim)
            }
            outputs.append(pred_dict)

        return outputs

    def analytical_param_count(self, lvl=0):
        """Calculate total parameter count for the head."""
        indent = "  " * lvl
        total_params = 0

        if lvl >= 2:
            print(f"{indent}MapDetectorHead '{self.name}':")

        # Input projection: (in_channels * embed_dims * 1 * 1) + embed_dims
        input_proj_params = self.in_channels * self.embed_dims + self.embed_dims
        total_params += input_proj_params
        if lvl >= 2:
            print(f"{indent}  Input projection: {input_proj_params:,}")

        # Query embedding: num_queries * embed_dims
        query_embed_params = self.num_queries * self.embed_dims
        total_params += query_embed_params
        if lvl >= 2:
            print(f"{indent}  Query embedding: {query_embed_params:,}")

        # Reference points embedding: Linear layer
        if hasattr(self, "ref_points_embed"):
            ref_points_params = self.ref_points_embed.analytical_param_count(lvl=0)
        else:
            ref_points_params = self.embed_dims * (self.num_points * 2) + (
                self.num_points * 2
            )
        total_params += ref_points_params
        if lvl >= 2:
            print(f"{indent}  Reference points: {ref_points_params:,}")

        # Classification branches
        for i, cls_branch in enumerate(self.cls_branches):
            cls_params = cls_branch.analytical_param_count(lvl=0)  # type: ignore[attr-defined]
            total_params += cls_params
            if lvl >= 2:
                print(f"{indent}  Cls branch {i}: {cls_params:,}")

        # Regression branches
        for i, reg_branch in enumerate(self.reg_branches):
            reg_params = reg_branch.analytical_param_count(lvl=0)  # type: ignore[attr-defined]
            total_params += reg_params
            if lvl >= 2:
                print(f"{indent}  Reg branch {i}: {reg_params:,}")

        # Transformer
        if self.transformer is not None and hasattr(
            self.transformer, "analytical_param_count"
        ):
            transformer_params = self.transformer.analytical_param_count(
                lvl=lvl + 1 if lvl >= 2 else 0
            )
            total_params += transformer_params
            if lvl >= 2:
                print(f"{indent}  Transformer: {transformer_params:,}")

        if lvl >= 1:
            print(f"{indent}Total MapDetectorHead params: {total_params:,}")

        return total_params

    # =========================================================================
    # Inference-time temporal tracking methods
    # (These operate on numpy data outside the TTSim computation graph)
    # =========================================================================

    def post_process(self, preds_dict, tokens, track_dict=None, thr=0.0):
        """Post-process predictions to extract final detection results.

        Filters predictions by score threshold and formats outputs.

        Args:
            preds_dict: Dict with 'lines', 'scores', 'hs_embeds' from last decoder layer
            tokens: List of sample tokens
            track_dict: Optional track query results dict
            thr: Score threshold for filtering (default: 0.0)

        Returns:
            results: List of dicts per batch with keys:
                'vectors', 'scores', 'labels', 'props', 'token',
                'track_scores', 'track_vectors', 'track_labels'
        """
        lines = preds_dict["lines"]
        scores = preds_dict["scores"]

        # Convert to numpy, handling SimTensor/memoryview/numpy/list
        def _extract(x):
            if hasattr(x, "data"):
                if x.data is not None:
                    return np.asarray(x.data).copy()
                elif hasattr(x, "shape") and x.shape is not None:
                    return np.zeros(x.shape, dtype=np.float32)
                else:
                    return None
            elif isinstance(x, np.ndarray):
                return x.copy()
            elif isinstance(x, list):
                return x  # already a list, handle per-element below
            else:
                try:
                    return np.asarray(x).copy()
                except (TypeError, ValueError):
                    return None

        lines_np = _extract(lines)
        scores_np = _extract(scores)

        # Flatten 4D reference points [bs, nq, num_points, 2] → [bs, nq, num_points*2]
        if isinstance(lines_np, np.ndarray) and lines_np.ndim == 4:
            lines_np = lines_np.reshape(lines_np.shape[0], lines_np.shape[1], -1)

        # Determine batch size and ensure list-of-arrays format
        if isinstance(lines_np, np.ndarray):
            if lines_np.ndim == 3:  # [bs, num_queries, coords]
                bs = lines_np.shape[0]
                lines_list = [lines_np[i] for i in range(bs)]
                scores_list = [scores_np[i] for i in range(bs)]
            elif lines_np.ndim == 2:  # [num_queries, coords] single batch
                bs = 1
                lines_list = [lines_np]
                scores_list = [scores_np]
            else:
                bs = 1
                lines_list = [
                    np.zeros((self.num_queries, 2 * self.num_points), dtype=np.float32)
                ]
                scores_list = [
                    np.zeros((self.num_queries, self.num_classes), dtype=np.float32)
                ]
        elif isinstance(lines_np, list):
            bs = len(lines_np)
            lines_list = lines_np
            scores_list = scores_np if isinstance(scores_np, list) else [scores_np]
        else:
            bs = 1
            lines_list = [
                np.zeros((self.num_queries, 2 * self.num_points), dtype=np.float32)
            ]
            scores_list = [
                np.zeros((self.num_queries, self.num_classes), dtype=np.float32)
            ]

        results = []
        for i in range(bs):
            # Get lines and scores for this batch item
            tmp_vectors = (
                lines_list[i]
                if isinstance(lines_list[i], np.ndarray)
                else _extract(lines_list[i])
            )
            if tmp_vectors is None:
                tmp_vectors = np.zeros(
                    (self.num_queries, 2 * self.num_points), dtype=np.float32
                )

            score_data = (
                scores_list[i]
                if isinstance(scores_list[i], np.ndarray)
                else _extract(scores_list[i])
            )
            if score_data is None:
                score_data = np.zeros(
                    (self.num_queries, self.num_classes), dtype=np.float32
                )

            # Set up prop flags (track queries vs detection queries)
            tmp_prop_flags = np.zeros(tmp_vectors.shape[0], dtype=bool)
            if tmp_vectors.shape[0] > self.num_queries:
                tmp_prop_flags[: -self.num_queries] = True

            num_preds = tmp_vectors.shape[0]
            num_points2 = (
                tmp_vectors.shape[1] if tmp_vectors.ndim > 1 else 2 * self.num_points
            )
            tmp_vectors = tmp_vectors.reshape(num_preds, num_points2 // 2, 2)

            # Score thresholding (sigmoid classification)
            tmp_scores = np.max(score_data, axis=-1)
            tmp_labels = np.argmax(score_data, axis=-1)
            tmp_scores = 1.0 / (1.0 + np.exp(-tmp_scores))  # sigmoid
            pos = tmp_scores > thr

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_flags = tmp_prop_flags[pos]

            if len(tmp_scores) == 0:
                single_result = {
                    "vectors": [],
                    "scores": [],
                    "labels": [],
                    "props": [],
                    "token": tokens[i],
                }
            else:
                single_result = {
                    "vectors": tmp_vectors,
                    "scores": tmp_scores,
                    "labels": tmp_labels,
                    "props": tmp_prop_flags,
                    "token": tokens[i],
                }

            # Track query analysis
            if track_dict is not None and len(track_dict.get("lines", [])) > 0:
                tmp_track_scores = track_dict["scores"][i]
                tmp_track_vectors = track_dict["lines"][i]
                tmp_track_labels = np.argmax(tmp_track_scores, axis=-1)
                tmp_track_scores_max = np.max(tmp_track_scores, axis=-1)
                tmp_track_scores_max = 1.0 / (1.0 + np.exp(-tmp_track_scores_max))
                single_result["track_scores"] = tmp_track_scores_max
                single_result["track_vectors"] = tmp_track_vectors
                single_result["track_labels"] = tmp_track_labels
            else:
                single_result["track_scores"] = []
                single_result["track_vectors"] = []
                single_result["track_labels"] = []

            results.append(single_result)

        return results

    def prepare_temporal_propagation(
        self,
        preds_dict,
        scene_name,
        local_idx,
        memory_bank=None,
        thr_track=0.1,
        thr_det=0.5,
    ):
        """Prepare track queries for propagation to the next frame.

        Selects high-confidence predictions and stores them for use as
        track queries in the next frame's inference.

        Args:
            preds_dict: Dict with 'lines', 'scores', 'hs_embeds'
            scene_name: Current scene identifier
            local_idx: Current frame index within the sequence
            memory_bank: Optional VectorInstanceMemory for tracking
            thr_track: Score threshold for tracked queries (default: 0.1)
            thr_det: Score threshold for new detections (default: 0.5)

        Returns:
            save_pos_results: Dict of positive predictions for analysis
        """
        lines = preds_dict["lines"]
        queries = preds_dict["hs_embeds"]
        scores = preds_dict["scores"]

        first_frame = local_idx == 0

        # Extract data (handle SimTensors, memoryview, and numpy)
        def _to_numpy(x):
            if hasattr(x, "data"):
                # SimTensor — extract .data or fall back to zeros from .shape
                if x.data is not None:
                    return np.asarray(x.data).copy()
                elif hasattr(x, "shape") and x.shape is not None:
                    return np.zeros(x.shape, dtype=np.float32)
                else:
                    return np.zeros(0, dtype=np.float32)
            elif isinstance(x, np.ndarray):
                return x.copy()
            elif isinstance(x, list) and len(x) > 0:
                return _to_numpy(x[0])
            else:
                try:
                    return np.asarray(x).copy()
                except (TypeError, ValueError):
                    return x

        tmp_vectors = (
            _to_numpy(lines[0]) if isinstance(lines, list) else _to_numpy(lines)
        )

        # queries (hs_embeds) may be a SimTensor (bs, nq, dim) — index [0] after extraction
        _q = _to_numpy(queries[0]) if isinstance(queries, list) else _to_numpy(queries)
        tmp_queries = _q[0] if _q.ndim == 3 else _q  # remove batch dim

        score_data = (
            _to_numpy(scores[0]) if isinstance(scores, list) else _to_numpy(scores)
        )

        # Sigmoid classification scores
        tmp_scores = np.max(score_data, axis=-1)
        tmp_labels = np.argmax(score_data, axis=-1)
        tmp_scores = 1.0 / (1.0 + np.exp(-tmp_scores))

        # Separate thresholds for tracked vs detection queries
        num_total = len(tmp_scores)
        if num_total > self.num_queries:
            pos_track = tmp_scores[: -self.num_queries] > thr_track
            pos_det = tmp_scores[-self.num_queries :] > thr_det
            pos = np.concatenate([pos_track, pos_det])
        else:
            pos = tmp_scores > thr_det
            pos_track = np.array([], dtype=bool)
            pos_det = pos  # all queries are detection queries

        pos_vectors = tmp_vectors[pos]
        pos_labels = tmp_labels[pos]
        pos_queries = tmp_queries[pos]
        pos_scores = tmp_scores[pos]

        # Assign global IDs for tracking
        if first_frame:
            global_ids = np.arange(len(pos_vectors))
            num_instance = len(pos_vectors)
        else:
            if hasattr(self, "prop_info") and self.prop_info is not None:
                prop_ids = self.prop_info["global_ids"]
                prop_num_instance = self.prop_info["num_instance"]
                global_ids_track = (
                    prop_ids[pos_track] if len(pos_track) > 0 else np.array([])
                )
                num_newborn = int(pos_det.sum()) if len(pos_det) > 0 else 0
                global_ids_newborn = np.arange(num_newborn) + prop_num_instance
                global_ids = np.concatenate(
                    [global_ids_track, global_ids_newborn]
                ).astype(np.int64)
                num_instance = prop_num_instance + num_newborn
            else:
                global_ids = np.arange(len(pos_vectors))
                num_instance = len(pos_vectors)

        self.prop_info = {
            "vectors": pos_vectors,
            "queries": pos_queries,
            "scores": pos_scores,
            "labels": pos_labels,
            "scene_name": scene_name,
            "local_idx": local_idx,
            "global_ids": global_ids,
            "num_instance": num_instance,
        }

        # Update memory bank
        if memory_bank is not None:
            if first_frame:
                num_tracks = 0
            else:
                num_tracks = getattr(self, "prop_active_tracks", 0)
            pos_out_inds = np.where(pos)[0]
            prev_out = {
                "hs_embeds": _to_numpy(queries),
                "scores": _to_numpy(scores),
            }
            memory_bank.update_memory(
                0, first_frame, pos_out_inds, prev_out, local_idx, num_tracks
            )
            self.prop_active_tracks = len(pos_out_inds)

        save_pos_results = {
            "vectors": pos_vectors,
            "scores": pos_scores,
            "labels": pos_labels,
            "global_ids": global_ids,
            "scene_name": scene_name,
            "local_idx": local_idx,
            "num_instance": num_instance,
        }

        return save_pos_results

    def get_track_info(self, scene_name, local_idx):
        """Retrieve track query info from the previous frame for propagation.

        Uses stored prop_info from prepare_temporal_propagation to create
        track query inputs for the current frame.

        Args:
            scene_name: Current scene name (must match stored info)
            local_idx: Current frame index (must be stored_idx + 1)

        Returns:
            track_info: List of dicts (one per batch, bs=1) with keys:
                'track_query_hs_embeds': [num_tracks, embed_dims]
                'track_query_boxes': [num_tracks, num_points*2]
        """
        if not hasattr(self, "prop_info") or self.prop_info is None:
            return None

        prop_info = self.prop_info
        # Verify scene continuity
        assert prop_info["scene_name"] == scene_name and (
            prop_info["local_idx"] + 1 == local_idx
            or prop_info["local_idx"] == local_idx
        ), (
            f"Scene/frame mismatch: stored=({prop_info['scene_name']}, {prop_info['local_idx']}), "
            f"requested=({scene_name}, {local_idx})"
        )

        vectors = prop_info["vectors"]
        queries = prop_info["queries"]

        target = {
            "track_query_hs_embeds": queries,
            "track_query_boxes": vectors,
        }
        track_info = [target]

        return track_info

    def clear_temporal_cache(self):
        """Clear stored temporal propagation information."""
        self.prop_info = None
