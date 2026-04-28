#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of FusionADTrack for FusionAD.

Extends SimNN.Module to provide:
  - Image backbone (ResNet-101) + FPN neck
  - LiDAR backbone (SparseEncoderHD)
  - BEV encoding via PerceptionTransformer (BEVFormerEncoder)
  - Detection via BEVFormerTrackHead (6-layer decoder + cls/reg/traj branches)
  - Query Interaction Module (QIM) for multi-frame tracking
  - Memory Bank for historical track queries
  - Runtime tracker for score-based filtering

Inference pipeline:
  1. extract_img_feat -> backbone + neck
  2. extract_pts_feat -> LiDAR backbone
  3. simple_test_track -> BEV encode + detect + track
  4. upsample_bev_if_tiny -> upscale for small BEV grids

"""

#----------------------------------PyTorch----------------------------------------#

# import torch
# import torch.nn as nn
# from mmcv.runner import force_fp32, auto_fp16
# from torch.nn import functional as F
# from mmdet.models import DETECTORS
# from mmdet3d.core import bbox3d2result
# from mmdet3d.core.bbox.coders import build_bbox_coder
# from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
# from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
# import copy
# import math
# from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# from mmdet.models import build_loss
# from einops import rearrange
# from mmdet.models.utils.transformer import inverse_sigmoid
# from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase
#
# @DETECTORS.register_module()
# class FusionADTrack(MVXTwoStageDetector):
#     """UniAD tracking part
#     """
#     def __init__(
#         self,
#         use_grid_mask=False,
#         pts_voxel_layer=None,
#         pts_voxel_encoder=None,
#         pts_middle_encoder=None,
#         pts_backbone=None,
#         pts_neck=None,
#         img_backbone=None,
#         img_neck=None,
#         pts_bbox_head=None,
#         train_cfg=None,
#         test_cfg=None,
#         pretrained=None,
#         video_test_mode=False,
#         loss_cfg=None,
#         qim_args=dict(
#             qim_type="QIMBase",
#             merger_dropout=0,
#             update_query_pos=False,
#             fp_ratio=0.3,
#             random_drop=0.1,
#         ),
#         mem_args=dict(
#             memory_bank_type="MemoryBank",
#             memory_bank_score_thresh=0.0,
#             memory_bank_len=4,
#         ),
#         bbox_coder=dict(
#             type="DETRTrack3DCoder",
#             post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#             pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
#             max_num=300,
#             num_classes=10,
#             score_threshold=0.0,
#             with_nms=False,
#             iou_thres=0.3,
#         ),
#         pc_range=None,
#         embed_dims=256,
#         num_query=900,
#         num_classes=10,
#         vehicle_id_list=None,
#         score_thresh=0.2,
#         filter_score_thresh=0.1,
#         miss_tolerance=5,
#         gt_iou_threshold=0.0,
#
#         freeze_img_modules=False,   # * Remember to use it
#         freeze_bev_encoder=False,
#         queue_length=3,
#     ):
#         super(FusionADTrack, self).__init__(
#             pts_voxel_layer=pts_voxel_layer,
#             pts_voxel_encoder=pts_voxel_encoder,
#             pts_middle_encoder=pts_middle_encoder,
#             pts_backbone=pts_backbone,
#             pts_neck=pts_neck,
#             img_backbone=img_backbone,
#             img_neck=img_neck,
#             pts_bbox_head=pts_bbox_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             pretrained=pretrained,
#         )
#
#         self.grid_mask = GridMask(
#             True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
#         )
#         self.use_grid_mask = use_grid_mask
#         self.fp16_enabled = False
#         self.embed_dims = embed_dims
#         self.num_query = num_query
#         self.freeze_img_modules = freeze_img_modules
#         self.num_classes = num_classes
#         self.vehicle_id_list = vehicle_id_list
#         self.pc_range = pc_range
#         self.queue_length = queue_length
#         if self.freeze_img_modules:
#             self.img_backbone.eval()
#             self.img_neck.eval()
#
#         # temporal
#         self.video_test_mode = video_test_mode
#         assert self.video_test_mode
#
#         self.prev_frame_info = {
#             "prev_bev": None,
#             "scene_token": None,
#             "prev_pos": 0,
#             "prev_angle": 0,
#         }
#         self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)
#         self.reference_points = nn.Linear(self.embed_dims, 3)
#         self.bbox_size_fc = nn.Linear(self.embed_dims, 3)
#
#         self.mem_bank_len = mem_args["memory_bank_len"]
#         self.memory_bank = None
#         self.track_base = RuntimeTrackerBase(
#             score_thresh=score_thresh,
#             filter_score_thresh=filter_score_thresh,
#             miss_tolerance=miss_tolerance,
#         )  # hyper-param for removing inactive queries
#
#         self.query_interact = QueryInteractionModule(
#             qim_args,
#             dim_in=embed_dims,
#             hidden_dim=embed_dims,
#             dim_out=embed_dims,
#         )
#
#         self.bbox_coder = build_bbox_coder(bbox_coder)
#
#         self.memory_bank = MemoryBank(
#             mem_args,
#             dim_in=embed_dims,
#             hidden_dim=embed_dims,
#             dim_out=embed_dims,
#         )
#         self.mem_bank_len = (
#             0 if self.memory_bank is None else self.memory_bank.max_his_length
#         )
#         self.criterion = build_loss(loss_cfg)
#         self.test_track_instances = None
#         self.l2g_r_mat = None
#         self.l2g_t = None
#         self.gt_iou_threshold = gt_iou_threshold
#         self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
#         self.freeze_bev_encoder = freeze_bev_encoder
#         self.voxelize_reduce = True
#
#
#
#     @torch.no_grad()
#     @force_fp32()
#     def voxelize(self, points):
#         feats, coords, sizes = [], [], []
#         for k, res in enumerate(points):
#             f, c, n = self.pts_voxel_layer(res)
#             feats.append(f)
#             coords.append(F.pad(c, (1, 0), mode="constant", value=k))
#             sizes.append(n)
#
#         feats = torch.cat(feats, dim=0)
#         coords = torch.cat(coords, dim=0)
#         sizes = torch.cat(sizes, dim=0)
#
#         if self.voxelize_reduce:
#             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
#             feats = feats.contiguous()
#
#         return feats, coords, sizes
#
#     @force_fp32()
#     def extract_pts_feat(self, pts):
#         feats, coords, sizes = self.voxelize(pts)
#         batch_size = coords[-1, 0] + 1
#         x = self.pts_backbone(feats, coords, batch_size)
#
#         return x
#
#     def extract_img_feat(self, img, len_queue=None):
#         """Extract features of images."""
#         if img is None:
#             return None
#         assert img.dim() == 5
#         B, N, C, H, W = img.size()
#         img = img.reshape(B * N, C, H, W)
#         if self.use_grid_mask:
#             img = self.grid_mask(img)
#         img_feats = self.img_backbone(img)
#         if isinstance(img_feats, dict):
#             img_feats = list(img_feats.values())
#         if self.with_img_neck:
#             img_feats = self.img_neck(img_feats)
#
#         img_feats_reshaped = []
#         for img_feat in img_feats:
#             _, c, h, w = img_feat.size()
#             if len_queue is not None:
#                 img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
#             else:
#                 img_feat_reshaped = img_feat.view(B, N, c, h, w)
#             img_feats_reshaped.append(img_feat_reshaped)
#         return img_feats_reshaped
#
#     @auto_fp16(apply_to=("img"))
#     def extract_feat(self, img, len_queue=None):
#         """Extract features from images and points."""
#         if self.freeze_img_modules:
#             with torch.no_grad():
#                 img_feats = self.extract_img_feat(img, len_queue=len_queue)
#         else:
#             img_feats = self.extract_img_feat(img, len_queue=len_queue)
#         return img_feats
#
#     def _generate_empty_tracks(self):
#         track_instances = Instances((1, 1))
#         num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
#         device = self.query_embedding.weight.device
#         query = self.query_embedding.weight
#         track_instances.ref_pts = self.reference_points(query[..., : dim // 2])
#
#         # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
#         box_sizes = self.bbox_size_fc(query[..., : dim // 2])
#         pred_boxes_init = torch.zeros(
#             (len(track_instances), 10), dtype=torch.float, device=device
#         )
#
#         pred_boxes_init[..., 2:4] = box_sizes[..., 0:2]
#         pred_boxes_init[..., 5:6] = box_sizes[..., 2:3]
#
#         track_instances.query = query
#
#         track_instances.output_embedding = torch.zeros(
#             (num_queries, dim >> 1), device=device
#         )
#
#         track_instances.obj_idxes = torch.full(
#             (len(track_instances),), -1, dtype=torch.long, device=device
#         )
#         track_instances.matched_gt_idxes = torch.full(
#             (len(track_instances),), -1, dtype=torch.long, device=device
#         )
#         track_instances.disappear_time = torch.zeros(
#             (len(track_instances),), dtype=torch.long, device=device
#         )
#
#         track_instances.iou = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         track_instances.scores = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         track_instances.track_scores = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         # xy, wl, z, h, sin, cos, vx, vy, vz
#         track_instances.pred_boxes = pred_boxes_init
#
#         track_instances.pred_logits = torch.zeros(
#             (len(track_instances), self.num_classes), dtype=torch.float, device=device
#         )
#
#         mem_bank_len = self.mem_bank_len
#         track_instances.mem_bank = torch.zeros(
#             (len(track_instances), mem_bank_len, dim // 2),
#             dtype=torch.float32,
#             device=device,
#         )
#         track_instances.mem_padding_mask = torch.ones(
#             (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
#         )
#         track_instances.save_period = torch.zeros(
#             (len(track_instances),), dtype=torch.float32, device=device
#         )
#
#         return track_instances.to(self.query_embedding.weight.device)
#
#     def velo_update(
#         self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
#     ):
#         """
#         Args:
#             ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
#             velocity (Tensor): (num_query, 2). m/s
#                 in lidar frame. vx, vy
#             global2lidar (np.Array) [4,4].
#         Outs:
#             ref_pts (Tensor): (num_query, 3).  in inevrse sigmoid space
#         """
#         # logging.info(l2g_r1.type(), l2g_t1.type(), ref_pts.type())
#         time_delta = time_delta.type(torch.float)
#         num_query = ref_pts.size(0)
#         velo_pad_ = velocity.new_zeros((num_query, 1))
#         velo_pad = torch.cat((velocity, velo_pad_), dim=-1)
#
#         reference_points = ref_pts.sigmoid().clone()
#         pc_range = self.pc_range
#         reference_points[..., 0:1] = (
#             reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
#         )
#         reference_points[..., 1:2] = (
#             reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
#         )
#         reference_points[..., 2:3] = (
#             reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
#         )
#
#         reference_points = reference_points + velo_pad * time_delta
#
#         ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
#
#         g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)
#
#         ref_pts = ref_pts @ g2l_r
#
#         ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
#             pc_range[3] - pc_range[0]
#         )
#         ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
#             pc_range[4] - pc_range[1]
#         )
#         ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
#             pc_range[5] - pc_range[2]
#         )
#
#         ref_pts = inverse_sigmoid(ref_pts)
#
#         return ref_pts
#
#     def _copy_tracks_for_loss(self, tgt_instances):
#         device = self.query_embedding.weight.device
#         track_instances = Instances((1, 1))
#
#         track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)
#
#         track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
#         track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)
#
#         track_instances.scores = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         track_instances.track_scores = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         track_instances.pred_boxes = torch.zeros(
#             (len(track_instances), 10), dtype=torch.float, device=device
#         )
#         track_instances.iou = torch.zeros(
#             (len(track_instances),), dtype=torch.float, device=device
#         )
#         track_instances.pred_logits = torch.zeros(
#             (len(track_instances), self.num_classes), dtype=torch.float, device=device
#         )
#
#         track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
#         return track_instances.to(device)
#
#     def get_history_bev(self, imgs_queue, img_metas_list, prev_points=None):
#         """
#         Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
#         """
#         self.eval()
#         with torch.no_grad():
#             prev_bev = None
#             bs, len_queue, num_cams, C, H, W = imgs_queue.shape
#             imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
#             img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
#             for i in range(len_queue):
#                 img_metas = [each[i] for each in img_metas_list]
#                 img_feats = [each_scale[:, i] for each_scale in img_feats_list]
#                 if prev_points is not None:
#                     pts_feats = self.extract_pts_feat([prev_points[i]])
#                 else:
#                     pts_feats = None
#                 prev_bev, _ = self.pts_bbox_head.get_bev_features(
#                     mlvl_feats=img_feats,
#                     img_metas=img_metas,
#                     prev_bev=prev_bev,
#                     pts_feats=pts_feats)
#         self.train()
#         return prev_bev
#
#     # Generate bev using bev_encoder in BEVFormer
#     def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None, points=None, prev_points=None):
#         if prev_img is not None and prev_img_metas is not None:
#             assert prev_bev is None
#             prev_bev = self.get_history_bev(prev_img, prev_img_metas, prev_points=prev_points)
#
#         img_feats = self.extract_feat(img=imgs)
#         if points is not None:
#             pts_feats = self.extract_pts_feat(points)
#         else:
#             pts_feats = None
#
#         if self.freeze_bev_encoder:
#             with torch.no_grad():
#                 bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
#                     mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev, pts_feats=pts_feats)
#         else:
#             bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
#                     mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev, pts_feats=pts_feats)
#
#         if bev_embed.shape[1] == self.bev_h * self.bev_w:
#             bev_embed = bev_embed.permute(1, 0, 2)
#
#         assert bev_embed.shape[0] == self.bev_h * self.bev_w
#         return bev_embed, bev_pos, prev_bev
#
#     @auto_fp16(apply_to=("img", "prev_bev"))
#     def _forward_single(
#         self,
#         points,
#         prev_points,
#         img,
#         img_metas,
#         track_instances,
#         prev_img,
#         prev_img_metas,
#         l2g_r1=None,
#         l2g_t1=None,
#         l2g_r2=None,
#         l2g_t2=None,
#         time_delta=None,
#         all_query_embeddings=None,
#         all_matched_indices=None,
#         all_instances_pred_logits=None,
#         all_instances_pred_boxes=None,
#     ):
#         """
#         Perform forward only on one frame. Called in  forward_train
#         Warnning: Only Support BS=1
#         Args:
#             img: shape [B, num_cam, 3, H, W]
#             if l2g_r2 is None or l2g_t2 is None:
#                 it means this frame is the end of the training clip,
#                 so no need to call velocity update
#         """
#         # NOTE: You can replace BEVFormer with other BEV encoder and generate bev_embed here
#         bev_embed, bev_pos, _ = self.get_bevs(
#             img, img_metas,
#             prev_img=prev_img, prev_img_metas=prev_img_metas, points=points, prev_points=prev_points,
#         )
#
#         det_output = self.pts_bbox_head.get_detections(
#             bev_embed,
#             object_query_embeds=track_instances.query,
#             ref_points=track_instances.ref_pts,
#             img_metas=img_metas,
#         )
#
#         output_classes = det_output["all_cls_scores"]
#         output_coords = det_output["all_bbox_preds"]
#         output_past_trajs = det_output["all_past_traj_preds"]
#         last_ref_pts = det_output["last_ref_points"]
#         query_feats = det_output["query_feats"]
#
#         out = {
#             "pred_logits": output_classes[-1],
#             "pred_boxes": output_coords[-1],
#             "pred_past_trajs": output_past_trajs[-1],
#             "ref_pts": last_ref_pts,
#             "bev_embed": bev_embed,
#             "bev_pos": bev_pos
#         }
#         with torch.no_grad():
#             track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
#
#         # Step-1 Update track instances with current prediction
#         # [nb_dec, bs, num_query, xxx]
#         nb_dec = output_classes.size(0)
#
#         # the track id will be assigned by the matcher.
#         track_instances_list = [
#             self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
#         ]
#         track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
#         velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
#         if l2g_r2 is not None:
#             ref_pts = self.velo_update(
#                 last_ref_pts[0],
#                 velo,
#                 l2g_r1,
#                 l2g_t1,
#                 l2g_r2,
#                 l2g_t2,
#                 time_delta=time_delta,
#             )
#         else:
#             ref_pts = last_ref_pts[0]
#
#         dim = track_instances.query.shape[-1]
#         track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
#         track_instances.ref_pts[...,:2] = ref_pts[...,:2]
#
#         track_instances_list.append(track_instances)
#
#         for i in range(nb_dec):
#             track_instances = track_instances_list[i]
#
#             track_instances.scores = track_scores
#             track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
#             track_instances.pred_boxes = output_coords[i, 0]  # [300, box_dim]
#             track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300,past_steps, 2]
#
#             out["track_instances"] = track_instances
#             track_instances, matched_indices = self.criterion.match_for_single_frame(
#                 out, i, if_step=(i == (nb_dec - 1))
#             )
#             all_query_embeddings.append(query_feats[i][0])
#             all_matched_indices.append(matched_indices)
#             all_instances_pred_logits.append(output_classes[i, 0])
#             all_instances_pred_boxes.append(output_coords[i, 0])   # Not used
#
#         active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
#         out.update(self.select_active_track_query(track_instances, active_index, img_metas))
#         out.update(self.select_sdc_track_query(track_instances[900], img_metas))
#
#         # memory bank
#         if self.memory_bank is not None:
#             track_instances = self.memory_bank(track_instances)
#         # Step-2 Update track instances using matcher
#
#         tmp = {}
#         tmp["init_track_instances"] = self._generate_empty_tracks()
#         tmp["track_instances"] = track_instances
#         out_track_instances = self.query_interact(tmp)
#         out["track_instances"] = out_track_instances
#         return out
#
#     def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
#         result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
#         result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
#         result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
#         return result_dict
#
#     def select_sdc_track_query(self, sdc_instance, img_metas):
#         out = dict()
#         result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
#         out["sdc_boxes_3d"] = result_dict['boxes_3d']
#         out["sdc_scores_3d"] = result_dict['scores_3d']
#         out["sdc_track_scores"] = result_dict['track_scores']
#         out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
#         out["sdc_embedding"] = sdc_instance.output_embedding[0]
#         return out
#
#     @auto_fp16(apply_to=("img", "points"))
#     def forward_track_train(self,
#                             points,
#                             img,
#                             gt_bboxes_3d,
#                             gt_labels_3d,
#                             gt_past_traj,
#                             gt_past_traj_mask,
#                             gt_inds,
#                             gt_sdc_bbox,
#                             gt_sdc_label,
#                             l2g_t,
#                             l2g_r_mat,
#                             img_metas,
#                             timestamp):
#         """Forward funciton
#         Args:
#         Returns:
#         """
#         track_instances = self._generate_empty_tracks()
#         num_frame = img.size(1)
#         # init gt instances!
#         gt_instances_list = []
#
#         for i in range(num_frame):
#             gt_instances = Instances((1, 1))
#             boxes = gt_bboxes_3d[0][i].tensor.to(img.device)
#             # normalize gt bboxes here!
#             boxes = normalize_bbox(boxes, self.pc_range)
#             sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
#             sd_boxes = normalize_bbox(sd_boxes, self.pc_range)
#             gt_instances.boxes = boxes
#             gt_instances.labels = gt_labels_3d[0][i]
#             gt_instances.obj_ids = gt_inds[0][i]
#             gt_instances.past_traj = gt_past_traj[0][i].float()
#             gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()
#             gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)  # boxes.shape[0] sometimes 0
#             gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
#             gt_instances_list.append(gt_instances)
#
#         self.criterion.initialize_for_single_clip(gt_instances_list)
#
#         out = dict()
#
#         for i in range(num_frame):
#             prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
#             prev_img_metas = copy.deepcopy(img_metas)
#
#             if points is not None:
#                 prev_points = points[0][:i] if i != 0 else points[0][:1]
#                 points_single = [points[0][i]]
#             else:
#                 prev_points = None
#                 points_single = None
#
#             # TODO: Generate prev_bev in an RNN way.
#
#             img_single = torch.stack([img_[i] for img_ in img], dim=0)
#             img_metas_single = [copy.deepcopy(img_metas[0][i])]
#             if i == num_frame - 1:
#                 l2g_r2 = None
#                 l2g_t2 = None
#                 time_delta = None
#             else:
#                 l2g_r2 = l2g_r_mat[0][i + 1]
#                 l2g_t2 = l2g_t[0][i + 1]
#                 time_delta = timestamp[0][i + 1] - timestamp[0][i]
#             all_query_embeddings = []
#             all_matched_idxes = []
#             all_instances_pred_logits = []
#             all_instances_pred_boxes = []
#             frame_res = self._forward_single(
#                 points_single,
#                 prev_points,
#                 img_single,
#                 img_metas_single,
#                 track_instances,
#                 prev_img,
#                 prev_img_metas,
#                 l2g_r_mat[0][i],
#                 l2g_t[0][i],
#                 l2g_r2,
#                 l2g_t2,
#                 time_delta,
#                 all_query_embeddings,
#                 all_matched_idxes,
#                 all_instances_pred_logits,
#                 all_instances_pred_boxes,
#             )
#             # all_query_embeddings: len=dec nums, N*256
#             # all_matched_idxes: len=dec nums, N*2
#             track_instances = frame_res["track_instances"]
#             if i == num_frame - 1:
#                 get_keys = ["bev_embed", "bev_pos",
#                             "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
#                             "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
#                 out.update({k: frame_res[k] for k in get_keys})
#         losses = self.criterion.losses_dict
#         return losses, out
#
#     def upsample_bev_if_tiny(self, outs_track):
#         if outs_track["bev_embed"].size(0) == 100 * 100:
#             # For tiny model
#             # bev_emb
#             bev_embed = outs_track["bev_embed"] # [10000, 1, 256]
#             dim, _, _ = bev_embed.size()
#             w = h = int(math.sqrt(dim))
#             assert h == w == 100
#
#             bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
#             bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
#             bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')
#             outs_track["bev_embed"] = bev_embed
#
#             # prev_bev
#             prev_bev = outs_track.get("prev_bev", None)
#             if prev_bev is not None:
#                 if self.training:
#                     #  [1, 10000, 256]
#                     prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
#                     prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
#                     prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
#                     outs_track["prev_bev"] = prev_bev
#                 else:
#                     #  [10000, 1, 256]
#                     prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
#                     prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
#                     prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
#                     outs_track["prev_bev"] = prev_bev
#
#             # bev_pos
#             bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
#             bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
#             outs_track["bev_pos"] = bev_pos
#         return outs_track
#
#
#     def _inference_single(
#         self,
#         points,
#         img,
#         img_metas,
#         track_instances,
#         prev_bev=None,
#         l2g_r1=None,
#         l2g_t1=None,
#         l2g_r2=None,
#         l2g_t2=None,
#         time_delta=None,
#     ):
#         """
#         img: B, num_cam, C, H, W = img.shape
#         """
#
#         """ velo update """
#         active_inst = track_instances[track_instances.obj_idxes >= 0]
#         other_inst = track_instances[track_instances.obj_idxes < 0]
#
#         if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
#             ref_pts = active_inst.ref_pts
#             velo = active_inst.pred_boxes[:, -2:]
#             ref_pts = self.velo_update(
#                 ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
#             )
#             ref_pts = ref_pts.squeeze(0)
#             dim = active_inst.query.shape[-1]
#             active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
#             active_inst.ref_pts[...,:2] = ref_pts[...,:2]
#
#         track_instances = Instances.cat([other_inst, active_inst])
#
#         # NOTE: You can replace BEVFormer with other BEV encoder and generate bev_embed here
#         bev_embed, bev_pos, _ = self.get_bevs(img, img_metas, prev_bev=prev_bev, points=points)
#         det_output = self.pts_bbox_head.get_detections(
#             bev_embed,
#             object_query_embeds=track_instances.query,
#             ref_points=track_instances.ref_pts,
#             img_metas=img_metas,
#         )
#         output_classes = det_output["all_cls_scores"]
#         output_coords = det_output["all_bbox_preds"]
#         last_ref_pts = det_output["last_ref_points"]
#         query_feats = det_output["query_feats"]
#
#         out = {
#             "pred_logits": output_classes,
#             "pred_boxes": output_coords,
#             "ref_pts": last_ref_pts,
#             "bev_embed": bev_embed,
#             "query_embeddings": query_feats,
#             "all_past_traj_preds": det_output["all_past_traj_preds"],
#             "bev_pos": bev_pos,
#         }
#
#         """ update track instances with predict results """
#         track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
#         # each track will be assigned an unique global id by the track base.
#         track_instances.scores = track_scores
#         # track_instances.track_scores = track_scores  # [300]
#         track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
#         track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
#         track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
#         track_instances.ref_pts = last_ref_pts[0]
#         # hard_code: assume the 901 query is sdc query
#         track_instances.obj_idxes[900] = -2
#         """ update track base """
#         self.track_base.update(track_instances, None)
#
#         active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)    # filter out sleep objects
#         out.update(self.select_active_track_query(track_instances, active_index, img_metas))
#         out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))
#
#         """ update with memory_bank """
#         if self.memory_bank is not None:
#             track_instances = self.memory_bank(track_instances)
#
#         """  Update track instances using matcher """
#         tmp = {}
#         tmp["init_track_instances"] = self._generate_empty_tracks()
#         tmp["track_instances"] = track_instances
#         out_track_instances = self.query_interact(tmp)
#         out["track_instances_fordet"] = track_instances
#         out["track_instances"] = out_track_instances
#         out["track_obj_idxes"] = track_instances.obj_idxes
#         return out
#
#     def simple_test_track(
#         self,
#         points=None,
#         img=None,
#         l2g_t=None,
#         l2g_r_mat=None,
#         img_metas=None,
#         timestamp=None,
#     ):
#         """only support bs=1 and sequential input"""
#
#         bs = img.size(0)
#         # img_metas = img_metas[0]
#
#         """ init track instances for first frame """
#         if (
#             self.test_track_instances is None
#             or img_metas[0]["scene_token"] != self.scene_token
#         ):
#             self.timestamp = timestamp
#             self.scene_token = img_metas[0]["scene_token"]
#             self.prev_bev = None
#             track_instances = self._generate_empty_tracks()
#             time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
#
#         else:
#             track_instances = self.test_track_instances
#             time_delta = timestamp - self.timestamp
#             l2g_r1 = self.l2g_r_mat
#             l2g_t1 = self.l2g_t
#             l2g_r2 = l2g_r_mat
#             l2g_t2 = l2g_t
#
#         """ get time_delta and l2g r/t infos """
#         """ update frame info for next frame"""
#         self.timestamp = timestamp
#         self.l2g_t = l2g_t
#         self.l2g_r_mat = l2g_r_mat
#
#         """ predict and update """
#         prev_bev = self.prev_bev
#         frame_res = self._inference_single(
#             points,
#             img,
#             img_metas,
#             track_instances,
#             prev_bev,
#             l2g_r1,
#             l2g_t1,
#             l2g_r2,
#             l2g_t2,
#             time_delta,
#         )
#
#         self.prev_bev = frame_res["bev_embed"]
#         track_instances = frame_res["track_instances"]
#         track_instances_fordet = frame_res["track_instances_fordet"]
#
#         self.test_track_instances = track_instances
#         results = [dict()]
#         get_keys = ["bev_embed", "bev_pos",
#                     "track_query_embeddings", "track_bbox_results",
#                     "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
#         if self.with_motion_head:
#             get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
#         results[0].update({k: frame_res[k] for k in get_keys})
#         results = self._det_instances2results(track_instances_fordet, results, img_metas)
#         return results
#
#     def _track_instances2results(self, track_instances, img_metas, with_mask=True):
#         bbox_dict = dict(
#             cls_scores=track_instances.pred_logits,
#             bbox_preds=track_instances.pred_boxes,
#             track_scores=track_instances.scores,
#             obj_idxes=track_instances.obj_idxes,
#         )
#         # bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask)[0]
#         bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
#         bboxes = bboxes_dict["bboxes"]
#         # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
#         bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
#         labels = bboxes_dict["labels"]
#         scores = bboxes_dict["scores"]
#         bbox_index = bboxes_dict["bbox_index"]
#
#         track_scores = bboxes_dict["track_scores"]
#         obj_idxes = bboxes_dict["obj_idxes"]
#         result_dict = dict(
#             boxes_3d=bboxes.to("cpu"),
#             scores_3d=scores.cpu(),
#             labels_3d=labels.cpu(),
#             track_scores=track_scores.cpu(),
#             bbox_index=bbox_index.cpu(),
#             track_ids=obj_idxes.cpu(),
#             mask=bboxes_dict["mask"].cpu(),
#             track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
#         )
#         return result_dict
#
#     def _det_instances2results(self, instances, results, img_metas):
#         """
#         Outs:
#         active_instances. keys:
#         - 'pred_logits':
#         - 'pred_boxes': normalized bboxes
#         - 'scores'
#         - 'obj_idxes'
#         out_dict. keys:
#             - boxes_3d (torch.Tensor): 3D boxes.
#             - scores (torch.Tensor): Prediction scores.
#             - labels_3d (torch.Tensor): Box labels.
#             - attrs_3d (torch.Tensor, optional): Box attributes.
#             - track_ids
#             - tracking_score
#         """
#         # filter out sleep querys
#         if instances.pred_logits.numel() == 0:
#             return [None]
#         bbox_dict = dict(
#             cls_scores=instances.pred_logits,
#             bbox_preds=instances.pred_boxes,
#             track_scores=instances.scores,
#             obj_idxes=instances.obj_idxes,
#         )
#         bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
#         bboxes = bboxes_dict["bboxes"]
#         bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
#         labels = bboxes_dict["labels"]
#         scores = bboxes_dict["scores"]
#
#         track_scores = bboxes_dict["track_scores"]
#         obj_idxes = bboxes_dict["obj_idxes"]
#         result_dict = results[0]
#         result_dict_det = dict(
#             boxes_3d_det=bboxes.to("cpu"),
#             scores_3d_det=scores.cpu(),
#             labels_3d_det=labels.cpu(),
#         )
#         if result_dict is not None:
#             result_dict.update(result_dict_det)
#         else:
#             result_dict = None
#
#         return [result_dict]
#

#-------------------------------TTSIM-------------------------------------------#

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import copy
import math
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from ..modules.builder_utils import InverseSigmoid
from ..dense_heads.track_head_plugin.modules import MemoryBank, QueryInteractionModule


class Instances:
    """Lightweight container mirroring detectron2 Instances for tracking."""
    def __init__(self, image_size):
        self._image_size = image_size
        self._fields = {}

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._fields[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"Instances has no field '{name}'")

    def __len__(self):
        for v in self._fields.values():
            if hasattr(v, '__len__'):
                return len(v)
        return 0

    def __getitem__(self, idx):
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, '__getitem__'):
                ret._fields[k] = v[idx]
            else:
                ret._fields[k] = v
        return ret

    @staticmethod
    def cat(instance_list):
        assert len(instance_list) > 0
        ret = Instances(instance_list[0]._image_size)
        all_keys = set()
        for inst in instance_list:
            all_keys.update(inst._fields.keys())
        for k in all_keys:
            vals = [inst._fields[k] for inst in instance_list if k in inst._fields]
            if isinstance(vals[0], SimTensor):
                ret._fields[k] = T.cat(vals, dim=0)
            elif isinstance(vals[0], np.ndarray):
                ret._fields[k] = np.concatenate(vals, axis=0)
            else:
                ret._fields[k] = vals[0]
        return ret


class RuntimeTrackerBase:
    """Runtime tracker for score-based query filtering (not part of ONNX graph)."""
    def __init__(self, score_thresh=0.2, filter_score_thresh=0.1, miss_tolerance=5):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance

    def update(self, track_instances, gt_instances):
        pass


class FusionADTrack(SimNN.Module):
    """UniAD tracking part — TTSim conversion.

    Original: projects/mmdet3d_plugin/fusionad/detectors/fusionad_track.py
    """

    pts_voxel_layer: SimNN.Module
    scene_token: str | None

    @property
    def with_motion_head(self) -> bool:
        return hasattr(self, 'motion_head') and self.motion_head is not None

    def __init__(
        self,
        name,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        qim_args=dict(
            qim_type="QIMBase",
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,

        freeze_img_modules=False,
        freeze_bev_encoder=False,
        queue_length=3,

        # TTSim-specific: sub-modules must be passed as constructed objects
        grid_mask_module=None,
        img_backbone_module=None,
        img_neck_module=None,
        pts_backbone_module=None,
        pts_bbox_head_module=None,
        bev_h=200,
        bev_w=200,
    ):
        super().__init__()
        self.name = name

        self.grid_mask_module = grid_mask_module

        self.use_grid_mask = use_grid_mask
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.freeze_img_modules = freeze_img_modules
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length

        self.img_backbone = img_backbone_module
        self.img_neck = img_neck_module
        self.pts_backbone = pts_backbone_module
        self.pts_bbox_head = pts_bbox_head_module
        self.with_img_neck = (img_neck_module is not None)

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.query_embedding = F.Embedding(
            f'{name}.query_embedding', tbl_size=self.num_query + 1, emb_dim=self.embed_dims * 2)

        self.reference_points = SimNN.Linear(
            f'{name}.reference_points', in_features=self.embed_dims, out_features=3)

        self.bbox_size_fc = SimNN.Linear(
            f'{name}.bbox_size_fc', in_features=self.embed_dims, out_features=3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.memory_bank = None
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )

        self.query_interact = QueryInteractionModule(
            f'{name}.query_interact',
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            update_query_pos=qim_args.get('update_query_pos', False),
        )

        self.bbox_coder_cfg = bbox_coder

        self.memory_bank = MemoryBank(
            f'{name}.memory_bank',
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )

        self.loss_cfg = loss_cfg

        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.freeze_bev_encoder = freeze_bev_encoder
        self.voxelize_reduce = True

        self.sigmoid_op = F.Sigmoid(f'{name}.sigmoid')
        self.inverse_sigmoid_op = InverseSigmoid(f'{name}.inverse_sigmoid')
        self.resize_op = F.Resize(f'{name}.upsample', scale_factor=2, mode='nearest',
                                  nearest_mode='floor',
                                  coordinate_transformation_mode='asymmetric')

        # Ops for velo_update
        self.velo_cat = F.ConcatX(f'{name}.velo_cat', axis=-1)
        self.velo_add = F.Add(f'{name}.velo_add')
        self.velo_sub = F.Sub(f'{name}.velo_sub')
        self.velo_mul = F.Mul(f'{name}.velo_mul')
        self.velo_div = F.Div(f'{name}.velo_div')
        self.velo_matmul1 = F.MatMul(f'{name}.velo_matmul1')
        self.velo_matmul2 = F.MatMul(f'{name}.velo_matmul2')

        super().link_op2module()

    def analytical_param_count(self, lvl=0):
        total = 0
        # Learnable sub-modules
        if self.img_backbone is not None:
            total += self.img_backbone.analytical_param_count()
        if self.img_neck is not None:
            total += self.img_neck.analytical_param_count()
        if self.pts_backbone is not None:
            total += self.pts_backbone.analytical_param_count()
        if self.pts_bbox_head is not None:
            total += self.pts_bbox_head.analytical_param_count()
        total += (self.num_query + 1) * (self.embed_dims * 2)  # query_embedding
        total += self.reference_points.analytical_param_count(0)
        total += self.bbox_size_fc.analytical_param_count(0)
        total += self.query_interact.analytical_param_count()
        if self.memory_bank is not None:
            total += self.memory_bank.analytical_param_count()
        return total


    def voxelize(self, points):
        """Voxelize point cloud inputs.

        Note: pts_voxel_layer is a runtime op; this method uses
        pre-constructed sub-modules. torch.no_grad / force_fp32 decorators
        are not applicable in ttsim.
        """
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            f, c, n = self.pts_voxel_layer(res)
            feats.append(f)
            pad_const = F._from_data(
                f'{self.name}.vox_pad_{k}',
                np.array([1, 0, 0, 0], dtype=np.int64), is_const=True)
            pad_op = F.Pad(f'{self.name}.vox_pad_op_{k}')
            coords.append(pad_op(c, pad_const))
            sizes.append(n)

        feats = T.cat(feats, dim=0)
        coords = T.cat(coords, dim=0)
        sizes = T.cat(sizes, dim=0)

        if self.voxelize_reduce:
            reduce_sum = F.ReduceSum(f'{self.name}.vox_reduce_sum', axis=1, keepdims=0)
            feats_sum = reduce_sum(feats)
            sizes_shape = F._from_data(
                f'{self.name}.vox_sizes_shape',
                np.array([-1, 1], dtype=np.int64), is_const=True)
            sizes_reshape_op = F.Reshape(f'{self.name}.vox_sizes_reshape')
            sizes_reshaped = sizes_reshape_op(sizes, sizes_shape)
            div_op = F.Div(f'{self.name}.vox_div')
            feats = div_op(feats_sum, sizes_reshaped)

        return feats, coords, sizes

    def extract_pts_feat(self, pts):
        feats, coords, sizes = self.voxelize(pts)
        batch_size = coords[-1, 0] + 1
        x = self.pts_backbone(feats, coords, batch_size)
        return x

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert len(img.shape) == 5
        B, N, C, H, W = img.shape
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask_module(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.shape
            if len_queue is not None:
                new_shape = F._from_data(
                    f'{self.name}.img_feat_shape_q',
                    np.array([B // len_queue, len_queue, N, c, h, w], dtype=np.int64), is_const=True)
                reshape_op = F.Reshape(f'{self.name}.img_feat_reshape_q')
                img_feat_reshaped = reshape_op(img_feat, new_shape)
            else:
                new_shape = F._from_data(
                    f'{self.name}.img_feat_shape',
                    np.array([B, N, c, h, w], dtype=np.int64), is_const=True)
                reshape_op = F.Reshape(f'{self.name}.img_feat_reshape')
                img_feat_reshaped = reshape_op(img_feat, new_shape)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def extract_feat(self, img, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, len_queue=len_queue)
        return img_feats

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries = self.num_query + 1
        dim = self.embed_dims * 2

        query_indices = F._from_data(
            f'{self.name}.query_indices',
            np.arange(num_queries, dtype=np.int64), is_const=True)
        query = self.query_embedding(query_indices)

        query_half = query[..., : dim // 2]
        track_instances.ref_pts = self.reference_points(query_half)

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        box_sizes = self.bbox_size_fc(query_half)

        pred_boxes_init = F._from_data(
            f'{self.name}.pred_boxes_init',
            np.zeros((num_queries, 10), dtype=np.float32))

        track_instances.query = query
        track_instances.box_sizes = box_sizes
        track_instances.pred_boxes = pred_boxes_init

        track_instances.output_embedding = F._from_shape(
            f'{self.name}.output_embedding', [num_queries, dim // 2])

        track_instances.obj_idxes = F._from_data(
            f'{self.name}.obj_idxes',
            np.full((num_queries,), -1, dtype=np.int64))

        track_instances.matched_gt_idxes = F._from_data(
            f'{self.name}.matched_gt_idxes',
            np.full((num_queries,), -1, dtype=np.int64))

        track_instances.disappear_time = F._from_data(
            f'{self.name}.disappear_time',
            np.zeros((num_queries,), dtype=np.int64))

        track_instances.iou = F._from_shape(
            f'{self.name}.iou', [num_queries])

        track_instances.scores = F._from_shape(
            f'{self.name}.scores', [num_queries])

        track_instances.track_scores = F._from_shape(
            f'{self.name}.track_scores', [num_queries])

        track_instances.pred_logits = F._from_shape(
            f'{self.name}.pred_logits', [num_queries, self.num_classes])

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = F._from_shape(
            f'{self.name}.mem_bank', [num_queries, mem_bank_len, dim // 2])

        track_instances.mem_padding_mask = F._from_data(
            f'{self.name}.mem_padding_mask',
            np.ones((num_queries, mem_bank_len), dtype=np.bool_))

        track_instances.save_period = F._from_shape(
            f'{self.name}.save_period', [num_queries])

        return track_instances

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """
        Args:
            ref_pts (SimTensor): (num_query, 3).  in inverse sigmoid space
            velocity (SimTensor): (num_query, 2). m/s in lidar frame. vx, vy
        Outs:
            ref_pts (SimTensor): (num_query, 3).  in inverse sigmoid space
        """
        num_query = ref_pts.shape[0]

        velo_pad_ = F._from_shape(f'{self.name}.velo_pad_zeros', [num_query, 1])

        velo_pad = self.velo_cat(velocity, velo_pad_)

        reference_points = self.sigmoid_op(ref_pts)

        pc_range = self.pc_range
        scale_data = np.array(
            [pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]],
            dtype=np.float32)
        offset_data = np.array(
            [pc_range[0], pc_range[1], pc_range[2]], dtype=np.float32)
        scale_t = F._from_data(f'{self.name}.velo_scale', scale_data, is_const=True)
        offset_t = F._from_data(f'{self.name}.velo_offset', offset_data, is_const=True)
        reference_points = self.velo_add(
            self.velo_mul(reference_points, scale_t), offset_t)

        velo_td = self.velo_mul(velo_pad, time_delta)
        reference_points = self.velo_add(reference_points, velo_td)

        ref_pts = self.velo_matmul1(reference_points, l2g_r1)
        ref_pts = self.velo_add(ref_pts, l2g_t1)
        ref_pts = self.velo_sub(ref_pts, l2g_t2)

        g2l_r = l2g_r2  # runtime must provide the inverse

        ref_pts = self.velo_matmul2(ref_pts, g2l_r)

        ref_pts = self.velo_sub(ref_pts, offset_t)
        ref_pts = self.velo_div(ref_pts, scale_t)

        ref_pts = self.inverse_sigmoid_op(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        """Copy track instances for loss computation (runtime-only)."""
        track_instances = Instances((1, 1))
        num = len(tgt_instances)

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = F._from_shape(
            f'{self.name}.copy_scores', [num])

        track_instances.track_scores = F._from_shape(
            f'{self.name}.copy_track_scores', [num])

        track_instances.pred_boxes = F._from_shape(
            f'{self.name}.copy_pred_boxes', [num, 10])

        track_instances.iou = F._from_shape(
            f'{self.name}.copy_iou', [num])

        track_instances.pred_logits = F._from_shape(
            f'{self.name}.copy_pred_logits', [num, self.num_classes])

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances

    def get_history_bev(self, imgs_queue, img_metas_list, prev_points=None):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
        img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each_scale[:, i] for each_scale in img_feats_list]
            if prev_points is not None:
                pts_feats = self.extract_pts_feat([prev_points[i]])
            else:
                pts_feats = None
            prev_bev, _ = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats,
                img_metas=img_metas,
                prev_bev=prev_bev,
                pts_feats=pts_feats)


    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None, points=None, prev_points=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas, prev_points=prev_points)

        img_feats = self.extract_feat(img=imgs)
        if points is not None:
            pts_feats = self.extract_pts_feat(points)
        else:
            pts_feats = None

        bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
            mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev, pts_feats=pts_feats)

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            perm_op = F.Transpose(f'{self.name}.bev_perm_102', perm=[1, 0, 2])
            bev_embed = perm_op(bev_embed)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos, prev_bev

    def _forward_single(
        self,
        points,
        prev_points,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """
        Perform forward only on one frame. Called in forward_train
        Warning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        # NOTE: You can replace BEVFormer with other BEV encoder and generate bev_embed here
        bev_embed, bev_pos, _ = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas, points=points, prev_points=prev_points,
        )

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        output_past_trajs = det_output["all_past_traj_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }

        track_scores_input = output_classes[-1, 0, :]
        track_scores = self.sigmoid_op(track_scores_input)

        # Step-1 Update track instances with current prediction
        # [nb_dec, bs, num_query, xxx]
        nb_dec = output_classes.shape[0]

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 3]
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim // 2])

        track_instances_list.append(track_instances)

        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]  # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]    # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300, past_steps, 2]

            out["track_instances"] = track_instances
            all_query_embeddings.append(query_feats[i][0])
            # all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])

        # memory bank: extract fields, run temporal attn, write back
        if self.memory_bank is not None and len(track_instances) > 0:
            key_padding_mask = track_instances.mem_padding_mask
            valid_idxes = key_padding_mask[:, -1] == 0
            embed = track_instances.output_embedding[valid_idxes]
            if len(embed) > 0:
                prev_embed = track_instances.mem_bank[valid_idxes]
                kpm = key_padding_mask[valid_idxes]
                updated = self.memory_bank(embed, prev_embed, key_padding_mask=kpm)
                track_instances.output_embedding = track_instances.output_embedding.copy()
                track_instances.output_embedding[valid_idxes] = updated
        # Step-2 Update track instances using matcher

        # query_interact: extract active tracks, refine, merge
        active_tracks = track_instances[track_instances.obj_idxes >= 0]
        if len(active_tracks) > 0:
            updated_query = self.query_interact(
                active_tracks.query, active_tracks.output_embedding)
            active_tracks.query = updated_query
        init_tracks = self._generate_empty_tracks()
        out_track_instances = Instances.cat([init_tracks, active_tracks])
        out["track_instances"] = out_track_instances
        return out

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        """Runtime-only: select active track queries by index mask."""
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        """Runtime-only: select SDC (self-driving car) track query."""
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def forward_track_train(self,
                            points,
                            img,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp):
        """Forward function
        Args:
        Returns:
        """
        track_instances = self._generate_empty_tracks()
        num_frame = img.shape[1]
        # init gt instances!
        gt_instances_list = []

        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i]
            sd_boxes = gt_sdc_bbox[0][i]
            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]
            gt_instances.past_traj = gt_past_traj[0][i]
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i]
            gt_instances.sdc_boxes = sd_boxes
            gt_instances.sdc_labels = gt_sdc_label[0][i]
            gt_instances_list.append(gt_instances)

        out = dict()

        for i in range(num_frame):
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)

            if points is not None:
                prev_points = points[0][:i] if i != 0 else points[0][:1]
                points_single = [points[0][i]]
            else:
                prev_points = None
                points_single = None

            # TODO: Generate prev_bev in an RNN way.

            img_single = T.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]
            all_query_embeddings: list[object] = []
            all_matched_idxes: list[object] = []
            all_instances_pred_logits: list[object] = []
            all_instances_pred_boxes: list[object] = []
            frame_res = self._forward_single(
                points_single,
                prev_points,
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )
            # all_query_embeddings: len=dec nums, N*256
            # all_matched_idxes: len=dec nums, N*2
            track_instances = frame_res["track_instances"]
            if i == num_frame - 1:
                get_keys = ["bev_embed", "bev_pos",
                            "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                            "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
                out.update({k: frame_res[k] for k in get_keys if k in frame_res})
        losses: dict[str, object] = {}
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].shape[0] == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"]  # [10000, 1, 256]
            dim = bev_embed.shape[0]
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            shp1 = F._from_data(f'{self.name}.bev_up_shp1',
                np.array([h, w, bev_embed.shape[1], bev_embed.shape[2]], dtype=np.int64), is_const=True)
            bev_embed = F.Reshape(f'{self.name}.bev_up_reshape1')(bev_embed, shp1)
            bev_embed = F.Transpose(f'{self.name}.bev_up_perm1', perm=[2, 3, 0, 1])(bev_embed)

            bev_embed = self.resize_op(bev_embed)  # [1, 256, 200, 200]

            b_, c_, h2, w2 = bev_embed.shape
            bev_embed = F.Transpose(f'{self.name}.bev_up_perm2', perm=[2, 3, 0, 1])(bev_embed)
            shp2 = F._from_data(f'{self.name}.bev_up_shp2',
                np.array([h2 * w2, b_, c_], dtype=np.int64), is_const=True)
            bev_embed = F.Reshape(f'{self.name}.bev_up_reshape2')(bev_embed, shp2)
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if len(prev_bev.shape) == 3 and prev_bev.shape[0] != h * w:
                    #  [1, 10000, 256] → training layout
                    shp_t1 = F._from_data(f'{self.name}.prev_t_shp1',
                        np.array([prev_bev.shape[0], h, w, prev_bev.shape[2]], dtype=np.int64), is_const=True)
                    prev_bev = F.Reshape(f'{self.name}.prev_t_reshape1')(prev_bev, shp_t1)
                    prev_bev = F.Transpose(f'{self.name}.prev_t_perm1', perm=[0, 3, 1, 2])(prev_bev)
                    prev_bev = self.resize_op(prev_bev)
                    b_, c_, h2, w2 = prev_bev.shape
                    prev_bev = F.Transpose(f'{self.name}.prev_t_perm2', perm=[0, 2, 3, 1])(prev_bev)
                    shp_t2 = F._from_data(f'{self.name}.prev_t_shp2',
                        np.array([b_, h2 * w2, c_], dtype=np.int64), is_const=True)
                    prev_bev = F.Reshape(f'{self.name}.prev_t_reshape2')(prev_bev, shp_t2)
                else:
                    #  [10000, 1, 256] → inference layout
                    shp_i1 = F._from_data(f'{self.name}.prev_i_shp1',
                        np.array([h, w, prev_bev.shape[1], prev_bev.shape[2]], dtype=np.int64), is_const=True)
                    prev_bev = F.Reshape(f'{self.name}.prev_i_reshape1')(prev_bev, shp_i1)
                    prev_bev = F.Transpose(f'{self.name}.prev_i_perm1', perm=[2, 3, 0, 1])(prev_bev)
                    prev_bev = self.resize_op(prev_bev)
                    b_, c_, h2, w2 = prev_bev.shape
                    prev_bev = F.Transpose(f'{self.name}.prev_i_perm2', perm=[2, 3, 0, 1])(prev_bev)
                    shp_i2 = F._from_data(f'{self.name}.prev_i_shp2',
                        np.array([h2 * w2, b_, c_], dtype=np.int64), is_const=True)
                    prev_bev = F.Reshape(f'{self.name}.prev_i_reshape2')(prev_bev, shp_i2)
                outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = self.resize_op(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _inference_single(
        self,
        points,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim // 2])

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and generate bev_embed here
        bev_embed, bev_pos, _ = self.get_bevs(img, img_metas, prev_bev=prev_bev, points=points)
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores_input = output_classes[-1, 0, :]
        track_scores = self.sigmoid_op(track_scores_input)

        # each track will be assigned an unique global id by the track base.
        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]    # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]

        """ update track base """
        self.track_base.update(track_instances, None)

        """ update with memory_bank """
        if self.memory_bank is not None and len(track_instances) > 0:
            key_padding_mask = track_instances.mem_padding_mask
            valid_idxes = key_padding_mask[:, -1] == 0
            embed = track_instances.output_embedding[valid_idxes]
            if len(embed) > 0:
                prev_embed = track_instances.mem_bank[valid_idxes]
                kpm = key_padding_mask[valid_idxes]
                updated = self.memory_bank(embed, prev_embed, key_padding_mask=kpm)
                track_instances.output_embedding = track_instances.output_embedding.copy()
                track_instances.output_embedding[valid_idxes] = updated

        """  Update track instances using matcher """
        active_tracks = track_instances[track_instances.obj_idxes >= 0]
        if len(active_tracks) > 0:
            updated_query = self.query_interact(
                active_tracks.query, active_tracks.output_embedding)
            active_tracks.query = updated_query
        init_tracks = self._generate_empty_tracks()
        out_track_instances = Instances.cat([init_tracks, active_tracks])
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        points=None,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """only support bs=1 and sequential input"""

        bs = img.shape[0]

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None

        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t

        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._inference_single(
            points,
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances
        results: list[dict] = [dict()]
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_bbox_results",
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys if k in frame_res})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        """Runtime-only: decode track instances to result dicts."""
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = bbox_dict  # placeholder — runtime must decode

        bboxes = bboxes_dict.get("bboxes", bboxes_dict.get("bbox_preds"))
        labels = bboxes_dict.get("labels", None)
        scores = bboxes_dict.get("scores", None)
        bbox_index = bboxes_dict.get("bbox_index", None)
        track_scores = bboxes_dict.get("track_scores", None)
        obj_idxes = bboxes_dict.get("obj_idxes", None)

        result_dict = dict(
            boxes_3d=bboxes,
            scores_3d=scores,
            labels_3d=labels,
            track_scores=track_scores,
            bbox_index=bbox_index,
            track_ids=obj_idxes,
            mask=bboxes_dict.get("mask", None),
            track_bbox_results=[[bboxes, scores, labels, bbox_index, bboxes_dict.get("mask", None)]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Runtime-only: convert detection instances to result dicts.

        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d: 3D boxes.
            - scores: Prediction scores.
            - labels_3d: Box labels.
            - track_ids
            - tracking_score
        """
        if hasattr(instances, 'pred_logits') and len(instances.pred_logits.shape) > 0 and instances.pred_logits.shape[0] == 0:
            return [None]

        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = bbox_dict  # placeholder — runtime must decode

        bboxes = bboxes_dict.get("bboxes", bboxes_dict.get("bbox_preds"))
        labels = bboxes_dict.get("labels", None)
        scores = bboxes_dict.get("scores", None)
        track_scores = bboxes_dict.get("track_scores", None)
        obj_idxes = bboxes_dict.get("obj_idxes", None)

        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes,
            scores_3d_det=scores,
            labels_3d_det=labels,
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]
