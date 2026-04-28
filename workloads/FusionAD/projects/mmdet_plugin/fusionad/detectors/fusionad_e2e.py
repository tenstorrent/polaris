#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of FusionAD end-to-end model.

Top-level orchestrator that extends FusionADTrack with:
  - seg_head (PansegformerHead): panoptic segmentation
  - motion_head (MotionHead): trajectory prediction
  - occ_head (OccHead): occupancy prediction
  - planning_head (PlanningHeadSingleMode): ego planning

Inference pipeline:
  1. img_backbone -> img_neck (FPN) -> multi-scale features
  2. pts_backbone (SparseEncoderHD) -> LiDAR BEV features
  3. BEVFormerEncoder (6L) -> unified BEV embedding
  4. DetectionTransformerDecoder (6L) -> detection + tracking
  5. PansegformerHead -> panoptic segmentation masks
  6. MotionHead -> multi-modal trajectory forecasting
  7. OccHead -> future occupancy grids
  8. PlanningHeadSingleMode -> ego trajectory planning

"""

#----------------------------------PyTorch----------------------------------------#

# import torch
# from mmcv.runner import auto_fp16
# from mmdet.models import DETECTORS
# import copy
# import os
# from ..dense_heads.seg_head_plugin import IOU
# from .fusionad_track import FusionADTrack
# from mmdet.models.builder import build_head
#
# @DETECTORS.register_module()
# class FusionAD(FusionADTrack):
#     """
#     FusionAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
#     """
#     def __init__(
#         self,
#         seg_head=None,
#         motion_head=None,
#         occ_head=None,
#         planning_head=None,
#         task_loss_weight=dict(
#             track=1.0,
#             map=1.0,
#             motion=1.0,
#             occ=1.0,
#             planning=1.0
#         ),
#         freeze_track=True,
#         freeze_seg=True,
#         freeze_occ=False,
#         freeze_motion=True,
#         **kwargs,
#     ):
#         super(FusionAD, self).__init__(**kwargs)
#         if seg_head:
#             self.seg_head = build_head(seg_head)
#         if occ_head:
#             self.occ_head = build_head(occ_head)
#         if motion_head:
#             self.motion_head = build_head(motion_head)
#         if planning_head:
#             self.planning_head = build_head(planning_head)
#
#         self.freeze_track = freeze_track
#         self.freeze_seg = freeze_seg
#         self.freeze_occ = freeze_occ
#         self.freeze_motion = freeze_motion
#
#         self.task_loss_weight = task_loss_weight
#         assert set(task_loss_weight.keys()) == \
#                {'track', 'occ', 'motion', 'map', 'planning'}
#
#     @property
#     def with_planning_head(self):
#         return hasattr(self, 'planning_head') and self.planning_head is not None
#
#     @property
#     def with_occ_head(self):
#         return hasattr(self, 'occ_head') and self.occ_head is not None
#
#     @property
#     def with_motion_head(self):
#         return hasattr(self, 'motion_head') and self.motion_head is not None
#
#     @property
#     def with_seg_head(self):
#         return hasattr(self, 'seg_head') and self.seg_head is not None
#
#     def forward_dummy(self, img):
#         dummy_metas = None
#         return self.forward_test(img=img, img_metas=[[dummy_metas]])
#
#     def forward(self, return_loss=True, **kwargs):
#         """Calls either forward_train or forward_test depending on whether
#         return_loss=True.
#         Note this setting will change the expected inputs. When
#         `return_loss=True`, img and img_metas are single-nested (i.e.
#         torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
#         img_metas should be double nested (i.e.  list[torch.Tensor],
#         list[list[dict]]), with the outer list indicating test time
#         augmentations.
#         """
#         if return_loss:
#             return self.forward_train(**kwargs)
#         else:
#             return self.forward_test(**kwargs)
#
#
#     # Add the subtask loss to the whole model loss
#     @auto_fp16(apply_to=('img', 'points'))
#     def forward_train(self,
#                       points=None,
#                       img=None,
#                       img_metas=None,
#                       gt_bboxes_3d=None,
#                       gt_labels_3d=None,
#                       gt_inds=None,
#                       l2g_t=None,
#                       l2g_r_mat=None,
#                       timestamp=None,
#                       gt_lane_labels=None,
#                       gt_lane_bboxes=None,
#                       gt_lane_masks=None,
#                       gt_fut_traj=None,
#                       gt_fut_traj_mask=None,
#                       gt_past_traj=None,
#                       gt_past_traj_mask=None,
#                       gt_sdc_bbox=None,
#                       gt_sdc_label=None,
#                       gt_sdc_fut_traj=None,
#                       gt_sdc_fut_traj_mask=None,
#
#                       # Occ_gt
#                       gt_segmentation=None,
#                       gt_instance=None,
#                       gt_occ_img_is_valid=None,
#
#                       #planning
#                       sdc_planning=None,
#                       sdc_planning_mask=None,
#                       command=None,
#
#                       # fut gt for planning
#                       gt_future_boxes=None,
#                       **kwargs,  # [1, 9]
#                       ):
#         """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.
#
#             Args:
#             img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
#             img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
#             gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
#             gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
#             gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
#             l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
#             l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
#             timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
#             gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
#             gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
#             gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
#             gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
#             gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
#             gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
#             gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
#             gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
#             gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
#             gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
#             gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
#             gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
#             gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
#             gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
#             gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
#             sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
#             sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
#             command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
#             gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
#             gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
#
#             Returns:
#                 dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary
#                     is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
#         """
#         losses = dict()
#         len_queue = img.size(1)
#
#         if self.freeze_track:
#             with torch.no_grad():
#                 _, outs_track = self.forward_track_train(points, img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
#                                                             l2g_t, l2g_r_mat, img_metas, timestamp)
#         else:
#             losses_track, outs_track = self.forward_track_train(points, img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
#                                                             l2g_t, l2g_r_mat, img_metas, timestamp)
#             losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
#             losses.update(losses_track)
#
#         # Upsample bev for tiny version
#         outs_track = self.upsample_bev_if_tiny(outs_track)
#
#         bev_embed = outs_track["bev_embed"]
#         bev_pos  = outs_track["bev_pos"]
#
#         img_metas = [each[len_queue-1] for each in img_metas]
#
#         outs_seg = dict()
#         if self.with_seg_head:
#             if self.freeze_seg:
#                 with torch.no_grad():
#                     _, outs_seg = self.seg_head.forward_train(bev_embed, img_metas, gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
#
#             else:
#                 losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
#                                                             gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
#
#                 losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
#                 losses.update(losses_seg)
#
#         outs_motion = dict()
#         # Forward Motion Head
#         if self.with_motion_head:
#             if self.freeze_motion:
#                 with torch.no_grad():
#                     ret_dict_motion = self.motion_head.forward_train(bev_embed,
#                                                             gt_bboxes_3d, gt_labels_3d,
#                                                             gt_fut_traj, gt_fut_traj_mask,
#                                                             gt_sdc_fut_traj, gt_sdc_fut_traj_mask,
#                                                             outs_track=outs_track, outs_seg=outs_seg
#                                                         )
#                     outs_motion = ret_dict_motion["outs_motion"]
#                     outs_motion['bev_pos'] = bev_pos
#
#             else:
#                 ret_dict_motion = self.motion_head.forward_train(bev_embed,
#                                                             gt_bboxes_3d, gt_labels_3d,
#                                                             gt_fut_traj, gt_fut_traj_mask,
#                                                             gt_sdc_fut_traj, gt_sdc_fut_traj_mask,
#                                                             outs_track=outs_track, outs_seg=outs_seg
#                                                         )
#                 losses_motion = ret_dict_motion["losses"]
#                 outs_motion = ret_dict_motion["outs_motion"]
#                 outs_motion['bev_pos'] = bev_pos
#                 losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
#                 losses.update(losses_motion)
#
#         # Forward Occ Head
#         if self.with_occ_head:
#             if outs_motion['track_query'].shape[1] == 0:
#                 outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
#                 outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
#                 outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
#                 outs_motion['all_matched_idxes'] = [[-1]]
#             if self.freeze_occ:
#                 pass
#             else:
#                 losses_occ = self.occ_head.forward_train(
#                                 bev_embed,
#                                 outs_motion,
#                                 gt_inds_list=gt_inds,
#                                 gt_segmentation=gt_segmentation,
#                                 gt_instance=gt_instance,
#                                 gt_img_is_valid=gt_occ_img_is_valid,
#                             )
#                 losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
#                 losses.update(losses_occ)
#
#
#         # Forward Plan Head
#         if self.with_planning_head:
#             ego_info = torch.from_numpy(img_metas[0]['can_bus'])
#             ego_info[:7] = 0
#             outs_planning = self.planning_head.forward_train(bev_embed, outs_motion,sdc_planning, sdc_planning_mask, ego_info, command, gt_future_boxes)
#             losses_planning = outs_planning['losses']
#             losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
#             losses.update(losses_planning)
#
#         for k,v in losses.items():
#             losses[k] = torch.nan_to_num(v)
#         return losses
#
#     def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
#         loss_factor = self.task_loss_weight[prefix]
#         loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
#         return loss_dict
#
#     def forward_test(self,
#                      points=None,
#                      img=None,
#                      img_metas=None,
#                      l2g_t=None,
#                      l2g_r_mat=None,
#                      timestamp=None,
#                      gt_lane_labels=None,
#                      gt_lane_masks=None,
#                      rescale=False,
#                      # planning gt(for evaluation only)
#                      sdc_planning=None,
#                      sdc_planning_mask=None,
#                      command=None,
#
#                      # Occ_gt (for evaluation only)
#                      gt_segmentation=None,
#                      gt_instance=None,
#                      gt_occ_img_is_valid=None,
#                      **kwargs
#                     ):
#         """Test function
#         """
#         for var, name in [(img_metas, 'img_metas')]:
#             if not isinstance(var, list):
#                 raise TypeError('{} must be a list, but got {}'.format(
#                     name, type(var)))
#         img = [img] if img is None else img
#
#         if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
#             # the first sample of each scene is truncated
#             self.prev_frame_info['prev_bev'] = None
#         # update idx
#         self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']
#
#         # do not use temporal information
#         if not self.video_test_mode:
#             self.prev_frame_info['prev_bev'] = None
#
#         # Get the delta of ego position and angle between two timestamps.
#         tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
#         tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
#         # first frame
#         if self.prev_frame_info['scene_token'] is None:
#             img_metas[0][0]['can_bus'][:3] = 0
#             img_metas[0][0]['can_bus'][-1] = 0
#         # following frames
#         else:
#             img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
#             img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
#         self.prev_frame_info['prev_pos'] = tmp_pos
#         self.prev_frame_info['prev_angle'] = tmp_angle
#
#         img = img[0]
#         points = points[0]
#         img_metas = img_metas[0]
#         timestamp = timestamp[0] if timestamp is not None else None
#
#         result = [dict() for i in range(len(img_metas))]
#         result_track = self.simple_test_track(points, img, l2g_t, l2g_r_mat, img_metas, timestamp)
#
#         # Upsample bev for tiny model
#         result_track[0] = self.upsample_bev_if_tiny(result_track[0])
#
#         bev_embed = result_track[0]["bev_embed"]
#
#         if self.with_seg_head:
#             result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)
#
#         if self.with_motion_head:
#             result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
#             outs_motion['bev_pos'] = result_track[0]['bev_pos']
#
#         outs_occ = dict()
#         if self.with_occ_head:
#             occ_no_query = outs_motion['track_query'].shape[1] == 0
#             outs_occ = self.occ_head.forward_test(
#                 bev_embed,
#                 outs_motion,
#                 no_query = occ_no_query,
#                 gt_segmentation=gt_segmentation,
#                 gt_instance=gt_instance,
#                 gt_img_is_valid=gt_occ_img_is_valid,
#             )
#             result[0]['occ'] = outs_occ
#
#         if self.with_planning_head:
#             ego_info = torch.from_numpy(img_metas[0]['can_bus'])
#             ego_info[:7] = 0
#             planning_gt=dict(
#                 segmentation=gt_segmentation,
#                 sdc_planning=sdc_planning[0]['planning'],
#                 sdc_planning_mask=sdc_planning_mask,
#                 command=command
#             )
#             result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, ego_info,sdc_planning[0]['past_planning'],command)
#             result[0]['planning'] = dict(
#                 planning_gt=planning_gt,
#                 result_planning=result_planning,
#             )
#
#         if self.with_seg_head:
#             del result_seg[0]['args_tuple']
#
#         pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
#         result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)
#
#         if self.with_seg_head:
#             result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox'])
#         if self.with_motion_head:
#             result_motion[0] = pop_elem_in_result(result_motion[0])
#         if self.with_occ_head:
#             result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
#                 pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
#
#         for i, res in enumerate(result):
#             res['token'] = img_metas[i]['sample_idx']
#             res.update(result_track[i])
#             if self.with_motion_head:
#                 res.update(result_motion[i])
#             if self.with_seg_head:
#                 res.update(result_seg[i])
#
#         return result
#
#
# def pop_elem_in_result(task_result:dict, pop_list:list=None):
#     all_keys = list(task_result.keys())
#     for k in all_keys:
#         if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
#             task_result.pop(k)
#
#     if pop_list is not None:
#         for pop_k in pop_list:
#             task_result.pop(pop_k, None)
#     return task_result


#-------------------------------TTSIM-------------------------------------------#

import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import copy
import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.detectors.fusionad_track import FusionADTrack
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.panseg_head import PansegformerHead
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.motion_head import MotionHead
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.occ_head import OccHead
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.planning_head import PlanningHeadSingleMode
from workloads.FusionAD.projects.mmdet_plugin.fusionad.dense_heads.track_head import BEVFormerTrackHead
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.resnet import ResNetBackbone, Bottleneck
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.fpn import FPN
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.encoder import BEVFormerEncoder
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.decoder import DetectionTransformerDecoder
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.transformer import PerceptionTransformer
from workloads.FusionAD.projects.mmdet_plugin.models.backbones.sparse_encoder_hd import SparseEncoderHD


def pop_elem_in_result(task_result, pop_list=None):
    """Remove query/embedding keys and any explicitly listed keys from a result dict."""
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result


class FusionAD(FusionADTrack):
    """
    FusionAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting,
    Occupancy Prediction and Planning for Autonomous Driving.

    TTSim conversion of the full FusionAD end-to-end model.

    This is the top-level orchestrator that extends FusionADTrack with:
      - seg_head (PansegformerHead): panoptic segmentation
      - motion_head (MotionHead): trajectory prediction
      - occ_head (OccHead): occupancy prediction
      - planning_head (PlanningHeadSingleMode): ego planning

    The inference pipeline:
      1. simple_test_track  → bev_embed, bev_pos, track results
      2. upsample_bev_if_tiny → (upscale for tiny models)
      3. seg_head(bev_embed) → seg results
      4. motion_head(bev_embed, track_query, ...) → motion results
      5. occ_head(bev_embed, ins_query) → occ results
      6. planning_head(bev_embed, bev_pos, ...) → planning results

    Supports two calling conventions:
      1. Polaris mode:  FusionAD(name_str, cfg_dict)
      2. Direct mode:   FusionAD(name=..., seg_head_module=..., **kwargs)

    Original: projects/mmdet3d_plugin/fusionad/detectors/fusionad_e2e.py
    """

    def __init__(
        self,
        *args,
        # Sub-modules (direct mode only)
        seg_head_module=None,
        motion_head_module=None,
        occ_head_module=None,
        planning_head_module=None,
        **kwargs,
    ):
        # Detect polaris calling convention: FusionAD("instance_name", {cfg_dict})
        if len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            self._init_from_polaris(args[0], args[1])
        elif len(args) == 0:
            self._init_direct(
                seg_head_module=seg_head_module,
                motion_head_module=motion_head_module,
                occ_head_module=occ_head_module,
                planning_head_module=planning_head_module,
                **kwargs,
            )
        else:
            raise TypeError(
                f"FusionAD expects either (name: str, cfg: dict) or keyword args, "
                f"got {len(args)} positional args"
            )

    def _init_direct(self, seg_head_module=None, motion_head_module=None,
                     occ_head_module=None, planning_head_module=None, **kwargs):
        """Original construction: pre-built sub-modules passed as keyword args."""
        super().__init__(**kwargs)

        self.seg_head = seg_head_module
        self.occ_head = occ_head_module
        self.motion_head = motion_head_module
        self.planning_head = planning_head_module

        super().link_op2module()

    def _init_from_polaris(self, name, cfg):
        """Polaris construction: build everything from (name, cfg) dict."""
        # Extract params from merged cfg
        embed_dims = cfg.get('embed_dims', 256)
        num_query = cfg.get('num_query', 900)
        num_classes = cfg.get('num_classes', 10)
        num_cams = cfg.get('num_cams', 6)
        bev_h = cfg.get('bev_h', 200)
        bev_w = cfg.get('bev_w', 200)
        pc_range = cfg.get('pc_range', [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0])
        video_test_mode = cfg.get('video_test_mode', True)
        queue_length = cfg.get('queue_length', 3)
        predict_steps = cfg.get('predict_steps', 12)
        predict_modes = cfg.get('predict_modes', 6)
        planning_steps = cfg.get('planning_steps', 6)
        occ_n_future = cfg.get('occ_n_future', 4)

        # Store image dimensions for create_input_tensors
        self._img_height = cfg.get('img_height', 900)
        self._img_width = cfg.get('img_width', 1600)
        self._num_cams = num_cams
        self._bs = cfg.get('bs', 1)

        # LiDAR config
        self._use_lidar = cfg.get('use_lidar', True)
        self._lidar_in_channels = cfg.get('lidar_in_channels', 5)
        self._lidar_sparse_shape = cfg.get('lidar_sparse_shape', [41, 256, 256])

        # Build image backbone (ResNet-101)
        backbone_layers = cfg.get('backbone_layers', [3, 4, 23, 3])
        backbone_out_indices = cfg.get('backbone_out_indices', (1, 2, 3))
        img_backbone = ResNetBackbone(
            f'{name}.img_backbone',
            {
                'img_channels': 3,
                'layers': backbone_layers,
                'out_indices': backbone_out_indices,
                'stage_with_dcn': (False, False, True, True),
                'dcn_deform_groups': 1,
            },
        )

        # Build FPN neck
        # out_indices=(1,2,3) → channels [512, 1024, 2048]
        stage_channels = [64 * Bottleneck.expansion,   # stage0: 256
                          128 * Bottleneck.expansion,  # stage1: 512
                          256 * Bottleneck.expansion,  # stage2: 1024
                          512 * Bottleneck.expansion]  # stage3: 2048
        fpn_in_channels = [stage_channels[i] for i in backbone_out_indices]
        img_neck = FPN(
            f'{name}.img_neck',
            {
                'in_channels': fpn_in_channels,
                'out_channels': embed_dims,
                'num_outs': 4,
            },
        )

        # Build LiDAR backbone (SparseEncoderHD) if fusion mode
        pts_backbone = None
        if self._use_lidar:
            pts_backbone = SparseEncoderHD(
                f'{name}.pts_backbone',
                in_channels=self._lidar_in_channels,
                sparse_shape=self._lidar_sparse_shape,
                output_channels=embed_dims,
                order=('conv', 'norm', 'act'),
                encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
                encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (0, 1, 1)), (0, 0)),
                block_type='basicblock',
            )

        # Build BEV encoder (BEVFormerEncoder with deformable attention)
        # NOTE: created as local var; assigned to self AFTER super().__init__()
        num_bev_encoder_layers = cfg.get('num_bev_encoder_layers', 6)
        fpn_num_outs = cfg.get('fpn_num_outs', 4)

        if self._use_lidar:
            # Fusion mode: BEVFormerFusionLayer with pts_cross_attn
            layer_cfg = dict(
                type='BEVFormerFusionLayer',
                attn_cfgs=[
                    dict(type='PtsCrossAttention',
                         embed_dims=embed_dims, num_levels=1),
                    dict(type='SpatialCrossAttention',
                         pc_range=pc_range, embed_dims=embed_dims,
                         num_cams=num_cams,
                         deformable_attention=dict(
                             embed_dims=embed_dims, num_heads=8,
                             num_levels=fpn_num_outs, num_points=8)),
                    dict(type='TemporalSelfAttention',
                         embed_dims=embed_dims, num_levels=1),
                ],
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=embed_dims,
                    feedforward_channels=embed_dims * 2,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                operation_order=('pts_cross_attn', 'norm', 'cross_attn',
                                 'norm', 'self_attn', 'norm', 'ffn', 'norm'),
            )
        else:
            # Camera-only mode: BEVFormerLayer
            layer_cfg = dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(type='TemporalSelfAttention',
                         embed_dims=embed_dims, num_levels=1),
                    dict(type='SpatialCrossAttention',
                         pc_range=pc_range, embed_dims=embed_dims,
                         num_cams=num_cams,
                         deformable_attention=dict(
                             embed_dims=embed_dims, num_heads=8,
                             num_levels=fpn_num_outs, num_points=8)),
                ],
                feedforward_channels=embed_dims * 2,
                ffn_dropout=0.1,  # type: ignore[dict-item]
                operation_order=('self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'),
            )

        bev_encoder = BEVFormerEncoder(
            f'{name}.bev_enc',
            num_layers=num_bev_encoder_layers,
            layer_cfg=layer_cfg,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
        )

        # Build detection decoder (6-layer transformer decoder)
        num_decoder_layers = cfg.get('num_decoder_layers', 6)
        decoder_layer_cfg = dict(
            type='DetrTransformerDecoderLayer',
            attn_cfgs=[
                dict(type='MultiheadAttention',
                     embed_dims=embed_dims, num_heads=8, dropout=0.1),
                dict(type='CustomMSDeformableAttention',
                     embed_dims=embed_dims, num_levels=1),
            ],
            feedforward_channels=embed_dims * 2,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'cross_attn',
                             'norm', 'ffn', 'norm'),
        )
        decoder = DetectionTransformerDecoder(
            f'{name}.decoder',
            num_layers=num_decoder_layers,
            layer_cfg=decoder_layer_cfg,
            return_intermediate=True,
        )

        # Build PerceptionTransformer (wraps encoder + decoder)
        transformer = PerceptionTransformer(
            name=f'{name}.perception_transformer',
            encoder=bev_encoder,
            decoder=decoder,
            embed_dims=embed_dims,
            num_feature_levels=fpn_num_outs,
            num_cams=num_cams,
        )
        # Set synthetic embeddings (graph-only mode, no pretrained weights)
        transformer.level_embeds = np.random.randn(
            fpn_num_outs, embed_dims).astype(np.float32)
        transformer.cams_embeds = np.random.randn(
            num_cams, embed_dims).astype(np.float32)

        # Build BEVFormerTrackHead (6-layer decoder, cls/reg/traj branches)
        past_steps = cfg.get('past_steps', 4)
        fut_steps = cfg.get('fut_steps', 4)
        track_head = BEVFormerTrackHead(
            f'{name}.track_head',
            embed_dims=embed_dims,
            num_reg_fcs=2,
            cls_out_channels=num_classes,
            code_size=10,
            bev_h=bev_h,
            bev_w=bev_w,
            past_steps=past_steps,
            fut_steps=fut_steps,
            pc_range=pc_range,
            num_pred=num_decoder_layers,
            num_query=num_query,
            with_box_refine=True,
            transformer=transformer,
        )
        # Set synthetic embedding data (graph-only mode)
        track_head.bev_embedding_data = np.random.randn(
            bev_h * bev_w, embed_dims).astype(np.float32)
        track_head.query_embedding_data = np.random.randn(
            num_query, 2 * embed_dims).astype(np.float32)

        # Initialize parent (FusionADTrack)
        super().__init__(
            name=name,
            embed_dims=embed_dims,
            num_query=num_query,
            num_classes=num_classes,
            pc_range=pc_range,
            video_test_mode=video_test_mode,
            queue_length=queue_length,
            bev_h=bev_h,
            bev_w=bev_w,
            img_backbone_module=img_backbone,
            img_neck_module=img_neck,
            pts_backbone_module=pts_backbone,
            pts_bbox_head_module=track_head,
            score_thresh=0.4,
            filter_score_thresh=0.35,
            qim_args=dict(
                qim_type="QIMBase",
                merger_dropout=0,
                update_query_pos=True,
                fp_ratio=0.3,
                random_drop=0.1,
            ),
            mem_args=dict(
                memory_bank_type="MemoryBank",
                memory_bank_score_thresh=0.0,
                memory_bank_len=4,
            ),
        )

        # Build seg head (PansegformerHead)
        num_enc_layers = 6
        num_dec_layers = 6
        nhead = 8
        num_seg_levels = 4
        ffn_channels = embed_dims * 2  # _ffn_dim_ = 512
        seg_transformer_cfg = dict(
            embed_dims=embed_dims,
            num_feature_levels=num_seg_levels,
            encoder_cfg=dict(
                type='DetrTransformerEncoder',
                num_layers=num_enc_layers,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=embed_dims, num_heads=nhead,
                        num_levels=num_seg_levels),
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=embed_dims,
                        feedforward_channels=ffn_channels, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder_cfg=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=num_dec_layers,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(type='MultiheadAttention',
                             embed_dims=embed_dims, num_heads=nhead, dropout=0.1),
                        dict(type='MultiScaleDeformableAttention',
                             embed_dims=embed_dims, num_heads=nhead,
                             num_levels=num_seg_levels)],
                    ffn_cfgs=dict(
                        type='FFN', embed_dims=embed_dims,
                        feedforward_channels=ffn_channels, ffn_drop=0.1),
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm'))),
        )
        self.seg_head = PansegformerHead(
            f'{name}.seg_head',
            embed_dims=embed_dims,
            num_query=300,
            num_things_classes=3,
            num_stuff_classes=1,
            bev_h=bev_h,
            bev_w=bev_w,
            canvas_size=(bev_h, bev_w),
            num_decoder_layers=num_dec_layers,
            num_dec_things=4,
            num_dec_stuff=6,
            pos_encoding_num_feats=embed_dims // 2,
            transformer_cfg=seg_transformer_cfg,
        )

        # Build motion head (MotionHead)
        group_id_list = [[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]]
        transformerlayers_cfg = dict(
            attn_cfgs=[dict(
                type='MotionDeformableAttention',
                num_steps=predict_steps,
                embed_dims=embed_dims,
                num_levels=1,
                num_heads=8,
                num_points=4,
                sample_index=-1,
                bev_range=pc_range)],
            ffn_cfgs=dict(
                type='FFN', embed_dims=embed_dims,
                feedforward_channels=embed_dims * 2,
                num_fcs=2, ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
            embed_dims=embed_dims,
        )
        self.motion_head = MotionHead(
            f'{name}.motion_head',
            predict_steps=predict_steps,
            transformerlayers=transformerlayers_cfg,
            num_cls_fcs=3,
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dims=embed_dims,
            num_anchor=predict_modes,
            det_layer_num=6,
            group_id_list=group_id_list,
            pc_range=pc_range,
            num_classes=num_classes,
            num_layers=3,
        )
        # Set synthetic anchors (no pretrained weights in graph-only mode)
        num_groups = len(group_id_list)
        self.motion_head.kmeans_anchors = np.random.randn(
            num_groups, predict_modes, predict_steps, 2).astype(np.float32)
        self.motion_head.learnable_motion_query_embedding_data = np.random.randn(
            predict_modes * num_groups, embed_dims).astype(np.float32)

        # Build occ head (OccHead)
        occflow_grid_conf = {
            'xbound': [-50.0, 50.0, 0.5],
            'ybound': [-50.0, 50.0, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.occ_head = OccHead(
            f'{name}.occ_head',
            n_future=occ_n_future,
            grid_conf=occflow_grid_conf,
            bev_size=(bev_h, bev_w),
            bev_emb_dim=embed_dims,
            bev_proj_dim=embed_dims,
            embed_dims=embed_dims,
        )

        # Build planning head (PlanningHeadSingleMode)
        self.planning_head = PlanningHeadSingleMode(
            name=f'{name}.planning_head',
            embed_dims=embed_dims,
            planning_steps=planning_steps,
            bev_h=bev_h,
            bev_w=bev_w,
        )

        super().link_op2module()

    # -----------------------------------------------------------------
    # Polaris interface methods
    # -----------------------------------------------------------------
    def create_input_tensors(self):
        """Create input tensors for polaris graph simulation.

        Inputs:
          - img: stacked multi-camera images [B*num_cams, 3, H, W]
          - voxels: dense voxel grid [B, C, D, H, W] (only when use_lidar=True)
        """
        self.input_tensors = {
            'img': F._from_shape(
                'img',
                [self._bs * self._num_cams, 3, self._img_height, self._img_width],
                is_param=False,
                np_dtype=np.float32,
            ),
        }
        if self._use_lidar:
            D_vox, H_vox, W_vox = self._lidar_sparse_shape
            self.input_tensors['voxels'] = F._from_shape(
                'voxels',
                [self._bs, self._lidar_in_channels, D_vox, H_vox, W_vox],
                is_param=False,
                np_dtype=np.float32,
            )
        for _, t in self.input_tensors.items():
            t.set_module(self)

    def get_forward_graph(self):
        """Return the forward computation graph for polaris analysis."""
        return super()._get_forward_graph(self.input_tensors)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------
    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def __call__(
        self,
        points=None,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        rescale=False,
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        **kwargs,
    ):
        """
        End-to-end inference for one frame.

        When called with no arguments (polaris mode), uses self.input_tensors
        and synthetic data to build the computation graph.

        Pipeline:
          1. Track: simple_test_track → bev_embed, detection/tracking results
          2. Seg:   seg_head(bev_embed) → seg results
          3. Motion: motion_head(bev_embed, track_outs, seg_outs) → motion results
          4. Occ:   occ_head(bev_embed, motion_outs) → occupancy results
          5. Plan:  planning_head(bev_embed, motion_outs, occ_outs, ...) → planning

        All sub-heads are optional (skipped when None).
        """
        # ---- Polaris graph-construction mode (no args) ----
        if img_metas is None and hasattr(self, 'input_tensors'):
            return self._forward_polaris()

        for var, var_name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{var_name} must be a list, but got {type(var)}')
        img = [img] if img is None else img

        # Scene token management
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Compute ego-motion delta (can_bus)
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        points = points[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result: list[dict] = [dict() for _ in range(len(img_metas))]

        # --- Stage 1: Tracking ---
        result_track = self.simple_test_track(
            points, img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample BEV for tiny model
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]

        # --- Stage 2: Segmentation ---
        result_seg = None
        if self.with_seg_head:
            result_seg = self.seg_head(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        # --- Stage 3: Motion ---
        result_motion = None
        outs_motion = dict()
        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head(
                bev_embed,
                outs_track=result_track[0],
                outs_seg=result_seg[0] if result_seg else {})
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        # --- Stage 4: Occupancy ---
        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion.get('track_query') is not None and \
                           hasattr(outs_motion['track_query'], 'shape') and \
                           outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ

        # --- Stage 5: Planning ---
        if self.with_planning_head:
            ego_info_np = img_metas[0]['can_bus'].copy().astype(np.float32)
            ego_info_np[:7] = 0
            ego_info = F._from_data(
                f'{self.name}.ego_info', ego_info_np, is_const=True)

            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning[0]['planning'],
                sdc_planning_mask=sdc_planning_mask,
                command=command,
            )
            result_planning = self.planning_head(
                bev_embed, outs_motion, outs_occ, ego_info,
                sdc_planning[0]['past_planning'], command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        # --- Post-processing: clean up intermediate keys ---
        if self.with_seg_head and result_seg is not None:
            result_seg[0].pop('args_tuple', None)

        pop_track_list = [
            'prev_bev', 'bev_pos', 'bev_embed',
            'track_query_embeddings', 'sdc_embedding',
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head and result_seg is not None:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox'])
        if self.with_motion_head and result_motion is not None:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(
                result[0].get('occ', {}),
                pop_list=[
                    'seg_out_mask', 'flow_out', 'future_states_occ',
                    'pred_ins_masks', 'pred_raw_occ',
                    'pred_ins_logits', 'pred_ins_sigmoid',
                ])

        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head and result_motion is not None:
                res.update(result_motion[i])
            if self.with_seg_head and result_seg is not None:
                res.update(result_seg[i])

        return result

    # -----------------------------------------------------------------
    # Polaris graph-construction forward
    # -----------------------------------------------------------------
    def _forward_polaris(self):
        """
        Simplified forward pass for polaris graph analysis.

        Called when __call__ is invoked with no arguments (polaris mode).
        Full pipeline: img_backbone → FPN → PerceptionTransformer(encoder) →
                       BEVFormerTrackHead(decoder + cls/reg/traj) → task heads.
        """
        n = self.name
        img = self.input_tensors['img']  # [B*num_cams, 3, H, W]
        bs = self._bs
        nc = self._num_cams
        D = self.embed_dims
        bev_h, bev_w = self.bev_h, self.bev_w

        # --- Image Backbone: [B*nc, 3, H, W] → multi-scale features ---
        img_feats = self.img_backbone(img)

        # --- FPN Neck: multi-scale → uniform 256-ch features ---
        fpn_feats = self.img_neck(img_feats)

        # --- LiDAR Backbone: [B, C, D, H, W] → [B, D_out, Hp, Wp] ---
        pts_feats = None
        if self._use_lidar and 'voxels' in self.input_tensors:
            voxels = self.input_tensors['voxels']
            pts_feats = self.pts_backbone(voxels)
            # pts_feats: [B, embed_dims, Hp, Wp]

        # --- Reshape FPN features for PerceptionTransformer ---
        # FPN outputs: list of [B*nc, D, H_l, W_l]
        # Transformer expects: list of [bs, nc, D, H_l, W_l]
        mlvl_feats = []
        for lvl, feat in enumerate(fpn_feats):
            _, D_feat, fh, fw = feat.shape
            feat_reshaped = feat.reshape(bs, nc, D_feat, fh, fw)
            mlvl_feats.append(feat_reshaped)

        # BEV positional encoding: [bs, D, bev_h, bev_w] wrapped as SimTensor
        bev_pos_t = F._from_data(
            f'{n}.bev_pos_enc',
            np.zeros((bs, D, bev_h, bev_w), dtype=np.float32),
            is_param=True)
        setattr(self, bev_pos_t.name, bev_pos_t)

        # --- BEV Encoder via TrackHead → PerceptionTransformer → Encoder ---
        bev_embed, _ = self.pts_bbox_head.get_bev_features(
            mlvl_feats, bev_pos_t, prev_bev=None,
            img_metas=None, pts_feats=pts_feats)

        # Encoder returns [bs, nq, D]; transpose to [nq, bs, D] for decoder/heads
        bev_embed = bev_embed.transpose(0, 1)
        if bev_embed.data is None:
            bev_embed.data = np.zeros(bev_embed.shape, dtype=np.float32)
        setattr(self, bev_embed.name, bev_embed)

        # --- Detection Decoder via TrackHead (6-layer decoder + cls/reg/traj) ---
        # Object query embeddings: [num_query, 2*embed_dims]
        query_embeds_np = self.pts_bbox_head.query_embedding_data
        detection_results = self.pts_bbox_head.get_detections(
            bev_embed, object_query_embeds=query_embeds_np,
            ref_points=None, img_metas=None)

        # --- Seg head ---
        seg_result = None
        if self.with_seg_head:
            seg_result = self.seg_head(bev_embed)

        # --- Motion head ---
        motion_outs = {}
        if self.with_motion_head:
            num_agents = 10  # synthetic agent count
            num_dec_layers = 6
            predict_modes = self.motion_head.num_anchor

            # Synthetic track queries: (B, num_dec, A, D)
            track_query = F._from_data(
                f'{n}.syn_track_query',
                np.random.randn(bs, num_dec_layers, num_agents, D).astype(np.float32))
            setattr(self, track_query.name, track_query)

            # Synthetic lane queries from seg head
            num_lane_queries = 300
            lane_query = F._from_data(
                f'{n}.syn_lane_query',
                np.random.randn(bs, num_lane_queries, D).astype(np.float32))
            lane_query_pos = F._from_data(
                f'{n}.syn_lane_query_pos',
                np.random.randn(bs, num_lane_queries, D).astype(np.float32))
            setattr(self, lane_query.name, lane_query)
            setattr(self, lane_query_pos.name, lane_query_pos)

            # Synthetic bbox results (numpy, for coordinate transforms)
            # Each element: (bboxes, scores, labels, bbox_index, mask)
            # bboxes: (A, 9) — [cx, cy, cz, w, l, h, rot_sin, rot_cos, vel]
            pc = self.pc_range
            syn_bboxes = np.zeros((num_agents, 9), dtype=np.float32)
            syn_bboxes[:, 0] = np.linspace(pc[0] + 1, pc[3] - 1, num_agents)  # cx
            syn_bboxes[:, 1] = np.linspace(pc[1] + 1, pc[4] - 1, num_agents)  # cy
            syn_scores = np.ones(num_agents, dtype=np.float32)
            syn_labels = np.zeros(num_agents, dtype=np.int64)
            syn_bbox_index = np.arange(num_agents, dtype=np.int64)
            syn_mask = np.ones(num_agents, dtype=bool)
            track_bbox_results = [(syn_bboxes, syn_scores, syn_labels,
                                   syn_bbox_index, syn_mask)]

            motion_outs = self.motion_head(
                bev_embed, track_query, lane_query, lane_query_pos,
                track_bbox_results)

        # --- Occ head ---
        if self.with_occ_head:
            # Instance query: (B, Q, D) — from motion head or synthetic
            num_ins_queries = 10
            ins_query = F._from_data(
                f'{n}.syn_ins_query',
                np.random.randn(bs, num_ins_queries, D).astype(np.float32))
            setattr(self, ins_query.name, ins_query)

            occ_result = self.occ_head(bev_embed, ins_query)

        # --- Planning head ---
        if self.with_planning_head:
            predict_modes = getattr(self.motion_head, 'num_anchor', 6) if self.with_motion_head else 6
            num_dec_layers_motion = 3

            # BEV positional encoding: [bs, D, bev_h, bev_w]
            bev_pos_np = np.zeros(
                (bs, D, bev_h, bev_w), dtype=np.float32)
            # SDC trajectory query: [num_dec, bs, P, D]
            sdc_traj_query_np = np.random.randn(
                num_dec_layers_motion, bs, predict_modes, D).astype(np.float32)
            # SDC track query: [bs, D]
            sdc_track_query_np = np.random.randn(bs, D).astype(np.float32)
            # Ego info: [bs, 18]
            ego_info_np = np.zeros((bs, 18), dtype=np.float32)
            # Past planning: [bs, 18]
            past_planning_np = np.zeros((bs, 18), dtype=np.float32)
            command = 2  # straight

            # Set required weights if not already set
            if self.planning_head.navi_embed_weight is None:
                self.planning_head.navi_embed_weight = np.random.randn(
                    3, D).astype(np.float32)
            if self.planning_head.pos_embed_weight is None:
                self.planning_head.pos_embed_weight = np.random.randn(
                    1, D).astype(np.float32)

            plan_result = self.planning_head.forward(
                bev_embed, bev_pos_np,
                sdc_traj_query_np, sdc_track_query_np,
                ego_info_np, past_planning_np, command)

        return {}

    # -----------------------------------------------------------------
    # Analytical parameter count
    # -----------------------------------------------------------------
    def analytical_param_count(self, lvl=0):
        """Total learnable parameters in FusionAD (all sub-modules)."""
        indent = "  " * lvl
        total = 0

        # Parent (FusionADTrack) params
        track_params = super().analytical_param_count(lvl=max(0, lvl - 1))
        total += track_params
        if lvl >= 2:
            logger.debug(f"{indent}  FusionADTrack: {track_params:,}")

        # Seg head
        if self.with_seg_head and hasattr(self.seg_head, 'analytical_param_count'):
            sh = self.seg_head.analytical_param_count()
            total += sh
            if lvl >= 2:
                logger.debug(f"{indent}  seg_head: {sh:,}")

        # Motion head
        if self.with_motion_head and hasattr(self.motion_head, 'analytical_param_count'):
            mh = self.motion_head.analytical_param_count()
            total += mh
            if lvl >= 2:
                logger.debug(f"{indent}  motion_head: {mh:,}")

        # Occ head
        if self.with_occ_head and hasattr(self.occ_head, 'analytical_param_count'):
            oh = self.occ_head.analytical_param_count()
            total += oh
            if lvl >= 2:
                logger.debug(f"{indent}  occ_head: {oh:,}")

        # Planning head
        if self.with_planning_head and hasattr(self.planning_head, 'analytical_param_count'):
            ph = self.planning_head.analytical_param_count()
            total += ph
            if lvl >= 2:
                logger.debug(f"{indent}  planning_head: {ph:,}")

        if lvl >= 1:
            logger.debug(f"{indent}Total FusionAD params: {total:,}")

        return total
