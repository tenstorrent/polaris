#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from workloads.ttnn.vadv2.reference.utils import denormalize_2d_bbox, denormalize_2d_pts, denormalize_bbox


class MapNMSFreeCoder(SimNN.Module):
    def __init__(
        self, pc_range, voxel_size=None, post_center_range=None, max_num=100, score_threshold=None, num_classes=10
    ):
        super().__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        max_num = self.max_num

        cls_scores = ttnn.sigmoid(cls_scores)
        cls_scores = ttnn.reshape(cls_scores, (-1,))
        max_num_tensor = ttnn.Tensor(shape=[1], dtype=ttnn.int64, device=cls_scores.device, data=np.array([max_num], dtype=np.int64))
        scores, indexs = ttnn.topk(cls_scores, max_num_tensor)

        num_classes_tensor = ttnn.full(shape=list(indexs.shape), dtype=ttnn.int64, device=cls_scores.device, fill_value=self.num_classes, layout=ttnn.Layout.TILE_LAYOUT)
        labels = ttnn.sub(indexs, ttnn.div(indexs, num_classes_tensor))
        bbox_index = ttnn.div(indexs, num_classes_tensor)
        bbox_preds = ttnn.Tensor(shape=[bbox_index.shape[0], bbox_preds.shape[1]], dtype=ttnn.bfloat16, device=bbox_preds.device)
        pts_preds = ttnn.Tensor(shape=[bbox_index.shape[0], pts_preds.shape[1]], dtype=ttnn.bfloat16, device=pts_preds.device)

        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)  # num_q,num_p,2
        final_scores = scores
        final_preds = labels
        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            boxes3d = final_box_preds #[mask]
            scores = final_scores #[mask]
            pts = final_pts_preds #[mask]
            labels = final_preds #[mask]
            predictions_dict = {
                "map_bboxes": boxes3d,
                "map_scores": scores,
                "map_labels": labels,
                "map_pts": pts,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        device = preds_dicts["map_all_cls_scores"].device
        all_cls_scores = ttnn.Tensor(shape=list(preds_dicts["map_all_cls_scores"].shape[2:]), dtype=ttnn.bfloat16, device=device)
        all_bbox_preds = ttnn.Tensor(shape=list(preds_dicts["map_all_bbox_preds"].shape[2:]), dtype=ttnn.bfloat16, device=device)
        all_pts_preds = ttnn.Tensor(shape=list(preds_dicts["map_all_pts_preds"].shape[2:]), dtype=ttnn.bfloat16, device=device)
        predictions_list = []

        # assume batch size is 1
        predictions_list.append(self.decode_single(all_cls_scores, all_bbox_preds, all_pts_preds))
        return predictions_list


class CustomNMSFreeCoder(SimNN.Module):
    def __init__(
        self, pc_range, voxel_size=None, post_center_range=None, max_num=100, score_threshold=None, num_classes=10
    ):
        super().__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, traj_preds):
        max_num = self.max_num

        cls_scores = ttnn.sigmoid(cls_scores)
        cls_scores = ttnn.reshape(cls_scores, (-1,))
        max_num_tensor = ttnn.Tensor(shape=[1], dtype=ttnn.int64, device=cls_scores.device, data=np.array([max_num], dtype=np.int64))
        scores, indexs = ttnn.topk(cls_scores, max_num_tensor)
        num_classes_tensor = ttnn.full(shape=list(indexs.shape), dtype=ttnn.int64, device=cls_scores.device, fill_value=self.num_classes, layout=ttnn.Layout.TILE_LAYOUT)
        labels = ttnn.sub(indexs, ttnn.div(indexs, num_classes_tensor))
        bbox_index = ttnn.div(indexs, num_classes_tensor)
        bbox_preds = ttnn.Tensor(shape=[bbox_index.shape[0], bbox_preds.shape[1]], dtype=ttnn.bfloat16, device=bbox_preds.device)
        traj_preds = ttnn.Tensor(shape=[bbox_index.shape[0], traj_preds.shape[1]], dtype=ttnn.bfloat16, device=traj_preds.device)

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels
        final_traj_preds = traj_preds

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            # assume mask is all true
            boxes3d = final_box_preds #[mask]
            scores = final_scores #[mask]
            labels = final_preds #[mask]
            trajs = final_traj_preds #[mask]

            predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels, "trajs": trajs}

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only " "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = ttnn.Tensor(shape=list(preds_dicts["all_cls_scores"].shape[2:]), dtype=ttnn.bfloat16, device=preds_dicts["all_cls_scores"].device)
        all_bbox_preds = ttnn.Tensor(shape=list(preds_dicts["all_bbox_preds"].shape[2:]), dtype=ttnn.bfloat16, device=preds_dicts["all_bbox_preds"].device)
        all_traj_preds = ttnn.Tensor(shape=list(preds_dicts["all_traj_preds"].shape[2:]), dtype=ttnn.bfloat16, device=preds_dicts["all_traj_preds"].device)
        predictions_list = []
        predictions_list.append(self.decode_single(all_cls_scores, all_bbox_preds, all_traj_preds))    # assume batch size is 1
        return predictions_list
