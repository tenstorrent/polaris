#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor

from workloads.bevdepth.layers.backbones.BaseLSSFPN import build_backbone, build_neck
from workloads.bevdepth.layers.heads.CenterHead import CenterHeadGroup


#class BEVDepthHead(CenterHead):
class BEVDepthHead(SimNN.Module):
    """Head for BevDepth.
    Args:
        in_channels(int)         : Number of channels after bev_neck.
        tasks(dict)              : Tasks for head.
        bbox_coder(dict)         : Config of bbox coder.
        common_heads(dict)       : Config of head for each task.
        loss_cls(dict)           : Config of classification loss.
        loss_bbox(dict)          : Config of regression loss.
        gaussian_overlap(float)  : Gaussian overlap used for `get_targets`.
        min_radius(int)          : Min radius used for `get_targets`.
        train_cfg(dict)          : Config used in the training process.
        test_cfg(dict)           : Config used in the test process.
        bev_backbone_conf(dict)  : Cnfig of bev_backbone.
        bev_neck_conf(dict)      : Cnfig of bev_neck.
    """
    def __init__(
        self, name,
        bev_backbone_conf,
        bev_neck_conf,
        in_channels=64, #256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
    ):
        #super(BEVDepthHead, self).__init__(in_channels=in_channels, tasks=tasks, bbox_coder=bbox_coder,
        #                                   common_heads=common_heads, loss_cls=loss_cls, loss_bbox=loss_bbox,
        #                                   separate_head=separate_head,)
        super().__init__()
        self.name             = name
        self.trunk            = build_backbone(name + '.trunk', bev_backbone_conf)
        self.neck             = build_neck(name + '.neck', bev_neck_conf)
        self.gaussian_overlap = gaussian_overlap
        self.min_radius       = min_radius
        self.train_cfg        = train_cfg
        self.test_cfg         = test_cfg
        self.in_channels      = in_channels
        self.tasks            = tasks
        self.bbox_coder       = bbox_coder
        self.common_heads     = common_heads
        self.loss_cls         = loss_cls
        self.loss_bbox        = loss_bbox
        self.separate_head    = separate_head
        super().link_op2module()

    def __call__(self, x):
        trunk_outs = self.trunk(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = []
        for xi,x in enumerate(fpn_output):
            center_head_group = CenterHeadGroup(self.name + f'.centerheadgrp_{xi}',
                                                in_channels   = self.in_channels,
                                                tasks         = self.tasks,
                                                bbox_coder    = self.bbox_coder,
                                                common_heads  = self.common_heads,
                                                loss_cls      = self.loss_cls,
                                                loss_bbox     = self.loss_bbox,
                                                separate_head = self.separate_head,
                                                )
            self._submodules[center_head_group.name] = center_head_group
            y = center_head_group(x)
            ret_values.append(y)
        return ret_values

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device='cuda')

            anno_box = gt_bboxes_3d.new_zeros(
                (max_objs, len(self.train_cfg['code_weights'])),
                dtype=torch.float32,
                device='cuda')

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device='cuda')
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device='cuda')

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                        x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device='cuda')
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device='cuda'),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = targets
        return_loss = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                                         min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict[0].keys():
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel']),
                    dim=1,
                )
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot']),
                    dim=1,
                )
            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)
            return_loss += loss_bbox
            return_loss += loss_heatmap
        return return_loss

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(batch_heatmap,
                                          batch_rots,
                                          batch_rotc,
                                          batch_hei,
                                          batch_dim,
                                          batch_vel,
                                          reg=batch_reg,
                                          task_id=task_id)
            assert self.test_cfg['nms_type'] in [
                'size_aware_circle', 'circle', 'rotate'
            ]
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size']),
                                        dtype=torch.long,
                                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            elif self.test_cfg['nms_type'] == 'size_aware_circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    boxes_2d = boxes3d[:, [0, 1, 3, 4, 6]]
                    boxes = torch.cat([boxes_2d, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        size_aware_circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['thresh_scale'][task_id],
                            post_max_size=self.test_cfg['post_max_size'],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

if __name__ == '__main__':
    import numpy as np

    H         = 900
    W         = 1600
    final_dim = (256, 704)
    img_conf  = dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True)

    backbone_conf = {
        'x_bound'  : [-51.2, 51.2, 0.8],
        'y_bound'  : [-51.2, 51.2, 0.8],
        'z_bound'  : [-5, 3, 8],
        'd_bound'  : [2.0, 58.0, 0.5],
        'final_dim': final_dim,
        'output_channels'  : 80,
        'downsample_factor': 16,
        'img_backbone_conf': dict(
            type='ResNet',
            depth=50,
            frozen_stages=0,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            ),
        'img_neck_conf': dict(
            type='SECONDFPN',
            in_channels=[256, 512, 1024, 2048],
            upsample_strides=[0.25, 0.5, 1, 2],
            out_channels=[128, 128, 128, 128],
            ),
        #'depth_net_conf': dict(in_channels=512, mid_channels=512)
        'depth_net_conf': dict(in_channels=128, mid_channels=512)
        }

    ida_aug_conf = {
        'resize_lim' : (0.386, 0.55),
        'final_dim'  : final_dim,
        'rot_lim'    : (-5.4, 5.4),
        'H'          : H,
        'W'          : W,
        'rand_flip'  : True,
        'bot_pct_lim': (0.0, 0.0),
        'cams'       : ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams'      : 6,
    }

    bda_aug_conf = {
        'rot_lim'      : (-22.5, 22.5),
        'scale_lim'    : (0.95, 1.05),
        'flip_dx_ratio': 0.5,
        'flip_dy_ratio': 0.5
    }

    bev_backbone = dict(type='ResNet',
                        in_channels=80,
                        depth=18,
                        num_stages=3,
                        strides=(1, 2, 2),
                        dilations=(1, 1, 1),
                        out_indices=[0, 1, 2],
                        norm_eval=False,
                        num_channels=160,)

    bev_neck = dict(type='SECONDFPN',
                    #in_channels=[80, 160, 320, 640],
                    in_channels=[256, 512, 1024, 2048],
                    upsample_strides=[1, 2, 4, 8],
                    out_channels=[64, 64, 64, 64])

    CLASSES = [
            'car',
            'truck',
            'construction_vehicle',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'traffic_cone',
            ]

    TASKS = [
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ]

    common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))

    bbox_coder = dict(
        type='CenterPointBBoxCoder',
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        code_size=9,
    )

    train_cfg = dict(
        point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
        grid_size=[512, 512, 1],
        voxel_size=[0.2, 0.2, 8],
        out_size_factor=4,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
    )

    test_cfg = dict(
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=[0.2, 0.2, 8],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
    )

    head_conf = {
        'bev_backbone_conf': bev_backbone,
        'bev_neck_conf': bev_neck,
        'tasks': TASKS,
        'common_heads': common_heads,
        'bbox_coder': bbox_coder,
        'train_cfg': train_cfg,
        'test_cfg': test_cfg,
        'in_channels': 64, #256,  # Equal to bev_neck output_channels.
        'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
        'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        'gaussian_overlap': 0.1,
        'min_radius': 2,
    }

    my_model = BEVDepthHead('xxx', **head_conf)
    X = F._from_shape('X', [5, 160, 128, 128])
    Y = my_model(X)
    for y in Y:
        print('YYY-->', y)
