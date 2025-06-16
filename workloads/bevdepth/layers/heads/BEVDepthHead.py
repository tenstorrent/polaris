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

class BEVDepthHead(SimNN.Module):
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
