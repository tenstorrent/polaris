#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))


import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor

from typing import Dict, List, Optional, Tuple, Union
import copy

# ref: https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/centerpoint_head.py

class TaskHead(SimNN.Module):
    def __init__(self, name, head_name, num_conv, in_channels, head_conv, classes, final_kernel):
        super().__init__()
        self.name      = name
        self.head_name = head_name
        conv_layers    = [
                F.Conv2d(name + '.shared_conv', in_channels, in_channels, kernel_size=3, padding=1),
                F.BatchNorm2d(name + '.shared_bn', in_channels)
                ]
        c_in = in_channels
        for i in range(num_conv - 1):
            _conv1 = F.Conv2d(name + f'._conv1_{i}', c_in, head_conv,
                              kernel_size=final_kernel,
                              stride=1,
                              padding=final_kernel // 2)
            _bn1   = F.BatchNorm2d(name + f'._bn1_{i}', head_conv)
            _conv2 = F.Conv2d(name + f'._conv2_{i}', head_conv, classes,
                              kernel_size=final_kernel,
                              stride=1,
                              padding=final_kernel // 2,
                              bias=True)
            conv_layers.append(_conv1)
            conv_layers.append(_bn1)
            conv_layers.append(_conv2)
            c_in = head_conv
        self.oplist = F.SimOpHandleList(conv_layers)
        super().link_op2module()
        return

    def __call__(self, x):
        return self.oplist(x)

class CenterHeadGroup(SimNN.Module):
    def __init__(self, name,
                 in_channels        = [128],
                 tasks              = None,
                 bbox_coder         = None,
                 common_heads       = dict(),
                 loss_cls           = dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
                 loss_bbox          = dict(type='mmdet.L1Loss', reduction='none', loss_weight=0.25),
                 separate_head      = dict(type='mmdet.SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel = 64,
                 num_heatmap_convs  = 2,
                 conv_cfg           = dict(type='Conv2d'),
                 norm_cfg           = dict(type='BN2d'),
                 bias               = 'auto',
                 norm_bbox          = True,
                 train_cfg          = None,
                 test_cfg           = None,
                 **kwargs):

        super().__init__()
        head_conv    = 64
        final_kernel = 3
        num_classes              = [len(t['class_names']) for t in tasks]
        self.name                = name
        self.class_names         = [t['class_names'] for t in tasks]
        self.train_cfg           = train_cfg
        self.test_cfg            = test_cfg
        self.in_channels         = in_channels
        self.num_classes         = num_classes
        self.norm_bbox           = norm_bbox
        self.loss_cls            = loss_cls
        self.loss_bbox           = loss_bbox
        self.bbox_coder          = bbox_coder
        self.num_anchor_per_locs = [n for n in num_classes]

        self.task_heads_tbl = {} #type: ignore

        for num_cls_count, num_cls in enumerate(num_classes):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            for head, (classes, num_conv) in common_heads.items():
                if head not in self.task_heads_tbl:
                    self.task_heads_tbl[head] = {}
                self.task_heads_tbl[head][num_cls_count] = \
                   TaskHead(name + f'.num_cls_{num_cls}_{num_cls_count}.head_{head}',
                            head, num_conv, share_conv_channel, head_conv, classes, final_kernel)

        for head,obj in self.task_heads_tbl.items():
            for cls_id, task in obj.items():
                self._submodules[task.name] = task
        super().link_op2module()

    def __call__(self, x: SimTensor):
        Y = {head: {} for head in self.task_heads_tbl} #type: ignore
        for head,obj in self.task_heads_tbl.items():
            for cls_id, task in obj.items():
                Y[head][cls_id] = task(x)
        return Y

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

    bev_backbone = dict(type='ResNet', in_channels=80, depth=18, num_stages=3,
                        strides=(1, 2, 2),
                        dilations=(1, 1, 1),
                        out_indices=[0, 1, 2],
                        norm_eval=False,
                        base_channels=160,)

    bev_neck = dict(type='SECONDFPN',
                    in_channels=[80, 160, 320, 640],
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

    #inputs
    X0 = F._from_shape('X0', [5, 64, 32, 32])
    X1 = F._from_shape('X1', [5, 64, 16, 16])
    X2 = F._from_shape('X2', [5, 64,  8,  8])
    X3 = F._from_shape('X3', [5, 64,  4,  4])
    X = [X0, X1, X2, X3]

    #th_cfg = dict(final_kernel= 3, in_channels= 64, classes=2, num_conv=2, head_conv=64)
    #for i,x in enumerate(X):
    #    th = TaskHead(f'task_head_{i}', 'ppp',  **th_cfg)
    #    y = th(x)
    #    print(y)
    #exit(0)

    for i,x in enumerate(X):
        ch = CenterHeadGroup('center_head', **head_conf)
        y = ch(x)
        #for h in y:
        #    print(">> HEAD=", h)
        #    for c in y[h]:
        #        print("    >> CLS=", c, "TENSOR=", y[h][c])
        #print('\n', '-'*80)
