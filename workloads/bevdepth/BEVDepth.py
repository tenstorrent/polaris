#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor

from workloads.bevdepth.layers.backbones.BaseLSSFPN import BaseLSSFPN
from workloads.bevdepth.layers.heads.BEVDepthHead import BEVDepthHead

class BaseBEVDepth(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name                  = name
        self.bs                    = cfg['bs']
        self.backbone_conf_yaml    = cfg['backbone_conf_yaml']
        self.head_conf_yaml        = cfg['head_conf_yaml']
        self.is_train_depth        = cfg.get('is_train_depth', False)
        self.bs                    = cfg.get('bs',                 1)
        self.num_sweeps            = cfg.get('num_sweeps',         2)
        self.num_cameras           = cfg.get('num_cameras',        6)
        self.img_channels          = cfg.get('img_channels',       3)
        self.img_height            = cfg.get('img_height',        64)
        self.img_width             = cfg.get('img_width',         64)
        self.training              = cfg.get('training',       False)
        self.backbone_conf         = parse_yaml(self.backbone_conf_yaml)
        self.head_conf             = parse_yaml(self.head_conf_yaml)

        self.backbone              = BaseLSSFPN(name + '.backbone', **self.backbone_conf)
        self.head                  = BEVDepthHead(name + '.head',   **self.head_conf)
        super().link_op2module()
        return

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        EB  = self.bs
        ENS = self.num_sweeps
        ENC = self.num_cameras
        EIC = self.img_channels
        EH  = self.img_height
        EW  = self.img_width
        self.input_tensors = {
                'sweep_imgs'         : F._from_shape('sweep_imgs',         [EB, ENS, ENC, EIC, EH, EW]),
                'sensor2ego_mats'    : F._from_shape('sensor2ego_mats',    [EB, ENS, ENC, 4, 4]),
                'intrin_mats'        : F._from_shape('intrin_mats',        [EB, ENS, ENC, 4, 4]),
                'ida_mats'           : F._from_shape('ida_mats',           [EB, ENS, ENC, 4, 4]),
                'sensor2sensor_mats' : F._from_shape('sensor2sensor_mats', [EB, ENS, ENC, 4, 4]),
                'bda_mat'            : F._from_shape('bda_mat',            [EB, 4, 4]),
                }
        for _,t in self.input_tensors.items():
            t.is_param = False
            t.set_module(self)
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self, lvl=0):
        return 0

    def __call__(self):
        sweep_imgs         = self.input_tensors['sweep_imgs']
        sensor2ego_mats    = self.input_tensors['sensor2ego_mats']
        intrin_mats        = self.input_tensors['intrin_mats']
        ida_mats           = self.input_tensors['ida_mats']
        sensor2sensor_mats = self.input_tensors['sensor2sensor_mats']
        bda_mat            = self.input_tensors['bda_mat']


        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(sweep_imgs,
                                          sensor2ego_mats,
                                          intrin_mats,
                                          ida_mats,
                                          sensor2sensor_mats,
                                          bda_mat)
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(sweep_imgs,
                              sensor2ego_mats,
                              intrin_mats,
                              ida_mats,
                              sensor2sensor_mats,
                              bda_mat)
            preds = self.head(x)
            return preds

if __name__ == '__main__':
    import numpy as np

    """
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
            'rot_lim'    : (-5.4, 5.4),
            'bot_pct_lim': (0.0, 0.0),
            'final_dim'  : final_dim,
            'H'          : img_height,
            'W'          : img_width,
            'rand_flip'  : True,
            'cams'       : cameras,
            'Ncams'      : num_cameras,
            }

    bda_aug_conf = {
            'rot_lim'      : (-22.5, 22.5),
            'scale_lim'    : (0.95, 1.05),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
            }

    bev_backbone = dict(
            type='ResNet',
            in_channels=80,
            depth=18, num_stages=3,
            strides=(1, 2, 2),
            dilations=(1, 1, 1),
            out_indices=[0, 1, 2],
            norm_eval=False,
            num_channels=160,
            )

    bev_neck = dict(
            type='SECONDFPN',
            #in_channels=[80, 160, 320, 640]
            in_channels=[256, 512, 1024, 2048], #because my BEVDepthHead.trunk (ResNet.stage1.chnls = 256)
            upsample_strides=[1, 2, 4, 8],
            out_channels=[64, 64, 64, 64]
            )

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
        'bev_neck_conf'    : bev_neck,
        'tasks'            : TASKS,
        'common_heads'     : common_heads,
        'bbox_coder'       : bbox_coder,
        'train_cfg'        : train_cfg,
        'test_cfg'         : test_cfg,
        'in_channels'      : 256,  # Equal to bev_neck output_channels.
        'loss_cls'         : dict(type='GaussianFocalLoss', reduction='mean'),
        'loss_bbox'        : dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        'gaussian_overlap' : 0.1,
        'min_radius'       : 2,
    }
    """

    img_height   = 32
    img_width    = 32
    img_channels = 3
    final_dim    = (256, 704)
    img_conf     = dict(img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True)
    cameras      = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    num_cameras  = len(cameras)

    bevdepth_model_cfg = {
            'bs'                  : 1,
            'num_sweeps'          : 2,
            'img_height'          : img_height,
            'img_width'           : img_width,
            'img_channels'        : img_channels,
            'is_train_depth'      : False,
            'training'            : False,
            'num_cameras'         : num_cameras,
            'head_conf_yaml'      : 'config/bevdepth_cfgs/bevdepth_head.yaml',
            'backbone_conf_yaml'  : 'config/bevdepth_cfgs/bevdepth_backbone.yaml'
            }


    bevdepth = BaseBEVDepth('bevdepth', bevdepth_model_cfg)
    bevdepth.create_input_tensors()
    bevdepth_out = bevdepth()

    print(bevdepth)
    print('\n', '-'*50)
    for i,heads in enumerate(bevdepth_out):
        print(f"FINAL OUTPUT[{i}]:")
        for j,(head,obj) in enumerate(heads.items()):
            print(f"  {head}:")
            for k, (X,Y) in enumerate(obj.items()):
                print(f"    {X}: {Y.name}, {Y.shape}")
    print('\n', '-'*50)
    #gg = bevdepth.get_forward_graph()
    #gg.graph2onnx('bevdepth.onnx', do_model_check=False)
