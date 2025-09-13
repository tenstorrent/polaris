#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import copy
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml
from ttsim.ops import SimTensor

from workloads.bevformer.layers.backbones.BEVFormerBackbone import BEVFormerBackbone
from workloads.bevformer.layers.transformer.BEVFormerTransformer import BEVFormerTransformer
from workloads.bevformer.layers.heads.BEVFormerHead import BEVFormerHead

class BEVFormer(SimNN.Module):
    """Main BEVFormer model for camera-based 3D object detection"""

    def __init__(self, name, cfg):
        super(BEVFormer, self).__init__()
        self.name = name

        # Parse configuration
        self.backbone_conf_yaml = cfg['backbone_conf_yaml']
        self.transformer_conf_yaml = cfg['transformer_conf_yaml']
        self.head_conf_yaml = cfg['head_conf_yaml']

        # Model configuration
        self.bs = cfg.get('bs', 1)
        self.num_sweeps = cfg.get('num_sweeps', 2)
        self.num_cameras = cfg.get('num_cameras', 6)
        self.img_channels = cfg.get('img_channels', 3)
        self.img_height = cfg.get('img_height', 256)
        self.img_width = cfg.get('img_width', 704)
        self.training = cfg.get('training', False)

        # Temporal handling
        self.video_test_mode = cfg.get('video_test_mode', True)
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        # Load configurations
        self.backbone_conf = parse_yaml(self.backbone_conf_yaml)
        self.transformer_conf = parse_yaml(self.transformer_conf_yaml)
        self.head_conf = parse_yaml(self.head_conf_yaml)

        # Build model components
        self.backbone = BEVFormerBackbone(name + '.backbone', self.backbone_conf)
        self.neck = None  # FPN is included in backbone for BEVFormer
        self.pts_bbox_head = BEVFormerHead(name + '.head', **self.head_conf)
        # Align with BEVDepth naming for structural parity
        self.head = self.pts_bbox_head

        # Build transformer for head
        self.pts_bbox_head.transformer = BEVFormerTransformer(
            name + '.transformer', **self.transformer_conf
        )

        super().link_op2module()

    def set_batch_size(self, new_bs):
        """Update batch size"""
        self.bs = new_bs

    def create_input_tensors(self):
        """Create input tensor specifications"""
        EB = self.bs
        ES = self.num_sweeps
        EC = self.num_cameras
        EIC = self.img_channels
        EH = self.img_height
        EW = self.img_width

        self.input_tensors = {
            'sweep_imgs': F._from_shape('sweep_imgs', [EB, ES, EC, EIC, EH, EW]),
            'sensor2ego_mats': F._from_shape('sensor2ego_mats', [EB, ES, EC, 4, 4]),
            'intrin_mats': F._from_shape('intrin_mats', [EB, ES, EC, 4, 4]),
            'ida_mats': F._from_shape('ida_mats', [EB, ES, EC, 4, 4]),
            'sensor2sensor_mats': F._from_shape('sensor2sensor_mats', [EB, ES, EC, 4, 4]),
            'bda_mat': F._from_shape('bda_mat', [EB, 4, 4]),
        }

        # Add can_bus for ego-motion
        self.input_tensors['can_bus'] = F._from_shape('can_bus', [EB, 18])

        for _, t in self.input_tensors.items():
            t.is_param = False
            t.set_module(self)

        return self.input_tensors

    def extract_img_feat(self, img):
        """Extract image features using backbone"""
        return self.backbone(img)

    def forward_pts_train(self, img_feats, gt_bboxes_3d=None, gt_labels_3d=None,
                         img_metas=None, prev_bev=None):
        """Forward pass for training"""
        return self.pts_bbox_head(img_feats, img_metas, prev_bev)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain historical BEV features for temporal fusion"""
        prev_bev = None
        bs, len_queue, num_cams, C, H, W = imgs_queue.shape
        imgs_queue = F.Reshape('imgs_queue_reshape', imgs_queue,
                              (bs * len_queue, num_cams, C, H, W))

        # Extract features for all frames in queue
        img_feats_list = self.extract_img_feat(imgs_queue)

        # Process each frame sequentially to build temporal context
        for i in range(len_queue):
            img_metas = [each[i] for each in img_metas_list]
            img_feats = [each[:, i] for each in img_feats_list]

            # Get BEV features for this frame
            prev_bev = self.pts_bbox_head.transformer.get_bev_features(
                img_feats, self.pts_bbox_head.bev_embedding,
                self.pts_bbox_head.positional_encoding, prev_bev
            )
        return prev_bev

    def __call__(self, mode='inference', **kwargs):
        """
        Main forward pass

        Args:
            mode: 'inference' or 'training'
            **kwargs: Input tensors and metadata

        Returns:
            Model predictions
        """
        if mode == 'training':
            return self._forward_train(**kwargs)
        else:
            return self._forward_inference(**kwargs)

    def _forward_train(self, sweep_imgs=None, sensor2ego_mats=None, intrin_mats=None,
                      ida_mats=None, sensor2sensor_mats=None, bda_mat=None,
                      can_bus=None, img_metas=None, **kwargs):
        """Training forward pass"""
        # Extract image features
        img_feats = self.extract_img_feat(sweep_imgs)

        # Get temporal context (simplified for training)
        prev_bev = None
        if self.num_sweeps > 1:
            prev_bev = self.obtain_history_bev(sweep_imgs, img_metas)

        # Forward through detection head
        losses = self.forward_pts_train(img_feats, prev_bev=prev_bev)

        return losses

    def _forward_inference(self, sweep_imgs=None, sensor2ego_mats=None, intrin_mats=None,
                          ida_mats=None, sensor2sensor_mats=None, bda_mat=None,
                          can_bus=None, img_metas=None, **kwargs):
        """Inference forward pass"""
        # Auto-wire default inputs if none were provided (Polaris runner path)
        if sweep_imgs is None:
            if not hasattr(self, 'input_tensors') or self.input_tensors is None:
                self.create_input_tensors()
            sweep_imgs = self.input_tensors.get('sweep_imgs')
            sensor2ego_mats = self.input_tensors.get('sensor2ego_mats')
            intrin_mats = self.input_tensors.get('intrin_mats')
            ida_mats = self.input_tensors.get('ida_mats')
            sensor2sensor_mats = self.input_tensors.get('sensor2sensor_mats')
            bda_mat = self.input_tensors.get('bda_mat')
            can_bus = self.input_tensors.get('can_bus')
            img_metas = None

        # Extract image features
        img_feats = self.extract_img_feat(sweep_imgs)

        # Handle temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Process ego-motion (use img_metas only to avoid SimTensor len checks)
        if img_metas is not None and len(img_metas) > 0:
            self._update_ego_motion(img_metas=img_metas)

        # Forward through detection head
        preds = self.pts_bbox_head(img_feats, img_metas,
                                 prev_bev=self.prev_frame_info['prev_bev'])

        # When running under Polaris script, img_metas is None: return preds (graph build path)
        if img_metas is None:
            # Update temporal state if available
            if self.video_test_mode and preds.get('bev_embed', None) is not None:
                self.prev_frame_info['prev_bev'] = preds['bev_embed']
            return preds

        # Get bounding boxes for normal path
        bbox_list = self.pts_bbox_head.get_bboxes(preds, img_metas)

        # Update temporal state
        if self.video_test_mode and preds['bev_embed'] is not None:
            self.prev_frame_info['prev_bev'] = preds['bev_embed']

        return bbox_list

    def _update_ego_motion(self, img_metas):
        """Update ego-motion information for temporal tracking"""
        if img_metas is not None and len(img_metas) > 0:
            # Expect BEVFormer-style can_bus in img_metas[0]
            can_bus_vec = img_metas[0].get('can_bus', [0.0]*18)
            current_pos = can_bus_vec[:3]
            current_angle = can_bus_vec[-1]

            # Update previous frame info
            self.prev_frame_info['prev_pos'] = current_pos
            self.prev_frame_info['prev_angle'] = current_angle

    def analytical_param_count(self):
        """Return analytical parameter count (simplified)"""
        return 0  # Would be implemented to count actual parameters

    def get_forward_graph(self):
        """Expose graph for Polaris runner like other workloads"""
        if not hasattr(self, 'input_tensors') or self.input_tensors is None:
            self.create_input_tensors()
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    # (Removed duplicate __call__ to satisfy mypy no-redef)

if __name__ == '__main__':
    # Example usage and configuration
    import numpy as np

    # BEVFormer configuration
    bevformer_cfg = {
        'bs': 1,
        'num_sweeps': 2,
        'num_cameras': 6,
        'img_channels': 3,
        'img_height': 256,
        'img_width': 704,
        'video_test_mode': True,
        'backbone_conf_yaml': 'config/bevformer_cfgs/bevformer_backbone.yaml',
        'transformer_conf_yaml': 'config/bevformer_cfgs/bevformer_transformer.yaml',
        'head_conf_yaml': 'config/bevformer_cfgs/bevformer_head.yaml'
    }

    # Create model
    bevformer = BEVFormer('bevformer', bevformer_cfg)
    bevformer.create_input_tensors()

    # Example forward pass
    outputs = bevformer(mode='inference')
    print("BEVFormer model created successfully!")
    print(f"Output structure: {type(outputs)}")
