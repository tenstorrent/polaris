#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import numpy as np
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.utils.common import parse_yaml

from workloads.bevformer.layers.backbones.ResNet import ResNet
from workloads.bevformer.layers.backbones.FPN import FPN

class BEVFormerBackbone(SimNN.Module):
    """BEVFormer backbone combining ResNet and FPN for multi-camera feature extraction"""

    def __init__(self, name, cfg):
        super(BEVFormerBackbone, self).__init__()
        self.name = name

        # Parse configuration
        self.img_backbone_conf = cfg['img_backbone_conf']
        self.img_neck_conf = cfg['img_neck_conf']

        # Build components
        self.img_backbone = self._build_img_backbone(name + '.img_backbone', self.img_backbone_conf)
        self.img_neck = self._build_img_neck(name + '.img_neck', self.img_neck_conf)

        super().link_op2module()

    def _build_img_backbone(self, name, cfg):
        """Build image backbone (ResNet)"""
        backbone_type = cfg.get('type', 'ResNet')
        if backbone_type == 'ResNet':
            return ResNet(name, cfg)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def _build_img_neck(self, name, cfg):
        """Build image neck (FPN)"""
        neck_type = cfg.get('type', 'FPN')
        if neck_type == 'FPN':
            return FPN(name, **cfg)
        else:
            raise ValueError(f"Unsupported neck type: {neck_type}")

    def __call__(self, img):
        """
        Forward pass through backbone

        Args:
            img: Input images [B, N, C, H, W]

        Returns:
            Multi-scale feature maps
        """
        # For Polaris, we assume the tensor shape based on the model configuration
        # The input tensor should have shape [B, S, N, C, H, W]
        input_shape = self._get_input_shape(img)
        B, S, N, C, H, W = input_shape  # [B, S, N, C, H, W]

        # If multiple sweeps are provided, select the last sweep along axis=1
        if S > 1:
            starts = F._from_data(f'{self.name}.slice_starts', np.array([S - 1], dtype=np.int64))
            ends   = F._from_data(f'{self.name}.slice_ends',   np.array([S], dtype=np.int64))
            axes   = F._from_data(f'{self.name}.slice_axes',   np.array([1], dtype=np.int64))
            steps  = F._from_data(f'{self.name}.slice_steps',  np.array([1], dtype=np.int64))
            img    = F.SliceF(f'{self.name}.select_last_sweep')(img, starts, ends, axes, steps)
            # Recompute shape after slice
            input_shape = self._get_input_shape(img)
            B, S, N, C, H, W = input_shape

        # For now, we handle single-sweep in inference. Training path can add queue handling.
        assert S == 1, f"Expected num_sweeps==1 for inference path, got {S}"

        # Reshape for backbone processing [B*N, C, H, W] using a registered op handle
        reshape_in = F.ReshapeFixed(f'{self.name}.reshape_input', [B * N, C, H, W])
        reshape_in.set_module(self)
        self.reshape_input_op: Any = reshape_in
        img_reshaped = self.reshape_input_op(img)

        # Extract features using backbone
        backbone_features = self.img_backbone(img_reshaped)

        # Process through neck (FPN)
        neck_features = self.img_neck(backbone_features)

        # Reshape back to [B, N, C', H', W'] for each feature level
        output_features = []
        for i, feat in enumerate(neck_features):
            # Infer output shape for each feature level
            BN, C_feat, H_feat, W_feat = self._get_output_shape(feat, i)
            # BN should equal B*N when S==1
            reshape_out = F.ReshapeFixed(f'{self.name}.reshape_output_{i}', [B, N, C_feat, H_feat, W_feat])
            reshape_out.set_module(self)
            setattr(self, f'reshape_output_op_{i}', reshape_out)
            feat_reshaped = getattr(self, f'reshape_output_op_{i}')(feat)
            output_features.append(feat_reshaped)

        return output_features

    def _get_input_shape(self, img_tensor):
        """
        Get input tensor shape for Polaris compatibility

        Since Polaris SimTensors may not have direct .shape access,
        we infer the shape from the model configuration.
        """
        # The expected input shape for BEVFormer backbone is [B, N, C, H, W]
        # We can infer this from the configuration or tensor metadata

        # Try to get shape from tensor if available
        if hasattr(img_tensor, 'shape'):
            return img_tensor.shape

        # Otherwise, infer from configuration
        # This is a simplified assumption - in practice you might need
        # to get this information from the tensor metadata or configuration
        batch_size = 1  # Default batch size
        num_sweeps = 2  # From model config
        num_cameras = 6  # From model config
        img_channels = 3  # RGB images
        img_height = 256  # From backbone config
        img_width = 704  # From backbone config

        return (batch_size, num_sweeps, num_cameras, img_channels, img_height, img_width)

    def _get_output_shape(self, feat_tensor, level_idx):
        """
        Get output tensor shape for a specific feature level

        Args:
            feat_tensor: Feature tensor from backbone
            level_idx: Feature level index (0, 1, 2, 3)

        Returns:
            Tuple of (BN, C, H, W) for the feature tensor
        """
        # Try to get shape from tensor if available
        if hasattr(feat_tensor, 'shape'):
            return feat_tensor.shape

        # Otherwise, infer from typical FPN output dimensions
        # These are typical output dimensions for ResNet + FPN
        batch_size = 1
        num_sweeps = 2
        num_cameras = 6

        # Feature dimensions for different levels
        level_configs = [
            (256, 64, 176),   # Level 0: 1/4 resolution
            (256, 32, 88),    # Level 1: 1/8 resolution
            (256, 16, 44),    # Level 2: 1/16 resolution
            (256, 8, 22),     # Level 3: 1/32 resolution
        ]

        if level_idx < len(level_configs):
            C, H, W = level_configs[level_idx]
        else:
            # Default for additional levels
            C, H, W = 256, 8, 22

        BN = batch_size * num_sweeps * num_cameras
        return (BN, C, H, W)
