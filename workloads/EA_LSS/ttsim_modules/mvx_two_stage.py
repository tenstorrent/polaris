#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of MVXTwoStageDetector.

Original file: mmdet3d/models/detectors/mvx_two_stage.py

MVXTwoStageDetector is a generic multi-modality 3D detector base class.
It builds and holds up to eight optional sub-modules:
    img_backbone, img_neck,
    pts_voxel_encoder, pts_backbone, pts_neck,
    pts_bbox_head

In TTSim this module accepts **pre-built** TTSim sub-module instances and
aggregates their ``analytical_param_count`` values.  ``__call__`` chains
sub-modules in a simple pts → img → detection-head order where available.

Concrete sub-classes (EALSS, EALSS_CAM) override ``__call__`` to implement
the full multi-modal inference graph.

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import _from_shape

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.base_3d_detector import Base3DDetector


class MVXTwoStageDetector(Base3DDetector):
    """
    Generic two-stage multi-modality 3D detector.

    Sub-modules are accepted as keyword arguments.  Any module not provided
    is simply absent and contributes 0 to the param count.

    Args:
        name (str): Module prefix.
        img_backbone: Camera image backbone (TTSim Module or None).
        img_neck: Camera image neck (TTSim Module or None).
        pts_voxel_encoder: Voxel feature encoder (TTSim Module or None).
        pts_backbone: LiDAR backbone (TTSim Module or None).
        pts_neck: LiDAR neck (TTSim Module or None).
        pts_bbox_head: Detection head (TTSim Module or None).
    """

    _CHILD_ATTRS = [
        "img_backbone", "img_neck",
        "pts_voxel_encoder", "pts_backbone", "pts_neck",
        "pts_bbox_head",
    ]

    def __init__(
        self,
        name: str,
        img_backbone=None,
        img_neck=None,
        pts_voxel_encoder=None,
        pts_backbone=None,
        pts_neck=None,
        pts_bbox_head=None,
    ):
        super().__init__(name)
        for attr, val in [
            ("img_backbone",      img_backbone),
            ("img_neck",          img_neck),
            ("pts_voxel_encoder", pts_voxel_encoder),
            ("pts_backbone",      pts_backbone),
            ("pts_neck",          pts_neck),
            ("pts_bbox_head",     pts_bbox_head),
        ]:
            if val is not None:
                setattr(self, attr, val)
        super().link_op2module()

    def __call__(self, x):
        """
        Simple passthrough that runs the detection head if present.

        Concrete sub-classes (EALSS, EALSS_CAM) fully override this.

        Args:
            x (SimTensor): BEV feature [B, C, H, W].

        Returns:
            SimTensor or dict: detection head output if available, else x.
        """
        if hasattr(self, "pts_bbox_head"):
            return self.pts_bbox_head(x)
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        p = 0
        for attr in self._CHILD_ATTRS:
            if hasattr(self, attr):
                p += getattr(self, attr).analytical_param_count(lvl + 1)
        return p
