#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of TransFusionDetector.

Original file: mmdet3d/models/detectors/transfusion.py

TransFusionDetector extends MVXTwoStageDetector with overrides specific
to the TransFusion pipeline:
  - extract_img_feat: identical to MVXTwoStageDetector
  - extract_pts_feat: uses dynamic voxelization (same graph as parent)
  - simple_test / aug_test: pass both pts_feats and img_feats to head

NOTE: This file models the **detector** orchestrator.  The individual
      sub-modules (TransFusionHead, TransformerDecoderLayer, etc.) are
      defined in ttsim_modules/transfusion_head.py.

In TTSim, TransFusionDetector adds no learnable parameters beyond the
sub-modules supplied to its parent MVXTwoStageDetector.  ``__call__``
takes BEV features and runs them through ``pts_bbox_head``.

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.mvx_two_stage import MVXTwoStageDetector


class TransFusionDetector(MVXTwoStageDetector):
    """
    TransFusion 3D detector.

    No additional learnable parameters beyond the pre-built sub-modules
    passed through MVXTwoStageDetector.

    ``__call__`` accepts BEV features and returns detection predictions
    from ``pts_bbox_head``.

    Args:
        name (str): Module prefix.
        **kwargs: Forwarded to MVXTwoStageDetector.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, bev_feat):
        """
        Forward pass through detection head.

        Args:
            bev_feat (SimTensor): [B, C, H, W] fused BEV feature.

        Returns:
            dict: Prediction dict from pts_bbox_head (or bev_feat if no head).
        """
        if hasattr(self, "pts_bbox_head"):
            return self.pts_bbox_head(bev_feat)
        return bev_feat
