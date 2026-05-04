#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim versions of MVXFasterRCNN and DynamicMVXFasterRCNN.

Original file: mmdet3d/models/detectors/mvx_faster_rcnn.py

Both classes are defined in the same source file and are thin wrappers
over MVXTwoStageDetector that add no new learnable parameters:

  MVXFasterRCNN          — static voxelization version
  DynamicMVXFasterRCNN   — dynamic voxelization version (same params)

In TTSim neither class introduces additional sub-modules.  ``__call__``
and ``analytical_param_count`` are inherited directly from
MVXTwoStageDetector.

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


class MVXFasterRCNN(MVXTwoStageDetector):
    """
    Multi-modality VoxelNet using Faster R-CNN (static voxelization).

    No additional learnable parameters beyond MVXTwoStageDetector.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)


class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """
    Multi-modality VoxelNet with dynamic voxelization.

    No additional learnable parameters beyond MVXTwoStageDetector.
    The dynamic voxelization logic (scatter ops) is data-dependent and
    not modelled in TTSim shape inference.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
