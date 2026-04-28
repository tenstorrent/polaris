
# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
#
#
# def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
#     """
#     Parameters
#     ----------
#         x_bounds: Forward direction in the ego-car.
#         y_bounds: Sides
#         z_bounds: Height
#
#     Returns
#     -------
#         bev_resolution: Bird's-eye view bev_resolution
#         bev_start_position Bird's-eye view first element
#         bev_dimension Bird's-eye view tensor spatial dimension
#     """
#     bev_resolution = torch.tensor(
#         [row[2] for row in [x_bounds, y_bounds, z_bounds]])
#     bev_start_position = torch.tensor(
#         [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
#     bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
#                                  for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)
#
#     return bev_resolution, bev_start_position, bev_dimension
#
#
# def gen_dx_bx(xbound, ybound, zbound):
#     dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
#     bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
#     nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
#
#     return dx, bx, nx
#
# # Instance utils
# def update_instance_ids(instance_seg, old_ids, new_ids):
#     """
#     Parameters
#     ----------
#         instance_seg: torch.Tensor arbitrary shape
#         old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
#         new_ids: 1D tensor with the new ids, aligned with old_ids
#
#     Returns
#         new_instance_seg: torch.Tensor same shape as instance_seg with new ids
#     """
#     indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
#     for old_id, new_id in zip(old_ids, new_ids):
#         indices[old_id] = new_id
#
#     return indices[instance_seg].long()
#
#
# def make_instance_seg_consecutive(instance_seg):
#     # Make the indices of instance_seg consecutive
#     unique_ids = torch.unique(instance_seg)  # include background
#     new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
#     instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
#     return instance_seg
#
#
# def predict_instance_segmentation_and_trajectories(
#                                     foreground_masks,
#                                     ins_sigmoid,
#                                     vehicles_id=1,
#                                     ):
#     if foreground_masks.dim() == 5 and foreground_masks.shape[2] == 1:
#         foreground_masks = foreground_masks.squeeze(2)  # [b, t, h, w]
#     foreground_masks = foreground_masks == vehicles_id  # [b, t, h, w]  Only these places have foreground id
#
#     argmax_ins = ins_sigmoid.argmax(dim=1)  # long, [b, t, h, w], ins_id starts from 0
#     argmax_ins = argmax_ins + 1 # [b, t, h, w], ins_id starts from 1
#     instance_seg = (argmax_ins * foreground_masks.float()).long()  # bg is 0, fg starts with 1
#
#     # Make the indices of instance_seg consecutive
#     instance_seg = make_instance_seg_consecutive(instance_seg).long()
#
#     return instance_seg
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#------------------------------Pytorch------------------------------------------------#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
#     """
#     Parameters
#     ----------
#         x_bounds: Forward direction in the ego-car.
#         y_bounds: Sides
#         z_bounds: Height

#     Returns
#     -------
#         bev_resolution: Bird's-eye view bev_resolution
#         bev_start_position Bird's-eye view first element
#         bev_dimension Bird's-eye view tensor spatial dimension
#     """
#     bev_resolution = torch.tensor(
#         [row[2] for row in [x_bounds, y_bounds, z_bounds]])
#     bev_start_position = torch.tensor(
#         [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
#     bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
#                                  for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

#     return bev_resolution, bev_start_position, bev_dimension


# def gen_dx_bx(xbound, ybound, zbound):
#     dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
#     bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
#     nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

#     return dx, bx, nx

# # Instance utils
# def update_instance_ids(instance_seg, old_ids, new_ids):
#     """
#     Parameters
#     ----------
#         instance_seg: torch.Tensor arbitrary shape
#         old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
#         new_ids: 1D tensor with the new ids, aligned with old_ids

#     Returns
#         new_instance_seg: torch.Tensor same shape as instance_seg with new ids
#     """
#     indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
#     for old_id, new_id in zip(old_ids, new_ids):
#         indices[old_id] = new_id

#     return indices[instance_seg].long()


# def make_instance_seg_consecutive(instance_seg):
#     # Make the indices of instance_seg consecutive
#     unique_ids = torch.unique(instance_seg)  # include background
#     new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
#     instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
#     return instance_seg


# def predict_instance_segmentation_and_trajectories(
#                                     foreground_masks,
#                                     ins_sigmoid,
#                                     vehicles_id=1,
#                                     ):
#     if foreground_masks.dim() == 5 and foreground_masks.shape[2] == 1:
#         foreground_masks = foreground_masks.squeeze(2)  # [b, t, h, w]
#     foreground_masks = foreground_masks == vehicles_id  # [b, t, h, w]  Only these places have foreground id

#     argmax_ins = ins_sigmoid.argmax(dim=1)  # long, [b, t, h, w], ins_id starts from 0
#     argmax_ins = argmax_ins + 1 # [b, t, h, w], ins_id starts from 1
#     instance_seg = (argmax_ins * foreground_masks.float()).long()  # bg is 0, fg starts with 1

#     # Make the indices of instance_seg consecutive
#     instance_seg = make_instance_seg_consecutive(instance_seg).long()

#     return instance_seg

#------------------------------TTsim-------------------------------------------------#
"""
TTSim conversion of occ_head_plugin/utils.py

All functions are numpy-only (init-time or post-processing).
No graph ops needed — these never run inside the TTSim forward graph.

Note: calculate_birds_eye_view_parameters and
      predict_instance_segmentation_and_trajectories are already
      defined in modules.py (the primary consumers). This file
      provides the remaining helpers for completeness.
"""

import numpy as np
from loguru import logger


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """Compute BEV grid resolution, start position, and dimension."""
    bev_resolution = np.array(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = np.array(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = np.array(
        [(row[1] - row[0]) / row[2]
         for row in [x_bounds, y_bounds, z_bounds]], dtype=np.int64)
    return bev_resolution, bev_start_position, bev_dimension


def gen_dx_bx(xbound, ybound, zbound):
    """Generate dx (resolution), bx (start center), nx (grid counts)."""
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.array([(row[1] - row[0]) / row[2]
                   for row in [xbound, ybound, zbound]], dtype=np.int64)
    return dx, bx, nx


def update_instance_ids(instance_seg, old_ids, new_ids):
    """Remap instance IDs in a segmentation array.

    Parameters
    ----------
    instance_seg : np.ndarray, arbitrary shape
    old_ids : 1-D array of old IDs (all must be present in instance_seg)
    new_ids : 1-D array of new IDs, aligned with old_ids

    Returns
    -------
    np.ndarray same shape as instance_seg with remapped IDs
    """
    lut = np.arange(int(old_ids.max()) + 1, dtype=np.int64)
    for old, new in zip(old_ids, new_ids):
        lut[int(old)] = int(new)
    return lut[instance_seg].astype(np.int64)


def make_instance_seg_consecutive(instance_seg):
    """Make instance IDs consecutive starting from 0."""
    unique_ids = np.unique(instance_seg)
    new_ids = np.arange(len(unique_ids), dtype=np.int64)
    return update_instance_ids(instance_seg, unique_ids, new_ids)


def predict_instance_segmentation_and_trajectories(
        foreground_masks, ins_sigmoid, vehicles_id=1):
    """Post-processing: assign instance IDs within foreground regions.

    Parameters
    ----------
    foreground_masks : np.ndarray (B, T, 1, H, W) or (B, T, H, W)
    ins_sigmoid : np.ndarray (B, Q, T, H, W)

    Returns
    -------
    np.ndarray (B, T, H, W) int64, consecutive instance IDs (0 = background)
    """
    if foreground_masks.ndim == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)
    foreground_masks = (foreground_masks == vehicles_id)

    argmax_ins = ins_sigmoid.argmax(axis=1)   # (B, T, H, W)
    argmax_ins = argmax_ins + 1               # IDs start from 1
    instance_seg = (argmax_ins * foreground_masks.astype(np.float32)).astype(np.int64)

    instance_seg = make_instance_seg_consecutive(instance_seg)
    return instance_seg


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Self-test: occ_head_plugin/utils.py")
    logger.info("=" * 60)

    passed = 0

    # 1. calculate_birds_eye_view_parameters
    res, start, dim = calculate_birds_eye_view_parameters(
        [-50, 50, 0.5], [-50, 50, 0.5], [-10, 10, 20])
    assert np.isclose(res[0], 0.5) and dim[0] == 200
    passed += 1
    logger.debug("  [OK] calculate_birds_eye_view_parameters")

    # 2. gen_dx_bx
    dx, bx, nx = gen_dx_bx([-50, 50, 0.5], [-50, 50, 0.5], [-10, 10, 20])
    assert np.isclose(dx[0], 0.5) and nx[0] == 200
    passed += 1
    logger.debug("  [OK] gen_dx_bx")

    # 3. update_instance_ids
    seg = np.array([0, 1, 2, 3, 1, 2])
    new = update_instance_ids(seg, np.array([0, 1, 2, 3]),
                              np.array([0, 10, 20, 30]))
    assert np.array_equal(new, [0, 10, 20, 30, 10, 20])
    passed += 1
    logger.debug("  [OK] update_instance_ids")

    # 4. make_instance_seg_consecutive
    seg2 = np.array([0, 5, 5, 10, 10, 0])
    cons = make_instance_seg_consecutive(seg2)
    assert set(cons.tolist()) == {0, 1, 2}
    passed += 1
    logger.debug("  [OK] make_instance_seg_consecutive")

    # 5. predict_instance_segmentation_and_trajectories
    B, Q, T, H, W = 1, 3, 2, 8, 8
    fg = np.zeros((B, T, 1, H, W), dtype=np.int64)
    fg[:, :, :, 2:6, 2:6] = 1
    ins = np.random.rand(B, Q, T, H, W).astype(np.float32)
    out = predict_instance_segmentation_and_trajectories(fg, ins)
    assert out.shape == (B, T, H, W) and out.dtype == np.int64
    assert np.all(out[:, :, :2, :] == 0)  # border = background
    passed += 1
    logger.debug("  [OK] predict_instance_segmentation_and_trajectories")

    logger.info(f"\n{passed}/5 tests passed.")
