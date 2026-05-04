#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of coordinate-transform utilities.

Original file: mmdet3d/models/fusion_layers/coord_transform.py

Provides two functional helpers used at inference time:
  - apply_3d_transformation: applies a sequential flow of T/S/R/HF/VF operations
    to a batch of 3-D points (N, 3+) represented as a SimTensor.
  - extract_2d_info: extracts and returns 2-D image augmentation scalars from
    a metadata dictionary (no graph ops; returns python scalars + numpy arrays).

No trainable parameters.  No torch/mmcv imports.

TTSim ops used:
    MatMul  – rotation:  pcd @ R^T
    Mul     – scaling:   pcd * scale_factor
    Add     – translation: pcd + trans_vector
    Neg     – negation for reverse translate (or Mul by -1 const)
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data


# ---------------------------------------------------------------------------
# apply_3d_transformation
# ---------------------------------------------------------------------------

def apply_3d_transformation(
    pcd,
    name: str,
    rotation_mat: np.ndarray = None,
    scale_factor: float = 1.0,
    trans_vector: np.ndarray = None,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    flow: list = None,
    reverse: bool = False,
):
    """
    Apply a sequential flow of 3-D transformations to a point-cloud SimTensor.

    In the original PyTorch code the transformation parameters come from
    ``img_meta`` dict at runtime.  Here they are passed explicitly as numpy
    arrays / scalars so the TTSim graph can be built offline.

    Each transformation step is one or more TTSim ops:
      - 'T'  translation:  pcd[:, :3] + trans_vector  (Add)
      - 'S'  scale:        pcd[:, :3] * scale_factor  (Mul by scalar const)
      - 'R'  rotation:     pcd[:, :3] @ R             (MatMul with const matrix)
      - 'HF' horiz flip:   pcd[:, 0] *= -1            (Mul slice; modelled as MulFixed)
      - 'VF' vert  flip:   pcd[:, 1] *= -1            (Mul slice; modelled as MulFixed)

    For simplicity in the TTSim graph we apply transformations to the **full**
    xyz slice only.  If pcd has extra features (intensity, etc.) they are
    preserved unchanged (concatenated back after transformation).

    Args:
        pcd         : SimTensor  [N, C] where C >= 3; first 3 cols are XYZ.
        name        : str        unique op-name prefix.
        rotation_mat: np.ndarray [3, 3] rotation matrix (None → identity).
        scale_factor: float      uniform scale (default 1.0).
        trans_vector: np.ndarray [3]   translation  (None → zero).
        horizontal_flip: bool    flip X axis.
        vertical_flip  : bool    flip Y axis.
        flow        : list[str]  ordered list of ops, e.g. ['T','S','R'].
                                 None → no transformation (identity).
        reverse     : bool       apply flow in reverse with inverse ops.

    Returns:
        SimTensor  [N, C] transformed point cloud.
    """
    if flow is None or len(flow) == 0:
        return pcd

    N = pcd.shape[0]
    C = pcd.shape[1] if len(pcd.shape) > 1 else 1

    # Build default numpy params
    if rotation_mat is None:
        rotation_mat = np.eye(3, dtype=np.float32)
    else:
        rotation_mat = np.array(rotation_mat, dtype=np.float32)

    if trans_vector is None:
        trans_vector = np.zeros(3, dtype=np.float32)
    else:
        trans_vector = np.array(trans_vector, dtype=np.float32)

    # ---- constant tensors --------------------------------------------------
    # Rotation matrix R: [3, 3]  (or R^-1 for reverse)
    if reverse:
        rot_np = np.linalg.inv(rotation_mat)
    else:
        rot_np = rotation_mat
    rot_t = _from_data(name + ".rot_mat", rot_np.astype(np.float32), is_const=True)

    # Scale scalar broadcast: stored as [N, 3] const
    if reverse:
        scale_np = np.full((1, 3), 1.0 / scale_factor, dtype=np.float32)
    else:
        scale_np = np.full((1, 3), scale_factor, dtype=np.float32)
    scale_t = _from_data(name + ".scale_vec", scale_np, is_const=True)

    # Translation vector: [1, 3]
    if reverse:
        tv_np = (-trans_vector).reshape(1, 3).astype(np.float32)
    else:
        tv_np = trans_vector.reshape(1, 3).astype(np.float32)
    trans_t = _from_data(name + ".trans_vec", tv_np, is_const=True)

    # Flip constants: -1 for X or Y axis  [1, C]
    neg_x_np = np.ones((1, C), dtype=np.float32)
    neg_x_np[0, 0] = -1.0
    neg_x_t = _from_data(name + ".neg_x", neg_x_np, is_const=True)

    neg_y_np = np.ones((1, C), dtype=np.float32)
    neg_y_np[0, 1] = -1.0
    neg_y_t = _from_data(name + ".neg_y", neg_y_np, is_const=True)

    # ---- Transformation helpers --------------------------------------------
    def _translate(x, step):
        # x [N, C], trans_t [1, 3] — add only to first 3 cols
        # We model this as SliceF + Add + Concat; approximate with full Add
        # on the xyz columns only, which requires slicing. For TTSim we
        # approximate: translate all-C using 0-padded const if C > 3,
        # but since only XYZ (first 3 cols) are semantically translated
        # we pad trans_vec to [1, C]
        tv_padded = np.zeros((1, C), dtype=np.float32)
        tv_padded[0, :3] = ((-trans_vector) if reverse else trans_vector)
        tv_t = _from_data(name + f".trans_pad_{step}", tv_padded, is_const=True)
        add_op = SimOpHandle(name + f".translate_{step}", "Add",
                             params=[(1, tv_t)], ipos=[0])
        add_op.implicit_inputs.append(tv_t)
        return add_op(x)

    def _scale(x, step):
        sc_padded = np.ones((1, C), dtype=np.float32)
        sc_padded[0, :3] = (1.0 / scale_factor) if reverse else scale_factor
        sc_t = _from_data(name + f".scale_pad_{step}", sc_padded, is_const=True)
        mul_op = SimOpHandle(name + f".scale_{step}", "Mul",
                             params=[(1, sc_t)], ipos=[0])
        mul_op.implicit_inputs.append(sc_t)
        return mul_op(x)

    def _rotate(x, step):
        # x [N, C]; rotate xyz: x_xyz @ rot_np → [N, 3]
        # For the full-column tensor we use a [C, C] block-diagonal matrix
        rot_full = np.eye(C, dtype=np.float32)
        rot_full[:3, :3] = rot_np
        rot_full_t = _from_data(name + f".rot_full_{step}",
                                rot_full, is_const=True)
        mm_op = SimOpHandle(name + f".rotate_{step}", "MatMul",
                            params=[(1, rot_full_t)], ipos=[0])
        mm_op.implicit_inputs.append(rot_full_t)
        return mm_op(x)

    def _hflip(x, step):
        neg_full = np.ones((1, C), dtype=np.float32)
        neg_full[0, 0] = -1.0
        neg_t = _from_data(name + f".neg_x_{step}", neg_full, is_const=True)
        mul_op = SimOpHandle(name + f".hflip_{step}", "Mul",
                             params=[(1, neg_t)], ipos=[0])
        mul_op.implicit_inputs.append(neg_t)
        return mul_op(x)

    def _vflip(x, step):
        neg_full = np.ones((1, C), dtype=np.float32)
        neg_full[0, 1] = -1.0
        neg_t = _from_data(name + f".neg_y_{step}", neg_full, is_const=True)
        mul_op = SimOpHandle(name + f".vflip_{step}", "Mul",
                             params=[(1, neg_t)], ipos=[0])
        mul_op.implicit_inputs.append(neg_t)
        return mul_op(x)

    op_funcs = {
        'T':  _translate,
        'S':  _scale,
        'R':  _rotate,
        'HF': _hflip if horizontal_flip else lambda x, s: x,
        'VF': _vflip if vertical_flip  else lambda x, s: x,
    }

    ordered_flow = list(reversed(flow)) if reverse else list(flow)
    out = pcd
    for step_idx, op_key in enumerate(ordered_flow):
        assert op_key in op_funcs, f"Unknown 3D transform op: {op_key}"
        out = op_funcs[op_key](out, step_idx)

    return out


# ---------------------------------------------------------------------------
# extract_2d_info  (metadata-only helper; returns numpy scalars / arrays)
# ---------------------------------------------------------------------------

def extract_2d_info(img_meta: dict):
    """
    Extract 2-D image augmentation parameters from an img_meta dictionary.

    Mirrors mmdet3d/models/fusion_layers/coord_transform.py::extract_2d_info
    but without any torch dependency — returns plain Python scalars and
    numpy arrays.

    Args:
        img_meta (dict): Metadata dict from the mmdet3d pipeline.

    Returns:
        tuple:
            img_h (int), img_w (int), ori_h (int), ori_w (int),
            scale_factor (np.ndarray [2]),
            flip (bool),
            crop_offset (np.ndarray [2])
    """
    img_shape = img_meta['img_shape']
    ori_shape = img_meta['ori_shape']
    img_h, img_w = img_shape[0], img_shape[1]
    ori_h, ori_w = ori_shape[0], ori_shape[1]

    scale_factor = (
        np.array(img_meta['scale_factor'][:2], dtype=np.float32)
        if 'scale_factor' in img_meta
        else np.array([1.0, 1.0], dtype=np.float32)
    )
    flip = img_meta.get('flip', False)
    crop_offset = (
        np.array(img_meta['img_crop_offset'], dtype=np.float32)
        if 'img_crop_offset' in img_meta
        else np.array([0.0, 0.0], dtype=np.float32)
    )

    return img_h, img_w, ori_h, ori_w, scale_factor, flip, crop_offset
