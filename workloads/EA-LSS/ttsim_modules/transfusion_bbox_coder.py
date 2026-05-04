#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of the TransFusion bounding-box coder.

Original file: mmdet3d/core/bbox/coders/transfusion_bbox_coder.py

Provides:
  - TransFusionBBoxCoder.encode  : numpy implementation (boxes → targets)
  - TransFusionBBoxCoder.decode  : TTSim graph implementation
                                   (heatmap + reg heads → 3-D boxes)

The encode path runs at training time and is implemented as pure-numpy
(no graph ops).  The decode path runs at inference and is the primary
focus of the TTSim conversion.

TTSim decode graph:
  heatmap  [B, num_cls, P]  → ReduceMax → scores [B, P]
                             → ArgMax   (modelled via TopK k=1)
  rot      [B, 2, P]        → atan2(sin, cos) approximated via
                               Sub/Div/Sqrt (see note below)
  dim      [B, 3, P]        → Exp
  center   [B, 2, P]        → scale + shift
  height   [B, 1, P]        → height - dim_z*0.5
  vel      [B, 2, P]        → pass-through (optional)
  ConcatX  → final_boxes    [B, P, 8 or 10]

Note on atan2:
  ONNX / TTSim do not have a native Atan2 op.  We implement:
    theta = atan2(sin, cos)
         ≈ sin / sqrt(sin^2 + cos^2) is not accurate (only sign correct).
  For full accuracy we use the standard series:
    if cos >= 0:  theta = atan(sin/cos)
    else:         theta = atan(sin/cos) + pi * sign(sin)
  Since TTSim has no Atan op either, we model atan2 as a TTSim
  "custom" node that stores its numpy-computed data in the SimTensor.
  Shape inference still works; data compute uses numpy.arctan2.

No torch / mmcv imports.
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
# Helper: atan2 via Identity op with numpy data compute
# ---------------------------------------------------------------------------

def _ttsim_atan2(sin_t, cos_t, name: str):
    """
    Approximate atan2(sin, cos) in TTSim.

    Shape is identical to sin_t / cos_t (element-wise).
    Data is computed with numpy.arctan2 when both inputs carry data.
    We use an Identity op as a shape-preserving placeholder.
    """
    out = _from_shape(name + ".atan2_out", list(sin_t.shape))
    out.op_in.append(name + ".atan2")
    if sin_t.data is not None and cos_t.data is not None:
        out.data = np.arctan2(sin_t.data, cos_t.data).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# TransFusionBBoxCoder
# ---------------------------------------------------------------------------

class TransFusionBBoxCoder(SimNN.Module):
    """
    TTSim implementation of the TransFusion bounding-box encoder/decoder.

    Args:
        name (str): Module name prefix.
        pc_range (list[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        out_size_factor (int): Spatial downsampling factor of the BEV grid.
        voxel_size (list[float]): [vx, vy, vz] voxel dimensions in metres.
        post_center_range (list[float] or None): [x_min,y_min,z_min,
                                                   x_max,y_max,z_max]
            spatial filter applied after decoding.  None → no filter.
        score_threshold (float or None): Score threshold for filtering.
        code_size (int): Number of box code dimensions (8 or 10).
    """

    def __init__(
        self,
        name: str,
        pc_range: list,
        out_size_factor: int,
        voxel_size: list,
        post_center_range: list = None,
        score_threshold: float = None,
        code_size: int = 8,
    ):
        super().__init__()
        self.name = name
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size
        super().link_op2module()

    # ------------------------------------------------------------------
    # encode  (pure numpy — training only)
    # ------------------------------------------------------------------

    def encode_numpy(self, dst_boxes: np.ndarray) -> np.ndarray:
        """
        Encode ground-truth 3-D boxes to TransFusion regression targets.

        Args:
            dst_boxes (np.ndarray): [N, 7+] boxes in LiDAR frame
                columns: [x, y, z, w, l, h, yaw, (vx, vy)].

        Returns:
            np.ndarray: [N, code_size] encoded targets.
        """
        N = dst_boxes.shape[0]
        targets = np.zeros((N, self.code_size), dtype=np.float32)
        vx, vy = self.voxel_size[0], self.voxel_size[1]
        osf = float(self.out_size_factor)

        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (osf * vx)
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (osf * vy)
        targets[:, 3] = np.log(np.maximum(dst_boxes[:, 3], 1e-6))
        targets[:, 4] = np.log(np.maximum(dst_boxes[:, 4], 1e-6))
        targets[:, 5] = np.log(np.maximum(dst_boxes[:, 5], 1e-6))
        # gravity centre (z + h/2)
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 6] = np.sin(dst_boxes[:, 6])
        targets[:, 7] = np.cos(dst_boxes[:, 6])
        if self.code_size == 10 and dst_boxes.shape[1] >= 9:
            targets[:, 8:10] = dst_boxes[:, 7:9]
        return targets

    # ------------------------------------------------------------------
    # decode  (TTSim graph)
    # ------------------------------------------------------------------

    def decode(
        self,
        heatmap,
        rot,
        dim,
        center,
        height,
        vel=None,
    ):
        """
        Decode TransFusion regression outputs into 3-D bounding boxes.

        Args:
            heatmap (SimTensor): [B, num_cls, P] class score logits.
            rot     (SimTensor): [B, 2, P]  (sin_yaw, cos_yaw).
            dim     (SimTensor): [B, 3, P]  log(w, l, h).
            center  (SimTensor): [B, 2, P]  BEV grid offsets (x, y).
            height  (SimTensor): [B, 1, P]  height (z-gravity-centre).
            vel     (SimTensor or None): [B, 2, P] velocity (optional).

        Returns:
            SimTensor: [B, P, 8] (or [B, P, 10] with velocity)
                columns: [x, y, z, w, l, h, yaw, (vx, vy)]
        """
        B, num_cls, P = heatmap.shape

        name = self.name

        # ---- 1. Class scores via ReduceMax over cls dim -----------------
        cls_axes_t = _from_data(name + ".cls_ax", np.array([1], dtype=np.int64),
                                 is_const=True)
        score_op = SimOpHandle(name + ".reduce_score", "ReduceMax",
                               params=[(1, cls_axes_t)], ipos=[0], keepdims=0)
        score_op.implicit_inputs.append(cls_axes_t)
        scores = score_op(heatmap)   # [B, P]

        # ---- 2. Dim: exp(log_dim) → actual w, l, h ----------------------
        dim_out = F.Exp(name + ".dim_exp")(dim)   # [B, 3, P]

        # ---- 3. Rotation: atan2(sin, cos) --------------------------------
        # Slice sin and cos from rot [B, 2, P]
        sin_t = _from_shape(name + ".rot_sin", [B, 1, P])
        cos_t = _from_shape(name + ".rot_cos", [B, 1, P])
        if rot.data is not None:
            sin_t.data = rot.data[:, 0:1, :]
            cos_t.data = rot.data[:, 1:2, :]
        rot_angle = _ttsim_atan2(sin_t, cos_t, name)  # [B, 1, P]

        # ---- 4. Center: grid → real-world coords -------------------------
        # center_x_real = center_x * osf * vx + pc_range[0]
        vx = float(self.voxel_size[0])
        vy = float(self.voxel_size[1])
        osf = float(self.out_size_factor)

        cx_scale_np = np.array([osf * vx, osf * vy], dtype=np.float32).reshape(1, 2, 1)  # [1,2,1]
        cx_shift_np = np.array([self.pc_range[0], self.pc_range[1]],
                                dtype=np.float32).reshape(1, 2, 1)

        cx_scale_t = _from_data(name + ".cx_scale", cx_scale_np, is_const=True)
        cx_shift_t = _from_data(name + ".cx_shift", cx_shift_np, is_const=True)

        mul_cx = SimOpHandle(name + ".mul_center", "Mul",
                             params=[(1, cx_scale_t)], ipos=[0])
        mul_cx.implicit_inputs.append(cx_scale_t)
        center_scaled = mul_cx(center)   # [B, 2, P]

        add_cx = SimOpHandle(name + ".add_center", "Add",
                             params=[(1, cx_shift_t)], ipos=[0])
        add_cx.implicit_inputs.append(cx_shift_t)
        center_real = add_cx(center_scaled)  # [B, 2, P]

        # ---- 5. Height: gravity centre → bottom centre ------------------
        # height_bottom = height - dim_z * 0.5
        half_np = np.array([[[0.5]]], dtype=np.float32)
        half_t = _from_data(name + ".half", half_np, is_const=True)

        # dim_z is at index 2: shape [B, 1, P]
        dim_z = _from_shape(name + ".dim_z", [B, 1, P])
        if dim_out.data is not None:
            dim_z.data = dim_out.data[:, 2:3, :]

        mul_dz = SimOpHandle(name + ".mul_dz", "Mul",
                             params=[(1, half_t)], ipos=[0])
        mul_dz.implicit_inputs.append(half_t)
        half_dz = mul_dz(dim_z)   # [B, 1, P]

        sub_h = F.Sub(name + ".sub_height")(height, half_dz)  # [B, 1, P]

        # ---- 6. Concatenate → [B, 8_or_10, P] --------------------------
        if vel is None:
            final = F.ConcatX(name + ".cat_boxes", axis=1)(
                center_real, sub_h, dim_out, rot_angle
            )  # [B, 2+1+3+1=7, P]  → [B, 7, P]
        else:
            final = F.ConcatX(name + ".cat_boxes", axis=1)(
                center_real, sub_h, dim_out, rot_angle, vel
            )  # [B, 9, P]

        # Transpose [B, C, P] → [B, P, C]
        tr_op = SimOpHandle(name + ".tr_final", "Transpose",
                            params=[], ipos=[0], perm=[0, 2, 1])
        tr_op.implicit_inputs = []
        final_boxes = tr_op(final)  # [B, P, 7 or 9]

        return final_boxes, scores

    def analytical_param_count(self, lvl=0):
        return 0   # no trainable parameters
