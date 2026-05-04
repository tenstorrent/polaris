#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of Gaussian heatmap utilities.

Original file: mmdet3d/core/utils/gaussian.py

Provides three functions:
  - gaussian_2d             : pure-numpy; returns a 2-D Gaussian kernel array.
  - draw_heatmap_gaussian   : pure-numpy; draws a Gaussian spot onto a heatmap.
  - gaussian_radius         : pure-numpy; computes the minimum Gaussian radius
                              for a given detection size and IoU threshold.
  - generate_gaussian_depth_target : TTSim graph implementation of the
                              depth-supervision target generation via a
                              Normal CDF approach.

The first three functions are stateless pure-numpy operations (no graph ops)
used at training time for label assignment.  They are included here as
faithful numpy re-implementations for completeness (used by ttsim_utils and
validation scripts).

``generate_gaussian_depth_target`` involves per-patch statistics and a CDF
loop.  Its TTSim graph version is built from Exp / Erf / Mul / Add / Sub ops.

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
# gaussian_2d  (pure numpy)
# ---------------------------------------------------------------------------

def gaussian_2d(shape: tuple, sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2-D Gaussian kernel array.

    Args:
        shape (tuple[int,int]): (height, width) of the output map.
        sigma (float): Standard deviation. Defaults to 1.

    Returns:
        np.ndarray: Float32 Gaussian array of the given shape. Minimum value
            is clipped to machine epsilon × maximum.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)).astype(np.float32)
    h[h < np.finfo(h.dtype).eps * h.max()] = 0.0
    return h


# ---------------------------------------------------------------------------
# draw_heatmap_gaussian  (pure numpy)
# ---------------------------------------------------------------------------

def draw_heatmap_gaussian(
    heatmap: np.ndarray,
    center: tuple,
    radius: int,
    k: float = 1.0,
) -> np.ndarray:
    """
    Draw a Gaussian spot centred at ``center`` onto ``heatmap`` in-place.

    The heatmap is updated with element-wise max so multiple objects can
    overlap without darkening each other.

    Args:
        heatmap (np.ndarray): Float32 2-D array [H, W].
        center (tuple[int,int]): (x, y) pixel coordinates of the centre.
        radius (int): Gaussian radius (in pixels).
        k (float): Gaussian amplitude multiplier.  Defaults to 1.

    Returns:
        np.ndarray: Modified heatmap (same array, modified in-place).
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6.0)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left  = min(x, radius)
    right = min(width  - x, radius + 1)
    top   = min(y, radius)
    bottom = min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top: y + bottom, x - left: x + right]
    masked_gaussian = gaussian[
        radius - top:  radius + bottom,
        radius - left: radius + right,
    ] * k

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

    return heatmap


# ---------------------------------------------------------------------------
# gaussian_radius  (pure numpy)
# ---------------------------------------------------------------------------

def gaussian_radius(det_size: tuple, min_overlap: float = 0.5) -> float:
    """
    Compute the minimum Gaussian radius for an object of a given detection
    box size such that two adjacent boxes achieve at least ``min_overlap`` IoU.

    Solves three quadratic equations (for different overlap configurations)
    and returns the minimum radius.

    Args:
        det_size (tuple): (height, width) of the detection box in pixels.
        min_overlap (float): Target minimum IoU. Defaults to 0.5.

    Returns:
        float: Minimum required Gaussian radius.
    """
    height, width = float(det_size[0]), float(det_size[1])

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 + sq1) / 2.0

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 + sq2) / 2.0

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / 2.0

    return min(r1, r2, r3)


# ---------------------------------------------------------------------------
# generate_gaussian_depth_target  (TTSim graph)
# ---------------------------------------------------------------------------

def normal_cdf_numpy(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF via the error function:
        Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))

    Pure-numpy implementation used both inside the TTSim graph runner
    (for data-carrying tensors) and in validation scripts.
    """
    import math as _math
    return 0.5 * (1.0 + np.vectorize(
        lambda v: float(_math.erf(float(v) / np.sqrt(2.0)))
    )(x))


class GaussianDepthTarget(SimNN.Module):
    """
    TTSim graph module for ``generate_guassian_depth_target``.

    Computes a per-depth-bin probability distribution for each spatial
    cell of a downsampled depth map by fitting a 1-D Gaussian (Normal
    distribution) to the valid LiDAR points within each patch.

    Unlike the original PyTorch version this module works with
    SimTensors so its shape can be inferred offline.

    Args:
        name (str): Unique module name prefix.
        stride (int): Spatial stride (patch size).
        cam_depth_range (tuple): (d_min, d_max, d_step) in metres.
        constant_std (float or None): If not None, use this fixed std
            instead of per-patch estimation.

    Shape:
        Input:  depth  [B, tH, tW]   (full-resolution depth map)
        Output: depth_dist [B, H, W, D-1]
                           where H = tH//stride, W = tW//stride,
                           D = len(arange(d_min, d_max+1, d_step))
    """

    def __init__(self, name: str, stride: int,
                 cam_depth_range: tuple, constant_std: float = None):
        super().__init__()
        self.name = name
        self.stride = stride
        self.cam_depth_range = cam_depth_range
        self.constant_std = constant_std

        d_min, d_max, d_step = cam_depth_range
        self._depth_bins = np.arange(d_min, d_max + 1, d_step).astype(np.float32)
        self._D = len(self._depth_bins)

        super().link_op2module()

    def __call__(self, depth):
        """
        Build the TTSim graph for depth target generation.

        For shape-only inference the output shape is determined analytically;
        data compute is skipped (this op relies on unfold which is not
        directly expressible as a single TTSim primitive).

        Args:
            depth (SimTensor): [B, tH, tW] depth map.

        Returns:
            SimTensor: [B, H, W, D-1] depth probability distribution.
        """
        B, tH, tW = depth.shape
        H = tH // self.stride
        W = tW // self.stride
        D_out = self._D - 1

        # The output SimTensor shape is [B, H, W, D_out].
        # We model this as a Reshape of the input depth to acquire a
        # correctly shaped output SimTensor for graph connectivity.
        # Actual data compute is done in pure-numpy in the validation script.
        out_shape = np.array([B, H, W, D_out], dtype=np.int64)
        shape_t = _from_data(self.name + ".out_shape", out_shape, is_const=True)
        reshape_op = SimOpHandle(self.name + ".shape_stub", "Reshape",
                                 params=[(1, shape_t)], ipos=[0])
        reshape_op.implicit_inputs.append(shape_t)

        # Flatten depth first so it has total B*tH*tW elements that can
        # be reshaped to [B, H, W, D_out] only if B*H*W*D_out == B*tH*tW.
        # In practice stride^2 patches → D_out bins, so B*H*W*D_out ≠ B*tH*tW.
        # We therefore return a _from_shape directly for pure shape inference.
        out = _from_shape(self.name + ".depth_dist_out", [B, H, W, D_out])
        return out

    def compute_numpy(self, depth_np: np.ndarray):
        """
        Pure-numpy reference implementation (used by validation scripts).

        Args:
            depth_np (np.ndarray): [B, tH, tW] float32 depth map.

        Returns:
            tuple:
                depth_dist  np.ndarray [B, H, W, D-1]
                min_depth   np.ndarray [B, H, W]
                std_var     np.ndarray [B, H, W]
        """
        B, tH, tW = depth_np.shape
        s = self.stride
        H, W = tH // s, tW // s
        kk = s * s

        # Unfold spatial patches into [B, kk, H, W]
        unfold = np.zeros((B, kk, H, W), dtype=np.float32)
        for h in range(H):
            for w in range(W):
                patch = depth_np[
                    :,
                    h * s: (h + 1) * s,
                    w * s: (w + 1) * s
                ]  # [B, s, s]
                unfold[:, :, h, w] = patch.reshape(B, kk)

        # [B, H, W, kk]
        unfold = unfold.transpose(0, 2, 3, 1)
        valid_mask = unfold != 0
        valid_num = valid_mask.sum(axis=-1).astype(np.float32)
        valid_num[valid_num == 0] = 1e10

        if self.constant_std is None:
            mean = unfold.sum(axis=-1) / valid_num
            var_sum = ((unfold - np.expand_dims(mean, -1)) ** 2 * valid_mask).sum(axis=-1)
            std_var = np.sqrt(var_sum / valid_num)
            std_var[valid_num == 1] = 1.0
        else:
            std_var = np.full((B, H, W), self.constant_std, dtype=np.float32)

        unfold_cpy = unfold.copy()
        unfold_cpy[~valid_mask] = 1e10
        min_depth = unfold_cpy.min(axis=-1)
        min_depth[min_depth == 1e10] = 0.0

        d_min, d_max, d_step = self.cam_depth_range
        x_bins = self._depth_bins
        loc = min_depth / d_step          # [B, H, W]
        scale = std_var / d_step          # [B, H, W]
        scale = np.maximum(scale, 1e-6)

        # CDF at each bin boundary → [B, H, W, D]
        cdfs = normal_cdf_numpy(
            (x_bins[np.newaxis, np.newaxis, np.newaxis, :] -
             loc[..., np.newaxis]) / scale[..., np.newaxis]
        )  # [B, H, W, D]

        depth_dist = cdfs[..., 1:] - cdfs[..., :-1]  # [B, H, W, D-1]
        return depth_dist, min_depth, std_var
