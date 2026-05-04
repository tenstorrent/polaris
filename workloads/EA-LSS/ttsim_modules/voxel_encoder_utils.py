#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of VFE (Voxel Feature Encoder) building blocks.

Original file: mmdet3d/models/voxel_encoders/utils.py

Provides:
  - get_paddings_indicator  : pure-numpy boolean mask, no graph ops
  - VFELayer                : SimNN.Module  – Linear + BN1d + ReLU + optional MaxPool
  - DynamicVFELayer         : SimNN.Module  – Linear + BN1d + ReLU (no pool)

Both VFE layers use the same sub-modules as mlp.py's ConvModule1d /
BatchNorm1d, so we import those directly rather than re-implementing.

TTSim ops used (via sub-modules):
  Conv (kernel=1) → BatchNormalization → Relu
  ReduceMax for aggregation (keepdims=1) implemented via SimOpHandle.
  Tile  for broadcasting aggregated features back to point dimension.
  ConcatX for cat_max path.

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

# Re-use the already-verified BatchNorm1d and the linear (Conv1d) layer
# from mlp.py so we don't duplicate implementation.
_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.mlp import BatchNorm1d, ConvModule1d


# ---------------------------------------------------------------------------
# get_paddings_indicator  (pure-numpy utility)
# ---------------------------------------------------------------------------

def get_paddings_indicator(actual_num: np.ndarray, max_num: int, axis: int = 0) -> np.ndarray:
    """
    Create a boolean mask that indicates which point-slots within a padded
    voxel tensor are valid (i.e. contain actual points).

    This is a pure-numpy implementation — no SimTensor / graph ops involved.

    Args:
        actual_num (np.ndarray): 1-D int array of shape [N] giving the
            number of points in each voxel.
        max_num (int): Maximum number of points per voxel (padded size M).
        axis (int): Axis along which to unsqueeze actual_num (default 0).

    Returns:
        np.ndarray: Boolean mask of shape [N, M]. True means the slot
            contains a real point.
    """
    actual_num = np.expand_dims(actual_num.astype(np.int32), axis=axis + 1)
    max_num_range = np.arange(max_num, dtype=np.int32)
    # broadcast: actual_num [N, 1] > max_num_range [M,]  → [N, M]
    return actual_num > max_num_range


# ---------------------------------------------------------------------------
# VFELayer
# ---------------------------------------------------------------------------

class VFELayer(SimNN.Module):
    """
    TTSim implementation of the Voxel Feature Encoder layer.

    Graph:
        input [N, M, C_in]
          → Transpose to [N, C_in, M]
          → ConvModule1d (Linear-equivalent kernel=1 + BN1d + ReLU) → [N, C_out, M]
          → Transpose back to [N, M, C_out]                           (pointwise)
          → [optional] ReduceMax over M → [N, 1, C_out]              (aggregated)
          → [optional] Tile to [N, M, C_out]                          (repeated)
          → [optional] ConcatX([pointwise, repeated]) → [N, M, 2*C_out]

    Args:
        name (str): Unique module name prefix.
        in_channels (int): Input feature dimension (C_in).
        out_channels (int): Output feature dimension (C_out) before cat_max.
        norm_cfg (dict): Batch-norm config; eps and momentum are used.
        max_out (bool): If True, aggregate via max-pool over point dimension.
        cat_max (bool): If max_out is True, whether to concatenate max-pooled
            and pointwise features (output C = 2*out_channels).
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        norm_cfg: dict = None,
        max_out: bool = True,
        cat_max: bool = True,
    ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(eps=1e-3, momentum=0.01)

        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_out = max_out
        self.cat_max = cat_max

        eps = norm_cfg.get('eps', 1e-3)
        momentum = norm_cfg.get('momentum', 0.01)

        self.conv_module = ConvModule1d(
            name + ".conv_module",
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            with_bn=False,   # BN created separately to allow eps/momentum
            with_relu=True,
        )
        # BatchNorm1d with VFE-specific eps / momentum
        from ttsim_modules.mlp import BatchNorm1d as _BN1d
        self._bn = _BN1d(name + ".bn", out_channels, eps=eps, momentum=momentum)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (SimTensor): [N, M, C_in] voxel features.

        Returns:
            SimTensor:
                - max_out=False           → [N, M, C_out]  pointwise
                - max_out=True, cat_max=F → [N, 1, C_out]  aggregated
                - max_out=True, cat_max=T → [N, M, 2*C_out] cat
        """
        N, M, C_in = x.shape

        # Permute [N, M, C] → [N, C, M] for Conv1d
        perm_t = _from_data(
            self.name + ".perm012", np.array([0, 2, 1], dtype=np.int64), is_const=True
        )
        tr_op = SimOpHandle(self.name + ".tr_in", "Transpose",
                            params=[], ipos=[0], perm=[0, 2, 1])
        tr_op.implicit_inputs = []
        x_t = tr_op(x)  # [N, C_in, M]

        # Conv + BN + ReLU
        pw_conv = self.conv_module(x_t)  # [N, C_out, M]
        pw = self._bn(pw_conv)           # [N, C_out, M] (ReLU already in conv_module)

        # Permute back [N, C_out, M] → [N, M, C_out]
        tr_back_op = SimOpHandle(self.name + ".tr_out", "Transpose",
                                 params=[], ipos=[0], perm=[0, 2, 1])
        tr_back_op.implicit_inputs = []
        pointwise = tr_back_op(pw)  # [N, M, C_out]

        if not self.max_out:
            return pointwise

        # ReduceMax over M-dim (dim=1) → [N, 1, C_out]
        axes_t = _from_data(self.name + ".reduce_axes",
                             np.array([1], dtype=np.int64), is_const=True)
        rmax_op = SimOpHandle(self.name + ".reducemax", "ReduceMax",
                              params=[(1, axes_t)], ipos=[0], keepdims=1)
        rmax_op.implicit_inputs.append(axes_t)
        aggregated = rmax_op(pointwise)  # [N, 1, C_out]

        if not self.cat_max:
            return aggregated

        # Tile aggregated → [N, M, C_out]
        repeats_t = _from_data(self.name + ".tile_reps",
                                np.array([1, M, 1], dtype=np.int64), is_const=True)
        tile_op = SimOpHandle(self.name + ".tile", "Tile",
                              params=[(1, repeats_t)], ipos=[0])
        tile_op.implicit_inputs.append(repeats_t)
        repeated = tile_op(aggregated)  # [N, M, C_out]

        # Concatenate along C dim (axis=2)
        cat_out = F.ConcatX(self.name + ".cat_max", axis=2)(pointwise, repeated)
        return cat_out

    def analytical_param_count(self, lvl=0):
        return self.conv_module.analytical_param_count(lvl + 1)


# ---------------------------------------------------------------------------
# DynamicVFELayer
# ---------------------------------------------------------------------------

class DynamicVFELayer(SimNN.Module):
    """
    TTSim implementation of the Dynamic Voxel Feature Encoder layer.

    Operates on un-padded points — input is [M_total, C_in] (all points
    across batch concatenated).  Applies Linear + BN + ReLU, no pooling.

    Graph:
        input [M, C_in]
          → Unsqueeze → [1, C_in, M]   (treat M as spatial dim for Conv1d)
          → ConvModule1d → [1, C_out, M]
          → Squeeze → [M, C_out]
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        norm_cfg: dict = None,
    ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(eps=1e-3, momentum=0.01)

        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        eps = norm_cfg.get('eps', 1e-3)
        momentum = norm_cfg.get('momentum', 0.01)

        self.conv_module = ConvModule1d(
            name + ".conv_module",
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            with_bn=False,   # BN created separately to allow eps/momentum
            with_relu=True,
        )
        # BatchNorm1d with VFE-specific eps / momentum
        from ttsim_modules.mlp import BatchNorm1d as _BN1d
        self._bn = _BN1d(name + ".bn", out_channels, eps=eps, momentum=momentum)

        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [M, C_in] — flat list of all points.

        Returns:
            SimTensor: [M, C_out] — per-point features after Linear+BN+ReLU.
        """
        M, C_in = x.shape

        # Transpose [M, C] → [C, M] then unsqueeze → [1, C, M]
        tr_op = SimOpHandle(self.name + ".tr_in", "Transpose",
                            params=[], ipos=[0], perm=[1, 0])
        tr_op.implicit_inputs = []
        x_t = tr_op(x)  # [C_in, M]

        ax0_t = _from_data(self.name + ".ax0",
                            np.array([0], dtype=np.int64), is_const=True)
        unsq_op = SimOpHandle(self.name + ".unsqueeze", "Unsqueeze",
                              params=[(1, ax0_t)], ipos=[0])
        unsq_op.implicit_inputs.append(ax0_t)
        x_3d = unsq_op(x_t)  # [1, C_in, M]

        pw_conv = self.conv_module(x_3d)   # [1, C_out, M]
        pw = self._bn(pw_conv)             # [1, C_out, M]

        # Squeeze back: [1, C_out, M] → [C_out, M]
        sq_ax_t = _from_data(self.name + ".sq_ax",
                              np.array([0], dtype=np.int64), is_const=True)
        sq_op = SimOpHandle(self.name + ".squeeze", "Squeeze",
                            params=[(1, sq_ax_t)], ipos=[0])
        sq_op.implicit_inputs.append(sq_ax_t)
        x_2d = sq_op(pw)  # [C_out, M]

        tr_back_op = SimOpHandle(self.name + ".tr_out", "Transpose",
                                 params=[], ipos=[0], perm=[1, 0])
        tr_back_op.implicit_inputs = []
        out = tr_back_op(x_2d)  # [M, C_out]
        return out

    def analytical_param_count(self, lvl=0):
        return self.conv_module.analytical_param_count(lvl + 1)
