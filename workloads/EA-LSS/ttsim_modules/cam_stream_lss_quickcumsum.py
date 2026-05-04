#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of QuickCumsum (voxel-pooling helper).

Original file: mmdet3d/models/detectors/cam_stream_lss.py (lines 86-130)

QuickCumsum is a custom ``torch.autograd.Function`` that implements an
efficient cumulative-sum aggregation trick for voxel feature pooling.
Given a sorted list of features (x) and their voxel indices (ranks), it
sums all features that fall into the same voxel in a single CumSum pass:

    1. Compute cumulative sum of features along the point dimension.
    2. Identify voxel boundaries (where ranks[i] != ranks[i-1]).
    3. Keep only boundary rows → one row per voxel.
    4. Recover per-voxel sum via first-difference of the boundary rows.

The backward pass uses a reverse cumsum over the gradient.

TTSim conversion note:
    The output size of QuickCumsum is **data-dependent** (number of
    unique voxels is not known at graph-build time).  Therefore:
      - For shape inference we return a SimTensor with shape
        [V, C] where V = x.shape[0] (worst case: every point is
        in a unique voxel).
      - For data compute we run the full numpy implementation.

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import _from_shape, _from_data


# ---------------------------------------------------------------------------
# cumsum_trick  (pure numpy)
# ---------------------------------------------------------------------------

def cumsum_trick_numpy(
    x: np.ndarray,
    geom_feats: np.ndarray,
    ranks: np.ndarray,
):
    """
    Pure-numpy reimplementation of ``cumsum_trick``.

    Args:
        x          (np.ndarray): [N, C] per-point features sorted by rank.
        geom_feats (np.ndarray): [N, D] geometric features (BEV coordinates).
        ranks      (np.ndarray): [N]   voxel index for each point (sorted).

    Returns:
        tuple:
            x_voxel   np.ndarray [V, C] – aggregated voxel features.
            geom_out  np.ndarray [V, D] – one geom row per voxel.
    """
    x_cum = np.cumsum(x, axis=0)
    kept = np.ones(x.shape[0], dtype=bool)
    kept[:-1] = ranks[1:] != ranks[:-1]

    x_kept = x_cum[kept]
    g_kept = geom_feats[kept]

    # First difference to recover per-voxel sums
    x_voxel = np.concatenate([x_kept[:1], x_kept[1:] - x_kept[:-1]], axis=0)
    return x_voxel, g_kept


# ---------------------------------------------------------------------------
# QuickCumsum  (TTSim module)
# ---------------------------------------------------------------------------

class QuickCumsum(SimNN.Module):
    """
    TTSim module wrapping the QuickCumsum voxel-pooling aggregation.

    Since the output size is data-dependent TTSim shape inference returns
    a worst-case shape: the full N rows (every point in its own voxel).
    Actual data is computed via the pure-numpy implementation above.

    Args:
        name (str): Unique module name prefix.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        super().link_op2module()

    def __call__(self, x, geom_feats, ranks=None):
        """
        Build TTSim graph node for QuickCumsum.

        Args:
            x          (SimTensor): [N, C] per-point features.
            geom_feats (SimTensor): [N, D] geometric features.
            ranks      (SimTensor or None): Not used for shape inference.

        Returns:
            tuple(SimTensor, SimTensor):
                x_out    [N, C] (worst-case shape — actual V <= N)
                geom_out [N, D]
        """
        N, C = x.shape
        _, D = geom_feats.shape if len(geom_feats.shape) == 2 else (N, 1)

        # Shape-only output tensors (data-dependent output size)
        x_out = _from_shape(self.name + ".x_out", [N, C])
        g_out = _from_shape(self.name + ".geom_out", [N, D])

        # Data compute if inputs have data
        if x.data is not None and geom_feats.data is not None and ranks is not None:
            ranks_np = ranks.data if hasattr(ranks, 'data') else ranks
            if ranks_np is not None:
                xv, gv = cumsum_trick_numpy(x.data, geom_feats.data, ranks_np)
                x_out.data = xv.astype(np.float32)
                g_out.data = gv.astype(np.float32)
                # Update shape to actual output size
                x_out.shape = list(xv.shape)
                g_out.shape = list(gv.shape)

        return x_out, g_out

    def analytical_param_count(self, lvl=0):
        return 0  # no learnable parameters
