#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of VFE (Voxel Feature Encoder) — HardSimpleVFE and HardSimpleVFE_ATT.

Original file: mmdet3d/models/voxel_encoders/voxel_encoder.py

HardSimpleVFE:
    Simple mean over points in a voxel.
    Input:  [N, M, num_features]  (N voxels, M points each)
    Output: [N, num_features]     (mean features)
    Params: 0

HardSimpleVFE_ATT:
    Decorates raw point features (cluster center, pillar center, num_points),
    applies temporal attention (VoxelFeature_TA) for contextual aggregation,
    then compresses each voxel to a fixed-size descriptor via PFNLayer.

    VoxelFeature_TA:
        Two rounds of:
        - PACALayer (point-wise x channel-wise attention via PALayer + CALayer)
        - VALayer   (voxel-wise attention)
        - FC projection
    PFNLayer: Linear + BN1d + ReLU + ReduceMax -> [N, 1, out_filters] -> squeeze

    Default config:
        num_features=5, dim_ca=12, dim_pa=10, reduction_r=8, boost_c_dim=32
        PFN: 32 -> 32

    Input:  [N, M, 5]   (raw point cloud voxel features)
    Output: [N, 32]     (compressed voxel descriptors)
    Params: 4322

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

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim_modules.mlp import BatchNorm1d, ConvModule1d
from ttsim_modules.swin_transformer import _LinearModule


# ---------------------------------------------------------------------------
# Helpers — verified patterns from voxel_encoder_utils.py
# ---------------------------------------------------------------------------

def _transpose3(name, x, perm):
    """Transpose a 3-D tensor x using given perm via SimOpHandle."""
    op = SimOpHandle(name, "Transpose", params=[], ipos=[0], perm=perm)
    op.implicit_inputs = []
    return op(x)


def _reduce_max_axis(name, x, axis: int, keepdims: int = 1):
    """ReduceMax over a single axis."""
    axes_t = _from_data(name + ".ax", np.array([axis], dtype=np.int64), is_const=True)
    op = SimOpHandle(name + ".rmax", "ReduceMax",
                     params=[(1, axes_t)], ipos=[0], keepdims=keepdims)
    op.implicit_inputs.append(axes_t)
    return op(x)


def _unsqueeze(name, x, axis: int):
    """Insert a size-1 dim at axis."""
    ax_t = _from_data(name + ".ax", np.array([axis], dtype=np.int64), is_const=True)
    op = SimOpHandle(name + ".unsq", "Unsqueeze", params=[(1, ax_t)], ipos=[0])
    op.implicit_inputs.append(ax_t)
    return op(x)


def _squeeze(name, x, axis: int):
    """Remove a size-1 dim at axis."""
    ax_t = _from_data(name + ".ax", np.array([axis], dtype=np.int64), is_const=True)
    op = SimOpHandle(name + ".sq", "Squeeze", params=[(1, ax_t)], ipos=[0])
    op.implicit_inputs.append(ax_t)
    return op(x)


def _tile(name, x, repeats):
    """Tile x by repeats (list of ints)."""
    rep_t = _from_data(name + ".reps", np.array(repeats, dtype=np.int64), is_const=True)
    op = SimOpHandle(name + ".tile", "Tile", params=[(1, rep_t)], ipos=[0])
    op.implicit_inputs.append(rep_t)
    return op(x)


# ---------------------------------------------------------------------------
# PALayer  (point-axis attention)
# ---------------------------------------------------------------------------

class _PALayer(SimNN.Module):
    """
    Point-axis attention layer.
    x [N, M, C] -> max over C -> [N, M] -> FC(M, mid) -> FC(mid, M) -> [N, M, 1]

    Parameters: 2 x (dim_pa * mid + mid + mid * dim_pa + dim_pa)
    With default dim_pa=10, reduction_pa=1:
        fc1: Linear(10, 10) + bias = 110
        fc2: Linear(10, 10) + bias = 110
        Total: 220
    """

    def __init__(self, name: str, dim_pa: int, reduction_pa: int):
        super().__init__()
        self.name = name
        self.dim_pa = dim_pa
        self.mid = max(1, dim_pa // reduction_pa)

        self.fc1 = _LinearModule(name + ".fc1", dim_pa, self.mid, bias=True)
        self.fc2 = _LinearModule(name + ".fc2", self.mid, dim_pa, bias=True)
        super().link_op2module()

    def __call__(self, x):
        # x: [N, M, C]
        # max over C (dim=2) keepdims=0 -> [N, M]
        y = _reduce_max_axis(self.name + ".maxC", x, axis=2, keepdims=0)
        # FC over the M dimension: [N, M] -> [N, mid] -> [N, M]
        y = self.fc1(y)
        y = SimOpHandle(self.name + ".relu1", "Relu", params=[], ipos=[0])(y)
        y = self.fc2(y)                    # [N, M]
        # unsqueeze last dim: [N, M] -> [N, M, 1]
        y = _unsqueeze(self.name + ".us", y, axis=2)
        return y                           # [N, M, 1]

    def analytical_param_count(self, lvl=0):
        return self.fc1.analytical_param_count() + self.fc2.analytical_param_count()


# ---------------------------------------------------------------------------
# CALayer  (channel-axis attention)
# ---------------------------------------------------------------------------

class _CALayer(SimNN.Module):
    """
    Channel-axis attention layer.
    x [N, M, C] -> max over M -> [N, C] -> FC(C, mid) -> FC(mid, C) -> [N, 1, C]

    With default dim_ca=12, reduction_ca=1:
        fc1: Linear(12, 12) + bias = 156
        fc2: Linear(12, 12) + bias = 156
        Total: 312
    """

    def __init__(self, name: str, dim_ca: int, reduction_ca: int):
        super().__init__()
        self.name = name
        self.dim_ca = dim_ca
        self.mid = max(1, dim_ca // reduction_ca)

        self.fc1 = _LinearModule(name + ".fc1", dim_ca, self.mid, bias=True)
        self.fc2 = _LinearModule(name + ".fc2", self.mid, dim_ca, bias=True)
        super().link_op2module()

    def __call__(self, x):
        # x: [N, M, C]
        # max over M (dim=1) keepdims=0 -> [N, C]
        y = _reduce_max_axis(self.name + ".maxM", x, axis=1, keepdims=0)
        y = self.fc1(y)
        y = SimOpHandle(self.name + ".relu1", "Relu", params=[], ipos=[0])(y)
        y = self.fc2(y)                     # [N, C]
        # unsqueeze dim 1: [N, C] -> [N, 1, C]
        y = _unsqueeze(self.name + ".us", y, axis=1)
        return y                            # [N, 1, C]

    def analytical_param_count(self, lvl=0):
        return self.fc1.analytical_param_count() + self.fc2.analytical_param_count()


# ---------------------------------------------------------------------------
# PACALayer (point-channel attention)
# ---------------------------------------------------------------------------

class _PACALayer(SimNN.Module):
    """
    Combined point-channel attention: PA x CA applied element-wise with sigmoid.
    x [N, M, C] -> PACA weight -> sigmoid -> x * weight

    Parameters: PA + CA params
    """

    def __init__(self, name: str, dim_ca: int, dim_pa: int, reduction_r: int):
        super().__init__()
        self.name = name
        self.pa = _PALayer(name + ".pa", dim_pa, max(1, dim_pa // reduction_r))
        self.ca = _CALayer(name + ".ca", dim_ca, max(1, dim_ca // reduction_r))
        super().link_op2module()

    def __call__(self, x):
        pa_w = self.pa(x)                               # [N, M, 1]
        ca_w = self.ca(x)                               # [N, 1, C]

        # Broadcast multiply: [N, M, 1] * [N, 1, C] -> [N, M, C]
        paca_w = SimOpHandle(self.name + ".mul_paca", "Mul",
                             params=[], ipos=[0, 1])(pa_w, ca_w)
        paca_n = F.Sigmoid(self.name + ".sig")(paca_w)  # [N, M, C]
        out = SimOpHandle(self.name + ".mul_x", "Mul",
                          params=[], ipos=[0, 1])(x, paca_n)
        return out, paca_n                              # both [N, M, C]

    def analytical_param_count(self, lvl=0):
        return self.pa.analytical_param_count() + self.ca.analytical_param_count()


# ---------------------------------------------------------------------------
# VALayer  (voxel-axis attention)
# ---------------------------------------------------------------------------

class _VALayer(SimNN.Module):
    """
    Voxel-wise attention.
    Inputs:
        voxel_center [N, 1, 3]
        paca_feat    [N, M, C]
    Output: weight [N, 1, 1]

    Graph:
        cat([paca_feat, tile(voxel_center, M)]) -> [N, M, C+3]
        fc1: Linear(C+3, 1) + ReLU  -> [N, M, 1]
        transpose(perm=[0,2,1])      -> [N, 1, M]
        fc2: Linear(M, 1)            -> [N, 1, 1]
        Sigmoid                      -> weight

    Parameters (c_num=12, p_num=10): fc1=(12+3)*1+1=16, fc2=10+1=11 -> 27
    """

    def __init__(self, name: str, c_num: int, p_num: int):
        super().__init__()
        self.name = name
        self.c_num = c_num
        self.p_num = p_num

        self.fc1 = _LinearModule(name + ".fc1", c_num + 3, 1, bias=True)
        self.fc2 = _LinearModule(name + ".fc2", p_num, 1, bias=True)
        super().link_op2module()

    def __call__(self, voxel_center, paca_feat):
        # voxel_center: [N, 1, 3], paca_feat: [N, M, C]
        N, M, C = paca_feat.shape

        # Tile voxel_center to [N, M, 3]
        ctr_tiled = _tile(self.name + ".tile", voxel_center, [1, M, 1])

        # Concat [N, M, C+3]
        concat = F.ConcatX(self.name + ".cat", axis=2)(paca_feat, ctr_tiled)

        # fc1: [N, M, C+3] -> [N, M, 1]
        y = self.fc1(concat)
        y = SimOpHandle(self.name + ".relu1", "Relu", params=[], ipos=[0])(y)

        # transpose [N, M, 1] -> [N, 1, M]
        y_t = _transpose3(self.name + ".tr", y, perm=[0, 2, 1])

        # fc2: [N, 1, M] -> [N, 1, 1]
        y2 = self.fc2(y_t)
        weight = F.Sigmoid(self.name + ".sig")(y2)
        return weight                               # [N, 1, 1]

    def analytical_param_count(self, lvl=0):
        return self.fc1.analytical_param_count() + self.fc2.analytical_param_count()


# ---------------------------------------------------------------------------
# VoxelFeature_TA
# ---------------------------------------------------------------------------

class _VoxelFeature_TA(SimNN.Module):
    """
    Temporal attention aggregator for voxel features.

    Hardcoded config matching EALSS source:
        dim_ca=12, dim_pa=10, reduction_r=8, boost_c_dim=32

    Input:
        voxel_center [N, 1, 3]
        x            [N, M, 12]   (decorated features)
    Output: [N, M, 32]

    Parameters: PACALayer1(532) + VALayer1(27) + FC1(800)
                + PACALayer2(772) + VALayer2(47) + FC2(1056)  = 3234
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        dim_ca, dim_pa, reduction_r, boost = 12, 10, 8, 32

        self.paca1 = _PACALayer(name + ".paca1", dim_ca, dim_pa, reduction_r)
        self.va1   = _VALayer(name + ".va1", c_num=dim_ca, p_num=dim_pa)
        self.fc1   = _LinearModule(name + ".fc1", 2 * dim_ca, boost, bias=True)

        self.paca2 = _PACALayer(name + ".paca2", boost, dim_pa, reduction_r)
        self.va2   = _VALayer(name + ".va2", c_num=boost, p_num=dim_pa)
        self.fc2   = _LinearModule(name + ".fc2", boost, boost, bias=True)

        super().link_op2module()

    def __call__(self, voxel_center, x):
        # Round 1
        paca1_out, _  = self.paca1(x)                           # [N, M, 12]
        va1_w          = self.va1(voxel_center, paca1_out)       # [N, 1, 1]
        paca1_feat     = SimOpHandle(self.name + ".mul1", "Mul",
                                     params=[], ipos=[0, 1])(va1_w, paca1_out)    # [N, M, 12]
        out1 = F.ConcatX(self.name + ".cat1", axis=2)(paca1_feat, x)             # [N, M, 24]
        out1 = self.fc1(out1)                                    # [N, M, 32]
        out1 = SimOpHandle(self.name + ".relu1", "Relu", params=[], ipos=[0])(out1)

        # Round 2
        paca2_out, _  = self.paca2(out1)                         # [N, M, 32]
        va2_w          = self.va2(voxel_center, paca2_out)       # [N, 1, 1]
        paca2_feat     = SimOpHandle(self.name + ".mul2", "Mul",
                                     params=[], ipos=[0, 1])(va2_w, paca2_out)    # [N, M, 32]
        out2 = SimOpHandle(self.name + ".add2", "Add",
                            params=[], ipos=[0, 1])(out1, paca2_feat)             # [N, M, 32]
        out2 = self.fc2(out2)
        out  = SimOpHandle(self.name + ".relu2", "Relu", params=[], ipos=[0])(out2)
        return out                                                # [N, M, 32]

    def analytical_param_count(self, lvl=0):
        return (self.paca1.analytical_param_count()
                + self.va1.analytical_param_count()
                + self.fc1.analytical_param_count()
                + self.paca2.analytical_param_count()
                + self.va2.analytical_param_count()
                + self.fc2.analytical_param_count())


# ---------------------------------------------------------------------------
# PFNLayer
# ---------------------------------------------------------------------------

class _PFNLayer(SimNN.Module):
    """
    Pillar Feature Net layer.

    Input  [N, M, in_channels]
    -> Transpose -> [N, in_ch, M]
    -> Conv1d(in, out, k=1, no bias) + BN1d + ReLU -> [N, out, M]
    -> Transpose -> [N, M, out]
    -> ReduceMax over M -> [N, 1, out]

    Parameters: in_ch * out_ch (conv weight, no bias) + 2 * out_ch (BN) = 1088 for 32->32
    """

    def __init__(self, name: str, in_channels: int, out_channels: int):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ConvModule1d (bias=False) + separate BN
        self.conv = ConvModule1d(name + ".conv", in_channels, out_channels,
                                 kernel_size=1, with_bn=False, with_relu=True, bias=False)
        self.bn = BatchNorm1d(name + ".bn", out_channels)
        super().link_op2module()

    def __call__(self, x):
        # x: [N, M, C_in]
        x_t = _transpose3(self.name + ".tr_in", x, perm=[0, 2, 1])  # [N, C_in, M]
        y   = self.conv(x_t)                                          # [N, out, M]
        y   = self.bn(y)                                              # [N, out, M]
        y   = _transpose3(self.name + ".tr_out", y, perm=[0, 2, 1])  # [N, M, out]
        y   = _reduce_max_axis(self.name + ".rmax", y, axis=1, keepdims=1)  # [N, 1, out]
        return y

    def analytical_param_count(self, lvl=0):
        return self.in_channels * self.out_channels + 2 * self.out_channels


# ---------------------------------------------------------------------------
# HardSimpleVFE
# ---------------------------------------------------------------------------

class HardSimpleVFE(SimNN.Module):
    """
    Simple mean-pooling voxel encoder.

    Input:  [N, M, num_features]
    Output: [N, num_features]
    Parameters: 0
    """

    def __init__(self, name: str, num_features: int = 4):
        super().__init__()
        self.name = name
        self.num_features = num_features
        super().link_op2module()

    def __call__(self, features, num_points=None):
        N, M, C = features.shape
        out = _reduce_max_axis(self.name + ".mean", features, axis=1, keepdims=0)
        return out      # [N, C]  (approximate mean by max for TTSim FLOPs parity)

    def analytical_param_count(self, lvl=0):
        return 0


# ---------------------------------------------------------------------------
# HardSimpleVFE_ATT
# ---------------------------------------------------------------------------

class HardSimpleVFE_ATT(SimNN.Module):
    """
    Attention-augmented voxel encoder (EALSS default config).

    Input:  [N, M=10, 5]
    Output: [N, 32]
    Parameters: 4322 (VoxelFeature_TA: 3234 + PFNLayer: 1088)
    """

    def __init__(self, name: str, num_features: int = 5, max_points: int = 10):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.max_points = max_points

        self.vfe_ta = _VoxelFeature_TA(name + ".vfe_ta")
        self.pfn    = _PFNLayer(name + ".pfn", in_channels=32, out_channels=32)

        super().link_op2module()

    def __call__(self, features, num_points=None):
        """
        Args:
            features (SimTensor): [N, M, 5]
        Returns:
            SimTensor: [N, 32]
        """
        N, M, C = features.shape

        # ---- Feature decoration: [N, M, 5] -> [N, M, 12] ----
        # cluster offset: subtract per-voxel mean position
        clust_ax = _from_data(self.name + ".ca", np.array([1], dtype=np.int64), is_const=True)
        pmean_op = SimOpHandle(self.name + ".pmean", "ReduceMean",
                               params=[(1, clust_ax)], ipos=[0], keepdims=1)
        pmean_op.implicit_inputs.append(clust_ax)
        pts_mean = pmean_op(features)

        f_cluster = SimOpHandle(self.name + ".clust_sub", "Sub",
                                params=[], ipos=[0, 1])(features, pts_mean)     # [N, M, 5]

        # Pillar-center + num-pts placeholder: [N, M, 4]
        f_extra = _from_shape(self.name + ".f_extra", [N, M, 4])

        # concat -> [N, M, 9] then concat with original [N, M, 5] = [N, M, .. ] but
        # EALSS decorates [x,y,z,r + dx,dy,dz + px,py,pz + n] = 5+3+3+1 = 12
        # We approximate: features[5] + f_cluster[N,M,3] + f_extra[N,M,4]  = [N,M,12]
        # Use only first 3 cols of f_cluster as positional offsets
        f_clust3 = _from_shape(self.name + ".f_clust3", [N, M, 3])  # proxy for slice
        voxel_feats = F.ConcatX(self.name + ".decorate", axis=2)(
            features, f_clust3, f_extra
        )                                                               # [N, M, 12]

        # voxel_center proxy [N, 1, 3]
        voxel_center = _from_shape(self.name + ".vctr", [N, 1, 3])

        # ---- VoxelFeature_TA -> [N, M, 32] ----
        agg = self.vfe_ta(voxel_center, voxel_feats)

        # ---- PFNLayer -> [N, 1, 32] ----
        out = self.pfn(agg)

        # Squeeze dim 1 -> [N, 32]
        out_sq = _squeeze(self.name + ".sq_out", out, axis=1)
        return out_sq

    def analytical_param_count(self, lvl=0):
        return self.vfe_ta.analytical_param_count() + self.pfn.analytical_param_count()
