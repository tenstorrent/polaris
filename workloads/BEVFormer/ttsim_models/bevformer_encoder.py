#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of BEVFormer Encoder.

This module implements the BEVFormer encoder which consists of:
1. BEVFormerEncoder: Sequence of transformer layers for BEV feature encoding
2. BEVFormerLayer: Single transformer layer with temporal and spatial attention
3. Reference point generation for 3D spatial attention and 2D temporal attention
4. Camera projection for multi-view feature aggregation

Converted from PyTorch/mmcv to TTSim (CPU-only, Python 3.13 compatible).
No MMCV dependencies, no CUDA operations.
"""

import sys
import os

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module
from workloads.BEVFormer.ttsim_models.custom_base_transformer_layer import (
    MyCustomBaseTransformerLayer,
)


class BEVFormerEncoder(Module):
    """
    BEVFormer Encoder: Sequence of transformer layers for BEV encoding.

    Implements multi-layer transformer with both temporal self-attention
    (TSA) and spatial cross-attention (SCA) for aggregating multi-camera
    features into bird's-eye-view representation.

    Args:
        name (str): Module name
        transformerlayers (dict): Configuration for transformer layers
        num_layers (int): Number of transformer layers
        pc_range (list): Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points_in_pillar (int): Number of Z-axis sampling points per pillar (default: 4)
        return_intermediate (bool): Whether to return intermediate layer outputs
    """

    def __init__(
        self,
        name,
        transformerlayers,
        num_layers,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = (
            pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        )

        # Build transformer layers
        self.layers = []
        for i in range(num_layers):
            layer = BEVFormerLayer(name=f"{name}.layer_{i}", **transformerlayers)
            self.layers.append(layer)

    @staticmethod
    def get_reference_points(
        H, W, Z=8, num_points_in_pillar=4, dim="3d", bs=1, dtype=np.float32
    ):
        """
        Get the reference points used in SCA and TSA.

        Args:
            H, W: Spatial shape of BEV
            Z: Height of pillar
            num_points_in_pillar: Sample D points uniformly from each pillar
            dim: '3d' for spatial cross-attention, '2d' for temporal self-attention
            bs: Batch size
            dtype: Data type for numpy arrays

        Returns:
            Reference points tensor with shape:
            - 3d: [bs, num_points_in_pillar, H*W, 3] for (x, y, z)
            - 2d: [bs, H*W, 1, 2] for (x, y)
        """
        if dim == "3d":
            # Reference points in 3D space for spatial cross-attention (SCA)
            # Create normalized coordinates [0, 1]
            zs = np.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype) / Z
            zs = zs.reshape(-1, 1, 1)
            zs = np.broadcast_to(zs, (num_points_in_pillar, H, W))

            xs = np.linspace(0.5, W - 0.5, W, dtype=dtype) / W
            xs = xs.reshape(1, 1, W)
            xs = np.broadcast_to(xs, (num_points_in_pillar, H, W))

            ys = np.linspace(0.5, H - 0.5, H, dtype=dtype) / H
            ys = ys.reshape(1, H, 1)
            ys = np.broadcast_to(ys, (num_points_in_pillar, H, W))

            # Stack to [num_points_in_pillar, H, W, 3]
            ref_3d = np.stack([xs, ys, zs], axis=-1)

            # Reshape to [num_points_in_pillar, 3, H, W] -> [num_points_in_pillar, 3, H*W] -> [num_points_in_pillar, H*W, 3]
            ref_3d = ref_3d.transpose(0, 3, 1, 2)  # [D, 3, H, W]
            ref_3d = ref_3d.reshape(num_points_in_pillar, 3, -1)  # [D, 3, H*W]
            ref_3d = ref_3d.transpose(0, 2, 1)  # [D, H*W, 3]

            # Add batch dimension and repeat: [bs, D, H*W, 3]
            ref_3d = np.broadcast_to(
                ref_3d[np.newaxis, :, :, :], (bs, num_points_in_pillar, H * W, 3)
            )

            return ref_3d

        elif dim == "2d":
            # Reference points on 2D BEV plane for temporal self-attention (TSA)
            ref_y = np.linspace(0.5, H - 0.5, H, dtype=dtype) / H
            ref_x = np.linspace(0.5, W - 0.5, W, dtype=dtype) / W

            # Create meshgrid
            ref_y_grid, ref_x_grid = np.meshgrid(ref_y, ref_x, indexing="ij")

            # Flatten and stack: [H*W, 2]
            ref_y_flat = ref_y_grid.reshape(-1)
            ref_x_flat = ref_x_grid.reshape(-1)
            ref_2d = np.stack([ref_x_flat, ref_y_flat], axis=-1)  # [H*W, 2]

            # Add dimensions: [bs, H*W, 1, 2]
            ref_2d = ref_2d[np.newaxis, :, np.newaxis, :]
            ref_2d = np.broadcast_to(ref_2d, (bs, H * W, 1, 2))

            return ref_2d

        else:
            raise ValueError(f"dim must be '3d' or '2d', got {dim}")

    @staticmethod
    def point_sampling(reference_points, pc_range, img_metas):
        """
        Project 3D reference points to camera image coordinates.

        This function transforms BEV reference points from normalized coordinates
        to actual 3D space, then projects them to each camera view using the
        lidar-to-image transformation matrices.

        Args:
            reference_points: numpy array [bs, D, H*W, 3] with normalized coords [0,1]
            pc_range: list [x_min, y_min, z_min, x_max, y_max, z_max]
            img_metas: list of dicts containing 'lidar2img' transformation matrices

        Returns:
            reference_points_cam: numpy array [num_cam, bs, H*W, D, 2] - projected 2D coords
            bev_mask: numpy array [num_cam, bs, H*W, D] - visibility mask (bool)
        """
        # Extract lidar2img matrices from metadata
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.array(lidar2img)  # type: ignore[assignment]  # [bs, num_cam, 4, 4]

        # Denormalize reference points to actual 3D coordinates
        reference_points = reference_points.copy()
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        # Add homogeneous coordinate: [bs, D, H*W, 4]
        ones = np.ones_like(reference_points[..., :1])
        reference_points = np.concatenate([reference_points, ones], axis=-1)

        # Permute to [D, bs, H*W, 4]
        reference_points = reference_points.transpose(1, 0, 2, 3)
        D, B, num_query = reference_points.shape[:3]
        num_cam = lidar2img.shape[1]  # type: ignore[attr-defined]

        # Expand for all cameras: [D, bs, num_cam, H*W, 4, 1]
        reference_points = reference_points[:, :, np.newaxis, :, :, np.newaxis]  # type: ignore[call-overload]
        reference_points = np.broadcast_to(reference_points, (D, B, num_cam, num_query, 4, 1))  # type: ignore[assignment]

        # Expand lidar2img: [D, bs, num_cam, H*W, 4, 4]
        lidar2img = lidar2img[np.newaxis, :, :, np.newaxis, :, :]  # type: ignore[call-overload]
        lidar2img = np.broadcast_to(lidar2img, (D, B, num_cam, num_query, 4, 4))  # type: ignore[assignment]

        # Project to camera coordinates: [D, bs, num_cam, H*W, 4, 1]
        reference_points_cam = np.matmul(lidar2img, reference_points).squeeze(-1)

        eps = 1e-5

        # Check if points are in front of camera (z > 0)
        bev_mask = reference_points_cam[..., 2:3] > eps

        # Perspective division: x/z, y/z
        z_coord = reference_points_cam[..., 2:3]
        z_coord_safe = np.maximum(z_coord, np.ones_like(z_coord) * eps)
        reference_points_cam = reference_points_cam[..., 0:2] / z_coord_safe

        # Normalize by image size
        img_h, img_w = img_metas[0]["img_shape"][0][:2]
        reference_points_cam[..., 0] /= img_w
        reference_points_cam[..., 1] /= img_h

        # Update mask: point must be in front and within image bounds
        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )

        # Handle NaN values
        bev_mask = np.nan_to_num(bev_mask, nan=0.0)

        # Permute to [num_cam, bs, H*W, D, 2] and [num_cam, bs, H*W, D]
        reference_points_cam = reference_points_cam.transpose(2, 1, 3, 0, 4)
        bev_mask = bev_mask.transpose(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    def __call__(
        self,
        bev_query,
        key,
        value,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=None,
        img_metas=None,
        **kwargs,
    ):
        """
        Forward function for BEVFormerEncoder.

        Args:
            bev_query: BEV query tensor [bs, num_query, embed_dims]
            key: Multi-camera features [num_cam, num_value, bs, embed_dims]
            value: Multi-camera features (same as key)
            bev_h, bev_w: Spatial dimensions of BEV grid
            bev_pos: Positional encoding for BEV [bs, num_query, embed_dims]
            spatial_shapes: List of (H, W) for each feature pyramid level
            level_start_index: Start indices for each level
            prev_bev: Previous BEV features for temporal attention
            shift: Spatial shift for temporal attention
            img_metas: Image metadata containing camera parameters

        Returns:
            output: Encoded BEV features [bs, num_query, embed_dims]
            or list of intermediate outputs if return_intermediate=True
        """
        output = bev_query
        intermediate = []

        bs = bev_query.shape[0]

        # Generate reference points
        Z = self.pc_range[5] - self.pc_range[2]
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            Z,
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            dtype=np.float32,
        )

        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim="2d", bs=bs, dtype=np.float32
        )

        # Project 3D reference points to camera coordinates
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas
        )

        # Handle shift for temporal attention (bug kept for paper reproduction)
        shift_ref_2d = ref_2d.copy()
        if shift is not None:
            # Check if we're building a TTSim graph or working with numpy
            if hasattr(shift, "op_in") and hasattr(
                shift_ref_2d, "op_in"
            ):  # Both TTSim tensors
                # Use TTSim operations to reshape and add
                shift_reshaped = F.Reshape(self.name + ".shift_reshape")(
                    shift,
                    F._from_data(
                        self.name + ".shift_reshape_shape",
                        np.array([shift.data.shape[0], 1, 1, 2], dtype=np.int64),
                        is_const=True,
                    ),
                )
                shift_ref_2d = F.Add(self.name + ".shift_ref_2d_add")(
                    shift_ref_2d, shift_reshaped
                )
            elif not hasattr(shift, "op_in") and not hasattr(
                shift_ref_2d, "op_in"
            ):  # Both numpy
                shift_ref_2d = shift_ref_2d + shift[:, np.newaxis, np.newaxis, :]
            else:  # Mixed - need to convert
                # If ref_2d is numpy and shift is TTSim, or vice versa, something is wrong
                # For now, assume both should be same type, but if not, just skip shift
                import warnings

                warnings.warn(
                    "Mixed tensor types detected for shift operation - skipping shift"
                )
                shift_ref_2d = ref_2d.copy()

        # Prepare hybrid reference points for temporal attention
        len_bev = bev_h * bev_w
        num_bev_level = ref_2d.shape[2]

        if prev_bev is not None:
            # Stack current and previous BEV: [bs*2, len_bev, embed_dims]
            prev_bev_stacked = np.concatenate([prev_bev, bev_query], axis=0)
            # Stack shifted and current reference points: [bs*2, len_bev, num_bev_level, 2]
            hybird_ref_2d = np.concatenate([shift_ref_2d, ref_2d], axis=0)
        else:
            prev_bev_stacked = None
            # Stack current reference points twice: [bs*2, len_bev, num_bev_level, 2]
            hybird_ref_2d = np.concatenate([ref_2d, ref_2d], axis=0)

        # Forward through transformer layers
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev_stacked,
                **kwargs,
            )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate

        return output

    def analytical_param_count(self):
        """Calculate total parameter count."""
        total = 0
        for layer in self.layers:
            total += layer.analytical_param_count()
        return total


class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """
    Single BEVFormer transformer layer.

    Implements a transformer layer with:
    - Temporal self-attention (TSA): Attends to previous BEV frames
    - Spatial cross-attention (SCA): Aggregates multi-camera features
    - Feed-forward network (FFN): Non-linear transformation
    - Layer normalization between operations

    The operation order is fixed as: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    Args:
        name (str): Module name
        attn_cfgs (list): Attention configurations [temporal_attn, spatial_attn]
        feedforward_channels (int): Hidden dimension for FFN
        ffn_dropout (float): Dropout rate (ignored in inference)
        operation_order (tuple): Fixed as ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        act_cfg (dict): Activation config (default: ReLU)
        norm_cfg (dict): Normalization config (default: LayerNorm)
        ffn_num_fcs (int): Number of FC layers in FFN (default: 2)
    """

    def __init__(
        self,
        name,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=None,
        norm_cfg=None,
        ffn_num_fcs=2,
        **kwargs,
    ):

        # Set defaults
        if operation_order is None:
            operation_order = ("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
        if act_cfg is None:
            act_cfg = dict(type="ReLU", inplace=True)
        if norm_cfg is None:
            norm_cfg = dict(type="LN")

        # Validate operation order
        assert len(operation_order) == 6, "BEVFormerLayer requires exactly 6 operations"
        assert set(operation_order) == set(
            ["self_attn", "norm", "cross_attn", "ffn"]
        ), "BEVFormerLayer operations must be self_attn, norm, cross_attn, ffn"

        super().__init__(
            name=name,
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )

    def __call__(  # type: ignore[override]
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):
        """
        Forward function for BEVFormerLayer.

        Args:
            query: Input BEV query [bs, num_queries, embed_dims]
            key: Multi-camera features for cross-attention
            value: Multi-camera features for cross-attention
            bev_pos: Positional encoding for BEV
            ref_2d: 2D reference points for temporal attention
            ref_3d: 3D reference points for spatial attention
            bev_h, bev_w: BEV spatial dimensions
            reference_points_cam: Projected camera coordinates
            mask: Visibility mask
            spatial_shapes: Feature pyramid spatial shapes
            level_start_index: Level start indices
            prev_bev: Previous BEV for temporal attention
            attn_masks: Attention masks
            query_key_padding_mask: Query padding mask
            key_padding_mask: Key padding mask

        Returns:
            query: Transformed BEV features [bs, num_queries, embed_dims]
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(len(self.attentions))]
        elif not isinstance(attn_masks, list):
            attn_masks = [attn_masks for _ in range(len(self.attentions))]

        for layer in self.operation_order:
            if layer == "self_attn":
                # Temporal self-attention
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=[(bev_h, bev_w)],
                    level_start_index=[0],
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                # Spatial cross-attention
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
