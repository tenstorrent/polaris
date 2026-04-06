#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Perception Transformer for BEVFormer.

This module implements the top-level transformer that orchestrates the
BEV encoder and detection decoder for end-to-end 3D object detection.

Original: projects/mmdet3d_plugin/bevformer/modules/transformer.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270
"""

import numpy as np
import warnings

# TTSim imports
from ttsim.front.functional import op as F
from ttsim.front.functional import sim_nn as F_nn
from ttsim.front.functional.sim_nn import Module

# Import converted modules
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention


class PerceptionTransformer(Module):
    """Perception Transformer for BEVFormer (TTSim version).

    Orchestrates the BEV encoder and detection decoder to transform
    multi-camera image features into 3D object detections.

    Args:
        encoder (Module): Pre-built BEVFormer encoder module.
        decoder (Module): Pre-built detection transformer decoder module.
        embed_dims (int): Embedding dimension. Default: 256.
        num_feature_levels (int): Number of feature pyramid levels. Default: 4.
        num_cams (int): Number of cameras. Default: 6.
        rotate_prev_bev (bool): Whether to rotate previous BEV by ego motion. Default: True.
        use_shift (bool): Whether to shift BEV by ego translation. Default: True.
        use_can_bus (bool): Whether to use CAN bus signals. Default: True.
        can_bus_norm (bool): Whether to normalize CAN bus with LayerNorm. Default: True.
        use_cams_embeds (bool): Whether to add camera embeddings. Default: True.
        rotate_center (list): Center point for BEV rotation [x, y]. Default: [100, 100].
        name (str): Module name. Default: None.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        embed_dims=256,
        num_feature_levels=4,
        num_cams=6,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=None,
        name=None,
    ):

        super().__init__()
        self.name = name if name else "perception_transformer"

        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center if rotate_center is not None else [100, 100]

        self.init_layers()

    def init_layers(self):
        """Initialize learnable parameters and sub-modules."""
        # Level embeddings for multi-scale features (num_feature_levels, embed_dims)
        # These will be loaded from pre-trained weights
        self.level_embeds = None  # Placeholder for (num_feature_levels, embed_dims)

        # Camera embeddings (num_cams, embed_dims)
        self.cams_embeds = None  # Placeholder for (num_cams, embed_dims)

        # Reference points prediction from query embeddings
        self.reference_points = F_nn.Linear(
            name=f"{self.name}_reference_points",
            in_features=self.embed_dims,
            out_features=3,  # (x, y, z) coordinates
            bias=True,
        )

        # CAN bus MLP: 18 -> embed_dims//2 -> embed_dims
        self.can_bus_fc1 = F_nn.Linear(
            name=f"{self.name}_can_bus_fc1",
            in_features=18,
            out_features=self.embed_dims // 2,
            bias=True,
        )
        self.can_bus_fc2 = F_nn.Linear(
            name=f"{self.name}_can_bus_fc2",
            in_features=self.embed_dims // 2,
            out_features=self.embed_dims,
            bias=True,
        )

        if self.can_bus_norm:
            self.can_bus_norm_layer = F.LayerNorm(
                f"{self.name}_can_bus_norm",
                self.embed_dims,
                epsilon=1e-5,
            )

    def init_weights(self):
        """Initialize weights (placeholder for TTSim inference).

        In TTSim, weights are loaded from pre-trained checkpoints.
        This method exists for API compatibility.
        """
        pass

    def get_bev_features(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=None,
        bev_pos=None,
        prev_bev=None,
        img_metas=None,
        **kwargs,
    ):
        """Extract BEV features from multi-camera multi-scale features.

        Args:
            mlvl_feats (list[Tensor]): Multi-level multi-camera features.
                Each element has shape [bs, num_cams, C, H, W].
            bev_queries (Tensor): BEV query embeddings with shape [bev_h*bev_w, embed_dims].
            bev_h (int): Height of BEV grid.
            bev_w (int): Width of BEV grid.
            grid_length (list): Physical size of each BEV grid cell [y, x] in meters. Default: [0.512, 0.512].
            bev_pos (Tensor): BEV positional encodings with shape [bs, embed_dims, bev_h, bev_w].
            prev_bev (Tensor): Previous BEV features for temporal fusion. Default: None.
            img_metas (list[dict]): Image meta information including CAN bus signals.

        Returns:
            Tensor: BEV features with shape [bs, bev_h*bev_w, embed_dims].
        """
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Get batch size from first feature level
        mlvl_feats_shape = F.Shape(
            mlvl_feats[0], name=f"{self.name}_mlvl_feats_shape"
        )
        bs = F.SliceF(mlvl_feats_shape, [0], [1], [0], name=f"{self.name}_bs")  # type: ignore[call-arg,misc]

        # Expand bev_queries to batch: [bev_h*bev_w, embed_dims] -> [bev_h*bev_w, bs, embed_dims]
        bev_queries = F.Unsqueeze(
            bev_queries, [1], name=f"{self.name}_bev_queries_unsqueeze"
        )
        # Repeat for batch size
        # Note: Tile operation repeats along dimension
        bev_queries = F.Tile(
            bev_queries, [1, bs, 1], name=f"{self.name}_bev_queries_tile"
        )

        # Reshape bev_pos: [bs, embed_dims, bev_h, bev_w] -> [bs, embed_dims, bev_h*bev_w]
        bev_pos = F.Reshape(
            bev_pos,
            [
                bs,
                F._from_data(
                    f"{self.name}_embed_dims",
                    np.array([self.embed_dims]),
                    is_const=True,
                ),
                F._from_data(
                    f"{self.name}_bev_hw", np.array([bev_h * bev_w]), is_const=True
                ),
            ],
            name=f"{self.name}_bev_pos_reshape",
        )
        # Permute to [bev_h*bev_w, bs, embed_dims]
        bev_pos = F.Transpose(
            bev_pos, perm=[2, 0, 1], name=f"{self.name}_bev_pos_transpose"
        )

        # Process ego motion (shift and rotation)
        # Note: In TTSim, img_metas would need to be passed as tensors
        # For now, we'll assume shift is passed as a tensor [bs, 2]
        shift = kwargs.get("shift", None)
        if shift is None:
            # Create zero shift if not provided
            shift = F._from_data(
                f"{self.name}_zero_shift",
                np.zeros((1, 2), dtype=np.float32),
                is_const=True,
            )

        # Handle previous BEV rotation if provided
        if prev_bev is not None and self.rotate_prev_bev:
            # Apply rotation to prev_bev
            # In TTSim, we would need to implement rotation using affine transform or interpolation
            # For now, pass through without rotation (would need GridSample-based rotation)
            warnings.warn(
                "BEV rotation not fully implemented in TTSim version. Using unrotated prev_bev."
            )
            # prev_bev shape: [bev_h*bev_w, bs, embed_dims] or [bs, bev_h*bev_w, embed_dims]
            pass

        # Process CAN bus signals if provided
        if self.use_can_bus and kwargs.get("can_bus") is not None:
            can_bus = kwargs["can_bus"]  # [bs, 18]

            # CAN bus MLP: 18 -> embed_dims//2 -> embed_dims
            can_bus_features = self.can_bus_fc1(can_bus)
            can_bus_features = F.Relu(f"{self.name}_can_bus_relu1")(can_bus_features)

            can_bus_features = self.can_bus_fc2(can_bus_features)
            can_bus_features = F.Relu(f"{self.name}_can_bus_relu2")(can_bus_features)

            if self.can_bus_norm:
                can_bus_features = self.can_bus_norm_layer(can_bus_features)

            # Add to bev_queries: [bs, embed_dims] -> [1, bs, embed_dims]
            can_bus_features = F.Unsqueeze(
                can_bus_features, [0], name=f"{self.name}_can_bus_unsqueeze"
            )
            bev_queries = F.Add(
                bev_queries,
                can_bus_features,
                name=f"{self.name}_bev_queries_add_can_bus",
            )

        # Process multi-level features
        feat_flatten = []
        spatial_shapes = []

        for lvl, feat in enumerate(mlvl_feats):
            # feat shape: [bs, num_cams, C, H, W]
            feat_shape = F.Shape(feat, name=f"{self.name}_feat_l{lvl}_shape")
            num_cam = F.SliceF(feat_shape, [0], [1], [1], name=f"{self.name}_num_cam_l{lvl}")  # type: ignore[call-arg,misc]
            c = F.SliceF(feat_shape, [0], [1], [2], name=f"{self.name}_c_l{lvl}")  # type: ignore[call-arg,misc]
            h = F.SliceF(feat_shape, [0], [1], [3], name=f"{self.name}_h_l{lvl}")  # type: ignore[call-arg,misc]
            w = F.SliceF(feat_shape, [0], [1], [4], name=f"{self.name}_w_l{lvl}")  # type: ignore[call-arg,misc]

            spatial_shapes.append((h, w))

            # Flatten spatial dimensions: [bs, num_cams, C, H, W] -> [bs, num_cams, C, H*W]
            hw = F.Mul(h, w, name=f"{self.name}_hw_l{lvl}")
            feat = F.Reshape(
                feat, [bs, num_cam, c, hw], name=f"{self.name}_feat_l{lvl}_flatten"
            )

            # Permute to [num_cams, bs, H*W, C]
            feat = F.Transpose(
                feat, perm=[1, 0, 3, 2], name=f"{self.name}_feat_l{lvl}_transpose"
            )

            # Add camera embeddings if enabled
            if self.use_cams_embeds and self.cams_embeds is not None:
                # cams_embeds: [num_cams, embed_dims]
                # Expand to [num_cams, 1, 1, embed_dims]
                cams_embeds_expanded = F.Unsqueeze(
                    self.cams_embeds,
                    [1, 2],
                    name=f"{self.name}_cams_embeds_l{lvl}_unsqueeze",
                )
                feat = F.Add(
                    feat, cams_embeds_expanded, name=f"{self.name}_feat_l{lvl}_add_cams"
                )

            # Add level embeddings if available
            if self.level_embeds is not None:
                # level_embeds: [num_feature_levels, embed_dims]
                # Get embedding for this level
                level_embed = F.SliceF(
                    self.level_embeds,
                    [lvl],
                    [lvl + 1],
                    [0],  # type: ignore[call-arg,misc]
                    name=f"{self.name}_level_embed_l{lvl}",
                )
                # Expand to [1, 1, 1, embed_dims]
                level_embed = F.Unsqueeze(
                    level_embed,
                    [0, 1, 2],
                    name=f"{self.name}_level_embed_l{lvl}_unsqueeze",
                )
                feat = F.Add(
                    feat, level_embed, name=f"{self.name}_feat_l{lvl}_add_level"
                )

            feat_flatten.append(feat)

        # Concatenate all levels: [num_cams, bs, sum(H*W), C]
        if len(feat_flatten) > 1:
            feat_flatten = F.ConcatX(
                feat_flatten,
                axis=2,  # type: ignore[assignment]
                name=f"{self.name}_feat_flatten_concat",
            )
        else:
            feat_flatten = feat_flatten[0]

        # Prepare spatial_shapes and level_start_index
        # These would typically be passed as tensors
        # spatial_shapes: [num_levels, 2]
        # level_start_index: [num_levels]

        # Permute feat_flatten to [num_cams, sum(H*W), bs, C]
        feat_flatten = F.Transpose(
            feat_flatten,
            perm=[0, 2, 1, 3],
            name=f"{self.name}_feat_flatten_final_transpose",
        )

        # Call encoder
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,  # key and value are the same
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=kwargs.get("spatial_shapes"),
            level_start_index=kwargs.get("level_start_index"),
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )

        return bev_embed

    def forward(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=None,
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for PerceptionTransformer.

        Args:
            mlvl_feats (list[Tensor]): Multi-level multi-camera features.
                Each element has shape [bs, num_cams, C, H, W].
            bev_queries (Tensor): BEV query embeddings with shape [bev_h*bev_w, embed_dims].
            object_query_embed (Tensor): Object query embeddings with shape [num_query, 2*embed_dims].
                Contains both query and query_pos concatenated.
            bev_h (int): Height of BEV grid.
            bev_w (int): Width of BEV grid.
            grid_length (list): Physical size of BEV grid cell. Default: [0.512, 0.512].
            bev_pos (Tensor): BEV positional encodings with shape [bs, embed_dims, bev_h, bev_w].
            reg_branches (nn.ModuleList): Regression heads for bbox refinement.
            cls_branches (nn.ModuleList): Classification heads.
            prev_bev (Tensor): Previous BEV features for temporal fusion.

        Returns:
            tuple: Contains the following elements:
                - bev_embed (Tensor): BEV features with shape [bs, bev_h*bev_w, embed_dims].
                - inter_states (Tensor): Decoder intermediate states.
                - init_reference_out (Tensor): Initial reference points.
                - inter_references_out (Tensor): Refined reference points per layer.
        """
        if grid_length is None:
            grid_length = [0.512, 0.512]

        # Extract BEV features from multi-camera inputs
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )
        # bev_embed shape: [bev_h*bev_w, bs, embed_dims] or [bs, bev_h*bev_w, embed_dims]

        # Get batch size
        bev_embed_shape = F.Shape(bev_embed, name=f"{self.name}_bev_embed_shape")
        # Assume bev_embed is [bs, bev_h*bev_w, embed_dims]
        bs = F.SliceF(bev_embed_shape, [0], [1], [0], name=f"{self.name}_bs_decoder")  # type: ignore[call-arg,misc]

        # Split object_query_embed into query and query_pos
        # object_query_embed: [num_query, 2*embed_dims]
        # Split along last dimension
        query_pos = F.SliceF(
            object_query_embed,
            [0],
            [self.embed_dims],
            [1],  # type: ignore[call-arg,misc]
            name=f"{self.name}_query_pos_slice",
        )
        query = F.SliceF(
            object_query_embed,
            [self.embed_dims],
            [2 * self.embed_dims],
            [1],  # type: ignore[call-arg,misc]
            name=f"{self.name}_query_slice",
        )

        # Expand to batch: [num_query, embed_dims] -> [num_query, bs, embed_dims]
        query_pos = F.Unsqueeze(
            query_pos, [1], name=f"{self.name}_query_pos_unsqueeze"
        )
        query_pos = F.Tile(query_pos, [1, bs, 1], name=f"{self.name}_query_pos_tile")

        query = F.Unsqueeze(query, [1], name=f"{self.name}_query_unsqueeze")
        query = F.Tile(query, [1, bs, 1], name=f"{self.name}_query_tile")

        # Predict initial reference points from query_pos
        reference_points = self.reference_points(query_pos)
        reference_points = F.Sigmoid(
            f"{self.name}_reference_points_sigmoid"
        )(reference_points)
        init_reference_out = reference_points

        # Permute for decoder
        # query: [num_query, bs, embed_dims] -> [num_query, bs, embed_dims] (already correct)
        # bev_embed: [bs, bev_h*bev_w, embed_dims] -> [bev_h*bev_w, bs, embed_dims]
        bev_embed = F.Transpose(
            bev_embed, perm=[1, 0, 2], name=f"{self.name}_bev_embed_transpose"
        )

        # Prepare spatial_shapes for decoder: [[bev_h, bev_w]]
        # This would be a tensor of shape [1, 2]
        bev_spatial_shape = F._from_data(
            f"{self.name}_bev_spatial_shape",
            np.array([[bev_h, bev_w]], dtype=np.int64),
            is_const=True,
        )

        # Level start index: [0]
        level_start_idx = F._from_data(
            f"{self.name}_level_start_idx", np.array([0], dtype=np.int64), is_const=True
        )

        # Call decoder
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=bev_spatial_shape,
            level_start_index=level_start_idx,
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


def rotate_bev_with_affine(
    prev_bev, rotation_angle, bev_h, bev_w, rotate_center, name="rotate_bev"
):
    """Rotate BEV features using affine transformation (TTSim implementation).

    This is a helper function to rotate BEV features for temporal alignment.
    Uses GridSample with affine transformation matrix.

    Args:
        prev_bev (Tensor): Previous BEV features with shape [bev_h*bev_w, 1, embed_dims].
        rotation_angle (float): Rotation angle in radians.
        bev_h (int): BEV grid height.
        bev_w (int): BEV grid width.
        rotate_center (list): Rotation center [cx, cy] in grid coordinates.
        name (str): Operation name prefix.

    Returns:
        Tensor: Rotated BEV features with same shape as input.
    """
    # Reshape prev_bev: [bev_h*bev_w, 1, embed_dims] -> [1, embed_dims, bev_h, bev_w]
    prev_bev = F.Reshape(
        prev_bev,
        [
            F._from_data(f"{name}_1", np.array([1]), is_const=True),
            F._from_data(f"{name}_bev_h", np.array([bev_h]), is_const=True),
            F._from_data(f"{name}_bev_w", np.array([bev_w]), is_const=True),
            F._from_data(f"{name}_embed_dims_shape", np.array([-1]), is_const=True),
        ],
        name=f"{name}_reshape_4d",
    )
    prev_bev = F.Transpose(
        prev_bev, perm=[0, 3, 1, 2], name=f"{name}_transpose_bchw"
    )

    # Create rotation grid
    # This would require implementing an affine grid generator using rotation matrix
    # For now, return unrotated (full implementation would use GridSample with rotation grid)
    warnings.warn(f"BEV rotation not fully implemented. Returning unrotated BEV.")

    # Reshape back: [1, embed_dims, bev_h, bev_w] -> [bev_h*bev_w, 1, embed_dims]
    prev_bev = F.Transpose(
        prev_bev, perm=[0, 2, 3, 1], name=f"{name}_transpose_back"
    )
    prev_bev = F.Reshape(
        prev_bev,
        [
            F._from_data(f"{name}_bev_hw", np.array([bev_h * bev_w]), is_const=True),
            F._from_data(f"{name}_1_final", np.array([1]), is_const=True),
            F._from_data(f"{name}_embed_dims_final", np.array([-1]), is_const=True),
        ],
        name=f"{name}_reshape_final",
    )

    return prev_bev


def analytical_param_count(embed_dims=256, num_feature_levels=4, num_cams=6):
    """Calculate the analytical parameter count for PerceptionTransformer.

    This does not include encoder and decoder parameters as they are
    passed as pre-built modules.

    Args:
        embed_dims (int): Embedding dimensions. Default: 256.
        num_feature_levels (int): Number of feature pyramid levels. Default: 4.
        num_cams (int): Number of cameras. Default: 6.

    Returns:
        int: Total number of parameters (excluding encoder/decoder).
    """
    # level_embeds: [num_feature_levels, embed_dims]
    level_embeds_params = num_feature_levels * embed_dims

    # cams_embeds: [num_cams, embed_dims]
    cams_embeds_params = num_cams * embed_dims

    # reference_points Linear: embed_dims -> 3
    reference_points_params = embed_dims * 3 + 3

    # can_bus_fc1: 18 -> embed_dims//2
    can_bus_fc1_params = 18 * (embed_dims // 2) + (embed_dims // 2)

    # can_bus_fc2: embed_dims//2 -> embed_dims
    can_bus_fc2_params = (embed_dims // 2) * embed_dims + embed_dims

    # can_bus_norm (LayerNorm): 2 * embed_dims (weight + bias)
    can_bus_norm_params = 2 * embed_dims

    total = (
        level_embeds_params
        + cams_embeds_params
        + reference_points_params
        + can_bus_fc1_params
        + can_bus_fc2_params
        + can_bus_norm_params
    )

    return total
