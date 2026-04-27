#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Multi-Scale Deformable Attention.

This is a CPU-only implementation converted from the PyTorch version in mmcv.
The core algorithm is based on the `multi_scale_deformable_attn_pytorch` function.

Original: mmcv/ops/multi_scale_deform_attn.py
Reference: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
           https://arxiv.org/pdf/2010.04159.pdf
"""

import sys
import os

# Add ttsim to path - navigate to polaris root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module, Linear
from ttsim.ops.desc.data_compute import (
    _numpy_multi_scale_deformable_attn,
)


def multi_scale_deformable_attn_ttsim(
    name,
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    debug=False,
):
    """
    TTSim CPU implementation of multi-scale deformable attention.

    This function samples features from multiple scales at deformable positions
    and aggregates them using attention weights. It's the core operation used in
    Deformable DETR and BEVFormer.

    Args:
        name (str): Operation name prefix
        value (Tensor): Feature values with shape [bs, num_keys, num_heads, embed_dims_per_head]
        value_spatial_shapes (Tensor or list): Spatial shapes [(H1, W1), (H2, W2), ...]
                                               Shape: [num_levels, 2]
        sampling_locations (Tensor): Normalized sampling coordinates [0, 1]
                                    Shape: [bs, num_queries, num_heads, num_levels, num_points, 2]
        attention_weights (Tensor): Attention weights
                                   Shape: [bs, num_queries, num_heads, num_levels, num_points]

    Returns:
        Tensor: Output features with shape [bs, num_queries, embed_dims]
               where embed_dims = num_heads * embed_dims_per_head

    Algorithm:
        1. Split value tensor by spatial shapes into per-level features
        2. For each level:
           a. Reshape value to [bs*num_heads, embed_dims_per_head, H, W]
           b. Get sampling grid for this level
           c. Apply bilinear sampling (grid_sample)
        3. Stack sampled values from all levels
        4. Apply attention weights and aggregate
        5. Reshape to output format
    """

    # Get dimensions
    # value: [bs, num_keys, num_heads, embed_dims_per_head]
    bs = value.shape[0]
    num_keys = value.shape[1]
    num_heads = value.shape[2]
    embed_dims_per_head = value.shape[3]

    # sampling_locations: [bs, num_queries, num_heads, num_levels, num_points, 2]
    num_queries = sampling_locations.shape[1]
    num_levels = sampling_locations.shape[3]
    num_points = sampling_locations.shape[4]

    # Parse spatial shapes
    if isinstance(value_spatial_shapes, list):
        spatial_shapes_list = value_spatial_shapes
    else:
        # Convert tensor to list of tuples
        spatial_shapes_list = [
            (int(value_spatial_shapes.data[i, 0]), int(value_spatial_shapes.data[i, 1]))
            for i in range(num_levels)
        ]

    # Early return: if all inputs carry data, compute numerically and skip graph building
    if (
        value.data is not None
        and sampling_locations.data is not None
        and attention_weights.data is not None
    ):
        result = _numpy_multi_scale_deformable_attn(
            value.data,
            spatial_shapes_list,
            sampling_locations.data,
            attention_weights.data,
        )
        return F._from_data(name + ".output", result)

    def _check_data(tensor, label):
        if debug and tensor.data is None:
            raise RuntimeError(f"{label} has no data")

    # Step 1: Split value by levels
    # Calculate split sizes: [H1*W1, H2*W2, ...]
    split_sizes = [H * W for H, W in spatial_shapes_list]

    # Split value along dimension 1 (num_keys)
    value_list = []
    start_idx = 0
    for size in split_sizes:
        # Slice: value[:, start_idx:start_idx+size, :, :]
        end_idx = start_idx + size
        value_slice_starts = F._from_data(
            name + f".value_split_{start_idx}.starts",
            np.array([start_idx], dtype=np.int64),
            is_const=True,
        )
        value_slice_ends = F._from_data(
            name + f".value_split_{start_idx}.ends",
            np.array([end_idx], dtype=np.int64),
            is_const=True,
        )
        value_slice_axes = F._from_data(
            name + f".value_split_{start_idx}.axes",
            np.array([1], dtype=np.int64),
            is_const=True,
        )
        value_slice_steps = F._from_data(
            name + f".value_split_{start_idx}.steps",
            np.array([1], dtype=np.int64),
            is_const=True,
        )
        value_level = F.SliceF(
            name + f".value_split_{start_idx}",
            out_shape=[bs, end_idx - start_idx, num_heads, embed_dims_per_head],
        )(
            value,
            value_slice_starts,
            value_slice_ends,
            value_slice_axes,
            value_slice_steps,
        )
        _check_data(value_level, f"value_level_{start_idx}")
        value_list.append(value_level)
        start_idx = end_idx

    # Step 2: Normalize sampling locations to [-1, 1] for grid_sample
    # sampling_grids = 2 * sampling_locations - 1
    two_const = F._from_data(
        name + ".two", np.array(2.0, dtype=np.float32), is_const=True
    )
    one_const = F._from_data(
        name + ".one", np.array(1.0, dtype=np.float32), is_const=True
    )

    sampling_grids = F.Mul(name + ".sampling_mul2")(sampling_locations, two_const)
    sampling_grids = F.Sub(name + ".sampling_sub1")(sampling_grids, one_const)
    _check_data(sampling_grids, "sampling_grids")

    # Step 3: Process each level
    sampling_value_list = []

    for level, (H, W) in enumerate(spatial_shapes_list):
        # Get value for this level: [bs, H*W, num_heads, embed_dims_per_head]
        value_l = value_list[level]

        # Reshape value_l for grid_sample:
        # [bs, H*W, num_heads, embed_dims_per_head] ->
        # [bs, H*W, num_heads*embed_dims_per_head] ->
        # [bs, num_heads*embed_dims_per_head, H*W] ->
        # [bs*num_heads, embed_dims_per_head, H, W]

        # Flatten heads and embed_dims: [bs, H*W, num_heads*embed_dims_per_head]
        value_l_flat = F.Reshape(name + f".value_l{level}_flat1")(
            value_l,
            F._from_data(
                name + f".value_l{level}_shape1",
                np.array([bs, H * W, num_heads * embed_dims_per_head], dtype=np.int64),
                is_const=True,
            ),
        )

        # Transpose: [bs, num_heads*embed_dims_per_head, H*W]
        value_l_trans = F.Transpose(name + f".value_l{level}_trans", perm=[0, 2, 1])(
            value_l_flat
        )

        # Reshape to image format: [bs*num_heads, embed_dims_per_head, H, W]
        value_l_img = F.Reshape(name + f".value_l{level}_img")(
            value_l_trans,
            F._from_data(
                name + f".value_l{level}_img_shape",
                np.array([bs * num_heads, embed_dims_per_head, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        # Get sampling grid for this level
        # sampling_grids: [bs, num_queries, num_heads, num_levels, num_points, 2]
        # Extract level: [bs, num_queries, num_heads, num_points, 2]
        grid_slice_starts = F._from_data(
            name + f".sampling_grid_l{level}.starts",
            np.array([level], dtype=np.int64),
            is_const=True,
        )
        grid_slice_ends = F._from_data(
            name + f".sampling_grid_l{level}.ends",
            np.array([level + 1], dtype=np.int64),
            is_const=True,
        )
        grid_slice_axes = F._from_data(
            name + f".sampling_grid_l{level}.axes",
            np.array([3], dtype=np.int64),
            is_const=True,
        )
        grid_slice_steps = F._from_data(
            name + f".sampling_grid_l{level}.steps",
            np.array([1], dtype=np.int64),
            is_const=True,
        )
        sampling_grid_l = F.SliceF(
            name + f".sampling_grid_l{level}",
            out_shape=[bs, num_queries, num_heads, 1, num_points, 2],
        )(
            sampling_grids,
            grid_slice_starts,
            grid_slice_ends,
            grid_slice_axes,
            grid_slice_steps,
        )
        _check_data(sampling_grid_l, f"sampling_grid_l{level}_slice")

        # Squeeze level dimension: [bs, num_queries, num_heads, num_points, 2]
        squeeze_axes = F._from_data(
            name + f".sampling_grid_l{level}_sq.axes",
            np.array([3], dtype=np.int64),
            is_const=True,
        )
        sampling_grid_l = F.Squeeze(name + f".sampling_grid_l{level}_sq")(
            sampling_grid_l, squeeze_axes
        )
        _check_data(sampling_grid_l, f"sampling_grid_l{level}_sq")

        # Transpose to [bs, num_heads, num_queries, num_points, 2]
        sampling_grid_l = F.Transpose(
            name + f".sampling_grid_l{level}_trans", perm=[0, 2, 1, 3, 4]
        )(sampling_grid_l)

        # Flatten batch and heads: [bs*num_heads, num_queries, num_points, 2]
        sampling_grid_l_flat = F.Reshape(name + f".sampling_grid_l{level}_flat")(
            sampling_grid_l,
            F._from_data(
                name + f".sampling_grid_l{level}_flat_shape",
                np.array([bs * num_heads, num_queries, num_points, 2], dtype=np.int64),
                is_const=True,
            ),
        )
        _check_data(sampling_grid_l_flat, f"sampling_grid_l{level}_flat")

        # Apply bilinear sampling using GridSample
        # value_l_img: [bs*num_heads, embed_dims_per_head, H, W]
        # sampling_grid_l_flat: [bs*num_heads, num_queries, num_points, 2]
        # output: [bs*num_heads, embed_dims_per_head, num_queries, num_points]
        sampling_value_l = F.GridSample(
            name + f".grid_sample_l{level}",
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )(value_l_img, sampling_grid_l_flat)
        _check_data(sampling_value_l, f"sampling_value_l{level}")

        sampling_value_list.append(sampling_value_l)

    # Step 4: Stack and aggregate
    # Stack along level dimension: [bs*num_heads, embed_dims_per_head, num_queries, num_levels, num_points]
    unsqueezed_values = []
    for level, sampling_value_l in enumerate(sampling_value_list):
        unsq_axes = F._from_data(
            name + f".sampling_value_l{level}.unsq_axes",
            np.array([3], dtype=np.int64),
            is_const=True,
        )
        sampling_value_l_unsq = F.Unsqueeze(name + f".sampling_value_l{level}.unsq")(
            sampling_value_l, unsq_axes
        )
        unsqueezed_values.append(sampling_value_l_unsq)

    if len(unsqueezed_values) == 1:
        stacked_values = unsqueezed_values[0]
    else:
        stacked_values = F.ConcatX(name + ".concat_levels", axis=3)(*unsqueezed_values)

    # Flatten levels and points: [bs*num_heads, embed_dims_per_head, num_queries, num_levels*num_points]
    stacked_values_flat = F.Reshape(name + ".stacked_flat")(
        stacked_values,
        F._from_data(
            name + ".stacked_flat_shape",
            np.array(
                [
                    bs * num_heads,
                    embed_dims_per_head,
                    num_queries,
                    num_levels * num_points,
                ],
                dtype=np.int64,
            ),
            is_const=True,
        ),
    )
    _check_data(stacked_values_flat, "stacked_values_flat")

    # Reshape attention weights
    # [bs, num_queries, num_heads, num_levels, num_points] ->
    # [bs, num_heads, num_queries, num_levels, num_points] ->
    # [bs*num_heads, 1, num_queries, num_levels*num_points]
    attn_trans = F.Transpose(name + ".attn_trans", perm=[0, 2, 1, 3, 4])(
        attention_weights
    )
    attn_flat = F.Reshape(name + ".attn_flat")(
        attn_trans,
        F._from_data(
            name + ".attn_flat_shape",
            np.array(
                [bs * num_heads, 1, num_queries, num_levels * num_points],
                dtype=np.int64,
            ),
            is_const=True,
        ),
    )
    _check_data(attn_flat, "attn_flat")

    # Multiply: [bs*num_heads, embed_dims_per_head, num_queries, num_levels*num_points]
    weighted = F.Mul(name + ".weighted")(stacked_values_flat, attn_flat)
    _check_data(weighted, "weighted")

    # Sum over sampling points: [bs*num_heads, embed_dims_per_head, num_queries]
    aggregated = F.ReduceSum(name + ".aggregate", axis=3, keepdims=False)(weighted)
    _check_data(aggregated, "aggregated")

    # Reshape to [bs, num_heads*embed_dims_per_head, num_queries]
    output = F.Reshape(name + ".output_reshape")(
        aggregated,
        F._from_data(
            name + ".output_shape",
            np.array(
                [bs, num_heads * embed_dims_per_head, num_queries], dtype=np.int64
            ),
            is_const=True,
        ),
    )

    # Transpose to [bs, num_queries, num_heads*embed_dims_per_head]
    output = F.Transpose(name + ".output_final", perm=[0, 2, 1])(output)
    _check_data(output, "output_final")

    return output


class MultiScaleDeformableAttention(Module):
    """
    TTSim implementation of Multi-Scale Deformable Attention module.

    Used in Deformable DETR and BEVFormer for efficient multi-scale feature aggregation
    with learnable sampling positions.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per head per level. Default: 4
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: False
        value_proj_ratio (float): Value projection expansion ratio. Default: 1.0

    Input Shapes:
        query: [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
        value: [num_key, bs, embed_dims] or [bs, num_key, embed_dims] if batch_first
        reference_points: [bs, num_query, num_levels, 2] - normalized coordinates in [0, 1]
        spatial_shapes: [num_levels, 2] - (H, W) for each level
        level_start_index: [num_levels] - starting index for each level in value

    Output Shape:
        [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        dropout=0.1,
        batch_first=False,
        value_proj_ratio=1.0,
    ):
        super().__init__(name)  # type: ignore[call-arg]
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout_rate = dropout
        self.batch_first = batch_first
        self.value_proj_ratio = value_proj_ratio

        # Projections
        self.sampling_offsets = Linear(
            name + ".sampling_offsets",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points * 2,
        )

        self.attention_weights = Linear(
            name + ".attention_weights",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points,
        )

        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = Linear(
            name + ".value_proj", in_features=embed_dims, out_features=value_proj_size
        )

        self.output_proj = Linear(
            name + ".output_proj", in_features=value_proj_size, out_features=embed_dims
        )

    def __call__(
        self,
        query,
        value=None,
        identity=None,
        query_pos=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        """
        Forward pass of Multi-Scale Deformable Attention.

        Args:
            query: Query features
            value: Value features (if None, uses query)
            identity: Identity for residual connection (if None, uses query)
            query_pos: Positional encoding for query
            reference_points: Normalized reference points [bs, num_query, num_levels, 2]
            spatial_shapes: Spatial shapes for each level
            level_start_index: Starting indices for each level

        Returns:
            Output features with residual connection and dropout
        """

        if value is None:
            value = query

        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            query = F.Add(self.name + ".query_pos_add")(query, query_pos)

        # Handle batch_first flag
        if not self.batch_first:
            # Convert from [num_query, bs, embed_dims] to [bs, num_query, embed_dims]
            query = F.Transpose(self.name + ".query_transpose", perm=[1, 0, 2])(query)
            value = F.Transpose(self.name + ".value_transpose", perm=[1, 0, 2])(value)

        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]

        # Project value
        value = self.value_proj(value)

        # Reshape value: [bs, num_value, num_heads, embed_dims_per_head]
        embed_dims_per_head = value.shape[2] // self.num_heads
        value = F.Reshape(self.name + ".value_reshape")(
            value,
            F._from_data(
                self.name + ".value_reshape_shape",
                np.array(
                    [bs, num_value, self.num_heads, embed_dims_per_head], dtype=np.int64
                ),
                is_const=True,
            ),
        )

        # Compute sampling offsets
        # [bs, num_query, num_heads * num_levels * num_points * 2]
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = F.Reshape(self.name + ".offsets_reshape")(
            sampling_offsets,
            F._from_data(
                self.name + ".offsets_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        self.num_points,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Compute attention weights
        # [bs, num_query, num_heads * num_levels * num_points]
        attention_weights = self.attention_weights(query)
        attention_weights = F.Reshape(self.name + ".attn_reshape")(
            attention_weights,
            F._from_data(
                self.name + ".attn_shape",
                np.array(
                    [bs, num_query, self.num_heads, self.num_levels * self.num_points],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Apply softmax
        attention_weights = F.Softmax(self.name + ".attn_softmax", axis=-1)(
            attention_weights
        )

        # Reshape attention weights back
        attention_weights = F.Reshape(self.name + ".attn_reshape2")(
            attention_weights,
            F._from_data(
                self.name + ".attn_shape2",
                np.array(
                    [bs, num_query, self.num_heads, self.num_levels, self.num_points],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Compute sampling locations
        # reference_points: [bs, num_query, num_levels, 2]
        if reference_points.shape[-1] == 2:
            # Normalize offsets by spatial shapes
            # offset_normalizer: [num_levels, 2] with [W, H] format
            offset_normalizer_data = np.array(
                [[float(W), float(H)] for H, W in spatial_shapes], dtype=np.float32
            )
            offset_normalizer = F._from_data(
                self.name + ".offset_normalizer", offset_normalizer_data, is_const=True
            )

            # Expand dimensions for broadcasting
            # reference_points: [bs, num_query, 1, num_levels, 1, 2]
            ref_pts = F.Unsqueeze(self.name + ".ref_unsq1")(
                reference_points,
                F._from_data(
                    self.name + ".ax2", np.array([2], dtype=np.int64), is_const=True
                ),
            )
            ref_pts = F.Unsqueeze(self.name + ".ref_unsq2")(
                ref_pts,
                F._from_data(
                    self.name + ".ax4", np.array([4], dtype=np.int64), is_const=True
                ),
            )

            # Normalize offsets: sampling_offsets / offset_normalizer
            # offset_normalizer: [1, 1, 1, num_levels, 1, 2]
            norm = F.Unsqueeze(self.name + ".norm_unsq1")(
                offset_normalizer,
                F._from_data(
                    self.name + ".ax0", np.array([0], dtype=np.int64), is_const=True
                ),
            )
            norm = F.Unsqueeze(self.name + ".norm_unsq2")(
                norm,
                F._from_data(
                    self.name + ".ax0_2", np.array([0], dtype=np.int64), is_const=True
                ),
            )
            norm = F.Unsqueeze(self.name + ".norm_unsq3")(
                norm,
                F._from_data(
                    self.name + ".ax0_3", np.array([0], dtype=np.int64), is_const=True
                ),
            )

            normalized_offsets = F.Div(self.name + ".norm_offsets")(
                sampling_offsets, norm
            )

            # sampling_locations = reference_points + normalized_offsets
            sampling_locations = F.Add(self.name + ".sampling_locs")(
                ref_pts, normalized_offsets
            )
        else:
            # Handle 4D reference points (with width and height)
            raise NotImplementedError(
                "4D reference points not yet implemented in TTSim"
            )

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            self.name + ".msda",
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
        )

        # Output projection
        output = self.output_proj(output)

        # Apply dropout (in inference, this is identity)
        if self.dropout_rate > 0:
            output = F.Dropout(self.name + ".dropout", ratio=self.dropout_rate)(output)

        # Handle batch_first flag for output
        if not self.batch_first:
            # Convert back to [num_query, bs, embed_dims]
            output = F.Transpose(self.name + ".output_transpose", perm=[1, 0, 2])(
                output
            )
            identity = F.Transpose(self.name + ".identity_transpose", perm=[1, 0, 2])(
                identity
            )

        # Residual connection
        output = F.Add(self.name + ".residual")(output, identity)

        return output

    def analytical_param_count(self, lvl=0):
        """Calculate total number of parameters."""
        count = 0
        count += self.sampling_offsets.analytical_param_count(lvl)
        count += self.attention_weights.analytical_param_count(lvl)
        count += self.value_proj.analytical_param_count(lvl)
        count += self.output_proj.analytical_param_count(lvl)
        return count
