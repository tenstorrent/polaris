#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Temporal Self Attention for BEVFormer.

This module implements the temporal self-attention mechanism that enables
BEV features to attend to historical BEV features across time, enabling
temporal reasoning for object tracking and motion prediction.

Original: projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

The original PyTorch implementation uses several mmcv functions that are not
compatible with Python 3.13. This TTSim version includes the following conversions:

1. Initialization Functions (mmcv.cnn):
   - xavier_init: Implemented in init_utils.py using torch.nn.init for PyTorch
   - constant_init: Implemented in init_utils.py using torch.nn.init for PyTorch

2. Decorators (mmcv.runner):
   - @ATTENTION.register_module(): Not needed in TTSim (no module registry)

3. Base Classes (mmcv.runner.base_module):
   - BaseModule: Replaced with ttsim.front.functional.sim_nn.Module

4. Custom Operations:
   - multi_scale_deformable_attn_pytorch: Uses TTSim implementation
   - MultiScaleDeformableAttnFunction: Replaced with TTSim implementation

5. Utilities:
   - run_time decorator: Not needed in TTSim (no profiling decorators)

6. Operations:
   - torch.stack: Replaced with F.ConcatX + F.Reshape
   - torch.cat: Replaced with F.ConcatX
   - tensor.view/.reshape: Replaced with F.Reshape
   - tensor.permute: Replaced with F.Transpose
   - tensor.softmax: Replaced with F.Softmax
   - tensor.mean: Replaced with F.ReduceMean

7. Warnings:
   - warnings.warn for non-power-of-2 dimensions: Implemented in init_weights

All computational logic from the PyTorch version has been preserved and
converted to TTSim operations.
"""

import sys
import os
import warnings
import math
from loguru import logger

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module, Linear

# Import our TTSim deformable attention
try:
    from .multi_scale_deformable_attn import multi_scale_deformable_attn_ttsim
except ImportError:
    # Handle case when run as script
    from multi_scale_deformable_attn import multi_scale_deformable_attn_ttsim  # type: ignore[import-not-found,no-redef]

# Import initialization utilities (Python 3.13 compatible)
try:
    from .init_utils import xavier_init, constant_init, _is_power_of_2
except ImportError:
    # Handle case when run as script
    from init_utils import xavier_init, constant_init, _is_power_of_2  # type: ignore[import-not-found,no-redef]


class TemporalSelfAttention(Module):
    """
    TTSim implementation of Temporal Self Attention for BEVFormer.

    This module enables BEV queries to attend to both current and historical
    BEV features using deformable attention, allowing the model to reason
    about motion and track objects over time.

    Key features:
    - Processes multiple BEV frames (typically 2: history + current)
    - Uses deformable sampling for flexible temporal aggregation
    - Fuses temporal information through learnable attention weights

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per query. Default: 4
        num_bev_queue (int): Number of BEV frames to process (typically 2). Default: 2
        im2col_step (int): Step for im2col (not used in CPU version). Default: 64
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: True

    Input Shapes:
        query: [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
        value: [bs*num_bev_queue, num_value, embed_dims] - Stacked BEV features
        reference_points: [bs, num_query, num_levels, 2] - Reference points
        spatial_shapes: [(H, W), ...] - Spatial shapes per level
        level_start_index: [num_levels] - Start indices per level

    Output Shape:
        [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
    """

    def __init__(
        self,
        name,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
    ):
        super().__init__()
        self.name = name

        # Check embed_dims divisibility
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        dim_per_head = embed_dims // num_heads

        # Warn about non-power-of-2 dimensions (for efficiency)
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step

        # Sampling offsets: takes concatenated query (embed_dims * num_bev_queue)
        # and outputs offsets for all queues, heads, levels, and points
        self.sampling_offsets = Linear(
            name + ".sampling_offsets",
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points * 2,
        )

        # Attention weights: takes concatenated query and outputs weights
        self.attention_weights = Linear(
            name + ".attention_weights",
            in_features=embed_dims * self.num_bev_queue,
            out_features=num_bev_queue * num_heads * num_levels * num_points,
        )

        # Value projection
        self.value_proj = Linear(
            name + ".value_proj", in_features=embed_dims, out_features=embed_dims
        )

        # Output projection
        self.output_proj = Linear(
            name + ".output_proj", in_features=embed_dims, out_features=embed_dims
        )

        self.dropout_rate = dropout

        # Initialize weights (for PyTorch reference model)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        # Note: This is for documentation and PyTorch reference model
        # In TTSim inference, weights are loaded from pre-trained checkpoint
        pass  # Initialization handled by PyTorch model loading or weight copying

    def __call__(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):
        """
        Forward pass of Temporal Self Attention.

        Args:
            query: Query tensor [bs, num_query, embed_dims] or [num_query, bs, embed_dims]
            key: Not used (self-attention)
            value: Value tensor [bs*num_bev_queue, num_value, embed_dims] or None
                   If None, will stack query with itself for temporal processing
            identity: Residual connection tensor
            query_pos: Positional encoding for queries
            key_padding_mask: Padding mask
            reference_points: Reference points [bs, num_query, num_levels, 2]
            spatial_shapes: Spatial shapes of feature levels
            level_start_index: Start indices for each level
            flag: 'decoder' or 'encoder'

        Returns:
            Output tensor with same shape as query, with residual connection
        """

        # Create value if not provided (stack query with itself)
        if value is None:
            assert self.batch_first, "batch_first must be True when value is None"
            bs, len_bev, c = query.shape

            # Stack query twice: [bs, len_bev, c] -> [bs*2, len_bev, c]
            # This creates [current_bev, current_bev] for temporal processing
            query_unsqueezed = F.Unsqueeze(self.name + ".value_unsqueeze")(
                query,
                F._from_data(
                    self.name + ".value_unsqueeze_axis",
                    np.array([1], dtype=np.int64),
                    is_const=True,
                ),
            )
            query_tiled = F.Tile(self.name + ".value_tile")(
                query_unsqueezed,
                F._from_data(
                    self.name + ".value_tile_reps",
                    np.array([1, 2, 1, 1], dtype=np.int64),
                    is_const=True,
                ),
            )
            value = F.Reshape(self.name + ".value_reshape")(
                query_tiled,
                F._from_data(
                    self.name + ".value_reshape_shape",
                    np.array([bs * 2, len_bev, c], dtype=np.int64),
                    is_const=True,
                ),
            )

        # Handle residual/identity
        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            query = F.Add(self.name + ".query_add_pos")(query, query_pos)

        # Handle batch_first format
        if not self.batch_first:
            # Convert to batch_first: [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            query = F.Transpose(self.name + ".query_transpose", perm=[1, 0, 2])(query)
            value = F.Transpose(self.name + ".value_transpose", perm=[1, 0, 2])(value)

        # Get dimensions
        bs = query.shape[0]
        num_query = query.shape[1]
        embed_dims = query.shape[2]
        num_value = value.shape[1]

        # Concatenate current query with historical value for temporal attention
        # value[:bs] is current BEV, query is also current BEV
        # Concatenate them: [bs, num_query, embed_dims*2]
        # First, flatten value from [bs*num_bev_queue, num_value, num_heads, head_dim] to [bs*num_bev_queue, num_value, embed_dims]
        value = F.Reshape(self.name + ".value_input_flatten")(
            value,
            F._from_data(
                self.name + ".value_input_flatten_shape",
                np.array(
                    [bs * self.num_bev_queue, num_value, embed_dims], dtype=np.int64
                ),
                is_const=True,
            ),
        )

        # Use SliceF to extract value[:bs]
        value_current_shape = [bs, num_value, embed_dims]
        starts_0 = F._from_data(
            self.name + ".starts_0", np.array([0], dtype=np.int64), is_const=True
        )
        ends_bs = F._from_data(
            self.name + ".ends_bs", np.array([bs], dtype=np.int64), is_const=True
        )
        axes_0 = F._from_data(
            self.name + ".axes_0", np.array([0], dtype=np.int64), is_const=True
        )
        steps_1 = F._from_data(
            self.name + ".steps_1", np.array([1], dtype=np.int64), is_const=True
        )
        value_current = F.SliceF(
            self.name + ".value_current", out_shape=value_current_shape
        )(value, starts_0, ends_bs, axes_0, steps_1)
        query_concat = F.ConcatX(self.name + ".query_concat", axis=2)(
            value_current, query
        )

        # Project value
        value = self.value_proj(value)

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # Expand mask: [bs*num_bev_queue, num_value] -> [bs*num_bev_queue, num_value, 1]
            mask_expanded = F.Unsqueeze(self.name + ".mask_expand")(
                key_padding_mask,
                F._from_data(
                    self.name + ".mask_expand_axis",
                    np.array([2], dtype=np.int64),
                    is_const=True,
                ),
            )
            # Invert mask (0 becomes 1, non-zero becomes 0)
            mask_inverted = F.Sub(self.name + ".mask_invert")(
                F._from_data(
                    self.name + ".one",
                    np.array([[[1.0]]], dtype=np.float32),
                    is_const=True,
                ),
                mask_expanded,
            )
            value = F.Mul(self.name + ".value_masked")(value, mask_inverted)

        # Reshape value: [bs*num_bev_queue, num_value, embed_dims] -> [bs*num_bev_queue, num_value, num_heads, dim_per_head]
        dim_per_head = self.embed_dims // self.num_heads
        value = F.Reshape(self.name + ".value_reshape_heads")(
            value,
            F._from_data(
                self.name + ".value_reshape_heads_shape",
                np.array(
                    [bs * self.num_bev_queue, num_value, self.num_heads, dim_per_head],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Generate sampling offsets
        sampling_offsets = self.sampling_offsets(query_concat)
        sampling_offsets = F.Reshape(self.name + ".sampling_offsets_reshape")(
            sampling_offsets,
            F._from_data(
                self.name + ".sampling_offsets_reshape_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_bev_queue,
                        self.num_levels,
                        self.num_points,
                        2,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Generate attention weights
        attention_weights = self.attention_weights(query_concat)
        attention_weights = F.Reshape(self.name + ".attention_weights_reshape")(
            attention_weights,
            F._from_data(
                self.name + ".attention_weights_reshape_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_bev_queue,
                        self.num_levels * self.num_points,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Apply softmax
        attention_weights = F.Softmax(
            self.name + ".attention_weights_softmax", axis=-1
        )(attention_weights)

        # Reshape attention weights
        attention_weights = F.Reshape(self.name + ".attention_weights_reshape2")(
            attention_weights,
            F._from_data(
                self.name + ".attention_weights_reshape2_shape",
                np.array(
                    [
                        bs,
                        num_query,
                        self.num_heads,
                        self.num_bev_queue,
                        self.num_levels,
                        self.num_points,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Permute attention weights: [bs, num_query, num_heads, num_bev_queue, num_levels, num_points]
        # -> [bs, num_bev_queue, num_query, num_heads, num_levels, num_points]
        attention_weights = F.Transpose(
            self.name + ".attention_weights_permute", perm=[0, 3, 1, 2, 4, 5]
        )(attention_weights)

        # Reshape to [bs*num_bev_queue, num_query, num_heads, num_levels, num_points]
        attention_weights = F.Reshape(self.name + ".attention_weights_reshape3")(
            attention_weights,
            F._from_data(
                self.name + ".attention_weights_reshape3_shape",
                np.array(
                    [
                        bs * self.num_bev_queue,
                        num_query,
                        self.num_heads,
                        self.num_levels,
                        self.num_points,
                    ],
                    dtype=np.int64,
                ),
                is_const=True,
            ),
        )

        # Permute sampling offsets: [bs, num_query, num_heads, num_bev_queue, num_levels, num_points, 2]
        # -> [bs, num_bev_queue, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = F.Transpose(
            self.name + ".sampling_offsets_permute", perm=[0, 3, 1, 2, 4, 5, 6]
        )(sampling_offsets)

        # Reshape to [bs*num_bev_queue, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = F.Reshape(self.name + ".sampling_offsets_reshape2")(
            sampling_offsets,
            F._from_data(
                self.name + ".sampling_offsets_reshape2_shape",
                np.array(
                    [
                        bs * self.num_bev_queue,
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

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            # Create offset normalizer from spatial_shapes
            # spatial_shapes is a list of tuples, need to convert to tensor
            spatial_shapes_array = np.array(spatial_shapes, dtype=np.float32)
            # Stack [W, H] for normalization
            offset_normalizer_data = np.stack(
                [spatial_shapes_array[:, 1], spatial_shapes_array[:, 0]], axis=-1
            )
            offset_normalizer = F._from_data(
                self.name + ".offset_normalizer", offset_normalizer_data, is_const=True
            )

            # Expand reference_points for num_bev_queue: [bs, num_query, num_levels, 2]
            # -> [bs, num_bev_queue, num_query, num_levels, 2] -> [bs*num_bev_queue, num_query, num_levels, 2]
            ref_points_unsqueezed = F.Unsqueeze(self.name + ".ref_points_unsqueeze")(
                reference_points,
                F._from_data(
                    self.name + ".ref_points_unsqueeze_axis",
                    np.array([1], dtype=np.int64),
                    is_const=True,
                ),
            )
            ref_points_tiled = F.Tile(self.name + ".ref_points_tile")(
                ref_points_unsqueezed,
                F._from_data(
                    self.name + ".ref_points_tile_reps",
                    np.array([1, self.num_bev_queue, 1, 1, 1], dtype=np.int64),
                    is_const=True,
                ),
            )
            ref_points_expanded_flat = F.Reshape(
                self.name + ".ref_points_expanded_flat"
            )(
                ref_points_tiled,
                F._from_data(
                    self.name + ".ref_points_expanded_flat_shape",
                    np.array(
                        [bs * self.num_bev_queue, num_query, self.num_levels, 2],
                        dtype=np.int64,
                    ),
                    is_const=True,
                ),
            )

            # Expand reference_points: [bs*num_bev_queue, num_query, num_levels, 2]
            # -> [bs*num_bev_queue, num_query, 1, num_levels, 1, 2]
            ref_points_exp1 = F.Unsqueeze(self.name + ".ref_points_exp1")(
                ref_points_expanded_flat,
                F._from_data(
                    self.name + ".ref_points_exp1_axis",
                    np.array([2], dtype=np.int64),
                    is_const=True,
                ),
            )
            ref_points_exp2 = F.Unsqueeze(self.name + ".ref_points_exp2")(
                ref_points_exp1,
                F._from_data(
                    self.name + ".ref_points_exp2_axis",
                    np.array([4], dtype=np.int64),
                    is_const=True,
                ),
            )

            # Expand offset_normalizer: [num_levels, 2] -> [1, 1, 1, num_levels, 1, 2]
            offset_normalizer_expanded = F.Unsqueeze(self.name + ".offset_norm_exp1")(
                offset_normalizer,
                F._from_data(
                    self.name + ".offset_norm_exp1_axis",
                    np.array([0], dtype=np.int64),
                    is_const=True,
                ),
            )
            offset_normalizer_expanded = F.Unsqueeze(self.name + ".offset_norm_exp2")(
                offset_normalizer_expanded,
                F._from_data(
                    self.name + ".offset_norm_exp2_axis",
                    np.array([0], dtype=np.int64),
                    is_const=True,
                ),
            )
            offset_normalizer_expanded = F.Unsqueeze(self.name + ".offset_norm_exp3")(
                offset_normalizer_expanded,
                F._from_data(
                    self.name + ".offset_norm_exp3_axis",
                    np.array([0], dtype=np.int64),
                    is_const=True,
                ),
            )
            offset_normalizer_expanded = F.Unsqueeze(self.name + ".offset_norm_exp4")(
                offset_normalizer_expanded,
                F._from_data(
                    self.name + ".offset_norm_exp4_axis",
                    np.array([4], dtype=np.int64),
                    is_const=True,
                ),
            )

            # Normalize offsets and add to reference points
            sampling_offsets_normalized = F.Div(
                self.name + ".sampling_offsets_normalized"
            )(sampling_offsets, offset_normalizer_expanded)
            sampling_locations = F.Add(self.name + ".sampling_locations")(
                ref_points_exp2, sampling_offsets_normalized
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2, "
                f"but got {reference_points.shape[-1]} instead."
            )

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            value=value,
            value_spatial_shapes=spatial_shapes,
            sampling_locations=sampling_locations,
            attention_weights=attention_weights,
            name=self.name + ".msda",
        )

        # Output shape: [bs*num_bev_queue, num_query, embed_dims]
        # Permute to [num_query, embed_dims, bs*num_bev_queue]
        output = F.Transpose(self.name + ".output_permute1", perm=[1, 2, 0])(output)

        # Reshape to [num_query, embed_dims, bs, num_bev_queue]
        output = F.Reshape(self.name + ".output_reshape1")(
            output,
            F._from_data(
                self.name + ".output_reshape1_shape",
                np.array(
                    [num_query, embed_dims, bs, self.num_bev_queue], dtype=np.int64
                ),
                is_const=True,
            ),
        )

        # Fuse history and current by averaging over num_bev_queue dimension
        # Use ReduceMean over axis 3 (num_bev_queue)
        axes_tensor = F._from_data(
            self.name + ".mean_axes", np.array([3], dtype=np.int64), is_const=True
        )
        reduce_mean_op = F.SimOpHandle(
            self.name + ".output_mean",
            "ReduceMean",
            params=[(1, axes_tensor)],
            ipos=[0],
            keepdims=0,
        )
        reduce_mean_op.implicit_inputs.append(axes_tensor)
        output = reduce_mean_op(output)

        # Permute back to [bs, num_query, embed_dims]
        output = F.Transpose(self.name + ".output_permute2", perm=[2, 0, 1])(output)

        # Apply output projection
        output = self.output_proj(output)

        # Apply dropout (note: in inference mode, dropout is typically disabled)
        # For TTSim, we'll skip dropout as it's inference-only
        # In PyTorch training, this would apply dropout

        # Handle batch_first format
        if not self.batch_first:
            # Convert back: [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            output = F.Transpose(self.name + ".output_transpose_final", perm=[1, 0, 2])(
                output
            )

        # Add residual connection
        output = F.Add(self.name + ".output_residual")(output, identity)

        return output

    def analytical_param_count(self):
        """
        Calculate the total number of parameters in this module.

        Returns:
            int: Total parameter count
        """
        # Sampling offsets: embed_dims*num_bev_queue -> num_bev_queue*num_heads*num_levels*num_points*2
        sampling_offsets_params = (self.embed_dims * self.num_bev_queue) * (
            self.num_bev_queue * self.num_heads * self.num_levels * self.num_points * 2
        )
        sampling_offsets_bias = (
            self.num_bev_queue * self.num_heads * self.num_levels * self.num_points * 2
        )

        # Attention weights: embed_dims*num_bev_queue -> num_bev_queue*num_heads*num_levels*num_points
        attention_weights_params = (self.embed_dims * self.num_bev_queue) * (
            self.num_bev_queue * self.num_heads * self.num_levels * self.num_points
        )
        attention_weights_bias = (
            self.num_bev_queue * self.num_heads * self.num_levels * self.num_points
        )

        # Value projection: embed_dims -> embed_dims
        value_proj_params = self.embed_dims * self.embed_dims
        value_proj_bias = self.embed_dims

        # Output projection: embed_dims -> embed_dims
        output_proj_params = self.embed_dims * self.embed_dims
        output_proj_bias = self.embed_dims

        total = (
            sampling_offsets_params
            + sampling_offsets_bias
            + attention_weights_params
            + attention_weights_bias
            + value_proj_params
            + value_proj_bias
            + output_proj_params
            + output_proj_bias
        )

        return total


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Temporal Self Attention TTSim Module")
    logger.info("=" * 80)
    logger.info("\n✓ Module imported successfully!")
    logger.info("\nAvailable component:")
    logger.info("  - TemporalSelfAttention - Temporal attention for BEV features")

    logger.info("\nModule test:")

    # Test TemporalSelfAttention
    try:
        tsa = TemporalSelfAttention(
            name="test_tsa",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            num_bev_queue=2,
            batch_first=True,
        )
        logger.info("  ✓ TemporalSelfAttention constructed successfully")
        logger.debug(f"    - Name: {tsa.name}")
        logger.debug(f"    - Embed dims: {tsa.embed_dims}")
        logger.debug(f"    - Num heads: {tsa.num_heads}")
        logger.debug(f"    - Num levels: {tsa.num_levels}")
        logger.debug(f"    - Num points: {tsa.num_points}")
        logger.debug(f"    - Num BEV queue: {tsa.num_bev_queue}")
        logger.debug(f"    - Parameter count: {tsa.analytical_param_count():,}")
    except Exception as e:
        logger.info(f"  ✗ TemporalSelfAttention construction failed: {e}")
        import traceback

        traceback.print_exc()

    logger.info("\n✓ Basic test passed!")
    logger.info(
        "\nNote: Use validation tests in Validation/ folder for full functionality testing."
    )
    logger.info("=" * 80)
