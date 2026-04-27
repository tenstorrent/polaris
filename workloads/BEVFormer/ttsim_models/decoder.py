#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
BEVFormer Decoder - TTSim Implementation

Converted from PyTorch to TTSim for CPU-based inference.
This module implements the detection transformer decoder used in BEVFormer
for 3D object detection.

Original: projects/mmdet3d_plugin/bevformer/modules/decoder.py
"""

import numpy as np
import warnings
from ttsim.front.functional.sim_nn import Module, Linear
import ttsim.front.functional.op as F


def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse function of sigmoid: logit(x) = log(x / (1-x))

    Args:
        x: Input tensor (numpy array or TTSim tensor)
        eps: Small value to avoid numerical issues

    Returns:
        Tensor with inverse sigmoid applied

    Note: This is used to convert normalized coordinates back to logits
    for iterative refinement in the decoder.
    """
    # Clamp to [0, 1] range
    x = F.Maximum("inverse_sigmoid_clamp_min")(x, F.Constant(0.0))
    x = F.Minimum("inverse_sigmoid_clamp_max")(x, F.Constant(1.0))

    # Clamp to avoid log(0)
    x1 = F.Maximum("inverse_sigmoid_eps1")(x, F.Constant(eps))
    x2_temp = F.Sub("inverse_sigmoid_sub")(F.Constant(1.0), x)
    x2 = F.Maximum("inverse_sigmoid_eps2")(x2_temp, F.Constant(eps))

    # log(x1 / x2)
    ratio = F.Div("inverse_sigmoid_div")(x1, x2)
    result = F.Log("inverse_sigmoid_log")(ratio)

    return result

    # log(x1 / x2)
    ratio = F.Div(x1, x2)  # type: ignore[unreachable]
    result = F.Log(ratio)

    return result


class CustomMSDeformableAttention(Module):
    """
    Custom Multi-Scale Deformable Attention for BEVFormer Decoder.

    This is similar to the one used in the encoder but optimized for decoder usage.
    Implements deformable attention with learnable sampling locations and attention weights.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension (default: 256)
        num_heads (int): Number of attention heads (default: 8)
        num_levels (int): Number of feature pyramid levels (default: 4)
        num_points (int): Number of sampling points per head (default: 4)
        dropout (float): Dropout rate (default: 0.1, ignored in inference)
        batch_first (bool): Whether batch dimension is first (default: False)
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
    ):
        super().__init__()
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        dim_per_head = embed_dims // num_heads

        # Check if dim_per_head is power of 2 (more efficient)
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    f"invalid input for _is_power_of_2: {n} (type: {type(n)})"
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient."
            )

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.batch_first = batch_first

        # Learnable parameters for sampling offsets
        self.sampling_offsets = Linear(
            name=f"{name}_sampling_offsets",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points * 2,
            bias=True,
        )

        # Learnable parameters for attention weights
        self.attention_weights = Linear(
            name=f"{name}_attention_weights",
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points,
            bias=True,
        )

        # Value projection
        self.value_proj = Linear(
            name=f"{name}_value_proj",
            in_features=embed_dims,
            out_features=embed_dims,
            bias=True,
        )

        # Output projection
        self.output_proj = Linear(
            name=f"{name}_output_proj",
            in_features=embed_dims,
            out_features=embed_dims,
            bias=True,
        )

    def init_weights(self):
        """
        Initialize weights (for reference only - not executed in TTSim inference).

        In production, weights are loaded from pre-trained checkpoints.
        This method documents the initialization strategy used during training.
        """
        pass

    def forward(
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
        **kwargs,
    ):
        """
        Forward pass of multi-scale deformable attention.

        Args:
            query: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
            key: Not used (kept for compatibility)
            value: [num_key, bs, embed_dims] or [bs, num_key, embed_dims]
            identity: Residual connection input (same shape as query)
            query_pos: Positional encoding for query
            key_padding_mask: [bs, num_key] - mask for padded positions
            reference_points: [bs, num_query, num_levels, 2 or 4] - normalized reference points
            spatial_shapes: [num_levels, 2] - (H, W) for each level
            level_start_index: [num_levels] - start index for each level

        Returns:
            output: [num_query, bs, embed_dims] or [bs, num_query, embed_dims]
        """
        # Handle default values
        if value is None:
            value = query

        if identity is None:
            identity = query

        # Add positional encoding to query
        if query_pos is not None:
            query = F.Add(query, query_pos)

        # Convert to batch-first format if needed
        if not self.batch_first:
            # [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            query = F.Transpose(query, perm=(1, 0, 2))
            value = F.Transpose(value, perm=(1, 0, 2))

        # Get shapes
        bs = F.Shape(query, 0)
        num_query = F.Shape(query, 1)
        num_value = F.Shape(value, 1)

        # Project value: [bs, num_value, embed_dims]
        value = self.value_proj(value)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask dimensions: [bs, num_value] -> [bs, num_value, 1]
            mask_expanded = F.Unsqueeze(key_padding_mask, axis=-1)
            # Masked fill with 0
            value = F.Where(mask_expanded, F.Constant(0.0), value)

        # Reshape value: [bs, num_value, num_heads, dim_per_head]
        dim_per_head = self.embed_dims // self.num_heads
        value = F.Reshape(value, [bs, num_value, self.num_heads, dim_per_head])

        # Generate sampling offsets: [bs, num_query, num_heads, num_levels, num_points, 2]
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = F.Reshape(
            sampling_offsets,
            [bs, num_query, self.num_heads, self.num_levels, self.num_points, 2],
        )

        # Generate attention weights: [bs, num_query, num_heads, num_levels * num_points]
        attention_weights = self.attention_weights(query)
        attention_weights = F.Reshape(
            attention_weights,
            [bs, num_query, self.num_heads, self.num_levels * self.num_points],
        )

        # Apply softmax to attention weights
        attention_weights = F.Softmax(attention_weights, axis=-1)

        # Reshape attention weights: [bs, num_query, num_heads, num_levels, num_points]
        attention_weights = F.Reshape(
            attention_weights,
            [bs, num_query, self.num_heads, self.num_levels, self.num_points],
        )

        # Compute sampling locations
        ref_points_shape = F.Shape(reference_points, -1)

        if ref_points_shape == 2:
            # reference_points: [bs, num_query, num_levels, 2]
            # Normalize offsets by spatial shapes
            # offset_normalizer: [num_levels, 2] with [W, H] order
            spatial_shapes_wh = F.ConcatX(
                [
                    F.SliceF(spatial_shapes, [0, 1], [self.num_levels, 2], [1, 1]),  # type: ignore[call-arg,misc] # W
                    F.SliceF(spatial_shapes, [0, 0], [self.num_levels, 1], [1, 1]),  # type: ignore[call-arg,misc] # H
                ],  # type: ignore[call-arg,misc] # H
                axis=1,
            )

            # reference_points: [bs, num_query, 1, num_levels, 1, 2]
            ref_points_expanded = F.Unsqueeze(
                F.Unsqueeze(reference_points, axis=2), axis=4
            )

            # offset_normalizer: [1, 1, 1, num_levels, 1, 2]
            offset_norm = F.Unsqueeze(
                F.Unsqueeze(F.Unsqueeze(spatial_shapes_wh, axis=0), axis=0), axis=0
            )
            offset_norm = F.Unsqueeze(offset_norm, axis=4)

            # Compute sampling locations
            normalized_offsets = F.Div(sampling_offsets, offset_norm)
            sampling_locations = F.Add(ref_points_expanded, normalized_offsets)

        elif ref_points_shape == 4:
            # reference_points: [bs, num_query, num_levels, 4] (x, y, w, h)
            # Extract (x, y) and (w, h)
            ref_xy = F.SliceF(
                reference_points,
                [0, 0, 0, 0],  # type: ignore[call-arg,misc]
                [bs, num_query, self.num_levels, 2],
                [1, 1, 1, 1],
            )
            ref_wh = F.SliceF(
                reference_points,
                [0, 0, 0, 2],  # type: ignore[call-arg,misc]
                [bs, num_query, self.num_levels, 2],
                [1, 1, 1, 1],
            )

            # Expand dimensions: [bs, num_query, 1, num_levels, 1, 2]
            ref_xy_exp = F.Unsqueeze(F.Unsqueeze(ref_xy, axis=2), axis=4)
            ref_wh_exp = F.Unsqueeze(F.Unsqueeze(ref_wh, axis=2), axis=4)

            # sampling_locations = ref_xy + offsets / num_points * ref_wh * 0.5
            scale = F.Constant(0.5 / self.num_points)
            scaled_offsets = F.Mul(F.Mul(sampling_offsets, ref_wh_exp), scale)
            sampling_locations = F.Add(ref_xy_exp, scaled_offsets)
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, "
                f"but got {ref_points_shape} instead."
            )

        # Apply multi-scale deformable attention (use CPU implementation)
        output = self._ms_deform_attn_core(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
        )

        # Output projection
        output = self.output_proj(output)

        # Convert back to original format if needed
        if not self.batch_first:
            # [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            output = F.Transpose(output, perm=(1, 0, 2))

        # Add residual connection
        output = F.Add(output, identity)

        return output

    def _ms_deform_attn_core(
        self,
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
    ):
        """
        Core multi-scale deformable attention computation.

        This is a TTSim implementation using bilinear sampling via GridSample.

        Args:
            value: [bs, num_value, num_heads, dim_per_head]
            spatial_shapes: [num_levels, 2]
            level_start_index: [num_levels]
            sampling_locations: [bs, num_query, num_heads, num_levels, num_points, 2]
            attention_weights: [bs, num_query, num_heads, num_levels, num_points]

        Returns:
            output: [bs, num_query, embed_dims]
        """
        bs = F.Shape(value, 0)
        num_query = F.Shape(sampling_locations, 1)
        num_heads = F.Shape(value, 2)
        dim_per_head = F.Shape(value, 3)
        num_levels = self.num_levels
        num_points = self.num_points

        output_list = []

        # Process each level
        for lvl in range(num_levels):
            # Get spatial shape for this level
            H = spatial_shapes[lvl, 0]
            W = spatial_shapes[lvl, 1]

            # Extract value for this level
            start_idx = level_start_index[lvl] if lvl < len(level_start_index) else 0
            end_idx = (
                level_start_index[lvl + 1]
                if (lvl + 1) < len(level_start_index)
                else F.Shape(value, 1)
            )

            value_l = F.SliceF(
                value,
                [0, start_idx, 0, 0],  # type: ignore[call-arg,misc]
                [bs, end_idx - start_idx, num_heads, dim_per_head],
                [1, 1, 1, 1],
            )

            # Reshape value_l: [bs, H, W, num_heads, dim_per_head]
            value_l = F.Reshape(value_l, [bs, H, W, num_heads, dim_per_head])

            # Get sampling locations for this level: [bs, num_query, num_heads, num_points, 2]
            sampling_locations_l = F.SliceF(  # type: ignore[call-arg,misc]
                sampling_locations,
                [0, 0, 0, lvl, 0, 0],
                [bs, num_query, num_heads, 1, num_points, 2],
                [1, 1, 1, 1, 1, 1],
            )
            sampling_locations_l = F.Squeeze(sampling_locations_l, axis=3)

            # Reshape for grid_sample: [bs * num_heads, num_query * num_points, 2]
            sampling_locations_l = F.Reshape(
                sampling_locations_l, [bs * num_heads, num_query * num_points, 2]
            )

            # Reshape value for grid_sample: [bs * num_heads, H, W, dim_per_head]
            value_l = F.Reshape(value_l, [bs * num_heads, H, W, dim_per_head])

            # Apply bilinear sampling using GridSample
            # GridSample expects coordinates in [-1, 1] range, we have [0, 1]
            # Convert: coord_normalized = coord * 2 - 1
            sampling_locations_normalized = F.Sub(F.Mul(sampling_locations_l, 2.0), 1.0)

            # Sample features
            sampled_l = F.GridSample(
                f"msda.grid_sample_l{lvl}",
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )(value_l, sampling_locations_normalized)

            # Reshape back: [bs, num_heads, num_query, num_points, dim_per_head]
            sampled_l = F.Reshape(
                sampled_l, [bs, num_heads, num_query, num_points, dim_per_head]
            )

            # Transpose to match attention weights: [bs, num_query, num_heads, num_points, dim_per_head]
            sampled_l = F.Transpose(sampled_l, perm=(0, 2, 1, 3, 4))

            output_list.append(sampled_l)

        # Stack all levels: [bs, num_query, num_heads, num_levels, num_points, dim_per_head]
        output = F.Stack(output_list, axis=3)

        # Expand attention weights: [bs, num_query, num_heads, num_levels, num_points, 1]
        attention_weights_exp = F.Unsqueeze(attention_weights, axis=-1)

        # Apply attention: weighted sum over levels and points
        output = F.Mul(output, attention_weights_exp)
        output = F.ReduceSum(output, axes=(3, 4))  # Sum over levels and points

        # Reshape to [bs, num_query, embed_dims]
        output = F.Reshape(output, [bs, num_query, num_heads * dim_per_head])

        return output

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        count = 0
        count += self.sampling_offsets.analytical_param_count(lvl=0)
        count += self.attention_weights.analytical_param_count(lvl=0)
        count += self.value_proj.analytical_param_count(lvl=0)
        count += self.output_proj.analytical_param_count(lvl=0)
        return count


class DetectionTransformerDecoder(Module):
    """
    Detection Transformer Decoder for BEVFormer.

    Implements iterative refinement of object queries using multi-scale
    deformable attention. Each layer refines the query features and
    optionally updates reference points through regression branches.

    Args:
        name (str): Module name
        transformerlayers (dict): Configuration for transformer layers
        num_layers (int): Number of decoder layers
        return_intermediate (bool): Whether to return intermediate outputs
            from all layers (default: False)
    """

    def __init__(self, name, transformerlayers, num_layers, return_intermediate=False):
        super().__init__()
        self.name = name
        self.return_intermediate = return_intermediate
        self.num_layers = num_layers

        # Build decoder layers
        self.layers = []
        for i in range(num_layers):
            # Each layer is a CustomMSDeformableAttention module
            layer = self._build_layer(f"{name}_layer{i}", transformerlayers)
            self.layers.append(layer)

    def _build_layer(self, layer_name, layer_cfg):
        """
        Build a single decoder layer.

        Args:
            layer_name (str): Name for the layer
            layer_cfg (dict): Layer configuration

        Returns:
            CustomMSDeformableAttention: Decoder layer module
        """
        # Extract configuration
        embed_dims = layer_cfg.get("embed_dims", 256)
        num_heads = layer_cfg.get("num_heads", 8)
        num_levels = layer_cfg.get("num_levels", 4)
        num_points = layer_cfg.get("num_points", 4)
        dropout = layer_cfg.get("dropout", 0.1)
        batch_first = layer_cfg.get("batch_first", False)

        return CustomMSDeformableAttention(
            name=layer_name,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        reg_branches=None,
        **kwargs,
    ):
        """
        Forward pass through decoder layers.

        Args:
            query: [num_query, bs, embed_dims] - Object queries
            key: Feature keys (not used in deformable attention)
            value: [num_value, bs, embed_dims] - Feature values
            query_pos: Positional encoding for queries
            key_pos: Not used
            attn_masks: Attention masks
            query_key_padding_mask: Not used
            key_padding_mask: [bs, num_key] - Mask for padded features
            reference_points: [bs, num_query, 2 or 3] - Initial reference points
            spatial_shapes: [num_levels, 2] - Spatial shapes of feature levels
            level_start_index: [num_levels] - Start indices for each level
            reg_branches: Optional regression branches for iterative refinement

        Returns:
            output: [num_query, bs, embed_dims] or list of intermediate outputs
            reference_points: Final or list of intermediate reference points
        """
        output = query
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            # Prepare reference points for this layer
            # Take only (x, y) coordinates and expand for multi-level
            # reference_points_input: [bs, num_query, num_levels, 2]
            reference_points_xy = F.SliceF(  # type: ignore[call-arg,misc]
                reference_points,
                [0, 0, 0],
                [F.Shape(reference_points, 0), F.Shape(reference_points, 1), 2],
                [1, 1, 1],
            )
            reference_points_input = F.Unsqueeze(reference_points_xy, axis=2)

            # Forward through layer
            output = layer(
                query=output,
                key=key,
                value=value,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                **kwargs,
            )

            # Permute for regression: [num_query, bs, embed_dims] -> [bs, num_query, embed_dims]
            output = F.Transpose(output, perm=(1, 0, 2))

            # Apply regression branch for iterative refinement
            if reg_branches is not None:
                # reg_branches[lid](output) predicts delta for reference points
                # tmp: [bs, num_query, output_dims]
                tmp = reg_branches[lid](output)

                # reference_points should have shape [bs, num_query, 3] (x, y, z)
                assert F.Shape(reference_points, -1) == 3

                # Create new reference points
                new_reference_points = F.Constant(0.0)  # Placeholder

                # Update (x, y) coordinates
                # new_xy = sigmoid(inverse_sigmoid(old_xy) + delta_xy)
                old_xy = F.SliceF(
                    reference_points,
                    [0, 0, 0],  # type: ignore[call-arg,misc]
                    [F.Shape(reference_points, 0), F.Shape(reference_points, 1), 2],
                    [1, 1, 1],
                )
                delta_xy = F.SliceF(
                    tmp,
                    [0, 0, 0],  # type: ignore[call-arg,misc]
                    [F.Shape(tmp, 0), F.Shape(tmp, 1), 2],
                    [1, 1, 1],
                )

                new_xy = F.Add(delta_xy, inverse_sigmoid(old_xy))
                new_xy = F.Sigmoid(new_xy)

                # Update z coordinate
                old_z = F.SliceF(
                    reference_points,
                    [0, 0, 2],  # type: ignore[call-arg,misc]
                    [F.Shape(reference_points, 0), F.Shape(reference_points, 1), 1],
                    [1, 1, 1],
                )
                delta_z = F.SliceF(
                    tmp,
                    [0, 0, 4],  # type: ignore[call-arg,misc]
                    [F.Shape(tmp, 0), F.Shape(tmp, 1), 1],
                    [1, 1, 1],
                )

                new_z = F.Add(delta_z, inverse_sigmoid(old_z))
                new_z = F.Sigmoid(new_z)

                # Concatenate to form new reference points
                new_reference_points = F.ConcatX([new_xy, new_z], axis=-1)

                # Detach for next iteration (stop gradient in training)
                reference_points = new_reference_points

            # Permute back: [bs, num_query, embed_dims] -> [num_query, bs, embed_dims]
            output = F.Transpose(output, perm=(1, 0, 2))

            # Store intermediate results if needed
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # Return results
        if self.return_intermediate:
            return F.Stack(intermediate, axis=0), F.Stack(
                intermediate_reference_points, axis=0
            )

        return output, reference_points

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        return sum(layer.analytical_param_count() for layer in self.layers)
