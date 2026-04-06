#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RelPositionEmbedding module for BEVFormer model - TTSim conversion
Converted from PyTorch to TTSim
Original file: workloads/BEVFormer/projects/mmdet3d_plugin/models/utils/position_embedding.py
"""

import os, sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import math
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class RelPositionEmbedding(SimNN.Module):
    """
    Relative Position Embedding module for spatial feature encoding.

    This module generates 2D position embeddings for spatial features using
    sinusoidal position encodings based on normalized coordinates.

    Args:
        name (str): Module name for TTSim graph
        num_pos_feats (int): Number of position features/embedding dimensions (default: 64)
        pos_norm (bool): Whether to apply LayerNorm to position embeddings (default: True)

    Forward Input:
        tensor: Input tensor of shape [B, C, H, W]

    Forward Output:
        x_pos: Position embeddings of shape [H*W, num_pos_feats]

    Notes:
        - Uses cosine/sine encoding for both x and y spatial dimensions
        - Normalizes coordinates to [0, 1] range before encoding
        - Applies pi scaling for better numerical properties
        - Output can be optionally normalized with LayerNorm
        - Position embeddings are computed at module call time based on input spatial dimensions
    """

    def __init__(self, name, num_pos_feats=64, pos_norm=True):
        super().__init__()
        self.name = name
        self.num_pos_feats = num_pos_feats
        self.pos_norm = pos_norm

        # Linear layer to project 4D coordinate features to num_pos_feats dimensions
        # Input: [cos(y*pi), sin(y*pi), cos(x*pi), sin(x*pi)] -> 4 features
        # Output: num_pos_feats features
        self.fc = SimNN.Linear(
            name=name + ".fc",
            in_features=4,
            out_features=self.num_pos_feats,
            bias=False,
        )

        # Optional LayerNorm for position embeddings
        if self.pos_norm:
            self.norm = F.LayerNorm(name + ".norm", self.num_pos_feats)
        else:
            self.norm = None

        # Link operations to module
        super().link_op2module()

    def __call__(self, tensor):
        """
        Generate position embeddings for input tensor spatial dimensions.

        This implementation mimics PyTorch behavior:
        ```python
        # PyTorch original:
        B, C, H, W = tensor.shape
        y_range = torch.arange(H) / float(H - 1)
        y_axis = torch.stack((torch.cos(y_range * math.pi), torch.sin(y_range * math.pi)), dim=1)
        y_axis = y_axis.reshape(H, 1, 2).repeat(1, W, 1).reshape(H * W, 2)
        x_range = torch.arange(W) / float(W - 1)
        x_axis = torch.stack((torch.cos(x_range * math.pi), torch.sin(x_range * math.pi)), dim=1)
        x_axis = x_axis.reshape(1, W, 2).repeat(H, 1, 1).reshape(H * W, 2)
        x_pos = torch.cat((y_axis, x_axis), dim=1)  # [H*W, 4]
        x_pos = self.fc(x_pos)  # [H*W, num_pos_feats]
        if self.pos_norm:
            x_pos = self.norm(x_pos)
        ```

        Args:
            tensor: Input tensor of shape [B, C, H, W]

        Returns:
            x_pos: Position embeddings of shape [H*W, num_pos_feats]
        """

        # Extract spatial dimensions from input tensor
        # tensor shape: [B, C, H, W]
        B, C, H, W = tensor.shape

        # ==================== Y-AXIS ENCODING ====================
        # Create normalized y-axis coordinate range [0, 1]
        # y_range shape: [H]
        # Values: [0/(H-1), 1/(H-1), ..., (H-1)/(H-1)] = [0, ..., 1]
        y_range_data = np.arange(H, dtype=np.float32) / float(H - 1)
        y_range = F._from_data(self.name + ".y_range", y_range_data, is_const=True)

        # Scale by pi: y_range * pi
        pi_value = math.pi
        pi_tensor = F._from_data(
            self.name + ".pi", np.array(pi_value, dtype=np.float32), is_const=True
        )
        y_range_scaled = F.Mul(self.name + ".y_range_scaled")(y_range, pi_tensor)

        # Compute cos(y*pi) and sin(y*pi)
        y_cos = F.Cos(self.name + ".y_cos")(y_range_scaled)
        y_sin = F.Sin(self.name + ".y_sin")(y_range_scaled)

        # Unsqueeze to add dimension for stacking: [H] -> [H, 1]
        y_cos_unsqueezed = F.Unsqueeze(self.name + ".y_cos_unsq")(
            y_cos,
            F._from_data(
                self.name + ".axis_1", np.array([1], dtype=np.int64), is_const=True
            ),
        )
        y_sin_unsqueezed = F.Unsqueeze(self.name + ".y_sin_unsq")(
            y_sin,
            F._from_data(
                self.name + ".axis_1_2", np.array([1], dtype=np.int64), is_const=True
            ),
        )

        # Concatenate cos and sin along last axis: [H, 1] + [H, 1] -> [H, 2]
        y_axis = F.ConcatX(self.name + ".y_axis_concat", axis=1)(
            y_cos_unsqueezed, y_sin_unsqueezed
        )

        # Reshape to [H, 1, 2] for tiling
        y_axis_reshaped = F.Reshape(self.name + ".y_axis_reshape")(
            y_axis,
            F._from_data(
                self.name + ".y_reshape_shape",
                np.array([H, 1, 2], dtype=np.int64),
                is_const=True,
            ),
        )

        # Tile across width dimension: [H, 1, 2] -> [H, W, 2]
        # Tile pattern: [1, W, 1] means repeat 1x along H, Wx along W, 1x along feature dim
        y_axis_tiled = F.Tile(self.name + ".y_axis_tile")(
            y_axis_reshaped,
            F._from_data(
                self.name + ".y_tile_reps",
                np.array([1, W, 1], dtype=np.int64),
                is_const=True,
            ),
        )

        # Reshape to [H*W, 2] for final concatenation
        y_axis_final = F.Reshape(self.name + ".y_axis_final_reshape")(
            y_axis_tiled,
            F._from_data(
                self.name + ".y_final_shape",
                np.array([H * W, 2], dtype=np.int64),
                is_const=True,
            ),
        )

        # ==================== X-AXIS ENCODING ====================
        # Create normalized x-axis coordinate range [0, 1]
        # x_range shape: [W]
        x_range_data = np.arange(W, dtype=np.float32) / float(W - 1)
        x_range = F._from_data(self.name + ".x_range", x_range_data, is_const=True)

        # Scale by pi: x_range * pi
        x_range_scaled = F.Mul(self.name + ".x_range_scaled")(x_range, pi_tensor)

        # Compute cos(x*pi) and sin(x*pi)
        x_cos = F.Cos(self.name + ".x_cos")(x_range_scaled)
        x_sin = F.Sin(self.name + ".x_sin")(x_range_scaled)

        # Unsqueeze to add dimension for stacking: [W] -> [W, 1]
        x_cos_unsqueezed = F.Unsqueeze(self.name + ".x_cos_unsq")(
            x_cos,
            F._from_data(
                self.name + ".axis_1_3", np.array([1], dtype=np.int64), is_const=True
            ),
        )
        x_sin_unsqueezed = F.Unsqueeze(self.name + ".x_sin_unsq")(
            x_sin,
            F._from_data(
                self.name + ".axis_1_4", np.array([1], dtype=np.int64), is_const=True
            ),
        )

        # Concatenate cos and sin along last axis: [W, 1] + [W, 1] -> [W, 2]
        x_axis = F.ConcatX(self.name + ".x_axis_concat", axis=1)(
            x_cos_unsqueezed, x_sin_unsqueezed
        )

        # Reshape to [1, W, 2] for tiling
        x_axis_reshaped = F.Reshape(self.name + ".x_axis_reshape")(
            x_axis,
            F._from_data(
                self.name + ".x_reshape_shape",
                np.array([1, W, 2], dtype=np.int64),
                is_const=True,
            ),
        )

        # Tile across height dimension: [1, W, 2] -> [H, W, 2]
        # Tile pattern: [H, 1, 1] means repeat Hx along H, 1x along W, 1x along feature dim
        x_axis_tiled = F.Tile(self.name + ".x_axis_tile")(
            x_axis_reshaped,
            F._from_data(
                self.name + ".x_tile_reps",
                np.array([H, 1, 1], dtype=np.int64),
                is_const=True,
            ),
        )

        # Reshape to [H*W, 2] for final concatenation
        x_axis_final = F.Reshape(self.name + ".x_axis_final_reshape")(
            x_axis_tiled,
            F._from_data(
                self.name + ".x_final_shape",
                np.array([H * W, 2], dtype=np.int64),
                is_const=True,
            ),
        )

        # ==================== COMBINE AND PROJECT ====================
        # Concatenate y and x features along last dimension
        # Result: [H*W, 4] where each row is [cos(y*pi), sin(y*pi), cos(x*pi), sin(x*pi)]
        x_pos = F.ConcatX(self.name + ".concat_xy", axis=1)(y_axis_final, x_axis_final)

        # Project from 4D to num_pos_feats dimensions using linear layer
        # Shape: [H*W, 4] -> [H*W, num_pos_feats]
        x_pos = self.fc(x_pos)

        # Apply optional LayerNorm
        if self.pos_norm:
            x_pos = self.norm(x_pos)

        return x_pos

    def analytical_param_count(self):
        """
        Calculate the number of trainable parameters in this module.

        Returns:
            int: Total parameter count

        Parameters:
            - fc.weight: [4, num_pos_feats] = 4 * num_pos_feats
            - norm.scale: [num_pos_feats] if pos_norm else 0
            - norm.bias: [num_pos_feats] if pos_norm else 0
        """
        # Linear layer weight (no bias)
        param_count = 4 * self.num_pos_feats

        # LayerNorm parameters (scale and bias)
        if self.pos_norm:
            param_count += 2 * self.num_pos_feats

        return param_count


# Helper function for creating RelPositionEmbedding module
def create_rel_position_embedding(name, num_pos_feats=64, pos_norm=True):
    """
    Factory function to create RelPositionEmbedding module.

    Args:
        name (str): Module name for TTSim graph
        num_pos_feats (int): Number of position features (default: 64)
        pos_norm (bool): Whether to apply LayerNorm (default: True)

    Returns:
        RelPositionEmbedding: Initialized position embedding module

    Example:
        >>> pos_embed = create_rel_position_embedding('bevformer.pos_embed', num_pos_feats=256)
        >>> # Input tensor: [batch=1, channels=256, height=200, width=200]
        >>> input_tensor = F._from_shape('input', [1, 256, 200, 200])
        >>> # Output: [40000, 256] position embeddings for 200x200 grid
        >>> pos_embeddings = pos_embed(input_tensor)
    """
    return RelPositionEmbedding(name, num_pos_feats, pos_norm)


if __name__ == "__main__":
    """
    Test script for RelPositionEmbedding module.

    This script creates a simple test case to verify the module construction
    and forward pass shape inference.
    """
    logger.info("=" * 80)
    logger.info("Testing RelPositionEmbedding TTSim Module")
    logger.info("=" * 80)

    # Create position embedding module
    module_name = "test_rel_pos_embed"
    num_features = 64
    use_norm = True

    logger.info("\nCreating RelPositionEmbedding:")
    logger.info(f"  - Module name: {module_name}")
    logger.info(f"  - Number of position features: {num_features}")
    logger.info(f"  - Position normalization: {use_norm}")

    pos_embed = RelPositionEmbedding(
        name=module_name, num_pos_feats=num_features, pos_norm=use_norm
    )

    # Create test input tensor [B, C, H, W]
    batch_size = 1
    channels = 256
    height = 200
    width = 200

    logger.info("\nCreating test input tensor:")
    logger.info(f"  - Shape: [{batch_size}, {channels}, {height}, {width}]")

    input_tensor = F._from_shape("test_input", [batch_size, channels, height, width])

    # Run forward pass
    logger.info("\nRunning forward pass...")
    output = pos_embed(input_tensor)

    # Expected output shape
    expected_shape = [height * width, num_features]
    logger.info(f"\nExpected output shape: {expected_shape}")
    logger.info(f"Actual output shape: {output.shape}")

    # Calculate parameter count
    param_count = pos_embed.analytical_param_count()
    logger.info(f"\nParameter count: {param_count}")
    logger.info(
        f"  - Linear layer (fc): 4 * {num_features} = {4 * num_features}"
    )
    if use_norm:
        logger.info(
            f"  - LayerNorm (scale + bias): 2 * {num_features} = {2 * num_features}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Test completed successfully!")
    logger.info("=" * 80)
