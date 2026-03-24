#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for CustomMSDeformableAttention TTSim module.

This test suite validates the TTSim implementation of CustomMSDeformableAttention
against PyTorch reference implementation with comprehensive numerical comparison.
"""

import sys
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

# PyTorch imports (for reference implementation)
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import math

# TTSim imports
from workloads.MapTracker.plugin.models.transformer_utils.custom_msdeformable_attention import (
    CustomMSDeformableAttention,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.init_utils import (
    xavier_init,
    constant_init,
)
import ttsim.front.functional.op as F
from ttsim.ops.tensor import SimTensor

print("=" * 80)
print("CustomMSDeformableAttention TTSim Module Test Suite")
print("=" * 80)
print()


# =============================================================================
# PyTorch Deformable Attention Core Function
# =============================================================================


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Pure PyTorch CPU implementation of multi-scale deformable attention.

    This is the core deformable attention algorithm used for numerical validation.
    Copied from maptracker fp16_dattn.py to avoid mmcv dependencies.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of each feature map,
            has shape (num_levels, 2), last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    # Split value by spatial levels
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Convert sampling locations from [0, 1] to grid_sample range [-1, 1]
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # Reshape value for grid_sample
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )

        # Get sampling grid for this level
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # Bilinear sampling at the deformed positions
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F_torch.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # Aggregate with attention weights
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    # Weighted sum of sampled values
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )

    return output.transpose(1, 2).contiguous()


# =============================================================================
# PyTorch Reference Implementation
# =============================================================================


class CustomMSDeformableAttentionPyTorch(nn.Module):
    """PyTorch reference implementation for comparison."""

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        use_sampling_offsets=True,
        batch_first=False,
    ):
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads")

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.dropout_rate = dropout
        self.use_sampling_offsets = use_sampling_offsets
        self.batch_first = batch_first

        # Linear layers
        if use_sampling_offsets:
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2
            )

        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )

        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        if self.use_sampling_offsets:
            constant_init(self.sampling_offsets, 0.0)
            thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
                2.0 * math.pi / self.num_heads
            )
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (
                (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
                .view(self.num_heads, 1, 1, 2)
                .repeat(1, self.num_levels, self.num_points, 1)
            )
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1

            self.sampling_offsets.bias.data = grid_init.view(-1)

        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

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
        """Forward pass."""

        if value is None:
            value = query
        # Note: identity is kept as None unless provided (no residual by default)
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        # Value projection
        value = self.value_proj(value)

        # Apply key padding mask
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.view(bs, num_value, self.num_heads, -1)

        # Sampling offsets
        if self.use_sampling_offsets:
            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
            )
        else:
            sampling_offsets = query.new_zeros(
                (bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
            )

        # Attention weights
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Offset normalizer: [W, H] from spatial_shapes
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )

        # Reference points shape check
        _, _, num_ref_points, _ = reference_points.shape

        # Broadcast ALL reference points (correct PyTorch source behavior)
        # (bs, num_query, num_points, 2) -> (bs, num_query, 1, 1, num_points, 2)
        reference_points = reference_points[:, :, None, None, :, :]

        # Compute sampling locations
        sampling_locations = reference_points + (
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )

        # Use actual PyTorch deformable attention implementation
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        # Output projection
        output = self.output_proj(output)

        # Dropout
        output = self.dropout(output)

        # Residual connection (only if identity is provided)
        if identity is not None:
            output = output + identity

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


# =============================================================================
# Helper Functions
# =============================================================================


def initialize_linear_weights_with_data(linear_layer, weight_data, bias_data=None):
    """
    Initialize a TTSim Linear layer's weights with actual data for numerical validation.

    Args:
        linear_layer: TTSim Linear layer instance
        weight_data: NumPy array of shape [out_features, in_features] (same layout as PyTorch)
        bias_data: Optional NumPy array of shape [out_features]
    """
    # Ensure data is contiguous and correct dtype
    weight_data = np.ascontiguousarray(weight_data, dtype=np.float32)

    # Replace the param tensor with one that has data
    linear_layer.param = F._from_data(
        linear_layer.param.name, weight_data, is_const=True
    )
    linear_layer.param.is_param = True
    linear_layer.param.set_module(linear_layer)
    linear_layer._tensors[linear_layer.param.name] = linear_layer.param

    if bias_data is not None and linear_layer.bias is not None:
        bias_data = np.ascontiguousarray(bias_data, dtype=np.float32)
        linear_layer.bias = F._from_data(
            linear_layer.bias.name, bias_data, is_const=True
        )
        linear_layer.bias.is_param = True
        linear_layer.bias.set_module(linear_layer)
        linear_layer._tensors[linear_layer.bias.name] = linear_layer.bias


def initialize_ttsim_attention_params(ttsim_model, pytorch_model):
    """
    Initialize TTSim model parameters from PyTorch model.

    Copies initialized weights from PyTorch (after init_weights() has been called)
    to TTSim Linear layers.

    Args:
        ttsim_model: TTSim CustomMSDeformableAttention instance
        pytorch_model: PyTorch CustomMSDeformableAttentionPyTorch instance (already initialized)
    """

    # Sampling offsets (if used)
    if ttsim_model.use_sampling_offsets and pytorch_model.use_sampling_offsets:
        # PyTorch weight is [out_features, in_features], same as TTSim
        weight_np = pytorch_model.sampling_offsets.weight.detach().cpu().numpy().copy()
        bias_np = pytorch_model.sampling_offsets.bias.detach().cpu().numpy().copy()
        print(
            f"     - sampling_offsets: weight shape {weight_np.shape}, bias shape {bias_np.shape}"
        )
        initialize_linear_weights_with_data(
            ttsim_model.sampling_offsets, weight_np, bias_np
        )

    # Attention weights
    weight_np = pytorch_model.attention_weights.weight.detach().cpu().numpy().copy()
    bias_np = pytorch_model.attention_weights.bias.detach().cpu().numpy().copy()
    print(
        f"     - attention_weights: weight shape {weight_np.shape}, bias shape {bias_np.shape}"
    )
    initialize_linear_weights_with_data(
        ttsim_model.attention_weights, weight_np, bias_np
    )

    # Value projection
    weight_np = pytorch_model.value_proj.weight.detach().cpu().numpy().copy()
    bias_np = pytorch_model.value_proj.bias.detach().cpu().numpy().copy()
    print(
        f"     - value_proj: weight shape {weight_np.shape}, bias shape {bias_np.shape}"
    )
    initialize_linear_weights_with_data(ttsim_model.value_proj, weight_np, bias_np)

    # Output projection
    weight_np = pytorch_model.output_proj.weight.detach().cpu().numpy().copy()
    bias_np = pytorch_model.output_proj.bias.detach().cpu().numpy().copy()
    print(
        f"     - output_proj: weight shape {weight_np.shape}, bias shape {bias_np.shape}"
    )
    initialize_linear_weights_with_data(ttsim_model.output_proj, weight_np, bias_np)


def compare_outputs(pytorch_output, ttsim_output, name="Output", rtol=1e-5, atol=1e-6):
    """Compare PyTorch and TTSim outputs using np.allclose."""

    if isinstance(ttsim_output, SimTensor):
        ttsim_np = ttsim_output.data
        if ttsim_np is None:
            print(f"\n[X] ERROR: TTSim output.data is None!")
            print(
                f"  This means the computation graph was built but data wasn't computed."
            )
            print(f"  The tensor is: {ttsim_output}")
            return False
    else:
        ttsim_np = ttsim_output

    pytorch_np = pytorch_output.detach().numpy()

    print(f"\n{name} Comparison:")
    print(f"  PyTorch shape: {pytorch_output.shape}")
    print(f"  PyTorch stats: mean={pytorch_np.mean():.6f}, std={pytorch_np.std():.6f}")
    print(f"  PyTorch range: [{pytorch_np.min():.6f}, {pytorch_np.max():.6f}]")
    print(f"  TTSim shape: {ttsim_np.shape}")
    print(f"  TTSim stats: mean={ttsim_np.mean():.6f}, std={ttsim_np.std():.6f}")
    print(f"  TTSim range: [{ttsim_np.min():.6f}, {ttsim_np.max():.6f}]")

    diff = np.abs(pytorch_np - ttsim_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = np.median(diff)

    print(f"\n  Absolute differences:")
    print(f"    Max:    {max_diff:.6e}")
    print(f"    Mean:   {mean_diff:.6e}")
    print(f"    Median: {median_diff:.6e}")

    # Use np.allclose for proper floating-point comparison
    if np.allclose(pytorch_np, ttsim_np, rtol=rtol, atol=atol):
        print(f"\n  [PASS] [PASS] Outputs match (rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"\n  [FAIL] [FAIL] Outputs don't match (rtol={rtol}, atol={atol})")
        print(f"     Max difference: {max_diff:.6e}")

        # Show example of largest difference
        max_idx = np.unravel_index(diff.argmax(), diff.shape)
        print(f"\n  Largest diff at {max_idx}:")
        print(f"    PyTorch: {pytorch_np[max_idx]:.6f}")
        print(f"    TTSim:   {ttsim_np[max_idx]:.6f}")
        print(f"    Diff:    {diff[max_idx]:.6f}")
        return False


# =============================================================================
# Test 1: Module Construction
# =============================================================================


def test_custom_attention_construction():
    """Test CustomMSDeformableAttention construction."""
    print("\n" + "=" * 80)
    print("TEST 1: CustomMSDeformableAttention Construction")
    print("=" * 80)

    try:
        # Test with default parameters
        print("\n1. Testing default configuration:")
        model = CustomMSDeformableAttention(
            name="test_attention",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            dropout=0.1,
            use_sampling_offsets=True,
            batch_first=True,
        )

        print(f"   [OK] Module constructed successfully")
        print(f"   - Module name: {model.name}")
        print(f"   - Embed dims: {model.embed_dims}")
        print(f"   - Num heads: {model.num_heads}")
        print(f"   - Num levels: {model.num_levels}")
        print(f"   - Num points: {model.num_points}")
        print(f"   - Use sampling offsets: {model.use_sampling_offsets}")
        print(f"   - Batch first: {model.batch_first}")

        # Test parameter counting
        print("\n2. Testing parameter counting:")
        param_count = model.analytical_param_count(lvl=2)
        print(f"   [OK] Total parameters: {param_count:,}")

        # Test without sampling offsets
        print("\n3. Testing without sampling offsets:")
        model_no_offsets = CustomMSDeformableAttention(
            name="test_attention_no_offsets",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            dropout=0.1,
            use_sampling_offsets=False,
            batch_first=True,
        )
        print(f"   [OK] Module without sampling offsets constructed")
        param_count_no_offsets = model_no_offsets.analytical_param_count(lvl=1)

        print(f"\n   Parameter reduction: {param_count - param_count_no_offsets:,}")

        print("\n" + "=" * 80)
        print("TEST 1: [OK] PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n[X] TEST 1 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 2: Forward Pass
# =============================================================================


def test_custom_attention_forward():
    """Test CustomMSDeformableAttention forward pass."""
    print("\n" + "=" * 80)
    print("TEST 2: CustomMSDeformableAttention Forward Pass")
    print("=" * 80)

    try:
        # Configuration
        batch_size = 2
        num_query = 100
        embed_dims = 128
        num_heads = 4
        num_levels = 3
        num_points = 4

        # Spatial shapes must be defined before num_value
        # spatial_shapes: (num_levels, 2) as [H, W]
        spatial_shapes_np = np.array([[50, 50], [25, 25], [13, 13]], dtype=np.int32)
        # num_value must equal sum of H*W across all levels
        num_value = int(
            np.sum(spatial_shapes_np[:, 0] * spatial_shapes_np[:, 1])
        )  # 2500 + 625 + 169 = 3294

        print(f"\nConfiguration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Num queries: {num_query}")
        print(f"  Num values: {num_value} (computed from spatial_shapes)")
        print(f"  Embed dims: {embed_dims}")
        print(f"  Num heads: {num_heads}")
        print(f"  Num levels: {num_levels}")
        print(f"  Num points: {num_points}")
        print(f"  Spatial shapes: {spatial_shapes_np.tolist()}")

        # Create PyTorch model
        print("\n1. Creating PyTorch reference model...")
        model_pytorch = CustomMSDeformableAttentionPyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=0.0,  # Disable dropout for testing
            use_sampling_offsets=True,
            batch_first=True,
        )
        model_pytorch.eval()

        # Verify PyTorch weights are initialized
        print(
            f"   PyTorch sampling_offsets weight: {model_pytorch.sampling_offsets.weight.shape}"
        )
        print(
            f"   PyTorch sampling_offsets bias: {model_pytorch.sampling_offsets.bias.shape}"
        )
        print(
            f"   PyTorch attention_weights weight: {model_pytorch.attention_weights.weight.shape}"
        )
        print(f"   PyTorch value_proj weight: {model_pytorch.value_proj.weight.shape}")
        print(
            f"   PyTorch output_proj weight: {model_pytorch.output_proj.weight.shape}"
        )

        # Create TTSim model
        print("2. Creating TTSim model...")
        model_ttsim = CustomMSDeformableAttention(
            name="test_forward_attention",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=0.0,
            use_sampling_offsets=True,
            batch_first=True,
        )

        # Create test inputs
        print("3. Creating test inputs...")
        np.random.seed(42)

        query_np = np.random.randn(batch_size, num_query, embed_dims).astype(np.float32)
        value_np = np.random.randn(batch_size, num_value, embed_dims).astype(np.float32)

        # Reference points: (bs, num_query, num_points, 2)
        reference_points_np = np.random.rand(
            batch_size, num_query, num_points, 2
        ).astype(np.float32)

        # Note: spatial_shapes already defined above when computing num_value

        # Level start indices
        level_start_index_np = np.array([0, 2500, 3125], dtype=np.int32)

        # Convert to PyTorch
        query_torch = torch.from_numpy(query_np)
        value_torch = torch.from_numpy(value_np)
        reference_points_torch = torch.from_numpy(reference_points_np)
        spatial_shapes_torch = torch.from_numpy(spatial_shapes_np)
        level_start_index_torch = torch.from_numpy(level_start_index_np)

        # Convert to TTSim
        query_ttsim = F._from_data("query", query_np, is_const=False)
        value_ttsim = F._from_data("value", value_np, is_const=False)
        reference_points_ttsim = F._from_data(
            "reference_points", reference_points_np, is_const=False
        )
        spatial_shapes_ttsim = F._from_data(
            "spatial_shapes", spatial_shapes_np, is_const=True
        )
        level_start_index_ttsim = F._from_data(
            "level_start_index", level_start_index_np, is_const=True
        )

        # Initialize TTSim params from PyTorch
        print("4. Initializing TTSim parameters from PyTorch...")
        print(
            f"   Copying {model_pytorch.sampling_offsets.weight.numel():,} sampling_offsets weights"
        )
        print(
            f"   Copying {model_pytorch.attention_weights.weight.numel():,} attention_weights weights"
        )
        print(
            f"   Copying {model_pytorch.value_proj.weight.numel():,} value_proj weights"
        )
        print(
            f"   Copying {model_pytorch.output_proj.weight.numel():,} output_proj weights"
        )
        initialize_ttsim_attention_params(model_ttsim, model_pytorch)
        print("   [OK] All weights copied successfully")

        # Run PyTorch forward
        print("5. Running PyTorch forward pass...")
        with torch.no_grad():
            output_pytorch = model_pytorch(
                query=query_torch,
                value=value_torch,
                reference_points=reference_points_torch,
                spatial_shapes=spatial_shapes_torch,
                level_start_index=level_start_index_torch,
            )

        # Run TTSim forward
        print("6. Running TTSim forward pass...")
        output_ttsim = model_ttsim(
            query=query_ttsim,
            value=value_ttsim,
            reference_points=reference_points_ttsim,
            spatial_shapes=spatial_shapes_ttsim,
            level_start_index=level_start_index_ttsim,
        )

        # Compare outputs
        print("7. Comparing outputs...")
        match = compare_outputs(
            output_pytorch, output_ttsim, "Attention Output", rtol=1e-5, atol=1e-5
        )

        if match:
            print("\n" + "=" * 80)
            print("TEST 2: [OK] PASSED")
            print("=" * 80)
            return True
        else:
            print("\n" + "=" * 80)
            print("TEST 2: [X] FAILED - Outputs don't match")
            print("=" * 80)
            return False

    except Exception as e:
        print(f"\n[X] TEST 2 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# TEST 3: Multi-Point Reference Points Verification
# =============================================================================


def test_multipoint_reference_points():
    """Verify that ALL reference points are used when num_ref_pts == num_points.


    This test creates reference_points whose points are far apart
    (corners of the BEV grid) with num_ref_pts == num_points to ensure
    all points are used in the matching case.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Multi-Point Reference Points (all points vs first-only)")
    print("=" * 80)

    try:
        batch_size = 2
        num_query = 20
        embed_dims = 128
        num_heads = 4
        num_levels = 2
        num_points = 4  # each query has 4 distinct reference points

        spatial_shapes_np = np.array([[16, 16], [8, 8]], dtype=np.int32)
        num_value = int(np.sum(spatial_shapes_np[:, 0] * spatial_shapes_np[:, 1]))

        np.random.seed(123)

        # Create reference points with VERY different locations per point
        # Point 0 near top-left, point 1 near top-right, etc.
        ref_pts_np = np.zeros((batch_size, num_query, num_points, 2), dtype=np.float32)
        ref_pts_np[:, :, 0, :] = 0.1  # top-left
        ref_pts_np[:, :, 1, :] = 0.9  # bottom-right
        ref_pts_np[:, :, 2, 0] = 0.1
        ref_pts_np[:, :, 2, 1] = 0.9  # top-right
        ref_pts_np[:, :, 3, 0] = 0.9
        ref_pts_np[:, :, 3, 1] = 0.1  # bottom-left

        query_np = np.random.randn(batch_size, num_query, embed_dims).astype(np.float32)
        value_np = np.random.randn(batch_size, num_value, embed_dims).astype(np.float32)

        # --- PyTorch reference (correct: uses ALL points) ---
        model_pt = CustomMSDeformableAttentionPyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=0.0,
            batch_first=True,
        )
        model_pt.eval()

        # --- TTSim model ---
        model_tt = CustomMSDeformableAttention(
            name="test_multiref",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=0.0,
            batch_first=True,
        )
        initialize_ttsim_attention_params(model_tt, model_pt)

        # Run PyTorch
        with torch.no_grad():
            out_pt = model_pt(
                query=torch.from_numpy(query_np),
                value=torch.from_numpy(value_np),
                reference_points=torch.from_numpy(ref_pts_np),
                spatial_shapes=torch.from_numpy(spatial_shapes_np),
            ).numpy()

        # Run TTSim
        out_tt = model_tt(
            query=F._from_data("t3_q", query_np, is_const=False),
            value=F._from_data("t3_v", value_np, is_const=False),
            reference_points=F._from_data("t3_ref", ref_pts_np, is_const=False),
            spatial_shapes=F._from_data("t3_ss", spatial_shapes_np, is_const=True),
        ).data

        match = compare_outputs(
            torch.from_numpy(out_pt),
            out_tt,
            "Multi-point reference output",
            rtol=1e-4,
            atol=1e-4,
        )

        if match:
            print("\n[OK] TEST 3 PASSED: All reference points correctly used")
        else:
            print("\n[X] TEST 3 FAILED")
        return match

    except Exception as e:
        print(f"\n[X] TEST 3 FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Run All Tests
# =============================================================================

if __name__ == "__main__":
    print("\nRunning CustomMSDeformableAttention Test Suite...")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Construction", test_custom_attention_construction()))
    results.append(("Forward Pass", test_custom_attention_forward()))
    results.append(("Multi-Point Reference Points", test_multipoint_reference_points()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [X]")
    print("=" * 80)
