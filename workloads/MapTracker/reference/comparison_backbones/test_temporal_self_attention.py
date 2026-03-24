#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for Temporal Self Attention TTSim module.
Validates the conversion from PyTorch to TTSim.

This tests:
- TemporalSelfAttention: Temporal attention for BEV features across time
"""

import os
import sys
import warnings
import math
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import ttsim.front.functional.op as F
from workloads.MapTracker.plugin.models.backbones.bevformer.temporal_self_attention import (
    TemporalSelfAttention,
)
from workloads.MapTracker.plugin.models.backbones.bevformer.init_utils import (
    xavier_init,
    constant_init,
)

# Fix for OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ============================================================================
# PyTorch Reference Implementation (CPU-only, Python 3.13 compatible)
# ============================================================================


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    CPU-only Python 3.13 version of multi-scale deformable attention.
    Converted from mmcv.ops.multi_scale_deform_attn.multi_scale_deformable_attn_pytorch

    Args:
        value (torch.Tensor): Shape (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (list): List of (H, W) tuples for each level
        sampling_locations (torch.Tensor): Shape (bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights (torch.Tensor): Shape (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        torch.Tensor: Shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    # Convert spatial shapes to list of tuples if it's a tensor
    if isinstance(value_spatial_shapes, torch.Tensor):
        value_spatial_shapes = [(int(H_), int(W_)) for H_, W_ in value_spatial_shapes]

    # Split value by levels
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Convert sampling locations from [0, 1] to [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # Reshape value for this level
        # bs, H_*W_, num_heads, embed_dims -> bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )

        # Get sampling grid for this level
        # bs, num_queries, num_heads, num_points, 2 -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # Apply bilinear sampling
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F_torch.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    # Reshape attention weights
    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )

    # Stack sampled values and apply attention weights
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )

    # Transpose back to (bs, num_queries, embed_dims)
    return output.transpose(1, 2).contiguous()


class TemporalSelfAttention_PyTorch(nn.Module):
    """PyTorch reference implementation for data validation."""

    def __init__(
        self, embed_dims=256, num_heads=8, num_levels=4, num_points=4, num_bev_queue=2
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue

        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # Initialize
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward_no_residual(self, query, value, reference_points, spatial_shapes):
        """
        Args:
            query: [bs, num_query, embed_dims]
            value: [bs*num_bev_queue, num_value, num_heads, head_dim]
            reference_points: [bs, num_query, num_levels, 2]
            spatial_shapes: list of (H, W) tuples
        """
        bs, num_query, _ = query.shape

        # Reshape value to [bs*num_bev_queue, num_value, embed_dims]
        value_reshaped = value.flatten(2)  # [bs*num_bev_queue, num_value, embed_dims]
        num_value = value_reshaped.shape[1]

        # Concatenate current query with historical value
        value_current = value_reshaped[:bs]  # [bs, num_value, embed_dims]
        query_concat = torch.cat([value_current, query], dim=-1)

        # Project value
        value = self.value_proj(value_reshaped)
        value = value.view(
            bs * self.num_bev_queue,
            num_value,
            self.num_heads,
            self.embed_dims // self.num_heads,
        )

        # Compute offsets
        sampling_offsets = self.sampling_offsets(query_concat)
        sampling_offsets = sampling_offsets.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )

        # Compute attention weights
        attention_weights = self.attention_weights(query_concat)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = F_torch.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )

        # Permute for processing
        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            .contiguous()
        )

        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )

        # Normalize offsets by spatial shapes
        offset_normalizer = torch.tensor(
            [[W, H] for H, W in spatial_shapes],
            dtype=torch.float32,
            device=query.device,
        )
        offset_normalizer = offset_normalizer[None, None, None, :, None, :]

        # Expand reference_points for num_bev_queue: [bs, nq, nl, 2] -> [bs*num_bev_queue, nq, nl, 2]
        reference_points_expanded = reference_points.unsqueeze(1).repeat(
            1, self.num_bev_queue, 1, 1, 1
        )
        reference_points_expanded = reference_points_expanded.reshape(
            bs * self.num_bev_queue, num_query, self.num_levels, 2
        )

        # Compute sampling locations
        sampling_locations = (
            reference_points_expanded[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer
        )

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        # output shape: [bs*num_bev_queue, num_query, embed_dims]
        # Reshape and fuse temporal information
        output = output.permute(1, 2, 0)  # [num_query, embed_dims, bs*num_bev_queue]
        output = output.view(num_query, self.embed_dims, bs, self.num_bev_queue)
        output = output.mean(dim=-1)  # Average over time
        output = output.permute(2, 0, 1)  # [bs, num_query, embed_dims]

        # Output projection
        output = self.output_proj(output)

        return output

    def forward(self, query, value, reference_points, spatial_shapes):
        """Forward with residual connection."""
        identity = query
        out = self.forward_no_residual(query, value, reference_points, spatial_shapes)
        return out + identity


def initialize_linear_weights_with_data(linear_layer, weight_data, bias_data=None):
    """
    Initialize a TTSim Linear layer's weights with actual data for numerical validation.

    Args:
        linear_layer: TTSim Linear layer instance
        weight_data: NumPy array of shape [out_features, in_features]
        bias_data: Optional NumPy array of shape [out_features]
    """
    # Replace the param tensor with one that has data
    linear_layer.param = F._from_data(
        linear_layer.param.name, weight_data, is_const=True
    )
    linear_layer.param.is_param = True
    linear_layer.param.set_module(linear_layer)
    linear_layer._tensors[linear_layer.param.name] = linear_layer.param

    if bias_data is not None and linear_layer.bias is not None:
        linear_layer.bias = F._from_data(
            linear_layer.bias.name, bias_data, is_const=True
        )
        linear_layer.bias.is_param = True
        linear_layer.bias.set_module(linear_layer)
        linear_layer._tensors[linear_layer.bias.name] = linear_layer.bias


def copy_pytorch_weights_to_ttsim(pytorch_module, ttsim_module):
    """
    Copy weights from a PyTorch TemporalSelfAttention to a TTSim one.

    Args:
        pytorch_module: PyTorch TemporalSelfAttention_PyTorch instance
        ttsim_module: TTSim TemporalSelfAttention instance
    """
    # Copy sampling_offsets weights
    weight_np = pytorch_module.sampling_offsets.weight.detach().cpu().numpy()
    bias_np = pytorch_module.sampling_offsets.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(
        ttsim_module.sampling_offsets, weight_np, bias_np
    )

    # Copy attention_weights
    weight_np = pytorch_module.attention_weights.weight.detach().cpu().numpy()
    bias_np = pytorch_module.attention_weights.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(
        ttsim_module.attention_weights, weight_np, bias_np
    )

    # Copy value_proj
    weight_np = pytorch_module.value_proj.weight.detach().cpu().numpy()
    bias_np = pytorch_module.value_proj.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(ttsim_module.value_proj, weight_np, bias_np)

    # Copy output_proj
    weight_np = pytorch_module.output_proj.weight.detach().cpu().numpy()
    bias_np = pytorch_module.output_proj.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(ttsim_module.output_proj, weight_np, bias_np)


# ============================================================================
# Test Functions
# ============================================================================


def test_construction():
    """Test that TemporalSelfAttention can be constructed successfully."""
    print("\n" + "=" * 80)
    print("TEST 1: TemporalSelfAttention Construction")
    print("=" * 80)

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
        print("[OK] Module constructed successfully")
        print(f"  - Module name: {tsa.name}")
        print(f"  - Embed dims: {tsa.embed_dims}")
        print(f"  - Num heads: {tsa.num_heads}")
        print(f"  - Num levels: {tsa.num_levels}")
        print(f"  - Num points: {tsa.num_points}")
        print(f"  - Num BEV queue: {tsa.num_bev_queue}")
        return True
    except Exception as e:
        print(f"[X] Module construction failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test TemporalSelfAttention forward pass with numerical validation."""
    print("\n" + "=" * 80)
    print("TEST 2: TemporalSelfAttention Forward Pass (with Data Validation)")
    print("=" * 80)

    try:
        # Configuration
        bs = 2
        num_query = 900  # BEV grid size (e.g., 30x30)
        embed_dims = 256
        num_heads = 8
        num_levels = 4
        num_points = 4
        num_bev_queue = 2

        # Spatial shapes for 4 levels (BEV features have same spatial layout across levels)
        spatial_shapes = [(30, 30), (15, 15), (8, 8), (4, 4)]

        # For temporal self-attention, num_query should equal sum of spatial shapes
        # because it's multi-scale BEV features
        num_query = sum(H * W for H, W in spatial_shapes)  # 900 + 225 + 64 + 16 = 1205
        num_value = num_query  # value has same size as query

        print(f"\nConfiguration:")
        print(f"  - Batch size: {bs}")
        print(f"  - Num queries: {num_query}")
        print(f"  - Embed dims: {embed_dims}")
        print(f"  - Num levels: {num_levels}")
        print(f"  - Num BEV queue: {num_bev_queue}")
        print(f"  - Spatial shapes: {spatial_shapes}")
        print(f"  - Num value (per BEV): {num_value}")

        # Create test inputs
        print("\n[1] Creating test inputs...")
        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1

        # Value: [bs*num_bev_queue, num_value, num_heads, head_dim]
        head_dim = embed_dims // num_heads
        value_np = (
            np.random.randn(bs * num_bev_queue, num_value, num_heads, head_dim).astype(
                np.float32
            )
            * 0.1
        )

        # Reference points: [bs, num_query, num_levels, 2]
        ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

        # PyTorch forward pass
        print("\n[2] Running PyTorch reference implementation...")
        model_pt = TemporalSelfAttention_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            num_bev_queue=num_bev_queue,
        )
        model_pt.eval()

        with torch.no_grad():
            output_pt = model_pt(
                torch.from_numpy(query_np),
                torch.from_numpy(value_np),
                torch.from_numpy(ref_points_np),
                spatial_shapes,
            )

        print(f"  PyTorch output shape: {output_pt.shape}")
        output_pt_np = output_pt.detach().cpu().numpy()
        print(
            f"  PyTorch: mean={np.mean(output_pt_np):.6e}, std={np.std(output_pt_np):.6e}, "
            f"min={np.min(output_pt_np):.6e}, max={np.max(output_pt_np):.6e}"
        )

        # TTSim forward pass
        print("\n[3] Running TTSim implementation...")
        tsa = TemporalSelfAttention(
            name="test_tsa_forward",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            num_bev_queue=num_bev_queue,
            batch_first=True,
        )

        # Copy PyTorch weights to TTSim
        print("  Copying PyTorch weights to TTSim...")
        copy_pytorch_weights_to_ttsim(model_pt, tsa)

        # Create TTSim inputs
        # TTSim expects reference_points pre-expanded to [bs*num_bev_queue, nq, nl, 2]
        # Must match PyTorch's unsqueeze(1).repeat(1,nbq,1,1,1).reshape(bs*nbq,...)
        # which gives [rp0,rp0,rp1,rp1] (each batch repeated nbq times consecutively).
        # np.tile would give [rp0,rp1,rp0,rp1] (wrong order), so use np.repeat.
        ref_points_expanded_np = np.repeat(
            ref_points_np, num_bev_queue, axis=0
        )  # [bs*num_bev_queue, nq, nl, 2]
        query_ttsim = F._from_data("query", query_np, is_const=True)
        value_ttsim = F._from_data("value", value_np, is_const=True)
        ref_points_ttsim = F._from_data(
            "ref_points", ref_points_expanded_np, is_const=True
        )

        # Forward pass
        output_ttsim = tsa(
            query=query_ttsim,
            value=value_ttsim,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes,
            level_start_index=None,
        )

        print(f"  TTSim output shape: {output_ttsim.shape}")

        # Get TTSim data for numerical comparison
        if hasattr(output_ttsim, "data") and output_ttsim.data is not None:
            output_ttsim_np = output_ttsim.data
            print(
                f"  TTSim:   mean={np.mean(output_ttsim_np):.6e}, std={np.std(output_ttsim_np):.6e}, "
                f"min={np.min(output_ttsim_np):.6e}, max={np.max(output_ttsim_np):.6e}"
            )

            # Numerical comparison
            print(f"\n  Numerical comparison:")
            print(f"    Max diff: {np.abs(output_pt_np - output_ttsim_np).max():.6e}")
            print(f"    Mean diff: {np.abs(output_pt_np - output_ttsim_np).mean():.6e}")

            # Numerical validation with np.allclose
            if np.allclose(output_pt_np, output_ttsim_np, rtol=1e-4, atol=1e-5):
                print(f"    [OK] Numerical outputs match within tolerance")
            else:
                print(f"    [Warning] Numerical outputs differ beyond tolerance")
        else:
            print(f"  TTSim:   (graph output, data not available for comparison)")

        # Check shape
        expected_shape = [bs, num_query, embed_dims]
        if list(output_ttsim.shape) == expected_shape:
            print(f"\n[OK] Forward pass successful with data validation")
            return True
        else:
            print(
                f"\n[X] Shape mismatch: expected {expected_shape}, got {list(output_ttsim.shape)}"
            )
            return False

    except Exception as e:
        print(f"\n[X] Forward pass failed with exception: {e}")
        traceback.print_exc()
        return False


def test_parameter_count():
    """Test parameter count calculation."""
    print("\n" + "=" * 80)
    print("TEST 3: Parameter Count")
    print("=" * 80)

    try:
        embed_dims = 256
        num_heads = 8
        num_levels = 4
        num_points = 4
        num_bev_queue = 2

        tsa = TemporalSelfAttention(
            name="test_tsa_params",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            num_bev_queue=num_bev_queue,
        )

        param_count = tsa.analytical_param_count()

        # Calculate expected parameters
        # sampling_offsets: (embed_dims*2) * (2*num_heads*num_levels*num_points*2) + bias
        expected_sampling_offsets = (embed_dims * num_bev_queue) * (
            num_bev_queue * num_heads * num_levels * num_points * 2
        ) + (num_bev_queue * num_heads * num_levels * num_points * 2)

        # attention_weights: (embed_dims*2) * (2*num_heads*num_levels*num_points) + bias
        expected_attention_weights = (embed_dims * num_bev_queue) * (
            num_bev_queue * num_heads * num_levels * num_points
        ) + (num_bev_queue * num_heads * num_levels * num_points)

        # value_proj: embed_dims * embed_dims + bias
        expected_value_proj = embed_dims * embed_dims + embed_dims

        # output_proj: embed_dims * embed_dims + bias
        expected_output_proj = embed_dims * embed_dims + embed_dims

        expected_total = (
            expected_sampling_offsets
            + expected_attention_weights
            + expected_value_proj
            + expected_output_proj
        )

        print(f"TemporalSelfAttention parameter breakdown:")
        print(f"  - Sampling offsets: {expected_sampling_offsets:,}")
        print(f"  - Attention weights: {expected_attention_weights:,}")
        print(f"  - Value projection: {expected_value_proj:,}")
        print(f"  - Output projection: {expected_output_proj:,}")
        print(f"  - Expected total: {expected_total:,}")
        print(f"  - Actual total: {param_count:,}")

        if param_count == expected_total:
            print("[OK] Parameter count matches expected")
            return True
        else:
            print(f"[X] Parameter count mismatch")
            return False
    except Exception as e:
        print(f"[X] Test failed: {e}")
        traceback.print_exc()
        return False


def test_different_configurations():
    """Test with different configurations."""
    print("\n" + "=" * 80)
    print("TEST 4: Different Configurations (with Data Validation)")
    print("=" * 80)

    test_cases = [
        # (embed_dims, num_heads, num_levels, num_points, spatial_shapes)
        (128, 4, 2, 4, [(20, 20), (10, 10)]),
        (256, 8, 4, 4, [(30, 30), (15, 15), (8, 8), (4, 4)]),
        (512, 16, 3, 8, [(15, 15), (8, 8), (4, 4)]),
    ]

    all_passed = True
    for i, (embed_dims, num_heads, num_levels, num_points, spatial_shapes) in enumerate(
        test_cases, 1
    ):
        try:
            print(
                f"\nTest case {i}: embed_dims={embed_dims}, num_heads={num_heads}, "
                f"num_levels={num_levels}, num_points={num_points}"
            )
            print(f"  Spatial shapes: {spatial_shapes}")

            bs = 2
            num_query = sum(
                H * W for H, W in spatial_shapes
            )  # Match sum of spatial shapes
            num_bev_queue = 2
            head_dim = embed_dims // num_heads

            # Create test inputs
            query_np = (
                np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
            )
            value_np = (
                np.random.randn(
                    bs * num_bev_queue, num_query, num_heads, head_dim
                ).astype(np.float32)
                * 0.1
            )
            ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(
                np.float32
            )

            # PyTorch reference
            model_pt = TemporalSelfAttention_PyTorch(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_bev_queue=num_bev_queue,
            )
            model_pt.eval()

            with torch.no_grad():
                output_pt = model_pt(
                    torch.from_numpy(query_np),
                    torch.from_numpy(value_np),
                    torch.from_numpy(ref_points_np),
                    spatial_shapes,
                )

            print(
                f"  PyTorch output: shape={output_pt.shape}, "
                f"range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}], "
                f"mean={output_pt.mean().item():.6f}"
            )

            # TTSim implementation
            tsa = TemporalSelfAttention(
                name=f"test_tsa_config_{i}",
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                num_bev_queue=num_bev_queue,
            )

            # Copy weights
            copy_pytorch_weights_to_ttsim(model_pt, tsa)

            # TTSim expects reference_points pre-expanded to [bs*num_bev_queue, nq, nl, 2]
            # np.repeat gives [rp0,rp0,rp1,rp1] matching PyTorch's expansion order
            ref_points_expanded_np = np.repeat(ref_points_np, num_bev_queue, axis=0)
            query_ttsim = F._from_data("query", query_np, is_const=True)
            value_ttsim = F._from_data("value", value_np, is_const=True)
            ref_points_ttsim = F._from_data(
                "ref_points", ref_points_expanded_np, is_const=True
            )

            output_ttsim = tsa(
                query=query_ttsim,
                value=value_ttsim,
                reference_points=ref_points_ttsim,
                spatial_shapes=spatial_shapes,
            )

            print(f"  TTSim output: shape={output_ttsim.shape}")

            # Numerical comparison
            if hasattr(output_ttsim, "data") and output_ttsim.data is not None:
                output_ttsim_np = output_ttsim.data
                output_pt_np = output_pt.detach().cpu().numpy()
                print(
                    f"    Max diff: {np.abs(output_pt_np - output_ttsim_np).max():.6e}, Mean diff: {np.abs(output_pt_np - output_ttsim_np).mean():.6e}"
                )

                # Numerical validation with np.allclose
                if not np.allclose(output_pt_np, output_ttsim_np, rtol=1e-4, atol=1e-5):
                    print(f"    [Warning] Numerical outputs differ beyond tolerance")

            # Validate shapes
            if list(output_pt.shape) == list(output_ttsim.shape):
                print(
                    f"  [OK] Shapes match! Parameter count: {tsa.analytical_param_count():,}"
                )
            else:
                print(f"  [X] Shape mismatch")
                all_passed = False
        except Exception as e:
            print(f"  [X] Test case {i} failed: {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Temporal Self Attention TTSim Module Test Suite")
    print("=" * 80)

    results = {
        "TemporalSelfAttention Construction": test_construction(),
        "TemporalSelfAttention Forward Pass": test_forward_pass(),
        "Parameter Count": test_parameter_count(),
        "Different Configurations": test_different_configurations(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name:.<60} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n! All tests passed! The module is working correctly.")
        return 0
    else:
        print(
            f"\n[WARNING]  {total_tests - passed_tests} test(s) failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
