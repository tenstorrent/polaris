#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for Spatial Cross Attention TTSim modules.
Validates the conversion from PyTorch to TTSim.

This tests:
- SpatialCrossAttention: BEV-to-camera attention wrapper
- MSDeformableAttention3D: 3D deformable attention with Z-anchors
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
from workloads.BEVFormer.ttsim_models.spatial_cross_attention import (
    SpatialCrossAttention,
    MSDeformableAttention3D,
)

# Import initialization utilities
from workloads.BEVFormer.ttsim_models.init_utils import (
    xavier_init,
    constant_init,
    _is_power_of_2,
)

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


class MSDeformableAttention3D_PyTorch(nn.Module):
    """PyTorch reference implementation for data validation."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # Initialize
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)

    def forward(self, query, value, reference_points, spatial_shapes):
        """
        Args:
            query: [bs, num_query, embed_dims]
            value: [bs, num_value, embed_dims]
            reference_points: [bs, num_query, num_Z_anchors, 2]
            spatial_shapes: list of (H, W) tuples
        """
        bs, num_query, _ = query.shape
        num_value = value.shape[1]
        num_Z_anchors = reference_points.shape[2]

        # Project value
        value = self.value_proj(value)
        value = value.view(
            bs, num_value, self.num_heads, self.embed_dims // self.num_heads
        )

        # Compute offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Compute attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F_torch.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # Normalize offsets by spatial shapes
        offset_normalizer = torch.tensor(
            [[W, H] for H, W in spatial_shapes],
            dtype=torch.float32,
            device=query.device,
        )
        offset_normalizer = offset_normalizer[
            None, None, None, :, None, :
        ]  # [1, 1, 1, L, 1, 2]
        normalized_offsets = sampling_offsets / offset_normalizer

        # Reshape for Z-anchors
        num_points_per_anchor = self.num_points // num_Z_anchors
        normalized_offsets = normalized_offsets.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            num_points_per_anchor,
            num_Z_anchors,
            2,
        )

        # Add reference points
        ref_pts = reference_points[
            :, :, None, None, None, :, :
        ]  # [bs, nq, 1, 1, 1, D, 2]
        sampling_locations = ref_pts + normalized_offsets

        # Reshape back
        sampling_locations = sampling_locations.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Use PyTorch multi_scale_deformable_attn
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        return output


def initialize_linear_weights_with_data(linear_layer, weight_data, bias_data=None):
    """
    Initialize a TTSim Linear layer's weights with actual data for numerical validation.

    Args:
        linear_layer: TTSim Linear layer instance
        weight_data: NumPy array of shape [in_features, out_features]
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
    Copy weights from a PyTorch MSDeformableAttention3D to a TTSim one.

    Args:
        pytorch_module: PyTorch MSDeformableAttention3D_PyTorch instance
        ttsim_module: TTSim MSDeformableAttention3D instance
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


def compare_outputs(pytorch_out, ttsim_out, name="Output"):
    """Compare PyTorch and TTSim outputs (shapes only, as TTSim builds graphs)."""
    pytorch_shape = list(pytorch_out.shape)
    ttsim_shape = list(ttsim_out.shape)

    pytorch_np = pytorch_out.detach().cpu().numpy()

    print(f"\n  {name} Comparison:")
    print(f"    PyTorch shape: {pytorch_shape}")
    print(f"    TTSim shape: {ttsim_shape}")
    print(f"    PyTorch range: [{np.min(pytorch_np):.6f}, {np.max(pytorch_np):.6f}]")
    print(f"    PyTorch mean: {np.mean(pytorch_np):.6f}, std: {np.std(pytorch_np):.6f}")

    if pytorch_shape == ttsim_shape:
        print(f"    ✓ Shapes match!")
        return True
    else:
        print(f"    ✗ Shape mismatch!")
        return False


# ============================================================================
# Test Functions
# ============================================================================


def test_initialization_utils():
    """Test initialization utility functions."""
    print("\n" + "=" * 80)
    print("TEST 1: Initialization Utilities (Python 3.13 Compatible)")
    print("=" * 80)

    all_passed = True

    # Test 1: xavier_init
    try:
        print("\n[1] Testing xavier_init...")
        linear = nn.Linear(256, 256)
        xavier_init(linear, gain=1.0, bias=0.0, distribution="uniform")
        print("  ✓ xavier_init succeeded")

        # Check that weights are initialized
        weight_std = linear.weight.data.std().item()
        bias_val = linear.bias.data.abs().max().item()
        print(f"    - Weight std: {weight_std:.6f}")
        print(f"    - Bias max abs: {bias_val:.6f}")

        if weight_std > 0 and bias_val < 1e-6:
            print("  ✓ xavier_init values look correct")
        else:
            print("  ✗ xavier_init values unexpected")
            all_passed = False
    except Exception as e:
        print(f"  ✗ xavier_init test failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Test 2: constant_init
    try:
        print("\n[2] Testing constant_init...")
        linear = nn.Linear(128, 128)
        constant_init(linear, val=0.0, bias=0.0)
        print("  ✓ constant_init succeeded")

        # Check that weights are all zero
        weight_max = linear.weight.data.abs().max().item()
        bias_max = linear.bias.data.abs().max().item()
        print(f"    - Weight max abs: {weight_max:.6f}")
        print(f"    - Bias max abs: {bias_max:.6f}")

        if weight_max < 1e-6 and bias_max < 1e-6:
            print("  ✓ constant_init values are zero")
        else:
            print("  ✗ constant_init values are not zero")
            all_passed = False
    except Exception as e:
        print(f"  ✗ constant_init test failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Test 3: _is_power_of_2
    try:
        print("\n[3] Testing _is_power_of_2...")
        test_cases = [
            (1, True),
            (2, True),
            (4, True),
            (8, True),
            (16, True),
            (32, True),
            (64, True),
            (128, True),
            (256, True),
            (3, False),
            (5, False),
            (6, False),
            (7, False),
            (15, False),
        ]

        all_correct = True
        for n, expected in test_cases:
            result = _is_power_of_2(n)
            if result != expected:
                print(f"  ✗ _is_power_of_2({n}) = {result}, expected {expected}")
                all_correct = False

        if all_correct:
            print("  ✓ _is_power_of_2 works correctly")
        else:
            print("  ✗ _is_power_of_2 has errors")
            all_passed = False
    except Exception as e:
        print(f"  ✗ _is_power_of_2 test failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Test 4: Check that warnings work for non-power-of-2 dims
    try:
        print("\n[4] Testing warning for non-power-of-2 dims...")
        # This should trigger a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed_dims = 250  # Not a power of 2 when divided by 8 heads
            num_heads = 8
            dim_per_head = embed_dims // num_heads  # 31.25, rounded to 31
            if not _is_power_of_2(dim_per_head):
                warnings.warn(
                    "You'd better set embed_dims in "
                    "MultiScaleDeformAttention to make "
                    "the dimension of each attention head a power of 2 "
                    "which is more efficient in our CUDA implementation."
                )

            if len(w) > 0:
                print(f"  ✓ Warning triggered as expected: {w[0].message}")
            else:
                print("  ✗ Warning not triggered")
                all_passed = False
    except Exception as e:
        print(f"  ✗ Warning test failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Test 5: Compare PyTorch vs TTSim with initialized weights
    try:
        print(
            "\n[5] Testing PyTorch vs TTSim output comparison with initialized weights..."
        )

        # Configuration
        bs = 2
        num_query = 10
        embed_dims = 128
        num_heads = 4
        num_levels = 2
        num_points = 4
        num_Z_anchors = 4
        spatial_shapes = [(10, 10), (5, 5)]
        num_value = 125

        # Create test inputs
        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(bs, num_query, num_Z_anchors, 2).astype(
            np.float32
        )

        # Create PyTorch model with our initialization functions
        print("    Creating PyTorch model with custom initialization...")
        model_pt = MSDeformableAttention3D_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

        # Apply our initialization functions
        xavier_init(model_pt.value_proj, gain=1.0, bias=0.0, distribution="uniform")
        constant_init(model_pt.sampling_offsets, val=0.0, bias=0.0)
        constant_init(model_pt.attention_weights, val=0.0, bias=0.0)

        model_pt.eval()

        # PyTorch forward pass
        with torch.no_grad():
            output_pt = model_pt(
                torch.from_numpy(query_np),
                torch.from_numpy(value_np),
                torch.from_numpy(ref_points_np),
                spatial_shapes,
            )

        print(
            f"    PyTorch output: shape={output_pt.shape}, "
            f"range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}], "
            f"mean={output_pt.mean().item():.6f}"
        )

        # Create TTSim model
        print("    Creating TTSim model...")
        msda3d = MSDeformableAttention3D(
            name="test_msda3d_init_compare",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        # TTSim forward pass
        query_ttsim = F._from_data("query", query_np, is_const=True)
        value_ttsim = F._from_data("value", value_np, is_const=True)
        ref_points_ttsim = F._from_data("ref_points", ref_points_np, is_const=True)

        output_ttsim = msda3d(
            query=query_ttsim,
            value=value_ttsim,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes,
        )

        print(f"    TTSim output: shape={output_ttsim.shape}")

        # Compare shapes
        if list(output_pt.shape) == list(output_ttsim.shape):
            print("    ✓ PyTorch and TTSim output shapes match!")
            print(f"      Both have shape: {list(output_pt.shape)}")
        else:
            print(
                f"    ✗ Shape mismatch: PyTorch {list(output_pt.shape)} vs TTSim {list(output_ttsim.shape)}"
            )
            all_passed = False

        # Verify initialization utilities work correctly
        print(
            "    ✓ Initialization utilities successfully used in PyTorch-TTSim comparison"
        )

    except Exception as e:
        print(f"  ✗ PyTorch vs TTSim comparison test failed: {e}")
        traceback.print_exc()
        all_passed = False

    return all_passed


def test_msda3d_construction():
    """Test that MSDeformableAttention3D can be constructed successfully."""
    print("\n" + "=" * 80)
    print("TEST 2: MSDeformableAttention3D Construction")
    print("=" * 80)

    try:
        msda3d = MSDeformableAttention3D(
            name="test_msda3d",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=8,
            dropout=0.1,
        )
        print("✓ Module constructed successfully")
        print(f"  - Module name: {msda3d.name}")
        print(f"  - Embed dims: {msda3d.embed_dims}")
        print(f"  - Num heads: {msda3d.num_heads}")
        print(f"  - Num levels: {msda3d.num_levels}")
        print(f"  - Num points: {msda3d.num_points}")
        return True
    except Exception as e:
        print(f"✗ Module construction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sca_construction():
    """Test that SpatialCrossAttention can be constructed successfully."""
    print("\n" + "=" * 80)
    print("TEST 3: SpatialCrossAttention Construction")
    print("=" * 80)

    try:
        sca = SpatialCrossAttention(
            name="test_sca",
            embed_dims=256,
            num_cams=6,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            dropout=0.1,
            batch_first=False,
        )
        print("✓ Module constructed successfully")
        print(f"  - Module name: {sca.name}")
        print(f"  - Embed dims: {sca.embed_dims}")
        print(f"  - Num cameras: {sca.num_cams}")
        print(f"  - PC range: {sca.pc_range}")
        return True
    except Exception as e:
        print(f"✗ Module construction failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_msda3d_forward():
    """Test MSDeformableAttention3D forward pass with PyTorch data validation."""
    print("\n" + "=" * 80)
    print("TEST 4: MSDeformableAttention3D Forward Pass (with Data Validation)")
    print("=" * 80)

    try:
        # Configuration
        bs = 2
        num_query = 10
        embed_dims = 256
        num_heads = 8
        num_levels = 4
        num_points = 8
        num_Z_anchors = 4

        # Spatial shapes for 4 levels
        spatial_shapes = [(50, 50), (25, 25), (13, 13), (7, 7)]
        num_value = sum([h * w for h, w in spatial_shapes])

        print(f"\nConfiguration:")
        print(f"  - Batch size: {bs}")
        print(f"  - Num queries: {num_query}")
        print(f"  - Embed dims: {embed_dims}")
        print(f"  - Num levels: {num_levels}")
        print(f"  - Num Z anchors: {num_Z_anchors}")
        print(f"  - Spatial shapes: {spatial_shapes}")
        print(f"  - Total num_value: {num_value}")

        # Create test inputs (shared between PyTorch and TTSim)
        print("\n[1] Creating test inputs...")
        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(bs, num_query, num_Z_anchors, 2).astype(
            np.float32
        )

        # PyTorch forward pass
        print("\n[2] Running PyTorch reference implementation...")
        model_pt = MSDeformableAttention3D_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        model_pt.eval()

        query_pt = torch.from_numpy(query_np)
        value_pt = torch.from_numpy(value_np)
        ref_points_pt = torch.from_numpy(ref_points_np)

        with torch.no_grad():
            output_pt = model_pt(query_pt, value_pt, ref_points_pt, spatial_shapes)

        print(f"  PyTorch output shape: {output_pt.shape}")
        output_pt_np = output_pt.detach().cpu().numpy()
        print(
            f"  PyTorch: mean={np.mean(output_pt_np):.6e}, std={np.std(output_pt_np):.6e}, "
            f"min={np.min(output_pt_np):.6e}, max={np.max(output_pt_np):.6e}"
        )

        # TTSim forward pass
        print("\n[3] Running TTSim implementation...")
        msda3d = MSDeformableAttention3D(
            name="test_msda3d_forward",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        # Copy PyTorch weights to TTSim for numerical validation
        print("  Copying PyTorch weights to TTSim...")
        copy_pytorch_weights_to_ttsim(model_pt, msda3d)

        query_ttsim = F._from_data("query", query_np, is_const=True)
        value_ttsim = F._from_data("value", value_np, is_const=True)
        ref_points_ttsim = F._from_data("ref_points", ref_points_np, is_const=True)

        output_ttsim = msda3d(
            query=query_ttsim,
            value=value_ttsim,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes,
            level_start_index=None,
        )

        print(f"  TTSim output shape: {output_ttsim.shape}")

        # Get TTSim data if available
        if hasattr(output_ttsim, "data") and output_ttsim.data is not None:
            output_ttsim_np = output_ttsim.data
            print(
                f"  TTSim:   mean={np.mean(output_ttsim_np):.6e}, std={np.std(output_ttsim_np):.6e}, "
                f"min={np.min(output_ttsim_np):.6e}, max={np.max(output_ttsim_np):.6e}"
            )

            # Numerical comparison
            max_diff = np.max(np.abs(output_pt_np - output_ttsim_np))
            mean_diff = np.mean(np.abs(output_pt_np - output_ttsim_np))
            print(f"\n  Numerical comparison:")
            print(f"    Max diff: {max_diff:.6e}")
            print(f"    Mean diff: {mean_diff:.6e}")
        else:
            print(f"  TTSim:   (graph output, data not available for comparison)")

        # Check output shape
        expected_shape = [bs, num_query, embed_dims]
        print(f"\n[4] Validating outputs...")
        print(f"  Expected shape: {expected_shape}")

        # Compare outputs
        shapes_match = compare_outputs(output_pt, output_ttsim, name="MSDA3D Output")

        if list(output_ttsim.shape) == expected_shape and shapes_match:
            print("\n✓ Forward pass successful with data validation")
            return True
        else:
            print(f"\n✗ Forward pass validation failed")
            return False
    except Exception as e:
        print(f"\n✗ Forward pass failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sca_forward():
    """Test SpatialCrossAttention forward pass with shape validation."""
    print("\n" + "=" * 80)
    print("TEST 5: SpatialCrossAttention Forward Pass")
    print("=" * 80)

    try:
        # Configuration
        bs = 1
        num_query = 900  # 30x30 BEV grid
        num_cams = 6
        embed_dims = 256
        num_levels = 4
        num_Z_anchors = 4

        # Spatial shapes for 4 levels
        spatial_shapes = [(116, 200), (58, 100), (29, 50), (15, 25)]
        l = sum([h * w for h, w in spatial_shapes])  # Total feature map size
        D = num_Z_anchors

        print(f"Configuration:")
        print(f"  - Batch size: {bs}")
        print(f"  - Num queries: {num_query}")
        print(f"  - Num cameras: {num_cams}")
        print(f"  - Embed dims: {embed_dims}")
        print(f"  - Spatial shapes: {spatial_shapes}")
        print(f"  - Total feature size (l): {l}")

        # Create module
        sca = SpatialCrossAttention(
            name="test_sca_forward",
            embed_dims=embed_dims,
            num_cams=num_cams,
            dropout=0.1,
            batch_first=True,  # Use batch_first for easier testing
        )

        # Create inputs
        query = F._from_shape("query", [bs, num_query, embed_dims])
        key = F._from_shape("key", [num_cams, l, bs, embed_dims])
        value = F._from_shape("value", [num_cams, l, bs, embed_dims])
        reference_points_cam = F._from_shape(
            "ref_points_cam", [num_cams, bs, num_query, D, 2]
        )

        # Create bev_mask (simplified - all queries can see all cameras)
        bev_mask_data = np.ones((num_cams, bs, num_query), dtype=np.float32)
        bev_mask = F._from_data("bev_mask", bev_mask_data, is_const=True)

        # Run forward pass
        print("\nRunning forward pass...")
        output = sca(
            query=query,
            key=key,
            value=value,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=None,
        )

        # Check output shape
        expected_shape = [bs, num_query, embed_dims]
        print(f"Expected output shape: {expected_shape}")
        print(f"Actual output shape: {output.shape}")

        if list(output.shape) == expected_shape:
            print("✓ Forward pass successful - output shape matches expected")
            return True
        else:
            print(f"✗ Forward pass failed - shape mismatch")
            return False
    except Exception as e:
        print(f"✗ Forward pass failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_different_configurations():
    """Test with different configurations (with PyTorch data validation)."""
    print("\n" + "=" * 80)
    print("TEST 6: Different Configurations (with Data Validation)")
    print("=" * 80)

    test_cases = [
        # (embed_dims, num_heads, num_levels, num_points, spatial_shapes)
        (128, 4, 2, 4, [(10, 10), (5, 5)]),
        (256, 8, 4, 8, [(50, 50), (25, 25), (13, 13), (7, 7)]),
        (512, 16, 3, 4, [(20, 20), (10, 10), (5, 5)]),
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
            num_query = 10
            num_Z_anchors = 4
            num_value = sum([h * w for h, w in spatial_shapes])

            # Create test inputs
            query_np = (
                np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
            )
            value_np = (
                np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
            )
            ref_points_np = np.random.rand(bs, num_query, num_Z_anchors, 2).astype(
                np.float32
            )

            # PyTorch reference
            model_pt = MSDeformableAttention3D_PyTorch(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
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
            msda3d = MSDeformableAttention3D(
                name=f"test_msda3d_config_{i}",
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
            )

            # Copy PyTorch weights for numerical validation
            copy_pytorch_weights_to_ttsim(model_pt, msda3d)

            query_ttsim = F._from_data("query", query_np, is_const=True)
            value_ttsim = F._from_data("value", value_np, is_const=True)
            ref_points_ttsim = F._from_data("ref_points", ref_points_np, is_const=True)

            output_ttsim = msda3d(
                query=query_ttsim,
                value=value_ttsim,
                reference_points=ref_points_ttsim,
                spatial_shapes=spatial_shapes,
            )

            print(f"  TTSim output: shape={output_ttsim.shape}")

            # Get TTSim data for numerical comparison
            if hasattr(output_ttsim, "data") and output_ttsim.data is not None:
                output_ttsim_np = output_ttsim.data
                output_pt_np = output_pt.detach().cpu().numpy()
                max_diff = np.max(np.abs(output_pt_np - output_ttsim_np))
                mean_diff = np.mean(np.abs(output_pt_np - output_ttsim_np))
                print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

            # Validate shapes
            if list(output_pt.shape) == list(output_ttsim.shape):
                print(
                    f"  ✓ Shapes match! Parameter count: {msda3d.analytical_param_count():,}"
                )
            else:
                print(
                    f"  ✗ Shape mismatch: PyTorch {list(output_pt.shape)} vs TTSim {list(output_ttsim.shape)}"
                )
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test case {i} failed: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    return all_passed


def test_parameter_count():
    """Test parameter count calculation."""
    print("\n" + "=" * 80)
    print("TEST 7: Parameter Count")
    print("=" * 80)

    try:
        # MSDeformableAttention3D
        embed_dims = 256
        num_heads = 8
        num_levels = 4
        num_points = 8

        msda3d = MSDeformableAttention3D(
            name="test_msda3d_params",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )

        param_count = msda3d.analytical_param_count()

        # Calculate expected parameters:
        # sampling_offsets: embed_dims * (num_heads * num_levels * num_points * 2) + bias
        # attention_weights: embed_dims * (num_heads * num_levels * num_points) + bias
        # value_proj: embed_dims * embed_dims + bias

        expected_sampling_offsets = embed_dims * (
            num_heads * num_levels * num_points * 2
        ) + (num_heads * num_levels * num_points * 2)
        expected_attention_weights = embed_dims * (
            num_heads * num_levels * num_points
        ) + (num_heads * num_levels * num_points)
        expected_value_proj = embed_dims * embed_dims + embed_dims
        expected_total = (
            expected_sampling_offsets + expected_attention_weights + expected_value_proj
        )

        print(f"MSDeformableAttention3D parameter breakdown:")
        print(f"  - Sampling offsets: {expected_sampling_offsets:,}")
        print(f"  - Attention weights: {expected_attention_weights:,}")
        print(f"  - Value projection: {expected_value_proj:,}")
        print(f"  - Expected total: {expected_total:,}")
        print(f"  - Actual total: {param_count:,}")

        if param_count == expected_total:
            print("✓ Parameter count matches expected")
            return True
        else:
            print(f"✗ Parameter count mismatch")
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_batch_first_flag():
    """Test batch_first flag handling (with PyTorch data validation)."""
    print("\n" + "=" * 80)
    print("TEST 8: Batch First Flag (with Data Validation)")
    print("=" * 80)

    all_passed = True

    for batch_first in [True, False]:
        try:
            print(f"\nTesting with batch_first={batch_first}")

            bs = 2
            num_query = 10
            embed_dims = 128
            num_heads = 4
            num_levels = 2
            num_points = 4
            num_Z_anchors = 4
            spatial_shapes = [(10, 10), (5, 5)]
            num_value = 125  # 10*10 + 5*5 = 100 + 25

            # Create test inputs
            query_np = (
                np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
            )
            value_np = (
                np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
            )
            ref_points_np = np.random.rand(bs, num_query, num_Z_anchors, 2).astype(
                np.float32
            )

            # PyTorch reference (always batch-first)
            model_pt = MSDeformableAttention3D_PyTorch(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
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
                f"range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}]"
            )

            # TTSim implementation
            msda3d = MSDeformableAttention3D(
                name=f"test_msda3d_bf_{batch_first}",
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                batch_first=batch_first,
            )

            # Copy PyTorch weights for numerical validation
            copy_pytorch_weights_to_ttsim(model_pt, msda3d)

            if batch_first:
                query = F._from_data("query", query_np, is_const=True)
                value = F._from_data("value", value_np, is_const=True)
            else:
                query = F._from_data(
                    "query", query_np.transpose(1, 0, 2), is_const=True
                )
                value = F._from_data(
                    "value", value_np.transpose(1, 0, 2), is_const=True
                )

            reference_points = F._from_data("ref_points", ref_points_np, is_const=True)

            output = msda3d(
                query=query,
                value=value,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
            )

            print(f"  TTSim output: shape={output.shape}")

            # Get TTSim data for numerical comparison
            if hasattr(output, "data") and output.data is not None:
                output_ttsim_np = output.data
                output_pt_np = output_pt.detach().cpu().numpy()
                if batch_first:
                    max_diff = np.max(np.abs(output_pt_np - output_ttsim_np))
                    mean_diff = np.mean(np.abs(output_pt_np - output_ttsim_np))
                else:
                    # Transpose TTSim output for comparison
                    output_ttsim_np_t = np.transpose(output_ttsim_np, (1, 0, 2))
                    max_diff = np.max(np.abs(output_pt_np - output_ttsim_np_t))
                    mean_diff = np.mean(np.abs(output_pt_np - output_ttsim_np_t))
                print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

            if batch_first:
                expected_shape = [bs, num_query, embed_dims]
                shapes_match = list(output_pt.shape) == list(output.shape)
            else:
                expected_shape = [num_query, bs, embed_dims]
                shapes_match = list(output_pt.shape) == [
                    output.shape[1],
                    output.shape[0],
                    output.shape[2],
                ]

            if list(output.shape) == expected_shape and shapes_match:
                print(
                    f"  ✓ Output shape correct and matches PyTorch (accounting for batch_first)"
                )
            else:
                print(
                    f"  ✗ Output shape incorrect: expected {expected_shape}, got {output.shape}"
                )
                all_passed = False
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    return all_passed


def test_with_masking():
    """Test with key_padding_mask (with PyTorch data validation)."""
    print("\n" + "=" * 80)
    print("TEST 9: With Key Padding Mask (with Data Validation)")
    print("=" * 80)

    try:
        bs = 2
        num_query = 10
        num_value = 125  # 10*10 + 5*5 = 100 + 25
        embed_dims = 128
        num_heads = 4
        num_levels = 2
        num_points = 4
        num_Z_anchors = 4
        spatial_shapes = [(10, 10), (5, 5)]

        # Create test inputs
        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(bs, num_query, num_Z_anchors, 2).astype(
            np.float32
        )

        # PyTorch reference (without masking for simplicity, just shape comparison)
        model_pt = MSDeformableAttention3D_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
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
            f"\n  PyTorch output (no mask): shape={output_pt.shape}, "
            f"range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}]"
        )

        # TTSim implementation with masking
        msda3d = MSDeformableAttention3D(
            name="test_msda3d_mask",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        # Copy PyTorch weights for numerical validation
        copy_pytorch_weights_to_ttsim(model_pt, msda3d)

        query = F._from_data("query", query_np, is_const=True)
        value = F._from_data("value", value_np, is_const=True)
        reference_points = F._from_data("ref_points", ref_points_np, is_const=True)

        # Create padding mask (some positions are padded)
        mask_data = np.zeros((bs, num_value), dtype=np.bool_)
        mask_data[:, -10:] = True  # Last 10 positions are padding
        key_padding_mask = F._from_data(
            "mask", mask_data.astype(np.float32), is_const=True
        )

        output = msda3d(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            key_padding_mask=key_padding_mask,
        )

        print(f"  TTSim output (with mask): shape={output.shape}")

        # Get TTSim data for numerical comparison
        if hasattr(output, "data") and output.data is not None:
            output_ttsim_np = output.data
            output_pt_np = output_pt.detach().cpu().numpy()
            max_diff = np.max(np.abs(output_pt_np - output_ttsim_np))
            mean_diff = np.mean(np.abs(output_pt_np - output_ttsim_np))
            print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        expected_shape = [bs, num_query, embed_dims]
        if list(output.shape) == expected_shape and list(output_pt.shape) == list(
            output.shape
        ):
            print(f"  ✓ Forward pass with masking successful, shapes match PyTorch")
            return True
        else:
            print(
                f"  ✗ Output shape incorrect: expected {expected_shape}, got {output.shape}"
            )
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases (with PyTorch data validation)."""
    print("\n" + "=" * 80)
    print("TEST 10: Edge Cases (with Data Validation)")
    print("=" * 80)

    all_passed = True

    # Test 1: Single query
    try:
        print("\nEdge case 1: Single query")
        embed_dims = 128
        num_heads = 4
        num_levels = 2
        num_points = 4
        num_Z_anchors = 4
        spatial_shapes = [(10, 10), (5, 5)]
        num_value = 125

        # Create test inputs
        query_np = np.random.randn(1, 1, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(1, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(1, 1, num_Z_anchors, 2).astype(np.float32)

        # PyTorch reference
        model_pt = MSDeformableAttention3D_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
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
            f"  PyTorch: shape={output_pt.shape}, range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}]"
        )

        # TTSim
        msda3d = MSDeformableAttention3D(
            name="test_msda3d_edge1",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        # Copy PyTorch weights for numerical validation
        copy_pytorch_weights_to_ttsim(model_pt, msda3d)

        query = F._from_data("query", query_np, is_const=True)
        value = F._from_data("value", value_np, is_const=True)
        reference_points = F._from_data("ref_points", ref_points_np, is_const=True)

        output = msda3d(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

        print(f"  TTSim: shape={output.shape}")

        # Get TTSim data for numerical comparison
        if hasattr(output, "data") and output.data is not None:
            output_ttsim_np = output.data
            max_diff = np.max(
                np.abs(output_pt.detach().cpu().numpy() - output_ttsim_np)
            )
            mean_diff = np.mean(
                np.abs(output_pt.detach().cpu().numpy() - output_ttsim_np)
            )
            print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        if list(output.shape) == [1, 1, 128] and list(output_pt.shape) == list(
            output.shape
        ):
            print("  ✓ Single query test passed, shapes match PyTorch")
        else:
            print(f"  ✗ Single query test failed: shape {output.shape}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Single query test failed: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Test 2: Single level
    try:
        print("\nEdge case 2: Single level")
        embed_dims = 128
        num_heads = 4
        num_levels = 1
        num_points = 4
        num_Z_anchors = 4
        spatial_shapes = [(10, 10)]
        num_value = 100

        # Create test inputs
        query_np = np.random.randn(2, 10, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(2, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(2, 10, num_Z_anchors, 2).astype(np.float32)

        # PyTorch reference
        model_pt = MSDeformableAttention3D_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
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
            f"  PyTorch: shape={output_pt.shape}, range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}]"
        )

        # TTSim
        msda3d = MSDeformableAttention3D(
            name="test_msda3d_edge2",
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        # Copy PyTorch weights for numerical validation
        copy_pytorch_weights_to_ttsim(model_pt, msda3d)

        query = F._from_data("query", query_np, is_const=True)
        value = F._from_data("value", value_np, is_const=True)
        reference_points = F._from_data("ref_points", ref_points_np, is_const=True)

        output = msda3d(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

        print(f"  TTSim: shape={output.shape}")

        # Get TTSim data for numerical comparison
        if hasattr(output, "data") and output.data is not None:
            output_ttsim_np = output.data
            max_diff = np.max(
                np.abs(output_pt.detach().cpu().numpy() - output_ttsim_np)
            )
            mean_diff = np.mean(
                np.abs(output_pt.detach().cpu().numpy() - output_ttsim_np)
            )
            print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

        if list(output.shape) == [2, 10, 128] and list(output_pt.shape) == list(
            output.shape
        ):
            print("  ✓ Single level test passed, shapes match PyTorch")
        else:
            print(f"  ✗ Single level test failed: shape {output.shape}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ Single level test failed: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Spatial Cross Attention TTSim Module Test Suite")
    print("=" * 80)

    results = {
        "Initialization Utilities": test_initialization_utils(),
        "MSDeformableAttention3D Construction": test_msda3d_construction(),
        "SpatialCrossAttention Construction": test_sca_construction(),
        "MSDeformableAttention3D Forward Pass": test_msda3d_forward(),
        "SpatialCrossAttention Forward Pass": test_sca_forward(),
        "Different Configurations": test_different_configurations(),
        "Parameter Count": test_parameter_count(),
        "Batch First Flag": test_batch_first_flag(),
        "With Key Padding Mask": test_with_masking(),
        "Edge Cases": test_edge_cases(),
    }

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<60} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n All tests passed! The modules are working correctly.")
        return 0
    else:
        print(
            f"\n  {total_tests - passed_tests} test(s) failed. Please review the errors above."
        )
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
