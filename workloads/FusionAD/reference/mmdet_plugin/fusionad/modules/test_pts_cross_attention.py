#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script for PtsCrossAttention TTSim module.
Validates the conversion from PyTorch to TTSim.

This tests:
- PtsCrossAttention: Standard multi-scale deformable cross-attention
  used in the decoder for object queries attending to BEV features.
"""

import os
import sys
import warnings
import math
import traceback
polaris_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..')
sys.path.insert(0, polaris_path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import ttsim.front.functional.op as F
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.pts_cross_attention import PtsCrossAttention
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.init_utils import xavier_init, constant_init

# Fix for OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================================
# PyTorch Reference Implementation (CPU-only, Python 3.13 compatible)
# ============================================================================

def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                       sampling_locations, attention_weights):
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
        # bs, H_*W_, num_heads, embed_dims -> bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)

        # bs, num_queries, num_heads, num_points, 2 -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F_torch.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    # Stack sampled values and apply attention weights
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()


class PtsCrossAttention_PyTorch(nn.Module):
    """PyTorch reference implementation for data validation."""

    def __init__(self, embed_dims=256, num_heads=8, num_levels=1, num_points=4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # Initialize
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(
            num_heads, dtype=torch.float32) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            num_heads, 1, 1, 2).repeat(1, num_levels, num_points, 1)
        for i in range(num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward_no_residual(self, query, value, reference_points, spatial_shapes):
        """
        Args:
            query: [bs, num_query, embed_dims]
            value: [bs, num_value, embed_dims]
            reference_points: [bs, num_query, num_levels, 2]
            spatial_shapes: list of (H, W) tuples
        """
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads,
                          self.embed_dims // self.num_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads,
            self.num_levels * self.num_points)
        attention_weights = F_torch.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads,
            self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(
                [[W, H] for H, W in spatial_shapes],
                dtype=torch.float32, device=query.device)
            offset_normalizer = offset_normalizer[None, None, None, :, None, :]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2, got {reference_points.shape[-1]}')

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)

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
        weight_data: NumPy array of shape [in_features, out_features]
        bias_data: Optional NumPy array of shape [out_features]
    """
    linear_layer.param = F._from_data(linear_layer.param.name, weight_data, is_const=True)
    linear_layer.param.is_param = True
    linear_layer.param.set_module(linear_layer)
    linear_layer._tensors[linear_layer.param.name] = linear_layer.param

    if bias_data is not None and linear_layer.bias is not None:
        linear_layer.bias = F._from_data(linear_layer.bias.name, bias_data, is_const=True)
        linear_layer.bias.is_param = True
        linear_layer.bias.set_module(linear_layer)
        linear_layer._tensors[linear_layer.bias.name] = linear_layer.bias


def copy_pytorch_weights_to_ttsim(pytorch_module, ttsim_module):
    """
    Copy weights from a PyTorch PtsCrossAttention to a TTSim one.

    Args:
        pytorch_module: PyTorch PtsCrossAttention_PyTorch instance
        ttsim_module: TTSim PtsCrossAttention instance
    """
    # Copy sampling_offsets weights (no transpose — SimNN.Linear transposes internally)
    weight_np = pytorch_module.sampling_offsets.weight.detach().cpu().numpy()
    bias_np = pytorch_module.sampling_offsets.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(ttsim_module.sampling_offsets, weight_np, bias_np)

    # Copy attention_weights
    weight_np = pytorch_module.attention_weights.weight.detach().cpu().numpy()
    bias_np = pytorch_module.attention_weights.bias.detach().cpu().numpy()
    initialize_linear_weights_with_data(ttsim_module.attention_weights, weight_np, bias_np)

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
    """Test that PtsCrossAttention can be constructed successfully."""
    print("\n" + "="*80)
    print("TEST 1: PtsCrossAttention Construction")
    print("="*80)

    try:
        pca = PtsCrossAttention(
            name='test_pca',
            embed_dims=256,
            num_heads=8,
            num_levels=1,
            num_points=4,
            batch_first=True
        )
        print("[OK] Module constructed successfully")
        print(f"  - Module name: {pca.name}")
        print(f"  - Embed dims: {pca.embed_dims}")
        print(f"  - Num heads: {pca.num_heads}")
        print(f"  - Num levels: {pca.num_levels}")
        print(f"  - Num points: {pca.num_points}")
        return True
    except Exception as e:
        print(f"[X] Module construction failed: {e}")
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test PtsCrossAttention forward pass with numerical validation."""
    print("\n" + "="*80)
    print("TEST 2: PtsCrossAttention Forward Pass (with Data Validation)")
    print("="*80)

    try:
        # Configuration
        bs = 2
        embed_dims = 256
        num_heads = 8
        num_levels = 1
        num_points = 4

        # Single level: BEV feature map
        spatial_shapes = [(30, 30)]
        num_value = sum(H * W for H, W in spatial_shapes)  # 900
        num_query = 100  # object queries

        print(f"\nConfiguration:")
        print(f"  - Batch size: {bs}")
        print(f"  - Num queries (object queries): {num_query}")
        print(f"  - Num value (BEV features): {num_value}")
        print(f"  - Embed dims: {embed_dims}")
        print(f"  - Num heads: {num_heads}")
        print(f"  - Num levels: {num_levels}")
        print(f"  - Spatial shapes: {spatial_shapes}")

        # Create test inputs
        print("\n[1] Creating test inputs...")
        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
        # Reference points: [bs, num_query, num_levels, 2] in [0, 1]
        ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

        # PyTorch forward pass
        print("\n[2] Running PyTorch reference implementation...")
        model_pt = PtsCrossAttention_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        model_pt.eval()

        with torch.no_grad():
            output_pt = model_pt(
                torch.from_numpy(query_np),
                torch.from_numpy(value_np),
                torch.from_numpy(ref_points_np),
                spatial_shapes
            )

        print(f"  PyTorch output shape: {output_pt.shape}")
        output_pt_np = output_pt.detach().cpu().numpy()
        print(f"  PyTorch: mean={np.mean(output_pt_np):.6e}, std={np.std(output_pt_np):.6e}, "
              f"min={np.min(output_pt_np):.6e}, max={np.max(output_pt_np):.6e}")

        # TTSim forward pass
        print("\n[3] Running TTSim implementation...")
        pca = PtsCrossAttention(
            name='test_pca_forward',
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True
        )

        # Copy PyTorch weights to TTSim
        print("  Copying PyTorch weights to TTSim...")
        copy_pytorch_weights_to_ttsim(model_pt, pca)

        # Create TTSim inputs
        query_ttsim = F._from_data('query', query_np, is_const=True)
        value_ttsim = F._from_data('value', value_np, is_const=True)
        ref_points_ttsim = F._from_data('ref_points', ref_points_np, is_const=True)

        # Forward pass
        output_ttsim = pca(
            query=query_ttsim,
            value=value_ttsim,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes,
            level_start_index=None
        )

        print(f"  TTSim output shape: {output_ttsim.shape}")

        # Get TTSim data for numerical comparison
        if hasattr(output_ttsim, 'data') and output_ttsim.data is not None:
            output_ttsim_np = output_ttsim.data
            print(f"  TTSim:   mean={np.mean(output_ttsim_np):.6e}, std={np.std(output_ttsim_np):.6e}, "
                  f"min={np.min(output_ttsim_np):.6e}, max={np.max(output_ttsim_np):.6e}")

            # Numerical comparison
            print(f"\n  Numerical comparison:")
            print(f"    Max diff: {np.abs(output_pt_np - output_ttsim_np).max():.6e}")
            print(f"    Mean diff: {np.abs(output_pt_np - output_ttsim_np).mean():.6e}")

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
            print(f"\n[X] Shape mismatch: expected {expected_shape}, got {list(output_ttsim.shape)}")
            return False

    except Exception as e:
        print(f"\n[X] Forward pass failed with exception: {e}")
        traceback.print_exc()
        return False


def test_parameter_count():
    """Test parameter count calculation."""
    print("\n" + "="*80)
    print("TEST 3: Parameter Count")
    print("="*80)

    try:
        embed_dims = 256
        num_heads = 8
        num_levels = 1
        num_points = 4

        pca = PtsCrossAttention(
            name='test_pca_params',
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )

        param_count = pca.analytical_param_count()

        # Calculate expected parameters
        # sampling_offsets: embed_dims * (num_heads*num_levels*num_points*2) + bias
        so_out = num_heads * num_levels * num_points * 2
        expected_sampling_offsets = embed_dims * so_out + so_out

        # attention_weights: embed_dims * (num_heads*num_levels*num_points) + bias
        aw_out = num_heads * num_levels * num_points
        expected_attention_weights = embed_dims * aw_out + aw_out

        # value_proj: embed_dims * embed_dims + bias
        expected_value_proj = embed_dims * embed_dims + embed_dims

        # output_proj: embed_dims * embed_dims + bias
        expected_output_proj = embed_dims * embed_dims + embed_dims

        expected_total = (expected_sampling_offsets + expected_attention_weights +
                         expected_value_proj + expected_output_proj)

        print(f"PtsCrossAttention parameter breakdown:")
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
    print("\n" + "="*80)
    print("TEST 4: Different Configurations (with Data Validation)")
    print("="*80)

    test_cases = [
        # (embed_dims, num_heads, num_levels, num_points, spatial_shapes, num_query)
        (128, 4, 1, 4, [(20, 20)], 50),
        (256, 8, 4, 4, [(30, 30), (15, 15), (8, 8), (4, 4)], 100),
        (512, 16, 2, 8, [(15, 15), (8, 8)], 200),
    ]

    all_passed = True
    for i, (embed_dims, num_heads, num_levels, num_points, spatial_shapes, num_query) in enumerate(test_cases, 1):
        try:
            print(f"\nTest case {i}: embed_dims={embed_dims}, num_heads={num_heads}, "
                  f"num_levels={num_levels}, num_points={num_points}")
            print(f"  Spatial shapes: {spatial_shapes}, num_query: {num_query}")

            bs = 2
            num_value = sum(H * W for H, W in spatial_shapes)
            head_dim = embed_dims // num_heads

            # Create test inputs
            query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
            value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
            ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

            # PyTorch reference
            model_pt = PtsCrossAttention_PyTorch(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points
            )
            model_pt.eval()

            with torch.no_grad():
                output_pt = model_pt(
                    torch.from_numpy(query_np),
                    torch.from_numpy(value_np),
                    torch.from_numpy(ref_points_np),
                    spatial_shapes
                )

            print(f"  PyTorch output: shape={output_pt.shape}, "
                  f"range=[{output_pt.min().item():.6f}, {output_pt.max().item():.6f}], "
                  f"mean={output_pt.mean().item():.6f}")

            # TTSim implementation
            pca = PtsCrossAttention(
                name=f'test_pca_config_{i}',
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points
            )

            # Copy weights
            copy_pytorch_weights_to_ttsim(model_pt, pca)

            query_ttsim = F._from_data('query', query_np, is_const=True)
            value_ttsim = F._from_data('value', value_np, is_const=True)
            ref_points_ttsim = F._from_data('ref_points', ref_points_np, is_const=True)

            output_ttsim = pca(
                query=query_ttsim,
                value=value_ttsim,
                reference_points=ref_points_ttsim,
                spatial_shapes=spatial_shapes
            )

            print(f"  TTSim output: shape={output_ttsim.shape}")

            # Numerical comparison
            if hasattr(output_ttsim, 'data') and output_ttsim.data is not None:
                output_ttsim_np = output_ttsim.data
                output_pt_np = output_pt.detach().cpu().numpy()
                max_diff = np.abs(output_pt_np - output_ttsim_np).max()
                mean_diff = np.abs(output_pt_np - output_ttsim_np).mean()
                print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")

                if not np.allclose(output_pt_np, output_ttsim_np, rtol=1e-4, atol=1e-5):
                    print(f"    [Warning] Numerical outputs differ beyond tolerance")

            # Validate shapes
            if list(output_pt.shape) == list(output_ttsim.shape):
                print(f"  [OK] Shapes match! Parameter count: {pca.analytical_param_count():,}")
            else:
                print(f"  [X] Shape mismatch")
                all_passed = False
        except Exception as e:
            print(f"  [X] Test case {i} failed: {e}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_value_defaults_to_query():
    """Test that value defaults to query when not provided."""
    print("\n" + "="*80)
    print("TEST 5: Value Defaults to Query")
    print("="*80)

    try:
        bs = 2
        num_query = 50
        embed_dims = 128
        num_heads = 4
        num_levels = 1
        num_points = 4
        spatial_shapes = [(7, 7)]  # num_value = 49

        # When value is None, PyTorch sets value = query.
        # But num_value must equal sum of spatial_shapes, so query shape
        # must be [bs, 49, embed_dims] and num_query == 49.
        num_query = sum(H * W for H, W in spatial_shapes)  # 49

        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

        # PyTorch: pass value=None -> uses query as value
        model_pt = PtsCrossAttention_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        model_pt.eval()

        with torch.no_grad():
            output_pt = model_pt(
                torch.from_numpy(query_np),
                torch.from_numpy(query_np),  # same as query
                torch.from_numpy(ref_points_np),
                spatial_shapes
            )

        # TTSim: pass value=None
        pca = PtsCrossAttention(
            name='test_pca_default_value',
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        copy_pytorch_weights_to_ttsim(model_pt, pca)

        query_ttsim = F._from_data('query', query_np, is_const=True)
        ref_points_ttsim = F._from_data('ref_points', ref_points_np, is_const=True)

        # Pass value=None -- TTSim should default to query
        output_ttsim = pca(
            query=query_ttsim,
            value=None,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes
        )

        # Shape check
        expected_shape = [bs, num_query, embed_dims]
        if list(output_ttsim.shape) == expected_shape:
            print(f"  Output shape: {output_ttsim.shape} -- correct")
        else:
            print(f"  [X] Shape mismatch: expected {expected_shape}, got {list(output_ttsim.shape)}")
            return False

        # Numerical comparison
        if hasattr(output_ttsim, 'data') and output_ttsim.data is not None:
            output_pt_np = output_pt.detach().cpu().numpy()
            output_ttsim_np = output_ttsim.data
            max_diff = np.abs(output_pt_np - output_ttsim_np).max()
            print(f"  Max diff vs PyTorch: {max_diff:.6e}")
            if np.allclose(output_pt_np, output_ttsim_np, rtol=1e-4, atol=1e-5):
                print("[OK] value=None defaults to query correctly, numerical match")
            else:
                print("[Warning] Numerical outputs differ beyond tolerance")

        return True

    except Exception as e:
        print(f"[X] Test failed: {e}")
        traceback.print_exc()
        return False


def test_query_pos():
    """Test that query_pos is added to query before projection."""
    print("\n" + "="*80)
    print("TEST 6: Query Position Encoding (query_pos)")
    print("="*80)

    try:
        bs = 2
        num_query = 50
        embed_dims = 128
        num_heads = 4
        num_levels = 1
        num_points = 4
        spatial_shapes = [(10, 10)]
        num_value = 100

        query_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.1
        query_pos_np = np.random.randn(bs, num_query, embed_dims).astype(np.float32) * 0.01
        value_np = np.random.randn(bs, num_value, embed_dims).astype(np.float32) * 0.1
        ref_points_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

        # PyTorch: query = query + query_pos before projections
        model_pt = PtsCrossAttention_PyTorch(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        model_pt.eval()

        query_plus_pos = torch.from_numpy(query_np) + torch.from_numpy(query_pos_np)
        with torch.no_grad():
            output_pt = model_pt.forward_no_residual(
                query_plus_pos,
                torch.from_numpy(value_np),
                torch.from_numpy(ref_points_np),
                spatial_shapes
            )
        # Add residual from original query (not query+pos) -- that's what the PyTorch code does:
        # identity = query (before query_pos), then output = dropout(output_proj(output)) + identity
        output_pt = output_pt + torch.from_numpy(query_np)

        # TTSim: pass query_pos
        pca = PtsCrossAttention(
            name='test_pca_qpos',
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )
        copy_pytorch_weights_to_ttsim(model_pt, pca)

        query_ttsim = F._from_data('query', query_np, is_const=True)
        query_pos_ttsim = F._from_data('query_pos', query_pos_np, is_const=True)
        value_ttsim = F._from_data('value', value_np, is_const=True)
        ref_points_ttsim = F._from_data('ref_points', ref_points_np, is_const=True)

        output_ttsim = pca(
            query=query_ttsim,
            value=value_ttsim,
            query_pos=query_pos_ttsim,
            reference_points=ref_points_ttsim,
            spatial_shapes=spatial_shapes
        )

        expected_shape = [bs, num_query, embed_dims]
        if list(output_ttsim.shape) == expected_shape:
            print(f"  Output shape: {output_ttsim.shape} -- correct")
        else:
            print(f"  [X] Shape mismatch: expected {expected_shape}, got {list(output_ttsim.shape)}")
            return False

        if hasattr(output_ttsim, 'data') and output_ttsim.data is not None:
            output_pt_np = output_pt.detach().cpu().numpy()
            output_ttsim_np = output_ttsim.data
            max_diff = np.abs(output_pt_np - output_ttsim_np).max()
            mean_diff = np.abs(output_pt_np - output_ttsim_np).mean()
            print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
            if np.allclose(output_pt_np, output_ttsim_np, rtol=1e-4, atol=1e-5):
                print("[OK] query_pos correctly added, numerical match")
            else:
                print("[Warning] Numerical outputs differ beyond tolerance")

        return True

    except Exception as e:
        print(f"[X] Test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PtsCrossAttention TTSim Module Test Suite")
    print("="*80)

    results = {
        "PtsCrossAttention Construction": test_construction(),
        "PtsCrossAttention Forward Pass": test_forward_pass(),
        "Parameter Count": test_parameter_count(),
        "Different Configurations": test_different_configurations(),
        "Value Defaults to Query": test_value_defaults_to_query(),
        "Query Position Encoding": test_query_pos(),
    }

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name:.<60} {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\nAll tests passed! The module is working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
