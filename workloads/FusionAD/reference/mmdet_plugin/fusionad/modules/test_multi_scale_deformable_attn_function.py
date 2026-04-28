#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for Multi-Scale Deformable Attention TTSim implementation.

Compares the TTSim CPU-only implementation against PyTorch's CPU fallback
(converted from mmcv) to ensure correctness.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

import os, sys
polaris_path = os.path.join(os.path.dirname(__file__), '..', '..', '..','..', '..','..')
sys.path.insert(0, polaris_path)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import TTSim implementation
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multi_scale_deformable_attn_function import multi_scale_deformable_attn_ttsim


# ============================================================================
# PyTorch Reference Implementation (converted from mmcv for Python 3.13)
# ============================================================================

def multi_scale_deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Converted from mmcv.ops.multi_scale_deform_attn for Python 3.13 + PyTorch 2.10.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs, num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs, num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape

    # Split value by spatial levels
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    # Normalize sampling locations to [-1, 1] range for grid_sample
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs * num_heads, embed_dims, H_, W_)

        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)

        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)

    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)

    # Stack sampling values and aggregate
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(bs, num_heads * embed_dims, num_queries)

    return output.transpose(1, 2).contiguous()

# ============================================================================
# Test Helper Functions
# ============================================================================

def pytorch_to_ttsim_tensor(torch_tensor):
    """Convert PyTorch tensor to TTSim SimTensor."""
    import ttsim.front.functional.op as F_ttsim
    np_data = torch_tensor.detach().cpu().numpy()
    return F_ttsim._from_data('tensor', np_data)


def run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights):
    """Run PyTorch CPU implementation."""
    # Ensure all tensors are on CPU
    value = value.cpu()
    sampling_locations = sampling_locations.cpu()
    attention_weights = attention_weights.cpu()

    # Call PyTorch reference
    output = multi_scale_deformable_attn_pytorch(
        value, spatial_shapes,
        sampling_locations, attention_weights
    )

    return output


def run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights):
    """Run TTSim implementation."""
    # Convert PyTorch tensors to TTSim tensors
    value_ttsim = pytorch_to_ttsim_tensor(value)
    sampling_locations_ttsim = pytorch_to_ttsim_tensor(sampling_locations)
    attention_weights_ttsim = pytorch_to_ttsim_tensor(attention_weights)

    # Convert spatial_shapes
    spatial_shapes_list = [(int(h), int(w)) for h, w in spatial_shapes]

    # Call TTSim implementation
    output_ttsim = multi_scale_deformable_attn_ttsim(
        'msda_test',
        value_ttsim,
        spatial_shapes_list,
        sampling_locations_ttsim,
        attention_weights_ttsim
    )

    # Extract numpy data from TTSim tensor
    return torch.from_numpy(output_ttsim.data)


# ============================================================================
# Test Cases
# ============================================================================


def test_basic_single_level():
    """Test with single level (simplest case)."""
    bs, num_queries, num_heads, embed_dims_per_head = 2, 10, 8, 32
    num_levels, num_points = 1, 4
    H, W = 20, 20

    torch.manual_seed(42)

    # Create inputs
    value = torch.randn(bs, H*W, num_heads, embed_dims_per_head)
    spatial_shapes = torch.tensor([[H, W]], dtype=torch.long)

    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    sampling_locations = sampling_locations.clamp(0, 1)  # Ensure in valid range

    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

    # Run both implementations
    output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
    output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

    # Compare
    assert output_pytorch.shape == output_ttsim.shape, \
        f"Shape mismatch: PyTorch {output_pytorch.shape} vs TTSim {output_ttsim.shape}"

    # Convert to numpy for comparison
    output_pytorch_np = output_pytorch.detach().cpu().numpy()
    output_ttsim_np = output_ttsim.detach().cpu().numpy()

    print(f"\nSingle Level Test:")
    print(f"  Output shape: {output_pytorch.shape}")
    print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}, min: {output_pytorch.min().item():.6e}, max: {output_pytorch.max().item():.6e}")
    print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}, min: {output_ttsim.min().item():.6e}, max: {output_ttsim.max().item():.6e}")
    print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}")
    print(f"  Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

    assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
        f"Outputs do not match within tolerance"


def test_multi_level_4_levels():
    """Test with 4 levels (typical BEVFormer configuration)."""
    bs, num_queries, num_heads, embed_dims_per_head = 1, 20, 8, 32
    num_levels, num_points = 4, 4

    # Multi-scale spatial shapes (typical feature pyramid)
    spatial_shapes = torch.tensor([
        [50, 50],   # Level 0: H=50, W=50
        [25, 25],   # Level 1: H=25, W=25
        [13, 13],   # Level 2: H=13, W=13
        [7, 7],     # Level 3: H=7, W=7
    ], dtype=torch.long)

    num_keys = sum([h * w for h, w in spatial_shapes])

    torch.manual_seed(123)

    # Create inputs
    value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)

    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    sampling_locations = sampling_locations.clamp(0, 1)

    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

    # Run both implementations
    output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
    output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

    # Compare
    assert output_pytorch.shape == output_ttsim.shape

    # Convert to numpy for comparison
    output_pytorch_np = output_pytorch.detach().cpu().numpy()
    output_ttsim_np = output_ttsim.detach().cpu().numpy()

    print(f"\n4-Level Test:")
    print(f"  Output shape: {output_pytorch.shape}")
    print(f"  Spatial shapes: {spatial_shapes.tolist()}")
    print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}, min: {output_pytorch.min().item():.6e}, max: {output_pytorch.max().item():.6e}")
    print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}, min: {output_ttsim.min().item():.6e}, max: {output_ttsim.max().item():.6e}")
    print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}")
    print(f"  Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

    assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
        f"Outputs do not match within tolerance"


def test_batch_size_variations():
    """Test with different batch sizes."""
    for bs in [1, 2, 4]:
        num_queries, num_heads, embed_dims_per_head = 15, 8, 32
        num_levels, num_points = 2, 4
        spatial_shapes = torch.tensor([[20, 20], [10, 10]], dtype=torch.long)
        num_keys = sum([h * w for h, w in spatial_shapes])

        torch.manual_seed(456 + bs)

        value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)
        sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        sampling_locations = sampling_locations.clamp(0, 1)
        attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
        attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

        output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
        output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

        # Convert to numpy for comparison
        output_pytorch_np = output_pytorch.detach().cpu().numpy()
        output_ttsim_np = output_ttsim.detach().cpu().numpy()

        print(f"\nBatch Size {bs} Test:")
        print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}")
        print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}")
        print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}, Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

        assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
            f"Batch size {bs}: Outputs do not match within tolerance"


def test_num_heads_variations():
    """Test with different numbers of attention heads."""
    for num_heads in [4, 8, 16]:
        bs, num_queries, embed_dims_per_head = 2, 10, 32
        num_levels, num_points = 2, 4
        spatial_shapes = torch.tensor([[15, 15], [8, 8]], dtype=torch.long)
        num_keys = sum([h * w for h, w in spatial_shapes])

        torch.manual_seed(789 + num_heads)

        value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)
        sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        sampling_locations = sampling_locations.clamp(0, 1)
        attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
        attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

        output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
        output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

        # Convert to numpy for comparison
        output_pytorch_np = output_pytorch.detach().cpu().numpy()
        output_ttsim_np = output_ttsim.detach().cpu().numpy()

        print(f"\nNum Heads {num_heads} Test:")
        print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}")
        print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}")
        print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}, Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

        assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
            f"Num heads {num_heads}: Outputs do not match within tolerance"


def test_num_points_variations():
    """Test with different numbers of sampling points."""
    for num_points in [1, 4, 8]:
        bs, num_queries, num_heads, embed_dims_per_head = 2, 10, 8, 32
        num_levels = 2
        spatial_shapes = torch.tensor([[12, 12], [6, 6]], dtype=torch.long)
        num_keys = sum([h * w for h, w in spatial_shapes])

        torch.manual_seed(101 + num_points)

        value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)
        sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        sampling_locations = sampling_locations.clamp(0, 1)
        attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
        attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

        output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
        output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

        # Convert to numpy for comparison
        output_pytorch_np = output_pytorch.detach().cpu().numpy()
        output_ttsim_np = output_ttsim.detach().cpu().numpy()

        print(f"\nNum Points {num_points} Test:")
        print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}")
        print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}")
        print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}, Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

        assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
            f"Num points {num_points}: Outputs do not match within tolerance"


def test_boundary_sampling_locations():
    """Test with sampling locations at boundaries (0 and 1)."""
    bs, num_queries, num_heads, embed_dims_per_head = 1, 8, 8, 32
    num_levels, num_points = 2, 4
    spatial_shapes = torch.tensor([[10, 10], [5, 5]], dtype=torch.long)
    num_keys = sum([h * w for h, w in spatial_shapes])

    torch.manual_seed(202)

    value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)

    # Create sampling locations at boundaries
    sampling_locations = torch.zeros(bs, num_queries, num_heads, num_levels, num_points, 2)
    sampling_locations[..., 0, 0] = 0.0  # Left edge (point 0, x-coord)
    sampling_locations[..., 1, 0] = 1.0  # Right edge (point 1, x-coord)
    sampling_locations[..., 2, 1] = 0.0  # Top edge (point 2, y-coord)
    sampling_locations[..., 3, 1] = 1.0  # Bottom edge (point 3, y-coord)

    # Random weights
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

    output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
    output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

    # Convert to numpy for comparison
    output_pytorch_np = output_pytorch.detach().cpu().numpy()
    output_ttsim_np = output_ttsim.detach().cpu().numpy()

    print(f"\nBoundary Sampling Test:")
    print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}")
    print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}")
    print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}, Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

    assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
        f"Outputs do not match within tolerance"


def test_large_scale_bevformer_like():
    """Test with BEVFormer-like large scale configuration."""
    bs = 1
    num_queries = 900  # 30x30 BEV grid
    num_heads = 8
    embed_dims_per_head = 32
    num_levels = 4
    num_points = 8

    # BEVFormer-like multi-scale feature maps
    spatial_shapes = torch.tensor([
        [116, 200],  # Level 0
        [58, 100],   # Level 1
        [29, 50],    # Level 2
        [15, 25],    # Level 3
    ], dtype=torch.long)

    num_keys = sum([h * w for h, w in spatial_shapes])

    torch.manual_seed(999)

    value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
    sampling_locations = sampling_locations.clamp(0, 1)
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

    output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
    output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

    # Convert to numpy for comparison
    output_pytorch_np = output_pytorch.detach().cpu().numpy()
    output_ttsim_np = output_ttsim.detach().cpu().numpy()

    print(f"\nLarge Scale BEVFormer-like Test:")
    print(f"  Output shape: {output_pytorch.shape}")
    print(f"  Num queries: {num_queries}")
    print(f"  Num keys: {num_keys}")
    print(f"  PyTorch - mean: {output_pytorch.mean().item():.6e}, std: {output_pytorch.std().item():.6e}, min: {output_pytorch.min().item():.6e}, max: {output_pytorch.max().item():.6e}")
    print(f"  TTSim   - mean: {output_ttsim.mean().item():.6e}, std: {output_ttsim.std().item():.6e}, min: {output_ttsim.min().item():.6e}, max: {output_ttsim.max().item():.6e}")
    print(f"  Max diff: {np.abs(output_pytorch_np - output_ttsim_np).max():.6e}")
    print(f"  Mean diff: {np.abs(output_pytorch_np - output_ttsim_np).mean():.6e}")

    assert np.allclose(output_pytorch_np, output_ttsim_np, rtol=1e-4, atol=1e-5), \
        f"Outputs do not match within tolerance"


def test_output_shape_correctness():
    """Test that output shape is always correct."""
    test_configs = [
        (1, 10, 4, 32, 1, 4, [[10, 10]]),
        (2, 20, 8, 32, 2, 4, [[20, 20], [10, 10]]),
        (1, 50, 8, 32, 4, 8, [[40, 40], [20, 20], [10, 10], [5, 5]]),
    ]

    for config in test_configs:
        bs, num_queries, num_heads, embed_dims_per_head, num_levels, num_points, spatial_list = config
        spatial_shapes = torch.tensor(spatial_list, dtype=torch.long)
        num_keys = sum([h * w for h, w in spatial_shapes])

        value = torch.randn(bs, num_keys, num_heads, embed_dims_per_head)
        sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
        attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
        attention_weights = F.softmax(attention_weights.flatten(-2), dim=-1).view_as(attention_weights)

        output_pytorch = run_pytorch_msda(value, spatial_shapes, sampling_locations, attention_weights)
        output_ttsim = run_ttsim_msda(value, spatial_shapes, sampling_locations, attention_weights)

        expected_shape = torch.Size([bs, num_queries, num_heads * embed_dims_per_head])

        assert output_pytorch.shape == expected_shape, \
            f"PyTorch output shape mismatch: {output_pytorch.shape} vs {expected_shape}"
        assert output_ttsim.shape == expected_shape, \
            f"TTSim output shape mismatch: {output_ttsim.shape} vs {expected_shape}"

    print("\nShape Correctness Test: PASSED")


# ============================================================================
# TEST: Full MultiScaleDeformableAttention Module - Normalizer Shape Validation
# ============================================================================

def test_full_module_normalizer_shape():
    """
    Test the full MultiScaleDeformableAttention module
    to validate the offset normalizer produces correct 6D shape (1,1,1,nL,1,2).
    """
    import ttsim.front.functional.op as F_ttsim
    from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multi_scale_deformable_attn_function import (
        MultiScaleDeformableAttention
    )

    # Use num_levels != num_points to catch broadcasting bugs
    embed_dims = 256
    num_heads = 8
    num_levels = 4
    num_points = 8  # Different from num_levels!
    bs = 2
    num_query = 10

    print(f"\nFull Module Normalizer Shape Test:")
    print(f"  embed_dims={embed_dims}, num_heads={num_heads}")
    print(f"  num_levels={num_levels}, num_points={num_points} (intentionally different)")

    # Spatial shapes for each level
    spatial_shapes_list = [(20, 20), (10, 10), (5, 5), (3, 3)]
    num_value = sum(h * w for h, w in spatial_shapes_list)

    # Create TTSim module (batch_first=True for simplicity)
    msda = MultiScaleDeformableAttention(
        name='test_msda_full',
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        dropout=0.0,
        batch_first=True
    )

    # Initialize Linear layer weights with random data so data propagates
    for linear_name in ['sampling_offsets', 'attention_weights', 'value_proj', 'output_proj']:
        linear = getattr(msda, linear_name)
        fan_in, fan_out = linear.in_features, linear.out_features
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        linear.param.data = np.random.uniform(-limit, limit, linear.param.shape).astype(np.float32)
        if linear.bias is not None:
            linear.bias.data = np.zeros(linear.bias.shape, dtype=np.float32)

    # Create input tensors
    query = F_ttsim._from_data('query', np.random.randn(bs, num_query, embed_dims).astype(np.float32))
    value = F_ttsim._from_data('value', np.random.randn(bs, num_value, embed_dims).astype(np.float32))
    reference_points = F_ttsim._from_data('ref_pts',
        np.random.rand(bs, num_query, num_levels, 2).astype(np.float32))

    # Run module
    try:
        output = msda(
            query=query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes_list,
        )

        # Check output shape
        expected_shape = (bs, num_query, embed_dims)
        assert tuple(output.shape) == expected_shape, \
            f"Output shape mismatch: {output.shape} vs expected {expected_shape}"

        assert output.data is not None, "Output data should not be None"
        assert not np.isnan(output.data).any(), "Output contains NaN"
        assert not np.isinf(output.data).any(), "Output contains Inf"

        print(f"  Output shape: {output.shape} (expected {expected_shape})")
        print(f"  Output range: [{output.data.min():.4f}, {output.data.max():.4f}]")
        print(f"  [OK] Full module normalizer shape test PASSED")
    except Exception as e:
        print(f"  [FAIL] Full module test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# TEST: Full Module - Identity Residual in Non-Batch-First Mode
# ============================================================================

def test_full_module_identity_non_batch_first():
    """
    Test MultiScaleDeformableAttention with batch_first=False to validate
    the identity tensor is NOT incorrectly transposed before the residual add.
    """
    import ttsim.front.functional.op as F_ttsim
    from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multi_scale_deformable_attn_function import (
        MultiScaleDeformableAttention
    )

    embed_dims = 256
    num_heads = 8
    num_levels = 2
    num_points = 4
    bs = 2
    num_query = 15  # Different from bs to catch shape mismatches

    print(f"\nFull Module Identity Non-Batch-First Test:")
    print(f"  bs={bs}, num_query={num_query} (intentionally different)")
    print(f"  batch_first=False")

    spatial_shapes_list = [(10, 10), (5, 5)]
    num_value = sum(h * w for h, w in spatial_shapes_list)

    # Create TTSim module with batch_first=False
    msda = MultiScaleDeformableAttention(
        name='test_msda_nbf',
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        dropout=0.0,
        batch_first=False  # Non-batch-first mode
    )

    # Initialize Linear layer weights with random data so data propagates
    for linear_name in ['sampling_offsets', 'attention_weights', 'value_proj', 'output_proj']:
        linear = getattr(msda, linear_name)
        fan_in, fan_out = linear.in_features, linear.out_features
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        linear.param.data = np.random.uniform(-limit, limit, linear.param.shape).astype(np.float32)
        if linear.bias is not None:
            linear.bias.data = np.zeros(linear.bias.shape, dtype=np.float32)

    # Input in non-batch-first format: [num_query, bs, embed_dims]
    query = F_ttsim._from_data('query_nbf',
        np.random.randn(num_query, bs, embed_dims).astype(np.float32))
    value = F_ttsim._from_data('value_nbf',
        np.random.randn(num_value, bs, embed_dims).astype(np.float32))
    # reference_points is always [bs, num_query, num_levels, 2]
    reference_points = F_ttsim._from_data('ref_pts_nbf',
        np.random.rand(bs, num_query, num_levels, 2).astype(np.float32))

    # Provide an explicit identity tensor to test residual path
    identity = F_ttsim._from_data('identity_nbf',
        np.random.randn(num_query, bs, embed_dims).astype(np.float32))

    try:
        output = msda(
            query=query,
            value=value,
            identity=identity,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes_list,
        )

        # Output should be [num_query, bs, embed_dims] (non-batch-first)
        expected_shape = (num_query, bs, embed_dims)
        assert tuple(output.shape) == expected_shape, \
            f"Output shape mismatch: {output.shape} vs expected {expected_shape}"

        assert output.data is not None, "Output data should not be None"
        assert not np.isnan(output.data).any(), "Output contains NaN"

        print(f"  Output shape: {output.shape} (expected {expected_shape})")
        print(f"  [OK] Identity non-batch-first test PASSED")
    except Exception as e:
        print(f"  [FAIL] Identity non-batch-first test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_actual_levels_less_than_num_levels():
    """Test single-level MSDA matching the seg head / BEV encoder pattern.

    In FusionAD, the seg head and BEV encoder always use num_levels=1 with
    a single BEV feature map. The Linear layers are sized for exactly the
    provided number of levels. This test validates that pattern.
    """
    print("\n" + "-" * 40)
    print("Test: num_levels == 1 (seg head pattern)")
    print("-" * 40)

    from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multi_scale_deformable_attn_function import (
        MultiScaleDeformableAttention,
    )
    import ttsim.front.functional.op as F_ttsim

    bs, num_query, embed_dims = 1, 100, 256
    num_heads, num_levels, num_points = 8, 1, 4
    H, W = 25, 25

    msda = MultiScaleDeformableAttention(
        name='test_al',
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        batch_first=False,
    )

    # Initialize Linear layer weights so data propagates through the graph
    for linear_name in ['sampling_offsets', 'attention_weights', 'value_proj', 'output_proj']:
        linear = getattr(msda, linear_name)
        fan_in, fan_out = linear.in_features, linear.out_features
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        linear.param.data = np.random.uniform(-limit, limit, linear.param.shape).astype(np.float32)
        if linear.bias is not None:
            linear.bias.data = np.zeros(linear.bias.shape, dtype=np.float32)

    # Single-level spatial shapes
    spatial_shapes_list = [(H, W)]
    num_value = H * W  # 625 tokens for 1 level

    query_np = np.random.randn(num_query, bs, embed_dims).astype(np.float32)
    value_np = np.random.randn(num_value, bs, embed_dims).astype(np.float32)
    # Reference points: [bs, num_query, num_levels, 2]
    ref_pts_np = np.random.rand(bs, num_query, num_levels, 2).astype(np.float32)

    query = F_ttsim._from_data('test_al.query', query_np)
    value = F_ttsim._from_data('test_al.value', value_np)
    ref_pts = F_ttsim._from_data('test_al.ref_pts', ref_pts_np)

    try:
        output = msda(
            query=query,
            value=value,
            reference_points=ref_pts,
            spatial_shapes=spatial_shapes_list,
        )

        expected_shape = (num_query, bs, embed_dims)
        assert tuple(output.shape) == expected_shape, \
            f"Output shape mismatch: {output.shape} vs expected {expected_shape}"
        assert output.data is not None, "Output data should not be None"
        assert not np.isnan(output.data).any(), "Output contains NaN"

        print(f"  Output shape: {output.shape} (expected {expected_shape})")
        print(f"  num_levels={num_levels}")
        print(f"  [OK] single-level attention test PASSED")
    except Exception as e:
        print(f"  [FAIL] single-level attention test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    print("=" * 80)
    print("Multi-Scale Deformable Attention Validation Tests")
    print("=" * 80)

    test_basic_single_level()
    test_multi_level_4_levels()
    test_batch_size_variations()
    test_num_heads_variations()
    test_num_points_variations()
    test_boundary_sampling_locations()
    test_large_scale_bevformer_like()
    test_output_shape_correctness()

    print("\n" + "-" * 80)
    print("Full Module Tests (covers normalizer shape + identity transpose fixes)")
    print("-" * 80)

    test_full_module_normalizer_shape()
    test_full_module_identity_non_batch_first()

    print("\n" + "-" * 80)
    print("Single-Level Test (covers seg head / BEV encoder pattern)")
    print("-" * 80)

    test_actual_levels_less_than_num_levels()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
