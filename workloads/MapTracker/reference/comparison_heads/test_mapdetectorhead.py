#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script to validate MapDetectorHead TTSim implementation against PyTorch.

This script:
1. Creates PyTorch MapDetectorHead (inference only)
2. Creates TTSim MapDetectorHead with same architecture
3. Copies weights from PyTorch to TTSim
4. Compares outputs for same inputs
"""

import sys
import os

# Add paths
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn

# Import TTSim components
from workloads.MapTracker.plugin.models.heads.MapDetectorHead import (
    RegressionBranch,
    ClassificationBranch,
)


def test_regression_branch():
    """Test RegressionBranch against PyTorch."""
    print("\n" + "=" * 80)
    print("TEST 1: RegressionBranch")
    print("=" * 80)

    embed_dims = 256
    num_points = 20
    coord_dim = 2
    bs = 2
    num_queries = 100

    # Create PyTorch regression branch
    torch_reg = nn.Sequential(
        nn.Linear(embed_dims, 2 * embed_dims),
        nn.LayerNorm(2 * embed_dims),
        nn.ReLU(),
        nn.Linear(2 * embed_dims, 2 * embed_dims),
        nn.LayerNorm(2 * embed_dims),
        nn.ReLU(),
        nn.Linear(2 * embed_dims, num_points * coord_dim),
    )

    # Create TTSim regression branch
    ttsim_reg = RegressionBranch(embed_dims, num_points, coord_dim)

    # Force PyTorch LayerNorm to identity (weight=1, bias=0) since
    # TTSim LayerNorm is non-affine (normalize only).
    with torch.no_grad():
        torch_reg[1].weight.fill_(1.0)
        torch_reg[1].bias.fill_(0.0)
        torch_reg[4].weight.fill_(1.0)
        torch_reg[4].bias.fill_(0.0)

    # Copy Linear weights: PyTorch and TTSim both store [out, in]
    ttsim_reg.fc1.param.data = torch_reg[0].weight.detach().numpy()
    ttsim_reg.fc1.bias.data = torch_reg[0].bias.detach().numpy()
    ttsim_reg.fc2.param.data = torch_reg[3].weight.detach().numpy()
    ttsim_reg.fc2.bias.data = torch_reg[3].bias.detach().numpy()
    ttsim_reg.fc3.param.data = torch_reg[6].weight.detach().numpy()
    ttsim_reg.fc3.bias.data = torch_reg[6].bias.detach().numpy()

    # Create input
    x_np = np.random.randn(bs, num_queries, embed_dims).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    # PyTorch forward
    torch_reg.eval()
    with torch.no_grad():
        y_torch = torch_reg(x_torch)

    # TTSim forward
    import ttsim.front.functional.op as F

    x_ttsim = F._from_data("test_input", x_np, is_const=False)
    y_ttsim = ttsim_reg(x_ttsim)

    # Compare
    y_ttsim_np = y_ttsim.data
    y_torch_np = y_torch.numpy()

    diff = np.abs(y_ttsim_np - y_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {y_torch_np.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Parameter count
    param_count = ttsim_reg.analytical_param_count(lvl=1)

    # PyTorch param count
    torch_params = sum(p.numel() for p in torch_reg.parameters())
    print(f"PyTorch params: {torch_params:,}")

    if max_diff < 1e-5:
        print("[OK] RegressionBranch test PASSED")
    else:
        print("[X] RegressionBranch test FAILED")

    return max_diff < 1e-5


def test_classification_branch():
    """Test ClassificationBranch against PyTorch."""
    print("\n" + "=" * 80)
    print("TEST 2: ClassificationBranch")
    print("=" * 80)

    embed_dims = 256
    num_classes = 3
    bs = 2
    num_queries = 100

    # Create PyTorch classification branch
    torch_cls = nn.Linear(embed_dims, num_classes)

    # Create TTSim classification branch
    ttsim_cls = ClassificationBranch(embed_dims, num_classes)

    # Copy weights directly to model attributes
    ttsim_cls.fc.param.data = torch_cls.weight.detach().numpy()
    ttsim_cls.fc.bias.data = torch_cls.bias.detach().numpy()

    # Create input
    x_np = np.random.randn(bs, num_queries, embed_dims).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    # PyTorch forward
    torch_cls.eval()
    with torch.no_grad():
        y_torch = torch_cls(x_torch)

    # TTSim forward
    import ttsim.front.functional.op as F

    x_ttsim = F._from_data("test_input", x_np, is_const=False)
    y_ttsim = ttsim_cls(x_ttsim)

    # Compare
    y_ttsim_np = y_ttsim.data
    y_torch_np = y_torch.numpy()

    diff = np.abs(y_ttsim_np - y_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Input shape: {x_np.shape}")
    print(f"Output shape: {y_torch_np.shape}")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    # Parameter count
    param_count = ttsim_cls.analytical_param_count(lvl=1)

    # PyTorch param count
    torch_params = sum(p.numel() for p in torch_cls.parameters())
    print(f"PyTorch params: {torch_params:,}")

    if max_diff < 1e-6:
        print("[OK] ClassificationBranch test PASSED")
    else:
        print("[X] ClassificationBranch test FAILED")

    return max_diff < 1e-6


def test_head_components():
    """Test individual MapDetectorHead components."""
    print("\n" + "=" * 80)
    print("TEST 3: MapDetectorHead Component Construction")
    print("=" * 80)

    # Configuration
    num_queries = 100
    num_classes = 3
    in_channels = 128
    embed_dims = 256
    num_points = 20
    coord_dim = 2
    num_layers = 6

    from workloads.MapTracker.plugin.models.heads.MapDetectorHead import MapDetectorHead

    # Create head (without transformer for now)
    head = MapDetectorHead(
        num_queries=num_queries,
        num_classes=num_classes,
        in_channels=in_channels,
        embed_dims=embed_dims,
        num_points=num_points,
        coord_dim=coord_dim,
        num_layers=num_layers,
        different_heads=True,
        predict_refine=False,
        transformer=None,
    )

    print(f"Created MapDetectorHead:")
    print(f"  num_queries: {num_queries}")
    print(f"  num_classes: {num_classes}")
    print(f"  embed_dims: {embed_dims}")
    print(f"  num_points: {num_points}")
    print(f"  num_layers: {num_layers}")
    print(f"  Classification branches: {len(head.cls_branches)}")
    print(f"  Regression branches: {len(head.reg_branches)}")

    # Set dummy weights
    head.set_input_proj_weights(
        weight=np.random.randn(embed_dims, in_channels, 1, 1).astype(np.float32),
        bias=np.random.randn(embed_dims).astype(np.float32),
    )
    head.set_query_embedding_weights(
        weight=np.random.randn(num_queries, embed_dims).astype(np.float32)
    )
    head.set_ref_points_weights(
        weight=np.random.randn(num_points * 2, embed_dims).astype(np.float32),
        bias=np.random.randn(num_points * 2).astype(np.float32),
    )

    # Set branch weights directly on model attributes
    for i in range(num_layers):
        head.cls_branches[i].fc.param.data = np.random.randn(
            num_classes, embed_dims
        ).astype(np.float32)
        head.cls_branches[i].fc.bias.data = np.random.randn(num_classes).astype(
            np.float32
        )

        hidden_dims = 2 * embed_dims
        out_dims = num_points * coord_dim
        head.reg_branches[i].fc1.param.data = np.random.randn(
            hidden_dims, embed_dims
        ).astype(np.float32)
        head.reg_branches[i].fc1.bias.data = np.random.randn(hidden_dims).astype(
            np.float32
        )
        head.reg_branches[i].fc2.param.data = np.random.randn(
            hidden_dims, hidden_dims
        ).astype(np.float32)
        head.reg_branches[i].fc2.bias.data = np.random.randn(hidden_dims).astype(
            np.float32
        )
        head.reg_branches[i].fc3.param.data = np.random.randn(
            out_dims, hidden_dims
        ).astype(np.float32)
        head.reg_branches[i].fc3.bias.data = np.random.randn(out_dims).astype(
            np.float32
        )

    # Test BEV positional embedding generation
    bs, h, w = 2, 50, 100
    bev_features_np = np.random.randn(bs, in_channels, h, w).astype(np.float32)

    import ttsim.front.functional.op as F

    bev_features = F._from_data("bev_features", bev_features_np, is_const=False)

    # Test _prepare_context
    processed_features = head._prepare_context(bev_features)
    print(f"\nBEV features processing:")
    print(f"  Input shape: {bev_features_np.shape}")
    print(f"  Output shape: {processed_features.shape}")

    # Calculate parameter count
    param_count = head.analytical_param_count(lvl=2)

    print("\n[OK] MapDetectorHead component test PASSED")
    return True


def test_bev_pos_encoding_values():
    """Validate BEV positional encoding VALUES against correct SinePositionalEncoding.

    The TTSim _get_bev_pos_embed() now implements the correct algorithm
    matching mmcv SinePositionalEncoding(normalize=True, scale=2*pi):
      1. not_mask = ones(bs, H, W)
      2. y_embed = cumsum(not_mask, axis=1)  ->  1, 2, ..., H
      3. x_embed = cumsum(not_mask, axis=2)  ->  1, 2, ..., W
      4. Normalize by last element, scale to [0, 2*pi]
      5. Divide by temperature^(2i/num_feats), temperature=10000
      6. Interleave sin/cos at even/odd indices
      7. Concatenate [pos_y, pos_x] -> [bs, embed_dims, H, W]

    Previously the code used np.linspace normalized to [-1, 1], which
    produced completely wrong frequency content and spatial aliasing.

    Fix applied in MapDetectorHead._get_bev_pos_embed().
    """
    print("\n" + "=" * 80)
    print("TEST 4: BEV Positional Encoding Values")
    print("=" * 80)

    from workloads.MapTracker.plugin.models.heads.MapDetectorHead import MapDetectorHead
    import ttsim.front.functional.op as F_op

    embed_dims = 256
    bs, h, w = 1, 10, 20
    num_feats = embed_dims // 2  # 128

    # --- Compute CORRECT SinePositionalEncoding (cumsum-based) ---
    scale = 2.0 * np.pi
    temperature = 10000.0
    eps = 1e-6

    not_mask = np.ones((bs, h, w), dtype=np.float32)
    y_embed = np.cumsum(not_mask, axis=1)  # [bs, H, W]: 1..H along rows
    x_embed = np.cumsum(not_mask, axis=2)  # [bs, H, W]: 1..W along cols

    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = np.arange(num_feats, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    pos_x = x_embed[:, :, :, None] / dim_t  # [bs, h, w, num_feats]
    pos_y = y_embed[:, :, :, None] / dim_t  # [bs, h, w, num_feats]

    # Interleave sin/cos
    pos_x_enc = np.zeros_like(pos_x)
    pos_x_enc[:, :, :, 0::2] = np.sin(pos_x[:, :, :, 0::2])
    pos_x_enc[:, :, :, 1::2] = np.cos(pos_x[:, :, :, 1::2])

    pos_y_enc = np.zeros_like(pos_y)
    pos_y_enc[:, :, :, 0::2] = np.sin(pos_y[:, :, :, 0::2])
    pos_y_enc[:, :, :, 1::2] = np.cos(pos_y[:, :, :, 1::2])

    # Concatenate and permute: [bs, h, w, C] -> [bs, C, h, w]
    correct_pos = np.concatenate([pos_y_enc, pos_x_enc], axis=3)
    correct_pos = correct_pos.transpose(0, 3, 1, 2).astype(np.float32)

    # --- Compute TTSim's _get_bev_pos_embed ---
    head = MapDetectorHead(
        num_queries=10,
        num_classes=3,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_points=20,
        coord_dim=2,
        num_layers=1,
        different_heads=True,
        predict_refine=False,
        transformer=None,
    )

    ttsim_pos_tensor = head._get_bev_pos_embed(bs, h, w)
    ttsim_pos = ttsim_pos_tensor.data  # [bs, embed_dims, h, w]

    # --- Compare ---
    assert (
        correct_pos.shape == ttsim_pos.shape
    ), f"Shape mismatch: correct {correct_pos.shape} vs TTSim {ttsim_pos.shape}"

    diff = np.abs(correct_pos - ttsim_pos)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(
        f"  Correct (cumsum) range: [{correct_pos.min():.4f}, {correct_pos.max():.4f}]"
    )
    print(f"  TTSim (linspace)  range: [{ttsim_pos.min():.4f}, {ttsim_pos.max():.4f}]")
    print(f"  Max diff:  {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    passed = max_diff < 1e-5
    if passed:
        print("[OK] BEV positional encoding values match SinePositionalEncoding")
    else:
        print(
            "[X] BEV positional encoding values WRONG — "
            "using linspace[-1,1] instead of cumsum-based [0, 2*pi*scale]"
        )
    return passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MapDetectorHead TTSim Implementation Tests")
    print("=" * 80)

    results = []

    # Test 1: RegressionBranch
    try:
        results.append(("RegressionBranch", test_regression_branch()))
    except Exception as e:
        print(f"[X] RegressionBranch test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("RegressionBranch", False))

    # Test 2: ClassificationBranch
    try:
        results.append(("ClassificationBranch", test_classification_branch()))
    except Exception as e:
        print(f"[X] ClassificationBranch test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("ClassificationBranch", False))

    # Test 3: Head components
    try:
        results.append(("MapDetectorHead Components", test_head_components()))
    except Exception as e:
        print(f"[X] MapDetectorHead component test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("MapDetectorHead Components", False))

    # Test 4: BEV positional encoding values
    try:
        results.append(("BEV Pos Encoding Values", test_bev_pos_encoding_values()))
    except Exception as e:
        print(f"[X] BEV pos encoding test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("BEV Pos Encoding Values", False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED [OK]")
    else:
        print("SOME TESTS FAILED [X]")
    print("=" * 80)


if __name__ == "__main__":
    main()
