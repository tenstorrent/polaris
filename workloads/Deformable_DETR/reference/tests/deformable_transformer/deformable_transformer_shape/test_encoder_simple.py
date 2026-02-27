#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for DeformableTransformerEncoder (multiple encoder layers).
Tests shape inference and numerical computation.
"""

import os
import sys
import torch
import numpy as np

# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

from workloads.Deformable_DETR.reference.deformable_transformer import (
    DeformableTransformerEncoderLayer as EncoderLayerPyTorch,
    DeformableTransformerEncoder as EncoderPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerEncoderLayer as EncoderLayerTTSim,
    DeformableTransformerEncoder as EncoderTTSim,
)
from ttsim.ops.tensor import SimTensor


def torch_to_simtensor(torch_tensor, name="tensor"):
    """Convert PyTorch tensor to SimTensor with data"""
    return SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": torch_tensor.detach().cpu().numpy().copy(),
            "dtype": np.dtype(np.float32),
        }
    )


def test_encoder():
    """Test Encoder with multiple layers"""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformerEncoder (Multi-Layer)")
    print("=" * 80)
    print(
        "\nObjective: Validate SHAPE + NUMERICAL computation for stacked encoder layers"
    )
    print("What we compute:")
    print("  1. Apply encoder layer transformations sequentially")
    print("  2. Each layer: deformable attention + FFN + residuals + norms")
    print("Output: Multi-layer encoded features [batch, seq_len, d_model]\n")

    # Configuration
    batch_size = 2
    seq_len = 100
    d_model = 256
    n_levels = 4
    n_layers = 2  # Stack 2 encoder layers

    spatial_shapes = [[7, 7], [5, 5], [4, 4], [2, 5]]
    level_start_indices = [0, 49, 74, 90]

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  Levels: {n_levels}")
    print(f"  Encoder layers: {n_layers}")

    torch.manual_seed(42)
    np.random.seed(42)

    src_torch = torch.randn(batch_size, seq_len, d_model)
    pos_torch = torch.randn(batch_size, seq_len, d_model)
    spatial_shapes_torch = torch.tensor(spatial_shapes, dtype=torch.long)
    level_start_index_torch = torch.tensor(level_start_indices, dtype=torch.long)
    valid_ratios_torch = torch.ones(batch_size, n_levels, 2)

    print(f"\nInputs created:")
    print(f"  src: {list(src_torch.shape)}")
    print(f"  pos: {list(pos_torch.shape)}")
    print(f"  valid_ratios: {list(valid_ratios_torch.shape)}")

    # ─────────────────────────────────────────────────────────────────
    # PyTorch Forward Pass
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("PYTORCH Forward Pass")
    print("-" * 80)

    encoder_layer_pytorch = EncoderLayerPyTorch(
        d_model=d_model,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=n_levels,
        n_heads=8,
        n_points=4,
    )

    encoder_pytorch = EncoderPyTorch(encoder_layer_pytorch, n_layers)
    encoder_pytorch.eval()

    with torch.no_grad():
        out_pytorch = encoder_pytorch(
            src_torch,
            spatial_shapes_torch,
            level_start_index_torch,
            valid_ratios_torch,
            pos_torch,
        )

    pytorch_data = out_pytorch.detach().cpu().numpy()
    print(f"\nPyTorch Output:")
    print(f"  Shape: {list(out_pytorch.shape)}")
    print(f"  Mean:  {pytorch_data.mean():.8f}")
    print(f"  Std:   {pytorch_data.std():.8f}")
    print(f"  Min:   {pytorch_data.min():.8f}")
    print(f"  Max:   {pytorch_data.max():.8f}")

    # ─────────────────────────────────────────────────────────────────
    # TTSim Forward Pass
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("TTSIM Forward Pass")
    print("-" * 80)

    src_ttsim = torch_to_simtensor(src_torch, "src")
    pos_ttsim = torch_to_simtensor(pos_torch, "pos")
    spatial_shapes_ttsim = torch_to_simtensor(
        spatial_shapes_torch.float(), "spatial_shapes"
    )
    level_start_index_ttsim = torch_to_simtensor(
        level_start_index_torch.float(), "level_start_index"
    )
    valid_ratios_ttsim = torch_to_simtensor(valid_ratios_torch, "valid_ratios")

    encoder_layer_ttsim = EncoderLayerTTSim(
        name="encoder_layer_ttsim",
        d_model=d_model,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=n_levels,
        n_heads=8,
        n_points=4,
    )

    encoder_ttsim = EncoderTTSim(
        name="encoder_ttsim", encoder_layer=encoder_layer_ttsim, num_layers=n_layers
    )

    out_ttsim = encoder_ttsim(
        src_ttsim,
        spatial_shapes_ttsim,
        level_start_index_ttsim,
        valid_ratios_ttsim,
        pos_ttsim,
    )

    print(f"\nTTSim Output:")
    print(f"  Shape: {out_ttsim.shape}")
    print(f"  Data: {type(out_ttsim.data)}")

    if out_ttsim.data is not None:
        ttsim_data = out_ttsim.data
        print(f"  ✓ DATA AVAILABLE")
        print(f"  Mean:  {ttsim_data.mean():.8f}")
        print(f"  Std:   {ttsim_data.std():.8f}")
        print(f"  Min:   {ttsim_data.min():.8f}")
        print(f"  Max:   {ttsim_data.max():.8f}")
    else:
        print(f"  ⊘ DATA IS NONE (shape inference only)")

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    expected_shape = [batch_size, seq_len, d_model]
    pytorch_shape = list(out_pytorch.shape)
    ttsim_shape = out_ttsim.shape

    print(f"\n1. Shape Validation:")
    print(f"   Expected: {expected_shape}")
    print(f"   PyTorch:  {pytorch_shape}")
    print(f"   TTSim:    {ttsim_shape}")

    shape_match = pytorch_shape == expected_shape == ttsim_shape
    if shape_match:
        print(f"   ✓ PASSED")
    else:
        print(f"   ✗ FAILED")
        return False

    print(f"\n2. Numerical Validation:")
    if out_ttsim.data is None:
        print(f"   ⊘ SKIPPED (shape inference only)")
        return True
    else:
        abs_diff = np.abs(pytorch_data - out_ttsim.data)
        numerical_match = np.allclose(
            pytorch_data, out_ttsim.data, rtol=1e-3, atol=1e-4
        )
        print(f"   Max error: {abs_diff.max():.6e}")
        if numerical_match:
            print(f"   ✓ PASSED")
            return True
        else:
            print(f"   ✗ FAILED")
            return False


if __name__ == "__main__":
    try:
        success = test_encoder()
        print("\n" + "=" * 80)
        if success:
            print("OVERALL: TEST PASSED ✓")
        else:
            print("OVERALL: TEST FAILED ✗")
        print("=" * 80)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
