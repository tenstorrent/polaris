#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for DeformableTransformerDecoder (multiple decoder layers).
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
    DeformableTransformerDecoderLayer as DecoderLayerPyTorch,
    DeformableTransformerDecoder as DecoderPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformerDecoderLayer as DecoderLayerTTSim,
    DeformableTransformerDecoder as DecoderTTSim,
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


def test_decoder():
    """Test Decoder with multiple layers"""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformerDecoder (Multi-Layer)")
    print("=" * 80)
    print(
        "\nObjective: Validate SHAPE + NUMERICAL computation for stacked decoder layers"
    )
    print("What we compute:")
    print("  1. Apply decoder layer transformations sequentially")
    print("  2. Each layer: self-attention + cross-attention + FFN")
    print("Output: Multi-layer refined queries [batch, num_queries, d_model]\n")

    # Configuration
    batch_size = 2
    num_queries = 100
    src_seq_len = 200
    d_model = 256
    n_levels = 4
    n_layers = 2

    src_spatial_shapes = [[10, 10], [7, 7], [5, 5], [2, 13]]
    src_level_start_indices = [0, 100, 149, 174]

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num queries: {num_queries}")
    print(f"  Memory length: {src_seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  Levels: {n_levels}")
    print(f"  Decoder layers: {n_layers}")

    torch.manual_seed(42)
    np.random.seed(42)

    tgt_torch = torch.randn(batch_size, num_queries, d_model)
    query_pos_torch = torch.randn(batch_size, num_queries, d_model)
    # Reference points should be [batch, num_queries, 2] for decoder input
    # Decoder will expand to [batch, num_queries, n_levels, 2] internally
    reference_points_torch = torch.rand(batch_size, num_queries, 2)
    src_torch = torch.randn(batch_size, src_seq_len, d_model)
    src_spatial_shapes_torch = torch.tensor(src_spatial_shapes, dtype=torch.long)
    level_start_index_torch = torch.tensor(src_level_start_indices, dtype=torch.long)
    src_valid_ratios_torch = torch.ones(batch_size, n_levels, 2)

    print(f"\nInputs created:")
    print(f"  tgt: {list(tgt_torch.shape)}")
    print(f"  query_pos: {list(query_pos_torch.shape)}")
    print(f"  reference_points: {list(reference_points_torch.shape)}")
    print(f"  src: {list(src_torch.shape)}")

    # ─────────────────────────────────────────────────────────────────
    # PyTorch Forward Pass
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("PYTORCH Forward Pass")
    print("-" * 80)

    decoder_layer_pytorch = DecoderLayerPyTorch(
        d_model=d_model,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=n_levels,
        n_heads=8,
        n_points=4,
    )

    decoder_pytorch = DecoderPyTorch(decoder_layer_pytorch, n_layers)
    decoder_pytorch.eval()

    with torch.no_grad():
        out_pytorch, _ = decoder_pytorch(
            tgt_torch,
            reference_points_torch,
            src_torch,
            src_spatial_shapes_torch,
            level_start_index_torch,
            src_valid_ratios_torch,
            query_pos_torch,
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

    tgt_ttsim = torch_to_simtensor(tgt_torch, "tgt")
    query_pos_ttsim = torch_to_simtensor(query_pos_torch, "query_pos")
    # Expand reference points to [batch, num_queries, n_levels, 2] for TTSim
    reference_points_expanded = reference_points_torch.unsqueeze(2).repeat(
        1, 1, n_levels, 1
    )
    reference_points_ttsim = torch_to_simtensor(
        reference_points_expanded, "reference_points"
    )
    src_ttsim = torch_to_simtensor(src_torch, "src")
    src_spatial_shapes_ttsim = torch_to_simtensor(
        src_spatial_shapes_torch.float(), "src_spatial_shapes"
    )
    level_start_index_ttsim = torch_to_simtensor(
        level_start_index_torch.float(), "level_start_index"
    )
    src_valid_ratios_ttsim = torch_to_simtensor(
        src_valid_ratios_torch, "src_valid_ratios"
    )

    decoder_layer_ttsim = DecoderLayerTTSim(
        name="decoder_layer_ttsim",
        d_model=d_model,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=n_levels,
        n_heads=8,
        n_points=4,
    )

    decoder_ttsim = DecoderTTSim(
        name="decoder_ttsim", decoder_layer=decoder_layer_ttsim, num_layers=n_layers
    )

    out_ttsim, _ = decoder_ttsim(
        tgt_ttsim,
        reference_points_ttsim,
        src_ttsim,
        src_spatial_shapes_ttsim,
        level_start_index_ttsim,
        src_valid_ratios_ttsim,
        query_pos_ttsim,
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

    expected_shape = [batch_size, num_queries, d_model]
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
        success = test_decoder()
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
