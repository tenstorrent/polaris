#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for DeformableTransformer (full encoder-decoder architecture).
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
    DeformableTransformer as TransformerPyTorch,
)
from workloads.Deformable_DETR.models.deformable_transformer_ttsim import (
    DeformableTransformer as TransformerTTSim,
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


def test_full_transformer():
    """Test full DeformableTransformer"""

    print("\n" + "=" * 80)
    print("TEST: DeformableTransformer (Full Encoder-Decoder)")
    print("=" * 80)
    print(
        "\nObjective: Validate SHAPE + NUMERICAL computation for complete transformer"
    )
    print("What we compute:")
    print("  1. Encoder: Multi-level feature encoding with deformable attention")
    print("  2. Decoder: Query refinement through self & cross attention")
    print("  3. Complete object detection transformer pipeline")
    print("Output: Decoded queries [num_layers, batch, num_queries, d_model]\n")

    # Configuration
    batch_size = 2
    num_queries = 50  # Smaller for faster testing
    d_model = 256
    num_feature_levels = 4
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2

    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num queries: {num_queries}")
    print(f"  d_model: {d_model}")
    print(f"  Feature levels: {num_feature_levels}")
    print(f"  Encoder layers: {num_encoder_layers}")
    print(f"  Decoder layers: {num_decoder_layers}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Multi-scale features
    srcs = []
    masks = []
    pos_embeds = []
    for lvl in range(num_feature_levels):
        h = 56 // (2**lvl)
        w = 56 // (2**lvl)
        src = torch.randn(batch_size, d_model, h, w)
        mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
        pos_embed = torch.randn(batch_size, d_model, h, w)
        srcs.append(src)
        masks.append(mask)
        pos_embeds.append(pos_embed)

    query_embed = torch.randn(num_queries, d_model * 2)  # Split into tgt + query_pos

    print(f"\nMulti-scale features:")
    for i, src in enumerate(srcs):
        print(f"  Level {i}: {list(src.shape)}")
    print(f"\nQuery embeddings: {list(query_embed.shape)}")

    # ─────────────────────────────────────────────────────────────────
    # PyTorch Forward Pass
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("PYTORCH Forward Pass")
    print("-" * 80)

    transformer_pytorch = TransformerPyTorch(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=4,
        enc_n_points=4,
    )
    transformer_pytorch.eval()

    with torch.no_grad():
        hs, init_reference, inter_references, _, _ = transformer_pytorch(
            srcs, masks, pos_embeds, query_embed
        )

    pytorch_data = hs.detach().cpu().numpy()
    print(f"\nPyTorch Output:")
    print(f"  hs (decoder outputs): {list(hs.shape)}")
    print(f"  init_reference: {list(init_reference.shape)}")
    print(f"  inter_references: {list(inter_references.shape)}")
    print(f"\n  hs statistics:")
    print(f"    Mean:  {pytorch_data.mean():.8f}")
    print(f"    Std:   {pytorch_data.std():.8f}")
    print(f"    Min:   {pytorch_data.min():.8f}")
    print(f"    Max:   {pytorch_data.max():.8f}")

    has_nan = np.isnan(pytorch_data).any()
    print(f"    NaN values: {'YES ✗' if has_nan else 'NO ✓'}")

    # ─────────────────────────────────────────────────────────────────
    # TTSim Forward Pass
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("TTSIM Forward Pass")
    print("-" * 80)

    # Create transformer first
    transformer_ttsim = TransformerTTSim(
        name="transformer_ttsim",
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=num_feature_levels,
        dec_n_points=4,
        enc_n_points=4,
    )

    # Create inputs and link them to the transformer module
    srcs_ttsim = []
    for i, src in enumerate(srcs):
        src_sim = torch_to_simtensor(src, f"src_{i}")
        src_sim.set_module(transformer_ttsim)
        srcs_ttsim.append(src_sim)

    masks_ttsim = []
    for i, mask in enumerate(masks):
        mask_sim = torch_to_simtensor(mask.float(), f"mask_{i}")
        mask_sim.set_module(transformer_ttsim)
        masks_ttsim.append(mask_sim)

    pos_embeds_ttsim = []
    for i, pos_embed in enumerate(pos_embeds):
        pos_sim = torch_to_simtensor(pos_embed, f"pos_embed_{i}")
        pos_sim.set_module(transformer_ttsim)
        pos_embeds_ttsim.append(pos_sim)

    query_embed_ttsim = torch_to_simtensor(query_embed, "query_embed")
    query_embed_ttsim.set_module(transformer_ttsim)

    hs_ttsim, init_ref_ttsim, inter_ref_ttsim, _, _ = transformer_ttsim(
        srcs_ttsim, masks_ttsim, pos_embeds_ttsim, query_embed_ttsim
    )

    print(f"\nTTSim Output:")
    print(f"  hs shape: {hs_ttsim.shape}")
    print(f"  hs data: {type(hs_ttsim.data)}")

    if hs_ttsim.data is not None:
        ttsim_data = hs_ttsim.data
        print(f"  ✓ DATA AVAILABLE")
        print(f"    Mean:  {ttsim_data.mean():.8f}")
        print(f"    Std:   {ttsim_data.std():.8f}")
        print(f"    Min:   {ttsim_data.min():.8f}")
        print(f"    Max:   {ttsim_data.max():.8f}")
        has_nan_ttsim = np.isnan(ttsim_data).any()
        print(f"    NaN values: {'YES ✗' if has_nan_ttsim else 'NO ✓'}")
    else:
        print(f"  ⊘ DATA IS NONE (shape inference only)")

    # ─────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    expected_shape = [num_decoder_layers, batch_size, num_queries, d_model]
    pytorch_shape = list(hs.shape)
    ttsim_shape = hs_ttsim.shape

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

    print(f"\n2. NaN Check:")
    if has_nan:
        print(f"   ✗ FAILED: PyTorch output contains NaN values")
        return False
    else:
        print(f"   ✓ PASSED: No NaN values")

    print(f"\n3. Numerical Validation:")
    if hs_ttsim.data is None:
        print(f"   ⊘ SKIPPED (shape inference only)")
        return True
    else:
        if np.isnan(hs_ttsim.data).any():
            print(f"   ✗ FAILED: TTSim output contains NaN values")
            return False
        abs_diff = np.abs(pytorch_data - hs_ttsim.data)
        numerical_match = np.allclose(pytorch_data, hs_ttsim.data, rtol=1e-3, atol=1e-4)
        print(f"   Max error: {abs_diff.max():.6e}")
        if numerical_match:
            print(f"   ✓ PASSED")
            return True
        else:
            print(f"   ✗ FAILED")
            return False


if __name__ == "__main__":
    try:
        success = test_full_transformer()
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
