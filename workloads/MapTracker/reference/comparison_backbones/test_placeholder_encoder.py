#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test PlaceHolderEncoder against PyTorch reference.
"""

import os, sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

import numpy as np
import torch
import torch.nn as nn


# PyTorch version (inline to avoid mmcv import issues)
class PlaceHolderEncoder_torch(nn.Module):
    """PyTorch version of PlaceHolderEncoder"""

    def __init__(self, *args, embed_dims=None, **kwargs):
        super().__init__()
        self.embed_dims = embed_dims

    def forward(self, *args, query=None, **kwargs):
        return query


# Import ttsim version
from workloads.MapTracker.plugin.models.transformer_utils.base_transformer import (
    PlaceHolderEncoder as PlaceHolderEncoder_ttsim,
)


def test_placeholder_encoder():
    """
    Test PlaceHolderEncoder - should just return input unchanged.
    """
    print("=" * 80)
    print("Testing PlaceHolderEncoder")
    print("=" * 80)

    # Test parameters
    batch_size = 2
    num_queries = 100
    embed_dims = 256

    # Create random input query [num_queries, batch_size, embed_dims]
    np.random.seed(42)
    query_np = np.random.randn(num_queries, batch_size, embed_dims).astype(np.float32)
    query_torch = torch.from_numpy(query_np)

    # Create PyTorch model
    torch_encoder = PlaceHolderEncoder_torch(embed_dims=embed_dims)
    torch_encoder.eval()

    # Forward pass (PyTorch)
    with torch.no_grad():
        output_torch = torch_encoder(query=query_torch)

    # Convert to numpy
    output_torch_np = output_torch.numpy()

    # Create ttsim model
    ttsim_encoder = PlaceHolderEncoder_ttsim(embed_dims=embed_dims)

    # Forward pass (ttsim) - create ttsim tensor
    import ttsim.front.functional.op as F

    query_ttsim = F._from_data("query", query_np, is_const=True)
    ttsim_encoder._tensors[query_ttsim.name] = query_ttsim

    output_ttsim_tensor = ttsim_encoder(query=query_ttsim)

    # For placeholder encoder, output should be identical to input
    print(f"Input query shape: {query_np.shape}")
    print(f"PyTorch output shape: {output_torch_np.shape}")

    # Verify output is same as input
    input_output_diff = np.abs(query_np - output_torch_np).max()
    print(f"\nPyTorch input-output max diff: {input_output_diff}")

    # For ttsim, the output tensor should be the same object as input
    print(
        f"\nTTSim output is same object as input: {output_ttsim_tensor is query_ttsim}"
    )

    # Verify shapes match
    assert output_torch_np.shape == query_np.shape, "PyTorch output shape mismatch"

    # Verify PyTorch output equals input (placeholder encoder should be identity)
    assert (
        input_output_diff < 1e-6
    ), f"PyTorch placeholder encoder modified input! Diff: {input_output_diff}"

    # Verify ttsim returns same tensor object
    assert (
        output_ttsim_tensor is query_ttsim
    ), "TTSim placeholder encoder should return input unchanged"

    print("\n" + "=" * 80)
    print("[OK] PlaceHolderEncoder test PASSED")
    print("=" * 80)

    # Test parameter count
    param_count = ttsim_encoder.analytical_param_count(lvl=1)
    assert (
        param_count == 0
    ), f"PlaceHolderEncoder should have 0 params, got {param_count}"
    print(f"\n[OK] Parameter count test PASSED: {param_count} params")


if __name__ == "__main__":
    test_placeholder_encoder()
