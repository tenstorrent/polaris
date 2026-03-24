#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Embedder module validation: ttsim vs PyTorch comparison.
Style aligned with Focus test: prints input stats, runs PT + ttsim, injects config,
then does a numeric comparison with max/mean diffs + allclose.
"""

import os
import sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

import numpy as np
import torch

import ttsim.front.functional.op as F
from ttsim.ops import SimTensor
from workloads.MapTracker.plugin.models.utils.query_update import (
    Embedder as EmbedderTtsim,
)


# PyTorch Embedder (MapTracker-style)
class EmbedderPyTorch:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0

        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)


# ttsim Embedder


def create_input_tensor_with_data(name: str, data: np.ndarray) -> SimTensor:
    """Create a ttsim SimTensor with numpy data."""
    return F._from_data(name, data.astype(np.float32), is_param=False, is_const=False)


print("=" * 70)
print("Testing Embedder Data Execution")
print("Concat([x, sin(2^k x), cos(2^k x)]) vs ttsim Embedder")
print("=" * 70)

# ----------------------------------------------------------------------
# Test configuration
# ----------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

input_dims = 7
max_freq_log2 = 9
num_freqs = 10

batch_size = 2
input_data = np.random.randn(batch_size, input_dims).astype(np.float32)

print(f"\nInput shape: [{batch_size}, {input_dims}]")
print("Input stats:")
print(f"  Min:  {input_data.min():.6f}")
print(f"  Max:  {input_data.max():.6f}")
print(f"  Mean: {input_data.mean():.6f}")
print(f"  Std:  {input_data.std():.6f}")
print(f"  Input Data:\n{input_data}\n")

# Common tolerances
atol = 1e-6
rtol = 1e-5

# ----------------------------------------------------------------------
# TEST 1: PyTorch Embedder
# ----------------------------------------------------------------------
print("TEST 1: PyTorch Embedder")
print("-" * 70)

embed_kwargs = {
    "include_input": True,
    "input_dims": input_dims,
    "max_freq_log2": max_freq_log2,
    "num_freqs": num_freqs,
    "log_sampling": True,
    "periodic_fns": [torch.sin, torch.cos],
}

embedder_pt = EmbedderPyTorch(**embed_kwargs)

with torch.no_grad():
    input_pt = torch.from_numpy(input_data)
    output_pt = embedder_pt.embed(input_pt).cpu().numpy()

print(f"  PyTorch output shape: {output_pt.shape}")
print(f"  Expected out_dim:     {embedder_pt.out_dim}")
print("  PyTorch output stats:")
print(f"    Min:  {output_pt.min():.6f}")
print(f"    Max:  {output_pt.max():.6f}")
print(f"    Mean: {output_pt.mean():.6f}")
print()

print("  PyTorch output values:")
np.set_printoptions(precision=10, suppress=True)
print(output_pt)
np.set_printoptions()
print()

# ----------------------------------------------------------------------
# TEST 2: TTSim Embedder
# ----------------------------------------------------------------------
print("TEST 2: TTSim Embedder")
print("-" * 70)

embedder_tt = EmbedderTtsim(
    "test_embedder",
    include_input=True,
    input_dims=input_dims,
    max_freq_log2=max_freq_log2,
    num_freqs=num_freqs,
    log_sampling=True,
    periodic_fns=["sin", "cos"],
)

input_tt = create_input_tensor_with_data("input", input_data)
output_tt = embedder_tt(input_tt)

print(f"  Input shape:  {input_tt.shape}")
print(f"  Output shape: {output_tt.shape}")
print(f"  Output .data is None? {output_tt.data is None}")

if output_tt.data is not None:
    print("  [OK] SUCCESS: Output data was computed!")
    print(f"  TTSim output shape: {output_tt.data.shape}")
    print("  TTSim output stats:")
    print(f"    Min:  {output_tt.data.min():.6f}")
    print(f"    Max:  {output_tt.data.max():.6f}")
    print(f"    Mean: {output_tt.data.mean():.6f}")
    print()

    print("  TTSim output values:")
    np.set_printoptions(precision=10, suppress=True)
    print(output_tt.data)
    np.set_printoptions()
    print()
else:
    print("  [FAIL] FAILED: Output data is still None\n")

# ----------------------------------------------------------------------
# TEST 3: Numerical Comparison
# ----------------------------------------------------------------------
if output_tt.data is not None:
    print("TEST 3: Numerical Comparison")
    print("-" * 70)

    print(f"  Tolerance: atol={atol}, rtol={rtol}")

    is_close = np.allclose(output_tt.data, output_pt, atol=atol, rtol=rtol)
    diff = np.abs(output_tt.data - output_pt)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(f"  Max absolute difference:  {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")
    print(f"  Arrays match (allclose):  {is_close}")

    if is_close:
        print(f"  [PASS] TTSim matches PyTorch (within atol={atol}, rtol={rtol})")
    else:
        print(f"  [FAIL] Differences exceed tolerance (max diff: {max_diff:.10f})")
    print()

print("=" * 70)
print("Test Complete!")
print("=" * 70)
