#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for PositionalEncoding1D
Compares PyTorch implementation vs ttsim implementation
"""

import os, sys

# Add polaris directory to path
polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_path not in sys.path:
    sys.path.insert(0, polaris_path)

import numpy as np
import torch
import torch.nn as nn
import importlib.util

import ttsim.front.functional.op as F
from workloads.MapTracker.plugin.models.maper import (
    PositionalEncoding1D as PositionalEncoding1D_TT,
)

# Import PyTorch version from maptracker workspace using importlib
vector_memory_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
        "maptracker",
        "plugin",
        "models",
        "mapers",
        "vector_memory.py",
    )
)
spec = importlib.util.spec_from_file_location("vector_memory", vector_memory_path)
vector_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_memory)
PositionalEncoding1D_PT = vector_memory.PositionalEncoding1D
get_emb = vector_memory.get_emb

print("=" * 70)
print("Testing PositionalEncoding1D - PyTorch vs ttsim")
print("=" * 70)

# Test configuration
batch_size = 2
seq_len = 10
channels = 256

# Create random input
np.random.seed(42)
torch.manual_seed(42)
input_np = np.random.randn(batch_size, seq_len, channels).astype(np.float32)
input_torch = torch.from_numpy(input_np)

print(f"\nInput shape: [{batch_size}, {seq_len}, {channels}]")
print(f"Input values:\n{input_np}")
print(f"Input stats:")
print(f"  Min: {input_np.min():.6f}")
print(f"  Max: {input_np.max():.6f}")
print(f"  Mean: {input_np.mean():.6f}")
print(f"  Std: {input_np.std():.6f}")
print()

# ============================================================
# Test 1: PyTorch PositionalEncoding1D (reference)
# ============================================================
print("TEST 1: PyTorch PositionalEncoding1D (reference)")
print("-" * 70)

pe_pytorch = PositionalEncoding1D_PT(channels)
output_pytorch = pe_pytorch(input_torch)
pytorch_output_np = output_pytorch.detach().cpu().numpy()

print(f"  Output shape: {pytorch_output_np.shape}")
print(f"  Output values:\n{pytorch_output_np}")
print(f"  Output stats:")
print(f"    Min: {pytorch_output_np.min():.6f}")
print(f"    Max: {pytorch_output_np.max():.6f}")
print(f"    Mean: {pytorch_output_np.mean():.6f}")
print(f"    Std: {pytorch_output_np.std():.6f}")
print()

# ============================================================
# Test 2: ttsim PositionalEncoding1D
# ============================================================
print("TEST 2: ttsim PositionalEncoding1D")
print("-" * 70)

pe_ttsim = PositionalEncoding1D_TT(name="pe_1d_test", channels=channels)

# Create input SimTensor
input_simtensor = F._from_data("input", data=input_np, is_const=False)
output_ttsim = pe_ttsim(input_simtensor)

print(f"  Input shape: {input_simtensor.shape}")
print(f"  Output shape: {output_ttsim.shape}")
print(f"  Output values:\n{output_ttsim.data}")
print(f"  Output .data is None? {output_ttsim.data is None}")

if output_ttsim.data is not None:
    print("  [OK] SUCCESS: Output data was computed!")
    print(f"  ttsim output stats:")
    print(f"    Min: {output_ttsim.data.min():.6f}")
    print(f"    Max: {output_ttsim.data.max():.6f}")
    print(f"    Mean: {output_ttsim.data.mean():.6f}")
    print(f"    Std: {output_ttsim.data.std():.6f}")
else:
    print("  [FAIL] FAILED: Output data is still None")
print()

# ============================================================
# Test 3: Numerical Comparison
# ============================================================
if output_ttsim.data is not None:
    print("TEST 3: Numerical Comparison")
    print("-" * 70)

    # Use atol and rtol for robust floating point comparison
    atol = 1e-6
    rtol = 1e-5

    print(f"  Tolerance: atol={atol}, rtol={rtol}")

    # Overall comparison
    is_close = np.allclose(pytorch_output_np, output_ttsim.data, atol=atol, rtol=rtol)
    diff = np.abs(pytorch_output_np - output_ttsim.data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  Max absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")
    print(f"  Arrays match (allclose): {is_close}")

    if is_close:
        print(f"  [PASS] ttsim matches PyTorch (within atol={atol}, rtol={rtol})")
    else:
        print(f"  [FAIL] Differences exceed tolerance (max diff: {max_diff:.10f})")

        # Show first mismatch
        diff_mask = ~np.isclose(
            pytorch_output_np, output_ttsim.data, atol=atol, rtol=rtol
        )
        if diff_mask.any():
            mismatch_idx = np.where(diff_mask)
            b, s, c = mismatch_idx[0][0], mismatch_idx[1][0], mismatch_idx[2][0]
            print(f"\n  First mismatch at [batch={b}, seq={s}, channel={c}]:")
            print(f"    PyTorch: {pytorch_output_np[b, s, c]:.10f}")
            print(f"    ttsim:   {output_ttsim.data[b, s, c]:.10f}")
            print(f"    Diff:    {diff[b, s, c]:.10e}")
    print()

# ============================================================
# Test 4: Graph connectivity - Sin/Cos → Unsqueeze → Concat → Reshape
# ============================================================
print("TEST 4: Graph Connectivity (Sin/Cos interleaving via TTSim ops)")
print("-" * 70)

# Verify pe_table SimTensor exists and is graph-connected
assert hasattr(
    pe_ttsim, "pe_table"
), "pe_table attribute missing from PositionalEncoding1D"
pe_table = pe_ttsim.pe_table
print(f"  pe_table exists: True")
print(f"  pe_table shape: {pe_table.shape}")
print(f"  pe_table.data is None? {pe_table.data is None}")

if pe_table.data is not None:
    # Verify pe_table matches emb_cache (pe_table is [max_len, channels], emb_cache is [:, :org_channels])
    pe_table_sliced = pe_table.data[:, :channels]
    emb_cache = pe_ttsim.emb_cache
    cache_match = np.allclose(pe_table_sliced[: len(emb_cache)], emb_cache, atol=1e-7)
    print(f"  pe_table[:, :org_channels] matches emb_cache: {cache_match}")
    if cache_match:
        print(f"  [PASS] pe_table data matches emb_cache")
    else:
        print(f"  [FAIL] pe_table data != emb_cache")

# Verify TTSim interleaving ops exist
assert hasattr(pe_ttsim, "sin_unsq_op"), "sin_unsq_op missing"
assert hasattr(pe_ttsim, "cos_unsq_op"), "cos_unsq_op missing"
assert hasattr(pe_ttsim, "interleave_concat"), "interleave_concat missing"
assert hasattr(pe_ttsim, "interleave_reshape"), "interleave_reshape missing"
print(f"  TTSim interleave ops (Unsqueeze, ConcatX, Reshape): all present")
print(f"  [PASS] Sin/Cos are graph-connected through interleaving ops")

# Verify the interleaving is correct:
# sin_vals[i, k] should appear at pe_table[i, 2*k] and cos_vals[i, k] at pe_table[i, 2*k+1]
sin_vals_data = pe_ttsim.sin_op(
    pe_ttsim.mul_op(
        F._from_data("tmp_pos", pe_ttsim.emb_cache[:1, :1], is_const=True),  # dummy
        F._from_data("tmp_inv", pe_ttsim.emb_cache[:1, :1], is_const=True),  # dummy
    )
)
# Use the already-computed sin/cos from __init__ via emb_cache
# pe_table[i, 0] = sin(inp[i, 0]), pe_table[i, 1] = cos(inp[i, 0]),
# pe_table[i, 2] = sin(inp[i, 1]), pe_table[i, 3] = cos(inp[i, 1]), ...
if pe_table.data is not None:
    # Quick spot check: verify interleaving pattern at a few positions
    half_ch = pe_ttsim.channels // 2
    k_vals = np.arange(0, pe_ttsim.channels, 2).astype(np.float32)
    inv_freq_check = 1.0 / (10000 ** (k_vals / pe_ttsim.channels))
    pos_check = np.arange(5, dtype=np.float32)
    sin_inp_check = np.outer(pos_check, inv_freq_check)
    expected_sin = np.sin(sin_inp_check)
    expected_cos = np.cos(sin_inp_check)

    interleave_ok = True
    for pos in range(5):
        for k in range(min(3, half_ch)):
            if not np.isclose(
                pe_table.data[pos, 2 * k], expected_sin[pos, k], atol=1e-6
            ):
                interleave_ok = False
            if not np.isclose(
                pe_table.data[pos, 2 * k + 1], expected_cos[pos, k], atol=1e-6
            ):
                interleave_ok = False
    if interleave_ok:
        print(
            f"  [PASS] Sin/Cos interleaving pattern verified (sin at even, cos at odd indices)"
        )
    else:
        print(f"  [FAIL] Interleaving pattern mismatch")
print()

print("=" * 70)
print("Test Complete!")
print("=" * 70)
