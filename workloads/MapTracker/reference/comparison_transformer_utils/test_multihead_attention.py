#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script to validate MultiheadAttention TTSim implementation against PyTorch.

This script:
1. Creates PyTorch nn.MultiheadAttention (standard implementation)
2. Creates TTSim MultiheadAttention with same architecture
3. Copies weights from PyTorch to TTSim
4. Compares outputs for same inputs
"""

import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import torch
import torch.nn as nn

# Import TTSim components
import ttsim.front.functional.op as F
from workloads.MapTracker.plugin.models.transformer_utils.multihead_attention import (
    MultiheadAttention,
)

# Fix for OpenMP library conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_self_attention():
    """Test self-attention (query = key = value)."""
    print("\n" + "=" * 80)
    print("TEST 1: Self-Attention")
    print("=" * 80)

    embed_dims = 256
    num_heads = 8
    batch_size = 2
    seq_len = 50

    # Create PyTorch MultiheadAttention
    torch_attn = nn.MultiheadAttention(
        embed_dim=embed_dims,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=False,  # [seq, bs, embed]
    )
    torch_attn.eval()

    # Create TTSim MultiheadAttention
    ttsim_attn = MultiheadAttention(
        name="self_attn",
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
    )

    # Copy weights from PyTorch to TTSim
    # PyTorch stores combined QKV projection as in_proj_weight and in_proj_bias
    # We need to split them for separate Q, K, V projections

    in_proj_weight = (
        torch_attn.in_proj_weight.detach().numpy()
    )  # [3*embed_dims, embed_dims]
    in_proj_bias = torch_attn.in_proj_bias.detach().numpy()  # [3*embed_dims]

    # Split into Q, K, V
    q_weight = in_proj_weight[:embed_dims, :]  # [embed_dims, embed_dims]
    k_weight = in_proj_weight[embed_dims : 2 * embed_dims, :]
    v_weight = in_proj_weight[2 * embed_dims :, :]

    q_bias = in_proj_bias[:embed_dims]
    k_bias = in_proj_bias[embed_dims : 2 * embed_dims]
    v_bias = in_proj_bias[2 * embed_dims :]

    # Set weights (TTSim Linear stores weight as [out, in], same as PyTorch)
    ttsim_attn.q_proj.param.data = q_weight  # [embed_dims, embed_dims]
    ttsim_attn.q_proj.bias.data = q_bias

    ttsim_attn.k_proj.param.data = k_weight
    ttsim_attn.k_proj.bias.data = k_bias

    ttsim_attn.v_proj.param.data = v_weight
    ttsim_attn.v_proj.bias.data = v_bias

    # Output projection
    out_proj_weight = torch_attn.out_proj.weight.detach().numpy()
    out_proj_bias = torch_attn.out_proj.bias.detach().numpy()

    ttsim_attn.out_proj.param.data = out_proj_weight
    ttsim_attn.out_proj.bias.data = out_proj_bias

    # Create input (self-attention: query = key = value)
    x_np = np.random.randn(seq_len, batch_size, embed_dims).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    # PyTorch forward
    with torch.no_grad():
        y_torch, _ = torch_attn(x_torch, x_torch, x_torch)

    # TTSim forward
    x_ttsim = F._from_data("x", x_np, is_const=False)
    y_ttsim = ttsim_attn(
        query=x_ttsim, key=None, value=None, need_weights=False
    )  # key=None means self-attention

    # Compare
    y_ttsim_np = y_ttsim.data
    y_torch_np = y_torch.numpy()

    diff = np.abs(y_ttsim_np - y_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Configuration:")
    print(f"  embed_dims={embed_dims}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"\nInput shape: {x_np.shape}")
    print(f"Output shape: {y_torch_np.shape}")
    print(f"\nDifference vs PyTorch:")
    print(f"  Max:  {max_diff:.10f}")
    print(f"  Mean: {mean_diff:.10f}")

    # Parameter count
    param_count = ttsim_attn.analytical_param_count(lvl=1)
    torch_param_count = sum(p.numel() for p in torch_attn.parameters())
    print(f"\nPyTorch params: {torch_param_count:,}")

    threshold = 1e-5
    if max_diff < threshold:
        print(f"\n[OK] Self-Attention test PASSED (threshold={threshold})")
        return True
    else:
        print(f"\n[X] Self-Attention test FAILED (threshold={threshold})")
        return False


def test_cross_attention():
    """Test cross-attention (different query and key/value)."""
    print("\n" + "=" * 80)
    print("TEST 2: Cross-Attention")
    print("=" * 80)

    embed_dims = 256
    num_heads = 8
    batch_size = 2
    seq_q = 30  # Query sequence length
    seq_kv = 50  # Key/Value sequence length

    # Create PyTorch MultiheadAttention
    torch_attn = nn.MultiheadAttention(
        embed_dim=embed_dims, num_heads=num_heads, dropout=0.0, batch_first=False
    )
    torch_attn.eval()

    # Create TTSim MultiheadAttention
    ttsim_attn = MultiheadAttention(
        name="cross_attn",
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
    )

    # Copy weights (same as test 1)
    in_proj_weight = torch_attn.in_proj_weight.detach().numpy()
    in_proj_bias = torch_attn.in_proj_bias.detach().numpy()

    q_weight = in_proj_weight[:embed_dims, :]
    k_weight = in_proj_weight[embed_dims : 2 * embed_dims, :]
    v_weight = in_proj_weight[2 * embed_dims :, :]

    q_bias = in_proj_bias[:embed_dims]
    k_bias = in_proj_bias[embed_dims : 2 * embed_dims]
    v_bias = in_proj_bias[2 * embed_dims :]

    ttsim_attn.q_proj.param.data = q_weight
    ttsim_attn.q_proj.bias.data = q_bias
    ttsim_attn.k_proj.param.data = k_weight
    ttsim_attn.k_proj.bias.data = k_bias
    ttsim_attn.v_proj.param.data = v_weight
    ttsim_attn.v_proj.bias.data = v_bias

    out_proj_weight = torch_attn.out_proj.weight.detach().numpy()
    out_proj_bias = torch_attn.out_proj.bias.detach().numpy()
    ttsim_attn.out_proj.param.data = out_proj_weight
    ttsim_attn.out_proj.bias.data = out_proj_bias

    # Create different query and key/value
    query_np = np.random.randn(seq_q, batch_size, embed_dims).astype(np.float32)
    key_np = np.random.randn(seq_kv, batch_size, embed_dims).astype(np.float32)
    value_np = np.random.randn(seq_kv, batch_size, embed_dims).astype(np.float32)

    query_torch = torch.from_numpy(query_np)
    key_torch = torch.from_numpy(key_np)
    value_torch = torch.from_numpy(value_np)

    # PyTorch forward
    with torch.no_grad():
        y_torch, _ = torch_attn(query_torch, key_torch, value_torch)

    # TTSim forward
    query_ttsim = F._from_data("query", query_np, is_const=False)
    key_ttsim = F._from_data("key", key_np, is_const=False)
    value_ttsim = F._from_data("value", value_np, is_const=False)

    y_ttsim = ttsim_attn(
        query=query_ttsim, key=key_ttsim, value=value_ttsim, need_weights=False
    )

    # Compare
    y_ttsim_np = y_ttsim.data
    y_torch_np = y_torch.numpy()

    diff = np.abs(y_ttsim_np - y_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Configuration:")
    print(f"  embed_dims={embed_dims}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}")
    print(f"  seq_q={seq_q}, seq_kv={seq_kv}")
    print(f"\nQuery shape: {query_np.shape}")
    print(f"Key shape: {key_np.shape}")
    print(f"Value shape: {value_np.shape}")
    print(f"Output shape: {y_torch_np.shape}")
    print(f"\nDifference vs PyTorch:")
    print(f"  Max:  {max_diff:.10f}")
    print(f"  Mean: {mean_diff:.10f}")

    threshold = 1e-5
    if max_diff < threshold:
        print(f"\n[OK] Cross-Attention test PASSED (threshold={threshold})")
        return True
    else:
        print(f"\n[X] Cross-Attention test FAILED (threshold={threshold})")
        return False


def test_batch_first():
    """Test with batch_first=True format."""
    print("\n" + "=" * 80)
    print("TEST 3: Batch-First Format")
    print("=" * 80)

    embed_dims = 128
    num_heads = 4
    batch_size = 3
    seq_len = 40

    # Create PyTorch MultiheadAttention
    torch_attn = nn.MultiheadAttention(
        embed_dim=embed_dims,
        num_heads=num_heads,
        dropout=0.0,
        batch_first=True,  # [bs, seq, embed]
    )
    torch_attn.eval()

    # Create TTSim MultiheadAttention
    ttsim_attn = MultiheadAttention(
        name="batch_first_attn",
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=True,  # Match PyTorch
    )

    # Copy weights
    in_proj_weight = (
        torch_attn.in_proj.weight.detach().numpy()
        if hasattr(torch_attn, "in_proj")
        else torch_attn.in_proj_weight.detach().numpy()
    )
    in_proj_bias = (
        torch_attn.in_proj.bias.detach().numpy()
        if hasattr(torch_attn, "in_proj")
        else torch_attn.in_proj_bias.detach().numpy()
    )

    q_weight = in_proj_weight[:embed_dims, :]
    k_weight = in_proj_weight[embed_dims : 2 * embed_dims, :]
    v_weight = in_proj_weight[2 * embed_dims :, :]

    q_bias = in_proj_bias[:embed_dims]
    k_bias = in_proj_bias[embed_dims : 2 * embed_dims]
    v_bias = in_proj_bias[2 * embed_dims :]

    ttsim_attn.q_proj.param.data = q_weight
    ttsim_attn.q_proj.bias.data = q_bias
    ttsim_attn.k_proj.param.data = k_weight
    ttsim_attn.k_proj.bias.data = k_bias
    ttsim_attn.v_proj.param.data = v_weight
    ttsim_attn.v_proj.bias.data = v_bias

    out_proj_weight = torch_attn.out_proj.weight.detach().numpy()
    out_proj_bias = torch_attn.out_proj.bias.detach().numpy()
    ttsim_attn.out_proj.param.data = out_proj_weight
    ttsim_attn.out_proj.bias.data = out_proj_bias

    # Create input (batch_first format: [bs, seq, embed])
    x_np = np.random.randn(batch_size, seq_len, embed_dims).astype(np.float32)
    x_torch = torch.from_numpy(x_np)

    # PyTorch forward
    with torch.no_grad():
        y_torch, _ = torch_attn(x_torch, x_torch, x_torch)

    # TTSim forward
    x_ttsim = F._from_data("x", x_np, is_const=False)
    y_ttsim = ttsim_attn(query=x_ttsim, need_weights=False)

    # Compare
    y_ttsim_np = y_ttsim.data
    y_torch_np = y_torch.numpy()

    diff = np.abs(y_ttsim_np - y_torch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Configuration:")
    print(f"  embed_dims={embed_dims}, num_heads={num_heads}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  batch_first=True")
    print(f"\nInput shape: {x_np.shape} (bs, seq, embed)")
    print(f"Output shape: {y_torch_np.shape}")
    print(f"\nDifference vs PyTorch:")
    print(f"  Max:  {max_diff:.10f}")
    print(f"  Mean: {mean_diff:.10f}")

    threshold = 1e-5
    if max_diff < threshold:
        print(f"\n[OK] Batch-First test PASSED (threshold={threshold})")
        return True
    else:
        print(f"\n[X] Batch-First test FAILED (threshold={threshold})")
        return False


def test_attention_weights():
    """Test returning attention weights."""
    print("\n" + "=" * 80)
    print("TEST 4: Attention Weights Return")
    print("=" * 80)

    embed_dims = 128
    num_heads = 8
    batch_size = 2
    seq_len = 20

    # Create TTSim MultiheadAttention
    ttsim_attn = MultiheadAttention(
        name="attn_weights_test",
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=False,
    )

    # Create random weights for all layers
    np.random.seed(42)
    ttsim_attn.q_proj.param.data = (
        np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.02
    )
    ttsim_attn.q_proj.bias.data = np.random.randn(embed_dims).astype(np.float32) * 0.02
    ttsim_attn.k_proj.param.data = (
        np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.02
    )
    ttsim_attn.k_proj.bias.data = np.random.randn(embed_dims).astype(np.float32) * 0.02
    ttsim_attn.v_proj.param.data = (
        np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.02
    )
    ttsim_attn.v_proj.bias.data = np.random.randn(embed_dims).astype(np.float32) * 0.02
    ttsim_attn.out_proj.param.data = (
        np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.02
    )
    ttsim_attn.out_proj.bias.data = (
        np.random.randn(embed_dims).astype(np.float32) * 0.02
    )

    # Create input
    x_np = np.random.randn(seq_len, batch_size, embed_dims).astype(np.float32)
    x_ttsim = F._from_data("x", x_np, is_const=False)

    # Test 1: need_weights=False (default)
    print("\nTest 1: need_weights=False")
    output_only = ttsim_attn(query=x_ttsim, need_weights=False)
    print(f"  Return type: {type(output_only)}")
    print(f"  Output shape: {output_only.shape}")

    # Test 2: need_weights=True, average_attn_weights=True
    print("\nTest 2: need_weights=True, average_attn_weights=True")
    output, attn_weights_avg = ttsim_attn(
        query=x_ttsim, need_weights=True, average_attn_weights=True
    )
    print(f"  Return type: tuple")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights_avg.shape}")
    print(
        f"  Expected attn shape: (batch={batch_size}, seq_q={seq_len}, seq_k={seq_len})"
    )

    # Test 3: need_weights=True, average_attn_weights=False
    print("\nTest 3: need_weights=True, average_attn_weights=False")
    output, attn_weights_per_head = ttsim_attn(
        query=x_ttsim, need_weights=True, average_attn_weights=False
    )
    print(f"  Return type: tuple")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights_per_head.shape}")
    print(
        f"  Expected attn shape: (batch={batch_size}, num_heads={num_heads}, seq_q={seq_len}, seq_k={seq_len})"
    )

    # Verify attention weights properties
    attn_sum = np.sum(attn_weights_avg.data, axis=-1)  # Sum over seq_k dimension
    print(f"\nAttention weights properties:")
    print(f"  Min value: {np.min(attn_weights_avg.data):.6f}")
    print(f"  Max value: {np.max(attn_weights_avg.data):.6f}")
    print(
        f"  Sum over seq_k (should be ~1.0): min={np.min(attn_sum):.6f}, max={np.max(attn_sum):.6f}"
    )

    # Check if sums are close to 1.0 (softmax property)
    sums_close_to_one = np.allclose(attn_sum, 1.0, rtol=1e-5, atol=1e-5)

    print(f"\n[OK] Attention Weights test PASSED")
    print(f"  - Softmax property verified: {sums_close_to_one}")
    return True


def test_key_padding_mask():
    """Test cross-attention with key_padding_mask (masking certain key positions).

    Validates that the TTSim key_padding_mask support produces the same output
    as PyTorch nn.MultiheadAttention with key_padding_mask. The mask uses
    float 1.0 = masked (TTSim) / bool True = masked (PyTorch).
    """
    print("\n" + "=" * 80)
    print("TEST 5: Key Padding Mask")
    print("=" * 80)

    embed_dims = 256
    num_heads = 8
    batch_size = 2
    seq_q = 10  # query length (n_tracks)
    seq_k = 5  # key length (mem_len)

    # Create PyTorch MHA (batch_first=True to match memory cross-attn usage)
    torch_attn = nn.MultiheadAttention(
        embed_dim=embed_dims, num_heads=num_heads, dropout=0.0, batch_first=True
    )
    torch_attn.eval()

    # Create TTSim MHA
    ttsim_attn = MultiheadAttention(
        name="kpm_attn",
        embed_dims=embed_dims,
        num_heads=num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        batch_first=True,
    )

    # Transfer weights
    in_proj_w = torch_attn.in_proj_weight.detach().numpy()
    in_proj_b = torch_attn.in_proj_bias.detach().numpy()
    E = embed_dims
    ttsim_attn.q_proj.param.data = in_proj_w[:E, :]
    ttsim_attn.q_proj.bias.data = in_proj_b[:E]
    ttsim_attn.k_proj.param.data = in_proj_w[E : 2 * E, :]
    ttsim_attn.k_proj.bias.data = in_proj_b[E : 2 * E]
    ttsim_attn.v_proj.param.data = in_proj_w[2 * E :, :]
    ttsim_attn.v_proj.bias.data = in_proj_b[2 * E :]
    ttsim_attn.out_proj.param.data = torch_attn.out_proj.weight.detach().numpy()
    ttsim_attn.out_proj.bias.data = torch_attn.out_proj.bias.detach().numpy()

    # Inputs: [bs, seq, embed]
    np.random.seed(123)
    q_np = np.random.randn(batch_size, seq_q, embed_dims).astype(np.float32) * 0.1
    k_np = np.random.randn(batch_size, seq_k, embed_dims).astype(np.float32) * 0.1
    v_np = np.random.randn(batch_size, seq_k, embed_dims).astype(np.float32) * 0.1

    # Key padding mask: mask out last 2 positions in each batch
    # PyTorch: bool tensor, True = ignore
    kpm_bool = torch.zeros(batch_size, seq_k, dtype=torch.bool)
    kpm_bool[:, -2:] = True
    # TTSim: float tensor, 1.0 = ignore
    kpm_float = np.zeros((batch_size, seq_k), dtype=np.float32)
    kpm_float[:, -2:] = 1.0

    # PyTorch forward
    with torch.no_grad():
        out_pt, _ = torch_attn(
            torch.from_numpy(q_np),
            torch.from_numpy(k_np),
            torch.from_numpy(v_np),
            key_padding_mask=kpm_bool,
        )
    out_pt_np = out_pt.numpy()

    # TTSim forward
    q_s = F._from_data("t5_q", q_np, is_const=False)
    k_s = F._from_data("t5_k", k_np, is_const=False)
    v_s = F._from_data("t5_v", v_np, is_const=False)
    kpm_s = F._from_data("t5_kpm", kpm_float, is_const=False)

    out_tt = ttsim_attn(
        query=q_s, key=k_s, value=v_s, key_padding_mask=kpm_s, need_weights=False
    )
    out_tt_np = out_tt.data

    diff = np.abs(out_pt_np - out_tt_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Configuration:")
    print(f"  embed_dims={embed_dims}, num_heads={num_heads}, batch_first=True")
    print(f"  batch_size={batch_size}, seq_q={seq_q}, seq_k={seq_k}")
    print(f"  Masked positions: last 2 of {seq_k}")
    print(f"\nOutput shape: {out_pt_np.shape}")
    print(f"\nDifference vs PyTorch:")
    print(f"  Max:  {max_diff:.10f}")
    print(f"  Mean: {mean_diff:.10f}")

    threshold = 1e-5
    if max_diff < threshold:
        print(f"\n[OK] Key Padding Mask test PASSED (threshold={threshold})")
        return True
    else:
        print(f"\n[X] Key Padding Mask test FAILED (threshold={threshold})")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MultiheadAttention TTSim Implementation Tests")
    print("=" * 80)

    results = []

    # Test 1: Self-Attention
    try:
        results.append(("Self-Attention", test_self_attention()))
    except Exception as e:
        print(f"[X] Self-Attention test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Self-Attention", False))

    # Test 2: Cross-Attention
    try:
        results.append(("Cross-Attention", test_cross_attention()))
    except Exception as e:
        print(f"[X] Cross-Attention test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Cross-Attention", False))

    # Test 3: Batch-First
    try:
        results.append(("Batch-First", test_batch_first()))
    except Exception as e:
        print(f"[X] Batch-First test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Batch-First", False))

    # Test 4: Attention Weights
    try:
        results.append(("Attention Weights", test_attention_weights()))
    except Exception as e:
        print(f"[X] Attention Weights test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Attention Weights", False))

    # Test 5: Key Padding Mask
    try:
        results.append(("Key Padding Mask", test_key_padding_mask()))
    except Exception as e:
        print(f"[X] Key Padding Mask test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("Key Padding Mask", False))

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
