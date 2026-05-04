#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for MultiheadAttention TTSim module.
Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_multihead_attention.py

Weight injection notes:
  - TTSim treats input as [..., seq, embed], so use [N, L, E] (batch-first).
  - in_proj_weight stored as [E, 3E] → inject PyTorch [3E, E] transposed.
  - out_proj stored as [E, E] → inject PyTorch [E, E] transposed (TTSim does
    'output @ W_out' without transposing, PyTorch F.linear does x @ W.T).
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.multihead_attention import MultiheadAttention
from reference.Validation.ttsim_utils import print_header, print_test, compare_arrays


# ---------------------------------------------------------------------------
# Helper: inject PyTorch-style weights into a TTSim MultiheadAttention module
# ---------------------------------------------------------------------------
def inject_mha_weights(mha, in_proj_w_pt, in_proj_b, out_proj_w, out_proj_b):
    """
    Args:
        in_proj_w_pt: [3E, E] — same layout as PyTorch in_proj_weight
        in_proj_b:    [3E]
        out_proj_w:   [E, E] — same layout as PyTorch out_proj.weight
        out_proj_b:   [E]
    """
    mha._mha._tensors['in_proj_weight'].data = in_proj_w_pt.T  # → [E, 3E]
    mha._mha._tensors['in_proj_bias'].data   = in_proj_b
    # TTSim MHA does 'output @ W_out' (no transpose) while PyTorch does x @ W.T
    mha._mha._submodules['out_proj']._tensors['param'].data = out_proj_w.T
    mha._mha._submodules['out_proj']._tensors['bias'].data  = out_proj_b


def pt_mha_forward(in_proj_w_pt, in_proj_b, out_proj_w, out_proj_b, x_np, n_heads):
    """Run torch.nn.MultiheadAttention (batch_first=True) with injected weights."""
    E = x_np.shape[-1]
    pt = nn.MultiheadAttention(E, n_heads, bias=True, batch_first=True)
    pt.in_proj_weight.data = torch.tensor(in_proj_w_pt)
    pt.in_proj_bias.data   = torch.tensor(in_proj_b)
    pt.out_proj.weight.data = torch.tensor(out_proj_w)
    pt.out_proj.bias.data   = torch.tensor(out_proj_b)
    pt.eval()
    with torch.no_grad():
        out, _ = pt(torch.tensor(x_np), torch.tensor(x_np), torch.tensor(x_np))
    return out.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construction():
    print_header("Test 1: Construction and param count")
    mha = MultiheadAttention("mha1", embed_dim=128, num_heads=4)
    p = mha.analytical_param_count()
    expected = 4 * 128**2 + 4 * 128   # 4E² + 4E
    ok = p == expected
    print_test("MHA(E=128) param count", f"got {p:,} expected {expected:,}")
    return ok


def test_output_shape_self_attn():
    print_header("Test 2: Self-attention output shape [N, L, E] (batch-first)")
    E, N, L = 64, 2, 20
    mha = MultiheadAttention("mha2", embed_dim=E, num_heads=2)
    q = _from_shape("q2", [N, L, E])
    attn_out, _ = mha(q, q, q)
    expected = [N, L, E]
    ok = list(attn_out.shape) == expected
    print_test("Self-attn output shape", f"got {list(attn_out.shape)} expected {expected}")
    return ok


def test_qkv_projection():
    """Test QKV projection: compare per-head linear step to PyTorch."""
    print_header("Test 3: QKV projection — PyTorch vs TTSim")
    E, N, L = 16, 2, 10
    rng = np.random.RandomState(0)

    in_proj_w = (rng.randn(3*E, E) * 0.1).astype(np.float32)
    in_proj_b = (rng.randn(3*E) * 0.1).astype(np.float32)
    out_proj_w = (rng.randn(E, E) * 0.1).astype(np.float32)
    out_proj_b = (rng.randn(E) * 0.1).astype(np.float32)
    x_np = (rng.randn(N, L, E) * 0.1).astype(np.float32)

    # PyTorch QKV projection (manual)
    pt_x = torch.tensor(x_np.reshape(N * L, E))
    pt_in_proj_w = torch.tensor(in_proj_w)
    pt_in_proj_b = torch.tensor(in_proj_b)
    pt_q_proj = nn.functional.linear(pt_x, pt_in_proj_w[:E],   pt_in_proj_b[:E])
    pt_k_proj = nn.functional.linear(pt_x, pt_in_proj_w[E:2*E], pt_in_proj_b[E:2*E])
    pt_v_proj = nn.functional.linear(pt_x, pt_in_proj_w[2*E:],  pt_in_proj_b[2*E:])
    print(f"  PyTorch Q proj: shape={list(pt_q_proj.shape)}, "
          f"sample={pt_q_proj.numpy().flatten()[:4]}")

    # TTSim manual QKV (using in_proj stored as [E, 3E])
    W_in = in_proj_w.T  # [E, 3E]
    ts_q = x_np.reshape(N * L, E) @ W_in[:, :E]   + in_proj_b[:E]
    ts_k = x_np.reshape(N * L, E) @ W_in[:, E:2*E] + in_proj_b[E:2*E]
    ts_v = x_np.reshape(N * L, E) @ W_in[:, 2*E:]  + in_proj_b[2*E:]
    print(f"  TTSim  Q proj: shape={list(ts_q.shape)}, "
          f"sample={ts_q.flatten()[:4]}")

    ok = compare_arrays(pt_q_proj.numpy(), ts_q, "Q projection")
    ok = compare_arrays(pt_k_proj.numpy(), ts_k, "K projection") and ok
    ok = compare_arrays(pt_v_proj.numpy(), ts_v, "V projection") and ok
    return ok


def test_full_forward_comparison():
    """End-to-end forward pass: TTSim vs PyTorch."""
    print_header("Test 4: Full forward — PyTorch vs TTSim")
    E, N, L = 16, 2, 10
    rng = np.random.RandomState(1)

    in_proj_w = (rng.randn(3*E, E) * 0.1).astype(np.float32)
    in_proj_b = (rng.randn(3*E) * 0.1).astype(np.float32)
    out_proj_w = (rng.randn(E, E) * 0.1).astype(np.float32)
    out_proj_b = (rng.randn(E) * 0.1).astype(np.float32)
    x_np = (rng.randn(N, L, E) * 0.1).astype(np.float32)

    # PyTorch reference (batch_first=True)
    pt_out = pt_mha_forward(in_proj_w, in_proj_b, out_proj_w, out_proj_b, x_np, n_heads=2)
    print(f"  PyTorch output: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim
    mha = MultiheadAttention("mha4", embed_dim=E, num_heads=2)
    inject_mha_weights(mha, in_proj_w, in_proj_b, out_proj_w, out_proj_b)
    x_t = _from_data("x4", x_np)
    ts_out, _ = mha(x_t, x_t, x_t)
    print(f"  TTSim  output: shape={list(ts_out.shape)}, "
          f"sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "MHA full forward")
    return ok


def test_cross_attention():
    """Cross-attention: query and key/value from different sources."""
    print_header("Test 5: Cross-attention — PyTorch vs TTSim")
    E, N, L, S = 16, 2, 6, 10
    rng = np.random.RandomState(2)

    in_proj_w = (rng.randn(3*E, E) * 0.1).astype(np.float32)
    in_proj_b = (rng.randn(3*E) * 0.1).astype(np.float32)
    out_proj_w = (rng.randn(E, E) * 0.1).astype(np.float32)
    out_proj_b = (rng.randn(E) * 0.1).astype(np.float32)
    q_np = (rng.randn(N, L, E) * 0.1).astype(np.float32)
    kv_np = (rng.randn(N, S, E) * 0.1).astype(np.float32)

    # PyTorch cross-attention (batch_first=True)
    pt = nn.MultiheadAttention(E, 2, bias=True, batch_first=True)
    pt.in_proj_weight.data = torch.tensor(in_proj_w)
    pt.in_proj_bias.data   = torch.tensor(in_proj_b)
    pt.out_proj.weight.data = torch.tensor(out_proj_w)
    pt.out_proj.bias.data   = torch.tensor(out_proj_b)
    pt.eval()
    with torch.no_grad():
        pt_out, _ = pt(torch.tensor(q_np), torch.tensor(kv_np), torch.tensor(kv_np))
    pt_out = pt_out.numpy()
    print(f"  PyTorch output: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim
    mha = MultiheadAttention("mha5", embed_dim=E, num_heads=2)
    inject_mha_weights(mha, in_proj_w, in_proj_b, out_proj_w, out_proj_b)
    q_t  = _from_data("q5",  q_np)
    kv_t = _from_data("kv5", kv_np)
    ts_out, _ = mha(q_t, kv_t, kv_t)
    print(f"  TTSim  output: shape={list(ts_out.shape)}, "
          f"sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "Cross-attn forward")
    return ok


def test_param_count_e256():
    print_header("Test 6: Param count for E=256, 8 heads")
    E = 256
    mha = MultiheadAttention("mha6", embed_dim=E, num_heads=8)
    expected = 4 * E**2 + 4 * E
    got = mha.analytical_param_count()
    ok = got == expected
    print_test(f"MHA(E={E}) param count", f"got {got:,} expected {expected:,}")
    return ok


def test_batch_size_independence():
    print_header("Test 7: Various batch sizes — shape check")
    E, L = 64, 50
    mha = MultiheadAttention("mha7", embed_dim=E, num_heads=2)
    ok = True
    for N in [1, 4, 8]:
        q = _from_shape(f"qb{N}", [N, L, E])
        attn_out, _ = mha(q, q, q)
        match = list(attn_out.shape) == [N, L, E]
        print_test(f"N={N}", f"got {list(attn_out.shape)}")
        if not match: ok = False
    return ok


if __name__ == "__main__":
    tests = [
        ("construction",            test_construction),
        ("output_shape_self_attn",  test_output_shape_self_attn),
        ("qkv_projection",          test_qkv_projection),
        ("full_forward_comparison", test_full_forward_comparison),
        ("cross_attention",         test_cross_attention),
        ("param_count_e256",        test_param_count_e256),
        ("batch_size_independence", test_batch_size_independence),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
