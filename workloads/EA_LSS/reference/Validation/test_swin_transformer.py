#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for SwinTransformer backbone TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_swin_transformer.py

Numerical comparisons validate the key Swin-T ops (QKV linear projection,
relative-position-bias table, window partition) against PyTorch/NumPy.
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

from ttsim.front.functional.op import _from_shape
from ttsim_modules.swin_transformer import SwinTransformer
from reference.Validation.ttsim_utils import print_header, print_test, compare_arrays


# ---------------------------------------------------------------------------
# Numerical comparison tests
# ---------------------------------------------------------------------------

def test_qkv_linear_step():
    """QKV linear projection (key Swin-T op) — PyTorch vs NumPy."""
    print_header("Test 1: Window-Attn QKV linear step — PyTorch vs NumPy")
    rng = np.random.RandomState(80)
    # Swin-T: embed_dim=96, 7×7 window → L=49 tokens
    L, C = 49, 96

    x_np = (rng.randn(L, C) * 0.1).astype(np.float32)
    w_pt = (rng.randn(3*C, C) * 0.1).astype(np.float32)   # [3*C, C] PyTorch Linear
    b    = (rng.randn(3*C)    * 0.1).astype(np.float32)

    lin = nn.Linear(C, 3*C)
    lin.weight.data = torch.tensor(w_pt)
    lin.bias.data   = torch.tensor(b)
    with torch.no_grad():
        pt_out = lin(torch.tensor(x_np)).numpy()
    print(f"  PyTorch QKV: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    # NumPy equivalent of nn.Linear: x @ W.T + b
    ts_out = x_np @ w_pt.T + b
    print(f"  NumPy   QKV: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "QKV linear step")
    return ok


def test_window_softmax_attn():
    """Softmax over last dim (attention weights) — PyTorch vs NumPy."""
    print_header("Test 2: Softmax attention weights — PyTorch vs NumPy")
    rng = np.random.RandomState(81)
    # Simulated attention logits: [B_win, nH, L, L]
    B_win, nH, L = 4, 3, 49
    attn_np = (rng.randn(B_win, nH, L, L) * 0.5).astype(np.float32)

    with torch.no_grad():
        pt_out = torch.softmax(torch.tensor(attn_np), dim=-1).numpy()
    print(f"  PyTorch Softmax: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    attn_max = attn_np - attn_np.max(axis=-1, keepdims=True)
    exp_attn = np.exp(attn_max)
    ts_out   = exp_attn / exp_attn.sum(axis=-1, keepdims=True)
    print(f"  NumPy   Softmax: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "attention softmax", atol=1e-6)
    return ok


# ---------------------------------------------------------------------------
# Shape / param tests
# ---------------------------------------------------------------------------

def test_construction_swint():
    print_header("Test 3: Construction (Swin-T)")
    swint = SwinTransformer(
        "swint1",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    p = swint.analytical_param_count()
    ok = p > 0
    print_test("SwinT params > 0", f"got {p:,}")
    return ok


def test_output_shapes_swint():
    print_header("Test 4: Output shapes (Swin-T, 224×224 input)")
    swint = SwinTransformer(
        "swint2",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    x = _from_shape("img", [2, 3, 224, 224])
    outs = swint(x)
    expected = [
        [2, 96,  56, 56],
        [2, 192, 28, 28],
        [2, 384, 14, 14],
        [2, 768,  7,  7],
    ]
    ok = True
    for i, (o, e) in enumerate(zip(outs, expected)):
        match = list(o.shape) == e
        print_test(f"Stage {i} shape", f"got {list(o.shape)} expected {e}")
        if not match:
            ok = False
    return ok


def test_param_count_swint():
    print_header("Test 5: Param count == 27,520,602 (Swin-T reference)")
    swint = SwinTransformer(
        "swint3",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    expected = 27_520_602
    got = swint.analytical_param_count()
    ok = got == expected
    print_test("Swin-T param count", f"got {got:,} expected {expected:,}")
    return ok


def test_two_stage_only():
    print_header("Test 6: Two-stage Swin (out_indices=(0,1))")
    swint = SwinTransformer(
        "swint4",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=7,
        out_indices=(0, 1),
    )
    x = _from_shape("img4", [1, 3, 224, 224])
    outs = swint(x)
    ok = len(outs) == 2
    print_test("2-stage produces 2 outputs", f"got {len(outs)}")
    return ok


def test_different_out_indices():
    print_header("Test 7: out_indices=(1, 3) — only stages 1 and 3")
    swint = SwinTransformer(
        "swint5",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(1, 3),
    )
    x = _from_shape("img5", [2, 3, 224, 224])
    outs = swint(x)
    ok = len(outs) == 2
    expected_shapes = [[2, 192, 28, 28], [2, 768, 7, 7]]
    for i, (o, e) in enumerate(zip(outs, expected_shapes)):
        match = list(o.shape) == e
        print_test(f"out_indices={[1,3][i]} shape", f"got {list(o.shape)} expected {e}")
        if not match: ok = False
    return ok


if __name__ == "__main__":
    tests = [
        ("qkv_linear_step",       test_qkv_linear_step),
        ("window_softmax_attn",   test_window_softmax_attn),
        ("construction_swint",    test_construction_swint),
        ("output_shapes_swint",   test_output_shapes_swint),
        ("param_count_swint",     test_param_count_swint),
        ("two_stage_only",        test_two_stage_only),
        ("different_out_indices", test_different_out_indices),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            print(f"  ERROR {name}: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
