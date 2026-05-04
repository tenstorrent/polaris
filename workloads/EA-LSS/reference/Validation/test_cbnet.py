#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for CBSwinTransformer TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_cbnet.py

Numerical comparisons validate the patch embedding Conv2d and FFN linear
steps (key CBSwin ops) against PyTorch.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.cbnet import CBSwinTransformer
from Reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays, ttsim_conv2d,
)

_SWIN_KWARGS = dict(
    pretrain_img_size=224,
    patch_size=4,
    in_chans=3,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    out_indices=(0, 1, 2, 3),
    patch_norm=True,
)


# ---------------------------------------------------------------------------
# Numerical comparison tests
# ---------------------------------------------------------------------------

def test_patch_embed_conv_step():
    """Patch embedding Conv2d (patch_size=4, in→embed) — PyTorch vs ttsim_conv2d."""
    print_header("Test 1: Patch Embedding Conv2d step — PyTorch vs ttsim_conv2d")
    rng = np.random.RandomState(100)
    B, in_chans, H, W = 1, 3, 8, 8
    embed_dim, patch_size = 96, 4

    x_np = (rng.randn(B, in_chans, H, W) * 0.5).astype(np.float32)
    w_np = (rng.randn(embed_dim, in_chans, patch_size, patch_size) * 0.1).astype(np.float32)
    b_np = (rng.randn(embed_dim) * 0.1).astype(np.float32)

    # PyTorch: Conv2d with stride=patch_size → [B, embed_dim, H/4, W/4]
    conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
    conv.weight.data = torch.tensor(w_np)
    conv.bias.data   = torch.tensor(b_np)
    with torch.no_grad():
        pt_out = conv(torch.tensor(x_np)).numpy()
    print(f"  PyTorch PatchEmbed: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = ttsim_conv2d(x_np, w_np, bias=b_np, stride=patch_size, padding=0)
    print(f"  TTSim   PatchEmbed: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "patch embed conv step")
    return ok


def test_ffn_linear_step():
    """FFN linear projection (key Swin MLP op) — PyTorch vs NumPy."""
    print_header("Test 2: Swin FFN linear step — PyTorch vs NumPy")
    rng = np.random.RandomState(101)
    L, C, ffn_ratio = 49, 96, 4   # 7×7 window, embed=96, mlp_ratio=4

    x_np = (rng.randn(L, C) * 0.1).astype(np.float32)
    # FFN expands: C → C*ffn_ratio
    w_pt = (rng.randn(C * ffn_ratio, C) * 0.1).astype(np.float32)
    b    = (rng.randn(C * ffn_ratio) * 0.1).astype(np.float32)

    lin = nn.Linear(C, C * ffn_ratio)
    lin.weight.data = torch.tensor(w_pt)
    lin.bias.data   = torch.tensor(b)
    with torch.no_grad():
        pt_out = lin(torch.tensor(x_np)).numpy()
    print(f"  PyTorch FFN:  shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = x_np @ w_pt.T + b
    print(f"  NumPy   FFN:  shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "Swin FFN linear step")
    return ok


# ---------------------------------------------------------------------------
# Shape / param tests
# ---------------------------------------------------------------------------

def test_cbswin_construction():
    print_header("Test 3: CBSwinTransformer construction")
    cb = CBSwinTransformer("cb_con", embed_dim=96, cb_del_stages=1, **_SWIN_KWARGS)
    p  = cb.analytical_param_count()
    ok = p == 55_682_580
    print_test(f"CBSwinTransformer params == 55,682,580", f"got {p:,}", ok)
    return ok


def test_cbswin_output_shapes():
    print_header("Test 4: CBSwinTransformer output shapes")
    cb   = CBSwinTransformer("cb_shp", embed_dim=96, cb_del_stages=1, **_SWIN_KWARGS)
    x    = _from_shape("cb_x", [1, 3, 128, 128])
    outs = cb(x)
    expected_shapes = [[1, 96, 32, 32], [1, 192, 16, 16], [1, 384, 8, 8], [1, 768, 4, 4]]
    all_ok = True
    for i, (out, exp) in enumerate(zip(outs, expected_shapes)):
        ok = list(out.shape) == exp
        print_test(f"out[{i}] shape", f"got {out.shape} expected {exp}", ok)
        if not ok:
            all_ok = False
    return all_ok


def test_cbswin_num_outputs():
    print_header("Test 5: CBSwinTransformer produces 4 feature maps")
    cb  = CBSwinTransformer("cb_nout", embed_dim=96, cb_del_stages=1, **_SWIN_KWARGS)
    x   = _from_shape("cb_n_x", [1, 3, 64, 64])
    outs = cb(x)
    ok  = len(outs) == 4
    print_test(f"len(outs) == 4", f"got {len(outs)}", ok)
    return ok


if __name__ == "__main__":
    tests = [
        ("patch_embed_conv",  test_patch_embed_conv_step),
        ("ffn_linear_step",   test_ffn_linear_step),
        ("cbswin_construct",  test_cbswin_construction),
        ("cbswin_shapes",     test_cbswin_output_shapes),
        ("cbswin_num_outs",   test_cbswin_num_outputs),
    ]
    results = {}
    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            results[name] = False

    print("\n" + "="*60)
    passed = sum(results.values())
    total  = len(results)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
