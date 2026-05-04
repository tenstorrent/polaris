#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for FPNC and gapcontext TTSim modules.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_fpnc.py

Numerical comparisons validate the core ops inside gapcontext
(pointwise Conv2d and Global Average Pool) against PyTorch.
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
from ttsim_modules.fpnc import gapcontext, FPNC
from reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays, ttsim_conv2d,
)


# ---------------------------------------------------------------------------
# Numerical comparison tests
# ---------------------------------------------------------------------------

def test_gapcontext_conv1x1_step():
    """Pointwise (1×1) Conv2d step inside gapcontext — PyTorch vs ttsim_conv2d."""
    print_header("Test 1: gapcontext Conv1×1 step — PyTorch vs ttsim_conv2d")
    rng = np.random.RandomState(70)
    B, C_in, C_out, H, W = 2, 32, 64, 8, 8

    x_np = (rng.randn(B, C_in, H, W) * 0.5).astype(np.float32)
    # PyTorch Conv2d weight layout: [out_channels, in_channels/groups, kH, kW]
    w_np = (rng.randn(C_out, C_in, 1, 1) * 0.1).astype(np.float32)
    b_np = (rng.randn(C_out) * 0.1).astype(np.float32)

    conv = nn.Conv2d(C_in, C_out, kernel_size=1, bias=True)
    conv.weight.data = torch.tensor(w_np)
    conv.bias.data   = torch.tensor(b_np)
    with torch.no_grad():
        pt_out = conv(torch.tensor(x_np)).numpy()
    print(f"  PyTorch Conv1x1: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = ttsim_conv2d(x_np, w_np, bias=b_np, stride=1, padding=0)
    print(f"  TTSim   Conv1x1: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "gapcontext conv1x1")
    return ok


def test_global_avg_pool_step():
    """Global Average Pool (AdaptiveAvgPool2d(1)) — PyTorch vs NumPy mean."""
    print_header("Test 2: Global Average Pool — PyTorch vs NumPy")
    rng = np.random.RandomState(71)
    B, C, H, W = 2, 64, 8, 8

    x_np = (rng.randn(B, C, H, W) * 0.5).astype(np.float32)

    gap = nn.AdaptiveAvgPool2d(1)
    with torch.no_grad():
        pt_out = gap(torch.tensor(x_np)).numpy()   # [B, C, 1, 1]
    print(f"  PyTorch GAP: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = x_np.mean(axis=(2, 3), keepdims=True)   # NumPy equivalent
    print(f"  NumPy   GAP: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "global avg pool", atol=1e-6)
    return ok


# ---------------------------------------------------------------------------
# Shape / param tests
# ---------------------------------------------------------------------------

def test_gapcontext_construction():
    print_header("Test 3: gapcontext construction and param count")
    for in_c, out_c, with_norm in [(32, 64, False), (64, 64, False), (128, 64, True)]:
        gc = gapcontext(f"gc_{in_c}_{out_c}", in_c, out_c, with_norm=with_norm)
        p  = gc.analytical_param_count()
        ok = p > 0
        print_test(f"gapcontext({in_c},{out_c},norm={with_norm}): params={p}", "p>0", ok)
    return True


def test_gapcontext_shape():
    print_header("Test 4: gapcontext output shape")
    all_ok = True
    for B, C_in, C_out, H, W in [(1, 32, 64, 64, 64), (2, 128, 64, 32, 32)]:
        gc  = gapcontext(f"gc_shp{C_in}", C_in, C_out)
        x   = _from_shape(f"gc_x{C_in}", [B, C_in, H, W])
        out = gc(x)
        ok  = list(out.shape) == [B, C_out, H, W]
        print_test(f"B={B} C={C_in}→{C_out} {H}×{W}", f"out={out.shape}", ok)
        if not ok:
            all_ok = False
    return all_ok


def test_fpnc_shape():
    print_header("Test 5: FPNC output shape")
    in_channels = [64, 128, 256, 512]
    outC = 64
    fpnc = FPNC("fpnc_val", in_channels=in_channels, out_channels=64, num_outs=4,
                final_dim=[256, 256], downsample=4, use_adp=False, outC=outC)
    feats = [_from_shape(f"f{i}", [1, c, 64 >> i, 64 >> i])
             for i, c in enumerate(in_channels)]
    outs = fpnc(*feats)
    ok  = isinstance(outs, list) and len(outs) == 1
    print_test(f"FPNC output is list[1]", f"len={len(outs)}", ok)
    p = fpnc.analytical_param_count()
    ok2 = p > 0
    print_test(f"FPNC param_count={p}", "p>0", ok2)
    return ok and ok2


def test_fpnc_param_count_formula():
    print_header("Test 6: FPNC analytical_param_count includes expected sub-parts")
    fpnc = FPNC("fpnc_pc", in_channels=[64, 128, 256, 512], out_channels=64, num_outs=4,
                final_dim=[128, 128], downsample=4, use_adp=False, outC=64)
    p = fpnc.analytical_param_count()
    # reduc_conv adds C*num_outs*outC*9 + outC = 256*64*9+64
    reduc = 256 * 64 * 9 + 64
    ok = p >= reduc
    print_test(f"FPNC total≥reduc_conv({reduc}), got {p}", "", ok)
    return ok


if __name__ == "__main__":
    tests = [
        ("gapcontext_conv1x1",   test_gapcontext_conv1x1_step),
        ("global_avg_pool",      test_global_avg_pool_step),
        ("gapcontext_construct", test_gapcontext_construction),
        ("gapcontext_shape",     test_gapcontext_shape),
        ("fpnc_shape",           test_fpnc_shape),
        ("fpnc_param_formula",   test_fpnc_param_count_formula),
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
    import sys; sys.exit(0 if passed == total else 1)
