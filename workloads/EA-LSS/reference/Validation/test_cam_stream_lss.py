#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for LiftSplatShoot TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_cam_stream_lss.py

Numerical comparisons validate the depthwise and pointwise Conv2d steps
(BEVConv building blocks) against PyTorch.
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
from ttsim_modules.cam_stream_lss import LiftSplatShoot
from Reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays, ttsim_conv2d,
)


def _default_lss(tag="v"):
    return LiftSplatShoot(
        f"lss_{tag}",
        lss=False,
        final_dim=(900, 1600),
        camera_depth_range=[4.0, 45.0, 1.0],
        pc_range=[-50, -50, -5, 50, 50, 3],
        downsample=4,
        grid=3,
        inputC=256,
        camC=64,
    )


# ---------------------------------------------------------------------------
# Numerical comparison tests
# ---------------------------------------------------------------------------

def test_depthwise_conv_step():
    """Depthwise Conv2d (BEVConv DW step) — PyTorch vs ttsim_conv2d."""
    print_header("Test 1: Depthwise Conv2d step — PyTorch vs ttsim_conv2d")
    rng = np.random.RandomState(90)
    B, C, H, W = 1, 32, 8, 8

    x_np = (rng.randn(B, C, H, W) * 0.5).astype(np.float32)
    # Depthwise: one filter per input channel → weight [C, 1, kH, kW]
    w_np = (rng.randn(C, 1, 3, 3) * 0.1).astype(np.float32)
    b_np = (rng.randn(C) * 0.1).astype(np.float32)

    conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=True)
    conv.weight.data = torch.tensor(w_np)
    conv.bias.data   = torch.tensor(b_np)
    with torch.no_grad():
        pt_out = conv(torch.tensor(x_np)).numpy()
    print(f"  PyTorch DW Conv: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = ttsim_conv2d(x_np, w_np, bias=b_np, stride=1, padding=1, groups=C)
    print(f"  TTSim   DW Conv: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "depthwise conv step")
    return ok


def test_pointwise_conv_step():
    """Pointwise Conv2d (BEVConv PW channel-mix step) — PyTorch vs ttsim_conv2d."""
    print_header("Test 2: Pointwise Conv2d step — PyTorch vs ttsim_conv2d")
    rng = np.random.RandomState(91)
    B, C_in, C_out, H, W = 1, 32, 64, 8, 8

    x_np = (rng.randn(B, C_in, H, W) * 0.5).astype(np.float32)
    w_np = (rng.randn(C_out, C_in, 1, 1) * 0.1).astype(np.float32)
    b_np = (rng.randn(C_out) * 0.1).astype(np.float32)

    conv = nn.Conv2d(C_in, C_out, kernel_size=1, bias=True)
    conv.weight.data = torch.tensor(w_np)
    conv.bias.data   = torch.tensor(b_np)
    with torch.no_grad():
        pt_out = conv(torch.tensor(x_np)).numpy()
    print(f"  PyTorch PW Conv: shape={list(pt_out.shape)}, sample={pt_out.flatten()[:4]}")

    ts_out = ttsim_conv2d(x_np, w_np, bias=b_np, stride=1, padding=0)
    print(f"  TTSim   PW Conv: shape={list(ts_out.shape)}, sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "pointwise conv step")
    return ok


# ---------------------------------------------------------------------------
# Shape / param tests
# ---------------------------------------------------------------------------

def test_lss_param_count():
    print_header("Test 3: LiftSplatShoot param count (expected 7,357,108)")
    lss = _default_lss("pc")
    p   = lss.analytical_param_count()
    ok  = p == 7_357_108
    print_test(f"LiftSplatShoot params == 7,357,108", f"got {p:,}", ok)
    return ok


def test_lss_depth_bins():
    print_header("Test 4: D = 41 for depth range [4, 45, 1]")
    lss = _default_lss("D")
    ok  = lss.D == 41
    print_test(f"lss.D == 41", f"got {lss.D}", ok)
    return ok


def test_lss_cz():
    print_header("Test 5: cz = 128 for default params")
    lss = _default_lss("cz")
    ok  = lss.cz == 128
    print_test(f"lss.cz == 128", f"got {lss.cz}", ok)
    return ok


def test_lss_output_shape():
    print_header("Test 6: LiftSplatShoot output shape")
    lss = _default_lss("shp")
    x   = _from_shape("lss_x", [6, 256, 32, 88])
    out = lss(x)
    ok  = list(out.shape) == [6, 256, 32, 88]
    print_test(f"LiftSplatShoot output shape [6,256,32,88]", f"got {out.shape}", ok)
    return ok


if __name__ == "__main__":
    tests = [
        ("depthwise_conv_step",  test_depthwise_conv_step),
        ("pointwise_conv_step",  test_pointwise_conv_step),
        ("lss_param_count",      test_lss_param_count),
        ("lss_depth_bins",       test_lss_depth_bins),
        ("lss_cz",               test_lss_cz),
        ("lss_output_shape",     test_lss_output_shape),
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
