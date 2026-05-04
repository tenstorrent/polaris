#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for gaussian TTSim module.

Validates the TTSim conversion of
mmdet3d/core/utils/gaussian.py.

Test Coverage:
  1.  gaussian_2d values      – centre value == k, edge ~ 0
  2.  gaussian_radius math    – output is a positive number
  3.  draw_heatmap shape      – unchanged [H,W]
  4.  draw_heatmap values     – center pixel is highest
  5.  GaussianDepthTarget shape – [B, H//s, W//s, D-1]
  6.  GaussianDepthTarget data  – compute_numpy vs manual formula
"""

import os
import sys
import math

_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_val_dir      = os.path.dirname(__file__)
for p in (_polaris_root, _ealss_root, _val_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.gaussian import (
    gaussian_2d, draw_heatmap_gaussian, gaussian_radius,
    GaussianDepthTarget
)
from ttsim_utils import compare_arrays, print_header, print_test


# ============================================================================
# Tests
# ============================================================================

def test_gaussian_2d_values():
    print_header("TEST 1: gaussian_2d values")
    g = gaussian_2d((7, 7), sigma=1.0)
    assert g.shape == (7, 7), f"Expected (7,7), got {g.shape}"
    cx, cy = 3, 3
    assert np.isclose(g[cy, cx], 1.0, atol=1e-5), \
        f"Center value should be 1.0, got {g[cy,cx]:.5f}"
    corner = g[0, 0]
    assert corner < 0.01, f"Corner should be close to 0, got {corner:.4f}"
    print_test("PASS", f"center={g[cy,cx]:.4f}, corner={corner:.5f}")


def test_gaussian_radius():
    print_header("TEST 2: gaussian_radius")
    r = gaussian_radius((40.0, 60.0), min_overlap=0.5)
    assert isinstance(r, (int, float, np.floating)), f"Expected number, got {type(r)}"
    assert r > 0, f"Radius should be positive, got {r}"
    print_test("PASS", f"radius={r:.2f}")


def test_draw_heatmap_shape():
    print_header("TEST 3: draw_heatmap_gaussian shape")
    hm = np.zeros((40, 80), dtype=np.float32)
    draw_heatmap_gaussian(hm, center=(20, 15), radius=5)
    assert hm.shape == (40, 80), f"Shape changed to {hm.shape}"
    print_test("PASS", f"shape={hm.shape}")


def test_draw_heatmap_center_is_max():
    print_header("TEST 4: draw_heatmap center pixel is max")
    hm = np.zeros((40, 80), dtype=np.float32)
    cx, cy = 30, 18
    draw_heatmap_gaussian(hm, center=(cx, cy), radius=4, k=1)
    hy, hx = np.unravel_index(np.argmax(hm), hm.shape)
    assert hm[cy, cx] == hm.max(), \
        f"Max at ({hx},{hy}) but center is ({cx},{cy})"
    print_test("PASS", f"max at center ({cx},{cy})={hm[cy,cx]:.4f}")


def test_gaussian_depth_target_shape():
    print_header("TEST 5: GaussianDepthTarget shape")
    stride = 16
    dmin, dmax, dstep = 2.0, 58.0, 0.5
    B, tH, tW = 2, 256, 704
    gdt = GaussianDepthTarget("gdt_s", stride, cam_depth_range=(dmin, dmax, dstep))
    dummy_depth = _from_shape("gdt_depth_s", [B, tH, tW])
    out = gdt(dummy_depth)

    D_bins = len(np.arange(dmin, dmax + 1, dstep))
    tgt_H, tgt_W = tH // stride, tW // stride
    assert out.shape == [B, tgt_H, tgt_W, D_bins - 1], \
        f"Expected [{B},{tgt_H},{tgt_W},{D_bins-1}], got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_gaussian_depth_target_data():
    print_header("TEST 6: GaussianDepthTarget data vs compute_numpy")
    stride = 4
    dmin, dmax, dstep = 1.0, 10.0, 1.0
    B, tH, tW = 1, 64, 64
    gdt = GaussianDepthTarget("gdt_d", stride, cam_depth_range=(dmin, dmax, dstep))

    rng = np.random.RandomState(21)
    depth_np = rng.uniform(dmin + 0.5, dmax - 0.5, size=(B, tH, tW)).astype(np.float32)

    depth_t = _from_data("gdt_d_dep", depth_np)
    out = gdt(depth_t)

    # compute_numpy returns (depth_dist, min_depth, std_var) — use first element
    result = gdt.compute_numpy(depth_np)
    ref = result[0]
    assert ref is not None, "compute_numpy returned None"
    assert ref.shape == tuple(out.shape), \
        f"Shape mismatch: ref={ref.shape}, ttsim={out.shape}"
    print_test("PASS", f"shape={ref.shape}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("gaussian_2d_values",         test_gaussian_2d_values),
        ("gaussian_radius",            test_gaussian_radius),
        ("draw_heatmap_shape",         test_draw_heatmap_shape),
        ("draw_heatmap_center_is_max", test_draw_heatmap_center_is_max),
        ("gdt_shape",                  test_gaussian_depth_target_shape),
        ("gdt_data",                   test_gaussian_depth_target_data),
    ]:
        try:
            fn()
            results[name] = "PASS"
        except Exception as exc:
            results[name] = f"FAIL: {exc}"
            import traceback; traceback.print_exc()

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v == "PASS")
    for k, v in results.items():
        print(f"  {'✓' if v=='PASS' else '✗'} {k}: {v}")
    print(f"\n  {passed}/{len(results)} tests passed")
    import sys; sys.exit(0 if passed == len(results) else 1)
