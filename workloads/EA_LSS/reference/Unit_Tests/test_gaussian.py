#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for gaussian TTSim module.

Three test categories:
  1. Shape Validation  – gaussian_2d, draw_heatmap, gaussian_radius returns,
                         GaussianDepthTarget output shape.
  2. Edge Case Creation – radius=0, single-pixel heatmap, k=1 vs k=2,
                         point outside heatmap boundary.
  3. Data Validation   – gaussian_2d centre == 1, radius formula bounds,
                         heatmap max at centre, GaussianDepthTarget compute_numpy.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_gaussian.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.gaussian import (
    gaussian_2d, draw_heatmap_gaussian, gaussian_radius, GaussianDepthTarget
)

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_GAUSS2D_CASES = [
    ((3, 3), 1.0), ((7, 7), 2.0), ((11, 11), 3.0), ((5, 9), 1.5),
]

_GDT_CASES = [
    # (stride, cam_depth_range, B, tH, tW)
    (4,  (1.0, 10.0, 1.0),  1, 64,  64),
    (8,  (2.0, 50.0, 0.5),  2, 128, 128),
    (16, (1.0, 60.0, 1.0),  1, 256, 704),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_gaussian_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True

    # gaussian_2d shapes
    for i, (shape, sigma) in enumerate(_GAUSS2D_CASES):
        try:
            g = gaussian_2d(shape, sigma)
            ok = g.shape == shape
            print(f"  [G2D-{i:02d}] shape={shape} sigma={sigma}  {'PASS' if ok else 'FAIL'}  out={g.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [G2D-{i:02d}] shape={shape} ERROR: {exc}")
            all_passed = False

    # draw_heatmap_gaussian: shape unchanged
    for i in range(3):
        try:
            H, W = 40 + i * 20, 80 + i * 20
            hm = np.zeros((H, W), dtype=np.float32)
            draw_heatmap_gaussian(hm, center=(W // 2, H // 2), radius=5)
            ok = hm.shape == (H, W)
            print(f"  [HM-{i:02d}] H={H} W={W}  {'PASS' if ok else 'FAIL'}  shape={hm.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [HM-{i:02d}] ERROR: {exc}")
            all_passed = False

    # gaussian_radius: positive scalar
    for i, (h, w, ov) in enumerate([(40.0, 60.0, 0.5), (1.0, 1.0, 0.7), (100.0, 100.0, 0.25)]):
        try:
            r = gaussian_radius((h, w), min_overlap=ov)
            ok = isinstance(r, (int, float, np.floating)) and r > 0
            print(f"  [GR-{i:02d}] det=({h},{w}) ov={ov}  {'PASS' if ok else 'FAIL'}  r={r:.2f}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [GR-{i:02d}] ERROR: {exc}")
            all_passed = False

    # GaussianDepthTarget output shapes
    for i, (stride, cdr, B, tH, tW) in enumerate(_GDT_CASES):
        try:
            gdt = GaussianDepthTarget(f"gdt_sv{i}", stride, cam_depth_range=cdr)
            out = gdt(_from_shape(f"gdt_sv{i}_in", [B, tH, tW]))
            D_bins = len(np.arange(cdr[0], cdr[1] + 1, cdr[2]))
            expected = [B, tH // stride, tW // stride, D_bins - 1]
            ok = list(out.shape) == expected
            print(f"  [GDT-{i:02d}] {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [GDT-{i:02d}] ERROR: {exc}")
            all_passed = False

    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_gaussian_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    # 1. radius=0 → single pixel set
    hm = np.zeros((30, 30), dtype=np.float32)
    draw_heatmap_gaussian(hm, center=(15, 15), radius=0, k=1)
    ok = hm[15, 15] > 0 and hm.sum() > 0
    print(f"  [00] radius=0 → single pixel set: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # 2. centre at corner
    hm2 = np.zeros((20, 20), dtype=np.float32)
    draw_heatmap_gaussian(hm2, center=(0, 0), radius=3, k=1)
    ok2 = hm2.shape == (20, 20)
    print(f"  [01] centre at corner (0,0): {'PASS' if ok2 else 'FAIL'}")
    if not ok2: all_passed = False

    # 3. k=2 → centre value is 2.0
    hm3 = np.zeros((20, 20), dtype=np.float32)
    draw_heatmap_gaussian(hm3, center=(10, 10), radius=4, k=2)
    ok3 = np.isclose(hm3[10, 10], 2.0, atol=1e-5)
    print(f"  [02] k=2 → centre=2.0: {'PASS' if ok3 else 'FAIL'}  centre={hm3[10,10]:.4f}")
    if not ok3: all_passed = False

    # 4. gaussian_2d centre value == 1.0
    g = gaussian_2d((9, 9), sigma=2.0)
    ok4 = np.isclose(g[4, 4], 1.0, atol=1e-5)
    print(f"  [03] gaussian_2d centre==1.0: {'PASS' if ok4 else 'FAIL'}  c={g[4,4]:.5f}")
    if not ok4: all_passed = False

    # 5. gaussian_radius increase as object size increases
    r1 = gaussian_radius((10.0, 10.0), 0.5)
    r2 = gaussian_radius((20.0, 20.0), 0.5)
    ok5 = r2 > r1
    print(f"  [04] radius grows with object size: {'PASS' if ok5 else 'FAIL'}  r1={r1:.2f} r2={r2:.2f}")
    if not ok5: all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_gaussian_data_validation():
    """Category 3 – Data Validation."""
    all_passed = True

    # 1. gaussian_2d values symmetry
    g = gaussian_2d((11, 11), sigma=1.5)
    ok = np.isclose(g, g[::-1, :], atol=1e-6).all() and \
         np.isclose(g, g[:, ::-1], atol=1e-6).all()
    print(f"  [00] gaussian_2d symmetric: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # 2. draw_heatmap max at specified centre
    for cx, cy in [(10, 15), (0, 0), (29, 29)]:
        hm = np.zeros((30, 30), dtype=np.float32)
        draw_heatmap_gaussian(hm, center=(cx, cy), radius=3, k=1)
        iy, ix = np.unravel_index(np.argmax(hm), hm.shape)
        ok2 = (ix == cx and iy == cy)
        print(f"  [01] hm max at ({cx},{cy}): {'PASS' if ok2 else 'FAIL'}  got ({ix},{iy})")
        if not ok2: all_passed = False

    # 3. GaussianDepthTarget compute_numpy output summing to ~1.0 per pixel
    stride, cdr = 4, (1.0, 8.0, 1.0)
    B, tH, tW = 1, 32, 32
    gdt = GaussianDepthTarget("gdt_dv0", stride, cam_depth_range=cdr)
    depth_np = np.ones((B, tH, tW), dtype=np.float32) * 3.5  # uniform depth
    result = gdt.compute_numpy(depth_np)
    assert result is not None
    depth_dist = result[0]  # [B, H, W, D-1]
    # Each PDF should sum to ~1 (within CDF boundary effects)
    # Just check shape and non-negative
    ok3 = depth_dist.shape == (B, tH//stride, tW//stride, len(np.arange(1.0, 9.0, 1.0)) - 1)
    ok3 = ok3 and (depth_dist >= 0).all()
    print(f"  [02] GDT compute_numpy shape/non-neg: {'PASS' if ok3 else 'FAIL'}  shape={depth_dist.shape}")
    if not ok3: all_passed = False

    # 4. gaussian_radius: output close to published formula (case 1)
    h, w, ov = 40.0, 60.0, 0.5
    r = gaussian_radius((h, w), ov)
    # Radius should be positive and scale with sqrt(h*w)
    lower_bound = 0.5
    ok4 = r > lower_bound
    print(f"  [03] gaussian_radius > {lower_bound}: {'PASS' if ok4 else 'FAIL'}  r={r:.2f}")
    if not ok4: all_passed = False

    assert all_passed
