#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for FPN neck TTSim module.

Validates the TTSim conversion of mmdet3d/models/necks/fpn.py.
Each test compares the TTSim graph output (.data) against the equivalent
PyTorch operation at every step of the forward pass.

Test Coverage:
  1.  Construction               – param count > 0
  2.  Lateral Conv2d Step        – TTSim lateral Conv2d vs torch.nn.Conv2d
  3.  FPN Conv2d Step            – TTSim FPN Conv2d vs torch.nn.Conv2d
  4.  Full 2-level forward       – complete FPN output vs PyTorch reference
  5.  4-level output shapes      – shape validation for 4 input levels
  6.  Param count formula        – analytical vs manual computation
  7.  with_norm BN delta         – BN adds correct extra params
  8.  Extra output via maxpool   – num_outs > num_ins adds pooled level
  9.  start_level skips level    – start_level=1 skips first backbone level

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_fpn.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.fpn import FPN
from reference.Validation.ttsim_utils import (
    compare_arrays, print_header, print_test,
    ttsim_conv2d, ttsim_add,
)

# ============================================================================
# PyTorch reference helpers
# ============================================================================

def pt_conv2d(x_np, w_np, b_np=None, stride=1, padding=0):
    in_ch, out_ch = w_np.shape[1], w_np.shape[0]
    kH, kW = w_np.shape[2], w_np.shape[3]
    m = nn.Conv2d(in_ch, out_ch, (kH, kW), stride=stride, padding=padding,
                  bias=(b_np is not None))
    m.weight.data = torch.tensor(w_np)
    if b_np is not None:
        m.bias.data = torch.tensor(b_np)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(x_np)).numpy()


def make_w(shape, rng, scale=0.05):
    return (rng.randn(*shape) * scale).astype(np.float32)


def inject_fpn_weights(fpn, name, in_channels, out_channels, num_levels, rng):
    """Inject weights into FPN _tensors; return matching weight dict."""
    w = {}
    for i in range(num_levels):
        in_ch = in_channels[i]
        key_lw = f'{name}.lat_conv{i}.param'
        key_lb = f'{name}.lat_conv{i}.bias'
        w_lat = make_w((out_channels, in_ch, 1, 1), rng)
        b_lat = make_w((out_channels,), rng)
        fpn._tensors[key_lw].data = w_lat
        if key_lb in fpn._tensors:
            fpn._tensors[key_lb].data = b_lat
        w[f'lat{i}_w'] = w_lat
        w[f'lat{i}_b'] = b_lat

        key_fw = f'{name}.fpn_conv{i}.param'
        key_fb = f'{name}.fpn_conv{i}.bias'
        w_fpn = make_w((out_channels, out_channels, 3, 3), rng)
        b_fpn = make_w((out_channels,), rng)
        fpn._tensors[key_fw].data = w_fpn
        if key_fb in fpn._tensors:
            fpn._tensors[key_fb].data = b_fpn
        w[f'fpn{i}_w'] = w_fpn
        w[f'fpn{i}_b'] = b_fpn
    return w


# ============================================================================
# Tests
# ============================================================================

def test_construction():
    print_header("Test 1: Construction (default)")
    fpn = FPN("fpn1", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4)
    p = fpn.analytical_param_count()
    ok = p > 0
    print_test("FPN() params > 0", f"got {p:,}", ok)
    return ok


def test_lateral_conv2d_step():
    print_header("Test 2: Lateral Conv2d Step — TTSim vs PyTorch")
    print_test("TTSim lateral Conv2d (.data) vs torch.nn.Conv2d (1×1 kernel)")
    rng = np.random.RandomState(20)
    B, in_ch, out_ch, H, W = 2, 64, 32, 16, 16
    x_np = make_w((B, in_ch, H, W), rng, scale=0.5)
    w_np = make_w((out_ch, in_ch, 1, 1), rng)
    b_np = make_w((out_ch,), rng)

    pt_out    = pt_conv2d(x_np, w_np, b_np, stride=1, padding=0)
    ttsim_out = ttsim_conv2d(x_np, w_np, b_np, stride=1, padding=0)

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "Lateral Conv2d 1×1")


def test_fpn_conv2d_step():
    print_header("Test 3: FPN Conv2d Step — TTSim vs PyTorch")
    print_test("TTSim FPN Conv2d (.data) vs torch.nn.Conv2d (3×3, pad=1)")
    rng = np.random.RandomState(21)
    B, ch, H, W = 2, 32, 16, 16
    x_np = make_w((B, ch, H, W), rng, scale=0.5)
    w_np = make_w((ch, ch, 3, 3), rng)
    b_np = make_w((ch,), rng)

    pt_out    = pt_conv2d(x_np, w_np, b_np, stride=1, padding=1)
    ttsim_out = ttsim_conv2d(x_np, w_np, b_np, stride=1, padding=1)

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  PyTorch sample: {pt_out.flatten()[:4]}")
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  TTSim   sample: {ttsim_out.flatten()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "FPN Conv2d 3×3")


def test_full_2level_forward():
    print_header("Test 4: Full 2-Level FPN Forward — TTSim vs PyTorch")
    print_test("TTSim FPN outputs (.data) vs manual PyTorch lateral+upsample+fpn_conv")
    rng = np.random.RandomState(23)
    name = "fpn_t4"
    in_channels = [64, 128]
    out_ch = 32
    B, H0, W0 = 2, 16, 16

    fpn = FPN(name, in_channels=in_channels, out_channels=out_ch,
              num_outs=2, with_norm=False)
    w = inject_fpn_weights(fpn, name, in_channels, out_ch, 2, rng)

    # Fine (level 0) and coarse (level 1) feature maps
    f0_np = make_w((B, in_channels[0], H0,   W0),   rng, scale=0.5)
    f1_np = make_w((B, in_channels[1], H0//2, W0//2), rng, scale=0.5)

    # PyTorch reference: lateral → top-down merge → fpn conv
    lat1    = pt_conv2d(f1_np, w['lat1_w'], w['lat1_b'])
    lat0    = pt_conv2d(f0_np, w['lat0_w'], w['lat0_b'])
    ups1    = F_torch.interpolate(torch.tensor(lat1), scale_factor=2, mode='nearest').numpy()
    merged0 = lat0 + ups1
    pt_out0 = pt_conv2d(merged0, w['fpn0_w'], w['fpn0_b'], stride=1, padding=1)
    pt_out1 = pt_conv2d(lat1,    w['fpn1_w'], w['fpn1_b'], stride=1, padding=1)

    print(f"  PyTorch out[0]: shape={pt_out0.shape}, sample={pt_out0.flatten()[:4]}")
    print(f"  PyTorch out[1]: shape={pt_out1.shape}, sample={pt_out1.flatten()[:4]}")

    # TTSim forward
    f0_t = _from_data("f0_t4", f0_np)
    f1_t = _from_data("f1_t4", f1_np)
    outs = fpn(f0_t, f1_t)
    tt0, tt1 = outs[0].data, outs[1].data
    print(f"  TTSim   out[0]: shape={tt0.shape}, sample={tt0.flatten()[:4]}")
    print(f"  TTSim   out[1]: shape={tt1.shape}, sample={tt1.flatten()[:4]}")

    ok0 = compare_arrays(pt_out0, tt0, "FPN out[0]")
    ok1 = compare_arrays(pt_out1, tt1, "FPN out[1]")
    return ok0 and ok1


def test_output_shapes_4level():
    print_header("Test 5: 4-level output shapes")
    fpn = FPN("fpn5", in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4)
    x0 = _from_shape("f0", [2, 256,  128, 128])
    x1 = _from_shape("f1", [2, 512,   64,  64])
    x2 = _from_shape("f2", [2, 1024,  32,  32])
    x3 = _from_shape("f3", [2, 2048,  16,  16])
    outs = fpn(x0, x1, x2, x3)
    expected = [
        [2, 256, 128, 128],
        [2, 256,  64,  64],
        [2, 256,  32,  32],
        [2, 256,  16,  16],
    ]
    ok = True
    for i, (o, e) in enumerate(zip(outs, expected)):
        match = list(o.shape) == e
        print_test(f"FPN out[{i}]", f"got {list(o.shape)} expected {e}", match)
        if not match: ok = False
    return ok


def test_param_count_lateral_fpn():
    print_header("Test 6: Param count (lateral + fpn convs, no norm)")
    in_ch = [256, 512]
    out_ch = 128
    fpn = FPN("fpn6", in_channels=in_ch, out_channels=out_ch, num_outs=2, with_norm=False)
    lat_params = sum(c * out_ch + out_ch for c in in_ch)
    fpn_params = len(in_ch) * (out_ch * out_ch * 9 + out_ch)
    expected = lat_params + fpn_params
    got = fpn.analytical_param_count()
    ok = got == expected
    print_test("FPN param count (2 levels, no norm)",
               f"got {got:,} expected {expected:,}", ok)
    return ok


def test_with_norm():
    print_header("Test 7: with_norm=True adds BN params")
    fpn_no_norm = FPN("fpn7a", in_channels=[256, 512], out_channels=128,
                      num_outs=2, with_norm=False)
    fpn_norm    = FPN("fpn7b", in_channels=[256, 512], out_channels=128,
                      num_outs=2, with_norm=True)
    p_no = fpn_no_norm.analytical_param_count()
    p_with = fpn_norm.analytical_param_count()
    extra_expected = 4 * 128
    ok = (p_with - p_no) == extra_expected
    print_test("BN delta", f"got {p_with - p_no} expected {extra_expected}", ok)
    return ok


def test_extra_output_via_pool():
    print_header("Test 8: num_outs > num_ins adds pooled extra level")
    fpn = FPN("fpn8", in_channels=[256, 512], out_channels=256, num_outs=3)
    x0 = _from_shape("g0", [2, 256, 64, 64])
    x1 = _from_shape("g1", [2, 512, 32, 32])
    outs = fpn(x0, x1)
    ok = len(outs) == 3
    print_test("num_outs extra level",
               f"got {len(outs)} outputs, shapes: {[list(o.shape) for o in outs]}", ok)
    return ok


def test_start_level():
    print_header("Test 9: start_level=1 skips first backbone level")
    fpn = FPN("fpn9", in_channels=[64, 128, 256, 512], out_channels=128,
              num_outs=3, start_level=1, end_level=4)
    x0 = _from_shape("s0", [2,  64, 128, 128])
    x1 = _from_shape("s1", [2, 128,  64,  64])
    x2 = _from_shape("s2", [2, 256,  32,  32])
    x3 = _from_shape("s3", [2, 512,  16,  16])
    outs = fpn(x0, x1, x2, x3)
    ok = len(outs) == 3
    print_test("start_level=1", f"got {len(outs)} outputs", ok)
    return ok


if __name__ == "__main__":
    tests = [
        ("construction",           test_construction),
        ("lateral_conv2d_step",    test_lateral_conv2d_step),
        ("fpn_conv2d_step",        test_fpn_conv2d_step),
        ("full_2level_forward",    test_full_2level_forward),
        ("output_shapes_4level",   test_output_shapes_4level),
        ("param_count_lateral_fpn",test_param_count_lateral_fpn),
        ("with_norm",              test_with_norm),
        ("extra_output_via_pool",  test_extra_output_via_pool),
        ("start_level",            test_start_level),
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
