#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for SECONDFPN neck TTSim module.
Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_second_fpn.py

Note: TTSim ConvTranspose2d tracks shapes only (no data compute).
      Numerical step comparisons use pure-NumPy reference vs PyTorch.
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
from ttsim_modules.second_fpn import SECONDFPN
from reference.Validation.ttsim_utils import print_header, print_test, compare_arrays

BN_EPS = 1e-3  # SECONDFPN default BN epsilon


# ---------------------------------------------------------------------------
# Pure-numpy ConvTranspose2d (extracted from PyTorch semantics)
# ---------------------------------------------------------------------------
def conv_transpose2d_numpy(x, weight, stride=2):
    """
    Pure-NumPy ConvTranspose2d: x [N,C_in,H,W], weight [C_in, C_out, kH, kW].
    Numerically equivalent to torch.nn.ConvTranspose2d with padding=0, no bias.
    """
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, kH, kW = weight.shape
    assert C_in == C_in_w
    H_out = (H_in - 1) * stride + kH
    W_out = (W_in - 1) * stride + kW
    out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for i in range(H_in):
        for j in range(W_in):
            patch = np.einsum('nc,cokh,kw->nokh', x[:, :, i, j], weight, np.eye(kH, dtype=np.float32)[:, :kW])
            # For each input position (i,j) → accumulate into output region
            for ki in range(kH):
                for kj in range(kW):
                    h_out = i * stride + ki
                    w_out = j * stride + kj
                    out[:, :, h_out, w_out] += np.einsum(
                        'nc,co->no', x[:, :, i, j], weight[:, :, ki, kj]
                    )
    return out


def batchnorm2d_numpy(x, scale, bias_bn, rm, rv, eps=BN_EPS):
    """BN2d inference: (x - mean) / sqrt(var + eps) * scale + bias."""
    mu    = rm   [np.newaxis, :, np.newaxis, np.newaxis]
    sigma = np.sqrt(rv[np.newaxis, :, np.newaxis, np.newaxis] + eps)
    g     = scale   [np.newaxis, :, np.newaxis, np.newaxis]
    b     = bias_bn [np.newaxis, :, np.newaxis, np.newaxis]
    return (x - mu) / sigma * g + b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construction():
    print_header("Test 1: Construction (default config)")
    sfpn = SECONDFPN("sfpn1")
    p = sfpn.analytical_param_count()
    print_test("SECONDFPN() param_count > 0", f"got {p:,}")
    return p > 0


def test_conv_transpose_step():
    """ConvTranspose2d step: pure-NumPy vs PyTorch."""
    print_header("Test 2: ConvTranspose2d step — PyTorch vs NumPy reference")
    rng = np.random.RandomState(20)
    in_ch, out_ch, k = 16, 32, 2
    x_np = (rng.randn(1, in_ch, 4, 4) * 0.5).astype(np.float32)
    # weight shape: [in_ch, out_ch, k, k]
    w_np = (rng.randn(in_ch, out_ch, k, k) * 0.1).astype(np.float32)

    # PyTorch
    pt_ct = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=k, bias=False)
    pt_ct.weight.data = torch.tensor(w_np)
    with torch.no_grad():
        pt_out = pt_ct(torch.tensor(x_np)).numpy()
    print(f"  PyTorch ConvTranspose2d: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # NumPy reference
    np_out = conv_transpose2d_numpy(x_np, w_np, stride=k)
    print(f"  NumPy  ConvTranspose2d: shape={list(np_out.shape)}, "
          f"sample={np_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, np_out, "ConvTranspose2d step", atol=1e-4)
    return ok


def test_bn_after_deconv():
    """BN2d step after ConvTranspose2d: PyTorch vs NumPy."""
    print_header("Test 3: BN2d after ConvTranspose — PyTorch vs NumPy")
    rng = np.random.RandomState(21)
    in_ch, out_ch, k = 16, 32, 2
    x_np = (rng.randn(1, in_ch, 4, 4) * 0.5).astype(np.float32)
    w_np = (rng.randn(in_ch, out_ch, k, k) * 0.1).astype(np.float32)
    bn_scale = np.ones(out_ch, np.float32)
    bn_bias  = np.zeros(out_ch, np.float32)
    bn_mean  = np.zeros(out_ch, np.float32)
    bn_var   = np.ones(out_ch, np.float32)

    # PyTorch: ConvTranspose → BN → ReLU
    pt_ct = nn.ConvTranspose2d(in_ch, out_ch, k, stride=k, bias=False)
    pt_ct.weight.data = torch.tensor(w_np)
    pt_bn = nn.BatchNorm2d(out_ch, eps=BN_EPS, momentum=0.01)
    pt_bn.weight.data = torch.tensor(bn_scale)
    pt_bn.bias.data   = torch.tensor(bn_bias)
    pt_bn.running_mean.data = torch.tensor(bn_mean)
    pt_bn.running_var.data  = torch.tensor(bn_var)
    pt_bn.eval()
    with torch.no_grad():
        pt_deconv = pt_ct(torch.tensor(x_np))
        pt_bn_out = pt_bn(pt_deconv).numpy()
        pt_relu   = torch.relu(torch.tensor(pt_bn_out)).numpy()
    print(f"  PyTorch BN→ReLU: shape={list(pt_relu.shape)}, "
          f"sample={pt_relu.flatten()[:4]}")

    # NumPy reference
    np_deconv = conv_transpose2d_numpy(x_np, w_np, stride=k)
    np_bn     = batchnorm2d_numpy(np_deconv, bn_scale, bn_bias, bn_mean, bn_var)
    np_relu   = np.maximum(0, np_bn)
    print(f"  NumPy  BN→ReLU: shape={list(np_relu.shape)}, "
          f"sample={np_relu.flatten()[:4]}")

    ok = compare_arrays(pt_relu, np_relu, "ConvTranspose→BN→ReLU", atol=1e-4)
    return ok


def test_output_shape_default():
    print_header("Test 4: Default config shapes (TTSim shape tracking)")
    sfpn = SECONDFPN("sfpn4", in_channels=[128,128,256],
                     out_channels=[256,256,256], upsample_strides=[1,2,4])
    x0 = _from_shape("d0", [2, 128, 64, 64])
    x1 = _from_shape("d1", [2, 128, 32, 32])
    x2 = _from_shape("d2", [2, 256, 16, 16])
    out = sfpn(x0, x1, x2)
    expected = [2, 768, 64, 64]
    ok = list(out.shape) == expected
    print_test("SECONDFPN out", f"got {list(out.shape)} expected {expected}")
    return ok


def test_stride1_identity():
    print_header("Test 5: Stride-1 deblock (identity in H/W)")
    sfpn = SECONDFPN("sfpn5", in_channels=[128], out_channels=[256], upsample_strides=[1])
    x = _from_shape("d5", [2, 128, 32, 32])
    out = sfpn(x)
    expected = [2, 256, 32, 32]
    ok = list(out.shape) == expected
    print_test("Stride-1 keeps H/W", f"got {list(out.shape)} expected {expected}")
    return ok


def test_stride2_doubling():
    print_header("Test 6: Stride-2 deblock (×2 upsampling)")
    sfpn = SECONDFPN("sfpn6", in_channels=[128], out_channels=[256], upsample_strides=[2])
    x = _from_shape("d6", [2, 128, 16, 16])
    out = sfpn(x)
    expected = [2, 256, 32, 32]
    ok = list(out.shape) == expected
    print_test("Stride-2 doubles H/W", f"got {list(out.shape)} expected {expected}")
    return ok


def test_param_count():
    print_header("Test 7: Param count formula")
    sfpn = SECONDFPN("sfpn7", in_channels=[128,128,256],
                     out_channels=[256,256,256], upsample_strides=[1,2,4])
    # sum_i (in_i * out_i * k_i^2 + 2*out_i)
    expected = (128*256*1*1 + 2*256) + (128*256*2*2 + 2*256) + (256*256*4*4 + 2*256)
    got = sfpn.analytical_param_count()
    ok = got == expected
    print_test("SECONDFPN param count", f"got {got:,} expected {expected:,}")
    return ok

if __name__ == "__main__":
    tests = [
        ("construction",              test_construction),
        ("conv_transpose_step",       test_conv_transpose_step),
        ("bn_after_deconv",           test_bn_after_deconv),
        ("output_shape_default",      test_output_shape_default),
        ("stride1_identity",          test_stride1_identity),
        ("stride2_doubling",          test_stride2_doubling),
        ("param_count",               test_param_count),
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
