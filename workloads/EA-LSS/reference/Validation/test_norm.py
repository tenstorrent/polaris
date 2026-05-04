#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for norm TTSim module.

Validates the TTSim conversion of mmdet3d/ops/norm.py.

Test Coverage:
  1.  NaiveSyncBatchNorm1d shape – 2D and 3D inputs
  2.  NaiveSyncBatchNorm2d shape – 4D inputs
  3.  NaiveSyncBatchNorm3d shape – 5D inputs
  4.  BN1d data vs torch         – numerical agreement
  5.  BN2d data vs torch
"""

import os
import sys

_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_val_dir      = os.path.dirname(__file__)
for p in (_polaris_root, _ealss_root, _val_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn as tn
from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.norm import (
    NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d, NaiveSyncBatchNorm3d
)
from ttsim_utils import compare_arrays, print_header, print_test


# ============================================================================
# Helpers
# ============================================================================

rng = np.random.RandomState(7)


def _inject_bn_weights(bn_mod, C, nd="1d"):
    """Write known weights into a NaiveSyncBatchNormXd."""
    w = lambda *s: rng.randn(*s).astype(np.float32) * 0.2 + 1.0
    # BN1d uses a nested ._bn; BN2d/3d expose attributes directly.
    if nd == "1d":
        inner = bn_mod._bn
    else:
        inner = bn_mod
    inner.scale.data        = w(C)
    inner.bias_bn.data      = rng.randn(C).astype(np.float32) * 0.05
    inner.running_mean.data = rng.randn(C).astype(np.float32) * 0.1
    inner.running_var.data  = (rng.rand(C).astype(np.float32) * 0.5 + 0.5)


def _torch_bn(x_np, scale, bias, mean, var, nd, eps=1e-5, momentum=0.1):
    """Torch BN reference (eval mode)."""
    C = scale.shape[0]
    if nd == "1d":
        m = tn.BatchNorm1d(C, eps=eps, momentum=momentum)
    elif nd == "2d":
        m = tn.BatchNorm2d(C, eps=eps, momentum=momentum)
    else:
        m = tn.BatchNorm3d(C, eps=eps, momentum=momentum)
    m.weight.data       = torch.tensor(scale)
    m.bias.data         = torch.tensor(bias)
    m.running_mean.data = torch.tensor(mean)
    m.running_var.data  = torch.tensor(var)
    m.eval()
    with torch.no_grad():
        out = m(torch.tensor(x_np))
    return out.numpy()


# ============================================================================
# Tests
# ============================================================================

def test_bn1d_shape_2d():
    print_header("TEST 1: NaiveSyncBatchNorm1d shape (N,C)")
    C = 32
    bn = NaiveSyncBatchNorm1d("bn1d_2d", C)
    out = bn(_from_shape("bn1d_2d_in", [8, C]))
    assert out.shape == [8, C], f"Expected [8,{C}], got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_bn1d_shape_3d():
    print_header("TEST 2: NaiveSyncBatchNorm1d shape (N,C,L)")
    C = 16
    bn = NaiveSyncBatchNorm1d("bn1d_3d", C)
    out = bn(_from_shape("bn1d_3d_in", [4, C, 50]))
    assert out.shape == [4, C, 50], f"Expected [4,{C},50], got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_bn2d_shape():
    print_header("TEST 3: NaiveSyncBatchNorm2d shape (N,C,H,W)")
    C = 24
    bn = NaiveSyncBatchNorm2d("bn2d", C)
    out = bn(_from_shape("bn2d_in", [2, C, 14, 14]))
    assert out.shape == [2, C, 14, 14], f"Got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_bn3d_shape():
    print_header("TEST 4: NaiveSyncBatchNorm3d shape (N,C,D,H,W)")
    C = 8
    bn = NaiveSyncBatchNorm3d("bn3d", C)
    out = bn(_from_shape("bn3d_in", [1, C, 4, 8, 8]))
    assert out.shape == [1, C, 4, 8, 8], f"Got {out.shape}"
    print_test("PASS", f"shape={out.shape}")


def test_bn1d_data():
    print_header("TEST 5: NaiveSyncBatchNorm1d data vs torch")
    C = 12
    bn = NaiveSyncBatchNorm1d("bn1d_dat", C)
    _inject_bn_weights(bn, C, "1d")

    x_np = rng.randn(5, C, 20).astype(np.float32)
    out  = bn(_from_data("bn1d_dat_in", x_np))
    assert out.data is not None, "BN1d data is None"

    ref = _torch_bn(x_np,
                    bn._bn.scale.data, bn._bn.bias_bn.data,
                    bn._bn.running_mean.data, bn._bn.running_var.data,
                    "1d", eps=1e-3, momentum=0.01)
    compare_arrays(ref, out.data, "BN1d vs torch", rtol=1e-4, atol=1e-4)


def test_bn2d_data():
    print_header("TEST 6: NaiveSyncBatchNorm2d data vs torch")
    C = 8
    bn = NaiveSyncBatchNorm2d("bn2d_dat", C)
    _inject_bn_weights(bn, C, "2d")

    x_np = rng.randn(3, C, 6, 6).astype(np.float32)
    out  = bn(_from_data("bn2d_dat_in", x_np))
    assert out.data is not None, "BN2d data is None"

    ref = _torch_bn(x_np,
                    bn.scale.data, bn.bias_bn.data,
                    bn.running_mean.data, bn.running_var.data,
                    "2d", eps=1e-5, momentum=0.1)
    compare_arrays(ref, out.data, "BN2d vs torch", rtol=1e-4, atol=1e-4)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    results = {}
    for name, fn in [
        ("bn1d_shape_2d",  test_bn1d_shape_2d),
        ("bn1d_shape_3d",  test_bn1d_shape_3d),
        ("bn2d_shape",     test_bn2d_shape),
        ("bn3d_shape",     test_bn3d_shape),
        ("bn1d_data",      test_bn1d_data),
        ("bn2d_data",      test_bn2d_data),
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
