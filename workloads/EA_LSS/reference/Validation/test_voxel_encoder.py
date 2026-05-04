#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for HardSimpleVFE and HardSimpleVFE_ATT TTSim modules.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_voxel_encoder.py

Note: TTSim transpose and reduce-max ops do not propagate data.
      Numerical comparisons are done with pure NumPy vs PyTorch references.
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

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.voxel_encoder import HardSimpleVFE, HardSimpleVFE_ATT
from reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays,
    conv1d_numpy, batchnorm1d_numpy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pfn_numpy(x_np, conv_w, bn_scale, bn_bias, bn_mean, bn_var):
    """
    Pure-NumPy PFN forward: [V,P,C] → Conv1d→BN→max → [V,1,C_out].
    Matches _PFNLayer.__call__ which does:
        transpose [V,P,C] → [V,C,P] → Conv1d → BN1d → transpose → max(axis=1)
    """
    x = x_np.transpose(0, 2, 1)                 # [V, C, P]
    x = conv1d_numpy(x, conv_w, bias=None)       # [V, C_out, P]
    x = batchnorm1d_numpy(x, bn_scale, bn_bias, bn_mean, bn_var)
    x = x.transpose(0, 2, 1)                    # [V, P, C_out]
    x = x.max(axis=1, keepdims=True)            # [V, 1, C_out]
    return x


def pt_pfn_forward(x_np, conv_w, bn_scale, bn_bias, bn_mean, bn_var):
    """PyTorch reference for PFN."""
    C_out = conv_w.shape[0]
    x_pt = torch.tensor(x_np).permute(0, 2, 1)  # [V, C, P]
    c = nn.Conv1d(x_np.shape[2], C_out, 1, bias=False)
    c.weight.data = torch.tensor(conv_w)
    bn = nn.BatchNorm1d(C_out, eps=1e-5, momentum=0.1)
    bn.weight.data = torch.tensor(bn_scale)
    bn.bias.data   = torch.tensor(bn_bias)
    bn.running_mean.data = torch.tensor(bn_mean)
    bn.running_var.data  = torch.tensor(bn_var)
    bn.eval()
    with torch.no_grad():
        y = bn(c(x_pt))                         # [V, C_out, P]
        y = y.permute(0, 2, 1)                  # [V, P, C_out]
        y = y.max(dim=1, keepdim=True)[0]       # [V, 1, C_out]
    return y.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_mean_aggregation():
    """HardSimpleVFE mean step: PyTorch mean vs NumPy mean."""
    print_header("Test 1: HardSimpleVFE mean aggregation — PyTorch vs NumPy")
    rng = np.random.RandomState(40)
    V, P, C = 5, 8, 4
    x_np = (rng.randn(V, P, C) * 0.5).astype(np.float32)

    # PyTorch reference
    pt_mean = torch.tensor(x_np).mean(dim=1).numpy()
    print(f"  PyTorch mean: shape={list(pt_mean.shape)}, "
          f"sample={pt_mean.flatten()[:4]}")

    # NumPy (TTSim equivalent)
    np_mean = x_np.mean(axis=1)
    print(f"  NumPy  mean: shape={list(np_mean.shape)}, "
          f"sample={np_mean.flatten()[:4]}")

    ok = compare_arrays(pt_mean, np_mean, "VFE mean aggregation")
    return ok


def test_pfn_conv1d_step():
    """PFN Conv1d step: PyTorch vs conv1d_numpy."""
    print_header("Test 2: PFN Conv1d step — PyTorch vs TTSim")
    rng = np.random.RandomState(41)
    V, P, C_in, C_out = 5, 4, 32, 32
    x_np = (rng.randn(V, C_in, P) * 0.5).astype(np.float32)  # [V,C,P]
    w    = (rng.randn(C_out, C_in, 1) * 0.1).astype(np.float32)

    # PyTorch
    c = nn.Conv1d(C_in, C_out, 1, bias=False)
    c.weight.data = torch.tensor(w)
    with torch.no_grad():
        pt_out = c(torch.tensor(x_np)).numpy()
    print(f"  PyTorch Conv1d: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # conv1d_numpy
    ts_out = conv1d_numpy(x_np, w, bias=None)
    print(f"  TTSim  Conv1d: shape={list(ts_out.shape)}, "
          f"sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "PFN Conv1d step")
    return ok


def test_pfn_bn_step():
    """PFN BN1d step: PyTorch vs batchnorm1d_numpy."""
    print_header("Test 3: PFN BN1d step — PyTorch vs TTSim")
    rng = np.random.RandomState(42)
    V, C, P = 5, 32, 4
    x_np = (rng.randn(V, C, P) * 0.5).astype(np.float32)
    s = (rng.randn(C) * 0.1 + 1.0).astype(np.float32)
    b = (rng.randn(C) * 0.1).astype(np.float32)
    m = (rng.randn(C) * 0.1).astype(np.float32)
    v = (np.abs(rng.randn(C)) + 0.5).astype(np.float32)

    # PyTorch
    bn = nn.BatchNorm1d(C, eps=1e-5, momentum=0.1)
    bn.weight.data = torch.tensor(s)
    bn.bias.data   = torch.tensor(b)
    bn.running_mean.data = torch.tensor(m)
    bn.running_var.data  = torch.tensor(v)
    bn.eval()
    with torch.no_grad():
        pt_out = bn(torch.tensor(x_np)).numpy()
    print(f"  PyTorch BN1d: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    ts_out = batchnorm1d_numpy(x_np, s, b, m, v)
    print(f"  TTSim  BN1d: shape={list(ts_out.shape)}, "
          f"sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "PFN BN1d step", atol=1e-5)
    return ok


def test_pfn_full_forward():
    """PFN full forward: PyTorch vs NumPy (Conv1d→BN→max)."""
    print_header("Test 4: PFN full forward — PyTorch vs NumPy")
    rng = np.random.RandomState(43)
    V, P, C = 5, 4, 32
    x_np = (rng.randn(V, P, C) * 0.5).astype(np.float32)  # [V,P,C]
    w    = (rng.randn(C, C, 1) * 0.1).astype(np.float32)
    s    = np.ones(C, np.float32)
    b    = np.zeros(C, np.float32)
    m    = np.zeros(C, np.float32)
    v    = np.ones(C, np.float32)

    pt_out = pt_pfn_forward(x_np, w, s, b, m, v)
    print(f"  PyTorch PFN: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    np_out = pfn_numpy(x_np, w, s, b, m, v)
    print(f"  NumPy  PFN: shape={list(np_out.shape)}, "
          f"sample={np_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, np_out, "PFN full forward", atol=1e-5)
    return ok


def test_hard_simple_vfe_shape():
    print_header("Test 5: HardSimpleVFE output shape (TTSim)")
    vfe = HardSimpleVFE("vfe5", num_features=4)
    ok_p = vfe.analytical_param_count() == 0
    print_test("HardSimpleVFE params == 0", f"got {vfe.analytical_param_count()}")
    x   = _from_shape("vfe_x", [100, 10, 4])
    out = vfe(x)
    ok_s = list(out.shape) == [100, 4]
    print_test("HardSimpleVFE shape [100,10,4]→[100,4]", f"got {list(out.shape)}")
    return ok_p and ok_s


def test_hard_simple_vfe_att_shape():
    print_header("Test 6: HardSimpleVFE_ATT output shape (TTSim)")
    vfe_att = HardSimpleVFE_ATT("vfe_att6", num_features=5)
    p = vfe_att.analytical_param_count()
    print_test("HardSimpleVFE_ATT params", f"got {p:,}")
    x   = _from_shape("vfe_att_x", [200, 10, 5])
    out = vfe_att(x)
    ok  = list(out.shape) == [200, 32]
    print_test(f"HardSimpleVFE_ATT [200,10,5]→[200,32]", f"got {list(out.shape)}")
    return ok


if __name__ == "__main__":
    tests = [
        ("mean_aggregation",          test_mean_aggregation),
        ("pfn_conv1d_step",           test_pfn_conv1d_step),
        ("pfn_bn_step",               test_pfn_bn_step),
        ("pfn_full_forward",          test_pfn_full_forward),
        ("hard_simple_vfe_shape",     test_hard_simple_vfe_shape),
        ("hard_simple_vfe_att_shape", test_hard_simple_vfe_att_shape),
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
