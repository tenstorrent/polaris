#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for PositionEmbeddingLearned TTSim module.
Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_position_embedding_learned.py

Architecture: input [B,P,ic] → Conv1d(ic→F,k=1) → BN1d → ReLU → Conv1d(F→F,k=1) → [B,F,P]
Weight injection via submodule attributes:
    pel.conv0.conv_weight.data, pel.conv0.conv_bias.data
    pel.bn0.scale/bias_bn/running_mean/running_var.data
    pel.conv1.conv_weight.data, pel.conv1.conv_bias.data
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

from ttsim.front.functional.op import _from_shape, _from_data
from ttsim_modules.position_embedding_learned import PositionEmbeddingLearned
from reference.Validation.ttsim_utils import (
    print_header, print_test, compare_arrays,
    conv1d_numpy, batchnorm1d_numpy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inject_pel_weights(pel, w0, b0, bn_scale, bn_bias, bn_mean, bn_var, w1, b1):
    """Inject PyTorch-compatible weights into a PositionEmbeddingLearned module."""
    pel.conv0.conv_weight.data = w0
    pel.conv0.conv_bias.data   = b0
    pel.bn0.scale.data         = bn_scale
    pel.bn0.bias_bn.data       = bn_bias
    pel.bn0.running_mean.data  = bn_mean
    pel.bn0.running_var.data   = bn_var
    pel.conv1.conv_weight.data = w1
    pel.conv1.conv_bias.data   = b1


def pt_pel_forward(x_np, ic, F, w0, b0, bn_scale, bn_bias, bn_mean, bn_var, w1, b1):
    """
    PyTorch reference for PEL:
      input [B,P,ic] → permute to [B,ic,P] → Conv1d→BN→ReLU→Conv1d → [B,F,P]
    """
    # Permute input to [B,ic,P]
    x_pt = torch.tensor(x_np).permute(0, 2, 1)
    c0 = nn.Conv1d(ic, F, 1, bias=True)
    c0.weight.data = torch.tensor(w0)
    c0.bias.data   = torch.tensor(b0)
    bn = nn.BatchNorm1d(F, eps=1e-5, momentum=0.1)
    bn.weight.data = torch.tensor(bn_scale)
    bn.bias.data   = torch.tensor(bn_bias)
    bn.running_mean.data = torch.tensor(bn_mean)
    bn.running_var.data  = torch.tensor(bn_var)
    bn.eval()
    c1 = nn.Conv1d(F, F, 1, bias=True)
    c1.weight.data = torch.tensor(w1)
    c1.bias.data   = torch.tensor(b1)
    with torch.no_grad():
        out = c1(torch.relu(bn(c0(x_pt))))
    return out.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construction_default():
    print_header("Test 1: Construction (default: input_channel=3, num_pos_feats=288)")
    pel = PositionEmbeddingLearned("pel1")
    p = pel.analytical_param_count()
    # Conv0: 3*288 + 288; BN: 2*288; Conv1: 288*288 + 288
    expected = 3*288 + 288 + 2*288 + 288*288 + 288
    ok = p == expected
    print_test("PEL default param count", f"got {p:,} expected {expected:,}")
    return ok


def test_conv1d_step():
    """First Conv1d step comparison: PyTorch vs TTSim conv1d_numpy."""
    print_header("Test 2: Conv1d step — PyTorch vs TTSim")
    rng = np.random.RandomState(30)
    ic, F = 3, 16
    x_np = (rng.randn(2, ic, 10) * 0.5).astype(np.float32)  # [B, ic, P]
    w0 = (rng.randn(F, ic, 1) * 0.1).astype(np.float32)
    b0 = (rng.randn(F) * 0.1).astype(np.float32)

    # PyTorch
    c0 = nn.Conv1d(ic, F, 1, bias=True)
    c0.weight.data = torch.tensor(w0)
    c0.bias.data   = torch.tensor(b0)
    with torch.no_grad():
        pt_out = c0(torch.tensor(x_np)).numpy()
    print(f"  PyTorch Conv1d: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim conv1d_numpy
    ts_out = conv1d_numpy(x_np, w0, b0)
    print(f"  TTSim  Conv1d: shape={list(ts_out.shape)}, "
          f"sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "Conv1d step")
    return ok


def test_bn_step():
    """BN1d step comparison: PyTorch vs TTSim batchnorm1d_numpy."""
    print_header("Test 3: BatchNorm1d step — PyTorch vs TTSim")
    rng = np.random.RandomState(31)
    F = 16
    x_np = (rng.randn(2, F, 10) * 0.5).astype(np.float32)  # [B, F, P]
    bn_scale = (rng.randn(F) * 0.1 + 1.0).astype(np.float32)
    bn_bias  = (rng.randn(F) * 0.1).astype(np.float32)
    bn_mean  = (rng.randn(F) * 0.1).astype(np.float32)
    bn_var   = (np.abs(rng.randn(F)) + 0.5).astype(np.float32)

    # PyTorch
    bn = nn.BatchNorm1d(F, eps=1e-5, momentum=0.1)
    bn.weight.data = torch.tensor(bn_scale)
    bn.bias.data   = torch.tensor(bn_bias)
    bn.running_mean.data = torch.tensor(bn_mean)
    bn.running_var.data  = torch.tensor(bn_var)
    bn.eval()
    with torch.no_grad():
        pt_out = bn(torch.tensor(x_np)).numpy()
    print(f"  PyTorch BN1d: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim batchnorm1d_numpy
    ts_out = batchnorm1d_numpy(x_np, bn_scale, bn_bias, bn_mean, bn_var)
    print(f"  TTSim  BN1d: shape={list(ts_out.shape)}, "
          f"sample={ts_out.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out, "BN1d step", atol=1e-5)
    return ok


def test_full_forward():
    """End-to-end PEL forward: TTSim vs PyTorch."""
    print_header("Test 4: Full PEL forward — PyTorch vs TTSim")
    rng = np.random.RandomState(32)
    ic, F, B, P = 3, 16, 2, 10

    w0 = (rng.randn(F, ic, 1) * 0.1).astype(np.float32)
    b0 = (rng.randn(F) * 0.1).astype(np.float32)
    bn_scale = np.ones(F, np.float32)
    bn_bias  = np.zeros(F, np.float32)
    bn_mean  = np.zeros(F, np.float32)
    bn_var   = np.ones(F, np.float32)
    w1 = (rng.randn(F, F, 1) * 0.1).astype(np.float32)
    b1 = (rng.randn(F) * 0.1).astype(np.float32)
    x_np = (rng.randn(B, P, ic) * 0.5).astype(np.float32)  # [B, P, ic]

    # PyTorch reference
    pt_out = pt_pel_forward(x_np, ic, F, w0, b0, bn_scale, bn_bias, bn_mean, bn_var, w1, b1)
    print(f"  PyTorch output: shape={list(pt_out.shape)}, "
          f"sample={pt_out.flatten()[:4]}")

    # TTSim
    pel = PositionEmbeddingLearned('pel4', input_channel=ic, num_pos_feats=F)
    inject_pel_weights(pel, w0, b0, bn_scale, bn_bias, bn_mean, bn_var, w1, b1)
    x_t = _from_data('x4', x_np)
    ts_out = pel(x_t)
    print(f"  TTSim  output: shape={list(ts_out.shape)}, "
          f"sample={ts_out.data.flatten()[:4]}")

    ok = compare_arrays(pt_out, ts_out.data, "PEL full forward", atol=1e-5)
    return ok


def test_output_shape_default():
    print_header("Test 5: Output shape [B, num_pos_feats, P]")
    B, P = 2, 200
    pel = PositionEmbeddingLearned("pel5", input_channel=3, num_pos_feats=288)
    x = _from_shape("xyz", [B, P, 3])
    out = pel(x)
    expected = [B, 288, P]
    ok = list(out.shape) == expected
    print_test("PEL output shape", f"got {list(out.shape)} expected {expected}")
    return ok


def test_param_count_formula():
    print_header("Test 6: Param count formula for various configs")
    ok = True
    for ic, F in [(3, 64), (6, 128), (3, 288)]:
        pel = PositionEmbeddingLearned(f"pel6_{ic}_{F}", input_channel=ic, num_pos_feats=F)
        expected = ic * F + F + 2 * F + F * F + F   # conv0 + BN + conv1
        got = pel.analytical_param_count()
        match = got == expected
        print_test(f"PEL(ic={ic}, F={F})", f"got {got:,} expected {expected:,}")
        if not match: ok = False
    return ok


if __name__ == "__main__":
    tests = [
        ("construction_default", test_construction_default),
        ("conv1d_step",          test_conv1d_step),
        ("bn_step",              test_bn_step),
        ("full_forward",         test_full_forward),
        ("output_shape_default", test_output_shape_default),
        ("param_count_formula",  test_param_count_formula),
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
