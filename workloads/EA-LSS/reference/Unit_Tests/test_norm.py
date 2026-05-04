#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for norm TTSim module.

Three test categories:
  1. Shape Validation  – BN1d (2D and 3D inputs), BN2d, BN3d shape preservation.
  2. Edge Case Creation – single-element batch, C=1, large C, 5-D input (BN3d).
  3. Data Validation   – numerical correctness vs torch.nn.BatchNorm*.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_norm.py -v
"""

import os
import sys
import logging

import numpy as np
import pytest
import torch
import torch.nn as tn

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.norm import (
    NaiveSyncBatchNorm1d, NaiveSyncBatchNorm2d, NaiveSyncBatchNorm3d
)

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _ll; _ll.remove(); _ll.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(13)


def _rw(*s): return (rng.randn(*s).astype(np.float32) * 0.2 + 1.0)
def _rp(*s): return (rng.rand(*s).astype(np.float32) * 0.5 + 0.5)   # positive

def _inject(mod, C, is_1d=True):
    inner = mod._bn if is_1d else mod
    inner.scale.data        = _rw(C)
    inner.bias_bn.data      = rng.randn(C).astype(np.float32) * 0.1
    inner.running_mean.data = rng.randn(C).astype(np.float32) * 0.1
    inner.running_var.data  = _rp(C)

def _pt_bn(x_np, C, scale, bias, rm, rv, nd, eps=1e-5, momentum=0.1):
    cls = {1: tn.BatchNorm1d, 2: tn.BatchNorm2d, 3: tn.BatchNorm3d}[nd]
    m = cls(C, eps=eps, momentum=momentum)
    m.weight.data[:] = torch.tensor(scale)
    m.bias.data[:]   = torch.tensor(bias)
    m.running_mean.data[:] = torch.tensor(rm)
    m.running_var.data[:]  = torch.tensor(rv)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(x_np)).numpy()


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

_SHAPE_CASES = [
    # (name, cls, input_shape, nd)
    ("BN1d  [N,C]",        NaiveSyncBatchNorm1d, [8, 16],     1),
    ("BN1d  [N,C,L]",      NaiveSyncBatchNorm1d, [4, 16, 50], 1),
    ("BN1d  [1,C,L]",      NaiveSyncBatchNorm1d, [1, 8, 10],  1),
    ("BN2d  [N,C,H,W]",    NaiveSyncBatchNorm2d, [3, 16, 8, 8], 2),
    ("BN2d  [1,C,14,14]",  NaiveSyncBatchNorm2d, [1, 24, 14, 14], 2),
    ("BN3d  [N,C,D,H,W]",  NaiveSyncBatchNorm3d, [2, 4, 4, 8, 8], 3),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_norm_shape_validation():
    """Category 1 – Shape Validation."""
    all_passed = True
    for i, (name, cls, shape, nd) in enumerate(_SHAPE_CASES):
        try:
            C = shape[1]
            mod = cls(f"bn_sv{i}", C)
            x_s = _from_shape(f"bn_sv{i}_in", shape)
            out = mod(x_s)
            ok  = list(out.shape) == shape
            x_d = _from_data(f"bn_dv{i}_in",
                              rng.randn(*shape).astype(np.float32))
            out_d = mod(x_d)
            ok = ok and list(out_d.shape) == shape
            # Note: out_d.data may be None until weights are injected; that
            # is expected here (shape validation only).
            print(f"  [{i:02d}] {name:25s} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:25s} ERROR: {exc}")
            all_passed = False
    assert all_passed


# ===========================================================================
# Category 2 – Edge Cases
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_norm_edge_cases():
    """Category 2 – Edge Cases."""
    all_passed = True

    cases = [
        ("BN1d C=1 [2,1]",        NaiveSyncBatchNorm1d, [2, 1]),
        ("BN1d large C [1,256,L]", NaiveSyncBatchNorm1d, [1, 256, 20]),
        ("BN2d C=1",               NaiveSyncBatchNorm2d, [2, 1, 4, 4]),
        ("BN3d C=2",               NaiveSyncBatchNorm3d, [1, 2, 2, 4, 4]),
        ("BN1d param_count=2C",    NaiveSyncBatchNorm1d, None),   # param count
    ]

    for i, (name, cls, shape) in enumerate(cases):
        try:
            if shape is None:
                # Param count test
                C = 32
                mod = cls(f"bn_ec_pc", C)
                pc  = mod.analytical_param_count()
                ok  = pc == 2 * C
                print(f"  [{i:02d}] {name:30s} {'PASS' if ok else 'FAIL'}  pc={pc}")
            else:
                C   = shape[1]
                mod = cls(f"bn_ec{i}", C)
                x_s = _from_shape(f"bn_ec{i}_in", shape)
                out = mod(x_s)
                ok  = list(out.shape) == shape
                print(f"  [{i:02d}] {name:30s} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:30s} ERROR: {exc}")
            all_passed = False

    assert all_passed


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_norm_data_validation():
    """Category 3 – Data Validation vs torch.nn.BatchNorm*."""
    all_passed = True

    cases = [
        ("BN1d 2D",  NaiveSyncBatchNorm1d, [5, 12],    1, True,  1e-3, 0.01),
        ("BN1d 3D",  NaiveSyncBatchNorm1d, [5, 12, 20],1, True,  1e-3, 0.01),
        ("BN2d",     NaiveSyncBatchNorm2d, [3, 8, 6, 6], 2, False, 1e-5, 0.1),
        ("BN3d",     NaiveSyncBatchNorm3d, [2, 4, 3, 4, 4], 3, False, 1e-5, 0.1),
    ]

    for i, (name, cls, shape, nd, is_1d, eps, mom) in enumerate(cases):
        try:
            C   = shape[1]
            mod = cls(f"bn_data{i}", C, eps=eps, momentum=mom)
            _inject(mod, C, is_1d=is_1d)

            x_np = rng.randn(*shape).astype(np.float32)
            out  = mod(_from_data(f"bn_data{i}_in", x_np))
            assert out.data is not None, "data is None"

            inner = mod._bn if is_1d else mod
            ref = _pt_bn(x_np, C,
                         inner.scale.data, inner.bias_bn.data,
                         inner.running_mean.data, inner.running_var.data,
                         nd, eps=eps, momentum=mom)
            max_diff = float(np.max(np.abs(ref - out.data)))
            ok = np.allclose(ref, out.data, atol=1e-4)
            print(f"  [{i:02d}] {name:12s} {'PASS' if ok else 'FAIL'}  max_diff={max_diff:.3e}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:12s} ERROR: {exc}")
            all_passed = False

    assert all_passed
