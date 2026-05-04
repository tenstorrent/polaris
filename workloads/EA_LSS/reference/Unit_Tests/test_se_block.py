#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SE_Block TTSim module.

Three test categories:
  1. Shape Validation  – output shape always equals input shape [B, C, H, W].
  2. Edge Case Creation – C=1, large C, non-square spatial dims, B=1.
  3. Data Validation   – param count formula C²+C and shape-with-data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_se_block.py -v
"""

import os, sys, logging
import numpy as np
import pytest

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

from ttsim.front.functional.op import _from_data, _from_shape
from ttsim_modules.se_block import SE_Block

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(42)


# Shape cases: (name, B, C, H, W)
_SHAPE_CASES = [
    ("Default [2,64,32,32]",     2,  64, 32, 32),
    ("B=1 [1,128,8,8]",          1, 128,  8,  8),
    ("B=4 [4,32,16,16]",         4,  32, 16, 16),
    ("Non-square [2,64,20,40]",  2,  64, 20, 40),
    ("Large C [1,256,4,4]",      1, 256,  4,  4),
    ("C=16 [2,16,64,64]",        2,  16, 64, 64),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_se_block_shape_validation():
    """Category 1 – output shape always equals input shape."""
    all_passed = True
    for i, (name, B, C, H, W) in enumerate(_SHAPE_CASES):
        try:
            blk = SE_Block(f"se_sv{i}", channels=C)
            x   = _from_shape(f"se_sv{i}_in", [B, C, H, W])
            out = blk(x)
            ok  = list(out.shape) == [B, C, H, W]
            print(f"  [{i:02d}] {name:30s} {'PASS' if ok else 'FAIL'}  shape={out.shape}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:30s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_se_block_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True
    cases = [
        ("C=1",            1,  1, 8, 8,  1*1 + 1),
        ("C=4",            2,  4, 4, 4,  4*4 + 4),
        ("H=W=1 (1×1 featuremap)", 2, 32, 1, 1, 32*32 + 32),
        ("C=512 large",    1, 512, 2, 2, 512*512 + 512),
    ]
    for i, (name, B, C, H, W, exp_params) in enumerate(cases):
        try:
            blk = SE_Block(f"se_ec{i}", channels=C)
            x   = _from_shape(f"se_ec{i}_in", [B, C, H, W])
            out = blk(x)
            shape_ok  = list(out.shape) == [B, C, H, W]
            param_ok  = blk.analytical_param_count() == exp_params
            ok = shape_ok and param_ok
            print(f"  [{i:02d}] {name:35s} {'PASS' if ok else 'FAIL'}  "
                  f"shape={out.shape} params={blk.analytical_param_count()} exp={exp_params}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i:02d}] {name:35s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_se_block_data_validation():
    """Category 3 – param count formula and data-carrying inputs."""
    all_passed = True
    # Param count: C*C + C = C*(C+1)
    for C in [16, 32, 64, 128, 256]:
        blk      = SE_Block(f"se_dv_c{C}", channels=C)
        expected = C * C + C
        got      = blk.analytical_param_count()
        ok       = got == expected
        print(f"  SE_Block(C={C:3d}).params  got={got:,}  expected={expected:,}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # Shape preservation with _from_data input
    x_np = rng.randn(2, 64, 8, 8).astype(np.float32)
    blk  = SE_Block("se_dv_data", channels=64)
    out  = blk(_from_data("se_dv_x", x_np))
    ok   = list(out.shape) == [2, 64, 8, 8]
    print(f"  SE_Block data input shape OK: {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    assert all_passed
