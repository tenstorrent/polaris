#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FPN neck TTSim module.

Three test categories:
  1. Shape Validation  – all output levels have shape [B, out_channels, H_i, W_i].
  2. Edge Case Creation – start_level, extra pooled levels, with_norm/with_act.
  3. Data Validation   – param count formulas and data-carrying inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_fpn.py -v
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
from ttsim_modules.fpn import FPN

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(11)


# (name, in_chs, out_ch, num_outs, start_level, end_level,
#  input_shapes, expected_shapes)
_SHAPE_CASES = [
    (
        "4-level ResNet-like",
        [256, 512, 1024, 2048], 256, 4, 0, -1,
        [[2, 256, 128, 128], [2, 512, 64, 64],
         [2, 1024, 32, 32], [2, 2048, 16, 16]],
        [[2, 256, 128, 128], [2, 256, 64, 64],
         [2, 256, 32, 32],   [2, 256, 16, 16]],
    ),
    (
        "2-level simple",
        [64, 128], 64, 2, 0, -1,
        [[1, 64, 32, 32], [1, 128, 16, 16]],
        [[1, 64, 32, 32], [1, 64, 16, 16]],
    ),
    (
        "start_level=1",
        [64, 128, 256, 512], 128, 3, 1, -1,
        [[2, 64, 128, 128], [2, 128, 64, 64],
         [2, 256, 32, 32],  [2, 512, 16, 16]],
        [[2, 128, 64, 64], [2, 128, 32, 32], [2, 128, 16, 16]],
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_fpn_shape_validation():
    """Category 1 – shape validation."""
    all_passed = True
    for i, (name, in_chs, out_ch, num_outs, sl, el, inp, exp) in enumerate(_SHAPE_CASES):
        try:
            fpn  = FPN(f"fpn_sv{i}", in_channels=in_chs, out_channels=out_ch,
                       num_outs=num_outs, start_level=sl, end_level=el)
            xs   = [_from_shape(f"fpn_sv{i}_x{j}", s) for j, s in enumerate(inp)]
            outs = fpn(*xs)
            ok   = (len(outs) == len(exp) and
                    all(list(o.shape) == e for o, e in zip(outs, exp)))
            print(f"  [{i}] {name:30s} {'PASS' if ok else 'FAIL'}  "
                  f"shapes={[list(o.shape) for o in outs]}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {name:30s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_fpn_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: extra pooled output via max-pool when num_outs > num backbone levels
    fpn = FPN("fpn_ec1", in_channels=[256, 512], out_channels=256, num_outs=3)
    x0  = _from_shape("fpn_ec1_x0", [2, 256, 64, 64])
    x1  = _from_shape("fpn_ec1_x1", [2, 512, 32, 32])
    outs = fpn(x0, x1)
    ok1  = len(outs) == 3
    print(f"  Extra pooled level: num_outs=3  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: with_norm=True (param delta = +out_ch per conv layer)
    fpn_nn = FPN("fpn_ec2_nn", in_channels=[128, 256], out_channels=64,
                 num_outs=2, with_norm=False)
    fpn_wn = FPN("fpn_ec2_wn", in_channels=[128, 256], out_channels=64,
                 num_outs=2, with_norm=True)
    delta_expected = 4 * 64   # 4 conv layers × net +64 per conv (BN adds 2C, bias removed C)
    delta_got = fpn_wn.analytical_param_count() - fpn_nn.analytical_param_count()
    ok2 = delta_got == delta_expected
    print(f"  with_norm delta: got={delta_got} exp={delta_expected}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: single backbone level
    fpn_s = FPN("fpn_ec3", in_channels=[256], out_channels=128, num_outs=1)
    xs    = [_from_shape("fpn_ec3_x0", [2, 256, 32, 32])]
    outs  = fpn_s(*xs)
    ok3   = len(outs) == 1 and list(outs[0].shape) == [2, 128, 32, 32]
    print(f"  Single level: {list(outs[0].shape)}  {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_fpn_data_validation():
    """Category 3 – param count formula and data-carrying inputs."""
    all_passed = True

    # Param count: lateral = sum(in_i*C+C), fpn = num_used*(C*C*9+C) where C = out_channels
    in_chs, out_ch = [256, 512, 1024, 2048], 256
    fpn  = FPN("fpn_dv_pc", in_channels=in_chs, out_channels=out_ch, num_outs=4,
               with_norm=False)
    lat  = sum(c * out_ch + out_ch for c in in_chs)
    fpnc = len(in_chs) * (out_ch * out_ch * 9 + out_ch)
    exp  = lat + fpnc
    got  = fpn.analytical_param_count()
    ok   = got == exp
    print(f"  Param count (4-level): got={got:,} exp={exp:,}  {'PASS' if ok else 'FAIL'}")
    all_passed = all_passed and ok

    # Data input
    fpn2  = FPN("fpn_dv2", in_channels=[64, 128], out_channels=64, num_outs=2)
    xs_np = [
        rng.randn(1, 64, 16, 16).astype(np.float32),
        rng.randn(1, 128, 8, 8).astype(np.float32),
    ]
    outs  = fpn2(*[_from_data(f"fpn_dv2_x{j}", x) for j, x in enumerate(xs_np)])
    ok2   = all(list(o.shape) == [1, 64, 16 >> j, 16 >> j] for j, o in enumerate(outs))
    print(f"  Data input shapes: {[list(o.shape) for o in outs]}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
