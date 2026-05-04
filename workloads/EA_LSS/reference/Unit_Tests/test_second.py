#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SECOND backbone TTSim module.

Three test categories:
  1. Shape Validation  – output list has correct shapes for various configs.
  2. Edge Case Creation – single stage, deep stages, in_channels variation.
  3. Data Validation   – param count formula and data-carrying inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_second.py -v
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
from ttsim_modules.second import SECOND, SECONDStage

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(7)


# Shape cases: (name, in_ch, out_chs, layer_nums, strides, input_shape, expected_shapes)
_SHAPE_CASES = [
    (
        "Default 3-stage",
        128, [128, 128, 256], [3, 5, 5], [2, 2, 2],
        [1, 128, 256, 256],
        [[1, 128, 128, 128], [1, 128, 64, 64], [1, 256, 32, 32]],
    ),
    (
        "2-stage small",
        64, [64, 128], [2, 3], [2, 2],
        [2, 64, 64, 64],
        [[2, 64, 32, 32], [2, 128, 16, 16]],
    ),
    (
        "Single stage",
        32, [32], [1], [2],
        [1, 32, 32, 32],
        [[1, 32, 16, 16]],
    ),
    (
        "Stride 1 (no spatial reduction)",
        16, [16], [2], [1],
        [2, 16, 16, 16],
        [[2, 16, 16, 16]],
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_second_shape_validation():
    """Category 1 – shape validation."""
    all_passed = True
    for i, (name, in_ch, out_chs, lnums, strides, inp_shape, exp_shapes) in enumerate(_SHAPE_CASES):
        try:
            model = SECOND(f"sec_sv{i}", in_channels=in_ch, out_channels=out_chs,
                           layer_nums=lnums, layer_strides=strides)
            x    = _from_shape(f"sec_sv{i}_in", inp_shape)
            outs = model(x)
            ok   = len(outs) == len(exp_shapes) and all(
                list(o.shape) == e for o, e in zip(outs, exp_shapes)
            )
            print(f"  [{i}] {name:35s} {'PASS' if ok else 'FAIL'}  "
                  f"shapes={[list(o.shape) for o in outs]}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {name:35s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_second_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: SECONDStage param count with layer_num=0 and layer_num=2
    stage0 = SECONDStage("edge_s0", 64, 128, stride=2, layer_num=0, bias=False)
    # Only the initial strided conv + BN: 3x3 kernel, 64→128
    exp0 = 9 * 64 * 128 + 2 * 128
    ok0  = stage0.analytical_param_count() == exp0
    print(f"  SECONDStage(ln=0) params: got={stage0.analytical_param_count()} "
          f"exp={exp0}  {'PASS' if ok0 else 'FAIL'}")
    all_passed = all_passed and ok0

    # Edge 2: SECONDStage with layer_num=2 (1 extra conv block on top of the 1 initial)
    stage2 = SECONDStage("edge_s2", 64, 128, stride=1, layer_num=2, bias=False)
    # Initial conv: 9*64*128 + 2*128
    # 2 more convs: 2 × (9*128*128 + 2*128)
    exp2 = (9 * 64 * 128 + 2 * 128) + 2 * (9 * 128 * 128 + 2 * 128)
    ok2  = stage2.analytical_param_count() == exp2
    print(f"  SECONDStage(ln=2) params: got={stage2.analytical_param_count()} "
          f"exp={exp2}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: num_outputs == len(out_channels)
    model = SECOND("edge_m", in_channels=16, out_channels=[16, 32, 64],
                   layer_nums=[1, 1, 1], layer_strides=[2, 2, 2])
    x    = _from_shape("edge_in", [1, 16, 32, 32])
    outs = model(x)
    ok3  = len(outs) == 3
    print(f"  num_outputs==3: {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_second_data_validation():
    """Category 3 – param count and data-carrying inputs."""
    all_passed = True

    # Data-carrying input shape check
    x_np  = rng.randn(2, 64, 32, 32).astype(np.float32)
    model = SECOND("sec_dv", in_channels=64, out_channels=[64, 128],
                   layer_nums=[1, 2], layer_strides=[2, 2])
    outs  = model(_from_data("sec_dv_in", x_np))
    ok    = list(outs[0].shape) == [2, 64, 16, 16] and list(outs[1].shape) == [2, 128, 8, 8]
    print(f"  Data-input shapes: {[list(o.shape) for o in outs]}  {'PASS' if ok else 'FAIL'}")
    all_passed = all_passed and ok

    # Param count check for 2-stage config
    model2 = SECOND("sec_pc", in_channels=32, out_channels=[32, 64],
                    layer_nums=[0, 1], layer_strides=[2, 2])
    exp = (  # stage0: initial(32→32,k=3,s=2)+BN
              9 * 32 * 32 + 2 * 32
              # stage1: initial(32→64,k=3,s=2)+BN + 1 extra(64→64,k=3)+BN
            + 9 * 32 * 64 + 2 * 64
            + 9 * 64 * 64 + 2 * 64
          )
    got   = model2.analytical_param_count()
    ok2   = got == exp
    print(f"  Param count 2-stage: got={got} exp={exp}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
