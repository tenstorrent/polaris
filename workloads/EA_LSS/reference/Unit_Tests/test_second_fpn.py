#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SECONDFPN neck TTSim module.

Three test categories:
  1. Shape Validation  – output is ConcatX of N deblocks; H/W matches stride.
  2. Edge Case Creation – stride=1, single input, 4-scale fusion.
  3. Data Validation   – param count formula (no bias) and data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_second_fpn.py -v
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
from ttsim_modules.second_fpn import SECONDFPN

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(99)


# (name, in_chs, out_chs, strides, input_shapes, expected_out_shape)
_SHAPE_CASES = [
    (
        "Default 3-scale stride[1,2,4]",
        [128, 128, 256], [256, 256, 256], [1, 2, 4],
        [[2, 128, 64, 64], [2, 128, 32, 32], [2, 256, 16, 16]],
        [2, 768, 64, 64],
    ),
    (
        "2-scale stride[1,2]",
        [64, 128], [128, 128], [1, 2],
        [[1, 64, 32, 32], [1, 128, 16, 16]],
        [1, 256, 32, 32],
    ),
    (
        "Single input stride=1",
        [128], [256], [1],
        [[2, 128, 16, 16]],
        [2, 256, 16, 16],
    ),
    (
        "Single input stride=2",
        [128], [256], [2],
        [[2, 128, 8, 8]],
        [2, 256, 16, 16],
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_second_fpn_shape_validation():
    """Category 1 – shape validation."""
    all_passed = True
    for i, (name, in_chs, out_chs, strides, inp_shapes, exp) in enumerate(_SHAPE_CASES):
        try:
            sfpn = SECONDFPN(f"sfpn_sv{i}",
                             in_channels=in_chs, out_channels=out_chs,
                             upsample_strides=strides)
            xs  = [_from_shape(f"sfpn_sv{i}_x{j}", s) for j, s in enumerate(inp_shapes)]
            out = sfpn(*xs)
            ok  = list(out.shape) == exp
            print(f"  [{i}] {name:35s} {'PASS' if ok else 'FAIL'}  "
                  f"got={list(out.shape)} exp={exp}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {name:35s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_second_fpn_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: 4-scale fusion
    sfpn = SECONDFPN("sfpn_4s", in_channels=[64, 128, 256, 512],
                     out_channels=[128, 128, 128, 128], upsample_strides=[1, 2, 4, 8])
    xs  = [
        _from_shape("sfpn_4s_x0", [1,  64, 64, 64]),
        _from_shape("sfpn_4s_x1", [1, 128, 32, 32]),
        _from_shape("sfpn_4s_x2", [1, 256, 16, 16]),
        _from_shape("sfpn_4s_x3", [1, 512,  8,  8]),
    ]
    out  = sfpn(*xs)
    exp  = [1, 512, 64, 64]
    ok1  = list(out.shape) == exp
    print(f"  4-scale fusion: got={list(out.shape)} exp={exp}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: output channel count is sum(out_channels)
    sfpn2 = SECONDFPN("sfpn_sum", in_channels=[128, 128, 256],
                      out_channels=[64, 128, 256], upsample_strides=[1, 2, 4])
    xs2  = [
        _from_shape("sfpn_sum_x0", [2, 128, 32, 32]),
        _from_shape("sfpn_sum_x1", [2, 128, 16, 16]),
        _from_shape("sfpn_sum_x2", [2, 256,  8,  8]),
    ]
    out2 = sfpn2(*xs2)
    ok2  = out2.shape[1] == sum([64, 128, 256])
    print(f"  Sum channels: got={out2.shape[1]} exp={sum([64, 128, 256])}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_second_fpn_data_validation():
    """Category 3 – param count formula and data inputs."""
    all_passed = True

    # Param count: each deblock = ConvTranspose2d(in_i * out_i * k_i^2, no bias) + BN(2*out_i)
    in_chs, out_chs, strides = [128, 128, 256], [256, 256, 256], [1, 2, 4]
    sfpn = SECONDFPN("sfpn_pc", in_channels=in_chs, out_channels=out_chs,
                     upsample_strides=strides)
    expected = sum(
        i * o * s * s + 2 * o
        for i, o, s in zip(in_chs, out_chs, strides)
    )
    got = sfpn.analytical_param_count()
    ok  = got == expected
    print(f"  Param count: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
    all_passed = all_passed and ok

    # Data-carrying input
    sfpn2 = SECONDFPN("sfpn_dv", in_channels=[64], out_channels=[128], upsample_strides=[2])
    x_np  = rng.randn(1, 64, 8, 8).astype(np.float32)
    out   = sfpn2(_from_data("sfpn_dv_x", x_np))
    ok2   = list(out.shape) == [1, 128, 16, 16]
    print(f"  Data input shape: {list(out.shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
