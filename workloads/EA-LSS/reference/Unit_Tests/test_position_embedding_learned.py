#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PositionEmbeddingLearned TTSim module.

Three test categories:
  1. Shape Validation  – output is [B, num_pos_feats, P] for various (ic, F, B, P).
  2. Edge Case Creation – 6-D coordinates, F=1, F=512, P=1.
  3. Data Validation   – param count formula ic*F+F+2F+F²+F and data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_position_embedding_learned.py -v
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
from ttsim_modules.position_embedding_learned import PositionEmbeddingLearned

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(17)


# (name, input_channel, num_pos_feats, B, P)
_SHAPE_CASES = [
    ("Default ic=3  F=288 B=2 P=200", 3,   288, 2, 200),
    ("ic=6  F=128   B=1 P=100",        6,   128, 1, 100),
    ("ic=3  F=64    B=4 P=50",         3,    64, 4,  50),
    ("ic=2  F=32    B=1 P=400",        2,    32, 1, 400),
    ("ic=3  F=256   B=2 P=10",         3,   256, 2,  10),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_position_embedding_learned_shape_validation():
    """Category 1 – output shape [B, F, P]."""
    all_passed = True
    for i, (name, ic, F, B, P) in enumerate(_SHAPE_CASES):
        try:
            pel  = PositionEmbeddingLearned(f"pel_sv{i}", input_channel=ic, num_pos_feats=F)
            x    = _from_shape(f"pel_sv{i}_in", [B, P, ic])
            out  = pel(x)
            ok   = list(out.shape) == [B, F, P]
            print(f"  [{i}] {name:35s} {'PASS' if ok else 'FAIL'}  got={list(out.shape)}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {name:35s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_position_embedding_learned_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: P=1 (single point)
    pel = PositionEmbeddingLearned("pel_ec1", input_channel=3, num_pos_feats=64)
    x   = _from_shape("pel_ec1_in", [2, 1, 3])
    out = pel(x)
    ok1 = list(out.shape) == [2, 64, 1]
    print(f"  P=1 single point: {list(out.shape)}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: 6-D coords (lidar + camera concat)
    pel2 = PositionEmbeddingLearned("pel_ec2", input_channel=6, num_pos_feats=288)
    x2   = _from_shape("pel_ec2_in", [1, 200, 6])
    out2 = pel2(x2)
    ok2  = list(out2.shape) == [1, 288, 200]
    print(f"  ic=6 6-D coords: {list(out2.shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: F same as ic → no bottleneck
    pel3 = PositionEmbeddingLearned("pel_ec3", input_channel=64, num_pos_feats=64)
    x3   = _from_shape("pel_ec3_in", [2, 100, 64])
    out3 = pel3(x3)
    ok3  = list(out3.shape) == [2, 64, 100]
    print(f"  ic==F=64: {list(out3.shape)}  {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_position_embedding_learned_data_validation():
    """Category 3 – param count formula and data inputs."""
    all_passed = True

    # Param formula: ic*F + F(bias) + 2*F(BN) + F*F + F(bias) = ic*F + F²+ 4F
    for ic, F in [(3, 64), (6, 128), (3, 288)]:
        pel      = PositionEmbeddingLearned(f"pel_dv_{ic}_{F}",
                                             input_channel=ic, num_pos_feats=F)
        expected = ic * F + F + 2 * F + F * F + F
        got      = pel.analytical_param_count()
        ok       = got == expected
        print(f"  PEL(ic={ic}, F={F}) params: got={got:,} exp={expected:,}  "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # Data input
    ic, F, B, P = 3, 128, 2, 50
    pel   = PositionEmbeddingLearned("pel_dv_data", input_channel=ic, num_pos_feats=F)
    x_np  = rng.randn(B, P, ic).astype(np.float32)
    out   = pel(_from_data("pel_dv_in", x_np))
    ok_d  = list(out.shape) == [B, F, P]
    print(f"  Data input shape: {list(out.shape)}  {'PASS' if ok_d else 'FAIL'}")
    all_passed = all_passed and ok_d

    assert all_passed
