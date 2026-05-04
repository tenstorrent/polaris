#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MultiheadAttention TTSim module.

Three test categories:
  1. Shape Validation  – output shape [L, N, E] for self- and cross-attention.
  2. Edge Case Creation – embed_dim=1, num_heads=1, large embed_dim, long sequences.
  3. Data Validation   – param count formula 4E²+4E and data-carrying inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_multihead_attention.py -v
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
from ttsim_modules.multihead_attention import MultiheadAttention

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(3)


# (name, embed_dim, num_heads, L, N, S)
_SHAPE_CASES = [
    ("E=128 H=4 L=200 N=2 S=200",  128, 4,  200, 2, 200),
    ("E=256 H=8 L=100 N=1 S=100",  256, 8,  100, 1, 100),
    ("E=64  H=2 L=50  N=4 S=100",   64, 2,   50, 4, 100),
    ("E=32  H=1 L=10  N=2 S=10",    32, 1,   10, 2,  10),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_shape_validation():
    """Category 1 – output shape [L, N, E]."""
    all_passed = True
    for i, (name, E, H, L, N, S) in enumerate(_SHAPE_CASES):
        try:
            mha  = MultiheadAttention(f"mha_sv{i}", embed_dim=E, num_heads=H)
            q    = _from_shape(f"mha_sv{i}_q", [L, N, E])
            k    = _from_shape(f"mha_sv{i}_k", [S, N, E])
            v    = _from_shape(f"mha_sv{i}_v", [S, N, E])
            out, _ = mha(q, k, v)
            ok   = list(out.shape) == [L, N, E]
            print(f"  [{i}] {name:30s} {'PASS' if ok else 'FAIL'}  got={list(out.shape)}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  [{i}] {name:30s} ERROR: {exc}")
            all_passed = False
    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: E=16, single head
    mha = MultiheadAttention("mha_ec1", embed_dim=16, num_heads=1)
    q = _from_shape("mha_ec1_q", [10, 2, 16])
    out, _ = mha(q, q, q)
    ok1 = list(out.shape) == [10, 2, 16]
    print(f"  E=16 H=1: {list(out.shape)}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: large embed_dim
    mha2 = MultiheadAttention("mha_ec2", embed_dim=512, num_heads=8)
    q2   = _from_shape("mha_ec2_q", [100, 1, 512])
    out2, _ = mha2(q2, q2, q2)
    ok2  = list(out2.shape) == [100, 1, 512]
    print(f"  E=512 H=8: {list(out2.shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: L ≠ S (cross-attention)
    mha3 = MultiheadAttention("mha_ec3", embed_dim=64, num_heads=4)
    q3   = _from_shape("mha_ec3_q", [50, 2, 64])
    k3   = _from_shape("mha_ec3_k", [200, 2, 64])
    v3   = _from_shape("mha_ec3_v", [200, 2, 64])
    out3, _ = mha3(q3, k3, v3)
    ok3  = list(out3.shape) == [50, 2, 64]
    print(f"  Cross-attn L=50 S=200: {list(out3.shape)}  {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_multihead_attention_data_validation():
    """Category 3 – param count formula and data-carrying inputs."""
    all_passed = True

    # Param count: 4E² + 4E
    for E in [32, 64, 128, 256]:
        mha      = MultiheadAttention(f"mha_dv_e{E}", embed_dim=E, num_heads=4)
        expected = 4 * E * E + 4 * E
        got      = mha.analytical_param_count()
        ok       = got == expected
        print(f"  MHA(E={E:3d}) params: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # Data input
    E, L, N = 128, 50, 2
    mha  = MultiheadAttention("mha_dv_data", embed_dim=E, num_heads=4)
    q_np = rng.randn(L, N, E).astype(np.float32)
    out, _ = mha(_from_data("mha_dv_q", q_np),
                 _from_data("mha_dv_k", q_np),
                 _from_data("mha_dv_v", q_np))
    ok2  = list(out.shape) == [L, N, E]
    print(f"  Data input shape: {list(out.shape)}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
