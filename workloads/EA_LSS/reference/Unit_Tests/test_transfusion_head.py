#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TransformerDecoderLayer and TransFusionHead TTSim modules.

Three test categories:
  1. Shape Validation  – decoder output [B,C,Pq] and TransFusionHead pred shapes.
  2. Edge Case Creation – cross_only=True, different d_model, varying proposals.
  3. Data Validation   – TDL param count (233,088 exact) and _from_data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_transfusion_head.py -v
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
from ttsim_modules.transfusion_head import TransformerDecoderLayer, TransFusionHead

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_tdl(tag="sv", cross_only=False, d=128, nhead=8, ffn=256):
    return TransformerDecoderLayer(
        f"tdl_{tag}", d_model=d, nhead=nhead, dim_feedforward=ffn,
        cross_only=cross_only
    )


def _default_tfh(tag="sv", in_channels=1024, hidden=128, num_classes=10,
                 num_proposals=200, num_dec=1):
    return TransFusionHead(
        f"tfh_{tag}",
        in_channels=in_channels,
        hidden_channel=hidden,
        num_classes=num_classes,
        num_proposals=num_proposals,
        num_decoder_layers=num_dec,
        initialize_by_heatmap=True,
        fuse_img=False,
    )


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_head_shape_validation():
    """Category 1 – TransformerDecoderLayer and TransFusionHead output shapes."""
    all_passed = True

    # TDL: [B,C,Pq] output
    tdl_cases = [
        ("B=1,C=128,Pq=200,Pk=100", 1, 128, 200, 100),
        ("B=2,C=128,Pq=50,Pk=400",  2, 128,  50, 400),
    ]
    for name, B, C, Pq, Pk in tdl_cases:
        try:
            tdl   = _default_tdl(f"sv_{B}_{C}_{Pq}", d=C)
            query = _from_shape(f"tdl_sv_q_{Pq}", [B, C, Pq])
            key   = _from_shape(f"tdl_sv_k_{Pk}", [B, C, Pk])
            qpos  = _from_shape(f"tdl_sv_qp_{Pq}", [B, Pq, 2])
            kpos  = _from_shape(f"tdl_sv_kp_{Pk}", [B, Pk, 2])
            out   = tdl(query, key, qpos, kpos)
            ok    = list(out.shape) == [B, C, Pq]
            print(f"  TDL {name}: got={list(out.shape)} exp=[{B},{C},{Pq}]  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  TDL {name}: ERROR {exc}")
            all_passed = False

    # TransFusionHead: output is a dict with prediction keys
    try:
        tfh  = _default_tfh("sv")
        x    = _from_shape("tfh_sv_bev", [1, 1024, 128, 128])
        preds = tfh(x)
        ok_dict = isinstance(preds, dict)
        ok_keys = "center" in preds and "heatmap" in preds
        print(f"  TransFusionHead output is dict: {'PASS' if ok_dict else 'FAIL'}")
        print(f"  TransFusionHead has 'center'+'heatmap' keys: {'PASS' if ok_keys else 'FAIL'}")
        if not ok_dict: all_passed = False
        if not ok_keys: all_passed = False
        for k, v in preds.items():
            print(f"    pred[{k}] shape={list(v.shape)}")
    except Exception as exc:
        print(f"  TransFusionHead shape test ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_head_edge_cases():
    """Category 2 – cross_only=True, varied d_model, TransFusionHead 2 decoder layers."""
    all_passed = True

    # Edge 1: TDL cross_only=True (no self-attention)
    try:
        tdl   = _default_tdl("ec_co", cross_only=True)
        query = _from_shape("tdl_co_q", [1, 128, 200])
        key   = _from_shape("tdl_co_k", [1, 128, 500])
        qpos  = _from_shape("tdl_co_qp", [1, 200, 2])
        kpos  = _from_shape("tdl_co_kp", [1, 500, 2])
        out   = tdl(query, key, qpos, kpos)
        ok    = list(out.shape) == [1, 128, 200]
        print(f"  TDL cross_only=True: got={list(out.shape)} exp=[1,128,200]  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  TDL cross_only=True ERROR: {exc}")
        all_passed = False

    # Edge 2: TDL with non-default d_model=64
    try:
        tdl   = _default_tdl("ec_d64", d=64, nhead=4, ffn=128)
        query = _from_shape("tdl_d64_q", [2, 64, 100])
        key   = _from_shape("tdl_d64_k", [2, 64, 200])
        qpos  = _from_shape("tdl_d64_qp", [2, 100, 2])
        kpos  = _from_shape("tdl_d64_kp", [2, 200, 2])
        out   = tdl(query, key, qpos, kpos)
        ok    = list(out.shape) == [2, 64, 100]
        print(f"  TDL d=64: got={list(out.shape)} exp=[2,64,100]  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  TDL d=64 ERROR: {exc}")
        all_passed = False

    # Edge 3: TransFusionHead with 2 decoder layers
    try:
        tfh   = _default_tfh("ec_2dec", num_dec=2)
        x     = _from_shape("tfh_ec2_bev", [1, 1024, 64, 64])
        preds = tfh(x)
        ok    = isinstance(preds, dict) and len(preds) > 0
        print(f"  TransFusionHead 2-layers: dict with {len(preds)} keys  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  TransFusionHead 2-layers ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_transfusion_head_data_validation():
    """Category 3 – exact TDL param count 233,088 and TransFusionHead params > 0."""
    all_passed = True

    # TDL default (d=128, nhead=8, ffn=256, not cross_only): 233,088
    tdl      = _default_tdl("dv")
    got      = tdl.analytical_param_count()
    expected = 233_088
    ok       = got == expected
    print(f"  TDL params: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # TDL cross_only=True: fewer params (no self_attn, no norm1)
    tdl_co   = _default_tdl("dv_co", cross_only=True)
    got_co   = tdl_co.analytical_param_count()
    ok_co    = got_co < expected
    print(f"  TDL cross_only params ({got_co:,}) < full ({expected:,}): {'PASS' if ok_co else 'FAIL'}")
    if not ok_co: all_passed = False

    # TransFusionHead total params must be positive
    tfh      = _default_tfh("dv")
    got_tfh  = tfh.analytical_param_count()
    ok_tfh   = got_tfh > 0
    print(f"  TransFusionHead params: {got_tfh:,}  {'PASS' if ok_tfh else 'FAIL'}")
    if not ok_tfh: all_passed = False

    # _from_data input to TDL
    q_np  = rng.randn(1, 128, 200).astype(np.float32)
    k_np  = rng.randn(1, 128, 500).astype(np.float32)
    qp_np = rng.randn(1, 200, 2).astype(np.float32)
    kp_np = rng.randn(1, 500, 2).astype(np.float32)
    tdl   = _default_tdl("dv_data")
    out   = tdl(
        _from_data("dv_q", q_np),
        _from_data("dv_k", k_np),
        _from_data("dv_qp", qp_np),
        _from_data("dv_kp", kp_np),
    )
    ok_data = list(out.shape) == [1, 128, 200]
    print(f"  TDL _from_data: shape={list(out.shape)}  {'PASS' if ok_data else 'FAIL'}")
    if not ok_data: all_passed = False

    assert all_passed
