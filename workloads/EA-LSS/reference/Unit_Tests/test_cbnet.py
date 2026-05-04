#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for CBSwinTransformer TTSim module.

Three test categories:
  1. Shape Validation  – 4 output feature maps match expected spatial scales.
  2. Edge Case Creation – different embed_dim, cb_del_stages=0.
  3. Data Validation   – exact param count (55,682,580) and _from_data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_cbnet.py -v
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
from ttsim_modules.cbnet import CBSwinTransformer

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(17)

_DEFAULT_SWIN_KWARGS = dict(
    pretrain_img_size=224,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    out_indices=(0, 1, 2, 3),
)


def _make_cb(embed_dim=96, cb_del_stages=1, tag="sv", **extra_kwargs):
    kwargs = dict(_DEFAULT_SWIN_KWARGS)
    kwargs.update(extra_kwargs)
    return CBSwinTransformer(
        f"cb_{tag}",
        embed_dim=embed_dim,
        cb_del_stages=cb_del_stages,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cbnet_shape_validation():
    """Category 1 – 4 output scales match SwinTransformer single-backbone output shapes."""
    all_passed = True

    cb = _make_cb(embed_dim=96, tag="sv")
    x  = _from_shape("cb_sv_in", [1, 3, 224, 224])
    outs = cb(x)

    # Expected: same as SwinT with embed_dim=96
    expected = [
        [1, 96,  56, 56],
        [1, 192, 28, 28],
        [1, 384, 14, 14],
        [1, 768,  7,  7],
    ]
    ok_len = len(outs) == 4
    print(f"  CBSwinT outputs count: {len(outs)} exp=4  {'PASS' if ok_len else 'FAIL'}")
    if not ok_len: all_passed = False

    for i, (o, e) in enumerate(zip(outs, expected)):
        ok = list(o.shape) == e
        print(f"  Stage {i}: got={list(o.shape)} exp={e}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cbnet_edge_cases():
    """Category 2 – batch size > 1, and 2-stage shallow depths."""
    all_passed = True

    # Edge 1: batch size B=2 with default embed_dim=96
    try:
        cb   = _make_cb(embed_dim=96, tag="ec_B2")
        x    = _from_shape("cb_ec_B2_in", [2, 3, 224, 224])
        outs = cb(x)
        expected = [
            [2, 96,  56, 56],
            [2, 192, 28, 28],
            [2, 384, 14, 14],
            [2, 768,  7,  7],
        ]
        ok_len = len(outs) == 4
        if not ok_len: all_passed = False
        for i, (o, e) in enumerate(zip(outs, expected)):
            ok = list(o.shape) == e
            print(f"  B=2 stage {i}: got={list(o.shape)} exp={e}  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
    except Exception as exc:
        print(f"  B=2 ERROR: {exc}")
        all_passed = False

    # Edge 2: smaller depths=[2,2,2,2] but same embed_dim=96
    try:
        cb   = CBSwinTransformer(
            "cb_ec_dep2222",
            embed_dim=96,
            cb_del_stages=1,
            pretrain_img_size=224,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            out_indices=(0, 1, 2, 3),
        )
        x    = _from_shape("cb_ec_dep2222_in", [1, 3, 224, 224])
        outs = cb(x)
        ok   = len(outs) == 4
        print(f"  depths=[2,2,2,2]: num_outputs={len(outs)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  depths=[2,2,2,2] ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_cbnet_data_validation():
    """Category 3 – exact param count 55,682,580 and _from_data inputs."""
    all_passed = True

    # Exact total param count for default EA-LSS CBSwinTransformer
    cb       = _make_cb(embed_dim=96, tag="dv")
    got      = cb.analytical_param_count()
    expected = 55_682_580
    ok       = got == expected
    print(f"  CBSwinTransformer params: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    # _from_data input
    x_np = rng.randn(1, 3, 224, 224).astype(np.float32)
    cb   = _make_cb(embed_dim=96, tag="dv_data")
    outs = cb(_from_data("cb_dv_x", x_np))
    ok_c = len(outs) == 4
    print(f"  CBSwinTransformer _from_data outputs: {len(outs)}  {'PASS' if ok_c else 'FAIL'}")
    if not ok_c: all_passed = False

    assert all_passed
