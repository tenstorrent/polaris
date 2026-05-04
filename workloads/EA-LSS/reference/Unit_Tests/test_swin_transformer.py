#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SwinTransformer backbone TTSim module.

Three test categories:
  1. Shape Validation  – 4 expected output shapes at each stage scale.
  2. Edge Case Creation – 2-stage only, subset out_indices, different embed_dims.
  3. Data Validation   – exact Swin-T param count (27,520,602) and data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_swin_transformer.py -v
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
from ttsim_modules.swin_transformer import SwinTransformer

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(5)


@pytest.mark.unit
@pytest.mark.opunit
def test_swin_transformer_shape_validation():
    """Category 1 – Swin-T output shapes at all 4 stages."""
    all_passed = True
    swint = SwinTransformer(
        "swt_sv",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    x    = _from_shape("swt_sv_in", [2, 3, 224, 224])
    outs = swint(x)
    expected = [
        [2, 96,  56, 56],
        [2, 192, 28, 28],
        [2, 384, 14, 14],
        [2, 768,  7,  7],
    ]
    for i, (o, e) in enumerate(zip(outs, expected)):
        ok = list(o.shape) == e
        print(f"  Stage {i}: got={list(o.shape)} exp={e}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_swin_transformer_edge_cases():
    """Category 2 – edge cases."""
    all_passed = True

    # Edge 1: 2-stage only
    swint = SwinTransformer(
        "swt_ec1",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=7,
        out_indices=(0, 1),
    )
    x    = _from_shape("swt_ec1_in", [1, 3, 224, 224])
    outs = swint(x)
    ok1  = len(outs) == 2
    print(f"  2-stage outputs: {len(outs)}  {'PASS' if ok1 else 'FAIL'}")
    all_passed = all_passed and ok1

    # Edge 2: subset out_indices=(1,3) only stages 1 and 3
    swint2 = SwinTransformer(
        "swt_ec2",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(1, 3),
    )
    x2    = _from_shape("swt_ec2_in", [2, 3, 224, 224])
    outs2 = swint2(x2)
    ok2   = (len(outs2) == 2 and
             list(outs2[0].shape) == [2, 192, 28, 28] and
             list(outs2[1].shape) == [2, 768,  7,  7])
    print(f"  out_indices=(1,3): {[list(o.shape) for o in outs2]}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    # Edge 3: embed_dim=128 doubles channels
    swint3 = SwinTransformer(
        "swt_ec3",
        pretrain_img_size=224,
        embed_dim=128,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    x3    = _from_shape("swt_ec3_in", [1, 3, 224, 224])
    outs3 = swint3(x3)
    ok3   = list(outs3[0].shape) == [1, 128, 56, 56] and list(outs3[-1].shape) == [1, 1024, 7, 7]
    print(f"  embed_dim=128 stages: {[list(o.shape) for o in outs3]}  {'PASS' if ok3 else 'FAIL'}")
    all_passed = all_passed and ok3

    assert all_passed


@pytest.mark.unit
@pytest.mark.opunit
def test_swin_transformer_data_validation():
    """Category 3 – exact Swin-T param count and data-carrying inputs."""
    all_passed = True

    # Param count == 27,520,602 (Swin-T reference)
    swint = SwinTransformer(
        "swt_dv_pc",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    expected = 27_520_602
    got      = swint.analytical_param_count()
    ok       = got == expected
    print(f"  Swin-T param count: got={got:,} exp={expected:,}  {'PASS' if ok else 'FAIL'}")
    all_passed = all_passed and ok

    # Data-carrying input: just verify shape
    x_np  = rng.randn(1, 3, 224, 224).astype(np.float32)
    swint2 = SwinTransformer(
        "swt_dv_data",
        pretrain_img_size=224,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    )
    outs  = swint2(_from_data("swt_dv_in", x_np))
    ok2   = list(outs[0].shape) == [1, 96, 56, 56] and list(outs[-1].shape) == [1, 768, 7, 7]
    print(f"  Data input shapes: {[list(o.shape) for o in outs]}  {'PASS' if ok2 else 'FAIL'}")
    all_passed = all_passed and ok2

    assert all_passed
