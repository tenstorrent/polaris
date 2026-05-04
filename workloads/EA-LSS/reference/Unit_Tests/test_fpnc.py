#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for gapcontext + FPNC neck TTSim modules.

Three test categories:
  1. Shape Validation  – gapcontext and FPNC produce correct output shapes.
  2. Edge Case Creation – various in_channels, use_adp flag, single input.
  3. Data Validation   – param count formulas and _from_data inputs.

Run all:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_fpnc.py -v
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
from ttsim_modules.fpnc import gapcontext, FPNC

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _l; _l.remove(); _l.add(sys.stderr, level="ERROR")
    except Exception: pass
except Exception: pass

rng = np.random.RandomState(7)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fpn_inputs(in_channels, num_feats, B, H, W):
    """Return a list of TTSim tensors mimicking FPN inputs at multiple scales."""
    feats = []
    for i, C in enumerate(in_channels):
        feats.append(_from_shape(f"fpnc_in_{i}", [B, C, H >> i, W >> i]))
    return feats


def _make_fpnc(in_channels, out_channels, num_outs, tag="sv", use_adp=False,
               outC=256, final_dim=(900, 1600), downsample=4):
    return FPNC(
        f"fpnc_{tag}",
        in_channels=in_channels,
        out_channels=out_channels,
        num_outs=num_outs,
        use_adp=use_adp,
        outC=outC,
        final_dim=final_dim,
        downsample=downsample,
    )



# ---------------------------------------------------------------------------
# Category 1 – Shape Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_fpnc_shape_validation():
    """Category 1 – output shapes are deterministic and correct."""
    all_passed = True

    # gapcontext: output channels = out_channels, spatial unchanged
    # gapcontext uses with_norm= not norm=
    cases_gap = [
        ("gap_default [1,32,64,64]", 1, 32,  64, 32),
        ("gap_big [2,128,32,32]",    2, 128, 32, 64),
        ("gap_small [1,16,8,8]",     1, 16,   8, 16),
    ]
    for name, B, C_in, H, C_out in cases_gap:
        try:
            W = H
            gc = gapcontext(f"gc_sv_{C_in}", C_in, C_out, with_norm=True)
            x  = _from_shape(f"gc_sv_{C_in}_in", [B, C_in, H, W])
            o  = gc(x)
            ok = list(o.shape) == [B, C_out, H, W]
            print(f"  gapcontext {name}: got={list(o.shape)} exp=[{B},{C_out},{H},{W}]  {'PASS' if ok else 'FAIL'}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"  gapcontext {name}: ERROR {exc}")
            all_passed = False

    # FPNC: returns a list[1] with shape [B, outC, tH, tW] after reduction conv
    # FPNC is called with *inputs (individual tensors, not a list)
    in_channels = [256, 512, 1024, 2048]
    out_channels_fpn = 64
    outC = 256
    B, H, W = 1, 64, 64
    try:
        fpnc  = _make_fpnc(in_channels, out_channels_fpn, 4, "sv_4", outC=outC)
        feats = _make_fpn_inputs(in_channels, 4, B, H, W)
        outs  = fpnc(*feats)
        ok_len = len(outs) == 1
        print(f"  FPNC num_outs=4 → 1 output after reduc_conv: len={len(outs)}  {'PASS' if ok_len else 'FAIL'}")
        if not ok_len: all_passed = False
        # Output should have outC channels
        ok_c = outs[0].shape[1] == outC
        print(f"  FPNC out shape={list(outs[0].shape)} exp_C={outC}  {'PASS' if ok_c else 'FAIL'}")
        if not ok_c: all_passed = False
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FPNC shape test ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 2 – Edge Cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_fpnc_edge_cases():
    """Category 2 – edge cases: single-level, use_adp, no norm in gapcontext."""
    all_passed = True

    # Edge 1: gapcontext without norm
    try:
        gc = gapcontext("gc_ec_nonorm", 64, 64, with_norm=False)
        x  = _from_shape("gc_ec_nonorm_in", [2, 64, 16, 16])
        o  = gc(x)
        ok = list(o.shape) == [2, 64, 16, 16]
        print(f"  gapcontext(norm=False): {list(o.shape)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"  gapcontext(norm=False) ERROR: {exc}")
        all_passed = False

    # Edge 2: FPNC with single-level FPN input (returns 1 element)
    try:
        fpnc  = _make_fpnc([256], 64, 1, "ec_single", outC=128)
        feat  = _from_shape("fp_ec_single_0", [1, 256, 64, 64])
        outs  = fpnc(feat)
        ok    = len(outs) == 1
        print(f"  FPNC single-level: len={len(outs)}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FPNC single-level ERROR: {exc}")
        all_passed = False

    # Edge 3: FPNC with use_adp=True
    # Use final_dim=(64,64), downsample=4 → target_size=(16,16);
    # feature maps at 64×64, 32×32, 16×16 are all ≥ target so AdaptiveAvgPool2d works
    try:
        in_channels = [256, 512, 1024]
        fpnc  = _make_fpnc(in_channels, 64, 3, "ec_adp", use_adp=True, outC=256,
                           final_dim=(64, 64), downsample=4)
        feats = _make_fpn_inputs(in_channels, 3, 1, 64, 64)
        outs  = fpnc(*feats)
        ok    = len(outs) == 1
        print(f"  FPNC use_adp=True: len={len(outs)} (1 after reduc_conv)  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FPNC use_adp=True ERROR: {exc}")
        all_passed = False

    assert all_passed


# ---------------------------------------------------------------------------
# Category 3 – Data Validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.opunit
def test_fpnc_data_validation():
    """Category 3 – gapcontext param count and _from_data input shapes."""
    all_passed = True

    # gapcontext param formula (with_norm=True, in==out==C):
    #   gap_conv:  C*C  (no bias when with_norm)
    #   gap_bn:    2*C  (scale + bias)
    #   out_conv:  C*C  (no bias when with_norm)
    #   out_bn:    2*C
    # Total: 2*C² + 4*C
    for C in [16, 32, 64]:
        gc       = gapcontext(f"gc_dv_{C}", C, C, with_norm=True)
        expected = 2*C*C + 4*C
        got      = gc.analytical_param_count()
        ok       = got == expected
        print(f"  gapcontext(C={C:3d},with_norm) params: got={got}  exp={expected}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # gapcontext param formula (with_norm=False, in==out==C):
    #   gap_conv:  C*C + C  (with bias)
    #   out_conv:  C*C + C
    # Total: 2*C² + 2*C
    for C in [16, 32]:
        gc       = gapcontext(f"gc_dv_nonorm_{C}", C, C, with_norm=False)
        expected = 2*C*C + 2*C
        got      = gc.analytical_param_count()
        ok       = got == expected
        print(f"  gapcontext(C={C:3d},no_norm) params: got={got}  exp={expected}  {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False

    # _from_data input: gapcontext passes through shapes correctly
    x_np = rng.randn(1, 32, 8, 8).astype(np.float32)
    gc   = gapcontext("gc_dv_data", 32, 64, with_norm=True)
    o    = gc(_from_data("gc_dv_x", x_np))
    ok   = list(o.shape) == [1, 64, 8, 8]
    print(f"  gapcontext _from_data output shape: {list(o.shape)}  {'PASS' if ok else 'FAIL'}")
    if not ok: all_passed = False

    assert all_passed
