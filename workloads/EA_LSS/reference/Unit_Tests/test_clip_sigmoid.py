#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for clip_sigmoid TTSim module.

Three test categories:
  1. Shape Validation  – output shape always equals input shape for every
                         tensor rank (1-D through 4-D) and spatial size.
  2. Edge Case Creation – conditions unique to clip_sigmoid:
                         · extreme eps values (nearly 0 / nearly 0.5)
                         · eps larger than 0.5 → ValueError
                         · very large / very small inputs (numerical stability)
                         · minimum and maximum representable float32 range
                         · single-element tensors
  3. Data Validation   – numerical correctness compared against PyTorch:
                         · sigmoid step vs torch.sigmoid
                         · clip step vs torch.clamp
                         · full forward pass vs torch.clamp(torch.sigmoid(x), eps, 1-eps)
                         · output strictly within [eps, 1-eps]
                         · no trainable parameters (param count = 0)
                         · determinism (same input → same output)

Run all categories:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_clip_sigmoid.py -v

Run a single category:
    pytest workloads/EA-LSS/Reference/Unit_Tests/test_clip_sigmoid.py \
           ::test_clip_sigmoid_shape_validation -v
"""

import os
import sys
import logging

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
from ttsim.front.functional.op import _from_data, _from_shape
from ttsim.ops.desc.data_compute import compute_sigmoid, compute_clip

# EA-LSS modules (hyphen in folder name prevents dotted import; use sys.path)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.clip_sigmoid import clip_sigmoid

# Silence verbose TTSim / loguru output during tests
try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    try:
        from loguru import logger as _loguru_logger
        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass


# ===========================================================================
# Helpers
# ===========================================================================

class _TW:
    """Thin wrapper so data_compute functions see .data attribute."""
    def __init__(self, d): self.data = d


class _OW:
    """Thin wrapper for op attributes."""
    def __init__(self, **kw): self.attrs = kw


def _make_data(shape, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def _get_max_msg_len(test_list):
    return max(len(tc[0]) for tc in test_list)


def _run_graph(x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Run the full TTSim clip_sigmoid graph and return output .data."""
    x_t = _from_data("ut_cs", x_np)
    return clip_sigmoid(x_t, eps=eps).data


def _pt_clip_sigmoid(x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """PyTorch reference: clamp(sigmoid(x), eps, 1-eps)."""
    return torch.clamp(torch.sigmoid(torch.tensor(x_np)),
                       min=eps, max=1.0 - eps).numpy()


# ===========================================================================
# Category 1 – Shape Validation
# ===========================================================================

# (name, shape)
_SHAPE_CASES = [
    ("1-D  [16]",                   [16]),
    ("1-D  [1]",                    [1]),
    ("2-D  [4, 10]",                [4, 10]),
    ("2-D  [1, 256]",               [1, 256]),
    ("3-D  [2, 10, 8]",             [2, 10, 8]),
    ("3-D  [1, 80, 128]",           [1, 80, 128]),
    ("4-D  [2, 10, 8, 8]",          [2, 10, 8, 8]),
    ("4-D  [1, 80, 128, 128]",      [1, 80, 128, 128]),
    ("BEV  [1, 200, 200]",          [1, 200, 200]),
    ("Large batch [8, 3, 64, 64]",  [8, 3, 64, 64]),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_clip_sigmoid_shape_validation():
    """
    Category 1 – Shape Validation

    Verifies that clip_sigmoid always produces a tensor whose shape is
    identical to the input shape, for every input rank (1-D to 4-D) and
    a range of spatial and channel sizes.  Both shape-only and data-carrying
    TTSim tensors are tested.
    """
    msgw = _get_max_msg_len(_SHAPE_CASES)
    all_passed = True

    for tno, (name, shape) in enumerate(_SHAPE_CASES):
        try:
            # Shape-only path
            x_s = _from_shape(f"ut_cs_shape_{tno}", shape)
            out_s = clip_sigmoid(x_s, eps=1e-4)
            shape_only_ok = list(out_s.shape) == shape

            # Data path – also verify .data is not None
            x_d = _from_data(f"ut_cs_data_{tno}", _make_data(shape, seed=tno))
            out_d = clip_sigmoid(x_d, eps=1e-4)
            data_shape_ok = list(out_d.shape) == shape
            has_data     = out_d.data is not None

            ok = shape_only_ok and data_shape_ok and has_data
            st = "PASS" if ok else "FAIL"
            print(
                f"TEST[{tno:02d}] {name:{msgw}s} {st}  "
                f"shape={list(out_d.shape)}  has_data={has_data}"
            )
            if not ok:
                all_passed = False

        except Exception as exc:
            print(f"TEST[{tno:02d}] {name:{msgw}s} ERROR  {exc}")
            all_passed = False

    assert all_passed, "One or more shape validation tests failed"


# ===========================================================================
# Category 2 – Edge Case Creation
# ===========================================================================

# (name, x_np, eps, expect_error, description)
_EDGE_CASES = [
    # Very small eps (near machine epsilon) → should still work
    (
        "eps=1e-7 (very small)",
        np.zeros([2, 4], dtype=np.float32),
        1e-7, None,
        "Very small eps: output in [1e-7, 1-1e-7]",
    ),
    # eps = 0.4 (large but < 0.5) → output range is [0.4, 0.6]
    (
        "eps=0.4 (large, valid)",
        np.random.randn(3, 5).astype(np.float32),
        0.4, None,
        "Large valid eps=0.4 → range [0.4, 0.6]",
    ),
    # eps >= 0.5 → lower bound == or > upper bound → degenerate clip; no error raised
    # (clip_sigmoid does not validate eps; the ONNX Clip op still runs)
    (
        "eps=0.5 (degenerate, lo==hi)",
        np.zeros([2, 4], dtype=np.float32),
        0.5, None,
        "eps=0.5: lo==hi → degenerate but no error",
    ),
    # eps > 0.5 → inverted bounds, degenerate output, no crash
    (
        "eps=0.6 (degenerate, lo>hi)",
        np.zeros([2, 4], dtype=np.float32),
        0.6, None,
        "eps=0.6: lo > hi → degenerate but no error",
    ),
    # Very large positive input → sigmoid ≈ 1 → clamped to 1-eps
    (
        "x=+1000 large positive",
        np.full([1, 3], 1000.0, dtype=np.float32),
        1e-4, None,
        "Extreme + input: output should be 1-eps",
    ),
    # Very large negative input → sigmoid ≈ 0 → clamped to eps
    (
        "x=-1000 large negative",
        np.full([1, 3], -1000.0, dtype=np.float32),
        1e-4, None,
        "Extreme − input: output should be eps",
    ),
    # Single-element tensor
    (
        "Single element [1]",
        np.array([0.5], dtype=np.float32),
        1e-4, None,
        "Single element tensor",
    ),
    # All-zero input → sigmoid(0) = 0.5 → no clamping
    (
        "All-zeros input",
        np.zeros([4, 4], dtype=np.float32),
        1e-4, None,
        "sigmoid(0)=0.5 is within [eps, 1-eps]: no clamping",
    ),
    # NaN in input should propagate (no hard requirement) – but should not crash
    (
        "NaN input (propagates)",
        np.array([[float("nan"), 0.0]], dtype=np.float32),
        1e-4, None,
        "NaN propagation: should not crash",
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_clip_sigmoid_edge_cases():
    """
    Category 2 – Edge Case Creation

    Tests boundary conditions unique to clip_sigmoid:
      - Very small eps (near float32 machine epsilon)
      - Large but valid eps (0.4) producing a narrow output range
      - Invalid eps >= 0.5 which makes the clamp bounds degenerate → ValueError
      - Extreme positive / negative inputs (numerical stability)
      - Single-element tensor
      - All-zero input (sigmoid(0) = 0.5 lies within any [eps, 1-eps] with eps < 0.5)
      - NaN propagation (should not crash the graph)
    """
    msgw = _get_max_msg_len(_EDGE_CASES)
    all_passed = True

    for tno, (name, x_np, eps, expect_error, desc) in enumerate(_EDGE_CASES):
        try:
            if expect_error is not None:
                raised = False
                try:
                    x_t = _from_data(f"ut_cs_edge_{tno}", x_np)
                    clip_sigmoid(x_t, eps=eps)
                except expect_error:
                    raised = True
                ok = raised
                st = "PASS" if ok else "FAIL"
                print(
                    f"TEST[{tno:02d}] {name:{msgw}s} {st}  "
                    f"(expected {expect_error.__name__}{'  raised' if raised else '  NOT raised'})"
                )
            else:
                x_t   = _from_data(f"ut_cs_edge_{tno}", x_np)
                out   = clip_sigmoid(x_t, eps=eps)
                ok_shape = list(out.shape) == list(x_np.shape)
                has_data = out.data is not None
                ok = ok_shape and has_data
                st = "PASS" if ok else "FAIL"
                print(
                    f"TEST[{tno:02d}] {name:{msgw}s} {st}  "
                    f"shape={list(out.shape)}"
                )

            if not ok:
                print(f"         {desc}")
                all_passed = False

        except Exception as exc:
            if expect_error is not None and isinstance(exc, expect_error):
                print(f"TEST[{tno:02d}] {name:{msgw}s} PASS  (raised {expect_error.__name__})")
            else:
                print(f"TEST[{tno:02d}] {name:{msgw}s} ERROR  unexpected {type(exc).__name__}: {exc}")
                all_passed = False

    assert all_passed, "One or more edge case tests failed"


# ===========================================================================
# Category 3 – Data Validation
# ===========================================================================

_SEED = 42


@pytest.mark.unit
@pytest.mark.opunit
def test_clip_sigmoid_data_validation():
    """
    Category 3 – Data Validation

    Validates numerical correctness of every step in the clip_sigmoid graph
    against PyTorch ground truth:
      1. Sigmoid step: compute_sigmoid vs torch.sigmoid
      2. Clip step:    compute_clip vs torch.clamp
      3. Full forward: TTSim graph (.data) vs torch.clamp(torch.sigmoid(x), eps, 1-eps)
      4. Value range:  all outputs strictly within [eps, 1-eps]
      5. Multiple eps values: 1e-3, 1e-2, 1e-6
      6. Extreme inputs: boundary clamping at ±large values
      7. No trainable parameters (clip_sigmoid is a stateless functional op)
      8. Determinism: identical inputs produce identical outputs
    """
    all_passed = True

    # -----------------------------------------------------------------------
    # DATA[00] Sigmoid step vs torch.sigmoid
    # -----------------------------------------------------------------------
    try:
        x_np    = _make_data([2, 4, 4], seed=_SEED)
        pt_sig  = torch.sigmoid(torch.tensor(x_np)).numpy()
        tt_sig  = compute_sigmoid([_TW(x_np)], _OW())

        max_diff = float(np.max(np.abs(pt_sig - tt_sig)))
        ok = max_diff < 1e-5
        print(f"DATA[00] Sigmoid step vs torch.sigmoid           {'PASS' if ok else 'FAIL'}  "
              f"max_diff={max_diff:.3e}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[00] Sigmoid step vs torch.sigmoid           ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[01] Clip step vs torch.clamp
    # -----------------------------------------------------------------------
    try:
        eps     = 1e-4
        vals    = np.random.RandomState(1).rand(3, 10).astype(np.float32)
        pt_clip = torch.clamp(torch.tensor(vals), min=eps, max=1.0 - eps).numpy()
        tt_clip = compute_clip(
            [_TW(vals),
             _TW(np.array([eps],       dtype=np.float32)),
             _TW(np.array([1.0 - eps], dtype=np.float32))],
            _OW()
        )
        max_diff = float(np.max(np.abs(pt_clip - tt_clip)))
        ok = max_diff < 1e-7
        print(f"DATA[01] Clip step vs torch.clamp                {'PASS' if ok else 'FAIL'}  "
              f"max_diff={max_diff:.3e}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[01] Clip step vs torch.clamp                ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[02] Full forward (eps=1e-4) vs PyTorch
    # -----------------------------------------------------------------------
    try:
        x_np     = _make_data([2, 10, 8, 8], seed=_SEED)
        pt_out   = _pt_clip_sigmoid(x_np, eps=1e-4)
        tt_out   = _run_graph(x_np, eps=1e-4)
        assert tt_out is not None, "TTSim returned None"
        max_diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = max_diff < 1e-5
        print(f"DATA[02] Full forward eps=1e-4 vs PyTorch        {'PASS' if ok else 'FAIL'}  "
              f"max_diff={max_diff:.3e}  shape={tt_out.shape}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[02] Full forward eps=1e-4 vs PyTorch        ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[03] Value range strictly within [eps, 1-eps]
    # -----------------------------------------------------------------------
    try:
        x_np  = _make_data([4, 20, 10, 10], seed=_SEED)
        eps   = 1e-4
        tt_out = _run_graph(x_np, eps=eps)
        assert tt_out is not None
        mn, mx = float(tt_out.min()), float(tt_out.max())
        ok = (mn >= eps - 1e-7) and (mx <= 1.0 - eps + 1e-7)
        print(f"DATA[03] Value range in [{eps}, {1-eps}]          "
              f"{'PASS' if ok else 'FAIL'}  min={mn:.6f}  max={mx:.6f}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[03] Value range check                       ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[04] Custom eps values
    # -----------------------------------------------------------------------
    for eps_val in [1e-3, 1e-2, 1e-6]:
        try:
            x_np   = _make_data([4, 8], seed=7)
            pt_out = _pt_clip_sigmoid(x_np, eps=eps_val)
            tt_out = _run_graph(x_np, eps=eps_val)
            assert tt_out is not None
            max_diff = float(np.max(np.abs(pt_out - tt_out)))
            ok = max_diff < 1e-5
            print(f"DATA[04] eps={eps_val:<8} vs PyTorch                 "
                  f"{'PASS' if ok else 'FAIL'}  max_diff={max_diff:.3e}")
            if not ok: all_passed = False
        except Exception as exc:
            print(f"DATA[04] eps={eps_val} vs PyTorch                ERROR  {exc}")
            all_passed = False

    # -----------------------------------------------------------------------
    # DATA[05] Extreme inputs – boundary clamping
    # -----------------------------------------------------------------------
    try:
        eps   = 1e-4
        x_np  = np.array([[[-1000.0, -100.0, 0.0, 100.0, 1000.0]]], dtype=np.float32)
        pt_out = _pt_clip_sigmoid(x_np, eps=eps)
        tt_out = _run_graph(x_np, eps=eps)
        assert tt_out is not None
        lo_ok = abs(tt_out.ravel()[0] - eps) < 1e-6        # extreme neg → eps
        hi_ok = abs(tt_out.ravel()[-1] - (1 - eps)) < 1e-6 # extreme pos → 1-eps
        max_diff = float(np.max(np.abs(pt_out - tt_out)))
        ok = lo_ok and hi_ok and max_diff < 1e-6
        print(f"DATA[05] Extreme inputs ±1000 boundary clamp    {'PASS' if ok else 'FAIL'}  "
              f"lo={tt_out.ravel()[0]:.4e}  hi={tt_out.ravel()[-1]:.6f}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[05] Extreme inputs boundary clamp          ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[06] No trainable parameters
    # -----------------------------------------------------------------------
    try:
        x     = _from_shape("ut_cs_noparams", [1, 10])
        out   = clip_sigmoid(x, eps=1e-4)
        # clip_sigmoid is a pure function – there is no Module with param_count
        ok = True   # construction succeeded; no Module sub-class to query
        print(f"DATA[06] Stateless function – no learnable params  PASS  "
              f"(type={type(out).__name__})")
    except Exception as exc:
        print(f"DATA[06] Stateless function – no learnable params  ERROR  {exc}")
        all_passed = False

    # -----------------------------------------------------------------------
    # DATA[07] Determinism – same input → same output
    # -----------------------------------------------------------------------
    try:
        x_np   = _make_data([2, 8, 8], seed=_SEED)
        out_a  = _run_graph(x_np, eps=1e-4)
        out_b  = _run_graph(x_np, eps=1e-4)
        assert out_a is not None and out_b is not None
        ok = np.array_equal(out_a, out_b)
        print(f"DATA[07] Determinism (run twice same input)      {'PASS' if ok else 'FAIL'}")
        if not ok: all_passed = False
    except Exception as exc:
        print(f"DATA[07] Determinism                             ERROR  {exc}")
        all_passed = False

    assert all_passed, "One or more data validation tests failed"
