#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for clip_sigmoid TTSim module.

Validates the TTSim conversion of mmdet3d/models/utils/clip_sigmoid.py.
Each test compares the TTSim graph output (.data) against the equivalent
PyTorch operation at every step of the forward pass.

Test Coverage:
  1.  Module Construction         – TTSim graph built without error; shape correct
  2.  Output Shape Validation     – shape preserved for 1-D through 4-D inputs
  3.  Sigmoid Step                – TTSim Sigmoid op vs torch.sigmoid
  4.  Clip Step                   – TTSim Clip op  vs torch.clamp
  5.  Full Forward (eps=1e-4)     – TTSim clip_sigmoid vs torch.clamp(torch.sigmoid())
  6.  Custom eps values           – eps=1e-3, 1e-2, 1e-6
  7.  Value Range Check           – all outputs strictly within [eps, 1-eps]
  8.  Extreme Inputs              – numerical stability at ±100
  9.  No Trainable Parameters     – clip_sigmoid is a stateless function
"""

import os
import sys

# Polaris root (contains ttsim/)
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _polaris_root not in sys.path:
    sys.path.insert(0, _polaris_root)

# EA-LSS root (contains ttsim_modules/ and Reference/)
_ealss_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)

import numpy as np
import torch

# TTSim imports
import ttsim.front.functional.op as F
from ttsim.front.functional.op import _from_data, _from_shape, Sigmoid
from ttsim.ops.desc.data_compute import compute_sigmoid, compute_clip

# EA-LSS TTSim module under test
from ttsim_modules.clip_sigmoid import clip_sigmoid

# Validation utilities (same dir as this script)
_val_dir = os.path.dirname(__file__)
if _val_dir not in sys.path:
    sys.path.insert(0, _val_dir)
from ttsim_utils import compare_arrays, print_header, print_test

# ============================================================================
# PyTorch reference helpers
# ============================================================================

def pt_sigmoid(x_np: np.ndarray) -> np.ndarray:
    """PyTorch sigmoid reference."""
    return torch.sigmoid(torch.tensor(x_np)).numpy()


def pt_clip(x_np: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """PyTorch clamp reference."""
    return torch.clamp(torch.tensor(x_np), min=lo, max=hi).numpy()


def pt_clip_sigmoid(x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """PyTorch clip_sigmoid reference: clamp(sigmoid(x), eps, 1-eps)."""
    return torch.clamp(torch.sigmoid(torch.tensor(x_np)),
                       min=eps, max=1.0 - eps).numpy()


# ============================================================================
# TTSim graph runners
# ============================================================================

class _TW:
    """Thin wrapper so data_compute functions see .data attribute."""
    def __init__(self, d): self.data = d


class _OW:
    """Thin wrapper for op attributes."""
    def __init__(self, **kw): self.attrs = kw


def _ttsim_sigmoid_step(x_np: np.ndarray) -> np.ndarray:
    """Run just the Sigmoid op via data_compute."""
    return compute_sigmoid([_TW(x_np)], _OW())


def _ttsim_clip_step(x_np: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Run just the Clip op via data_compute."""
    from ttsim.ops.desc.data_compute import compute_clip
    lo_t = _TW(np.array([lo], dtype=np.float32))
    hi_t = _TW(np.array([hi], dtype=np.float32))
    return compute_clip([_TW(x_np), lo_t, hi_t], _OW())


def _ttsim_graph(x_np: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Run the full TTSim clip_sigmoid graph and return output .data."""
    x_tensor = _from_data("cs_input", x_np)
    out = clip_sigmoid(x_tensor, eps=eps)
    return out.data


# ============================================================================
# Helpers
# ============================================================================

def _make_input(shape, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


# ============================================================================
# Tests
# ============================================================================

def test_construction():
    print_header("TEST 1: Module Construction")
    print_test("Building TTSim clip_sigmoid graph (shape-only, no data)")

    x = _from_shape("input_shape_only", [2, 10, 8, 8])
    out = clip_sigmoid(x, eps=1e-4)

    assert out.shape == [2, 10, 8, 8], f"Expected [2, 10, 8, 8], got {out.shape}"
    print(f"  ✓ Graph built successfully")
    print(f"  ✓ Input  shape : {x.shape}")
    print(f"  ✓ Output shape : {out.shape}")
    print(f"  ✓ Output tensor name : {out.name}")
    return True


def test_output_shape():
    print_header("TEST 2: Output Shape Validation (various shapes)")
    print_test("Shape preserved for each input shape; PyTorch shape used as ground truth")

    test_shapes = [
        [4],
        [3, 10],
        [2, 10, 8],
        [2, 10, 8, 8],
        [1, 80, 128, 128],
    ]

    passed = True
    for shape in test_shapes:
        x_np  = _make_input(shape)
        # PyTorch reference shape
        pt_out = pt_clip_sigmoid(x_np)
        # TTSim graph shape (data path)
        x_sim = _from_data("cs_shape_test", x_np)
        out   = clip_sigmoid(x_sim, eps=1e-4)
        ttsim_shape = list(out.shape)
        ok = (ttsim_shape == shape) and (list(pt_out.shape) == shape)
        status = "✓" if ok else "✗"
        print(f"  {status} Input {shape} → TTSim {ttsim_shape}  PyTorch {list(pt_out.shape)}")
        if not ok:
            passed = False
    return passed


def test_step_sigmoid():
    print_header("TEST 3: Step-by-step — Sigmoid")
    print_test("TTSim Sigmoid op (data_compute.compute_sigmoid) vs torch.sigmoid")

    x_np = _make_input([2, 4, 4], seed=10)

    pt_out    = pt_sigmoid(x_np)
    ttsim_out = _ttsim_sigmoid_step(x_np)

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  PyTorch sample: {pt_out.ravel()[:4]}")
    print(f"  TTSim   sample: {ttsim_out.ravel()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "sigmoid step", rtol=1e-5, atol=1e-5)


def test_step_clip():
    print_header("TEST 4: Step-by-step — Clip")
    print_test("TTSim Clip op (data_compute.compute_clip) vs torch.clamp")

    eps = 1e-4
    rng = np.random.RandomState(42)
    # Use post-sigmoid values ∈ [0,1]
    sig_values = rng.rand(3, 10).astype(np.float32)

    pt_out    = pt_clip(sig_values, eps, 1.0 - eps)
    ttsim_out = _ttsim_clip_step(sig_values, eps, 1.0 - eps)

    print(f"  PyTorch shape : {pt_out.shape}")
    print(f"  TTSim   shape : {ttsim_out.shape}")
    print(f"  PyTorch sample: {pt_out.ravel()[:4]}")
    print(f"  TTSim   sample: {ttsim_out.ravel()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "clip step", atol=1e-7)


def test_full_forward_default_eps():
    print_header("TEST 5: Full Forward Pass (eps=1e-4)")
    print_test("TTSim clip_sigmoid graph (.data) vs torch.clamp(torch.sigmoid(x), eps, 1-eps)")

    x_np = _make_input([2, 10, 8, 8], seed=123)
    eps  = 1e-4

    pt_out    = pt_clip_sigmoid(x_np, eps=eps)
    ttsim_out = _ttsim_graph(x_np, eps=eps)

    print(f"  PyTorch shape  : {pt_out.shape}")
    print(f"  TTSim   shape  : {ttsim_out.shape if ttsim_out is not None else 'None'}")

    assert ttsim_out is not None, "TTSim data compute returned None — data_compute fix required!"
    print(f"  PyTorch sample : {pt_out.ravel()[:4]}")
    print(f"  TTSim   sample : {ttsim_out.ravel()[:4]}")
    return compare_arrays(pt_out, ttsim_out, "full forward (eps=1e-4)", rtol=1e-5, atol=1e-5)


def test_custom_eps():
    print_header("TEST 6: Custom eps values")
    passed = True

    for eps in [1e-3, 1e-2, 1e-6]:
        print_test(f"eps = {eps}")
        x_np      = _make_input([4, 8], seed=7)
        pt_out    = pt_clip_sigmoid(x_np, eps=eps)
        ttsim_out = _ttsim_graph(x_np, eps=eps)
        assert ttsim_out is not None, f"TTSim returned None for eps={eps}"
        ok = compare_arrays(pt_out, ttsim_out, f"eps={eps}", rtol=1e-5, atol=1e-5)
        passed = passed and ok

    return passed


def test_value_range():
    print_header("TEST 7: Value Range Check")
    print_test("All output values in [eps, 1-eps] — verified by both PyTorch and TTSim")

    eps  = 1e-4
    x_np = _make_input([4, 20, 10, 10])

    pt_out    = pt_clip_sigmoid(x_np, eps=eps)
    ttsim_out = _ttsim_graph(x_np, eps=eps)
    assert ttsim_out is not None, "TTSim data compute returned None"

    for label, arr in [("PyTorch", pt_out), ("TTSim  ", ttsim_out)]:
        mn, mx = float(arr.min()), float(arr.max())
        ok_range = (mn >= eps - 1e-7) and (mx <= 1.0 - eps + 1e-7)
        status = "✓" if ok_range else "✗"
        print(f"  {status} {label}  min={mn:.6f}  max={mx:.6f}  range=[{eps}, {1-eps}]")
        if not ok_range:
            return False

    return compare_arrays(pt_out, ttsim_out, "value range", rtol=1e-5, atol=1e-5)


def test_extreme_inputs():
    print_header("TEST 8: Extreme Input Values (±100)")
    print_test("TTSim Clip correctly clamps sigmoid(±100) to eps / 1-eps")

    eps  = 1e-4
    x_np = np.array([[[-100.0, -50.0, 0.0, 50.0, 100.0]]], dtype=np.float32)

    pt_out    = pt_clip_sigmoid(x_np, eps=eps)
    ttsim_out = _ttsim_graph(x_np, eps=eps)
    assert ttsim_out is not None, "TTSim data compute returned None"

    print(f"  Input   : {x_np.ravel()}")
    print(f"  PyTorch : {pt_out.ravel()}")
    print(f"  TTSim   : {ttsim_out.ravel()}")

    # Clamp assertions
    assert abs(ttsim_out.ravel()[0] - eps) < 1e-6, "large negative not clamped to eps"
    assert abs(ttsim_out.ravel()[-1] - (1 - eps)) < 1e-6, "large positive not clamped to 1-eps"
    print("  ✓ Boundary clamping verified")

    return compare_arrays(pt_out, ttsim_out, "extreme inputs", atol=1e-6)


def test_no_trainable_params():
    print_header("TEST 9: No Trainable Parameters")
    print_test("clip_sigmoid is a stateless functional op — no learnable weights")

    x = _from_shape("x_no_params", [1, 10])
    out = clip_sigmoid(x, eps=1e-4)

    print(f"  ✓ clip_sigmoid is a stateless function (no Module wrapping)")
    print(f"  ✓ Output tensor type : {type(out).__name__}")
    return True


# ============================================================================
# Main runner
# ============================================================================

if __name__ == "__main__":
    results = {}

    tests = [
        ("Construction",        test_construction),
        ("Output Shape",        test_output_shape),
        ("Sigmoid Step",        test_step_sigmoid),
        ("Clip Step",           test_step_clip),
        ("Full Forward",        test_full_forward_default_eps),
        ("Custom eps",          test_custom_eps),
        ("Value Range",         test_value_range),
        ("Extreme Inputs",      test_extreme_inputs),
        ("No Trainable Params", test_no_trainable_params),
    ]

    passed_all = True
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:
            print(f"\n  ✗ Test raised exception: {e}")
            import traceback
            traceback.print_exc()
            ok = False
        results[name] = ok
        passed_all = passed_all and ok

    print_header("SUMMARY")
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    total  = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    sys.exit(0 if passed_all else 1)
