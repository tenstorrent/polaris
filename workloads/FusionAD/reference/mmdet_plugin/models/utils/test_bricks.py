#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for bricks.py `run_time` decorator.
Tests shape and numerical equivalence between the PyTorch (torch) version
and the TTSIM version.

The `run_time` decorator wraps arbitrary callables with timing instrumentation.
Both versions must:
  - Return identical outputs (shape + values) for the same inputs
  - Correctly accumulate timing and count statistics
"""

import logging
import time
from collections import defaultdict

import numpy as np
import torch


# ===========================================================================
# PyTorch version (original bricks.py — uses torch.cuda.synchronize)
# ===========================================================================
class BricksTorch:
    """Namespace that holds the PyTorch version of the run_time decorator."""

    time_maps = defaultdict(lambda: 0.)
    count_maps = defaultdict(lambda: 0.)

    @staticmethod
    def run_time(name):
        def middle(fn):
            def wrapper(*args, **kwargs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.time()
                res = fn(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                BricksTorch.time_maps['%s : %s' % (name, fn.__name__)] += time.time() - start
                BricksTorch.count_maps['%s : %s' % (name, fn.__name__)] += 1
                logging.info(
                    "%s : %s takes up %f " % (
                        name,
                        fn.__name__,
                        BricksTorch.time_maps['%s : %s' % (name, fn.__name__)]
                        / BricksTorch.count_maps['%s : %s' % (name, fn.__name__)]
                    )
                )
                return res
            return wrapper
        return middle


# ===========================================================================
# TTSIM version (bricks_ttsim.py — no torch dependency, synchronous)
# ===========================================================================
class BricksTTSIM:
    """Namespace that holds the TTSIM version of the run_time decorator."""

    time_maps = defaultdict(lambda: 0.)
    count_maps = defaultdict(lambda: 0.)

    @staticmethod
    def run_time(name):
        def middle(fn):
            def wrapper(*args, **kwargs):
                # ttsim execution is synchronous — no cuda.synchronize() needed
                start = time.time()
                res = fn(*args, **kwargs)
                # ttsim execution is synchronous — no cuda.synchronize() needed
                BricksTTSIM.time_maps['%s : %s' % (name, fn.__name__)] += time.time() - start
                BricksTTSIM.count_maps['%s : %s' % (name, fn.__name__)] += 1
                logging.info(
                    "%s : %s takes up %f " % (
                        name,
                        fn.__name__,
                        BricksTTSIM.time_maps['%s : %s' % (name, fn.__name__)]
                        / BricksTTSIM.count_maps['%s : %s' % (name, fn.__name__)]
                    )
                )
                return res
            return wrapper
        return middle


# Convenience aliases matching the old import names
bricks_torch = BricksTorch
bricks_ttsim = BricksTTSIM


# ---------------------------------------------------------------------------
# Helper: a simple function to be decorated (returns known outputs)
# ---------------------------------------------------------------------------
def _make_sample_fn():
    """Return a plain function that performs element-wise squaring on numpy
    arrays.  This lets us compare outputs numerically."""
    def sample_fn(x):
        return x ** 2
    return sample_fn


def _make_tensor_fn():
    """Return a function that takes a torch Tensor / numpy array and returns
    its square.  Used for shape + numerical validation with tensor-like data."""
    def tensor_fn(x):
        return x ** 2
    return tensor_fn


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("bricks.py `run_time` Decorator Validation: PyTorch vs TTSIM")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)
    all_pass = True

    # ------------------------------------------------------------------
    # Test 1: Scalar input/output — basic decorator behaviour
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 1: Scalar input / output")
    print("-" * 70)

    # Reset global counters for both versions
    bricks_torch.time_maps.clear()
    bricks_torch.count_maps.clear()
    bricks_ttsim.time_maps.clear()
    bricks_ttsim.count_maps.clear()

    fn_pt = bricks_torch.run_time("test")(_make_sample_fn())
    fn_tt = bricks_ttsim.run_time("test")(_make_sample_fn())

    scalar_in = 7.0
    out_pt = fn_pt(scalar_in)
    out_tt = fn_tt(scalar_in)

    print(f"  Input:       {scalar_in}")
    print(f"  PyTorch out: {out_pt}")
    print(f"  TTSIM  out:  {out_tt}")

    scalar_match = (out_pt == out_tt)
    print(f"  Value match: {'[PASS]' if scalar_match else '[FAIL]'}")
    if not scalar_match:
        all_pass = False

    # Verify counters were incremented
    pt_count = list(bricks_torch.count_maps.values())
    tt_count = list(bricks_ttsim.count_maps.values())
    count_match = pt_count == tt_count
    print(f"  Count maps match: {'[PASS]' if count_match else '[FAIL]'}  PT={pt_count}  TTSIM={tt_count}")
    if not count_match:
        all_pass = False

    # ------------------------------------------------------------------
    # Test 2: 1-D numpy array — shape + numerical validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 2: 1-D numpy array (shape + numerical)")
    print("-" * 70)

    arr_in = np.random.randn(128).astype(np.float32)

    fn_pt2 = bricks_torch.run_time("arr_test")(_make_sample_fn())
    fn_tt2 = bricks_ttsim.run_time("arr_test")(_make_sample_fn())

    out_pt2 = fn_pt2(arr_in.copy())
    out_tt2 = fn_tt2(arr_in.copy())

    shape_match = (out_pt2.shape == out_tt2.shape)
    print(f"  Input shape:  {arr_in.shape}")
    print(f"  PT  output shape: {out_pt2.shape}")
    print(f"  TT  output shape: {out_tt2.shape}")
    print(f"  Shape match: {'[PASS]' if shape_match else '[FAIL]'}")
    if not shape_match:
        all_pass = False

    atol, rtol = 1e-6, 1e-6
    diff = np.abs(out_pt2 - out_tt2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Tolerance: atol={atol}, rtol={rtol}")
    print(f"  Max  absolute difference: {max_diff:.10f}")
    print(f"  Mean absolute difference: {mean_diff:.10f}")

    num_match = np.allclose(out_pt2, out_tt2, atol=atol, rtol=rtol)
    print(f"  Numerical match: {'[PASS]' if num_match else '[FAIL]'}")
    if not num_match:
        all_pass = False

    # ------------------------------------------------------------------
    # Test 3: Multi-dimensional numpy array — shape + numerical validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 3: Multi-dimensional numpy array (shape + numerical)")
    print("-" * 70)

    tensor_in = np.random.randn(2, 3, 64, 64).astype(np.float32)

    fn_pt3 = bricks_torch.run_time("tensor_test")(_make_tensor_fn())
    fn_tt3 = bricks_ttsim.run_time("tensor_test")(_make_tensor_fn())

    out_pt3 = fn_pt3(tensor_in.copy())
    out_tt3 = fn_tt3(tensor_in.copy())

    shape_match3 = (out_pt3.shape == out_tt3.shape)
    print(f"  Input shape:  {tensor_in.shape}")
    print(f"  PT  output shape: {out_pt3.shape}")
    print(f"  TT  output shape: {out_tt3.shape}")
    print(f"  Shape match: {'[PASS]' if shape_match3 else '[FAIL]'}")
    if not shape_match3:
        all_pass = False

    diff3 = np.abs(out_pt3 - out_tt3)
    max_diff3 = np.max(diff3)
    mean_diff3 = np.mean(diff3)
    print(f"  Tolerance: atol={atol}, rtol={rtol}")
    print(f"  Max  absolute difference: {max_diff3:.10f}")
    print(f"  Mean absolute difference: {mean_diff3:.10f}")
    print(f"  Output stats PT:   min={out_pt3.min():.6f}, max={out_pt3.max():.6f}, mean={out_pt3.mean():.6f}")
    print(f"  Output stats TT:   min={out_tt3.min():.6f}, max={out_tt3.max():.6f}, mean={out_tt3.mean():.6f}")

    num_match3 = np.allclose(out_pt3, out_tt3, atol=atol, rtol=rtol)
    print(f"  Numerical match: {'[PASS]' if num_match3 else '[FAIL]'}")
    if not num_match3:
        all_pass = False

    # ------------------------------------------------------------------
    # Test 4: Torch tensor input — shape + numerical validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 4: Torch tensor input (shape + numerical)")
    print("-" * 70)

    t_in = torch.randn(4, 16, dtype=torch.float32)

    def square_torch(x):
        return x ** 2

    fn_pt4 = bricks_torch.run_time("torch_tensor")(square_torch)
    fn_tt4 = bricks_ttsim.run_time("torch_tensor")(square_torch)

    out_pt4 = fn_pt4(t_in.clone())
    out_tt4 = fn_tt4(t_in.clone())

    out_pt4_np = out_pt4.numpy()
    out_tt4_np = out_tt4.numpy()

    shape_match4 = (out_pt4_np.shape == out_tt4_np.shape)
    print(f"  Input shape:  {t_in.shape}")
    print(f"  PT  output shape: {out_pt4_np.shape}")
    print(f"  TT  output shape: {out_tt4_np.shape}")
    print(f"  Shape match: {'[PASS]' if shape_match4 else '[FAIL]'}")
    if not shape_match4:
        all_pass = False

    diff4 = np.abs(out_pt4_np - out_tt4_np)
    max_diff4 = np.max(diff4)
    mean_diff4 = np.mean(diff4)
    print(f"  Tolerance: atol={atol}, rtol={rtol}")
    print(f"  Max  absolute difference: {max_diff4:.10f}")
    print(f"  Mean absolute difference: {mean_diff4:.10f}")

    num_match4 = np.allclose(out_pt4_np, out_tt4_np, atol=atol, rtol=rtol)
    print(f"  Numerical match: {'[PASS]' if num_match4 else '[FAIL]'}")
    if not num_match4:
        all_pass = False

    # ------------------------------------------------------------------
    # Test 5: Multiple calls — cumulative counter validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 5: Multiple invocations — counter accumulation")
    print("-" * 70)

    bricks_torch.time_maps.clear()
    bricks_torch.count_maps.clear()
    bricks_ttsim.time_maps.clear()
    bricks_ttsim.count_maps.clear()

    fn_pt5 = bricks_torch.run_time("multi")(_make_sample_fn())
    fn_tt5 = bricks_ttsim.run_time("multi")(_make_sample_fn())

    n_calls = 10
    for i in range(n_calls):
        _ = fn_pt5(float(i))
        _ = fn_tt5(float(i))

    pt_counts = dict(bricks_torch.count_maps)
    tt_counts = dict(bricks_ttsim.count_maps)
    print(f"  Number of calls: {n_calls}")
    print(f"  PT count_maps:  {pt_counts}")
    print(f"  TT count_maps:  {tt_counts}")

    counts_match = all(
        pt_counts[k] == tt_counts[k] for k in pt_counts if k in tt_counts
    )
    print(f"  Count values match: {'[PASS]' if counts_match else '[FAIL]'}")
    if not counts_match:
        all_pass = False

    # Verify time_maps keys match
    keys_match = set(bricks_torch.time_maps.keys()) == set(bricks_ttsim.time_maps.keys())
    print(f"  Timer keys match:  {'[PASS]' if keys_match else '[FAIL]'}")
    if not keys_match:
        all_pass = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: [PASS] ALL TESTS PASSED — TTSIM matches PyTorch")
    else:
        print("OVERALL: [FAIL] SOME TESTS FAILED — see details above")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
