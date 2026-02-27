#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Position Encoding — Numerical Compute Validation: PyTorch vs TTSim.

Exhaustive element-wise comparison across multiple configurations,
spatial sizes, mask patterns, and parameter settings.

Tests:
   1. PositionEmbeddingSine — standard normalised config
   2. PositionEmbeddingSine — unnormalised (scale = default)
   3. PositionEmbeddingSine — non-square spatial sizes
   4. PositionEmbeddingSine — varying mask patterns
   5. PositionEmbeddingSine — y / x component separation
   6. PositionEmbeddingSine — parameter sweeps (num_pos_feats, temperature)
   7. PositionEmbeddingSine — edge cases (1×1, single batch, large)
   8. PositionEmbeddingLearned — shape validation (numerics differ by design)
   9. build_position_encoding — factory function verification

Usage:
  python test_position_encoding_numerical.py
"""

import os, sys, math
import torch
import numpy as np

# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── PyTorch imports ──────────────────────────────────────────────────────────
from workloads.Deformable_DETR.reference.position_encoding import (
    PositionEmbeddingSine as SinePT,
    PositionEmbeddingLearned as LearnedPT,
    build_position_encoding as build_pt,
)
from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPT

# ── TTSim imports ────────────────────────────────────────────────────────────
from workloads.Deformable_DETR.models.position_encoding_ttsim import (
    PositionEmbeddingSine as SineTT,
    PositionEmbeddingLearned as LearnedTT,
    build_position_encoding as build_tt,
)
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTT
from ttsim.ops.tensor import SimTensor

# ═══════════════════════════════════════════════════════════════════════════════
# Globals
# ═══════════════════════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _sim(t, name="t"):
    """Torch tensor → SimTensor."""
    d = t.detach().cpu().numpy().copy()
    return SimTensor(
        {"name": name, "shape": list(d.shape), "data": d, "dtype": d.dtype}
    )


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def subsection(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


def compare_numerical(label, pt_data, tt_data, atol=1e-5, rtol=1e-4):
    """Element-wise numerical comparison; update global counters."""
    global PASS_COUNT, FAIL_COUNT

    if isinstance(pt_data, torch.Tensor):
        pt_np = pt_data.detach().cpu().numpy()
    else:
        pt_np = np.asarray(pt_data)

    if isinstance(tt_data, SimTensor):
        tt_np = tt_data.data
    elif isinstance(tt_data, np.ndarray):
        tt_np = tt_data
    else:
        tt_np = np.asarray(tt_data)

    if tt_np is None:
        print(f"  ⚠ SKIP {label}: TTSim data is None (shape-inference only)")
        return False

    abs_diff = np.abs(pt_np.astype(np.float64) - tt_np.astype(np.float64))
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    ok = np.allclose(pt_np, tt_np, atol=atol, rtol=rtol)

    tag = "PASS" if ok else "FAIL"
    sym = "✓" if ok else "✗"
    print(f"  {sym} {tag} {label}:  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    if not ok:
        idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"         worst @ {idx}: PT={pt_np[idx]:.8f}  TT={tt_np[idx]:.8f}")

    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def check_shape(label, expected, actual):
    """Shape comparison; update global counters."""
    global PASS_COUNT, FAIL_COUNT
    ok = list(expected) == list(actual)
    sym = "✓" if ok else "✗"
    tag = "PASS" if ok else "FAIL"
    print(f"  {sym} {tag} {label}:  expected {list(expected)}  got {list(actual)}")
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    return ok


def make_inputs(B, C, H, W, mask_fn=None, seed=42):
    """Create matching PyTorch & TTSim NestedTensor inputs.

    mask_fn(B, H, W) → bool mask array.  None ⇒ all-False (no masking).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_pt = torch.randn(B, C, H, W)
    if mask_fn is not None:
        m_pt = mask_fn(B, H, W)
        if isinstance(m_pt, np.ndarray):
            m_pt = torch.from_numpy(m_pt)
    else:
        m_pt = torch.zeros(B, H, W, dtype=torch.bool)

    nt_pt = NestedTensorPT(x_pt, m_pt)

    x_tt = _sim(x_pt, "input")
    m_tt = m_pt.numpy() if isinstance(m_pt, torch.Tensor) else m_pt
    nt_tt = NestedTensorTT(x_tt, m_tt)

    return nt_pt, nt_tt


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — PositionEmbeddingSine: standard normalised config
# ═══════════════════════════════════════════════════════════════════════════════


def test_01_sine_normalised():
    section("Test 1: PositionEmbeddingSine — normalised (standard)")

    B, C, H, W = 2, 256, 28, 28
    D = 128  # num_pos_feats

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=True)
    pt.eval()
    tt = SineTT("sine_norm", num_pos_feats=D, temperature=10000, normalize=True)

    nt_pt, nt_tt = make_inputs(B, C, H, W)

    with torch.no_grad():
        out_pt = pt(nt_pt)
    out_tt = tt(nt_tt)

    all_ok = True
    all_ok &= check_shape("output shape", out_pt.shape, out_tt.shape)
    all_ok &= check_shape("expected [B,2D,H,W]", [B, 2 * D, H, W], out_tt.shape)
    all_ok &= compare_numerical("full output", out_pt, out_tt)

    # value range: sin/cos → [-1, 1]
    global PASS_COUNT, FAIL_COUNT
    if out_tt.data is not None:
        vmin, vmax = out_tt.data.min(), out_tt.data.max()
        ok = (-1.0 - 1e-6 <= vmin) and (vmax <= 1.0 + 1e-6)
        sym = "✓" if ok else "✗"
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {sym} {tag} value range: [{vmin:.6f}, {vmax:.6f}]  (expected ⊆ [-1,1])"
        )
        if ok:
            PASS_COUNT += 1
        else:
            FAIL_COUNT += 1
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — PositionEmbeddingSine: unnormalised
# ═══════════════════════════════════════════════════════════════════════════════


def test_02_sine_unnormalised():
    section("Test 2: PositionEmbeddingSine — unnormalised")

    B, C, H, W = 1, 64, 16, 16
    D = 32

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=False)
    pt.eval()
    tt = SineTT("sine_unnorm", num_pos_feats=D, temperature=10000, normalize=False)

    nt_pt, nt_tt = make_inputs(B, C, H, W, seed=99)

    with torch.no_grad():
        out_pt = pt(nt_pt)
    out_tt = tt(nt_tt)

    all_ok = True
    all_ok &= check_shape("output shape", out_pt.shape, out_tt.shape)
    all_ok &= compare_numerical("full output", out_pt, out_tt)

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — PositionEmbeddingSine: non-square spatial sizes
# ═══════════════════════════════════════════════════════════════════════════════


def test_03_sine_nonsquare():
    section("Test 3: PositionEmbeddingSine — non-square spatial dimensions")

    D = 64
    configs = [
        ("wide", 1, 128, 14, 28),
        ("tall", 2, 128, 32, 8),
        ("skinny", 1, 64, 50, 3),
        ("small_rect", 2, 256, 7, 11),
    ]

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=True)
    pt.eval()
    tt = SineTT("sine_nonsq", num_pos_feats=D, temperature=10000, normalize=True)

    all_ok = True
    for name, B, C, H, W in configs:
        subsection(f"Config: {name}  [{B},{C},{H},{W}]")
        nt_pt, nt_tt = make_inputs(B, C, H, W, seed=hash(name) % 2**31)

        with torch.no_grad():
            out_pt = pt(nt_pt)
        out_tt = tt(nt_tt)

        ok = check_shape(f"{name} shape", out_pt.shape, out_tt.shape)
        ok &= compare_numerical(f"{name} values", out_pt, out_tt)
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — PositionEmbeddingSine: varying mask patterns
# ═══════════════════════════════════════════════════════════════════════════════


def test_04_sine_masks():
    section("Test 4: PositionEmbeddingSine — varying mask patterns")

    B, C, H, W = 2, 128, 20, 20
    D = 64

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=True)
    pt.eval()
    tt = SineTT("sine_masks", num_pos_feats=D, temperature=10000, normalize=True)

    mask_fns = {
        "no_mask": lambda b, h, w: np.zeros((b, h, w), dtype=bool),
        "top_rows": lambda b, h, w: np.concatenate(
            [
                np.ones((b, h // 4, w), dtype=bool),
                np.zeros((b, h - h // 4, w), dtype=bool),
            ],
            axis=1,
        ),
        "left_cols": lambda b, h, w: np.concatenate(
            [
                np.ones((b, h, w // 3), dtype=bool),
                np.zeros((b, h, w - w // 3), dtype=bool),
            ],
            axis=2,
        ),
        "checkerboard": lambda b, h, w: np.array(
            [
                [[(r + c) % 2 == 0 for c in range(w)] for r in range(h)]
                for _ in range(b)
            ],
            dtype=bool,
        ),
        "bottom_right": lambda b, h, w: _corner_mask(b, h, w),
    }

    all_ok = True
    for mname, mfn in mask_fns.items():
        subsection(f"Mask: {mname}")
        nt_pt, nt_tt = make_inputs(B, C, H, W, mask_fn=mfn, seed=12345)

        with torch.no_grad():
            out_pt = pt(nt_pt)
        out_tt = tt(nt_tt)

        ok = check_shape(f"{mname} shape", out_pt.shape, out_tt.shape)
        ok &= compare_numerical(f"{mname} values", out_pt, out_tt)
        all_ok &= ok

    return all_ok


def _corner_mask(b, h, w):
    """Bottom-right quadrant masked."""
    m = np.zeros((b, h, w), dtype=bool)
    m[:, h // 2 :, w // 2 :] = True
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — PositionEmbeddingSine: y / x component separation
# ═══════════════════════════════════════════════════════════════════════════════


def test_05_sine_components():
    section("Test 5: PositionEmbeddingSine — y/x component separation")

    B, C, H, W = 2, 256, 24, 24
    D = 128

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=True)
    pt.eval()
    tt = SineTT("sine_comp", num_pos_feats=D, temperature=10000, normalize=True)

    nt_pt, nt_tt = make_inputs(B, C, H, W)

    with torch.no_grad():
        out_pt = pt(nt_pt)
    out_tt = tt(nt_tt)

    pt_np = out_pt.detach().cpu().numpy()
    tt_np = out_tt.data

    all_ok = True
    # y-component: channels [0, D)
    all_ok &= compare_numerical(
        "y-pos component", pt_np[:, :D, :, :], tt_np[:, :D, :, :]
    )
    # x-component: channels [D, 2D)
    all_ok &= compare_numerical(
        "x-pos component", pt_np[:, D:, :, :], tt_np[:, D:, :, :]
    )

    # Verify y-pos is constant across columns (within same row, unmasked)
    global PASS_COUNT, FAIL_COUNT
    y_col_0 = tt_np[0, :D, :, 0]
    y_col_5 = tt_np[0, :D, :, 5]
    y_match = np.allclose(y_col_0, y_col_5, atol=1e-6)
    sym = "✓" if y_match else "✗"
    tag = "PASS" if y_match else "FAIL"
    print(f"  {sym} {tag} y-pos constant across columns (col 0 vs col 5)")
    if y_match:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= y_match

    # Verify x-pos is constant across rows (within same column, unmasked)
    x_row_0 = tt_np[0, D:, 0, :]
    x_row_5 = tt_np[0, D:, 5, :]
    x_match = np.allclose(x_row_0, x_row_5, atol=1e-6)
    sym = "✓" if x_match else "✗"
    tag = "PASS" if x_match else "FAIL"
    print(f"  {sym} {tag} x-pos constant across rows (row 0 vs row 5)")
    if x_match:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= x_match

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — PositionEmbeddingSine: parameter sweeps
# ═══════════════════════════════════════════════════════════════════════════════


def test_06_sine_param_sweep():
    section("Test 6: PositionEmbeddingSine — parameter sweeps (D, temperature)")

    B, C, H, W = 1, 128, 12, 12

    sweeps = [
        ("D=32  T=10000", 32, 10000),
        ("D=64  T=10000", 64, 10000),
        ("D=128 T=10000", 128, 10000),
        ("D=256 T=10000", 256, 10000),
        ("D=64  T=100", 64, 100),
        ("D=64  T=5000", 64, 5000),
        ("D=64  T=20000", 64, 20000),
    ]

    all_ok = True
    for name, D, T in sweeps:
        subsection(f"Sweep: {name}")
        pt = SinePT(num_pos_feats=D, temperature=T, normalize=True)
        pt.eval()
        tt = SineTT(f"sine_{name}", num_pos_feats=D, temperature=T, normalize=True)

        nt_pt, nt_tt = make_inputs(B, C, H, W, seed=42)

        with torch.no_grad():
            out_pt = pt(nt_pt)
        out_tt = tt(nt_tt)

        ok = check_shape(f"{name} shape", out_pt.shape, out_tt.shape)
        ok &= check_shape(f"{name} expected [B,2D,H,W]", [B, 2 * D, H, W], out_tt.shape)
        ok &= compare_numerical(f"{name} values", out_pt, out_tt)
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — PositionEmbeddingSine: edge cases
# ═══════════════════════════════════════════════════════════════════════════════


def test_07_sine_edge_cases():
    section("Test 7: PositionEmbeddingSine — edge cases")

    D = 64

    cases = [
        ("1×1 spatial", 1, 64, 1, 1),
        ("single batch", 1, 128, 16, 16),
        ("1×W strip", 2, 64, 1, 32),
        ("H×1 strip", 2, 64, 32, 1),
        ("large spatial", 1, 128, 64, 64),
        ("bs=4 standard", 4, 256, 14, 14),
    ]

    pt = SinePT(num_pos_feats=D, temperature=10000, normalize=True)
    pt.eval()
    tt = SineTT("sine_edge", num_pos_feats=D, temperature=10000, normalize=True)

    all_ok = True
    for name, B, C, H, W in cases:
        subsection(f"Edge: {name}  [{B},{C},{H},{W}]")
        nt_pt, nt_tt = make_inputs(B, C, H, W, seed=7)

        with torch.no_grad():
            out_pt = pt(nt_pt)
        out_tt = tt(nt_tt)

        ok = check_shape(f"{name} shape", out_pt.shape, out_tt.shape)
        ok &= compare_numerical(f"{name} values", out_pt, out_tt)
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 8 — PositionEmbeddingLearned: shape validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_08_learned_numerical():
    section("Test 8: PositionEmbeddingLearned — numerical validation (weights synced)")

    configs = [
        ("standard", 2, 256, 28, 28, 128),
        ("small", 1, 64, 10, 10, 32),
        ("non_square", 2, 128, 14, 28, 64),
        ("large_D", 1, 512, 20, 20, 256),
    ]

    all_ok = True
    for name, B, C, H, W, D in configs:
        subsection(f"Config: {name}  [{B},{C},{H},{W}]  D={D}")

        pt = LearnedPT(num_pos_feats=D)
        pt.eval()
        tt = LearnedTT(f"learned_{name}", num_pos_feats=D)

        # Sync weights from PyTorch to TTSim
        tt.row_embed_weight = pt.row_embed.weight.detach().cpu().numpy().copy()
        tt.col_embed_weight = pt.col_embed.weight.detach().cpu().numpy().copy()

        nt_pt, nt_tt = make_inputs(B, C, H, W, seed=42)

        with torch.no_grad():
            out_pt = pt(nt_pt)
        out_tt = tt(nt_tt)

        ok = check_shape(f"{name} shape match", out_pt.shape, out_tt.shape)
        ok &= check_shape(f"{name} expected [B,2D,H,W]", [B, 2 * D, H, W], out_tt.shape)
        ok &= compare_numerical(f"{name} values", out_pt, out_tt)
        all_ok &= ok

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Test 9 — build_position_encoding: factory function
# ═══════════════════════════════════════════════════════════════════════════════


def test_09_build_factory():
    section("Test 9: build_position_encoding — factory function")

    global PASS_COUNT, FAIL_COUNT

    # ── 9a: sine ─────────────────────────────────────────────────────────
    subsection("9a: build (sine)")

    class ArgsSine:
        hidden_dim = 256
        position_embedding = "sine"

    pe_pt = build_pt(ArgsSine())
    pe_tt = build_tt(ArgsSine())

    ok_type = isinstance(pe_pt, SinePT) and isinstance(pe_tt, SineTT)
    sym = "✓" if ok_type else "✗"
    tag = "PASS" if ok_type else "FAIL"
    print(
        f"  {sym} {tag} type check: PT={type(pe_pt).__name__}  TT={type(pe_tt).__name__}"
    )
    if ok_type:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

    # forward pass + numerical check
    B, H, W = 2, 14, 14
    nt_pt, nt_tt = make_inputs(B, ArgsSine.hidden_dim, H, W, seed=42)
    pe_pt.eval()
    with torch.no_grad():
        out_pt = pe_pt(nt_pt)
    out_tt = pe_tt(nt_tt)

    all_ok = ok_type
    all_ok &= check_shape("sine factory shape", out_pt.shape, out_tt.shape)
    all_ok &= compare_numerical("sine factory values", out_pt, out_tt)

    # ── 9b: learned ──────────────────────────────────────────────────────
    subsection("9b: build (learned)")

    class ArgsLearned:
        hidden_dim = 256
        position_embedding = "learned"

    pe_pt2 = build_pt(ArgsLearned())
    pe_tt2 = build_tt(ArgsLearned())

    ok_type2 = isinstance(pe_pt2, LearnedPT) and isinstance(pe_tt2, LearnedTT)
    sym = "✓" if ok_type2 else "✗"
    tag = "PASS" if ok_type2 else "FAIL"
    print(
        f"  {sym} {tag} type check: PT={type(pe_pt2).__name__}  TT={type(pe_tt2).__name__}"
    )
    if ok_type2:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= ok_type2

    pe_pt2.eval()
    with torch.no_grad():
        out_pt2 = pe_pt2(nt_pt)
    out_tt2 = pe_tt2(nt_tt)
    all_ok &= check_shape("learned factory shape", out_pt2.shape, out_tt2.shape)

    # ── 9c: invalid ──────────────────────────────────────────────────────
    subsection("9c: build (invalid)")

    class ArgsInvalid:
        hidden_dim = 256
        position_embedding = "banana"

    raised = False
    try:
        build_pt(ArgsInvalid())
    except ValueError:
        raised = True
    sym = "✓" if raised else "✗"
    tag = "PASS" if raised else "FAIL"
    print(f"  {sym} {tag} PyTorch raises ValueError for unsupported type")
    if raised:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= raised

    raised2 = False
    try:
        build_tt(ArgsInvalid())
    except ValueError:
        raised2 = True
    sym = "✓" if raised2 else "✗"
    tag = "PASS" if raised2 else "FAIL"
    print(f"  {sym} {tag} TTSim raises ValueError for unsupported type")
    if raised2:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= raised2

    # ── 9d: v2/v3 aliases ────────────────────────────────────────────────
    subsection("9d: v2/v3 aliases")

    class ArgsV2:
        hidden_dim = 128
        position_embedding = "v2"

    class ArgsV3:
        hidden_dim = 128
        position_embedding = "v3"

    ok_v2 = isinstance(build_pt(ArgsV2()), SinePT) and isinstance(
        build_tt(ArgsV2()), SineTT
    )
    sym = "✓" if ok_v2 else "✗"
    print(f"  {sym} {'PASS' if ok_v2 else 'FAIL'} v2 alias → Sine")
    if ok_v2:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= ok_v2

    ok_v3 = isinstance(build_pt(ArgsV3()), LearnedPT) and isinstance(
        build_tt(ArgsV3()), LearnedTT
    )
    sym = "✓" if ok_v3 else "✗"
    print(f"  {sym} {'PASS' if ok_v3 else 'FAIL'} v3 alias → Learned")
    if ok_v3:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    all_ok &= ok_v3

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 80)
    print("  Position Encoding — Numerical Compute Validation")
    print("  PyTorch vs TTSim")
    print("=" * 80)

    results = {}

    results["01_Sine_normalised"] = test_01_sine_normalised()
    results["02_Sine_unnormalised"] = test_02_sine_unnormalised()
    results["03_Sine_nonsquare"] = test_03_sine_nonsquare()
    results["04_Sine_masks"] = test_04_sine_masks()
    results["05_Sine_components"] = test_05_sine_components()
    results["06_Sine_param_sweep"] = test_06_sine_param_sweep()
    results["07_Sine_edge_cases"] = test_07_sine_edge_cases()
    results["08_Learned_numerical"] = test_08_learned_numerical()
    results["09_build_factory"] = test_09_build_factory()

    # ── Summary ──────────────────────────────────────────────────────────
    section("SUMMARY")

    for name, ok in results.items():
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {name}")

    all_ok = all(results.values())
    print(
        f"\n  Total checks: {PASS_COUNT + FAIL_COUNT}  |  Passed: {PASS_COUNT}  |  Failed: {FAIL_COUNT}"
    )
    print(f"\n  OVERALL: {'ALL PASSED ✓' if all_ok else 'SOME FAILED ✗'}")

    return all_ok


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
