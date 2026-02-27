#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for position encoding modules — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPES EXPLAINED:
================================================================================

NUMERICAL VALIDATION:
    - Compares actual OUTPUT VALUES between PyTorch and TTSim
    - Uses np.allclose() with tolerance: rtol=1e-4, atol=1e-5
    - Reports: max absolute difference, mean absolute difference
    - PASS if: all values within tolerance
    - FAIL if: any value exceeds tolerance

SHAPE VALIDATION ONLY:
    - Compares only OUTPUT DIMENSIONS (shapes) between PyTorch and TTSim
    - NO numerical value comparison
    - PASS if: all shapes match exactly
    - FAIL if: any shape mismatch detected

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: PositionEmbeddingSine — NUMERICAL VALIDATION
    Formula: PE(pos,2i) = sin(pos/T^(2i/D)), PE(pos,2i+1) = cos(pos/T^(2i/D))
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6),
                minimum_input (1x1), non-square, no-mask, partial-mask,
                unnormalised, parameter sweeps
    WHY NUMERICAL: Pure numpy math operations (sin/cos), deterministic

MODULE 2: PositionEmbeddingLearned — NUMERICAL VALIDATION (weights synced)
    Architecture: Two Embedding(50, D) tables → lookup → broadcast → concat
    Edge Cases: positive, negative, zeros, mixed, small, large,
                minimum_input (4x4), non-square, large-D
    WHY NUMERICAL: Simple lookup/broadcast, weights synced from PyTorch

MODULE 3: build_position_encoding — SHAPE + TYPE VALIDATION
    Factory function: maps 'sine'/'v2' → Sine, 'learned'/'v3' → Learned
    Edge Cases: sine, learned, v2/v3 aliases, invalid type
    WHY SHAPE: Factory correctness validated via type check + shape match

MODULE 4: PositionEmbeddingSine Masks — NUMERICAL VALIDATION
    Tests how different mask patterns affect cumulative sum and final encoding
    Edge Cases: no_mask, top_rows, left_cols, checkerboard, corner
    WHY NUMERICAL: Mask handling is critical for correct pos encoding

MODULE 5: PositionEmbeddingSine Components — NUMERICAL VALIDATION
    Validates y/x component separation and spatial invariance properties
    Tests: y-pos constant across columns, x-pos constant across rows
    WHY NUMERICAL: Verifies structural correctness of encoding

================================================================================
EDGE CASES TESTED (MANDATORY — all numerical modules):
================================================================================

'positive'       — Standard positive values (1.0 - 2.0) - baseline test
'negative'       — All negative values (-2.0 to -1.0) - tests sign handling
'zeros'          — All zeros - tests zero-mask cumsum edge case
'mixed'          — Mix of positive/negative values - tests real-world distribution
'small'          — Very small values (~1e-6) - tests numerical precision near zero
'large'          — Very large values (~1e6) - tests numerical overflow handling
'minimum_input'  — Smallest valid input size - degenerate/boundary case

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_position_encoding_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_position_encoding_unit.py
================================================================================
"""

import os
import sys
import math
import pytest
import torch
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# PyTorch implementations
from workloads.Deformable_DETR.reference.position_encoding import (
    PositionEmbeddingSine as PositionEmbeddingSinePyTorch,
    PositionEmbeddingLearned as PositionEmbeddingLearnedPyTorch,
    build_position_encoding as build_position_encoding_pytorch,
)

# TTSim implementations
from workloads.Deformable_DETR.models.position_encoding_ttsim import (
    PositionEmbeddingSine as PositionEmbeddingSineTTSim,
    PositionEmbeddingLearned as PositionEmbeddingLearnedTTSim,
    build_position_encoding as build_position_encoding_ttsim,
)

# Utilities
from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPyTorch
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTTSim
from ttsim.ops.tensor import SimTensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 1e-4
ATOL = 1e-5
SEED = 42


# ---------------------------------------------------------------------------
# Terminal Colors (ANSI escape codes)
# ---------------------------------------------------------------------------
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def success(s):
        return f"{Colors.GREEN}{s}{Colors.RESET}"

    @staticmethod
    def fail(s):
        return f"{Colors.RED}{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def warn(s):
        return f"{Colors.YELLOW}{s}{Colors.RESET}"

    @staticmethod
    def warning(s):
        return f"{Colors.YELLOW}{s}{Colors.RESET}"

    @staticmethod
    def info(s):
        return f"{Colors.CYAN}{s}{Colors.RESET}"

    @staticmethod
    def header(s):
        return f"{Colors.BOLD}{Colors.MAGENTA}{s}{Colors.RESET}"

    @staticmethod
    def dim(s):
        return f"{Colors.DIM}{s}{Colors.RESET}"

    @staticmethod
    def bold(s):
        return f"{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def cyan(s):
        return f"{Colors.CYAN}{s}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Visual separators
# ---------------------------------------------------------------------------
SUMMARY_LINE = "═" * 65
DIVIDER_LINE = "─" * 65


# ---------------------------------------------------------------------------
# Edge case descriptions
# ---------------------------------------------------------------------------
EDGE_CASE_DESC = {
    "positive": "Standard positive values (1.0 - 2.0) - baseline test",
    "negative": "All negative values (-2.0 to -1.0) - tests sign handling",
    "zeros": "All zeros - tests zero-mask cumsum edge case",
    "mixed": "Mix of positive/negative values - tests real-world distribution",
    "small": "Very small values (~1e-6) - tests numerical precision near zero",
    "large": "Very large values (~1e6) - tests numerical overflow handling",
    "minimum_input": "Smallest valid input size - degenerate/boundary case",
    "baseline": "Standard configuration with random input",
    "scale": "Different spatial resolution",
    "batch": "Multi-batch test",
    "no_mask": "No masked positions — uniform cumsum",
    "top_rows": "Top rows masked — shifts y cumsum",
    "left_cols": "Left columns masked — shifts x cumsum",
    "checkerboard": "Alternating masked pixels — irregular cumsum",
    "bottom_right": "Bottom-right quadrant masked",
    "non_square": "Non-square spatial dimensions",
    "unnormalised": "Unnormalised position embedding (no scale)",
}


# ---------------------------------------------------------------------------
# Report data collector  (populated by tests, consumed by _write_report)
# ---------------------------------------------------------------------------
REPORT_SECTIONS = []  # list of section dicts
FAILED_TESTS = []  # track all failures for summary
TEST_RESULTS = []  # detailed per-test results for report
MODULE_STATS = {}  # {module: {shape_passed, shape_total, num_passed, num_total}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def torch_to_simtensor(tensor: torch.Tensor, name: str = "tensor") -> SimTensor:
    """Convert a PyTorch tensor to a SimTensor."""
    return SimTensor(
        {
            "name": name,
            "shape": list(tensor.shape),
            "data": tensor.detach().cpu().numpy().copy(),
            "dtype": np.dtype(np.float32),
        }
    )


def _to_numpy(x):
    """Coerce PyTorch tensor / SimTensor / ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, SimTensor):
        return x.data
    return np.asarray(x)


def generate_test_data(shape, data_type):
    """Generate test data based on type."""
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) + 1.0
    elif data_type == "negative":
        return -np.random.rand(*shape).astype(np.float32) - 1.0
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)
    elif data_type == "small":
        return np.random.rand(*shape).astype(np.float32) * 1e-6
    elif data_type == "large":
        return np.random.rand(*shape).astype(np.float32) * 1e6
    else:
        return np.random.randn(*shape).astype(np.float32)


def _fmt_samples(arr, n=10):
    """First *n* values, formatted for a markdown table cell."""
    flat = np.asarray(arr).flatten()
    return ", ".join(f"{v:.6f}" for v in flat[:n])


def _compact_shape(shape):
    """Format shape as [1,64,8,8] (no spaces) for inline display."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def make_nested_tensors(x_torch, mask_torch):
    """Create matching PyTorch and TTSim NestedTensors."""
    nt_pt = NestedTensorPyTorch(x_torch, mask_torch)
    x_tt = torch_to_simtensor(x_torch, "input")
    mask_tt = mask_torch.detach().cpu().numpy()
    nt_tt = NestedTensorTTSim(x_tt, mask_tt)
    return nt_pt, nt_tt


def generate_mask(B, H, W, mask_type="no_mask"):
    """Generate boolean mask tensor of type mask_type."""
    if mask_type == "no_mask":
        return torch.zeros(B, H, W, dtype=torch.bool)
    elif mask_type == "top_rows":
        m = torch.zeros(B, H, W, dtype=torch.bool)
        m[:, : max(1, H // 4), :] = True
        return m
    elif mask_type == "left_cols":
        m = torch.zeros(B, H, W, dtype=torch.bool)
        m[:, :, : max(1, W // 3)] = True
        return m
    elif mask_type == "checkerboard":
        m = np.array(
            [
                [[(r + c) % 2 == 0 for c in range(W)] for r in range(H)]
                for _ in range(B)
            ],
            dtype=bool,
        )
        return torch.from_numpy(m)
    elif mask_type == "bottom_right":
        m = torch.zeros(B, H, W, dtype=torch.bool)
        m[:, H // 2 :, W // 2 :] = True
        return m
    else:
        return torch.zeros(B, H, W, dtype=torch.bool)


def print_test_linear(
    module,
    edge_case,
    edge_desc,
    input_shape,
    shape_line,
    shape_ok,
    is_numerical=True,
    num_ok=None,
    max_diff=None,
    mean_diff=None,
    rtol=RTOL,
    atol=ATOL,
    failure_reason="",
    pt_data=None,
    tt_data=None,
    input_samples=None,
):
    """Print test result in clean tree-style linear format."""
    passed = shape_ok and (num_ok if is_numerical else True)

    print(f"\nMODULE: {Colors.bold(module)}")
    print(f"├─ EDGE CASE: {Colors.warn(edge_case)} ({edge_desc})")
    print(f"├─ INPUT: {input_shape}")

    # Show input float samples when provided
    if input_samples:
        for sname, sarr in input_samples.items():
            flat = np.asarray(sarr).flatten()
            sstr = ", ".join(f"{v:.6f}" for v in flat[:5])
            print(f"├─ INPUT {sname}[0:5]: [{sstr}]")

    # Shape line
    shape_status = Colors.success("✓ MATCH") if shape_ok else Colors.fail("✗ MISMATCH")
    print(f"├─ SHAPE: {shape_line} → {shape_status}")

    # Numerical line
    if is_numerical and max_diff is not None:
        if num_ok:
            num_status = Colors.success(f"✓ PASS (tol: rtol={rtol}, atol={atol})")
        else:
            num_status = Colors.fail("✗ FAIL")
        print(
            f"├─ NUMERICAL: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} → {num_status}"
        )

    # Sample output values for numerical tests
    if is_numerical and pt_data is not None and tt_data is not None:
        pt_arr = pt_data.flat[:5] if hasattr(pt_data, "flat") else pt_data
        tt_arr = tt_data.flat[:5] if hasattr(tt_data, "flat") else tt_data
        pt_str = ", ".join(f"{v:.6f}" for v in pt_arr)
        tt_str = ", ".join(f"{v:.6f}" for v in tt_arr)
        print(f"├─ PT OUTPUT[0:5]: [{pt_str}]")
        print(f"├─ TT OUTPUT[0:5]: [{tt_str}]")

    # Failure reason
    if not passed and failure_reason:
        print(f"├─ FAILURE REASON: {Colors.fail(failure_reason)}")

    # Result
    result_str = Colors.success("✓ PASS") if passed else Colors.fail("✗ FAIL")
    print(f"└─ RESULT: {result_str}")


def print_summary():
    """Print the final summary table."""
    print(f"\n{SUMMARY_LINE}")
    print("SUMMARY")
    print(SUMMARY_LINE)
    print(f"{'MODULE':<32}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

    total_sp = total_st = total_np = total_nt = 0
    all_passed = True

    for name, stats in MODULE_STATS.items():
        sp, st = stats["shape_passed"], stats["shape_total"]
        total_sp += sp
        total_st += st

        shape_str = f"{sp}/{st}"

        if stats["num_total"] is not None:
            np_, nt = stats["num_passed"], stats["num_total"]
            total_np += np_
            total_nt += nt
            num_str = f"{np_}/{nt}"
            mod_pass = (sp == st) and (np_ == nt)
        else:
            num_str = "N/A"
            mod_pass = sp == st

        if not mod_pass:
            all_passed = False

        status = Colors.success("✓ PASS") if mod_pass else Colors.fail("✗ FAIL")
        print(f"{name:<32}{shape_str:<12}{num_str:<12}{status}")

    print(DIVIDER_LINE)

    total_num_str = f"{total_np}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("✓ PASS") if all_passed else Colors.fail("✗ FAIL")
    print(f"{'TOTAL':<32}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

    if FAILED_TESTS:
        print(f"\n{Colors.fail('FAILED TESTS:')}")
        for ft in FAILED_TESTS:
            diff_str = f"max_diff={ft['max_diff']:.2e}" if ft.get("max_diff") else ""
            atol_val = ft.get("atol", ATOL)
            gt_str = f" > atol={atol_val}" if ft.get("max_diff") else ""
            print(f"- {ft['module']} | {ft['edge_case']} values | {diff_str}{gt_str}")

    print(SUMMARY_LINE)


# ---------------------------------------------------------------------------
# Report tee-stream (only when running as __main__)
# ---------------------------------------------------------------------------
class _TeeStream:
    """Write to both a file and the real stdout."""

    def __init__(self, file_handle, original_stdout):
        self._file = file_handle
        self._stdout = original_stdout
        self.encoding = getattr(original_stdout, "encoding", "utf-8")

    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._stdout.fileno()


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — PositionEmbeddingSine (numerical, multiple edge cases)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_sine = "test_position_embedding_sine"
test_cases_sine = [
    # (description, batch, channels, height, width, num_pos_feats, temperature, normalize, data_type, mask_type, category)
    # --- Baseline tests ---
    (
        "Sine 28x28 normalised",
        2,
        256,
        28,
        28,
        128,
        10000,
        True,
        "positive",
        "no_mask",
        "baseline",
    ),
    (
        "Sine 16x16 normalised",
        1,
        64,
        16,
        16,
        32,
        10000,
        True,
        "positive",
        "no_mask",
        "baseline",
    ),
    (
        "Sine 28x28 batch=4",
        4,
        256,
        28,
        28,
        128,
        10000,
        True,
        "positive",
        "no_mask",
        "batch",
    ),
    (
        "Sine 14x28 non-square",
        2,
        128,
        14,
        28,
        64,
        10000,
        True,
        "positive",
        "no_mask",
        "non_square",
    ),
    (
        "Sine 32x8 non-square",
        1,
        64,
        32,
        8,
        64,
        10000,
        True,
        "positive",
        "no_mask",
        "non_square",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "Sine negative input",
        1,
        128,
        16,
        16,
        64,
        10000,
        True,
        "negative",
        "no_mask",
        "edge_value",
    ),
    (
        "Sine zero input",
        1,
        128,
        16,
        16,
        64,
        10000,
        True,
        "zeros",
        "no_mask",
        "edge_value",
    ),
    (
        "Sine mixed input",
        1,
        128,
        16,
        16,
        64,
        10000,
        True,
        "mixed",
        "no_mask",
        "edge_value",
    ),
    (
        "Sine small values (1e-6)",
        1,
        128,
        16,
        16,
        64,
        10000,
        True,
        "small",
        "no_mask",
        "edge_value",
    ),
    (
        "Sine large values (1e6)",
        1,
        128,
        16,
        16,
        64,
        10000,
        True,
        "large",
        "no_mask",
        "edge_value",
    ),
    # --- Edge case: Minimum input ---
    (
        "Sine 1x1 minimum",
        1,
        64,
        1,
        1,
        32,
        10000,
        True,
        "positive",
        "no_mask",
        "minimum_input",
    ),
    (
        "Sine 1xW strip",
        1,
        64,
        1,
        16,
        32,
        10000,
        True,
        "positive",
        "no_mask",
        "edge_shape",
    ),
    (
        "Sine Hx1 strip",
        1,
        64,
        16,
        1,
        32,
        10000,
        True,
        "positive",
        "no_mask",
        "edge_shape",
    ),
    # --- Edge case: Unnormalised ---
    (
        "Sine unnormalised",
        1,
        64,
        16,
        16,
        32,
        10000,
        False,
        "positive",
        "no_mask",
        "unnormalised",
    ),
    # --- Edge case: Parameter sweeps ---
    (
        "Sine D=256 T=10000",
        1,
        128,
        12,
        12,
        256,
        10000,
        True,
        "positive",
        "no_mask",
        "param_sweep",
    ),
    (
        "Sine D=64 T=100",
        1,
        128,
        12,
        12,
        64,
        100,
        True,
        "positive",
        "no_mask",
        "param_sweep",
    ),
    (
        "Sine D=64 T=20000",
        1,
        128,
        12,
        12,
        64,
        20000,
        True,
        "positive",
        "no_mask",
        "param_sweep",
    ),
]


@pytest.mark.unit
def test_position_embedding_sine():
    """Test PositionEmbeddingSine: numerical validation across data types, shapes, and parameters."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        ch,
        h,
        w,
        D,
        T,
        normalize,
        data_type,
        mask_type,
        category,
    ) in enumerate(test_cases_sine):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        shape = [batch, ch, h, w]
        test_data = generate_test_data(shape, data_type)
        mask_torch = generate_mask(batch, h, w, mask_type)

        # PyTorch
        x_torch = torch.from_numpy(test_data)
        nt_pt, nt_tt = make_nested_tensors(x_torch, mask_torch)

        pt_mod = PositionEmbeddingSinePyTorch(
            num_pos_feats=D, temperature=T, normalize=normalize
        )
        pt_mod.eval()
        tt_mod = PositionEmbeddingSineTTSim(
            f"sine_test_{tno}", num_pos_feats=D, temperature=T, normalize=normalize
        )

        # Forward
        with torch.no_grad():
            out_pt = pt_mod(nt_pt)
        out_tt = tt_mod(nt_tt)

        # Validation
        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        # Print
        print_test_linear(
            module="PositionEmbeddingSine",
            edge_case=data_type if category.startswith("edge_value") else category,
            edge_desc=EDGE_CASE_DESC.get(
                data_type if category.startswith("edge_value") else category, tmsg
            ),
            input_shape=shape,
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=RTOL,
            atol=ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
            input_samples={"input": test_data, "mask": mask_torch.numpy()},
        )

        # Capture for report
        TEST_RESULTS.append(
            {
                "module": "PositionEmbeddingSine",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, tmsg),
                "input_shape": list(shape),
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "pt_stats": {
                    "mean": float(pt_d.mean()),
                    "std": float(pt_d.std()),
                    "min": float(pt_d.min()),
                    "max": float(pt_d.max()),
                },
                "tt_stats": {
                    "mean": float(tt_d.mean()),
                    "std": float(tt_d.std()),
                    "min": float(tt_d.min()),
                    "max": float(tt_d.max()),
                },
                "passed": ok,
                "note": f"D={D}, T={T}, normalize={normalize}, mask={mask_type}",
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "PositionEmbeddingSine",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        # Report row
        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{shape}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        # Detail block
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, tmsg)}\n\n"
            f"**Config:** D={D}, T={T}, normalize={normalize}, mask={mask_type}\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- input: `[{_fmt_samples(test_data)}]`\n"
            f"- mask:  `[{_fmt_samples(mask_torch.numpy().astype(np.float32))}]`\n\n"
            f"| Metric | PyTorch | TTSim | Difference |\n"
            f"|:-------|--------:|------:|----------:|\n"
            f"| Mean | {pt_d.mean():.8f} | {tt_d.mean():.8f} | {abs(pt_d.mean()-tt_d.mean()):.2e} |\n"
            f"| Std  | {pt_d.std():.8f} | {tt_d.std():.8f} | {abs(pt_d.std()-tt_d.std()):.2e} |\n"
            f"| Min  | {pt_d.min():.8f} | {tt_d.min():.8f} | {abs(pt_d.min()-tt_d.min()):.2e} |\n"
            f"| Max  | {pt_d.max():.8f} | {tt_d.max():.8f} | {abs(pt_d.max()-tt_d.max()):.2e} |\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    # Module stats
    MODULE_STATS["PositionEmbeddingSine"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_sine),
        "num_passed": num_passed,
        "num_total": len(test_cases_sine),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "PositionEmbeddingSine",
            "description": "Sinusoidal position encoding — deterministic sin/cos computation",
            "passed": passed,
            "total": len(test_cases_sine),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_sine
    ), f"PositionEmbeddingSine: {passed}/{len(test_cases_sine)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — PositionEmbeddingLearned (numerical, weights synced)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_learned = "test_position_embedding_learned"
test_cases_learned = [
    # (description, batch, channels, height, width, num_pos_feats, data_type, category)
    # --- Baseline tests ---
    ("Learned 28x28 standard", 2, 256, 28, 28, 128, "positive", "baseline"),
    ("Learned 10x10 small", 1, 64, 10, 10, 32, "positive", "baseline"),
    ("Learned 14x28 non-square", 2, 128, 14, 28, 64, "positive", "non_square"),
    ("Learned 28x28 batch=4", 4, 256, 28, 28, 128, "positive", "batch"),
    ("Learned 20x20 D=256", 1, 512, 20, 20, 256, "positive", "param_sweep"),
    # --- Edge case: Values (mandatory) ---
    ("Learned negative input", 1, 128, 16, 16, 64, "negative", "edge_value"),
    ("Learned zero input", 1, 128, 16, 16, 64, "zeros", "edge_value"),
    ("Learned mixed input", 1, 128, 16, 16, 64, "mixed", "edge_value"),
    ("Learned small values (1e-6)", 1, 128, 16, 16, 64, "small", "edge_value"),
    ("Learned large values (1e6)", 1, 128, 16, 16, 64, "large", "edge_value"),
    # --- Edge case: Minimum input ---
    ("Learned 4x4 minimum", 1, 64, 4, 4, 32, "positive", "minimum_input"),
]


@pytest.mark.unit
def test_position_embedding_learned():
    """Test PositionEmbeddingLearned: numerical validation with weights synced from PyTorch."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, D, data_type, category) in enumerate(
        test_cases_learned
    ):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        shape = [batch, ch, h, w]
        test_data = generate_test_data(shape, data_type)
        mask_torch = torch.zeros(batch, h, w, dtype=torch.bool)

        x_torch = torch.from_numpy(test_data)
        nt_pt, nt_tt = make_nested_tensors(x_torch, mask_torch)

        # Create modules
        pt_mod = PositionEmbeddingLearnedPyTorch(num_pos_feats=D)
        pt_mod.eval()
        tt_mod = PositionEmbeddingLearnedTTSim(f"learned_test_{tno}", num_pos_feats=D)

        # Sync weights from PyTorch to TTSim
        tt_mod.row_embed_weight = pt_mod.row_embed.weight.detach().cpu().numpy().copy()
        tt_mod.col_embed_weight = pt_mod.col_embed.weight.detach().cpu().numpy().copy()

        # Forward
        with torch.no_grad():
            out_pt = pt_mod(nt_pt)
        out_tt = tt_mod(nt_tt)

        # Validation
        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        # Print
        print_test_linear(
            module="PositionEmbeddingLearned",
            edge_case=data_type if category.startswith("edge_value") else category,
            edge_desc=EDGE_CASE_DESC.get(
                data_type if category.startswith("edge_value") else category, tmsg
            ),
            input_shape=shape,
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=RTOL,
            atol=ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
            input_samples={"input": test_data},
        )

        # Capture for report
        TEST_RESULTS.append(
            {
                "module": "PositionEmbeddingLearned",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, tmsg),
                "input_shape": list(shape),
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "pt_stats": {
                    "mean": float(pt_d.mean()),
                    "std": float(pt_d.std()),
                    "min": float(pt_d.min()),
                    "max": float(pt_d.max()),
                },
                "tt_stats": {
                    "mean": float(tt_d.mean()),
                    "std": float(tt_d.std()),
                    "min": float(tt_d.min()),
                    "max": float(tt_d.max()),
                },
                "passed": ok,
                "note": f"D={D}, weights synced from PyTorch",
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "PositionEmbeddingLearned",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        # Report row
        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{shape}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        # Detail block
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, tmsg)}\n\n"
            f"**Config:** D={D}, weights synced from PyTorch\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- input: `[{_fmt_samples(test_data)}]`\n\n"
            f"| Metric | PyTorch | TTSim | Difference |\n"
            f"|:-------|--------:|------:|----------:|\n"
            f"| Mean | {pt_d.mean():.8f} | {tt_d.mean():.8f} | {abs(pt_d.mean()-tt_d.mean()):.2e} |\n"
            f"| Std  | {pt_d.std():.8f} | {tt_d.std():.8f} | {abs(pt_d.std()-tt_d.std()):.2e} |\n"
            f"| Min  | {pt_d.min():.8f} | {tt_d.min():.8f} | {abs(pt_d.min()-tt_d.min()):.2e} |\n"
            f"| Max  | {pt_d.max():.8f} | {tt_d.max():.8f} | {abs(pt_d.max()-tt_d.max()):.2e} |\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    # Module stats
    MODULE_STATS["PosEmbLearned"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_learned),
        "num_passed": num_passed,
        "num_total": len(test_cases_learned),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "PositionEmbeddingLearned",
            "description": "Learned position encoding — Embedding lookup with synced weights",
            "passed": passed,
            "total": len(test_cases_learned),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_learned
    ), f"PositionEmbeddingLearned: {passed}/{len(test_cases_learned)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — PositionEmbeddingSine Mask Variations (numerical)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_masks = "test_sine_mask_variations"
test_cases_masks = [
    # (description, batch, channels, height, width, mask_type, data_type, category)
    # --- Mask patterns ---
    ("Sine no mask", 2, 128, 20, 20, "no_mask", "positive", "no_mask"),
    ("Sine top rows masked", 2, 128, 20, 20, "top_rows", "positive", "top_rows"),
    ("Sine left columns masked", 2, 128, 20, 20, "left_cols", "positive", "left_cols"),
    (
        "Sine checkerboard mask",
        2,
        128,
        20,
        20,
        "checkerboard",
        "positive",
        "checkerboard",
    ),
    (
        "Sine bottom-right masked",
        2,
        128,
        20,
        20,
        "bottom_right",
        "positive",
        "bottom_right",
    ),
    # --- Mask + edge value combos (mandatory) ---
    ("Sine mask + negative", 1, 128, 16, 16, "top_rows", "negative", "edge_value"),
    ("Sine mask + zeros", 1, 128, 16, 16, "top_rows", "zeros", "edge_value"),
    ("Sine mask + mixed", 1, 128, 16, 16, "checkerboard", "mixed", "edge_value"),
    ("Sine mask + small", 1, 128, 16, 16, "left_cols", "small", "edge_value"),
    ("Sine mask + large", 1, 128, 16, 16, "bottom_right", "large", "edge_value"),
    # --- Minimum input with mask ---
    ("Sine 4x4 mask minimum", 1, 64, 4, 4, "top_rows", "positive", "minimum_input"),
]


@pytest.mark.unit
def test_sine_mask_variations():
    """Test PositionEmbeddingSine with different mask patterns and edge cases."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    D = 64
    T = 10000

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, mask_type, data_type, category) in enumerate(
        test_cases_masks
    ):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        shape = [batch, ch, h, w]
        test_data = generate_test_data(shape, data_type)
        mask_torch = generate_mask(batch, h, w, mask_type)

        x_torch = torch.from_numpy(test_data)
        nt_pt, nt_tt = make_nested_tensors(x_torch, mask_torch)

        pt_mod = PositionEmbeddingSinePyTorch(
            num_pos_feats=D, temperature=T, normalize=True
        )
        pt_mod.eval()
        tt_mod = PositionEmbeddingSineTTSim(
            f"sine_mask_{tno}", num_pos_feats=D, temperature=T, normalize=True
        )

        with torch.no_grad():
            out_pt = pt_mod(nt_pt)
        out_tt = tt_mod(nt_tt)

        # Validation
        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        # Determine edge label
        edge_label = mask_type if not category.startswith("edge_value") else data_type
        edge_desc_val = EDGE_CASE_DESC.get(edge_label, tmsg)

        print_test_linear(
            module="SineMaskVariations",
            edge_case=edge_label,
            edge_desc=edge_desc_val,
            input_shape=shape,
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=RTOL,
            atol=ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
            input_samples={
                "input": test_data,
                "mask": mask_torch.numpy().astype(np.float32),
            },
        )

        TEST_RESULTS.append(
            {
                "module": "SineMaskVariations",
                "validation_type": "NUMERICAL",
                "edge_case": f"{mask_type}+{data_type}",
                "edge_desc": f"mask={mask_type}, data={data_type}",
                "input_shape": list(shape),
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "pt_stats": {
                    "mean": float(pt_d.mean()),
                    "std": float(pt_d.std()),
                    "min": float(pt_d.min()),
                    "max": float(pt_d.max()),
                },
                "tt_stats": {
                    "mean": float(tt_d.mean()),
                    "std": float(tt_d.std()),
                    "min": float(tt_d.min()),
                    "max": float(tt_d.max()),
                },
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, f"{mask_type}+{data_type}", mx))
            FAILED_TESTS.append(
                {
                    "module": "SineMaskVariations",
                    "test": tmsg,
                    "edge_case": f"{mask_type}+{data_type}",
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        masked_pct = float(mask_torch.sum()) / mask_torch.numel() * 100
        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {mask_type} | {data_type} | `{shape}` "
            f"| {masked_pct:.0f}% | {mx:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Mask:** `{mask_type}` ({masked_pct:.0f}% masked) | **Data:** `{data_type}`\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- input: `[{_fmt_samples(test_data)}]`\n"
            f"- mask:  `[{_fmt_samples(mask_torch.numpy().astype(np.float32))}]`\n\n"
            f"| Metric | PyTorch | TTSim | Difference |\n"
            f"|:-------|--------:|------:|----------:|\n"
            f"| Mean | {pt_d.mean():.8f} | {tt_d.mean():.8f} | {abs(pt_d.mean()-tt_d.mean()):.2e} |\n"
            f"| Std  | {pt_d.std():.8f} | {tt_d.std():.8f} | {abs(pt_d.std()-tt_d.std()):.2e} |\n"
            f"| Min  | {pt_d.min():.8f} | {tt_d.min():.8f} | {abs(pt_d.min()-tt_d.min()):.2e} |\n"
            f"| Max  | {pt_d.max():.8f} | {tt_d.max():.8f} | {abs(pt_d.max()-tt_d.max()):.2e} |\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["SineMaskVariations"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_masks),
        "num_passed": num_passed,
        "num_total": len(test_cases_masks),
    }

    hdr = (
        "| # | Test Case | Mask | Data | Input | Masked% | Max Diff | Result |\n"
        "|:--|:----------|:-----|:-----|:------|:--------|:---------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "SineMaskVariations",
            "description": "Sine position encoding with varying mask patterns and edge case data",
            "passed": passed,
            "total": len(test_cases_masks),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_masks
    ), f"SineMaskVariations: {passed}/{len(test_cases_masks)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — PositionEmbeddingSine Component Analysis (y/x separation)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_components = "test_sine_component_analysis"
test_cases_components = [
    # (description, batch, channels, height, width, num_pos_feats, category)
    ("Components 24x24 standard", 2, 256, 24, 24, 128, "baseline"),
    ("Components 12x12 small", 1, 128, 12, 12, 64, "baseline"),
    ("Components 8x16 non-square", 2, 128, 8, 16, 64, "non_square"),
    # --- Minimum input ---
    ("Components 4x4 minimum", 1, 64, 4, 4, 32, "minimum_input"),
]


@pytest.mark.unit
def test_sine_component_analysis():
    """Test PositionEmbeddingSine y/x component separation and spatial invariance."""
    _seed()

    D_default = 128
    T = 10000

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, D, category) in enumerate(test_cases_components):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        shape = [batch, ch, h, w]
        x_torch = torch.randn(*shape)
        mask_torch = torch.zeros(batch, h, w, dtype=torch.bool)

        nt_pt, nt_tt = make_nested_tensors(x_torch, mask_torch)

        pt_mod = PositionEmbeddingSinePyTorch(
            num_pos_feats=D, temperature=T, normalize=True
        )
        pt_mod.eval()
        tt_mod = PositionEmbeddingSineTTSim(
            f"sine_comp_{tno}", num_pos_feats=D, temperature=T, normalize=True
        )

        with torch.no_grad():
            out_pt = pt_mod(nt_pt)
        out_tt = tt_mod(nt_tt)

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
        shape_ok = pt_shape == tt_shape

        # Full numerical comparison
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))

        # Component checks
        # y-component: channels [0, D)
        pt_y = pt_d[:, :D, :, :]
        tt_y = tt_d[:, :D, :, :]
        y_ok = bool(np.allclose(pt_y, tt_y, rtol=RTOL, atol=ATOL))

        # x-component: channels [D, 2D)
        pt_x = pt_d[:, D:, :, :]
        tt_x = tt_d[:, D:, :, :]
        x_ok = bool(np.allclose(pt_x, tt_x, rtol=RTOL, atol=ATOL))

        # Spatial invariance: y-pos constant across columns (no mask)
        y_col_invariant = True
        if w > 1:
            y_col_0 = tt_d[0, :D, :, 0]
            y_col_mid = tt_d[0, :D, :, w // 2]
            y_col_invariant = bool(np.allclose(y_col_0, y_col_mid, atol=1e-6))

        # Spatial invariance: x-pos constant across rows (no mask)
        x_row_invariant = True
        if h > 1:
            x_row_0 = tt_d[0, D:, 0, :]
            x_row_mid = tt_d[0, D:, h // 2, :]
            x_row_invariant = bool(np.allclose(x_row_0, x_row_mid, atol=1e-6))

        ok = (
            shape_ok
            and num_ok
            and y_ok
            and x_ok
            and y_col_invariant
            and x_row_invariant
        )
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(ok)

        reason = ""
        if not shape_ok:
            reason = "Shape mismatch"
        elif not num_ok:
            reason = f"Full numerical mismatch: max_diff={mx:.2e}"
        elif not y_ok:
            reason = "Y-component mismatch"
        elif not x_ok:
            reason = "X-component mismatch"
        elif not y_col_invariant:
            reason = "Y-pos NOT constant across columns"
        elif not x_row_invariant:
            reason = "X-pos NOT constant across rows"

        # Print detailed component analysis
        print(f"\nMODULE: {Colors.bold('SineComponentAnalysis')}")
        print(f"├─ EDGE CASE: {Colors.warn(category)} ({tmsg})")
        print(f"├─ INPUT: {shape}")
        print(
            f"├─ SHAPE: PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)} → "
            f"{Colors.success('✓ MATCH') if shape_ok else Colors.fail('✗ MISMATCH')}"
        )
        print(
            f"├─ NUMERICAL: max_diff={mx:.2e}, mean_diff={mn:.2e} → "
            f"{Colors.success('✓ PASS') if num_ok else Colors.fail('✗ FAIL')}"
        )
        print(
            f"├─ Y-COMPONENT [0:{D}]: "
            f"{Colors.success('✓ MATCH') if y_ok else Colors.fail('✗ MISMATCH')}"
        )
        print(
            f"├─ X-COMPONENT [{D}:{2*D}]: "
            f"{Colors.success('✓ MATCH') if x_ok else Colors.fail('✗ MISMATCH')}"
        )
        print(
            f"├─ Y-POS COLUMN INVARIANT: "
            f"{Colors.success('✓ YES') if y_col_invariant else Colors.fail('✗ NO')}"
        )
        print(
            f"├─ X-POS ROW INVARIANT: "
            f"{Colors.success('✓ YES') if x_row_invariant else Colors.fail('✗ NO')}"
        )
        if not ok and reason:
            print(f"├─ FAILURE REASON: {Colors.fail(reason)}")
        result_str = Colors.success("✓ PASS") if ok else Colors.fail("✗ FAIL")
        print(f"└─ RESULT: {result_str}")

        TEST_RESULTS.append(
            {
                "module": "SineComponentAnalysis",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": list(shape),
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": ok,
                "max_diff": mx,
                "mean_diff": mn,
                "pt_stats": None,
                "tt_stats": None,
                "passed": ok,
                "note": f"y_ok={y_ok}, x_ok={x_ok}, y_col={y_col_invariant}, x_row={x_row_invariant}",
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, mx))
            FAILED_TESTS.append(
                {
                    "module": "SineComponentAnalysis",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | `{shape}` | D={D} | y={y_ok} x={x_ok} | "
            f"col_inv={y_col_invariant} row_inv={x_row_invariant} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Config:** D={D}, T={T}, normalize=True, no mask\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}`\n\n"
            f"| Check | Result |\n"
            f"|:------|:-------|\n"
            f"| Shape match | {'✅' if shape_ok else '❌'} |\n"
            f"| Full numerical | {'✅' if num_ok else '❌'} (max_diff={mx:.2e}) |\n"
            f"| Y-component (channels 0:{D}) | {'✅' if y_ok else '❌'} |\n"
            f"| X-component (channels {D}:{2*D}) | {'✅' if x_ok else '❌'} |\n"
            f"| Y-pos constant across columns | {'✅' if y_col_invariant else '❌'} |\n"
            f"| X-pos constant across rows | {'✅' if x_row_invariant else '❌'} |\n\n"
        )

    MODULE_STATS["SineComponents"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_components),
        "num_passed": num_passed,
        "num_total": len(test_cases_components),
    }

    hdr = (
        "| # | Test Case | Input | D | Components | Invariance | Result |\n"
        "|:--|:----------|:------|:--|:-----------|:-----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "SineComponentAnalysis",
            "description": "Y/X component separation and spatial invariance verification",
            "passed": passed,
            "total": len(test_cases_components),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_components
    ), f"SineComponentAnalysis: {passed}/{len(test_cases_components)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — build_position_encoding factory (shape + type validation)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_factory = "test_build_position_encoding"
test_cases_factory = [
    # (description, position_embedding, hidden_dim, expected_pt_type, expected_tt_type, should_raise, category)
    (
        "Factory sine",
        "sine",
        256,
        "PositionEmbeddingSine",
        "PositionEmbeddingSine",
        False,
        "baseline",
    ),
    (
        "Factory learned",
        "learned",
        256,
        "PositionEmbeddingLearned",
        "PositionEmbeddingLearned",
        False,
        "baseline",
    ),
    (
        "Factory v2 alias",
        "v2",
        256,
        "PositionEmbeddingSine",
        "PositionEmbeddingSine",
        False,
        "alias",
    ),
    (
        "Factory v3 alias",
        "v3",
        256,
        "PositionEmbeddingLearned",
        "PositionEmbeddingLearned",
        False,
        "alias",
    ),
    (
        "Factory v2 hidden=128",
        "v2",
        128,
        "PositionEmbeddingSine",
        "PositionEmbeddingSine",
        False,
        "param_sweep",
    ),
    (
        "Factory sine hidden=512",
        "sine",
        512,
        "PositionEmbeddingSine",
        "PositionEmbeddingSine",
        False,
        "param_sweep",
    ),
    ("Factory invalid type", "banana", 256, None, None, True, "error"),
]


@pytest.mark.unit
def test_build_position_encoding():
    """Test build_position_encoding factory: type dispatch, aliases, shape, and error handling."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    failed_cases = []

    B, H, W = 2, 14, 14

    for tno, (
        tmsg,
        pos_emb,
        hd,
        exp_pt_type,
        exp_tt_type,
        should_raise,
        category,
    ) in enumerate(test_cases_factory):

        class Args:
            hidden_dim = hd
            position_embedding = pos_emb

        if should_raise:
            # Test that both PyTorch and TTSim raise ValueError
            pt_raised = False
            tt_raised = False
            try:
                build_position_encoding_pytorch(Args())
            except ValueError:
                pt_raised = True
            try:
                build_position_encoding_ttsim(Args())
            except ValueError:
                tt_raised = True

            ok = pt_raised and tt_raised
            passed += int(ok)
            shape_passed += int(ok)

            reason = ""
            if not pt_raised:
                reason = "PyTorch did NOT raise ValueError"
            elif not tt_raised:
                reason = "TTSim did NOT raise ValueError"

            print(f"\nMODULE: {Colors.bold('build_position_encoding')}")
            print(f"├─ CONFIG: position_embedding='{pos_emb}', hidden_dim={hd}")
            print(
                f"├─ PT RAISES: {Colors.success('✓ YES') if pt_raised else Colors.fail('✗ NO')}"
            )
            print(
                f"├─ TT RAISES: {Colors.success('✓ YES') if tt_raised else Colors.fail('✗ NO')}"
            )
            if not ok and reason:
                print(f"├─ FAILURE REASON: {Colors.fail(reason)}")
            result_str = Colors.success("✓ PASS") if ok else Colors.fail("✗ FAIL")
            print(f"└─ RESULT: {result_str}")

            TEST_RESULTS.append(
                {
                    "module": "build_position_encoding",
                    "validation_type": "ERROR HANDLING",
                    "edge_case": "invalid_type",
                    "edge_desc": f'Should raise ValueError for "{pos_emb}"',
                    "input_shape": None,
                    "pt_shape": None,
                    "tt_shape": None,
                    "shape_ok": ok,
                    "num_ok": None,
                    "max_diff": None,
                    "mean_diff": None,
                    "pt_stats": None,
                    "tt_stats": None,
                    "passed": ok,
                }
            )

            if not ok:
                failed_cases.append((tno, tmsg, "error_handling", 0))
                FAILED_TESTS.append(
                    {
                        "module": "build_position_encoding",
                        "test": tmsg,
                        "edge_case": "error_handling",
                        "max_diff": 0,
                    }
                )

            tag = "✅ PASS" if ok else "🔴 **FAIL**"
            rows.append(f"| {tno} | {tmsg} | `{pos_emb}` | {hd} | ValueError | {tag} |")

            status_badge = "🟢" if ok else "🔴"
            detail_blocks.append(
                f"---\n\n"
                f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
                f"**Config:** position_embedding='{pos_emb}', hidden_dim={hd}\n\n"
                f"**Expected:** ValueError raised\n\n"
                f"| Framework | Raised | Result |\n"
                f"|:----------|:-------|:-------|\n"
                f"| PyTorch | {'✅ Yes' if pt_raised else '❌ No'} | {'✅' if pt_raised else '❌'} |\n"
                f"| TTSim   | {'✅ Yes' if tt_raised else '❌ No'} | {'✅' if tt_raised else '❌'} |\n\n"
            )
            continue

        # Normal build
        pe_pt = build_position_encoding_pytorch(Args())
        pe_tt = build_position_encoding_ttsim(Args())

        # Type check
        type_ok = (
            type(pe_pt).__name__ == exp_pt_type and type(pe_tt).__name__ == exp_tt_type
        )

        # Shape check via forward pass
        C = hd
        x_torch = torch.randn(B, C, H, W)
        mask_torch = torch.zeros(B, H, W, dtype=torch.bool)
        nt_pt, nt_tt = make_nested_tensors(x_torch, mask_torch)

        pe_pt.eval()
        with torch.no_grad():
            out_pt = pe_pt(nt_pt)
        out_tt = pe_tt(nt_tt)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
        shape_ok = pt_shape == tt_shape

        # For sine: also check numerical match
        num_ok = None
        mx = None
        mn = None
        if pos_emb in ("sine", "v2"):
            pt_d = _to_numpy(out_pt)
            tt_d = _to_numpy(out_tt)
            abs_diff = np.abs(pt_d - tt_d)
            mx = float(abs_diff.max())
            mn = float(abs_diff.mean())
            num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))

        ok = type_ok and shape_ok and (num_ok if num_ok is not None else True)
        passed += int(ok)
        shape_passed += int(shape_ok)

        reason = ""
        if not type_ok:
            reason = (
                f"Type mismatch: PT={type(pe_pt).__name__} vs expected {exp_pt_type}"
            )
        elif not shape_ok:
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif num_ok is not None and not num_ok:
            reason = f"Numerical mismatch: max_diff={mx:.2e}"

        print(f"\nMODULE: {Colors.bold('build_position_encoding')}")
        print(f"├─ CONFIG: position_embedding='{pos_emb}', hidden_dim={hd}")
        print(
            f"├─ TYPE: PT={type(pe_pt).__name__} | TT={type(pe_tt).__name__} → "
            f"{Colors.success('✓ MATCH') if type_ok else Colors.fail('✗ MISMATCH')}"
        )
        print(
            f"├─ SHAPE: PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)} → "
            f"{Colors.success('✓ MATCH') if shape_ok else Colors.fail('✗ MISMATCH')}"
        )
        if num_ok is not None:
            print(
                f"├─ NUMERICAL: max_diff={mx:.2e} → "
                f"{Colors.success('✓ PASS') if num_ok else Colors.fail('✗ FAIL')}"
            )
        if not ok and reason:
            print(f"├─ FAILURE REASON: {Colors.fail(reason)}")
        result_str = Colors.success("✓ PASS") if ok else Colors.fail("✗ FAIL")
        print(f"└─ RESULT: {result_str}")

        TEST_RESULTS.append(
            {
                "module": "build_position_encoding",
                "validation_type": "TYPE + SHAPE",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": [B, C, H, W],
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "pt_stats": None,
                "tt_stats": None,
                "passed": ok,
                "note": f"type_ok={type_ok}",
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, mx or 0))
            FAILED_TESTS.append(
                {
                    "module": "build_position_encoding",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx or 0,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        num_str = f"{mx:.2e}" if mx is not None else "N/A"
        rows.append(
            f"| {tno} | {tmsg} | `{pos_emb}` | {hd} | `{pt_shape}` | {num_str} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Config:** position_embedding='{pos_emb}', hidden_dim={hd}\n\n"
            f"| Check | Expected | Actual | Result |\n"
            f"|:------|:---------|:-------|:-------|\n"
            f"| PT type | {exp_pt_type} | {type(pe_pt).__name__} | {'✅' if type_ok else '❌'} |\n"
            f"| TT type | {exp_tt_type} | {type(pe_tt).__name__} | {'✅' if type_ok else '❌'} |\n"
            f"| Shape | `{pt_shape}` | `{tt_shape}` | {'✅' if shape_ok else '❌'} |\n"
            f"| Numerical | — | {num_str} | {'✅' if (num_ok is None or num_ok) else '❌'} |\n\n"
        )

    MODULE_STATS["build_factory"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_factory),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Type | Hidden Dim | Output Shape | Numerical | Result |\n"
        "|:--|:----------|:-----|:-----------|:-------------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "build_position_encoding",
            "description": "Factory function — type dispatch, aliases, error handling",
            "passed": passed,
            "total": len(test_cases_factory),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_factory
    ), f"build_position_encoding: {passed}/{len(test_cases_factory)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Shape Inference (data=None)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_shape = "test_shape_inference"
test_cases_shape = [
    # (description, batch, channels, height, width, num_pos_feats, module_type, category)
    ("Shape sine 28x28", 2, 256, 28, 28, 128, "sine", "baseline"),
    ("Shape sine 14x14", 1, 128, 14, 14, 64, "sine", "baseline"),
    ("Shape sine 7x11", 2, 256, 7, 11, 128, "sine", "non_square"),
    ("Shape sine 1x1 minimum", 1, 64, 1, 1, 32, "sine", "minimum_input"),
    ("Shape learned 28x28", 2, 256, 28, 28, 128, "learned", "baseline"),
    ("Shape learned 10x10", 1, 128, 10, 10, 64, "learned", "baseline"),
    ("Shape learned 4x4 min", 1, 64, 4, 4, 32, "learned", "minimum_input"),
]


@pytest.mark.unit
def test_shape_inference():
    """Test position encoding shape inference (data=None mode)."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, D, mod_type, category) in enumerate(
        test_cases_shape
    ):
        # PyTorch reference (with data, for expected shape)
        x_torch = torch.randn(batch, ch, h, w)
        mask_torch = torch.zeros(batch, h, w, dtype=torch.bool)
        nt_pt = NestedTensorPyTorch(x_torch, mask_torch)

        if mod_type == "sine":
            pt_mod = PositionEmbeddingSinePyTorch(
                num_pos_feats=D, temperature=10000, normalize=True
            )
            pt_mod.eval()
        else:
            pt_mod = PositionEmbeddingLearnedPyTorch(num_pos_feats=D)
            pt_mod.eval()

        with torch.no_grad():
            out_pt = pt_mod(nt_pt)
        expected_shape = list(out_pt.shape)

        # TTSim shape inference (data=None)
        x_sim_none = SimTensor(
            {
                "name": "input_shape_only",
                "shape": [batch, ch, h, w],
                "data": None,
                "dtype": np.dtype(np.float32),
            }
        )
        # Mask as None-data too — but mask needs actual data for sine cumsum
        # For shape inference test, provide mask data (zeros) since sine needs it
        mask_np = np.zeros((batch, h, w), dtype=bool)
        nt_tt_shape = NestedTensorTTSim(x_sim_none, mask_np)

        if mod_type == "sine":
            tt_mod = PositionEmbeddingSineTTSim(
                f"shape_{mod_type}_{tno}",
                num_pos_feats=D,
                temperature=10000,
                normalize=True,
            )
        else:
            tt_mod = PositionEmbeddingLearnedTTSim(
                f"shape_{mod_type}_{tno}", num_pos_feats=D
            )

        out_tt = tt_mod(nt_tt_shape)
        tt_shape = out_tt.shape

        shape_ok = expected_shape == tt_shape

        ok = shape_ok
        passed += int(ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: expected={expected_shape} got={tt_shape}"

        print_test_linear(
            module=f"ShapeInference ({mod_type})",
            edge_case=category,
            edge_desc=tmsg,
            input_shape=[batch, ch, h, w],
            shape_line=f"Expected={_compact_shape(expected_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=False,
            failure_reason=reason,
        )

        TEST_RESULTS.append(
            {
                "module": f"ShapeInference ({mod_type})",
                "validation_type": "SHAPE ONLY",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": [batch, ch, h, w],
                "pt_shape": expected_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": None,
                "max_diff": None,
                "mean_diff": None,
                "pt_stats": None,
                "tt_stats": None,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, 0))
            FAILED_TESTS.append(
                {
                    "module": f"ShapeInference ({mod_type})",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": 0,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {mod_type} | `[{batch},{ch},{h},{w}]` | `{expected_shape}` | `{tt_shape}` | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Module:** {mod_type} | **Config:** D={D}\n\n"
            f"**Input Shape:** `[{batch},{ch},{h},{w}]` (data=None)\n\n"
            f"| Expected Shape | TTSim Shape | Match |\n"
            f"|:---------------|:------------|:------|\n"
            f"| `{expected_shape}` | `{tt_shape}` | {'✅' if shape_ok else '❌'} |\n\n"
        )

    MODULE_STATS["ShapeInference"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_shape),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Type | Input | Expected | TTSim | Result |\n"
        "|:--|:----------|:-----|:------|:---------|:------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "ShapeInference",
            "description": "Shape inference validation (data=None mode)",
            "passed": passed,
            "total": len(test_cases_shape),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_shape
    ), f"ShapeInference: {passed}/{len(test_cases_shape)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-runner with markdown report
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a simple, module-wise markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# Position Encoding Unit Test Report",
        f"**PyTorch vs TTSim Comparison** | **{total_passed}/{total_tests} passed** | {status}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exit Code: {exit_code}",
        "",
        "---",
        "",
    ]

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Module | Passed | Total | Status |")
    lines.append("|--------|--------|-------|--------|")
    for s in REPORT_SECTIONS:
        mod_status = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"| {s['name']} | {s['passed']} | {s['total']} | {mod_status} |")
    lines.append("")
    lines.append(f"**Total: {total_passed}/{total_tests} tests passed**")
    lines.append("")

    # Failed tests
    if FAILED_TESTS:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Tests")
        lines.append("")
        lines.append("| Module | Test | Edge Case | Max Diff |")
        lines.append("|--------|------|-----------|----------|")
        for ft in FAILED_TESTS:
            diff_str = f"{ft['max_diff']:.2e}" if ft.get("max_diff") else "N/A"
            lines.append(
                f"| {ft['module']} | {ft['test']} | {ft['edge_case']} | {diff_str} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Module details
    for s in REPORT_SECTIONS:
        mod_status = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"## {s['name']} ({s['passed']}/{s['total']} {mod_status})")
        if s.get("description"):
            lines.append(f"*{s['description']}*")
        lines.append("")
        lines.append(s["table"])
        lines.append("")

        failed = s.get("failed_cases", [])
        if failed:
            lines.append("**Failed Cases:**")
            for tno, tmsg, edge, diff in failed:
                diff_str = f"{diff:.2e}" if diff else "N/A"
                lines.append(f"- [{tno}] {tmsg} - {edge} (diff: {diff_str})")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append(f"- Tolerance: rtol={RTOL}, atol={ATOL}")
    lines.append(f"- Random Seed: {SEED}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sync_globals_from_pytest():
    """After pytest.main() re-imports this file under its real module name,
    copy the populated global collectors back into __main__ so that
    print_summary() and _write_markdown_report() see the test results.
    """
    this_file = os.path.normcase(os.path.abspath(__file__))
    main_mod = sys.modules.get("__main__")

    for mod in list(sys.modules.values()):
        if mod is main_mod or mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        if os.path.normcase(os.path.abspath(mod_file)) == this_file:
            for attr in (
                "MODULE_STATS",
                "REPORT_SECTIONS",
                "FAILED_TESTS",
                "TEST_RESULTS",
            ):
                src = getattr(mod, attr, None)
                dst = globals()[attr]
                if src is not None and src is not dst:
                    if isinstance(dst, dict):
                        dst.update(src)
                    elif isinstance(dst, list) and src:
                        dst.extend(src)
            break


if __name__ == "__main__":
    report_dir = os.path.join(os.path.dirname(__file__), "..", "unit_test_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "position_encoding_unit_test_report.md")
    output_path = os.path.join(report_dir, "position_encoding_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    # Print overall header
    print(f"\n{SUMMARY_LINE}")
    print(f"POSITION ENCODING UNIT TEST SUITE - PyTorch vs TTSim")
    print(f"{SUMMARY_LINE}\n")

    # Run pytest
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    # Sync globals back from the pytest-imported copy
    _sync_globals_from_pytest()

    # Final summary
    print_summary()

    # Restore stdout and close tee file
    sys.stdout = _original_stdout
    _tee_file.close()

    # Write structured markdown report
    _write_markdown_report(report_path, exit_code)

    print(f"\n{Colors.cyan(f'[Markdown report : {report_path}]')}")
    print(f"{Colors.cyan(f'[Full output log  : {output_path}]')}\n")
    sys.exit(exit_code)
