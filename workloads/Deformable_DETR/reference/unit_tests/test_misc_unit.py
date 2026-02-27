#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for misc utility modules — PyTorch vs TTSim comparison WITH EDGE CASES.

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

MODULE 1: NestedTensor — NUMERICAL VALIDATION
    Purpose: Data container bundling tensors with padding masks
    Methods: __init__, decompose, __repr__
    Edge Cases: negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Simple data wrapping, no computation

MODULE 2: interpolate (nearest) — NUMERICAL VALIDATION
    Purpose: Resize tensors using nearest neighbor interpolation
    Formula: output[i,j] = input[floor(i * H_in / H_out), floor(j * W_in / W_out)]
    Edge Cases: negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Pixel selection is deterministic, exact match expected

MODULE 3: interpolate (bilinear) — NUMERICAL VALIDATION
    Purpose: Resize tensors using bilinear interpolation
    Formula: weighted average of 4 nearest neighbours
    Edge Cases: negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: TTSim has PyTorch-compatible bilinear implementation

MODULE 4: interpolate (downsampling) — NUMERICAL VALIDATION
    Purpose: Reduce spatial resolution using nearest neighbor
    Edge Cases: negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Deterministic pixel selection, exact match expected

MODULE 5: nested_tensor_from_tensor_list — NUMERICAL VALIDATION
    Purpose: Batch variable-sized tensors with zero-padding and mask
    Process: Find max dims → pad with zeros → create boolean mask
    Edge Cases: negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Simple copy + zero-padding, exact match expected

MODULE 6: inverse_sigmoid — NUMERICAL VALIDATION
    Purpose: Compute logit (inverse sigmoid) function
    Formula: log(clamp(x, eps) / clamp(1-x, eps))
    Edge Cases: near_zero, near_one, mid_range, boundary, mixed, minimum_input
    WHY NUMERICAL: Elementwise math, fast to compute

MODULE 7: Shape Inference — SHAPE VALIDATION ONLY
    Purpose: Verify TTSim produces correct output shapes with data=None
    Modules: interpolate, nested_tensor_from_tensor_list
    Edge Cases: various configurations including minimum_input
    WHY SHAPE: Tests shape-inference-only code path (no numerical data)

================================================================================
EDGE CASES TESTED (MANDATORY — all numerical modules):
================================================================================

'positive'       — Standard positive values (1.0 - 2.0) - baseline test
'negative'       — All negative values (-2.0 to -1.0) - tests sign handling
'zeros'          — All zeros - tests zero-value propagation
'mixed'          — Mix of positive/negative values - tests real-world distribution
'small'          — Very small values (~1e-6) - tests numerical precision near zero
'large'          — Very large values (~1e6) - tests numerical overflow handling
'minimum_input'  — Smallest valid input size - tests degenerate/boundary case

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_misc_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_misc_unit.py
================================================================================
"""

import os
import sys
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
from workloads.Deformable_DETR.reference.misc import (
    NestedTensor as NestedTensorPyTorch,
    interpolate as interpolate_pytorch,
    nested_tensor_from_tensor_list as nested_tensor_from_tensor_list_pytorch,
    inverse_sigmoid as inverse_sigmoid_pytorch,
)

# TTSim implementations
from workloads.Deformable_DETR.util.misc_ttsim import (
    NestedTensor as NestedTensorTTSim,
    interpolate as interpolate_ttsim,
    nested_tensor_from_tensor_list as nested_tensor_from_tensor_list_ttsim,
    inverse_sigmoid as inverse_sigmoid_ttsim,
)

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
    "zeros": "All zeros - tests zero-value propagation",
    "mixed": "Mix of positive/negative values - tests real-world distribution",
    "small": "Very small values (~1e-6) - tests numerical precision near zero",
    "large": "Very large values (~1e6) - tests numerical overflow handling",
    "minimum_input": "Smallest valid input size - degenerate/boundary case",
    "near_zero": "Values near 0 (0.01-0.1) - tests logit near -inf",
    "near_one": "Values near 1 (0.9-0.99) - tests logit near +inf",
    "mid_range": "Values around 0.5 - tests logit near 0",
    "boundary": "Values at exact 0 and 1 - tests clamping",
    "uniform": "Uniform [0,1] values - tests full sigmoid range",
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
def _seed(offset=0):
    torch.manual_seed(SEED + offset)
    np.random.seed(SEED + offset)


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
    elif data_type == "near_zero":
        return np.random.rand(*shape).astype(np.float32) * 0.09 + 0.01
    elif data_type == "near_one":
        return np.random.rand(*shape).astype(np.float32) * 0.09 + 0.9
    elif data_type == "mid_range":
        return np.random.rand(*shape).astype(np.float32) * 0.2 + 0.4
    elif data_type == "boundary":
        d = np.zeros(shape, dtype=np.float32)
        d.flat[::2] = 0.0
        d.flat[1::2] = 1.0
        return d
    elif data_type == "uniform":
        return np.random.rand(*shape).astype(np.float32)
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
    """Print test result in clean tree-style linear format.

    Args:
        input_samples: dict of {name: ndarray} with raw input float
                       samples to display (e.g. {'input': test_data}).
    """
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
        pt_arr = np.asarray(pt_data).flat[:5]
        tt_arr = np.asarray(tt_data).flat[:5]
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
    print(f"{'MODULE':<35}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

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
        print(f"{name:<35}{shape_str:<12}{num_str:<12}{status}")

    print(DIVIDER_LINE)

    total_num_str = f"{total_np}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("✓ PASS") if all_passed else Colors.fail("✗ FAIL")
    print(f"{'TOTAL':<35}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

    if FAILED_TESTS:
        print(f"\n{Colors.fail('FAILED TESTS:')}")
        for ft in FAILED_TESTS:
            diff_str = (
                f"max_diff={ft['max_diff']:.2e}"
                if ft.get("max_diff") is not None
                else ""
            )
            atol_val = ft.get("atol", ATOL)
            gt_str = f" > atol={atol_val}" if ft.get("max_diff") is not None else ""
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
#  TEST 1 — NestedTensor (data container)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_nested = "test_nested_tensor"
test_cases_nested = [
    # (description, batch, channels, height, width, data_type, category)
    # --- Baseline tests ---
    ("Basic 3ch 56x56", 2, 3, 56, 56, "positive", "baseline"),
    ("Single batch 3ch 32x32", 1, 3, 32, 32, "positive", "baseline"),
    ("Multi-channel 64ch", 2, 64, 16, 16, "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("Negative values", 2, 3, 16, 16, "negative", "edge_value"),
    ("Zero values", 2, 3, 16, 16, "zeros", "edge_value"),
    ("Mixed positive/negative", 2, 3, 16, 16, "mixed", "edge_value"),
    ("Very small values (1e-6)", 2, 3, 16, 16, "small", "edge_value"),
    ("Very large values (1e6)", 2, 3, 16, 16, "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum spatial 1x1", 1, 1, 1, 1, "positive", "minimum_input"),
    ("Single pixel batch=2", 2, 3, 1, 1, "mixed", "minimum_input"),
]


@pytest.mark.unit
def test_nested_tensor():
    """Test NestedTensor: data container wrapping tensors + mask, decompose method."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, data_type, category) in enumerate(
        test_cases_nested
    ):
        _seed(tno)
        shape = [batch, ch, h, w]
        test_data = generate_test_data(shape, data_type)

        # Create tensors
        x_torch = torch.from_numpy(test_data)
        mask_torch = torch.zeros(batch, h, w, dtype=torch.bool)
        # Set some mask pixels for non-trivial testing
        if h > 1 and w > 1:
            mask_torch[:, : min(3, h), :] = True

        x_ttsim = torch_to_simtensor(x_torch, "tensor")
        mask_np = mask_torch.cpu().numpy()

        # Create NestedTensors
        nt_pt = NestedTensorPyTorch(x_torch, mask_torch)
        nt_tt = NestedTensorTTSim(x_ttsim, mask_np)

        # Decompose
        t_pt, m_pt = nt_pt.decompose()
        t_tt, m_tt = nt_tt.decompose()

        # Shape validation
        pt_t_shape = list(t_pt.shape)
        tt_t_shape = (
            list(t_tt.shape) if isinstance(t_tt, SimTensor) else list(t_tt.shape)
        )
        pt_m_shape = list(m_pt.shape)
        tt_m_shape = list(m_tt.shape)
        shape_ok = (pt_t_shape == tt_t_shape) and (pt_m_shape == tt_m_shape)

        # Numerical validation — tensors
        pt_d = _to_numpy(t_pt)
        tt_d = _to_numpy(t_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))

        # Mask validation
        mask_ok = np.array_equal(m_pt.cpu().numpy(), m_tt)
        num_ok = num_ok and mask_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: tensor PT={pt_t_shape} vs TT={tt_t_shape}, mask PT={pt_m_shape} vs TT={tt_m_shape}"
        elif not mask_ok:
            reason = "Mask values differ"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        print_test_linear(
            module="NestedTensor",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=shape,
            shape_line=f"tensor: PT={_compact_shape(pt_t_shape)} | TT={_compact_shape(tt_t_shape)}, mask: PT={_compact_shape(pt_m_shape)} | TT={_compact_shape(tt_m_shape)}",
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

        TEST_RESULTS.append(
            {
                "module": "NestedTensor",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": list(shape),
                "pt_shape": pt_t_shape,
                "tt_shape": tt_t_shape,
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
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "NestedTensor",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}` | `{pt_t_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{shape}` → **Tensor Shape:** `{pt_t_shape}` | **Mask Shape:** `{pt_m_shape}`\n\n"
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

    MODULE_STATS["NestedTensor"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_nested),
        "num_passed": num_passed,
        "num_total": len(test_cases_nested),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "NestedTensor",
            "description": "Data container bundling tensors with padding masks — decompose + mask validation",
            "passed": passed,
            "total": len(test_cases_nested),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_nested
    ), f"NestedTensor: {passed}/{len(test_cases_nested)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — interpolate (nearest neighbor, upsampling)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_interp_nearest = "test_interpolate_nearest"
test_cases_interp_nearest = [
    # (description, batch, channels, h_in, w_in, h_out, w_out, data_type, category)
    # --- Baseline tests ---
    ("2x upsample 10→20", 2, 3, 10, 10, 20, 20, "positive", "baseline"),
    ("3x upsample 8→24", 1, 3, 8, 8, 24, 24, "positive", "baseline"),
    ("Non-square 8x10→16x20", 1, 3, 8, 10, 16, 20, "positive", "baseline"),
    ("64 channels", 1, 64, 8, 8, 16, 16, "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("Negative values", 2, 3, 10, 10, 20, 20, "negative", "edge_value"),
    ("Zero values", 2, 3, 10, 10, 20, 20, "zeros", "edge_value"),
    ("Mixed positive/negative", 2, 3, 10, 10, 20, 20, "mixed", "edge_value"),
    ("Very small values (1e-6)", 2, 3, 10, 10, 20, 20, "small", "edge_value"),
    ("Very large values (1e6)", 2, 3, 10, 10, 20, 20, "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum 1x1→2x2", 1, 1, 1, 1, 2, 2, "positive", "minimum_input"),
    ("Minimum 1x1→1x1 (identity)", 1, 1, 1, 1, 1, 1, "mixed", "minimum_input"),
]


@pytest.mark.unit
def test_interpolate_nearest():
    """Test interpolate (nearest): numerical validation across edge cases."""
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
        h_in,
        w_in,
        h_out,
        w_out,
        data_type,
        category,
    ) in enumerate(test_cases_interp_nearest):
        _seed(tno)
        shape = [batch, ch, h_in, w_in]
        test_data = generate_test_data(shape, data_type)

        x_torch = torch.from_numpy(test_data)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        target_size = (h_out, w_out)

        # Forward
        with torch.no_grad():
            out_pt = interpolate_pytorch(x_torch, size=target_size, mode="nearest")
        out_tt = interpolate_ttsim(x_ttsim, size=target_size, mode="nearest")

        # Validation
        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape if isinstance(out_tt, SimTensor) else list(out_tt.shape)
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

        print_test_linear(
            module="interpolate (nearest)",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=f"{shape} → {target_size}",
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

        TEST_RESULTS.append(
            {
                "module": "interpolate (nearest)",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
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
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "interpolate (nearest)",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}→{list(target_size)}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}` (target={target_size})\n\n"
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

    MODULE_STATS["interpolate (nearest)"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_interp_nearest),
        "num_passed": num_passed,
        "num_total": len(test_cases_interp_nearest),
    }

    hdr = (
        "| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "interpolate (nearest)",
            "description": "Nearest neighbor upsampling — deterministic pixel selection",
            "passed": passed,
            "total": len(test_cases_interp_nearest),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_interp_nearest
    ), f"interpolate (nearest): {passed}/{len(test_cases_interp_nearest)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — interpolate (bilinear)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_interp_bilinear = "test_interpolate_bilinear"
test_cases_interp_bilinear = [
    # (description, batch, channels, h_in, w_in, h_out, w_out, data_type, category)
    # --- Baseline tests ---
    ("2x upsample 8→16", 1, 3, 8, 8, 16, 16, "positive", "baseline"),
    ("3x upsample 4→12", 1, 3, 4, 4, 12, 12, "positive", "baseline"),
    ("Non-square 6x8→12x16", 1, 3, 6, 8, 12, 16, "positive", "baseline"),
    ("Multi-batch", 2, 3, 8, 8, 16, 16, "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("Negative values", 1, 3, 8, 8, 16, 16, "negative", "edge_value"),
    ("Zero values", 1, 3, 8, 8, 16, 16, "zeros", "edge_value"),
    ("Mixed positive/negative", 1, 3, 8, 8, 16, 16, "mixed", "edge_value"),
    ("Very small values (1e-6)", 1, 3, 8, 8, 16, 16, "small", "edge_value"),
    ("Very large values (1e6)", 1, 3, 8, 8, 16, 16, "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum 2x2→4x4", 1, 1, 2, 2, 4, 4, "positive", "minimum_input"),
]


@pytest.mark.unit
def test_interpolate_bilinear():
    """Test interpolate (bilinear): numerical validation across edge cases."""
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
        h_in,
        w_in,
        h_out,
        w_out,
        data_type,
        category,
    ) in enumerate(test_cases_interp_bilinear):
        _seed(tno)
        shape = [batch, ch, h_in, w_in]
        test_data = generate_test_data(shape, data_type)

        # Tolerance — bilinear may have small float differences
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        x_torch = torch.from_numpy(test_data)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        target_size = (h_out, w_out)

        with torch.no_grad():
            out_pt = interpolate_pytorch(
                x_torch, size=target_size, mode="bilinear", align_corners=False
            )
        out_tt = interpolate_ttsim(
            x_ttsim, size=target_size, mode="bilinear", align_corners=False
        )

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape if isinstance(out_tt, SimTensor) else list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=rtol, atol=atol))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="interpolate (bilinear)",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=f"{shape} → {target_size}",
            shape_line=f"PyTorch={_compact_shape(pt_shape)} | TTSim={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=rtol,
            atol=atol,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
            input_samples={"input": test_data},
        )

        TEST_RESULTS.append(
            {
                "module": "interpolate (bilinear)",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
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
                "note": (
                    f"relaxed tolerance for large values"
                    if data_type == "large"
                    else ""
                ),
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "interpolate (bilinear)",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        tol_info = (
            f" *(relaxed: rtol={rtol}, atol={atol})*" if data_type == "large" else ""
        )
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}→{list(target_size)}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}{tol_info}\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}` (target={target_size})\n\n"
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

    MODULE_STATS["interpolate (bilinear)"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_interp_bilinear),
        "num_passed": num_passed,
        "num_total": len(test_cases_interp_bilinear),
    }

    hdr = (
        "| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "interpolate (bilinear)",
            "description": "Bilinear upsampling — PyTorch-compatible weighted average interpolation",
            "passed": passed,
            "total": len(test_cases_interp_bilinear),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_interp_bilinear
    ), f"interpolate (bilinear): {passed}/{len(test_cases_interp_bilinear)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — interpolate (downsampling, nearest)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_interp_down = "test_interpolate_downsample"
test_cases_interp_down = [
    # (description, batch, channels, h_in, w_in, h_out, w_out, data_type, category)
    # --- Baseline tests ---
    ("4x downsample 32→8", 2, 3, 32, 32, 8, 8, "positive", "baseline"),
    ("2x downsample 16→8", 1, 3, 16, 16, 8, 8, "positive", "baseline"),
    ("Non-square 24x32→6x8", 1, 3, 24, 32, 6, 8, "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("Negative values", 2, 3, 32, 32, 8, 8, "negative", "edge_value"),
    ("Zero values", 2, 3, 32, 32, 8, 8, "zeros", "edge_value"),
    ("Mixed positive/negative", 2, 3, 32, 32, 8, 8, "mixed", "edge_value"),
    ("Very small values (1e-6)", 2, 3, 32, 32, 8, 8, "small", "edge_value"),
    ("Very large values (1e6)", 2, 3, 32, 32, 8, 8, "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum 2x2→1x1", 1, 1, 2, 2, 1, 1, "positive", "minimum_input"),
    ("Identity 4x4→4x4", 1, 3, 4, 4, 4, 4, "mixed", "minimum_input"),
]


@pytest.mark.unit
def test_interpolate_downsample():
    """Test interpolate (downsample nearest): numerical validation across edge cases."""
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
        h_in,
        w_in,
        h_out,
        w_out,
        data_type,
        category,
    ) in enumerate(test_cases_interp_down):
        _seed(tno)
        shape = [batch, ch, h_in, w_in]
        test_data = generate_test_data(shape, data_type)

        x_torch = torch.from_numpy(test_data)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        target_size = (h_out, w_out)

        with torch.no_grad():
            out_pt = interpolate_pytorch(x_torch, size=target_size, mode="nearest")
        out_tt = interpolate_ttsim(x_ttsim, size=target_size, mode="nearest")

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape if isinstance(out_tt, SimTensor) else list(out_tt.shape)
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

        print_test_linear(
            module="interpolate (downsample)",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=f"{shape} → {target_size}",
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

        TEST_RESULTS.append(
            {
                "module": "interpolate (downsample)",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
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
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "interpolate (downsample)",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}→{list(target_size)}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{shape}` → **Output Shape:** `{pt_shape}` (target={target_size})\n\n"
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

    MODULE_STATS["interpolate (downsample)"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_interp_down),
        "num_passed": num_passed,
        "num_total": len(test_cases_interp_down),
    }

    hdr = (
        "| # | Test Case | Data Type | Input→Target | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "interpolate (downsample)",
            "description": "Nearest neighbor downsampling — reducing spatial resolution",
            "passed": passed,
            "total": len(test_cases_interp_down),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_interp_down
    ), f"interpolate (downsample): {passed}/{len(test_cases_interp_down)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — nested_tensor_from_tensor_list
# ═══════════════════════════════════════════════════════════════════════════════

test_name_ntftl = "test_nested_tensor_from_tensor_list"
test_cases_ntftl = [
    # (description, tensor_shapes, data_type, category)
    # tensor_shapes is list of (C, H, W) for each tensor in the batch
    # --- Baseline tests ---
    (
        "3 tensors variable sizes",
        [(3, 10, 15), (3, 12, 18), (3, 8, 20)],
        "positive",
        "baseline",
    ),
    ("2 tensors same size", [(3, 16, 16), (3, 16, 16)], "positive", "baseline"),
    (
        "4 tensors different H/W",
        [(3, 8, 12), (3, 10, 10), (3, 6, 14), (3, 12, 8)],
        "positive",
        "baseline",
    ),
    ("Single channel", [(1, 10, 10), (1, 8, 12)], "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("Negative values", [(3, 10, 15), (3, 12, 18)], "negative", "edge_value"),
    ("Zero values", [(3, 10, 15), (3, 12, 18)], "zeros", "edge_value"),
    ("Mixed positive/negative", [(3, 10, 15), (3, 12, 18)], "mixed", "edge_value"),
    ("Very small values (1e-6)", [(3, 10, 15), (3, 12, 18)], "small", "edge_value"),
    ("Very large values (1e6)", [(3, 10, 15), (3, 12, 18)], "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum 1x1 tensors", [(1, 1, 1), (1, 1, 1)], "positive", "minimum_input"),
    ("Minimum mixed 1x1 & 2x3", [(1, 1, 1), (1, 2, 3)], "mixed", "minimum_input"),
]


@pytest.mark.unit
def test_nested_tensor_from_tensor_list():
    """Test nested_tensor_from_tensor_list: padding + mask with edge cases."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, tensor_shapes, data_type, category) in enumerate(test_cases_ntftl):
        _seed(tno)

        # Generate tensors
        pt_tensors = []
        tt_tensors = []
        input_samples_dict = {}
        for i, (c, h, w) in enumerate(tensor_shapes):
            data = generate_test_data((c, h, w), data_type)
            pt_tensors.append(torch.from_numpy(data))
            tt_tensors.append(torch_to_simtensor(torch.from_numpy(data), f"tensor{i}"))
            if i < 3:  # Show first 3 tensor samples
                input_samples_dict[f"tensor{i}"] = data

        # Forward
        with torch.no_grad():
            nested_pt = nested_tensor_from_tensor_list_pytorch(pt_tensors)
        nested_tt = nested_tensor_from_tensor_list_ttsim(tt_tensors)

        # Decompose
        t_pt, m_pt = nested_pt.decompose()
        t_tt, m_tt = nested_tt.decompose()

        # Shape validation
        pt_shape = list(t_pt.shape)
        tt_shape = t_tt.shape if isinstance(t_tt, SimTensor) else list(t_tt.shape)
        pt_mask_shape = list(m_pt.shape)
        tt_mask_shape = list(m_tt.shape)
        shape_ok = (pt_shape == tt_shape) and (pt_mask_shape == tt_mask_shape)

        # Numerical validation — padded tensors
        pt_d = _to_numpy(t_pt)
        tt_d = _to_numpy(t_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))

        # Mask validation
        mask_ok = np.array_equal(m_pt.cpu().numpy(), m_tt)
        num_ok = num_ok and mask_ok

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: tensor PT={pt_shape} vs TT={tt_shape}, mask PT={pt_mask_shape} vs TT={tt_mask_shape}"
        elif not mask_ok:
            diff_count = int(np.sum(m_pt.cpu().numpy() != m_tt))
            reason = f"Mask mismatch: {diff_count} pixels differ"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        shapes_str = " + ".join(f"{list(s)}" for s in tensor_shapes)
        print_test_linear(
            module="nested_tensor_from_tensor_list",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=shapes_str,
            shape_line=f"tensor: PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}, mask: PT={_compact_shape(pt_mask_shape)} | TT={_compact_shape(tt_mask_shape)}",
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
            input_samples=input_samples_dict,
        )

        TEST_RESULTS.append(
            {
                "module": "nested_tensor_from_tensor_list",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": [list(s) for s in tensor_shapes],
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
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "nested_tensor_from_tensor_list",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{[list(s) for s in tensor_shapes]}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        # Build input samples string for detail
        input_strs = []
        for i, (c, h, w) in enumerate(tensor_shapes):
            if i < 3:
                input_strs.append(
                    f"- tensor{i} `[{c},{h},{w}]`: `[{_fmt_samples(input_samples_dict.get(f'tensor{i}', np.zeros(1)))}]`"
                )
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Tensors:** {len(tensor_shapes)} tensors → **Batched Shape:** `{pt_shape}` | **Mask:** `{pt_mask_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n" + "\n".join(input_strs) + "\n\n"
            f"| Metric | PyTorch | TTSim | Difference |\n"
            f"|:-------|--------:|------:|----------:|\n"
            f"| Mean | {pt_d.mean():.8f} | {tt_d.mean():.8f} | {abs(pt_d.mean()-tt_d.mean()):.2e} |\n"
            f"| Std  | {pt_d.std():.8f} | {tt_d.std():.8f} | {abs(pt_d.std()-tt_d.std()):.2e} |\n"
            f"| Min  | {pt_d.min():.8f} | {tt_d.min():.8f} | {abs(pt_d.min()-tt_d.min()):.2e} |\n"
            f"| Max  | {pt_d.max():.8f} | {tt_d.max():.8f} | {abs(pt_d.max()-tt_d.max()):.2e} |\n\n"
            f"**Mask Comparison:**\n"
            f"- Masks identical: `{mask_ok}`\n"
            f"- PyTorch masked pixels: `{int(m_pt.sum().item())}/{m_pt.numel()}`\n"
            f"- TTSim masked pixels:   `{int(m_tt.sum())}/{m_tt.size}`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["nested_tensor_from_list"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_ntftl),
        "num_passed": num_passed,
        "num_total": len(test_cases_ntftl),
    }

    hdr = (
        "| # | Test Case | Data Type | Input Shapes | Batched Shape | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------------|:--------------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "nested_tensor_from_tensor_list",
            "description": "Batch variable-sized tensors with zero-padding and boolean mask",
            "passed": passed,
            "total": len(test_cases_ntftl),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_ntftl
    ), f"nested_tensor_from_tensor_list: {passed}/{len(test_cases_ntftl)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — inverse_sigmoid
# ═══════════════════════════════════════════════════════════════════════════════

test_name_inv_sig = "test_inverse_sigmoid"
test_cases_inv_sig = [
    # (description, shape, data_type, category)
    # --- Baseline tests ---
    ("Mid-range [0.4-0.6]", (2, 4, 8), "mid_range", "baseline"),
    ("Uniform [0,1]", (2, 4, 8), "uniform", "baseline"),
    ("Multi-batch", (4, 8, 16), "uniform", "baseline"),
    # --- Edge case: Values (mandatory — adapted for sigmoid domain) ---
    ("Near-zero values [0.01-0.1]", (2, 4, 8), "near_zero", "edge_value"),
    ("Near-one values [0.9-0.99]", (2, 4, 8), "near_one", "edge_value"),
    ("Boundary 0 and 1", (2, 4, 8), "boundary", "edge_value"),
    ("Zero values", (2, 4, 8), "zeros", "edge_value"),
    ("Negative values (clamped)", (2, 4, 8), "negative", "edge_value"),
    ("Large values >1 (clamped)", (2, 4, 8), "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum 1-element", (1,), "mid_range", "minimum_input"),
    ("Minimum 1x1x1", (1, 1, 1), "uniform", "minimum_input"),
]


@pytest.mark.unit
def test_inverse_sigmoid():
    """Test inverse_sigmoid: logit function numerical validation across edge cases."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, shape, data_type, category) in enumerate(test_cases_inv_sig):
        _seed(tno)
        test_data = generate_test_data(shape, data_type)

        x_torch = torch.from_numpy(test_data)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        # Forward
        with torch.no_grad():
            out_pt = inverse_sigmoid_pytorch(x_torch)
        out_tt = inverse_sigmoid_ttsim(x_ttsim)

        # Shape validation
        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape if isinstance(out_tt, SimTensor) else list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

        # Numerical validation
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

        print_test_linear(
            module="inverse_sigmoid",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, category),
            input_shape=list(shape),
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

        TEST_RESULTS.append(
            {
                "module": "inverse_sigmoid",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
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
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "inverse_sigmoid",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{list(shape)}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{list(shape)}` → **Output Shape:** `{pt_shape}`\n\n"
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

    MODULE_STATS["inverse_sigmoid"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_inv_sig),
        "num_passed": num_passed,
        "num_total": len(test_cases_inv_sig),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "inverse_sigmoid",
            "description": "Logit function — inverse of sigmoid, with clamping for numerical stability",
            "passed": passed,
            "total": len(test_cases_inv_sig),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_inv_sig
    ), f"inverse_sigmoid: {passed}/{len(test_cases_inv_sig)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Shape Inference Only (data=None)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_shape = "test_shape_inference"
test_cases_shape = [
    # (description, function, kwargs, expected_shape, category)
    (
        "interpolate nearest 2x up",
        "interpolate",
        {"shape": [2, 3, 10, 10], "size": (20, 20), "mode": "nearest"},
        [2, 3, 20, 20],
        "baseline",
    ),
    (
        "interpolate bilinear 2x up",
        "interpolate",
        {"shape": [1, 3, 8, 8], "size": (16, 16), "mode": "bilinear"},
        [1, 3, 16, 16],
        "baseline",
    ),
    (
        "interpolate 4x downsample",
        "interpolate",
        {"shape": [2, 3, 32, 32], "size": (8, 8), "mode": "nearest"},
        [2, 3, 8, 8],
        "baseline",
    ),
    (
        "interpolate scale_factor=2",
        "interpolate",
        {"shape": [1, 3, 10, 10], "scale_factor": 2.0, "mode": "nearest"},
        [1, 3, 20, 20],
        "baseline",
    ),
    (
        "interpolate non-square",
        "interpolate",
        {"shape": [1, 64, 6, 8], "size": (12, 16), "mode": "nearest"},
        [1, 64, 12, 16],
        "baseline",
    ),
    (
        "nested_tensor 3 tensors",
        "nested_tensor_from_tensor_list",
        {"tensor_shapes": [(3, 10, 15), (3, 12, 18), (3, 8, 20)]},
        [3, 3, 12, 20],
        "baseline",
    ),
    (
        "interpolate minimum 1x1→2x2",
        "interpolate",
        {"shape": [1, 1, 1, 1], "size": (2, 2), "mode": "nearest"},
        [1, 1, 2, 2],
        "minimum_input",
    ),
    (
        "nested_tensor minimum 1x1",
        "nested_tensor_from_tensor_list",
        {"tensor_shapes": [(1, 1, 1), (1, 2, 3)]},
        [2, 1, 2, 3],
        "minimum_input",
    ),
]


@pytest.mark.unit
def test_shape_inference():
    """Test shape inference mode (data=None) for interpolate and nested_tensor_from_tensor_list."""
    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (tmsg, func_name, kwargs, expected_shape, category) in enumerate(
        test_cases_shape
    ):
        _seed(tno)

        try:
            if func_name == "interpolate":
                shape = kwargs["shape"]
                x_ttsim = SimTensor(
                    {
                        "name": "input",
                        "shape": shape,
                        "data": None,
                        "dtype": np.dtype(np.float32),
                    }
                )
                interp_kwargs = {k: v for k, v in kwargs.items() if k != "shape"}
                out_tt = interpolate_ttsim(x_ttsim, **interp_kwargs)
                tt_shape = (
                    out_tt.shape
                    if isinstance(out_tt, SimTensor)
                    else list(out_tt.shape)
                )
                data_is_none = (
                    (out_tt.data is None) if isinstance(out_tt, SimTensor) else False
                )

            elif func_name == "nested_tensor_from_tensor_list":
                tensor_shapes = kwargs["tensor_shapes"]
                tt_tensors = []
                for i, (c, h, w) in enumerate(tensor_shapes):
                    tt_tensors.append(
                        SimTensor(
                            {
                                "name": f"tensor{i}",
                                "shape": [c, h, w],
                                "data": None,
                                "dtype": np.dtype(np.float32),
                            }
                        )
                    )
                nested_tt = nested_tensor_from_tensor_list_ttsim(tt_tensors)
                t_tt, m_tt = nested_tt.decompose()
                tt_shape = (
                    t_tt.shape if isinstance(t_tt, SimTensor) else list(t_tt.shape)
                )
                data_is_none = (
                    (t_tt.data is None) if isinstance(t_tt, SimTensor) else False
                )

            shape_ok = tt_shape == expected_shape
            ok = shape_ok
            passed += int(ok)

            reason = ""
            if not shape_ok:
                reason = f"Shape mismatch: expected={expected_shape} got={tt_shape}"

            print_test_linear(
                module="Shape Inference",
                edge_case=category,
                edge_desc=f"{func_name}: {tmsg}",
                input_shape=str(kwargs),
                shape_line=f"expected={_compact_shape(expected_shape)} | got={_compact_shape(tt_shape)}, data=None: {data_is_none}",
                shape_ok=shape_ok,
                is_numerical=False,
                failure_reason=reason,
            )

            TEST_RESULTS.append(
                {
                    "module": "Shape Inference",
                    "validation_type": "SHAPE ONLY",
                    "edge_case": category,
                    "edge_desc": f"{func_name}: {tmsg}",
                    "input_shape": str(kwargs),
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
                failed_cases.append((tno, tmsg, "shape_mismatch", 0))
                FAILED_TESTS.append(
                    {
                        "module": "Shape Inference",
                        "test": tmsg,
                        "edge_case": "shape_mismatch",
                        "max_diff": None,
                    }
                )

            tag = "✅ PASS" if ok else "🔴 **FAIL**"
            rows.append(
                f"| {tno} | {tmsg} | `{_compact_shape(expected_shape)}` | `{_compact_shape(tt_shape)}` | {data_is_none} | {tag} |"
            )

            status_badge = "🟢" if ok else "🔴"
            detail_blocks.append(
                f"---\n\n"
                f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
                f"**Function:** `{func_name}`\n\n"
                f"**Config:** `{kwargs}`\n\n"
                f"**Expected Shape:** `{expected_shape}` | **Actual Shape:** `{tt_shape}`\n\n"
                f"**data=None:** `{data_is_none}`\n\n"
            )

        except Exception as e:
            import traceback

            passed_this = False
            reason = f"Exception: {str(e)}"
            print_test_linear(
                module="Shape Inference",
                edge_case=category,
                edge_desc=f"{func_name}: {tmsg}",
                input_shape=str(kwargs),
                shape_line=f"expected={_compact_shape(expected_shape)} | ERROR",
                shape_ok=False,
                is_numerical=False,
                failure_reason=reason,
            )
            failed_cases.append((tno, tmsg, "error", 0))
            FAILED_TESTS.append(
                {
                    "module": "Shape Inference",
                    "test": tmsg,
                    "edge_case": "error",
                    "max_diff": None,
                }
            )
            tag = "🔴 **FAIL**"
            rows.append(
                f"| {tno} | {tmsg} | `{_compact_shape(expected_shape)}` | ERROR | — | {tag} |"
            )
            detail_blocks.append(
                f"---\n\n"
                f"### 🔴 TEST[{tno}] {tmsg}\n\n"
                f"**Exception:**\n```\n{traceback.format_exc()}\n```\n\n"
            )

    MODULE_STATS["Shape Inference"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_shape),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Expected Shape | Actual Shape | data=None | Result |\n"
        "|:--|:----------|:---------------|:-------------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Shape Inference",
            "description": "Shape-only validation (data=None) for interpolate and nested_tensor_from_tensor_list",
            "passed": passed,
            "total": len(test_cases_shape),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_shape
    ), f"Shape Inference: {passed}/{len(test_cases_shape)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-runner with markdown report
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a simple, module-wise markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# Misc Utilities Unit Test Report",
        f"**PyTorch vs TTSim Comparison** | **{total_passed}/{total_tests} passed** | {status}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exit Code: {exit_code}",
        "",
        "---",
        "",
    ]

    # Summary first - module wise pass/fail
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

    # Failed tests (if any)
    if FAILED_TESTS:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Tests")
        lines.append("")
        lines.append("| Module | Test | Edge Case | Max Diff |")
        lines.append("|--------|------|-----------|----------|")
        for ft in FAILED_TESTS:
            diff_str = (
                f"{ft['max_diff']:.2e}" if ft.get("max_diff") is not None else "N/A"
            )
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

        # Show table
        lines.append(s["table"])
        lines.append("")

        # Show failed cases for this module
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

    pytest may register the module under various names (basename, dotted path,
    or even a conftest-mangled path), so we search all of sys.modules for any
    module whose __file__ matches ours.
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
    report_path = os.path.join(report_dir, "misc_unit_test_report.md")
    output_path = os.path.join(report_dir, "misc_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    # Print overall header
    print(f"\n{SUMMARY_LINE}")
    print(f"MISC UTILITIES UNIT TEST SUITE - PyTorch vs TTSim")
    print(f"{SUMMARY_LINE}\n")

    # Run pytest — tests populate REPORT_SECTIONS / MODULE_STATS / etc.
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    # Sync globals back from the pytest-imported copy of this module
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
