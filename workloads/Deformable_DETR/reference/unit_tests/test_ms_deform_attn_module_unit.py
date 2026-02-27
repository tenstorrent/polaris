#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MSDeformAttn MODULE — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPES EXPLAINED:
================================================================================

NUMERICAL VALIDATION:
    - Compares actual OUTPUT VALUES between PyTorch and TTSim
    - Uses np.allclose() with tolerance: rtol=1e-4, atol=1e-5
    - Reports: max absolute difference, mean absolute difference
    - PASS if: all values within tolerance
    - FAIL if: any value exceeds tolerance
    - WHY USE: When computations are fast enough for full numerical comparison

SHAPE VALIDATION ONLY:
    - Compares only OUTPUT DIMENSIONS (shapes) between PyTorch and TTSim
    - NO numerical value comparison
    - PASS if: all shapes match exactly
    - FAIL if: any shape mismatch detected
    - WHY USE: Validates shape-inference path (data=None)

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: Linear (Value Projection) — NUMERICAL VALIDATION
    Operation: value_proj(input_flatten) → [N, Len_in, d_model]
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6)
    WHY NUMERICAL: Single linear layer, fast to compute

MODULE 2: Linear (Sampling Offsets) — NUMERICAL VALIDATION
    Operation: sampling_offsets(query) → [N, Len_q, n_heads*n_levels*n_points*2]
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6)
    WHY NUMERICAL: Single linear layer, fast to compute

MODULE 3: Linear (Attention Weights + Softmax) — NUMERICAL VALIDATION
    Operation: attention_weights(query) → softmax → [N, Len_q, n_heads, n_levels, n_points]
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6)
    WHY NUMERICAL: Linear + softmax, verifies probability normalization

MODULE 4: Sampling Locations — NUMERICAL VALIDATION
    Operation: compute sampling locations from reference_points + offsets
    Edge Cases: positive, negative, zeros, mixed, small, large,
                boundary_coords (near 0/1), center_coords, corner_coords, 4d_ref_points
    WHY NUMERICAL: Pure arithmetic on coordinates

MODULE 5: MSDeformAttn End-to-End (2D ref points) — NUMERICAL VALIDATION
    Operation: Full forward pass with 2D reference points
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input
    WHY NUMERICAL: Validates the entire pipeline

MODULE 6: MSDeformAttn End-to-End (4D ref points) — NUMERICAL VALIDATION
    Operation: Full forward pass with 4D reference points
    Edge Cases: positive, negative, zeros, mixed, small, large
    WHY NUMERICAL: Validates 4D branch of sampling location computation

MODULE 7: MSDeformAttn Shape Inference — SHAPE VALIDATION ONLY
    Operation: Forward pass with data=None
    Edge Cases: standard, small_config, single_level, many_heads,
                single_query, large_batch, minimum_input
    WHY SHAPE: Validates shape-inference path

================================================================================
EDGE CASES TESTED (MANDATORY — all numerical modules):
================================================================================

'positive'       — Standard positive values (1.0 - 2.0) - baseline test
'negative'       — All negative values (-2.0 to -1.0) - tests sign handling
'zeros'          — All zeros - tests zero input behavior
'mixed'          — Mix of positive/negative values - tests real-world distribution
'small'          — Very small values (~1e-6) - tests numerical precision near zero
'large'          — Very large values (~1e6) - tests numerical overflow handling
'minimum_input'  — Smallest valid input size - tests degenerate/boundary case

--- Additional Geometry Edge Cases ---
'boundary_coords'       — Sampling near 0.0 and 1.0 — grid edge behavior
'center_coords'         — All locations at 0.5 — center-only sampling
'corner_coords'         — All at 0.0 or 1.0 — extreme corner positions
'4d_ref_points'         — 4D reference points (x, y, w, h)

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_ms_deform_attn_module_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_ms_deform_attn_module_unit.py
================================================================================
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as TF
import numpy as np
from datetime import datetime
from types import SimpleNamespace

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
from workloads.Deformable_DETR.reference.ms_deform_attn import (
    MSDeformAttn as MSDeformAttnPyTorch,
)
from workloads.Deformable_DETR.reference.ms_deform_attn_func import (
    ms_deform_attn_core_pytorch,
)

# TTSim implementations
from workloads.Deformable_DETR.models.ops.modules.ms_deform_attn_ttsim import (
    MSDeformAttn as MSDeformAttnTTSim,
    Linear as LinearTTSim,
)
from workloads.Deformable_DETR.models.ops.functions.ms_deform_attn_func_ttsim import (
    ms_deform_attn_core_ttsim,
)

# Utilities
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.helpers import unary_fwd

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
SUMMARY_LINE = "=" * 65
DIVIDER_LINE = "-" * 65


# ---------------------------------------------------------------------------
# Edge case descriptions
# ---------------------------------------------------------------------------
EDGE_CASE_DESC = {
    "positive": "Standard positive values (1.0 - 2.0) - baseline test",
    "negative": "All negative values (-2.0 to -1.0) - tests sign handling",
    "zeros": "All zeros - tests zero input behavior",
    "mixed": "Mix of positive/negative values - tests real-world distribution",
    "small": "Very small values (~1e-6) - tests numerical precision near zero",
    "large": "Very large values (~1e6) - tests numerical overflow handling",
    "minimum_input": "Smallest valid input size - degenerate/boundary case",
    "boundary_coords": "Sampling near 0.0 and 1.0 - grid edge behavior",
    "center_coords": "All locations at 0.5 - center-only sampling",
    "corner_coords": "All at 0.0 or 1.0 - extreme corner positions",
    "4d_ref_points": "4D reference points (x, y, w, h) - box-based references",
    "standard": "Standard multi-level configuration - baseline correctness",
    "small_config": "Small configuration for quick test",
    "single_level": "Single feature level - minimal level loop",
    "many_heads": "Many attention heads - head scalability",
    "single_query": "Single query - minimal query count",
    "large_batch": "Large batch size - batch scalability",
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
def _seed(extra=0):
    torch.manual_seed(SEED + extra)
    np.random.seed(SEED + extra)


def _to_numpy(x):
    """Coerce PyTorch tensor / SimTensor / ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, SimTensor):
        return x.data
    return np.asarray(x)


def _compact_shape(shape):
    """Format shape as [1,64,8,8] (no spaces) for inline display."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def _fmt_samples(arr, n=10):
    """First *n* values, formatted for a markdown table cell."""
    flat = np.asarray(arr).flatten()
    return ", ".join(f"{v:.6f}" for v in flat[:n])


def generate_test_data(shape, data_type):
    """Generate test data based on type for query/input_flatten (general tensors)."""
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


def generate_ref_points(shape, data_type):
    """Generate reference points in [0, 1] range (or specific patterns for coords)."""
    if data_type == "boundary_coords":
        # Near 0 and 1
        rp = np.random.rand(*shape).astype(np.float32)
        rp = np.where(rp > 0.5, 0.99, 0.01)
        return rp
    elif data_type == "center_coords":
        return np.full(shape, 0.5, dtype=np.float32)
    elif data_type == "corner_coords":
        rp = np.random.rand(*shape).astype(np.float32)
        return np.where(rp > 0.5, 1.0, 0.0).astype(np.float32)
    else:
        # Standard uniform [0, 1]
        return np.random.rand(*shape).astype(np.float32)


def _create_modules(d_model, n_levels, n_heads, n_points):
    """Create matching PyTorch and TTSim MSDeformAttn modules with synchronized weights."""
    module_pt = MSDeformAttnPyTorch(
        d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    )
    module_pt.eval()

    module_tt = MSDeformAttnTTSim(
        d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
    )

    # Copy weights from PyTorch to TTSim (SimNN.Linear uses .param.data / .bias.data)
    module_tt.sampling_offsets.param.data = (
        module_pt.sampling_offsets.weight.data.cpu().numpy()
    )
    module_tt.sampling_offsets.bias.data = (
        module_pt.sampling_offsets.bias.data.cpu().numpy()
    )
    module_tt.attention_weights.param.data = (
        module_pt.attention_weights.weight.data.cpu().numpy()
    )
    module_tt.attention_weights.bias.data = (
        module_pt.attention_weights.bias.data.cpu().numpy()
    )
    module_tt.value_proj.param.data = (
        module_pt.value_proj.weight.data.cpu().numpy()
    )
    module_tt.value_proj.bias.data = (
        module_pt.value_proj.bias.data.cpu().numpy()
    )
    module_tt.output_proj.param.data = (
        module_pt.output_proj.weight.data.cpu().numpy()
    )
    module_tt.output_proj.bias.data = (
        module_pt.output_proj.bias.data.cpu().numpy()
    )

    return module_pt, module_tt


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
    print(f"|- EDGE CASE: {Colors.warn(edge_case)} ({edge_desc})")
    print(f"|- INPUT: {input_shape}")

    # Show input float samples when provided
    if input_samples:
        for sname, sarr in input_samples.items():
            flat = np.asarray(sarr).flatten()
            sstr = ", ".join(f"{v:.6f}" for v in flat[:5])
            print(f"|- INPUT {sname}[0:5]: [{sstr}]")

    # Shape line
    shape_status = Colors.success("V MATCH") if shape_ok else Colors.fail("X MISMATCH")
    print(f"|- SHAPE: {shape_line} -> {shape_status}")

    # Numerical line
    if is_numerical and max_diff is not None:
        if num_ok:
            num_status = Colors.success(f"V PASS (tol: rtol={rtol}, atol={atol})")
        else:
            num_status = Colors.fail("X FAIL")
        print(
            f"|- NUMERICAL: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} -> {num_status}"
        )

    # Sample output values for numerical tests
    if is_numerical and pt_data is not None and tt_data is not None:
        pt_arr = np.asarray(pt_data).flat[:5]
        tt_arr = np.asarray(tt_data).flat[:5]
        pt_str = ", ".join(f"{v:.6f}" for v in pt_arr)
        tt_str = ", ".join(f"{v:.6f}" for v in tt_arr)
        print(f"|- PT OUTPUT[0:5]: [{pt_str}]")
        print(f"|- TT OUTPUT[0:5]: [{tt_str}]")

    # Failure reason
    if not passed and failure_reason:
        print(f"|- FAILURE REASON: {Colors.fail(failure_reason)}")

    # Result
    result_str = Colors.success("V PASS") if passed else Colors.fail("X FAIL")
    print(f"|_ RESULT: {result_str}")


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

        status = Colors.success("V PASS") if mod_pass else Colors.fail("X FAIL")
        print(f"{name:<35}{shape_str:<12}{num_str:<12}{status}")

    print(DIVIDER_LINE)

    total_num_str = f"{total_np}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("V PASS") if all_passed else Colors.fail("X FAIL")
    print(f"{'TOTAL':<35}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

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


# ===========================================================================
#  Default test configuration
# ===========================================================================
DEFAULT_D_MODEL = 256
DEFAULT_N_LEVELS = 4
DEFAULT_N_HEADS = 8
DEFAULT_N_POINTS = 4
DEFAULT_BATCH = 2
DEFAULT_LEN_Q = 50
DEFAULT_SPATIAL = np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32)


def _compute_len_in(spatial_shapes):
    return int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))


# ===========================================================================
#  TEST 1 — Linear (Value Projection): NUMERICAL VALIDATION
# ===========================================================================
test_cases_value_proj = [
    # (description, data_type, category)
    ("Baseline positive", "positive", "baseline"),
    ("Negative values", "negative", "edge_value"),
    ("Zero values", "zeros", "edge_value"),
    ("Mixed pos/neg", "mixed", "edge_value"),
    ("Very small (1e-6)", "small", "edge_value"),
    ("Very large (1e6)", "large", "edge_value"),
]


@pytest.mark.unit
def test_value_projection():
    """Test value_proj linear layer: shape + numerical validation across data types."""
    _seed()
    d_model = DEFAULT_D_MODEL
    N = DEFAULT_BATCH
    Len_in = _compute_len_in(DEFAULT_SPATIAL)

    module_pt, module_tt = _create_modules(
        d_model, DEFAULT_N_LEVELS, DEFAULT_N_HEADS, DEFAULT_N_POINTS
    )

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, data_type, category) in enumerate(test_cases_value_proj):
        _seed(tno)

        test_data = generate_test_data([N, Len_in, d_model], data_type)

        # Determine tolerance
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        # PyTorch
        x_torch = torch.from_numpy(test_data)
        with torch.no_grad():
            out_pt = module_pt.value_proj(x_torch)

        # TTSim
        x_sim = SimTensor(
            {
                "name": "input",
                "shape": list(test_data.shape),
                "data": test_data.copy(),
                "dtype": np.float32,
            }
        )
        out_tt = module_tt.value_proj(x_sim)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
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
            module="ValueProjection",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
            input_shape=[N, Len_in, d_model],
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
            input_samples={"input_flatten": test_data},
        )

        TEST_RESULTS.append(
            {
                "module": "ValueProjection",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": [N, Len_in, d_model],
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
                    "module": "ValueProjection",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        tol_info = (
            f" *(relaxed: rtol={rtol}, atol={atol})*" if data_type == "large" else ""
        )
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{[N, Len_in, d_model]}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )
        status_badge = "PASS" if ok else "FAIL"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` - {EDGE_CASE_DESC.get(data_type, 'N/A')}{tol_info}\n\n"
            f"**Input Shape:** `{[N, Len_in, d_model]}` -> **Output Shape:** `{pt_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- input_flatten: `[{_fmt_samples(test_data)}]`\n\n"
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

    MODULE_STATS["ValueProjection"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_value_proj),
        "num_passed": num_passed,
        "num_total": len(test_cases_value_proj),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "ValueProjection",
            "description": "value_proj linear layer (d_model -> d_model)",
            "passed": passed,
            "total": len(test_cases_value_proj),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_value_proj
    ), f"ValueProjection: {passed}/{len(test_cases_value_proj)} passed"


# ===========================================================================
#  TEST 2 — Linear (Sampling Offsets): NUMERICAL VALIDATION
# ===========================================================================
test_cases_sampling_offsets = [
    ("Baseline positive", "positive", "baseline"),
    ("Negative values", "negative", "edge_value"),
    ("Zero values", "zeros", "edge_value"),
    ("Mixed pos/neg", "mixed", "edge_value"),
    ("Very small (1e-6)", "small", "edge_value"),
    ("Very large (1e6)", "large", "edge_value"),
]


@pytest.mark.unit
def test_sampling_offsets():
    """Test sampling_offsets linear layer: shape + numerical across data types."""
    _seed()
    d_model = DEFAULT_D_MODEL
    n_heads, n_levels, n_points = DEFAULT_N_HEADS, DEFAULT_N_LEVELS, DEFAULT_N_POINTS
    N = DEFAULT_BATCH
    Len_q = DEFAULT_LEN_Q

    module_pt, module_tt = _create_modules(d_model, n_levels, n_heads, n_points)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, data_type, category) in enumerate(test_cases_sampling_offsets):
        _seed(tno)

        test_data = generate_test_data([N, Len_q, d_model], data_type)
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        # PyTorch
        x_torch = torch.from_numpy(test_data)
        with torch.no_grad():
            out_pt = module_pt.sampling_offsets(x_torch)

        # TTSim
        x_sim = SimTensor(
            {
                "name": "query",
                "shape": list(test_data.shape),
                "data": test_data.copy(),
                "dtype": np.float32,
            }
        )
        out_tt = module_tt.sampling_offsets(x_sim)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
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
            module="SamplingOffsets",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
            input_shape=[N, Len_q, d_model],
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
            input_samples={"query": test_data},
        )

        TEST_RESULTS.append(
            {
                "module": "SamplingOffsets",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": [N, Len_q, d_model],
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
                    "module": "SamplingOffsets",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{[N, Len_q, d_model]}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )
        status_badge = "PASS" if ok else "FAIL"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` - {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{[N, Len_q, d_model]}` -> **Output Shape:** `{pt_shape}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- query: `[{_fmt_samples(test_data)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["SamplingOffsets"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_sampling_offsets),
        "num_passed": num_passed,
        "num_total": len(test_cases_sampling_offsets),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "SamplingOffsets",
            "description": "sampling_offsets linear (d_model -> n_heads*n_levels*n_points*2)",
            "passed": passed,
            "total": len(test_cases_sampling_offsets),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_sampling_offsets
    ), f"SamplingOffsets: {passed}/{len(test_cases_sampling_offsets)} passed"


# ===========================================================================
#  TEST 3 — Attention Weights + Softmax: NUMERICAL VALIDATION
# ===========================================================================
test_cases_attn_weights = [
    ("Baseline positive", "positive", "baseline"),
    ("Negative values", "negative", "edge_value"),
    ("Zero values", "zeros", "edge_value"),
    ("Mixed pos/neg", "mixed", "edge_value"),
    ("Very small (1e-6)", "small", "edge_value"),
    ("Very large (1e6)", "large", "edge_value"),
]


@pytest.mark.unit
def test_attention_weights_softmax():
    """Test attention_weights linear + softmax: shape + numerical across data types."""
    _seed()
    d_model = DEFAULT_D_MODEL
    n_heads, n_levels, n_points = DEFAULT_N_HEADS, DEFAULT_N_LEVELS, DEFAULT_N_POINTS
    N = DEFAULT_BATCH
    Len_q = DEFAULT_LEN_Q

    module_pt, module_tt = _create_modules(d_model, n_levels, n_heads, n_points)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, data_type, category) in enumerate(test_cases_attn_weights):
        _seed(tno)

        test_data = generate_test_data([N, Len_q, d_model], data_type)
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        # PyTorch: attention_weights(query) -> view -> softmax -> view
        x_torch = torch.from_numpy(test_data)
        with torch.no_grad():
            aw_pt = module_pt.attention_weights(x_torch)
            aw_pt = aw_pt.view(N, Len_q, n_heads, n_levels * n_points)
            aw_pt = TF.softmax(aw_pt, -1)
            aw_pt = aw_pt.view(N, Len_q, n_heads, n_levels, n_points)

        # TTSim: attention_weights(query) -> reshape -> softmax -> reshape
        x_sim = SimTensor(
            {
                "name": "query",
                "shape": list(test_data.shape),
                "data": test_data.copy(),
                "dtype": np.float32,
            }
        )
        aw_tt = module_tt.attention_weights(x_sim)
        aw_tt.shape = [N, Len_q, n_heads, n_levels * n_points]
        aw_tt.data = aw_tt.data.reshape(aw_tt.shape)

        aw_softmax = SimTensor(
            {"name": "aw_softmax", "shape": None, "data": None, "dtype": None}
        )
        op_softmax = SimpleNamespace(
            attrs={"axis": -1}, optype="Softmax", name="softmax", precision="fp32"
        )
        unary_fwd([aw_tt], [aw_softmax], op_softmax)
        aw_tt = aw_softmax

        aw_tt.shape = [N, Len_q, n_heads, n_levels, n_points]
        aw_tt.data = aw_tt.data.reshape(aw_tt.shape)

        pt_shape = list(aw_pt.shape)
        tt_shape = aw_tt.shape
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(aw_pt)
        tt_d = _to_numpy(aw_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=rtol, atol=atol))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        # Verify softmax property: sum to 1 (last dim)
        pt_sum = pt_d.sum(axis=-1)
        tt_sum = tt_d.sum(axis=-1)
        sum_ok = bool(
            np.allclose(pt_sum, 1.0, atol=1e-5) and np.allclose(tt_sum, 1.0, atol=1e-5)
        )

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"
        if not sum_ok:
            reason += " | Softmax sums deviate from 1.0"

        print_test_linear(
            module="AttnWeights+Softmax",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
            input_shape=[N, Len_q, d_model],
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
            input_samples={"query": test_data},
        )

        TEST_RESULTS.append(
            {
                "module": "AttnWeights+Softmax",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": [N, Len_q, d_model],
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
                "softmax_sum_ok": sum_ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "AttnWeights+Softmax",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{[N, Len_q, d_model]}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'PASS' if ok else 'FAIL'} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` - {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Shape:** `{[N, Len_q, d_model]}` -> **Output Shape:** `{pt_shape}`\n\n"
            f"**Softmax sum check:** {'PASS' if sum_ok else 'FAIL'}\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- query: `[{_fmt_samples(test_data)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["AttnWeights+Softmax"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_attn_weights),
        "num_passed": num_passed,
        "num_total": len(test_cases_attn_weights),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "AttnWeights+Softmax",
            "description": "attention_weights linear + softmax normalization",
            "passed": passed,
            "total": len(test_cases_attn_weights),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_attn_weights
    ), f"AttnWeights+Softmax: {passed}/{len(test_cases_attn_weights)} passed"


# ===========================================================================
#  TEST 4 — Sampling Locations: NUMERICAL VALIDATION
# ===========================================================================
test_cases_sampling_locs = [
    # (description, data_type, ref_points_dim, category)
    ("Baseline positive 2D", "positive", 2, "baseline"),
    ("Negative values 2D", "negative", 2, "edge_value"),
    ("Zero values 2D", "zeros", 2, "edge_value"),
    ("Mixed pos/neg 2D", "mixed", 2, "edge_value"),
    ("Very small (1e-6) 2D", "small", 2, "edge_value"),
    ("Very large (1e6) 2D", "large", 2, "edge_value"),
    ("Boundary coords 2D", "boundary_coords", 2, "edge_coord"),
    ("Center coords 2D", "center_coords", 2, "edge_coord"),
    ("Corner coords 2D", "corner_coords", 2, "edge_coord"),
    ("4D reference points", "positive", 4, "edge_shape"),
]


@pytest.mark.unit
def test_sampling_locations():
    """Test sampling location computation: shape + numerical across data/coord types."""
    _seed()
    d_model = DEFAULT_D_MODEL
    n_heads, n_levels, n_points = DEFAULT_N_HEADS, DEFAULT_N_LEVELS, DEFAULT_N_POINTS
    N = DEFAULT_BATCH
    Len_q = DEFAULT_LEN_Q
    spatial_shapes = DEFAULT_SPATIAL

    module_pt, module_tt = _create_modules(d_model, n_levels, n_heads, n_points)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, data_type, ref_dim, category) in enumerate(
        test_cases_sampling_locs
    ):
        _seed(tno)

        # Generate query for sampling offsets
        query_np = generate_test_data([N, Len_q, d_model], data_type)
        ref_pts_np = generate_ref_points([N, Len_q, n_levels, ref_dim], data_type)

        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        # PyTorch: sampling_offsets(query) -> view + compute sampling_locations
        q_torch = torch.from_numpy(query_np)
        rp_torch = torch.from_numpy(ref_pts_np)
        ss_torch = torch.from_numpy(spatial_shapes)

        with torch.no_grad():
            so_pt = module_pt.sampling_offsets(q_torch).view(
                N, Len_q, n_heads, n_levels, n_points, 2
            )
            if ref_dim == 2:
                offset_norm = torch.stack([ss_torch[..., 1], ss_torch[..., 0]], -1)
                sl_pt = (
                    rp_torch[:, :, None, :, None, :]
                    + so_pt / offset_norm[None, None, None, :, None, :]
                )
            else:
                sl_pt = (
                    rp_torch[:, :, None, :, None, :2]
                    + so_pt / n_points * rp_torch[:, :, None, :, None, 2:] * 0.5
                )

        # TTSim: sampling_offsets(query) -> reshape + _compute_sampling_locations
        q_sim = SimTensor(
            {
                "name": "query",
                "shape": list(query_np.shape),
                "data": query_np.copy(),
                "dtype": np.float32,
            }
        )
        so_tt = module_tt.sampling_offsets(q_sim)
        so_tt.shape = [N, Len_q, n_heads, n_levels, n_points, 2]
        so_tt.data = so_tt.data.reshape(so_tt.shape)

        rp_sim = SimTensor(
            {
                "name": "ref_pts",
                "shape": list(ref_pts_np.shape),
                "data": ref_pts_np.copy(),
                "dtype": np.float32,
            }
        )
        ss_sim = SimTensor(
            {
                "name": "spatial_shapes",
                "shape": list(spatial_shapes.shape),
                "data": spatial_shapes.copy(),
                "dtype": np.int32,
            }
        )
        sl_tt = module_tt._compute_sampling_locations(rp_sim, so_tt, ss_sim)

        pt_shape = list(sl_pt.shape)
        tt_shape = sl_tt.shape
        shape_ok = pt_shape == tt_shape

        pt_d = _to_numpy(sl_pt)
        tt_d = _to_numpy(sl_tt)
        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=rtol, atol=atol))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        edge_label = data_type if ref_dim == 2 else "4d_ref_points"
        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PyTorch={pt_shape} vs TTSim={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={atol}"

        print_test_linear(
            module="SamplingLocations",
            edge_case=edge_label,
            edge_desc=EDGE_CASE_DESC.get(
                edge_label, EDGE_CASE_DESC.get(data_type, "Standard test")
            ),
            input_shape=f"query={[N,Len_q,d_model]} ref_pts={[N,Len_q,n_levels,ref_dim]}",
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
            input_samples={"query": query_np, "ref_points": ref_pts_np},
        )

        TEST_RESULTS.append(
            {
                "module": "SamplingLocations",
                "validation_type": "NUMERICAL",
                "edge_case": edge_label,
                "edge_desc": EDGE_CASE_DESC.get(edge_label, ""),
                "input_shape": f"query=[{N},{Len_q},{d_model}] ref_pts=[{N},{Len_q},{n_levels},{ref_dim}]",
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
            failed_cases.append((tno, tmsg, edge_label, mx))
            FAILED_TESTS.append(
                {
                    "module": "SamplingLocations",
                    "test": tmsg,
                    "edge_case": edge_label,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {edge_label} | ref_dim={ref_dim} | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'PASS' if ok else 'FAIL'} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{edge_label}` - {EDGE_CASE_DESC.get(edge_label, 'N/A')}\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- query: `[{_fmt_samples(query_np)}]`\n"
            f"- ref_points: `[{_fmt_samples(ref_pts_np)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["SamplingLocations"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_sampling_locs),
        "num_passed": num_passed,
        "num_total": len(test_cases_sampling_locs),
    }

    hdr = (
        "| # | Test Case | Edge Case | Ref Dim | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:--------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "SamplingLocations",
            "description": "Sampling location computation from reference points + offsets",
            "passed": passed,
            "total": len(test_cases_sampling_locs),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_sampling_locs
    ), f"SamplingLocations: {passed}/{len(test_cases_sampling_locs)} passed"


# ===========================================================================
#  TEST 5 — MSDeformAttn End-to-End (2D ref points): NUMERICAL VALIDATION
# ===========================================================================
test_cases_e2e_2d = [
    # (description, data_type, d_model, n_levels, n_heads, n_points, N, Len_q, spatial_shapes, category)
    (
        "Baseline positive",
        "positive",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "baseline",
    ),
    (
        "Negative values",
        "negative",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "edge_value",
    ),
    (
        "Zero values",
        "zeros",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "edge_value",
    ),
    (
        "Mixed pos/neg",
        "mixed",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "edge_value",
    ),
    (
        "Very small (1e-6)",
        "small",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "edge_value",
    ),
    (
        "Very large (1e6)",
        "large",
        256,
        4,
        8,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "edge_value",
    ),
    (
        "Minimum input",
        "positive",
        128,
        2,
        4,
        2,
        1,
        10,
        np.array([[4, 4], [2, 2]], dtype=np.int32),
        "minimum_input",
    ),
]


@pytest.mark.unit
def test_e2e_2d_ref_points():
    """Test full MSDeformAttn module with 2D reference points: all mandatory edge cases."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        data_type,
        d_model,
        n_levels,
        n_heads,
        n_points,
        N,
        Len_q,
        spatial_shapes,
        category,
    ) in enumerate(test_cases_e2e_2d):
        _seed(tno)

        Len_in = _compute_len_in(spatial_shapes)
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        module_pt, module_tt = _create_modules(d_model, n_levels, n_heads, n_points)

        query_np = generate_test_data([N, Len_q, d_model], data_type)
        ref_pts_np = generate_ref_points([N, Len_q, n_levels, 2], data_type)
        input_flat_np = generate_test_data([N, Len_in, d_model], data_type)

        # Compute level start index for PyTorch
        level_start = np.concatenate(
            [np.array([0]), np.cumsum([int(h * w) for h, w in spatial_shapes])[:-1]]
        )
        level_start_torch = torch.from_numpy(level_start.astype(np.int64))

        # PyTorch forward
        q_torch = torch.from_numpy(query_np)
        rp_torch = torch.from_numpy(ref_pts_np)
        if_torch = torch.from_numpy(input_flat_np)
        ss_torch = torch.from_numpy(spatial_shapes)

        with torch.no_grad():
            out_pt = module_pt(q_torch, rp_torch, if_torch, ss_torch, level_start_torch)

        # TTSim forward
        q_sim = SimTensor(
            {
                "name": "query",
                "shape": list(query_np.shape),
                "data": query_np.copy(),
                "dtype": np.float32,
            }
        )
        rp_sim = SimTensor(
            {
                "name": "ref_pts",
                "shape": list(ref_pts_np.shape),
                "data": ref_pts_np.copy(),
                "dtype": np.float32,
            }
        )
        if_sim = SimTensor(
            {
                "name": "input_flatten",
                "shape": list(input_flat_np.shape),
                "data": input_flat_np.copy(),
                "dtype": np.float32,
            }
        )
        ss_sim = SimTensor(
            {
                "name": "spatial_shapes",
                "shape": list(spatial_shapes.shape),
                "data": spatial_shapes.copy(),
                "dtype": np.int32,
            }
        )

        out_tt = module_tt.forward(q_sim, rp_sim, if_sim, ss_sim)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
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
            module="MSDeformAttn E2E (2D)",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(
                data_type, EDGE_CASE_DESC.get(category, "Standard test")
            ),
            input_shape=f"query={[N,Len_q,d_model]} flat={[N,Len_in,d_model]}",
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
            input_samples={
                "query": query_np,
                "input_flatten": input_flat_np,
                "ref_points": ref_pts_np,
            },
        )

        TEST_RESULTS.append(
            {
                "module": "MSDeformAttn E2E (2D)",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": f"query=[{N},{Len_q},{d_model}] flat=[{N},{Len_in},{d_model}]",
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
                "note": f"relaxed tolerance" if data_type == "large" else "",
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, data_type, mx))
            FAILED_TESTS.append(
                {
                    "module": "MSDeformAttn E2E (2D)",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        tol_info = (
            f" *(relaxed: rtol={rtol}, atol={atol})*" if data_type == "large" else ""
        )
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | d={d_model} L={n_levels} "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'PASS' if ok else 'FAIL'} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` - {EDGE_CASE_DESC.get(data_type, 'N/A')}{tol_info}\n\n"
            f"**Config:** d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, "
            f"n_points={n_points}, N={N}, Len_q={Len_q}\n\n"
            f"**Spatial Shapes:** `{spatial_shapes.tolist()}`\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- query: `[{_fmt_samples(query_np)}]`\n"
            f"- input_flatten: `[{_fmt_samples(input_flat_np)}]`\n"
            f"- ref_points: `[{_fmt_samples(ref_pts_np)}]`\n\n"
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

    MODULE_STATS["MSDeformAttn E2E (2D)"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_e2e_2d),
        "num_passed": num_passed,
        "num_total": len(test_cases_e2e_2d),
    }

    hdr = (
        "| # | Test Case | Data Type | Config | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "MSDeformAttn E2E (2D)",
            "description": "Full MSDeformAttn end-to-end with 2D reference points",
            "passed": passed,
            "total": len(test_cases_e2e_2d),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_e2e_2d
    ), f"MSDeformAttn E2E (2D): {passed}/{len(test_cases_e2e_2d)} passed"


# ===========================================================================
#  TEST 6 — MSDeformAttn End-to-End (4D ref points): NUMERICAL VALIDATION
# ===========================================================================
test_cases_e2e_4d = [
    # (description, data_type, category)
    ("Baseline positive 4D", "positive", "baseline"),
    ("Negative values 4D", "negative", "edge_value"),
    ("Zero values 4D", "zeros", "edge_value"),
    ("Mixed pos/neg 4D", "mixed", "edge_value"),
    ("Very small (1e-6) 4D", "small", "edge_value"),
    ("Very large (1e6) 4D", "large", "edge_value"),
]


@pytest.mark.unit
def test_e2e_4d_ref_points():
    """Test full MSDeformAttn module with 4D reference points: all mandatory edge cases."""
    d_model, n_levels, n_heads, n_points = 256, 4, 8, 4
    N, Len_q = 2, 50
    spatial_shapes = DEFAULT_SPATIAL
    Len_in = _compute_len_in(spatial_shapes)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, data_type, category) in enumerate(test_cases_e2e_4d):
        _seed(tno)
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        module_pt, module_tt = _create_modules(d_model, n_levels, n_heads, n_points)

        query_np = generate_test_data([N, Len_q, d_model], data_type)
        # 4D ref points: (x, y, w, h) in [0, 1]
        ref_pts_np = np.random.rand(N, Len_q, n_levels, 4).astype(np.float32)
        # Ensure w, h > 0 for valid reference boxes
        ref_pts_np[..., 2:] = np.clip(ref_pts_np[..., 2:], 0.1, 0.9)
        input_flat_np = generate_test_data([N, Len_in, d_model], data_type)

        level_start = np.concatenate(
            [np.array([0]), np.cumsum([int(h * w) for h, w in spatial_shapes])[:-1]]
        )
        level_start_torch = torch.from_numpy(level_start.astype(np.int64))

        # PyTorch
        q_torch = torch.from_numpy(query_np)
        rp_torch = torch.from_numpy(ref_pts_np)
        if_torch = torch.from_numpy(input_flat_np)
        ss_torch = torch.from_numpy(spatial_shapes)

        with torch.no_grad():
            out_pt = module_pt(q_torch, rp_torch, if_torch, ss_torch, level_start_torch)

        # TTSim
        q_sim = SimTensor(
            {
                "name": "query",
                "shape": list(query_np.shape),
                "data": query_np.copy(),
                "dtype": np.float32,
            }
        )
        rp_sim = SimTensor(
            {
                "name": "ref_pts",
                "shape": list(ref_pts_np.shape),
                "data": ref_pts_np.copy(),
                "dtype": np.float32,
            }
        )
        if_sim = SimTensor(
            {
                "name": "input_flatten",
                "shape": list(input_flat_np.shape),
                "data": input_flat_np.copy(),
                "dtype": np.float32,
            }
        )
        ss_sim = SimTensor(
            {
                "name": "spatial_shapes",
                "shape": list(spatial_shapes.shape),
                "data": spatial_shapes.copy(),
                "dtype": np.int32,
            }
        )

        out_tt = module_tt.forward(q_sim, rp_sim, if_sim, ss_sim)

        pt_shape = list(out_pt.shape)
        tt_shape = out_tt.shape
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
            module="MSDeformAttn E2E (4D)",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
            input_shape=f"query={[N,Len_q,d_model]} flat={[N,Len_in,d_model]} ref_dim=4",
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
            input_samples={
                "query": query_np,
                "input_flatten": input_flat_np,
                "ref_points_4d": ref_pts_np,
            },
        )

        TEST_RESULTS.append(
            {
                "module": "MSDeformAttn E2E (4D)",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, ""),
                "input_shape": f"query=[{N},{Len_q},{d_model}] flat=[{N},{Len_in},{d_model}] ref_dim=4",
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
                    "module": "MSDeformAttn E2E (4D)",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        tag = "PASS" if ok else "**FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {data_type} | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'PASS' if ok else 'FAIL'} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` - {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- query: `[{_fmt_samples(query_np)}]`\n"
            f"- input_flatten: `[{_fmt_samples(input_flat_np)}]`\n"
            f"- ref_points_4d: `[{_fmt_samples(ref_pts_np)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["MSDeformAttn E2E (4D)"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_e2e_4d),
        "num_passed": num_passed,
        "num_total": len(test_cases_e2e_4d),
    }

    hdr = (
        "| # | Test Case | Data Type | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "MSDeformAttn E2E (4D)",
            "description": "Full MSDeformAttn end-to-end with 4D reference points (x, y, w, h)",
            "passed": passed,
            "total": len(test_cases_e2e_4d),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_e2e_4d
    ), f"MSDeformAttn E2E (4D): {passed}/{len(test_cases_e2e_4d)} passed"


# ===========================================================================
#  TEST 7 — Shape Inference (data=None): SHAPE VALIDATION ONLY
# ===========================================================================
test_cases_shape_inf = [
    # (description, d_model, n_levels, n_heads, n_points, N, Len_q, spatial_shapes, category)
    (
        "Standard 4-level",
        256,
        4,
        8,
        4,
        2,
        100,
        np.array([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=np.int32),
        "standard",
    ),
    (
        "Small config",
        128,
        2,
        4,
        2,
        1,
        50,
        np.array([[32, 32], [16, 16]], dtype=np.int32),
        "small_config",
    ),
    (
        "Single level",
        256,
        1,
        8,
        4,
        2,
        100,
        np.array([[50, 50]], dtype=np.int32),
        "single_level",
    ),
    (
        "Many heads (16)",
        256,
        4,
        16,
        4,
        2,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "many_heads",
    ),
    (
        "Single query",
        256,
        4,
        8,
        4,
        1,
        1,
        np.array([[8, 8], [4, 4], [2, 2], [1, 1]], dtype=np.int32),
        "single_query",
    ),
    (
        "Large batch (4)",
        256,
        4,
        8,
        4,
        4,
        50,
        np.array([[16, 16], [8, 8], [4, 4], [2, 2]], dtype=np.int32),
        "large_batch",
    ),
    (
        "Minimum input",
        64,
        1,
        2,
        1,
        1,
        1,
        np.array([[2, 2]], dtype=np.int32),
        "minimum_input",
    ),
]


@pytest.mark.unit
def test_shape_inference():
    """Test MSDeformAttn shape inference (data=None) across various configurations."""
    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        d_model,
        n_levels,
        n_heads,
        n_points,
        N,
        Len_q,
        spatial_shapes,
        category,
    ) in enumerate(test_cases_shape_inf):

        Len_in = _compute_len_in(spatial_shapes)

        module_tt = MSDeformAttnTTSim(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points
        )

        # Shape-only inputs (data=None)
        q_sim = SimTensor(
            {
                "name": "query",
                "shape": [N, Len_q, d_model],
                "data": None,
                "dtype": np.float32,
            }
        )
        rp_sim = SimTensor(
            {
                "name": "ref_pts",
                "shape": [N, Len_q, n_levels, 2],
                "data": None,
                "dtype": np.float32,
            }
        )
        if_sim = SimTensor(
            {
                "name": "input_flatten",
                "shape": [N, Len_in, d_model],
                "data": None,
                "dtype": np.float32,
            }
        )
        ss_sim = SimTensor(
            {
                "name": "spatial_shapes",
                "shape": [n_levels, 2],
                "data": None,
                "dtype": np.int32,
            }
        )

        out_tt = module_tt.forward(q_sim, rp_sim, if_sim, ss_sim)

        expected_shape = [N, Len_q, d_model]
        actual_shape = out_tt.shape
        shape_ok = expected_shape == actual_shape
        passed += int(shape_ok)

        reason = ""
        if not shape_ok:
            reason = (
                f"Shape mismatch: expected={expected_shape} vs actual={actual_shape}"
            )

        print_test_linear(
            module="ShapeInference",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, tmsg),
            input_shape=f"query={[N,Len_q,d_model]} flat={[N,Len_in,d_model]}",
            shape_line=f"Expected={_compact_shape(expected_shape)} | Actual={_compact_shape(actual_shape)}",
            shape_ok=shape_ok,
            is_numerical=False,
            failure_reason=reason,
        )

        TEST_RESULTS.append(
            {
                "module": "ShapeInference",
                "validation_type": "SHAPE ONLY",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": f"query=[{N},{Len_q},{d_model}] flat=[{N},{Len_in},{d_model}]",
                "pt_shape": expected_shape,
                "tt_shape": actual_shape,
                "shape_ok": shape_ok,
                "num_ok": None,
                "max_diff": None,
                "mean_diff": None,
                "pt_stats": None,
                "tt_stats": None,
                "passed": shape_ok,
            }
        )

        if not shape_ok:
            failed_cases.append((tno, tmsg, category, 0))
            FAILED_TESTS.append(
                {
                    "module": "ShapeInference",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": 0,
                }
            )

        tag = "PASS" if shape_ok else "**FAIL**"
        cfg_str = f"d={d_model} L={n_levels} M={n_heads} P={n_points}"
        rows.append(
            f"| {tno} | {tmsg} | {cfg_str} | `{expected_shape}` | `{actual_shape}` | {tag} |"
        )
        detail_blocks.append(
            f"---\n\n### {'PASS' if shape_ok else 'FAIL'} TEST[{tno}] {tmsg}\n\n"
            f"**Config:** d_model={d_model}, n_levels={n_levels}, n_heads={n_heads}, "
            f"n_points={n_points}, N={N}, Len_q={Len_q}\n\n"
            f"**Spatial Shapes:** `{spatial_shapes.tolist()}`\n\n"
            f"**Expected Shape:** `{expected_shape}`\n"
            f"**Actual Shape:** `{actual_shape}`\n\n"
        )

    MODULE_STATS["ShapeInference"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_shape_inf),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Config | Expected | Actual | Result |\n"
        "|:--|:----------|:-------|:---------|:-------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "ShapeInference",
            "description": "MSDeformAttn shape inference (data=None) across configurations",
            "passed": passed,
            "total": len(test_cases_shape_inf),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_shape_inf
    ), f"ShapeInference: {passed}/{len(test_cases_shape_inf)} passed"


# ===========================================================================
#  Markdown report writer
# ===========================================================================


def _write_markdown_report(report_path, exit_code):
    """Generate a simple, module-wise markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# MSDeformAttn Module Unit Test Report",
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

    # Failed tests (if any)
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
    lines.append(f"- Default tolerance: rtol={RTOL}, atol={ATOL}")
    lines.append(f"- Large value tolerance: rtol=1e-3, atol=1e-2")
    lines.append(f"- Random Seed: {SEED}")
    lines.append(
        f"- Default config: d_model={DEFAULT_D_MODEL}, n_levels={DEFAULT_N_LEVELS}, "
        f"n_heads={DEFAULT_N_HEADS}, n_points={DEFAULT_N_POINTS}"
    )
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sync_globals_from_pytest():
    """After pytest.main() re-imports this file under its real module name,
    copy the populated global collectors back into __main__."""
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
    report_path = os.path.join(report_dir, "ms_deform_attn_module_unit_test_report.md")
    output_path = os.path.join(report_dir, "ms_deform_attn_module_unit_test_output.md")

    # Tee stdout -> terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    # Print overall header
    print(f"\n{SUMMARY_LINE}")
    print(f"MSDeformAttn MODULE UNIT TEST SUITE - PyTorch vs TTSim")
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
