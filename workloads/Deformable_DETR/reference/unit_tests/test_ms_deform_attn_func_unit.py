#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ms_deform_attn_core — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPES EXPLAINED:
================================================================================

SHAPE VALIDATION:
    - Runs TTSim in shape-inference mode (data=None)
    - Verifies output shape matches expected [N, Lq, M*D]
    - PASS if: shapes match
    - FAIL if: shape mismatch

NUMERICAL VALIDATION:
    - Compares actual OUTPUT VALUES between PyTorch and TTSim
    - Uses np.allclose() with tolerance: rtol=1e-4, atol=1e-5
    - Reports: max absolute difference, mean absolute difference
    - PASS if: all values within tolerance
    - FAIL if: any value exceeds tolerance

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: Shape Inference — SHAPE VALIDATION
    Verifies TTSim produces correct output shape [N, Lq, M*D] in
    shape-inference mode (data=None) across different configurations.
    Edge Cases: standard_multi_level, single_level, many_points,
                large_batch, single_query, many_heads,
                non_square_spatial, minimal_config,
                single_element_pooling, minimum_input
    WHY SHAPE: Validates the shape-inference path is correct

MODULE 2: End-to-End Numerical — NUMERICAL VALIDATION
    Compares full ms_deform_attn_core_pytorch output vs
    ms_deform_attn_core_ttsim output (data mode).
    Edge Cases: positive, negative, zeros, mixed, small, large,
                single_level, many_points, large_batch, many_heads,
                single_query, uniform_attention, concentrated_attention,
                boundary_coords, center_coords, corner_coords,
                single_element_pooling, minimum_input
    Tolerance: rtol=1e-4, atol=1e-5 (relaxed for large values)
    WHY NUMERICAL: Validates the entire pipeline end-to-end

MODULE 3: Per-Level Grid Sample — NUMERICAL VALIDATION
    Compares intermediate per-level grid_sample outputs between
    PyTorch and TTSim to isolate per-level numerical accuracy.
    Edge Cases: standard_4_levels, single_level, two_levels_asymmetric,
                negative_values, zeros_values, mixed_values,
                small_values, large_values,
                single_element_pooling, minimum_input
    Tolerance: rtol=1e-4, atol=1e-5 (relaxed for large values)
    WHY NUMERICAL: Isolates the grid_sample accuracy per level

MODULE 4: Intermediate Steps — NUMERICAL VALIDATION
    Compares step-by-step intermediates (sampling_grids, stacked,
    flattened, attention_reshaped, weighted, output).
    Edge Cases: standard_config, small_config, negative_values,
                zeros_values, mixed_values, small_values,
                large_values, single_element_pooling, minimum_input
    Tolerance: rtol=1e-4, atol=1e-5 (relaxed for large values)
    WHY NUMERICAL: Pin-points which step diverges first

================================================================================
EDGE CASES TESTED:
================================================================================

--- Value Data Types (backbone-style) ---
'positive'              — Standard positive values (1.0-2.0) — baseline
'negative'              — All negative values (-2.0 to -1.0) — sign handling
'zeros'                 — All zeros — zero feature maps
'mixed'                 — Mix of positive/negative — real-world distribution
'small'                 — Very small values (~1e-6) — precision near zero
'large'                 — Very large values (~1e6) — overflow handling

--- Geometry ---
'standard'              — Multi-level, typical dims — baseline correctness
'single_level'          — L=1, single feature level — minimal level loop
'many_points'           — P=8 sampling points — more interpolation
'large_batch'           — N=4 batch size — batch scalability
'single_query'          — Lq=1 — minimal query count
'many_heads'            — M=16 attention heads — head scalability
'non_square_spatial'    — Non-square spatial dims (H!=W) — asymmetric grids
'minimal_config'        — Tiny dims for quick smoke test
'two_levels_asymmetric' — 2 levels with very different spatial sizes

--- Attention Weights ---
'uniform_attention'     — Equal attention weights — no weighting bias
'concentrated_attention'— One-hot attention (all weight on first point)

--- Sampling Coordinates ---
'boundary_coords'       — Sampling near 0.0 and 1.0 — grid edge behavior
'center_coords'         — All locations at 0.5 — center-only sampling
'corner_coords'         — All at 0.0 or 1.0 — extreme corner positions

--- Pooling / Size Edge Cases ---
'single_element_pooling' — P=1, single sampling point — minimal interpolation
'minimum_input'          — Smallest valid config (all dims minimal) — degenerate case

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_ms_deform_attn_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_ms_deform_attn_unit.py
================================================================================
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F
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

# PyTorch implementation
from workloads.Deformable_DETR.reference.ms_deform_attn_func import (
    ms_deform_attn_core_pytorch,
)

# TTSim implementation
from workloads.Deformable_DETR.models.ops.functions.ms_deform_attn_func_ttsim import (
    ms_deform_attn_core_ttsim,
)
from ttsim.ops.tensor import SimTensor
from ttsim.ops.desc.nn import grid_sample_fwd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 1e-4
ATOL = 1e-5
LARGE_RTOL = 1e-3  # relaxed for large-value tests
LARGE_ATOL = 1e-2
SEED = 42

# Expected number of pytest test functions (used to detect crashed tests)
EXPECTED_MODULES = [
    "ShapeInference",
    "E2E_Numerical",
    "PerLevelGridSample",
    "IntermediateSteps",
]


# ---------------------------------------------------------------------------
# Terminal Colors (ANSI escape codes)
# ---------------------------------------------------------------------------
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"

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
    def bold(s):
        return f"{Colors.BOLD}{s}{Colors.RESET}"

    @staticmethod
    def cyan(s):
        return f"{Colors.CYAN}{s}{Colors.RESET}"

    @staticmethod
    def dim(s):
        return f"{Colors.DIM}{s}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Visual separators
# ---------------------------------------------------------------------------
SUMMARY_LINE = "═" * 65
DIVIDER_LINE = "─" * 65


# ---------------------------------------------------------------------------
# Edge case descriptions
# ---------------------------------------------------------------------------
EDGE_CASE_DESC = {
    # Value data types (backbone-style)
    "positive": "Standard positive values (1.0-2.0) — baseline test",
    "negative": "All negative values (-2.0 to -1.0) — sign handling",
    "zeros": "All zeros — zero feature maps edge case",
    "mixed": "Mix of positive/negative — real-world distribution",
    "small": "Very small values (~1e-6) — precision near zero",
    "large": "Very large values (~1e6) — overflow handling",
    # Geometry
    "standard": "Multi-level typical dims — baseline correctness",
    "single_level": "L=1, single feature level — minimal level loop",
    "many_points": "P=8 sampling points — more interpolation",
    "large_batch": "N=4 batch size — batch scalability",
    "single_query": "Lq=1 — minimal query count",
    "many_heads": "M=16 attention heads — head scalability",
    "non_square_spatial": "Non-square spatial dims (H!=W) — asymmetric grids",
    "minimal_config": "Tiny dims for quick smoke test",
    "two_levels_asymmetric": "2 levels with very different spatial sizes",
    "standard_multi_level": "4 levels, typical multi-scale — baseline shape",
    "standard_4_levels": "4 levels — per-level grid_sample check",
    "standard_config": "Standard config — full intermediate comparison",
    "small_config": "Tiny dims for quick smoke test",
    # Attention
    "uniform_attention": "Equal attention weights — no weighting bias",
    "concentrated_attention": "One-hot attention (all weight on first point)",
    # Coordinates
    "boundary_coords": "Sampling near 0.0 and 1.0 — grid edge behavior",
    "center_coords": "All locations at 0.5 — center-only sampling",
    "corner_coords": "All at 0.0 or 1.0 — extreme corner positions",
    # Composite
    "negative_values": "Negative feature values with standard geometry",
    "small_values": "Very small feature values with standard geometry",
    "large_values": "Very large feature values — overflow risk",
    "zeros_values": "All-zero feature values — zero output edge case",
    "mixed_values": "Mix of positive/negative feature values",
    # Pooling / Size
    "single_element_pooling": "P=1, single sampling point — minimal interpolation",
    "minimum_input": "Smallest valid config (all dims minimal) — degenerate case",
}


# ---------------------------------------------------------------------------
# Report data collectors
# ---------------------------------------------------------------------------
REPORT_SECTIONS = []
FAILED_TESTS = []
TEST_RESULTS = []
MODULE_STATS = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seed(extra=0):
    torch.manual_seed(SEED + extra)
    np.random.seed(SEED + extra)


def _compact_shape(shape):
    """Format shape as [N,M] (no spaces)."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def _fmt_samples(arr, n=10):
    """First *n* flat values, formatted for a markdown table cell."""
    flat = np.asarray(arr).flatten()
    return ", ".join(f"{v:.6f}" for v in flat[:n])


def generate_value_data(shape, value_mode):
    """Generate value tensor data based on mode (backbone-style).

    Args:
        shape: tuple, e.g. (N, S, M, D)
        value_mode: one of 'positive', 'negative', 'zeros', 'mixed',
                    'small', 'large', 'random'
    Returns:
        numpy float32 array of given shape
    """
    if value_mode == "positive":
        return np.random.rand(*shape).astype(np.float32) + 1.0
    elif value_mode == "negative":
        return -np.random.rand(*shape).astype(np.float32) - 1.0
    elif value_mode == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif value_mode == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)
    elif value_mode == "small":
        return np.random.rand(*shape).astype(np.float32) * 1e-6
    elif value_mode == "large":
        return np.random.rand(*shape).astype(np.float32) * 1e6
    else:  # 'random'
        return np.random.randn(*shape).astype(np.float32)


def _wrap_sim(arr, name="tensor"):
    """Wrap a numpy ndarray as a SimTensor.

    Needed because numpy ndarray has a .data attribute (memoryview buffer)
    that the TTSim function's ``hasattr(x, 'data')`` check intercepts
    before the ``isinstance(np.ndarray)`` check.  Wrapping as SimTensor
    ensures the ``isinstance(SimTensor)`` branch fires first, returning
    the real numpy array via ``.data``.
    """
    return SimTensor(
        {
            "name": name,
            "shape": list(arr.shape),
            "data": arr,
            "dtype": arr.dtype,
        }
    )


def _generate_inputs(
    N,
    Lq,
    M,
    D,
    L,
    P,
    spatial_shapes,
    seed=SEED,
    value_mode="random",
    attention_mode="random",
    coord_mode="random",
):
    """Generate matched PyTorch & numpy inputs for ms_deform_attn.

    Args:
        value_mode: 'random', 'positive', 'negative', 'zeros', 'mixed',
                    'small', 'large'
        attention_mode: 'random', 'uniform', 'concentrated'
        coord_mode: 'random', 'boundary', 'center', 'corner'

    Returns:
        (value_pt, spatial_pt, sampling_pt, attn_pt,
         value_np, spatial_np, sampling_np, attn_np)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    spatial_np = np.array(spatial_shapes, dtype=np.int64)
    S = int(np.sum(spatial_np[:, 0] * spatial_np[:, 1]))

    # Value tensor
    value_np = generate_value_data((N, S, M, D), value_mode)

    # Sampling coordinates
    if coord_mode == "boundary":
        sampling_np = np.random.choice(
            [0.0, 0.25, 0.5, 0.75, 1.0], size=(N, Lq, M, L, P, 2)
        ).astype(np.float32)
    elif coord_mode == "center":
        sampling_np = np.full((N, Lq, M, L, P, 2), 0.5, dtype=np.float32)
    elif coord_mode == "corner":
        sampling_np = np.random.choice([0.0, 1.0], size=(N, Lq, M, L, P, 2)).astype(
            np.float32
        )
    else:  # 'random'
        sampling_np = np.random.rand(N, Lq, M, L, P, 2).astype(np.float32)

    # Attention weights
    if attention_mode == "uniform":
        attn_np = np.ones((N, Lq, M, L, P), dtype=np.float32) / (L * P)
    elif attention_mode == "concentrated":
        attn_np = np.zeros((N, Lq, M, L, P), dtype=np.float32)
        attn_np[:, :, :, 0, 0] = 1.0  # all weight on first level, first point
    else:  # 'random'
        attn_np = np.random.rand(N, Lq, M, L, P).astype(np.float32)
        attn_np = attn_np / attn_np.sum(axis=-1, keepdims=True)

    value_pt = torch.from_numpy(value_np.copy())
    spatial_pt = torch.from_numpy(spatial_np.copy())
    sampling_pt = torch.from_numpy(sampling_np.copy())
    attn_pt = torch.from_numpy(attn_np.copy())

    return (
        value_pt,
        spatial_pt,
        sampling_pt,
        attn_pt,
        value_np,
        spatial_np,
        sampling_np,
        attn_np,
    )


def _to_numpy(x):
    """Coerce PyTorch tensor / SimTensor / ndarray to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, SimTensor):
        return x.data
    return np.asarray(x)


def _pytorch_with_intermediates(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """PyTorch forward with all intermediates collected."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    intermediates = {}

    # 1. Split
    split_sizes = [H_ * W_ for H_, W_ in spatial_shapes]
    value_list = value.split(split_sizes, dim=1)

    # 2. Sampling grids
    sampling_grids = 2 * sampling_locations - 1
    intermediates["sampling_grids"] = sampling_grids.detach().cpu().numpy()

    # 3-4. Per-level grid sample
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(spatial_shapes):
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        intermediates[f"level_{lid_}_grid_sample"] = (
            sampling_value_l_.detach().cpu().numpy()
        )
        sampling_value_list.append(sampling_value_l_)

    # 5. Stack
    stacked = torch.stack(sampling_value_list, dim=-2)
    intermediates["stacked"] = stacked.detach().cpu().numpy()

    # 6. Flatten
    flattened = stacked.flatten(-2)
    intermediates["flattened"] = flattened.detach().cpu().numpy()

    # 7. Attention reshape
    attn_reshaped = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    intermediates["attention_reshaped"] = attn_reshaped.detach().cpu().numpy()

    # 8. Weighted sum
    weighted = (flattened * attn_reshaped).sum(-1)
    intermediates["weighted"] = weighted.detach().cpu().numpy()

    # 9-11. Final reshape
    output = weighted.view(N_, M_ * D_, Lq_).transpose(1, 2).contiguous()
    intermediates["output"] = output.detach().cpu().numpy()

    return output, intermediates


def _ttsim_with_intermediates(value_np, spatial_np, sampling_np, attn_np):
    """TTSim forward with all intermediates collected."""
    N_, S_, M_, D_ = value_np.shape
    _, Lq_, _, L_, P_, _ = sampling_np.shape

    intermediates = {}

    # 1. Split
    split_sizes = [int(H * W) for H, W in spatial_np]
    value_list = np.split(value_np, np.cumsum(split_sizes)[:-1], axis=1)

    # 2. Sampling grids
    sampling_grids = 2.0 * sampling_np - 1.0
    intermediates["sampling_grids"] = sampling_grids

    # 3-4. Per-level grid sample
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(spatial_np):
        H_, W_ = int(H_), int(W_)
        value_l_data = value_list[lid_]
        value_l_flat = value_l_data.reshape(N_, H_ * W_, M_ * D_)
        value_l_trans = value_l_flat.transpose(0, 2, 1)
        value_l_reshaped = value_l_trans.reshape(N_ * M_, D_, H_, W_)

        sampling_grid_l = sampling_grids[:, :, :, lid_, :, :]
        sampling_grid_l_trans = sampling_grid_l.transpose(0, 2, 1, 3, 4)
        sampling_grid_l_flat = sampling_grid_l_trans.reshape(N_ * M_, Lq_, P_, 2)

        input_t = SimpleNamespace(
            shape=list(value_l_reshaped.shape),
            data=value_l_reshaped,
            dtype=value_l_reshaped.dtype,
        )
        grid_t = SimpleNamespace(
            shape=list(sampling_grid_l_flat.shape),
            data=sampling_grid_l_flat,
            dtype=sampling_grid_l_flat.dtype,
        )
        output_t = SimpleNamespace(shape=None, data=None, dtype=None)
        op = SimpleNamespace(
            attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
            optype="GridSample",
        )
        grid_sample_fwd([input_t, grid_t], [output_t], op)

        intermediates[f"level_{lid_}_grid_sample"] = output_t.data
        sampling_value_list.append(output_t.data)

    # 5. Stack
    stacked = np.stack(sampling_value_list, axis=-2)
    intermediates["stacked"] = stacked

    # 6. Flatten
    flattened = stacked.reshape(N_ * M_, D_, Lq_, L_ * P_)
    intermediates["flattened"] = flattened

    # 7. Attention reshape
    attn_trans = attn_np.transpose(0, 2, 1, 3, 4)
    attn_reshaped = attn_trans.reshape(N_ * M_, 1, Lq_, L_ * P_)
    intermediates["attention_reshaped"] = attn_reshaped

    # 8. Weighted sum
    weighted = (flattened * attn_reshaped).sum(axis=-1)
    intermediates["weighted"] = weighted

    # 9-11. Final reshape
    output_viewed = weighted.reshape(N_, M_ * D_, Lq_)
    output_transposed = output_viewed.transpose(0, 2, 1)
    output_final = np.ascontiguousarray(output_transposed)
    intermediates["output"] = output_final

    return output_final, intermediates


# ---------------------------------------------------------------------------
# Tree-style output
# ---------------------------------------------------------------------------
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
                       samples to display (e.g. value, sampling, attn).
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

    shape_status = Colors.success("✓ MATCH") if shape_ok else Colors.fail("✗ MISMATCH")
    print(f"├─ SHAPE: {shape_line} → {shape_status}")

    if is_numerical and max_diff is not None:
        if num_ok:
            num_status = Colors.success(f"✓ PASS (tol: rtol={rtol}, atol={atol})")
        else:
            num_status = Colors.fail("✗ FAIL")
        print(
            f"├─ NUMERICAL: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} → {num_status}"
        )

    if is_numerical and pt_data is not None and tt_data is not None:
        pt_arr = np.asarray(pt_data).flat[:5]
        tt_arr = np.asarray(tt_data).flat[:5]
        pt_str = ", ".join(f"{v:.6f}" for v in pt_arr)
        tt_str = ", ".join(f"{v:.6f}" for v in tt_arr)
        print(f"├─ PT OUTPUT[0:5]: [{pt_str}]")
        print(f"├─ TT OUTPUT[0:5]: [{tt_str}]")

    if not passed and failure_reason:
        print(f"├─ FAILURE REASON: {Colors.fail(failure_reason)}")

    result_str = Colors.success("✓ PASS") if passed else Colors.fail("✗ FAIL")
    print(f"└─ RESULT: {result_str}")


def print_summary(exit_code=0):
    """Print the final summary table.

    Args:
        exit_code: pytest exit code. If non-zero and some expected modules
                   are missing from MODULE_STATS, they are shown as CRASHED.
    """
    print(f"\n{SUMMARY_LINE}")
    print("SUMMARY")
    print(SUMMARY_LINE)
    print(f"{'MODULE':<28}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

    total_sp = total_st = total_np_ = total_nt = 0
    all_passed = True

    # Detect modules that crashed (never populated MODULE_STATS)
    missing_modules = [m for m in EXPECTED_MODULES if m not in MODULE_STATS]

    for name, stats in MODULE_STATS.items():
        sp, st = stats["shape_passed"], stats["shape_total"]
        total_sp += sp
        total_st += st
        shape_str = f"{sp}/{st}"

        if stats["num_total"] is not None:
            npx, nt = stats["num_passed"], stats["num_total"]
            total_np_ += npx
            total_nt += nt
            num_str = f"{npx}/{nt}"
            mod_pass = (sp == st) and (npx == nt)
        else:
            num_str = "N/A"
            mod_pass = sp == st

        if not mod_pass:
            all_passed = False

        status = Colors.success("✓ PASS") if mod_pass else Colors.fail("✗ FAIL")
        print(f"{name:<28}{shape_str:<12}{num_str:<12}{status}")

    # Show crashed modules as FAIL
    for name in missing_modules:
        all_passed = False
        print(f"{name:<28}{'---':<12}{'---':<12}{Colors.fail('✗ CRASHED')}")

    print(DIVIDER_LINE)

    # If exit_code is non-zero, force overall to FAIL
    if exit_code != 0:
        all_passed = False

    total_num_str = f"{total_np_}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("✓ PASS") if all_passed else Colors.fail("✗ FAIL")
    print(f"{'TOTAL':<28}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

    if missing_modules:
        print(f"\n{Colors.fail('CRASHED TESTS (never completed):')}")
        for name in missing_modules:
            print(
                f"  - {name} — test function raised an exception before recording results"
            )

    if FAILED_TESTS:
        print(f"\n{Colors.fail('FAILED TESTS:')}")
        for ft in FAILED_TESTS:
            diff_str = f"max_diff={ft['max_diff']:.2e}" if ft.get("max_diff") else ""
            gt_str = f" > atol={ft.get('atol', ATOL)}" if ft.get("max_diff") else ""
            print(f"  - {ft['module']} | {ft['edge_case']} | {diff_str}{gt_str}")

    print(SUMMARY_LINE)


# ---------------------------------------------------------------------------
# Report tee-stream
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
#  TEST 1 — Shape Inference
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_shape = [
    # (description, N, Lq, M, D, L, P, spatial_shapes, edge_category)
    (
        "Standard multi-level",
        2,
        100,
        8,
        32,
        4,
        4,
        [[50, 50], [25, 25], [13, 13], [7, 7]],
        "standard_multi_level",
    ),
    ("Single level L=1", 1, 50, 4, 16, 1, 2, [[32, 32]], "single_level"),
    ("Many points P=8", 1, 64, 4, 16, 2, 8, [[16, 16], [8, 8]], "many_points"),
    (
        "Large batch N=4",
        4,
        80,
        8,
        32,
        3,
        4,
        [[20, 20], [10, 10], [5, 5]],
        "large_batch",
    ),
    ("Single query Lq=1", 2, 1, 4, 16, 2, 4, [[8, 8], [4, 4]], "single_query"),
    ("Many heads M=16", 1, 50, 16, 8, 2, 4, [[16, 16], [8, 8]], "many_heads"),
    ("Non-square spatial", 1, 30, 4, 16, 2, 4, [[16, 8], [8, 4]], "non_square_spatial"),
    ("Minimal config", 1, 2, 1, 4, 1, 1, [[2, 2]], "minimal_config"),
    # --- Edge case: Pooling / Size ---
    ("Single element pool P=1", 1, 10, 2, 8, 1, 1, [[4, 4]], "single_element_pooling"),
    ("Minimum input size", 1, 1, 1, 4, 1, 1, [[1, 1]], "minimum_input"),
]


@pytest.mark.unit
def test_shape_inference():
    """Test TTSim shape-inference mode (data=None) for ms_deform_attn_core."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0

    for tno, (tmsg, N, Lq, M, D, L, P, sp_list, category) in enumerate(
        test_cases_shape
    ):
        spatial_np = np.array(sp_list, dtype=np.int64)
        S = int(np.sum(spatial_np[:, 0] * spatial_np[:, 1]))

        # Build SimTensors with data=None
        value_sim = SimTensor(
            {
                "name": "value",
                "shape": [N, S, M, D],
                "data": None,
                "dtype": np.float32,
            }
        )
        sampling_sim = SimTensor(
            {
                "name": "sampling_locations",
                "shape": [N, Lq, M, L, P, 2],
                "data": None,
                "dtype": np.float32,
            }
        )
        attn_sim = SimTensor(
            {
                "name": "attention_weights",
                "shape": [N, Lq, M, L, P],
                "data": None,
                "dtype": np.float32,
            }
        )

        output_sim = ms_deform_attn_core_ttsim(
            value_sim, spatial_np, sampling_sim, attn_sim
        )

        expected = [N, Lq, M * D]
        actual = list(output_sim.shape)
        shape_ok = expected == actual
        ok = shape_ok
        shape_passed += int(shape_ok)
        passed += int(ok)

        reason = ""
        if not shape_ok:
            reason = f"Expected {expected} got {actual}"

        inp_desc = (
            f"value[{N},{S},{M},{D}] "
            f"sampling[{N},{Lq},{M},{L},{P},2] "
            f"attn[{N},{Lq},{M},{L},{P}]"
        )

        print_test_linear(
            module="ShapeInference",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"Expected={_compact_shape(expected)} | Actual={_compact_shape(actual)}",
            shape_ok=shape_ok,
            is_numerical=False,
            failure_reason=reason,
        )

        TEST_RESULTS.append(
            {
                "module": "ShapeInference",
                "validation_type": "SHAPE",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "pt_shape": expected,
                "tt_shape": actual,
                "shape_ok": shape_ok,
                "num_ok": None,
                "max_diff": None,
                "mean_diff": None,
                "passed": ok,
            }
        )

        if not ok:
            FAILED_TESTS.append(
                {
                    "module": "ShapeInference",
                    "test": tmsg,
                    "edge_case": category,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` "
            f"| `{expected}` | `{actual}` | {tag} |"
        )
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}`\n\n"
            f"**Expected:** `{expected}` → **Actual:** `{actual}`\n\n"
        )

    MODULE_STATS["ShapeInference"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_shape),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Expected | Actual | Result |\n"
        "|:--|:----------|:----------|:------|:---------|:-------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "ShapeInference",
            "description": "Shape inference mode (data=None) — verifies output shape [N, Lq, M*D]",
            "passed": passed,
            "total": len(test_cases_shape),
            "failed_cases": [
                (i, t[0], t[-1], None)
                for i, t in enumerate(test_cases_shape)
                if not TEST_RESULTS[i]["passed"]
            ],
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_shape
    ), f"ShapeInference: {passed}/{len(test_cases_shape)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — End-to-End Numerical
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_e2e = [
    # (desc, N, Lq, M, D, L, P, spatial_shapes,
    #  value_mode, attn_mode, coord_mode, use_relaxed_tol, edge_category)
    # --- Baseline (standard geometry, random values) ---
    (
        "Standard 4 levels",
        2,
        100,
        8,
        32,
        4,
        4,
        [[50, 50], [25, 25], [13, 13], [7, 7]],
        "random",
        "random",
        "random",
        False,
        "standard",
    ),
    # --- Value data types (backbone-style edge cases) ---
    (
        "Positive values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "positive",
        "random",
        "random",
        False,
        "positive",
    ),
    (
        "Negative values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "negative",
        "random",
        "random",
        False,
        "negative",
    ),
    (
        "Zero values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "zeros",
        "random",
        "random",
        False,
        "zeros",
    ),
    (
        "Mixed positive/negative",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "mixed",
        "random",
        "random",
        False,
        "mixed",
    ),
    (
        "Small values (~1e-6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "small",
        "random",
        "random",
        False,
        "small",
    ),
    (
        "Large values (~1e6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "large",
        "random",
        "random",
        True,
        "large",
    ),
    # --- Geometry edge cases ---
    (
        "Single level L=1",
        1,
        50,
        4,
        16,
        1,
        2,
        [[32, 32]],
        "random",
        "random",
        "random",
        False,
        "single_level",
    ),
    (
        "Many sampling points P=8",
        1,
        64,
        4,
        16,
        2,
        8,
        [[16, 16], [8, 8]],
        "random",
        "random",
        "random",
        False,
        "many_points",
    ),
    (
        "Large batch N=4",
        4,
        80,
        8,
        32,
        3,
        4,
        [[20, 20], [10, 10], [5, 5]],
        "random",
        "random",
        "random",
        False,
        "large_batch",
    ),
    (
        "Many heads M=16",
        1,
        50,
        16,
        8,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "random",
        "random",
        False,
        "many_heads",
    ),
    (
        "Single query Lq=1",
        2,
        1,
        4,
        16,
        2,
        4,
        [[8, 8], [4, 4]],
        "random",
        "random",
        "random",
        False,
        "single_query",
    ),
    # --- Attention edge cases ---
    (
        "Uniform attention",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "uniform",
        "random",
        False,
        "uniform_attention",
    ),
    (
        "Concentrated attention",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "concentrated",
        "random",
        False,
        "concentrated_attention",
    ),
    # --- Coordinate edge cases ---
    (
        "Boundary coordinates",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "random",
        "boundary",
        False,
        "boundary_coords",
    ),
    (
        "Center coordinates",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "random",
        "center",
        False,
        "center_coords",
    ),
    (
        "Corner coordinates",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "random",
        "random",
        "corner",
        False,
        "corner_coords",
    ),
    # --- Edge case: Pooling / Size ---
    (
        "Single element pool P=1",
        1,
        10,
        2,
        8,
        1,
        1,
        [[4, 4]],
        "random",
        "random",
        "random",
        False,
        "single_element_pooling",
    ),
    (
        "Minimum input size",
        1,
        1,
        1,
        4,
        1,
        1,
        [[2, 2]],
        "random",
        "random",
        "random",
        False,
        "minimum_input",
    ),
]


@pytest.mark.unit
def test_e2e_numerical():
    """Test end-to-end numerical accuracy: PyTorch vs TTSim."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        N,
        Lq,
        M,
        D,
        L,
        P,
        sp_list,
        value_mode,
        attn_mode,
        coord_mode,
        use_relaxed_tol,
        category,
    ) in enumerate(test_cases_e2e):

        spatial_shapes = np.array(sp_list, dtype=np.int64)

        (
            value_pt,
            spatial_pt,
            sampling_pt,
            attn_pt,
            value_np,
            spatial_np,
            sampling_np,
            attn_np,
        ) = _generate_inputs(
            N,
            Lq,
            M,
            D,
            L,
            P,
            sp_list,
            seed=SEED + tno,
            value_mode=value_mode,
            attention_mode=attn_mode,
            coord_mode=coord_mode,
        )

        # Select tolerance
        cur_rtol = LARGE_RTOL if use_relaxed_tol else RTOL
        cur_atol = LARGE_ATOL if use_relaxed_tol else ATOL

        # PyTorch
        pt_out = ms_deform_attn_core_pytorch(value_pt, spatial_pt, sampling_pt, attn_pt)
        pt_d = pt_out.detach().cpu().numpy()

        # TTSim — wrap ALL numpy inputs as SimTensors so the TTSim
        # function's isinstance(SimTensor) check fires first (plain
        # ndarray.data returns a memoryview which breaks arithmetic).
        tt_out = ms_deform_attn_core_ttsim(
            _wrap_sim(value_np, "value"),
            _wrap_sim(spatial_np, "spatial_shapes"),
            _wrap_sim(sampling_np, "sampling_locations"),
            _wrap_sim(attn_np, "attention_weights"),
        )
        tt_d = _to_numpy(tt_out)

        # Compare
        pt_shape = list(pt_d.shape)
        tt_shape = list(tt_d.shape)
        shape_ok = pt_shape == tt_shape

        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=cur_rtol, atol=cur_atol))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={cur_atol}"

        S = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))
        inp_desc = (
            f"value[{N},{S},{M},{D}] "
            f"sampling[{N},{Lq},{M},{L},{P},2] "
            f"attn[{N},{Lq},{M},{L},{P}]"
        )

        tol_note = " (relaxed)" if use_relaxed_tol else ""
        print_test_linear(
            module="E2E_Numerical",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=cur_rtol,
            atol=cur_atol,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
            input_samples={"value": value_np, "sampling": sampling_np, "attn": attn_np},
        )

        TEST_RESULTS.append(
            {
                "module": "E2E_Numerical",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "pt_shape": pt_shape,
                "tt_shape": tt_shape,
                "shape_ok": shape_ok,
                "num_ok": num_ok,
                "max_diff": mx,
                "mean_diff": mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, mx))
            FAILED_TESTS.append(
                {
                    "module": "E2E_Numerical",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}` → **Output Shape:** `{pt_shape}`{tol_note}\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- value:    `[{_fmt_samples(value_np)}]`\n"
            f"- sampling: `[{_fmt_samples(sampling_np)}]`\n"
            f"- attn:     `[{_fmt_samples(attn_np)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["E2E_Numerical"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_e2e),
        "num_passed": num_passed,
        "num_total": len(test_cases_e2e),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "E2E_Numerical",
            "description": "End-to-end numerical validation — full ms_deform_attn_core pipeline",
            "passed": passed,
            "total": len(test_cases_e2e),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_e2e
    ), f"E2E_Numerical: {passed}/{len(test_cases_e2e)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Per-Level Grid Sample
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_perlevel = [
    # (description, N, Lq, M, D, L, P, spatial_shapes, value_mode, edge_category)
    (
        "Standard 4 levels",
        2,
        100,
        8,
        32,
        4,
        4,
        [[50, 50], [25, 25], [13, 13], [7, 7]],
        "random",
        "standard_4_levels",
    ),
    ("Single level", 1, 50, 4, 16, 1, 2, [[32, 32]], "random", "single_level"),
    (
        "Two levels asymmetric",
        2,
        64,
        4,
        16,
        2,
        4,
        [[32, 32], [4, 4]],
        "random",
        "two_levels_asymmetric",
    ),
    (
        "Negative values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "negative",
        "negative_values",
    ),
    (
        "Small values (~1e-6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "small",
        "small_values",
    ),
    # --- Edge case: Values (mandatory) ---
    ("Zero values", 2, 50, 4, 16, 2, 4, [[16, 16], [8, 8]], "zeros", "zeros_values"),
    (
        "Mixed positive/negative",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "mixed",
        "mixed_values",
    ),
    (
        "Large values (~1e6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "large",
        "large_values",
    ),
    # --- Edge case: Pooling / Size ---
    (
        "Single element pool P=1",
        1,
        10,
        2,
        8,
        1,
        1,
        [[4, 4]],
        "random",
        "single_element_pooling",
    ),
    ("Minimum input size", 1, 1, 1, 4, 1, 1, [[2, 2]], "random", "minimum_input"),
]


@pytest.mark.unit
def test_per_level_grid_sample():
    """Test per-level grid_sample outputs: PyTorch vs TTSim."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, N, Lq, M, D, L, P, sp_list, value_mode, category) in enumerate(
        test_cases_perlevel
    ):
        spatial_shapes = np.array(sp_list, dtype=np.int64)

        # Relaxed tolerance for large-value tests
        use_relaxed_tol = category == "large_values"
        cur_rtol = LARGE_RTOL if use_relaxed_tol else RTOL
        cur_atol = LARGE_ATOL if use_relaxed_tol else ATOL

        (
            value_pt,
            spatial_pt,
            sampling_pt,
            attn_pt,
            value_np,
            spatial_np,
            sampling_np,
            attn_np,
        ) = _generate_inputs(
            N, Lq, M, D, L, P, sp_list, seed=SEED + 100 + tno, value_mode=value_mode
        )

        _, pt_inter = _pytorch_with_intermediates(
            value_pt, spatial_pt, sampling_pt, attn_pt
        )
        _, tt_inter = _ttsim_with_intermediates(
            value_np, spatial_np, sampling_np, attn_np
        )

        # Compare each level's grid_sample output
        all_levels_ok = True
        level_details = []
        worst_mx = 0.0
        worst_mn = 0.0

        for lid in range(L):
            key = f"level_{lid}_grid_sample"
            pt_val = pt_inter[key]
            tt_val = tt_inter[key]

            pt_s = list(pt_val.shape)
            tt_s = list(tt_val.shape)
            s_ok = pt_s == tt_s

            abs_d = np.abs(pt_val - tt_val)
            mx_l = float(abs_d.max())
            mn_l = float(abs_d.mean())
            n_ok = bool(np.allclose(pt_val, tt_val, rtol=cur_rtol, atol=cur_atol))
            l_ok = s_ok and n_ok

            if mx_l > worst_mx:
                worst_mx = mx_l
                worst_mn = mn_l

            if not l_ok:
                all_levels_ok = False

            level_details.append(
                {
                    "level": lid,
                    "shape": pt_s,
                    "max_diff": mx_l,
                    "mean_diff": mn_l,
                    "shape_ok": s_ok,
                    "num_ok": n_ok,
                    "passed": l_ok,
                }
            )

        ok = all_levels_ok
        shape_passed += int(ok)
        num_passed += int(ok)
        passed += int(ok)

        reason = ""
        if not ok:
            for ld in level_details:
                if not ld["passed"]:
                    reason = f"Level {ld['level']} max_diff={ld['max_diff']:.2e}"
                    break

        S = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))
        inp_desc = (
            f"value[{N},{S},{M},{D}] "
            f"sampling[{N},{Lq},{M},{L},{P},2] "
            f"L={L} levels"
        )

        # Use the final output for sample display
        pt_out_data = pt_inter["output"]
        tt_out_data = tt_inter["output"]

        tol_note = " (relaxed)" if use_relaxed_tol else ""
        print_test_linear(
            module="PerLevelGridSample",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"{L} levels checked — worst max_diff={worst_mx:.2e}",
            shape_ok=ok,
            is_numerical=True,
            num_ok=ok,
            max_diff=worst_mx,
            mean_diff=worst_mn,
            rtol=cur_rtol,
            atol=cur_atol,
            failure_reason=reason,
            pt_data=pt_out_data,
            tt_data=tt_out_data,
            input_samples={"value": value_np, "sampling": sampling_np, "attn": attn_np},
        )

        TEST_RESULTS.append(
            {
                "module": "PerLevelGridSample",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "pt_shape": "per-level",
                "tt_shape": "per-level",
                "shape_ok": ok,
                "num_ok": ok,
                "max_diff": worst_mx,
                "mean_diff": worst_mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, worst_mx))
            FAILED_TESTS.append(
                {
                    "module": "PerLevelGridSample",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": worst_mx,
                    "atol": cur_atol,
                }
            )

        # Build per-level detail for markdown
        level_rows = []
        for ld in level_details:
            ltag = "✅" if ld["passed"] else "🔴"
            level_rows.append(
                f"  - Level {ld['level']}: shape=`{ld['shape']}`, "
                f"max_diff={ld['max_diff']:.2e}, mean_diff={ld['mean_diff']:.2e} {ltag}"
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` "
            f"| {worst_mx:.2e} | {worst_mn:.2e} | {tag} |"
        )
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}`{tol_note}\n\n"
            f"**Per-Level Results:**\n" + "\n".join(level_rows) + "\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- value:    `[{_fmt_samples(value_np)}]`\n"
            f"- sampling: `[{_fmt_samples(sampling_np)}]`\n"
            f"- attn:     `[{_fmt_samples(attn_np)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_out_data)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_out_data)}]`\n\n"
        )

    MODULE_STATS["PerLevelGridSample"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_perlevel),
        "num_passed": num_passed,
        "num_total": len(test_cases_perlevel),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Worst Max Diff | Worst Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:---------------|:----------------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "PerLevelGridSample",
            "description": "Per-level grid_sample output — isolates interpolation accuracy per feature level",
            "passed": passed,
            "total": len(test_cases_perlevel),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_perlevel
    ), f"PerLevelGridSample: {passed}/{len(test_cases_perlevel)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Intermediate Steps
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_intermediate = [
    # (desc, N, Lq, M, D, L, P, spatial_shapes, value_mode,
    #  use_relaxed_tol, edge_category)
    (
        "Standard config",
        2,
        100,
        8,
        32,
        4,
        4,
        [[50, 50], [25, 25], [13, 13], [7, 7]],
        "random",
        False,
        "standard_config",
    ),
    (
        "Small config",
        1,
        10,
        2,
        8,
        2,
        2,
        [[4, 4], [2, 2]],
        "random",
        False,
        "small_config",
    ),
    (
        "Negative values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "negative",
        False,
        "negative_values",
    ),
    (
        "Large values (~1e6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "large",
        True,
        "large_values",
    ),
    # --- Edge case: Values (mandatory) ---
    (
        "Zero values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "zeros",
        False,
        "zeros_values",
    ),
    (
        "Mixed values",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "mixed",
        False,
        "mixed_values",
    ),
    (
        "Small values (~1e-6)",
        2,
        50,
        4,
        16,
        2,
        4,
        [[16, 16], [8, 8]],
        "small",
        False,
        "small_values",
    ),
    # --- Edge case: Pooling / Size ---
    (
        "Single element pool P=1",
        1,
        10,
        2,
        8,
        1,
        1,
        [[4, 4]],
        "random",
        False,
        "single_element_pooling",
    ),
    (
        "Minimum input size",
        1,
        1,
        1,
        4,
        1,
        1,
        [[2, 2]],
        "random",
        False,
        "minimum_input",
    ),
]


@pytest.mark.unit
def test_intermediate_steps():
    """Test step-by-step intermediate outputs: PyTorch vs TTSim."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    step_keys = [
        ("sampling_grids", "Sampling Grids [0,1]→[-1,1]"),
        ("stacked", "Stacked Grid Sample Outputs"),
        ("flattened", "Flattened Stacked"),
        ("attention_reshaped", "Attention Reshaped"),
        ("weighted", "Weighted Sum"),
        ("output", "Final Output"),
    ]

    for tno, (
        tmsg,
        N,
        Lq,
        M,
        D,
        L,
        P,
        sp_list,
        value_mode,
        use_relaxed_tol,
        category,
    ) in enumerate(test_cases_intermediate):
        spatial_shapes = np.array(sp_list, dtype=np.int64)

        cur_rtol = LARGE_RTOL if use_relaxed_tol else RTOL
        cur_atol = LARGE_ATOL if use_relaxed_tol else ATOL

        (
            value_pt,
            spatial_pt,
            sampling_pt,
            attn_pt,
            value_np,
            spatial_np,
            sampling_np,
            attn_np,
        ) = _generate_inputs(
            N, Lq, M, D, L, P, sp_list, seed=SEED + 200 + tno, value_mode=value_mode
        )

        _, pt_inter = _pytorch_with_intermediates(
            value_pt, spatial_pt, sampling_pt, attn_pt
        )
        _, tt_inter = _ttsim_with_intermediates(
            value_np, spatial_np, sampling_np, attn_np
        )

        all_steps_ok = True
        step_details = []
        worst_mx = 0.0
        worst_mn = 0.0

        for key, desc in step_keys:
            if key not in pt_inter or key not in tt_inter:
                continue
            pt_val = pt_inter[key]
            tt_val = tt_inter[key]

            pt_s = list(pt_val.shape)
            tt_s = list(tt_val.shape)
            s_ok = pt_s == tt_s

            abs_d = np.abs(pt_val - tt_val)
            mx_s = float(abs_d.max())
            mn_s = float(abs_d.mean())
            n_ok = bool(np.allclose(pt_val, tt_val, rtol=cur_rtol, atol=cur_atol))
            s_passed = s_ok and n_ok

            if mx_s > worst_mx:
                worst_mx = mx_s
                worst_mn = mn_s

            if not s_passed:
                all_steps_ok = False

            step_details.append(
                {
                    "key": key,
                    "desc": desc,
                    "shape": pt_s,
                    "max_diff": mx_s,
                    "mean_diff": mn_s,
                    "shape_ok": s_ok,
                    "num_ok": n_ok,
                    "passed": s_passed,
                }
            )

        ok = all_steps_ok
        shape_passed += int(ok)
        num_passed += int(ok)
        passed += int(ok)

        reason = ""
        if not ok:
            for sd in step_details:
                if not sd["passed"]:
                    reason = f"{sd['desc']} max_diff={sd['max_diff']:.2e}"
                    break

        S = int(np.sum(spatial_shapes[:, 0] * spatial_shapes[:, 1]))
        inp_desc = (
            f"value[{N},{S},{M},{D}] "
            f"sampling[{N},{Lq},{M},{L},{P},2] "
            f"{len(step_keys)} steps"
        )

        pt_final = pt_inter["output"]
        tt_final = tt_inter["output"]

        tol_note = " (relaxed)" if use_relaxed_tol else ""
        print_test_linear(
            module="IntermediateSteps",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"{len(step_keys)} steps checked — worst max_diff={worst_mx:.2e}",
            shape_ok=ok,
            is_numerical=True,
            num_ok=ok,
            max_diff=worst_mx,
            mean_diff=worst_mn,
            rtol=cur_rtol,
            atol=cur_atol,
            failure_reason=reason,
            pt_data=pt_final,
            tt_data=tt_final,
            input_samples={"value": value_np, "sampling": sampling_np, "attn": attn_np},
        )

        TEST_RESULTS.append(
            {
                "module": "IntermediateSteps",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "pt_shape": "step-by-step",
                "tt_shape": "step-by-step",
                "shape_ok": ok,
                "num_ok": ok,
                "max_diff": worst_mx,
                "mean_diff": worst_mn,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, worst_mx))
            FAILED_TESTS.append(
                {
                    "module": "IntermediateSteps",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": worst_mx,
                    "atol": ATOL,
                }
            )

        # Build per-step detail for markdown
        step_rows = []
        for sd in step_details:
            stag = "✅" if sd["passed"] else "🔴"
            step_rows.append(
                f"  - {sd['desc']}: shape=`{sd['shape']}`, "
                f"max_diff={sd['max_diff']:.2e}, mean_diff={sd['mean_diff']:.2e} {stag}"
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` "
            f"| {worst_mx:.2e} | {worst_mn:.2e} | {tag} |"
        )
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}`{tol_note}\n\n"
            f"**Step-by-Step Results:**\n" + "\n".join(step_rows) + "\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- value:    `[{_fmt_samples(value_np)}]`\n"
            f"- sampling: `[{_fmt_samples(sampling_np)}]`\n"
            f"- attn:     `[{_fmt_samples(attn_np)}]`\n\n"
            f"**Output Float Samples [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_final)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_final)}]`\n\n"
        )

    MODULE_STATS["IntermediateSteps"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_intermediate),
        "num_passed": num_passed,
        "num_total": len(test_cases_intermediate),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Worst Max Diff | Worst Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:---------------|:----------------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "IntermediateSteps",
            "description": "Step-by-step intermediate comparison — pin-points first diverging step",
            "passed": passed,
            "total": len(test_cases_intermediate),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_intermediate
    ), f"IntermediateSteps: {passed}/{len(test_cases_intermediate)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Markdown report + self-runner
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a structured markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    # If some modules crashed, they won't be in REPORT_SECTIONS
    missing = [
        m for m in EXPECTED_MODULES if m not in [s["name"] for s in REPORT_SECTIONS]
    ]
    has_crashes = len(missing) > 0

    status = (
        "PASS"
        if (total_passed == total_tests and exit_code == 0 and not has_crashes)
        else "FAIL"
    )

    lines = [
        "# MS Deform Attn Core Unit Test Report",
        f"**PyTorch vs TTSim Comparison** | **{total_passed}/{total_tests} passed** | {status}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exit Code: {exit_code}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Module | Passed | Total | Status |")
    lines.append("|--------|--------|-------|--------|")
    for s in REPORT_SECTIONS:
        ms = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"| {s['name']} | {s['passed']} | {s['total']} | {ms} |")
    for m in missing:
        lines.append(f"| {m} | --- | --- | CRASHED |")
    lines.append("")
    lines.append(f"**Total: {total_passed}/{total_tests} tests passed**")
    if has_crashes:
        lines.append(
            f"\n**WARNING: {len(missing)} module(s) crashed before "
            f"recording results: {', '.join(missing)}**"
        )
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
            diff = f"{ft['max_diff']:.2e}" if ft.get("max_diff") else "N/A"
            lines.append(
                f"| {ft['module']} | {ft['test']} | {ft['edge_case']} | {diff} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Per-module details
    for s in REPORT_SECTIONS:
        ms = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"## {s['name']} ({s['passed']}/{s['total']} {ms})")
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

        if s.get("details"):
            lines.append(s["details"])

        lines.append("---")
        lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append(f"- Tolerance: rtol={RTOL}, atol={ATOL}")
    lines.append(
        f"- Relaxed Tolerance (large values): rtol={LARGE_RTOL}, atol={LARGE_ATOL}"
    )
    lines.append(f"- Random Seed: {SEED}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sync_globals_from_pytest():
    """Sync global collectors from the pytest-imported copy back to __main__."""
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
    report_path = os.path.join(report_dir, "ms_deform_attn_unit_test_report.md")
    output_path = os.path.join(report_dir, "ms_deform_attn_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    print(f"\n{SUMMARY_LINE}")
    print(f"MS DEFORM ATTN CORE UNIT TEST SUITE - PyTorch vs TTSim")
    print(f"{SUMMARY_LINE}\n")

    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    _sync_globals_from_pytest()

    # Pass exit_code so summary can detect crashed modules
    print_summary(exit_code=exit_code)

    sys.stdout = _original_stdout
    _tee_file.close()

    _write_markdown_report(report_path, exit_code)

    print(f"\n{Colors.cyan(f'[Markdown report : {report_path}]')}")
    print(f"{Colors.cyan(f'[Full output log  : {output_path}]')}\n")
    sys.exit(exit_code)
