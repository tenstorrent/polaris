#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for backbone modules — PyTorch vs TTSim comparison WITH EDGE CASES.

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
    - WHY USE: When Conv2d in TTSim uses slow nested-loop numpy implementation
              (Full backbone would take hours for numerical validation)

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: FrozenBatchNorm2d — NUMERICAL VALIDATION
    Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    Edge Cases: positive, negative, zeros, mixed, small (~1e-6), large (~1e6)
    WHY NUMERICAL: Simple elementwise operations, fast to compute

MODULE 2: ResNetBottleneck — NUMERICAL VALIDATION (pretrained weights)
    Architecture: Conv1x1→BN→ReLU → Conv3x3→BN→ReLU → Conv1x1→BN + Shortcut
    Edge Cases: positive, negative, zeros, mixed, small, large, minimum_input (4x4)
    Note: Large values use relaxed tolerance (rtol=1e-3, atol=1e-2)
    WHY NUMERICAL: 8x8 input is small enough for full numerical validation

MODULE 3: Backbone (ResNet50) — SHAPE VALIDATION ONLY
    Output: Feature maps from layer1, layer2, layer3, layer4
    Edge Cases: baseline, scale (64x64), batch, minimum_input (16x16)
    WHY SHAPE: Conv2d in TTSim uses slow nested-loop numpy; numerical would take hours
    Block-level numerical validation is done in MODULE 2

MODULE 4: Joiner (Backbone + Position Encoding) — SHAPE VALIDATION ONLY
    Output: Multi-scale feature maps + positional encodings
    Edge Cases: baseline, scale (64x64), batch, minimum_input (16x16)
    WHY SHAPE: Same reason as Backbone

MODULE 5: Positional Encoding — NUMERICAL VALIDATION
    Formula: PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d))
    Edge Cases: baseline, scale (64x64), batch, minimum_input (8x8)
    WHY NUMERICAL: Pure numpy math operations (sin/cos), no Conv2d involved

================================================================================
EDGE CASES TESTED (MANDATORY — all numerical modules):
================================================================================

'positive'       — Standard positive values (1.0 - 2.0) - baseline test
'negative'       — All negative values (-2.0 to -1.0) - tests sign handling
'zeros'          — All zeros - tests division by variance edge case
'mixed'          — Mix of positive/negative values - tests real-world distribution
'small'          — Very small values (~1e-6) - tests numerical precision near zero
'large'          — Very large values (~1e6) - tests numerical overflow handling
'minimum_input'  — Smallest valid input size - tests degenerate/boundary case

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_backbone_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_backbone_unit.py
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
import torchvision
from workloads.Deformable_DETR.reference.backbone import (
    FrozenBatchNorm2d as FrozenBatchNorm2dPyTorch,
    Backbone as BackbonePyTorch,
    Joiner as JoinerPyTorch,
)

# TTSim implementations
from workloads.Deformable_DETR.models.backbone_ttsim import (
    FrozenBatchNorm2d as FrozenBatchNorm2dTTSim,
    ResNetBottleneck as ResNetBottleneckTTSim,
    Backbone as BackboneTTSim,
    Joiner as JoinerTTSim,
    Sequential,
)

# Utilities
from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPyTorch
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTTSim
from workloads.Deformable_DETR.reference.position_encoding import (
    build_position_encoding as build_position_encoding_pytorch,
)
from workloads.Deformable_DETR.models.position_encoding_ttsim import (
    build_position_encoding as build_position_encoding_ttsim,
)
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.op as F

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
    "zeros": "All zeros - tests division by variance edge case",
    "mixed": "Mix of positive/negative values - tests real-world distribution",
    "small": "Very small values (~1e-6) - tests numerical precision near zero",
    "large": "Very large values (~1e6) - tests numerical overflow handling",
    "minimum_input": "Smallest valid input size - degenerate/boundary case",
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
def get_max_test_msg_len(TL):
    return max(len(x[0]) for x in TL)


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


def _create_dummy_args():
    class Args:
        hidden_dim = 256
        position_embedding = "sine"
        lr_backbone = 0.0
        masks = False
        num_feature_levels = 4
        backbone = "resnet50"
        dilation = False

    return Args()


def _transfer_bn(pt_bn, tt_bn):
    """Copy FrozenBatchNorm2d parameters from PyTorch to TTSim."""
    tt_bn.set_parameters(
        weight=pt_bn.weight.detach().cpu().numpy(),
        bias=pt_bn.bias.detach().cpu().numpy(),
        running_mean=pt_bn.running_mean.detach().cpu().numpy(),
        running_var=pt_bn.running_var.detach().cpu().numpy(),
    )


def _transfer_conv(pt_conv, tt_conv):
    """Copy Conv2d weights from PyTorch to TTSim SimOpHandle."""
    if hasattr(pt_conv, "weight"):
        tt_conv.params[0][1].data = pt_conv.weight.detach().cpu().numpy().copy()
        if (
            hasattr(pt_conv, "bias")
            and pt_conv.bias is not None
            and len(tt_conv.params) > 1
        ):
            tt_conv.params[1][1].data = pt_conv.bias.detach().cpu().numpy().copy()


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
    return ", ".join(f"{v:.6f}" for v in arr.flat[:n])


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
    print(f"{'MODULE':<28}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

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
        print(f"{name:<28}{shape_str:<12}{num_str:<12}{status}")

    print(DIVIDER_LINE)

    total_num_str = f"{total_np}/{total_nt}" if total_nt > 0 else "N/A"
    overall = Colors.success("✓ PASS") if all_passed else Colors.fail("✗ FAIL")
    print(f"{'TOTAL':<28}{total_sp}/{total_st:<11} {total_num_str:<12}{overall}")

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
#  TEST 1 — FrozenBatchNorm2d (multiple test cases, avgpool2d style)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_bn = "test_frozen_batchnorm2d"
test_cases_bn = [
    # (description, batch, channels, height, width, data_type, category)
    # --- Baseline tests ---
    ("Basic 64ch 8x8", 1, 64, 8, 8, "positive", "baseline"),
    ("Multi-batch 64ch", 2, 64, 8, 8, "positive", "baseline"),
    ("128 channels", 1, 128, 8, 8, "positive", "baseline"),
    ("Non-square 8x16", 1, 64, 8, 16, "positive", "shape"),
    # --- Edge case: Values ---
    ("Negative values", 1, 64, 8, 8, "negative", "edge_value"),
    ("Zero values", 1, 64, 8, 8, "zeros", "edge_value"),
    ("Mixed positive/negative", 1, 64, 8, 8, "mixed", "edge_value"),
    ("Very small values (1e-6)", 1, 64, 8, 8, "small", "edge_value"),
    ("Very large values (1e6)", 1, 64, 8, 8, "large", "edge_value"),
    # --- Edge case: Shapes ---
    ("Minimum spatial 1x1", 1, 64, 1, 1, "positive", "edge_shape"),
    ("Single channel", 1, 1, 8, 8, "positive", "edge_shape"),
    ("Large spatial 32x32", 1, 64, 32, 32, "positive", "edge_shape"),
]


@pytest.mark.unit
def test_frozen_batchnorm2d():
    """Test FrozenBatchNorm2d: shape + numerical validation across data types."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, data_type, category) in enumerate(test_cases_bn):
        shape = [batch, ch, h, w]
        test_data = generate_test_data(shape, data_type)

        # PyTorch
        x_torch = torch.from_numpy(test_data)
        bn_pt = FrozenBatchNorm2dPyTorch(ch)
        bn_pt.eval()

        # TTSim
        x_ttsim = torch_to_simtensor(x_torch, "input")
        bn_tt = FrozenBatchNorm2dTTSim("bn_test", ch)
        _transfer_bn(bn_pt, bn_tt)

        # Forward
        with torch.no_grad():
            out_pt = bn_pt(x_torch)
        out_tt = bn_tt(x_ttsim)

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

        # Print linear output
        print_test_linear(
            module="FrozenBatchNorm2d",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
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

        # Capture detailed results for report
        TEST_RESULTS.append(
            {
                "module": "FrozenBatchNorm2d",
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
                    "module": "FrozenBatchNorm2d",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        # Report table row
        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        # Report detail block
        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'N/A')}\n\n"
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

    # Module stats for summary
    MODULE_STATS["FrozenBatchNorm2d"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_bn),
        "num_passed": num_passed,
        "num_total": len(test_cases_bn),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "FrozenBatchNorm2d",
            "description": "Frozen Batch Normalization layer - normalizes with fixed running statistics",
            "passed": passed,
            "total": len(test_cases_bn),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_bn
    ), f"FrozenBatchNorm2d: {passed}/{len(test_cases_bn)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — ResNetBottleneck (pretrained weights, 8x8)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_bottleneck = "test_resnet_bottleneck"
test_cases_bottleneck = [
    # (description, batch, in_ch, out_ch, height, width, stride, data_type, category)
    # --- Baseline tests ---
    ("layer1[0] 64->256 stride=1", 1, 64, 256, 8, 8, 1, "positive", "baseline"),
    ("layer1[0] batch=2", 2, 64, 256, 8, 8, 1, "positive", "baseline"),
    # --- Edge case: Values (mandatory) ---
    ("layer1[0] negative input", 1, 64, 256, 8, 8, 1, "negative", "edge_value"),
    ("layer1[0] zero values", 1, 64, 256, 8, 8, 1, "zeros", "edge_value"),
    ("layer1[0] mixed input", 1, 64, 256, 8, 8, 1, "mixed", "edge_value"),
    ("layer1[0] small values", 1, 64, 256, 8, 8, 1, "small", "edge_value"),
    (
        "layer1[0] large values",
        1,
        64,
        256,
        8,
        8,
        1,
        "large",
        "edge_value",
    ),  # relaxed tolerance
    # --- Edge case: Minimum input size ---
    ("layer1[0] minimum 4x4", 1, 64, 256, 4, 4, 1, "positive", "edge_shape"),
]


@pytest.mark.unit
def test_resnet_bottleneck():
    """Test ResNetBottleneck: shape + numerical (pretrained weights, 8x8 input)."""
    # Load pretrained ResNet50 once
    resnet = torchvision.models.resnet50(
        pretrained=True, norm_layer=FrozenBatchNorm2dPyTorch
    )
    resnet.eval()
    block_pt = resnet.layer1[0]

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (
        tmsg,
        batch,
        in_ch,
        out_ch,
        h,
        w,
        stride,
        data_type,
        category,
    ) in enumerate(test_cases_bottleneck):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        shape = [batch, in_ch, h, w]
        test_data = generate_test_data(shape, data_type)

        # Determine tolerance
        rtol = 1e-3 if data_type == "large" else RTOL
        atol = 1e-2 if data_type == "large" else ATOL

        x_torch = torch.from_numpy(test_data)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        # TTSim — build matching bottleneck
        downsample = Sequential(
            "downsample",
            [
                F.Conv2d(
                    "downsample.0",
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                FrozenBatchNorm2dTTSim("downsample.1", out_ch),
            ],
        )
        block_tt = ResNetBottleneckTTSim(
            "bottleneck_test",
            in_channels=in_ch,
            out_channels=out_ch,
            stride=stride,
            downsample=downsample,
        )

        # Transfer weights
        _transfer_conv(block_pt.conv1, block_tt.conv1)
        _transfer_conv(block_pt.conv2, block_tt.conv2)
        _transfer_conv(block_pt.conv3, block_tt.conv3)
        _transfer_bn(block_pt.bn1, block_tt.bn1)
        _transfer_bn(block_pt.bn2, block_tt.bn2)
        _transfer_bn(block_pt.bn3, block_tt.bn3)
        if block_pt.downsample is not None:
            ds = block_tt.downsample.modules_list
            _transfer_conv(block_pt.downsample[0], ds[0])
            _transfer_bn(block_pt.downsample[1], ds[1])

        # Forward
        with torch.no_grad():
            out_pt = block_pt(x_torch)
        out_tt = block_tt(x_ttsim)

        # Validation
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

        # Print linear output
        print_test_linear(
            module="ResNetBottleneck",
            edge_case=data_type,
            edge_desc=EDGE_CASE_DESC.get(data_type, "Standard test"),
            input_shape=shape,
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

        # Capture detailed results for report
        TEST_RESULTS.append(
            {
                "module": "ResNetBottleneck",
                "validation_type": "NUMERICAL",
                "edge_case": data_type,
                "edge_desc": EDGE_CASE_DESC.get(data_type, "Standard test"),
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
                    "module": "ResNetBottleneck",
                    "test": tmsg,
                    "edge_case": data_type,
                    "max_diff": mx,
                    "atol": atol,
                }
            )

        # Report
        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        row_style = "" if ok else "**"
        tol_info = (
            f" *(relaxed: rtol={rtol}, atol={atol})*" if data_type == "large" else ""
        )
        rows.append(
            f"| {row_style}{tno}{row_style} | {row_style}{tmsg}{row_style} | {data_type} | `{shape}` | `{pt_shape}` "
            f"| {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{data_type}` — {EDGE_CASE_DESC.get(data_type, 'Standard test')}{tol_info}\n\n"
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

    # Module stats for summary
    MODULE_STATS["ResNetBottleneck"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_bottleneck),
        "num_passed": num_passed,
        "num_total": len(test_cases_bottleneck),
    }

    hdr = (
        "| # | Test Case | Data Type | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "ResNetBottleneck",
            "description": "ResNet50 Bottleneck block with pretrained ImageNet weights",
            "passed": passed,
            "total": len(test_cases_bottleneck),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_bottleneck
    ), f"ResNetBottleneck: {passed}/{len(test_cases_bottleneck)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Backbone (shape + mask only, 32x32)
# ═══════════════════════════════════════════════════════════════════════════════

test_name_backbone = "test_backbone_shapes"
test_cases_backbone = [
    # (description, batch, channels, height, width, category)
    # --- Baseline tests ---
    ("ResNet50 32x32 batch=1", 1, 3, 32, 32, "baseline"),
    ("ResNet50 64x64 batch=1", 1, 3, 64, 64, "scale"),
    ("ResNet50 32x32 batch=2", 2, 3, 32, 32, "batch"),
    # --- Edge case: Minimum input size ---
    ("ResNet50 16x16 minimum", 1, 3, 16, 16, "minimum_input"),
]


@pytest.mark.unit
def test_backbone_shapes():
    """Test Backbone (ResNet50): shape + mask validation for all output layers."""
    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, category) in enumerate(test_cases_backbone):
        _seed()

        x_torch = torch.randn(batch, ch, h, w)
        mask = torch.zeros(batch, h, w, dtype=torch.bool)

        nt_pt = NestedTensorPyTorch(x_torch, mask)
        nt_tt = NestedTensorTTSim(torch_to_simtensor(x_torch, "input"), mask.numpy())

        bb_pt = BackbonePyTorch(
            name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        bb_pt.eval()
        bb_tt = BackboneTTSim(
            name="backbone_test",
            resnet_name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )

        with torch.no_grad():
            out_pt = bb_pt(nt_pt)
        out_tt = bb_tt(nt_tt)

        pt_keys = sorted(out_pt.keys())
        tt_keys = sorted(out_tt.keys())
        keys_match = pt_keys == tt_keys

        all_shapes_match = True
        all_masks_match = True
        layer_details = []
        layer_rows = []
        matched_layers = 0

        for key in pt_keys:
            pt_ts = list(out_pt[key].tensors.shape)
            tt_ts = list(out_tt[key].tensors.shape)
            shape_ok = pt_ts == tt_ts

            pt_ms = (
                list(out_pt[key].mask.shape) if out_pt[key].mask is not None else None
            )
            tt_ms = (
                list(out_tt[key].mask.shape) if out_tt[key].mask is not None else None
            )
            mask_ok = pt_ms == tt_ms

            all_shapes_match = all_shapes_match and shape_ok
            all_masks_match = all_masks_match and mask_ok
            if shape_ok and mask_ok:
                matched_layers += 1
            layer_details.append((key, pt_ts, tt_ts, shape_ok, pt_ms, tt_ms, mask_ok))

            s_tag = "✅" if (shape_ok and mask_ok) else "❌"
            layer_rows.append(
                f"| {key} | `{pt_ts}` | `{tt_ts}` | `{pt_ms}` | `{tt_ms}` | {s_tag} |"
            )

        overall = keys_match and all_shapes_match and all_masks_match
        passed += int(overall)

        reason = ""
        if not keys_match:
            reason = f"Layer key mismatch: PT={pt_keys} vs TT={tt_keys}"
        elif not all_shapes_match:
            reason = "One or more tensor shapes do not match"
        elif not all_masks_match:
            reason = "One or more mask shapes do not match"

        # Print linear output
        print_test_linear(
            module="Backbone",
            edge_case=category,
            edge_desc=tmsg,
            input_shape=[batch, ch, h, w],
            shape_line=f"{matched_layers}/{len(pt_keys)} layers match",
            shape_ok=overall,
            is_numerical=False,
            failure_reason=reason,
        )

        # Capture detailed results for report
        TEST_RESULTS.append(
            {
                "module": "Backbone (ResNet50)",
                "validation_type": "SHAPE ONLY",
                "edge_case": "random",
                "edge_desc": "Shape validation with random input",
                "input_shape": [batch, ch, h, w],
                "pt_shape": f"{len(pt_keys)} layers",
                "tt_shape": f"{len(tt_keys)} layers",
                "shape_ok": overall,
                "num_ok": None,
                "max_diff": None,
                "mean_diff": None,
                "pt_stats": None,
                "tt_stats": None,
                "passed": overall,
                "layer_details": layer_details,
            }
        )

        if not overall:
            failed_cases.append((tno, tmsg, "shape_mismatch", 0))
            FAILED_TESTS.append(
                {
                    "module": "Backbone",
                    "test": tmsg,
                    "edge_case": "shape_mismatch",
                    "max_diff": 0,
                }
            )

        tag = "✅ PASS" if overall else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | `[{batch},{ch},{h},{w}]` | {len(pt_keys)} layers | {tag} |"
        )

        lyr_hdr = (
            "| Layer | PT Tensor | TT Tensor | PT Mask | TT Mask | Match |\n"
            "|:------|:----------|:----------|:--------|:--------|:------|"
        )
        status_badge = "🟢" if overall else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Input Shape:** `[{batch},{ch},{h},{w}]`\n\n"
            f"**Layer Output Shapes:**\n\n"
            + lyr_hdr
            + "\n"
            + "\n".join(layer_rows)
            + "\n\n"
        )

    # Module stats for summary
    MODULE_STATS["Backbone"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_backbone),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Input | Layers | Result |\n"
        "|:--|:----------|:------|:-------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Backbone Shapes",
            "description": "ResNet50 backbone shape validation for all feature map layers",
            "passed": passed,
            "total": len(test_cases_backbone),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_backbone
    ), f"Backbone Shapes: {passed}/{len(test_cases_backbone)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Joiner feature-map shapes + mask shapes
# ═══════════════════════════════════════════════════════════════════════════════

test_name_joiner = "test_joiner_shapes"
test_cases_joiner = [
    # (description, batch, channels, height, width, category)
    # --- Baseline tests ---
    ("Joiner 32x32 batch=1", 1, 3, 32, 32, "baseline"),
    ("Joiner 64x64 batch=1", 1, 3, 64, 64, "scale"),
    ("Joiner 32x32 batch=2", 2, 3, 32, 32, "batch"),
    # --- Edge case: Minimum input size ---
    ("Joiner 16x16 minimum", 1, 3, 16, 16, "minimum_input"),
]


@pytest.mark.unit
def test_joiner_shapes():
    """Test Joiner: feature-map shape + mask validation."""
    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (tmsg, batch, ch, h, w, category) in enumerate(test_cases_joiner):
        _seed()
        args = _create_dummy_args()

        x_torch = torch.randn(batch, ch, h, w)
        mask = torch.zeros(batch, h, w, dtype=torch.bool)

        nt_pt = NestedTensorPyTorch(x_torch, mask)
        nt_tt = NestedTensorTTSim(torch_to_simtensor(x_torch, "input"), mask.numpy())

        bb_pt = BackbonePyTorch(
            name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        pos_pt = build_position_encoding_pytorch(args)
        joiner_pt = JoinerPyTorch(bb_pt, pos_pt)
        joiner_pt.eval()

        bb_tt = BackboneTTSim(
            name="backbone_test",
            resnet_name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        pos_tt = build_position_encoding_ttsim(args)
        joiner_tt = JoinerTTSim("joiner_test", bb_tt, pos_tt)

        with torch.no_grad():
            feats_pt, pos_pt_out = joiner_pt(nt_pt)
        feats_tt, pos_tt_out = joiner_tt(nt_tt)

        count_match = (len(feats_pt) == len(feats_tt)) and (
            len(pos_pt_out) == len(pos_tt_out)
        )
        all_ok = count_match
        matched_feats = 0
        matched_pos = 0

        feat_rows = []
        for i in range(min(len(feats_pt), len(feats_tt))):
            pt_s = list(feats_pt[i].tensors.shape)
            tt_s = list(feats_tt[i].tensors.shape)
            s_ok = pt_s == tt_s
            pt_m = (
                list(feats_pt[i].mask.shape) if feats_pt[i].mask is not None else None
            )
            tt_m = (
                list(feats_tt[i].mask.shape) if feats_tt[i].mask is not None else None
            )
            m_ok = pt_m == tt_m
            all_ok = all_ok and s_ok and m_ok
            if s_ok and m_ok:
                matched_feats += 1

            f_tag = "✅" if (s_ok and m_ok) else "❌"
            feat_rows.append(
                f"| feat[{i}] | `{pt_s}` | `{tt_s}` | `{pt_m}` | `{tt_m}` | {f_tag} |"
            )

        pos_rows_detail = []
        for i in range(min(len(pos_pt_out), len(pos_tt_out))):
            pt_ps = list(pos_pt_out[i].shape)
            tt_ps = list(pos_tt_out[i].shape)
            ps_ok = pt_ps == tt_ps
            all_ok = all_ok and ps_ok
            if ps_ok:
                matched_pos += 1

            p_tag = "✅" if ps_ok else "❌"
            pos_rows_detail.append(
                f"| pos[{i}] | `{pt_ps}` | `{tt_ps}` | — | — | {p_tag} |"
            )

        passed += int(all_ok)

        reason = ""
        if not count_match:
            reason = f"Count mismatch: feats PT={len(feats_pt)} vs TT={len(feats_tt)}"
        elif not all_ok:
            reason = "One or more shape mismatches detected"

        total_feats = min(len(feats_pt), len(feats_tt))
        total_pos = min(len(pos_pt_out), len(pos_tt_out))

        # Print linear output
        print_test_linear(
            module="Joiner",
            edge_case=category,
            edge_desc=tmsg,
            input_shape=[batch, ch, h, w],
            shape_line=f"{matched_feats}/{total_feats} feats + {matched_pos}/{total_pos} pos",
            shape_ok=all_ok,
            is_numerical=False,
            failure_reason=reason,
        )

        # Capture detailed results for report
        TEST_RESULTS.append(
            {
                "module": "Joiner (Backbone + Position Encoding)",
                "validation_type": "SHAPE ONLY",
                "edge_case": "random",
                "edge_desc": "Feature map + position encoding shape validation",
                "input_shape": [batch, ch, h, w],
                "pt_shape": f"{len(feats_pt)} feats, {len(pos_pt_out)} pos",
                "tt_shape": f"{len(feats_tt)} feats, {len(pos_tt_out)} pos",
                "shape_ok": all_ok,
                "num_ok": None,
                "max_diff": None,
                "mean_diff": None,
                "pt_stats": None,
                "tt_stats": None,
                "passed": all_ok,
                "feat_rows": feat_rows,
                "pos_rows": pos_rows_detail,
            }
        )

        if not all_ok:
            failed_cases.append((tno, tmsg, "shape_mismatch", 0))
            FAILED_TESTS.append(
                {
                    "module": "Joiner",
                    "test": tmsg,
                    "edge_case": "shape_mismatch",
                    "max_diff": 0,
                }
            )

        tag = "✅ PASS" if all_ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | `[{batch},{ch},{h},{w}]` | {len(feats_pt)} feats | {tag} |"
        )

        tbl_hdr = (
            "| Item | PT Shape | TT Shape | PT Mask | TT Mask | Match |\n"
            "|:-----|:---------|:---------|:--------|:--------|:------|"
        )
        status_badge = "🟢" if all_ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Input Shape:** `[{batch},{ch},{h},{w}]`\n\n"
            f"**Feature Maps & Position Encodings:**\n\n"
            + tbl_hdr
            + "\n"
            + "\n".join(feat_rows + pos_rows_detail)
            + "\n\n"
        )

    # Module stats for summary
    MODULE_STATS["Joiner"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_joiner),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Input | Features | Result |\n"
        "|:--|:----------|:------|:---------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Joiner Shapes",
            "description": "Joiner module shape validation for feature maps and position encodings",
            "passed": passed,
            "total": len(test_cases_joiner),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_joiner
    ), f"Joiner Shapes: {passed}/{len(test_cases_joiner)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Joiner positional encoding numerical
# ═══════════════════════════════════════════════════════════════════════════════

test_name_pos = "test_joiner_pos_enc_numerical"
test_cases_pos = [
    # (description, batch, height, width, category)
    # --- Baseline tests ---
    ("Pos-enc 32x32 batch=1", 1, 32, 32, "baseline"),
    ("Pos-enc 64x64 batch=1", 1, 64, 64, "scale"),
    ("Pos-enc 32x32 batch=2", 2, 32, 32, "batch"),
    # --- Edge case: Minimum input size ---
    ("Pos-enc 8x8 minimum", 1, 8, 8, "minimum_input"),
]


@pytest.mark.unit
def test_joiner_pos_enc_numerical():
    """Test Joiner: full numerical validation for sine positional encoding."""
    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, batch, h, w, category) in enumerate(test_cases_pos):
        _seed()
        args = _create_dummy_args()
        ch = 3

        x_torch = torch.randn(batch, ch, h, w)
        mask = torch.zeros(batch, h, w, dtype=torch.bool)

        nt_pt = NestedTensorPyTorch(x_torch, mask)
        nt_tt = NestedTensorTTSim(torch_to_simtensor(x_torch, "input"), mask.numpy())

        bb_pt = BackbonePyTorch(
            name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        pos_pt = build_position_encoding_pytorch(args)
        joiner_pt = JoinerPyTorch(bb_pt, pos_pt)
        joiner_pt.eval()

        bb_tt = BackboneTTSim(
            name="backbone_test",
            resnet_name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        pos_tt = build_position_encoding_ttsim(args)
        joiner_tt = JoinerTTSim("joiner_test", bb_tt, pos_tt)

        with torch.no_grad():
            _, pos_pt_out = joiner_pt(nt_pt)
        _, pos_tt_out = joiner_tt(nt_tt)

        all_ok = len(pos_pt_out) == len(pos_tt_out)
        layers_shape_ok = len(pos_pt_out) == len(pos_tt_out)
        pos_detail_rows = []
        max_diff_all = 0.0
        mean_diff_all = 0.0

        for i in range(min(len(pos_pt_out), len(pos_tt_out))):
            pt_d = _to_numpy(pos_pt_out[i])
            tt_d = _to_numpy(pos_tt_out[i])
            shape_ok = list(pos_pt_out[i].shape) == list(pos_tt_out[i].shape)

            if tt_d is None:
                num_ok = False
                mx = float("nan")
                mn = float("nan")
            else:
                adiff = np.abs(pt_d - tt_d)
                mx = float(adiff.max())
                mn = float(adiff.mean())
                max_diff_all = max(max_diff_all, mx)
                mean_diff_all = mn
                num_ok = bool(np.allclose(pt_d, tt_d, rtol=RTOL, atol=ATOL))

            ok = shape_ok and num_ok
            all_ok = all_ok and ok

            p_tag = "✅" if ok else "❌"
            pos_detail_rows.append(
                f"| pos[{i}] | `{list(pos_pt_out[i].shape)}` | {mx:.2e} | {mn:.2e} "
                f"| {_fmt_samples(pt_d)} | {_fmt_samples(tt_d)} | {p_tag} |"
            )

        passed += int(all_ok)
        shape_passed += int(layers_shape_ok)
        num_passed += int(all_ok)

        reason = ""
        if len(pos_pt_out) != len(pos_tt_out):
            reason = (
                f"Layer count mismatch: PT={len(pos_pt_out)} vs TT={len(pos_tt_out)}"
            )
        elif not all_ok:
            reason = f"max_diff={max_diff_all:.2e} exceeds atol={ATOL}"

        # Print linear output — use first layer's data for samples
        x_np = x_torch.detach().cpu().numpy()
        _pt_sample = _to_numpy(pos_pt_out[0]) if pos_pt_out else None
        _tt_sample = _to_numpy(pos_tt_out[0]) if pos_tt_out else None
        print_test_linear(
            module="PositionalEncoding",
            edge_case=category,
            edge_desc=tmsg,
            input_shape=[batch, ch, h, w],
            shape_line=f"PyTorch={len(pos_pt_out)} layers | TTSim={len(pos_tt_out)} layers",
            shape_ok=layers_shape_ok,
            is_numerical=True,
            num_ok=all_ok,
            max_diff=max_diff_all,
            mean_diff=mean_diff_all,
            rtol=RTOL,
            atol=ATOL,
            failure_reason=reason,
            pt_data=_pt_sample,
            tt_data=_tt_sample,
            input_samples={"input": x_np, "mask": mask.numpy()},
        )

        # Capture detailed results for report
        TEST_RESULTS.append(
            {
                "module": "Positional Encoding",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": f"Position encoding numerical validation",
                "input_shape": [batch, ch, h, w],
                "pt_shape": f"{len(pos_pt_out)} layers",
                "tt_shape": f"{len(pos_tt_out)} layers",
                "shape_ok": len(pos_pt_out) == len(pos_tt_out),
                "num_ok": all_ok,
                "max_diff": max_diff_all,
                "mean_diff": mean_diff_all,
                "pt_stats": None,
                "tt_stats": None,
                "passed": all_ok,
                "pos_detail_rows": pos_detail_rows,
            }
        )

        if not all_ok:
            failed_cases.append((tno, tmsg, "numerical", max_diff_all))
            FAILED_TESTS.append(
                {
                    "module": "PositionalEncoding",
                    "test": tmsg,
                    "edge_case": "numerical",
                    "max_diff": max_diff_all,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if all_ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | `[{batch},{h},{w}]` | {len(pos_pt_out)} layers | {tag} |"
        )

        p_hdr = (
            "| Layer | Shape | Max Diff | Mean Diff | PT Sample[0:10] | TT Sample[0:10] | Match |\n"
            "|:------|:------|:---------|:----------|:----------------|:----------------|:------|"
        )
        status_badge = "🟢" if all_ok else "🔴"
        detail_blocks.append(
            f"---\n\n"
            f"### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Input Shape:** `[{batch},{h},{w}]` | **Category:** {category}\n\n"
            f"**Input Float Samples [0:10]:**\n"
            f"- input: `[{_fmt_samples(x_np)}]`\n\n"
            f"**Position Encoding Layers:**\n\n"
            + p_hdr
            + "\n"
            + "\n".join(pos_detail_rows)
            + "\n\n"
        )

    # Module stats for summary
    MODULE_STATS["PositionalEncoding"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_pos),
        "num_passed": num_passed,
        "num_total": len(test_cases_pos),
    }

    hdr = (
        "| # | Test Case | Input | Pos Layers | Result |\n"
        "|:--|:----------|:------|:-----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "Positional Encoding Numerical",
            "description": "Numerical validation for sinusoidal position encodings",
            "passed": passed,
            "total": len(test_cases_pos),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_pos
    ), f"Positional Encoding: {passed}/{len(test_cases_pos)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-runner with markdown report
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a simple, module-wise markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# Backbone Unit Test Report",
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
            diff_str = f"{ft['max_diff']:.2e}" if ft["max_diff"] else "N/A"
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
    report_path = os.path.join(report_dir, "backbone_unit_test_report.md")
    output_path = os.path.join(report_dir, "backbone_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    # Print overall header
    print(f"\n{SUMMARY_LINE}")
    print(f"BACKBONE UNIT TEST SUITE - PyTorch vs TTSim")
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
