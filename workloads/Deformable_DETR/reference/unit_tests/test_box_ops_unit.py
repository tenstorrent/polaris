#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for box operations — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPE: NUMERICAL for all functions
================================================================================

All box operations are deterministic math (add, subtract, multiply, divide,
min, max, clamp).  No Conv2d or heavy neural-net ops, so every test does
full numerical comparison:

    np.allclose(pytorch_output, ttsim_output, rtol=1e-5, atol=1e-7)

================================================================================
FUNCTIONS TESTED:
================================================================================

FUNCTION 1: box_cxcywh_to_xyxy — Convert (cx, cy, w, h) → (x0, y0, x1, y1)
    Formula: x0=cx-w/2, y0=cy-h/2, x1=cx+w/2, y1=cy+h/2

FUNCTION 2: box_xyxy_to_cxcywh — Convert (x0, y0, x1, y1) → (cx, cy, w, h)
    Formula: cx=(x0+x1)/2, cy=(y0+y1)/2, w=x1-x0, h=y1-y0

FUNCTION 3: box_area — Compute area of xyxy boxes
    Formula: area = (x1 - x0) * (y1 - y0)

FUNCTION 4: box_iou — Pairwise Intersection over Union
    Formula: IoU = intersection / union

FUNCTION 5: generalized_box_iou — Generalized IoU (GIoU)
    Formula: GIoU = IoU - (enclosing_area - union) / enclosing_area

FUNCTION 6: masks_to_boxes — Extract xyxy bounding boxes from binary masks
    Algorithm: min/max of True-pixel coordinates

================================================================================
EDGE CASES TESTED PER FUNCTION:
================================================================================

'standard'     — Normal well-formed inputs, baseline correctness check
'single'       — Single box / single pair — minimum valid input
'many'         — Many boxes / large batch — scalability check
'zero_size'    — Zero-width or zero-height boxes — degenerate geometry
'large_coords' — Large coordinate values (~1e4) — overflow / precision
'small_coords' — Very small coordinate values (~1e-6) — underflow / precision
'identical'    — Identical boxes — exact-overlap special case (IoU=1)
'no_overlap'   — Fully disjoint boxes — zero intersection special case
'enclosed'     — One box fully inside another — containment special case
'full_mask'    — Entire image is True — max bounding box
'single_pixel' — A single True pixel in mask — point bounding box
'empty_mask'   — All-False mask — degenerate input

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_box_ops_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_box_ops_unit.py
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
from workloads.Deformable_DETR.reference.box_ops import (
    box_cxcywh_to_xyxy as box_cxcywh_to_xyxy_pytorch,
    box_xyxy_to_cxcywh as box_xyxy_to_cxcywh_pytorch,
    box_iou as box_iou_pytorch,
    generalized_box_iou as generalized_box_iou_pytorch,
    masks_to_boxes as masks_to_boxes_pytorch,
)
from torchvision.ops.boxes import box_area as box_area_pytorch

# TTSim implementations
from workloads.Deformable_DETR.util.box_ops_ttsim import (
    box_cxcywh_to_xyxy as box_cxcywh_to_xyxy_ttsim,
    box_xyxy_to_cxcywh as box_xyxy_to_cxcywh_ttsim,
    box_area as box_area_ttsim,
    box_iou as box_iou_ttsim,
    generalized_box_iou as generalized_box_iou_ttsim,
    masks_to_boxes as masks_to_boxes_ttsim,
)

from ttsim.ops.tensor import SimTensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RTOL = 1e-5
ATOL = 1e-7
SEED = 42


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
    "standard": "Well-formed typical boxes — baseline correctness",
    "single": "Single box / minimal valid input",
    "many": "Large batch of boxes — scalability check",
    "zero_size": "Zero-width or zero-height — degenerate geometry",
    "large_coords": "Large coordinates (~1e4) — overflow / precision",
    "small_coords": "Very small coordinates (~1e-6) — underflow / precision",
    "identical": "Identical boxes — IoU = 1.0 special case",
    "no_overlap": "Disjoint boxes — zero intersection",
    "enclosed": "One box fully inside another — containment",
    "full_mask": "Entire image True — maximum bounding box",
    "single_pixel": "Single True pixel — point bounding box",
    "empty_mask": "All-False mask — degenerate input",
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


def _compact_shape(shape):
    """Format shape as [N,4] (no spaces)."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def _fmt_samples(arr, n=10):
    """First *n* flat values, formatted for a markdown table cell."""
    return ", ".join(f"{v:.6f}" for v in arr.flat[:n])


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
):
    """Print test result in clean tree-style linear format."""
    passed = shape_ok and (num_ok if is_numerical else True)

    print(f"\nFUNCTION: {Colors.bold(module)}")
    print(f"├─ EDGE CASE: {Colors.warn(edge_case)} ({edge_desc})")
    print(f"├─ INPUT: {input_shape}")

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
        pt_arr = pt_data.flat[:5] if hasattr(pt_data, "flat") else pt_data
        tt_arr = tt_data.flat[:5] if hasattr(tt_data, "flat") else tt_data
        pt_str = ", ".join(f"{v:.6f}" for v in pt_arr)
        tt_str = ", ".join(f"{v:.6f}" for v in tt_arr)
        print(f"├─ PT OUTPUT[0:5]: [{pt_str}]")
        print(f"├─ TT OUTPUT[0:5]: [{tt_str}]")

    if not passed and failure_reason:
        print(f"├─ FAILURE REASON: {Colors.fail(failure_reason)}")

    result_str = Colors.success("✓ PASS") if passed else Colors.fail("✗ FAIL")
    print(f"└─ RESULT: {result_str}")


def print_summary():
    """Print the final summary table."""
    print(f"\n{SUMMARY_LINE}")
    print("SUMMARY")
    print(SUMMARY_LINE)
    print(f"{'FUNCTION':<28}{'SHAPE':<12}{'NUMERICAL':<12}TOTAL")

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
            print(f"- {ft['module']} | {ft['edge_case']} | {diff_str}{gt_str}")

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
#  TEST 1 — box_cxcywh_to_xyxy
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_cx2xy = [
    # (description, boxes_data, edge_category)
    (
        "Standard 3 boxes",
        [[0.5, 0.5, 0.4, 0.6], [0.25, 0.75, 0.3, 0.4], [0.8, 0.3, 0.2, 0.5]],
        "standard",
    ),
    ("Single box", [[0.5, 0.5, 1.0, 1.0]], "single"),
    ("Many boxes (20)", None, "many"),  # generated at runtime
    ("Zero-width box (w=0)", [[0.5, 0.5, 0.0, 0.4], [0.3, 0.3, 0.2, 0.0]], "zero_size"),
    (
        "Large coordinates (~1e4)",
        [[5000.0, 5000.0, 2000.0, 3000.0], [9999.0, 9999.0, 100.0, 100.0]],
        "large_coords",
    ),
    (
        "Small coordinates (~1e-6)",
        [[1e-6, 1e-6, 5e-7, 5e-7], [0.5, 0.5, 1e-6, 1e-6]],
        "small_coords",
    ),
]


@pytest.mark.unit
def test_box_cxcywh_to_xyxy():
    """Test box_cxcywh_to_xyxy: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, boxes_data, category) in enumerate(test_cases_cx2xy):
        if boxes_data is None:
            # Generate random boxes for 'many' case
            np.random.seed(SEED)
            boxes_data = np.random.rand(20, 4).astype(np.float32).tolist()

        boxes_torch = torch.tensor(boxes_data, dtype=torch.float32)
        boxes_sim = torch_to_simtensor(boxes_torch, "boxes_cxcywh")

        out_pt = box_cxcywh_to_xyxy_pytorch(boxes_torch)
        out_tt = box_cxcywh_to_xyxy_ttsim(boxes_sim)

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        pt_shape = list(out_pt.shape)
        tt_shape = list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

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
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        print_test_linear(
            module="box_cxcywh_to_xyxy",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=_compact_shape(list(boxes_torch.shape)),
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
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
        )

        TEST_RESULTS.append(
            {
                "module": "box_cxcywh_to_xyxy",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": list(boxes_torch.shape),
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
                    "module": "box_cxcywh_to_xyxy",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{list(boxes_torch.shape)}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input Shape:** `{list(boxes_torch.shape)}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["box_cxcywh_to_xyxy"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_cx2xy),
        "num_passed": num_passed,
        "num_total": len(test_cases_cx2xy),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "box_cxcywh_to_xyxy",
            "description": "Convert boxes from (cx,cy,w,h) to (x0,y0,x1,y1)",
            "passed": passed,
            "total": len(test_cases_cx2xy),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_cx2xy
    ), f"box_cxcywh_to_xyxy: {passed}/{len(test_cases_cx2xy)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — box_xyxy_to_cxcywh
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_xy2cx = [
    (
        "Standard 3 boxes",
        [[0.1, 0.2, 0.5, 0.7], [0.3, 0.3, 0.8, 0.9], [0.0, 0.0, 0.2, 0.3]],
        "standard",
    ),
    ("Single box", [[0.0, 0.0, 1.0, 1.0]], "single"),
    ("Many boxes (20)", None, "many"),
    ("Zero-area point box", [[0.5, 0.5, 0.5, 0.5]], "zero_size"),
    ("Large coordinates", [[100.0, 200.0, 5000.0, 8000.0]], "large_coords"),
    ("Small coordinates", [[1e-7, 2e-7, 3e-7, 4e-7]], "small_coords"),
]


@pytest.mark.unit
def test_box_xyxy_to_cxcywh():
    """Test box_xyxy_to_cxcywh: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, boxes_data, category) in enumerate(test_cases_xy2cx):
        if boxes_data is None:
            np.random.seed(SEED)
            raw = np.sort(np.random.rand(20, 2, 2).astype(np.float32), axis=-1)
            boxes_data = np.concatenate([raw[:, 0, :], raw[:, 1, :]], axis=-1).tolist()

        boxes_torch = torch.tensor(boxes_data, dtype=torch.float32)
        boxes_sim = torch_to_simtensor(boxes_torch, "boxes_xyxy")

        out_pt = box_xyxy_to_cxcywh_pytorch(boxes_torch)
        out_tt = box_xyxy_to_cxcywh_ttsim(boxes_sim)

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        pt_shape = list(out_pt.shape)
        tt_shape = list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

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
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        print_test_linear(
            module="box_xyxy_to_cxcywh",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=_compact_shape(list(boxes_torch.shape)),
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
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
        )

        TEST_RESULTS.append(
            {
                "module": "box_xyxy_to_cxcywh",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": list(boxes_torch.shape),
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
                    "module": "box_xyxy_to_cxcywh",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{list(boxes_torch.shape)}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input Shape:** `{list(boxes_torch.shape)}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["box_xyxy_to_cxcywh"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_xy2cx),
        "num_passed": num_passed,
        "num_total": len(test_cases_xy2cx),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "box_xyxy_to_cxcywh",
            "description": "Convert boxes from (x0,y0,x1,y1) to (cx,cy,w,h)",
            "passed": passed,
            "total": len(test_cases_xy2cx),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_xy2cx
    ), f"box_xyxy_to_cxcywh: {passed}/{len(test_cases_xy2cx)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — box_area
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_area = [
    ("Unit square", [[0.0, 0.0, 1.0, 1.0]], "standard"),
    (
        "Multiple boxes",
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.2, 0.5, 0.7],
            [0.3, 0.3, 0.8, 0.9],
            [0.0, 0.0, 0.5, 0.5],
        ],
        "standard",
    ),
    ("Single tiny box", [[0.4, 0.4, 0.6, 0.6]], "single"),
    ("Zero-area box (point)", [[0.5, 0.5, 0.5, 0.5]], "zero_size"),
    ("Zero-width box", [[0.3, 0.0, 0.3, 1.0]], "zero_size"),
    (
        "Large coordinate boxes",
        [[0.0, 0.0, 10000.0, 10000.0], [5000.0, 5000.0, 9999.0, 9999.0]],
        "large_coords",
    ),
    ("Small coordinate boxes", [[0.0, 0.0, 1e-6, 1e-6]], "small_coords"),
    ("Many boxes (50)", None, "many"),
]


@pytest.mark.unit
def test_box_area():
    """Test box_area: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, boxes_data, category) in enumerate(test_cases_area):
        if boxes_data is None:
            np.random.seed(SEED)
            xy0 = np.random.rand(50, 2).astype(np.float32)
            wh = np.random.rand(50, 2).astype(np.float32) * 0.5
            xy1 = xy0 + wh
            boxes_data = np.concatenate([xy0, xy1], axis=-1).tolist()

        boxes_torch = torch.tensor(boxes_data, dtype=torch.float32)
        boxes_sim = torch_to_simtensor(boxes_torch, "boxes")

        out_pt = box_area_pytorch(boxes_torch)
        out_tt = box_area_ttsim(boxes_sim)

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        pt_shape = list(out_pt.shape)
        tt_shape = list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

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
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        print_test_linear(
            module="box_area",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=_compact_shape(list(boxes_torch.shape)),
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
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
        )

        TEST_RESULTS.append(
            {
                "module": "box_area",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": list(boxes_torch.shape),
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
                    "module": "box_area",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{list(boxes_torch.shape)}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input Shape:** `{list(boxes_torch.shape)}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["box_area"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_area),
        "num_passed": num_passed,
        "num_total": len(test_cases_area),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "box_area",
            "description": "Compute area of boxes in (x0,y0,x1,y1) format: area=(x1-x0)*(y1-y0)",
            "passed": passed,
            "total": len(test_cases_area),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_area
    ), f"box_area: {passed}/{len(test_cases_area)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — box_iou
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_iou = [
    # (description, boxes1, boxes2, edge_category)
    (
        "Standard overlap",
        [[0.0, 0.0, 0.5, 0.5], [0.2, 0.2, 0.7, 0.7]],
        [[0.0, 0.0, 0.5, 0.5], [0.4, 0.4, 0.9, 0.9], [0.8, 0.8, 1.0, 1.0]],
        "standard",
    ),
    (
        "Identical boxes (IoU=1)",
        [[0.1, 0.1, 0.6, 0.6], [0.3, 0.3, 0.9, 0.9]],
        [[0.1, 0.1, 0.6, 0.6], [0.3, 0.3, 0.9, 0.9]],
        "identical",
    ),
    (
        "No overlap (disjoint)",
        [[0.0, 0.0, 0.2, 0.2]],
        [[0.8, 0.8, 1.0, 1.0]],
        "no_overlap",
    ),
    ("Single box pair", [[0.0, 0.0, 0.5, 0.5]], [[0.25, 0.25, 0.75, 0.75]], "single"),
    ("Enclosed box", [[0.0, 0.0, 1.0, 1.0]], [[0.2, 0.2, 0.8, 0.8]], "enclosed"),
    (
        "Large coordinate boxes",
        [[0.0, 0.0, 5000.0, 5000.0]],
        [[2500.0, 2500.0, 10000.0, 10000.0]],
        "large_coords",
    ),
]


@pytest.mark.unit
def test_box_iou():
    """Test box_iou: shape + numerical validation for IoU and union."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, b1_data, b2_data, category) in enumerate(test_cases_iou):
        b1_torch = torch.tensor(b1_data, dtype=torch.float32)
        b2_torch = torch.tensor(b2_data, dtype=torch.float32)
        b1_sim = torch_to_simtensor(b1_torch, "boxes1")
        b2_sim = torch_to_simtensor(b2_torch, "boxes2")

        iou_pt, union_pt = box_iou_pytorch(b1_torch, b2_torch)
        iou_tt, union_tt = box_iou_ttsim(b1_sim, b2_sim)

        iou_pt_d = _to_numpy(iou_pt)
        iou_tt_d = _to_numpy(iou_tt)
        union_pt_d = _to_numpy(union_pt)
        union_tt_d = _to_numpy(union_tt)

        iou_shape_ok = list(iou_pt.shape) == list(iou_tt.shape)
        union_shape_ok = list(union_pt.shape) == list(union_tt.shape)
        shape_ok = iou_shape_ok and union_shape_ok

        iou_diff = np.abs(iou_pt_d - iou_tt_d)
        union_diff = np.abs(union_pt_d - union_tt_d)
        mx = float(max(iou_diff.max(), union_diff.max()))
        mn = float((iou_diff.mean() + union_diff.mean()) / 2)
        iou_match = bool(np.allclose(iou_pt_d, iou_tt_d, rtol=RTOL, atol=ATOL))
        union_match = bool(np.allclose(union_pt_d, union_tt_d, rtol=RTOL, atol=ATOL))
        num_ok = iou_match and union_match

        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: IoU PT={list(iou_pt.shape)} vs TT={list(iou_tt.shape)}"
        elif not iou_match:
            reason = f"IoU max_diff={float(iou_diff.max()):.2e} exceeds atol={ATOL}"
        elif not union_match:
            reason = f"Union max_diff={float(union_diff.max()):.2e} exceeds atol={ATOL}"

        expected_shape = f"[{len(b1_data)},{len(b2_data)}]"
        print_test_linear(
            module="box_iou",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=f"boxes1={_compact_shape(list(b1_torch.shape))}, boxes2={_compact_shape(list(b2_torch.shape))}",
            shape_line=f"IoU: PT={_compact_shape(list(iou_pt.shape))} | TT={_compact_shape(list(iou_tt.shape))}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=RTOL,
            atol=ATOL,
            failure_reason=reason,
            pt_data=iou_pt_d,
            tt_data=iou_tt_d,
        )

        TEST_RESULTS.append(
            {
                "module": "box_iou",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": f"[{len(b1_data)},4]+[{len(b2_data)},4]",
                "pt_shape": list(iou_pt.shape),
                "tt_shape": list(iou_tt.shape),
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
                    "module": "box_iou",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{expected_shape}` "
            f"| `{list(iou_pt.shape)}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**IoU Matrix:**\n"
            f"- PyTorch: `[{_fmt_samples(iou_pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(iou_tt_d)}]`\n\n"
            f"**Union Matrix:**\n"
            f"- PyTorch: `[{_fmt_samples(union_pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(union_tt_d)}]`\n\n"
        )

    MODULE_STATS["box_iou"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_iou),
        "num_passed": num_passed,
        "num_total": len(test_cases_iou),
    }

    hdr = (
        "| # | Test Case | Edge Case | Expected | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:---------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "box_iou",
            "description": "Pairwise Intersection-over-Union and union area",
            "passed": passed,
            "total": len(test_cases_iou),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_iou
    ), f"box_iou: {passed}/{len(test_cases_iou)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — generalized_box_iou
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_giou = [
    (
        "Standard overlap",
        [[0.0, 0.0, 0.5, 0.5], [0.6, 0.6, 1.0, 1.0]],
        [[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.8, 0.8, 1.0, 1.0]],
        "standard",
    ),
    (
        "Identical boxes (GIoU=1)",
        [[0.1, 0.1, 0.5, 0.5]],
        [[0.1, 0.1, 0.5, 0.5]],
        "identical",
    ),
    (
        "No overlap — negative GIoU",
        [[0.0, 0.0, 0.1, 0.1]],
        [[0.9, 0.9, 1.0, 1.0]],
        "no_overlap",
    ),
    ("Enclosed box", [[0.0, 0.0, 1.0, 1.0]], [[0.3, 0.3, 0.7, 0.7]], "enclosed"),
    ("Single box pair", [[0.2, 0.2, 0.6, 0.6]], [[0.4, 0.4, 0.9, 0.9]], "single"),
    (
        "Large coordinate boxes",
        [[0.0, 0.0, 5000.0, 5000.0]],
        [[2500.0, 2500.0, 10000.0, 10000.0]],
        "large_coords",
    ),
]


@pytest.mark.unit
def test_generalized_box_iou():
    """Test generalized_box_iou (GIoU): shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, b1_data, b2_data, category) in enumerate(test_cases_giou):
        b1_torch = torch.tensor(b1_data, dtype=torch.float32)
        b2_torch = torch.tensor(b2_data, dtype=torch.float32)
        b1_sim = torch_to_simtensor(b1_torch, "boxes1")
        b2_sim = torch_to_simtensor(b2_torch, "boxes2")

        giou_pt = generalized_box_iou_pytorch(b1_torch, b2_torch)
        giou_tt = generalized_box_iou_ttsim(b1_sim, b2_sim)

        pt_d = _to_numpy(giou_pt)
        tt_d = _to_numpy(giou_tt)
        pt_shape = list(giou_pt.shape)
        tt_shape = list(giou_tt.shape)
        shape_ok = pt_shape == tt_shape

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
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        expected_shape = f"[{len(b1_data)},{len(b2_data)}]"
        print_test_linear(
            module="generalized_box_iou",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=f"boxes1={_compact_shape(list(b1_torch.shape))}, boxes2={_compact_shape(list(b2_torch.shape))}",
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
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
        )

        TEST_RESULTS.append(
            {
                "module": "generalized_box_iou",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": f"[{len(b1_data)},4]+[{len(b2_data)},4]",
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
                    "module": "generalized_box_iou",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{expected_shape}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**GIoU range:** [-1, 1]  (1.0 = identical, 0.0 = touching, -1.0 = maximally separated)\n\n"
            f"**GIoU Matrix:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["generalized_box_iou"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_giou),
        "num_passed": num_passed,
        "num_total": len(test_cases_giou),
    }

    hdr = (
        "| # | Test Case | Edge Case | Expected | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:---------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "generalized_box_iou",
            "description": "Generalized IoU: GIoU = IoU - (enclosing - union) / enclosing",
            "passed": passed,
            "total": len(test_cases_giou),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_giou
    ), f"generalized_box_iou: {passed}/{len(test_cases_giou)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — masks_to_boxes
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mask_cases():
    """Build mask test cases at import time."""
    cases = []

    # Standard — rectangle regions
    m0 = torch.zeros(3, 10, 10, dtype=torch.bool)
    m0[0, 2:5, 3:8] = True  # box [3,2,7,4]
    m0[1, 5:9, 1:5] = True  # box [1,5,4,8]
    m0[2, 0:2, 8:10] = True  # box [8,0,9,1]
    cases.append(("3 rectangle masks (10×10)", m0, "standard"))

    # Full mask — entire image True
    m1 = torch.ones(1, 8, 8, dtype=torch.bool)
    cases.append(("Full mask 8×8", m1, "full_mask"))

    # Single pixel
    m2 = torch.zeros(1, 16, 16, dtype=torch.bool)
    m2[0, 7, 7] = True
    cases.append(("Single pixel [7,7]", m2, "single_pixel"))

    # Multiple single-pixel masks
    m3 = torch.zeros(2, 10, 10, dtype=torch.bool)
    m3[0, 0, 0] = True
    m3[1, 9, 9] = True
    cases.append(("Corner pixels [0,0] & [9,9]", m3, "single_pixel"))

    # Large mask
    m4 = torch.zeros(2, 64, 64, dtype=torch.bool)
    m4[0, 10:50, 5:55] = True
    m4[1, 0:64, 0:64] = True
    cases.append(("Large 64×64 masks", m4, "large_coords"))

    return cases


test_cases_masks = _make_mask_cases()


@pytest.mark.unit
def test_masks_to_boxes():
    """Test masks_to_boxes: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, masks_torch, category) in enumerate(test_cases_masks):
        masks_sim = torch_to_simtensor(masks_torch.float(), "masks")

        out_pt = masks_to_boxes_pytorch(masks_torch)
        out_tt = masks_to_boxes_ttsim(masks_sim)

        pt_d = _to_numpy(out_pt)
        tt_d = _to_numpy(out_tt)
        pt_shape = list(out_pt.shape)
        tt_shape = list(out_tt.shape)
        shape_ok = pt_shape == tt_shape

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
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={ATOL}"

        print_test_linear(
            module="masks_to_boxes",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=_compact_shape(list(masks_torch.shape)),
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
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
        )

        TEST_RESULTS.append(
            {
                "module": "masks_to_boxes",
                "validation_type": "NUMERICAL",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": list(masks_torch.shape),
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
                    "module": "masks_to_boxes",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": ATOL,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{list(masks_torch.shape)}` "
            f"| `{pt_shape}` | {mx:.2e} | {mn:.2e} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Mask Shape:** `{list(masks_torch.shape)}` → **Boxes Shape:** `{pt_shape}`\n\n"
            f"**Extracted Boxes:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["masks_to_boxes"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_masks),
        "num_passed": num_passed,
        "num_total": len(test_cases_masks),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "masks_to_boxes",
            "description": "Extract xyxy bounding boxes from binary masks",
            "passed": passed,
            "total": len(test_cases_masks),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_masks
    ), f"masks_to_boxes: {passed}/{len(test_cases_masks)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Markdown report + self-runner
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a structured markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# Box Operations Unit Test Report",
        f"**PyTorch vs TTSim Comparison** | **{total_passed}/{total_tests} passed** | {status}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Exit Code: {exit_code}",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Function | Passed | Total | Status |")
    lines.append("|----------|--------|-------|--------|")
    for s in REPORT_SECTIONS:
        ms = "PASS" if s["passed"] == s["total"] else "FAIL"
        lines.append(f"| {s['name']} | {s['passed']} | {s['total']} | {ms} |")
    lines.append("")
    lines.append(f"**Total: {total_passed}/{total_tests} tests passed**")
    lines.append("")

    # Failed tests
    if FAILED_TESTS:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Tests")
        lines.append("")
        lines.append("| Function | Test | Edge Case | Max Diff |")
        lines.append("|----------|------|-----------|----------|")
        for ft in FAILED_TESTS:
            diff = f"{ft['max_diff']:.2e}" if ft.get("max_diff") else "N/A"
            lines.append(
                f"| {ft['module']} | {ft['test']} | {ft['edge_case']} | {diff} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Per-function details
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
    report_path = os.path.join(report_dir, "box_ops_unit_test_report.md")
    output_path = os.path.join(report_dir, "box_ops_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    print(f"\n{SUMMARY_LINE}")
    print(f"BOX OPERATIONS UNIT TEST SUITE - PyTorch vs TTSim")
    print(f"{SUMMARY_LINE}\n")

    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    _sync_globals_from_pytest()

    print_summary()

    sys.stdout = _original_stdout
    _tee_file.close()

    _write_markdown_report(report_path, exit_code)

    print(f"\n{Colors.cyan(f'[Markdown report : {report_path}]')}")
    print(f"{Colors.cyan(f'[Full output log  : {output_path}]')}\n")
    sys.exit(exit_code)
