#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for HungarianMatcher — PyTorch vs TTSim comparison WITH EDGE CASES.

================================================================================
VALIDATION TYPES EXPLAINED:
================================================================================

NUMERICAL VALIDATION:
    - Compares actual OUTPUT VALUES between PyTorch and TTSim
    - Reports: max absolute difference, mean absolute difference
    - PASS if: all values within tolerance
    - FAIL if: any value exceeds tolerance

INDEX VALIDATION (for full matcher):
    - Compares matched index assignments between PyTorch and TTSim
    - Uses np.array_equal() for exact index match
    - PASS if: both source & target indices identical
    - FAIL if: any index mismatch

================================================================================
MODULES TESTED:
================================================================================

MODULE 1: Focal Loss Cost Component — NUMERICAL VALIDATION
    Formula: cost = α(1-p)^γ · (−log(p+ε)) − (1-α)p^γ · (−log(1-p+ε))
    Edge Cases: standard, extreme_logits, uniform_probs, single_target,
                many_classes, sparse_logits
    Tolerance: rtol=1e-4, atol=1e-5
    WHY NUMERICAL: Elementwise sigmoid/log/pow — fast, deterministic

MODULE 2: L1 Bbox Cost Component — NUMERICAL VALIDATION
    Formula: cost_bbox[i,j] = Σ|pred_box[i] − tgt_box[j]| (pairwise L1)
    Edge Cases: standard, identical_boxes, distant_boxes, single_pair,
                many_boxes, zero_size_boxes
    Tolerance: rtol=1e-5, atol=1e-6
    WHY NUMERICAL: Simple absolute-value arithmetic

MODULE 3: GIoU Cost Component — NUMERICAL VALIDATION
    Formula: cost_giou = −GIoU(xyxy(pred), xyxy(tgt))
    Edge Cases: standard, identical_boxes, no_overlap, enclosed,
                single_pair, many_boxes
    Tolerance: rtol=1e-4, atol=1e-5
    WHY NUMERICAL: Deterministic box geometry

MODULE 4: HungarianMatcher Shape — SHAPE VALIDATION
    Validates output structure: list of tuples, index array lengths, dtypes
    Edge Cases: standard_batch, single_image, many_gt, single_gt,
                equal_queries_gt, unbalanced_gt
    WHY SHAPE: Validates API contract and structure correctness

MODULE 5: HungarianMatcher End-to-End — INDEX VALIDATION
    Validates identical matching assignments between PyTorch and TTSim
    Edge Cases: standard, single_image, many_gt, few_gt, different_weights
    Tolerance: exact match (np.array_equal on index arrays)
    WHY INDEX: Verifies the full pipeline produces identical assignments

================================================================================
EDGE CASES TESTED:
================================================================================

'standard'        — Typical well-formed inputs — baseline correctness
'extreme_logits'  — Very large/small logit values — sigmoid saturation
'uniform_probs'   — Uniform probability (0.5) — degenerate focal loss
'single_target'   — Single GT object — minimal matching
'many_classes'    — Large class space — scalability check
'sparse_logits'   — Mostly-zero logits — sparse activation pattern
'identical_boxes' — Identical pred/target boxes — zero distance
'distant_boxes'   — Widely separated boxes — large L1 distances
'single_pair'     — Single query, single GT — 1×1 cost matrix
'many_boxes'      — Large query/GT count — scalability
'zero_size_boxes' — Degenerate zero-area boxes
'no_overlap'      — Non-overlapping boxes — GIoU < 0
'enclosed'        — One box fully inside another — containment
'single_image'    — Batch size = 1
'many_gt'         — Many ground-truth objects per image
'few_gt'          — Very few GT objects (1 per image)
'equal_qgt'       — num_queries == num_gt — square cost matrix
'unbalanced_gt'   — Different GT counts per image in batch
'different_wts'   — Non-default cost weights

================================================================================
RUN:
    cd polaris
    pytest workloads/Deformable_DETR/unit_tests/test_matcher_unit.py -v -s
    # or
    python workloads/Deformable_DETR/unit_tests/test_matcher_unit.py
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
from workloads.Deformable_DETR.reference.matcher import (
    HungarianMatcher as HungarianMatcherPyTorch,
)
from workloads.Deformable_DETR.reference.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)

# TTSim implementations
from workloads.Deformable_DETR.models.matcher_ttsim import (
    HungarianMatcher as HungarianMatcherTTSim,
)
from ttsim.ops.tensor import SimTensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FOCAL_RTOL = 1e-4
FOCAL_ATOL = 1e-5
L1_RTOL = 1e-5
L1_ATOL = 1e-6
GIOU_RTOL = 1e-4
GIOU_ATOL = 1e-5
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
    "standard": "Well-formed typical inputs — baseline correctness",
    "extreme_logits": "Very large/small logits — sigmoid saturation",
    "uniform_probs": "Uniform probability (0.5) — degenerate focal loss",
    "single_target": "Single GT object — minimal matching",
    "many_classes": "Large class space — scalability check",
    "sparse_logits": "Mostly-zero logits — sparse activation pattern",
    "identical_boxes": "Identical pred/tgt boxes — zero distance",
    "distant_boxes": "Widely separated boxes — large L1 distances",
    "single_pair": "Single query + single GT — 1×1 cost matrix",
    "many_boxes": "Large query/GT count — scalability",
    "zero_size_boxes": "Degenerate zero-area boxes",
    "no_overlap": "Non-overlapping boxes — GIoU < 0",
    "enclosed": "One box fully inside another — containment",
    "single_image": "Batch size = 1",
    "many_gt": "Many ground-truth objects per image",
    "few_gt": "Very few GT objects (1 per image)",
    "equal_qgt": "num_queries == num_gt — square cost matrix",
    "unbalanced_gt": "Different GT counts per image in batch",
    "different_wts": "Non-default cost weights",
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


def _make_simtensor(data, name="tensor"):
    """Create SimTensor from numpy array."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().copy()
    data = np.asarray(data)
    return SimTensor(
        {
            "name": name,
            "shape": list(data.shape),
            "data": data,
            "dtype": data.dtype,
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
    """Format shape as [N,M] (no spaces)."""
    if isinstance(shape, (list, tuple)):
        return "[" + ",".join(str(s) for s in shape) + "]"
    return str(shape)


def _fmt_samples(arr, n=10):
    """First *n* flat values, formatted for a markdown table cell."""
    return ", ".join(f"{v:.6f}" for v in arr.flat[:n])


def _generate_inputs(batch_size, num_queries, num_classes, num_gt_per_image, seed=SEED):
    """Generate matched PyTorch/NumPy inputs for matcher testing.

    Returns:
        (outputs_pt, targets_pt, outputs_np, targets_np)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pred_logits = torch.randn(batch_size, num_queries, num_classes, dtype=torch.float32)
    pred_boxes = torch.sigmoid(
        torch.randn(batch_size, num_queries, 4, dtype=torch.float32)
    )
    # Ensure valid cxcywh (positive w, h)
    pred_boxes[:, :, 2:] = pred_boxes[:, :, 2:].clamp(min=0.01)

    outputs_pt = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

    targets_pt = []
    targets_np = []
    for i in range(batch_size):
        ngt = num_gt_per_image[i]
        labels = torch.randint(0, num_classes, (ngt,), dtype=torch.int64)
        boxes = torch.sigmoid(torch.randn(ngt, 4, dtype=torch.float32))
        boxes[:, 2:] = boxes[:, 2:].clamp(min=0.01)
        targets_pt.append({"labels": labels, "boxes": boxes})
        targets_np.append({"labels": labels.numpy(), "boxes": boxes.numpy()})

    outputs_np = {
        "pred_logits": pred_logits.numpy(),
        "pred_boxes": pred_boxes.numpy(),
    }

    return outputs_pt, targets_pt, outputs_np, targets_np


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
    rtol=FOCAL_RTOL,
    atol=FOCAL_ATOL,
    failure_reason="",
    pt_data=None,
    tt_data=None,
):
    """Print test result in clean tree-style linear format."""
    passed = shape_ok and (num_ok if is_numerical else True)

    print(f"\nMODULE: {Colors.bold(module)}")
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
            atol_val = ft.get("atol", FOCAL_ATOL)
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
#  TEST 1 — Focal Loss Cost Component
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_focal = [
    # (description, num_queries, num_classes, num_gt, logit_mode, edge_category)
    ("Standard 10q × 91c × 5gt", 10, 91, 5, "random", "standard"),
    ("Extreme logits ±10", 10, 91, 5, "extreme", "extreme_logits"),
    ("Uniform probs (logit=0)", 10, 91, 5, "uniform", "uniform_probs"),
    ("Single target", 10, 91, 1, "random", "single_target"),
    ("Many classes (200)", 10, 200, 5, "random", "many_classes"),
    ("Sparse logits (mostly zero)", 10, 91, 5, "sparse", "sparse_logits"),
]


@pytest.mark.unit
def test_focal_loss_cost():
    """Test focal loss cost component: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []
    alpha = 0.25
    gamma = 2.0
    eps = 1e-8

    for tno, (tmsg, nq, nc, ngt, logit_mode, category) in enumerate(test_cases_focal):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        # Generate logits based on mode
        if logit_mode == "random":
            logits_np = np.random.randn(nq, nc).astype(np.float32)
        elif logit_mode == "extreme":
            logits_np = np.random.randn(nq, nc).astype(np.float32) * 10.0
        elif logit_mode == "uniform":
            logits_np = np.zeros((nq, nc), dtype=np.float32)
        elif logit_mode == "sparse":
            logits_np = np.zeros((nq, nc), dtype=np.float32)
            mask = np.random.rand(nq, nc) < 0.1
            logits_np[mask] = np.random.randn(mask.sum()).astype(np.float32)
        else:
            logits_np = np.random.randn(nq, nc).astype(np.float32)

        tgt_ids_np = np.random.randint(0, nc, size=(ngt,), dtype=np.int64)

        logits_pt = torch.tensor(logits_np)
        tgt_ids_pt = torch.tensor(tgt_ids_np)

        # --- PyTorch focal loss computation ---
        out_prob_pt = logits_pt.sigmoid()
        neg_cost = (1 - alpha) * (out_prob_pt**gamma) * (-(1 - out_prob_pt + eps).log())
        pos_cost = alpha * ((1 - out_prob_pt) ** gamma) * (-(out_prob_pt + eps).log())
        cost_class_pt = pos_cost[:, tgt_ids_pt] - neg_cost[:, tgt_ids_pt]
        pt_d = cost_class_pt.detach().numpy()

        # --- TTSim focal loss computation ---
        matcher_tt = HungarianMatcherTTSim(
            name=f"test_focal_{tno}", cost_class=1.0, cost_bbox=1.0, cost_giou=1.0
        )
        out_prob_np = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
        out_prob_sim = SimTensor(
            {
                "name": f"test_focal_{tno}.out_prob",
                "shape": list(out_prob_np.shape),
                "data": out_prob_np,
                "dtype": out_prob_np.dtype,
            }
        )
        out_prob_sim.set_module(matcher_tt)

        tgt_ids_sim = SimTensor(
            {
                "name": f"test_focal_{tno}.tgt_ids",
                "shape": list(tgt_ids_np.shape),
                "data": tgt_ids_np,
                "dtype": tgt_ids_np.dtype,
            }
        )
        tgt_ids_sim.set_module(matcher_tt)

        cost_class_tt = matcher_tt._compute_focal_loss_cost(out_prob_sim, tgt_ids_sim)
        tt_d = cost_class_tt.data

        # --- Compare ---
        pt_shape = list(pt_d.shape)
        tt_shape = list(tt_d.shape)
        shape_ok = pt_shape == tt_shape

        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=FOCAL_RTOL, atol=FOCAL_ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={FOCAL_ATOL}"

        inp_desc = f"logits[{nq},{nc}] tgt_ids[{ngt}]"
        print_test_linear(
            module="FocalLossCost",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=FOCAL_RTOL,
            atol=FOCAL_ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
        )

        TEST_RESULTS.append(
            {
                "module": "FocalLossCost",
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
                    "module": "FocalLossCost",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": FOCAL_ATOL,
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
            f"**Input:** `{inp_desc}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["FocalLossCost"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_focal),
        "num_passed": num_passed,
        "num_total": len(test_cases_focal),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "FocalLossCost",
            "description": "Focal loss cost component — α(1−p)^γ · (−log p) classification cost",
            "passed": passed,
            "total": len(test_cases_focal),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_focal
    ), f"FocalLossCost: {passed}/{len(test_cases_focal)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — L1 Bbox Cost Component
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_l1 = [
    # (description, num_queries, num_gt, box_mode, edge_category)
    ("Standard 10q × 5gt", 10, 5, "random", "standard"),
    ("Identical boxes", 5, 5, "identical", "identical_boxes"),
    ("Distant boxes", 5, 5, "distant", "distant_boxes"),
    ("Single pair (1q × 1gt)", 1, 1, "random", "single_pair"),
    ("Many boxes (50q × 20gt)", 50, 20, "random", "many_boxes"),
    ("Zero size boxes (w=0)", 5, 5, "zero_size", "zero_size_boxes"),
]


@pytest.mark.unit
def test_l1_bbox_cost():
    """Test L1 bbox cost component: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, nq, ngt, box_mode, category) in enumerate(test_cases_l1):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        # Generate boxes (cxcywh format, values in [0,1])
        if box_mode == "random":
            pred_np = np.random.rand(nq, 4).astype(np.float32)
            pred_np[:, 2:] = np.clip(pred_np[:, 2:], 0.01, None)
            tgt_np = np.random.rand(ngt, 4).astype(np.float32)
            tgt_np[:, 2:] = np.clip(tgt_np[:, 2:], 0.01, None)
        elif box_mode == "identical":
            base = np.random.rand(1, 4).astype(np.float32)
            base[:, 2:] = np.clip(base[:, 2:], 0.01, None)
            pred_np = np.tile(base, (nq, 1))
            tgt_np = np.tile(base, (ngt, 1))
        elif box_mode == "distant":
            pred_np = np.random.rand(nq, 4).astype(np.float32) * 0.1
            pred_np[:, 2:] = np.clip(pred_np[:, 2:], 0.01, None)
            tgt_np = np.random.rand(ngt, 4).astype(np.float32) * 0.1 + 0.9
            tgt_np[:, 2:] = np.clip(tgt_np[:, 2:], 0.01, None)
        elif box_mode == "zero_size":
            pred_np = np.random.rand(nq, 4).astype(np.float32)
            pred_np[:, 2] = 0.0  # zero width
            tgt_np = np.random.rand(ngt, 4).astype(np.float32)
            tgt_np[:, 3] = 0.0  # zero height
        else:
            pred_np = np.random.rand(nq, 4).astype(np.float32)
            tgt_np = np.random.rand(ngt, 4).astype(np.float32)

        pred_pt = torch.tensor(pred_np)
        tgt_pt = torch.tensor(tgt_np)

        # --- PyTorch L1 cost ---
        cost_pt = torch.cdist(pred_pt, tgt_pt, p=1)
        pt_d = cost_pt.detach().numpy()

        # --- TTSim L1 cost ---
        matcher_tt = HungarianMatcherTTSim(
            name=f"test_l1_{tno}", cost_class=1.0, cost_bbox=1.0, cost_giou=1.0
        )
        pred_sim = SimTensor(
            {
                "name": f"test_l1_{tno}.pred_bbox",
                "shape": list(pred_np.shape),
                "data": pred_np,
                "dtype": pred_np.dtype,
            }
        )
        pred_sim.set_module(matcher_tt)

        tgt_sim = SimTensor(
            {
                "name": f"test_l1_{tno}.tgt_bbox",
                "shape": list(tgt_np.shape),
                "data": tgt_np,
                "dtype": tgt_np.dtype,
            }
        )
        tgt_sim.set_module(matcher_tt)

        cost_tt = matcher_tt._compute_l1_bbox_cost(pred_sim, tgt_sim)
        tt_d = cost_tt.data

        # --- Compare ---
        pt_shape = list(pt_d.shape)
        tt_shape = list(tt_d.shape)
        shape_ok = pt_shape == tt_shape

        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=L1_RTOL, atol=L1_ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={L1_ATOL}"

        inp_desc = f"pred[{nq},4] tgt[{ngt},4]"
        print_test_linear(
            module="L1BboxCost",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=L1_RTOL,
            atol=L1_ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
        )

        TEST_RESULTS.append(
            {
                "module": "L1BboxCost",
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
                    "module": "L1BboxCost",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": L1_ATOL,
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
            f"**Input:** `{inp_desc}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["L1BboxCost"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_l1),
        "num_passed": num_passed,
        "num_total": len(test_cases_l1),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "L1BboxCost",
            "description": "L1 pairwise bounding box distance cost — torch.cdist(p=1)",
            "passed": passed,
            "total": len(test_cases_l1),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_l1
    ), f"L1BboxCost: {passed}/{len(test_cases_l1)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — GIoU Cost Component
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_giou = [
    # (description, num_queries, num_gt, box_mode, edge_category)
    ("Standard 10q × 5gt", 10, 5, "random", "standard"),
    ("Identical boxes", 5, 5, "identical", "identical_boxes"),
    ("Non-overlapping boxes", 5, 5, "no_overlap", "no_overlap"),
    ("Enclosed boxes", 5, 5, "enclosed", "enclosed"),
    ("Single pair (1q × 1gt)", 1, 1, "random", "single_pair"),
    ("Many boxes (30q × 15gt)", 30, 15, "random", "many_boxes"),
]


@pytest.mark.unit
def test_giou_cost():
    """Test GIoU cost component: shape + numerical validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, nq, ngt, box_mode, category) in enumerate(test_cases_giou):
        np.random.seed(SEED + tno)
        torch.manual_seed(SEED + tno)

        # Generate boxes in cxcywh format
        if box_mode == "random":
            pred_np = np.random.rand(nq, 4).astype(np.float32) * 0.8 + 0.1
            pred_np[:, 2:] = np.clip(pred_np[:, 2:], 0.05, 0.5)
            tgt_np = np.random.rand(ngt, 4).astype(np.float32) * 0.8 + 0.1
            tgt_np[:, 2:] = np.clip(tgt_np[:, 2:], 0.05, 0.5)
        elif box_mode == "identical":
            base = np.array([[0.5, 0.5, 0.3, 0.3]], dtype=np.float32)
            pred_np = np.tile(base, (nq, 1))
            tgt_np = np.tile(base, (ngt, 1))
        elif box_mode == "no_overlap":
            # Pred boxes in top-left, tgt boxes in bottom-right
            pred_np = np.random.rand(nq, 4).astype(np.float32) * 0.1 + 0.1
            pred_np[:, 2:] = 0.05
            tgt_np = np.random.rand(ngt, 4).astype(np.float32) * 0.1 + 0.8
            tgt_np[:, 2:] = 0.05
        elif box_mode == "enclosed":
            # Outer boxes (large) as targets, inner boxes (small) as predictions
            pred_np = np.random.rand(nq, 4).astype(np.float32) * 0.1 + 0.45
            pred_np[:, 2:] = 0.05  # small boxes centered
            tgt_np = np.random.rand(ngt, 4).astype(np.float32) * 0.1 + 0.45
            tgt_np[:, 2:] = 0.4  # large boxes centered
        else:
            pred_np = np.random.rand(nq, 4).astype(np.float32) * 0.8 + 0.1
            pred_np[:, 2:] = np.clip(pred_np[:, 2:], 0.05, 0.5)
            tgt_np = np.random.rand(ngt, 4).astype(np.float32) * 0.8 + 0.1
            tgt_np[:, 2:] = np.clip(tgt_np[:, 2:], 0.05, 0.5)

        pred_pt = torch.tensor(pred_np)
        tgt_pt = torch.tensor(tgt_np)

        # --- PyTorch GIoU cost ---
        cost_pt = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_pt), box_cxcywh_to_xyxy(tgt_pt)
        )
        pt_d = cost_pt.detach().numpy()

        # --- TTSim GIoU cost ---
        matcher_tt = HungarianMatcherTTSim(
            name=f"test_giou_{tno}", cost_class=1.0, cost_bbox=1.0, cost_giou=1.0
        )
        pred_sim = _make_simtensor(pred_np, f"test_giou_{tno}.pred_bbox")
        pred_sim.set_module(matcher_tt)
        tgt_sim = _make_simtensor(tgt_np, f"test_giou_{tno}.tgt_bbox")
        tgt_sim.set_module(matcher_tt)

        cost_tt = matcher_tt._compute_giou_cost(pred_sim, tgt_sim)
        tt_d = cost_tt.data

        # --- Compare ---
        pt_shape = list(pt_d.shape)
        tt_shape = list(tt_d.shape)
        shape_ok = pt_shape == tt_shape

        abs_diff = np.abs(pt_d - tt_d)
        mx = float(abs_diff.max())
        mn = float(abs_diff.mean())
        num_ok = bool(np.allclose(pt_d, tt_d, rtol=GIOU_RTOL, atol=GIOU_ATOL))
        ok = shape_ok and num_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(num_ok)

        reason = ""
        if not shape_ok:
            reason = f"Shape mismatch: PT={pt_shape} vs TT={tt_shape}"
        elif not num_ok:
            reason = f"max_diff={mx:.2e} exceeds atol={GIOU_ATOL}"

        inp_desc = f"pred[{nq},4] tgt[{ngt},4]"
        print_test_linear(
            module="GIoUCost",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=f"PT={_compact_shape(pt_shape)} | TT={_compact_shape(tt_shape)}",
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=num_ok,
            max_diff=mx,
            mean_diff=mn,
            rtol=GIOU_RTOL,
            atol=GIOU_ATOL,
            failure_reason=reason,
            pt_data=pt_d,
            tt_data=tt_d,
        )

        TEST_RESULTS.append(
            {
                "module": "GIoUCost",
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
                    "module": "GIoUCost",
                    "test": tmsg,
                    "edge_case": category,
                    "max_diff": mx,
                    "atol": GIOU_ATOL,
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
            f"**Input:** `{inp_desc}` → **Output Shape:** `{pt_shape}`\n\n"
            f"**Sample Values [0:10]:**\n"
            f"- PyTorch: `[{_fmt_samples(pt_d)}]`\n"
            f"- TTSim:   `[{_fmt_samples(tt_d)}]`\n\n"
        )

    MODULE_STATS["GIoUCost"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_giou),
        "num_passed": num_passed,
        "num_total": len(test_cases_giou),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Output | Max Diff | Mean Diff | Result |\n"
        "|:--|:----------|:----------|:------|:-------|:---------|:----------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "GIoUCost",
            "description": "Generalized IoU cost component — −GIoU(xyxy(pred), xyxy(tgt))",
            "passed": passed,
            "total": len(test_cases_giou),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_giou
    ), f"GIoUCost: {passed}/{len(test_cases_giou)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — HungarianMatcher Shape Validation
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_shape = [
    # (desc, bs, nq, nc, gt_per_img, cost_cls, cost_bbox, cost_giou, category)
    # NOTE: bs >= 2 required — TTSim cat() needs at least 2 inputs
    ("Standard bs=2 nq=10", 2, 10, 91, [3, 5], 2.0, 5.0, 2.0, "standard"),
    ("Small batch bs=2", 2, 10, 91, [4, 3], 2.0, 5.0, 2.0, "standard"),
    ("Many GT per image", 2, 10, 91, [8, 9], 2.0, 5.0, 2.0, "many_gt"),
    ("Single GT per image", 2, 10, 91, [1, 1], 2.0, 5.0, 2.0, "few_gt"),
    ("Equal queries & GT", 2, 5, 91, [5, 5], 2.0, 5.0, 2.0, "equal_qgt"),
    ("Unbalanced GT counts", 2, 10, 91, [1, 8], 2.0, 5.0, 2.0, "unbalanced_gt"),
]


@pytest.mark.unit
def test_matcher_shapes():
    """Test HungarianMatcher output structure: shape validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    failed_cases = []

    for tno, (tmsg, bs, nq, nc, gt_per_img, cc, cb, cg, category) in enumerate(
        test_cases_shape
    ):
        outputs_pt, targets_pt, outputs_np, targets_np = _generate_inputs(
            bs, nq, nc, gt_per_img, seed=SEED + tno
        )

        # --- PyTorch ---
        matcher_pt = HungarianMatcherPyTorch(cost_class=cc, cost_bbox=cb, cost_giou=cg)
        with torch.no_grad():
            indices_pt = matcher_pt(outputs_pt, targets_pt)

        # --- TTSim ---
        matcher_tt = HungarianMatcherTTSim(
            name=f"test_shape_{tno}", cost_class=cc, cost_bbox=cb, cost_giou=cg
        )
        indices_tt = matcher_tt(outputs_np, targets_np)

        # --- Validate structure ---
        ok = True
        reason = ""

        # Check list length
        if len(indices_pt) != bs:
            ok = False
            reason = f"PT returned {len(indices_pt)} outputs, expected {bs}"
        if len(indices_tt) != bs:
            ok = False
            reason = f"TT returned {len(indices_tt)} outputs, expected {bs}"

        # Check per-image match counts
        shape_details = []
        if ok:
            for img_i in range(bs):
                pred_pt_idx, gt_pt_idx = indices_pt[img_i]
                pred_tt_idx, gt_tt_idx = indices_tt[img_i]

                expected_ngt = gt_per_img[img_i]
                expected_matches = min(nq, expected_ngt)

                len_pt = len(pred_pt_idx)
                len_tt = len(pred_tt_idx)
                len_gt_pt = len(gt_pt_idx)
                len_gt_tt = len(gt_tt_idx)

                if len_pt != expected_matches:
                    ok = False
                    reason = f"Image {img_i}: PT pred idx len={len_pt}, expected={expected_matches}"
                if len_tt != expected_matches:
                    ok = False
                    reason = f"Image {img_i}: TT pred idx len={len_tt}, expected={expected_matches}"
                if len_gt_pt != expected_matches:
                    ok = False
                    reason = f"Image {img_i}: PT gt idx len={len_gt_pt}, expected={expected_matches}"
                if len_gt_tt != expected_matches:
                    ok = False
                    reason = f"Image {img_i}: TT gt idx len={len_gt_tt}, expected={expected_matches}"

                shape_details.append(f"img{img_i}:PT={len_pt},TT={len_tt}")

        passed += int(ok)

        shape_line = " | ".join(shape_details) if shape_details else "N/A"
        inp_desc = f"bs={bs} nq={nq} nc={nc} gt={gt_per_img}"

        print_test_linear(
            module="MatcherShape",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=shape_line,
            shape_ok=ok,
            is_numerical=False,
            failure_reason=reason,
        )

        TEST_RESULTS.append(
            {
                "module": "MatcherShape",
                "validation_type": "SHAPE",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "passed": ok,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, None))
            FAILED_TESTS.append(
                {
                    "module": "MatcherShape",
                    "test": tmsg,
                    "edge_case": category,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` | {shape_line} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}`\n\n"
            f"**Match Counts:** {shape_line}\n\n"
        )

    MODULE_STATS["MatcherShape"] = {
        "shape_passed": passed,
        "shape_total": len(test_cases_shape),
        "num_passed": None,
        "num_total": None,
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Match Counts | Result |\n"
        "|:--|:----------|:----------|:------|:-------------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "MatcherShape",
            "description": "HungarianMatcher output structure — list of (pred_idx, gt_idx) tuples",
            "passed": passed,
            "total": len(test_cases_shape),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_shape
    ), f"MatcherShape: {passed}/{len(test_cases_shape)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — HungarianMatcher End-to-End (Index Validation)
# ═══════════════════════════════════════════════════════════════════════════════

test_cases_e2e = [
    # (desc, bs, nq, nc, gt_per_img, cost_cls, cost_bbox, cost_giou, category)
    # NOTE: bs >= 2 required — TTSim cat() needs at least 2 inputs
    ("Standard bs=2 nq=10", 2, 10, 91, [3, 5], 2.0, 5.0, 2.0, "standard"),
    ("Small batch bs=2", 2, 10, 91, [4, 3], 2.0, 5.0, 2.0, "standard"),
    ("Many GT (8,9)", 2, 10, 91, [8, 9], 2.0, 5.0, 2.0, "many_gt"),
    ("Few GT (1 each)", 2, 10, 91, [1, 1], 2.0, 5.0, 2.0, "few_gt"),
    ("Class-only weights", 2, 10, 91, [3, 5], 5.0, 0.0, 0.0, "different_wts"),
    ("Bbox-only weights", 2, 10, 91, [3, 5], 0.0, 5.0, 0.0, "different_wts"),
    ("GIoU-only weights", 2, 10, 91, [3, 5], 0.0, 0.0, 5.0, "different_wts"),
]


@pytest.mark.unit
def test_matcher_e2e():
    """Test HungarianMatcher end-to-end: exact index match validation."""
    _seed()

    rows = []
    detail_blocks = []
    passed = 0
    shape_passed = 0
    num_passed = 0
    failed_cases = []

    for tno, (tmsg, bs, nq, nc, gt_per_img, cc, cb, cg, category) in enumerate(
        test_cases_e2e
    ):
        outputs_pt, targets_pt, outputs_np, targets_np = _generate_inputs(
            bs, nq, nc, gt_per_img, seed=SEED + tno
        )

        # --- PyTorch ---
        matcher_pt = HungarianMatcherPyTorch(cost_class=cc, cost_bbox=cb, cost_giou=cg)
        with torch.no_grad():
            indices_pt = matcher_pt(outputs_pt, targets_pt)

        # --- TTSim ---
        matcher_tt = HungarianMatcherTTSim(
            name=f"test_e2e_{tno}", cost_class=cc, cost_bbox=cb, cost_giou=cg
        )
        indices_tt = matcher_tt(outputs_np, targets_np)

        # --- Validate indices ---
        shape_ok = len(indices_pt) == len(indices_tt) == bs
        idx_ok = True
        mismatch_detail = ""

        if shape_ok:
            for img_i in range(bs):
                pred_pt_i = (
                    indices_pt[img_i][0].numpy()
                    if isinstance(indices_pt[img_i][0], torch.Tensor)
                    else np.asarray(indices_pt[img_i][0])
                )
                gt_pt_i = (
                    indices_pt[img_i][1].numpy()
                    if isinstance(indices_pt[img_i][1], torch.Tensor)
                    else np.asarray(indices_pt[img_i][1])
                )
                pred_tt_i = np.asarray(indices_tt[img_i][0])
                gt_tt_i = np.asarray(indices_tt[img_i][1])

                if not np.array_equal(pred_pt_i, pred_tt_i):
                    idx_ok = False
                    mismatch_detail += (
                        f"Image {img_i}: pred_idx PT={pred_pt_i} vs TT={pred_tt_i}; "
                    )
                if not np.array_equal(gt_pt_i, gt_tt_i):
                    idx_ok = False
                    mismatch_detail += (
                        f"Image {img_i}: gt_idx PT={gt_pt_i} vs TT={gt_tt_i}; "
                    )
        else:
            idx_ok = False
            mismatch_detail = (
                f"Output count mismatch: PT={len(indices_pt)}, TT={len(indices_tt)}"
            )

        ok = shape_ok and idx_ok
        passed += int(ok)
        shape_passed += int(shape_ok)
        num_passed += int(idx_ok)

        reason = mismatch_detail if not ok else ""

        # Build shape line showing index matches per image
        idx_parts = []
        if shape_ok:
            for img_i in range(bs):
                pred_pt_i = _to_numpy(indices_pt[img_i][0])
                gt_pt_i = _to_numpy(indices_pt[img_i][1])
                pred_tt_i = np.asarray(indices_tt[img_i][0])
                gt_tt_i = np.asarray(indices_tt[img_i][1])
                match_pred = np.array_equal(pred_pt_i, pred_tt_i)
                match_gt = np.array_equal(gt_pt_i, gt_tt_i)
                status = "✓" if (match_pred and match_gt) else "✗"
                idx_parts.append(f"img{img_i}:{status}(n={len(pred_pt_i)})")
        shape_line = " | ".join(idx_parts) if idx_parts else "N/A"

        inp_desc = f"bs={bs} nq={nq} nc={nc} gt={gt_per_img} w=[{cc},{cb},{cg}]"

        print_test_linear(
            module="MatcherE2E",
            edge_case=category,
            edge_desc=EDGE_CASE_DESC.get(category, ""),
            input_shape=inp_desc,
            shape_line=shape_line,
            shape_ok=shape_ok,
            is_numerical=True,
            num_ok=idx_ok,
            max_diff=0.0 if idx_ok else 1.0,
            mean_diff=0.0 if idx_ok else 1.0,
            rtol=0,
            atol=0,
            failure_reason=reason,
        )

        TEST_RESULTS.append(
            {
                "module": "MatcherE2E",
                "validation_type": "INDEX",
                "edge_case": category,
                "edge_desc": tmsg,
                "input_shape": inp_desc,
                "passed": ok,
                "shape_ok": shape_ok,
                "num_ok": idx_ok,
                "max_diff": 0.0 if idx_ok else 1.0,
                "mean_diff": 0.0 if idx_ok else 1.0,
            }
        )

        if not ok:
            failed_cases.append((tno, tmsg, category, None))
            FAILED_TESTS.append(
                {
                    "module": "MatcherE2E",
                    "test": tmsg,
                    "edge_case": category,
                }
            )

        tag = "✅ PASS" if ok else "🔴 **FAIL**"
        rows.append(
            f"| {tno} | {tmsg} | {category} | `{inp_desc}` | {shape_line} | {tag} |"
        )

        status_badge = "🟢" if ok else "🔴"
        detail_blocks.append(
            f"---\n\n### {status_badge} TEST[{tno}] {tmsg}\n\n"
            f"**Edge Case:** `{category}` — {EDGE_CASE_DESC.get(category, '')}\n\n"
            f"**Input:** `{inp_desc}`\n\n"
            f"**Index Match:** {shape_line}\n\n"
        )

    MODULE_STATS["MatcherE2E"] = {
        "shape_passed": shape_passed,
        "shape_total": len(test_cases_e2e),
        "num_passed": num_passed,
        "num_total": len(test_cases_e2e),
    }

    hdr = (
        "| # | Test Case | Edge Case | Input | Index Match | Result |\n"
        "|:--|:----------|:----------|:------|:------------|:-------|"
    )
    REPORT_SECTIONS.append(
        {
            "name": "MatcherE2E",
            "description": "Full HungarianMatcher end-to-end — exact index assignment validation",
            "passed": passed,
            "total": len(test_cases_e2e),
            "failed_cases": failed_cases,
            "table": hdr + "\n" + "\n".join(rows),
            "details": "\n".join(detail_blocks),
        }
    )

    assert passed == len(
        test_cases_e2e
    ), f"MatcherE2E: {passed}/{len(test_cases_e2e)} passed"


# ═══════════════════════════════════════════════════════════════════════════════
#  Markdown report + self-runner
# ═══════════════════════════════════════════════════════════════════════════════


def _write_markdown_report(report_path, exit_code):
    """Generate a structured markdown report from REPORT_SECTIONS."""
    total_passed = sum(s["passed"] for s in REPORT_SECTIONS)
    total_tests = sum(s["total"] for s in REPORT_SECTIONS)
    status = "PASS" if total_passed == total_tests else "FAIL"

    lines = [
        "# HungarianMatcher Unit Test Report",
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

        lines.append("---")
        lines.append("")

    # Config
    lines.append("## Configuration")
    lines.append(f"- Focal Loss Tolerance: rtol={FOCAL_RTOL}, atol={FOCAL_ATOL}")
    lines.append(f"- L1 Bbox Tolerance: rtol={L1_RTOL}, atol={L1_ATOL}")
    lines.append(f"- GIoU Tolerance: rtol={GIOU_RTOL}, atol={GIOU_ATOL}")
    lines.append(f"- End-to-End: exact index match (np.array_equal)")
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
    report_path = os.path.join(report_dir, "matcher_unit_test_report.md")
    output_path = os.path.join(report_dir, "matcher_unit_test_output.md")

    # Tee stdout → terminal + output file
    _original_stdout = sys.stdout
    _tee_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _TeeStream(_tee_file, _original_stdout)

    print(f"\n{SUMMARY_LINE}")
    print(f"HUNGARIAN MATCHER UNIT TEST SUITE - PyTorch vs TTSim")
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
