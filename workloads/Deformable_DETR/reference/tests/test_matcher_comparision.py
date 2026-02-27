#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Test Suite: PyTorch vs TTSim HungarianMatcher Validation

Purpose:
    Validates functional equivalence between PyTorch and TTSim implementations
    of HungarianMatcher module for bipartite matching in Deformable DETR.

Test Scope:
    - Shape correctness (output tensor dimensions)
    - Numerical equivalence (value comparison with tolerances)
    - Component-level validation (focal loss, L1, GIoU, Hungarian matching)

Architecture:
    - Tests exist outside TTSim core library
    - No impact on graph construction, scheduling, or simulation semantics
    - Numerical execution enabled only within test context
    - Deterministic inputs with fixed random seeds

Test Organization:
    1. Shape Validation: test_*_shape
       - Verifies output tensor ranks and dimension sizes
       - No numerical computation required

    2. Numerical Validation: test_*_numerical
       - Compares PyTorch vs TTSim numerical outputs
       - Uses explicit tolerance thresholds (atol, rtol)
       - Reports detailed diagnostics on mismatch

    3. Component Tests: test_*_component_*
       - Validates individual cost computation functions
       - Ensures decomposition preserves numerical equivalence

Execution:
    python workloads/Deformable_DETR/tests/test_matcher_validation.py

Output:
    - Console output with detailed comparison
    - Markdown report saved to workloads/Deformable_DETR/reports/
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add polaris root to path
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import PyTorch implementation
from workloads.Deformable_DETR.reference.matcher import (
    HungarianMatcher as PyTorchMatcher,
)
from workloads.Deformable_DETR.reference.box_ops import (
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)

# Import TTSim implementation
from workloads.Deformable_DETR.models.matcher_ttsim import (
    HungarianMatcher as TTSimMatcher,
)
from ttsim.ops.tensor import SimTensor

# ============================================================================
# Test Reporting Infrastructure
# ============================================================================


class TestReport:
    """Manages test execution reporting and markdown generation."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.console_width = 80

    def section(self, title):
        """Print section header."""
        print("\n" + "=" * self.console_width)
        print(f"  {title}")
        print("=" * self.console_width)

    def subsection(self, title):
        """Print subsection header."""
        print(f"\n{title}")
        print("-" * len(title))

    def log(self, message, indent=0):
        """Print log message with optional indentation."""
        prefix = "  " * indent
        print(f"{prefix}{message}")

    def log_tensor_info(self, name, tensor, indent=0, show_samples=False):
        """Log tensor information."""
        prefix = "  " * indent
        if isinstance(tensor, torch.Tensor):
            print(f"{prefix}{name}:")
            print(f"{prefix}  Shape: {list(tensor.shape)}")
            print(f"{prefix}  Dtype: {tensor.dtype}")
            print(f"{prefix}  Device: {tensor.device}")
            print(
                f"{prefix}  Min/Max: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]"
            )
            print(f"{prefix}  Mean: {tensor.mean().item():.4f}")
            if show_samples and tensor.numel() > 0:
                flat = tensor.flatten()
                samples = flat[: min(5, len(flat))].detach().numpy()
                print(f"{prefix}  Sample values: {samples}")
        elif isinstance(tensor, np.ndarray):
            print(f"{prefix}{name}:")
            print(f"{prefix}  Shape: {list(tensor.shape)}")
            print(f"{prefix}  Dtype: {tensor.dtype}")
            print(f"{prefix}  Min/Max: [{tensor.min():.4f}, {tensor.max():.4f}]")
            print(f"{prefix}  Mean: {tensor.mean():.4f}")
            if show_samples and tensor.size > 0:
                flat = tensor.flatten()
                samples = flat[: min(5, len(flat))]
                print(f"{prefix}  Sample values: {samples}")
        elif isinstance(tensor, SimTensor):
            print(f"{prefix}{name}:")
            print(f"{prefix}  Shape: {tensor.shape}")
            print(f"{prefix}  Dtype: {tensor.dtype}")
            if tensor.data is not None:
                print(
                    f"{prefix}  Min/Max: [{tensor.data.min():.4f}, {tensor.data.max():.4f}]"
                )
                print(f"{prefix}  Mean: {tensor.data.mean():.4f}")
                if show_samples and tensor.data.size > 0:
                    flat = tensor.data.flatten()
                    samples = flat[: min(5, len(flat))]
                    print(f"{prefix}  Sample values: {samples}")
            else:
                print(f"{prefix}  Data: None (shape-only mode)")

    def add_result(self, test_name, status, message, details=None):
        """Add test result."""
        self.results.append(
            {
                "test_name": test_name,
                "status": status,
                "message": message,
                "details": details or {},
            }
        )

        # Print immediate result
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"\n{status_symbol} {test_name}: {status}")
        print(f"  {message}")

    def generate_markdown_report(self, output_path):
        """Generate markdown report from test results."""
        report_lines = []
        report_lines.append("# HungarianMatcher Validation Report")
        report_lines.append(
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(
            f"\n**Duration:** {(datetime.now() - self.start_time).total_seconds():.2f}s"
        )

        # Summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = total - passed

        report_lines.append("\n## Summary")
        report_lines.append(f"\n- **Total Tests:** {total}")
        report_lines.append(f"- **Passed:** {passed}")
        report_lines.append(f"- **Failed:** {failed}")
        report_lines.append(f"- **Success Rate:** {(passed/total*100):.1f}%")

        # Detailed Results
        report_lines.append("\n## Test Results")

        for result in self.results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            report_lines.append(f"\n### {status_emoji} {result['test_name']}")
            report_lines.append(f"\n**Status:** {result['status']}")
            report_lines.append(f"\n**Message:** {result['message']}")

            if result["details"]:
                report_lines.append("\n**Details:**")
                for key, value in result["details"].items():
                    if isinstance(value, dict):
                        report_lines.append(f"\n- **{key}:**")
                        for k, v in value.items():
                            report_lines.append(f"  - {k}: `{v}`")
                    else:
                        report_lines.append(f"- **{key}:** `{value}`")

        # Recommendations
        report_lines.append("\n## Recommendations")
        if failed == 0:
            report_lines.append(
                "\n✅ All tests passed! The TTSim implementation is functionally equivalent to PyTorch."
            )
        else:
            report_lines.append(
                "\n⚠️ Some tests failed. Please review the details above and:"
            )
            report_lines.append("- Check tensor shapes and dtypes")
            report_lines.append("- Verify operation decomposition correctness")
            report_lines.append(
                "- Review tolerance thresholds for numerical comparisons"
            )

        # Write report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(report_lines), encoding="utf-8")

        return output_path


# ============================================================================
# Input Generation
# ============================================================================


def generate_deterministic_inputs(
    batch_size=2, num_queries=10, num_classes=91, num_gt_per_image=[3, 5], seed=42
):
    """
    Generate deterministic input tensors for matcher testing.

    Args:
        batch_size: Number of images in batch
        num_queries: Number of predicted queries per image
        num_classes: Number of object classes
        num_gt_per_image: List of ground truth counts per image
        seed: Random seed for reproducibility

    Returns:
        outputs_pt: Dict with pred_logits and pred_boxes (PyTorch tensors)
        targets_pt: List of target dicts (PyTorch tensors)
        outputs_ttsim: Dict with pred_logits and pred_boxes (NumPy arrays)
        targets_ttsim: List of target dicts (NumPy arrays)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate predictions (PyTorch)
    pred_logits_pt = torch.randn(
        batch_size, num_queries, num_classes, dtype=torch.float32
    )
    pred_boxes_pt = torch.rand(batch_size, num_queries, 4, dtype=torch.float32)
    # Ensure boxes are in valid [0, 1] range for cxcywh format
    pred_boxes_pt[:, :, :2] = torch.sigmoid(pred_boxes_pt[:, :, :2])  # center
    pred_boxes_pt[:, :, 2:] = torch.sigmoid(pred_boxes_pt[:, :, 2:])  # width/height

    outputs_pt = {"pred_logits": pred_logits_pt, "pred_boxes": pred_boxes_pt}

    # Generate targets (PyTorch)
    targets_pt = []
    for i in range(batch_size):
        num_gt = num_gt_per_image[i]
        labels = torch.randint(0, num_classes, (num_gt,), dtype=torch.int64)
        boxes = torch.rand(num_gt, 4, dtype=torch.float32)
        boxes[:, :2] = torch.sigmoid(boxes[:, :2])  # center
        boxes[:, 2:] = torch.sigmoid(boxes[:, 2:])  # width/height
        targets_pt.append({"labels": labels, "boxes": boxes})

    # Convert to NumPy for TTSim
    outputs_ttsim = {
        "pred_logits": pred_logits_pt.numpy(),
        "pred_boxes": pred_boxes_pt.numpy(),
    }

    targets_ttsim = []
    for t in targets_pt:
        targets_ttsim.append(
            {"labels": t["labels"].numpy(), "boxes": t["boxes"].numpy()}
        )

    return outputs_pt, targets_pt, outputs_ttsim, targets_ttsim


# ============================================================================
# Test 1: Full HungarianMatcher Shape Validation
# ============================================================================


def test_hungarian_matcher_shape(report):
    """
    Test: HungarianMatcher shape correctness

    Validates:
        - Output structure (list of tuples)
        - Number of matches per image
        - Index array shapes and dtypes

    Execution Mode: Shape-only (no numerical comparison)
    """
    report.section("TEST 1: HungarianMatcher Shape Validation")

    # Configuration
    batch_size = 2
    num_queries = 10
    num_classes = 91
    num_gt_per_image = [3, 5]
    seed = 42

    cost_class = 2.0
    cost_bbox = 5.0
    cost_giou = 2.0

    report.subsection("Configuration")
    report.log(f"Batch Size: {batch_size}")
    report.log(f"Num Queries: {num_queries}")
    report.log(f"Num Classes: {num_classes}")
    report.log(f"GT per Image: {num_gt_per_image}")
    report.log(f"Cost Weights: class={cost_class}, bbox={cost_bbox}, giou={cost_giou}")

    # Generate inputs
    report.subsection("Input Generation")
    outputs_pt, targets_pt, outputs_ttsim, targets_ttsim = (
        generate_deterministic_inputs(
            batch_size=batch_size,
            num_queries=num_queries,
            num_classes=num_classes,
            num_gt_per_image=num_gt_per_image,
            seed=seed,
        )
    )

    report.log("PyTorch Inputs:")
    report.log_tensor_info(
        "pred_logits", outputs_pt["pred_logits"], indent=1, show_samples=True
    )
    report.log_tensor_info(
        "pred_boxes", outputs_pt["pred_boxes"], indent=1, show_samples=True
    )

    # Print sample predictions for first image
    report.log("\n  Sample Predictions (Image 0, Query 0):")
    report.log(
        f"    Logits: {outputs_pt['pred_logits'][0, 0, :5].numpy()}... (showing first 5 classes)"
    )
    report.log(f"    Box (cxcywh): {outputs_pt['pred_boxes'][0, 0].numpy()}")

    # Print sample targets
    report.log("\n  Sample Targets:")
    for i, target in enumerate(targets_pt):
        report.log(f"    Image {i}: {len(target['labels'])} objects")
        if len(target["labels"]) > 0:
            report.log(
                f"      Object 0: class={target['labels'][0].item()}, box={target['boxes'][0].numpy()}"
            )

    report.log("\nTTSim Inputs:")
    report.log_tensor_info(
        "pred_logits", outputs_ttsim["pred_logits"], indent=1, show_samples=True
    )
    report.log_tensor_info(
        "pred_boxes", outputs_ttsim["pred_boxes"], indent=1, show_samples=True
    )

    # Create matchers
    report.subsection("Matcher Initialization")
    matcher_pt = PyTorchMatcher(
        cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou
    )
    matcher_ttsim = TTSimMatcher(
        name="test_matcher",
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )
    report.log("✓ PyTorch matcher created")
    report.log("✓ TTSim matcher created")

    # Run matchers
    report.subsection("Forward Pass")
    try:
        with torch.no_grad():
            indices_pt = matcher_pt(outputs_pt, targets_pt)
        report.log("✓ PyTorch forward pass completed")

        indices_ttsim = matcher_ttsim(outputs_ttsim, targets_ttsim)
        report.log("✓ TTSim forward pass completed")
    except Exception as e:
        report.add_result(
            "HungarianMatcher Shape",
            "FAIL",
            f"Forward pass failed: {str(e)}",
            {"error": str(e)},
        )
        return

    # Validate shapes
    report.subsection("Shape Validation")

    # Check output structure
    if len(indices_pt) != batch_size:
        report.add_result(
            "HungarianMatcher Shape",
            "FAIL",
            f"PyTorch: Expected {batch_size} outputs, got {len(indices_pt)}",
            {"expected": batch_size, "actual": len(indices_pt)},
        )
        return

    if len(indices_ttsim) != batch_size:
        report.add_result(
            "HungarianMatcher Shape",
            "FAIL",
            f"TTSim: Expected {batch_size} outputs, got {len(indices_ttsim)}",
            {"expected": batch_size, "actual": len(indices_ttsim)},
        )
        return

    report.log(f"✓ Both implementations returned {batch_size} outputs")

    # Check per-image matches
    shape_match = True
    for i in range(batch_size):
        pred_idx_pt, gt_idx_pt = indices_pt[i]
        pred_idx_ttsim, gt_idx_ttsim = indices_ttsim[i]

        report.log(f"\nImage {i}:")
        report.log(f"  Ground Truth Count: {num_gt_per_image[i]}")
        report.log(f"  PyTorch Matches: {len(pred_idx_pt)}")
        report.log(f"  TTSim Matches: {len(pred_idx_ttsim)}")

        # Show actual matched indices
        if isinstance(pred_idx_pt, torch.Tensor):
            pred_pt_arr = pred_idx_pt.cpu().numpy()
            gt_pt_arr = gt_idx_pt.cpu().numpy()
        else:
            pred_pt_arr = pred_idx_pt
            gt_pt_arr = gt_idx_pt
        report.log(f"  PyTorch: pred_idx={pred_pt_arr}, gt_idx={gt_pt_arr}")
        report.log(f"  TTSim:   pred_idx={pred_idx_ttsim}, gt_idx={gt_idx_ttsim}")

        if len(pred_idx_pt) != len(pred_idx_ttsim):
            report.log(f"  ✗ Match count mismatch!")
            shape_match = False
        else:
            report.log(f"  ✓ Match counts agree")

    if shape_match:
        report.add_result(
            "HungarianMatcher Shape",
            "PASS",
            "All shape validations passed",
            {
                "batch_size": batch_size,
                "num_queries": num_queries,
                "matches_per_image": [len(indices_pt[i][0]) for i in range(batch_size)],
            },
        )
    else:
        report.add_result(
            "HungarianMatcher Shape",
            "FAIL",
            "Shape mismatch detected",
            {"batch_size": batch_size, "num_queries": num_queries},
        )


# ============================================================================
# Test 2: Full HungarianMatcher Numerical Validation
# ============================================================================


def test_hungarian_matcher_numerical(report):
    """
    Test: HungarianMatcher numerical equivalence

    Validates:
        - Exact matching between PyTorch and TTSim implementations
        - Hungarian algorithm produces identical assignments
        - End-to-end pipeline preserves numerical accuracy

    Execution Mode: Numerical (data_compute enabled in test context)
    """
    report.section("TEST 2: HungarianMatcher Numerical Validation")

    # Configuration
    batch_size = 2
    num_queries = 10
    num_classes = 91
    num_gt_per_image = [3, 5]
    seed = 42

    cost_class = 2.0
    cost_bbox = 5.0
    cost_giou = 2.0

    # Generate inputs
    report.subsection("Input Generation")
    outputs_pt, targets_pt, outputs_ttsim, targets_ttsim = (
        generate_deterministic_inputs(
            batch_size=batch_size,
            num_queries=num_queries,
            num_classes=num_classes,
            num_gt_per_image=num_gt_per_image,
            seed=seed,
        )
    )

    # Create matchers
    report.subsection("Matcher Execution")
    matcher_pt = PyTorchMatcher(
        cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou
    )
    matcher_ttsim = TTSimMatcher(
        name="test_matcher",
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )

    # Run matchers
    try:
        with torch.no_grad():
            indices_pt = matcher_pt(outputs_pt, targets_pt)

        indices_ttsim = matcher_ttsim(outputs_ttsim, targets_ttsim)
    except Exception as e:
        report.add_result(
            "HungarianMatcher Numerical",
            "FAIL",
            f"Execution failed: {str(e)}",
            {"error": str(e)},
        )
        return

    # Compare indices
    report.subsection("Numerical Comparison")

    all_match = True
    comparison_details = {}

    for i in range(batch_size):
        pred_idx_pt, gt_idx_pt = indices_pt[i]
        pred_idx_ttsim, gt_idx_ttsim = indices_ttsim[i]

        # Convert to numpy for comparison
        if isinstance(pred_idx_pt, torch.Tensor):
            pred_idx_pt = pred_idx_pt.cpu().numpy()
        if isinstance(gt_idx_pt, torch.Tensor):
            gt_idx_pt = gt_idx_pt.cpu().numpy()

        report.log(f"\nImage {i}:")
        report.log(f"  PyTorch pred_indices: {pred_idx_pt}")
        report.log(f"  TTSim pred_indices:   {pred_idx_ttsim}")
        report.log(f"  PyTorch gt_indices: {gt_idx_pt}")
        report.log(f"  TTSim gt_indices:   {gt_idx_ttsim}")

        # Compare
        pred_match = np.array_equal(pred_idx_pt, pred_idx_ttsim)
        gt_match = np.array_equal(gt_idx_pt, gt_idx_ttsim)

        if pred_match and gt_match:
            report.log(f"  ✓ Indices match exactly")
            comparison_details[f"image_{i}"] = "MATCH"
        else:
            report.log(f"  ✗ Indices mismatch!")
            if not pred_match:
                report.log(f"    - pred_indices differ")
            if not gt_match:
                report.log(f"    - gt_indices differ")
            all_match = False
            comparison_details[f"image_{i}"] = "MISMATCH"

    if all_match:
        report.add_result(
            "HungarianMatcher Numerical",
            "PASS",
            "All indices match exactly between PyTorch and TTSim",
            comparison_details,
        )
    else:
        report.add_result(
            "HungarianMatcher Numerical",
            "FAIL",
            "Index mismatch detected",
            comparison_details,
        )


# ============================================================================
# Test 3: Focal Loss Cost Component
# ============================================================================


def test_focal_loss_cost_component(report):
    """
    Test: Focal Loss cost computation numerical equivalence

    Validates:
        - TTSim decomposition of focal loss matches PyTorch compact form
        - Primitive ops (Pow, Log, Mul, Sub, Add) preserve numerical accuracy

    Tolerance: atol=1e-5, rtol=1e-4
    """
    report.section("TEST 3: Focal Loss Cost Component")

    # Configuration
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = 2
    num_queries = 10
    num_classes = 91
    num_gt = 5

    report.subsection("Configuration")
    report.log(f"Batch Size: {batch_size}")
    report.log(f"Num Queries: {num_queries}")
    report.log(f"Num Classes: {num_classes}")
    report.log(f"Num Ground Truth: {num_gt}")
    report.log(f"Alpha: 0.25, Gamma: 2.0")

    # Generate inputs
    report.subsection("Input Generation")
    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    tgt_ids = torch.randint(0, num_classes, (num_gt,), dtype=torch.int64)

    report.log_tensor_info("PyTorch pred_logits", pred_logits, show_samples=True)
    report.log(f"Target IDs: {tgt_ids.numpy()}")
    report.log(
        f"Sample logits (query 0, first 5 classes): {pred_logits[0, 0, :5].numpy()}"
    )

    # PyTorch focal loss computation
    report.subsection("PyTorch Computation")
    out_prob = pred_logits.flatten(0, 1).sigmoid()

    alpha = 0.25
    gamma = 2.0

    neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    cost_class_pt = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

    report.log_tensor_info("PyTorch focal_loss_cost", cost_class_pt, show_samples=True)
    report.log(
        f"Sample costs (query 0, first 3 targets): {cost_class_pt[0, :3].numpy()}"
    )

    # TTSim focal loss computation
    report.subsection("TTSim Computation")
    matcher_ttsim = TTSimMatcher(
        name="test_focal_matcher", cost_class=1.0, cost_bbox=0.0, cost_giou=0.0
    )

    # Convert inputs
    out_prob_np = out_prob.detach().numpy()
    tgt_ids_np = tgt_ids.numpy()

    out_prob_sim = SimTensor(
        {
            "name": "test_focal_matcher.out_prob",
            "shape": list(out_prob_np.shape),
            "data": out_prob_np,
            "dtype": out_prob_np.dtype,
        }
    )
    out_prob_sim.set_module(matcher_ttsim)

    tgt_ids_sim = SimTensor(
        {
            "name": "test_focal_matcher.tgt_ids",
            "shape": list(tgt_ids_np.shape),
            "data": tgt_ids_np,
            "dtype": tgt_ids_np.dtype,
        }
    )
    tgt_ids_sim.set_module(matcher_ttsim)

    # Compute focal loss via TTSim
    cost_class_ttsim = matcher_ttsim._compute_focal_loss_cost(out_prob_sim, tgt_ids_sim)

    report.log_tensor_info("TTSim focal_loss_cost", cost_class_ttsim, show_samples=True)
    if cost_class_ttsim.data is not None:
        report.log(
            f"Sample costs (query 0, first 3 targets): {cost_class_ttsim.data[0, :3]}"
        )

    # Compare
    report.subsection("Numerical Comparison")
    cost_class_pt_np = cost_class_pt.detach().numpy()
    cost_class_ttsim_np = cost_class_ttsim.data

    report.log(f"PyTorch shape: {cost_class_pt_np.shape}")
    report.log(f"TTSim shape:   {cost_class_ttsim_np.shape}")

    abs_diff = np.abs(cost_class_pt_np - cost_class_ttsim_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    report.log(f"\nDifference Statistics:")
    report.log(f"  Max Absolute Difference: {max_diff:.6e}")
    report.log(f"  Mean Absolute Difference: {mean_diff:.6e}")

    atol = 1e-5
    rtol = 1e-4

    try:
        np.testing.assert_allclose(
            cost_class_ttsim_np, cost_class_pt_np, atol=atol, rtol=rtol
        )
        report.log(f"  ✓ Within tolerance (atol={atol}, rtol={rtol})")
        report.add_result(
            "Focal Loss Cost Component",
            "PASS",
            f"Numerical match within tolerance (max_diff={max_diff:.6e})",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
            },
        )
    except AssertionError as e:
        report.log(f"  ✗ Tolerance exceeded!")
        report.add_result(
            "Focal Loss Cost Component",
            "FAIL",
            f"Numerical mismatch (max_diff={max_diff:.6e} > tolerance)",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
                "error": str(e),
            },
        )


# ============================================================================
# Test 4: L1 Bbox Cost Component
# ============================================================================


def test_l1_bbox_cost_component(report):
    """
    Test: L1 bounding box cost computation numerical equivalence

    Validates:
        - F.Cdist(p=1.0) matches torch.cdist(p=1)
        - Pairwise L1 distance computation is correct

    Tolerance: atol=1e-6, rtol=1e-5
    """
    report.section("TEST 4: L1 Bbox Cost Component")

    # Configuration
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = 2
    num_queries = 10
    num_gt = 5

    report.subsection("Configuration")
    report.log(f"Batch Size: {batch_size}")
    report.log(f"Num Queries: {num_queries}")
    report.log(f"Num Ground Truth: {num_gt}")

    # Generate box coordinates
    report.subsection("Input Generation")
    pred_boxes = torch.rand(batch_size * num_queries, 4)
    tgt_boxes = torch.rand(num_gt, 4)

    report.log_tensor_info("PyTorch pred_boxes", pred_boxes, show_samples=True)
    report.log_tensor_info("PyTorch tgt_boxes", tgt_boxes, show_samples=True)
    report.log(f"Sample pred_box (query 0): {pred_boxes[0].numpy()}")
    report.log(f"Sample tgt_box (gt 0): {tgt_boxes[0].numpy()}")

    # PyTorch L1 cost
    report.subsection("PyTorch Computation")
    cost_bbox_pt = torch.cdist(pred_boxes, tgt_boxes, p=1)
    report.log_tensor_info("PyTorch L1 cost", cost_bbox_pt, show_samples=True)
    report.log(
        f"Sample L1 distances (query 0 to first 3 targets): {cost_bbox_pt[0, :3].numpy()}"
    )

    # TTSim L1 cost
    report.subsection("TTSim Computation")
    matcher_ttsim = TTSimMatcher(
        name="test_l1_matcher", cost_class=0.0, cost_bbox=1.0, cost_giou=0.0
    )

    pred_boxes_np = pred_boxes.numpy()
    tgt_boxes_np = tgt_boxes.numpy()

    pred_boxes_sim = SimTensor(
        {
            "name": "test_l1_matcher.pred_boxes",
            "shape": list(pred_boxes_np.shape),
            "data": pred_boxes_np,
            "dtype": pred_boxes_np.dtype,
        }
    )
    pred_boxes_sim.set_module(matcher_ttsim)

    tgt_boxes_sim = SimTensor(
        {
            "name": "test_l1_matcher.tgt_boxes",
            "shape": list(tgt_boxes_np.shape),
            "data": tgt_boxes_np,
            "dtype": tgt_boxes_np.dtype,
        }
    )
    tgt_boxes_sim.set_module(matcher_ttsim)

    cost_bbox_ttsim = matcher_ttsim._compute_l1_bbox_cost(pred_boxes_sim, tgt_boxes_sim)
    report.log_tensor_info("TTSim L1 cost", cost_bbox_ttsim, show_samples=True)
    if cost_bbox_ttsim.data is not None:
        report.log(
            f"Sample L1 distances (query 0 to first 3 targets): {cost_bbox_ttsim.data[0, :3]}"
        )

    # Compare
    report.subsection("Numerical Comparison")
    cost_bbox_pt_np = cost_bbox_pt.detach().numpy()
    cost_bbox_ttsim_np = cost_bbox_ttsim.data

    report.log(f"PyTorch shape: {cost_bbox_pt_np.shape}")
    report.log(f"TTSim shape:   {cost_bbox_ttsim_np.shape}")

    abs_diff = np.abs(cost_bbox_pt_np - cost_bbox_ttsim_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    report.log(f"\nDifference Statistics:")
    report.log(f"  Max Absolute Difference: {max_diff:.6e}")
    report.log(f"  Mean Absolute Difference: {mean_diff:.6e}")

    atol = 1e-6
    rtol = 1e-5

    try:
        np.testing.assert_allclose(
            cost_bbox_ttsim_np, cost_bbox_pt_np, atol=atol, rtol=rtol
        )
        report.log(f"  ✓ Within tolerance (atol={atol}, rtol={rtol})")
        report.add_result(
            "L1 Bbox Cost Component",
            "PASS",
            f"Numerical match within tolerance (max_diff={max_diff:.6e})",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
            },
        )
    except AssertionError as e:
        report.log(f"  ✗ Tolerance exceeded!")
        report.add_result(
            "L1 Bbox Cost Component",
            "FAIL",
            f"Numerical mismatch (max_diff={max_diff:.6e} > tolerance)",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
                "error": str(e),
            },
        )


# ============================================================================
# Test 5: GIoU Cost Component
# ============================================================================


def test_giou_cost_component(report):
    """
    Test: GIoU cost computation numerical equivalence

    Validates:
        - box_ops_ttsim.generalized_box_iou matches PyTorch implementation
        - Box format conversion (cxcywh → xyxy) preserves correctness

    Tolerance: atol=1e-5, rtol=1e-4
    """
    report.section("TEST 5: GIoU Cost Component")

    # Configuration
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size = 2
    num_queries = 10
    num_gt = 5

    report.subsection("Configuration")
    report.log(f"Batch Size: {batch_size}")
    report.log(f"Num Queries: {num_queries}")
    report.log(f"Num Ground Truth: {num_gt}")

    # Generate boxes in cxcywh format
    report.subsection("Input Generation")
    pred_boxes = torch.rand(batch_size * num_queries, 4)
    tgt_boxes = torch.rand(num_gt, 4)

    report.log_tensor_info("PyTorch pred_boxes (cxcywh)", pred_boxes, show_samples=True)
    report.log_tensor_info("PyTorch tgt_boxes (cxcywh)", tgt_boxes, show_samples=True)
    report.log(f"Sample pred_box (query 0, cxcywh): {pred_boxes[0].numpy()}")
    report.log(f"Sample tgt_box (gt 0, cxcywh): {tgt_boxes[0].numpy()}")

    # PyTorch GIoU cost
    report.subsection("PyTorch Computation")
    cost_giou_pt = -generalized_box_iou(
        box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes)
    )
    report.log_tensor_info("PyTorch GIoU cost", cost_giou_pt, show_samples=True)
    report.log(
        f"Sample GIoU costs (query 0 to first 3 targets): {cost_giou_pt[0, :3].numpy()}"
    )

    # TTSim GIoU cost
    report.subsection("TTSim Computation")
    matcher_ttsim = TTSimMatcher(
        name="test_giou_matcher", cost_class=0.0, cost_bbox=0.0, cost_giou=1.0
    )

    pred_boxes_np = pred_boxes.numpy()
    tgt_boxes_np = tgt_boxes.numpy()

    pred_boxes_sim = SimTensor(
        {
            "name": "test_giou_matcher.pred_boxes",
            "shape": list(pred_boxes_np.shape),
            "data": pred_boxes_np,
            "dtype": pred_boxes_np.dtype,
        }
    )
    pred_boxes_sim.set_module(matcher_ttsim)

    tgt_boxes_sim = SimTensor(
        {
            "name": "test_giou_matcher.tgt_boxes",
            "shape": list(tgt_boxes_np.shape),
            "data": tgt_boxes_np,
            "dtype": tgt_boxes_np.dtype,
        }
    )
    tgt_boxes_sim.set_module(matcher_ttsim)

    cost_giou_ttsim = matcher_ttsim._compute_giou_cost(pred_boxes_sim, tgt_boxes_sim)
    report.log_tensor_info("TTSim GIoU cost", cost_giou_ttsim, show_samples=True)
    if cost_giou_ttsim.data is not None:
        report.log(
            f"Sample GIoU costs (query 0 to first 3 targets): {cost_giou_ttsim.data[0, :3]}"
        )

    # Compare
    report.subsection("Numerical Comparison")
    cost_giou_pt_np = cost_giou_pt.detach().numpy()
    cost_giou_ttsim_np = cost_giou_ttsim.data

    report.log(f"PyTorch shape: {cost_giou_pt_np.shape}")
    report.log(f"TTSim shape:   {cost_giou_ttsim_np.shape}")

    abs_diff = np.abs(cost_giou_pt_np - cost_giou_ttsim_np)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    report.log(f"\nDifference Statistics:")
    report.log(f"  Max Absolute Difference: {max_diff:.6e}")
    report.log(f"  Mean Absolute Difference: {mean_diff:.6e}")

    atol = 1e-5
    rtol = 1e-4

    try:
        np.testing.assert_allclose(
            cost_giou_ttsim_np, cost_giou_pt_np, atol=atol, rtol=rtol
        )
        report.log(f"  ✓ Within tolerance (atol={atol}, rtol={rtol})")
        report.add_result(
            "GIoU Cost Component",
            "PASS",
            f"Numerical match within tolerance (max_diff={max_diff:.6e})",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
            },
        )
    except AssertionError as e:
        report.log(f"  ✗ Tolerance exceeded!")
        report.add_result(
            "GIoU Cost Component",
            "FAIL",
            f"Numerical mismatch (max_diff={max_diff:.6e} > tolerance)",
            {
                "max_diff": f"{max_diff:.6e}",
                "mean_diff": f"{mean_diff:.6e}",
                "atol": atol,
                "rtol": rtol,
                "error": str(e),
            },
        )


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("  HUNGARIAN MATCHER VALIDATION SUITE")
    print("  PyTorch vs TTSim Implementation Comparison")
    print("=" * 80)

    # Initialize reporter
    report = TestReport()

    # Run tests
    try:
        test_hungarian_matcher_shape(report)
        test_hungarian_matcher_numerical(report)
        test_focal_loss_cost_component(report)
        test_l1_bbox_cost_component(report)
        test_giou_cost_component(report)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()

    # Print summary
    report.section("FINAL SUMMARY")
    total = len(report.results)
    passed = sum(1 for r in report.results if r["status"] == "PASS")
    failed = total - passed

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED")

    # Generate markdown report
    report_dir = Path(__file__).parent.parent / "reports"
    report_path = report_dir / "matcher_validation.md"

    report.subsection("Report Generation")
    try:
        saved_path = report.generate_markdown_report(report_path)
        print(f"\n✓ Markdown report saved to: {saved_path}")
    except Exception as e:
        print(f"\n✗ Failed to save markdown report: {str(e)}")

    print("\n" + "=" * 80)

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
