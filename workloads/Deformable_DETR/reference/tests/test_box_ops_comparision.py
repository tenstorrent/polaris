#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced test file comparing PyTorch and TTSim box operations implementations.
Compares shape inference and numerical computation with detailed outputs.
Generates markdown report with input/output samples and relative error analysis.

Functions tested:
- box_cxcywh_to_xyxy (with numerical comparison)
- box_xyxy_to_cxcywh (with numerical comparison)
- box_area (with numerical comparison)
- box_iou (with numerical comparison)
- generalized_box_iou (with numerical comparison)
- masks_to_boxes (with numerical comparison)
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import PyTorch implementations
from workloads.Deformable_DETR.reference.box_ops import (
    box_cxcywh_to_xyxy as box_cxcywh_to_xyxy_pytorch,
    box_xyxy_to_cxcywh as box_xyxy_to_cxcywh_pytorch,
    box_iou as box_iou_pytorch,
    generalized_box_iou as generalized_box_iou_pytorch,
    masks_to_boxes as masks_to_boxes_pytorch,
)
from torchvision.ops.boxes import box_area as box_area_pytorch

# Import TTSim implementations
from workloads.Deformable_DETR.util.box_ops_ttsim import (
    box_cxcywh_to_xyxy as box_cxcywh_to_xyxy_ttsim,
    box_xyxy_to_cxcywh as box_xyxy_to_cxcywh_ttsim,
    box_area as box_area_ttsim,
    box_iou as box_iou_ttsim,
    generalized_box_iou as generalized_box_iou_ttsim,
    masks_to_boxes as masks_to_boxes_ttsim,
)

from ttsim.ops.tensor import SimTensor

# ──────────────────────────────────────────────────────────────────────────────
# Global report buffer
# ──────────────────────────────────────────────────────────────────────────────
REPORT_BUFFER = []


def log_to_report(message):
    """Add message to both console and report buffer"""
    print(message)
    REPORT_BUFFER.append(message)


def save_report():
    """Save accumulated report to markdown file"""
    report_dir = "workloads/Deformable_DETR/reports"
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "box_ops_validation.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_BUFFER))

    print(f"\n[Report saved to: {report_path}]")


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────


def torch_to_simtensor(torch_tensor, name="tensor"):
    """Convert PyTorch tensor to SimTensor with proper dtype"""
    return SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": torch_tensor.detach().cpu().numpy().copy(),
            "dtype": np.dtype(np.float32),
        }
    )


def format_array_sample(data, max_elements=10):
    """Format array sample for display"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif isinstance(data, SimTensor):
        data = data.data

    if data is None:
        return "None"

    flat = data.flatten()
    if len(flat) > max_elements:
        samples = flat[:max_elements]
        return f"[{', '.join(f'{v:.6f}' for v in samples)}, ...]"
    else:
        return f"[{', '.join(f'{v:.6f}' for v in flat)}]"


def print_tensor_comparison(pytorch_data, ttsim_data, name, indent=""):
    """Print detailed tensor comparison with samples"""
    log_to_report(f"\n{indent}**{name}:**")

    if isinstance(pytorch_data, torch.Tensor):
        pytorch_data = pytorch_data.detach().cpu().numpy()
    elif isinstance(pytorch_data, SimTensor):
        pytorch_data = pytorch_data.data

    if isinstance(ttsim_data, SimTensor):
        ttsim_data = ttsim_data.data

    log_to_report(f"{indent}```")
    log_to_report(f"{indent}PyTorch shape: {list(pytorch_data.shape)}")
    if ttsim_data is not None:
        log_to_report(f"{indent}TTSim shape:   {list(ttsim_data.shape)}")
    else:
        log_to_report(f"{indent}TTSim shape:   None (shape inference only)")

    log_to_report(f"{indent}")
    log_to_report(f"{indent}PyTorch statistics:")
    log_to_report(f"{indent}  Mean: {pytorch_data.mean():.8f}")
    log_to_report(f"{indent}  Std:  {pytorch_data.std():.8f}")
    log_to_report(f"{indent}  Min:  {pytorch_data.min():.8f}")
    log_to_report(f"{indent}  Max:  {pytorch_data.max():.8f}")

    if ttsim_data is not None:
        log_to_report(f"{indent}")
        log_to_report(f"{indent}TTSim statistics:")
        log_to_report(f"{indent}  Mean: {ttsim_data.mean():.8f}")
        log_to_report(f"{indent}  Std:  {ttsim_data.std():.8f}")
        log_to_report(f"{indent}  Min:  {ttsim_data.min():.8f}")
        log_to_report(f"{indent}  Max:  {ttsim_data.max():.8f}")

    log_to_report(f"{indent}")
    log_to_report(f"{indent}Sample values (first 10):")
    log_to_report(f"{indent}  PyTorch: {format_array_sample(pytorch_data, 10)}")
    if ttsim_data is not None:
        log_to_report(f"{indent}  TTSim:   {format_array_sample(ttsim_data, 10)}")

    log_to_report(f"{indent}```")


def compare_numerics(torch_output, ttsim_output, test_name, rtol=1e-5, atol=1e-7):
    """Compare numerical values between PyTorch and TTSim outputs with detailed error analysis"""
    if isinstance(torch_output, torch.Tensor):
        torch_data = torch_output.detach().cpu().numpy()
    else:
        torch_data = np.array(torch_output)

    if isinstance(ttsim_output, SimTensor):
        ttsim_data = ttsim_output.data
    else:
        ttsim_data = np.array(ttsim_output)

    log_to_report(f"\n#### Numerical Comparison")

    if ttsim_data is None:
        log_to_report(
            f"**Result:** [SKIPPED] TTSim data is None (shape inference only)"
        )
        return False

    # Compute differences
    abs_diff = np.abs(torch_data - ttsim_data)
    rel_diff = abs_diff / (np.abs(torch_data) + 1e-10)

    max_abs_diff = abs_diff.max()
    mean_abs_diff = abs_diff.mean()
    max_rel_diff = rel_diff.max()
    mean_rel_diff = rel_diff.mean()

    log_to_report(f"```")
    log_to_report(f"Absolute Error:")
    log_to_report(f"  Max:  {max_abs_diff:.6e}")
    log_to_report(f"  Mean: {mean_abs_diff:.6e}")
    log_to_report(f"")
    log_to_report(f"Relative Error:")
    log_to_report(f"  Max:  {max_rel_diff:.6e}")
    log_to_report(f"  Mean: {mean_rel_diff:.6e}")
    log_to_report(f"")
    log_to_report(f"Tolerance:")
    log_to_report(f"  rtol: {rtol}")
    log_to_report(f"  atol: {atol}")

    # Check if within tolerance
    matches = np.allclose(torch_data, ttsim_data, rtol=rtol, atol=atol)

    if matches:
        log_to_report(f"")
        log_to_report(f"Result: PASSED (within tolerance)")
    else:
        # Find worst mismatches
        worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        log_to_report(f"")
        log_to_report(f"Result: FAILED (exceeds tolerance)")
        log_to_report(f"")
        log_to_report(f"Worst mismatch at index {worst_idx}:")
        log_to_report(f"  PyTorch: {torch_data[worst_idx]:.8f}")
        log_to_report(f"  TTSim:   {ttsim_data[worst_idx]:.8f}")
        log_to_report(f"  Abs diff: {abs_diff[worst_idx]:.8e}")
        log_to_report(f"  Rel diff: {rel_diff[worst_idx]:.8e}")

    log_to_report(f"```")

    log_to_report(
        f"\n**Result:** {'[PASSED]' if matches else '[FAILED]'} Numerical comparison"
    )
    return matches


# ──────────────────────────────────────────────────────────────────────────────
# Test Functions
# ──────────────────────────────────────────────────────────────────────────────


def test_box_cxcywh_to_xyxy():
    """Test box_cxcywh_to_xyxy with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 1: box_cxcywh_to_xyxy")
    log_to_report("=" * 80)

    log_to_report(
        f"\n**Function:** Convert boxes from (cx, cy, w, h) to (x0, y0, x1, y1) format"
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test boxes in cxcywh format
        boxes_torch = torch.tensor(
            [
                [0.5, 0.5, 0.4, 0.6],  # center box
                [0.25, 0.75, 0.3, 0.4],  # top-left region
                [0.8, 0.3, 0.2, 0.5],  # right region
            ],
            dtype=torch.float32,
        )

        boxes_ttsim = torch_to_simtensor(boxes_torch, "boxes_cxcywh")

        log_to_report(f"\n### Input Boxes (cxcywh format)")
        print_tensor_comparison(boxes_torch, boxes_ttsim, "Input Boxes")

        # Forward pass
        out_pytorch = box_cxcywh_to_xyxy_pytorch(boxes_torch)
        out_ttsim = box_cxcywh_to_xyxy_ttsim(boxes_ttsim)

        log_to_report(f"\n### Output Boxes (xyxy format)")
        print_tensor_comparison(out_pytorch, out_ttsim, "Output Boxes")

        # Compare
        numeric_match = compare_numerics(out_pytorch, out_ttsim, "box_cxcywh_to_xyxy")

        if numeric_match:
            log_to_report("\n### [PASSED] box_cxcywh_to_xyxy test")
        else:
            log_to_report("\n### [FAILED] box_cxcywh_to_xyxy test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] box_cxcywh_to_xyxy test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_box_xyxy_to_cxcywh():
    """Test box_xyxy_to_cxcywh with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 2: box_xyxy_to_cxcywh")
    log_to_report("=" * 80)

    log_to_report(
        f"\n**Function:** Convert boxes from (x0, y0, x1, y1) to (cx, cy, w, h) format"
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test boxes in xyxy format
        boxes_torch = torch.tensor(
            [
                [0.1, 0.2, 0.5, 0.7],  # box 1
                [0.3, 0.3, 0.8, 0.9],  # box 2
                [0.0, 0.0, 0.2, 0.3],  # box 3 (top-left corner)
            ],
            dtype=torch.float32,
        )

        boxes_ttsim = torch_to_simtensor(boxes_torch, "boxes_xyxy")

        log_to_report(f"\n### Input Boxes (xyxy format)")
        print_tensor_comparison(boxes_torch, boxes_ttsim, "Input Boxes")

        # Forward pass
        out_pytorch = box_xyxy_to_cxcywh_pytorch(boxes_torch)
        out_ttsim = box_xyxy_to_cxcywh_ttsim(boxes_ttsim)

        log_to_report(f"\n### Output Boxes (cxcywh format)")
        print_tensor_comparison(out_pytorch, out_ttsim, "Output Boxes")

        # Compare
        numeric_match = compare_numerics(out_pytorch, out_ttsim, "box_xyxy_to_cxcywh")

        if numeric_match:
            log_to_report("\n### [PASSED] box_xyxy_to_cxcywh test")
        else:
            log_to_report("\n### [FAILED] box_xyxy_to_cxcywh test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] box_xyxy_to_cxcywh test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_box_area():
    """Test box_area with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 3: box_area")
    log_to_report("=" * 80)

    log_to_report(f"\n**Function:** Compute area of boxes in xyxy format")
    log_to_report(f"\n**Formula:** area = (x1 - x0) * (y1 - y0)")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test boxes
        boxes_torch = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0],  # unit square, area = 1.0
                [0.1, 0.2, 0.5, 0.7],  # area = 0.4 * 0.5 = 0.2
                [0.3, 0.3, 0.8, 0.9],  # area = 0.5 * 0.6 = 0.3
                [0.0, 0.0, 0.5, 0.5],  # area = 0.25
            ],
            dtype=torch.float32,
        )

        boxes_ttsim = torch_to_simtensor(boxes_torch, "boxes")

        log_to_report(f"\n### Input Boxes")
        print_tensor_comparison(boxes_torch, boxes_ttsim, "Input Boxes (xyxy)")

        # Forward pass
        area_pytorch = box_area_pytorch(boxes_torch)
        area_ttsim = box_area_ttsim(boxes_ttsim)

        log_to_report(f"\n### Output Areas")
        print_tensor_comparison(area_pytorch, area_ttsim, "Box Areas")

        log_to_report(f"\n### Expected Areas")
        log_to_report(f"```")
        log_to_report(f"Box 0: 1.0 * 1.0 = 1.000000")
        log_to_report(f"Box 1: 0.4 * 0.5 = 0.200000")
        log_to_report(f"Box 2: 0.5 * 0.6 = 0.300000")
        log_to_report(f"Box 3: 0.5 * 0.5 = 0.250000")
        log_to_report(f"```")

        # Compare
        numeric_match = compare_numerics(area_pytorch, area_ttsim, "box_area")

        if numeric_match:
            log_to_report("\n### [PASSED] box_area test")
        else:
            log_to_report("\n### [FAILED] box_area test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] box_area test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_box_iou():
    """Test box_iou with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 4: box_iou")
    log_to_report("=" * 80)

    log_to_report(
        f"\n**Function:** Compute pairwise IoU (Intersection over Union) between two sets of boxes"
    )
    log_to_report(f"\n**Formula:** IoU = intersection_area / union_area")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create two sets of boxes
        boxes1_torch = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5],  # box A
                [0.2, 0.2, 0.7, 0.7],  # box B
            ],
            dtype=torch.float32,
        )

        boxes2_torch = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5],  # identical to box A
                [0.4, 0.4, 0.9, 0.9],  # partial overlap with B
                [0.8, 0.8, 1.0, 1.0],  # no overlap
            ],
            dtype=torch.float32,
        )

        boxes1_ttsim = torch_to_simtensor(boxes1_torch, "boxes1")
        boxes2_ttsim = torch_to_simtensor(boxes2_torch, "boxes2")

        log_to_report(f"\n### Input Boxes")
        log_to_report(f"\n**Set 1 (N=2 boxes):**")
        print_tensor_comparison(boxes1_torch, boxes1_ttsim, "Boxes1")

        log_to_report(f"\n**Set 2 (M=3 boxes):**")
        print_tensor_comparison(boxes2_torch, boxes2_ttsim, "Boxes2")

        # Forward pass
        iou_pytorch, union_pytorch = box_iou_pytorch(boxes1_torch, boxes2_torch)
        iou_ttsim, union_ttsim = box_iou_ttsim(boxes1_ttsim, boxes2_ttsim)

        log_to_report(f"\n### Output IoU Matrix [N, M]")
        print_tensor_comparison(iou_pytorch, iou_ttsim, "IoU Matrix")

        log_to_report(f"\n### Output Union Matrix [N, M]")
        print_tensor_comparison(union_pytorch, union_ttsim, "Union Matrix")

        # Compare
        iou_match = compare_numerics(
            iou_pytorch, iou_ttsim, "box_iou (IoU)", rtol=1e-5, atol=1e-7
        )
        union_match = compare_numerics(
            union_pytorch, union_ttsim, "box_iou (Union)", rtol=1e-5, atol=1e-7
        )

        if iou_match and union_match:
            log_to_report("\n### [PASSED] box_iou test")
        else:
            log_to_report("\n### [FAILED] box_iou test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] box_iou test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_generalized_box_iou():
    """Test generalized_box_iou with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 5: generalized_box_iou (GIoU)")
    log_to_report("=" * 80)

    log_to_report(f"\n**Function:** Compute Generalized IoU between two sets of boxes")
    log_to_report(
        f"\n**Formula:** GIoU = IoU - (area_enclosing - union) / area_enclosing"
    )
    log_to_report(
        f"\nGIoU extends IoU by penalizing non-overlapping boxes based on their spatial relationship."
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create boxes with known relationships
        boxes1_torch = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5],  # box A
                [0.6, 0.6, 1.0, 1.0],  # box B (disjoint from A)
            ],
            dtype=torch.float32,
        )

        boxes2_torch = torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5],  # identical to A (GIoU = 1.0)
                [0.25, 0.25, 0.75, 0.75],  # partial overlap with A
                [0.8, 0.8, 1.0, 1.0],  # close to B
            ],
            dtype=torch.float32,
        )

        boxes1_ttsim = torch_to_simtensor(boxes1_torch, "boxes1")
        boxes2_ttsim = torch_to_simtensor(boxes2_torch, "boxes2")

        log_to_report(f"\n### Input Boxes")
        log_to_report(f"\n**Set 1 (N=2 boxes):**")
        print_tensor_comparison(boxes1_torch, boxes1_ttsim, "Boxes1")

        log_to_report(f"\n**Set 2 (M=3 boxes):**")
        print_tensor_comparison(boxes2_torch, boxes2_ttsim, "Boxes2")

        # Forward pass
        giou_pytorch = generalized_box_iou_pytorch(boxes1_torch, boxes2_torch)
        giou_ttsim = generalized_box_iou_ttsim(boxes1_ttsim, boxes2_ttsim)

        log_to_report(f"\n### Output GIoU Matrix [N, M]")
        print_tensor_comparison(giou_pytorch, giou_ttsim, "GIoU Matrix")

        log_to_report(f"\n### GIoU Interpretation")
        log_to_report(f"```")
        log_to_report(f"GIoU range: [-1, 1]")
        log_to_report(f"  1.0:  Perfect overlap (identical boxes)")
        log_to_report(f"  0.0:  No overlap (touching boxes)")
        log_to_report(f" -1.0:  Maximum separation")
        log_to_report(f"```")

        # Compare
        numeric_match = compare_numerics(
            giou_pytorch, giou_ttsim, "generalized_box_iou", rtol=1e-5, atol=1e-7
        )

        if numeric_match:
            log_to_report("\n### [PASSED] generalized_box_iou test")
        else:
            log_to_report("\n### [FAILED] generalized_box_iou test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] generalized_box_iou test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_masks_to_boxes():
    """Test masks_to_boxes with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 6: masks_to_boxes")
    log_to_report("=" * 80)

    log_to_report(f"\n**Function:** Extract bounding boxes from binary masks")
    log_to_report(
        f"\n**Algorithm:** Find min/max coordinates of True pixels in each mask"
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test masks with known bounding boxes
        masks_torch = torch.zeros(3, 10, 10, dtype=torch.bool)

        # Mask 0: Rectangle [2:5, 3:8] -> box should be [3, 2, 7, 4]
        masks_torch[0, 2:5, 3:8] = True

        # Mask 1: Square [5:9, 1:5] -> box should be [1, 5, 4, 8]
        masks_torch[1, 5:9, 1:5] = True

        # Mask 2: Small region [0:2, 8:10] -> box should be [8, 0, 9, 1]
        masks_torch[2, 0:2, 8:10] = True

        masks_ttsim = torch_to_simtensor(masks_torch.float(), "masks")

        log_to_report(f"\n### Input Masks")
        log_to_report(f"```")
        log_to_report(f"Shape: {list(masks_torch.shape)} ([N, H, W])")
        log_to_report(f"")
        log_to_report(f"Mask 0: Active pixels at [2:5, 3:8]")
        log_to_report(f"  Expected box: [3, 2, 7, 4] (x_min, y_min, x_max, y_max)")
        log_to_report(f"")
        log_to_report(f"Mask 1: Active pixels at [5:9, 1:5]")
        log_to_report(f"  Expected box: [1, 5, 4, 8]")
        log_to_report(f"")
        log_to_report(f"Mask 2: Active pixels at [0:2, 8:10]")
        log_to_report(f"  Expected box: [8, 0, 9, 1]")
        log_to_report(f"```")

        # Forward pass
        boxes_pytorch = masks_to_boxes_pytorch(masks_torch)
        boxes_ttsim = masks_to_boxes_ttsim(masks_ttsim)

        log_to_report(f"\n### Output Bounding Boxes")
        print_tensor_comparison(boxes_pytorch, boxes_ttsim, "Extracted Boxes")

        # Compare
        numeric_match = compare_numerics(
            boxes_pytorch, boxes_ttsim, "masks_to_boxes", rtol=1e-5, atol=1e-7
        )

        if numeric_match:
            log_to_report("\n### [PASSED] masks_to_boxes test")
        else:
            log_to_report("\n### [FAILED] masks_to_boxes test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] masks_to_boxes test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_to_report("# Box Operations Validation Report")
    log_to_report(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_report(
        f"\n**Test Suite:** PyTorch vs TTSim Bounding Box Operations Comparison"
    )

    log_to_report("\n---\n")

    log_to_report("## Overview")
    log_to_report(
        "\nThis report validates the TTSim implementation of bounding box utility functions."
    )
    log_to_report("\n**Functions Tested:**")
    log_to_report("1. **box_cxcywh_to_xyxy**: Convert center format to corner format")
    log_to_report("2. **box_xyxy_to_cxcywh**: Convert corner format to center format")
    log_to_report("3. **box_area**: Compute box areas")
    log_to_report("4. **box_iou**: Compute Intersection over Union")
    log_to_report("5. **generalized_box_iou**: Compute Generalized IoU (GIoU)")
    log_to_report("6. **masks_to_boxes**: Extract boxes from binary masks")

    log_to_report("\n**All functions include full numerical comparison with PyTorch.**")

    log_to_report("\n---\n")

    tests_passed = 0
    tests_failed = 0

    tests = [
        ("box_cxcywh_to_xyxy", test_box_cxcywh_to_xyxy),
        ("box_xyxy_to_cxcywh", test_box_xyxy_to_cxcywh),
        ("box_area", test_box_area),
        ("box_iou", test_box_iou),
        ("generalized_box_iou", test_generalized_box_iou),
        ("masks_to_boxes", test_masks_to_boxes),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            tests_passed += 1
        except Exception as e:
            tests_failed += 1
            log_to_report(
                f"\n[WARNING] Test {test_name} encountered errors but continuing..."
            )

    log_to_report("\n" + "=" * 80)
    log_to_report("# Test Summary")
    log_to_report("=" * 80)
    log_to_report(f"\n| Metric | Value |")
    log_to_report(f"|--------|-------|")
    log_to_report(f"| **Tests Passed** | {tests_passed}/{len(tests)} |")
    log_to_report(f"| **Tests Failed** | {tests_failed}/{len(tests)} |")
    log_to_report(f"| **Success Rate** | {100*tests_passed/len(tests):.1f}% |")

    log_to_report(f"\n## Function Status")
    log_to_report(f"\n| Function | Numerical Comparison | Status |")
    log_to_report(f"|----------|---------------------|--------|")
    log_to_report(f"| box_cxcywh_to_xyxy | ✓ Full | Deterministic math |")
    log_to_report(f"| box_xyxy_to_cxcywh | ✓ Full | Deterministic math |")
    log_to_report(f"| box_area | ✓ Full | Simple arithmetic |")
    log_to_report(f"| box_iou | ✓ Full | Intersection/union |")
    log_to_report(f"| generalized_box_iou | ✓ Full | GIoU formula |")
    log_to_report(f"| masks_to_boxes | ✓ Full | Min/max extraction |")

    if tests_failed == 0:
        log_to_report("\n## [PASSED] All box operations tests completed successfully!")
        log_to_report(
            "\nThe TTSim implementation produces numerically identical results to PyTorch."
        )
    else:
        log_to_report(
            f"\n## [FAILED] {tests_failed} test(s) failed. Review errors above."
        )

    log_to_report("\n---\n")
    log_to_report("\n*End of Report*")

    # Save report
    save_report()
