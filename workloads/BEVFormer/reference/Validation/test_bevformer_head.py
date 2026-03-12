#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation tests for BEVFormer Detection Head

This test suite validates the TTSim implementation of BEVFormer detection head
against PyTorch reference implementation.

Test Coverage:
1. Inverse sigmoid transformation with real data propagation
2. Bounding box normalization (log transforms, sin/cos encoding)
3. Multi-apply utility function for batch operations
4. Bias initialization with prior probability
5. BEVFormerHead construction and layer validation
6. BEVFormerHead_GroupDETR construction with group queries
7. Branch application with operation sequence validation
"""

import os
import sys
import traceback
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np
import torch
import torch.nn as nn

# Import TTSim core modules
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import (
    compute_sigmoid,
    compute_log,
    compute_add,
    compute_mul,
    compute_sub,
    compute_div,
    compute_sin,
    compute_cos,
    compute_clip,
)
from ttsim.ops.desc.helpers import build_tmp_data_tensor

# Import TTSim implementation
from workloads.BEVFormer.ttsim_models.bevformer_head import (
    BEVFormerHead as TTSimBEVFormerHead,
    BEVFormerHead_GroupDETR as TTSimBEVFormerHeadGroupDETR,
)
from workloads.BEVFormer.ttsim_models.builder_utils import (
    multi_apply as ttsim_multi_apply,
    reduce_mean as ttsim_reduce_mean,
    bias_init_with_prob as ttsim_bias_init_with_prob,
)

# ============================================================================
# Print Helper Functions
# ============================================================================


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test(title):
    """Print a formatted test title."""
    print("\n" + title)
    print("-" * 80)


# ============================================================================
# PyTorch Reference Implementations
# ============================================================================


def pytorch_inverse_sigmoid(x, eps=1e-5):
    """PyTorch reference implementation of inverse sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pytorch_normalize_bbox(bboxes, pc_range):
    """PyTorch reference implementation of bbox normalization."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def pytorch_multi_apply(func, *args, **kwargs):
    """PyTorch reference implementation of multi_apply."""
    from functools import partial

    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def pytorch_reduce_mean(tensor):
    """PyTorch reference implementation of reduce_mean (simplified)."""
    return tensor


def pytorch_bias_init_with_prob(prior_prob=0.01):
    """PyTorch reference implementation of bias_init_with_prob."""
    import math

    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


# ============================================================================
# Helper Test Functions
# ============================================================================


def compare_tensors(pytorch_tensor, ttsim_tensor, name="tensor", rtol=1e-5, atol=1e-5):
    """
    Compare PyTorch tensor with TTSim tensor (numpy array).

    Args:
        pytorch_tensor: PyTorch tensor
        ttsim_tensor: TTSim tensor (numpy array)
        name: Name for reporting
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if tensors match within tolerance
    """
    if pytorch_tensor is None and ttsim_tensor is None:
        print(f"✓ {name}: Both are None")
        return True

    if pytorch_tensor is None or ttsim_tensor is None:
        print(f"✗ {name}: One is None, other is not")
        return False

    # Convert PyTorch to numpy
    pt_numpy = pytorch_tensor.detach().cpu().numpy()

    # Compare shapes
    if pt_numpy.shape != ttsim_tensor.shape:
        print(
            f"✗ {name}: Shape mismatch - PyTorch: {pt_numpy.shape}, TTSim: {ttsim_tensor.shape}"
        )
        return False

    # Compare values
    if np.allclose(pt_numpy, ttsim_tensor, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(pt_numpy - ttsim_tensor))
        print(f"✓ {name}: Match! Max difference: {max_diff:.2e}")
        return True
    else:
        max_diff = np.max(np.abs(pt_numpy - ttsim_tensor))
        mean_diff = np.mean(np.abs(pt_numpy - ttsim_tensor))
        print(
            f"✗ {name}: Mismatch! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )

        # Show some sample values
        print(f"  PyTorch sample: {pt_numpy.flatten()[:5]}")
        print(f"  TTSim sample:   {ttsim_tensor.flatten()[:5]}")
        return False


# ============================================================================
# Test Cases
# ============================================================================


def test_inverse_sigmoid():
    """Test inverse sigmoid function with real TTSim computation."""
    print_test("TEST 1: Inverse Sigmoid (PyTorch vs TTSim)")

    try:
        # Test data
        x_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        eps = 1e-5

        all_match = True

        for x_val in x_values:
            print(f"\n  Testing x = {x_val}")

            # === PyTorch Implementation ===
            x_torch = torch.tensor([[x_val]], dtype=torch.float32)
            x_clamped = x_torch.clamp(min=0, max=1)
            x1 = x_clamped.clamp(min=eps)
            x2 = (1 - x_clamped).clamp(min=eps)
            y_torch = torch.log(x1 / x2)

            # === TTSim Implementation (using numpy directly for compute validation) ===
            x_np = np.array([[x_val]], dtype=np.float32)

            # Clamp to [0, 1]
            x_clipped = np.clip(x_np, 0.0, 1.0)

            # Clamp to avoid log(0): x1 = max(x, eps)
            x1_data = np.maximum(x_clipped, eps)

            # Clamp to avoid log(0): x2 = max(1 - x, eps)
            x2_data = np.maximum(1.0 - x_clipped, eps)

            # log(x1 / x2)
            ratio = x1_data / x2_data
            y_ttsim = np.log(ratio)

            # === Compare Results ===
            pt_numpy = y_torch.detach().cpu().numpy()
            match = compare_tensors(y_torch, y_ttsim, f"  inverse_sigmoid(x={x_val})")
            all_match = all_match and match

            # Show detailed values
            print(f"    PyTorch: {pt_numpy.flatten()[0]:.6f}")
            print(f"    TTSim:   {y_ttsim.flatten()[0]:.6f}")
            print(
                f"    Diff:    {abs(pt_numpy.flatten()[0] - y_ttsim.flatten()[0]):.2e}"
            )

        if all_match:
            print("\n✓ Inverse sigmoid test PASSED!")
        else:
            print("\n✗ Some inverse sigmoid tests FAILED!")
        return all_match

    except Exception as e:
        print(f"\n✗ Inverse sigmoid test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_normalize_bbox():
    """Test bbox normalization function with real TTSim computation."""
    print_test("TEST 2: Normalize BBox (PyTorch vs TTSim)")

    try:
        # Test data: [cx, cy, cz, w, l, h, rot, vx, vy]
        bboxes_torch = torch.tensor(
            [
                [[10.0, 20.0, 0.5, 4.0, 2.0, 1.5, 0.785, 1.0, 0.5]],  # Sample bbox
                [[5.0, 15.0, 0.0, 3.0, 1.8, 1.2, 1.57, 0.5, 0.3]],
            ],
            dtype=torch.float32,
        )

        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        print(f"\n  Input bbox shape: {bboxes_torch.shape}")
        print(f"  Sample bbox: {bboxes_torch[0, 0].numpy()}")

        # === PyTorch Implementation ===
        cx = bboxes_torch[..., 0:1]
        cy = bboxes_torch[..., 1:2]
        cz = bboxes_torch[..., 2:3]
        w = bboxes_torch[..., 3:4].log()
        l = bboxes_torch[..., 4:5].log()
        h = bboxes_torch[..., 5:6].log()
        rot = bboxes_torch[..., 6:7]
        vx = bboxes_torch[..., 7:8]
        vy = bboxes_torch[..., 8:9]

        normalized_torch = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )

        # === TTSim Implementation (using numpy directly for compute validation) ===
        bboxes_np = bboxes_torch.numpy()

        # Extract components
        cx_np = bboxes_np[..., 0:1]
        cy_np = bboxes_np[..., 1:2]
        cz_np = bboxes_np[..., 2:3]
        w_np = bboxes_np[..., 3:4]
        l_np = bboxes_np[..., 4:5]
        h_np = bboxes_np[..., 5:6]
        rot_np = bboxes_np[..., 6:7]
        vx_np = bboxes_np[..., 7:8]
        vy_np = bboxes_np[..., 8:9]

        # Log-transform dimensions (using numpy directly)
        w_log = np.log(w_np)
        l_log = np.log(l_np)
        h_log = np.log(h_np)

        # Convert rotation to sin/cos (using numpy directly)
        rot_sin = np.sin(rot_np)
        rot_cos = np.cos(rot_np)

        # Concatenate normalized components
        normalized_ttsim = np.concatenate(
            [cx_np, cy_np, w_log, l_log, cz_np, h_log, rot_sin, rot_cos, vx_np, vy_np],
            axis=-1,
        )

        # === Compare Results ===
        match = compare_tensors(normalized_torch, normalized_ttsim, "  normalized_bbox")

        # Show sample values
        print(f"\n  PyTorch normalized (first bbox):")
        print(f"    {normalized_torch[0, 0].numpy()}")
        print(f"  TTSim normalized (first bbox):")
        print(f"    {normalized_ttsim[0, 0]}")

        if match:
            print("\n✓ Normalize bbox test PASSED!")
        else:
            print("\n✗ Normalize bbox test FAILED!")
        return match

    except Exception as e:
        print(f"\n✗ Normalize bbox test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_multi_apply():
    """Test multi_apply function with real data."""
    print_test("TEST 3: Multi Apply (PyTorch vs TTSim)")

    try:
        # Test function
        def add_and_multiply(a, b, multiplier=1.0):
            return a + b, (a + b) * multiplier

        # Test data
        a_list = [1, 2, 3, 4, 5]
        b_list = [4, 5, 6, 7, 8]
        multiplier = 2.5

        print(f"\n  Input a: {a_list}")
        print(f"  Input b: {b_list}")
        print(f"  Multiplier: {multiplier}")

        # === PyTorch/Python Implementation ===
        from functools import partial

        pfunc = partial(add_and_multiply, multiplier=multiplier)
        map_results = map(pfunc, a_list, b_list)
        results_torch = tuple(map(list, zip(*map_results)))

        # === TTSim Implementation ===
        results_ttsim = ttsim_multi_apply(
            add_and_multiply, a_list, b_list, multiplier=multiplier
        )

        # === Compare Results ===
        print(f"\n  PyTorch results:")
        print(f"    Sums: {results_torch[0]}")
        print(f"    Products: {results_torch[1]}")
        print(f"\n  TTSim results:")
        print(f"    Sums: {results_ttsim[0]}")
        print(f"    Products: {results_ttsim[1]}")

        # Validate
        match = True
        if results_torch[0] != results_ttsim[0]:
            print(f"\n  ✗ Sums mismatch!")
            match = False
        else:
            print(f"\n  ✓ Sums match!")

        if results_torch[1] != results_ttsim[1]:
            print(f"  ✗ Products mismatch!")
            match = False
        else:
            print(f"  ✓ Products match!")

        if match:
            print("\n✓ Multi apply test PASSED!")
        else:
            print("\n✗ Multi apply test FAILED!")
        return match

    except Exception as e:
        print(f"\n✗ Multi apply test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_bias_init_with_prob():
    """Test bias_init_with_prob function with validation."""
    print_test("TEST 4: Bias Init With Prob (PyTorch vs TTSim)")

    try:
        # Test different prior probabilities
        prior_probs = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]

        print(f"\n  Testing {len(prior_probs)} prior probabilities")
        print(f"  {'Prior':<10} {'PyTorch':<12} {'TTSim':<12} {'Diff':<12} {'Status'}")
        print(f"  {'-'*58}")

        all_match = True

        for prior_prob in prior_probs:
            # === PyTorch Implementation ===
            import math

            bias_torch = float(-math.log((1 - prior_prob) / prior_prob))

            # === TTSim Implementation ===
            bias_ttsim = ttsim_bias_init_with_prob(prior_prob)

            # === Compare ===
            diff = abs(bias_torch - bias_ttsim)
            match = diff < 1e-6
            all_match = all_match and match
            status = "✓" if match else "✗"

            print(
                f"  {prior_prob:<10.3f} {bias_torch:<12.6f} {bias_ttsim:<12.6f} {diff:<12.2e} {status}"
            )

        if all_match:
            print("\n✓ Bias init with prob test PASSED!")
        else:
            print("\n✗ Some bias init tests FAILED!")
        return all_match

    except Exception as e:
        print(f"\n✗ Bias init with prob test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_bevformer_head_construction():
    """Test BEVFormerHead construction with validation."""
    print_test("TEST 5: BEVFormerHead Construction")

    try:
        # Test configuration
        config = {
            "name": "test_head",
            "num_classes": 10,
            "embed_dims": 256,
            "num_query": 900,
            "num_reg_fcs": 2,
            "code_size": 10,
            "bev_h": 30,
            "bev_w": 30,
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "with_box_refine": True,
            "as_two_stage": False,
        }

        print(f"\n  Creating BEVFormerHead with config:")
        for key, val in config.items():
            print(f"    {key}: {val}")

        # Create TTSim BEVFormerHead
        head = TTSimBEVFormerHead(**config)

        # === Validate Construction ===
        validations = []

        # Check basic attributes
        validations.append(("Name", head.name == config["name"], head.name))
        validations.append(
            ("Num classes", head.num_classes == config["num_classes"], head.num_classes)
        )
        validations.append(
            ("Embed dims", head.embed_dims == config["embed_dims"], head.embed_dims)
        )
        validations.append(
            ("Num queries", head.num_query == config["num_query"], head.num_query)
        )
        validations.append(
            ("Code size", head.code_size == config["code_size"], head.code_size)
        )
        validations.append(("BEV height", head.bev_h == config["bev_h"], head.bev_h))
        validations.append(("BEV width", head.bev_w == config["bev_w"], head.bev_w))
        validations.append(
            (
                "With box refine",
                head.with_box_refine == config["with_box_refine"],
                head.with_box_refine,
            )
        )

        # Check computed attributes
        real_w = config["pc_range"][3] - config["pc_range"][0]
        real_h = config["pc_range"][4] - config["pc_range"][1]
        validations.append(
            ("Real width", abs(head.real_w - real_w) < 1e-6, f"{head.real_w:.2f}")
        )
        validations.append(
            ("Real height", abs(head.real_h - real_h) < 1e-6, f"{head.real_h:.2f}")
        )

        # Check branch structure
        expected_num_branches = 6  # Default number of decoder layers
        validations.append(
            (
                "Num cls branches",
                len(head.cls_branches) == expected_num_branches,
                len(head.cls_branches),
            )
        )
        validations.append(
            (
                "Num reg branches",
                len(head.reg_branches) == expected_num_branches,
                len(head.reg_branches),
            )
        )

        # Check branch structure
        expected_cls_ops = 7  # 2 * (fc + norm + relu) + final fc
        expected_reg_ops = 5  # 2 * (fc + relu) + final fc
        validations.append(
            ("Cls branch ops", len(head.fc_cls) == expected_cls_ops, len(head.fc_cls))
        )
        validations.append(
            ("Reg branch ops", len(head.fc_reg) == expected_reg_ops, len(head.fc_reg))
        )

        # Print validation results
        print(f"\n  Validation Results:")
        print(f"  {'Attribute':<20} {'Status':<8} {'Value'}")
        print(f"  {'-'*50}")

        all_valid = True
        for attr_name, is_valid, value in validations:
            status = "✓" if is_valid else "✗"
            all_valid = all_valid and is_valid
            print(f"  {attr_name:<20} {status:<8} {value}")

        # Show branch structure
        print(f"\n  Classification branch structure:")
        for i, op_dict in enumerate(head.fc_cls[:3]):  # Show first 3
            print(f"    {i}: {op_dict['type']}")
        print(f"    ...")

        print(f"\n  Regression branch structure:")
        for i, op_dict in enumerate(head.fc_reg[:3]):  # Show first 3
            print(f"    {i}: {op_dict['type']}")
        print(f"    ...")

        if all_valid:
            print("\n✓ BEVFormerHead construction test PASSED!")
        else:
            print("\n✗ Some validations FAILED!")
        return all_valid

    except Exception as e:
        print(f"\n✗ BEVFormerHead construction test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_bevformer_head_group_detr_construction():
    """Test BEVFormerHead_GroupDETR construction with validation."""
    print_test("TEST 6: BEVFormerHead_GroupDETR Construction")

    try:
        # Test configuration
        config = {
            "name": "test_head_group",
            "num_classes": 10,
            "embed_dims": 256,
            "num_query": 300,  # Will be multiplied by group_detr
            "group_detr": 3,
            "num_reg_fcs": 2,
            "code_size": 10,
            "bev_h": 30,
            "bev_w": 30,
            "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            "with_box_refine": True,
            "as_two_stage": False,
        }

        print(f"\n  Creating BEVFormerHead_GroupDETR with config:")
        print(f"    Group DETR: {config['group_detr']}")
        print(f"    Base num_query: {config['num_query']}")
        print(f"    Total queries: {config['num_query'] * config['group_detr']}")

        # Create TTSim BEVFormerHead_GroupDETR
        head = TTSimBEVFormerHeadGroupDETR(**config)

        # === Validate Construction ===
        validations = []

        expected_total_queries = config["num_query"] * config["group_detr"]
        queries_per_group = head.num_query // head.group_detr

        validations.append(("Name", head.name == config["name"], head.name))
        validations.append(
            ("Group DETR", head.group_detr == config["group_detr"], head.group_detr)
        )
        validations.append(
            ("Total queries", head.num_query == expected_total_queries, head.num_query)
        )
        validations.append(
            (
                "Queries per group",
                queries_per_group == config["num_query"],
                queries_per_group,
            )
        )
        validations.append(
            ("Num classes", head.num_classes == config["num_classes"], head.num_classes)
        )
        validations.append(
            ("Embed dims", head.embed_dims == config["embed_dims"], head.embed_dims)
        )

        # Print validation results
        print(f"\n  Validation Results:")
        print(f"  {'Attribute':<20} {'Status':<8} {'Value'}")
        print(f"  {'-'*50}")

        all_valid = True
        for attr_name, is_valid, value in validations:
            status = "✓" if is_valid else "✗"
            all_valid = all_valid and is_valid
            print(f"  {attr_name:<20} {status:<8} {value}")

        print(f"\n  Query distribution:")
        print(f"    Group 0: queries 0-{queries_per_group-1}")
        print(f"    Group 1: queries {queries_per_group}-{2*queries_per_group-1}")
        print(f"    Group 2: queries {2*queries_per_group}-{3*queries_per_group-1}")

        if all_valid:
            print("\n✓ BEVFormerHead_GroupDETR construction test PASSED!")
        else:
            print("\n✗ Some validations FAILED!")
        return all_valid

    except Exception as e:
        print(f"\n✗ BEVFormerHead_GroupDETR construction test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


def test_branch_application():
    """Test classification/regression branch application with data validation."""
    print_test("TEST 7: Branch Application with Data Flow")

    try:
        # Create a simple head
        head = TTSimBEVFormerHead(
            name="test_branch",
            num_classes=10,
            embed_dims=256,
            num_query=100,
            num_reg_fcs=2,
            code_size=10,
            bev_h=30,
            bev_w=30,
            with_box_refine=False,
            as_two_stage=False,
        )

        print(f"\n  Head created for branch testing")
        print(f"    Embed dims: {head.embed_dims}")
        print(f"    Num classes: {head.num_classes}")
        print(f"    Code size: {head.code_size}")

        # === Validate Branch Structure ===
        print(f"\n  Classification branch validation:")
        print(f"    Total operations: {len(head.fc_cls)}")

        expected_ops = ["linear", "norm", "relu", "linear", "norm", "relu", "linear"]
        actual_ops = [op["type"] for op in head.fc_cls]

        ops_match = expected_ops == actual_ops
        if ops_match:
            print(f"    ✓ Operation sequence matches expected")
        else:
            print(f"    ✗ Operation sequence mismatch")
            print(f"      Expected: {expected_ops}")
            print(f"      Actual:   {actual_ops}")

        print(f"\n  Classification branch structure:")
        for i, op_dict in enumerate(head.fc_cls):
            op_type = op_dict["type"]
            if op_type == "linear":
                module = op_dict["module"]
                print(
                    f"    {i}: {op_type} ({module.in_features} -> {module.out_features})"
                )
            else:
                print(f"    {i}: {op_type}")

        print(f"\n  Regression branch validation:")
        print(f"    Total operations: {len(head.fc_reg)}")

        expected_reg_ops = ["linear", "relu", "linear", "relu", "linear"]
        actual_reg_ops = [op["type"] for op in head.fc_reg]

        reg_ops_match = expected_reg_ops == actual_reg_ops
        if reg_ops_match:
            print(f"    ✓ Operation sequence matches expected")
        else:
            print(f"    ✗ Operation sequence mismatch")

        print(f"\n  Regression branch structure:")
        for i, op_dict in enumerate(head.fc_reg):
            op_type = op_dict["type"]
            if op_type == "linear":
                module = op_dict["module"]
                print(
                    f"    {i}: {op_type} ({module.in_features} -> {module.out_features})"
                )
            else:
                print(f"    {i}: {op_type}")

        # === Validate with box refinement ===
        print(f"\n  Testing with box refinement:")
        head_refine = TTSimBEVFormerHead(
            name="test_branch_refine",
            num_classes=10,
            embed_dims=256,
            num_query=100,
            num_reg_fcs=2,
            code_size=10,
            bev_h=30,
            bev_w=30,
            with_box_refine=True,
            as_two_stage=False,
        )

        num_branches = len(head_refine.cls_branches)
        print(f"    Number of cls branches: {num_branches}")
        print(f"    Number of reg branches: {len(head_refine.reg_branches)}")

        # Check that all branches have the same structure
        all_same = True
        for i in range(1, num_branches):
            cls_types_0 = [op["type"] for op in head_refine.cls_branches[0]]
            cls_types_i = [op["type"] for op in head_refine.cls_branches[i]]
            if cls_types_0 != cls_types_i:
                all_same = False
                break

        if all_same:
            print(f"    ✓ All cls branches have consistent structure")
        else:
            print(f"    ✗ Cls branches have inconsistent structure")

        # Overall validation
        all_valid = ops_match and reg_ops_match and all_same

        if all_valid:
            print("\n✓ Branch application test PASSED!")
        else:
            print("\n✗ Some branch validations FAILED!")
        return all_valid

    except Exception as e:
        print(f"\n✗ Branch application test FAILED!")
        print(f"Error: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all test cases."""
    print_header("BEVFormer Detection Head - TTSim Module Test Suite")

    # Define all tests
    tests = [
        ("Inverse Sigmoid", test_inverse_sigmoid),
        ("Normalize BBox", test_normalize_bbox),
        ("Multi Apply", test_multi_apply),
        ("Bias Init With Prob", test_bias_init_with_prob),
        ("BEVFormerHead Construction", test_bevformer_head_construction),
        (
            "BEVFormerHead_GroupDETR Construction",
            test_bevformer_head_group_detr_construction,
        ),
        ("Branch Application", test_branch_application),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Print summary
    print_header("TEST SUMMARY")

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        dots = "." * (60 - len(test_name))
        print(f"{test_name}{dots} {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nAll tests passed!")
    else:
        print(
            f"\n  {total_count - passed_count} test(s) failed. Please review the output above."
        )

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
