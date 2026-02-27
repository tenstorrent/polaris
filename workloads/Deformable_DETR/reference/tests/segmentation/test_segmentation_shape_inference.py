#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Shape Inference Test for Deformable DETR Segmentation Modules.

This test validates shape consistency between PyTorch and TTSim implementations.
Tests are designed to identify and diagnose shape inference issues.

Critical Issues Found:
  1. Unsqueeze operation: ipos mismatch in MHAttentionMap
  2. Concat operation: Shape mismatch in MaskHeadSmallConv expand logic
  3. Tile operation: Batch dimension expansion issues

Modules Tested:
  1. MHAttentionMap - Multi-head attention map generation
  2. MaskHeadSmallConv - FPN-based mask prediction head
  3. DETRsegm - Complete segmentation wrapper
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Add project root to path (4 levels up: segmentation -> tests -> Deformable_DETR -> workloads -> polaris)
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
from workloads.Deformable_DETR.reference.segmentation import (
    MHAttentionMap as MHAttentionMapPyTorch,
    MaskHeadSmallConv as MaskHeadSmallConvPyTorch,
)

# Import TTSim implementations
from workloads.Deformable_DETR.models.segmentation_ttsim import (
    MHAttentionMap as MHAttentionMapTTSim,
    MaskHeadSmallConv as MaskHeadSmallConvTTSim,
    DETRsegm as DETRsegmTTSim,
)

from ttsim.ops.tensor import SimTensor

# ============================================================================
# Utility Functions
# ============================================================================


def torch_to_simtensor(torch_tensor, name="tensor"):
    """Convert PyTorch tensor to SimTensor with data attached."""
    return SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": torch_tensor.detach().cpu().numpy().copy(),
            "dtype": np.dtype(np.float32),
        }
    )


def print_section(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subsection(title, char="-", width=80):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def compare_shapes(pytorch_out, ttsim_out, module_name="Module"):
    """
    Compare output shapes between PyTorch and TTSim.

    Returns:
        bool: True if shapes match, False otherwise
    """
    pytorch_shape = list(pytorch_out.shape)
    ttsim_shape = list(ttsim_out.shape)

    match = pytorch_shape == ttsim_shape

    if match:
        print(f"\n✓ PASS - Shape Validation")
        print(f"  Expected: {pytorch_shape}")
        print(f"  Got:      {ttsim_shape}")
    else:
        print(f"\n✗ FAIL - Shape Mismatch")
        print(f"  Expected (PyTorch): {pytorch_shape}")
        print(f"  Got (TTSim):        {ttsim_shape}")
        print(f"  Module: {module_name}")

    return match


def diagnose_error(error, module_name):
    """Diagnose common TTSim errors and provide fixes."""
    error_msg = str(error)

    print(f"\n🔍 ERROR DIAGNOSIS for {module_name}:")
    print(f"  Error: {error_msg}")

    if "ipos" in error_msg and "don't match" in error_msg:
        print(f"\n  Issue: Unsqueeze operation expects different number of inputs")
        print(f"  Root Cause: TTSim Unsqueeze with axes parameter expects only 1 input")
        print(f"  Fix: Remove ipos parameter or check operation construction")

    elif "Incompatible shapes" in error_msg and "concat" in error_msg.lower():
        print(f"\n  Issue: Concatenation shape mismatch")
        print(f"  Root Cause: Expand/Tile operation not working correctly")
        print(f"  Fix: Verify batch dimension expansion logic")

    elif "Tile" in error_msg or "repeats" in error_msg:
        print(f"\n  Issue: Tile operation configuration error")
        print(f"  Root Cause: Repeats tensor shape or value incorrect")
        print(f"  Fix: Check repeats tensor construction and shape")

    print()


# ============================================================================
# Test 1: MHAttentionMap
# ============================================================================


def test_mh_attention_map():
    """
    Test Multi-Head Attention Map module.

    Known Issue: Unsqueeze operation ipos mismatch
    Expected Error: "Length for inputs 1 & ipos 2 don't match"
    """

    print_section("TEST 1: MHAttentionMap")

    # Configuration
    batch_size = 2
    num_queries = 100
    query_dim = 256
    hidden_dim = 256
    num_heads = 8
    H, W = 16, 16

    print("\nConfiguration:")
    print(f"  Batch size:   {batch_size}")
    print(f"  Num queries:  {num_queries}")
    print(f"  Query dim:    {query_dim}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Num heads:    {num_heads}")
    print(f"  Spatial size: {H} x {W}")

    try:
        # Create inputs
        torch.manual_seed(42)
        np.random.seed(42)

        q_torch = torch.randn(batch_size, num_queries, query_dim)
        k_torch = torch.randn(batch_size, query_dim, H, W)
        mask_torch = torch.zeros(batch_size, H, W, dtype=torch.bool)
        mask_torch[:, :2, :2] = True

        print(f"\nInput Shapes:")
        print(f"  q (queries): {list(q_torch.shape)}")
        print(f"  k (keys):    {list(k_torch.shape)}")
        print(f"  mask:        {list(mask_torch.shape)}")

        # PyTorch Forward
        print_subsection("PyTorch Forward Pass")

        attention_pytorch = MHAttentionMapPyTorch(
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
        )
        attention_pytorch.eval()

        with torch.no_grad():
            out_pytorch = attention_pytorch(q_torch, k_torch, mask=mask_torch)

        print(f"Output Shape: {list(out_pytorch.shape)}")
        print(f"Expected:     [{batch_size}, {num_queries}, {num_heads}, {H}, {W}]")

        # TTSim Forward
        print_subsection("TTSim Forward Pass")

        q_ttsim = torch_to_simtensor(q_torch, "q")
        k_ttsim = torch_to_simtensor(k_torch, "k")
        mask_ttsim = torch_to_simtensor(mask_torch.float(), "mask")

        attention_ttsim = MHAttentionMapTTSim(
            name="mh_attention_test",
            query_dim=query_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
        )

        out_ttsim = attention_ttsim(q_ttsim, k_ttsim, mask=mask_ttsim)

        print(f"Output Shape: {list(out_ttsim.shape)}")

        # Comparison
        print_subsection("Shape Comparison")

        result = compare_shapes(out_pytorch, out_ttsim, "MHAttentionMap")

        # Parameter count
        print(f"\nParameter Count:")
        pytorch_params = sum(p.numel() for p in attention_pytorch.parameters())
        ttsim_params = attention_ttsim.analytical_param_count()
        param_match = pytorch_params == ttsim_params
        print(f"  PyTorch: {pytorch_params:,}")
        print(f"  TTSim:   {ttsim_params:,}")
        print(f"  Match:   {'✓' if param_match else '✗'}")

        return result and param_match

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        diagnose_error(e, "MHAttentionMap")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Test 2: MaskHeadSmallConv
# ============================================================================


def test_mask_head_small_conv():
    """
    Test MaskHeadSmallConv module.

    Known Issue: Concat operation shape mismatch due to expand/tile
    Expected Error: "Incompatible shapes at dim 1"
    """

    print_section("TEST 2: MaskHeadSmallConv")

    # Configuration
    batch_size = 2
    num_queries = 100
    hidden_dim = 256
    num_heads = 8
    H, W = 16, 16

    dim = hidden_dim + num_heads
    fpn_dims = [1024, 512, 256]
    context_dim = hidden_dim

    print("\nConfiguration:")
    print(f"  Batch size:       {batch_size}")
    print(f"  Num queries:      {num_queries}")
    print(f"  Input dim:        {dim} (hidden_dim + num_heads)")
    print(f"  FPN dims:         {fpn_dims}")
    print(f"  Context dim:      {context_dim}")
    print(f"  Input spatial:    {H} x {W}")

    try:
        # Create inputs
        torch.manual_seed(42)
        np.random.seed(42)

        x_torch = torch.randn(batch_size, hidden_dim, H, W)
        bbox_mask_torch = torch.randn(batch_size, num_queries, num_heads, H, W)

        fpn0_torch = torch.randn(batch_size, fpn_dims[0], H * 2, W * 2)
        fpn1_torch = torch.randn(batch_size, fpn_dims[1], H * 4, W * 4)
        fpn2_torch = torch.randn(batch_size, fpn_dims[2], H * 8, W * 8)
        fpns_torch = [fpn0_torch, fpn1_torch, fpn2_torch]

        print(f"\nInput Shapes:")
        print(f"  x (src_proj): {list(x_torch.shape)}")
        print(f"  bbox_mask:    {list(bbox_mask_torch.shape)}")
        print(f"  fpns[0]:      {list(fpn0_torch.shape)}")
        print(f"  fpns[1]:      {list(fpn1_torch.shape)}")
        print(f"  fpns[2]:      {list(fpn2_torch.shape)}")

        # PyTorch Forward
        print_subsection("PyTorch Forward Pass")

        mask_head_pytorch = MaskHeadSmallConvPyTorch(
            dim=dim, fpn_dims=fpn_dims, context_dim=context_dim
        )
        mask_head_pytorch.eval()

        with torch.no_grad():
            out_pytorch = mask_head_pytorch(x_torch, bbox_mask_torch, fpns_torch)

        expected_batch = batch_size * num_queries
        print(f"Output Shape: {list(out_pytorch.shape)}")
        print(f"Expected:     [{expected_batch}, 1, {H*8}, {W*8}]")

        # TTSim Forward
        print_subsection("TTSim Forward Pass")

        x_ttsim = torch_to_simtensor(x_torch, "x")
        bbox_mask_ttsim = torch_to_simtensor(bbox_mask_torch, "bbox_mask")
        fpn0_ttsim = torch_to_simtensor(fpn0_torch, "fpn0")
        fpn1_ttsim = torch_to_simtensor(fpn1_torch, "fpn1")
        fpn2_ttsim = torch_to_simtensor(fpn2_torch, "fpn2")
        fpns_ttsim = [fpn0_ttsim, fpn1_ttsim, fpn2_ttsim]

        mask_head_ttsim = MaskHeadSmallConvTTSim(
            name="mask_head_test", dim=dim, fpn_dims=fpn_dims, context_dim=context_dim
        )

        out_ttsim = mask_head_ttsim(x_ttsim, bbox_mask_ttsim, fpns_ttsim)

        print(f"Output Shape: {list(out_ttsim.shape)}")

        # Comparison
        print_subsection("Shape Comparison")

        result = compare_shapes(out_pytorch, out_ttsim, "MaskHeadSmallConv")

        # Parameter count
        print(f"\nParameter Count:")
        pytorch_params = sum(p.numel() for p in mask_head_pytorch.parameters())
        ttsim_params = mask_head_ttsim.analytical_param_count()
        param_match = pytorch_params == ttsim_params
        print(f"  PyTorch: {pytorch_params:,}")
        print(f"  TTSim:   {ttsim_params:,}")
        print(f"  Match:   {'✓' if param_match else '✗'}")

        return result and param_match

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        diagnose_error(e, "MaskHeadSmallConv")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Test 3: DETRsegm
# ============================================================================


def test_detrsegm():
    """
    Test DETRsegm wrapper module.

    Known Issue: Inherits MaskHeadSmallConv concat error
    """

    print_section("TEST 3: DETRsegm (Integrated Module)")

    # Configuration
    batch_size = 2
    num_queries = 100
    hidden_dim = 256
    num_heads = 8
    H, W = 16, 16

    print("\nConfiguration:")
    print(f"  Batch size:   {batch_size}")
    print(f"  Num queries:  {num_queries}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Num heads:    {num_heads}")
    print(f"  Spatial size: {H} x {W}")

    try:
        # Mock DETR
        class MockDETR:
            def __init__(self):
                self.num_queries = num_queries

        mock_detr = MockDETR()

        # Create inputs
        torch.manual_seed(42)
        np.random.seed(42)

        features_torch = torch.randn(batch_size, hidden_dim, H, W)
        pos_torch = [torch.randn(batch_size, hidden_dim, H, W)]
        query_embed_torch = torch.randn(num_queries, hidden_dim)

        print(f"\nInput Shapes (Mocked):")
        print(f"  features:     {list(features_torch.shape)}")
        print(f"  pos:          {[list(p.shape) for p in pos_torch]}")
        print(f"  query_embed:  {list(query_embed_torch.shape)}")

        # TTSim Forward
        print_subsection("TTSim Forward Pass")

        features_ttsim = torch_to_simtensor(features_torch, "features")
        pos_ttsim = [torch_to_simtensor(pos_torch[0], "pos0")]
        query_embed_ttsim = torch_to_simtensor(query_embed_torch, "query_embed")

        detrsegm_ttsim = DETRsegmTTSim(
            name="detrsegm_test",
            detr=mock_detr,
            hidden_dim=hidden_dim,
            nheads=num_heads,
            freeze_detr=False,
        )

        samples_ttsim = [features_ttsim]

        out_ttsim = detrsegm_ttsim(
            samples_ttsim, [features_ttsim], pos_ttsim, query_embed_ttsim
        )

        pred_masks = out_ttsim["pred_masks"]

        print(f"Output Shape: {list(pred_masks.shape)}")
        print(f"Expected:     [{batch_size}, {num_queries}, H_out, W_out]")

        # Validation
        print_subsection("Shape Validation")

        expected_dims = 4
        expected_batch = batch_size
        expected_queries = num_queries

        shape_valid = (
            len(pred_masks.shape) == expected_dims
            and pred_masks.shape[0] == expected_batch
            and pred_masks.shape[1] == expected_queries
        )

        if shape_valid:
            print(f"\n✓ PASS - Output shape is valid")
            print(f"  Batch:    {pred_masks.shape[0]} (expected {expected_batch})")
            print(f"  Queries:  {pred_masks.shape[1]} (expected {expected_queries})")
            print(f"  Spatial:  {pred_masks.shape[2]} x {pred_masks.shape[3]}")
        else:
            print(f"\n✗ FAIL - Output shape is invalid")
            print(f"  Got:      {list(pred_masks.shape)}")
            print(f"  Expected: [B={expected_batch}, Q={expected_queries}, H, W]")

        print(f"\nParameter Count:")
        ttsim_params = detrsegm_ttsim.analytical_param_count()
        print(f"  TTSim (segmentation only): {ttsim_params:,}")

        return shape_valid

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        diagnose_error(e, "DETRsegm")
        import traceback

        traceback.print_exc()
        return False


# ============================================================================
# Test 4: Diagnostic Tests for Common Operations
# ============================================================================


def test_operation_diagnostics():
    """
    Run diagnostic tests for problematic TTSim operations.
    """

    print_section("TEST 4: Operation Diagnostics")

    all_passed = True

    # Test 4.1: Unsqueeze operation
    print_subsection("4.1: Unsqueeze Operation")

    try:
        import ttsim.front.functional.op as F

        # Create test tensor
        test_tensor = SimTensor(
            {
                "name": "test_unsqueeze",
                "shape": [2, 16, 16],
                "data": np.random.randn(2, 16, 16).astype(np.float32),
                "dtype": np.dtype(np.float32),
            }
        )

        print(f"Input shape: {test_tensor.shape}")

        # Test single unsqueeze - TTSim requires axes as a separate tensor input
        unsqueeze_op = F.Unsqueeze("test_unsqueeze_op")
        axes_tensor = F._from_data(
            "test_unsqueeze_op.axes", data=np.array([1], dtype=np.int64), is_const=True
        )
        result = unsqueeze_op(test_tensor, axes_tensor)

        print(f"After unsqueeze(axes=[1]): {result.shape}")
        print(f"Expected: [2, 1, 16, 16]")

        if result.shape == [2, 1, 16, 16]:
            print("✓ Unsqueeze works correctly")
        else:
            print("✗ Unsqueeze shape mismatch")
            all_passed = False

    except Exception as e:
        print(f"✗ Unsqueeze test failed: {str(e)}")
        all_passed = False

    # Test 4.2: Tile operation
    print_subsection("4.2: Tile Operation")

    try:
        import ttsim.front.functional.op as F

        test_tensor = SimTensor(
            {
                "name": "test_tile",
                "shape": [2, 256, 16, 16],
                "data": np.random.randn(2, 256, 16, 16).astype(np.float32),
                "dtype": np.dtype(np.float32),
            }
        )

        print(f"Input shape: {test_tensor.shape}")

        # Test tile along batch dimension
        tile_op = F.Tile("test_tile_op")
        repeats_tensor = F._from_data(
            "repeats", data=np.array([50, 1, 1, 1], dtype=np.int64), is_const=True
        )
        result = tile_op(test_tensor, repeats_tensor)

        print(f"After tile([50, 1, 1, 1]): {result.shape}")
        print(f"Expected: [100, 256, 16, 16]")

        if result.shape == [100, 256, 16, 16]:
            print("✓ Tile works correctly")
        else:
            print("✗ Tile shape mismatch")
            all_passed = False

    except Exception as e:
        print(f"✗ Tile test failed: {str(e)}")
        all_passed = False

    # Test 4.3: Concat operation
    print_subsection("4.3: Concat Operation")

    try:
        import ttsim.front.functional.op as F

        tensor1 = SimTensor(
            {
                "name": "concat_t1",
                "shape": [200, 256, 16, 16],
                "data": np.random.randn(200, 256, 16, 16).astype(np.float32),
                "dtype": np.dtype(np.float32),
            }
        )

        tensor2 = SimTensor(
            {
                "name": "concat_t2",
                "shape": [200, 8, 16, 16],
                "data": np.random.randn(200, 8, 16, 16).astype(np.float32),
                "dtype": np.dtype(np.float32),
            }
        )

        print(f"Input 1 shape: {tensor1.shape}")
        print(f"Input 2 shape: {tensor2.shape}")

        concat_op = F.ConcatX("test_concat", axis=1)
        result = concat_op(tensor1, tensor2)

        print(f"After concat(axis=1): {result.shape}")
        print(f"Expected: [200, 264, 16, 16]")

        if result.shape == [200, 264, 16, 16]:
            print("✓ Concat works correctly")
        else:
            print("✗ Concat shape mismatch")
            all_passed = False

    except Exception as e:
        print(f"✗ Concat test failed: {str(e)}")
        all_passed = False

    return all_passed


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all segmentation module tests with diagnostics."""

    start_time = datetime.now()

    print_section("DEFORMABLE DETR SEGMENTATION - COMPREHENSIVE TEST SUITE", "=", 80)
    print(f"\nStart Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjective: Validate shape consistency and diagnose TTSim errors")
    print("Focus: Shape validation + error diagnosis + operation testing")

    results = {}

    # Run operation diagnostics first
    results["Diagnostics"] = test_operation_diagnostics()

    # Test main modules
    results["MHAttentionMap"] = test_mh_attention_map()
    results["MaskHeadSmallConv"] = test_mask_head_small_conv()
    results["DETRsegm"] = test_detrsegm()

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_section("TEST SUMMARY", "=", 80)

    print("\nResults:")
    for module_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {module_name}")

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    print(f"\nStatistics:")
    print(f"  Total Tests:  {total_tests}")
    print(f"  Passed:       {passed_tests}")
    print(f"  Failed:       {total_tests - passed_tests}")
    print(f"  Pass Rate:    {(passed_tests/total_tests*100):.1f}%")
    print(f"  Duration:     {duration:.2f} seconds")

    print(f"\nEnd Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Error summary
    if not all(results.values()):
        print_section("ERROR SUMMARY & FIXES NEEDED", "=", 80)

        if not results.get("MHAttentionMap", True):
            print("\n1. MHAttentionMap - Unsqueeze ipos mismatch")
            print("   Fix: Check Unsqueeze operation construction in masked_fill_impl")
            print("   Line: segmentation_ttsim.py, unsqueeze operations")

        if not results.get("MaskHeadSmallConv", True):
            print("\n2. MaskHeadSmallConv - Concat shape mismatch")
            print("   Fix: Verify expand() function Tile logic")
            print("   Issue: x_expanded shape not matching bbox_mask_flat")
            print(
                "   Line: segmentation_ttsim.py, expand function and concat operation"
            )

        if not results.get("DETRsegm", True):
            print("\n3. DETRsegm - Inherits MaskHeadSmallConv errors")
            print("   Fix: Resolve MaskHeadSmallConv issues first")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print(" " * 25 + "🎉 ALL TESTS PASSED! 🎉")
    else:
        print(" " * 25 + "⚠️  TESTS FAILED - SEE DIAGNOSTICS  ⚠️")
    print("=" * 80 + "\n")

    return all_passed


class _TeeStream:
    """Write to both a file and the original stream simultaneously."""

    def __init__(self, original, filepath):
        self._original = original
        self._file = open(filepath, "w", encoding="utf-8")

    def write(self, text):
        self._original.write(text)
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    REPORT_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "reports")
    )
    os.makedirs(REPORT_DIR, exist_ok=True)
    REPORT_PATH = os.path.join(REPORT_DIR, "segmentation_shape_validation.md")

    tee = _TeeStream(sys.stdout, REPORT_PATH)
    sys.stdout = tee

    success = run_all_tests()
    print(f"\n\n*Report saved to: {REPORT_PATH}*")
    tee.close()
    sys.stdout = tee._original
    sys.exit(0 if success else 1)
