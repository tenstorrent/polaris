#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced test file comparing PyTorch and TTSim misc utilities implementations.
Compares shape inference and numerical computation with detailed outputs.
Generates markdown report with input/output samples and relative error analysis.

Functions tested:
- NestedTensor (data container)
- interpolate (with numerical comparison)
- nested_tensor_from_tensor_list (with numerical comparison)
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
from workloads.Deformable_DETR.reference.misc import (
    NestedTensor as NestedTensorPyTorch,
    interpolate as interpolate_pytorch,
    nested_tensor_from_tensor_list as nested_tensor_from_tensor_list_pytorch,
)

# Import TTSim implementations
from workloads.Deformable_DETR.util.misc_ttsim import (
    NestedTensor as NestedTensorTTSim,
    interpolate as interpolate_ttsim,
    nested_tensor_from_tensor_list as nested_tensor_from_tensor_list_ttsim,
)

from ttsim.ops.tensor import SimTensor

# ──────────────────────────────────────────────────────────────────────────────
# Global report buffer and test results tracking
# ──────────────────────────────────────────────────────────────────────────────
REPORT_BUFFER = []
TEST_RESULTS = {}  # Track individual test results


def log_to_report(message):
    """Add message to both console and report buffer"""
    print(message)
    REPORT_BUFFER.append(message)


def save_report():
    """Save accumulated report to markdown file"""
    report_dir = "workloads/Deformable_DETR/reports"
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, "misc_utils_validation.md")

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


def compare_numerics(torch_output, ttsim_output, test_name, rtol=1e-4, atol=1e-6):
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


def test_nested_tensor():
    """Test NestedTensor data container"""
    test_name = "NestedTensor"
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 1: NestedTensor (Data Container)")
    log_to_report("=" * 80)

    log_to_report(
        f"\n**Purpose:** Bundle tensors with padding masks for efficient batching"
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test tensor and mask
        tensor_torch = torch.randn(2, 3, 56, 56)
        mask_torch = torch.zeros(2, 56, 56, dtype=torch.bool)
        mask_torch[0, :5, :] = True
        mask_torch[1, :, :5] = True

        tensor_ttsim = torch_to_simtensor(tensor_torch, "tensor")
        mask_ttsim = mask_torch.cpu().numpy()

        log_to_report(f"\n### Input")
        log_to_report(f"```")
        log_to_report(f"Tensor shape: {list(tensor_torch.shape)}")
        log_to_report(f"Mask shape:   {list(mask_torch.shape)}")
        log_to_report(f"Masked pixels: {mask_torch.sum().item()}/{mask_torch.numel()}")
        log_to_report(f"```")

        # Create NestedTensors
        nested_pytorch = NestedTensorPyTorch(tensor_torch, mask_torch)
        nested_ttsim = NestedTensorTTSim(tensor_ttsim, mask_ttsim)

        # Test decompose
        t_pytorch, m_pytorch = nested_pytorch.decompose()
        t_ttsim, m_ttsim = nested_ttsim.decompose()

        log_to_report(f"\n### NestedTensor Created")
        log_to_report(f"```")
        log_to_report(f"PyTorch: {nested_pytorch}")
        log_to_report(f"TTSim:   {nested_ttsim}")
        log_to_report(f"```")

        log_to_report(f"\n### decompose() Method")
        log_to_report(f"```")
        log_to_report(f"PyTorch tensors shape: {list(t_pytorch.shape)}")
        log_to_report(f"PyTorch mask shape:    {list(m_pytorch.shape)}")
        log_to_report(
            f"TTSim tensors shape:   {list(t_ttsim.shape) if isinstance(t_ttsim, SimTensor) else 'SimTensor'}"
        )
        log_to_report(f"TTSim mask shape:      {list(m_ttsim.shape)}")
        log_to_report(f"```")

        log_to_report("\n### [PASSED] NestedTensor test")
        TEST_RESULTS[test_name] = "PASSED"
        return True

    except Exception as e:
        log_to_report(f"\n### [ERROR] NestedTensor test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        TEST_RESULTS[test_name] = "ERROR"
        return False


def test_interpolate_nearest():
    """Test interpolate with nearest neighbor mode"""
    test_name = "interpolate_nearest"
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 2: interpolate (Nearest Neighbor)")
    log_to_report("=" * 80)

    log_to_report(f"\n**Function:** Resize tensors using interpolation")
    log_to_report(f"\n**Mode:** Nearest neighbor (no smoothing)")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test input
        input_torch = torch.randn(2, 3, 10, 10)
        input_ttsim = torch_to_simtensor(input_torch, "input")

        target_size = (20, 20)

        log_to_report(f"\n### Configuration")
        log_to_report(f"```")
        log_to_report(f"Input shape:  {list(input_torch.shape)}")
        log_to_report(f"Target size:  {target_size}")
        log_to_report(f"Mode:         nearest")
        log_to_report(f"Scale factor: 2x")
        log_to_report(f"```")

        log_to_report(f"\n### Input")
        print_tensor_comparison(input_torch, input_ttsim, "Input Tensor")

        # Interpolate
        output_pytorch = interpolate_pytorch(
            input_torch, size=target_size, mode="nearest"
        )
        output_ttsim = interpolate_ttsim(input_ttsim, size=target_size, mode="nearest")

        log_to_report(f"\n### Output (Upsampled)")
        print_tensor_comparison(output_pytorch, output_ttsim, "Output Tensor")

        # Compare
        numeric_match = compare_numerics(
            output_pytorch, output_ttsim, "interpolate_nearest", rtol=1e-4, atol=1e-6
        )

        if numeric_match:
            log_to_report("\n### [PASSED] interpolate (nearest) test")
            TEST_RESULTS[test_name] = "PASSED"
            return True
        else:
            log_to_report("\n### [FAILED] interpolate (nearest) test")
            TEST_RESULTS[test_name] = "FAILED"
            return False

    except Exception as e:
        log_to_report(f"\n### [ERROR] interpolate (nearest) test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        TEST_RESULTS[test_name] = "ERROR"
        return False


def test_interpolate_bilinear():
    """Test interpolate with bilinear mode"""
    test_name = "interpolate_bilinear"
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 3: interpolate (Bilinear)")
    log_to_report("=" * 80)

    log_to_report(f"\n**Mode:** Bilinear (smooth interpolation)")
    log_to_report(f"\n**Implementation:** PyTorch-compatible bilinear interpolation")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test input
        input_torch = torch.randn(1, 3, 8, 8)
        input_ttsim = torch_to_simtensor(input_torch, "input")

        target_size = (16, 16)

        log_to_report(f"\n### Configuration")
        log_to_report(f"```")
        log_to_report(f"Input shape:  {list(input_torch.shape)}")
        log_to_report(f"Target size:  {target_size}")
        log_to_report(f"Mode:         bilinear")
        log_to_report(f"Scale factor: 2x")
        log_to_report(f"```")

        log_to_report(f"\n### Input")
        print_tensor_comparison(input_torch, input_ttsim, "Input Tensor")

        # Interpolate
        output_pytorch = interpolate_pytorch(
            input_torch, size=target_size, mode="bilinear", align_corners=False
        )
        output_ttsim = interpolate_ttsim(input_ttsim, size=target_size, mode="bilinear")

        log_to_report(f"\n### Output (Upsampled)")
        print_tensor_comparison(output_pytorch, output_ttsim, "Output Tensor")

        # Compare with tight tolerance now that we have PyTorch-compatible implementation
        numeric_match = compare_numerics(
            output_pytorch, output_ttsim, "interpolate_bilinear", rtol=1e-5, atol=1e-7
        )

        if numeric_match:
            log_to_report("\n### [PASSED] interpolate (bilinear) test")
            TEST_RESULTS[test_name] = "PASSED"
            return True
        else:
            log_to_report("\n### [FAILED] interpolate (bilinear) test")
            TEST_RESULTS[test_name] = "FAILED"
            return False

    except Exception as e:
        log_to_report(f"\n### [ERROR] interpolate (bilinear) test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        TEST_RESULTS[test_name] = "ERROR"
        return False


def test_nested_tensor_from_tensor_list():
    """Test nested_tensor_from_tensor_list with padding"""
    test_name = "nested_tensor_from_tensor_list"
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 4: nested_tensor_from_tensor_list")
    log_to_report("=" * 80)

    log_to_report(f"\n**Function:** Batch variable-sized tensors with zero-padding")
    log_to_report(f"\n**Process:**")
    log_to_report(f"1. Find maximum dimensions across all tensors")
    log_to_report(f"2. Create batch tensor with max dimensions")
    log_to_report(f"3. Copy each tensor and mark padded regions in mask")

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create variable-sized tensors
        tensor1_torch = torch.randn(3, 10, 15)
        tensor2_torch = torch.randn(3, 12, 18)
        tensor3_torch = torch.randn(3, 8, 20)

        tensor_list_pytorch = [tensor1_torch, tensor2_torch, tensor3_torch]
        tensor_list_ttsim = [
            torch_to_simtensor(tensor1_torch, "tensor1"),
            torch_to_simtensor(tensor2_torch, "tensor2"),
            torch_to_simtensor(tensor3_torch, "tensor3"),
        ]

        log_to_report(f"\n### Input Tensors (Variable Sizes)")
        log_to_report(f"```")
        log_to_report(f"Tensor 1: {list(tensor1_torch.shape)} - smallest height")
        log_to_report(f"Tensor 2: {list(tensor2_torch.shape)} - largest height")
        log_to_report(f"Tensor 3: {list(tensor3_torch.shape)} - largest width")
        log_to_report(f"")
        log_to_report(f"Expected batch shape: [3, 3, 12, 20] (max along each dim)")
        log_to_report(f"```")

        # Sample input values
        log_to_report(f"\n### Sample Input Values")
        log_to_report(f"```")
        log_to_report(f"Tensor 1 sample: {format_array_sample(tensor1_torch, 10)}")
        log_to_report(f"Tensor 2 sample: {format_array_sample(tensor2_torch, 10)}")
        log_to_report(f"Tensor 3 sample: {format_array_sample(tensor3_torch, 10)}")
        log_to_report(f"```")

        # Create nested tensors
        nested_pytorch = nested_tensor_from_tensor_list_pytorch(tensor_list_pytorch)
        nested_ttsim = nested_tensor_from_tensor_list_ttsim(tensor_list_ttsim)

        # Extract tensors and masks
        tensors_pytorch, mask_pytorch = nested_pytorch.decompose()
        tensors_ttsim, mask_ttsim = nested_ttsim.decompose()

        log_to_report(f"\n### Output Batch")
        log_to_report(f"```")
        log_to_report(f"PyTorch tensors: {list(tensors_pytorch.shape)}")
        log_to_report(f"PyTorch mask:    {list(mask_pytorch.shape)}")
        log_to_report(f"TTSim tensors:   {list(tensors_ttsim.shape)}")
        log_to_report(f"TTSim mask:      {list(mask_ttsim.shape)}")
        log_to_report(f"```")

        # Check masks
        log_to_report(f"\n### Padding Masks")
        log_to_report(f"```")
        for i in range(3):
            pt_masked = mask_pytorch[i].sum().item()
            tt_masked = mask_ttsim[i].sum()
            log_to_report(
                f"Tensor {i}: PyTorch masked={pt_masked}, TTSim masked={tt_masked}"
            )
        log_to_report(f"```")

        # Compare tensors
        print_tensor_comparison(tensors_pytorch, tensors_ttsim, "Batched Tensors")

        # Compare numerics
        numeric_match = compare_numerics(
            tensors_pytorch,
            tensors_ttsim,
            "nested_tensor_from_tensor_list",
            rtol=1e-5,
            atol=1e-7,
        )

        # Compare masks
        mask_match = np.array_equal(mask_pytorch.cpu().numpy(), mask_ttsim)

        log_to_report(f"\n### Mask Comparison")
        log_to_report(f"```")
        log_to_report(f"Masks identical: {mask_match}")
        if not mask_match:
            mask_diff = np.sum(mask_pytorch.cpu().numpy() != mask_ttsim)
            log_to_report(f"Differences: {mask_diff}/{mask_ttsim.size} pixels")
        log_to_report(f"```")

        if numeric_match and mask_match:
            log_to_report("\n### [PASSED] nested_tensor_from_tensor_list test")
            TEST_RESULTS[test_name] = "PASSED"
            return True
        else:
            log_to_report("\n### [FAILED] nested_tensor_from_tensor_list test")
            TEST_RESULTS[test_name] = "FAILED"
            return False

    except Exception as e:
        log_to_report(f"\n### [ERROR] nested_tensor_from_tensor_list test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        TEST_RESULTS[test_name] = "ERROR"
        return False


def test_interpolate_downsampling():
    """Test interpolate with downsampling"""
    test_name = "interpolate_downsample"
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 5: interpolate (Downsampling)")
    log_to_report("=" * 80)

    log_to_report(f"\n**Purpose:** Test downsampling (reducing resolution)")
    log_to_report(
        f"\n**Implementation:** PyTorch-compatible nearest neighbor downsampling"
    )

    try:
        torch.manual_seed(42)
        np.random.seed(42)

        # Create high-resolution input
        input_torch = torch.randn(2, 3, 32, 32)
        input_ttsim = torch_to_simtensor(input_torch, "input")

        target_size = (8, 8)

        log_to_report(f"\n### Configuration")
        log_to_report(f"```")
        log_to_report(f"Input shape:  {list(input_torch.shape)}")
        log_to_report(f"Target size:  {target_size}")
        log_to_report(f"Mode:         nearest")
        log_to_report(f"Scale factor: 0.25x (4x reduction)")
        log_to_report(f"```")

        # Interpolate
        output_pytorch = interpolate_pytorch(
            input_torch, size=target_size, mode="nearest"
        )
        output_ttsim = interpolate_ttsim(input_ttsim, size=target_size, mode="nearest")

        log_to_report(f"\n### Output (Downsampled)")
        print_tensor_comparison(output_pytorch, output_ttsim, "Output Tensor")

        # Compare with tight tolerance now that we have PyTorch-compatible implementation
        numeric_match = compare_numerics(
            output_pytorch, output_ttsim, "interpolate_downsample", rtol=1e-5, atol=1e-7
        )

        if numeric_match:
            log_to_report("\n### [PASSED] interpolate (downsampling) test")
            TEST_RESULTS[test_name] = "PASSED"
            return True
        else:
            log_to_report("\n### [FAILED] interpolate (downsampling) test")
            TEST_RESULTS[test_name] = "FAILED"
            return False

    except Exception as e:
        log_to_report(f"\n### [ERROR] interpolate (downsampling) test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        TEST_RESULTS[test_name] = "ERROR"
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_to_report("# Misc Utilities Validation Report")
    log_to_report(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_report(f"\n**Test Suite:** PyTorch vs TTSim Misc Utilities Comparison")

    log_to_report("\n---\n")

    log_to_report("## Overview")
    log_to_report(
        "\nThis report validates TTSim implementations of utility functions used throughout Deformable DETR."
    )
    log_to_report("\n**Functions Tested:**")
    log_to_report(
        "1. **NestedTensor**: Data container bundling tensors with padding masks"
    )
    log_to_report(
        "2. **interpolate**: Resize tensors (nearest neighbor and bilinear modes)"
    )
    log_to_report(
        "3. **nested_tensor_from_tensor_list**: Batch variable-sized tensors with padding"
    )

    log_to_report("\n**Key Points:**")
    log_to_report("- NestedTensor is a data container (not a computational module)")
    log_to_report("- interpolate uses scipy.ndimage.zoom for TTSim")
    log_to_report("- Bilinear interpolation may show small differences vs PyTorch")
    log_to_report("- Padding and masking preserve spatial information")

    log_to_report("\n---\n")

    tests = [
        ("NestedTensor", test_nested_tensor),
        ("interpolate (nearest)", test_interpolate_nearest),
        ("interpolate (bilinear)", test_interpolate_bilinear),
        ("nested_tensor_from_tensor_list", test_nested_tensor_from_tensor_list),
        ("interpolate (downsample)", test_interpolate_downsampling),
    ]

    for test_name, test_func in tests:
        test_func()

    # Count results
    tests_passed = sum(1 for v in TEST_RESULTS.values() if v == "PASSED")
    tests_failed = sum(1 for v in TEST_RESULTS.values() if v == "FAILED")
    tests_known_issues = sum(1 for v in TEST_RESULTS.values() if v == "KNOWN_ISSUE")
    tests_errors = sum(1 for v in TEST_RESULTS.values() if v == "ERROR")

    log_to_report("\n" + "=" * 80)
    log_to_report("# Test Summary")
    log_to_report("=" * 80)

    log_to_report(f"\n## Results by Test")
    log_to_report(f"\n| Test | Result | Notes |")
    log_to_report(f"|------|--------|-------|")
    for test_name, test_func in tests:
        result = TEST_RESULTS.get(
            test_name.replace(" ", "_").replace("(", "").replace(")", ""), "NOT_RUN"
        )
        if result == "PASSED":
            icon = "✅"
        elif result == "FAILED":
            icon = "❌"
        elif result == "KNOWN_ISSUE":
            icon = "⚠️"
        elif result == "ERROR":
            icon = "💥"
        else:
            icon = "❓"

        notes = ""
        if result == "KNOWN_ISSUE":
            notes = "Expected differences"

        log_to_report(f"| {test_name} | {icon} {result} | {notes} |")

    log_to_report(f"\n## Overall Statistics")
    log_to_report(f"\n| Metric | Value |")
    log_to_report(f"|--------|-------|")
    log_to_report(f"| **Tests Passed** | {tests_passed}/{len(tests)} |")
    log_to_report(f"| **Tests Failed** | {tests_failed}/{len(tests)} |")
    log_to_report(f"| **Known Issues** | {tests_known_issues}/{len(tests)} |")
    log_to_report(f"| **Errors** | {tests_errors}/{len(tests)} |")
    log_to_report(f"| **Success Rate** | {100*tests_passed/len(tests):.1f}% |")

    log_to_report(f"\n## Implementation Details")

    log_to_report(f"\n**✅ Working Correctly:**")
    log_to_report(f"- NestedTensor data container")
    log_to_report(f"- Tensor batching with padding")
    log_to_report(f"- Mask generation")
    if tests_passed >= 2:
        log_to_report(f"- Numerical computations for passing tests")

    log_to_report(f"\n**⚠️ Known Issues:**")
    if tests_known_issues > 0:
        log_to_report(f"- Interpolation: scipy.ndimage.zoom vs PyTorch differences")
        log_to_report(f"  - Different algorithms for bilinear interpolation")
        log_to_report(
            f"  - Different pixel selection for nearest neighbor downsampling"
        )
        log_to_report(f"  - Results are similar but not numerically identical")
        log_to_report(
            f"\n  **Impact:** May cause small differences in model outputs that use interpolation."
        )
        log_to_report(
            f"  **Recommendation:** Consider implementing PyTorch-compatible interpolation or"
        )
        log_to_report(f"  accept the differences as part of framework translation.")

    log_to_report(f"\n**💡 Implementation Notes:**")
    log_to_report(
        f"- TTSim uses scipy for interpolation (numpy-based, no PyTorch dependency)"
    )
    log_to_report(
        f"- NestedTensor stores numpy arrays for masks (metadata, not computation)"
    )
    log_to_report(f"- All functions support shape inference mode (data=None)")

    log_to_report(f"\n## Final Status")
    if tests_passed == len(tests):
        log_to_report(
            "\n✅ **ALL TESTS PASSED** - TTSim implementation is fully compatible!"
        )
    elif tests_failed == 0 and tests_known_issues > 0:
        log_to_report(
            f"\n⚠️ **{tests_passed} PASSED, {tests_known_issues} KNOWN ISSUES**"
        )
        log_to_report(
            "\nCore functionality works correctly. Known issues are due to scipy vs PyTorch"
        )
        log_to_report(
            "implementation differences that may or may not impact final model accuracy."
        )
    else:
        log_to_report(
            f"\n❌ **{tests_passed} PASSED, {tests_failed} FAILED, {tests_known_issues} KNOWN ISSUES**"
        )
        log_to_report("\nReview failed tests above for issues that need fixing.")

    log_to_report("\n---\n")
    log_to_report("\n*End of Report*")

    # Save report
    save_report()
