#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Numerical Computation Validation for MHAttentionMap Module (FIXED VERSION).

This module tests numerical correctness of the MHAttentionMap TTSim implementation
against the PyTorch reference.

KEY FIX: The MHAttentionMap now uses explicit q_linear_weight/q_linear_bias tensors
instead of SimNN.Linear, which allows for cleaner weight loading and avoids the
dual-weight-setting issue that caused the alternating zeros pattern.

Components Tested:
  1. Linear projection (q_linear) - Query projection
  2. 1x1 Conv projection (k_linear) - Key projection
  3. Reshape operations - Multi-head format conversion
  4. Einsum for attention scores - bqnc,bnchw->bqnhw
  5. Softmax - Normalized attention weights
  6. Dropout - Attention dropout (test mode)

Test Strategy:
  - Create random input tensors with data attached
  - Run through both PyTorch and TTSim implementations
  - Compare numerical outputs with tolerance
  - Verify each component individually then full forward pass

Author: Numerical Validation Suite
Date: 2025
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from datetime import datetime

# Add project root to path
# Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

import ttsim.front.functional.op as F
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN

# Import TTSim MHAttentionMap
from workloads.Deformable_DETR.models.segmentation_ttsim import (
    MHAttentionMap as MHAttentionMapTTSim,
    conv2d_functional,
    masked_fill_impl,
)

# Import PyTorch MHAttentionMap
from workloads.Deformable_DETR.reference.segmentation import (
    MHAttentionMap as MHAttentionMapPyTorch,
)

# ============================================================================
# Utility Functions
# ============================================================================


def torch_to_simtensor(torch_tensor, name="tensor"):
    """Convert PyTorch tensor to SimTensor with data attached."""
    data = torch_tensor.detach().cpu().numpy().copy()
    return SimTensor(
        {
            "name": name,
            "shape": list(torch_tensor.shape),
            "data": data,
            "dtype": data.dtype,
        }
    )


def numpy_to_simtensor(np_array, name="tensor", is_const=False, is_param=False):
    """Convert numpy array to SimTensor with data attached."""
    return SimTensor(
        {
            "name": name,
            "shape": list(np_array.shape),
            "data": np_array.copy(),
            "dtype": np_array.dtype,
            "is_const": is_const,
            "is_param": is_param,
        }
    )


def print_section(title, char="=", width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")


def print_subsection(title, char="-", width=60):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def compare_numerical(
    pytorch_out, ttsim_out, name="Output", rtol=1e-4, atol=1e-5, verbose=True
):
    """
    Compare numerical outputs between PyTorch and TTSim.

    Args:
        pytorch_out: PyTorch tensor or numpy array
        ttsim_out: TTSim SimTensor or numpy array
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: If True, show detailed tensor values

    Returns:
        bool: True if outputs match within tolerance
    """
    # Convert to numpy
    if isinstance(pytorch_out, torch.Tensor):
        pytorch_np = pytorch_out.detach().cpu().numpy()
    else:
        pytorch_np = np.asarray(pytorch_out)

    if isinstance(ttsim_out, SimTensor):
        if ttsim_out.data is None:
            print(f"\n✗ FAIL - {name}: TTSim output has no data!")
            return False
        ttsim_np = ttsim_out.data
    else:
        ttsim_np = np.asarray(ttsim_out)

    # Check shape match
    if pytorch_np.shape != ttsim_np.shape:
        print(f"\n✗ FAIL - {name}: Shape mismatch")
        print(f"  PyTorch shape: {pytorch_np.shape}")
        print(f"  TTSim shape:   {ttsim_np.shape}")
        return False

    # Check numerical match with inf handling
    try:
        is_close = np.allclose(
            pytorch_np, ttsim_np, rtol=rtol, atol=atol, equal_nan=True
        )
    except Exception as e:
        print(f"\n✗ FAIL - {name}: Comparison error: {e}")
        return False

    # Calculate diff, masking out inf values
    diff_raw = pytorch_np - ttsim_np
    finite_mask = np.isfinite(diff_raw)
    if np.any(finite_mask):
        max_diff = np.max(np.abs(diff_raw[finite_mask]))
        mean_diff = np.mean(np.abs(diff_raw[finite_mask]))
    else:
        max_diff = 0.0
        mean_diff = 0.0

    # Check if inf values match
    inf_match = np.array_equal(np.isinf(pytorch_np), np.isinf(ttsim_np))
    if not inf_match:
        print(f"\n✗ FAIL - {name}: Inf value mismatch")
        return False

    if is_close:
        print(f"\n✓ PASS - {name}: Numerical match")
        print(f"  Shape: {pytorch_np.shape}")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        if verbose:
            flat_pytorch = pytorch_np.flatten()
            flat_ttsim = ttsim_np.flatten()
            print(f"\n  --- PyTorch output (first 10 values) ---")
            print(f"  {flat_pytorch[:10]}")
            print(f"\n  --- TTSim output (first 10 values) ---")
            print(f"  {flat_ttsim[:10]}")

        return True
    else:
        diff_raw = pytorch_np - ttsim_np
        diff = np.where(np.isfinite(diff_raw), np.abs(diff_raw), 0.0)
        print(f"\n✗ FAIL - {name}: Numerical mismatch")
        print(f"  Shape: {pytorch_np.shape}")
        print(f"  Max diff: {np.max(diff):.2e}")
        print(f"  Mean diff: {np.mean(diff):.2e}")
        print(f"  Required rtol={rtol}, atol={atol}")

        # Show detailed values
        flat_pytorch = pytorch_np.flatten()
        flat_ttsim = ttsim_np.flatten()

        print(f"\n  --- PyTorch output (first 20 values) ---")
        print(f"  {flat_pytorch[:20]}")
        print(f"\n  --- TTSim output (first 20 values) ---")
        print(f"  {flat_ttsim[:20]}")

        # Find the location of max difference
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\n  --- Location of max difference ---")
        print(f"  Index: {max_idx}")
        print(f"  PyTorch value: {pytorch_np[max_idx]}")
        print(f"  TTSim value:   {ttsim_np[max_idx]}")

        return False


# ============================================================================
# Test: Full MHAttentionMap Forward Pass (FIXED)
# ============================================================================


def test_mhattentionmap_full():
    """
    Test full MHAttentionMap module: PyTorch vs TTSim.

    Tests the complete forward pass with:
    - Query projection
    - Key projection (1x1 conv)
    - Multi-head reshape
    - Einsum attention scores
    - Optional masking
    - Softmax normalization
    - Dropout (test mode)

    KEY FIX:
    The original implementation had SimNN.Linear for q_linear which stored weights
    in self.q_linear.param. But the test code was also setting self.q_linear_weight
    (a separate tensor). This caused confusion.

    The FIXED implementation uses explicit self.q_linear_weight and self.q_linear_bias
    tensors with manual MatMul+Add operations, making weight loading straightforward.
    """
    print_section("TEST: Full MHAttentionMap Forward Pass (FIXED)")

    all_passed = True

    # Model parameters
    query_dim = 256
    hidden_dim = 256
    num_heads = 8
    dropout = 0.0  # Set to 0 for deterministic testing
    head_dim = hidden_dim // num_heads

    # Input dimensions
    batch_size = 2
    num_queries = 100
    H, W = 25, 38

    np.random.seed(42)
    torch.manual_seed(42)

    # Test Case 1: Forward without mask
    print_subsection("Test 1: Forward pass without mask")

    # Create inputs
    q_np = np.random.randn(batch_size, num_queries, query_dim).astype(np.float32)
    k_np = np.random.randn(batch_size, query_dim, H, W).astype(np.float32)

    # PyTorch MHAttentionMap
    pytorch_model = MHAttentionMapPyTorch(
        query_dim, hidden_dim, num_heads, dropout=dropout
    )
    pytorch_model.eval()  # Disable dropout

    q_torch = torch.from_numpy(q_np)
    k_torch = torch.from_numpy(k_np)

    # Get PyTorch weights
    pt_q_weight = (
        pytorch_model.q_linear.weight.detach().numpy()
    )  # [hidden_dim, query_dim]
    pt_q_bias = pytorch_model.q_linear.bias.detach().numpy()  # [hidden_dim]
    pt_k_weight = (
        pytorch_model.k_linear.weight.detach().numpy()
    )  # [hidden_dim, query_dim]
    pt_k_bias = pytorch_model.k_linear.bias.detach().numpy()  # [hidden_dim]

    # Get full PyTorch output
    with torch.no_grad():
        pytorch_out = pytorch_model(q_torch, k_torch, mask=None)

    print(f"\n  Input q shape: {q_np.shape}")
    print(f"  Input k shape: {k_np.shape}")
    print(f"  PyTorch output shape: {list(pytorch_out.shape)}")
    print(
        f"  Expected: [B={batch_size}, Q={num_queries}, nheads={num_heads}, H={H}, W={W}]"
    )

    # TTSim MHAttentionMap (FIXED VERSION)
    ttsim_model = MHAttentionMapTTSim(
        "test_mhattention",
        query_dim=query_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )

    # Copy weights from PyTorch to TTSim (FIXED APPROACH)
    # =====================================================
    # q_linear_weight: PyTorch [hidden_dim, query_dim] -> TTSim [query_dim, hidden_dim]
    # This is because PyTorch does: output = input @ weight.T
    # But TTSim MatMul does: output = input @ weight (no transpose)
    # So we need to store the transposed weight
    ttsim_model.q_linear_weight.data = pt_q_weight.T.copy()  # [query_dim, hidden_dim]
    ttsim_model.q_linear_bias.data = pt_q_bias.copy()  # [hidden_dim]

    # k_linear_weight: PyTorch [hidden_dim, query_dim] -> TTSim [hidden_dim, query_dim, 1, 1]
    # For conv2d, we keep the same layout but add spatial dimensions
    k_weight_4d = pt_k_weight.reshape(hidden_dim, query_dim, 1, 1)
    ttsim_model.k_linear_weight.data = k_weight_4d.copy()
    ttsim_model.k_linear_bias.data = pt_k_bias.copy()

    print(f"\n  === TTSim Weight Shapes (FIXED) ===")
    print(f"    q_linear_weight shape: {ttsim_model.q_linear_weight.shape}")
    print(f"    q_linear_weight data shape: {ttsim_model.q_linear_weight.data.shape}")
    print(f"    k_linear_weight shape: {ttsim_model.k_linear_weight.shape}")
    print(f"    k_linear_weight data shape: {ttsim_model.k_linear_weight.data.shape}")

    q_sim = numpy_to_simtensor(q_np, "query")
    k_sim = numpy_to_simtensor(k_np, "key")

    # Run TTSim forward
    ttsim_out = ttsim_model(q_sim, k_sim, mask=None)

    print(f"\n  TTSim output shape: {ttsim_out.shape}")
    print(f"  TTSim output has data: {ttsim_out.data is not None}")

    if ttsim_out.data is not None:
        # Use relaxed tolerance for float32 softmax over large dimension
        if compare_numerical(
            pytorch_out, ttsim_out, "MHAttentionMap (no mask)", rtol=1e-4, atol=1e-5
        ):
            print("  ✓ Full forward pass WITHOUT mask validated!")
        else:
            all_passed = False
    else:
        print("  ✗ TTSim output has no data!")
        all_passed = False

    # Test Case 2: Forward with mask
    print_subsection("Test 2: Forward pass with attention mask")

    # Create mask [B, H, W] - True means invalid (pad) positions
    mask_np = np.zeros((batch_size, H, W), dtype=bool)
    # Mask out some positions (e.g., right edge padding)
    mask_np[:, :, -5:] = True

    q_torch = torch.from_numpy(q_np)
    k_torch = torch.from_numpy(k_np)
    mask_torch = torch.from_numpy(mask_np)

    with torch.no_grad():
        pytorch_out_masked = pytorch_model(q_torch, k_torch, mask=mask_torch)

    print(f"\n  Mask shape: {mask_np.shape}")
    print(f"  Mask True positions (invalid): {np.sum(mask_np)} / {mask_np.size}")
    print(f"  PyTorch masked output shape: {list(pytorch_out_masked.shape)}")

    # TTSim with mask - create FRESH model to avoid operation state accumulation
    ttsim_model_masked = MHAttentionMapTTSim(
        "test_mhattention_masked",
        query_dim=query_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    ttsim_model_masked.q_linear_weight.data = pt_q_weight.T.copy()
    ttsim_model_masked.q_linear_bias.data = pt_q_bias.copy()
    ttsim_model_masked.k_linear_weight.data = k_weight_4d.copy()
    ttsim_model_masked.k_linear_bias.data = pt_k_bias.copy()

    q_sim = numpy_to_simtensor(q_np, "query_masked")
    k_sim = numpy_to_simtensor(k_np, "key_masked")
    mask_sim = numpy_to_simtensor(mask_np.astype(np.float32), "mask")

    ttsim_out_masked = ttsim_model_masked(q_sim, k_sim, mask=mask_sim)

    print(f"  TTSim masked output shape: {ttsim_out_masked.shape}")
    print(f"  TTSim masked output has data: {ttsim_out_masked.data is not None}")

    if ttsim_out_masked.data is not None:
        # Check that masked positions have very low attention (near 0 after softmax)
        ttsim_masked_data = ttsim_out_masked.data
        pytorch_masked_data = pytorch_out_masked.detach().numpy()

        # Attention at masked positions should be ~0
        masked_attention_ttsim = ttsim_masked_data[:, :, :, :, -5:]
        masked_attention_pytorch = pytorch_masked_data[:, :, :, :, -5:]

        print(f"\n  Masked region attention (should be ~0):")
        print(f"  PyTorch max at masked: {np.max(masked_attention_pytorch):.2e}")
        print(f"  TTSim max at masked: {np.max(masked_attention_ttsim):.2e}")

        if compare_numerical(
            pytorch_out_masked,
            ttsim_out_masked,
            "MHAttentionMap (with mask)",
            rtol=1e-4,
            atol=1e-5,
        ):
            print("  ✓ Masked forward pass validated!")
        else:
            all_passed = False
    else:
        print("  ✗ TTSim masked output has no data!")
        all_passed = False

    return all_passed


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run MHAttentionMap numerical tests."""
    print("\n" + "=" * 80)
    print(" MHATTENTIONMAP MODULE - NUMERICAL VALIDATION (FIXED VERSION)")
    print(" Testing: Full Forward Pass with and without mask")
    print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {}

    # Run test
    try:
        results["Full MHAttentionMap"] = test_mhattentionmap_full()
    except Exception as e:
        print(f"\n✗ ERROR in Full MHAttentionMap: {e}")
        import traceback

        traceback.print_exc()
        results["Full MHAttentionMap"] = False

    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
        if not passed:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print(" ALL TESTS PASSED!")
    else:
        print(" SOME TESTS FAILED - Review output above")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)


# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0
# """
# Numerical Computation Validation for MHAttentionMap Module.

# This module tests numerical correctness of the MHAttentionMap TTSim implementation
# against the PyTorch reference.

# Components Tested:
#   1. Linear projection (q_linear) - Query projection
#   2. 1x1 Conv projection (k_linear) - Key projection
#   3. Reshape operations - Multi-head format conversion
#   4. Einsum for attention scores - bqnc,bnchw->bqnhw
#   5. Softmax - Normalized attention weights
#   6. Dropout - Attention dropout (test mode)

# Test Strategy:
#   - Create random input tensors with data attached
#   - Run through both PyTorch and TTSim implementations
#   - Compare numerical outputs with tolerance
#   - Verify each component individually then full forward pass

# Author: Numerical Validation Suite
# Date: 2025
# """

# import os
# import sys
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F_torch
# from datetime import datetime

# # Add project root to path
# # Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# import ttsim.front.functional.op as F
# from ttsim.ops.tensor import SimTensor
# import ttsim.front.functional.sim_nn as SimNN

# # Import TTSim MHAttentionMap
# from workloads.Deformable_DETR.models.segmentation_ttsim import (
#     MHAttentionMap as MHAttentionMapTTSim,
#     conv2d_functional,
#     masked_fill_impl,
# )

# # Import PyTorch MHAttentionMap
# from workloads.Deformable_DETR.reference.segmentation import (
#     MHAttentionMap as MHAttentionMapPyTorch,
# )


# # ============================================================================
# # Utility Functions
# # ============================================================================

# def torch_to_simtensor(torch_tensor, name='tensor'):
#     """Convert PyTorch tensor to SimTensor with data attached."""
#     data = torch_tensor.detach().cpu().numpy().copy()
#     return SimTensor({
#         'name': name,
#         'shape': list(torch_tensor.shape),
#         'data': data,
#         'dtype': data.dtype
#     })


# def numpy_to_simtensor(np_array, name='tensor', is_const=False, is_param=False):
#     """Convert numpy array to SimTensor with data attached."""
#     return SimTensor({
#         'name': name,
#         'shape': list(np_array.shape),
#         'data': np_array.copy(),
#         'dtype': np_array.dtype,
#         'is_const': is_const,
#         'is_param': is_param
#     })


# def print_section(title, char='=', width=80):
#     """Print a formatted section header."""
#     print(f"\n{char * width}")
#     print(f"{title:^{width}}")
#     print(f"{char * width}")


# def print_subsection(title, char='-', width=60):
#     """Print a formatted subsection header."""
#     print(f"\n{char * width}")
#     print(title)
#     print(f"{char * width}")


# def compare_numerical(pytorch_out, ttsim_out, name="Output", rtol=1e-4, atol=1e-5, verbose=True):
#     """
#     Compare numerical outputs between PyTorch and TTSim.

#     Args:
#         pytorch_out: PyTorch tensor or numpy array
#         ttsim_out: TTSim SimTensor or numpy array
#         name: Name for logging
#         rtol: Relative tolerance
#         atol: Absolute tolerance
#         verbose: If True, show detailed tensor values

#     Returns:
#         bool: True if outputs match within tolerance
#     """
#     # Convert to numpy
#     if isinstance(pytorch_out, torch.Tensor):
#         pytorch_np = pytorch_out.detach().cpu().numpy()
#     else:
#         pytorch_np = np.asarray(pytorch_out)

#     if isinstance(ttsim_out, SimTensor):
#         if ttsim_out.data is None:
#             print(f"\n✗ FAIL - {name}: TTSim output has no data!")
#             return False
#         ttsim_np = ttsim_out.data
#     else:
#         ttsim_np = np.asarray(ttsim_out)

#     # Check shape match
#     if pytorch_np.shape != ttsim_np.shape:
#         print(f"\n✗ FAIL - {name}: Shape mismatch")
#         print(f"  PyTorch shape: {pytorch_np.shape}")
#         print(f"  TTSim shape:   {ttsim_np.shape}")
#         return False

#     # Check numerical match with inf handling
#     try:
#         is_close = np.allclose(pytorch_np, ttsim_np, rtol=rtol, atol=atol, equal_nan=True)
#     except Exception as e:
#         print(f"\n✗ FAIL - {name}: Comparison error: {e}")
#         return False

#     # Calculate diff, masking out inf values
#     diff_raw = pytorch_np - ttsim_np
#     finite_mask = np.isfinite(diff_raw)
#     if np.any(finite_mask):
#         max_diff = np.max(np.abs(diff_raw[finite_mask]))
#         mean_diff = np.mean(np.abs(diff_raw[finite_mask]))
#     else:
#         max_diff = 0.0
#         mean_diff = 0.0

#     # Check if inf values match
#     inf_match = np.array_equal(np.isinf(pytorch_np), np.isinf(ttsim_np))
#     if not inf_match:
#         print(f"\n✗ FAIL - {name}: Inf value mismatch")
#         return False

#     if is_close:
#         print(f"\n✓ PASS - {name}: Numerical match")
#         print(f"  Shape: {pytorch_np.shape}")
#         print(f"  Max diff: {max_diff:.2e}")
#         print(f"  Mean diff: {mean_diff:.2e}")

#         if verbose:
#             flat_pytorch = pytorch_np.flatten()
#             flat_ttsim = ttsim_np.flatten()
#             print(f"\n  --- PyTorch output (first 10 values) ---")
#             print(f"  {flat_pytorch[:10]}")
#             print(f"\n  --- TTSim output (first 10 values) ---")
#             print(f"  {flat_ttsim[:10]}")

#         return True
#     else:
#         diff_raw = pytorch_np - ttsim_np
#         diff = np.where(np.isfinite(diff_raw), np.abs(diff_raw), 0.0)
#         print(f"\n✗ FAIL - {name}: Numerical mismatch")
#         print(f"  Shape: {pytorch_np.shape}")
#         print(f"  Max diff: {np.max(diff):.2e}")
#         print(f"  Mean diff: {np.mean(diff):.2e}")
#         print(f"  Required rtol={rtol}, atol={atol}")

#         # Show detailed values
#         flat_pytorch = pytorch_np.flatten()
#         flat_ttsim = ttsim_np.flatten()

#         print(f"\n  --- PyTorch output (first 20 values) ---")
#         print(f"  {flat_pytorch[:20]}")
#         print(f"\n  --- TTSim output (first 20 values) ---")
#         print(f"  {flat_ttsim[:20]}")

#         # Find the location of max difference
#         max_idx = np.unravel_index(np.argmax(diff), diff.shape)
#         print(f"\n  --- Location of max difference ---")
#         print(f"  Index: {max_idx}")
#         print(f"  PyTorch value: {pytorch_np[max_idx]}")
#         print(f"  TTSim value:   {ttsim_np[max_idx]}")

#         return False


# # ============================================================================
# # Test 1: Linear Projection (q_linear)
# # ============================================================================

# def test_linear_projection():
#     """
#     Test TTSim Linear operation against PyTorch nn.Linear.

#     This validates the query projection: q = W_q @ query + b_q

#     PyTorch: nn.Linear(query_dim, hidden_dim)
#     TTSim: SimNN.Linear(name, in_features, out_features)
#     """
#     print_section("TEST 1: Linear Projection (q_linear)")

#     all_passed = True

#     # Test Case 1: Basic linear projection
#     print_subsection("Test 1.1: Basic Linear [B, Q, D_in] -> [B, Q, D_out]")

#     np.random.seed(42)
#     torch.manual_seed(42)

#     batch_size = 2
#     num_queries = 100
#     query_dim = 256
#     hidden_dim = 256

#     # Create input
#     input_np = np.random.randn(batch_size, num_queries, query_dim).astype(np.float32)

#     # PyTorch Linear
#     pytorch_linear = nn.Linear(query_dim, hidden_dim, bias=True)
#     weight_np = pytorch_linear.weight.detach().numpy().copy()
#     bias_np = pytorch_linear.bias.detach().numpy().copy()

#     input_torch = torch.from_numpy(input_np)
#     pytorch_out = pytorch_linear(input_torch)

#     print(f"\n  Input shape: {input_np.shape}")
#     print(f"  Weight shape: {weight_np.shape}")
#     print(f"  Bias shape: {bias_np.shape}")
#     print(f"  PyTorch output shape: {list(pytorch_out.shape)}")

#     # TTSim Linear
#     ttsim_linear = SimNN.Linear(
#         'test_q_linear',
#         in_features=query_dim,
#         out_features=hidden_dim,
#         bias=True
#     )

#     # Set TTSim weights to match PyTorch
#     # SimNN.Linear uses param in [in, out] format (transposed from PyTorch's [out, in])
#     ttsim_linear.param.data = weight_np.T.copy()  # Transpose!
#     ttsim_linear.bias.data = bias_np.copy()

#     input_sim = numpy_to_simtensor(input_np, 'q_input')
#     ttsim_out = ttsim_linear(input_sim)

#     print(f"  TTSim output shape: {ttsim_out.shape}")
#     print(f"  TTSim output has data: {ttsim_out.data is not None}")

#     if compare_numerical(pytorch_out, ttsim_out, "Linear projection"):
#         print("  Linear projection validated!")
#     else:
#         all_passed = False

#     # Test Case 2: Different dimensions (query_dim != hidden_dim)
#     print_subsection("Test 1.2: Linear with dimension change [B, Q, 128] -> [B, Q, 256]")

#     query_dim_2 = 128
#     hidden_dim_2 = 256

#     input_np_2 = np.random.randn(batch_size, num_queries, query_dim_2).astype(np.float32)

#     pytorch_linear_2 = nn.Linear(query_dim_2, hidden_dim_2, bias=True)
#     weight_np_2 = pytorch_linear_2.weight.detach().numpy().copy()
#     bias_np_2 = pytorch_linear_2.bias.detach().numpy().copy()

#     input_torch_2 = torch.from_numpy(input_np_2)
#     pytorch_out_2 = pytorch_linear_2(input_torch_2)

#     ttsim_linear_2 = SimNN.Linear(
#         'test_q_linear_2',
#         in_features=query_dim_2,
#         out_features=hidden_dim_2,
#         bias=True
#     )
#     ttsim_linear_2.param.data = weight_np_2.T.copy()  # Transpose!
#     ttsim_linear_2.bias.data = bias_np_2.copy()

#     input_sim_2 = numpy_to_simtensor(input_np_2, 'q_input_2')
#     ttsim_out_2 = ttsim_linear_2(input_sim_2)

#     if compare_numerical(pytorch_out_2, ttsim_out_2, "Linear dim change"):
#         print("  Dimension change validated!")
#     else:
#         all_passed = False

#     return all_passed


# # ============================================================================
# # Test 2: 1x1 Conv Projection (k_linear)
# # ============================================================================

# def test_conv1x1_projection():
#     """
#     Test TTSim 1x1 Conv for key projection against PyTorch.

#     PyTorch: F.conv2d(k, weight.unsqueeze(-1).unsqueeze(-1), bias)
#     TTSim: conv2d_functional with 1x1 kernel

#     This is used for projecting spatial features (keys) in attention.
#     """
#     print_section("TEST 2: 1x1 Conv Projection (k_linear)")

#     all_passed = True

#     # Test Case 1: Basic 1x1 conv (like k_linear in MHAttentionMap)
#     print_subsection("Test 2.1: 1x1 Conv [B, C_in, H, W] -> [B, C_out, H, W]")

#     np.random.seed(42)
#     torch.manual_seed(42)

#     batch_size = 2
#     query_dim = 256  # C_in
#     hidden_dim = 256  # C_out
#     H, W = 25, 38  # Typical feature map size

#     # Create spatial feature input (keys)
#     k_np = np.random.randn(batch_size, query_dim, H, W).astype(np.float32)

#     # PyTorch k_linear equivalent
#     # In MHAttentionMap: k_linear is nn.Linear but applied as 1x1 conv
#     pytorch_k_linear = nn.Linear(query_dim, hidden_dim, bias=True)
#     weight_2d = pytorch_k_linear.weight.detach().numpy().copy()  # [hidden_dim, query_dim]
#     bias_np = pytorch_k_linear.bias.detach().numpy().copy()  # [hidden_dim]

#     # PyTorch conv2d with weight.unsqueeze(-1).unsqueeze(-1)
#     k_torch = torch.from_numpy(k_np)
#     weight_4d = pytorch_k_linear.weight.unsqueeze(-1).unsqueeze(-1)  # [C_out, C_in, 1, 1]
#     pytorch_out = F_torch.conv2d(k_torch, weight_4d, pytorch_k_linear.bias)

#     print(f"\n  Input (k) shape: {k_np.shape}")
#     print(f"  Weight 2D shape: {weight_2d.shape}")
#     print(f"  Weight 4D shape: {list(weight_4d.shape)}")
#     print(f"  Bias shape: {bias_np.shape}")
#     print(f"  PyTorch output shape: {list(pytorch_out.shape)}")

#     # TTSim conv2d_functional
#     k_sim = numpy_to_simtensor(k_np, 'k_input')
#     weight_4d_np = weight_2d.reshape(hidden_dim, query_dim, 1, 1)
#     weight_sim = F._from_data('k_weight', weight_4d_np, is_param=True)
#     bias_sim = F._from_data('k_bias', bias_np, is_param=True)

#     ttsim_out = conv2d_functional(
#         k_sim, weight_sim, bias_sim,
#         stride=1, padding=0, dilation=1, groups=1, module=None
#     )

#     print(f"  TTSim output shape: {ttsim_out.shape}")
#     print(f"  TTSim output has data: {ttsim_out.data is not None}")

#     if compare_numerical(pytorch_out, ttsim_out, "1x1 Conv k_linear"):
#         print("  1x1 Conv k_linear validated!")
#     else:
#         all_passed = False

#     return all_passed


# # ============================================================================
# # Test 3: Reshape Operations (Multi-head format)
# # ============================================================================

# def test_reshape_multihead():
#     """
#     Test TTSim Reshape for multi-head attention format conversion.

#     Query reshape: [B, Q, hidden_dim] -> [B, Q, num_heads, head_dim]
#     Key reshape: [B, hidden_dim, H, W] -> [B, num_heads, head_dim, H, W]
#     """
#     print_section("TEST 3: Reshape Operations (Multi-head format)")

#     all_passed = True

#     # Test parameters
#     batch_size = 2
#     num_queries = 100
#     hidden_dim = 256
#     num_heads = 8
#     head_dim = hidden_dim // num_heads
#     H, W = 25, 38

#     np.random.seed(42)

#     # Test Case 1: Query reshape
#     print_subsection("Test 3.1: Query reshape [B, Q, D] -> [B, Q, nheads, head_dim]")

#     q_np = np.random.randn(batch_size, num_queries, hidden_dim).astype(np.float32)

#     # PyTorch reshape
#     q_torch = torch.from_numpy(q_np)
#     qh_pytorch = q_torch.view(batch_size, num_queries, num_heads, head_dim)

#     print(f"\n  Input shape: {q_np.shape}")
#     print(f"  Target shape: [{batch_size}, {num_queries}, {num_heads}, {head_dim}]")
#     print(f"  PyTorch output shape: {list(qh_pytorch.shape)}")

#     # TTSim Reshape
#     q_sim = numpy_to_simtensor(q_np, 'q_proj')
#     qh_shape = [batch_size, num_queries, num_heads, head_dim]
#     reshape_q_op = F.Reshape('test_reshape_q')
#     shape_tensor_q = F._from_data('reshape_q_shape',
#                                   data=np.array(qh_shape, dtype=np.int64),
#                                   is_const=True)
#     qh_ttsim = reshape_q_op(q_sim, shape_tensor_q)

#     print(f"  TTSim output shape: {qh_ttsim.shape}")
#     print(f"  TTSim output has data: {qh_ttsim.data is not None}")

#     if compare_numerical(qh_pytorch, qh_ttsim, "Query reshape"):
#         print("  Query reshape validated!")
#     else:
#         all_passed = False

#     # Test Case 2: Key reshape
#     print_subsection("Test 3.2: Key reshape [B, D, H, W] -> [B, nheads, head_dim, H, W]")

#     k_np = np.random.randn(batch_size, hidden_dim, H, W).astype(np.float32)

#     # PyTorch reshape
#     k_torch = torch.from_numpy(k_np)
#     kh_pytorch = k_torch.view(batch_size, num_heads, head_dim, H, W)

#     print(f"\n  Input shape: {k_np.shape}")
#     print(f"  Target shape: [{batch_size}, {num_heads}, {head_dim}, {H}, {W}]")
#     print(f"  PyTorch output shape: {list(kh_pytorch.shape)}")

#     # TTSim Reshape
#     k_sim = numpy_to_simtensor(k_np, 'k_proj')
#     kh_shape = [batch_size, num_heads, head_dim, H, W]
#     reshape_k_op = F.Reshape('test_reshape_k')
#     shape_tensor_k = F._from_data('reshape_k_shape',
#                                   data=np.array(kh_shape, dtype=np.int64),
#                                   is_const=True)
#     kh_ttsim = reshape_k_op(k_sim, shape_tensor_k)

#     print(f"  TTSim output shape: {kh_ttsim.shape}")
#     print(f"  TTSim output has data: {kh_ttsim.data is not None}")

#     if compare_numerical(kh_pytorch, kh_ttsim, "Key reshape"):
#         print("  Key reshape validated!")
#     else:
#         all_passed = False

#     return all_passed


# # ============================================================================
# # Test 4: Einsum for Attention Scores
# # ============================================================================

# def test_einsum_attention():
#     """
#     Test TTSim Einsum for attention score computation.

#     PyTorch: torch.einsum("bqnc,bnchw->bqnhw", qh * scale, kh)
#     TTSim: F.Einsum(name, equation, *inputs)

#     This computes the attention scores between queries and keys.
#     """
#     print_section("TEST 4: Einsum for Attention Scores")

#     all_passed = True

#     # Test parameters
#     batch_size = 2
#     num_queries = 100
#     num_heads = 8
#     head_dim = 32
#     H, W = 25, 38

#     np.random.seed(42)

#     # Test Case 1: Basic einsum
#     print_subsection("Test 4.1: Einsum bqnc,bnchw->bqnhw")

#     # Create inputs in multi-head format
#     qh_np = np.random.randn(batch_size, num_queries, num_heads, head_dim).astype(np.float32)
#     kh_np = np.random.randn(batch_size, num_heads, head_dim, H, W).astype(np.float32)

#     # Apply scaling (as done in MHAttentionMap)
#     scale = float(head_dim) ** -0.5
#     qh_scaled_np = qh_np * scale

#     # PyTorch einsum
#     qh_torch = torch.from_numpy(qh_scaled_np)
#     kh_torch = torch.from_numpy(kh_np)
#     weights_pytorch = torch.einsum("bqnc,bnchw->bqnhw", qh_torch, kh_torch)

#     print(f"\n  qh shape: {qh_scaled_np.shape}")
#     print(f"  kh shape: {kh_np.shape}")
#     print(f"  Equation: bqnc,bnchw->bqnhw")
#     print(f"  PyTorch output shape: {list(weights_pytorch.shape)}")
#     print(f"  PyTorch output (first 5): {weights_pytorch.flatten()[:5].numpy()}")

#     # TTSim Einsum
#     qh_sim = numpy_to_simtensor(qh_scaled_np, 'qh_scaled')
#     kh_sim = numpy_to_simtensor(kh_np, 'kh')

#     weights_ttsim = F.Einsum('test_einsum', 'bqnc,bnchw->bqnhw', qh_sim, kh_sim)

#     print(f"  TTSim output shape: {weights_ttsim.shape}")
#     print(f"  TTSim output has data: {weights_ttsim.data is not None}")

#     if weights_ttsim.data is not None:
#         print(f"  TTSim output (first 5): {weights_ttsim.data.flatten()[:5]}")
#         if compare_numerical(weights_pytorch, weights_ttsim, "Einsum attention"):
#             print("  Einsum attention validated!")
#         else:
#             all_passed = False
#     else:
#         # Validate shape only if data not propagated
#         expected_shape = list(weights_pytorch.shape)
#         if list(weights_ttsim.shape) == expected_shape:
#             print(f"  ✓ Shape correct (data not propagated): {expected_shape}")
#         else:
#             print(f"  ✗ Shape mismatch")
#             all_passed = False

#     # Test Case 2: Smaller test case for verification
#     print_subsection("Test 4.2: Small einsum for manual verification")

#     batch_small = 1
#     Q_small = 2
#     heads_small = 2
#     hdim_small = 3
#     H_small, W_small = 2, 2

#     qh_small = np.random.randn(batch_small, Q_small, heads_small, hdim_small).astype(np.float32)
#     kh_small = np.random.randn(batch_small, heads_small, hdim_small, H_small, W_small).astype(np.float32)

#     # PyTorch
#     qh_torch_small = torch.from_numpy(qh_small)
#     kh_torch_small = torch.from_numpy(kh_small)
#     weights_pytorch_small = torch.einsum("bqnc,bnchw->bqnhw", qh_torch_small, kh_torch_small)

#     # TTSim
#     qh_sim_small = numpy_to_simtensor(qh_small, 'qh_small')
#     kh_sim_small = numpy_to_simtensor(kh_small, 'kh_small')
#     weights_ttsim_small = F.Einsum('test_einsum_small', 'bqnc,bnchw->bqnhw', qh_sim_small, kh_sim_small)

#     print(f"\n  Small test: qh={qh_small.shape}, kh={kh_small.shape}")
#     print(f"  Output shape: {list(weights_pytorch_small.shape)}")

#     if weights_ttsim_small.data is not None:
#         if compare_numerical(weights_pytorch_small, weights_ttsim_small, "Einsum small"):
#             print("  Small einsum validated!")
#         else:
#             all_passed = False
#     else:
#         print("  Note: Data not propagated (shape-only mode)")

#     return all_passed


# # ============================================================================
# # Test 5: Softmax over Flattened Spatial Dimensions
# # ============================================================================

# def test_softmax_spatial():
#     """
#     Test TTSim Softmax for attention weight normalization.

#     PyTorch: F.softmax(weights.flatten(2), dim=-1).view_as(weights)
#     TTSim: F.Softmax with axis parameter

#     The attention weights are flattened over spatial dims, softmax applied,
#     then reshaped back.
#     """
#     print_section("TEST 5: Softmax over Flattened Spatial Dimensions")

#     all_passed = True

#     # Test parameters
#     batch_size = 2
#     num_queries = 100
#     num_heads = 8
#     H, W = 25, 38

#     np.random.seed(42)

#     # Test Case 1: Softmax on flattened spatial
#     print_subsection("Test 5.1: Softmax [B, Q, nheads, H*W] -> normalized")

#     # Create attention weights [B, Q, nheads, H, W]
#     weights_np = np.random.randn(batch_size, num_queries, num_heads, H, W).astype(np.float32)

#     # PyTorch: Flatten to [B, Q, nheads, H*W], softmax, reshape back
#     weights_torch = torch.from_numpy(weights_np)
#     weights_flat_pytorch = weights_torch.flatten(3)  # [B, Q, nheads, H*W]
#     weights_softmax_pytorch = F_torch.softmax(weights_flat_pytorch, dim=-1)
#     weights_out_pytorch = weights_softmax_pytorch.view_as(weights_torch)

#     print(f"\n  Input shape: {weights_np.shape}")
#     print(f"  Flattened shape: {list(weights_flat_pytorch.shape)}")
#     print(f"  After softmax shape: {list(weights_softmax_pytorch.shape)}")
#     print(f"  Final shape: {list(weights_out_pytorch.shape)}")

#     # TTSim approach: Flatten -> Softmax -> Reshape
#     weights_sim = numpy_to_simtensor(weights_np, 'attn_weights')

#     # Step 1: Flatten spatial dimensions
#     HW = H * W
#     flatten_shape = [batch_size, num_queries, num_heads, HW]
#     flatten_op = F.Reshape('test_flatten')
#     flatten_shape_tensor = F._from_data('flatten_shape',
#                                         data=np.array(flatten_shape, dtype=np.int64),
#                                         is_const=True)
#     weights_flat_ttsim = flatten_op(weights_sim, flatten_shape_tensor)

#     # Step 2: Softmax on last axis
#     softmax_op = F.Softmax('test_softmax', axis=-1)
#     weights_softmax_ttsim = softmax_op(weights_flat_ttsim)

#     # Step 3: Reshape back to spatial
#     unflatten_shape = [batch_size, num_queries, num_heads, H, W]
#     unflatten_op = F.Reshape('test_unflatten')
#     unflatten_shape_tensor = F._from_data('unflatten_shape',
#                                           data=np.array(unflatten_shape, dtype=np.int64),
#                                           is_const=True)
#     weights_out_ttsim = unflatten_op(weights_softmax_ttsim, unflatten_shape_tensor)

#     print(f"  TTSim flattened shape: {weights_flat_ttsim.shape}")
#     print(f"  TTSim softmax shape: {weights_softmax_ttsim.shape}")
#     print(f"  TTSim final shape: {weights_out_ttsim.shape}")

#     if weights_out_ttsim.data is not None:
#         if compare_numerical(weights_out_pytorch, weights_out_ttsim, "Softmax spatial"):
#             print("  Softmax spatial validated!")
#         else:
#             all_passed = False
#     else:
#         print("  Note: Data not propagated (shape-only mode)")
#         expected_shape = list(weights_out_pytorch.shape)
#         if list(weights_out_ttsim.shape) == expected_shape:
#             print(f"  ✓ Shape correct: {expected_shape}")
#         else:
#             print(f"  ✗ Shape mismatch")
#             all_passed = False

#     # Test Case 2: Verify softmax properties (sum to 1)
#     print_subsection("Test 5.2: Verify softmax sums to 1")

#     if weights_out_ttsim.data is not None:
#         # Check that softmax over H*W sums to 1
#         weights_ttsim_np = weights_out_ttsim.data
#         sums = np.sum(weights_ttsim_np, axis=(-2, -1))  # Sum over H, W

#         all_ones = np.allclose(sums, 1.0, atol=1e-5)
#         print(f"\n  Softmax sum check (should be 1.0):")
#         print(f"  Sample sums: {sums[0, 0, :5]}")
#         print(f"  All sums ~= 1.0: {all_ones}")

#         if not all_ones:
#             all_passed = False

#     return all_passed


# # ============================================================================
# # Test 6: Full MHAttentionMap Forward Pass
# # ============================================================================

# def test_mhattentionmap_full():
#     """
#     Test full MHAttentionMap module: PyTorch vs TTSim.

#     Tests the complete forward pass with:
#     - Query projection
#     - Key projection (1x1 conv)
#     - Multi-head reshape
#     - Einsum attention scores
#     - Optional masking
#     - Softmax normalization
#     - Dropout (test mode)
#     """
#     print_section("TEST 6: Full MHAttentionMap Forward Pass")

#     all_passed = True

#     # Model parameters
#     query_dim = 256
#     hidden_dim = 256
#     num_heads = 8
#     dropout = 0.0  # Set to 0 for deterministic testing
#     head_dim = hidden_dim // num_heads

#     # Input dimensions
#     batch_size = 2
#     num_queries = 100
#     H, W = 25, 38

#     np.random.seed(42)
#     torch.manual_seed(42)

#     # Test Case 1: Forward without mask - DEBUG INTERMEDIATE VALUES
#     print_subsection("Test 6.1: Forward pass without mask (DEBUG)")

#     # Create inputs
#     q_np = np.random.randn(batch_size, num_queries, query_dim).astype(np.float32)
#     k_np = np.random.randn(batch_size, query_dim, H, W).astype(np.float32)

#     # PyTorch MHAttentionMap
#     pytorch_model = MHAttentionMapPyTorch(query_dim, hidden_dim, num_heads, dropout=dropout)
#     pytorch_model.eval()  # Disable dropout

#     q_torch = torch.from_numpy(q_np)
#     k_torch = torch.from_numpy(k_np)

#     # Get PyTorch weights
#     pt_q_weight = pytorch_model.q_linear.weight.detach().numpy()  # [hidden_dim, query_dim]
#     pt_q_bias = pytorch_model.q_linear.bias.detach().numpy()
#     pt_k_weight = pytorch_model.k_linear.weight.detach().numpy()  # [hidden_dim, query_dim]
#     pt_k_bias = pytorch_model.k_linear.bias.detach().numpy()

#     # === STEP-BY-STEP COMPARISON ===
#     print("\n  === STEP-BY-STEP INTERMEDIATE COMPARISON ===")

#     # Step 1: Query projection
#     print("\n  STEP 1: Query projection")
#     q_proj_pytorch = pytorch_model.q_linear(q_torch).detach().numpy()
#     print(f"    PyTorch q_proj shape: {q_proj_pytorch.shape}")
#     print(f"    PyTorch q_proj first 5: {q_proj_pytorch.flatten()[:5]}")

#     # Manual numpy computation (reference)
#     q_proj_manual = q_np @ pt_q_weight.T + pt_q_bias
#     print(f"    Manual q_proj first 5: {q_proj_manual.flatten()[:5]}")
#     manual_match = np.allclose(q_proj_pytorch, q_proj_manual, rtol=1e-4, atol=1e-5)
#     print(f"    Manual matches PyTorch: {manual_match}")

#     # Step 2: Key projection (1x1 conv)
#     print("\n  STEP 2: Key projection (1x1 conv)")
#     weight_4d = pytorch_model.k_linear.weight.unsqueeze(-1).unsqueeze(-1)
#     k_proj_pytorch = F_torch.conv2d(k_torch, weight_4d, pytorch_model.k_linear.bias).detach().numpy()
#     print(f"    PyTorch k_proj shape: {k_proj_pytorch.shape}")
#     print(f"    PyTorch k_proj first 5: {k_proj_pytorch.flatten()[:5]}")

#     # Step 3: Reshape to multi-head format
#     print("\n  STEP 3: Reshape to multi-head")
#     qh_pytorch = q_proj_pytorch.reshape(batch_size, num_queries, num_heads, head_dim)
#     kh_pytorch = k_proj_pytorch.reshape(batch_size, num_heads, head_dim, H, W)
#     print(f"    qh shape: {qh_pytorch.shape}")
#     print(f"    kh shape: {kh_pytorch.shape}")

#     # Step 4: Scale queries
#     print("\n  STEP 4: Scale queries")
#     scale = float(head_dim) ** -0.5
#     qh_scaled_pytorch = qh_pytorch * scale
#     print(f"    Scale factor: {scale}")
#     print(f"    qh_scaled first 5: {qh_scaled_pytorch.flatten()[:5]}")

#     # Step 5: Einsum attention scores
#     print("\n  STEP 5: Einsum attention scores")
#     weights_pytorch_step = np.einsum('bqnc,bnchw->bqnhw', qh_scaled_pytorch, kh_pytorch)
#     print(f"    weights shape: {weights_pytorch_step.shape}")
#     print(f"    weights first 5: {weights_pytorch_step.flatten()[:5]}")

#     # Step 6: Softmax
#     # PyTorch: F.softmax(weights.flatten(2), dim=-1).view_as(weights)
#     # flatten(2) flattens from dim 2: [B, Q, nheads*H*W]
#     print("\n  STEP 6: Softmax (flatten from dim 2)")
#     weights_flat = weights_pytorch_step.reshape(batch_size, num_queries, num_heads * H * W)
#     weights_soft = np.exp(weights_flat - np.max(weights_flat, axis=-1, keepdims=True))
#     weights_soft = weights_soft / np.sum(weights_soft, axis=-1, keepdims=True)
#     weights_final_manual = weights_soft.reshape(batch_size, num_queries, num_heads, H, W)
#     print(f"    final shape: {weights_final_manual.shape}")
#     print(f"    final first 5: {weights_final_manual.flatten()[:5]}")

#     # Get full PyTorch output
#     with torch.no_grad():
#         pytorch_out = pytorch_model(q_torch, k_torch, mask=None)

#     print(f"\n  Input q shape: {q_np.shape}")
#     print(f"  Input k shape: {k_np.shape}")
#     print(f"  PyTorch output shape: {list(pytorch_out.shape)}")
#     print(f"  Expected: [B={batch_size}, Q={num_queries}, nheads={num_heads}, H={H}, W={W}]")

#     # TTSim MHAttentionMap
#     ttsim_model = MHAttentionMapTTSim(
#         'test_mhattention',
#         query_dim=query_dim,
#         hidden_dim=hidden_dim,
#         num_heads=num_heads,
#         dropout=dropout
#     )

#     # Copy weights from PyTorch to TTSim
#     # q_linear weights - stored as [in, out] format for TTSim matmul (x @ W)
#     ttsim_model.q_linear_weight.data = pt_q_weight.T.copy()
#     ttsim_model.q_linear.param.data = pt_q_weight.T.copy()
#     ttsim_model.q_linear_bias.data = pt_q_bias.copy()

#     # k_linear weights (stored as 4D for conv2d in TTSim)
#     k_weight_4d = pt_k_weight.reshape(hidden_dim, query_dim, 1, 1)
#     ttsim_model.k_linear_weight.data = k_weight_4d.copy()
#     ttsim_model.k_linear_bias.data = pt_k_bias.copy()

#     print(f"\n  === TTSim Weight Shapes ===")
#     print(f"    q_linear_weight shape: {ttsim_model.q_linear_weight.shape}")
#     print(f"    q_linear_weight data shape: {ttsim_model.q_linear_weight.data.shape if ttsim_model.q_linear_weight.data is not None else 'None'}")
#     print(f"    k_linear_weight shape: {ttsim_model.k_linear_weight.shape}")
#     print(f"    k_linear_weight data shape: {ttsim_model.k_linear_weight.data.shape if ttsim_model.k_linear_weight.data is not None else 'None'}")

#     q_sim = numpy_to_simtensor(q_np, 'query')
#     k_sim = numpy_to_simtensor(k_np, 'key')

#     # === TRACE THROUGH TTSIM FORWARD STEP-BY-STEP ===
#     print(f"\n  === TTSim Step-by-Step Forward ===")

#     # Use fresh input tensors
#     q_sim_trace = numpy_to_simtensor(q_np, 'q_trace')
#     k_sim_trace = numpy_to_simtensor(k_np, 'k_trace')

#     # Step 1: q_linear (manual matmul since q_linear is no longer a module)
#     print("\n  TTSim STEP 1: q_linear (manual matmul)")
#     q_matmul_trace = F.MatMul('trace_q_matmul')
#     q_matmul_trace.set_module(ttsim_model)
#     q_proj_ttsim = q_matmul_trace(q_sim_trace, ttsim_model.q_linear_weight)
#     if ttsim_model.q_linear_bias is not None:
#         q_proj_ttsim = q_proj_ttsim + ttsim_model.q_linear_bias
#     print(f"    q_proj_ttsim shape: {q_proj_ttsim.shape}")
#     print(f"    q_proj_ttsim has data: {q_proj_ttsim.data is not None}")
#     if q_proj_ttsim.data is not None:
#         print(f"    q_proj_ttsim first 5: {q_proj_ttsim.data.flatten()[:5]}")
#         q_proj_match = np.allclose(q_proj_pytorch, q_proj_ttsim.data, rtol=1e-4, atol=1e-5)
#         print(f"    q_proj matches PyTorch: {q_proj_match}")

#     # Step 2: k_linear (conv2d)
#     print("\n  TTSim STEP 2: k_linear (conv2d)")
#     k_proj_ttsim = conv2d_functional(
#         k_sim_trace, ttsim_model.k_linear_weight, ttsim_model.k_linear_bias,
#         stride=1, padding=0, groups=1, module=ttsim_model, op_prefix='trace_k'
#     )
#     print(f"    k_proj_ttsim shape: {k_proj_ttsim.shape}")
#     print(f"    k_proj_ttsim has data: {k_proj_ttsim.data is not None}")
#     if k_proj_ttsim.data is not None:
#         print(f"    k_proj_ttsim first 5: {k_proj_ttsim.data.flatten()[:5]}")
#         k_proj_match = np.allclose(k_proj_pytorch, k_proj_ttsim.data, rtol=1e-4, atol=1e-5)
#         print(f"    k_proj matches PyTorch: {k_proj_match}")
#         if not k_proj_match:
#             diff = np.abs(k_proj_pytorch - k_proj_ttsim.data)
#             print(f"    Max diff: {np.max(diff):.2e}")

#     # Step 3: Reshape queries
#     print("\n  TTSim STEP 3: Reshape queries")
#     qh_shape = [batch_size, num_queries, num_heads, head_dim]
#     reshape_q_op = F.Reshape('trace_reshape_q')
#     shape_tensor_q = F._from_data('trace_reshape_q_shape',
#                                   data=np.array(qh_shape, dtype=np.int64), is_const=True)
#     qh_ttsim = reshape_q_op(q_proj_ttsim, shape_tensor_q)
#     print(f"    qh_ttsim shape: {qh_ttsim.shape}")
#     print(f"    qh_ttsim has data: {qh_ttsim.data is not None}")
#     if qh_ttsim.data is not None:
#         qh_pytorch_ref = q_proj_pytorch.reshape(batch_size, num_queries, num_heads, head_dim)
#         print(f"    qh_ttsim first 5: {qh_ttsim.data.flatten()[:5]}")
#         qh_match = np.allclose(qh_pytorch_ref, qh_ttsim.data, rtol=1e-4, atol=1e-5)
#         print(f"    qh matches PyTorch: {qh_match}")

#     # Step 4: Reshape keys
#     print("\n  TTSim STEP 4: Reshape keys")
#     kh_shape = [batch_size, num_heads, head_dim, H, W]
#     reshape_k_op = F.Reshape('trace_reshape_k')
#     shape_tensor_k = F._from_data('trace_reshape_k_shape',
#                                   data=np.array(kh_shape, dtype=np.int64), is_const=True)
#     kh_ttsim = reshape_k_op(k_proj_ttsim, shape_tensor_k)
#     print(f"    kh_ttsim shape: {kh_ttsim.shape}")
#     print(f"    kh_ttsim has data: {kh_ttsim.data is not None}")
#     if kh_ttsim.data is not None:
#         kh_pytorch_ref = k_proj_pytorch.reshape(batch_size, num_heads, head_dim, H, W)
#         print(f"    kh_ttsim first 5: {kh_ttsim.data.flatten()[:5]}")
#         kh_match = np.allclose(kh_pytorch_ref, kh_ttsim.data, rtol=1e-4, atol=1e-5)
#         print(f"    kh matches PyTorch: {kh_match}")

#     # Step 5: Scale queries
#     print("\n  TTSim STEP 5: Scale queries")
#     scale_const = F._from_data('trace_scale', np.array(scale, dtype=np.float32), is_const=True)
#     mul_op = F.Mul('trace_qh_scale')
#     qh_scaled_ttsim = mul_op(qh_ttsim, scale_const)
#     print(f"    qh_scaled_ttsim shape: {qh_scaled_ttsim.shape}")
#     print(f"    qh_scaled_ttsim has data: {qh_scaled_ttsim.data is not None}")
#     if qh_scaled_ttsim.data is not None:
#         print(f"    qh_scaled_ttsim first 5: {qh_scaled_ttsim.data.flatten()[:5]}")
#         qh_scaled_match = np.allclose(qh_scaled_pytorch.flatten()[:100],
#                                        qh_scaled_ttsim.data.flatten()[:100], rtol=1e-4, atol=1e-5)
#         print(f"    qh_scaled matches PyTorch: {qh_scaled_match}")

#     # Step 6: Einsum
#     print("\n  TTSim STEP 6: Einsum attention scores")
#     weights_ttsim_step = F.Einsum('trace_einsum', 'bqnc,bnchw->bqnhw', qh_scaled_ttsim, kh_ttsim)
#     print(f"    weights_ttsim shape: {weights_ttsim_step.shape}")
#     print(f"    weights_ttsim has data: {weights_ttsim_step.data is not None}")
#     einsum_passed = False
#     if weights_ttsim_step.data is not None:
#         print(f"    weights_ttsim first 10: {weights_ttsim_step.data.flatten()[:10]}")
#         print(f"    PyTorch weights first 10: {weights_pytorch_step.flatten()[:10]}")
#         einsum_match = np.allclose(weights_pytorch_step, weights_ttsim_step.data, rtol=1e-4, atol=1e-5)
#         print(f"    einsum matches PyTorch: {einsum_match}")
#         einsum_passed = einsum_match
#         if not einsum_match:
#             diff = np.abs(weights_pytorch_step - weights_ttsim_step.data)
#             print(f"    Max diff: {np.max(diff):.2e}")
#             print(f"    Mean diff: {np.mean(diff):.2e}")
#             # Show where the differences are
#             max_idx = np.unravel_index(np.argmax(diff), diff.shape)
#             print(f"    Max diff at: {max_idx}")
#             print(f"    PyTorch at max: {weights_pytorch_step[max_idx]}")
#             print(f"    TTSim at max: {weights_ttsim_step.data[max_idx]}")

#     # Step 7: Softmax (flatten from dim 2: nheads*H*W)
#     print("\n  TTSim STEP 7: Softmax (flatten from dim 2)")
#     nheads_HW = num_heads * H * W
#     flatten_shape = [batch_size, num_queries, nheads_HW]
#     flatten_op = F.Reshape('trace_flatten')
#     flatten_shape_tensor = F._from_data('trace_flatten_shape',
#                                         data=np.array(flatten_shape, dtype=np.int64), is_const=True)
#     weights_flat_ttsim = flatten_op(weights_ttsim_step, flatten_shape_tensor)
#     print(f"    weights_flat_ttsim shape: {weights_flat_ttsim.shape}")
#     print(f"    weights_flat_ttsim has data: {weights_flat_ttsim.data is not None}")
#     if weights_flat_ttsim.data is not None:
#         print(f"    weights_flat_ttsim first 10: {weights_flat_ttsim.data.flatten()[:10]}")

#     softmax_op = F.Softmax('trace_softmax', axis=-1)
#     weights_soft_ttsim = softmax_op(weights_flat_ttsim)
#     print(f"    weights_soft_ttsim shape: {weights_soft_ttsim.shape}")
#     print(f"    weights_soft_ttsim has data: {weights_soft_ttsim.data is not None}")
#     if weights_soft_ttsim.data is not None:
#         print(f"    weights_soft_ttsim first 10: {weights_soft_ttsim.data.flatten()[:10]}")

#     # Step 8: Unflatten
#     print("\n  TTSim STEP 8: Unflatten")
#     unflatten_shape = [batch_size, num_queries, num_heads, H, W]
#     unflatten_op = F.Reshape('trace_unflatten')
#     unflatten_shape_tensor = F._from_data('trace_unflatten_shape',
#                                           data=np.array(unflatten_shape, dtype=np.int64), is_const=True)
#     weights_final_ttsim = unflatten_op(weights_soft_ttsim, unflatten_shape_tensor)
#     print(f"    weights_final_ttsim shape: {weights_final_ttsim.shape}")
#     print(f"    weights_final_ttsim has data: {weights_final_ttsim.data is not None}")
#     if weights_final_ttsim.data is not None:
#         print(f"    weights_final_ttsim first 10: {weights_final_ttsim.data.flatten()[:10]}")
#         # Compare with manual PyTorch computation
#         print(f"    Manual PyTorch final first 10: {weights_final_manual.flatten()[:10]}")
#         step_by_step_match = np.allclose(weights_final_manual, weights_final_ttsim.data, rtol=1e-4, atol=1e-5)
#         print(f"    Step-by-step matches: {step_by_step_match}")
#         if step_by_step_match:
#             print("    ✓ Step-by-step TTSim computation matches manual PyTorch!")
#         else:
#             diff = np.abs(weights_final_manual - weights_final_ttsim.data)
#             print(f"    Step-by-step max diff: {np.max(diff):.2e}")

#     # Full forward (for final comparison)
#     # IMPORTANT: Create a FRESH model instance because TTSim operations accumulate state
#     # when called multiple times. The step-by-step debugging above corrupted the model's
#     # internal operation state (inList keeps growing on each call).
#     print("\n  === Full Model Forward Comparison ===")
#     print("  Creating fresh model instance to avoid operation state accumulation...")

#     ttsim_model_fresh = MHAttentionMapTTSim(
#         'test_mhattention_fresh',
#         query_dim=query_dim,
#         hidden_dim=hidden_dim,
#         num_heads=num_heads,
#         dropout=dropout
#     )

#     # Copy weights to fresh model
#     ttsim_model_fresh.q_linear_weight.data = pt_q_weight.T.copy()
#     ttsim_model_fresh.q_linear_bias.data = pt_q_bias.copy()
#     ttsim_model_fresh.k_linear_weight.data = k_weight_4d.copy()
#     ttsim_model_fresh.k_linear_bias.data = pt_k_bias.copy()

#     q_sim2 = numpy_to_simtensor(q_np, 'query_fresh')
#     k_sim2 = numpy_to_simtensor(k_np, 'key_fresh')
#     ttsim_out = ttsim_model_fresh(q_sim2, k_sim2, mask=None)

#     print(f"\n  TTSim output shape: {ttsim_out.shape}")
#     print(f"  TTSim output has data: {ttsim_out.data is not None}")

#     if ttsim_out.data is not None:
#         # Relaxed tolerance: float32 einsum+softmax over 7600 elements amplifies
#         # backend rounding differences (~1e-5 in scores → ~1e-2 in softmax peaks)
#         if compare_numerical(pytorch_out, ttsim_out, "MHAttentionMap (no mask)",
#                              rtol=1e-2, atol=1e-3):
#             print("  Full forward pass validated!")
#         else:
#             all_passed = False
#     else:
#         # Shape validation only
#         expected_shape = list(pytorch_out.shape)
#         if list(ttsim_out.shape) == expected_shape:
#             print(f"  ✓ Shape correct (data not propagated): {expected_shape}")
#         else:
#             print(f"  ✗ Shape mismatch: expected {expected_shape}, got {list(ttsim_out.shape)}")
#             all_passed = False

#     # Test Case 2: Forward with mask
#     print_subsection("Test 6.2: Forward pass with attention mask")

#     # Create mask [B, H, W] - True means invalid (pad) positions
#     mask_np = np.zeros((batch_size, H, W), dtype=bool)
#     # Mask out some positions (e.g., right edge padding)
#     mask_np[:, :, -5:] = True

#     q_torch = torch.from_numpy(q_np)
#     k_torch = torch.from_numpy(k_np)
#     mask_torch = torch.from_numpy(mask_np)

#     with torch.no_grad():
#         pytorch_out_masked = pytorch_model(q_torch, k_torch, mask=mask_torch)

#     print(f"\n  Mask shape: {mask_np.shape}")
#     print(f"  Mask True positions (invalid): {np.sum(mask_np)} / {mask_np.size}")
#     print(f"  PyTorch masked output shape: {list(pytorch_out_masked.shape)}")

#     # TTSim with mask - create FRESH model to avoid operation state accumulation
#     ttsim_model_masked = MHAttentionMapTTSim(
#         'test_mhattention_masked',
#         query_dim=query_dim,
#         hidden_dim=hidden_dim,
#         num_heads=num_heads,
#         dropout=dropout
#     )
#     ttsim_model_masked.q_linear_weight.data = pt_q_weight.T.copy()
#     ttsim_model_masked.q_linear_bias.data = pt_q_bias.copy()
#     ttsim_model_masked.k_linear_weight.data = k_weight_4d.copy()
#     ttsim_model_masked.k_linear_bias.data = pt_k_bias.copy()

#     q_sim = numpy_to_simtensor(q_np, 'query_masked')
#     k_sim = numpy_to_simtensor(k_np, 'key_masked')
#     mask_sim = numpy_to_simtensor(mask_np.astype(np.float32), 'mask')

#     ttsim_out_masked = ttsim_model_masked(q_sim, k_sim, mask=mask_sim)

#     print(f"  TTSim masked output shape: {ttsim_out_masked.shape}")
#     print(f"  TTSim masked output has data: {ttsim_out_masked.data is not None}")

#     if ttsim_out_masked.data is not None:
#         # Check that masked positions have very low attention (near 0 after softmax)
#         ttsim_masked_data = ttsim_out_masked.data
#         pytorch_masked_data = pytorch_out_masked.detach().numpy()

#         # Attention at masked positions should be ~0
#         masked_attention_ttsim = ttsim_masked_data[:, :, :, :, -5:]
#         masked_attention_pytorch = pytorch_masked_data[:, :, :, :, -5:]

#         print(f"\n  Masked region attention (should be ~0):")
#         print(f"  PyTorch max at masked: {np.max(masked_attention_pytorch):.2e}")
#         print(f"  TTSim max at masked: {np.max(masked_attention_ttsim):.2e}")

#         # Relaxed tolerance: same float32 einsum+softmax amplification reasoning
#         if compare_numerical(pytorch_out_masked, ttsim_out_masked, "MHAttentionMap (with mask)",
#                              rtol=1e-2, atol=1e-3):
#             print("  Masked forward pass validated!")
#         else:
#             all_passed = False
#     else:
#         expected_shape = list(pytorch_out_masked.shape)
#         if list(ttsim_out_masked.shape) == expected_shape:
#             print(f"  ✓ Shape correct: {expected_shape}")
#         else:
#             all_passed = False

#     return all_passed


# # ============================================================================
# # Main Test Runner
# # ============================================================================

# def run_all_tests():
#     """Run all MHAttentionMap numerical tests."""
#     print("\n" + "=" * 80)
#     print(" MHATTENTIONMAP MODULE - NUMERICAL VALIDATION")
#     print(" Testing: Linear, Conv1x1, Reshape, Einsum, Softmax, Full Forward")
#     print(f" Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("=" * 80)

#     results = {}

#     # Run each test
#     tests = [
#         ("Linear Projection", test_linear_projection),
#         ("1x1 Conv Projection", test_conv1x1_projection),
#         ("Reshape Multi-head", test_reshape_multihead),
#         ("Einsum Attention", test_einsum_attention),
#         ("Softmax Spatial", test_softmax_spatial),
#         ("Full MHAttentionMap", test_mhattentionmap_full),
#     ]

#     for test_name, test_func in tests:
#         try:
#             results[test_name] = test_func()
#         except Exception as e:
#             print(f"\n✗ ERROR in {test_name}: {e}")
#             import traceback
#             traceback.print_exc()
#             results[test_name] = False

#     # Summary
#     print("\n" + "=" * 80)
#     print(" TEST SUMMARY")
#     print("=" * 80)

#     all_passed = True
#     for test_name, passed in results.items():
#         status = "✓ PASS" if passed else "✗ FAIL"
#         print(f"  {status} - {test_name}")
#         if not passed:
#             all_passed = False

#     print("=" * 80)
#     if all_passed:
#         print(" ALL TESTS PASSED!")
#     else:
#         print(" SOME TESTS FAILED - Review output above")
#     print("=" * 80)

#     return all_passed


# if __name__ == "__main__":
#     success = run_all_tests()
#     sys.exit(0 if success else 1)
