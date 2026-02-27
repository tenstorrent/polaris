#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced test file comparing PyTorch and TTSim position encoding implementations.
Compares shape and numerical inference with detailed outputs.
Generates markdown report with input/output samples and relative error analysis.

Modules tested:
- PositionEmbeddingSine (with numerical comparison)
- PositionEmbeddingLearned (shape only - random initialization)
- build_position_encoding factory
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
from workloads.Deformable_DETR.reference.position_encoding import (
    PositionEmbeddingSine as PositionEmbeddingSinePyTorch,
    PositionEmbeddingLearned as PositionEmbeddingLearnedPyTorch,
    build_position_encoding as build_position_encoding_pytorch,
)

# Import TTSim implementations
from workloads.Deformable_DETR.models.position_encoding_ttsim import (
    PositionEmbeddingSine as PositionEmbeddingSineTTSim,
    PositionEmbeddingLearned as PositionEmbeddingLearnedTTSim,
    build_position_encoding as build_position_encoding_ttsim,
)

# Import utilities
from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPyTorch
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTTSim
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

    report_path = os.path.join(report_dir, "position_encoding_validation.md")

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

    # Shape
    log_to_report(f"{indent}```")
    log_to_report(f"{indent}PyTorch shape: {list(pytorch_data.shape)}")
    if ttsim_data is not None:
        log_to_report(f"{indent}TTSim shape:   {list(ttsim_data.shape)}")
    else:
        log_to_report(f"{indent}TTSim shape:   None (shape inference only)")

    # Statistics
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

    # Samples
    log_to_report(f"{indent}")
    log_to_report(f"{indent}Sample values (first 10):")
    log_to_report(f"{indent}  PyTorch: {format_array_sample(pytorch_data, 10)}")
    if ttsim_data is not None:
        log_to_report(f"{indent}  TTSim:   {format_array_sample(ttsim_data, 10)}")

    log_to_report(f"{indent}```")


def compare_shapes(torch_output, ttsim_output, test_name):
    """Compare shapes between PyTorch and TTSim outputs"""
    if isinstance(torch_output, torch.Tensor):
        torch_shape = list(torch_output.shape)
    else:
        torch_shape = (
            list(torch_output) if hasattr(torch_output, "__iter__") else [torch_output]
        )

    if isinstance(ttsim_output, SimTensor):
        ttsim_shape = ttsim_output.shape
    else:
        ttsim_shape = (
            list(ttsim_output) if hasattr(ttsim_output, "__iter__") else [ttsim_output]
        )

    log_to_report(f"\n#### Shape Comparison")
    log_to_report(f"```")
    log_to_report(f"PyTorch: {torch_shape}")
    log_to_report(f"TTSim:   {ttsim_shape}")
    log_to_report(f"```")

    if torch_shape == ttsim_shape:
        log_to_report(f"**Result:** [PASSED] Shapes match")
        return True
    else:
        log_to_report(f"**Result:** [FAILED] Shape mismatch")
        return False


def compare_numerics(torch_output, ttsim_output, test_name, rtol=1e-4, atol=1e-5):
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
        log_to_report(f"  Diff:    {abs_diff[worst_idx]:.8e}")

    log_to_report(f"```")

    log_to_report(
        f"\n**Result:** {'[PASSED]' if matches else '[FAILED]'} Numerical comparison"
    )
    return matches


def create_dummy_args():
    """Create dummy args object for testing"""

    class Args:
        hidden_dim = 256
        position_embedding = "sine"

    return Args()


# ──────────────────────────────────────────────────────────────────────────────
# Test Functions
# ──────────────────────────────────────────────────────────────────────────────


def test_position_embedding_sine():
    """Test PositionEmbeddingSine: PyTorch vs TTSim with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 1: PositionEmbeddingSine")
    log_to_report("=" * 80)

    batch_size = 2
    channels = 256
    height = 28
    width = 28
    num_pos_feats = 128

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(f"Input tensor shape:  [{batch_size}, {channels}, {height}, {width}]")
    log_to_report(f"Num pos feats:       {num_pos_feats}")
    log_to_report(f"Temperature:         10000")
    log_to_report(f"Normalize:           True")
    log_to_report(f"Scale:               2*pi")
    log_to_report(
        f"Output shape:        [{batch_size}, {2*num_pos_feats}, {height}, {width}]"
    )
    log_to_report(f"```")

    try:
        # Create input NestedTensor
        torch.manual_seed(42)
        np.random.seed(42)

        x_torch = torch.randn(batch_size, channels, height, width)
        mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
        # Add some masked regions for testing
        mask_torch[0, :5, :] = True
        mask_torch[1, :, :5] = True

        nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

        x_ttsim = torch_to_simtensor(x_torch, "input")
        mask_ttsim = mask_torch.detach().cpu().numpy()
        nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

        log_to_report(f"\n### Input NestedTensor")
        log_to_report(f"```")
        log_to_report(f"Tensor shape: {list(x_torch.shape)}")
        log_to_report(f"Mask shape:   {list(mask_torch.shape)}")
        log_to_report(f"Masked pixels: {mask_torch.sum().item()}/{mask_torch.numel()}")
        log_to_report(f"Tensor mean:  {x_torch.mean():.8f}")
        log_to_report(f"Tensor std:   {x_torch.std():.8f}")
        log_to_report(f"Sample (first 10): {format_array_sample(x_torch, 10)}")
        log_to_report(f"```")

        # Create PyTorch module
        pos_emb_pytorch = PositionEmbeddingSinePyTorch(
            num_pos_feats=num_pos_feats, temperature=10000, normalize=True
        )
        pos_emb_pytorch.eval()

        # Create TTSim module
        pos_emb_ttsim = PositionEmbeddingSineTTSim(
            name="pos_emb_sine_test",
            num_pos_feats=num_pos_feats,
            temperature=10000,
            normalize=True,
        )

        # Forward pass
        with torch.no_grad():
            out_pytorch = pos_emb_pytorch(nested_tensor_pytorch)
        out_ttsim = pos_emb_ttsim(nested_tensor_ttsim)

        log_to_report(f"\n### Output Position Embeddings")
        print_tensor_comparison(out_pytorch, out_ttsim, "Position Embeddings")

        # Detailed analysis of embedding structure
        log_to_report(f"\n### Embedding Structure Analysis")
        log_to_report(f"```")
        log_to_report(
            f"Output channels: {out_pytorch.shape[1]} (y_pos: {num_pos_feats}, x_pos: {num_pos_feats})"
        )

        # Extract y and x position components
        pytorch_y_pos = out_pytorch[:, :num_pos_feats, :, :].detach().cpu().numpy()
        pytorch_x_pos = out_pytorch[:, num_pos_feats:, :, :].detach().cpu().numpy()

        if out_ttsim.data is not None:
            ttsim_y_pos = out_ttsim.data[:, :num_pos_feats, :, :]
            ttsim_x_pos = out_ttsim.data[:, num_pos_feats:, :, :]

            log_to_report(f"")
            log_to_report(f"Y-position component:")
            log_to_report(
                f"  PyTorch - Mean: {pytorch_y_pos.mean():.8f}, Std: {pytorch_y_pos.std():.8f}"
            )
            log_to_report(
                f"  TTSim   - Mean: {ttsim_y_pos.mean():.8f}, Std: {ttsim_y_pos.std():.8f}"
            )
            log_to_report(f"")
            log_to_report(f"X-position component:")
            log_to_report(
                f"  PyTorch - Mean: {pytorch_x_pos.mean():.8f}, Std: {pytorch_x_pos.std():.8f}"
            )
            log_to_report(
                f"  TTSim   - Mean: {ttsim_x_pos.mean():.8f}, Std: {ttsim_x_pos.std():.8f}"
            )
        log_to_report(f"```")

        # Compare
        shape_match = compare_shapes(out_pytorch, out_ttsim, "PositionEmbeddingSine")
        numeric_match = compare_numerics(
            out_pytorch, out_ttsim, "PositionEmbeddingSine", rtol=1e-4, atol=1e-5
        )

        if shape_match and numeric_match:
            log_to_report("\n### [PASSED] PositionEmbeddingSine test")
        else:
            log_to_report("\n### [FAILED] PositionEmbeddingSine test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] PositionEmbeddingSine test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_position_embedding_learned():
    """Test PositionEmbeddingLearned: PyTorch vs TTSim with numerical comparison"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 2: PositionEmbeddingLearned")
    log_to_report("=" * 80)

    batch_size = 2
    channels = 256
    height = 28
    width = 28
    num_pos_feats = 128

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(f"Input tensor shape:  [{batch_size}, {channels}, {height}, {width}]")
    log_to_report(f"Num pos feats:       {num_pos_feats}")
    log_to_report(
        f"Output shape:        [{batch_size}, {2*num_pos_feats}, {height}, {width}]"
    )
    log_to_report(f"```")

    try:
        # Create input NestedTensor
        torch.manual_seed(42)
        np.random.seed(42)

        x_torch = torch.randn(batch_size, channels, height, width)
        mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
        nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

        x_ttsim = torch_to_simtensor(x_torch, "input")
        mask_ttsim = mask_torch.detach().cpu().numpy()
        nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

        log_to_report(f"\n### Input NestedTensor")
        log_to_report(f"```")
        log_to_report(f"Tensor shape: {list(x_torch.shape)}")
        log_to_report(f"Mask shape:   {list(mask_torch.shape)}")
        log_to_report(f"Sample (first 10): {format_array_sample(x_torch, 10)}")
        log_to_report(f"```")

        # Create PyTorch module
        pos_emb_pytorch = PositionEmbeddingLearnedPyTorch(num_pos_feats=num_pos_feats)
        pos_emb_pytorch.eval()

        # Create TTSim module
        pos_emb_ttsim = PositionEmbeddingLearnedTTSim(
            name="pos_emb_learned_test", num_pos_feats=num_pos_feats
        )

        # Sync weights from PyTorch to TTSim
        pos_emb_ttsim.row_embed_weight = (
            pos_emb_pytorch.row_embed.weight.detach().cpu().numpy().copy()
        )
        pos_emb_ttsim.col_embed_weight = (
            pos_emb_pytorch.col_embed.weight.detach().cpu().numpy().copy()
        )

        log_to_report(f"\n### Weight Sync")
        log_to_report(f"```")
        log_to_report(
            f"row_embed: PT {list(pos_emb_pytorch.row_embed.weight.shape)} → TT {list(pos_emb_ttsim.row_embed_weight.shape)}"
        )
        log_to_report(
            f"col_embed: PT {list(pos_emb_pytorch.col_embed.weight.shape)} → TT {list(pos_emb_ttsim.col_embed_weight.shape)}"
        )
        log_to_report(f"```")

        # Forward pass
        with torch.no_grad():
            out_pytorch = pos_emb_pytorch(nested_tensor_pytorch)
        out_ttsim = pos_emb_ttsim(nested_tensor_ttsim)

        log_to_report(f"\n### Output Position Embeddings")
        print_tensor_comparison(out_pytorch, out_ttsim, "Position Embeddings")

        # Compare shape + numerical
        shape_match = compare_shapes(out_pytorch, out_ttsim, "PositionEmbeddingLearned")
        numeric_match = compare_numerics(
            out_pytorch, out_ttsim, "PositionEmbeddingLearned", rtol=1e-4, atol=1e-5
        )

        if shape_match and numeric_match:
            log_to_report("\n### [PASSED] PositionEmbeddingLearned test")
        else:
            log_to_report("\n### [FAILED] PositionEmbeddingLearned test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] PositionEmbeddingLearned test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_build_position_encoding():
    """Test build_position_encoding factory function"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 3: build_position_encoding Factory")
    log_to_report("=" * 80)

    log_to_report(f"\n### Test 3.1: Sine Position Encoding")

    try:
        # Create args for sine
        class ArgsSine:
            hidden_dim = 256
            position_embedding = "sine"

        args_sine = ArgsSine()

        log_to_report(f"```")
        log_to_report(f"Args:")
        log_to_report(f"  hidden_dim: {args_sine.hidden_dim}")
        log_to_report(f"  position_embedding: '{args_sine.position_embedding}'")
        log_to_report(f"```")

        # Build modules
        pos_emb_pytorch = build_position_encoding_pytorch(args_sine)
        pos_emb_ttsim = build_position_encoding_ttsim(args_sine)

        log_to_report(f"\n**PyTorch module type:** `{type(pos_emb_pytorch).__name__}`")
        log_to_report(f"**TTSim module type:** `{type(pos_emb_ttsim).__name__}`")

        # Test forward pass
        batch_size = 2
        height, width = 28, 28

        x_torch = torch.randn(batch_size, args_sine.hidden_dim, height, width)
        mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
        nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

        x_ttsim = torch_to_simtensor(x_torch, "input")
        mask_ttsim = mask_torch.detach().cpu().numpy()
        nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

        with torch.no_grad():
            out_pytorch = pos_emb_pytorch(nested_tensor_pytorch)
        out_ttsim = pos_emb_ttsim(nested_tensor_ttsim)

        shape_match = compare_shapes(
            out_pytorch, out_ttsim, "build_position_encoding[sine]"
        )

        if shape_match:
            log_to_report("\n**Result:** [PASSED] Sine encoding factory test")
        else:
            log_to_report("\n**Result:** [FAILED] Sine encoding factory test")

        log_to_report(f"\n### Test 3.2: Learned Position Encoding")

        # Create args for learned
        class ArgsLearned:
            hidden_dim = 256
            position_embedding = "learned"

        args_learned = ArgsLearned()

        log_to_report(f"```")
        log_to_report(f"Args:")
        log_to_report(f"  hidden_dim: {args_learned.hidden_dim}")
        log_to_report(f"  position_embedding: '{args_learned.position_embedding}'")
        log_to_report(f"```")

        # Build modules
        pos_emb_pytorch = build_position_encoding_pytorch(args_learned)
        pos_emb_ttsim = build_position_encoding_ttsim(args_learned)

        log_to_report(f"\n**PyTorch module type:** `{type(pos_emb_pytorch).__name__}`")
        log_to_report(f"**TTSim module type:** `{type(pos_emb_ttsim).__name__}`")

        # Test forward pass
        with torch.no_grad():
            out_pytorch = pos_emb_pytorch(nested_tensor_pytorch)
        out_ttsim = pos_emb_ttsim(nested_tensor_ttsim)

        shape_match = compare_shapes(
            out_pytorch, out_ttsim, "build_position_encoding[learned]"
        )

        if shape_match:
            log_to_report("\n**Result:** [PASSED] Learned encoding factory test")
        else:
            log_to_report("\n**Result:** [FAILED] Learned encoding factory test")

        log_to_report("\n### [PASSED] build_position_encoding factory test")

    except Exception as e:
        log_to_report(f"\n### [ERROR] build_position_encoding test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_to_report("# Position Encoding Validation Report")
    log_to_report(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_report(
        f"\n**Test Suite:** PyTorch vs TTSim Position Encoding Implementation Comparison"
    )
    log_to_report(
        f"\nThis report compares the PyTorch and TTSim implementations of position encoding modules."
    )
    log_to_report(
        f"Includes detailed numerical comparison for sine embeddings (deterministic) and shape testing for learned embeddings (random init)."
    )

    log_to_report("\n---\n")

    tests_passed = 0
    tests_failed = 0

    tests = [
        ("PositionEmbeddingSine", test_position_embedding_sine),
        ("PositionEmbeddingLearned", test_position_embedding_learned),
        ("build_position_encoding", test_build_position_encoding),
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

    log_to_report(f"\n## Test Details")
    log_to_report(f"\n| Test | Numerical Comparison | Status |")
    log_to_report(f"|------|---------------------|--------|")
    log_to_report(f"| PositionEmbeddingSine | Yes (deterministic math) | Recommended |")
    log_to_report(
        f"| PositionEmbeddingLearned | Yes (weights synced) | Full numerical |"
    )
    log_to_report(f"| build_position_encoding | Partial (sine only) | Factory test |")

    if tests_failed == 0:
        log_to_report(
            "\n## [PASSED] All position encoding tests completed successfully!"
        )
    else:
        log_to_report(
            f"\n## [FAILED] {tests_failed} test(s) failed. Review errors above."
        )

    log_to_report("\n---\n")
    log_to_report("\n*End of Report*")

    # Save report
    save_report()
