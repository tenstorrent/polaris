#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive validation test comparing PyTorch and TTSim backbone implementations.

Features:
- Shape validation for ALL modules
- Numerical validation with reduced inputs (FrozenBatchNorm2d, ResNetBottleneck)
- Shape-only validation for large modules (Backbone, Joiner)
- Detailed input/output comparison with statistics
- Relative and absolute error analysis
- Markdown report generation

Modules tested:
- FrozenBatchNorm2d (numerical validation with 8x8 inputs)
- ResNetBottleneck (numerical validation with 8x8 inputs and pretrained weights)
- Backbone (shape validation only - 32x32 inputs)
- Joiner (shape validation only - 32x32 inputs)

Test methodology:
- Random inputs with fixed seed for reproducibility
- Reduced spatial dimensions for fast numerical validation
- Pretrained ResNet50 weights transferred to TTSim for ResNetBottleneck
- Shape-only validation for Backbone/Joiner due to slow Conv2d computation
- Tolerance: rtol=1e-4, atol=1e-5

Note: Full numerical validation of ResNet50 (224x224 inputs) would take hours
due to nested-loop NumPy Conv2d implementation in data_compute.py.
"""

import os
import sys
import torch
import numpy as np
from collections import OrderedDict
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
import torchvision
from workloads.Deformable_DETR.reference.backbone import (
    FrozenBatchNorm2d as FrozenBatchNorm2dPyTorch,
    Backbone as BackbonePyTorch,
    Joiner as JoinerPyTorch,
)

# Import TTSim implementations
from workloads.Deformable_DETR.models.backbone_ttsim import (
    FrozenBatchNorm2d as FrozenBatchNorm2dTTSim,
    ResNetBottleneck as ResNetBottleneckTTSim,
    Backbone as BackboneTTSim,
    Joiner as JoinerTTSim,
    Sequential,
)

# Import utilities
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

    report_path = os.path.join(report_dir, "backbone_validation.md")

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


def print_strict_outputs(pytorch_data, ttsim_data, name, num_samples=20):
    """Print STRICT PyTorch vs TTSim output comparison with raw values."""
    log_to_report(f"\n{'='*80}")
    log_to_report(f"  STRICT OUTPUT COMPARISON: {name}")
    log_to_report(f"{'='*80}")

    # Convert to numpy
    if isinstance(pytorch_data, torch.Tensor):
        pt_arr = pytorch_data.detach().cpu().numpy()
    elif isinstance(pytorch_data, SimTensor):
        pt_arr = pytorch_data.data
    else:
        pt_arr = np.array(pytorch_data)

    if isinstance(ttsim_data, SimTensor):
        tt_arr = ttsim_data.data
    elif ttsim_data is None:
        tt_arr = None
    else:
        tt_arr = np.array(ttsim_data)

    # PYTORCH OUTPUT
    log_to_report(f"\n┌{'─'*78}┐")
    log_to_report(f"│{'PYTORCH OUTPUT':^78}│")
    log_to_report(f"├{'─'*78}┤")
    log_to_report(f"│ Shape: {str(list(pt_arr.shape)):70} │")
    log_to_report(f"│ Dtype: {str(pt_arr.dtype):70} │")
    log_to_report(f"│ Mean:  {pt_arr.mean():70.8f} │")
    log_to_report(f"│ Std:   {pt_arr.std():70.8f} │")
    log_to_report(f"│ Min:   {pt_arr.min():70.8f} │")
    log_to_report(f"│ Max:   {pt_arr.max():70.8f} │")
    log_to_report(f"├{'─'*78}┤")
    log_to_report(f"│ First {num_samples} values (flattened):{' '*43}│")
    pt_flat = pt_arr.flatten()[:num_samples]
    for i in range(0, len(pt_flat), 5):
        chunk = pt_flat[i : i + 5]
        vals = "  ".join(f"{v:12.6f}" for v in chunk)
        log_to_report(f"│   [{i:3d}-{i+len(chunk)-1:3d}]: {vals:62} │")
    log_to_report(f"└{'─'*78}┘")

    # TTSIM OUTPUT
    log_to_report(f"\n┌{'─'*78}┐")
    log_to_report(f"│{'TTSIM OUTPUT':^78}│")
    log_to_report(f"├{'─'*78}┤")
    if tt_arr is not None:
        log_to_report(f"│ Shape: {str(list(tt_arr.shape)):70} │")
        log_to_report(f"│ Dtype: {str(tt_arr.dtype):70} │")
        log_to_report(f"│ Mean:  {tt_arr.mean():70.8f} │")
        log_to_report(f"│ Std:   {tt_arr.std():70.8f} │")
        log_to_report(f"│ Min:   {tt_arr.min():70.8f} │")
        log_to_report(f"│ Max:   {tt_arr.max():70.8f} │")
        log_to_report(f"├{'─'*78}┤")
        log_to_report(f"│ First {num_samples} values (flattened):{' '*43}│")
        tt_flat = tt_arr.flatten()[:num_samples]
        for i in range(0, len(tt_flat), 5):
            chunk = tt_flat[i : i + 5]
            vals = "  ".join(f"{v:12.6f}" for v in chunk)
            log_to_report(f"│   [{i:3d}-{i+len(chunk)-1:3d}]: {vals:62} │")
    else:
        log_to_report(f"│ {'(No data - shape inference only)':^76} │")
    log_to_report(f"└{'─'*78}┘")

    # SIDE-BY-SIDE COMPARISON (first 10 values)
    if tt_arr is not None:
        log_to_report(f"\n┌{'─'*78}┐")
        log_to_report(f"│{'SIDE-BY-SIDE COMPARISON (first 10 values)':^78}│")
        log_to_report(f"├{'─'*10}┬{'─'*22}┬{'─'*22}┬{'─'*22}┤")
        log_to_report(
            f"│{'Index':^10}│{'PyTorch':^22}│{'TTSim':^22}│{'Difference':^22}│"
        )
        log_to_report(f"├{'─'*10}┼{'─'*22}┼{'─'*22}┼{'─'*22}┤")
        pt_f = pt_arr.flatten()[:10]
        tt_f = tt_arr.flatten()[:10]
        for i in range(len(pt_f)):
            diff = abs(pt_f[i] - tt_f[i])
            log_to_report(f"│{i:^10}│{pt_f[i]:^22.8f}│{tt_f[i]:^22.8f}│{diff:^22.2e}│")
        log_to_report(f"└{'─'*10}┴{'─'*22}┴{'─'*22}┴{'─'*22}┘")

    log_to_report(f"{'='*80}\n")


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
        lr_backbone = 0.0
        masks = False
        num_feature_levels = 4
        backbone = "resnet50"
        dilation = False

    return Args()


def transfer_weights_from_pytorch_resnet(pytorch_resnet, ttsim_backbone):
    """Transfer weights from PyTorch ResNet to TTSim Backbone."""
    log_to_report(f"\n### Weight Transfer")
    log_to_report(f"Transferring weights from PyTorch ResNet50 to TTSim Backbone...")

    # Helper to transfer conv2d weights to SimOpHandle
    def transfer_conv(pt_conv, tt_conv):
        if hasattr(pt_conv, "weight"):
            weight_data = pt_conv.weight.detach().cpu().numpy().copy()
            # Set data on the parameter tensor in SimOpHandle
            if hasattr(tt_conv, "params") and len(tt_conv.params) > 0:
                # params is list of (position, tensor) tuples
                tt_conv.params[0][1].data = weight_data
            if hasattr(pt_conv, "bias") and pt_conv.bias is not None:
                bias_data = pt_conv.bias.detach().cpu().numpy().copy()
                if len(tt_conv.params) > 1:
                    tt_conv.params[1][1].data = bias_data

    # Helper to transfer BatchNorm weights
    def transfer_bn(pt_bn, tt_bn):
        tt_bn.set_parameters(
            weight=pt_bn.weight.detach().cpu().numpy(),
            bias=pt_bn.bias.detach().cpu().numpy(),
            running_mean=pt_bn.running_mean.detach().cpu().numpy(),
            running_var=pt_bn.running_var.detach().cpu().numpy(),
        )

    # Transfer stem: conv1, bn1
    stem_modules = ttsim_backbone.backbone_layers["stem"].modules_list
    transfer_conv(pytorch_resnet.conv1, stem_modules[0])  # conv1
    transfer_bn(pytorch_resnet.bn1, stem_modules[1])  # bn1

    # Transfer layer1, layer2, layer3, layer4
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        pt_layer = getattr(pytorch_resnet, layer_name)
        tt_layer = ttsim_backbone.backbone_layers[layer_name].modules_list

        for block_idx, (pt_block, tt_block) in enumerate(zip(pt_layer, tt_layer)):
            # Transfer conv1, bn1
            transfer_conv(pt_block.conv1, tt_block.conv1)
            transfer_bn(pt_block.bn1, tt_block.bn1)

            # Transfer conv2, bn2
            transfer_conv(pt_block.conv2, tt_block.conv2)
            transfer_bn(pt_block.bn2, tt_block.bn2)

            # Transfer conv3, bn3
            transfer_conv(pt_block.conv3, tt_block.conv3)
            transfer_bn(pt_block.bn3, tt_block.bn3)

            # Transfer downsample if exists
            if pt_block.downsample is not None and tt_block.downsample is not None:
                ds_modules = tt_block.downsample.modules_list
                transfer_conv(pt_block.downsample[0], ds_modules[0])  # conv
                transfer_bn(pt_block.downsample[1], ds_modules[1])  # bn

    log_to_report(f"[DONE] Weight transfer completed")


# ──────────────────────────────────────────────────────────────────────────────
# Test Functions
# ──────────────────────────────────────────────────────────────────────────────


def test_frozen_batchnorm2d():
    """Test FrozenBatchNorm2d: PyTorch vs TTSim"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 1: FrozenBatchNorm2d")
    log_to_report("=" * 80)

    batch_size = 1
    channels = 64
    height = 8
    width = 8

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(f"Input shape: [{batch_size}, {channels}, {height}, {width}]")
    log_to_report(f"Channels: {channels}")
    log_to_report(f"```")

    try:
        # Create input
        torch.manual_seed(42)
        np.random.seed(42)

        x_torch = torch.randn(batch_size, channels, height, width)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        # Create PyTorch module
        bn_pytorch = FrozenBatchNorm2dPyTorch(channels)
        bn_pytorch.eval()

        # Create TTSim module
        bn_ttsim = FrozenBatchNorm2dTTSim("bn_test", channels)

        # Copy parameters from PyTorch to TTSim
        bn_ttsim.set_parameters(
            weight=bn_pytorch.weight.detach().cpu().numpy(),
            bias=bn_pytorch.bias.detach().cpu().numpy(),
            running_mean=bn_pytorch.running_mean.detach().cpu().numpy(),
            running_var=bn_pytorch.running_var.detach().cpu().numpy(),
        )

        log_to_report(f"\n### Input Tensor")
        print_tensor_comparison(x_torch, x_ttsim, "Input")

        log_to_report(f"\n### Module Parameters")
        log_to_report(f"```")
        log_to_report(
            f"Weight:       mean={bn_pytorch.weight.mean():.6f}, std={bn_pytorch.weight.std():.6f}"
        )
        log_to_report(
            f"Bias:         mean={bn_pytorch.bias.mean():.6f}, std={bn_pytorch.bias.std():.6f}"
        )
        log_to_report(
            f"Running mean: mean={bn_pytorch.running_mean.mean():.6f}, std={bn_pytorch.running_mean.std():.6f}"
        )
        log_to_report(
            f"Running var:  mean={bn_pytorch.running_var.mean():.6f}, std={bn_pytorch.running_var.std():.6f}"
        )
        log_to_report(f"```")

        # Forward pass
        with torch.no_grad():
            out_pytorch = bn_pytorch(x_torch)
        out_ttsim = bn_ttsim(x_ttsim)

        log_to_report(f"\n### Output Tensor")
        print_tensor_comparison(out_pytorch, out_ttsim, "Output")

        # STRICT OUTPUT COMPARISON
        print_strict_outputs(out_pytorch, out_ttsim, "FrozenBatchNorm2d Output")

        # Compare
        shape_match = compare_shapes(out_pytorch, out_ttsim, "FrozenBatchNorm2d")
        numeric_match = compare_numerics(out_pytorch, out_ttsim, "FrozenBatchNorm2d")

        if shape_match and numeric_match:
            log_to_report("\n### [PASSED] FrozenBatchNorm2d test")
        else:
            log_to_report("\n### [FAILED] FrozenBatchNorm2d test")
            raise AssertionError(
                f"FrozenBatchNorm2d validation failed: shape_match={shape_match}, numeric_match={numeric_match}"
            )

    except Exception as e:
        log_to_report(f"\n### [ERROR] FrozenBatchNorm2d test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_resnet_bottleneck():
    """Test ResNetBottleneck: PyTorch vs TTSim with numerical validation"""
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 2: ResNetBottleneck (Numerical Validation)")
    log_to_report("=" * 80)
    log_to_report(
        "\n**Note:** Using reduced spatial dimensions (8x8) for faster numerical computation."
    )
    log_to_report(
        "Full-size validation (56x56) would take hours due to nested-loop Conv2d implementation.\n"
    )

    batch_size = 1
    in_channels = 256
    out_channels = 256
    height = 8
    width = 8

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(f"Input shape:      [{batch_size}, {in_channels}, {height}, {width}]")
    log_to_report(f"Output channels:  {out_channels}")
    log_to_report(f"Stride:           1")
    log_to_report(f"Note:             Reduced size for fast computation")
    log_to_report(f"```")

    try:
        # Create input
        torch.manual_seed(42)
        np.random.seed(42)

        x_torch = torch.randn(batch_size, in_channels, height, width)
        x_ttsim = torch_to_simtensor(x_torch, "input")

        log_to_report(f"\n### Input Tensor")
        print_tensor_comparison(x_torch, x_ttsim, "Input")

        # Create PyTorch Bottleneck from ResNet50 layer1[0]
        resnet_pytorch = torchvision.models.resnet50(
            pretrained=True, norm_layer=FrozenBatchNorm2dPyTorch
        )
        resnet_pytorch.eval()
        bottleneck_pytorch = resnet_pytorch.layer1[
            0
        ]  # First bottleneck: 64->256 with downsample

        # Adjust input for PyTorch bottleneck (layer1 expects 64 channels)
        x_torch_64 = torch.randn(batch_size, 64, height, width)
        x_ttsim_64 = torch_to_simtensor(x_torch_64, "input")

        log_to_report(f"\n### Input Tensor (adjusted to 64 channels for layer1[0])")
        print_tensor_comparison(x_torch_64, x_ttsim_64, "Input")

        # Create TTSim Bottleneck with same structure
        downsample = Sequential(
            "downsample",
            [
                F.Conv2d("downsample.0", 64, 256, kernel_size=1, stride=1, bias=False),
                FrozenBatchNorm2dTTSim("downsample.1", 256),
            ],
        )

        bottleneck_ttsim = ResNetBottleneckTTSim(
            "bottleneck_test",
            in_channels=64,
            out_channels=256,
            stride=1,
            downsample=downsample,
        )

        # Transfer weights from PyTorch to TTSim
        log_to_report(f"\n### Weight Transfer")
        log_to_report(f"Transferring weights from PyTorch bottleneck to TTSim...")

        # Transfer conv layers (set data on parameter tensors)
        bottleneck_ttsim.conv1.params[0][1].data = (
            bottleneck_pytorch.conv1.weight.detach().cpu().numpy().copy()
        )
        bottleneck_ttsim.conv2.params[0][1].data = (
            bottleneck_pytorch.conv2.weight.detach().cpu().numpy().copy()
        )
        bottleneck_ttsim.conv3.params[0][1].data = (
            bottleneck_pytorch.conv3.weight.detach().cpu().numpy().copy()
        )

        # Transfer BatchNorm layers
        bottleneck_ttsim.bn1.set_parameters(
            weight=bottleneck_pytorch.bn1.weight.detach().cpu().numpy(),
            bias=bottleneck_pytorch.bn1.bias.detach().cpu().numpy(),
            running_mean=bottleneck_pytorch.bn1.running_mean.detach().cpu().numpy(),
            running_var=bottleneck_pytorch.bn1.running_var.detach().cpu().numpy(),
        )
        bottleneck_ttsim.bn2.set_parameters(
            weight=bottleneck_pytorch.bn2.weight.detach().cpu().numpy(),
            bias=bottleneck_pytorch.bn2.bias.detach().cpu().numpy(),
            running_mean=bottleneck_pytorch.bn2.running_mean.detach().cpu().numpy(),
            running_var=bottleneck_pytorch.bn2.running_var.detach().cpu().numpy(),
        )
        bottleneck_ttsim.bn3.set_parameters(
            weight=bottleneck_pytorch.bn3.weight.detach().cpu().numpy(),
            bias=bottleneck_pytorch.bn3.bias.detach().cpu().numpy(),
            running_mean=bottleneck_pytorch.bn3.running_mean.detach().cpu().numpy(),
            running_var=bottleneck_pytorch.bn3.running_var.detach().cpu().numpy(),
        )

        # Transfer downsample
        if bottleneck_pytorch.downsample is not None:
            ds_modules = bottleneck_ttsim.downsample.modules_list
            ds_modules[0].params[0][1].data = (
                bottleneck_pytorch.downsample[0].weight.detach().cpu().numpy().copy()
            )
            ds_modules[1].set_parameters(
                weight=bottleneck_pytorch.downsample[1].weight.detach().cpu().numpy(),
                bias=bottleneck_pytorch.downsample[1].bias.detach().cpu().numpy(),
                running_mean=bottleneck_pytorch.downsample[1]
                .running_mean.detach()
                .cpu()
                .numpy(),
                running_var=bottleneck_pytorch.downsample[1]
                .running_var.detach()
                .cpu()
                .numpy(),
            )

        log_to_report(f"[DONE] Weight transfer completed")

        # Forward pass
        with torch.no_grad():
            out_pytorch = bottleneck_pytorch(x_torch_64)
        out_ttsim = bottleneck_ttsim(x_ttsim_64)

        log_to_report(f"\n### Output Tensor")
        print_tensor_comparison(out_pytorch, out_ttsim, "Output")

        # STRICT OUTPUT COMPARISON
        print_strict_outputs(out_pytorch, out_ttsim, "ResNetBottleneck Output")

        # Compare
        shape_match = compare_shapes(out_pytorch, out_ttsim, "ResNetBottleneck")
        numeric_match = compare_numerics(
            out_pytorch, out_ttsim, "ResNetBottleneck", rtol=1e-4, atol=1e-5
        )

        if shape_match and numeric_match:
            log_to_report("\n### [PASSED] ResNetBottleneck test")
        else:
            log_to_report("\n### [FAILED] ResNetBottleneck test")
            raise AssertionError(
                f"ResNetBottleneck validation failed: shape_match={shape_match}, numeric_match={numeric_match}"
            )

    except Exception as e:
        log_to_report(f"\n### [ERROR] ResNetBottleneck test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_backbone():
    """
    Test 3: Backbone (ResNet50) — Shape + Mask validation only.

    WHY NO NUMERICAL COMPARISON:
    - ResNet50 has ~53 bottleneck Conv2d layers, each running through TTSim's
      nested-loop numpy Conv2d (data_compute.py) which is extremely slow.
    - Test 2 (ResNetBottleneck) already validates Conv2d + FrozenBatchNorm2d +
      residual add numerics at *block level* with pretrained weights — proving
      the building blocks are numerically correct.
    - Running the full backbone numerically is therefore redundant and slow.

    WHAT IS VALIDATED:
    - Output shapes for all intermediate layers (layer2, layer3, layer4)
    - Output key names and count match between PyTorch and TTSim
    - Mask shapes match at every layer (interpolated from input mask)
    """
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 3: Backbone (ResNet50 - Shape + Mask Validation)")
    log_to_report("=" * 80)
    log_to_report("\n**Validation:** Shape + mask only (no numerical comparison).")
    log_to_report(
        "**Why:** Block-level numerical correctness already proven in Test 2 (ResNetBottleneck)."
    )
    log_to_report(
        "Full-backbone Conv2d through nested-loop numpy is both redundant and prohibitively slow.\n"
    )

    batch_size = 1
    channels = 3
    height = 32
    width = 32

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(
        f"Input shape:             [{batch_size}, {channels}, {height}, {width}]"
    )
    log_to_report(f"Backbone:                ResNet50")
    log_to_report(f"Return intermediate:     True")
    log_to_report(f"Train backbone:          False")
    log_to_report(f"Dilation:                False")
    log_to_report(
        f"Validation mode:         Shape + mask (numerical covered by Test 2)"
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
        print_tensor_comparison(x_torch, x_ttsim, "Input Tensor")
        log_to_report(f"\nMask shape: {list(mask_torch.shape)}")

        # Create PyTorch backbone
        log_to_report(f"\n### Loading Pretrained ResNet50")
        backbone_pytorch = BackbonePyTorch(
            name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        backbone_pytorch.eval()

        # Create TTSim backbone
        backbone_ttsim = BackboneTTSim(
            name="backbone_test",
            resnet_name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )

        # No weight transfer — shape validation only.
        # Numerical correctness of Conv2d + BN + residual is already validated
        # at block-level in Test 2 (ResNetBottleneck) with pretrained weights.
        log_to_report(f"\n### Weight Transfer: Skipped")
        log_to_report(
            f"Not needed for shape validation. Block-level numerics covered by Test 2."
        )

        # Forward pass
        log_to_report(f"\n### Forward Pass")
        with torch.no_grad():
            out_pytorch = backbone_pytorch(nested_tensor_pytorch)
        out_ttsim = backbone_ttsim(nested_tensor_ttsim)

        # Compare outputs for each layer
        log_to_report(f"\n### Output Layers Comparison")
        log_to_report(
            f"Number of layers: PyTorch={len(out_pytorch)}, TTSim={len(out_ttsim)}"
        )

        # Assert same number of output layers
        assert len(out_pytorch) == len(
            out_ttsim
        ), f"Output layer count mismatch: PyTorch={len(out_pytorch)}, TTSim={len(out_ttsim)}"

        # Assert same output keys
        pt_keys = sorted(out_pytorch.keys())
        tt_keys = sorted(out_ttsim.keys())
        assert (
            pt_keys == tt_keys
        ), f"Output key mismatch: PyTorch={pt_keys}, TTSim={tt_keys}"

        all_shape_match = True

        for layer_name in pt_keys:
            log_to_report(f"\n#### Layer: `{layer_name}`")

            # Extract tensors
            pytorch_nested = out_pytorch[layer_name]
            ttsim_nested = out_ttsim[layer_name]

            pytorch_tensor = pytorch_nested.tensors
            ttsim_tensor = ttsim_nested.tensors

            # STRICT OUTPUT COMPARISON
            print_strict_outputs(
                pytorch_tensor, ttsim_tensor, f"Backbone Layer [{layer_name}]"
            )

            log_to_report(f"\n**Tensor Shapes:**")
            log_to_report(f"```")
            log_to_report(f"PyTorch: {list(pytorch_tensor.shape)}")
            log_to_report(f"TTSim:   {list(ttsim_tensor.shape)}")
            log_to_report(f"```")

            # Compare tensor shapes
            shape_match = compare_shapes(
                pytorch_tensor, ttsim_tensor, f"Backbone[{layer_name}]"
            )
            all_shape_match = all_shape_match and shape_match

            # Compare mask shapes
            pt_mask = pytorch_nested.mask
            tt_mask = ttsim_nested.mask
            pt_mask_shape = list(pt_mask.shape) if pt_mask is not None else None
            tt_mask_shape = list(tt_mask.shape) if tt_mask is not None else None
            log_to_report(f"\n**Mask Shapes:**")
            log_to_report(f"```")
            log_to_report(f"PyTorch: {pt_mask_shape}")
            log_to_report(f"TTSim:   {tt_mask_shape}")
            log_to_report(f"```")
            assert (
                pt_mask_shape == tt_mask_shape
            ), f"Mask shape mismatch at layer {layer_name}: PyTorch={pt_mask_shape}, TTSim={tt_mask_shape}"
            log_to_report(f"**Result:** [PASSED] Mask shapes match")

            # NOTE: Numerical comparison intentionally omitted.
            # Block-level Conv2d + BN + residual numerics already validated in Test 2.
            log_to_report(
                f"[INFO] Numerical comparison not needed (validated at block level in Test 2)"
            )

        if all_shape_match:
            log_to_report(
                "\n### [PASSED] Backbone test - All layer shapes and masks match!"
            )
        else:
            log_to_report("\n### [FAILED] Backbone test - Shape mismatches found")
            raise AssertionError(f"Backbone shape validation failed")

    except Exception as e:
        log_to_report(f"\n### [ERROR] Backbone test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


def test_joiner():
    """
    Test 4: Joiner (Backbone + Position Embedding).

    FEATURE MAPS — Shape + Mask validation only:
    - Same rationale as Test 3: full-backbone Conv2d numerical validation is
      redundant (proven at block level in Test 2) and slow.

    POSITIONAL ENCODING — Full numerical validation:
    - Sine position encoding is pure numpy math (cumsum, sin, cos) with no
      Conv2d dependency. It is deterministic, weight-free, and fast.
    - Numerical parity between PyTorch and TTSim must be verified here since
      position encoding is NOT covered by any other test in this file.
    """
    log_to_report("\n" + "=" * 80)
    log_to_report("## Test 4: Joiner (Backbone + Position Embedding)")
    log_to_report("=" * 80)
    log_to_report(
        "\n**Feature maps:** Shape + mask validation only (Conv2d numerics covered by Test 2)."
    )
    log_to_report(
        "**Positional encoding:** Full numerical comparison (sine is weight-free numpy math).\n"
    )

    batch_size = 1
    channels = 3
    height = 32
    width = 32

    log_to_report(f"\n### Configuration")
    log_to_report(f"```")
    log_to_report(f"Input shape:         [{batch_size}, {channels}, {height}, {width}]")
    log_to_report(f"Position embedding:  sine")
    log_to_report(f"Hidden dim:          256")
    log_to_report(f"Backbone:            ResNet50")
    log_to_report(f"Feature maps:        Shape + mask only")
    log_to_report(f"Positional encoding: Full numerical")
    log_to_report(f"```")

    try:
        # Create dummy args
        args = create_dummy_args()

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
        print_tensor_comparison(x_torch, x_ttsim, "Input Tensor")
        log_to_report(f"\nMask shape: {list(mask_torch.shape)}")

        # Create PyTorch Joiner
        log_to_report(f"\n### Creating PyTorch Joiner (with pretrained backbone)")
        backbone_pytorch = BackbonePyTorch(
            name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        position_embedding_pytorch = build_position_encoding_pytorch(args)
        joiner_pytorch = JoinerPyTorch(backbone_pytorch, position_embedding_pytorch)
        joiner_pytorch.eval()

        # Create TTSim Joiner
        log_to_report(f"Creating TTSim Joiner...")
        backbone_ttsim = BackboneTTSim(
            name="backbone_test",
            resnet_name="resnet50",
            train_backbone=False,
            return_interm_layers=True,
            dilation=False,
        )
        position_embedding_ttsim = build_position_encoding_ttsim(args)
        joiner_ttsim = JoinerTTSim(
            "joiner_test", backbone_ttsim, position_embedding_ttsim
        )

        # No backbone weight transfer — feature maps are shape-only.
        # Block-level Conv2d + BN + residual numerics already validated in Test 2.
        log_to_report(f"\n### Weight Transfer: Skipped")
        log_to_report(f"Feature maps: shape-only. Positional encoding: weight-free.")

        # Forward pass
        log_to_report(f"\n### Forward Pass")
        log_to_report(f"Running forward pass on both implementations...")
        with torch.no_grad():
            out_pytorch, pos_pytorch = joiner_pytorch(nested_tensor_pytorch)
        out_ttsim, pos_ttsim = joiner_ttsim(nested_tensor_ttsim)

        # Compare feature maps
        log_to_report(f"\n### Feature Maps Comparison")
        log_to_report(
            f"Number of feature maps: PyTorch={len(out_pytorch)}, TTSim={len(out_ttsim)}"
        )

        # Assert same number of feature maps
        assert len(out_pytorch) == len(
            out_ttsim
        ), f"Feature map count mismatch: PyTorch={len(out_pytorch)}, TTSim={len(out_ttsim)}"

        all_shape_match = True

        for i, (pytorch_nested, ttsim_nested) in enumerate(zip(out_pytorch, out_ttsim)):
            log_to_report(f"\n#### Feature Map {i}")

            # Extract tensors
            pytorch_tensor = pytorch_nested.tensors
            ttsim_tensor = ttsim_nested.tensors

            # STRICT OUTPUT COMPARISON
            print_strict_outputs(
                pytorch_tensor, ttsim_tensor, f"Joiner Feature Map [{i}]"
            )

            log_to_report(f"```")
            log_to_report(f"PyTorch: {list(pytorch_tensor.shape)}")
            log_to_report(f"TTSim:   {list(ttsim_tensor.shape)}")
            log_to_report(f"```")

            # Compare feature map shapes
            shape_match = compare_shapes(
                pytorch_tensor, ttsim_tensor, f"Joiner[features_{i}]"
            )
            all_shape_match = all_shape_match and shape_match

            # NOTE: Feature map numerical comparison intentionally omitted.
            # Conv2d + BN + residual numerics already validated at block level in Test 2.

            # Compare mask shapes
            pt_mask = pytorch_nested.mask
            tt_mask = ttsim_nested.mask
            pt_mask_shape = list(pt_mask.shape) if pt_mask is not None else None
            tt_mask_shape = list(tt_mask.shape) if tt_mask is not None else None
            log_to_report(f"\n**Mask Shapes:**")
            log_to_report(f"```")
            log_to_report(f"PyTorch: {pt_mask_shape}")
            log_to_report(f"TTSim:   {tt_mask_shape}")
            log_to_report(f"```")
            assert (
                pt_mask_shape == tt_mask_shape
            ), f"Mask shape mismatch at feature map {i}: PyTorch={pt_mask_shape}, TTSim={tt_mask_shape}"
            log_to_report(f"**Result:** [PASSED] Mask shapes match")

        # Compare positional encodings (numerical - sine encoding is deterministic)
        log_to_report(f"\n### Positional Encodings Comparison")
        log_to_report(
            f"Number of positional encodings: PyTorch={len(pos_pytorch)}, TTSim={len(pos_ttsim)}"
        )

        # Assert same number of positional encodings
        assert len(pos_pytorch) == len(
            pos_ttsim
        ), f"Positional encoding count mismatch: PyTorch={len(pos_pytorch)}, TTSim={len(pos_ttsim)}"

        all_pos_match = True
        for i, (pytorch_pos, ttsim_pos) in enumerate(zip(pos_pytorch, pos_ttsim)):
            log_to_report(f"\n#### Positional Encoding {i}")

            # STRICT OUTPUT COMPARISON
            print_strict_outputs(pytorch_pos, ttsim_pos, f"Positional Encoding [{i}]")

            log_to_report(f"```")
            log_to_report(f"PyTorch: {list(pytorch_pos.shape)}")
            log_to_report(f"TTSim:   {list(ttsim_pos.shape)}")
            log_to_report(f"```")

            # Shape comparison
            shape_match = compare_shapes(pytorch_pos, ttsim_pos, f"Joiner[pos_{i}]")
            all_shape_match = all_shape_match and shape_match

            # Numerical comparison (sine position encoding is deterministic, no weights needed)
            pos_numeric_match = compare_numerics(
                pytorch_pos, ttsim_pos, f"Joiner[pos_{i}]", rtol=1e-4, atol=1e-5
            )
            all_pos_match = all_pos_match and pos_numeric_match

        if all_shape_match and all_pos_match:
            log_to_report(
                "\n### [PASSED] Joiner test - All shapes match and positional encodings numerically validated!"
            )
        elif all_shape_match:
            log_to_report(
                "\n### [PARTIAL] Joiner test - Shapes match but positional encoding numerical mismatch"
            )
            raise AssertionError(
                f"Joiner positional encoding numerical validation failed"
            )
        else:
            log_to_report("\n### [FAILED] Joiner test - Shape mismatches found")
            raise AssertionError(f"Joiner shape validation failed")

    except Exception as e:
        log_to_report(f"\n### [ERROR] Joiner test")
        log_to_report(f"```\n{str(e)}\n```")
        import traceback

        log_to_report(f"```python\n{traceback.format_exc()}\n```")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_to_report("# Backbone Validation Report")
    log_to_report(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_to_report(
        f"\n**Test Suite:** PyTorch vs TTSim Backbone Implementation Comparison"
    )
    log_to_report(
        f"\nThis report compares the PyTorch and TTSim implementations of the Deformable DETR backbone modules."
    )
    log_to_report(f"\n## Test Approach")
    log_to_report(f"- **FrozenBatchNorm2d**: Full numerical validation (8x8 inputs)")
    log_to_report(
        f"- **ResNetBottleneck**: Full numerical validation with pretrained weights (8x8 inputs)"
    )
    log_to_report(
        f"- **Backbone**: Shape + mask validation only (32x32 inputs). Numerical redundant — block-level proven in Test 2"
    )
    log_to_report(
        f"- **Joiner**: Shape + mask for feature maps. Full numerical for positional encoding (sine, weight-free numpy)"
    )
    log_to_report(
        f"\n**Rationale:** TTSim Conv2d runs through nested-loop numpy (data_compute.py), making full-backbone"
    )
    log_to_report(
        f"numerical validation both prohibitively slow and redundant since the building blocks are individually validated."
    )

    log_to_report("\n---\n")

    tests_passed = 0
    tests_failed = 0

    tests = [
        ("FrozenBatchNorm2d", test_frozen_batchnorm2d),
        ("ResNetBottleneck", test_resnet_bottleneck),
        ("Backbone", test_backbone),
        ("Joiner", test_joiner),
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

    if tests_failed == 0:
        log_to_report("\n## [PASSED] All backbone tests completed successfully!")
    else:
        log_to_report(
            f"\n## [FAILED] {tests_failed} test(s) failed. Review errors above."
        )

    log_to_report("\n---\n")
    log_to_report("\n*End of Report*")

    # Save report
    save_report()

    # #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0
# """
# Test file comparing PyTorch and TTSim backbone implementations.
# Compares shape and numerical inference module-wise with detailed outputs.
# Generates markdown report of all results.

# Modules tested:
# - FrozenBatchNorm2d
# - ResNetBottleneck
# - Backbone (ResNet)
# - Joiner
# """

# import os
# import sys
# import torch
# import numpy as np
# from collections import OrderedDict
# from datetime import datetime

# # Locate polaris root (contains pyproject.toml) regardless of run directory
_here = os.path.abspath(os.path.dirname(__file__))
_root = _here
while _root != os.path.dirname(_root) and not os.path.exists(
    os.path.join(_root, "pyproject.toml")
):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)

# # Import PyTorch implementations
# from workloads.Deformable_DETR.reference.backbone import (
#     FrozenBatchNorm2d as FrozenBatchNorm2dPyTorch,
#     Backbone as BackbonePyTorch,
#     Joiner as JoinerPyTorch,
# )

# # Import TTSim implementations
# from workloads.Deformable_DETR.models.backbone_ttsim import (
#     FrozenBatchNorm2d as FrozenBatchNorm2dTTSim,
#     ResNetBottleneck as ResNetBottleneckTTSim,
#     Backbone as BackboneTTSim,
#     Joiner as JoinerTTSim,
#     Sequential,
# )

# # Import utilities
# from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPyTorch
# from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTTSim
# from workloads.Deformable_DETR.reference.position_encoding import build_position_encoding as build_position_encoding_pytorch
# from workloads.Deformable_DETR.models.position_encoding_ttsim import build_position_encoding as build_position_encoding_ttsim
# from ttsim.ops.tensor import SimTensor
# import ttsim.front.functional.op as F


# # ──────────────────────────────────────────────────────────────────────────────
# # Global report buffer
# # ──────────────────────────────────────────────────────────────────────────────
# REPORT_BUFFER = []


# def log_to_report(message):
#     """Add message to both console and report buffer"""
#     print(message)
#     REPORT_BUFFER.append(message)


# def save_report():
#     """Save accumulated report to markdown file"""
#     report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
#     os.makedirs(report_dir, exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     report_path = os.path.join(report_dir, f"backbone_test_report_{timestamp}.md")

#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write('\n'.join(REPORT_BUFFER))

#     print(f"\n📄 Report saved to: {report_path}")


# # ──────────────────────────────────────────────────────────────────────────────
# # Helper Functions
# # ──────────────────────────────────────────────────────────────────────────────

# def torch_to_simtensor(torch_tensor, name='tensor'):
#     """Convert PyTorch tensor to SimTensor with proper dtype"""
#     return SimTensor({
#         'name': name,
#         'shape': list(torch_tensor.shape),
#         'data': torch_tensor.detach().cpu().numpy().copy(),
#         'dtype': np.dtype(np.float32)
#     })


# def print_tensor_stats(data, name, indent="  "):
#     """Print detailed statistics of a tensor"""
#     if isinstance(data, torch.Tensor):
#         data = data.detach().cpu().numpy()
#     elif isinstance(data, SimTensor):
#         data = data.data

#     if data is None:
#         msg = f"{indent}{name}: No data (shape inference only)"
#         log_to_report(msg)
#         return

#     msg_lines = [
#         f"{indent}{name}:",
#         f"{indent}  Shape: {list(data.shape)}",
#         f"{indent}  Mean: {data.mean():.6f}, Std: {data.std():.6f}",
#         f"{indent}  Min: {data.min():.6f}, Max: {data.max():.6f}",
#         f"{indent}  Sample values (first 5): {data.flatten()[:5].tolist()}"
#     ]
#     for msg in msg_lines:
#         log_to_report(msg)


# def compare_shapes(torch_output, ttsim_output, test_name):
#     """Compare shapes between PyTorch and TTSim outputs"""
#     if isinstance(torch_output, torch.Tensor):
#         torch_shape = list(torch_output.shape)
#     else:
#         torch_shape = list(torch_output) if hasattr(torch_output, '__iter__') else [torch_output]

#     if isinstance(ttsim_output, SimTensor):
#         ttsim_shape = ttsim_output.shape
#     else:
#         ttsim_shape = list(ttsim_output) if hasattr(ttsim_output, '__iter__') else [ttsim_output]

#     log_to_report(f"\n### {test_name} - Shape Comparison")
#     log_to_report(f"- **PyTorch shape:** `{torch_shape}`")
#     log_to_report(f"- **TTSim shape:** `{ttsim_shape}`")

#     if torch_shape == ttsim_shape:
#         log_to_report(f"- ✅ **Shapes match!**")
#         return True
#     else:
#         log_to_report(f"- ❌ **Shape mismatch!**")
#         return False


# def compare_numerics(torch_output, ttsim_output, test_name, rtol=1e-4, atol=1e-5):
#     """Compare numerical values between PyTorch and TTSim outputs"""
#     if isinstance(torch_output, torch.Tensor):
#         torch_data = torch_output.detach().cpu().numpy()
#     else:
#         torch_data = np.array(torch_output)

#     if isinstance(ttsim_output, SimTensor):
#         ttsim_data = ttsim_output.data
#     else:
#         ttsim_data = np.array(ttsim_output)

#     log_to_report(f"\n### {test_name} - Numerical Comparison")

#     if ttsim_data is None:
#         log_to_report(f"- ⚠️ **TTSim data is None (shape inference only)**")
#         return False

#     log_to_report(f"\n**PyTorch Statistics:**")
#     log_to_report(f"- Mean: {torch_data.mean():.6f}")
#     log_to_report(f"- Std: {torch_data.std():.6f}")
#     log_to_report(f"- Min: {torch_data.min():.6f}")
#     log_to_report(f"- Max: {torch_data.max():.6f}")

#     log_to_report(f"\n**TTSim Statistics:**")
#     log_to_report(f"- Mean: {ttsim_data.mean():.6f}")
#     log_to_report(f"- Std: {ttsim_data.std():.6f}")
#     log_to_report(f"- Min: {ttsim_data.min():.6f}")
#     log_to_report(f"- Max: {ttsim_data.max():.6f}")

#     # Compare numerics
#     diff = np.abs(torch_data - ttsim_data)
#     max_diff = diff.max()
#     mean_diff = diff.mean()

#     log_to_report(f"\n**Differences:**")
#     log_to_report(f"- Max absolute diff: {max_diff:.6e}")
#     log_to_report(f"- Mean absolute diff: {mean_diff:.6e}")

#     if np.allclose(torch_data, ttsim_data, rtol=rtol, atol=atol):
#         log_to_report(f"- ✅ **Numerics match within tolerance!** (rtol={rtol}, atol={atol})")
#         return True
#     else:
#         large_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
#         log_to_report(f"- ⚠️ **Numerical difference exceeds tolerance**")
#         log_to_report(f"- Largest diff at {large_diff_idx}:")
#         log_to_report(f"  - PyTorch: {torch_data[large_diff_idx]:.6f}")
#         log_to_report(f"  - TTSim: {ttsim_data[large_diff_idx]:.6f}")
#         return False


# def create_dummy_args():
#     """Create dummy args object for testing"""
#     class Args:
#         hidden_dim = 256
#         position_embedding = 'sine'
#         lr_backbone = 0.0
#         masks = False
#         num_feature_levels = 4
#         backbone = 'resnet50'
#         dilation = False
#     return Args()


# # ──────────────────────────────────────────────────────────────────────────────
# # Test Functions
# # ──────────────────────────────────────────────────────────────────────────────

# def test_frozen_batchnorm2d():
#     """Test FrozenBatchNorm2d: PyTorch vs TTSim"""
#     log_to_report("\n" + "="*80)
#     log_to_report("## Test 1: FrozenBatchNorm2d")
#     log_to_report("="*80)

#     batch_size = 2
#     channels = 64
#     height = 56
#     width = 56

#     log_to_report(f"\n**Configuration:**")
#     log_to_report(f"- Input shape: [{batch_size}, {channels}, {height}, {width}]")

#     try:
#         # Create input
#         x_torch = torch.randn(batch_size, channels, height, width)
#         x_ttsim = torch_to_simtensor(x_torch, 'input')

#         log_to_report(f"\n**Input:**")
#         print_tensor_stats(x_torch, "PyTorch Input")
#         print_tensor_stats(x_ttsim, "TTSim Input")

#         # Create PyTorch module
#         bn_pytorch = FrozenBatchNorm2dPyTorch(channels)
#         bn_pytorch.eval()

#         # Create TTSim module
#         bn_ttsim = FrozenBatchNorm2dTTSim('bn_test', channels)

#         # Copy parameters from PyTorch to TTSim
#         bn_ttsim.set_parameters(
#             weight=bn_pytorch.weight.detach().cpu().numpy(),
#             bias=bn_pytorch.bias.detach().cpu().numpy(),
#             running_mean=bn_pytorch.running_mean.detach().cpu().numpy(),
#             running_var=bn_pytorch.running_var.detach().cpu().numpy()
#         )

#         log_to_report(f"\n**Parameters:**")
#         log_to_report(f"- Weight mean: {bn_pytorch.weight.mean():.6f}")
#         log_to_report(f"- Bias mean: {bn_pytorch.bias.mean():.6f}")
#         log_to_report(f"- Running mean: {bn_pytorch.running_mean.mean():.6f}")
#         log_to_report(f"- Running var: {bn_pytorch.running_var.mean():.6f}")

#         # Forward pass
#         with torch.no_grad():
#             out_pytorch = bn_pytorch(x_torch)
#         out_ttsim = bn_ttsim(x_ttsim)

#         log_to_report(f"\n**Output:**")
#         print_tensor_stats(out_pytorch, "PyTorch Output")
#         print_tensor_stats(out_ttsim, "TTSim Output")

#         # Compare
#         shape_match = compare_shapes(out_pytorch, out_ttsim, "FrozenBatchNorm2d")
#         numeric_match = compare_numerics(out_pytorch, out_ttsim, "FrozenBatchNorm2d")

#         if shape_match and numeric_match:
#             log_to_report("\n✅ **FrozenBatchNorm2d test PASSED!**")
#         else:
#             log_to_report("\n⚠️ **FrozenBatchNorm2d test completed with warnings**")

#     except Exception as e:
#         log_to_report(f"\n❌ **FrozenBatchNorm2d test FAILED with error:**")
#         log_to_report(f"```\n{str(e)}\n```")
#         import traceback
#         log_to_report(f"```\n{traceback.format_exc()}\n```")
#         raise


# def test_resnet_bottleneck():
#     """Test ResNetBottleneck: TTSim only (no direct PyTorch equivalent in backbone.py)"""
#     log_to_report("\n" + "="*80)
#     log_to_report("## Test 2: ResNetBottleneck (TTSim Shape Inference)")
#     log_to_report("="*80)

#     batch_size = 2
#     in_channels = 64
#     out_channels = 256
#     height = 56
#     width = 56

#     log_to_report(f"\n**Configuration:**")
#     log_to_report(f"- Input shape: [{batch_size}, {in_channels}, {height}, {width}]")
#     log_to_report(f"- Output channels: {out_channels}")

#     try:
#         # Create input
#         x = torch.randn(batch_size, in_channels, height, width)
#         x_ttsim = torch_to_simtensor(x, 'input')

#         log_to_report(f"\n**Input:**")
#         print_tensor_stats(x_ttsim, "TTSim Input")

#         # Create TTSim module with downsample
#         downsample = Sequential('downsample', [
#             F.Conv2d('downsample.0', in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             FrozenBatchNorm2dTTSim('downsample.1', out_channels),
#         ])

#         bottleneck_ttsim = ResNetBottleneckTTSim(
#             'bottleneck_test',
#             in_channels=in_channels,
#             out_channels=out_channels,
#             stride=1,
#             downsample=downsample
#         )

#         # Forward pass
#         out_ttsim = bottleneck_ttsim(x_ttsim)

#         # Check output shape
#         expected_shape = [batch_size, out_channels, height, width]

#         log_to_report(f"\n**Output:**")
#         log_to_report(f"- Input shape: {x_ttsim.shape}")
#         log_to_report(f"- Output shape: {out_ttsim.shape}")
#         log_to_report(f"- Expected shape: {expected_shape}")

#         if out_ttsim.shape == expected_shape:
#             log_to_report(f"- ✅ **Shape is correct!**")
#             log_to_report("\n✅ **ResNetBottleneck test PASSED!**")
#         else:
#             log_to_report(f"- ❌ **Shape mismatch!**")
#             raise AssertionError(f"Shape mismatch: {out_ttsim.shape} vs {expected_shape}")

#     except Exception as e:
#         log_to_report(f"\n❌ **ResNetBottleneck test FAILED with error:**")
#         log_to_report(f"```\n{str(e)}\n```")
#         import traceback
#         log_to_report(f"```\n{traceback.format_exc()}\n```")
#         raise


# def test_backbone():
#     """Test Backbone (ResNet): PyTorch vs TTSim"""
#     log_to_report("\n" + "="*80)
#     log_to_report("## Test 3: Backbone (ResNet50)")
#     log_to_report("="*80)

#     batch_size = 2
#     channels = 3
#     height = 224
#     width = 224

#     log_to_report(f"\n**Configuration:**")
#     log_to_report(f"- Input shape: [{batch_size}, {channels}, {height}, {width}]")
#     log_to_report(f"- Backbone: ResNet50")
#     log_to_report(f"- Return intermediate layers: True")

#     try:
#         # Create input NestedTensor
#         x_torch = torch.randn(batch_size, channels, height, width)
#         mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
#         nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

#         x_ttsim = torch_to_simtensor(x_torch, 'input')
#         mask_ttsim = mask_torch.detach().cpu().numpy()
#         nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

#         log_to_report(f"\n**Input:**")
#         print_tensor_stats(x_torch, "PyTorch Input")
#         print_tensor_stats(x_ttsim, "TTSim Input")
#         log_to_report(f"  Mask shape: {list(mask_torch.shape)}")

#         # Create PyTorch backbone
#         backbone_pytorch = BackbonePyTorch(
#             name='resnet50',
#             train_backbone=False,
#             return_interm_layers=True,
#             dilation=False
#         )
#         backbone_pytorch.eval()

#         # Create TTSim backbone
#         backbone_ttsim = BackboneTTSim(
#             name='backbone_test',
#             resnet_name='resnet50',
#             train_backbone=False,
#             return_interm_layers=True,
#             dilation=False
#         )

#         # Forward pass
#         with torch.no_grad():
#             out_pytorch = backbone_pytorch(nested_tensor_pytorch)
#         out_ttsim = backbone_ttsim(nested_tensor_ttsim)

#         # Compare outputs for each layer
#         log_to_report(f"\n**Output Layers:** {len(out_pytorch)} layers")

#         all_match = True
#         for layer_name in sorted(out_pytorch.keys()):
#             log_to_report(f"\n### Layer: {layer_name}")

#             # Extract tensors
#             pytorch_nested = out_pytorch[layer_name]
#             ttsim_nested = out_ttsim[layer_name]

#             pytorch_tensor = pytorch_nested.tensors
#             ttsim_tensor = ttsim_nested.tensors

#             # Compare shapes
#             shape_match = compare_shapes(pytorch_tensor, ttsim_tensor, f"Backbone[{layer_name}]")
#             all_match = all_match and shape_match

#             # Print shapes
#             log_to_report(f"- Mask shape: PyTorch={list(pytorch_nested.mask.shape)}, TTSim={list(ttsim_nested.mask.shape) if ttsim_nested.mask is not None else 'None'}")
#             log_to_report(f"- ℹ️ Numerical comparison skipped (requires weight transfer)")

#         if all_match:
#             log_to_report("\n✅ **Backbone test PASSED!**")
#         else:
#             log_to_report("\n⚠️ **Backbone test completed with shape mismatches**")

#     except Exception as e:
#         log_to_report(f"\n❌ **Backbone test FAILED with error:**")
#         log_to_report(f"```\n{str(e)}\n```")
#         import traceback
#         log_to_report(f"```\n{traceback.format_exc()}\n```")
#         raise


# def test_joiner():
#     """Test Joiner: PyTorch vs TTSim"""
#     log_to_report("\n" + "="*80)
#     log_to_report("## Test 4: Joiner (Backbone + Position Embedding)")
#     log_to_report("="*80)

#     batch_size = 2
#     channels = 3
#     height = 224
#     width = 224

#     log_to_report(f"\n**Configuration:**")
#     log_to_report(f"- Input shape: [{batch_size}, {channels}, {height}, {width}]")
#     log_to_report(f"- Position embedding: sine")

#     try:
#         # Create dummy args
#         args = create_dummy_args()

#         # Create input NestedTensor
#         x_torch = torch.randn(batch_size, channels, height, width)
#         mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
#         nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

#         x_ttsim = torch_to_simtensor(x_torch, 'input')
#         mask_ttsim = mask_torch.detach().cpu().numpy()
#         nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

#         log_to_report(f"\n**Input:**")
#         print_tensor_stats(x_torch, "PyTorch Input")
#         print_tensor_stats(x_ttsim, "TTSim Input")

#         # Create PyTorch Joiner
#         backbone_pytorch = BackbonePyTorch(
#             name='resnet50',
#             train_backbone=False,
#             return_interm_layers=True,
#             dilation=False
#         )
#         position_embedding_pytorch = build_position_encoding_pytorch(args)
#         joiner_pytorch = JoinerPyTorch(backbone_pytorch, position_embedding_pytorch)
#         joiner_pytorch.eval()

#         # Create TTSim Joiner
#         backbone_ttsim = BackboneTTSim(
#             name='backbone_test',
#             resnet_name='resnet50',
#             train_backbone=False,
#             return_interm_layers=True,
#             dilation=False
#         )
#         position_embedding_ttsim = build_position_encoding_ttsim(args)
#         joiner_ttsim = JoinerTTSim('joiner_test', backbone_ttsim, position_embedding_ttsim)

#         # Forward pass
#         with torch.no_grad():
#             out_pytorch, pos_pytorch = joiner_pytorch(nested_tensor_pytorch)
#         out_ttsim, pos_ttsim = joiner_ttsim(nested_tensor_ttsim)

#         # Compare outputs
#         log_to_report(f"\n### Feature Maps: {len(out_pytorch)} outputs")

#         all_match = True
#         for i, (pytorch_nested, ttsim_nested) in enumerate(zip(out_pytorch, out_ttsim)):
#             log_to_report(f"\n**Feature Map {i}:**")

#             # Extract tensors
#             pytorch_tensor = pytorch_nested.tensors
#             ttsim_tensor = ttsim_nested.tensors

#             # Compare shapes
#             shape_match = compare_shapes(pytorch_tensor, ttsim_tensor, f"Joiner[features_{i}]")
#             all_match = all_match and shape_match

#         log_to_report(f"\n### Positional Encodings: {len(pos_pytorch)} outputs")

#         for i, (pytorch_pos, ttsim_pos) in enumerate(zip(pos_pytorch, pos_ttsim)):
#             log_to_report(f"\n**Positional Encoding {i}:**")

#             # Compare shapes
#             shape_match = compare_shapes(pytorch_pos, ttsim_pos, f"Joiner[pos_{i}]")
#             all_match = all_match and shape_match

#         if all_match:
#             log_to_report("\n✅ **Joiner test PASSED!**")
#         else:
#             log_to_report("\n⚠️ **Joiner test completed with shape mismatches**")

#     except Exception as e:
#         log_to_report(f"\n❌ **Joiner test FAILED with error:**")
#         log_to_report(f"```\n{str(e)}\n```")
#         import traceback
#         log_to_report(f"```\n{traceback.format_exc()}\n```")
#         raise


# # ──────────────────────────────────────────────────────────────────────────────
# # Run all tests
# # ──────────────────────────────────────────────────────────────────────────────

# if __name__ == '__main__':
#     log_to_report("# Backbone Test Report")
#     log_to_report(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     log_to_report(f"\n**Test Suite:** PyTorch vs TTSim Backbone Comparison")

#     tests_passed = 0
#     tests_failed = 0

#     tests = [
#         ("FrozenBatchNorm2d", test_frozen_batchnorm2d),
#         ("ResNetBottleneck", test_resnet_bottleneck),
#         ("Backbone", test_backbone),
#         ("Joiner", test_joiner),
#     ]

#     for test_name, test_func in tests:
#         try:
#             test_func()
#             tests_passed += 1
#         except Exception as e:
#             tests_failed += 1
#             log_to_report(f"\n⚠️ Test {test_name} encountered errors but continuing...")

#     log_to_report("\n" + "="*80)
#     log_to_report("# Test Summary")
#     log_to_report("="*80)
#     log_to_report(f"\n- **Tests Passed:** {tests_passed}/{len(tests)}")
#     log_to_report(f"- **Tests Failed:** {tests_failed}/{len(tests)}")

#     if tests_failed == 0:
#         log_to_report("\n✅ **All backbone tests PASSED!**")
#     else:
#         log_to_report(f"\n⚠️ **{tests_failed} test(s) failed. Review errors above.**")

#     # Save report
#     save_report()


# # #!/usr/bin/env python
# # # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # # SPDX-License-Identifier: Apache-2.0
# # """
# # Test file comparing PyTorch and TTSim backbone implementations.
# # Compares shape and numerical inference module-wise.

# # Modules tested:
# # - FrozenBatchNorm2d
# # - ResNetBottleneck
# # - Backbone (ResNet)
# # - Joiner
# # """

# # import os
# # import sys
# # import pytest
# # import torch
# # import numpy as np
# # from collections import OrderedDict

# # # Add paths
# # #sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# # import sys, os
# # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
# # # Import PyTorch implementations
# # from workloads.Deformable_DETR.reference.backbone import (
# #     FrozenBatchNorm2d as FrozenBatchNorm2dPyTorch,
# #     Backbone as BackbonePyTorch,
# #     Joiner as JoinerPyTorch,
# # )

# # # Import TTSim implementations
# # from workloads.Deformable_DETR.models.backbone_ttsim import (
# #     FrozenBatchNorm2d as FrozenBatchNorm2dTTSim,
# #     ResNetBottleneck as ResNetBottleneckTTSim,
# #     Backbone as BackboneTTSim,
# #     Joiner as JoinerTTSim,
# # )

# # # Import utilities
# # from workloads.Deformable_DETR.reference.misc import NestedTensor as NestedTensorPyTorch
# # from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor as NestedTensorTTSim
# # from workloads.Deformable_DETR.reference.position_encoding import build_position_encoding as build_position_encoding_pytorch
# # from workloads.Deformable_DETR.models.position_encoding_ttsim import build_position_encoding as build_position_encoding_ttsim
# # from ttsim.ops.tensor import SimTensor


# # # ──────────────────────────────────────────────────────────────────────────────
# # # Helper Functions
# # # ──────────────────────────────────────────────────────────────────────────────

# # # def torch_to_simtensor(torch_tensor, name='tensor'):
# # #     """Convert PyTorch tensor to SimTensor"""
# # #     return SimTensor({
# # #         'name': name,
# # #         'shape': list(torch_tensor.shape),
# # #         'data': torch_tensor.detach().cpu().numpy().copy(),
# # #         'dtype': np.float32
# # #     })

# # def torch_to_simtensor(torch_tensor, name='tensor'):
# #     """Convert PyTorch tensor to SimTensor"""
# #     return SimTensor({
# #         'name': name,
# #         'shape': list(torch_tensor.shape),
# #         'data': torch_tensor.detach().cpu().numpy().copy(),
# #         'dtype': np.dtype(np.float32)  # Changed: use np.dtype() wrapper
# #     })

# # def compare_shapes(torch_output, ttsim_output, test_name):
# #     """Compare shapes between PyTorch and TTSim outputs"""
# #     if isinstance(torch_output, torch.Tensor):
# #         torch_shape = list(torch_output.shape)
# #     else:
# #         torch_shape = list(torch_output)

# #     if isinstance(ttsim_output, SimTensor):
# #         ttsim_shape = ttsim_output.shape
# #     else:
# #         ttsim_shape = list(ttsim_output)

# #     print(f"\n{test_name} - Shape Comparison:")
# #     print(f"  PyTorch: {torch_shape}")
# #     print(f"  TTSim:   {ttsim_shape}")

# #     assert torch_shape == ttsim_shape, f"Shape mismatch: {torch_shape} vs {ttsim_shape}"
# #     print(f"  ✓ Shapes match!")


# # def compare_numerics(torch_output, ttsim_output, test_name, rtol=1e-4, atol=1e-5):
# #     """Compare numerical values between PyTorch and TTSim outputs"""
# #     if isinstance(torch_output, torch.Tensor):
# #         torch_data = torch_output.detach().cpu().numpy()
# #     else:
# #         torch_data = np.array(torch_output)

# #     if isinstance(ttsim_output, SimTensor):
# #         ttsim_data = ttsim_output.data
# #     else:
# #         ttsim_data = np.array(ttsim_output)

# #     if ttsim_data is None:
# #         print(f"\n{test_name} - Numerical Comparison:")
# #         print(f"  ⚠ TTSim data is None (shape inference only)")
# #         return

# #     print(f"\n{test_name} - Numerical Comparison:")
# #     print(f"  PyTorch stats: mean={torch_data.mean():.6f}, std={torch_data.std():.6f}, "
# #           f"min={torch_data.min():.6f}, max={torch_data.max():.6f}")
# #     print(f"  TTSim stats:   mean={ttsim_data.mean():.6f}, std={ttsim_data.std():.6f}, "
# #           f"min={ttsim_data.min():.6f}, max={ttsim_data.max():.6f}")

# #     # Compare numerics
# #     diff = np.abs(torch_data - ttsim_data)
# #     max_diff = diff.max()
# #     mean_diff = diff.mean()

# #     print(f"  Absolute diff: max={max_diff:.6e}, mean={mean_diff:.6e}")

# #     if not np.allclose(torch_data, ttsim_data, rtol=rtol, atol=atol):
# #         # Find locations of largest differences
# #         large_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
# #         print(f"  ⚠ Numerical difference exceeds tolerance!")
# #         print(f"    Largest diff at {large_diff_idx}: PyTorch={torch_data[large_diff_idx]:.6f}, "
# #               f"TTSim={ttsim_data[large_diff_idx]:.6f}")
# #         # Don't fail, just warn
# #         print(f"  ⚠ Warning: Numerical mismatch (continuing test)")
# #     else:
# #         print(f"  ✓ Numerics match within tolerance!")


# # def create_dummy_args():
# #     """Create dummy args object for testing"""
# #     class Args:
# #         hidden_dim = 256
# #         position_embedding = 'sine'
# #         lr_backbone = 0.0
# #         masks = False
# #         num_feature_levels = 4
# #         backbone = 'resnet50'
# #         dilation = False
# #     return Args()


# # # ──────────────────────────────────────────────────────────────────────────────
# # # Test Functions
# # # ──────────────────────────────────────────────────────────────────────────────

# # def test_frozen_batchnorm2d():
# #     """Test FrozenBatchNorm2d: PyTorch vs TTSim"""
# #     print("\n" + "="*80)
# #     print("Testing FrozenBatchNorm2d")
# #     print("="*80)

# #     batch_size = 2
# #     channels = 64
# #     height = 56
# #     width = 56

# #     # Create input
# #     x_torch = torch.randn(batch_size, channels, height, width)
# #     x_ttsim = torch_to_simtensor(x_torch, 'input')

# #     # Create PyTorch module
# #     bn_pytorch = FrozenBatchNorm2dPyTorch(channels)
# #     bn_pytorch.eval()

# #     # Create TTSim module
# #     bn_ttsim = FrozenBatchNorm2dTTSim('bn_test', channels)

# #     # Copy parameters from PyTorch to TTSim
# #     bn_ttsim.set_parameters(
# #         weight=bn_pytorch.weight.detach().cpu().numpy(),
# #         bias=bn_pytorch.bias.detach().cpu().numpy(),
# #         running_mean=bn_pytorch.running_mean.detach().cpu().numpy(),
# #         running_var=bn_pytorch.running_var.detach().cpu().numpy()
# #     )

# #     # Forward pass
# #     with torch.no_grad():
# #         out_pytorch = bn_pytorch(x_torch)
# #     out_ttsim = bn_ttsim(x_ttsim)

# #     # Compare
# #     compare_shapes(out_pytorch, out_ttsim, "FrozenBatchNorm2d")
# #     compare_numerics(out_pytorch, out_ttsim, "FrozenBatchNorm2d")

# #     print("\n✓ FrozenBatchNorm2d test passed!")


# # def test_resnet_bottleneck():
# #     """Test ResNetBottleneck: TTSim only (no direct PyTorch equivalent in backbone.py)"""
# #     print("\n" + "="*80)
# #     print("Testing ResNetBottleneck (TTSim)")
# #     print("="*80)

# #     batch_size = 2
# #     in_channels = 64
# #     out_channels = 256
# #     height = 56
# #     width = 56

# #     # Create input
# #     x = torch.randn(batch_size, in_channels, height, width)
# #     x_ttsim = torch_to_simtensor(x, 'input')

# #     # Create TTSim module with downsample
# #     from workloads.Deformable_DETR.models.backbone_ttsim import Sequential
# #     import ttsim.front.functional.op as F

# #     downsample = Sequential('downsample', [
# #         F.Conv2d('downsample.0', in_channels, out_channels, kernel_size=1, stride=1, bias=False),
# #         FrozenBatchNorm2dTTSim('downsample.1', out_channels),
# #     ])

# #     bottleneck_ttsim = ResNetBottleneckTTSim(
# #         'bottleneck_test',
# #         in_channels=in_channels,
# #         out_channels=out_channels,
# #         stride=1,
# #         downsample=downsample
# #     )

# #     # Forward pass
# #     out_ttsim = bottleneck_ttsim(x_ttsim)

# #     # Check output shape
# #     expected_shape = [batch_size, out_channels, height, width]
# #     print(f"\nResNetBottleneck - Shape Check:")
# #     print(f"  Input:    {x_ttsim.shape}")
# #     print(f"  Output:   {out_ttsim.shape}")
# #     print(f"  Expected: {expected_shape}")

# #     assert out_ttsim.shape == expected_shape, f"Shape mismatch: {out_ttsim.shape} vs {expected_shape}"
# #     print(f"  ✓ Shape is correct!")

# #     print("\n✓ ResNetBottleneck test passed!")


# # def test_backbone():
# #     """Test Backbone (ResNet): PyTorch vs TTSim"""
# #     print("\n" + "="*80)
# #     print("Testing Backbone (ResNet50)")
# #     print("="*80)

# #     batch_size = 2
# #     channels = 3
# #     height = 224
# #     width = 224

# #     # Create input NestedTensor
# #     x_torch = torch.randn(batch_size, channels, height, width)
# #     mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
# #     nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

# #     x_ttsim = torch_to_simtensor(x_torch, 'input')
# #     mask_ttsim = mask_torch.detach().cpu().numpy()
# #     nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

# #     # Create PyTorch backbone
# #     backbone_pytorch = BackbonePyTorch(
# #         name='resnet50',
# #         train_backbone=False,
# #         return_interm_layers=True,
# #         dilation=False
# #     )
# #     backbone_pytorch.eval()

# #     # Create TTSim backbone
# #     backbone_ttsim = BackboneTTSim(
# #         name='backbone_test',
# #         resnet_name='resnet50',
# #         train_backbone=False,
# #         return_interm_layers=True,
# #         dilation=False
# #     )

# #     # Forward pass
# #     with torch.no_grad():
# #         out_pytorch = backbone_pytorch(nested_tensor_pytorch)
# #     out_ttsim = backbone_ttsim(nested_tensor_ttsim)

# #     # Compare outputs for each layer
# #     print(f"\nBackbone outputs {len(out_pytorch)} layers")

# #     for layer_name in sorted(out_pytorch.keys()):
# #         print(f"\n--- Layer: {layer_name} ---")

# #         # Extract tensors
# #         pytorch_nested = out_pytorch[layer_name]
# #         ttsim_nested = out_ttsim[layer_name]

# #         pytorch_tensor = pytorch_nested.tensors
# #         ttsim_tensor = ttsim_nested.tensors

# #         # Compare shapes
# #         compare_shapes(pytorch_tensor, ttsim_tensor, f"Backbone[{layer_name}]")

# #         # Note: Numerical comparison requires weight transfer (not done here)
# #         print(f"  ℹ Numerical comparison skipped (requires weight transfer)")

# #     print("\n✓ Backbone test passed!")


# # def test_joiner():
# #     """Test Joiner: PyTorch vs TTSim"""
# #     print("\n" + "="*80)
# #     print("Testing Joiner")
# #     print("="*80)

# #     batch_size = 2
# #     channels = 3
# #     height = 224
# #     width = 224

# #     # Create dummy args
# #     args = create_dummy_args()

# #     # Create input NestedTensor
# #     x_torch = torch.randn(batch_size, channels, height, width)
# #     mask_torch = torch.zeros(batch_size, height, width, dtype=torch.bool)
# #     nested_tensor_pytorch = NestedTensorPyTorch(x_torch, mask_torch)

# #     x_ttsim = torch_to_simtensor(x_torch, 'input')
# #     mask_ttsim = mask_torch.detach().cpu().numpy()
# #     nested_tensor_ttsim = NestedTensorTTSim(x_ttsim, mask_ttsim)

# #     # Create PyTorch Joiner
# #     backbone_pytorch = BackbonePyTorch(
# #         name='resnet50',
# #         train_backbone=False,
# #         return_interm_layers=True,
# #         dilation=False
# #     )
# #     position_embedding_pytorch = build_position_encoding_pytorch(args)
# #     joiner_pytorch = JoinerPyTorch(backbone_pytorch, position_embedding_pytorch)
# #     joiner_pytorch.eval()

# #     # Create TTSim Joiner
# #     backbone_ttsim = BackboneTTSim(
# #         name='backbone_test',
# #         resnet_name='resnet50',
# #         train_backbone=False,
# #         return_interm_layers=True,
# #         dilation=False
# #     )
# #     position_embedding_ttsim = build_position_encoding_ttsim(args)
# #     joiner_ttsim = JoinerTTSim('joiner_test', backbone_ttsim, position_embedding_ttsim)

# #     # Forward pass
# #     with torch.no_grad():
# #         out_pytorch, pos_pytorch = joiner_pytorch(nested_tensor_pytorch)
# #     out_ttsim, pos_ttsim = joiner_ttsim(nested_tensor_ttsim)

# #     # Compare outputs
# #     print(f"\nJoiner outputs {len(out_pytorch)} feature maps")

# #     for i, (pytorch_nested, ttsim_nested) in enumerate(zip(out_pytorch, out_ttsim)):
# #         print(f"\n--- Feature Map {i} ---")

# #         # Extract tensors
# #         pytorch_tensor = pytorch_nested.tensors
# #         ttsim_tensor = ttsim_nested.tensors

# #         # Compare shapes
# #         compare_shapes(pytorch_tensor, ttsim_tensor, f"Joiner[features_{i}]")

# #     print(f"\nJoiner outputs {len(pos_pytorch)} positional encodings")

# #     for i, (pytorch_pos, ttsim_pos) in enumerate(zip(pos_pytorch, pos_ttsim)):
# #         print(f"\n--- Positional Encoding {i} ---")

# #         # Compare shapes
# #         compare_shapes(pytorch_pos, ttsim_pos, f"Joiner[pos_{i}]")

# #     print("\n✓ Joiner test passed!")


# # # ──────────────────────────────────────────────────────────────────────────────
# # # Run all tests
# # # ──────────────────────────────────────────────────────────────────────────────

# # if __name__ == '__main__':
# #     test_frozen_batchnorm2d()
# #     test_resnet_bottleneck()
# #     test_backbone()
# #     test_joiner()
# #     print("\n" + "="*80)
# #     print("All backbone tests passed! ✓")
# #     print("="*80)
