#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison script: MyResBlock PyTorch vs ttsim
Tests residual block with Conv3x3 + BN + ReLU
"""

import os, sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

import torch
import torch.nn as nn
import numpy as np

import ttsim.front.functional.op as F


def kaiming_init_weights(module):
    """Apply Kaiming initialization to Conv2d layers"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# PyTorch implementation
class MyResBlockPyTorch(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()
        padding = dilation

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


# ttsim implementation
from workloads.MapTracker.plugin.models.backbones.bevformer.temporal_net import (
    MyResBlock as MyResBlockTtsim,
)

print("=" * 70)
print("MyResBlock Comparison: PyTorch vs ttsim")
print("=" * 70)

# Configuration
inplanes = 64
planes = 64
batch_size = 2
height = 16
width = 16

print(f"\nConfiguration:")
print(f"  inplanes={inplanes}, planes={planes}")
print(f"  Input shape: [{batch_size}, {inplanes}, {height}, {width}]")

# Create random input
input_np = np.random.randn(batch_size, inplanes, height, width).astype(np.float32)
print(input_np)
# ============================================================
# TEST 1: PyTorch MyResBlock
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: PyTorch MyResBlock")
print("-" * 70)

resblock_pytorch = MyResBlockPyTorch(inplanes=inplanes, planes=planes)
resblock_pytorch.eval()

# Apply Kaiming initialization
kaiming_init_weights(resblock_pytorch)
print("  Applied Kaiming initialization to Conv2d layers")

input_torch = torch.from_numpy(input_np)

with torch.no_grad():
    output_pytorch = resblock_pytorch(input_torch)

output_pytorch_np = output_pytorch.numpy()
print(output_pytorch_np)

print(f"  Input shape: {input_torch.shape}")
print(f"  Output shape: {output_pytorch.shape}")
print(f"  Output stats:")
print(f"    Min:  {output_pytorch_np.min():.6f}")
print(f"    Max:  {output_pytorch_np.max():.6f}")
print(f"    Mean: {output_pytorch_np.mean():.6f}")
print(f"    Std:  {output_pytorch_np.std():.6f}")

# ============================================================
# TEST 2: ttsim MyResBlock
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: ttsim MyResBlock")
print("-" * 70)

resblock_ttsim = MyResBlockTtsim(name="resblock_test", inplanes=inplanes, planes=planes)

# Inject PyTorch weights into ttsim operators
# CRITICAL: Set .data on params[x][1], don't replace the tensors!

# Conv1: weight only (no bias)
resblock_ttsim.conv1.params[0][1].data = resblock_pytorch.conv1.weight.data.numpy()

# BN1: weight, bias, running_mean, running_var
resblock_ttsim.bn1.params[0][1].data = resblock_pytorch.bn1.weight.data.numpy()
resblock_ttsim.bn1.params[1][1].data = resblock_pytorch.bn1.bias.data.numpy()
resblock_ttsim.bn1.params[2][1].data = resblock_pytorch.bn1.running_mean.numpy()
resblock_ttsim.bn1.params[3][1].data = resblock_pytorch.bn1.running_var.numpy()

# Conv2: weight only (no bias)
resblock_ttsim.conv2.params[0][1].data = resblock_pytorch.conv2.weight.data.numpy()

# BN2: weight, bias, running_mean, running_var
resblock_ttsim.bn2.params[0][1].data = resblock_pytorch.bn2.weight.data.numpy()
resblock_ttsim.bn2.params[1][1].data = resblock_pytorch.bn2.bias.data.numpy()
resblock_ttsim.bn2.params[2][1].data = resblock_pytorch.bn2.running_mean.numpy()
resblock_ttsim.bn2.params[3][1].data = resblock_pytorch.bn2.running_var.numpy()

# Create input SimTensor with data
input_simtensor = F._from_data("input", data=input_np, is_const=False)
output_ttsim = resblock_ttsim(input_simtensor)

print(f"  Input shape: {input_simtensor.shape}")
print(f"  Output shape: {output_ttsim.shape}")
print(f"  Output .data is None? {output_ttsim.data is None}")
print(output_ttsim.data)

if output_ttsim.data is not None:
    print(f"  Output stats:")
    print(f"    Min:  {output_ttsim.data.min():.6f}")
    print(f"    Max:  {output_ttsim.data.max():.6f}")
    print(f"    Mean: {output_ttsim.data.mean():.6f}")
    print(f"    Std:  {output_ttsim.data.std():.6f}")
else:
    print("  WARNING: output.data is None (shape inference only)")

# ============================================================
# TEST 3: Numerical Comparison
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Numerical Comparison")
print("-" * 70)

if output_ttsim.data is not None:
    ttsim_output = output_ttsim.data

    # Compute difference
    diff = np.abs(output_pytorch_np - ttsim_output)

    print(f"  Absolute difference:")
    print(f"    Min:  {diff.min():.10e}")
    print(f"    Max:  {diff.max():.10e}")
    print(f"    Mean: {diff.mean():.10e}")
    print(f"    Std:  {diff.std():.10e}")

    # Check if outputs match within tolerance
    atol = 1e-5  # Relaxed from 1e-6 to account for floating-point precision
    rtol = 1e-5
    matches = np.allclose(output_pytorch_np, ttsim_output, atol=atol, rtol=rtol)

    if matches:
        print(f"\n  [PASS] Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"\n  [FAIL] Outputs differ beyond tolerance (atol={atol}, rtol={rtol})")

        # Show first mismatch
        diff_mask = ~np.isclose(output_pytorch_np, ttsim_output, atol=atol, rtol=rtol)
        if diff_mask.any():
            mismatch_idx = np.where(diff_mask)
            b, c, h, w = (
                mismatch_idx[0][0],
                mismatch_idx[1][0],
                mismatch_idx[2][0],
                mismatch_idx[3][0],
            )
            print(f"\n  First mismatch at [batch={b}, channel={c}, h={h}, w={w}]:")
            print(f"    PyTorch: {output_pytorch_np[b, c, h, w]:.10f}")
            print(f"    ttsim:   {ttsim_output[b, c, h, w]:.10f}")
            print(f"    Diff:    {diff[b, c, h, w]:.10e}")
else:
    print("  [SKIP] Cannot compare - ttsim output.data is None")

print()
print("=" * 70)
print("Test Complete!")
print("=" * 70)
