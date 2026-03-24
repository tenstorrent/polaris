#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison script: TemporalNet PyTorch vs ttsim
Tests temporal feature fusion with residual blocks
"""

import os, sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

import ttsim.front.functional.op as F

# Import ttsim TemporalNet
from workloads.MapTracker.plugin.models.backbones.bevformer.temporal_net import (
    TemporalNet as TemporalNetTtsim,
)


def kaiming_init_weights(module):
    """Apply Kaiming initialization to Conv2d layers"""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# PyTorch implementations (to avoid mmcv dependencies)
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
        self.relu = nn.ReLU(inplace=True)
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


class TemporalNetPyTorch(nn.Module):
    def __init__(self, history_steps, hidden_dims, num_blocks):
        super().__init__()
        self.history_steps = history_steps
        self.hidden_dims = hidden_dims
        self.num_blocks = num_blocks

        # Initial conv layer
        in_dims = (history_steps + 1) * hidden_dims
        self.conv_in = nn.Conv2d(
            in_dims, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(hidden_dims)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        layers = []
        for _ in range(num_blocks):
            layers.append(MyResBlockPyTorch(hidden_dims, hidden_dims))
        self.res_layer = nn.Sequential(*layers)

    def forward(self, history_feats, curr_feat):
        # Concatenate history and current features along time dimension
        input_feats = torch.cat([history_feats, curr_feat.unsqueeze(1)], dim=1)

        # Rearrange using einops
        input_feats = rearrange(input_feats, "b t c h w -> b (t c) h w")

        # Initial conv block
        out = self.conv_in(input_feats)
        out = self.bn(out)
        out = self.relu(out)

        # Residual blocks
        out = self.res_layer(out)

        # Conditional squeeze if curr_feat was 3D
        if curr_feat.dim() == 3:
            out = out.squeeze(0)

        return out


print("=" * 70)
print("TemporalNet Comparison: PyTorch vs ttsim")
print("=" * 70)

# Configuration
history_steps = 2  # Number of history frames
hidden_dims = 64  # Feature dimension (smaller for testing)
num_blocks = 2  # Number of residual blocks
batch_size = 2
height = 16
width = 16

print(f"\nConfiguration:")
print(f"  history_steps={history_steps}")
print(f"  hidden_dims={hidden_dims}")
print(f"  num_blocks={num_blocks}")
print(
    f"  History features shape: [{batch_size}, {history_steps}, {hidden_dims}, {height}, {width}]"
)
print(f"  Current feature shape: [{batch_size}, {hidden_dims}, {height}, {width}]")

# Create random inputs
np.random.seed(42)
torch.manual_seed(42)

history_feats_np = np.random.randn(
    batch_size, history_steps, hidden_dims, height, width
).astype(np.float32)
curr_feat_np = np.random.randn(batch_size, hidden_dims, height, width).astype(
    np.float32
)

# ============================================================
# TEST 1: PyTorch TemporalNet
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: PyTorch TemporalNet")
print("-" * 70)

temporal_net_pytorch = TemporalNetPyTorch(
    history_steps=history_steps, hidden_dims=hidden_dims, num_blocks=num_blocks
)
temporal_net_pytorch.eval()

# Apply Kaiming initialization
kaiming_init_weights(temporal_net_pytorch)
print("  Applied Kaiming initialization to Conv2d layers")

history_feats_torch = torch.from_numpy(history_feats_np)
curr_feat_torch = torch.from_numpy(curr_feat_np)

with torch.no_grad():
    output_pytorch = temporal_net_pytorch(history_feats_torch, curr_feat_torch)

output_pytorch_np = output_pytorch.numpy()

print(f"  History features shape: {history_feats_torch.shape}")
print(f"  Current feature shape: {curr_feat_torch.shape}")
print(f"  Output shape: {output_pytorch.shape}")
print(f" Output: {output_pytorch_np}")
print(f"  Output stats:")
print(f"    Min:  {output_pytorch_np.min():.6f}")
print(f"    Max:  {output_pytorch_np.max():.6f}")
print(f"    Mean: {output_pytorch_np.mean():.6f}")
print(f"    Std:  {output_pytorch_np.std():.6f}")

# ============================================================
# TEST 2: ttsim TemporalNet
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: ttsim TemporalNet")
print("-" * 70)

temporal_net_ttsim = TemporalNetTtsim(
    name="temporal_net_test",
    history_steps=history_steps,
    hidden_dims=hidden_dims,
    num_blocks=num_blocks,
)

# Inject PyTorch weights into ttsim operators

# Conv_in: weight only (no bias)
temporal_net_ttsim.conv_in.params[0][
    1
].data = temporal_net_pytorch.conv_in.weight.data.numpy()

# BN: weight, bias, running_mean, running_var
temporal_net_ttsim.bn.params[0][1].data = temporal_net_pytorch.bn.weight.data.numpy()
temporal_net_ttsim.bn.params[1][1].data = temporal_net_pytorch.bn.bias.data.numpy()
temporal_net_ttsim.bn.params[2][1].data = temporal_net_pytorch.bn.running_mean.numpy()
temporal_net_ttsim.bn.params[3][1].data = temporal_net_pytorch.bn.running_var.numpy()

# Residual blocks
for i, (block_ttsim, block_pytorch) in enumerate(
    zip(temporal_net_ttsim.res_blocks, temporal_net_pytorch.res_layer)
):
    # Conv1
    block_ttsim.conv1.params[0][1].data = block_pytorch.conv1.weight.data.numpy()
    # BN1
    block_ttsim.bn1.params[0][1].data = block_pytorch.bn1.weight.data.numpy()
    block_ttsim.bn1.params[1][1].data = block_pytorch.bn1.bias.data.numpy()
    block_ttsim.bn1.params[2][1].data = block_pytorch.bn1.running_mean.numpy()
    block_ttsim.bn1.params[3][1].data = block_pytorch.bn1.running_var.numpy()
    # Conv2
    block_ttsim.conv2.params[0][1].data = block_pytorch.conv2.weight.data.numpy()
    # BN2
    block_ttsim.bn2.params[0][1].data = block_pytorch.bn2.weight.data.numpy()
    block_ttsim.bn2.params[1][1].data = block_pytorch.bn2.bias.data.numpy()
    block_ttsim.bn2.params[2][1].data = block_pytorch.bn2.running_mean.numpy()
    block_ttsim.bn2.params[3][1].data = block_pytorch.bn2.running_var.numpy()

print("  Injected weights from PyTorch to ttsim")

# Create input SimTensors with data
history_feats_simtensor = F._from_data(
    "history_feats", data=history_feats_np, is_const=False
)
curr_feat_simtensor = F._from_data("curr_feat", data=curr_feat_np, is_const=False)
output_ttsim = temporal_net_ttsim(history_feats_simtensor, curr_feat_simtensor)

print(f"  History features shape: {history_feats_simtensor.shape}")
print(f"  Current feature shape: {curr_feat_simtensor.shape}")
print(f"  Output shape: {output_ttsim.shape}")
print(f"  Output .data is None? {output_ttsim.data is None}")
print(f"  Output .data: {output_ttsim.data}")

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
    # Relaxed tolerance for accumulated floating-point errors through multiple layers
    atol = 3e-5
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
    print("  [WARN]  [SKIP] Cannot compare - ttsim output.data is None")

print()
print("=" * 70)
print("Test Complete!")
print("=" * 70)
