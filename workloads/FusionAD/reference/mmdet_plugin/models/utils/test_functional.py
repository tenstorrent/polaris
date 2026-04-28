#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
FusionAD functional.py validation: TTSim vs PyTorch comparison.

Tests all 6 utility functions/classes:
  1. bivariate_gaussian_activation  /  BivariateGaussianActivation
  2. norm_points                    /  NormPoints
  3. pos2posemb2d                   /  Pos2PosEmb2D
  4. rot_2d                         /  Rot2D
  5. anchor_coordinate_transform    /  AnchorCoordinateTransform
  6. trajectory_coordinate_transform / TrajectoryCoordinateTransform

"""

import os
import sys
import math

# Add polaris root to sys.path for workloads imports
# Script location: polaris/workloads/FusionAD/Comparison/mmdet_plugin/models/utils/
polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
sys.path.insert(0, polaris_path)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from einops import rearrange

import ttsim.front.functional.op as F
from ttsim.ops import SimTensor

# Import TTSim classes
from workloads.FusionAD.projects.mmdet_plugin.models.utils.functional import (
    BivariateGaussianActivation,
    NormPoints,
    Pos2PosEmb2D,
    Rot2D,
    AnchorCoordinateTransform,
    TrajectoryCoordinateTransform,
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PyTorch reference implementations (inlined to avoid mmdet3d dependency)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def bivariate_gaussian_activation_pt(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def norm_points_pt(pos, pc_range):
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d_pt(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def rot_2d_pt(yaw):
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack(
        [torch.stack([cy, -sy]), torch.stack([sy, cy])]
    ).permute([2, 0, 1])
    return out


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def create_input_tensor(name: str, data: np.ndarray) -> SimTensor:
    """Create a TTSim SimTensor from numpy data."""
    return F._from_data(name, data.astype(np.float32), is_param=False, is_const=False)


def compare_outputs(name, pt_out, tt_data, atol=1e-6, rtol=1e-5):
    """
    Numerically compare PyTorch and TTSim outputs.
    Returns True if outputs match within tolerance.
    """
    print(f"\n{'-' * 60}")
    print(f"  Comparison: {name}")
    print(f"{'-' * 60}")

    if tt_data is None:
        print("  [FAIL] TTSim output data is None.")
        return False

    print(f"  PyTorch shape: {pt_out.shape}")
    print(f"  TTSim  shape:  {tt_data.shape}")

    if pt_out.shape != tt_data.shape:
        print(f"  [FAIL] Shape mismatch! PT {pt_out.shape} vs TT {tt_data.shape}")
        return False

    is_close = np.allclose(tt_data, pt_out, atol=atol, rtol=rtol)
    diff = np.abs(tt_data - pt_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(f"  Tolerance:     atol={atol}, rtol={rtol}")
    print(f"  Max abs diff:  {max_diff:.10f}")
    print(f"  Mean abs diff: {mean_diff:.10f}")
    print(f"  allclose:      {is_close}")

    if is_close:
        print("  [PASS] TTSim matches PyTorch")
    else:
        print(f"  [FAIL] Differences exceed tolerance (max diff: {max_diff:.10f})")
    return is_close


# Main Test

PI_CONST = 3.1415953  # matches FusionAD source

np.random.seed(42)
torch.manual_seed(42)

atol = 1e-6
rtol = 1e-5
results = {}

print("=" * 70)
print("FusionAD functional.py - TTSim vs PyTorch Comparison")
print("=" * 70)



# TEST 1: bivariate_gaussian_activation

print("\n\n" + "=" * 70)
print("TEST 1: bivariate_gaussian_activation")
print("=" * 70)

input_data = np.random.randn(4, 8, 5).astype(np.float32)
print(f"Input shape: {input_data.shape}")
print(f"Input stats: min={input_data.min():.6f}, max={input_data.max():.6f}, "
      f"mean={input_data.mean():.6f}")

# PyTorch
with torch.no_grad():
    pt_input = torch.from_numpy(input_data)
    pt_out = bivariate_gaussian_activation_pt(pt_input).cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")
print(f"  PyTorch output stats: min={pt_out.min():.6f}, max={pt_out.max():.6f}, "
      f"mean={pt_out.mean():.6f}")

# TTSim
tt_module = BivariateGaussianActivation('test_bga')
tt_input = create_input_tensor('bga_input', input_data)
tt_out = tt_module(tt_input)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")
    print(f"  TTSim  output stats: min={tt_out.data.min():.6f}, max={tt_out.data.max():.6f}, "
          f"mean={tt_out.data.mean():.6f}")

results['bivariate_gaussian_activation'] = compare_outputs(
    'bivariate_gaussian_activation', pt_out, tt_out.data, atol, rtol
)



# TEST 2: norm_points

print("\n\n" + "=" * 70)
print("TEST 2: norm_points")
print("=" * 70)

pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
pos_data = np.random.randn(6, 10, 2).astype(np.float32) * 50  # positions in plausible range
print(f"Input shape: {pos_data.shape}")
print(f"pc_range:    {pc_range}")
print(f"Input stats: min={pos_data.min():.6f}, max={pos_data.max():.6f}, "
      f"mean={pos_data.mean():.6f}")

# PyTorch
with torch.no_grad():
    pt_pos = torch.from_numpy(pos_data)
    pt_out = norm_points_pt(pt_pos, pc_range).cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")
print(f"  PyTorch output stats: min={pt_out.min():.6f}, max={pt_out.max():.6f}, "
      f"mean={pt_out.mean():.6f}")

# TTSim
tt_module = NormPoints('test_norm', pc_range)
tt_input = create_input_tensor('norm_input', pos_data)
tt_out = tt_module(tt_input)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")
    print(f"  TTSim  output stats: min={tt_out.data.min():.6f}, max={tt_out.data.max():.6f}, "
          f"mean={tt_out.data.mean():.6f}")

results['norm_points'] = compare_outputs(
    'norm_points', pt_out, tt_out.data, atol, rtol
)


# 
# TEST 3: pos2posemb2d
# 
print("\n\n" + "=" * 70)
print("TEST 3: pos2posemb2d")
print("=" * 70)

num_pos_feats = 64   # smaller for faster testing
temperature = 10000
pos_data = np.random.randn(3, 5, 2).astype(np.float32)
print(f"Input shape:   {pos_data.shape}")
print(f"num_pos_feats: {num_pos_feats}")
print(f"temperature:   {temperature}")
print(f"Input stats: min={pos_data.min():.6f}, max={pos_data.max():.6f}, "
      f"mean={pos_data.mean():.6f}")

# PyTorch
with torch.no_grad():
    pt_pos = torch.from_numpy(pos_data)
    pt_out = pos2posemb2d_pt(pt_pos, num_pos_feats, temperature).cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")
print(f"  PyTorch output stats: min={pt_out.min():.6f}, max={pt_out.max():.6f}, "
      f"mean={pt_out.mean():.6f}")

# TTSim
tt_module = Pos2PosEmb2D('test_posemb', num_pos_feats=num_pos_feats, temperature=temperature)
tt_input = create_input_tensor('posemb_input', pos_data)
tt_out = tt_module(tt_input)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")
    print(f"  TTSim  output stats: min={tt_out.data.min():.6f}, max={tt_out.data.max():.6f}, "
          f"mean={tt_out.data.mean():.6f}")

# NOTE: The TTSim Pos2PosEmb2D concatenates [sin, cos] halves while
#       PyTorch interleaves them via stack+flatten.  If this test fails,
#       the interleaving order is the likely cause and the conversion
#       should be updated to match the interleaved layout.
results['pos2posemb2d'] = compare_outputs(
    'pos2posemb2d', pt_out,
    tt_out.data if tt_out.data is not None else None,
    atol, rtol
)


# 
# TEST 4: rot_2d
# 
print("\n\n" + "=" * 70)
print("TEST 4: rot_2d")
print("=" * 70)

yaw_data = np.random.randn(8).astype(np.float32)
print(f"Input shape: {yaw_data.shape}")
print(f"Input stats: min={yaw_data.min():.6f}, max={yaw_data.max():.6f}, "
      f"mean={yaw_data.mean():.6f}")

# PyTorch: rot_2d takes [N], returns [N, 2, 2]
with torch.no_grad():
    pt_yaw = torch.from_numpy(yaw_data)
    pt_out = rot_2d_pt(pt_yaw).cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")

# TTSim: Rot2D expects [..., 1] shaped input
tt_module = Rot2D('test_rot2d')
yaw_data_reshaped = yaw_data.reshape(-1, 1)
tt_input = create_input_tensor('rot2d_input', yaw_data_reshaped)
tt_out = tt_module(tt_input)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")

    # Handle potential shape mismatch â€” TTSim may produce [N, 4] or [2N, 2]
    # instead of [N, 2, 2] depending on ConcatX axis=-2 behavior.
    tt_data = tt_out.data
    if tt_data.shape != pt_out.shape:
        print(f"  Note: Shape mismatch, attempting reshape {tt_data.shape} -> {pt_out.shape}")
        try:
            tt_data = tt_data.reshape(pt_out.shape)
        except ValueError:
            print(f"  [FAIL] Cannot reshape TTSim output {tt_out.data.shape} to {pt_out.shape}")
            results['rot_2d'] = False
            tt_data = None

    if tt_data is not None:
        results['rot_2d'] = compare_outputs('rot_2d', pt_out, tt_data, atol, rtol)
else:
    print("  [FAIL] TTSim output data is None.")
    results['rot_2d'] = False


# 
# TEST 5: anchor_coordinate_transform
# 
print("\n\n" + "=" * 70)
print("TEST 5: anchor_coordinate_transform")
print("=" * 70)

num_agents = 3
num_groups = 2
num_modes = 4
num_steps = 12

anchors_data = np.random.randn(num_groups, num_modes, num_steps, 2).astype(np.float32)
yaw_data = np.random.randn(num_agents, 1).astype(np.float32)
centers_data = np.random.randn(num_agents, 3).astype(np.float32)

print(f"Anchors shape:  {anchors_data.shape}  (G, M, T, 2)")
print(f"Yaw shape:      {yaw_data.shape}       (A, 1)")
print(f"Centers shape:  {centers_data.shape}    (A, 3)")

# PyTorch reference (manual â€” avoids bbox_results mock)
with torch.no_grad():
    anchors_pt = torch.from_numpy(anchors_data)
    yaw_pt = torch.from_numpy(yaw_data)
    centers_pt = torch.from_numpy(centers_data)

    transformed = anchors_pt[None, ...]                     # [1, G, M, T, 2]
    angle = yaw_pt - PI_CONST                               # [A, 1]
    rot_mat = rot_2d_pt(angle.squeeze(-1))                  # [A, 2, 2]
    rot_mat = rot_mat[:, None, None, :, :]                  # [A, 1, 1, 2, 2]
    transformed = rearrange(transformed, 'b g m t c -> b g m c t')
    transformed = torch.matmul(rot_mat, transformed)        # [A, G, M, 2, T]
    transformed = rearrange(transformed, 'b g m c t -> b g m t c')
    transformed = centers_pt[:, None, None, None, :2] + transformed
    pt_out = transformed.cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")

# TTSim
tt_module = AnchorCoordinateTransform('test_anchor_ct')
tt_anchors = create_input_tensor('anchor_in', anchors_data[None, ...])   # [1, G, M, T, 2]
tt_yaw = create_input_tensor('anchor_yaw', yaw_data)
# Pass centers pre-shaped with broadcasting dims, [:2] for xy only
tt_centers = create_input_tensor('anchor_centers', centers_data[:, None, None, None, :2])
tt_out = tt_module(tt_anchors, tt_yaw, tt_centers,
                   with_rotation_transform=True, with_translation_transform=True)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")

    tt_data = tt_out.data
    if tt_data.shape != pt_out.shape:
        print(f"  Note: Shape mismatch, attempting reshape {tt_data.shape} -> {pt_out.shape}")
        try:
            tt_data = tt_data.reshape(pt_out.shape)
        except ValueError:
            print(f"  [FAIL] Cannot reshape TTSim output {tt_out.data.shape} to {pt_out.shape}")
            results['anchor_coordinate_transform'] = False
            tt_data = None

    if tt_data is not None:
        results['anchor_coordinate_transform'] = compare_outputs(
            'anchor_coordinate_transform', pt_out, tt_data, atol, rtol
        )
else:
    print("  [FAIL] TTSim output data is None.")
    results['anchor_coordinate_transform'] = False



# TEST 6: trajectory_coordinate_transform

print("\n\n" + "=" * 70)
print("TEST 6: trajectory_coordinate_transform")
print("=" * 70)

num_agents = 3
num_groups = 2
num_proposals = 4
num_steps = 12

traj_data = np.random.randn(num_agents, num_groups, num_proposals, num_steps, 2).astype(np.float32)
yaw_data = np.random.randn(num_agents, 1).astype(np.float32)
centers_data = np.random.randn(num_agents, 3).astype(np.float32)

print(f"Trajectory shape: {traj_data.shape}  (A, G, P, T, 2)")
print(f"Yaw shape:        {yaw_data.shape}       (A, 1)")
print(f"Centers shape:    {centers_data.shape}    (A, 3)")

# PyTorch reference (manual â€” avoids bbox_results mock)
with torch.no_grad():
    traj_pt = torch.from_numpy(traj_data)
    yaw_pt = torch.from_numpy(yaw_data)
    centers_pt = torch.from_numpy(centers_data)

    transformed = traj_pt.clone()
    angle = -(yaw_pt - PI_CONST)                             # negate for inverse rotation
    rot_mat = rot_2d_pt(angle.squeeze(-1))                   # [A, 2, 2]
    rot_mat = rot_mat[:, None, None, :, :]                   # [A, 1, 1, 2, 2]
    transformed = rearrange(transformed, 'a g p t c -> a g p c t')
    transformed = torch.matmul(rot_mat, transformed)         # [A, G, P, 2, T]
    transformed = rearrange(transformed, 'a g p c t -> a g p t c')
    transformed = centers_pt[:, None, None, None, :2] + transformed
    pt_out = transformed.cpu().numpy()

print(f"  PyTorch output shape: {pt_out.shape}")

# TTSim
tt_module = TrajectoryCoordinateTransform('test_traj_ct')
tt_traj = create_input_tensor('traj_in', traj_data)
tt_yaw = create_input_tensor('traj_yaw', yaw_data)
# Pass centers pre-shaped with broadcasting dims, [:2] for xy only
tt_centers = create_input_tensor('traj_centers', centers_data[:, None, None, None, :2])
tt_out = tt_module(tt_traj, tt_yaw, tt_centers,
                   with_rotation_transform=True, with_translation_transform=True)

if tt_out.data is not None:
    print(f"  TTSim  output shape: {tt_out.data.shape}")

    tt_data = tt_out.data
    if tt_data.shape != pt_out.shape:
        print(f"  Note: Shape mismatch, attempting reshape {tt_data.shape} -> {pt_out.shape}")
        try:
            tt_data = tt_data.reshape(pt_out.shape)
        except ValueError:
            print(f"  [FAIL] Cannot reshape TTSim output {tt_out.data.shape} to {pt_out.shape}")
            results['trajectory_coordinate_transform'] = False
            tt_data = None

    if tt_data is not None:
        results['trajectory_coordinate_transform'] = compare_outputs(
            'trajectory_coordinate_transform', pt_out, tt_data, atol, rtol
        )
else:
    print("  [FAIL] TTSim output data is None.")
    results['trajectory_coordinate_transform'] = False


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_pass = True
for name, passed in results.items():
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {name}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("All 6 tests PASSED!")
else:
    failed = sum(1 for v in results.values() if not v)
    passed = sum(1 for v in results.values() if v)
    print(f"{passed}/{len(results)} tests passed, {failed} FAILED.")

print("=" * 70)

