#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for DiffMotionPlanningRefinementModule.
Tests shape inference and numerical equivalence between PyTorch and TTSIM.
"""

import os
import sys

# Add DiffusionDrive to path for navsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# Add TT-Polaris to path for ttsim imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../.."))

import numpy as np
import torch
import torch.nn as nn

from navsim.agents.diffusiondrive.reference.torch_code.navsim.agents.diffusiondrive.transfuser_model_v2 import (
    DiffMotionPlanningRefinementModule as DiffMotion_PyTorch,
)
from navsim.agents.diffusiondrive.transfuser_model_v2_ttsim import (
    DiffMotionPlanningRefinementModule as DiffMotion_TTSIM,
)

import ttsim.front.functional.op as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def inject_weights(ttsim_module, pytorch_module):
    """Inject PyTorch weights into TTSIM DiffMotionPlanningRefinementModule."""
    # Plan classification branch
    ttsim_module.plan_cls_linear1.params[0][1].data = pytorch_module.plan_cls_branch[
        0
    ].weight.data.T.numpy()
    ttsim_module.plan_cls_bias1.params[0][1].data = pytorch_module.plan_cls_branch[
        0
    ].bias.data.numpy()
    ttsim_module.plan_cls_ln1.params[0][1].data = pytorch_module.plan_cls_branch[
        2
    ].weight.data.numpy()
    ttsim_module.plan_cls_ln1.params[1][1].data = pytorch_module.plan_cls_branch[
        2
    ].bias.data.numpy()
    ttsim_module.plan_cls_linear2.params[0][1].data = pytorch_module.plan_cls_branch[
        3
    ].weight.data.T.numpy()
    ttsim_module.plan_cls_bias2.params[0][1].data = pytorch_module.plan_cls_branch[
        3
    ].bias.data.numpy()
    ttsim_module.plan_cls_ln2.params[0][1].data = pytorch_module.plan_cls_branch[
        5
    ].weight.data.numpy()
    ttsim_module.plan_cls_ln2.params[1][1].data = pytorch_module.plan_cls_branch[
        5
    ].bias.data.numpy()
    ttsim_module.plan_cls_linear3.params[0][1].data = pytorch_module.plan_cls_branch[
        6
    ].weight.data.T.numpy()
    ttsim_module.plan_cls_bias3.params[0][1].data = pytorch_module.plan_cls_branch[
        6
    ].bias.data.numpy()

    # Plan regression branch
    ttsim_module.plan_reg_linear1.params[0][1].data = pytorch_module.plan_reg_branch[
        0
    ].weight.data.T.numpy()
    ttsim_module.plan_reg_bias1.params[0][1].data = pytorch_module.plan_reg_branch[
        0
    ].bias.data.numpy()
    ttsim_module.plan_reg_linear2.params[0][1].data = pytorch_module.plan_reg_branch[
        2
    ].weight.data.T.numpy()
    ttsim_module.plan_reg_bias2.params[0][1].data = pytorch_module.plan_reg_branch[
        2
    ].bias.data.numpy()
    ttsim_module.plan_reg_linear3.params[0][1].data = pytorch_module.plan_reg_branch[
        4
    ].weight.data.T.numpy()
    ttsim_module.plan_reg_bias3.params[0][1].data = pytorch_module.plan_reg_branch[
        4
    ].bias.data.numpy()


def main():
    print("=" * 70)
    print("DiffMotionPlanningRefinementModule Validation: PyTorch vs TTSIM")
    print("=" * 70)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    embed_dims = 256
    ego_fut_ts = 8
    ego_fut_mode = 20
    batch_size = 2

    # Create PyTorch model
    print("\n--- PyTorch DiffMotionPlanningRefinementModule ---")
    model_pt = DiffMotion_PyTorch(
        embed_dims=embed_dims,
        ego_fut_ts=ego_fut_ts,
        ego_fut_mode=ego_fut_mode,
        if_zeroinit_reg=False,
    )
    model_pt.eval()

    # Generate random input
    traj_feature_data = np.random.randn(batch_size, ego_fut_mode, embed_dims).astype(
        np.float32
    )
    traj_feature_pt = torch.from_numpy(traj_feature_data)

    # Forward pass
    with torch.no_grad():
        plan_reg_pt, plan_cls_pt = model_pt(traj_feature_pt)

    plan_reg_pt_np = plan_reg_pt.numpy()
    plan_cls_pt_np = plan_cls_pt.numpy()

    print(f"Trajectory feature shape: {traj_feature_pt.shape}")
    print(f"Plan regression shape: {plan_reg_pt_np.shape}")
    print(f"Plan classification shape: {plan_cls_pt_np.shape}")
    print(
        f"Plan reg stats: min={plan_reg_pt_np.min():.6f}, max={plan_reg_pt_np.max():.6f}, mean={plan_reg_pt_np.mean():.6f}"
    )
    print(
        f"Plan cls stats: min={plan_cls_pt_np.min():.6f}, max={plan_cls_pt_np.max():.6f}, mean={plan_cls_pt_np.mean():.6f}"
    )

    # TTSIM model
    print("\n--- TTSIM DiffMotionPlanningRefinementModule ---")
    model_ttsim = DiffMotion_TTSIM(
        embed_dims=embed_dims,
        ego_fut_ts=ego_fut_ts,
        ego_fut_mode=ego_fut_mode,
        if_zeroinit_reg=False,
    )

    # Inject weights
    print("Injecting weights...")
    inject_weights(model_ttsim, model_pt)

    # Forward pass
    traj_feature_ttsim = F._from_data("traj_feature", traj_feature_data)
    traj_feature_ttsim.link_module = model_ttsim
    plan_reg_ttsim, plan_cls_ttsim = model_ttsim(traj_feature_ttsim)

    print(f"Trajectory feature shape: {traj_feature_ttsim.shape}")
    print(f"Plan regression shape: {plan_reg_ttsim.shape}")
    print(f"Plan classification shape: {plan_cls_ttsim.shape}")

    # Check if data is available
    if plan_reg_ttsim.data is not None and plan_cls_ttsim.data is not None:
        print(
            f"Plan reg stats: min={plan_reg_ttsim.data.min():.6f}, max={plan_reg_ttsim.data.max():.6f}, mean={plan_reg_ttsim.data.mean():.6f}"
        )
        print(
            f"Plan cls stats: min={plan_cls_ttsim.data.min():.6f}, max={plan_cls_ttsim.data.max():.6f}, mean={plan_cls_ttsim.data.mean():.6f}"
        )

        # Numerical comparison
        print("\n--- Numerical Comparison: PyTorch vs TTSIM ---")

        # Note: TTSIM plan_cls has shape [2, 20, 1] due to squeeze breaking .data computation
        # Expand PyTorch output to match for comparison
        plan_cls_pt_expanded = np.expand_dims(plan_cls_pt_np, axis=-1)

        atol = 1e-4
        rtol = 1e-4

        print(f"Tolerance: atol={atol}, rtol={rtol}")

        # Compare plan regression
        diff_reg = np.abs(plan_reg_pt_np - plan_reg_ttsim.data)
        max_diff_reg = np.max(diff_reg)
        mean_diff_reg = np.mean(diff_reg)

        print(f"\nPlan Regression:")
        print(f"  Max absolute difference: {max_diff_reg:.10f}")
        print(f"  Mean absolute difference: {mean_diff_reg:.10f}")

        is_close_reg = np.allclose(
            plan_reg_pt_np, plan_reg_ttsim.data, atol=atol, rtol=rtol
        )

        if is_close_reg:
            print(f"  PASS: TTSIM matches PyTorch for plan regression")
        else:
            print(f"  FAIL: Differences exceed tolerance for plan regression")

        # Compare plan classification
        diff_cls = np.abs(plan_cls_pt_np - plan_cls_ttsim.data)
        max_diff_cls = np.max(diff_cls)
        mean_diff_cls = np.mean(diff_cls)

        print(f"\nPlan Classification:")
        print(f"  Max absolute difference: {max_diff_cls:.10f}")
        print(f"  Mean absolute difference: {mean_diff_cls:.10f}")

        is_close_cls = np.allclose(
            plan_cls_pt_np, plan_cls_ttsim.data, atol=atol, rtol=rtol
        )

        if is_close_cls:
            print(f"  PASS: TTSIM matches PyTorch for plan classification")
        else:
            print(f"  FAIL: Differences exceed tolerance for plan classification")

        # Overall result
        print("\n" + "=" * 70)
        if is_close_reg and is_close_cls:
            print("OVERALL: PASS - All outputs match")
        else:
            print("OVERALL: FAIL - Some outputs don't match")
        print("=" * 70)
    else:
        print("  WARN: SKIPPED: No TTSIM data available")

    print()


if __name__ == "__main__":
    main()
