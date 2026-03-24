#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test: Segmentation Losses (PyTorch vs ttsim)
Tests MaskFocalLoss and MaskDiceLoss
"""

import os, sys

# Add polaris directory to path
polaris_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_path not in sys.path:
    sys.path.insert(0, polaris_path)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch

import ttsim.front.functional.op as F

# Import ttsim versions
from workloads.MapTracker.plugin.models.losses.seg_loss import (
    MaskFocalLoss as MaskFocalLossTtsim,
    MaskDiceLoss as MaskDiceLossTtsim,
)

# ============================================================
# PyTorch Reference Implementations
# ============================================================


class MaskFocalLossPyTorch(nn.Module):
    """Focal Loss for segmentation (PyTorch reference)"""

    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(
            self.gamma
        )
        loss = (
            F_torch.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        return loss.mean() * self.loss_weight


class MaskDiceLossPyTorch(nn.Module):
    """Dice Loss for segmentation (PyTorch reference)"""

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.smooth = 1e-5
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        bs, num_classes, h, w = pred.shape
        pred = pred.view(bs, num_classes, h * w)
        target = target.view(bs, num_classes, h * w)
        pred = pred.sigmoid()
        intersection = torch.sum(pred * target, dim=2)
        union = torch.sum(pred.pow(2), dim=2) + torch.sum(target, dim=2)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - torch.mean(dice_coef)
        return dice_loss * self.loss_weight


# ============================================================
# Comparison Tests
# ============================================================


def test_mask_focal_loss():
    """Test MaskFocalLoss: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 1: MaskFocalLoss")
    print("=" * 80)

    # Create test data
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(42)
    pred_np = np.random.randn(B, C, H, W).astype(np.float32)
    target_np = np.random.randint(0, 2, (B, C, H, W)).astype(np.float32)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = MaskFocalLossPyTorch(gamma=2.0, alpha=0.25, loss_weight=1.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch MaskFocalLoss: {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = MaskFocalLossTtsim(gamma=2.0, alpha=0.25, loss_weight=1.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    # Handle potential array output
    if isinstance(loss_ttsim.data, np.ndarray):
        if loss_ttsim.data.size == 1:
            loss_ttsim_value = float(loss_ttsim.data.flatten()[0])
        else:
            print(
                f"[WARN]  ttsim output shape: {loss_ttsim.data.shape}, expected scalar"
            )
            loss_ttsim_value = float(loss_ttsim.data.mean())
    else:
        loss_ttsim_value = float(loss_ttsim.data)
    print(f"ttsim MaskFocalLoss:   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: MaskFocalLoss matches!")
    else:
        print("[FAIL] FAIL: MaskFocalLoss mismatch!")

    return matches


def test_mask_dice_loss():
    """Test MaskDiceLoss: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 2: MaskDiceLoss")
    print("=" * 80)

    # Create test data
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(43)
    pred_np = np.random.randn(B, C, H, W).astype(np.float32)
    target_np = np.random.randint(0, 2, (B, C, H, W)).astype(np.float32)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = MaskDiceLossPyTorch(loss_weight=1.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch MaskDiceLoss: {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = MaskDiceLossTtsim(loss_weight=1.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    # Handle potential array output
    if isinstance(loss_ttsim.data, np.ndarray):
        if loss_ttsim.data.size == 1:
            loss_ttsim_value = float(loss_ttsim.data.flatten()[0])
        else:
            print(
                f"[WARN]  ttsim output shape: {loss_ttsim.data.shape}, expected scalar"
            )
            loss_ttsim_value = float(loss_ttsim.data.mean())
    else:
        loss_ttsim_value = float(loss_ttsim.data)
    print(f"ttsim MaskDiceLoss:   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: MaskDiceLoss matches!")
    else:
        print("[FAIL] FAIL: MaskDiceLoss mismatch!")

    return matches


def test_perfect_prediction():
    """Test edge case: perfect prediction (loss should be near zero)"""
    print("\n" + "=" * 80)
    print("TEST 3: Perfect Prediction (Edge Case)")
    print("=" * 80)

    # Create perfect prediction
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(49)
    target_np = np.random.randint(0, 2, (B, C, H, W)).astype(np.float32)
    # Logits that will give high probability for correct class
    pred_np = target_np * 10.0 - (1 - target_np) * 10.0

    # PyTorch Dice Loss
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = MaskDiceLossPyTorch(loss_weight=1.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch DiceLoss (perfect): {loss_pytorch.item():.6f}")

    # ttsim Dice Loss
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = MaskDiceLossTtsim(loss_weight=1.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    # Handle potential array output
    if isinstance(loss_ttsim.data, np.ndarray):
        if loss_ttsim.data.size == 1:
            loss_ttsim_value = float(loss_ttsim.data.flatten()[0])
        else:
            print(
                f"[WARN]  ttsim output shape: {loss_ttsim.data.shape}, expected scalar"
            )
            loss_ttsim_value = float(loss_ttsim.data.mean())
    else:
        loss_ttsim_value = float(loss_ttsim.data)
    print(f"ttsim DiceLoss (perfect):   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    # Both should be very small (near 0) and match
    if loss_pytorch.item() < 0.1 and loss_ttsim_value < 0.1 and matches:
        print("[PASS] PASS: Perfect prediction gives low loss!")
    else:
        print("[FAIL] FAIL: Perfect prediction edge case failed!")

    return loss_pytorch.item() < 0.1 and loss_ttsim_value < 0.1 and matches


def main():
    """Run all segmentation loss comparison tests"""
    print("\n" + "=" * 80)
    print("Segmentation Losses Comparison: PyTorch vs ttsim")
    print("=" * 80)

    results = []

    # Run all tests
    results.append(("MaskFocalLoss", test_mask_focal_loss()))
    results.append(("MaskDiceLoss", test_mask_dice_loss()))
    results.append(("Perfect Prediction", test_perfect_prediction()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {name}")

    total_pass = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")

    if total_pass == len(results):
        print(
            "\n[PASS] All tests passed! PyTorch and ttsim implementations match perfectly!"
        )
    else:
        print(f"\n[WARN]  {len(results) - total_pass} test(s) failed. Please review.")


if __name__ == "__main__":
    main()
