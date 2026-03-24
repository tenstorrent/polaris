#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test: DETR Losses (PyTorch vs ttsim)
Tests LinesL1Loss, MasksLoss, and LenLoss
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
from workloads.MapTracker.plugin.models.losses.detr_loss import (
    LinesL1Loss as LinesL1LossTtsim,
    MasksLoss as MasksLossTtsim,
    LenLoss as LenLossTtsim,
)

# ============================================================
# PyTorch Reference Implementations
# ============================================================


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss."""
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss


def l1_loss(pred, target, weight=None, reduction="mean", avg_factor=None):
    """L1 loss."""
    loss = F_torch.l1_loss(pred, target, reduction="none")
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def smooth_l1_loss(
    pred, target, weight=None, beta=1.0, reduction="mean", avg_factor=None
):
    """Smooth L1 loss."""
    assert beta > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LinesL1LossPyTorch(nn.Module):
    """L1/Smooth L1 loss for lines (PyTorch reference)"""

    def __init__(self, reduction="mean", loss_weight=1.0, beta=0.5):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(self, pred, target, weight=None, avg_factor=None):
        if self.beta > 0:
            loss = smooth_l1_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=self.reduction,
                avg_factor=avg_factor,
            )
        else:
            loss = l1_loss(
                pred, target, weight, reduction=self.reduction, avg_factor=avg_factor
            )

        num_points = pred.shape[-1] // 2
        loss = loss / num_points
        return loss * self.loss_weight


class MasksLossPyTorch(nn.Module):
    """Binary Cross Entropy loss (PyTorch reference)"""

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = F_torch.binary_cross_entropy_with_logits(
            pred, target.float(), reduction="none"
        )

        if weight is not None:
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_weight


class LenLossPyTorch(nn.Module):
    """Cross Entropy loss (PyTorch reference)"""

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None):
        loss = F_torch.cross_entropy(pred, target, reduction="none")

        if weight is not None:
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss * self.loss_weight


# ============================================================
# Comparison Tests
# ============================================================


def test_lines_l1_loss_smooth():
    """Test LinesL1Loss with Smooth L1: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 1: LinesL1Loss (Smooth L1, beta=0.5)")
    print("=" * 80)

    # Create test data (line coordinates)
    B, N, coords = 4, 50, 40  # batch, num_lines, coordinates (20 points * 2)
    np.random.seed(44)
    pred_np = np.random.randn(B, N, coords).astype(np.float32)
    target_np = np.random.randn(B, N, coords).astype(np.float32)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = LinesL1LossPyTorch(reduction="mean", loss_weight=1.0, beta=0.5)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch LinesL1Loss (Smooth): {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = LinesL1LossTtsim(reduction="mean", loss_weight=1.0, beta=0.5)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    loss_ttsim_value = (
        loss_ttsim.data.item()
        if hasattr(loss_ttsim.data, "item")
        else float(loss_ttsim.data)
    )
    print(f"ttsim LinesL1Loss (Smooth):   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: LinesL1Loss (Smooth) matches!")
    else:
        print("[FAIL] FAIL: LinesL1Loss (Smooth) mismatch!")

    return matches


def test_lines_l1_loss_standard():
    """Test LinesL1Loss with standard L1: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 2: LinesL1Loss (Standard L1, beta=0)")
    print("=" * 80)

    # Create test data
    B, N, coords = 4, 50, 40
    np.random.seed(45)
    pred_np = np.random.randn(B, N, coords).astype(np.float32)
    target_np = np.random.randn(B, N, coords).astype(np.float32)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = LinesL1LossPyTorch(reduction="mean", loss_weight=1.0, beta=0.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch LinesL1Loss (L1): {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = LinesL1LossTtsim(reduction="mean", loss_weight=1.0, beta=0.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    loss_ttsim_value = (
        loss_ttsim.data.item()
        if hasattr(loss_ttsim.data, "item")
        else float(loss_ttsim.data)
    )
    print(f"ttsim LinesL1Loss (L1):   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: LinesL1Loss (L1) matches!")
    else:
        print("[FAIL] FAIL: LinesL1Loss (L1) mismatch!")

    return matches


def test_masks_loss():
    """Test MasksLoss (BCE): PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 3: MasksLoss (BCE)")
    print("=" * 80)

    # Create test data
    B, nquery, npts = 4, 50, 20
    np.random.seed(46)
    pred_np = np.random.randn(B, nquery, npts).astype(np.float32)
    target_np = np.random.randint(0, 2, (B, nquery, npts)).astype(np.float32)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = MasksLossPyTorch(reduction="mean", loss_weight=1.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch MasksLoss: {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = MasksLossTtsim(reduction="mean", loss_weight=1.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    loss_ttsim_value = (
        loss_ttsim.data.item()
        if hasattr(loss_ttsim.data, "item")
        else float(loss_ttsim.data)
    )
    print(f"ttsim MasksLoss:   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: MasksLoss matches!")
    else:
        print("[FAIL] FAIL: MasksLoss mismatch!")

    return matches


def test_len_loss():
    """Test LenLoss (Cross Entropy): PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 4: LenLoss (Cross Entropy)")
    print("=" * 80)

    # Create test data
    batch_size = 100
    num_classes = 20
    np.random.seed(47)
    pred_np = np.random.randn(batch_size, num_classes).astype(np.float32)
    target_np = np.random.randint(0, num_classes, (batch_size,)).astype(np.int64)

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)

    loss_fn_pytorch = LenLossPyTorch(reduction="mean", loss_weight=1.0)
    loss_pytorch = loss_fn_pytorch(pred_torch, target_torch)
    print(f"PyTorch LenLoss: {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)

    loss_fn_ttsim = LenLossTtsim(reduction="mean", loss_weight=1.0)
    loss_ttsim = loss_fn_ttsim(pred_ttsim, target_ttsim)

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    loss_ttsim_value = (
        loss_ttsim.data.item()
        if hasattr(loss_ttsim.data, "item")
        else float(loss_ttsim.data)
    )
    print(f"ttsim LenLoss:   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: LenLoss matches!")
    else:
        print("[FAIL] FAIL: LenLoss mismatch!")

    return matches


def test_weighted_reduction():
    """Test weighted loss reduction with avg_factor"""
    print("\n" + "=" * 80)
    print("TEST 5: Weighted Reduction with avg_factor")
    print("=" * 80)

    # Create test data
    B, N, coords = 4, 50, 40
    np.random.seed(48)
    pred_np = np.random.randn(B, N, coords).astype(np.float32)
    target_np = np.random.randn(B, N, coords).astype(np.float32)
    weight_np = np.random.rand(B, N, coords).astype(np.float32)
    avg_factor = 100.0

    # PyTorch version
    pred_torch = torch.from_numpy(pred_np)
    target_torch = torch.from_numpy(target_np)
    weight_torch = torch.from_numpy(weight_np)

    loss_fn_pytorch = LinesL1LossPyTorch(reduction="mean", loss_weight=1.0, beta=0.5)
    loss_pytorch = loss_fn_pytorch(
        pred_torch, target_torch, weight=weight_torch, avg_factor=avg_factor
    )
    print(f"PyTorch Weighted Loss: {loss_pytorch.item():.6f}")

    # ttsim version
    pred_ttsim = F._from_data("pred", pred_np)
    target_ttsim = F._from_data("target", target_np)
    weight_ttsim = F._from_data("weight", weight_np)

    loss_fn_ttsim = LinesL1LossTtsim(reduction="mean", loss_weight=1.0, beta=0.5)
    loss_ttsim = loss_fn_ttsim(
        pred_ttsim, target_ttsim, weight=weight_ttsim, avg_factor=avg_factor
    )

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    loss_ttsim_value = (
        loss_ttsim.data.item()
        if hasattr(loss_ttsim.data, "item")
        else float(loss_ttsim.data)
    )
    print(f"ttsim Weighted Loss:   {loss_ttsim_value:.6f}")

    # Compare using np.allclose
    matches = np.allclose(loss_pytorch.item(), loss_ttsim_value, rtol=1e-5, atol=1e-4)
    diff = abs(loss_pytorch.item() - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: Weighted reduction matches!")
    else:
        print("[FAIL] FAIL: Weighted reduction mismatch!")

    return matches


def main():
    """Run all DETR loss comparison tests"""
    print("\n" + "=" * 80)
    print("DETR Losses Comparison: PyTorch vs ttsim")
    print("=" * 80)

    results = []

    # Run all tests
    results.append(("LinesL1Loss (Smooth)", test_lines_l1_loss_smooth()))
    results.append(("LinesL1Loss (L1)", test_lines_l1_loss_standard()))
    results.append(("MasksLoss (BCE)", test_masks_loss()))
    results.append(("LenLoss (CE)", test_len_loss()))
    results.append(("Weighted Reduction", test_weighted_reduction()))

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
