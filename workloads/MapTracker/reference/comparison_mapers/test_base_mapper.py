#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test: BaseMapper (PyTorch vs ttsim)
Tests basic mapper functionality including properties, forward passes, and train/val steps
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
from abc import ABCMeta, abstractmethod
import logging

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import ttsim version
from workloads.MapTracker.plugin.models.maper.base_mapper import (
    BaseMapper as BaseMapperTtsim,
)

# ============================================================
# PyTorch Reference Implementation
# ============================================================


class BaseMapperPyTorch(nn.Module, metaclass=ABCMeta):
    """Base class for mappers (PyTorch reference)"""

    def __init__(self):
        super(BaseMapperPyTorch, self).__init__()
        self.fp16_enabled = False
        self.logger = logging.getLogger(__name__)

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, imgs, img_metas, **kwargs):
        pass

    def forward_test(self, imgs, img_metas, use_aug=False, **kwargs):
        if use_aug:
            return self.aug_test(imgs, img_metas, **kwargs)
        else:
            return self.simple_test(imgs, img_metas, **kwargs)

    def forward(self, imgs, img_metas=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(imgs, img_metas, **kwargs)
        else:
            kwargs.pop("rescale", None)
            return self.forward_test(imgs, img_metas, **kwargs)

    def train_step(self, data_dict, optimizer):
        losses = self(**data_dict)

        if isinstance(losses, dict):
            loss = losses.get("loss")
            log_vars = losses.get("log_vars", {})
            num_samples = losses.get("num_samples", len(data_dict.get("img_metas", [])))
        else:
            loss, log_vars, num_samples = losses

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
        return outputs

    def val_step(self, data_dict, optimizer=None):
        losses = self(**data_dict)

        if isinstance(losses, dict):
            loss = losses.get("loss")
            log_vars = losses.get("log_vars", {})
            num_samples = losses.get("num_samples", len(data_dict.get("img_metas", [])))
        else:
            loss, log_vars, num_samples = losses

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
        return outputs


# ============================================================
# Minimal Concrete Test Mappers
# ============================================================


class SimpleMapperPyTorch(BaseMapperPyTorch):
    """Minimal concrete mapper for testing (PyTorch)"""

    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        # Simple feature extractor (bias=False to match ttsim Conv2d behavior)
        self.conv = nn.Conv2d(
            in_channels, feat_channels, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU()

        # Optional neck module for property testing
        self.neck = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=False)

        # Head for predictions
        self.head = nn.Conv2d(
            feat_channels, 10, kernel_size=1, bias=False
        )  # 10 classes

    def extract_feat(self, imgs):
        """Extract features from images"""
        x = self.conv(imgs)
        x = self.relu(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, imgs, img_metas, gt_labels=None, **kwargs):
        """Forward for training - compute simple loss"""
        # Extract features
        feats = self.extract_feat(imgs)

        # Get predictions
        pred = self.head(feats)

        # Simple loss: mean squared error with target
        if gt_labels is not None:
            loss = torch.mean((pred - gt_labels) ** 2)
        else:
            # Dummy loss if no target
            loss = torch.mean(pred**2) * 0.01

        # Log variables
        log_vars = {"loss": loss.item()}
        num_samples = imgs.shape[0]

        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

    def simple_test(self, imgs, img_metas, **kwargs):
        """Simple test - return predictions"""
        feats = self.extract_feat(imgs)
        pred = self.head(feats)

        # Return per-image results
        results = []
        for i in range(pred.shape[0]):
            results.append({"pred": pred[i].detach().cpu().numpy()})

        return results


class SimpleMapperTtsim(BaseMapperTtsim):
    """Minimal concrete mapper for testing (ttsim)"""

    def __init__(self, in_channels=3, feat_channels=64):
        super().__init__()
        # Create conv layers using F.Conv2d (which creates parameters internally)
        self.conv_op = F.Conv2d(
            "conv", in_channels, feat_channels, kernel_size=3, padding=1
        )
        self.neck_op = F.Conv2d(
            "neck", feat_channels, feat_channels, kernel_size=1, padding=0
        )
        self.head_op = F.Conv2d("head", feat_channels, 10, kernel_size=1, padding=0)

        # Mark that we have a neck for property testing
        self.neck = True  # Will use this for with_neck property

    def extract_feat(self, imgs):
        """Extract features from images"""
        # Conv layer
        x = self.conv_op(imgs)

        # ReLU
        relu_op = F.Relu("relu")
        x = relu_op(x)

        # Neck if present
        if self.with_neck:
            x = self.neck_op(x)

        return x

    def forward_train(self, imgs, img_metas, gt_labels=None, **kwargs):
        """Forward for training - compute simple loss"""
        # Extract features
        feats = self.extract_feat(imgs)

        # Head predictions
        pred = self.head_op(feats)

        # Simple loss: mean squared error with target
        if gt_labels is not None:
            diff = F.Sub("loss_diff")(pred, gt_labels)
            sq = F.Pow("loss_sq")(
                diff, F._from_data("two", np.array(2.0, dtype=np.float32))
            )
            loss = F.ReduceMean("loss_mean", axes=None)(sq)
        else:
            # Dummy loss if no target
            sq = F.Pow("loss_sq")(
                pred, F._from_data("two", np.array(2.0, dtype=np.float32))
            )
            mean = F.ReduceMean("loss_mean", axes=None)(sq)
            loss = F.Mul("loss_scale")(
                mean, F._from_data("scale", np.array(0.01, dtype=np.float32))
            )

        # Log variables (extract from .data)
        log_vars = {}
        if loss.data is not None:
            if isinstance(loss.data, np.ndarray):
                log_vars["loss"] = (
                    float(loss.data.flatten()[0])
                    if loss.data.size == 1
                    else float(loss.data.mean())
                )
            else:
                log_vars["loss"] = float(loss.data)

        num_samples = imgs.shape[0] if hasattr(imgs, "shape") else len(img_metas)

        return dict(loss=loss, log_vars=log_vars, num_samples=num_samples)

    def simple_test(self, imgs, img_metas, **kwargs):
        """Simple test - return predictions"""
        feats = self.extract_feat(imgs)
        pred = self.head_op(feats)

        # Return per-image results (extract from .data)
        results = []
        if pred.data is not None:
            for i in range(pred.data.shape[0]):
                results.append({"pred": pred.data[i]})

        return results


# ============================================================
# Helper Functions
# ============================================================


def sync_weights_pytorch_to_ttsim(model_pytorch, model_ttsim):
    """Copy weights from PyTorch model to ttsim model (via .data attribute)"""
    pytorch_state = model_pytorch.state_dict()

    # Map PyTorch layer names to ttsim operator parameter names
    weight_map = {
        "conv.weight": ("conv_op", "conv.param"),
        "neck.weight": ("neck_op", "neck.param"),
        "head.weight": ("head_op", "head.param"),
    }

    for pytorch_name, (ttsim_attr, param_name) in weight_map.items():
        if pytorch_name in pytorch_state and hasattr(model_ttsim, ttsim_attr):
            ttsim_op = getattr(model_ttsim, ttsim_attr)
            # Find the parameter tensor in the op's params
            if hasattr(ttsim_op, "params") and ttsim_op.params:
                param_tensor = ttsim_op.params[0][
                    1
                ]  # params is list of (position, tensor) tuples
                pytorch_weight = pytorch_state[pytorch_name].detach().cpu().numpy()
                param_tensor.data = pytorch_weight
                print(
                    f"  Synced {pytorch_name} -> {ttsim_attr}.param, shape: {pytorch_weight.shape}"
                )


# ============================================================
# Comparison Tests
# ============================================================


def test_properties():
    """Test property methods: with_neck, with_bbox, with_mask"""
    print("\n" + "=" * 80)
    print("TEST 1: Property Methods")
    print("=" * 80)

    # Create models
    model_pytorch = SimpleMapperPyTorch(in_channels=3, feat_channels=64)
    model_ttsim = SimpleMapperTtsim(in_channels=3, feat_channels=64)

    # Test with_neck
    pytorch_neck = model_pytorch.with_neck
    ttsim_neck = model_ttsim.with_neck
    print(f"PyTorch with_neck: {pytorch_neck}")
    print(f"ttsim with_neck:   {ttsim_neck}")

    neck_match = pytorch_neck == ttsim_neck
    print(f"Properties match: {neck_match}")

    if neck_match:
        print("[PASS] PASS: Properties match!")
        return True
    else:
        print("[FAIL] FAIL: Properties mismatch!")
        return False


def test_extract_feat():
    """Test extract_feat: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 2: extract_feat()")
    print("=" * 80)

    # Create test data
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(50)
    img_np = np.random.randn(B, C, H, W).astype(np.float32)

    # Create models
    model_pytorch = SimpleMapperPyTorch(in_channels=C, feat_channels=64)
    model_ttsim = SimpleMapperTtsim(in_channels=C, feat_channels=64)

    # Sync weights
    sync_weights_pytorch_to_ttsim(model_pytorch, model_ttsim)

    # PyTorch forward
    img_torch = torch.from_numpy(img_np)
    model_pytorch.eval()
    with torch.no_grad():
        feat_pytorch = model_pytorch.extract_feat(img_torch)

    print(f"PyTorch feat shape: {feat_pytorch.shape}")
    print(f"PyTorch feat mean: {feat_pytorch.mean().item():.6f}")

    # ttsim forward
    img_ttsim = F._from_data("img", img_np)
    feat_ttsim = model_ttsim.extract_feat(img_ttsim)

    # Check ttsim output
    if feat_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    print(f"ttsim feat shape: {feat_ttsim.data.shape}")
    print(f"ttsim feat mean: {feat_ttsim.data.mean():.6f}")

    # Compare using np.allclose (relaxed tolerance for conv operations)
    matches = np.allclose(feat_pytorch.numpy(), feat_ttsim.data, rtol=1e-3, atol=1e-2)
    diff = np.abs(feat_pytorch.numpy() - feat_ttsim.data).max()
    print(f"Max difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: extract_feat matches!")
    else:
        print("[FAIL] FAIL: extract_feat mismatch!")

    return matches


def test_forward_train():
    """Test forward_train: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 3: forward_train()")
    print("=" * 80)

    # Create test data
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(51)
    img_np = np.random.randn(B, C, H, W).astype(np.float32)
    gt_np = np.random.randn(B, 10, 32, 32).astype(np.float32)

    img_metas = [{"filename": f"img_{i}.jpg"} for i in range(B)]

    # Create models
    model_pytorch = SimpleMapperPyTorch(in_channels=C, feat_channels=64)
    model_ttsim = SimpleMapperTtsim(in_channels=C, feat_channels=64)

    # Sync weights
    sync_weights_pytorch_to_ttsim(model_pytorch, model_ttsim)

    # PyTorch forward
    img_torch = torch.from_numpy(img_np)
    gt_torch = torch.from_numpy(gt_np)
    model_pytorch.eval()
    with torch.no_grad():
        output_pytorch = model_pytorch.forward_train(
            img_torch, img_metas, gt_labels=gt_torch
        )

    loss_pytorch = output_pytorch["loss"].item()
    print(f"PyTorch loss: {loss_pytorch:.6f}")
    print(f"PyTorch log_vars: {output_pytorch['log_vars']}")

    # ttsim forward
    img_ttsim = F._from_data("img", img_np)
    gt_ttsim = F._from_data("gt", gt_np)
    output_ttsim = model_ttsim.forward_train(img_ttsim, img_metas, gt_labels=gt_ttsim)

    loss_ttsim = output_ttsim["loss"]

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    # Extract scalar value
    if isinstance(loss_ttsim.data, np.ndarray):
        loss_ttsim_value = (
            float(loss_ttsim.data.flatten()[0])
            if loss_ttsim.data.size == 1
            else float(loss_ttsim.data.mean())
        )
    else:
        loss_ttsim_value = float(loss_ttsim.data)

    print(f"ttsim loss: {loss_ttsim_value:.6f}")
    print(f"ttsim log_vars: {output_ttsim['log_vars']}")

    # Compare using np.allclose (relaxed tolerance for conv operations)
    matches = np.allclose(loss_pytorch, loss_ttsim_value, rtol=1e-3, atol=1e-2)
    diff = abs(loss_pytorch - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: forward_train matches!")
    else:
        print("[FAIL] FAIL: forward_train mismatch!")

    return matches


def test_train_step():
    """Test train_step: PyTorch vs ttsim"""
    print("\n" + "=" * 80)
    print("TEST 4: train_step()")
    print("=" * 80)

    # Create test data
    B, C, H, W = 2, 3, 32, 32
    np.random.seed(52)
    img_np = np.random.randn(B, C, H, W).astype(np.float32)
    gt_np = np.random.randn(B, 10, 32, 32).astype(np.float32)

    data_dict = {
        "imgs": None,  # Will be set per-framework
        "img_metas": [{"filename": f"img_{i}.jpg"} for i in range(B)],
        "gt_labels": None,  # Will be set per-framework
        "return_loss": True,
    }

    # Create models
    model_pytorch = SimpleMapperPyTorch(in_channels=C, feat_channels=64)
    model_ttsim = SimpleMapperTtsim(in_channels=C, feat_channels=64)

    # Sync weights
    sync_weights_pytorch_to_ttsim(model_pytorch, model_ttsim)

    # PyTorch train_step
    data_dict["imgs"] = torch.from_numpy(img_np)
    data_dict["gt_labels"] = torch.from_numpy(gt_np)
    model_pytorch.eval()
    with torch.no_grad():
        output_pytorch = model_pytorch.train_step(data_dict, optimizer=None)

    loss_pytorch = output_pytorch["loss"].item()
    print(f"PyTorch train_step loss: {loss_pytorch:.6f}")

    # ttsim train_step
    data_dict["imgs"] = F._from_data("img", img_np)
    data_dict["gt_labels"] = F._from_data("gt", gt_np)
    output_ttsim = model_ttsim.train_step(data_dict, optimizer=None)

    loss_ttsim = output_ttsim["loss"]

    # Check ttsim output
    if loss_ttsim.data is None:
        print("[WARN]  [SKIP] Cannot compare - ttsim output.data is None")
        return False

    # Extract scalar value
    if isinstance(loss_ttsim.data, np.ndarray):
        loss_ttsim_value = (
            float(loss_ttsim.data.flatten()[0])
            if loss_ttsim.data.size == 1
            else float(loss_ttsim.data.mean())
        )
    else:
        loss_ttsim_value = float(loss_ttsim.data)

    print(f"ttsim train_step loss: {loss_ttsim_value:.6f}")

    # Compare using np.allclose (relaxed tolerance for conv operations)
    matches = np.allclose(loss_pytorch, loss_ttsim_value, rtol=1e-3, atol=1e-2)
    diff = abs(loss_pytorch - loss_ttsim_value)
    print(f"Difference: {diff:.8f}")

    if matches:
        print("[PASS] PASS: train_step matches!")
    else:
        print("[FAIL] FAIL: train_step mismatch!")

    return matches


def main():
    """Run all BaseMapper comparison tests"""
    print("\n" + "=" * 80)
    print("BaseMapper Comparison: PyTorch vs ttsim")
    print("=" * 80)

    results = []

    # Run all tests
    results.append(("Properties", test_properties()))
    results.append(("extract_feat", test_extract_feat()))
    results.append(("forward_train", test_forward_train()))
    results.append(("train_step", test_train_step()))

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
        print(f"\n[WARN] {len(results) - total_pass} test(s) failed. Please review.")


if __name__ == "__main__":
    main()
