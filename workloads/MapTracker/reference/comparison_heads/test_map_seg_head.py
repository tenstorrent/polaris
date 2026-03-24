#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Comparison test for MapSegHead: TTSim vs PyTorch
Tests the semantic segmentation head for BEV features
"""

import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np

# ============================================================================
# PyTorch Reference Implementation
# ============================================================================


class MapSegHeadPyTorch(nn.Module):
    """
    PyTorch reference implementation of MapSegHead for validation.
    Semantic segmentation head that upsamples BEV features to canvas size.
    """

    def __init__(
        self,
        num_classes=3,
        in_channels=256,
        embed_dims=256,
        bev_size=(100, 50),
        canvas_size=(200, 100),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.bev_size = bev_size
        self.canvas_size = canvas_size

        # Calculate number of upsampling blocks
        assert (
            canvas_size[0] % bev_size[0] == 0
        ), "canvas size must be a multiple of bev size"
        self.num_up_blocks = int(np.log2(canvas_size[0] // bev_size[0]))

        self.cls_out_channels = num_classes

        # Initial convolution
        self.conv_in = nn.Conv2d(
            in_channels, embed_dims, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

        # Upsampling blocks
        self.conv_mid_layers = nn.ModuleList([])
        self.downsample_layers = nn.ModuleList([])

        for _ in range(self.num_up_blocks):
            conv_mid = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv_mid_layers.append(conv_mid)
            self.downsample_layers.append(
                nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
            )

        # Output convolution
        self.conv_out = nn.Conv2d(
            embed_dims, self.cls_out_channels, kernel_size=1, padding=0
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Just use default initialization for testing
        pass

    def forward_test(self, bev_features):
        """Forward pass for testing (no loss computation)."""
        x = self.relu(self.conv_in(bev_features))

        for conv_mid in self.conv_mid_layers:
            x = conv_mid(x)

        preds = self.conv_out(x)

        # Downsample features back to original BEV size
        seg_feats = x
        for downsample in self.downsample_layers:
            seg_feats = downsample(seg_feats)

        return preds, seg_feats


# ============================================================================
# Import TTSim Implementation
# ============================================================================

from workloads.MapTracker.plugin.models.heads.Map_Seg_Head import (
    MapSegHead as MapSegHeadTTSim,
)
import ttsim.front.functional.op as F

# ============================================================================
# Test Functions
# ============================================================================


def test_map_seg_head_construction():
    """Test that both PyTorch and TTSim modules can be constructed."""
    print("\n" + "=" * 80)
    print("TEST 1: Module Construction")
    print("=" * 80)

    # Test parameters (smaller config for faster testing)
    num_classes = 3
    in_channels = 64
    embed_dims = 64
    bev_size = (50, 25)
    canvas_size = (100, 50)

    print(f"\nConfiguration:")
    print(f"  Classes: {num_classes}")
    print(f"  Input channels: {in_channels}")
    print(f"  Embed dims: {embed_dims}")
    print(f"  BEV size: {bev_size}")
    print(f"  Canvas size: {canvas_size}")

    try:
        # PyTorch model
        model_pytorch = MapSegHeadPyTorch(
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dims=embed_dims,
            bev_size=bev_size,
            canvas_size=canvas_size,
        )
        model_pytorch.eval()
        print("\n[OK] PyTorch model constructed")

        # TTSim model
        model_ttsim = MapSegHeadTTSim(
            name="test_seg_head",
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dims=embed_dims,
            bev_size=bev_size,
            canvas_size=canvas_size,
            max_batch_size=4,
        )
        print("[OK] TTSim model constructed")

        print(f"\n[OK] TEST PASSED: Both models constructed successfully")
        return True

    except Exception as e:
        print(f"\n[X] TEST FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_map_seg_head_forward():
    """Test forward pass comparison between PyTorch and TTSim."""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass Comparison")
    print("=" * 80)

    # Test parameters (smaller config for faster testing)
    bs = 1
    num_classes = 3
    in_channels = 64
    embed_dims = 64
    bev_h, bev_w = 50, 25
    canvas_h, canvas_w = 100, 50

    print(f"\nTest configuration:")
    print(f"  Batch size: {bs}")
    print(f"  Input: [{bs}, {in_channels}, {bev_h}, {bev_w}]")
    print(f"  Expected output preds: [{bs}, {num_classes}, {canvas_h}, {canvas_w}]")
    print(f"  Expected output feats: [{bs}, {embed_dims}, {bev_h}, {bev_w}]")

    try:
        # Create models
        model_pytorch = MapSegHeadPyTorch(
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dims=embed_dims,
            bev_size=(bev_h, bev_w),
            canvas_size=(canvas_h, canvas_w),
        )
        model_pytorch.eval()

        model_ttsim = MapSegHeadTTSim(
            name="test_seg_head",
            num_classes=num_classes,
            in_channels=in_channels,
            embed_dims=embed_dims,
            bev_size=(bev_h, bev_w),
            canvas_size=(canvas_h, canvas_w),
            max_batch_size=4,
        )

        # Create input
        np.random.seed(42)
        torch.manual_seed(42)
        bev_features_np = np.random.randn(bs, in_channels, bev_h, bev_w).astype(
            np.float32
        )
        bev_features_torch = torch.from_numpy(bev_features_np)

        print(f"\n[OK] Input created: {bev_features_np.shape}")

        # ========== Inject PyTorch weights into TTSim ==========
        print("\n" + "-" * 80)
        print("Injecting PyTorch weights into TTSim model")
        print("-" * 80)

        # Initial conv
        conv_in_weight = model_pytorch.conv_in.weight.data.numpy()
        model_ttsim.conv_in.params[0][1].data = conv_in_weight
        print(f"[OK] Injected conv_in weight: {conv_in_weight.shape}")

        # Upsampling conv blocks
        num_up_blocks = model_ttsim.num_up_blocks
        for i in range(num_up_blocks):
            # Conv weight and bias (bias is separate parameter in TTSim)
            conv_weight = model_pytorch.conv_mid_layers[i][1].weight.data.numpy()
            conv_bias = model_pytorch.conv_mid_layers[i][1].bias.data.numpy()
            model_ttsim.conv_ups[i].params[0][1].data = conv_weight
            getattr(model_ttsim, f"conv_up_bias_{i}").data = (
                conv_bias  # Separate bias parameter
            )
            print(
                f"[OK] Injected conv_up_{i} weight: {conv_weight.shape}, bias: {conv_bias.shape}"
            )

        # Output conv (bias is separate parameter in TTSim)
        conv_out_weight = model_pytorch.conv_out.weight.data.numpy()
        conv_out_bias = model_pytorch.conv_out.bias.data.numpy()
        model_ttsim.conv_out.params[0][1].data = conv_out_weight
        model_ttsim.conv_out_bias.data = conv_out_bias  # Separate bias parameter
        print(
            f"[OK] Injected conv_out weight: {conv_out_weight.shape}, bias: {conv_out_bias.shape}"
        )

        print("\n[OK] All weights injected successfully")

        # ========== PyTorch Forward Pass ==========
        print("\n" + "-" * 80)
        print("PyTorch Forward Pass")
        print("-" * 80)

        with torch.no_grad():
            preds_pytorch, seg_feats_pytorch = model_pytorch.forward_test(
                bev_features_torch
            )

        print(f"Predictions shape: {preds_pytorch.shape}")
        print(
            f"Predictions range: [{preds_pytorch.min().item():.6f}, {preds_pytorch.max().item():.6f}]"
        )
        print(f"Seg features shape: {seg_feats_pytorch.shape}")
        print(
            f"Seg features range: [{seg_feats_pytorch.min().item():.6f}, {seg_feats_pytorch.max().item():.6f}]"
        )

        # ========== TTSim Forward Pass ==========
        print("\n" + "-" * 80)
        print("TTSim Forward Pass")
        print("-" * 80)

        bev_features_ttsim = F._from_data(
            "bev_features", bev_features_np, is_const=True
        )
        preds_ttsim, seg_feats_ttsim = model_ttsim(bev_features_ttsim)

        if preds_ttsim.data is None:
            print("[X] ERROR: TTSim predictions have no data!")
            return False

        if seg_feats_ttsim.data is None:
            print("[X] ERROR: TTSim seg features have no data!")
            return False

        print(f"Predictions shape: {preds_ttsim.shape}")
        print(
            f"Predictions range: [{preds_ttsim.data.min():.6f}, {preds_ttsim.data.max():.6f}]"
        )
        print(f"Seg features shape: {seg_feats_ttsim.shape}")
        print(
            f"Seg features range: [{seg_feats_ttsim.data.min():.6f}, {seg_feats_ttsim.data.max():.6f}]"
        )

        # ========== Compare Outputs ==========
        print("\n" + "=" * 80)
        print("Output Comparison")
        print("=" * 80)

        preds_pytorch_np = preds_pytorch.numpy()
        seg_feats_pytorch_np = seg_feats_pytorch.numpy()

        # Compare predictions
        preds_diff = np.abs(preds_ttsim.data - preds_pytorch_np)
        print(f"\nPredictions comparison:")
        print(f"  Max absolute diff: {preds_diff.max():.10e}")
        print(f"  Mean absolute diff: {preds_diff.mean():.10e}")
        print(f"  Median absolute diff: {np.median(preds_diff):.10e}")

        # Compare seg features
        feats_diff = np.abs(seg_feats_ttsim.data - seg_feats_pytorch_np)
        print(f"\nSeg features comparison:")
        print(f"  Max absolute diff: {feats_diff.max():.10e}")
        print(f"  Mean absolute diff: {feats_diff.mean():.10e}")
        print(f"  Median absolute diff: {np.median(feats_diff):.10e}")

        # Validate using np.allclose
        preds_match = np.allclose(
            preds_ttsim.data, preds_pytorch_np, rtol=1e-5, atol=1e-5
        )
        feats_match = np.allclose(
            seg_feats_ttsim.data, seg_feats_pytorch_np, rtol=1e-5, atol=1e-5
        )

        print(f"\nValidation (rtol=1e-5, atol=1e-5):")
        print(f"  Predictions match: {'[OK] YES' if preds_match else '[X] NO'}")
        print(f"  Seg features match: {'[OK] YES' if feats_match else '[X] NO'}")

        if not (preds_match and feats_match):
            if not preds_match:
                max_idx = np.unravel_index(preds_diff.argmax(), preds_diff.shape)
                print(f"\n  Predictions - Largest diff at {max_idx}:")
                print(f"    PyTorch: {preds_pytorch_np[max_idx]:.10f}")
                print(f"    TTSim:   {preds_ttsim.data[max_idx]:.10f}")
            if not feats_match:
                max_idx = np.unravel_index(feats_diff.argmax(), feats_diff.shape)
                print(f"\n  Features - Largest diff at {max_idx}:")
                print(f"    PyTorch: {seg_feats_pytorch_np[max_idx]:.10f}")
                print(f"    TTSim:   {seg_feats_ttsim.data[max_idx]:.10f}")

        # Return validation result
        if preds_match and feats_match:
            print(f"\n[OK] TEST PASSED: All outputs match within tolerance")
            return True
        else:
            print(f"\n[X] TEST FAILED: Outputs differ")
            return False

    except Exception as e:
        print(f"\n[X] TEST FAILED with exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_map_seg_head_different_sizes():
    """Test with different upsampling configurations."""
    print("\n" + "=" * 80)
    print("TEST 3: Different Upsampling Configurations")
    print("=" * 80)

    test_configs = [
        # (bev_size, canvas_size, description) - using smaller configs for speed
        ((25, 25), (50, 50), "2x upsampling (square)"),
        ((50, 25), (100, 50), "2x upsampling (rectangular)"),
    ]

    all_passed = True

    for bev_size, canvas_size, desc in test_configs:
        print(f"\n{desc}:")
        print(f"  BEV: {bev_size}, Canvas: {canvas_size}")

        try:
            bs = 1
            in_channels = 32  # Reduced from 128
            embed_dims = 32  # Reduced from 128
            num_classes = 3

            # Create models
            model_pytorch = MapSegHeadPyTorch(
                num_classes=num_classes,
                in_channels=in_channels,
                embed_dims=embed_dims,
                bev_size=bev_size,
                canvas_size=canvas_size,
            )
            model_pytorch.eval()

            model_ttsim = MapSegHeadTTSim(
                name=f"test_seg_{bev_size[0]}x{bev_size[1]}",
                num_classes=num_classes,
                in_channels=in_channels,
                embed_dims=embed_dims,
                bev_size=bev_size,
                canvas_size=canvas_size,
                max_batch_size=4,
            )

            # Create input
            np.random.seed(123)
            bev_features_np = np.random.randn(
                bs, in_channels, bev_size[0], bev_size[1]
            ).astype(np.float32)

            # Inject weights (bias is separate parameter in TTSim)
            model_ttsim.conv_in.params[0][
                1
            ].data = model_pytorch.conv_in.weight.data.numpy()
            num_up_blocks = model_ttsim.num_up_blocks
            for i in range(num_up_blocks):
                model_ttsim.conv_ups[i].params[0][1].data = (
                    model_pytorch.conv_mid_layers[i][1].weight.data.numpy()
                )
                getattr(model_ttsim, f"conv_up_bias_{i}").data = (
                    model_pytorch.conv_mid_layers[i][1].bias.data.numpy()
                )
            model_ttsim.conv_out.params[0][
                1
            ].data = model_pytorch.conv_out.weight.data.numpy()
            model_ttsim.conv_out_bias.data = model_pytorch.conv_out.bias.data.numpy()

            # Forward pass
            with torch.no_grad():
                preds_pytorch, seg_feats_pytorch = model_pytorch.forward_test(
                    torch.from_numpy(bev_features_np)
                )

            bev_features_ttsim = F._from_data(
                f"bev_features_{bev_size[0]}x{bev_size[1]}",
                bev_features_np,
                is_const=True,
            )
            preds_ttsim, seg_feats_ttsim = model_ttsim(bev_features_ttsim)

            # Compare
            preds_diff = np.abs(preds_ttsim.data - preds_pytorch.numpy())
            feats_diff = np.abs(seg_feats_ttsim.data - seg_feats_pytorch.numpy())

            preds_match = np.allclose(
                preds_ttsim.data, preds_pytorch.numpy(), rtol=1e-5, atol=1e-5
            )
            feats_match = np.allclose(
                seg_feats_ttsim.data, seg_feats_pytorch.numpy(), rtol=1e-5, atol=1e-5
            )

            print(f"  Preds diff: max={preds_diff.max():.6e}, match={preds_match}")
            print(f"  Feats diff: max={feats_diff.max():.6e}, match={feats_match}")

            if preds_match and feats_match:
                print(f"  [OK] PASSED")
            else:
                print(f"  [X] FAILED")
                all_passed = False

        except Exception as e:
            print(f"  [X] FAILED: {str(e)}")
            all_passed = False

    if all_passed:
        print(f"\n[OK] TEST PASSED: All configurations work correctly")
    else:
        print(f"\n[X] TEST FAILED: Some configurations failed")

    return all_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MapSegHead Test Suite: PyTorch vs TTSim")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Construction", test_map_seg_head_construction()))
    results.append(("Forward Pass", test_map_seg_head_forward()))
    results.append(("Different Sizes", test_map_seg_head_different_sizes()))

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results:
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{test_name:.<60} {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n! All tests passed!")
    else:
        print(f"\n[WARNING]  {total - passed} test(s) failed")
