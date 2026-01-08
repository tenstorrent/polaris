#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test runner for diffusers modules (doesn't require pytest).
Can be run directly with: python test_diffusers_standalone.py
"""

import os
import sys

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.diffusers.attention import BasicTransformerBlock
from workloads.diffusers.attention_processor import Attention
# Import diffusers modules
from workloads.diffusers.downsampling import Downsample2D
from workloads.diffusers.resnet import ResnetBlock2D
from workloads.diffusers.upsampling import Upsample2D


class RunnerForTest:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name, func):
        """Run a test function."""
        try:
            func()
            self.passed += 1
            self.tests.append((name, "PASS", None))
            print(f"✓ {name}")
        except AssertionError as e:
            self.failed += 1
            self.tests.append((name, "FAIL", str(e)))
            print(f"✗ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.tests.append((name, "ERROR", f"{type(e).__name__}: {e}"))
            print(f"✗ {name}: {type(e).__name__}: {e}")
    
    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Summary: {self.passed}/{total} passed")
        if self.failed > 0:
            print("\nFailed tests:")
            for name, status, error in self.tests:
                if status != "PASS":
                    print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


def test_downsample_non_conv_not_implemented():
    """Test that non-convolutional downsampling raises NotImplementedError."""
    try:
        Downsample2D(
            objname="test",
            channels=64,
            use_conv=False,
            out_channels=64
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "Non-convolutional downsampling" in str(e)


def test_downsample_channel_mismatch():
    """Test that channel mismatch raises ValueError."""
    try:
        Downsample2D(
            objname="test",
            channels=64,
            use_conv=False,
            out_channels=128
        )
        raise AssertionError("Expected ValueError")
    except ValueError as e:
        assert "64 != 128" in str(e)


def test_downsample_rms_norm():
    """Test that RMSNorm raises NotImplementedError."""
    try:
        Downsample2D(
            objname="test",
            channels=64,
            use_conv=True,
            norm_type="rms_norm"
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "RMSNorm" in str(e)


def test_upsample_layernorm():
    """Test that LayerNorm raises NotImplementedError."""
    try:
        Upsample2D(
            objname="test",
            channels=64,
            norm_type="ln_norm"
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "LayerNorm" in str(e)


def test_upsample_conv_transpose():
    """Test that ConvTranspose2d raises NotImplementedError."""
    try:
        Upsample2D(
            objname="test",
            channels=64,
            use_conv_transpose=True
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "ConvTranspose2d" in str(e)


def test_resnet_fir_upsampling():
    """Test that FIR upsampling raises NotImplementedError."""
    try:
        ResnetBlock2D(
            objname="test",
            in_channels=64,
            up=True,
            kernel="fir"  # type: ignore[arg-type]
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "FIR upsampling" in str(e)


def test_resnet_sde_vp_downsampling():
    """Test that SDE_VP downsampling raises NotImplementedError."""
    try:
        ResnetBlock2D(
            objname="test",
            in_channels=64,
            down=True,
            kernel="sde_vp"  # type: ignore[arg-type]
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "SDE_VP downsampling" in str(e)


def test_transformer_gated_attention():
    """Test that gated attention raises NotImplementedError."""
    try:
        BasicTransformerBlock(
            objname="test",
            dim=512,
            num_attention_heads=8,
            attention_head_dim=64,
            attention_type="gated"
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "Gated Self Attention" in str(e)


def test_attention_spatial_norm():
    """Test that spatial norm raises NotImplementedError."""
    try:
        Attention(
            objname="test",
            query_dim=512,
            heads=8,
            dim_head=64,
            spatial_norm_dim=512
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "spatial norm" in str(e)


def test_attention_qk_norm():
    """Test that qk_norm raises NotImplementedError."""
    try:
        Attention(
            objname="test",
            query_dim=512,
            heads=8,
            dim_head=64,
            qk_norm="layer_norm"
        )
        raise AssertionError("Expected NotImplementedError")
    except NotImplementedError as e:
        assert "qk_norm" in str(e)


def main():
    """Run all tests."""
    print("Running diffusers module tests...\n")
    
    runner = RunnerForTest()
    
    # Downsample2D tests
    runner.test("Downsample: Non-conv not implemented", test_downsample_non_conv_not_implemented)
    runner.test("Downsample: Channel mismatch", test_downsample_channel_mismatch)
    runner.test("Downsample: RMSNorm not supported", test_downsample_rms_norm)
    
    # Upsample2D tests
    runner.test("Upsample: LayerNorm not supported", test_upsample_layernorm)
    runner.test("Upsample: ConvTranspose2d not supported", test_upsample_conv_transpose)
    
    # ResnetBlock2D tests
    runner.test("Resnet: FIR upsampling not supported", test_resnet_fir_upsampling)
    runner.test("Resnet: SDE_VP downsampling not supported", test_resnet_sde_vp_downsampling)
    
    # BasicTransformerBlock tests
    runner.test("Transformer: Gated attention not supported", test_transformer_gated_attention)
    
    # Attention tests
    runner.test("Attention: Spatial norm not supported", test_attention_spatial_norm)
    runner.test("Attention: QK norm not supported", test_attention_qk_norm)
    
    success = runner.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

