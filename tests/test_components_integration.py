#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Polaris diffusion pipeline components.

Tests component loading, graph integration, and pipeline functionality.
"""

import tempfile
import os
from pathlib import Path
import json

import pytest
import numpy as np

# Import Polaris components
from pipelines.pipeline_utils import PolarisDiffusionPipeline
from pipelines.models import (
    BaseOnnxComponent,
    UNet2DConditionModelOnnx,
    AutoencoderKLOnnx,
    AutoencoderKLEncoderOnnx,
    AutoencoderKLDecoderOnnx,
    CLIPTextModelOnnx,
    CLIPTextModelWithProjectionOnnx,
    CLIPTokenizerHost
)
from pipelines.schedulers import EulerDiscreteScheduler
from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.tensor import SimTensor


class TestComponentIntegration:
    """Test integration of ONNX components with Polaris pipelines."""

    def test_pipeline_with_mock_components(self):
        """Test pipeline with mock ONNX components (no actual ONNX files)."""
        pipeline = PolarisDiffusionPipeline()

        # Create mock components (we'll test without actual ONNX files)
        scheduler = EulerDiscreteScheduler()

        # Register components
        pipeline.register_modules(scheduler=scheduler)

        # Check registration
        assert "scheduler" in pipeline._components
        assert pipeline.scheduler == scheduler

        # Test config
        config = pipeline.config
        assert "scheduler" in config

    def test_component_base_functionality(self):
        """Test base component functionality."""
        # Test with a mock ONNX file path (doesn't need to exist for basic tests)
        mock_path = "/tmp/mock_model.onnx"

        # Test that we can create components (they'll fail on load, but should initialize config)
        try:
            # This will fail because ONNX file doesn't exist, but should initialize properly
            component = BaseOnnxComponent(onnx_path=mock_path)
            assert False, "Should have failed due to missing ONNX file"
        except FileNotFoundError:
            pass  # Expected

    def test_unet_component_interface(self):
        """Test UNet component interface."""
        # Test expected shape calculations
        unet = UNet2DConditionModelOnnx.__new__(UNet2DConditionModelOnnx)  # Create without __init__
        unet.sample_size = 64
        unet.in_channels = 4
        unet.out_channels = 4

        # Test shape calculations
        latent_shape = unet.get_expected_latent_shape(batch_size=2, height=64, width=64)
        assert latent_shape == [2, 4, 64, 64]

        output_shape = unet.get_expected_output_shape(batch_size=2, height=64, width=64)
        assert output_shape == [2, 4, 64, 64]

    def test_vae_component_interface(self):
        """Test VAE component interface."""
        # Test encoder shape calculations
        encoder = AutoencoderKLEncoderOnnx.__new__(AutoencoderKLEncoderOnnx)
        encoder.scale_factor = 8
        encoder.in_channels = 3
        encoder.out_channels = 4

        # Test decoder shape calculations
        decoder = AutoencoderKLDecoderOnnx.__new__(AutoencoderKLDecoderOnnx)
        decoder.scale_factor = 8
        decoder.in_channels = 4
        decoder.out_channels = 3

    def test_text_encoder_tokenizer(self):
        """Test CLIP tokenizer host functionality."""
        try:
            tokenizer = CLIPTokenizerHost()

            # Test tokenization
            result = tokenizer("a test prompt", return_tensors="np")

            assert "input_ids" in result
            assert "attention_mask" in result
            assert isinstance(result["input_ids"], np.ndarray)
            assert isinstance(result["attention_mask"], np.ndarray)

        except ImportError:
            pytest.skip("transformers library not available")

    def test_sim_tensor_creation(self):
        """Test SimTensor creation through components."""
        from pipelines.pipeline_utils import PolarisDiffusionPipeline

        pipeline = PolarisDiffusionPipeline()

        # Create a tensor using pipeline helper
        tensor = pipeline._create_tensor(
            name="test_tensor",
            shape=[1, 4, 64, 64],
            dtype="float32"
        )

        assert isinstance(tensor, SimTensor)
        assert tensor.name == "test_tensor"
        assert tensor.shape == [1, 4, 64, 64]
        assert tensor.dtype == "float32"

    def test_scheduler_integration(self):
        """Test scheduler integration with pipeline."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=5)

        pipeline = PolarisDiffusionPipeline()
        pipeline.register_modules(scheduler=scheduler)

        # Test that scheduler is accessible
        assert pipeline.scheduler == scheduler
        assert pipeline.scheduler.num_inference_steps == 5

    def test_graph_creation(self):
        """Test WorkloadGraph creation."""
        graph = WorkloadGraph("test_graph")

        assert graph.get_node_count() == 0
        assert graph.get_edge_count() == 0
        assert graph._name == "test_graph"

    def test_pipeline_save_load_config(self):
        """Test pipeline config save and load."""
        pipeline = PolarisDiffusionPipeline()
        scheduler = EulerDiscreteScheduler()
        pipeline.register_modules(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save config
            pipeline.save_config(tmpdir)

            # Check file exists
            config_path = Path(tmpdir) / "model_index.json"
            assert config_path.exists()

            # Load and verify config
            with open(config_path, 'r') as f:
                saved_config = json.load(f)

            assert "scheduler" in saved_config
            assert saved_config["scheduler"][0] == "EulerDiscreteScheduler"


class TestComponentLoading:
    """Test component loading functionality."""

    def test_load_component_method(self):
        """Test the _load_component method with different component types."""
        # Test with valid component types (will fail on file loading, but should handle types correctly)
        test_cases = [
            ("unet", "UNet2DConditionModelOnnx", "/fake/path/unet.onnx"),
            ("vae", "AutoencoderKLOnnx", "/fake/path/vae.onnx"),
            ("text_encoder", "CLIPTextModelOnnx", "/fake/path/text_encoder.onnx"),
            ("text_encoder_2", "CLIPTextModelWithProjectionOnnx", "/fake/path/text_encoder_2.onnx"),
        ]

        for name, component_type, component_path in test_cases:
            try:
                PolarisDiffusionPipeline._load_component(
                    name, component_type, component_path, "/fake/base"
                )
                assert False, f"Should have failed for {component_type} due to missing file"
            except FileNotFoundError:
                pass  # Expected
            except ValueError as e:
                if "Unsupported component type" in str(e):
                    assert False, f"Component type {component_type} should be supported"

    def test_invalid_component_type(self):
        """Test loading with invalid component type."""
        with pytest.raises(ValueError, match="Unsupported component type"):
            PolarisDiffusionPipeline._load_component(
                "invalid", "InvalidType", "/fake/path", "/fake/base"
            )


class TestComponentGraphIntegration:
    """Test component integration with WorkloadGraph."""

    def test_tensor_operations(self):
        """Test tensor operations in graph context."""
        pipeline = PolarisDiffusionPipeline()
        graph = WorkloadGraph("test_graph")

        # Create tensors
        tensor1 = pipeline._create_tensor("input_tensor", [1, 4, 64, 64], "float32")
        tensor2 = pipeline._create_tensor("output_tensor", [1, 4, 64, 64], "float32")

        # Add tensors to graph
        graph.add_tensor(tensor1)
        graph.add_tensor(tensor2)

        # Verify tensors are in graph
        assert len(graph._tensors) == 2
        assert "input_tensor" in graph._tensors
        assert "output_tensor" in graph._tensors


if __name__ == "__main__":
    # Run basic integration tests
    print("Running Polaris components integration tests...")

    test_instance = TestComponentIntegration()

    # Test basic functionality
    test_instance.test_pipeline_with_mock_components()
    print("✓ Pipeline with mock components")

    test_instance.test_sim_tensor_creation()
    print("✓ SimTensor creation")

    test_instance.test_scheduler_integration()
    print("✓ Scheduler integration")

    test_instance.test_graph_creation()
    print("✓ Graph creation")

    test_instance.test_pipeline_save_load_config()
    print("✓ Pipeline save/load config")

    # Test component loading
    loading_tests = TestComponentLoading()
    loading_tests.test_load_component_method()
    print("✓ Component loading")

    loading_tests.test_invalid_component_type()
    print("✓ Invalid component type handling")

    # Test graph integration
    graph_tests = TestComponentGraphIntegration()
    graph_tests.test_tensor_operations()
    print("✓ Graph tensor operations")

    print("All integration tests passed! Component wrappers are working correctly.")
