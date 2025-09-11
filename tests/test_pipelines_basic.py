#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Basic test for Polaris diffusion pipelines scaffolding.
Validates that the base pipeline and scheduler classes work correctly.
"""

import tempfile
import os
from pathlib import Path

import pytest
import numpy as np

# Import Polaris pipeline components
from workloads.diffusers.SDXLPipelinePolaris import SDXLPipelinePolarisWorkload
from workloads.diffusers.schedulers.base import SchedulerBase
from workloads.diffusers.schedulers.euler_discrete import EulerDiscreteScheduler
from test_components_integration import PolarisDiffusionPipeline  # type: ignore[import]


class MockComponent:
    """Mock component for testing pipeline registration."""
    def __init__(self, name="mock"):
        self.name = name
        self.onnx_path = f"/tmp/{name}.onnx"


class TestPolarisDiffusionPipeline:
    """Test the base PolarisDiffusionPipeline class."""

    def test_pipeline_initialization(self):
        """Test that pipeline can be initialized."""
        pipeline = PolarisDiffusionPipeline()
        assert pipeline is not None
        assert pipeline._components == {}
        assert pipeline._execution_device is None

    def test_register_modules(self):
        """Test component registration."""
        pipeline = PolarisDiffusionPipeline()

        # Register mock components
        unet = MockComponent("unet")
        vae = MockComponent("vae")
        scheduler = EulerDiscreteScheduler()

        pipeline.register_modules(unet=unet, vae=vae, scheduler=scheduler)

        # Check components are registered
        assert "unet" in pipeline._components
        assert "vae" in pipeline._components
        assert "scheduler" in pipeline._components

        # Check config is updated (in-memory config uses tuples)
        assert pipeline.config["unet"] == ("MockComponent", "/tmp/unet.onnx")
        assert pipeline.config["vae"] == ("MockComponent", "/tmp/vae.onnx")
        assert pipeline.config["scheduler"] == ("EulerDiscreteScheduler", None)

        # Check attributes are set
        assert pipeline.unet == unet
        assert pipeline.vae == vae
        assert pipeline.scheduler == scheduler

    def test_save_load_config(self):
        """Test config save and load."""
        pipeline = PolarisDiffusionPipeline()

        # Register components
        unet = MockComponent("unet")
        pipeline.register_modules(unet=unet)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save config
            pipeline.save_config(tmpdir)

            # Check file exists
            config_path = Path(tmpdir) / "model_index.json"
            assert config_path.exists()

            # Load config
            loaded_config = PolarisDiffusionPipeline.load_config(tmpdir)
            assert loaded_config["unet"] == ["MockComponent", "/tmp/unet.onnx"]


class TestEulerDiscreteScheduler:
    """Test the EulerDiscreteScheduler class."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization with default parameters."""
        scheduler = EulerDiscreteScheduler()

        assert scheduler is not None
        assert scheduler.config["num_train_timesteps"] == 1000
        assert scheduler.config["beta_start"] == 0.0001
        assert scheduler.config["beta_end"] == 0.02
        assert scheduler.config["prediction_type"] == "epsilon"

    def test_scheduler_custom_params(self):
        """Test scheduler with custom parameters."""
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=500,
            beta_start=0.001,
            beta_end=0.01,
            prediction_type="sample"
        )

        assert scheduler.config["num_train_timesteps"] == 500
        assert scheduler.config["beta_start"] == 0.001
        assert scheduler.config["beta_end"] == 0.01
        assert scheduler.config["prediction_type"] == "sample"

    def test_set_timesteps(self):
        """Test timestep setting."""
        scheduler = EulerDiscreteScheduler()

        # Set timesteps
        scheduler.set_timesteps(num_inference_steps=10)

        assert scheduler.num_inference_steps == 10
        assert len(scheduler.timesteps) == 10
        assert scheduler.timesteps[0] > scheduler.timesteps[-1]  # Decreasing

    def test_scale_model_input(self):
        """Test model input scaling."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10)

        # Test scaling for first timestep
        timestep = scheduler.timesteps[0]
        scale_factor = scheduler.scale_model_input(None, timestep)

        assert isinstance(scale_factor, (float, np.floating))
        assert scale_factor > 0

    def test_step(self):
        """Test denoising step."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=10)

        # Test step computation
        timestep = scheduler.timesteps[0]
        step_params = scheduler.step(None, timestep, None)

        assert isinstance(step_params, dict)
        assert "sigma" in step_params
        assert "gamma" in step_params
        assert "prediction_type" in step_params
        assert "timestep" in step_params
        assert step_params["prediction_type"] == scheduler.config["prediction_type"]

    def test_scheduler_save_load(self):
        """Test scheduler save and load."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save scheduler
            scheduler.save_pretrained(tmpdir)

            # Load scheduler
            loaded_scheduler = EulerDiscreteScheduler.from_pretrained(tmpdir)

            assert loaded_scheduler.config["num_train_timesteps"] == 1000
            assert loaded_scheduler.config["beta_start"] == 0.0001


class TestSchedulerMixin:
    """Test the SchedulerMixin base class."""

    def test_scheduler_mixin(self):
        """Test basic SchedulerMixin functionality."""
        scheduler = EulerDiscreteScheduler()

        # Test compatibles
        compatibles = scheduler.compatibles
        assert isinstance(compatibles, list)
        assert "EulerDiscreteScheduler" in compatibles

    def test_begin_index(self):
        """Test begin index setting."""
        scheduler = EulerDiscreteScheduler()

        scheduler.set_begin_index(5)
        assert scheduler.begin_index == 5


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running Polaris pipelines basic tests...")

    # Test pipeline
    pipeline = TestPolarisDiffusionPipeline()
    pipeline.test_pipeline_initialization()
    pipeline.test_register_modules()
    pipeline.test_save_load_config()
    print("✓ Pipeline tests passed")

    # Test scheduler
    scheduler_tests = TestEulerDiscreteScheduler()
    scheduler_tests.test_scheduler_initialization()
    scheduler_tests.test_scheduler_custom_params()
    scheduler_tests.test_set_timesteps()
    scheduler_tests.test_scale_model_input()
    scheduler_tests.test_step()
    scheduler_tests.test_scheduler_save_load()
    print("✓ Scheduler tests passed")

    # Test mixin
    mixin_tests = TestSchedulerMixin()
    mixin_tests.test_scheduler_mixin()
    mixin_tests.test_begin_index()
    print("✓ Mixin tests passed")

    print("All basic tests passed! Polaris diffusion pipelines scaffolding is working.")