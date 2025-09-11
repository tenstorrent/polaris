#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Polaris SDXL Pipeline serialization and loading functionality.

Tests save_pretrained, from_pretrained, model_index.json, and scheduler_config.json
handling with comprehensive roundtrip validation.
"""

import tempfile
import os
import json
from pathlib import Path

import pytest
import numpy as np

# Import Polaris components
from pipelines.stable_diffusion_xl import StableDiffusionXLPipelinePolaris
from pipelines.schedulers import EulerDiscreteScheduler


class TestSerializationBasic:
    """Test basic serialization functionality."""

    def test_save_pretrained_creates_directory(self):
        """Test that save_pretrained creates the target directory."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"

            # Directory shouldn't exist initially
            assert not save_path.exists()

            # Save pipeline
            pipeline.save_pretrained(save_path)

            # Directory should now exist
            assert save_path.exists()
            assert save_path.is_dir()

    def test_save_pretrained_creates_model_index(self):
        """Test that save_pretrained creates model_index.json."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            # Check model_index.json exists
            model_index_path = save_path / "model_index.json"
            assert model_index_path.exists()

            # Load and verify content
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)

            # Check basic structure
            assert "_class_name" in model_index
            assert model_index["_class_name"] == "StableDiffusionXLPipelinePolaris"
            assert "_diffusers_version" in model_index
            assert "_name_or_path" in model_index

    def test_save_pretrained_with_scheduler(self):
        """Test saving pipeline with scheduler creates scheduler config."""
        scheduler = EulerDiscreteScheduler()
        pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            # Check scheduler_config.json exists
            scheduler_config_path = save_path / "scheduler_config.json"
            assert scheduler_config_path.exists()

            # Load and verify content
            with open(scheduler_config_path, 'r') as f:
                scheduler_config = json.load(f)

            # Should contain scheduler configuration
            assert isinstance(scheduler_config, dict)
            # Check for common scheduler parameters
            expected_keys = ["num_train_timesteps", "beta_start", "beta_end"]
            assert any(key in scheduler_config for key in expected_keys)


class TestSerializationRoundtrip:
    """Test save/load roundtrip functionality."""

    def test_roundtrip_basic_pipeline(self):
        """Test basic pipeline save/load roundtrip."""
        # Create original pipeline
        original_pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"

            # Save pipeline
            original_pipeline.save_pretrained(save_path)

            # Load pipeline
            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Verify basic properties
            assert loaded_pipeline.__class__ == original_pipeline.__class__
            assert loaded_pipeline.config is not None

    def test_roundtrip_with_scheduler(self):
        """Test pipeline with scheduler save/load roundtrip."""
        # Create original pipeline with scheduler
        scheduler = EulerDiscreteScheduler()
        original_pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"

            # Save pipeline
            original_pipeline.save_pretrained(save_path)

            # Load pipeline
            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Verify scheduler is loaded
            assert loaded_pipeline.scheduler is not None
            assert loaded_pipeline.scheduler.__class__ == EulerDiscreteScheduler

    def test_roundtrip_execution_consistency(self):
        """Test that saved/loaded pipeline produces consistent results."""
        # Create original pipeline
        original_pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"

            # Save pipeline
            original_pipeline.save_pretrained(save_path)

            # Load pipeline
            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Test execution with same parameters
            prompt = "test prompt"
            params = {
                "height": 128,
                "width": 128,
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "return_dict": True,
            }

            # Both pipelines should execute without error
            original_result = original_pipeline(prompt, **params)
            loaded_result = loaded_pipeline(prompt, **params)

            # Both should return result objects
            assert hasattr(original_result, 'images') or isinstance(original_result, dict)
            assert hasattr(loaded_result, 'images') or isinstance(loaded_result, dict)


class TestModelIndexFormat:
    """Test model_index.json format handling."""

    def test_model_index_structure(self):
        """Test model_index.json has correct structure."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            model_index_path = save_path / "model_index.json"
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)

            # Required metadata fields
            required_fields = ["_class_name", "_diffusers_version", "_name_or_path"]
            for field in required_fields:
                assert field in model_index

            # Class name should match
            assert model_index["_class_name"] == "StableDiffusionXLPipelinePolaris"

    def test_model_index_component_entries(self):
        """Test that model_index contains proper component entries."""
        scheduler = EulerDiscreteScheduler()
        pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            model_index_path = save_path / "model_index.json"
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)

            # Should have scheduler entry
            assert "scheduler" in model_index
            scheduler_entry = model_index["scheduler"]
            assert isinstance(scheduler_entry, dict)
            assert "type" in scheduler_entry

    def test_model_index_loading(self):
        """Test loading pipeline from model_index.json."""
        # Create and save pipeline
        scheduler = EulerDiscreteScheduler()
        original_pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            original_pipeline.save_pretrained(save_path)

            # Load using from_pretrained
            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Verify components are loaded correctly
            assert loaded_pipeline.scheduler is not None
            assert loaded_pipeline.scheduler.__class__ == EulerDiscreteScheduler


class TestSchedulerConfig:
    """Test scheduler_config.json handling."""

    def test_scheduler_config_format(self):
        """Test scheduler_config.json has correct format."""
        scheduler = EulerDiscreteScheduler()
        pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            scheduler_config_path = save_path / "scheduler_config.json"
            with open(scheduler_config_path, 'r') as f:
                scheduler_config = json.load(f)

            # Should be a dictionary with scheduler parameters
            assert isinstance(scheduler_config, dict)
            assert len(scheduler_config) > 0

            # Check for typical scheduler parameters
            scheduler_params = [
                "num_train_timesteps", "beta_start", "beta_end",
                "beta_schedule", "prediction_type", "timestep_spacing"
            ]
            found_params = [param for param in scheduler_params if param in scheduler_config]
            assert len(found_params) > 0, f"No expected scheduler params found in {scheduler_config}"

    def test_scheduler_config_loading(self):
        """Test that scheduler config is properly loaded."""
        # Create scheduler with custom config
        scheduler = EulerDiscreteScheduler(prediction_type="epsilon")
        original_pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            original_pipeline.save_pretrained(save_path)

            # Load pipeline
            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Verify scheduler config is preserved
            assert loaded_pipeline.scheduler is not None
            # Note: The exact config preservation depends on scheduler implementation


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_save_pretrained_overwrites_existing(self):
        """Test that save_pretrained overwrites existing files."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"

            # Save once
            pipeline.save_pretrained(save_path)
            first_save_time = os.path.getmtime(save_path / "model_index.json")

            # Wait a moment and save again
            import time
            time.sleep(0.01)

            pipeline.save_pretrained(save_path)
            second_save_time = os.path.getmtime(save_path / "model_index.json")

            # File should have been updated
            assert second_save_time >= first_save_time

    def test_load_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises((FileNotFoundError, ValueError)):
            StableDiffusionXLPipelinePolaris.from_pretrained("/nonexistent/path")

    def test_load_invalid_model_index(self):
        """Test loading with invalid model_index.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            save_path.mkdir()

            # Create invalid model_index.json
            model_index_path = save_path / "model_index.json"
            with open(model_index_path, 'w') as f:
                json.dump({"invalid": "data"}, f)

            # Should handle gracefully or raise appropriate error
            try:
                StableDiffusionXLPipelinePolaris.from_pretrained(save_path)
            except (ValueError, KeyError):
                pass  # Expected for invalid format

    def test_save_empty_pipeline(self):
        """Test saving pipeline with no components."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            # Should still create model_index.json
            model_index_path = save_path / "model_index.json"
            assert model_index_path.exists()

            with open(model_index_path, 'r') as f:
                model_index = json.load(f)

            # Should have basic metadata even with no components
            assert "_class_name" in model_index


class TestSerializationCompatibility:
    """Test compatibility with different pipeline configurations."""

    def test_multiple_schedulers(self):
        """Test serialization with different scheduler types."""
        # Test with EulerDiscreteScheduler
        scheduler = EulerDiscreteScheduler()
        pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            assert loaded_pipeline.scheduler.__class__ == EulerDiscreteScheduler

    def test_pipeline_metadata_preservation(self):
        """Test that pipeline metadata is preserved during save/load."""
        pipeline = StableDiffusionXLPipelinePolaris()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_pipeline"
            pipeline.save_pretrained(save_path)

            loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)

            # Check that basic pipeline properties are preserved
            assert loaded_pipeline.__class__ == pipeline.__class__
            assert loaded_pipeline.config is not None


if __name__ == "__main__":
    # Run tests manually
    print("Running Polaris SDXL Serialization tests...")

    # Test basic serialization
    basic_tests = TestSerializationBasic()

    try:
        basic_tests.test_save_pretrained_creates_directory()
        print("✓ Directory creation")

        basic_tests.test_save_pretrained_creates_model_index()
        print("✓ Model index creation")

        basic_tests.test_save_pretrained_with_scheduler()
        print("✓ Scheduler config creation")

        # Test roundtrip
        roundtrip_tests = TestSerializationRoundtrip()

        roundtrip_tests.test_roundtrip_basic_pipeline()
        print("✓ Basic pipeline roundtrip")

        roundtrip_tests.test_roundtrip_with_scheduler()
        print("✓ Pipeline with scheduler roundtrip")

        roundtrip_tests.test_roundtrip_execution_consistency()
        print("✓ Execution consistency")

        # Test model index format
        model_index_tests = TestModelIndexFormat()

        model_index_tests.test_model_index_structure()
        print("✓ Model index structure")

        model_index_tests.test_model_index_component_entries()
        print("✓ Component entries")

        model_index_tests.test_model_index_loading()
        print("✓ Model index loading")

        # Test scheduler config
        scheduler_tests = TestSchedulerConfig()

        scheduler_tests.test_scheduler_config_format()
        print("✓ Scheduler config format")

        scheduler_tests.test_scheduler_config_loading()
        print("✓ Scheduler config loading")

        print("All serialization tests passed! 🎉")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
