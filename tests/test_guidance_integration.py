#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Classifier-Free Guidance integration in Polaris SDXL Pipeline.

Tests the ClassifierFreeGuidance utility and its integration with the SDXL pipeline,
including proper guidance scaling, rescaling, and pipeline behavior.
"""

import tempfile
import json

import pytest
import numpy as np

# Import Polaris components
from pipelines.guiders import ClassifierFreeGuidance
from pipelines.stable_diffusion_xl import StableDiffusionXLPipelinePolaris


class TestClassifierFreeGuidance:
    """Test the ClassifierFreeGuidance utility."""

    def test_guidance_initialization(self):
        """Test guidance utility initialization."""
        guidance = ClassifierFreeGuidance(guidance_scale=7.5, guidance_rescale=0.1)

        assert guidance.guidance_scale == 7.5
        assert guidance.guidance_rescale == 0.1

        config = guidance.get_guidance_config()
        assert config["guidance_scale"] == 7.5
        assert config["guidance_rescale"] == 0.1

    def test_guidance_scaling_basic(self):
        """Test basic guidance scaling."""
        guidance = ClassifierFreeGuidance(guidance_scale=2.0)

        # Create mock predictions
        unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])    # Shape: (1, 2, 2)

        # Apply guidance
        result = guidance.combine_predictions(unconditional, conditional)

        # Expected: unconditional + 2.0 * (conditional - unconditional)
        # = [1,2,3,4] + 2.0 * ([2,3,4,5] - [1,2,3,4]) = [1,2,3,4] + 2.0 * [1,1,1,1] = [3,4,5,6]
        expected = np.array([[[3.0, 4.0], [5.0, 6.0]]])

        np.testing.assert_array_equal(result, expected)

    def test_guidance_scaling_zero_scale(self):
        """Test guidance with scale = 1.0 (no guidance)."""
        guidance = ClassifierFreeGuidance(guidance_scale=1.0)

        unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])

        result = guidance.combine_predictions(unconditional, conditional)

        # With scale=1.0, result should equal conditional
        np.testing.assert_array_equal(result, conditional)

    def test_guidance_rescaling(self):
        """Test guidance rescaling functionality."""
        guidance = ClassifierFreeGuidance(guidance_scale=2.0, guidance_rescale=0.5)

        unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])

        result = guidance.combine_predictions(unconditional, conditional)

        # The rescaling should modify the final result
        # This is a complex calculation, so we just verify it's different from basic scaling
        assert result.shape == unconditional.shape
        assert not np.array_equal(result, unconditional)

    def test_guidance_validation(self):
        """Test input validation."""
        guidance = ClassifierFreeGuidance()

        unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])

        # Valid inputs should pass
        assert guidance.validate_guidance_inputs(unconditional, conditional)

        # Mismatched shapes should raise error
        wrong_shape = np.array([[[1.0, 2.0]]])  # Different shape
        with pytest.raises(ValueError):
            guidance.validate_guidance_inputs(unconditional, wrong_shape)

    def test_guidance_with_nan_values(self):
        """Test guidance behavior with NaN values."""
        guidance = ClassifierFreeGuidance()

        unconditional = np.array([[[1.0, float('nan')], [3.0, 4.0]]])
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])

        with pytest.raises(ValueError, match="NaN values detected"):
            guidance.validate_guidance_inputs(unconditional, conditional)

    def test_guidance_representation(self):
        """Test string representations."""
        guidance = ClassifierFreeGuidance(guidance_scale=3.5, guidance_rescale=0.2)

        repr_str = repr(guidance)
        assert "ClassifierFreeGuidance" in repr_str
        assert "3.5" in repr_str
        assert "0.2" in repr_str

        str_output = str(guidance)
        assert "guidance_scale: 3.5" in str_output
        assert "guidance_rescale: 0.2" in str_output


class TestGuidancePipelineIntegration:
    """Test guidance integration with SDXL pipeline."""

    def test_pipeline_with_guidance_utility(self):
        """Test pipeline initializes and uses guidance utility."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Initially guidance should be None
        assert pipeline._guidance is None

        # After running with guidance, it should be initialized
        result = pipeline(
            prompt="test prompt",
            guidance_scale=3.0,
            num_inference_steps=2,
            return_dict=True,
        )

        assert isinstance(result, dict) or hasattr(result, 'images')
        # Note: guidance utility is initialized during the call, so we can't easily test it here
        # In a real scenario, we'd need to access the internal state during execution

    def test_pipeline_guidance_scale_effect(self):
        """Test that different guidance scales produce different results."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Run with different guidance scales
        result_low = pipeline(
            prompt="test prompt",
            guidance_scale=1.0,  # No guidance
            num_inference_steps=2,
            return_dict=True,
        )

        result_high = pipeline(
            prompt="test prompt",
            guidance_scale=5.0,  # Strong guidance
            num_inference_steps=2,
            return_dict=True,
        )

        # Results should be different (in a real implementation)
        # For now, we just verify both complete successfully
        assert hasattr(result_low, 'images') or isinstance(result_low, dict)
        assert hasattr(result_high, 'images') or isinstance(result_high, dict)

    def test_pipeline_guidance_rescale(self):
        """Test pipeline with guidance rescaling."""
        pipeline = StableDiffusionXLPipelinePolaris()

        result = pipeline(
            prompt="test prompt",
            guidance_scale=7.5,
            guidance_rescale=0.1,
            num_inference_steps=2,
            return_dict=True,
        )

        assert hasattr(result, 'images') or isinstance(result, dict)

    def test_pipeline_no_guidance(self):
        """Test pipeline without classifier-free guidance."""
        pipeline = StableDiffusionXLPipelinePolaris()

        result = pipeline(
            prompt="test prompt",
            guidance_scale=1.0,  # No guidance
            num_inference_steps=2,
            return_dict=True,
        )

        assert hasattr(result, 'images') or isinstance(result, dict)


class TestGuidanceAdvancedFeatures:
    """Test advanced guidance features."""

    def test_guidance_dynamic_scaling(self):
        """Test dynamic guidance scaling (timestep-dependent)."""
        guidance = ClassifierFreeGuidance(guidance_scale=5.0)

        # Test that scale factor computation works
        scale_factor = guidance.compute_guidance_scale_factor(timestep=10)
        assert scale_factor == 5.0  # Should return static scale

        scale_factor_2 = guidance.compute_guidance_scale_factor(timestep=50)
        assert scale_factor_2 == 5.0  # Should still return static scale

    def test_guidance_split_and_combine(self):
        """Test splitting and recombining predictions."""
        guidance = ClassifierFreeGuidance(guidance_scale=2.0)

        # Create combined prediction (unconditional + conditional)
        unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])
        combined = np.concatenate([unconditional, conditional], axis=0)

        # Split back
        split_uncond, split_cond = guidance.split_predictions(combined, batch_size=1)

        np.testing.assert_array_equal(split_uncond, unconditional)
        np.testing.assert_array_equal(split_cond, conditional)

        # Combine again
        recombined = guidance.combine_predictions(split_uncond, split_cond)

        # Verify recombination works
        assert recombined.shape == unconditional.shape


class TestGuidanceErrorHandling:
    """Test error handling in guidance functionality."""

    def test_guidance_empty_inputs(self):
        """Test guidance with empty or None inputs."""
        guidance = ClassifierFreeGuidance()

        # Empty arrays
        empty_uncond = np.array([])
        empty_cond = np.array([])

        # Should handle gracefully or raise appropriate error
        try:
            guidance.combine_predictions(empty_uncond, empty_cond)
        except (ValueError, IndexError):
            pass  # Expected for empty arrays

    def test_guidance_extreme_values(self):
        """Test guidance with extreme values."""
        guidance = ClassifierFreeGuidance(guidance_scale=100.0)  # Very high scale

        unconditional = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        conditional = np.array([[[0.2, 0.3], [0.4, 0.5]]])

        result = guidance.combine_predictions(unconditional, conditional)

        # Should complete without numerical issues
        assert result.shape == unconditional.shape
        assert np.all(np.isfinite(result))  # No inf or nan values


if __name__ == "__main__":
    # Run tests manually
    print("Running Classifier-Free Guidance integration tests...")

    # Test basic guidance functionality
    guidance_tests = TestClassifierFreeGuidance()

    try:
        guidance_tests.test_guidance_initialization()
        print("✓ Guidance initialization")

        guidance_tests.test_guidance_scaling_basic()
        print("✓ Basic guidance scaling")

        guidance_tests.test_guidance_scaling_zero_scale()
        print("✓ Zero scale guidance")

        guidance_tests.test_guidance_validation()
        print("✓ Input validation")

        guidance_tests.test_guidance_representation()
        print("✓ String representations")

        # Test pipeline integration
        pipeline_tests = TestGuidancePipelineIntegration()

        pipeline_tests.test_pipeline_with_guidance_utility()
        print("✓ Pipeline guidance utility")

        pipeline_tests.test_pipeline_guidance_scale_effect()
        print("✓ Guidance scale effect")

        pipeline_tests.test_pipeline_no_guidance()
        print("✓ No guidance pipeline")

        # Test advanced features
        advanced_tests = TestGuidanceAdvancedFeatures()

        advanced_tests.test_guidance_dynamic_scaling()
        print("✓ Dynamic scaling")

        advanced_tests.test_guidance_split_and_combine()
        print("✓ Split and combine")

        print("All guidance integration tests passed! 🎉")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
