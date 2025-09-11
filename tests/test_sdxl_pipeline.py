#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Stable Diffusion XL Pipeline in Polaris.

Tests the complete SDXL pipeline functionality including:
- Pipeline initialization and component registration
- Text encoding with mock tokenizers
- Diffusion process with mock components
- Output validation and formatting
- Integration with Polaris graph system
"""

import tempfile
import os
from pathlib import Path
import json

import pytest
import numpy as np

# Import Polaris SDXL pipeline
from pipelines.stable_diffusion_xl import (
    StableDiffusionXLPipelinePolaris,
    StableDiffusionXLPipelineOutput
)
from pipelines.schedulers import EulerDiscreteScheduler
from pipelines.models import (
    UNet2DConditionModelOnnx,
    AutoencoderKLOnnx,
    CLIPTextModelOnnx,
    CLIPTextModelWithProjectionOnnx,
    CLIPTokenizerHost
)


class TestStableDiffusionXLPipeline:
    """Test the Stable Diffusion XL Pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with mock components."""
        # Create mock components
        scheduler = EulerDiscreteScheduler()

        # Initialize pipeline
        pipeline = StableDiffusionXLPipelinePolaris(
            scheduler=scheduler,
        )

        # Verify initialization
        assert pipeline.scheduler == scheduler
        assert pipeline.config is not None
        assert "scheduler" in pipeline.components

    def test_pipeline_with_all_components(self):
        """Test pipeline with all SDXL components."""
        # Create mock components
        scheduler = EulerDiscreteScheduler()
        tokenizer = CLIPTokenizerHost()
        tokenizer_2 = CLIPTokenizerHost()

        # Create pipeline with all components
        pipeline = StableDiffusionXLPipelinePolaris(
            scheduler=scheduler,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
        )

        # Verify all components are registered
        components = pipeline.components
        assert components["scheduler"] == scheduler
        assert components["tokenizer"] == tokenizer
        assert components["tokenizer_2"] == tokenizer_2

    def test_text_encoding(self):
        """Test text encoding functionality."""
        pipeline = StableDiffusionXLPipelinePolaris()
        tokenizer = CLIPTokenizerHost()
        pipeline.tokenizer = tokenizer
        pipeline.tokenizer_2 = tokenizer

        # Test encoding with prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline._encode_prompt(
            prompt="a beautiful landscape",
            negative_prompt="blurry",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        # Verify output shapes
        assert prompt_embeds.shape[0] == 1  # batch_size
        assert negative_prompt_embeds.shape[0] == 1
        assert pooled_prompt_embeds.shape[0] == 1
        assert negative_pooled_prompt_embeds.shape[0] == 1

        # Verify embeddings are numpy arrays
        assert isinstance(prompt_embeds, np.ndarray)
        assert isinstance(pooled_prompt_embeds, np.ndarray)

    def test_prepare_latents(self):
        """Test latent preparation."""
        pipeline = StableDiffusionXLPipelinePolaris()
        scheduler = EulerDiscreteScheduler()
        pipeline.scheduler = scheduler

        # Test latent preparation
        latents = pipeline.prepare_latents(
            batch_size=2,
            num_channels_latents=4,
            height=128,
            width=128,
            dtype=np.float32,
        )

        # Verify shape and type
        assert latents.shape == (2, 4, 128, 128)
        assert latents.dtype == np.float32

    def test_pipeline_call_basic(self):
        """Test basic pipeline __call__ method."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Test with minimal parameters
        result = pipeline(
            prompt="test prompt",
            height=512,
            width=512,
            num_inference_steps=2,  # Very few steps for testing
            guidance_scale=1.0,  # Disable guidance for simplicity
            return_dict=True,
        )

        # Verify result
        assert isinstance(result, StableDiffusionXLPipelineOutput)
        assert hasattr(result, 'images')
        assert result.image_count > 0

    def test_pipeline_call_with_guidance(self):
        """Test pipeline with classifier-free guidance."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Test with guidance
        result = pipeline(
            prompt="a beautiful sunset",
            negative_prompt="dark, blurry",
            height=512,
            width=512,
            num_inference_steps=3,
            guidance_scale=7.5,
            return_dict=True,
        )

        # Verify result
        assert isinstance(result, StableDiffusionXLPipelineOutput)
        assert result.image_count == 1

    def test_pipeline_call_multiple_prompts(self):
        """Test pipeline with multiple prompts."""
        pipeline = StableDiffusionXLPipelinePolaris()

        prompts = ["sunset over mountains", "ocean waves"]
        result = pipeline(
            prompt=prompts,
            height=512,
            width=512,
            num_inference_steps=2,
            return_dict=True,
        )

        # Verify multiple images generated
        assert result.image_count == len(prompts)

    def test_pipeline_call_multiple_images_per_prompt(self):
        """Test generating multiple images per prompt."""
        pipeline = StableDiffusionXLPipelinePolaris()

        result = pipeline(
            prompt="a cat",
            height=256,
            width=256,
            num_inference_steps=2,
            num_images_per_prompt=3,
            return_dict=True,
        )

        # Verify multiple images per prompt
        assert result.image_count == 3

    def test_pipeline_output_validation(self):
        """Test pipeline output validation."""
        # Create mock output
        images = np.random.randn(1, 3, 512, 512).astype(np.float32)
        output = StableDiffusionXLPipelineOutput(images=images)

        # Test validation
        assert output.image_count == 1
        print(f"Actual image shape: {output.image_shape}")
        assert output.image_shape == (1, 3, 512, 512)

        # Test invalid shape should raise error
        with pytest.raises(ValueError):
            invalid_images = np.random.randn(1, 3, 512)  # Missing one dimension
            StableDiffusionXLPipelineOutput(images=invalid_images)

    def test_pipeline_output_to_pil(self):
        """Test conversion to PIL images."""
        # Create mock output with proper image range
        images = np.random.randn(1, 3, 64, 64).astype(np.float32)
        output = StableDiffusionXLPipelineOutput(images=images)

        # Test PIL conversion
        try:
            pil_images = output.to_pil()
            assert len(pil_images) == 1
        except ImportError:
            pytest.skip("PIL not available")

    def test_pipeline_with_scheduler(self):
        """Test pipeline with Euler scheduler."""
        scheduler = EulerDiscreteScheduler()
        pipeline = StableDiffusionXLPipelinePolaris(scheduler=scheduler)

        # Test pipeline execution
        try:
            result = pipeline(
                prompt="test",
                height=256,
                width=256,
                num_inference_steps=5,
                guidance_scale=1.0,  # Disable guidance to avoid concatenation issues
                return_dict=True,
            )
            assert isinstance(result, StableDiffusionXLPipelineOutput)
        except Exception as e:
            print(f"Pipeline with scheduler failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_pipeline_save_load_config(self):
        """Test pipeline configuration save and load."""
        pipeline = StableDiffusionXLPipelinePolaris()
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

    def test_pipeline_components_property(self):
        """Test pipeline components property."""
        pipeline = StableDiffusionXLPipelinePolaris()
        scheduler = EulerDiscreteScheduler()
        pipeline.scheduler = scheduler

        components = pipeline.components
        assert "scheduler" in components
        assert components["scheduler"] == scheduler


class TestSDXLPipelineIntegration:
    """Test SDXL pipeline integration with mock ONNX components."""

    def test_pipeline_with_mock_unet(self):
        """Test pipeline with mock UNet component."""
        # Create pipeline
        pipeline = StableDiffusionXLPipelinePolaris()

        # Create mock UNet (this would normally load from ONNX)
        # For testing, we just ensure the pipeline can handle it
        pipeline.unet = "mock_unet"  # Mock component

        # Test that pipeline can still run
        result = pipeline(
            prompt="test",
            num_inference_steps=1,
            return_dict=True,
        )

        assert isinstance(result, StableDiffusionXLPipelineOutput)

    def test_pipeline_with_mock_vae(self):
        """Test pipeline with mock VAE component."""
        pipeline = StableDiffusionXLPipelinePolaris()
        # Create a mock VAE that returns the input latents as mock images
        class MockVAE:
            def decode(self, latents):
                # Convert latents to mock RGB images
                if hasattr(latents, 'shape'):
                    batch_size, latent_channels, height, width = latents.shape
                    # Convert to image space (mock VAE decode)
                    image_height = height * 8  # VAE upsampling factor
                    image_width = width * 8
                    images = np.random.randn(batch_size, 3, image_height, image_width).astype(np.float32)
                    return {"sample": images}
                return {"sample": latents}

        pipeline.vae = MockVAE()

        result = pipeline(
            prompt="test",
            num_inference_steps=1,
            return_dict=True,
        )

        assert isinstance(result, StableDiffusionXLPipelineOutput)

    def test_pipeline_with_mock_text_encoders(self):
        """Test pipeline with mock text encoders."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Create mock text encoders
        class MockTextEncoder:
            def __call__(self, input_ids, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
                seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1 else 77
                embed_dim = 768
                return {
                    "last_hidden_state": np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
                }

        class MockTextEncoder2:
            def __call__(self, input_ids, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
                seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1 else 77
                embed_dim = 1280
                return {
                    "last_hidden_state": np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32),
                    "text_embeds": np.random.randn(batch_size, embed_dim).astype(np.float32)
                }

        pipeline.text_encoder = MockTextEncoder()
        pipeline.text_encoder_2 = MockTextEncoder2()

        result = pipeline(
            prompt="test",
            num_inference_steps=1,
            return_dict=True,
        )

        assert isinstance(result, StableDiffusionXLPipelineOutput)


class TestSDXLPipelineErrors:
    """Test error handling in SDXL pipeline."""

    def test_invalid_prompt_type(self):
        """Test error handling for invalid prompt types."""
        pipeline = StableDiffusionXLPipelinePolaris()

        # Should handle non-string prompts gracefully
        result = pipeline(
            prompt=123,  # Invalid prompt type
            num_inference_steps=1,
            return_dict=True,
        )

        assert isinstance(result, StableDiffusionXLPipelineOutput)

    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        pipeline = StableDiffusionXLPipelinePolaris()

        result = pipeline(
            prompt="",
            num_inference_steps=1,
            return_dict=True,
        )

        assert isinstance(result, StableDiffusionXLPipelineOutput)


if __name__ == "__main__":
    # Run tests manually
    print("Running SDXL Pipeline tests...")

    # Basic pipeline tests
    test_instance = TestStableDiffusionXLPipeline()

    try:
        test_instance.test_pipeline_initialization()
        print("✓ Pipeline initialization")

        test_instance.test_pipeline_with_all_components()
        print("✓ Pipeline with all components")

        test_instance.test_text_encoding()
        print("✓ Text encoding")

        test_instance.test_prepare_latents()
        print("✓ Prepare latents")

        test_instance.test_pipeline_call_basic()
        print("✓ Basic pipeline call")

        test_instance.test_pipeline_call_with_guidance()
        print("✓ Pipeline call with guidance")

        test_instance.test_pipeline_call_multiple_prompts()
        print("✓ Multiple prompts")

        test_instance.test_pipeline_call_multiple_images_per_prompt()
        print("✓ Multiple images per prompt")

        test_instance.test_pipeline_output_validation()
        print("✓ Output validation")

        test_instance.test_pipeline_with_scheduler()
        print("✓ Pipeline with scheduler")

        test_instance.test_pipeline_save_load_config()
        print("✓ Save/load config")

        test_instance.test_pipeline_components_property()
        print("✓ Components property")

        # Integration tests
        integration_tests = TestSDXLPipelineIntegration()
        integration_tests.test_pipeline_with_mock_unet()
        print("✓ Mock UNet integration")

        integration_tests.test_pipeline_with_mock_vae()
        print("✓ Mock VAE integration")

        integration_tests.test_pipeline_with_mock_text_encoders()
        print("✓ Mock text encoders integration")

        print("All SDXL pipeline tests passed! 🎉")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
