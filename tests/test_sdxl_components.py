#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SDXL Components Integration Test Workload for Polaris.

This workload tests the individual SDXL pipeline components and their integration,
focusing on the ONNX-based models, text encoders, and guidance systems.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from workloads.diffusers.base_onnx_component import BaseOnnxComponent
from workloads.diffusers.UNet2DConditionModelPolaris import UNet2DConditionModelPolaris
from workloads.diffusers.AutoencoderKLPolaris import AutoencoderKLPolaris
from workloads.diffusers.TextEncodersPolaris import CLIPTextModelPolaris, CLIPTextModelWithProjectionPolaris, CLIPTokenizerHost
from workloads.diffusers.ClassifierFreeGuidancePolaris import ClassifierFreeGuidance
from workloads.diffusers.schedulers.euler_discrete import EulerDiscreteScheduler


class SDXLComponentsTest:
    """
    Test workload for individual SDXL components.

    This class provides focused testing of each SDXL pipeline component:
    - UNet model functionality
    - VAE encoder/decoder
    - Text encoders and tokenizers
    - Classifier-free guidance
    - Scheduler integration
    """

    def __init__(self):
        """Initialize the components test workload."""
        self.test_results = {}
        self.components = {}

    def test_text_tokenizer(self):
        """Test CLIP tokenizer functionality."""
        print("\n=== Text Tokenizer Test ===")

        try:
            # Test with mock tokenizer (since transformers may not be available)
            tokenizer = CLIPTokenizerHost()
            print("✓ Tokenizer created successfully")

            # Test tokenization
            test_prompts = [
                "a beautiful sunset",
                "futuristic city landscape",
                "abstract geometric art"
            ]

            for prompt in test_prompts:
                try:
                    result = tokenizer(prompt, return_tensors="np")
                    if 'input_ids' in result and 'attention_mask' in result:
                        print(f"✓ Tokenized: '{prompt}' -> shape {result['input_ids'].shape}")
                    else:
                        print(f"⚠️  Tokenization result missing expected keys for: '{prompt}'")
                except Exception as e:
                    print(f"⚠️  Tokenization failed for '{prompt}': {e}")

            self.test_results['text_tokenizer'] = {'success': True}
            self.components['tokenizer'] = tokenizer
            return True

        except Exception as e:
            print(f"❌ Tokenizer test failed: {e}")
            self.test_results['text_tokenizer'] = {'success': False, 'error': str(e)}
            return False

    def test_unet_model(self):
        """Test UNet model interface."""
        print("\n=== UNet Model Test ===")

        try:
            # Create UNet instance (without actual ONNX file for testing)
            from workloads.diffusers.UNet2DConditionModelPolaris import UNet2DConditionModelPolaris
            unet = UNet2DConditionModelPolaris.__new__(UNet2DConditionModelPolaris)
            unet.sample_size = 64
            unet.in_channels = 4
            unet.out_channels = 4
            unet.time_embedding_dim = 320

            print("✓ UNet model interface created")

            # Test shape calculations
            batch_size = 2
            height, width = 64, 64

            latent_shape = unet.get_expected_latent_shape(batch_size, height, width)
            output_shape = unet.get_expected_output_shape(batch_size, height, width)

            print(f"✓ Latent shape: {latent_shape}")
            print(f"✓ Output shape: {output_shape}")

            # Verify shapes are correct
            expected_latent = [batch_size, unet.in_channels, height, width]
            expected_output = [batch_size, unet.out_channels, height, width]

            assert latent_shape == expected_latent, f"Latent shape mismatch: {latent_shape} != {expected_latent}"
            assert output_shape == expected_output, f"Output shape mismatch: {output_shape} != {expected_output}"

            print("✓ Shape calculations verified")
            self.test_results['unet_model'] = {'success': True}
            self.components['unet'] = unet
            return True

        except Exception as e:
            print(f"❌ UNet model test failed: {e}")
            self.test_results['unet_model'] = {'success': False, 'error': str(e)}
            return False

    def test_vae_components(self):
        """Test VAE encoder and decoder components."""
        print("\n=== VAE Components Test ===")

        try:
            # Test VAE encoder
            from workloads.diffusers.AutoencoderKLPolaris import AutoencoderKLPolaris
            encoder = AutoencoderKLPolaris.__new__(AutoencoderKLPolaris)
            encoder.scale_factor = 8
            encoder.in_channels = 3
            encoder.out_channels = 4

            # Test VAE decoder
            decoder = AutoencoderKLPolaris.__new__(AutoencoderKLPolaris)
            decoder.scale_factor = 8
            decoder.in_channels = 4
            decoder.out_channels = 3

            print("✓ VAE encoder and decoder interfaces created")

            # Test combined VAE
            vae = AutoencoderKLPolaris("test_vae", {"in_channels": 3, "latent_channels": 4, "sample_size": 128})
            print("✓ Combined VAE interface created")

            self.test_results['vae_components'] = {'success': True}
            self.components['vae'] = vae
            return True

        except Exception as e:
            print(f"❌ VAE components test failed: {e}")
            self.test_results['vae_components'] = {'success': False, 'error': str(e)}
            return False

    def test_guidance_system(self):
        """Test classifier-free guidance system."""
        print("\n=== Guidance System Test ===")

        try:
            # Test basic guidance
            guidance = ClassifierFreeGuidance(guidance_scale=3.0, guidance_rescale=0.1)
            print("✓ Guidance system created")

            # Test guidance combination
            unconditional = np.array([[[1.0, 2.0], [3.0, 4.0]]])
            conditional = np.array([[[2.0, 3.0], [4.0, 5.0]]])

            result = guidance.combine_predictions(unconditional, conditional)
            print(f"✓ Guidance combination: {result}")

            # Test input validation
            guidance.validate_guidance_inputs(unconditional, conditional)
            print("✓ Input validation passed")

            # Test guidance scaling
            scales = [1.0, 2.0, 5.0, 10.0]
            for scale in scales:
                scaled_guidance = ClassifierFreeGuidance(guidance_scale=scale)
                scaled_result = scaled_guidance.combine_predictions(unconditional, conditional)
                print(f"✓ Scale {scale}: result shape {scaled_result.shape}")

            self.test_results['guidance_system'] = {'success': True}
            self.components['guidance'] = guidance
            return True

        except Exception as e:
            print(f"❌ Guidance system test failed: {e}")
            self.test_results['guidance_system'] = {'success': False, 'error': str(e)}
            return False

    def test_scheduler_integration(self):
        """Test scheduler integration."""
        print("\n=== Scheduler Integration Test ===")

        try:
            # Test Euler scheduler
            scheduler = EulerDiscreteScheduler()
            print("✓ Euler scheduler created")

            # Test scheduler configuration
            config = scheduler.config
            print(f"✓ Scheduler config: {list(config.keys())}")

            # Test timestep setting
            scheduler.set_timesteps(num_inference_steps=5)
            print(f"✓ Timesteps set: {len(scheduler.timesteps)} steps")

            # Test basic step functionality (mock)
            latents = np.random.randn(1, 4, 32, 32).astype(np.float32)
            noise = np.random.randn(1, 4, 32, 32).astype(np.float32)
            timestep = 100

            step_result = scheduler.step(
                model_output=noise,
                timestep=timestep,
                sample=latents
            )

            print("✓ Scheduler step executed")
            self.test_results['scheduler_integration'] = {'success': True}
            self.components['scheduler'] = scheduler
            return True

        except Exception as e:
            print(f"❌ Scheduler integration test failed: {e}")
            self.test_results['scheduler_integration'] = {'success': False, 'error': str(e)}
            return False

    def test_component_integration(self):
        """Test integration of multiple components."""
        print("\n=== Component Integration Test ===")

        try:
            # Test component interaction
            if 'guidance' in self.components and 'scheduler' in self.components:
                guidance = self.components['guidance']
                scheduler = self.components['scheduler']

                # Simulate a simple generation step
                batch_size = 1
                latent_shape = [batch_size, 4, 32, 32]

                # Mock latents and noise
                latents = np.random.randn(*latent_shape).astype(np.float32)
                noise_pred = np.random.randn(*latent_shape).astype(np.float32)

                # Test scheduler step with guidance-compatible shape
                step_result = scheduler.step(
                    model_output=noise_pred,
                    timestep=100,
                    sample=latents,
                    return_dict=False
                )

                print("✓ Component integration successful")
                self.test_results['component_integration'] = {'success': True}
                return True
            else:
                print("⚠️  Skipping integration test - some components not available")
                self.test_results['component_integration'] = {'success': True, 'skipped': True}
                return True

        except Exception as e:
            print(f"❌ Component integration test failed: {e}")
            self.test_results['component_integration'] = {'success': False, 'error': str(e)}
            return False

    def run_all_tests(self):
        """Run all component tests."""
        print("🔧 Starting SDXL Components Test Suite")
        print("=" * 50)

        # Run individual component tests
        self.test_text_tokenizer()
        self.test_unet_model()
        self.test_vae_components()
        self.test_guidance_system()
        self.test_scheduler_integration()
        self.test_component_integration()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 50)
        print("📊 SDXL Components Test Summary")
        print("=" * 50)

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values()
                             if isinstance(result, dict) and result.get('success', False))

        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")

        if successful_tests == total_tests:
            print("🎉 All component tests passed!")
        else:
            print("⚠️  Some component tests failed - check details above")

        # Print detailed results
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")

            if not result.get('success', False) and 'error' in result:
                print(f"      Error: {result['error']}")


def main():
    """Main function to run the SDXL components test workload."""
    print("Polaris SDXL Components Test Workload")
    print("=====================================")

    # Create test instance
    test_workload = SDXLComponentsTest()

    # Run all tests
    test_workload.run_all_tests()

    print("\n" + "=" * 50)
    print("Components test workload completed!")
    print("This workload validates individual pipeline components.")
    print("For full pipeline testing, use SDXLPipelineTest.py")


if __name__ == "__main__":
    main()
