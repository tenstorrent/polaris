#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Complete SDXL Pipeline Test Workload for Polaris.

This workload demonstrates the full Stable Diffusion XL pipeline functionality
including text encoding, UNet denoising, VAE decoding, and classifier-free guidance.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import pipeline components
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.stable_diffusion_xl import StableDiffusionXLPipelinePolaris
from pipelines.schedulers import EulerDiscreteScheduler


class SDXLPipelineTest:
    """
    Complete test workload for SDXL pipeline functionality.

    This class provides comprehensive testing of the SDXL pipeline including:
    - Text-to-image generation with various parameters
    - Classifier-free guidance testing
    - Multiple prompt handling
    - Performance benchmarking
    - Serialization/deserialization testing
    """

    def __init__(self):
        """Initialize the SDXL pipeline test workload."""
        self.pipeline = None
        self.test_results = {}

    def setup_pipeline(self, **kwargs):
        """
        Set up the SDXL pipeline with specified configuration.

        Args:
            **kwargs: Pipeline configuration options
                - scheduler: Scheduler to use (default: EulerDiscreteScheduler)
                - guidance_scale: CFG scale (default: 7.5)
                - guidance_rescale: CFG rescale factor (default: 0.0)
        """
        print("Setting up SDXL Pipeline...")

        # Create scheduler
        scheduler = kwargs.get('scheduler', EulerDiscreteScheduler())

        # Create pipeline
        self.pipeline = StableDiffusionXLPipelinePolaris(
            scheduler=scheduler
        )

        # Configure guidance if specified
        guidance_scale = kwargs.get('guidance_scale', 7.5)
        guidance_rescale = kwargs.get('guidance_rescale', 0.0)

        print(f"✓ Pipeline created with scheduler: {scheduler.__class__.__name__}")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Guidance rescale: {guidance_rescale}")

        return self.pipeline

    def run_basic_generation_test(self, **kwargs):
        """
        Run basic text-to-image generation test.

        Args:
            **kwargs: Generation parameters
                - prompt: Text prompt (default: "a beautiful landscape")
                - height: Image height (default: 256)
                - width: Image width (default: 256)
                - num_inference_steps: Number of denoising steps (default: 5)
                - guidance_scale: CFG scale (default: 7.5)
        """
        print("\n=== Basic Generation Test ===")

        if self.pipeline is None:
            self.setup_pipeline()

        # Default test parameters
        prompt = kwargs.get('prompt', "a beautiful landscape at sunset")
        height = kwargs.get('height', 256)
        width = kwargs.get('width', 256)
        num_inference_steps = kwargs.get('num_inference_steps', 5)
        guidance_scale = kwargs.get('guidance_scale', 7.5)

        print(f"Generating image with prompt: '{prompt}'")
        print(f"Image size: {height}x{width}")
        print(f"Inference steps: {num_inference_steps}")
        print(f"Guidance scale: {guidance_scale}")

        try:
            # Generate image
            result = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                return_dict=True,
            )

            # Validate results
            if hasattr(result, 'images') and result.images is not None:
                print("✓ Generation successful!")
                print(f"  Generated {result.image_count} image(s)")
                print(f"  Image shape: {result.image_shape}")

                # Store test results
                self.test_results['basic_generation'] = {
                    'success': True,
                    'image_count': result.image_count,
                    'image_shape': result.image_shape,
                    'prompt': prompt,
                    'parameters': {
                        'height': height,
                        'width': width,
                        'steps': num_inference_steps,
                        'guidance_scale': guidance_scale
                    }
                }
                return True
            else:
                print("❌ Generation failed - no images returned")
                self.test_results['basic_generation'] = {'success': False, 'error': 'No images returned'}
                return False

        except Exception as e:
            print(f"❌ Generation failed with error: {e}")
            self.test_results['basic_generation'] = {'success': False, 'error': str(e)}
            return False

    def run_guidance_comparison_test(self):
        """Test different guidance scales to show the effect of CFG."""
        print("\n=== Guidance Scale Comparison Test ===")

        if self.pipeline is None:
            self.setup_pipeline()

        guidance_scales = [1.0, 3.0, 7.5, 15.0]
        prompt = "a futuristic city at night"

        print(f"Testing guidance scales with prompt: '{prompt}'")

        results = {}
        for scale in guidance_scales:
            print(f"\n--- Guidance Scale: {scale} ---")
            try:
                result = self.pipeline(
                    prompt=prompt,
                    height=128,
                    width=128,
                    num_inference_steps=3,
                    guidance_scale=scale,
                    return_dict=True,
                )

                if hasattr(result, 'images'):
                    print(f"✓ Success - Generated {result.image_count} image(s)")
                    results[scale] = {'success': True, 'shape': result.image_shape}
                else:
                    print("❌ Failed - No images generated")
                    results[scale] = {'success': False}

            except Exception as e:
                print(f"❌ Failed with error: {e}")
                results[scale] = {'success': False, 'error': str(e)}

        self.test_results['guidance_comparison'] = results
        successful_scales = sum(1 for r in results.values() if r.get('success', False))
        print(f"\n✓ Guidance comparison test completed: {successful_scales}/{len(guidance_scales)} scales successful")

    def run_multiple_prompts_test(self):
        """Test generation with multiple prompts."""
        print("\n=== Multiple Prompts Test ===")

        if self.pipeline is None:
            self.setup_pipeline()

        prompts = [
            "a serene mountain landscape",
            "a bustling city street",
            "an abstract geometric pattern"
        ]

        print(f"Testing with {len(prompts)} prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. '{prompt}'")

        try:
            result = self.pipeline(
                prompt=prompts,
                height=128,
                width=128,
                num_inference_steps=3,
                guidance_scale=5.0,
                return_dict=True,
            )

            if hasattr(result, 'images'):
                print(f"✓ Success - Generated {result.image_count} image(s) for {len(prompts)} prompts")
                self.test_results['multiple_prompts'] = {
                    'success': True,
                    'prompt_count': len(prompts),
                    'image_count': result.image_count,
                    'image_shape': result.image_shape
                }
                return True
            else:
                print("❌ Failed - No images generated")
                self.test_results['multiple_prompts'] = {'success': False}
                return False

        except Exception as e:
            print(f"❌ Failed with error: {e}")
            self.test_results['multiple_prompts'] = {'success': False, 'error': str(e)}
            return False

    def run_serialization_test(self, **kwargs):
        """Test pipeline serialization and loading."""
        print("\n=== Serialization Test ===")

        if self.pipeline is None:
            self.setup_pipeline()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "sdxl_test_pipeline"

            try:
                # Save pipeline
                print("Saving pipeline...")
                self.pipeline.save_pretrained(save_path)
                print("✓ Pipeline saved successfully")

                # Load pipeline
                print("Loading pipeline...")
                from pipelines.stable_diffusion_xl import StableDiffusionXLPipelinePolaris
                loaded_pipeline = StableDiffusionXLPipelinePolaris.from_pretrained(save_path)
                print("✓ Pipeline loaded successfully")

                # Test loaded pipeline
                print("Testing loaded pipeline...")
                result = loaded_pipeline(
                    prompt="test after loading",
                    height=64,
                    width=64,
                    num_inference_steps=2,
                    guidance_scale=3.0,
                    return_dict=True,
                )

                if hasattr(result, 'images'):
                    print("✓ Loaded pipeline works correctly")
                    self.test_results['serialization'] = {
                        'success': True,
                        'save_success': True,
                        'load_success': True,
                        'execution_success': True
                    }
                    return True
                else:
                    print("❌ Loaded pipeline execution failed")
                    self.test_results['serialization'] = {
                        'success': False,
                        'error': 'Loaded pipeline execution failed'
                    }
                    return False

            except Exception as e:
                print(f"❌ Serialization test failed: {e}")
                self.test_results['serialization'] = {'success': False, 'error': str(e)}
                return False

    def run_performance_test(self, **kwargs):
        """Run basic performance test."""
        print("\n=== Performance Test ===")

        if self.pipeline is None:
            self.setup_pipeline()

        import time

        prompt = kwargs.get('prompt', "performance test image")
        num_runs = kwargs.get('num_runs', 3)
        height = kwargs.get('height', 128)
        width = kwargs.get('width', 128)
        steps = kwargs.get('steps', 5)

        print(f"Running {num_runs} generations...")
        print(f"Parameters: {height}x{width}, {steps} steps")

        times = []
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...")
            start_time = time.time()

            try:
                result = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=5.0,
                    return_dict=True,
                )
                end_time = time.time()
                times.append(end_time - start_time)
                print(".2f")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                times.append(None)

        successful_runs = sum(1 for t in times if t is not None)
        if successful_runs > 0:
            avg_time = sum(t for t in times if t is not None) / successful_runs
            print(".2f")
            self.test_results['performance'] = {
                'success': True,
                'successful_runs': successful_runs,
                'total_runs': num_runs,
                'average_time': avg_time,
                'times': times
            }
        else:
            print("❌ All performance test runs failed")
            self.test_results['performance'] = {'success': False}

    def run_all_tests(self):
        """Run all available tests."""
        print("🚀 Starting SDXL Pipeline Test Suite")
        print("=" * 50)

        # Run all tests
        self.run_basic_generation_test()
        self.run_guidance_comparison_test()
        self.run_multiple_prompts_test()
        self.run_serialization_test()
        self.run_performance_test()

        # Print summary
        self.print_test_summary()

    def print_test_summary(self):
        """Print a summary of all test results."""
        print("\n" + "=" * 50)
        print("📊 SDXL Pipeline Test Summary")
        print("=" * 50)

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values()
                             if isinstance(result, dict) and result.get('success', False))

        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")

        if successful_tests == total_tests:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed - check details above")

        # Print detailed results
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅" if result.get('success', False) else "❌"
            print(f"  {status} {test_name.replace('_', ' ').title()}")


def main():
    """Main function to run the SDXL pipeline test workload."""
    print("Polaris SDXL Pipeline Test Workload")
    print("===================================")

    # Create test instance
    test_workload = SDXLPipelineTest()

    # Run all tests
    test_workload.run_all_tests()

    print("\n" + "=" * 50)
    print("Test workload completed!")
    print("Check the results above for detailed performance metrics.")


if __name__ == "__main__":
    main()
