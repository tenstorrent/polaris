#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple test for SDXL pipeline to debug import issues.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    # Test basic imports using correct paths
    from workloads.diffusers.SDXLPipelinePolaris import SDXLPipelinePolarisWorkload
    print("✓ SDXLPipelinePolarisWorkload imported")

    from workloads.diffusers.schedulers.euler_discrete import EulerDiscreteScheduler
    print("✓ EulerDiscreteScheduler imported")

    from workloads.diffusers.ClassifierFreeGuidancePolaris import ClassifierFreeGuidance
    print("✓ ClassifierFreeGuidance imported")

    # Test other SDXL components
    from workloads.diffusers.UNet2DConditionModelPolaris import UNet2DConditionModelPolaris
    print("✓ UNet2DConditionModelPolaris imported")

    from workloads.diffusers.AutoencoderKLPolaris import AutoencoderKLPolaris
    print("✓ AutoencoderKLPolaris imported")

    print("All imports successful!")

except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()
