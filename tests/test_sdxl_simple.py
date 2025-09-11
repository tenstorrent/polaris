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
    # Test basic imports using relative imports
    from pipelines.pipeline_utils import PolarisDiffusionPipeline
    print("✓ PolarisDiffusionPipeline imported")

    from pipelines.schedulers import EulerDiscreteScheduler
    print("✓ EulerDiscreteScheduler imported")

    from pipelines.models import BaseOnnxComponent
    print("✓ BaseOnnxComponent imported")

    # Test SDXL imports
    from pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
    print("✓ StableDiffusionXLPipelineOutput imported")

    from pipelines.stable_diffusion_xl import pipeline_stable_diffusion_xl
    print("✓ SDXL pipeline module imported")

    print("All imports successful!")

except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"Other error: {e}")
    import traceback
    traceback.print_exc()
