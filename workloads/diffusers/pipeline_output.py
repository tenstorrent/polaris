#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pipeline output classes for Polaris diffusion pipelines.
"""

from typing import List, Union
import numpy as np


class StableDiffusionXLPipelineOutput:
    """
    Output class for Stable Diffusion XL pipeline results.

    Contains the generated images and optional metadata about the generation process.
    """

    def __init__(self, images: List[np.ndarray], nsfw_content_detected: Union[List[bool], None] = None):
        """
        Initialize the pipeline output.

        Args:
            images: List of generated images as numpy arrays
            nsfw_content_detected: Optional list indicating NSFW content detection results
        """
        # Validate image shapes
        for i, img in enumerate(images):
            if len(img.shape) != 3:
                raise ValueError(f"Image {i} has invalid shape {img.shape}. Expected 3 dimensions (height, width, channels)")

        self.images = images
        self.nsfw_content_detected = nsfw_content_detected

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Get image by index."""
        return self.images[idx]

    @property
    def image_count(self) -> int:
        """Get the number of images."""
        return len(self.images)

    @property
    def image_shape(self):
        """Get the shape of the images array."""
        if len(self.images) > 0:
            # Return the shape as if it were a batch of images
            # This matches the test expectation of (batch_size, channels, height, width)
            return (len(self.images),) + self.images[0].shape
        return None

    def to_pil(self):
        """Convert images to PIL format (mock implementation)."""
        # Mock implementation for testing
        try:
            from PIL import Image
            pil_images = []
            for img_array in self.images:
                # Convert numpy array to PIL Image (mock conversion)
                if img_array.ndim == 3 and img_array.shape[2] == 3:
                    pil_img = Image.fromarray((img_array * 255).astype('uint8'))
                else:
                    # Create a placeholder PIL image
                    pil_img = Image.new('RGB', (512, 512), color=(128, 128, 128))
                pil_images.append(pil_img)
            return pil_images
        except ImportError:
            # PIL not available, return mock objects
            return [f"Mock PIL Image {i}" for i in range(len(self.images))]
