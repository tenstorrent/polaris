# Copyright 2025 The HuggingFace Team and Polaris Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from typing import Optional, Tuple, Union

from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.tensor import SimTensor


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance utility for Polaris diffusion pipelines.

    This class implements the classifier-free guidance technique that combines
    conditional and unconditional diffusion predictions to guide the generation
    process. It provides methods to compute guidance scaling and rescaling.

    Key Features:
    - Combine conditional and unconditional predictions
    - Apply guidance scaling and rescaling
    - Graph-based operations for Polaris integration
    - Support for various guidance configurations
    """

    def __init__(self, guidance_scale: float = 7.5, guidance_rescale: float = 0.0):
        """
        Initialize Classifier-Free Guidance.

        Args:
            guidance_scale: Scale factor for guidance (typically 1.0-20.0)
            guidance_rescale: Rescale factor to mitigate over-guidance (0.0-1.0)
        """
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale

    def combine_predictions(
        self,
        unconditional_prediction: Union[np.ndarray, SimTensor],
        conditional_prediction: Union[np.ndarray, SimTensor]
    ) -> Union[np.ndarray, SimTensor]:
        """
        Combine unconditional and conditional predictions using guidance.

        Args:
            unconditional_prediction: Prediction from unconditional model
            conditional_prediction: Prediction from conditional model

        Returns:
            Combined prediction tensor
        """
        # Apply guidance: prediction = uncond + guidance_scale * (cond - uncond)
        guided_prediction = unconditional_prediction + self.guidance_scale * (
            conditional_prediction - unconditional_prediction
        )

        # Apply guidance rescaling if specified
        if self.guidance_rescale > 0.0:
            guided_prediction = self._apply_guidance_rescale(
                unconditional_prediction, conditional_prediction, guided_prediction
            )

        return guided_prediction

    def _apply_guidance_rescale(
        self,
        unconditional_prediction: Union[np.ndarray, SimTensor],
        conditional_prediction: Union[np.ndarray, SimTensor],
        guided_prediction: Union[np.ndarray, SimTensor]
    ) -> Union[np.ndarray, SimTensor]:
        """
        Apply guidance rescaling to mitigate over-guidance artifacts.

        Args:
            unconditional_prediction: Original unconditional prediction
            conditional_prediction: Original conditional prediction
            guided_prediction: Current guided prediction

        Returns:
            Rescaled guided prediction
        """
        # Compute the standard deviation of the original predictions
        if isinstance(unconditional_prediction, np.ndarray):
            std_uncond = np.std(unconditional_prediction, axis=None, keepdims=True)
            std_cond = np.std(conditional_prediction, axis=None, keepdims=True)
            std_guided = np.std(guided_prediction, axis=None, keepdims=True)

            # Rescale based on standard deviation ratio
            if std_guided > 0:
                rescale_factor = (std_uncond + std_cond) / (2 * std_guided)
                guided_prediction = unconditional_prediction + rescale_factor * (
                    guided_prediction - unconditional_prediction
                )
        else:
            # For SimTensor, we would need to implement this as graph operations
            # For now, return the original guided prediction
            pass

        return guided_prediction

    def split_predictions(
        self,
        combined_prediction: Union[np.ndarray, SimTensor],
        batch_size: int
    ) -> Tuple[Union[np.ndarray, SimTensor], Union[np.ndarray, SimTensor]]:
        """
        Split combined predictions back into unconditional and conditional parts.

        Args:
            combined_prediction: Combined prediction tensor
            batch_size: Original batch size (before duplication)

        Returns:
            Tuple of (unconditional_prediction, conditional_prediction)
        """
        if isinstance(combined_prediction, np.ndarray):
            # Split along batch dimension
            unconditional_prediction = combined_prediction[:batch_size]
            conditional_prediction = combined_prediction[batch_size:2*batch_size]
        else:
            # For SimTensor, this would require graph operations
            # Return placeholders for now
            unconditional_prediction = combined_prediction
            conditional_prediction = combined_prediction

        return unconditional_prediction, conditional_prediction

    def compute_guidance_scale_factor(self, timestep: Optional[Union[float, int]] = None) -> float:
        """
        Compute dynamic guidance scale factor (can be extended for timestep-dependent scaling).

        Args:
            timestep: Current timestep (optional, for dynamic scaling)

        Returns:
            Guidance scale factor
        """
        # For now, return the static guidance scale
        # This can be extended to implement dynamic scaling based on timestep
        return self.guidance_scale

    def add_guidance_to_graph(
        self,
        graph: WorkloadGraph,
        unconditional_tensor_name: str,
        conditional_tensor_name: str,
        output_tensor_name: str,
        guidance_scale: Optional[float] = None
    ) -> str:
        """
        Add guidance computation to the WorkloadGraph.

        Args:
            graph: Target WorkloadGraph
            unconditional_tensor_name: Name of unconditional prediction tensor
            conditional_tensor_name: Name of conditional prediction tensor
            output_tensor_name: Name for output guided tensor
            guidance_scale: Optional guidance scale override

        Returns:
            Name of the output tensor
        """
        # Get or use default guidance scale
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        # In a full implementation, this would add graph operations for:
        # 1. Subtract unconditional from conditional
        # 2. Scale the difference by guidance_scale
        # 3. Add back to unconditional prediction

        # For now, create a placeholder operation
        # This would be replaced with actual graph operations in a full implementation

        # Mock implementation - in practice, this would create actual graph nodes
        # For demonstration, we'll just return the conditional tensor name
        # as the guided result

        return conditional_tensor_name

    def validate_guidance_inputs(
        self,
        unconditional_prediction: Union[np.ndarray, SimTensor],
        conditional_prediction: Union[np.ndarray, SimTensor]
    ) -> bool:
        """
        Validate that guidance inputs have compatible shapes.

        Args:
            unconditional_prediction: Unconditional prediction
            conditional_prediction: Conditional prediction

        Returns:
            True if inputs are valid
        """
        if isinstance(unconditional_prediction, np.ndarray) and isinstance(conditional_prediction, np.ndarray):
            if unconditional_prediction.shape != conditional_prediction.shape:
                raise ValueError(
                    f"Shape mismatch: unconditional {unconditional_prediction.shape} "
                    f"vs conditional {conditional_prediction.shape}"
                )

            # Check for reasonable value ranges
            if np.any(np.isnan(unconditional_prediction)) or np.any(np.isnan(conditional_prediction)):
                raise ValueError("NaN values detected in predictions")

            if np.any(np.isinf(unconditional_prediction)) or np.any(np.isinf(conditional_prediction)):
                raise ValueError("Infinite values detected in predictions")

        return True

    def get_guidance_config(self) -> dict:
        """
        Get current guidance configuration.

        Returns:
            Dictionary with guidance parameters
        """
        return {
            "guidance_scale": self.guidance_scale,
            "guidance_rescale": self.guidance_rescale,
        }

    def __repr__(self) -> str:
        """String representation of the guidance utility."""
        return (f"ClassifierFreeGuidance("
                f"guidance_scale={self.guidance_scale}, "
                f"guidance_rescale={self.guidance_rescale})")

    def __str__(self) -> str:
        """Detailed string representation."""
        config = self.get_guidance_config()
        lines = ["ClassifierFreeGuidance:"]
        for key, value in config.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
