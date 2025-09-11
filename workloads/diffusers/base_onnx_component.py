#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Base class for ONNX-based components in Polaris workloads.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class BaseOnnxComponent:
    """
    Base class for ONNX-based model components.

    Provides common functionality for loading and managing ONNX models
    within the Polaris workload framework.
    """

    def __init__(self, onnx_path: Union[str, Path], **kwargs: Any):
        """
        Initialize the ONNX component.

        Args:
            onnx_path: Path to the ONNX model file
            **kwargs: Additional configuration parameters
        """
        self.onnx_path = Path(onnx_path)
        self._config = kwargs  # Store config in private attribute
        self._model = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

        # Check if file exists (for backward compatibility with tests)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

    def load_model(self) -> None:
        """
        Load the ONNX model. Override in subclasses for specific loading logic.
        """
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        # Placeholder for actual ONNX loading logic
        # This would typically use onnxruntime or similar
        self._model = str(self.onnx_path)  # type: ignore[assignment] # Mock ONNX model loaded
        self._input_names = ["input"]  # Mock input names
        self._output_names = ["output"]  # Mock output names

    def get_input_names(self) -> list[str]:
        """Get the names of model inputs."""
        return self._input_names

    def get_output_names(self) -> list[str]:
        """Get the names of model outputs."""
        return self._output_names

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the model.

        Args:
            inputs: Dictionary of input tensors

        Returns:
            Dictionary of output tensors
        """
        if self._model is None:
            self.load_model()

        # Placeholder for actual inference logic
        # This would typically use the loaded ONNX model
        outputs = {}
        for output_name in self._output_names:
            # Mock output - in real implementation this would come from the model
            outputs[output_name] = np.array([])

        return outputs

    def validate_inputs(self, **inputs) -> None:
        """
        Validate input tensors before processing.

        Args:
            **inputs: Input tensors to validate

        Raises:
            ValueError: If inputs are invalid
        """
        # Basic validation - can be overridden by subclasses
        for key, value in inputs.items():
            if value is None:
                raise ValueError(f"Input '{key}' cannot be None")

    def create_tensor(self, name: str, shape: List[int], dtype: str = "float32", data: Optional[np.ndarray] = None) -> Any:
        """
        Create a tensor with the specified properties.

        Args:
            name: Tensor name
            shape: Tensor shape
            dtype: Data type
            data: Optional tensor data

        Returns:
            Created tensor object
        """
        # Import here to avoid circular imports
        from ttsim.ops.tensor import SimTensor

        cfg = {
            'name': name,
            'shape': shape,
            'dtype': dtype,
        }

        if data is not None:
            cfg['data'] = data  # type: ignore[assignment]

        return SimTensor(cfg)

    def add_to_graph(self, graph, input_mappings: Dict[str, str], output_mappings: Dict[str, str]) -> None:
        """
        Add this component to the workload graph.

        Args:
            graph: WorkloadGraph to add to
            input_mappings: Mapping of input names to tensor names
            output_mappings: Mapping of output names to tensor names
        """
        # This is a placeholder implementation
        # In a real implementation, this would add the ONNX subgraph to the graph
        pass
