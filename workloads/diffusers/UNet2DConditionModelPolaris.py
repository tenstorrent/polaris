#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from .base_onnx_component import BaseOnnxComponent
from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.tensor import SimTensor


class UNet2DConditionModelPolaris(BaseOnnxComponent):
    """
    ONNX-based UNet2DConditionModel wrapper for Polaris diffusion pipelines.

    This class wraps a UNet model exported to ONNX format and provides
    the interface expected by diffusion pipelines. It handles the typical
    UNet inputs (latent, timestep, text embeddings) and integrates with
    Polaris' WorkloadGraph system.

    Typical UNet inputs:
    - latent_model_input: (batch_size, channels, height, width)
    - timestep: (batch_size,) - timestep embeddings
    - encoder_hidden_states: (batch_size, seq_len, embed_dim) - text embeddings
    - additional conditioning (optional)

    Typical UNet outputs:
    - sample: (batch_size, channels, height, width) - predicted noise
    """

    def __init__(
        self,
        onnx_path: Union[str, "Path"],
        sample_size: int = 128,
        in_channels: int = 4,
        out_channels: int = 4,
        time_embedding_dim: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize UNet ONNX component.

        Args:
            onnx_path: Path to UNet ONNX model
            sample_size: Spatial size of the model (height/width)
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_embedding_dim: Dimension of timestep embeddings
            **kwargs: Additional configuration
        """
        # Store model configuration
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim

        super().__init__(
            onnx_path=onnx_path,
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim,
            **kwargs
        )

    def forward(
        self,
        sample: Union[SimTensor, np.ndarray],
        timestep: Union[SimTensor, np.ndarray, float, int],
        encoder_hidden_states: Optional[Union[SimTensor, np.ndarray]] = None,
        encoder_attention_mask: Optional[Union[SimTensor, np.ndarray]] = None,
        **kwargs
    ):
        """
        Forward pass through the UNet model.

        Args:
            sample: Input latent tensor (batch_size, in_channels, height, width)
            timestep: Timestep value(s) for conditioning
            encoder_hidden_states: Text embeddings (batch_size, seq_len, embed_dim)
            encoder_attention_mask: Attention mask for text embeddings
            **kwargs: Additional conditioning inputs

        Returns:
            Predicted noise tensor (batch_size, out_channels, height, width)
        """
        # Validate inputs
        inputs = {
            'sample': sample,
            'timestep': timestep,
        }

        if encoder_hidden_states is not None:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        if encoder_attention_mask is not None:
            inputs['encoder_attention_mask'] = encoder_attention_mask

        # Add any additional inputs from kwargs
        inputs.update(kwargs)

        self.validate_inputs(**inputs)

        # In simulation mode, we don't actually run the model
        # Instead, we create output tensors with expected shapes
        if isinstance(sample, SimTensor):
            batch_size = sample.shape[0]
            height = sample.shape[2] if len(sample.shape) > 2 else self.sample_size
            width = sample.shape[3] if len(sample.shape) > 3 else self.sample_size
        elif isinstance(sample, np.ndarray):
            batch_size = sample.shape[0]
            height = sample.shape[2] if len(sample.shape) > 2 else self.sample_size
            width = sample.shape[3] if len(sample.shape) > 3 else self.sample_size
        else:
            # Fallback for shape inference
            batch_size = 1
            height = self.sample_size
            width = self.sample_size

        # Create output shape
        output_shape = [batch_size, self.out_channels, height, width]

        # Create output tensor
        output = self.create_tensor(
            name="unet_output",
            shape=output_shape,
            dtype="float32"
        )

        return output

    def add_call(
        self,
        graph: WorkloadGraph,
        sample: Union[SimTensor, str],
        timestep: Union[SimTensor, str, float, int],
        encoder_hidden_states: Optional[Union[SimTensor, str]] = None,
        encoder_attention_mask: Optional[Union[SimTensor, str]] = None,
        output_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add UNet call to the WorkloadGraph.

        Args:
            graph: Target WorkloadGraph
            sample: Input latent tensor or tensor name
            timestep: Timestep value or tensor name
            encoder_hidden_states: Text embeddings or tensor name
            encoder_attention_mask: Attention mask or tensor name
            output_name: Name for output tensor
            **kwargs: Additional conditioning inputs

        Returns:
            Name of output tensor
        """
        if output_name is None:
            output_name = f"unet_output_{id(self)}"

        # Prepare input mappings for the ONNX subgraph
        input_mappings = {}

        # Map sample input
        if isinstance(sample, str):
            input_mappings['sample'] = sample
        elif isinstance(sample, SimTensor):
            # Add sample tensor to graph if not already present
            if sample.name not in [t.name for t in graph._tensors.values()]:
                graph.add_tensor(sample)
            input_mappings['sample'] = sample.name

        # Map timestep input
        timestep_name = f"timestep_{id(self)}"
        if isinstance(timestep, (float, int)):
            # Create timestep tensor
            timestep_tensor = self.create_tensor(
                name=timestep_name,
                shape=[1],
                dtype="float32",
                data=np.array([timestep], dtype=np.float32),
                is_const=True
            )
            graph.add_tensor(timestep_tensor)
        elif isinstance(timestep, str):
            input_mappings['timestep'] = timestep
        elif isinstance(timestep, SimTensor):
            if timestep.name not in [t.name for t in graph._tensors.values()]:
                graph.add_tensor(timestep)
            input_mappings['timestep'] = timestep.name

        # Map encoder hidden states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, str):
                input_mappings['encoder_hidden_states'] = encoder_hidden_states
            elif isinstance(encoder_hidden_states, SimTensor):
                if encoder_hidden_states.name not in [t.name for t in graph._tensors.values()]:
                    graph.add_tensor(encoder_hidden_states)
                input_mappings['encoder_hidden_states'] = encoder_hidden_states.name

        # Map attention mask if provided
        if encoder_attention_mask is not None:
            if isinstance(encoder_attention_mask, str):
                input_mappings['encoder_attention_mask'] = encoder_attention_mask
            elif isinstance(encoder_attention_mask, SimTensor):
                if encoder_attention_mask.name not in [t.name for t in graph._tensors.values()]:
                    graph.add_tensor(encoder_attention_mask)
                input_mappings['encoder_attention_mask'] = encoder_attention_mask.name

        # Map additional inputs from kwargs
        for key, value in kwargs.items():
            if isinstance(value, str):
                input_mappings[key] = value
            elif isinstance(value, SimTensor):
                if value.name not in [t.name for t in graph._tensors.values()]:
                    graph.add_tensor(value)
                input_mappings[key] = value.name

        # Define output mapping
        output_mappings = {'sample': output_name}

        # Add component to graph
        self.add_to_graph(graph, input_mappings, output_mappings)

        return output_name

    def get_expected_latent_shape(self, batch_size: int = 1, height: Optional[int] = None,
                                width: Optional[int] = None) -> List[int]:
        """
        Get expected latent tensor shape for given parameters.

        Args:
            batch_size: Batch size
            height: Height of latent (defaults to sample_size)
            width: Width of latent (defaults to sample_size)

        Returns:
            Expected shape [batch_size, channels, height, width]
        """
        height = height or self.sample_size
        width = width or self.sample_size
        return [batch_size, self.in_channels, height, width]

    def get_expected_output_shape(self, batch_size: int = 1, height: Optional[int] = None,
                                width: Optional[int] = None) -> List[int]:
        """
        Get expected output tensor shape for given parameters.

        Args:
            batch_size: Batch size
            height: Height of output (defaults to sample_size)
            width: Width of output (defaults to sample_size)

        Returns:
            Expected shape [batch_size, channels, height, width]
        """
        height = height or self.sample_size
        width = width or self.sample_size
        return [batch_size, self.out_channels, height, width]

    @property
    def config(self):
        """Get model configuration."""
        return {
            'sample_size': self.sample_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'time_embedding_dim': self.time_embedding_dim,
            'onnx_path': str(self.onnx_path)
        }
