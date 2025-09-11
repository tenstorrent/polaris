#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris-compatible AutoencoderKL for SDXL VAE operations.

This module provides VAE encoding/decoding operations that integrate
with the Polaris TTSIM workload system.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import Polaris TTSIM components
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.graph.wl_graph import WorkloadGraph


class AutoencoderKLPolaris(SimNN.Module):
    """
    Polaris-compatible AutoencoderKL for SDXL VAE operations.

    This class implements VAE encoding and decoding operations using
    TTSIM-compatible operations for integration with Polaris workloads.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        """
        Initialize the VAE component.

        Args:
            name: Component name identifier
            cfg: Configuration dictionary
        """
        super().__init__()

        self.name = name
        self.cfg = cfg

        # Extract VAE configuration
        self.in_channels = cfg.get('in_channels', 3)
        self.latent_channels = cfg.get('latent_channels', 4)
        self.sample_size = cfg.get('sample_size', 256)
        self.block_out_channels = cfg.get('block_out_channels', [128, 256, 512, 512])
        self.layers_per_block = cfg.get('layers_per_block', 2)
        self.scaling_factor = cfg.get('scaling_factor', 0.13025)
        self.norm_num_groups = cfg.get('norm_num_groups', 32)
        self.act_fn = cfg.get('act_fn', 'silu')

        # Build encoder and decoder operations
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        """Build encoder operations."""
        # Encoder operations (simplified for TTSIM compatibility)
        self.encoder_conv_in = F.Conv2d(f"{self.name}_encoder_conv_in",
                                       self.in_channels, self.block_out_channels[0], 3, stride=1, padding=1)
        self.encoder_conv_in.set_module(self)

        # Encoder blocks
        self.encoder_blocks = []
        for i, out_ch in enumerate(self.block_out_channels):
            for layer in range(self.layers_per_block):
                # Convolution layers with residual connections
                conv1 = F.Conv2d(f"{self.name}_encoder_block_{i}_layer_{layer}_conv1",
                                out_ch, out_ch, 3, stride=1, padding=1)
                conv2 = F.Conv2d(f"{self.name}_encoder_block_{i}_layer_{layer}_conv2",
                                out_ch, out_ch, 3, stride=1, padding=1)
                conv1.set_module(self)
                conv2.set_module(self)

                # Downsampling for all but last block
                downsample = None
                if layer == 0 and i < len(self.block_out_channels) - 1:
                    downsample = F.Conv2d(f"{self.name}_encoder_down_{i}",
                                         out_ch, self.block_out_channels[i+1], 3, stride=2, padding=1)
                    downsample.set_module(self)

                self.encoder_blocks.append({
                    'conv1': conv1,
                    'conv2': conv2,
                    'downsample': downsample
                })

        # Final encoder convolution to latent space
        self.encoder_conv_out = F.Conv2d(f"{self.name}_encoder_conv_out",
                                        self.block_out_channels[-1], self.latent_channels * 2, 3, stride=1, padding=1)
        self.encoder_conv_out.set_module(self)

    def _build_decoder(self):
        """Build decoder operations."""
        # Decoder operations (simplified for TTSIM compatibility)
        self.decoder_conv_in = F.Conv2d(f"{self.name}_decoder_conv_in",
                                       self.latent_channels, self.block_out_channels[-1], 3, stride=1, padding=1)
        self.decoder_conv_in.set_module(self)

        # Decoder blocks (reverse order)
        self.decoder_blocks = []
        for i in reversed(range(len(self.block_out_channels))):
            in_ch = self.block_out_channels[i]
            for layer in range(self.layers_per_block):
                # Convolution layers with residual connections
                conv1 = F.Conv2d(f"{self.name}_decoder_block_{i}_layer_{layer}_conv1",
                                in_ch, in_ch, 3, stride=1, padding=1)
                conv2 = F.Conv2d(f"{self.name}_decoder_block_{i}_layer_{layer}_conv2",
                                in_ch, in_ch, 3, stride=1, padding=1)
                conv1.set_module(self)
                conv2.set_module(self)

                # Upsampling for all but last block
                upsample = None
                if layer == self.layers_per_block - 1 and i > 0:
                    upsample = F.Resize(f"{self.name}_decoder_up_{i}", scale_factor=2.0, mode="nearest")
                    upsample.set_module(self)

                self.decoder_blocks.append({
                    'conv1': conv1,
                    'conv2': conv2,
                    'upsample': upsample
                })

        # Final decoder convolution to RGB
        self.decoder_conv_out = F.Conv2d(f"{self.name}_decoder_conv_out",
                                        self.block_out_channels[0], self.in_channels, 3, stride=1, padding=1)
        self.decoder_conv_out.set_module(self)

    def add_encode_call(self, workload_graph: WorkloadGraph, input_tensor: SimTensor) -> SimTensor:
        """
        Add VAE encoding operations to the workload graph.

        Args:
            workload_graph: TTSIM workload graph
            input_tensor: Input image tensor [batch, channels, height, width]

        Returns:
            Encoded latent tensor with mean and log variance
        """
        # Encoder forward pass
        x = self.encoder_conv_in(input_tensor)

        # Encoder blocks
        for block in self.encoder_blocks:
            # Residual connection
            residual = x
            x = block['conv1'](x)
            x = F.Gelu(f"{block['conv1'].name}_gelu")(x)
            x = block['conv2'](x)
            x = x + residual  # Residual connection

            # Downsampling if needed
            if block['downsample'] is not None:
                x = block['downsample'](x)

        # Final convolution to latent space (mean and log variance)
        latents = self.encoder_conv_out(x)

        return latents

    def add_decode_call(self, workload_graph: WorkloadGraph, latent_tensor: SimTensor) -> SimTensor:
        """
        Add VAE decoding operations to the workload graph.

        Args:
            workload_graph: TTSIM workload graph
            latent_tensor: Input latent tensor [batch, latent_channels, height, width]

        Returns:
            Decoded image tensor
        """
        # Decoder forward pass
        x = self.decoder_conv_in(latent_tensor)

        # Decoder blocks
        for block in self.decoder_blocks:
            # If channel count mismatches expected in this block, insert a 1x1 projection BEFORE residual
            try:
                # params holds tuples (pos, tensor); extract conv weight tensor
                weight_tensor = None
                for pos, pt in block['conv1'].params:
                    if pt.name.endswith('.param'):
                        weight_tensor = pt
                        break
                if weight_tensor is not None:
                    expected_in_c = weight_tensor.shape[1]
                    current_in_c = x.shape[1]
                    if current_in_c != expected_in_c:
                        proj = F.Conv2d(f"{block['conv1'].name}_chanproj", current_in_c, expected_in_c, 1, stride=1, padding=0)
                        proj.set_module(self)
                        x = proj(x)
            except Exception:
                pass

            # Residual connection after channel alignment
            residual = x
            x = block['conv1'](x)
            x = F.Gelu(f"{block['conv1'].name}_gelu")(x)
            x = block['conv2'](x)
            x = x + residual  # Residual connection

            # Upsampling if needed
            if block['upsample'] is not None:
                x = block['upsample'](x)

        # Final convolution to RGB
        images = self.decoder_conv_out(x)

        # Apply tanh activation for image output
        images = F.Tanh(f"{self.name}_tanh")(images)

        return images

    def analytical_param_count(self, lvl: int = 0) -> int:
        """Estimate parameter count for analytical purposes."""
        # Rough parameter count estimation
        total_params = 0

        # Encoder parameters
        for i, out_ch in enumerate(self.block_out_channels):
            in_ch = self.in_channels if i == 0 else self.block_out_channels[i-1]
            # Conv layers
            total_params += (in_ch * out_ch * 9) * self.layers_per_block  # 3x3 conv
            total_params += (out_ch * out_ch * 9) * self.layers_per_block  # 3x3 conv

        # Final encoder conv
        total_params += self.block_out_channels[-1] * self.latent_channels * 2 * 9

        # Decoder parameters (similar to encoder but reversed)
        for i in reversed(range(len(self.block_out_channels))):
            out_ch = self.block_out_channels[i]
            in_ch = self.latent_channels if i == len(self.block_out_channels) - 1 else self.block_out_channels[i+1]
            total_params += (in_ch * out_ch * 9) * self.layers_per_block
            total_params += (out_ch * out_ch * 9) * self.layers_per_block

        # Final decoder conv
        total_params += self.block_out_channels[0] * self.in_channels * 9

        return total_params


# Logging setup
import logging
LOG = logging.getLogger(__name__)
INFO = LOG.info
DEBUG = LOG.debug
ERROR = LOG.error
WARNING = LOG.warning