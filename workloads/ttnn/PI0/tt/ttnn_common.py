# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Common utility functions for TTSim PI0 implementation.
This module provides shared helper functions used across the PI0 model:
    - Sinusoidal positional embeddings for flow matching timesteps
    - Safe tensor operations with dtype handling
    - Device-aware computations
"""
import math
from typing import Optional
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice


def get_ttnn_dtype(precision: str) -> ttnn.DataType:
    """
    Convert precision string to TTNN dtype.
    Args:
        precision: "bfloat16", "float32", "bfloat8_b", etc.
    Returns:
        TTNN data type
    """
    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": getattr(ttnn, "bfloat4_b", ttnn.bfloat8_b),
    }
    return dtype_map.get(precision, ttnn.bfloat16)


def create_sinusoidal_pos_embedding_ttnn(
    time: ttnn.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: Optional[TTNNDevice] = None,
    indices: Optional[ttnn.Tensor] = None,
) -> ttnn.Tensor:
    """
    Create sinusoidal positional embeddings for timesteps (pure ttsim version).
    All computations are done on device using TTNN operations.
    Args:
        time: TTNN tensor of shape (batch_size,) with timestep values
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for sinusoidal encoding
        max_period: Maximum period for sinusoidal encoding
        device: TTNNDevice (uses time's device if not specified)
        indices: Pre-allocated index tensor of shape (half_dim,)
    Returns:
        TTNN tensor of shape (batch_size, dimension) with sinusoidal embeddings
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    
    if device is None:
        # Get device from time tensor with proper null handling
        _device = time.device()   # type: ignore[misc, operator]
        if _device is None:
            raise ValueError("time tensor must have a device")
        device = _device
    
    half_dim = dimension // 2
    # Create fraction [0, 1/(n-1), 2/(n-1), ..., 1] using TTNN
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
    if half_dim > 1:
        fraction = ttnn.multiply(indices, 1.0 / (half_dim - 1))
    else:
        fraction = indices  # Edge case: half_dim == 1
    # Compute scaling_factor = (2π / min_period) * exp(-fraction * log_ratio)
    # Avoids ttnn.reciprocal (not available in ttsim) by negating the exponent.
    log_ratio = math.log(max_period / min_period)
    exponent = ttnn.multiply(fraction, -log_ratio)
    inv_period_ratio = ttnn.exp(exponent)
    scaling_factor = ttnn.multiply(inv_period_ratio, (2 * math.pi) / min_period)
    # Reshape for broadcasting: scaling_factor [half_dim] -> [1, half_dim]
    scaling_factor = ttnn.reshape(scaling_factor, (1, half_dim))
    # Reshape time for broadcasting: [batch] -> [batch, 1]
    time_reshaped = ttnn.reshape(time, (-1, 1))
    # Compute sin input: time * scaling_factor (broadcasts to [batch, half_dim])
    sin_input = ttnn.matmul(time_reshaped, scaling_factor)
    # Compute sin and cos
    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)
    # Concatenate to get [batch, dimension]
    embeddings = ttnn.concat(sin_emb, cos_emb, axis=-1)
    # Clean up intermediate tensors
    # ttnn.deallocate(indices)
    ttnn.deallocate(fraction)
    ttnn.deallocate(exponent)
    ttnn.deallocate(inv_period_ratio)
    ttnn.deallocate(scaling_factor)
    ttnn.deallocate(sin_input)
    return embeddings


def safe_cat_ttnn(
    tensors: list,
    dim: int = -1,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Safely concatenate TTNN tensors.
    Args:
        tensors: List of TTNN tensors to concatenate
        dim: Dimension along which to concatenate
        memory_config: Optional memory config for output
    Returns:
        Concatenated TTNN tensor
    """
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")
    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG
    return ttnn.concat(*tensors, axis=dim, memory_config=memory_config)


def compute_position_ids_ttnn(
    pad_masks: ttnn.Tensor,
    device: Optional[TTNNDevice] = None,
) -> ttnn.Tensor:
    """
    Compute position IDs from padding masks (ttsim version).
    Args:
        pad_masks: Boolean TTNN tensor (batch_size, seq_len)
        device: TTNNDevice
    Returns:
        Position IDs TTNN tensor (batch_size, seq_len)
    Note:
        ttnn.cumsum and ttnn.moreh_cumsum are both unavailable in ttsim.
        ttsim is a shape-tracking simulator so the exact numerical values
        do not matter — only the output shape must be correct.
        We return a placeholder tensor of the same shape as pad_masks.
    """
    shape = pad_masks.shape
    if shape is None:
        raise ValueError("pad_masks must have a valid shape")
    output_shape = list(shape)
    position_ids = ttnn.Tensor(shape=output_shape, device=device, dtype=ttnn.bfloat16)
    return position_ids


def ttnn_to_ttnn(tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Identity conversion — ttsim tensors are already in TTNN format.
    In the original TT Metal implementation this converted a TTNN tensor
    to a PyTorch tensor via ttnn.to_torch(). In the ttsim context there
    is no separate PyTorch runtime, so this function is a no-op.
    Args:
        tensor: TTNN (ttsim) tensor
    Returns:
        Same TTNN tensor (unchanged)
    """
    return tensor


# Alias kept for API compatibility with callers expecting ttnn_to_torch
ttnn_to_torch = ttnn_to_ttnn


def torch_to_ttnn(
    tensor: ttnn.Tensor,
    device: TTNNDevice,
    dtype: Optional[ttnn.DataType] = None,
    layout: Optional[ttnn.Layout] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """
    Ensure a ttsim tensor is on device with the correct dtype/layout.
    In the original TT Metal implementation this called ttnn.from_torch()
    to move a PyTorch tensor onto device. In the ttsim context the input
    is already a ttsim Tensor; this function re-wraps it with the requested
    layout/memory config when needed.
    Args:
        tensor: Input ttsim Tensor
        device: TTNNDevice to place the tensor on
        dtype: TTNN data type (default: bfloat16)
        layout: TTNN layout (default: TILE_LAYOUT)
        memory_config: Memory configuration (default: DRAM)
    Returns:
        TTNN tensor on the target device
    """
    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    # In ttsim, from_torch accepts ttsim tensors directly as a passthrough
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def tensor_1d_to_2d_ttnn(tensor, device, dtype):
    flat = ttnn.reshape(tensor, (-1,))  # ALWAYS flatten first

    features = flat.shape[-1]           # NEVER infer from original tensor

    return ttnn.reshape(flat, (1, features))

# Default exports — match the original ttnn_common.py aliases
create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_ttnn
safe_cat = safe_cat_ttnn
compute_position_ids = compute_position_ids_ttnn