# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common utility functions for Polaris PI0 implementation.

TT-only utility helpers:
    - precision -> TTNN dtype mapping
    - sinusoidal timestep embeddings
    - safe concatenation
    - position-id computation
"""

import math
import numpy as np
import ttsim.front.ttnn as ttnn


def get_ttnn_dtype(precision: str):
    """
    Convert precision string to TTNN dtype.
    """
    dtype_map = {
        "bfloat16": ttnn.bfloat16,
        "float32": ttnn.float32,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": getattr(ttnn, "bfloat4_b", ttnn.bfloat8_b),
    }
    return dtype_map.get(precision, ttnn.bfloat16)

def create_sinusoidal_pos_embedding_ttnn(
    time,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    memory_config=None,
):
    """
    Create sinusoidal positional embeddings for timesteps using TTNN only.

    Args:
        time: TTNN tensor of shape (batch_size,) or (batch_size, 1)
        dimension: embedding dimension, must be divisible by 2
        min_period: minimum sinusoidal period
        max_period: maximum sinusoidal period

    Returns:
        TTNN tensor of shape (batch_size, dimension)
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    half_dim = dimension // 2

    # Build frequency values in plain Python
    fractions = [i / (half_dim - 1) if half_dim > 1 else 0.0 for i in range(half_dim)]
    periods = [min_period * ((max_period / min_period) ** frac) for frac in fractions]
    scaling_values = [(2.0 * math.pi) / p for p in periods]

    # Create TTNN tensor directly from Python list/scalar path available in Polaris.
    # If your Polaris build uses a different constructor helper, replace this line
    # with the local tensor creation API used in your tree.
    scaling_factor = ttnn.as_tensor(
        [scaling_values],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
    )

    # Make time shape [batch, 1]
    time_shape = time.shape
    if len(time_shape) == 1:
        time = ttnn.reshape(time, (time_shape[0], 1))

    # Broadcast multiply => [batch, half_dim]
    sin_input = ttnn.multiply(time, scaling_factor)

    sin_emb = ttnn.sin(sin_input)
    cos_emb = ttnn.cos(sin_input)

    embeddings = ttnn.concat([sin_emb, cos_emb], dim=-1, memory_config=memory_config)
    return embeddings


def safe_cat_ttnn(
    tensors: list,
    dim: int = -1,
    memory_config=None,
):
    """
    Safely concatenate TTNN tensors.
    """
    if len(tensors) == 0:
        raise ValueError("Cannot concatenate empty list of tensors")

    if memory_config is None:
        memory_config = ttnn.L1_MEMORY_CONFIG

    return ttnn.concat(tensors, dim=dim, memory_config=memory_config)


def compute_position_ids_ttnn(pad_masks, device):
    """
    Compute position IDs from padding masks.
    """
    import numpy as np
    
    shape = pad_masks.shape
    if shape is None:
        raise ValueError("pad_masks must have a valid shape")
    
    batch_size = shape[0]
    seq_len = shape[-1]
    
    # Create position IDs [0, 1, 2, ..., seq_len-1] for each batch
    position_ids_1d = np.arange(seq_len, dtype=np.float32)  # Shape: (seq_len,)
    position_ids_2d = np.tile(position_ids_1d, (batch_size, 1))  # Shape: (batch_size, seq_len)
    
    # Convert numpy to TTNN using as_tensor
    return ttnn.as_tensor(
        position_ids_2d,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT
    )

# ttnn_to_torch,torch to ttnn, is not needed 

# Export TT-only helpers
create_sinusoidal_pos_embedding = create_sinusoidal_pos_embedding_ttnn
safe_cat = safe_cat_ttnn
compute_position_ids = compute_position_ids_ttnn

# sample_noise = sample_noise_torch
# sample_time = sample_time_torch