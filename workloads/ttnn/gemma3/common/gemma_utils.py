#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import math
import numpy as np
import ttsim.front.ttnn as ttnn


def nearest_32(x):
    """Round up to nearest multiple of 32."""
    return math.ceil(x / 32) * 32


def nearest_y(x, y):
    """Round up x to nearest multiple of y."""
    return math.ceil(x / y) * y


def pad_to_multiple(x, multiple):
    """Pad x to be a multiple of the given value."""
    return ((x + multiple - 1) // multiple) * multiple


def is_blackhole():
    ARCH_NAME = ttnn.get_arch_name()
    return "blackhole" in ARCH_NAME


def to_numpy(tensor):
    """Convert any tensor type to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, 'numpy'):
        return tensor.numpy()
    if hasattr(tensor, 'detach'):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def state_dict_to_ttnn(
    state_dict_value,
    dtype,
    device,
    layout=None,
    memory_config=None,
):
    """
    Convert a state dict value (numpy array or torch tensor) to a ttnn tensor.
    
    Args:
        state_dict_value: The value from state_dict (numpy array, torch tensor, or ttnn tensor)
        dtype: Target ttnn dtype
        device: Target device
        layout: Target layout (default: TILE_LAYOUT)
        memory_config: Target memory config (default: DRAM_MEMORY_CONFIG)
    
    Returns:
        ttnn.Tensor
    """
    if layout is None:
        layout = ttnn.TILE_LAYOUT
    if memory_config is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    
    # If already a ttnn tensor, just return it (maybe convert layout/device)
    if hasattr(state_dict_value, 'get_layout'):
        # It's already a ttnn tensor
        return state_dict_value
    
    # Convert to numpy first
    np_array = to_numpy(state_dict_value)
    
    # Convert to ttnn tensor
    return ttnn.as_tensor(
        np_array,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )