#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python 3.13 compatible weight initialization utilities.
Converted from mmcv/mmengine initialization functions.

These are used for initializing PyTorch model weights that will be
loaded into TTSim models for inference.
"""

import math
import warnings
import numpy as np


def xavier_uniform_(tensor, gain=1.0, distribution="uniform"):
    """
    Xavier/Glorot uniform initialization.

    Args:
        tensor: numpy array or torch tensor to initialize
        gain: scaling factor
        distribution: 'uniform' or 'normal' (only uniform supported here)

    Reference:
        Understanding the difficulty of training deep feedforward neural networks
        - Glorot, X. & Bengio, Y. (2010)
    """
    if hasattr(tensor, "numpy"):
        # PyTorch tensor
        import torch.nn.init as init # type: ignore[import-not-found]
        if distribution == 'uniform':
            init.xavier_uniform_(tensor, gain=gain)
        else:
            init.xavier_normal_(tensor, gain=gain)
    else:
        # Numpy array
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

        if distribution == "uniform":
            a = math.sqrt(3.0) * std
            with np.errstate(all="ignore"):
                tensor[:] = np.random.uniform(-a, a, tensor.shape)
        else:
            with np.errstate(all="ignore"):
                tensor[:] = np.random.normal(0, std, tensor.shape)

    return tensor


def constant_(tensor, val=0.0):
    """
    Fill tensor with constant value.

    Args:
        tensor: numpy array or torch tensor to initialize
        val: constant value
    """
    if hasattr(tensor, "numpy"):
        # PyTorch tensor
        import torch.nn.init as init # type: ignore[import-not-found]
        init.constant_(tensor, val)
    else:
        # Numpy array
        tensor[:] = val

    return tensor


def normal_(tensor, mean=0.0, std=1.0):
    """
    Fill tensor with values drawn from normal distribution N(mean, std).

    Args:
        tensor: numpy array or torch tensor to initialize
        mean: mean of the normal distribution
        std: standard deviation of the normal distribution
    """
    if hasattr(tensor, "numpy"):
        pass
        ## PyTorch tensor
        # import torch.nn.init as init
        # init.normal_(tensor, mean=mean, std=std)
    else:
        # Numpy array
        with np.errstate(all="ignore"):
            tensor[:] = np.random.normal(mean, std, tensor.shape)

    return tensor


def xavier_init(module, gain=1.0, bias=0.0, distribution="uniform"):
    """
    Initialize a module with Xavier initialization.

    Args:
        module: nn.Module with weight and optionally bias
        gain: scaling factor for Xavier init
        bias: constant value for bias initialization
        distribution: 'uniform' or 'normal'
    """
    if hasattr(module, "weight") and module.weight is not None:
        xavier_uniform_(module.weight.data, gain=gain, distribution=distribution)

    if hasattr(module, "bias") and module.bias is not None:
        constant_(module.bias.data, bias)


def constant_init(module, val=0.0, bias=0.0):
    """
    Initialize a module with constant values.

    Args:
        module: nn.Module with weight and optionally bias
        val: constant value for weight initialization
        bias: constant value for bias initialization
    """
    if hasattr(module, "weight") and module.weight is not None:
        constant_(module.weight.data, val)

    if hasattr(module, "bias") and module.bias is not None:
        constant_(module.bias.data, bias)


def _calculate_fan_in_and_fan_out(tensor):
    """
    Calculate fan_in and fan_out for a tensor.
    Supports 2D and higher dimensional tensors.

    Args:
        tensor: weight tensor (numpy array or torch tensor)

    Returns:
        tuple: (fan_in, fan_out)
    """
    dimensions = len(tensor.shape)

    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out cannot be computed for tensor"
            f" with fewer than 2 dimensions, got {dimensions}"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1

    if dimensions > 2:
        # For conv layers: shape is (out_channels, in_channels, k_h, k_w, ...)
        receptive_field_size = np.prod(tensor.shape[2:])

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _is_power_of_2(n):
    """
    Check if n is a power of 2.

    Args:
        n: integer to check

    Returns:
        bool: True if n is a power of 2
    """
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


# Export commonly used functions
__all__ = [
    "xavier_init",
    "constant_init",
    "xavier_uniform_",
    "constant_",
    "normal_",
    "_is_power_of_2",
    "_calculate_fan_in_and_fan_out",
]
