#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim Utility Functions for BEVFormer Validation

This module provides wrapper functions around TTSim compute operations
with a cleaner, more efficient interface that avoids creating mock objects.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from ttsim.ops.desc.data_compute import (
    compute_add,
    compute_sub,
    compute_mul,
    compute_div,
    compute_relu,
    compute_sigmoid,
    compute_softmax,
    compute_reshape,
    compute_transpose,
    compute_resize,
    compute_sin,
    compute_cos,
    compute_exp,
    compute_log,
    compute_sqrt,
    compute_reducemean,
    compute_reducesum,
    compute_matmul,
    compute_concat,
    compute_tanh,
    compute_pow,
    compute_clip,
    compute_unsqueeze,
    compute_squeeze,
    compute_tile,
    compute_maxpool2d,
    compute_conv2d,
)

# ============================================================================
# Helper Classes for TTSim Operations
# ============================================================================


class TensorWrapper:
    """Lightweight wrapper for numpy arrays to work with TTSim compute functions."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data


class OpWrapper:
    """Lightweight wrapper for operation attributes."""

    __slots__ = ("attrs",)

    def __init__(self, **attrs):
        self.attrs = attrs


# ============================================================================
# Basic Activation Functions
# ============================================================================


def ttsim_relu(x):
    """ReLU activation: max(0, x)"""
    return compute_relu([TensorWrapper(x)], OpWrapper())


def ttsim_sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return compute_sigmoid([TensorWrapper(x)], OpWrapper())


def ttsim_tanh(x):
    """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return compute_tanh([TensorWrapper(x)], OpWrapper())


def ttsim_softmax(x, axis=-1):
    """Softmax activation: exp(x) / sum(exp(x))"""
    return compute_softmax([TensorWrapper(x)], OpWrapper(axis=axis))


# ============================================================================
# Element-wise Operations
# ============================================================================


def ttsim_add(a, b):
    """Element-wise addition: a + b"""
    return compute_add([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_sub(a, b):
    """Element-wise subtraction: a - b"""
    return compute_sub([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_mul(a, b):
    """Element-wise multiplication: a * b"""
    return compute_mul([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_div(a, b):
    """Element-wise division: a / b"""
    return compute_div([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


# ============================================================================
# Mathematical Functions
# ============================================================================


def ttsim_sqrt(x):
    """Square root: sqrt(x)"""
    return compute_sqrt([TensorWrapper(x)], OpWrapper())


def ttsim_exp(x):
    """Exponential: exp(x)"""
    return compute_exp([TensorWrapper(x)], OpWrapper())


def ttsim_log(x):
    """Natural logarithm: ln(x)"""
    return compute_log([TensorWrapper(x)], OpWrapper())


def ttsim_sin(x):
    """Sine: sin(x)"""
    return compute_sin([TensorWrapper(x)], OpWrapper())


def ttsim_cos(x):
    """Cosine: cos(x)"""
    return compute_cos([TensorWrapper(x)], OpWrapper())


def ttsim_pow(a, b):
    """Power: a^b"""
    return compute_pow([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_clip(x, min_val, max_val):
    """Clip values to range [min_val, max_val]"""
    min_tensor = TensorWrapper(np.array(min_val, dtype=x.dtype))
    max_tensor = TensorWrapper(np.array(max_val, dtype=x.dtype))
    return compute_clip([TensorWrapper(x), min_tensor, max_tensor], OpWrapper())


# ============================================================================
# Shape Operations
# ============================================================================


def ttsim_reshape(x, shape):
    """Reshape tensor to new shape"""
    shape_tensor = TensorWrapper(np.array(shape, dtype=np.int64))
    return compute_reshape([TensorWrapper(x), shape_tensor], OpWrapper())


def ttsim_transpose(x, axes):
    """Transpose tensor according to axes permutation"""
    return compute_transpose([TensorWrapper(x)], OpWrapper(perm=axes))


def ttsim_unsqueeze(x, axes):
    """Add dimensions at specified axes"""
    axes_tensor = TensorWrapper(np.array(axes, dtype=np.int64))
    return compute_unsqueeze([TensorWrapper(x), axes_tensor], OpWrapper())


def ttsim_squeeze(x, axes=None):
    """Remove dimensions at specified axes or all dimensions of size 1"""
    if axes is not None:
        axes_tensor = TensorWrapper(np.array(axes, dtype=np.int64))
        return compute_squeeze([TensorWrapper(x), axes_tensor], OpWrapper())
    else:
        return compute_squeeze([TensorWrapper(x)], OpWrapper())


# ============================================================================
# Aggregation Operations
# ============================================================================


def ttsim_reducemean(x, axis=None, keepdims=True):
    """Compute mean along specified axes"""
    keepdims_val = 1 if keepdims else 0

    if axis is not None:
        axis_list = [axis] if isinstance(axis, int) else axis
        axes_tensor = TensorWrapper(np.array(axis_list, dtype=np.int64))
        return compute_reducemean(
            [TensorWrapper(x), axes_tensor], OpWrapper(keepdims=keepdims_val)
        )
    else:
        return compute_reducemean(
            [TensorWrapper(x)], OpWrapper(keepdims=keepdims_val, noop_with_empty_axes=0)
        )


def ttsim_reducesum(x, axis=None, keepdims=True):
    """Compute sum along specified axes"""
    keepdims_val = 1 if keepdims else 0

    if axis is not None:
        axis_list = [axis] if isinstance(axis, int) else axis
        axes_tensor = TensorWrapper(np.array(axis_list, dtype=np.int64))
        return compute_reducesum(
            [TensorWrapper(x), axes_tensor], OpWrapper(keepdims=keepdims_val)
        )
    else:
        return compute_reducesum(
            [TensorWrapper(x)], OpWrapper(keepdims=keepdims_val, noop_with_empty_axes=0)
        )


# ============================================================================
# Matrix Operations
# ============================================================================


def ttsim_matmul(a, b):
    """Matrix multiplication: a @ b"""
    return compute_matmul([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_concat(arrays, axis=0):
    """Concatenate arrays along specified axis"""
    tensors = [TensorWrapper(arr) for arr in arrays]
    return compute_concat(tensors, OpWrapper(axis=axis))


# ============================================================================
# High-level Operations
# ============================================================================


def ttsim_layernorm(x, normalized_shape, eps=1e-5):
    """
    Layer normalization: (x - mean) / sqrt(var + eps)

    Args:
        x: Input array
        normalized_shape: Shape over which to normalize (currently unused, normalizes over last dim)
        eps: Small constant for numerical stability
    """
    # Compute mean and variance
    mean = ttsim_reducemean(x, axis=-1, keepdims=True)

    # Compute variance: Var = E[(X - mean)^2]
    x_centered = ttsim_sub(x, mean)
    x_squared = ttsim_mul(x_centered, x_centered)
    var = ttsim_reducemean(x_squared, axis=-1, keepdims=True)

    # Normalize: (x - mean) / sqrt(var + eps)
    var_eps = ttsim_add(var, np.array(eps, dtype=x.dtype))
    std = ttsim_sqrt(var_eps)
    normalized = ttsim_div(x_centered, std)

    return normalized


def ttsim_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    2D Convolution

    Args:
        x: Input tensor [N, C_in, H, W]
        weight: Convolution kernel [C_out, C_in/groups, kH, kW]
        bias: Optional bias [C_out]
        stride: Stride (int or [stride_h, stride_w])
        padding: Padding (int or [pad_h, pad_w, pad_h, pad_w])
        dilation: Dilation (int or [dilation_h, dilation_w])
        groups: Number of blocked connections
    """
    # Normalize parameters to lists
    stride_list = [stride, stride] if isinstance(stride, int) else stride
    padding_list = (
        [padding, padding, padding, padding] if isinstance(padding, int) else padding
    )
    dilation_list = [dilation, dilation] if isinstance(dilation, int) else dilation

    op = OpWrapper(
        strides=stride_list, pads=padding_list, dilations=dilation_list, group=groups
    )

    if bias is not None:
        return compute_conv2d(
            [TensorWrapper(x), TensorWrapper(weight), TensorWrapper(bias)], op
        )
    else:
        return compute_conv2d([TensorWrapper(x), TensorWrapper(weight)], op)


def ttsim_maxpool2d(x, kernel_size, stride=None, padding=0):
    """
    2D Max Pooling

    Args:
        x: Input tensor [N, C, H, W]
        kernel_size: Size of pooling kernel (int or [kH, kW])
        stride: Stride (int or [stride_h, stride_w]), defaults to kernel_size
        padding: Padding (int or [pad_h, pad_w, pad_h, pad_w])
    """
    kernel_list = (
        [kernel_size, kernel_size] if isinstance(kernel_size, int) else kernel_size
    )

    if stride is None:
        stride_list = kernel_list
    else:
        stride_list = [stride, stride] if isinstance(stride, int) else stride

    padding_list = (
        [padding, padding, padding, padding] if isinstance(padding, int) else padding
    )

    op = OpWrapper(kernel_shape=kernel_list, strides=stride_list, pads=padding_list)

    return compute_maxpool2d([TensorWrapper(x)], op)


def ttsim_interpolate(x, size=None, scale_factor=None, mode="nearest"):
    """
    Interpolate (resize) tensor

    Args:
        x: Input tensor [N, C, H, W]
        size: Target size [H_out, W_out]
        scale_factor: Scale factor (float or [scale_h, scale_w])
        mode: Interpolation mode ('nearest', 'linear', 'bilinear', etc.)
    """
    if scale_factor is not None:
        op = OpWrapper(mode=mode, scale_factor=scale_factor)
    elif size is not None:
        # Calculate scale factor from size
        H_in, W_in = x.shape[-2:]
        H_out, W_out = size
        scale_h = H_out / H_in
        scale_w = W_out / W_in
        op = OpWrapper(mode=mode, scale_factor=[scale_h, scale_w])
    else:
        raise ValueError("Either size or scale_factor must be provided")

    return compute_resize([TensorWrapper(x)], op)


# ============================================================================
# Comparison and Validation Helpers
# ============================================================================


def compare_arrays(pytorch_arr, ttsim_arr, name="array", rtol=1e-5, atol=1e-6):
    """
    Compare PyTorch tensor with TTSim/numpy array.

    Args:
        pytorch_arr: PyTorch tensor or numpy array
        ttsim_arr: TTSim output (numpy array)
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if arrays match within tolerance

    Note: Relaxed tolerances (rtol=1e-5, atol=1e-6) account for accumulated
    floating point errors in multi-step operations.
    """
    import torch

    if isinstance(pytorch_arr, torch.Tensor):
        pt_numpy = pytorch_arr.detach().cpu().numpy()
    else:
        pt_numpy = pytorch_arr

    if pt_numpy.shape != ttsim_arr.shape:
        print(
            f"  ✗ {name}: Shape mismatch - PyTorch: {pt_numpy.shape}, TTSim: {ttsim_arr.shape}"
        )
        return False

    if np.allclose(pt_numpy, ttsim_arr, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(pt_numpy - ttsim_arr))
        print(f"  ✓ {name}: Match! Max difference: {max_diff:.2e}")
        return True
    else:
        max_diff = np.max(np.abs(pt_numpy - ttsim_arr))
        mean_diff = np.mean(np.abs(pt_numpy - ttsim_arr))
        print(
            f"  ✗ {name}: Mismatch! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )
        return False


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test(title):
    """Print a formatted test title."""
    print("\n" + title)
    print("-" * 80)
