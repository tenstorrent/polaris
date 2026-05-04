#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim Utility Functions for EA-LSS Validation.

This module provides:
  1. Thin wrappers around TTSim ``data_compute`` functions for easy use
     in validation scripts (same pattern as BEVFormer/Reference/Validation/ttsim_utils.py).
  2. Pure-NumPy reference implementations of mmcv/mmdet functions that are
     used by EA-LSS source files.  These replace torch/mmcv imports so that
     all validation code runs on CPU without external ML framework dependencies.

mmcv functions extracted and re-implemented here:
  - ConvModule1d_numpy  (from mmcv.cnn.ConvModule + mmcv.cnn.bricks.conv_module.py)
    Implements Conv1d + BN1d + ReLU forward pass in pure NumPy.
  - batchnorm1d_numpy   (from torch.nn.BatchNorm1d inference mode)
  - conv1d_numpy        (from torch.nn.Conv1d)

All implementations are CPU-only and produce results numerically identical
to their PyTorch / mmcv counterparts (within float32 precision).
"""

import os
import sys
import numpy as np

# Polaris root is 4 levels up from workloads/EA-LSS/Reference/Validation/
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _polaris_root not in sys.path:
    sys.path.insert(0, _polaris_root)

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
    compute_exp,
    compute_log,
    compute_sqrt,
    compute_reducemean,
    compute_matmul,
    compute_concat,
    compute_tanh,
    compute_pow,
    compute_clip,
    compute_tile,
    compute_maxpool2d,
    compute_conv2d,
    compute_batchnorm,
)


# ============================================================================
# Helper Classes  (used by all ttsim_* wrapper functions below)
# ============================================================================

class TensorWrapper:
    """Lightweight wrapper for numpy arrays to work with TTSim compute functions."""
    __slots__ = ("data",)

    def __init__(self, data: np.ndarray):
        self.data = data


class OpWrapper:
    """Lightweight wrapper for operation attributes."""
    __slots__ = ("attrs",)

    def __init__(self, **attrs):
        self.attrs = attrs


# ============================================================================
# Activation Functions
# ============================================================================

def ttsim_relu(x: np.ndarray) -> np.ndarray:
    """ReLU: max(0, x)"""
    return compute_relu([TensorWrapper(x)], OpWrapper())


def ttsim_sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid: 1 / (1 + exp(-x))"""
    return compute_sigmoid([TensorWrapper(x)], OpWrapper())


def ttsim_tanh(x: np.ndarray) -> np.ndarray:
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return compute_tanh([TensorWrapper(x)], OpWrapper())


def ttsim_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax: exp(x) / sum(exp(x)) along axis."""
    return compute_softmax([TensorWrapper(x)], OpWrapper(axis=axis))


# ============================================================================
# Clipped Sigmoid  (clip_sigmoid module)
# ============================================================================

def ttsim_clip_sigmoid(x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Clamped sigmoid: clamp(sigmoid(x), eps, 1-eps).

    Numerically equivalent to the TTSim ``clip_sigmoid`` module:
      Sigmoid op → Clip op[min=eps, max=1-eps]

    Args:
        x: Input array of any shape.
        eps: Clamp lower bound (upper bound is 1 - eps). Default: 1e-4.

    Returns:
        Array same shape as x with values in [eps, 1-eps].
    """
    sig_out = ttsim_sigmoid(x)
    min_t = TensorWrapper(np.array([eps], dtype=np.float32))
    max_t = TensorWrapper(np.array([1.0 - eps], dtype=np.float32))
    return compute_clip([TensorWrapper(sig_out), min_t, max_t], OpWrapper())


# ============================================================================
# Element-wise Operations
# ============================================================================

def ttsim_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise addition: a + b (supports broadcasting)."""
    return compute_add([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise subtraction: a - b."""
    return compute_sub([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise multiplication: a * b."""
    return compute_mul([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division: a / b."""
    return compute_div([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


# ============================================================================
# Math Functions
# ============================================================================

def ttsim_sqrt(x: np.ndarray) -> np.ndarray:
    return compute_sqrt([TensorWrapper(x)], OpWrapper())


def ttsim_exp(x: np.ndarray) -> np.ndarray:
    return compute_exp([TensorWrapper(x)], OpWrapper())


def ttsim_log(x: np.ndarray) -> np.ndarray:
    return compute_log([TensorWrapper(x)], OpWrapper())


def ttsim_sin(x: np.ndarray) -> np.ndarray:
    """sin(x) — pure numpy (no TTSim compute_sin available)."""
    return np.sin(x).astype(x.dtype)


def ttsim_cos(x: np.ndarray) -> np.ndarray:
    """cos(x) — pure numpy (no TTSim compute_cos available)."""
    return np.cos(x).astype(x.dtype)


def ttsim_pow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return compute_pow([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_clip(x: np.ndarray, min_val, max_val) -> np.ndarray:
    """Clip x to [min_val, max_val]."""
    min_t = TensorWrapper(np.array(min_val, dtype=x.dtype))
    max_t = TensorWrapper(np.array(max_val, dtype=x.dtype))
    return compute_clip([TensorWrapper(x), min_t, max_t], OpWrapper())


# ============================================================================
# Shape Operations
# ============================================================================

def ttsim_reshape(x: np.ndarray, shape) -> np.ndarray:
    shape_t = TensorWrapper(np.array(shape, dtype=np.int64))
    return compute_reshape([TensorWrapper(x), shape_t], OpWrapper())


def ttsim_transpose(x: np.ndarray, axes) -> np.ndarray:
    return compute_transpose([TensorWrapper(x)], OpWrapper(perm=axes))


def ttsim_unsqueeze(x: np.ndarray, axes) -> np.ndarray:
    """Unsqueeze (insert dim) — pure numpy implementation."""
    axes_list = [axes] if isinstance(axes, int) else list(axes)
    out = x
    for ax in sorted(axes_list):
        out = np.expand_dims(out, axis=ax)
    return out


def ttsim_squeeze(x: np.ndarray, axes=None) -> np.ndarray:
    """Squeeze (remove dim) — pure numpy implementation."""
    if axes is None:
        return np.squeeze(x)
    axes_list = [axes] if isinstance(axes, int) else list(axes)
    out = x
    for ax in sorted(axes_list, reverse=True):
        out = np.squeeze(out, axis=ax)
    return out


# ============================================================================
# Aggregation
# ============================================================================

def ttsim_reducemean(x: np.ndarray, axis=None, keepdims: bool = True) -> np.ndarray:
    keepdims_val = 1 if keepdims else 0
    if axis is not None:
        axis_list = [axis] if isinstance(axis, int) else list(axis)
        axes_t = TensorWrapper(np.array(axis_list, dtype=np.int64))
        return compute_reducemean([TensorWrapper(x), axes_t],
                                  OpWrapper(keepdims=keepdims_val))
    return compute_reducemean([TensorWrapper(x)],
                              OpWrapper(keepdims=keepdims_val,
                                        noop_with_empty_axes=0))


def ttsim_reducesum(x: np.ndarray, axis=None, keepdims: bool = True) -> np.ndarray:
    """ReduceSum — pure numpy (no compute_reducesum available)."""
    if axis is None:
        result = np.sum(x)
        if keepdims:
            result = np.reshape(result, [1] * x.ndim)
    else:
        result = np.sum(x, axis=axis, keepdims=keepdims)
    return result.astype(x.dtype)


# ============================================================================
# Matrix Operations
# ============================================================================

def ttsim_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return compute_matmul([TensorWrapper(a), TensorWrapper(b)], OpWrapper())


def ttsim_concat(arrays, axis: int = 0) -> np.ndarray:
    return compute_concat([TensorWrapper(a) for a in arrays],
                          OpWrapper(axis=axis))


def ttsim_layernorm(x: np.ndarray, dim: int = -1, eps: float = 1e-5,
                    weight: np.ndarray = None, bias: np.ndarray = None) -> np.ndarray:
    """Layer normalisation over the last (or specified) axis, matching nn.LayerNorm."""
    mean = x.mean(axis=dim, keepdims=True)
    var  = ((x - mean) ** 2).mean(axis=dim, keepdims=True)
    y    = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


# ============================================================================
# Convolution Helpers
# ============================================================================

def ttsim_conv2d(x: np.ndarray, weight: np.ndarray,
                 bias: np.ndarray = None,
                 stride=1, padding=0, dilation=1, groups=1) -> np.ndarray:
    """2-D convolution via TTSim compute_conv2d."""
    stride_l  = [stride, stride]   if isinstance(stride,   int) else stride
    pad_l     = [padding]*4        if isinstance(padding,  int) else padding
    dilation_l= [dilation, dilation] if isinstance(dilation, int) else dilation
    op = OpWrapper(strides=stride_l, pads=pad_l, dilations=dilation_l, group=groups)
    inputs = [TensorWrapper(x), TensorWrapper(weight)]
    if bias is not None:
        inputs.append(TensorWrapper(bias))
    return compute_conv2d(inputs, op)


# ============================================================================
# mmcv / torch extracted — pure-NumPy reference implementations
# (extracted from mmcv.cnn.bricks.conv_module and torch.nn internals)
# ============================================================================

def conv1d_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> np.ndarray:
    """
    Pure-NumPy 1-D convolution.

    Extracted from torch.nn.Conv1d forward logic.
    Supports stride, padding, dilation and groups.

    Uses vectorised NumPy (einsum + strided slicing) — no Python
    loops over spatial positions, so it is fast for large L.

    Args:
        x:        Input  [N, C_in, L]
        weight:   Kernel [C_out, C_in/groups, K]
        bias:     Optional bias [C_out]
        stride:   Convolution stride.
        padding:  Zero-padding on each side.
        dilation: Kernel dilation.
        groups:   Number of blocked connections.

    Returns:
        np.ndarray of shape [N, C_out, L_out]
    """
    N, C_in, L = x.shape
    C_out, C_per_group, K = weight.shape
    assert C_in == C_per_group * groups, (
        f"C_in={C_in} must equal C_per_group*groups={C_per_group}*{groups}")

    # Apply padding
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    K_eff = dilation * (K - 1) + 1
    L_padded = x.shape[-1]
    L_out = (L_padded - K_eff) // stride + 1

    c_out_g = C_out // groups
    out = np.zeros((N, C_out, L_out), dtype=np.float32)

    for g in range(groups):
        x_g = x[:, g * C_per_group: (g + 1) * C_per_group, :]  # [N, Cg_in, L_padded]
        w_g = weight[g * c_out_g: (g + 1) * c_out_g]            # [Cg_out, Cg_in, K]

        for k in range(K):
            # For kernel position k, gather input slice at strided+dilated positions
            # Positions in the padded sequence: k*d, k*d+stride, ..., k*d+(L_out-1)*stride
            k_start = k * dilation
            x_slice = x_g[:, :, k_start: k_start + L_out * stride: stride]  # [N, Cg_in, L_out]
            # Accumulate: out[n, co, l] += sum_ci w_g[co, ci, k] * x_slice[n, ci, l]
            out[:, g * c_out_g: (g + 1) * c_out_g, :] += np.einsum(
                'nci,oc->noi', x_slice, w_g[:, :, k], optimize=True
            )

    if bias is not None:
        out = out + bias[np.newaxis, :, np.newaxis]

    return out


def batchnorm1d_numpy(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Pure-NumPy 1-D Batch Normalization (inference mode).

    Extracted from torch.nn.BatchNorm1d inference path.
    Handles input of shape (N, C) or (N, C, L).

    Formula:
        y = (x - running_mean) / sqrt(running_var + eps) * scale + bias

    Args:
        x:            Input [N, C] or [N, C, L].
        scale:        Learnable γ of shape [C].
        bias:         Learnable β of shape [C].
        running_mean: Running mean [C].
        running_var:  Running variance [C].
        eps:          Numerical stability constant.

    Returns:
        Normalized np.ndarray with same shape as x.
    """
    ndim = x.ndim
    # Reshape stats to broadcast correctly
    if ndim == 2:
        mu    = running_mean[np.newaxis, :]         # [1, C]
        var   = running_var [np.newaxis, :]         # [1, C]
        gamma = scale       [np.newaxis, :]         # [1, C]
        beta  = bias        [np.newaxis, :]         # [1, C]
    else:  # ndim == 3: shape (N, C, L)
        mu    = running_mean[np.newaxis, :, np.newaxis]  # [1, C, 1]
        var   = running_var [np.newaxis, :, np.newaxis]  # [1, C, 1]
        gamma = scale       [np.newaxis, :, np.newaxis]  # [1, C, 1]
        beta  = bias        [np.newaxis, :, np.newaxis]  # [1, C, 1]

    x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def ConvModule1d_numpy(
    x: np.ndarray,
    conv_weight: np.ndarray,
    conv_bias: np.ndarray,
    bn_scale: np.ndarray,
    bn_bias: np.ndarray,
    bn_running_mean: np.ndarray,
    bn_running_var: np.ndarray,
    stride: int = 1,
    padding: int = 0,
    eps: float = 1e-5,
    with_bn: bool = True,
    with_relu: bool = True,
) -> np.ndarray:
    """
    Pure-NumPy equivalent of mmcv.cnn.ConvModule for 1-D convolutions.

    Extracted from mmcv/cnn/bricks/conv_module.py (OpenMMLab):
        forward():
            if self.with_explicit_padding:
                x = self.padding_layer(x)
            x = self.conv(x,            # nn.Conv1d forward
                          activated_groups=self.norm_cfg is not None)
            if self.with_norm:
                x = self.norm(x)        # BatchNorm1d forward (inference)
            if self.with_activation:
                x = self.activate(x)    # ReLU
            return x

    This function implements exactly that sequence for:
        conv_cfg=dict(type='Conv1d')
        norm_cfg=dict(type='BN1d')
        act_cfg=dict(type='ReLU')

    Args:
        x:               Input [N, C_in, L].
        conv_weight:     Conv1d weight [C_out, C_in, kernel_size].
        conv_bias:       Conv1d bias [C_out] or None.
        bn_scale:        BN gamma [C_out]. Pass None to skip BN.
        bn_bias:         BN beta  [C_out].
        bn_running_mean: BN running mean [C_out].
        bn_running_var:  BN running var  [C_out].
        stride:          Conv stride.
        padding:         Conv padding.
        eps:             BN epsilon.
        with_bn:         Whether to apply BatchNorm.
        with_relu:       Whether to apply ReLU.

    Returns:
        np.ndarray of same shape as the mmcv.cnn.ConvModule output.
    """
    # 1. Conv1d
    out = conv1d_numpy(x, conv_weight, conv_bias, stride=stride, padding=padding)

    # 2. BatchNorm1d (inference mode uses running stats)
    if with_bn and bn_scale is not None:
        out = batchnorm1d_numpy(out, bn_scale, bn_bias,
                                bn_running_mean, bn_running_var, eps=eps)

    # 3. ReLU
    if with_relu:
        out = np.maximum(0, out)

    return out


# ============================================================================
# Comparison and Validation Helpers
# ============================================================================

def compare_arrays(
    reference: np.ndarray,
    ttsim_out: np.ndarray,
    name: str = "array",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """
    Compare a reference (numpy / PyTorch-derived) array against a TTSim output.

    Args:
        reference:  Expected values (numpy array or PyTorch tensor with .numpy()).
        ttsim_out:  Actual output from TTSim compute function (numpy array).
        name:       Label for this comparison (used in printed output).
        rtol:       Relative tolerance for np.allclose.
        atol:       Absolute tolerance for np.allclose.

    Returns:
        True if arrays match within tolerance, False otherwise.
    """
    # Accept PyTorch tensors as reference
    try:
        import torch
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().cpu().numpy()
    except ImportError:
        pass

    if reference.shape != ttsim_out.shape:
        print(f"  ✗ {name}: Shape mismatch — reference {reference.shape}, "
              f"TTSim {ttsim_out.shape}")
        return False

    if np.allclose(reference, ttsim_out, rtol=rtol, atol=atol):
        max_diff = float(np.max(np.abs(reference - ttsim_out)))
        print(f"  ✓ {name}: PASS  (max_diff={max_diff:.3e})")
        return True
    else:
        max_diff  = float(np.max(np.abs(reference - ttsim_out)))
        mean_diff = float(np.mean(np.abs(reference - ttsim_out)))
        print(f"  ✗ {name}: FAIL  (max_diff={max_diff:.3e}, "
              f"mean_diff={mean_diff:.3e}, rtol={rtol}, atol={atol})")
        return False


def print_header(title: str):
    """Print a wide section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test(title: str, msg: str = "", ok: bool = True):
    """Print a test-step header with optional detail and pass/fail indicator."""
    status = "✓" if ok else "✗"
    text = f"{status} {title}" + (f"  {msg}" if msg else "")
    print("\n" + text)
    print("-" * 80)
