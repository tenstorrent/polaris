#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Layer-specific Operations

This module provides high-level operations for different architectural components:
- Backbone operations (ResNet stem, residual blocks)
- FPN operations (feature pyramid network)
- Transformer operations (attention, FFN)
- Detection head operations
"""

import numpy as np
import torch
from .ttsim_utils import (
    ttsim_conv2d,
    ttsim_relu,
    ttsim_maxpool2d,
    ttsim_add,
    ttsim_interpolate,
    ttsim_matmul,
    ttsim_softmax,
    ttsim_layernorm,
    ttsim_reshape,
    ttsim_transpose,
    ttsim_mul,
    ttsim_reducemean,
    ttsim_div,
    compare_arrays,
)

# ============================================================================
# Backbone Operations
# ============================================================================


def backbone_stem(img_flat_ttsim, img_flat_torch, conv1_weight_np, verbose=True):
    """
    Backbone stem: Conv + ReLU + MaxPool

    Args:
        img_flat_ttsim: Input for TTSim [B*num_cams, 3, H, W]
        img_flat_torch: Input for PyTorch [B*num_cams, 3, H, W]
        conv1_weight_np: Convolution weights
        verbose: Print comparison results

    Returns:
        tuple: (ttsim_output, torch_output, validation_success)
    """
    # Step 1: Conv7x7 (using 3x3 for speed)
    conv1_output_torch = torch.nn.functional.conv2d(
        img_flat_torch, torch.from_numpy(conv1_weight_np), stride=1, padding=1
    )
    conv1_output_ttsim = ttsim_conv2d(
        img_flat_ttsim, conv1_weight_np, bias=None, stride=1, padding=1
    )

    if verbose:
        print(
            f"    Step 1: Conv7x7 - Input {img_flat_torch.shape} -> Output {conv1_output_torch.shape}"
        )
    match1 = (
        compare_arrays(
            conv1_output_torch, conv1_output_ttsim, "Conv7x7", rtol=1e-5, atol=1e-6
        )
        if verbose
        else True
    )

    # Step 2: ReLU activation
    relu_output_torch = torch.relu(conv1_output_torch)
    relu_output_ttsim = ttsim_relu(conv1_output_ttsim)

    if verbose:
        print(f"    Step 2: ReLU - Activating negative values to zero")
        print(
            f"      PyTorch: min={relu_output_torch.min():.4f}, max={relu_output_torch.max():.4f}"
        )
        print(
            f"      TTSim:   min={relu_output_ttsim.min():.4f}, max={relu_output_ttsim.max():.4f}"
        )
    match2 = (
        compare_arrays(relu_output_torch, relu_output_ttsim, "ReLU")
        if verbose
        else True
    )

    # Step 3: MaxPool
    output_torch = torch.nn.functional.max_pool2d(
        relu_output_torch, 2, stride=1, padding=0
    )
    output_ttsim = ttsim_maxpool2d(
        relu_output_ttsim, kernel_size=2, stride=1, padding=0
    )

    if verbose:
        print(
            f"    Step 3: MaxPool3x3 - Downsampling {relu_output_torch.shape} -> {output_torch.shape}"
        )
    match3 = compare_arrays(output_torch, output_ttsim, "MaxPool") if verbose else True

    return output_ttsim, output_torch, (match1 and match2 and match3)


# ============================================================================
# FPN Operations
# ============================================================================


def fpn_top_down_fusion(
    top_feature_ttsim, top_feature_torch, lateral_ttsim, lateral_torch, verbose=True
):
    """
    FPN top-down pathway: upsample top feature and add to lateral connection

    Args:
        top_feature_ttsim: Higher-level feature (TTSim)
        top_feature_torch: Higher-level feature (PyTorch)
        lateral_ttsim: Lateral connection feature (TTSim)
        lateral_torch: Lateral connection feature (PyTorch)
        verbose: Print comparison results

    Returns:
        tuple: (fused_ttsim, fused_torch, validation_success)
    """
    # Upsample top feature to match lateral size
    upsampled_torch = torch.nn.functional.interpolate(
        top_feature_torch, size=lateral_torch.shape[-2:], mode="nearest"
    )
    upsampled_ttsim = ttsim_interpolate(
        top_feature_ttsim, size=lateral_ttsim.shape[-2:], mode="nearest"
    )

    # Fuse features
    fused_torch = lateral_torch + upsampled_torch
    fused_ttsim = ttsim_add(lateral_ttsim, upsampled_ttsim)

    match = True
    if verbose:
        match = compare_arrays(
            fused_torch, fused_ttsim, "FPN Fusion", rtol=1e-5, atol=1e-6
        )

    return fused_ttsim, fused_torch, match


def fpn_stride2_downsample(feature_ttsim, feature_torch, verbose=True):
    """
    Create extra FPN level by stride-2 max pooling

    Args:
        feature_ttsim: Input feature (TTSim)
        feature_torch: Input feature (PyTorch)
        verbose: Print comparison results

    Returns:
        tuple: (downsampled_ttsim, downsampled_torch, validation_success)
    """
    downsampled_torch = torch.nn.functional.max_pool2d(feature_torch, 1, stride=2)
    downsampled_ttsim = ttsim_maxpool2d(
        feature_ttsim, kernel_size=1, stride=2, padding=0
    )

    match = True
    if verbose:
        match = compare_arrays(
            downsampled_torch, downsampled_ttsim, "FPN Downsample", rtol=1e-5, atol=1e-6
        )

    return downsampled_ttsim, downsampled_torch, match


# ============================================================================
# Transformer Operations
# ============================================================================


def multi_head_attention(
    query_ttsim,
    query_torch,
    key_ttsim,
    key_torch,
    value_ttsim,
    value_torch,
    num_heads,
    verbose=True,
):
    """
    Multi-head attention computation

    Args:
        query_ttsim, query_torch: Query tensors [B, N, D]
        key_ttsim, key_torch: Key tensors [B, M, D]
        value_ttsim, value_torch: Value tensors [B, M, D]
        num_heads: Number of attention heads
        verbose: Print comparison results

    Returns:
        tuple: (output_ttsim, output_torch, validation_success)
    """
    B, N, D = query_torch.shape
    _, M, _ = key_torch.shape
    head_dim = D // num_heads

    # Reshape for multi-head: [B, num_heads, N, head_dim]
    q_torch = query_torch.reshape(B, N, num_heads, head_dim).transpose(1, 2)
    k_torch = key_torch.reshape(B, M, num_heads, head_dim).transpose(1, 2)
    v_torch = value_torch.reshape(B, M, num_heads, head_dim).transpose(1, 2)

    q_ttsim = ttsim_reshape(query_ttsim, (B, N, num_heads, head_dim))
    q_ttsim = ttsim_transpose(q_ttsim, (0, 2, 1, 3))
    k_ttsim = ttsim_reshape(key_ttsim, (B, M, num_heads, head_dim))
    k_ttsim = ttsim_transpose(k_ttsim, (0, 2, 1, 3))
    v_ttsim = ttsim_reshape(value_ttsim, (B, M, num_heads, head_dim))
    v_ttsim = ttsim_transpose(v_ttsim, (0, 2, 1, 3))

    # Attention scores: Q @ K^T / sqrt(d_k)
    scale = 1.0 / np.sqrt(head_dim)
    attn_torch = torch.matmul(q_torch, k_torch.transpose(-2, -1)) * scale

    # For TTSim, we need to transpose K and multiply
    from .ttsim_utils import ttsim_div

    k_t_ttsim = ttsim_transpose(k_ttsim, (0, 1, 3, 2))
    attn_ttsim = ttsim_matmul(q_ttsim, k_t_ttsim)
    attn_ttsim = ttsim_div(
        attn_ttsim, np.array(np.sqrt(head_dim), dtype=attn_ttsim.dtype)
    )

    # Softmax
    attn_weights_torch = torch.softmax(attn_torch, dim=-1)
    attn_weights_ttsim = ttsim_softmax(attn_ttsim, axis=-1)

    # Apply attention to values
    output_torch = torch.matmul(attn_weights_torch, v_torch)
    output_ttsim = ttsim_matmul(attn_weights_ttsim, v_ttsim)

    # Reshape back: [B, num_heads, N, head_dim] -> [B, N, D]
    output_torch = output_torch.transpose(1, 2).reshape(B, N, D)
    output_ttsim = ttsim_transpose(output_ttsim, (0, 2, 1, 3))
    output_ttsim = ttsim_reshape(output_ttsim, (B, N, D))

    match = True
    if verbose:
        match = compare_arrays(
            output_torch, output_ttsim, "Multi-Head Attention", rtol=1e-4, atol=1e-5
        )

    return output_ttsim, output_torch, match


def feedforward_network(
    x_ttsim,
    x_torch,
    linear1_weight,
    linear2_weight,
    linear1_bias=None,
    linear2_bias=None,
    verbose=True,
):
    """
    Feedforward network: Linear -> ReLU -> Linear

    Args:
        x_ttsim, x_torch: Input tensors [B, N, D]
        linear1_weight: First linear layer weight [D, D_ffn] (TTSim format)
        linear2_weight: Second linear layer weight [D_ffn, D] (TTSim format)
        linear1_bias: Optional bias for first layer [D_ffn]
        linear2_bias: Optional bias for second layer [D]
        verbose: Print comparison results

    Returns:
        tuple: (output_ttsim, output_torch, validation_success)
    """
    from .ttsim_utils import ttsim_add, ttsim_relu

    # Step 1: First linear layer (expand)
    if verbose:
        print(
            f"     * Step 1: Linear expand ({linear1_weight.shape[0]} -> {linear1_weight.shape[1]})"
        )

    # PyTorch: uses weight @ x format
    h_torch = x_torch @ torch.from_numpy(linear1_weight)
    if linear1_bias is not None:
        h_torch = h_torch + torch.from_numpy(linear1_bias)

    # TTSim: uses matmul
    h_ttsim = ttsim_matmul(x_ttsim, linear1_weight)
    if linear1_bias is not None:
        h_ttsim = ttsim_add(h_ttsim, linear1_bias)

    match1 = (
        compare_arrays(h_torch, h_ttsim, "       Linear1", rtol=1e-5, atol=1e-6)
        if verbose
        else True
    )

    # Step 2: ReLU activation
    if verbose:
        print(f"     * Step 2: ReLU activation")
    h_torch = torch.relu(h_torch)
    h_ttsim = ttsim_relu(h_ttsim)
    match2 = (
        compare_arrays(h_torch, h_ttsim, "       ReLU", rtol=1e-5, atol=1e-6)
        if verbose
        else True
    )

    # Step 3: Second linear layer (contract)
    if verbose:
        print(
            f"     * Step 3: Linear contract ({linear2_weight.shape[0]} -> {linear2_weight.shape[1]})"
        )

    output_torch = h_torch @ torch.from_numpy(linear2_weight)
    if linear2_bias is not None:
        output_torch = output_torch + torch.from_numpy(linear2_bias)

    output_ttsim = ttsim_matmul(h_ttsim, linear2_weight)
    if linear2_bias is not None:
        output_ttsim = ttsim_add(output_ttsim, linear2_bias)

    match3 = (
        compare_arrays(
            output_torch, output_ttsim, "       Linear2", rtol=1e-5, atol=1e-6
        )
        if verbose
        else True
    )

    match = match1 and match2 and match3
    return output_ttsim, output_torch, match


def transformer_layer_with_residual(
    x_ttsim, x_torch, operation_func, *args, verbose=True
):
    """
    Apply transformer operation with residual connection: output = x + operation(x)

    Args:
        x_ttsim, x_torch: Input tensors
        operation_func: Function that takes (ttsim_input, torch_input, *args) and returns (ttsim_out, torch_out, match)
        *args: Additional arguments for operation_func
        verbose: Print comparison results

    Returns:
        tuple: (output_ttsim, output_torch, validation_success)
    """
    # Apply operation
    op_out_ttsim, op_out_torch, match1 = operation_func(
        x_ttsim, x_torch, *args, verbose=verbose
    )

    # Add residual
    output_torch = x_torch + op_out_torch
    output_ttsim = ttsim_add(x_ttsim, op_out_ttsim)

    match2 = True
    if verbose:
        match2 = compare_arrays(
            output_torch, output_ttsim, "Residual Add", rtol=1e-5, atol=1e-6
        )

    return output_ttsim, output_torch, (match1 and match2)


# ============================================================================
# Detection Head Operations
# ============================================================================


def classification_head(
    features_ttsim, features_torch, fc_weight, fc_bias=None, verbose=True
):
    """
    Classification head: Linear -> Softmax (for multi-class classification)
    or Linear -> Sigmoid (for binary classification)

    Args:
        features_ttsim, features_torch: Input features [B, N, D]
        fc_weight: Classification weight [num_classes, D]
        fc_bias: Optional classification bias [num_classes]
        verbose: Print comparison results

    Returns:
        tuple: (logits_ttsim, logits_torch, validation_success)
    """
    # Linear projection
    logits_torch = torch.nn.functional.linear(
        features_torch,
        torch.from_numpy(fc_weight),
        torch.from_numpy(fc_bias) if fc_bias is not None else None,
    )

    # TTSim: [B, N, D] @ [D, num_classes] -> [B, N, num_classes]
    features_flat_ttsim = ttsim_reshape(features_ttsim, (-1, features_ttsim.shape[-1]))
    w_t = fc_weight.T
    logits_flat_ttsim = ttsim_matmul(features_flat_ttsim, w_t)

    if fc_bias is not None:
        logits_flat_ttsim = ttsim_add(logits_flat_ttsim, fc_bias)

    logits_ttsim = ttsim_reshape(
        logits_flat_ttsim, features_ttsim.shape[:2] + (fc_weight.shape[0],)
    )

    match = True
    if verbose:
        match = compare_arrays(
            logits_torch, logits_ttsim, "Classification Head", rtol=1e-5, atol=1e-6
        )

    return logits_ttsim, logits_torch, match


def regression_head(
    features_ttsim, features_torch, fc_weight, fc_bias=None, verbose=True
):
    """
    Regression head: Linear (for bounding box regression)

    Args:
        features_ttsim, features_torch: Input features [B, N, D]
        fc_weight: Regression weight [code_size, D]
        fc_bias: Optional regression bias [code_size]
        verbose: Print comparison results

    Returns:
        tuple: (regression_ttsim, regression_torch, validation_success)
    """
    # Linear projection
    regression_torch = torch.nn.functional.linear(
        features_torch,
        torch.from_numpy(fc_weight),
        torch.from_numpy(fc_bias) if fc_bias is not None else None,
    )

    # TTSim: [B, N, D] @ [D, code_size] -> [B, N, code_size]
    features_flat_ttsim = ttsim_reshape(features_ttsim, (-1, features_ttsim.shape[-1]))
    w_t = fc_weight.T
    regression_flat_ttsim = ttsim_matmul(features_flat_ttsim, w_t)

    if fc_bias is not None:
        regression_flat_ttsim = ttsim_add(regression_flat_ttsim, fc_bias)

    regression_ttsim = ttsim_reshape(
        regression_flat_ttsim, features_ttsim.shape[:2] + (fc_weight.shape[0],)
    )

    match = True
    if verbose:
        match = compare_arrays(
            regression_torch, regression_ttsim, "Regression Head", rtol=1e-5, atol=1e-6
        )

    return regression_ttsim, regression_torch, match
