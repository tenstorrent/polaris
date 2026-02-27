#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Data computation helpers for shape inference functions"""

import numpy as np
from typing import List, Optional

##############################################################################
# MS DEFORMABLE ATTENTION ADDITIONS for data_compute.py
##############################################################################
# Add these 2 functions to ttsim/ops/desc/data_compute.py
# These are generic compute helpers that can be reused for other models.
##############################################################################

import numpy as np
from typing import List, Optional


def try_compute_data(compute_func, iTList, op):
    """
    Wrapper to safely compute data if all inputs have data.

    Args:
        compute_func: Function that computes output data from inputs
        iTList: List of input tensors
        op: SimOp instance

    Returns:
        Computed numpy array if all inputs have data, None otherwise
    """
    # Check if all required inputs have data
    if all(t.data is not None for t in iTList):
        try:
            return compute_func(iTList, op)
        except Exception as e:
            # Data computation failed, return None
            # Shape inference still works!
            import warnings

            warnings.warn(f"Data computation failed for {op.optype}: {e}")
            return None


#     return None


def compute_maxpool2d(iTList, op) -> np.ndarray:
    """
    Compute MaxPool2d output using pure NumPy.

    Args:
        iTList: [X] where X is [N, C, H, W]
        op: SimOp with attrs kernel_shape, strides, pads

    Returns:
        Y: MaxPool output [N, C, H_out, W_out]
    """
    X = iTList[0].data

    # Get pooling parameters
    kernel_shape = op.attrs.get("kernel_shape", [2, 2])
    strides = op.attrs.get("strides", kernel_shape)
    pads = op.attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]

    N, C, H_in, W_in = X.shape
    Kh, Kw = kernel_shape

    # Apply padding
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(
            X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant", constant_values=-np.inf
        )
    else:
        X_padded = X

    # Calculate output size
    H_out = (H_in + pads[0] + pads[2] - Kh) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - Kw) // strides[1] + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # Perform max pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    pool_region = X_padded[
                        n, c, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c, h, w] = np.max(pool_region)

    return Y


def compute_concat(iTList, op) -> np.ndarray:
    """
    Compute concatenation along specified axis.

    Args:
        iTList: List of input tensors to concatenate
        op: SimOp with attrs axis

    Returns:
        Y: Concatenated output
    """
    axis = op.attrs.get("axis", 1)
    arrays = [t.data for t in iTList]
    return np.concatenate(arrays, axis=axis)


def compute_expand(iTList, op, target_shape) -> np.ndarray:
    """
    Compute expand (broadcast) output.

    Args:
        iTList: [A, B] where:
            A: input tensor to expand
            B: tensor containing target shape
        op: SimOp
        target_shape: list of target dimensions

    Returns:
        Y: Expanded (broadcasted) tensor
    """
    X = iTList[0].data
    # Use numpy broadcast_to for expansion, then copy to ensure contiguous memory
    # broadcast_to returns a view with stride=0 for repeated dims which can cause issues
    return np.ascontiguousarray(np.broadcast_to(X, target_shape))


def compute_add(iTList, op) -> np.ndarray:
    """Element-wise addition with broadcasting"""
    return iTList[0].data + iTList[1].data


def compute_resize(iTList, op) -> np.ndarray | None:
    """
    Compute Resize output using nearest neighbor interpolation.

    Args:
        iTList: [X, roi, scales] where:
            X: input tensor [N, C, H, W] (4D) or any rank
            roi: region of interest (empty for our use case)
            scales: scale factors for spatial dimensions
        op: SimOp with attrs mode (default 'nearest')

    Returns:
        Y: Resized output tensor
    """
    X = iTList[0].data
    scales = iTList[2].data  # [scale_h, scale_w] for spatial dims
    mode = op.attrs.get("mode", "nearest")

    if X.ndim == 4:
        # [N, C, H, W] case - most common
        N, C, H_in, W_in = X.shape
        # scales contains [scale_h, scale_w] for spatial dimensions only
        scale_h = float(scales[-2]) if len(scales) >= 2 else float(scales[-1])
        scale_w = float(scales[-1])
        H_out = int(H_in * scale_h)
        W_out = int(W_in * scale_w)

        if mode == "nearest":
            # Nearest neighbor interpolation
            Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
            for h in range(H_out):
                for w in range(W_out):
                    src_h = int(h / scale_h)
                    src_w = int(w / scale_w)
                    # Clamp to valid range
                    src_h = min(src_h, H_in - 1)
                    src_w = min(src_w, W_in - 1)
                    Y[:, :, h, w] = X[:, :, src_h, src_w]
            return Y
        else:
            # Other modes not implemented - return None to indicate
            return None
    else:
        # General case for other ranks - not implemented
        return None


def compute_mul(iTList, op) -> np.ndarray:
    """Element-wise multiplication with broadcasting"""
    return iTList[0].data * iTList[1].data


def compute_multihead_attention(iTList, op):
    """
    Compute multi-head attention output numerically.

    Implements scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    When projection weights are provided in op.attrs (in_proj_weight_data, out_proj_weight_data),
    applies input and output projections to match PyTorch nn.MultiheadAttention behavior:
        projected_q = query @ W_q + b_q
        projected_k = key   @ W_k + b_k
        projected_v = value  @ W_v + b_v
        attn_output = Attention(projected_q, projected_k, projected_v)
        output = attn_output @ W_out + b_out

    Args:
        iTList: [query, key, value, key_padding_mask (optional), attn_mask (optional)]
            query: [*, tgt_len, embed_dim]
            key: [*, src_len, embed_dim]
            value: [*, src_len, embed_dim]
        op: SimOp with attrs (embed_dim, num_heads, dropout)
            Optional attrs for projections:
                in_proj_weight_data: [embed_dim, 3*embed_dim] (TTSim transposed format)
                in_proj_bias_data: [3*embed_dim]
                out_proj_weight_data: [embed_dim, embed_dim] (TTSim transposed format)
                out_proj_bias_data: [embed_dim]

    Returns:
        (output, attn_weights)
            output: [*, tgt_len, embed_dim]
            attn_weights: [*, num_heads, tgt_len, src_len]
    """
    query = iTList[0].data
    key = iTList[1].data
    value = iTList[2].data

    embed_dim = op.attrs.get("embed_dim")
    num_heads = op.attrs.get("num_heads")
    dropout_p = op.attrs.get("dropout", 0.0)

    # Get optional masks
    key_padding_mask = (
        iTList[3].data if len(iTList) > 3 and iTList[3].data is not None else None
    )
    attn_mask = (
        iTList[4].data if len(iTList) > 4 and iTList[4].data is not None else None
    )

    # Get optional projection weights from attrs
    W_in = op.attrs.get("in_proj_weight_data", None)
    b_in = op.attrs.get("in_proj_bias_data", None)
    W_out = op.attrs.get("out_proj_weight_data", None)
    b_out = op.attrs.get("out_proj_bias_data", None)

    head_dim = embed_dim // num_heads
    assert embed_dim == num_heads * head_dim, "embed_dim must be divisible by num_heads"

    # Apply input projections if weights are available
    # in_proj_weight is [E, 3E] in TTSim format: query @ W[:, :E] gives projected query
    if W_in is not None:
        E = embed_dim
        query = query @ W_in[:, :E]
        if b_in is not None:
            query = query + b_in[:E]
        key = key @ W_in[:, E : 2 * E]
        if b_in is not None:
            key = key + b_in[E : 2 * E]
        value = value @ W_in[:, 2 * E :]
        if b_in is not None:
            value = value + b_in[2 * E :]

    # Get shapes
    batch_shape = query.shape[:-2]
    tgt_len = query.shape[-2]
    src_len = key.shape[-2]
    batch_size = np.prod(batch_shape) if len(batch_shape) > 0 else 1

    # Reshape for multi-head: [batch, seq, embed] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = query.reshape(list(batch_shape) + [tgt_len, num_heads, head_dim])
    k = key.reshape(list(batch_shape) + [src_len, num_heads, head_dim])
    v = value.reshape(list(batch_shape) + [src_len, num_heads, head_dim])

    # Transpose to [batch, heads, seq, head_dim]
    axes = list(range(len(batch_shape))) + [
        len(batch_shape) + 1,
        len(batch_shape),
        len(batch_shape) + 2,
    ]
    q = np.transpose(q, axes)
    k = np.transpose(k, axes)
    v = np.transpose(v, axes)

    # Scaled dot-product attention: scores = Q @ K^T / sqrt(d_k)
    # [batch, heads, tgt_len, head_dim] @ [batch, heads, head_dim, src_len] = [batch, heads, tgt_len, src_len]
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) / np.sqrt(head_dim)

    # Apply attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Apply key padding mask if provided
    if key_padding_mask is not None:
        # Reshape mask to [batch, 1, 1, src_len] for broadcasting
        mask_reshape = [int(batch_size), 1, 1, int(src_len)]
        key_padding_mask_expanded = key_padding_mask.reshape(mask_reshape)
        scores = np.where(key_padding_mask_expanded, -1e9, scores)

    # Softmax over source sequence dimension
    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / (np.sum(attn_weights, axis=-1, keepdims=True) + 1e-9)

    # Apply dropout to attention weights (training mode)
    # For inference/shape inference, skip dropout or use dropout_p=0
    if dropout_p > 0:
        mask = np.random.binomial(1, 1 - dropout_p, attn_weights.shape)
        attn_weights = attn_weights * mask / (1 - dropout_p)

    # Apply attention: [batch, heads, tgt_len, src_len] @ [batch, heads, src_len, head_dim] = [batch, heads, tgt_len, head_dim]
    output = np.matmul(attn_weights, v)

    # Reshape back: [batch, heads, tgt_len, head_dim] -> [batch, tgt_len, heads, head_dim] -> [batch, tgt_len, embed]
    axes_back = list(range(len(batch_shape))) + [
        len(batch_shape) + 1,
        len(batch_shape),
        len(batch_shape) + 2,
    ]
    output = np.transpose(output, axes_back)
    output = output.reshape(list(batch_shape) + [tgt_len, embed_dim])

    # Apply output projection if weights are available
    # out_proj_weight is [E, E] in TTSim format: output @ W gives projected output
    if W_out is not None:
        output = output @ W_out
        if b_out is not None:
            output = output + b_out

    return output, attn_weights


def compute_dropout(iTList, op):
    """
    Compute Dropout output: applies dropout mask to input during training.

    During training: output = input * mask / (1 - ratio)
    During inference or ratio=0: output = input (identity)

    Args:
        iTList: [X, ratio (optional), training_mode (optional)]
            X: [*] - Input tensor of any shape
            ratio: scalar - Dropout probability (default: 0.5)
            training_mode: bool - Whether in training mode (default: False)
        op: SimOp with attrs:
            - seed: int - Random seed for reproducibility (default: 1.0)
            - prob: float - Dropout probability (if set by F.Dropout)
            - train_mode: bool - Training mode (if set by F.Dropout)

    Returns:
        output: [*] - Dropped-out tensor (or identity if ratio=0 or training_mode=False)
        mask: [*] - Boolean mask (if return_mask=True)

    Note: Returns tuple (output, mask) if len(oTList) == 2, else just output
    """
    X = iTList[0].data
    seed = op.attrs.get("seed", 1.0)

    # Priority 1: Check op.attrs for 'prob' and 'train_mode' (set by F.Dropout)
    # Priority 2: Fall back to iTList for ONNX-style inputs
    # Priority 3: Use safe defaults (no dropout)
    ratio = op.attrs.get("prob", None)
    training_mode = op.attrs.get("train_mode", None)

    # If not in attrs, try to get from iTList (ONNX-style)
    if ratio is None:
        if len(iTList) >= 2 and iTList[1].data is not None:
            ratio = float(iTList[1].data)
        else:
            ratio = 0.0  # Default: no dropout (safer default)

    if training_mode is None:
        if len(iTList) >= 3 and iTList[2].data is not None:
            training_mode = bool(iTList[2].data)
        else:
            training_mode = False  # Default: inference mode

    # Apply dropout only if:
    # 1. In training mode (training_mode=True)
    # 2. AND ratio > 0 (non-zero dropout probability)
    # Otherwise, dropout is a no-op (identity function)
    if not training_mode or ratio <= 0:
        # No dropout: output = input (identity)
        np_out = X.copy()
        np_mask_out = np.ones(X.shape, dtype=bool)
    else:
        # Apply dropout with mask
        np.random.seed(int(seed))
        mask = np.random.uniform(0, 1.0, X.shape) >= ratio
        scale = 1.0 / (1.0 - ratio)
        np_out = mask * X * scale
        np_mask_out = mask.astype(bool)

    return np_out, np_mask_out


def compute_layernorm(iTList, op) -> np.ndarray:
    """
    Compute LayerNormalization output: (X - mean) / sqrt(var + eps)  * scale + bias

    Args:
        iTList: [X, scale, bias (optional)]
            X: [*, normalized_shape] - Input tensor
            scale: [normalized_shape] - Learnable scale parameter
            bias: [normalized_shape] - Learnable bias parameter (optional)
        op: SimOp with attrs:
            - axis: int - First dimension to normalize (default: -1)
            - epsilon: float - Small value for numerical stability (default: 1e-5)

    Returns:
        Y: [*, normalized_shape] - Normalized output

    Note: Normalized axes are [axis, ..., rank-1]
    """
    X = iTList[0].data
    scale = iTList[1].data
    bias = iTList[2].data if len(iTList) >= 3 else None

    axis = op.attrs.get("axis", -1)
    epsilon = op.attrs.get("epsilon", 1e-5)

    # Handle negative axis
    if axis < 0:
        axis += X.ndim

    # Compute normalized axes
    normalized_axes = tuple(range(axis, X.ndim))

    # Compute mean and variance over normalized axes
    mean = np.mean(X, axis=normalized_axes, keepdims=True)
    variance = np.var(X, axis=normalized_axes, keepdims=True)

    # Normalize: (X - mean) / sqrt(var + eps)
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)

    # Scale and shift
    # Broadcasting: scale and bias have shape [normalized_shape]
    # X_normalized has shape [*, normalized_shape]
    # Need to reshape scale/bias for broadcasting
    broadcast_shape = [1] * axis + list(scale.shape)
    scale_reshaped = scale.reshape(broadcast_shape)

    Y = X_normalized * scale_reshaped

    if bias is not None:
        bias_reshaped = bias.reshape(broadcast_shape)
        Y = Y + bias_reshaped

    return Y


def compute_mish(iTList, op) -> np.ndarray:
    """
    Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

    Uses numerical stability tricks:
    - Clip input to prevent overflow in exp
    - Use log1p for better precision
    """
    X = iTList[0].data
    # Clip to prevent overflow in exp
    X_clipped = np.clip(X, -20, 20)
    # softplus(x) = ln(1 + e^x) - use log1p for stability
    softplus = np.log1p(np.exp(X_clipped))
    return X * np.tanh(softplus)


def compute_sigmoid(iTList, op) -> np.ndarray:
    """Sigmoid activation: 1 / (1 + e^(-x))"""
    X = iTList[0].data
    return 1.0 / (1.0 + np.exp(-np.clip(X, -20, 20)))


def compute_relu(iTList, op) -> np.ndarray:
    """ReLU activation: max(0, x)"""
    return np.maximum(0, iTList[0].data)


def compute_identity(iTList, op) -> np.ndarray:
    """Identity operation: returns input unchanged"""
    return iTList[0].data.copy()


def compute_batchnorm(iTList, op) -> np.ndarray:
    """
    BatchNorm: (x - mean) / sqrt(var + eps) * scale + bias

    Args:
        iTList: [X, scale, bias, mean, var] where X is [N, C, H, W]
        op: SimOp with attrs epsilon

    Returns:
        Y: Normalized output [N, C, H, W]
    """
    X = iTList[0].data  # [N, C, H, W]
    scale = iTList[1].data  # [C]
    bias = iTList[2].data  # [C]
    mean = iTList[3].data  # [C]
    var = iTList[4].data  # [C]

    eps = op.attrs.get("epsilon", 1e-5)

    # Normalize
    X_normalized = (X - mean.reshape(1, -1, 1, 1)) / np.sqrt(
        var.reshape(1, -1, 1, 1) + eps
    )

    # Scale and shift
    return scale.reshape(1, -1, 1, 1) * X_normalized + bias.reshape(1, -1, 1, 1)


def compute_conv2d(iTList, op) -> np.ndarray:
    """
    Compute Conv2d output using pure NumPy.

    Args:
        iTList: [X, W] or [X, W, B] where:
            X: input [N, C_in, H, W]
            W: weights [C_out, C_in/group, Kh, Kw]
            B: optional bias [C_out]
        op: SimOp with attrs strides, pads, dilations, group

    Returns:
        Y: Conv output [N, C_out, H_out, W_out]
    """
    X = iTList[0].data
    W = iTList[1].data
    B = iTList[2].data if len(iTList) > 2 else None

    strides = op.attrs.get("strides", [1, 1])
    pads = op.attrs.get("pads", [0, 0, 0, 0])  # [top, left, bottom, right]
    dilations = op.attrs.get("dilations", [1, 1])
    group = op.attrs.get("group", 1)

    N, C_in, H_in, W_in = X.shape
    C_out, C_per_group, Kh, Kw = W.shape

    # Apply padding
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    else:
        X_padded = X

    # Calculate output dimensions
    H_out = (H_in + pads[0] + pads[2] - dilations[0] * (Kh - 1) - 1) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - dilations[1] * (Kw - 1) - 1) // strides[1] + 1

    Y = np.zeros((N, C_out, H_out, W_out), dtype=X.dtype)

    if group == 1:
        # Standard convolution
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * strides[0]
                        w_start = w * strides[1]

                        # Extract receptive field with dilation
                        conv_sum = 0.0
                        for kh in range(Kh):
                            for kw in range(Kw):
                                h_idx = h_start + kh * dilations[0]
                                w_idx = w_start + kw * dilations[1]
                                for c_in in range(C_in):
                                    conv_sum += (
                                        X_padded[n, c_in, h_idx, w_idx]
                                        * W[c_out, c_in, kh, kw]
                                    )
                        Y[n, c_out, h, w] = conv_sum
    else:
        # Grouped convolution
        C_in_per_group = C_in // group
        C_out_per_group = C_out // group

        for g in range(group):
            c_in_start = g * C_in_per_group
            c_in_end = (g + 1) * C_in_per_group
            c_out_start = g * C_out_per_group
            c_out_end = (g + 1) * C_out_per_group

            for n in range(N):
                for c_out_local in range(C_out_per_group):
                    c_out = c_out_start + c_out_local
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * strides[0]
                            w_start = w * strides[1]

                            conv_sum = 0.0
                            for kh in range(Kh):
                                for kw in range(Kw):
                                    h_idx = h_start + kh * dilations[0]
                                    w_idx = w_start + kw * dilations[1]
                                    for c_in_local in range(C_in_per_group):
                                        c_in = c_in_start + c_in_local
                                        conv_sum += (
                                            X_padded[n, c_in, h_idx, w_idx]
                                            * W[c_out, c_in_local, kh, kw]
                                        )
                            Y[n, c_out, h, w] = conv_sum

    # Add bias if present
    if B is not None:
        Y += B.reshape(1, -1, 1, 1)

    return Y


def compute_matmul(iTList, op) -> np.ndarray:
    """Matrix multiplication with broadcasting"""
    A = iTList[0].data
    B = iTList[1].data
    return np.matmul(A, B)


def compute_linear(iTList, op) -> np.ndarray:
    """
    Compute Linear layer (fully connected): y = xW^T + b

    Args:
        iTList: [input, weight] or [input, weight, bias]
            input: [..., in_features]
            weight: [out_features, in_features]
            bias: [out_features] (optional)

    Returns:
        output: [..., out_features]
    """
    input_data = iTList[0].data
    weight_data = iTList[1].data

    # Compute xW^T
    output = np.matmul(input_data, weight_data.T)

    # Add bias if present
    if len(iTList) > 2 and iTList[2].data is not None:
        bias_data = iTList[2].data
        output = output + bias_data

    return output


def compute_avgpool2d(iTList, op) -> np.ndarray:
    """
    Compute AvgPool2d output using pure NumPy.

    Args:
        iTList: [X] where X is [N, C, H, W]
        op: SimOp with attrs kernel_shape, strides, pads

    Returns:
        Y: AvgPool output [N, C, H_out, W_out]
    """
    X = iTList[0].data

    kernel_shape = op.attrs.get("kernel_shape", [2, 2])
    strides = op.attrs.get("strides", kernel_shape)
    pads = op.attrs.get("pads", [0, 0, 0, 0])

    N, C, H_in, W_in = X.shape
    Kh, Kw = kernel_shape

    # Apply padding
    pad_h = (pads[0], pads[2])
    pad_w = (pads[1], pads[3])

    if any(p > 0 for p in pads):
        X_padded = np.pad(X, ((0, 0), (0, 0), pad_h, pad_w), mode="constant")
    else:
        X_padded = X

    # Calculate output size
    H_out = (H_in + pads[0] + pads[2] - Kh) // strides[0] + 1
    W_out = (W_in + pads[1] + pads[3] - Kw) // strides[1] + 1

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    # Perform average pooling
    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * strides[0]
                    w_start = w * strides[1]
                    pool_region = X_padded[
                        n, c, h_start : h_start + Kh, w_start : w_start + Kw
                    ]
                    Y[n, c, h, w] = np.mean(pool_region)

    return Y


def compute_slice(iTList, op) -> np.ndarray:
    """
    Compute Slice operation.

    Args:
        iTList: [data, starts, ends, axes, steps]
        op: SimOp

    Returns:
        Y: Sliced output
    """
    data = iTList[0].data
    starts = iTList[1].data.astype(np.int64)
    ends = iTList[2].data.astype(np.int64)
    axes = iTList[3].data.astype(np.int64) if len(iTList) > 3 else None
    steps = iTList[4].data.astype(np.int64) if len(iTList) > 4 else None

    # Build slice objects
    slices = [slice(None)] * len(data.shape)

    if axes is None:
        axes = np.arange(len(starts))

    for i, axis in enumerate(axes):
        start = starts[i]
        end = ends[i]
        step = steps[i] if steps is not None else 1
        slices[axis] = slice(start, end, step)

    return data[tuple(slices)]


def compute_transpose(iTList, op) -> np.ndarray:
    """
    Compute Transpose operation.

    Args:
        iTList: [data]
        op: SimOp with attrs perm

    Returns:
        Y: Transposed output
    """
    # Check if this is shape inference only
    if iTList[0].data is None:
        input_shape = iTList[0].shape
        perm = op.attrs.get("perm", None)
        if perm is None:
            perm = list(range(len(input_shape) - 1, -1, -1))
        output_shape = tuple(input_shape[i] for i in perm)
        return np.empty(output_shape)

    data = iTList[0].data
    perm = op.attrs.get("perm", None)

    if perm is None:
        # Default: reverse all dimensions
        perm = list(range(len(data.shape) - 1, -1, -1))

    return np.transpose(data, perm)


def compute_split(iTList, op):
    """
    Compute Split operation (returns list of arrays).

    Args:
        iTList: [data] or [data, split]
        op: SimOp with attrs axis, split

    Returns:
        List[np.ndarray]: Split outputs
    """
    data = iTList[0].data
    axis = op.attrs.get("axis", 0)
    split = op.attrs.get("split", None)

    if split is None:
        # Equal split - get num_outputs from op
        num_outputs = len(op.outList)
        return np.array_split(data, num_outputs, axis=axis)
    else:
        # Unequal split
        split_indices = np.cumsum(split)[:-1]
        return np.split(data, split_indices, axis=axis)


def compute_softmax(iTList, op) -> np.ndarray:
    """
    Compute Softmax along specified axis.

    Args:
        iTList: [X]
        op: SimOp with attrs axis

    Returns:
        Y: Softmax output
    """
    # Check if this is shape inference only
    if iTList[0].data is None:
        # Shape inference: output has same shape as input
        return np.empty(iTList[0].shape)

    X = iTList[0].data
    axis = op.attrs.get("axis", -1)

    # Numerical stability: subtract max
    X_max = np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X - X_max)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def compute_sub(iTList, op) -> np.ndarray:
    """Element-wise subtraction with broadcasting"""
    return iTList[0].data - iTList[1].data


def compute_div(iTList, op) -> np.ndarray:
    """Element-wise division with broadcasting"""
    return iTList[0].data / iTList[1].data


def compute_sqrt(iTList, op) -> np.ndarray:
    """Element-wise square root"""
    return np.sqrt(iTList[0].data)


def compute_tanh(iTList, op) -> np.ndarray:
    """Tanh activation"""
    return np.tanh(iTList[0].data)


def compute_exp(iTList, op) -> np.ndarray:
    """Element-wise exponential"""
    return np.exp(iTList[0].data)


def compute_log(iTList, op) -> np.ndarray:
    """Element-wise natural logarithm"""
    return np.log(iTList[0].data)


def compute_pow(iTList, op) -> np.ndarray:
    """Element-wise power"""
    return np.power(iTList[0].data, iTList[1].data)


def compute_clip(iTList, op) -> np.ndarray:
    """
    Clip values to range [min, max].

    Args:
        iTList: [X, min, max] or [X]
        op: SimOp

    Returns:
        Y: Clipped output
    """
    X = iTList[0].data
    min_val = iTList[1].data if len(iTList) > 1 else -np.inf
    max_val = iTList[2].data if len(iTList) > 2 else np.inf
    return np.clip(X, min_val, max_val)


def compute_reducemean(iTList, op) -> np.ndarray:
    """
    Compute ReduceMean (average) along specified axes.

    Args:
        iTList: [X] or [X, axes] where axes is int64 array
        op: SimOp with attrs keepdims, noop_with_empty_axes

    Returns:
        Y: Reduced output
    """
    X = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None
    keepdims = op.attrs.get("keepdims", 1)
    noop = op.attrs.get("noop_with_empty_axes", 0)

    if axes is None:
        if noop:
            return X.copy()
        else:
            # Reduce over all axes
            axes = None
    else:
        # Convert to tuple for np.mean
        axes = tuple(int(a) for a in axes)

    return np.mean(X, axis=axes, keepdims=bool(keepdims))


# def compute_relu6(iTList, op) -> np.ndarray:
#     """ReLU6 activation: min(max(0, x), 6) = clip(x, 0, 6)"""
#     return np.clip(iTList[0].data, 0, 6)


def compute_upsample(iTList, op) -> np.ndarray:
    """
    Compute Upsample using nearest neighbor interpolation.

    Args:
        iTList: [X] where X is [N, C, H, W]
        op: SimOp with attrs mode, scale_factor, nearest_mode

    Returns:
        Y: Resized output [N, C, H_out, W_out]
    """
    X = iTList[0].data

    mode = op.attrs.get("mode", "nearest")
    scale_factor = op.attrs.get("scale_factor", 2)
    nearest_mode = op.attrs.get("nearest_mode", "floor")

    N, C, H_in, W_in = X.shape

    if isinstance(scale_factor, (list, tuple)):
        scale_h, scale_w = scale_factor[-2], scale_factor[-1]
    else:
        scale_h = scale_w = scale_factor

    H_out = int(H_in * scale_h)
    W_out = int(W_in * scale_w)

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)

    if mode == "nearest":
        for h in range(H_out):
            for w in range(W_out):
                if nearest_mode == "floor":
                    src_h = int(np.floor(h / scale_h))
                    src_w = int(np.floor(w / scale_w))
                elif nearest_mode == "ceil":
                    src_h = int(np.ceil(h / scale_h))
                    src_w = int(np.ceil(w / scale_w))
                else:
                    src_h = int(np.round(h / scale_h))
                    src_w = int(np.round(w / scale_w))

                src_h = min(max(0, src_h), H_in - 1)
                src_w = min(max(0, src_w), W_in - 1)

                Y[:, :, h, w] = X[:, :, src_h, src_w]

    return Y


def compute_l1_loss(iTList, op) -> np.ndarray:
    """
    Compute L1 Loss (Mean Absolute Error).

    Formula: mean(|predictions - targets|) or element-wise |predictions - targets|

    Args:
        iTList: [predictions, targets]
        op: SimOp with optional attr 'reduction' ('none', 'mean', 'sum')

    Returns:
        Output tensor with L1 loss
    """
    predictions = iTList[0].data
    targets = iTList[1].data
    reduction = op.attrs.get("reduction", "mean")

    # Compute element-wise absolute difference
    diff = np.abs(predictions - targets)

    # Apply reduction
    if reduction == "none":
        return diff
    elif reduction == "mean":
        return np.mean(diff)
    elif reduction == "sum":
        return np.sum(diff)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_binary_cross_entropy_with_logits(iTList, op) -> np.ndarray:
    """
    Compute Binary Cross Entropy with Logits.

    Formula: -[target * log(sigmoid(input)) + (1 - target) * log(1 - sigmoid(input))]
    This is numerically stable version that combines sigmoid + BCE.

    Args:
        iTList: [input, target]
            input: logits (unnormalized predictions)
            target: ground truth labels (0 or 1)
        op: SimOp with optional attr 'reduction' ('none', 'mean', 'sum')

    Returns:
        Output tensor with BCE loss
    """
    input_data = iTList[0].data
    target_data = iTList[1].data
    reduction = op.attrs.get("reduction", "mean")

    # Numerically stable computation
    # BCE = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    # Simplified: max(x, 0) - x * y + log(1 + exp(-|x|))
    max_val = np.maximum(input_data, 0)
    loss = max_val - input_data * target_data + np.log(1 + np.exp(-np.abs(input_data)))

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_groupnorm(iTList, op) -> np.ndarray:
    """
    Compute GroupNormalization.

    Normalizes input over groups of channels.
    Formula: y = (x - mean) / sqrt(var + eps) * scale + bias
    where mean and var are computed over spatial dims within each group.

    Args:
        iTList: [X, scale, bias] or [X, scale] where:
            X: input [N, C, ...] (any spatial dimensions)
            scale: [C] scale parameters
            bias: [C] bias parameters (optional)
        op: SimOp with attrs:
            - num_groups: int (number of groups)
            - epsilon: float (default 1e-5)

    Returns:
        Y: Normalized output [N, C, ...]
    """
    X = iTList[0].data
    scale = iTList[1].data
    bias = iTList[2].data if len(iTList) > 2 else None

    num_groups = op.attrs.get("num_groups", 1)
    epsilon = op.attrs.get("epsilon", 1e-5)

    N, C = X.shape[:2]
    spatial_shape = X.shape[2:]

    assert (
        C % num_groups == 0
    ), f"Channels {C} must be divisible by num_groups {num_groups}"

    # Reshape X to [N, num_groups, C // num_groups, ...spatial...]
    G = num_groups
    C_per_group = C // G
    X_grouped = X.reshape(N, G, C_per_group, *spatial_shape)

    # Compute mean and variance over (C_per_group, ...spatial) axes
    # axes to reduce: (2, 3, 4, ...) for (C_per_group, H, W, ...)
    reduce_axes = tuple(range(2, X_grouped.ndim))
    mean = X_grouped.mean(axis=reduce_axes, keepdims=True)
    var = X_grouped.var(axis=reduce_axes, keepdims=True)

    # Normalize
    X_norm = (X_grouped - mean) / np.sqrt(var + epsilon)

    # Reshape back to [N, C, ...spatial...]
    X_norm = X_norm.reshape(N, C, *spatial_shape)

    # Apply scale and bias (broadcast over spatial dimensions)
    scale_shape = [1, C] + [1] * len(spatial_shape)
    scale_broadcast = scale.reshape(scale_shape)
    Y = X_norm * scale_broadcast

    if bias is not None:
        bias_broadcast = bias.reshape(scale_shape)
        Y = Y + bias_broadcast

    return Y


def compute_einsum(iTList, op) -> np.ndarray:
    """
    Compute Einstein summation.

    Performs generalized tensor contraction using einsum notation.

    Args:
        iTList: List of input tensors
        op: SimOp with required attr 'subscripts' (einsum notation string)
            Example: "bqnc,bnchw->bqnhw"

    Returns:
        Output tensor from einsum operation
    """
    subscripts = op.attrs.get("subscripts", "")
    assert subscripts, "Einsum requires 'subscripts' attribute"

    # Extract data from tensors
    operands = [t.data for t in iTList]

    # Use numpy's einsum for computation
    result = np.einsum(subscripts, *operands)

    return result


def compute_cdist(iTList, op) -> np.ndarray:
    """
    Compute pairwise distance between two collections of row vectors.

    Args:
        iTList: [x1, x2]
            x1: [..., P, M] - First collection of row vectors
            x2: [..., R, M] - Second collection of row vectors
        op: SimOp with optional attr 'p' (norm order, default 2.0)
            p=1: Manhattan distance (L1)
            p=2: Euclidean distance (L2)

    Returns:
        output: [..., P, R] - Pairwise distances
        output[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||_p
    """
    x1_data = iTList[0].data
    x2_data = iTList[1].data
    p = op.attrs.get("p", 2.0)

    # Get shapes
    batch_shape = x1_data.shape[:-2]
    P = x1_data.shape[-2]
    R = x2_data.shape[-2]
    M = x1_data.shape[-1]

    # Reshape for broadcasting: [..., P, 1, M] and [..., 1, R, M]
    x1_expanded = x1_data[..., :, np.newaxis, :]  # [..., P, 1, M]
    x2_expanded = x2_data[..., np.newaxis, :, :]  # [..., 1, R, M]

    # Compute difference
    diff = x1_expanded - x2_expanded  # [..., P, R, M]

    # Compute distance based on p-norm
    if p == 1:
        # L1 distance: sum of absolute differences
        dist = np.sum(np.abs(diff), axis=-1)  # [..., P, R]
    elif p == 2:
        # L2 distance: sqrt of sum of squared differences
        dist = np.sqrt(np.sum(diff**2, axis=-1))  # [..., P, R]
    else:
        # General p-norm: (sum of |diff|^p)^(1/p)
        dist = np.sum(np.abs(diff) ** p, axis=-1) ** (1.0 / p)  # [..., P, R]

    return dist


def compute_diag(iTList, op) -> np.ndarray:
    """
    Compute diagonal extraction or construction.

    Args:
        iTList: [input_tensor]
        op: SimOp with optional attr 'diagonal' (offset, default 0)

    Returns:
        - If input is 1D: 2D matrix with input on diagonal
        - If input is 2D: 1D vector of diagonal elements
    """
    input_data = iTList[0].data
    diagonal = op.attrs.get("diagonal", 0)

    # Use numpy's diag function
    return np.diag(input_data, k=diagonal)


def compute_glu(iTList, op) -> np.ndarray:
    """
    Compute Gated Linear Unit (GLU).
    GLU(x) = x[:, :n] * sigmoid(x[:, n:]) where input is split in half.

    Args:
        iTList: [input_tensor]
            input_tensor: [..., 2*dim, ...] where split dimension must be even
        op: SimOp with optional attr 'dim' (default -1)

    Returns:
        Output tensor with half size along split dimension
    """
    input_data = iTList[0].data
    dim = op.attrs.get("dim", -1)

    # Normalize dimension
    if dim < 0:
        dim += input_data.ndim

    # Split input in half along specified dimension
    split_size = input_data.shape[dim] // 2

    # Create slicing tuples
    first_half_slice = [slice(None)] * input_data.ndim
    first_half_slice[dim] = slice(0, split_size)

    second_half_slice = [slice(None)] * input_data.ndim
    second_half_slice[dim] = slice(split_size, None)

    # Compute GLU: first_half * sigmoid(second_half)
    first_half = input_data[tuple(first_half_slice)]
    second_half = input_data[tuple(second_half_slice)]

    # Sigmoid activation
    sigmoid_second = 1.0 / (1.0 + np.exp(-second_half))

    # Element-wise multiplication
    output = first_half * sigmoid_second

    return output


def compute_inverse_sigmoid(iTList, op) -> np.ndarray:
    """
    Compute inverse sigmoid (logit function).

    Formula: log(x / (1 - x))

    Args:
        iTList: [input_tensor]
        op: SimOp

    Returns:
        Output tensor with inverse sigmoid applied

    Note:
        - Clamps input to (eps, 1-eps) for numerical stability
        - eps default is 1e-5
    """
    x = iTList[0].data
    eps = op.attrs.get("eps", 1e-5)

    # Clamp to avoid log(0) or log(inf)
    x_clamped = np.clip(x, eps, 1 - eps)

    # Compute inverse sigmoid: log(x / (1 - x))
    return np.log(x_clamped / (1 - x_clamped))


def compute_tile(iTList, op) -> np.ndarray:
    """
    Compute Tile operation (repeat array along axes).

    Args:
        iTList: [data, repeats] where repeats is int64 array
        op: SimOp

    Returns:
        Y: Tiled output
    """
    data = iTList[0].data
    repeats = iTList[1].data.astype(np.int64)
    return np.tile(data, repeats)


def compute_meshgrid(iTList, op) -> np.ndarray:
    """
    Create coordinate grid for Detect module.
    Used in YOLOv4 Detect for anchor decoding.

    Args:
        iTList: [ny, nx] coordinate ranges or empty (uses attrs)
        op: SimOp with attrs ny, nx

    Returns:
        Grid array [1, 1, ny, nx, 2] with [x, y] coordinates
    """
    if len(iTList) >= 2:
        ny = int(iTList[0].data)
        nx = int(iTList[1].data)
    else:
        ny = op.attrs.get("ny", 20)
        nx = op.attrs.get("nx", 20)

    # Create coordinate arrays
    y_coords = np.arange(ny, dtype=np.float32)
    x_coords = np.arange(nx, dtype=np.float32)

    # Create meshgrid using 'ij' indexing (matrix indexing)
    # torch.meshgrid([arange(ny), arange(nx)], indexing='ij')
    yv, xv = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Stack as [xv, yv] along last axis
    # torch.stack((xv, yv), 2) creates [..., 2] dimension with [x, y]
    grid = np.stack([xv, yv], axis=2)

    # Reshape to (1, 1, ny, nx, 2)
    grid = grid.reshape(1, 1, ny, nx, 2)

    return grid


# used in polaris\workloads\Deformable_DETR\models\ops\functions\ms_deform_attn_func_ttsim.py
def compute_grid_sample(iTList, op) -> np.ndarray:
    """
    Compute grid_sample operation for bilinear interpolation.
    Vectorized implementation that exactly matches PyTorch's F.grid_sample
    with mode='bilinear', padding_mode='zeros', align_corners=False.

    Args:
        iTList: [input, grid] where:
            input: [N, C, H_in, W_in] - input feature map
            grid: [N, H_out, W_out, 2] - sampling grid with (x, y) coordinates in [-1, 1]
        op: SimOp with attrs mode, padding_mode, align_corners

    Returns:
        output: [N, C, H_out, W_out] - sampled values
    """
    input_tensor = iTList[0].data  # [N, C, H_in, W_in]
    grid = iTList[1].data  # [N, H_out, W_out, 2]

    mode = op.attrs.get("mode", "bilinear")
    padding_mode = op.attrs.get("padding_mode", "zeros")
    align_corners = op.attrs.get("align_corners", False)

    N, C, H_in, W_in = input_tensor.shape
    _, H_out, W_out, _ = grid.shape

    # Extract x and y coordinates from grid
    grid_x = grid[..., 0]  # [N, H_out, W_out]
    grid_y = grid[..., 1]  # [N, H_out, W_out]

    # Convert normalized grid [-1, 1] to pixel coordinates
    # PyTorch formula for align_corners=False:
    #   pixel_coord = ((grid + 1) * size - 1) / 2
    # This maps: -1 -> -0.5, 0 -> (size-1)/2, 1 -> size-0.5
    if align_corners:
        # Map [-1, 1] to [0, size-1]
        x = ((grid_x + 1) / 2) * (W_in - 1)
        y = ((grid_y + 1) / 2) * (H_in - 1)
    else:
        # Map [-1, 1] to [-0.5, size-0.5] - matches PyTorch exactly
        x = ((grid_x + 1) * W_in - 1) / 2
        y = ((grid_y + 1) * H_in - 1) / 2

    if mode == "bilinear":
        # Compute floor coordinates for bilinear interpolation
        x0 = np.floor(x).astype(np.int64)  # [N, H_out, W_out]
        y0 = np.floor(y).astype(np.int64)  # [N, H_out, W_out]
        x1 = x0 + 1
        y1 = y0 + 1

        # Compute interpolation weights
        wx1 = x - x0.astype(np.float64)  # weight for x1
        wx0 = 1.0 - wx1  # weight for x0
        wy1 = y - y0.astype(np.float64)  # weight for y1
        wy0 = 1.0 - wy1  # weight for y0

        # Create validity masks for each corner (padding_mode='zeros')
        # Points outside [0, size-1] should contribute 0
        valid_x0 = (x0 >= 0) & (x0 < W_in)  # [N, H_out, W_out]
        valid_x1 = (x1 >= 0) & (x1 < W_in)
        valid_y0 = (y0 >= 0) & (y0 < H_in)
        valid_y1 = (y1 >= 0) & (y1 < H_in)

        valid_00 = valid_y0 & valid_x0  # top-left
        valid_01 = valid_y0 & valid_x1  # top-right
        valid_10 = valid_y1 & valid_x0  # bottom-left
        valid_11 = valid_y1 & valid_x1  # bottom-right

        # Clamp coordinates for safe indexing (will be zeroed by validity mask)
        x0_safe = np.clip(x0, 0, W_in - 1)
        x1_safe = np.clip(x1, 0, W_in - 1)
        y0_safe = np.clip(y0, 0, H_in - 1)
        y1_safe = np.clip(y1, 0, H_in - 1)

        # Create batch indices for advanced indexing
        batch_idx = np.arange(N)[:, None, None]  # [N, 1, 1]
        batch_idx = np.broadcast_to(batch_idx, (N, H_out, W_out))  # [N, H_out, W_out]

        # Initialize output
        output = np.zeros((N, C, H_out, W_out), dtype=input_tensor.dtype)

        # Gather values from input tensor for each corner
        # input_tensor: [N, C, H_in, W_in]
        # We need to index [batch_idx, :, y_safe, x_safe] for all channels
        for c in range(C):
            # Get values at 4 corners using advanced indexing
            v00 = input_tensor[batch_idx, c, y0_safe, x0_safe]  # [N, H_out, W_out]
            v01 = input_tensor[batch_idx, c, y0_safe, x1_safe]
            v10 = input_tensor[batch_idx, c, y1_safe, x0_safe]
            v11 = input_tensor[batch_idx, c, y1_safe, x1_safe]

            # Apply validity masks (zeros for out-of-bounds)
            v00 = np.where(valid_00, v00, 0.0)
            v01 = np.where(valid_01, v01, 0.0)
            v10 = np.where(valid_10, v10, 0.0)
            v11 = np.where(valid_11, v11, 0.0)

            # Bilinear interpolation
            output[:, c, :, :] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (
                wx0 * v10 + wx1 * v11
            )

        return output

    elif mode == "nearest":
        # Round to nearest integer coordinates
        x_near = np.round(x).astype(np.int64)
        y_near = np.round(y).astype(np.int64)

        # Validity mask
        valid = (x_near >= 0) & (x_near < W_in) & (y_near >= 0) & (y_near < H_in)

        # Clamp for safe indexing
        x_safe = np.clip(x_near, 0, W_in - 1)
        y_safe = np.clip(y_near, 0, H_in - 1)

        # Create batch indices
        batch_idx = np.arange(N)[:, None, None]
        batch_idx = np.broadcast_to(batch_idx, (N, H_out, W_out))

        # Initialize output
        output = np.zeros((N, C, H_out, W_out), dtype=input_tensor.dtype)

        for c in range(C):
            vals = input_tensor[batch_idx, c, y_safe, x_safe]
            output[:, c, :, :] = np.where(valid, vals, 0.0)

        return output

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def compute_view_reshape(iTList, op) -> np.ndarray | None:
    """
    View/reshape operation (no data copy, just shape change).

    Args:
        iTList: [data, shape_tensor] where shape_tensor contains the target shape
        op: SimOp with optional attrs new_shape

    Returns:
        reshaped: tensor with new shape
    """
    # Get shape from either attributes or shape tensor
    new_shape = op.attrs.get("new_shape", None)

    if new_shape is None and len(iTList) > 1:
        # Extract from shape tensor (standard reshape operation)
        shape_tensor = iTList[1]
        if shape_tensor.data is not None:
            new_shape = [int(x) for x in shape_tensor.data]

    if new_shape is None:
        raise ValueError(
            "new_shape attribute or shape tensor required for view_reshape"
        )

    # Check if this is shape inference only
    if iTList[0].data is None:
        return None

    data = iTList[0].data
    # Use np.ascontiguousarray to ensure contiguous memory layout
    # This fixes issues when data comes from views/slices
    return np.ascontiguousarray(np.reshape(data, new_shape))


def compute_stack(iTList, op) -> np.ndarray:
    """
    Stack tensors along a new dimension.

    Args:
        iTList: List of tensors to stack
        op: SimOp with attrs axis

    Returns:
        stacked: stacked tensor
    """
    axis = op.attrs.get("axis", 0)
    arrays = [t.data for t in iTList]
    return np.stack(arrays, axis=axis)


def compute_relu6(iTList, op) -> np.ndarray:
    """ReLU6 activation: min(max(0, x), 6) = clip(x, 0, 6)"""
    return np.clip(iTList[0].data, 0, 6)


def compute_reshape(iTList, op) -> np.ndarray:
    """
    Compute Reshape operation.

    Args:
        iTList: [data, shape]
        op: SimOp

    Returns:
        Y: Reshaped output
    """
    data = iTList[0].data
    new_shape = iTList[1].data.astype(np.int64)
    return np.reshape(data, new_shape)


def compute_bbox_center_decode(iTList, op) -> np.ndarray:
    """
    Decode bounding box center coordinates using grid-based offset and stride.
    Commonly used in anchor-based object detection (YOLO, SSD, etc.).
    Formula: (sigmoid(xy) * 2.0 - 0.5 + grid) * stride

    Args:
        iTList: [xy_sigmoid, grid, stride] where:
            xy_sigmoid: [bs, na, ny, nx, 2] - sigmoid activated xy predictions
            grid: [1, 1, ny, nx, 2] or [bs, na, ny, nx, 2] - coordinate grid
            stride: scalar - detection layer stride
        op: SimOp

    Returns:
        xy_decoded: [bs, na, ny, nx, 2] - decoded xy coordinates in image space
    """
    xy_sigmoid = iTList[0].data  # [bs, na, ny, nx, 2]
    grid = iTList[1].data  # [1, 1, ny, nx, 2] or [bs, na, ny, nx, 2]
    stride = iTList[2].data  # scalar

    # Formula: (xy * 2.0 - 0.5 + grid) * stride
    xy_decoded = (xy_sigmoid * 2.0 - 0.5 + grid) * stride

    return xy_decoded


def compute_bbox_size_decode(iTList, op) -> np.ndarray:
    """
    Decode bounding box width and height using anchor-based scaling.
    Commonly used in anchor-based object detection (YOLO, SSD, etc.).
    Formula: ((sigmoid(wh) * 2.0) ** 2) * anchor_grid

    Args:
        iTList: [wh_sigmoid, anchor_grid] where:
            wh_sigmoid: [bs, na, ny, nx, 2] - sigmoid activated wh predictions
            anchor_grid: [1, na, 1, 1, 2] - anchor dimensions for this layer
        op: SimOp

    Returns:
        wh_decoded: [bs, na, ny, nx, 2] - decoded wh dimensions in image space
    """
    wh_sigmoid = iTList[0].data  # [bs, na, ny, nx, 2]
    anchor_grid = iTList[1].data  # [1, na, 1, 1, 2]

    # Formula: ((wh * 2.0) ** 2) * anchor_grid
    wh_decoded = ((wh_sigmoid * 2.0) ** 2) * anchor_grid

    return wh_decoded
