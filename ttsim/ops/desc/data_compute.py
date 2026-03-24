#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Data computation helpers for shape inference functions"""

import numpy as np
from typing import Any, List, Optional

##############################################################################
# MS DEFORMABLE ATTENTION ADDITIONS for data_compute.py
##############################################################################
# Add these 2 functions to ttsim/ops/desc/data_compute.py
# These are generic compute helpers that can be reused for other models.
##############################################################################

import numpy as np
from typing import Any, List, Optional


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
    Compute Resize (Upsample/Downsample) using nearest or bilinear interpolation.

    Args:
        iTList: [X, roi, scales] where X is [N, C, H, W], scales holds per-axis factors
        op: SimOp with attrs mode, nearest_mode, align_corners, coordinate_transformation_mode

    Returns:
        Y: Resized output [N, C, H_out, W_out]
    """
    X = iTList[0].data

    mode = op.attrs.get("mode", "nearest")
    nearest_mode = op.attrs.get("nearest_mode", "floor")

    # Normalize mode: ONNX uses 'linear' for bilinear interpolation
    if mode == "linear":
        mode = "bilinear"

    # Read scale factors from the scales param tensor (iTList[2])
    scale_factor: Any
    if len(iTList) >= 3 and iTList[2].data is not None:
        scales_data = iTList[2].data
        scale_h = float(scales_data[-2])
        scale_w = float(scales_data[-1])
        scale_factor = [scale_h, scale_w]
    else:
        scale_factor = op.attrs.get("scale_factor", 2)

    N, C, H_in, W_in = X.shape

    if isinstance(scale_factor, (list, tuple)):
        scale_h, scale_w = scale_factor[-2], scale_factor[-1]
    elif isinstance(scale_factor, (int, float)):
        scale_h = scale_w = scale_factor
    else:
        scale_h = scale_w = float(scale_factor)

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

    elif mode == "bilinear":
        # Read align_corners from ONNX-compatible coordinate_transformation_mode
        ctm = op.attrs.get("coordinate_transformation_mode", None)
        if ctm is not None:
            align_corners = (ctm == "align_corners")
        else:
            align_corners = op.attrs.get("align_corners", False)

        if align_corners:
            if H_out > 1:
                row_scale = (H_in - 1) / (H_out - 1)
            else:
                row_scale = 0.0
            if W_out > 1:
                col_scale = (W_in - 1) / (W_out - 1)
            else:
                col_scale = 0.0
        else:
            row_scale = 1.0 / scale_h
            col_scale = 1.0 / scale_w

        for h in range(H_out):
            if align_corners:
                src_h = h * row_scale
            else:
                src_h = (h + 0.5) * row_scale - 0.5

            src_h = max(0.0, min(src_h, H_in - 1))
            h0 = int(np.floor(src_h))
            h1 = min(h0 + 1, H_in - 1)
            fh = src_h - h0

            for w in range(W_out):
                if align_corners:
                    src_w = w * col_scale
                else:
                    src_w = (w + 0.5) * col_scale - 0.5

                src_w = max(0.0, min(src_w, W_in - 1))
                w0 = int(np.floor(src_w))
                w1 = min(w0 + 1, W_in - 1)
                fw = src_w - w0

                Y[:, :, h, w] = (
                    X[:, :, h0, w0] * (1 - fh) * (1 - fw)
                    + X[:, :, h0, w1] * (1 - fh) * fw
                    + X[:, :, h1, w0] * fh * (1 - fw)
                    + X[:, :, h1, w1] * fh * fw
                )

    else:
        raise ValueError(f"compute_resize: unsupported mode '{mode}'")

    return Y


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


def _numpy_bilinear_interpolate(X, H_out, W_out, align_corners):
    """Pure numpy bilinear interpolation matching PyTorch F.interpolate.

    Args:
        X: (N, C, H_in, W_in) float array
        H_out, W_out: target spatial dimensions
        align_corners: bool — coordinate mapping convention

    Returns:
        Y: (N, C, H_out, W_out) float array
    """
    N, C, H_in, W_in = X.shape

    # Build destination coordinate grids
    dst_y = np.arange(H_out, dtype=np.float64)
    dst_x = np.arange(W_out, dtype=np.float64)

    # Map destination coords → source coords (PyTorch conventions)
    if align_corners:
        if H_out > 1:
            src_y = dst_y * (H_in - 1) / (H_out - 1)
        else:
            src_y = np.zeros_like(dst_y)
        if W_out > 1:
            src_x = dst_x * (W_in - 1) / (W_out - 1)
        else:
            src_x = np.zeros_like(dst_x)
    else:
        src_y = (dst_y + 0.5) * H_in / H_out - 0.5
        src_x = (dst_x + 0.5) * W_in / W_out - 0.5

    # Clamp to valid range
    src_y = np.clip(src_y, 0, H_in - 1)
    src_x = np.clip(src_x, 0, W_in - 1)

    # Integer and fractional parts
    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, H_in - 1)
    x1 = np.minimum(x0 + 1, W_in - 1)

    fy = (src_y - y0).astype(X.dtype)  # (H_out,)
    fx = (src_x - x0).astype(X.dtype)  # (W_out,)

    # Broadcast: fy → (1, 1, H_out, 1), fx → (1, 1, 1, W_out)
    fy = fy[None, None, :, None]
    fx = fx[None, None, None, :]

    # Gather the four corner values  (N, C, H_out, W_out)
    Ia = X[:, :, y0, :][:, :, :, x0]
    Ib = X[:, :, y1, :][:, :, :, x0]
    Ic = X[:, :, y0, :][:, :, :, x1]
    Id = X[:, :, y1, :][:, :, :, x1]

    # Bilinear blend
    Y = (Ia * (1 - fy) * (1 - fx) +
         Ib * fy * (1 - fx) +
         Ic * (1 - fy) * fx +
         Id * fy * fx)

    return Y


def compute_upsample(iTList, op) -> np.ndarray:
    """
    Upsample/Downsample computation.

    Supports 'nearest' and 'linear' (bilinear 2D) modes with
    align_corners control.  Works for any scale factor (integer,
    fractional, >1 or <1) and for 2D or 4D inputs.

    Scale factor resolution order:
        1. iTList[1].data  (scales tensor from graph wiring)
        2. op.attrs['scales']
        3. op.attrs['scale_factor']  (scalar, list, or tuple)
        4. default 2.0

    Args:
        iTList: [X] or [X, scales] where X is input (N,C,H,W) or (H,W)
        op: SimOp with attrs mode, align_corners, scale_factor / scales

    Returns:
        Y: Resampled output
    """
    X = iTList[0].data
    mode = op.attrs.get('mode', 'nearest')
    align_corners = op.attrs.get('align_corners', True)

    # --- resolve scale factors -----------------------------------------
    if len(iTList) > 1 and iTList[1].data is not None:
        scales = iTList[1].data
        sf_h, sf_w = float(scales[-2]), float(scales[-1])
    else:
        scales = op.attrs.get('scales', None)
        if scales is not None:
            sf_h, sf_w = float(scales[-2]), float(scales[-1])
        else:
            sf = op.attrs.get('scale_factor', 2.0)
            if isinstance(sf, (list, tuple)):
                sf_h, sf_w = float(sf[-2]), float(sf[-1])
            else:
                sf_h = sf_w = float(sf)

    # --- compute output size -------------------------------------------
    if len(X.shape) == 4:
        N, C, H, W = X.shape
    elif len(X.shape) == 2:
        H, W = X.shape
    else:
        raise NotImplementedError(
            f"Unsupported input shape for upsample: {X.shape}")

    H_out = int(H * sf_h)
    W_out = int(W * sf_w)

    # --- interpolation -------------------------------------------------
    if mode in ('linear', 'bilinear'):
        if len(X.shape) != 4:
            raise NotImplementedError(
                "Bilinear upsample only supports 4D (N,C,H,W) input")
        Y = _numpy_bilinear_interpolate(X, H_out, W_out, align_corners)

    elif mode == 'nearest':
        # Vectorised coordinate mapping (works for any scale)
        src_y = np.floor(np.arange(H_out) * H / H_out).astype(np.int64)
        src_x = np.floor(np.arange(W_out) * W / W_out).astype(np.int64)
        src_y = np.clip(src_y, 0, H - 1)
        src_x = np.clip(src_x, 0, W - 1)

        if len(X.shape) == 4:
            Y = X[:, :, src_y, :][:, :, :, src_x]
        else:  # 2D
            Y = X[src_y, :][:, src_x]
    else:
        raise ValueError(f"Unsupported upsample mode: {mode}")

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

        if padding_mode == "border":
            # Border mode: clamp coordinates and use all values (no zeroing)
            x0_safe = np.clip(x0, 0, W_in - 1)
            x1_safe = np.clip(x1, 0, W_in - 1)
            y0_safe = np.clip(y0, 0, H_in - 1)
            y1_safe = np.clip(y1, 0, H_in - 1)

            batch_idx = np.arange(N)[:, None, None]
            batch_idx = np.broadcast_to(batch_idx, (N, H_out, W_out))

            output = np.zeros((N, C, H_out, W_out), dtype=input_tensor.dtype)

            for c in range(C):
                v00 = input_tensor[batch_idx, c, y0_safe, x0_safe]
                v01 = input_tensor[batch_idx, c, y0_safe, x1_safe]
                v10 = input_tensor[batch_idx, c, y1_safe, x0_safe]
                v11 = input_tensor[batch_idx, c, y1_safe, x1_safe]

                output[:, c, :, :] = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (
                    wx0 * v10 + wx1 * v11
                )

            return output

        # Default: padding_mode='zeros'
        # Create validity masks for each corner
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


def compute_unsqueeze(iTList, op) -> np.ndarray:
    """
    Compute Unsqueeze operation (add dimensions of size 1).

    Args:
        iTList: [data, axes] where axes specifies dimensions to add
        op: SimOp

    Returns:
        Y: Output with added dimensions
    """
    data = iTList[0].data
    axes = iTList[1].data  # Array of axes to unsqueeze

    # Build new shape by inserting 1s at specified axes
    new_shape = list(data.shape)
    for axis in sorted(axes):
        # Handle negative indices
        if axis < 0:
            axis = len(new_shape) + axis + 1
        new_shape.insert(axis, 1)

    return np.reshape(data, new_shape)


def compute_sin(iTList, op) -> np.ndarray:
    """Element-wise sine"""
    return np.sin(iTList[0].data)


def compute_cos(iTList, op) -> np.ndarray:
    """Element-wise cosine"""
    return np.cos(iTList[0].data)


def compute_abs(iTList, op) -> np.ndarray:
    """Element-wise absolute value"""
    return np.abs(iTList[0].data)


def compute_neg(iTList, op) -> np.ndarray:
    """Element-wise negation"""
    return -iTList[0].data


def compute_less(iTList, op) -> np.ndarray:
    """Element-wise less than comparison"""
    return iTList[0].data < iTList[1].data


def compute_where(iTList, op) -> np.ndarray:
    """
    Select elements from X or Y based on condition.

    Args:
        iTList: [condition, X, Y]
        op: SimOp

    Returns:
        output: X where condition is True, Y otherwise
    """
    condition = iTList[0].data
    X = iTList[1].data
    Y = iTList[2].data
    return np.where(condition, X, Y)


def compute_scatter_nd(iTList, op) -> np.ndarray:
    """
    Scatter updates into a copy of data at positions specified by indices.

    Follows ONNX ScatterND semantics (reduction='none'):
      output = copy(data)
      output[indices] = updates

    Args:
        iTList: [data, indices, updates]
            data:    (any shape) base tensor
            indices: (*idx_shape, K) last dim indexes into first K dims of data
            updates: (*idx_shape, *data.shape[K:])  values to write
        op: SimOp (attrs 'reduction' optional, default 'none')

    Returns:
        output: same shape as data, with scattered updates
    """
    data    = iTList[0].data.copy()
    indices = iTList[1].data.astype(np.int64)
    updates = iTList[2].data
    reduction = op.attrs.get('reduction', 'none')

    # indices shape: (*idx_shape, K)
    K = indices.shape[-1]
    idx_shape = indices.shape[:-1]

    # Flatten the index batch dimensions so we can iterate
    flat_indices = indices.reshape(-1, K)
    flat_updates = updates.reshape(-1, *data.shape[K:])

    for i in range(flat_indices.shape[0]):
        idx = tuple(flat_indices[i])
        if reduction == 'none':
            data[idx] = flat_updates[i]
        elif reduction == 'add':
            data[idx] += flat_updates[i]
        elif reduction == 'mul':
            data[idx] *= flat_updates[i]
        elif reduction == 'max':
            data[idx] = np.maximum(data[idx], flat_updates[i])
        elif reduction == 'min':
            data[idx] = np.minimum(data[idx], flat_updates[i])
        else:
            data[idx] = flat_updates[i]

    return data


def compute_gather(iTList, op) -> np.ndarray:
    """
    Gather elements along an axis based on indices.

    Args:
        iTList: [data, indices]
        op: SimOp with attrs axis

    Returns:
        output: Gathered elements
    """
    data = iTList[0].data
    indices = iTList[1].data.astype(np.int64)
    axis = op.attrs.get('axis', 0)

    # Expand indices to match data's number of dimensions for np.take_along_axis
    # ONNX Gather: output_shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
    # We need to expand indices shape to match this
    data_rank = len(data.shape)
    indices_rank = len(indices.shape)

    if indices_rank < data_rank:
        # Need to expand indices to have same rank as data
        # Insert new axes before and after the indices dimensions
        new_shape = [1] * axis + list(indices.shape) + [1] * (data_rank - axis - indices_rank)
        indices = indices.reshape(new_shape)

    return np.take_along_axis(data, indices, axis=axis)


def compute_squeeze(iTList, op) -> np.ndarray:
    """
    Remove dimensions of size 1.

    Args:
        iTList: [data] or [data, axes]
        op: SimOp

    Returns:
        output: Squeezed array
    """
    data = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None
    if axes is not None:
        axes = tuple(int(a) for a in axes)
        return np.squeeze(data, axis=axes)
    return np.squeeze(data)


def compute_reducesum(iTList, op) -> np.ndarray:
    """
    Compute ReduceSum along specified axes.

    Args:
        iTList: [X] or [X, axes] where axes is int64 array
        op: SimOp with attrs keepdims, noop_with_empty_axes

    Returns:
        Y: Reduced output
    """
    X = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None
    keepdims = op.attrs.get('keepdims', 1)
    noop = op.attrs.get('noop_with_empty_axes', 0)

    if axes is None:
        if noop:
            return X.copy()
        else:
            # Reduce over all axes
            axes = None
    else:
        # Convert to tuple for np.sum
        axes = tuple(int(a) for a in axes)

    return np.sum(X, axis=axes, keepdims=bool(keepdims))


def compute_reducemax(iTList, op) -> np.ndarray:
    """
    Compute ReduceMax along specified axes.

    Args:
        iTList: [X] or [X, axes] where axes is int64 array
        op: SimOp with attrs keepdims, noop_with_empty_axes

    Returns:
        Y: Reduced output with max values
    """
    X = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None
    keepdims = op.attrs.get('keepdims', 1)
    noop = op.attrs.get('noop_with_empty_axes', 0)

    if axes is None:
        if noop:
            return X.copy()
        else:
            # Reduce over all axes
            axes = None
    else:
        # Convert to tuple for np.max
        axes = tuple(int(a) for a in axes)

    return np.max(X, axis=axes, keepdims=bool(keepdims))


def compute_argmax(iTList, op) -> np.ndarray:
    """
    Compute ArgMax - indices of maximum values along an axis.

    Args:
        iTList: [X] where X is input tensor
        op: SimOp with attrs axis, keepdims, select_last_index

    Returns:
        Y: Indices of maximum values (int64)
    """
    X = iTList[0].data
    axis = op.attrs.get('axis', -1)
    keepdims = op.attrs.get('keepdims', 1)
    select_last_index = op.attrs.get('select_last_index', 0)

    if select_last_index:
        # Find last occurrence of max value
        # Reverse along axis, find first max, then reverse index
        X_reversed = np.flip(X, axis=axis)
        indices_reversed = np.argmax(X_reversed, axis=axis, keepdims=bool(keepdims))
        # Convert back to original indices
        indices = X.shape[axis] - 1 - indices_reversed
    else:
        # Find first occurrence (default numpy behavior)
        indices = np.argmax(X, axis=axis, keepdims=bool(keepdims))

    # Ensure result is always an ndarray (even for 0-d scalars)
    result = indices.astype(np.int64)
    return np.asarray(result)

def compute_argmin(iTList, op) -> np.ndarray:
    """
    Compute ArgMin - indices of minimum values along an axis.

    Args:
        iTList: [X] where X is input tensor
        op: SimOp with attrs axis, keepdims, select_last_index

    Returns:
        Y: Indices of minimum values (int64)
    """
    X = iTList[0].data
    axis = op.attrs.get('axis', -1)
    keepdims = op.attrs.get('keepdims', 1)
    select_last_index = op.attrs.get('select_last_index', 0)

    if select_last_index:
        X_reversed = np.flip(X, axis=axis)
        indices_reversed = np.argmin(X_reversed, axis=axis, keepdims=bool(keepdims))
        indices = X.shape[axis] - 1 - indices_reversed
    else:
        indices = np.argmin(X, axis=axis, keepdims=bool(keepdims))

    result = indices.astype(np.int64)
    return np.asarray(result)

def compute_conv_transpose2d(iTList, op) -> np.ndarray:
    """
    Compute ConvTranspose2d output using pure NumPy.

    Args:
        iTList: [X, W] or [X, W, B] where:
            X: input  [N, C_in, H_in, W_in]
            W: weights [C_in, C_out/groups, kH, kW]   (PyTorch ConvTranspose2d layout)
            B: optional bias [C_out]
        op: SimOp with attrs strides, padding, output_padding, dilation, groups

    Returns:
        Y: ConvTranspose output [N, C_out, H_out, W_out]
    """
    X = iTList[0].data
    W = iTList[1].data
    B = iTList[2].data if len(iTList) > 2 else None

    stride = tuple(op.attrs.get("strides", (1, 1)))
    padding = tuple(op.attrs.get("padding", (0, 0)))
    output_padding = tuple(op.attrs.get("output_padding", (0, 0)))
    dilation = tuple(op.attrs.get("dilation", (1, 1)))
    groups = op.attrs.get("groups", 1)

    N, C_in, H_in, W_in = X.shape
    C_in_w, C_out_per_group, kH, kW = W.shape
    C_out = C_out_per_group * groups

    # Full output size (before padding crop)
    full_H = (H_in - 1) * stride[0] + dilation[0] * (kH - 1) + 1
    full_W = (W_in - 1) * stride[1] + dilation[1] * (kW - 1) + 1

    full_output = np.zeros((N, C_out, full_H, full_W), dtype=np.float64)

    if groups == 1:
        for n in range(N):
            for c_in in range(C_in):
                for h_in in range(H_in):
                    for w_in in range(W_in):
                        val = X[n, c_in, h_in, w_in]
                        for c_out in range(C_out):
                            for ki in range(kH):
                                for kj in range(kW):
                                    fh = h_in * stride[0] + ki * dilation[0]
                                    fw = w_in * stride[1] + kj * dilation[1]
                                    full_output[n, c_out, fh, fw] += (
                                        val * W[c_in, c_out, ki, kj]
                                    )
    else:
        C_in_per_group = C_in // groups
        for g in range(groups):
            c_in_start = g * C_in_per_group
            c_out_start = g * C_out_per_group
            for n in range(N):
                for c_in_local in range(C_in_per_group):
                    c_in = c_in_start + c_in_local
                    for h_in in range(H_in):
                        for w_in in range(W_in):
                            val = X[n, c_in, h_in, w_in]
                            for c_out_local in range(C_out_per_group):
                                c_out = c_out_start + c_out_local
                                for ki in range(kH):
                                    for kj in range(kW):
                                        fh = h_in * stride[0] + ki * dilation[0]
                                        fw = w_in * stride[1] + kj * dilation[1]
                                        full_output[n, c_out, fh, fw] += (
                                            val * W[c_in_local, c_out_local, ki, kj]
                                        )

    # Crop: top/left by padding, bottom/right by (padding - output_padding)
    crop_top = padding[0]
    crop_left = padding[1]
    h_end = full_H - (padding[0] - output_padding[0])
    w_end = full_W - (padding[1] - output_padding[1])

    Y = full_output[:, :, crop_top:h_end, crop_left:w_end].copy()

    if B is not None:
        Y += B.reshape(1, -1, 1, 1)

    return Y.astype(X.dtype)


def compute_elementwise_max(iTList, op) -> np.ndarray:
    """Element-wise maximum with broadcasting"""
    result = iTList[0].data
    for t in iTList[1:]:
        result = np.maximum(result, t.data)
    return result


def compute_elementwise_min(iTList, op) -> np.ndarray:
    """Element-wise minimum with broadcasting"""
    result = iTList[0].data
    for t in iTList[1:]:
        result = np.minimum(result, t.data)
    return result

def compute_elementwise_sum(iTList, op) -> np.ndarray:
    """Element-wise sum with broadcasting"""
    result = iTList[0].data.copy()
    for t in iTList[1:]:
        result = result + t.data
    return result

def compute_elementwise_mean(iTList, op) -> np.ndarray:
    """Element-wise mean with broadcasting"""
    result = iTList[0].data.copy()
    for t in iTList[1:]:
        result = result + t.data
    return result / len(iTList)

def compute_atan(iTList, op) -> np.ndarray:
    """Element-wise arctangent"""
    return np.arctan(iTList[0].data)


def compute_pad(iTList, op) -> np.ndarray:
    """Pad last 2 dims of input tensor.

    iTList[0] = data (N, C, H, W)
    iTList[1] = pads (4,)  [H_begin, W_begin, H_end, W_end]
    """
    X = iTList[0].data
    pads = [int(x) for x in iTList[1].data.tolist()]
    mode = op.attrs.get('mode', 'constant')
    value = op.attrs.get('value', 0)

    rank = X.ndim
    pad_before = [0] * (rank - 2) + pads[:2]
    pad_after = [0] * (rank - 2) + pads[2:]
    pad_widths = [(pb, pa) for pb, pa in zip(pad_before, pad_after)]

    if mode == 'constant':
        return np.pad(X, pad_widths, mode='constant', constant_values=value)
    elif mode == 'reflect':
        return np.pad(X, pad_widths, mode='reflect')
    elif mode == 'edge':
        return np.pad(X, pad_widths, mode='edge')
    else:
        return np.pad(X, pad_widths, mode='constant', constant_values=value)


def compute_gelu(iTList, op) -> np.ndarray:
    """GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    x = iTList[0].data
    try:
        from scipy.special import erf  # type: ignore[import-untyped]
    except ImportError:
        from math import erf as _erf
        erf = np.vectorize(_erf)  # type: ignore[assignment]
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(x.dtype)


def compute_atan2(iTList, op) -> np.ndarray:
    """
    Element-wise arctangent of y/x with correct quadrant handling.

    Args:
        iTList: [y, x] where y and x are tensors
        op: SimOp

    Returns:
        Array of angles in radians, in range [-pi, pi]
    """
    y = iTList[0].data
    x = iTList[1].data
    return np.arctan2(y, x)


def compute_cumsum(iTList, op) -> np.ndarray:
    X = iTList[0].data
    axis_data = iTList[1].data
    axis = int(axis_data.item()) if axis_data.size == 1 else int(axis_data[0])
    exclusive = op.attrs.get("exclusive", 0)
    reverse = op.attrs.get("reverse", 0)
    if reverse:
        X_work = np.flip(X, axis=axis)
    else:
        X_work = X
    if exclusive:
        result = np.cumsum(X_work, axis=axis)
        result = np.roll(result, 1, axis=axis)
        slc: list = [slice(None)] * len(X.shape)
        slc[axis] = 0  # type: ignore[list-item]
        result[tuple(slc)] = 0
    else:
        result = np.cumsum(X_work, axis=axis)
    if reverse:
        result = np.flip(result, axis=axis)
    return result


def compute_floor(iTList, op) -> np.ndarray:
    return np.floor(iTList[0].data)


def compute_mod(iTList, op) -> np.ndarray:
    return np.fmod(iTList[0].data, iTList[1].data)


def compute_reducemin(iTList, op) -> np.ndarray:
    X = iTList[0].data
    keepdims_bool = bool(op.attrs.get("keepdims", 1))
    if len(iTList) > 1:
        axes = tuple(iTList[1].data.flatten().astype(int))
        return np.min(X, axis=axes, keepdims=keepdims_bool)
    return np.min(X, keepdims=keepdims_bool)


def compute_cast(iTList, op) -> np.ndarray:
    X = iTList[0].data
    to_dtype_code = op.attrs.get("to")
    ONNX_DTYPE_MAP = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        10: np.float16,
        11: np.float64,
        12: np.uint32,
        13: np.uint64,
    }
    return X.astype(ONNX_DTYPE_MAP.get(to_dtype_code, np.float32))


def compute_nonzero(iTList, op) -> np.ndarray:
    X = iTList[0].data
    indices = np.nonzero(X)
    if len(indices) > 0 and len(indices[0]) > 0:
        return np.stack(indices, axis=0).astype(np.int64)
    return np.zeros((len(X.shape), 0), dtype=np.int64)


def compute_shape(iTList, op) -> np.ndarray:
    return np.array(iTList[0].data.shape, dtype=np.int64)


def compute_sign(iTList, op) -> np.ndarray:
    """Element-wise sign: -1 if x<0, 0 if x==0, 1 if x>0"""
    return np.sign(iTList[0].data).astype(iTList[0].data.dtype)


# ---------------------------------------------------------------------------
# Pure-numpy helpers for numerical validation / inference
# ---------------------------------------------------------------------------


def _numpy_grid_sample_bilinear(input_t, grid):
    """
    Numpy equivalent of:
        F.grid_sample(input_t, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    Args:
        input_t: np.ndarray [N, C, H_in, W_in]
        grid:    np.ndarray [N, H_out, W_out, 2]  -- (x, y) coords in [-1, 1]

    Returns:
        np.ndarray [N, C, H_out, W_out]
    """
    N, C, H_in, W_in = input_t.shape
    _, H_out, W_out, _ = grid.shape

    gx = grid[..., 0].astype(np.float32)  # [N, H_out, W_out] -- x maps to W
    gy = grid[..., 1].astype(np.float32)  # [N, H_out, W_out] -- y maps to H

    # align_corners=False: pixel = (g + 1) / 2 * size - 0.5
    px = (gx + 1.0) * 0.5 * W_in - 0.5
    py = (gy + 1.0) * 0.5 * H_in - 0.5

    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional weights (kept in float32 to match PyTorch precision)
    wx = (px - x0.astype(np.float32))[:, np.newaxis, :, :]  # [N,1,H_out,W_out]
    wy = (py - y0.astype(np.float32))[:, np.newaxis, :, :]

    def gather(xi, yi):
        """Gather pixels; out-of-bounds positions produce zero (padding_mode='zeros')."""
        valid = (xi >= 0) & (xi < W_in) & (yi >= 0) & (yi < H_in)  # [N,H_out,W_out]
        xi_c = np.clip(xi, 0, W_in - 1)
        yi_c = np.clip(yi, 0, H_in - 1)
        # Advanced index: result[n, c, h, w] = input_t[n, c, yi_c[n,h,w], xi_c[n,h,w]]
        n_idx = np.arange(N).reshape(N, 1, 1, 1)
        c_idx = np.arange(C).reshape(1, C, 1, 1)
        yi_bc = yi_c[:, np.newaxis, :, :]
        xi_bc = xi_c[:, np.newaxis, :, :]
        vals = input_t[n_idx, c_idx, yi_bc, xi_bc].astype(np.float32)  # [N,C,H_out,W_out]
        return vals * valid[:, np.newaxis, :, :]

    v00 = gather(x0, y0)
    v10 = gather(x1, y0)
    v01 = gather(x0, y1)
    v11 = gather(x1, y1)

    out = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return out.astype(input_t.dtype)


def _numpy_multi_scale_deformable_attn(
    value_data, spatial_shapes_list, sampling_locs_data, attn_weights_data
):
    """
    Pure numpy computation of multi-scale deformable attention.

    Numerically equivalent to the PyTorch reference
    ``multi_scale_deformable_attn_pytorch`` used in the validation tests.

    Args:
        value_data:          np.ndarray [bs, num_keys, num_heads, embed_dims_per_head]
        spatial_shapes_list: list of (H, W) tuples, len == num_levels
        sampling_locs_data:  np.ndarray [bs, num_queries, num_heads, num_levels, num_points, 2]
                             -- coordinates in [0, 1]
        attn_weights_data:   np.ndarray [bs, num_queries, num_heads, num_levels, num_points]

    Returns:
        np.ndarray [bs, num_queries, num_heads * embed_dims_per_head]
    """
    bs, _, num_heads, embed_dims_per_head = value_data.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locs_data.shape

    # 1. Split value by level
    value_list = []
    start = 0
    for H, W in spatial_shapes_list:
        size = H * W
        value_list.append(value_data[:, start : start + size, :, :])
        start += size

    # 2. Normalise sampling locations [0,1] -> [-1,1]
    sampling_grids = 2.0 * sampling_locs_data.astype(np.float32) - 1.0

    sampling_value_list = []
    for level, (H, W) in enumerate(spatial_shapes_list):
        # value_l: [bs, H*W, num_heads, embed_dims_per_head]
        value_l = value_list[level]

        # -> [bs, H*W, num_heads*embed_dims_per_head]
        val_flat = value_l.reshape(bs, H * W, num_heads * embed_dims_per_head)
        # -> [bs, num_heads*embed_dims_per_head, H*W]
        val_trans = np.ascontiguousarray(val_flat.transpose(0, 2, 1))
        # -> [bs*num_heads, embed_dims_per_head, H, W]
        val_img = val_trans.reshape(bs * num_heads, embed_dims_per_head, H, W)

        # grid for this level: [bs, num_queries, num_heads, num_points, 2]
        grid_l = sampling_grids[:, :, :, level, :, :]
        # -> [bs, num_heads, num_queries, num_points, 2]
        grid_l = np.ascontiguousarray(grid_l.transpose(0, 2, 1, 3, 4))
        # -> [bs*num_heads, num_queries, num_points, 2]
        grid_l = grid_l.reshape(bs * num_heads, num_queries, num_points, 2)

        # bilinear sample: [bs*num_heads, embed_dims_per_head, num_queries, num_points]
        sampled = _numpy_grid_sample_bilinear(val_img, grid_l)
        sampling_value_list.append(sampled)

    # 3. Stack levels and aggregate
    # [bs*num_heads, embed_dims_per_head, num_queries, num_levels, num_points]
    stacked = np.stack(sampling_value_list, axis=-2)
    # [bs*num_heads, embed_dims_per_head, num_queries, num_levels*num_points]
    stacked_flat = stacked.reshape(
        bs * num_heads, embed_dims_per_head, num_queries, num_levels * num_points
    )

    # attn: [bs, num_queries, num_heads, num_levels, num_points]
    #    -> [bs, num_heads, num_queries, num_levels, num_points]
    #    -> [bs*num_heads, 1, num_queries, num_levels*num_points]
    attn = np.ascontiguousarray(attn_weights_data.transpose(0, 2, 1, 3, 4))
    attn = attn.reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    ).astype(np.float32)

    # [bs*num_heads, embed_dims_per_head, num_queries]
    output = (stacked_flat * attn).sum(axis=-1)
    # [bs, num_heads*embed_dims_per_head, num_queries]
    output = output.reshape(bs, num_heads * embed_dims_per_head, num_queries)
    # [bs, num_queries, num_heads*embed_dims_per_head]
    output = np.ascontiguousarray(output.transpose(0, 2, 1))
    return output.astype(value_data.dtype)
