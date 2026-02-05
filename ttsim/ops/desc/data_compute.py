#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Data computation helpers for shape inference functions"""

import numpy as np


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
    return None


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


def compute_add(iTList, op) -> np.ndarray:
    """Element-wise addition with broadcasting"""
    return iTList[0].data + iTList[1].data


def compute_mul(iTList, op) -> np.ndarray:
    """Element-wise multiplication with broadcasting"""
    return iTList[0].data * iTList[1].data


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
            c_out_start = g * C_out_per_group

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


def compute_transpose(iTList, op) -> np.ndarray:
    """
    Compute Transpose operation.

    Args:
        iTList: [data]
        op: SimOp with attrs perm

    Returns:
        Y: Transposed output
    """
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


def compute_relu6(iTList, op) -> np.ndarray:
    """ReLU6 activation: min(max(0, x), 6) = clip(x, 0, 6)"""
    return np.clip(iTList[0].data, 0, 6)


def compute_resize(iTList, op) -> np.ndarray:
    """
    Compute Resize (Upsample) using nearest neighbor interpolation.

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
