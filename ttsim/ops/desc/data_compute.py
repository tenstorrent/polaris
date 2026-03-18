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


def compute_sin(iTList, op) -> np.ndarray:
    """Element-wise sine"""
    data = iTList[0].data
    result = np.sin(data)
    # Preserve non-floating dtypes: only cast back when input is floating
    if np.issubdtype(data.dtype, np.floating):
        result = result.astype(data.dtype)
    return result


def compute_cos(iTList, op) -> np.ndarray:
    """Element-wise cosine"""
    data = iTList[0].data
    result = np.cos(data)
    # Preserve non-floating dtypes: only cast back when input is floating
    if np.issubdtype(data.dtype, np.floating):
        result = result.astype(data.dtype)
    return result


def compute_atan(iTList, op) -> np.ndarray:
    """Element-wise arctangent"""
    return np.arctan(iTList[0].data).astype(iTList[0].data.dtype)


def compute_sign(iTList, op) -> np.ndarray:
    """Element-wise sign: -1 if x<0, 0 if x==0, 1 if x>0"""
    return np.sign(iTList[0].data).astype(iTList[0].data.dtype)


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

    # Ensure variance is non-negative (avoid numerical issues)
    var = np.maximum(var, 0.0)

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
    keepdims = op.attrs.get("keepdims", 1)
    noop = op.attrs.get("noop_with_empty_axes", 0)

    if axes is None:
        if noop:
            return X.copy()
        else:
            axes = None
    else:
        axes = tuple(int(a) for a in axes)

    return np.sum(X, axis=axes, keepdims=bool(keepdims))


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


def compute_unsqueeze(iTList, op) -> np.ndarray:
    """
    Add dimension(s) to array at specified axis.

    Args:
        iTList: [data, axes] where axes is int64 array
        op: SimOp

    Returns:
        Y: Array with added dimensions
    """
    data = iTList[0].data
    axes = iTList[1].data

    if np.isscalar(axes) or axes.ndim == 0:
        axes = [int(axes)]
    else:
        axes = [int(a) for a in axes]

    axes = sorted(axes)

    result = data
    for axis in axes:
        result = np.expand_dims(result, axis=axis)

    return result


def compute_squeeze(iTList, op) -> np.ndarray:
    """
    Remove dimension(s) from array at specified axes.

    Args:
        iTList: [data, axes] where axes is int64 array
        op: SimOp

    Returns:
        Y: Array with removed dimensions
    """
    data = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None

    if axes is None:
        return np.squeeze(data)

    if np.isscalar(axes) or axes.ndim == 0:
        axes = [int(axes)]
    else:
        axes = [int(a) for a in axes]

    axes = tuple(axes)
    return np.squeeze(data, axis=axes)


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


def compute_gridsample(iTList, op) -> np.ndarray:
    """
    Compute GridSample output using bilinear interpolation.

    This is a CPU-only implementation of torch.nn.functional.grid_sample
    with bilinear interpolation mode and zeros padding mode.

    Args:
        iTList: [input, grid] where:
            input: [N, C, H_in, W_in] - input feature map
            grid: [N, H_out, W_out, 2] - sampling locations (x, y) in [-1, 1]
        op: SimOp with attrs mode, padding_mode, align_corners

    Returns:
        output: [N, C, H_out, W_out] - sampled features
    """
    input_data = iTList[0].data  # [N, C, H_in, W_in]
    grid_data = iTList[1].data  # [N, H_out, W_out, 2]

    mode = op.attrs.get("mode", "bilinear")
    padding_mode = op.attrs.get("padding_mode", "zeros")
    align_corners = op.attrs.get("align_corners", 0)

    N, C, H_in, W_in = input_data.shape
    N_grid, H_out, W_out, _ = grid_data.shape

    assert N == N_grid, f"Batch size mismatch"
    assert grid_data.shape[3] == 2, f"Grid must have 2 coordinates (x, y)"

    if align_corners:
        grid_x = ((grid_data[..., 0] + 1) / 2) * (W_in - 1)
        grid_y = ((grid_data[..., 1] + 1) / 2) * (H_in - 1)
    else:
        grid_x = ((grid_data[..., 0] + 1) * W_in - 1) / 2
        grid_y = ((grid_data[..., 1] + 1) * H_in - 1) / 2

    output = np.zeros((N, C, H_out, W_out), dtype=input_data.dtype)

    if mode == "nearest":
        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    x = grid_x[n, h_out, w_out]
                    y = grid_y[n, h_out, w_out]
                    ix = int(np.round(x))
                    iy = int(np.round(y))
                    if padding_mode == "zeros":
                        if 0 <= ix < W_in and 0 <= iy < H_in:
                            output[n, :, h_out, w_out] = input_data[n, :, iy, ix]
                    elif padding_mode == "border":
                        ix = np.clip(ix, 0, W_in - 1)
                        iy = np.clip(iy, 0, H_in - 1)
                        output[n, :, h_out, w_out] = input_data[n, :, iy, ix]

    elif mode == "bilinear":
        for n in range(N):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    x = grid_x[n, h_out, w_out]
                    y = grid_y[n, h_out, w_out]
                    x0 = int(np.floor(x))
                    x1 = x0 + 1
                    y0 = int(np.floor(y))
                    y1 = y0 + 1
                    wx1 = x - x0
                    wx0 = 1.0 - wx1
                    wy1 = y - y0
                    wy0 = 1.0 - wy1
                    for c in range(C):
                        val = 0.0
                        if padding_mode == "zeros":
                            if 0 <= x0 < W_in and 0 <= y0 < H_in:
                                val += wx0 * wy0 * input_data[n, c, y0, x0]
                            if 0 <= x1 < W_in and 0 <= y0 < H_in:
                                val += wx1 * wy0 * input_data[n, c, y0, x1]
                            if 0 <= x0 < W_in and 0 <= y1 < H_in:
                                val += wx0 * wy1 * input_data[n, c, y1, x0]
                            if 0 <= x1 < W_in and 0 <= y1 < H_in:
                                val += wx1 * wy1 * input_data[n, c, y1, x1]
                        elif padding_mode == "border":
                            val += (
                                wx0
                                * wy0
                                * input_data[
                                    n,
                                    c,
                                    np.clip(y0, 0, H_in - 1),
                                    np.clip(x0, 0, W_in - 1),
                                ]
                            )
                            val += (
                                wx1
                                * wy0
                                * input_data[
                                    n,
                                    c,
                                    np.clip(y0, 0, H_in - 1),
                                    np.clip(x1, 0, W_in - 1),
                                ]
                            )
                            val += (
                                wx0
                                * wy1
                                * input_data[
                                    n,
                                    c,
                                    np.clip(y1, 0, H_in - 1),
                                    np.clip(x0, 0, W_in - 1),
                                ]
                            )
                            val += (
                                wx1
                                * wy1
                                * input_data[
                                    n,
                                    c,
                                    np.clip(y1, 0, H_in - 1),
                                    np.clip(x1, 0, W_in - 1),
                                ]
                            )
                        output[n, c, h_out, w_out] = val
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return output


def compute_gather(iTList, op) -> np.ndarray:
    X = iTList[0].data
    indices = iTList[1].data
    axis = op.attrs.get("axis", 0)
    return np.take(X, indices, axis=axis)


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


def compute_gelu(iTList, op) -> np.ndarray:
    X = iTList[0].data
    return (
        0.5
        * X
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (X + 0.044715 * np.power(X, 3))))
    )


def compute_mod(iTList, op) -> np.ndarray:
    return np.fmod(iTList[0].data, iTList[1].data)


def compute_reducemin(iTList, op) -> np.ndarray:
    X = iTList[0].data
    keepdims_bool = bool(op.attrs.get("keepdims", 1))
    if len(iTList) > 1:
        axes = tuple(iTList[1].data.flatten().astype(int))
        return np.min(X, axis=axes, keepdims=keepdims_bool)
    return np.min(X, keepdims=keepdims_bool)


def compute_where(iTList, op) -> np.ndarray:
    return np.where(iTList[0].data, iTList[1].data, iTList[2].data)


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


def compute_upsample(iTList, op) -> np.ndarray:
    X = iTList[0].data
    scales = iTList[1].data
    mode = op.attrs.get("mode", "nearest")
    if mode != "nearest":
        raise NotImplementedError(f"Unsupported upsample mode: {mode}")
    if len(X.shape) == 4:
        output = np.repeat(X, int(scales[2]), axis=2)
        return np.repeat(output, int(scales[3]), axis=3)
    elif len(X.shape) == 2:
        output = np.repeat(X, int(scales[0]), axis=0)
        return np.repeat(output, int(scales[1]), axis=1)
    raise NotImplementedError(f"Unsupported input shape for upsample: {X.shape}")


def compute_shape(iTList, op) -> np.ndarray:
    return np.array(iTList[0].data.shape, dtype=np.int64)


def compute_abs(iTList, op) -> np.ndarray:
    return np.abs(iTList[0].data)


def compute_neg(iTList, op) -> np.ndarray:
    return np.negative(iTList[0].data)


def compute_less(iTList, op) -> np.ndarray:
    return iTList[0].data < iTList[1].data


def compute_reducemax(iTList, op) -> np.ndarray:
    X = iTList[0].data
    axes = iTList[1].data if len(iTList) > 1 else None
    keepdims = op.attrs.get("keepdims", 1)
    noop = op.attrs.get("noop_with_empty_axes", 0)

    if axes is None:
        if noop:
            return X.copy()
        else:
            axes = None
    else:
        axes = tuple(int(a) for a in axes)

    return np.max(X, axis=axes, keepdims=bool(keepdims))


def compute_scatter_nd(iTList, op) -> np.ndarray:
    data = iTList[0].data
    indices = iTList[1].data
    updates = iTList[2].data
    reduction = op.attrs.get("reduction", "none")

    output = data.copy()
    K = indices.shape[-1]
    flat_idx = indices.reshape(-1, K)
    flat_upd = updates.reshape(-1, *data.shape[K:])
    for i in range(flat_idx.shape[0]):
        idx = tuple(flat_idx[i])
        if reduction == "none":
            output[idx] = flat_upd[i]
        elif reduction == "add":
            output[idx] += flat_upd[i]
        elif reduction == "mul":
            output[idx] *= flat_upd[i]
    return output


def compute_groupnorm(iTList, op) -> np.ndarray:
    X = iTList[0].data
    weight = iTList[1].data
    bias = iTList[2].data if len(iTList) > 2 else None
    num_groups = op.attrs.get("num_groups")
    eps = op.attrs.get("eps", 1e-5)

    N, C, H, W = X.shape
    G = num_groups
    x_g = X.reshape(N, G, C // G, H, W)
    mean = np.mean(x_g, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_g, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_g - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)

    result = x_norm * weight.reshape(1, C, 1, 1)
    if bias is not None:
        result = result + bias.reshape(1, C, 1, 1)
    return result


def compute_layernorm(iTList, op) -> np.ndarray:
    X = iTList[0].data
    scale = iTList[1].data
    bias = iTList[2].data if len(iTList) > 2 else None
    axis = op.attrs.get("axis", -1)
    eps = op.attrs.get("epsilon", 1e-5)

    rank = len(X.shape)
    if axis < 0:
        axis += rank
    norm_axes = tuple(range(axis, rank))
    mean = np.mean(X, axis=norm_axes, keepdims=True)
    var = np.var(X, axis=norm_axes, keepdims=True)
    x_norm = (X - mean) / np.sqrt(var + eps)
    result = x_norm * scale
    if bias is not None:
        result = result + bias
    return result


def compute_argmax(iTList, op) -> np.ndarray:
    X = iTList[0].data
    axis = op.attrs.get("axis", 0)
    keepdims = op.attrs.get("keepdims", 1)
    select_last = op.attrs.get("select_last_index", 0)

    rank = len(X.shape)
    if axis < 0:
        axis += rank
    if select_last:
        X_rev = np.flip(X, axis=axis)
        idx_rev = np.argmax(X_rev, axis=axis, keepdims=bool(keepdims))
        return (X.shape[axis] - 1 - idx_rev).astype(np.int64)
    return np.argmax(X, axis=axis, keepdims=bool(keepdims)).astype(np.int64)


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
