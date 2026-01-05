#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim conversion of util/misc.py

Design principles:
  - Shape inference only (always compute output shape)
  - Returns SimTensor objects (not SimpleNamespace)
  - Model-specific utilities implemented directly (no custom library calls)
  - Numerical computation optional (when data available)

Includes:
  - NestedTensorTTSim (data container for tensors + mask)
  - interpolate() (F.interpolate equivalent using scipy)
  - nested_tensor_from_tensor_list() (batch NestedTensor creation with padding)
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from ttsim.ops.tensor import SimTensor, shape_as_optional_list
import ttsim.front.functional.op as F

# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers (no-ops in TTSim — single-process simulation)
# ──────────────────────────────────────────────────────────────────────────────


def is_dist_avail_and_initialized():
    """Always False in TTSim — no distributed training."""
    return False


def get_world_size():
    """Always 1 in TTSim — single process."""
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy
# ──────────────────────────────────────────────────────────────────────────────


def accuracy(output, target, topk=(1,)):
    """
    Computes precision@k for the specified values of k.

    TTSim version using NumPy. Mirrors PyTorch's accuracy() from misc.py.

    Args:
        output: SimTensor or ndarray [N, num_classes] — prediction scores
        target: SimTensor or ndarray [N] — ground truth class indices
        topk: Tuple of k values

    Returns:
        List of accuracy percentages (as floats) for each k
    """
    # Extract numpy data
    if isinstance(output, SimTensor):
        out_data = output.data
    elif isinstance(output, np.ndarray):
        out_data = output
    else:
        out_data = np.asarray(output)

    if isinstance(target, SimTensor):
        tgt_data = target.data
    elif isinstance(target, np.ndarray):
        tgt_data = target
    else:
        tgt_data = np.asarray(target)

    if tgt_data is None or tgt_data.size == 0:
        return [0.0]

    maxk = max(topk)
    batch_size = tgt_data.shape[0]

    # Top-k indices along last axis (descending)
    pred = np.argsort(-out_data, axis=1)[:, :maxk]  # [N, maxk]
    pred = pred.T  # [maxk, N]
    correct = pred == tgt_data.reshape(1, -1)  # broadcast

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).astype(np.float64).sum()
        res.append(float(correct_k * 100.0 / batch_size))
    return res


# ──────────────────────────────────────────────────────────────────────────────
# inverse_sigmoid
# ──────────────────────────────────────────────────────────────────────────────


def inverse_sigmoid(x, eps=1e-5):
    """
    Inverse sigmoid (logit) function.

    Mirrors PyTorch:
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)

    Args:
        x: SimTensor or ndarray
        eps: Clamping epsilon

    Returns:
        SimTensor with logit values
    """
    if isinstance(x, SimTensor):
        x_data = x.data
        x_shape = shape_as_optional_list(x.shape)
        x_dtype = x.dtype if hasattr(x, "dtype") else np.float32
    elif isinstance(x, np.ndarray):
        x_data = x
        x_shape = list(x.shape)
        x_dtype = x.dtype
    else:
        x_data = np.asarray(x)
        x_shape = list(x_data.shape)
        x_dtype = x_data.dtype

    if x_data is not None:
        clamped = np.clip(x_data, 0.0, 1.0)
        x1 = np.clip(clamped, eps, None)
        x2 = np.clip(1.0 - clamped, eps, None)
        result_data = np.log(x1 / x2).astype(np.float32)
    else:
        result_data = None

    return SimTensor(
        {
            "name": "inverse_sigmoid_output",
            "shape": x_shape,
            "data": result_data,
            "dtype": x_dtype,
        }
    )


class NestedTensor:
    """
    Lightweight NestedTensor stand-in for TTSim.
    Bundles tensors + mask together, provides decompose() method.
    NOT a SimNN.Module (it's a data container like tuple/dict).

    Attributes:
        tensors: SimTensor or tensor-like object with shape and optional data
        mask: Optional numpy boolean array [B, H, W]
    """

    def __init__(
        self, tensors: Union[SimTensor, np.ndarray], mask: Optional[np.ndarray] = None
    ):
        self.tensors = tensors
        self.mask = mask

    def decompose(self):
        """Return (tensors, mask) tuple."""
        return self.tensors, self.mask

    def __repr__(self):
        if isinstance(self.tensors, SimTensor):
            tensors_shape_repr = repr(shape_as_optional_list(self.tensors.shape))
        elif hasattr(self.tensors, "shape"):
            tensors_shape_repr = repr(self.tensors.shape)
        else:
            tensors_shape_repr = "?"
        return f"NestedTensorTTSim(tensors_shape={tensors_shape_repr}, mask_shape={self.mask.shape if self.mask is not None else None})"


def interpolate(
    input_data, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    TTSim version of F.interpolate with PyTorch-compatible behavior.
    Mirrors torch.nn.functional.interpolate behavior exactly.

    Design: Shape inference always performed, numerical computation when data available.

    Args:
        input_data : SimTensor or ndarray [N, C, H, W]
        size       : tuple (H_out, W_out)
        scale_factor : float or tuple
        mode       : 'nearest' or 'bilinear'
        align_corners : bool (for bilinear mode)

    Returns:
        SimTensor with output shape and optional interpolated data
    """
    # Extract shape and data
    if isinstance(input_data, SimTensor):
        input_shape = shape_as_optional_list(input_data.shape)
        data = input_data.data
        dtype = input_data.dtype if hasattr(input_data, "dtype") else np.float32
    elif isinstance(input_data, np.ndarray):
        input_shape = list(input_data.shape)
        data = input_data
        dtype = input_data.dtype
    else:
        # Handle other tensor-like objects
        input_shape = list(input_data.shape) if hasattr(input_data, "shape") else None
        data = input_data.data if hasattr(input_data, "data") else input_data
        dtype = input_data.dtype if hasattr(input_data, "dtype") else np.float32

    assert input_shape is not None, "interpolate requires a known input shape"

    # Shape inference: Compute output shape
    if size is not None:
        target_h, target_w = size
        output_shape = list(input_shape[:-2]) + [target_h, target_w]
    elif scale_factor is not None:
        if isinstance(scale_factor, (list, tuple)):
            sf_h, sf_w = scale_factor[-2:]
        else:
            sf_h = sf_w = scale_factor
        target_h = int(input_shape[-2] * sf_h)
        target_w = int(input_shape[-1] * sf_w)
        output_shape = list(input_shape[:-2]) + [target_h, target_w]
    else:
        # No resize
        output_shape = input_shape

    # Check if empty using .numel() method (matching PyTorch's behavior)
    num_elements = (
        input_data.numel()
        if isinstance(input_data, SimTensor)
        else (data.size if data is not None else 0)
    )

    # Handle empty tensor or shape inference only
    if data is None or num_elements == 0:
        return SimTensor(
            {
                "name": "interpolate_output",
                "shape": output_shape,
                "data": None,
                "dtype": dtype,
            }
        )

    # Numerical computation: Apply PyTorch-compatible interpolation
    if size is None and scale_factor is None:
        # No resize needed
        target_h, target_w = input_shape[-2:]

    # Get input dimensions
    ndim = len(input_shape)
    if ndim == 4:
        in_h, in_w = input_shape[2], input_shape[3]
    elif ndim == 3:
        in_h, in_w = input_shape[1], input_shape[2]
    elif ndim == 2:
        in_h, in_w = input_shape[0], input_shape[1]
    else:
        raise ValueError(f"Unsupported input dimensions: {ndim}")

    # Apply PyTorch-compatible interpolation
    if mode == "nearest":
        out = _interpolate_nearest_pytorch_compatible(
            data, in_h, in_w, target_h, target_w, ndim
        )
    elif mode == "bilinear":
        if align_corners is None:
            align_corners = False
        out = _interpolate_bilinear_pytorch_compatible(
            data, in_h, in_w, target_h, target_w, ndim, align_corners
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return SimTensor(
        {
            "name": "interpolate_output",
            "shape": list(out.shape),
            "data": out,
            "dtype": out.dtype,
        }
    )


def _interpolate_nearest_pytorch_compatible(data, in_h, in_w, out_h, out_w, ndim):
    """
    PyTorch-compatible nearest neighbor interpolation.
    Uses the same pixel selection strategy as PyTorch.

    PyTorch uses: floor(i * in_size / out_size) for nearest neighbor
    This ensures proper pixel replication for upsampling and selection for downsampling.
    """
    # Create output array
    if ndim == 4:
        n, c = data.shape[0], data.shape[1]
        out = np.zeros((n, c, out_h, out_w), dtype=data.dtype)
    elif ndim == 3:
        c = data.shape[0]
        out = np.zeros((c, out_h, out_w), dtype=data.dtype)
    else:
        out = np.zeros((out_h, out_w), dtype=data.dtype)

    # PyTorch's nearest neighbor mapping
    for i in range(out_h):
        # Simple and accurate: floor(i * in_size / out_size)
        src_i = int((i * in_h) // out_h)
        src_i = min(src_i, in_h - 1)  # Clamp to valid range

        for j in range(out_w):
            src_j = int((j * in_w) // out_w)
            src_j = min(src_j, in_w - 1)

            if ndim == 4:
                out[:, :, i, j] = data[:, :, src_i, src_j]
            elif ndim == 3:
                out[:, i, j] = data[:, src_i, src_j]
            else:
                out[i, j] = data[src_i, src_j]

    return out


def _interpolate_bilinear_pytorch_compatible(
    data, in_h, in_w, out_h, out_w, ndim, align_corners
):
    """
    PyTorch-compatible bilinear interpolation.
    Matches PyTorch's coordinate mapping and interpolation logic.
    """
    # Create output array
    if ndim == 4:
        n, c = data.shape[0], data.shape[1]
        out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    elif ndim == 3:
        c = data.shape[0]
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
    else:
        out = np.zeros((out_h, out_w), dtype=np.float32)

    # Compute coordinate mapping (matching PyTorch logic)
    if align_corners:
        if out_h > 1:
            scale_h = (in_h - 1) / (out_h - 1)
        else:
            scale_h = 0.0
        if out_w > 1:
            scale_w = (in_w - 1) / (out_w - 1)
        else:
            scale_w = 0.0
    else:
        scale_h = in_h / out_h
        scale_w = in_w / out_w

    for i in range(out_h):
        for j in range(out_w):
            # Compute source coordinates (matching PyTorch)
            if align_corners:
                src_y = i * scale_h
                src_x = j * scale_w
            else:
                src_y = (i + 0.5) * scale_h - 0.5
                src_x = (j + 0.5) * scale_w - 0.5

            # Clamp coordinates
            src_y = max(0.0, min(src_y, in_h - 1))
            src_x = max(0.0, min(src_x, in_w - 1))

            # Get integer and fractional parts
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, in_h - 1)
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, in_w - 1)

            wy1 = src_y - y0
            wy0 = 1.0 - wy1
            wx1 = src_x - x0
            wx0 = 1.0 - wx1

            # Bilinear interpolation
            if ndim == 4:
                out[:, :, i, j] = (
                    wy0 * wx0 * data[:, :, y0, x0]
                    + wy0 * wx1 * data[:, :, y0, x1]
                    + wy1 * wx0 * data[:, :, y1, x0]
                    + wy1 * wx1 * data[:, :, y1, x1]
                )
            elif ndim == 3:
                out[:, i, j] = (
                    wy0 * wx0 * data[:, y0, x0]
                    + wy0 * wx1 * data[:, y0, x1]
                    + wy1 * wx0 * data[:, y1, x0]
                    + wy1 * wx1 * data[:, y1, x1]
                )
            else:
                out[i, j] = (
                    wy0 * wx0 * data[y0, x0]
                    + wy0 * wx1 * data[y0, x1]
                    + wy1 * wx0 * data[y1, x0]
                    + wy1 * wx1 * data[y1, x1]
                )

    return out


def _max_by_axis(the_list):
    """
    Helper function to find max along each axis.
    Mirrors PyTorch's _max_by_axis from util/misc.py.

    Args:
        the_list : List[List[int]]  - list of shapes
    Returns:
        List[int]  - max size along each dimension
    """
    # Create a copy to avoid modifying the input list (unlike PyTorch version)
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(
    tensor_list: List[Union[SimTensor, np.ndarray]],
) -> NestedTensor:
    """
    Create NestedTensorTTSim from a list of tensors with potentially different sizes.
    Mirrors util.misc.nested_tensor_from_tensor_list() from PyTorch code.

    Design: Shape inference always performed, numerical padding when data available.

    Args:
        tensor_list : List of SimTensor or ndarrays [C, H, W]

    Returns:
        NestedTensorTTSim with:
            tensors : SimTensor [B, C, max_H, max_W] - padded with zeros
            mask    : ndarray [B, max_H, max_W] - True where padded
    """
    # Extract shapes and check if we have data
    shapes = []
    has_data = True

    for item in tensor_list:
        if isinstance(item, SimTensor):
            # Make a copy to avoid reference issues
            shapes.append(shape_as_optional_list(item.shape))
            if item.data is None:
                has_data = False
        elif isinstance(item, np.ndarray):
            shapes.append(list(item.shape))
        else:
            shapes.append(list(item.shape) if hasattr(item, "shape") else None)  # type: ignore[unreachable]

    if shapes[0] is None or len(shapes[0]) != 3:
        raise ValueError("Only 3D tensors [C, H, W] supported")

    # Shape inference: Find max size
    max_size = _max_by_axis(shapes)
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape

    # Determine dtype
    first_item = tensor_list[0]
    if isinstance(first_item, SimTensor):
        dtype = first_item.dtype if hasattr(first_item, "dtype") else np.float32
    elif isinstance(first_item, np.ndarray):
        dtype = first_item.dtype
    else:
        dtype = first_item.dtype if hasattr(first_item, "dtype") else np.float32  # type: ignore[unreachable]

    # Shape inference only (no data)
    if not has_data:
        tensors_simtensor = SimTensor(
            {
                "name": "nested_tensor",
                "shape": batch_shape,
                "data": None,
                "dtype": dtype,
            }
        )
        # Mask shape inference - mask remains NumPy array (metadata, not computation)
        mask = np.ones((b, h, w), dtype=bool)
        return NestedTensor(tensors_simtensor, mask)

    # Numerical computation: Pad tensors and create mask using ttsim operations
    if shapes[0] is not None and len(shapes[0]) == 3:
        # Create padded tensor using ttsim's zeros operation
        tensor_simtensor = F.zeros("nested_tensor_padded", batch_shape, dtype=dtype)
        tensor = tensor_simtensor.data  # Access underlying NumPy array for manipulation

        # Mask remains NumPy array (it's metadata, not a computational tensor)
        mask = np.ones((b, h, w), dtype=bool)

        # Copy each tensor and set mask
        for i, item in enumerate(tensor_list):
            # Extract data and use pre-computed shapes for dimensions
            if isinstance(item, SimTensor):
                img_data = item.data
            elif isinstance(item, np.ndarray):
                img_data = item
            else:
                img_data = item.data if hasattr(item, "data") else item  # type: ignore[unreachable]

            # Use pre-computed shape from shapes list
            _item_shape = shapes[i]
            assert _item_shape is not None
            img_c, img_h, img_w = _item_shape
            tensor[i, :, :img_h, :img_w] = img_data
            mask[i, :img_h, :img_w] = False
    else:
        raise ValueError("Only 3D tensors [C, H, W] supported")

    # Return as SimTensor (wrapping the modified data)
    tensors_simtensor = SimTensor(
        {
            "name": "nested_tensor",
            "shape": list(tensor.shape),
            "data": tensor,
            "dtype": tensor.dtype,
        }
    )
    return NestedTensor(tensors_simtensor, mask)
