#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim conversion of ms_deform_attn_func.py

Provides shape inference and numerical computation for ms_deform_attn_core
using ttsim primitives and operations.

Mirrors the PyTorch implementation:
    - value.split() -> tensor split operation
    - F.grid_sample() -> ttsim grid_sample_fwd
    - torch.stack() -> ttsim stack operation
    - tensor operations (transpose, reshape, flatten)
"""

import sys
import os
import numpy as np
from types import SimpleNamespace

# Add polaris root to path
_file = os.path.abspath(__file__)
_funcs = os.path.dirname(_file)  # functions
_ops = os.path.dirname(_funcs)  # ops
_models = os.path.dirname(_ops)  # models
_detr = os.path.dirname(_models)  # Deformable_DETR
_wl = os.path.dirname(_detr)  # workloads
_polaris = os.path.dirname(_wl)  # polaris
if _polaris not in sys.path:
    sys.path.insert(0, _polaris)

# Import ttsim operations
from ttsim.ops.desc.nn import grid_sample_fwd
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.tensor_op as T


def ms_deform_attn_core_ttsim(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    TTSim implementation of ms_deform_attn_core_pytorch.

    Performs shape inference and numerical computation using ttsim operations.
    Mirrors PyTorch implementation line-by-line.

    Args:
        value: SimTensor [N, S, M, D]
               Flattened multi-scale feature maps where S = sum(H_i * W_i)
        value_spatial_shapes: SimTensor or ndarray [L, 2]
               Spatial dimensions [[H_0, W_0], [H_1, W_1], ..., [H_{L-1}, W_{L-1}]]
        sampling_locations: SimTensor [N, Lq, M, L, P, 2]
               Normalized sampling coordinates in [0, 1] range
        attention_weights: SimTensor [N, Lq, M, L, P]
               Attention weights (should be softmax normalized)

    Returns:
        output: SimTensor [N, Lq, M*D]
                Aggregated multi-scale deformable attention output

    Data Flow (mirrors PyTorch):
        1. Split value by levels: [N, S, M, D] -> List[[N, H_i*W_i, M, D]]
        2. Convert sampling coords: [0,1] -> [-1,1]
        3. Per-level processing:
           - Flatten: [N, H*W, M, D] -> [N, H*W, M*D]
           - Transpose: [N, M*D, H*W]
           - Reshape: [N*M, D, H, W]
           - Grid sample: [N*M, D, Lq, P]
        4. Stack levels: [N*M, D, Lq, L, P]
        5. Flatten last 2 dims: [N*M, D, Lq, L*P]
        6. Apply attention: [N*M, D, Lq, L*P] * [N*M, 1, Lq, L*P] -> sum -> [N*M, D, Lq]
        7. Reshape output: [N, M*D, Lq]
        8. Transpose: [N, Lq, M*D]
        9. Contiguous
    """
    import numpy as np

    # Convert inputs to SimTensor if needed
    if not isinstance(value, SimTensor):
        value = SimTensor(
            {
                "name": "value",
                "shape": (
                    list(value.shape) if hasattr(value, "shape") else value["shape"]
                ),
                "data": value.data if hasattr(value, "data") else value.get("data"),
                "dtype": (
                    value.dtype
                    if hasattr(value, "dtype")
                    else value.get("dtype", np.dtype(np.float32))
                ),
            }
        )

    if not isinstance(sampling_locations, SimTensor):
        sampling_locations = SimTensor(
            {
                "name": "sampling_locations",
                "shape": (
                    list(sampling_locations.shape)
                    if hasattr(sampling_locations, "shape")
                    else sampling_locations["shape"]
                ),
                "data": (
                    sampling_locations.data
                    if hasattr(sampling_locations, "data")
                    else sampling_locations.get("data")
                ),
                "dtype": (
                    sampling_locations.dtype
                    if hasattr(sampling_locations, "dtype")
                    else sampling_locations.get("dtype", np.dtype(np.float32))
                ),
            }
        )

    if not isinstance(attention_weights, SimTensor):
        attention_weights = SimTensor(
            {
                "name": "attention_weights",
                "shape": (
                    list(attention_weights.shape)
                    if hasattr(attention_weights, "shape")
                    else attention_weights["shape"]
                ),
                "data": (
                    attention_weights.data
                    if hasattr(attention_weights, "data")
                    else attention_weights.get("data")
                ),
                "dtype": (
                    attention_weights.dtype
                    if hasattr(attention_weights, "dtype")
                    else attention_weights.get("dtype", np.dtype(np.float32))
                ),
            }
        )

    # Extract spatial shapes data (always needed for split computation)
    if isinstance(value_spatial_shapes, SimTensor):
        spatial_data = value_spatial_shapes.data
    elif hasattr(value_spatial_shapes, "data"):
        spatial_data = value_spatial_shapes.data
    elif isinstance(value_spatial_shapes, np.ndarray):
        spatial_data = value_spatial_shapes
    else:
        raise ValueError(
            "value_spatial_shapes must be SimTensor, have .data attribute, or be ndarray"
        )

    # Get dimensions from shapes
    N_, S_, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape

    # ────────────────────────────────────────────────────────────────────
    # SHAPE INFERENCE MODE (data=None)
    # ────────────────────────────────────────────────────────────────────
    # Check if any required tensor is missing data (shape inference only)
    if (
        value.data is None
        or sampling_locations.data is None
        or attention_weights.data is None
    ):
        return SimTensor(
            {
                "name": "ms_deform_attn_output",
                "shape": [N_, Lq_, M_ * D_],
                "data": None,
                "dtype": (
                    value.dtype if hasattr(value, "dtype") else np.dtype(np.float32)
                ),
            }
        )

    # ────────────────────────────────────────────────────────────────────
    # NUMERICAL COMPUTATION MODE (data available)
    # Uses ttsim operations with data computation
    # ────────────────────────────────────────────────────────────────────

    # Step 1: Split value by spatial levels
    # value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    split_sizes = [int(H * W) for H, W in spatial_data]
    value_data = value.data
    value_list = np.split(value_data, np.cumsum(split_sizes)[:-1], axis=1)

    # Step 2: Convert sampling locations from [0, 1] to [-1, 1]
    # sampling_grids = 2 * sampling_locations - 1
    sampling_grids_data = 2.0 * sampling_locations.data - 1.0

    # Step 3-4: Per-level grid sampling
    sampling_value_list = []

    for lid_, (H_, W_) in enumerate(spatial_data):
        H_, W_ = int(H_), int(W_)

        # value_l_ processing:
        # [N, H*W, M, D] -> flatten(2) -> [N, H*W, M*D] -> transpose(1,2) -> [N, M*D, H*W] -> reshape -> [N*M, D, H, W]
        value_l_data = value_list[lid_]  # [N, H*W, M, D]

        # flatten(2): flatten dimensions 2 and beyond
        value_l_flat = value_l_data.reshape(N_, H_ * W_, M_ * D_)  # [N, H*W, M*D]

        # transpose(1, 2)
        value_l_trans = value_l_flat.transpose(0, 2, 1)  # [N, M*D, H*W]

        # reshape to [N*M, D, H, W]
        value_l_reshaped = value_l_trans.reshape(N_ * M_, D_, H_, W_)

        # sampling_grid_l_ processing:
        # sampling_grids_data: [N, Lq, M, L, P, 2]
        # Select level lid_: [N, Lq, M, P, 2] -> transpose(1,2) -> [N, M, Lq, P, 2] -> flatten(0,1) -> [N*M, Lq, P, 2]
        sampling_grid_l_data = sampling_grids_data[
            :, :, :, lid_, :, :
        ]  # [N, Lq, M, P, 2] - explicit indexing
        sampling_grid_l_trans = sampling_grid_l_data.transpose(
            0, 2, 1, 3, 4
        )  # [N, M, Lq, P, 2]
        sampling_grid_l_flat = sampling_grid_l_trans.reshape(N_ * M_, Lq_, P_, 2)

        # Grid sample using ttsim's grid_sample_fwd
        # F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        input_t = SimpleNamespace(
            shape=list(value_l_reshaped.shape),
            data=value_l_reshaped,
            dtype=value_l_reshaped.dtype,
        )
        grid_t = SimpleNamespace(
            shape=list(sampling_grid_l_flat.shape),
            data=sampling_grid_l_flat,
            dtype=sampling_grid_l_flat.dtype,
        )
        output_t = SimpleNamespace(shape=None, data=None, dtype=None)

        op = SimpleNamespace(
            attrs={"mode": "bilinear", "padding_mode": "zeros", "align_corners": False},
            optype="GridSample",
        )

        # Call ttsim's grid_sample_fwd (handles both shape inference and computation)
        grid_sample_fwd([input_t, grid_t], [output_t], op)
        sampling_value_l_ = output_t.data  # [N*M, D, Lq, P]

        sampling_value_list.append(sampling_value_l_)

    # Step 5: Stack all levels
    # torch.stack(sampling_value_list, dim=-2) -> [N*M, D, Lq, L, P]
    stacked = np.stack(sampling_value_list, axis=-2)  # [N*M, D, Lq, L, P]

    # Step 6: Flatten last 2 dimensions
    # .flatten(-2) -> [N*M, D, Lq, L*P]
    flattened = stacked.reshape(N_ * M_, D_, Lq_, L_ * P_)

    # Step 7: Reshape attention weights
    # attention_weights: [N, Lq, M, L, P]
    # -> transpose(1, 2): [N, M, Lq, L, P]
    # -> reshape: [N*M, 1, Lq, L*P]
    attn_data = attention_weights.data
    attn_trans = attn_data.transpose(0, 2, 1, 3, 4)  # [N, M, Lq, L, P]
    attn_reshaped = attn_trans.reshape(N_ * M_, 1, Lq_, L_ * P_)

    # Step 8: Apply attention weights and sum
    # (flattened * attention_weights).sum(-1)
    # [N*M, D, Lq, L*P] * [N*M, 1, Lq, L*P] -> sum(-1) -> [N*M, D, Lq]
    weighted = (flattened * attn_reshaped).sum(axis=-1)  # [N*M, D, Lq]

    # Step 9: Reshape to final output
    # .view(N, M*D, Lq)
    output_viewed = weighted.reshape(N_, M_ * D_, Lq_)

    # Step 10: Transpose
    # .transpose(1, 2) -> [N, Lq, M*D]
    output_transposed = output_viewed.transpose(0, 2, 1)  # [N, Lq, M*D]

    # Step 11: Contiguous (in NumPy, ensure C-contiguous)
    output_final = np.ascontiguousarray(output_transposed)

    # Return as SimTensor
    return SimTensor(
        {
            "name": "ms_deform_attn_output",
            "shape": list(output_final.shape),
            "data": output_final,
            "dtype": output_final.dtype,
        }
    )
