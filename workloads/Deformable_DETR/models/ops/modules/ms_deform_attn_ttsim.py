#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim implementation of Multi-Scale Deformable Attention Module.

Provides shape inference for MSDeformAttn module using ONLY ttsim primitives.
Mirrors the PyTorch implementation structure without adding extra logic.

CRITICAL DESIGN:
- This file uses ONLY ttsim shape inference APIs (from ttsim.ops.desc.*)
- Numerical computation is handled internally by ttsim functions which call data_compute
- Test files will import from this module for shape inference validation
"""

import sys
import os
import numpy as np
import math
import warnings

# Add polaris to path for ttsim imports
_file = os.path.abspath(__file__)
_modules = os.path.dirname(_file)
_ops = os.path.dirname(_modules)
_models = os.path.dirname(_ops)
_detr = os.path.dirname(_models)
_wl = os.path.dirname(_detr)
_polaris = os.path.dirname(_wl)
if _polaris not in sys.path:
    sys.path.insert(0, _polaris)

# Import ONLY ttsim shape inference functions (NOT data_compute directly)
from ttsim.ops.desc.helpers import unary_fwd
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T

# Import the core function from the TTSim function module
# Use direct import since path is already set up correctly
from workloads.Deformable_DETR.models.ops.functions.ms_deform_attn_func_ttsim import (
    ms_deform_attn_core_ttsim,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions (mirrors PyTorch utilities)
# ══════════════════════════════════════════════════════════════════════════════
def _is_power_of_2(n):
    """Check if n is a power of 2."""
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


# ══════════════════════════════════════════════════════════════════════════════
# Initialization Helper Functions (for parameter initialization only)-
# ══════════════════════════════════════════════════════════════════════════════


# all these are numerical compute / nothing related to shape inference
def xavier_uniform_(tensor_shape):
    """Xavier uniform initialization (mirrors torch.nn.init.xavier_uniform_)."""
    fan_in, fan_out = tensor_shape[1], tensor_shape[0]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return np.random.uniform(-a, a, tensor_shape).astype(np.float32)


def constant_(shape, val):
    """Constant initialization (mirrors torch.nn.init.constant_)."""
    return np.full(shape, val, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MSDeformAttn Module (mirrors nn.Module)
# ══════════════════════════════════════════════════════════════════════════════
class MSDeformAttn(SimNN.Module):
    """
    TTSim implementation of Multi-Scale Deformable Attention Module.
    Strictly mirrors the PyTorch MSDeformAttn class from ms_deform_attn.py.
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, name="msda"):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(
                    d_model, n_heads
                )
            )
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.name: str = name
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Create linear layers using SimNN.Linear (unique scoped names fix tensor collision)
        self.sampling_offsets = SimNN.Linear(
            f"{name}.sampling_offsets", d_model, n_heads * n_levels * n_points * 2
        )
        self.attention_weights = SimNN.Linear(
            f"{name}.attention_weights", d_model, n_heads * n_levels * n_points
        )
        self.value_proj = SimNN.Linear(f"{name}.value_proj", d_model, d_model)
        self.output_proj = SimNN.Linear(f"{name}.output_proj", d_model, d_model)

        self._reset_parameters()
        super().link_op2module()

    def _reset_parameters(self):
        """Initialize parameters (mirrors PyTorch initialization)."""
        assert self.sampling_offsets.bias is not None
        assert self.attention_weights.bias is not None
        assert self.value_proj.bias is not None
        assert self.output_proj.bias is not None
        # sampling_offsets: constant_(weight, 0.) and custom bias init
        # SimNN.Linear stores weight as .param (SimTensor) and bias as .bias (SimTensor)
        self.sampling_offsets.param.data = constant_(
            (self.n_heads * self.n_levels * self.n_points * 2, self.d_model), 0.0
        )

        # Grid initialization for sampling_offsets.bias
        thetas = np.arange(self.n_heads, dtype=np.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
        grid_init = (
            grid_init / np.max(np.abs(grid_init), axis=-1, keepdims=True)
        ).reshape(self.n_heads, 1, 1, 2)
        grid_init = np.tile(grid_init, (1, self.n_levels, self.n_points, 1))
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)

        # attention_weights: constant_(weight, 0.) and constant_(bias, 0.)
        self.attention_weights.param.data = constant_(
            (self.n_heads * self.n_levels * self.n_points, self.d_model), 0.0
        )
        self.attention_weights.bias.data = constant_(
            (self.n_heads * self.n_levels * self.n_points,), 0.0
        )

        # value_proj: xavier_uniform_(weight) and constant_(bias, 0.)
        self.value_proj.param.data = xavier_uniform_((self.d_model, self.d_model))
        self.value_proj.bias.data = constant_((self.d_model,), 0.0)

        # output_proj: xavier_uniform_(weight) and constant_(bias, 0.)
        self.output_proj.param.data = xavier_uniform_((self.d_model, self.d_model))
        self.output_proj.bias.data = constant_((self.d_model,), 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index=None,
        input_padding_mask=None,
    ):
        """
        Forward pass (mirrors PyTorch forward).

        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        # Convert inputs to SimpleNamespace if needed
        query = self._to_tensor(query)
        reference_points = self._to_tensor(reference_points)
        input_flatten = self._to_tensor(input_flatten)
        input_spatial_shapes = self._to_tensor(input_spatial_shapes)

        N, Len_q, _ = query.shape
        N_in, Len_in, _ = input_flatten.shape

        # Shape validation (only when data is available)
        if input_spatial_shapes.data is not None:
            assert (
                input_spatial_shapes.data[:, 0] * input_spatial_shapes.data[:, 1]
            ).sum() == Len_in

        # value = self.value_proj(input_flatten)
        value = self.value_proj(input_flatten)

        # if input_padding_mask is not None: value = value.masked_fill(input_padding_mask[..., None], float(0))
        if input_padding_mask is not None:
            input_padding_mask = self._to_tensor(input_padding_mask)
            # For numerical computation: apply mask
            if value.data is not None and input_padding_mask.data is not None:
                # Expand mask: [N, Len_in] -> [N, Len_in, 1]
                mask_expanded = input_padding_mask.data[..., None]  # Add dimension
                # Apply mask: value[mask] = 0.0
                value.data = np.where(mask_expanded, 0.0, value.data)

        # value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value.shape = [N, Len_in, self.n_heads, self.d_model // self.n_heads]
        if value.data is not None:
            value.data = value.data.reshape(value.shape)

        # sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets.shape = [
            N,
            Len_q,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        ]
        if sampling_offsets.data is not None:
            sampling_offsets.data = sampling_offsets.data.reshape(
                sampling_offsets.shape
            )

        # attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = self.attention_weights(query)
        attention_weights.shape = [
            N,
            Len_q,
            self.n_heads,
            self.n_levels * self.n_points,
        ]
        if attention_weights.data is not None:
            attention_weights.data = attention_weights.data.reshape(
                attention_weights.shape
            )

        # attention_weights = F.softmax(attention_weights, -1)
        # Use ttsim's unary_fwd for softmax
        from types import SimpleNamespace

        attention_weights_softmax = SimTensor(
            {
                "name": f"{self.name}.attention_weights_softmax",
                "shape": None,
                "data": None,
                "dtype": None,
            }
        )
        op_softmax = SimpleNamespace(
            attrs={"axis": -1}, optype="Softmax", name="softmax", precision="fp32"
        )
        unary_fwd([attention_weights], [attention_weights_softmax], op_softmax)
        attention_weights = attention_weights_softmax

        # .view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        attention_weights.set_shape([N, Len_q, self.n_heads, self.n_levels, self.n_points])
        if attention_weights.data is not None:
            attention_weights.data = attention_weights.data.reshape(
                attention_weights.shape
            )

        # Compute sampling_locations
        if (
            reference_points.data is not None
            and sampling_offsets.data is not None
            and input_spatial_shapes.data is not None
        ):
            sampling_locations = self._compute_sampling_locations(
                reference_points, sampling_offsets, input_spatial_shapes
            )
        else:
            # Shape inference
            sampling_locations = SimTensor(
                {
                    "name": "sampling_locations",
                    "shape": [N, Len_q, self.n_heads, self.n_levels, self.n_points, 2],
                    "data": None,
                    "dtype": np.dtype(np.float32),
                }
            )

        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = ms_deform_attn_core_ttsim(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        # Register the output tensor with a unique module-scoped name so that
        # get_tensors() traversal can find it and output_proj.matmul's input is
        # present in the WorkloadGraph when add_op() validates it.
        output.name = f"{self.name}.ms_deform_attn_output"
        output.link_module = self
        self._tensors[output.name] = output
        # output = self.output_proj(output)
        output = self.output_proj(output)

        return output

    def __call__(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index=None,
        input_padding_mask=None,
    ):
        """Make MSDeformAttn callable like PyTorch nn.Module"""
        return self.forward(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )

    def _compute_sampling_locations(
        self, reference_points, sampling_offsets, input_spatial_shapes
    ):
        """
        Compute sampling locations (mirrors PyTorch logic).

        This exactly mirrors the logic in the PyTorch forward() method:
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        """
        ref_data = reference_points.data
        offset_data = sampling_offsets.data
        spatial_data = input_spatial_shapes.data

        if ref_data.shape[-1] == 2:
            # offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            offset_normalizer = np.stack(
                [spatial_data[:, 1], spatial_data[:, 0]], axis=-1
            )

            # sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations_data = (
                ref_data[:, :, None, :, None, :]
                + offset_data / offset_normalizer[None, None, None, :, None, :]
            )
        elif ref_data.shape[-1] == 4:
            # sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations_data = (
                ref_data[:, :, None, :, None, :2]
                + offset_data / self.n_points * ref_data[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    ref_data.shape[-1]
                )
            )

        return SimTensor(
            {
                "name": "sampling_locations",
                "shape": list(sampling_locations_data.shape),
                "data": sampling_locations_data,
                "dtype": np.dtype(np.float32),
            }
        )

    def _to_tensor(self, x):
        """Convert input to SimTensor."""
        from types import SimpleNamespace

        if isinstance(x, SimTensor):
            return x
        elif isinstance(x, SimpleNamespace):
            # Handle SimpleNamespace with shape/data/dtype attributes
            return SimTensor(
                {
                    "name": f"{self.name}.input",
                    "shape": x.shape if hasattr(x, "shape") else [],
                    "data": x.data if hasattr(x, "data") else None,
                    "dtype": x.dtype if hasattr(x, "dtype") else np.dtype(np.float32),
                }
            )
        elif isinstance(x, np.ndarray):
            return SimTensor(
                {"name": f"{self.name}.input", "shape": list(x.shape), "data": x, "dtype": x.dtype}
            )
        else:
            arr = np.array(x)
            return SimTensor(
                {
                    "name": f"{self.name}.input",
                    "shape": list(arr.shape),
                    "data": arr,
                    "dtype": arr.dtype,
                }
            )
