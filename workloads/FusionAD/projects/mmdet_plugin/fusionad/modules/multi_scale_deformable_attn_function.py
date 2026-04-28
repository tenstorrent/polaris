#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Multi-Scale Deformable Attention.

This is a CPU-only implementation converted from the PyTorch version in mmcv.
The core algorithm is based on the `multi_scale_deformable_attn_pytorch` function.
"""

#-------------------------------PyTorch--------------------------------

# import torch
# from torch.cuda.amp import custom_bwd, custom_fwd
# from torch.autograd.function import Function, once_differentiable
# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# class MultiScaleDeformableAttnFunction_fp16(Function):
#
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, im2col_step):
#         """GPU version of multi-scale deformable attention.
#
#         Args:
#             value (Tensor): The value has shape
#                 (bs, num_keys, mum_heads, embed_dims//num_heads)
#             value_spatial_shapes (Tensor): Spatial shape of
#                 each feature map, has shape (num_levels, 2),
#                 last dimension 2 represent (h, w)
#             sampling_locations (Tensor): The location of sampling points,
#                 has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points, 2),
#                 the last dimension 2 represent (x, y).
#             attention_weights (Tensor): The weight of sampling points used
#                 when calculate the attention, has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points),
#             im2col_step (Tensor): The step used in image to column.
#
#         Returns:
#             Tensor: has shape (bs, num_queries, embed_dims)
#         """
#         ctx.im2col_step = im2col_step
#         output = ext_module.ms_deform_attn_forward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             im2col_step=ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes,
#                               value_level_start_index, sampling_locations,
#                               attention_weights)
#         return output
#
#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         """GPU version of backward function.
#
#         Args:
#             grad_output (Tensor): Gradient
#                 of output tensor of forward.
#
#         Returns:
#              Tuple[Tensor]: Gradient
#                 of input tensors in forward.
#         """
#         value, value_spatial_shapes, value_level_start_index, \
#             sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value = torch.zeros_like(value)
#         grad_sampling_loc = torch.zeros_like(sampling_locations)
#         grad_attn_weight = torch.zeros_like(attention_weights)
#
#         ext_module.ms_deform_attn_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             grad_output.contiguous(),
#             grad_value,
#             grad_sampling_loc,
#             grad_attn_weight,
#             im2col_step=ctx.im2col_step)
#
#         return grad_value, None, None, \
#             grad_sampling_loc, grad_attn_weight, None


# class MultiScaleDeformableAttnFunction_fp32(Function):
#
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index,
#                 sampling_locations, attention_weights, im2col_step):
#         """GPU version of multi-scale deformable attention.
#
#         Args:
#             value (Tensor): The value has shape
#                 (bs, num_keys, mum_heads, embed_dims//num_heads)
#             value_spatial_shapes (Tensor): Spatial shape of
#                 each feature map, has shape (num_levels, 2),
#                 last dimension 2 represent (h, w)
#             sampling_locations (Tensor): The location of sampling points,
#                 has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points, 2),
#                 the last dimension 2 represent (x, y).
#             attention_weights (Tensor): The weight of sampling points used
#                 when calculate the attention, has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points),
#             im2col_step (Tensor): The step used in image to column.
#
#         Returns:
#             Tensor: has shape (bs, num_queries, embed_dims)
#         """
#
#         ctx.im2col_step = im2col_step
#         output = ext_module.ms_deform_attn_forward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             im2col_step=ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes,
#                               value_level_start_index, sampling_locations,
#                               attention_weights)
#         return output
#
#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         """GPU version of backward function.
#
#         Args:
#             grad_output (Tensor): Gradient
#                 of output tensor of forward.
#
#         Returns:
#              Tuple[Tensor]: Gradient
#                 of input tensors in forward.
#         """
#         value, value_spatial_shapes, value_level_start_index, \
#             sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value = torch.zeros_like(value)
#         grad_sampling_loc = torch.zeros_like(sampling_locations)
#         grad_attn_weight = torch.zeros_like(attention_weights)
#
#         ext_module.ms_deform_attn_backward(
#             value,
#             value_spatial_shapes,
#             value_level_start_index,
#             sampling_locations,
#             attention_weights,
#             grad_output.contiguous(),
#             grad_value,
#             grad_sampling_loc,
#             grad_attn_weight,
#             im2col_step=ctx.im2col_step)
#
#         return grad_value, None, None, \
#             grad_sampling_loc, grad_attn_weight, None

#-------------------------------TTSIM-----------------------------------

import sys
import os

# Add ttsim to path - navigate to polaris root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..','..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class MultiScaleDeformableAttnCore(SimNN.Module):
    """
    TTSim Module implementing the core multi-scale deformable attention computation.

    All ops/tensors are registered via self.xxx = ... (__setattr__), no _reg_tensor/_reg_op needed.

    Args:
        name (str): Module name prefix
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __call__(self, value, value_spatial_shapes, sampling_locations, attention_weights, debug=False):
        """
        Forward pass of multi-scale deformable attention core.

        - Ops registered via setattr (triggers __setattr__ → _op_hndls) + set_module(self)
        - Data tensors registered via self._tensors[name] = tensor (YOLO pattern)
        - Output tensors auto-register via link_module (no manual setattr needed)
        """
        name = self.name

        # Get dimensions
        bs = value.shape[0]
        num_keys = value.shape[1]
        num_heads = value.shape[2]
        embed_dims_per_head = value.shape[3]
        num_queries = sampling_locations.shape[1]
        num_levels = sampling_locations.shape[3]   # from attention Linear (may be > actual)
        num_points = sampling_locations.shape[4]

        # Parse spatial shapes — actual_levels may be < num_levels
        if isinstance(value_spatial_shapes, list):
            spatial_shapes_list = value_spatial_shapes
        else:
            ss_rows = value_spatial_shapes.data.shape[0] if hasattr(value_spatial_shapes, 'data') and value_spatial_shapes.data is not None else num_levels
            spatial_shapes_list = [(int(value_spatial_shapes.data[i, 0]),
                                   int(value_spatial_shapes.data[i, 1]))
                                  for i in range(ss_rows)]
        actual_levels = len(spatial_shapes_list)

        def _check_data(tensor, label):
            if debug and tensor.data is None:
                raise RuntimeError(f"{label} has no data")

        # Step 1: Split value by levels
        split_sizes = [H * W for H, W in spatial_shapes_list]
        value_list = []
        start_idx = 0
        for size in split_sizes:
            end_idx = start_idx + size
            _s = F._from_data(name + f'.value_split_{start_idx}.starts', np.array([start_idx], dtype=np.int64), is_const=True)
            self._tensors[_s.name] = _s
            _e = F._from_data(name + f'.value_split_{start_idx}.ends', np.array([end_idx], dtype=np.int64), is_const=True)
            self._tensors[_e.name] = _e
            _a = F._from_data(name + f'.value_split_{start_idx}.axes', np.array([1], dtype=np.int64), is_const=True)
            self._tensors[_a.name] = _a
            _st = F._from_data(name + f'.value_split_{start_idx}.steps', np.array([1], dtype=np.int64), is_const=True)
            self._tensors[_st.name] = _st
            _op = F.SliceF(name + f'.value_split_{start_idx}', out_shape=[bs, end_idx - start_idx, num_heads, embed_dims_per_head])
            setattr(self, _op.name, _op)
            _op.set_module(self)
            value_level = _op(value, _s, _e, _a, _st)
            _check_data(value_level, f"value_level_{start_idx}")
            value_list.append(value_level)
            start_idx = end_idx

        # Step 2: Normalize sampling locations to [-1, 1]
        self.two_const = F._from_data(name + '.two', np.array([2.0], dtype=np.float32), is_const=True)
        self.one_const = F._from_data(name + '.one', np.array([1.0], dtype=np.float32), is_const=True)

        self.sampling_mul2_op = F.Mul(name + '.sampling_mul2')
        self.sampling_mul2_op.set_module(self)
        sampling_grids = self.sampling_mul2_op(sampling_locations, self.two_const)
        self.sampling_sub1_op = F.Sub(name + '.sampling_sub1')
        self.sampling_sub1_op.set_module(self)
        sampling_grids = self.sampling_sub1_op(sampling_grids, self.one_const)
        _check_data(sampling_grids, "sampling_grids")

        # Step 3: Process each level
        sampling_value_list = []

        for level, (H, W) in enumerate(spatial_shapes_list):
            value_l = value_list[level]

            _shape1 = F._from_data(name + f'.value_l{level}_shape1', np.array([bs, H*W, num_heads * embed_dims_per_head], dtype=np.int64), is_const=True)
            self._tensors[_shape1.name] = _shape1
            _op_flat1 = F.Reshape(name + f'.value_l{level}_flat1')
            setattr(self, _op_flat1.name, _op_flat1)
            _op_flat1.set_module(self)
            value_l_flat = _op_flat1(value_l, _shape1)

            _op_trans = F.Transpose(name + f'.value_l{level}_trans', perm=[0, 2, 1])
            setattr(self, _op_trans.name, _op_trans)
            _op_trans.set_module(self)
            value_l_trans = _op_trans(value_l_flat)

            _img_shape = F._from_data(name + f'.value_l{level}_img_shape', np.array([bs * num_heads, embed_dims_per_head, H, W], dtype=np.int64), is_const=True)
            self._tensors[_img_shape.name] = _img_shape
            _op_img = F.Reshape(name + f'.value_l{level}_img')
            setattr(self, _op_img.name, _op_img)
            _op_img.set_module(self)
            value_l_img = _op_img(value_l_trans, _img_shape)

            # Extract sampling grid for this level
            _gs = F._from_data(name + f'.sampling_grid_l{level}.starts', np.array([level], dtype=np.int64), is_const=True)
            self._tensors[_gs.name] = _gs
            _ge = F._from_data(name + f'.sampling_grid_l{level}.ends', np.array([level + 1], dtype=np.int64), is_const=True)
            self._tensors[_ge.name] = _ge
            _ga = F._from_data(name + f'.sampling_grid_l{level}.axes', np.array([3], dtype=np.int64), is_const=True)
            self._tensors[_ga.name] = _ga
            _gst = F._from_data(name + f'.sampling_grid_l{level}.steps', np.array([1], dtype=np.int64), is_const=True)
            self._tensors[_gst.name] = _gst
            _op_gslice = F.SliceF(name + f'.sampling_grid_l{level}', out_shape=[bs, num_queries, num_heads, 1, num_points, 2])
            setattr(self, _op_gslice.name, _op_gslice)
            _op_gslice.set_module(self)
            sampling_grid_l = _op_gslice(sampling_grids, _gs, _ge, _ga, _gst)
            _check_data(sampling_grid_l, f"sampling_grid_l{level}_slice")

            _sq_axes = F._from_data(name + f'.sampling_grid_l{level}_sq.axes', np.array([3], dtype=np.int64), is_const=True)
            self._tensors[_sq_axes.name] = _sq_axes
            _op_sq = F.Squeeze(name + f'.sampling_grid_l{level}_sq')
            setattr(self, _op_sq.name, _op_sq)
            _op_sq.set_module(self)
            sampling_grid_l = _op_sq(sampling_grid_l, _sq_axes)
            _check_data(sampling_grid_l, f"sampling_grid_l{level}_sq")

            _op_gtrans = F.Transpose(name + f'.sampling_grid_l{level}_trans', perm=[0, 2, 1, 3, 4])
            setattr(self, _op_gtrans.name, _op_gtrans)
            _op_gtrans.set_module(self)
            sampling_grid_l = _op_gtrans(sampling_grid_l)

            _gf_shape = F._from_data(name + f'.sampling_grid_l{level}_flat_shape', np.array([bs * num_heads, num_queries, num_points, 2], dtype=np.int64), is_const=True)
            self._tensors[_gf_shape.name] = _gf_shape
            _op_gflat = F.Reshape(name + f'.sampling_grid_l{level}_flat')
            setattr(self, _op_gflat.name, _op_gflat)
            _op_gflat.set_module(self)
            sampling_grid_l_flat = _op_gflat(sampling_grid_l, _gf_shape)
            _check_data(sampling_grid_l_flat, f"sampling_grid_l{level}_flat")

            _op_gs = F.GridSample(name + f'.grid_sample_l{level}', mode='bilinear', padding_mode='zeros', align_corners=False)
            setattr(self, _op_gs.name, _op_gs)
            _op_gs.set_module(self)
            sampling_value_l = _op_gs(value_l_img, sampling_grid_l_flat)
            _check_data(sampling_value_l, f"sampling_value_l{level}")

            sampling_value_list.append(sampling_value_l)

        # Step 4: Stack and aggregate
        unsqueezed_values = []
        for level, sampling_value_l in enumerate(sampling_value_list):
            _unsq_axes = F._from_data(name + f'.sampling_value_l{level}.unsq_axes', np.array([3], dtype=np.int64), is_const=True)
            self._tensors[_unsq_axes.name] = _unsq_axes
            _op_unsq = F.Unsqueeze(name + f'.sampling_value_l{level}.unsq')
            setattr(self, _op_unsq.name, _op_unsq)
            _op_unsq.set_module(self)
            sampling_value_l_unsq = _op_unsq(sampling_value_l, _unsq_axes)
            unsqueezed_values.append(sampling_value_l_unsq)

        if len(unsqueezed_values) == 1:
            stacked_values = unsqueezed_values[0]
        else:
            self.concat_levels_op = F.ConcatX(name + '.concat_levels', axis=3)
            self.concat_levels_op.set_module(self)
            stacked_values = self.concat_levels_op(*unsqueezed_values)

        self.stacked_flat_shape = F._from_data(name + '.stacked_flat_shape', np.array([bs * num_heads, embed_dims_per_head, num_queries, actual_levels * num_points], dtype=np.int64), is_const=True)
        self.stacked_flat_op = F.Reshape(name + '.stacked_flat')
        self.stacked_flat_op.set_module(self)
        stacked_values_flat = self.stacked_flat_op(stacked_values, self.stacked_flat_shape)
        _check_data(stacked_values_flat, "stacked_values_flat")

        # Slice attention weights to actual_levels when fewer levels than
        # the attention Linear produces (PyTorch CUDA kernel does this via
        # spatial_shapes.shape[0]).
        if actual_levels < num_levels:
            _aw_s = F._from_data(name + '.aw_slice.starts', np.array([0], dtype=np.int64), is_const=True)
            self._tensors[_aw_s.name] = _aw_s
            _aw_e = F._from_data(name + '.aw_slice.ends', np.array([actual_levels], dtype=np.int64), is_const=True)
            self._tensors[_aw_e.name] = _aw_e
            _aw_a = F._from_data(name + '.aw_slice.axes', np.array([3], dtype=np.int64), is_const=True)
            self._tensors[_aw_a.name] = _aw_a
            _aw_st = F._from_data(name + '.aw_slice.steps', np.array([1], dtype=np.int64), is_const=True)
            self._tensors[_aw_st.name] = _aw_st
            _aw_op = F.SliceF(name + '.aw_slice', out_shape=[bs, num_queries, num_heads, actual_levels, num_points])
            setattr(self, _aw_op.name, _aw_op)
            _aw_op.set_module(self)
            attention_weights = _aw_op(attention_weights, _aw_s, _aw_e, _aw_a, _aw_st)

        self.attn_trans_op = F.Transpose(name + '.attn_trans', perm=[0, 2, 1, 3, 4])
        self.attn_trans_op.set_module(self)
        attn_trans = self.attn_trans_op(attention_weights)
        self.attn_flat_shape = F._from_data(name + '.attn_flat_shape', np.array([bs * num_heads, 1, num_queries, actual_levels * num_points], dtype=np.int64), is_const=True)
        self.attn_flat_op = F.Reshape(name + '.attn_flat')
        self.attn_flat_op.set_module(self)
        attn_flat = self.attn_flat_op(attn_trans, self.attn_flat_shape)
        _check_data(attn_flat, "attn_flat")

        self.weighted_op = F.Mul(name + '.weighted')
        self.weighted_op.set_module(self)
        weighted = self.weighted_op(stacked_values_flat, attn_flat)
        _check_data(weighted, "weighted")

        self.aggregate_op = F.ReduceSum(name + '.aggregate', axis=3, keepdims=False)
        self.aggregate_op.set_module(self)
        aggregated = self.aggregate_op(weighted)
        _check_data(aggregated, "aggregated")

        self.output_shape_tensor = F._from_data(name + '.output_shape', np.array([bs, num_heads * embed_dims_per_head, num_queries], dtype=np.int64), is_const=True)
        self.output_reshape_op = F.Reshape(name + '.output_reshape')
        self.output_reshape_op.set_module(self)
        output = self.output_reshape_op(aggregated, self.output_shape_tensor)

        self.output_final_op = F.Transpose(name + '.output_final', perm=[0, 2, 1])
        self.output_final_op.set_module(self)
        output = self.output_final_op(output)
        _check_data(output, "output_final")

        return output


def multi_scale_deformable_attn_ttsim(
    name, value, value_spatial_shapes,
    sampling_locations, attention_weights,
    debug=False, parent_module=None):
    """
    Backward-compatible wrapper around MultiScaleDeformableAttnCore.

    Creates a MultiScaleDeformableAttnCore submodule on parent_module (if provided)
    and delegates to it. For new code, use MultiScaleDeformableAttnCore directly as a submodule.
    """
    if parent_module is not None:
        if not hasattr(parent_module, '_msda_core_' + name):
            core = MultiScaleDeformableAttnCore(name)
            setattr(parent_module, '_msda_core_' + name, core)
        core = getattr(parent_module, '_msda_core_' + name)
    else:
        core = MultiScaleDeformableAttnCore(name)
    return core(value, value_spatial_shapes, sampling_locations, attention_weights, debug)


class MultiScaleDeformableAttention(SimNN.Module):
    """
    TTSim implementation of Multi-Scale Deformable Attention module.

    Used in Deformable DETR and BEVFormer for efficient multi-scale feature aggregation
    with learnable sampling positions.

    Args:
        name (str): Module name
        embed_dims (int): Embedding dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        num_levels (int): Number of feature pyramid levels. Default: 4
        num_points (int): Number of sampling points per head per level. Default: 4
        dropout (float): Dropout rate. Default: 0.1
        batch_first (bool): If True, batch dimension is first. Default: False
        value_proj_ratio (float): Value projection expansion ratio. Default: 1.0

    Input Shapes:
        query: [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
        value: [num_key, bs, embed_dims] or [bs, num_key, embed_dims] if batch_first
        reference_points: [bs, num_query, num_levels, 2] - normalized coordinates in [0, 1]
        spatial_shapes: [num_levels, 2] - (H, W) for each level
        level_start_index: [num_levels] - starting index for each level in value

    Output Shape:
        [num_query, bs, embed_dims] or [bs, num_query, embed_dims] if batch_first
    """

    def __init__(self,
                 name,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 dropout=0.1,
                 batch_first=False,
                 value_proj_ratio=1.0):
        super().__init__()
        self.name = name

        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                           f'but got {embed_dims} and {num_heads}')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.dropout_rate = dropout
        self.batch_first = batch_first
        self.value_proj_ratio = value_proj_ratio

        # Projections
        self.sampling_offsets = SimNN.Linear(
            name + '.sampling_offsets',
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points * 2
        )

        self.attention_weights = SimNN.Linear(
            name + '.attention_weights',
            in_features=embed_dims,
            out_features=num_heads * num_levels * num_points
        )

        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = SimNN.Linear(
            name + '.value_proj',
            in_features=embed_dims,
            out_features=value_proj_size
        )

        self.output_proj = SimNN.Linear(
            name + '.output_proj',
            in_features=value_proj_size,
            out_features=embed_dims
        )

        # Pre-create ops for __call__
        self.query_pos_add = F.Add(name + '.query_pos_add')

        # Batch-first transpose ops
        if not batch_first:
            self.query_transpose_in = F.Transpose(name + '.query_transpose', perm=[1, 0, 2])
            self.value_transpose_in = F.Transpose(name + '.value_transpose', perm=[1, 0, 2])
            self.output_transpose_out = F.Transpose(name + '.output_transpose', perm=[1, 0, 2])
            self.identity_transpose_out = F.Transpose(name + '.identity_transpose', perm=[1, 0, 2])

        # Value reshape
        self.value_reshape_op = F.Reshape(name + '.value_reshape')

        # Offsets/attention reshapes and softmax
        self.offsets_reshape_op = F.Reshape(name + '.offsets_reshape')
        self.attn_reshape_op = F.Reshape(name + '.attn_reshape')
        self.attn_softmax_op = F.Softmax(name + '.attn_softmax', axis=-1)
        self.attn_reshape2_op = F.Reshape(name + '.attn_reshape2')

        # Reference point unsqueeze ops
        self.ref_unsq1 = F.Unsqueeze(name + '.ref_unsq1')
        self.ref_unsq2 = F.Unsqueeze(name + '.ref_unsq2')
        self.ax2_tensor = F._from_data(name + '.ax2', np.array([2], dtype=np.int64), is_const=True)
        self.ax4_tensor = F._from_data(name + '.ax4', np.array([4], dtype=np.int64), is_const=True)

        # Normalizer unsqueeze ops: (num_levels, 2) -> (1, 1, 1, num_levels, 1, 2)
        self.norm_unsq1 = F.Unsqueeze(name + '.norm_unsq1')
        self.norm_unsq2 = F.Unsqueeze(name + '.norm_unsq2')
        self.norm_unsq3 = F.Unsqueeze(name + '.norm_unsq3')
        self.norm_unsq4 = F.Unsqueeze(name + '.norm_unsq4')
        self.ax0_tensor = F._from_data(name + '.ax0', np.array([0], dtype=np.int64), is_const=True)
        self.ax0_2_tensor = F._from_data(name + '.ax0_2', np.array([0], dtype=np.int64), is_const=True)
        self.ax0_3_tensor = F._from_data(name + '.ax0_3', np.array([0], dtype=np.int64), is_const=True)
        self.ax_neg2_tensor = F._from_data(name + '.ax_neg2', np.array([-2], dtype=np.int64), is_const=True)

        # Normalize and combine
        self.norm_offsets_div = F.Div(name + '.norm_offsets')
        self.sampling_locs_add = F.Add(name + '.sampling_locs')

        # Dropout and residual
        if dropout > 0:
            self.dropout_op = F.Dropout(name + '.dropout', dropout, True)
        self.residual_add = F.Add(name + '.residual')

    def __call__(self,
                 query,
                 value=None,
                 identity=None,
                 query_pos=None,
                 reference_points=None,
                 spatial_shapes=None,
                 level_start_index=None,
                 **kwargs):
        """
        Forward pass of Multi-Scale Deformable Attention.

        Args:
            query: Query features
            value: Value features (if None, uses query)
            identity: Identity for residual connection (if None, uses query)
            query_pos: Positional encoding for query
            reference_points: Normalized reference points [bs, num_query, num_levels, 2]
            spatial_shapes: Spatial shapes for each level
            level_start_index: Starting indices for each level

        Returns:
            Output features with residual connection and dropout
        """

        if value is None:
            value = query

        if identity is None:
            identity = query

        # Add positional encoding
        if query_pos is not None:
            query = self.query_pos_add(query, query_pos)

        # Handle batch_first flag
        if not self.batch_first:
            query = self.query_transpose_in(query)
            value = self.value_transpose_in(value)

        bs = query.shape[0]
        num_query = query.shape[1]
        num_value = value.shape[1]

        # Project value
        value = self.value_proj(value)

        # Reshape value: [bs, num_value, num_heads, embed_dims_per_head]
        embed_dims_per_head = value.shape[2] // self.num_heads
        self.value_reshape_shape = F._from_data(
            self.name + '.value_reshape_shape',
            np.array([bs, num_value, self.num_heads, embed_dims_per_head], dtype=np.int64),
            is_const=True
        )
        value = self.value_reshape_op(value, self.value_reshape_shape)

        # Compute sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        self.offsets_shape_tensor = F._from_data(
            self.name + '.offsets_shape',
            np.array([bs, num_query, self.num_heads, self.num_levels, self.num_points, 2], dtype=np.int64),
            is_const=True
        )
        sampling_offsets = self.offsets_reshape_op(sampling_offsets, self.offsets_shape_tensor)

        # Compute attention weights
        attention_weights = self.attention_weights(query)
        self.attn_shape_tensor = F._from_data(
            self.name + '.attn_shape',
            np.array([bs, num_query, self.num_heads, self.num_levels * self.num_points], dtype=np.int64),
            is_const=True
        )
        attention_weights = self.attn_reshape_op(attention_weights, self.attn_shape_tensor)

        # Apply softmax
        attention_weights = self.attn_softmax_op(attention_weights)

        # Reshape attention weights back
        self.attn_shape2_tensor = F._from_data(
            self.name + '.attn_shape2',
            np.array([bs, num_query, self.num_heads, self.num_levels, self.num_points], dtype=np.int64),
            is_const=True
        )
        attention_weights = self.attn_reshape2_op(attention_weights, self.attn_shape2_tensor)

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer_data = np.array(
                [[float(W), float(H)] for H, W in spatial_shapes],
                dtype=np.float32
            )
            self.offset_normalizer = F._from_data(
                self.name + '.offset_normalizer',
                offset_normalizer_data,
                is_const=True
            )

            # Expand reference_points: [bs, num_query, 1, num_levels, 1, 2]
            ref_pts = self.ref_unsq1(reference_points, self.ax2_tensor)
            ref_pts = self.ref_unsq2(ref_pts, self.ax4_tensor)

            # Expand normalizer: (num_levels, 2) -> (1, 1, 1, num_levels, 1, 2)
            norm = self.norm_unsq1(self.offset_normalizer, self.ax0_tensor)    # (1, num_levels, 2)
            norm = self.norm_unsq2(norm, self.ax0_2_tensor)                     # (1, 1, num_levels, 2)
            norm = self.norm_unsq3(norm, self.ax0_3_tensor)                     # (1, 1, 1, num_levels, 2)
            norm = self.norm_unsq4(norm, self.ax_neg2_tensor)                   # (1, 1, 1, num_levels, 1, 2)

            normalized_offsets = self.norm_offsets_div(sampling_offsets, norm)
            sampling_locations = self.sampling_locs_add(ref_pts, normalized_offsets)
        else:
            raise NotImplementedError("4D reference points not yet implemented in TTSim")

        # Apply multi-scale deformable attention
        output = multi_scale_deformable_attn_ttsim(
            self.name + '.msda',
            value,
            spatial_shapes,
            sampling_locations,
            attention_weights,
            parent_module=self
        )

        # Output projection
        output = self.output_proj(output)

        # Apply dropout
        if self.dropout_rate > 0:
            output = self.dropout_op(output)

        # Handle batch_first flag for output
        if not self.batch_first:
            output = self.output_transpose_out(output)
            # Note: identity is already in the original layout (nq, bs, E)
            # so do NOT transpose it again

        # Residual connection
        output = self.residual_add(output, identity)

        return output

    def analytical_param_count(self):
        """Calculate total number of parameters."""
        count = 0
        count += self.sampling_offsets.analytical_param_count(0)
        count += self.attention_weights.analytical_param_count(0)
        count += self.value_proj.analytical_param_count(0)
        count += self.output_proj.analytical_param_count(0)
        return count
