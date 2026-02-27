#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim-Compatible Implementation of Deformable DETR Segmentation Modules

FIXED VERSION: Bypasses Dropout when prob=0.0 to avoid TTSim Dropout data_compute bug.

This module provides TTSim-compatible implementations of the segmentation components
from Deformable DETR, maintaining functional equivalence with the PyTorch original
while adhering to Polaris TTSim simulation framework requirements.

Modules:
    - MHAttentionMap: Multi-head attention map for generating spatial attention weights
    - MaskHeadSmallConv: FPN-based convolutional mask prediction head
    - DETRsegm: Complete segmentation model wrapper
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T
from ttsim.ops import SimTensor

# ============================================================================
# Helper Functions for Missing PyTorch Operations
# ============================================================================


def masked_fill_impl(tensor, mask, value, module=None, call_id=None):
    """
    TTSim implementation of torch.masked_fill.

    Decomposes masked_fill into TTSim primitives:
        result = tensor * (1 - mask) + value * mask

    Special handling for infinite values (-inf, inf) to avoid nan from inf*0.
    """
    op_id = call_id if call_id is not None else id(tensor)

    value_const_name = f"masked_fill_value_{op_id}"
    value_tensor = F._from_data(
        value_const_name, np.array(value, dtype=np.float32), is_const=True
    )

    ones_name = f"masked_fill_ones_{op_id}"
    ones_tensor = F._from_data(
        ones_name, np.array(1.0, dtype=np.float32), is_const=True
    )

    sub_op = F.Sub(f"masked_fill_sub_{op_id}")
    if module is not None:
        sub_op.set_module(module)
    inverted_mask = sub_op(ones_tensor, mask)

    mul_op1 = F.Mul(f"masked_fill_mul1_{op_id}")
    if module is not None:
        mul_op1.set_module(module)
    kept_values = mul_op1(tensor, inverted_mask)

    mul_op2 = F.Mul(f"masked_fill_mul2_{op_id}")
    if module is not None:
        mul_op2.set_module(module)
    filled_values = mul_op2(value_tensor, mask)

    add_op = F.Add(f"masked_fill_add_{op_id}")
    if module is not None:
        add_op.set_module(module)
    result = add_op(kept_values, filled_values)

    # Special handling for infinite values
    if np.isinf(value) and tensor.data is not None and mask.data is not None:
        mask_broadcast = np.broadcast_to(mask.data, tensor.data.shape)
        result.data = np.where(mask_broadcast > 0, value, tensor.data).astype(
            np.float32
        )

    return result


def interpolate_nearest(tensor, size, module=None, call_id=None):
    """TTSim implementation of F.interpolate with mode='nearest'.

    Delegates to SimTensor.interpolate() (which uses F.Resize internally).
    Computes scale factors from the target size and input spatial dimensions.
    """
    if module is not None:
        tensor.set_module(module)

    if hasattr(tensor, "shape") and len(tensor.shape) == 4:
        H_in, W_in = tensor.shape[2], tensor.shape[3]
        H_out, W_out = size
        scale_h = H_out / H_in
        scale_w = W_out / W_in
    else:
        scale_h = float(size[0])
        scale_w = float(size[1])

    return tensor.interpolate(scale_factor=[scale_h, scale_w])


def conv2d_functional(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    module=None,
    call_id=None,
):
    """TTSim functional convolution — equivalent to torch.nn.functional.conv2d.

    F.Conv2d (module-level API) stores its own internal parameter tensors and
    cannot accept externally supplied weight/bias SimTensors.  This helper
    creates a SimOpHandle with the external weight/bias passed as params,
    preserving numerical data for end-to-end verification.
    """
    import ttsim.utils.common as common

    op_id = call_id if call_id is not None else id(input)
    op_name = f"conv2d_functional_{op_id}"

    stride = common.make_tuple(stride, 2)
    padding = common.make_tuple(padding, 2 * 2)
    dilation = common.make_tuple(dilation, 2)

    params_list = [(1, weight)]
    if bias is not None:
        params_list.append((2, bias))

    conv_op = F.SimOpHandle(
        op_name,
        "Conv",
        params=params_list,
        ipos=[0],
        group=groups,
        strides=stride,
        pads=padding,
        dilations=dilation,
    )

    if module is not None:
        conv_op.set_module(module)

    result = conv_op(input)
    result.link_module = module

    return result


# ============================================================================
# MHAttentionMap: Multi-Head Attention Map (FIXED)
# ============================================================================


class MHAttentionMap(SimNN.Module):
    """
    Multi-Head Attention Map Module (TTSim-Compatible).

    FIXED: Bypasses Dropout when prob=0.0 to avoid TTSim Dropout data_compute bug.

    Computes spatial attention weights using multi-head attention mechanism.
    Returns only the attention softmax weights (no value multiplication).
    """

    def __init__(self, name, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.name = name
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout
        self.head_dim = hidden_dim // num_heads
        self.normalize_fact = float(self.head_dim) ** -0.5

        # Query linear projection weights
        self.q_linear_weight = F._from_shape(
            name + ".q_linear.weight", [query_dim, hidden_dim], is_param=True
        )
        self.q_linear_bias = (
            F._from_shape(name + ".q_linear.bias", [hidden_dim], is_param=True)
            if bias
            else None
        )

        # Key conv2d projection weights
        self.k_linear_weight = F._from_shape(
            name + ".k_linear.weight", [hidden_dim, query_dim, 1, 1], is_param=True
        )
        self.k_linear_bias = (
            F._from_shape(name + ".k_linear.bias", [hidden_dim], is_param=True)
            if bias
            else None
        )

        # Only create Dropout if prob > 0
        # FIX: TTSim Dropout data_compute has a bug that applies dropout even when train_mode=False
        if dropout > 0.0:
            self.dropout = F.Dropout(name + ".dropout", prob=dropout, train_mode=False)
        else:
            self.dropout = None  # Bypass dropout entirely

        super().link_op2module()

    def __call__(self, q, k, mask=None):
        """Forward pass for attention map computation."""
        call_id = id(q)

        # Step 1: Project queries [B, Q, query_dim] -> [B, Q, hidden_dim]
        q_matmul_op = F.MatMul(f"{self.name}.q_matmul_{call_id}")
        q_matmul_op.set_module(self)
        q_proj = q_matmul_op(q, self.q_linear_weight)

        if self.q_linear_bias is not None:
            q_add_op = F.Add(f"{self.name}.q_bias_add_{call_id}")
            q_add_op.set_module(self)
            q_proj = q_add_op(q_proj, self.q_linear_bias)

        # Step 2: Project keys using 1x1 conv
        k_proj = conv2d_functional(
            k,
            self.k_linear_weight,
            self.k_linear_bias,
            stride=1,
            padding=0,
            groups=1,
            module=self,
            call_id=call_id,
        )

        # Step 3: Reshape queries to multi-head format
        B = q_proj.shape[0]
        Q = q_proj.shape[1]
        qh_shape = [B, Q, self.num_heads, self.head_dim]
        reshape_q_op = F.Reshape(f"{self.name}.reshape_q_{call_id}")
        reshape_q_op.set_module(self)
        shape_tensor_q = F._from_data(
            f"{self.name}.reshape_q.shape_{call_id}",
            data=np.array(qh_shape, dtype=np.int64),
            is_const=True,
        )
        qh = reshape_q_op(q_proj, shape_tensor_q)

        # Step 4: Reshape keys to multi-head format
        H = k_proj.shape[2]
        W = k_proj.shape[3]
        kh_shape = [k_proj.shape[0], self.num_heads, self.head_dim, H, W]
        reshape_k_op = F.Reshape(f"{self.name}.reshape_k_{call_id}")
        reshape_k_op.set_module(self)
        shape_tensor_k = F._from_data(
            f"{self.name}.reshape_k.shape_{call_id}",
            data=np.array(kh_shape, dtype=np.int64),
            is_const=True,
        )
        kh = reshape_k_op(k_proj, shape_tensor_k)

        # Step 5: Apply scaling to queries
        scale_const = F._from_data(
            f"{self.name}.scale_{call_id}",
            np.array(self.normalize_fact, dtype=np.float32),
            is_const=True,
        )
        mul_op = F.Mul(f"{self.name}.qh_scale_{call_id}")
        mul_op.set_module(self)
        qh_scaled = mul_op(qh, scale_const)

        # Step 6: Compute attention scores using einsum
        weights = F.Einsum(
            f"{self.name}.einsum_{call_id}", "bqnc,bnchw->bqnhw", qh_scaled, kh
        )

        # Step 7: Apply mask if provided
        if mask is not None:
            unsqueeze1_op = F.Unsqueeze(f"{self.name}.mask_unsqueeze1_{call_id}")
            unsqueeze1_op.set_module(self)
            axes1_tensor = F._from_data(
                f"{self.name}.mask_unsqueeze1.axes_{call_id}",
                data=np.array([1], dtype=np.int64),
                is_const=True,
            )
            mask_u1 = unsqueeze1_op(mask, axes1_tensor)

            # FIX: F.Unsqueeze doesn't propagate data — do it manually
            if mask.data is not None:
                mask_u1.data = np.expand_dims(mask.data, axis=1)

            unsqueeze2_op = F.Unsqueeze(f"{self.name}.mask_unsqueeze2_{call_id}")
            unsqueeze2_op.set_module(self)
            axes2_tensor = F._from_data(
                f"{self.name}.mask_unsqueeze2.axes_{call_id}",
                data=np.array([1], dtype=np.int64),
                is_const=True,
            )
            mask_expanded = unsqueeze2_op(mask_u1, axes2_tensor)

            # FIX: F.Unsqueeze doesn't propagate data — do it manually
            if mask_u1.data is not None:
                mask_expanded.data = np.expand_dims(mask_u1.data, axis=1)

            weights = masked_fill_impl(
                weights, mask_expanded, float("-inf"), module=self, call_id=call_id
            )

        # Step 8: Flatten for softmax
        nheads = self.num_heads
        flatten_shape = [B, Q, nheads * H * W]
        flatten_op = F.Reshape(f"{self.name}.flatten_spatial_{call_id}")
        flatten_op.set_module(self)
        flatten_shape_tensor = F._from_data(
            f"{self.name}.flatten_spatial.shape_{call_id}",
            data=np.array(flatten_shape, dtype=np.int64),
            is_const=True,
        )
        weights_flat = flatten_op(weights, flatten_shape_tensor)

        # Step 9: Apply softmax
        softmax_op = F.Softmax(f"{self.name}.softmax_{call_id}", axis=-1)
        softmax_op.set_module(self)
        weights_soft = softmax_op(weights_flat)

        # Step 10: Unflatten
        unflatten_shape = [B, Q, nheads, H, W]
        unflatten_op = F.Reshape(f"{self.name}.unflatten_spatial_{call_id}")
        unflatten_op.set_module(self)
        unflatten_shape_tensor = F._from_data(
            f"{self.name}.unflatten_spatial.shape_{call_id}",
            data=np.array(unflatten_shape, dtype=np.int64),
            is_const=True,
        )
        weights_spatial = unflatten_op(weights_soft, unflatten_shape_tensor)

        # Step 11: Apply dropout (FIXED: skip if prob=0)
        if self.dropout is not None:
            weights_out = self.dropout(weights_spatial)
        else:
            # FIX: When dropout prob=0, just pass through (identity)
            weights_out = weights_spatial

        return weights_out

    def analytical_param_count(self, lvl=0):
        """Calculate total parameter count."""
        q_params = self.q_linear_weight.shape[0] * self.q_linear_weight.shape[1]
        if self.q_linear_bias is not None:
            q_params += self.hidden_dim
        k_params = self.hidden_dim * self.k_linear_weight.shape[1] * 1 * 1
        if self.k_linear_bias is not None:
            k_params += self.hidden_dim
        return q_params + k_params


# ============================================================================
# MaskHeadSmallConv: FPN-Based Mask Prediction Head
# ============================================================================


class MaskHeadSmallConv(SimNN.Module):
    """
    Small Convolutional Mask Head with FPN Integration (TTSim-Compatible).
    """

    def __init__(self, name, dim, fpn_dims, context_dim):
        super().__init__()
        self.name = name
        self.dim = dim
        self.fpn_dims = fpn_dims
        self.context_dim = context_dim

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.inter_dims = inter_dims

        self.lay1 = F.Conv2d(
            name + ".lay1", dim, dim, kernel_size=3, padding=1, bias=True
        )
        self.gn1 = SimNN.GroupNorm(name + ".gn1", num_groups=8, num_channels=dim)

        self.lay2 = F.Conv2d(
            name + ".lay2", dim, inter_dims[1], kernel_size=3, padding=1, bias=True
        )
        self.gn2 = SimNN.GroupNorm(
            name + ".gn2", num_groups=8, num_channels=inter_dims[1]
        )

        self.lay3 = F.Conv2d(
            name + ".lay3",
            inter_dims[1],
            inter_dims[2],
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.gn3 = SimNN.GroupNorm(
            name + ".gn3", num_groups=8, num_channels=inter_dims[2]
        )

        self.lay4 = F.Conv2d(
            name + ".lay4",
            inter_dims[2],
            inter_dims[3],
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.gn4 = SimNN.GroupNorm(
            name + ".gn4", num_groups=8, num_channels=inter_dims[3]
        )

        self.lay5 = F.Conv2d(
            name + ".lay5",
            inter_dims[3],
            inter_dims[4],
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.gn5 = SimNN.GroupNorm(
            name + ".gn5", num_groups=8, num_channels=inter_dims[4]
        )

        self.out_lay = F.Conv2d(
            name + ".out_lay", inter_dims[4], 1, kernel_size=3, padding=1, bias=True
        )

        self.adapter1 = F.Conv2d(
            name + ".adapter1",
            fpn_dims[0],
            inter_dims[1],
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.adapter2 = F.Conv2d(
            name + ".adapter2",
            fpn_dims[1],
            inter_dims[2],
            kernel_size=1,
            padding=0,
            bias=True,
        )
        self.adapter3 = F.Conv2d(
            name + ".adapter3",
            fpn_dims[2],
            inter_dims[3],
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.relu1 = F.Relu(name + ".relu1")
        self.relu2 = F.Relu(name + ".relu2")
        self.relu3 = F.Relu(name + ".relu3")
        self.relu4 = F.Relu(name + ".relu4")
        self.relu5 = F.Relu(name + ".relu5")

        super().link_op2module()

    def __call__(self, x, bbox_mask, fpns):
        """Forward pass for mask head."""
        call_id = id(x)

        def expand(tensor, length, tensor_id):
            length_int = int(length)
            B, C, H, W = tensor.shape
            unique_id = f"{call_id}_{tensor_id}"

            unsqueeze_op = F.Unsqueeze(f"{self.name}.expand_unsqueeze_{unique_id}")
            unsqueeze_op.set_module(self)
            axes_tensor = F._from_data(
                f"{self.name}.expand_unsqueeze.axes_{unique_id}",
                data=np.array([1], dtype=np.int64),
                is_const=True,
            )
            tensor_5d = unsqueeze_op(tensor, axes_tensor)

            # FIX: F.Unsqueeze doesn't propagate data — do it manually
            if tensor.data is not None:
                tensor_5d.data = np.expand_dims(tensor.data, axis=1)

            tile_op = F.Tile(f"{self.name}.expand_tile_{unique_id}")
            tile_op.set_module(self)
            repeats_tensor = F._from_data(
                f"{self.name}.expand_tile.repeats_{unique_id}",
                data=np.array([1, length_int, 1, 1, 1], dtype=np.int64),
                is_const=True,
            )
            tensor_tiled = tile_op(tensor_5d, repeats_tensor)

            reshape_op = F.Reshape(f"{self.name}.expand_reshape_{unique_id}")
            reshape_op.set_module(self)
            new_shape = [B * length_int, C, H, W]
            shape_tensor = F._from_data(
                f"{self.name}.expand_reshape.shape_{unique_id}",
                data=np.array(new_shape, dtype=np.int64),
                is_const=True,
            )
            tensor_flat = reshape_op(tensor_tiled, shape_tensor)

            return tensor_flat

        if len(bbox_mask.shape) > 4:
            expand_factor = bbox_mask.shape[1]
        else:
            expand_factor = bbox_mask.shape[0] / x.shape[0]

        x_expanded = expand(x, expand_factor, "x_main")

        if len(bbox_mask.shape) > 4:
            B_mask, Q_mask = bbox_mask.shape[0], bbox_mask.shape[1]
            nheads, H_mask, W_mask = (
                bbox_mask.shape[2],
                bbox_mask.shape[3],
                bbox_mask.shape[4],
            )
            flatten_shape = [B_mask * Q_mask, nheads, H_mask, W_mask]
            flatten_bbox_op = F.Reshape(f"{self.name}.flatten_bbox_mask_{call_id}")
            flatten_bbox_op.set_module(self)
            flatten_shape_tensor = F._from_data(
                f"{self.name}.flatten_bbox_mask.shape_{call_id}",
                data=np.array(flatten_shape, dtype=np.int64),
                is_const=True,
            )
            bbox_mask_flat = flatten_bbox_op(bbox_mask, flatten_shape_tensor)
        else:
            bbox_mask_flat = bbox_mask

        concat_op = F.ConcatX(f"{self.name}.concat_input_{call_id}", axis=1)
        concat_op.set_module(self)
        x_in = concat_op(x_expanded, bbox_mask_flat)

        x1 = self.lay1(x_in)
        x1 = self.gn1(x1)
        x1 = self.relu1(x1)

        x2 = self.lay2(x1)
        x2 = self.gn2(x2)
        x2 = self.relu2(x2)

        cur_fpn0 = self.adapter1(fpns[0])
        if cur_fpn0.shape[0] != x2.shape[0]:
            cur_fpn0 = expand(cur_fpn0, x2.shape[0] / cur_fpn0.shape[0], "fpn0")
        target_size = (cur_fpn0.shape[2], cur_fpn0.shape[3])
        x2_up = interpolate_nearest(
            x2, target_size, module=self, call_id=f"{call_id}_fpn0"
        )
        add_op1 = F.Add(f"{self.name}.add_fpn0_{call_id}")
        add_op1.set_module(self)
        x2 = add_op1(cur_fpn0, x2_up)

        x3 = self.lay3(x2)
        x3 = self.gn3(x3)
        x3 = self.relu3(x3)

        cur_fpn1 = self.adapter2(fpns[1])
        if cur_fpn1.shape[0] != x3.shape[0]:
            cur_fpn1 = expand(cur_fpn1, x3.shape[0] / cur_fpn1.shape[0], "fpn1")
        target_size = (cur_fpn1.shape[2], cur_fpn1.shape[3])
        x3_up = interpolate_nearest(
            x3, target_size, module=self, call_id=f"{call_id}_fpn1"
        )
        add_op2 = F.Add(f"{self.name}.add_fpn1_{call_id}")
        add_op2.set_module(self)
        x3 = add_op2(cur_fpn1, x3_up)

        x4 = self.lay4(x3)
        x4 = self.gn4(x4)
        x4 = self.relu4(x4)

        cur_fpn2 = self.adapter3(fpns[2])
        if cur_fpn2.shape[0] != x4.shape[0]:
            cur_fpn2 = expand(cur_fpn2, x4.shape[0] / cur_fpn2.shape[0], "fpn2")
        target_size = (cur_fpn2.shape[2], cur_fpn2.shape[3])
        x4_up = interpolate_nearest(
            x4, target_size, module=self, call_id=f"{call_id}_fpn2"
        )
        add_op3 = F.Add(f"{self.name}.add_fpn2_{call_id}")
        add_op3.set_module(self)
        x4 = add_op3(cur_fpn2, x4_up)

        x5 = self.lay5(x4)
        x5 = self.gn5(x5)
        x5 = self.relu5(x5)

        out = self.out_lay(x5)

        return out

    def analytical_param_count(self, lvl=0):
        """Calculate total parameter count."""
        param_count = 0
        param_count += (3 * 3) * self.dim * self.dim + self.dim
        param_count += (3 * 3) * self.dim * self.inter_dims[1] + self.inter_dims[1]
        param_count += (3 * 3) * self.inter_dims[1] * self.inter_dims[
            2
        ] + self.inter_dims[2]
        param_count += (3 * 3) * self.inter_dims[2] * self.inter_dims[
            3
        ] + self.inter_dims[3]
        param_count += (3 * 3) * self.inter_dims[3] * self.inter_dims[
            4
        ] + self.inter_dims[4]
        param_count += (3 * 3) * self.inter_dims[4] * 1 + 1
        param_count += self.fpn_dims[0] * self.inter_dims[1] + self.inter_dims[1]
        param_count += self.fpn_dims[1] * self.inter_dims[2] + self.inter_dims[2]
        param_count += self.fpn_dims[2] * self.inter_dims[3] + self.inter_dims[3]
        param_count += 2 * self.dim
        param_count += 2 * self.inter_dims[1]
        param_count += 2 * self.inter_dims[2]
        param_count += 2 * self.inter_dims[3]
        param_count += 2 * self.inter_dims[4]
        return param_count


# ============================================================================
# DETRsegm: Complete Segmentation Model Wrapper
# ============================================================================


class DETRsegm(SimNN.Module):
    """Deformable DETR Segmentation Model Wrapper (TTSim-Compatible)."""

    def __init__(self, name, detr, hidden_dim, nheads, freeze_detr=False):
        super().__init__()
        self.name = name
        self.detr = detr
        self.hidden_dim = hidden_dim
        self.nheads = nheads

        self.bbox_attention = MHAttentionMap(
            name + ".bbox_attention",
            query_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=nheads,
            dropout=0.0,
        )

        self.mask_head = MaskHeadSmallConv(
            name + ".mask_head",
            dim=hidden_dim + nheads,
            fpn_dims=[1024, 512, 256],
            context_dim=hidden_dim,
        )

        super().link_op2module()

    def __call__(self, samples, features, pos, query_embed_weight):
        """Forward pass for segmentation model."""
        call_id = id(samples)

        src = features[-1]
        mask = None

        B, C, H, W = src.shape
        num_queries = (
            query_embed_weight.shape[0] if hasattr(query_embed_weight, "shape") else 100
        )

        hs_last = F._from_shape(
            f"{self.name}.hs_last_placeholder_{call_id}",
            [B, num_queries, self.hidden_dim],
        )

        memory = F._from_shape(
            f"{self.name}.memory_placeholder_{call_id}", [B, self.hidden_dim, H, W]
        )

        src_proj = F._from_shape(
            f"{self.name}.src_proj_placeholder_{call_id}", [B, self.hidden_dim, H, W]
        )

        bbox_mask = self.bbox_attention(hs_last, memory, mask=mask)

        fpn0 = F._from_shape(
            f"{self.name}.fpn0_placeholder_{call_id}", [B, 1024, H // 2, W // 2]
        )
        fpn1 = F._from_shape(
            f"{self.name}.fpn1_placeholder_{call_id}", [B, 512, H // 4, W // 4]
        )
        fpn2 = F._from_shape(
            f"{self.name}.fpn2_placeholder_{call_id}", [B, 256, H // 8, W // 8]
        )
        fpns = [fpn0, fpn1, fpn2]

        seg_masks = self.mask_head(src_proj, bbox_mask, fpns)

        H_out, W_out = seg_masks.shape[-2], seg_masks.shape[-1]
        reshape1_shape = [B, num_queries, 1, H_out, W_out]
        reshape1_op = F.Reshape(f"{self.name}.reshape_masks1_{call_id}")
        reshape1_op.set_module(self)
        reshape1_shape_tensor = F._from_data(
            f"{self.name}.reshape_masks1.shape_{call_id}",
            data=np.array(reshape1_shape, dtype=np.int64),
            is_const=True,
        )
        masks_5d = reshape1_op(seg_masks, reshape1_shape_tensor)

        squeeze_op = F.Squeeze(f"{self.name}.squeeze_masks_{call_id}")
        squeeze_op.set_module(self)
        squeeze_axes_tensor = F._from_data(
            f"{self.name}.squeeze_masks.axes_{call_id}",
            data=np.array([2], dtype=np.int64),
            is_const=True,
        )
        outputs_seg_masks = squeeze_op(masks_5d, squeeze_axes_tensor)

        # FIX: F.Squeeze doesn't propagate data — do it manually
        if masks_5d.data is not None:
            outputs_seg_masks.data = np.squeeze(masks_5d.data, axis=2)

        out = {"pred_masks": outputs_seg_masks}

        return out

    def analytical_param_count(self, lvl=0):
        """Calculate total parameter count."""
        bbox_params = self.bbox_attention.analytical_param_count(lvl)
        mask_params = self.mask_head.analytical_param_count(lvl)
        return bbox_params + mask_params


# ============================================================================
# Post-Processing Modules (Added in Deformable DETR)
# ============================================================================

# Loss functions (dice_loss, sigmoid_focal_loss) are added in Deformable DETR
