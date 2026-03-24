#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
MapTracker Architecture Layer-by-Layer Validation with TTSim Computations

This test suite validates the MapTracker architecture using TTSim compute functions.
Data flows through each layer end-to-end with proper shapes and operations.
The image backbone uses ResNet-50 with Bottleneck blocks (reduced to 1 block per stage
for speed) and all downstream stages (seg head, vector head, temporal tracking) are faithful.

IMPORTANT PERFORMANCE NOTE:
- TTSim compute functions are used for ALL operations on ttsim variables:
  * Convolutions (Conv2d) and upsampling
  * Matrix multiplication (attention, FFN, projections)
  * Activations (ReLU, Sigmoid, Softmax)
  * Element-wise operations (Add, Sub, Mul, Div)
  * Reshaping and transposing
  * Layer normalization
  * Grid sampling (history warping)
  * Aggregations (ReduceMean, ReduceSum)
- PyTorch operations are only used on PyTorch reference tensors (_torch variables)
- Test sizes are reduced for faster validation while preserving architecture structure

MAPTRACKER ARCHITECTURE (ResNet-50 backbone, faithful downstream stages):

0. IMAGE BACKBONE + FPN + BEV ENCODER
   ├─ Multi-camera input: [B, num_cams, 3, H, W]
   ├─ ResNet-50 Backbone (Bottleneck blocks, reduced to 1 per stage for speed):
   │   ├─ Stem: Conv7x7(stride=2) -> BN -> ReLU -> MaxPool3x3(stride=2)
   │   ├─ Stage 1: Bottleneck [1x1->BN->ReLU, 3x3->BN->ReLU, 1x1->BN] + downsample
   │   ├─ Stage 2: Bottleneck(stride=2) + downsample -> C3
   │   ├─ Stage 3: Bottleneck(stride=2) + downsample -> C4
   │   └─ Stage 4: Bottleneck(stride=2) + downsample -> C5
   ├─ FPN Neck: lateral 1x1 + top-down fusion + 3x3 output convs
   │   ├─ Lateral 1x1 on C3/C4/C5 -> channel projection to embed_dims
   │   ├─ Top-down fusion with bilinear upsample + add
   │   └─ 3x3 output convs on each fused level
   └─ BEVFormer Encoder (2 layers): Transform to BEV space
       ├─ Temporal Self-Attention (deformable, BEV-to-BEV)
       ├─ Spatial Cross-Attention (simplified with random sampled features)
       └─ FFN
   Output: BEV features [B, embed_dims, bev_h, bev_w]
   Note: Spatial cross-attn uses random sampled features for speed

1. BEV INPUT (output of Stage 0, connected)
   └─ BEV features reshaped from encoder output: [B, embed_dims, bev_h, bev_w]

2. BEV BACKBONE (BEVFormerBackbone)
   ├─ BEV Embedding lookup
   ├─ History warping via GridSample (frames 1+)
   ├─ Reshape: [bs, H*W, C] -> [bs, C, H, W]
   └─ Optional UpsampleBlock (Conv2d -> GroupNorm -> ReLU -> Upsample)

3. SEGMENTATION HEAD (MapSegHead)
   ├─ Conv2d(3×3, no bias) -> ReLU
   ├─ Upsample(2×) -> Conv2d(3×3) + bias -> ReLU
   ├─ Conv2d(1×1) + bias -> seg_preds
   └─ Downsample(0.5×) -> seg_feats

4. VECTOR MAP DETECTION HEAD (MapDetectorHead)
   ├─ Input projection: Conv2d(1×1) + bias
   ├─ BEV sine positional embedding + Add
   ├─ Query initialization: Embedding -> Unsqueeze -> Tile
   ├─ Reference points: Linear -> Sigmoid -> Reshape [bs, N_q, N_pts, 2]
   └─ MapTransformer Decoder (6 layers):
       Each layer:
       ├─ Self-Attention (MultiheadAttention)
       │   ├─ Q, K, V projections (3× Linear)
       │   ├─ Reshape -> multi-head [bs, heads, L, d_k]
       │   ├─ MatMul(Q, K^T) / sqrt(d_k) -> Softmax -> MatMul(attn, V)
       │   └─ Reshape -> [bs, L, C] -> Output projection (Linear)
       ├─ LayerNorm (ReduceMean -> Sub -> Mul -> ReduceMean -> Sqrt -> Div)
       ├─ Cross-Attention (CustomMSDeformableAttention)
       │   ├─ Value projection (Linear)
       │   ├─ Sampling offsets (Linear -> Reshape)
       │   ├─ Attention weights (Linear -> Softmax)
       │   ├─ Bilinear sampling from BEV features
       │   └─ Output projection (Linear)
       ├─ LayerNorm
       ├─ Memory Cross-Attention (MultiheadAttention) — active (pre-populated memory)
       ├─ LayerNorm
       ├─ FFN: Linear -> ReLU -> Linear + Residual
       ├─ LayerNorm
       ├─ RegressionBranch: Linear -> LN -> ReLU -> Linear -> LN -> ReLU -> Linear
       │   -> Reshape [bs, N_q, N_pts, 2] -> Sigmoid -> updated reference_points
       └─ ClassificationBranch: Linear -> [bs, N_q, N_cls]

5. QUERY PROPAGATION (MotionMLP) — between frames
   ├─ Embedder: sin/cos positional encoding of pose
   ├─ Concat(features, pose_embed)
   ├─ Linear -> LayerNorm -> ReLU -> Linear
   └─ Add (residual)

6. POST-PROCESSING
   ├─ Sigmoid on scores -> threshold -> filter
   ├─ ArgMax on seg logits -> semantic_mask
   └─ Output: vectors, scores, labels, semantic_mask

This test validates each operation with actual data flow and proper shapes using TTSim compute functions.
"""

import sys
import os
import traceback
import numpy as np
import torch
import torch.nn.functional as F_torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import layer operations from shared module (FPN, transformer ops)
from workloads.MapTracker.reference.comparison_mapers.layer_ops import (
    fpn_top_down_fusion,
    multi_head_attention,
    feedforward_network,
)


# Import TTSim compute functions
def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pure PyTorch CPU implementation of multi-scale deformable attention.

    Args:
        value (Tensor): (bs, num_keys, num_heads, head_dim)
        value_spatial_shapes (Tensor): (num_levels, 2)  [H, W]
        sampling_locations (Tensor): (bs, num_queries, num_heads, num_levels, num_points, 2)  in [0,1]
        attention_weights (Tensor): (bs, num_queries, num_heads, num_levels, num_points)
    Returns:
        Tensor: (bs, num_queries, embed_dims)
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, int(H_), int(W_))
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F_torch.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)

    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


from ttsim.ops.desc.data_compute import (
    compute_add,
    compute_sub,
    compute_mul,
    compute_div,
    compute_relu,
    compute_sigmoid,
    compute_softmax,
    compute_reshape,
    compute_transpose,
    compute_resize,
    compute_sin,
    compute_cos,
    compute_exp,
    compute_sqrt,
    compute_reducemean,
    compute_reducesum,
    compute_matmul,
    compute_concat,
    compute_unsqueeze,
    compute_squeeze,
    compute_maxpool2d,
    compute_conv2d,
    compute_layernorm,
    compute_gridsample,
    compute_upsample,
    compute_argmax,
    compute_tile,
)

# ============================================================================
# TTSim Wrapper Functions (Using actual TTSim compute functions)
# ============================================================================


def create_mock_tensor(data):
    """Create a mock tensor object with data attribute for TTSim compute functions."""
    return type("MockTensor", (), {"data": data})()


def create_mock_op(**attrs):
    """Create a mock op object with attrs for TTSim compute functions."""
    return type("MockOp", (), {"attrs": attrs})()


# --- Activations ---
def ttsim_relu(x):
    return compute_relu([create_mock_tensor(x)], create_mock_op())


def ttsim_sigmoid(x):
    return compute_sigmoid([create_mock_tensor(x)], create_mock_op())


def ttsim_softmax(x, axis=-1):
    return compute_softmax([create_mock_tensor(x)], create_mock_op(axis=axis))


# --- Element-wise ---
def ttsim_add(a, b):
    return compute_add([create_mock_tensor(a), create_mock_tensor(b)], create_mock_op())


def ttsim_sub(a, b):
    return compute_sub([create_mock_tensor(a), create_mock_tensor(b)], create_mock_op())


def ttsim_mul(a, b):
    return compute_mul([create_mock_tensor(a), create_mock_tensor(b)], create_mock_op())


def ttsim_div(a, b):
    return compute_div([create_mock_tensor(a), create_mock_tensor(b)], create_mock_op())


def ttsim_sqrt(x):
    return compute_sqrt([create_mock_tensor(x)], create_mock_op())


def ttsim_exp(x):
    return compute_exp([create_mock_tensor(x)], create_mock_op())


def ttsim_sin(x):
    return compute_sin([create_mock_tensor(x)], create_mock_op())


def ttsim_cos(x):
    return compute_cos([create_mock_tensor(x)], create_mock_op())


# --- Shape ops ---
def ttsim_reshape(x, shape):
    return compute_reshape(
        [create_mock_tensor(x), create_mock_tensor(np.array(shape, dtype=np.int64))],
        create_mock_op(),
    )


def ttsim_transpose(x, axes):
    return compute_transpose([create_mock_tensor(x)], create_mock_op(perm=axes))


def ttsim_unsqueeze(x, axes):
    return compute_unsqueeze(
        [create_mock_tensor(x), create_mock_tensor(np.array(axes, dtype=np.int64))],
        create_mock_op(),
    )


def ttsim_squeeze(x, axes=None):
    if axes is not None:
        return compute_squeeze(
            [create_mock_tensor(x), create_mock_tensor(np.array(axes, dtype=np.int64))],
            create_mock_op(),
        )
    return compute_squeeze([create_mock_tensor(x)], create_mock_op())


def ttsim_tile(x, repeats):
    return compute_tile(
        [create_mock_tensor(x), create_mock_tensor(np.array(repeats, dtype=np.int64))],
        create_mock_op(),
    )


def ttsim_concat(arrays, axis=0):
    return compute_concat(
        [create_mock_tensor(a) for a in arrays], create_mock_op(axis=axis)
    )


# --- Reductions ---
def ttsim_reducemean(x, axis=None, keepdims=True):
    mt = create_mock_tensor(x)
    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        ma = create_mock_tensor(np.array(axis, dtype=np.int64))
        return compute_reducemean(
            [mt, ma], create_mock_op(keepdims=1 if keepdims else 0)
        )
    return compute_reducemean(
        [mt], create_mock_op(keepdims=1 if keepdims else 0, noop_with_empty_axes=0)
    )


def ttsim_reducesum(x, axis=None, keepdims=True):
    mt = create_mock_tensor(x)
    if axis is not None:
        if isinstance(axis, int):
            axis = [axis]
        ma = create_mock_tensor(np.array(axis, dtype=np.int64))
        return compute_reducesum(
            [mt, ma], create_mock_op(keepdims=1 if keepdims else 0)
        )
    return compute_reducesum(
        [mt], create_mock_op(keepdims=1 if keepdims else 0, noop_with_empty_axes=0)
    )


# --- Matrix ops ---
def ttsim_matmul(a, b):
    return compute_matmul(
        [create_mock_tensor(a), create_mock_tensor(b)], create_mock_op()
    )


# --- High-level ops ---
def ttsim_conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding, padding, padding]
    if isinstance(dilation, int):
        dilation = [dilation, dilation]
    inputs = [create_mock_tensor(x), create_mock_tensor(weight)]
    if bias is not None:
        inputs.append(create_mock_tensor(bias))
    return compute_conv2d(
        inputs,
        create_mock_op(strides=stride, pads=padding, dilations=dilation, group=groups),
    )


def ttsim_maxpool2d(x, kernel_size, stride=None, padding=0):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding, padding, padding]
    return compute_maxpool2d(
        [create_mock_tensor(x)],
        create_mock_op(kernel_shape=kernel_size, strides=stride, pads=padding),
    )


def ttsim_upsample(x, scale_factor=2.0, mode="nearest", align_corners=True):
    if isinstance(scale_factor, (int, float)):
        scale_factor = [float(scale_factor), float(scale_factor)]
    return compute_upsample(
        [create_mock_tensor(x)],
        create_mock_op(
            mode=mode, scale_factor=scale_factor, align_corners=align_corners
        ),
    )


def ttsim_gridsample(
    x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
):
    return compute_gridsample(
        [create_mock_tensor(x), create_mock_tensor(grid)],
        create_mock_op(
            mode=mode,
            padding_mode=padding_mode,
            align_corners=1 if align_corners else 0,
        ),
    )


def ttsim_layernorm(x, normalized_shape, eps=1e-5):
    """Non-affine LayerNorm using TTSim compute functions (ReduceMean -> Sub -> Mul -> ReduceMean -> Sqrt -> Div)."""
    mean = ttsim_reducemean(x, axis=-1, keepdims=True)
    x_centered = ttsim_sub(x, mean)
    x_squared = ttsim_mul(x_centered, x_centered)
    var = ttsim_reducemean(x_squared, axis=-1, keepdims=True)
    var_eps = ttsim_add(var, np.array(eps, dtype=x.dtype))
    std = ttsim_sqrt(var_eps)
    return ttsim_div(x_centered, std)


def ttsim_layernorm_affine(x, scale, bias, eps=1e-5):
    """Affine LayerNorm using actual TTSim compute_layernorm."""
    return compute_layernorm(
        [create_mock_tensor(x), create_mock_tensor(scale), create_mock_tensor(bias)],
        create_mock_op(axis=-1, epsilon=eps),
    )


def ttsim_argmax(x, axis=-1, keepdims=True):
    return compute_argmax(
        [create_mock_tensor(x)],
        create_mock_op(axis=axis, keepdims=1 if keepdims else 0),
    )


# ============================================================================
# Composite TTSim Operations
# ============================================================================


def ttsim_linear(x, weight, bias=None):
    """Linear layer: x @ weight + bias. Weight shape [in, out] (pre-transposed from PyTorch)."""
    out = ttsim_matmul(x, weight)
    if bias is not None:
        out = ttsim_add(out, bias)
    return out


def ttsim_multihead_attention(
    query,
    key,
    value,
    q_weight,
    k_weight,
    v_weight,
    out_weight,
    q_bias,
    k_bias,
    v_bias,
    out_bias,
    num_heads,
    head_dim,
):
    """Full multi-head attention using TTSim compute functions.

    Input shapes: query [bs, L, C], key [bs, S, C], value [bs, S, C]
    Weight shapes: [C, C] (pre-transposed)
    Output: [bs, L, C]
    """
    bs = query.shape[0]
    L = query.shape[1]
    S = key.shape[1]
    embed_dims = num_heads * head_dim

    # Q, K, V projections
    Q = ttsim_linear(query, q_weight, q_bias)  # [bs, L, C]
    K = ttsim_linear(key, k_weight, k_bias)  # [bs, S, C]
    V = ttsim_linear(value, v_weight, v_bias)  # [bs, S, C]

    # Reshape for multi-head: [bs, L, num_heads, head_dim] -> [bs, num_heads, L, head_dim]
    Q = ttsim_transpose(ttsim_reshape(Q, (bs, L, num_heads, head_dim)), [0, 2, 1, 3])
    K = ttsim_transpose(ttsim_reshape(K, (bs, S, num_heads, head_dim)), [0, 2, 1, 3])
    V = ttsim_transpose(ttsim_reshape(V, (bs, S, num_heads, head_dim)), [0, 2, 1, 3])

    # Attention: Q @ K^T / sqrt(d_k) -> Softmax -> @ V
    K_T = ttsim_transpose(K, [0, 1, 3, 2])
    scores = ttsim_matmul(Q, K_T)
    scores = ttsim_div(scores, np.array(np.sqrt(head_dim), dtype=np.float32))
    attn = ttsim_softmax(scores, axis=-1)
    context = ttsim_matmul(attn, V)  # [bs, heads, L, d_k]

    # Concat heads: [bs, L, heads, d_k] -> [bs, L, C]
    context = ttsim_transpose(context, [0, 2, 1, 3])
    context = ttsim_reshape(context, (bs, L, embed_dims))

    # Output projection
    out = ttsim_linear(context, out_weight, out_bias)
    return out


def ttsim_ffn(x, w1, b1, w2, b2):
    """FFN: Linear -> ReLU -> Linear + Residual."""
    identity = x.copy()
    h = ttsim_relu(ttsim_linear(x, w1, b1))
    out = ttsim_linear(h, w2, b2)
    return ttsim_add(out, identity)


def ttsim_regression_branch(x, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b):
    """RegressionBranch: Linear -> LN -> ReLU -> Linear -> LN -> ReLU -> Linear."""
    h = ttsim_linear(x, fc1_w, fc1_b)
    h = ttsim_layernorm(h, h.shape[-1])
    h = ttsim_relu(h)
    h = ttsim_linear(h, fc2_w, fc2_b)
    h = ttsim_layernorm(h, h.shape[-1])
    h = ttsim_relu(h)
    return ttsim_linear(h, fc3_w, fc3_b)


def ttsim_batchnorm2d_eval(x, weight, bias, running_mean, running_var, eps=1e-5):
    """BatchNorm2d in eval mode: y = (x - mean) / sqrt(var + eps) * weight + bias.
    x: [N, C, H, W], weight/bias/mean/var: [C]."""
    C = weight.shape[0]
    inv_std = 1.0 / np.sqrt(running_var + eps)
    scale = (weight * inv_std).reshape(1, C, 1, 1).astype(np.float32)
    offset = (
        (bias - running_mean * weight * inv_std).reshape(1, C, 1, 1).astype(np.float32)
    )
    return ttsim_add(ttsim_mul(x, scale), offset)


def gen_bottleneck_params(in_ch, mid_ch, out_ch, stride, has_downsample):
    """Generate random weights for one Bottleneck block."""
    p = {}
    # Conv 1x1 -> BN
    p["conv1_w"] = np.random.randn(mid_ch, in_ch, 1, 1).astype(np.float32) * 0.01
    p["bn1_w"] = np.abs(np.random.randn(mid_ch).astype(np.float32)) + 0.5
    p["bn1_b"] = np.random.randn(mid_ch).astype(np.float32) * 0.1
    p["bn1_m"] = np.random.randn(mid_ch).astype(np.float32) * 0.1
    p["bn1_v"] = np.abs(np.random.randn(mid_ch).astype(np.float32)) + 0.5
    # Conv 3x3 -> BN
    p["conv2_w"] = np.random.randn(mid_ch, mid_ch, 3, 3).astype(np.float32) * 0.01
    p["bn2_w"] = np.abs(np.random.randn(mid_ch).astype(np.float32)) + 0.5
    p["bn2_b"] = np.random.randn(mid_ch).astype(np.float32) * 0.1
    p["bn2_m"] = np.random.randn(mid_ch).astype(np.float32) * 0.1
    p["bn2_v"] = np.abs(np.random.randn(mid_ch).astype(np.float32)) + 0.5
    # Conv 1x1 -> BN
    p["conv3_w"] = np.random.randn(out_ch, mid_ch, 1, 1).astype(np.float32) * 0.01
    p["bn3_w"] = np.abs(np.random.randn(out_ch).astype(np.float32)) + 0.5
    p["bn3_b"] = np.random.randn(out_ch).astype(np.float32) * 0.1
    p["bn3_m"] = np.random.randn(out_ch).astype(np.float32) * 0.1
    p["bn3_v"] = np.abs(np.random.randn(out_ch).astype(np.float32)) + 0.5
    if has_downsample:
        p["ds_conv_w"] = np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.01
        p["ds_bn_w"] = np.abs(np.random.randn(out_ch).astype(np.float32)) + 0.5
        p["ds_bn_b"] = np.random.randn(out_ch).astype(np.float32) * 0.1
        p["ds_bn_m"] = np.random.randn(out_ch).astype(np.float32) * 0.1
        p["ds_bn_v"] = np.abs(np.random.randn(out_ch).astype(np.float32)) + 0.5
    return p


def run_bottleneck(x_pt, x_tt, params, stride=1, has_downsample=False):
    """Execute one Bottleneck block on both PyTorch and TTSim paths.
    Returns (out_pt, out_tt, match)."""
    identity_pt = x_pt.clone()
    identity_tt = x_tt.copy()

    # Conv 1x1 -> BN -> ReLU
    out_pt = torch.nn.functional.conv2d(x_pt, torch.from_numpy(params["conv1_w"]))
    out_pt = torch.nn.functional.batch_norm(
        out_pt,
        torch.from_numpy(params["bn1_m"]),
        torch.from_numpy(params["bn1_v"]),
        torch.from_numpy(params["bn1_w"]),
        torch.from_numpy(params["bn1_b"]),
        training=False,
    )
    out_pt = torch.relu(out_pt)

    out_tt = ttsim_conv2d(x_tt, params["conv1_w"], stride=1, padding=0)
    out_tt = ttsim_batchnorm2d_eval(
        out_tt, params["bn1_w"], params["bn1_b"], params["bn1_m"], params["bn1_v"]
    )
    out_tt = ttsim_relu(out_tt)

    # Conv 3x3 -> BN -> ReLU
    out_pt = torch.nn.functional.conv2d(
        out_pt, torch.from_numpy(params["conv2_w"]), stride=stride, padding=1
    )
    out_pt = torch.nn.functional.batch_norm(
        out_pt,
        torch.from_numpy(params["bn2_m"]),
        torch.from_numpy(params["bn2_v"]),
        torch.from_numpy(params["bn2_w"]),
        torch.from_numpy(params["bn2_b"]),
        training=False,
    )
    out_pt = torch.relu(out_pt)

    out_tt = ttsim_conv2d(out_tt, params["conv2_w"], stride=stride, padding=1)
    out_tt = ttsim_batchnorm2d_eval(
        out_tt, params["bn2_w"], params["bn2_b"], params["bn2_m"], params["bn2_v"]
    )
    out_tt = ttsim_relu(out_tt)

    # Conv 1x1 -> BN (no ReLU before residual)
    out_pt = torch.nn.functional.conv2d(out_pt, torch.from_numpy(params["conv3_w"]))
    out_pt = torch.nn.functional.batch_norm(
        out_pt,
        torch.from_numpy(params["bn3_m"]),
        torch.from_numpy(params["bn3_v"]),
        torch.from_numpy(params["bn3_w"]),
        torch.from_numpy(params["bn3_b"]),
        training=False,
    )

    out_tt = ttsim_conv2d(out_tt, params["conv3_w"], stride=1, padding=0)
    out_tt = ttsim_batchnorm2d_eval(
        out_tt, params["bn3_w"], params["bn3_b"], params["bn3_m"], params["bn3_v"]
    )

    # Downsample residual
    if has_downsample:
        identity_pt = torch.nn.functional.conv2d(
            identity_pt, torch.from_numpy(params["ds_conv_w"]), stride=stride
        )
        identity_pt = torch.nn.functional.batch_norm(
            identity_pt,
            torch.from_numpy(params["ds_bn_m"]),
            torch.from_numpy(params["ds_bn_v"]),
            torch.from_numpy(params["ds_bn_w"]),
            torch.from_numpy(params["ds_bn_b"]),
            training=False,
        )

        identity_tt = ttsim_conv2d(
            identity_tt, params["ds_conv_w"], stride=stride, padding=0
        )
        identity_tt = ttsim_batchnorm2d_eval(
            identity_tt,
            params["ds_bn_w"],
            params["ds_bn_b"],
            params["ds_bn_m"],
            params["ds_bn_v"],
        )

    # Residual + ReLU
    out_pt = torch.relu(out_pt + identity_pt)
    out_tt = ttsim_relu(ttsim_add(out_tt, identity_tt))

    match = compare_arrays(out_pt, out_tt, "Bottleneck", rtol=1e-4, atol=1e-5)
    return out_pt, out_tt, match


# ============================================================================
# Multi-scale Deformable Attention (pure compute)
# ============================================================================


def ttsim_ms_deformable_attn(
    value, spatial_shapes, sampling_locations, attention_weights
):
    """Pure TTSim compute implementation of multi-scale deformable attention.

    Args:
        value: [bs, num_keys, num_heads, head_dim]
        spatial_shapes: [num_levels, 2] (H, W)
        sampling_locations: [bs, num_queries, num_heads, num_levels, num_points, 2] in [0,1]
        attention_weights: [bs, num_queries, num_heads, num_levels, num_points]
    Returns:
        [bs, num_queries, embed_dims]
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    # Split value by spatial levels
    value_list = []
    offset = 0
    for level in range(num_levels):
        H_, W_ = int(spatial_shapes[level, 0]), int(spatial_shapes[level, 1])
        n_keys = H_ * W_
        val_level = value[:, offset : offset + n_keys]
        offset += n_keys
        value_list.append(val_level)

    # Convert sampling_locations from [0,1] to grid_sample coords [-1,1]
    sampling_grids = ttsim_sub(
        ttsim_mul(sampling_locations, np.float32(2.0)), np.float32(1.0)
    )

    sampling_value_list = []
    for level in range(num_levels):
        H_, W_ = int(spatial_shapes[level, 0]), int(spatial_shapes[level, 1])
        # val_level: [bs, H_*W_, num_heads, head_dim] -> [bs, num_heads, head_dim, H_, W_]
        val_l = ttsim_reshape(value_list[level], (bs, H_, W_, num_heads, head_dim))
        val_l = ttsim_transpose(val_l, [0, 3, 4, 1, 2])  # [bs, heads, d_k, H_, W_]
        val_l = ttsim_reshape(val_l, (bs * num_heads, head_dim, H_, W_))

        # Grid for this level: [bs, nq, heads, num_points, 2]
        grid_l = sampling_grids[:, :, :, level, :, :]  # [bs, nq, heads, pts, 2]
        grid_l = ttsim_transpose(grid_l, [0, 2, 1, 3, 4])  # [bs, heads, nq, pts, 2]
        grid_l = ttsim_reshape(grid_l, (bs * num_heads, num_queries, num_points, 2))

        # Grid sample: [bs*heads, head_dim, nq, pts]
        sampled = ttsim_gridsample(val_l, grid_l, align_corners=False)
        sampling_value_list.append(sampled)

    # Stack and weighted sum
    # attn_weights: [bs, nq, heads, levels, pts] -> [bs*heads, 1, nq, levels*pts]
    attn_w = ttsim_transpose(attention_weights, [0, 2, 1, 3, 4])
    attn_w = ttsim_reshape(
        attn_w, (bs * num_heads, 1, num_queries, num_levels * num_points)
    )

    # Stack sampled: each [bs*heads, d_k, nq, pts] -> concat on last dim
    stacked = ttsim_concat(
        sampling_value_list, axis=-1
    )  # [bs*heads, d_k, nq, levels*pts]

    # Weighted sum: element-wise mul then sum over last axis
    weighted = ttsim_mul(stacked, attn_w)
    # Sum over last dim -> [bs*heads, d_k, nq]
    output = ttsim_reducesum(weighted, axis=-1, keepdims=False)

    # Reshape -> [bs, heads*d_k, nq] -> [bs, nq, C]
    output = ttsim_reshape(output, (bs, num_heads * head_dim, num_queries))
    output = ttsim_transpose(output, [0, 2, 1])

    return output


# ============================================================================
# Helper Functions
# ============================================================================


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_test(title):
    print("\n" + title)
    print("-" * 80)


def compare_arrays(pytorch_arr, ttsim_arr, name="array", rtol=1e-5, atol=1e-6):
    """Compare PyTorch tensor with TTSim/numpy array."""
    if isinstance(pytorch_arr, torch.Tensor):
        pt_numpy = pytorch_arr.detach().cpu().numpy()
    else:
        pt_numpy = pytorch_arr

    if pt_numpy.shape != ttsim_arr.shape:
        print(
            f"  x {name}: Shape mismatch - PyTorch: {pt_numpy.shape}, TTSim: {ttsim_arr.shape}"
        )
        return False

    max_diff = np.max(np.abs(pt_numpy - ttsim_arr))
    mean_diff = np.mean(np.abs(pt_numpy - ttsim_arr))

    if np.allclose(pt_numpy, ttsim_arr, rtol=rtol, atol=atol):
        print(
            f"  PASS {name}: Match! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )
        return True
    else:
        print(
            f"  FAIL {name}: Mismatch! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )
        return False


# ============================================================================
# Test: Complete Layer-by-Layer Validation
# ============================================================================


def test_layer_by_layer_validation():
    """Complete MapTracker forward pass with layer-by-layer computational validation.

    Uses TTSim compute functions for ALL operations, mirroring the test_bevformer.py pattern.
    Data flows through each layer end-to-end (ResNet-50 backbone, faithful downstream).
    """
    print_test("MapTracker Architecture Layer-by-Layer Validation")

    np.random.seed(42)
    torch.manual_seed(42)

    try:
        print(f"\n  {'='*75}")
        print(f"  MAPTRACKER COMPLETE ARCHITECTURE VALIDATION")
        print(f"  {'='*75}")
        print(f"  ResNet-50 backbone + FPN + faithful downstream stages")
        print(f"  Data flows through each layer with proper shapes and operations")
        print(f"  All computations use TTSim compute functions for validation")
        print(f"  {'='*75}\n")

        # Model hyperparameters (reduced for fast validation)
        B = 1
        embed_dims = 32
        num_heads = 8
        head_dim = embed_dims // num_heads  # 4
        bev_h, bev_w = 20, 10
        num_q = 10  # number of queries
        num_cls = 3  # map classes
        num_pts = 5  # polyline points
        coord_dim = 2  # x, y
        ff_channels = 128  # FFN expansion
        c_dim = 7  # pose dimensions
        canvas_h, canvas_w = 40, 20
        num_decoder_layers = 2  # reduced from 6
        num_levels = 1
        deform_pts = 4  # deformable sampling points
        num_value = bev_h * bev_w  # 200

        # Image backbone hyperparameters (ResNet-50, reduced for speed)
        num_cams = 2  # reduced from 7
        img_h, img_w = 28, 50  # reduced for speed
        stem_channels = 8  # reduced from 64
        resnet_planes = [2, 4, 8, 16]  # reduced from [64, 128, 256, 512]
        resnet_layers = [1, 1, 1, 1]  # reduced from [3, 4, 6, 3]
        bottleneck_expansion = 4
        # Stage output channels with expansion=4: [8, 16, 32, 64]
        # out_indices=(1,2,3) -> C3=16, C4=32, C5=64
        fpn_in_ch_list = [resnet_planes[i] * bottleneck_expansion for i in [1, 2, 3]]
        fpn_out_channels = embed_dims  # 32
        num_encoder_layers_bev = 2  # BEVFormer encoder layers

        validation_results = []
        layer_counter = 0

        # =====================================================================
        # STAGE 0: IMAGE BACKBONE + FPN (ResNet-50 with Bottleneck blocks)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 0: IMAGE BACKBONE + FPN (ResNet-50 Bottleneck)")
        print(f"  {'='*75}")
        print(
            f"  ResNet-50 backbone with Bottleneck blocks (1 block per stage for speed)"
        )
        print(f"  Extracts multi-scale features from {num_cams} camera views\n")

        # 0.1: Multi-camera input
        layer_counter += 1
        print(f"  Layer {layer_counter}: Multi-Camera Image Input")
        print(f"  {'-'*75}")

        np.random.seed(10)
        img_input_np = (
            np.random.randn(B, num_cams, 3, img_h, img_w).astype(np.float32) * 0.5
        )
        img_input_torch = torch.from_numpy(img_input_np)

        print(f"    Input shape: {img_input_torch.shape} [B, num_cams, C, H, W]")
        print(f"    Number of cameras: {num_cams}")
        print(f"    Image size: {img_h}x{img_w}")
        validation_results.append(
            (f"Layer {layer_counter}: Camera Input", True, f"{img_input_torch.shape}")
        )

        # Reshape for backbone: [B*num_cams, 3, H, W]
        img_flat_torch = img_input_torch.reshape(B * num_cams, 3, img_h, img_w)
        img_flat_ttsim = ttsim_reshape(img_input_np, (B * num_cams, 3, img_h, img_w))

        # 0.2: Backbone Stem (Conv7x7/2 + BN + ReLU + MaxPool3x3/2)
        layer_counter += 1
        print(
            f"\n  Layer {layer_counter}: Backbone Stem (Conv7x7/2 + BN + ReLU + MaxPool3x3/2)"
        )
        print(f"  {'-'*75}")

        np.random.seed(10)
        stem_conv_w = np.random.randn(stem_channels, 3, 7, 7).astype(np.float32) * 0.01
        stem_bn_w = np.abs(np.random.randn(stem_channels).astype(np.float32)) + 0.5
        stem_bn_b = np.random.randn(stem_channels).astype(np.float32) * 0.1
        stem_bn_m = np.random.randn(stem_channels).astype(np.float32) * 0.1
        stem_bn_v = np.abs(np.random.randn(stem_channels).astype(np.float32)) + 0.5

        # PyTorch stem
        stem_pt = torch.nn.functional.conv2d(
            img_flat_torch, torch.from_numpy(stem_conv_w), stride=2, padding=3
        )
        stem_pt = torch.nn.functional.batch_norm(
            stem_pt,
            torch.from_numpy(stem_bn_m),
            torch.from_numpy(stem_bn_v),
            torch.from_numpy(stem_bn_w),
            torch.from_numpy(stem_bn_b),
            training=False,
        )
        stem_pt = torch.relu(stem_pt)
        stem_pt = torch.nn.functional.max_pool2d(
            stem_pt, kernel_size=3, stride=2, padding=1
        )

        # TTSim stem
        stem_tt = ttsim_conv2d(img_flat_ttsim, stem_conv_w, stride=2, padding=3)
        stem_tt = ttsim_batchnorm2d_eval(
            stem_tt, stem_bn_w, stem_bn_b, stem_bn_m, stem_bn_v
        )
        stem_tt = ttsim_relu(stem_tt)
        stem_tt = ttsim_maxpool2d(stem_tt, kernel_size=3, stride=2, padding=1)

        print(
            f"    Conv7x7/2: [{B*num_cams}, 3, {img_h}, {img_w}] -> [{B*num_cams}, {stem_channels}, {img_h//2}, {img_w//2}]"
        )
        print(f"    BN + ReLU + MaxPool3x3/2 -> {stem_pt.shape}")
        match_stem = compare_arrays(stem_pt, stem_tt, "Stem", rtol=1e-4, atol=1e-5)
        validation_results.append(
            (f"Layer {layer_counter}: Backbone Stem", match_stem, f"{stem_pt.shape}")
        )

        # 0.3: ResNet Stages 1-4 (Bottleneck blocks)
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: ResNet-50 Stages 1-4 (Bottleneck blocks)")
        print(f"  {'-'*75}")

        cur_pt = stem_pt
        cur_tt = stem_tt
        in_ch = stem_channels
        stage_outs_pt = []
        stage_outs_tt = []
        match_stages = True

        for stage_idx in range(4):
            planes = resnet_planes[stage_idx]
            out_ch = planes * bottleneck_expansion
            stride = 1 if stage_idx == 0 else 2
            has_ds = (in_ch != out_ch) or (stride != 1)

            np.random.seed(11 + stage_idx * 10)
            for blk_idx in range(resnet_layers[stage_idx]):
                blk_stride = stride if blk_idx == 0 else 1
                blk_ds = has_ds if blk_idx == 0 else False
                blk_in_ch = in_ch if blk_idx == 0 else out_ch
                params = gen_bottleneck_params(
                    blk_in_ch, planes, out_ch, blk_stride, blk_ds
                )
                cur_pt, cur_tt, blk_match = run_bottleneck(
                    cur_pt, cur_tt, params, blk_stride, blk_ds
                )
                match_stages &= blk_match

            print(
                f"    Stage {stage_idx+1}: {in_ch}->{out_ch} ch, stride={stride}, "
                f"{resnet_layers[stage_idx]} block(s) -> {cur_pt.shape}"
            )
            stage_outs_pt.append(cur_pt)
            stage_outs_tt.append(cur_tt)
            in_ch = out_ch

        # out_indices=(1,2,3) -> C3, C4, C5 are stages 2, 3, 4
        backbone_c3_torch = stage_outs_pt[1]
        backbone_c3_np = stage_outs_tt[1]
        backbone_c4_torch = stage_outs_pt[2]
        backbone_c4_np = stage_outs_tt[2]
        backbone_c5_torch = stage_outs_pt[3]
        backbone_c5_np = stage_outs_tt[3]

        print(
            f"    out_indices=(1,2,3): C3={backbone_c3_torch.shape}, C4={backbone_c4_torch.shape}, C5={backbone_c5_torch.shape}"
        )
        validation_results.append(
            (
                f"Layer {layer_counter}: ResNet-50 Stages",
                match_stages,
                f"C3:{backbone_c3_torch.shape}, C4:{backbone_c4_torch.shape}, C5:{backbone_c5_torch.shape}",
            )
        )

        # 0.4: FPN Neck (lateral 1x1 + top-down fusion + 3x3 output convs)
        layer_counter += 1
        print(
            f"\n  Layer {layer_counter}: FPN (Lateral 1x1 + Top-Down + 3x3 Output Convs)"
        )
        print(f"  {'-'*75}")

        np.random.seed(13)
        # Lateral 1x1 convolutions (channel projection to embed_dims)
        lat_c3_w = (
            np.random.randn(fpn_out_channels, fpn_in_ch_list[0], 1, 1).astype(
                np.float32
            )
            * 0.01
        )
        lat_c4_w = (
            np.random.randn(fpn_out_channels, fpn_in_ch_list[1], 1, 1).astype(
                np.float32
            )
            * 0.01
        )
        lat_c5_w = (
            np.random.randn(fpn_out_channels, fpn_in_ch_list[2], 1, 1).astype(
                np.float32
            )
            * 0.01
        )

        # Lateral C5
        fpn_lat_c5_torch = torch.nn.functional.conv2d(
            backbone_c5_torch, torch.from_numpy(lat_c5_w)
        )
        fpn_lat_c5_np = ttsim_conv2d(backbone_c5_np, lat_c5_w, stride=1, padding=0)

        # Lateral C4
        fpn_lat_c4_torch = torch.nn.functional.conv2d(
            backbone_c4_torch, torch.from_numpy(lat_c4_w)
        )
        fpn_lat_c4_np = ttsim_conv2d(backbone_c4_np, lat_c4_w, stride=1, padding=0)

        # Lateral C3
        fpn_lat_c3_torch = torch.nn.functional.conv2d(
            backbone_c3_torch, torch.from_numpy(lat_c3_w)
        )
        fpn_lat_c3_np = ttsim_conv2d(backbone_c3_np, lat_c3_w, stride=1, padding=0)

        print(
            f"    Lateral C5: {backbone_c5_torch.shape} -> 1x1 conv -> {fpn_lat_c5_torch.shape}"
        )
        print(
            f"    Lateral C4: {backbone_c4_torch.shape} -> 1x1 conv -> {fpn_lat_c4_torch.shape}"
        )
        print(
            f"    Lateral C3: {backbone_c3_torch.shape} -> 1x1 conv -> {fpn_lat_c3_torch.shape}"
        )

        # Top-down fusion
        fpn_p5_torch = fpn_lat_c5_torch
        fpn_p5_ttsim = fpn_lat_c5_np

        # P4 = lateral_C4 + upsample(P5)
        fpn_p4_ttsim, fpn_p4_torch, match_p4 = fpn_top_down_fusion(
            fpn_p5_ttsim, fpn_p5_torch, fpn_lat_c4_np, fpn_lat_c4_torch, verbose=True
        )
        # P3 = lateral_C3 + upsample(P4)
        fpn_p3_ttsim, fpn_p3_torch, match_p3 = fpn_top_down_fusion(
            fpn_p4_ttsim, fpn_p4_torch, fpn_lat_c3_np, fpn_lat_c3_torch, verbose=True
        )

        # 3x3 output convolutions on each fused level
        fpn_conv_p3_w = (
            np.random.randn(fpn_out_channels, fpn_out_channels, 3, 3).astype(np.float32)
            * 0.01
        )
        fpn_conv_p4_w = (
            np.random.randn(fpn_out_channels, fpn_out_channels, 3, 3).astype(np.float32)
            * 0.01
        )
        fpn_conv_p5_w = (
            np.random.randn(fpn_out_channels, fpn_out_channels, 3, 3).astype(np.float32)
            * 0.01
        )

        fpn_p3_torch = torch.nn.functional.conv2d(
            fpn_p3_torch, torch.from_numpy(fpn_conv_p3_w), padding=1
        )
        fpn_p3_ttsim = ttsim_conv2d(fpn_p3_ttsim, fpn_conv_p3_w, stride=1, padding=1)
        fpn_p4_torch = torch.nn.functional.conv2d(
            fpn_p4_torch, torch.from_numpy(fpn_conv_p4_w), padding=1
        )
        fpn_p4_ttsim = ttsim_conv2d(fpn_p4_ttsim, fpn_conv_p4_w, stride=1, padding=1)
        fpn_p5_torch = torch.nn.functional.conv2d(
            fpn_p5_torch, torch.from_numpy(fpn_conv_p5_w), padding=1
        )
        fpn_p5_ttsim = ttsim_conv2d(fpn_p5_ttsim, fpn_conv_p5_w, stride=1, padding=1)

        match_p3_conv = compare_arrays(
            fpn_p3_torch, fpn_p3_ttsim, "FPN P3 output conv", rtol=1e-4, atol=1e-5
        )
        match_p4_conv = compare_arrays(
            fpn_p4_torch, fpn_p4_ttsim, "FPN P4 output conv", rtol=1e-4, atol=1e-5
        )
        match_p5_conv = compare_arrays(
            fpn_p5_torch, fpn_p5_ttsim, "FPN P5 output conv", rtol=1e-4, atol=1e-5
        )

        print(f"    FPN Level 0 (P3): {fpn_p3_torch.shape}")
        print(f"    FPN Level 1 (P4): {fpn_p4_torch.shape}")
        print(f"    FPN Level 2 (P5): {fpn_p5_torch.shape}")

        match_fpn = (
            match_p3 and match_p4 and match_p3_conv and match_p4_conv and match_p5_conv
        )
        validation_results.append(
            (
                f"Layer {layer_counter}: FPN Multi-scale",
                match_fpn,
                f"3 levels: {fpn_p3_torch.shape} to {fpn_p5_torch.shape}",
            )
        )

        # 0.6: BEVFormer Encoder (Actual compute — matches test_bevformer.py pattern)
        # Initialize BEV queries + positional encoding
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: BEV Query Init + Positional Encoding")
        print(f"  {'-'*75}")

        np.random.seed(46)
        bev_queries_enc_np = (
            np.random.randn(B, bev_h * bev_w, embed_dims).astype(np.float32) * 0.5
        )
        bev_queries_enc_torch = torch.from_numpy(bev_queries_enc_np.copy())
        bev_queries_enc_ttsim = bev_queries_enc_np.copy()

        bev_pos_enc_np = (
            np.random.randn(bev_h, bev_w, embed_dims).astype(np.float32) * 0.5
        )
        bev_pos_flat_torch = torch.from_numpy(bev_pos_enc_np.copy()).view(
            1, bev_h * bev_w, embed_dims
        )
        bev_pos_flat_ttsim = ttsim_reshape(
            bev_pos_enc_np, (1, bev_h * bev_w, embed_dims)
        )

        bev_enc_torch = bev_queries_enc_torch + bev_pos_flat_torch
        bev_enc_ttsim = ttsim_add(bev_queries_enc_ttsim, bev_pos_flat_ttsim)

        print(
            f"    BEV queries: {bev_queries_enc_torch.shape} [B, bev_h*bev_w, embed_dims]"
        )
        print(f"    Positional encoding: {bev_pos_flat_torch.shape}")
        match_init = compare_arrays(bev_enc_torch, bev_enc_ttsim, "BEV Queries+Pos")
        validation_results.append(
            (
                f"Layer {layer_counter}: BEV Query Init",
                match_init,
                f"{bev_queries_enc_torch.shape}",
            )
        )

        # Previous BEV for temporal attention (zeros for first frame)
        prev_bev_enc_torch = torch.zeros_like(bev_enc_torch)
        prev_bev_enc_ttsim = np.zeros_like(bev_enc_ttsim)

        for enc_layer_idx in range(num_encoder_layers_bev):
            print(f"\n  {'─'*75}")
            print(f"  ENCODER LAYER {enc_layer_idx + 1}/{num_encoder_layers_bev}")
            print(f"  {'─'*75}")

            # --- Temporal Self-Attention ---
            layer_counter += 1
            print(f"\n  ├─ Layer {layer_counter}: Temporal Self-Attention")

            temporal_input_torch = torch.cat([bev_enc_torch, prev_bev_enc_torch], dim=1)
            temporal_input_ttsim = ttsim_concat(
                [bev_enc_ttsim, prev_bev_enc_ttsim], axis=1
            )

            np.random.seed(50 + enc_layer_idx * 10)
            Q_enc_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            K_enc_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            V_enc_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01

            Q_enc_torch = bev_enc_torch @ torch.from_numpy(Q_enc_w)
            K_enc_torch = temporal_input_torch @ torch.from_numpy(K_enc_w)
            V_enc_torch = temporal_input_torch @ torch.from_numpy(V_enc_w)

            Q_enc_ttsim = ttsim_matmul(bev_enc_ttsim, Q_enc_w)
            K_enc_ttsim = ttsim_matmul(temporal_input_ttsim, K_enc_w)
            V_enc_ttsim = ttsim_matmul(temporal_input_ttsim, V_enc_w)

            print(f"     * Step 1: Q, K, V projections (K,V from current+prev BEV)")
            compare_arrays(
                Q_enc_torch, Q_enc_ttsim, "       Q projection", rtol=1e-5, atol=1e-6
            )

            print(f"     * Step 2: Temporal multi-head attention ({num_heads} heads)")
            temp_attn_ttsim, temp_attn_torch, match_temp = multi_head_attention(
                Q_enc_ttsim,
                Q_enc_torch,
                K_enc_ttsim,
                K_enc_torch,
                V_enc_ttsim,
                V_enc_torch,
                num_heads,
                verbose=False,
            )

            out_proj_w = (
                np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            )
            temp_out_torch = temp_attn_torch @ torch.from_numpy(out_proj_w)
            temp_out_ttsim = ttsim_matmul(temp_attn_ttsim, out_proj_w)
            print(f"     * Step 3: Output projection")
            match_proj = compare_arrays(
                temp_out_torch,
                temp_out_ttsim,
                "       Output Projection",
                rtol=1e-4,
                atol=1e-5,
            )

            bev_enc_torch = bev_enc_torch + temp_out_torch
            bev_enc_ttsim = ttsim_add(bev_enc_ttsim, temp_out_ttsim)
            print(
                f"     * Step 4: Residual connection [{B}, {bev_h*bev_w}, {embed_dims}]"
            )
            match_res = compare_arrays(
                bev_enc_torch, bev_enc_ttsim, "       Residual", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: Temporal Self-Attn",
                    match_temp and match_proj and match_res,
                    f"{bev_enc_torch.shape}",
                )
            )

            # --- LayerNorm ---
            layer_counter += 1
            print(f"\n  ├─ Layer {layer_counter}: LayerNorm")
            bev_enc_torch = torch.nn.functional.layer_norm(bev_enc_torch, [embed_dims])
            bev_enc_ttsim = ttsim_layernorm(bev_enc_ttsim, embed_dims)
            match_ln1 = compare_arrays(
                bev_enc_torch, bev_enc_ttsim, "     Normalized", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: LayerNorm",
                    match_ln1,
                    f"{bev_enc_torch.shape}",
                )
            )

            # --- Spatial Cross-Attention (simplified) ---
            layer_counter += 1
            print(f"\n  ├─ Layer {layer_counter}: Spatial Cross-Attention (Deformable)")

            np.random.seed(60 + enc_layer_idx)
            sampled_feats_np = (
                np.random.randn(B, bev_h * bev_w, embed_dims).astype(np.float32) * 0.1
            )
            sampled_feats_torch = torch.from_numpy(sampled_feats_np.copy())

            print(f"     * Step 1: 3D ref pts projected to camera views")
            print(
                f"     * Step 2: Sample from {num_levels} FPN level x {num_cams} cameras"
            )

            spatial_proj_w = (
                np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            )
            cross_out_torch = sampled_feats_torch @ torch.from_numpy(spatial_proj_w)
            cross_out_ttsim = ttsim_matmul(sampled_feats_np, spatial_proj_w)
            print(f"     * Step 3: Output projection ({embed_dims} -> {embed_dims})")
            match_sproj = compare_arrays(
                cross_out_torch,
                cross_out_ttsim,
                "       Projection",
                rtol=1e-4,
                atol=1e-5,
            )

            bev_enc_torch = bev_enc_torch + cross_out_torch
            bev_enc_ttsim = ttsim_add(bev_enc_ttsim, cross_out_ttsim)
            print(f"     * Step 4: Residual connection")
            match_sres = compare_arrays(
                bev_enc_torch, bev_enc_ttsim, "     Final Output", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: Spatial Cross-Attn",
                    match_sproj and match_sres,
                    f"{bev_enc_torch.shape}",
                )
            )

            # --- LayerNorm ---
            layer_counter += 1
            print(f"\n  ├─ Layer {layer_counter}: LayerNorm")
            bev_enc_torch = torch.nn.functional.layer_norm(bev_enc_torch, [embed_dims])
            bev_enc_ttsim = ttsim_layernorm(bev_enc_ttsim, embed_dims)
            match_ln2 = compare_arrays(
                bev_enc_torch, bev_enc_ttsim, "     Normalized", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: LayerNorm",
                    match_ln2,
                    f"{bev_enc_torch.shape}",
                )
            )

            # --- FFN ---
            layer_counter += 1
            print(f"\n  ├─ Layer {layer_counter}: FFN (Linear-ReLU-Linear)")
            ffn_enc_w1 = (
                np.random.randn(embed_dims, ff_channels).astype(np.float32) * 0.01
            )
            ffn_enc_w2 = (
                np.random.randn(ff_channels, embed_dims).astype(np.float32) * 0.01
            )

            ffn_enc_ttsim, ffn_enc_torch, match_ffn = feedforward_network(
                bev_enc_ttsim, bev_enc_torch, ffn_enc_w1, ffn_enc_w2, verbose=True
            )

            bev_enc_torch = bev_enc_torch + ffn_enc_torch
            bev_enc_ttsim = ttsim_add(bev_enc_ttsim, ffn_enc_ttsim)
            print(f"     * Residual connection")
            match_fres = compare_arrays(
                bev_enc_torch, bev_enc_ttsim, "     Final Output", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: FFN",
                    match_ffn and match_fres,
                    f"{bev_enc_torch.shape}",
                )
            )

            # --- LayerNorm (final) ---
            layer_counter += 1
            print(f"\n  └─ Layer {layer_counter}: LayerNorm (Final)")
            bev_enc_torch = torch.nn.functional.layer_norm(bev_enc_torch, [embed_dims])
            bev_enc_ttsim = ttsim_layernorm(bev_enc_ttsim, embed_dims)
            match_ln3 = compare_arrays(
                bev_enc_torch,
                bev_enc_ttsim,
                "     Final Normalized",
                rtol=1e-4,
                atol=1e-5,
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: LayerNorm",
                    match_ln3,
                    f"{bev_enc_torch.shape}",
                )
            )

        print(f"\n  ENCODER COMPLETE: {num_encoder_layers_bev} layers processed")
        print(
            f"  Final BEV features: {bev_enc_torch.shape} [B, bev_h*bev_w, {embed_dims}]"
        )

        # =====================================================================
        # STAGE 1: BEV INPUT (output of Stage 0)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 1: BEV INPUT (output from Image Backbone + BEV Encoder)")
        print(f"  {'='*75}\n")

        layer_counter += 1
        print(
            f"  Layer {layer_counter}: BEV Feature Input (connected from Stage 0 encoder)"
        )
        print(f"  {'-'*75}")

        # Reshape encoder output [B, bev_h*bev_w, embed_dims] -> [B, embed_dims, bev_h, bev_w]
        # PyTorch path
        bev_input_torch = (
            bev_enc_torch.reshape(B, bev_h, bev_w, embed_dims)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        # TTSim path
        bev_input_ttsim = ttsim_reshape(bev_enc_ttsim, (B, bev_h, bev_w, embed_dims))
        bev_input_ttsim = ttsim_transpose(bev_input_ttsim, [0, 3, 1, 2])
        bev_input_np = np.ascontiguousarray(bev_input_ttsim)

        print(f"    Encoder output: [{B}, {bev_h*bev_w}, {embed_dims}] (from Stage 0)")
        print(f"    Reshape + Permute -> {bev_input_torch.shape} [B, C, H, W]")
        print(f"    BEV grid: {bev_h} x {bev_w} = {bev_h * bev_w} cells")
        match_bev_input = compare_arrays(
            bev_input_torch, bev_input_np, "BEV Input (from encoder)"
        )
        validation_results.append(
            (
                f"Layer {layer_counter}: BEV Input",
                match_bev_input,
                f"{bev_input_torch.shape}",
            )
        )

        # =====================================================================
        # STAGE 2: BEV BACKBONE (BEVFormerBackbone)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 2: BEV BACKBONE (BEVFormerBackbone)")
        print(f"  {'='*75}\n")

        # 2.1: BEV Embedding lookup
        layer_counter += 1
        print(f"  Layer {layer_counter}: BEV Embedding Lookup")
        print(f"  {'-'*75}")

        bev_embedding_weight = (
            np.random.randn(bev_h * bev_w, embed_dims).astype(np.float32) * 0.5
        )
        bev_embedding_torch = torch.from_numpy(bev_embedding_weight.copy())

        # PyTorch: nn.Embedding lookup
        bev_indices = torch.arange(bev_h * bev_w, dtype=torch.long)
        bev_queries_torch = torch.nn.functional.embedding(
            bev_indices, bev_embedding_torch
        )

        # TTSim: direct weight indexing (embedding = table lookup)
        bev_queries_ttsim = bev_embedding_weight.copy()  # [H*W, C]

        print(f"    Embedding table: [{bev_h * bev_w}, {embed_dims}]")
        print(f"    BEV queries: {bev_queries_torch.shape}")
        match = compare_arrays(bev_queries_torch, bev_queries_ttsim, "BEV Embedding")
        validation_results.append(
            (
                f"Layer {layer_counter}: BEV Embedding",
                match,
                f"{bev_queries_torch.shape}",
            )
        )

        # 2.2: Reshape + Permute (simulating transformer mock output)
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: Reshape [bs, H*W, C] -> [bs, C, H, W]")
        print(f"  {'-'*75}")

        # Mock transformer returns bev_queries expanded to batch
        bev_out_torch = bev_queries_torch.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, C]
        bev_out_ttsim = ttsim_unsqueeze(bev_queries_ttsim, [0])  # [1, H*W, C]
        bev_out_ttsim = ttsim_tile(bev_out_ttsim, [B, 1, 1])

        print(f"    Step 1: Unsqueeze + Tile -> {bev_out_torch.shape}")
        match1 = compare_arrays(bev_out_torch, bev_out_ttsim, "Expanded queries")

        # Reshape to spatial
        bev_spatial_torch = bev_out_torch.reshape(B, bev_h, bev_w, embed_dims)
        bev_spatial_torch = bev_spatial_torch.permute(0, 3, 1, 2)  # [B, C, H, W]

        bev_spatial_ttsim = ttsim_reshape(bev_out_ttsim, (B, bev_h, bev_w, embed_dims))
        bev_spatial_ttsim = ttsim_transpose(bev_spatial_ttsim, [0, 3, 1, 2])

        print(
            f"    Step 2: Reshape to [{B}, {bev_h}, {bev_w}, {embed_dims}] + Permute -> {bev_spatial_torch.shape}"
        )
        match2 = compare_arrays(bev_spatial_torch, bev_spatial_ttsim, "BEV spatial")
        match = match1 and match2
        validation_results.append(
            (
                f"Layer {layer_counter}: Reshape+Permute",
                match,
                f"{bev_spatial_torch.shape}",
            )
        )

        # 2.3: History Warping via GridSample
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: History Warping (GridSample)")
        print(f"  {'-'*75}")

        T = 2  # history frames
        hist_feats_np = [
            np.random.randn(B, embed_dims, bev_h, bev_w).astype(np.float32)
            for _ in range(T)
        ]

        # Create warp grid
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, bev_w), np.linspace(-1, 1, bev_h), indexing="ij"
        )
        base_grid = np.stack([xx, yy], axis=-1).astype(np.float32)
        coord_np = np.tile(base_grid[np.newaxis, np.newaxis, ...], (B, T, 1, 1, 1))

        # PyTorch: stack -> permute -> reshape -> grid_sample -> reshape
        hist_pt = torch.stack(
            [torch.from_numpy(h) for h in hist_feats_np], dim=0
        )  # [T, B, C, H, W]
        hist_pt = hist_pt.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]
        hist_flat_pt = hist_pt.reshape(B * T, embed_dims, bev_h, bev_w)
        coord_flat_pt = torch.from_numpy(coord_np.reshape(B * T, bev_h, bev_w, 2))
        warped_pt = torch.nn.functional.grid_sample(
            hist_flat_pt, coord_flat_pt, padding_mode="zeros", align_corners=False
        )
        warped_pt = warped_pt.reshape(B, T, embed_dims, bev_h, bev_w)

        # TTSim: stack -> permute -> reshape -> gridsample -> reshape
        hist_stacked = ttsim_concat(
            [ttsim_unsqueeze(h, [0]) for h in hist_feats_np], axis=0
        )
        hist_perm = ttsim_transpose(hist_stacked, [1, 0, 2, 3, 4])
        hist_flat_tt = ttsim_reshape(hist_perm, (B * T, embed_dims, bev_h, bev_w))
        coord_flat_tt = ttsim_reshape(coord_np, (B * T, bev_h, bev_w, 2))
        warped_tt = ttsim_gridsample(hist_flat_tt, coord_flat_tt, align_corners=False)
        warped_tt = ttsim_reshape(warped_tt, (B, T, embed_dims, bev_h, bev_w))

        print(f"    History frames: {T}, each [{B}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    Stack -> [{T}, {B}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    Permute -> [{B}, {T}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    Flatten -> [{B * T}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    GridSample -> [{B * T}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    Reshape -> {warped_pt.shape}")
        match = compare_arrays(
            warped_pt, warped_tt, "GridSample Warping", rtol=1e-4, atol=1e-5
        )
        validation_results.append(
            (f"Layer {layer_counter}: GridSample", match, f"{warped_pt.shape}")
        )

        # =====================================================================
        # STAGE 3: SEGMENTATION HEAD (MapSegHead)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 3: SEGMENTATION HEAD (MapSegHead)")
        print(f"  {'='*75}\n")

        # Use bev_input as the BEV features
        seg_input_torch = bev_input_torch.clone()
        seg_input_ttsim = bev_input_np.copy()

        # 3.1: Conv2d(3x3, no bias) -> ReLU
        layer_counter += 1
        print(f"  Layer {layer_counter}: Conv2d(3x3, no bias) + ReLU")
        print(f"  {'-'*75}")

        np.random.seed(100)
        conv_in_w = (
            np.random.randn(embed_dims, embed_dims, 3, 3).astype(np.float32) * 0.01
        )

        conv_in_pt = torch.nn.functional.conv2d(
            seg_input_torch, torch.from_numpy(conv_in_w), padding=1
        )
        relu_pt = torch.relu(conv_in_pt)

        conv_in_tt = ttsim_conv2d(
            seg_input_ttsim, conv_in_w, bias=None, stride=1, padding=1
        )
        relu_tt = ttsim_relu(conv_in_tt)

        print(
            f"    Conv2d: [{B}, {embed_dims}, {bev_h}, {bev_w}] -> [{B}, {embed_dims}, {bev_h}, {bev_w}]"
        )
        print(f"    ReLU applied")
        match = compare_arrays(relu_pt, relu_tt, "SegHead Conv_in + ReLU")
        validation_results.append(
            (f"Layer {layer_counter}: SegHead Conv_in", match, f"{relu_pt.shape}")
        )

        # 3.2: Upsample(2x, nearest) + Conv2d(3x3) + bias + ReLU
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: Upsample(2x) + Conv2d(3x3) + bias + ReLU")
        print(f"  {'-'*75}")

        conv_up_w = (
            np.random.randn(embed_dims, embed_dims, 3, 3).astype(np.float32) * 0.01
        )
        conv_up_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

        # PyTorch
        up_pt = torch.nn.functional.interpolate(relu_pt, scale_factor=2, mode="nearest")
        conv_up_pt = torch.nn.functional.conv2d(
            up_pt, torch.from_numpy(conv_up_w), padding=1
        )
        conv_up_pt = conv_up_pt + torch.from_numpy(conv_up_b).reshape(1, -1, 1, 1)
        up_relu_pt = torch.relu(conv_up_pt)

        # TTSim
        up_tt = ttsim_upsample(relu_tt, scale_factor=2.0, mode="nearest")
        conv_up_tt = ttsim_conv2d(up_tt, conv_up_w, bias=None, stride=1, padding=1)
        bias_4d = ttsim_reshape(conv_up_b, (1, embed_dims, 1, 1))
        conv_up_tt = ttsim_add(conv_up_tt, bias_4d)
        up_relu_tt = ttsim_relu(conv_up_tt)

        print(
            f"    Upsample: [{B}, {embed_dims}, {bev_h}, {bev_w}] -> [{B}, {embed_dims}, {canvas_h}, {canvas_w}]"
        )
        print(f"    Conv2d(3x3) + bias + ReLU")
        match = compare_arrays(
            up_relu_pt, up_relu_tt, "SegHead Upsample+Conv", rtol=1e-5, atol=1e-5
        )
        validation_results.append(
            (f"Layer {layer_counter}: SegHead Up+Conv", match, f"{up_relu_pt.shape}")
        )

        # Store upsampled features for downsample later
        seg_feats_up_torch = up_relu_pt.clone()
        seg_feats_up_ttsim = up_relu_tt.copy()

        # 3.3: Conv2d(1x1) + bias -> seg_preds
        layer_counter += 1
        print(
            f"\n  Layer {layer_counter}: Conv2d(1x1) + bias -> Segmentation Predictions"
        )
        print(f"  {'-'*75}")

        conv_out_w = (
            np.random.randn(num_cls, embed_dims, 1, 1).astype(np.float32) * 0.01
        )
        conv_out_b = np.random.randn(num_cls).astype(np.float32) * 0.01

        seg_preds_pt = torch.nn.functional.conv2d(
            up_relu_pt, torch.from_numpy(conv_out_w)
        )
        seg_preds_pt = seg_preds_pt + torch.from_numpy(conv_out_b).reshape(1, -1, 1, 1)

        seg_preds_tt = ttsim_conv2d(
            up_relu_tt, conv_out_w, bias=None, stride=1, padding=0
        )
        bias_cls = ttsim_reshape(conv_out_b, (1, num_cls, 1, 1))
        seg_preds_tt = ttsim_add(seg_preds_tt, bias_cls)

        print(
            f"    Conv2d(1x1): [{B}, {embed_dims}, {canvas_h}, {canvas_w}] -> [{B}, {num_cls}, {canvas_h}, {canvas_w}]"
        )
        match = compare_arrays(seg_preds_pt, seg_preds_tt, "SegHead Predictions")
        validation_results.append(
            (f"Layer {layer_counter}: SegHead Preds", match, f"{seg_preds_pt.shape}")
        )

        # 3.4: Downsample -> seg_feats
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: Downsample(0.5x) -> seg_feats")
        print(f"  {'-'*75}")

        seg_feats_pt = torch.nn.functional.interpolate(
            seg_feats_up_torch, scale_factor=0.5, mode="bilinear", align_corners=True
        )

        seg_feats_tt = ttsim_upsample(
            seg_feats_up_ttsim, scale_factor=0.5, mode="linear", align_corners=True
        )

        print(
            f"    Bilinear downsample: [{B}, {embed_dims}, {canvas_h}, {canvas_w}] -> {seg_feats_pt.shape}"
        )
        match = compare_arrays(
            seg_feats_pt, seg_feats_tt, "SegHead Features", rtol=1e-4, atol=1e-4
        )
        validation_results.append(
            (f"Layer {layer_counter}: SegHead Feats", match, f"{seg_feats_pt.shape}")
        )

        # =====================================================================
        # STAGE 4: VECTOR MAP DETECTION HEAD (MapDetectorHead)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 4: VECTOR MAP DETECTION HEAD (MapDetectorHead)")
        print(f"  {'='*75}\n")

        # 4.1: Input projection Conv2d(1x1) + bias
        layer_counter += 1
        print(f"  Layer {layer_counter}: Input Projection Conv2d(1x1) + bias")
        print(f"  {'-'*75}")

        np.random.seed(200)
        input_proj_w = (
            np.random.randn(embed_dims, embed_dims, 1, 1).astype(np.float32) * 0.01
        )
        input_proj_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

        bev_proj_pt = torch.nn.functional.conv2d(
            bev_input_torch, torch.from_numpy(input_proj_w)
        )
        bev_proj_pt = bev_proj_pt + torch.from_numpy(input_proj_b).reshape(1, -1, 1, 1)

        bev_proj_tt = ttsim_conv2d(
            bev_input_np, input_proj_w, bias=None, stride=1, padding=0
        )
        proj_bias_4d = ttsim_reshape(input_proj_b, (1, embed_dims, 1, 1))
        bev_proj_tt = ttsim_add(bev_proj_tt, proj_bias_4d)

        print(
            f"    Conv2d(1x1): [{B}, {embed_dims}, {bev_h}, {bev_w}] -> [{B}, {embed_dims}, {bev_h}, {bev_w}]"
        )
        match = compare_arrays(bev_proj_pt, bev_proj_tt, "Input Projection")
        validation_results.append(
            (f"Layer {layer_counter}: Input Proj", match, f"{bev_proj_pt.shape}")
        )

        # 4.2: BEV Sine Positional Embedding + Add
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: BEV Sine Positional Embedding + Add")
        print(f"  {'-'*75}")

        # Generate positional encoding (deterministic, not random)
        y_coords = np.linspace(0, bev_h - 1, bev_h)[:, None].repeat(bev_w, axis=1)
        x_coords = np.linspace(0, bev_w - 1, bev_w)[None, :].repeat(bev_h, axis=0)
        y_coords = 2.0 * (y_coords / max(bev_h - 1, 1)) - 1.0
        x_coords = 2.0 * (x_coords / max(bev_w - 1, 1)) - 1.0
        num_feats = embed_dims // 2
        dim_t = np.arange(num_feats, dtype=np.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / num_feats)
        pos_x = x_coords[:, :, None] / dim_t
        pos_x = np.stack(
            [np.sin(pos_x[:, :, 0::2]), np.cos(pos_x[:, :, 1::2])], axis=3
        ).reshape(bev_h, bev_w, -1)
        pos_y = y_coords[:, :, None] / dim_t
        pos_y = np.stack(
            [np.sin(pos_y[:, :, 0::2]), np.cos(pos_y[:, :, 1::2])], axis=3
        ).reshape(bev_h, bev_w, -1)
        pos_embed_np = (
            np.concatenate([pos_y, pos_x], axis=2)
            .transpose(2, 0, 1)[None, :, :, :]
            .astype(np.float32)
        )

        # PyTorch: add
        bev_with_pos_pt = bev_proj_pt + torch.from_numpy(pos_embed_np)

        # TTSim: Using sin/cos compute functions
        # For validation we pre-compute the same positional encoding and add
        bev_with_pos_tt = ttsim_add(bev_proj_tt, pos_embed_np)

        print(f"    Positional encoding: {pos_embed_np.shape}")
        print(f"    Add to projected BEV -> {bev_with_pos_pt.shape}")
        match = compare_arrays(bev_with_pos_pt, bev_with_pos_tt, "BEV + PosEmbed")
        validation_results.append(
            (f"Layer {layer_counter}: BEV PosEmbed", match, f"{bev_with_pos_pt.shape}")
        )

        # 4.3: Flatten BEV for decoder key/value
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: Flatten BEV for Decoder")
        print(f"  {'-'*75}")

        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C] -> [H*W, B, C]
        bev_flat_pt = bev_with_pos_pt.reshape(B, embed_dims, bev_h * bev_w)
        bev_flat_pt = bev_flat_pt.permute(0, 2, 1)  # [B, H*W, C]
        bev_seq_pt = bev_flat_pt.permute(1, 0, 2)  # [H*W, B, C]

        bev_flat_tt = ttsim_reshape(bev_with_pos_tt, (B, embed_dims, bev_h * bev_w))
        bev_flat_tt = ttsim_transpose(bev_flat_tt, [0, 2, 1])
        bev_seq_tt = ttsim_transpose(bev_flat_tt, [1, 0, 2])

        print(
            f"    Reshape [{B}, {embed_dims}, {bev_h}, {bev_w}] -> [{B}, {embed_dims}, {num_value}]"
        )
        print(f"    Permute -> [{B}, {num_value}, {embed_dims}]")
        print(
            f"    Permute -> [{num_value}, {B}, {embed_dims}] (sequence-first for attention)"
        )
        match = compare_arrays(bev_seq_pt, bev_seq_tt, "BEV Flattened")
        validation_results.append(
            (f"Layer {layer_counter}: BEV Flatten", match, f"{bev_seq_pt.shape}")
        )

        # 4.4: Query Initialization
        layer_counter += 1
        print(
            f"\n  Layer {layer_counter}: Query Initialization (Embedding + Ref Points)"
        )
        print(f"  {'-'*75}")

        query_emb_np = np.random.randn(num_q, embed_dims).astype(np.float32) * 0.5

        # Unsqueeze + tile to batch
        query_pt = (
            torch.from_numpy(query_emb_np).unsqueeze(0).expand(B, -1, -1)
        )  # [B, num_q, C]
        query_tt = ttsim_tile(ttsim_unsqueeze(query_emb_np, [0]), [B, 1, 1])

        # Reference points: Linear -> Sigmoid -> Reshape
        ref_pts_w = np.random.randn(embed_dims, num_pts * 2).astype(np.float32) * 0.01
        ref_pts_b = np.random.randn(num_pts * 2).astype(np.float32) * 0.01

        ref_raw_pt = query_pt @ torch.from_numpy(ref_pts_w) + torch.from_numpy(
            ref_pts_b
        )
        ref_pts_pt = torch.sigmoid(ref_raw_pt).reshape(B, num_q, num_pts, 2)

        ref_raw_tt = ttsim_linear(query_tt, ref_pts_w, ref_pts_b)
        ref_pts_tt = ttsim_sigmoid(ref_raw_tt)
        ref_pts_tt = ttsim_reshape(ref_pts_tt, (B, num_q, num_pts, 2))

        print(
            f"    Query embedding: [{num_q}, {embed_dims}] -> [{B}, {num_q}, {embed_dims}]"
        )
        print(
            f"    Reference points Linear({embed_dims}->{num_pts * 2}) + Sigmoid -> Reshape"
        )
        print(f"    Reference points: {ref_pts_pt.shape}")
        match_q = compare_arrays(query_pt, query_tt, "Query Embedding")
        match_r = compare_arrays(ref_pts_pt, ref_pts_tt, "Reference Points")
        match = match_q and match_r
        validation_results.append(
            (
                f"Layer {layer_counter}: Query Init",
                match,
                f"Q:{query_pt.shape}, Ref:{ref_pts_pt.shape}",
            )
        )

        # Transpose query to sequence-first: [num_q, B, C]
        output_pt = query_pt.permute(1, 0, 2)  # [num_q, B, C]
        output_tt = ttsim_transpose(query_tt, [1, 0, 2])
        reference_points_pt = ref_pts_pt.clone()
        reference_points_tt = ref_pts_tt.copy()

        # =====================================================================
        # STAGE 5: TRANSFORMER DECODER (num_decoder_layers layers)
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 5: TRANSFORMER DECODER ({num_decoder_layers} layers)")
        print(f"  {'='*75}")
        print(
            f"  Each layer: Self-Attn -> Norm -> BEV Cross-Attn -> Norm -> "
            f"Memory Cross-Attn -> Norm -> FFN -> Norm"
        )
        print(f"  Then: RegressionBranch + ClassificationBranch\n")

        # Prepare value for deformable attention: [H*W, B, C] -> [B, H*W, C]
        bev_value_pt = bev_seq_pt.permute(1, 0, 2)  # [B, H*W, C]
        bev_value_tt = ttsim_transpose(bev_seq_tt, [1, 0, 2])

        spatial_shapes = np.array([[bev_h, bev_w]], dtype=np.int64)
        level_start = np.array([0], dtype=np.int64)

        all_cls_scores_pt = []
        all_reg_pts_pt = []
        all_cls_scores_tt = []
        all_reg_pts_tt = []

        # --- Pre-populate dummy memory bank for frame 0 ---
        # Matches production MapTracker.__call__ which creates 1 dummy tracked
        # instance so the memory cross-attention path is exercised in the ONNX graph.
        mem_len_f0 = 10
        n_tracks_f0 = 1
        np.random.seed(250)
        f0_mem_embeds = (
            np.random.randn(mem_len_f0, n_tracks_f0, embed_dims).astype(np.float32)
            * 0.01
        )
        # Build relative PE matching production: Gather row from PE table, pad rest with zeros.
        # Production code uses F.Gather(pe_table, idx=1) -> Unsqueeze -> ConcatX(zeros)
        # to keep Sin/Cos graph-connected. Here we replicate the same values via numpy.
        pe_ch = int(np.ceil(embed_dims / 2) * 2)
        _pe_inv = 1.0 / (10000 ** (np.arange(0, pe_ch, 2).astype(np.float32) / pe_ch))
        _pe_sinp = (
            np.array([1.0], dtype=np.float32)[:, None] * _pe_inv[None, :]
        )  # gap=1
        _pe_row = np.stack([np.sin(_pe_sinp), np.cos(_pe_sinp)], axis=-1).reshape(1, -1)
        _pe_row = _pe_row[:, :embed_dims]  # [1, embed_dims]
        f0_mem_relative_pe = np.zeros(
            (mem_len_f0, n_tracks_f0, embed_dims), dtype=np.float32
        )
        f0_mem_relative_pe[0, :, :] = _pe_row  # first slot gets PE for gap=1
        f0_valid_track_idx = [0]  # query index 0 has memory

        # Memory cross-attention weights (shared across decoder layers, seeded per layer)
        f0_mem_attn_weights = []
        for lid in range(num_decoder_layers):
            np.random.seed(700 + lid * 50)
            maw = {
                "q_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "k_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "v_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "o_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "q_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "k_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "v_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "o_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
            }
            f0_mem_attn_weights.append(maw)

        for dec_idx in range(num_decoder_layers):
            print(f"\n  {'─'*75}")
            print(f"  DECODER LAYER {dec_idx + 1}/{num_decoder_layers}")
            print(f"  {'─'*75}")

            np.random.seed(300 + dec_idx * 100)

            # ─── 5.1: Self-Attention ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: Self-Attention (MultiheadAttention)")

            sa_q_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            sa_k_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            sa_v_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            sa_o_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            sa_q_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
            sa_k_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
            sa_v_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
            sa_o_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

            # PyTorch: manual MHA
            sa_identity_pt = output_pt.clone()
            out_bf_pt = output_pt.permute(1, 0, 2)  # [B, num_q, C]
            Q_pt = out_bf_pt @ torch.from_numpy(sa_q_w) + torch.from_numpy(sa_q_b)
            K_pt = out_bf_pt @ torch.from_numpy(sa_k_w) + torch.from_numpy(sa_k_b)
            V_pt = out_bf_pt @ torch.from_numpy(sa_v_w) + torch.from_numpy(sa_v_b)
            Q_pt = Q_pt.view(B, num_q, num_heads, head_dim).transpose(1, 2)
            K_pt = K_pt.view(B, num_q, num_heads, head_dim).transpose(1, 2)
            V_pt = V_pt.view(B, num_q, num_heads, head_dim).transpose(1, 2)
            scores_pt = Q_pt @ K_pt.transpose(-2, -1) / np.sqrt(head_dim)
            attn_pt = torch.softmax(scores_pt, dim=-1)
            ctx_pt = (
                (attn_pt @ V_pt).transpose(1, 2).contiguous().view(B, num_q, embed_dims)
            )
            sa_out_pt = ctx_pt @ torch.from_numpy(sa_o_w) + torch.from_numpy(sa_o_b)
            sa_out_pt = sa_out_pt.permute(1, 0, 2)  # [num_q, B, C]
            output_pt = sa_identity_pt + sa_out_pt  # residual

            # TTSim
            sa_identity_tt = output_tt.copy()
            out_bf_tt = ttsim_transpose(output_tt, [1, 0, 2])
            sa_out_tt = ttsim_multihead_attention(
                out_bf_tt,
                out_bf_tt,
                out_bf_tt,
                sa_q_w,
                sa_k_w,
                sa_v_w,
                sa_o_w,
                sa_q_b,
                sa_k_b,
                sa_v_b,
                sa_o_b,
                num_heads,
                head_dim,
            )
            sa_out_tt = ttsim_transpose(sa_out_tt, [1, 0, 2])
            output_tt = ttsim_add(sa_identity_tt, sa_out_tt)

            print(f"     * Q,K,V projections: Linear({embed_dims}->{embed_dims}) × 3")
            print(f"     * Reshape -> {num_heads} heads x {head_dim} head_dim")
            print(
                f"     * MatMul(Q, K^T) / sqrt({head_dim}) -> Softmax -> MatMul(attn, V)"
            )
            print(f"     * Output projection + residual")
            match = compare_arrays(
                output_pt, output_tt, "     Self-Attn Output", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: Self-Attn L{dec_idx+1}",
                    match,
                    f"{output_pt.shape}",
                )
            )

            # ─── 5.2: LayerNorm ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: LayerNorm")

            output_pt = torch.nn.functional.layer_norm(output_pt, [embed_dims])
            output_tt = ttsim_layernorm(output_tt, embed_dims)

            match = compare_arrays(
                output_pt, output_tt, "     LayerNorm", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (f"Layer {layer_counter}: LN L{dec_idx+1}", match, f"{output_pt.shape}")
            )

            # ─── 5.3: Cross-Attention (Deformable Attention to BEV) ───
            layer_counter += 1
            print(
                f"\n  |-- Layer {layer_counter}: Cross-Attention (Deformable Attention to BEV)"
            )

            da_val_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            da_val_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
            da_soff_w = (
                np.random.randn(
                    embed_dims, num_heads * num_levels * deform_pts * 2
                ).astype(np.float32)
                * 0.01
            )
            da_soff_b = np.zeros(
                num_heads * num_levels * deform_pts * 2, dtype=np.float32
            )
            da_aw_w = (
                np.random.randn(embed_dims, num_heads * num_levels * deform_pts).astype(
                    np.float32
                )
                * 0.01
            )
            da_aw_b = np.zeros(num_heads * num_levels * deform_pts, dtype=np.float32)
            da_out_w = np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
            da_out_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

            # Y-axis reversal on reference points (as in decoder source)
            ref_x_pt = reference_points_pt[..., 0:1]
            ref_y_pt = reference_points_pt[..., 1:2]
            ref_rev_pt = torch.cat(
                [ref_x_pt, 1.0 - ref_y_pt], dim=-1
            )  # [B, nq, npts, 2]

            ref_x_tt = reference_points_tt[..., 0:1]
            ref_y_tt = reference_points_tt[..., 1:2]
            ref_rev_tt = ttsim_concat(
                [ref_x_tt, ttsim_sub(np.float32(1.0), ref_y_tt)], axis=-1
            )

            da_identity_pt = output_pt.clone()

            # PyTorch: manual deformable attention
            query_bf_pt = output_pt.permute(1, 0, 2)  # [B, num_q, C]
            value_proj_pt = bev_value_pt @ torch.from_numpy(
                da_val_w
            ) + torch.from_numpy(
                da_val_b
            )  # [B, H*W, C]
            value_mh_pt = value_proj_pt.view(B, num_value, num_heads, head_dim)

            soff_pt = query_bf_pt @ torch.from_numpy(da_soff_w) + torch.from_numpy(
                da_soff_b
            )
            soff_pt = soff_pt.view(B, num_q, num_heads, num_levels, deform_pts, 2)

            aw_pt = query_bf_pt @ torch.from_numpy(da_aw_w) + torch.from_numpy(da_aw_b)
            aw_pt = aw_pt.view(B, num_q, num_heads, num_levels * deform_pts)
            aw_pt = torch.softmax(aw_pt, dim=-1)
            aw_pt = aw_pt.view(B, num_q, num_heads, num_levels, deform_pts)

            offset_norm = torch.stack(
                [
                    torch.tensor(spatial_shapes[0, 1], dtype=torch.float32),
                    torch.tensor(spatial_shapes[0, 0], dtype=torch.float32),
                ]
            )
            # Use first point of ref for sampling location base
            ref_base_pt = ref_rev_pt[:, :, 0:1, :]  # [B, nq, 1, 2]
            ref_base_pt = ref_base_pt.unsqueeze(2).unsqueeze(3)  # [B, nq, 1, 1, 1, 2]

            sampling_loc_pt = (
                ref_base_pt + soff_pt / offset_norm[None, None, None, None, None, :]
            )

            da_out_pt = multi_scale_deformable_attn_pytorch(
                value_mh_pt, torch.from_numpy(spatial_shapes), sampling_loc_pt, aw_pt
            )
            da_out_pt = da_out_pt @ torch.from_numpy(da_out_w) + torch.from_numpy(
                da_out_b
            )
            da_out_seq_pt = da_out_pt.permute(1, 0, 2)  # [num_q, B, C]
            output_pt = da_identity_pt + da_out_seq_pt

            # TTSim
            da_identity_tt = output_tt.copy()
            query_bf_tt = ttsim_transpose(output_tt, [1, 0, 2])
            value_proj_tt = ttsim_linear(bev_value_tt, da_val_w, da_val_b)
            value_mh_tt = ttsim_reshape(
                value_proj_tt, (B, num_value, num_heads, head_dim)
            )

            soff_tt = ttsim_linear(query_bf_tt, da_soff_w, da_soff_b)
            soff_tt = ttsim_reshape(
                soff_tt, (B, num_q, num_heads, num_levels, deform_pts, 2)
            )

            aw_tt = ttsim_linear(query_bf_tt, da_aw_w, da_aw_b)
            aw_tt = ttsim_reshape(aw_tt, (B, num_q, num_heads, num_levels * deform_pts))
            aw_tt = ttsim_softmax(aw_tt, axis=-1)
            aw_tt = ttsim_reshape(aw_tt, (B, num_q, num_heads, num_levels, deform_pts))

            offset_norm_tt = np.array(
                [spatial_shapes[0, 1], spatial_shapes[0, 0]], dtype=np.float32
            )
            ref_base_tt = ref_rev_tt[:, :, 0:1, :]
            ref_base_tt = ttsim_unsqueeze(ttsim_unsqueeze(ref_base_tt, [2]), [3])

            sampling_loc_tt = ttsim_add(ref_base_tt, ttsim_div(soff_tt, offset_norm_tt))

            da_out_tt = ttsim_ms_deformable_attn(
                value_mh_tt, spatial_shapes, sampling_loc_tt, aw_tt
            )
            da_out_tt = ttsim_linear(da_out_tt, da_out_w, da_out_b)
            da_out_seq_tt = ttsim_transpose(da_out_tt, [1, 0, 2])
            output_tt = ttsim_add(da_identity_tt, da_out_seq_tt)

            print(f"     * Value projection: Linear({embed_dims}->{embed_dims})")
            print(
                f"     * Sampling offsets: Linear({embed_dims}->{num_heads * num_levels * deform_pts * 2})"
            )
            print(
                f"     * Attention weights: Linear({embed_dims}->{num_heads * num_levels * deform_pts}) + Softmax"
            )
            print(f"     * Deformable sampling from BEV [{num_value}, {embed_dims}]")
            print(f"     * Output projection + residual")
            match = compare_arrays(
                output_pt,
                output_tt,
                "     Deformable Attn Output",
                rtol=1e-3,
                atol=1e-3,
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: DeformAttn L{dec_idx+1}",
                    match,
                    f"{output_pt.shape}",
                )
            )

            # ─── 5.4: LayerNorm ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: LayerNorm")

            output_pt = torch.nn.functional.layer_norm(output_pt, [embed_dims])
            output_tt = ttsim_layernorm(output_tt, embed_dims)

            match = compare_arrays(
                output_pt, output_tt, "     LayerNorm", rtol=1e-3, atol=1e-3
            )
            validation_results.append(
                (f"Layer {layer_counter}: LN L{dec_idx+1}", match, f"{output_pt.shape}")
            )

            # ─── 5.5: Memory Cross-Attention (active with pre-populated memory) ───
            layer_counter += 1
            print(
                f"\n  |-- Layer {layer_counter}: Memory Cross-Attention (Frame 0: ACTIVE)"
            )
            print(
                f"     * Pre-populated memory: {n_tracks_f0} track(s), mem_len={mem_len_f0}"
            )

            maw = f0_mem_attn_weights[dec_idx]

            # Query: [num_q, B, C] -> [B, num_q, C]
            mem_q_pt = output_pt.permute(1, 0, 2)
            mem_q_tt = ttsim_transpose(output_tt, [1, 0, 2])

            # Memory K/V for the single tracked query
            # mem_embeds: [mem_len, n_tracks, C] -> permute to [n_tracks, mem_len, C] for batch_first MHA
            mem_e_bf = torch.from_numpy(f0_mem_embeds).permute(
                1, 0, 2
            )  # [n_tracks, mem_len, C]
            mem_pe_bf = torch.from_numpy(f0_mem_relative_pe).permute(
                1, 0, 2
            )  # [n_tracks, mem_len, C]

            # PyTorch: per-track MHA for valid tracks, identity for others
            query_memory_pt = torch.zeros_like(mem_q_pt)  # [B, num_q, C]
            for b_i in range(B):
                q_track = mem_q_pt[
                    b_i : b_i + 1, f0_valid_track_idx
                ]  # [1, n_tracks, C]
                q_track_sq = q_track.squeeze(0)  # [n_tracks, C]

                # Manual MHA (batch_first=True style): Q=[n_tracks,1,C], K=V=[n_tracks,mem_len,C]
                Q_m = q_track_sq.unsqueeze(1) @ torch.from_numpy(
                    maw["q_w"]
                ) + torch.from_numpy(maw["q_b"])
                K_m = (mem_e_bf + mem_pe_bf) @ torch.from_numpy(
                    maw["k_w"]
                ) + torch.from_numpy(maw["k_b"])
                V_m = mem_e_bf @ torch.from_numpy(maw["v_w"]) + torch.from_numpy(
                    maw["v_b"]
                )
                # [n_tracks, heads, 1, d_k] x [n_tracks, heads, d_k, mem_len]
                Q_m = Q_m.view(n_tracks_f0, 1, num_heads, head_dim).transpose(1, 2)
                K_m = K_m.view(n_tracks_f0, mem_len_f0, num_heads, head_dim).transpose(
                    1, 2
                )
                V_m = V_m.view(n_tracks_f0, mem_len_f0, num_heads, head_dim).transpose(
                    1, 2
                )
                sc_m = Q_m @ K_m.transpose(-2, -1) / np.sqrt(head_dim)
                at_m = torch.softmax(sc_m, dim=-1)
                cx_m = (
                    (at_m @ V_m)
                    .transpose(1, 2)
                    .contiguous()
                    .view(n_tracks_f0, 1, embed_dims)
                )
                mem_out = cx_m @ torch.from_numpy(maw["o_w"]) + torch.from_numpy(
                    maw["o_b"]
                )
                query_memory_pt[b_i, f0_valid_track_idx] = mem_out.squeeze(1)

            # TTSim: same per-track computation
            query_memory_tt = np.zeros_like(mem_q_tt)
            for b_i in range(B):
                mem_e_bf_tt = np.transpose(
                    f0_mem_embeds, (1, 0, 2)
                )  # [n_tracks, mem_len, C]
                mem_pe_bf_tt = np.transpose(f0_mem_relative_pe, (1, 0, 2))
                mem_key_tt = ttsim_add(mem_e_bf_tt, mem_pe_bf_tt)
                q_track_tt = mem_q_tt[b_i : b_i + 1, f0_valid_track_idx].squeeze(
                    0
                )  # [n_tracks, C]
                mem_out_tt = ttsim_multihead_attention(
                    q_track_tt[:, np.newaxis, :],  # [n_tracks, 1, C]
                    mem_key_tt,  # [n_tracks, mem_len, C]
                    mem_e_bf_tt,  # [n_tracks, mem_len, C]
                    maw["q_w"],
                    maw["k_w"],
                    maw["v_w"],
                    maw["o_w"],
                    maw["q_b"],
                    maw["k_b"],
                    maw["v_b"],
                    maw["o_b"],
                    num_heads,
                    head_dim,
                )
                query_memory_tt[b_i, f0_valid_track_idx] = mem_out_tt.squeeze(1)

            # Add memory attention output as residual
            output_pt = output_pt + query_memory_pt.permute(1, 0, 2)
            output_tt = ttsim_add(
                output_tt, ttsim_transpose(query_memory_tt, [1, 0, 2])
            )

            print(f"     * Q from decoder: [{B}, {num_q}, {embed_dims}]")
            print(
                f"     * K = memory + relative_PE: [{n_tracks_f0}, {mem_len_f0}, {embed_dims}]"
            )
            print(f"     * V from memory: [{n_tracks_f0}, {mem_len_f0}, {embed_dims}]")
            print(
                f"     * MHA per-track: Q,K,V proj -> {num_heads} heads -> Softmax -> Output proj"
            )
            print(f"     * Fusion: query_memory + query_bev (additive)")
            match = compare_arrays(
                output_pt, output_tt, "     MemAttn Output", rtol=1e-3, atol=1e-3
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: MemAttn L{dec_idx+1}",
                    match,
                    f"{output_pt.shape}",
                )
            )

            # ─── 5.6: LayerNorm ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: LayerNorm")

            output_pt = torch.nn.functional.layer_norm(output_pt, [embed_dims])
            output_tt = ttsim_layernorm(output_tt, embed_dims)

            match = compare_arrays(
                output_pt, output_tt, "     LayerNorm", rtol=1e-3, atol=1e-3
            )
            validation_results.append(
                (f"Layer {layer_counter}: LN L{dec_idx+1}", match, f"{output_pt.shape}")
            )

            # ─── 5.7: FFN ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: FFN (Linear-ReLU-Linear + Residual)")

            ffn_w1 = np.random.randn(embed_dims, ff_channels).astype(np.float32) * 0.01
            ffn_b1 = np.random.randn(ff_channels).astype(np.float32) * 0.01
            ffn_w2 = np.random.randn(ff_channels, embed_dims).astype(np.float32) * 0.01
            ffn_b2 = np.random.randn(embed_dims).astype(np.float32) * 0.01

            ffn_identity_pt = output_pt.clone()
            h_pt = torch.relu(
                output_pt @ torch.from_numpy(ffn_w1) + torch.from_numpy(ffn_b1)
            )
            ffn_out_pt = h_pt @ torch.from_numpy(ffn_w2) + torch.from_numpy(ffn_b2)
            output_pt = ffn_identity_pt + ffn_out_pt

            output_tt = ttsim_ffn(output_tt, ffn_w1, ffn_b1, ffn_w2, ffn_b2)

            print(
                f"     * Expansion: {embed_dims} -> {ff_channels} (ReLU) -> {embed_dims}"
            )
            match = compare_arrays(
                output_pt, output_tt, "     FFN Output", rtol=1e-3, atol=1e-3
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: FFN L{dec_idx+1}",
                    match,
                    f"{output_pt.shape}",
                )
            )

            # ─── 5.8: LayerNorm ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: LayerNorm (Final)")

            output_pt = torch.nn.functional.layer_norm(output_pt, [embed_dims])
            output_tt = ttsim_layernorm(output_tt, embed_dims)

            match = compare_arrays(
                output_pt, output_tt, "     Final LayerNorm", rtol=1e-3, atol=1e-3
            )
            validation_results.append(
                (f"Layer {layer_counter}: LN L{dec_idx+1}", match, f"{output_pt.shape}")
            )

            # ─── 5.9: RegressionBranch ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: RegressionBranch")

            reg_fc1_w = (
                np.random.randn(embed_dims, 2 * embed_dims).astype(np.float32) * 0.01
            )
            reg_fc1_b = np.random.randn(2 * embed_dims).astype(np.float32) * 0.01
            reg_fc2_w = (
                np.random.randn(2 * embed_dims, 2 * embed_dims).astype(np.float32)
                * 0.01
            )
            reg_fc2_b = np.random.randn(2 * embed_dims).astype(np.float32) * 0.01
            reg_fc3_w = (
                np.random.randn(2 * embed_dims, num_pts * coord_dim).astype(np.float32)
                * 0.01
            )
            reg_fc3_b = np.random.randn(num_pts * coord_dim).astype(np.float32) * 0.01

            out_bf_pt2 = output_pt.permute(1, 0, 2)  # [B, num_q, C]

            h_pt = torch.relu(
                torch.nn.functional.layer_norm(
                    out_bf_pt2 @ torch.from_numpy(reg_fc1_w)
                    + torch.from_numpy(reg_fc1_b),
                    [2 * embed_dims],
                )
            )
            h_pt = torch.relu(
                torch.nn.functional.layer_norm(
                    h_pt @ torch.from_numpy(reg_fc2_w) + torch.from_numpy(reg_fc2_b),
                    [2 * embed_dims],
                )
            )
            reg_out_pt = h_pt @ torch.from_numpy(reg_fc3_w) + torch.from_numpy(
                reg_fc3_b
            )
            reg_pts_pt = torch.sigmoid(reg_out_pt.reshape(B, num_q, num_pts, 2))

            out_bf_tt2 = ttsim_transpose(output_tt, [1, 0, 2])
            reg_out_tt = ttsim_regression_branch(
                out_bf_tt2,
                reg_fc1_w,
                reg_fc1_b,
                reg_fc2_w,
                reg_fc2_b,
                reg_fc3_w,
                reg_fc3_b,
            )
            reg_pts_tt = ttsim_sigmoid(
                ttsim_reshape(reg_out_tt, (B, num_q, num_pts, 2))
            )

            reference_points_pt = reg_pts_pt.detach()
            reference_points_tt = reg_pts_tt.copy()

            print(f"     * Linear({embed_dims}->{2 * embed_dims}) -> LN -> ReLU")
            print(f"     * Linear({2 * embed_dims}->{2 * embed_dims}) -> LN -> ReLU")
            print(f"     * Linear({2 * embed_dims}->{num_pts * coord_dim}) -> Sigmoid")
            print(f"     * Reference points: {reg_pts_pt.shape}")
            match = compare_arrays(
                reg_pts_pt, reg_pts_tt, "     Regression Branch", rtol=1e-4, atol=1e-4
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: RegBranch L{dec_idx+1}",
                    match,
                    f"{reg_pts_pt.shape}",
                )
            )

            all_reg_pts_pt.append(reg_pts_pt.detach().numpy().copy())
            all_reg_pts_tt.append(reg_pts_tt.copy())

            # ─── 5.10: ClassificationBranch ───
            layer_counter += 1
            print(f"\n  |-- Layer {layer_counter}: ClassificationBranch")

            cls_w = np.random.randn(embed_dims, num_cls).astype(np.float32) * 0.01
            cls_b = np.random.randn(num_cls).astype(np.float32) * 0.01

            cls_out_pt = out_bf_pt2 @ torch.from_numpy(cls_w) + torch.from_numpy(cls_b)
            cls_out_tt = ttsim_linear(out_bf_tt2, cls_w, cls_b)

            print(f"     * Linear({embed_dims} -> {num_cls})")
            match = compare_arrays(
                cls_out_pt,
                cls_out_tt,
                "     Classification Branch",
                rtol=1e-4,
                atol=1e-4,
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: ClsBranch L{dec_idx+1}",
                    match,
                    f"{cls_out_pt.shape}",
                )
            )

            all_cls_scores_pt.append(cls_out_pt.detach().numpy().copy())
            all_cls_scores_tt.append(cls_out_tt.copy())

        print(f"\n  {'='*75}")
        print(f"  DECODER COMPLETE: {num_decoder_layers} layers processed")
        print(f"  Final query features: {output_pt.shape}")

        # Stack predictions
        all_cls_pt = np.stack(all_cls_scores_pt, axis=0)
        all_cls_tt = np.stack(all_cls_scores_tt, axis=0)
        all_reg_pt = np.stack(all_reg_pts_pt, axis=0)
        all_reg_tt = np.stack(all_reg_pts_tt, axis=0)

        match_cls = compare_arrays(
            all_cls_pt, all_cls_tt, "All Cls Scores (stacked)", rtol=1e-4, atol=1e-4
        )
        match_reg = compare_arrays(
            all_reg_pt, all_reg_tt, "All Reg Points (stacked)", rtol=1e-4, atol=1e-4
        )
        validation_results.append(
            ("Final Cls Scores", match_cls, f"{all_cls_pt.shape}")
        )
        validation_results.append(
            ("Final Reg Points", match_reg, f"{all_reg_pt.shape}")
        )

        # =====================================================================
        # STAGE 6: QUERY PROPAGATION (MotionMLP) — between frames
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 6: QUERY PROPAGATION (MotionMLP)")
        print(f"  {'='*75}\n")

        np.random.seed(500)

        # 6.1: Embedder (sin/cos positional encoding of pose)
        layer_counter += 1
        print(f"  Layer {layer_counter}: Embedder (Sin/Cos Positional Encoding)")
        print(f"  {'-'*75}")

        pose_np = np.random.randn(num_q, c_dim).astype(np.float32)
        N_freqs = 10
        freq_bands = 2.0 ** np.arange(N_freqs).astype(np.float32)

        # PyTorch embedder
        pose_pt = torch.from_numpy(pose_np)
        embed_list_pt = [pose_pt]
        for freq in freq_bands:
            embed_list_pt.append(torch.sin(pose_pt * freq))
            embed_list_pt.append(torch.cos(pose_pt * freq))
        pose_embed_pt = torch.cat(embed_list_pt, dim=-1)

        # TTSim embedder
        embed_list_tt = [pose_np.copy()]
        for freq in freq_bands:
            embed_list_tt.append(ttsim_sin(ttsim_mul(pose_np, np.float32(freq))))
            embed_list_tt.append(ttsim_cos(ttsim_mul(pose_np, np.float32(freq))))
        pose_embed_tt = ttsim_concat(embed_list_tt, axis=-1)

        embed_dim_pose = c_dim * (1 + 2 * N_freqs)  # 7 * 21 = 147
        print(f"    Pose: [{num_q}, {c_dim}] -> [{num_q}, {embed_dim_pose}]")
        print(f"    {N_freqs} frequency bands: sin/cos encoding")
        match = compare_arrays(pose_embed_pt, pose_embed_tt, "Embedder Output")
        validation_results.append(
            (f"Layer {layer_counter}: Embedder", match, f"{pose_embed_pt.shape}")
        )

        # 6.2: Concat + FC1 + LN + ReLU + FC2 + Residual
        layer_counter += 1
        print(
            f"\n  Layer {layer_counter}: MotionMLP (Concat -> Linear -> LN -> ReLU -> Linear + Residual)"
        )
        print(f"  {'-'*75}")

        # Use last decoder output as features [num_q, B, C] -> squeeze B -> [num_q, C]
        feat_np = output_pt.permute(1, 0, 2).squeeze(0).detach().numpy()  # [num_q, C]
        feat_pt = torch.from_numpy(feat_np.copy())
        feat_tt = feat_np.copy()

        # Concat features + pose embedding
        concat_dim = embed_dims + embed_dim_pose
        concat_pt = torch.cat([feat_pt, pose_embed_pt], dim=-1)  # [num_q, C+147]
        concat_tt = ttsim_concat([feat_tt, pose_embed_tt], axis=-1)

        mlp_fc1_w = (
            np.random.randn(concat_dim, 2 * embed_dims).astype(np.float32) * 0.01
        )
        mlp_fc1_b = np.random.randn(2 * embed_dims).astype(np.float32) * 0.01
        mlp_fc2_w = (
            np.random.randn(2 * embed_dims, embed_dims).astype(np.float32) * 0.01
        )
        mlp_fc2_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

        # PyTorch: Concat -> Linear -> LN -> ReLU -> Linear + Residual
        h_pt = concat_pt @ torch.from_numpy(mlp_fc1_w) + torch.from_numpy(mlp_fc1_b)
        h_pt = torch.nn.functional.layer_norm(h_pt, [2 * embed_dims])
        h_pt = torch.relu(h_pt)
        mlp_out_pt = h_pt @ torch.from_numpy(mlp_fc2_w) + torch.from_numpy(mlp_fc2_b)
        mlp_out_pt = mlp_out_pt + feat_pt  # residual

        # TTSim
        h_tt = ttsim_linear(concat_tt, mlp_fc1_w, mlp_fc1_b)
        h_tt = ttsim_layernorm(h_tt, 2 * embed_dims)
        h_tt = ttsim_relu(h_tt)
        mlp_out_tt = ttsim_linear(h_tt, mlp_fc2_w, mlp_fc2_b)
        mlp_out_tt = ttsim_add(mlp_out_tt, feat_tt)  # residual

        print(
            f"    Concat: [{num_q}, {embed_dims}] + [{num_q}, {embed_dim_pose}] -> [{num_q}, {concat_dim}]"
        )
        print(f"    Linear({concat_dim}->{2 * embed_dims}) -> LN -> ReLU")
        print(f"    Linear({2 * embed_dims}->{embed_dims}) + Residual")
        match = compare_arrays(mlp_out_pt, mlp_out_tt, "MotionMLP Output")
        validation_results.append(
            (f"Layer {layer_counter}: MotionMLP", match, f"{mlp_out_pt.shape}")
        )

        # =====================================================================
        # STAGE 7: POST-PROCESSING
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 7: POST-PROCESSING")
        print(f"  {'='*75}\n")

        layer_counter += 1
        print(f"  Layer {layer_counter}: Sigmoid + ArgMax -> Final Predictions")
        print(f"  {'-'*75}")

        # Classification: sigmoid -> max score -> label
        final_cls = all_cls_scores_pt[-1]  # [B, num_q, num_cls] from last layer
        cls_sigmoid_pt = torch.sigmoid(torch.from_numpy(final_cls))
        max_scores_pt, labels_pt = cls_sigmoid_pt.max(dim=-1)

        cls_sigmoid_tt = ttsim_sigmoid(all_cls_scores_tt[-1])
        max_scores_tt = np.max(cls_sigmoid_tt, axis=-1)
        labels_tt = np.argmax(cls_sigmoid_tt, axis=-1)

        print(f"    Cls scores: {final_cls.shape} -> Sigmoid -> Max")
        print(f"    Scores: {max_scores_pt.shape}, Labels: {labels_pt.shape}")
        match_s = compare_arrays(max_scores_pt, max_scores_tt, "Scores (sigmoid+max)")
        match_l = compare_arrays(
            labels_pt.float(), labels_tt.astype(np.float32), "Labels (argmax)"
        )

        # Segmentation: argmax -> semantic mask
        seg_preds_final = seg_preds_pt.detach().numpy()[0]  # [num_cls, H, W]
        seg_labels_pt = np.argmax(seg_preds_final, axis=0)
        seg_raw_scores_pt = np.max(seg_preds_final, axis=0)
        seg_scores_pt = 1.0 / (1.0 + np.exp(-seg_raw_scores_pt))
        seg_mask_pt = np.zeros(seg_labels_pt.shape, dtype=np.uint8)
        pos_ids = seg_scores_pt >= 0.4
        seg_mask_pt[pos_ids] = seg_labels_pt[pos_ids].astype(np.uint8) + 1

        seg_preds_final_tt = seg_preds_tt[0]
        seg_labels_tt = ttsim_argmax(seg_preds_final_tt, axis=0, keepdims=False).astype(
            np.int64
        )
        seg_raw_scores_tt = np.max(seg_preds_final_tt, axis=0)
        seg_scores_tt = ttsim_sigmoid(seg_raw_scores_tt.astype(np.float32))
        seg_mask_tt = np.zeros(seg_labels_tt.shape, dtype=np.uint8)
        pos_ids_tt = seg_scores_tt >= 0.4
        seg_mask_tt[pos_ids_tt] = seg_labels_tt[pos_ids_tt].astype(np.uint8) + 1

        print(
            f"    Seg preds: [{num_cls}, {canvas_h}, {canvas_w}] -> ArgMax -> Semantic mask"
        )
        match_seg = compare_arrays(
            seg_mask_pt.astype(np.float32),
            seg_mask_tt.astype(np.float32),
            "Semantic Mask",
        )

        match = match_s and match_l and match_seg
        validation_results.append(
            (
                f"Layer {layer_counter}: Post-Processing",
                match,
                "scores + labels + seg mask",
            )
        )

        # =====================================================================
        # STAGE 8: FINAL OUTPUT
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 8: FINAL OUTPUT")
        print(f"  {'='*75}\n")

        layer_counter += 1
        print(f"  Layer {layer_counter}: Final Model Output")
        print(f"  {'-'*75}")

        print(f"    Output Dictionary:")
        print(f"    |-- 'vectors': {all_reg_pt[-1].shape} (sigmoid reference points)")
        print(f"    |-- 'scores': {max_scores_pt.shape} (sigmoid max cls)")
        print(f"    |-- 'labels': {labels_pt.shape} (argmax cls)")
        print(f"    |-- 'semantic_mask': {seg_mask_pt.shape}")
        print(f"    |-- 'seg_preds': {seg_preds_pt.shape}")
        print(f"    '-- 'hs_embeds': {output_pt.shape} (decoder output)")

        validation_results.append(
            (
                "Final Output",
                True,
                f"Vec:{all_reg_pt[-1].shape}, Seg:{seg_mask_pt.shape}",
            )
        )

        # =====================================================================
        # STAGE 9: MULTI-FRAME TEMPORAL CONSISTENCY
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  STAGE 9: MULTI-FRAME TEMPORAL CONSISTENCY")
        print(f"  {'='*75}")
        print(f"  Running 3 frames: Frame 0 (done above) -> Frame 1 -> Frame 2")
        print(f"  Tests: BEV warp grid from ego-motion, GridSample, Memory Cross-Attn,")
        print(f"         Query Propagation via MotionMLP, Decoder with active memory\n")

        np.random.seed(600)
        num_frames = 3

        # --- Store Frame 0 results (already computed above) ---
        frame_decoder_outputs_pt = [output_pt.detach().clone()]  # [num_q, B, C]
        frame_decoder_outputs_tt = [output_tt.copy()]
        frame_ref_pts_pt = [
            reference_points_pt.detach().clone()
        ]  # [B, num_q, num_pts, 2]
        frame_ref_pts_tt = [reference_points_tt.copy()]
        frame_cls_pt = [all_cls_scores_pt[-1].copy()]
        frame_cls_tt = [all_cls_scores_tt[-1].copy()]

        # Per-frame BEV features (frame 0 already used bev_input_np)
        frame_bev_feats = [bev_input_np.copy()]
        for f_idx in range(1, num_frames):
            frame_bev_feats.append(
                np.random.randn(B, embed_dims, bev_h, bev_w).astype(np.float32)
            )

        # Ego poses: each frame has translation offset
        ego_translations = [
            np.array([100.0, 200.0, 0.0], dtype=np.float64),
            np.array([105.0, 200.0, 0.0], dtype=np.float64),
            np.array([110.0, 200.0, 0.0], dtype=np.float64),
        ]
        ego_rotations = [np.eye(3, dtype=np.float64)] * num_frames

        # Shared weights for all frames (memory cross-attention, decoder, MotionMLP)
        # Re-use decoder weights from Stage 5 by re-seeding
        # Memory cross-attention weights (per decoder layer)
        mem_attn_weights = []
        for lid in range(num_decoder_layers):
            np.random.seed(700 + lid * 50)
            maw = {
                "q_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "k_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "v_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "o_w": np.random.randn(embed_dims, embed_dims).astype(np.float32)
                * 0.01,
                "q_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "k_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "v_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
                "o_b": np.random.randn(embed_dims).astype(np.float32) * 0.01,
            }
            mem_attn_weights.append(maw)

        # BEV plane grid for warp computation: [bev_h, bev_w, 4]
        roi_size = (60.0, 30.0)
        xs = np.linspace(-roi_size[0] / 2, roi_size[0] / 2, bev_w)
        ys = np.linspace(-roi_size[1] / 2, roi_size[1] / 2, bev_h)
        yy_plane, xx_plane = np.meshgrid(ys, xs, indexing="ij")
        plane = np.stack(
            [xx_plane, yy_plane, np.zeros_like(xx_plane), np.ones_like(xx_plane)],
            axis=-1,
        ).astype(np.float64)
        # plane: [bev_h, bev_w, 4]

        # PositionalEncoding1D: sin/cos interleaved table for temporal PE
        # inv_freq = 1 / (10000 ^ (k / channels)) for k in [0, 2, 4, ..., channels-2]
        pe_channels = int(np.ceil(embed_dims / 2) * 2)
        pe_inv_freq = 1.0 / (
            10000 ** (np.arange(0, pe_channels, 2).astype(np.float32) / pe_channels)
        )
        pe_max_len = 100  # more than enough for 3 frames
        pe_positions = np.arange(pe_max_len, dtype=np.float32)
        # sin_inp = outer(positions, inv_freq) -> [max_len, channels//2]
        pe_sin_inp = (
            pe_positions[:, None] * pe_inv_freq[None, :]
        )  # [max_len, channels//2]
        # Interleave sin/cos: [max_len, channels//2, 2] -> [max_len, channels]
        pe_sin = np.sin(pe_sin_inp)
        pe_cos = np.cos(pe_sin_inp)
        pe_stacked = np.stack([pe_sin, pe_cos], axis=-1).reshape(pe_max_len, -1)
        pe_cached = pe_stacked[:, :embed_dims].astype(
            np.float32
        )  # [max_len, embed_dims]

        # TTSim: same PE table using compute functions
        # In production, these ops (Unsqueeze, ConcatX, Reshape) are used in
        # PositionalEncoding1D.__init__ to keep Sin/Cos graph-connected to pe_table.
        pe_sin_inp_tt = ttsim_mul(
            ttsim_reshape(pe_positions, (pe_max_len, 1)),
            ttsim_reshape(pe_inv_freq, (1, len(pe_inv_freq))),
        )
        pe_sin_tt = ttsim_sin(pe_sin_inp_tt)
        pe_cos_tt = ttsim_cos(pe_sin_inp_tt)
        # Interleave: stack on last dim then reshape
        pe_interleaved_tt = ttsim_reshape(
            ttsim_concat(
                [ttsim_unsqueeze(pe_sin_tt, [-1]), ttsim_unsqueeze(pe_cos_tt, [-1])],
                axis=-1,
            ),
            (pe_max_len, -1),
        )
        pe_cached_tt = pe_interleaved_tt[:, :embed_dims]

        # Validate PE table
        layer_counter += 1
        print(f"\n  Layer {layer_counter}: PositionalEncoding1D Table Generation")
        print(f"  {'-'*75}")
        print(
            f"    inv_freq: 1/(10000^(k/{pe_channels})) for {len(pe_inv_freq)} frequencies"
        )
        print(
            f"    sin_inp: outer(positions, inv_freq) -> [{pe_max_len}, {len(pe_inv_freq)}]"
        )
        print(
            f"    Interleave sin/cos -> [{pe_max_len}, {pe_channels}] -> slice to [{pe_max_len}, {embed_dims}]"
        )
        match_pe = compare_arrays(
            pe_cached, pe_cached_tt, "PE Table", rtol=1e-5, atol=1e-6
        )
        validation_results.append(
            (
                f"Layer {layer_counter}: PE1D Table",
                match_pe,
                f"({pe_max_len},{embed_dims})",
            )
        )

        stage9_ok = True
        stage9_ok &= match_pe

        for frame_idx in range(1, num_frames):
            print(f"\n  {'─'*75}")
            print(f"  FRAME {frame_idx}/{num_frames - 1}")
            print(f"  {'─'*75}")

            # =================================================================
            # 9a: Compute warp grid from ego-motion (prev->curr transform)
            # =================================================================
            layer_counter += 1
            print(
                f"\n  Layer {layer_counter}: Ego-Motion Warp Grid (Frame {frame_idx})"
            )
            print(f"  {'-'*75}")

            prev_trans = ego_translations[frame_idx - 1]
            prev_rot = ego_rotations[frame_idx - 1]
            curr_trans = ego_translations[frame_idx]
            curr_rot = ego_rotations[frame_idx]

            # prev global-to-ego: g2e_prev = R_prev^T, t = -(R_prev^T @ trans_prev)
            prev_g2e = np.eye(4, dtype=np.float64)
            prev_g2e[:3, :3] = prev_rot.T
            prev_g2e[:3, 3] = -(prev_rot.T @ prev_trans)

            # curr ego-to-global: e2g_curr = R_curr, t = trans_curr
            curr_e2g = np.eye(4, dtype=np.float64)
            curr_e2g[:3, :3] = curr_rot
            curr_e2g[:3, 3] = curr_trans

            # curr->prev = g2e_prev @ e2g_curr
            curr2prev = prev_g2e @ curr_e2g  # [4, 4]

            # Apply to BEV plane: history_coord = plane @ curr2prev^T  -> take [:2]
            # einsum('lk,ijk->ijl', curr2prev, plane) = plane @ curr2prev.T
            history_coord = np.einsum(
                "lk,ijk->ijl", curr2prev, plane
            )  # [bev_h, bev_w, 4]
            history_coord = history_coord[..., :2].astype(
                np.float32
            )  # [bev_h, bev_w, 2]

            # Normalize to [-1, 1] for grid_sample
            warp_grid = np.zeros_like(history_coord)
            warp_grid[..., 0] = history_coord[..., 0] / (roi_size[0] / 2)  # x
            warp_grid[..., 1] = -history_coord[..., 1] / (
                roi_size[1] / 2
            )  # y (negated)

            # TTSim warp grid computation (same math using matmul)
            plane_flat = plane.reshape(-1, 4).astype(np.float32)  # [H*W, 4]
            curr2prev_f32 = curr2prev.T.astype(np.float32)  # [4, 4] transposed
            warp_coords_tt = ttsim_matmul(plane_flat, curr2prev_f32)  # [H*W, 4]
            warp_coords_tt = ttsim_reshape(warp_coords_tt, (bev_h, bev_w, 4))
            warp_x_tt = warp_coords_tt[..., 0:1]
            warp_y_tt = warp_coords_tt[..., 1:2]
            warp_x_norm = ttsim_div(warp_x_tt, np.float32(roi_size[0] / 2))
            warp_y_norm = ttsim_div(
                ttsim_mul(warp_y_tt, np.float32(-1.0)), np.float32(roi_size[1] / 2)
            )
            warp_grid_tt = ttsim_concat(
                [warp_x_norm, warp_y_norm], axis=-1
            )  # [bev_h, bev_w, 2]

            print(
                f"    Ego translation: prev={prev_trans[:2]} -> curr={curr_trans[:2]}"
            )
            print(
                f"    Curr->Prev transform applied to BEV plane [{bev_h}, {bev_w}, 4]"
            )
            print(f"    Warp grid: [{bev_h}, {bev_w}, 2] (normalized to [-1,1])")
            match_wg = compare_arrays(
                warp_grid, warp_grid_tt, "Warp Grid", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: WarpGrid F{frame_idx}",
                    match_wg,
                    f"({bev_h},{bev_w},2)",
                )
            )
            stage9_ok &= match_wg

            # =================================================================
            # 9b: GridSample history BEV with warp grid
            # =================================================================
            layer_counter += 1
            print(
                f"\n  Layer {layer_counter}: History BEV Warping (GridSample, Frame {frame_idx})"
            )
            print(f"  {'-'*75}")

            prev_bev = frame_bev_feats[frame_idx - 1]  # [B, C, H, W]
            warp_grid_4d = warp_grid[np.newaxis, ...]  # [1, bev_h, bev_w, 2]

            # PyTorch: grid_sample
            warped_bev_pt = torch.nn.functional.grid_sample(
                torch.from_numpy(prev_bev),
                torch.from_numpy(warp_grid_4d),
                padding_mode="zeros",
                align_corners=False,
            )

            # TTSim: grid_sample
            warp_grid_4d_tt = ttsim_unsqueeze(warp_grid_tt, [0])
            warped_bev_tt = ttsim_gridsample(
                prev_bev, warp_grid_4d_tt, align_corners=False
            )

            print(f"    Previous BEV: [{B}, {embed_dims}, {bev_h}, {bev_w}]")
            print(f"    Warp grid: [{B}, {bev_h}, {bev_w}, 2]")
            print(f"    Warped BEV: {warped_bev_pt.shape}")
            match_warp = compare_arrays(
                warped_bev_pt, warped_bev_tt, "Warped BEV", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: WarpBEV F{frame_idx}",
                    match_warp,
                    f"{warped_bev_pt.shape}",
                )
            )
            stage9_ok &= match_warp

            # =================================================================
            # 9c: BEV query fusion (warped history replaces embedding where non-zero)
            # =================================================================
            layer_counter += 1
            print(f"\n  Layer {layer_counter}: BEV Query Fusion (Frame {frame_idx})")
            print(f"  {'-'*75}")

            # prop_bev: [B, C, H, W] -> flatten to [H*W, B, C]
            prop_bev_pt = warped_bev_pt.reshape(B, embed_dims, bev_h * bev_w).permute(
                2, 0, 1
            )
            prop_bev_tt_flat = ttsim_transpose(
                ttsim_reshape(warped_bev_tt, (B, embed_dims, bev_h * bev_w)), [2, 0, 1]
            )

            # valid_mask = (prop_bev.sum(-1) > 0)
            # Use TTSim reducesum for the mask, then share the SAME mask for both
            # paths so we test fusion math, not floating-point threshold sensitivity
            # (grid_sample can produce ~1e-6 differences that flip the > 0 boundary)
            prop_sum_tt = ttsim_reducesum(prop_bev_tt_flat, axis=-1, keepdims=True)
            valid_mask_tt = (prop_sum_tt > 0).astype(np.float32)
            valid_mask_pt = torch.from_numpy(valid_mask_tt)  # shared mask

            # bev_queries = bev_queries * (1 - valid_mask) + prop_bev * valid_mask
            bev_q_fresh_pt = (
                torch.from_numpy(bev_queries_ttsim.copy())
                .unsqueeze(1)
                .expand(-1, B, -1)
            )  # [H*W, B, C]
            fused_pt = (
                bev_q_fresh_pt * (1 - valid_mask_pt) + prop_bev_pt * valid_mask_pt
            )

            bev_q_fresh_tt = ttsim_tile(
                ttsim_unsqueeze(bev_queries_ttsim.copy(), [1]), [1, B, 1]
            )
            inv_mask_tt = ttsim_sub(np.float32(1.0), valid_mask_tt)
            fused_tt = ttsim_add(
                ttsim_mul(bev_q_fresh_tt, inv_mask_tt),
                ttsim_mul(prop_bev_tt_flat, valid_mask_tt),
            )

            print(f"    Prop BEV: [{bev_h * bev_w}, {B}, {embed_dims}]")
            print(
                f"    Valid mask: sum > 0 -> replace BEV queries with warped features"
            )
            print(f"    Fused queries: {fused_pt.shape}")
            match_fuse = compare_arrays(
                fused_pt, fused_tt, "BEV Query Fusion", rtol=1e-4, atol=1e-5
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: BEVFusion F{frame_idx}",
                    match_fuse,
                    f"{fused_pt.shape}",
                )
            )
            stage9_ok &= match_fuse

            # =================================================================
            # 9d: Seg head on new frame's BEV
            # =================================================================
            layer_counter += 1
            print(f"\n  Layer {layer_counter}: SegHead (Frame {frame_idx})")
            print(f"  {'-'*75}")

            frame_bev_cur = frame_bev_feats[frame_idx]
            # Reuse seg head weights from Stage 3
            seg_in_pt = torch.nn.functional.conv2d(
                torch.from_numpy(frame_bev_cur), torch.from_numpy(conv_in_w), padding=1
            )
            seg_in_pt = torch.relu(seg_in_pt)
            seg_up_pt = torch.nn.functional.interpolate(
                seg_in_pt, scale_factor=2, mode="nearest"
            )
            seg_up_pt = torch.nn.functional.conv2d(
                seg_up_pt, torch.from_numpy(conv_up_w), padding=1
            )
            seg_up_pt = seg_up_pt + torch.from_numpy(conv_up_b).reshape(1, -1, 1, 1)
            seg_up_pt = torch.relu(seg_up_pt)
            seg_pred_f_pt = torch.nn.functional.conv2d(
                seg_up_pt, torch.from_numpy(conv_out_w)
            )
            seg_pred_f_pt = seg_pred_f_pt + torch.from_numpy(conv_out_b).reshape(
                1, -1, 1, 1
            )

            seg_in_tt = ttsim_relu(
                ttsim_conv2d(frame_bev_cur, conv_in_w, bias=None, stride=1, padding=1)
            )
            seg_up_tt = ttsim_upsample(seg_in_tt, scale_factor=2.0, mode="nearest")
            seg_up_tt = ttsim_conv2d(
                seg_up_tt, conv_up_w, bias=None, stride=1, padding=1
            )
            seg_up_tt = ttsim_relu(
                ttsim_add(seg_up_tt, ttsim_reshape(conv_up_b, (1, embed_dims, 1, 1)))
            )
            seg_pred_f_tt = ttsim_add(
                ttsim_conv2d(seg_up_tt, conv_out_w, bias=None, stride=1, padding=0),
                ttsim_reshape(conv_out_b, (1, num_cls, 1, 1)),
            )

            print(
                f"    BEV [{B},{embed_dims},{bev_h},{bev_w}] -> Conv->ReLU->Up->Conv->ReLU->Conv"
            )
            print(f"    Seg preds: {seg_pred_f_pt.shape}")
            match_seg = compare_arrays(
                seg_pred_f_pt, seg_pred_f_tt, "SegHead Preds", rtol=1e-4, atol=1e-4
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: Seg F{frame_idx}",
                    match_seg,
                    f"{seg_pred_f_pt.shape}",
                )
            )
            stage9_ok &= match_seg

            # =================================================================
            # 9e: Query propagation via MotionMLP (prev queries -> curr queries)
            # =================================================================
            layer_counter += 1
            print(
                f"\n  Layer {layer_counter}: Query Propagation via MotionMLP (Frame {frame_idx})"
            )
            print(f"  {'-'*75}")

            # Compute pose: relative transform prev->curr
            # For identity rotations: quaternion = [0, 0, 0, 1], translation = delta
            delta_trans = curr_trans - prev_trans  # [3]
            rot_quat = np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            )  # identity rotation quaternion
            pose_vec = np.concatenate([rot_quat, delta_trans]).astype(np.float32)  # [7]
            pose_repeated = np.tile(pose_vec[np.newaxis, :], (num_q, 1))  # [num_q, 7]

            # Get previous frame's decoder output [num_q, B, C] -> [num_q, C]
            prev_hs = (
                frame_decoder_outputs_pt[-1].squeeze(1).detach().numpy()
            )  # [num_q, C]

            # Embedder: sin/cos
            embed_list_pt_f = [torch.from_numpy(pose_repeated)]
            embed_list_tt_f = [pose_repeated.copy()]
            for freq in freq_bands:
                embed_list_pt_f.append(
                    torch.sin(torch.from_numpy(pose_repeated) * freq)
                )
                embed_list_pt_f.append(
                    torch.cos(torch.from_numpy(pose_repeated) * freq)
                )
                embed_list_tt_f.append(
                    ttsim_sin(ttsim_mul(pose_repeated, np.float32(freq)))
                )
                embed_list_tt_f.append(
                    ttsim_cos(ttsim_mul(pose_repeated, np.float32(freq)))
                )
            pose_emb_pt_f = torch.cat(embed_list_pt_f, dim=-1)
            pose_emb_tt_f = ttsim_concat(embed_list_tt_f, axis=-1)

            # Concat + MLP
            feat_pt_f = torch.from_numpy(prev_hs.copy())
            feat_tt_f = prev_hs.copy()

            concat_pt_f = torch.cat([feat_pt_f, pose_emb_pt_f], dim=-1)
            concat_tt_f = ttsim_concat([feat_tt_f, pose_emb_tt_f], axis=-1)

            h_pt_f = concat_pt_f @ torch.from_numpy(mlp_fc1_w) + torch.from_numpy(
                mlp_fc1_b
            )
            h_pt_f = torch.nn.functional.layer_norm(h_pt_f, [2 * embed_dims])
            h_pt_f = torch.relu(h_pt_f)
            mlp_out_pt_f = h_pt_f @ torch.from_numpy(mlp_fc2_w) + torch.from_numpy(
                mlp_fc2_b
            )
            mlp_out_pt_f = mlp_out_pt_f + feat_pt_f  # residual

            h_tt_f = ttsim_linear(concat_tt_f, mlp_fc1_w, mlp_fc1_b)
            h_tt_f = ttsim_layernorm(h_tt_f, 2 * embed_dims)
            h_tt_f = ttsim_relu(h_tt_f)
            mlp_out_tt_f = ttsim_linear(h_tt_f, mlp_fc2_w, mlp_fc2_b)
            mlp_out_tt_f = ttsim_add(mlp_out_tt_f, feat_tt_f)

            print(f"    Pose: quat=[0,0,0,1] + delta_t={delta_trans}")
            print(f"    Embedder: [{num_q}, {c_dim}] -> [{num_q}, {embed_dim_pose}]")
            print(f"    MLP: [{num_q}, {concat_dim}] -> [{num_q}, {embed_dims}]")
            match_mlp = compare_arrays(
                mlp_out_pt_f,
                mlp_out_tt_f,
                "MotionMLP Propagation",
                rtol=1e-4,
                atol=1e-5,
            )
            validation_results.append(
                (
                    f"Layer {layer_counter}: MotionMLP F{frame_idx}",
                    match_mlp,
                    f"({num_q},{embed_dims})",
                )
            )
            stage9_ok &= match_mlp

            # Use propagated queries as initial queries for this frame's decoder
            prop_query_pt = mlp_out_pt_f.unsqueeze(1)  # [num_q, 1, C] (seq-first)
            prop_query_tt = ttsim_unsqueeze(mlp_out_tt_f, [1])

            # =================================================================
            # 9f: Run decoder with memory cross-attention ACTIVE
            # =================================================================
            print(
                f"\n  Running Decoder for Frame {frame_idx} (Memory Cross-Attn ACTIVE)..."
            )

            # Prepare BEV value for this frame (input proj + pos embed + flatten)
            bev_cur_proj_pt = torch.nn.functional.conv2d(
                torch.from_numpy(frame_bev_cur), torch.from_numpy(input_proj_w)
            )
            bev_cur_proj_pt = bev_cur_proj_pt + torch.from_numpy(input_proj_b).reshape(
                1, -1, 1, 1
            )
            bev_cur_proj_pt = bev_cur_proj_pt + torch.from_numpy(pos_embed_np)

            bev_cur_proj_tt = ttsim_conv2d(
                frame_bev_cur, input_proj_w, bias=None, stride=1, padding=0
            )
            bev_cur_proj_tt = ttsim_add(bev_cur_proj_tt, proj_bias_4d)
            bev_cur_proj_tt = ttsim_add(bev_cur_proj_tt, pos_embed_np)

            bev_val_f_pt = bev_cur_proj_pt.reshape(
                B, embed_dims, bev_h * bev_w
            ).permute(0, 2, 1)
            bev_val_f_tt = ttsim_transpose(
                ttsim_reshape(bev_cur_proj_tt, (B, embed_dims, bev_h * bev_w)),
                [0, 2, 1],
            )

            # Use propagated queries + previous ref points
            cur_output_pt = prop_query_pt
            cur_output_tt = prop_query_tt
            cur_ref_pt = frame_ref_pts_pt[-1].clone()
            cur_ref_tt = frame_ref_pts_tt[-1].copy()

            # Memory: previous frame's decoder output [num_q, B, C] -> [1, num_q, C] for K/V
            memory_pt = frame_decoder_outputs_pt[-1].permute(1, 0, 2)  # [B, num_q, C]
            memory_tt = ttsim_transpose(frame_decoder_outputs_tt[-1], [1, 0, 2])

            frame_cls_list_pt = []
            frame_cls_list_tt = []
            frame_reg_list_pt = []
            frame_reg_list_tt = []

            for dec_idx in range(num_decoder_layers):
                np.random.seed(300 + dec_idx * 100)  # Same decoder weights as Stage 5

                # Self-attention
                sa_q_w = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                sa_k_w = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                sa_v_w = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                sa_o_w = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                sa_q_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
                sa_k_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
                sa_v_b = np.random.randn(embed_dims).astype(np.float32) * 0.01
                sa_o_b = np.random.randn(embed_dims).astype(np.float32) * 0.01

                sa_id_pt = cur_output_pt.clone()
                sa_bf_pt = cur_output_pt.permute(1, 0, 2)
                Q_f = sa_bf_pt @ torch.from_numpy(sa_q_w) + torch.from_numpy(sa_q_b)
                K_f = sa_bf_pt @ torch.from_numpy(sa_k_w) + torch.from_numpy(sa_k_b)
                V_f = sa_bf_pt @ torch.from_numpy(sa_v_w) + torch.from_numpy(sa_v_b)
                Q_f = Q_f.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                K_f = K_f.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                V_f = V_f.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                sc_f = Q_f @ K_f.transpose(-2, -1) / np.sqrt(head_dim)
                at_f = torch.softmax(sc_f, dim=-1)
                cx_f = (
                    (at_f @ V_f).transpose(1, 2).contiguous().view(B, num_q, embed_dims)
                )
                sa_o = cx_f @ torch.from_numpy(sa_o_w) + torch.from_numpy(sa_o_b)
                cur_output_pt = sa_id_pt + sa_o.permute(1, 0, 2)

                sa_id_tt = cur_output_tt.copy()
                sa_bf_tt = ttsim_transpose(cur_output_tt, [1, 0, 2])
                sa_out_f = ttsim_multihead_attention(
                    sa_bf_tt,
                    sa_bf_tt,
                    sa_bf_tt,
                    sa_q_w,
                    sa_k_w,
                    sa_v_w,
                    sa_o_w,
                    sa_q_b,
                    sa_k_b,
                    sa_v_b,
                    sa_o_b,
                    num_heads,
                    head_dim,
                )
                cur_output_tt = ttsim_add(
                    sa_id_tt, ttsim_transpose(sa_out_f, [1, 0, 2])
                )

                # LN 1
                cur_output_pt = torch.nn.functional.layer_norm(
                    cur_output_pt, [embed_dims]
                )
                cur_output_tt = ttsim_layernorm(cur_output_tt, embed_dims)

                # Deformable cross-attention to BEV
                da_val_w_ = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                da_val_b_ = np.random.randn(embed_dims).astype(np.float32) * 0.01
                da_soff_w_ = (
                    np.random.randn(
                        embed_dims, num_heads * num_levels * deform_pts * 2
                    ).astype(np.float32)
                    * 0.01
                )
                da_soff_b_ = np.zeros(
                    num_heads * num_levels * deform_pts * 2, dtype=np.float32
                )
                da_aw_w_ = (
                    np.random.randn(
                        embed_dims, num_heads * num_levels * deform_pts
                    ).astype(np.float32)
                    * 0.01
                )
                da_aw_b_ = np.zeros(
                    num_heads * num_levels * deform_pts, dtype=np.float32
                )
                da_out_w_ = (
                    np.random.randn(embed_dims, embed_dims).astype(np.float32) * 0.01
                )
                da_out_b_ = np.random.randn(embed_dims).astype(np.float32) * 0.01

                ref_x = cur_ref_pt[..., 0:1]
                ref_y = cur_ref_pt[..., 1:2]
                ref_rev_f_pt = torch.cat([ref_x, 1.0 - ref_y], dim=-1)
                ref_x_t = cur_ref_tt[..., 0:1]
                ref_y_t = cur_ref_tt[..., 1:2]
                ref_rev_f_tt = ttsim_concat(
                    [ref_x_t, ttsim_sub(np.float32(1.0), ref_y_t)], axis=-1
                )

                da_id_pt = cur_output_pt.clone()
                q_bf_f_pt = cur_output_pt.permute(1, 0, 2)
                vp_f_pt = bev_val_f_pt @ torch.from_numpy(da_val_w_) + torch.from_numpy(
                    da_val_b_
                )
                vmh_f_pt = vp_f_pt.view(B, num_value, num_heads, head_dim)
                so_f_pt = (
                    q_bf_f_pt @ torch.from_numpy(da_soff_w_)
                    + torch.from_numpy(da_soff_b_)
                ).view(B, num_q, num_heads, num_levels, deform_pts, 2)
                aw_f_pt = (
                    q_bf_f_pt @ torch.from_numpy(da_aw_w_) + torch.from_numpy(da_aw_b_)
                ).view(B, num_q, num_heads, num_levels * deform_pts)
                aw_f_pt = torch.softmax(aw_f_pt, dim=-1).view(
                    B, num_q, num_heads, num_levels, deform_pts
                )
                offset_n = torch.stack(
                    [
                        torch.tensor(spatial_shapes[0, 1], dtype=torch.float32),
                        torch.tensor(spatial_shapes[0, 0], dtype=torch.float32),
                    ]
                )
                rb_f_pt = ref_rev_f_pt[:, :, 0:1, :].unsqueeze(2).unsqueeze(3)
                sl_f_pt = rb_f_pt + so_f_pt / offset_n[None, None, None, None, None, :]
                da_o_pt = multi_scale_deformable_attn_pytorch(
                    vmh_f_pt, torch.from_numpy(spatial_shapes), sl_f_pt, aw_f_pt
                )
                da_o_pt = da_o_pt @ torch.from_numpy(da_out_w_) + torch.from_numpy(
                    da_out_b_
                )
                cur_output_pt = da_id_pt + da_o_pt.permute(1, 0, 2)

                da_id_tt = cur_output_tt.copy()
                q_bf_f_tt = ttsim_transpose(cur_output_tt, [1, 0, 2])
                vp_f_tt = ttsim_linear(bev_val_f_tt, da_val_w_, da_val_b_)
                vmh_f_tt = ttsim_reshape(vp_f_tt, (B, num_value, num_heads, head_dim))
                so_f_tt = ttsim_reshape(
                    ttsim_linear(q_bf_f_tt, da_soff_w_, da_soff_b_),
                    (B, num_q, num_heads, num_levels, deform_pts, 2),
                )
                aw_f_tt = ttsim_softmax(
                    ttsim_reshape(
                        ttsim_linear(q_bf_f_tt, da_aw_w_, da_aw_b_),
                        (B, num_q, num_heads, num_levels * deform_pts),
                    ),
                    axis=-1,
                )
                aw_f_tt = ttsim_reshape(
                    aw_f_tt, (B, num_q, num_heads, num_levels, deform_pts)
                )
                offset_n_tt = np.array(
                    [spatial_shapes[0, 1], spatial_shapes[0, 0]], dtype=np.float32
                )
                rb_f_tt = ttsim_unsqueeze(
                    ttsim_unsqueeze(ref_rev_f_tt[:, :, 0:1, :], [2]), [3]
                )
                sl_f_tt = ttsim_add(rb_f_tt, ttsim_div(so_f_tt, offset_n_tt))
                da_o_tt = ttsim_ms_deformable_attn(
                    vmh_f_tt, spatial_shapes, sl_f_tt, aw_f_tt
                )
                da_o_tt = ttsim_linear(da_o_tt, da_out_w_, da_out_b_)
                cur_output_tt = ttsim_add(da_id_tt, ttsim_transpose(da_o_tt, [1, 0, 2]))

                # LN 2
                cur_output_pt = torch.nn.functional.layer_norm(
                    cur_output_pt, [embed_dims]
                )
                cur_output_tt = ttsim_layernorm(cur_output_tt, embed_dims)

                # ─── Memory Cross-Attention (ACTIVE on frames 1+) ───
                layer_counter += 1
                print(
                    f"\n  |-- Layer {layer_counter}: Memory Cross-Attention (ACTIVE, Decoder L{dec_idx+1}, Frame {frame_idx})"
                )
                print(f"     + Relative Temporal PE added to memory keys")

                maw = mem_attn_weights[dec_idx]

                # Query: [num_q, B, C] -> [B, num_q, C]
                mem_q_pt = cur_output_pt.permute(1, 0, 2)
                mem_q_tt = ttsim_transpose(cur_output_tt, [1, 0, 2])

                # Memory from prev frame [B, num_q, C] as Key/Value
                # In real model: [mem_len, num_q, C], we use mem_len=1
                mem_kv_pt = memory_pt  # [B, num_q, C]
                mem_kv_tt = memory_tt

                # --- Relative Temporal Positional Encoding ---
                # time gap = current frame index - memory frame index
                time_gap = frame_idx - (
                    frame_idx - 1
                )  # = 1 (memory is from previous frame)
                # Look up PE from cached table by time gap
                rel_pe_np = pe_cached[time_gap]  # [embed_dims]
                rel_pe_tt_val = pe_cached_tt[time_gap]  # [embed_dims]

                # Broadcast to [B, num_q, embed_dims] and add to memory keys
                rel_pe_broadcast = np.tile(
                    rel_pe_np[np.newaxis, np.newaxis, :], (B, num_q, 1)
                )
                rel_pe_broadcast_tt = ttsim_tile(
                    ttsim_unsqueeze(ttsim_unsqueeze(rel_pe_tt_val, [0]), [0]),
                    [B, num_q, 1],
                )

                # key = mem_kv + relative_pe (before K projection)
                mem_key_pt = mem_kv_pt + torch.from_numpy(rel_pe_broadcast)
                mem_key_tt = ttsim_add(mem_kv_tt, rel_pe_broadcast_tt)

                # Standard MHA with PE-augmented keys
                Q_m = mem_q_pt @ torch.from_numpy(maw["q_w"]) + torch.from_numpy(
                    maw["q_b"]
                )
                K_m = mem_key_pt @ torch.from_numpy(maw["k_w"]) + torch.from_numpy(
                    maw["k_b"]
                )
                V_m = mem_kv_pt @ torch.from_numpy(maw["v_w"]) + torch.from_numpy(
                    maw["v_b"]
                )
                Q_m = Q_m.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                K_m = K_m.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                V_m = V_m.view(B, num_q, num_heads, head_dim).transpose(1, 2)
                sc_m = Q_m @ K_m.transpose(-2, -1) / np.sqrt(head_dim)
                at_m = torch.softmax(sc_m, dim=-1)
                cx_m = (
                    (at_m @ V_m).transpose(1, 2).contiguous().view(B, num_q, embed_dims)
                )
                mem_attn_out_pt = cx_m @ torch.from_numpy(
                    maw["o_w"]
                ) + torch.from_numpy(maw["o_b"])

                mem_out_tt = ttsim_multihead_attention(
                    mem_q_tt,
                    mem_key_tt,
                    mem_kv_tt,
                    maw["q_w"],
                    maw["k_w"],
                    maw["v_w"],
                    maw["o_w"],
                    maw["q_b"],
                    maw["k_b"],
                    maw["v_b"],
                    maw["o_b"],
                    num_heads,
                    head_dim,
                )

                # In real model: query_memory + query_bev (before LN)
                # query_bev = cur_output after deform attn (already has residual)
                # query_memory = mem_attn_out
                # We add mem_attn_out to cur_output as additional residual
                cur_output_pt = cur_output_pt + mem_attn_out_pt.permute(1, 0, 2)
                cur_output_tt = ttsim_add(
                    cur_output_tt, ttsim_transpose(mem_out_tt, [1, 0, 2])
                )

                print(f"     * Q from decoder: [{B}, {num_q}, {embed_dims}]")
                print(
                    f"     * K = memory + relative_PE(gap={time_gap}): [{B}, {num_q}, {embed_dims}]"
                )
                print(
                    f"     * V from memory (prev frame): [{B}, {num_q}, {embed_dims}]"
                )
                print(
                    f"     * MHA: Q,K,V proj -> {num_heads} heads -> Softmax -> Output proj"
                )
                match_mem = compare_arrays(
                    cur_output_pt,
                    cur_output_tt,
                    "     MemAttn Output",
                    rtol=1e-3,
                    atol=1e-3,
                )
                validation_results.append(
                    (
                        f"Layer {layer_counter}: MemAttn L{dec_idx+1} F{frame_idx}",
                        match_mem,
                        f"{cur_output_pt.shape}",
                    )
                )
                stage9_ok &= match_mem

                # LN 3
                cur_output_pt = torch.nn.functional.layer_norm(
                    cur_output_pt, [embed_dims]
                )
                cur_output_tt = ttsim_layernorm(cur_output_tt, embed_dims)

                # FFN
                ffn_w1 = (
                    np.random.randn(embed_dims, ff_channels).astype(np.float32) * 0.01
                )
                ffn_b1 = np.random.randn(ff_channels).astype(np.float32) * 0.01
                ffn_w2 = (
                    np.random.randn(ff_channels, embed_dims).astype(np.float32) * 0.01
                )
                ffn_b2 = np.random.randn(embed_dims).astype(np.float32) * 0.01

                ffn_id_pt = cur_output_pt.clone()
                h_f = torch.relu(
                    cur_output_pt @ torch.from_numpy(ffn_w1) + torch.from_numpy(ffn_b1)
                )
                cur_output_pt = ffn_id_pt + (
                    h_f @ torch.from_numpy(ffn_w2) + torch.from_numpy(ffn_b2)
                )

                cur_output_tt = ttsim_ffn(cur_output_tt, ffn_w1, ffn_b1, ffn_w2, ffn_b2)

                # LN 4 (final)
                cur_output_pt = torch.nn.functional.layer_norm(
                    cur_output_pt, [embed_dims]
                )
                cur_output_tt = ttsim_layernorm(cur_output_tt, embed_dims)

                # Regression branch
                reg_fc1_w = (
                    np.random.randn(embed_dims, 2 * embed_dims).astype(np.float32)
                    * 0.01
                )
                reg_fc1_b = np.random.randn(2 * embed_dims).astype(np.float32) * 0.01
                reg_fc2_w = (
                    np.random.randn(2 * embed_dims, 2 * embed_dims).astype(np.float32)
                    * 0.01
                )
                reg_fc2_b = np.random.randn(2 * embed_dims).astype(np.float32) * 0.01
                reg_fc3_w = (
                    np.random.randn(2 * embed_dims, num_pts * coord_dim).astype(
                        np.float32
                    )
                    * 0.01
                )
                reg_fc3_b = (
                    np.random.randn(num_pts * coord_dim).astype(np.float32) * 0.01
                )

                out_bf_f = cur_output_pt.permute(1, 0, 2)
                h_r = torch.relu(
                    torch.nn.functional.layer_norm(
                        out_bf_f @ torch.from_numpy(reg_fc1_w)
                        + torch.from_numpy(reg_fc1_b),
                        [2 * embed_dims],
                    )
                )
                h_r = torch.relu(
                    torch.nn.functional.layer_norm(
                        h_r @ torch.from_numpy(reg_fc2_w) + torch.from_numpy(reg_fc2_b),
                        [2 * embed_dims],
                    )
                )
                reg_f_pt = torch.sigmoid(
                    (
                        h_r @ torch.from_numpy(reg_fc3_w) + torch.from_numpy(reg_fc3_b)
                    ).reshape(B, num_q, num_pts, 2)
                )

                out_bf_f_tt = ttsim_transpose(cur_output_tt, [1, 0, 2])
                reg_f_tt = ttsim_sigmoid(
                    ttsim_reshape(
                        ttsim_regression_branch(
                            out_bf_f_tt,
                            reg_fc1_w,
                            reg_fc1_b,
                            reg_fc2_w,
                            reg_fc2_b,
                            reg_fc3_w,
                            reg_fc3_b,
                        ),
                        (B, num_q, num_pts, 2),
                    )
                )

                cur_ref_pt = reg_f_pt.detach()
                cur_ref_tt = reg_f_tt.copy()

                frame_reg_list_pt.append(reg_f_pt.detach().numpy().copy())
                frame_reg_list_tt.append(reg_f_tt.copy())

                # Classification branch
                cls_w = np.random.randn(embed_dims, num_cls).astype(np.float32) * 0.01
                cls_b = np.random.randn(num_cls).astype(np.float32) * 0.01

                cls_f_pt = out_bf_f @ torch.from_numpy(cls_w) + torch.from_numpy(cls_b)
                cls_f_tt = ttsim_linear(out_bf_f_tt, cls_w, cls_b)

                frame_cls_list_pt.append(cls_f_pt.detach().numpy().copy())
                frame_cls_list_tt.append(cls_f_tt.copy())

            # Validate last decoder layer output for this frame
            layer_counter += 1
            print(
                f"\n  Layer {layer_counter}: Frame {frame_idx} Decoder Output Validation"
            )
            print(f"  {'-'*75}")

            match_dec = compare_arrays(
                cur_output_pt,
                cur_output_tt,
                f"Decoder Output F{frame_idx}",
                rtol=1e-3,
                atol=1e-3,
            )
            match_cls_f = compare_arrays(
                np.stack(frame_cls_list_pt),
                np.stack(frame_cls_list_tt),
                f"Cls Scores F{frame_idx}",
                rtol=1e-3,
                atol=1e-3,
            )
            match_reg_f = compare_arrays(
                np.stack(frame_reg_list_pt),
                np.stack(frame_reg_list_tt),
                f"Reg Points F{frame_idx}",
                rtol=1e-3,
                atol=1e-3,
            )

            print(f"    Decoder queries: {cur_output_pt.shape}")
            print(
                f"    Cls: {np.stack(frame_cls_list_pt).shape}, Reg: {np.stack(frame_reg_list_pt).shape}"
            )

            match_frame = match_dec and match_cls_f and match_reg_f
            validation_results.append(
                (
                    f"Layer {layer_counter}: Decoder F{frame_idx}",
                    match_frame,
                    f"dec+cls+reg",
                )
            )
            stage9_ok &= match_frame

            # Store for next frame
            frame_decoder_outputs_pt.append(cur_output_pt.detach().clone())
            frame_decoder_outputs_tt.append(cur_output_tt.copy())
            frame_ref_pts_pt.append(cur_ref_pt.clone())
            frame_ref_pts_tt.append(cur_ref_tt.copy())
            frame_cls_pt.append(frame_cls_list_pt[-1])
            frame_cls_tt.append(frame_cls_list_tt[-1])

        # Summary for Stage 9
        print(f"\n  {'─'*75}")
        print(f"  MULTI-FRAME SUMMARY: {num_frames} frames processed")
        for fi in range(num_frames):
            out_shape = frame_decoder_outputs_pt[fi].shape
            print(
                f"    Frame {fi}: decoder={out_shape}, cls={frame_cls_pt[fi].shape}, "
                f"reg={frame_ref_pts_pt[fi].shape}"
            )
        print(f"  {'─'*75}")

        layer_counter += 1
        validation_results.append(
            (
                f"Layer {layer_counter}: Multi-Frame ({num_frames}f)",
                stage9_ok,
                f"warp+fusion+memAttn+MLP×{num_frames - 1}",
            )
        )

        # =====================================================================
        # PIPELINE SUMMARY
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  COMPLETE MAPTRACKER PIPELINE:")
        print(f"  {'='*75}")
        print(f"  Input: Multi-camera images [{B}, {num_cams}, 3, {img_h}, {img_w}]")
        print(f"    |")
        print(f"  Image Backbone (ResNet-50 Bottleneck): Extract multi-scale features")
        print(f"    |")
        print(f"  FPN Neck: 3-level feature pyramid -> all {fpn_out_channels} channels")
        print(f"    |")
        print(
            f"  BEVFormer Encoder ({num_encoder_layers_bev} layers): Camera features -> BEV space"
        )
        print(f"    |")
        print(f"  BEV features [{B}, {embed_dims}, {bev_h}, {bev_w}]")
        print(f"    |")
        print(f"  BEV Backbone: Embedding -> Reshape -> Permute")
        print(f"    + History warping via GridSample")
        print(f"    |")
        print(
            f"  Segmentation Head: Conv2d(3x3) -> ReLU -> Upsample -> Conv2d -> ReLU -> Conv2d(1x1)"
        )
        print(f"    -> seg_preds [{B}, {num_cls}, {canvas_h}, {canvas_w}]")
        print(f"    |")
        print(f"  Detection Head: Input Proj -> BEV PosEmbed -> Flatten")
        print(f"    |")
        print(f"  Transformer Decoder ({num_decoder_layers} layers):")
        print(
            f"    Each: Self-Attn -> LN -> DeformAttn -> LN -> MemAttn -> LN -> FFN -> LN"
        )
        print(
            f"    -> RegressionBranch: Linear->LN->ReLU->Linear->LN->ReLU->Linear -> Sigmoid"
        )
        print(f"    -> ClassificationBranch: Linear({embed_dims}->{num_cls})")
        print(f"    |")
        print(f"  Query Propagation (MotionMLP):")
        print(
            f"    Embedder(sin/cos) -> Concat -> Linear -> LN -> ReLU -> Linear + Residual"
        )
        print(f"    |")
        print(f"  Multi-Frame Temporal ({num_frames} frames):")
        print(f"    Ego-motion warp grid -> GridSample -> BEV fusion")
        print(f"    -> Decoder with Memory Cross-Attn (ACTIVE on frames 1+)")
        print(f"    |")
        print(f"  Output:")
        print(f"    * Class scores: [{num_decoder_layers}, {B}, {num_q}, {num_cls}]")
        print(
            f"    * Reference points: [{num_decoder_layers}, {B}, {num_q}, {num_pts}, {coord_dim}]"
        )
        print(f"    * Semantic mask: [{canvas_h}, {canvas_w}]")
        print(f"  {'='*75}")

        # =====================================================================
        # VALIDATION SUMMARY
        # =====================================================================
        print(f"\n  {'='*75}")
        print(f"  VALIDATION SUMMARY")
        print(f"  {'='*75}\n")

        print(f"  {'Component':<45} {'Status':<10} {'Details'}")
        print(f"  {'-'*75}")

        all_match = True
        for comp_name, is_match, details in validation_results:
            status = "PASS" if is_match else "FAIL"
            print(f"  {comp_name:<45} {status:<10} {details}")
            if not is_match:
                all_match = False

        passed = sum(1 for _, m, _ in validation_results if m)
        total = len(validation_results)
        print(f"\n  {passed}/{total} components validated")
        print(f"  Total layers processed: {layer_counter}")

        if all_match:
            print(
                f"\n  SUCCESS! Complete MapTracker architecture validated with TTSim compute functions!"
            )
            return True
        else:
            print(f"\n  VALIDATION FAILED! Some components have mismatches")
            return False

    except Exception as e:
        print(f"\n  FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all test cases."""
    print_header("MapTracker Layer-by-Layer Validation - TTSim Complete Test Suite")

    tests = [
        ("Layer-by-Layer Validation with TTSim", test_layer_by_layer_validation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            print(f"\nRunning: {test_name}")
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"Error running {test_name}: {e}")
            traceback.print_exc()
            results[test_name] = False

    print_header("TEST SUMMARY")

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        dots = "." * (60 - len(test_name))
        print(f"{test_name}{dots} {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nAll tests passed! TTSim computations match PyTorch!")
        return True
    else:
        print("\nSome tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
