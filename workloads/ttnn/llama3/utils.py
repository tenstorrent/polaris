#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
import os, sys

from loguru import logger
from numpy import reshape
from ttsim.front.ttnn.tensor import DataType, Tensor, Layout
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

def nlp_create_qkv_heads(
    xqkv_fused,
    num_heads=1,
    num_kv_heads=1,
    transpose_k_heads=False,
    memory_config=None,
    ):
    # Assume xqkv_fused.shape = [batch, seq_len, 3 * num_heads * head_dim]
    [batch, seq_groups, seq_len, fused_dim] = xqkv_fused.shape
    head_dim = fused_dim // (num_heads + 2 * num_kv_heads)
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim

    # Split Q, K, V
    # slicing to use instead of new Tensor creation
    # q = xqkv_fused[:, :, :q_end]
    # k = xqkv_fused[:, :, q_end:k_end]
    # v = xqkv_fused[:, :, k_end:]
    q_shape = [batch, seq_groups, seq_len, q_end]
    q = Tensor(shape=q_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))  # Simulate tensor creation
    k_shape = [batch, seq_groups, seq_len, k_end - q_end]
    k = Tensor(shape=k_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))  # Simulate tensor creation
    v_shape = [batch, seq_groups, seq_len, fused_dim - k_end]
    v = Tensor(shape=v_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))  # Simulate tensor creation

    # Reshape Q: [batch, seq_len, num_heads * head_dim] -> [batch, num_heads, seq_len, head_dim]
    q = ttnn.permute(ttnn.reshape(q, (batch, seq_len, num_heads, head_dim)), (0, 2, 1, 3))

    # Reshape K, V: [batch, seq_len, num_kv_heads * head_dim] -> [batch, num_kv_heads, seq_len, head_dim]
    k = ttnn.permute(ttnn.reshape(k, (batch, seq_len, num_kv_heads, head_dim)), (0, 2, 1, 3))
    v = ttnn.permute(ttnn.reshape(v, (batch, seq_len, num_kv_heads, head_dim)), (0, 2, 1, 3))

    if transpose_k_heads:
        # For multi-query attention, expand K/V to match Q heads
        # TODO: Review Suggestion
        # The parameter is named transpose_k_heads but the operation being performed
        # is expanding K/V heads for multi-query attention, not transposing. 
        # The parameter name is misleading and should be renamed to something like
        # expand_kv_heads or the logic should be corrected.
        k = k.repeat(1, num_heads // num_kv_heads, 1, 1)
        v = v.repeat(1, num_heads // num_kv_heads, 1, 1)

    return q, k, v

def nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads,
    num_kv_heads,
    memory_config=None,
):
    """
    Splits fused QKV tensor into Q, K, V heads for decode mode.

    Args:
        xqkv_fused: Input fused tensor of shape [batch, seq_len, 3 * num_heads * head_dim]
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        memory_config: Optional memory configuration

    Returns:
        q, k, v tensors split and reshaped for attention
    """
    batch_dim, seq_len, batch, fused_dim = xqkv_fused.shape
    head_dim = fused_dim // (num_heads + 2 * num_kv_heads)
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim
    assert head_dim * (num_heads + 2 * num_kv_heads) == fused_dim, \
        "Dimension mismatch between heads and fused projection size"

    # Simulate slicing Q, K, V from fused tensor
    q_shape = [batch, seq_len, q_end]
    k_shape = [batch, seq_len, k_end - q_end]
    v_shape = [batch, seq_len, fused_dim - k_end]
    q = Tensor(shape=q_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))
    k = Tensor(shape=k_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))
    v = Tensor(shape=v_shape, device=xqkv_fused.device, dtype=DataType.from_numpy(xqkv_fused.dtype))

    # Reshape Q: [batch, seq_len, num_heads * head_dim] -> [1, batch, num_heads, head_dim]
    q = ttnn.reshape(q, (batch, num_heads, head_dim)).unsqueeze(0)  # Add a dimension for decode mode

    # Reshape K, V: [batch, seq_len, num_kv_heads * head_dim] -> [1, batch, num_kv_heads, head_dim]
    k = ttnn.reshape(k, (batch, num_kv_heads, head_dim)).unsqueeze(0)  # Add a dimension for decode mode
    v = ttnn.reshape(v, (batch, num_kv_heads, head_dim)).unsqueeze(0)  # Add a dimension for decode mode

    return q, k, v

import math as mth

def rotary_embedding_llama(
    head_pre_rot,
    rot_mat1,
    rot_mat2,
    transformation_mats=None,
    is_decode_mode=False,
):
    """
    Applies rotary embedding to the input tensor using the formula:
    output = x * cos + (x @ trans_mat) * sin

    Args:
        head_pre_rot: Input tensor (x)
        rot_mat1: Cosine matrix (cos)
        rot_mat2: Sine matrix (sin)
        transformation_mats: Transformation matrix (trans_mat), optional
        is_decode_mode: Not used in this implementation

    Returns:
        Tensor with rotary embedding applied
    """
    x = head_pre_rot
    cos = rot_mat1
    sin = rot_mat2

    if transformation_mats is not None:
        trans_mat = transformation_mats
        # Compute how many matmul calls of trans_mat size are needed for input x
        chunk_size = trans_mat.shape[-2:]  # Assuming trans_mat is [..., N, N]
        input_last2 = x.shape[-2:]
        num_matmul_calls = mth.ceil(input_last2[0] / chunk_size[0]) * mth.ceil(input_last2[1] / chunk_size[1])
        if (num_matmul_calls > 0):
            logger.info(f'Below are stats for each matmul call of size {chunk_size} on input of size {input_last2}, a total of {num_matmul_calls} calls')
            for i in range(num_matmul_calls):
                in1 = Tensor(shape=trans_mat.shape, device=x.device, dtype=DataType.from_numpy(trans_mat.dtype))
                dummy_out = ttnn.matmul(in1, in1)
                logger.info(f'  matmul call {i+1}/{num_matmul_calls} done, w output shape {dummy_out.shape}')
        x_rot = x  # No actual computation, just reporting the count
    else:
        # If no transformation matrix is provided, assume identity (no rotation)
        x_rot = x

    out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(x_rot, sin))
    return out

def scaled_dot_product_attention(
    q_heads_1QSD_8b,
    k_heads_1KSD_8b,
    v_heads_1VSD_8b,
    is_causal=True,
    scale=None,
    compute_kernel_config=None,
    program_config=None,
):
    """
    Implements scaled dot-product attention.

    Args:
        q_heads_1QSD_8b: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        k_heads_1KSD_8b: Key tensor of shape [batch, num_heads, seq_len, head_dim]
        v_heads_1VSD_8b: Value tensor of shape [batch, num_heads, seq_len, head_dim]
        is_causal: Whether to apply causal masking
        scale: Scaling factor for attention scores
        compute_kernel_config: Optional compute kernel config
        program_config: Optional program config

    Returns:
        Output tensor after attention
    """
    # MatMul Q x K^T
    # For MQA, repeat K along num_heads axis to match Q's num_heads
    num_heads = q_heads_1QSD_8b.shape[1]
    num_kv_heads = k_heads_1KSD_8b.shape[1]
    if num_heads != num_kv_heads:
        k_heads_expanded = k_heads_1KSD_8b.repeat(1, num_heads // num_kv_heads, 1, 1)
    else:
        k_heads_expanded = k_heads_1KSD_8b
    k_transposed = ttnn.permute(k_heads_expanded, (0, 1, 3, 2))  # [batch, num_heads, head_dim, seq_len]
    attn_scores = ttnn.matmul(q_heads_1QSD_8b, k_transposed)     # [batch, num_heads, seq_len, seq_len]

    # Scale
    if scale is not None:
        if not isinstance(scale, Tensor):
            scale = Tensor(shape=attn_scores.shape, dtype=DataType.FLOAT32, device=q_heads_1QSD_8b.device)
        attn_scores = ttnn.div(attn_scores, scale)

    # Causal mask
    ### commenting for now
    # if is_causal: # for now lets assume its not causal
    #     seq_len = attn_scores.shape[-1]
    #     mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.bool_)
    #     mask_tensor = Tensor(shape=mask.shape, dtype=DataType.BOOL, device=attn_scores.device, data=mask)
    #     attn_scores = where(mask_tensor, neg(float('inf')), attn_scores)

    # Softmax
    attn_probs = ttnn.softmax(attn_scores)

    # MatMul with V
    # For MQA, repeat V along num_heads axis to match Q's num_heads
    num_heads = q_heads_1QSD_8b.shape[1]
    num_kv_heads = v_heads_1VSD_8b.shape[1]
    if num_heads != num_kv_heads:
        v_heads_expanded = v_heads_1VSD_8b.repeat(1, num_heads // num_kv_heads, 1, 1)
    else:
        v_heads_expanded = v_heads_1VSD_8b
    output = ttnn.matmul(attn_probs, v_heads_expanded)

    return output

def nlp_concat_heads(attn_output_1QSD, memory_config=None):
    """
    Concatenates attention output heads into a single tensor.

    Args:
        attn_output_1QSD: Tensor of shape [batch, num_heads, seq_len, head_dim]
        memory_config: Optional memory configuration

    Returns:
        Tensor of shape [batch, seq_len, num_heads * head_dim]
    """
    batch, num_heads, seq_len, head_dim = attn_output_1QSD.shape
    # Permute to [batch, seq_len, num_heads, head_dim]
    permuted = ttnn.permute(attn_output_1QSD, (0, 2, 1, 3))
    # Reshape to [batch, seq_len, num_heads * head_dim]
    output = ttnn.reshape(permuted, (batch, seq_len, num_heads * head_dim))
    return output

def nlp_concat_heads_decode(
    attn_output_11BH,
    num_heads,
):
    """
    Concatenates attention output heads into a single tensor for decode mode.

    Args:
        attn_output_11BH: Tensor of shape [1, batch, num_heads, head_dim]
        num_heads: Number of heads

    Returns:
        Tensor of shape [1, batch, num_heads * head_dim]
    """
    batch, seq_len, attm_heads, head_dim = attn_output_11BH.shape
    # Reshape to [1, batch, num_heads * head_dim]
    output = ttnn.reshape(attn_output_11BH, (batch, seq_len, attm_heads * head_dim))
    return output

def scaled_dot_product_attention_decode(
    q_heads_1BQD,
    keys,
    values,
    cur_pos_tensor=None,
    scale=None,
    program_config=None,
    compute_kernel_config=None,
    memory_config=None,
):
    # GQA: For each query head, use its corresponding KV head for Q @ K^T
    num_query_heads = q_heads_1BQD.shape[2]
    num_kv_heads = keys.shape[1]
    # if num_query_heads > num_kv_heads:
    #     logger.info(f'Implementation of GQA {num_query_heads / num_kv_heads} mapping for decode mode')

    def gqa_matmul(q_head, kv_head, transposed=True):
        # MatMul Q x K^T: [1, num_query_heads, seq_len_q, head_dim] x [1, num_query_heads, head_dim, seq_len_k] -> [1, num_query_heads, seq_len_q, seq_len_k]
        # GQA 3:1 mapping: For each query head, use its corresponding KV head (i // 3)
        batch_size, seq_len_q, num_query_heads, qdim = q_head.shape
        _, num_kv_heads, seq_len_k, kdim = kv_head.shape
        for i in range(num_query_heads):
            #kv_head_idx = i // 3
            # slicing not suported yet. so, create new tensor
            #q_head = q_heads_1BQD[:, i:i+1, :, :]           # [1, 1, num_q_heads, dim]
            q_head = Tensor(shape=(batch_size, 1, 1, qdim), device=q_heads_1BQD.device, dtype=DataType.from_numpy(q_heads_1BQD.dtype))
            #k_head = keys_transposed[:, kv_head_idx:kv_head_idx+1, :, :]  # [1, 1, head_dim, seq_len_k]
            if transposed:
                k_head = Tensor(shape=(batch_size, 1, kdim, seq_len_k), device=keys.device, dtype=DataType.from_numpy(keys.dtype))
            else:
                k_head = Tensor(shape=(batch_size, 1, seq_len_k, kdim), device=keys.device, dtype=DataType.from_numpy(keys.dtype))
            attn_score = ttnn.matmul(q_head, k_head)
            bs, slq, _, ldim = attn_score.shape
        attn_scores = Tensor(shape=(bs, slq, num_query_heads, ldim), device=q_heads_1BQD.device, dtype=DataType.from_numpy(q_heads_1BQD.dtype))
        return attn_scores

    attn_scores = gqa_matmul(q_heads_1BQD, keys, transposed=True)  # [1, num_query_heads, seq_len_q, seq_len_k]
    # Scale
    if scale is not None:
        if not isinstance(scale, Tensor):
            scale = Tensor(shape=attn_scores.shape, dtype=DataType.FLOAT32, device=q_heads_1BQD.device)
        attn_scores = ttnn.div(attn_scores, scale)

    import numpy as np
    # Causal mask for decode: only attend to cur_pos_tensor
    if cur_pos_tensor is not None:
        seq_len_k = attn_scores.shape[-1]
        mask = np.ones((seq_len_k,), dtype=np.bool_)
        mask[:] = True
        #mask[cur_pos_tensor[0]:] = False  # Only allow up to cur_pos
        mask[0:] = False
        mask_tensor = Tensor(shape=mask.shape, dtype=DataType.BOOL, device=attn_scores.device, data=mask)
        mask_tensor = ttnn.reshape(mask_tensor, (1, 1, 1, seq_len_k))
        fill_tensor = Tensor(shape=attn_scores.shape, dtype=DataType.from_numpy(attn_scores.dtype), device=attn_scores.device, data=np.full(attn_scores.shape, float('inf')))
        attn_scores = ttnn.where(mask_tensor, attn_scores, fill_tensor)
    # Softmax
    attn_probs = ttnn.softmax(attn_scores)
    # MatMul with V: [1, seq_len_q, num_query_heads, seq_len_k] x [1, num_kv_heads, seq_len_k, head_dim] -> [1, seq_len_q, num_query_heads, head_dim]
    output = gqa_matmul(attn_probs, values, transposed=False)  # [1, seq_len_q, num_query_heads, head_dim]
    return output

def paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor, page_table):
    pass
