#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.ttnn_shim import permute_op
from ttsim.front.ttnn.tensor import Tensor, DataType
from loguru import logger

_LAYOUT_OPERATOR_TODO_KEYS: set[str] = set()


def _warn_layout_operator_todo_once(key: str, message: str) -> None:
    if key in _LAYOUT_OPERATOR_TODO_KEYS:
        return
    _LAYOUT_OPERATOR_TODO_KEYS.add(key)
    logger.debug("[ttnn/operators layout TODO] {}", message)


_warn_layout_operator_todo_once(
    "ttnn_optimized_sharded_vit_wh_file",
    "This file uses explicit to_layout / .layout where operator outputs lack correct layout metadata; set layout in operators and remove workarounds.",
)


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = config.patch_size
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    _warn_layout_operator_todo_once(
        "opt_vit_patch_fold_tolayout",
        "fold (and preceding ops) should set output layout; remove explicit to_layout.",
    )
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.matmul(pixel_values, parameters.projection.weight)
    _warn_layout_operator_todo_once(
        "opt_vit_patch_proj_matmul_layout",
        "matmul should set output layout metadata; remove explicit TILE assignment (patch projection).",
    )
    patch_embedding_output.layout = ttnn.TILE_LAYOUT
    logger.debug('matmul patch_embedding shape {} = pixel_values shape {} @ projection.weight shape {}',
                 patch_embedding_output.shape, pixel_values.shape, parameters.projection.weight.shape)
    patch_embedding_output = patch_embedding_output + parameters.projection.bias
    _warn_layout_operator_todo_once(
        "opt_vit_patch_matmul_add_tolayout",
        "matmul/add should set output layout; remove explicit to_layout.",
    )

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, patch_size_sq_trpl))

    return patch_embedding_output


def vit_embeddings(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    parameters = parameters.vit.embeddings

    patch_embeddings = vit_patch_embeddings(config, pixel_values, parameters=parameters.patch_embeddings)
    cls_token.layout = ttnn.ROW_MAJOR_LAYOUT
    embedding_output = ttnn.concat(cls_token, patch_embeddings, axis=1)
    _warn_layout_operator_todo_once(
        "opt_vit_embeddings_concat_layout",
        "concat should set output layout metadata; remove explicit TILE assignment (embeddings).",
    )
    embedding_output.layout = ttnn.TILE_LAYOUT
    embedding_output = embedding_output + position_embeddings
    _warn_layout_operator_todo_once(
        "opt_vit_embeddings_add_layout",
        "add should set output layout metadata; remove explicit TILE assignment (position embeddings).",
    )
    embedding_output.layout = ttnn.TILE_LAYOUT

    return embedding_output


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Fused QKV projection: matmul + bias
    query_key_value = hidden_states @ parameters.attention.query_key_value.weight
    _warn_layout_operator_todo_once(
        "opt_vit_attn_qkv_matmul_layout",
        "matmul should set output layout; remove explicit TILE assignment (QKV projection).",
    )
    query_key_value.layout = ttnn.TILE_LAYOUT
    logger.debug('matmul qkv shape {} = hidden_states shape {} @ qkv.weight shape {}',
                 query_key_value.shape, hidden_states.shape, parameters.attention.query_key_value.weight.shape)
    query_key_value = query_key_value + parameters.attention.query_key_value.bias
    query_key_value.layout = ttnn.TILE_LAYOUT

    # Split QKV into separate Q, K, V tensors and reshape to head-major layout
    q = Tensor(shape=[batch_size, seq_len, hidden_size], device=query_key_value.device,
               dtype=DataType.from_numpy(query_key_value.dtype))
    k = Tensor(shape=[batch_size, seq_len, hidden_size], device=query_key_value.device,
               dtype=DataType.from_numpy(query_key_value.dtype))
    v = Tensor(shape=[batch_size, seq_len, hidden_size], device=query_key_value.device,
               dtype=DataType.from_numpy(query_key_value.dtype))

    # Q: [B, S, D] -> [B, S, heads, hd] -> [B, heads, S, hd]
    q = ttnn.to_layout(ttnn.reshape(q, (batch_size, seq_len, num_heads, head_size)),
                       layout=ttnn.TILE_LAYOUT)
    q = permute_op(q, (0, 2, 1, 3))
    q.layout = ttnn.TILE_LAYOUT

    # K: [B, S, D] -> [B, S, heads, hd] -> [B, heads, S, hd] -> [B, heads, hd, S]
    k = ttnn.to_layout(ttnn.reshape(k, (batch_size, seq_len, num_heads, head_size)),
                       layout=ttnn.TILE_LAYOUT)
    k = permute_op(k, (0, 2, 1, 3))
    k.layout = ttnn.TILE_LAYOUT
    # HW split_query_key_value_and_split_heads returns K transposed
    k = permute_op(k, (0, 1, 3, 2))
    k.layout = ttnn.TILE_LAYOUT

    # V: [B, S, D] -> [B, S, heads, hd] -> [B, heads, S, hd]
    v = ttnn.to_layout(ttnn.reshape(v, (batch_size, seq_len, num_heads, head_size)),
                       layout=ttnn.TILE_LAYOUT)
    v = permute_op(v, (0, 2, 1, 3))
    v.layout = ttnn.TILE_LAYOUT

    # Q @ K^T
    attention_scores = q @ k
    _warn_layout_operator_todo_once(
        "opt_vit_attn_scores_matmul_layout",
        "matmul should set output layout; remove explicit assignment (attention scores).",
    )
    attention_scores.layout = ttnn.TILE_LAYOUT
    logger.debug('matmul attention_scores shape {} = q shape {} @ k shape {}',
                 attention_scores.shape, q.shape, k.shape)

    # Scale
    scale = 1.0 / (head_size ** 0.5)
    attention_scores = attention_scores * scale
    attention_scores.layout = ttnn.TILE_LAYOUT

    # Softmax
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    _warn_layout_operator_todo_once(
        "opt_vit_attn_softmax_layout",
        "softmax should set output layout; remove explicit assignment.",
    )
    attention_probs.layout = ttnn.TILE_LAYOUT

    # attention_probs @ V
    context_layer = attention_probs @ v
    _warn_layout_operator_todo_once(
        "opt_vit_attn_context_matmul_layout",
        "matmul should set output layout; remove explicit assignment (context).",
    )
    context_layer.layout = ttnn.TILE_LAYOUT
    logger.debug('matmul context_layer shape {} = attention_probs shape {} @ v shape {}',
                 context_layer.shape, attention_probs.shape, v.shape)

    # Concatenate heads: [B, heads, S, hd] -> [B, S, heads, hd] -> [B, S, D]
    context_layer = permute_op(context_layer, (0, 2, 1, 3))
    context_layer.layout = ttnn.TILE_LAYOUT
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, seq_len, hidden_size))
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    # Output dense projection: matmul + bias
    self_output = context_layer @ parameters.output.dense.weight
    _warn_layout_operator_todo_once(
        "opt_vit_attn_output_dense_matmul_layout",
        "matmul should set output layout; remove explicit assignment (attention output dense).",
    )
    self_output.layout = ttnn.TILE_LAYOUT
    logger.debug('matmul self_output shape {} = context_layer shape {} @ output.dense.weight shape {}',
                 self_output.shape, context_layer.shape, parameters.output.dense.weight.shape)
    self_output = self_output + parameters.output.dense.bias
    self_output.layout = ttnn.TILE_LAYOUT

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    logger.debug('matmul output shape {} = hidden_states shape {} @ weight shape {}',
                 output.shape, hidden_states.shape, parameters.dense.weight.shape)
    output.layout = hidden_states.layout
    output = output + parameters.dense.bias
    output.layout = hidden_states.layout
    # HW ff1_matmul_program_config has fused_activation=(GELU, True); decompose here
    output = ttnn.gelu(output)
    output.layout = hidden_states.layout
    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    logger.debug('matmul output shape {} = hidden_states shape {} @ weight shape {}',
                 output.shape, hidden_states.shape, parameters.dense.weight.shape)
    output.layout = hidden_states.layout
    output = output + parameters.dense.bias
    output.layout = ttnn.TILE_LAYOUT
    _warn_layout_operator_todo_once(
        "opt_vit_output_residual_add_layout",
        "add/residual path should set output layout metadata; remove explicit TILE assignment (vit_output).",
    )
    output = output + residual
    output.layout = ttnn.TILE_LAYOUT
    return output


def vit_feedforward(
    config,
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = vit_intermediate(config, hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_layernorm_before_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (layernorm_before).",
    )
    layernorm_before_output.layout = ttnn.TILE_LAYOUT

    multi_head_attention_output = vit_attention(
        config,
        layernorm_before_output,
        parameters=parameters.attention,
    )

    multi_head_attention_output = multi_head_attention_output + hidden_states
    _warn_layout_operator_todo_once(
        "opt_vit_layer_residual_add_layout",
        "residual add should set output layout metadata; remove explicit TILE assignment (vit_layer).",
    )
    multi_head_attention_output.layout = ttnn.TILE_LAYOUT

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_layernorm_after_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (layernorm_after).",
    )
    layernorm_after_output.layout = ttnn.TILE_LAYOUT

    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    embeddings,
    parameters,
):
    encoder_input = embeddings

    for index, encoder_parameters in enumerate(parameters.layer):
        encoder_output = vit_layer(
            config,
            encoder_input,
            encoder_parameters,
        )
        encoder_input = encoder_output

    return encoder_output


def vit(
    config,
    pixel_values,
    cls_token,
    position_embeddings,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_final_layernorm_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (final norm).",
    )
    output.layout = ttnn.TILE_LAYOUT

    # Classifier
    classifier_output = output @ parameters.classifier.weight
    logger.debug('matmul classifier_output shape {} = output shape {} @ classifier.weight shape {}',
                 classifier_output.shape, output.shape, parameters.classifier.weight.shape)
    classifier_output.layout = output.layout
    classifier_output = classifier_output + parameters.classifier.bias
    classifier_output.layout = output.layout

    return classifier_output
