#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
import ttsim.front.ttnn as ttnn
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
    ndims = len(pixel_values.shape)
    if ndims == 4:
        batch_size, img_h, img_w_over_patch, fold_c = pixel_values.shape
        patch_size = config.patch_size
        patch_count = img_h // patch_size  # 14
    else:
        raise ValueError(f"Expected 4D input, got {ndims}D: {pixel_values.shape}")
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.sharded_to_interleaved(pixel_values, ttnn.L1_MEMORY_CONFIG)
    _warn_layout_operator_todo_once(
        "opt_vit_patch_fold_tolayout",
        "fold (and preceding ops) should set output layout; remove explicit to_layout.",
    )
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    # ttnn.linear emits a single fused MatMul SimOp (matmul + bias), matching
    # HW's MatmulDeviceOperation.  Previously decomposed as matmul → add.
    # output_dtype=bfloat16: on HW the patch projection matmul accumulates in
    # BF16 (not activations_dtype), matching profiler OUT0=BFLOAT16.
    patch_embedding_output = ttnn.linear(
        pixel_values, parameters.projection.weight,
        bias=parameters.projection.bias,
        output_dtype=ttnn.bfloat16,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_patch_proj_linear_layout",
        "linear should set output layout metadata; remove explicit TILE assignment (patch projection).",
    )
    patch_embedding_output.layout = ttnn.TILE_LAYOUT
    logger.debug('linear patch_embedding shape {} = pixel_values shape {} @ projection.weight shape {}',
                 patch_embedding_output.shape, pixel_values.shape, parameters.projection.weight.shape)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Host-side view change (no SimOp) to recover [B, patches, hidden] from
    # the flattened fold shape [1,1,B*patches,hidden], matching HW's implicit
    # reshape between Untilize and Concat.
    patch_embedding_output.set_shape([batch_size, patch_count_all, patch_size_sq_trpl])

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

    # Explicit tilize after concat matches HW's TilizeWithValPadding (F:7168).
    embedding_output = ttnn.to_layout(embedding_output, layout=ttnn.TILE_LAYOUT)

    embedding_output = embedding_output + position_embeddings
    _warn_layout_operator_todo_once(
        "opt_vit_embeddings_add_layout",
        "add should set output layout metadata; remove explicit TILE assignment (position embeddings).",
    )
    embedding_output.layout = ttnn.TILE_LAYOUT

    embedding_output = ttnn.interleaved_to_sharded(
        embedding_output,
        ttnn.create_sharded_memory_config(
            embedding_output.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    return embedding_output


def vit_attention(
    config,
    hidden_states,
    parameters,
):
    num_heads = config.num_attention_heads  # 12
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    # Fused QKV projection — single MatMul SimOp (matmul + bias), matching HW.
    query_key_value = ttnn.linear(
        hidden_states, parameters.attention.query_key_value.weight,
        bias=parameters.attention.query_key_value.bias,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_attn_qkv_linear_layout",
        "linear should set output layout; remove explicit TILE assignment (QKV projection).",
    )
    query_key_value.layout = ttnn.TILE_LAYOUT
    logger.debug('linear qkv shape {} = hidden_states shape {} @ qkv.weight shape {}',
                 query_key_value.shape, hidden_states.shape, parameters.attention.query_key_value.weight.shape)

    query_key_value = ttnn.reshard(
        query_key_value,
        ttnn.create_sharded_memory_config(
            query_key_value.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Single CreateQKVHeads op replaces the previous decomposed sequence of
    # 3×(Tensor → reshape → permute) + extra permute for K transpose.
    # HW's split_query_key_value_and_split_heads maps to this single op.
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        query_key_value,
        num_heads=num_heads,
        transpose_k_heads=True,
    )
    logger.debug('nlp_create_qkv_heads: q {} k {} v {}', q.shape, k.shape, v.shape)

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

    # Single ConcatHeads op replaces the previous decomposed permute → reshape.
    # HW's concatenate_heads maps to this single op.
    context_layer = ttnn.experimental.nlp_concat_heads(context_layer)
    context_layer.layout = ttnn.TILE_LAYOUT

    context_layer = ttnn.reshard(
        context_layer,
        ttnn.create_sharded_memory_config(
            context_layer.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Output dense projection — fused matmul + bias, matching HW.
    self_output = ttnn.linear(
        context_layer, parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
    )
    _warn_layout_operator_todo_once(
        "opt_vit_attn_output_dense_linear_layout",
        "linear should set output layout; remove explicit assignment (attention output dense).",
    )
    self_output.layout = ttnn.TILE_LAYOUT
    logger.debug('linear self_output shape {} = context_layer shape {} @ output.dense.weight shape {}',
                 self_output.shape, context_layer.shape, parameters.output.dense.weight.shape)

    return self_output


def vit_intermediate(
    config,
    hidden_states,
    *,
    parameters,
):
    # Fused matmul + bias + GELU as a single MatMul SimOp.  HW's
    # ff1_matmul_program_config has fused_activation=(GELU, True).
    output = ttnn.linear(
        hidden_states, parameters.dense.weight,
        bias=parameters.dense.bias,
        activation="gelu",
    )
    logger.debug('linear(+gelu) output shape {} = hidden_states shape {} @ weight shape {}',
                 output.shape, hidden_states.shape, parameters.dense.weight.shape)
    output.layout = hidden_states.layout
    return output


def vit_output(
    config,
    hidden_states,
    residual,
    *,
    parameters,
):
    # Fused matmul + bias as a single MatMul SimOp, matching HW.
    output = ttnn.linear(
        hidden_states, parameters.dense.weight,
        bias=parameters.dense.bias,
    )
    logger.debug('linear output shape {} = hidden_states shape {} @ weight shape {}',
                 output.shape, hidden_states.shape, parameters.dense.weight.shape)
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
        compute_kernel_config=config.program_configs["ln_compute_config"],
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
        compute_kernel_config=config.program_configs["ln_compute_config"],
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

    output = ttnn.reshard(
        output,
        ttnn.create_sharded_memory_config(
            output.shape,
            core_grid=config.core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Classifier — fused matmul + bias, matching HW.
    classifier_output = ttnn.linear(
        output, parameters.classifier.weight,
        bias=parameters.classifier.bias,
    )
    logger.debug('linear classifier_output shape {} = output shape {} @ classifier.weight shape {}',
                 classifier_output.shape, output.shape, parameters.classifier.weight.shape)
    classifier_output.layout = output.layout

    return classifier_output
