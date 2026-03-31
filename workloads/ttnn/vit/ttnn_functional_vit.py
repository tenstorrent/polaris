#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.ttnn_shim import permute_op
from loguru import logger

_LAYOUT_OPERATOR_TODO_KEYS: set[str] = set()


def _warn_layout_operator_todo_once(key: str, message: str) -> None:
    if key in _LAYOUT_OPERATOR_TODO_KEYS:
        return
    _LAYOUT_OPERATOR_TODO_KEYS.add(key)
    logger.warning("[ttnn/operators layout TODO] {}", message)


# TODO(ttnn/operators): This file uses explicit to_layout / .layout assignments where operator outputs
# do not yet carry correct layout metadata. Remove those workarounds once each relevant op sets layout.
_warn_layout_operator_todo_once(
    "ttnn_functional_vit_file",
    "This file uses explicit to_layout / .layout where operator outputs lack correct layout metadata; set layout in operators and remove workarounds.",
)


def vit_patch_embeddings(config, pixel_values, *, parameters, unittest_check=False):
    # batch_size, img_c, img_h, img_w = pixel_values.shape # NCHW
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 16
    patch_count = img_h // patch_size  # 14
    patch_size_sq_trpl = int(patch_size * patch_size * 3)  # 768
    patch_count_all = int(patch_count * patch_count)  # 196
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    # TODO(ttnn/operators): fold (and preceding ops) should set output layout; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_patch_fold_tolayout",
        "fold (and preceding ops) should set output layout; remove explicit to_layout.",
    )
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    if unittest_check:
        parameters = parameters.vit.embeddings.patch_embeddings

    patch_embedding_output = ttnn.matmul(pixel_values, parameters.projection.weight)
    _warn_layout_operator_todo_once(
        "vit_patch_proj_matmul_layout",
        "matmul should set output layout metadata; remove explicit TILE assignment (patch projection).",
    )
    patch_embedding_output.layout = ttnn.TILE_LAYOUT
    patch_embedding_output = patch_embedding_output + parameters.projection.bias

    # TODO(ttnn/operators): matmul/add should set output layout; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_patch_matmul_add_tolayout",
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
        "vit_embeddings_concat_layout",
        "concat should set output layout metadata; remove explicit TILE assignment (embeddings).",
    )
    embedding_output.layout = ttnn.TILE_LAYOUT
    embedding_output = embedding_output + position_embeddings
    _warn_layout_operator_todo_once(
        "vit_embeddings_add_layout",
        "add should set output layout metadata; remove explicit TILE assignment (position embeddings).",
    )
    embedding_output.layout = ttnn.TILE_LAYOUT

    return embedding_output


def vit_layernorm_before(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_before.weight,
        bias=parameters.layernorm_before.bias,
    )
    _warn_layout_operator_todo_once(
        "vit_layernorm_before_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (layernorm_before).",
    )
    attention_output.layout = ttnn.TILE_LAYOUT
    return attention_output


def vit_layernorm_after(
    config,
    hidden_states,
    *,
    parameters,
):
    attention_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layernorm_after.weight,
        bias=parameters.layernorm_after.bias,
    )
    _warn_layout_operator_todo_once(
        "vit_layernorm_after_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (layernorm_after).",
    )
    attention_output.layout = ttnn.TILE_LAYOUT

    return attention_output


def vit_attention(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = hidden_states @ parameters.attention.query.weight
    logger.warning('matmul query shape {} = hidden_states shape {} @ parameters.attention.query.weight shape {}', query.shape, hidden_states.shape, parameters.attention.query.weight.shape)
    # TODO(ttnn/operators): matmul should stamp output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_query_matmul_layout",
        "matmul should stamp output layout; remove explicit assignment (query projection).",
    )
    query.layout = hidden_states.layout
    query = query + parameters.attention.query.bias
    # query.layout = ttnn.TILE_LAYOUT
    # TODO(ttnn/operators): add/bias should preserve or set layout; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_query_add_tolayout_rm",
        "add/bias should preserve or set layout; remove explicit to_layout (query).",
    )
    query = ttnn.to_layout(query, layout=ttnn.ROW_MAJOR_LAYOUT)
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    # TODO(ttnn/operators): reshape should set or require layout internally; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_query_reshape_tolayout_tile",
        "reshape should set or require layout internally; remove explicit to_layout (query).",
    )
    query = ttnn.to_layout(query, layout=ttnn.TILE_LAYOUT)
    query = permute_op(query, (0, 2, 1, 3))
    # TODO(ttnn/operators): permute should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_query_permute_layout",
        "permute should set output layout; remove explicit assignment (query).",
    )
    query.layout = ttnn.TILE_LAYOUT

    key = hidden_states @ parameters.attention.key.weight
    logger.warning('matmul key shape {} = hidden_states shape {} @ parameters.attention.key.weight shape {}', key.shape, hidden_states.shape, parameters.attention.key.weight.shape)
    # TODO(ttnn/operators): matmul should stamp output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_key_matmul_layout",
        "matmul should stamp output layout; remove explicit assignment (key projection).",
    )
    key.layout = hidden_states.layout
    logger.warning('key shape {}', key.shape)
    logger.warning('bias shape {}', parameters.attention.key.bias.shape)
    key = key + parameters.attention.key.bias
    logger.warning('key result shape {}', key.shape)
    # TODO(ttnn/operators): add should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_key_add_layout",
        "add should set output layout; remove explicit assignment (key bias).",
    )
    key.layout = ttnn.TILE_LAYOUT
    # TODO(ttnn/operators): add/bias path should not require manual layout; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_key_add_tolayout_rm",
        "add/bias path should not require manual layout; remove explicit to_layout (key).",
    )
    key = ttnn.to_layout(key, layout=ttnn.ROW_MAJOR_LAYOUT)
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    # TODO(ttnn/operators): reshape should set layout metadata; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_key_reshape_tolayout_tile",
        "reshape should set layout metadata; remove explicit to_layout (key).",
    )
    key = ttnn.to_layout(key, layout=ttnn.TILE_LAYOUT)
    key = permute_op(key, (0, 2, 3, 1))
    # TODO(ttnn/operators): permute should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_key_permute_layout",
        "permute should set output layout; remove explicit assignment (key).",
    )
    key.layout = ttnn.TILE_LAYOUT

    value = hidden_states @ parameters.attention.value.weight
    logger.warning('matmul value shape {} = hidden_states shape {} @ parameters.attention.value.weight shape {}', value.shape, hidden_states.shape, parameters.attention.value.weight.shape)
    # TODO(ttnn/operators): matmul should stamp output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_value_matmul_layout",
        "matmul should stamp output layout; remove explicit assignment (value projection).",
    )
    value.layout = hidden_states.layout
    value = value + parameters.attention.value.bias
    # TODO(ttnn/operators): add should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_value_add_layout",
        "add should set output layout; remove explicit assignment (value bias).",
    )
    value.layout = ttnn.TILE_LAYOUT
    # TODO(ttnn/operators): add/bias path should not require manual layout; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_value_add_tolayout_rm",
        "add/bias path should not require manual layout; remove explicit to_layout (value).",
    )
    value = ttnn.to_layout(value, layout=ttnn.ROW_MAJOR_LAYOUT)
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    # TODO(ttnn/operators): reshape should set layout metadata; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_value_reshape_tolayout_tile",
        "reshape should set layout metadata; remove explicit to_layout (value).",
    )
    value = ttnn.to_layout(value, layout=ttnn.TILE_LAYOUT)
    value = permute_op(value, (0, 2, 1, 3))
    # TODO(ttnn/operators): permute should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_value_permute_layout",
        "permute should set output layout; remove explicit assignment (value).",
    )
    value.layout = ttnn.TILE_LAYOUT

    attention_scores = query @ key
    logger.warning('matmul attention_scores shape {} = query shape {} @ key shape {}', attention_scores.shape, query.shape, key.shape)
    # TODO(ttnn/operators): matmul should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_scores_matmul_layout",
        "matmul should set output layout; remove explicit assignment (attention scores).",
    )
    attention_scores.layout = ttnn.TILE_LAYOUT
    logger.warning("query layout {} key layout {}", query.layout, key.layout)
    logger.warning("attention_scores layout {}", attention_scores.layout)
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        logger.warning("attention mark shape {} layout {}", attention_mask.shape, attention_mask.layout)
        logger.warning("attention scores shape {} layout {}", attention_scores.shape, attention_scores.layout)
        attention_scores = attention_scores + attention_mask
    logger.warning("attention scores shape {} layout {}", attention_scores.shape, attention_scores.layout)
    # TODO(ttnn/operators): mul/add/mask broadcast should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_scores_post_scale_mask_layout",
        "mul/add/mask broadcast should set output layout; remove explicit assignment (attention scores).",
    )
    attention_scores.layout = ttnn.TILE_LAYOUT

    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    # TODO(ttnn/operators): softmax should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_softmax_layout",
        "softmax should set output layout; remove explicit assignment.",
    )
    attention_probs.layout = ttnn.TILE_LAYOUT

    context_layer = attention_probs @ value

    logger.warning('matmul context_layer shape {} = attention_probs shape {} @ value shape {}', context_layer.shape, attention_probs.shape, value.shape)
    # TODO: Remove this hack after fixing the issue with permute and reshape.
    # TODO(ttnn/operators): matmul should set output layout; remove explicit assignment when permute/reshape fixed.
    _warn_layout_operator_todo_once(
        "vit_attn_context_matmul_layout",
        "matmul should set output layout; remove explicit assignment (context); fix permute/reshape hack.",
    )
    context_layer.layout = ttnn.TILE_LAYOUT
    # TODO: end
    logger.debug("context_layer: {} layout {} before permute", context_layer.name, context_layer.layout)
    context_layer = permute_op(context_layer, (0, 2, 1, 3))
    # TODO: Remove this hack after fixing the issue with permute and reshape.
    # TODO(ttnn/operators): permute should set output layout; remove explicit assignment when op is fixed.
    _warn_layout_operator_todo_once(
        "vit_attn_context_permute_layout",
        "permute should set output layout; remove explicit assignment (context); fix permute/reshape hack.",
    )
    context_layer.layout = ttnn.TILE_LAYOUT
    context_layer.set_shape(context_layer.shape)
    # TODO: end
    logger.debug("context_layer: {} layout {} after permute", context_layer.name, context_layer.layout)
    # TODO(ttnn/operators): permute should emit correct layout for downstream reshape; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_context_tolayout_rm",
        "permute should emit correct layout for downstream reshape; remove explicit to_layout (ROW_MAJOR).",
    )
    context_layer = ttnn.to_layout(context_layer, ttnn.ROW_MAJOR_LAYOUT)
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    # TODO(ttnn/operators): reshape should set layout metadata; remove explicit to_layout.
    _warn_layout_operator_todo_once(
        "vit_attn_context_reshape_tolayout_tile",
        "reshape should set layout metadata; remove explicit to_layout (context merge heads).",
    )
    context_layer = ttnn.to_layout(context_layer, ttnn.TILE_LAYOUT)

    self_output = context_layer
    logger.warning("self_output layout {}", self_output.layout)
    self_output = self_output @ parameters.attention.output.dense.weight

    logger.warning('matmul self_output shape {} = context_layer shape {} @ parameters.attention.output.dense.weight shape {}', self_output.shape, context_layer.shape, parameters.attention.output.dense.weight.shape)
    # TODO(ttnn/operators): matmul should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_output_dense_matmul_layout",
        "matmul should set output layout; remove explicit assignment (attention output dense).",
    )
    self_output.layout = ttnn.TILE_LAYOUT
    self_output = self_output + parameters.attention.output.dense.bias
    # TODO(ttnn/operators): add should set output layout; remove explicit assignment.
    _warn_layout_operator_todo_once(
        "vit_attn_output_dense_bias_layout",
        "add should set output layout; remove explicit assignment (attention output bias).",
    )
    self_output.layout = ttnn.TILE_LAYOUT
    return self_output


def vit_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = hidden_states @ parameters.dense.weight
    logger.warning('matmul output shape {} = hidden_states shape {} @ parameters.dense.weight shape {}', output.shape, hidden_states.shape, parameters.dense.weight.shape)
    # TODO: placeholder for matmul layout
    output.layout = hidden_states.layout
    output = output + parameters.dense.bias
    # TODO: placeholder for add layout
    output.layout = hidden_states.layout
    output = ttnn.gelu(output)
    # TODO: placeholder for gelu layout
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
    logger.warning('matmul output shape {} = hidden_states shape {} @ parameters.dense.weight shape {}', output.shape, hidden_states.shape, parameters.dense.weight.shape)
    output.layout = hidden_states.layout
    output = output + parameters.dense.bias
    output.layout = ttnn.TILE_LAYOUT
    _warn_layout_operator_todo_once(
        "vit_output_residual_add_layout",
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
    intermediate = vit_intermediate(hidden_states, parameters=parameters.intermediate)
    hidden_states = vit_output(config, intermediate, attention_output, parameters=parameters.output)
    return hidden_states


def vit_layer(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    layernorm_before_output = vit_layernorm_before(
        config,
        hidden_states,
        parameters=parameters,
    )
    attention_output = vit_attention(
        config,
        layernorm_before_output,
        attention_mask,
        parameters=parameters,
    )
    attention_output = attention_output + hidden_states
    _warn_layout_operator_todo_once(
        "vit_layer_residual_add_layout",
        "residual add should set output layout metadata; remove explicit TILE assignment (vit_layer).",
    )
    attention_output.layout = ttnn.TILE_LAYOUT
    layernorm_after_output = vit_layernorm_after(
        config,
        attention_output,
        parameters=parameters,
    )
    feedforward_output = vit_feedforward(
        config,
        layernorm_after_output,
        attention_output,
        parameters=parameters,
    )

    return feedforward_output


def vit_encoder(
    config,
    hidden_states,
    attention_mask,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = vit_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
        )
        encoder_input = encoder_output
    return encoder_output


def vit(
    config,
    pixel_values,
    attention_mask,
    cls_token,
    position_embeddings,
    *,
    parameters,
):
    embeddings_output = vit_embeddings(config, pixel_values, cls_token, position_embeddings, parameters=parameters)

    hidden_states = vit_encoder(
        config,
        embeddings_output,
        attention_mask=attention_mask,
        parameters=parameters.vit.encoder,
    )

    # Final LayerNorm
    output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.vit.layernorm.weight,
        bias=parameters.vit.layernorm.bias,
    )
    _warn_layout_operator_todo_once(
        "vit_final_layernorm_layout",
        "layer_norm should set output layout metadata; remove explicit TILE assignment (final norm).",
    )
    output.layout = ttnn.TILE_LAYOUT
    # Classifier
    classifier_output = output @ parameters.classifier.weight
    logger.warning('matmul classifier_output shape {} = output shape {} @ parameters.classifier.weight shape {}', classifier_output.shape, output.shape, parameters.classifier.weight.shape)
    # TODO: placeholder for matmul layout
    classifier_output.layout = output.layout
    classifier_output = classifier_output + parameters.classifier.bias
    # TODO: placeholder for add layout
    classifier_output.layout = output.layout

    return classifier_output
