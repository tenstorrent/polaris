#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

#from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask

import ttsim.front.ttnn as ttnn


def bert_attention(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    num_heads = config.num_attention_heads
    batch_size, sequence_size, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query = ttnn.linear(
        hidden_states,
        parameters.self.query.weight,
        bias=parameters.self.query.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = ttnn.permute(query, (0, 2, 1, 3))

    key = ttnn.linear(
        hidden_states,
        parameters.self.key.weight,
        bias=parameters.self.key.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.permute(key, (0, 2, 3, 1))

    value = ttnn.linear(
        hidden_states,
        parameters.self.value.weight,
        bias=parameters.self.value.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.permute(value, (0, 2, 1, 3))

    attention_scores = ttnn.matmul(query, key)
    attention_scores = ttnn.to_device(attention_scores, device)
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = ttnn.to_layout(attention_scores, ttnn.TILE_LAYOUT)
        attention_mask = ttnn.to_layout(attention_mask, ttnn.TILE_LAYOUT)

        attention_scores = attention_scores + attention_mask

    #creates unknown attribute 'dim' in Softmax during onnx dump
    attention_probs = ttnn.softmax(attention_scores)#, dim=-1)

    context_layer = attention_probs @ value

    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))

    self_output = context_layer
    self_output = ttnn.linear(
        self_output,
        parameters.output.dense.weight,
        bias=parameters.output.dense.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    attention_output = ttnn.layer_norm(
        hidden_states + self_output,
        weight=parameters.output.LayerNorm.weight,
        bias=parameters.output.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return attention_output


def bert_intermediate(
    hidden_states,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        activation="gelu",
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return output


def bert_output(
    config,
    hidden_states,
    residual,
    device=None,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.dense.weight,
        bias=parameters.dense.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    output = ttnn.layer_norm(
        output + residual,
        weight=parameters.LayerNorm.weight,
        bias=parameters.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return output


def bert_feedforward(
    config,
    hidden_states,
    device=None,
    *,
    parameters,
):
    intermediate = bert_intermediate(hidden_states, parameters=parameters.intermediate, device=device)
    hidden_states = bert_output(config, intermediate, hidden_states, parameters=parameters.output, device=device)
    return hidden_states


def bert_layer(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    attention_output = bert_attention(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.attention,
        device=device,
    )

    feedforward_output = bert_feedforward(
        config,
        attention_output,
        parameters=parameters,
        device=device,
    )

    return feedforward_output


def bert_encoder(
    config,
    hidden_states,
    attention_mask,
    device=None,
    *,
    parameters,
):
    encoder_input = hidden_states
    encoder_output = None
    for encoder_parameters in parameters.layer:
        encoder_output = bert_layer(
            config,
            encoder_input,
            attention_mask,
            parameters=encoder_parameters,
            device=device,
        )
        encoder_input = encoder_output
    return encoder_output


def bert(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
):
    word_embeddings        = ttnn.embedding(input_ids,      parameters.embeddings.word_embeddings.weight)
    token_type_embeddings  = ttnn.embedding(token_type_ids, parameters.embeddings.token_type_embeddings.weight)
    position_embeddings    = ttnn.embedding(position_ids,   parameters.embeddings.position_embeddings.weight)

    word_embeddings        = ttnn.to_layout(word_embeddings,       ttnn.TILE_LAYOUT)
    token_type_embeddings  = ttnn.to_layout(token_type_embeddings, ttnn.TILE_LAYOUT)
    position_embeddings    = ttnn.to_layout(position_embeddings,   ttnn.TILE_LAYOUT)

    embeddings = word_embeddings + token_type_embeddings + position_embeddings

    hidden_states = ttnn.layer_norm(
        embeddings,
        weight=parameters.embeddings.LayerNorm.weight,
        bias=parameters.embeddings.LayerNorm.bias,
        epsilon=config.layer_norm_eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = bert_encoder(
        config,
        hidden_states,
        attention_mask,
        parameters=parameters.encoder,
        device=device,
    )

    return hidden_states


def bert_for_question_answering(
    config,
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device=None,
    *,
    parameters,
    name="bert",
):
    bert_output = bert(
        config,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        device=device,
        parameters = parameters,
        #parameters=parameters[name], #ERROR dict2obj not subscriptable...
    )

    qa_outputs = bert_output
    qa_outputs = ttnn.linear(
        qa_outputs,
        parameters.qa_outputs.weight,
        bias=parameters.qa_outputs.bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    return qa_outputs


def preprocess_inputs(
    input_ids,
    token_type_ids,
    position_ids,
    attention_mask,
    device,
):
    batch_size, _ = input_ids.shape

    input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    position_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    if attention_mask is not None:
        raise NotImplementedError("attention_mask behavior not implemented yet!!")
        #attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, torch.float32)
        #attention_mask = attention_mask.expand((batch_size, -1, -1, -1))
        #attention_mask = torch.clamp(attention_mask, min=-100000)
        #attention_mask = ttnn.from_torch(
        #    attention_mask,
        #    dtype=ttnn.bfloat16,
        #    layout=ttnn.TILE_LAYOUT,
        #    device=device,
        #    memory_config=ttnn.L1_MEMORY_CONFIG,
        #)

    return input_ids, token_type_ids, position_ids, attention_mask


def custom_preprocessor(torch_model, name):
    return {}

if __name__ == '__main__':
    from ttsim.utils.common import parse_yaml, dict2obj
    from workloads.ttnn.bert.utils import create_bert_params, create_bert_inputs

    def filter_ttnn_attrs(attrs_dict):
        onnx_attrs = {
                k: v for k, v in attrs_dict.items() if not isinstance(v,
                                                                      (
                                                                          ttnn.Tensor,
                                                                          ttnn.WormholeComputeKernelConfig,
                                                                          ttnn.MemoryConfig
                                                                          )
                                                                      )
                }
        return onnx_attrs

    yamlfile         = 'config/ttnn/bert/bert_tiny.yaml'
    configs          = parse_yaml(yamlfile)
    batch_size       = 8
    sequence_size    = 128
    for cfg_name, cfg_dict in configs.items():
        print(cfg_name)

        cfg_obj          = dict2obj(cfg_dict)
        device           = ttnn.open_device(l1_small_size=24576, device_id=0)
        parameters       = create_bert_params(cfg_obj, device=device, dtype=ttnn.bfloat16.to_numpy)
        ttnn_bert_inputs = create_bert_inputs(batch_size, sequence_size, device)
        ttnn_output      = bert_for_question_answering(cfg_obj, **ttnn_bert_inputs, parameters=parameters, device=device,)

        print("    ttnn_output=", ttnn_output.shape)

        #check graph via onnx dump
        onnxfilename = cfg_name.replace('/', '_') + '.onnx'
        g = device.get_graph()
        g.graph2onnx(onnxfilename, do_model_check=True, filter_op_attrs=filter_ttnn_attrs)

        ttnn.close_device(device)
