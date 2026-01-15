#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tests.pcc.test_tt_backbone import Parameter
from workloads.ttnn.vadv2.tt.tt_fpn import ConvParams
import re

spec = """
head:
head.positional_encoding:
head.positional_encoding.row_embed:
head.positional_encoding.row_embed.weight: tensor with shape Shape([100, 128])
head.positional_encoding.col_embed:
head.positional_encoding.col_embed.weight: tensor with shape Shape([100, 128])
head.motion_decoder:
head.motion_decoder.layers:
head.motion_decoder.layers.layer0:
head.motion_decoder.layers.layer0.attentions:
head.motion_decoder.layers.layer0.attentions.attn0:
head.motion_decoder.layers.layer0.attentions.attn0.in_proj:
head.motion_decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.motion_decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.motion_decoder.layers.layer0.attentions.attn0.out_proj:
head.motion_decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.motion_decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.motion_decoder.layers.layer0.ffn:
head.motion_decoder.layers.layer0.ffn.ffn0:
head.motion_decoder.layers.layer0.ffn.ffn0.linear1:
head.motion_decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.motion_decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.motion_decoder.layers.layer0.ffn.ffn0.linear2:
head.motion_decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.motion_decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.motion_decoder.layers.layer0.norms:
head.motion_decoder.layers.layer0.norms.norm0:
head.motion_decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.motion_decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.motion_decoder.layers.layer0.norms.norm1:
head.motion_decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.motion_decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.motion_map_decoder:
head.motion_map_decoder.layers:
head.motion_map_decoder.layers.layer0:
head.motion_map_decoder.layers.layer0.attentions:
head.motion_map_decoder.layers.layer0.attentions.attn0:
head.motion_map_decoder.layers.layer0.attentions.attn0.in_proj:
head.motion_map_decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.motion_map_decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.motion_map_decoder.layers.layer0.attentions.attn0.out_proj:
head.motion_map_decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.motion_map_decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.motion_map_decoder.layers.layer0.ffn:
head.motion_map_decoder.layers.layer0.ffn.ffn0:
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear1:
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear2:
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.motion_map_decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.motion_map_decoder.layers.layer0.norms:
head.motion_map_decoder.layers.layer0.norms.norm0:
head.motion_map_decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.motion_map_decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.motion_map_decoder.layers.layer0.norms.norm1:
head.motion_map_decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.motion_map_decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.ego_map_decoder:
head.ego_map_decoder.layers:
head.ego_map_decoder.layers.layer0:
head.ego_map_decoder.layers.layer0.attentions:
head.ego_map_decoder.layers.layer0.attentions.attn0:
head.ego_map_decoder.layers.layer0.attentions.attn0.in_proj:
head.ego_map_decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.ego_map_decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.ego_map_decoder.layers.layer0.attentions.attn0.out_proj:
head.ego_map_decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.ego_map_decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.ego_map_decoder.layers.layer0.ffn:
head.ego_map_decoder.layers.layer0.ffn.ffn0:
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear1:
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear2:
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.ego_map_decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.ego_map_decoder.layers.layer0.norms:
head.ego_map_decoder.layers.layer0.norms.norm0:
head.ego_map_decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.ego_map_decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.ego_map_decoder.layers.layer0.norms.norm1:
head.ego_map_decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.ego_map_decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.ego_agent_decoder:
head.ego_agent_decoder.layers:
head.ego_agent_decoder.layers.layer0:
head.ego_agent_decoder.layers.layer0.attentions:
head.ego_agent_decoder.layers.layer0.attentions.attn0:
head.ego_agent_decoder.layers.layer0.attentions.attn0.in_proj:
head.ego_agent_decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.ego_agent_decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.ego_agent_decoder.layers.layer0.attentions.attn0.out_proj:
head.ego_agent_decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.ego_agent_decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.ego_agent_decoder.layers.layer0.ffn:
head.ego_agent_decoder.layers.layer0.ffn.ffn0:
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear1:
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear2:
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.ego_agent_decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.ego_agent_decoder.layers.layer0.norms:
head.ego_agent_decoder.layers.layer0.norms.norm0:
head.ego_agent_decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.ego_agent_decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.ego_agent_decoder.layers.layer0.norms.norm1:
head.ego_agent_decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.ego_agent_decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.lane_encoder:
head.lane_encoder.lmlp_0:
head.lane_encoder.lmlp_0.linear:
head.lane_encoder.lmlp_0.linear.weight: tensor with shape Shape([256, 128])
head.lane_encoder.lmlp_0.linear.bias: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_0.norm:
head.lane_encoder.lmlp_0.norm.weight: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_0.norm.bias: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_1:
head.lane_encoder.lmlp_1.linear:
head.lane_encoder.lmlp_1.linear.weight: tensor with shape Shape([256, 128])
head.lane_encoder.lmlp_1.linear.bias: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_1.norm:
head.lane_encoder.lmlp_1.norm.weight: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_1.norm.bias: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_2:
head.lane_encoder.lmlp_2.linear:
head.lane_encoder.lmlp_2.linear.weight: tensor with shape Shape([256, 128])
head.lane_encoder.lmlp_2.linear.bias: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_2.norm:
head.lane_encoder.lmlp_2.norm.weight: tensor with shape Shape([1, 128])
head.lane_encoder.lmlp_2.norm.bias: tensor with shape Shape([1, 128])
head.transformer:
head.transformer.encoder:
head.transformer.encoder.layers:
head.transformer.encoder.layers.layer0:
head.transformer.encoder.layers.layer0.attentions:
head.transformer.encoder.layers.layer0.attentions.attn0:
head.transformer.encoder.layers.layer0.attentions.attn0.sampling_offsets:
head.transformer.encoder.layers.layer0.attentions.attn0.sampling_offsets.weight: tensor with shape Shape([512, 128])
head.transformer.encoder.layers.layer0.attentions.attn0.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer0.attentions.attn0.attention_weights:
head.transformer.encoder.layers.layer0.attentions.attn0.attention_weights.weight: tensor with shape Shape([512, 64])
head.transformer.encoder.layers.layer0.attentions.attn0.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer0.attentions.attn0.value_proj:
head.transformer.encoder.layers.layer0.attentions.attn0.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer0.attentions.attn0.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.attentions.attn0.output_proj:
head.transformer.encoder.layers.layer0.attentions.attn0.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer0.attentions.attn0.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.attentions.attn1:
head.transformer.encoder.layers.layer0.attentions.attn1.sampling_offsets:
head.transformer.encoder.layers.layer0.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 128])
head.transformer.encoder.layers.layer0.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer0.attentions.attn1.attention_weights:
head.transformer.encoder.layers.layer0.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 64])
head.transformer.encoder.layers.layer0.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer0.attentions.attn1.value_proj:
head.transformer.encoder.layers.layer0.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer0.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.attentions.attn1.output_proj:
head.transformer.encoder.layers.layer0.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer0.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.ffn:
head.transformer.encoder.layers.layer0.ffn.ffn0:
head.transformer.encoder.layers.layer0.ffn.ffn0.linear1:
head.transformer.encoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.encoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.encoder.layers.layer0.ffn.ffn0.linear2:
head.transformer.encoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.encoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms:
head.transformer.encoder.layers.layer0.norms.norm0:
head.transformer.encoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms.norm1:
head.transformer.encoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms.norm2:
head.transformer.encoder.layers.layer0.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer0.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1:
head.transformer.encoder.layers.layer1.attentions:
head.transformer.encoder.layers.layer1.attentions.attn0:
head.transformer.encoder.layers.layer1.attentions.attn0.sampling_offsets:
head.transformer.encoder.layers.layer1.attentions.attn0.sampling_offsets.weight: tensor with shape Shape([512, 128])
head.transformer.encoder.layers.layer1.attentions.attn0.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer1.attentions.attn0.attention_weights:
head.transformer.encoder.layers.layer1.attentions.attn0.attention_weights.weight: tensor with shape Shape([512, 64])
head.transformer.encoder.layers.layer1.attentions.attn0.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer1.attentions.attn0.value_proj:
head.transformer.encoder.layers.layer1.attentions.attn0.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer1.attentions.attn0.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.attentions.attn0.output_proj:
head.transformer.encoder.layers.layer1.attentions.attn0.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer1.attentions.attn0.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.attentions.attn1:
head.transformer.encoder.layers.layer1.attentions.attn1.sampling_offsets:
head.transformer.encoder.layers.layer1.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 128])
head.transformer.encoder.layers.layer1.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer1.attentions.attn1.attention_weights:
head.transformer.encoder.layers.layer1.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 64])
head.transformer.encoder.layers.layer1.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer1.attentions.attn1.value_proj:
head.transformer.encoder.layers.layer1.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer1.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.attentions.attn1.output_proj:
head.transformer.encoder.layers.layer1.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer1.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.ffn:
head.transformer.encoder.layers.layer1.ffn.ffn0:
head.transformer.encoder.layers.layer1.ffn.ffn0.linear1:
head.transformer.encoder.layers.layer1.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.encoder.layers.layer1.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.encoder.layers.layer1.ffn.ffn0.linear2:
head.transformer.encoder.layers.layer1.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.encoder.layers.layer1.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms:
head.transformer.encoder.layers.layer1.norms.norm0:
head.transformer.encoder.layers.layer1.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms.norm1:
head.transformer.encoder.layers.layer1.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms.norm2:
head.transformer.encoder.layers.layer1.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer1.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2:
head.transformer.encoder.layers.layer2.attentions:
head.transformer.encoder.layers.layer2.attentions.attn0:
head.transformer.encoder.layers.layer2.attentions.attn0.sampling_offsets:
head.transformer.encoder.layers.layer2.attentions.attn0.sampling_offsets.weight: tensor with shape Shape([512, 128])
head.transformer.encoder.layers.layer2.attentions.attn0.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer2.attentions.attn0.attention_weights:
head.transformer.encoder.layers.layer2.attentions.attn0.attention_weights.weight: tensor with shape Shape([512, 64])
head.transformer.encoder.layers.layer2.attentions.attn0.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer2.attentions.attn0.value_proj:
head.transformer.encoder.layers.layer2.attentions.attn0.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer2.attentions.attn0.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.attentions.attn0.output_proj:
head.transformer.encoder.layers.layer2.attentions.attn0.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer2.attentions.attn0.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.attentions.attn1:
head.transformer.encoder.layers.layer2.attentions.attn1.sampling_offsets:
head.transformer.encoder.layers.layer2.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 128])
head.transformer.encoder.layers.layer2.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 128])
head.transformer.encoder.layers.layer2.attentions.attn1.attention_weights:
head.transformer.encoder.layers.layer2.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 64])
head.transformer.encoder.layers.layer2.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 64])
head.transformer.encoder.layers.layer2.attentions.attn1.value_proj:
head.transformer.encoder.layers.layer2.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer2.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.attentions.attn1.output_proj:
head.transformer.encoder.layers.layer2.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.encoder.layers.layer2.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.ffn:
head.transformer.encoder.layers.layer2.ffn.ffn0:
head.transformer.encoder.layers.layer2.ffn.ffn0.linear1:
head.transformer.encoder.layers.layer2.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.encoder.layers.layer2.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.encoder.layers.layer2.ffn.ffn0.linear2:
head.transformer.encoder.layers.layer2.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.encoder.layers.layer2.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms:
head.transformer.encoder.layers.layer2.norms.norm0:
head.transformer.encoder.layers.layer2.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms.norm1:
head.transformer.encoder.layers.layer2.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms.norm2:
head.transformer.encoder.layers.layer2.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.encoder.layers.layer2.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder:
head.transformer.decoder.layers:
head.transformer.decoder.layers.layer0:
head.transformer.decoder.layers.layer0.attentions:
head.transformer.decoder.layers.layer0.attentions.attn0:
head.transformer.decoder.layers.layer0.attentions.attn0.in_proj:
head.transformer.decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.decoder.layers.layer0.attentions.attn0.out_proj:
head.transformer.decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.attentions.attn1:
head.transformer.decoder.layers.layer0.attentions.attn1.sampling_offsets:
head.transformer.decoder.layers.layer0.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.decoder.layers.layer0.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.decoder.layers.layer0.attentions.attn1.attention_weights:
head.transformer.decoder.layers.layer0.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.decoder.layers.layer0.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.decoder.layers.layer0.attentions.attn1.value_proj:
head.transformer.decoder.layers.layer0.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer0.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.attentions.attn1.output_proj:
head.transformer.decoder.layers.layer0.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer0.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.ffn:
head.transformer.decoder.layers.layer0.ffn.ffn0:
head.transformer.decoder.layers.layer0.ffn.ffn0.linear1:
head.transformer.decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.decoder.layers.layer0.ffn.ffn0.linear2:
head.transformer.decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms:
head.transformer.decoder.layers.layer0.norms.norm0:
head.transformer.decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms.norm1:
head.transformer.decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms.norm2:
head.transformer.decoder.layers.layer0.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer0.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1:
head.transformer.decoder.layers.layer1.attentions:
head.transformer.decoder.layers.layer1.attentions.attn0:
head.transformer.decoder.layers.layer1.attentions.attn0.in_proj:
head.transformer.decoder.layers.layer1.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.decoder.layers.layer1.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.decoder.layers.layer1.attentions.attn0.out_proj:
head.transformer.decoder.layers.layer1.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer1.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.attentions.attn1:
head.transformer.decoder.layers.layer1.attentions.attn1.sampling_offsets:
head.transformer.decoder.layers.layer1.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.decoder.layers.layer1.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.decoder.layers.layer1.attentions.attn1.attention_weights:
head.transformer.decoder.layers.layer1.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.decoder.layers.layer1.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.decoder.layers.layer1.attentions.attn1.value_proj:
head.transformer.decoder.layers.layer1.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer1.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.attentions.attn1.output_proj:
head.transformer.decoder.layers.layer1.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer1.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.ffn:
head.transformer.decoder.layers.layer1.ffn.ffn0:
head.transformer.decoder.layers.layer1.ffn.ffn0.linear1:
head.transformer.decoder.layers.layer1.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.decoder.layers.layer1.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.decoder.layers.layer1.ffn.ffn0.linear2:
head.transformer.decoder.layers.layer1.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.decoder.layers.layer1.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms:
head.transformer.decoder.layers.layer1.norms.norm0:
head.transformer.decoder.layers.layer1.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms.norm1:
head.transformer.decoder.layers.layer1.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms.norm2:
head.transformer.decoder.layers.layer1.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer1.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2:
head.transformer.decoder.layers.layer2.attentions:
head.transformer.decoder.layers.layer2.attentions.attn0:
head.transformer.decoder.layers.layer2.attentions.attn0.in_proj:
head.transformer.decoder.layers.layer2.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.decoder.layers.layer2.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.decoder.layers.layer2.attentions.attn0.out_proj:
head.transformer.decoder.layers.layer2.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer2.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.attentions.attn1:
head.transformer.decoder.layers.layer2.attentions.attn1.sampling_offsets:
head.transformer.decoder.layers.layer2.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.decoder.layers.layer2.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.decoder.layers.layer2.attentions.attn1.attention_weights:
head.transformer.decoder.layers.layer2.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.decoder.layers.layer2.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.decoder.layers.layer2.attentions.attn1.value_proj:
head.transformer.decoder.layers.layer2.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer2.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.attentions.attn1.output_proj:
head.transformer.decoder.layers.layer2.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.decoder.layers.layer2.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.ffn:
head.transformer.decoder.layers.layer2.ffn.ffn0:
head.transformer.decoder.layers.layer2.ffn.ffn0.linear1:
head.transformer.decoder.layers.layer2.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.decoder.layers.layer2.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.decoder.layers.layer2.ffn.ffn0.linear2:
head.transformer.decoder.layers.layer2.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.decoder.layers.layer2.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms:
head.transformer.decoder.layers.layer2.norms.norm0:
head.transformer.decoder.layers.layer2.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms.norm1:
head.transformer.decoder.layers.layer2.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms.norm2:
head.transformer.decoder.layers.layer2.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.decoder.layers.layer2.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder:
head.transformer.map_decoder.layers:
head.transformer.map_decoder.layers.layer0:
head.transformer.map_decoder.layers.layer0.attentions:
head.transformer.map_decoder.layers.layer0.attentions.attn0:
head.transformer.map_decoder.layers.layer0.attentions.attn0.in_proj:
head.transformer.map_decoder.layers.layer0.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.map_decoder.layers.layer0.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.map_decoder.layers.layer0.attentions.attn0.out_proj:
head.transformer.map_decoder.layers.layer0.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer0.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.attentions.attn1:
head.transformer.map_decoder.layers.layer0.attentions.attn1.sampling_offsets:
head.transformer.map_decoder.layers.layer0.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.map_decoder.layers.layer0.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.map_decoder.layers.layer0.attentions.attn1.attention_weights:
head.transformer.map_decoder.layers.layer0.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.map_decoder.layers.layer0.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.map_decoder.layers.layer0.attentions.attn1.value_proj:
head.transformer.map_decoder.layers.layer0.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer0.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.attentions.attn1.output_proj:
head.transformer.map_decoder.layers.layer0.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer0.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.ffn:
head.transformer.map_decoder.layers.layer0.ffn.ffn0:
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear1:
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear2:
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.map_decoder.layers.layer0.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms:
head.transformer.map_decoder.layers.layer0.norms.norm0:
head.transformer.map_decoder.layers.layer0.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms.norm1:
head.transformer.map_decoder.layers.layer0.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms.norm2:
head.transformer.map_decoder.layers.layer0.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer0.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1:
head.transformer.map_decoder.layers.layer1.attentions:
head.transformer.map_decoder.layers.layer1.attentions.attn0:
head.transformer.map_decoder.layers.layer1.attentions.attn0.in_proj:
head.transformer.map_decoder.layers.layer1.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.map_decoder.layers.layer1.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.map_decoder.layers.layer1.attentions.attn0.out_proj:
head.transformer.map_decoder.layers.layer1.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer1.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.attentions.attn1:
head.transformer.map_decoder.layers.layer1.attentions.attn1.sampling_offsets:
head.transformer.map_decoder.layers.layer1.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.map_decoder.layers.layer1.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.map_decoder.layers.layer1.attentions.attn1.attention_weights:
head.transformer.map_decoder.layers.layer1.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.map_decoder.layers.layer1.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.map_decoder.layers.layer1.attentions.attn1.value_proj:
head.transformer.map_decoder.layers.layer1.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer1.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.attentions.attn1.output_proj:
head.transformer.map_decoder.layers.layer1.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer1.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.ffn:
head.transformer.map_decoder.layers.layer1.ffn.ffn0:
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear1:
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear2:
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.map_decoder.layers.layer1.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms:
head.transformer.map_decoder.layers.layer1.norms.norm0:
head.transformer.map_decoder.layers.layer1.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms.norm1:
head.transformer.map_decoder.layers.layer1.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms.norm2:
head.transformer.map_decoder.layers.layer1.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer1.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2:
head.transformer.map_decoder.layers.layer2.attentions:
head.transformer.map_decoder.layers.layer2.attentions.attn0:
head.transformer.map_decoder.layers.layer2.attentions.attn0.in_proj:
head.transformer.map_decoder.layers.layer2.attentions.attn0.in_proj.weight: tensor with shape Shape([256, 768])
head.transformer.map_decoder.layers.layer2.attentions.attn0.in_proj.bias: tensor with shape Shape([1, 768])
head.transformer.map_decoder.layers.layer2.attentions.attn0.out_proj:
head.transformer.map_decoder.layers.layer2.attentions.attn0.out_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer2.attentions.attn0.out_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.attentions.attn1:
head.transformer.map_decoder.layers.layer2.attentions.attn1.sampling_offsets:
head.transformer.map_decoder.layers.layer2.attentions.attn1.sampling_offsets.weight: tensor with shape Shape([256, 64])
head.transformer.map_decoder.layers.layer2.attentions.attn1.sampling_offsets.bias: tensor with shape Shape([1, 64])
head.transformer.map_decoder.layers.layer2.attentions.attn1.attention_weights:
head.transformer.map_decoder.layers.layer2.attentions.attn1.attention_weights.weight: tensor with shape Shape([256, 32])
head.transformer.map_decoder.layers.layer2.attentions.attn1.attention_weights.bias: tensor with shape Shape([1, 32])
head.transformer.map_decoder.layers.layer2.attentions.attn1.value_proj:
head.transformer.map_decoder.layers.layer2.attentions.attn1.value_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer2.attentions.attn1.value_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.attentions.attn1.output_proj:
head.transformer.map_decoder.layers.layer2.attentions.attn1.output_proj.weight: tensor with shape Shape([256, 256])
head.transformer.map_decoder.layers.layer2.attentions.attn1.output_proj.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.ffn:
head.transformer.map_decoder.layers.layer2.ffn.ffn0:
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear1:
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear1.weight: tensor with shape Shape([256, 512])
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear1.bias: tensor with shape Shape([1, 512])
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear2:
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear2.weight: tensor with shape Shape([512, 256])
head.transformer.map_decoder.layers.layer2.ffn.ffn0.linear2.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms:
head.transformer.map_decoder.layers.layer2.norms.norm0:
head.transformer.map_decoder.layers.layer2.norms.norm0.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms.norm0.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms.norm1:
head.transformer.map_decoder.layers.layer2.norms.norm1.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms.norm1.bias: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms.norm2:
head.transformer.map_decoder.layers.layer2.norms.norm2.weight: tensor with shape Shape([1, 256])
head.transformer.map_decoder.layers.layer2.norms.norm2.bias: tensor with shape Shape([1, 256])
head.transformer.reference_points:
head.transformer.reference_points.weight: tensor with shape Shape([256, 3])
head.transformer.reference_points.bias: tensor with shape Shape([1, 3])
head.transformer.map_reference_points:
head.transformer.map_reference_points.weight: tensor with shape Shape([256, 2])
head.transformer.map_reference_points.bias: tensor with shape Shape([1, 2])
head.transformer.can_bus_mlp:
head.transformer.can_bus_mlp.0:
head.transformer.can_bus_mlp.0.weight: tensor with shape Shape([18, 128])
head.transformer.can_bus_mlp.0.bias: tensor with shape Shape([1, 128])
head.transformer.can_bus_mlp.1:
head.transformer.can_bus_mlp.1.weight: tensor with shape Shape([128, 256])
head.transformer.can_bus_mlp.1.bias: tensor with shape Shape([1, 256])
head.transformer.can_bus_mlp.norm:
head.transformer.can_bus_mlp.norm.weight: tensor with shape Shape([1, 256])
head.transformer.can_bus_mlp.norm.bias: tensor with shape Shape([1, 256])
head.transformer.level_embeds: tensor with shape Shape([4, 256])
head.transformer.cams_embeds: tensor with shape Shape([6, 256])
head.pos_mlp_sa:
head.pos_mlp_sa.weight: tensor with shape Shape([2, 256])
head.pos_mlp_sa.bias: tensor with shape Shape([1, 256])
head.pos_mlp:
head.pos_mlp.weight: tensor with shape Shape([2, 256])
head.pos_mlp.bias: tensor with shape Shape([1, 256])
head.ego_agent_pos_mlp:
head.ego_agent_pos_mlp.weight: tensor with shape Shape([2, 256])
head.ego_agent_pos_mlp.bias: tensor with shape Shape([1, 256])
head.ego_map_pos_mlp:
head.ego_map_pos_mlp.weight: tensor with shape Shape([2, 256])
head.ego_map_pos_mlp.bias: tensor with shape Shape([1, 256])
head.bev_embedding:
head.bev_embedding.weight: tensor with shape Shape([10000, 256])
head.query_embedding:
head.query_embedding.weight: tensor with shape Shape([300, 512])
head.map_instance_embedding:
head.map_instance_embedding.weight: tensor with shape Shape([100, 512])
head.map_pts_embedding:
head.map_pts_embedding.weight: tensor with shape Shape([20, 512])
head.motion_mode_query:
head.motion_mode_query.weight: tensor with shape Shape([6, 256])
head.ego_query:
head.ego_query.weight: tensor with shape Shape([1, 256])
head.branches:
head.branches.cls_branches:
head.branches.cls_branches.0:
head.branches.cls_branches.0.0:
head.branches.cls_branches.0.0.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.0.0.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.1_norm:
head.branches.cls_branches.0.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.2:
head.branches.cls_branches.0.2.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.0.2.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.3_norm:
head.branches.cls_branches.0.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.0.4:
head.branches.cls_branches.0.4.weight: tensor with shape Shape([256, 10])
head.branches.cls_branches.0.4.bias: tensor with shape Shape([1, 10])
head.branches.cls_branches.1:
head.branches.cls_branches.1.0:
head.branches.cls_branches.1.0.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.1.0.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.1_norm:
head.branches.cls_branches.1.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.2:
head.branches.cls_branches.1.2.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.1.2.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.3_norm:
head.branches.cls_branches.1.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.1.4:
head.branches.cls_branches.1.4.weight: tensor with shape Shape([256, 10])
head.branches.cls_branches.1.4.bias: tensor with shape Shape([1, 10])
head.branches.cls_branches.2:
head.branches.cls_branches.2.0:
head.branches.cls_branches.2.0.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.2.0.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.1_norm:
head.branches.cls_branches.2.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.2:
head.branches.cls_branches.2.2.weight: tensor with shape Shape([256, 256])
head.branches.cls_branches.2.2.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.3_norm:
head.branches.cls_branches.2.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.cls_branches.2.4:
head.branches.cls_branches.2.4.weight: tensor with shape Shape([256, 10])
head.branches.cls_branches.2.4.bias: tensor with shape Shape([1, 10])
head.branches.reg_branches:
head.branches.reg_branches.0:
head.branches.reg_branches.0.0:
head.branches.reg_branches.0.0.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.0.0.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.0.1:
head.branches.reg_branches.0.1.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.0.1.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.0.2:
head.branches.reg_branches.0.2.weight: tensor with shape Shape([256, 10])
head.branches.reg_branches.0.2.bias: tensor with shape Shape([1, 10])
head.branches.reg_branches.1:
head.branches.reg_branches.1.0:
head.branches.reg_branches.1.0.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.1.0.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.1.1:
head.branches.reg_branches.1.1.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.1.1.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.1.2:
head.branches.reg_branches.1.2.weight: tensor with shape Shape([256, 10])
head.branches.reg_branches.1.2.bias: tensor with shape Shape([1, 10])
head.branches.reg_branches.2:
head.branches.reg_branches.2.0:
head.branches.reg_branches.2.0.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.2.0.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.2.1:
head.branches.reg_branches.2.1.weight: tensor with shape Shape([256, 256])
head.branches.reg_branches.2.1.bias: tensor with shape Shape([1, 256])
head.branches.reg_branches.2.2:
head.branches.reg_branches.2.2.weight: tensor with shape Shape([256, 10])
head.branches.reg_branches.2.2.bias: tensor with shape Shape([1, 10])
head.branches.traj_branches:
head.branches.traj_branches.0:
head.branches.traj_branches.0.0:
head.branches.traj_branches.0.0.weight: tensor with shape Shape([512, 512])
head.branches.traj_branches.0.0.bias: tensor with shape Shape([1, 512])
head.branches.traj_branches.0.1:
head.branches.traj_branches.0.1.weight: tensor with shape Shape([512, 512])
head.branches.traj_branches.0.1.bias: tensor with shape Shape([1, 512])
head.branches.traj_branches.0.2:
head.branches.traj_branches.0.2.weight: tensor with shape Shape([512, 12])
head.branches.traj_branches.0.2.bias: tensor with shape Shape([1, 12])
head.branches.map_cls_branches:
head.branches.map_cls_branches.0:
head.branches.map_cls_branches.0.0:
head.branches.map_cls_branches.0.0.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.0.0.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.1_norm:
head.branches.map_cls_branches.0.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.2:
head.branches.map_cls_branches.0.2.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.0.2.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.3_norm:
head.branches.map_cls_branches.0.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.0.4:
head.branches.map_cls_branches.0.4.weight: tensor with shape Shape([256, 3])
head.branches.map_cls_branches.0.4.bias: tensor with shape Shape([1, 3])
head.branches.map_cls_branches.1:
head.branches.map_cls_branches.1.0:
head.branches.map_cls_branches.1.0.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.1.0.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.1_norm:
head.branches.map_cls_branches.1.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.2:
head.branches.map_cls_branches.1.2.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.1.2.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.3_norm:
head.branches.map_cls_branches.1.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.1.4:
head.branches.map_cls_branches.1.4.weight: tensor with shape Shape([256, 3])
head.branches.map_cls_branches.1.4.bias: tensor with shape Shape([1, 3])
head.branches.map_cls_branches.2:
head.branches.map_cls_branches.2.0:
head.branches.map_cls_branches.2.0.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.2.0.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.1_norm:
head.branches.map_cls_branches.2.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.2:
head.branches.map_cls_branches.2.2.weight: tensor with shape Shape([256, 256])
head.branches.map_cls_branches.2.2.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.3_norm:
head.branches.map_cls_branches.2.3_norm.weight: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.3_norm.bias: tensor with shape Shape([1, 256])
head.branches.map_cls_branches.2.4:
head.branches.map_cls_branches.2.4.weight: tensor with shape Shape([256, 3])
head.branches.map_cls_branches.2.4.bias: tensor with shape Shape([1, 3])
head.branches.map_reg_branches:
head.branches.map_reg_branches.0:
head.branches.map_reg_branches.0.0:
head.branches.map_reg_branches.0.0.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.0.0.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.0.1:
head.branches.map_reg_branches.0.1.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.0.1.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.0.2:
head.branches.map_reg_branches.0.2.weight: tensor with shape Shape([256, 2])
head.branches.map_reg_branches.0.2.bias: tensor with shape Shape([1, 2])
head.branches.map_reg_branches.1:
head.branches.map_reg_branches.1.0:
head.branches.map_reg_branches.1.0.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.1.0.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.1.1:
head.branches.map_reg_branches.1.1.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.1.1.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.1.2:
head.branches.map_reg_branches.1.2.weight: tensor with shape Shape([256, 2])
head.branches.map_reg_branches.1.2.bias: tensor with shape Shape([1, 2])
head.branches.map_reg_branches.2:
head.branches.map_reg_branches.2.0:
head.branches.map_reg_branches.2.0.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.2.0.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.2.1:
head.branches.map_reg_branches.2.1.weight: tensor with shape Shape([256, 256])
head.branches.map_reg_branches.2.1.bias: tensor with shape Shape([1, 256])
head.branches.map_reg_branches.2.2:
head.branches.map_reg_branches.2.2.weight: tensor with shape Shape([256, 2])
head.branches.map_reg_branches.2.2.bias: tensor with shape Shape([1, 2])
head.branches.ego_fut_decoder:
head.branches.ego_fut_decoder.0:
head.branches.ego_fut_decoder.0.0:
head.branches.ego_fut_decoder.0.0.weight: tensor with shape Shape([512, 512])
head.branches.ego_fut_decoder.0.0.bias: tensor with shape Shape([1, 512])
head.branches.ego_fut_decoder.1:
head.branches.ego_fut_decoder.2:
head.branches.ego_fut_decoder.2.0:
head.branches.ego_fut_decoder.2.0.weight: tensor with shape Shape([512, 512])
head.branches.ego_fut_decoder.2.0.bias: tensor with shape Shape([1, 512])
head.branches.ego_fut_decoder.3:
head.branches.ego_fut_decoder.4:
head.branches.ego_fut_decoder.4.0:
head.branches.ego_fut_decoder.4.0.weight: tensor with shape Shape([512, 36])
head.branches.ego_fut_decoder.4.0.bias: tensor with shape Shape([1, 36])
head.branches.agent_fus_mlp:
head.branches.agent_fus_mlp.0:
head.branches.agent_fus_mlp.0.weight: tensor with shape Shape([3072, 256])
head.branches.agent_fus_mlp.0.bias: tensor with shape Shape([1, 256])
head.branches.agent_fus_mlp.1_norm:
head.branches.agent_fus_mlp.1_norm.weight: tensor with shape Shape([1, 256])
head.branches.agent_fus_mlp.1_norm.bias: tensor with shape Shape([1, 256])
head.branches.agent_fus_mlp.2:
head.branches.agent_fus_mlp.3:
head.branches.agent_fus_mlp.3.weight: tensor with shape Shape([256, 256])
head.branches.agent_fus_mlp.3.bias: tensor with shape Shape([1, 256])
head.branches.traj_cls_branches:
head.branches.traj_cls_branches.0:
head.branches.traj_cls_branches.0.0:
head.branches.traj_cls_branches.0.0.weight: tensor with shape Shape([512, 512])
head.branches.traj_cls_branches.0.0.bias: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.1_norm:
head.branches.traj_cls_branches.0.1_norm.weight: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.1_norm.bias: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.2:
head.branches.traj_cls_branches.0.2.weight: tensor with shape Shape([512, 512])
head.branches.traj_cls_branches.0.2.bias: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.3_norm:
head.branches.traj_cls_branches.0.3_norm.weight: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.3_norm.bias: tensor with shape Shape([1, 512])
head.branches.traj_cls_branches.0.4:
head.branches.traj_cls_branches.0.4.weight: tensor with shape Shape([512, 1])
head.branches.traj_cls_branches.0.4.bias: tensor with shape Shape([1, 1])
img_neck:
img_neck.lateral_convs:
img_neck.lateral_convs.conv:
img_neck.lateral_convs.conv.weight: tensor with shape Shape([256, 2048, 1, 1])
img_neck.lateral_convs.conv.bias: tensor with shape Shape([256])
img_neck.fpn_convs:
img_neck.fpn_convs.conv:
img_neck.fpn_convs.conv.weight: tensor with shape Shape([256, 256, 3, 3])
img_neck.fpn_convs.conv.bias: tensor with shape Shape([256])
img_backbone:
img_backbone.conv1:
img_backbone.conv1.weight: tensor with shape Shape([64, 3, 7, 7])
img_backbone.conv1.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_0:
img_backbone.layer1_0.conv1:
img_backbone.layer1_0.conv1.weight: tensor with shape Shape([64, 64, 1, 1])
img_backbone.layer1_0.conv1.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_0.conv2:
img_backbone.layer1_0.conv2.weight: tensor with shape Shape([64, 64, 3, 3])
img_backbone.layer1_0.conv2.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_0.conv3:
img_backbone.layer1_0.conv3.weight: tensor with shape Shape([256, 64, 1, 1])
img_backbone.layer1_0.conv3.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer1_0.downsample:
img_backbone.layer1_0.downsample.weight: tensor with shape Shape([256, 64, 1, 1])
img_backbone.layer1_0.downsample.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer1_1:
img_backbone.layer1_1.conv1:
img_backbone.layer1_1.conv1.weight: tensor with shape Shape([64, 256, 1, 1])
img_backbone.layer1_1.conv1.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_1.conv2:
img_backbone.layer1_1.conv2.weight: tensor with shape Shape([64, 64, 3, 3])
img_backbone.layer1_1.conv2.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_1.conv3:
img_backbone.layer1_1.conv3.weight: tensor with shape Shape([256, 64, 1, 1])
img_backbone.layer1_1.conv3.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer1_2:
img_backbone.layer1_2.conv1:
img_backbone.layer1_2.conv1.weight: tensor with shape Shape([64, 256, 1, 1])
img_backbone.layer1_2.conv1.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_2.conv2:
img_backbone.layer1_2.conv2.weight: tensor with shape Shape([64, 64, 3, 3])
img_backbone.layer1_2.conv2.bias: tensor with shape Shape([1, 1, 1, 64])
img_backbone.layer1_2.conv3:
img_backbone.layer1_2.conv3.weight: tensor with shape Shape([256, 64, 1, 1])
img_backbone.layer1_2.conv3.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer2_0:
img_backbone.layer2_0.conv1:
img_backbone.layer2_0.conv1.weight: tensor with shape Shape([128, 256, 1, 1])
img_backbone.layer2_0.conv1.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_0.conv2:
img_backbone.layer2_0.conv2.weight: tensor with shape Shape([128, 128, 3, 3])
img_backbone.layer2_0.conv2.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_0.conv3:
img_backbone.layer2_0.conv3.weight: tensor with shape Shape([512, 128, 1, 1])
img_backbone.layer2_0.conv3.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer2_0.downsample:
img_backbone.layer2_0.downsample.weight: tensor with shape Shape([512, 256, 1, 1])
img_backbone.layer2_0.downsample.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer2_1:
img_backbone.layer2_1.conv1:
img_backbone.layer2_1.conv1.weight: tensor with shape Shape([128, 512, 1, 1])
img_backbone.layer2_1.conv1.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_1.conv2:
img_backbone.layer2_1.conv2.weight: tensor with shape Shape([128, 128, 3, 3])
img_backbone.layer2_1.conv2.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_1.conv3:
img_backbone.layer2_1.conv3.weight: tensor with shape Shape([512, 128, 1, 1])
img_backbone.layer2_1.conv3.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer2_2:
img_backbone.layer2_2.conv1:
img_backbone.layer2_2.conv1.weight: tensor with shape Shape([128, 512, 1, 1])
img_backbone.layer2_2.conv1.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_2.conv2:
img_backbone.layer2_2.conv2.weight: tensor with shape Shape([128, 128, 3, 3])
img_backbone.layer2_2.conv2.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_2.conv3:
img_backbone.layer2_2.conv3.weight: tensor with shape Shape([512, 128, 1, 1])
img_backbone.layer2_2.conv3.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer2_3:
img_backbone.layer2_3.conv1:
img_backbone.layer2_3.conv1.weight: tensor with shape Shape([128, 512, 1, 1])
img_backbone.layer2_3.conv1.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_3.conv2:
img_backbone.layer2_3.conv2.weight: tensor with shape Shape([128, 128, 3, 3])
img_backbone.layer2_3.conv2.bias: tensor with shape Shape([1, 1, 1, 128])
img_backbone.layer2_3.conv3:
img_backbone.layer2_3.conv3.weight: tensor with shape Shape([512, 128, 1, 1])
img_backbone.layer2_3.conv3.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer3_0:
img_backbone.layer3_0.conv1:
img_backbone.layer3_0.conv1.weight: tensor with shape Shape([256, 512, 1, 1])
img_backbone.layer3_0.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_0.conv2:
img_backbone.layer3_0.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_0.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_0.conv3:
img_backbone.layer3_0.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_0.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_0.downsample:
img_backbone.layer3_0.downsample.weight: tensor with shape Shape([1024, 512, 1, 1])
img_backbone.layer3_0.downsample.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_1:
img_backbone.layer3_1.conv1:
img_backbone.layer3_1.conv1.weight: tensor with shape Shape([256, 1024, 1, 1])
img_backbone.layer3_1.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_1.conv2:
img_backbone.layer3_1.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_1.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_1.conv3:
img_backbone.layer3_1.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_1.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_2:
img_backbone.layer3_2.conv1:
img_backbone.layer3_2.conv1.weight: tensor with shape Shape([256, 1024, 1, 1])
img_backbone.layer3_2.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_2.conv2:
img_backbone.layer3_2.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_2.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_2.conv3:
img_backbone.layer3_2.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_2.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_3:
img_backbone.layer3_3.conv1:
img_backbone.layer3_3.conv1.weight: tensor with shape Shape([256, 1024, 1, 1])
img_backbone.layer3_3.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_3.conv2:
img_backbone.layer3_3.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_3.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_3.conv3:
img_backbone.layer3_3.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_3.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_4:
img_backbone.layer3_4.conv1:
img_backbone.layer3_4.conv1.weight: tensor with shape Shape([256, 1024, 1, 1])
img_backbone.layer3_4.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_4.conv2:
img_backbone.layer3_4.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_4.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_4.conv3:
img_backbone.layer3_4.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_4.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer3_5:
img_backbone.layer3_5.conv1:
img_backbone.layer3_5.conv1.weight: tensor with shape Shape([256, 1024, 1, 1])
img_backbone.layer3_5.conv1.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_5.conv2:
img_backbone.layer3_5.conv2.weight: tensor with shape Shape([256, 256, 3, 3])
img_backbone.layer3_5.conv2.bias: tensor with shape Shape([1, 1, 1, 256])
img_backbone.layer3_5.conv3:
img_backbone.layer3_5.conv3.weight: tensor with shape Shape([1024, 256, 1, 1])
img_backbone.layer3_5.conv3.bias: tensor with shape Shape([1, 1, 1, 1024])
img_backbone.layer4_0:
img_backbone.layer4_0.conv1:
img_backbone.layer4_0.conv1.weight: tensor with shape Shape([512, 1024, 1, 1])
img_backbone.layer4_0.conv1.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_0.conv2:
img_backbone.layer4_0.conv2.weight: tensor with shape Shape([512, 512, 3, 3])
img_backbone.layer4_0.conv2.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_0.conv3:
img_backbone.layer4_0.conv3.weight: tensor with shape Shape([2048, 512, 1, 1])
img_backbone.layer4_0.conv3.bias: tensor with shape Shape([1, 1, 1, 2048])
img_backbone.layer4_0.downsample:
img_backbone.layer4_0.downsample.weight: tensor with shape Shape([2048, 1024, 1, 1])
img_backbone.layer4_0.downsample.bias: tensor with shape Shape([1, 1, 1, 2048])
img_backbone.layer4_1:
img_backbone.layer4_1.conv1:
img_backbone.layer4_1.conv1.weight: tensor with shape Shape([512, 2048, 1, 1])
img_backbone.layer4_1.conv1.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_1.conv2:
img_backbone.layer4_1.conv2.weight: tensor with shape Shape([512, 512, 3, 3])
img_backbone.layer4_1.conv2.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_1.conv3:
img_backbone.layer4_1.conv3.weight: tensor with shape Shape([2048, 512, 1, 1])
img_backbone.layer4_1.conv3.bias: tensor with shape Shape([1, 1, 1, 2048])
img_backbone.layer4_2:
img_backbone.layer4_2.conv1:
img_backbone.layer4_2.conv1.weight: tensor with shape Shape([512, 2048, 1, 1])
img_backbone.layer4_2.conv1.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_2.conv2:
img_backbone.layer4_2.conv2.weight: tensor with shape Shape([512, 512, 3, 3])
img_backbone.layer4_2.conv2.bias: tensor with shape Shape([1, 1, 1, 512])
img_backbone.layer4_2.conv3:
img_backbone.layer4_2.conv3.weight: tensor with shape Shape([2048, 512, 1, 1])
img_backbone.layer4_2.conv3.bias: tensor with shape Shape([1, 1, 1, 2048])
""".strip()

def tp(shape, device=None):
        return ttnn._rand(shape=shape, dtype=ttnn.bfloat16, device=device)

def build_nested_parameter_dict(spec_text: str, device=None):
    """
    Parse lines like:
      'head.positional_encoding.row_embed.weight: tensor with shape Shape([100, 128])'
    into a nested dict:
      parameter['head']['positional_encoding']['row_embed']['weight'] = tp(100, 128)
    """
    line_re = re.compile(
        r"^(?P<name>.*?):\s*tensor with shape Shape\(\[(?P<shape>[0-9,\s]+)\]\)\s*$"
    )

    parameter: dict = {}

    for raw_line in spec_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = line_re.match(line)
        if not m:
            # Skip container/module lines like 'head:' or 'head.motion_decoder.layers:'
            continue

        full_name = m.group("name")  # e.g. 'head.positional_encoding.row_embed.weight'
        shape_str = m.group("shape") # e.g. '100, 128'

        # Convert '100, 128' -> (100, 128)
        tshape = tuple(int(x.strip()) for x in shape_str.split(","))

        # Build nested structure
        keys = full_name.split(".")
        d = parameter
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = tp(tshape, device=device)

    # parameter.conv_args["img_backbone"] = {}
    conv_args_param = Parameter()
    parameter['conv_args'] = conv_args_param.conv_args
    parameter['img_backbone'] = conv_args_param.res_model
    parameter['conv_args_img_neck'] = ConvParams()
    return parameter

def create_vadv2_model_parameters_vad(device=None):
    return build_nested_parameter_dict(spec, device=device)
