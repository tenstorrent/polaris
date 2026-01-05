#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.attention import Attention
from workloads.ttnn.llama3.rope import get_prefill_rot_mat, get_rot_transformation_mat
from workloads.ttnn.llama3.model_config import ModelArgs
from loguru import logger

def test_attention_inference():
    mesh_device = ttnn.open_device(device_id=0)#, device_type="ttnn", device_name=None)
    max_seq_len = 256
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)

    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        mesh_device,
        max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim, device=mesh_device)

    transformation_mat_torch = ttnn._rand(transformation_mat_torch.shape, device=mesh_device, dtype=dtype)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=None, #ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    generation_start_pos = 0
    generation_length = 3
    all_tests_pass = True

    # Setup page table
    page_table_tt = None
    paged_attention_config = None
    state_dict = None

    tt_model = Attention(
        mesh_device,
        state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        configuration=model_args,
        paged_attention_config=paged_attention_config,
    )

    attention_input = ttnn._rand((1, 1, 256, model_args.dim), device=mesh_device, dtype=ttnn.bfloat16)

    tt_out = tt_model(
        attention_input,
        current_pos=None,
        rot_mats=rot_mats,
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
    )
    tt_out = ttnn.to_torch(
        tt_out,# mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape)
    )

    if (tt_out.shape == [1, 1, 256, model_args.dim]):
        logger.info(f"TT Output shape matches expected: {tt_out.shape} == [1, 1, 256, {model_args.dim}]")
    else:
        logger.info(f"TT Output shape mismatch: {tt_out.shape} != [1, 1, 256, {model_args.dim}]")

    check_kv_cache = False  # set to false for simulation

    # logger.info("\nGenerating computation graph...")
    # g = mesh_device.get_graph()
    # g.graph2onnx('test_attn_prefill.onnx', do_model_check=False)

if __name__ == "__main__":
    test_attention_inference()
    logger.info("Attention test completed!")
