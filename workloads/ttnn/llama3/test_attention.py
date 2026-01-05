#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
import typing
from numpy import shape
from loguru import logger
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.attention import Attention
from workloads.ttnn.llama3.rope import RotarySetup, precompute_freqs
from workloads.ttnn.llama3.model_config import ModelArgs

def filter_ttnn_attrs(attrs_dict):
    return {k: v for k, v in attrs_dict.items() if not (isinstance(v, ttnn.Tensor) or k == "layout" or k == "memory_config")}

def test_attention_inference():
    mesh_device = ttnn.open_device(device_id=0)#, device_type="ttnn", device_name=None)
    max_seq_len = 256
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    dtype = ttnn.bfloat8_b
    batch_size = 1
    dtype = ttnn.bfloat8_b
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1  # For the unit test, just run a single layer

    first_layer_prefix = f"Attention 0"
    seq_len = 1
    
    generation_start_pos = 0
    generation_length = 1

    # Setup RoPE transformation matrices
    rope_setup = RotarySetup(
        mesh_device,
        batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

    page_table_tt = None
    paged_attention_config = None

    ## commenting out paged attention for simulation
    # if paged_attention: # set to false for simulation
    #     paged_attention_config = PagedAttentionConfig(
    #         block_size=page_params["page_block_size"],
    #         max_num_blocks=page_params["page_max_num_blocks"],
    #     )

    #     # Implied shuffling of blocks
    #     permutation = torch.randperm(paged_attention_config.max_num_blocks)
    #     # Page table which maps virtual blocks to physical
    #     reverse_permutation = torch.argsort(permutation)
    #     page_table = reverse_permutation.reshape(
    #         model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
    #     )
    #     page_table_tt = ttnn.from_torch(
    #         page_table,
    #         device=mesh_device,
    #         dtype=ttnn.int32,
    #         layout=ttnn.ROW_MAJOR_LAYOUT,
    #         mesh_mapper=ttnn.ShardTensor2dMesh(
    #             mesh_device,
    #             dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
    #             mesh_shape=model_args.cluster_shape,
    #         ),
    #     )

    from typing import Dict
    state_dict: typing.Dict[str, int] = {} # Placeholder for actual state_dict loading logic
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

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
        device=mesh_device,
    )

    # Initial positions
    current_pos = ttnn.Tensor(shape=(batch_size,), device=mesh_device, dtype=ttnn.uint32)#, data=[generation_start_pos] * batch_size)
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.uint32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        data=current_pos.data,
    )

    for i in range(generation_length):
        pt_attention_input = ttnn._rand(shape=(batch_size, seq_len, model_args.dim), device=mesh_device, dtype=dtype)
        tt_attention_input = pt_attention_input.clone()
        attention_input = model_args.prepare_residual_tensor_decode(
            tt_attention_input,
            None, #model_args.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            force_replicated=False if model_args.is_galaxy else True,
        )

        # Get cos/sin matrices for the current position of each user
        current_pos = current_pos.unsqueeze(0)
        rot_mats = rope_setup.get_rot_mats(current_pos)
        
        tt_out = tt_model(
            attention_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        if (tt_out.shape[0] != seq_len) or (tt_out.shape[1] != batch_size) or (tt_out.shape[2] != 1) or (tt_out.shape[3] != (model_args.head_dim * model_args.n_heads)):
            logger.info(f"tt_out shape: {tt_out.shape}, Tests Failed!")
        else:
            logger.info(f"tt_out shape: {tt_out.shape}, Tests Passed!")

        current_pos = ttnn.Tensor(shape=(batch_size,), device=mesh_device, dtype=ttnn.uint32)
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=mesh_device,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        check_kv_cache = False
        ## commenting out kv cache check for simulation
        # if check_kv_cache: # set to false for simulation
        #     # PyTorch output --------------------------------------------------------------------
        #     pytorch_layer_present = [
        #         reference_model.cache_k.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
        #         reference_model.cache_v.clone().permute(0, 2, 1, 3),  # [batch_size, n_kv_heads, seq, head_dim]
        #     ]
        #     # TT hardware execution -------------------------------------------------------------
        #     if paged_attention:
        #         tt_layer_present = [
        #             (
        #                 ttnn.to_torch(
        #                     cache,
        #                     mesh_composer=ttnn.ConcatMesh2dToTensor(
        #                         mesh_device,
        #                         dims=(1, 3) if model_args.is_galaxy else (0, 1),
        #                         mesh_shape=model_args.cluster_shape,
        #                     ),
        #                 )[reverse_permutation][:, : model_args.n_kv_heads, :, : model_args.head_dim]
        #                 .reshape(
        #                     model_args.max_batch_size,
        #                     paged_attention_config.max_num_blocks // model_args.max_batch_size,
        #                     model_args.n_kv_heads,
        #                     paged_attention_config.block_size,
        #                     model_args.head_dim,
        #                 )
        #                 .transpose(1, 2)
        #                 .reshape(model_args.max_batch_size, model_args.n_kv_heads, -1, model_args.head_dim)[
        #                     :batch_size, ...
        #                 ]
        #             )
        #             for cache in tt_model.layer_past
        #         ]
        #     else:
        #         tt_layer_present = [
        #             ttnn.to_torch(
        #                 cache,
        #                 mesh_composer=ttnn.ConcatMesh2dToTensor(
        #                     mesh_device,
        #                     dims=(1, 0) if model_args.is_galaxy else (0, 1),
        #                     mesh_shape=model_args.cluster_shape,
        #                 ),
        #             )[:batch_size, :, :, :]
        #             for cache in tt_model.layer_past
        #         ]
        #     for label, cache_pt, cache_tt in zip(["K", "V"], pytorch_layer_present, tt_layer_present):
        #         cache_length_to_check = min(model_args.max_seq_len, generation_start_pos + i + 1)
        #         cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
        #         cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
        #         does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
        #         logger.info(f"{label} cache output: {output_pcc}")
        #         if does_pass:
        #             logger.info(f"{label} cache Passed!")
        #         else:
        #             logger.warning(f"{label} Cache Failed! PCC value is lower than {pcc}")
        #             all_tests_pass = False

    # Graph generation failed - TypeError(f"'{value}' is not an accepted attribute value.")
    # logger.info("\nGenerating computation graph...")
    # g = mesh_device.get_graph()
    # g.graph2onnx('test_attn.onnx', do_model_check=False,
    #             filter_op_attrs=filter_ttnn_attrs)

if __name__ == "__main__":
    test_attention_inference()
    logger.info("All tests completed!")
