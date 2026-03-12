#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from loguru import logger
from workloads.ttnn.tt_transformers.rope import get_prefill_rot_mat, get_rot_transformation_mat
from workloads.ttnn.tt_transformers.decoder import TransformerBlock as TtTransformerBlock
from workloads.ttnn.tt_transformers.model_config import ModelArgs


def test_mixtral_decoder_inference(model_name, mesh_device):
    dtype = ttnn.bfloat8_b
    mode = "prefill"
    batch = 1
    max_seq_len = 4096

    model_args = ModelArgs(model_name=model_name, mesh_device=mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Initialize TT model
    rot_mats = get_prefill_rot_mat(
        model_args.head_dim,
        mesh_device,
        max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling.factor if model_args.rope_scaling else None, # type: ignore[attr-defined]
        model_args.orig_context_len,
    )
    transformation_mat_torch = get_rot_transformation_mat(model_args.head_dim, device=mesh_device)
    transformation_mats_prefill = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    transformation_mats = {"prefill": transformation_mats_prefill}

    tt_model = TtTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        args=model_args,
    )

    generation_length = 10

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")
        decode_input = ttnn._rand(shape=(1, batch, max_seq_len, int(model_args.dim)//8), device=mesh_device, dtype=ttnn.float32) # type: ignore[arg-type]

        # Run TT model
        tt_out = tt_model(
            decode_input,
            None,
            rot_mats,
            user_id=0,
            mode=mode,
        )
        logger.info(f"TT output shape is: {tt_out.shape}")
        assert tt_out.shape == [1, batch, max_seq_len, int(model_args.dim)//8], f"Expected output shape {(1, batch, max_seq_len, int(model_args.dim)//8)}, but got {tt_out.shape}" # type: ignore[arg-type]
        logger.info(f"Token {i} generated successfully with correct output shape.")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_mixtral_decoder_inference(mesh_device=mesh_device, model_name="mixtral-8x7B")
