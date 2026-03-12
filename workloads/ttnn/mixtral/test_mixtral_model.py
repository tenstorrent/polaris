#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.model import Transformer
from loguru import logger

def test_model_inference(
    wlname: str,
    mesh_device,
    cfg: dict
):
    max_seq_len = 256
    batch_size = 1
    dtype = ttnn.bfloat8_b
    layers = cfg.get('layers')

    model_args = ModelArgs(
        model_name=cfg.get('model_name'),
        mesh_device=mesh_device,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    iterations = 4

    if layers is not None:
        model_args.n_layers = layers
    state_dict = model_args.load_state_dict()

    encoded_prompts = [[1619, 1117, 1032, 2137]]
    generation_start_pos = 0
    generation_length = iterations
    page_table_tt = None
    paged_attention_config = None

    # Load TTNN model
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    logger.info("Model and caches loaded.")

    seqlen = 1
    batch = model_args.max_batch_size
    encoded_prompts_tensor = ttnn._rand(shape=(len(encoded_prompts), batch), device=mesh_device, dtype=ttnn.int32)
    tt_decode_input = tt_model.embd(encoded_prompts_tensor).view(seqlen, batch, -1)

    generation_pos = [generation_start_pos for _ in range(batch)]
    current_pos = ttnn._rand(shape=(len(generation_pos),), device=mesh_device, dtype=ttnn.int32)
    current_pos = current_pos.unsqueeze(0)
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        logger.info(f"[Model] Generating token {i}")
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            None,
            args=model_args,
        )
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)

        logger.info(f"input shape is {decode_input.shape}")
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        assert tt_out.shape == [1, 1, 1, model_args.vocab_size//8], f"Expected output shape {(1, 1, 1, model_args.vocab_size//8)}, but got {tt_out.shape}"
        logger.info(f"Output shape is correct [1, 1, 1, {model_args.vocab_size//8}]")
        ttnn.deallocate(tt_out)

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_model_inference(
        wlname="mixtral_model_inference",
        mesh_device=mesh_device,
        cfg={'model_name': "mixtral-8x7b", 'layers': 2},
    )
