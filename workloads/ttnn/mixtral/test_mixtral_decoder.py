#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.rope import RotarySetup
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.decoder import TransformerBlock as TtTransformerBlock
from loguru import logger

def test_mixtral_decoder_inference(mesh_device, model_name, layer_idx):
    """
    b: batch
    s: sequence length
    h: hidden size
    """

    batch = 32
    dtype = ttnn.bfloat8_b

    if batch == 32:
        generation_start_pos = 15000
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 30000
        max_seq_len = 32768
    elif batch == 1:
        generation_start_pos = 0
        max_seq_len = 128
    else:
        raise ValueError(f"Batch size {batch} not supported")

    model_args = ModelArgs(model_name=model_name, mesh_device=mesh_device, max_seq_len=max_seq_len, max_batch_size=batch)
    model_args.use_qk_fused = False
    state_dict = model_args.load_state_dict()

    # Initialize TT model
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling,
    )
    tt_model = TtTransformerBlock(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=layer_idx,
        dtype=dtype,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=rope_setup.get_both_trans_mats(),
    )

    generation_length = 10
    seqlen = 1

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        pt_decode_input_bsh = ttnn._rand(shape=(batch, seqlen, model_args.dim), device=mesh_device, dtype=ttnn.float32)
        start_pos = generation_start_pos + i
        start_pos_ids = ttnn.full(shape=(batch,), fill_value=start_pos, device=mesh_device, layout=ttnn.Layout.TILE_LAYOUT, dtype=ttnn.int32)
        start_pos_ids = start_pos_ids.unsqueeze(0)
        current_pos_tensor = ttnn.from_torch(
            start_pos_ids,
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

        tt_decode_input = pt_decode_input_bsh.clone()
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            None,
            args=model_args
        )
        rot_mats = rope_setup.get_rot_mats(start_pos_ids)

        # Run TT model
        logger.info(f"Decode input shape: {decode_input.shape}")
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode='decode',
        )

        logger.info(f"TT output shape is: {tt_out.shape}")
        assert tt_out.shape == [1, seqlen, batch, 512], \
                f"Expected output shape {(1, seqlen, batch, 512)}, but got {tt_out.shape}"
        logger.info(f"Token {i} generated successfully with correct output shape.")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_mixtral_decoder_inference(mesh_device, model_name="mixtral-8x7B", layer_idx=0)
