# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import ttsim.front.ttnn as ttnn
from ttsim.ops.tensor import require_shape_list
from ttsim.front.ttnn.tensor import DataType
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.decoder import TransformerBlock
from workloads.ttnn.tt_transformers.rope import RotarySetup

def test_decoder_prefill_inference(wln, mesh_device, gcfg):
    logger.info("Starting Polaris Decoder PREFILL Smoke Test (Mistral-7B Config)...")

    batch_size = 1
    seq_len = 128  # 128 tokens for Prefill phase
    model_args = ModelArgs(mesh_device, model_name="Mistral-7B")

    # Initialize the PRODUCTION RotarySetup
    rope_setup = RotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    logger.info("RotarySetup Initialized.")

    # RoPE Transformation Matrix
    trans_mat_raw = ttnn._rand(shape=[1, 1, 32, 32], device=mesh_device, dtype=ttnn.bfloat16)
    transformation_mats = {"prefill": ttnn.to_layout(trans_mat_raw, ttnn.TILE_LAYOUT)}

    # Initialize the REAL TT Model
    tt_model = TransformerBlock(
        mesh_device=mesh_device,
        state_dict={},
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        transformation_mats=transformation_mats,
        args=model_args
    )
    logger.info("TransformerBlock Initialized Successfully.")

    # 1. Input Tensor [Batch, 1, Seq, Dim]
    logger.info(f"Creating Input Tensor: [1, 1, {seq_len}, {model_args.dim}]")
    input_raw = ttnn._rand(shape=[batch_size, 1, seq_len, model_args.dim], device=mesh_device, dtype=ttnn.bfloat16)
    tt_input = ttnn.to_layout(input_raw, ttnn.TILE_LAYOUT)

    # 2. RoPE matrices via get_rot_mats (Mirrored from Decode script exactly)
    current_pos = ttnn.Tensor(shape=(1, seq_len), device=mesh_device, dtype=ttnn.int32)

    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    rot_mats = rope_setup.get_rot_mats(current_pos_tensor)
    logger.info("Running Forward Prefill Pass...")

    # Forward Pass
    tt_out = tt_model(
        tt_input,
        current_pos=None,
        rot_mats=rot_mats,
        mode="prefill"
    )

    actual_shape = require_shape_list(tt_out.shape, "decoder output must have a known shape")
    logger.info(f"Output Prefill Shape: {actual_shape}")

    # 3. Generic shape validation
    in_shape = require_shape_list(tt_input.shape, "decoder input must have a known shape")
    if actual_shape == in_shape:
        logger.success("Prefill Test Passed! Output perfectly matches input shape.")
    else:
        logger.error(f"Shape Mismatch! Expected {in_shape}, got {actual_shape}")

    ttnn.deallocate(tt_out)
    logger.info("Done.")
