#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
from workloads.ttnn.mixtral.mixtral_rmsnorm import RMSNorm
from workloads.ttnn.tt_transformers.model_config import ModelArgs

def test_rms_norm_inference(model_name, mesh_device):
    dtype = ttnn.bfloat16
    max_seq_len = 128000
    seq_len = 128
    batch_size = 32
    norm_type = "ffn"

    model_args = ModelArgs(mesh_device, model_name=model_name, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    tt_inner_norm = RMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=None,
        weight_key=f"{norm_type}_norm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
    )

    input = ttnn._rand(shape=(1, batch_size, seq_len, 4096//model_args.num_experts), device=mesh_device, dtype=ttnn.uint32)

    tt_input = ttnn.from_torch(
        input,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.Layout.TILE_LAYOUT,
        mesh_mapper=None,
    )

    logger.info(f"input shape: {tt_input.shape}")
    tt_output = tt_inner_norm(tt_input)
    logger.info(f"output shape: {tt_output.shape}")
    assert tt_output.shape == [1, batch_size, seq_len, 4096], f"Expected output shape {(1, batch_size, seq_len, 4096)}, but got {tt_output.shape}"
    logger.info("Test passed!")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_rms_norm_inference(model_name='mixtral-8x7B', mesh_device=mesh_device)