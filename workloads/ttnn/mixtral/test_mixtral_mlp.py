#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
from workloads.ttnn.mixtral.mixtral_mlp import TtMixtralMLP
from workloads.ttnn.tt_transformers.model_config import ModelArgs

def test_mixtral_mlp_inference(mesh_device, model_name="mixtral-8x7B", mode="prefill"):
    seqlen = 32
    dtypes = {
        "w1": ttnn.bfloat8_b,
        "w2": ttnn.bfloat8_b,
        "w3": ttnn.bfloat8_b,
    }

    model_args = ModelArgs(mesh_device, model_name=model_name)
    model_args.n_layers = 32
    state_dict = None
    batch = 32
    tt_model = TtMixtralMLP(mesh_device=mesh_device, state_dict=state_dict, args=model_args, layer_num=0, dtypes=dtypes)
    torch_input = ttnn._rand(shape=(1, batch, seqlen, model_args.dim), device=mesh_device, dtype=ttnn.bfloat16)

    if mode == "decode":
        tt_input = ttnn.as_tensor(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=None,
        )
    else: # prefill
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    logger.info(f"Running Mixtral MLP forward pass for {mode}...")
    logger.info(f"TT input shape is: {tt_input.shape}")
    tt_output = tt_model(tt_input, mode)
    logger.info(f"TT output shape is: {tt_output.shape}")
    assert tt_output.shape == [1, batch, seqlen, model_args.dim], f"Expected output shape {(1, batch, seqlen, model_args.dim)}, but got {tt_output.shape}"
    logger.info("Test passed!")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_mixtral_mlp_inference(mesh_device, model_name="mixtral-8x7B", mode="prefill")
    test_mixtral_mlp_inference(mesh_device, model_name="mixtral-8x7B", mode="decode")
