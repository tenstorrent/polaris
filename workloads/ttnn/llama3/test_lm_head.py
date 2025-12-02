#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.lm_head import LMHead
from workloads.ttnn.llama3.model_config import ModelArgs

def test_lm_head_inference():
    seq_len = 32
    batch_size = 1
    dtype = ttnn.bfloat8_b
    mesh_device = ttnn.open_device(device_id=0)  # Assuming device_id 0 for simplicity
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len)
    model_args.n_layers = 1
    model_args.WEIGHTS_DTYPE = dtype

    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=None, #state_dict,
        state_dict_prefix=None, #state_dict_prefix,
        weight_cache_path=None, #model_args.weight_cache_path(dtype),
    )

    torch_input = ttnn._rand(shape=(1, 1, seq_len, model_args.dim), device=mesh_device, dtype=dtype)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=None, #model_args.model_config["LM_HEAD_INPUT_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model(tt_input)
    if (tt_output.shape == [1, 1, 32, 128256]):
        logger.info(f"LM Head output shape {tt_output.shape} is as expected.")
    else:
        logger.info(f"LM Head output shape is not as expected: {tt_output.shape}")

if __name__ == "__main__":
    test_lm_head_inference()
