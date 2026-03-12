#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
from workloads.ttnn.mixtral.mixtral_mlp import TtMixtralMLP
from workloads.ttnn.mixtral.mixtral_moe import TtMoeLayer
from workloads.ttnn.tt_transformers.model_config import ModelArgs

def test_mixtral_moe_inference(mesh_device, model_name, mode):
    iterations = 1
    dtype = ttnn.bfloat8_b
    model_args = ModelArgs(mesh_device, model_name=model_name)
    state_dict = model_args.load_state_dict()
    model_args.n_layers = 1
    layer_num = 0

    tt_model = TtMoeLayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        experts=TtMixtralMLP(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=model_args,
            layer_num=layer_num,
            dtypes={
                "w1": dtype,
                "w2": dtype,
                "w3": dtype,
            },
        ),
        args=model_args,
        layer_num=layer_num,
        dtype=dtype,
        tt_ccl=None,
    )

    seqlen = 1
    batch = 32

    for i in range(iterations):
        logger.info(f"{mode} Generating token {i}")
        _input = ttnn._rand(shape=(seqlen, batch, model_args.dim), device=mesh_device, dtype=ttnn.bfloat16)

        # TT Model Output
        logger.info(f"Starting TT Mixtral MOE")
        logger.info(f"TT input shape is: {_input.shape}")
        tt_out = tt_model(_input, mode)
        logger.info(f"output shape is {tt_out.shape}")
        assert tt_out.shape == [1, seqlen, batch, model_args.dim // model_args.num_experts], f"Expected output shape {(1, seqlen, batch, model_args.dim // model_args.num_experts)}, but got {tt_out.shape}" #type: ignore[operator]
        logger.info(f"Passed! TT Mixtral MOE {mode}.")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_mixtral_moe_inference(mesh_device=mesh_device, model_name="mixtral-8x7B", mode="prefill")
    test_mixtral_moe_inference(mesh_device=mesh_device, model_name="mixtral-8x7B", mode="decode")
