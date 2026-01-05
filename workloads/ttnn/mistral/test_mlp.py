# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from ttsim.ops.tensor import require_shape_list
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.mlp import MLP

def test_mlp_inference():
    logger.info("Initializing TT-Metal Parity MLP Test (Mistral-7B Config)...")

    # 1. Define batch_size (it was missing!)
    batch_size = 1
    seq_len = 128
    mesh_device = ttnn.open_device(device_id=0)

    # 2. Load args (This sets dim=4096 automatically)
    model_args = ModelArgs(mesh_device, model_name="Mistral-7B")

    logger.info("Initializing Real MLP module from fork...")
    tt_model = MLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict={},
        weight_cache_path=None,
        layer_num=0,
        dtype=ttnn.bfloat8_b,
        model_config=None
    )

    logger.info("Creating Input Tensor...")
    # 3. Use model_args.dim here (Dynamic!)
    input_raw = ttnn._rand(
        shape=[batch_size, 1, seq_len, model_args.dim],
        device=mesh_device,
        dtype=ttnn.bfloat16
    )
    tt_input = ttnn.to_layout(input_raw, ttnn.TILE_LAYOUT)

    logger.info("Running MLP Forward Pass...")

    tt_output = tt_model.forward(tt_input, mode="prefill")
    actual_shape = require_shape_list(tt_output.shape, "MLP output must have a known shape")
    logger.info(f"Output Shape: {actual_shape}")

    # 4. Compare against model_args.dim
    if actual_shape[-1] == model_args.dim:
        logger.success(f"MLP Test Passed! Output dim is {model_args.dim}.")
    else:
        logger.error(f"MLP Test Failed! Expected {model_args.dim}, got {actual_shape[-1]}")

    ttnn.deallocate(tt_output)
    logger.info("Done.")

if __name__ == "__main__":
    test_mlp_inference()