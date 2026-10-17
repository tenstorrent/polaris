# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.embedding import Embedding

def test_embedding():
    logger.info("Initializing TT-Polaris Embedding Test (Standardized for Simulator)...")
    batch_size = 32
    mesh_device = ttnn.open_device(device_id=0)

    # Setup Model Args
    # This will automatically load dim=4096 and vocab_size=32000 from model_config.py
    model_args = ModelArgs(mesh_device, model_name="Mistral-7B")

    # DYNAMICALLY get the dimension from the config
    expected_dim = model_args.dim
    logger.info(f"Initializing Embedding module (Dim: {expected_dim})...")

    # Initialize the module using the real class from the fork
    # The Embedding class will handle generating its own random weights if state_dict is empty
    tt_emb = Embedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=None,
        state_dict={},
        dtype=ttnn.bfloat16
    )

    logger.info("Embedding module initialized successfully.")

    # Input Tensor (Random Token IDs)
    tt_input = ttnn._rand(shape=[batch_size], device=mesh_device, dtype=ttnn.uint32)

    logger.info("Running Forward Pass...")
    tt_output = tt_emb(tt_input)

    actual_shape = list(tt_output.shape)
    logger.info(f"Output Shape: {actual_shape}")

    # Validation
    if actual_shape[-1] == expected_dim and actual_shape[-2] == batch_size:
        logger.success(f"Embedding Test Passed! Output shape {actual_shape} matches expected dimensions.")
    else:
        logger.error(f"Test Failed! Expected [..., {batch_size}, {expected_dim}], got {actual_shape}")

    # Cleanup
    ttnn.deallocate(tt_output)
    logger.info("Done.")

if __name__ == "__main__":
    test_embedding()
