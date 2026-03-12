#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.tt_transformers.model_config import ModelArgs
from workloads.ttnn.tt_transformers.model import Transformer
from loguru import logger

def test_model_inference(wlname: str, mesh_device, cfg: dict):
    model_name = cfg.get('model_name')
    max_seq_len = 128
    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # Use instruct weights instead of general weights
    instruct = True

    model_args = ModelArgs(
        mesh_device,
        model_name,
        instruct=instruct,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=None,
        weight_cache_path=model_args.weight_cache_path(dtype),
        paged_attention_config=None,
    )

    tt_prefill_input = ttnn._rand(shape=[1, 1, 128, model_args.dim//8], device=mesh_device, dtype=ttnn.bfloat16) # type: ignore[operator]
    logger.info(f"Running TT model...")
    generation_pos = [0 for _ in range(batch_size)]
    current_pos = ttnn._rand(shape=(len(generation_pos),), device=mesh_device, dtype=ttnn.int32).unsqueeze(0)
    rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
    tt_output_torch = tt_model.ttnn_prefill_forward(
        tt_prefill_input,
        rot_mats,
        0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None
    )
    logger.info(f"Input shape is {tt_prefill_input.shape}")
    logger.info(f"TT output shape: {tt_output_torch.shape}")
    assert tt_output_torch.shape == tt_prefill_input.shape, f"Expected output shape {tt_prefill_input.shape} but got {tt_output_torch.shape}"
    logger.info("TT model inference successful with expected output shape.")

if __name__ == "__main__":
    mesh_device = ttnn.open_device(device_id=0)
    test_model_inference(wlname='mixtral_model_prefill', mesh_device=mesh_device, cfg={'model_name': "mixtral-8x7b"})
