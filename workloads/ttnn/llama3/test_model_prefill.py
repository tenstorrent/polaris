# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import bz2
from loguru import logger
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.model import Transformer
from workloads.ttnn.llama3.model_config import ModelArgs

def test_model_inference(model_name: str = "llama3-8B"):
    paged_attention = False
    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    optimizations = None
    max_seq_len = 128 * 1024
    seq_len = 128 # or 256
    if model_name == "llama3-8B":
        num_layers = 32
    elif model_name == "llama3-3B":
        num_layers = 28
    elif model_name == "llama3-1B":
        num_layers = 16
    elif model_name == "llama3-70B":
        num_layers = 80

    mesh_device = ttnn.open_device(device_id=0)

    dtype = ttnn.bfloat8_b
    batch_size = 1  # For prefill we only support batch_size = 1

    # Use instruct weights instead of general weights
    instruct = True

    paged_attention_config = None
    
    # Load TTNN model
    logger.info(f"Loading TT model...")

    tt_model_args = ModelArgs(
        mesh_device,
        model_name=model_name,
        instruct=instruct,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    if num_layers is not None:
        tt_model_args.n_layers = num_layers

    from typing import Dict
    state_dict: Dict[str, int] = {}
 
    model = Transformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    encoded_prompt_tensor = ttnn._rand(shape=[batch_size, seq_len], device=mesh_device, dtype=ttnn.int32)
    print(f'encoded prompt tensor shape is {encoded_prompt_tensor.shape}')
    # tt_prefill_input = encoded_prompt_tensor.unsqueeze(0)
    tt_prefill_input = encoded_prompt_tensor
    tt_prefill_input = ttnn.reshape(tt_prefill_input, (1, 1, 1, -1))
    
    bsz, head_dim, _, last_dim = model.rope_setup.cos_matrix.shape
    S = tt_prefill_input.shape[-1]
    model.rope_setup.cos_matrix = ttnn._rand(shape=[bsz, head_dim, S, last_dim], device=mesh_device, dtype=ttnn.float32)
    model.rope_setup.sin_matrix = ttnn._rand(shape=[bsz, head_dim, S, last_dim], device=mesh_device, dtype=ttnn.float32)

    tt_rot_mats_prefill = [
        model.rope_setup.cos_matrix, # [:, :, start_pos : start_pos + S, :],
        model.rope_setup.sin_matrix, # [:, :, start_pos : start_pos + S, :],
    ]

    tokens_embd = model.embd(tt_prefill_input)
    tokens_embd = tokens_embd.squeeze(0)

    #print(f'tokens_embd shape is {tokens_embd.shape}')
    tt_output_torch = model.ttnn_prefill_forward(
        tokens_embd,
        rot_mats=tt_rot_mats_prefill,
        user_id=0,
        get_last_token=seq_len - 1,
    )

    if (tt_output_torch.shape == [1, 1, seq_len, 128256]):
        print(f"Output shape is correct [1, 1, {seq_len}, 128256]")
    else:
        print(f'Output shape is incorrect. It is {tt_output_torch.shape}. Should have been [1, 1, {seq_len}, 128256]')

    logger.info(f"Finished running TT model.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "llama3-3B"
    test_model_inference(model_name=model_name)
