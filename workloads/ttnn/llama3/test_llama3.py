#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.model import Transformer
from workloads.ttnn.llama3.model_config import ModelArgs

# def filter_ttnn_attrs(attrs_dict):
#     return {k: v for k, v in attrs_dict.items() if not (isinstance(v, ttnn.Tensor) or k == "layout" or k == "memory_config")}

def test_model_inference(model_name: str = "llama3-8B"):
    paged_attention = False
    page_params = [{"page_block_size": 32, "page_max_num_blocks": 1024}]
    batch_size = 1
    if model_name == "llama3-8B":
        max_seq_len = 4096 #256
        layers = 32
    elif model_name == "llama3-3B":
        max_seq_len = 2048 #256
        layers = 28
    elif model_name == "llama3-1B":
        max_seq_len = 2048 #256
        layers = 16
    elif model_name == "llama3-70B":
        max_seq_len = 8192
        layers = 80
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    mesh_device = ttnn.open_device(device_id=0)
    weights = "random"
    
    dtype = ttnn.bfloat8_b
    instruct = False  # True if weights == "instruct" else False
    dummy_weights = True if weights == "random" else False
    cache_pcc = False # layers == 1 and not dummy_weights

    model_args = ModelArgs(
        mesh_device,
        model_name=model_name,
        instruct=instruct,
        dummy_weights=dummy_weights,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    iterations = 1

    if layers is not None:
        model_args.n_layers = layers
    state_dict = None #model_args.load_state_dict()

    prompts = ["This is a test"] * model_args.max_batch_size
    model_name = ""
    encoded_prompts = [128000]
    generation_start_pos = 0
    generation_length = iterations
    page_table_tt = None
    paged_attention_config = None

    # Load TTNN model
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=None, #model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )
    print("Model and caches loaded.")

    seqlen = 1  # Generating one token per user at a time
    batch = model_args.max_batch_size

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = ttnn._rand(shape=(batch, len(encoded_prompts)), device=mesh_device, dtype=ttnn.int32)
    tt_decode_input = tt_model.embd(encoded_prompts_tensor).view(batch, seqlen, -1)
    
    # Initial positions
    generation_pos = [generation_start_pos for _ in range(batch)]
    current_pos = ttnn._rand(shape=(len(generation_pos),), device=mesh_device, dtype=ttnn.int32)
    current_pos = current_pos.unsqueeze(0)
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    for i in range(generation_length):
        print(f"[Model] Generating token {i}")
        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            None, #model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        rot_mats = tt_model.rope_setup.get_rot_mats(current_pos)
        # Run TT model
        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_output_torch = ttnn.permute(ttnn.to_torch(tt_out), (1, 2, 0, 3)).squeeze(2)#[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
        
        if (tt_output_torch.shape == [1, 32, 128256]): # 128256 is the vocab_size for llama3 8B and llama3 3B, 1B
            print(f'tt_output_torch is correctly shaped: {tt_output_torch.shape}')
        else:
            print(f'tt_output_torch is incorrectly shaped: {tt_output_torch.shape}')
        
        ttnn.deallocate(tt_out)
    # print("Generating Model Graph...")
    # g = mesh_device.get_graph()
    # g.graph2onnx('ttnn_llama32_model.onnx', do_model_check=False,
    #                 filter_op_attrs=filter_ttnn_attrs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "llama3-3B"
    test_model_inference(model_name=model_name)
