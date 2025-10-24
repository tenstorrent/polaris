#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.embedding import Embedding
from workloads.ttnn.llama3.model_config import ModelArgs

def test_embedding():
    dtype = ttnn.bfloat16
    max_seq_len = 128
    batch_size = 1
    mesh_device = ttnn.open_device(device_id=0)  # Assuming device_id 0 for simplicity
    model_args = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    model_args.n_layers = 1

    state_dict = None #model_args.load_state_dict()

    if model_args.is_vision():
        layer_name = "text_model.tok_embeddings.weight"
    else:
        layer_name = "tok_embeddings.weight"

    tt_emb = Embedding(
        mesh_device=mesh_device,
        args=model_args,
        weight_cache_path=model_args.weight_cache_path(dtype),
        state_dict=state_dict,
        dtype=dtype,
        dim=model_args.dim,
    )

    pt_input = ttnn._rand(shape=(32,2), device=mesh_device, dtype=ttnn.uint32)

    tt_input = ttnn.from_torch(
        pt_input.squeeze(1),
        device=mesh_device,
        mesh_mapper=None, #ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_output = tt_emb(tt_input)
    print(f'tt_output: {tt_output.shape}')

    if (tt_output.shape == [32, 2, model_args.dim]):
        print("embedding Passed!")
    else:
        print("embedding Failed!")

if __name__ == "__main__":
    test_embedding()
    print("\n All tests completed!")
