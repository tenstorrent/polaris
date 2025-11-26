#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn

class ModelArgs:
    def __init__(self, mesh_device, model_name="llama3-8B", max_batch_size=1, max_seq_len=256, instruct=False, dummy_weights=False):
        self.mesh_device = mesh_device
        if model_name == "llama3-8B":
            self.model_name = "llama3-8B"
            self.dim = 4096 # llama3 8B
            self.n_heads = 32 # llama3 8B
            self.head_dim = 128
        elif model_name == "llama3-3B":
            self.model_name = "llama3-3B"
            self.dim = 3072 # llama3 3B
            self.n_heads = 24 # llama3 3B
            self.head_dim = 128
        elif model_name == "llama3-1B":
            self.model_name = "llama3-1B"
            self.dim = 2048 # llama3 1B
            self.n_heads = 32 # llama3 1B
            self.head_dim = 64
        elif model_name == "llama3-70B":
            self.model_name = "llama3-70B"
            self.dim = 8192
            self.n_heads = 64
            self.head_dim = 128

        self.n_kv_heads = 8 # llama3 8B and llama3 3B and llama3 1B and llama3 70B
        self.rms_norm_add_unit_offset = False
        self.num_devices = 1
        self.max_batch_size = max_batch_size
        self.num_reduce_scatter_links = 1
        self.arch_name = ttnn.get_arch_name()
        self.n_layers = 1
        self.compute_kernel_config_hifi2=ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi2_fp16=ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi4=ttnn.MathFidelity.HiFi4
        self.max_grid_size = ttnn.CoreGrid([ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(7,7))])
        self.MAX_QKV_MM_SEQ_LEN = 2048
        self.num_all_gather_links = 1
        self.instruct=instruct
        self.qkv_size = 5120
        self.ccl_dtype = ttnn.bfloat8_b
        self.tile_size = 32
        self.min_kv_prefill_shard_seqlen = 256.0
        self.max_seq_len = max_seq_len
        self.rope_scaling_factor = None
        self.orig_context_len = None
        self.rope_theta = 500000.0
        self.model_config = None
        self.is_multichip = False
        self.dummy_weights = True
        self.cluster_shape = [1,1]
        self.query_pre_attn_scalar = None
        self.is_galaxy = False
        self.norm_eps = 1e-5
        self.is_distributed_norm = False
        self.vocab_size = 128256
        self.padded_vocab_size = None
        self.checkpoint_type = "simulation"
        self.WEIGHTS_DTYPE = ttnn.bfloat8_b

    def weight_cache_path(self, dtype):
        return None
    
    def ccl_topology(self):
        return None
    
    def prepare_residual_tensor_decode(self, x, input_mem_cfg, force_replicated=False, on_host=False):
        batch = x.shape[0]
        seq_len = x.shape[1]
        assert x.shape[2] == self.dim

        x = ttnn.transpose(x, 0, 1).unsqueeze(1)
        if batch < 32: # pad to 32 for small batches
            x = ttnn._rand(shape=(1, seq_len, 32, self.dim), device=self.mesh_device, dtype=ttnn.bfloat16)
        return x
    
    def is_vision(self):
        return False # llama3 3B
    
    def is_simulation(self):
        return True
