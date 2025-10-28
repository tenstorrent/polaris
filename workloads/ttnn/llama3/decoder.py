#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
from numpy import shape
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.attention import Attention
from workloads.ttnn.llama3.mlp import MLP
from workloads.ttnn.llama3.rmsnorm import RMSNorm

class TransformerBlock():
    def __init__(
        self,
        args,
        mesh_device,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = None #args.get_model_config()

        self.layer_num = layer_num

        self.attention = Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
        )
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix="", #args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="attention_norm",
            is_distributed=self.args.is_distributed_norm,
            add_unit_offset=self.args.rms_norm_add_unit_offset,
            sharded_program_config=None, #self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
            sharded_output_config=None, #self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
            ccl_topology=None, #self.args.ccl_topology(),
        )
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix="", #args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="ffn_norm",
            is_distributed=self.args.is_distributed_norm,
            add_unit_offset=self.args.rms_norm_add_unit_offset,
            sharded_program_config=None, #self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
            sharded_output_config=None, #self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            ccl_topology=None, #self.args.ccl_topology(),
        )

    def __call__(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        skip_mem_cfg = None #self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        attn_in = self.attention_norm(x, mode)
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        h = ttnn.add(x, attn_out, memory_config=skip_mem_cfg) #, dtype=ttnn.bfloat16 if TG else None)
        ttnn.deallocate(attn_out)
        if mode == "prefill":
            pass #x.deallocate(True)

        ff_in = self.ff_norm(h, mode)
        if TG and mode == "decode":
            ff_in = ttnn.to_memory_config(ff_in, memory_config=None) # self.model_config["MLP_ACT_MEMCFG"])
        ff_out = self.feed_forward.forward(ff_in, mode)
        out = ttnn.add(
            h,
            ff_out,
            memory_config=skip_mem_cfg,
            # ttnn.bfloat16,
        )
        return out
