#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
 
import os, sys
from ttsim.front.ttnn.tensor import DataType
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.llama3.attention import tt_all_reduce

def pad_to_size(tensor, dim, size):
    """Pads the tensor to the specified size along the given dimension."""
    return tensor

def OpGroup(args, kwargs):
    pass

def TensorGroup(args, kwargs):
    pass

class MLP():
    def __init__(
        self, mesh_device, args, state_dict, weight_cache_path, layer_num, dtype, model_config, state_dict_prefix=None
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.args = args
        self.dim = args.dim
        self.model_config = model_config
        self.layer_num = layer_num
        self.w1 = ttnn.Tensor(shape=(self.dim, int(3.5 * self.dim)), device=mesh_device, dtype=ttnn.bfloat16)
        self.w2 = ttnn.Tensor(shape=(int(3.5 * self.dim), self.dim), device=mesh_device, dtype=ttnn.bfloat16)
        self.w3 = ttnn.Tensor(shape=(self.dim, int(3.5 * self.dim)), device=mesh_device, dtype=ttnn.bfloat16)

    def forward(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        TG = self.args.is_galaxy
        w1_out = ttnn.linear(
            x,
            self.w1,
            #dtype=ttnn.bfloat16 if TG else ttnn.bfloat16, # activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
            compute_kernel_config=None, #li_ff1_3_compute_kernel_cfg,
            program_config=None, #pc_1,
            memory_config=None, #memory_config,
        )

        w3_out = ttnn.linear(
            x,
            self.w3,
            #dtype=ttnn.bfloat16 if TG else ttnn.bfloat16, #activation_dtype or ttnn.bfloat16,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_3 else None,
            compute_kernel_config=None, #li_ff1_3_compute_kernel_cfg,
            program_config=None, #pc_3,
            memory_config=None, #memory_config,
        )
        ttnn.deallocate(x)

        ## commenting out TG specific code for simulation
        # if TG:
        #     if self.dim == 8192 or mode == "prefill":
        #         input_mem_cfg = w1_out.memory_config()
        #         w1_out = ttnn.reduce_scatter(
        #             w1_out,
        #             dim=3,
        #             math_op=ttnn.ReduceType.Sum,
        #             num_links=self.args.num_reduce_scatter_links,
        #             cluster_axis=1,
        #             mesh_device=self.mesh_device,
        #             topology=ttnn.Topology.Linear,
        #             memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
        #         )
        #         w3_out = ttnn.reduce_scatter(
        #             w3_out,
        #             dim=3,
        #             math_op=ttnn.ReduceType.Sum,
        #             num_links=1,
        #             cluster_axis=1,
        #             mesh_device=self.mesh_device,
        #             topology=ttnn.Topology.Linear,
        #             memory_config=self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] if mode == "decode" else None,
        #         )
        #     else:
        #         w1_out = tt_all_reduce(
        #             w1_out,
        #             self.mesh_device,
        #             cluster_axis=1,
        #             num_all_gather_links=2,
        #             sharded=True if mode == "decode" else False,
        #             topology=self.args.ccl_topology(),
        #             memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
        #         )
        #         w3_out = tt_all_reduce(
        #             w3_out,
        #             self.mesh_device,
        #             cluster_axis=1,
        #             num_all_gather_links=2,
        #             sharded=True if mode == "decode" else False,
        #             topology=self.args.ccl_topology(),
        #             memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
        #         )

        w2_in = ttnn.multiply(
            w1_out,
            w3_out,
            input_tensor_a_activations=None, #[self.activation_type],
            #dtype=ttnn.bfloat16, #activation_dtype or ttnn.bfloat8_b,
            memory_config=None, #w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        ## commenting out TG specific code for simulation
        # if TG and (self.dim == 8192 or mode == "prefill"):
        #     w2_in = ttnn.all_gather(
        #         w2_in,
        #         3,
        #         num_links=2,
        #         cluster_axis=1,
        #         mesh_device=self.mesh_device,
        #         topology=ttnn.Topology.Linear,
        #         memory_config=input_mem_cfg,
        #     )
        #     if mode == "decode":
        #         w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=None, #li_ff2_compute_kernel_cfg,
            dtype=self.args.ccl_dtype if TG else ttnn.bfloat16, # activation_dtype or ttnn.bfloat16,
            program_config=None, #pc_2,
            memory_config=None, #memory_config,
            core_grid=None,  # FIXME: validate on TG ttnn.CoreGrid(y=8, x=8) if not pc_2 else None,
        )
        ttnn.deallocate(w2_in)

        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            cluster_axis=0,
            dim=0 if (TG and self.dim < 8192) else 3,
            num_reduce_scatter_links=self.args.num_reduce_scatter_links,
            num_all_gather_links=self.args.num_all_gather_links,
            sharded=(mode == "decode"),
            dtype=self.args.ccl_dtype,
            use_composite=True if self.dim == 8192 else False,
            topology=self.args.ccl_topology(),
        )

        return w2_out_reduced

def __call__(self, x: ttnn.Tensor, mode) -> ttnn.Tensor:
        """
        Call the forward method of the MLP class.
        """
        return self.forward(x, mode)
