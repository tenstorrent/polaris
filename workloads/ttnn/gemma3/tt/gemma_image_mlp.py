"""
This is the FeedForward submodule for vision block in Gemma-3-4b-it
We have reused the TtLlamaImageFeedForward with few changes in CoreGrid and program_config configurations
"""
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule


class TtGemmaImageFeedForward(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        tt_ccl=None,  # Optional for API compatibility
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.model_config = args.get_model_config()

        # Helper to transpose weight (on host, before converting to ttnn)
        def get_transposed_weight(name, suffix):
            weight = self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]
            # Convert to numpy if needed
            if hasattr(weight, "numpy"):
                weight = weight.numpy()
            elif hasattr(weight, "detach"):
                weight = weight.detach().cpu().numpy()
            weight = np.asarray(weight)
            # Transpose last two dimensions
            return np.swapaxes(weight, -2, -1)

        # Helper to get bias (no transpose needed)
        def get_bias(name, suffix):
            bias = self.state_dict[f"{state_dict_prefix}{name}.{suffix}"]
            if hasattr(bias, "numpy"):
                bias = bias.numpy()
            elif hasattr(bias, "detach"):
                bias = bias.detach().cpu().numpy()
            return np.asarray(bias)

        # Cache file name helper
        def get_cache_name(name, suffix):
            if args.dummy_weights or weight_cache_path is None:
                return None
            return weight_cache_path / f"{state_dict_prefix}{name}.{suffix}"

        # Helper to convert tensor to interleaved format on device
        def as_interleaved_tensor(name, suffix, tensor_dtype, dim=None):
            if suffix == "weight":
                host_tensor = get_transposed_weight(name, suffix)
            else:
                host_tensor = get_bias(name, suffix)
            # Determine mesh mapper
            is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
            if is_mesh_device and dim is not None:
                mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=dim)  # type: ignore[attr-defined]
            elif is_mesh_device:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
            else:
                mesh_mapper = None
            return ttnn.as_tensor(
                host_tensor,
                dtype=tensor_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_cache_name(name, suffix),
                mesh_mapper=mesh_mapper,
            )

        # Sharded weights
        self.c_fc_weight = as_interleaved_tensor("c_fc", "weight", dtype, dim=-1)
        self.c_fc_bias = as_interleaved_tensor("c_fc", "bias", ttnn.bfloat16, dim=-1)
        self.c_fc_bias = ttnn.reshape(self.c_fc_bias, (1, -1))
        self.c_proj_weight = as_interleaved_tensor("c_proj", "weight", dtype, dim=-2)
        self.c_proj_bias = as_interleaved_tensor("c_proj", "bias", ttnn.bfloat16, dim=None)

    def _get_shape_tuple(self, tensor):
        """Get shape as tuple from various tensor types."""
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            if hasattr(shape, '__iter__'):
                return tuple(shape)
            return shape
        if hasattr(tensor, 'get_shape'):
            return tuple(tensor.get_shape())
        return ()

    def forward(self, x):
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        # Get input shape safely
        x_shape = self._get_shape_tuple(x)

        # Handle 4D tensor: [1, B, seq_len, hidden_dim] or [B, 1, seq_len, hidden_dim]
        if len(x_shape) == 4:
            if x_shape[0] == 1 and x_shape[1] > 1:
                # Shape is [1, B, seq_len, hidden_dim]
                batch_size = x_shape[1]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
            elif x_shape[1] == 1:
                # Shape is [B, 1, seq_len, hidden_dim]
                batch_size = x_shape[0]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
            else:
                # Shape is [B, ?, seq_len, hidden_dim] - use first dim as batch
                batch_size = x_shape[0]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
        elif len(x_shape) == 3:
            batch_size = x_shape[0]
            seq_len = x_shape[1]
            hidden_dim = x_shape[2]
        else:
            batch_size = 1
            seq_len = x_shape[-2] if len(x_shape) >= 2 else 1
            hidden_dim = x_shape[-1]

        # Calculate total elements to verify reshape is valid
        input_elements = 1
        for dim in x_shape:
            input_elements *= dim
        output_elements = batch_size * seq_len * hidden_dim

        # Reshape to 3D for linear: (batch, seq, hidden)
        x_in = ttnn.reshape(x, (batch_size, seq_len, hidden_dim))

        # Get program configs with fallback to None
        if "IMAGE_MLP_FC_PROGCFG" in self.model_config and self.model_config["IMAGE_MLP_FC_PROGCFG"] is not None:
            try:
                pc_1 = self.model_config["IMAGE_MLP_FC_PROGCFG"](seq_len, seq_len)
            except Exception:
                pc_1 = None
        else:
            pc_1 = None

        if "IMAGE_MLP_PROJ_PROGCFG" in self.model_config and self.model_config["IMAGE_MLP_PROJ_PROGCFG"] is not None:
            try:
                pc_2 = self.model_config["IMAGE_MLP_PROJ_PROGCFG"](seq_len, seq_len)
            except Exception:
                pc_2 = None
        else:
            pc_2 = None

        # First linear layer
        c_fc_out = ttnn.linear(
            x_in,
            self.c_fc_weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            program_config=pc_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.c_fc_bias is not None:
            c_fc_out = ttnn.add(c_fc_out, self.c_fc_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        c_fc_out = ttnn.gelu(c_fc_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Second linear layer
        c_proj_out = ttnn.linear(
            c_fc_out,
            self.c_proj_weight,
            compute_kernel_config=self.args.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            program_config=pc_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape to 4D: (batch, 1, seq, hidden)
        c_proj_out = ttnn.reshape(c_proj_out, (batch_size, 1, seq_len, -1))

        # All reduce for multi-device
        if self.args.num_devices > 1 and self.tt_ccl is not None:
            try:
                w2_out_gathered = ttnn.experimental.all_gather_async(  # type: ignore[attr-defined]
                    c_proj_out,
                    persistent_output_buffer=None,
                    dim=1,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=4 if self.args.is_galaxy else 1,
                    topology=ttnn.Topology.Ring,  # type: ignore[attr-defined]
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
                pre_bias_output = ttnn.experimental.fast_reduce_nc(  # type: ignore[attr-defined]
                    w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
                )
            except AttributeError:
                pre_bias_output = c_proj_out
        else:
            pre_bias_output = c_proj_out

        output = ttnn.add(pre_bias_output, self.c_proj_bias)
        return output