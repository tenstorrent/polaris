# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
This is the ImageAttention block for Gemma-3-4b-it
We have reused the TTLlamaImageAttention with some modification.
We have made the linears (Q,K,V) to be executed separately and added bias support for O_projection, along with few
configuration changes.
"""


import numpy as np

import ttsim.front.ttnn as ttnn

from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.common.gemma_utils import nearest_32


class TtGemmaImageAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        tt_ccl=None,  # Optional for API compatibility
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim
        self.n_heads = configuration.vision_attn_n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_kv_heads = self.n_heads
        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices
        self.dtype = dtype
        self.grid_size = configuration.max_grid_size
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa
        self.configuration = configuration
        self.model_config = configuration.get_model_config()

        # Cache file name helper
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}{name}"

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        # Assertions for device splitting
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        # Get weights and convert to numpy
        wq = self._to_numpy(self.state_dict[wq_str])
        wk = self._to_numpy(self.state_dict[wk_str])
        wv = self._to_numpy(self.state_dict[wv_str])
        wo = self._to_numpy(self.state_dict[wo_str])

        # Pad head dim to multiple of 32
        wq_padded = self._pad_head_dim(wq, heads_out=True)
        wk_padded = self._pad_head_dim(wk, heads_out=True)
        wv_padded = self._pad_head_dim(wv, heads_out=True)
        wo_padded = self._pad_head_dim(wo, heads_out=False)

        # Chunk weights for multi-device
        wq_chunked = np.array_split(wq_padded, configuration.num_devices, axis=0)
        wk_chunked = np.array_split(wk_padded, configuration.num_devices, axis=0)
        wv_chunked = np.array_split(wv_padded, configuration.num_devices, axis=0)

        self.qkv_program_config = lambda seq_len, MAX_MM_SEQ_LEN: None

        # Build concatenated QKV weight
        qkv_per_device = []
        for i in range(configuration.num_devices):
            wq_t = np.swapaxes(wq_chunked[i], -2, -1)
            wk_t = np.swapaxes(wk_chunked[i], -2, -1)
            wv_t = np.swapaxes(wv_chunked[i], -2, -1)
            qkv_concat = np.concatenate([wq_t, wk_t, wv_t], axis=-1)
            qkv_per_device.append(qkv_concat)

        wqkv_combined = np.concatenate(qkv_per_device, axis=-1)

        # Determine mesh mapper
        is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"

        self.wqkv = ttnn.as_tensor(
            wqkv_combined,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wqkv_sharded"),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if is_mesh_device else None, # type: ignore[attr-defined]
        )

        # Process biases if they exist
        bq_str = f"{state_dict_prefix}wq.bias"
        bk_str = f"{state_dict_prefix}wk.bias"
        bv_str = f"{state_dict_prefix}wv.bias"
        bo_str = f"{state_dict_prefix}wo.bias"

        if bq_str in self.state_dict:
            bq = self._to_numpy(self.state_dict[bq_str])
            bk = self._to_numpy(self.state_dict[bk_str])
            bv = self._to_numpy(self.state_dict[bv_str])

            bq_padded = self._pad_head_dim_bias(bq)
            bk_padded = self._pad_head_dim_bias(bk)
            bv_padded = self._pad_head_dim_bias(bv)

            bq_chunked = np.array_split(bq_padded, configuration.num_devices, axis=0)
            bk_chunked = np.array_split(bk_padded, configuration.num_devices, axis=0)
            bv_chunked = np.array_split(bv_padded, configuration.num_devices, axis=0)

            bqkv_per_device = []
            for i in range(configuration.num_devices):
                bqkv_concat = np.concatenate([bq_chunked[i], bk_chunked[i], bv_chunked[i]], axis=-1)
                bqkv_per_device.append(bqkv_concat)

            bqkv_combined = np.concatenate(bqkv_per_device, axis=-1)

            self.bqkv = ttnn.as_tensor(
                bqkv_combined,
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("bqkv_sharded"),
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1) if is_mesh_device else None, # type: ignore[attr-defined]
            )
        else:
            self.bqkv = None

        # Output projection weight
        wo_transposed = np.swapaxes(wo_padded, -2, -1)

        self.wo = ttnn.as_tensor(
            wo_transposed,
            device=mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("wo_sharded"),
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-2) if is_mesh_device else None, # type: ignore[attr-defined]
        )

        # Output projection bias
        if bo_str in self.state_dict:
            bo = self._to_numpy(self.state_dict[bo_str])

            self.bo = ttnn.as_tensor(
                bo,
                device=mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name("bo_replicated"),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None,
            )
        else:
            self.bo = None

        self.scale = self.head_dim ** -0.5

    def _to_numpy(self, tensor):
        """Convert tensor to numpy array for preprocessing."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def _pad_head_dim(self, weight, heads_out=True):
        """Pad head dim to multiple of 32."""
        dim = weight.shape[1]
        assert weight.shape[0] == dim

        padded_head_dim = nearest_32(self.head_dim)
        padding_size = padded_head_dim - self.head_dim

        if padding_size > 0:
            if heads_out:
                weight = np.swapaxes(weight, -1, -2)

            weight = weight.reshape(dim, self.n_heads, self.head_dim)
            padding = np.zeros((dim, self.n_heads, padding_size), dtype=weight.dtype)
            weight = np.concatenate([weight, padding], axis=-1)
            weight = weight.reshape(dim, self.n_heads * padded_head_dim)

            if heads_out:
                weight = np.swapaxes(weight, -1, -2)

        return weight

    def _pad_head_dim_bias(self, bias):
        """Pad 1D bias to match padded head dim."""
        dim = bias.shape[0]
        expected = self.n_heads * self.head_dim
        assert dim == expected, f"Expected bias shape ({expected}), got {dim}"

        padded_head_dim = nearest_32(self.head_dim)
        padding_size = padded_head_dim - self.head_dim

        if padding_size > 0:
            bias = bias.reshape(self.n_heads, self.head_dim)
            padding = np.zeros((self.n_heads, padding_size), dtype=bias.dtype)
            bias = np.concatenate([bias, padding], axis=-1)
            bias = bias.reshape(self.n_heads * padded_head_dim)

        return bias

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        batch_size = x_11SH.shape[0]

        # Reshape if needed
        if len(x_11SH.shape) == 3:
            x_11SH = ttnn.reshape(x_11SH, (batch_size, 1, seq_len, -1))

        MAX_MM_SEQ_LEN = seq_len

        if seq_len > MAX_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, (batch_size, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1))

        # QKV linear
        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.qkv_program_config(seq_len, MAX_MM_SEQ_LEN),
        )

        if self.bqkv is not None:
            xqkv_fused = ttnn.add(xqkv_fused, self.bqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(x_11SH)

        # Create QKV heads
        q_heads_1QSD, k_heads_1KSD, v_heads_1VSD = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(xqkv_fused)

        # SDPA
        # SDPA - handle missing SDPAProgramConfig in simulator
        # SDPA - handle missing scaled_dot_product_attention in simulator
        if hasattr(ttnn, 'transformer') and hasattr(ttnn.transformer, 'scaled_dot_product_attention'):
            # Use ttnn's native SDPA
            if hasattr(ttnn, 'SDPAProgramConfig'):
                sdpa_cfg = ttnn.SDPAProgramConfig( # type: ignore[attr-defined]
                    compute_with_storage_grid_size=(8, 8),
                    q_chunk_size=256,
                    k_chunk_size=256,
                    exp_approx_mode=False,
                )
            else:
                sdpa_cfg = None

            attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD,
                k_heads_1KSD,
                v_heads_1VSD,
                is_causal=False,
                scale=self.scale,
                attn_mask=mask,
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_kernel_config_sdpa,
            )
        else:
            # Fallback: manual attention computation for simulator
            # Q @ K^T
            k_transposed = ttnn.transpose(k_heads_1KSD, -2, -1)
            attn_weights = ttnn.matmul(q_heads_1QSD, k_transposed)

            # Scale
            attn_weights = ttnn.multiply(attn_weights, self.scale)

            # Apply mask if provided
            if mask is not None:
                attn_weights = ttnn.add(attn_weights, mask)

            # Softmax
            attn_weights = ttnn.softmax(attn_weights, dim=-1)

            # Attention @ V
            attn_output_1QSD = ttnn.matmul(attn_weights, v_heads_1VSD)

            # Cleanup
            ttnn.deallocate(k_transposed)
            ttnn.deallocate(attn_weights)

        ttnn.deallocate(q_heads_1QSD)
        ttnn.deallocate(k_heads_1KSD)
        ttnn.deallocate(v_heads_1VSD)

        # Concat heads
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output_1QSD)

        # Reshape for long sequences
        if seq_len > MAX_MM_SEQ_LEN:
            attn_output_11SH = ttnn.reshape(
                attn_output_11SH, (batch_size, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1)
            )

        # Output projection
        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.qkv_program_config(seq_len, MAX_MM_SEQ_LEN),
        )

        if seq_len > MAX_MM_SEQ_LEN:
            output_11SH = ttnn.reshape(output_11SH, (batch_size, 1, seq_len, -1))

        ttnn.deallocate(attn_output_11SH)

        # All reduce for multi-device
        if self.num_devices > 1 and self.tt_ccl is not None:
            output_all_reduce = _all_reduce_helper(self, output_11SH)
            ttnn.deallocate(output_11SH)
        else:
            output_all_reduce = output_11SH

        # Add output bias
        if self.bo is not None:
            output_after_bias = ttnn.add(output_all_reduce, self.bo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(output_all_reduce)
        else:
            output_after_bias = output_all_reduce

        return output_after_bias


def _all_reduce_helper(caller, input_tensor):
    """Helper for multi-device all-reduce operation."""
    if caller.tt_ccl is None:
        return input_tensor

    w2_out_gathered = ttnn.experimental.all_gather_async( # type: ignore[attr-defined]
        input_tensor,
        persistent_output_buffer=None,
        dim=1,
        multi_device_global_semaphore=caller.tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=4 if caller.configuration.is_galaxy else 1,
        topology=ttnn.Topology.Ring, # type: ignore[attr-defined]
        barrier_semaphore=caller.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    pre_bias_output = ttnn.experimental.fast_reduce_nc( # type: ignore[attr-defined]
        w2_out_gathered, dims=[1], output=None, compute_kernel_config=None
    )

    ttnn.deallocate(w2_out_gathered)
    return pre_bias_output
