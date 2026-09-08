#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
This is the modified version of the RMSNorm for Gemma-3-4b-it model.
We have modified the RMSNorm implementation equivalent to RMSNorm in Gemma-3-4b-it.
We have handled the unit offset addition in the RMSNorm implementation directly into the TTNN Weights
"""
import os
import sys
from pathlib import Path

import numpy as np
import ttsim.front.ttnn as ttnn

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule

# -----------------------------------------------------------------------------
# Compatibility shims
# -----------------------------------------------------------------------------
class _TopologyShim:
    Ring = "ring"
    Linear = "linear"
    Mesh = "mesh"


if not hasattr(ttnn, "Topology"):
    ttnn.Topology = _TopologyShim  # type: ignore[attr-defined]

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


def _reshape_host_weight(weight, dim):
    """
    Convert host weight to shape [1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT].
    Supports torch tensors and numpy arrays.
    """
    if hasattr(weight, "unsqueeze"):  # torch.Tensor
        return weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
    arr = np.asarray(weight)
    return arr.reshape(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])


def _add_unit_offset(weight):
    if hasattr(weight, "__add__") and not isinstance(weight, np.ndarray):
        return weight + 1.0
    return np.asarray(weight) + 1.0


def _as_tensor_compat(
    host_weight,
    *,
    device,
    dtype,
    layout,
    memory_config,
    cache_file_name=None,
    mesh_mapper=None,
):
    """
    Best-effort wrapper around ttnn.as_tensor for Polaris/ttsim.
    """
    kwargs = {
        "device": device,
        "dtype": dtype,
        "layout": layout,
        "memory_config": memory_config,
    }
    if cache_file_name is not None:
        kwargs["cache_file_name"] = cache_file_name
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper
    try:
        return ttnn.as_tensor(host_weight, **kwargs)
    except TypeError:
        # Older / reduced ttsim builds may not support all kwargs.
        kwargs.pop("mesh_mapper", None)
        kwargs.pop("cache_file_name", None)
        return ttnn.as_tensor(host_weight, **kwargs)


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a MeshDevice and sharding within devices.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-06,
        add_unit_offset=True,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring if hasattr(ttnn, "Topology") else None,  # type: ignore[attr-defined]
    ):
        super().__init__()
        self.eps = eps
        self.dim = dim  # Store dim for later use
        self.is_distributed = is_distributed
        self.ccl_topology = ccl_topology  # type: ignore[attr-defined]
        self.weight_dtype = weight_dtype

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        host_weight = _reshape_host_weight(state_dict[weight_name], dim)
        if add_unit_offset:
            host_weight = _add_unit_offset(host_weight)

        cache_name = None if weight_cache_path is None else Path(weight_cache_path) / weight_name

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        replicate_mapper = None
        shard_mapper = None
        if is_mesh_device and hasattr(ttnn, "ReplicateTensorToMesh"):
            replicate_mapper = ttnn.ReplicateTensorToMesh(device)
        if is_mesh_device and hasattr(ttnn, "ShardTensor2dMesh"):
            shard_mapper = ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))

        self.weight = _as_tensor_compat(
            host_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name,
            mesh_mapper=replicate_mapper,
        )

        if self.is_distributed:
            self.weight_distributed = _as_tensor_compat(
                host_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=cache_name,
                mesh_mapper=shard_mapper,
            )

        self.sharded_output_config = sharded_output_config
        self.sharded_program_config = sharded_program_config
        self.output_mem_config = output_mem_config

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x, mode, in_sharded=False, out_sharded=False):
        program_config = self.sharded_program_config if in_sharded else None
        memory_config = self.sharded_output_config if out_sharded else None

        # Handle is_distributed as both bool flag and callable function
        if callable(self.is_distributed):
            distributed = bool(self.is_distributed(mode))
        else:
            distributed = bool(self.is_distributed)

        # Branch based on distributed flag
        if distributed:
            # Distributed path - use _distributed_rmsnorm
            if in_sharded:
                assert False, "Distributed RMSNorm does not support sharded inputs"
            weight = self.weight_distributed
            x = self._distributed_rmsnorm(
                x,
                epsilon=self.eps,
                weight=weight,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
        else:
            # Non-distributed path - use ttnn.rms_norm directly
            if out_sharded:
                assert False, "Non-sharded version of RMSNorm cannot output a sharded tensor"
            weight = self.weight
            # Calculate dim from weight shape
            weight_shape = tuple(weight.shape) if hasattr(weight, "shape") else ()
            if len(weight_shape) >= 2:
                dim = weight_shape[-2] * weight_shape[-1]
            else:
                dim = self.dim  # Fallback to stored dim

            x = ttnn.rms_norm(
                x,
                epsilon=self.eps,
                weight_tensor=weight,
                memory_config=memory_config,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dim=dim,
            )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        return x

    def _distributed_rmsnorm(
        self,
        inp,
        epsilon=1e-6,
        weight=None,
        program_config=None,
        memory_config=None,
        compute_kernel_config=None,
    ):
        """
        Custom distributed RMSNorm implementation.
        Only used when is_distributed is True.
        Assumes sharded input that needs to be converted to interleaved first.
        """
        inp = ttnn.sharded_to_interleaved(inp)

        xnorm = ttnn.pow(inp, 2)
        xnorm = ttnn.mean(xnorm, dim=-1, keepdim=True)
        xnorm = 1.0 / ttnn.sqrt(xnorm + epsilon)
        xnorm = ttnn.multiply(inp, xnorm)

        weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)
        weight = ttnn.reshape(weight, [1, 1, 1, -1])
        weight = ttnn.to_layout(weight, ttnn.TILE_LAYOUT, dtype=self.weight_dtype)

        output = ttnn.multiply(xnorm, weight)

        if memory_config is not None:
            output = ttnn.to_memory_config(output, memory_config)

        ttnn.deallocate(xnorm)
        ttnn.deallocate(inp)

        return output