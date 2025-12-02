#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from loguru import logger
import ttsim.front.ttnn as ttnn
import math

def nearest_32(x):
    return math.ceil(x / 32) * 32

def get_rot_transformation_mat(dhead, device=None):
    # TODO: Review suggestion
    # The function parameter dhead is ignored and hardcoded to 32. 
    # Either use the parameter value or remove it from the function signature to avoid confusion.
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = ttnn._rand([1, 1, dhead, dhead], device=device, dtype=ttnn.float32)
    # rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    # rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix

def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = position_ids.unsqueeze(1).expand(-1, cos.shape[-1])
    cos = cos.gather(0, position_id_expanded)
    sin = sin.gather(0, position_id_expanded)
    cos = ttnn.stack([cos, cos], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    sin = ttnn.stack([sin, sin], dim=-1).flatten(-2).unsqueeze(0).unsqueeze(0)
    return cos, sin

def apply_scaling(freqs: ttnn.Tensor, scale_factor: float, orig_context_len: int, device=None):
    ## pass, dummy call
    # low_freq_factor = 1
    # high_freq_factor = 4

    # low_freq_wavelen = orig_context_len / low_freq_factor
    # high_freq_wavelen = orig_context_len / high_freq_factor
    # new_freqs = []
    # for freq in freqs:
    #     wavelen = 2 * math.pi / freq
    #     if wavelen < high_freq_wavelen:
    #         new_freqs.append(freq)
    #     elif wavelen > low_freq_wavelen:
    #         new_freqs.append(freq / scale_factor)
    #     else:
    #         assert low_freq_wavelen != high_freq_wavelen
    #         smooth = (orig_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    #         new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return freqs

import numpy as np
def precompute_freqs(dim: int, end: int, theta, scale_factor, orig_context_len, device=None):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = ttnn.Tensor(shape=freqs.shape, dtype=ttnn.float32, device=device)
    t = ttnn.arange(end, device=device)
    if scale_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, orig_context_len, device)
    else:
        freqs = ttnn.Tensor(shape=freqs.shape, dtype=ttnn.float32, device=device)
    freqs = ttnn.outer(t, freqs).float()
    return ttnn.cos(freqs), ttnn.sin(freqs)

def compute_gather_cos_sin(dhead, end, theta, scale_factor, orig_context_len, position_ids, device=None):
    cos, sin = precompute_freqs(dhead, end, theta, scale_factor, orig_context_len, device=device)
    return gather_cos_sin(position_ids, cos, sin)

def get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0):
    cos, sin = precompute_freqs(
        head_dim, seq_len * 2, theta=theta, scale_factor=scale_factor, orig_context_len=orig_context_len, device=mesh_device
    )
    cos_gathered, sin_gathered = gather_cos_sin(ttnn.arange(start_pos, start_pos + seq_len, device=mesh_device), cos, sin)
    assert cos_gathered.size() == (1, 1, seq_len, head_dim)
    assert sin_gathered.size() == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_torch(
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=None, # ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_gathereds = ttnn.from_torch(
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=None, # ttnn.ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats

class RotarySetup():
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor,  # use None to disable rope scaling
        orig_context_len,  # only used if scaling enabled
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = False# isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = max(self.batch_size // list(device.shape)[1], 1)
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        # Generate the cos/sin matrices needed for ttnn.embedding op
        cos_matrix, sin_matrix = compute_gather_cos_sin(
            dhead=head_dim,
            end=max_seq_len * 2,
            theta=rope_theta,
            scale_factor=scale_factor,
            orig_context_len=orig_context_len,
            position_ids=ttnn.arange(max_seq_len, device=device),
            device=device,
        )
        cos_matrix = ttnn._rand(cos_matrix.shape, device=device, dtype=datatype)
        sin_matrix = ttnn._rand(sin_matrix.shape, device=device, dtype=datatype)

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        num_cores_x, num_cores_y = 8, 4
        self.batch_grid = (
            ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}
                    )
            if ttnn.get_arch_name() == "blackhole"
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )
        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE, device=device).repeat(
            1,
            1,
            batch_size,
            1,
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        trans_mat = ttnn._rand(trans_mat.shape, device=device, dtype=datatype)
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=(
                ttnn.ShardTensor2dMesh(
                    device,
                    dims=(None, 2) if (self.num_devices == 32 and batch_size > 1) else (None, None),
                    mesh_shape=list(device.shape),
                )
                if self.is_mesh_device
                else None
            ),
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim, device=device)
        prefill_trans_mat_torch = ttnn._rand(prefill_trans_mat_torch.shape, device=device, dtype=datatype)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, ttnn.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"

        batch = position_idxs.shape[0]
        position_idxs = ttnn.reshape(position_idxs, [1, batch])  # [1, 1, 1, batch]
        assert position_idxs.shape == [1, batch], "position idxs must be a [1, batch] tensor"
        # assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = ttnn.pad(position_idxs, (0, pad_size), "constant", 0)
        position_idxs = ttnn._rand(position_idxs.shape, device=self.device, dtype=ttnn.uint32)

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
            )

        return rot_idxs, position_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        rot_idxs = position_idxs
        assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        # logger.info(f'cos_mat is {self.cos_matrix.shape}, sin_mat is {self.sin_matrix.shape}, rot_idxs is {rot_idxs.shape}')
        cos = ttnn.embedding(rot_idxs, self.cos_matrix.squeeze(0).squeeze(0), layout=embedding_layout)  # [1, batch, head_dim]
        sin = ttnn.embedding(rot_idxs, self.sin_matrix.squeeze(0).squeeze(0), layout=embedding_layout)  # [1, batch, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        cos = ttnn.transpose(cos, 1, 2)  # [1, batch, 1[32], head_dim]
        sin = ttnn.transpose(sin, 1, 2)  # [1, batch, 1[32], head_dim]

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        cos = ttnn.interleaved_to_sharded(cos, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]
        sin = ttnn.interleaved_to_sharded(sin, mem_config)  # [1, 1 (= batch / shard_num_cores), 1[32], self.head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
