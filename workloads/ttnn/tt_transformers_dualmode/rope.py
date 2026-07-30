#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode RoPE for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/rope.py + tt-metal
models/tt_transformers/tt/{rope.py,common.py}.

Audit fixes vs the shim-only base:
  - The shim-only RotarySetup.__init__ computed cos/sin via ttnn.arange/outer/cos/sin and
    then DISCARDED the values (overwrote them with ttnn._rand). Those compute calls emit
    SimOps that do NOT exist on HW: tt-metal precomputes the matrices on host with torch and
    only transfers them via from_torch (no profiled device ops at init). Dual-mode fabricates
    the cos/sin/trans matrices as dummy weights DIRECTLY (no init-time device ops), matching
    HW and the dummy-weights perf-capture contract (design doc §8b).
  - The rope APPLICATION is a single ttnn.rotary_embedding_llama op emitted in attention.py
    (matches the capture's RotaryEmbeddingLlamaDeviceOperation, ops 27-28), NOT the shim-only
    utils.rotary_embedding_llama matmul-decomposition. rope.py only provides the matrices.
  - Matrix shapes mirror tt-metal common.py: prefill rot_mats = [cos, sin] each
    (1,1,seq_len,head_dim) (common.py:445-451); trans_mat uses a single tile (1,1,32,32)
    (common.py:473-477, dhead forced to 32).
  - rope values are dummy: irrelevant to perf (rope is one fixed-shape op) and accuracy is
    not validated here (perf-capture only). Real rope math is intentionally not replicated.
"""
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

TILE = 32


def nearest_32(x):
    return math.ceil(x / 32) * 32


def _dummy(shape, device, dtype):
    """Dummy matrix of the given shape (config-only, no HF checkpoint), fabricated directly
    in both modes. Values are irrelevant to perf and accuracy is not validated, so real rope
    math is not computed. ttnn.zeros also tolerates device=None (host tensor) — callers such
    as get_rot_transformation_mat() may omit a device, which the old from_torch path crashed on.
    """
    return ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _shape_list(t):
    """Mode-aware shape-as-list (ttsim require_shape_list is shim-only)."""
    if IS_POLARIS:
        from ttsim.ops.tensor import require_shape_list
        return require_shape_list(t.shape, 'shape must be known')
    return list(t.shape)


def get_rot_transformation_mat(dhead=32, device=None, datatype=None):
    # ROPE op uses a single tile (tt-metal common.py:473-477, dhead forced to 32).
    datatype = datatype if datatype is not None else ttnn.bfloat16
    return _dummy((1, 1, TILE, TILE), device, datatype)


def get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0):
    """Prefill rot_mats = [cos, sin], each (1,1,seq_len,head_dim) (tt-metal common.py:445-451)."""
    cos_gathereds = _dummy((1, 1, seq_len, head_dim), mesh_device, ttnn.bfloat16)
    sin_gathereds = _dummy((1, 1, seq_len, head_dim), mesh_device, ttnn.bfloat16)
    return [cos_gathereds, sin_gathereds]


class RotarySetup:
    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float,
        scale_factor,            # None to disable rope scaling
        orig_context_len=None,   # only used if scaling enabled
        datatype=None,
    ):
        datatype = datatype if datatype is not None else ttnn.bfloat16
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.device = device
        self.is_mesh_device = False
        self.num_devices = 1
        self.batch_size_per_device_group = batch_size
        self.core_grid = device.compute_with_storage_grid_size()

        # cos/sin caches: (1,1,max_seq_len,head_dim). Dummy (see module audit note) — the
        # shim-only init-time arange/outer/cos/sin compute is dropped (it over-emitted ops).
        self.cos_matrix = _dummy((1, 1, max_seq_len, head_dim), device, datatype)
        self.sin_matrix = _dummy((1, 1, max_seq_len, head_dim), device, datatype)

        num_cores_x, num_cores_y = 8, 4
        self.batch_grid = (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))})
            if ttnn.get_arch_name() == 'blackhole'
            else ttnn.num_cores_to_corerangeset(batch_size, self.core_grid, row_wise=True)
        )

        # Transformation matrices: single tile (1,1,32,32) decode + prefill (tt-metal forces dhead=32).
        self.transformation_mat = _dummy((1, 1, TILE, TILE), device, datatype)
        self.transformation_mat_prefill = _dummy((1, 1, TILE, TILE), device, datatype)

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, 'Transformation matrix not initialized'
        assert self.transformation_mat_prefill is not None, 'Prefill transformation matrix not initialized'
        return {'decode': self.transformation_mat, 'prefill': self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, ttnn.Tensor), 'Position ids must be a tensor'
        dims0 = _shape_list(position_idxs)
        assert len(dims0) == 1, 'position idxs must be a [batch] tensor'
        batch = dims0[0]
        position_idxs = ttnn.reshape(position_idxs, [1, batch])
        assert _shape_list(position_idxs) == [1, batch], 'position idxs must be a [1, batch] tensor'

        pad_size = nearest_32(batch) - batch
        position_idxs = ttnn.pad(position_idxs, (0, pad_size), 'constant', 0)
        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
        )
        return rot_idxs, position_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        """Decode rot_mats via embedding + transpose + interleaved_to_sharded (real device ops)."""
        device = self.device
        rot_idxs = position_idxs
        assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, 'rot_idxs must be a [1, batch] tensor'
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(rot_idxs, self.cos_matrix.squeeze(0).squeeze(0), layout=embedding_layout)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix.squeeze(0).squeeze(0), layout=embedding_layout)

        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(TILE, self.head_dim),
            core_grid=self.batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.interleaved_to_sharded(cos, mem_config)
        sin = ttnn.interleaved_to_sharded(sin, mem_config)

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]
