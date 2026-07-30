#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode Attention for the llama3 tt_transformers port (single-chip, paged).

Basis: shim-only workloads/ttnn/tt_transformers/attention.py, audited against tt-metal
models/tt_transformers/tt/attention.py and the BH prefill capture
(project_llama3_prefill_op_sequence). Calls the PR1 (#468) single device ops directly.

Audit fixes vs the shim-only base:
  - Single-chip (num_devices=1): all TG / mesh-shard (ShardTensor2dMesh) / CCL machinery in
    the shim-only base is dead and is dropped. The fused QKV weight is one dummy
    (1,1,dim,qkv_size) and wo is (1,1,dim,dim) — no per-device cat loop. tt_all_reduce is the
    identity from ccl.py (design doc §8c).
  - Rope is a SINGLE ttnn.experimental.rotary_embedding_llama op per Q/K (capture ops 27-28),
    replacing the shim-only utils.rotary_embedding_llama matmul-decomposition (which did NOT
    match HW).
  - PAGED prefill: fills the paged KV cache via ttnn.experimental.paged_fill_cache for K and V
    (capture ops 33-34) before the prefill SDPA, matching the capture and the paged_attention=1
    pin. The shim-only forward_prefill was non-paged (it ignored page_table / kv_cache).
  - Weights use args.WEIGHTS_DTYPE (bfloat8_b) per ModelArgs / tt-metal llama3; the shim-only
    base used bfloat16. (dtype parity with the capture is re-verified during LUT correlation.)
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]

from workloads.ttnn.tt_transformers_dualmode.ccl import tt_all_reduce
from workloads.ttnn.tt_transformers_dualmode.model_config import dram_sharded_weight_memcfg


def _dummy_weight(shape, device, dtype):
    """Config-only dummy weight (no HF checkpoint), fabricated directly on-device in both
    modes. Values are irrelevant to perf and accuracy is not validated, so we avoid
    materializing a large host torch tensor on the HW path.

    Weights are DRAM width-sharded (see dram_sharded_weight_memcfg) so the matmul's
    input_1 memory tag matches the silicon capture (DRAM_WIDTH_SHARDED, not INTERLEAVED)."""
    w = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = dram_sharded_weight_memcfg(device, int(shape[-2]), int(shape[-1]))
    if IS_POLARIS:
        w._memory_config = cfg
    else:
        w = ttnn.to_memory_config(w, cfg)
    return w


def _is_bf16(tensor):
    """True if tensor is bf16. tensor.dtype is a DataType on the shim and a ttnn dtype on
    HW; ttnn.bfloat16 resolves correctly in both, so a direct compare works for both modes
    (the previous DataType.from_numpy() path mis-classified shim DataTypes as FLOAT32)."""
    return tensor.dtype == ttnn.bfloat16


class Attention:
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.n_kv_heads = configuration.n_kv_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.paged_attention_config = paged_attention_config
        self.transformation_mats = transformation_mats
        self.dtype = dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        # single-chip: no head/device splitting
        self.n_local_heads = self.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.kv_cache_dtype = ttnn.bfloat16
        self.activation_dtype = ttnn.bfloat16
        # qkv / o / sdpa run at HiFi2 in decode under the performance preset (tt-metal
        # DecodersPrecision default: LI_QKV_DECODE / LI_O_DECODE / SDPA_DECODE = HIFI2).
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        wdtype = configuration.WEIGHTS_DTYPE

        # Fused QKV weight (1,1,dim,qkv_size) and output proj (1,1,dim,dim) — dummy.
        self.wqkv = _dummy_weight((1, 1, self.hidden_size, configuration.qkv_size), mesh_device, wdtype)
        self.wo = _dummy_weight((1, 1, self.hidden_size, self.hidden_size), mesh_device, wdtype)

        # q/k norm are identity in llama3 (no QK-norm)
        self.q_norm = lambda x, mode: x
        self.k_norm = lambda x, mode: x

        if not use_paged_kv_cache:
            self.init_kv_cache(configuration, weight_cache_path, device=mesh_device)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar ** -0.5
        else:
            self.scale = self.head_dim ** -0.5

    def init_kv_cache(self, configuration, weight_cache_path, device=None):
        """Empty (dummy) KV cache on device. Paged shape per paged_attention_config."""
        if self.paged_attention_config:
            shape = [
                self.paged_attention_config.max_num_blocks,
                self.n_local_kv_heads,
                self.paged_attention_config.block_size,
                self.head_dim,
            ]
        else:
            shape = [self.max_batch_size, self.n_local_kv_heads, self.max_seq_len, self.head_dim]
        self.layer_past = [
            ttnn.zeros(shape, dtype=self.kv_cache_dtype, device=device, layout=ttnn.TILE_LAYOUT)
            for _ in range(2)
        ]

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, 'Seqlen must be divisible by 128'

        # reshape long sequences so the QKV matmul fits on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f'seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}')
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        # Fused QKV projection (capture op 25)
        xqkv_fused = ttnn.linear(
            x_11SH, self.wqkv, memory_config=None, compute_kernel_config=None, program_config=None,
        )
        xqkv_fused = tt_all_reduce(xqkv_fused)  # identity (single-chip)
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])
        ttnn.deallocate(x_11SH)

        # Split into heads (capture op 26)
        q_heads_pre, k_heads_pre, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False, memory_config=None,
        )
        q_heads_pre = self.q_norm(q_heads_pre, mode='prefill')
        k_heads_pre = self.k_norm(k_heads_pre, mode='prefill')
        ttnn.deallocate(xqkv_fused)

        # Rotary embeddings: ONE op each for Q, K (capture ops 27-28)
        if not _is_bf16(q_heads_pre):
            q_heads_pre = ttnn.typecast(q_heads_pre, dtype=ttnn.bfloat16)
        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre, rot_mats[0], rot_mats[1], self.transformation_mats['prefill'], is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_pre)
        if not _is_bf16(k_heads_pre):
            k_heads_pre = ttnn.typecast(k_heads_pre, dtype=ttnn.bfloat16)
        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre, rot_mats[0], rot_mats[1], self.transformation_mats['prefill'], is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_pre)

        # Typecast K/V for cache + SDPA (capture ops 29-30)
        k_heads_8b = ttnn.typecast(k_heads, dtype=ttnn.bfloat8_b)
        v_heads_8b = ttnn.typecast(v_heads, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # PAGED KV-cache fill for K and V (capture ops 33-34)
        kc = kv_cache if kv_cache is not None else self.layer_past
        ttnn.experimental.paged_fill_cache(kc[0], k_heads_8b, page_table=page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kc[1], v_heads_8b, page_table=page_table, batch_idx=user_id)

        # Prefill SDPA on the in-flight heads (capture op 36)
        q_heads_8b = ttnn.typecast(q_heads, dtype=self.activation_dtype)
        ttnn.deallocate(q_heads)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads_8b, k_heads_8b, v_heads_8b, is_causal=True, scale=self.scale,
            compute_kernel_config=None, program_config=None,
        )
        ttnn.deallocate(q_heads_8b)
        ttnn.deallocate(k_heads_8b)
        ttnn.deallocate(v_heads_8b)

        attn_output = ttnn.reshape(attn_output, [1, self.n_local_heads, -1, self.head_dim])

        # Concat heads (capture op 37)
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=None)
        # Output projection (capture op 38)
        output = ttnn.linear(
            attn_output, self.wo, compute_kernel_config=None, dtype=self.activation_dtype,
            program_config=None, memory_config=None,
        )
        ttnn.deallocate(attn_output)
        return tt_all_reduce(output)  # identity (single-chip)

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        """Paged decode: create-heads-decode -> fused rope -> fused cache update -> paged SDPA -> concat."""
        xqkv_fused = ttnn.linear(x, self.wqkv, memory_config=None,
                                 compute_kernel_config=self.compute_kernel_config_hifi2, program_config=None)
        xqkv_fused = tt_all_reduce(xqkv_fused)
        # createqkv-decode requires interleaved bf16 input (capture STS, row 33)
        xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused)
        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused, num_heads=self.n_local_heads, num_kv_heads=self.n_local_kv_heads,
            head_dim=self.head_dim, memory_config=None,
        )
        ttnn.deallocate(xqkv_fused)
        # rope-decode input must match the transformation_mat shard spec (capture Reshard row 35)
        q_heads = ttnn.reshard(q_heads, ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=self.mesh_device.compute_with_storage_grid_size(),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ))
        q_heads, k_heads = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads, k_heads, rot_mats[0], rot_mats[1], self.transformation_mats['decode'],
        )
        kc = kv_cache if kv_cache is not None else self.layer_past
        ttnn.experimental.paged_fused_update_cache(
            kc[0], k_heads, kc[1], v_heads, update_idxs_tensor=current_pos, page_table=page_table,
        )
        attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads, kc[0], kc[1], page_table_tensor=page_table, cur_pos_tensor=current_pos, scale=self.scale,
        )
        # sdpa-decode outputs DRAM-interleaved; concat-heads needs L1 height-sharded (capture ITS, row 39)
        attn_output = ttnn.interleaved_to_sharded(attn_output, ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=self.mesh_device.compute_with_storage_grid_size(),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ))
        attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.n_local_heads)
        output = ttnn.linear(attn_output, self.wo, compute_kernel_config=self.compute_kernel_config_hifi2,
                             dtype=self.activation_dtype, program_config=None, memory_config=None)
        return tt_all_reduce(output)

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode='decode', page_table=None,
                chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        if mode == 'prefill':
            return self.forward_prefill(x, rot_mats, user_id, page_table=page_table,
                                        chunk_page_table=chunk_page_table, chunk_start_idx=chunk_start_idx,
                                        kv_cache=kv_cache)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def __call__(self, x, current_pos=None, rot_mats=None, user_id=0, mode='prefill', page_table=None,
                 chunk_page_table=None, chunk_start_idx=None, kv_cache=None):
        return self.forward(x, current_pos, rot_mats=rot_mats, user_id=user_id, mode=mode,
                            page_table=page_table, chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx, kv_cache=kv_cache)
