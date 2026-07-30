#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode ModelArgs for the llama3 tt_transformers port.

Based on the shim-only workloads/ttnn/tt_transformers/model_config.py, made
dual-mode (IS_POLARIS gate) and audited against tt-metal
models/tt_transformers/tt/model_config.py. Config-only + dummy weights — the
Polaris path never reads an HF checkpoint (see design doc §5).

Audit fixes vs the shim-only base:
  - qkv_size now COMPUTED as head_dim*(2*n_kv_heads + n_heads) per tt-metal
    model_config.py:660 -> 6144 for llama3-8B (shim-only hardcoded a stale 5120).
  - hidden_dim (MLP intermediate, tt-metal `calculate_hidden_dim`) added: 14336
    for llama3-8B (absent in the shim-only base).
Scoped to the llama3 family (the workload); non-llama3 variants live in the
untouched shim-only model_config for mixtral et al.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]


def dram_sharded_weight_memcfg(mesh_device, k, n):
    """DRAM width-sharded memory config for a (k, n) weight tensor.

    Mirrors tt-metal ModelArgs.create_dram_sharded_mem_config (models/tt_transformers/
    tt/model_config.py): matmul weights are DRAM width-sharded on silicon, so the
    matmul's input_1 memory tag in the capture is ``DEV_*_DRAM_WIDTH_SHARDED``. Without
    this the shim emits ``DRAM_INTERLEAVED`` and every matmul misses the LUT on the
    weight-memory field.

    Polaris only needs the canonical (WIDTH_SHARDED, DRAM) tag for LUT-key parity — the
    ShardSpec details are not part of the key. The HW branch builds the full ShardSpec
    across the DRAM grid to match tt-metal. NOTE: the HW branch is not exercised in the
    Polaris CI here (no silicon at authoring time); it reproduces tt-metal verbatim.
    """
    if IS_POLARIS:
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM)
    import math
    dram_cores = mesh_device.dram_grid_size().x
    padded_size = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))}
    )
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


# llama3 family config: (dim, n_heads, n_kv_heads, head_dim, hidden_dim/MLP-intermediate)
_LLAMA3_CONFIGS = {
    'llama3-8b': {'dim': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 14336},
    'llama3-3b': {'dim': 3072, 'n_heads': 24, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 8192},
    'llama3-1b': {'dim': 2048, 'n_heads': 32, 'n_kv_heads': 8, 'head_dim': 64, 'hidden_dim': 8192},
    'llama3-70b': {'dim': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'head_dim': 128, 'hidden_dim': 28672},
}


class ModelArgs:
    def __init__(self, mesh_device, model_name='llama3-8B', max_batch_size=1, max_seq_len=256,
                 instruct=False, dummy_weights=True):
        self.mesh_device = mesh_device

        name_lower = model_name.lower()
        cfg = _LLAMA3_CONFIGS.get(name_lower, _LLAMA3_CONFIGS['llama3-8b'])
        self.model_name = 'llama3-8B' if name_lower not in _LLAMA3_CONFIGS else model_name

        # 1. Architecture (audited vs tt-metal llama3 config)
        self.dim = cfg['dim']
        self.n_heads = cfg['n_heads']
        self.n_kv_heads = cfg['n_kv_heads']
        self.head_dim = cfg['head_dim']
        self.hidden_dim = cfg['hidden_dim']        # MLP intermediate (tt-metal calculate_hidden_dim)
        self.hidden_size = self.dim                # alias used by some callers
        self.norm_eps = 1e-5
        self.vocab_size = 128256
        self.n_layers = 1                          # capture/runner overrides (1-layer device-perf)
        self.num_devices = 1
        self.num_experts = 1
        self.moe = False

        # qkv_size = head_dim * (2*n_kv_heads + n_heads)  (tt-metal model_config.py:660)
        # llama3-8B -> 128 * (16 + 32) = 6144  (matches the HW fused-QKV proj 4096->6144)
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)

        # 2. Simulator/general settings
        self.rms_norm_add_unit_offset = False
        self.max_batch_size = max_batch_size
        self.num_reduce_scatter_links = 1
        self.arch_name = ttnn.get_arch_name()
        self.compute_kernel_config_hifi2 = ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi2_fp16 = ttnn.MathFidelity.HiFi2
        self.compute_kernel_config_hifi4 = ttnn.MathFidelity.HiFi4
        # LoFi is used by the FF1/FF3 (gate/up) matmuls under the performance preset
        # (tt-metal DecodersPrecision.performance: OpGroup.LI_FF1_FF3 = LOFI).
        self.compute_kernel_config_lofi = ttnn.MathFidelity.LoFi
        self.max_grid_size = ttnn.CoreGrid([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        self.MAX_QKV_MM_SEQ_LEN = 2048
        self.num_all_gather_links = 1
        self.instruct = instruct
        self.ccl_dtype = ttnn.bfloat8_b
        self.tile_size = 32
        # min_kv_prefill_shard_seqlen = (TILE_SIZE*8*8) / n_kv_heads  (tt-metal model_config.py:661)
        self.min_kv_prefill_shard_seqlen = (32 * 8 * 8) / self.n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = 500000.0
        # Llama-3.1 llama3-type rope scaling (HF meta-llama/Llama-3.1-8B-Instruct
        # config.json: rope_scaling = {rope_type: llama3, factor: 8, low_freq_factor: 1,
        # high_freq_factor: 4, original_max_position_embeddings: 8192}). RotarySetup
        # (rope.py) consumes rope_scaling_factor + orig_context_len; both None disables
        # scaling. Wiring the real values makes the cos/sin tables match the reference
        # model — shape/LUT-key-neutral (values only), but corrects rope fidelity.
        self.rope_scaling_factor = 8.0
        self.orig_context_len = 8192
        self.rope_scaling = {
            'rope_type': 'llama3', 'factor': 8.0,
            'low_freq_factor': 1.0, 'high_freq_factor': 4.0,
            'original_max_position_embeddings': 8192,
        }
        self.model_config = None
        self.is_multichip = False
        self.dummy_weights = dummy_weights
        self.cluster_shape = [1, 1]
        self.use_qk_fused = False
        self.use_fused_qkv_op = True
        self.query_pre_attn_scalar = None
        self.is_galaxy = False
        self.is_distributed_norm = False
        self.padded_vocab_size = None
        self.checkpoint_type = 'simulation'
        self.WEIGHTS_DTYPE = ttnn.bfloat8_b
        # MLP gate/up projections (w1/w3, FF1/FF3) are BFP4 on silicon; down_proj (w2)
        # and attention stay BFP8. Mirrors tt-metal decoders_optimizations ff1_3_dtype
        # ("All models use bfp4 in FF1 and FF3 MLPs", model_config.py) and the capture
        # (the 14336-wide w1/w3 matmuls have input_1 dtype BFLOAT4_B).
        self.FF1_FF3_WEIGHTS_DTYPE = ttnn.bfloat4_b
        # lm_head matmul output is downcast to bfp8 (tt-metal ModelArgs.lm_head_dtype).
        self.lm_head_dtype = ttnn.bfloat8_b

    def weight_cache_path(self, dtype):
        return None

    def ccl_topology(self):
        return None

    def prepare_residual_tensor_decode(self, x, input_mem_cfg, args=None, force_replicated=False, on_host=False):
        batch = x.shape[0]
        seq_len = x.shape[1]
        assert x.shape[2] == self.dim
        x = ttnn.transpose(x, 0, 1).unsqueeze(0)
        return x

    def load_state_dict(self):
        return None

    def is_vision(self):
        return False

    def is_simulation(self):
        return True
