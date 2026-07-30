#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode LMHead for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/lm_head.py (simulation path).

Audit vs the shim-only base:
  - Simulation path only (num_splits=1, num_experts=1): the galaxy / column-split / dram-
    sharded-matmul machinery is dropped (dead for our single-chip, lean ModelArgs). The
    output weight is one dummy (dim, vocab_size).
  - The capture splits the lm_head matmul into column chunks wrapped by InterleavedToSharded
    / ShardedToInterleaved (ops 48-59) for L1 sizing. This port emits the single-matmul sim
    path + one sharded_to_interleaved; the chunk-split is a Phase-5 correlation parity item
    (see project_llama3_prefill_op_sequence).
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
    modes. For llama3-8B the output weight (dim x vocab) is large, so the HW path avoids
    materializing it as a host torch tensor; values are irrelevant (accuracy not validated).

    Weights are DRAM width-sharded (see dram_sharded_weight_memcfg) so the matmul's
    input_1 memory tag matches the silicon capture (DRAM_WIDTH_SHARDED, not INTERLEAVED)."""
    w = ttnn.zeros(list(shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = dram_sharded_weight_memcfg(device, int(shape[-2]), int(shape[-1]))
    if IS_POLARIS:
        w._memory_config = cfg
    else:
        w = ttnn.to_memory_config(w, cfg)
    return w


class LMHead:
    def __init__(self, args, mesh_device, dtype, state_dict, state_dict_prefix, weight_cache_path):
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        # Column-split the vocab into chunks (the max_columns_per_device split).
        # The HW column-tiles the vocab (L1 sizing): num_splits and chunk width are
        # arch-dependent (tt-metal get_lm_head_max_columns_per_device: BH single-chip
        # 128256//8=16032 -> 8 chunks; WH 668*num_cores=42752 -> 3 chunks), each a
        # DRAM-sharded matmul + ShardedToInterleaved, then Concat back to full vocab.
        #
        # DUAL-MODE split of responsibility:
        #  - POLARIS path: emit ONE canonical GEMM (dim x vocab). The arch-specific
        #    column split is realized in the backend (wl2archmap op_split_spec, issue
        #    #477) where the arch is known, keeping this front-end graph device-independent.
        #  - HW path: must do the real column split here — a single dim x vocab GEMM would
        #    L1-OOM on silicon. (num_splits=3 is WH-correct; BH-HW needs the formula-based
        #    split — tracked follow-up for the HW-validation runs, not exercised here.)
        if IS_POLARIS:
            self.output_weight = _dummy_weight((args.dim, args.vocab_size), mesh_device, args.WEIGHTS_DTYPE)
        else:
            self.num_splits = 3 if args.vocab_size % 3 == 0 else 1
            split_size = args.vocab_size // self.num_splits
            self.output_weights = [
                _dummy_weight((args.dim, split_size), mesh_device, args.WEIGHTS_DTYPE)
                for _ in range(self.num_splits)
            ]

    def forward(self, x):
        # lm_head matmul explicitly downcasts its output to bfp8 (tt-metal lm_head.py:
        # dtype=lm_head_dtype, default bfloat8_b) so the downstream ShardedToInterleaved
        # sees a bf8 activation in the capture — pass it rather than inheriting bf16.
        lm_head_dtype = getattr(self.args, 'lm_head_dtype', ttnn.bfloat8_b)
        if IS_POLARIS:
            # Single logical GEMM (dim x vocab); the backend column-split transform
            # expands it to N matmuls + N ShardedToInterleaved + Concat per arch.
            output = ttnn.linear(x, self.output_weight,
                                 compute_kernel_config=self.args.compute_kernel_config_hifi2,
                                 program_config=None, memory_config=None, dtype=lm_head_dtype)
            return tt_all_reduce(output)  # identity (single-chip)
        outputs = []
        for w in self.output_weights:
            o = ttnn.linear(x, w, compute_kernel_config=self.args.compute_kernel_config_hifi2,
                            program_config=None, memory_config=None, dtype=lm_head_dtype)
            o = ttnn.sharded_to_interleaved(o, memory_config=None)  # capture STS rows 54/56/58
            outputs.append(o)
        # combine the column chunks back to full vocab (capture Concat row 59)
        output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-1)
        return tt_all_reduce(output)  # identity (single-chip)

    def __call__(self, x):
        return self.forward(x)
