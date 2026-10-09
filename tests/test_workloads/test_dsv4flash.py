#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4-Flash TTSIM workload (workloads/deepseek_v4_flash/)."""
from typing import Any, Dict

import pytest

from workloads.deepseek_v4_flash.dsv4f_model import Dsv4FlashMoe

# Published total for deepseek-ai/DeepSeek-V4-Flash is 284B; the 1 MTP layer
# is excluded by design and the simulated base model lands at ~284.3B.
EXPECTED_TOTAL_PARAMS = 284_000_000_000
PARAM_COUNT_TOLERANCE = 0.05

TINY_CFG: Dict[str, Any] = {
    'vocab_size': 256,
    'hidden_size': 64,
    'num_hidden_layers': 6,
    'num_hash_layers': 2,
    'compress_ratios': [0, 4, 128, 4, 128, 0],
    'num_attention_heads': 4,
    'head_dim': 32,
    'q_lora_rank': 16,
    'o_lora_rank': 16,
    'o_groups': 2,
    'sliding_window': 4,
    'n_routed_experts': 8,
    'num_experts_per_tok': 2,
    'moe_intermediate_size': 32,
    'n_shared_experts': 1,
    'index_n_heads': 2,
    'index_head_dim': 8,
    'index_topk': 2,
    'hc_mult': 4,
    'bs': 1,
    'seq_len': 8,
    'rms_norm_eps': 1e-6,
}


@pytest.mark.unit
def test_dsv4f_tiny_forward():
    model = Dsv4FlashMoe('dsv4f_tiny', TINY_CFG)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [TINY_CFG['bs'], TINY_CFG['seq_len'], TINY_CFG['vocab_size']]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_dsv4f_param_count():
    model = Dsv4FlashMoe('dsv4f_flash', {'bs': 1, 'seq_len': 1})
    n_params = model.analytical_param_count()
    assert EXPECTED_TOTAL_PARAMS * (1 - PARAM_COUNT_TOLERANCE) < n_params \
        < EXPECTED_TOTAL_PARAMS * (1 + PARAM_COUNT_TOLERANCE), \
        f'param count {n_params:,} outside tolerance of {EXPECTED_TOTAL_PARAMS:,}'


@pytest.mark.unit
def test_dsv4f_decode():
    model = Dsv4FlashMoe('dsv4f_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 129280]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_dsv4f_prefill():
    model = Dsv4FlashMoe('dsv4f_prefill', {'bs': 1, 'seq_len': 128})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 128, 129280]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_dsv4f_layer_kind_counts():
    model = Dsv4FlashMoe('dsv4f_counts', {'bs': 1, 'seq_len': 1})
    # compress_ratios: [0, 0, (4, 128) x 20, 0]
    assert model.n_window_layers == 3  # sliding-window-only (ratio 0)
    assert model.n_csa_layers == 20  # ratio 4 + Lightning Indexer
    assert model.n_hca_layers == 20  # ratio 128, all blocks
    assert model.n_window_layers + model.n_csa_layers + model.n_hca_layers == 43
    # proxy blocks cover every (gate_kind, compress_ratio) layer type exactly once
    assert sum(model.block_counts.values()) == 43
    assert len(model.block_keys) == len(set(model.block_keys))