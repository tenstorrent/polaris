#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3 TTSIM workload (workloads/glm5_3/)."""
from typing import Any, Dict

import pytest

from workloads.glm5_3.glm53_model import Glm53Moe

# Published total for zai-org/GLM-5.3 is 753.3B *including* the 1 MTP layer,
# which this perf model excludes by design; the simulated base lands at ~747B.
EXPECTED_TOTAL_PARAMS = 747_000_000_000
PARAM_COUNT_TOLERANCE = 0.05

TINY_CFG: Dict[str, Any] = {
    'vocab_size': 256,
    'hidden_size': 64,
    'num_hidden_layers': 8,
    'first_k_dense_replace': 1,
    'intermediate_size': 128,
    'num_attention_heads': 4,
    'num_key_value_heads': 4,
    'q_lora_rank': 32,
    'kv_lora_rank': 16,
    'qk_nope_head_dim': 8,
    'qk_rope_head_dim': 4,
    'v_head_dim': 16,
    'n_routed_experts': 8,
    'num_experts_per_tok': 2,
    'moe_intermediate_size': 32,
    'n_shared_experts': 1,
    'index_n_heads': 2,
    'index_head_dim': 8,
    'index_topk': 4,
    'bs': 1,
    'seq_len': 8,
    'rms_norm_eps': 1e-5,
}


@pytest.mark.unit
def test_glm53_tiny_forward():
    model = Glm53Moe('glm53_tiny', TINY_CFG)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [TINY_CFG['bs'], TINY_CFG['seq_len'], TINY_CFG['vocab_size']]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_glm53_param_count():
    model = Glm53Moe('glm53_full', {'bs': 1, 'seq_len': 1})
    n_params = model.analytical_param_count()
    assert EXPECTED_TOTAL_PARAMS * (1 - PARAM_COUNT_TOLERANCE) < n_params \
        < EXPECTED_TOTAL_PARAMS * (1 + PARAM_COUNT_TOLERANCE), \
        f'param count {n_params:,} outside tolerance of {EXPECTED_TOTAL_PARAMS:,}'


@pytest.mark.unit
def test_glm53_decode():
    model = Glm53Moe('glm53_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 154880]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_glm53_prefill():
    model = Glm53Moe('glm53_prefill', {'bs': 1, 'seq_len': 128})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 128, 154880]
    gg = model.get_forward_graph()
    assert gg is not None


@pytest.mark.unit
def test_glm53_layer_kind_counts():
    model = Glm53Moe('glm53_counts', {'bs': 1, 'seq_len': 1})
    # indexer_types: full at 0-3 and every 4th after -> 22 full, 56 shared
    assert model.n_full_idx == 22
    assert model.n_shared_moe == 56
    assert model.n_dense_full + model.n_moe_full + model.n_shared_moe == 78