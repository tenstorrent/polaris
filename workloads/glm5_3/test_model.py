#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test for the GLM-5.3 TTSIM workload."""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.glm5_3.glm53_model import Glm53Moe

# Published total for zai-org/GLM-5.3 is 753.3B *including* the 1 MTP layer,
# which this perf model excludes by design; the simulated base model lands
# at ~747B (0.8% under the checkpoint count).
EXPECTED_TOTAL_PARAMS = 747_000_000_000
PARAM_COUNT_TOLERANCE = 0.05

TINY_CFG = {
    'vocab_size': 256, 'hidden_size': 64, 'num_hidden_layers': 8,
    'first_k_dense_replace': 1, 'intermediate_size': 128,
    'num_attention_heads': 4, 'num_key_value_heads': 4,
    'q_lora_rank': 32, 'kv_lora_rank': 16,
    'qk_nope_head_dim': 8, 'qk_rope_head_dim': 4, 'v_head_dim': 16,
    'n_routed_experts': 8, 'num_experts_per_tok': 2,
    'moe_intermediate_size': 32, 'n_shared_experts': 1,
    'index_n_heads': 2, 'index_head_dim': 8, 'index_topk': 4,
    'bs': 1, 'seq_len': 8, 'rms_norm_eps': 1e-5,
}


def test_tiny_forward():
    model = Glm53Moe('glm53_tiny', TINY_CFG)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [TINY_CFG['bs'], TINY_CFG['seq_len'], TINY_CFG['vocab_size']]
    gg = model.get_forward_graph()
    assert gg is not None
    print(f'tiny forward OK: output={out.shape}, params={model.analytical_param_count():,}')


def test_glm53_decode():
    model = Glm53Moe('glm53_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 154880], f'decode output shape mismatch: {out.shape}'
    n_params = model.analytical_param_count()
    assert EXPECTED_TOTAL_PARAMS * (1 - PARAM_COUNT_TOLERANCE) < n_params \
        < EXPECTED_TOTAL_PARAMS * (1 + PARAM_COUNT_TOLERANCE), \
        f'param count {n_params:,} outside tolerance of {EXPECTED_TOTAL_PARAMS:,}'
    gg = model.get_forward_graph()
    assert gg is not None
    print(f'GLM-5.3 decode OK: output={out.shape}, params={n_params:,}')


def test_glm53_prefill_short():
    # L <= index_topk: full attention window, indexer top-k is a no-op selection
    model = Glm53Moe('glm53_prefill_2k', {'bs': 1, 'seq_len': 512})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 512, 154880]
    print(f'GLM-5.3 prefill(512) OK: output={out.shape}')


def test_glm53_prefill_sparse_window():
    # L > index_topk: exercises the DSA sparse window slicing path
    model = Glm53Moe('glm53_prefill_8k', {'bs': 1, 'seq_len': 4096, 'index_topk': 2048})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 4096, 154880]
    print(f'GLM-5.3 prefill(4096, topk=2048) OK: output={out.shape}')


if __name__ == '__main__':
    test_tiny_forward()
    test_glm53_decode()
    test_glm53_prefill_short()
    test_glm53_prefill_sparse_window()
    print('All GLM-5.3 workload tests passed!')