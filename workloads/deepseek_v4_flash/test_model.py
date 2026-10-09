#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone smoke test for the DeepSeek-V4-Flash TTSIM workload."""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from workloads.deepseek_v4_flash.dsv4f_model import Dsv4FlashMoe

# Published total for deepseek-ai/DeepSeek-V4-Flash is 284B (the 1 MTP layer
# is excluded by design; the simulated base model lands at ~284.3B).
EXPECTED_TOTAL_PARAMS = 284_000_000_000
PARAM_COUNT_TOLERANCE = 0.05

TINY_CFG = {
    'vocab_size': 256, 'hidden_size': 64, 'num_hidden_layers': 6,
    'num_hash_layers': 2, 'compress_ratios': [0, 4, 128, 4, 128, 0],
    'num_attention_heads': 4, 'head_dim': 32,
    'q_lora_rank': 16, 'o_lora_rank': 16, 'o_groups': 2,
    'sliding_window': 4,
    'n_routed_experts': 8, 'num_experts_per_tok': 2,
    'moe_intermediate_size': 32, 'n_shared_experts': 1,
    'index_n_heads': 2, 'index_head_dim': 8, 'index_topk': 2,
    'hc_mult': 4, 'bs': 1, 'seq_len': 8, 'rms_norm_eps': 1e-6,
}


def test_tiny_forward():
    model = Dsv4FlashMoe('dsv4f_tiny', TINY_CFG)
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [TINY_CFG['bs'], TINY_CFG['seq_len'], TINY_CFG['vocab_size']]
    gg = model.get_forward_graph()
    assert gg is not None
    print(f'tiny forward OK: output={out.shape}, params={model.analytical_param_count():,}')


def test_dsv4f_decode():
    model = Dsv4FlashMoe('dsv4f_decode', {'bs': 1, 'seq_len': 1})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 1, 129280], f'decode output shape mismatch: {out.shape}'
    n_params = model.analytical_param_count()
    assert EXPECTED_TOTAL_PARAMS * (1 - PARAM_COUNT_TOLERANCE) < n_params \
        < EXPECTED_TOTAL_PARAMS * (1 + PARAM_COUNT_TOLERANCE), \
        f'param count {n_params:,} outside tolerance of {EXPECTED_TOTAL_PARAMS:,}'
    gg = model.get_forward_graph()
    assert gg is not None
    print(f'V4-Flash decode OK: output={out.shape}, params={n_params:,}')


def test_dsv4f_prefill_short():
    # window larger than seq: CSA/HCA compressors emit blocks; indexer topk is
    # a no-op selection (all blocks selected)
    model = Dsv4FlashMoe('dsv4f_prefill_128', {'bs': 1, 'seq_len': 128})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 128, 129280]
    print(f'V4-Flash prefill(128) OK: output={out.shape}')


def test_dsv4f_prefill_sparse_window():
    # L > window: exercises the CSA top-512 cap (S = 128 + 512) and the HCA
    # all-blocks window (S = 128 + 16); indexer topk still a no-op at C == 512
    model = Dsv4FlashMoe('dsv4f_prefill_2k', {'bs': 1, 'seq_len': 2048})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 2048, 129280]
    print(f'V4-Flash prefill(2048) OK: output={out.shape}')


def test_dsv4f_prefill_indexer_topk():
    # C > index_topk: the Lightning-Indexer top-512 selection op runs
    model = Dsv4FlashMoe('dsv4f_prefill_8k', {'bs': 1, 'seq_len': 4096})
    model.create_input_tensors()
    out = model()
    assert list(out.shape) == [1, 4096, 129280]
    print(f'V4-Flash prefill(4096, indexer topk) OK: output={out.shape}')


if __name__ == '__main__':
    test_tiny_forward()
    test_dsv4f_decode()
    test_dsv4f_prefill_short()
    test_dsv4f_prefill_sparse_window()
    test_dsv4f_prefill_indexer_topk()
    print('All DeepSeek-V4-Flash workload tests passed!')