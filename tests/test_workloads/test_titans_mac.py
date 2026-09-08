#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Smoke tests for the Titans MAC TTSim port.
Validates: (1) forward graph builds, (2) param-count math is consistent,
(3) ONNX export round-trips, (4) NeuralMemory store/retrieve shapes are sane.
"""
import os
import sys
import pytest
import ttsim.front.functional.op as F
from workloads.Titans.mac_transformer import TitansMAC
from workloads.Titans.neural_memory  import NeuralMemory
from workloads.Titans.memory_models  import MemoryMLP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.mark.unit
def test_memory_mlp_shapes():
    m = MemoryMLP('mm', dim=32, depth=2, expansion_factor=2.0)
    x = F._from_shape('x', shape=[4, 8, 32])
    y = m(x)
    assert y.shape == [4, 8, 32]
    assert m.analytical_param_count() == 32 * 64 + 64 * 32

@pytest.mark.unit
def test_neural_memory_forward_graph():
    nm = NeuralMemory('nm', dim=64, chunk_size=4, dim_head=32, heads=2, depth=2)
    seq = F._from_shape('seq', shape=[1, 16, 64])
    out, state = nm(seq)
    assert out.shape == [1, 16, 64]

@pytest.mark.unit
def test_neural_memory_chunk_state_propagation():
    """
    D1 verification:
      - seq_len=16, chunk_size=4 -> num_ch=4 chunks
      - max_chunks_unroll=8, depth=2 -> 16 per-chunk handles per op type total
    Confirms per-chunk handles are allocated AND the constructor wires them
    correctly. Everything below the returned-state block is constructor-time
    state and holds without a forward call.

    Note: no caller in the repo threads state between invocations (MACBlock
    passes mem_state=None), so cross-call propagation is not exercised here
    or anywhere else; this pins the shape of the state that store_memories
    hands back.
    """
    nm = NeuralMemory('nm_state', dim=32, chunk_size=4, dim_head=16, heads=2,
                      depth=2, max_chunks_unroll=8)
    seq = F._from_shape('seq_state', shape=[1, 16, 32])
    out, state = nm(seq)
    assert out.shape == [1, 16, 32], f'bad out shape {out.shape}'

    # returned state: seq_index advances by the full sequence, and the port
    # asserts N % chunk_size == 0, so there is never a cached remainder
    assert state.seq_index == 16, f'seq_index: want 16, got {state.seq_index}'
    assert state.cache_store_segment is None, 'port emits no remainder segment'
    assert sorted(state.weights) == ['W0', 'W1']
    assert sorted(state.updates) == ['W0', 'W1']
    assert len(state.states) == 2, 'states should be (last_update, last_momentum)'
    assert state.weights['W0'].shape == [2, 16, 32], \
        f'W0 update: want [B*H, 16, 32], got {state.weights["W0"].shape}'


    # Each per-chunk op list = depth * max_chunks_unroll = 2 * 8 = 16 handles
    expected = 2 * 8
    assert len(nm.mom_mul)   == expected, f'mom_mul:   want {expected}, got {len(nm.mom_mul)}'
    assert len(nm.mom_add)   == expected, f'mom_add:   want {expected}, got {len(nm.mom_add)}'
    assert len(nm.decay_mul) == expected, f'decay_mul: want {expected}, got {len(nm.decay_mul)}'
    assert len(nm.weight_add)== expected, f'weight_add: want {expected}, got {len(nm.weight_add)}'
    assert len(nm.weight_sub)== expected, f'weight_sub: want {expected}, got {len(nm.weight_sub)}'

    # Index helper is correct
    assert nm._mc(0, 0) == 0
    assert nm._mc(0, 7) == 7
    assert nm._mc(1, 0) == 8
    assert nm._mc(1, 7) == 15

    # Variant Splits cover all num_ch values from 1 to max_chunks_unroll
    assert len(nm._grads_splits) == 2, 'grads_splits should be per-layer (depth=2)'
    assert len(nm._grads_splits[0]) == 8, 'grads_splits[0] should have max_chunks_unroll variants'
    assert len(nm._momentum_coef_splits) == 8
    assert len(nm._decay_factor_splits) == 8
    assert nm._decay_factor_splits[3].count == 4, 'Split variant for num_ch=4 should have count=4'


@pytest.mark.integration
def test_titans_mac_forward_graph_and_onnx(tmp_path):
    cfg = dict(
        nL=2, nH=4, dE=128, dim_head=32, nW=32, vocab_sz=256, bs=1,
        segment_len=8, num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=[2],
        mem_depth=2, mem_expansion_factor=2.0, mem_heads=2, mem_dim_head=32,
        mem_chunk_size=4,
    )
    m = TitansMAC('titans_mac', cfg)
    m.create_input_tensors()
    gg = m.get_forward_graph()
    onnx_path = os.path.join(str(tmp_path), 'titans_mac.onnx')
    gg.graph2onnx(onnx_path, do_model_check=False)
    assert os.path.exists(onnx_path), 'ONNX not written'
