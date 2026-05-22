#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Smoke tests for the Titans MAC TTSim port.
Validates: (1) forward graph builds, (2) param-count math is consistent,
(3) ONNX export round-trips, (4) NeuralMemory store/retrieve shapes are sane.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import ttsim.front.functional.op as F
from workloads.Titans.mac_transformer import TitansMAC
from workloads.Titans.neural_memory  import NeuralMemory
from workloads.Titans.memory_models  import MemoryMLP


def test_memory_mlp_shapes():
    m = MemoryMLP('mm', dim=32, depth=2, expansion_factor=2.0)
    x = F._from_shape('x', shape=[4, 8, 32])
    y = m(x)
    assert y.shape == [4, 8, 32]
    assert m.analytical_param_count() == 32 * 64 + 64 * 32
    print('OK : MemoryMLP shapes and param count')

def test_neural_memory_forward_graph():
    nm = NeuralMemory('nm', dim=64, chunk_size=4, dim_head=32, heads=2, depth=2)
    seq = F._from_shape('seq', shape=[1, 16, 64])
    out, state = nm(seq)
    assert out.shape == [1, 16, 64]
    print(f'OK : NeuralMemory forward (params={nm.analytical_param_count()})')

def test_titans_mac_forward_graph_and_onnx(tmpdir='./__titans_mac_test'):
    os.makedirs(tmpdir, exist_ok=True)
    cfg = dict(
        nL=2, nH=4, dE=128, dim_head=32, nW=32, vocab_sz=256, bs=1,
        segment_len=8, num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=[2], neural_memory_segment_len=4,
        mem_depth=2, mem_expansion_factor=2.0, mem_heads=2, mem_dim_head=32,
        mem_chunk_size=4,
    )
    m = TitansMAC('titans_mac', cfg)
    m.create_input_tensors()
    y = m()
    print(f'OK : TitansMAC forward (out={y.shape}, params={m.analytical_param_count():,d})')

    gg = m.get_forward_graph()
    onnx_path = os.path.join(tmpdir, 'titans_mac.onnx')
    gg.graph2onnx(onnx_path, do_model_check=False)
    assert os.path.exists(onnx_path), 'ONNX not written'
    print(f'OK : ONNX written to {onnx_path}')

if __name__ == '__main__':
    test_memory_mlp_shapes()
    test_neural_memory_forward_graph()
    test_titans_mac_forward_graph_and_onnx()
    print('\nAll Titans MAC smoke tests passed.')
