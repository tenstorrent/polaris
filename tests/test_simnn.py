#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from workloads.BasicLLM import BasicLLM
import ttsim.front.functional.op as F

@pytest.mark.unit
def test_simnn():
    llm_cfg = {
        'vocab_sz'             : 100,
        'drop_prob'            : 0.1,
        'bs'                   : 7,
        'nL'                   : 4,
        'nH'                   : 3,
        'dE'                   : 15,
        'nW'                   : 11,
        'attn_type'            : 'bidir',
        'norm_type'            : 'layer',
        'positional_encoding'  : 'learned',
        'use_segment_embedding': True,
        }

    basic_llm = BasicLLM('basic_llm', llm_cfg)
    basic_llm.create_input_tensors()
    llm_out = basic_llm() #all intermediate tensor shape/data fixed after __call__ call

    llm_graph = basic_llm.get_forward_graph()
    assert len(llm_graph._ops)            == 33
    assert len(llm_graph._tensors)        == 65
    assert len(llm_graph._input_tensors)  == 30
    assert len(llm_graph._input_nodes)    == 22
    assert len(llm_graph._output_nodes)   == 1
    assert len(llm_graph._output_tensors) == 1
