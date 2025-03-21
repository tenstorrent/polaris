import numpy as np
from workloads.BasicLLM import BasicLLM
import ttsim.front.functional.op as F

def test_simnn():
    llm_cfg = {
        'vocab_sz'    : 100,
        'bs'          : 7,
        'nW'          : 11,
        'nH'          : 3,
        'dE'          : 15,
        'nL'          : 4,
        'embd_pdrop'  : 0.1,
        'attn_pdrop'  : 0.1,
        'resid_pdrop' : 0.1,
        'mlp_pdrop'   : 0.1,
        }

    basic_llm = BasicLLM('basic_llm', llm_cfg)
    basic_llm.create_input_tensors()
    llm_out = basic_llm() #all intermediate tensor shape/data fixed after __call__ call

    llm_graph = basic_llm.get_forward_graph()
    assert len(llm_graph._ops) == 31
    assert len(llm_graph._tensors) == 62
    assert len(llm_graph._input_tensors) == 29
    assert len(llm_graph._input_nodes) == 21
    assert len(llm_graph._output_nodes) == 1
    assert len(llm_graph._output_tensors) == 1
