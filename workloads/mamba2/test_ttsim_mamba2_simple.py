#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from workloads.mamba2.ttsim_mamba2_simple import Mamba2Simple
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

d_model_sz = 128

model = Mamba2Simple(
    # This module uses roughly 3 * expand * d_model^2 parameters
    objname='mamba2_simple',
    d_model=d_model_sz, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
    batch_size=1, # Batch size
    seq_len=16,   # Sequence length
)
# Define input tensor x with shape (batch_size, sequence_length, d_model)
batch_size = 1
sequence_length = 16
# x = F._from_shape('input', [batch_size, sequence_length, d_model_sz])

model.create_input_tensors()
y = model()
print(f'input shape is [{model.batch_size}, {model.seq_len}, {model.d_model}] and output shape is {y.shape}')
assert y.shape == [model.batch_size, model.seq_len, model.d_model], 'Test failed!'
print('Test passed!')

# gg = model.get_forward_graph()
# print('Dumping ONNX...')
# gg.graph2onnx(f'mamba2_simple.onnx', do_model_check=False)
