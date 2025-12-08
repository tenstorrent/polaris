#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 HuggingFace Inc.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN

ACT2CLS = {
    "swish": SimNN.Silu,
    "silu": SimNN.Silu,
}

def get_activation(act_fn: str) -> SimNN.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]('act_fn')
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


class GELU(SimNN.Module):
    def __init__(self, name: str, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.name = name
        self.proj = ttsimF.Linear(f'{self.name}_proj', dim_in, dim_out, bias=bias, module=self)
        self.approximate = approximate
        super().link_op2module()
        self._op_hndls[self.proj.name] = self.proj

    def gelu(self, gate: SimNN.SimTensor) -> SimNN.SimTensor:
        op = ttsimF.Gelu(f'{self.name}_geluop')
        self._op_hndls[op.name] = op
        return op(gate)

    def __call__(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        self._tensors[hidden_states.name] = hidden_states
        return hidden_states
