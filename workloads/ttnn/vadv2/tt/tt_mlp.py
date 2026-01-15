#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN

class TtMLP(SimNN.Module):
    def __init__(self, params, device, in_channels, hidden_unit, verbose=False):
        super().__init__()
        self.params = params
        self.device = device

    def __call__(self, x):
        x = ttnn.linear(x, self.params['linear']['weight'], bias=self.params['linear']['bias'], memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.layer_norm(
            x, weight=self.params['norm']['weight'], bias=self.params['norm']['bias'], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.relu(x)
        return x
