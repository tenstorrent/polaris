#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt.tt_utils import DictAsAttr

class TtFFN:
    def __init__(self, params, device):
        self.device = device
        params = DictAsAttr(params, depth=1)
        self.linear1_weight = params.linear1.weight
        self.linear2_weight = params.linear2.weight
        self.linear1_bias = params.linear1.bias
        self.linear2_bias = params.linear2.bias

    def __call__(self, x, identity=None):
        if identity is None:
            identity = x

        # First linear + ReLU
        x = ttnn.linear(x, self.linear1_weight, bias=self.linear1_bias)
        x = ttnn.relu(x)

        # Second linear
        x = ttnn.linear(x, self.linear2_weight, bias=self.linear2_bias)

        # Residual connection
        x = ttnn.add(x, identity)
        ttnn.deallocate(identity)
        return x
