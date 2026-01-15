#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt.tt_mlp import TtMLP
import ttsim.front.functional.sim_nn as SimNN

class TtLaneNet(SimNN.Module):
    def __init__(self, params, device, in_channels, hidden_unit, num_subgraph_layers):
        super().__init__()
        self.params = params
        self.device = device
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = []
        for i in range(num_subgraph_layers):
            layer_name = f"lmlp_{i}"
            self.layer_seq.append(TtMLP(params[layer_name], device, in_channels, hidden_unit))
            in_channels = hidden_unit * 2  # Update if output is concatenated like in LaneNet

    def __call__(self, pts_lane_feats):
        x = pts_lane_feats
        for layer in self.layer_seq:
            if isinstance(layer, TtMLP):
                x = layer(x)
                x_max = ttnn.max(x, dim=-2) # [0]
                x_max = ttnn.unsqueeze(x_max, 2)
                x_max = ttnn.repeat(x_max, (1, 1, x.shape[2], 1))
                x = ttnn.concat(x, x_max, axis=-1)

        x_max = ttnn.max(x, dim=-2).squeeze(0) #[0]

        return x_max
