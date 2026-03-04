#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class ToyLinearMSE(SimNN.Module):
    """
    2-layer MLP + MSE-style loss:
      h = GELU(W1 x)
      y = W2 h
      loss = mean((y - target)^2)
    """

    def __init__(self, objname, cfg):
        super().__init__()
        self.name       = objname
        self.in_dim     = cfg["in_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.out_dim    = cfg["out_dim"]
        self.bs         = cfg["bs"]

        self.lin1 = F.Linear(f"{self.name}.lin1", self.in_dim, self.hidden_dim)
        self.act  = F.Gelu(f"{self.name}.gelu")
        self.lin2 = F.Linear(f"{self.name}.lin2", self.hidden_dim, self.out_dim)

        self.sub  = F.Sub(f"{self.name}.sub")
        self.mul  = F.Mul(f"{self.name}.mul")
        self.mean = F.Mean(f"{self.name}.mean", dim=None)

        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
            "x": F._from_shape("x", [self.bs, self.in_dim],
                               is_param=False, np_dtype=None),
            "target": F._from_shape("target", [self.bs, self.out_dim],
                                    is_param=False, np_dtype=None),
        }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self, x=None, target=None):
        if x is None:
            x = self.input_tensors["x"]
        if target is None:
            target = self.input_tensors["target"]

        h   = self.lin1(x)
        h   = self.act(h)
        y   = self.lin2(h)
        err = self.sub(y, target)
        sq  = self.mul(err, err)
        loss = self.mean(sq)

        return loss
