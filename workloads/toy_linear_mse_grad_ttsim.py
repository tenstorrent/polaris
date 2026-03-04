#!/usr/bin/env python
# Simple backward graph for last layer of ToyLinearMSE:
# Inputs: h, y, target
# Outputs: grad_W2 approx (no GELU grad yet)

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class ToyLinearMSE_LastLayerGrad(SimNN.Module):
    """
    Backward for last linear layer of 2-layer MLP + MSE:
      e       = y - target
      grad_y  = 2 * e / N
      grad_W2 = h^T @ grad_y
      grad_b2 = sum(grad_y, axis=0)
    """

    def __init__(self, objname, cfg):
        super().__init__()
        self.name    = objname
        self.bs      = cfg["bs"]
        self.out_dim = cfg["out_dim"]
        self.hidden_dim = cfg["hidden_dim"]

        # Ops for backward
        self.sub   = F.Sub(f"{self.name}.sub")     # e = y - target
        self.matmul_w2 = F.MatMul(f"{self.name}.gradW2")  # h^T @ grad_y
        self.reducesum_b2 = F.ReduceSum(f"{self.name}.gradb2", 0)
        self.hT_op        = F.Transpose(f"{self.name}.hT", perm=[1, 0])
        super().link_op2module()

    def create_input_tensors(self):
        # shapes match forward:
        # h:      [bs, hidden_dim]
        # y,tgt:  [bs, out_dim]
        self.input_tensors = {
            "h": F._from_shape("h", [self.bs, self.hidden_dim],
                               is_param=False, np_dtype=None),
            "y": F._from_shape("y", [self.bs, self.out_dim],
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

    def __call__(self, h=None, y=None, target=None):
        if h is None:
            h = self.input_tensors["h"]
        if y is None:
            y = self.input_tensors["y"]
        if target is None:
            target = self.input_tensors["target"]

        # e = y - target
        e = self.sub(y, target)      # [bs, out_dim]

        # toy approx: grad_y ~ e
        grad_y = e

        # h^T: [hidden_dim, bs], grad_y: [bs, out_dim]
        h_T = self.hT_op(h)
        grad_W2 = self.matmul_w2(h_T, grad_y)     # [hidden_dim, out_dim]

        # grad_b2 = sum over batch
        grad_b2 = self.reducesum_b2(grad_y)       # [out_dim]

        return grad_W2, grad_b2
