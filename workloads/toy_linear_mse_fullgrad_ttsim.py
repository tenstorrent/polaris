#!/usr/bin/env python
# Full backward graph for 2-layer MLP + MSE:
# Inputs: x, h, y, target, W2
# Outputs: grad_W2, grad_b2, grad_W1, grad_b1  (approximate)

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

class ToyLinearMSE_FullGrad(SimNN.Module):
    """
    2-layer MLP + MSE backward (approx):
      Forward:
        h = GELU(W1 x)
        y = W2 h
        loss = mean((y - t)^2)

      Backward (approx):
        e       = y - t
        grad_y  ~ e
        grad_W2 = h^T @ grad_y
        grad_b2 = sum(grad_y, axis=0)
        grad_h  = grad_y @ W2
        grad_h_pre ~ grad_h         (skip exact GELU')
        grad_W1 = grad_h_pre^T @ x
        grad_b1 = sum(grad_h_pre, axis=0)
    """

    def __init__(self, objname, cfg):
        super().__init__()
        self.name       = objname
        self.bs         = cfg["bs"]
        self.in_dim     = cfg["in_dim"]
        self.hidden_dim = cfg["hidden_dim"]
        self.out_dim    = cfg["out_dim"]

        # Ops
        self.sub         = F.Sub(f"{self.name}.sub")            # e = y - target
        self.matmul_W2   = F.MatMul(f"{self.name}.gradW2")      # h^T @ grad_y
        self.matmul_h    = F.MatMul(f"{self.name}.gradH")       # grad_y @ W2
        self.matmul_W1   = F.MatMul(f"{self.name}.gradW1")      # grad_h_pre^T @ x
        self.reducesum_b2 = F.ReduceSum(f"{self.name}.gradb2", 0)
        self.reducesum_b1 = F.ReduceSum(f"{self.name}.gradb1", 0)
        self.hT_op       = F.Transpose(f"{self.name}.hT", perm=[1, 0])
        self.ghT_op      = F.Transpose(f"{self.name}.ghT", perm=[1, 0])

        super().link_op2module()

    def create_input_tensors(self):
        # x:      [bs, in_dim]
        # h:      [bs, hidden_dim]
        # y,tgt:  [bs, out_dim]
        # W2:     [out_dim, hidden_dim]
        self.input_tensors = {
            "x": F._from_shape("x", [self.bs, self.in_dim],
                               is_param=False, np_dtype=None),
            "h": F._from_shape("h", [self.bs, self.hidden_dim],
                               is_param=False, np_dtype=None),
            "y": F._from_shape("y", [self.bs, self.out_dim],
                               is_param=False, np_dtype=None),
            "target": F._from_shape("target", [self.bs, self.out_dim],
                                    is_param=False, np_dtype=None),
            "W2": F._from_shape("W2", [self.out_dim, self.hidden_dim],
                                is_param=False, np_dtype=None),
        }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self, x=None, h=None, y=None, target=None, W2=None):
        if x is None:
            x = self.input_tensors["x"]
        if h is None:
            h = self.input_tensors["h"]
        if y is None:
            y = self.input_tensors["y"]
        if target is None:
            target = self.input_tensors["target"]
        if W2 is None:
            W2 = self.input_tensors["W2"]

        # e = y - target
        e = self.sub(y, target)          # [bs, out_dim]

        # toy approx: grad_y ~ e
        grad_y = e                       # [bs, out_dim]

        # grad_W2 = h^T @ grad_y
        h_T = self.hT_op(h)              # [hidden_dim, bs]
        grad_W2 = self.matmul_W2(h_T, grad_y)   # [hidden_dim, out_dim]

        # grad_b2 = sum over batch
        grad_b2 = self.reducesum_b2(grad_y)     # [out_dim]

        # grad_h = grad_y @ W2
        grad_h = self.matmul_h(grad_y, W2)      # [bs, hidden_dim]

        # For now, approximate GELU' as 1 → grad_h_pre ~ grad_h
        grad_h_pre = grad_h

        # grad_W1 = grad_h_pre^T @ x
        gh_T = self.ghT_op(grad_h_pre)          # [hidden_dim, bs]
        grad_W1 = self.matmul_W1(gh_T, x)      # [hidden_dim, in_dim]

        # grad_b1 = sum over batch
        grad_b1 = self.reducesum_b1(grad_h_pre) # [hidden_dim]

        return grad_W2, grad_b2, grad_W1, grad_b1
