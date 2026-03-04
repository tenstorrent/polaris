#!/usr/bin/env python3
import torch
import torch.nn as nn

class ToyLinearMSE(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=8):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.act  = nn.GELU()          # or nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, x, target):
        h = self.lin1(x)
        h = self.act(h)
        y = self.lin2(h)
        return self.loss(y, target)

def main():
    model = ToyLinearMSE()
    model.eval()

    bs, in_dim, out_dim = 4, 16, 8
    x = torch.randn(bs, in_dim)
    target = torch.randn(bs, out_dim)

    torch.onnx.export(
        model,
        (x, target),
        "workloads/onnx/linear_mseloss.onnx",
        input_names=["x", "target"],
        output_names=["loss"],
        opset_version=18,
        dynamic_axes=None,
    )
    print("Saved workloads/onnx/linear_mseloss.onnx")

if __name__ == "__main__":
    main()
