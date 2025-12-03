#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

## Example of exporting ResNet50 to ONNX with dynamic batch size

# import torch
# import torchvision.models as models

# device = torch.device("cpu")

# model = models.resnet50(weights=None).to(device)
# model.eval()

# dummy_input = torch.randn(1, 3, 224, 224, device=device)

# torch.onnx.export(
#     model,
#     dummy_input,
#     "resnet50.onnx",
#     opset_version=17,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
# )

# print("Saved resnet50.onnx")
