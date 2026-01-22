#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# import argparse
# import torch
# from torchvision.models import vit_b_16

# def export_vit_onnx(
#     batch_size: int = 8,
#     height: int = 224,
#     width: int = 224,
#     output_path: str = "workloads/onnx/vit-base-fixed-224.onnx",
# ) -> None:
#     model = vit_b_16(weights=None)
#     model.eval()
#     dummy = torch.randn(batch_size, 3, height, width)
#     torch.onnx.export(
#         model,
#         dummy,
#         output_path,
#         input_names=["input"],
#         output_names=["logits"],
#         opset_version=18,          
#         dynamic_axes=None,         # all dims fixed for Polaris
#         do_constant_folding=True,
#     )

#     print(f"Saved ViT-B/16 ONNX to {output_path}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--batch", type=int, default=8)
#     p.add_argument("--height", type=int, default=224)
#     p.add_argument("--width", type=int, default=224)
#     p.add_argument(
#         "--output",
#         type=str,
#         default="workloads/onnx/vit-base-fixed-224.onnx",
#     )
#     args = p.parse_args()

#     export_vit_onnx(
#         batch_size=args.batch,
#         height=args.height,
#         width=args.width,
#         output_path=args.output,
#     )