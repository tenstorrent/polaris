# #!/usr/bin/env python3
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# import argparse
# import torch
# from transformers import DetrForObjectDetection

# def export_detr_onnx(
#     batch_size: int = 1,
#     height: int = 800,
#     width: int = 800,
#     output_path: str = "workloads/onnx/detr-fixed-800.onnx",
#     hf_model_name: str = "facebook/detr-resnet-50",
# ) -> None:
#     model = DetrForObjectDetection.from_pretrained(hf_model_name)
#     model.eval()

#     #  Dummy input: [bs, 3, H, W]
#     dummy = torch.randn(batch_size, 3, height, width)

#     torch.onnx.export(
#         model,
#         dummy,                       
#         output_path,
#         input_names=["pixel_values"],
#         output_names=["logits", "boxes"],
#         opset_version=18,
#         dynamic_axes=None,           
#         do_constant_folding=True,
#     )

#     print(f"Saved HF DETR ONNX to {output_path}")

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--batch", type=int, default=1)
#     p.add_argument("--height", type=int, default=800)
#     p.add_argument("--width", type=int, default=800)
#     p.add_argument(
#         "--output",
#         type=str,
#         default="workloads/onnx/detr-fixed-800.onnx",
#     )
#     p.add_argument(
#         "--hf-model",
#         type=str,
#         default="facebook/detr-resnet-50",
#         help="Hugging Face model id to export",
#     )
#     args = p.parse_args()

#     export_detr_onnx(
#         batch_size=args.batch,
#         height=args.height,
#         width=args.width,
#         output_path=args.output,
#         hf_model_name=args.hf_model,
#     )