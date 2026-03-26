# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0

# """
# Export Qwen3-VL-2B-Instruct *text decoder* to a fixed-shape ONNX graph
# (bs=1, seq_len=128) for Polaris perf projections.

# Key points:
# - Wrap the HF model so forward() returns ONLY logits (no DynamicCache).
# - Disable KV cache via config + forward(use_cache=False).
# - Fixed shapes: input_ids [1, 128], attention_mask [1, 128].
# """

# import torch
# from torch import nn
# from transformers import AutoConfig, Qwen3VLForConditionalGeneration

# MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
# SEQ_LEN = 128
# OUT_PATH = "workloads/onnx/qwen3_vl_2b_text_fixed-128.onnx"

# class Qwen3VLTextOnly(nn.Module):
#     """Thin wrapper that hides DynamicCache and returns only logits."""

#     def __init__(self, base_model: Qwen3VLForConditionalGeneration):
#         super().__init__()
#         self.base_model = base_model

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
#         # Explicitly disable cache in forward so outputs don’t contain DynamicCache
#         out = self.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=False,
#         )
#         return out.logits

# def main() -> None:
#     print(f"[Qwen3-VL-2B] Loading config for {MODEL_ID} ...")
#     cfg = AutoConfig.from_pretrained(MODEL_ID)
#     if hasattr(cfg, "use_cache"):
#         cfg.use_cache = False

#     print(f"[Qwen3-VL-2B] Loading model from_pretrained ({MODEL_ID}) ...")
#     base = Qwen3VLForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         torch_dtype=torch.float16,
#         device_map=None,  # keep on CPU
#     )
#     base.eval()

#     wrapped = Qwen3VLTextOnly(base)
#     wrapped.eval()

#     # Dummy text inputs: [bs=1, seq_len=128]
#     input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
#     attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

#     print(f"[Qwen3-VL-2B] Exporting fixed-shape ONNX to {OUT_PATH} ...")
#     torch.onnx.export(
#         wrapped,
#         (input_ids, attention_mask),
#         OUT_PATH,
#         input_names=["input_ids", "attention_mask"],
#         output_names=["logits"],
#         opset_version=18,
#         dynamic_axes=None,   # all dims fixed
#         do_constant_folding=True,
#     )
#     print("[Qwen3-VL-2B] Saved:", OUT_PATH)

# if __name__ == "__main__":
#     main()