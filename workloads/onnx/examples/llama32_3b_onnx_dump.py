#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# import torch
# from transformers import AutoModelForCausalLM, AutoConfig

# MODEL_ID = "meta-llama/Llama-3.2-3B"

# SEQ_LEN = 128
# OUT_PATH = "workloads/onnx/llama3_3b_fixed-128.onnx"

# def main():
#     print(f"Loading config for {MODEL_ID}...")
#     cfg = AutoConfig.from_pretrained(MODEL_ID)
#     cfg.use_cache = False  

#     print("Building model from config (random weights) ...")
#     model = AutoModelForCausalLM.from_config(cfg)
#     model = model.to(dtype=torch.float16)  
#     model.eval()

#     input_ids = torch.zeros(1, SEQ_LEN, dtype=torch.long)
#     attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.long)

#     print(f"Exporting to {OUT_PATH}...")
#     torch.onnx.export(
#         model,
#         (input_ids, attention_mask),
#         OUT_PATH,
#         input_names=["input_ids", "attention_mask"],
#         output_names=["logits"],
#         opset_version=18,        
#         dynamic_axes=None,      
#         do_constant_folding=True,
#     )
#     print("Saved:", OUT_PATH)

# if __name__ == "__main__":
#     main()