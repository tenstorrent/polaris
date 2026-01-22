#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ## Example of exporting BERT to ONNX with fixed input shapes

# import torch
# from transformers import BertTokenizer, BertModel

# model_name = "bert-base-uncased"

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertModel.from_pretrained(model_name)
# model.eval()

# # Choose fixed sequence length
# SEQ_LEN = 128

# # Create dummy input with fixed size
# dummy_input = {
#    "input_ids": torch.zeros(1, SEQ_LEN, dtype=torch.long),
#    "attention_mask": torch.ones(1, SEQ_LEN, dtype=torch.long)
# }

# # Export ONNX with fixed shapes
# torch.onnx.export(
#    model,
#    (dummy_input["input_ids"], dummy_input["attention_mask"]),
#    "bert-fixed-128.onnx",
#    input_names=["input_ids", "attention_mask"],
#    output_names=["output"],
#    opset_version=14,
#    dynamic_axes=None  #  No dynamic axes → fixed graph
# )

# print("Saved fixed-shape ONNX: bert-fixed-128.onnx")

