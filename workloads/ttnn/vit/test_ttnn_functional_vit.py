#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vit.ttnn_functional_vit import vit

def run_vit():
    device = ttnn.open_device(device_id=0)
    class Config:
        def __init__(self):
            self.num_hidden_layers = 12
            self.num_attention_heads = 12

    config = Config()

    # Parameters dictionary entries:
    class Parameters:
        def __init__(self):
            self.vit = self.Vit()
            self.layernorm_before = self.Vit.LayerNorm()
            self.layernorm_after = self.Vit.LayerNorm()
            self.classifier = self.Classifier()
        class Vit:
            def __init__(self):
                self.embeddings = self.Embeddings()
                self.encoder = self.Encoder()
                self.layernorm = self.LayerNorm()
            class Embeddings:
                def __init__(self):
                    self.patch_embeddings = self.PatchEmbeddings()
                    self.cls_token = ttnn._rand(shape=[1, 1, 768], dtype=ttnn.bfloat16, device=device)
                    self.position_embeddings = ttnn._rand(shape=[1, 197, 768], dtype=ttnn.bfloat16, device=device)
                class PatchEmbeddings:
                    def __init__(self):
                        self.projection = self.Projection()

                    class Projection:
                        def __init__(self):
                            self.weight = ttnn._rand(shape=[1024, 768], dtype=ttnn.bfloat16, device=device)
                            self.bias = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)
            class DenseIntermediate:
                def __init__(self):
                    self.weight = ttnn._rand(shape=[768, 3072], dtype=ttnn.bfloat16, device=device)
                    self.bias = ttnn._rand(shape=[1, 3072], dtype=ttnn.bfloat16, device=device)
            class Intermediate:
                def __init__(self):
                    self.dense = Parameters.Vit.DenseIntermediate()
            class Attention:
                def __init__(self):
                    self.query = self.QKV()
                    self.key = self.QKV()
                    self.value = self.QKV()
                    self.output = self.Out()
                class QKV:
                    def __init__(self):
                        self.weight = ttnn._rand(shape=[768, 768], dtype=ttnn.bfloat16, device=device)
                        self.bias = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)
                class Dense:
                    def __init__(self):
                        self.weight = ttnn._rand(shape=[768, 768], dtype=ttnn.bfloat16, device=device)
                        self.bias = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)
                class Out:
                    def __init__(self):
                        self.dense = Parameters.Vit.Attention.Dense()
                class VitOut:
                    def __init__(self):
                        self.dense = Parameters.Vit.Attention.VitOutDense()
                class VitOutDense:
                    def __init__(self):
                        self.weight = ttnn._rand(shape=[3072, 768], dtype=ttnn.bfloat16, device=device)
                        self.bias = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)
            class Layer:
                def __init__(self):
                    self.layernorm_before = Parameters.Vit.LayerNorm()
                    self.layernorm_after = Parameters.Vit.LayerNorm()
                    self.attention = Parameters.Vit.Attention()
                    self.intermediate = Parameters.Vit.Intermediate()
                    self.output = Parameters.Vit.Attention.VitOut()
            class Encoder:
                def __init__(self):
                    self.layer = [Parameters.Vit.Layer() for _ in range(config.num_hidden_layers)]

            class LayerNorm:
                def __init__(self):
                    self.weight = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)
                    self.bias = ttnn._rand(shape=[1, 768], dtype=ttnn.bfloat16, device=device)

        class Classifier:
            def __init__(self):
                self.weight = ttnn._rand(shape=[768, 1152], dtype=ttnn.bfloat16, device=device)
                self.bias = ttnn._rand(shape=[1, 1152], dtype=ttnn.bfloat16, device=device)

    parameters = Parameters()
    pixel_values = ttnn._rand(shape=[8, 224, 14, 64], dtype=ttnn.bfloat16, device=device)
    cls_token = ttnn._rand(shape=[8, 1, 768], dtype=ttnn.bfloat16, device=device)
    position_embeddings = ttnn._rand(shape=[8, 197, 768], dtype=ttnn.bfloat16, device=device)

    output = vit(
        config,
        pixel_values,
        None,
        cls_token,
        position_embeddings,
        parameters=parameters,
    )
    print('Input shape:', pixel_values.shape)
    print("Output shape:", output.shape)
    if output.shape == [8, 197, 1152]:
        return True
    else:
        return False

if __name__ == "__main__":
    if run_vit():
        print("Test Passed")
    else:
        print("Test Failed")
