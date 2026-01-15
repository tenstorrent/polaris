#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from workloads.ttnn.vadv2.tt.tt_transformer import TtVADPerceptionTransformer
from ttsim.front.ttnn import DataType, Layout, Shape
from loguru import logger

mesh_device = ttnn.open_device(device_id=0)

class dummyModel(SimNN.Module):
    def __init__(self, device, x=0):
        super().__init__()
        self.device = device
        self.name = "dummyModel"
        self.x = x

    def forward(self, x):
        return x

def test_vadv2_transformer(device=mesh_device):
    parameter = {
    "encoder": {
        "layers": {
        "layer0": {
            "attentions": {
            "attn0": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([512, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([512, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer1": {
            "attentions": {
            "attn0": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([512, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([512, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer2": {
            "attentions": {
            "attn0": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([512, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([512, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        },
    },
    "decoder": {
        "layers": {
        "layer0": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer1": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer2": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        },
    },
    "map_decoder": {
        "layers": {
        "layer0": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer1": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        "layer2": {
            "attentions": {
            "attn0": {
                "in_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 768]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "out_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            "attn1": {
                "sampling_offsets": {
                "weight": ttnn.Tensor(shape=Shape([256, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 64]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "attention_weights": {
                "weight": ttnn.Tensor(shape=Shape([256, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 32]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "value_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "output_proj": {
                "weight": ttnn.Tensor(shape=Shape([256, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "ffn": {
            "ffn0": {
                "linear1": {
                "weight": ttnn.Tensor(shape=Shape([256, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 512]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
                "linear2": {
                "weight": ttnn.Tensor(shape=Shape([512, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                },
            },
            },
            "norms": {
            "norm0": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm1": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            "norm2": {
                "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
                "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
            },
            },
        },
        },
    },
    "reference_points": {
        "weight": ttnn.Tensor(shape=Shape([256, 3]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        "bias":   ttnn.Tensor(shape=Shape([1, 3]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
    },
    "map_reference_points": {
        "weight": ttnn.Tensor(shape=Shape([256, 2]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        "bias":   ttnn.Tensor(shape=Shape([1, 2]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
    },
    "can_bus_mlp": {
        "0": {
        "weight": ttnn.Tensor(shape=Shape([18, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        "bias":   ttnn.Tensor(shape=Shape([1, 128]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        },
        "1": {
        "weight": ttnn.Tensor(shape=Shape([128, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        },
        "norm": {
        "weight": ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        "bias":   ttnn.Tensor(shape=Shape([1, 256]), layout=Layout.TILE_LAYOUT, dtype=DataType.BFLOAT16, device=device),
        },
    },
    }

    dummyModelObj = dummyModel(device, x=3)
    bev_h = 100
    bev_w = 100
    grid_length = (0.6, 0.3)
    bev_queries = ttnn._rand(shape=(bev_h * bev_w, 256), dtype=ttnn.bfloat16, device=device)
    object_query_embed = ttnn._rand(shape=(300, 512), dtype=ttnn.bfloat16, device=device)
    map_query_embed = ttnn._rand(shape=(2000, 512), dtype=ttnn.bfloat16, device=device)
    mlvl_feats = []
    a = ttnn._rand(shape=(1, 6, 256, 12, 20), dtype=ttnn.bfloat16, device=device)
    mlvl_feats.append(a)
    bev_pos = ttnn._rand(shape=(1, 256, 100, 100), dtype=ttnn.bfloat16, device=device)
    img_metas = [
        {
            "can_bus": np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.9686697,
                    -0.60694152,
                    -0.07634412,
                    9.87149385,
                    -0.02108691,
                    -0.01243972,
                    -0.023067,
                    8.5640597,
                    0.0,
                    0.0,
                    5.78155401,
                    0.0,
                ]
            ),
            "lidar2img": [
                np.array(
                    [
                        [2.48597954e02, 1.68129905e02, 6.55251068e00, -7.08702279e01],
                        [-3.64025219e00, 1.07359713e02, -2.45107509e02, -1.28941576e02],
                        [-1.17025046e-02, 9.98471159e-01, 5.40221896e-02, -4.25203639e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [2.72989308e02, -1.23852972e02, -8.06783283e00, -9.23285717e01],
                        [7.58924673e01, 6.40614553e01, -2.47958947e02, -1.38511256e02],
                        [8.43406855e-01, 5.36312055e-01, 3.21598489e-02, -6.10371854e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [6.47396684e00, 3.00630854e02, 1.55246365e01, -6.04875770e01],
                        [-7.78640394e01, 6.40883103e01, -2.47490601e02, -1.35884951e02],
                        [-8.23415292e-01, 5.65940098e-01, 4.12196894e-02, -5.29677094e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-1.60796449e02, -1.70144772e02, -5.28753263e00, -1.74159198e02],
                        [-2.16465632e00, -8.90571925e01, -1.62979489e02, -1.41736848e02],
                        [-8.33350064e-03, -9.99200442e-01, -3.91028008e-02, -1.01645350e00],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [-2.37313222e02, 1.84652288e02, 1.06528318e01, -1.25068238e02],
                        [-9.25251029e01, -2.05081174e01, -2.50495434e02, -1.12365691e02],
                        [-9.47586752e-01, -3.19482867e-01, 3.16948959e-03, -4.32527296e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
                np.array(
                    [
                        [5.70378465e01, -2.93855304e02, -1.19126859e01, -5.45200638e01],
                        [8.89472086e01, -2.45651403e01, -2.50078534e02, -1.17649223e02],
                        [9.24052925e-01, -3.82246554e-01, -3.70989150e-03, -4.64645142e-01],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                ),
            ],
            "img_shape": [(192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3), (192, 320, 3)],
        }
    ]
    use_shift = True
    delta_x = np.array([each["can_bus"][0] for each in img_metas])  # type: ignore
    delta_y = np.array([each["can_bus"][1] for each in img_metas])  # type: ignore
    ego_angle = np.array([each["can_bus"][-2] / np.pi * 180 for each in img_metas])  # type: ignore
    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]
    translation_length = np.sqrt(delta_x**2 + delta_y**2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
    shift_y = shift_y * use_shift
    shift_x = shift_x * use_shift
    shift = bev_queries.new_tensor([shift_x, shift_y])
    shift.set_module(dummyModelObj)
    shift = shift.permute([1, 0])  # xy, bs -> bs, xy

    ttnn_model = TtVADPerceptionTransformer(
        params=parameter,
        device=device,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        decoder=True,
        map_decoder=True,
        embed_dims=256,
        params_branches=None,
    )
    can_bus = bev_queries.new_tensor([each["can_bus"] for each in img_metas])  # [:, :]
    can_bus = ttnn.from_torch(can_bus, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    logger.debug('input shape is ', mlvl_feats[0].shape)
    ttnn_outputs = ttnn_model(
        mlvl_feats,
        bev_queries,
        object_query_embed,
        map_query_embed,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        reg_branches=None,
        map_reg_branches=None,
        img_metas=img_metas,
        shift=shift,
        can_bus=can_bus,
    )

    logger.debug("Checking outputs...")

    if (mlvl_feats[0].shape == [1, 6, 256, 12, 20] and
        ttnn_outputs[0].shape == [10000, 1, 256] and
        ttnn_outputs[1].shape == [3, 300, 1, 256] and
        ttnn_outputs[2].shape == [1, 300, 3] and
        ttnn_outputs[3].shape == [3, 1, 300, 3] and
        ttnn_outputs[4].shape == [3, 2000, 1, 256] and
        ttnn_outputs[5].shape == [1, 2000, 2] and
        ttnn_outputs[6].shape == [3, 1, 2000, 2]):
        logger.debug("Test passed: Output shapes are as expected.")
        return 0
    else:
        raise AssertionError("Test failed: Output shapes are not as expected.")

if __name__ == "__main__":
    test_vadv2_transformer()
    logger.debug("Test completed successfully.")
