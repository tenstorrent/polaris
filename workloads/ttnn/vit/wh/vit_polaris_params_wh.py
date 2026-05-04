#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synthetic parameter trees and config for Polaris-side ViT (optimized sharded).

Shared by ``run_ttnn_optimized_sharded_vit_wh.py`` and
``vit_test_infra_polaris_wh.py`` so that parameter shapes stay in sync.
"""

import pathlib
import types

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.config import WormholeComputeKernelConfig
from ttsim.front.ttnn.op import MathFidelity

from workloads.common.hf_config import load_hf_config

config_dict = load_hf_config(
    "google/vit-base-patch16-224",
    cache_dir=pathlib.Path(__file__).parent.parent / "common",
)

config_obj = types.SimpleNamespace(**config_dict)
config_obj.core_grid = ttnn.CoreGrid(
    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
)
config_obj.program_configs = {
    "ln_compute_config": WormholeComputeKernelConfig(
        math_fidelity=MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ),
}



def make_info(weight_shape, bias_shape, *, report_dtype=None):
    """Create a weight/bias namespace.

    *report_dtype* sets ``_ttnn_dtype`` on the tensors, which affects both
    stats/CSV output and byte-count / bandwidth accounting (``element_size()``
    prefers ``_ttnn_dtype``).  This is intentional: HW stores linear (MatMul)
    weights as BFLOAT8_B, so perf modeling should account them at 1 byte/element.
    LayerNorm gamma/beta remain at the default BFLOAT16.
    """
    w = ttnn.Tensor(shape=weight_shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.Tensor(shape=bias_shape, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
    if report_dtype is not None:
        w._ttnn_dtype = report_dtype
        b._ttnn_dtype = report_dtype
    return types.SimpleNamespace(weight=w, bias=b)


# ---------------------------------------------------------------------------
# Synthetic parameter tree builders for the optimized sharded ViT.
# These mirror the parameter tree produced by preprocess_model_parameters
# with custom_preprocessor on the HuggingFace ViT model.
# Key difference from functional ViT: fused QKV weights and double
# attention.attention nesting.
# ---------------------------------------------------------------------------


def _polaris_vit_embeddings_patch_parameters():
    """Embeddings.patch_embeddings subtree (projection only)."""
    hidden = config_dict["hidden_size"]
    return types.SimpleNamespace(
        patch_embeddings=types.SimpleNamespace(
            projection=make_info(
                weight_shape=ttnn.Shape([1024, hidden]),
                bias_shape=ttnn.Shape([1, hidden]),
                report_dtype=ttnn.DataType.BFLOAT8_B,
            )
        )
    )


def polaris_parameters_vit_patch_embeddings():
    """Full parameters root for vit_patch_embeddings(..., unittest_check=True)."""
    return types.SimpleNamespace(
        vit=types.SimpleNamespace(embeddings=_polaris_vit_embeddings_patch_parameters())
    )


class Parameters_attention_optimized:
    """ViTAttention parameter tree with fused QKV (optimized sharded variant).

    Tree structure matches vit_attention() access paths:
      parameters.attention.query_key_value.weight  [768, 2304]
      parameters.attention.query_key_value.bias     [1, 2304]
      parameters.output.dense.weight                [768, 768]
      parameters.output.dense.bias                  [1, 768]
    """
    def __init__(self):
        hidden = config_dict["hidden_size"]
        qkv = make_info(
            weight_shape=ttnn.Shape([hidden, hidden * 3]),
            bias_shape=ttnn.Shape([1, hidden * 3]),
            report_dtype=ttnn.DataType.BFLOAT8_B,
        )
        dense = make_info(
            weight_shape=ttnn.Shape([hidden, hidden]),
            bias_shape=ttnn.Shape([1, hidden]),
            report_dtype=ttnn.DataType.BFLOAT8_B,
        )
        self.attention = types.SimpleNamespace(query_key_value=qkv)
        self.output = types.SimpleNamespace(dense=dense)


class Parameters_dense_intermediate:
    def __init__(self):
        hidden = config_dict["hidden_size"]
        intermediate = config_dict["intermediate_size"]
        self.dense = make_info(
            weight_shape=ttnn.Shape([hidden, intermediate]),
            bias_shape=ttnn.Shape([1, intermediate]),
            report_dtype=ttnn.DataType.BFLOAT8_B,
        )


class Parameters_dense_output:
    def __init__(self):
        hidden = config_dict["hidden_size"]
        intermediate = config_dict["intermediate_size"]
        self.dense = make_info(
            weight_shape=ttnn.Shape([intermediate, hidden]),
            bias_shape=ttnn.Shape([1, hidden]),
            report_dtype=ttnn.DataType.BFLOAT8_B,
        )


def _polaris_vit_encoder_layer_parameters():
    """Single ViT encoder block parameters matching the optimized sharded model.

    Full layer tree:
      layer.layernorm_before.{weight,bias}
      layer.layernorm_after.{weight,bias}
      layer.attention.attention.query_key_value.{weight,bias}  (double attention!)
      layer.attention.output.dense.{weight,bias}
      layer.intermediate.dense.{weight,bias}
      layer.output.dense.{weight,bias}
    """
    hidden = config_dict["hidden_size"]
    attn = Parameters_attention_optimized()
    return types.SimpleNamespace(
        layernorm_before=make_info(
            weight_shape=ttnn.Shape([1, hidden]),
            bias_shape=ttnn.Shape([1, hidden]),
        ),
        layernorm_after=make_info(
            weight_shape=ttnn.Shape([1, hidden]),
            bias_shape=ttnn.Shape([1, hidden]),
        ),
        attention=attn,
        intermediate=Parameters_dense_intermediate(),
        output=Parameters_dense_output(),
    )


def polaris_parameters_vit_encoder():
    """Encoder-only parameters for vit_encoder (stack of transformer blocks)."""
    num_layers = config_dict["num_hidden_layers"]
    return types.SimpleNamespace(
        layer=[_polaris_vit_encoder_layer_parameters() for _ in range(num_layers)]
    )


def polaris_vit_parameters(*, num_labels: int = 1152):
    """Full ViT parameter tree for POLARIS.
    Note: classifier is padded from 1000 to 1152 for tile alignment."""
    hidden = config_dict["hidden_size"]

    embeddings = _polaris_vit_embeddings_patch_parameters()
    encoder = polaris_parameters_vit_encoder()
    layernorm = make_info(
        weight_shape=ttnn.Shape([1, hidden]),
        bias_shape=ttnn.Shape([1, hidden]),
    )
    vit_ns = types.SimpleNamespace(
        embeddings=embeddings,
        encoder=encoder,
        layernorm=layernorm,
    )
    classifier = make_info(
        weight_shape=ttnn.Shape([hidden, num_labels]),
        bias_shape=ttnn.Shape([1, num_labels]),
        report_dtype=ttnn.DataType.BFLOAT8_B,
    )
    return types.SimpleNamespace(vit=vit_ns, classifier=classifier)
