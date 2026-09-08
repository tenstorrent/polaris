#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dual-mode Embedding for the llama3 tt_transformers port.

Basis: shim-only workloads/ttnn/tt_transformers/embedding.py, made dual-mode and
audited against tt-metal models/tt_transformers/tt/embedding.py.

Audit vs the shim-only base:
  - Embedding table is 2D (vocab, dim): the shim's ttnn.embedding op requires a 2D weight
    (rope.get_rot_mats likewise does `.squeeze(0).squeeze(0)` before ttnn.embedding), and
    real ttnn.embedding takes a 2D (num_embeddings, embedding_dim) weight. (The capture's
    EmbeddingsDeviceOperation reports a 4D in1 (1x1x128256x4096), but a 4D table breaks the
    shim's embedding shape inference, so 2D is used on both paths — op output is unchanged.)
  - Dummy weights (config-only, no HF checkpoint): shim path uses shim-native ttnn._rand;
    HW path a torch dummy. memory_config left default (DRAM); tt-metal uses EMB_WEIGHTS_MEMCFG
    via get_model_config, which the lean ModelArgs omits.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

IS_POLARIS = os.getenv('IRD_ARCH_NAME', '') == ''
if IS_POLARIS:
    import ttsim.front.ttnn as ttnn
else:
    import ttnn  # type: ignore[import-not-found, no-redef]


class Embedding:
    def __init__(self, mesh_device, args, weight_cache_path, state_dict, dtype, dim=None):
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        if dim is None:
            dim = args.dim

        # Dummy embedding table, 2D (vocab, dim) — required by the shim/real ttnn.embedding op.
        # Fabricated directly on-device in both modes: for llama3-8B this table is enormous
        # (128256 x 4096), so the HW path avoids allocating it as a host torch tensor first.
        # Values are irrelevant (accuracy not validated). mesh_mapper / cache_file_name were
        # no-ops at single-device cluster [1, 1].
        self.weights = ttnn.zeros(
            [args.vocab_size, dim],
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )

    def forward(self, x: 'ttnn.Tensor', memory_config=None) -> 'ttnn.Tensor':
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
        return ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)

    def __call__(self, x, memory_config=None):
        return self.forward(x, memory_config)
