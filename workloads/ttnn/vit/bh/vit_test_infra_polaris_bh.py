#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side equivalent of ``vit_test_infra.py`` from tt-metal (BH variant).

Provides ``create_test_infra`` with the same interface so that
``test_vit_device_perf_bh.py`` can run in dual-mode.  Mirrors
``vit_test_infra_polaris_wh.py`` (Wormhole) but pulls config and parameters
from ``vit_polaris_params_bh.py`` and binds the BH polaris model.
"""

import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]

from ttsim.front.ttnn.device import set_default_device
from ttsim.front.ttnn.tensor import ttnn_random
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.memory import MemoryConfig

import workloads.ttnn.vit.bh.ttnn_optimized_sharded_vit_bh as ttnn_optimized_sharded_vit
from workloads.ttnn.vit.bh.vit_polaris_params_bh import (
    config_dict,
    config_obj,
    polaris_vit_parameters,
)


class VitTestInfra:
    """Polaris BH mirror of the upstream VitTestInfra.

    Builds synthetic config, parameters, cls_token, position_embeddings, and
    random pixel_values with the same shapes the HW path uses.
    """

    def __init__(
        self,
        device,
        batch_size,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
        use_random_input_tensor=False,
        model_location_generator=None,
    ):
        torch.manual_seed(0)
        set_default_device(device)

        self.device = device
        self.batch_size = batch_size
        self.config = config_obj

        image_size = config_dict["image_size"]
        patch_size = config_dict["patch_size"]
        hidden = config_dict["hidden_size"]
        sequence_len = 1 + (image_size // patch_size) ** 2  # 197
        num_labels = 1152

        self.parameters = polaris_vit_parameters(num_labels=num_labels)

        torch_cls_token = ttnn_random(
            (batch_size, 1, hidden), -0.1, 0.1, dtype=torch.bfloat16,
        )
        torch_position_embeddings = ttnn_random(
            (batch_size, sequence_len, hidden), -0.1, 0.1, dtype=torch.bfloat16,
        )
        self.cls_token = ttnn.from_torch(torch_cls_token, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.position_embeddings = ttnn.from_torch(
            torch_position_embeddings, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
        )

        # Pixel values already in post-reshape shape [B, H, W/patch, 4*patch]
        # to avoid emitting host-side Permute/Pad/Reshape SimOps.
        self.torch_pixel_values = ttnn_random(
            (batch_size, image_size, image_size // patch_size, 4 * patch_size),
            -1, 1, dtype=torch.bfloat16,
        )
        self.input_tensor = None

    def setup_l1_sharded_input(self, device, torch_pixel_values=None, mesh_mapper=None, mesh_composer=None):
        if torch_pixel_values is None:
            torch_pixel_values = self.torch_pixel_values
        tt_inputs_host = ttnn.from_torch(
            torch_pixel_values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        input_mem_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, _torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer,
        )
        sharded_mem_config_DRAM = MemoryConfig.DRAM
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        input_tensor = tt_input_tensor if tt_input_tensor is not None else self.input_tensor
        self.output_tensor = ttnn_optimized_sharded_vit.vit(
            self.config,
            input_tensor,
            self.cls_token,
            self.position_embeddings,
            parameters=self.parameters,
        )
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
    use_random_input_tensor=False,
    model_location_generator=None,
):
    return VitTestInfra(
        device,
        batch_size,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
        use_random_input_tensor,
        model_location_generator=model_location_generator,
    )
