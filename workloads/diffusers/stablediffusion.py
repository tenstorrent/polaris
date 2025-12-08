#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 HuggingFace Inc.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
from loguru import logger

from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from workloads.diffusers.autoencoder_kl import AutoencoderKL
from workloads.diffusers.unet_2d_condition import UNet2DConditionModel

class SDModel(SimNN.Module):
    def __init__(self, name, cfg):
        super().__init__()
        self.name           = name
        self.bs             = cfg.get('bs', 1)
        self.vae = AutoencoderKL(  # random weights
            objname="vae",
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),  # type: ignore[arg-type]
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),    # type: ignore[arg-type]
            block_out_channels=(128, 256, 512, 512),    # type: ignore[arg-type]
            latent_channels=4,
            norm_num_groups=32,
            sample_size=32,
        )

        self.unet = UNet2DConditionModel(  # random weights
            objname="unet",
            sample_size=32,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),  # type: ignore[arg-type]
            down_block_types=(  # type: ignore[arg-type]
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(    # type: ignore[arg-type]
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=512,
            attention_head_dim=8,
        )

        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
                'txt_embeds': F._from_shape('txt_embeds',  [2, 77, 512], np_dtype=np.int64),
                'latents': F._from_shape('latents',  [1, 4, 64, 64], np_dtype=np.int64)
                }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self):
        text_embeddings = self.input_tensors['txt_embeds']
        latents = self.input_tensors['latents']
        for i in range(1):
            t = F._from_shape(f'timestep_{i}', shape=[])
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings)
        logger.info(f'ttsim unet input - latents is of shape {latents.shape}')
        logger.info(f'ttsim unet output - noise_pred shape is {noise_pred.shape}')
        image = self.vae.decode(noise_pred)
        return image

def test_model() -> None:
    sd_model = SDModel('StableDiffusion', {'bs': 1})
    sd_model.create_input_tensors()
    image = sd_model()
    if (image.shape == [1, 3, 512, 512]):
        logger.info(f'ttsim SD output image shape is as expected: {image.shape}')
    else:
        logger.error(f'Error: ttsim SD output image shape is NOT as expected: {image.shape}. Expected shape is [1, 3, 512, 512]')

    # print('Generating model graph...')
    # gg = sd_model.get_forward_graph()
    # onnx_ofilename = f'stable_diffusion_.onnx'
    # gg.graph2onnx(onnx_ofilename, do_model_check=False)

if __name__ == "__main__":
    test_model()
