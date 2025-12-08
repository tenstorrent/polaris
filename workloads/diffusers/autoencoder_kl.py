#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from .attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
)
from .vae import Decoder, DecoderOutput


class AutoencoderKL(SimNN.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "ResnetBlock2D"]

    def __init__(
        self,
        objname: str,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()
        self.name = objname

        # pass init params to Decoder
        self.decoder = Decoder(
            objname="decoder",
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = F.Conv2d('conv1', 2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = F.Conv2d('conv2', latent_channels, latent_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        self.tile_overlap_factor = 0.25
        super().link_op2module()

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def _decode(self, z: SimNN.SimTensor, return_dict: bool = True) -> Union[DecoderOutput, SimNN.SimTensor]:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)   # type: ignore[return-value]

        return dec

    def decode(
        self, z: SimNN.SimTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, SimNN.SimTensor]:
        decoded = self._decode(z)

        if not return_dict:
            return (decoded,)   # type: ignore[return-value]

        return decoded
