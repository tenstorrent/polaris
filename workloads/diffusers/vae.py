#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from .unet_2d_blocks import (
    UNetMidBlock2D,
    get_up_block,
)


class DecoderOutput():
    sample: SimNN.SimTensor
    commit_loss: Optional[SimNN.SimTensor] = None


class Decoder(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.name = objname
        self.conv_in = F.Conv2d(
            'conv_in_decoder',
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            objname="mid_block",
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,    # type: ignore[arg-type]
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        up_block = [0] * len(up_block_types)
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block[i] = get_up_block( # type: ignore[call-overload]
                up_block_type,
                i,
                self.name,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,    # type: ignore[arg-type]
                resnet_time_scale_shift=norm_type,
            )
        self.up_blocks = SimNN.ModuleList(up_block)

        # out
        self.conv_norm_out = SimNN.GroupNorm("grp_nrm", num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = SimNN.Silu("silu_op")
        self.conv_out = F.Conv2d("conv2d", block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False
        super().link_op2module()

    def __call__(
        self,
        sample: SimNN.SimTensor,
        latent_embeds: Optional[SimNN.SimTensor] = None,
    ) -> SimNN.SimTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        if self.gradient_checkpointing:
            raise NotImplementedError("Gradient checkpointing is not supported yet!")
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        self._tensors[sample.name] = sample
        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
