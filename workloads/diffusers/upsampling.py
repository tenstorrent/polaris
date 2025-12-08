#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN


class Upsample2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.objname = objname
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            raise NotImplementedError("LayerNorm not supported!")
        elif norm_type == "rms_norm":
            raise NotImplementedError("RMSNorm not supported!")
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            raise NotImplementedError("ConvTranspose2d not supported!")
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = ttsimF.Conv2d(f'{self.objname}.conv2d', self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        if name == "conv":
            self.conv = conv

        super().link_op2module()

    def __call__(self, hidden_states: SimNN.SimTensor, output_size: Optional[int] = None, *args, **kwargs) -> SimNN.SimTensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"Upsample2D: Input channel mismatch. Expected {self.channels}, got {hidden_states.shape[1]}")

        if self.interpolate:
            hidden_states = hidden_states.interpolate(scale_factor=2.0, mode="nearest") # type: ignore[attr-defined]
            self._tensors[hidden_states.name] = hidden_states

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states) # type: ignore[misc]

        return hidden_states
