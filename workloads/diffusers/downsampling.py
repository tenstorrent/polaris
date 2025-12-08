#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN


class Downsample2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = objname

        if norm_type == "ln_norm":
            self.norm = ttsimF.LayerNorm(f'{self.name}_layernorm1', channels)
        elif norm_type == "rms_norm":
            raise NotImplementedError("RMSNorm not supported!")
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = ttsimF.Conv2d(f'{self.name}_conv',
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
        else:
            if self.channels != self.out_channels:
                raise ValueError(f'Non-convolutional downsampling requires channels == out_channels, got {self.channels} != {self.out_channels}')
            raise NotImplementedError(f'Non-convolutional downsampling (use_conv=False) is not implemented')

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

        super().link_op2module()

    def __call__(self, hidden_states: SimNN.SimTensor, *args, **kwargs) -> SimNN.SimTensor:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"Downsample2D: Input channel mismatch. Expected {self.channels}, got {hidden_states.shape[1]}")

        if self.norm is not None:
            raise NotImplementedError("Normalization in Downsample2D forward pass is not implemented")

        self._tensors[hidden_states.name] = hidden_states
        hidden_states = self.conv(hidden_states)

        return hidden_states
