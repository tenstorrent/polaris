#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN

from .downsampling import (  # noqa
    Downsample2D,
)
from .upsampling import (  # noqa
    Upsample2D,
)


class ResnetBlock2D(SimNN.Module):
    def __init__(
        self,
        # *,
        objname: str,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[SimNN.SimTensor] = None, # type: ignore[assignment]
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.non_linearity = non_linearity
        self.name = objname
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = SimNN.GroupNorm(f'{self.name}_resnetblock2d_norm1', num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = ttsimF.Conv2d(f'{self.name}_resnetblock2d_conv2d', in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = ttsimF.Linear(f'{self.name}_linop1', temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = ttsimF.Linear(f'{self.name}_linop2', temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None # type: ignore[unreachable]

        self.norm2 = SimNN.GroupNorm(f'{self.name}_resnetblock2d_norm2', num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = ttsimF.Dropout(f'{self.name}_dropout', dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = ttsimF.Conv2d(f'{self.name}_conv2d', out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity1 = SimNN.Silu(f'{self.name}_act_fn1')
        self.nonlinearity2 = SimNN.Silu(f'{self.name}_act_fn2')
        self.special_nonlinearity = SimNN.Silu(f'{self.name}_act_fn3')

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                raise NotImplementedError("FIR upsampling not supported!")
            elif kernel == "sde_vp":
                raise NotImplementedError("SDE_VP upsampling not supported!")
            else:
                self.upsample = Upsample2D(f'{self.name}_upsample', in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                raise NotImplementedError("FIR downsampling not supported!")
            elif kernel == "sde_vp":
                raise NotImplementedError("SDE_VP downsampling not supported!")
            else:
                # TODO: Cursor review comment
                # Downsample path raises NotImplementedError when non-convolutional downsampling is selected
                # The code attempts to create a Downsample2D with use_conv=False, but the Downsample2D implementation
                # (line 51 in downsampling.py) raises NotImplementedError unconditionally when use_conv=False.
                # This means any ResnetBlock2D instance with down=True and kernel not in ["fir", "sde_vp"] will 
                # raise NotImplementedError during initialization rather than during forward pass, making the module unusable.
                self.downsample = Downsample2D(f'{self.name}_downsample', in_channels, use_conv=False, padding=1)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.addop1 = ttsimF.Add(f'{self.name}_add_op_1')
        self.divop1 = ttsimF.Div(f'{self.name}_div_op_1')

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = ttsimF.Conv2d(
                f'{self.name}_conv_shortcut',
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

        super().link_op2module()

    def __call__(self, input_tensor: SimNN.SimTensor, temb: SimNN.SimTensor, *args, **kwargs) -> SimNN.SimTensor:
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity1(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity2(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb # type: ignore[operator]
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            raise NotImplementedError('scale_shift not implemented!')
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.special_nonlinearity(hidden_states) if self.non_linearity == "silu" else hidden_states

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous()) # type: ignore[attr-defined]

        div_tensor = SimNN.SimTensor({
            'name': f'{self.name}_div_tensor',
            'shape': input_tensor.shape,
            'dtype': input_tensor.dtype,
        })
        self._tensors[div_tensor.name] = div_tensor
        self._tensors[input_tensor.name] = input_tensor
        self._tensors[hidden_states.name] = hidden_states
        output_tensor = self.divop1(self.addop1(input_tensor, hidden_states), div_tensor)
        self._tensors[output_tensor.name] = output_tensor

        return output_tensor
