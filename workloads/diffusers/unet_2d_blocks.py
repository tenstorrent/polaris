#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Tuple, Union, List, TYPE_CHECKING

import os, sys

from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN
from .attention_processor import Attention
from .resnet import (
    Downsample2D,
    ResnetBlock2D,
    Upsample2D,
)
from .transformer_2d import Transformer2DModel


def get_down_block(
    down_block_type: str,
    index: int,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        if TYPE_CHECKING:
            assert resnet_groups is not None and downsample_padding is not None and num_attention_heads is not None # for mypy
        return CrossAttnDownBlock2D(
            objname=f"CrossAttnDownBlock2D_{index}",
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    elif down_block_type == "DownBlock2D":
        if TYPE_CHECKING:
            assert resnet_groups is not None and downsample_padding is not None # for mypy
        return DownBlock2D(
            objname=f"down_block_2d_{index}",
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    prefix_name: str,
    temb_channels: int,
    in_channels: int,
    resnet_eps: float,
    resnet_act_fn: str,
    resnet_groups: int,
    output_scale_factor: float = 1.0,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    mid_block_only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = 1,
    dropout: float = 0.0,
):
    if mid_block_type == "UNetMidBlock2DCrossAttn":
        if TYPE_CHECKING:
            assert cross_attention_dim is not None and num_attention_heads is not None # for mypy
        return UNetMidBlock2DCrossAttn(
            f'{prefix_name}_UNetMidBlock2DCrossAttn',
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resnet_groups=resnet_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )
    elif mid_block_type == "UNetMidBlock2D":
        return UNetMidBlock2D(
            f'{prefix_name}_UNetMidBlock2D',
            in_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            num_layers=0,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            add_attention=False,
        )
    else:
        raise ValueError(f"unknown mid_block_type : {mid_block_type}")


def get_up_block(
    up_block_type: str,
    index: int,
    prefix_name: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> SimNN.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type

    if up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        if TYPE_CHECKING:
            assert resnet_groups is not None and num_attention_heads is not None # for mypy
        return CrossAttnUpBlock2D(
            objname=f"{prefix_name}_cross_attn_{index}",
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
        )
    elif up_block_type == "UpBlock2D":
        assert resnet_groups is not None # for mypy
        return UpBlock2D(
            objname=f"{prefix_name}_up_block_{index}",
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "UpDecoderBlock2D":
        assert resnet_groups is not None # for mypy
        return UpDecoderBlock2D(
            objname=f"{prefix_name}_updecoderblock2d_{index}",
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        self.name = objname
        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        if resnet_time_scale_shift == "spatial":
            raise NotImplementedError("ResNet time scale shift 'spatial' (resnet_time_scale_shift='spatial') is not supported")

        if attention_head_dim is None:
            attention_head_dim = in_channels    # type: ignore[unreachable]

        self.attentions = SimNN.ModuleList([Attention(
                        f"{self.name}_attn_block_{_}",
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    ) for _ in range(num_layers) if self.add_attention])

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                        objname=f"{self.name}_resnet_block_{_}",
                        in_channels=in_channels,
                        out_channels=in_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    ) for _ in range(num_layers) if resnet_time_scale_shift != "spatial"])

        self.gradient_checkpointing = False
        super().link_op2module()

    def __call__(self, hidden_states: SimNN.SimTensor, temb: Optional[SimNN.SimTensor] = None) -> SimNN.SimTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class UNetMidBlock2DCrossAttn(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        temb_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, List[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_groups_out: Optional[int] = None,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        self.name = objname
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnet_groups_out = resnet_groups_out or resnet_groups

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                objname=f"{self.name}_ResnetBlock2D_{i}",
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                groups_out=resnet_groups_out,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            ) for i in range(num_layers + 1)])

        self.attentions = SimNN.ModuleList([Transformer2DModel(
                        f"{self.name}_Transformer2DModel_{i}",
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups_out,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    ) for i in range(num_layers) if not dual_cross_attention])

        self.gradient_checkpointing = False
        super().link_op2module()

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        temb: Optional[SimNN.SimTensor] = None,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
    ) -> SimNN.SimTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.", extra={"once": True})

        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class CrossAttnDownBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, List[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        self.name = objname

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        in_chans = [0] * num_layers
        for i in range(num_layers):
            in_chans[i] = in_channels if i == 0 else out_channels

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                    objname=f"{self.name}_ResnetBlock2D_{i}",
                    in_channels=in_chans[i],
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) for i in range(num_layers)])

        self.attentions = SimNN.ModuleList([Transformer2DModel(
                        f"{self.name}_Transformer2DModel_{i}",
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    ) for i in range(num_layers) if not dual_cross_attention])

        self.downsamplers: Optional[SimNN.ModuleList] = None

        if add_downsample:
            self.downsamplers = SimNN.ModuleList(
                [
                    Downsample2D(f'{self.name}_Downsample2D',
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        super().link_op2module()

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        temb: Optional[SimNN.SimTensor] = None,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
        additional_residuals: Optional[SimNN.SimTensor] = None,
    ) -> Tuple[SimNN.SimTensor, Tuple[SimNN.SimTensor, ...]]:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.", extra={"once": True})

        output_states: Tuple[SimNN.SimTensor, ...] = ()

        blocks = list(zip(self.resnets, self.attentions))

        for i, (resnet, attn) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals  # type: ignore[operator]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class DownBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()
        self.name = objname
        in_chan = [0] * num_layers
        for i in range(num_layers):
            in_chan[i] = in_channels if i == 0 else out_channels

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                    objname=f"{self.name}_ResnetBlock2D_{i}",
                    in_channels=in_chan[i],
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) for i in range(num_layers)])

        self.downsamplers: Optional[SimNN.ModuleList] = None
        if add_downsample:
            self.downsamplers = SimNN.ModuleList(
                [
                    Downsample2D(f'{self.name}_Downsample2D',
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False
        super().link_op2module()

    def __call__(
        self, hidden_states: SimNN.SimTensor, temb: Optional[SimNN.SimTensor] = None, *args, **kwargs
    ) -> Tuple[SimNN.SimTensor, Tuple[SimNN.SimTensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            logger.warning("scale 1.0.0 {}", deprecation_message, extra={"once": True})

        output_states: Tuple[SimNN.SimTensor, ...] = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, List[int]] = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
    ):
        super().__init__()
        self.name = objname

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        res_skip_channels = [0] * num_layers
        resnet_in_channels = [0] * num_layers
        for i in range(num_layers):
            res_skip_channels[i] = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels[i] = prev_output_channel if i == 0 else out_channels

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                    objname=f'{self.name}_ResnetBlock2D_{i}',
                    in_channels=resnet_in_channels[i] + res_skip_channels[i],
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                ) for i in range(num_layers)])

        self.attentions = SimNN.ModuleList([Transformer2DModel(
                        f'{self.name}_Transformer2DModel_{i}',
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    ) for i in range(num_layers) if not dual_cross_attention])

        self.upsamplers: Optional[SimNN.ModuleList] = None
        if add_upsample:
            self.upsamplers = SimNN.ModuleList([Upsample2D(f'{self.name}_Upsample2D', out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx
        self.cat = ttsimF.ConcatX(f'{self.name}.catop', axis = 1)
        super().link_op2module()

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        res_hidden_states_tuple: Tuple[SimNN.SimTensor, ...],
        temb: Optional[SimNN.SimTensor] = None,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
    ) -> SimNN.SimTensor:

        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            self._tensors[res_hidden_states.name] = res_hidden_states
            self._tensors[hidden_states.name] = hidden_states
            hidden_states = self.cat(hidden_states, res_hidden_states)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.name = objname

        res_skip_channels = [0] * num_layers
        resnet_in_channels = [0] * num_layers

        for i in range(num_layers):
            res_skip_channels[i] = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels[i] = prev_output_channel if i == 0 else out_channels

        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                                            objname=f"{self.name}.resnet_block_{i}",
                                            in_channels=resnet_in_channels[i] + res_skip_channels[i],
                                            out_channels=out_channels,
                                            temb_channels=temb_channels,
                                            eps=resnet_eps,
                                            groups=resnet_groups,
                                            dropout=dropout,
                                            time_embedding_norm=resnet_time_scale_shift,
                                            non_linearity=resnet_act_fn,
                                            output_scale_factor=output_scale_factor,
                                            pre_norm=resnet_pre_norm,
                                        ) for i in range(num_layers)])

        self.upsamplers: Optional[SimNN.ModuleList] = None
        if add_upsample:
            self.upsamplers = SimNN.ModuleList([Upsample2D(f'{self.name}.upsample', out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx
        self.cat = ttsimF.ConcatX(f'{self.name}.catop', axis = 1)
        super().link_op2module()

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        res_hidden_states_tuple: Tuple[SimNN.SimTensor, ...],
        temb: Optional[SimNN.SimTensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> SimNN.SimTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            logger.warning("scale 1.0.0 {}", deprecation_message, extra={"once": True})

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            self._tensors[res_hidden_states.name] = res_hidden_states
            self._tensors[hidden_states.name] = hidden_states
            hidden_states = self.cat(hidden_states, res_hidden_states)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpDecoderBlock2D(SimNN.Module):
    def __init__(
        self,
        objname: str,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()
        self.name = objname

        input_channels = [0] * num_layers
        for i in range(num_layers):
            input_channels[i] = in_channels if i == 0 else out_channels

        if TYPE_CHECKING:
            assert temb_channels is not None # for mypy
        self.resnets = SimNN.ModuleList([ResnetBlock2D(
                        objname=f"{self.name}.updecoderblock2d_{i}",
                        in_channels=input_channels[i],
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    ) for i in range(num_layers)])

        self.upsamplers: Optional[SimNN.ModuleList] = None
        if add_upsample:
            self.upsamplers = SimNN.ModuleList([Upsample2D(f'{self.name}_upsample', out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx
        super().link_op2module()

    def __call__(self, hidden_states: SimNN.SimTensor, temb: Optional[SimNN.SimTensor] = None) -> SimNN.SimTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
