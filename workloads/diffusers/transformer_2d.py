#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN
from .attention import BasicTransformerBlock


class Transformer2DModel(SimNN.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock"]
    _skip_layerwise_casting_patterns = ["latent_image_embedding", "norm"]

    def __init__(
        self,
        objname: str,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,   # type: ignore[assignment]
        interpolation_scale: float = None,  # type: ignore[assignment]
        use_additional_conditions: Optional[bool] = None,
    ):
        super().__init__()

        self.name = objname
        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        self.permuteop_0231 = ttsimF.permute(f'{self.name}.perm_op', (0, 2, 3, 1))
        self.permuteop_0312 = ttsimF.permute(f'{self.name}.perm_op2', (0, 3, 1, 2))
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized and not self.is_input_patches:
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            norm_type = "ada_norm"

        # Set some common variables used across the board.
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale = interpolation_scale
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False
        self.norm_num_groups = norm_num_groups
        self.num_layers = num_layers
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.attention_bias = attention_bias
        self.only_cross_attention = only_cross_attention
        self.double_self_attention = double_self_attention
        self.upcast_attention = upcast_attention
        self.norm_type = norm_type
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.attention_type = attention_type

        if use_additional_conditions is None:
            if norm_type == "ada_norm_single" and sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 2. Initialize the right blocks.
        # These functions follow a common structure:
        # a. Initialize the input blocks. b. Initialize the transformer blocks.
        # c. Initialize the output blocks and other projection blocks when necessary.
        if self.is_input_continuous:
            self._init_continuous_input(norm_type=norm_type)
        elif self.is_input_vectorized:
            raise NotImplementedError("vectorized input not supported!")
        elif self.is_input_patches:
            raise NotImplementedError("patched input not supported!")

    def _init_continuous_input(self, norm_type):
        self.norm = SimNN.GroupNorm(f'{self.name}.grpnrm1',
            num_groups=self.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        self.proj_in = ttsimF.Conv2d(f'{self.name}.conv_in', self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = SimNN.ModuleList(
            [
                BasicTransformerBlock(
                    f"{self.name}_BasicTransformerBlock{i}",
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    attention_type=self.attention_type,
                )
                for i in range(self.num_layers)
            ]
        )

        self.proj_out = ttsimF.Conv2d(f'{self.name}.conv_out', self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)


    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        timestep: Optional[SimNN.SimTensor] = None,
        added_cond_kwargs: Dict[str, SimNN.SimTensor] = None,   # type: ignore[assignment]
        class_labels: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,  # type: ignore[assignment]
        attention_mask: Optional[SimNN.SimTensor] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
        return_dict: bool = True,
    ):
        if attention_mask is not None:
            raise NotImplementedError('attention_mask with 2 dims not supported in ttsim!')

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            raise NotImplementedError('encoder_attention_mask with 2 dims not supported in ttsim!')

        # 1. Input
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            raise NotImplementedError("vectorized input forward pass not implemented!")
        elif self.is_input_patches:
            raise NotImplementedError("patched input forward pass not implemented!")

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            raise NotImplementedError("vectorized input forward pass not implemented!")
        elif self.is_input_patches:
            raise NotImplementedError("patched input forward pass not implemented!")

        if not return_dict:
            return (output,)

        return output

    def _operate_on_continuous_inputs(self, hidden_states):
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            self._tensors[hidden_states.name] = hidden_states
            hidden_states = self.permuteop_0231(hidden_states)
            hidden_states.set_module(self)
            hidden_states = hidden_states.reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = self.permuteop_0231(hidden_states).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim

    def _get_output_for_continuous_inputs(self, hidden_states, residual, batch_size, height, width, inner_dim):
        if not self.use_linear_projection:
            hidden_states = (
                self.permuteop_0312(hidden_states.reshape(batch_size, height, width, inner_dim)).contiguous()
            )
            self._tensors[hidden_states.name] = hidden_states
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch_size, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual
        return output
