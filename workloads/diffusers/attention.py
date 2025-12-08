#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN
from .activations import GELU
from .attention_processor import Attention, AttentionProcessor


class BasicTransformerBlock(SimNN.Module):
    def __init__(
        self,
        objname: str,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continuous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.name = objname
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.double_self_attention = double_self_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positional_embeddings` must also be defined."
            )

        self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = ttsimF.LayerNorm(f'{self.name}_norm1', dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            objname=f"{self.name}_SelfAttention",
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = ttsimF.LayerNorm(f'{self.name}_norm2', dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            self.attn2 = Attention(
                objname=f"{self.name}_CrossAttention",
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
        else:
            if norm_type == "ada_norm_single":  # For Latte
                self.norm2 = ttsimF.LayerNorm(f'{self.name}_norm2', dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
            else:
                self.norm2 = None
            self.attn2 = None   # type: ignore[assignment]

        # 3. Feed-forward
        self.norm3 = ttsimF.LayerNorm(f'{self.name}_norm3', dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        
        self.ff = FeedForward(
            f"{self.name}_transformer_block_ff",
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            raise NotImplementedError("Gated Self Attention not supported!")

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            raise NotImplementedError("Single AdaNorm not supported!")

        self._chunk_size = None
        self._chunk_dim = 0

        super().link_op2module()

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size   #type: ignore[assignment]
        self._chunk_dim = dim

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        attention_mask: Optional[SimNN.SimTensor] = None,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        encoder_attention_mask: Optional[SimNN.SimTensor] = None,
        timestep: Optional[SimNN.SimTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,  #type: ignore[assignment]
        class_labels: Optional[SimNN.SimTensor] = None,
        added_cond_kwargs: Optional[Dict[str, SimNN.SimTensor]] = None,
    ) -> SimNN.SimTensor:

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            raise NotImplementedError("ada_norm_continuous not implemented!")
        elif self.norm_type == "ada_norm_single":
            raise NotImplementedError("Single AdaNorm not supported!")
        else:
            raise ValueError("Incorrect norm used")

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states # type: ignore[operator]
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)    # type: ignore

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            raise NotImplementedError("GLIGEN not supported!")

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])    # type: ignore[index]
            else:
                raise ValueError("Incorrect norm")

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states # type: ignore[operator]

        # 4. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            raise NotImplementedError("ada_norm_continuous not implemented!")
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            raise NotImplementedError("Chunked feed forward not supported!")
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states   # type: ignore[operator]
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1)    # type: ignore[attr-defined]

        return hidden_states


class FeedForward(SimNN.Module):
    def __init__(
        self,
        objname: str,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.name = objname
        self.act_fn = GELU(f'{self.name}_geluop1', dim, inner_dim, bias=bias)
        self.linop = ttsimF.Linear(f'{self.name}_linop', inner_dim, dim_out, bias=bias)
        super().link_op2module()

    def __call__(self, hidden_states: SimNN.SimTensor, *args, **kwargs) -> SimNN.SimTensor:
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linop(hidden_states)
        return hidden_states
