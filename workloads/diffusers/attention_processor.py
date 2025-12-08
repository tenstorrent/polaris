#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
from typing import Optional, Union
from loguru import logger

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import ttsim.front.functional.op as ttsimF
import ttsim.front.functional.sim_nn as SimNN

xformers = None
XLA_AVAILABLE = False


class Attention(SimNN.Module):
    def __init__(
        self,
        objname: str,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor = None,
        out_dim: int = None,    # type: ignore[assignment]
        out_context_dim: int = None, # type: ignore[assignment]
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.name = objname
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if norm_num_groups is not None:
            self.group_norm = SimNN.GroupNorm(f"{self.name}_grpnrm", num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None # type: ignore[assignment]

        if spatial_norm_dim is not None:
            raise NotImplementedError('spatial norm not supported!')
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        else:
            raise NotImplementedError('qk_norm not None is not supported yet!')

        # cross_attention_norm is assumed to be None
        self.norm_cross = None
        self.to_q = ttsimF.Linear(f"{self.name}_linearop1", query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            self.to_k = ttsimF.Linear(f"{self.name}_linearop2", self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = ttsimF.Linear(f"{self.name}_linearop3", self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        self.add_q_proj = None
        self.add_k_proj = None
        self.add_v_proj = None

        if not self.pre_only:
            linop1 = ttsimF.Linear(f"{self.name}_linoper1", self.inner_dim, self.out_dim, bias=out_bias, module=self)
            self._op_hndls[linop1.name] = linop1
            self.to_out = [linop1]

            dpoutop = ttsimF.Dropout(f"{self.name}_dropoutop", dropout, module=self)
            self._op_hndls[dpoutop.name] = dpoutop
            self.to_out.append(dpoutop)
        else:
            self.to_out = None  # type: ignore[assignment]
        
        self.to_add_out = None
        self.norm_added_q = None
        self.norm_added_k = None

        if processor is None:
            processor = (
                AttnProcessor2_0(self)
            )
        self.set_processor(processor)
        super().link_op2module()

    def set_processor(self, processor) -> None:
        self.processor = processor

    def __call__(
        self,
        hidden_states: SimNN.SimTensor,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        **cross_attention_kwargs,
    ) -> SimNN.SimTensor:
        
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                "cross_attention_kwargs {} are not expected by {} and will be ignored.",
                unused_kwargs, self.processor.__class__.__name__
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )


class AttnProcessor2_0(SimNN.Module):
    def __init__(self, parent_module):
        super().__init__()
        self.name = "AttnProcessor2_0"
        self.bmmop1 = SimNN.bmm(f'{self.name}_bmmop1')
        self.bmmop2 = SimNN.bmm(f'{self.name}_bmmop2')
        self.transpose_op = ttsimF.Transpose(f'{self.name}_transpose_op', perm=(0, 2, 1))
        self.transpose_op.set_module(parent_module)
        self.softmaxop = ttsimF.Softmax(f'{self.name}_softmaxop')
        self.softmaxop.set_module(parent_module)
        self.transpose_op2 = ttsimF.Transpose(f'{self.name}_transpose_op2', perm=(0, 2, 1, 3))
        self.transpose_op2.set_module(parent_module)
        self.transpose_op3 = ttsimF.Transpose(f'{self.name}_transpose_op3', perm=(0, 2, 1, 3))
        self.transpose_op3.set_module(parent_module)
        self.transpose_op4 = ttsimF.Transpose(f'{self.name}_transpose_op4', perm=(0, 2, 1, 3))
        self.transpose_op4.set_module(parent_module)
        self.transpose_op5 = ttsimF.Transpose(f'{self.name}_transpose_op5', perm=(0, 2, 1, 3))
        self.transpose_op5.set_module(parent_module)
        super().link_op2module()

    def scaled_dot_product_attention(
            self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    ):
        attn_scores = self.bmmop1(query, key.transpose(-1, -2)) # * scale
        scale = ttsimF._from_shape(f'{self.name}_scale', attn_scores.shape)
        attn_scores = attn_scores * scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_probs = self.softmaxop(attn_scores)
        attn_output = self.bmmop2(attn_probs, value)
        return attn_output

    def __call__(
        self,
        attn: Attention,
        hidden_states: SimNN.SimTensor,
        encoder_hidden_states: Optional[SimNN.SimTensor] = None,
        attention_mask: Optional[SimNN.SimTensor] = None,
        temb: Optional[SimNN.SimTensor] = None,
        *args,
        **kwargs,
    ) -> SimNN.SimTensor:

        residual = hidden_states

        # input_ndim = hidden_states.ndim
        input_ndim = len(hidden_states.shape) if isinstance(hidden_states, SimNN.SimTensor) else hidden_states.ndim
        
        if input_ndim == 4:
            raise NotImplementedError('not implemented! only works for 3 dims!')

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            raise NotImplementedError('not implemented! attention mask not supported!')

        if attn.group_norm is not None:
            raise NotImplementedError('not implemented! group norm not supported!')

        query = attn.to_q(hidden_states)   # type: ignore[unreachable]
        self._tensors[query.name] = query

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        query = self.transpose_op3(query)

        key = key.view(batch_size, -1, attn.heads, head_dim)
        key = self.transpose_op4(key)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        value = self.transpose_op5(value)

        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = self.transpose_op2(hidden_states)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim) # type: ignore[attr-defined]

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        self._tensors[hidden_states.name] = hidden_states
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states.set_module(self)

        if attn.residual_connection:
            hidden_states = hidden_states + residual # type: ignore[operator]

        attn_res_factor = ttsimF._from_shape(f'{self.name}_attn_res_factor', hidden_states.shape)
        hidden_states.set_module(self)
        hidden_states = hidden_states / attn_res_factor # type: ignore[operator]

        return hidden_states


ADDED_KV_ATTENTION_PROCESSORS = (
)

CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor2_0,
)

AttentionProcessor = Union[
    AttnProcessor2_0,
]
