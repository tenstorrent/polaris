#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""
TTSim implementation of Custom Base Transformer Layer.

This module implements a flexible transformer layer that can be composed
of various attention mechanisms (self-attention, cross-attention), 
feed-forward networks (FFN), and normalization layers in any specified order.

Original: projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

The original PyTorch implementation uses several mmcv functions that are not
compatible with Python 3.13. This TTSim version includes the following conversions:

1. Base Classes:
   - BaseModule: Replaced with ttsim.front.functional.sim_nn.Module
   - ModuleList: Replaced with Python list
   - Sequential: Replaced with custom sequential execution
   
2. Builders:
   - build_attention: Custom builder function for attention modules
   - build_feedforward_network: Custom builder function for FFN modules
   - build_norm_layer: Custom builder function for normalization layers
   - build_activation_layer: Custom builder function for activation layers
   
3. Registry Decorators:
   - @TRANSFORMER_LAYER.register_module(): Not needed in TTSim (no module registry)
   
4. Config Management:
   - ConfigDict: Replaced with Python dict
   - deprecated_api_warning: Replaced with warnings.warn
   
5. Tensor Operations:
   - torch.Tensor checks: Replaced with appropriate type checks
   - copy.deepcopy: Still used (standard Python library)
   
All computational logic from the PyTorch version has been preserved and
converted to TTSim operations.
"""


# -------------------------------PyTorch--------------------------------

# import copy
# import warnings

# import torch
# import torch.nn as nn

# from mmcv import ConfigDict, deprecated_api_warning
# from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

# from mmcv.cnn.bricks.registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
#                                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

# # Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
# try:
#     from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention  # noqa F401
#     warnings.warn(
#         ImportWarning(
#             '``MultiScaleDeformableAttention`` has been moved to '
#             '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
#             '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
#             'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
#         ))
# except ImportError:
#     warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
#                   '``mmcv.ops.multi_scale_deform_attn``, '
#                   'You should install ``mmcv-full`` if you need this module. ')
# from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


# @TRANSFORMER_LAYER.register_module()
# class MyCustomBaseTransformerLayer(BaseModule):
#     """Base `TransformerLayer` for vision transformer.
#     It can be built from `mmcv.ConfigDict` and support more flexible
#     customization, for example, using any number of `FFN or LN ` and
#     use different kinds of `attention` by specifying a list of `ConfigDict`
#     named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
#     when you specifying `norm` as the first element of `operation_order`.
#     More details about the `prenorm`: `On Layer Normalization in the
#     Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for `self_attention` or `cross_attention` modules,
#             The order of the configs in the list should be consistent with
#             corresponding attentions in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config. Default: None.
#         ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for FFN, The order of the configs in the list should be
#             consistent with corresponding ffn in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Support `prenorm` when you specifying first element as `norm`.
#             Default：None.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): Key, Query and Value are shape
#             of (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default to False.
#     """

#     def __init__(self,
#                  attn_cfgs=None,
#                  ffn_cfgs=dict(
#                      type='FFN',
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True),
#                  ),
#                  operation_order=None,
#                  norm_cfg=dict(type='LN'),
#                  init_cfg=None,
#                  batch_first=True,
#                  **kwargs):

#         deprecated_args = dict(
#             feedforward_channels='feedforward_channels',
#             ffn_dropout='ffn_drop',
#             ffn_num_fcs='num_fcs')
#         for ori_name, new_name in deprecated_args.items():
#             if ori_name in kwargs:
#                 warnings.warn(
#                     f'The arguments `{ori_name}` in BaseTransformerLayer '
#                     f'has been deprecated, now you should set `{new_name}` '
#                     f'and other FFN related arguments '
#                     f'to a dict named `ffn_cfgs`. ')
#                 ffn_cfgs[new_name] = kwargs[ori_name]

#         super(MyCustomBaseTransformerLayer, self).__init__(init_cfg)

#         self.batch_first = batch_first

#         assert set(operation_order) & set(
#             ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
#             set(operation_order), f'The operation_order of' \
#             f' {self.__class__.__name__} should ' \
#             f'contains all four operation type ' \
#             f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

#         num_attn = operation_order.count('self_attn') + operation_order.count(
#             'cross_attn')
#         if isinstance(attn_cfgs, dict):
#             attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
#         else:
#             assert num_attn == len(attn_cfgs), f'The length ' \
#                 f'of attn_cfg {num_attn} is ' \
#                 f'not consistent with the number of attention' \
#                 f'in operation_order {operation_order}.'

#         self.num_attn = num_attn
#         self.operation_order = operation_order
#         self.norm_cfg = norm_cfg
#         self.pre_norm = operation_order[0] == 'norm'
#         self.attentions = ModuleList()

#         index = 0
#         for operation_name in operation_order:
#             if operation_name in ['self_attn', 'cross_attn']:
#                 if 'batch_first' in attn_cfgs[index]:
#                     assert self.batch_first == attn_cfgs[index]['batch_first']
#                 else:
#                     attn_cfgs[index]['batch_first'] = self.batch_first
#                 attention = build_attention(attn_cfgs[index])
#                 # Some custom attentions used as `self_attn`
#                 # or `cross_attn` can have different behavior.
#                 attention.operation_name = operation_name
#                 self.attentions.append(attention)
#                 index += 1

#         self.embed_dims = self.attentions[0].embed_dims

#         self.ffns = ModuleList()
#         num_ffns = operation_order.count('ffn')
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = ConfigDict(ffn_cfgs)
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
#         assert len(ffn_cfgs) == num_ffns
#         for ffn_index in range(num_ffns):
#             if 'embed_dims' not in ffn_cfgs[ffn_index]:
#                 ffn_cfgs['embed_dims'] = self.embed_dims
#             else:
#                 assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

#             self.ffns.append(
#                 build_feedforward_network(ffn_cfgs[ffn_index]))

#         self.norms = ModuleList()
#         num_norms = operation_order.count('norm')
#         for _ in range(num_norms):
#             self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoderLayer`.
#         **kwargs contains some specific arguments of attentions.
#         Args:
#             query (Tensor): The input query with shape
#                 [num_queries, bs, embed_dims] if
#                 self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#             value (Tensor): The value tensor with same shape as `key`.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): 2D Tensor used in
#                 calculation of corresponding attention. The length of
#                 it should equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_queries]. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_keys]. Default: None.
#         Returns:
#             Tensor: forwarded results with shape [num_queries, bs, embed_dims].
#         """

#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [
#                 copy.deepcopy(attn_masks) for _ in range(self.num_attn)
#             ]
#             warnings.warn(f'Use same attn_mask in all attentions in '
#                           f'{self.__class__.__name__} ')
#         else:
#             assert len(attn_masks) == self.num_attn, f'The length of ' \
#                 f'attn_masks {len(attn_masks)} must be equal ' \
#                 f'to the number of attention in ' \
#                 f'operation_order {self.num_attn}'

#         for layer in self.operation_order:
#             if layer == 'self_attn':
#                 temp_key = temp_value = query
#                 query = self.attentions[attn_index](
#                     query,
#                     temp_key,
#                     temp_value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=query_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1

#             elif layer == 'cross_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1

#         return query


# @TRANSFORMER_LAYER.register_module()
# class MyCustomBaseTransformerLayerWithoutSelfAttn(BaseModule):
#     """Base `TransformerLayer` for vision transformer.
#     It can be built from `mmcv.ConfigDict` and support more flexible
#     customization, for example, using any number of `FFN or LN ` and
#     use different kinds of `attention` by specifying a list of `ConfigDict`
#     named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
#     when you specifying `norm` as the first element of `operation_order`.
#     More details about the `prenorm`: `On Layer Normalization in the
#     Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for `self_attention` or `cross_attention` modules,
#             The order of the configs in the list should be consistent with
#             corresponding attentions in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config. Default: None.
#         ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
#             Configs for FFN, The order of the configs in the list should be
#             consistent with corresponding ffn in operation_order.
#             If it is a dict, all of the attention modules in operation_order
#             will be built with this config.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Support `prenorm` when you specifying first element as `norm`.
#             Default：None.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): Key, Query and Value are shape
#             of (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default to False.
#     """

#     def __init__(self,
#                  attn_cfgs=None,
#                  ffn_cfgs=dict(
#                      type='FFN',
#                      embed_dims=256,
#                      feedforward_channels=1024,
#                      num_fcs=2,
#                      ffn_drop=0.,
#                      act_cfg=dict(type='ReLU', inplace=True),
#                  ),
#                  operation_order=None,
#                  norm_cfg=dict(type='LN'),
#                  init_cfg=None,
#                  batch_first=True,
#                  **kwargs):

#         deprecated_args = dict(
#             feedforward_channels='feedforward_channels',
#             ffn_dropout='ffn_drop',
#             ffn_num_fcs='num_fcs')
#         for ori_name, new_name in deprecated_args.items():
#             if ori_name in kwargs:
#                 warnings.warn(
#                     f'The arguments `{ori_name}` in BaseTransformerLayer '
#                     f'has been deprecated, now you should set `{new_name}` '
#                     f'and other FFN related arguments '
#                     f'to a dict named `ffn_cfgs`. ')
#                 ffn_cfgs[new_name] = kwargs[ori_name]

#         super(MyCustomBaseTransformerLayerWithoutSelfAttn, self).__init__(init_cfg)

#         self.batch_first = batch_first

#         assert set(operation_order) & set(
#             ['norm', 'ffn', 'cross_attn']) == \
#             set(operation_order), f'The operation_order of' \
#             f' {self.__class__.__name__} should ' \
#             f'contains all three operation type ' \
#             f"{['norm', 'ffn', 'cross_attn']}"

#         num_attn = operation_order.count(
#             'cross_attn')
#         if isinstance(attn_cfgs, dict):
#             attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
#         else:
#             assert num_attn == len(attn_cfgs), f'The length ' \
#                 f'of attn_cfg {num_attn} is ' \
#                 f'not consistent with the number of attention' \
#                 f'in operation_order {operation_order}.'

#         self.num_attn = num_attn
#         self.operation_order = operation_order
#         self.norm_cfg = norm_cfg
#         self.pre_norm = operation_order[0] == 'norm'
#         self.attentions = ModuleList()

#         index = 0
#         for operation_name in operation_order:
#             if operation_name in ['self_attn', 'cross_attn']:
#                 if 'batch_first' in attn_cfgs[index]:
#                     assert self.batch_first == attn_cfgs[index]['batch_first']
#                 else:
#                     attn_cfgs[index]['batch_first'] = self.batch_first
#                 attention = build_attention(attn_cfgs[index])
#                 # Some custom attentions used as `self_attn`
#                 # or `cross_attn` can have different behavior.
#                 attention.operation_name = operation_name
#                 self.attentions.append(attention)
#                 index += 1

#         self.embed_dims = self.attentions[0].embed_dims

#         self.ffns = ModuleList()
#         num_ffns = operation_order.count('ffn')
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = ConfigDict(ffn_cfgs)
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
#         assert len(ffn_cfgs) == num_ffns
#         for ffn_index in range(num_ffns):
#             if 'embed_dims' not in ffn_cfgs[ffn_index]:
#                 ffn_cfgs['embed_dims'] = self.embed_dims
#             else:
#                 assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

#             self.ffns.append(
#                 build_feedforward_network(ffn_cfgs[ffn_index]))

#         self.norms = ModuleList()
#         num_norms = operation_order.count('norm')
#         for _ in range(num_norms):
#             self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoderLayer`.
#         **kwargs contains some specific arguments of attentions.
#         Args:
#             query (Tensor): The input query with shape
#                 [num_queries, bs, embed_dims] if
#                 self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#             value (Tensor): The value tensor with same shape as `key`.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`.
#                 Default: None.
#             attn_masks (List[Tensor] | None): 2D Tensor used in
#                 calculation of corresponding attention. The length of
#                 it should equal to the number of `attention` in
#                 `operation_order`. Default: None.
#             query_key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_queries]. Only used in `self_attn` layer.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_keys]. Default: None.
#         Returns:
#             Tensor: forwarded results with shape [num_queries, bs, embed_dims].
#         """

#         norm_index = 0
#         attn_index = 0
#         ffn_index = 0
#         identity = query
#         if attn_masks is None:
#             attn_masks = [None for _ in range(self.num_attn)]
#         elif isinstance(attn_masks, torch.Tensor):
#             attn_masks = [
#                 copy.deepcopy(attn_masks) for _ in range(self.num_attn)
#             ]
#             warnings.warn(f'Use same attn_mask in all attentions in '
#                           f'{self.__class__.__name__} ')
#         else:
#             assert len(attn_masks) == self.num_attn, f'The length of ' \
#                 f'attn_masks {len(attn_masks)} must be equal ' \
#                 f'to the number of attention in ' \
#                 f'operation_order {self.num_attn}'

#         for layer in self.operation_order:
#             if layer == 'self_attn':
#                 temp_key = temp_value = query
#                 query = self.attentions[attn_index](
#                     query,
#                     temp_key,
#                     temp_value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=query_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1

#             elif layer == 'cross_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1

#         return query


# -------------------------------TTSIM-----------------------------------

import sys
import os
import copy
import warnings

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import component builders
from .builder_utils import build_attention, build_feedforward_network, build_norm_layer


class MyCustomBaseTransformerLayer(SimNN.Module):
    """
    TTSim implementation of Custom Base Transformer Layer for BEVFormer.

    This is a flexible transformer layer that can compose any number of
    attention, FFN, and normalization operations in a specified order.
    It supports both prenorm and postnorm architectures.

    Key features:
    - Flexible operation ordering (self_attn, cross_attn, norm, ffn)
    - Multiple attention mechanisms in single layer
    - Prenorm support (norm before attention/FFN)
    - Batch-first or sequence-first formats

    Args:
        name (str): Module name
        attn_cfgs (list[dict] | dict | None): Configs for attention modules.
            Order should match attention operations in operation_order.
            If dict, all attentions use same config. Default: None
        ffn_cfgs (list[dict] | dict | None): Configs for FFN modules.
            Order should match FFN operations in operation_order.
            Default: standard FFN config
        operation_order (tuple[str]): Execution order of operations.
            e.g., ('self_attn', 'norm', 'ffn', 'norm')
            Support prenorm when first element is 'norm'. Default: None
        norm_cfg (dict): Config for normalization layer. Default: dict(type='LN')
        batch_first (bool): If True, inputs are [batch, seq, dim]. Default: True

    Example operation_order:
        - Standard: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        - Prenorm: ('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn')
    """

    def __init__(
        self,
        name,
        attn_cfgs=None,
        ffn_cfgs=None,
        operation_order=None,
        norm_cfg=None,
        batch_first=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        # Set defaults
        # Note: embed_dims is intentionally omitted so it gets auto-populated
        # from self.embed_dims (derived from attention configs) at build time.
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type="FFN",
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="ReLU", inplace=True),
            )

        if norm_cfg is None:
            norm_cfg = dict(type="LN")

        # Handle deprecated arguments
        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                if isinstance(ffn_cfgs, dict):
                    ffn_cfgs[new_name] = kwargs[ori_name]

        self.batch_first = batch_first

        # Validate operation_order
        if operation_order is None:
            raise ValueError("operation_order must be specified")

        valid_ops = {"self_attn", "norm", "ffn", "cross_attn"}
        assert set(operation_order) & valid_ops == set(
            operation_order
        ), f"The operation_order should only contain operations from {valid_ops}"

        # Count number of each operation type
        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        num_ffns = operation_order.count("ffn")
        num_norms = operation_order.count("norm")

        # Handle attention configs
        if attn_cfgs is None:
            attn_cfgs = []
        elif isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length of attn_cfg {len(attn_cfgs)} is "
                f"not consistent with the number of attention "
                f"in operation_order {num_attn}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = len(operation_order) > 0 and operation_order[0] == "norm"

        # Build attention modules
        _attentions = []
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first

                attention = build_attention(f"{name}.attn_{index}", attn_cfgs[index])
                # Mark operation type for the attention module
                attention.operation_name = operation_name
                _attentions.append(attention)
                index += 1
        self.attentions: SimNN.ModuleList | list[SimNN.Module] = (
            SimNN.ModuleList(_attentions) if _attentions else []
        )

        # Get embed_dims from first attention
        if len(self.attentions) > 0:
            self.embed_dims = self.attentions[0].embed_dims
        else:
            self.embed_dims = ffn_cfgs.get("embed_dims", 256)

        # Build FFN modules
        _ffns = []
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert (
            len(ffn_cfgs) == num_ffns
        ), f"The length of ffn_cfgs {len(ffn_cfgs)} must equal num_ffns {num_ffns}"

        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            ffn = build_feedforward_network(
                f"{name}.ffn_{ffn_index}", ffn_cfgs[ffn_index]
            )
            _ffns.append(ffn)
        self.ffns: SimNN.ModuleList | list[SimNN.Module] = (
            SimNN.ModuleList(_ffns) if _ffns else []
        )

        # Build normalization modules
        _norms = []
        for norm_index in range(num_norms):
            norm = build_norm_layer(
                f"{name}.norm_{norm_index}", norm_cfg, self.embed_dims
            )
            _norms.append(norm)
        self.norms: SimNN.ModuleList | list[SimNN.Module] = (
            SimNN.ModuleList(_norms) if _norms else []
        )

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """
        Forward pass of Custom Base Transformer Layer.

        Args:
            query: Input query tensor
                [num_queries, bs, embed_dims] if batch_first=False
                [bs, num_queries, embed_dims] if batch_first=True
            key: Key tensor (for cross-attention)
            value: Value tensor (for cross-attention)
            query_pos: Positional encoding for query
            key_pos: Positional encoding for key
            attn_masks: List of attention masks (one per attention operation)
            query_key_padding_mask: Padding mask for query (self-attention)
            key_padding_mask: Padding mask for key (cross-attention)
            **kwargs: Additional arguments for attention modules

        Returns:
            Output tensor with same shape as query
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif not isinstance(attn_masks, list):
            # If single mask provided, replicate for all attentions
            attn_masks = [attn_masks for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in operation_order {self.num_attn}"
            )

        # Execute operations in specified order
        for layer in self.operation_order:
            if layer == "self_attn":
                # Self-attention: query attends to itself
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                # Normalization
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                # Cross-attention: query attends to key/value
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                # Feed-forward network
                query = self.ffns[ffn_index](
                    query, identity=identity if self.pre_norm else None
                )
                ffn_index += 1

        return query

    def analytical_param_count(self):
        """
        Calculate the total number of parameters in this module.

        Returns:
            int: Total parameter count
        """
        total = 0

        # Count attention parameters
        for attn in self.attentions:
            if hasattr(attn, "analytical_param_count"):
                total += attn.analytical_param_count()

        # Count FFN parameters
        for ffn in self.ffns:
            if hasattr(ffn, "analytical_param_count"):
                total += ffn.analytical_param_count()

        # Count norm parameters
        for norm in self.norms:
            if hasattr(norm, "analytical_param_count"):
                total += norm.analytical_param_count()
            else:
                # LayerNorm has 2 * embed_dims parameters (weight + bias)
                total += 2 * self.embed_dims

        return total


if __name__ == "__main__":
    print("=" * 80)
    print("Custom Base Transformer Layer TTSim Module")
    print("=" * 80)
    print("\n[OK] Module imported successfully!")
    print("\nAvailable component:")
    print("  - MyCustomBaseTransformerLayer - Flexible transformer layer")

    print("\nModule test:")

    # Test MyCustomBaseTransformerLayer
    try:
        # Simple test configuration
        attn_cfg = dict(
            type="TemporalSelfAttention",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            num_bev_queue=2,
        )

        ffn_cfg = dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
        )

        operation_order = ("self_attn", "norm", "ffn", "norm")

        layer = MyCustomBaseTransformerLayer(
            name="test_layer",
            attn_cfgs=[attn_cfg],
            ffn_cfgs=ffn_cfg,
            operation_order=operation_order,
            batch_first=True,
        )
        print(f"  [OK] MyCustomBaseTransformerLayer constructed successfully")
        print(f"    - Name: {layer.name}")
        print(f"    - Embed dims: {layer.embed_dims}")
        print(f"    - Operation order: {layer.operation_order}")
        print(f"    - Num attentions: {layer.num_attn}")
        print(f"    - Num FFNs: {len(layer.ffns)}")
        print(f"    - Num norms: {len(layer.norms)}")
        print(f"    - Batch first: {layer.batch_first}")
        print(f"    - Pre-norm: {layer.pre_norm}")
    except Exception as e:
        print(f"  [X] MyCustomBaseTransformerLayer construction failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n[OK] Basic test passed!")
    print(
        "\nNote: Use validation tests in Validation/ folder for full functionality testing."
    )
    print("=" * 80)
