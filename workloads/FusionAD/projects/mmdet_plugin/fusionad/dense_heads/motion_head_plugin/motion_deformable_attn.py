
# =============================================================================
# ORIGINAL TORCH CODE (from FusionAD)
# Source: FusionAD/projects/mmdet3d_plugin/fusionad/dense_heads/motion_head_plugin/motion_deformable_attn.py
# =============================================================================
# #---------------------------------------------------------------------------------#
# # UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# # Source code: https://github.com/OpenDriveLab/UniAD                              #
# # Copyright (c) OpenDriveLab. All rights reserved.                                #
# #---------------------------------------------------------------------------------#
#
# # Modifications:
# # - Modified by FusionAD on 2023.5
# # - Added extended support from FusionAD (https://arxiv.org/abs/2308.01006)
#
# import copy
# import warnings
# import torch
# import math
# import torch.nn as nn
#
# from einops import rearrange, repeat
# from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
# from mmcv.cnn import xavier_init, constant_init
# from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER
# from mmcv.cnn.bricks.transformer import build_attention, build_feedforward_network, build_norm_layer
# from mmcv.cnn.bricks.drop import build_dropout
# from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
# from mmcv.utils import ConfigDict, deprecated_api_warning
# from projects.mmdet3d_plugin.fusionad.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
#
#
# @TRANSFORMER_LAYER.register_module()
# class MotionTransformerAttentionLayer(BaseModule):
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
#
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
#                  batch_first=False,
#                  **kwargs):
#
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
#                     f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
#                 ffn_cfgs[new_name] = kwargs[ori_name]
#
#         super().__init__(init_cfg)
#
#         self.batch_first = batch_first
#
#         num_attn = operation_order.count('self_attn') + operation_order.count(
#             'cross_attn')
#         if isinstance(attn_cfgs, dict):
#             attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
#         else:
#             assert num_attn == len(attn_cfgs), f'The length ' \
#                 f'of attn_cfg {num_attn} is ' \
#                 f'not consistent with the number of attention' \
#                 f'in operation_order {operation_order}.'
#
#         self.num_attn = num_attn
#         self.operation_order = operation_order
#         self.norm_cfg = norm_cfg
#         self.pre_norm = operation_order[0] == 'norm'
#         self.attentions = ModuleList()
#
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
#
#         self.embed_dims = self.attentions[0].embed_dims
#
#         self.ffns = ModuleList()
#         num_ffns = operation_order.count('ffn')
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = ConfigDict(ffn_cfgs)
#         if isinstance(ffn_cfgs, dict):
#             ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
#         assert len(ffn_cfgs) == num_ffns
#         for ffn_index in range(num_ffns):
#             if 'embed_dims' not in ffn_cfgs[ffn_index]:
#                 ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
#             else:
#                 assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
#             self.ffns.append(
#                 build_feedforward_network(ffn_cfgs[ffn_index],
#                                           dict(type='FFN')))
#
#         self.norms = ModuleList()
#         num_norms = operation_order.count('norm')
#         for _ in range(num_norms):
#             self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
#
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
#
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
#                         f'attn_masks {len(attn_masks)} must be equal ' \
#                         f'to the number of attention in ' \
#                         f'operation_order {self.num_attn}'
#
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
#
#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1
#
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
#
#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1
#
#         return query
#
# @ATTENTION.register_module()
# class MotionDeformableAttention(BaseModule):
#     """An attention module used in Deformable-Detr.
#
#     `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
#     <https://arxiv.org/pdf/2010.04159.pdf>`_.
#
#     Args:
#         embed_dims (int): The embedding dimension of Attention.
#             Default: 256.
#         num_heads (int): Parallel attention heads. Default: 64.
#         num_levels (int): The number of feature map used in
#             Attention. Default: 4.
#         num_points (int): The number of sampling points for
#             each query in each head. Default: 4.
#         im2col_step (int): The step used in image_to_column.
#             Default: 64.
#         dropout (float): A Dropout layer on `inp_identity`.
#             Default: 0.1.
#         batch_first (bool): Key, Query and Value are shape of
#             (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default to False.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: None.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#     """
#
#     def __init__(self,
#                  embed_dims=256,
#                  num_heads=8,
#                  num_levels=4,
#                  num_points=4,
#                  num_steps=1,
#                  sample_index=-1,
#                  im2col_step=64,
#                  dropout=0.1,
#                  bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
#                  voxel_size=[0.2, 0.2, 8],
#                  batch_first=True,
#                  norm_cfg=None,
#                  init_cfg=None):
#         super().__init__(init_cfg)
#         if embed_dims % num_heads != 0:
#             raise ValueError(f'embed_dims must be divisible by num_heads, '
#                              f'but got {embed_dims} and {num_heads}')
#         dim_per_head = embed_dims // num_heads
#         self.norm_cfg = norm_cfg
#         self.dropout = nn.Dropout(dropout)
#         self.batch_first = batch_first
#         self.fp16_enabled = False
#         self.bev_range = bev_range
#
#         # you'd better set dim_per_head to a power of 2
#         # which is more efficient in the CUDA implementation
#         def _is_power_of_2(n):
#             if (not isinstance(n, int)) or (n < 0):
#                 raise ValueError(
#                     'invalid input for _is_power_of_2: {} (type: {})'.format(
#                         n, type(n)))
#             return (n & (n - 1) == 0) and n != 0
#
#         if not _is_power_of_2(dim_per_head):
#             warnings.warn(
#                 "You'd better set embed_dims in "
#                 'MultiScaleDeformAttention to make '
#                 'the dimension of each attention head a power of 2 '
#                 'which is more efficient in our CUDA implementation.')
#
#         self.im2col_step = im2col_step
#         self.embed_dims = embed_dims
#         self.num_levels = num_levels
#         self.num_heads = num_heads
#         self.num_points = num_points
#         self.num_steps = num_steps
#         self.sample_index = sample_index
#         self.sampling_offsets = nn.Linear(
#             embed_dims, num_heads * num_steps * num_levels * num_points * 2)
#         self.attention_weights = nn.Linear(embed_dims,
#                                            num_heads * num_steps * num_levels * num_points)
#         self.value_proj = nn.Linear(embed_dims, embed_dims)
#         self.output_proj = Sequential(nn.Linear(num_steps*embed_dims, embed_dims),
#                                       nn.LayerNorm(embed_dims),
#                                       nn.ReLU(inplace=True)
#                                      )
#         self.init_weights()
#
#     def init_weights(self):
#         """Default initialization for Parameters of Module."""
#         constant_init(self.sampling_offsets, 0.)
#         thetas = torch.arange(
#             self.num_heads,
#             dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
#         grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
#         grid_init = (grid_init /
#                      grid_init.abs().max(-1, keepdim=True)[0]).view(
#             self.num_heads, 1, 1, 1,
#             2).repeat(1, self.num_steps, self.num_levels, self.num_points, 1)
#         for i in range(self.num_points):
#             grid_init[:, :, :, i, :] *= i + 1
#
#         self.sampling_offsets.bias.data = grid_init.view(-1)
#         constant_init(self.attention_weights, val=0., bias=0.)
#         xavier_init(self.value_proj, distribution='uniform', bias=0.)
#         xavier_init(self.output_proj, distribution='uniform', bias=0.)
#         self._is_init = True
#
#     @deprecated_api_warning({'residual': 'identity'},
#                             cls_name='MultiScaleDeformableAttention')
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_padding_mask=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 bbox_results=None,
#                 reference_trajs=None,
#                 flag='decoder',
#                 **kwargs):
#         """Forward Function of MultiScaleDeformAttention.
#
#         Args:
#             query (Tensor): Query of Transformer with shape
#                 (num_query, bs, embed_dims).
#             key (Tensor): The key tensor with shape
#                 `(num_key, bs, embed_dims)`.
#             value (Tensor): The value tensor with shape
#                 `(num_key, bs, embed_dims)`.
#             identity (Tensor): The tensor used for addition, with the
#                 same shape as `query`. Default None. If None,
#                 `query` will be used.
#             query_pos (Tensor): The positional encoding for `query`.
#                 Default: None.
#             key_pos (Tensor): The positional encoding for `key`. Default
#                 None.
#             reference_points (Tensor):  The normalized reference
#                 points with shape (bs, num_query, num_levels, 2),
#                 all elements is range in [0, 1], top-left (0,0),
#                 bottom-right (1, 1), including padding area.
#                 or (N, Length_{query}, num_levels, 4), add
#                 additional two dimensions is (w, h) to
#                 form reference boxes.
#             key_padding_mask (Tensor): ByteTensor for `query`, with
#                 shape [bs, num_key].
#             spatial_shapes (Tensor): Spatial shape of features in
#                 different levels. With shape (num_levels, 2),
#                 last dimension represents (h, w).
#             level_start_index (Tensor): The start index of each level.
#                 A tensor has shape ``(num_levels, )`` and can be represented
#                 as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
#
#         Returns:
#              Tensor: forwarded results with shape [num_query, bs, embed_dims].
#         """
#         bs, num_agent, num_mode, _ = query.shape
#         num_query = num_agent * num_mode
#         if value is None:
#             value = query
#         if identity is None:
#             identity = query
#         if query_pos is not None:
#             query = query + query_pos
#         query = torch.flatten(query, start_dim=1, end_dim=2)
#
#         value = value.permute(1, 0, 2)
#         bs, num_value, _ = value.shape
#         assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
#
#         value = self.value_proj(value)
#         if key_padding_mask is not None:
#             value = value.masked_fill(key_padding_mask[..., None], 0.0)
#         value = value.view(bs, num_value, self.num_heads, -1)
#         sampling_offsets = self.sampling_offsets(query).view(
#             bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points, 2)
#         attention_weights = self.attention_weights(query).view(
#             bs, num_query, self.num_heads, self.num_steps, self.num_levels * self.num_points)
#         attention_weights = attention_weights.softmax(-1)
#
#         attention_weights = attention_weights.view(bs, num_query,
#                                                    self.num_heads,
#                                                    self.num_steps,
#                                                    self.num_levels,
#                                                    self.num_points)
#         # bs, n_query, n_head, n_steps, N_level, N_points, 2
#         # BS NUM_AGENT NUM_MODE 12 NUM_LEVEL  2
#         if reference_trajs.shape[-1] == 2:
#             reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]
#             reference_trajs_ego = self.agent_coords_to_ego_coords(copy.deepcopy(reference_trajs), bbox_results).detach()
#             reference_trajs_ego = torch.flatten(reference_trajs_ego, start_dim=1, end_dim=2)
#             reference_trajs_ego = reference_trajs_ego[:, :, None, :, :, None, :]
#             reference_trajs_ego[..., 0] -= self.bev_range[0]
#             reference_trajs_ego[..., 1] -= self.bev_range[1]
#             reference_trajs_ego[..., 0] /= (self.bev_range[3] - self.bev_range[0])
#             reference_trajs_ego[..., 1] /= (self.bev_range[4] - self.bev_range[1])
#             offset_normalizer = torch.stack(
#                 [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
#             sampling_locations = reference_trajs_ego \
#                 + sampling_offsets \
#                 / offset_normalizer[None, None, None, None, :, None, :]
#
#             sampling_locations = rearrange(sampling_locations, 'bs nq nh ns nl np c -> bs nq ns nh nl np c') # permute([0,1,3,2,4,5,6])
#             attention_weights = rearrange(attention_weights, 'bs nq nh ns nl np -> bs nq ns nh nl np') #.permute([0,1,3,2,4,5])
#             sampling_locations = sampling_locations.reshape(bs, num_query*self.num_steps, self.num_heads, self.num_levels, self.num_points, 2)
#             attention_weights = attention_weights.reshape(bs, num_query*self.num_steps, self.num_heads, self.num_levels, self.num_points)
#
#         else:
#             raise ValueError(
#                 f'Last dim of reference_trajs must be'
#                 f' 2 or 4, but get {reference_trajs.shape[-1]} instead.')
#         if torch.cuda.is_available() and value.is_cuda:
#
#             # using fp16 deformable attention is unstable because it performs many sum operations
#             if value.dtype == torch.float16:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             else:
#                 MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
#             output = MultiScaleDeformableAttnFunction.apply(
#                 value, spatial_shapes, level_start_index, sampling_locations,
#                 attention_weights, self.im2col_step)
#         else:
#             output = multi_scale_deformable_attn_pytorch(
#                 value, spatial_shapes, sampling_locations, attention_weights)
#         output = output.view(bs, num_query, self.num_steps, -1)
#         output = torch.flatten(output, start_dim=2, end_dim=3)
#         output = self.output_proj(output)
#         output = output.view(bs, num_agent, num_mode, -1)
#         return self.dropout(output) + identity
#
#     def agent_coords_to_ego_coords(self, reference_trajs, bbox_results):
#         batch_size = len(bbox_results)
#         reference_trajs_ego = []
#         for i in range(batch_size):
#             boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
#             det_centers = boxes_3d.gravity_center.to(reference_trajs.device)
#             batch_reference_trajs = reference_trajs[i]
#             batch_reference_trajs += det_centers[:, None, None, None, :2]
#             reference_trajs_ego.append(batch_reference_trajs)
#         return torch.stack(reference_trajs_ego)
#
#     def rot_2d(self, yaw):
#         sy, cy = torch.sin(yaw), torch.cos(yaw)
#         out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2,0,1])
#         return out
#
# @ATTENTION.register_module()
# class CustomModeMultiheadAttention(BaseModule):
#     """A wrapper for ``torch.nn.MultiheadAttention``.
#     This module implements MultiheadAttention with identity connection,
#     and positional encoding  is also passed as input.
#     Args:
#         embed_dims (int): The embedding dimension.
#         num_heads (int): Parallel attention heads.
#         attn_drop (float): A Dropout layer on attn_output_weights.
#             Default: 0.0.
#         proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
#             Default: 0.0.
#         dropout_layer (obj:`ConfigDict`): The dropout_layer used
#             when adding the shortcut.
#         init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
#             Default: None.
#         batch_first (bool): When it is True,  Key, Query and Value are shape of
#             (batch, n, embed_dim), otherwise (n, batch, embed_dim).
#              Default to False.
#     """
#
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  attn_drop=0.,
#                  proj_drop=0.,
#                  dropout_layer=dict(type='Dropout', drop_prob=0.),
#                  init_cfg=None,
#                  **kwargs):
#         super().__init__(init_cfg)
#         if 'dropout' in kwargs:
#             warnings.warn(
#                 'The arguments `dropout` in MultiheadAttention '
#                 'has been deprecated, now you can separately '
#                 'set `attn_drop`(float), proj_drop(float), '
#                 'and `dropout_layer`(dict) ', DeprecationWarning)
#             attn_drop = kwargs['dropout']
#             dropout_layer['drop_prob'] = kwargs.pop('dropout')
#
#         self.embed_dims = embed_dims
#         self.num_heads = num_heads
#
#         self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
#
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.dropout_layer = build_dropout(
#             dropout_layer) if dropout_layer else nn.Identity()
#
#     @deprecated_api_warning({'residual': 'identity'},
#                             cls_name='MultiheadAttention')
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_mask=None,
#                 key_padding_mask=None,
#                 **kwargs):
#         """Forward function for `MultiheadAttention`.
#         **kwargs allow passing a more general data flow when combining
#         with other operations in `transformerlayer`.
#         Args:
#             query (Tensor): The input query with shape [num_queries, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#                 If None, the ``query`` will be used. Defaults to None.
#             value (Tensor): The value tensor with same shape as `key`.
#                 Same in `nn.MultiheadAttention.forward`. Defaults to None.
#                 If None, the `key` will be used.
#             identity (Tensor): This tensor, with the same shape as x,
#                 will be used for the identity link.
#                 If None, `x` will be used. Defaults to None.
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `x`. If not None, it will
#                 be added to `x` before forward function. Defaults to None.
#             key_pos (Tensor): The positional encoding for `key`, with the
#                 same shape as `key`. Defaults to None. If not None, it will
#                 be added to `key` before forward function. If None, and
#                 `query_pos` has the same shape as `key`, then `query_pos`
#                 will be used for `key_pos`. Defaults to None.
#             attn_mask (Tensor): ByteTensor mask with shape [num_queries,
#                 num_keys]. Same in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
#                 Defaults to None.
#         Returns:
#             Tensor: forwarded results with shape
#             [num_queries, bs, embed_dims]
#             if self.batch_first is False, else
#             [bs, num_queries embed_dims].
#         """
#         query_pos = query_pos.unsqueeze(1)
#         key_pos = key_pos.unsqueeze(1)
#         bs, n_agent, n_query, D = query.shape
#         if key is None:
#             key = query
#         if value is None:
#             value = key
#         if identity is None:
#             identity = query
#         if key_pos is None:
#             if query_pos is not None:
#                 # use query_pos if key_pos is not available
#                 if query_pos.shape == key.shape:
#                     key_pos = query_pos
#                 else:
#                     warnings.warn(f'position encoding of key is'
#                                   f'missing in {self.__class__.__name__}.')
#         if query_pos is not None:
#             query = query + query_pos
#         if key_pos is not None:
#             key = key + key_pos
#
#         # Because the dataflow('key', 'query', 'value') of
#         # ``torch.nn.MultiheadAttention`` is (num_query, batch,
#         # embed_dims), We should adjust the shape of dataflow from
#         # batch_first (batch, num_query, embed_dims) to num_query_first
#         # (num_query ,batch, embed_dims), and recover ``attn_output``
#         # from num_query_first to batch_first.
#         query = torch.flatten(query, start_dim=0, end_dim=1)
#         key = torch.flatten(key, start_dim=0, end_dim=1)
#         value = torch.flatten(value, start_dim=0, end_dim=1)
#         identity = torch.flatten(identity, start_dim=0, end_dim=1)
#
#         query = query.transpose(0, 1)
#         key = key.transpose(0, 1)
#         value = value.transpose(0, 1)
#
#         out = self.attn(
#             query=query,
#             key=key,
#             value=value,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask)[0]
#
#         out = out.transpose(0, 1)
#         out = identity + self.dropout_layer(self.proj_drop(out))
#
#         return out.view(bs, n_agent, n_query, D)
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of motion-specific deformable attention modules for FusionAD.

Inference-only conversion.  Training-specific logic (dropout, gradient ops,
CUDA autocast) is omitted.

Original:
  projects/mmdet3d_plugin/fusionad/dense_heads/motion_head_plugin/motion_deformable_attn.py

Classes:
  - MotionDeformableAttention       : Deformable attention on BEV for motion prediction.
  - CustomModeMultiheadAttention    : Standard MHA wrapper for (B, A, P, D) shaped inputs.
  - MotionTransformerAttentionLayer : Flexible transformer layer composing attention + FFN + norms.
"""

import sys
import os
from loguru import logger
import copy
import warnings
import math

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add motion_head_plugin directory
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add dense_heads directory
dense_heads_dir = os.path.abspath(os.path.join(current_dir, '..'))
if dense_heads_dir not in sys.path:
    sys.path.insert(0, dense_heads_dir)

# Add fusionad directory
fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

# Add polaris root for ttsim
polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.builder_utils import (
    build_attention,
    build_feedforward_network,
    build_norm_layer,
    LayerNorm,
    FFN,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multihead_attention import (
    MultiheadAttention,
)
from workloads.FusionAD.projects.mmdet_plugin.fusionad.modules.multi_scale_deformable_attn_function import (
    multi_scale_deformable_attn_ttsim,
)


# ======================================================================
# MotionDeformableAttention
# ======================================================================

class MotionDeformableAttention(SimNN.Module):
    """
    Deformable attention on BEV features for motion prediction.

    Each agent's mode query generates sampling offsets and attention weights,
    which are used to sample from BEV feature maps via multi-scale deformable
    attention.  Reference trajectories in agent-local coordinates are converted
    to ego (BEV) coordinates before sampling.

    Key differences from the generic MultiScaleDeformableAttention:
      - Input query is (B, A, P, D) — per-agent, per-mode.
      - Reference points come from predicted trajectories (reference_trajs).
      - Agent-local → ego coordinate conversion using bbox_results.
      - num_steps dimension for temporal multi-step sampling.
      - output_proj is Sequential(Linear → LayerNorm → ReLU).

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_levels (int): Number of BEV feature levels. Default: 1.
        num_points (int): Number of sampling points per head per level. Default: 4.
        num_steps (int): Number of future time-steps for sampling. Default: 1.
        sample_index (int): Which time-step to sample from reference_trajs. Default: -1.
        bev_range (list): BEV range [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size (list): Voxel size [dx, dy, dz] (unused in forward).
    """

    def __init__(self, name, embed_dims=256, num_heads=8,
                 num_levels=1, num_points=4, num_steps=1,
                 sample_index=-1,
                 bev_range=None, voxel_size=None,
                 batch_first=True, **kwargs):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index
        self.batch_first = batch_first

        if bev_range is None:
            bev_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.bev_range = bev_range

        if embed_dims % num_heads != 0:
            raise ValueError(
                f'embed_dims must be divisible by num_heads, '
                f'but got {embed_dims} and {num_heads}')

        # Learnable projections
        self.sampling_offsets = SimNN.Linear(
            f'{name}.sampling_offsets',
            in_features=embed_dims,
            out_features=num_heads * num_steps * num_levels * num_points * 2)

        self.attention_weights = SimNN.Linear(
            f'{name}.attention_weights',
            in_features=embed_dims,
            out_features=num_heads * num_steps * num_levels * num_points)

        self.value_proj = SimNN.Linear(
            f'{name}.value_proj',
            in_features=embed_dims,
            out_features=embed_dims)

        # output_proj: Linear → LayerNorm → ReLU
        self.output_proj_linear = SimNN.Linear(
            f'{name}.output_proj_linear',
            in_features=num_steps * embed_dims,
            out_features=embed_dims)
        self.output_proj_ln = LayerNorm(f'{name}.output_proj_ln', embed_dims)
        self.output_proj_relu = F.Relu(f'{name}.output_proj_relu')

        # Reshape / transpose / arithmetic ops
        self.flatten_query = F.Reshape(f'{name}.flatten_query')
        self.permute_value = F.Transpose(f'{name}.permute_value', perm=[1, 0, 2])
        self.reshape_value = F.Reshape(f'{name}.reshape_value')
        self.reshape_offsets = F.Reshape(f'{name}.reshape_offsets')
        self.reshape_attn_w = F.Reshape(f'{name}.reshape_attn_w')
        self.softmax_attn = F.Softmax(f'{name}.softmax_attn', axis=-1)
        self.reshape_attn_w2 = F.Reshape(f'{name}.reshape_attn_w2')

        # Reference trajectory processing ops
        self.sub_bev_x = F.Sub(f'{name}.sub_bev_x')
        self.sub_bev_y = F.Sub(f'{name}.sub_bev_y')
        self.div_range_x = F.Div(f'{name}.div_range_x')
        self.div_range_y = F.Div(f'{name}.div_range_y')
        self.div_normalizer = F.Div(f'{name}.div_normalizer')
        self.add_ref_offset = F.Add(f'{name}.add_ref_offset')

        # Rearrange / reshape for final deformable attn
        self.rearrange_locs = F.Transpose(f'{name}.rearrange_locs',
                                          perm=[0, 1, 3, 2, 4, 5, 6])
        self.rearrange_attn = F.Transpose(f'{name}.rearrange_attn',
                                          perm=[0, 1, 3, 2, 4, 5])
        self.reshape_locs_final = F.Reshape(f'{name}.reshape_locs_final')
        self.reshape_attn_final = F.Reshape(f'{name}.reshape_attn_final')

        # Output reshape
        self.reshape_output_steps = F.Reshape(f'{name}.reshape_output_steps')
        self.reshape_output_flat = F.Reshape(f'{name}.reshape_output_flat')
        self.reshape_output_back = F.Reshape(f'{name}.reshape_output_back')

        # Residual add (identity + output)
        self.residual_add = F.Add(f'{name}.residual_add')

        super().link_op2module()

    def _r(self, obj):
        """Register a dynamically-created SimTensor or op into the graph."""
        from ttsim.ops import SimTensor
        if isinstance(obj, SimTensor):
            self._tensors[obj.name] = obj
        elif isinstance(obj, (F.SimOpHandle, F.SplitOpHandle,
                              F.VariadicInputOpHandle,
                              F.MultiOutputSimOpHandle)):
            self._op_hndls[obj.name] = obj
            obj.set_module(self)
        return obj

    def __call__(self, query, key=None, value=None, identity=None,
                 query_pos=None, key_padding_mask=None,
                 spatial_shapes=None, level_start_index=None,
                 bbox_results=None, reference_trajs=None,
                 flag='decoder', **kwargs):
        """
        Forward pass.

        Args:
            query: SimTensor (B, A, P, D) — per-agent mode queries.
            value: SimTensor (num_bev, B, D) — BEV features (seq-first).
            identity: SimTensor — residual identity (defaults to query).
            query_pos: SimTensor (B, A, P, D) — optional positional encoding.
            spatial_shapes: numpy array (num_levels, 2) — (H, W) per level.
            level_start_index: numpy array (num_levels,) — start indices.
            bbox_results: list of bbox tuples for agent→ego coordinate conversion.
            reference_trajs: SimTensor (B, A, P, num_steps_full, num_levels, 2)
                             — predicted trajectories in agent-local coords.

        Returns:
            SimTensor (B, A, P, D) — updated query after BEV deformable attention + residual.
        """
        B, num_agent, num_mode, D = query.shape
        num_query = num_agent * num_mode

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            self._add_pos = self._r(F.Add(f'{self.name}._add_pos'))
            query = self._add_pos(query, query_pos)

        # Flatten query: (B, A, P, D) → (B, A*P, D)
        _flat_shape = self._r(F._from_data(
            f'{self.name}._flat_query_shape',
            np.array([B, num_query, D], dtype=np.int64), is_const=True))
        q = self.flatten_query(query, _flat_shape)

        # Value: (num_bev, B, D) → (B, num_bev, D)
        val = self.permute_value(value)
        num_value = val.shape[1]

        # Value projection → (B, num_bev, num_heads, D//num_heads)
        val = self.value_proj(val)
        _val_shape = self._r(F._from_data(
            f'{self.name}._val_shape',
            np.array([B, num_value, self.num_heads,
                      D // self.num_heads], dtype=np.int64), is_const=True))
        val = self.reshape_value(val, _val_shape)

        # Sampling offsets: (B, num_query, num_heads, num_steps, num_levels, num_points, 2)
        offsets = self.sampling_offsets(q)
        _off_shape = self._r(F._from_data(
            f'{self.name}._off_shape',
            np.array([B, num_query, self.num_heads, self.num_steps,
                      self.num_levels, self.num_points, 2], dtype=np.int64),
            is_const=True))
        offsets = self.reshape_offsets(offsets, _off_shape)

        # Attention weights: softmax over (num_levels * num_points)
        attn_w = self.attention_weights(q)
        _aw_shape = self._r(F._from_data(
            f'{self.name}._aw_shape',
            np.array([B, num_query, self.num_heads, self.num_steps,
                      self.num_levels * self.num_points], dtype=np.int64),
            is_const=True))
        attn_w = self.reshape_attn_w(attn_w, _aw_shape)
        attn_w = self.softmax_attn(attn_w)
        _aw2_shape = self._r(F._from_data(
            f'{self.name}._aw2_shape',
            np.array([B, num_query, self.num_heads, self.num_steps,
                      self.num_levels, self.num_points], dtype=np.int64),
            is_const=True))
        attn_w = self.reshape_attn_w2(attn_w, _aw2_shape)

        # ------------------------------------------------------------------
        # Reference trajectory → ego coordinates → normalized BEV [0,1]
        # ------------------------------------------------------------------
        # reference_trajs: (B, A, P, T, L, 2)  — agent-local offsets
        # Select sample_index time-step
        ref = self._select_sample_index(reference_trajs)
        # Convert agent-local to ego coords using bbox_results
        ref_ego = self._agent_to_ego(ref, bbox_results)
        # Flatten (B, A, P, ...) → (B, A*P, ...)
        ref_ego = self._flatten_ref(ref_ego, B, num_query)
        # Expand dims: (B, nq, 1, 1, 1, 2) for broadcasting with offsets
        ref_ego = self._expand_ref(ref_ego, B, num_query)
        # Normalize to [0, 1] in BEV range
        ref_ego = self._normalize_to_bev(ref_ego)

        # Compute offset_normalizer from spatial_shapes
        offset_norm = self._make_offset_normalizer(spatial_shapes)

        # sampling_locations = ref_ego + offsets / normalizer
        norm_offsets = self.div_normalizer(offsets, offset_norm)
        sampling_locations = self.add_ref_offset(ref_ego, norm_offsets)

        # Rearrange: (bs nq nh ns nl np c) → (bs nq ns nh nl np c)
        sampling_locations = self.rearrange_locs(sampling_locations)
        attn_w = self.rearrange_attn(attn_w)

        # Flatten num_steps into num_query: (bs, nq*ns, nh, nl, np, 2)
        _sl_shape = self._r(F._from_data(
            f'{self.name}._sl_shape',
            np.array([B, num_query * self.num_steps, self.num_heads,
                      self.num_levels, self.num_points, 2], dtype=np.int64),
            is_const=True))
        sampling_locations = self.reshape_locs_final(sampling_locations, _sl_shape)

        _aw_final_shape = self._r(F._from_data(
            f'{self.name}._aw_final_shape',
            np.array([B, num_query * self.num_steps, self.num_heads,
                      self.num_levels, self.num_points], dtype=np.int64),
            is_const=True))
        attn_w = self.reshape_attn_final(attn_w, _aw_final_shape)

        # Call multi-scale deformable attention
        if isinstance(spatial_shapes, np.ndarray):
            sp_list = [(int(spatial_shapes[i, 0]), int(spatial_shapes[i, 1]))
                       for i in range(self.num_levels)]
        else:
            sp_list = spatial_shapes

        output = multi_scale_deformable_attn_ttsim(
            f'{self.name}.msda',
            val, sp_list, sampling_locations, attn_w,
            parent_module=self)

        # output: (B, nq*ns, D) → (B, nq, ns, D) → (B, nq, ns*D)
        _step_shape = self._r(F._from_data(
            f'{self.name}._step_shape',
            np.array([B, num_query, self.num_steps, D], dtype=np.int64),
            is_const=True))
        output = self.reshape_output_steps(output, _step_shape)

        _flat_out_shape = self._r(F._from_data(
            f'{self.name}._flat_out_shape',
            np.array([B, num_query, self.num_steps * D], dtype=np.int64),
            is_const=True))
        output = self.reshape_output_flat(output, _flat_out_shape)

        # output_proj: Linear → LN → ReLU
        output = self.output_proj_linear(output)
        output = self.output_proj_ln(output)
        output = self.output_proj_relu(output)

        # Reshape back: (B, A*P, D) → (B, A, P, D)
        _back_shape = self._r(F._from_data(
            f'{self.name}._back_shape',
            np.array([B, num_agent, num_mode, D], dtype=np.int64),
            is_const=True))
        output = self.reshape_output_back(output, _back_shape)

        # Residual: output + identity (dropout omitted for inference)
        output = self.residual_add(output, identity)

        return output

    # ------------------------------------------------------------------
    # Helper methods for reference trajectory processing
    # ------------------------------------------------------------------

    def _select_sample_index(self, reference_trajs):
        """Select the sample_index-th time-step from reference_trajs.

        Input:  (B, A, P, T, L, 2)
        Output: (B, A, P, 1, L, 2) — single time-step selected.
        """
        # Use numpy indexing on the data level — this is a constant slice
        idx = self.sample_index
        _s = self._r(F._from_data(f'{self.name}.ref_slice.starts',
                          np.array([idx], dtype=np.int64), is_const=True))
        # For index -1, end=None means "up to the end" → encode as shape[3]
        if idx == -1:
            end_val = reference_trajs.shape[3]
        else:
            end_val = idx + 1
        _e = self._r(F._from_data(f'{self.name}.ref_slice.ends',
                          np.array([end_val], dtype=np.int64), is_const=True))
        _a = self._r(F._from_data(f'{self.name}.ref_slice.axes',
                          np.array([3], dtype=np.int64), is_const=True))
        _st = self._r(F._from_data(f'{self.name}.ref_slice.steps',
                           np.array([1], dtype=np.int64), is_const=True))

        B, A, P = reference_trajs.shape[0], reference_trajs.shape[1], reference_trajs.shape[2]
        L = reference_trajs.shape[4]
        _op = self._r(F.SliceF(f'{self.name}.ref_slice',
                       out_shape=[B, A, P, 1, L, 2]))
        return _op(reference_trajs, _s, _e, _a, _st)

    def _agent_to_ego(self, ref, bbox_results):
        """Convert agent-local reference trajectories to ego coordinates.

        Uses bbox_results to get agent center positions and adds them.

        Input ref: SimTensor (B, A, P, 1, L, 2) or data thereof.
        Returns: numpy array with ego coords applied (run at data level).
        """
        # This operation mixes numpy data (bbox centers) with tensor data.
        # For TTSim shape tracking we create a new tensor with the result.
        ref_data = ref.data if hasattr(ref, 'data') and ref.data is not None else None

        if ref_data is not None and bbox_results is not None:
            batch_size = ref_data.shape[0]
            centers_list = []
            for i in range(batch_size):
                boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
                if hasattr(boxes_3d, 'gravity_center'):
                    det_centers = boxes_3d.gravity_center.cpu().numpy()
                else:
                    det_centers = np.array(boxes_3d)
                n_agents = ref_data.shape[1]
                c = det_centers[:n_agents, :2].reshape(-1, 1, 1, 1, 2)
                centers_list.append(c)
            centers_np = np.stack(centers_list, axis=0).astype(np.float32)
        else:
            centers_np = np.zeros(list(ref.shape), dtype=np.float32)

        centers_t = self._r(F._from_data(f'{self.name}.ego_centers',
                                 centers_np, is_const=True))
        add_op = self._r(F.Add(f'{self.name}.ego_add'))
        return add_op(ref, centers_t)

    def _flatten_ref(self, ref_ego, B, num_query):
        """Flatten (B, A, P, 1, L, 2) → (B, A*P, 1, L, 2)."""
        shape = list(ref_ego.shape)
        _shape = self._r(F._from_data(
            f'{self.name}._ref_flat_shape',
            np.array([B, num_query] + shape[3:], dtype=np.int64), is_const=True))
        _op = self._r(F.Reshape(f'{self.name}.ref_flatten'))
        return _op(ref_ego, _shape)

    def _expand_ref(self, ref_ego, B, num_query):
        """Expand (B, nq, 1, L, 2) → (B, nq, 1, 1, L, 1, 2) for broadcasting.

        Mirrors PyTorch: reference_trajs_ego[:, :, None, :, :, None, :]
        Adds dims at position 2 (num_heads) and position 5 (num_points).
        """
        _ax1 = self._r(F._from_data(f'{self.name}._ref_unsq_ax1',
                            np.array([2], dtype=np.int64), is_const=True))
        _op1 = self._r(F.Unsqueeze(f'{self.name}.ref_unsq1'))
        out = _op1(ref_ego, _ax1)  # (B, nq, 1, 1, L, 2)

        _ax2 = self._r(F._from_data(f'{self.name}._ref_unsq_ax2',
                            np.array([5], dtype=np.int64), is_const=True))
        _op2 = self._r(F.Unsqueeze(f'{self.name}.ref_unsq2'))
        return _op2(out, _ax2)  # (B, nq, 1, 1, L, 1, 2)

    def _normalize_to_bev(self, ref_ego):
        """Normalize ego coordinates to [0, 1] within BEV range.

        ref_ego[..., 0] = (x - x_min) / (x_max - x_min)
        ref_ego[..., 1] = (y - y_min) / (y_max - y_min)
        """
        bev = self.bev_range
        # Broadcast constants: [..., 2] where channel 0=x, channel 1=y
        bev_min = np.array([bev[0], bev[1]], dtype=np.float32)
        bev_range = np.array([bev[3] - bev[0], bev[4] - bev[1]], dtype=np.float32)

        min_t = self._r(F._from_data(f'{self.name}.bev_min',
                                     bev_min, is_const=True))
        range_t = self._r(F._from_data(f'{self.name}.bev_range',
                                       bev_range, is_const=True))
        sub_op = self._r(F.Sub(f'{self.name}.ref_sub_min'))
        div_op = self._r(F.Div(f'{self.name}.ref_div_range'))
        return div_op(sub_op(ref_ego, min_t), range_t)

    def _make_offset_normalizer(self, spatial_shapes):
        """Create offset normalizer tensor: [[W, H]] from spatial_shapes.

        Reverse order (W, H) to match sampling offset convention.
        Shape: (1, 1, 1, 1, num_levels, 1, 2) for broadcasting.
        """
        if isinstance(spatial_shapes, np.ndarray):
            # spatial_shapes: (num_levels, 2) with [H, W]
            norm_data = spatial_shapes[:, ::-1].copy().astype(np.float32)
        else:
            norm_data = np.array([[s[1], s[0]] for s in spatial_shapes],
                                 dtype=np.float32)
        # Reshape: (num_levels, 2) → (1, 1, 1, 1, num_levels, 1, 2)
        norm_data = norm_data.reshape(1, 1, 1, 1, self.num_levels, 1, 2)
        return self._r(F._from_data(f'{self.name}.offset_normalizer',
                            norm_data, is_const=True))

    def analytical_param_count(self):
        """Total parameter count for MotionDeformableAttention."""
        D = self.embed_dims
        H = self.num_heads
        S = self.num_steps
        L = self.num_levels
        P = self.num_points
        count = 0
        # sampling_offsets: (D, H*S*L*P*2) + bias
        off_out = H * S * L * P * 2
        count += D * off_out + off_out
        # attention_weights: (D, H*S*L*P) + bias
        aw_out = H * S * L * P
        count += D * aw_out + aw_out
        # value_proj: (D, D) + bias
        count += D * D + D
        # output_proj_linear: (S*D, D) + bias
        count += S * D * D + D
        return count


# ======================================================================
# CustomModeMultiheadAttention
# ======================================================================

class CustomModeMultiheadAttention(SimNN.Module):
    """
    MHA wrapper for (B, A, P, D) shaped inputs used in motion head.

    Adds positional encodings via unsqueeze, flattens B*A to batch dim,
    transposes to (P, B*A, D) for seq-first MHA, then unflattens back.

    PyTorch equivalent:
        nn.MultiheadAttention(embed_dims, num_heads) with custom reshaping.

    Args:
        name (str): Module name.
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, name, embed_dims=256, num_heads=8, **kwargs):
        super().__init__()
        self.name = name
        self.embed_dims = embed_dims
        self.num_heads = num_heads

        # Standard MHA (batch_first=False — PyTorch convention for this module)
        self.attn = MultiheadAttention(
            f'{name}.attn', embed_dims=embed_dims, num_heads=num_heads,
            batch_first=False, bias=True)

        # Pos encoding unsqueeze + add
        self.unsq_qpos = F.Unsqueeze(f'{name}.unsq_qpos')
        self.unsq_kpos = F.Unsqueeze(f'{name}.unsq_kpos')
        self.add_qpos = F.Add(f'{name}.add_qpos')
        self.add_kpos = F.Add(f'{name}.add_kpos')

        # Flatten / unflatten / transpose ops
        self.flatten_q = F.Reshape(f'{name}.flatten_q')
        self.flatten_k = F.Reshape(f'{name}.flatten_k')
        self.flatten_v = F.Reshape(f'{name}.flatten_v')
        self.flatten_id = F.Reshape(f'{name}.flatten_id')
        self.trans_q = F.Transpose(f'{name}.trans_q', perm=[1, 0, 2])
        self.trans_k = F.Transpose(f'{name}.trans_k', perm=[1, 0, 2])
        self.trans_v = F.Transpose(f'{name}.trans_v', perm=[1, 0, 2])
        self.trans_out = F.Transpose(f'{name}.trans_out', perm=[1, 0, 2])
        self.reshape_back = F.Reshape(f'{name}.reshape_back')

        # Residual add (identity + dropout(proj_drop(out)))
        self.residual_add = F.Add(f'{name}.residual_add')

        super().link_op2module()

    def _r(self, obj):
        """Register a dynamically-created SimTensor or op into the graph."""
        from ttsim.ops import SimTensor
        if isinstance(obj, SimTensor):
            self._tensors[obj.name] = obj
        elif isinstance(obj, (F.SimOpHandle, F.SplitOpHandle,
                              F.VariadicInputOpHandle,
                              F.MultiOutputSimOpHandle)):
            self._op_hndls[obj.name] = obj
            obj.set_module(self)
        return obj

    def __call__(self, query, key=None, value=None, identity=None,
                 query_pos=None, key_pos=None,
                 attn_mask=None, key_padding_mask=None, **kwargs):
        """
        Forward pass.

        Args:
            query: SimTensor (B, A, P, D).
            key: SimTensor (B, A, P, D) or None (defaults to query).
            value: SimTensor (B, A, P, D) or None (defaults to key).
            identity: SimTensor for residual (defaults to query).
            query_pos: SimTensor (B, A, D) — broadcast-unsqueezed to (B, A, 1, D).
            key_pos: SimTensor (B, A, D) — broadcast-unsqueezed to (B, A, 1, D).

        Returns:
            SimTensor (B, A, P, D).
        """
        B, n_agent, n_query, D = query.shape

        # Unsqueeze pos encodings: (B, A, D) → (B, A, 1, D)
        _unsq_ax = self._r(F._from_data(f'{self.name}._unsq_ax',
                                np.array([2], dtype=np.int64), is_const=True))
        if query_pos is not None:
            qp = self.unsq_qpos(query_pos, _unsq_ax)
        if key_pos is not None:
            kp = self.unsq_kpos(key_pos, _unsq_ax)

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None:
            if list(query_pos.shape) == list(key.shape):
                kp = qp

        # Add positional encodings
        if query_pos is not None:
            query = self.add_qpos(query, qp)
        if key_pos is not None:
            key = self.add_kpos(key, kp)

        # Flatten: (B, A, P, D) → (B*A, P, D)
        _flat_shape = self._r(F._from_data(
            f'{self.name}._flat_shape',
            np.array([B * n_agent, n_query, D], dtype=np.int64), is_const=True))
        q = self.flatten_q(query, _flat_shape)
        k = self.flatten_k(key, _flat_shape)
        v = self.flatten_v(value, _flat_shape)
        ident = self.flatten_id(identity, _flat_shape)

        # Transpose: (B*A, P, D) → (P, B*A, D)
        q = self.trans_q(q)
        k = self.trans_k(k)
        v = self.trans_v(v)

        # MHA (seq-first)
        out = self.attn(q, key=k, value=v)

        # Transpose back: (P, B*A, D) → (B*A, P, D)
        out = self.trans_out(out)

        # Residual: identity + out (dropout omitted for inference)
        out = self.residual_add(ident, out)

        # Reshape: (B*A, P, D) → (B, A, P, D)
        _back_shape = self._r(F._from_data(
            f'{self.name}._back_shape',
            np.array([B, n_agent, n_query, D], dtype=np.int64), is_const=True))
        out = self.reshape_back(out, _back_shape)

        return out

    def analytical_param_count(self):
        D = self.embed_dims
        # MHA: Q, K, V, Out projections each (D*D + D)
        return 4 * (D * D + D)


# ======================================================================
# MotionTransformerAttentionLayer
# ======================================================================

class MotionTransformerAttentionLayer(SimNN.Module):
    """
    Flexible transformer layer composing attention + FFN + norms.

    Used in the motion prediction head for BEV deformable cross-attention.
    Very similar to MyCustomBaseTransformerLayer but with the specific
    attention types used in motion prediction (MotionDeformableAttention,
    CustomModeMultiheadAttention).

    Config example (from fusion_base_e2e.py):
        operation_order=('cross_attn', 'norm', 'ffn', 'norm')
        attn_cfgs=[dict(type='MotionDeformableAttention', ...)]

    Args:
        name (str): Module name.
        attn_cfgs (list[dict] | dict): Attention configs.
        ffn_cfgs (dict): FFN config.
        operation_order (tuple[str]): Execution order of operations.
        norm_cfg (dict): Normalization config. Default: dict(type='LN').
        batch_first (bool): Whether inputs are batch-first. Default: True.
        feedforward_channels (int): Deprecated — use ffn_cfgs instead.
        ffn_dropout (float): Deprecated — use ffn_cfgs instead.
    """

    def __init__(self, name, attn_cfgs=None,
                 ffn_cfgs=None,
                 operation_order=None,
                 norm_cfg=None, batch_first=True,
                 embed_dims=256,
                 **kwargs):
        super().__init__()
        self.name = name
        self.batch_first = batch_first

        if norm_cfg is None:
            norm_cfg = dict(type='LN')

        # Handle deprecated kwargs
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type='FFN',
                embed_dims=embed_dims,
                feedforward_channels=kwargs.get('feedforward_channels', 512),
                num_fcs=kwargs.get('num_fcs', 2),
                ffn_drop=kwargs.get('ffn_dropout', 0.1),
                act_cfg=dict(type='ReLU', inplace=True),
            )
        if isinstance(ffn_cfgs, dict):
            if 'embed_dims' not in ffn_cfgs:
                ffn_cfgs['embed_dims'] = embed_dims

        if operation_order is None:
            operation_order = ('cross_attn', 'norm', 'ffn', 'norm')
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == 'norm'

        # Count operations
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        num_ffns = operation_order.count('ffn')
        num_norms = operation_order.count('norm')
        self.num_attn = num_attn

        # Build attention modules
        if attn_cfgs is None:
            attn_cfgs = []
        elif isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            attn_cfgs = list(attn_cfgs)

        _attentions = []
        attn_idx = 0
        for op_name in operation_order:
            if op_name in ('self_attn', 'cross_attn'):
                cfg = attn_cfgs[attn_idx]
                attn = self._build_motion_attention(
                    f'{name}.attn_{attn_idx}', cfg)
                attn.operation_name = op_name
                _attentions.append(attn)
                attn_idx += 1
        self.attentions: SimNN.ModuleList | list = SimNN.ModuleList(_attentions) if _attentions else []

        # Get embed_dims
        if _attentions:
            self.embed_dims = _attentions[0].embed_dims
        else:
            self.embed_dims = embed_dims

        # Build FFNs
        _ffns = []
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs_list = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        else:
            ffn_cfgs_list = list(ffn_cfgs)
        for fi in range(num_ffns):
            cfg = ffn_cfgs_list[fi]
            if 'embed_dims' not in cfg:
                cfg['embed_dims'] = self.embed_dims
            ffn = build_feedforward_network(f'{name}.ffn_{fi}', cfg)
            _ffns.append(ffn)
        self.ffns: SimNN.ModuleList | list = SimNN.ModuleList(_ffns) if _ffns else []

        # Build norms
        _norms = []
        for ni in range(num_norms):
            norm = build_norm_layer(f'{name}.norm_{ni}', norm_cfg,
                                    self.embed_dims)
            _norms.append(norm)
        self.norms: SimNN.ModuleList | list = SimNN.ModuleList(_norms) if _norms else []

    def _build_motion_attention(self, name, cfg):
        """Build attention module for motion head.

        Supports MotionDeformableAttention and CustomModeMultiheadAttention.
        Falls back to the generic build_attention for other types.
        """
        attn_type = cfg.get('type', 'MotionDeformableAttention')

        if attn_type == 'MotionDeformableAttention':
            return MotionDeformableAttention(
                name=name,
                embed_dims=cfg.get('embed_dims', 256),
                num_heads=cfg.get('num_heads', 8),
                num_levels=cfg.get('num_levels', 1),
                num_points=cfg.get('num_points', 4),
                num_steps=cfg.get('num_steps', 1),
                sample_index=cfg.get('sample_index', -1),
                bev_range=cfg.get('bev_range', None),
                voxel_size=cfg.get('voxel_size', None),
                batch_first=cfg.get('batch_first', True),
            )
        elif attn_type == 'CustomModeMultiheadAttention':
            return CustomModeMultiheadAttention(
                name=name,
                embed_dims=cfg.get('embed_dims', 256),
                num_heads=cfg.get('num_heads', 8),
            )
        else:
            return build_attention(name, cfg)

    def __call__(self, query, key=None, value=None,
                 query_pos=None, key_pos=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 **kwargs):
        """
        Forward pass.

        Executes operations in the order specified by operation_order.

        Args:
            query: SimTensor — main query tensor.
            key: SimTensor — key for cross-attention.
            value: SimTensor — value for cross-attention.
            query_pos: SimTensor — positional encoding for query.
            key_pos: SimTensor — positional encoding for key.
            attn_masks: list of attention masks.
            **kwargs: forwarded to attention modules (bbox_results,
                      reference_trajs, spatial_shapes, etc.)

        Returns:
            SimTensor — updated query.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query, temp_key, temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query, key, value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def analytical_param_count(self):
        total = 0
        for attn in self.attentions:
            total += attn.analytical_param_count()  # type: ignore[union-attr]
        for ffn in self.ffns:
            total += ffn.analytical_param_count()  # type: ignore[union-attr]
        return total


# ======================================================================
# Self-test
# ======================================================================

if __name__ == '__main__':
    logger.info("Motion Deformable Attention — TTSim (FusionAD)")
    logger.info("=" * 70)

    D = 256
    H = 8
    num_levels = 1
    num_points = 4
    num_steps = 12
    B = 1
    A = 8
    P = 6
    bev_h, bev_w = 200, 200
    num_bev = bev_h * bev_w
    ok = True

    # --- Test 1: MotionDeformableAttention construction ---
    try:
        mda = MotionDeformableAttention(
            'test_mda', embed_dims=D, num_heads=H,
            num_levels=num_levels, num_points=num_points,
            num_steps=num_steps, sample_index=-1)
        pc = mda.analytical_param_count()
        logger.debug(f"[OK] MotionDeformableAttention  params={pc:,}")
    except Exception as e:
        logger.debug(f"[X]  MotionDeformableAttention construction FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False

    # --- Test 2: CustomModeMultiheadAttention ---
    try:
        cmma = CustomModeMultiheadAttention(
            'test_cmma', embed_dims=D, num_heads=H)
        q = F._from_data('q_cmma', np.random.randn(B, A, P, D).astype(np.float32))
        qpos = F._from_data('qpos_cmma', np.random.randn(B, A, D).astype(np.float32))
        kpos = F._from_data('kpos_cmma', np.random.randn(B, A, D).astype(np.float32))
        out = cmma(q, query_pos=qpos, key_pos=kpos)
        assert list(out.shape) == [B, A, P, D], f"Bad shape: {out.shape}"
        logger.debug(
            f"[OK] CustomModeMultiheadAttention  shape={out.shape}  "
            f"params={cmma.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X]  CustomModeMultiheadAttention FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False

    # --- Test 3: MotionTransformerAttentionLayer construction ---
    try:
        mtal = MotionTransformerAttentionLayer(
            'test_mtal',
            embed_dims=D,
            attn_cfgs=[dict(
                type='MotionDeformableAttention',
                num_steps=num_steps,
                embed_dims=D,
                num_levels=num_levels,
                num_heads=H,
                num_points=num_points,
                sample_index=-1)],
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm'))
        pc = mtal.analytical_param_count()
        logger.debug(f"[OK] MotionTransformerAttentionLayer  params={pc:,}")
    except Exception as e:
        logger.debug(f"[X]  MotionTransformerAttentionLayer construction FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False

    # --- Test 4: CustomModeMultiheadAttention — no pos ---
    try:
        out2 = cmma(q)
        assert list(out2.shape) == [B, A, P, D], f"Bad shape: {out2.shape}"
        logger.debug(f"[OK] CustomModeMultiheadAttention (no pos)  shape={out2.shape}")
    except Exception as e:
        logger.debug(f"[X]  CustomModeMultiheadAttention (no pos) FAILED: {e}")
        import traceback; traceback.print_exc()
        ok = False

    logger.info("=" * 70)
    logger.info("ALL OK" if ok else "SOME FAILURES")
