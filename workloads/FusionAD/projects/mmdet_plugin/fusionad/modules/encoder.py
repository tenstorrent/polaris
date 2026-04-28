#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of the BEVFormer Encoder for FusionAD.

Contains three classes:
  - BEVFormerEncoder   : Sequence of encoder layers + reference-point generation
  - BEVFormerLayer     : Single encoder layer  (TSA -> norm -> SCA -> norm -> FFN -> norm)
  - BEVFormerFusionLayer : Same, but adds a pts_cross_attn operation for LiDAR features

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

1. Base Classes:
   - TransformerLayerSequence -> SimNN.Module with layer list
   - MyCustomBaseTransformerLayer -> inherited directly (TTSim version)
   - BaseModule -> SimNN.Module
2. Registry / Decorators:
   - @TRANSFORMER_LAYER_SEQUENCE.register_module() -> removed
   - @TRANSFORMER_LAYER.register_module() -> removed
   - @force_fp32 / @auto_fp16 -> removed (inference only)
3. Builders:
   - build_attention -> builder_utils.build_attention
   - build_feedforward_network -> builder_utils.build_feedforward_network
   - build_norm_layer -> builder_utils.build_norm_layer
4. Modules:
   - ModuleList -> SimNN.ModuleList
5. Operations:
   - torch.linspace / meshgrid -> numpy equivalents
   - torch.matmul (lidar2img projection) -> numpy (preprocessing)
   - torch.stack / permute / reshape -> TTSim ops or numpy
"""
#----------------------------------PyTorch----------------------------------------

# from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
# import copy
# import warnings
# from mmcv.cnn.bricks.registry import (ATTENTION,
#                                       TRANSFORMER_LAYER,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import TransformerLayerSequence
# from mmcv.runner import force_fp32, auto_fp16
# import numpy as np
# import torch
# import cv2 as cv
# import mmcv
# import torch
# from mmcv import ConfigDict
# from mmcv.cnn import build_norm_layer
# from mmcv.runner.base_module import BaseModule, ModuleList
# from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention
# from mmcv.utils import TORCH_VERSION, digit_version
# from mmcv.utils import ext_loader
# ext_module = ext_loader.load_ext(
#     '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class BEVFormerEncoder(TransformerLayerSequence):

#     """
#     Attention with both self and cross
#     Implements the decoder in DETR transformer.
#     Args:
#         return_intermediate (bool): Whether to return intermediate outputs.
#         coder_norm_cfg (dict): Config of last normalization layer. Default：
#             `LN`.
#     """

#     def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
#                  **kwargs):

#         super(BEVFormerEncoder, self).__init__(*args, **kwargs)
#         self.return_intermediate = return_intermediate

#         self.num_points_in_pillar = num_points_in_pillar
#         self.pc_range = pc_range
#         self.fp16_enabled = False

#     @staticmethod
#     def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
#         """Get the reference points used in SCA and TSA.
#         Args:
#             H, W: spatial shape of bev.
#             Z: hight of pillar.
#             D: sample D points uniformly from each pillar.
#             device (obj:`device`): The device where
#                 reference_points should be.
#         Returns:
#             Tensor: reference points used in decoder, has \
#                 shape (bs, num_keys, num_levels, 2).
#         """

#         # reference points in 3D space, used in spatial cross-attention (SCA)
#         if dim == '3d':
#             zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
#                                 device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
#             xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
#                                 device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
#             ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
#                                 device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
#             ref_3d = torch.stack((xs, ys, zs), -1)
#             ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
#             ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
#             return ref_3d

#         # reference points on 2D bev plane, used in temporal self-attention (TSA).
#         elif dim == '2d':
#             ref_y, ref_x = torch.meshgrid(
#                 torch.linspace(
#                     0.5, H - 0.5, H, dtype=dtype, device=device),
#                 torch.linspace(
#                     0.5, W - 0.5, W, dtype=dtype, device=device)
#             )
#             ref_y = ref_y.reshape(-1)[None] / H
#             ref_x = ref_x.reshape(-1)[None] / W
#             ref_2d = torch.stack((ref_x, ref_y), -1)
#             ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
#             return ref_2d

#     # This function must use fp32!!!
#     @force_fp32(apply_to=('reference_points', 'img_metas'))
#     def point_sampling(self, reference_points, pc_range,  img_metas):

#         lidar2img = []
#         for img_meta in img_metas:
#             lidar2img.append(img_meta['lidar2img'])
#         lidar2img = np.asarray(lidar2img)
#         lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
#         reference_points = reference_points.clone()

#         reference_points[..., 0:1] = reference_points[..., 0:1] * \
#             (pc_range[3] - pc_range[0]) + pc_range[0]
#         reference_points[..., 1:2] = reference_points[..., 1:2] * \
#             (pc_range[4] - pc_range[1]) + pc_range[1]
#         reference_points[..., 2:3] = reference_points[..., 2:3] * \
#             (pc_range[5] - pc_range[2]) + pc_range[2]

#         reference_points = torch.cat(
#             (reference_points, torch.ones_like(reference_points[..., :1])), -1)

#         reference_points = reference_points.permute(1, 0, 2, 3)
#         D, B, num_query = reference_points.size()[:3]
#         num_cam = lidar2img.size(1)

#         reference_points = reference_points.view(
#             D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

#         lidar2img = lidar2img.view(
#             1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

#         reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
#                                             reference_points.to(torch.float32)).squeeze(-1)
#         eps = 1e-5

#         bev_mask = (reference_points_cam[..., 2:3] > eps)
#         reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
#             reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

#         reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
#         reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

#         bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
#                     & (reference_points_cam[..., 1:2] < 1.0)
#                     & (reference_points_cam[..., 0:1] < 1.0)
#                     & (reference_points_cam[..., 0:1] > 0.0))
#         if digit_version(TORCH_VERSION) >= digit_version('1.8'):
#             bev_mask = torch.nan_to_num(bev_mask)
#         else:
#             bev_mask = bev_mask.new_tensor(
#                 np.nan_to_num(bev_mask.cpu().numpy()))

#         reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
#         bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

#         return reference_points_cam, bev_mask

#     @auto_fp16()
#     def forward(self,
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_h=None,
#                 bev_w=None,
#                 bev_pos=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 valid_ratios=None,
#                 prev_bev=None,
#                 shift=0.,
#                 img_metas=None,
#                 **kwargs):
#         """Forward function for `TransformerDecoder`.
#         Args:
#             bev_query (Tensor): Input BEV query with shape
#                 `(num_query, bs, embed_dims)`.
#             key & value (Tensor): Input multi-cameta features with shape
#                 (num_cam, num_value, bs, embed_dims)
#             reference_points (Tensor): The reference
#                 points of offset. has shape
#                 (bs, num_query, 4) when as_two_stage,
#                 otherwise has shape ((bs, num_query, 2).
#             valid_ratios (Tensor): The radios of valid
#                 points on the feature map, has shape
#                 (bs, num_levels, 2)
#         Returns:
#             Tensor: Results with shape [1, num_query, bs, embed_dims] when
#                 return_intermediate is `False`, otherwise it has shape
#                 [num_layers, num_query, bs, embed_dims].
#         """

#         output = bev_query
#         intermediate = []

#         ref_3d = self.get_reference_points(
#             bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
#         ref_2d = self.get_reference_points(
#             bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

#         reference_points_cam, bev_mask = self.point_sampling(
#             ref_3d, self.pc_range, img_metas)

#         # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
#         shift_ref_2d = ref_2d  # .clone()
#         shift_ref_2d += shift[:, None, None, :]

#         # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
#         bev_query = bev_query.permute(1, 0, 2)
#         bev_pos = bev_pos.permute(1, 0, 2)
#         bs, len_bev, num_bev_level, _ = ref_2d.shape
#         if prev_bev is not None:
#             prev_bev = prev_bev.permute(1, 0, 2)
#             prev_bev = torch.stack(
#                 [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
#             hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)
#         else:
#             hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
#                 bs*2, len_bev, num_bev_level, 2)

#         for lid, layer in enumerate(self.layers):
#             output = layer(
#                 bev_query,
#                 key,
#                 value,
#                 *args,
#                 bev_pos=bev_pos,
#                 ref_2d=hybird_ref_2d,
#                 ref_3d=ref_3d,
#                 pts_ref_2d=ref_2d,
#                 bev_h=bev_h,
#                 bev_w=bev_w,
#                 spatial_shapes=spatial_shapes,
#                 level_start_index=level_start_index,
#                 reference_points_cam=reference_points_cam,
#                 bev_mask=bev_mask,
#                 prev_bev=prev_bev,
#                 **kwargs)

#             bev_query = output
#             if self.return_intermediate:
#                 intermediate.append(output)

#         if self.return_intermediate:
#             return torch.stack(intermediate)

#         return output


# @TRANSFORMER_LAYER.register_module()
# class BEVFormerLayer(MyCustomBaseTransformerLayer):
#     """Implements decoder layer in DETR transformer.
#     Args:
#         attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
#             Configs for self_attention or cross_attention, the order
#             should be consistent with it in `operation_order`. If it is
#             a dict, it would be expand to the number of attention in
#             `operation_order`.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         ffn_dropout (float): Probability of an element to be zeroed
#             in ffn. Default 0.0.
#         operation_order (tuple[str]): The execution order of operation
#             in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
#             Default: None
#         act_cfg (dict): The activation config for FFNs. Default: `LN`
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: `LN`.
#         ffn_num_fcs (int): The number of fully-connected layers in FFNs.
#             Default: 2.
#     """

#     def __init__(self,
#                  attn_cfgs,
#                  feedforward_channels,
#                  ffn_dropout=0.0,
#                  operation_order=None,
#                  act_cfg=dict(type='ReLU', inplace=True),
#                  norm_cfg=dict(type='LN'),
#                  ffn_num_fcs=2,
#                  **kwargs):
#         super(BEVFormerLayer, self).__init__(
#             attn_cfgs=attn_cfgs,
#             feedforward_channels=feedforward_channels,
#             ffn_dropout=ffn_dropout,
#             operation_order=operation_order,
#             act_cfg=act_cfg,
#             norm_cfg=norm_cfg,
#             ffn_num_fcs=ffn_num_fcs,
#             **kwargs)
#         self.fp16_enabled = False
#         assert len(operation_order) == 6
#         assert set(operation_order) == set(
#             ['self_attn', 'norm', 'cross_attn', 'ffn'])

#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 bev_pos=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 ref_2d=None,
#                 ref_3d=None,
#                 bev_h=None,
#                 bev_w=None,
#                 reference_points_cam=None,
#                 mask=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 prev_bev=None,
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
#                                                      f'attn_masks {len(attn_masks)} must be equal ' \
#                                                      f'to the number of attention in ' \
#                 f'operation_order {self.num_attn}'

#         for layer in self.operation_order:
#             # temporal self attention
#             if layer == 'self_attn':

#                 query = self.attentions[attn_index](
#                     query,
#                     prev_bev,
#                     prev_bev,
#                     identity if self.pre_norm else None,
#                     query_pos=bev_pos,
#                     key_pos=bev_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     reference_points=ref_2d,
#                     spatial_shapes=torch.tensor(
#                         [[bev_h, bev_w]], device=query.device),
#                     level_start_index=torch.tensor([0], device=query.device),
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1

#             # spaital cross attention
#             elif layer == 'cross_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     reference_points=ref_3d,
#                     reference_points_cam=reference_points_cam,
#                     mask=mask,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     spatial_shapes=spatial_shapes,
#                     level_start_index=level_start_index,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1

#         return query




# @TRANSFORMER_LAYER.register_module()
# class BEVFormerFusionLayer(BaseModule):
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

#         super(BEVFormerFusionLayer, self).__init__(init_cfg)

#         self.batch_first = batch_first

#         assert set(operation_order) & set(
#             ['self_attn', 'norm', 'ffn', 'cross_attn', 'pts_cross_attn']) == \
#             set(operation_order), f'The operation_order of' \
#             f' {self.__class__.__name__} should ' \
#             f'contains all four operation type ' \
#             f"{['self_attn', 'norm', 'ffn', 'cross_attn', 'pts_cross_attn']}"

#         num_attn = operation_order.count('self_attn') + operation_order.count(
#             'cross_attn') + operation_order.count('pts_cross_attn')
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
#             if operation_name in ['self_attn', 'cross_attn', 'pts_cross_attn']:
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
#                 bev_pos=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_masks=None,
#                 query_key_padding_mask=None,
#                 key_padding_mask=None,
#                 ref_2d=None,
#                 ref_3d=None,
#                 bev_h=None,
#                 bev_w=None,
#                 reference_points_cam=None,
#                 mask=None,
#                 spatial_shapes=None,
#                 level_start_index=None,
#                 prev_bev=None,
#                 pts_ref_2d=None,
#                 pts_feats=None,
#                 pts_spatial_shapes=None,
#                 pts_level_start_index=None,
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
#                                                      f'attn_masks {len(attn_masks)} must be equal ' \
#                                                      f'to the number of attention in ' \
#                 f'operation_order {self.num_attn}'

#         for layer in self.operation_order:
#             # temporal self attention
#             if layer == 'self_attn':

#                 query = self.attentions[attn_index](
#                     query,
#                     prev_bev,
#                     prev_bev,
#                     identity if self.pre_norm else None,
#                     query_pos=bev_pos,
#                     key_pos=bev_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     reference_points=ref_2d,
#                     spatial_shapes=torch.tensor(
#                         [[bev_h, bev_w]], device=query.device),
#                     level_start_index=torch.tensor([0], device=query.device),
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'pts_cross_attn':

#                 query = self.attentions[attn_index](
#                     query,
#                     pts_feats,
#                     pts_feats,
#                     identity if self.pre_norm else None,
#                     query_pos=bev_pos,
#                     key_pos=bev_pos,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=query_key_padding_mask,
#                     reference_points=pts_ref_2d,
#                     spatial_shapes=pts_spatial_shapes,
#                     level_start_index=pts_level_start_index,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'norm':
#                 query = self.norms[norm_index](query)
#                 norm_index += 1

#             # spaital cross attention
#             elif layer == 'cross_attn':
#                 query = self.attentions[attn_index](
#                     query,
#                     key,
#                     value,
#                     identity if self.pre_norm else None,
#                     query_pos=query_pos,
#                     key_pos=key_pos,
#                     reference_points=ref_3d,
#                     reference_points_cam=reference_points_cam,
#                     mask=mask,
#                     attn_mask=attn_masks[attn_index],
#                     key_padding_mask=key_padding_mask,
#                     spatial_shapes=spatial_shapes,
#                     level_start_index=level_start_index,
#                     **kwargs)
#                 attn_index += 1
#                 identity = query

#             elif layer == 'ffn':
#                 query = self.ffns[ffn_index](
#                     query, identity if self.pre_norm else None)
#                 ffn_index += 1

#         return query

#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger
import copy
import warnings
import math

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN

# Import sub-modules
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .builder_utils import (build_attention, build_feedforward_network,
                            build_norm_layer)


# ======================================================================
# BEVFormerEncoder
# ======================================================================

class BEVFormerEncoder(SimNN.Module):
    """
    TTSim implementation of BEVFormerEncoder.

    Sequence of BEVFormerLayer (or BEVFormerFusionLayer) modules.
    Handles reference-point generation and camera projection as
    pure-numpy preprocessing before handing off to the layers.

    Args:
        name (str): Module name.
        num_layers (int): Number of stacked encoder layers.
        layer_cfg (dict): Config dict for each layer (type, attn_cfgs, ...).
        pc_range (list[float]): Point-cloud range [x0,y0,z0,x1,y1,z1].
        num_points_in_pillar (int): Z-samples for 3-D reference points. Default 4.
        return_intermediate (bool): Return outputs from every layer. Default False.
    """

    def __init__(self,
                 name,
                 num_layers,
                 layer_cfg,
                 pc_range,
                 num_points_in_pillar=4,
                 return_intermediate=False):
        super().__init__()
        self.name = name
        self.num_layers = num_layers
        self.pc_range = list(pc_range)
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate

        # Build layers
        _layers = []
        for i in range(num_layers):
            _layers.append(build_encoder_layer(f'{name}.layer_{i}', layer_cfg))
        self.layers = SimNN.ModuleList(_layers)

    # ------------------------------------------------------------------
    # Reference-point helpers  (pure numpy -- no learnable parameters)
    # ------------------------------------------------------------------

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4,
                             dim='3d', bs=1):
        """
        Generate reference-point grids used by TSA (2-D) and SCA (3-D).

        Args:
            H, W (int): BEV spatial size.
            Z (float): Pillar height range (pc_range[5] - pc_range[2]).
            num_points_in_pillar (int): Samples along Z axis.
            dim (str): '3d' for SCA, '2d' for TSA.
            bs (int): Batch size.

        Returns:
            numpy.ndarray:
                '3d' -> [bs, D, H*W, 3]   (D = num_points_in_pillar)
                '2d' -> [bs, H*W, 1, 2]
        """
        if dim == '3d':
            D = num_points_in_pillar
            zs = np.linspace(0.5, Z - 0.5, D).astype(np.float32) / Z
            xs = np.linspace(0.5, W - 0.5, W).astype(np.float32) / W
            ys = np.linspace(0.5, H - 0.5, H).astype(np.float32) / H

            # Build [D, H, W, 3] grid
            zs_g = zs[:, None, None] * np.ones((1, H, W), dtype=np.float32)
            xs_g = np.ones((D, H, 1), dtype=np.float32) * xs[None, None, :]
            ys_g = np.ones((D, 1, W), dtype=np.float32) * ys[None, :, None]
            ref_3d = np.stack([xs_g, ys_g, zs_g], axis=-1)       # [D, H, W, 3]

            # [D, H, W, 3] -> [D, 3, H, W] -> [D, H*W, 3]
            ref_3d = ref_3d.transpose(0, 3, 1, 2).reshape(D, 3, -1).transpose(0, 2, 1)
            # [1, D, H*W, 3] -> [bs, D, H*W, 3]
            ref_3d = np.tile(ref_3d[None], (bs, 1, 1, 1))
            return ref_3d

        elif dim == '2d':
            ref_y = np.linspace(0.5, H - 0.5, H).astype(np.float32) / H
            ref_x = np.linspace(0.5, W - 0.5, W).astype(np.float32) / W
            ref_x_g, ref_y_g = np.meshgrid(ref_x, ref_y)          # each [H, W]
            ref_x_flat = ref_x_g.reshape(-1)                       # [H*W]
            ref_y_flat = ref_y_g.reshape(-1)
            ref_2d = np.stack([ref_x_flat, ref_y_flat], axis=-1)   # [H*W, 2]
            ref_2d = np.tile(ref_2d[None], (bs, 1, 1))             # [bs, H*W, 2]
            ref_2d = ref_2d[:, :, None, :]                         # [bs, H*W, 1, 2]
            return ref_2d

        else:
            raise ValueError(f"dim must be '3d' or '2d', got '{dim}'")

    @staticmethod
    def point_sampling(reference_points, pc_range, lidar2img, img_shape):
        """
        Project 3-D reference points into each camera view (pure numpy).

        Args:
            reference_points (np.ndarray): [bs, D, num_query, 3]  in [0, 1].
            pc_range (list[float]): [x0,y0,z0,x1,y1,z1].
            lidar2img (np.ndarray): [B, num_cam, 4, 4].
            img_shape (tuple): (img_H, img_W) -- pixel dimensions.

        Returns:
            reference_points_cam (np.ndarray): [num_cam, B, num_query, D, 2]
            bev_mask             (np.ndarray): [num_cam, B, num_query, D]
        """
        rp = reference_points.copy().astype(np.float32)
        rp[..., 0:1] = rp[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        rp[..., 1:2] = rp[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        rp[..., 2:3] = rp[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        # Homogeneous: [bs, D, nq, 4]
        rp = np.concatenate([rp, np.ones_like(rp[..., :1])], axis=-1)

        # -> [D, bs, nq, 4]
        rp = rp.transpose(1, 0, 2, 3)
        D, B, nq = rp.shape[:3]
        num_cam = lidar2img.shape[1]

        # Expand for cameras
        rp = rp[:, :, np.newaxis, :, :, np.newaxis]                 # [D,B,1,nq,4,1]
        rp = np.tile(rp, (1, 1, num_cam, 1, 1, 1))                 # [D,B,N,nq,4,1]

        l2i = lidar2img[np.newaxis, :, :, np.newaxis, :, :]         # [1,B,N,1,4,4]
        l2i = np.tile(l2i, (D, 1, 1, nq, 1, 1)).astype(np.float32) # [D,B,N,nq,4,4]

        # Project
        rpc = np.matmul(l2i, rp.astype(np.float32)).squeeze(-1)     # [D,B,N,nq,4]

        eps = 1e-5
        bev_mask = (rpc[..., 2:3] > eps).astype(np.float32)

        # Perspective divide
        rpc_xy = rpc[..., 0:2] / np.maximum(rpc[..., 2:3], eps)

        # Normalise by image size
        rpc_xy[..., 0] /= img_shape[1]   # width
        rpc_xy[..., 1] /= img_shape[0]   # height

        # Validity mask
        bev_mask *= (rpc_xy[..., 1:2] > 0).astype(np.float32)
        bev_mask *= (rpc_xy[..., 1:2] < 1).astype(np.float32)
        bev_mask *= (rpc_xy[..., 0:1] < 1).astype(np.float32)
        bev_mask *= (rpc_xy[..., 0:1] > 0).astype(np.float32)
        bev_mask = np.nan_to_num(bev_mask)

        # Permute to [num_cam, B, nq, D, 2]  and  [num_cam, B, nq, D]
        rpc_xy = rpc_xy.transpose(2, 1, 3, 0, 4)       # [N,B,nq,D,2]
        bev_mask = bev_mask.transpose(2, 1, 3, 0, 4).squeeze(-1)  # [N,B,nq,D]

        return rpc_xy.astype(np.float32), bev_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self,
                 bev_query,
                 key,
                 value,
                 *args,
                 bev_h=None,
                 bev_w=None,
                 bev_pos=None,
                 spatial_shapes=None,
                 level_start_index=None,
                 prev_bev=None,
                 shift=0.,
                 lidar2img=None,
                 img_shape=None,
                 **kwargs):
        """
        Forward pass of BEVFormerEncoder.

        Args:
            bev_query:   SimTensor [num_query, bs, embed_dims]
            key:         SimTensor [num_cam, l, bs, embed_dims]
            value:       SimTensor [num_cam, l, bs, embed_dims]
            bev_h, bev_w (int): BEV spatial dimensions
            bev_pos:     SimTensor [num_query, bs, embed_dims]
            spatial_shapes: list of (H, W) for multi-scale image features
            level_start_index: list of ints or None
            prev_bev:    SimTensor [num_query, bs, embed_dims] or None
            shift:       np.ndarray [bs, 2] or scalar 0. (ego-motion offset)
            lidar2img:   np.ndarray [B, num_cam, 4, 4]
            img_shape:   tuple (H, W) image pixel dimensions

        Returns:
            SimTensor [bs, num_query, embed_dims]
            (or stacked intermediate outputs if return_intermediate)
        """
        bs = bev_query.shape[1]
        num_query = bev_query.shape[0]
        embed_dims = bev_query.shape[2]

        # ---- 1. Reference points (numpy constants) ----
        Z_range = self.pc_range[5] - self.pc_range[2]
        ref_3d_np = self.get_reference_points(
            bev_h, bev_w, Z=Z_range,
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=bs)                                  # [bs, D, nq, 3]
        ref_2d_np = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs)                    # [bs, nq, 1, 2]

        # ---- 2. Camera projection (numpy preprocessing) ----
        if lidar2img is not None and img_shape is not None:
            reference_points_cam_np, bev_mask_np = self.point_sampling(
                ref_3d_np, self.pc_range, lidar2img, img_shape)
        else:
            # When camera data is unavailable, create dummy placeholders
            num_cam = 6
            D = self.num_points_in_pillar
            nq = bev_h * bev_w
            reference_points_cam_np = np.zeros(
                (num_cam, bs, nq, D, 2), dtype=np.float32)
            bev_mask_np = np.zeros(
                (num_cam, bs, nq, D), dtype=np.float32)

        # ---- 3. Shift ref_2d (bug-compatible: in-place alias) ----
        shift_np = np.array(shift, dtype=np.float32) if not isinstance(shift, np.ndarray) else shift.astype(np.float32)
        if shift_np.ndim == 0:
            shift_np = np.zeros((bs, 2), dtype=np.float32)
        ref_2d_np = ref_2d_np + shift_np[:, None, None, :]  # broadcast add

        # hybird_ref_2d: stack two copies -> [bs*2, nq, 1, 2]
        nq_2d = ref_2d_np.shape[1]
        hybird_ref_2d_np = np.stack(
            [ref_2d_np, ref_2d_np], axis=1
        ).reshape(bs * 2, nq_2d, 1, 2)

        # Unshifted ref_2d for pts_ref_2d (BEVFormerFusionLayer needs it)
        pts_ref_2d_np = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs)                    # [bs, nq, 1, 2]

        # ---- 4. Wrap numpy as const SimTensors ----
        ref_3d = F._from_data(self.name + '.ref_3d', ref_3d_np, is_const=True)
        setattr(self, ref_3d.name, ref_3d)
        hybird_ref_2d = F._from_data(self.name + '.hybird_ref_2d',
                                     hybird_ref_2d_np, is_const=True)
        setattr(self, hybird_ref_2d.name, hybird_ref_2d)
        ref_cam = F._from_data(self.name + '.ref_cam',
                               reference_points_cam_np, is_const=True)
        setattr(self, ref_cam.name, ref_cam)
        # Reduce bev_mask from [num_cam, bs, nq, D] to [num_cam, bs, nq]
        # by taking max along D axis (equivalent to PyTorch sum(-1) > 0
        # for 0/1 masks). SCA expects rank-3 input.
        bev_mask = F._from_data(self.name + '.bev_mask',
                                bev_mask_np.max(axis=-1).astype(np.float32),
                                is_const=True)
        setattr(self, bev_mask.name, bev_mask)
        pts_ref_2d = F._from_data(self.name + '.pts_ref_2d',
                                  pts_ref_2d_np, is_const=True)
        setattr(self, pts_ref_2d.name, pts_ref_2d)

        # ---- 5. Permute bev_query / bev_pos to batch-first ----
        _pq = F.Transpose(self.name + '.bq_perm', perm=[1, 0, 2])
        setattr(self, _pq.name, _pq)
        output = _pq(bev_query)
        setattr(self, output.name, output)

        _pp = F.Transpose(self.name + '.bp_perm', perm=[1, 0, 2])
        setattr(self, _pp.name, _pp)
        bev_pos_bf = _pp(bev_pos)
        setattr(self, bev_pos_bf.name, bev_pos_bf)

        # ---- 6. Handle prev_bev temporal stacking ----
        if prev_bev is not None:
            # prev_bev: [nq, bs, ed] -> [bs, nq, ed]
            _ppb = F.Transpose(self.name + '.pb_perm', perm=[1, 0, 2])
            setattr(self, _ppb.name, _ppb)
            prev_bev_bf = _ppb(prev_bev)
            setattr(self, prev_bev_bf.name, prev_bev_bf)

            # Stack [prev_bev_bf, output] along dim=1
            _ax1 = F._from_data(self.name + '.ax1',
                                np.array([1], dtype=np.int64), is_const=True)
            setattr(self, _ax1.name, _ax1)
            _usq1 = F.Unsqueeze(self.name + '.pb_unsq')
            setattr(self, _usq1.name, _usq1)
            pb_exp = _usq1(prev_bev_bf, _ax1)
            setattr(self, pb_exp.name, pb_exp)
            _usq2 = F.Unsqueeze(self.name + '.bq_unsq')
            setattr(self, _usq2.name, _usq2)
            bq_exp = _usq2(output, _ax1)
            setattr(self, bq_exp.name, bq_exp)

            _cat = F.ConcatX(self.name + '.bev_stack', axis=1)
            setattr(self, _cat.name, _cat)
            stacked = _cat(pb_exp, bq_exp)
            setattr(self, stacked.name, stacked)

            _flat_shp = F._from_data(
                self.name + '.bev_flat_shp',
                np.array([bs * 2, num_query, embed_dims], dtype=np.int64),
                is_const=True)
            setattr(self, _flat_shp.name, _flat_shp)
            _flat = F.Reshape(self.name + '.bev_flat')
            setattr(self, _flat.name, _flat)
            prev_bev_stacked = _flat(stacked, _flat_shp)
            setattr(self, prev_bev_stacked.name, prev_bev_stacked)
        else:
            prev_bev_stacked = None

        # ---- 7. Layer loop ----
        intermediate = []

        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                key,
                value,
                bev_pos=bev_pos_bf,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                pts_ref_2d=pts_ref_2d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=ref_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev_stacked,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate  # list of SimTensors

        return output

    # ------------------------------------------------------------------
    def analytical_param_count(self):
        """Total learnable parameters across all layers."""
        return sum(layer.analytical_param_count() for layer in self.layers)  # type: ignore[attr-defined]


# ======================================================================
# BEVFormerLayer
# ======================================================================

class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """
    TTSim implementation of a single BEVFormer encoder layer.

    Operation order (fixed): self_attn -> norm -> cross_attn -> norm -> ffn -> norm

    - self_attn  : TemporalSelfAttention  (BEV temporal reasoning)
    - cross_attn : SpatialCrossAttention  (camera feature projection)
    - ffn        : Feed-forward network
    - norm       : LayerNorm

    Inherits module construction from MyCustomBaseTransformerLayer and
    overrides __call__ to route BEV-specific arguments to each attention
    type correctly.

    Args:
        name (str): Module name.
        attn_cfgs (list[dict]): Configs for [TSA, SCA] attention modules.
        feedforward_channels (int): FFN hidden dim. Default 2048.
        ffn_dropout (float): FFN dropout. Default 0.0.
        operation_order (tuple[str]): Operation sequence. Default 6-element.
        act_cfg (dict): Activation config. Default ReLU.
        norm_cfg (dict): Norm config. Default LN.
        ffn_num_fcs (int): Number of FC layers in FFN. Default 2.
    """

    def __init__(self,
                 name,
                 attn_cfgs,
                 feedforward_channels=2048,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=None,
                 norm_cfg=None,
                 ffn_num_fcs=2,
                 **kwargs):
        if operation_order is None:
            operation_order = ('self_attn', 'norm', 'cross_attn',
                               'norm', 'ffn', 'norm')
        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if act_cfg is None:
            act_cfg = dict(type='ReLU', inplace=True)

        # Build ffn config from individual parameters
        ffn_cfgs = dict(
            type='FFN',
            feedforward_channels=feedforward_channels,
            num_fcs=ffn_num_fcs,
            ffn_drop=ffn_dropout,
            act_cfg=act_cfg,
        )

        super().__init__(
            name=name,
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            batch_first=True)

        assert len(operation_order) == 6
        assert set(operation_order) == {'self_attn', 'norm', 'cross_attn', 'ffn'}

    # ------------------------------------------------------------------

    def __call__(self,
                 query,
                 key=None,
                 value=None,
                 *args,
                 bev_pos=None,
                 query_pos=None,
                 key_pos=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 ref_2d=None,
                 ref_3d=None,
                 bev_h=None,
                 bev_w=None,
                 reference_points_cam=None,
                 mask=None,
                 bev_mask=None,
                 spatial_shapes=None,
                 level_start_index=None,
                 prev_bev=None,
                 **kwargs):
        """
        Forward pass of BEVFormerLayer.

        Dispatches arguments to the correct attention module:
          - self_attn  (TSA): key/value = prev_bev, reference_points = ref_2d,
            spatial_shapes = [(bev_h,bev_w)], query_pos = bev_pos
          - cross_attn (SCA): key/value = camera features, ref_3d,
            reference_points_cam, bev_mask, spatial_shapes

        Args:
            query:      SimTensor [bs, num_query, embed_dims]
            key, value: SimTensors (camera features)
            bev_pos:    SimTensor positional encoding
            ref_2d:     SimTensor [bs*2, nq, 1, 2] hybrid BEV ref points
            ref_3d:     SimTensor [bs, D, nq, 3] 3D ref points
            bev_h, bev_w: int BEV grid size
            reference_points_cam: SimTensor [N, B, nq, D, 2]
            bev_mask:   SimTensor [N, B, nq, D]
            spatial_shapes: list of (H, W) for image features
            level_start_index: list[int]
            prev_bev:   SimTensor [bs*2, nq, ed] (stacked prev+current) or None

        Returns:
            SimTensor [bs, num_query, embed_dims]
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif not isinstance(attn_masks, list):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn

        for layer in self.operation_order:

            # ---- Temporal Self-Attention (TSA) ----
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,                       # key
                    prev_bev,                       # value
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=[(bev_h, bev_w)],
                    level_start_index=None,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- Layer Norm ----
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # ---- Spatial Cross-Attention (SCA) ----
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- FFN ----
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    # ------------------------------------------------------------------
    def analytical_param_count(self):
        """Total learnable parameters (delegates to parent)."""""
        return super().analytical_param_count()


# ======================================================================
# BEVFormerFusionLayer
# ======================================================================

class BEVFormerFusionLayer(SimNN.Module):
    """
    TTSim implementation of BEVFormerFusionLayer.

    Extends the standard BEVFormerLayer with a ``pts_cross_attn``
    operation for attending to LiDAR BEV features via PtsCrossAttention.

    Typical operation_order:
        ('self_attn', 'norm', 'pts_cross_attn', 'norm',
         'cross_attn', 'norm', 'ffn', 'norm')

    Args:
        name (str): Module name.
        attn_cfgs (list[dict] | dict): Attention configs (one per attn op).
        ffn_cfgs (dict): FFN config.
        operation_order (tuple[str]): Must include 'pts_cross_attn'.
        norm_cfg (dict): Norm config. Default LN.
        batch_first (bool): Default True.
    """

    def __init__(self,
                 name,
                 attn_cfgs=None,
                 ffn_cfgs=None,
                 operation_order=None,
                 norm_cfg=None,
                 batch_first=True,
                 **kwargs):
        super().__init__()
        self.name = name
        self.batch_first = batch_first

        if norm_cfg is None:
            norm_cfg = dict(type='LN')
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='ReLU', inplace=True),
            )
        if operation_order is None:
            raise ValueError("operation_order must be specified")

        # Handle deprecated kwargs -> ffn_cfgs
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'Argument `{ori_name}` is deprecated; '
                    f'use `{new_name}` inside ffn_cfgs instead.')
                if isinstance(ffn_cfgs, dict):
                    ffn_cfgs[new_name] = kwargs.pop(ori_name)

        # Validate operation_order
        valid_ops = {'self_attn', 'norm', 'ffn', 'cross_attn', 'pts_cross_attn'}
        assert set(operation_order) <= valid_ops, \
            f'operation_order contains unsupported ops: {set(operation_order) - valid_ops}'

        num_attn = (operation_order.count('self_attn')
                    + operation_order.count('cross_attn')
                    + operation_order.count('pts_cross_attn'))

        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), \
                f'len(attn_cfgs)={len(attn_cfgs)} != num_attn={num_attn}'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'

        # ---- Build attention modules ----
        _attentions = []
        index = 0
        for op_name in operation_order:
            if op_name in ('self_attn', 'cross_attn', 'pts_cross_attn'):
                cfg = copy.deepcopy(attn_cfgs[index])
                if 'batch_first' in cfg:
                    assert self.batch_first == cfg['batch_first']
                else:
                    cfg['batch_first'] = self.batch_first
                attention = build_attention(
                    f'{name}.attn_{index}', cfg)
                attention.operation_name = op_name
                _attentions.append(attention)
                index += 1
        self.attentions = SimNN.ModuleList(_attentions)

        # Get embed_dims
        self.embed_dims = self.attentions[0].embed_dims

        # ---- Build FFN modules ----
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns, \
            f'len(ffn_cfgs)={len(ffn_cfgs)} != num_ffns={num_ffns}'
        _ffns = []
        for fi in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[fi]:
                ffn_cfgs[fi]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[fi]['embed_dims'] == self.embed_dims
            _ffns.append(
                build_feedforward_network(f'{name}.ffn_{fi}', ffn_cfgs[fi]))
        self.ffns: SimNN.ModuleList | list = SimNN.ModuleList(_ffns) if _ffns else []

        # ---- Build norm modules ----
        num_norms = operation_order.count('norm')
        _norms = []
        for ni in range(num_norms):
            _norms.append(
                build_norm_layer(f'{name}.norm_{ni}', norm_cfg, self.embed_dims))
        self.norms: SimNN.ModuleList | list = SimNN.ModuleList(_norms) if _norms else []

    # ------------------------------------------------------------------

    def __call__(self,
                 query,
                 key=None,
                 value=None,
                 *args,
                 bev_pos=None,
                 query_pos=None,
                 key_pos=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 ref_2d=None,
                 ref_3d=None,
                 bev_h=None,
                 bev_w=None,
                 reference_points_cam=None,
                 mask=None,
                 bev_mask=None,
                 spatial_shapes=None,
                 level_start_index=None,
                 prev_bev=None,
                 pts_ref_2d=None,
                 pts_feats=None,
                 pts_spatial_shapes=None,
                 pts_level_start_index=None,
                 **kwargs):
        """
        Forward pass of BEVFormerFusionLayer (includes pts_cross_attn).

        Additional args over BEVFormerLayer:
            pts_ref_2d:            SimTensor [bs, nq, 1, 2]
            pts_feats:             SimTensor (LiDAR BEV features)
            pts_spatial_shapes:    list of (H, W) for LiDAR features
            pts_level_start_index: list[int]

        Returns:
            SimTensor [bs, num_query, embed_dims]
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif not isinstance(attn_masks, list):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
        else:
            assert len(attn_masks) == self.num_attn

        for layer in self.operation_order:

            # ---- Temporal Self-Attention (TSA) ----
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=[(bev_h, bev_w)] if bev_h and bev_w else None,
                    level_start_index=None,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- PtsCrossAttention (LiDAR features) ----
            elif layer == 'pts_cross_attn':
                query = self.attentions[attn_index](
                    query,
                    pts_feats,
                    pts_feats,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=pts_ref_2d,
                    spatial_shapes=pts_spatial_shapes,
                    level_start_index=pts_level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- Layer Norm ----
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # ---- Spatial Cross-Attention (SCA) ----
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    bev_mask=bev_mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # ---- FFN ----
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    # ------------------------------------------------------------------
    def analytical_param_count(self):
        """Total learnable parameters across all sub-modules."""
        total = 0
        for attn in self.attentions:
            if hasattr(attn, 'analytical_param_count'):
                total += attn.analytical_param_count()
        for ffn in self.ffns:
            if hasattr(ffn, 'analytical_param_count'):
                total += ffn.analytical_param_count()
        for norm in self.norms:
            if hasattr(norm, 'analytical_param_count'):
                total += norm.analytical_param_count()
            else:
                total += 2 * self.embed_dims
        return total


# ======================================================================
# Factory
# ======================================================================

def build_encoder_layer(name, cfg):
    """
    Build a single encoder layer from a config dict.

    Args:
        name (str): Module name.
        cfg (dict): Must contain 'type' ('BEVFormerLayer' or
                    'BEVFormerFusionLayer') and the corresponding init args.

    Returns:
        BEVFormerLayer or BEVFormerFusionLayer instance.
    """
    cfg = copy.deepcopy(cfg)
    layer_type = cfg.pop('type', 'BEVFormerLayer')

    if layer_type == 'BEVFormerLayer':
        return BEVFormerLayer(name=name, **cfg)
    elif layer_type == 'BEVFormerFusionLayer':
        return BEVFormerFusionLayer(name=name, **cfg)
    else:
        raise ValueError(f"Unsupported encoder layer type: {layer_type}")


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("BEVFormer Encoder TTSim Module")
    logger.info("=" * 70)

    # ------ Test BEVFormerLayer ------
    try:
        attn_cfgs = [
            dict(type='TemporalSelfAttention',
                 embed_dims=256, num_heads=8, num_levels=1,
                 num_points=4, num_bev_queue=2),
            dict(type='SpatialCrossAttention',
                 embed_dims=256, num_cams=6,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 deformable_attention=dict(
                     embed_dims=256, num_heads=8,
                     num_levels=4, num_points=8)),
        ]
        layer = BEVFormerLayer(
            name='test_layer',
            attn_cfgs=attn_cfgs,
            feedforward_channels=1024,
            ffn_dropout=0.0,
            ffn_num_fcs=2)
        logger.debug("[OK] BEVFormerLayer constructed")
        logger.debug(f"  embed_dims      = {layer.embed_dims}")
        logger.debug(f"  operation_order = {layer.operation_order}")
        logger.debug(f"  num_attn        = {layer.num_attn}")
        logger.debug(f"  param_count     = {layer.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X] BEVFormerLayer failed: {e}")
        import traceback; traceback.print_exc()

    # ------ Test BEVFormerEncoder ------
    try:
        layer_cfg = dict(
            type='BEVFormerLayer',
            attn_cfgs=[
                dict(type='TemporalSelfAttention',
                     embed_dims=256, num_heads=8, num_levels=1,
                     num_points=4, num_bev_queue=2),
                dict(type='SpatialCrossAttention',
                     embed_dims=256, num_cams=6,
                     pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                     deformable_attention=dict(
                         embed_dims=256, num_heads=8,
                         num_levels=4, num_points=8)),
            ],
            feedforward_channels=1024,
            ffn_dropout=0.0,
            ffn_num_fcs=2,
        )
        encoder = BEVFormerEncoder(
            name='test_encoder',
            num_layers=6,
            layer_cfg=layer_cfg,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            num_points_in_pillar=4,
            return_intermediate=False)
        logger.debug("\n[OK] BEVFormerEncoder constructed")
        logger.debug(f"  num_layers  = {encoder.num_layers}")
        logger.debug(f"  param_count = {encoder.analytical_param_count():,}")
    except Exception as e:
        logger.debug(f"[X] BEVFormerEncoder failed: {e}")
        import traceback; traceback.print_exc()

    # ------ Test get_reference_points ------
    try:
        ref_3d = BEVFormerEncoder.get_reference_points(
            50, 50, Z=8, num_points_in_pillar=4, dim='3d', bs=2)
        ref_2d = BEVFormerEncoder.get_reference_points(
            50, 50, dim='2d', bs=2)
        logger.debug("\n[OK] get_reference_points")
        logger.debug(f"  ref_3d shape: {ref_3d.shape}   (expected [2, 4, 2500, 3])")
        logger.debug(f"  ref_2d shape: {ref_2d.shape}   (expected [2, 2500, 1, 2])")
    except Exception as e:
        logger.debug(f"[X] get_reference_points failed: {e}")
        import traceback; traceback.print_exc()

    # ------ Test point_sampling ------
    try:
        lidar2img = np.random.randn(2, 6, 4, 4).astype(np.float32)
        rpc, mask = BEVFormerEncoder.point_sampling(
            ref_3d,
            [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            lidar2img,
            (900, 1600))
        logger.debug("\n[OK] point_sampling")
        logger.debug(f"  ref_points_cam shape: {rpc.shape}  (expected [6, 2, 2500, 4, 2])")
        logger.debug(f"  bev_mask shape:       {mask.shape}  (expected [6, 2, 2500, 4])")
    except Exception as e:
        logger.debug(f"[X] point_sampling failed: {e}")
        import traceback; traceback.print_exc()

    logger.info("\n" + "=" * 70)
