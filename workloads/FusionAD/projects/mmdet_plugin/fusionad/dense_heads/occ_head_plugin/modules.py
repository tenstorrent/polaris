
"""
TTSim implementation of OccHead plugin modules for FusionAD.

Inference-only conversion. Training-specific logic omitted.

Classes:
  - BevFeatureSlicer   : Grid-sample a sub-region of BEV features.
  - MLP                : Multi-layer perceptron (Linear + ReLU).
  - SimpleConv2d       : Sequential Conv2d layers with BN+ReLU.
  - CVT_DecoderBlock   : Upsample + Conv + BN, optional residual.
  - CVT_Decoder        : Stack of CVT_DecoderBlocks.
  - UpsamplingAdd      : Upsample + Conv + BN + skip add.
  - Bottleneck         : Residual bottleneck with optional down/upsample.
"""

# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================

# import torch
# from torch import nn
# import torch.utils.checkpoint as checkpoint
# from .utils import calculate_birds_eye_view_parameters
# import torch.nn.functional as F
# from mmcv.runner import BaseModule
# from mmcv.cnn import ConvModule, build_conv_layer
# from einops import rearrange
# from collections import OrderedDict
#
# # Grid sampler
# # Sample a smaller receptive-field bev from larger one
# class BevFeatureSlicer(nn.Module):
#     def __init__(self, grid_conf, map_grid_conf):
#         super().__init__()
#         if grid_conf == map_grid_conf:
#             self.identity_mapping = True
#         else:
#             self.identity_mapping = False
#
#             bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
#                 grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
#             )
#
#             map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
#                 map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
#             )
#
#             self.map_x = torch.arange(
#                 map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])
#
#             self.map_y = torch.arange(
#                 map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])
#
#             # convert to normalized coords
#             self.norm_map_x = self.map_x / (- bev_start_position[0])
#             self.norm_map_y = self.map_y / (- bev_start_position[1])
#
#             tmp_m, tmp_n = torch.meshgrid(
#                 self.norm_map_x, self.norm_map_y)  # indexing 'ij'
#             tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results
#
#             self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)
#
#     def forward(self, x):
#         # x: bev feature map tensor of shape (b, c, h, w)
#         if self.identity_mapping:
#             return x
#         else:
#             grid = self.map_grid.unsqueeze(0).type_as(
#                 x).repeat(x.shape[0], 1, 1, 1)  # (b, h, w, 2)
#
#             return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)
#
# # General layers
# class MLP(nn.Module):
#     """Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
#
# class SimpleConv2d(BaseModule):
#     def __init__(self, in_channels,
#                        out_channels,
#
#                        conv_channels=64,
#                        num_conv=1,
#                        conv_cfg=dict(type='Conv2d'),
#                        norm_cfg=dict(type='BN2d'),
#                        bias='auto',
#                        init_cfg=None,
#                        ):
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#             'behavior, init_cfg is not allowed to be set'
#         super().__init__(init_cfg=init_cfg)
#         self.out_channels = out_channels
#         if num_conv == 1:
#             conv_channels = in_channels
#
#         conv_layers = []
#         c_in = in_channels
#         for i in range(num_conv-1):
#             conv_layers.append(
#                 ConvModule(
#                     c_in,
#                     conv_channels,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=bias,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                 )
#             )
#             c_in = conv_channels
#         # No norm and relu in last conv
#         conv_layers.append(
#             build_conv_layer(
#                 conv_cfg,
#                 conv_channels,
#                 out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 bias=True
#             )
#         )
#         self.conv_layers = nn.Sequential(*conv_layers)
#
#         if init_cfg is None:
#             self.init_cfg = dict(type='Kaiming', layer='Conv2d')
#
#     def forward(self, x):
#         b, c_in, h_in, w_in = x.size()
#         out = self.conv_layers(x)
#         assert out.size() == (b, self.out_channels, h_in, w_in)  # sanity check
#         return out
#
# # Decoder
# class CVT_DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
#         super().__init__()
#
#         dim = out_channels // factor
#
#         if upsample:
#             self.conv = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(dim),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels))
#         else:
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(dim),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels))
#
#         if residual:
#             self.up = nn.Conv2d(skip_dim, out_channels, 1)
#         else:
#             self.up = None
#
#         self.with_relu = with_relu
#         if self.with_relu:
#             self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x, skip):
#         x = self.conv(x)
#
#         if self.up is not None:
#             up = self.up(skip)
#             up = F.interpolate(up, x.shape[-2:])
#
#             x = x + up
#         if self.with_relu:
#             return self.relu(x)
#         return x
#
# class CVT_Decoder(BaseModule):
#     def __init__(self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False, init_cfg=None):
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#             'behavior, init_cfg is not allowed to be set'
#         super().__init__(init_cfg=init_cfg)
#
#         layers = []
#         channels = dim
#
#         for i, out_channels in enumerate(blocks):
#             with_relu = i < len(blocks) - 1  # if not last block, with relu
#             layer = CVT_DecoderBlock(channels, out_channels, dim, residual, factor, upsample, with_relu=with_relu)
#             layers.append(layer)
#
#             channels = out_channels
#
#         self.layers = nn.Sequential(*layers)
#         self.out_channels = channels
#         self.use_checkpoint = use_checkpoint
#
#         if init_cfg is None:
#             self.init_cfg = dict(type='Kaiming', layer='Conv2d')
#
#     def forward(self, x):
#         b, t = x.size(0), x.size(1)
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
#         y = x
#         for layer in self.layers:
#             if self.use_checkpoint:
#                 y = checkpoint(layer, y, x)
#             else:
#                 y = layer(y, x)
#
#         y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)
#         return y
#
#
# # Conv modules
# class UpsamplingAdd(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()
#         self.upsample_layer = nn.Sequential(
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels),
#         )
#
#     def forward(self, x, x_skip):
#         x = self.upsample_layer(x)
#         return x + x_skip
#
# class Interpolate(nn.Module):
#     def __init__(self, scale_factor: int = 2):
#         super().__init__()
#         self._interpolate = nn.functional.interpolate
#         self._scale_factor = scale_factor
#
#     def forward(self, x):
#         return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)
#
# class Bottleneck(nn.Module):
#     """
#     Defines a bottleneck module with a residual connection
#     """
#
#     def __init__(
#         self,
#         in_channels,
#         out_channels=None,
#         kernel_size=3,
#         dilation=1,
#         groups=1,
#         upsample=False,
#         downsample=False,
#         dropout=0.0,
#     ):
#         super().__init__()
#         self._downsample = downsample
#         bottleneck_channels = int(in_channels / 2)
#         out_channels = out_channels or in_channels
#         padding_size = ((kernel_size - 1) * dilation + 1) // 2
#
#         # Define the main conv operation
#         assert dilation == 1
#         if upsample:
#             assert not downsample, 'downsample and upsample not possible simultaneously.'
#             bottleneck_conv = nn.ConvTranspose2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=1,
#                 stride=2,
#                 output_padding=padding_size,
#                 padding=padding_size,
#                 groups=groups,
#             )
#         elif downsample:
#             bottleneck_conv = nn.Conv2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=dilation,
#                 stride=2,
#                 padding=padding_size,
#                 groups=groups,
#             )
#         else:
#             bottleneck_conv = nn.Conv2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=dilation,
#                 padding=padding_size,
#                 groups=groups,
#             )
#
#         self.layers = nn.Sequential(
#             OrderedDict(
#                 [
#                     # First projection with 1x1 kernel
#                     ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
#                     ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
#                                                        nn.ReLU(inplace=True))),
#                     # Second conv block
#                     ('conv', bottleneck_conv),
#                     ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
#                     # Final projection with 1x1 kernel
#                     ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
#                     ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
#                                                      nn.ReLU(inplace=True))),
#                     # Regulariser
#                     ('dropout', nn.Dropout2d(p=dropout)),
#                 ]
#             )
#         )
#
#         if out_channels == in_channels and not downsample and not upsample:
#             self.projection = None
#         else:
#             projection = OrderedDict()
#             if upsample:
#                 projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
#             elif downsample:
#                 projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
#             projection.update(
#                 {
#                     'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                     'bn_skip_proj': nn.BatchNorm2d(out_channels),
#                 }
#             )
#             self.projection = nn.Sequential(projection)
#
#     def forward(self, *args):
#         (x,) = args
#         x_residual = self.layers(x)
#         if self.projection is not None:
#             if self._downsample:
#                 # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
#                 x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
#             return x_residual + self.projection(x)
#         return x_residual + x
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#--------------------------------PyTorch------------------------------------------
#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# Modifications:
# - Modified by FusionAD on 2023.5
# - Added extended support from FusionAD (https://arxiv.org/abs/2308.01006)

# import torch
# from torch import nn
# import torch.utils.checkpoint as checkpoint
# from .utils import calculate_birds_eye_view_parameters
# import torch.nn.functional as F
# from mmcv.runner import BaseModule
# from mmcv.cnn import ConvModule, build_conv_layer
# from einops import rearrange
# from collections import OrderedDict

# # Grid sampler
# # Sample a smaller receptive-field bev from larger one
# class BevFeatureSlicer(nn.Module):
#     def __init__(self, grid_conf, map_grid_conf):
#         super().__init__()
#         if grid_conf == map_grid_conf:
#             self.identity_mapping = True
#         else:
#             self.identity_mapping = False

#             bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
#                 grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
#             )

#             map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
#                 map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
#             )

#             self.map_x = torch.arange(
#                 map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

#             self.map_y = torch.arange(
#                 map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

#             # convert to normalized coords
#             self.norm_map_x = self.map_x / (- bev_start_position[0])
#             self.norm_map_y = self.map_y / (- bev_start_position[1])

#             tmp_m, tmp_n = torch.meshgrid(
#                 self.norm_map_x, self.norm_map_y)  # indexing 'ij'
#             tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results

#             self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

#     def forward(self, x):
#         # x: bev feature map tensor of shape (b, c, h, w)
#         if self.identity_mapping:
#             return x
#         else:
#             grid = self.map_grid.unsqueeze(0).type_as(
#                 x).repeat(x.shape[0], 1, 1, 1)  # (b, h, w, 2)

#             return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)

# # General layers
# class MLP(nn.Module):
#     """Very simple multi-layer perceptron (also called FFN)"""

#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(
#             nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
#         )

#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x

# class SimpleConv2d(BaseModule):
#     def __init__(self, in_channels,
#                        out_channels,

#                        conv_channels=64,
#                        num_conv=1,
#                        conv_cfg=dict(type='Conv2d'),
#                        norm_cfg=dict(type='BN2d'),
#                        bias='auto',
#                        init_cfg=None,
#                        ):
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#             'behavior, init_cfg is not allowed to be set'
#         super().__init__(init_cfg=init_cfg)
#         self.out_channels = out_channels
#         if num_conv == 1:
#             conv_channels = in_channels

#         conv_layers = []
#         c_in = in_channels
#         for i in range(num_conv-1):
#             conv_layers.append(
#                 ConvModule(
#                     c_in,
#                     conv_channels,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=bias,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                 )
#             )
#             c_in = conv_channels
#         # No norm and relu in last conv
#         conv_layers.append(
#             build_conv_layer(
#                 conv_cfg,
#                 conv_channels,
#                 out_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0,
#                 bias=True
#             )
#         )
#         self.conv_layers = nn.Sequential(*conv_layers)

#         if init_cfg is None:
#             self.init_cfg = dict(type='Kaiming', layer='Conv2d')

#     def forward(self, x):
#         b, c_in, h_in, w_in = x.size()
#         out = self.conv_layers(x)
#         assert out.size() == (b, self.out_channels, h_in, w_in)  # sanity check
#         return out

# # Decoder
# class CVT_DecoderBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, skip_dim, residual, factor, upsample, with_relu=True):
#         super().__init__()

#         dim = out_channels // factor

#         if upsample:
#             self.conv = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(dim),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels))
#         else:
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
#                 nn.BatchNorm2d(dim),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_channels))

#         if residual:
#             self.up = nn.Conv2d(skip_dim, out_channels, 1)
#         else:
#             self.up = None

#         self.with_relu = with_relu
#         if self.with_relu:
#             self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, skip):
#         x = self.conv(x)

#         if self.up is not None:
#             up = self.up(skip)
#             up = F.interpolate(up, x.shape[-2:])

#             x = x + up
#         if self.with_relu:
#             return self.relu(x)
#         return x

# class CVT_Decoder(BaseModule):
#     def __init__(self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False, init_cfg=None):
#         assert init_cfg is None, 'To prevent abnormal initialization ' \
#             'behavior, init_cfg is not allowed to be set'
#         super().__init__(init_cfg=init_cfg)

#         layers = []
#         channels = dim

#         for i, out_channels in enumerate(blocks):
#             with_relu = i < len(blocks) - 1  # if not last block, with relu
#             layer = CVT_DecoderBlock(channels, out_channels, dim, residual, factor, upsample, with_relu=with_relu)
#             layers.append(layer)

#             channels = out_channels

#         self.layers = nn.Sequential(*layers)
#         self.out_channels = channels
#         self.use_checkpoint = use_checkpoint

#         if init_cfg is None:
#             self.init_cfg = dict(type='Kaiming', layer='Conv2d')

#     def forward(self, x):
#         b, t = x.size(0), x.size(1)
#         x = rearrange(x, 'b t c h w -> (b t) c h w')
#         y = x
#         for layer in self.layers:
#             if self.use_checkpoint:
#                 y = checkpoint(layer, y, x)
#             else:
#                 y = layer(y, x)

#         y = rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)
#         return y


# # Conv modules
# class UpsamplingAdd(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super().__init__()
#         self.upsample_layer = nn.Sequential(
#             nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(out_channels),
#         )

#     def forward(self, x, x_skip):
#         x = self.upsample_layer(x)
#         return x + x_skip

# class Interpolate(nn.Module):
#     def __init__(self, scale_factor: int = 2):
#         super().__init__()
#         self._interpolate = nn.functional.interpolate
#         self._scale_factor = scale_factor

#     def forward(self, x):
#         return self._interpolate(x, scale_factor=self._scale_factor, mode='bilinear', align_corners=False)

# class Bottleneck(nn.Module):
#     """
#     Defines a bottleneck module with a residual connection
#     """

#     def __init__(
#         self,
#         in_channels,
#         out_channels=None,
#         kernel_size=3,
#         dilation=1,
#         groups=1,
#         upsample=False,
#         downsample=False,
#         dropout=0.0,
#     ):
#         super().__init__()
#         self._downsample = downsample
#         bottleneck_channels = int(in_channels / 2)
#         out_channels = out_channels or in_channels
#         padding_size = ((kernel_size - 1) * dilation + 1) // 2

#         # Define the main conv operation
#         assert dilation == 1
#         if upsample:
#             assert not downsample, 'downsample and upsample not possible simultaneously.'
#             bottleneck_conv = nn.ConvTranspose2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=1,
#                 stride=2,
#                 output_padding=padding_size,
#                 padding=padding_size,
#                 groups=groups,
#             )
#         elif downsample:
#             bottleneck_conv = nn.Conv2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=dilation,
#                 stride=2,
#                 padding=padding_size,
#                 groups=groups,
#             )
#         else:
#             bottleneck_conv = nn.Conv2d(
#                 bottleneck_channels,
#                 bottleneck_channels,
#                 kernel_size=kernel_size,
#                 bias=False,
#                 dilation=dilation,
#                 padding=padding_size,
#                 groups=groups,
#             )

#         self.layers = nn.Sequential(
#             OrderedDict(
#                 [
#                     # First projection with 1x1 kernel
#                     ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
#                     ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels),
#                                                        nn.ReLU(inplace=True))),
#                     # Second conv block
#                     ('conv', bottleneck_conv),
#                     ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
#                     # Final projection with 1x1 kernel
#                     ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
#                     ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels),
#                                                      nn.ReLU(inplace=True))),
#                     # Regulariser
#                     ('dropout', nn.Dropout2d(p=dropout)),
#                 ]
#             )
#         )

#         if out_channels == in_channels and not downsample and not upsample:
#             self.projection = None
#         else:
#             projection = OrderedDict()
#             if upsample:
#                 projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
#             elif downsample:
#                 projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
#             projection.update(
#                 {
#                     'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#                     'bn_skip_proj': nn.BatchNorm2d(out_channels),
#                 }
#             )
#             self.projection = nn.Sequential(projection)

#     def forward(self, *args):
#         (x,) = args
#         x_residual = self.layers(x)
#         if self.projection is not None:
#             if self._downsample:
#                 # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
#                 x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
#             return x_residual + self.projection(x)
#         return x_residual + x

#--------------------------------TTsim------------------------------------------

import sys
import os
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))

dense_heads_dir = os.path.abspath(os.path.join(current_dir, '..'))
if dense_heads_dir not in sys.path:
    sys.path.insert(0, dense_heads_dir)

fusionad_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if fusionad_dir not in sys.path:
    sys.path.insert(0, fusionad_dir)

polaris_root = os.path.abspath(
    os.path.join(current_dir, '..', '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ─── Utility (init-time only, plain numpy) ────────────────────────────────

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """Compute resolution, start position, and dimension for BEV grid."""
    bev_resolution = np.array(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=np.float32)
    bev_start_position = np.array(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]],
        dtype=np.float32)
    bev_dimension = np.array(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=np.int64)
    return bev_resolution, bev_start_position, bev_dimension


# ─── BevFeatureSlicer ─────────────────────────────────────────────────────

class BevFeatureSlicer(SimNN.Module):
    """
    Sample a sub-region of BEV features via grid_sample.

    If grid_conf == map_grid_conf, acts as identity.
    Otherwise precomputes a normalised sampling grid at init and uses
    F.GridSample in forward.
    """

    def __init__(self, name, grid_conf, map_grid_conf):
        super().__init__()
        self.name = name

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
            self.identity_op = F.Identity(f'{name}.identity')
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, _ = \
                calculate_birds_eye_view_parameters(
                    grid_conf['xbound'], grid_conf['ybound'],
                    grid_conf['zbound'])

            map_bev_resolution, map_bev_start_position, _ = \
                calculate_birds_eye_view_parameters(
                    map_grid_conf['xbound'], map_grid_conf['ybound'],
                    map_grid_conf['zbound'])

            map_x = np.arange(
                map_bev_start_position[0],
                map_grid_conf['xbound'][1], map_bev_resolution[0])
            map_y = np.arange(
                map_bev_start_position[1],
                map_grid_conf['ybound'][1], map_bev_resolution[1])

            norm_map_x = map_x / (-bev_start_position[0])
            norm_map_y = map_y / (-bev_start_position[1])

            tmp_m, tmp_n = np.meshgrid(norm_map_x, norm_map_y, indexing='ij')
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # 'xy' mode

            # grid: (1, H_out, W_out, 2) — tiled to batch in forward
            grid_np = np.stack([tmp_m, tmp_n], axis=2).astype(np.float32)
            grid_np = grid_np[np.newaxis, ...]

            self.grid_const = F._from_data(
                f'{name}.grid', grid_np, is_const=True)
            self.grid_sample = F.GridSample(
                f'{name}.grid_sample', mode='bilinear',
                padding_mode='zeros', align_corners=True)
            self.tile_op = F.Tile(f'{name}.tile')

        super().link_op2module()

    def __call__(self, x):
        if self.identity_mapping:
            return self.identity_op(x)

        b = x.shape[0]
        self._tile_repeats = F._from_data(
            f'{self.name}._tile_repeats',
            np.array([b, 1, 1, 1], dtype=np.int64), is_const=True)
        grid = self.tile_op(self.grid_const, self._tile_repeats)
        return self.grid_sample(x, grid)


# ─── MLP ──────────────────────────────────────────────────────────────────

class MLP(SimNN.Module):
    """Multi-layer perceptron: chain of Linear + ReLU (last layer no ReLU)."""

    def __init__(self, name, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.name = name
        self.num_layers = num_layers

        h = [hidden_dim] * (num_layers - 1)
        dims = list(zip([input_dim] + h, h + [output_dim]))

        self.linears = []
        self.relus = []
        for i, (n_in, n_out) in enumerate(dims):
            lin = SimNN.Linear(f'{name}.layer_{i}',
                               in_features=n_in, out_features=n_out)
            setattr(self, f'layer_{i}', lin)
            self.linears.append(lin)
            if i < num_layers - 1:
                relu = F.Relu(f'{name}.relu_{i}')
                setattr(self, f'relu_{i}', relu)
                self.relus.append(relu)

        super().link_op2module()

    def __call__(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < self.num_layers - 1:
                x = self.relus[i](x)
        return x


# ─── SimpleConv2d ─────────────────────────────────────────────────────────

class SimpleConv2d(SimNN.Module):
    """
    Sequential Conv2d layers.
    (num_conv - 1) × Conv3x3 + BN + ReLU, then 1 × Conv1x1 (with bias).
    """

    def __init__(self, name, in_channels, out_channels,
                 conv_channels=64, num_conv=1):
        super().__init__()
        self.name = name
        self.out_channels = out_channels

        if num_conv == 1:
            conv_channels = in_channels

        self.conv_blocks = []
        c_in = in_channels
        for i in range(num_conv - 1):
            conv = F.Conv2d(f'{name}.conv_{i}', c_in, conv_channels,
                            kernel_size=3, stride=1, padding=1, bias=False)
            bn = F.BatchNorm2d(f'{name}.bn_{i}', conv_channels)
            relu = F.Relu(f'{name}.relu_{i}')
            setattr(self, f'conv_{i}', conv)
            setattr(self, f'bn_{i}', bn)
            setattr(self, f'relu_{i}', relu)
            self.conv_blocks.append((conv, bn, relu))
            c_in = conv_channels

        # Final 1x1 conv with bias, no BN/ReLU
        self.final_conv = F.Conv2d(
            f'{name}.final_conv', conv_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=True)

        super().link_op2module()

    def __call__(self, x):
        for conv, bn, relu in self.conv_blocks:
            x = conv(x)
            x = bn(x)
            x = relu(x)
        x = self.final_conv(x)
        return x


# ─── CVT_DecoderBlock ─────────────────────────────────────────────────────

class CVT_DecoderBlock(SimNN.Module):
    """
    Decoder block: optional upsample + Conv3x3+BN+ReLU + Conv1x1+BN,
    optional residual (Conv1x1 on skip + interpolate), optional final ReLU.

    Args:
        residual_scale: scale factor for interpolating skip to match output.
    """

    def __init__(self, name, in_channels, out_channels, skip_dim,
                 residual, factor, upsample, with_relu=True,
                 residual_scale=2):
        super().__init__()
        self.name = name
        self.has_residual = residual
        self.has_upsample = upsample
        self.with_relu = with_relu

        dim = out_channels // factor

        # Main conv path
        if upsample:
            self.up_conv = F.Resize(
                f'{name}.up', scale_factor=2,
                mode='bilinear', align_corners=True)

        self.conv1 = F.Conv2d(f'{name}.conv1', in_channels, dim,
                              kernel_size=3, padding=1, bias=False)
        self.bn1 = F.BatchNorm2d(f'{name}.bn1', dim)
        self.relu1 = F.Relu(f'{name}.relu1')
        self.conv2 = F.Conv2d(f'{name}.conv2', dim, out_channels,
                              kernel_size=1, padding=0, bias=False)
        self.bn2 = F.BatchNorm2d(f'{name}.bn2', out_channels)

        # Residual path
        if residual:
            self.skip_conv = F.Conv2d(
                f'{name}.skip_conv', skip_dim, out_channels,
                kernel_size=1, bias=True)
            self.skip_interp = F.Resize(
                f'{name}.skip_interp',
                scale_factor=residual_scale,
                mode='bilinear', align_corners=False)
            self.add_op = F.Add(f'{name}.add')

        if with_relu:
            self.final_relu = F.Relu(f'{name}.final_relu')

        super().link_op2module()

    def __call__(self, x, skip):
        # Main path
        if self.has_upsample:
            x = self.up_conv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Residual
        if self.has_residual:
            up = self.skip_conv(skip)
            up = self.skip_interp(up)
            x = self.add_op(x, up)

        if self.with_relu:
            x = self.final_relu(x)
        return x


# ─── CVT_Decoder ──────────────────────────────────────────────────────────

class CVT_Decoder(SimNN.Module):
    """
    Stack of CVT_DecoderBlocks applied over (B*T) BEV features.

    Input:  (B, T, C, H, W)
    Output: (B, T, C_out, H_out, W_out)
    """

    def __init__(self, name, dim, blocks, residual=True, factor=2,
                 upsample=True):
        super().__init__()
        self.name = name
        channels = dim

        self.block_list = []
        cumulative_scale = 1
        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1
            if upsample:
                cumulative_scale *= 2
            res_scale = cumulative_scale if residual else 1

            block = CVT_DecoderBlock(
                f'{name}.block_{i}', channels, out_channels, dim,
                residual, factor, upsample, with_relu=with_relu,
                residual_scale=res_scale)
            setattr(self, f'block_{i}', block)
            self.block_list.append(block)
            channels = out_channels

        self.out_channels = channels

        # Reshape ops for (B,T,C,H,W) <-> (B*T,C,H,W)
        self.reshape_flat = F.Reshape(f'{name}.reshape_flat')
        self.reshape_back = F.Reshape(f'{name}.reshape_back')

        super().link_op2module()

    def __call__(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        self._flat_shape = F._from_data(
            f'{self.name}._flat_shape',
            np.array([B * T, C, H, W], dtype=np.int64), is_const=True)
        y = self.reshape_flat(x, self._flat_shape)

        skip = y  # skip is always the original flattened input
        for block in self.block_list:
            y = block(y, skip)

        _, C_out, H_out, W_out = y.shape
        self._back_shape = F._from_data(
            f'{self.name}._back_shape',
            np.array([B, T, C_out, H_out, W_out], dtype=np.int64),
            is_const=True)
        y = self.reshape_back(y, self._back_shape)
        return y


# ─── UpsamplingAdd ────────────────────────────────────────────────────────

class UpsamplingAdd(SimNN.Module):
    """Upsample(2x) + Conv1x1 + BN, then element-wise add with skip."""

    def __init__(self, name, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.name = name

        self.up = F.Resize(
            f'{name}.up', scale_factor=scale_factor,
            mode='bilinear', align_corners=False)
        self.conv = F.Conv2d(
            f'{name}.conv', in_channels, out_channels,
            kernel_size=1, padding=0, bias=False)
        self.bn = F.BatchNorm2d(f'{name}.bn', out_channels)
        self.add_op = F.Add(f'{name}.add')

        super().link_op2module()

    def __call__(self, x, x_skip):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.add_op(x, x_skip)
        return x


# ─── Bottleneck ───────────────────────────────────────────────────────────

class Bottleneck(SimNN.Module):
    """
    Residual bottleneck module.

    Main path: Conv1x1↓ → BN+ReLU → Conv3x3 → BN+ReLU → Conv1x1↑ → BN+ReLU → Dropout
    Skip path: optional MaxPool2d + Conv1x1 + BN (when dims change or down/upsample)
    """

    def __init__(self, name, in_channels, out_channels=None,
                 kernel_size=3, dilation=1, groups=1,
                 upsample=False, downsample=False, dropout=0.0):
        super().__init__()
        self.name = name
        self._downsample = downsample
        self._upsample = upsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        assert dilation == 1

        # ── Main path ──
        # 1x1 down-project
        self.conv_down = F.Conv2d(
            f'{name}.conv_down', in_channels, bottleneck_channels,
            kernel_size=1, bias=False)
        self.bn_down = F.BatchNorm2d(f'{name}.bn_down', bottleneck_channels)
        self.relu_down = F.Relu(f'{name}.relu_down')

        # Middle conv (3x3, possibly strided)
        if upsample:
            assert not downsample
            self.mid_conv = F.ConvTranspose2d(
                f'{name}.mid_conv', bottleneck_channels,
                bottleneck_channels, kernel_size=kernel_size, stride=2)
        elif downsample:
            self.mid_conv = F.Conv2d(
                f'{name}.mid_conv', bottleneck_channels,
                bottleneck_channels, kernel_size=kernel_size, stride=2,
                padding=padding_size, groups=groups, bias=False)
        else:
            self.mid_conv = F.Conv2d(
                f'{name}.mid_conv', bottleneck_channels,
                bottleneck_channels, kernel_size=kernel_size,
                padding=padding_size, groups=groups, bias=False)

        self.bn_mid = F.BatchNorm2d(f'{name}.bn_mid', bottleneck_channels)
        self.relu_mid = F.Relu(f'{name}.relu_mid')

        # 1x1 up-project
        self.conv_up = F.Conv2d(
            f'{name}.conv_up', bottleneck_channels, out_channels,
            kernel_size=1, bias=False)
        self.bn_up = F.BatchNorm2d(f'{name}.bn_up', out_channels)
        self.relu_up = F.Relu(f'{name}.relu_up')

        # Dropout (train_mode=False => pass-through at inference)
        self.dropout_op = F.Dropout(f'{name}.dropout', dropout, False)

        # ── Skip / projection path ──
        self.has_projection = not (
            out_channels == in_channels
            and not downsample and not upsample)

        if self.has_projection:
            if upsample:
                self.skip_interp = F.Resize(
                    f'{name}.skip_interp', scale_factor=2,
                    mode='bilinear', align_corners=False)
            elif downsample:
                self.skip_pool = F.MaxPool2d(
                    f'{name}.skip_pool', kernel_size=2, stride=2)
                self.pad_op = F.Pad(f'{name}.pad')
                # Pre-create pad values for odd-dim case
                self._pad_vals = F._from_data(
                    f'{name}._pad_vals',
                    np.array([0, 0, 1, 1], dtype=np.int64),
                    is_const=True)

            self.skip_conv = F.Conv2d(
                f'{name}.skip_conv', in_channels, out_channels,
                kernel_size=1, bias=False)
            self.skip_bn = F.BatchNorm2d(f'{name}.skip_bn', out_channels)

        self.add_op = F.Add(f'{name}.add')

        super().link_op2module()

    def __call__(self, x):
        # Main path
        res = self.conv_down(x)
        res = self.bn_down(res)
        res = self.relu_down(res)

        res = self.mid_conv(res)
        res = self.bn_mid(res)
        res = self.relu_mid(res)

        res = self.conv_up(res)
        res = self.bn_up(res)
        res = self.relu_up(res)

        res = self.dropout_op(res)

        # Skip path
        if self.has_projection:
            skip = x
            if self._downsample:
                # Pad odd dimensions so MaxPool2d produces matching size
                h_pad = skip.shape[-2] % 2
                w_pad = skip.shape[-1] % 2
                if h_pad or w_pad:
                    skip = self.pad_op(skip, self._pad_vals)
                skip = self.skip_pool(skip)
            elif self._upsample:
                skip = self.skip_interp(skip)
            skip = self.skip_conv(skip)
            skip = self.skip_bn(skip)
            return self.add_op(res, skip)

        return self.add_op(res, x)


# ─── Post-processing (numpy, not in graph) ────────────────────────────────

def predict_instance_segmentation_and_trajectories(
        foreground_masks, ins_sigmoid, vehicles_id=1):
    """
    Post-processing for panoptic instance segmentation.
    Operates on numpy arrays after the graph forward pass.
    """
    if foreground_masks.ndim == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)
    foreground_masks = (foreground_masks == vehicles_id)

    argmax_ins = ins_sigmoid.argmax(axis=1) + 1
    instance_seg = (argmax_ins * foreground_masks.astype(np.float32)).astype(
        np.int64)

    # Make consecutive
    unique_ids = np.unique(instance_seg)
    new_seg = np.zeros_like(instance_seg)
    for new_id, old_id in enumerate(unique_ids):
        if old_id == 0:
            continue
        new_seg[instance_seg == old_id] = new_id
    return new_seg


# ─── Self-test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("=== occ_head_plugin/modules.py  self-test ===\n")

    # 1. BevFeatureSlicer (identity)
    bevformer_conf = {
        'xbound': [-51.2, 51.2, 0.512],
        'ybound': [-51.2, 51.2, 0.512],
        'zbound': [-10.0, 10.0, 20.0],
    }
    slicer_id = BevFeatureSlicer('slicer_id', bevformer_conf, bevformer_conf)
    x_bev = F._from_data('x_bev', np.random.randn(1, 256, 200, 200).astype(np.float32))
    out = slicer_id(x_bev)
    assert list(out.shape) == [1, 256, 200, 200], f"BevFeatureSlicer identity: {out.shape}"
    logger.debug(f"  BevFeatureSlicer identity: {out.shape}  ✓")

    # 2. BevFeatureSlicer (grid sample)
    occflow_conf = {
        'xbound': [-50.0, 50.0, 0.5],
        'ybound': [-50.0, 50.0, 0.5],
        'zbound': [-10.0, 10.0, 20.0],
    }
    slicer_gs = BevFeatureSlicer('slicer_gs', bevformer_conf, occflow_conf)
    out_gs = slicer_gs(x_bev)
    assert list(out_gs.shape) == [1, 256, 200, 200], f"BevFeatureSlicer grid: {out_gs.shape}"
    logger.debug(f"  BevFeatureSlicer grid:     {out_gs.shape}  ✓")

    # 3. MLP
    mlp = MLP('mlp', 256, 256, 64, num_layers=3)
    x_mlp = F._from_data('x_mlp', np.random.randn(2, 10, 256).astype(np.float32))
    out_mlp = mlp(x_mlp)
    assert list(out_mlp.shape) == [2, 10, 64], f"MLP: {out_mlp.shape}"
    logger.debug(f"  MLP (256→64, 3 layers):    {out_mlp.shape}  ✓")

    # 4. SimpleConv2d
    sc = SimpleConv2d('sc', in_channels=256, out_channels=256,
                      conv_channels=256, num_conv=4)
    x_sc = F._from_data('x_sc', np.random.randn(1, 256, 200, 200).astype(np.float32))
    out_sc = sc(x_sc)
    assert list(out_sc.shape) == [1, 256, 200, 200], f"SimpleConv2d: {out_sc.shape}"
    logger.debug(f"  SimpleConv2d (4 convs):    {out_sc.shape}  ✓")

    # 5. CVT_Decoder
    cvt = CVT_Decoder('cvt', dim=256, blocks=[256, 256])
    x_cvt = F._from_data('x_cvt', np.random.randn(2, 5, 256, 50, 50).astype(np.float32))
    out_cvt = cvt(x_cvt)
    assert list(out_cvt.shape) == [2, 5, 256, 200, 200], f"CVT_Decoder: {out_cvt.shape}"
    logger.debug(f"  CVT_Decoder (2 blocks):    {out_cvt.shape}  ✓")

    # 6. UpsamplingAdd
    ua = UpsamplingAdd('ua', in_channels=256, out_channels=256)
    x_ua = F._from_data('x_ua', np.random.randn(1, 256, 25, 25).astype(np.float32))
    x_skip = F._from_data('x_skip', np.random.randn(1, 256, 50, 50).astype(np.float32))
    out_ua = ua(x_ua, x_skip)
    assert list(out_ua.shape) == [1, 256, 50, 50], f"UpsamplingAdd: {out_ua.shape}"
    logger.debug(f"  UpsamplingAdd:             {out_ua.shape}  ✓")

    # 7. Bottleneck (no downsample)
    bn_nods = Bottleneck('bn_nods', in_channels=256)
    x_bn = F._from_data('x_bn0', np.random.randn(1, 256, 50, 50).astype(np.float32))
    out_bn = bn_nods(x_bn)
    assert list(out_bn.shape) == [1, 256, 50, 50], f"Bottleneck no-ds: {out_bn.shape}"
    logger.debug(f"  Bottleneck (no ds):        {out_bn.shape}  ✓")

    # 8. Bottleneck (downsample, even)
    bn_ds = Bottleneck('bn_ds', in_channels=256, downsample=True)
    x_bn_e = F._from_data('x_bn1', np.random.randn(1, 256, 50, 50).astype(np.float32))
    out_bn_ds = bn_ds(x_bn_e)
    assert list(out_bn_ds.shape) == [1, 256, 25, 25], f"Bottleneck ds-even: {out_bn_ds.shape}"
    logger.debug(f"  Bottleneck (ds, even 50):  {out_bn_ds.shape}  ✓")

    # 9. Bottleneck (downsample, odd)
    bn_ds2 = Bottleneck('bn_ds2', in_channels=256, downsample=True)
    x_bn_o = F._from_data('x_bn2', np.random.randn(1, 256, 25, 25).astype(np.float32))
    out_bn_ds2 = bn_ds2(x_bn_o)
    # Conv path: input 25, kernel=3, stride=2, padding=1 => (25+2*1-3)/2+1 = 13
    # Pool path: pad to 26, MaxPool2d(2,2) => 13
    assert list(out_bn_ds2.shape) == [1, 256, 13, 13], f"Bottleneck ds-odd: {out_bn_ds2.shape}"
    logger.debug(f"  Bottleneck (ds, odd 25):   {out_bn_ds2.shape}  ✓")

    logger.info("\n=== All self-tests passed ===")
