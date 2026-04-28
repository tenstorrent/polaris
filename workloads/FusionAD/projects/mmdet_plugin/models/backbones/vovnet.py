#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSIM version of VoVNet backbone.

Converts the FusionAD VoVNet (Pytorch + mmcv/mmdet) to TTSIM.
All nn.Module subclasses become SimNN.Module, torch ops become F.* ops,
mmcv / mmdet decorators and registries are removed.

"""
# =============================================================================
# ORIGINAL TORCH CODE
# =============================================================================
#
# from collections import OrderedDict
# from mmcv.runner import BaseModule
# from mmdet.models.builder import BACKBONES
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.modules.batchnorm import _BatchNorm
#
#
# VoVNet19_slim_dw_eSE = {
#     'stem': [64, 64, 64],
#     'stage_conv_ch': [64, 80, 96, 112],
#     'stage_out_ch': [112, 256, 384, 512],
#     "layer_per_block": 3,
#     "block_per_stage": [1, 1, 1, 1],
#     "eSE": True,
#     "dw": True
# }
#
# VoVNet19_dw_eSE = {
#     'stem': [64, 64, 64],
#     "stage_conv_ch": [128, 160, 192, 224],
#     "stage_out_ch": [256, 512, 768, 1024],
#     "layer_per_block": 3,
#     "block_per_stage": [1, 1, 1, 1],
#     "eSE": True,
#     "dw": True
# }
#
# VoVNet19_slim_eSE = {
#     'stem': [64, 64, 128],
#     'stage_conv_ch': [64, 80, 96, 112],
#     'stage_out_ch': [112, 256, 384, 512],
#     'layer_per_block': 3,
#     'block_per_stage': [1, 1, 1, 1],
#     'eSE': True,
#     "dw": False
# }
#
# VoVNet19_eSE = {
#     'stem': [64, 64, 128],
#     "stage_conv_ch": [128, 160, 192, 224],
#     "stage_out_ch": [256, 512, 768, 1024],
#     "layer_per_block": 3,
#     "block_per_stage": [1, 1, 1, 1],
#     "eSE": True,
#     "dw": False
# }
#
# VoVNet39_eSE = {
#     'stem': [64, 64, 128],
#     "stage_conv_ch": [128, 160, 192, 224],
#     "stage_out_ch": [256, 512, 768, 1024],
#     "layer_per_block": 5,
#     "block_per_stage": [1, 1, 2, 2],
#     "eSE": True,
#     "dw": False
# }
#
# VoVNet57_eSE = {
#     'stem': [64, 64, 128],
#     "stage_conv_ch": [128, 160, 192, 224],
#     "stage_out_ch": [256, 512, 768, 1024],
#     "layer_per_block": 5,
#     "block_per_stage": [1, 1, 4, 3],
#     "eSE": True,
#     "dw": False
# }
#
# VoVNet99_eSE = {
#     'stem': [64, 64, 128],
#     "stage_conv_ch": [128, 160, 192, 224],
#     "stage_out_ch": [256, 512, 768, 1024],
#     "layer_per_block": 5,
#     "block_per_stage": [1, 3, 9, 3],
#     "eSE": True,
#     "dw": False
# }
#
# _STAGE_SPECS = {
#     "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
#     "V-19-dw-eSE": VoVNet19_dw_eSE,
#     "V-19-slim-eSE": VoVNet19_slim_eSE,
#     "V-19-eSE": VoVNet19_eSE,
#     "V-39-eSE": VoVNet39_eSE,
#     "V-57-eSE": VoVNet57_eSE,
#     "V-99-eSE": VoVNet99_eSE,
# }
#
#
# def dw_conv3x3(in_channels, out_channels, module_name, postfix, stride=1, kernel_size=3, padding=1):
#     """3x3 convolution with padding"""
#     return [
#         (
#             '{}_{}/dw_conv3x3'.format(module_name, postfix),
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 groups=out_channels,
#                 bias=False
#             )
#         ),
#         (
#             '{}_{}/pw_conv1x1'.format(module_name, postfix),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
#         ),
#         ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
#         ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
#     ]
#
#
# def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
#     """3x3 convolution with padding"""
#     return [
#         (
#             f"{module_name}_{postfix}/conv",
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 groups=groups,
#                 bias=False,
#             ),
#         ),
#         (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
#         (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
#     ]
#
#
# def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
#     """1x1 convolution with padding"""
#     return [
#         (
#             f"{module_name}_{postfix}/conv",
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 groups=groups,
#                 bias=False,
#             ),
#         ),
#         (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
#         (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
#     ]
#
#
# class Hsigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(Hsigmoid, self).__init__()
#         self.inplace = inplace
#
#     def forward(self, x):
#         return F.relu6(x + 3.0, inplace=self.inplace) / 6.0
#
#
# class eSEModule(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(eSEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
#         self.hsigmoid = Hsigmoid()
#
#     def forward(self, x):
#         input = x
#         x = self.avg_pool(x)
#         x = self.fc(x)
#         x = self.hsigmoid(x)
#         return input * x
#
#
# class _OSA_module(nn.Module):
#     def __init__(
#         self, in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE=False, identity=False, depthwise=False
#     ):
#
#         super(_OSA_module, self).__init__()
#
#         self.identity = identity
#         self.depthwise = depthwise
#         self.isReduced = False
#         self.layers = nn.ModuleList()
#         in_channel = in_ch
#         if self.depthwise and in_channel != stage_ch:
#             self.isReduced = True
#             self.conv_reduction = nn.Sequential(
#                 OrderedDict(conv1x1(in_channel, stage_ch, "{}_reduction".format(module_name), "0"))
#             )
#         for i in range(layer_per_block):
#             if self.depthwise:
#                 self.layers.append(nn.Sequential(OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))))
#             else:
#                 self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
#             in_channel = stage_ch
#
#         # feature aggregation
#         in_channel = in_ch + layer_per_block * stage_ch
#         self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat")))
#
#         self.ese = eSEModule(concat_ch)
#
#     def forward(self, x):
#
#         identity_feat = x
#
#         output = []
#         output.append(x)
#         if self.depthwise and self.isReduced:
#             x = self.conv_reduction(x)
#         for layer in self.layers:
#             x = layer(x)
#             output.append(x)
#
#         x = torch.cat(output, dim=1)
#         xt = self.concat(x)
#
#         xt = self.ese(xt)
#
#         if self.identity:
#             xt = xt + identity_feat
#
#         return xt
#
#
# class _OSA_stage(nn.Sequential):
#     def __init__(
#         self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False
#     ):
#
#         super(_OSA_stage, self).__init__()
#
#         if not stage_num == 2:
#             self.add_module("Pooling", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
#
#         if block_per_stage != 1:
#             SE = False
#         module_name = f"OSA{stage_num}_1"
#         self.add_module(
#             module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, SE, depthwise=depthwise)
#         )
#         for i in range(block_per_stage - 1):
#             if i != block_per_stage - 2:  # last block
#                 SE = False
#             module_name = f"OSA{stage_num}_{i + 2}"
#             self.add_module(
#                 module_name,
#                 _OSA_module(
#                     concat_ch,
#                     stage_ch,
#                     concat_ch,
#                     layer_per_block,
#                     module_name,
#                     SE,
#                     identity=True,
#                     depthwise=depthwise
#                 ),
#             )
#
#
# @BACKBONES.register_module()
# class VoVNet(BaseModule):
#     def __init__(self, spec_name, input_ch=3, out_features=None,
#                  frozen_stages=-1, norm_eval=True, pretrained=None, init_cfg=None):
#         """
#         Args:
#             input_ch(int) : the number of input channel
#             out_features (list[str]): name of the layers whose outputs should
#                 be returned in forward. Can be anything in "stem", "stage2" ...
#         """
#         super(VoVNet, self).__init__(init_cfg)
#         self.frozen_stages = frozen_stages
#         self.norm_eval = norm_eval
#
#         if isinstance(pretrained, str):
#             warnings.warn('DeprecationWarning: pretrained is deprecated, '
#                           'please use "init_cfg" instead')
#             self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
#         stage_specs = _STAGE_SPECS[spec_name]
#
#         stem_ch = stage_specs["stem"]
#         config_stage_ch = stage_specs["stage_conv_ch"]
#         config_concat_ch = stage_specs["stage_out_ch"]
#         block_per_stage = stage_specs["block_per_stage"]
#         layer_per_block = stage_specs["layer_per_block"]
#         SE = stage_specs["eSE"]
#         depthwise = stage_specs["dw"]
#
#         self._out_features = out_features
#
#         # Stem module
#         conv_type = dw_conv3x3 if depthwise else conv3x3
#         stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
#         stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
#         stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
#         self.add_module("stem", nn.Sequential((OrderedDict(stem))))
#         current_stirde = 4
#         self._out_feature_strides = {"stem": current_stirde, "stage2": current_stirde}
#         self._out_feature_channels = {"stem": stem_ch[2]}
#
#         stem_out_ch = [stem_ch[2]]
#         in_ch_list = stem_out_ch + config_concat_ch[:-1]
#         # OSA stages
#         self.stage_names = []
#         for i in range(4):  # num_stages
#             name = "stage%d" % (i + 2)  # stage 2 ... stage 5
#             self.stage_names.append(name)
#             self.add_module(
#                 name,
#                 _OSA_stage(
#                     in_ch_list[i],
#                     config_stage_ch[i],
#                     config_concat_ch[i],
#                     block_per_stage[i],
#                     layer_per_block,
#                     i + 2,
#                     SE,
#                     depthwise,
#                 ),
#             )
#
#             self._out_feature_channels[name] = config_concat_ch[i]
#             if not i == 0:
#                 self._out_feature_strides[name] = current_stirde = int(current_stirde * 2)
#
#         # initialize weights
#         # self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#
#     def forward(self, x):
#         outputs = {}
#         x = self.stem(x)
#         if "stem" in self._out_features:
#             outputs["stem"] = x
#         for name in self.stage_names:
#             x = getattr(self, name)(x)
#             if name in self._out_features:
#                 outputs[name] = x
#
#         return outputs
#
#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             m = getattr(self, 'stem')
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False
#
#         for i in range(1, self.frozen_stages + 1):
#             m = getattr(self, f'stage{i+1}')
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False
#
#     def train(self, mode=True):
#         """Convert the model into training mode while keep normalization layer
#         freezed."""
#         super(VoVNet, self).train(mode)
#         self._freeze_stages()
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 # trick: eval have effect on BatchNorm only
#                 if isinstance(m, _BatchNorm):
#                     m.eval()
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================




import os
import sys
from typing import Any

import numpy as np

_POLARIS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, _POLARIS_DIR)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ---------------------------------------------------------------------------
# Architecture configs (unchanged from original)
# ---------------------------------------------------------------------------

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_dw_eSE = {
    'stem': [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True
}

VoVNet19_slim_eSE = {
    'stem': [64, 64, 128],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE': True,
    "dw": False
}

VoVNet19_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False
}

VoVNet39_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False
}

VoVNet57_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False
}

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


# ---------------------------------------------------------------------------
# Helper: Conv-BN-ReLU building blocks
# ---------------------------------------------------------------------------

class DWConv3x3Block(SimNN.Module):
    """Depthwise-separable 3x3 conv block: DWConv3x3 → PWConv1x1 → BN → ReLU.

    Replaces the original ``dw_conv3x3`` helper which returned an OrderedDict
    list for ``nn.Sequential``.
    """

    def __init__(self, name, in_channels, out_channels, stride=1,
                 kernel_size=3, padding=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Depthwise conv
        self.dw_conv = F.Conv2d(
            f'{name}/dw_conv3x3', in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=out_channels, bias=False)
        # Pointwise conv
        self.pw_conv = F.Conv2d(
            f'{name}/pw_conv1x1', in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn = F.BatchNorm2d(f'{name}/pw_norm', out_channels)
        self.relu = F.Relu(f'{name}/pw_relu')

        super().link_op2module()

    def __call__(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def analytical_param_count(self, lvl=0):
        k = 3
        dw_params = self.out_channels * k * k  # depthwise (groups = out_channels)
        pw_params = self.in_channels * self.out_channels  # 1x1 pointwise
        bn_params = 2 * self.out_channels
        return dw_params + pw_params + bn_params


class Conv3x3Block(SimNN.Module):
    """Conv3x3 → BN → ReLU.

    Replaces the original ``conv3x3`` helper.
    """

    def __init__(self, name, in_channels, out_channels, stride=1,
                 groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups

        self.conv = F.Conv2d(
            f'{name}/conv', in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups, bias=False)
        self.bn = F.BatchNorm2d(f'{name}/norm', out_channels)
        self.relu = F.Relu(f'{name}/relu')

        super().link_op2module()

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def analytical_param_count(self, lvl=0):
        k = self.kernel_size
        conv_params = (self.in_channels * self.out_channels * k * k) // self.groups
        bn_params = 2 * self.out_channels
        return conv_params + bn_params


class Conv1x1Block(SimNN.Module):
    """Conv1x1 → BN → ReLU.

    Replaces the original ``conv1x1`` helper.
    """

    def __init__(self, name, in_channels, out_channels, stride=1,
                 groups=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.conv = F.Conv2d(
            f'{name}/conv', in_channels, out_channels,
            kernel_size=1, stride=stride, padding=0,
            groups=groups, bias=False)
        self.bn = F.BatchNorm2d(f'{name}/norm', out_channels)
        self.relu = F.Relu(f'{name}/relu')

        super().link_op2module()

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def analytical_param_count(self, lvl=0):
        conv_params = (self.in_channels * self.out_channels) // self.groups
        bn_params = 2 * self.out_channels
        return conv_params + bn_params


# ---------------------------------------------------------------------------
# Hsigmoid & eSE
# ---------------------------------------------------------------------------

class Hsigmoid(SimNN.Module):
    """Hard-sigmoid: relu6(x + 3) / 6."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.add_const = F.Add(f'{name}/add3')
        self.relu6 = F.Relu6(f'{name}/relu6')
        self.div_const = F.Div(f'{name}/div6')

        super().link_op2module()

    def __call__(self, x):
        three = F._from_data(f'{self.name}/three',
                             np.array([3.0], dtype=np.float32), is_const=True)
        six = F._from_data(f'{self.name}/six',
                           np.array([6.0], dtype=np.float32), is_const=True)
        x = self.add_const(x, three)
        x = self.relu6(x)
        x = self.div_const(x, six)
        return x

    def analytical_param_count(self, lvl=0):
        return 0


class eSEModule(SimNN.Module):
    """Effective Squeeze-Excitation module.

    AdaptiveAvgPool2d(1) → Conv1x1 → Hsigmoid → element-wise multiply.
    """

    def __init__(self, name, channel):
        super().__init__()
        self.name = name
        self.avg_pool = F.AdaptiveAvgPool2d(f'{name}/avg_pool', output_size=1)
        self.fc = F.Conv2d(f'{name}/fc', channel, channel,
                           kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid(f'{name}/hsigmoid')
        self.mul = F.Mul(f'{name}/mul')

        super().link_op2module()

    def __call__(self, x):
        inp = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return self.mul(inp, x)

    def analytical_param_count(self, lvl=0):
        # fc: Conv2d(channel, channel, 1) — has bias by default
        fc_params = self.fc.attrs.get('in_channels', 0) * self.fc.attrs.get('out_channels', 0) + \
                    self.fc.attrs.get('out_channels', 0)
        return fc_params


# ---------------------------------------------------------------------------
# OSA module & stage
# ---------------------------------------------------------------------------

class _OSA_module(SimNN.Module):
    """One-Shot Aggregation module."""

    def __init__(self, name, in_ch, stage_ch, concat_ch,
                 layer_per_block, SE=False, identity=False,
                 depthwise=False):
        super().__init__()
        self.name = name
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False

        # Optional channel reduction for depthwise path
        if self.depthwise and in_ch != stage_ch:
            self.isReduced = True
            self.conv_reduction = Conv1x1Block(
                f'{name}_reduction_0', in_ch, stage_ch)

        # OSA layers
        layers: list[SimNN.Module] = []
        in_channel = in_ch
        for i in range(layer_per_block):
            if self.depthwise:
                layers.append(DWConv3x3Block(
                    f'{name}_{i}', stage_ch, stage_ch))
            else:
                layers.append(Conv3x3Block(
                    f'{name}_{i}', in_channel, stage_ch))
            in_channel = stage_ch
        self.layers = SimNN.ModuleList(layers)

        # Feature aggregation: concat → 1x1 conv
        agg_in_ch = in_ch + layer_per_block * stage_ch
        self.concat_conv = Conv1x1Block(
            f'{name}_concat', agg_in_ch, concat_ch)

        # eSE
        self.ese = eSEModule(f'{name}/ese', concat_ch)

        # Residual add (when identity=True)
        self.add = F.Add(f'{name}/identity_add')

        # Concat for aggregation
        self.cat = F.ConcatX(f'{name}/cat', axis=1)

        super().link_op2module()

    def __call__(self, x):
        identity_feat = x

        output = [x]
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = self.cat(*output)
        xt = self.concat_conv(x)

        xt = self.ese(xt)

        if self.identity:
            xt = self.add(xt, identity_feat)

        return xt

    def analytical_param_count(self, lvl=0):
        total = 0
        if self.depthwise and self.isReduced:
            total += self.conv_reduction.analytical_param_count(lvl + 1)
        for layer in self.layers:
            total += layer.analytical_param_count(lvl + 1)  # type: ignore[attr-defined]
        total += self.concat_conv.analytical_param_count(lvl + 1)
        total += self.ese.analytical_param_count(lvl + 1)
        return total


class _OSA_stage(SimNN.Module):
    """One stage of VoVNet: optional MaxPool → sequence of _OSA_modules."""

    def __init__(self, name, in_ch, stage_ch, concat_ch,
                 block_per_stage, layer_per_block, stage_num,
                 SE=False, depthwise=False):
        super().__init__()
        self.name = name
        self.has_pool = (stage_num != 2)

        if self.has_pool:
            self.pool = F.MaxPool2d(f'{name}/pool', kernel_size=3,
                                    stride=2, ceil_mode=True)

        blocks = []
        # First block
        module_name = f'OSA{stage_num}_1'
        se_flag = SE if block_per_stage == 1 else False
        blocks.append(_OSA_module(
            module_name, in_ch, stage_ch, concat_ch,
            layer_per_block, SE=se_flag, depthwise=depthwise))

        # Remaining blocks (with identity shortcut)
        for i in range(block_per_stage - 1):
            se_flag = SE if (i == block_per_stage - 2) else False
            module_name = f'OSA{stage_num}_{i + 2}'
            blocks.append(_OSA_module(
                module_name, concat_ch, stage_ch, concat_ch,
                layer_per_block, SE=se_flag, identity=True,
                depthwise=depthwise))

        self.blocks = SimNN.ModuleList(blocks)

        super().link_op2module()

    def __call__(self, x):
        if self.has_pool:
            x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        return x

    def analytical_param_count(self, lvl=0):
        total = 0
        for block in self.blocks:
            total += block.analytical_param_count(lvl + 1)  # type: ignore[attr-defined]
        return total


# ---------------------------------------------------------------------------
# VoVNet backbone
# ---------------------------------------------------------------------------

class VoVNet(SimNN.Module):
    """VoVNet backbone converted to TTSIM.

    Args:
        spec_name (str): Architecture variant key (e.g. ``"V-99-eSE"``).
        input_ch (int): Number of input channels (default 3).
        out_features (list[str]): Names of layers whose outputs to return
            (e.g. ``["stage2", "stage3", "stage4", "stage5"]``).
    """

    def __init__(self, spec_name, input_ch=3, out_features=None):
        super().__init__()
        self.name = 'VoVNet'

        stage_specs: Any = _STAGE_SPECS[spec_name]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features or []

        # --- Stem ---
        conv_type_cls = DWConv3x3Block if depthwise else Conv3x3Block
        self.stem_1 = Conv3x3Block('stem_1', input_ch, stem_ch[0], stride=2)
        self.stem_2 = conv_type_cls('stem_2', stem_ch[0], stem_ch[1], stride=1)
        self.stem_3 = conv_type_cls('stem_3', stem_ch[1], stem_ch[2], stride=2)

        current_stride = 4
        self._out_feature_strides = {
            "stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}

        # --- OSA stages ---
        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]

        self.stage_names = []
        stages = []
        for i in range(4):
            sname = f'stage{i + 2}'
            self.stage_names.append(sname)
            stages.append(_OSA_stage(
                sname, in_ch_list[i], config_stage_ch[i],
                config_concat_ch[i], block_per_stage[i],
                layer_per_block, i + 2, SE, depthwise))
            self._out_feature_channels[sname] = config_concat_ch[i]
            if i != 0:
                current_stride = int(current_stride * 2)
                self._out_feature_strides[sname] = current_stride

        self.stages = SimNN.ModuleList(stages)

        super().link_op2module()

    def __call__(self, x):
        outputs = {}

        # Stem
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        if "stem" in self._out_features:
            outputs["stem"] = x

        # Stages
        for sname, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if sname in self._out_features:
                outputs[sname] = x

        return outputs

    def analytical_param_count(self, lvl=0):
        total = 0
        total += self.stem_1.analytical_param_count(lvl + 1)
        total += self.stem_2.analytical_param_count(lvl + 1)
        total += self.stem_3.analytical_param_count(lvl + 1)
        for stage in self.stages:
            total += stage.analytical_param_count(lvl + 1)  # type: ignore[attr-defined]
        return total
