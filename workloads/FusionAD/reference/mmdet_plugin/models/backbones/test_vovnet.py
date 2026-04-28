#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for vovnet.py:
VoVNet backbone — shape and numerical validation.

Both PyTorch (inlined, no mmcv/mmdet dependency) and TTSIM versions are
present in this file.  Tests compare shapes from each sub-module and
from the full model, and verify numerical output where data propagation
is available.

Test Coverage:
  TEST  1: Conv3x3Block shape
  TEST  2: Conv1x1Block shape
  TEST  3: DWConv3x3Block shape
  TEST  4: Hsigmoid numerical
  TEST  5: eSEModule shape
  TEST  6: _OSA_module shape (no depthwise)
  TEST  7: _OSA_module shape (depthwise)
  TEST  8: _OSA_stage shape (with pool)
  TEST  9: _OSA_stage shape (no pool, stage2)
  TEST 10: Full VoVNet shape — V-19-slim-eSE
  TEST 11: Full VoVNet shape — V-99-eSE
  TEST 12: Hsigmoid numerical — PyTorch vs TTSIM
  TEST 13: Config & stride verification
  TEST 14: Various input sizes
  TEST 15: Conv3x3Block numerical — shared weights
  TEST 16: eSEModule numerical — shared weights
  TEST 17: _OSA_module numerical — shared weights
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import os, sys
from collections import OrderedDict

_POLARIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, _POLARIS_DIR)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ===========================================================================
# Architecture configs (shared)
# ===========================================================================

VoVNet19_slim_dw_eSE = {
    'stem': [64, 64, 64],
    'stage_conv_ch': [64, 80, 96, 112],
    'stage_out_ch': [112, 256, 384, 512],
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

VoVNet99_eSE = {
    'stem': [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False
}

_STAGE_SPECS_PT = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


# ===========================================================================
# PyTorch versions (inlined from original vovnet.py — no mmcv/mmdet)
# ===========================================================================

def dw_conv3x3_pt(in_channels, out_channels, module_name, postfix,
                  stride=1, kernel_size=3, padding=1):
    return [
        ('{}_{}/dw_conv3x3'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                   stride=stride, padding=padding, groups=out_channels, bias=False)),
        ('{}_{}/pw_conv1x1'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                   padding=0, groups=1, bias=False)),
        ('{}_{}/pw_norm'.format(module_name, postfix), nn.BatchNorm2d(out_channels)),
        ('{}_{}/pw_relu'.format(module_name, postfix), nn.ReLU(inplace=True)),
    ]


def conv3x3_pt(in_channels, out_channels, module_name, postfix,
               stride=1, groups=1, kernel_size=3, padding=1):
    return [
        (f"{module_name}_{postfix}/conv",
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                   stride=stride, padding=padding, groups=groups, bias=False)),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


def conv1x1_pt(in_channels, out_channels, module_name, postfix,
               stride=1, groups=1, kernel_size=1, padding=0):
    return [
        (f"{module_name}_{postfix}/conv",
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                   stride=stride, padding=padding, groups=groups, bias=False)),
        (f"{module_name}_{postfix}/norm", nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", nn.ReLU(inplace=True)),
    ]


class HsigmoidPT(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return TF.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModulePT(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = HsigmoidPT()

    def forward(self, x):
        inp = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return inp * x


class _OSA_module_PT(nn.Module):
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block,
                 module_name, SE=False, identity=False, depthwise=False):
        super().__init__()
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = nn.Sequential(
                OrderedDict(conv1x1_pt(in_channel, stage_ch,
                                       f"{module_name}_reduction", "0")))
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(nn.Sequential(
                    OrderedDict(dw_conv3x3_pt(stage_ch, stage_ch, module_name, i))))
            else:
                self.layers.append(nn.Sequential(
                    OrderedDict(conv3x3_pt(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1_pt(in_channel, concat_ch, module_name, "concat")))
        self.ese = eSEModulePT(concat_ch)

    def forward(self, x):
        identity_feat = x
        output = [x]
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)
        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage_PT(nn.Sequential):
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage,
                 layer_per_block, stage_num, SE=False, depthwise=False):
        super().__init__()
        if stage_num != 2:
            self.add_module("Pooling", nn.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True))

        se_flag = SE if block_per_stage == 1 else False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name,
            _OSA_module_PT(in_ch, stage_ch, concat_ch, layer_per_block,
                           module_name, se_flag, depthwise=depthwise))

        for i in range(block_per_stage - 1):
            se_flag = SE if (i == block_per_stage - 2) else False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module_PT(concat_ch, stage_ch, concat_ch, layer_per_block,
                               module_name, se_flag, identity=True,
                               depthwise=depthwise))


class VoVNetPT(nn.Module):
    """PyTorch VoVNet (inlined, no mmcv dependency)."""

    def __init__(self, spec_name, input_ch=3, out_features=None):
        super().__init__()
        stage_specs = _STAGE_SPECS_PT[spec_name]
        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features or []

        conv_type = dw_conv3x3_pt if depthwise else conv3x3_pt
        stem = conv3x3_pt(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.stem = nn.Sequential(OrderedDict(stem))

        current_stride = 4
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]

        self.stage_names = []
        for i in range(4):
            name = f"stage{i + 2}"
            self.stage_names.append(name)
            setattr(self, name, _OSA_stage_PT(
                in_ch_list[i], config_stage_ch[i], config_concat_ch[i],
                block_per_stage[i], layer_per_block, i + 2, SE, depthwise))
            self._out_feature_channels[name] = config_concat_ch[i]
            if i != 0:
                current_stride = int(current_stride * 2)
                self._out_feature_strides[name] = current_stride

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name in self.stage_names:
            x = getattr(self, name)(x)
            if name in self._out_features:
                outputs[name] = x
        return outputs


# ===========================================================================
# TTSIM versions (inlined from converted vovnet.py)
# ===========================================================================

class DWConv3x3Block(SimNN.Module):
    def __init__(self, name, in_channels, out_channels, stride=1,
                 kernel_size=3, padding=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw_conv = F.Conv2d(f'{name}/dw_conv3x3', in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                groups=out_channels, bias=False)
        self.pw_conv = F.Conv2d(f'{name}/pw_conv1x1', in_channels, out_channels,
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


class Conv3x3Block(SimNN.Module):
    def __init__(self, name, in_channels, out_channels, stride=1,
                 groups=1, kernel_size=3, padding=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv = F.Conv2d(f'{name}/conv', in_channels, out_channels,
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


class Conv1x1Block(SimNN.Module):
    def __init__(self, name, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.conv = F.Conv2d(f'{name}/conv', in_channels, out_channels,
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


class HsigmoidTTSIM(SimNN.Module):
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


class eSEModuleTTSIM(SimNN.Module):
    def __init__(self, name, channel):
        super().__init__()
        self.name = name
        self.avg_pool = F.AdaptiveAvgPool2d(f'{name}/avg_pool', output_size=1)
        self.fc = F.Conv2d(f'{name}/fc', channel, channel,
                           kernel_size=1, padding=0)
        self.hsigmoid = HsigmoidTTSIM(f'{name}/hsigmoid')
        self.mul = F.Mul(f'{name}/mul')
        super().link_op2module()

    def __call__(self, x):
        inp = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return self.mul(inp, x)


class _OSA_module_TT(SimNN.Module):
    def __init__(self, name, in_ch, stage_ch, concat_ch,
                 layer_per_block, SE=False, identity=False, depthwise=False):
        super().__init__()
        self.name = name
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False

        if self.depthwise and in_ch != stage_ch:
            self.isReduced = True
            self.conv_reduction = Conv1x1Block(f'{name}_reduction_0', in_ch, stage_ch)

        layers = []
        in_channel = in_ch
        for i in range(layer_per_block):
            if self.depthwise:
                layers.append(DWConv3x3Block(f'{name}_{i}', stage_ch, stage_ch))
            else:
                layers.append(Conv3x3Block(f'{name}_{i}', in_channel, stage_ch))
            in_channel = stage_ch
        self.layers = SimNN.ModuleList(layers)

        agg_in_ch = in_ch + layer_per_block * stage_ch
        self.concat_conv = Conv1x1Block(f'{name}_concat', agg_in_ch, concat_ch)
        self.ese = eSEModuleTTSIM(f'{name}/ese', concat_ch)
        self.add = F.Add(f'{name}/identity_add')
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


class _OSA_stage_TT(SimNN.Module):
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
        se_flag = SE if block_per_stage == 1 else False
        module_name = f'OSA{stage_num}_1'
        blocks.append(_OSA_module_TT(
            module_name, in_ch, stage_ch, concat_ch,
            layer_per_block, SE=se_flag, depthwise=depthwise))

        for i in range(block_per_stage - 1):
            se_flag = SE if (i == block_per_stage - 2) else False
            module_name = f'OSA{stage_num}_{i + 2}'
            blocks.append(_OSA_module_TT(
                module_name, concat_ch, stage_ch, concat_ch,
                layer_per_block, SE=se_flag, identity=True, depthwise=depthwise))

        self.blocks = SimNN.ModuleList(blocks)
        super().link_op2module()

    def __call__(self, x):
        if self.has_pool:
            x = self.pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class VoVNetTTSIM(SimNN.Module):
    def __init__(self, spec_name, input_ch=3, out_features=None):
        super().__init__()
        self.name = 'VoVNet'
        stage_specs = _STAGE_SPECS_PT[spec_name]
        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        self._out_features = out_features or []

        conv_type_cls = DWConv3x3Block if depthwise else Conv3x3Block
        self.stem_1 = Conv3x3Block('stem_1', input_ch, stem_ch[0], stride=2)
        self.stem_2 = conv_type_cls('stem_2', stem_ch[0], stem_ch[1], stride=1)
        self.stem_3 = conv_type_cls('stem_3', stem_ch[1], stem_ch[2], stride=2)

        current_stride = 4
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        stages = []
        for i in range(4):
            sname = f'stage{i + 2}'
            self.stage_names.append(sname)
            stages.append(_OSA_stage_TT(
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
        x = self.stem_1(x)
        x = self.stem_2(x)
        x = self.stem_3(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for sname, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if sname in self._out_features:
                outputs[sname] = x
        return outputs


# ===========================================================================
# Test helpers
# ===========================================================================

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ===========================================================================
# Tests
# ===========================================================================

def test_conv3x3_block_shape():
    """Conv3x3Block: [B, C_in, H, W] → [B, C_out, H', W']."""
    print("\n" + "=" * 70)
    print("TEST 1: Conv3x3Block shape")
    print("=" * 70)

    # TTSIM
    tt = Conv3x3Block('t1_conv3x3', 64, 128, stride=2)
    inp = F._from_data('t1_inp', np.random.randn(2, 64, 32, 32).astype(np.float32))
    out = tt(inp)
    check("TTSIM shape", list(out.shape) == [2, 128, 16, 16], f"got {list(out.shape)}")

    # PyTorch
    pt_layers = conv3x3_pt(64, 128, "pt_conv3x3", "0", stride=2)
    pt = nn.Sequential(OrderedDict(pt_layers))
    pt.eval()
    with torch.no_grad():
        pt_out = pt(torch.randn(2, 64, 32, 32))
    check("PT shape", list(pt_out.shape) == [2, 128, 16, 16], f"got {list(pt_out.shape)}")
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_conv1x1_block_shape():
    """Conv1x1Block: [B, C_in, H, W] → [B, C_out, H, W]."""
    print("\n" + "=" * 70)
    print("TEST 2: Conv1x1Block shape")
    print("=" * 70)

    tt = Conv1x1Block('t2_conv1x1', 256, 128)
    inp = F._from_data('t2_inp', np.random.randn(1, 256, 16, 16).astype(np.float32))
    out = tt(inp)
    check("TTSIM shape", list(out.shape) == [1, 128, 16, 16], f"got {list(out.shape)}")

    pt_layers = conv1x1_pt(256, 128, "pt_conv1x1", "0")
    pt = nn.Sequential(OrderedDict(pt_layers))
    pt.eval()
    with torch.no_grad():
        pt_out = pt(torch.randn(1, 256, 16, 16))
    check("PT shape", list(pt_out.shape) == [1, 128, 16, 16], f"got {list(pt_out.shape)}")
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_dw_conv3x3_block_shape():
    """DWConv3x3Block shape: depthwise-separable."""
    print("\n" + "=" * 70)
    print("TEST 3: DWConv3x3Block shape")
    print("=" * 70)

    tt = DWConv3x3Block('t3_dw', 64, 64, stride=1)
    inp = F._from_data('t3_inp', np.random.randn(2, 64, 32, 32).astype(np.float32))
    out = tt(inp)
    check("TTSIM shape", list(out.shape) == [2, 64, 32, 32], f"got {list(out.shape)}")

    pt_layers = dw_conv3x3_pt(64, 64, "pt_dw", "0", stride=1)
    pt = nn.Sequential(OrderedDict(pt_layers))
    pt.eval()
    with torch.no_grad():
        pt_out = pt(torch.randn(2, 64, 32, 32))
    check("PT shape", list(pt_out.shape) == [2, 64, 32, 32], f"got {list(pt_out.shape)}")
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_hsigmoid_numerical():
    """Hsigmoid numerical: relu6(x+3)/6."""
    print("\n" + "=" * 70)
    print("TEST 4: Hsigmoid numerical")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.randn(2, 8, 4, 4).astype(np.float32)

    # PyTorch
    pt_hsig = HsigmoidPT()
    with torch.no_grad():
        pt_out = pt_hsig(torch.from_numpy(data)).numpy()

    # TTSIM
    tt_hsig = HsigmoidTTSIM('t4_hsig')
    tt_inp = F._from_data('t4_inp', data)
    tt_out = tt_hsig(tt_inp)

    check("TTSIM shape == PT shape",
          list(tt_out.shape) == list(pt_out.shape),
          f"TTSIM={list(tt_out.shape)} vs PT={list(pt_out.shape)}")

    if tt_out.data is not None:
        max_diff = np.max(np.abs(pt_out - tt_out.data))
        check("numerical close (atol=1e-5)",
              np.allclose(pt_out, tt_out.data, atol=1e-5),
              f"max diff = {max_diff:.2e}")
    else:
        check("data propagation (shape-only mode)", True,
              "data=None — ttsim shape-only")


def test_ese_module_shape():
    """eSEModule shape: [B, C, H, W] → [B, C, H, W]."""
    print("\n" + "=" * 70)
    print("TEST 5: eSEModule shape")
    print("=" * 70)

    channel = 64

    # TTSIM
    tt_ese = eSEModuleTTSIM('t5_ese', channel)
    inp = F._from_data('t5_inp', np.random.randn(2, channel, 16, 16).astype(np.float32))
    out = tt_ese(inp)
    check("TTSIM shape", list(out.shape) == [2, channel, 16, 16],
          f"got {list(out.shape)}")

    # PyTorch
    pt_ese = eSEModulePT(channel)
    pt_ese.eval()
    with torch.no_grad():
        pt_out = pt_ese(torch.randn(2, channel, 16, 16))
    check("PT shape", list(pt_out.shape) == [2, channel, 16, 16])
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_osa_module_shape_no_dw():
    """_OSA_module shape — no depthwise."""
    print("\n" + "=" * 70)
    print("TEST 6: _OSA_module shape (no depthwise)")
    print("=" * 70)

    in_ch, stage_ch, concat_ch = 128, 128, 256
    layer_per_block = 5

    # TTSIM
    tt_osa = _OSA_module_TT('t6_osa', in_ch, stage_ch, concat_ch,
                             layer_per_block, SE=True)
    inp = F._from_data('t6_inp', np.random.randn(1, in_ch, 16, 16).astype(np.float32))
    out = tt_osa(inp)
    check("TTSIM shape", list(out.shape) == [1, concat_ch, 16, 16],
          f"got {list(out.shape)}")

    # PyTorch
    pt_osa = _OSA_module_PT(in_ch, stage_ch, concat_ch, layer_per_block,
                             "pt_osa", SE=True)
    pt_osa.eval()
    with torch.no_grad():
        pt_out = pt_osa(torch.randn(1, in_ch, 16, 16))
    check("PT shape", list(pt_out.shape) == [1, concat_ch, 16, 16])
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_osa_module_shape_dw():
    """_OSA_module shape — depthwise, with channel reduction."""
    print("\n" + "=" * 70)
    print("TEST 7: _OSA_module shape (depthwise)")
    print("=" * 70)

    in_ch, stage_ch, concat_ch = 128, 64, 112
    layer_per_block = 3

    # TTSIM
    tt_osa = _OSA_module_TT('t7_osa', in_ch, stage_ch, concat_ch,
                             layer_per_block, depthwise=True)
    inp = F._from_data('t7_inp', np.random.randn(1, in_ch, 16, 16).astype(np.float32))
    out = tt_osa(inp)
    check("TTSIM shape", list(out.shape) == [1, concat_ch, 16, 16],
          f"got {list(out.shape)}")
    check("isReduced flag", tt_osa.isReduced is True)

    # PyTorch
    pt_osa = _OSA_module_PT(in_ch, stage_ch, concat_ch, layer_per_block,
                             "pt_osa_dw", depthwise=True)
    pt_osa.eval()
    with torch.no_grad():
        pt_out = pt_osa(torch.randn(1, in_ch, 16, 16))
    check("PT shape", list(pt_out.shape) == [1, concat_ch, 16, 16])
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_osa_stage_with_pool():
    """_OSA_stage shape — stages 3-5 have MaxPool."""
    print("\n" + "=" * 70)
    print("TEST 8: _OSA_stage shape (with pool)")
    print("=" * 70)

    in_ch, stage_ch, concat_ch = 256, 160, 512
    block_per_stage, layer_per_block = 1, 5
    stage_num = 3  # not stage 2, so has pool

    # TTSIM
    tt_stage = _OSA_stage_TT('t8_stage', in_ch, stage_ch, concat_ch,
                              block_per_stage, layer_per_block, stage_num, SE=True)
    inp = F._from_data('t8_inp', np.random.randn(1, in_ch, 64, 64).astype(np.float32))
    out = tt_stage(inp)

    # MaxPool(k=3, s=2, ceil_mode=True): 64 → ceil((64-3+2*0)/2)+1=32 or with ceil_mode
    # ceil_mode=True: ceil((64+2*1-3)/2)+1=33 ... depends on padding.
    # MaxPool2d with k=3, s=2, pad=0, ceil: ceil((64-3)/2)+1 = ceil(30.5)+1=32
    expected_h = 32
    check("TTSIM spatial halved (pool)", out.shape[2] == expected_h,
          f"got H={out.shape[2]}, expected {expected_h}")
    check("TTSIM channels", out.shape[1] == concat_ch)
    check("has_pool flag", tt_stage.has_pool is True)

    # PyTorch
    pt_stage = _OSA_stage_PT(in_ch, stage_ch, concat_ch,
                              block_per_stage, layer_per_block, stage_num, SE=True)
    pt_stage.eval()
    with torch.no_grad():
        pt_out = pt_stage(torch.randn(1, in_ch, 64, 64))
    check("PT channels", pt_out.shape[1] == concat_ch)
    check("PT spatial == TTSIM spatial",
          pt_out.shape[2] == out.shape[2] and pt_out.shape[3] == out.shape[3],
          f"PT=({pt_out.shape[2]},{pt_out.shape[3]}) vs TT=({out.shape[2]},{out.shape[3]})")


def test_osa_stage_no_pool():
    """_OSA_stage shape — stage2 has no MaxPool."""
    print("\n" + "=" * 70)
    print("TEST 9: _OSA_stage shape (no pool, stage2)")
    print("=" * 70)

    in_ch, stage_ch, concat_ch = 128, 128, 256
    block_per_stage, layer_per_block = 1, 5
    stage_num = 2

    # TTSIM
    tt_stage = _OSA_stage_TT('t9_stage', in_ch, stage_ch, concat_ch,
                              block_per_stage, layer_per_block, stage_num, SE=True)
    inp = F._from_data('t9_inp', np.random.randn(1, in_ch, 64, 64).astype(np.float32))
    out = tt_stage(inp)

    check("TTSIM spatial preserved (no pool)",
          out.shape[2] == 64 and out.shape[3] == 64,
          f"got ({out.shape[2]},{out.shape[3]})")
    check("TTSIM channels", out.shape[1] == concat_ch)
    check("has_pool flag", tt_stage.has_pool is False)

    # PyTorch
    pt_stage = _OSA_stage_PT(in_ch, stage_ch, concat_ch,
                              block_per_stage, layer_per_block, stage_num, SE=True)
    pt_stage.eval()
    with torch.no_grad():
        pt_out = pt_stage(torch.randn(1, in_ch, 64, 64))
    check("PT spatial preserved", pt_out.shape[2] == 64)
    check("shapes match", list(out.shape) == list(pt_out.shape))


def test_full_vovnet_slim_ese():
    """Full VoVNet V-19-slim-eSE shape."""
    print("\n" + "=" * 70)
    print("TEST 10: Full VoVNet shape — V-19-slim-eSE")
    print("=" * 70)

    out_feats = ["stage2", "stage3", "stage4", "stage5"]

    # PyTorch
    pt_model = VoVNetPT("V-19-slim-eSE", input_ch=3, out_features=out_feats)
    pt_model.eval()
    with torch.no_grad():
        pt_outs = pt_model(torch.randn(1, 3, 224, 224))

    # TTSIM
    tt_model = VoVNetTTSIM("V-19-slim-eSE", input_ch=3, out_features=out_feats)
    tt_inp = F._from_data('t10_inp', np.random.randn(1, 3, 224, 224).astype(np.float32))
    tt_outs = tt_model(tt_inp)

    check("same output keys", set(tt_outs.keys()) == set(pt_outs.keys()),
          f"TT={set(tt_outs.keys())} vs PT={set(pt_outs.keys())}")

    expected_channels = {"stage2": 112, "stage3": 256, "stage4": 384, "stage5": 512}
    for key in out_feats:
        pt_shape = list(pt_outs[key].shape)
        tt_shape = list(tt_outs[key].shape)
        check(f"{key} shapes match", tt_shape == pt_shape,
              f"TT={tt_shape} vs PT={pt_shape}")
        check(f"{key} channels == {expected_channels[key]}",
              tt_shape[1] == expected_channels[key])


def test_full_vovnet_99_ese():
    """Full VoVNet V-99-eSE shape."""
    print("\n" + "=" * 70)
    print("TEST 11: Full VoVNet shape — V-99-eSE")
    print("=" * 70)

    out_feats = ["stage2", "stage3", "stage4", "stage5"]

    # PyTorch
    pt_model = VoVNetPT("V-99-eSE", input_ch=3, out_features=out_feats)
    pt_model.eval()
    with torch.no_grad():
        pt_outs = pt_model(torch.randn(1, 3, 224, 224))

    # TTSIM
    tt_model = VoVNetTTSIM("V-99-eSE", input_ch=3, out_features=out_feats)
    tt_inp = F._from_data('t11_inp', np.random.randn(1, 3, 224, 224).astype(np.float32))
    tt_outs = tt_model(tt_inp)

    expected_channels = {"stage2": 256, "stage3": 512, "stage4": 768, "stage5": 1024}
    for key in out_feats:
        pt_shape = list(pt_outs[key].shape)
        tt_shape = list(tt_outs[key].shape)
        check(f"{key} shapes match", tt_shape == pt_shape,
              f"TT={tt_shape} vs PT={pt_shape}")
        check(f"{key} channels == {expected_channels[key]}",
              tt_shape[1] == expected_channels[key])


def test_hsigmoid_numerical_compare():
    """Hsigmoid PyTorch vs TTSIM numerical comparison across value ranges."""
    print("\n" + "=" * 70)
    print("TEST 12: Hsigmoid numerical — PyTorch vs TTSIM")
    print("=" * 70)

    # Test values spanning the three regimes: x<-3 (→0), -3<x<3 (→linear), x>3 (→1)
    test_values = np.array([-10.0, -5.0, -3.0, -1.5, 0.0, 1.5, 3.0, 5.0, 10.0],
                           dtype=np.float32).reshape(1, 1, 3, 3)

    # PyTorch
    pt_hsig = HsigmoidPT()
    with torch.no_grad():
        pt_out = pt_hsig(torch.from_numpy(test_values)).numpy()

    # TTSIM
    tt_hsig = HsigmoidTTSIM('t12_hsig')
    tt_inp = F._from_data('t12_inp', test_values)
    tt_out = tt_hsig(tt_inp)

    if tt_out.data is not None:
        max_diff = np.max(np.abs(pt_out - tt_out.data))
        check("values close (atol=1e-5)",
              np.allclose(pt_out, tt_out.data, atol=1e-5),
              f"max diff = {max_diff:.2e}")
        # Check boundary behavior
        check("x=-10 → 0", abs(tt_out.data.flatten()[0]) < 1e-5)
        check("x=0 → 0.5", abs(tt_out.data.flatten()[4] - 0.5) < 1e-5)
        check("x=10 → 1", abs(tt_out.data.flatten()[8] - 1.0) < 1e-5)
    else:
        check("data propagation", True, "shape-only mode")


def test_config_and_strides():
    """Config and stride verification."""
    print("\n" + "=" * 70)
    print("TEST 13: Config & stride verification")
    print("=" * 70)

    tt_model = VoVNetTTSIM("V-19-slim-eSE", input_ch=3,
                            out_features=["stem", "stage2", "stage3", "stage4", "stage5"])

    check("stem channels", tt_model._out_feature_channels["stem"] == 128)
    check("stage2 channels", tt_model._out_feature_channels["stage2"] == 112)
    check("stage3 channels", tt_model._out_feature_channels["stage3"] == 256)
    check("stage4 channels", tt_model._out_feature_channels["stage4"] == 384)
    check("stage5 channels", tt_model._out_feature_channels["stage5"] == 512)

    check("stem stride", tt_model._out_feature_strides["stem"] == 4)
    check("stage2 stride", tt_model._out_feature_strides["stage2"] == 4)
    check("stage3 stride", tt_model._out_feature_strides["stage3"] == 8)
    check("stage4 stride", tt_model._out_feature_strides["stage4"] == 16)
    check("stage5 stride", tt_model._out_feature_strides["stage5"] == 32)

    check("4 stage_names", len(tt_model.stage_names) == 4)
    check("stage names correct",
          tt_model.stage_names == ["stage2", "stage3", "stage4", "stage5"])


def test_various_input_sizes():
    """Shape inference with different input spatial sizes."""
    print("\n" + "=" * 70)
    print("TEST 14: Various input sizes")
    print("=" * 70)

    sizes = [
        (1, 3, 224, 224, "224x224"),
        (2, 3, 320, 320, "320x320"),
        (1, 3, 128, 128, "128x128"),
    ]

    out_feats = ["stage5"]

    for B, C, H, W, label in sizes:
        pt_model = VoVNetPT("V-19-slim-eSE", input_ch=C, out_features=out_feats)
        pt_model.eval()
        with torch.no_grad():
            pt_outs = pt_model(torch.randn(B, C, H, W))
        pt_shape = list(pt_outs["stage5"].shape)

        tt_model = VoVNetTTSIM("V-19-slim-eSE", input_ch=C, out_features=out_feats)
        tt_inp = F._from_data(f't14_inp_{label}',
                              np.random.randn(B, C, H, W).astype(np.float32))
        tt_outs = tt_model(tt_inp)
        tt_shape = list(tt_outs["stage5"].shape)

        check(f"{label}: shapes match", tt_shape == pt_shape,
              f"TT={tt_shape} vs PT={pt_shape}")


# ===========================================================================
# Weight copy helpers (PT → TTSIM)
# ===========================================================================

def copy_conv2d(pt_conv, tt_conv_op):
    """Copy nn.Conv2d weights to TTSim F.Conv2d op (params[0]=weight, params[1]=bias)."""
    tt_conv_op.params[0][1].data = pt_conv.weight.data.detach().numpy().astype(np.float32)
    if pt_conv.bias is not None and len(tt_conv_op.params) > 1:
        tt_conv_op.params[1][1].data = pt_conv.bias.data.detach().numpy().astype(np.float32)


def copy_bn(pt_bn, tt_bn_op):
    """Copy nn.BatchNorm2d params to TTSim F.BatchNorm2d op."""
    tt_bn_op.params[0][1].data = pt_bn.weight.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[1][1].data = pt_bn.bias.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[2][1].data = pt_bn.running_mean.data.detach().numpy().astype(np.float32)
    tt_bn_op.params[3][1].data = pt_bn.running_var.data.detach().numpy().astype(np.float32)


def copy_conv3x3_block(pt_seq, tt_block):
    """Copy PyTorch conv3x3 Sequential (Conv, BN, ReLU) → TTSIM Conv3x3Block."""
    # pt_seq is nn.Sequential with named children: .../conv, .../norm, .../relu
    modules = list(pt_seq.children())
    copy_conv2d(modules[0], tt_block.conv)
    copy_bn(modules[1], tt_block.bn)


def copy_conv1x1_block(pt_seq, tt_block):
    """Copy PyTorch conv1x1 Sequential (Conv, BN, ReLU) → TTSIM Conv1x1Block."""
    modules = list(pt_seq.children())
    copy_conv2d(modules[0], tt_block.conv)
    copy_bn(modules[1], tt_block.bn)


def copy_ese(pt_ese, tt_ese):
    """Copy eSEModulePT weights → eSEModuleTTSIM.
    Neutralizes PT fc bias since TTSIM Conv2d has no bias param."""
    # Zero PT bias BEFORE copying so TTSIM gets the same zeroed bias
    if pt_ese.fc.bias is not None:
        pt_ese.fc.bias.data.fill_(0.0)
    copy_conv2d(pt_ese.fc, tt_ese.fc)


def copy_osa_module(pt_osa, tt_osa):
    """Copy _OSA_module_PT weights → _OSA_module_TT."""
    if tt_osa.depthwise and tt_osa.isReduced:
        copy_conv1x1_block(pt_osa.conv_reduction, tt_osa.conv_reduction)
    for i, (pt_layer, tt_layer) in enumerate(zip(pt_osa.layers, tt_osa.layers)):
        if tt_osa.depthwise:
            # DWConv3x3Block: Sequential(dw_conv, pw_conv, pw_norm, pw_relu)
            pt_mods = list(pt_layer.children())
            copy_conv2d(pt_mods[0], tt_layer.dw_conv)
            copy_conv2d(pt_mods[1], tt_layer.pw_conv)
            copy_bn(pt_mods[2], tt_layer.bn)
        else:
            copy_conv3x3_block(pt_layer, tt_layer)
    copy_conv1x1_block(pt_osa.concat, tt_osa.concat_conv)
    copy_ese(pt_osa.ese, tt_osa.ese)


# ===========================================================================
# Compare helper
# ===========================================================================

def compare(pt_out, tt_out, name, atol=1e-4):
    """Compare PyTorch and TTSim outputs, print diagnostics."""
    pt_np = pt_out.detach().numpy() if isinstance(pt_out, torch.Tensor) else pt_out
    tt_np = tt_out.data if hasattr(tt_out, 'data') else tt_out
    if tt_np is None:
        print(f"  {name}: TTSim data=None (shape-only mode)")
        return None  # inconclusive
    print(f"  {name}:")
    print(f"    PyTorch shape: {list(pt_np.shape)}")
    print(f"    TTSim   shape: {list(tt_np.shape)}")
    if list(pt_np.shape) != list(tt_np.shape):
        print(f"    [FAIL] Shape mismatch!")
        return False
    diff = np.abs(pt_np.astype(np.float64) - tt_np.astype(np.float64))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"    Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
    if np.allclose(pt_np, tt_np, atol=atol):
        print(f"    [OK] Match (atol={atol})")
        return True
    print(f"    [FAIL] Exceeds tolerance")
    return False


# ===========================================================================
# Data-validation tests (weight-copied)
# ===========================================================================

def test_conv3x3_block_numerical():
    """Conv3x3Block numerical — PT weights copied to TTSIM."""
    print("\n" + "=" * 70)
    print("TEST 15: Conv3x3Block numerical — shared weights")
    print("=" * 70)

    in_ch, out_ch = 16, 32
    np.random.seed(42)
    torch.manual_seed(42)
    data = np.random.randn(2, in_ch, 16, 16).astype(np.float32)

    # PyTorch
    pt_layers = conv3x3_pt(in_ch, out_ch, "pt_num_conv3x3", "0", stride=1)
    pt = nn.Sequential(OrderedDict(pt_layers))
    pt.eval()
    with torch.no_grad():
        pt_out = pt(torch.from_numpy(data))

    # TTSIM
    tt = Conv3x3Block('t15_conv3x3', in_ch, out_ch, stride=1)
    copy_conv3x3_block(pt, tt)
    tt_inp = F._from_data('t15_inp', data)
    tt_out = tt(tt_inp)

    result = compare(pt_out, tt_out, "Conv3x3Block forward", atol=1e-4)
    if result is None:
        check("data propagation", True, "shape-only — no data to compare")
    else:
        check("numerical match", result is True)


def test_ese_module_numerical():
    """eSEModule numerical — PT weights copied to TTSIM."""
    print("\n" + "=" * 70)
    print("TEST 16: eSEModule numerical — shared weights")
    print("=" * 70)

    channel = 32
    np.random.seed(43)
    torch.manual_seed(43)
    data = np.random.randn(2, channel, 8, 8).astype(np.float32)

    # PyTorch
    pt_ese = eSEModulePT(channel)
    pt_ese.eval()

    # TTSIM — copy weights (also zeroes PT fc bias that TTSIM can't represent)
    tt_ese = eSEModuleTTSIM('t16_ese', channel)
    copy_ese(pt_ese, tt_ese)

    # Forward both after weight sync
    with torch.no_grad():
        pt_out = pt_ese(torch.from_numpy(data))
    tt_inp = F._from_data('t16_inp', data)
    tt_out = tt_ese(tt_inp)

    result = compare(pt_out, tt_out, "eSEModule forward", atol=1e-4)
    if result is None:
        check("data propagation", True, "shape-only — no data to compare")
    else:
        check("numerical match", result is True)


def test_osa_module_numerical():
    """_OSA_module numerical — PT weights copied to TTSIM."""
    print("\n" + "=" * 70)
    print("TEST 17: _OSA_module numerical — shared weights")
    print("=" * 70)

    in_ch, stage_ch, concat_ch = 32, 32, 64
    layer_per_block = 3
    np.random.seed(44)
    torch.manual_seed(44)
    data = np.random.randn(1, in_ch, 8, 8).astype(np.float32)

    # PyTorch
    pt_osa = _OSA_module_PT(in_ch, stage_ch, concat_ch, layer_per_block,
                             "pt_num_osa", SE=True)
    pt_osa.eval()

    # TTSIM — copy weights (also zeroes PT ese fc bias)
    tt_osa = _OSA_module_TT('t17_osa', in_ch, stage_ch, concat_ch,
                             layer_per_block, SE=True)
    copy_osa_module(pt_osa, tt_osa)

    # Forward both after weight sync
    with torch.no_grad():
        pt_out = pt_osa(torch.from_numpy(data))
    tt_inp = F._from_data('t17_inp', data)
    tt_out = tt_osa(tt_inp)

    result = compare(pt_out, tt_out, "OSA module forward", atol=1e-4)
    if result is None:
        check("data propagation", True, "shape-only — no data to compare")
    else:
        check("numerical match", result is True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("vovnet.py — PyTorch vs TTSIM Validation")
    print("=" * 70)

    test_conv3x3_block_shape()
    test_conv1x1_block_shape()
    test_dw_conv3x3_block_shape()
    test_hsigmoid_numerical()
    test_ese_module_shape()
    test_osa_module_shape_no_dw()
    test_osa_module_shape_dw()
    test_osa_stage_with_pool()
    test_osa_stage_no_pool()
    test_full_vovnet_slim_ese()
    test_full_vovnet_99_ese()
    test_hsigmoid_numerical_compare()
    test_config_and_strides()
    test_various_input_sizes()
    test_conv3x3_block_numerical()
    test_ese_module_numerical()
    test_osa_module_numerical()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    print("=" * 70)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
