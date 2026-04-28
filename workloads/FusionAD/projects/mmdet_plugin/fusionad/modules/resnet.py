# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
ResNet-101 backbone for FusionAD — TTSim implementation.

Architecture: 4-stage ResNet with Bottleneck blocks.
  - Stem:   Conv7x7/2 + BN + ReLU + MaxPool/2
  - Stage0: 3  blocks, planes=64,  stride=1  → 256  ch, stride 4
  - Stage1: 4  blocks, planes=128, stride=2  → 512  ch, stride 8
  - Stage2: 23 blocks, planes=256, stride=2  → 1024 ch, stride 16
  - Stage3: 3  blocks, planes=512, stride=2  → 2048 ch, stride 32

FusionAD uses out_indices=(1, 2, 3) to return stages 1-3:
  → [512, 1024, 2048] channel outputs

"""

import os
import sys

polaris_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


class Bottleneck(SimNN.Module):
    """ResNet bottleneck block: 1x1 → 3x3 → 1x1 + residual.

    When use_dcn=True the 3×3 conv is preceded by a conv_offset that
    predicts deformable offsets + modulation masks (DCNv2 pattern).
    The actual deformable sampling is approximated by a regular Conv2d
    (same I/O shapes) since TTSIM has no native DeformConv op.
    """

    expansion = 4

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['out_channels']
        self.stride = cfg.get('stride', 1)
        self.downsample = cfg.get('downsample', None)
        self.use_dcn = cfg.get('use_dcn', False)
        dcn_deform_groups = cfg.get('dcn_deform_groups', 1)

        # --- 1×1 reduce ---
        self.conv0 = F.Conv2d(f'{name}.conv0', self.in_channels,
                              self.out_channels, kernel_size=1, padding=0,
                              stride=1)
        self.bn0 = F.BatchNorm2d(f'{name}.bn0', self.out_channels)

        # --- DCNv2 offset branch (optional) ---
        if self.use_dcn:
            # ModulatedDeformConvPack.conv_offset:
            #   in_ch  = out_channels (intermediate planes)
            #   out_ch = deform_groups * 3 * K * K  (offsets + mask)
            k = 3
            offset_channels = dcn_deform_groups * 3 * k * k  # 27 for dg=1
            self.conv_offset = F.Conv2d(
                f'{name}.conv_offset', self.out_channels, offset_channels,
                kernel_size=k, padding=1, stride=self.stride)
            self.offset_sigmoid = F.Sigmoid(f'{name}.offset_sigmoid')

        # --- 3×3 conv (replaces deformable conv with regular conv) ---
        self.conv1 = F.Conv2d(f'{name}.conv1', self.out_channels,
                              self.out_channels, kernel_size=3, padding=1,
                              stride=self.stride)
        self.bn1 = F.BatchNorm2d(f'{name}.bn1', self.out_channels)

        # --- 1×1 expand ---
        self.conv2 = F.Conv2d(f'{name}.conv2', self.out_channels,
                              self.out_channels * Bottleneck.expansion,
                              kernel_size=1, padding=0, stride=1)
        self.bn2 = F.BatchNorm2d(f'{name}.bn2',
                                 self.out_channels * Bottleneck.expansion)

        self.relu = F.Relu(f'{name}.relu')

        if self.downsample is not None:
            xi = self.downsample['in_channels']
            xo = self.downsample['out_channels']
            xs = self.downsample['stride']
            self.conv_ds = F.Conv2d(f'{name}.conv_ds', xi, xo,
                                    kernel_size=1, padding=0, stride=xs)
            self.bn_ds = F.BatchNorm2d(f'{name}.bn_ds', xo)

        super().link_op2module()

    def __call__(self, x):
        # 1×1 reduce
        y = self.bn0(self.conv0(x))

        # DCNv2 offset branch (side computation in the graph)
        if self.use_dcn:
            _off = self.conv_offset(y)
            _off = self.offset_sigmoid(_off)   # mask portion uses sigmoid

        # 3×3 conv (approximates deformable conv)
        y = self.bn1(self.conv1(y))

        # 1×1 expand
        y = self.bn2(self.conv2(y))

        # Residual
        if self.downsample is None:
            z = y + x
        else:
            x = self.conv_ds(x)
            x = self.bn_ds(x)
            z = y + x
        return self.relu(z)


class ResNetBackbone(SimNN.Module):
    """4-stage ResNet producing multi-scale features.

    For FusionAD (ResNet-101): layers=[3, 4, 23, 3], out_indices=(1, 2, 3)
    Output channels per stage: [256, 512, 1024, 2048]
    With out_indices=(1,2,3) returns: [512, 1024, 2048]

    Args:
        name: Module name prefix.
        cfg: Dict with keys:
            - layers: list of 4 ints (default [3, 4, 23, 3] for ResNet-101)
            - img_channels: input channels (default 3)
            - out_indices: tuple of stage indices to output (default (1, 2, 3))
    """

    def __init__(self, name, cfg):
        super().__init__()
        self.name = name
        self.in_channels = 64
        layers = cfg.get('layers', [3, 4, 23, 3])
        img_channels = cfg.get('img_channels', 3)
        self.out_indices = cfg.get('out_indices', (1, 2, 3))
        # DCNv2: stage_with_dcn is a tuple of bools per stage
        stage_with_dcn = cfg.get('stage_with_dcn', (False, False, False, False))
        dcn_deform_groups = cfg.get('dcn_deform_groups', 1)

        # Stem: Conv7x7/2 + BN + ReLU + MaxPool/2
        self.conv1 = F.Conv2d(f'{name}.conv1', img_channels, 64,
                              kernel_size=7, stride=2, padding=3)
        self.bn1 = F.BatchNorm2d(f'{name}.bn1', 64)
        self.relu = F.Relu(f'{name}.relu')
        self.maxpool = F.MaxPool2d(f'{name}.maxpool',
                                   kernel_size=3, stride=2, padding=1)

        self.stage1 = SimNN.ModuleList(
            self._make_stage(f'{name}.layer1', layers[0], 64, stride=1,
                             use_dcn=stage_with_dcn[0], dcn_dg=dcn_deform_groups))
        self.stage2 = SimNN.ModuleList(
            self._make_stage(f'{name}.layer2', layers[1], 128, stride=2,
                             use_dcn=stage_with_dcn[1], dcn_dg=dcn_deform_groups))
        self.stage3 = SimNN.ModuleList(
            self._make_stage(f'{name}.layer3', layers[2], 256, stride=2,
                             use_dcn=stage_with_dcn[2], dcn_dg=dcn_deform_groups))
        self.stage4 = SimNN.ModuleList(
            self._make_stage(f'{name}.layer4', layers[3], 512, stride=2,
                             use_dcn=stage_with_dcn[3], dcn_dg=dcn_deform_groups))

        super().link_op2module()

    def _make_stage(self, name, num_blocks, planes, stride,
                    use_dcn=False, dcn_dg=1):
        blocks = []
        exp = Bottleneck.expansion
        downsample_cfg = None
        if stride != 1 or self.in_channels != planes * exp:
            downsample_cfg = {
                'in_channels': self.in_channels,
                'out_channels': planes * exp,
                'stride': stride,
            }
        blocks.append(Bottleneck(f'{name}.0', {
            'in_channels': self.in_channels,
            'out_channels': planes,
            'stride': stride,
            'downsample': downsample_cfg,
            'use_dcn': use_dcn,
            'dcn_deform_groups': dcn_dg,
        }))
        self.in_channels = planes * exp
        for i in range(1, num_blocks):
            blocks.append(Bottleneck(f'{name}.{i}', {
                'in_channels': self.in_channels,
                'out_channels': planes,
                'use_dcn': use_dcn,
                'dcn_deform_groups': dcn_dg,
            }))
        return blocks

    def __call__(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        outs = []
        c2 = x
        for blk in self.stage1:
            c2 = blk(c2)
        outs.append(c2)   # stage 0: stride 4,  256 ch
        c3 = c2
        for blk in self.stage2:
            c3 = blk(c3)
        outs.append(c3)   # stage 1: stride 8,  512 ch
        c4 = c3
        for blk in self.stage3:
            c4 = blk(c4)
        outs.append(c4)   # stage 2: stride 16, 1024 ch
        c5 = c4
        for blk in self.stage4:
            c5 = blk(c5)
        outs.append(c5)   # stage 3: stride 32, 2048 ch
        return [outs[i] for i in self.out_indices]

    def analytical_param_count(self):
        return 0  # Counted via graph analysis
