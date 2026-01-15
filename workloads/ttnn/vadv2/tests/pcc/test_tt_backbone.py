#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../'))
import ttsim.front.ttnn as ttnn
from workloads.ttnn.vadv2.tt import tt_backbone
from loguru import logger

mesh_device = ttnn.open_device(device_id=0)

class ConvParams:
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, stride=None, padding=None, dilation=None, batch_size=1, input_height=None, input_width=None, groups=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size is not None else [3, 3]
        self.stride = stride if stride is not None else [1, 1]
        self.padding = padding if padding is not None else [1, 1]
        self.dilation = dilation if dilation is not None else [1, 1]
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.groups = groups

class MaxPoolParams:
    def __init__(self, kernel_size=None, stride=None, padding=None, dilation=None, batch_size=1, input_channels=None, input_height=None, input_width=None, dtype=ttnn.bfloat16):
        self.kernel_size = kernel_size if kernel_size is not None else 2
        self.stride = stride if stride is not None else 2
        self.padding = padding if padding is not None else 0
        self.dilation = dilation if dilation is not None else 1
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.dtype = dtype

class ConvPth:
    def __init__(self, wshape, bshape):
        self.weight = ttnn._rand(wshape, dtype=ttnn.float32, device=mesh_device)
        self.bias = ttnn._rand(bshape, dtype=ttnn.float32, device=mesh_device)

class ConvOne:
    def __init__(self, cp1, cp2, cp3, ds0=None):
        self.conv1 = cp1
        self.conv2 = cp2
        self.conv3 = cp3
        self.downsample = [ds0 if ds0 is not None else None]

class ResOne:
    def __init__(self, c1, c2, c3, d1=None):
        self.conv1 = c1
        self.conv2 = c2
        self.conv3 = c3
        self.downsample = d1 if d1 is not None else None

class Parameter:
    class ConvArgs:
        def __init__(self):
            self.conv1 = ConvParams(in_channels=3, out_channels=64, kernel_size=[7,7], stride=[2,2], padding=[3,3], dilation=[1,1], batch_size=6, input_height=384, input_width=640)
            cp11_1 = ConvParams(in_channels=64, out_channels=64, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp11_2 = ConvParams(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp11_3 = ConvParams(in_channels=64, out_channels=256, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            ds1_0 = ConvParams(in_channels=64, out_channels=256, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp12_1 = ConvParams(in_channels=256, out_channels=64, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp12_2 = ConvParams(in_channels=64, out_channels=64, kernel_size=[3,3], stride=[1,1], padding=[1,1], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp12_3 = ConvParams(in_channels=64, out_channels=256, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            self.layer1 = [ConvOne(cp11_1, cp11_2, cp11_3, ds0=ds1_0), ConvOne(cp12_1, cp12_2, cp12_3), ConvOne(cp12_1, cp12_2, cp12_3)]

            cp21_1 = ConvParams(in_channels=256, out_channels=128, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp21_2 = ConvParams(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp21_3 = ConvParams(in_channels=128, out_channels=512, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            ds2_0 = ConvParams(in_channels=256, out_channels=512, kernel_size=[1,1], stride=[2,2], padding=[0,0], dilation=[1,1], batch_size=6, input_height=96, input_width=160)
            cp22_1 = ConvParams(in_channels=512, out_channels=128, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            cp22_2 = ConvParams(in_channels=128, out_channels=128, kernel_size=[3,3], stride=[1,1], padding=[1,1], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            cp22_3 = ConvParams(in_channels=128, out_channels=512, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            self.layer2 = [ConvOne(cp21_1, cp21_2, cp21_3, ds0=ds2_0), ConvOne(cp22_1, cp22_2, cp22_3), ConvOne(cp22_1, cp22_2, cp22_3), ConvOne(cp22_1, cp22_2, cp22_3)]

            cp31_1 = ConvParams(in_channels=512, out_channels=256, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            cp31_2 = ConvParams(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            cp31_3 = ConvParams(in_channels=256, out_channels=1024, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            ds3_0 = ConvParams(in_channels=512, out_channels=1024, kernel_size=[1,1], stride=[2,2], padding=[0,0], dilation=[1,1], batch_size=6, input_height=48, input_width=80)
            cp32_1 = ConvParams(in_channels=1024, out_channels=256, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            cp32_2 = ConvParams(in_channels=256, out_channels=256, kernel_size=[3,3], stride=[1,1], padding=[1,1], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            cp32_3 = ConvParams(in_channels=256, out_channels=1024, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            self.layer3 = [ConvOne(cp31_1, cp31_2, cp31_3, ds0=ds3_0), ConvOne(cp32_1, cp32_2, cp32_3), ConvOne(cp32_1, cp32_2, cp32_3), ConvOne(cp32_1, cp32_2, cp32_3), ConvOne(cp32_1, cp32_2, cp32_3), ConvOne(cp32_1, cp32_2, cp32_3)]

            cp41_1 = ConvParams(in_channels=1024, out_channels=512, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            cp41_2 = ConvParams(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[2,2], padding=[1,1], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            cp41_3 = ConvParams(in_channels=512, out_channels=2048, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=12, input_width=20)
            ds4_0 = ConvParams(in_channels=1024, out_channels=2048, kernel_size=[1,1], stride=[2,2], padding=[0,0], dilation=[1,1], batch_size=6, input_height=24, input_width=40)
            cp42_1 = ConvParams(in_channels=2048, out_channels=512, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=12, input_width=20)
            cp42_2 = ConvParams(in_channels=512, out_channels=512, kernel_size=[3,3], stride=[1,1], padding=[1,1], dilation=[1,1], batch_size=6, input_height=12, input_width=20)
            cp42_3 = ConvParams(in_channels=512, out_channels=2048, kernel_size=[1,1], stride=[1,1], padding=[0,0], dilation=[1,1], batch_size=6, input_height=12, input_width=20)
            self.layer4 = [ConvOne(cp41_1, cp41_2, cp41_3, ds0=ds4_0), ConvOne(cp42_1, cp42_2, cp42_3), ConvOne(cp42_1, cp42_2, cp42_3)]
            self.maxpool = MaxPoolParams(kernel_size=3, stride=2, padding=1, dilation=1, batch_size=6, input_channels=64, input_height=192, input_width=320)

    class ResModel:
        def __init__(self):
            self.conv1 = ConvPth([64, 3, 7, 7], [64])
            self.layer1_0 = ResOne(
                ConvPth([64, 64, 1, 1], [64]),
                ConvPth([64, 64, 3, 3], [64]),
                ConvPth([256, 64, 1, 1], [256]),
                d1=ConvPth([256, 64, 1, 1], [256]),
            )
            self.layer1_1 = ResOne(
                ConvPth([64, 256, 1, 1], [64]),
                ConvPth([64, 64, 3, 3], [64]),
                ConvPth([256, 64, 1, 1], [256]),
            )
            self.layer1_2 = ResOne(
                ConvPth([64, 256, 1, 1], [64]),
                ConvPth([64, 64, 3, 3], [64]),
                ConvPth([256, 64, 1, 1], [256]),
            )
            self.layer2_0 = ResOne(
                ConvPth([128, 256, 1, 1], [128]),
                ConvPth([128, 128, 3, 3], [128]),
                ConvPth([512, 128, 1, 1], [512]),
                d1=ConvPth([512, 256, 1, 1], [512]),
            )
            self.layer2_1 = ResOne(
                ConvPth([128, 512, 1, 1], [128]),
                ConvPth([128, 128, 3, 3], [128]),
                ConvPth([512, 128, 1, 1], [512]),
            )
            self.layer2_2 = ResOne(
                ConvPth([128, 512, 1, 1], [128]),
                ConvPth([128, 128, 3, 3], [128]),
                ConvPth([512, 128, 1, 1], [512]),
            )
            self.layer2_3 = ResOne(
                ConvPth([128, 512, 1, 1], [128]),
                ConvPth([128, 128, 3, 3], [128]),
                ConvPth([512, 128, 1, 1], [512]),
            )
            self.layer3_0 = ResOne(
                ConvPth([256, 512, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
                d1=ConvPth([1024, 512, 1, 1], [1024]),
            )
            self.layer3_1 = ResOne(
                ConvPth([256, 1024, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
            )
            self.layer3_2 = ResOne(
                ConvPth([256, 1024, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
            )
            self.layer3_3 = ResOne(
                ConvPth([256, 1024, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
            )
            self.layer3_4 = ResOne(
                ConvPth([256, 1024, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
            )
            self.layer3_5 = ResOne(
                ConvPth([256, 1024, 1, 1], [256]),
                ConvPth([256, 256, 3, 3], [256]),
                ConvPth([1024, 256, 1, 1], [1024]),
            )
            self.layer4_0 = ResOne(
                ConvPth([512, 1024, 1, 1], [512]),
                ConvPth([512, 512, 3, 3], [512]),
                ConvPth([2048, 512, 1, 1], [2048]),
                d1=ConvPth([2048, 1024, 1, 1], [2048]),
            )
            self.layer4_1 = ResOne(
                ConvPth([512, 2048, 1, 1], [512]),
                ConvPth([512, 512, 3, 3], [512]),
                ConvPth([2048, 512, 1, 1], [2048]),
            )
            self.layer4_2 = ResOne(
                ConvPth([512, 2048, 1, 1], [512]),
                ConvPth([512, 512, 3, 3], [512]),
                ConvPth([2048, 512, 1, 1], [2048]),
            )
    conv_args = ConvArgs()
    res_model = ResModel()

def test_vadv2_backbone(device):
    torch_input = ttnn._rand([6, 3, 384, 640], dtype=ttnn.bfloat16, device=device)
    ttnn_input_tensor = torch_input
    logger.debug("ttnn_input_tensor shape:", ttnn_input_tensor.shape, "and dtype:", ttnn_input_tensor.dtype)
    parameter = Parameter()
    ttnn_model = tt_backbone.TtResnet50(parameter.conv_args, parameter.res_model, device)
    ttnn_output = ttnn_model(ttnn_input_tensor, batch_size=6)[0]
    logger.debug("ttnn_output shape:", ttnn_output.shape)

if __name__ == "__main__":
    test_vadv2_backbone(mesh_device)