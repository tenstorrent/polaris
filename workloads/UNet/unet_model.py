#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

""" Full assembly of the parts to form the complete network """

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from loguru import logger

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from workloads.UNet.unet_parts import *


class UNet(SimNN.Module):
    def __init__(self, objname, cfg):
        super().__init__()
        self.name = objname
        self.n_channels = cfg['n_channels']
        self.n_classes = cfg['n_classes']
        self.bilinear = cfg['bilinear']
        self.img_size = cfg['img_size']

        self.inc = (DoubleConv(f'{self.name}_DoubleConv', self.n_channels, 64))
        self.down1 = (Down(f'{self.name}_Down1', 64, 128))
        self.down2 = (Down(f'{self.name}_Down2', 128, 256))
        self.down3 = (Down(f'{self.name}_Down3', 256, 512))
        factor = 2 if self.bilinear else 1
        self.down4 = (Down(f'{self.name}_Down4', 512, 1024 // factor))
        self.up1 = (Up(f'{self.name}_Up1', 1024, 512 // factor, self.bilinear))
        self.up2 = (Up(f'{self.name}_Up2', 512, 256 // factor, self.bilinear))
        self.up3 = (Up(f'{self.name}_Up3', 256, 128 // factor, self.bilinear))
        self.up4 = (Up(f'{self.name}_Up4', 128, 64, self.bilinear))
        self.outc = (OutConv(f'{self.name}_OutConv', 64, self.n_classes))
        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
                'x_in': F._from_shape('input_tensor', shape=[self.n_classes, self.n_channels, self.img_size, self.img_size]),
                }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def __call__(self, x=None):
        x = self.input_tensors['x_in'] if x is None else x
        logger.info(f'Input shape: {x.shape}')
        return self.forward(x)
