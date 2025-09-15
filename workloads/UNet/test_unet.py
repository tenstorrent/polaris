#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
from workloads.UNet.unet_model import UNet

def test_unet_forward_random():
    n_channels = 3
    n_classes = 2
    img_size = 256
    bilinear = False

    x = F._from_shape(name='input_tensor', shape=[2, n_channels, img_size, img_size])
    print(f'bilinear is {bilinear}')
    model = UNet('unet', n_channels, n_classes, bilinear=bilinear)
    model.create_input_tensors()
    out = model(x)
    print(f'output shape is {out.shape}')
    assert out.shape[0] == 2
    assert out.shape[1] == n_classes
    assert out.shape[2] == img_size and out.shape[3] == img_size
    print('UNet test passed!')

    # gg = model.get_forward_graph()
    # print('Dumping ONNX...')
    # gg.graph2onnx(f'unet_model.onnx', do_model_check=True)

if __name__ == "__main__":
    test_unet_forward_random()
