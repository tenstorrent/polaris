#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.utils.common import parse_yaml

class SecondFPN(SimNN.Module):
    def __init__(self, name,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=[128, 128, 128, 128],
                 upsample_strides=[0.25, 0.5, 1, 2],
                 ):
        super().__init__()
        assert len(in_channels) == 4, f"SecondFPN ERROR-1"
        assert len(in_channels) == len(upsample_strides) == len(out_channels), f"SecondFPN ERROR-2"

        self.name             = name
        self.in_channels      = in_channels
        self.upsample_strides = upsample_strides
        self.out_channels     = out_channels

        # Lateral 1x1 convolutions to reduce input channels to out_channels
        self.lateral_convs = F.SimOpHandleList([
            F.Conv2d(name + f'.lconv{i}', ic, oc, kernel_size=1)
            for i,(ic, oc) in enumerate(zip(in_channels, out_channels))
            ])


        # 3x3 convolutions for post-processing fused features
        op_list = []
        for i,oc in enumerate(out_channels):
            op_list.append(F.Conv2d(name + f'.fpn_conv{i}', oc, oc, kernel_size=3, padding=1))
            op_list.append(F.BatchNorm2d(name + f'.fpn_bn{i}', oc))
            op_list.append(F.Relu(name + f'.fpn_relu{i}'))
        self.fpn_convs = F.SimOpHandleList(op_list)

        # Batch normalization for lateral connections
        self.lateral_norms = F.SimOpHandleList([
            F.BatchNorm2d(name + f'.lnorm{i}', oc)
            for i,oc in enumerate(out_channels)
            ])

        # upsampling/downsampling
        self.resize_ops = F.SimOpHandleList([
            F.Resize(name + f'.interpolate{i}',
                     scale_factor = upsample_strides[i+1]/upsample_strides[i],
                     mode='nearest',
                     nearest_mode='floor',
                     coordinate_transformation_mode='asymmetric')
            for i in range(len(upsample_strides)-1)
            ])
        self.resize_plus_ops = F.SimOpHandleList([
            F.Add(name + f'.plus{i}')
            for i in range(len(upsample_strides)-1)
            ])

        super().link_op2module()

    def __call__(self, inputs):
        # Step 1: Apply lateral 1x1 convolutions
        laterals = []
        for i, (input_feature, lateral_conv, lateral_norm) in enumerate(
            zip(inputs, self.lateral_convs, self.lateral_norms)
        ):
            lateral = lateral_conv(input_feature)  # Reduce channels to 128
            lateral = lateral_norm(lateral)
            laterals.append(lateral)

        # Step 2: Top-down pathway with upsampling/downsampling
        outputs     = [None] * len(laterals)
        last_inner  = laterals[-1]  # Start with C5 (highest level)
        outputs[-1] = last_inner   # P5 (no upsampling yet)

        # Top-down fusion
        for i in range(len(laterals) - 2, -1, -1):  # From C4 to C2
            # Upsample the higher-level feature
            top_down = self.resize_ops[i](last_inner)
            # Add to lateral feature
            last_inner = self.resize_plus_ops[i](laterals[i],top_down)
            outputs[i] = last_inner

        # Step 3: Adjust resolutions based on upsample_strides
        final_outputs = []
        for i, op in enumerate(outputs):
            # Apply 3x3 convolution for refinement
            op = self.fpn_convs[3*i+0](op)
            op = self.fpn_convs[3*i+1](op)
            op = self.fpn_convs[3*i+2](op)
            final_outputs.append(op)

        return final_outputs

# Example usage
if __name__ == "__main__":
    # Create SecondFPN
    neck = SecondFPN('SecondFPN',
                     in_channels=[256, 512, 1024, 2048],
                     upsample_strides=[0.25, 0.5, 1, 2],
                     out_channels=[128, 128, 128, 128]
    )

    # Example input feature maps (from ResNet-50, 704x1280 input)
    inputs = [
            #F._from_shape('C2', [1,  256, 176, 320]),
            #F._from_shape('C3', [1,  512,  88, 160]),
            #F._from_shape('C4', [1, 1024,  44,  80]),
            #F._from_shape('C5', [1, 2048,  22,  40]),

            F._from_shape('C2', [5,  256, 32, 32]),
            F._from_shape('C3', [5,  512, 16, 16]),
            F._from_shape('C4', [5, 1024,  8,  8]),
            F._from_shape('C5', [5, 2048,  4,  4]),

            #F._from_shape('C2', [5,  80, 32, 32]),
            #F._from_shape('C3', [5, 160, 16, 16]),
            #F._from_shape('C4', [5, 320,  8,  8]),
            #F._from_shape('C5', [5, 640,  4,  4]),
    ]

    # Forward pass
    outputs = neck(inputs)

    for i, out in enumerate(outputs):
        print(f"P{i+2} shape: {out.shape}")
    gg = neck._get_forward_graph({'inputs': inputs})
    gg.graph2onnx(f'bevdepth_backbone_second_fpn.onnx', do_model_check=True)
