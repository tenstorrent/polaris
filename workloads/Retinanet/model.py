#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Updated RetinaNet for TT-Polaris Hardware Tracing

""" Parts of the Retinanet model """
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from workloads.Retinanet.anchor import Anchors
from typing import Type, Union, List

from workloads.Retinanet.utils import BasicBlock, Bottleneck, Downsample, BBoxTransform, ClipBoxes

class PyramidFeatures(SimNN.Module):
    def __init__(self, objname, C3_size, C4_size, C5_size, feature_size=256):
        super().__init__()
        self.name = objname

        self.P5_upsampled = F.Upsample(f"{self.name}_P5_upsample", scale_factor=2, mode="nearest")
        self.P4_upsampled = F.Upsample(f"{self.name}_P4_upsample", scale_factor=2, mode="nearest")
        
        # Using the standard pattern caused the TT-Polaris backend to lose
        # input-to-output connectivity, triggering 'tuple' attribute errors during get_ordered_nodes() calls.

        self.P5_upsampled.ipos = [0, 1]
        self.P4_upsampled.ipos = [0, 1]

        self.P5_1 = F.Conv2d(f"{self.name}_P5_1", C5_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.P5_2 = F.Conv2d(f"{self.name}_P5_2", feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.P4_1 = F.Conv2d(f"{self.name}_P4_1", C4_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.P4_2 = F.Conv2d(f"{self.name}_P4_2", feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.P3_1 = F.Conv2d(f"{self.name}_P3_1", C3_size, feature_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.P3_2 = F.Conv2d(f"{self.name}_P3_2", feature_size, feature_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.P6 = F.Conv2d(f"{self.name}_P6", C5_size, feature_size, kernel_size=3, stride=2, padding=1, bias=False)
        self.P7_1 = F.Relu(f"{self.name}_P7_relu")
        self.P7_2 = F.Conv2d(f"{self.name}_P7_2", feature_size, feature_size, kernel_size=3, stride=2, padding=1, bias=False)
        super().link_op2module()

    def __call__(self, inputs):
        C3, C4, C5 = inputs

        # P5
        p5_x = self.P5_1(C5)
        # We pass the target tensor (C4) as a second argument to Upsample to satisfy TT-Polaris backend hardware assertions regarding output shape.
        p5_upsampled_x = self.P5_upsampled(p5_x, C4)  # type: ignore[call-arg]
        p5_out = self.P5_2(p5_x)

        # P4
        p4_x = self.P4_1(C4)
        p4_x = p4_x + p5_upsampled_x
        # Passing C3 as the second argument is required for TT-Polaris backend assertions.
        p4_upsampled_x = self.P4_upsampled(p4_x, C3) # type: ignore[call-arg]
        p4_out = self.P4_2(p4_x)

        # P3
        p3_x = self.P3_1(C3)
        p3_x = p3_x + p4_upsampled_x
        p3_out = self.P3_2(p3_x)

        # P6 & P7
        p6_out = self.P6(C5)
        p7_out = self.P7_1(p6_out)
        p7_out = self.P7_2(p7_out)

        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class RegressionModel(SimNN.Module):
    def __init__(self, objname, num_features_in, num_anchors=9, feature_size=256):
        super().__init__()
        self.name = objname
        self.num_anchors = num_anchors

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1", num_features_in, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act1 = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act2 = F.Relu(f"{self.name}_relu2")

        self.conv3 = F.Conv2d(
            f"{self.name}_conv3", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act3 = F.Relu(f"{self.name}_relu3")

        self.conv4 = F.Conv2d(
            f"{self.name}_conv4", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act4 = F.Relu(f"{self.name}_relu4")

        self.output = F.Conv2d(
            f"{self.name}_out", feature_size, num_anchors * 4, kernel_size=3, padding=1, bias=False,
        )

        super().link_op2module()

    def __call__(self, x):
        out = self.conv1(x); out = self.act1(out)
        out = self.conv2(out); out = self.act2(out)
        out = self.conv3(out); out = self.act3(out)
        out = self.conv4(out); out = self.act4(out)
        out = self.output(out)

        # out is B x (A*4) x H x W → B x H x W x (A*4)
        out = out.permute([0, 2, 3, 1])
        b = out.shape[0]

        return out.view(b, -1, 4)
    
class ClassificationModel(SimNN.Module):
    def __init__(self, objname, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super().__init__()
        self.name        = objname
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1", num_features_in, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act1 = F.Relu(f"{self.name}_relu1")

        self.conv2 = F.Conv2d(
            f"{self.name}_conv2", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act2 = F.Relu(f"{self.name}_relu2")

        self.conv3 = F.Conv2d(
            f"{self.name}_conv3", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act3 = F.Relu(f"{self.name}_relu3")

        self.conv4 = F.Conv2d(
            f"{self.name}_conv4", feature_size, feature_size, kernel_size=3, padding=1, bias=False,
        )
        self.act4 = F.Relu(f"{self.name}_relu4")

        self.output = F.Conv2d(
            f"{self.name}_out", feature_size, num_anchors * num_classes, kernel_size=3, padding=1, bias=False,
        )
        self.output_act = F.Sigmoid(f"{self.name}_sigmoid")
        super().link_op2module()

    def __call__(self, x):
        out = self.conv1(x); out = self.act1(out)
        out = self.conv2(out); out = self.act2(out)
        out = self.conv3(out); out = self.act3(out)
        out = self.conv4(out); out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # B x (A*C) x H x W → B x H x W x (A*C)
        out1 = out.permute([0, 2, 3, 1])
        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        return out2.view(batch_size, -1, self.num_classes)
    
class ResNet(SimNN.Module):
    def __init__(self, objname, block, layers):
        super().__init__()
        self.name     = objname
        self.inplanes = 64

        self.conv1 = F.Conv2d(
            f"{self.name}_conv1", in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False,
        )
        self.bn1   = F.BatchNorm2d(f"{self.name}_bn1", 64)
        self.relu1  = F.Relu(f"{self.name}_relu1")
        self.maxpool = F.MaxPool2d(
            f"{self.name}_maxpool", kernel_size=3, stride=2, padding=1,
        )

        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        super().link_op2module()
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []

        outplanes = planes * block.expansion

        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = Downsample(
                f"{self.name}_down_{planes}", self.inplanes, outplanes, stride,
        )
        layers.append(block(
            f"{self.name}_layer{planes}_0",   # objname
            self.inplanes,                    # inplanes
            planes,                           # planes
            stride=stride,
            downsample=downsample,
        ))
        self.inplanes = outplanes

        for i in range(1, blocks):
            layers.append(block(
                f"{self.name}_layer{planes}_{i}",
                self.inplanes,
                planes,
                stride=1,
                downsample=None,
        ))
        mlist = SimNN.ModuleList(layers)
        for m in mlist:
            self._submodules[m.name] = m
        return mlist
    
    def _run_layer(self, layer_list, x):
        """
        Helper to iterate through the ModuleList
        """
        for module in layer_list:
            x = module(x)
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        C2 = self._run_layer(self.layer1, x)
        C3 = self._run_layer(self.layer2, C2)
        C4 = self._run_layer(self.layer3, C3)
        C5 = self._run_layer(self.layer4, C4)

        return C3, C4, C5
    
class RetinaNet(SimNN.Module):
    def __init__(self, objname, cfg):
        super().__init__()
        self.name = objname
        self.num_classes = cfg["num_classes"]
        self.img_size = cfg["img_size"]
        depth = cfg.get("resnet_depth", 50)

        block: Type[Union[BasicBlock, Bottleneck]]
        layers: List[int]

        # Select ResNet block type, layer configuration, and C3/C4/C5 feature sizes based on resnet_depth.
        if depth == 18:
            block, layers = BasicBlock, [2, 2, 2, 2]
            C3_size, C4_size, C5_size = 128, 256, 512
        elif depth == 34:
            block, layers = BasicBlock, [3, 4, 6, 3]
            C3_size, C4_size, C5_size = 128, 256, 512
        elif depth == 50:
            block, layers = Bottleneck, [3, 4, 6, 3]
            C3_size, C4_size, C5_size = 512, 1024, 2048
        elif depth == 101:
            block, layers = Bottleneck, [3, 4, 23, 3]
            C3_size, C4_size, C5_size = 512, 1024, 2048
        elif depth == 152:
            block, layers = Bottleneck, [3, 8, 36, 3]
            C3_size, C4_size, C5_size = 512, 1024, 2048
        else:
            raise ValueError(
                f"Unsupported resnet_depth {depth}. Supported depths are: 18, 34, 50, 101, 152."
            )

        # Backbone and heads
        self.backbone = ResNet(f"{self.name}_backbone", block, layers)
        self.fpn = PyramidFeatures(f"{self.name}_fpn", C3_size, C4_size, C5_size)
        self.regressionModel = RegressionModel(f"{self.name}_reg", 256)
        self.classificationModel = ClassificationModel(
            f"{self.name}_cls", 256, num_classes=self.num_classes
        )

        # Anchors and post-processing
        self.anchors = Anchors(f"{self.name}_anchors")
        self.regressBoxes = BBoxTransform(f"{self.name}_bbox_trans")
        self.clipBoxes = ClipBoxes(
            f"{self.name}_clip_boxes",
            max_x=float(self.img_size - 1),
            max_y=float(self.img_size - 1),
        )

        # Concats for regression and classification (created once, registered by link_op2module)
        self.reg_concat = F.ConcatX(f"{self.name}_reg_concat", axis=1)
        self.cls_concat = F.ConcatX(f"{self.name}_cls_concat", axis=1)

        super().link_op2module()

    def create_input_tensors(self) -> None:
        """
        Create default input tensors for TT-Polaris/TTSIM:
        - anchors_in: anchors as a SimTensor
        - x_in: dummy image input   <-- THIS LINE IS THE FIX
        """
        # Anchors as numpy -> SimTensor
        anchors_np = self.anchors(self.img_size, self.img_size)  # [1, N, 4]
        anchors_t = F._from_data(f"{self.name}_anchors_tensor", anchors_np)
        
        x_in = F._from_shape(
            f"{self.name}_x_in",
            [1, 3, self.img_size, self.img_size],
        )
        
        self.input_tensors = {
            "anchors_in": anchors_t,
            "x_in": x_in,
        }

    def __call__(self):
        if not hasattr(self, "input_tensors"):
            self.create_input_tensors()

        x = self.input_tensors["x_in"]

        # Backbone + FPN
        C3, C4, C5 = self.backbone.forward(x)
        features = self.fpn([C3, C4, C5])

        # Regression tensor: concat over FPN levels
        regression = self.reg_concat(
            *[self.regressionModel(f) for f in features]
        )

        # Classification tensor: concat over FPN levels
        classification = self.cls_concat(
            *[self.classificationModel(f) for f in features]
        )

        # Anchors as SimTensor input (created in create_input_tensors)
        anchors_t = self.input_tensors["anchors_in"]

        # BBox transform + clipping
        transformed_anchors = self.regressBoxes(anchors_t, regression)
        final_bbox_coords = self.clipBoxes(transformed_anchors, x)

        return classification, regression, final_bbox_coords

    def get_forward_graph(self):
        # Ensure inputs are present (just in case)
        if not hasattr(self, "input_tensors"):
            self.create_input_tensors()

        # Standard pattern: seeds = input_tensors (x_in + anchors_in)
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self, lvl: int = 0) -> int:
        # TT-Polaris just logs this; stub is fine
        return 0
    
def resnet18_backbone(objname):
    # ResNet-18: BasicBlock, [2,2,2,2]
    return ResNet(objname, BasicBlock, [2, 2, 2, 2])

def resnet34_backbone(objname):
    # ResNet-34: BasicBlock, [3,4,6,3]
    return ResNet(objname, BasicBlock, [3, 4, 6, 3])

def resnet50_backbone(objname):
    # ResNet-50: Bottleneck, [3,4,6,3]
    return ResNet(objname, Bottleneck, [3, 4, 6, 3])

def resnet101_backbone(objname):
    # ResNet-101: Bottleneck, [3,4,23,3]
    return ResNet(objname, Bottleneck, [3, 4, 23, 3])

def resnet152_backbone(objname):
    # ResNet-152: Bottleneck, [3,8,36,3]
    return ResNet(objname, Bottleneck, [3, 8, 36, 3])
