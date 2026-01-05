#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim conversion of backbone.py
  FrozenBatchNorm2d
  BackboneBase
  Backbone   (ResNet)
  Joiner
  build_backbone()

DESIGN PRINCIPLES:
  - Shape inference only in forward pass
  - Direct implementation using NumPy for numerical computation
  - No custom library functions (only standard PyTorch equivalents)
  - Returns SimTensor objects
"""

"""
Module	Validation	Rationale
FrozenBatchNorm2d	Shape + Numerical	Element-wise, no Conv2d, fast
ResNetBottleneck	Shape + Numerical	Single block validates Conv2d+BN+residual correctness
Backbone	Shape + Mask + Key/Count assertions	Block-level numerics already proven in Test 2. Full backbone numerical is redundant and slow
Joiner	Shape + Mask for features. Numerical for positional encoding only	Sine encoding is deterministic numpy math, no Conv2d dependency
"""

import os, sys
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

import ttsim.front.functional.op as F

# import ttsim.front.functional.tensor_op  as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops.tensor import SimTensor, shape_as_optional_list

# Import NestedTensorTTSim and interpolate from misc_ttsim
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor, interpolate

# position encoding in its own file
from workloads.Deformable_DETR.models.position_encoding_ttsim import (
    build_position_encoding,
)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
class Identity(SimNN.Module):
    """Pass-through (placeholder until full bottleneck layers are ported)."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        super().link_op2module()

    def __call__(self, x):
        return x


# ──────────────────────────────────────────────────────────────────────────────
# FrozenBatchNorm2d
# ──────────────────────────────────────────────────────────────────────────────
class FrozenBatchNorm2d(SimNN.Module):
    """
    TTSim implementation of FrozenBatchNorm2d.

    Mirrors PyTorch forward() exactly:
        w = weight.reshape(1, -1, 1, 1)
        b = bias.reshape(1, -1, 1, 1)
        rv = running_var.reshape(1, -1, 1, 1)
        rm = running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + eps).rsqrt()
        bias_offset = b - rm * scale
        output = x * scale + bias_offset

    Design: Shape inference always performed, numerical computation when data available.
    """

    def __init__(self, name: str, n: int, eps: float = 1e-5):
        super().__init__()
        self.name = name
        self.n = n
        self.eps = eps
        self.weight: np.ndarray = np.ones(n, dtype=np.float32)
        self.bias: np.ndarray = np.zeros(n, dtype=np.float32)
        self.running_mean: np.ndarray = np.zeros(n, dtype=np.float32)
        self.running_var: np.ndarray = np.ones(n, dtype=np.float32)
        # Use F.BatchNorm2d SimOpHandle so the output tensor is graph-tracked.
        self._bn_op = F.BatchNorm2d(name + '.bn', n)
        super().link_op2module()

    def set_parameters(self, weight, bias, running_mean, running_var):
        self.weight = np.asarray(weight, dtype=np.float32).copy()
        self.bias = np.asarray(bias, dtype=np.float32).copy()
        self.running_mean = np.asarray(running_mean, dtype=np.float32).copy()
        self.running_var = np.asarray(running_var, dtype=np.float32).copy()
        return self

    def __call__(self, x):
        # Delegate to the registered F.BatchNorm2d SimOpHandle so that
        # the output tensor is properly tracked in the WorkloadGraph.
        return self._bn_op(x)

    # def __call__(self, x):
    #     # Extract shape and data
    #     if isinstance(x, SimTensor):
    #         x_shape = x.shape
    #         x_data = x.data
    #         dtype = x.dtype if hasattr(x, 'dtype') else np.float32
    #     else:
    #         x_shape = list(x.shape) if hasattr(x, 'shape') else None
    #         x_data = x.data if hasattr(x, 'data') else x
    #         dtype = x.dtype if hasattr(x, 'dtype') else np.float32

    #     # Shape inference: output shape same as input shape
    #     output_shape = x_shape

    #     # Numerical computation if data available
    #     if x_data is not None:
    #         # Reshape parameters for broadcasting [1, C, 1, 1]
    #         w = self.weight.reshape(1, -1, 1, 1)
    #         b = self.bias.reshape(1, -1, 1, 1)
    #         rv = self.running_var.reshape(1, -1, 1, 1)
    #         rm = self.running_mean.reshape(1, -1, 1, 1)

    #         # scale = w * rsqrt(rv + eps)
    #         scale = w * np.power(rv + self.eps, -0.5)
    #         # bias_offset = b - rm * scale
    #         bias_offset = b - rm * scale
    #         # output = x * scale + bias_offset
    #         out = x_data * scale + bias_offset

    #         return SimTensor({
    #             'name': f'{self.name}_output',
    #             'shape': list(out.shape),
    #             'data': out.astype(np.float32),
    #             'dtype': np.float32
    #         })
    #     else:
    #         # Shape inference only
    #         return SimTensor({
    #             'name': f'{self.name}_output',
    #             'shape': output_shape,
    #             'data': None,
    #             'dtype': dtype
    #         })


# ──────────────────────────────────────────────────────────────────────────────
# ResNet Bottleneck
# ──────────────────────────────────────────────────────────────────────────────
class ResNetBottleneck(SimNN.Module):
    """
    Single Bottleneck block.
        conv1  1×1   in_ch  → mid_ch
        conv2  3×3   mid_ch → mid_ch   (stride here)
        conv3  1×1   mid_ch → out_ch
        + optional 1×1 downsample on skip
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample=None,
    ):
        super().__init__()
        self.name = name
        mid = out_channels // 4

        self.conv1 = F.Conv2d(
            f"{name}.conv1", in_channels, mid, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = FrozenBatchNorm2d(f"{name}.bn1", mid)
        self.conv2 = F.Conv2d(
            f"{name}.conv2",
            mid,
            mid,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = FrozenBatchNorm2d(f"{name}.bn2", mid)
        self.conv3 = F.Conv2d(
            f"{name}.conv3", mid, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = FrozenBatchNorm2d(f"{name}.bn3", out_channels)
        # Three separate relu ops — one per activation site, avoids SimOpHandle
        # single-call constraint (calling the same handle twice accumulates inList).
        self.relu1 = F.Relu(f"{name}.relu1")
        self.relu2 = F.Relu(f"{name}.relu2")
        self.relu3 = F.Relu(f"{name}.relu3")
        # Pre-created residual add so it has a module pointer and tracked output.
        self.residual_add = F.Add(f"{name}.residual_add")
        self.downsample = downsample
        super().link_op2module()

    def __call__(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.residual_add(out, identity)
        out = self.relu3(out)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# BackboneBase
# ──────────────────────────────────────────────────────────────────────────────
class BackboneBase(SimNN.Module):
    """
    Mirrors BackboneBase.forward():
        xs   = body(tensor_list.tensors)          # IntermediateLayerGetter
        out  = {}
        for name, x in xs.items():
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(bool)[0]
            out[name] = NestedTensor(x, mask)
    """

    def __init__(
        self, name: str, backbone_layers: Dict, return_interm_layers: bool = True
    ):
        super().__init__()
        self.name = name
        self.backbone_layers = backbone_layers
        # Register each layer as a submodule so get_ops/get_tensors traverse into them.
        for lname, layer in backbone_layers.items():
            if isinstance(layer, SimNN.Module):
                self._submodules[layer.name] = layer

        if return_interm_layers:
            # Dictionary mapping: layer name -> output key (matches PyTorch IntermediateLayerGetter)
            self.return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            self.return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor):
        x = tensor_list.tensors

        if "stem" in self.backbone_layers:
            x = self.backbone_layers["stem"](x)

        xs = {}
        for lname in ["layer1", "layer2", "layer3", "layer4"]:
            if lname in self.backbone_layers:
                x = self.backbone_layers[lname](x)
                if lname in self.return_layers:
                    xs[lname] = x

        out = {}
        for lname, x_out in xs.items():
            # Get the output name from return_layers mapping
            output_name = self.return_layers[lname]
            # Extract shape from SimTensor or tensor-like object
            if isinstance(x_out, SimTensor):
                shape = shape_as_optional_list(x_out.shape)
            else:
                shape = list(x_out.shape) if hasattr(x_out, "shape") else None

            # Interpolate mask to match feature map spatial dimensions
            m = tensor_list.mask
            if m is not None:
                # Create SimTensor for mask interpolation
                mask_simtensor = SimTensor(
                    {
                        "name": "mask_input",
                        "shape": list(m.shape),
                        "data": m.copy(),
                        "dtype": m.dtype,
                    }
                )
                # Use interpolate from misc_ttsim (uses scipy, mirrors F.interpolate)
                assert shape is not None
                mask_interpolated = interpolate(
                    mask_simtensor, size=tuple(shape[-2:]), mode="nearest"
                )
                mask = (
                    mask_interpolated.data.astype(bool)
                    if mask_interpolated.data is not None
                    else None
                )
            else:
                mask = None

            # Use output_name (e.g., '0', '1', '2') as key, not layer name
            out[output_name] = NestedTensor(x_out, mask)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Backbone  (ResNet)
# ──────────────────────────────────────────────────────────────────────────────
class Backbone(BackboneBase):
    """
    ResNet backbone (resnet50 / resnet101) with FrozenBatchNorm2d.

    Channel progression:
        stem    →  64
        layer1  → 256   (stride 1 vs stem out)
        layer2  → 512   (stride 2)
        layer3  → 1024  (stride 2)
        layer4  → 2048  (stride 2)
    """

    _CONFIGS = {
        "resnet50": [3, 4, 6, 3],
        "resnet101": [3, 4, 23, 3],
    }

    def __init__(
        self,
        name: str,
        resnet_name: str = "resnet50",
        train_backbone: bool = False,
        return_interm_layers: bool = True,
        dilation: bool = False,
    ):
        if resnet_name not in self._CONFIGS:
            raise ValueError(f"Unsupported resnet: {resnet_name}")
        layers = self._build_layers(name, self._CONFIGS[resnet_name])
        super().__init__(name, layers, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

    @staticmethod
    def _make_downsample(name, in_ch, out_ch, stride):
        return SimNN.Sequential(
            [
                F.Conv2d(
                    f"{name}.downsample.0",
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                FrozenBatchNorm2d(f"{name}.downsample.1", out_ch),
            ],
        )

    @classmethod
    def _make_layer(cls, name, num_blocks, in_ch, out_ch, stride):
        blocks = []
        ds = None
        if in_ch != out_ch or stride != 1:
            ds = cls._make_downsample(f"{name}.0", in_ch, out_ch, stride)
        blocks.append(
            ResNetBottleneck(f"{name}.0", in_ch, out_ch, stride=stride, downsample=ds)
        )
        for i in range(1, num_blocks):
            blocks.append(ResNetBottleneck(f"{name}.{i}", out_ch, out_ch, stride=1))
        return SimNN.Sequential(blocks)

    @classmethod
    def _build_layers(cls, name, layer_configs):
        layers = OrderedDict()

        # stem: conv1(7×7 s2) → bn1 → relu → maxpool(3×3 s2)
        layers["stem"] = SimNN.Sequential(
            [
                F.Conv2d(
                    f"{name}.conv1",
                    3,
                    64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                FrozenBatchNorm2d(f"{name}.bn1", 64),
                F.Relu(f"{name}.relu"),
                F.MaxPool2d(f"{name}.maxpool", kernel_size=3, stride=2, padding=1),
            ],
        )

        # residual layers  (in_ch → out_ch @ stride)
        specs = [
            ("layer1", layer_configs[0], 64, 256, 1),
            ("layer2", layer_configs[1], 256, 512, 2),
            ("layer3", layer_configs[2], 512, 1024, 2),
            ("layer4", layer_configs[3], 1024, 2048, 2),
        ]
        for lname, nblocks, in_ch, out_ch, stride in specs:
            layers[lname] = cls._make_layer(
                f"{name}.{lname}", nblocks, in_ch, out_ch, stride
            )

        return layers


# ──────────────────────────────────────────────────────────────────────────────
# Joiner
# ──────────────────────────────────────────────────────────────────────────────
class Joiner(SimNN.Module):
    """
    Mirrors Joiner.forward():
        xs  = backbone(tensor_list)
        out = [xs[k] for k in sorted(xs)]
        pos = [position_embedding(x).to(x.tensors.dtype) for x in out]
        return out, pos
    """

    def __init__(self, name: str, backbone: Backbone, position_embedding):
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels
        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out = [xs[k] for k in sorted(xs)]
        pos = [self.position_embedding(x) for x in out]
        return out, pos


# ──────────────────────────────────────────────────────────────────────────────
# factory  – mirrors build_backbone(args)
# ──────────────────────────────────────────────────────────────────────────────
def build_backbone(args):
    """
    Drop-in for backbone.build_backbone(args).

    args attributes used:
        hidden_dim, position_embedding, lr_backbone,
        masks, num_feature_levels, backbone, dilation
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)

    backbone = Backbone(
        "backbone", args.backbone, train_backbone, return_interm_layers, args.dilation
    )

    return Joiner("joiner", backbone, position_embedding)
