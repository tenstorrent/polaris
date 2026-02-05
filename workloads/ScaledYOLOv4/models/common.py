#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import numpy as np
import math


def DWConv(name, c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(name, c1, c2, k, s, g=math.gcd(c1, c2), act=act)


def autopad(k, p=None):
    """Auto-padding to 'same'"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MP(SimNN.Module):
    """MaxPooling downsampling layer"""

    def __init__(self, name, k=2):
        super().__init__()
        self.name = name

        if not hasattr(self, "_ops"):
            self._ops = {}

        self.m = F.MaxPool2d(self.name + ".maxpool", kernel_size=k, stride=k)
        super().link_op2module()

    def __call__(self, x):
        self._ops[self.m.name] = self.m
        result = self.m(x)
        self._tensors[result.name] = result
        return result


class Flatten(SimNN.Module):
    """Flatten layer - reshapes [B,C,H,W] → [B,C*H*W]"""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.reshape = F.Reshape(self.name + ".flatten")
        super().link_op2module()

    def __call__(self, x):
        B = x.shape[0]
        # Flatten all dimensions after batch: [B, C, H, W] → [B, C*H*W]
        flattened_size = 1
        for dim in x.shape[1:]:
            flattened_size *= dim
        new_shape = F._from_data(
            self.name + ".shape",
            np.array([B, flattened_size], dtype=np.int64),
            is_const=True,
        )
        self._tensors[new_shape.name] = new_shape
        return self.reshape(x, new_shape)


class Concat(SimNN.Module):
    """Concatenate tensors along dimension (default: channel dimension)"""

    def __init__(self, name, dimension=1):
        super().__init__()
        self.name = name
        self.d = dimension

        if not hasattr(self, "_ops"):
            self._ops = {}

        self.concat = F.ConcatX(self.name + ".concat", axis=self.d)
        super().link_op2module()

    def __call__(self, x):
        # x is a list of tensors to concatenate
        self._ops[self.concat.name] = self.concat
        result = self.concat(*x)
        self._tensors[result.name] = result
        return result

    def analytical_param_count(self, lvl):
        return 0


class Conv(SimNN.Module):
    def __init__(self, name, c1, c2, k=1, s=1, p=None, g=1, act="Mish"):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.name = name
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        self.stride = s
        self.padding = p
        self.group = g
        assert act in ["Mish", "Identity"], f"Illegal activation=({act})!!"
        self.conv = F.Conv2d(
            self.name + ".conv",
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias=False,
        )
        self.bn = F.BatchNorm2d(self.name + ".bn", c2)
        self.act = (
            F.Mish(self.name + ".mish")
            if act == "Mish"
            else F.Identity(self.name + ".identity")
        )
        super().link_op2module()

    def __call__(self, x):
        y = self.conv(x)
        z = self.bn(y)
        o = self.act(z)
        return o

    def fuseforward(self, x):
        # Fused forward: Conv → Activation (skips BN)
        y = self.conv(x)
        o = self.act(y)
        return o

    def analytical_param_count(self, lvl):
        # Assumes bias=False, because we use BatchNorm2d
        # For grouped conv: each group has (c1/g) input channels and (c2/g) output channels
        conv_params = (
            (self.in_channels // self.group)
            * self.out_channels
            * self.kernel_size
            * self.kernel_size
        )
        bn_params = 2 * self.out_channels  # (weight, bias)
        return conv_params + bn_params


class ConvSig(SimNN.Module):
    def __init__(self, name, c1, c2, k=1, s=1, p=None, g=1, act=True):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.name = name
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k
        self.stride = s
        self.padding = p
        self.group = g

        self.conv = F.Conv2d(
            self.name + ".conv",
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias=False,
        )
        self.act = (
            F.Sigmoid(self.name + ".sigmoid")
            if act
            else F.Identity(self.name + ".identity")
        )
        super().link_op2module()

    def __call__(self, x):
        y = self.conv(x)
        return self.act(y)

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def analytical_param_count(self, lvl):
        # Conv params only (no BatchNorm in ConvSig)
        conv_params = (
            (self.in_channels // self.group)
            * self.out_channels
            * self.kernel_size
            * self.kernel_size
        )
        return conv_params


class SPP(SimNN.Module):
    """Spatial Pyramid Pooling - Multi-scale feature extraction"""

    def __init__(self, name, c1, c2, k=(5, 9, 13)):
        super().__init__()
        self.name = name
        c_ = c1 // 2  # hidden channels

        # Conv modules
        self.cv1 = Conv(self.name + ".cv1", c1, c_, 1, 1)
        self.cv2 = Conv(self.name + ".cv2", c_ * (len(k) + 1), c2, 1, 1)

        # MaxPool layers with different kernel sizes - store as individual attributes
        self.m0 = F.MaxPool2d(
            self.name + ".maxpool_0", kernel_size=k[0], stride=1, padding=k[0] // 2
        )
        self.m1 = F.MaxPool2d(
            self.name + ".maxpool_1", kernel_size=k[1], stride=1, padding=k[1] // 2
        )
        self.m2 = F.MaxPool2d(
            self.name + ".maxpool_2", kernel_size=k[2], stride=1, padding=k[2] // 2
        )

        # Concat operator
        self.concat = Concat(self.name + ".concat", dimension=1)

        super().link_op2module()

    def __call__(self, x):
        x = self.cv1(x)
        # Concatenate [x, pool1(x), pool2(x), pool3(x)]
        p0 = self.m0(x)
        p1 = self.m1(x)
        p2 = self.m2(x)
        spp_out = self.concat([x, p0, p1, p2])
        return self.cv2(spp_out)

    def analytical_param_count(self, lvl):
        params = 0
        params += self.cv1.analytical_param_count(lvl)
        params += self.cv2.analytical_param_count(lvl)
        return params


class SPPCSP(SimNN.Module):
    """CSP SPP - Cross Stage Partial Spatial Pyramid Pooling"""

    def __init__(self, name, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        self.name = name
        self.in_channels = c1
        c_ = int(2 * c2 * e)  # hidden channels

        # Conv modules (Conv2d + BatchNorm + Mish)
        self.cv1 = Conv(self.name + ".cv1", c1, c_, 1, 1)
        self.cv3 = Conv(self.name + ".cv3", c_, c_, 3, 1)
        self.cv4 = Conv(self.name + ".cv4", c_, c_, 1, 1)
        self.cv5 = Conv(self.name + ".cv5", 4 * c_, c_, 1, 1)
        self.cv6 = Conv(self.name + ".cv6", c_, c_, 3, 1)
        self.cv7 = Conv(self.name + ".cv7", 2 * c_, c2, 1, 1)

        # Bare Conv2d (no BatchNorm, no activation)
        self.cv2 = F.Conv2d(
            self.name + ".cv2", c1, c_, kernel_size=1, stride=1, bias=False
        )

        # MaxPool layers with different kernel sizes - store as individual attributes
        self.m0 = F.MaxPool2d(
            self.name + ".maxpool_0", kernel_size=k[0], stride=1, padding=k[0] // 2
        )
        self.m1 = F.MaxPool2d(
            self.name + ".maxpool_1", kernel_size=k[1], stride=1, padding=k[1] // 2
        )
        self.m2 = F.MaxPool2d(
            self.name + ".maxpool_2", kernel_size=k[2], stride=1, padding=k[2] // 2
        )

        # Standalone BatchNorm + Mish
        self.bn = F.BatchNorm2d(self.name + ".bn", 2 * c_)
        self.act = F.Mish(self.name + ".mish")

        # Concat operators
        self.concat1 = Concat(self.name + ".concat1", dimension=1)  # For SPP outputs
        self.concat2 = Concat(self.name + ".concat2", dimension=1)  # For CSP merge

        super().link_op2module()

    def __call__(self, x):
        # SPP path
        x1 = self.cv1(x)
        x1 = self.cv3(x1)
        x1 = self.cv4(x1)

        # Spatial pyramid pooling - concat [x1, pool1(x1), pool2(x1), pool3(x1)]
        p0 = self.m0(x1)
        p1 = self.m1(x1)
        p2 = self.m2(x1)
        spp_out = self.concat1([x1, p0, p1, p2])

        y1 = self.cv5(spp_out)
        y1 = self.cv6(y1)

        # CSP path
        y2 = self.cv2(x)

        # Merge CSP paths
        merged = self.concat2([y1, y2])
        normalized = self.bn(merged)
        activated = self.act(normalized)
        output = self.cv7(activated)

        return output

    def analytical_param_count(self, lvl):
        # 6 Conv modules + 1 bare Conv2d + 1 BatchNorm
        params = 0
        params += self.cv1.analytical_param_count(lvl)
        params += self.cv3.analytical_param_count(lvl)
        params += self.cv4.analytical_param_count(lvl)
        params += self.cv5.analytical_param_count(lvl)
        params += self.cv6.analytical_param_count(lvl)
        params += self.cv7.analytical_param_count(lvl)

        # cv2 bare Conv2d parameters
        # cv2 is SimOpHandle, use stored values
        c1 = self.cv1.in_channels
        c2 = self.cv1.out_channels
        k = 1
        params += c1 * c2 * k * k

        # bn parameters
        # BatchNorm2d (2 * c_) - weight and bias
        # self.cv7.out_channels is c2, e=0.5, so 2 * (2 * c2 * 0.5) = 2 * c2
        params += 2 * c2

        return params


class Focus(SimNN.Module):
    """Focus - Pixel space to channel space conversion (1 layer)"""

    def __init__(self, name, c1, c2, k=1, s=1, p=None, g=1, act="Mish"):
        super().__init__()
        self.name = name

        if not hasattr(self, "_ops"):
            self._ops = {}

        # Conv module (processes 4× channels after slicing)
        self.conv = Conv(self.name + ".conv", c1 * 4, c2, k, s, p, g, act)

        super().link_op2module()

    def __call__(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape

        # Axes for spatial slicing only (ONNX compatibility)
        axes = F._from_data(
            self.name + ".axes", np.array([2, 3], dtype=np.int64), is_const=True
        )
        self._tensors[axes.name] = axes

        # Create slice operators (4 patterns for checkerboard slicing)
        # Pattern 1: x[..., ::2, ::2] - even rows, even cols
        slice1 = F.SliceF(self.name + ".slice1", out_shape=[B, C, H // 2, W // 2])
        starts1 = F._from_data(
            self.name + ".starts1", np.array([0, 0], dtype=np.int64), is_const=True
        )
        ends1 = F._from_data(
            self.name + ".ends1", np.array([H, W], dtype=np.int64), is_const=True
        )
        steps1 = F._from_data(
            self.name + ".steps1", np.array([2, 2], dtype=np.int64), is_const=True
        )

        # Pattern 2: x[..., 1::2, ::2] - odd rows, even cols
        slice2 = F.SliceF(self.name + ".slice2", out_shape=[B, C, H // 2, W // 2])
        starts2 = F._from_data(
            self.name + ".starts2", np.array([1, 0], dtype=np.int64), is_const=True
        )
        ends2 = F._from_data(
            self.name + ".ends2", np.array([H, W], dtype=np.int64), is_const=True
        )
        steps2 = F._from_data(
            self.name + ".steps2", np.array([2, 2], dtype=np.int64), is_const=True
        )

        # Pattern 3: x[..., ::2, 1::2] - even rows, odd cols
        slice3 = F.SliceF(self.name + ".slice3", out_shape=[B, C, H // 2, W // 2])
        starts3 = F._from_data(
            self.name + ".starts3", np.array([0, 1], dtype=np.int64), is_const=True
        )
        ends3 = F._from_data(
            self.name + ".ends3", np.array([H, W], dtype=np.int64), is_const=True
        )
        steps3 = F._from_data(
            self.name + ".steps3", np.array([2, 2], dtype=np.int64), is_const=True
        )

        # Pattern 4: x[..., 1::2, 1::2] - odd rows, odd cols
        slice4 = F.SliceF(self.name + ".slice4", out_shape=[B, C, H // 2, W // 2])
        starts4 = F._from_data(
            self.name + ".starts4", np.array([1, 1], dtype=np.int64), is_const=True
        )
        ends4 = F._from_data(
            self.name + ".ends4", np.array([H, W], dtype=np.int64), is_const=True
        )
        steps4 = F._from_data(
            self.name + ".steps4", np.array([2, 2], dtype=np.int64), is_const=True
        )

        # Register all tensors
        for t in [
            starts1,
            ends1,
            steps1,
            starts2,
            ends2,
            steps2,
            starts3,
            ends3,
            steps3,
            starts4,
            ends4,
            steps4,
        ]:
            self._tensors[t.name] = t

        # Register slice operators
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        self.slice4 = slice4
        for op in [self.slice1, self.slice2, self.slice3, self.slice4]:
            self._ops[op.name] = op

        # Perform slicing with explicit axes
        s1 = self.slice1(x, starts1, ends1, axes, steps1)  # [B, C, H/2, W/2]
        s2 = self.slice2(x, starts2, ends2, axes, steps2)  # [B, C, H/2, W/2]
        s3 = self.slice3(x, starts3, ends3, axes, steps3)  # [B, C, H/2, W/2]
        s4 = self.slice4(x, starts4, ends4, axes, steps4)  # [B, C, H/2, W/2]

        # Register slice outputs
        for t in [s1, s2, s3, s4]:
            self._tensors[t.name] = t

        # Concatenate along channel dimension: [B, 4C, H/2, W/2]
        self.concat_op = F.ConcatX(self.name + ".concat", axis=1)
        self._ops[self.concat_op.name] = self.concat_op
        concatenated = self.concat_op(s1, s2, s3, s4)
        self._tensors[concatenated.name] = concatenated

        # Apply convolution
        return self.conv(concatenated)

    def analytical_param_count(self, lvl):
        return self.conv.analytical_param_count(lvl)


class Classify(SimNN.Module):
    """Classification head - converts feature maps to class predictions

    Transforms spatial features x(b,c1,h,w) to class logits x(b,c2).

    Architecture:
    1. AdaptiveAvgPool2d(1) - Global pooling: (b,c1,h,w) → (b,c1,1,1)
    2. Conv2d(c1→c2, k×k)   - Feature transform: (b,c1,1,1) → (b,c2,1,1)
    3. Flatten              - Remove spatial dims: (b,c2,1,1) → (b,c2)
    """

    def __init__(self, name, c1, c2, k=1, s=1, p=None, g=1):
        # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.name = name
        self.in_channels = c1
        self.out_channels = c2
        self.kernel_size = k

        # AdaptiveAvgPool2d(1) - global average pooling
        self.aap = F.AdaptiveAvgPool2d(self.name + ".aap", output_size=1)

        # Conv2d for feature transformation (no BatchNorm in Classify)
        self.conv = F.Conv2d(
            self.name + ".conv",
            c1,
            c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p),
            groups=g,
            bias=False,
        )

        # Flatten layer
        self.flat = Flatten(self.name + ".flat")

        # Concat for handling list inputs
        self.concat = Concat(self.name + ".concat", dimension=1)

        super().link_op2module()

    def __call__(self, x):
        # Handle list of tensors: concatenate along channel dimension
        if isinstance(x, list):
            # Apply AdaptiveAvgPool to each tensor in the list
            pooled_list = [self.aap(y) for y in x]
            # Concatenate: [(b,c1,1,1), (b,c2,1,1), ...] → (b,c1+c2+...,1,1)
            z = self.concat(pooled_list)
        else:
            # Single tensor: just pool
            z = self.aap(x)

        # Conv2d: (b,c_pooled,1,1) → (b,c2,1,1)
        conv_out = self.conv(z)

        # Flatten: (b,c2,1,1) → (b,c2)
        return self.flat(conv_out)

    def analytical_param_count(self, lvl):
        # Only Conv2d parameters (AdaptiveAvgPool and Flatten have no params)
        conv_params = (
            (self.in_channels // self.conv.groups)
            * self.out_channels
            * self.kernel_size
            * self.kernel_size
        )
        return conv_params


class VoVCSP(SimNN.Module):
    """VoV-CSP (VoVNet + Cross Stage Partial) - Efficient feature extraction

    Architecture:
    1. Chunk input into 2 halves along channel dimension
    2. Process second half through 2 Conv layers
    3. Concatenate intermediate + final
    4. Merge with 1×1 Conv
    """

    def __init__(self, name, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.name = name
        c_ = int(c2)  # hidden channels

        if not hasattr(self, "_ops"):
            self._ops = {}

        # Conv modules
        self.cv1 = Conv(self.name + ".cv1", c1 // 2, c_ // 2, 3, 1)
        self.cv2 = Conv(self.name + ".cv2", c_ // 2, c_ // 2, 3, 1)
        self.cv3 = Conv(self.name + ".cv3", c_, c2, 1, 1)

        # Concat operator
        self.concat = Concat(self.name + ".concat", dimension=1)

        super().link_op2module()

    def __call__(self, x):
        # x shape: [B, c1, H, W]
        B, C, H, W = x.shape

        # Chunk into 2 halves: [B, c1//2, H, W] and [B, c1//2, H, W]
        # We only use the second half (x1)
        half_c = C // 2

        # Slice second half: x[:, c1//2:, :, :]
        # We're slicing along dimension 1 (channel dimension)
        self.slice_x1 = F.SliceF(self.name + ".slice_x1", out_shape=[B, half_c, H, W])

        # Axes: which dimensions to slice (dimension 1 = channel)
        axes = F._from_data(
            self.name + ".axes", np.array([1], dtype=np.int64), is_const=True
        )
        starts = F._from_data(
            self.name + ".starts", np.array([half_c], dtype=np.int64), is_const=True
        )
        ends = F._from_data(
            self.name + ".ends", np.array([C], dtype=np.int64), is_const=True
        )
        steps = F._from_data(
            self.name + ".steps", np.array([1], dtype=np.int64), is_const=True
        )

        # Register slice operation and tensors
        self._ops[self.slice_x1.name] = self.slice_x1
        self._tensors[axes.name] = axes
        self._tensors[starts.name] = starts
        self._tensors[ends.name] = ends
        self._tensors[steps.name] = steps

        # Get second half
        x1 = self.slice_x1(x, starts, ends, axes, steps)  # [B, c1//2, H, W]
        self._tensors[x1.name] = x1

        # Process through conv layers
        x1 = self.cv1(x1)  # [B, c_//2, H, W]
        x2 = self.cv2(x1)  # [B, c_//2, H, W]

        # Concatenate x1 and x2: [B, c_, H, W]
        concatenated = self.concat([x1, x2])

        # Final 1×1 conv
        return self.cv3(concatenated)

    def analytical_param_count(self, lvl):
        params = 0
        params += self.cv1.analytical_param_count(lvl)
        params += self.cv2.analytical_param_count(lvl)
        params += self.cv3.analytical_param_count(lvl)
        return params


# -------------------TTSIM----------------------#


class Upsample(SimNN.Module):
    """
    Wrapper for nn.Upsample using F.Resize (nn.Upsample maps to onnx.Resize).
    Supports size or scale_factor based upsampling with various modes.
    YAML config format: [size, scale_factor, mode] e.g. [None, 2, 'nearest']
    """

    def __init__(self, name, size=None, scale_factor=None, mode="nearest"):
        super().__init__()
        self.name = name
        self.in_channels = None
        # Either size or scale_factor must be provided
        assert (
            size is not None or scale_factor is not None
        ), f"Upsample requires either 'size' or 'scale_factor'"
        self.resize = F.Resize(name + ".upsample", scale_factor=scale_factor, mode=mode)

        super().link_op2module()

    def analytical_param_count(self, lvl):
        return 0

    def __call__(self, x):
        return self.resize(x)


class Bottleneck(SimNN.Module):
    def __init__(
        self, name, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        self.name = name
        self.in_channels = None
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.cv1 = Conv(name + ".cv1", c1, c_, 1, 1)
        self.cv2 = Conv(name + ".cv2", c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

        if self.add:
            self.residual_add = F.Add(name + ".res_add")

        super().link_op2module()

    def __call__(self, x):
        y = self.cv1(x)
        z = self.cv2(y)
        if self.add:
            o = self.residual_add(x, z)
        else:
            o = z
        return o

    def analytical_param_count(self, lvl):
        return self.cv1.analytical_param_count(
            lvl + 1
        ) + self.cv2.analytical_param_count(lvl + 1)


class BottleneckCSP(SimNN.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, name, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.name = name
        self.in_channels = c1
        c_ = int(c2 * e)  # hidden channels
        self.c_ = c_
        self.cv1 = Conv(name + ".cv1", c1, c_, 1, 1)
        self.cv2 = F.Conv2d(name + ".cv2", c1, c_, kernel_size=1, stride=1, bias=False)
        self.cv3 = F.Conv2d(name + ".cv3", c_, c_, kernel_size=1, stride=1, bias=False)
        # Store in/out channels for SimOpHandle
        self.cv2_in_channels = c1
        self.cv2_out_channels = c_
        self.cv3_in_channels = c_
        self.cv3_out_channels = c_
        self.cv4 = Conv(name + ".cv4", 2 * c_, c2, 1, 1)
        self.bn = F.BatchNorm2d(name + ".bn", 2 * c_)
        self.act = F.Mish(name + ".mish")
        self.m = SimNN.ModuleList(
            [Bottleneck(name + f".m{i}", c_, c_, shortcut, g, e=1.0) for i in range(n)]
        )
        self.cat = F.ConcatX(name + ".cat", axis=1)

        super().link_op2module()

    def __call__(self, x):
        y1 = self.cv1(x)
        for m_ in self.m:
            y1 = m_(y1)
        y1 = self.cv3(y1)
        y2 = self.cv2(x)
        y = self.cat(y1, y2)
        y = self.bn(y)
        y = self.act(y)
        y = self.cv4(y)
        return y

    def analytical_param_count(self, lvl):
        # cv2: F.Conv2d (c1 -> c_, 1x1, bias=False)
        cv2_params = self.cv2_in_channels * self.cv2_out_channels * 1 * 1
        # cv3: F.Conv2d (c_ -> c_, 1x1, bias=False)
        cv3_params = self.cv3_in_channels * self.cv3_out_channels * 1 * 1
        # bn: BatchNorm2d (2 * c_) - weight and bias
        bn_params = 2 * (2 * self.c_)
        return (
            self.cv1.analytical_param_count(lvl + 1)
            + cv2_params
            + cv3_params
            + self.cv4.analytical_param_count(lvl + 1)
            + bn_params
            + sum([m_.analytical_param_count(lvl + 1) for m_ in self.m]) # type: ignore[attr-defined]
        )


class BottleneckCSP2(SimNN.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, name, c1, c2, n=1, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.name = name
        self.in_channels = c1
        c_ = int(c2)
        self.c_ = c_
        self.c_ = c_
        self.c_ = c_
        self.cv1 = Conv(name + ".cv1", c1, c_, 1, 1)
        self.cv2 = F.Conv2d(name + ".cv2", c_, c_, kernel_size=1, stride=1, bias=False)
        self.cv3 = Conv(name + ".cv3", 2 * c_, c2, 1, 1)
        # Store in/out channels for SimOpHandle
        self.cv2_in_channels = c_
        self.cv2_out_channels = c_
        self.bn = F.BatchNorm2d(name + ".bn", 2 * c_)
        self.act = F.Mish(name + ".mish")
        self.m = SimNN.ModuleList(
            [Bottleneck(name + f".m{i}", c_, c_, shortcut, g, e=1.0) for i in range(n)]
        )
        self.cat = F.ConcatX(name + ".cat", axis=1)

        super().link_op2module()

    def __call__(self, x):
        x1 = self.cv1(x)
        y1 = x1
        for m_ in self.m:
            y1 = m_(y1)
        y2 = self.cv2(x1)
        y = self.cat(y1, y2)
        y = self.bn(y)
        y = self.act(y)
        return self.cv3(y)

    def analytical_param_count(self, lvl):
        # cv2: F.Conv2d (c_ -> c_, 1x1, bias=False)
        cv2_params = self.cv2_in_channels * self.cv2_out_channels * 1 * 1
        # bn: BatchNorm2d (2 * c_) - weight and bias
        bn_params = 2 * (2 * self.c_)
        return (
            self.cv1.analytical_param_count(lvl + 1)
            + cv2_params
            + self.cv3.analytical_param_count(lvl + 1)
            + bn_params
            + sum([m_.analytical_param_count(lvl + 1) for m_ in self.m]) # type: ignore[attr-defined]
        )


class DWConvLayer(SimNN.Module):
    """Depthwise Convolution Layer: DWConv (groups=in_channels) + BatchNorm"""

    def __init__(self, name, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = in_channels
        self.kernel_size = 3
        self.stride = stride

        # Depthwise conv: groups = in_channels, kernel=3x3, padding=1
        self.dwconv = F.Conv2d(
            name + ".dwconv",
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=bias,
        )
        self.norm = F.BatchNorm2d(name + ".norm", in_channels)

        super().link_op2module()

    def __call__(self, x):
        y = self.dwconv(x)
        y = self.norm(y)
        return y

    def analytical_param_count(self, lvl):
        # Depthwise conv: groups=in_channels, so each group has 1 input channel
        # kernel is 3x3, output channels = in_channels (one filter per group)
        dwconv_params = self.in_channels * 1 * self.kernel_size * self.kernel_size
        # BatchNorm: weight + bias
        bn_params = 2 * self.in_channels
        return dwconv_params + bn_params


class ConvLayer(SimNN.Module):
    """Conv + BatchNorm + ReLU6 layer"""

    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel=3,
        stride=1,
        dropout=0.1,
        bias=False,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel
        self.stride = stride

        self.conv = F.Conv2d(
            name + ".conv",
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=1,
            bias=bias,
        )
        self.norm = F.BatchNorm2d(name + ".norm", out_channels)
        self.relu6 = F.Relu6(name + ".relu6")

        super().link_op2module()

    def __call__(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu6(y)
        return y

    def analytical_param_count(self, lvl):
        # Conv: in_channels * out_channels * kernel * kernel
        conv_params = (
            self.in_channels * self.out_channels * self.kernel_size * self.kernel_size
        )
        # BatchNorm: weight + bias
        bn_params = 2 * self.out_channels
        return conv_params + bn_params


class CombConvLayer(SimNN.Module):
    """Combined Conv Layer: ConvLayer + DWConvLayer"""

    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel=1,
        stride=1,
        dropout=0.1,
        bias=False,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer1 = ConvLayer(
            name + ".layer1", in_channels, out_channels, kernel=kernel
        )
        self.layer2 = DWConvLayer(
            name + ".layer2", out_channels, out_channels, stride=stride
        )

        super().link_op2module()

    def __call__(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return y

    def analytical_param_count(self, lvl):
        return self.layer1.analytical_param_count(
            lvl + 1
        ) + self.layer2.analytical_param_count(lvl + 1)
