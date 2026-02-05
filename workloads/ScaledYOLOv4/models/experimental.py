#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# This file contains experimental modules
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from workloads.ScaledYOLOv4.models.common import Conv, DWConv

class CrossConv(SimNN.Module):
    # Cross Convolution Downsample
    def __init__(self, name, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # name, ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        self.name = name
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(self.name + '.cv1', c1, c_, (1, k), (1, s))
        self.cv2 = Conv(self.name + '.cv2', c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2
        super().link_op2module()

    def __call__(self, x):

        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

    def analytical_param_count(self, lvl):
        # CrossConv has two Conv blocks: cv1 and cv2
        cv1_params = self.cv1.analytical_param_count(lvl)
        cv2_params = self.cv2.analytical_param_count(lvl)
        return cv1_params + cv2_params


class C3(SimNN.Module):
    # Cross Convolution CSP
    def __init__(self, name, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        # name, ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.name = name
        self.c_ = int(c2 * e)  # hidden channels - store for analytical_param_count
        c_ = self.c_
        self.cv1 = Conv(self.name + '.cv1', c1, c_, 1, 1)
        self.cv2 = Conv(self.name + '.cv2', c1, c_, 1, 1)
        self.cv3 = Conv(self.name + '.cv3', c_, c_, 1, 1)
        self.cv4 = Conv(self.name + '.cv4', 2 * c_, c2, 1, 1)
        self.bn = F.BatchNorm2d(self.name + '.bn', 2 * c_)
        self.act = F.LeakyReLU(self.name + '.act', alpha=0.1)

        # Create CrossConv sequence
        self.m = SimNN.ModuleList([
            CrossConv(self.name + f'.CrossConv_{i}', c_, c_, 3, 1, g, 1.0, shortcut)
            for i in range(n)
        ])

        super().link_op2module()

    def __call__(self, x):
        # y1 = self.cv3(self.m(self.cv1(x)))
        y1_intermediate = self.cv1(x)
        for cross_conv in self.m:
            y1_intermediate = cross_conv(y1_intermediate)
        y1 = self.cv3(y1_intermediate)

        # y2 = self.cv2(x)
        y2 = self.cv2(x)

        # Concatenate along channel dimension (axis=1)
        y_cat = T.cat([y1, y2], dim=1)

        # Apply BatchNorm
        y_bn = self.bn(y_cat)

        # Apply LeakyReLU activation
        y_act = self.act(y_bn)

        # Apply final convolution
        return self.cv4(y_act)

    def analytical_param_count(self, lvl):
        # C3 has cv1, cv2, cv3, cv4 Conv blocks, bn, and n CrossConv modules
        cv1_params = self.cv1.analytical_param_count(lvl)
        cv2_params = self.cv2.analytical_param_count(lvl)
        cv3_params = self.cv3.analytical_param_count(lvl)
        cv4_params = self.cv4.analytical_param_count(lvl)

        # BatchNorm2d has 2 * channels learnable parameters (scale and bias)
        # The bn input has 2 * c_ channels (from concatenation)
        bn_params = 2 * (2 * self.c_)  # 2 * c_ is the input channels to bn

        # Count params from all CrossConv modules
        crossconv_params = sum(m.analytical_param_count(lvl) for m in self.m) # type: ignore[attr-defined]

        return cv1_params + cv2_params + cv3_params + cv4_params + bn_params + crossconv_params


class Sum(SimNN.Module):
    def analytical_param_count(self, lvl):
            # Only count weights if self.weight is True
        if self.weight:
            return self.w.shape[0]  # Returns n-1 (number of weight elements)
        return 0
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, name, n, weight=False):  # name, n: number of inputs, weight: apply weights boolean
        super().__init__()
        self.name = name
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            # Initialize weights: -arange(1, n) / 2 = [-0.5, -1.0, -1.5, ...]
            init_weights = -np.arange(1.0, n, dtype=np.float32) / 2
            self.w = F._from_data(self.name + '.w', init_weights, is_param=True)
            self.sigmoid = F.Sigmoid(self.name + '.sigmoid')
            self.mul_by_2 = F.MulFixed(self.name + '.mul2', 'scale', np.float32(2.0))
        super().link_op2module()

    def __call__(self, x):
        # Set link_module for all input tensors
        for tensor in x:
            if tensor.link_module is None:
                tensor.link_module = self

        y = x[0]  # no weight
        if self.weight:
            # Apply sigmoid and scale by 2
            w = self.mul_by_2(self.sigmoid(self.w))
            for i in self.iter:
                # Get w[i] by slicing - w has shape [n-1], need tuple for indexing
                w_i = w[(slice(i, i+1),)]  # Shape: [1]
                # Multiply and accumulate: x[i+1] * w[i]
                y = y + x[i + 1] * w_i
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(SimNN.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, name, c1, c2, k=1, s=1, g=1, act=True):  # name, ch_in, ch_out, kernel, stride, groups, activation
        super().__init__()
        self.name = name
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(self.name + '.cv1', c1, c_, k, s, None, g, act='Mish' if act else 'Identity')
        self.cv2 = Conv(self.name + '.cv2', c_, c_, 5, 1, None, c_, act='Mish' if act else 'Identity')
        super().link_op2module()

    def __call__(self, x):
        y = self.cv1(x)
        return T.cat([y, self.cv2(y)], dim=1)

    def analytical_param_count(self, lvl):
        # GhostConv has cv1 and cv2
        cv1_params = self.cv1.analytical_param_count(lvl)
        cv2_params = self.cv2.analytical_param_count(lvl)
        return cv1_params + cv2_params


class GhostBottleneck(SimNN.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, name, c1, c2, k, s):
        super().__init__()
        self.name = name
        c_ = c2 // 2
        self.s = s

        # Build conv modules list
        conv_list = []
        conv_list.append(GhostConv(self.name + '.conv_0', c1, c_, 1, 1))

        # Add DWConv or skip for middle layer
        if s == 2:
            conv_list.append(DWConv(self.name + '.conv_1', c_, c_, s))

        conv_list.append(GhostConv(self.name + '.conv_2', c_, c2, 1, 1, act=False))
        self.conv_modules = SimNN.ModuleList(conv_list)

        # Build shortcut modules list
        shortcut_list = []
        if s == 2:
            shortcut_list.append(DWConv(self.name + '.shortcut_0', c1, c1, s))
            shortcut_list.append(Conv(self.name + '.shortcut_1', c1, c2, 1, 1, None, 1, 'Identity'))
            self.shortcut_modules = SimNN.ModuleList(shortcut_list)

        super().link_op2module()

    def __call__(self, x):
        # Forward through conv path
        y = self.conv_modules[0](x)  # GhostConv

        if self.s == 2:
            # DWConv in middle (when stride=2)
            y = self.conv_modules[1](y)  # DWConv
            y = self.conv_modules[2](y)  # GhostConv
        else:
            # Skip middle layer (identity)
            y = self.conv_modules[1](y)  # GhostConv

        # Forward through shortcut path
        if self.s == 2:
            shortcut = self.shortcut_modules[0](x)  # DWConv
            shortcut = self.shortcut_modules[1](shortcut)  # Conv
        else:
            # Identity
            shortcut = x

        # Add conv output and shortcut
        return y + shortcut

    def analytical_param_count(self, lvl):
        # Sum params from all conv modules
        conv_params = sum(m.analytical_param_count(lvl) for m in self.conv_modules) # type: ignore[attr-defined]

        # Sum params from shortcut modules if used
        shortcut_params = 0
        if self.s == 2:
            shortcut_params = sum(m.analytical_param_count(lvl) for m in self.shortcut_modules) # type: ignore[attr-defined]

        return conv_params + shortcut_params


class MixConv2d(SimNN.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, name, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super().__init__()
        self.name = name
        self.c1 = c1  # Store for analytical_param_count
        self.c2 = c2
        self.k = k  # Store kernel sizes
        groups = len(k)

        if equal_ch:  # equal c_ per group
            # Use numpy instead of torch for computation
            i = np.linspace(0, groups - 1E-6, c2).astype(np.float32)
            i_floored = np.floor(i)
            c_ = []
            for g in range(groups):
                c_.append(int(np.sum(i_floored == g)))
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0]
            c_ = [int(round(val)) for val in c_]

        # Store conv ops as individual attributes (not in a list) so they're properly registered
        self.m_channels = []
        self.m_ops = []  # Keep track of operations for analytical_param_count
        for g in range(groups):
            # Use naming pattern m.0, m.1, etc. to match PyTorch ModuleList convention
            conv_op = F.Conv2d(self.name + f'.m.{g}', c1, int(c_[g]), kernel_size=k[g], stride=s, padding=k[g] // 2, bias=False)
            # Store as attribute with unique name (m0, m1, etc.)
            setattr(self, f'm{g}', conv_op)
            self.m_ops.append(f'm{g}')
            self.m_channels.append(int(c_[g]))

        self.bn = F.BatchNorm2d(self.name + '.bn', c2)
        self.act = F.LeakyReLU(self.name + '.act', alpha=0.1)
        super().link_op2module()

    def __call__(self, x):
        # Apply each convolution and concatenate results
        conv_results = []
        for op_name in self.m_ops:
            conv_op = getattr(self, op_name)
            result = conv_op(x)
            conv_results.append(result)

        # Concatenate along channel dimension (axis=1)
        y_cat = T.cat(conv_results, dim=1)

        # Apply BatchNorm
        y_bn = self.bn(y_cat)

        # Apply LeakyReLU activation
        y_act = self.act(y_bn)

        # Add residual connection
        return x + y_act

    def analytical_param_count(self, lvl):
        # MixConv2d has multiple conv modules and a BatchNorm
        # Each conv has no bias, so params = kernel * kernel * in_channels * out_channels
        conv_params = 0
        for i, out_ch in enumerate(self.m_channels):
            kernel_size = self.k[i]
            # params = k * k * c1 * c_[i]
            conv_params += kernel_size * kernel_size * self.c1 * out_ch

        # BatchNorm2d has 2 * channels learnable parameters (scale and bias)
        bn_params = 2 * self.c2
        return conv_params + bn_params
