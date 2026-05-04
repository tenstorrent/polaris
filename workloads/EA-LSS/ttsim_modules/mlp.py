#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim version of the MLP (Multi-Layer Perceptron) module.

Original file: mmdet3d/models/utils/mlp.py

The MLP applies stacked 1-D point-wise convolutions (Conv1d with kernel_size=1),
each followed by Batch Normalization 1D and ReLU activation.  This is equivalent
to a per-point linear projection with normalisation on features of shape (B, C, N).

Dependency replaced:
    mmcv.cnn.ConvModule  →  ConvModule1d (defined below as a TTSim Module)
    torch.nn             →  ttsim.front.functional.sim_nn / op
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data, Sigmoid


# ---------------------------------------------------------------------------
# BatchNorm1d
# ---------------------------------------------------------------------------

class BatchNorm1d(SimNN.Module):
    """
    TTSim implementation of 1-D Batch Normalization.

    Operates on input of shape (N, C, L) — batch × channels × length.
    At inference time (the only mode ttsim supports) the running statistics
    (mean / var) are used directly; there is no online update.

    Learnable parameters: scale (γ) of shape [C], bias (β) of shape [C].
    Running statistics (not trained): running_mean [C], running_var [C].

    Args:
        name (str): Unique module name used to prefix all internal op/tensor names.
        num_features (int): Number of channels C.
        eps (float): Small constant added to variance for numerical stability.
                     Default: 1e-5.
        momentum (float): Momentum for running statistics update (inference-only;
                          stored for completeness). Default: 0.1.

    Shape:
        - Input:  (N, C, L)
        - Output: (N, C, L) — same shape as input
    """

    def __init__(self, name: str, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable affine parameters (gamma / beta)
        self.scale = _from_shape(name + ".scale", [num_features], is_param=True)
        self.scale.op_in.append(name)

        self.bias_bn = _from_shape(name + ".bias", [num_features], is_param=True)
        self.bias_bn.op_in.append(name)

        # Running statistics (inference-only; treated as params in the graph)
        self.running_mean = _from_shape(name + ".running_mean",
                                        [num_features], is_param=True)
        self.running_mean.op_in.append(name)

        self.running_var = _from_shape(name + ".running_var",
                                       [num_features], is_param=True)
        self.running_var.op_in.append(name)

        # ONNX BatchNormalization: inputs are (X, scale, bias, mean, var)
        self.bn_op = SimOpHandle(
            name + ".bn",
            "BatchNormalization",
            params=[
                (1, self.scale),
                (2, self.bias_bn),
                (3, self.running_mean),
                (4, self.running_var),
            ],
            ipos=[0],
            epsilon=eps,
        )

        super().link_op2module()

    def __call__(self, x):
        """Forward pass: normalize x using running statistics and affine params.

        Args:
            x (SimTensor): Input of shape (N, C, L).

        Returns:
            SimTensor: Normalized output of shape (N, C, L).
        """
        return self.bn_op(x)

    def analytical_param_count(self, lvl: int = 0) -> int:
        # scale + bias (gamma + beta); running stats are not "trainable" params
        return 2 * self.num_features


# ---------------------------------------------------------------------------
# ConvModule1d  (replaces mmcv.cnn.ConvModule for 1-D case)
# ---------------------------------------------------------------------------

class ConvModule1d(SimNN.Module):
    """
    TTSim replacement for ``mmcv.cnn.ConvModule`` configured for 1-D convolutions.

    Applies Conv1d (kernel_size=1) → optional BatchNorm1d → optional ReLU.

    This is the exact sequence that mmcv.cnn.ConvModule produces when called with:
        conv_cfg=dict(type='Conv1d')
        norm_cfg=dict(type='BN1d')
        act_cfg=dict(type='ReLU')

    The convolution uses ONNX "Conv" with 1-D kernel (shape [C_out, C_in, 1]).
    All mmcv / torch dependencies are eliminated.

    Args:
        name (str): Unique module name.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 1.
        stride (int): Convolution stride. Default: 1.
        padding (int): Zero-padding added to both sides. Default: 0.
        dilation (int): Kernel dilation. Default: 1.
        groups (int): Blocked connections. Default: 1.
        bias (bool): Whether conv has a bias term. Default: True.
        with_bn (bool): Whether to include BatchNorm1d. Default: True.
        with_relu (bool): Whether to include ReLU activation. Default: True.

    Shape:
        - Input:  (N, C_in, L)
        - Output: (N, C_out, L_out)  where L_out depends on kernel/stride/padding.
                  For kernel_size=1, padding=0, stride=1 → L_out = L.
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_bn: bool = True,
        with_relu: bool = True,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.with_bn = with_bn
        self.with_relu = with_relu

        # ------------------------------------------------------------------
        # Conv1d weight: [C_out, C_in / groups, kernel_size]
        # ------------------------------------------------------------------
        conv_weight = _from_shape(
            name + ".conv.weight",
            [out_channels, in_channels // groups, kernel_size],
            is_param=True,
        )
        conv_weight.op_in.append(name + ".conv")

        conv_params = [(1, conv_weight)]

        if bias:
            conv_bias = _from_shape(
                name + ".conv.bias", [out_channels], is_param=True
            )
            conv_bias.op_in.append(name + ".conv")
            conv_params.append((2, conv_bias))
            self.conv_bias = conv_bias
        else:
            self.conv_bias = None

        self.conv_weight = conv_weight

        # ONNX Conv for 1-D: strides/pads/dilations are length-1 lists
        self.conv_op = SimOpHandle(
            name + ".conv",
            "Conv",
            params=conv_params,
            ipos=[0],
            strides=[stride],
            pads=[padding, padding],
            dilations=[dilation],
            group=groups,
        )

        # ------------------------------------------------------------------
        # (Optional) BatchNorm1d
        # ------------------------------------------------------------------
        if with_bn:
            self.bn = BatchNorm1d(name + ".bn", out_channels)
        else:
            self.bn = None

        # ------------------------------------------------------------------
        # (Optional) ReLU
        # ------------------------------------------------------------------
        if with_relu:
            self.relu_op = F.Relu(name + ".relu")
        else:
            self.relu_op = None

        super().link_op2module()

    def __call__(self, x):
        """Forward: x → Conv1d → [BN1d] → [ReLU].

        Args:
            x (SimTensor): Input of shape (N, C_in, L).

        Returns:
            SimTensor: Output of shape (N, C_out, L_out).
        """
        x = self.conv_op(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu_op is not None:
            x = self.relu_op(x)
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        count = self.in_channels * self.out_channels * self.kernel_size
        if self.conv_bias is not None:
            count += self.out_channels
        if self.bn is not None:
            count += self.bn.analytical_param_count()
        return count


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(SimNN.Module):
    """
    TTSim version of the MLP module.

    Passes features of shape (B, C, N) through a sequence of 1-D point-wise
    convolution layers, each with Batch Normalisation and ReLU activation.

    Replaces ``mmcv.cnn.ConvModule`` with the TTSim-native ``ConvModule1d``.
    No torch / mmcv / mmdet dependencies.

    Args:
        name (str): Unique module name.
        in_channel (int): Number of input feature channels. Default: 18.
        conv_channels (tuple[int]): Output channels for each successive layer.
                                    Default: (256, 256).
        with_bn (bool): Include BatchNorm1d in each layer. Default: True.
        with_relu (bool): Include ReLU in each layer. Default: True.
        bias (bool): Include bias in each Conv1d. Default: True.

    Shape:
        - Input:  (B, in_channel, N)
        - Output: (B, conv_channels[-1], N)

    Example:
        >>> from ttsim.front.functional.op import _from_shape
        >>> mlp = MLP("mlp", in_channel=18, conv_channels=(256, 256))
        >>> x = _from_shape("x", [2, 18, 128])
        >>> out = mlp(x)
        >>> print(out.shape)  # [2, 256, 128]
    """

    def __init__(
        self,
        name: str,
        in_channel: int = 18,
        conv_channels: tuple = (256, 256),
        with_bn: bool = True,
        with_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.name = name
        self.in_channel = in_channel
        self.conv_channels = tuple(conv_channels)

        # Build sequential list of ConvModule1d layers
        self.layers: list = []
        prev_ch = in_channel
        for i, out_ch in enumerate(conv_channels):
            layer = ConvModule1d(
                name=f"{name}.layer{i}",
                in_channels=prev_ch,
                out_channels=out_ch,
                kernel_size=1,
                padding=0,
                bias=bias,
                with_bn=with_bn,
                with_relu=with_relu,
            )
            self.layers.append(layer)
            # Register as attribute so link_op2module picks them up
            setattr(self, f"layer{i}", layer)
            prev_ch = out_ch

        super().link_op2module()

    def __call__(self, x):
        """Forward: pass x through all Conv1d-BN-ReLU layers sequentially.

        Args:
            x (SimTensor): Input of shape (B, in_channel, N).

        Returns:
            SimTensor: Output of shape (B, conv_channels[-1], N).
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        return sum(l.analytical_param_count() for l in self.layers)
