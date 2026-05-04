#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of FFN (Feed-Forward Network) detection head.

Original file: mmdet3d/models/dense_heads/transfusion_head.py (class at ~line 160)

FFN is a multi-head prediction network that maps BEV query features to
per-attribute outputs (center, height, dim, rot, vel, heatmap, …).

For each attribute head defined in ``heads``:
    heads[head_name] = (num_classes, num_conv)

The network is:
    (num_conv-1) × ConvModule1d(c_in → head_conv, k=1, BN1d, ReLU)
    +   1        × Conv1d(head_conv → num_classes, k=1, bias=True)

Input shape:  [B, in_channels, P]  (per-proposal features)
Output:       dict  {head_name: SimTensor of shape [B, num_classes, P]}

Parameters per head:
    Intermediate ConvModule1d:
        Conv1d(c_in  → head_conv, k=1, no bias) + BN1d(head_conv):
            c_in * head_conv + 2 * head_conv
        Subsequent Conv1d(head_conv → head_conv) + BN1d:
            head_conv^2 + 2 * head_conv
    Final Conv1d(head_conv → num_classes, bias=True):
        head_conv * num_classes + num_classes

No torch / mmcv imports.
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
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.mlp import BatchNorm1d, ConvModule1d


class FFN(SimNN.Module):
    """
    TTSim FFN prediction head.

    Implements per-attribute Conv1d stacks that decode transformer query
    features (shape [B, in_channels, P]) into per-attribute predictions.

    Args:
        name (str): Unique module name prefix.
        in_channels (int): Number of input channels.
        heads (dict): {head_name: (num_classes, num_conv)}.
            Example: {'center': (2, 2), 'heatmap': (10, 2)}.
        head_conv (int): Intermediate channel width. Default: 64.
        final_kernel (int): Convolution kernel size. Default: 1.

    Shape:
        - Input:  (B, in_channels, P)
        - Output: dict{head_name: (B, num_classes, P)}

    Notes:
        - bias='auto' in the original → conv has no bias when BN follows,
          bias=True for the final conv layer.
        - final_kernel=1 so padding = final_kernel//2 = 0.
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        heads: dict,
        head_conv: int = 64,
        final_kernel: int = 1,
    ):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.heads = heads            # {head_name: (num_classes, num_conv)}
        self.head_conv = head_conv
        self.final_kernel = final_kernel

        for head_name, (num_classes, num_conv) in heads.items():
            c_in = in_channels

            # --- Intermediate ConvModule1d blocks ---
            for i in range(num_conv - 1):
                conv_module = ConvModule1d(
                    f"{name}.{head_name}.cm{i}",
                    in_channels=c_in,
                    out_channels=head_conv,
                    kernel_size=final_kernel,
                    with_bn=False,   # BN created separately for eps/momentum control
                    with_relu=True,
                )
                setattr(self, f"{head_name}_cm{i}", conv_module)

                bn_module = BatchNorm1d(
                    f"{name}.{head_name}.bn{i}",
                    num_features=head_conv,
                )
                setattr(self, f"{head_name}_bn{i}", bn_module)
                c_in = head_conv

            # --- Final Conv1d (no BN, bias=True) ---
            # The original always uses head_conv as input to the final layer
            # (even for num_conv=1), so overwrite c_in if needed.
            final_in = head_conv if num_conv >= 1 else in_channels
            pad = final_kernel // 2
            final_conv = ConvModule1d(
                f"{name}.{head_name}.final",
                in_channels=final_in,
                out_channels=num_classes,
                kernel_size=final_kernel,
                with_bn=False,
                with_relu=False,
            )
            setattr(self, f"{head_name}_final", final_conv)

        super().link_op2module()

    def __call__(self, x):
        """
        Forward pass.

        Args:
            x (SimTensor): [B, in_channels, P]

        Returns:
            dict{str: SimTensor}: Each value is [B, num_classes_head, P].
        """
        ret = {}
        for head_name, (num_classes, num_conv) in self.heads.items():
            h = x
            for i in range(num_conv - 1):
                cm = getattr(self, f"{head_name}_cm{i}")
                bn = getattr(self, f"{head_name}_bn{i}")
                h = bn(cm(h))
            final = getattr(self, f"{head_name}_final")
            h = final(h)
            ret[head_name] = h
        return ret

    def analytical_param_count(self, lvl: int = 0) -> int:
        """
        Per head:
          Intermediate cm_i (no bias + BN):
              c_in_i * head_conv * k   (conv, no bias, k=final_kernel)
              2 * head_conv            (BN scale + bias)
          Final conv (bias=True, no BN):
              head_conv * num_classes * k + num_classes
        """
        total = 0
        for head_name, (num_classes, num_conv) in self.heads.items():
            c_in = self.in_channels
            k = self.final_kernel

            for i in range(num_conv - 1):
                # Conv1d(c_in → head_conv), no bias since BN follows
                total += c_in * self.head_conv * k
                # BN
                total += 2 * self.head_conv
                c_in = self.head_conv

            # Final Conv1d(head_conv → num_classes, bias=True)
            final_in = self.head_conv if num_conv >= 1 else self.in_channels
            total += final_in * num_classes * k + num_classes

        return total
