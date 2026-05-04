#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of NaiveSyncBatchNorm variants.

Original file: mmdet3d/ops/norm.py

Provides:
  - NaiveSyncBatchNorm1d  (registered as 'naiveSyncBN1d')
  - NaiveSyncBatchNorm2d  (registered as 'naiveSyncBN2d')
  - NaiveSyncBatchNorm3d  (registered as 'naiveSyncBN3d')

At **inference** (the only mode TTSim targets) all three variants reduce to
standard batch normalisation using stored running_mean / running_var:

    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias

The distributed AllReduce logic is never executed at inference time; it
therefore has no TTSim equivalent and is simply omitted.

Implementation:
    Each class wraps the BatchNorm1d module from mlp.py (which already
    implements the full BN graph via SimOpHandle("BatchNormalization")).
    For 2-D and 3-D inputs extra Transpose ops are added to massage the
    trailing spatial dimensions, keeping the channel dimension at position 1
    as required by the ONNX BatchNormalization spec.

No torch / mmcv imports.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import SimOpHandle, _from_shape, _from_data

_ealss_root = os.path.abspath(os.path.join(current_dir, ".."))
if _ealss_root not in sys.path:
    sys.path.insert(0, _ealss_root)
from ttsim_modules.mlp import BatchNorm1d


# ---------------------------------------------------------------------------
# NaiveSyncBatchNorm1d
# ---------------------------------------------------------------------------

class NaiveSyncBatchNorm1d(SimNN.Module):
    """
    TTSim NaiveSyncBatchNorm1d — inference-only standard BN for 2-D or 3-D
    tensors of shape (N, C) or (N, C, L).

    Equivalent to torch.nn.BatchNorm1d in eval() mode.

    Args:
        name (str): Unique op-name prefix.
        num_features (int): C — number of channels / features.
        eps (float): Variance epsilon. Default: 1e-5.
        momentum (float): Momentum (unused at inference; stored). Default: 0.1.
    """

    def __init__(self, name: str, num_features: int,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self._bn = BatchNorm1d(name + ".bn", num_features, eps=eps, momentum=momentum)
        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): (N, C) or (N, C, L).
        Returns:
            SimTensor: same shape as input.
        """
        # BatchNorm1d handles both (N, C) and (N, C, L) via the fixed
        # compute_batchnorm (dimension-agnostic reshape).
        return self._bn(x)

    def set_weights(self, weight_np, bias_np, running_mean_np, running_var_np):
        """Inject inference-mode statistics and affine params."""
        self._bn.set_weights(weight_np, bias_np, running_mean_np, running_var_np)

    def analytical_param_count(self, lvl=0):
        return self._bn.analytical_param_count(lvl + 1)


# ---------------------------------------------------------------------------
# NaiveSyncBatchNorm2d
# ---------------------------------------------------------------------------

class NaiveSyncBatchNorm2d(SimNN.Module):
    """
    TTSim NaiveSyncBatchNorm2d — inference-only standard BN for 4-D tensors
    of shape (N, C, H, W).

    Uses the same SimOpHandle("BatchNormalization") as NaiveSyncBatchNorm1d.

    Args:
        name (str): Unique op-name prefix.
        num_features (int): C.
        eps (float): Default: 1e-5.
        momentum (float): Default: 0.1.
    """

    def __init__(self, name: str, num_features: int,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.name = name
        self.num_features = num_features

        self.scale = _from_shape(name + ".scale", [num_features], is_param=True)
        self.scale.op_in.append(name)
        self.bias_bn = _from_shape(name + ".bias", [num_features], is_param=True)
        self.bias_bn.op_in.append(name)
        self.running_mean = _from_shape(name + ".running_mean",
                                        [num_features], is_param=True)
        self.running_mean.op_in.append(name)
        self.running_var = _from_shape(name + ".running_var",
                                       [num_features], is_param=True)
        self.running_var.op_in.append(name)

        self.eps = eps
        self.momentum = momentum
        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [N, C, H, W].
        Returns:
            SimTensor: [N, C, H, W].
        """
        bn_op = SimOpHandle(
            self.name + ".batchnorm",
            "BatchNormalization",
            params=[
                (1, self.scale),
                (2, self.bias_bn),
                (3, self.running_mean),
                (4, self.running_var),
            ],
            ipos=[0],
            epsilon=self.eps,
        )
        bn_op.implicit_inputs.extend(
            [self.scale, self.bias_bn, self.running_mean, self.running_var]
        )
        return bn_op(x)

    def set_weights(self, weight_np, bias_np, running_mean_np, running_var_np):
        self.scale.data = weight_np.astype(np.float32)
        self.bias_bn.data = bias_np.astype(np.float32)
        self.running_mean.data = running_mean_np.astype(np.float32)
        self.running_var.data = running_var_np.astype(np.float32)

    def analytical_param_count(self, lvl=0):
        return 2 * self.num_features  # weight + bias (running stats not trained)


# ---------------------------------------------------------------------------
# NaiveSyncBatchNorm3d
# ---------------------------------------------------------------------------

class NaiveSyncBatchNorm3d(SimNN.Module):
    """
    TTSim NaiveSyncBatchNorm3d — inference-only standard BN for 5-D tensors
    of shape (N, C, D, H, W).

    Uses the same SimOpHandle("BatchNormalization") as the 2-D variant,
    relying on the dimension-agnostic ``compute_batchnorm`` in data_compute.py.

    Args:
        name (str): Unique op-name prefix.
        num_features (int): C.
        eps (float): Default: 1e-5.
        momentum (float): Default: 0.1.
    """

    def __init__(self, name: str, num_features: int,
                 eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.name = name
        self.num_features = num_features

        self.scale = _from_shape(name + ".scale", [num_features], is_param=True)
        self.scale.op_in.append(name)
        self.bias_bn = _from_shape(name + ".bias", [num_features], is_param=True)
        self.bias_bn.op_in.append(name)
        self.running_mean = _from_shape(name + ".running_mean",
                                        [num_features], is_param=True)
        self.running_mean.op_in.append(name)
        self.running_var = _from_shape(name + ".running_var",
                                       [num_features], is_param=True)
        self.running_var.op_in.append(name)

        self.eps = eps
        self.momentum = momentum
        super().link_op2module()

    def __call__(self, x):
        """
        Args:
            x (SimTensor): [N, C, D, H, W].
        Returns:
            SimTensor: [N, C, D, H, W].
        """
        bn_op = SimOpHandle(
            self.name + ".batchnorm",
            "BatchNormalization",
            params=[
                (1, self.scale),
                (2, self.bias_bn),
                (3, self.running_mean),
                (4, self.running_var),
            ],
            ipos=[0],
            epsilon=self.eps,
        )
        bn_op.implicit_inputs.extend(
            [self.scale, self.bias_bn, self.running_mean, self.running_var]
        )
        return bn_op(x)

    def set_weights(self, weight_np, bias_np, running_mean_np, running_var_np):
        self.scale.data = weight_np.astype(np.float32)
        self.bias_bn.data = bias_np.astype(np.float32)
        self.running_mean.data = running_mean_np.astype(np.float32)
        self.running_var.data = running_var_np.astype(np.float32)

    def analytical_param_count(self, lvl=0):
        return 2 * self.num_features
