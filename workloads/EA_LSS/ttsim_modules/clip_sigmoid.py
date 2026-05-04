#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim version of clip_sigmoid utility.

Clamped sigmoid function to avoid numerical instability in focal-loss computation.
All values are clipped to [eps, 1-eps] after sigmoid to prevent log(0) in loss.

Original file: mmdet3d/models/utils/clip_sigmoid.py
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
from ttsim.front.functional.op import SimOpHandle, _from_data, Sigmoid


def clip_sigmoid(x, eps=1e-4):
    """
    Sigmoid function for input feature with value clamping.

    Computes sigmoid(x) and then clamps the result to [eps, 1-eps] to avoid
    numerical instability (e.g. log(0)) in subsequent focal-loss calculations.

    This is a functional operation (no learnable parameters) that maps directly
    to two TTSim graph ops: Sigmoid -> Clip.

    Args:
        x (SimTensor): Input feature tensor of any shape, e.g. [B, N, H, W].
        eps (float): Lower bound of the clamp range. Upper bound is 1 - eps.
                     Defaults to 1e-4.

    Returns:
        SimTensor: Output tensor with same shape as input, values in [eps, 1-eps].

    Example:
        >>> import numpy as np
        >>> from ttsim.front.functional.op import _from_data
        >>> from ttsim_modules.clip_sigmoid import clip_sigmoid
        >>> x = _from_data("input", np.random.randn(2, 10, 8, 8).astype(np.float32))
        >>> out = clip_sigmoid(x, eps=1e-4)
        >>> # out.shape == (2, 10, 8, 8), values in [1e-4, 1-1e-4]
    """
    # Step 1: Apply sigmoid: y = 1 / (1 + exp(-x))
    sigmoid_op = Sigmoid(x.name + ".sigmoid")
    sigmoid_out = sigmoid_op(x)

    # Step 2: Clamp result to [eps, 1-eps].
    # ONNX Clip op takes (X, min, max) at positions 0, 1, 2.
    # We bake min/max in as constant params (same pattern as F.Relu6).
    min_tensor = _from_data(
        x.name + ".clip_min",
        np.array([eps], dtype=np.float32),
        is_const=True,
    )
    max_tensor = _from_data(
        x.name + ".clip_max",
        np.array([1.0 - eps], dtype=np.float32),
        is_const=True,
    )

    clip_op = SimOpHandle(
        x.name + ".clip",
        "Clip",
        params=[(1, min_tensor), (2, max_tensor)],
        ipos=[0],
    )
    # Register constant inputs so they appear in the graph
    clip_op.implicit_inputs.extend([min_tensor, max_tensor])

    clipped = clip_op(sigmoid_out)
    return clipped
