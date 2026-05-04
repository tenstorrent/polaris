#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim version of Base3DDetector.

Original file: mmdet3d/models/detectors/base.py

Base3DDetector is an abstract shell that:
  - Routes forward() to forward_train / forward_test
  - Provides show_results() for visualization

In TTSim there are no learnable parameters and no training / test routing.
This module is a pure structural shell used as base class for all EA-LSS
detector TTSim modules.

Params: 0
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.op import _from_shape


class Base3DDetector(SimNN.Module):
    """
    Abstract base class for all 3D detectors.

    In TTSim this serves as a pass-through shell with no learnable weights.
    Concrete sub-classes override ``__call__`` to implement the full
    inference graph.

    Args:
        name (str): Module prefix.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        super().link_op2module()

    def __call__(self, x):
        """Identity passthrough – overridden by concrete detectors."""
        return x

    def analytical_param_count(self, lvl: int = 0) -> int:
        return 0
