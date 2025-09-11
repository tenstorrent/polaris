#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.graph.wl_graph import WorkloadGraph


class _BaseConditioning(SimNN.Module):
    """
    Minimal functional conditioning module.

    This produces a simple feature map from the current latent sample and
    scales it by a configurable strength. It serves as a ControlNet/T2I-Adapter
    placeholder for graph construction and performance estimation in TTSIM.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.strength = float(cfg.get('conditioning_strength', 1.0))
        self.hidden_channels = int(cfg.get('conditioning_hidden_channels', 8))

    def add_call(self, workload_graph: WorkloadGraph, sample: SimTensor,
                 encoder_hidden_states: Optional[SimTensor] = None, **kwargs) -> SimTensor:
        conv1 = F.Conv2d(f"{self.name}_conv1", sample.shape[1], self.hidden_channels, 3, stride=1, padding=1)
        act1 = F.Gelu(f"{self.name}_gelu1")
        conv2 = F.Conv2d(f"{self.name}_conv2", self.hidden_channels, sample.shape[1], 3, stride=1, padding=1)
        conv1.set_module(self)
        act1.set_module(self)
        conv2.set_module(self)

        x = conv1(sample)
        x = act1(x)
        x = conv2(x)

        s = F._from_data(f"{self.name}_strength", data=np.array(self.strength, dtype=np.float32), is_const=True)
        s.set_module(self)
        mul = F.Mul(f"{self.name}_scale")
        mul.set_module(self)
        return mul(x, s)


class ControlNetPolaris(_BaseConditioning):
    pass


class T2IAdapterPolaris(_BaseConditioning):
    pass


