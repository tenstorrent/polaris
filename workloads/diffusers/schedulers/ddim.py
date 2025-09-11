#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from .base import SchedulerBase


class DDIMScheduler(SchedulerBase):
    """
    Minimal DDIM scheduler adapter implementing SchedulerBase.
    Note: This is a placeholder with Euler-like behavior for now.
    """

    def __init__(self, **config: Any) -> None:
        super().__init__(**config)

    def step(self, model_output, timestep, sample) -> Dict[str, Any]:
        # DDIM: use eta=0 (deterministic) approximation: x_{t-1} = x_t + (sigma_next - sigma) * pred
        idx = self._timestep_to_index.get(int(timestep), 0)
        sigma = self._sigmas[idx] if self._sigmas else 1.0
        sigma_next = self._sigmas[idx + 1] if (self._sigmas and idx + 1 < len(self._sigmas)) else 0.0
        gamma = float(abs(sigma_next - sigma))
        return {"gamma": max(gamma, 1e-3)}


