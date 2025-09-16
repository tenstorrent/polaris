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
        # Set DDIM-specific defaults
        config.setdefault('eta', 0.0)  # Default to deterministic
        config.setdefault('prediction_type', 'epsilon')
        super().__init__(**config)

    def step(self, model_output, timestep, sample) -> Dict[str, Any]:
        # DDIM step parameters
        idx = self._timestep_to_index.get(int(timestep), 0)

        # Get alpha cumulative products
        alpha_cum = self._abar[idx] if self._abar else 1.0
        alpha_cum_prev = self._abar[idx + 1] if (self._abar and idx + 1 < len(self._abar)) else 1.0

        # Get sigma values
        sigma = self._sigmas[idx] if self._sigmas else 0.0
        sigma_prev = self._sigmas[idx + 1] if (self._sigmas and idx + 1 < len(self._sigmas)) else 0.0

        # Provide a generic gamma for downstream consumers expecting an update scale
        dt = float(abs(sigma_prev - sigma))
        gamma = max(dt, 1e-3)

        return {
            'alpha_cum': float(alpha_cum),
            'alpha_cum_prev': float(alpha_cum_prev),
            'eta': self.config.get('eta', 0.0),
            'prediction_type': self.config.get('prediction_type', 'epsilon'),
            'timestep': int(timestep),
            'sigma': float(sigma),
            'sigma_prev': float(sigma_prev),
            'gamma': gamma,
        }


