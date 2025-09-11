#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from .base import SchedulerBase


class HeunDiscreteScheduler(SchedulerBase):
    """
    Minimal Heun scheduler adapter implementing SchedulerBase.
    Placeholder implementation matching base behavior for now.
    """

    def __init__(self, **config: Any) -> None:
        config.setdefault('prediction_type', 'epsilon')
        super().__init__(**config)

    def step(self, model_output, timestep, sample) -> Dict[str, Any]:
        # Heun scheduler step parameters
        idx = self._timestep_to_index.get(int(timestep), 0)

        sigma = self._sigmas[idx] if self._sigmas else 1.0
        sigma_next = self._sigmas[idx + 1] if (self._sigmas and idx + 1 < len(self._sigmas)) else 0.0
        dt = float(abs(sigma_next - sigma))
        gamma = max(1.5 * dt, 1e-3)

        return {
            'sigma': float(sigma),
            'sigma_next': float(sigma_next),
            'gamma': gamma,
            'prediction_type': self.config.get('prediction_type', 'epsilon'),
            'is_heun': True,
            'order': 2,
            'timestep': int(timestep)
        }


