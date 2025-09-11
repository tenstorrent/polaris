#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any

from .base import SchedulerBase


class EulerDiscreteScheduler(SchedulerBase):
    """
    Euler Discrete scheduler adapter implementing SchedulerBase.
    Keeps the same behavior as the previous minimal stub for compatibility.
    """

    def __init__(self, **config):
        # Set default values if not provided
        config.setdefault('num_train_timesteps', 1000)
        config.setdefault('beta_start', 1e-4)
        config.setdefault('beta_end', 2e-2)
        config.setdefault('prediction_type', 'epsilon')
        config.setdefault('beta_schedule', 'scaled_linear')

        super().__init__(**config)
        self.num_inference_steps = 0

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Override to store num_inference_steps."""
        super().set_timesteps(num_inference_steps)
        self.num_inference_steps = num_inference_steps

    def step(self, model_output, timestep, sample) -> Dict[str, Any]:
        """Euler step with proper parameter format."""
        idx = self._timestep_to_index.get(int(timestep), 0)
        sigma = self._sigmas[idx] if self._sigmas else 1.0
        sigma_next = self._sigmas[idx + 1] if (self._sigmas and idx + 1 < len(self._sigmas)) else 0.0
        dt = float(abs(sigma_next - sigma))
        gamma = max(dt, 1e-3)

        return {
            'sigma': float(sigma),
            'gamma': gamma,
            'prediction_type': self.config.get('prediction_type', 'epsilon'),
            'timestep': int(timestep)
        }

    def save_pretrained(self, save_directory: str | os.PathLike) -> None:
        """Save scheduler configuration to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = save_directory / "scheduler_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs) -> "EulerDiscreteScheduler":
        """Load scheduler from pretrained directory."""
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_path = pretrained_model_name_or_path / "scheduler_config.json"

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # Merge with any additional kwargs
        config.update(kwargs)
        return cls(**config)

    @property
    def compatibles(self):
        """Get compatible schedulers."""
        return ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler"]

    def set_begin_index(self, begin_index: int = 0) -> None:
        """Set the begin index for timestep sampling."""
        self.begin_index = begin_index


