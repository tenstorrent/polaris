#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


class SchedulerBase:
    """
    Minimal scheduler interface for Polaris workloads.

    Provides a common API compatible with TTSIM graph generation:
    - set_timesteps(num_inference_steps)
    - scale_model_input(latents, timestep) -> float | tensor-like
    - step(model_output, timestep, sample) -> dict (must include 'gamma')
    """

    def __init__(self, **config: Any) -> None:
        self.config: Dict[str, Any] = config
        self.num_train_timesteps: int = int(config.get('num_train_timesteps', 1000))
        self.beta_start: float = float(config.get('beta_start', 1e-4))
        self.beta_end: float = float(config.get('beta_end', 2e-2))
        self.beta_schedule: str = str(config.get('beta_schedule', 'scaled_linear'))
        # Number of inference steps used in the last call to set_timesteps
        # Exposed for tests and downstream consumers
        self.num_inference_steps: int = 0
        self.timesteps: List[int] = []
        # Precomputed schedules
        self._betas: np.ndarray | None = None
        self._alphas: np.ndarray | None = None
        self._alphas_cumprod: np.ndarray | None = None
        self._sigmas_all: np.ndarray | None = None
        # Selected-step views
        self._sigmas: List[float] = []
        self._abar: List[float] = []
        self._timestep_to_index: Dict[int, int] = {}

    def set_timesteps(self, num_inference_steps: int) -> None:
        # Record for external visibility
        self.num_inference_steps = int(num_inference_steps)
        step = max(self.num_train_timesteps // max(int(num_inference_steps), 1), 1)
        self.timesteps = list(range(self.num_train_timesteps - 1, -1, -step))[:num_inference_steps]
        # Build base schedules on first use
        if self._betas is None:
            self._build_base_schedules()
        # Slice schedules for selected timesteps
        assert self._sigmas_all is not None, "_sigmas_all should be initialized by _build_base_schedules"
        assert self._alphas_cumprod is not None, "_alphas_cumprod should be initialized by _build_base_schedules"
        sigmas = [float(self._sigmas_all[t]) for t in self.timesteps]
        abar = [float(self._alphas_cumprod[t]) for t in self.timesteps]
        self._sigmas = sigmas
        self._abar = abar
        self._timestep_to_index = {t: i for i, t in enumerate(self.timesteps)}

    def scale_model_input(self, latents, timestep) -> float:
        # Karras-style scaling approximation: x / sqrt(sigma^2 + 1)
        idx = self._timestep_to_index.get(int(timestep), 0)
        sigma = self._sigmas[idx] if self._sigmas else 1.0
        return float(1.0 / np.sqrt(sigma * sigma + 1.0))

    def step(self, model_output, timestep, sample) -> Dict[str, Any]:
        # Default: Euler-like step using sigma differences
        if not self._sigmas:
            return {"gamma": 1.0}
        idx = self._timestep_to_index.get(int(timestep), 0)
        sigma_cur = self._sigmas[idx]
        sigma_next = self._sigmas[idx + 1] if idx + 1 < len(self._sigmas) else 0.0
        dt = abs(sigma_next - sigma_cur)
        gamma = max(float(dt), 1e-3)
        return {"gamma": gamma}

    # Internal helpers
    def _build_base_schedules(self) -> None:
        # Simple beta schedules
        if self.beta_schedule == 'scaled_linear':
            # Align with common practice: scale linearly in sqrt space
            sqrt_beta = np.linspace(np.sqrt(self.beta_start), np.sqrt(self.beta_end), self.num_train_timesteps)
            betas = np.clip(sqrt_beta * sqrt_beta, 1e-8, 0.999)
        else:
            betas = np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # Convert to sigma schedule (variance-preserving)
        sigmas_all = np.sqrt((1.0 - alphas_cumprod) / np.maximum(alphas_cumprod, 1e-8))

        self._betas = betas
        self._alphas = alphas
        self._alphas_cumprod = alphas_cumprod
        self._sigmas_all = sigmas_all


