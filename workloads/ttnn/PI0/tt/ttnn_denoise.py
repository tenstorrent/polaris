# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Denoising module - TTNN/TTSim Implementation.
"""

from typing import Callable, Optional
import ttsim.front.ttnn as ttnn
from workloads.ttnn.PI0.common.configs import DenoiseConfig


class DenoisingModuleTTNN:
    """
    Flow matching denoising module using TTNN/TTSim.
    """

    def __init__(
        self,
        config: DenoiseConfig,
        forward_fn: Callable,
        device,
    ):
        self.config = config
        self.forward_fn = forward_fn
        self.device = device

        # Precompute scalar timestep values
        self.timesteps = []
        self.dt_values = []

        num_steps = self.config.num_steps

        for i in range(num_steps + 1):
            t = 1.0 - (i / num_steps)
            self.timesteps.append(t)

        for i in range(num_steps):
            dt = self.timesteps[i + 1] - self.timesteps[i]
            self.dt_values.append(dt)

    def sample_noise(
        self,
        batch_size: int,
    ):
        """
        Create initial noise tensor.

        Polaris/TTSim does not support random generation,
        so use zeros placeholder tensor.
        """

        return ttnn.zeros(
            (
                batch_size,
                self.config.action_horizon,
                self.config.action_dim,
            ),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def denoise_step(
        self,
        x_t,
        timestep: float,
        dt: float,
        prefix_kv_cache=None,
        **forward_kwargs,
    ):
        """
        Single Euler denoising step.
        """

        v_t = self.forward_fn(
            x_t,
            timestep,
            prefix_kv_cache,
            **forward_kwargs,
        )

        velocity_scaled = ttnn.multiply(v_t, dt)

        x_next = ttnn.add(
            x_t,
            velocity_scaled,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return x_next

    def sample_actions(
        self,
        batch_size: int,
        prefix_kv_cache=None,
        **forward_kwargs,
    ):
        """
        Full denoising loop.
        """

        x_t = self.sample_noise(batch_size)

        for i in range(self.config.num_steps):

            timestep = self.timesteps[i]
            dt = self.dt_values[i]

            x_t = self.denoise_step(
                x_t=x_t,
                timestep=timestep,
                dt=dt,
                prefix_kv_cache=prefix_kv_cache,
                **forward_kwargs,
            )

        return x_t


class KVCacheManagerTTNN:
    """
    Simplified KV cache manager for TTNN/TTSim.
    """

    def __init__(self):
        self.cache = []
        self.seq_len = 0

    def initialize(self):
        self.cache = []
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        new_k,
        new_v,
    ):
        """
        Store KV tensors.
        """

        if layer_idx >= len(self.cache):
            self.cache.append((new_k, new_v))
        else:
            self.cache[layer_idx] = (new_k, new_v)

    def get(
        self,
        layer_idx: int,
    ):
        """
        Retrieve KV tensors.
        """

        return self.cache[layer_idx]

    def increment_seq_len(
        self,
        delta: int,
    ):
        self.seq_len += delta