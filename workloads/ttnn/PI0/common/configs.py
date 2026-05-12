# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common configurations for PI0 model components on Polaris.

This module contains dataclass configs shared across the Polaris PI0 workload.
These configs are framework-agnostic and can be used by reference code,
weight-loading code, and Polaris runtime/model wrappers.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GemmaConfig:
    """Configuration for Gemma transformer."""

    width: int = 2048
    depth: int = 18
    mlp_dim: int = 16384
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0

    @classmethod
    def gemma_2b(cls) -> "GemmaConfig":
        """Gemma 2B configuration used for the VLM backbone."""
        return cls(
            width=2048,
            depth=18,
            mlp_dim=16384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,  # 8 x 128 = 1024 width
        )

    @classmethod
    def gemma_300m(cls) -> "GemmaConfig":
        """Gemma 300M configuration used for the action expert."""
        return cls(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=128,  # 8 x 128 = 1024 width
        )


@dataclass
class SigLIPConfig:
    """Configuration for SigLIP vision encoder."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-6

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass
class SuffixConfig:
    """Configuration for suffix embedding."""

    action_dim: int = 32
    action_horizon: int = 50
    expert_width: int = 1024
    state_dim: int = 32
    time_emb_dim: int = 1024
    pi05: bool = False


@dataclass
class PrefixConfig:
    """Configuration for prefix embedding."""

    vlm_hidden_size: int = 2048
    num_image_tokens: int = 256  # Tokens per image from SigLIP
    max_lang_tokens: int = 512


@dataclass
class PaliGemmaConfig:
    """Configuration for the PaliGemma backbone."""

    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)
    max_seq_len: int = 2048


@dataclass
class DenoiseConfig:
    """Configuration for denoising."""

    num_steps: int = 10
    noise_scale: float = 1.0
    action_dim: int = 32
    action_horizon: int = 50



@dataclass
class PI0ModelConfig:
    """Top-level PI0 model configuration for Polaris."""

    # Core dimensions
    action_dim: int = 32
    action_horizon: int = 50
    state_dim: int = 32

    # Model variants
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    # Runtime / processing
    precision: str = "bfloat16"
    num_denoising_steps: int = 10
    max_seq_len: int = 2048

    # PI05 mode (uses adaRMS instead of fused action-time)
    pi05: bool = False

    # Component configs
    vlm_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_2b)
    expert_config: GemmaConfig = field(default_factory=GemmaConfig.gemma_300m)
    siglip_config: SigLIPConfig = field(default_factory=SigLIPConfig)

    @staticmethod
    def _build_vlm_config(variant: str) -> GemmaConfig:
        if variant == "gemma_2b":
            return GemmaConfig.gemma_2b()
        raise ValueError(f"Unsupported paligemma_variant: {variant}")

    @staticmethod
    def _build_expert_config(variant: str) -> GemmaConfig:
        if variant == "gemma_300m":
            return GemmaConfig.gemma_300m()
        raise ValueError(f"Unsupported action_expert_variant: {variant}")

    def __post_init__(self):
        # Only derive configs from variants if caller did not explicitly override them.
        if self.vlm_config == GemmaConfig.gemma_2b() and self.paligemma_variant != "gemma_2b":
            self.vlm_config = self._build_vlm_config(self.paligemma_variant)

        if self.expert_config == GemmaConfig.gemma_300m() and self.action_expert_variant != "gemma_300m":
            self.expert_config = self._build_expert_config(self.action_expert_variant)

    @property
    def prefix_config(self) -> PrefixConfig:
        return PrefixConfig(
            vlm_hidden_size=self.vlm_config.width,
            num_image_tokens=self.siglip_config.num_patches,
            max_lang_tokens=512,
        )

    @property
    def suffix_config(self) -> SuffixConfig:
        return SuffixConfig(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            expert_width=self.expert_config.width,
            state_dim=self.state_dim,
            time_emb_dim=self.expert_config.width,
            pi05=self.pi05,
        )

    @property
    def denoise_config(self) -> DenoiseConfig:
        return DenoiseConfig(
            num_steps=self.num_denoising_steps,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )

    @property
    def paligemma_config(self) -> PaliGemmaConfig:
        return PaliGemmaConfig(
            vlm_config=self.vlm_config,
            expert_config=self.expert_config,
            siglip_config=self.siglip_config,
            max_seq_len=self.max_seq_len,
        )
    
    #in tt metal code --def __post_init__(self): this part unconditionally
    # overwrites everything It ignores: paligemma_variant ,action_expert_variant ,any configs passed by the user