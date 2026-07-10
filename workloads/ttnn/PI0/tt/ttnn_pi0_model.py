# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Main PI0 model - TTSim Implementation (Inference / perf-model, shape-tracking).

This assembles the REAL PI0 components (SigLIP vision tower + PaliGemma VLM +
action expert + prefix/suffix embedders) into a Polaris workload, mirroring the
tt-metal reference `sample_actions` flow.

Scope note (per review):
- ttsim is a shape-tracking performance simulator, so weights/inputs are
  synthetic zeros of the correct shape; only the op graph + shapes matter.
- KV cache is intentionally OUT OF SCOPE for this first port. forward_vlm /
  forward_expert are run with use_cache=False (no cross-attention cache). The
  VLM prefix forward is still executed so its compute is modeled; its output is
  not fed into the expert (that would require the KV-cache path).
"""
from typing import Any, Dict, List

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

from workloads.ttnn.PI0.common.configs import (
    PI0ModelConfig,
    SigLIPConfig,
    GemmaConfig,
    SuffixConfig,
    PaliGemmaConfig,
)
from workloads.ttnn.PI0.tt.ttnn_paligemma import PaliGemmaBackboneTTNN
from workloads.ttnn.PI0.tt.ttnn_prefix import PrefixEmbeddingTTNN
from workloads.ttnn.PI0.tt.ttnn_suffix import SuffixEmbeddingTTNN


# =============================================================================
# Synthetic weight builders (perf model -> shapes only, values are zeros)
# Key patterns copied from the component PCC tests, which pass under ttsim.
# =============================================================================
def _z(shape: List[int], device: TTNNDevice) -> ttnn.Tensor:
    return ttnn.zeros(shape, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def _build_siglip_weights(cfg: SigLIPConfig, device: TTNNDevice) -> Dict[str, ttnn.Tensor]:
    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    w: Dict[str, ttnn.Tensor] = {
        "patch_embedding.weight": _z([hidden, 3, cfg.patch_size, cfg.patch_size], device),
        "patch_embedding.bias": _z([1, hidden], device),
        "position_embedding.weight": _z([1, num_patches, hidden], device),
        "post_layernorm.weight": _z([1, hidden], device),
        "post_layernorm.bias": _z([1, hidden], device),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{i}."
        w[f"{p}layer_norm1.weight"] = _z([1, hidden], device)
        w[f"{p}layer_norm1.bias"] = _z([1, hidden], device)
        w[f"{p}layer_norm2.weight"] = _z([1, hidden], device)
        w[f"{p}layer_norm2.bias"] = _z([1, hidden], device)
        w[f"{p}self_attn.q_proj.weight"] = _z([hidden, hidden], device)
        w[f"{p}self_attn.q_proj.bias"] = _z([1, hidden], device)
        w[f"{p}self_attn.k_proj.weight"] = _z([hidden, hidden], device)
        w[f"{p}self_attn.k_proj.bias"] = _z([1, hidden], device)
        w[f"{p}self_attn.v_proj.weight"] = _z([hidden, hidden], device)
        w[f"{p}self_attn.v_proj.bias"] = _z([1, hidden], device)
        w[f"{p}self_attn.out_proj.weight"] = _z([hidden, hidden], device)
        w[f"{p}self_attn.out_proj.bias"] = _z([1, hidden], device)
        w[f"{p}mlp.fc1.weight"] = _z([hidden, inter], device)
        w[f"{p}mlp.fc1.bias"] = _z([1, inter], device)
        w[f"{p}mlp.fc2.weight"] = _z([inter, hidden], device)
        w[f"{p}mlp.fc2.bias"] = _z([1, hidden], device)
    return w


def _build_gemma_weights(cfg: GemmaConfig, device: TTNNDevice) -> Dict[str, ttnn.Tensor]:
    width = cfg.width
    mlp_dim = cfg.mlp_dim
    q_dim = cfg.num_heads * cfg.head_dim
    kv_dim = cfg.num_kv_heads * cfg.head_dim
    w: Dict[str, ttnn.Tensor] = {
        "model.embed_tokens.weight": _z([10000, width], device),
        "model.norm.weight": _z([1, width], device),
    }
    for i in range(cfg.depth):
        p = f"model.layers.{i}."
        w[f"{p}input_layernorm.weight"] = _z([1, width], device)
        w[f"{p}post_attention_layernorm.weight"] = _z([1, width], device)
        w[f"{p}self_attn.q_proj.weight"] = _z([width, q_dim], device)
        w[f"{p}self_attn.k_proj.weight"] = _z([width, kv_dim], device)
        w[f"{p}self_attn.v_proj.weight"] = _z([width, kv_dim], device)
        w[f"{p}self_attn.o_proj.weight"] = _z([q_dim, width], device)
        w[f"{p}mlp.gate_proj.weight"] = _z([width, mlp_dim], device)
        w[f"{p}mlp.up_proj.weight"] = _z([width, mlp_dim], device)
        w[f"{p}mlp.down_proj.weight"] = _z([mlp_dim, width], device)
    return w


def _build_projector_weights(in_size: int, out_size: int, device: TTNNDevice) -> Dict[str, ttnn.Tensor]:
    return {
        "linear.weight": _z([out_size, in_size], device),
        "linear.bias": _z([1, out_size], device),
    }


def _build_paligemma_weights(cfg: PaliGemmaConfig, device: TTNNDevice) -> Dict[str, Dict[str, ttnn.Tensor]]:
    return {
        "vlm_vision": _build_siglip_weights(cfg.siglip_config, device),
        "vlm_language": _build_gemma_weights(cfg.vlm_config, device),
        "vlm_projector": _build_projector_weights(
            cfg.siglip_config.hidden_size, cfg.vlm_config.width, device
        ),
        "action_expert": _build_gemma_weights(cfg.expert_config, device),
    }


def _build_suffix_weights(cfg: SuffixConfig, device: TTNNDevice) -> Dict[str, ttnn.Tensor]:
    return {
        "action_in_proj.weight": _z([cfg.action_dim, cfg.expert_width], device),
        "action_in_proj.bias": _z([1, cfg.expert_width], device),
        "action_out_proj.weight": _z([cfg.expert_width, cfg.action_dim], device),
        "action_out_proj.bias": _z([1, cfg.action_dim], device),
        "state_proj.weight": _z([cfg.state_dim, cfg.expert_width], device),
        "state_proj.bias": _z([1, cfg.expert_width], device),
        "action_time_mlp_in.weight": _z([cfg.expert_width * 2, cfg.time_emb_dim], device),
        "action_time_mlp_in.bias": _z([1, cfg.time_emb_dim], device),
        "action_time_mlp_out.weight": _z([cfg.time_emb_dim, cfg.expert_width], device),
        "action_time_mlp_out.bias": _z([1, cfg.expert_width], device),
    }


# =============================================================================
# PI0 model
# =============================================================================
class PI0ModelTTNN:
    """Complete PI0 model (TTSim port) driving the real component stack."""

    def __init__(
        self,
        device: TTNNDevice,
        action_dim: int = 32,
        action_horizon: int = 50,
        state_dim: int = 32,
        num_denoising_steps: int = 10,
        max_seq_len: int = 2048,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        pi05: bool = False,
        bs: int = 1,
        image_size: int = 224,
        patch_size: int = 14,
        num_images: int = 1,
        lang_seq_len: int = 256,
        **_ignored: Any,
    ) -> None:
        self.device = device
        self.bs = bs
        self.num_images = num_images
        self.lang_seq_len = lang_seq_len

        # ---- Config ----
        cfg = PI0ModelConfig(
            action_dim=action_dim,
            action_horizon=action_horizon,
            state_dim=state_dim,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            num_denoising_steps=num_denoising_steps,
            max_seq_len=max_seq_len,
            pi05=pi05,
        )
        # Honor image/patch size from workload params.
        cfg.siglip_config = SigLIPConfig(image_size=image_size, patch_size=patch_size)
        self.config = cfg
        self.pi05 = pi05
        self.num_denoising_steps = num_denoising_steps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # ---- Build real components with synthetic weights ----
        self._init_components()
        self._init_inputs()

    def _init_components(self) -> None:
        pali_cfg = self.config.paligemma_config
        pali_weights = _build_paligemma_weights(pali_cfg, self.device)
        self.backbone = PaliGemmaBackboneTTNN(pali_cfg, pali_weights, self.device)

        suffix_cfg = self.config.suffix_config
        suffix_weights = _build_suffix_weights(suffix_cfg, self.device)
        self.suffix_embedding = SuffixEmbeddingTTNN(suffix_cfg, suffix_weights, self.device)

        prefix_cfg = self.config.prefix_config
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_cfg,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def _init_inputs(self) -> None:
        bs = self.bs
        # Synthetic image inputs (one per image).
        self.images = [
            _z([bs, 3, self.image_size, self.image_size], self.device)
            for _ in range(self.num_images)
        ]
        self.img_masks = [_z([bs, 1], self.device) for _ in range(self.num_images)]
        # Language tokens: embedding indices. NOTE(review): if ttsim.embedding
        # requires integer indices, change dtype/layout here (e.g. uint32,
        # ROW_MAJOR_LAYOUT). Starting from the shape the components expect.
        self.lang_tokens = ttnn.zeros(
            [bs, self.lang_seq_len],
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.lang_masks = _z([bs, self.lang_seq_len], self.device)
        # Robot state + initial noisy action.
        self.state = _z([bs, self.config.state_dim], self.device)
        self.x_t = _z([bs, self.config.action_horizon, self.config.action_dim], self.device)
        # Timestep tensor (batch_size,). Values irrelevant for shape sim.
        self.timestep = ttnn.zeros(
            [bs], device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def _scale_velocity(self, velocity: ttnn.Tensor, dt: float) -> ttnn.Tensor:
        dt_t = ttnn.full(
            velocity.shape, dt, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        scaled = ttnn.multiply(velocity, dt_t)
        ttnn.deallocate(dt_t)
        return scaled

    def sample_actions(self) -> ttnn.Tensor:
        """Full denoising loop mirroring the tt-metal reference (no KV cache)."""
        # Step 1: prefix (images + language) -> VLM forward (models vision+VLM).
        prefix_embs, _prefix_pad, _prefix_att = self.prefix_embedding.embed_prefix(
            self.images, self.img_masks, self.lang_tokens, self.lang_masks
        )
        # use_cache=False: prefix compute is modeled; no KV cache is built.
        _vlm_out, _ = self.backbone.forward_vlm(prefix_embs)

        x_t = self.x_t
        num_steps = self.num_denoising_steps
        dt = -1.0 / num_steps  # go from t=1.0 -> 0.0

        token_select = None
        if not self.pi05:
            token_select = _z(
                [self.bs, self.config.action_horizon, self.config.action_horizon + 1],
                self.device,
            )

        for _i in range(num_steps):
            # Step 2: suffix (state + noisy action + timestep).
            suffix_embs, _pad, _att, _cond = self.suffix_embedding.embed_suffix(
                self.state, x_t, self.timestep
            )
            # Step 3: expert forward (no KV cache / cross-attention).
            expert_output, _ = self.backbone.forward_expert(suffix_embs)

            # Step 4: extract action output (skip state token in PI0 mode).
            if not self.pi05:
                action_output = ttnn.matmul(token_select, expert_output)
            else:
                action_output = expert_output

            # Step 5: project to velocity + Euler step.
            velocity = self.suffix_embedding.project_output(action_output)
            velocity_scaled = self._scale_velocity(velocity, dt)
            x_t = ttnn.add(x_t, velocity_scaled)

        return x_t

    def __call__(self) -> ttnn.Tensor:
        return self.sample_actions()


# =============================================================================
# Polaris entry point
# =============================================================================
def run_pi0_model(wlname: str, device: TTNNDevice, cfg: Dict[str, Any]) -> ttnn.Tensor:
    """Polaris workload entry point: called as run_pi0_model(wlname, device, cfg)."""
    model = PI0ModelTTNN(device=device, **cfg)
    return model.sample_actions()
