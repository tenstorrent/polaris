# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SigLIP Vision Tower - TTSim Implementation (FIXED VERSION)

Fixes included:
- Patch embedding weight lookup crash
- Empty weight dict handling
- Robust checkpoint key matching
- Undefined variable bugs
"""

import math
from typing import Dict, Optional

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.PI0.common.configs import SigLIPConfig


# ============================================================================
# Helper
# ============================================================================

def nearest_32(x: int) -> int:
    return ((x + 31) // 32) * 32


# ============================================================================
# Patch Embedding
# ============================================================================

class PatchEmbeddingTTNN:
    def __init__(
        self,
        config: SigLIPConfig,
        weights: Dict[str, ttnn.Tensor],
        device: TTNNDevice,
    ):
        self.config = config
        self.device = device
        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size

        if not weights:
            raise ValueError("PatchEmbeddingTTNN received EMPTY weights dict")

        # -----------------------------
        # Robust weight lookup
        # -----------------------------
        possible_weight_keys = [
            "patch_embedding.weight",
            "vision_model.embeddings.patch_embedding.weight",
            "embeddings.patch_embedding.weight",
            "conv.weight",
        ]

        conv_weight = None
        conv_bias = None

        for k in possible_weight_keys:
            if k in weights:
                conv_weight = weights[k]
                break

        bias_keys = [
            "patch_embedding.bias",
            "vision_model.embeddings.patch_embedding.bias",
            "embeddings.patch_embedding.bias",
            "conv.bias",
        ]

        for k in bias_keys:
            if k in weights:
                conv_bias = weights[k]
                break

        if conv_weight is None:
            raise ValueError(
                "Patch embedding weight not found.\n"
                f"Available keys sample: {list(weights.keys())[:20]}"
            )
        
        # -----------------------------
        # reshape conv -> linear
        # -----------------------------
        out_channels = conv_weight.shape[0]  # type: ignore[index]
        in_channels = conv_weight.shape[1] # type: ignore[index] 

        in_features = (
            in_channels * conv_weight.shape[2] * conv_weight.shape[3] # type: ignore[index]
        )

        self.in_features = in_features
        self.in_features_padded = nearest_32(in_features)

        # (out, c, h, w) -> (out, h, w, c)
        w = ttnn.permute(conv_weight, (0, 2, 3, 1))
        w = ttnn.reshape(w, (out_channels, in_features))
        w = ttnn.permute(w, (1, 0))

        self.linear_weight = w
        self.linear_bias = conv_bias

    def _unfold_conv2d(self, x):
        b, h, w, c = x.shape
        ps = self.patch_size

        ph = h // ps
        pw = w // ps

        x = ttnn.reshape(
            x,
            (b, ph, ps, pw, ps, c),
        )

        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))

        x = ttnn.reshape(
            x,
            (b, ph * pw, ps * ps * c),
        )

        return x

    def forward(self, pixel_values):
        x = ttnn.permute(pixel_values, (0, 2, 3, 1))
        x = self._unfold_conv2d(x)

        out = ttnn.linear(
            x,
            self.linear_weight,
            bias=self.linear_bias,
        )

        return out


# ============================================================================
# Attention
# ============================================================================

# ============================================================================
# Attention (FIXED)
# ============================================================================
class SigLIPAttentionTTNN:
    def __init__(self, config, weights, device):
        self.config = config
        self.device = device
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Get individual Q, K, V weights
        self.q_weight = weights.get("self_attn.q_proj.weight")
        self.k_weight = weights.get("self_attn.k_proj.weight")
        self.v_weight = weights.get("self_attn.v_proj.weight")
        self.q_bias = weights.get("self_attn.q_proj.bias")
        self.k_bias = weights.get("self_attn.k_proj.bias")
        self.v_bias = weights.get("self_attn.v_proj.bias")
        
        self.wo = weights.get("self_attn.out_proj.weight")
        self.bo = weights.get("self_attn.out_proj.bias")

    def forward(self, x):
        shape = x.shape
        if shape is None:
            raise ValueError("x must have a valid shape")
        
        b = shape[0]
        s = shape[1]
        h = shape[2] if len(shape) > 2 else self.hidden_size

        # Separate Q, K, V projections (instead of fused QKV)
        q = ttnn.linear(x, self.q_weight, bias=self.q_bias)
        k = ttnn.linear(x, self.k_weight, bias=self.k_bias)
        v = ttnn.linear(x, self.v_weight, bias=self.v_bias)

        # Reshape Q: [b, s, hidden] -> [b, num_heads, s, head_dim]
        q = ttnn.reshape(q, (b, s, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))  # [b, num_heads, s, head_dim]

        # Reshape K: [b, s, hidden] -> [b, num_heads, s, head_dim]
        k = ttnn.reshape(k, (b, s, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))  # [b, num_heads, s, head_dim]

        # Reshape V: [b, s, hidden] -> [b, num_heads, s, head_dim]
        v = ttnn.reshape(v, (b, s, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))  # [b, num_heads, s, head_dim]

        # Transpose K for attention: [b, num_heads, head_dim, s]
        k_t = ttnn.permute(k, (0, 1, 3, 2))

        # Attention scores: Q @ K^T
        attn = ttnn.matmul(q, k_t)
        attn = ttnn.multiply(attn, self.scale)
        attn = ttnn.softmax(attn, dim=-1)

        # Attention output: attn @ V
        out = ttnn.matmul(attn, v)

        # Reshape back: [b, num_heads, s, head_dim] -> [b, s, hidden]
        out = ttnn.permute(out, (0, 2, 1, 3))  # [b, s, num_heads, head_dim]
        out = ttnn.reshape(out, (b, s, self.hidden_size))

        # Output projection
        out = ttnn.linear(out, self.wo, bias=self.bo)

        return out
# ============================================================================
# MLP
# ============================================================================

class SigLIPMLPTTNN:
    def __init__(self, config, weights, device):
        self.fc1_w = weights.get("mlp.fc1.weight")
        self.fc1_b = weights.get("mlp.fc1.bias")

        self.fc2_w = weights.get("mlp.fc2.weight")
        self.fc2_b = weights.get("mlp.fc2.bias")

    def forward(self, x):
        x = ttnn.linear(x, self.fc1_w, bias=self.fc1_b)
        x = ttnn.gelu(x)
        x = ttnn.linear(x, self.fc2_w, bias=self.fc2_b)
        return x


# ============================================================================
# Block
# ============================================================================

class SigLIPBlockTTNN:
    def __init__(self, config, weights, device):
        self.ln1_w = weights.get("layer_norm1.weight")
        self.ln1_b = weights.get("layer_norm1.bias")

        self.ln2_w = weights.get("layer_norm2.weight")
        self.ln2_b = weights.get("layer_norm2.bias")

        self.attn = SigLIPAttentionTTNN(config, weights, device)
        self.mlp = SigLIPMLPTTNN(config, weights, device)

    def forward(self, x):
        n = ttnn.layer_norm(
            x,
            weight=self.ln1_w,
            bias=self.ln1_b,
            epsilon=1e-5,
        )

        x = ttnn.add(x, self.attn.forward(n))

        n = ttnn.layer_norm(
            x,
            weight=self.ln2_w,
            bias=self.ln2_b,
            epsilon=1e-5,
        )

        x = ttnn.add(x, self.mlp.forward(n))

        return x


# ============================================================================
# Vision Tower
# ============================================================================

class SigLIPVisionTowerTTNN:
    def __init__(self, config, weights, device):
        self.config = config
        self.device = device

        if not weights:
            raise ValueError(
                "SigLIPVisionTowerTTNN received EMPTY vlm_vision weights"
            )

        self.patch_embed = PatchEmbeddingTTNN(config, weights, device)

        self.position_embedding = weights.get(
            "position_embedding.weight"
        ) or weights.get(
            "vision_model.embeddings.position_embedding.weight"
        )

        self.blocks = []

        for i in range(config.num_hidden_layers):
            layer_weights = self._get_layer(weights, i)

            self.blocks.append(
                SigLIPBlockTTNN(config, layer_weights, device)
            )

        self.post_ln_w = weights.get("post_layernorm.weight")
        self.post_ln_b = weights.get("post_layernorm.bias")

    def _get_layer(self, weights, idx):
        prefix_list = [
            f"vision_model.encoder.layers.{idx}.",
            f"encoder.layers.{idx}.",
        ]

        out = {}

        for k, v in weights.items():
            for p in prefix_list:
                if k.startswith(p):
                    out[k[len(p):]] = v

        return out

    def forward(self, x):
        x = self.patch_embed.forward(x)

        if self.position_embedding is not None:
            x = ttnn.add(x, self.position_embedding)

        for b in self.blocks:
            x = b.forward(x)

        if self.post_ln_w is not None:
            x = ttnn.layer_norm(
                x,
                weight=self.post_ln_w,
                bias=self.post_ln_b,
                epsilon=1e-5,
            )

        return x


# ============================================================================
# Projector
# ============================================================================

class MultiModalProjectorTTNN:
    def __init__(self, weights, device):
        self.weight = ttnn.transpose(
            weights["linear.weight"], -2, -1
        )
        self.bias = weights.get("linear.bias")

    def forward(self, x):
        return ttnn.linear(x, self.weight, bias=self.bias)


# aliases
PatchEmbedding = PatchEmbeddingTTNN
SigLIPVisionTower = SigLIPVisionTowerTTNN
MultiModalProjector = MultiModalProjectorTTNN