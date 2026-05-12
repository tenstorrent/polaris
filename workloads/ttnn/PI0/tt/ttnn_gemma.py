# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Gemma transformer blocks - TTSim Implementation.
This module implements Gemma 2B style transformer layers using ttsim.front.ttnn:
    - RMSNorm (pre-normalization)
    - Multi-Query Attention (MQA) with num_kv_heads=1
    - GeGLU MLP (gated GELU activation)
    - Rotary Position Embeddings (RoPE)
Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1
NOTE: This is a TTSim port - shape tracking only, no numerical computation.
Hardware-specific optimizations (multicast, compute kernels) are removed.
"""
import math
from typing import Dict, Optional, Tuple
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.PI0.common.configs import GemmaConfig


# ============================================================================
# RMSNorm (TTSim)
# ============================================================================
def rms_norm_ttnn(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    RMSNorm implementation for TTSim.

    TTSim rms_norm API is currently incomplete/broken for some shapes,
    so we fall back to shape-preserving multiply.
    """

    try:
        return ttnn.rms_norm(x, weight)
    except Exception:
        return ttnn.multiply(x, weight)

# ============================================================================
# Rotary Position Embeddings (TTSim)
# ============================================================================
def precompute_freqs_cis_ttnn(
    head_dim: int,
    max_seq_len: int,
    device: TTNNDevice,
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Precompute cos and sin for rotary embeddings using TTSim operations.
    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        device: TTNNDevice
        base: Base for frequency computation
    Returns:
        Tuple of (cos, sin) each of shape (1, 1, max_seq_len, head_dim)
    """
    half_dim = head_dim // 2
    # TTSim arange only supports (length) or (start, end), not step
    # Create indices [0, 1, 2, ..., half_dim-1] then multiply by 2 to get [0, 2, 4, ...]
    indices_half = ttnn.arange(0, half_dim, device=device, dtype=ttnn.float32)
    indices_half = ttnn.to_layout(indices_half, ttnn.TILE_LAYOUT)
    indices = ttnn.multiply(indices_half, 2.0)  # [0, 2, 4, ..., head_dim-2]
    ttnn.deallocate(indices_half)
    # freqs = 1.0 / (base ** (indices / head_dim))
    # Use exp(-x * log(base)) instead of pow then reciprocal
    exponents = ttnn.multiply(indices, 1.0 / head_dim)
    ttnn.deallocate(indices)
    neg_exp = ttnn.multiply(exponents, -math.log(base))
    ttnn.deallocate(exponents)
    freqs = ttnn.exp(neg_exp)  # Shape: [half_dim]
    ttnn.deallocate(neg_exp)
    # Compute positions: [0, 1, 2, ..., max_seq_len-1]
    t = ttnn.arange(0, max_seq_len, device=device, dtype=ttnn.float32)
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
    # Outer product: t[i] * freqs[j] -> [max_seq_len, half_dim]
    t_col = ttnn.reshape(t, (max_seq_len, 1))
    ttnn.deallocate(t)
    freqs_row = ttnn.reshape(freqs, (1, half_dim))
    ttnn.deallocate(freqs)
    freqs_outer = ttnn.multiply(t_col, freqs_row)
    ttnn.deallocate(t_col)
    ttnn.deallocate(freqs_row)
    # Compute cos/sin
    cos_half = ttnn.cos(freqs_outer)
    sin_half = ttnn.sin(freqs_outer)
    ttnn.deallocate(freqs_outer)
    # Repeat for full head_dim
    cos_2d = ttnn.concat(cos_half, cos_half, axis=-1)
    sin_2d = ttnn.concat(sin_half, sin_half, axis=-1)
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)
    # Reshape to [1, 1, seq, head_dim]
    cos = ttnn.reshape(cos_2d, (1, 1, max_seq_len, head_dim))
    sin = ttnn.reshape(sin_2d, (1, 1, max_seq_len, head_dim))
    ttnn.deallocate(cos_2d)
    ttnn.deallocate(sin_2d)
    return cos, sin


# ============================================================================
# Multi-Query Attention (TTSim)
# ============================================================================
class GemmaAttentionTTNN:
    """
    TTSim-compatible Gemma Attention.

    No slice()
    No hardware configs
    Uses native TTNN APIs when available
    Falls back safely for TTSim
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device

        #
        # Weights
        #

        self.wqkv = weights.get("self_attn.wqkv")

        self.q_proj = weights.get("self_attn.q_proj.weight")
        self.k_proj = weights.get("self_attn.k_proj.weight")
        self.v_proj = weights.get("self_attn.v_proj.weight")

        self.o_proj = weights["self_attn.o_proj.weight"]

        #
        # Config
        #

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.width

        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.cos_meta = cos_meta
        self.sin_meta = sin_meta

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ):

        batch_size = hidden_states.shape[0] # type: ignore[index]
        seq_len = hidden_states.shape[1]  #type: ignore[index]

        #
        # Convert to 4D
        #

        if len(hidden_states.shape) == 3:  #type: ignore
            hidden_states = ttnn.reshape(
                hidden_states,
                (batch_size, 1, seq_len, self.hidden_size),
            )

        #
        # QKV projections
        #
        # IMPORTANT:
        # Since slice() does not exist in TTSim,
        # avoid fused-QKV fallback splitting.
        #

        q = None
        k = None
        v = None

        #
        # Try fused QKV + native head creation
        #

        if self.wqkv is not None:

            try:
                xqkv = ttnn.linear(hidden_states, self.wqkv)

                q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                    xqkv,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    transpose_k_heads=False,
                )

            except Exception:
                pass

        #
        # Fallback to separate projections
        #

        if q is None:

            q = ttnn.linear(hidden_states, self.q_proj)

            k = ttnn.linear(hidden_states, self.k_proj)

            v = ttnn.linear(hidden_states, self.v_proj)

            q = ttnn.reshape(
                q,
                (batch_size, self.num_heads, seq_len, self.head_dim),
            )

            k = ttnn.reshape(
                k,
                (batch_size, self.num_kv_heads, seq_len, self.head_dim),
            )

            v = ttnn.reshape(
                v,
                (batch_size, self.num_kv_heads, seq_len, self.head_dim),
            )

        #
        # RoPE
        #

        try:

            q_rope = ttnn.experimental.rotary_embedding( # type: ignore[attr-defined]
                q,
                cos,
                sin,
            )

            k_rope = ttnn.experimental.rotary_embedding( # type: ignore[attr-defined]
                k,
                cos,
                sin,
            )

        except Exception:

            #
            # TTSim fallback
            #

            q_rope = q
            k_rope = k

        #
        # KV cache
        #

        if past_key_value is not None:

            past_k, past_v = past_key_value

            try:

                k_rope = ttnn.concat(
                    [past_k, k_rope],
                    dim=2,
                )

                v = ttnn.concat(
                    [past_v, v],
                    dim=2,
                )

            except Exception:

                #
                # Older concat API
                #

                k_rope = ttnn.concat(
                    past_k,
                    k_rope,
                    axis=2,
                )

                v = ttnn.concat(
                    past_v,
                    v,
                    axis=2,
                )

        new_cache = (k_rope, v) if use_cache else None

        #
        # Attention
        #

        try:

            attn_output = ttnn.transformer.scaled_dot_product_attention( # type: ignore[attr-defined]
                q_rope,
                k_rope,
                v,
                attn_mask=attention_mask,
                is_causal=False,
                scale=self.scale,
            )

        except Exception:

            #
            # Manual attention fallback
            #

            if self.num_kv_heads < self.num_heads:

                repeat_factor = self.num_heads // self.num_kv_heads

                k_rope = ttnn.repeat(
                    k_rope,
                    (1, repeat_factor, 1, 1),
                )

                v = ttnn.repeat(
                    v,
                    (1, repeat_factor, 1, 1),
                )

            k_t = ttnn.permute(
                k_rope,
                (0, 1, 3, 2),
            )

            attn_weights = ttnn.matmul(
                q_rope,
                k_t,
            )

            attn_weights = ttnn.multiply(
                attn_weights,
                self.scale,
            )

            if attention_mask is not None:

                attn_weights = ttnn.add(
                    attn_weights,
                    attention_mask,
                )

            attn_weights = ttnn.softmax(
                attn_weights,
                dim=-1,
            )

            attn_output = ttnn.matmul(
                attn_weights,
                v,
            )

        #
        # Concat heads
        #

        try:

            attn_concat = ttnn.experimental.nlp_concat_heads(
                attn_output,
            )

        except Exception:

            attn_concat = ttnn.reshape(
                attn_output,
                (batch_size, 1, seq_len, self.hidden_size),
            )

        #
        # Output projection
        #

        output = ttnn.linear(
            attn_concat,
            self.o_proj,
        )

        #
        # Back to 3D
        #

        output = ttnn.reshape(
            output,
            (batch_size, seq_len, self.hidden_size),
        )

        return output, new_cache
    
# ============================================================================
# GeGLU MLP (TTSim)
# ============================================================================
class GemmaMLPTTNN:
    """
    Gemma MLP with GeGLU activation using TTSim.
    Simplified version without chunking or hardware optimizations.
    """
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        device: TTNNDevice,
    ):
        """
        Initialize MLP.
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            device: TTNNDevice
        """
        self.config = config
        self.device = device
        self.gate_proj = weights.get("mlp.gate_proj.weight")
        self.up_proj = weights.get("mlp.up_proj.weight")
        self.down_proj = weights.get("mlp.down_proj.weight")
        self.mlp_dim = config.mlp_dim
        self.hidden_size = config.width

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass (TTSim version).
        Args:
            x: Input tensor [batch, seq, hidden] or [batch, 1, seq, hidden]
        Returns:
            Output tensor of same shape
        """
        # Gate projection
        gate = ttnn.linear(x, self.gate_proj)
        # Up projection
        up = ttnn.linear(x, self.up_proj)
        # GELU activation on gate
        gate_activated = ttnn.gelu(gate)
        ttnn.deallocate(gate)
        # Element-wise multiply
        hidden_out = ttnn.multiply(gate_activated, up)
        ttnn.deallocate(gate_activated)
        ttnn.deallocate(up)
        # Down projection
        output = ttnn.linear(hidden_out, self.down_proj)
        ttnn.deallocate(hidden_out)
        return output


# ============================================================================
# Full Transformer Block (TTSim)
# ============================================================================
class GemmaBlockTTNN:
    """
    Complete Gemma transformer block using TTSim.
    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """
    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: TTNNDevice,
        cos: Optional[ttnn.Tensor] = None,
        sin: Optional[ttnn.Tensor] = None,
    ):
        """
        Initialize transformer block.
        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNNDevice
            cos: Precomputed cos for RoPE
            sin: Precomputed sin for RoPE
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.input_layernorm_weight = weights.get("input_layernorm.weight")
        self.post_attention_layernorm_weight = weights.get("post_attention_layernorm.weight")
        self.attention = GemmaAttentionTTNN(config, weights, layer_idx, device, cos, sin)
        self.mlp = GemmaMLPTTNN(config, weights, device)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass through transformer block.
        Args:
            hidden_states: Input tensor
            cos, sin: RoPE embeddings
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache
        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm (shape-preserving)
        # Add null check for weight
        if self.input_layernorm_weight is None:
            raise ValueError("input_layernorm_weight must not be None")
        normed = rms_norm_ttnn(
            hidden_states,
            self.input_layernorm_weight,
            self.config.rms_norm_eps,
        )
        # Attention with residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        hidden_states = ttnn.add(hidden_states, attn_output)
        ttnn.deallocate(attn_output)
        # Pre-MLP norm (shape-preserving)
        # Add null check for weight
        if self.post_attention_layernorm_weight is None:
            raise ValueError("post_attention_layernorm_weight must not be None")
        normed = rms_norm_ttnn(
            hidden_states,
            self.post_attention_layernorm_weight,
            self.config.rms_norm_eps,
        )
        # MLP with residual
        mlp_output = self.mlp.forward(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(hidden_states, mlp_output)
        ttnn.deallocate(mlp_output)
        return hidden_states, new_cache


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN