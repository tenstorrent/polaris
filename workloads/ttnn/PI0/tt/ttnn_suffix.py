# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Suffix Embedding module - TTSim Implementation
Ported from ttnn_suffix.py to use ttsim.front.ttnn instead of ttnn.
No PyTorch dependency — all tensors are ttsim Tensors.
"""
from typing import Dict, Optional, Tuple
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.PI0.common.configs import SuffixConfig
from workloads.ttnn.PI0.tt.ttnn_common import create_sinusoidal_pos_embedding_ttnn, tensor_1d_to_2d_ttnn


class SuffixEmbeddingTTNN:
    """
    TTSim implementation of suffix embedding.
    Uses ttsim.front.ttnn operations. No PyTorch dependency.
    """
    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, ttnn.Tensor],
        device: TTNNDevice,
    ):
        """
        Initialize suffix embedding with ttsim weights.
        Args:
            config: Suffix configuration
            weights: Dictionary with ttsim Tensor weights
            device: TTNNDevice
        """
        self.config = config
        self.device = device
        self.weights = weights
        # Pre-compute attention mask pattern
        # Attention mask is constant: [1, 1, 0, ..., 0] for PI0 or [1, 0, ..., 0] for PI05
        att_mask_pattern = []
        if not config.pi05:
            att_mask_pattern.append(1)  # State token
        att_mask_pattern.append(1)  # First action token
        att_mask_pattern.extend([0] * (config.action_horizon - 1))  # Remaining action tokens
        suffix_len = len(att_mask_pattern)
        pad_len = ((suffix_len + 31) // 32) * 32
        # Create base zeros mask on device
        att_mask_ttnn = ttnn.zeros(
            (1, pad_len),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        # Fill leading positions with ones based on pattern
        num_ones = sum(att_mask_pattern)
        if num_ones > 0:
            ones_tensor = ttnn.ones(
                (1, num_ones),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            # Pad ones to full length
            ones_padded = ttnn.pad(
                ones_tensor,
                [0, 0, 0, pad_len - num_ones],
                value=0.0,
            )
            ttnn.deallocate(ones_tensor)
            ttnn.deallocate(att_mask_ttnn)
            att_mask_ttnn = ones_padded
        self._att_mask_pattern = att_mask_ttnn
        self._att_mask_suffix_len = suffix_len
        half_dim = config.expert_width // 2
        # Pre-compute index tensor for sinusoidal embedding
        self.indices = ttnn.arange(
            0, half_dim,
            device=device,
            dtype=ttnn.float32,
        )

    def embed_actions(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embed noisy actions using ttnn.linear.
        Args:
            noisy_actions: ttsim Tensor (batch_size, action_horizon, action_dim)
        Returns:
            ttsim Tensor (batch_size, action_horizon, expert_width)
        """
        return ttnn.linear(
            noisy_actions,
            self.weights["action_in_proj.weight"],
            bias=self.weights["action_in_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def embed_state(self, state: ttnn.Tensor) -> Optional[ttnn.Tensor]:
        """
        Embed robot state (PI0 only, not PI05).
        Args:
            state: ttsim Tensor (batch_size, state_dim)
        Returns:
            ttsim Tensor (batch_size, 1, expert_width) or None for PI05
        """
        if self.config.pi05:
            return None
        state_emb = ttnn.linear(
            state,
            self.weights["state_proj.weight"],
            bias=self.weights["state_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Add null check for shape
        shape = state_emb.shape
        if shape is None:
            raise ValueError("state_emb must have a valid shape")
        # Add sequence dimension: (batch, expert_width) -> (batch, 1, expert_width)
        return ttnn.reshape(state_emb, (shape[0], 1, shape[-1]))

    def embed_timestep(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """
        Create sinusoidal timestep embedding.
        Args:
            timestep: ttsim Tensor (batch_size,)
        Returns:
            ttsim Tensor (batch_size, expert_width)
        """
        return create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
            indices=self.indices,
        )

    def fuse_action_time(
        self,
        action_emb: ttnn.Tensor,
        time_emb: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Fuse action and time embeddings.
        Args:
            action_emb: ttsim Tensor (batch_size, action_horizon, expert_width)
            time_emb:   ttsim Tensor (batch_size, expert_width)
        Returns:
            Tuple of (fused_emb, adarms_cond)
        """
        if self.config.pi05:
            return action_emb, time_emb
        
        # Add null check for shape
        shape = action_emb.shape
        if shape is None:
            raise ValueError("action_emb must have a valid shape")
        batch_size = shape[0]
        action_horizon = shape[1]
        
        # Expand time embedding to match action sequence length
        time_expanded = ttnn.reshape(time_emb, (batch_size, 1, -1))
        time_expanded = ttnn.repeat(time_expanded, (1, action_horizon, 1))
        # Concatenate along feature dimension
        concat = ttnn.concat(action_emb, time_expanded, axis=-1)
        # MLP: Linear -> SiLU -> Linear
        x = ttnn.linear(
            concat,
            self.weights["action_time_mlp_in.weight"],
            bias=self.weights["action_time_mlp_in.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.silu(x)
        x = ttnn.linear(
            x,
            self.weights["action_time_mlp_out.weight"],
            bias=self.weights["action_time_mlp_out.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return x, None

    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
        state_emb: Optional[ttnn.Tensor] = None,
        time_emb: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Create suffix embeddings.
        Args:
            state: ttsim Tensor (batch_size, state_dim) or None
            noisy_actions: ttsim Tensor (batch_size, action_horizon, action_dim)
            timestep: ttsim Tensor (batch_size,). Unused when time_emb is provided.
            state_emb: Optional pre-computed state embedding (batch_size, 1, expert_width).
            time_emb: Optional pre-computed timestep embedding (batch_size, expert_width).
        Returns:
            Tuple of (suffix_embs, pad_masks, att_masks, adarms_cond)
            - suffix_embs:  (batch_size, suffix_len, expert_width)
            - pad_masks:    None (no padding)
            - att_masks:    (batch_size, suffix_len)
            - adarms_cond:  Optional conditioning for adaptive RMSNorm
        """
        # Add null check for noisy_actions.shape
        na_shape = noisy_actions.shape
        if na_shape is None:
            raise ValueError("noisy_actions must have a valid shape")
        batch_size = na_shape[0]
        
        embs = []
        # Embed state (PI0 only)
        if not self.config.pi05:
            if state_emb is None:
                state_emb = self.embed_state(state)
            if state_emb is not None:
                embs.append(state_emb)
        # Embed timestep (skip if pre-computed)
        if time_emb is None:
            time_emb = self.embed_timestep(timestep)
        # Embed actions and fuse with timestep
        action_emb = self.embed_actions(noisy_actions)
        action_time_emb, adarms_cond = self.fuse_action_time(action_emb, time_emb)
        embs.append(action_time_emb)
        # Concatenate along sequence dimension
        if len(embs) > 1:
            suffix_embs = ttnn.concat(*embs, axis=1)
        else:
            suffix_embs = embs[0]
        
        # Add null check for shape
        shape = suffix_embs.shape
        if shape is None:
            raise ValueError("suffix_embs must have a valid shape")
        suffix_len = shape[1]
        
        suffix_pad_masks = None  # No padding
        # Create attention masks
        suffix_att_masks = ttnn.zeros(
            (batch_size, suffix_len),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond

    def project_output(self, expert_output: ttnn.Tensor) -> ttnn.Tensor:
        """
        Project expert output back to action dimension.
        Args:
            expert_output: ttsim Tensor (batch_size, action_horizon, expert_width)
        Returns:
            ttsim Tensor (batch_size, action_horizon, action_dim)
        """
        return ttnn.linear(
            expert_output,
            self.weights["action_out_proj.weight"],
            bias=self.weights["action_out_proj.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


def convert_suffix_weights_to_ttnn(
    ttsim_weights: Dict[str, ttnn.Tensor],
    device: TTNNDevice,
    dtype: Optional[ttnn.DataType] = None,
) -> Dict[str, ttnn.Tensor]:
    """
    Convert ttsim suffix weights to the correct layout/dtype.
    In the original TT Metal implementation this accepted PyTorch tensors
    and transferred them to device. In the ttsim context the inputs are
    already ttsim Tensors; this function applies transpose and layout
    adjustments using ttsim ops.
    Args:
        ttsim_weights: Dictionary of ttsim Tensors
        device: TTNNDevice
        dtype: Override dtype (default: bfloat8_b for weights, bfloat16 for bias)
    Returns:
        Dictionary of correctly formatted ttsim Tensors
    """
    ttnn_weights = {}
    for key, value in ttsim_weights.items():
        if "bias" in key:
            # Bias: reshape 1D [out] -> [1, out]
            ttnn_weights[key] = tensor_1d_to_2d_ttnn(value, device, dtype=ttnn.bfloat16)
        else:
            # Weight: transpose [out, in] -> [in, out] for TTNN matmul convention
            transposed = ttnn.permute(value, (1, 0))
            ttnn_weights[key] = ttnn.to_layout(transposed, ttnn.TILE_LAYOUT)
    return ttnn_weights


# Default export alias
SuffixEmbedding = SuffixEmbeddingTTNN