# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PaliGemma backbone wrapper - TTSim Implementation
This module combines vision, language, and action expert components:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)
The dual-expert architecture shares attention layers:
    - VLM and Expert compute separate Q, K, V
    - K, V are concatenated for shared attention
    - Outputs are split and processed through separate MLPs
NOTE: This is a TTSim port - shape tracking only, no numerical computation.
Hardware-specific optimizations are removed.
"""
from typing import Any, Dict, List, Optional, Tuple
import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice
from workloads.ttnn.PI0.common.configs import PaliGemmaConfig

from workloads.ttnn.PI0.tt.ttnn_common import tensor_1d_to_2d_ttnn
from workloads.ttnn.PI0.tt.ttnn_gemma import (
    GemmaBlockTTNN,
    rms_norm_ttnn,
    precompute_freqs_cis_ttnn,
)
from workloads.ttnn.PI0.tt.ttnn_siglip import (
    SigLIPVisionTowerTTNN,
    MultiModalProjectorTTNN,
)


class PaliGemmaBackboneTTNN:
    """
    PaliGemma backbone using TTSim operations.
    """
    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, ttnn.Tensor]],
        device: TTNNDevice,
    ):
        """
        Initialize PaliGemma backbone with TTSim.
        Args:
            config: PaliGemma configuration
            weights: Categorized weight tensors
            device: TTNNDevice
        """
        self.config = config
        self.device = device
        # VLM embedding weights
        self.vlm_embed_tokens = weights["vlm_language"].get("model.embed_tokens.weight")
        if self.vlm_embed_tokens is None:
            self.vlm_embed_tokens = weights["vlm_language"].get("lm_head.weight")
        # Final norm weights
        self.vlm_norm = weights["vlm_language"].get("model.norm.weight")
        self.expert_norm = weights["action_expert"].get("model.norm.weight")
        # Initialize vision tower
        self.vision_tower = SigLIPVisionTowerTTNN(
            config.siglip_config,
            weights["vlm_vision"],
            device,
        )
        # Initialize projector
        self.mm_projector = MultiModalProjectorTTNN(weights["vlm_projector"], device)
        # Precompute RoPE embeddings for VLM
        self.cos, self.sin = precompute_freqs_cis_ttnn(
            config.vlm_config.head_dim,
            config.max_seq_len,
            device,
        )
        # Precompute RoPE embeddings for Expert
        self.expert_cos, self.expert_sin = precompute_freqs_cis_ttnn(
            config.expert_config.head_dim,
            config.max_seq_len,
            device,
        )
        # Initialize VLM transformer blocks
        self.vlm_blocks: List[GemmaBlockTTNN] = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_block_weights(weights["vlm_language"], i)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    config.vlm_config,
                    block_weights,
                    i,
                    device,
                    self.cos,
                    self.sin,
                )
            )
        # Initialize Expert transformer blocks
        self.expert_blocks: List[GemmaBlockTTNN] = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_block_weights(weights["action_expert"], i)
            self.expert_blocks.append(
                GemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    device,
                    self.expert_cos,
                    self.expert_sin,
                )
            )

    def _get_block_weights(
        self,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:

        prefix = f"model.layers.{layer_idx}."
        block_weights: Dict[str, ttnn.Tensor] = {}

        # Create fused QKV if available
        q = weights.get(f"{prefix}self_attn.q_proj.weight")
        k = weights.get(f"{prefix}self_attn.k_proj.weight")
        v = weights.get(f"{prefix}self_attn.v_proj.weight")

        if q is not None and k is not None and v is not None:
            try:
                block_weights["self_attn.wqkv"] = ttnn.concat(
                    [q, k, v],
                    dim=-1,
                )
            except Exception:
                pass

        for key, value in weights.items():

            if not key.startswith(prefix):
                continue

            if value is None:
                continue # type: ignore[unreachable]

            new_key = key[len(prefix):] 
            # Skip individual QKV if fused exists
            if new_key in [
                "self_attn.q_proj.weight",

                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ]:
                continue

            # Pre-add Gemma RMSNorm +1 offset
            if (
                "layernorm.weight" in new_key
                or "norm.weight" in new_key
           ):
                try:
                    value = ttnn.add(
                        value,
                        1.0,
                    )
                except Exception:
                    pass
                pass

            block_weights[new_key] = value

        return block_weights
    
    def embed_image(self, pixel_values: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embed images through vision tower and projector.
        Args:
            pixel_values: TTNN tensor (batch_size, channels, height, width)
        Returns:
            TTNN tensor (batch_size, num_patches, vlm_width)
        """
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)

    def embed_language_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embed language tokens using TTSim embedding op.

        Args:
            token_ids: TTNN tensor of token IDs [batch, seq]

        Returns:
            TTNN tensor [batch, seq, hidden]
        """
        if self.vlm_embed_tokens is None:
            raise ValueError("vlm_embed_tokens is None")

        return ttnn.embedding(token_ids, self.vlm_embed_tokens)

    def forward_vlm(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through VLM backbone.
        Args:
            hidden_states: Prefix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from previous forward
            use_cache: Whether to return updated cache
        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = [] if use_cache else None
        
        for i, block in enumerate(self.vlm_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                self.cos,
                self.sin,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache and new_kv is not None and new_cache is not None:
                new_cache.append(new_kv)
        
        # Final norm
        if self.vlm_norm is not None:
            hidden_states = rms_norm_ttnn(
                hidden_states,
                self.vlm_norm,
                self.config.vlm_config.rms_norm_eps,
            )
        return hidden_states, new_cache

    def forward_expert(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through action expert.
        Args:
            hidden_states: Suffix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from VLM prefix (for cross-attention)
            use_cache: Whether to return updated cache
        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = [] if use_cache else None
        
        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                self.expert_cos,
                self.expert_sin,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache and new_kv is not None and new_cache is not None:
                new_cache.append(new_kv)
        
        # Final norm
        if self.expert_norm is not None:
            hidden_states = rms_norm_ttnn(
                hidden_states,
                self.expert_norm,
                self.config.expert_config.rms_norm_eps,
            )
        return hidden_states, new_cache

    def forward_shared_attention(
        self,
        prefix_embs: ttnn.Tensor,
        suffix_embs: ttnn.Tensor,
        prefix_mask: Optional[ttnn.Tensor] = None,
        suffix_mask: Optional[ttnn.Tensor] = None,
        prefix_position_ids: Optional[ttnn.Tensor] = None,
        suffix_position_ids: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass with shared attention between VLM and Expert.
        Args:
            prefix_embs: VLM prefix embeddings (TTNN tensor)
            suffix_embs: Expert suffix embeddings (TTNN tensor)
            prefix_mask: Prefix attention mask
            suffix_mask: Suffix attention mask
            prefix_position_ids: Prefix positions
            suffix_position_ids: Suffix positions
        Returns:
            Tuple of (vlm_output, expert_output)
        """
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            prefix_position_ids,
            use_cache=True,
        )
        # Process suffix through expert
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            suffix_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        return vlm_output, expert_output


# Default export
PaliGemmaBackbone = PaliGemmaBackboneTTNN