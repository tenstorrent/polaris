# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Vision Model (SigLIP-based Vision Encoder)
"""
import numpy as np
import ttsim.front.ttnn as ttnn
from loguru import logger

from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings
from workloads.ttnn.gemma3.tt.gemma_image_transformer import TtGemmaImageTransformer
from workloads.ttnn.tt_transformers.llama_layernorm import TtLayerNorm


class TtSiglipGemmaVisionModel(LightweightModule):
    """
    Complete SigLIP-based vision model for Gemma3.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        dtype,
        configuration,
        weight_cache_path=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.configuration = configuration

        # Model dimensions from config
        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.in_channels = configuration.vision_in_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Vision embeddings
        self.embeddings = TtSiglipVisionEmbeddings(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}embeddings.",
            dtype=dtype,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.in_channels,
            hidden_dim=self.width,
            bias=True,
        )

        # Vision transformer encoder
        self.encoder = TtGemmaImageTransformer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}encoder.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            layers=self.layers,
            block_key="layers",
        )

        # Post layer norm
        self.ln_post = TtLayerNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_post.",
            weight_cache_path=weight_cache_path if weight_cache_path else configuration.weight_cache_path(dtype),
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

    def _get_shape_tuple(self, tensor):
        """Get shape as tuple from various tensor types."""
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            if hasattr(shape, '__iter__'):
                return tuple(shape)
            return shape
        if hasattr(tensor, 'get_shape'):
            return tuple(tensor.get_shape())
        return ()

    def forward(self, images):
        """
        Forward pass for vision model.

        Args:
            images: Input images [B, C, H, W]

        Returns:
            Vision embeddings (shape depends on internal processing)
        """
        # Get batch size from input
        shape = self._get_shape_tuple(images)
        bsz = shape[0] if len(shape) >= 1 else 1

        # Get embeddings
        x = self.embeddings(images)

        # Create attention mask
        embed_shape = self._get_shape_tuple(x)
        if len(embed_shape) == 4:
            seq_len = embed_shape[2]
        elif len(embed_shape) == 3:
            seq_len = embed_shape[1]
        else:
            seq_len = self.num_patches

        mask_np = np.zeros((bsz, 1, seq_len, seq_len), dtype=np.float32)
        tt_mask = ttnn.Tensor(
            mask_np,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x = self.encoder(x, mask=tt_mask)
        ttnn.deallocate(tt_mask)
        x = self.ln_post(x)
        
        return x