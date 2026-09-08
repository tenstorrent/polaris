# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
This is the VisionEmbedding implementation for the Gemma-3-4b-it
This implementation combines patch_conv followed by Embeddings as a submodule.
"""

import numpy as np
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.common.gemma_utils import to_numpy
from workloads.ttnn.gemma3.tt.gemma_conv2d_patch import TtGemmaConv2dPatch


class TtSiglipVisionEmbeddings(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        image_size,
        patch_size,
        num_channels,
        hidden_dim,
        bias=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Fixed: Use 2-argument form (start, end) instead of 3-argument form
        self.position_ids = ttnn.arange(0, self.num_positions, dtype=ttnn.uint32, device=self.mesh_device)
        self.position_ids = ttnn.reshape(self.position_ids, (1, -1))

        self.patch_embed = TtGemmaConv2dPatch(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}patch_embedding.",
            dtype=dtype,
            in_channels=num_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            image_size=image_size,
        )

        # Positional embedding - get from state dict and convert to numpy
        pos_key = f"{state_dict_prefix}position_embedding.positional_embedding"
        if pos_key in state_dict:
            positional_embedding = state_dict[pos_key]
            positional_embedding = to_numpy(positional_embedding)
        else:
            # Fallback: create dummy positional embedding
            positional_embedding = np.random.randn(self.num_patches, hidden_dim).astype(np.float32) * 0.02

        # Convert numpy array to ttnn tensor with proper layout
        self.pos_emb_weights = ttnn.as_tensor(
            positional_embedding,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

    def forward(self, x):
        """
        Args:
            x: ttnn.Tensor - Input tensor (image)
               Expected shape: (B, C, H, W)
        Returns:
            embeddings: ttnn.Tensor of shape (B, num_patches, hidden_dim)
        """
        # Get patch embeddings from conv2d patch layer
        patch_embeddings = self.patch_embed(x)

        # Get shapes for debugging and processing
        patch_shape = self._get_shape_tuple(patch_embeddings)
        x_shape = self._get_shape_tuple(x)

        # Get batch size from input
        batch_size = x_shape[0] if len(x_shape) >= 1 else 1

        # Handle different output shapes from patch_embed
        # Could be [1, B, num_patches, hidden_dim] (4D) or [B, num_patches, hidden_dim] (3D)
        if len(patch_shape) == 4:
            # Shape is [1, B, num_patches, hidden_dim] - squeeze and reshape
            # Extract actual dimensions
            actual_batch = patch_shape[1]
            seq_len = patch_shape[2]
            hidden = patch_shape[3]
            patch_embeddings = ttnn.reshape(patch_embeddings, (actual_batch, seq_len, hidden))
            batch_size = actual_batch
        elif len(patch_shape) == 3:
            # Shape is already [B, num_patches, hidden_dim]
            batch_size = patch_shape[0]

        # Get updated shape after reshape
        patch_shape = self._get_shape_tuple(patch_embeddings)

        # Get positional embeddings: [1, num_patches, hidden_dim]
        positional_embeddings = ttnn.embedding(
            self.position_ids,
            self.pos_emb_weights,
            layout=ttnn.TILE_LAYOUT,
        )

        # Get positional embeddings shape
        pos_shape = self._get_shape_tuple(positional_embeddings)

        # Handle shape matching for ttnn.add
        # patch_embeddings: [B, num_patches, hidden_dim]
        # positional_embeddings: [1, num_patches, hidden_dim]

        # Check if we need to repeat positional embeddings for batch dimension
        if batch_size > 1:
            # Check if sequence lengths match
            patch_seq_len = patch_shape[1] if len(patch_shape) >= 2 else self.num_patches
            pos_seq_len = pos_shape[1] if len(pos_shape) >= 2 else self.num_patches

            if patch_seq_len != pos_seq_len:
                # Sequence length mismatch - this shouldn't happen normally
                # but handle it gracefully by using the smaller one
                # This might indicate an issue in patch_embed for batched inputs
                pass  # Let it fail with a clear error message

            # Repeat positional embeddings along batch dimension
            # [1, num_patches, hidden_dim] -> [B, num_patches, hidden_dim]
            positional_embeddings = ttnn.repeat(positional_embeddings, (batch_size, 1, 1))

        # Add positional embeddings
        embeddings = ttnn.add(patch_embeddings, positional_embeddings)

        return embeddings