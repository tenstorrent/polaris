#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSim conversion of position_encoding.py
  PositionEmbeddingSineTTSim
  PositionEmbeddingLearnedTTSim
  build_position_encoding_ttsim()

DESIGN PRINCIPLES:
  - Use numpy operations for simplicity (position encoding is not performance-critical)
  - Return SimTensor objects for compatibility with TTSim pipeline
  - Support both shape inference and numerical computation modes
"""

import os, sys, math
import numpy as np
from typing import Optional

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

# Import ttsim operations
from ttsim.ops.tensor import SimTensor, require_shape_list
import ttsim.front.functional.sim_nn as SimNN
from workloads.Deformable_DETR.util.misc_ttsim import NestedTensor


# ──────────────────────────────────────────────────────────────────────────────
# PositionEmbeddingSineTTSim
# ──────────────────────────────────────────────────────────────────────────────
class PositionEmbeddingSine(SimNN.Module):
    """
    TTSim implementation of PositionEmbeddingSine.

    Uses ttsim operations for shape inference and numerical computation.
    Mirrors PyTorch forward() logic using ttsim operations.
    """

    def __init__(
        self,
        name: str,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.name = name
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi
        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor):
        """
        Forward pass using numpy operations for simplicity.

        Args:
            tensor_list: NestedTensor with tensors and mask

        Returns:
            Position embeddings [B, 2*num_pos_feats, H, W]
        """
        x = tensor_list.tensors
        mask = tensor_list.mask

        # Extract shape information
        if isinstance(x, SimTensor):
            B, C, H, W = require_shape_list(
                x.shape, "SimTensor in NestedTensor must have a known shape"
            )
        else:
            _shape = x.shape if hasattr(x, "shape") else (2, 3, 224, 224)
            B, C, H, W = _shape[0], _shape[1], _shape[2], _shape[3]

        # mask is Optional[np.ndarray] — keep as-is so shape-inference (None) path is reachable
        mask_data: Optional[np.ndarray] = mask

        # Compute position embeddings using numpy
        if mask_data is not None:
            # not_mask = ~mask (invert)
            not_mask = (
                ~mask_data
                if mask_data.dtype == bool
                else (1.0 - mask_data.astype(np.float32))
            )

            not_mask = not_mask.astype(np.float32)

            # Cumulative sum to get positions (keep float32 to match PyTorch precision)
            y_embed = np.cumsum(not_mask, axis=1).astype(np.float32)
            x_embed = np.cumsum(not_mask, axis=2).astype(np.float32)

            # Normalize if requested
            if self.normalize:
                eps = np.float32(1e-6)
                scale = np.float32(self.scale)
                y_embed = ((y_embed - np.float32(0.5)) / (y_embed[:, -1:, :] + eps) * scale).astype(np.float32)
                x_embed = ((x_embed - np.float32(0.5)) / (x_embed[:, :, -1:] + eps) * scale).astype(np.float32)

            # Compute position encoding
            dim_t_exp = np.float32(2) * (np.arange(self.num_pos_feats, dtype=np.float32) // np.float32(2)) / np.float32(self.num_pos_feats)
            dim_t = np.array([np.float32(math.pow(self.temperature, float(e))) for e in dim_t_exp], dtype=np.float32)

            # Expand dimensions: [B,H,W,1] / [D] -> [B,H,W,D]
            pos_x = (x_embed[:, :, :, None] / dim_t).astype(np.float32)
            pos_y = (y_embed[:, :, :, None] / dim_t).astype(np.float32)

            # Apply sin to even indices, cos to odd indices
            stacked_x = np.stack(
                (np.sin(pos_x[:, :, :, 0::2]), np.cos(pos_x[:, :, :, 1::2])), axis=4
            )
            pos_x = stacked_x.reshape(stacked_x.shape[0], stacked_x.shape[1], stacked_x.shape[2], -1)
            stacked_y = np.stack(
                (np.sin(pos_y[:, :, :, 0::2]), np.cos(pos_y[:, :, :, 1::2])), axis=4
            )
            pos_y = stacked_y.reshape(stacked_y.shape[0], stacked_y.shape[1], stacked_y.shape[2], -1)

            # Concatenate and permute to [B, 2*D, H, W]
            pos = np.concatenate((pos_y, pos_x), axis=3).transpose(0, 3, 1, 2)

            # Return as SimTensor
            result = SimTensor(
                {
                    "name": f"{self.name}.pos_output",
                    "shape": list(pos.shape),
                    "data": pos.astype(np.float32),
                    "dtype": np.dtype(np.float32),
                }
            )
        else:
            # Shape inference only
            result = SimTensor(
                {
                    "name": f"{self.name}.pos_output",
                    "shape": [B, 2 * self.num_pos_feats, H, W],
                    "data": None,
                    "dtype": np.dtype(np.float32),
                }
            )

        self._tensors[result.name] = result
        return result

        # cos_op_y = F.Cos(f'{self.name}.cos_y')
        # pos_y_cos = cos_op_y(pos_y_odd)

        # pos_y_stacked = stack([pos_y_sin, pos_y_cos], dim=4)  # [B, H, W, D/2, 2]
        # pos_y_flat = pos_y_stacked.flatten(3)  # [B, H, W, D]

        # # Concatenate and permute
        # pos = cat([pos_y_flat, pos_x_flat], dim=3)  # [B, H, W, 2*D]
        # pos = pos.permute([0, 3, 1, 2])  # [B, 2*D, H, W]

        # return pos


# ──────────────────────────────────────────────────────────────────────────────
# PositionEmbeddingLearnedTTSim
# ──────────────────────────────────────────────────────────────────────────────
class PositionEmbeddingLearned(SimNN.Module):
    """
    TTSim implementation of PositionEmbeddingLearned.

    Mirrors PyTorch: two Embedding tables of shape [50, num_pos_feats]
    (row_embed and col_embed).  Weights are stored as numpy arrays so
    they can be synced from the PyTorch model for numerical validation.
    """

    def __init__(self, name: str, num_pos_feats: int = 256):
        super().__init__()
        self.name = name
        self.num_pos_feats = num_pos_feats

        self.row_embed_weight = np.random.uniform(0, 1, (50, num_pos_feats)).astype(
            np.float32
        )
        self.col_embed_weight = np.random.uniform(0, 1, (50, num_pos_feats)).astype(
            np.float32
        )

        super().link_op2module()

    def __call__(self, tensor_list: NestedTensor):
        """
        Forward pass — mirrors PyTorch PositionEmbeddingLearned.forward().

        Args:
            tensor_list: NestedTensor with tensors

        Returns:
            Position embeddings [B, 2*num_pos_feats, H, W]
        """
        x = tensor_list.tensors

        # Extract shape
        if isinstance(x, SimTensor):
            B, C, H, W = require_shape_list(
                x.shape, "SimTensor in NestedTensor must have a known shape"
            )
            x_data = x.data
        else:
            B, C, H, W = x.shape if hasattr(x, "shape") else (2, 3, 224, 224)
            x_data = x

        if x_data is not None:
            # Index into stored embedding tables (same as nn.Embedding lookup)
            # col_embed(arange(W)) → [W, D]
            x_emb = self.col_embed_weight[:W]  # [W, D]
            # row_embed(arange(H)) → [H, D]
            y_emb = self.row_embed_weight[:H]  # [H, D]

            # Broadcast: x_emb → [H, W, D],  y_emb → [H, W, D]
            # Matches PyTorch:
            #   x_emb.unsqueeze(0).repeat(h, 1, 1)  → [H, W, D]
            #   y_emb.unsqueeze(1).repeat(1, w, 1)  → [H, W, D]
            x_emb_grid = np.tile(x_emb[np.newaxis, :, :], (H, 1, 1))  # [H, W, D]
            y_emb_grid = np.tile(y_emb[:, np.newaxis, :], (1, W, 1))  # [H, W, D]

            # cat dim=-1  →  [H, W, 2D]
            pos = np.concatenate([x_emb_grid, y_emb_grid], axis=2)

            # permute(2,0,1) → [2D, H, W]  then broadcast to [B, 2D, H, W]
            pos = pos.transpose(2, 0, 1)
            pos = np.tile(pos[np.newaxis, :, :, :], (B, 1, 1, 1))

            result = SimTensor(
                {
                    "name": f"{self.name}.pos_output",
                    "shape": list(pos.shape),
                    "data": pos.astype(np.float32),
                    "dtype": np.dtype(np.float32),
                }
            )
        else:
            # Shape inference only
            result = SimTensor(
                {
                    "name": f"{self.name}.pos_output",
                    "shape": [B, 2 * self.num_pos_feats, H, W],
                    "data": None,
                    "dtype": np.dtype(np.float32),
                }
            )

        self._tensors[result.name] = result
        return result


# ──────────────────────────────────────────────────────────────────────────────
# factory  – mirrors build_position_encoding(args)
# ──────────────────────────────────────────────────────────────────────────────
def build_position_encoding(args):
    """Build position encoding module (ttsim version)"""
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        return PositionEmbeddingSine("position_embedding", N_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        return PositionEmbeddingLearned("position_embedding", N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")
