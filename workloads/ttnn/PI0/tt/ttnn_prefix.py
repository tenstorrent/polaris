# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Prefix Embedding module - TTSim Implementation
"""
import math
from typing import Callable, List, Optional, Tuple

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device as TTNNDevice

from workloads.ttnn.PI0.common.configs import PrefixConfig


class PrefixEmbeddingTTNN:
    """
    TTSim implementation of prefix embedding.
    """

    def __init__(
        self,
        config: PrefixConfig,
        device: TTNNDevice,
        embed_image_fn: Optional[Callable] = None,
        embed_language_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.device = device
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn

        self.prefix_att_masks = ttnn.zeros(
            (1, 544),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def embed_images(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
    ) -> Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")

        image_embs = []
        expanded_masks = []

        for img, mask in zip(images, img_masks):
            img_emb = self.embed_image_fn(img)
            image_embs.append(img_emb)

            shape = img_emb.shape
            if shape is None:
                raise ValueError("img_emb must have a valid shape")
            batch_size = int(shape[0])
            num_tokens = int(shape[1])

            # Create expanded mask using zeros and avoid shape issues
            mask_reshaped = ttnn.reshape(mask, [batch_size, 1])

            expanded_mask = ttnn.repeat(
                mask_reshaped,
                [1, num_tokens]
      )
            expanded_masks.append(expanded_mask)

        return image_embs, expanded_masks

    def embed_language(
        self,
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if self.embed_language_fn is None:
            raise RuntimeError("embed_language_fn not set")

        lang_emb = self.embed_language_fn(lang_tokens)

        shape = lang_emb.shape
        if shape is None:
            raise ValueError("lang_emb must have a valid shape")
        hidden_dim = shape[-1]
        scale = math.sqrt(hidden_dim)

        return ttnn.multiply(lang_emb, scale)

    def embed_prefix(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        embs = []
        pad_masks = []

        # Process images
        if images and self.embed_image_fn is not None:
            image_embs, img_pad_masks = self.embed_images(images, img_masks)
            for img_emb, img_pad_mask in zip(image_embs, img_pad_masks):
                embs.append(img_emb)
                pad_masks.append(img_pad_mask)

        # Process language
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens, lang_masks)
            embs.append(lang_emb)
            
            # Create 2D mask for language with same rank as image masks
            lang_shape = lang_emb.shape
            if lang_shape is None:
                raise ValueError("lang_emb must have a valid shape")
            batch_size = int(lang_shape[0])
            lang_seq_len = int(lang_shape[1])
            
            # Create 2D mask directly
            lang_mask_2d = ttnn.zeros(
                [batch_size, lang_seq_len],  # Use list instead of tuple
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pad_masks.append(lang_mask_2d)

        # Concatenate embeddings
        if len(embs) > 1:
            prefix_embs = ttnn.concat(*embs, dim=1)
        else:
            prefix_embs = embs[0]

        # Concatenate pad masks
        if len(pad_masks) > 1:
            prefix_pad_masks = ttnn.concat(*pad_masks, dim=1)
        else:
            prefix_pad_masks = pad_masks[0]

        prefix_att_masks = self.prefix_att_masks

        return prefix_embs, prefix_pad_masks, prefix_att_masks


# Default export
PrefixEmbedding = PrefixEmbeddingTTNN