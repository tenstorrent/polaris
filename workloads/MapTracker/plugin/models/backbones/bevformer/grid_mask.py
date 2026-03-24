#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GridMask TTSim Module
Converted from PyTorch: projects/mmdet3d_plugin/models/utils/grid_mask.py

GridMask is a data augmentation technique that randomly drops out regions of the input
in a grid pattern. This helps the model become more robust by preventing overfitting
to specific spatial patterns.
"""

# -------------------------------PyTorch--------------------------------

# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# from mmcv.runner import force_fp32, auto_fp16

# class Grid(object):
#     def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
#         self.use_h = use_h
#         self.use_w = use_w
#         self.rotate = rotate
#         self.offset = offset
#         self.ratio = ratio
#         self.mode=mode
#         self.st_prob = prob
#         self.prob = prob

#     def set_prob(self, epoch, max_epoch):
#         self.prob = self.st_prob * epoch / max_epoch

#     def __call__(self, img, label):
#         if np.random.rand() > self.prob:
#             return img, label
#         h = img.size(1)
#         w = img.size(2)
#         self.d1 = 2
#         self.d2 = min(h, w)
#         hh = int(1.5*h)
#         ww = int(1.5*w)
#         d = np.random.randint(self.d1, self.d2)
#         if self.ratio == 1:
#             self.l = np.random.randint(1, d)
#         else:
#             self.l = min(max(int(d*self.ratio+0.5),1),d-1)
#         mask = np.ones((hh, ww), np.float32)
#         st_h = np.random.randint(d)
#         st_w = np.random.randint(d)
#         if self.use_h:
#             for i in range(hh//d):
#                 s = d*i + st_h
#                 t = min(s+self.l, hh)
#                 mask[s:t,:] *= 0
#         if self.use_w:
#             for i in range(ww//d):
#                 s = d*i + st_w
#                 t = min(s+self.l, ww)
#                 mask[:,s:t] *= 0
#
#         r = np.random.randint(self.rotate)
#         mask = Image.fromarray(np.uint8(mask))
#         mask = mask.rotate(r)
#         mask = np.asarray(mask)
#         mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
#
#         mask = torch.from_numpy(mask).float()
#         if self.mode == 1:
#             mask = 1-mask
#
#         mask = mask.expand_as(img)
#         if self.offset:
#             offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float()
#             offset = (1 - mask) * offset
#             img = img * mask + offset
#         else:
#             img = img * mask
#
#         return img, label


# class GridMask(nn.Module):
#     def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
#         super(GridMask, self).__init__()
#         self.use_h = use_h
#         self.use_w = use_w
#         self.rotate = rotate
#         self.offset = offset
#         self.ratio = ratio
#         self.mode = mode
#         self.st_prob = prob
#         self.prob = prob
#         self.fp16_enable = False
#
#     def set_prob(self, epoch, max_epoch):
#         self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5
#
#     def set_ratio_and_prob(self, ratio, prob):
#         self.prob = prob
#         self.ratio = ratio
#
#     @auto_fp16()
#     def forward(self, x):
#         if np.random.rand() > self.prob or not self.training:
#             return x
#         n,c,h,w = x.size()
#         x = x.view(-1,h,w)
#         hh = int(1.5*h)
#         ww = int(1.5*w)
#         d = np.random.randint(2, h)
#         self.l = min(max(int(d*self.ratio+0.5),1),d-1)
#         mask = np.ones((hh, ww), np.float32)
#         st_h = np.random.randint(d)
#         st_w = np.random.randint(d)
#         if self.use_h:
#             for i in range(hh//d):
#                 s = d*i + st_h
#                 t = min(s+self.l, hh)
#                 mask[s:t,:] *= 0
#         if self.use_w:
#             for i in range(ww//d):
#                 s = d*i + st_w
#                 t = min(s+self.l, ww)
#                 mask[:,s:t] *= 0
#
#         r = np.random.randint(self.rotate)
#         mask = Image.fromarray(np.uint8(mask))
#         mask = mask.rotate(r)
#         mask = np.asarray(mask)
#         mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]
#
#         mask = torch.from_numpy(mask).to(x.dtype).cuda()
#
#         if self.mode == 1:
#             mask = 1-mask
#
#         mask = mask.expand_as(x)
#         if self.offset:
#             offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
#             x = x * mask + offset * (1 - mask)
#         else:
#             x = x * mask
#
#         return x.view(n,c,h,w)

# -------------------------------TTSIM-----------------------------------

import numpy as np
from PIL import Image
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.op as F


class GridMask(SimNN.Module):
    """
    GridMask data augmentation module for TTSim.

    Applies a grid-based masking pattern to input images, dropping out rectangular regions
    in a regular grid pattern. The grid can be applied horizontally, vertically, or both,
    with optional rotation.

    Original PyTorch implementation: mmdet3d_plugin/models/utils/grid_mask.py

    Args:
        name: Module name for TTSim graph
        use_h: If True, apply grid mask along height dimension
        use_w: If True, apply grid mask along width dimension
        rotate: Maximum rotation angle in degrees (0-360)
        offset: If True, add random offset to masked regions instead of zeros
        ratio: Ratio of masked region size to grid spacing (0-1)
        mode: 0 = keep non-masked regions, 1 = invert (keep masked regions)
        prob: Probability of applying the mask (0-1)
        training: Whether module is in training mode

    Attributes:
        use_h: Apply mask along height
        use_w: Apply mask along width
        rotate: Max rotation angle
        offset: Use random offset
        ratio: Mask to spacing ratio
        mode: Masking mode
        st_prob: Starting probability
        prob: Current probability
        training: Training flag

    Shape:
        - Input: [N, C, H, W] - Batch of images
        - Output: [N, C, H, W] - Masked images (same shape as input)

    Examples:
        >>> grid_mask = GridMask(
        ...     name='grid_mask',
        ...     use_h=True,
        ...     use_w=True,
        ...     rotate=1,
        ...     ratio=0.5,
        ...     prob=0.7,
        ...     training=True
        ... )
        >>> x = F._from_shape('input', [2, 3, 224, 224])
        >>> masked = grid_mask(x, seed=42)  # Deterministic with seed
        >>> print(masked.shape)  # [2, 3, 224, 224]
    """

    def __init__(
        self,
        name: str,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.0,
        training: bool = True,
    ):
        """Initialize GridMask module."""
        super().__init__()

        self.name = name
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob  # Starting probability
        self.prob = prob  # Current probability
        self.training = training

        # Link operations to module
        super().link_op2module()

    def set_prob(self, epoch: int, max_epoch: int):
        """
        Update probability based on training progress.

        Args:
            epoch: Current epoch number
            max_epoch: Total number of epochs
        """
        self.prob = self.st_prob * epoch / max_epoch

    def set_training(self, mode: bool):
        """Set training mode."""
        self.training = mode

    def _generate_mask(self, h: int, w: int, seed: int | None = None) -> np.ndarray:
        """
        Generate grid mask pattern using NumPy.

        Args:
            h: Image height
            w: Image width
            seed: Random seed for deterministic behavior (used in validation)

        Returns:
            mask: Binary mask array of shape [h, w]
        """
        # Set random seed if provided (for validation/testing)
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Skip masking with probability (1 - prob)
        if rng.rand() > self.prob or not self.training:
            return np.ones((h, w), dtype=np.float32)

        # Create larger canvas for rotation
        hh = int(1.5 * h)
        ww = int(1.5 * w)

        # Random grid spacing between 2 and min(h, w)
        d = rng.randint(2, min(h, w))

        # Calculate mask line width based on ratio
        if self.ratio == 1:
            l = rng.randint(1, d)
        else:
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)

        # Initialize mask (all ones = keep all pixels)
        mask: np.ndarray = np.ones((hh, ww), dtype=np.float32)

        # Random starting positions
        st_h = rng.randint(d)
        st_w = rng.randint(d)

        # Apply horizontal grid lines
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + l, hh)
                mask[s:t, :] = 0

        # Apply vertical grid lines
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + l, ww)
                mask[:, s:t] = 0

        # Apply random rotation
        if self.rotate > 0:
            r = rng.randint(self.rotate)
            mask_img = Image.fromarray(np.uint8(mask * 255))
            mask_img = mask_img.rotate(r)
            mask = np.asarray(mask_img).astype(np.float32) / 255.0

        # Crop to original size
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        # Invert mask if mode == 1
        if self.mode == 1:
            mask = 1 - mask

        return mask

    def __call__(self, x, seed: int | None = None):
        """
        Apply grid mask to input tensor.

        Args:
            x: Input tensor of shape [N, C, H, W]
            seed: Random seed for deterministic masking (used in validation)

        Returns:
            Masked output tensor of shape [N, C, H, W]
        """
        # Get input shape
        N, C, H, W = x.shape

        # Generate mask for this batch
        mask_np = self._generate_mask(H, W, seed=seed)

        # Create TTSim tensor from mask
        # Mask shape: [H, W] -> need to broadcast to [N, C, H, W]
        mask_tensor = F._from_data(
            self.name + ".mask", mask_np.astype(np.float32), is_const=True
        )

        # Add batch dimension: [H, W] -> [1, H, W]
        mask_unsqueezed = F.Unsqueeze(self.name + ".mask_unsq")(
            mask_tensor,
            F._from_data(
                self.name + ".unsq_axis", np.array([0], dtype=np.int64), is_const=True
            ),
        )

        # Reshape input to [N*C, H, W] for easier processing
        x_reshaped = F.Reshape(self.name + ".reshape_in")(
            x,
            F._from_data(
                self.name + ".reshape_shape_in",
                np.array([N * C, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        # Tile mask to [N*C, H, W]
        mask_expanded = F.Tile(self.name + ".mask_tile")(
            mask_unsqueezed,
            F._from_data(
                self.name + ".tile_reps",
                np.array([N * C, 1, 1], dtype=np.int64),
                is_const=True,
            ),
        )

        # Apply mask: x * mask
        if self.offset and seed is not None:
            # Add random offset to masked regions
            # offset = 2 * (rand(H, W) - 0.5) in range [-1, 1]
            offset_np = 2 * (np.random.RandomState(seed + 1).rand(H, W) - 0.5)
            offset_tensor = F._from_data(
                self.name + ".offset", offset_np.astype(np.float32), is_const=True
            )

            # Unsqueeze offset: [H, W] -> [1, H, W]
            offset_unsqueezed = F.Unsqueeze(self.name + ".offset_unsq")(
                offset_tensor,
                F._from_data(
                    self.name + ".offset_unsq_axis",
                    np.array([0], dtype=np.int64),
                    is_const=True,
                ),
            )

            # Tile offset to [N*C, H, W]
            offset_expanded = F.Tile(self.name + ".offset_tile")(
                offset_unsqueezed,
                F._from_data(
                    self.name + ".offset_tile_reps",
                    np.array([N * C, 1, 1], dtype=np.int64),
                    is_const=True,
                ),
            )

            # Compute: x * mask + offset * (1 - mask)
            masked_x = F.Mul(self.name + ".mul_mask")(x_reshaped, mask_expanded)

            # (1 - mask)
            one = F._from_data(
                self.name + ".one", np.array(1.0, dtype=np.float32), is_const=True
            )
            inv_mask = F.Sub(self.name + ".inv_mask")(one, mask_expanded)

            # offset * (1 - mask)
            offset_part = F.Mul(self.name + ".mul_offset")(offset_expanded, inv_mask)

            # x * mask + offset * (1 - mask)
            output_reshaped = F.Add(self.name + ".add_offset")(masked_x, offset_part)
        else:
            # Simple masking: x * mask
            output_reshaped = F.Mul(self.name + ".mul")(x_reshaped, mask_expanded)

        # Reshape back to [N, C, H, W]
        output = F.Reshape(self.name + ".reshape_out")(
            output_reshaped,
            F._from_data(
                self.name + ".reshape_shape_out",
                np.array([N, C, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        return output

    def analytical_param_count(self) -> int:
        """
        Calculate total number of trainable parameters.

        GridMask has no trainable parameters - it's a data augmentation module.

        Returns:
            0 (no parameters)
        """
        return 0
