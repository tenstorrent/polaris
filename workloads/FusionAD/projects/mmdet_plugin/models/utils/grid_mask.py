
#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TTSIM conversion of grid_mask.py - preserves all logic, inputs, and outputs.

CONVERTED CLASSES:
------------------
1. Grid       - Plain object that generates + applies a grid mask to an image
2. GridMask   - SimNN.Module version for use inside model graphs

CONVERSION NOTES:
-----------------
- torch.from_numpy(...).float()  → F._from_data(name, arr)
- mask.expand_as(img)            → F.Tile to broadcast [1,H,W] -> [N*C,H,W]
- img * mask                     → F.Mul
- 1 - mask                       → F.Sub(one, mask)
- x.view(n,c,h,w)               → F.Reshape
- x.size()                       → x.shape
- nn.Module                      → SimNN.Module
- self.training                  → explicit flag stored on the instance
- @auto_fp16 / @force_fp32       → removed (not applicable in ttsim)
- torch.cuda calls               → removed (ttsim is synchronous / no CUDA)
- Mask generation logic (numpy + PIL) is preserved exactly as-is
"""

#------------------------------PyTorch (original)---------------------------
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# from mmcv.runner import force_fp32, auto_fp16
#
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
#
#     def set_prob(self, epoch, max_epoch):
#         self.prob = self.st_prob * epoch / max_epoch
#
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
#
#
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
#     def set_prob(self, epoch, max_epoch):
#         self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5
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
#         if self.mode == 1:
#             mask = 1-mask
#         mask = mask.expand_as(x)
#         if self.offset:
#             offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
#             x = x * mask + offset * (1 - mask)
#         else:
#             x = x * mask
#
#         return x.view(n,c,h,w)
#---------------------------------------------------------------------------

import numpy as np
from PIL import Image

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ============================================================================
# 1. Grid — plain Python object (not an nn.Module / SimNN.Module)
# ============================================================================
class Grid(object):
    """Grid data-augmentation transform (TTSIM version).

    Generates a grid mask and applies it element-wise to an image tensor.
    This class is NOT a SimNN.Module — it operates on SimTensors using F ops.
    """

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        # Internal call counter for unique op naming
        self._call_count = 0

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    # -- internal: numpy mask generation (identical to PyTorch version) ------
    def _generate_mask(self, h, w):
        """Generate a grid mask of shape [h, w] using numpy + PIL (same logic as original)."""
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask: np.ndarray = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask_img = Image.fromarray(np.uint8(mask))
        mask_img = mask_img.rotate(r)
        mask = np.asarray(mask_img).astype(np.float32)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask

        return mask

    def __call__(self, img, label):
        """
        Apply grid mask to *img* (a SimTensor of shape [C, H, W]).

        Args:
            img:   SimTensor [C, H, W]
            label: passed through unchanged

        Returns:
            (masked_img, label)
        """
        if np.random.rand() > self.prob:
            return img, label

        self._call_count += 1
        cc = self._call_count

        # img shape: [C, H, W]
        h, w = img.shape[1], img.shape[2]
        C = img.shape[0]

        mask_np = self._generate_mask(h, w)  # [H, W]

        # Create mask SimTensor [1, H, W] for broadcasting with [C, H, W]
        mask_tensor = F._from_data(
            f'grid.mask_c{cc}',
            mask_np.reshape(1, h, w),
            is_const=True,
        )

        # Tile to [C, H, W]
        mask_expanded = F.Tile(f'grid.mask_tile_c{cc}')(
            mask_tensor,
            F._from_data(f'grid.tile_reps_c{cc}', np.array([C, 1, 1], dtype=np.int64), is_const=True),
        )

        if self.offset:
            offset_np = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            offset_tensor = F._from_data(
                f'grid.offset_c{cc}',
                offset_np.reshape(1, h, w),
                is_const=True,
            )
            offset_expanded = F.Tile(f'grid.offset_tile_c{cc}')(
                offset_tensor,
                F._from_data(f'grid.offset_tile_reps_c{cc}', np.array([C, 1, 1], dtype=np.int64), is_const=True),
            )

            # inv_mask = 1 - mask
            one = F._from_data(f'grid.one_c{cc}', np.array([1.0], dtype=np.float32), is_const=True)
            inv_mask = F.Sub(f'grid.inv_mask_c{cc}')(one, mask_expanded)

            # offset_part = (1 - mask) * offset
            offset_part = F.Mul(f'grid.mul_offset_c{cc}')(inv_mask, offset_expanded)

            # img * mask + offset_part
            masked = F.Mul(f'grid.mul_mask_c{cc}')(img, mask_expanded)
            img_out = F.Add(f'grid.add_offset_c{cc}')(masked, offset_part)
        else:
            img_out = F.Mul(f'grid.mul_c{cc}')(img, mask_expanded)

        return img_out, label


# ============================================================================
# 2. GridMask — SimNN.Module (replaces nn.Module)
# ============================================================================
class GridMask(SimNN.Module):
    """GridMask data-augmentation module (TTSIM version).

    Applies a grid-based masking pattern to a batch of images.

    Args:
        name:    Module name for the ttsim graph
        use_h:   Apply grid mask along height
        use_w:   Apply grid mask along width
        rotate:  Maximum rotation angle in degrees
        offset:  If True, fill masked regions with random values instead of 0
        ratio:   Ratio of masked line width to grid spacing
        mode:    0 = mask out grid lines, 1 = invert
        prob:    Probability of applying the mask

    Shape:
        Input:  [N, C, H, W]
        Output: [N, C, H, W]
    """

    def __init__(self, name, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super().__init__()
        self.name = name
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.training = True
        self._call_count = 0

        super().link_op2module()

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def set_training(self, mode: bool):
        self.training = mode

    # -- internal: numpy mask generation (identical to PyTorch version) ------
    def _generate_mask(self, h, w):
        """Generate a grid mask of shape [h, w]."""
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask: np.ndarray = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask_img = Image.fromarray(np.uint8(mask))
        mask_img = mask_img.rotate(r)
        mask = np.asarray(mask_img).astype(np.float32)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask

        return mask

    def __call__(self, x):
        """
        Apply grid mask to input batch.

        Args:
            x: SimTensor of shape [N, C, H, W]

        Returns:
            SimTensor of shape [N, C, H, W]
        """
        N, C, H, W = x.shape

        if np.random.rand() > self.prob or not self.training:
            return x

        self._call_count += 1
        cc = self._call_count

        mask_np = self._generate_mask(H, W)  # [H, W]

        # --- reshape x: [N, C, H, W] -> [N*C, H, W] ---
        x_reshaped = F.Reshape(f'{self.name}.reshape_in_c{cc}')(
            x,
            F._from_data(
                f'{self.name}.reshape_in_shape_c{cc}',
                np.array([N * C, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        # --- mask tensor [1, H, W] -> tile to [N*C, H, W] ---
        mask_tensor = F._from_data(
            f'{self.name}.mask_c{cc}',
            mask_np.reshape(1, H, W),
            is_const=True,
        )
        mask_expanded = F.Tile(f'{self.name}.mask_tile_c{cc}')(
            mask_tensor,
            F._from_data(
                f'{self.name}.tile_reps_c{cc}',
                np.array([N * C, 1, 1], dtype=np.int64),
                is_const=True,
            ),
        )

        # --- apply mask ---
        if self.offset:
            offset_np = (2 * (np.random.rand(H, W) - 0.5)).astype(np.float32)
            offset_tensor = F._from_data(
                f'{self.name}.offset_c{cc}',
                offset_np.reshape(1, H, W),
                is_const=True,
            )
            offset_expanded = F.Tile(f'{self.name}.offset_tile_c{cc}')(
                offset_tensor,
                F._from_data(
                    f'{self.name}.offset_tile_reps_c{cc}',
                    np.array([N * C, 1, 1], dtype=np.int64),
                    is_const=True,
                ),
            )

            # x * mask
            masked_x = F.Mul(f'{self.name}.mul_mask_c{cc}')(x_reshaped, mask_expanded)

            # (1 - mask)
            one = F._from_data(f'{self.name}.one_c{cc}', np.array([1.0], dtype=np.float32), is_const=True)
            inv_mask = F.Sub(f'{self.name}.inv_mask_c{cc}')(one, mask_expanded)

            # offset * (1 - mask)
            offset_part = F.Mul(f'{self.name}.mul_offset_c{cc}')(offset_expanded, inv_mask)

            # x * mask + offset * (1 - mask)
            output_reshaped = F.Add(f'{self.name}.add_offset_c{cc}')(masked_x, offset_part)
        else:
            output_reshaped = F.Mul(f'{self.name}.mul_c{cc}')(x_reshaped, mask_expanded)

        # --- reshape back: [N*C, H, W] -> [N, C, H, W] ---
        output = F.Reshape(f'{self.name}.reshape_out_c{cc}')(
            output_reshaped,
            F._from_data(
                f'{self.name}.reshape_out_shape_c{cc}',
                np.array([N, C, H, W], dtype=np.int64),
                is_const=True,
            ),
        )

        return output
