#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for grid_mask.py: Grid and GridMask classes.
Tests shape and numerical equivalence between PyTorch and TTSIM versions.

Both versions share the same numpy+PIL mask generation logic. The difference
is only in how the mask is applied to the tensor (torch ops vs ttsim F ops).
To get deterministic, comparable results we seed numpy identically for both.
"""

import functools
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import os, sys
_POLARIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
sys.path.insert(0, _POLARIS_DIR)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ===========================================================================
# Manual reimplementations of mmcv decorators
# These replace `from mmcv.runner import force_fp32, auto_fp16` so the code
# can run without the mmcv package while preserving the original API.
# ===========================================================================

def auto_fp16(apply_to=None):
    """Manual reimplementation of mmcv.runner.auto_fp16.

    When ``self.fp16_enabled`` is True, casts floating-point tensor inputs
    to ``torch.float16`` before calling the wrapped method.
    When False (default) the decorator is a no-op.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, 'fp16_enabled', False):
                return fn(self, *args, **kwargs)
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                    new_args.append(arg.half())
                else:
                    new_args.append(arg)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    new_kwargs[k] = v.half()
                else:
                    new_kwargs[k] = v
            return fn(self, *new_args, **new_kwargs)
        return wrapper
    return decorator


def force_fp32(apply_to=None):
    """Manual reimplementation of mmcv.runner.force_fp32.

    Casts floating-point tensor inputs to ``torch.float32`` before calling
    the wrapped method, ensuring computation stays in full precision.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            new_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                    new_args.append(arg.float())
                else:
                    new_args.append(arg)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    new_kwargs[k] = v.float()
                else:
                    new_kwargs[k] = v
            return fn(self, *new_args, **new_kwargs)
        return wrapper
    return decorator


# ===========================================================================
# PyTorch versions (inlined from original grid_mask.py)
# NOTE: torch.cuda.synchronize / .cuda() guarded / removed for CPU-only env
# NOTE: mmcv decorators manually reimplemented above (no mmcv dependency)
# ===========================================================================

class Grid_PyTorch(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label
        h = img.size(1)
        w = img.size(2)
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
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
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).float()
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask

        return img, label


class GridMask_PyTorch(nn.Module):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        super(GridMask_PyTorch, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enabled = False

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    @auto_fp16()
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
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
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.from_numpy(mask).to(x.dtype)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


# ===========================================================================
# TTSIM versions (inlined from converted grid_mask.py)
# ===========================================================================

class Grid_TTSIM(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self._call_count = 0

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def _generate_mask(self, h, w):
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
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
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask).astype(np.float32)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        return mask

    def __call__(self, img, label):
        if np.random.rand() > self.prob:
            return img, label

        self._call_count += 1
        cc = self._call_count

        h, w = img.shape[1], img.shape[2]
        C = img.shape[0]

        mask_np = self._generate_mask(h, w)

        mask_tensor = F._from_data(
            f'grid.mask_c{cc}', mask_np.reshape(1, h, w), is_const=True)
        mask_expanded = F.Tile(f'grid.mask_tile_c{cc}')(
            mask_tensor,
            F._from_data(f'grid.tile_reps_c{cc}', np.array([C, 1, 1], dtype=np.int64), is_const=True))

        if self.offset:
            offset_np = (2 * (np.random.rand(h, w) - 0.5)).astype(np.float32)
            offset_tensor = F._from_data(
                f'grid.offset_c{cc}', offset_np.reshape(1, h, w), is_const=True)
            offset_expanded = F.Tile(f'grid.offset_tile_c{cc}')(
                offset_tensor,
                F._from_data(f'grid.offset_tile_reps_c{cc}', np.array([C, 1, 1], dtype=np.int64), is_const=True))

            one = F._from_data(f'grid.one_c{cc}', np.array(1.0, dtype=np.float32), is_const=True)
            inv_mask = F.Sub(f'grid.inv_mask_c{cc}')(one, mask_expanded)
            offset_part = F.Mul(f'grid.mul_offset_c{cc}')(inv_mask, offset_expanded)
            masked = F.Mul(f'grid.mul_mask_c{cc}')(img, mask_expanded)
            img_out = F.Add(f'grid.add_offset_c{cc}')(masked, offset_part)
        else:
            img_out = F.Mul(f'grid.mul_c{cc}')(img, mask_expanded)

        return img_out, label


class GridMask_TTSIM(SimNN.Module):
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

    def set_training(self, mode):
        self.training = mode

    def _generate_mask(self, h, w):
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
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
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask).astype(np.float32)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        return mask

    def __call__(self, x):
        N, C, H, W = x.shape
        if np.random.rand() > self.prob or not self.training:
            return x

        self._call_count += 1
        cc = self._call_count

        mask_np = self._generate_mask(H, W)

        x_reshaped = F.Reshape(f'{self.name}.reshape_in_c{cc}')(
            x,
            F._from_data(f'{self.name}.reshape_in_shape_c{cc}',
                          np.array([N * C, H, W], dtype=np.int64), is_const=True))

        mask_tensor = F._from_data(
            f'{self.name}.mask_c{cc}', mask_np.reshape(1, H, W), is_const=True)
        mask_expanded = F.Tile(f'{self.name}.mask_tile_c{cc}')(
            mask_tensor,
            F._from_data(f'{self.name}.tile_reps_c{cc}',
                          np.array([N * C, 1, 1], dtype=np.int64), is_const=True))

        if self.offset:
            offset_np = (2 * (np.random.rand(H, W) - 0.5)).astype(np.float32)
            offset_tensor = F._from_data(
                f'{self.name}.offset_c{cc}', offset_np.reshape(1, H, W), is_const=True)
            offset_expanded = F.Tile(f'{self.name}.offset_tile_c{cc}')(
                offset_tensor,
                F._from_data(f'{self.name}.offset_tile_reps_c{cc}',
                              np.array([N * C, 1, 1], dtype=np.int64), is_const=True))

            masked_x = F.Mul(f'{self.name}.mul_mask_c{cc}')(x_reshaped, mask_expanded)
            one = F._from_data(f'{self.name}.one_c{cc}', np.array(1.0, dtype=np.float32), is_const=True)
            inv_mask = F.Sub(f'{self.name}.inv_mask_c{cc}')(one, mask_expanded)
            offset_part = F.Mul(f'{self.name}.mul_offset_c{cc}')(offset_expanded, inv_mask)
            output_reshaped = F.Add(f'{self.name}.add_offset_c{cc}')(masked_x, offset_part)
        else:
            output_reshaped = F.Mul(f'{self.name}.mul_c{cc}')(x_reshaped, mask_expanded)

        output = F.Reshape(f'{self.name}.reshape_out_c{cc}')(
            output_reshaped,
            F._from_data(f'{self.name}.reshape_out_shape_c{cc}',
                          np.array([N, C, H, W], dtype=np.int64), is_const=True))
        return output


# ===========================================================================
# Helpers
# ===========================================================================

def _generate_mask_numpy(h, w, use_h, use_w, ratio, mode, rotate, seed):
    """Deterministic mask generation identical to both versions."""
    rng = np.random.RandomState(seed)
    hh = int(1.5 * h)
    ww = int(1.5 * w)
    d = rng.randint(2, min(h, w))
    if ratio == 1:
        l = rng.randint(1, d)
    else:
        l = min(max(int(d * ratio + 0.5), 1), d - 1)
    mask = np.ones((hh, ww), np.float32)
    st_h = rng.randint(d)
    st_w = rng.randint(d)
    if use_h:
        for i in range(hh // d):
            s = d * i + st_h
            t = min(s + l, hh)
            mask[s:t, :] = 0
    if use_w:
        for i in range(ww // d):
            s = d * i + st_w
            t = min(s + l, ww)
            mask[:, s:t] = 0

    r = rng.randint(rotate) if rotate > 0 else 0
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask).astype(np.float32)
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

    if mode == 1:
        mask = 1 - mask
    return mask


def _apply_mask_pytorch(x_np, mask, offset_on, offset_np=None):
    """Apply mask using PyTorch ops (CPU), return numpy result [N,C,H,W]."""
    x = torch.from_numpy(x_np).float()
    n, c, h, w = x.shape
    x_flat = x.view(-1, h, w)
    mask_t = torch.from_numpy(mask).float().expand_as(x_flat)
    if offset_on and offset_np is not None:
        offset_t = torch.from_numpy(offset_np).float()
        x_flat = x_flat * mask_t + offset_t.expand_as(x_flat) * (1 - mask_t)
    else:
        x_flat = x_flat * mask_t
    return x_flat.view(n, c, h, w).numpy()


def _apply_mask_ttsim(x_np, mask, offset_on, offset_np=None, tag='test'):
    """Apply mask using TTSIM F ops, return numpy result [N,C,H,W]."""
    N, C, H, W = x_np.shape
    x_tensor = F._from_data(f'{tag}.input', x_np)

    # reshape [N,C,H,W] -> [N*C,H,W]
    x_reshaped = F.Reshape(f'{tag}.resh_in')(
        x_tensor,
        F._from_data(f'{tag}.resh_in_s', np.array([N * C, H, W], dtype=np.int64), is_const=True))

    mask_t = F._from_data(f'{tag}.mask', mask.reshape(1, H, W), is_const=True)
    mask_exp = F.Tile(f'{tag}.tile')(
        mask_t,
        F._from_data(f'{tag}.tile_r', np.array([N * C, 1, 1], dtype=np.int64), is_const=True))

    if offset_on and offset_np is not None:
        off_t = F._from_data(f'{tag}.off', offset_np.reshape(1, H, W), is_const=True)
        off_exp = F.Tile(f'{tag}.off_tile')(
            off_t,
            F._from_data(f'{tag}.off_tile_r', np.array([N * C, 1, 1], dtype=np.int64), is_const=True))

        masked = F.Mul(f'{tag}.mul_m')(x_reshaped, mask_exp)
        one = F._from_data(f'{tag}.one', np.array(1.0, dtype=np.float32), is_const=True)
        inv = F.Sub(f'{tag}.inv')(one, mask_exp)
        off_part = F.Mul(f'{tag}.mul_off')(off_exp, inv)
        out_flat = F.Add(f'{tag}.add')(masked, off_part)
    else:
        out_flat = F.Mul(f'{tag}.mul')(x_reshaped, mask_exp)

    out = F.Reshape(f'{tag}.resh_out')(
        out_flat,
        F._from_data(f'{tag}.resh_out_s', np.array([N, C, H, W], dtype=np.int64), is_const=True))

    return np.array(out.data) if out.data is not None else None


# ===========================================================================
# Main validation
# ===========================================================================

def main():
    print("=" * 70)
    print("grid_mask.py Validation: PyTorch vs TTSIM")
    print("=" * 70)

    all_pass = True
    atol, rtol = 1e-6, 1e-6

    # ------------------------------------------------------------------
    # Test 1 — GridMask: simple masking (no offset, mode=0)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 1: GridMask — simple masking (offset=False, mode=0)")
    print("-" * 70)

    N, C, H, W = 2, 3, 32, 32
    seed = 42
    np.random.seed(seed)
    x_np = np.random.randn(N, C, H, W).astype(np.float32)

    mask = _generate_mask_numpy(H, W, use_h=True, use_w=True,
                                ratio=0.5, mode=0, rotate=1, seed=seed)

    out_pt = _apply_mask_pytorch(x_np, mask, offset_on=False)
    out_tt = _apply_mask_ttsim(x_np, mask, offset_on=False, tag='t1')

    shape_ok = (out_pt.shape == out_tt.shape)
    print(f"  Input shape:  {x_np.shape}")
    print(f"  Mask shape:   {mask.shape}")
    print(f"  PT  output shape: {out_pt.shape}")
    print(f"  TT  output shape: {out_tt.shape}")
    print(f"  Shape match: {'[PASS]' if shape_ok else '[FAIL]'}")
    if not shape_ok:
        all_pass = False

    if out_tt is not None:
        diff = np.abs(out_pt - out_tt)
        print(f"  Tolerance: atol={atol}, rtol={rtol}")
        print(f"  Max  abs diff:  {diff.max():.10f}")
        print(f"  Mean abs diff:  {diff.mean():.10f}")
        print(f"  PT  stats: min={out_pt.min():.6f}, max={out_pt.max():.6f}, mean={out_pt.mean():.6f}")
        print(f"  TT  stats: min={out_tt.min():.6f}, max={out_tt.max():.6f}, mean={out_tt.mean():.6f}")
        num_ok = np.allclose(out_pt, out_tt, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok else '[FAIL]'}")
        if not num_ok:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 2 — GridMask: inverted mode (mode=1)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 2: GridMask — inverted mode (mode=1)")
    print("-" * 70)

    mask_inv = _generate_mask_numpy(H, W, use_h=True, use_w=True,
                                    ratio=0.5, mode=1, rotate=1, seed=99)

    out_pt2 = _apply_mask_pytorch(x_np, mask_inv, offset_on=False)
    out_tt2 = _apply_mask_ttsim(x_np, mask_inv, offset_on=False, tag='t2')

    shape_ok2 = (out_pt2.shape == out_tt2.shape)
    print(f"  PT  output shape: {out_pt2.shape}")
    print(f"  TT  output shape: {out_tt2.shape}")
    print(f"  Shape match: {'[PASS]' if shape_ok2 else '[FAIL]'}")
    if not shape_ok2:
        all_pass = False

    if out_tt2 is not None:
        diff2 = np.abs(out_pt2 - out_tt2)
        print(f"  Max  abs diff:  {diff2.max():.10f}")
        print(f"  Mean abs diff:  {diff2.mean():.10f}")
        num_ok2 = np.allclose(out_pt2, out_tt2, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok2 else '[FAIL]'}")
        if not num_ok2:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 3 — GridMask: with offset
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 3: GridMask — with offset (offset=True)")
    print("-" * 70)

    mask3 = _generate_mask_numpy(H, W, use_h=True, use_w=True,
                                 ratio=0.5, mode=0, rotate=1, seed=77)
    np.random.seed(77)
    offset_np = (2 * (np.random.rand(H, W) - 0.5)).astype(np.float32)

    out_pt3 = _apply_mask_pytorch(x_np, mask3, offset_on=True, offset_np=offset_np)
    out_tt3 = _apply_mask_ttsim(x_np, mask3, offset_on=True, offset_np=offset_np, tag='t3')

    shape_ok3 = (out_pt3.shape == out_tt3.shape)
    print(f"  PT  output shape: {out_pt3.shape}")
    print(f"  TT  output shape: {out_tt3.shape}")
    print(f"  Shape match: {'[PASS]' if shape_ok3 else '[FAIL]'}")
    if not shape_ok3:
        all_pass = False

    if out_tt3 is not None:
        diff3 = np.abs(out_pt3 - out_tt3)
        print(f"  Max  abs diff:  {diff3.max():.10f}")
        print(f"  Mean abs diff:  {diff3.mean():.10f}")
        num_ok3 = np.allclose(out_pt3, out_tt3, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok3 else '[FAIL]'}")
        if not num_ok3:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 4 — Grid class: simple masking on [C,H,W] tensor
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 4: Grid class — [C,H,W] tensor, offset=False")
    print("-" * 70)

    C4, H4, W4 = 3, 24, 24
    np.random.seed(55)
    img_np = np.random.randn(C4, H4, W4).astype(np.float32)
    label_np = np.array([1, 0, 1])

    # PyTorch path
    np.random.seed(123)
    grid_pt = Grid_PyTorch(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.)
    img_pt = torch.from_numpy(img_np.copy()).float()
    out_pt4, lbl_pt4 = grid_pt(img_pt, label_np.copy())
    out_pt4_np = out_pt4.numpy()

    # TTSIM path — same seed
    np.random.seed(123)
    grid_tt = Grid_TTSIM(use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.)
    img_tt = F._from_data('grid_t4_img', img_np.copy())
    out_tt4, lbl_tt4 = grid_tt(img_tt, label_np.copy())
    out_tt4_np = out_tt4.data if out_tt4.data is not None else None

    if out_tt4_np is not None:
        shape_ok4 = (list(out_pt4_np.shape) == list(out_tt4.shape))
        print(f"  Input shape:  [{C4},{H4},{W4}]")
        print(f"  PT  output shape: {out_pt4_np.shape}")
        print(f"  TT  output shape: {list(out_tt4.shape)}")
        print(f"  Shape match: {'[PASS]' if shape_ok4 else '[FAIL]'}")
        if not shape_ok4:
            all_pass = False

        diff4 = np.abs(out_pt4_np - out_tt4_np)
        print(f"  Max  abs diff:  {diff4.max():.10f}")
        print(f"  Mean abs diff:  {diff4.mean():.10f}")
        print(f"  PT  stats: min={out_pt4_np.min():.6f}, max={out_pt4_np.max():.6f}, mean={out_pt4_np.mean():.6f}")
        print(f"  TT  stats: min={out_tt4_np.min():.6f}, max={out_tt4_np.max():.6f}, mean={out_tt4_np.mean():.6f}")
        num_ok4 = np.allclose(out_pt4_np, out_tt4_np, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok4 else '[FAIL]'}")
        if not num_ok4:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 5 — Grid class: with offset
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 5: Grid class — [C,H,W] tensor, offset=True")
    print("-" * 70)

    np.random.seed(200)
    grid_pt5 = Grid_PyTorch(use_h=True, use_w=True, rotate=1, offset=True, ratio=0.5, mode=0, prob=1.)
    img_pt5 = torch.from_numpy(img_np.copy()).float()
    out_pt5, _ = grid_pt5(img_pt5, label_np.copy())
    out_pt5_np = out_pt5.numpy()

    np.random.seed(200)
    grid_tt5 = Grid_TTSIM(use_h=True, use_w=True, rotate=1, offset=True, ratio=0.5, mode=0, prob=1.)
    img_tt5 = F._from_data('grid_t5_img', img_np.copy())
    out_tt5, _ = grid_tt5(img_tt5, label_np.copy())
    out_tt5_np = out_tt5.data if out_tt5.data is not None else None

    if out_tt5_np is not None:
        shape_ok5 = (list(out_pt5_np.shape) == list(out_tt5.shape))
        print(f"  PT  output shape: {out_pt5_np.shape}")
        print(f"  TT  output shape: {list(out_tt5.shape)}")
        print(f"  Shape match: {'[PASS]' if shape_ok5 else '[FAIL]'}")
        if not shape_ok5:
            all_pass = False

        diff5 = np.abs(out_pt5_np - out_tt5_np)
        print(f"  Max  abs diff:  {diff5.max():.10f}")
        print(f"  Mean abs diff:  {diff5.mean():.10f}")
        num_ok5 = np.allclose(out_pt5_np, out_tt5_np, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok5 else '[FAIL]'}")
        if not num_ok5:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 6 — GridMask class: larger batch, different spatial size
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 6: GridMask — larger batch [4,8,64,64]")
    print("-" * 70)

    N6, C6, H6, W6 = 4, 8, 64, 64
    np.random.seed(300)
    x6_np = np.random.randn(N6, C6, H6, W6).astype(np.float32)

    mask6 = _generate_mask_numpy(H6, W6, use_h=True, use_w=True,
                                 ratio=0.5, mode=0, rotate=1, seed=301)

    out_pt6 = _apply_mask_pytorch(x6_np, mask6, offset_on=False)
    out_tt6 = _apply_mask_ttsim(x6_np, mask6, offset_on=False, tag='t6')

    shape_ok6 = (out_pt6.shape == out_tt6.shape)
    print(f"  Input shape:  ({N6},{C6},{H6},{W6})")
    print(f"  PT  output shape: {out_pt6.shape}")
    print(f"  TT  output shape: {out_tt6.shape}")
    print(f"  Shape match: {'[PASS]' if shape_ok6 else '[FAIL]'}")
    if not shape_ok6:
        all_pass = False

    if out_tt6 is not None:
        diff6 = np.abs(out_pt6 - out_tt6)
        print(f"  Max  abs diff:  {diff6.max():.10f}")
        print(f"  Mean abs diff:  {diff6.mean():.10f}")
        print(f"  PT  stats: min={out_pt6.min():.6f}, max={out_pt6.max():.6f}, mean={out_pt6.mean():.6f}")
        print(f"  TT  stats: min={out_tt6.min():.6f}, max={out_tt6.max():.6f}, mean={out_tt6.mean():.6f}")
        num_ok6 = np.allclose(out_pt6, out_tt6, atol=atol, rtol=rtol)
        print(f"  Numerical match: {'[PASS]' if num_ok6 else '[FAIL]'}")
        if not num_ok6:
            all_pass = False
    else:
        print("  [WARN] TTSIM data is None — skipped")
        all_pass = False

    # ------------------------------------------------------------------
    # Test 7 — set_prob validation
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 7: set_prob — probability update consistency")
    print("-" * 70)

    gm_pt = GridMask_PyTorch(use_h=True, use_w=True, prob=0.8)
    gm_tt = GridMask_TTSIM(name='gm_prob', use_h=True, use_w=True, prob=0.8)

    gm_pt.set_prob(5, 10)
    gm_tt.set_prob(5, 10)

    prob_match = (gm_pt.prob == gm_tt.prob)
    print(f"  epoch=5, max_epoch=10")
    print(f"  PT  prob: {gm_pt.prob}")
    print(f"  TT  prob: {gm_tt.prob}")
    print(f"  Prob match: {'[PASS]' if prob_match else '[FAIL]'}")
    if not prob_match:
        all_pass = False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: [PASS] ALL TESTS PASSED — TTSIM matches PyTorch")
    else:
        print("OVERALL: [FAIL] SOME TESTS FAILED — see details above")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
