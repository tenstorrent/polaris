
# =============================================================================
# ORIGINAL TORCH CODE (from FusionAD)
# Source: FusionAD/projects/mmdet3d_plugin/core/bbox/util.py
# =============================================================================
# import torch 
#
#
# def normalize_bbox(bboxes, pc_range):
#
#     cx = bboxes[..., 0:1]
#     cy = bboxes[..., 1:2]
#     cz = bboxes[..., 2:3]
#     w = bboxes[..., 3:4].log()
#     l = bboxes[..., 4:5].log()
#     h = bboxes[..., 5:6].log()
#
#     rot = bboxes[..., 6:7]
#     if bboxes.size(-1) > 7:
#         vx = bboxes[..., 7:8] 
#         vy = bboxes[..., 8:9]
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
#         )
#     else:
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
#         )
#     return normalized_bboxes
#
# def denormalize_bbox(normalized_bboxes, pc_range):
#     # rotation 
#     rot_sine = normalized_bboxes[..., 6:7]
#
#     rot_cosine = normalized_bboxes[..., 7:8]
#     rot = torch.atan2(rot_sine, rot_cosine)
#
#     # center in the bev
#     cx = normalized_bboxes[..., 0:1]
#     cy = normalized_bboxes[..., 1:2]
#     cz = normalized_bboxes[..., 4:5]
#
#     # size
#     w = normalized_bboxes[..., 2:3]
#     l = normalized_bboxes[..., 3:4]
#     h = normalized_bboxes[..., 5:6]
#
#     w = w.exp() 
#     l = l.exp() 
#     h = h.exp() 
#     if normalized_bboxes.size(-1) > 8:
#          # velocity 
#         vx = normalized_bboxes[:, 8:9]
#         vy = normalized_bboxes[:, 9:10]
#         denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
#     else:
#         denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
#     return denormalized_bboxes
# =============================================================================
# END OF ORIGINAL TORCH CODE
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of bbox utilities for FusionAD.

Contains SimNN.Module classes that register all ops in the TTSim graph,
so they can be traced and simulated on Tenstorrent hardware.

Original: projects/mmdet3d_plugin/core/bbox/util.py

Classes:
  - NormalizeBbox    : Convert raw bbox to normalized form (graph ops).
  - DenormalizeBbox  : Inverse of NormalizeBbox (graph ops).

Also provides numpy convenience functions for host-side use:
  - normalize_bbox_np
  - denormalize_bbox_np
"""

#-------------------------------PyTorch--------------------------------

# import torch
#
#
# def normalize_bbox(bboxes, pc_range):
#
#     cx = bboxes[..., 0:1]
#     cy = bboxes[..., 1:2]
#     cz = bboxes[..., 2:3]
#     w = bboxes[..., 3:4].log()
#     l = bboxes[..., 4:5].log()
#     h = bboxes[..., 5:6].log()
#
#     rot = bboxes[..., 6:7]
#     if bboxes.size(-1) > 7:
#         vx = bboxes[..., 7:8]
#         vy = bboxes[..., 8:9]
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
#         )
#     else:
#         normalized_bboxes = torch.cat(
#             (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
#         )
#     return normalized_bboxes
#
# def denormalize_bbox(normalized_bboxes, pc_range):
#     # rotation
#     rot_sine = normalized_bboxes[..., 6:7]
#
#     rot_cosine = normalized_bboxes[..., 7:8]
#     rot = torch.atan2(rot_sine, rot_cosine)
#
#     # center in the bev
#     cx = normalized_bboxes[..., 0:1]
#     cy = normalized_bboxes[..., 1:2]
#     cz = normalized_bboxes[..., 4:5]
#
#     # size
#     w = normalized_bboxes[..., 2:3]
#     l = normalized_bboxes[..., 3:4]
#     h = normalized_bboxes[..., 5:6]
#
#     w = w.exp()
#     l = l.exp()
#     h = h.exp()
#     if normalized_bboxes.size(-1) > 8:
#          # velocity
#         vx = normalized_bboxes[:, 8:9]
#         vy = normalized_bboxes[:, 9:10]
#         denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
#     else:
#         denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
#     return denormalized_bboxes


#-------------------------------TTSIM-----------------------------------

import sys
import os
from loguru import logger

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..','..'))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN


# ======================================================================
# NormalizeBbox  (TTSim graph module)
# ======================================================================

class NormalizeBbox(SimNN.Module):
    """
    TTSim implementation of normalize_bbox.

    Input  layout: [..., cx, cy, cz, w, l, h, rot,  vx, vy]  (9-dim)
                   [..., cx, cy, cz, w, l, h, rot]            (7-dim)
    Output layout: [..., cx, cy, log(w), log(l), cz, log(h),
                        sin(rot), cos(rot),  vx, vy]          (10-dim)
                   [..., cx, cy, log(w), log(l), cz, log(h),
                        sin(rot), cos(rot)]                    (8-dim)

    Args:
        name (str): Module name.
        has_velocity (bool): Whether input has velocity (last dim > 7).
    """

    def __init__(self, name, has_velocity=True):
        super().__init__()
        self.name = name
        self.has_velocity = has_velocity

        # Unary ops
        self.log_w   = F.Log(f'{name}.log_w')
        self.log_l   = F.Log(f'{name}.log_l')
        self.log_h   = F.Log(f'{name}.log_h')
        self.sin_rot = F.Sin(f'{name}.sin_rot')
        self.cos_rot = F.Cos(f'{name}.cos_rot')

        # Concat — 10-dim with velocity, 8-dim without
        n_parts = 10 if has_velocity else 8
        self.concat = F.ConcatX(f'{name}.concat', axis=-1)

        super().link_op2module()

    def __call__(self, bboxes):
        """
        Args:
            bboxes: SimTensor [..., 9] (velocity) or [..., 7] (no velocity).
        Returns:
            SimTensor [..., 10] or [..., 8].
        """
        ndim = len(bboxes.shape)
        last = bboxes.shape[-1]

        # Helper to slice [..., start:end] from bboxes
        def _slice(field_name, start, end):
            starts = [0] * (ndim - 1) + [start]
            ends   = [int(s) for s in bboxes.shape[:-1]] + [end]
            axes   = list(range(ndim))
            out_shape = list(bboxes.shape[:-1]) + [end - start]
            st = F._from_data(f'{self.name}.{field_name}_st',
                              np.array(starts, dtype=np.int64), is_const=True)
            setattr(self, st.name, st)
            en = F._from_data(f'{self.name}.{field_name}_en',
                              np.array(ends, dtype=np.int64), is_const=True)
            setattr(self, en.name, en)
            ax = F._from_data(f'{self.name}.{field_name}_ax',
                              np.array(axes, dtype=np.int64), is_const=True)
            setattr(self, ax.name, ax)
            sl = F.SliceF(f'{self.name}.{field_name}_sl', out_shape=out_shape)
            setattr(self, sl.name, sl)
            result = sl(bboxes, st, en, ax)
            setattr(self, result.name, result)
            return result

        cx  = _slice('cx',  0, 1)
        cy  = _slice('cy',  1, 2)
        cz  = _slice('cz',  2, 3)
        w   = _slice('w',   3, 4)
        l   = _slice('l',   4, 5)
        h   = _slice('h',   5, 6)
        rot = _slice('rot', 6, 7)

        log_w = self.log_w(w);       setattr(self, log_w.name, log_w)
        log_l = self.log_l(l);       setattr(self, log_l.name, log_l)
        log_h = self.log_h(h);       setattr(self, log_h.name, log_h)
        sin_r = self.sin_rot(rot);   setattr(self, sin_r.name, sin_r)
        cos_r = self.cos_rot(rot);   setattr(self, cos_r.name, cos_r)

        if self.has_velocity:
            vx = _slice('vx', 7, 8)
            vy = _slice('vy', 8, 9)
            out = self.concat(cx, cy, log_w, log_l, cz, log_h,
                              sin_r, cos_r, vx, vy)
        else:
            out = self.concat(cx, cy, log_w, log_l, cz, log_h,
                              sin_r, cos_r)
        setattr(self, out.name, out)
        return out


# ======================================================================
# DenormalizeBbox  (TTSim graph module)
# ======================================================================

class DenormalizeBbox(SimNN.Module):
    """
    TTSim implementation of denormalize_bbox (inverse of normalize).

    Implements atan2(y, x) using the identity:
        atan2(y, x) = 2 * atan( y / (sqrt(x² + y²) + x) )
    This is numerically stable for all quadrants except the
    singular point x < 0, y = 0 (angle = ±π).

    Input  layout: [..., cx, cy, log(w), log(l), cz, log(h),
                        sin(rot), cos(rot),  vx, vy]  (10-dim)
    Output layout: [..., cx, cy, cz, w, l, h, rot,  vx, vy]  (9-dim)

    Args:
        name (str): Module name.
        has_velocity (bool): Whether input has velocity (last dim > 8).
    """

    def __init__(self, name, has_velocity=True):
        super().__init__()
        self.name = name
        self.has_velocity = has_velocity

        # Exp for w, l, h
        self.exp_w = F.Exp(f'{name}.exp_w')
        self.exp_l = F.Exp(f'{name}.exp_l')
        self.exp_h = F.Exp(f'{name}.exp_h')

        # atan2(y, x) = 2 * atan(y / (sqrt(x² + y²) + x))
        self.sin_sq  = F.Mul(f'{name}.sin_sq')     # y²
        self.cos_sq  = F.Mul(f'{name}.cos_sq')     # x²
        self.sum_sq  = F.Add(f'{name}.sum_sq')     # x² + y²
        self.sqrt_op = F.Sqrt(f'{name}.sqrt')      # sqrt(x² + y²)
        self.denom   = F.Add(f'{name}.denom')      # sqrt(x²+y²) + x
        self.ratio   = F.Div(f'{name}.ratio')      # y / denom
        self.atan_op = F.Atan(f'{name}.atan')        # atan(ratio)
        self.two_const = F._from_data(f'{name}.two',
                                      np.array([2.0], dtype=np.float32),
                                      is_const=True)
        self.mul_two = F.Mul(f'{name}.mul_two')     # 2 * atan(...)

        # Concat
        self.concat = F.ConcatX(f'{name}.concat', axis=-1)

        super().link_op2module()

    def __call__(self, normalized_bboxes):
        """
        Args:
            normalized_bboxes: SimTensor [..., 10] or [..., 8].
        Returns:
            SimTensor [..., 9] or [..., 7].
        """
        ndim = len(normalized_bboxes.shape)

        def _slice(field_name, start, end):
            starts = [0] * (ndim - 1) + [start]
            ends   = [int(s) for s in normalized_bboxes.shape[:-1]] + [end]
            axes   = list(range(ndim))
            out_shape = list(normalized_bboxes.shape[:-1]) + [end - start]
            st = F._from_data(f'{self.name}.{field_name}_st',
                              np.array(starts, dtype=np.int64), is_const=True)
            setattr(self, st.name, st)
            en = F._from_data(f'{self.name}.{field_name}_en',
                              np.array(ends, dtype=np.int64), is_const=True)
            setattr(self, en.name, en)
            ax = F._from_data(f'{self.name}.{field_name}_ax',
                              np.array(axes, dtype=np.int64), is_const=True)
            setattr(self, ax.name, ax)
            sl = F.SliceF(f'{self.name}.{field_name}_sl', out_shape=out_shape)
            setattr(self, sl.name, sl)
            result = sl(normalized_bboxes, st, en, ax)
            setattr(self, result.name, result)
            return result

        cx        = _slice('cx',   0, 1)
        cy        = _slice('cy',   1, 2)
        log_w     = _slice('logw', 2, 3)
        log_l     = _slice('logl', 3, 4)
        cz        = _slice('cz',   4, 5)
        log_h     = _slice('logh', 5, 6)
        rot_sine  = _slice('sins', 6, 7)
        rot_cos   = _slice('cosr', 7, 8)

        # Exp for dimensions
        w = self.exp_w(log_w); setattr(self, w.name, w)
        l = self.exp_l(log_l); setattr(self, l.name, l)
        h = self.exp_h(log_h); setattr(self, h.name, h)

        # atan2(rot_sine, rot_cos) via 2 * atan(y / (sqrt(x²+y²) + x))
        y_sq = self.sin_sq(rot_sine, rot_sine);  setattr(self, y_sq.name, y_sq)
        x_sq = self.cos_sq(rot_cos, rot_cos);    setattr(self, x_sq.name, x_sq)
        sq_sum = self.sum_sq(x_sq, y_sq);        setattr(self, sq_sum.name, sq_sum)
        hyp = self.sqrt_op(sq_sum);              setattr(self, hyp.name, hyp)
        den = self.denom(hyp, rot_cos);          setattr(self, den.name, den)
        rat = self.ratio(rot_sine, den);         setattr(self, rat.name, rat)
        at  = self.atan_op(rat);                 setattr(self, at.name, at)
        rot = self.mul_two(self.two_const, at);  setattr(self, rot.name, rot)

        if self.has_velocity:
            vx = _slice('vx', 8, 9)
            vy = _slice('vy', 9, 10)
            out = self.concat(cx, cy, cz, w, l, h, rot, vx, vy)
        else:
            out = self.concat(cx, cy, cz, w, l, h, rot)
        setattr(self, out.name, out)
        return out


# ======================================================================
# Numpy convenience functions (for host-side / test use)
# ======================================================================

def normalize_bbox_np(bboxes, pc_range):
    """Numpy version of normalize_bbox (host-side convenience)."""
    cx  = bboxes[..., 0:1]
    cy  = bboxes[..., 1:2]
    cz  = bboxes[..., 2:3]
    w   = np.log(bboxes[..., 3:4])
    l   = np.log(bboxes[..., 4:5])
    h   = np.log(bboxes[..., 5:6])
    rot = bboxes[..., 6:7]
    if bboxes.shape[-1] > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        return np.concatenate(
            (cx, cy, w, l, cz, h, np.sin(rot), np.cos(rot), vx, vy), axis=-1)
    else:
        return np.concatenate(
            (cx, cy, w, l, cz, h, np.sin(rot), np.cos(rot)), axis=-1)


def denormalize_bbox_np(normalized_bboxes, pc_range):
    """Numpy version of denormalize_bbox (host-side convenience)."""
    rot_sine   = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = np.arctan2(rot_sine, rot_cosine)
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    w = np.exp(normalized_bboxes[..., 2:3])
    l = np.exp(normalized_bboxes[..., 3:4])
    h = np.exp(normalized_bboxes[..., 5:6])
    if normalized_bboxes.shape[-1] > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        return np.concatenate([cx, cy, cz, w, l, h, rot, vx, vy], axis=-1)
    else:
        return np.concatenate([cx, cy, cz, w, l, h, rot], axis=-1)


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("bbox/util.py — self-test (TTSim modules)")
    logger.info("=" * 60)

    np.random.seed(42)
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # ---- NormalizeBbox ----
    logger.info("\n--- NormalizeBbox (with velocity) ---")
    bboxes_np = np.array([[1.0, 2.0, 3.0, 0.5, 1.2, 0.8, 0.7, 1.5, -0.3]],
                          dtype=np.float32)
    norm_mod = NormalizeBbox('test_norm', has_velocity=True)
    bboxes_t = F._from_data('test_bboxes', bboxes_np, is_const=True)
    norm_out = norm_mod(bboxes_t)
    expected = normalize_bbox_np(bboxes_np, pc_range)
    assert np.allclose(norm_out.data, expected, atol=1e-6), \
        f"NormalizeBbox mismatch:\n  got {norm_out.data}\n  exp {expected}"
    logger.info(f"  [OK] shape {norm_out.shape}, matches numpy reference")

    # ---- DenormalizeBbox ----
    logger.info("\n--- DenormalizeBbox (with velocity) ---")
    denorm_mod = DenormalizeBbox('test_denorm', has_velocity=True)
    denorm_out = denorm_mod(norm_out)
    assert np.allclose(denorm_out.data, bboxes_np, atol=1e-5), \
        f"DenormalizeBbox mismatch:\n  got {denorm_out.data}\n  exp {bboxes_np}"
    logger.info(f"  [OK] shape {denorm_out.shape}, round-trip matches original")

    # ---- Without velocity ----
    logger.info("\n--- NormalizeBbox (no velocity) ---")
    bboxes7_np = bboxes_np[:, :7]
    norm_mod7 = NormalizeBbox('test_norm7', has_velocity=False)
    bboxes7_t = F._from_data('test_bboxes7', bboxes7_np, is_const=True)
    norm7_out = norm_mod7(bboxes7_t)
    expected7 = normalize_bbox_np(bboxes7_np, pc_range)
    assert np.allclose(norm7_out.data, expected7, atol=1e-6)
    logger.info(f"  [OK] shape {norm7_out.shape}")

    logger.info("\n--- DenormalizeBbox (no velocity) ---")
    denorm_mod7 = DenormalizeBbox('test_denorm7', has_velocity=False)
    denorm7_out = denorm_mod7(norm7_out)
    assert np.allclose(denorm7_out.data, bboxes7_np, atol=1e-5)
    logger.info(f"  [OK] shape {denorm7_out.shape}, round-trip matches")

    logger.info("\n" + "=" * 60)
    logger.info("[OK] All self-tests passed!")
    logger.info("=" * 60)
