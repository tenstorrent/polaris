#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os, sys
import numpy as np
from typing import Any, Dict, List, Optional

# Ensure repo root on path if needed
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.tensor_op import *

class Conv2d_BN(SimNN.Module):
    def __init__(self, name: str, a: int, b: int, ks: int = 1, stride: int = 1, pad: int = 0,
                 dilation: int = 1, groups: int = 1, resolution: int = -10000):
        super().__init__()
        self.name = name
        self.in_ch = int(a)
        self.out_ch = int(b)
        self.ks = int(ks)
        self.conv = F.Conv2d(name + '.conv', a, b, kernel_size=ks, stride=stride,
                             padding=pad, dilation=dilation, groups=groups, bias=False)
        self.bn = F.BatchNorm2d(name + '.bn', b)
        super().link_op2module()

    def __call__(self, x):
        return self.bn(self.conv(x))

    def analytical_param_count(self, lvl=0):
        return self.in_ch * self.out_ch * self.ks * self.ks + 2 * self.out_ch


class Linear_BN(SimNN.Module):
    def __init__(self, name: str, a: int, b: int, resolution: int = -100000, bn_weight_init: Optional[float] = None):
        super().__init__()
        self.name = name
        self.in_features = int(a)
        self.out_features = int(b)
        self.lin = F.Linear(name + '.lin', a, b, bias=False)
        self.bn2d = F.BatchNorm2d(name + '.bn', b)
        if bn_weight_init is not None and hasattr(self.bn2d, "set_weight_init"):
            self.bn2d.set_weight_init(bn_weight_init)
        super().link_op2module()

    def __call__(self, x):
        B, N, _ = x.shape
        x = self.lin(x)
        outshape = x.shape
        x = x.reshape([B * N, x.shape[-1], 1, 1])
        x = self.bn2d(x)
        return x.reshape([B, N, outshape[-1]])

    def analytical_param_count(self, lvl=0):
        return self.in_features * self.out_features + 2 * self.out_features


class BN_Linear(SimNN.Module):
    def __init__(self, name: str, a: int, b: int, bias: bool = True):
        super().__init__()
        self.name = name
        self.bn_features = int(a)
        self.in_features = int(a)
        self.out_features = int(b)
        self.use_bias = bool(bias)
        self.bn2d = F.BatchNorm2d(name + '.bn', a)
        self.lin = F.Linear(name + '.lin', a, b, bias=bias)
        super().link_op2module()

    def __call__(self, x):
        if len(x.shape) == 3:
            B, N, C = x.shape
            assert N == 1, "BN_Linear expects [B,C] or [B,1,C]"
            x = x.reshape([B, C])
        B, C = x.shape
        x = x.reshape([B, C, 1, 1])
        x = self.bn2d(x).reshape([B, C])
        return self.lin(x)

    def analytical_param_count(self, lvl=0):
        return 2 * self.bn_features + self.in_features * self.out_features + (self.out_features if self.use_bias else 0)


def b16(name: str, n: int, resolution: int = 224) -> SimNN.Module:
    class _Patch(SimNN.Module):
        def __init__(self):
            super().__init__()
            self.name = name
            self.c1 = Conv2d_BN(f'{name}.c1', 3, n // 8, 3, 2, 1, resolution=resolution)
            self.a1 = F.Hardswish(f'{name}.a1'); self.a1.set_module(self)
            self.c2 = Conv2d_BN(f'{name}.c2', n // 8, n // 4, 3, 2, 1, resolution=resolution // 2)
            self.a2 = F.Hardswish(f'{name}.a2'); self.a2.set_module(self)
            self.c3 = Conv2d_BN(f'{name}.c3', n // 4, n // 2, 3, 2, 1, resolution=resolution // 4)
            self.a3 = F.Hardswish(f'{name}.a3'); self.a3.set_module(self)
            self.c4 = Conv2d_BN(f'{name}.c4', n // 2, n, 3, 2, 1, resolution=resolution // 8)
            self._submodules[self.c1.name] = self.c1
            self._submodules[self.c2.name] = self.c2
            self._submodules[self.c3.name] = self.c3
            self._submodules[self.c4.name] = self.c4
            super().link_op2module()

        def __call__(self, x):
            y = self.c1(x); y = self.a1(y)
            y = self.c2(y); y = self.a2(y)
            y = self.c3(y); y = self.a3(y)
            y = self.c4(y)
            return y

        def analytical_param_count(self, lvl=0):
            return (self.c1.analytical_param_count(lvl + 1) +
                    self.c2.analytical_param_count(lvl + 1) +
                    self.c3.analytical_param_count(lvl + 1) +
                    self.c4.analytical_param_count(lvl + 1))
    return _Patch()


class Residual(SimNN.Module):
    def __init__(self, name: str, m: Any, drop: float):
        super().__init__()
        self.name = name
        self.drop = float(drop)
        self.training = False

        self.is_seq3 = isinstance(m, tuple) and len(m) == 3
        if self.is_seq3:
            fc1, act, fc2 = m
            self.fc1 = fc1
            self.act = act
            self.fc2 = fc2
            self._submodules[self.fc1.name] = self.fc1
            self._submodules[self.fc2.name] = self.fc2
        else:
            self.m = m
            self._submodules[name + '.m'] = m

        self.add = F.Add(name + '.add')
        self.mul = F.Mul(name + '.mul')
        super().link_op2module()

    def _apply_m(self, x):
        if self.is_seq3:
            return self.fc2(self.act(self.fc1(x)))
        return self.m(x)

    def __call__(self, x):
        if self.training and self.drop > 0.0:
            inv_keep = F._from_data(self.name + '.invkeep', np.array(1.0 / (1.0 - self.drop), dtype=np.float32))
            self._tensors[inv_keep.name] = inv_keep
            return self.add(x, self.mul(self._apply_m(x), inv_keep))
        return self.add(x, self._apply_m(x))

    def analytical_param_count(self, lvl=0):
        if self.is_seq3:
            c1 = self.fc1.analytical_param_count(lvl + 1) if hasattr(self.fc1, 'analytical_param_count') else 0
            c2 = self.fc2.analytical_param_count(lvl + 1) if hasattr(self.fc2, 'analytical_param_count') else 0
            return c1 + c2
        return self.m.analytical_param_count(lvl + 1) if hasattr(self.m, 'analytical_param_count') else 0


class Attention(SimNN.Module):
    def __init__(self, name: str, dim: int, key_dim: int, num_heads: int = 8,
                 attn_ratio: float = 4.0, activation: Optional[str] = "silu",
                 resolution: int = 14):
        super().__init__()
        self.name = name
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * self.num_heads
        self.nh_kd = self.key_dim * self.num_heads
        self.scale_val = np.array(self.key_dim ** -0.5, dtype=np.float32)

        # qkv and output projection
        self.qkv  = Linear_BN(name + '.qkv', dim, self.dh + 2 * self.nh_kd, resolution=resolution)
        self.proj = Linear_BN(name + '.proj', self.dh, dim, resolution=resolution, bn_weight_init=0.0)

        # core ops
        self.softmax = F.Softmax(name + '.softmax', axis=-1)
        self.matmul_qk = F.MatMul(name + '.matmul_qk')
        self.matmul_av = F.MatMul(name + '.matmul_av')
        self.scale_mul = F.Mul(name + '.scale')
        self.transpose_kt = F.Transpose(name + '.transpose_kt', perm=[0, 1, 3, 2])
        self.transpose_out = F.Transpose(name + '.transpose_out', perm=[0, 2, 1, 3])

        # gather and reshape/transpose
        self.gather_q = F.Gather(name + '.gq', axis=2)
        self.gather_k = F.Gather(name + '.gk', axis=2)
        self.gather_v = F.Gather(name + '.gv', axis=2)
        self.reshape_q = F.Reshape(name + '.q.R'); self.Tq = F.Transpose(name + '.q.T', perm=[0, 2, 1, 3])
        self.reshape_k = F.Reshape(name + '.k.R'); self.Tk = F.Transpose(name + '.k.T', perm=[0, 2, 1, 3])
        self.reshape_v = F.Reshape(name + '.v.R'); self.Tv = F.Transpose(name + '.v.T', perm=[0, 2, 1, 3])
        self.reshape_out = F.Reshape(name + '.out.R')

        self.act = F.Hardswish(name + '.act'); self.act.set_module(self)
        super().link_op2module()

    def __call__(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x)

        # split ranges
        q_w = k_w = self.nh_kd; v_w = self.dh
        q_idx = F._from_data(self.name + '.q.idx', np.arange(0, q_w, dtype=np.int64))
        k_idx = F._from_data(self.name + '.k.idx', np.arange(q_w, q_w + k_w, dtype=np.int64))
        v_idx = F._from_data(self.name + '.v.idx', np.arange(q_w + k_w, q_w + k_w + v_w, dtype=np.int64))
        self._tensors[q_idx.name] = q_idx; self._tensors[k_idx.name] = k_idx; self._tensors[v_idx.name] = v_idx

        # gather q,k,v
        q = self.gather_q(qkv, q_idx); k = self.gather_k(qkv, k_idx); v = self.gather_v(qkv, v_idx)

        # [B,N,H,·] -> [B,H,N,·]
        shp_q = F._from_data(self.name + '.q.shp', np.array([B, N, self.num_heads, self.key_dim], dtype=np.int64))
        shp_k = F._from_data(self.name + '.k.shp', np.array([B, N, self.num_heads, self.key_dim], dtype=np.int64))
        shp_v = F._from_data(self.name + '.v.shp', np.array([B, N, self.num_heads, self.d], dtype=np.int64))
        for s in (shp_q, shp_k, shp_v): self._tensors[s.name] = s
        q = self.Tq(self.reshape_q(q, shp_q)); k = self.Tk(self.reshape_k(k, shp_k)); v = self.Tv(self.reshape_v(v, shp_v))

        # attention
        kt = self.transpose_kt(k)
        scale = F._from_data(self.name + '.scale.val', self.scale_val); self._tensors[scale.name] = scale
        attn = self.softmax(self.scale_mul(self.matmul_qk(q, kt), scale))

        # output
        y = self.matmul_av(attn, v)
        y = self.transpose_out(y)
        shp_out = F._from_data(self.name + '.out.shp', np.array([B, y.shape[1], self.dh], dtype=np.int64)); self._tensors[shp_out.name] = shp_out
        y = self.reshape_out(y, shp_out)
        y = self.act(y)
        return self.proj(y)

    def analytical_param_count(self, lvl=0):
        return self.qkv.analytical_param_count(lvl + 1) + self.proj.analytical_param_count(lvl + 1)


class Subsample(SimNN.Module):
    def __init__(self, name: str, stride: int, resolution: int):
        super().__init__()
        self.name = name
        self.stride = int(stride)
        self.resolution = int(resolution)
        self.reshape4 = F.Reshape(name + '.to4')
        self.reshape3 = F.Reshape(name + '.to3')
        self.gather_row = F.Gather(name + '.grow', axis=1)
        self.gather_col = F.Gather(name + '.gcol', axis=2)
        super().link_op2module()

    def __call__(self, x):
        B, N, C = x.shape
        R = self.resolution
        s4 = F._from_data(self.name + '.s4', np.array([B, R, R, C], dtype=np.int64)); self._tensors[s4.name] = s4
        t = self.reshape4(x, s4)

        rows = F._from_data(self.name + '.rows', np.arange(0, R, self.stride, dtype=np.int64))
        cols = F._from_data(self.name + '.cols', np.arange(0, R, self.stride, dtype=np.int64))
        self._tensors[rows.name] = rows; self._tensors[cols.name] = cols

        t = self.gather_row(t, rows)
        t = self.gather_col(t, cols)

        s3 = F._from_data(self.name + '.s3', np.array([B, -1, C], dtype=np.int64)); self._tensors[s3.name] = s3
        return self.reshape3(t, s3)

    def analytical_param_count(self, lvl=0):
        return 0


class AttentionSubsample(SimNN.Module):
    def __init__(self, name: str, in_dim: int, out_dim: int, key_dim: int, num_heads: int = 8,
        attn_ratio: float = 2.0, activation: Optional[str] = "silu", stride: int = 2,
        resolution: int = 14, resolution_: int = 7):
        super().__init__()
        self.name = name
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * self.num_heads
        self.scale_val = np.array(self.key_dim ** -0.5, dtype=np.float32)
        self.resolution_ = int(resolution_)
        self.resolution_2 = self.resolution_ * self.resolution_

        self.kv   = Linear_BN(name + '.kv', in_dim, self.num_heads * self.key_dim + self.dh, resolution=resolution)
        self.qsub = Subsample(name + '.q.sub', stride, resolution)
        self.qfc  = Linear_BN(name + '.q.fc', in_dim, self.num_heads * self.key_dim, resolution=resolution_)
        self.proj = Linear_BN(name + '.proj', self.dh, out_dim, resolution=resolution_, bn_weight_init=0.0)

        self.softmax = F.Softmax(name + '.softmax', axis=-1)
        self.matmul_qk = F.MatMul(name + '.matmul_qk')
        self.matmul_av = F.MatMul(name + '.matmul_av')
        self.scale_mul = F.Mul(name + '.scale')
        self.transpose_kt = F.Transpose(name + '.transpose_kt', perm=[0, 1, 3, 2])
        self.transpose_out = F.Transpose(name + '.transpose_out', perm=[0, 2, 1, 3])

        self.gather_k = F.Gather(name + '.gk', axis=2)
        self.gather_v = F.Gather(name + '.gv', axis=2)
        self.reshape_k = F.Reshape(name + '.k.R'); self.Tk = F.Transpose(name + '.k.T', perm=[0, 2, 1, 3])
        self.reshape_v = F.Reshape(name + '.v.R'); self.Tv = F.Transpose(name + '.v.T', perm=[0, 2, 1, 3])
        self.reshape_q = F.Reshape(name + '.q.R'); self.Tq = F.Transpose(name + '.q.T', perm=[0, 2, 1, 3])
        self.reshape_out = F.Reshape(name + '.out.R')

        self.act = F.Hardswish(name + '.act'); self.act.set_module(self)
        super().link_op2module()

    def __call__(self, x):
        B, N, _ = x.shape
        kv = self.kv(x)

        # split kv -> k, v
        k_w = self.num_heads * self.key_dim; v_w = self.dh
        k_idx = F._from_data(self.name + '.k.idx', np.arange(0, k_w, dtype=np.int64))
        v_idx = F._from_data(self.name + '.v.idx', np.arange(k_w, k_w + v_w, dtype=np.int64))
        self._tensors[k_idx.name] = k_idx; self._tensors[v_idx.name] = v_idx
        k = self.gather_k(kv, k_idx); v = self.gather_v(kv, v_idx)

        shp_k = F._from_data(self.name + '.k.shp', np.array([B, N, self.num_heads, self.key_dim], dtype=np.int64))
        shp_v = F._from_data(self.name + '.v.shp', np.array([B, N, self.num_heads, self.d], dtype=np.int64))
        self._tensors[shp_k.name] = shp_k; self._tensors[shp_v.name] = shp_v
        k = self.Tk(self.reshape_k(k, shp_k)); v = self.Tv(self.reshape_v(v, shp_v))

        # q path
        q = self.qfc(self.qsub(x))
        shp_q = F._from_data(self.name + '.q.shp', np.array([B, self.resolution_2, self.num_heads, self.key_dim], dtype=np.int64))
        self._tensors[shp_q.name] = shp_q
        q = self.Tq(self.reshape_q(q, shp_q))

        # attention
        kt = self.transpose_kt(k)
        scale = F._from_data(self.name + '.scale.val', self.scale_val); self._tensors[scale.name] = scale
        attn = self.softmax(self.scale_mul(self.matmul_qk(q, kt), scale))

        # output
        y = self.matmul_av(attn, v)
        y = self.transpose_out(y)
        shp_out = F._from_data(self.name + '.out.shp', np.array([B, y.shape[1], self.dh], dtype=np.int64)); self._tensors[shp_out.name] = shp_out
        y = self.reshape_out(y, shp_out)
        y = self.act(y)
        return self.proj(y)

    def analytical_param_count(self, lvl=0):
        return self.kv.analytical_param_count(lvl + 1) + self.qfc.analytical_param_count(lvl + 1) + self.proj.analytical_param_count(lvl + 1)


class LeViT(SimNN.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        c = dict(cfg)

        img_h = int(c.get("img_height", 224))
        img_w = int(c.get("img_width", img_h))
        patch = int(c.get("patch_size", 16))
        in_ch = int(c.get("img_channels", 3))
        num_classes = int(c.get("num_classes", 1000))
        embed_dim = [int(x) for x in c.get("dims", [192])]
        key_dim = [int(x) for x in c.get("key_dim", [64])]
        depth = [int(x) for x in c.get("depths", [12])]
        heads = [int(x) for x in c.get("heads", [3])]
        attn_ratio = [float(x) for x in c.get("attn_ratio", [2])]
        mlp_ratio = [float(x) for x in c.get("mlp_ratio", [2])]
        down_ops = list(c.get("down_ops", []))
        distillation = bool(c.get("distillation", True))
        drop_path = float(c.get("drop_path", 0.0))
        bs = int(c.get("bs", 1))

        self.bs = bs
        self.in_channels = in_ch
        self.in_height = img_h
        self.in_width = img_w
        self.num_classes = num_classes
        self.distillation = distillation

        # Patch stem (CNN) aligns with levit.py via b16 backbone
        self.patch = b16('levit.patch', n=embed_dim[0], resolution=self.in_height)
        self._submodules['levit.patch'] = self.patch

        # Build blocks sequence, mirroring levit.py’s loop structure
        self.blocks: List[SimNN.Module] = []
        resolution = img_h // patch
        down_ops = list(down_ops) + [['']]  # sentinel

        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(zip(embed_dim, key_dim, depth, heads, attn_ratio, mlp_ratio, down_ops)):
            for bi in range(dpth):
                attn = Attention(f'levit.attn_s{i}b{bi}', ed, kd, nh, ar, 'silu', resolution)
                self._submodules[attn.name] = attn
                blk_attn = Residual(f'levit.blk_attn_s{i}b{bi}', attn, drop_path)
                self.blocks.append(blk_attn); self._submodules[blk_attn.name] = blk_attn

                if mr > 0.0:
                    h = int(ed * mr)
                    scope = f'levit.mlp_s{i}b{bi}'
                    fc1 = Linear_BN(f'{scope}.fc1', ed, h, resolution)
                    act = F.Hardswish(f'{scope}.act'); act.set_module(self)
                    fc2 = Linear_BN(f'{scope}.fc2', h, ed, resolution, bn_weight_init=0.0)
                    self._submodules[fc1.name] = fc1; self._submodules[fc2.name] = fc2
                    blk_mlp = Residual(f'levit.blk_mlp_s{i}b{bi}', (fc1, act, fc2), drop_path)
                    self.blocks.append(blk_mlp); self._submodules[blk_mlp.name] = blk_mlp

            if isinstance(do, list) and len(do) > 0 and do[0] == 'Subsample':
                stride = int(do[5])
                resolution_ = (resolution - 1) // stride + 1
                ds = AttentionSubsample(f'levit.down_s{i}', embed_dim[i], embed_dim[i + 1],
                                        key_dim=do[1], num_heads=do[2], attn_ratio=do[3],
                                        activation='silu', stride=stride,
                                        resolution=resolution, resolution_=resolution_)
                self.blocks.append(ds); self._submodules[ds.name] = ds
                resolution = resolution_

                if do[4] > 0:
                    h = int(embed_dim[i + 1] * do[4])
                    scope = f'levit.post_s{i}'
                    fc1 = Linear_BN(f'{scope}.fc1', embed_dim[i + 1], h, resolution)
                    act = F.Hardswish(f'{scope}.act'); act.set_module(self)
                    fc2 = Linear_BN(f'{scope}.fc2', h, embed_dim[i + 1], resolution, bn_weight_init=0.0)
                    self._submodules[fc1.name] = fc1; self._submodules[fc2.name] = fc2
                    blk_post = Residual(f'levit.blk_post_s{i}', (fc1, act, fc2), drop_path)
                    self.blocks.append(blk_post); self._submodules[blk_post.name] = blk_post

        # Avg over tokens (N), same as x.mean(1) in levit.py
        self.avg_sum = F.ReduceSum('levit.avg.sum', axis=1)
        self.avg_mul = F.Mul('levit.avg.mul')

        # Classifier heads
        self.head = BN_Linear('levit.head', embed_dim[-1], num_classes) if num_classes > 0 else (lambda t: t)
        if isinstance(self.head, SimNN.Module): self._submodules['levit.head'] = self.head
        self.head_dist = BN_Linear('levit.head_dist', embed_dim[-1], num_classes) if (num_classes > 0 and self.distillation) else None
        if self.head_dist is not None: self._submodules['levit.head_dist'] = self.head_dist
        self.head_add = F.Add('levit.head.add')
        self.mul_head = F.Mul('levit.head.mul')

        # Stem reshape to [B,N,C], transpose to match levit.py path
        self.R3 = F.Reshape('levit.R3')
        self.T = F.Transpose('levit.T', perm=[0, 2, 1])

        self.training = False
        super().link_op2module()

    def set_batch_size(self, new_bs: int):
        self.bs = int(new_bs)

    def create_input_tensors(self):
        self.input_tensors = {'levit_input': F._from_shape('levit_input', [self.bs, self.in_channels, self.in_height, self.in_width])}

    def __call__(self):
        x = self.input_tensors['levit_input']
        y = self.patch(x)  # [B,Cs,Hs,Ws]
        B, Cs, Hs, Ws = y.shape
        N = Hs * Ws

        # Flatten(2).transpose(1,2) -> [B,N,C]
        s3 = F._from_data('levit.s3', np.array([B, Cs, N], dtype=np.int64)); self._tensors[s3.name] = s3
        y = self.T(self.R3(y, s3))

        # Sequential blocks
        z = y
        for blk in self.blocks:
            z = blk(z)

        # Mean over tokens (dim=1)
        N_live = z.shape[1]
        invN = F._from_data('levit.avg.invN', np.array(1.0 / float(N_live), dtype=np.float32)); self._tensors[invN.name] = invN
        z = self.avg_mul(self.avg_sum(z), invN)

        # Heads (distillation optional)
        if self.head_dist is not None and self.num_classes > 0 and self.distillation:
            y1 = self.head(z) if not callable(self.head) else self.head(z)
            y2 = self.head_dist(z)
            half = F._from_data('levit.half', np.array(0.5, dtype=np.float32)); self._tensors[half.name] = half
            out = self.mul_head(self.head_add(y1, y2), half)
        else:
            out = self.head(z) if not callable(self.head) else self.head(z)
        return out

    def get_forward_graph(self):
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self, lvl=0):
        cnt = 0

        # Patch
        patch = self.patch
        if hasattr(patch, "analytical_param_count"):
            cnt += patch.analytical_param_count(lvl + 1)

        # Blocks
        for blk in self.blocks:
            if hasattr(blk, "analytical_param_count"):
                cnt += blk.analytical_param_count(lvl + 1)

        # Head: either BN_Linear or a lambda
        head = self.head
        if hasattr(head, "analytical_param_count"):
            cnt += head.analytical_param_count(lvl + 1)

        # Head dist: Optional[BN_Linear]
        head_dist = self.head_dist
        if head_dist is not None:
            cnt += head_dist.analytical_param_count(lvl + 1)

        return cnt


# Legacy alias for YAML
LEVIT = LeViT


if __name__ == "__main__":
    # Initialize WL->Arch mapping from YAML (robust path regardless of CWD)
    from pathlib import Path
    from ttsim.config.wl2archmap import get_wlmapspec_from_yaml

    cfg_yaml = Path(__file__).resolve().parents[2] / "config" / "wl2archmapping.yaml"
    print(f"Using wl2archmapping: {cfg_yaml}")
    get_wlmapspec_from_yaml(str(cfg_yaml))

    # Model config
    cfg = {
        "img_channels": 3,
        "img_height": 224,
        "img_width": 224,
        "patch_size": 16,
        "dims": [128, 256, 384],
        "key_dim": [16, 16, 16],
        "depths": [4, 4, 4],
        "heads": [4, 8, 12],
        "attn_ratio": [2, 2, 2],
        "mlp_ratio": [2, 2, 2],
        "down_ops": [
            ["Subsample", 16, 8, 4, 2, 2],
            ["Subsample", 16, 16, 4, 2, 2],
        ],
        "num_classes": 1000,
        "distillation": True,
        "drop_path": 0.0,
        "bs": 1,
    }

    # Build and run
    model = LeViT("test_levit", cfg)
    model.set_batch_size(1)
    model.create_input_tensors()
    out = model()
    print("Output shape:", out.shape)
    print("Analytical parameter count:", model.analytical_param_count())
    gg = model.get_forward_graph()
    print("Dumping ONNX Graph to levit.onnx")
    gg.graph2onnx("levit.onnx", do_model_check=False)
