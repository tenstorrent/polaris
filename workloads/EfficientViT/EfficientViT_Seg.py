#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os, sys
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.front.functional.tensor_op import *

# -------------------- Small op wrappers (baseline-aligned) --------------------

class AddLayer(SimNN.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name: str = name
        self.add = F.Add(name + ".op")
        super().link_op2module()

    def __call__(self, x: Any, y: Any) -> Any:
        return self.add(x, y)


class UpSampleLayer(SimNN.Module):
    def __init__(self, name: str, factor: int, mode: str = "tile", size: Optional[List[int]] = None) -> None:
        super().__init__()
        self.name: str = name
        self.factor: int = int(factor) if factor is not None else 1
        self.mode: str = str(mode)
        self.size: Optional[List[int]] = list(size) if size is not None else None
        if self.factor > 1:
            ones = np.ones((1, 1, 1, self.factor, 1, self.factor), dtype=np.float32)
            self.k = F._from_data(name + ".ones", ones)  # type: ignore[no-untyped-call]
            self.mul = F.Mul(name + ".mul")
        else:
            self.k = None
            self.mul = None
        super().link_op2module()

    def __call__(self, x: Any) -> Any:
        if self.size is not None:
            Ht, Wt = int(self.size[0]), int(self.size[1])
            if tuple(x.shape[-2:]) == (Ht, Wt):  # type: ignore[index]
                return x
        if self.factor <= 1:
            return x
        B, C, H, W = x.shape  # type: ignore[assignment]
        x6 = x.reshape([B, C, H, 1, W, 1])  # type: ignore[no-untyped-call]
        y6 = self.mul(x6, self.k)  # type: ignore[misc]
        y  = y6.reshape([B, C, H * self.factor, W * self.factor])  # type: ignore[no-untyped-call]
        return y

# -------------------- Core layers and blocks --------------------

class ConvLayer(SimNN.Module):
    def __init__(self, name: str, in_ch: int, out_ch: int, ks: int = 1, stride: int = 1, pad: int = 0,
                 groups: int = 1, activation: Optional[str] = "hswish", bias: bool = False) -> None:
        super().__init__()
        self.name: str = name
        self.ic = int(in_ch); self.oc = int(out_ch)
        self.ks = int(ks); self.stride = int(stride); self.pad = int(pad)
        self.groups = int(groups); self.use_bias = bool(bias)

        self.conv = F.Conv2d(name + ".conv", self.ic, self.oc,
                             kernel_size=self.ks, stride=self.stride, padding=self.pad,
                             dilation=1, groups=self.groups, bias=self.use_bias)
        self.bn = F.BatchNorm2d(name + ".bn", self.oc)

        self.act = None
        if activation not in (None, "none"):
            a = str(activation).lower()
            if a == "gelu":
                self.act = F.Gelu(name + ".act")
            elif a == "relu":
                self.act = F.Relu(name + ".act")
            elif a in ("hswish",):
                self.act = F.Hardswish(name + ".act")
            else:
                self.act = F.Gelu(name + ".act")
            if hasattr(self.act, "set_module"):
                self.act.set_module(self)  # type: ignore[call-arg]

        super().link_op2module()

    def __call__(self, x: Any) -> Any:
        y = self.conv(x)
        y = self.bn(y)
        if self.act is not None:
            y = self.act(y)
        return y

    def analytical_param_count(self, lvl: int = 0) -> int:
        conv_w = self.oc * (self.ic // max(1, self.groups)) * self.ks * self.ks
        conv_b = self.oc if self.use_bias else 0
        bn = 2 * self.oc
        return int(conv_w + conv_b + bn)


class MBConv(SimNN.Module):
    def __init__(self, name: str, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, activation: str = "hswish") -> None:
        super().__init__()
        self.name: str = name
        pad = int(kernel) // 2
        self.use_res: bool = (int(in_ch) == int(out_ch) and int(stride) == 1)
        self.dw = ConvLayer(name + ".dw", in_ch, in_ch, ks=kernel, stride=stride, pad=pad,
                            groups=in_ch, activation=activation)
        self.pw = ConvLayer(name + ".pw", in_ch, out_ch, ks=1, stride=1, pad=0,
                            groups=1, activation=activation)
        self._submodules[self.dw.name] = self.dw
        self._submodules[self.pw.name] = self.pw
        if self.use_res:
            self.add: Optional[AddLayer] = AddLayer(name + ".add")
            self._submodules[self.add.name] = self.add  # type: ignore[index]
        else:
            self.add = None
        super().link_op2module()

    def __call__(self, x: Any) -> Any:
        y = self.pw(self.dw(x))
        return self.add(x, y) if (self.use_res and self.add is not None) else y

    def analytical_param_count(self, lvl: int = 0) -> int:
        return self.dw.analytical_param_count(lvl+1) + self.pw.analytical_param_count(lvl+1)

# -------------------- Backbone --------------------

class EfficientViTBackbone(SimNN.Module):
    def __init__(self, name: str, in_ch: int = 3, stem_out: int = 16, dims: Tuple[int, int, int, int] = (32,64,128,128),
                 blocks: Tuple[int, int, int, int] = (1,2,2,2), activation: str = "hswish") -> None:
        super().__init__()
        self.name: str = name
        self.stem1 = ConvLayer(name + ".stem1", in_ch, stem_out, ks=3, stride=2, pad=1, activation=activation)
        self.stem2 = ConvLayer(name + ".stem2", stem_out, stem_out, ks=3, stride=1, pad=1, activation=activation)
        self._submodules[self.stem1.name] = self.stem1
        self._submodules[self.stem2.name] = self.stem2

        self.stages: List[List[SimNN.Module]] = []
        in_planes = stem_out
        for si, (c_out, n_blk) in enumerate(zip(dims, blocks)):
            stage: List[SimNN.Module] = []
            first_stride = 2
            blk0 = MBConv(f"{name}.s{si}.b0", in_planes, c_out, kernel=3, stride=first_stride, activation=activation)
            stage.append(blk0); self._submodules[blk0.name] = blk0
            in_planes = c_out
            for bi in range(1, n_blk):
                blk = MBConv(f"{name}.s{si}.b{bi}", in_planes, c_out, kernel=3, stride=1, activation=activation)
                stage.append(blk); self._submodules[blk.name] = blk
            self.stages.append(stage)

        super().link_op2module()

    def __call__(self, x: Any) -> Dict[str, Any]:
        feed: Dict[str, Any] = {}
        y = self.stem2(self.stem1(x))      # after stem: OS=2
        for blk in self.stages[0]:
            y = blk(y)
        feed["stage2"] = y                 # OS=4
        for blk in self.stages[1]:
            y = blk(y)
        feed["stage3"] = y                 # OS=8
        for blk in self.stages[2]:
            y = blk(y)
        feed["stage4"] = y                 # OS=16
        for blk in self.stages[3]:
            y = blk(y)
        feed["stage5"] = y                 # OS=32
        return feed

    def create_input_tensors(self, bs: int, c: int, h: int, w: int) -> Dict[str, Any]:
        return {"seg_input": F._from_shape("seg_input", [bs, c, h, w])}  # type: ignore[no-untyped-call]

    def analytical_param_count(self, lvl: int = 0) -> int:
        cnt = 0
        if hasattr(self.stem1, "analytical_param_count"):
            cnt += self.stem1.analytical_param_count(lvl+1)  # type: ignore[misc]
        if hasattr(self.stem2, "analytical_param_count"):
            cnt += self.stem2.analytical_param_count(lvl+1)  # type: ignore[misc]
        for stage in self.stages:
            for blk in stage:
                if hasattr(blk, "analytical_param_count"):
                    cnt += blk.analytical_param_count(lvl+1)  # type: ignore[misc]
        return int(cnt)

# -------------------- Segmentation head --------------------

class SegHead(SimNN.Module):
    def __init__(self,
                 name: str,
                 fid_list: List[str],
                 in_channel_list: List[int],
                 stride_list: List[int],
                 head_stride: int,
                 head_width: int,
                 head_depth: int,
                 expand_ratio: float,
                 middle_op: str,
                 final_expand: Optional[float],
                 n_classes: int,
                 activation: str = "hswish") -> None:
        super().__init__()
        self.name: str = name
        self.fid_list: List[str] = list(fid_list)
        self.stride_list: List[int] = [int(s) for s in stride_list]
        self.head_stride: int = int(head_stride)
        self.head_width: int = int(head_width)
        self.n_classes: int = int(n_classes)
        self.activation: str = activation

        if not isinstance(self.fid_list, (list, tuple)) or len(self.fid_list) == 0:
            raise ValueError(f"{name}.fid_list: list is empty or invalid — Provide feature ids e.g., ['stage4','stage3','stage2']")
        if not isinstance(self.stride_list, (list, tuple)) or len(self.stride_list) == 0:
            raise ValueError(f"{name}.stride_list: list is empty or invalid — Provide strides per feature id")
        if not isinstance(in_channel_list, (list, tuple)) or len(in_channel_list) == 0:
            raise ValueError(f"{name}.in_channel_list: list is empty or invalid — Provide channels per feature id")
        Ls = [len(self.fid_list), len(self.stride_list), len(in_channel_list)]
        if len(set(Ls)) != 1:
            raise ValueError(f"{name}.argcheck: lists must have same length, got lengths={Ls}")

        self.in_adapters: List[ConvLayer] = []
        self.in_ups: List[Optional[int]] = []
        self.in_downs: List[Optional[int]] = []
        self.down_ops: List[Optional[ConvLayer]] = []
        self.add_mods: List[AddLayer] = []
        self.up_mod_cache: Dict[int, UpSampleLayer] = {}

        for i, (cin, s) in enumerate(zip(in_channel_list, self.stride_list)):
            conv = ConvLayer(f"{name}.in{i}", int(cin), self.head_width, ks=1, stride=1, pad=0, activation=None)
            self.in_adapters.append(conv); self._submodules[conv.name] = conv

            s = int(s)
            if s > self.head_stride:
                if (s % self.head_stride) != 0:
                    raise ValueError(f"{name}: non-integer upsample factor from stride {s} to head_stride {self.head_stride}")
                factor_up = s // self.head_stride
                self.in_ups.append(factor_up); self.in_downs.append(None); self.down_ops.append(None)
            elif s < self.head_stride:
                if (self.head_stride % s) != 0:
                    raise ValueError(f"{name}: non-integer downsample factor from stride {s} to head_stride {self.head_stride}")
                factor_down = self.head_stride // s
                self.in_ups.append(None); self.in_downs.append(factor_down)
                dconv = ConvLayer(f"{name}.down{i}", self.head_width, self.head_width, ks=1, stride=factor_down, pad=0, activation=None)
                self.down_ops.append(dconv); self._submodules[dconv.name] = dconv
            else:
                self.in_ups.append(None); self.in_downs.append(None); self.down_ops.append(None)

        if len(self.fid_list) >= 2:
            for i in range(len(self.fid_list) - 1):
                addm = AddLayer(f"{name}.add{i}")
                self.add_mods.append(addm)
                self._submodules[addm.name] = addm

        self.middle_blocks: List[MBConv] = []
        mid_op = str(middle_op).lower()
        if mid_op in ("mbconv", "mb", "res", "fmbconv", "fmb"):
            for i in range(int(head_depth)):
                mb = MBConv(f"{name}.mid.mb{i}", self.head_width, self.head_width, kernel=3, stride=1, activation=self.activation)
                self.middle_blocks.append(mb); self._submodules[mb.name] = mb
        else:
            raise NotImplementedError(f"{name}: middle_op={middle_op}")

        self.final_expand: Optional[ConvLayer] = None

        fe_any: object = final_expand
        fe_ratio: Optional[float]
        if fe_any is None:
            fe_ratio = None
        elif isinstance(fe_any, (int, float)):
            fe_ratio = float(fe_any)
        elif isinstance(fe_any, str):
            fe_str: str = fe_any  # type: ignore[assignment]
            s_str: str = fe_str.strip().lower()
            if s_str in ("none", ""):
                fe_ratio = None
            else:
                try:
                    fe_ratio = float(fe_str)
                except ValueError:
                    raise ValueError(f"{name}: final_expand invalid string '{fe_str}'")
        else:
            raise TypeError(f"{name}: final_expand unsupported type {type(fe_any)}")

        if fe_ratio is not None:
            fe = int(round(self.head_width * fe_ratio))
            self.final_expand = ConvLayer(f"{name}.expand", self.head_width, fe, ks=1, stride=1, pad=0,
                                          activation=self.activation, bias=True)
            self._submodules[self.final_expand.name] = self.final_expand
            out_in = fe
        else:
            out_in = self.head_width

        self.out = F.Conv2d(f"{self.name}.out", int(out_in), self.n_classes, kernel_size=1, stride=1, padding=0, bias=True)

        super().link_op2module()

    def _upsample(self, name: str, x: Any, factor: int) -> Any:
        if factor <= 1:
            return x
        mod = self.up_mod_cache.get(factor, None)
        if mod is None:
            mod = UpSampleLayer(name, factor, mode="tile", size=None)
            self.up_mod_cache[factor] = mod
            self._submodules[mod.name] = mod
            mod.link_op2module()  # type: ignore[no-untyped-call]
        return mod(x)

    def __call__(self, feed: Dict[str, Any]) -> Dict[str, Any]:
        for fid in self.fid_list:
            if fid not in feed:
                raise KeyError(f"{self.name}: required feature '{fid}' not found in backbone feed. Available keys: {list(feed.keys())}")

        xs: List[Any] = []
        for i, fid in enumerate(self.fid_list):
            xi = feed[fid]
            xi = self.in_adapters[i](xi)
            if self.in_downs[i] is not None:
                d = self.down_ops[i]
                if d is not None:
                    xi = d(xi)
            if self.in_ups[i] is not None:
                factor_up_val = self.in_ups[i]
                factor_int: int = int(factor_up_val) if factor_up_val is not None else 1
                up_name: str = f"{self.name}.up{i}"
                xi = self._upsample(up_name, xi, factor_int)
            xs.append(xi)

        y = xs[0]
        for i in range(1, len(xs)):
            y = self.add_mods[i - 1](y, xs[i])

        for mb in self.middle_blocks:
            y = mb(y)

        if self.final_expand is not None:
            y = self.final_expand(y)
        y = self.out(y)
        return {"segout": y}

    def analytical_param_count(self, lvl: int = 0) -> int:
        cnt = 0
        for ca in self.in_adapters:
            cnt += ca.analytical_param_count(lvl+1)
        for d in self.down_ops:
            if d is not None and hasattr(d, "analytical_param_count"):
                cnt += d.analytical_param_count(lvl+1)  # type: ignore[misc]
        for mb in self.middle_blocks:
            cnt += mb.analytical_param_count(lvl+1)
        if self.final_expand is not None:
            cnt += self.final_expand.analytical_param_count(lvl+1)
            out_in = self.final_expand.oc
        else:
            out_in = self.head_width
        cnt += (out_in * self.n_classes) + self.n_classes
        return int(cnt)

# -------------------- Top model (Seg wrapper) --------------------

class EfficientViT_Seg(SimNN.Module):
    def __init__(self, name: str, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.name: str = name
        c: Dict[str, Any] = dict(cfg)

        self.bs: int  = int(c.get("bs", 1))
        self.in_c: int = int(c.get("img_channels", 3))
        self.H: int    = int(c.get("img_height", 512))
        self.W: int    = int(c.get("img_width", self.H))

        # BEGIN minimal mypy fix: enforce exact 4-tuples for dims and blocks
        dims_list = [int(x) for x in c.get("dims", [32, 64, 128, 128])]
        if len(dims_list) != 4:
            raise ValueError(f"cfg.dims must have length 4, got {len(dims_list)}: {dims_list}")
        dims_t: Tuple[int, int, int, int] = (dims_list[0], dims_list[1], dims_list[2], dims_list[3])

        blocks_list = [int(x) for x in c.get("blocks", [1, 2, 2, 2])]
        if len(blocks_list) != 4:
            raise ValueError(f"cfg.blocks must have length 4, got {len(blocks_list)}: {blocks_list}")
        blocks_t: Tuple[int, int, int, int] = (blocks_list[0], blocks_list[1], blocks_list[2], blocks_list[3])
        # END minimal mypy fix

        stem_out = int(c.get("stem_out", 16))
        activation = str(c.get("activation", "hswish"))

        self.backbone = EfficientViTBackbone("seg.backbone",
                                             in_ch=self.in_c, stem_out=stem_out,
                                             dims=dims_t, blocks=blocks_t,
                                             activation=activation)
        self._submodules[self.backbone.name] = self.backbone

        fid_list = list(c.get("fid_list", ["stage4", "stage3", "stage2"]))
        if not isinstance(fid_list, (list, tuple)) or len(fid_list) == 0:
            raise ValueError("cfg.fid_list: Valid options include 'stage2','stage3','stage4','stage5'")

        in_channel_list_cfg = c.get("in_channel_list", "auto")
        if in_channel_list_cfg == "auto" or in_channel_list_cfg is None:
            stage2, stage3, stage4, stage5 = dims_list[0], dims_list[1], dims_list[2], dims_list[3]
            chan_map: Dict[str, int] = {"stage2": stage2, "stage3": stage3, "stage4": stage4, "stage5": stage5}
            try:
                in_channel_list = [int(chan_map[f]) for f in fid_list]
            except KeyError as e:
                raise KeyError(f"in_channel_list auto-resolve failed: unknown fid '{e.args[0]}'. "
                               f"Expected one of {list(chan_map.keys())}, got {fid_list}")
            if len(in_channel_list) == 0:
                raise ValueError("in_channel_list(auto): Resolved empty channel list from dims/fid_list")
        else:
            in_channel_list = [int(x) for x in in_channel_list_cfg]  # type: ignore[assignment]
            if len(in_channel_list) == 0:
                raise ValueError("cfg.in_channel_list: Provide non-empty list or use 'auto'")

        stride_list_cfg = c.get("stride_list", "auto")
        if stride_list_cfg == "auto" or stride_list_cfg is None:
            stride_map: Dict[str, int] = {"stage2": 4, "stage3": 8, "stage4": 16, "stage5": 32}
            try:
                stride_list = [int(stride_map[f]) for f in fid_list]
            except KeyError as e:
                raise KeyError(f"stride_list auto-resolve failed: unknown fid '{e.args[0]}'. "
                               f"Expected one of {list(stride_map.keys())}, got {fid_list}")
            if len(stride_list) == 0:
                raise ValueError("stride_list(auto): Resolved empty stride list from fid_list")
        else:
            stride_list = [int(x) for x in stride_list_cfg]  # type: ignore[assignment]
            if len(stride_list) == 0:
                raise ValueError("cfg.stride_list: Provide non-empty list or use 'auto'")

        if len(set([len(fid_list), len(in_channel_list), len(stride_list)])) != 1:
            raise ValueError(f"cfg(list lengths): lists must have same length, got lengths={[len(fid_list), len(in_channel_list), len(stride_list)]}")

        head_stride   = int(c.get("head_stride", 8))
        head_width    = int(c.get("head_width", 32))
        head_depth    = int(c.get("head_depth", 1))
        expand_ratio  = float(c.get("expand_ratio", 4))
        middle_op     = str(c.get("middle_op", "mbconv"))
        final_expand  = c.get("final_expand", 4)
        if isinstance(final_expand, str) and final_expand.strip().lower() in ("none", ""):
            final_expand = None
        n_classes     = int(c.get("n_classes", 19))

        self.head = SegHead("seg.head",
                            fid_list=fid_list,
                            in_channel_list=in_channel_list,
                            stride_list=stride_list,
                            head_stride=head_stride,
                            head_width=head_width,
                            head_depth=head_depth,
                            expand_ratio=expand_ratio,
                            middle_op=middle_op,
                            final_expand=final_expand,  # type: ignore[arg-type]
                            n_classes=n_classes,
                            activation=activation)
        self._submodules[self.head.name] = self.head

        self.training: bool = False
        super().link_op2module()

    def create_input_tensors(self) -> None:
        self.input_tensors = self.backbone.create_input_tensors(self.bs, self.in_c, self.H, self.W)

    def __call__(self) -> Any:
        x = self.input_tensors["seg_input"]
        feed = self.backbone(x)
        outd = self.head(feed)
        return outd["segout"]

    def get_forward_graph(self) -> Any:
        return super()._get_forward_graph(self.input_tensors)

    def analytical_param_count(self, lvl: int = 0) -> int:
        return int(self.backbone.analytical_param_count(lvl+1) + self.head.analytical_param_count(lvl+1))

EFFICIENTVIT_SEG = EfficientViT_Seg

# -------------------- Standalone smoke test --------------------

if __name__ == "__main__":
    cfg: Dict[str, Any] = {
        "img_height": 512, "img_width": 512, "img_channels": 3, "bs": 1,
        "dims": [32, 64, 128, 128], "blocks": [1, 2, 2, 2], "stem_out": 16, "activation": "hswish",
        "fid_list": ["stage4", "stage3", "stage2"],
        "in_channel_list": "auto",
        "stride_list": "auto",
        "head_stride": 8, "head_width": 32, "head_depth": 1,
        "expand_ratio": 4, "middle_op": "mbconv", "final_expand": 4,
        "n_classes": 19,
    }
    model = EfficientViT_Seg("effvit_seg_test", cfg)
    model.create_input_tensors()
    y = model()
    print("Output shape:", y.shape)
    print(f"param_count: {model.analytical_param_count():,d}")
    gg = model.get_forward_graph()
    gg.graph2onnx("efficientvit_seg.onnx", do_model_check=False)
