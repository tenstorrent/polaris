#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Authors: HARMAN Connected Services, Inc.


"""YOLO Model - TTSim conversion"""

import os, sys

# Add polaris root to path (need to go up 3 levels: Yolo -> ScaledYolov4 -> workloads -> polaris)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import math
from copy import deepcopy
from pathlib import Path
import numpy as np
import yaml
from loguru import logger

import typing
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from workloads.ScaledYOLOv4.models.common import (
    Conv,
    Concat,
    SPP,
    SPPCSP,
    Focus,
    VoVCSP,
    MP,
    Bottleneck,
    BottleneckCSP,
    BottleneckCSP2,
    DWConv,
    Upsample,
)

# Import experimental modules
from workloads.ScaledYOLOv4.models.experimental import (
    CrossConv,
    C3,
    MixConv2d,
    Sum,
    GhostConv,
    GhostBottleneck,
)
from workloads.ScaledYOLOv4.models.utils import general
from workloads.ScaledYOLOv4.models.utils import utils


class Detect(SimNN.Module):
    """YOLO Detection Layer - TTSim version

    Converts feature maps from multiple scales into detection outputs.
    Matches YOLOv7 pattern: all ops created in __init__ and stored in SimOpHandleList.
    """

    def __init__(self, name, nc=80, anchors=(), ch=()):
        """
        Args:
            name: Module name
            nc: Number of classes
            anchors: Anchor boxes [nl, na, 2] where nl=num layers, na=num anchors
            ch: Input channels for each detection layer
        """
        super().__init__()
        self.name = name
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (x,y,w,h,obj,cls1,...,clsN)
        self.nl = len(anchors)  # number of detection layers (3 for yolov4)
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.ch = ch  # input channels for each detection layer
        self.in_channels = ch
        self.training = True  # training mode flag
        self.export = False  # onnx export flag

        # Store anchors as const tensors
        self.anchors_np = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.anchor_grid_np = self.anchors_np.reshape(self.nl, 1, -1, 1, 1, 2)

        # Create all ops in __init__ following YOLOv7 pattern
        # For each detection layer: Conv2d, Reshape, Transpose
        oplist = []
        for i, x in enumerate(ch):
            oplist.append(F.Conv2d(name + f".m{i}", x, self.no * self.na, 1))
            oplist.append(F.Reshape(name + f".reshape_{i}"))
            oplist.append(F.Transpose(name + f".transpose_{i}", perm=[0, 1, 3, 4, 2]))
        self.m = F.SimOpHandleList(oplist)

        super().link_op2module()

    def analytical_param_count(self, lvl):
        # Conv2d params: in_channels * out_channels + bias
        return sum([(x + 1) * self.no * self.na for x in self.ch])

    def __call__(self, x):
        """Forward pass through detection layer

        Args:
            x: List of feature maps from different scales
                [tensor1(bs,c1,h1,w1), tensor2(bs,c2,h2,w2), tensor3(bs,c3,h3,w3)]

        Returns:
            Training mode: List of detection outputs (raw)
        """
        self.training |= self.export

        for i in range(self.nl):
            # Apply conv: (bs, ch_in, ny, nx) -> (bs, na*no, ny, nx)
            x[i] = self.m[3 * i](x[i])  # conv
            bs, _, ny, nx = x[i].shape

            # Create shape tensor for reshape
            _tmp_x = F._from_data(
                f"tmp_x_{i}",
                is_const=True,
                data=np.array([bs, self.na, self.no, ny, nx], dtype=np.int64),
            )
            _tmp_x.op_in.append(self.m[3 * i + 1].name)
            self._tensors[_tmp_x.name] = _tmp_x

            # Reshape: (bs, na*no, ny, nx) -> (bs, na, no, ny, nx)
            x[i] = self.m[3 * i + 1](x[i], _tmp_x)

            # Transpose: (bs, na, no, ny, nx) -> (bs, na, ny, nx, no)
            x[i] = self.m[3 * i + 2](x[i])

            if not self.training:  # inference
                # TODO: implement inference mode if needed
                pass

        out = x
        return out


class Model(SimNN.Module):
    """YOLO Model - TTSim version

    Builds the complete YOLO model from a YAML configuration file.
    """

    def __init__(self, name="yolo", cfg=None, ch=3, nc=None):
        """
        Args:
            name: Model name
            cfg: Path to YAML config file or dict
            ch: Number of input channels
            nc: Number of classes (overrides config if provided)
        """
        super().__init__()
        self.name = name
        self.bs = cfg.get("bs", 1)
        self.yaml_cfg_path = cfg.get("yaml_cfg_path", None)
        self.in_channels = cfg.get("in_channels", 3)
        self.in_resolution = cfg.get("in_resolution", 640)
        self.nc = cfg.get("nc", None)

        # Load configuration
        if self.yaml_cfg_path:
            from ttsim.utils.common import parse_yaml

            yaml_cfg = parse_yaml(self.yaml_cfg_path)
            self.yaml = yaml_cfg
            if self.nc is not None:
                self.yaml["nc"] = self.nc
        else:
            self.yaml_file = Path(cfg).name if not isinstance(cfg, dict) else None
            if not isinstance(cfg, dict):
                with open(cfg) as f:
                    self.yaml = yaml.load(f, Loader=yaml.FullLoader)
            else:
                self.yaml = cfg

        if nc and nc != self.yaml["nc"]:
            logger.debug(f'Overriding {cfg} nc={self.yaml["nc"]} with nc={nc}')
            self.yaml["nc"] = nc

        # Parse and build model
        self.model, self.save = parse_model(self.name, deepcopy(self.yaml), ch=[ch])
        self.layers = self.model

        # Build strides and anchors for Detect layer
        # Match PyTorch logic: m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
        m = self.model[-1]  # Last layer should be Detect()
        if isinstance(m, Detect):
            # Create dummy input to compute strides
            # In ttsim, we need to infer output shapes to calculate strides
            # The stride for each detection layer is: input_size / output_feature_map_size
            # Standard YOLO strides are [8, 16, 32] for 3 detection layers
            # This matches: 256/32=8, 256/16=16, 256/8=32

            # For a more accurate calculation, we would need to actually run forward,
            # but in ttsim during __init__, we typically set these based on architecture
            # For YOLOv4, the standard strides are well-known
            if m.nl == 3:
                m.stride = np.array([8.0, 16.0, 32.0], dtype=np.float32)
            elif m.nl == 4:
                m.stride = np.array([8.0, 16.0, 32.0, 64.0], dtype=np.float32)
            else:
                # Generalized: assume exponential growth
                m.stride = np.array(
                    [2.0 ** (i + 3) for i in range(m.nl)], dtype=np.float32
                )

            # Adjust anchors by stride (matches PyTorch: m.anchors /= m.stride.view(-1, 1, 1))
            m.anchors_np = m.anchors_np / m.stride.reshape(-1, 1, 1)
            m.anchor_grid_np = m.anchors_np.reshape(m.nl, 1, -1, 1, 1, 2)

            # Check anchor order
            general.check_anchor_order(m)

            # Store stride reference in model for later use
            self.stride = m.stride

            # Initialize biases (matches PyTorch: self._initialize_biases())
            self._initialize_biases()

        # We create submodules dynamically in this constructor via parse_model
        # so setattr is unable to register these. Below code is to explicitly
        # record these submodules (matching YOLOv7 pattern)
        for LL in self.layers:
            self._submodules[LL.name] = LL

        super().link_op2module()

    def _initialize_biases(self, cf=None):
        """Initialize biases into Detect() layer

        Args:
            cf: Class frequency (optional)

        Note: In ttsim, bias initialization is more limited than PyTorch.
        This function modifies the bias values stored in the Conv2d operations.
        The logic matches PyTorch's bias initialization for YOLO detection.
        """
        m = self.model[-1]  # Detect() module
        if not isinstance(m, Detect):
            return

        # For each detection layer conv and its corresponding stride
        for mi, s in zip(m.m, m.stride): # type: ignore[arg-type]
            # In ttsim, we need to access and modify the bias parameter
            # The bias shape is (na * no,) where na=num_anchors, no=num_outputs
            # After reshaping to (na, no), we modify specific indices:
            # - Index 4: objectness confidence
            # - Indices 5+: class probabilities

            if hasattr(mi, "bias") and mi.bias is not None:
                # Get bias as numpy array and reshape
                # bias shape: (na * no,) -> (na, no)
                bias_data = mi.bias.data if hasattr(mi.bias, "data") else mi.bias

                if isinstance(bias_data, np.ndarray):
                    b = bias_data.reshape(m.na, -1)

                    # obj (8 objects per 640 image)
                    # PyTorch: b[:, 4] += math.log(8 / (640 / s) ** 2)
                    b[:, 4] += math.log(8 / (640 / s) ** 2)

                    # cls (class probabilities)
                    # PyTorch: b[:, 5:] += math.log(0.6 / (m.nc - 0.99))
                    if cf is None:
                        b[:, 5:] += math.log(0.6 / (m.nc - 0.99))
                    else:
                        b[:, 5:] += np.log(cf / cf.sum())

                    # Update bias back to flattened form
                    if hasattr(mi.bias, "data"):
                        mi.bias.data = b.reshape(-1)
                    else:
                        mi.bias = b.reshape(-1)

    def create_input_tensors(self):
        self.input_tensors = {
            "yolo_input": F._from_shape(
                "yolo_input",
                [self.bs, self.in_channels, self.in_resolution, self.in_resolution],
            )
        }
        return

    def fuse(self):
        """Fuse Conv2d() + BatchNorm2d() layers for optimization

        Note: In ttsim, layer fusion is typically handled at the graph optimization level.
        This function is provided for API compatibility but may not perform actual fusion.
        """
        # In ttsim, we would need to traverse the module tree and fuse layers
        # This is more complex than PyTorch and typically done during graph compilation
        self.info()
        return self

    def info(self):
        """Print model information"""
        utils.model_info(self)

    def __call__(self, x=None, augment=False, profile=False):
        # Match PyTorch/YOLOv7 pattern: __call__ delegates to forward
        return self.forward(x, augment=augment, profile=profile)

    def forward(self, x=None, augment=False, profile=False):
        """
        Forward pass for YOLO model (TTSim version)
        Args:
            x: Input tensor (if None, use self.input_tensors['yolo_input'])
            augment: Whether to run augmented inference (scales/flips)
            profile: Whether to profile execution (not implemented)
        Returns:
            Detection outputs (possibly augmented)
        """
        if x is None:
            x = self.input_tensors["yolo_input"]
        if augment:
            img_size = x.shape[-2:]
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []
            for si, fi in zip(s, f):
                xi = self._scale_img(x, si, fi)
                yi = (
                    self.forward_once(xi)[0]
                    if isinstance(self.forward_once(xi), (tuple, list))
                    else self.forward_once(xi)
                )
                # De-scale boxes
                if hasattr(yi, "shape") and yi.shape[-1] >= 4:
                    yi[..., :4] /= si
                    if fi == 2:
                        yi[..., 1] = img_size[0] - yi[..., 1]
                    elif fi == 3:
                        yi[..., 0] = img_size[1] - yi[..., 0]
                y.append(yi)
            # Concatenate outputs along dim 1 (batch)
            import numpy as np

            y_cat = (
                np.concatenate(y, axis=1)
                if all(isinstance(yy, np.ndarray) for yy in y)
                else y
            )
            return y_cat, None
        else:
            return self.forward_once(x, profile=profile)

    @typing.no_type_check
    def _scale_img(self, x, scale, flip_axis):
        # Helper to scale and flip input (numpy version)
        import numpy as np

        if flip_axis is not None:
            x = np.flip(x, axis=flip_axis)
        if scale == 1:
            return x
        # Assume x shape: (bs, c, h, w)
        bs, c, h, w = x.shape
        new_h, new_w = int(h * scale), int(w * scale)
        # Use numpy for resizing (nearest neighbor)
        x_scaled = np.zeros((bs, c, new_h, new_w), dtype=x.dtype)
        for b in range(bs):
            for ch in range(c):
                x_scaled[b, ch] = np.array(
                    np.round(
                        np.array(
                            np.array(
                                np.array(
                                    np.array(
                                        np.interp(
                                            np.linspace(0, h - 1, new_h),
                                            np.arange(h),
                                            x[b, ch, :, :],
                                        )
                                    ).T
                                )
                            ).T
                        )
                    )
                )
        return x_scaled

    def forward_once(self, x, profile=False):
        """Single forward pass

        Args:
            x: Input tensor
            profile: Profile execution (not implemented)

        Returns:
            Detection outputs from Detect layer
        """
        y = []  # type: ignore[var-annotated]

        for i, m in enumerate(self.model):
            # Get input from previous layer or earlier layers
            if hasattr(m, "f") and m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # Multiple inputs from different layers
                    x = [x if j == -1 else y[j] for j in m.f]

            # Forward through layer
            x = m(x)

            # Save output if needed for later layers
            if hasattr(m, "i"):
                y.append(x if m.i in self.save else None)
            else:
                y.append(x)

        return x

    def get_forward_graph(self):
        # Ensure ops are registered by running forward pass with input tensors
        if not hasattr(self, "input_tensors") or "yolo_input" not in self.input_tensors:
            self.create_input_tensors()
        _ = self.__call__(self.input_tensors["yolo_input"])
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def analytical_param_count(self, lvl=0):
        return sum([m.analytical_param_count(lvl + 1) for m in self.layers])


def parse_model(model_name, d, ch):
    """Parse model from YAML configuration

    Args:
        model_name: Name for the model
        d: Model dictionary from YAML
        ch: Input channels [3]

    Returns:
        Tuple of (model_layers, save_list)
    """
    anchors = d["anchors"]
    nc = d["nc"]
    gd = d["depth_multiple"]
    gw = d["width_multiple"]

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers = []
    save = [] # type: ignore[var-annotated]
    c2 = ch[-1]  # output channels

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # Get module class
        m_str = m
        if isinstance(m, str):
            # Map string names to actual classes
            module_map = {
                "Conv": Conv,
                "Focus": Focus,
                "Concat": Concat,
                "SPP": SPP,
                "SPPCSP": SPPCSP,
                "VoVCSP": VoVCSP,
                "MP": MP,
                "Bottleneck": Bottleneck,
                "BottleneckCSP": BottleneckCSP,
                "BottleneckCSP2": BottleneckCSP2,
                "DWConv": DWConv,
                "Upsample": Upsample,
                "nn.Upsample": Upsample,  # Handle nn.Upsample from yaml
                "Detect": Detect,
                "CrossConv": CrossConv,
                "MixConv2d": MixConv2d,
                "C3": C3,
                "Sum": Sum,
                "GhostConv": GhostConv,
                "GhostBottleneck": GhostBottleneck,
            }
            m = module_map.get(m_str, None)
            if m is None:
                raise ValueError(f"Unknown module type: {m_str}")

        # Process arguments
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except Exception:
                pass

        # Apply depth multiplier
        n = max(round(n * gd), 1) if n > 1 else n

        # Build module arguments
        if m in [
            Conv,
            SPP,
            SPPCSP,
            Focus,
            VoVCSP,
            Bottleneck,
            BottleneckCSP,
            BottleneckCSP2,
            DWConv,
            MixConv2d,
            CrossConv,
            C3,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = general.make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]

            if m in [SPPCSP, VoVCSP, BottleneckCSP, BottleneckCSP2, C3]:
                args.insert(2, n)
                n = 1

        elif m in [GhostConv, GhostBottleneck]:
            c1, c2 = ch[f], args[0]
            c2 = general.make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]

        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])

        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)

        else:
            c2 = ch[f]

        # Create module instance(s)
        if n > 1:
            # Multiple instances - create a list
            m_list = []
            for idx in range(n):
                module_name = f"{model_name}.layer{i}.{idx}"
                m_list.append(m(module_name, *args))
            m_ = SimNN.ModuleList(m_list)
        else:
            module_name = f"{model_name}.layer{i}"
            m_ = m(module_name, *args)

        # Set module metadata
        t = m_str if isinstance(m_str, str) else str(m.__name__)
        np_count = 0  # TODO: Calculate parameter count properly

        if hasattr(m_, "analytical_param_count"):
            try:
                np_count = m_.analytical_param_count(0)
            except Exception:
                pass

        m_.i = i # type: ignore[attr-defined]
        m_.f = f # type: ignore[attr-defined]
        m_.type = t # type: ignore[attr-defined]
        m_.np = np_count # type: ignore[attr-defined]

        # Track which outputs to save
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)

    return SimNN.ModuleList(layers), sorted(save)


if __name__ == "__main__":
    cfg_url = "https://raw.githubusercontent.com/WongKinYiu/ScaledYOLOv4/refs/heads/yolov4-large/models/"
    cfgs = [
        "yolov4-csp.yaml",
        "yolov4-p5.yaml",
        "yolov4-p6.yaml",
        "yolov4-p7.yaml",
    ]
    ch = 3  # input channels for all variants

    for cfg_file in cfgs:
        cfg_path = os.path.join(cfg_url, cfg_file)

        cfg = {"yaml_cfg_path": cfg_path}
        model = Model("yolo", cfg)
        param_count = model.analytical_param_count()
        logger.debug(f"    #params= {param_count/1e6:.2f}M")
        model.create_input_tensors()
        logger.debug(f"    input shape= {model.input_tensors['yolo_input'].shape}")
        model_out = model()
        logger.debug(f"    output shapes= {[y.shape for y in model_out]}")
        # gg = model.get_forward_graph()
        # out_onnx_file = cfg_file.replace('.yaml', '.onnx')
        # gg.graph2onnx(out_onnx_file, do_model_check=False)
