#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YOLO Model module validation: ttsim vs PyTorch comparison.
"""

import os
import sys

# Add polaris root to path (4 levels up from this file)
polaris_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
import yaml

import ttsim.front.functional.op as F
from ttsim.ops import SimTensor
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

import logging

try:
    # Silence python logging coming from ttsim modules (only show ERROR+)
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    # If the project uses loguru, remove default sinks and keep only ERROR+
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass

RTOL = 1e-2  # Relaxed tolerance for full model (accounts for BatchNorm differences)
ATOL = 1e-2

np.set_printoptions(precision=10, suppress=True, linewidth=150)


def truncate_values(arr: np.ndarray, n: int = 5) -> str:
    """Return first n values of flattened array as string, with scale info."""
    if arr is None:
        return "N/A"
    flat = arr.flatten()[:n]
    max_abs = np.abs(arr).max()
    mean_abs = np.abs(arr).mean()
    # Use scientific notation for very small values
    if max_abs < 1e-3 and max_abs > 0:
        values_str = "[" + ", ".join(f"{v:.2e}" for v in flat) + ", ...]"
    else:
        values_str = "[" + ", ".join(f"{v:.6f}" for v in flat) + ", ...]"
    return f"{values_str} (max={max_abs:.2e}, mean={mean_abs:.2e})"


# ------------------- PyTorch YOLO Implementation ----------------------#


def autopad(k, p=None):
    """Auto-padding to 'same'"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Mish(nn.Module):
    """Mish activation function - used in ScaledYOLOv4"""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class Conv(nn.Module):
    """Standard convolution with Mish activation"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            Mish()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def DWConv(c1, c2, k=1, s=1, act=True):
    """Depthwise convolution"""
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    """Standard bottleneck"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class BottleneckCSP2(nn.Module):
    """CSP Bottleneck variant"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class VoVCSP(nn.Module):
    """CSP VoVNet"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(VoVCSP, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1 // 2, c_ // 2, 3, 1)
        self.cv2 = Conv(c_ // 2, c_ // 2, 3, 1)
        self.cv3 = Conv(c_, c2, 1, 1)

    def forward(self, x):
        _, x1 = x.chunk(2, dim=1)
        x1 = self.cv1(x1)
        x2 = self.cv2(x1)
        return self.cv3(torch.cat((x1, x2), dim=1))


class SPP(nn.Module):
    """Spatial pyramid pooling"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPCSP(nn.Module):
    """CSP SPP"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(nn.Module):
    """Focus width and height information into channels"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )


class Concat(nn.Module):
    """Concatenate tensors"""

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Detect(nn.Module):
    """YOLO detection layer"""

    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1, bias=False) for x in ch
        )
        self.export = False
        self.training = True

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def check_anchor_order(m):
    """Check anchor order against stride order"""
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print("Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def make_divisible(x, divisor):
    """Make x divisible by divisor"""
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    """Initialize model weights"""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def parse_model_pytorch(d, ch):
    """Parse model from config dict"""
    anchors, nc, gd, gw = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
    )
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n
        if m in [
            nn.Conv2d,
            Conv,
            Bottleneck,
            SPP,
            DWConv,
            Focus,
            BottleneckCSP,
            BottleneckCSP2,
            SPPCSP,
            VoVCSP,
        ]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np_count = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np_count
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class ModelPyTorch(nn.Module):
    """PyTorch YOLO Model"""

    def __init__(self, cfg=None, ch=3, nc=None):
        super(ModelPyTorch, self).__init__()
        self.yaml_cfg_path = cfg.get("yaml_cfg_path", None)

        from ttsim.utils.common import parse_yaml

        self.yaml = parse_yaml(self.yaml_cfg_path)

        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.model, self.save = parse_model_pytorch(deepcopy(self.yaml), ch=[ch])

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 64
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride

        initialize_weights(self)

    def forward(self, x, augment=False, profile=False):
        return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y = []
        self.layer_outputs = []  # Store all intermediate outputs
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)
            # Store intermediate output for comparison
            if isinstance(x, list):
                self.layer_outputs.append([out.clone() for out in x])
            else:
                self.layer_outputs.append(x.clone() if hasattr(x, "clone") else x)
        return x


# ------------------- ttsim ----------------------#
from workloads.ScaledYOLOv4.models.yolo import Model as ModelTtsim


def calculate_pytorch_memory_stats(pytorch_model, input_data, iterations=50):
    """Calculate memory performance metrics for PyTorch full YOLO model.

    Args:
        pytorch_model: Full PyTorch YOLO model
        input_data: Input numpy array
        iterations: Number of iterations for timing (reduced for full model)
    """
    input_torch = torch.from_numpy(input_data)
    pytorch_model.eval()

    # Set Detect layer to training mode for raw outputs
    for m in pytorch_model.modules():
        if isinstance(m, Detect):
            m.training = True

    # Warmup
    with torch.no_grad():
        for _ in range(2):  # Reduced warmup for full model
            _ = pytorch_model(input_torch)

    # Measure execution time
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            outputs = pytorch_model(input_torch.clone())
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) / iterations * 1000
    execution_time_s = execution_time_ms / 1000

    # Calculate data movement (bytes)
    # Input data
    input_bytes = input_data.nbytes

    # Weight data: all parameters in full model
    weight_bytes = 0
    for param in pytorch_model.parameters():
        weight_bytes += param.numel() * 4  # fp32 = 4 bytes

    # Output data: sum of all detection outputs (multi-scale)
    if isinstance(outputs, list):
        output_bytes = sum(out.numel() * 4 for out in outputs)  # fp32 = 4 bytes
    else:
        output_bytes = outputs.numel() * 4

    total_data_movement_bytes = input_bytes + weight_bytes + output_bytes
    data_movement_MB = total_data_movement_bytes / 1e6

    # Throughput
    inferences_per_sec = 1.0 / execution_time_s

    # Calculate total operations for full YOLO model
    # This is a comprehensive count across all layers (Conv, Bottleneck, CSP, SPP, etc.)
    total_operations = 0

    # Iterate through all Conv2d layers in the model
    for module in pytorch_model.modules():
        if isinstance(module, nn.Conv2d):
            # Need to find the input/output shapes for this layer
            # For simplicity, we'll estimate based on parameters
            # Real implementation would track activations
            if hasattr(module, "weight"):
                K = module.kernel_size[0] * module.kernel_size[1]
                C_in = module.in_channels
                C_out = module.out_channels
                # Rough estimate: assuming average spatial size reduction
                # This is approximate since we don't track exact feature map sizes
                ops = 2 * K * C_in * C_out * 64  # Approximate spatial size
                total_operations += ops
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: 4 ops per element
            if hasattr(module, "num_features"):
                ops = 4 * module.num_features * 64  # Approximate spatial size
                total_operations += ops

    # Add detection head operations (reshape/permute)
    if isinstance(outputs, list):
        for out in outputs:
            total_operations += out.numel()  # Memory operations

    # Additional comparable metrics
    arithmetic_intensity = (
        total_operations / total_data_movement_bytes
        if total_data_movement_bytes > 0
        else 0
    )
    read_write_ratio = input_bytes / output_bytes if output_bytes > 0 else 0
    minimum_data = input_bytes + output_bytes
    memory_traffic_ratio = (
        total_data_movement_bytes / minimum_data if minimum_data > 0 else 1.0
    )

    return {
        "execution_time_ms": execution_time_ms,
        "inferences_per_sec": inferences_per_sec,
        "data_movement_MB": data_movement_MB,
        "input_bytes": input_bytes,
        "weight_bytes": weight_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_data_movement_bytes,
        "total_operations": total_operations,
        "arithmetic_intensity": arithmetic_intensity,
        "read_write_ratio": read_write_ratio,
        "memory_traffic_ratio": memory_traffic_ratio,
    }


def calculate_ttsim_memory_stats(ttsim_model, device):
    """Calculate memory performance metrics for ttsim full YOLO model."""
    # Get all operations from all layers in the model
    ops_dict = {}

    # Iterate through all layers and collect operations
    for layer in ttsim_model.model:
        layer_ops = {}
        if hasattr(layer, "get_ops"):
            layer.get_ops(layer_ops)
            ops_dict.update(layer_ops)

    # Helper function to convert precision to string format
    def normalize_precision(prec):
        """Convert precision to string format expected by device config."""
        if prec is None:
            return "fp16"
        # Handle numpy dtype objects
        if hasattr(prec, "name"):
            prec = prec.name
        # Convert to string and normalize
        prec_str = str(prec).lower()
        # Map common numpy dtype names to config precision names
        dtype_map = {
            "float32": "fp16",  # Use fp16 as fallback since fp32 not in config
            "float16": "fp16",
            "bfloat16": "bf16",
            "int8": "int8",
            "int32": "int32",
        }
        return dtype_map.get(prec_str, "fp16")

    # Set precision and compute pipe for all operations
    for op_name, op in ops_dict.items():
        op.precision = normalize_precision(op.precision)
        if op.uses_compute_pipe is None:
            # Conv uses matrix pipe, others use vector pipe
            if "conv" in op.optype.lower():
                op.uses_compute_pipe = "matrix"
            else:
                op.uses_compute_pipe = "vector"

    # Execute all operations to get performance stats
    total_compute_cycles = 0
    total_mem_rd_cycles = 0
    total_mem_wr_cycles = 0
    total_inBytes = 0
    total_outBytes = 0
    total_operations = 0
    num_memory_ops = 0

    for op_name, op in ops_dict.items():
        if op.perf_stats is not None:
            device.execute_op(op)
            total_compute_cycles += op.compute_cycles
            total_mem_rd_cycles += op.mem_rd_cycles
            total_mem_wr_cycles += op.mem_wr_cycles
            total_inBytes += op.perf_stats["inBytes"]
            total_outBytes += op.perf_stats["outBytes"]

            # Count total operations (MACs, adds, etc.)
            if "instrs" in op.perf_stats:
                for instr, count in op.perf_stats["instrs"].items():
                    total_operations += count

            # Count memory operations
            if op.mem_rd_cycles > 0 or op.mem_wr_cycles > 0:
                num_memory_ops += 1

    # Calculate metrics
    total_mem_cycles = total_mem_rd_cycles + total_mem_wr_cycles
    total_bytes = total_inBytes + total_outBytes

    # Execution time based on max(compute, memory) bottleneck
    ideal_cycles = max(total_compute_cycles, total_mem_cycles)
    execution_time_ms = ideal_cycles / device.freq_MHz / 1e3
    execution_time_s = execution_time_ms / 1000

    # Bandwidth calculations
    peak_bw_GBps = device.simconfig_obj.peak_bandwidth(freq_units="GHz")
    effective_bw_GBps = peak_bw_GBps * device.DG_MEMORY_UTIL_CONSTANT

    actual_bandwidth_GBps = 0.0
    if execution_time_s > 0:
        actual_bandwidth_GBps = (total_bytes / execution_time_s) / 1e9

    memory_efficiency = 0.0
    if effective_bw_GBps > 0:
        memory_efficiency = actual_bandwidth_GBps / effective_bw_GBps

    # Throughput
    inferences_per_sec = 0.0
    if execution_time_s > 0:
        inferences_per_sec = 1.0 / execution_time_s

    # Bottleneck
    bottleneck = "COMPUTE" if total_compute_cycles >= total_mem_cycles else "MEMORY"

    # Utilization
    mem_rd_util = total_mem_rd_cycles / ideal_cycles if ideal_cycles > 0 else 0
    mem_wr_util = total_mem_wr_cycles / ideal_cycles if ideal_cycles > 0 else 0

    # Additional Memory-Focused Metrics

    # 1. Memory Bandwidth Utilization (achieved vs peak, not just effective)
    mem_bw_utilization = actual_bandwidth_GBps / peak_bw_GBps if peak_bw_GBps > 0 else 0

    # 2. Arithmetic Intensity (ops/byte) - key roofline model metric
    arithmetic_intensity = total_operations / total_bytes if total_bytes > 0 else 0

    # 3. Read/Write Ratio - shows memory traffic balance
    read_write_ratio = total_inBytes / total_outBytes if total_outBytes > 0 else 0

    # 4. Bytes per Cycle - actual data transfer rate
    bytes_per_cycle = total_bytes / total_mem_cycles if total_mem_cycles > 0 else 0

    # 5. Memory Traffic Ratio - data reuse efficiency
    # Minimum data = input + output once (no intermediate transfers)
    minimum_data = total_inBytes + total_outBytes
    memory_traffic_ratio = total_bytes / minimum_data if minimum_data > 0 else 1.0

    # 6. Average Memory Latency - cycles per memory operation
    avg_memory_latency = total_mem_cycles / num_memory_ops if num_memory_ops > 0 else 0

    # 7. Memory Pressure Score - how memory-bound (0-1 scale)
    memory_pressure = (
        total_mem_cycles / (total_mem_cycles + total_compute_cycles)
        if (total_mem_cycles + total_compute_cycles) > 0
        else 0
    )

    return {
        "peak_bandwidth_GBps": peak_bw_GBps,
        "effective_bandwidth_GBps": effective_bw_GBps,
        "actual_bandwidth_GBps": actual_bandwidth_GBps,
        "memory_efficiency": memory_efficiency,
        "execution_time_ms": execution_time_ms,
        "inferences_per_sec": inferences_per_sec,
        "data_movement_MB": total_bytes / 1e6,
        "total_bytes": total_bytes,
        "memory_cycles": total_mem_cycles,
        "compute_cycles": total_compute_cycles,
        "ideal_cycles": ideal_cycles,
        "bottleneck": bottleneck,
        "mem_rd_util": mem_rd_util * device.DG_MEMORY_UTIL_CONSTANT,
        "mem_wr_util": mem_wr_util * device.DG_MEMORY_UTIL_CONSTANT,
        "mem_rd_cycles": total_mem_rd_cycles,
        "mem_wr_cycles": total_mem_wr_cycles,
        "input_bytes": total_inBytes,
        "output_bytes": total_outBytes,
        # Additional metrics
        "mem_bw_utilization": mem_bw_utilization,
        "arithmetic_intensity": arithmetic_intensity,
        "read_write_ratio": read_write_ratio,
        "bytes_per_cycle": bytes_per_cycle,
        "memory_traffic_ratio": memory_traffic_ratio,
        "avg_memory_latency": avg_memory_latency,
        "memory_pressure": memory_pressure,
        "total_operations": total_operations,
        "num_memory_ops": num_memory_ops,
    }


def compare_memory_stats(pytorch_stats, ttsim_stats):
    """Compare PyTorch and ttsim memory statistics."""
    print(f"\n{'='*60}")
    print("Memory Performance Comparison")
    print(f"{'='*60}")

    print(f"\n-- Execution Time --")
    print(f"PyTorch: {pytorch_stats['execution_time_ms']:.6f} ms")
    print(f"ttsim:   {ttsim_stats['execution_time_ms']:.6f} ms")

    print(f"\n-- Throughput (Inferences/sec) --")
    print(f"PyTorch: {pytorch_stats['inferences_per_sec']:.2f}")
    print(f"ttsim:   {ttsim_stats['inferences_per_sec']:.2f}")

    print(f"\n-- Data Movement --")
    print(f"PyTorch: {pytorch_stats['data_movement_MB']:.3f} MB")
    print(f"ttsim:   {ttsim_stats['data_movement_MB']:.3f} MB")

    print(f"\n-- Total Operations --")
    print(f"PyTorch: {pytorch_stats['total_operations']:,}")
    print(f"ttsim:   {ttsim_stats['total_operations']:,}")

    print(f"\n-- Arithmetic Intensity (ops/byte) --")
    print(f"PyTorch: {pytorch_stats['arithmetic_intensity']:.4f}")
    print(f"ttsim:   {ttsim_stats['arithmetic_intensity']:.4f}")

    print(f"\n-- Read/Write Ratio --")
    print(f"PyTorch: {pytorch_stats['read_write_ratio']:.2f}")
    print(f"ttsim:   {ttsim_stats['read_write_ratio']:.2f}")

    print(f"\n-- Memory Traffic Ratio --")
    print(f"PyTorch: {pytorch_stats['memory_traffic_ratio']:.2f}x")
    print(f"ttsim:   {ttsim_stats['memory_traffic_ratio']:.2f}x")

    print(f"\n-- ttsim-Only Memory Analysis --")
    print(f"Memory Efficiency (vs Effective):  {ttsim_stats['memory_efficiency']:.1%}")
    print(f"Memory BW Utilization (vs Peak):   {ttsim_stats['mem_bw_utilization']:.1%}")
    print(f"Bottleneck:                         {ttsim_stats['bottleneck']}")
    print(f"Memory Pressure Score:              {ttsim_stats['memory_pressure']:.3f}")

    print(f"\n-- ttsim Memory Cycles Breakdown --")
    print(f"Compute Cycles:    {ttsim_stats['compute_cycles']}")
    print(f"Memory Cycles:     {ttsim_stats['memory_cycles']}")
    print(f"  Read Cycles:     {ttsim_stats['mem_rd_cycles']}")
    print(f"  Write Cycles:    {ttsim_stats['mem_wr_cycles']}")
    print(f"Memory Read Util:  {ttsim_stats['mem_rd_util']:.1%}")
    print(f"Memory Write Util: {ttsim_stats['mem_wr_util']:.1%}")

    print(f"\n-- ttsim Additional Metrics --")
    print(f"Bytes per Cycle:       {ttsim_stats['bytes_per_cycle']:.2f} bytes/cycle")
    print(f"Avg Memory Latency:    {ttsim_stats['avg_memory_latency']:.1f} cycles/op")

    print(f"{'='*60}\n")

    return {"pytorch_stats": pytorch_stats, "ttsim_stats": ttsim_stats}


def transfer_weights_recursive(pt_module, tt_module, prefix="", verbose=False):
    """Recursively transfer weights from PyTorch to ttsim."""
    transferred = 0

    # Get all tensors from ttsim module
    tt_tensors = {}
    if hasattr(tt_module, "get_tensors"):
        tt_module.get_tensors(tt_tensors)

    if verbose:
        print(f"\n  ttsim tensors for {prefix} ({len(tt_tensors)} total):")
        for k in list(tt_tensors.keys())[:15]:
            print(f"    {k}: has_data={tt_tensors[k].data is not None}")

    # Get PyTorch state dict
    pt_state = pt_module.state_dict()

    for pt_key, pt_weight in pt_state.items():
        if "num_batches_tracked" in pt_key:
            continue

        # Build ttsim key - transform PyTorch naming to ttsim naming
        tt_key = pt_key

        # BatchNorm transforms - handle both .bn.weight and bn.weight patterns
        tt_key = re.sub(r"\.bn\.weight$", ".bn.scale", tt_key)
        tt_key = re.sub(r"^bn\.weight$", "bn.scale", tt_key)  # Top-level bn.weight
        tt_key = tt_key.replace(".running_mean", ".input_mean")
        tt_key = tt_key.replace(".running_var", ".input_var")

        # Conv weight transform
        if ".conv.weight" in tt_key:
            tt_key = tt_key.replace(".conv.weight", ".conv.param")
        elif tt_key == "conv.weight":  # Top-level conv.weight
            tt_key = "conv.param"
        elif tt_key.endswith(".weight") and ".bn" not in tt_key and "bn." not in tt_key:
            tt_key = tt_key.replace(".weight", ".param")

        # Module list indexing: m.0. -> m0.
        tt_key = re.sub(r"m\.(\d+)\.", r"m\1.", tt_key)

        # Full key with prefix
        full_key = f"{prefix}.{tt_key}" if prefix else tt_key

        # Try to find matching ttsim tensor - check multiple patterns
        matched = False
        for tt_tensor_key, tt_tensor in tt_tensors.items():
            # Pattern 1: exact match with full key
            if tt_tensor_key == full_key:
                matched = True
            # Pattern 2: ttsim key ends with transformed key
            elif tt_tensor_key.endswith(tt_key):
                matched = True
            # Pattern 3: ttsim key ends with full key
            elif tt_tensor_key.endswith(full_key):
                matched = True
            # Pattern 4: ttsim key contains the key parts
            elif tt_key in tt_tensor_key:
                matched = True

            if matched:
                pt_weight_np = pt_weight.detach().cpu().numpy()
                tt_tensor.shape = list(pt_weight_np.shape)
                tt_tensor.dtype = pt_weight_np.dtype
                tt_tensor.data = pt_weight_np
                transferred += 1
                if verbose:
                    print(f"  ✓ {pt_key} -> {tt_tensor_key}")
                break

        if not matched and verbose:
            print(f"  ✗ {pt_key} (tried: {full_key})")

    return transferred


def validate_model_module(verbose: bool = False):
    """Validate full YOLO Model by comparing PyTorch output with ttsim output."""
    print("\n-- YOLO Model Validation --")

    np.random.seed(42)
    torch.manual_seed(42)

    cfg_url = "https://raw.githubusercontent.com/WongKinYiu/ScaledYOLOv4/refs/heads/yolov4-large/models/"
    cfg_file = "yolov4-p5.yaml"
    config_path = os.path.join(cfg_url, cfg_file)
    print(f"Using config: {config_path}")

    # Input configuration - minimum 64x64 for YOLOv4-P5
    batch_size = 1
    input_channels = 3
    input_height = 64
    input_width = 64

    input_shape = [batch_size, input_channels, input_height, input_width]
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print(f"Input shape: {input_shape}")

    # PyTorch Model
    print("\nCreating PyTorch model...")
    cfg = {"yaml_cfg_path": config_path}
    model_pt = ModelPyTorch(cfg)
    model_pt.eval()

    # Set Detect layer to training mode for raw outputs
    for m in model_pt.modules():
        if isinstance(m, Detect):
            m.training = True

    input_torch = torch.from_numpy(input_data)
    with torch.no_grad():
        output_pt = model_pt(input_torch)

    # Get PyTorch outputs
    if isinstance(output_pt, list):
        pytorch_outputs = [out.detach().cpu().numpy() for out in output_pt]
    else:
        pytorch_outputs = [output_pt.detach().cpu().numpy()]

    print(f"PyTorch output shapes: {[out.shape for out in pytorch_outputs]}")

    # ttsim Model
    print("\nCreating ttsim model...")
    model_tt = ModelTtsim(name="test_yolo", cfg=cfg)
    model_tt.model[-1].training = True  # Set Detect to training mode

    # Create input tensor
    input_tt = F._from_data("input", input_data, is_param=False, is_const=False)

    # Transfer weights
    print("\nTransferring weights...")
    total_transferred = 0
    for i, (pt_layer, tt_layer) in enumerate(zip(model_pt.model, model_tt.model)):
        transferred = transfer_weights_recursive(
            pt_layer, tt_layer, prefix=tt_layer.name, verbose=verbose
        )
        total_transferred += transferred
    print(f"Transferred {total_transferred} weight tensors")

    # Forward pass
    num_layers = len(model_tt.model)
    print(f"\nRunning ttsim forward pass ({num_layers} layers)...")

    y_tt = []
    x_tt = input_tt

    for i, m in enumerate(model_tt.model):
        layer_name = m.name.split(".")[-1] if hasattr(m, "name") else type(m).__name__
        print(f"  [{i+1}/{num_layers}] {layer_name}")

        if hasattr(m, "f") and m.f != -1:
            if isinstance(m.f, int):
                x_tt = y_tt[m.f]
            else:
                x_tt = [x_tt if j == -1 else y_tt[j] for j in m.f]

        x_tt = m(x_tt)

        if hasattr(m, "i"):
            y_tt.append(x_tt if m.i in model_tt.save else None)
        else:
            y_tt.append(x_tt)

    print("Forward pass complete.")
    output_tt = x_tt

    # Get ttsim outputs
    if isinstance(output_tt, list):
        ttsim_outputs = output_tt
    else:
        ttsim_outputs = [output_tt]

    print(f"ttsim output shapes: {[out.shape for out in ttsim_outputs]}")

    # Compare final detection outputs
    print("\n-- Comparison --")
    all_passed = True
    results_per_layer = []

    print(f"\n--- Final Detection Outputs ({len(pytorch_outputs)} scales) ---")
    for i, (pt_out, tt_out) in enumerate(zip(pytorch_outputs, ttsim_outputs)):
        tt_data = tt_out.data if hasattr(tt_out, "data") else tt_out

        print(f"\n--- Detection Layer {i} ---")
        print(f"PyTorch shape: {pt_out.shape}")
        print(
            f"ttsim shape:   {list(tt_out.shape) if hasattr(tt_out, 'shape') else 'N/A'}"
        )

        pt_shape = list(pt_out.shape)
        tt_shape = list(tt_out.shape) if hasattr(tt_out, "shape") else None

        if tt_data is not None:
            # Check shape
            if pt_shape != list(tt_data.shape):
                print(f"FAIL: Shape mismatch")
                results_per_layer.append(
                    {
                        "layer": f"Detect_{i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": False,
                        "reason": "shape_mismatch",
                    }
                )
                all_passed = False
                continue

            # Check values
            abs_diff = np.abs(pt_out - tt_data)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            is_close = np.allclose(pt_out, tt_data, rtol=RTOL, atol=ATOL)

            print(f"Max diff: {max_diff:.10f}, Mean diff: {mean_diff:.10f}")
            print(f"PyTorch values: {truncate_values(pt_out)}")
            print(f"ttsim values:   {truncate_values(tt_data)}")

            if is_close:
                print(f"Layer {i}: PASS")
                results_per_layer.append(
                    {
                        "layer": f"Detect_{i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": True,
                        "max_diff": float(max_diff),
                    }
                )
            else:
                print(f"Layer {i}: FAIL")
                results_per_layer.append(
                    {
                        "layer": f"Detect_{i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": False,
                        "max_diff": float(max_diff),
                    }
                )
                all_passed = False
        else:
            print(f"Layer {i}: ttsim output has no data (shape-only mode)")
            # Shape-only validation
            if pt_shape == tt_shape:
                print(f"Layer {i}: PASS (shape-only)")
                results_per_layer.append(
                    {
                        "layer": f"Detect_{i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": True,
                        "shape_only": True,
                    }
                )
            else:
                print(f"Layer {i}: FAIL (shape mismatch)")
                results_per_layer.append(
                    {
                        "layer": f"Detect_{i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": False,
                        "reason": "shape_mismatch",
                    }
                )
                all_passed = False

    # Build result
    result = {
        "passed": all_passed,
        "input_shape": input_shape,
        "input_values": truncate_values(input_data),
        "config": cfg_file,
        "num_layers": len(model_pt.model),
        "weights_transferred": total_transferred,
        "pytorch_output_shape": [list(out.shape) for out in pytorch_outputs],
        "pytorch_output_values": [truncate_values(out) for out in pytorch_outputs],
        "ttsim_output_shape": [list(out.shape) for out in ttsim_outputs],
        "ttsim_output_values": [
            (
                truncate_values(out.data)
                if hasattr(out, "data") and out.data is not None
                else "N/A"
            )
            for out in ttsim_outputs
        ],
        "layers": results_per_layer,
        "detection_layers": len(pytorch_outputs),
    }

    # Summary
    passed_layers = sum(1 for r in results_per_layer if r.get("passed", False))
    print(f"\n-- Summary --")
    print(f"Total layers: {len(model_pt.model)}")
    print(f"Compared layers: {len(results_per_layer)}")
    print(f"Passed layers: {passed_layers}/{len(results_per_layer)}")
    print(f"Detection outputs: {len(pytorch_outputs)} scales")

    print(f"\n-- Overall Result: {'PASS' if all_passed else 'FAIL'} --")

    # Memory performance estimation
    print(f"\n{'='*60}")
    print("Memory Work Estimates (Full YOLO Model - All Layers)")
    print(f"{'='*60}\n")

    try:
        # Load device configuration
        config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        print(f"Using device: {device.devname} ({device.name})")
        print(f"Device frequency: {device.freq_MHz} MHz")
        print(f"Memory frequency: {device.memfreq_MHz} MHz")
        print(f"Note: Metrics aggregated across {num_layers} model layers\n")

        # Calculate PyTorch memory stats
        pytorch_mem_stats = calculate_pytorch_memory_stats(
            model_pt, input_data, iterations=5
        )

        # Calculate ttsim memory stats
        ttsim_mem_stats = calculate_ttsim_memory_stats(model_tt, device)

        # Compare stats
        memory_comparison = compare_memory_stats(pytorch_mem_stats, ttsim_mem_stats)

        # Add memory stats to result
        result["memory_stats"] = memory_comparison

    except Exception as e:
        print(f"Warning: Could not calculate memory estimates: {e}")
        import traceback

        traceback.print_exc()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("YOLO Model Validation")
    print("=" * 60 + "\n")

    result = validate_model_module(verbose=args.verbose)

    print("\n" + "=" * 60)
    if result.get("passed", False):
        print(f"Output Validation: PASS")
    else:
        print(f"Output Validation: FAIL")

    if "memory_stats" in result:
        print(f"Memory Estimates: Calculated")

    print("=" * 60)
