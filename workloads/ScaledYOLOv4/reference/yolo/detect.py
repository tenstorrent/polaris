#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Detect module validation: ttsim vs PyTorch comparison.
"""

import os
import sys

# Add polaris root to path (4 levels up from this file)
polaris_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import re
import time
import numpy as np
import torch
import torch.nn as nn

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

RTOL = 1e-5
ATOL = 1e-6


def build_name_mapping(pytorch_name: str) -> str:
    """Convert PyTorch weight name to ttsim tensor name pattern."""
    name = pytorch_name
    name = re.sub(r"m\.(\d+)\.", r"m\1.", name)
    name = re.sub(r"\.bn\.weight$", ".bn.scale", name)
    name = re.sub(r"^bn\.weight$", "bn.scale", name)
    name = name.replace(".running_mean", ".input_mean")
    name = name.replace(".running_var", ".input_var")
    if ".conv.weight" in name:
        name = name.replace(".conv.weight", ".conv.param")
    elif name.endswith(".weight") and ".bn" not in name and "bn." not in name:
        name = name.replace(".weight", ".param")
    return name


def create_input_tensor_with_data(
    name: str, shape: list, data: np.ndarray = None
) -> SimTensor:
    """Create a ttsim SimTensor with numpy data."""
    if data is None:
        data = np.random.randn(*shape).astype(np.float32)
    return F._from_data(name, data, is_param=False, is_const=False)


def truncate_values(arr: np.ndarray, n: int = 5) -> str:
    """Return first n values of flattened array as string."""
    flat = arr.flatten()[:n]
    return "[" + ", ".join(f"{v:.6f}" for v in flat) + ", ...]"


def compare_outputs(
    pytorch_output_np: np.ndarray,
    ttsim_output: np.ndarray,
    module_name: str = "Module",
    input_data: list = None,
    verbose: bool = False,
) -> dict:
    """Compare PyTorch and ttsim outputs. Returns dict with passed, shapes, values."""
    # For Detect, input is a list of tensors
    if input_data is not None and isinstance(input_data, list):
        input_shape = [list(inp.shape) for inp in input_data]
        input_values = [truncate_values(inp) for inp in input_data]
    else:
        input_shape = None
        input_values = "N/A"

    pytorch_shape = list(pytorch_output_np.shape)
    ttsim_shape = list(ttsim_output.shape)
    pytorch_values = truncate_values(pytorch_output_np)
    ttsim_values = truncate_values(ttsim_output)

    if input_data is not None:
        print(f"\n-- Input --")
        print(f"Shapes: {input_shape}")

    shape_match = pytorch_shape == ttsim_shape
    if not shape_match:
        print(
            f"FAIL: {module_name} shape mismatch - PyTorch: {pytorch_shape}, ttsim: {ttsim_shape}"
        )
        return {
            "passed": False,
            "input_shape": input_shape,
            "input_values": str(input_values),
            "pytorch_output_shape": pytorch_shape,
            "pytorch_output_values": pytorch_values,
            "ttsim_output_shape": ttsim_shape,
            "ttsim_output_values": ttsim_values,
        }

    abs_diff = np.abs(pytorch_output_np - ttsim_output)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    is_close = np.allclose(pytorch_output_np, ttsim_output, rtol=RTOL, atol=ATOL)

    print(f"\n-- Output --")
    print(f"PyTorch: shape={pytorch_shape}, values={pytorch_values}")
    print(f"ttsim:   shape={ttsim_shape}, values={ttsim_values}")
    print(f"\n-- Comparison --")
    print(f"Max diff: {max_diff:.10f}, Mean diff: {mean_diff:.10f}")
    print(f"rtol={RTOL}, atol={ATOL}")

    if is_close:
        print(f"Result: PASS")
        return {
            "passed": True,
            "input_shape": input_shape,
            "input_values": str(input_values),
            "pytorch_output_shape": pytorch_shape,
            "pytorch_output_values": pytorch_values,
            "ttsim_output_shape": ttsim_shape,
            "ttsim_output_values": ttsim_values,
            "values_match": True,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
        }
    else:
        diff_mask = ~np.isclose(pytorch_output_np, ttsim_output, rtol=RTOL, atol=ATOL)
        num_diff = diff_mask.sum()
        total = diff_mask.size
        print(
            f"Result: FAIL - {num_diff}/{total} elements differ ({100*num_diff/total:.2f}%)"
        )
        return {
            "passed": False,
            "input_shape": input_shape,
            "input_values": str(input_values),
            "pytorch_output_shape": pytorch_shape,
            "pytorch_output_values": pytorch_values,
            "ttsim_output_shape": ttsim_shape,
            "ttsim_output_values": ttsim_values,
            "values_match": False,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
        }


# ------------------- PyTorch ----------------------#


class DetectPyTorch(nn.Module):
    """PyTorch version of YOLO Detect layer."""

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.stride = None
        self.nc = nc
        self.no = nc + 5  # outputs per anchor: x, y, w, h, obj, cls1...clsN
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl
        self.training = True  # Set True for training mode output
        self.export = False

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))

        # Conv2d for each detection layer (bias=False to match ttsim F.Conv2d)
        self.m = nn.ModuleList(
            nn.Conv2d(x, self.no * self.na, 1, bias=False) for x in ch
        )

    def forward(self, x):
        """Forward pass in training mode - returns raw outputs.

        Args:
            x: List of feature maps [(bs,c1,h1,w1), (bs,c2,h2,w2), (bs,c3,h3,w3)]

        Returns:
            List of reshaped outputs [(bs,na,h1,w1,no), (bs,na,h2,w2,no), (bs,na,h3,w3,no)]
        """
        for i in range(self.nl):
            # Conv: (bs, ch_in, ny, nx) -> (bs, na*no, ny, nx)
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape

            # Reshape + Permute: (bs, na*no, ny, nx) -> (bs, na, ny, nx, no)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )

        return x


# ------------------- ttsim ----------------------#
from workloads.ScaledYOLOv4.models.yolo import Detect as DetectTtsim


def calculate_pytorch_memory_stats(pytorch_module, input_data_list, iterations=100):
    """Calculate memory performance metrics for PyTorch Detect module.

    Args:
        pytorch_module: PyTorch Detect module
        input_data_list: List of numpy arrays (3 feature maps)
        iterations: Number of iterations for timing
    """
    input_torch_list = [torch.from_numpy(inp.copy()) for inp in input_data_list]
    pytorch_module.eval()
    pytorch_module.training = True  # Training mode for raw outputs

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_module(input_torch_list.copy())

    # Measure execution time
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            outputs = pytorch_module([inp.clone() for inp in input_torch_list])
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) / iterations * 1000
    execution_time_s = execution_time_ms / 1000

    # Calculate data movement (bytes)
    # Input data: sum of all input feature maps
    input_bytes = sum(inp.nbytes for inp in input_data_list)

    # Weight data: all parameters in Detect (3 Conv2d layers)
    weight_bytes = 0
    for param in pytorch_module.parameters():
        weight_bytes += param.numel() * 4  # fp32 = 4 bytes

    # Output data: sum of all output tensors
    output_bytes = sum(out.numel() * 4 for out in outputs)  # fp32 = 4 bytes

    total_data_movement_bytes = input_bytes + weight_bytes + output_bytes
    data_movement_MB = total_data_movement_bytes / 1e6

    # Throughput
    inferences_per_sec = 1.0 / execution_time_s

    # Calculate total operations for PyTorch Detect module
    # Detect has 3 detection layers, each with:
    # - Conv2d (1x1 kernel)
    # - Reshape + Permute (minimal compute, mainly memory reorganization)
    total_operations = 0

    for i, inp in enumerate(input_data_list):
        H = inp.shape[2]
        W = inp.shape[3]

        # Conv2d (1x1 kernel)
        conv = pytorch_module.m[i]
        K = conv.kernel_size[0] * conv.kernel_size[1]  # 1x1 = 1
        C_in = conv.in_channels
        C_out = conv.out_channels

        # Conv operations: 2 operations per MAC (multiply + accumulate)
        conv_ops = 2 * K * C_in * C_out * H * W
        total_operations += conv_ops

        # Reshape + Permute: minimal compute (1 read + 1 write per element)
        # Output shape: (bs, na, ny, nx, no)
        reshape_ops = C_out * H * W  # Just counting memory operations
        total_operations += reshape_ops

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


def calculate_ttsim_memory_stats(ttsim_module, device):
    """Calculate memory performance metrics for ttsim module."""
    # Get all operations from the module
    ops_dict = {}
    ttsim_module.get_ops(ops_dict)

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
            # Conv uses matrix pipe, Reshape/Transpose use vector pipe
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


def validate_detect_module(verbose: bool = False):
    """Validate Detect module by comparing PyTorch output with ttsim .data output."""
    print("\n-- Detect Module Validation --")

    np.random.seed(42)
    torch.manual_seed(42)

    # Configuration
    nc = 80  # number of classes
    anchors = [
        [10, 13, 16, 30, 33, 23],  # P3/8
        [30, 61, 62, 45, 59, 119],  # P4/16
        [116, 90, 156, 198, 373, 326],  # P5/32
    ]
    ch = [128, 256, 512]  # input channels for each detection layer

    nl = len(anchors)  # 3
    na = len(anchors[0]) // 2  # 3
    no = nc + 5  # 85

    # Input feature map shapes (smaller for faster testing)
    input_shapes = [(1, 128, 8, 8), (1, 256, 4, 4), (1, 512, 2, 2)]  # P3  # P4  # P5

    input_data = [np.random.randn(*shape).astype(np.float32) for shape in input_shapes]

    print(f"Input shapes: {[inp.shape for inp in input_data]}")
    print(f"Config: nc={nc}, nl={nl}, na={na}, no={no}, ch={ch}")

    # PyTorch
    pytorch_module = DetectPyTorch(nc=nc, anchors=anchors, ch=ch)
    pytorch_module.eval()
    pytorch_module.training = True  # Training mode for raw outputs

    input_torch = [torch.from_numpy(inp.copy()) for inp in input_data]
    with torch.no_grad():
        pytorch_outputs = pytorch_module(input_torch)
    pytorch_outputs_np = [out.detach().cpu().numpy() for out in pytorch_outputs]

    # ttsim
    module_name = "test_detect"
    ttsim_module = DetectTtsim(module_name, nc=nc, anchors=anchors, ch=ch)
    ttsim_module.training = True  # Training mode

    input_tensors = [
        create_input_tensor_with_data(f"input_{i}", list(inp.shape), inp)
        for i, inp in enumerate(input_data)
    ]

    # Inject weights from PyTorch Conv layers into ttsim
    # In ttsim Detect, self.m is [Conv0, Reshape0, Transpose0, Conv1, Reshape1, Transpose1, ...]
    # Conv layers are at indices 0, 3, 6 (i.e., 3*i)
    for i, conv_pt in enumerate(pytorch_module.m):
        conv_idx = 3 * i
        conv_op = ttsim_module.m[conv_idx]

        # Inject weight - params is [(position, tensor)]
        weight = conv_pt.weight.data.numpy()
        conv_op.params[0][1].data = weight
        conv_op.params[0][1].shape = list(weight.shape)

    # Forward pass
    ttsim_outputs = ttsim_module(input_tensors)

    # Compare each output layer
    all_passed = True
    results_per_layer = []

    for i, (pt_out, tt_out) in enumerate(zip(pytorch_outputs_np, ttsim_outputs)):
        print(f"\n--- Layer {i} ---")
        print(f"PyTorch shape: {pt_out.shape}")
        print(f"ttsim shape:   {tt_out.shape}")

        pt_shape = list(pt_out.shape)
        tt_shape = list(tt_out.shape)

        if tt_out.data is not None:
            ttsim_data = tt_out.data
            abs_diff = np.abs(pt_out - ttsim_data)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            is_close = np.allclose(pt_out, ttsim_data, rtol=RTOL, atol=ATOL)

            print(f"Max diff: {max_diff:.10f}, Mean diff: {mean_diff:.10f}")

            if is_close:
                print(f"Layer {i}: PASS")
                results_per_layer.append(
                    {
                        "layer": f"P{3+i}",
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
                        "layer": f"P{3+i}",
                        "type": "Detect",
                        "pytorch_shape": pt_shape,
                        "ttsim_shape": tt_shape,
                        "passed": False,
                        "max_diff": float(max_diff),
                    }
                )
                all_passed = False
        else:
            print(f"Layer {i}: ttsim output has no data")
            results_per_layer.append(
                {
                    "layer": f"P{3+i}",
                    "type": "Detect",
                    "pytorch_shape": pt_shape,
                    "ttsim_shape": tt_shape,
                    "passed": False,
                    "max_diff": None,
                }
            )
            all_passed = False

    # Build overall result
    result = {
        "passed": all_passed,
        "input_shape": [list(inp.shape) for inp in input_data],
        "input_values": "multi-scale feature maps",
        "pytorch_output_shape": [list(out.shape) for out in pytorch_outputs_np],
        "pytorch_output_values": [truncate_values(out) for out in pytorch_outputs_np],
        "ttsim_output_shape": [list(out.shape) for out in ttsim_outputs],
        "ttsim_output_values": [
            truncate_values(out.data) if out.data is not None else "N/A"
            for out in ttsim_outputs
        ],
        "layers": results_per_layer,
    }

    print(f"\n-- Overall Result: {'PASS' if all_passed else 'FAIL'} --")

    # Memory performance estimation
    print(f"\n{'='*60}")
    print("Memory Work Estimates (Aggregated across all detection layers)")
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
        print(f"Note: Metrics aggregated across {nl} detection layers\n")

        # Calculate PyTorch memory stats
        pytorch_mem_stats = calculate_pytorch_memory_stats(
            pytorch_module, input_data, iterations=100
        )

        # Calculate ttsim memory stats
        ttsim_mem_stats = calculate_ttsim_memory_stats(ttsim_module, device)

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
    print("\n" + "=" * 60)
    print("Detect Module Validation")
    print("=" * 60 + "\n")

    result = validate_detect_module(verbose=True)

    print("\n" + "=" * 60)
    if result.get("passed", False):
        print(f"Output Validation: PASS")
    else:
        print(f"Output Validation: FAIL")

    if "memory_stats" in result:
        print(f"Memory Estimates: Calculated")

    print("=" * 60)
