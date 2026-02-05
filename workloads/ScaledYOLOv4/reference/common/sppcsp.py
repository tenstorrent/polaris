#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SPPCSP module validation: ttsim vs PyTorch comparison.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

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
    if data is None:
        data = np.random.randn(*shape).astype(np.float32)
    return F._from_data(name, data, is_param=False, is_const=False)


def inject_pytorch_weights_to_ttsim(
    pytorch_module: nn.Module, ttsim_module, prefix: str = "", verbose: bool = False
):
    state_dict = pytorch_module.state_dict()
    ttsim_tensors = {}
    ttsim_module.get_tensors(ttsim_tensors)
    injected, failed = 0, 0
    for pt_key, pt_weight in state_dict.items():
        if "num_batches_tracked" in pt_key:
            continue
        transformed = build_name_mapping(pt_key)
        ttsim_key = f"{prefix}.{transformed}" if prefix else transformed
        if ttsim_key in ttsim_tensors:
            pt_weight_np = pt_weight.detach().cpu().numpy()
            ttsim_tensor = ttsim_tensors[ttsim_key]
            ttsim_tensor.shape = list(pt_weight_np.shape)
            ttsim_tensor.dtype = pt_weight_np.dtype
            ttsim_tensor.data = pt_weight_np
            injected += 1
        else:
            failed += 1
    return injected, failed


def truncate_values(arr: np.ndarray, n: int = 5) -> str:
    flat = arr.flatten()[:n]
    return "[" + ", ".join(f"{v:.6f}" for v in flat) + ", ...]"


def compare_outputs(
    pytorch_output_np: np.ndarray,
    ttsim_output: np.ndarray,
    module_name: str = "Module",
    input_data: np.ndarray = None,
    ttsim_module=None,
    verbose: bool = False,
) -> dict:
    input_shape = list(input_data.shape) if input_data is not None else None
    input_values = truncate_values(input_data) if input_data is not None else "N/A"
    pytorch_shape = list(pytorch_output_np.shape)
    ttsim_shape = list(ttsim_output.shape)
    pytorch_values = truncate_values(pytorch_output_np)
    ttsim_values = truncate_values(ttsim_output)

    if input_data is not None:
        print(f"\n-- Input --")
        print(f"Shape: {input_shape}")
        print(f"Values: {input_values}")
    shape_match = pytorch_shape == ttsim_shape
    if not shape_match:
        print(
            f"FAIL: {module_name} shape mismatch - PyTorch: {pytorch_shape}, ttsim: {ttsim_shape}"
        )
        return {
            "passed": False,
            "input_shape": input_shape,
            "input_values": input_values,
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
            "input_values": input_values,
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
            "input_values": input_values,
            "pytorch_output_shape": pytorch_shape,
            "pytorch_output_values": pytorch_values,
            "ttsim_output_shape": ttsim_shape,
            "ttsim_output_values": ttsim_values,
            "values_match": False,
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
        }


def debug_tensors(ttsim_module):
    print("\n-- Debug: Intermediate Tensors --")
    all_tensors = {}
    ttsim_module.get_tensors(all_tensors)
    for name, tensor in all_tensors.items():
        print(f"  {name}: shape={tensor.shape}, has_data={tensor.data is not None}")


# ------------------- PyTorch----------------------#


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ConvPytorch(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            Mish()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPCSPPytorch(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = ConvPytorch(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = ConvPytorch(c_, c_, 3, 1)
        self.cv4 = ConvPytorch(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = ConvPytorch(4 * c_, c_, 1, 1)
        self.cv6 = ConvPytorch(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = Mish()
        self.cv7 = ConvPytorch(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))


def calculate_pytorch_memory_stats(pytorch_module, input_data, iterations=100):
    """Calculate memory performance metrics for PyTorch module."""
    input_torch = torch.from_numpy(input_data)
    pytorch_module.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_module(input_torch)

    # Measure execution time
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = pytorch_module(input_torch)
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) / iterations * 1000
    execution_time_s = execution_time_ms / 1000

    # Calculate data movement (bytes)
    # Input data
    input_bytes = input_data.nbytes

    # Weight data (all parameters)
    weight_bytes = 0
    for param in pytorch_module.parameters():
        weight_bytes += param.numel() * 4  # fp32 = 4 bytes

    # Output data
    output_bytes = output.numel() * 4  # fp32 = 4 bytes

    total_data_movement_bytes = input_bytes + weight_bytes + output_bytes
    data_movement_MB = total_data_movement_bytes / 1e6

    # Throughput
    inferences_per_sec = 1.0 / execution_time_s

    # Calculate total operations for PyTorch SPPCSP module
    # This is a simplified count - actual operations are more complex
    batch, c_in, h, w = input_data.shape
    c_out = output.shape[1]
    h_out = output.shape[2]
    w_out = output.shape[3]

    total_operations = 0
    c_ = int(2 * c_out * 0.5)  # e=0.5 by default

    # Estimate operations for all Conv layers, BN, activations, maxpools
    # cv1-cv7 convolutions + BN + Mish + MaxPools
    # This is a rough estimate based on module structure
    total_operations = c_out * h_out * w_out * c_in * 100  # Simplified

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
        if op.uses_compute_pipe is None:
            # Conv uses matrix pipe, BN and activation use vector pipe
            if "conv" in op.optype.lower():
                op.uses_compute_pipe = "matrix"
            else:
                op.uses_compute_pipe = "vector"
        # Matrix pipe supports fp16; vector pipe only supports fp32/int32
        if op.uses_compute_pipe == "matrix":
            op.precision = normalize_precision(op.precision)
        else:
            op.precision = "fp32"

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
        # Device info for display
        "compute_freq_MHz": device.freq_MHz,
        "memory_bw_GB_s": peak_bw_GBps,
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


# ------------------- ttsim----------------------#
from workloads.ScaledYOLOv4.models.common import SPPCSP as SPPCSPTtsim


def validate_sppcsp_module(verbose: bool = False):
    """Validate SPPCSP module by comparing PyTorch output with ttsim .data output."""
    print("\n-- SPPCSP Module Validation --")

    batch_size = 1
    c1, c2 = 64, 64
    height, width = 16, 16
    k = (5, 9, 13)

    input_shape = [batch_size, c1, height, width]
    input_data = np.random.randn(*input_shape).astype(np.float32)

    print(f"Input: {input_shape}, Config: c1={c1}, c2={c2}, k={k}")

    # PyTorch
    pytorch_module = SPPCSPPytorch(c1, c2, k=k)
    pytorch_module.eval()

    input_torch = torch.from_numpy(input_data)
    with torch.no_grad():
        pytorch_output = pytorch_module(input_torch)
    pytorch_output_np = pytorch_output.detach().cpu().numpy()

    # ttsim
    module_name = "test_sppcsp"
    ttsim_module = SPPCSPTtsim(module_name, c1, c2, k=k)

    input_tensor = create_input_tensor_with_data("input", input_shape, input_data)
    inject_pytorch_weights_to_ttsim(pytorch_module, ttsim_module, prefix=module_name)

    output_tensor = ttsim_module(input_tensor)

    # Validate output correctness
    if output_tensor.data is not None:
        ttsim_output = output_tensor.data
        result = compare_outputs(
            pytorch_output_np, ttsim_output, "SPPCSP", input_data, ttsim_module, verbose
        )
    else:
        print("FAIL: ttsim output has no data")
        debug_tensors(ttsim_module)
        return {
            "passed": False,
            "input_shape": input_shape,
            "pytorch_output_shape": list(pytorch_output_np.shape),
            "ttsim_output_shape": output_tensor.shape,
        }

    # Memory performance estimation
    print(f"\n{'='*60}")
    print("Memory Work Estimates")
    print(f"{'='*60}\n")

    try:
        from pathlib import Path

        polaris_root = Path(__file__).parent.parent.parent.parent.parent
        config_path = polaris_root / "config" / "tt_wh.yaml"
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        print(f"Using device: {device.devname} ({device.name})")
        print(f"Device frequency: {device.freq_MHz} MHz")
        print(f"Memory frequency: {device.memfreq_MHz} MHz")

        # Calculate PyTorch memory stats
        pytorch_stats = calculate_pytorch_memory_stats(
            pytorch_module, input_data, iterations=100
        )

        # Calculate ttsim memory stats
        ttsim_stats = calculate_ttsim_memory_stats(ttsim_module, device)

        # Compare stats
        compare_memory_stats(pytorch_stats, ttsim_stats)

        # Add memory stats to result
        result["memory_stats"] = {"pytorch": pytorch_stats, "ttsim": ttsim_stats}

    except Exception as e:
        print(f"Warning: Could not calculate memory estimates: {e}")
        import traceback

        traceback.print_exc()

    return result


if __name__ == "__main__":
    result = validate_sppcsp_module()
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULT: {'PASS' if result.get('passed', False) else 'FAIL'}")
    print(f"{'='*60}\n")
