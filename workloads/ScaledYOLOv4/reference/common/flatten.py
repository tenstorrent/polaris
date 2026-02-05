#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Flatten module validation: ttsim vs PyTorch comparison.
"""

import os
import sys

# Add polaris root to path
polaris_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

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


def create_input_tensor_with_data(
    name: str, shape: list, data: np.ndarray = None
) -> SimTensor:
    if data is None:
        data = np.random.randn(*shape).astype(np.float32)
    return F._from_data(name, data, is_param=False, is_const=False)


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
    """Compare PyTorch and ttsim outputs. Returns dict with passed, shapes, values."""
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
# Flatten uses torch.flatten or nn.Flatten

# ------------------- ttsim----------------------#
from workloads.ScaledYOLOv4.models.common import Flatten as FlattenTtsim


def calculate_pytorch_memory_stats(input_data, iterations=100):
    """Calculate memory performance metrics for PyTorch flatten operation."""
    input_torch = torch.from_numpy(input_data)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = input_torch.flatten(1)

    # Measure execution time
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            output = input_torch.flatten(1)
    end_time = time.perf_counter()

    execution_time_ms = (end_time - start_time) / iterations * 1000
    execution_time_s = execution_time_ms / 1000

    # Calculate data movement (bytes) - just input and output, no weights
    input_bytes = input_data.nbytes
    weight_bytes = 0  # Flatten has no weights
    output_bytes = output.numel() * 4  # fp32 = 4 bytes

    total_data_movement_bytes = input_bytes + output_bytes
    data_movement_MB = total_data_movement_bytes / 1e6

    # Throughput
    inferences_per_sec = 1.0 / execution_time_s

    # Reshape is essentially zero operations (just memory layout change)
    total_operations = 0

    # Additional metrics
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
    ops_dict = {}
    ttsim_module.get_ops(ops_dict)

    def normalize_precision(prec):
        if prec is None:
            return "fp16"
        if hasattr(prec, "name"):
            prec = prec.name
        prec_str = str(prec).lower()
        dtype_map = {
            "float32": "fp16",
            "float16": "fp16",
            "bfloat16": "bf16",
            "int8": "int8",
            "int32": "int32",
        }
        return dtype_map.get(prec_str, "fp16")

    for op_name, op in ops_dict.items():
        if op.uses_compute_pipe is None:
            op.uses_compute_pipe = "vector"
        if op.uses_compute_pipe == "matrix":
            op.precision = normalize_precision(op.precision)
        else:
            op.precision = "fp32"

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

            if "instrs" in op.perf_stats:
                for instr, count in op.perf_stats["instrs"].items():
                    total_operations += count

            if op.mem_rd_cycles > 0 or op.mem_wr_cycles > 0:
                num_memory_ops += 1

    total_mem_cycles = total_mem_rd_cycles + total_mem_wr_cycles
    total_bytes = total_inBytes + total_outBytes

    ideal_cycles = max(total_compute_cycles, total_mem_cycles)
    execution_time_ms = ideal_cycles / device.freq_MHz / 1e3
    execution_time_s = execution_time_ms / 1000

    peak_bw_GBps = device.simconfig_obj.peak_bandwidth(freq_units="GHz")
    effective_bw_GBps = peak_bw_GBps * device.DG_MEMORY_UTIL_CONSTANT

    actual_bandwidth_GBps = 0.0
    if execution_time_s > 0:
        actual_bandwidth_GBps = (total_bytes / execution_time_s) / 1e9

    memory_efficiency = 0.0
    if effective_bw_GBps > 0:
        memory_efficiency = actual_bandwidth_GBps / effective_bw_GBps

    inferences_per_sec = 0.0
    if execution_time_s > 0:
        inferences_per_sec = 1.0 / execution_time_s

    bottleneck = "COMPUTE" if total_compute_cycles >= total_mem_cycles else "MEMORY"
    mem_rd_util = total_mem_rd_cycles / ideal_cycles if ideal_cycles > 0 else 0
    mem_wr_util = total_mem_wr_cycles / ideal_cycles if ideal_cycles > 0 else 0

    mem_bw_utilization = actual_bandwidth_GBps / peak_bw_GBps if peak_bw_GBps > 0 else 0
    arithmetic_intensity = total_operations / total_bytes if total_bytes > 0 else 0
    read_write_ratio = total_inBytes / total_outBytes if total_outBytes > 0 else 0
    bytes_per_cycle = total_bytes / total_mem_cycles if total_mem_cycles > 0 else 0
    minimum_data = total_inBytes + total_outBytes
    memory_traffic_ratio = total_bytes / minimum_data if minimum_data > 0 else 1.0
    avg_memory_latency = total_mem_cycles / num_memory_ops if num_memory_ops > 0 else 0
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


def validate_flatten_module(verbose: bool = False):
    """Validate Flatten module by comparing PyTorch output with ttsim .data output."""
    print("\n-- Flatten Module Validation --")

    batch_size = 2
    channels = 16
    height, width = 4, 4

    input_shape = [batch_size, channels, height, width]
    input_data = np.random.randn(*input_shape).astype(np.float32)

    expected_output_size = channels * height * width
    print(f"Input: {input_shape}")
    print(f"Expected output: [{batch_size}, {expected_output_size}]")

    # PyTorch
    input_torch = torch.from_numpy(input_data)
    with torch.no_grad():
        pytorch_output = input_torch.flatten(1)  # Flatten from dim 1 onwards
    pytorch_output_np = pytorch_output.detach().cpu().numpy()

    # ttsim
    module_name = "test_flatten"
    ttsim_module = FlattenTtsim(module_name)

    input_tensor = create_input_tensor_with_data("input", input_shape, input_data)
    output_tensor = ttsim_module(input_tensor)

    if output_tensor.data is not None:
        ttsim_output = output_tensor.data
        result = compare_outputs(
            pytorch_output_np,
            ttsim_output,
            "Flatten",
            input_data,
            ttsim_module,
            verbose,
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

        config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        print(f"Using device: {device.devname} ({device.name})")
        print(f"Device frequency: {device.freq_MHz} MHz")
        print(f"Memory frequency: {device.memfreq_MHz} MHz")

        pytorch_mem_stats = calculate_pytorch_memory_stats(input_data, iterations=100)
        ttsim_mem_stats = calculate_ttsim_memory_stats(ttsim_module, device)
        compare_memory_stats(pytorch_mem_stats, ttsim_mem_stats)

        result["memory_stats"] = {
            "pytorch": pytorch_mem_stats,
            "ttsim": ttsim_mem_stats,
        }

    except Exception as e:
        print(f"Warning: Could not calculate memory estimates: {e}")
        import traceback

        traceback.print_exc()

    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Flatten Module Validation")
    print("=" * 60 + "\n")

    result = validate_flatten_module()

    print("\n" + "=" * 60)
    if result.get("passed", False):
        print(f"Output Validation: PASS")
    else:
        print(f"Output Validation: FAIL")

    if "memory_stats" in result:
        print(f"Memory Estimates: Calculated")

    print("=" * 60)
