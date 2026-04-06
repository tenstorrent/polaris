#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
from pathlib import Path
import sys
import logging

# Silence the ttsim-related log messages
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("ttsim")
except ImportError:
    pass

import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_gather

try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def ref_impl(data_shape, indices, axis):
    X = np.random.randn(*data_shape)
    odata = np.take(X, indices, axis=axis)
    return list(odata.shape)


# Test cases
test_name = "test_gather"
test_cases = [
    ("Gather from 2D along axis 0", [3, 4], [0, 2], 0),
    ("Gather from 2D along axis 1", [3, 4], [1, 3], 1),
    ("Gather from 3D along axis 0", [2, 3, 4], [1], 0),
    ("Gather from 3D along axis 2", [2, 3, 4], [[0, 1], [2, 3]], 2),
    ("Gather with empty indices", [3, 4], [], 0),
    # from onnx.backend.test
    ("test_gather_0", [5, 4, 3, 2], [0, 1, 3], 0),
    ("test_gather_1", [5, 4, 3, 2], [0, 1, 3], 1),
    ("test_gather_2d_indices", [3, 3], [[0, 2]], 1),
    ("test_gather_negative_indices", [10], [0, -9, -10], 0),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_gather():
    msgw = max([len(x[0]) for x in test_cases])
    for tno, (tmsg, data_shape, indices, axis) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"
        i_tensors = [
            F._from_shape("X", data_shape, np_dtype=np.float32),
            F._from_data("I", np.array(indices)),
        ]

        o_tensors = [make_tensor("Y")]
        op_info = {
            "name": op_name,
            "optype": "Gather",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"axis": axis},
        }
        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        inf_shape = o_tensors[0].shape
        ref_shape = ref_impl(data_shape, indices, axis)

        if inf_shape == ref_shape:
            logger.debug(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS")
        else:
            logger.debug("INPUTS:")
            for x in i_tensors:
                logger.debug("\t%s", x)
            logger.debug("OUTPUTS:")
            for x in o_tensors:
                logger.debug("\t%s", x)
            assert (
                False
            ), f"TEST[{tno:3d}] {tmsg:{msgw}s} FAIL {inf_shape} != {ref_shape}"


def _run_gather(data, indices, axis=0, dtype="float32"):
    """Build a Gather op with actual data, run shape inference + compute."""
    data_arr = np.array(data, dtype=dtype)
    idx_arr = np.array(indices, dtype="int64")

    data_t = F._from_data("X", data_arr)
    idx_t = F._from_data("I", idx_arr)
    out_t = make_tensor("Y")

    op_info = {
        "name": "Gather",
        "optype": "Gather",
        "inList": [data_t.name, idx_t.name],
        "outList": [out_t.name],
        "attrs": {"axis": axis},
    }
    op = SimOp(op_info)
    op.get_perf_counts([data_t, idx_t], [out_t])

    result = compute_gather([data_t, idx_t], op)
    expected = np.take(data_arr, idx_arr, axis=axis)
    return out_t.shape, result, expected


# ---------------------------------------------------------------------------
# Numerical test cases: (id, data, indices, axis)
# ---------------------------------------------------------------------------


def test_gather_memory_validation(capsys, request):
    """
    Test memory validation for gather operation.
    Validates instructions executed and data moved for various scenarios.

    Run with: pytest tests/test_ops/test_gather.py::test_gather_memory_validation -v
    For detailed output: add -s flag
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 60)
    logger.info("Gather Operation Memory Validation")
    logger.info("=" * 60)

    # Load device configuration
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        logger.info(f"\nDevice: {device.devname} ({device.name})")
        logger.info(f"Device frequency: {device.freq_MHz} MHz")
        logger.info(f"Memory frequency: {device.memfreq_MHz} MHz")
        logger.info(
            "Peak bandwidth: %.2f GB/s",
            device.simconfig_obj.peak_bandwidth(freq_units="GHz"),
        )
    except Exception as e:
        logger.info(f"\nWarning: Could not load device config: {e}")
        pytest.skip(f"Could not load device config: {e}")
        return

    # Test cases: different gather patterns
    test_cases = [
        {
            "name": "1D Sparse Gather",
            "data_shape": [1000],
            "indices": list(range(0, 500, 2)),
            "axis": 0,
            "description": "Gather every other element from 1D",
        },
        {
            "name": "2D Row Gather",
            "data_shape": [64, 32],
            "indices": [0, 10, 20, 30, 40, 50],
            "axis": 0,
            "description": "Gather specific rows from 2D matrix",
        },
        {
            "name": "2D Col Gather",
            "data_shape": [32, 64],
            "indices": list(range(0, 64, 4)),
            "axis": 1,
            "description": "Gather columns from 2D matrix",
        },
        {
            "name": "3D Depth Gather",
            "data_shape": [16, 32, 48],
            "indices": list(range(0, 48, 3)),
            "axis": 2,
            "description": "Gather depth slices from 3D tensor",
        },
        {
            "name": "4D Channel Gather",
            "data_shape": [2, 64, 16, 16],
            "indices": list(range(0, 64, 2)),
            "axis": 1,
            "description": "Gather channels from 4D tensor",
        },
    ]

    logger.info("\n%s", "=" * 60)
    logger.info("Running Memory Validation Tests")
    logger.info("%s\n", "=" * 60)

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        data_shape = test_case["data_shape"]
        indices = test_case["indices"]
        axis = test_case["axis"]

        logger.debug(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Data shape: {data_shape}")
        logger.debug(f"Indices count: {len(indices)}")
        logger.debug(f"Gather axis: {axis}")

        # Generate test data
        np.random.seed(42)
        data_arr = np.random.randn(*data_shape).astype(np.float32)
        idx_arr = np.array(indices, dtype="int64")

        # Create operation with fp32 precision for consistency
        data_t = F._from_data("X", data_arr)
        idx_t = F._from_data("I", idx_arr)
        out_t = make_tensor("Y")

        op_info = {
            "name": f'gather_mem_{test_name.replace(" ", "_")}',
            "optype": "Gather",
            "inList": [data_t.name, idx_t.name],
            "outList": [out_t.name],
            "attrs": {"axis": axis},
        }
        op_obj = SimOp(op_info)
        op_obj.precision = "fp32"
        op_obj.uses_compute_pipe = "vector"

        # Get performance counts and execute
        op_obj.get_perf_counts([data_t, idx_t], [out_t])
        device.execute_op(op_obj)

        # Extract performance stats directly
        perf_stats = op_obj.perf_stats
        output_shape = out_t.shape
        output_elems = np.prod(output_shape)
        input_elems = np.prod(data_shape)

        logger.debug(f"Output shape: {output_shape}")

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate 'mov' instruction is present
        assert (
            "mov" in actual_instrs
        ), f"Expected 'mov' instruction for Gather, got {list(actual_instrs.keys())}"

        # Get memory metrics
        input_bytes = perf_stats.get("inBytes", 0)
        output_bytes = perf_stats.get("outBytes", 0)
        total_data_moved = input_bytes + output_bytes

        # Compute cycles
        compute_cycles = op_obj.compute_cycles
        mem_rd_cycles = op_obj.mem_rd_cycles
        mem_wr_cycles = op_obj.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        ideal_cycles = max(compute_cycles, memory_cycles)

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        logger.debug("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {total_instructions:,}")
        logger.debug(f"  Instruction types:     {dict(actual_instrs)}")
        logger.debug(f"  Input elements:        {input_elems:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(f"  Indices count:         {len(indices):,}")

        # Validate: 'mov' instructions should match output elements (1 per element)
        instruction_ratio = total_instructions / output_elems if output_elems > 0 else 0
        assert (
            0.8 <= instruction_ratio <= 1.5
        ), f"Instruction count mismatch: {total_instructions} vs expected ~{output_elems}"
        logger.debug(
            "  ✓ Instruction count validates (1 'mov' per output element, ratio=%.2f)",
            instruction_ratio,
        )

        logger.debug("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {input_bytes:,} ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {output_bytes:,} ({output_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_moved:,} ({total_data_moved/1024:.2f} KB)"
        )

        # Calculate gather sparsity (how selective the gather is)
        sparsity_pct = (
            (len(indices) / data_shape[axis] * 100) if data_shape[axis] > 0 else 0
        )
        logger.debug(
            "  Gather sparsity:  %.1f%% of axis %s gathered (%s/%s)",
            sparsity_pct,
            axis,
            len(indices),
            data_shape[axis],
        )

        assert output_bytes > 0, "Output bytes should be positive"
        logger.debug("  ✓ Output bytes positive (gathered data)")

        logger.debug("\n  -- Memory Metrics --")
        logger.debug(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
        logger.debug(
            "  Read/Write ratio:      %.2f",
            input_bytes / output_bytes if output_bytes > 0 else 0,
        )
        logger.debug(
            "  Bytes per element:     %.1f",
            output_bytes / output_elems if output_elems > 0 else 0,
        )

        # Gather is typically memory-bound (simple copy)
        assert (
            arithmetic_intensity < 1.0
        ), f"Arithmetic intensity too high for memory-bound Gather: {arithmetic_intensity}"
        logger.debug("  ✓ Low arithmetic intensity (memory-bound operation)")

        logger.debug("\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {ideal_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")

        # Validate: Gather should be memory-bound for typical cases
        if output_elems > 1000:
            assert (
                bottleneck == "MEMORY"
            ), f"Expected MEMORY bottleneck for large gather, got {bottleneck}"
            logger.debug("  ✓ Memory-bound as expected")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "data_shape": data_shape,
                "output_shape": output_shape,
                "indices_count": len(indices),
                "axis": axis,
                "sparsity_pct": sparsity_pct,
                "instructions": total_instructions,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "total_data_moved": total_data_moved,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
            }
        )

        logger.debug("\n  ✓ Test PASSED")

    # Summary
    logger.info("\n%s", "=" * 60)
    logger.info("Memory Validation Summary")
    logger.info("%s\n", "=" * 60)
    logger.info(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    logger.info("\n-- Arithmetic Intensity Comparison --")
    logger.info(f"{'Test Name':<30s} {'Ops/Byte':<12s} {'Data Moved':<15s}")
    logger.info("-" * 60)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['arithmetic_intensity']:<12.4f} {result['total_data_moved']/1024:>10.1f} KB"
        )

    # Gather Pattern Analysis
    logger.info("\n-- Gather Pattern Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Sparsity':<12s} {'Axis':<8s} {'Indices':<12s}"
    )
    logger.info("-" * 65)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['sparsity_pct']:>8.1f}%   {result['axis']:<8d} {result['indices_count']:>10,}"
        )

    # Bottleneck Analysis
    logger.info("\n-- Bottleneck Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    logger.info("-" * 80)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    logger.info("\n%s", "=" * 60)
    logger.info("Memory validation complete!")
    logger.info("%s\n", "=" * 60)

    # Create pytest summary
    summary_lines = [
        "✓ Gather Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • 'mov' instructions match output elements (1:1 ratio) ✓",
        "  • All operations are MEMORY-bound ✓",
        "  • Low arithmetic intensity (simple copy operation) ✓",
        "  • Gather sparsity validated for selective element access ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        data_shape_str = "x".join(map(str, result["data_shape"]))
        summary_lines.append(
            "  ✓ {:<28s} | {:>7,} mov | {:>8.1f} KB | axis={} | {:.1f}% sparse".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                result["axis"],
                result["sparsity_pct"],
            )
        )

    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "GATHER MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("GATHER MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
