#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
import logging
from loguru import logger

# Silence the ttsim-related log messages
try:
    from loguru import logger as _loguru_logger

    _loguru_logger.disable("ttsim")
except ImportError:
    pass
from pathlib import Path

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_layernorm

try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Layer normalization's reference implementation
def _layer_normalization(X, W, B, axis=-1, epsilon=1e-5):  # type: ignore
    X_shape = X.shape
    X_rank = len(X_shape)
    if axis < 0:
        # If axis = -1 and rank of X is 4,
        # the axis is changed to -1 + 4 = 3,
        # which means the last axis.
        axis = axis + X_rank
    unsqueezed_rank = X_rank - axis
    reduction_shape = X_shape[0:axis] + (1,) * unsqueezed_rank

    # Parameter used to convert N-D tensor layer
    # normalization to equivalent 2-D matirx operations.
    row_number = 1
    col_number = 1
    for i in range(X_rank):
        if i < axis:
            row_number *= X_shape[i]
        else:
            col_number *= X_shape[i]

    # After reshaping input tensor X into a matrix,
    # layer normalization is equivalent to conducting
    # standardization on each column vector (s.t. each
    # column has zero mean and unit variance).
    x_mat = np.reshape(X, (row_number, col_number))
    # This computes mean for every x_mat's column.
    x_mean = np.sum(x_mat, axis=1, keepdims=True) / col_number
    x_diff = x_mat - x_mean
    x_squared_diff = x_diff * x_diff
    # This computes variance for every x_mat's column.
    variance = np.sum(x_squared_diff, axis=1, keepdims=True) / col_number
    variance_eps = variance + epsilon
    std_dev = np.sqrt(variance_eps)
    inv_std_dev = np.reciprocal(std_dev)
    # Standardization step. y_mat is zero-mean and unit-variance.
    y_mat = x_diff * inv_std_dev
    # Apply affine transform on normalization outcome.
    # W is linear coefficient while B is bias.
    Y = np.reshape(y_mat, X_shape) * W + B
    # Matrix-level operations' outputs should be reshaped
    # to compensate the initial tensor-to-matrix reshape.
    X_mean = np.reshape(x_mean, reduction_shape)
    X_inv_std_dev = np.reshape(inv_std_dev, reduction_shape)

    return Y, X_mean, X_inv_std_dev


def calculate_normalized_shape(X_shape, axis):  # type: ignore
    X_rank = len(X_shape)
    if axis < 0:
        axis = axis + X_rank
    return X_shape[axis:]


# Test cases
test_name = "test_layernorm"
test_cases = [
    {
        "name": f"test_layer_normalization_4d",
        "x": [2, 3, 4, 5],
        "in": ["X", "W", "B"],
        "out": ["Y", "Mean", "InvStdDev"],
    },
    {
        "name": "test_layer_normalization_default_axis",
        "x": [2, 3, 4, 5],
        "in": ["X", "W", "B"],
        "out": ["Y", "Mean", "InvStdDev"],
    },
    {
        "name": "test_layer_normalization_2d",
        "x": [3, 4],
        "in": ["X", "W", "B"],
        "out": ["Y", "Mean", "InvStdDev"],
    },
    {
        "name": f"test_layer_normalization_3d_epsilon",
        "x": [2, 3, 5],
        "in": ["X", "W", "B"],
        "out": ["Y", "Mean", "InvStdDev"],
        "eps": 1e-1,
    },
]


@pytest.mark.unit
@pytest.mark.opunit
def test_layernorm():
    for trec in test_cases:
        tname = trec["name"]  # type: ignore
        if tname.endswith("default_axis"):
            axes = [-1]
            names = [tname]
        else:
            xrank = len(trec["x"])  # type: ignore
            axes = [i for i in range(xrank)]
            axes += [i - xrank for i in range(xrank)]
            names = [
                f"{tname}_neg_axis_{-a}" if a < 0 else f"{tname}_axis_{a}" for a in axes
            ]
        trec["axes"] = axes  # type: ignore
        trec["names"] = names  # type: ignore
    msgw = max([len(y) for x in test_cases for y in x["names"]])  # type: ignore
    for tno, trec in enumerate(test_cases):
        for cno, axis in enumerate(trec["axes"]):  # type: ignore
            test_name = trec["names"][cno]  # type: ignore
            op_name = f"{test_name}_{tno}_{cno}"

            XShape = trec["x"]  # type: ignore
            normalized_shape = calculate_normalized_shape(XShape, axis)
            X = np.random.randn(*XShape).astype(np.float32)
            W = np.random.randn(*normalized_shape).astype(np.float32)
            B = np.random.randn(*normalized_shape).astype(np.float32)
            attrs = {"axis": axis}
            if "eps" in trec:  # type: ignore
                eps = trec["eps"]  # type: ignore
                attrs["epsilon"] = eps
                Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, eps)
            else:
                Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)
            o0Shape = list(Y.shape)
            o1Shape = list(mean.shape)
            o2Shape = list(inv_std_dev.shape)

            i_tensors = [
                F._from_shape("X", XShape, np_dtype=np.float32),  # data
                F._from_shape("W", normalized_shape, np_dtype=np.float32),  # scale
                F._from_shape("B", normalized_shape, np_dtype=np.float32),  # bias
            ]
            o_tensors = [
                make_tensor("Y"),
                make_tensor("mean"),
                make_tensor("inv_std_dev"),
            ]
            op_info = {
                "name": op_name,
                "optype": "LayerNormalization",
                "inList": [x.name for x in i_tensors],
                "outList": [x.name for x in o_tensors],
                "attrs": attrs,
            }
            op_obj = SimOp(op_info)
            for x in i_tensors:
                x.op_in = [op_name]
            for x in o_tensors:
                x.op_out = [op_name]
            op_obj.get_perf_counts(i_tensors, o_tensors)

            assert (
                o_tensors[0].shape == o0Shape
            ), f"Y shape mismatch: {o_tensors[0].shape} != {o0Shape}"
            assert (
                o_tensors[1].shape == o1Shape
            ), f"mean shape mismatch: {o_tensors[1].shape} != {o1Shape}"
            assert (
                o_tensors[2].shape == o2Shape
            ), f"inv_std_dev shape mismatch: {o_tensors[2].shape} != {o2Shape}"

            eps_str = f"{eps:.2f}" if "eps" in trec else "-"  # type: ignore
            logger.debug(
                f"TEST[{tno:3d}] CASE[{cno:4d}] {test_name:{msgw}s} axis={axis:3d} eps={eps_str} PASS"
            )


def _ref_layernorm(X, scale, bias, axis=-1, epsilon=1e-5):
    """Independent layer normalization reference."""
    if axis < 0:
        axis += X.ndim
    normalized_axes = tuple(range(axis, X.ndim))
    mean = np.mean(X, axis=normalized_axes, keepdims=True)
    var = np.var(X, axis=normalized_axes, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    Y = X_norm * scale
    if bias is not None:
        Y = Y + bias
    return Y


# ---------------------------------------------------------------------------
# Helper to run through SimOp with data
# ---------------------------------------------------------------------------


def _run_layernorm(data, scale, bias=None, axis=-1, epsilon=1e-5, tag="ln"):
    """Run LayerNormalization through SimOp and return (actual, expected, oT)."""
    if bias is not None:
        i_tensors = [
            F._from_data("X", data),
            F._from_data("scale", scale),
            F._from_data("bias", bias),
        ]
        in_names = ["X", "scale", "bias"]
    else:
        i_tensors = [
            F._from_data("X", data),
            F._from_data("scale", scale),
        ]
        in_names = ["X", "scale"]

    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "LayerNormalization",
        "inList": in_names,
        "outList": ["Y"],
        "attrs": {
            "axis": axis,
            "epsilon": epsilon,
        },
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)
    o_tensors[0].data = compute_layernorm(i_tensors, op)

    actual = o_tensors[0].data
    expected = _ref_layernorm(data, scale, bias, axis=axis, epsilon=epsilon)
    return actual, expected, o_tensors[0]


# ===================================================================
# Numerical validation
# ===================================================================


def test_layernorm_memory_validation(capsys, request):
    """
    Test memory validation for LayerNormalization operation.
    Validates instructions and data movement for normalization computation.

    This test validates:
    1. Instructions: Multiple instruction types (add, sub, mul, div, mac, rsqrt)
    2. Data Movement: Reads input+scale+bias, writes output (same size as input)
    3. Normalization: Per-axis statistics computation

    Run with: pytest tests/test_ops/test_layernorm.py::test_layernorm_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 80)
    logger.info("LayerNormalization Operation Memory Validation")
    logger.info("=" * 80)

    # Load device configuration once
    polaris_root = Path(__file__).parent.parent.parent
    config_path = polaris_root / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        logger.debug(f"\nDevice: {device.devname} ({device.name})")
        logger.debug(f"Frequency: {device.freq_MHz} MHz")
        logger.debug(
            f"Peak Bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        pytest.skip(f"Could not load device config: {e}")

    # Test cases: different tensor shapes and normalization axes
    test_cases = [
        {
            "name": "2D Matrix",
            "shape": (32, 64),
            "axis": -1,
            "description": "2D matrix normalizing last dimension",
        },
        {
            "name": "3D Tensor",
            "shape": (8, 16, 32),
            "axis": -1,
            "description": "3D tensor normalizing last dimension",
        },
        {
            "name": "4D Batch",
            "shape": (2, 8, 16, 32),
            "axis": -1,
            "description": "4D tensor (typical for transformers)",
        },
        {
            "name": "2D Wide",
            "shape": (16, 128),
            "axis": 1,
            "description": "2D wide matrix (large normalized dimension)",
        },
        {
            "name": "3D Multi-Axis",
            "shape": (4, 16, 64),
            "axis": 1,
            "description": "3D normalizing middle axis",
        },
    ]

    logger.info(f"\n{'='*80}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        axis = test_case["axis"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Shape: {shape}")
        logger.debug(f"Axis: {axis}")

        # Calculate normalized shape
        ax = axis if axis >= 0 else axis + len(shape)
        norm_shape = shape[ax:]

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(*shape).astype(np.float32)
        scale = np.random.randn(*norm_shape).astype(np.float32)
        bias = np.random.randn(*norm_shape).astype(np.float32)

        # Create operation with fp32 precision for consistency
        x_t = F._from_data("X", X)
        scale_t = F._from_data("scale", scale)
        bias_t = F._from_data("bias", bias)
        out_t = make_tensor("Y")

        op_info = {
            "name": f'layernorm_mem_{test_name.replace(" ", "_")}',
            "optype": "LayerNormalization",
            "inList": [x_t.name, scale_t.name, bias_t.name],
            "outList": [out_t.name],
            "attrs": {"axis": axis, "epsilon": 1e-5},
        }
        op = SimOp(op_info)
        op.precision = "fp32"
        op.uses_compute_pipe = "vector"

        # Get performance counts and execute
        op.get_perf_counts([x_t, scale_t, bias_t], [out_t])
        device.execute_op(op)

        # Verify correctness
        normalized_axes = tuple(range(ax, len(shape)))
        mean = np.mean(X, axis=normalized_axes, keepdims=True)
        var = np.var(X, axis=normalized_axes, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + 1e-5)
        expected_output = X_norm * scale + bias
        actual_output = out_t.data
        np.testing.assert_allclose(
            actual_output,
            expected_output,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"LayerNorm output mismatch for {test_name}",
        )

        # Extract performance stats directly
        perf_stats = op.perf_stats
        num_elements = int(np.prod(shape))
        norm_size = int(np.prod(norm_shape))

        # Validate output shape
        assert out_t.shape == list(
            shape
        ), f"Output shape {out_t.shape} != expected {list(shape)}"
        logger.debug(f"Output shape: {out_t.shape}")
        logger.debug(
            f"Normalized dimensions: {norm_shape} ({norm_size} elements per instance)"
        )

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate expected instruction types
        expected_instr_types = {"add", "sub", "mul", "div", "mac", "rsqrt"}
        actual_instr_types = set(actual_instrs.keys())
        assert actual_instr_types.issubset(
            expected_instr_types
        ), f"Unexpected instructions: {actual_instr_types - expected_instr_types}"

        # Get memory metrics
        input_bytes = perf_stats.get("inBytes", 0)
        output_bytes = perf_stats.get("outBytes", 0)
        total_data_moved = input_bytes + output_bytes

        # Compute cycles
        compute_cycles = op.compute_cycles
        mem_rd_cycles = op.mem_rd_cycles
        mem_wr_cycles = op.mem_wr_cycles
        memory_cycles = mem_rd_cycles + mem_wr_cycles
        ideal_cycles = max(compute_cycles, memory_cycles)

        # Arithmetic intensity
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        logger.info("\n  -- Instructions & Operations --")
        logger.debug(f"  Instructions executed: {total_instructions:,}")
        logger.debug("  Instruction breakdown:")
        for instr_name in sorted(actual_instrs.keys()):
            logger.debug(
                f"    {instr_name:>6s}: {actual_instrs[instr_name]:>12,}"
            )
        logger.debug(f"  Input elements:        {num_elements:,}")
        logger.debug(f"  Output elements:       {num_elements:,}")

        # Validate: LayerNorm has high instruction count (multiple ops per element)
        assert (
            total_instructions >= num_elements
        ), f"Too few instructions: {total_instructions} vs {num_elements} elements"
        logger.debug(
            "  ✓ Complex normalization validated (multiple ops per element)"
        )

        logger.info("\n  -- Data Movement --")
        logger.debug(
            f"  Input bytes:      {input_bytes:,} ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output bytes:     {output_bytes:,} ({output_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved: {total_data_moved:,} ({total_data_moved/1024:.2f} KB)"
        )

        assert (
            abs(output_bytes - num_elements * 4) <= 4
        ), f"Output bytes mismatch: {output_bytes} vs expected {num_elements * 4}"
        logger.debug("  ✓ Input/Output size validated (fp32)")

        logger.info("\n  -- Memory Metrics --")
        logger.debug(
            f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte"
        )
        logger.debug(
            f"  Read/Write ratio:      {input_bytes/output_bytes if output_bytes > 0 else 0:.2f}"
        )
        logger.debug(f"  Instructions/element:  {total_instructions/num_elements:.2f}")

        # LayerNorm has moderate arithmetic intensity (higher than simple ops)
        assert (
            arithmetic_intensity > 0.5
        ), f"Arithmetic intensity too low for LayerNorm: {arithmetic_intensity}"
        logger.debug("  ✓ Moderate arithmetic intensity (higher than simple ops)")

        logger.info("\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {ideal_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")

        # LayerNorm bottleneck can vary based on problem size
        logger.debug(f"  ✓ Bottleneck identified: {bottleneck}")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "axis": axis,
                "norm_size": norm_size,
                "num_elements": num_elements,
                "instructions": total_instructions,
                "instr_breakdown": dict(actual_instrs),
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "total_data_moved": total_data_moved,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
            }
        )

        logger.info("\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*80}\n")
    logger.info(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    logger.info("\n-- Arithmetic Intensity Comparison --")
    logger.info(f"{'Test Name':<30s} {'Ops/Byte':<12s} {'Ops/Element':<15s}")
    logger.info("-" * 60)
    for result in all_results:
        ops_per_elem = result["instructions"] / result["num_elements"]
        logger.debug(
            f"{result['test_name']:<30s} {result['arithmetic_intensity']:<12.4f} {ops_per_elem:>12.2f}"
        )

    # Normalization Configuration Analysis
    logger.info("\n-- Normalization Configuration Analysis --")
    logger.info(f"{'Test Name':<30s} {'Axis':<8s} {'Norm Size':<12s} {'Shape':<25s}")
    logger.info("-" * 80)
    for result in all_results:
        shape_str = "×".join(map(str, result["shape"]))
        logger.debug(
            f"{result['test_name']:<30s} {result['axis']:<8d} {result['norm_size']:>10,}   {shape_str}"
        )

    # Bottleneck Analysis
    logger.info("\n-- Bottleneck Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    logger.info("-" * 80)
    for result in all_results:
        logger.debug(
            f"{result['test_name']:<30s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*80}\n")

    # Create pytest summary
    summary_lines = [
        "✓ LayerNorm Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • Multiple instruction types validated (add, sub, mul, div, mac, rsqrt) ✓",
        "  • Bottleneck analysis completed (varies by problem size) ✓",
        "  • Moderate arithmetic intensity (higher than simple ops) ✓",
        "  • Axis-based normalization verified ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        shape_str = "×".join(map(str, result["shape"]))
        summary_lines.append(
            "  ✓ {:<28s} | {:>9,} ops | {:>8.1f} KB | axis={} | {}".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                result["axis"],
                shape_str,
            )
        )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "LAYERNORM MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("LAYERNORM MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
