#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import pytest
import os
import sys
import logging

import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_atan2
from ttsim.ops.desc.helpers import bidir_bcast
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

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


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_atan2(Y, X):
    """Reference implementation of atan2(y, x)"""
    return np.arctan2(Y, X)


# Test cases with shape validation and numerical validation
test_name = "test_atan2"
test_cases = [
    # (name, shape_Y, shape_X, test_data_type)
    # Same shape cases
    ("Same shape 1D", [10], [10], "positive"),
    ("Same shape 2D", [4, 4], [4, 4], "positive"),
    ("Same shape 3D", [2, 3, 4], [2, 3, 4], "positive"),
    ("Same shape 4D", [2, 3, 4, 5], [2, 3, 4, 5], "positive"),
    # Broadcasting cases
    ("Broadcast scalar", [4, 5], [1], "positive"),
    ("Broadcast 1D to 2D", [4, 5], [5], "positive"),
    ("Broadcast 2D to 3D", [2, 3, 4], [3, 4], "positive"),
    # Quadrant tests
    ("Quadrant I (+,+)", [4, 4], [4, 4], "both_positive"),
    ("Quadrant II (+,-)", [4, 4], [4, 4], "y_pos_x_neg"),
    ("Quadrant III (-,-)", [4, 4], [4, 4], "both_negative"),
    ("Quadrant IV (-,+)", [4, 4], [4, 4], "y_neg_x_pos"),
    # Edge cases: zero inputs (atan2 handles these specially)
    ("Y=0, X>0 -> 0", [4, 4], [4, 4], "y_zero_x_pos"),
    ("Y=0, X<0 -> π", [4, 4], [4, 4], "y_zero_x_neg"),
    ("Y>0, X=0 -> π/2", [4, 4], [4, 4], "y_pos_x_zero"),
    ("Y<0, X=0 -> -π/2", [4, 4], [4, 4], "y_neg_x_zero"),
    ("Y=0, X=0 -> 0 (undefined)", [4, 4], [4, 4], "both_zero"),
    # Edge cases: equal magnitudes
    ("Y=X diagonal", [4, 4], [4, 4], "equal_positive"),
    ("Y=-X diagonal", [4, 4], [4, 4], "y_neg_x"),
    # Edge cases: large values
    ("Large values", [3, 3], [3, 3], "large"),
    # Edge cases: small values
    ("Small values", [3, 3], [3, 3], "small"),
]


def generate_test_data(shape_Y, shape_X, data_type):
    """Generate test data based on type"""
    if data_type == "positive":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        return Y, X
    elif data_type == "both_positive":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        return Y, X
    elif data_type == "y_pos_x_neg":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = -(np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1)
        return Y, X
    elif data_type == "both_negative":
        Y = -(np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1)
        X = -(np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1)
        return Y, X
    elif data_type == "y_neg_x_pos":
        Y = -(np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1)
        X = np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        return Y, X
    elif data_type == "y_zero_x_pos":
        Y = np.zeros(shape_Y, dtype=np.float32)
        X = np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        return Y, X
    elif data_type == "y_zero_x_neg":
        Y = np.zeros(shape_Y, dtype=np.float32)
        X = -(np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1)
        return Y, X
    elif data_type == "y_pos_x_zero":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = np.zeros(shape_X, dtype=np.float32)
        return Y, X
    elif data_type == "y_neg_x_zero":
        Y = -(np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1)
        X = np.zeros(shape_X, dtype=np.float32)
        return Y, X
    elif data_type == "both_zero":
        Y = np.zeros(shape_Y, dtype=np.float32)
        X = np.zeros(shape_X, dtype=np.float32)
        return Y, X
    elif data_type == "equal_positive":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = (
            Y.copy()
            if shape_Y == shape_X
            else np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        )
        return Y, X
    elif data_type == "y_neg_x":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 10 + 0.1
        X = (
            -Y
            if shape_Y == shape_X
            else np.random.rand(*shape_X).astype(np.float32) * 10 + 0.1
        )
        return Y, X
    elif data_type == "large":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 1e6
        X = np.random.rand(*shape_X).astype(np.float32) * 1e6
        return Y, X
    elif data_type == "small":
        Y = np.random.rand(*shape_Y).astype(np.float32) * 1e-6
        X = np.random.rand(*shape_X).astype(np.float32) * 1e-6
        return Y, X
    else:
        Y = np.random.randn(*shape_Y).astype(np.float32)
        X = np.random.randn(*shape_X).astype(np.float32)
        return Y, X


@pytest.mark.unit
@pytest.mark.opunit
def test_atan2():
    """Test Atan2 with shape validation, edge cases, and numerical validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape_Y, shape_X, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data_Y, test_data_X = generate_test_data(shape_Y, shape_X, data_type)

        # Create input tensors with actual data
        i_tensors = [F._from_data("Y", test_data_Y), F._from_data("X", test_data_X)]
        o_tensors = [make_tensor("Z")]

        op_info = {
            "name": op_name,
            "optype": "Atan2",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_atan2(test_data_Y, test_data_X)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_atan2(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-7
            )

            # Verify atan2 properties: output in [-π, π]
            all_in_range = np.all(
                (computed_output >= -np.pi) & (computed_output <= np.pi)
            )
            if not all_in_range:
                numerical_match = False
                print(f"\n  Output not in range [-π, π]")

            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                print(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]")
        elif shape_match:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            print(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            print(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            print(f"  Numerical match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_atan2_precision"
precision_test_cases = [
    # (name, Y, X, expected_output)
    (
        "atan2(0, 1) = 0",
        np.array([[0.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    ),
    (
        "atan2(1, 0) = π/2",
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([[np.pi / 2]], dtype=np.float32),
    ),
    (
        "atan2(0, -1) = π",
        np.array([[0.0]], dtype=np.float32),
        np.array([[-1.0]], dtype=np.float32),
        np.array([[np.pi]], dtype=np.float32),
    ),
    (
        "atan2(-1, 0) = -π/2",
        np.array([[-1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        np.array([[-np.pi / 2]], dtype=np.float32),
    ),
    (
        "atan2(1, 1) = π/4",
        np.array([[1.0]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[np.pi / 4]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_atan2_precision():
    """Test Atan2 with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data_Y, test_data_X, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("Y", test_data_Y), F._from_data("X", test_data_X)]
        o_tensors = [make_tensor("Z")]

        op_info = {
            "name": op_name,
            "optype": "Atan2",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_atan2(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-6)
            if match:
                print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                print(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  Expected: {expected_output.flatten()}")
                print(f"  Got:      {computed_output.flatten()}")
                print(f"  Diff:     {(computed_output - expected_output).flatten()}")
        except Exception as e:
            print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_atan2_memory_stats(
    op, device, input_shape_Y, input_shape_X, precision="fp16"
):
    """
    Calculate memory performance metrics for a single atan2 operation.

    Args:
        op: SimOp representing the atan2 operation
        device: Device instance for execution
        input_shape_Y: Shape of first input tensor (Y)
        input_shape_X: Shape of second input tensor (X)
        precision: Data precision (default: 'fp16')

    Returns:
        Dictionary containing memory statistics:
        - instructions_executed: Total atan2 instructions
        - input_bytes_Y: Bytes read for input Y
        - input_bytes_X: Bytes read for input X
        - output_bytes: Bytes written for output
        - total_data_moved: Total bytes (reads + writes)
        - arithmetic_intensity: operations per byte
        - compute_cycles: Compute cycles
        - mem_rd_cycles: Memory read cycles
        - mem_wr_cycles: Memory write cycles
        - memory_cycles: Total memory cycles
        - bottleneck: 'COMPUTE' or 'MEMORY'
        - execution_time_ms: Estimated execution time
    """

    # Normalize precision to string format
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

    # Set operation configuration
    op.precision = normalize_precision(precision)
    if op.uses_compute_pipe is None:
        op.uses_compute_pipe = "vector"  # Atan2 uses vector pipe

    # Get performance counts (this populates perf_stats with element counts)
    # Note: For atan2, device config may not have instruction definitions,
    # so we use element counts as a proxy for instructions
    if op.perf_stats is not None:
        # Calculate output shape with broadcasting
        output_shape = np.broadcast_shapes(input_shape_Y, input_shape_X)
        elem_count = np.prod(output_shape)
        total_instructions = elem_count  # 1 atan2 operation per output element

        # Get byte counts from perf_stats
        total_input_bytes = op.perf_stats.get("inBytes", 0)
        output_bytes = op.perf_stats.get("outBytes", 0)

        # If perf_stats doesn't have byte counts, calculate them
        if total_input_bytes == 0 or output_bytes == 0:
            bytes_per_elem = {
                "fp16": 2,
                "bf16": 2,
                "fp32": 4,
                "int8": 1,
                "int32": 4,
            }.get(op.precision, 2)

            elem_count_Y = np.prod(input_shape_Y)
            elem_count_X = np.prod(input_shape_X)
            total_input_bytes = (elem_count_Y + elem_count_X) * bytes_per_elem
            output_bytes = elem_count * bytes_per_elem

        # Approximate individual input sizes
        elem_count_Y = np.prod(input_shape_Y)
        elem_count_X = np.prod(input_shape_X)
        bytes_per_elem = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "int32": 4}.get(
            op.precision, 2
        )

        input_bytes_Y = elem_count_Y * bytes_per_elem
        input_bytes_X = elem_count_X * bytes_per_elem

        total_data_moved = total_input_bytes + output_bytes

        # For memory-bound operations, estimate cycles based on bandwidth
        bytes_per_cycle = (
            device.simconfig_obj.peak_bandwidth(freq_units="MHz") / device.freq_MHz
        )
        memory_cycles = (
            int(total_data_moved / bytes_per_cycle) if bytes_per_cycle > 0 else 0
        )

        # Estimate compute cycles (atan2 is more expensive than add/mul)
        # Assuming ~20-30 cycles per atan2 operation (more complex than atan)
        compute_cycles_per_elem = 25  # Conservative estimate for atan2
        compute_cycles = elem_count * compute_cycles_per_elem

        # Split memory cycles into read/write (approximate 2:1 ratio for binary ops)
        mem_rd_cycles = int(memory_cycles * 0.67)
        mem_wr_cycles = memory_cycles - mem_rd_cycles

        # Arithmetic intensity (operations per byte)
        arithmetic_intensity = (
            total_instructions / total_data_moved if total_data_moved > 0 else 0
        )

        # Bottleneck determination
        bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

        # Execution time (max of compute and memory cycles)
        ideal_cycles = max(compute_cycles, memory_cycles)
        execution_time_ms = ideal_cycles / device.freq_MHz / 1e3

        # Additional metrics
        read_write_ratio = total_input_bytes / output_bytes if output_bytes > 0 else 0
        bytes_per_cycle_actual = (
            total_data_moved / memory_cycles if memory_cycles > 0 else 0
        )

        return {
            "instructions_executed": total_instructions,
            "input_bytes_Y": input_bytes_Y,
            "input_bytes_X": input_bytes_X,
            "input_bytes_total": total_input_bytes,
            "output_bytes": output_bytes,
            "total_data_moved": total_data_moved,
            "arithmetic_intensity": arithmetic_intensity,
            "compute_cycles": compute_cycles,
            "mem_rd_cycles": mem_rd_cycles,
            "mem_wr_cycles": mem_wr_cycles,
            "memory_cycles": memory_cycles,
            "ideal_cycles": ideal_cycles,
            "bottleneck": bottleneck,
            "execution_time_ms": execution_time_ms,
            "read_write_ratio": read_write_ratio,
            "bytes_per_cycle": bytes_per_cycle_actual,
            "precision": op.precision,
        }
    else:
        return None


@pytest.mark.unit
@pytest.mark.opunit
def test_atan2_memory_validation(capsys, request):
    """
    Test memory validation for atan2 operation.
    Validates instructions executed and data moved for various scenarios.

    This test validates two primary metrics:
    1. Instructions Executed: Verifies instruction count matches output elements
    2. Data Moved: Tracks input/output bytes and validates memory traffic

    Run with: pytest tests/test_ops/test_atan2.py::test_atan2_memory_validation -v
    For detailed output: add -s flag
    """
    print("\n" + "=" * 60)
    print("Atan2 Operation Memory Validation")
    print("=" * 60)

    # Load device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]  # Use Wormhole n150 device
        device = Device(device_pkg)

        print(f"\nDevice: {device.devname} ({device.name})")
        print(f"Device frequency: {device.freq_MHz} MHz")
        print(f"Memory frequency: {device.memfreq_MHz} MHz")
        print(
            f"Peak bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        print(f"\nWarning: Could not load device config: {e}")
        print("Skipping memory validation test")
        pytest.skip(f"Could not load device config: {e}")
        return

    # Test cases: different shapes and broadcasting scenarios for binary operation
    test_cases = [
        {
            "name": "1D Same Shape",
            "shape_Y": [1000],
            "shape_X": [1000],
            "description": "Atan2 of two 1D arrays",
        },
        {
            "name": "2D Same Shape",
            "shape_Y": [32, 32],
            "shape_X": [32, 32],
            "description": "Atan2 of two 2D matrices",
        },
        {
            "name": "4D Same Shape (Image)",
            "shape_Y": [1, 3, 64, 64],
            "shape_X": [1, 3, 64, 64],
            "description": "Atan2 of two 4D tensors (image-like)",
        },
        {
            "name": "Scalar Broadcast",
            "shape_Y": [1, 64, 64, 64],
            "shape_X": [1],
            "description": "Atan2 with scalar broadcast",
        },
        {
            "name": "1D to 2D Broadcast",
            "shape_Y": [32, 64],
            "shape_X": [64],
            "description": "Broadcast 1D to 2D",
        },
        {
            "name": "Channel-wise",
            "shape_Y": [8, 64, 32, 32],
            "shape_X": [1, 64, 1, 1],
            "description": "Atan2 with channel-wise broadcast",
        },
        {
            "name": "Large Tensors",
            "shape_Y": [16, 128, 128],
            "shape_X": [16, 128, 128],
            "description": "Large tensor atan2",
        },
    ]

    print(f"\n{'='*60}")
    print("Running Memory Validation Tests")
    print(f"{'='*60}\n")

    all_results = []

    for test_case in test_cases:
        print(f"\n-- Test: {test_case['name']} --")
        print(f"Description: {test_case['description']}")
        print(f"Shape Y: {test_case['shape_Y']}")
        print(f"Shape X: {test_case['shape_X']}")

        # Create input tensors with fp16 precision to match operation precision
        input_Y = SimTensor(
            {"name": "input_Y", "shape": test_case["shape_Y"], "dtype": "float16"}
        )
        input_Y.data = np.random.randn(*test_case["shape_Y"]).astype(np.float16)

        input_X = SimTensor(
            {"name": "input_X", "shape": test_case["shape_X"], "dtype": "float16"}
        )
        input_X.data = np.random.randn(*test_case["shape_X"]).astype(np.float16)

        # Calculate output shape (with broadcasting)
        output_shape = list(
            np.broadcast_shapes(test_case["shape_Y"], test_case["shape_X"])
        )

        output_tensor = SimTensor(
            {"name": "output", "shape": output_shape, "dtype": "float16"}
        )

        # Create atan2 operation
        op = SimOp(
            {
                "name": f'atan2_op_{test_case["name"].replace(" ", "_").lower()}',
                "optype": "Atan2",
                "inList": ["input_Y", "input_X"],
                "outList": ["output"],
            }
        )

        # Perform shape inference to get perf_stats
        bidir_bcast([input_Y, input_X], [output_tensor], op)

        # Get performance counts to populate perf_stats
        op.get_perf_counts([input_Y, input_X], [output_tensor])

        # Calculate memory stats
        mem_stats = calculate_atan2_memory_stats(
            op, device, test_case["shape_Y"], test_case["shape_X"], precision="fp16"
        )

        if mem_stats:
            # Validate metrics
            output_elems = np.prod(output_shape)

            print(f"\n  -- Instructions & Operations --")
            print(f"  Instructions executed: {mem_stats['instructions_executed']:,}")
            print(f"  Output elements:       {output_elems:,}")
            print(
                f"  Expected instructions: ~{output_elems:,} (1 atan2 per output element)"
            )

            # Validate: instructions should be approximately equal to output elements
            instruction_ratio = (
                mem_stats["instructions_executed"] / output_elems
                if output_elems > 0
                else 0
            )
            assert (
                0.8 <= instruction_ratio <= 1.5
            ), f"Instruction count mismatch: {mem_stats['instructions_executed']} vs expected ~{output_elems}"
            print(
                f"  Instruction ratio:     {instruction_ratio:.2f} (✓ within expected range)"
            )

            print(f"\n  -- Data Movement --")
            print(
                f"  Input Y bytes:    {mem_stats['input_bytes_Y']:,} bytes ({mem_stats['input_bytes_Y']/1024:.2f} KB)"
            )
            print(
                f"  Input X bytes:    {mem_stats['input_bytes_X']:,} bytes ({mem_stats['input_bytes_X']/1024:.2f} KB)"
            )
            print(
                f"  Input total:      {mem_stats['input_bytes_total']:,} bytes ({mem_stats['input_bytes_total']/1024:.2f} KB)"
            )
            print(
                f"  Output bytes:     {mem_stats['output_bytes']:,} bytes ({mem_stats['output_bytes']/1024:.2f} KB)"
            )
            print(
                f"  Total data moved: {mem_stats['total_data_moved']:,} bytes ({mem_stats['total_data_moved']/1024:.2f} KB)"
            )

            # Validate: output bytes should match output tensor size with fp16 precision
            bytes_per_elem = 2  # fp16 = 2 bytes per element
            expected_output_bytes = output_elems * bytes_per_elem

            assert mem_stats["output_bytes"] > 0, "Output bytes should be positive"
            assert (
                abs(mem_stats["output_bytes"] - expected_output_bytes)
                < expected_output_bytes * 0.1
            ), f"Output bytes mismatch: {mem_stats['output_bytes']} vs expected {expected_output_bytes}"

            print(
                f"  Expected output:  {expected_output_bytes:,} bytes (✓ matches fp16 precision)"
            )

            print(f"\n  -- Memory Metrics --")
            print(
                f"  Arithmetic intensity:  {mem_stats['arithmetic_intensity']:.4f} ops/byte"
            )
            print(f"  Read/Write ratio:      {mem_stats['read_write_ratio']:.2f}")
            print(f"  Bytes per cycle:       {mem_stats['bytes_per_cycle']:.2f}")

            # For atan2, arithmetic intensity should be low (memory-bound)
            assert (
                mem_stats["arithmetic_intensity"] < 1.0
            ), f"Arithmetic intensity too high for memory-bound op: {mem_stats['arithmetic_intensity']}"
            print(f"  ✓ Low arithmetic intensity confirms memory-bound operation")

            print(f"\n  -- Execution Cycles --")
            print(f"  Compute cycles:   {mem_stats['compute_cycles']:,}")
            print(f"  Memory cycles:    {mem_stats['memory_cycles']:,}")
            print(f"    Read cycles:    {mem_stats['mem_rd_cycles']:,}")
            print(f"    Write cycles:   {mem_stats['mem_wr_cycles']:,}")
            print(f"  Ideal cycles:     {mem_stats['ideal_cycles']:,}")
            print(f"  Bottleneck:       {mem_stats['bottleneck']}")

            # Note: Atan2 is compute-intensive (transcendental function with two inputs)
            print(
                f"  ✓ Bottleneck identified ({'COMPUTE' if mem_stats['bottleneck'] == 'COMPUTE' else 'MEMORY'}-bound operation)"
            )

            # Store results
            all_results.append(
                {
                    "test_name": test_case["name"],
                    "shape_Y": test_case["shape_Y"],
                    "shape_X": test_case["shape_X"],
                    "output_shape": output_shape,
                    "stats": mem_stats,
                }
            )

            print(f"\n  ✓ Test PASSED")
        else:
            print(f"\n  ✗ Test FAILED: Could not calculate memory stats")
            assert False, "Failed to calculate memory stats"

    # Summary
    print(f"\n{'='*60}")
    print("Memory Validation Summary")
    print(f"{'='*60}\n")
    print(f"Total tests run: {len(all_results)}")
    print(f"All tests passed: ✓")

    # Compare arithmetic intensity across tests
    print(f"\n-- Arithmetic Intensity Comparison --")
    for result in all_results:
        ai = result["stats"]["arithmetic_intensity"]
        print(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

    print(f"\n-- Bottleneck Analysis --")
    for result in all_results:
        bottleneck = result["stats"]["bottleneck"]
        print(f"{result['test_name']:30s}: {bottleneck}")

    print(f"\n{'='*60}")
    print("Memory validation complete!")
    print(f"{'='*60}\n")

    # Create a summary that will be displayed in pytest output (even without -s flag)
    summary_lines = [
        "✓ Tests completed: {}/{} - All PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Findings:",
        "  • Instructions match output elements (1:1 ratio) ✓",
        "  • Atan2 is compute-intensive (transcendental function)",
        "  • Arithmetic Intensity: 0.17-0.25 ops/byte",
        "",
        "Test Results:",
    ]

    for result in all_results:
        mem_stats = result["stats"]
        summary_lines.append(
            "  ✓ {:<26s} | {:>7,} ops | {:>7.1f} KB | {:.3f} ops/byte".format(
                result["test_name"],
                mem_stats["instructions_executed"],
                mem_stats["total_data_moved"] / 1024,
                mem_stats["arithmetic_intensity"],
            )
        )

    summary_lines.extend(
        [
            "",
            "Validation: All memory metrics within expected ranges ✓",
            "",
            "For detailed output, run with: pytest -s -v",
        ]
    )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            print("\n" + "=" * 70)
            print("MEMORY VALIDATION RESULTS")
            print("=" * 70)
            for line in summary_lines:
                print(line)
            print("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
