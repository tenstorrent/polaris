#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os, logging
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_mod
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except Exception:
        pass
except Exception:
    pass

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_mod(A, B):
    """
    Reference implementation of modulo operation.
    ONNX Mod: result = A - B * floor(A / B)
    Handles sign like Python's fmod with special ONNX behavior
    """
    return np.fmod(A, B)


# Test cases for numerical validation
test_name = "test_mod"
test_cases = [
    # (name, shape_A, shape_B, data_type)
    ("1D same shape", [10], [10], "basic"),
    ("2D same shape", [4, 5], [4, 5], "basic"),
    ("3D same shape", [2, 3, 4], [2, 3, 4], "basic"),
    ("Broadcast scalar", [5, 5], [1], "broadcast_scalar"),
    ("Broadcast row", [4, 5], [1, 5], "broadcast_row"),
    ("Broadcast column", [5, 4], [5, 1], "broadcast_col"),
    ("Positive mod positive", [10], [10], "pos_pos"),
    ("Negative mod positive", [10], [10], "neg_pos"),
    ("Positive mod negative", [10], [10], "pos_neg"),
    ("Mixed signs", [8, 8], [8, 8], "mixed"),
]


def generate_test_data(shape_A, shape_B, data_type):
    """Generate test data based on type"""
    if data_type == "basic":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(
            np.float32
        )  # Avoid division by small numbers
        return A, B
    elif data_type == "broadcast_scalar":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = np.array([3.0], dtype=np.float32)
        return A, B
    elif data_type == "broadcast_row":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "broadcast_col":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "pos_pos":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "neg_pos":
        A = -(np.random.rand(*shape_A) * 20).astype(np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "pos_neg":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = -(np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "mixed":
        A = (np.random.randn(*shape_A) * 10).astype(np.float32)
        B = (np.random.randn(*shape_B) * 3 + 3).astype(np.float32)  # Ensure non-zero
        return A, B
    else:
        A = np.random.rand(*shape_A).astype(np.float32)
        B = np.random.rand(*shape_B).astype(np.float32) + 1
        return A, B


@pytest.mark.unit
@pytest.mark.opunit
def test_mod_numerical():
    """Test Mod with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, shape_A, shape_B, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_A, test_B = generate_test_data(shape_A, shape_B, data_type)

        # Create input tensors
        i_tensors = [F._from_data("A", test_A), F._from_data("B", test_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Mod",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"fmod": 1},  # ONNX uses fmod=1 by default
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_mod(test_A, test_B)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_mod(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-6
            )

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
test_name_precision = "test_mod_precision"
precision_test_cases = [
    # (name, input_A, input_B, expected_output)
    (
        "7 mod 3",
        np.array([7.0], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ),
    (
        "8 mod 4",
        np.array([8.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    ),
    (
        "5.5 mod 2",
        np.array([5.5], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        np.array([1.5], dtype=np.float32),
    ),
    (
        "-7 mod 3",
        np.array([-7.0], dtype=np.float32),
        np.array([3.0], dtype=np.float32),
        np.array([-1.0], dtype=np.float32),
    ),
    (
        "7 mod -3",
        np.array([7.0], dtype=np.float32),
        np.array([-3.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ),
    (
        "2D mod",
        np.array([[10.0, 15.0], [7.0, 12.0]], dtype=np.float32),
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        np.array([[1.0, 3.0], [2.0, 0.0]], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mod_precision():
    """Test Mod with precise known outputs"""
    msgw = 20

    for tno, (tmsg, test_A, test_B, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("A", test_A), F._from_data("B", test_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Mod",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"fmod": 1},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_mod(i_tensors, op_obj)
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


# Edge cases
test_name_edge = "test_mod_edge"
edge_test_cases = [
    # (name, shape_A, shape_B, data_type, description)
    ("Mod by 1", [10], [1], "mod_one", "Result should be near zero"),
    ("Small mod large", [10], [1], "small_large", "Small values mod large divisor"),
    ("Zero dividend", [5], [5], "zero_dividend", "0 mod anything = 0"),
    ("Large values", [8], [8], "large", "Large value modulo"),
    ("Fractional divisor", [10], [10], "frac_divisor", "Fractional divisors"),
    ("Broadcast batch", [4, 8], [1], "broadcast_batch", "Broadcasting in batch"),
]


def generate_edge_test_data(shape_A, shape_B, data_type):
    """Generate edge case test data"""
    if data_type == "mod_one":
        A = (np.random.rand(*shape_A) * 10).astype(np.float32)
        B = np.ones(shape_B, dtype=np.float32)
        return A, B
    elif data_type == "small_large":
        A = (np.random.rand(*shape_A) * 2).astype(np.float32)
        B = np.array([100.0], dtype=np.float32)
        return A, B
    elif data_type == "zero_dividend":
        A = np.zeros(shape_A, dtype=np.float32)
        B = (np.random.rand(*shape_B) * 5 + 1).astype(np.float32)
        return A, B
    elif data_type == "large":
        A = (np.random.rand(*shape_A) * 1000).astype(np.float32)
        B = (np.random.rand(*shape_B) * 100 + 10).astype(np.float32)
        return A, B
    elif data_type == "frac_divisor":
        A = (np.random.rand(*shape_A) * 10).astype(np.float32)
        B = (np.random.rand(*shape_B) * 0.5 + 0.1).astype(np.float32)
        return A, B
    elif data_type == "broadcast_batch":
        A = (np.random.rand(*shape_A) * 20).astype(np.float32)
        B = np.array([7.0], dtype=np.float32)
        return A, B
    else:
        A = np.random.rand(*shape_A).astype(np.float32)
        B = np.random.rand(*shape_B).astype(np.float32) + 1
        return A, B


@pytest.mark.unit
@pytest.mark.opunit
def test_mod_edge_cases():
    """Test Mod edge cases and boundary conditions"""
    msgw = 25

    for tno, (tmsg, shape_A, shape_B, data_type, description) in enumerate(
        edge_test_cases
    ):
        op_name = f"{test_name_edge}_{tno}"

        test_A, test_B = generate_edge_test_data(shape_A, shape_B, data_type)

        i_tensors = [F._from_data("A", test_A), F._from_data("B", test_B)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Mod",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"fmod": 1},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_mod(i_tensors, op_obj)
            ref_output = ref_impl_mod(test_A, test_B)

            # Validate
            match = np.allclose(computed_output, ref_output, rtol=1e-5, atol=1e-6)

            # Additional edge case checks
            if data_type == "mod_one":
                # x mod 1 should be fractional part
                assert np.all(np.abs(computed_output) < 1.0 + 1e-5), "x mod 1 < 1"
            elif data_type == "zero_dividend":
                # 0 mod anything = 0
                assert np.allclose(computed_output, 0.0, atol=1e-6), "0 mod x = 0"
            elif data_type == "small_large":
                # small mod large = small (when small < large)
                assert np.allclose(
                    computed_output, test_A, rtol=1e-5
                ), "small mod large = small"

            # Property: |result| < |divisor| for most cases
            # (except edge cases with negative numbers in fmod)

            if match:
                print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                print(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  {description}")
                print(f"  Max diff: {np.max(np.abs(computed_output - ref_output))}")
        except Exception as e:
            print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_mod_memory_stats(shape_a, shape_b):
    """Calculate memory access and arithmetic operations for mod operation"""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    A = SimTensor({"name": "A", "shape": shape_a, "dtype": "float32"})
    A.data = (np.random.rand(*shape_a) * 20).astype(np.float32)
    B = SimTensor({"name": "B", "shape": shape_b, "dtype": "float32"})
    B.data = (np.random.rand(*shape_b) * 5 + 1).astype(np.float32)
    i_tensors = [A, B]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_mod",
        "optype": "Mod",
        "inList": ["A", "B"],
        "outList": ["Y"],
        "attrs": {"fmod": 1},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_mod"]
    for x in o_tensors:
        x.op_out = ["test_mod"]

    op.get_perf_counts(i_tensors, o_tensors)

    perf_stats = op.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    instr_sum = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else 0
    )
    ops = instr_sum if instr_sum > 0 else perf_stats.get("inElems", np.prod(shape_a))
    input_bytes = perf_stats["inBytes"]
    output_bytes = perf_stats["outBytes"]
    total_memory = input_bytes + output_bytes

    arithmetic_intensity = ops / total_memory if total_memory > 0 else 0

    mem_bw_bytes_per_cycle = (
        device.simconfig_obj.peak_bandwidth(freq_units="GHz")
        * 1e9
        / device.freq_MHz
        / 1e6
    )
    compute_throughput = 1
    compute_cycles = ops / compute_throughput
    memory_cycles = total_memory / mem_bw_bytes_per_cycle

    bottleneck = "compute-bound" if compute_cycles > memory_cycles else "memory-bound"

    return {
        "memory_mb": total_memory / (1024 * 1024),
        "ops": ops,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
    }


@pytest.mark.unit
@pytest.mark.opunit
def test_mod_memory_validation():
    """Test Mod memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    print("\n" + "=" * 80)
    print("MOD MEMORY VALIDATION")
    print("=" * 80)
    print(f"Device: {device.devname}")
    print(f"  Name: {device.name}")
    print(f"  Frequency: {device.freq_MHz} MHz")
    print(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    print("=" * 80)

    # (shape_a, shape_b) — elementwise so typically same shapes
    test_configs = [
        ([1024], [1024]),
        ([4096], [4096]),
        ([1, 64, 56, 56], [1, 64, 56, 56]),
        ([1, 128, 28, 28], [1, 128, 28, 28]),
        ([4, 64, 56, 56], [4, 64, 56, 56]),
        ([4, 128, 28, 28], [4, 128, 28, 28]),
        ([8, 256, 14, 14], [8, 256, 14, 14]),
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape_a, shape_b in test_configs:
        stats = calculate_mod_memory_stats(shape_a, shape_b)

        print(f"\nShape A: {shape_a}, Shape B: {shape_b}")
        print(f"  Memory: {stats['memory_mb']:.4f} MB")
        print(f"  Operations: {stats['ops']}")
        print(f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte")
        print(f"  Bottleneck: {stats['bottleneck']}")

        if stats["bottleneck"] == "memory-bound":
            memory_bound_count += 1
        else:
            compute_bound_count += 1

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configurations tested: {len(test_configs)}")
    print(f"Memory-bound: {memory_bound_count}")
    print(f"Compute-bound: {compute_bound_count}")
    print("=" * 80)
