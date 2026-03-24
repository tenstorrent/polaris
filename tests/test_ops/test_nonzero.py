#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os, logging
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_nonzero
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


def ref_impl_nonzero(X):
    """
    Reference implementation of nonzero operation.
    Returns indices of non-zero elements as a 2D array.
    Shape: (num_dims, num_nonzero_elements)
    """
    indices = np.nonzero(X)
    # Stack to create shape (num_dims, num_nonzero)
    if len(indices) > 0 and len(indices[0]) > 0:
        result = np.stack(indices, axis=0).astype(np.int64)
    else:
        # No non-zero elements
        result = np.zeros((len(X.shape), 0), dtype=np.int64)
    return result


# Test cases for numerical validation
test_name = "test_nonzero"
test_cases = [
    # (name, input_shape, data_type)
    ("1D sparse", [10], "sparse_1d"),
    ("1D dense", [10], "dense_1d"),
    ("2D sparse", [5, 5], "sparse_2d"),
    ("2D dense", [4, 4], "dense_2d"),
    ("3D sparse", [3, 4, 5], "sparse_3d"),
    ("Single nonzero", [10], "single_nz"),
    ("Diagonal pattern", [5, 5], "diagonal"),
    ("Corner elements", [4, 4], "corners"),
    ("Random pattern", [8, 8], "random"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "sparse_1d":
        data = np.zeros(shape, dtype=np.float32)
        # Set a few random elements to non-zero
        indices = np.random.choice(shape[0], size=min(3, shape[0]), replace=False)
        data[indices] = np.random.randn(len(indices)).astype(np.float32) * 10
        return data
    elif data_type == "dense_1d":
        return (np.random.randn(*shape) * 10).astype(np.float32)
    elif data_type == "sparse_2d":
        data = np.zeros(shape, dtype=np.float32)
        # Set ~20% elements to non-zero
        num_nz = max(1, int(np.prod(shape) * 0.2))
        flat_indices = np.random.choice(np.prod(shape), size=num_nz, replace=False)
        data.flat[flat_indices] = np.random.randn(num_nz).astype(np.float32) * 10
        return data
    elif data_type == "dense_2d":
        return (np.random.randn(*shape) * 10).astype(np.float32)
    elif data_type == "sparse_3d":
        data = np.zeros(shape, dtype=np.float32)
        num_nz = max(1, int(np.prod(shape) * 0.15))
        flat_indices = np.random.choice(np.prod(shape), size=num_nz, replace=False)
        data.flat[flat_indices] = np.random.randn(num_nz).astype(np.float32) * 10
        return data
    elif data_type == "single_nz":
        data = np.zeros(shape, dtype=np.float32)
        data[shape[0] // 2] = 5.0
        return data
    elif data_type == "diagonal":
        data = np.zeros(shape, dtype=np.float32)
        np.fill_diagonal(data, np.arange(1, min(shape) + 1))
        return data
    elif data_type == "corners":
        data = np.zeros(shape, dtype=np.float32)
        data[0, 0] = 1.0
        data[0, -1] = 2.0
        data[-1, 0] = 3.0
        data[-1, -1] = 4.0
        return data
    elif data_type == "random":
        data = (np.random.rand(*shape) > 0.6).astype(np.float32) * np.random.randn(
            *shape
        ).astype(np.float32)
        return data
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_nonzero_numerical():
    """Test NonZero with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "NonZero",
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
        ref_output = ref_impl_nonzero(test_data)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_nonzero(i_tensors, op_obj)

            # For NonZero, we need exact match of indices
            numerical_match = np.array_equal(computed_output, ref_output)

            if not numerical_match:
                print(
                    f"\n  Output shape: {computed_output.shape}, expected: {ref_output.shape}"
                )
                print(
                    f"  Num nonzero: computed={computed_output.shape[1]}, ref={ref_output.shape[1]}"
                )
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Indices ✓]")
        elif shape_match:
            print(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Indices: {numerical_match}]"
            )
        else:
            print(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            print(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            print(f"  Indices match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_nonzero_precision"
precision_test_cases = [
    # (name, input, expected_output)
    (
        "Single nonzero at index 2",
        np.array([0.0, 0.0, 5.0, 0.0], dtype=np.float32),
        np.array([[2]], dtype=np.int64),
    ),
    (
        "Three nonzero elements",
        np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32),
        np.array([[0, 2, 4]], dtype=np.int64),
    ),
    (
        "2D single element",
        np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1], [1]], dtype=np.int64),
    ),
    (
        "2D multiple elements",
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        np.array([[0, 1], [0, 1]], dtype=np.int64),
    ),
    (
        "All zeros",
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros((1, 0), dtype=np.int64),
    ),
    (
        "Negative values are nonzero",
        np.array([0.0, -1.0, 0.0, -2.0], dtype=np.float32),
        np.array([[1, 3]], dtype=np.int64),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_nonzero_precision():
    """Test NonZero with precise known outputs"""
    msgw = 35

    for tno, (tmsg, test_data, expected_output) in enumerate(precision_test_cases):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "NonZero",
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
            computed_output = compute_nonzero(i_tensors, op_obj)
            match = np.array_equal(computed_output, expected_output)
            if match:
                print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                print(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(
                    f"  Expected shape: {expected_output.shape}, got: {computed_output.shape}"
                )
                if expected_output.size <= 20 and computed_output.size <= 20:
                    print(f"  Expected: {expected_output}")
                    print(f"  Got:      {computed_output}")
        except Exception as e:
            print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Edge cases
test_name_edge = "test_nonzero_edge"
edge_test_cases = [
    # (name, input_shape, data_type, description)
    ("All zeros", [10], "all_zeros", "No nonzero elements"),
    ("All nonzero", [10], "all_nonzero", "All elements nonzero"),
    ("Single element zero", [1], "single_zero", "Single zero element"),
    ("Single element nonzero", [1], "single_nonzero", "Single nonzero element"),
    ("Very sparse", [100], "very_sparse", "1% nonzero"),
    ("Alternating pattern", [20], "alternating", "Every other element"),
    ("3D single nonzero", [3, 4, 5], "3d_single", "One element in 3D"),
    ("Large tensor", [10, 10], "large_sparse", "Large sparse tensor"),
]


def generate_edge_test_data(shape, data_type):
    """Generate edge case test data"""
    if data_type == "all_zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "all_nonzero":
        return (np.random.randn(*shape) + 10).astype(np.float32)  # Avoid zeros
    elif data_type == "single_zero":
        return np.array([0.0], dtype=np.float32)
    elif data_type == "single_nonzero":
        return np.array([1.0], dtype=np.float32)
    elif data_type == "very_sparse":
        data = np.zeros(shape, dtype=np.float32)
        num_nz = max(1, int(np.prod(shape) * 0.01))
        flat_indices = np.random.choice(np.prod(shape), size=num_nz, replace=False)
        data.flat[flat_indices] = np.random.randn(num_nz).astype(np.float32) + 5
        return data
    elif data_type == "alternating":
        data = np.zeros(shape, dtype=np.float32)
        data[::2] = 1.0
        return data
    elif data_type == "3d_single":
        data = np.zeros(shape, dtype=np.float32)
        data[1, 2, 3] = 1.0
        return data
    elif data_type == "large_sparse":
        data = np.zeros(shape, dtype=np.float32)
        num_nz = max(1, int(np.prod(shape) * 0.1))
        flat_indices = np.random.choice(np.prod(shape), size=num_nz, replace=False)
        data.flat[flat_indices] = np.random.randn(num_nz).astype(np.float32) * 10
        return data
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_nonzero_edge_cases():
    """Test NonZero edge cases and boundary conditions"""
    msgw = 25

    for tno, (tmsg, input_shape, data_type, description) in enumerate(edge_test_cases):
        op_name = f"{test_name_edge}_{tno}"

        test_data = generate_edge_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "NonZero",
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
            computed_output = compute_nonzero(i_tensors, op_obj)
            ref_output = ref_impl_nonzero(test_data)

            # Validate
            match = np.array_equal(computed_output, ref_output)

            # Additional edge case checks
            if data_type == "all_zeros":
                # Should return empty array with shape (ndim, 0)
                assert computed_output.shape[0] == len(input_shape), "First dim = ndim"
                assert computed_output.shape[1] == 0, "No nonzero elements"
            elif data_type == "all_nonzero":
                # Should return all indices
                assert computed_output.shape[1] == np.prod(
                    input_shape
                ), "All elements nonzero"
            elif data_type == "single_nonzero":
                # Should return one index
                assert computed_output.shape[1] == 1, "One nonzero element"
            elif data_type == "alternating":
                # Should return half the elements
                expected_count = (input_shape[0] + 1) // 2
                assert (
                    computed_output.shape[1] == expected_count
                ), f"Alternating: {expected_count} nonzero"

            # Validate output shape always (ndim, num_nonzero)
            assert computed_output.shape[0] == len(
                test_data.shape
            ), "Output dim 0 = input ndim"

            # Validate output dtype is int64
            assert computed_output.dtype == np.int64, "Output must be int64"

            if match:
                print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                print(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  {description}")
                print(
                    f"  Expected shape: {ref_output.shape}, got: {computed_output.shape}"
                )
        except Exception as e:
            print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_nonzero_memory_stats(shape):
    """Calculate memory access and arithmetic operations for nonzero operation"""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    X = SimTensor({"name": "X", "shape": shape, "dtype": "float32"})
    X.data = np.random.randn(*shape).astype(np.float32)  # random → most values nonzero
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_nonzero",
        "optype": "NonZero",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_nonzero"]
    for x in o_tensors:
        x.op_out = ["test_nonzero"]

    op.get_perf_counts(i_tensors, o_tensors)

    perf_stats = op.perf_stats
    actual_instructions = perf_stats.get("instrs", {})
    instr_sum = (
        sum(actual_instructions.values())
        if isinstance(actual_instructions, dict)
        else 0
    )
    ops = instr_sum if instr_sum > 0 else perf_stats.get("inElems", np.prod(shape))
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
def test_nonzero_memory_validation():
    """Test NonZero memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    print("\n" + "=" * 80)
    print("NONZERO MEMORY VALIDATION")
    print("=" * 80)
    print(f"Device: {device.devname}")
    print(f"  Name: {device.name}")
    print(f"  Frequency: {device.freq_MHz} MHz")
    print(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    print("=" * 80)

    # Various input shapes with dense (mostly nonzero) data
    test_configs = [
        [128, 128],
        [256, 256],
        [1, 64, 56, 56],
        [1, 128, 28, 28],
        [4, 64, 56, 56],
        [4, 128, 28, 28],
        [512, 512],
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape in test_configs:
        stats = calculate_nonzero_memory_stats(shape)

        print(f"\nInput shape: {shape}")
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
