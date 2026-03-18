#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import logging

# Silence the avgpool2d-related log messages
try:
    from loguru import logger

    logger.disable("ttsim")
except ImportError:
    pass

import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_reducemean
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_reducemean(X, axes, keepdims):
    """Reference implementation of ReduceMean"""
    if axes is None:
        return np.mean(X, keepdims=keepdims)
    return np.mean(X, axis=tuple(axes), keepdims=keepdims)


# Test cases for numerical validation
test_name = "test_mean"
test_cases = [
    # (name, input_shape, axes, keepdims, data_type)
    ("1D reduce all", [10], None, False, "mixed"),
    ("1D keep dims", [10], [0], True, "mixed"),
    ("2D reduce axis 0", [4, 5], [0], False, "mixed"),
    ("2D reduce axis 1", [4, 5], [1], False, "mixed"),
    ("2D reduce all", [4, 5], None, False, "mixed"),
    ("2D keep dims axis 0", [4, 5], [0], True, "mixed"),
    ("3D reduce axis 0", [2, 3, 4], [0], False, "mixed"),
    ("3D reduce axis 1", [2, 3, 4], [1], False, "mixed"),
    ("3D reduce axis 2", [2, 3, 4], [2], False, "mixed"),
    ("3D reduce axes [0,2]", [2, 3, 4], [0, 2], False, "mixed"),
    ("4D reduce axes [1,3]", [2, 3, 4, 5], [1, 3], True, "mixed"),
    ("Positive values", [5, 5], [1], False, "positive"),
    ("Negative values", [5, 5], [0], False, "negative"),
]


def generate_test_data(shape, data_type):
    """Generate test data based on type"""
    if data_type == "mixed":
        return (np.random.randn(*shape) * 10).astype(np.float32)
    elif data_type == "positive":
        return (np.random.rand(*shape) * 10).astype(np.float32)
    elif data_type == "negative":
        return -(np.random.rand(*shape) * 10).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_mean_numerical():
    """Test ReduceMean with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, axes, keepdims, data_type) in enumerate(test_cases):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, data_type)

        # Create input tensors
        i_tensors = [F._from_data("X", test_data)]

        # Add axes tensor if specified
        if axes is not None:
            axes_tensor = F._from_data(
                "axes", np.array(axes, dtype=np.int64), is_param=False, is_const=True
            )
            i_tensors.append(axes_tensor)

        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"keepdims": 1 if keepdims else 0},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_reducemean(test_data, axes, keepdims)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_reducemean(i_tensors, op_obj)
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
test_name_precision = "test_mean_precision"
precision_test_cases = [
    # (name, input, axes, keepdims, expected_output)
    (
        "Mean of [1,2,3]",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        [0],
        False,
        np.array(2.0, dtype=np.float32),
    ),
    (
        "Mean 2D rows",
        np.array([[1.0, 5.0], [2.0, 4.0]], dtype=np.float32),
        [1],
        False,
        np.array([3.0, 3.0], dtype=np.float32),
    ),
    (
        "Mean 2D cols",
        np.array([[1.0, 5.0], [3.0, 7.0]], dtype=np.float32),
        [0],
        False,
        np.array([2.0, 6.0], dtype=np.float32),
    ),
    (
        "Mean all with keepdims",
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
        None,
        True,
        np.array([[5.0]], dtype=np.float32),
    ),
    (
        "Negative values",
        np.array([[-4.0, -2.0], [-6.0, -8.0]], dtype=np.float32),
        [1],
        False,
        np.array([-3.0, -7.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_mean_precision():
    """Test ReduceMean with precise known outputs"""
    msgw = 30

    for tno, (tmsg, test_data, axes, keepdims, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]

        if axes is not None:
            axes_tensor = F._from_data(
                "axes", np.array(axes, dtype=np.int64), is_param=False, is_const=True
            )
            i_tensors.append(axes_tensor)

        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"keepdims": 1 if keepdims else 0},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_reducemean(i_tensors, op_obj)
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
test_name_edge = "test_mean_edge"
edge_test_cases = [
    # (name, input_shape, axes, keepdims, data_type, description)
    ("All same values", [5, 5], [1], False, "same", "All elements identical"),
    ("Single element", [1], [0], False, "single", "Single element tensor"),
    ("All zeros", [4, 4], [0], False, "zeros", "All zero values"),
    ("Symmetric around zero", [10], [0], False, "symmetric", "Sum to zero"),
    ("Large range", [8, 8], [1], True, "large_range", "Wide value range"),
    ("Reduce all dims", [3, 4, 5], None, False, "reduce_all", "Reduce to scalar"),
]


def generate_edge_test_data(shape, data_type):
    """Generate edge case test data"""
    if data_type == "same":
        return np.full(shape, 5.0, dtype=np.float32)
    elif data_type == "single":
        return np.array([7.0], dtype=np.float32)
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "symmetric":
        # Create symmetric data that sums to approximately zero
        data = np.linspace(-5, 5, np.prod(shape), dtype=np.float32).reshape(shape)
        return data
    elif data_type == "large_range":
        return (np.random.rand(*shape) * 1000 - 500).astype(np.float32)
    elif data_type == "reduce_all":
        return (np.random.randn(*shape) * 10).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_mean_edge_cases():
    """Test ReduceMean edge cases and boundary conditions"""
    msgw = 25

    for tno, (tmsg, input_shape, axes, keepdims, data_type, description) in enumerate(
        edge_test_cases
    ):
        op_name = f"{test_name_edge}_{tno}"

        test_data = generate_edge_test_data(input_shape, data_type)

        i_tensors = [F._from_data("X", test_data)]

        if axes is not None:
            axes_tensor = F._from_data(
                "axes", np.array(axes, dtype=np.int64), is_param=False, is_const=True
            )
            i_tensors.append(axes_tensor)

        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "ReduceMean",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"keepdims": 1 if keepdims else 0},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_reducemean(i_tensors, op_obj)
            ref_output = ref_impl_reducemean(test_data, axes, keepdims)

            # Validate
            match = np.allclose(computed_output, ref_output, rtol=1e-5, atol=1e-6)

            # Additional edge case checks
            if data_type == "same":
                # Mean of same values = that value
                assert np.allclose(
                    computed_output, 5.0, atol=1e-6
                ), "Mean of same values"
            elif data_type == "single":
                # Mean of single element = that element
                assert np.allclose(
                    computed_output, test_data[0], atol=1e-6
                ), "Mean of single element"
            elif data_type == "zeros":
                # Mean of zeros = 0
                assert np.allclose(computed_output, 0.0, atol=1e-6), "Mean of zeros"

            # Mean should be between min and max
            if axes is None:
                test_min = np.min(test_data)
                test_max = np.max(test_data)
                assert np.all(
                    (computed_output >= test_min - 1e-5)
                    & (computed_output <= test_max + 1e-5)
                ), "Mean in range [min, max]"

            if match:
                print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                print(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  {description}")
                print(f"  Max diff: {np.max(np.abs(computed_output - ref_output))}")
        except Exception as e:
            print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_mean_memory_stats(shape, axes, keepdims=1):
    """Calculate memory access and arithmetic operations for reducemean operation"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    X = SimTensor({"name": "X", "shape": shape, "dtype": "float32"})
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_mean",
        "optype": "ReduceMean",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {"axes": axes, "keepdims": keepdims},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_mean"]
    for x in o_tensors:
        x.op_out = ["test_mean"]

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
def test_mean_memory_validation():
    """Test ReduceMean memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    print("\n" + "=" * 80)
    print("MEAN (REDUCEMEAN) MEMORY VALIDATION")
    print("=" * 80)
    print(f"Device: {device.devname}")
    print(f"  Name: {device.name}")
    print(f"  Frequency: {device.freq_MHz} MHz")
    print(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    print("=" * 80)

    # (input_shape, axes, keepdims)
    test_configs = [
        ([64, 64], [1], 1),
        ([128, 128], [0], 1),
        ([32, 32, 32], [2], 1),
        ([32, 32, 32], [1, 2], 1),
        ([16, 16, 16, 16], [2, 3], 1),
        ([4, 64, 56, 56], [2, 3], 1),
        ([8, 256, 14, 14], [2, 3], 1),
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape, axes, keepdims in test_configs:
        stats = calculate_mean_memory_stats(shape, axes, keepdims)

        print(f"\nShape: {shape}, Axes: {axes}")
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
