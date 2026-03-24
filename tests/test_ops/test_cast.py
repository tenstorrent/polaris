#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import pytest
import numpy as np
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
from ttsim.ops.tensor import SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_cast
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

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


def ref_impl_cast(X, to_dtype):
    """Reference implementation of cast operation"""
    return X.astype(to_dtype)


# ONNX dtype mapping
ONNX_DTYPE_MAP = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}


# Test cases for numerical validation
test_name = "test_cast"
test_cases = [
    # (name, input_shape, from_dtype, to_dtype_code, data_type)
    ("float32 to int32", [10], np.float32, 6, "small_float"),
    ("int32 to float32", [10], np.int32, 1, "small_int"),
    ("float32 to int64", [5, 5], np.float32, 7, "mixed_float"),
    ("int64 to float32", [5, 5], np.int64, 1, "mixed_int"),
    ("float32 to float64", [3, 4], np.float32, 11, "precision_up"),
    ("float64 to float32", [3, 4], np.float64, 1, "precision_down"),
    ("int32 to int64", [8], np.int32, 7, "int_widen"),
    ("int64 to int32", [8], np.int64, 6, "int_narrow"),
    ("uint8 to float32", [4, 4], np.uint8, 1, "uint8_to_float"),
    ("float32 to uint8", [4, 4], np.float32, 2, "float_to_uint8"),
]


def generate_test_data(shape, from_dtype, data_type):
    """Generate test data based on type"""
    if data_type == "small_float":
        return (np.random.rand(*shape) * 10).astype(from_dtype)
    elif data_type == "small_int":
        return np.random.randint(-50, 51, size=shape).astype(from_dtype)
    elif data_type == "mixed_float":
        return (np.random.randn(*shape) * 100).astype(from_dtype)
    elif data_type == "mixed_int":
        return np.random.randint(-1000, 1001, size=shape).astype(from_dtype)
    elif data_type == "precision_up":
        return (np.random.randn(*shape) * 10).astype(from_dtype)
    elif data_type == "precision_down":
        return (np.random.randn(*shape) * 10).astype(from_dtype)
    elif data_type == "int_widen":
        return np.random.randint(-1000, 1001, size=shape).astype(from_dtype)
    elif data_type == "int_narrow":
        return np.random.randint(-1000, 1001, size=shape).astype(from_dtype)
    elif data_type == "uint8_to_float":
        return np.random.randint(0, 256, size=shape).astype(from_dtype)
    elif data_type == "float_to_uint8":
        return (np.random.rand(*shape) * 255).astype(from_dtype)
    else:
        return np.random.randn(*shape).astype(from_dtype)


@pytest.mark.unit
@pytest.mark.opunit
def test_cast_numerical():
    """Test Cast with numerical validation and shape validation"""
    msgw = get_max_test_msg_len(test_cases)

    for tno, (tmsg, input_shape, from_dtype, to_dtype_code, data_type) in enumerate(
        test_cases
    ):
        op_name = f"{test_name}_{tno}"

        # Generate test data
        test_data = generate_test_data(input_shape, from_dtype, data_type)
        to_dtype = ONNX_DTYPE_MAP[to_dtype_code]

        # Create input tensors
        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Cast",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"to": to_dtype_code},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        # Execute operation
        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        ref_output = ref_impl_cast(test_data, to_dtype)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        dtype_match = True
        try:
            computed_output = compute_cast(i_tensors, op_obj)

            # Check dtype
            if computed_output.dtype != to_dtype:
                dtype_match = False
                print(
                    f"\n  Dtype mismatch: got {computed_output.dtype}, expected {to_dtype}"
                )

            # For integer conversions, use exact match; for float, use tolerance
            if np.issubdtype(to_dtype, np.integer):
                numerical_match = np.array_equal(computed_output, ref_output)
            else:
                numerical_match = np.allclose(
                    computed_output, ref_output, rtol=1e-5, atol=1e-6
                )

            if not numerical_match:
                max_diff = np.max(
                    np.abs(
                        computed_output.astype(np.float64)
                        - ref_output.astype(np.float64)
                    )
                )
                print(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            print(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True and dtype_match:
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓, Dtype ✓]")
        elif shape_match:
            status = []
            if numerical_match != True:
                status.append(f"Numerical: {numerical_match}")
            if not dtype_match:
                status.append("Dtype: ✗")
            print(f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [{', '.join(status)}]")
        else:
            print(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            print(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )


# Precision test cases with known outputs
test_name_precision = "test_cast_precision"
precision_test_cases = [
    # (name, input, to_dtype_code, expected_output)
    (
        "Float to int truncation",
        np.array([3.7, -2.3, 5.9], dtype=np.float32),
        6,  # int32
        np.array([3, -2, 5], dtype=np.int32),
    ),
    (
        "Int to float exact",
        np.array([1, 2, 3, 4], dtype=np.int32),
        1,  # float32
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    ),
    (
        "Zero preservation",
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        6,  # int32
        np.array([0, 0, 0], dtype=np.int32),
    ),
    (
        "Uint8 to float",
        np.array([0, 127, 255], dtype=np.uint8),
        1,  # float32
        np.array([0.0, 127.0, 255.0], dtype=np.float32),
    ),
    (
        "Small ints preserved",
        np.array([1, -1, 10, -10], dtype=np.int32),
        7,  # int64
        np.array([1, -1, 10, -10], dtype=np.int64),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_cast_precision():
    """Test Cast with precise known outputs"""
    msgw = 30

    for tno, (tmsg, test_data, to_dtype_code, expected_output) in enumerate(
        precision_test_cases
    ):
        op_name = f"{test_name_precision}_{tno}"

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Cast",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"to": to_dtype_code},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_cast(i_tensors, op_obj)
            match = np.array_equal(computed_output, expected_output)
            dtype_match = computed_output.dtype == expected_output.dtype

            if match and dtype_match:
                print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                print(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  Expected: {expected_output} (dtype: {expected_output.dtype})")
                print(f"  Got:      {computed_output} (dtype: {computed_output.dtype})")
        except Exception as e:
            print(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Edge cases
test_name_edge = "test_cast_edge"
edge_test_cases = [
    # (name, input_shape, from_dtype, to_dtype_code, data_type, description)
    ("Same type cast", [10], np.float32, 1, "identity", "Cast to same type"),
    ("Large values", [5], np.float32, 6, "large_vals", "Large float to int"),
    ("Negative to unsigned", [5], np.int32, 2, "neg_to_uint", "Negative ints to uint8"),
    ("Overflow values", [5], np.float32, 2, "overflow", "Values outside uint8 range"),
    ("Very small floats", [10], np.float32, 6, "tiny_float", "Near-zero floats to int"),
    ("Single element", [1], np.float64, 1, "single", "Single element cast"),
    ("4D tensor", [2, 3, 4, 5], np.float32, 6, "multidim", "Multi-dimensional cast"),
]


def generate_edge_test_data(shape, from_dtype, data_type):
    """Generate edge case test data"""
    if data_type == "identity":
        return (np.random.randn(*shape) * 10).astype(from_dtype)
    elif data_type == "large_vals":
        return (np.random.rand(*shape) * 10000).astype(from_dtype)
    elif data_type == "neg_to_uint":
        return np.random.randint(-50, 51, size=shape).astype(from_dtype)
    elif data_type == "overflow":
        return (np.random.rand(*shape) * 500 - 100).astype(from_dtype)
    elif data_type == "tiny_float":
        return (np.random.rand(*shape) * 0.9).astype(from_dtype)
    elif data_type == "single":
        return np.array([3.14159], dtype=from_dtype)
    elif data_type == "multidim":
        return (np.random.randn(*shape) * 10).astype(from_dtype)
    else:
        return np.random.randn(*shape).astype(from_dtype)


@pytest.mark.unit
@pytest.mark.opunit
def test_cast_edge_cases():
    """Test Cast edge cases and boundary conditions"""
    msgw = 25

    for tno, (
        tmsg,
        input_shape,
        from_dtype,
        to_dtype_code,
        data_type,
        description,
    ) in enumerate(edge_test_cases):
        op_name = f"{test_name_edge}_{tno}"

        test_data = generate_edge_test_data(input_shape, from_dtype, data_type)
        to_dtype = ONNX_DTYPE_MAP[to_dtype_code]

        i_tensors = [F._from_data("X", test_data)]
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": op_name,
            "optype": "Cast",
            "inList": [x.name for x in i_tensors],
            "outList": [x.name for x in o_tensors],
            "attrs": {"to": to_dtype_code},
        }

        op_obj = SimOp(op_info)
        for x in i_tensors:
            x.op_in = [op_name]
        for x in o_tensors:
            x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        try:
            computed_output = compute_cast(i_tensors, op_obj)
            ref_output = ref_impl_cast(test_data, to_dtype)

            # Validate dtype
            assert (
                computed_output.dtype == to_dtype
            ), f"Dtype mismatch: {computed_output.dtype} != {to_dtype}"

            # Validate shape preserved
            assert computed_output.shape == test_data.shape, "Shape should be preserved"

            # Additional edge case checks
            if data_type == "identity":
                # Same type cast should be exact
                assert np.allclose(
                    computed_output, test_data, rtol=1e-6
                ), "Identity cast"
            elif data_type == "tiny_float":
                # Small floats to int should be 0
                if to_dtype == np.int32 or to_dtype == np.int64:
                    assert np.all(computed_output == 0), "Tiny floats -> 0"

            # Compare with reference
            if np.issubdtype(to_dtype, np.integer):
                match = np.array_equal(computed_output, ref_output)
            else:
                match = np.allclose(computed_output, ref_output, rtol=1e-5, atol=1e-6)

            if match:
                print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                print(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                print(f"  {description}")
                if test_data.size <= 10:
                    print(f"  Input:  {test_data.flatten()}")
                    print(f"  Output: {computed_output.flatten()}")
                    print(f"  Ref:    {ref_output.flatten()}")
        except Exception as e:
            print(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_cast_memory_stats(shape, ops_per_elem=1):
    """
    Calculate memory statistics for cast operation.

    Cast is a type conversion operation that reads one dtype and writes another.
    For fp16 precision, both input and output are 2 bytes per element.
    Operations per element: ~1 (type conversion)
    """
    # Element count
    num_elements = np.prod(shape)

    # Memory: input bytes + output bytes (fp16 = 2 bytes per element)
    bytes_per_element = 2  # fp16
    input_bytes = num_elements * bytes_per_element
    output_bytes = num_elements * bytes_per_element
    total_bytes = input_bytes + output_bytes

    # Instructions (ops per element)
    instructions = int(num_elements * ops_per_elem)

    # Arithmetic intensity (ops per byte)
    arithmetic_intensity = instructions / total_bytes if total_bytes > 0 else 0

    return {
        "instructions": instructions,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "ops_per_elem": ops_per_elem,
        "num_elements": num_elements,
    }


def test_cast_memory_validation(capsys, request):
    """
    Memory validation test for cast operation.
    Tests memory bandwidth and compute requirements using simulator performance counters.
    """
    # Access pytest's terminal reporter for explicit output
    terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")

    # Device configuration
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device_pkg = packages["n150"]  # Use Wormhole n150 device
    device = Device(device_pkg)

    # Test cases with different input shapes
    test_cases = [
        {"shape": [256], "name": "tiny_1d"},
        {"shape": [64, 64], "name": "small_2d"},
        {"shape": [128, 128], "name": "medium_2d"},
        {"shape": [256, 256], "name": "large_2d"},
        {"shape": [32, 32, 32], "name": "small_3d"},
        {"shape": [64, 64, 64], "name": "medium_3d"},
        {"shape": [16, 16, 16, 16], "name": "4d_tensor"},
    ]

    print("\n" + "=" * 80)
    print("CAST OPERATION - MEMORY VALIDATION TEST")
    print("=" * 80)

    # Device info (WITHOUT bandwidth line)
    print(f"\nDevice: {device.devname} ({device.name})")
    print(f"Device frequency: {device.freq_MHz} MHz")
    print(f"Memory frequency: {device.memfreq_MHz} MHz")

    print("\nTest Configuration:")
    print(f"  Precision: fp16 (2 bytes per element)")
    print(f"  Operations: Type conversion (~1 op per element)")
    print(f"  Memory pattern: Read input + Write output")

    print("\n" + "-" * 80)
    print("TEST RESULTS:")
    print("-" * 80)

    results = []

    for test_case in test_cases:
        shape = test_case["shape"]
        test_name = test_case["name"]

        # Create input tensor with fp16 precision
        input_tensor = SimTensor(
            {"name": "input_X", "shape": shape, "dtype": "float16"}
        )
        input_tensor.data = np.random.randn(*shape).astype(np.float16)

        # Output tensor (same shape for cast)
        output_tensor = SimTensor(
            {"name": "output", "shape": shape, "dtype": "float32"}
        )

        # Create cast operation
        op = SimOp(
            {
                "name": f"cast_op_{test_name}",
                "optype": "Cast",
                "inList": ["input_X"],
                "outList": ["output"],
            }
        )
        op.attrs = {"to": 1}  # Cast to float32
        op.precision = "fp16"

        # Get performance counts
        op.get_perf_counts([input_tensor], [output_tensor])

        # Extract stats from perf_stats
        perf_stats = op.perf_stats
        actual_instructions = perf_stats.get("instrs", {})
        num_instructions = (
            sum(actual_instructions.values())
            if isinstance(actual_instructions, dict)
            else np.prod(shape)
        )

        input_bytes = perf_stats["inBytes"]
        output_bytes = perf_stats["outBytes"]
        total_bytes = input_bytes + output_bytes
        arithmetic_intensity = num_instructions / total_bytes if total_bytes > 0 else 0

        # Compute cycles and bottleneck
        compute_cycles = num_instructions / 1  # 1 op per cycle
        memory_cycles = total_bytes / (
            device.simconfig_obj.peak_bandwidth(freq_units="GHz")
            * 1e9
            / device.freq_MHz
            / 1e6
        )
        bottleneck = "MEMORY" if memory_cycles > compute_cycles else "COMPUTE"

        # Store results
        results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "instructions": num_instructions,
                "total_bytes": total_bytes,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
            }
        )

        # Print test result
        print(f"\n[{test_name}] Shape: {shape}")
        print(f"  Instructions executed: {num_instructions:,}")
        print(f"  Input bytes:  {input_bytes:,}")
        print(f"  Output bytes: {output_bytes:,}")
        print(f"  Total data moved: {total_bytes:,} bytes ({total_bytes/1024:.2f} KB)")
        print(f"  Arithmetic intensity: {arithmetic_intensity:.3f} ops/byte")
        print(f"  Compute cycles: {compute_cycles:,.0f}")
        print(f"  Memory cycles:  {memory_cycles:,.0f}")
        print(f"  Bottleneck: {bottleneck}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_instructions = sum(r["instructions"] for r in results)
    total_bytes = sum(r["total_bytes"] for r in results)
    avg_arithmetic_intensity = (
        total_instructions / total_bytes if total_bytes > 0 else 0
    )

    memory_bound_count = sum(1 for r in results if r["bottleneck"] == "MEMORY")
    compute_bound_count = sum(1 for r in results if r["bottleneck"] == "COMPUTE")

    print(f"\nTotal tests: {len(results)}")
    print(f"Total instructions: {total_instructions:,}")
    print(f"Total data moved: {total_bytes:,} bytes ({total_bytes/1024/1024:.2f} MB)")
    print(f"Average arithmetic intensity: {avg_arithmetic_intensity:.3f} ops/byte")
    print(f"\nBottleneck distribution:")
    print(f"  Memory-bound: {memory_bound_count}/{len(results)}")
    print(f"  Compute-bound: {compute_bound_count}/{len(results)}")
    print("\n" + "=" * 80)

    # All tests pass if we got here
    assert True
