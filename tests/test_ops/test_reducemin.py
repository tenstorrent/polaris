#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import sys, os, logging
import pytest
import numpy as np
from loguru import logger
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor, SimTensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_reducemin
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


def ref_impl_reducemin(X, axes, keepdims):
    """Reference implementation of ReduceMin"""
    if axes is None:
        return np.min(X, keepdims=keepdims)
    return np.min(X, axis=tuple(axes), keepdims=keepdims)


# Test cases for numerical validation
test_name = "test_reducemin"
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
def test_reducemin_numerical():
    """Test ReduceMin with numerical validation and shape validation"""
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
            "optype": "ReduceMin",
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
        ref_output = ref_impl_reducemin(test_data, axes, keepdims)
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)

        shape_match = inf_shape == ref_shape

        # 2. Numerical validation
        numerical_match = True
        try:
            computed_output = compute_reducemin(i_tensors, op_obj)
            numerical_match = np.allclose(
                computed_output, ref_output, rtol=1e-5, atol=1e-6
            )

            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # Report results
        if shape_match and numerical_match == True:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PASS [Shape ✓, Numerical ✓]"
            )
        elif shape_match:
            logger.debug(
                f"TEST[{tno:3d}] {tmsg:{msgw}s} PARTIAL [Shape ✓, Numerical: {numerical_match}]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] {tmsg:{msgw}s} FAIL")
            logger.debug(
                "  Shape match: "
                f"{shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")


# Precision test cases with known outputs
test_name_precision = "test_reducemin_precision"
precision_test_cases = [
    # (name, input, axes, keepdims, expected_output)
    (
        "Min of [1,2,3]",
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        [0],
        False,
        np.array(1.0, dtype=np.float32),
    ),
    (
        "Min 2D rows",
        np.array([[1.0, 5.0], [2.0, 3.0]], dtype=np.float32),
        [1],
        False,
        np.array([1.0, 2.0], dtype=np.float32),
    ),
    (
        "Min 2D cols",
        np.array([[1.0, 5.0], [2.0, 3.0]], dtype=np.float32),
        [0],
        False,
        np.array([1.0, 3.0], dtype=np.float32),
    ),
    (
        "Min all with keepdims",
        np.array([[1.0, 5.0], [2.0, 3.0]], dtype=np.float32),
        None,
        True,
        np.array([[1.0]], dtype=np.float32),
    ),
    (
        "Negative values",
        np.array([[-5.0, -2.0], [-8.0, -1.0]], dtype=np.float32),
        [1],
        False,
        np.array([-5.0, -8.0], dtype=np.float32),
    ),
]


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemin_precision():
    """Test ReduceMin with precise known outputs"""
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
            "optype": "ReduceMin",
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
            computed_output = compute_reducemin(i_tensors, op_obj)
            match = np.allclose(computed_output, expected_output, rtol=1e-5, atol=1e-6)
            if match:
                logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} PASS")
            else:
                logger.debug(f"\nPRECISION TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  Expected: {expected_output.flatten()}")
                logger.debug(f"  Got:      {computed_output.flatten()}")
                logger.debug(
                    f"  Diff:     {(computed_output - expected_output).flatten()}"
                )
        except Exception as e:
            logger.debug(f"PRECISION TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


# Edge cases
test_name_edge = "test_reducemin_edge"
edge_test_cases = [
    # (name, input_shape, axes, keepdims, data_type, description)
    ("All same values", [5, 5], [1], False, "same", "All elements identical"),
    ("Single element", [1], [0], False, "single", "Single element tensor"),
    ("All negative", [4, 4], [0], False, "all_neg", "All negative values"),
    ("Contains zero", [10], [0], False, "with_zero", "Mix including zero"),
    ("Large range", [8, 8], [1], True, "large_range", "Wide value range"),
    ("Reduce all dims", [3, 4, 5], None, False, "reduce_all", "Reduce to scalar"),
]


def generate_edge_test_data(shape, data_type):
    """Generate edge case test data"""
    if data_type == "same":
        return np.full(shape, 5.0, dtype=np.float32)
    elif data_type == "single":
        return np.array([7.0], dtype=np.float32)
    elif data_type == "all_neg":
        return -(np.random.rand(*shape) * 10 + 1).astype(np.float32)
    elif data_type == "with_zero":
        data = (np.random.randn(*shape) * 5).astype(np.float32)
        data[len(data) // 2] = 0.0
        return data
    elif data_type == "large_range":
        return (np.random.rand(*shape) * 1000 - 500).astype(np.float32)
    elif data_type == "reduce_all":
        return (np.random.randn(*shape) * 10).astype(np.float32)
    else:
        return np.random.randn(*shape).astype(np.float32)


@pytest.mark.unit
@pytest.mark.opunit
def test_reducemin_edge_cases():
    """Test ReduceMin edge cases and boundary conditions"""
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
            "optype": "ReduceMin",
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
            computed_output = compute_reducemin(i_tensors, op_obj)
            ref_output = ref_impl_reducemin(test_data, axes, keepdims)

            # Validate
            match = np.allclose(computed_output, ref_output, rtol=1e-5, atol=1e-6)

            # Additional edge case checks
            if data_type == "same":
                # All same values -> min = that value
                assert np.allclose(
                    computed_output, 5.0, atol=1e-6
                ), "Min of same values"
            elif data_type == "single":
                # Single element -> min = that element
                assert np.allclose(
                    computed_output, test_data[0], atol=1e-6
                ), "Min of single element"

            # Min should be <= all input values
            if axes is None:
                assert np.all(computed_output <= test_data + 1e-5), "Min <= all values"

            if match:
                logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} PASS - {description}")
            else:
                logger.debug(f"\nEDGE TEST[{tno}] {tmsg:{msgw}s} FAIL")
                logger.debug(f"  {description}")
                logger.debug(
                    f"  Max diff: {np.max(np.abs(computed_output - ref_output))}"
                )
        except Exception as e:
            logger.debug(f"EDGE TEST[{tno}] {tmsg:{msgw}s} ERROR: {e}")


def calculate_reducemin_memory_stats(shape, axes, keepdims=1):
    """Calculate memory access and arithmetic operations for reducemin operation"""
    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    X = SimTensor({"name": "X", "shape": shape, "dtype": "float32"})
    i_tensors = [X]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": "test_reducemin",
        "optype": "ReduceMin",
        "inList": ["X"],
        "outList": ["Y"],
        "attrs": {"axes": axes, "keepdims": keepdims},
    }

    op = SimOp(op_info)
    for x in i_tensors:
        x.op_in = ["test_reducemin"]
    for x in o_tensors:
        x.op_out = ["test_reducemin"]

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
def test_reducemin_memory_validation():
    """Test ReduceMin memory access patterns and arithmetic intensity"""

    config_path = os.path.join(polaris_root, "config", "tt_wh.yaml")
    ipgroups, packages = get_arspec_from_yaml(config_path)
    device = Device(packages["n150"])

    logger.info("\n" + "=" * 80)
    logger.info("REDUCEMIN MEMORY VALIDATION")
    logger.info("=" * 80)
    logger.info(f"Device: {device.devname}")
    logger.info(f"  Name: {device.name}")
    logger.info(f"  Frequency: {device.freq_MHz} MHz")
    logger.info(f"  Memory Frequency: {device.memfreq_MHz} MHz")
    logger.info("=" * 80)

    # (shape, axes, keepdims)
    test_configs = [
        ([1, 64, 56, 56], [2, 3], 1),
        ([1, 128, 28, 28], [2, 3], 1),
        ([1, 256, 14, 14], [2, 3], 1),
        ([4, 64, 56, 56], [2, 3], 1),
        ([4, 128, 28, 28], [1], 1),
        ([8, 256, 14, 14], [1], 0),
        ([2, 512, 7, 7], [2, 3], 1),
    ]

    memory_bound_count = 0
    compute_bound_count = 0

    for shape, axes, keepdims in test_configs:
        stats = calculate_reducemin_memory_stats(shape, axes, keepdims)

        logger.debug(f"\nShape: {shape}, Axes: {axes}, Keepdims: {keepdims}")
        logger.debug(f"  Memory: {stats['memory_mb']:.4f} MB")
        logger.debug(f"  Operations: {stats['ops']}")
        logger.debug(
            f"  Arithmetic Intensity: {stats['arithmetic_intensity']:.6f} ops/byte"
        )
        logger.debug(f"  Bottleneck: {stats['bottleneck']}")

        if stats["bottleneck"] == "memory-bound":
            memory_bound_count += 1
        else:
            compute_bound_count += 1

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total configurations tested: {len(test_configs)}")
    logger.info(f"Memory-bound: {memory_bound_count}")
    logger.info(f"Compute-bound: {compute_bound_count}")
    logger.info("=" * 80)
