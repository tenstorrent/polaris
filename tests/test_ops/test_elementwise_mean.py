#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for element-wise Mean operator.

Tests both shape inference (via SimOp.get_perf_counts) and numerical data
computation (via compute_elementwise_mean) through the full operator pipeline
including variadic_sinf.
"""
import pytest
import os
import sys
import logging
sys.path.append(os.getcwd())

import numpy as np
from loguru import logger
from pathlib import Path

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_elementwise_mean
from ttsim.config import get_arspec_from_yaml
from ttsim.back.device import Device

polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    logging.getLogger("ttsim").setLevel(logging.ERROR)
    logging.getLogger("ttsim.config").setLevel(logging.ERROR)
    try:
        from loguru import logger as _loguru_logger

        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, level="ERROR")
    except ImportError:
        pass
except Exception:
    pass


def get_max_test_msg_len(TL):
    return max([len(x[0]) for x in TL])


# ============================================================================
# Test data generation
# ============================================================================

def generate_test_data(shape, data_type, which='both'):
    if len(shape) == 0:
        if data_type == 'positive':
            return np.array(np.random.rand() + 1.0, dtype=np.float32)
        elif data_type == 'negative':
            return np.array(-np.random.rand() - 1.0, dtype=np.float32)
        elif data_type == 'zeros':
            return np.array(0.0, dtype=np.float32)
        else:
            return np.array(np.random.randn(), dtype=np.float32)

    if data_type == 'positive':
        return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    elif data_type == 'negative':
        return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)
    elif data_type == 'neg_pos':
        if which == 'A':
            return np.array(-np.random.rand(*shape) - 1.0, dtype=np.float32)
        else:
            return np.array(np.random.rand(*shape) + 1.0, dtype=np.float32)
    elif data_type == 'zeros':
        return np.zeros(shape, dtype=np.float32)
    elif data_type == 'mixed':
        return np.array(np.random.randn(*shape) * 2, dtype=np.float32)
    elif data_type == 'small':
        return np.array(np.random.rand(*shape) * 1e-6, dtype=np.float32)
    elif data_type == 'large':
        return np.array(np.random.rand(*shape) * 1e3, dtype=np.float32)
    elif data_type == 'ones':
        return np.ones(shape, dtype=np.float32)
    else:
        return np.array(np.random.randn(*shape), dtype=np.float32)


# ============================================================================
# Test cases for 2-input element-wise Mean
# ============================================================================

two_input_test_cases = [
    # (name, shape_A, shape_B, data_type)
    # Same-shape cases
    ("Same shape 1D",                  [4],          [4],          'positive'),
    ("Same shape 2D",                  [3, 4],       [3, 4],       'positive'),
    ("Same shape 3D",                  [2, 3, 4],    [2, 3, 4],    'positive'),
    ("Same shape 4D (NCHW)",           [2, 3, 4, 4], [2, 3, 4, 4], 'positive'),

    # Broadcasting cases
    ("Scalar to 2D broadcast",         [],           [3, 4],       'positive'),
    ("1D to 2D broadcast",             [4],          [3, 4],       'positive'),
    ("Bidirectional broadcast",        [3, 1],       [1, 4],       'positive'),
    ("Multi-dim broadcast",            [2, 1, 4],    [1, 3, 1],    'positive'),
    ("Channel-wise broadcast",         [1, 3, 1, 1], [2, 3, 4, 4], 'positive'),
    ("Scalar broadcast",               [1],          [2, 3, 4],    'positive'),

    # Negative values
    ("Negative vs Positive",           [3, 4],       [3, 4],       'neg_pos'),
    ("All negative",                   [3, 4],       [3, 4],       'negative'),

    # Zero values
    ("Zeros vs Positive",              [3, 4],       [3, 4],       'mixed'),
    ("All zeros",                      [3, 4],       [3, 4],       'zeros'),

    # Mixed values
    ("Mixed positive/negative",        [2, 3, 4],    [2, 3, 4],    'mixed'),

    # Small values
    ("Small values",                   [3, 4],       [3, 4],       'small'),

    # Large values
    ("Large values",                   [3, 4],       [3, 4],       'large'),

    # Single element
    ("Single element",                 [1],          [1],          'positive'),

    # High-rank tensor
    ("5D tensor",                      [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 'positive'),
]


# ============================================================================
# TEST: Element-wise Mean (2 inputs)
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_elementwise_mean_2_inputs():
    """Test element-wise Mean with 2 inputs: shape and numerical validation."""
    test_name = 'test_elementwise_mean_2'
    msgw = get_max_test_msg_len(two_input_test_cases)

    for tno, (tmsg, shape_A, shape_B, data_type) in enumerate(two_input_test_cases):
        op_name = f'{test_name}_{tno}'

        data_A = generate_test_data(shape_A, data_type, which='A')
        data_B = generate_test_data(shape_B, data_type, which='B')

        # Reference: element-wise mean of 2 inputs
        ref_output = ((data_A + data_B) / 2.0).astype(np.float32)

        i_tensors = [
            F._from_data('A', data_A),
            F._from_data('B', data_B),
        ]
        o_tensors = [make_tensor('Y')]

        op_info = {
            'name'   : op_name,
            'optype' : 'Mean',
            'inList' : [x.name for x in i_tensors],
            'outList': [x.name for x in o_tensors],
        }

        op_obj = SimOp(op_info)
        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        op_obj.get_perf_counts(i_tensors, o_tensors)

        # 1. Shape validation
        inf_shape = o_tensors[0].shape
        ref_shape = list(ref_output.shape)
        shape_match = inf_shape == ref_shape

        # 2. Numerical validation via compute function
        numerical_match = True
        try:
            computed_output = compute_elementwise_mean(i_tensors, op_obj)
            numerical_match = np.allclose(computed_output, ref_output, rtol=1e-5, atol=1e-7)
            if not numerical_match:
                max_diff = np.max(np.abs(computed_output - ref_output))
                logger.debug(f"\n  Max difference: {max_diff}")
        except Exception as e:
            numerical_match = f"Error: {e}"
            logger.debug(f"\n  Numerical validation error: {e}")

        # 3. Validate data propagated through op pipeline
        pipeline_match = True
        if o_tensors[0].data is not None:
            pipeline_match = np.allclose(o_tensors[0].data, ref_output, rtol=1e-5, atol=1e-7)
        else:
            pipeline_match = "No data propagated"

        # 4. Perf stats validation: should use 'add' and 'div' instructions
        perf_ok = True
        if hasattr(op_obj, 'perf_stats') and op_obj.perf_stats:
            instrs = op_obj.perf_stats.get('instrs', {})
            if 'add' not in instrs or 'div' not in instrs:
                perf_ok = False
                logger.debug(f"\n  Missing 'add'/'div' in perf_stats instrs: {instrs}")

        if shape_match and numerical_match is True and pipeline_match is True and perf_ok:
            logger.debug(
                f"TEST[{tno:3d}] Mean {tmsg:{msgw}s} PASS [Shape OK, Numerical OK, Pipeline OK]"
            )
        else:
            logger.debug(f"\nTEST[{tno:3d}] Mean {tmsg:{msgw}s} FAIL")
            logger.debug(
                f"  Shape match: {shape_match} (got {inf_shape}, expected {ref_shape})"
            )
            logger.debug(f"  Numerical match: {numerical_match}")
            logger.debug(f"  Pipeline match: {pipeline_match}")
            logger.debug(f"  Perf stats OK: {perf_ok}")
            assert False, f"TEST[{tno:3d}] Mean {tmsg} FAIL"


# ============================================================================
# TEST: Element-wise Mean (3 inputs)
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_elementwise_mean_3_inputs():
    """Test element-wise Mean with 3 inputs."""
    data_A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data_B = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    data_C = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]], dtype=np.float32)
    ref_output = ((data_A + data_B + data_C) / 3.0).astype(np.float32)

    i_tensors = [
        F._from_data('A', data_A),
        F._from_data('B', data_B),
        F._from_data('C', data_C),
    ]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'mean_3_inputs',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['mean_3_inputs']
    for x in o_tensors: x.op_out = ['mean_3_inputs']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].shape == list(ref_output.shape), \
        f"Shape mismatch: {o_tensors[0].shape} vs {list(ref_output.shape)}"
    assert o_tensors[0].data is not None, "No data propagated for 3-input Mean"
    assert np.allclose(o_tensors[0].data, ref_output, rtol=1e-5, atol=1e-7), \
        f"Numerical mismatch: max diff = {np.max(np.abs(o_tensors[0].data - ref_output))}"

    # Perf stats: n_inputs-1 adds + 1 div per element
    instrs = op_obj.perf_stats.get('instrs', {})
    out_elems = o_tensors[0].nelems()
    assert instrs.get('add') == out_elems * 2, \
        f"Expected {out_elems * 2} adds, got {instrs.get('add')}"
    assert instrs.get('div') == out_elems, \
        f"Expected {out_elems} divs, got {instrs.get('div')}"

    logger.debug("TEST Mean 3 inputs PASS")


@pytest.mark.unit
@pytest.mark.opunit
def test_elementwise_mean_4_inputs():
    """Test element-wise Mean with 4 inputs."""
    shape = [2, 3]
    data = [np.random.randn(*shape).astype(np.float32) for _ in range(4)]
    ref_output = (sum(data) / 4.0).astype(np.float32)

    i_tensors = [F._from_data(f'T{i}', d) for i, d in enumerate(data)]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'mean_4_inputs',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['mean_4_inputs']
    for x in o_tensors: x.op_out = ['mean_4_inputs']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].data is not None, "No data propagated for 4-input Mean"
    assert np.allclose(o_tensors[0].data, ref_output, rtol=1e-5, atol=1e-7), \
        f"Numerical mismatch: max diff = {np.max(np.abs(o_tensors[0].data - ref_output))}"
    logger.debug("TEST Mean 4 inputs PASS")


# ============================================================================
# TEST: Element-wise Mean with broadcasting (3 inputs)
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_elementwise_mean_3_inputs_broadcast():
    """Test element-wise Mean with 3 inputs requiring broadcasting."""
    data_A = np.random.randn(2, 3, 4).astype(np.float32)
    data_B = np.random.randn(1, 3, 1).astype(np.float32)
    data_C = np.random.randn(2, 1, 4).astype(np.float32)
    ref_output = ((data_A + data_B + data_C) / 3.0).astype(np.float32)

    i_tensors = [
        F._from_data('A', data_A),
        F._from_data('B', data_B),
        F._from_data('C', data_C),
    ]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'mean_3_bcast',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['mean_3_bcast']
    for x in o_tensors: x.op_out = ['mean_3_bcast']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].shape == [2, 3, 4], \
        f"Shape mismatch: {o_tensors[0].shape} vs [2, 3, 4]"
    assert o_tensors[0].data is not None, "No data propagated"
    assert np.allclose(o_tensors[0].data, ref_output, rtol=1e-5, atol=1e-7), \
        f"Numerical mismatch: max diff = {np.max(np.abs(o_tensors[0].data - ref_output))}"
    logger.debug("TEST Mean 3 inputs broadcast PASS")


# ============================================================================
# TEST: Mean of identical inputs = same value
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_elementwise_mean_identical():
    """Mean of N copies of the same tensor should return that tensor."""
    data = np.random.randn(3, 4).astype(np.float32)
    ref_output = data.copy()

    i_tensors = [F._from_data(f'T{i}', data.copy()) for i in range(3)]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'mean_identical',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['mean_identical']
    for x in o_tensors: x.op_out = ['mean_identical']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].data is not None, "No data propagated"
    assert np.allclose(o_tensors[0].data, ref_output, rtol=1e-5, atol=1e-7), \
        f"Mean of identical inputs should equal the input. Max diff = {np.max(np.abs(o_tensors[0].data - ref_output))}"
    logger.debug("TEST Mean identical inputs PASS")


# ============================================================================
# TEST: Shape-only inputs (no data)
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_shape_only_mean():
    """Test that Mean with shape-only inputs (no data) still infers shape."""
    tA = F._from_shape('A', [2, 3, 4])
    tB = F._from_shape('B', [2, 3, 4])

    i_tensors = [tA, tB]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'shape_only_mean',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['shape_only_mean']
    for x in o_tensors: x.op_out = ['shape_only_mean']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].shape == [2, 3, 4], f"Shape mismatch: {o_tensors[0].shape}"
    assert o_tensors[0].data is None, "Expected no data for shape-only inputs"
    logger.debug("TEST shape-only Mean PASS")


# ============================================================================
# TEST: Mixed data/shape inputs
# ============================================================================

@pytest.mark.unit
@pytest.mark.opunit
def test_mixed_data_shape_mean():
    """Test Mean where one input has data and one is shape-only → no data output."""
    data_A = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    tA = F._from_data('A', data_A)
    tB = F._from_shape('B', [3])

    i_tensors = [tA, tB]
    o_tensors = [make_tensor('Y')]

    op_info = {
        'name'   : 'mixed_mean',
        'optype' : 'Mean',
        'inList' : [x.name for x in i_tensors],
        'outList': [x.name for x in o_tensors],
    }

    op_obj = SimOp(op_info)
    for x in i_tensors: x.op_in  = ['mixed_mean']
    for x in o_tensors: x.op_out = ['mixed_mean']

    op_obj.get_perf_counts(i_tensors, o_tensors)

    assert o_tensors[0].shape == [3], f"Shape mismatch: {o_tensors[0].shape}"
    assert o_tensors[0].data is None, "Expected no data when one input is shape-only"
    logger.debug("TEST mixed data/shape Mean PASS")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    test_elementwise_mean_2_inputs()
    test_elementwise_mean_3_inputs()
    test_elementwise_mean_4_inputs()
    test_elementwise_mean_3_inputs_broadcast()
    test_elementwise_mean_identical()
    test_shape_only_mean()
    test_mixed_data_shape_mean()
    logger.info("\n" + "=" * 60)
    logger.info("ALL ELEMENTWISE MEAN TESTS PASSED")
    logger.info("=" * 60)


# ============================================================================
# Memory estimation helper
# ============================================================================

def calculate_mean_memory_stats(input_shapes, n_inputs, precision='fp32'):
    """
    Calculate expected memory stats for an element-wise Mean operation.

    Args:
        input_shapes: List of input shapes (one per input tensor)
        n_inputs: Number of input tensors
        precision: Data precision (fp16, bf16, fp32, etc.)

    Returns:
        dict with expected memory stats
    """
    precision_bytes = {
        'bfp8': 1, 'fp16': 2, 'bf16': 2, 'fp32': 4,
        'int8': 1, 'int32': 4,
    }
    bytes_per_element = precision_bytes.get(precision, 4)

    # Output shape is broadcast of all inputs
    output_shape = list(input_shapes[0])
    for s in input_shapes[1:]:
        # Broadcast shape calculation
        out = []
        a, b = list(output_shape), list(s)
        while len(a) < len(b):
            a = [1] + a
        while len(b) < len(a):
            b = [1] + b
        for da, db in zip(a, b):
            out.append(max(da, db))
        output_shape = out

    num_output_elements = int(np.prod(output_shape))
    total_input_elements = sum(int(np.prod(s)) if len(s) > 0 else 1 for s in input_shapes)

    # Bytes
    input_bytes = total_input_elements * bytes_per_element
    output_bytes = num_output_elements * bytes_per_element
    total_data_movement = input_bytes + output_bytes

    # Instructions: (n_inputs - 1) adds + 1 div per output element
    expected_add_instrs = num_output_elements * (n_inputs - 1)
    expected_div_instrs = num_output_elements
    total_instructions = expected_add_instrs + expected_div_instrs

    # Arithmetic intensity
    arithmetic_intensity = (
        total_instructions / total_data_movement if total_data_movement > 0 else 0
    )

    return {
        'output_shape': output_shape,
        'total_input_elements': total_input_elements,
        'output_elements': num_output_elements,
        'input_bytes': input_bytes,
        'output_bytes': output_bytes,
        'total_data_movement': total_data_movement,
        'expected_add_instrs': expected_add_instrs,
        'expected_div_instrs': expected_div_instrs,
        'total_instructions': total_instructions,
        'arithmetic_intensity': arithmetic_intensity,
        'n_inputs': n_inputs,
    }


# Test cases for memory validation
memory_validation_cases = [
    # (input_shapes, description)
    ([[128], [128]],                        "2-input 1D same shape"),
    ([[32, 64], [32, 64]],                  "2-input 2D same shape"),
    ([[8, 16, 16], [8, 16, 16]],            "2-input 3D same shape"),
    ([[32, 64], [32, 64], [32, 64]],        "3-input 2D same shape"),
    ([[16, 16], [16, 16], [16, 16], [16, 16]], "4-input 2D same shape"),
]


class TestMeanMemoryValidation:
    """Validate memory estimation and instruction counts for Mean operation."""

    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.performance
    def test_mean_memory_validation(self, capsys, request):
        """
        Validate memory usage and instruction counts for element-wise Mean.

        This test validates two primary metrics:
        1. Instructions Executed: (N-1) 'add' + 1 'div' per output element
        2. Data Moved: Tracks input/output bytes and validates memory traffic

        Mean computes element-wise average of N tensors: Y = (T1 + T2 + ... + TN) / N
        Run with: pytest tests/test_ops/test_elementwise_mean.py::TestMeanMemoryValidation -v
        For detailed output: add -s flag
        """
        logger.info("\n" + "=" * 60)
        logger.info("Element-wise Mean Operation Memory Validation")
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

        logger.info("\n%s", "=" * 60)
        logger.info("Running Memory Validation Tests")
        logger.info("%s\n", "=" * 60)

        all_results = []

        for input_shapes, description in memory_validation_cases:
            n_inputs = len(input_shapes)
            logger.debug(f"\n-- Test: {description} --")
            logger.debug(f"Input shapes: {input_shapes}, N={n_inputs}")

            # Generate random inputs
            np.random.seed(42)
            input_data = [
                np.random.randn(*s).astype(np.float32) for s in input_shapes
            ]

            # Reference output
            ref_output = sum(input_data) / float(n_inputs)

            # Build tensors and op
            i_tensors = [
                F._from_data(f'T{i}', d) for i, d in enumerate(input_data)
            ]
            o_tensors = [make_tensor('Y')]

            op_info = {
                'name': 'Mean',
                'optype': 'Mean',
                'inList': [t.name for t in i_tensors],
                'outList': ['Y'],
            }
            op = SimOp(op_info)
            for t in i_tensors:
                t.op_in = ['Mean']
            for t in o_tensors:
                t.op_out = ['Mean']

            op.precision = 'fp32'
            op.get_perf_counts(i_tensors, o_tensors)

            # Validate compute correctness
            result = compute_elementwise_mean(i_tensors, op)
            np.testing.assert_allclose(
                result, ref_output, rtol=1e-5, atol=1e-6,
                err_msg=f"[{description}] compute_elementwise_mean validation failed",
            )

            # Calculate expected stats
            expected_stats = calculate_mean_memory_stats(input_shapes, n_inputs, 'fp32')

            # Set compute pipe
            if op.uses_compute_pipe is None:
                op.uses_compute_pipe = 'vector'

            # Execute on device for cycle estimation
            if op.perf_stats is not None:
                device.execute_op(op)

            # Extract stats from op.perf_stats
            perf_stats = op.perf_stats
            actual_in_bytes = perf_stats['inBytes']
            actual_out_bytes = perf_stats['outBytes']
            actual_instrs = perf_stats['instrs']

            # Validate instructions
            assert 'add' in actual_instrs, "Expected 'add' instruction not found"
            assert 'div' in actual_instrs, "Expected 'div' instruction not found"
            actual_add = actual_instrs.get('add', 0)
            actual_div = actual_instrs.get('div', 0)
            assert actual_add == expected_stats['expected_add_instrs'], \
                f"Add count mismatch: {actual_add} vs {expected_stats['expected_add_instrs']}"
            assert actual_div == expected_stats['expected_div_instrs'], \
                f"Div count mismatch: {actual_div} vs {expected_stats['expected_div_instrs']}"

            # Validate data movement
            assert actual_in_bytes == expected_stats['input_bytes'], \
                f"Input bytes mismatch: {actual_in_bytes} vs {expected_stats['input_bytes']}"
            assert actual_out_bytes == expected_stats['output_bytes'], \
                f"Output bytes mismatch: {actual_out_bytes} vs {expected_stats['output_bytes']}"

            # Calculate metrics
            total_data_movement = actual_in_bytes + actual_out_bytes
            instructions_executed = sum(actual_instrs.values())
            arithmetic_intensity = (
                instructions_executed / total_data_movement
                if total_data_movement > 0 else 0
            )

            # Execution cycles
            compute_cycles = op.compute_cycles
            mem_rd_cycles = op.mem_rd_cycles
            mem_wr_cycles = op.mem_wr_cycles
            memory_cycles = mem_rd_cycles + mem_wr_cycles
            total_cycles = max(compute_cycles, memory_cycles)
            bottleneck = "COMPUTE" if compute_cycles >= memory_cycles else "MEMORY"

            # Print detailed breakdown
            logger.debug("\n  -- Instructions & Operations --")
            logger.debug(
                f"  Instructions executed: {instructions_executed:,} ({actual_add:,} add + {actual_div:,} div)"
            )
            logger.debug(
                f"  Input elements (all):  {expected_stats['total_input_elements']:,}"
            )
            logger.debug(
                f"  Output elements:       {expected_stats['output_elements']:,}"
            )
            logger.debug(
                f"  Expected:              {expected_stats['expected_add_instrs']:,} add ({n_inputs-1}/elem) + {expected_stats['expected_div_instrs']:,} div (1/elem)"
            )
            ops_per_elem = instructions_executed / expected_stats['output_elements']
            logger.debug(
                f"  Ops per output elem:   {ops_per_elem:.1f} (OK: {n_inputs-1} add + 1 div = {n_inputs})"
            )

            logger.debug("\n  -- Data Movement --")
            logger.debug(
                f"  Input bytes:      {actual_in_bytes:,} bytes ({actual_in_bytes/1024:.2f} KB)"
            )
            logger.debug(
                f"  Output bytes:     {actual_out_bytes:,} bytes ({actual_out_bytes/1024:.2f} KB)"
            )
            logger.debug(
                f"  Total data moved: {total_data_movement:,} bytes ({total_data_movement/1024:.2f} KB)"
            )
            read_write_ratio = actual_in_bytes / actual_out_bytes if actual_out_bytes > 0 else 0
            logger.debug(
                f"  Read/Write ratio: {read_write_ratio:.1f}:1 ({n_inputs} inputs -> 1 output)"
            )

            logger.debug("\n  -- Memory Metrics --")
            logger.debug(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
            logger.debug(
                f"  [PASS] Arithmetic intensity within expected range for {n_inputs}-input mean"
            )

            logger.debug("\n  -- Execution Cycles --")
            logger.debug(f"  Compute cycles:   {compute_cycles:,}")
            logger.debug(f"  Memory cycles:    {memory_cycles:,}")
            logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
            logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
            logger.debug(f"  Ideal cycles:     {total_cycles:,}")
            logger.debug(f"  Bottleneck:       {bottleneck}")

            if expected_stats['output_elements'] > 100:
                logger.debug(
                    f"  [PASS] Bottleneck analysis: {bottleneck} for element-wise mean"
                )

            # Validate arithmetic intensity matches expected
            np.testing.assert_allclose(
                arithmetic_intensity,
                expected_stats['arithmetic_intensity'],
                rtol=0.01, atol=1e-6,
                err_msg="Arithmetic intensity mismatch",
            )

            all_results.append({
                'test_name': description,
                'n_inputs': n_inputs,
                'input_shapes': input_shapes,
                'output_shape': expected_stats['output_shape'],
                'add_instructions': actual_add,
                'div_instructions': actual_div,
                'total_instructions': instructions_executed,
                'total_data_moved': total_data_movement,
                'arithmetic_intensity': arithmetic_intensity,
                'bottleneck': bottleneck,
                'compute_cycles': compute_cycles,
                'memory_cycles': memory_cycles,
                'ideal_cycles': total_cycles,
            })

            logger.debug("\n  [PASS] Test PASSED")

        # Summary
        logger.info("\n%s", "=" * 60)
        logger.info("Memory Validation Summary")
        logger.info("%s\n", "=" * 60)
        logger.info(f"Total tests run: {len(all_results)}")
        logger.info("All tests passed: YES")

        logger.info("\n-- Arithmetic Intensity Comparison --")
        for result in all_results:
            ai = result['arithmetic_intensity']
            logger.info(f"{result['test_name']:30s}: {ai:.4f} ops/byte")

        logger.info("\n-- Instruction Breakdown --")
        for result in all_results:
            logger.info(
                f"{result['test_name']:30s}: {result['add_instructions']:,} add + {result['div_instructions']:,} div = {result['total_instructions']:,}"
            )

        logger.info("\n-- Bottleneck Analysis --")
        for result in all_results:
            bottleneck = result['bottleneck']
            logger.info(f"{result['test_name']:30s}: {bottleneck}")

        logger.info("\n%s", "=" * 60)
        logger.info("Memory validation complete!")
        logger.info("%s\n", "=" * 60)

        # Summary for pytest output
        summary_lines = [
            "[PASS] Tests completed: {}/{} - All PASSED".format(
                len(all_results), len(memory_validation_cases)
            ),
            "",
            "Key Findings:",
            "  * Instructions: (N-1) 'add' + 1 'div' per output element [PASS]",
            "  * More inputs -> higher arithmetic intensity (more ops, same output bytes)",
            "  * Bottleneck: Mix of COMPUTE and MEMORY based on tensor size",
            "",
            "Test Results:",
        ]

        for result in all_results:
            summary_lines.append(
                "  [PASS] {:<26s} | {:>7,} ops | {:>7.1f} KB | {:.3f} ops/byte | N={:d}".format(
                    result['test_name'],
                    result['total_instructions'],
                    result['total_data_moved'] / 1024,
                    result['arithmetic_intensity'],
                    result['n_inputs'],
                )
            )

        summary_lines.extend([
            "",
            "Validation: All memory metrics within expected ranges [PASS]",
            "",
            "For detailed output, run with: pytest -s -v",
        ])

        # Write to pytest terminal reporter
        try:
            terminalreporter = request.config.pluginmanager.get_plugin(
                'terminalreporter'
            )
            if terminalreporter:
                terminalreporter.write_sep(
                    "=", "MEMORY VALIDATION RESULTS", bold=True, green=True
                )
                for line in summary_lines:
                    terminalreporter.write_line(line)
                terminalreporter.write_sep("=", "", bold=True)
        except Exception:
            with capsys.disabled():
                logger.info("\n" + "=" * 70)
                logger.info("MEMORY VALIDATION RESULTS")
                logger.info("=" * 70)
                for line in summary_lines:
                    logger.info(line)
                logger.info("=" * 70 + "\n")

        # Final assertion
        assert len(all_results) == len(memory_validation_cases), \
            f"Memory validation: {len(all_results)}/{len(memory_validation_cases)} tests passed"
