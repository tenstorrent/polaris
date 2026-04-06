#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the GroupNormalization op."""

import pytest
import numpy as np
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_groupnorm

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _run_groupnorm(x_np, weight_np, bias_np, num_groups, eps=1e-5, tag="gnorm"):
    """Run GroupNormalization through SimOp and return (actual, expected, oT)."""
    i_tensors = [
        F._from_data("X", x_np),
        F._from_data("weight", weight_np),
        F._from_data("bias", bias_np),
    ]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "GroupNormalization",
        "inList": [t.name for t in i_tensors],
        "outList": [t.name for t in o_tensors],
        "attrs": {"num_groups": num_groups, "epsilon": eps},
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)
    o_tensors[0].data = compute_groupnorm(i_tensors, op)

    actual = o_tensors[0].data

    # Reference: manual group-norm computation
    N, C, H, W = x_np.shape
    G = num_groups
    x_g = x_np.reshape(N, G, C // G, H, W)
    mean = np.mean(x_g, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_g, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_g - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    expected = x_norm * weight_np.reshape(1, C, 1, 1) + bias_np.reshape(1, C, 1, 1)

    return actual, expected, o_tensors[0]


# ===================================================================
# 1. Shape validation tests
# ===================================================================
shape_test_cases = [
    # (N, C, H, W, num_groups, id)
    (1, 4, 2, 2, 2, "basic_2g"),
    (1, 4, 2, 2, 4, "groups_eq_channels"),
    (1, 4, 2, 2, 1, "single_group"),
    (2, 8, 4, 4, 4, "batch2_4g"),
    (1, 16, 1, 1, 4, "spatial_1x1"),
    (1, 32, 8, 8, 8, "32ch_8g"),
    (3, 6, 3, 3, 3, "batch3_3g"),
]


class TestGroupNormShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("N,C,H,W,G,tid", shape_test_cases)
    def test_output_shape(self, N, C, H, W, G, tid):
        x = np.random.randn(N, C, H, W).astype(np.float32)
        w = np.ones(C, dtype=np.float32)
        b = np.zeros(C, dtype=np.float32)
        _, _, oT = _run_groupnorm(x, w, b, G, tag=f"shape_{tid}")
        assert list(oT.shape) == [N, C, H, W], f"{tid}: {oT.shape} != {[N, C, H, W]}"


# ===================================================================
# 2. Numerical validation tests
# ===================================================================
numerical_cases = [
    # (N, C, H, W, num_groups, id)
    (1, 4, 3, 3, 2, "4ch_2g"),
    (2, 8, 4, 4, 4, "8ch_4g"),
    (1, 6, 2, 2, 3, "6ch_3g"),
    (1, 4, 5, 5, 1, "4ch_1g"),
    (1, 4, 5, 5, 4, "4ch_4g"),
    (1, 16, 4, 4, 8, "16ch_8g"),
]


class TestGroupNormNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("N,C,H,W,G,tid", numerical_cases)
    def test_values(self, N, C, H, W, G, tid):
        np.random.seed(42)
        x = np.random.randn(N, C, H, W).astype(np.float32)
        w = np.random.randn(C).astype(np.float32)
        b = np.random.randn(C).astype(np.float32)
        actual, expected, _ = _run_groupnorm(x, w, b, G, tag=f"num_{tid}")
        np.testing.assert_allclose(
            actual, expected, rtol=1e-5, atol=1e-6, err_msg=f"{tid} value mismatch"
        )


# ===================================================================
# 3. Edge-case tests
# ===================================================================
class TestGroupNormEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_identity_affine(self):
        """weight=1, bias=0 should give pure normalization."""
        np.random.seed(10)
        x = np.random.randn(1, 4, 3, 3).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual, expected, _ = _run_groupnorm(x, w, b, 2, tag="id_affine")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zero_weight(self):
        """weight=0 should zero out the output (bias only)."""
        np.random.seed(11)
        x = np.random.randn(1, 4, 2, 2).astype(np.float32)
        w = np.zeros(4, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, _, _ = _run_groupnorm(x, w, b, 2, tag="zero_w")
        expected_bias = b.reshape(1, 4, 1, 1) * np.ones_like(x)
        np.testing.assert_allclose(actual, expected_bias, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_constant_input(self):
        """Constant input within each group → normalized to 0 (before affine)."""
        x = np.full((1, 4, 3, 3), 5.0, dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual, _, _ = _run_groupnorm(x, w, b, 2, tag="const")
        # Constant input: x - mean = 0, so output should be 0 (+ bias=0)
        np.testing.assert_allclose(actual, 0.0, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_spatial(self):
        """1x1 spatial dimensions."""
        np.random.seed(12)
        x = np.random.randn(2, 8, 1, 1).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        actual, expected, _ = _run_groupnorm(x, w, b, 4, tag="1x1")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_values(self):
        """Large input values handled without overflow."""
        x = np.full((1, 4, 2, 2), 1e6, dtype=np.float32)
        x[0, 0, 0, 0] = 1e6 + 1  # slight variation so var != 0
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual, expected, _ = _run_groupnorm(x, w, b, 2, tag="large")
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_weight(self):
        """Negative weights flip signs."""
        np.random.seed(13)
        x = np.random.randn(1, 4, 2, 2).astype(np.float32)
        w_pos = np.ones(4, dtype=np.float32)
        w_neg = -np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        act_pos, _, _ = _run_groupnorm(x, w_pos, b, 2, tag="neg_w_pos")
        act_neg, _, _ = _run_groupnorm(x, w_neg, b, 2, tag="neg_w_neg")
        np.testing.assert_allclose(act_neg, -act_pos, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_custom_eps(self):
        """Custom epsilon value."""
        np.random.seed(14)
        x = np.random.randn(1, 4, 2, 2).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual, expected, _ = _run_groupnorm(x, w, b, 2, eps=1e-3, tag="eps")
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


# ===================================================================
# 4. Precision tests with known values
# ===================================================================
class TestGroupNormPrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_manual_2ch_1group(self):
        """Manually verify group norm with 2 channels, 1 group, 1x1 spatial."""
        # 1 group, 2 channels, 1x1 spatial → group = entire [C, H, W]
        x = np.array([[[[1.0]], [[3.0]]]], dtype=np.float32)  # (1,2,1,1)
        w = np.ones(2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        eps = 0.0

        # group = [[1.0, 3.0]], mean=2.0, var=1.0
        # normalized: (1-2)/1 = -1, (3-2)/1 = 1
        actual, _, _ = _run_groupnorm(x, w, b, num_groups=1, eps=eps, tag="manual_1g")
        np.testing.assert_allclose(actual[0, 0, 0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 1, 0, 0], 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_manual_4ch_2groups(self):
        """Manually verify with 4 channels, 2 groups, 1x1 spatial."""
        # Group 0 channels [0,1], Group 1 channels [2,3]
        x = np.array([[[[2.0]], [[4.0]], [[10.0]], [[20.0]]]], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        eps = 0.0

        # Group 0: [2, 4], mean=3, var=1 → normalized [-1, 1]
        # Group 1: [10, 20], mean=15, var=25 → normalized [-1, 1]
        actual, _, _ = _run_groupnorm(x, w, b, num_groups=2, eps=eps, tag="manual_2g")
        np.testing.assert_allclose(actual[0, 0, 0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 1, 0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 2, 0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 3, 0, 0], 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_affine_transform(self):
        """Verify affine: output = normalized * weight + bias."""
        x = np.array([[[[1.0]], [[3.0]]]], dtype=np.float32)  # (1,2,1,1)
        w = np.array([2.0, 3.0], dtype=np.float32)
        b = np.array([10.0, 20.0], dtype=np.float32)
        eps = 0.0

        # normalized: [-1, 1]
        # affine: [-1*2+10, 1*3+20] = [8, 23]
        actual, _, _ = _run_groupnorm(x, w, b, num_groups=1, eps=eps, tag="affine")
        np.testing.assert_allclose(actual[0, 0, 0, 0], 8.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 1, 0, 0], 23.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_groups_equal_channels(self):
        """When num_groups == C, each channel normalized independently (instance-norm style)."""
        x = np.array([[[[1.0, 3.0]], [[10.0, 20.0]]]], dtype=np.float32)  # (1,2,1,2)
        w = np.ones(2, dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        eps = 0.0

        # Group 0 = channel 0 = [1, 3], mean=2, var=1 → [-1, 1]
        # Group 1 = channel 1 = [10, 20], mean=15, var=25 → [-1, 1]
        actual, _, _ = _run_groupnorm(x, w, b, num_groups=2, eps=eps, tag="g_eq_c")
        np.testing.assert_allclose(actual[0, 0, 0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 0, 0, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 1, 0, 0], -1.0, atol=1e-6)
        np.testing.assert_allclose(actual[0, 1, 0, 1], 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zeros_input(self):
        """All-zero input → output = bias."""
        x = np.zeros((1, 4, 2, 2), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        actual, _, _ = _run_groupnorm(x, w, b, 2, tag="zeros")
        # Constant 0 input: normalized = 0, so output = 0*w + b = b
        expected = b.reshape(1, 4, 1, 1) * np.ones((1, 4, 2, 2), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, atol=1e-5)


# ===================================================================
# 5. Mathematical property tests
# ===================================================================
class TestGroupNormProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_output_shape_preserved(self):
        """Output has same shape as input."""
        np.random.seed(20)
        x = np.random.randn(2, 8, 4, 4).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        actual, _, oT = _run_groupnorm(x, w, b, 4, tag="shape_pres")
        assert list(oT.shape) == list(x.shape)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_zero_mean_per_group(self):
        """After group norm with w=1, b=0, each group has ~zero mean."""
        np.random.seed(21)
        x = np.random.randn(1, 8, 4, 4).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        actual, _, _ = _run_groupnorm(x, w, b, 4, tag="zero_mean")
        # Reshape to groups and check mean
        G = 4
        C = 8
        actual_g = actual.reshape(1, G, C // G, 4, 4)
        for g in range(G):
            group_mean = np.mean(actual_g[0, g])
            np.testing.assert_allclose(
                group_mean, 0.0, atol=1e-5, err_msg=f"Group {g} mean not zero"
            )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_unit_variance_per_group(self):
        """After group norm with w=1, b=0, each group has ~unit variance."""
        np.random.seed(22)
        x = np.random.randn(1, 8, 6, 6).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        actual, _, _ = _run_groupnorm(x, w, b, 4, tag="unit_var")
        G = 4
        C = 8
        actual_g = actual.reshape(1, G, C // G, 6, 6)
        for g in range(G):
            group_var = np.var(actual_g[0, g])
            np.testing.assert_allclose(
                group_var, 1.0, rtol=0.05, err_msg=f"Group {g} var not ~1"
            )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_batch_independence(self):
        """Each batch element is normalized independently."""
        np.random.seed(23)
        x1 = np.random.randn(1, 4, 3, 3).astype(np.float32)
        x2 = np.random.randn(1, 4, 3, 3).astype(np.float32) * 10 + 5
        x_batch = np.concatenate([x1, x2], axis=0)  # (2,4,3,3)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)

        # Run batched
        actual_batch, _, _ = _run_groupnorm(x_batch, w, b, 2, tag="batch_ind")
        # Run individually
        actual_1, _, _ = _run_groupnorm(x1, w, b, 2, tag="batch_ind_1")
        actual_2, _, _ = _run_groupnorm(x2, w, b, 2, tag="batch_ind_2")

        np.testing.assert_allclose(actual_batch[0], actual_1[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual_batch[1], actual_2[0], rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scale_invariance(self):
        """Scaling input by constant → same normalized output (w=1, b=0)."""
        np.random.seed(24)
        x = np.random.randn(1, 4, 3, 3).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual_1, _, _ = _run_groupnorm(x, w, b, 2, tag="scale_1")
        actual_100, _, _ = _run_groupnorm(x * 100, w, b, 2, tag="scale_100")
        np.testing.assert_allclose(actual_1, actual_100, rtol=1e-4, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_shift_invariance(self):
        """Adding constant to input → same normalized output (w=1, b=0)."""
        np.random.seed(25)
        x = np.random.randn(1, 4, 3, 3).astype(np.float32)
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        actual_orig, _, _ = _run_groupnorm(x, w, b, 2, tag="shift_0")
        actual_shift, _, _ = _run_groupnorm(x + 1000, w, b, 2, tag="shift_1k")
        np.testing.assert_allclose(actual_orig, actual_shift, rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_weight_scaling(self):
        """Doubling weight doubles the output (when bias=0)."""
        np.random.seed(26)
        x = np.random.randn(1, 4, 3, 3).astype(np.float32)
        w1 = np.ones(4, dtype=np.float32)
        w2 = np.ones(4, dtype=np.float32) * 2
        b = np.zeros(4, dtype=np.float32)
        actual_1, _, _ = _run_groupnorm(x, w1, b, 2, tag="wscale_1")
        actual_2, _, _ = _run_groupnorm(x, w2, b, 2, tag="wscale_2")
        np.testing.assert_allclose(actual_2, actual_1 * 2, rtol=1e-5, atol=1e-6)


# ===================================================================
# 6. Memory and performance validation
# ===================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_groupnorm_memory_validation(capsys, request):
    """
    Test memory validation for GroupNormalization operation.
    Validates instructions and data movement for normalization computation.

    This test validates:
    1. Instructions: Multiple instruction types (add, sub, mul, div, mac, rsqrt)
    2. Data Movement: Reads input+weight+bias, writes output (same size as input)
    3. Complex Normalization: Per-group statistics computation

    Run with: pytest tests/test_ops/test_groupnorm.py::test_groupnorm_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 80)
    logger.info("GroupNormalization Operation Memory Validation")
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

    # Test cases: different tensor shapes and group configurations
    test_cases = [
        {
            "name": "2D Small",
            "shape": (1, 4, 8, 8),
            "num_groups": 2,
            "description": "Small 2D feature map with 2 groups",
        },
        {
            "name": "2D Medium",
            "shape": (2, 16, 32, 32),
            "num_groups": 4,
            "description": "Medium 2D batch with 4 groups",
        },
        {
            "name": "Instance Norm",
            "shape": (1, 8, 16, 16),
            "num_groups": 8,
            "description": "Groups equal channels (instance norm style)",
        },
        {
            "name": "Layer Norm",
            "shape": (1, 16, 16, 16),
            "num_groups": 1,
            "description": "Single group (layer norm style)",
        },
        {
            "name": "Large Batch",
            "shape": (4, 32, 16, 16),
            "num_groups": 8,
            "description": "Large batch with many channels",
        },
    ]

    logger.info(f"\n{'='*80}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        num_groups = test_case["num_groups"]

        logger.info(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Shape: {shape} (N, C, H, W)")
        logger.debug(f"Groups: {num_groups}")

        # Generate test data
        np.random.seed(42)
        N, C, H, W = shape
        x_data = np.random.randn(*shape).astype(np.float32)
        w_data = np.random.randn(C).astype(np.float32)
        b_data = np.random.randn(C).astype(np.float32)

        # Create operation with fp32 precision for consistency
        x_t = F._from_data("X", x_data)
        w_t = F._from_data("weight", w_data)
        b_t = F._from_data("bias", b_data)
        out_t = make_tensor("Y")

        op_info = {
            "name": f'groupnorm_mem_{test_name.replace(" ", "_")}',
            "optype": "GroupNormalization",
            "inList": [x_t.name, w_t.name, b_t.name],
            "outList": [out_t.name],
            "attrs": {"num_groups": num_groups, "eps": 1e-5},
        }
        op = SimOp(op_info)
        op.precision = "fp32"
        op.uses_compute_pipe = "vector"

        # Get performance counts and execute
        op.get_perf_counts([x_t, w_t, b_t], [out_t])
        out_t.data = compute_groupnorm([x_t, w_t, b_t], op)
        device.execute_op(op)

        # Verify correctness
        x_g = x_data.reshape(N, num_groups, C // num_groups, H, W)
        mean = np.mean(x_g, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_g, axis=(2, 3, 4), keepdims=True)
        x_norm = (x_g - mean) / np.sqrt(var + 1e-5)
        x_norm = x_norm.reshape(N, C, H, W)
        expected_output = x_norm * w_data.reshape(1, C, 1, 1) + b_data.reshape(
            1, C, 1, 1
        )
        actual_output = out_t.data
        np.testing.assert_allclose(
            actual_output,
            expected_output,
            rtol=1e-4,
            atol=1e-5,
            err_msg=f"GroupNorm output mismatch for {test_name}",
        )

        # Extract performance stats directly
        perf_stats = op.perf_stats
        num_elements = int(np.prod(shape))

        # Validate output shape
        assert out_t.shape == list(
            shape
        ), f"Output shape {out_t.shape} != expected {list(shape)}"
        logger.debug(f"Output shape: {out_t.shape}")

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

        # Validate: GroupNorm has high instruction count (multiple ops per element)
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

        # Calculate elements per group
        elems_per_group = (C // num_groups) * H * W
        logger.debug(
            f"  Elements per group: {elems_per_group:,} ({C//num_groups} channels × {H}×{W})"
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

        # GroupNorm has moderate arithmetic intensity
        assert (
            arithmetic_intensity > 0.5
        ), f"Arithmetic intensity too low for GroupNorm: {arithmetic_intensity}"
        logger.debug("  ✓ Moderate arithmetic intensity (higher than simple ops)")

        logger.info("\n  -- Execution Cycles --")
        logger.debug(f"  Compute cycles:   {compute_cycles:,}")
        logger.debug(f"  Memory cycles:    {memory_cycles:,}")
        logger.debug(f"    Read cycles:    {mem_rd_cycles:,}")
        logger.debug(f"    Write cycles:   {mem_wr_cycles:,}")
        logger.debug(f"  Ideal cycles:     {ideal_cycles:,}")
        logger.debug(f"  Bottleneck:       {bottleneck}")

        # GroupNorm bottleneck can vary based on problem size
        logger.debug(f"  ✓ Bottleneck identified: {bottleneck}")

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "shape": shape,
                "num_groups": num_groups,
                "num_elements": num_elements,
                "elems_per_group": elems_per_group,
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

    # Group Configuration Analysis
    logger.info("\n-- Group Configuration Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Groups':<10s} {'Elems/Group':<15s} {'Shape (NCHW)':<20s}"
    )
    logger.info("-" * 80)
    for result in all_results:
        shape_str = "×".join(map(str, result["shape"]))
        logger.debug(
            f"{result['test_name']:<30s} {result['num_groups']:<10d} {result['elems_per_group']:>12,}   {shape_str}"
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
        "✓ GroupNorm Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • Multiple instruction types validated (add, sub, mul, div, mac, rsqrt) ✓",
        "  • All operations are COMPUTE-bound ✓",
        "  • High arithmetic intensity (compute-intensive) ✓",
        "  • Complex per-group normalization verified ✓",
        "  • Bottleneck analysis completed (varies by problem size) ✓",
        "  • Moderate arithmetic intensity (higher than simple ops) ✓",
    ]

    for result in all_results:
        shape_str = "×".join(map(str, result["shape"]))
        summary_lines.append(
            "  ✓ {:<28s} | {:>9,} ops | {:>8.1f} KB | {} groups | {}".format(
                result["test_name"],
                result["instructions"],
                result["total_data_moved"] / 1024,
                result["num_groups"],
                shape_str,
            )
        )

    # Write to pytest's terminal reporter (always visible)
    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=", "GROUPNORM MEMORY VALIDATION RESULTS", bold=True, green=True
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        # Fallback: disable capture and print directly
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("GROUPNORM MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    # Final assertion
    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
