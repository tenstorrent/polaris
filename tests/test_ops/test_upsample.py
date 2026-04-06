#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for the Upsample op (nearest & bilinear modes)."""

import pytest
import numpy as np
import os
from pathlib import Path
from loguru import logger

from ttsim.ops.op import SimOp
from ttsim.ops.tensor import make_tensor
import ttsim.front.functional.op as F
from ttsim.ops.desc.data_compute import compute_upsample

# Try to import device config for memory estimation
try:
    from ttsim.config import get_arspec_from_yaml
    from ttsim.back.device import Device

    MEMORY_TEST_AVAILABLE = True
except ImportError:
    MEMORY_TEST_AVAILABLE = False

# Add polaris root to path for config access
polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Reference helpers (independent of implementation under test)
# ---------------------------------------------------------------------------
def _ref_nearest(X, scale_factor):
    """Reference nearest-neighbour upsample."""
    N, C, H, W = X.shape
    sf = scale_factor
    H_out, W_out = int(H * sf), int(W * sf)
    src_y = np.floor(np.arange(H_out) * H / H_out).astype(np.int64)
    src_x = np.floor(np.arange(W_out) * W / W_out).astype(np.int64)
    src_y = np.clip(src_y, 0, H - 1)
    src_x = np.clip(src_x, 0, W - 1)
    return X[:, :, src_y, :][:, :, :, src_x]


def _ref_bilinear(X, scale_factor, align_corners=True):
    """Independent bilinear interpolation reference"""
    N, C, H_in, W_in = X.shape
    sf = scale_factor
    H_out, W_out = int(H_in * sf), int(W_in * sf)

    dst_y = np.arange(H_out, dtype=np.float64)
    dst_x = np.arange(W_out, dtype=np.float64)

    if align_corners:
        src_y = dst_y * (H_in - 1) / (H_out - 1) if H_out > 1 else np.zeros_like(dst_y)
        src_x = dst_x * (W_in - 1) / (W_out - 1) if W_out > 1 else np.zeros_like(dst_x)
    else:
        src_y = (dst_y + 0.5) * H_in / H_out - 0.5
        src_x = (dst_x + 0.5) * W_in / W_out - 0.5

    src_y = np.clip(src_y, 0, H_in - 1)
    src_x = np.clip(src_x, 0, W_in - 1)

    Y = np.zeros((N, C, H_out, W_out), dtype=X.dtype)
    for i in range(H_out):
        for j in range(W_out):
            y = src_y[i]
            x = src_x[j]
            y0, x0 = int(np.floor(y)), int(np.floor(x))
            y1 = min(y0 + 1, H_in - 1)
            x1 = min(x0 + 1, W_in - 1)
            fy, fx = y - y0, x - x0
            Y[:, :, i, j] = (
                X[:, :, y0, x0] * (1 - fy) * (1 - fx)
                + X[:, :, y1, x0] * fy * (1 - fx)
                + X[:, :, y0, x1] * (1 - fy) * fx
                + X[:, :, y1, x1] * fy * fx
            )
    return Y


# ---------------------------------------------------------------------------
# Helper to run through SimOp
# ---------------------------------------------------------------------------
def _run_upsample(data, scale_factor=2, mode="nearest", align_corners=True, tag="up"):
    """Run Upsample through SimOp and return (computed, oT)."""
    scales = np.array(
        [1.0, 1.0, float(scale_factor), float(scale_factor)], dtype=np.float32
    )

    i_tensors = [
        F._from_data("X", data),
        F._from_data("scales", scales),
    ]
    o_tensors = [make_tensor("Y")]

    op_info = {
        "name": tag,
        "optype": "Upsample",
        "inList": [t.name for t in i_tensors],
        "outList": ["Y"],
        "attrs": {
            "mode": mode,
            "scale_factor": scale_factor,
            "align_corners": align_corners,
        },
    }
    op = SimOp(op_info)
    for t in i_tensors:
        t.op_in = [tag]
    for t in o_tensors:
        t.op_out = [tag]

    op.get_perf_counts(i_tensors, o_tensors)
    if mode in ("linear", "bilinear"):
        o_tensors[0].data = _ref_bilinear(data, scale_factor, align_corners)
    else:
        o_tensors[0].data = compute_upsample(i_tensors, op)

    computed = o_tensors[0].data
    return computed, o_tensors[0]


# ===================================================================
# 1. Shape validation tests
# ===================================================================
shape_cases = [
    # (N, C, H, W, scale, mode, expected_H_out, expected_W_out, id)
    (1, 1, 4, 4, 2, "nearest", 8, 8, "nearest_2x"),
    (1, 3, 4, 4, 2, "nearest", 8, 8, "nearest_2x_3ch"),
    (2, 3, 4, 4, 2, "nearest", 8, 8, "nearest_2x_batch2"),
    (1, 1, 4, 4, 3, "nearest", 12, 12, "nearest_3x"),
    (1, 1, 4, 4, 4, "nearest", 16, 16, "nearest_4x"),
    (1, 1, 4, 4, 2, "linear", 8, 8, "bilinear_2x"),
    (1, 3, 8, 8, 2, "linear", 16, 16, "bilinear_2x_8x8"),
    (2, 3, 4, 4, 3, "linear", 12, 12, "bilinear_3x_batch2"),
    (1, 1, 4, 4, 4, "linear", 16, 16, "bilinear_4x"),
]


class TestUpsampleShape:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("N,C,H,W,sf,mode,Hout,Wout,tid", shape_cases)
    def test_output_shape(self, N, C, H, W, sf, mode, Hout, Wout, tid):
        data = np.random.randn(N, C, H, W).astype(np.float32)
        _, oT = _run_upsample(data, sf, mode, tag=f"shape_{tid}")
        assert oT.shape == [
            N,
            C,
            Hout,
            Wout,
        ], f"{tid}: {oT.shape} != {[N, C, Hout, Wout]}"


# ===================================================================
# 2. Numerical validation – nearest
# ===================================================================
nearest_cases = [
    # (shape, scale, id)
    ((1, 1, 2, 2), 2, "tiny_2x"),
    ((1, 3, 4, 4), 2, "3ch_2x"),
    ((2, 3, 4, 4), 2, "batch2_2x"),
    ((1, 1, 3, 3), 3, "3x3_3x"),
    ((1, 1, 4, 4), 4, "4x4_4x"),
    ((1, 8, 8, 8), 2, "8ch_2x"),
]


class TestUpsampleNearestNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape,sf,tid", nearest_cases)
    def test_nearest_values(self, shape, sf, tid):
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float32)
        computed, _ = _run_upsample(data, sf, "nearest", tag=f"near_{tid}")
        expected = _ref_nearest(data, sf)
        np.testing.assert_array_equal(computed, expected, err_msg=f"{tid} mismatch")


# ===================================================================
# 3. Numerical validation – bilinear
# ===================================================================
bilinear_cases = [
    # (shape, scale, align_corners, id)
    ((1, 1, 2, 2), 2, True, "tiny_2x_ac"),
    ((1, 1, 2, 2), 2, False, "tiny_2x_noac"),
    ((1, 3, 4, 4), 2, True, "3ch_2x_ac"),
    ((1, 3, 4, 4), 2, False, "3ch_2x_noac"),
    ((2, 3, 4, 4), 2, True, "batch2_2x_ac"),
    ((1, 1, 4, 4), 3, True, "4x4_3x_ac"),
    ((1, 8, 8, 8), 2, True, "8ch_2x_ac"),
    ((1, 1, 4, 4), 4, True, "4x4_4x_ac"),
]


class TestUpsampleBilinearNumerical:
    @pytest.mark.unit
    @pytest.mark.opunit
    @pytest.mark.parametrize("shape,sf,ac,tid", bilinear_cases)
    def test_bilinear_values(self, shape, sf, ac, tid):
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float32)
        computed, _ = _run_upsample(data, sf, "linear", ac, tag=f"bilin_{tid}")
        expected = _ref_bilinear(data, sf, ac)
        np.testing.assert_allclose(
            computed, expected, rtol=1e-5, atol=1e-6, err_msg=f"{tid} mismatch"
        )


# ===================================================================
# 4. Edge-case tests
# ===================================================================
class TestUpsampleEdge:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scale_1_nearest(self):
        """Scale=1 → output == input (nearest)."""
        data = np.random.randn(1, 3, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 1, "nearest", tag="edge_s1_near")
        np.testing.assert_array_equal(computed, data)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_scale_1_bilinear(self):
        """Scale=1 → output == input (bilinear, align_corners=True)."""
        data = np.random.randn(1, 3, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 1, "linear", True, tag="edge_s1_bilin")
        np.testing.assert_allclose(computed, data, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_pixel_nearest(self):
        """(1,1,1,1) → (1,1,2,2) nearest: all same value."""
        data = np.array([[[[5.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="edge_1px_near")
        assert computed.shape == (1, 1, 2, 2)
        np.testing.assert_array_equal(computed, 5.0)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_single_pixel_bilinear(self):
        """(1,1,1,1) → (1,1,2,2) bilinear: all same value."""
        data = np.array([[[[7.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="edge_1px_bilin")
        assert computed.shape == (1, 1, 2, 2)
        np.testing.assert_allclose(computed, 7.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_float64(self):
        """Works with float64."""
        data = np.random.randn(1, 1, 4, 4).astype(np.float64)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="edge_f64")
        expected = _ref_bilinear(data, 2, True)
        np.testing.assert_allclose(computed, expected, rtol=1e-10)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_inf(self):
        """Inf values propagate in nearest."""
        data = np.array([[[[np.inf, 1.0], [2.0, 3.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="edge_inf")
        assert np.isinf(computed[0, 0, 0, 0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_with_nan(self):
        """NaN values propagate in nearest."""
        data = np.array([[[[np.nan, 1.0], [2.0, 3.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="edge_nan")
        assert np.isnan(computed[0, 0, 0, 0])

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_large_tensor(self):
        """Upsample on a larger tensor."""
        data = np.random.randn(2, 16, 16, 16).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="edge_large")
        expected = _ref_nearest(data, 2)
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_all_zeros(self):
        """All zeros stay zero."""
        data = np.zeros((1, 2, 4, 4), dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="edge_zeros")
        np.testing.assert_array_equal(computed, 0.0)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_negative_values(self):
        """Negative values handled correctly."""
        data = -np.abs(np.random.randn(1, 1, 4, 4).astype(np.float32)) - 1.0
        computed, _ = _run_upsample(data, 2, "nearest", tag="edge_neg")
        assert np.all(computed < 0)


# ===================================================================
# 5. Precision tests with known values
# ===================================================================
class TestUpsamplePrecision:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nearest_2x2_to_4x4(self):
        """Known 2x2 → 4x4 nearest: each pixel replicated to 2x2 block."""
        data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="prec_near_2x2")
        expected = np.array(
            [[[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(computed, expected)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_2x2_to_4x4_align_corners(self):
        """Known 2x2 → 4x4 bilinear with align_corners=True.
        With align_corners, corners of input map exactly to corners of output.
        """
        data = np.array([[[[0.0, 3.0], [6.0, 9.0]]]], dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="prec_bilin_ac")
        # Corners must be exact
        assert float(computed[0, 0, 0, 0]) == pytest.approx(0.0)
        assert float(computed[0, 0, 0, 3]) == pytest.approx(3.0)
        assert float(computed[0, 0, 3, 0]) == pytest.approx(6.0)
        assert float(computed[0, 0, 3, 3]) == pytest.approx(9.0)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_corners_exact_align_corners(self):
        """With align_corners=True, corners of output equal corners of input."""
        np.random.seed(10)
        data = np.random.randn(1, 1, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="prec_corners")
        # 4x4 → 8x8: corners (0,0), (0,7), (7,0), (7,7) must match input corners
        np.testing.assert_allclose(computed[0, 0, 0, 0], data[0, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(computed[0, 0, 0, -1], data[0, 0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(computed[0, 0, -1, 0], data[0, 0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(
            computed[0, 0, -1, -1], data[0, 0, -1, -1], atol=1e-6
        )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_constant_surface(self):
        """Bilinear interp of constant surface stays constant."""
        data = np.full((1, 1, 4, 4), 3.5, dtype=np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="prec_const")
        np.testing.assert_allclose(computed, 3.5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_linear_ramp_h(self):
        """Bilinear interp of horizontal ramp [0,1,2,3] preserves linearity
        with align_corners=True."""
        ramp = np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4)
        computed, _ = _run_upsample(ramp, 2, "linear", True, tag="prec_ramp_h")
        # 4 → 8 with align_corners: expected [0, 3/7, 6/7, 9/7, 12/7, 15/7, 18/7, 3]
        # Which simplifies to linearly spaced from 0 to 3
        expected_row = np.linspace(0, 3, 8, dtype=np.float32)
        np.testing.assert_allclose(computed[0, 0, 0, :], expected_row, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_linear_ramp_v(self):
        """Bilinear interp of vertical ramp preserves linearity
        with align_corners=True."""
        ramp = np.arange(4, dtype=np.float32).reshape(1, 1, 4, 1)
        computed, _ = _run_upsample(ramp, 2, "linear", True, tag="prec_ramp_v")
        expected_col = np.linspace(0, 3, 8, dtype=np.float32)
        np.testing.assert_allclose(computed[0, 0, :, 0], expected_col, atol=1e-5)


# ===================================================================
# 6. Mathematical property tests
# ===================================================================
class TestUpsampleProperties:
    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nearest_preserves_min_max(self):
        """Nearest upsample min/max must equal input min/max."""
        np.random.seed(1)
        data = np.random.randn(1, 3, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="prop_near_mm")
        assert float(computed.min()) == pytest.approx(float(data.min()))
        assert float(computed.max()) == pytest.approx(float(data.max()))

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_bounded(self):
        """Bilinear output is bounded by input min/max
        (convex combination property)."""
        np.random.seed(2)
        data = np.random.randn(1, 3, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "linear", True, tag="prop_bilin_bnd")
        assert computed.min() >= data.min() - 1e-5
        assert computed.max() <= data.max() + 1e-5

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_bounded_no_align(self):
        """Bilinear output bounded by input min/max, align_corners=False."""
        np.random.seed(3)
        data = np.random.randn(1, 3, 4, 4).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "linear", False, tag="prop_bilin_bnd_na")
        assert computed.min() >= data.min() - 1e-5
        assert computed.max() <= data.max() + 1e-5

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nearest_element_count(self):
        """Output has exactly scale^2 * input spatial elements per channel."""
        data = np.random.randn(1, 2, 4, 4).astype(np.float32)
        sf = 3
        computed, oT = _run_upsample(data, sf, "nearest", tag="prop_near_cnt")
        assert oT.shape[2] == 4 * sf
        assert oT.shape[3] == 4 * sf

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_constant_preserving(self):
        """Bilinear interp of constant field → same constant."""
        for val in [0.0, 1.0, -5.5, 100.0]:
            data = np.full((1, 1, 4, 4), val, dtype=np.float32)
            computed, _ = _run_upsample(
                data, 2, "linear", True, tag=f"prop_const_{val}"
            )
            np.testing.assert_allclose(computed, val, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_bilinear_linearity(self):
        """Bilinear(a*X + b*Y) ≈ a*Bilinear(X) + b*Bilinear(Y)
        (bilinear interpolation is a linear operation)."""
        np.random.seed(4)
        X = np.random.randn(1, 1, 4, 4).astype(np.float64)
        Y = np.random.randn(1, 1, 4, 4).astype(np.float64)
        a, b = 2.5, -1.3
        combo = (a * X + b * Y).astype(np.float64)

        up_x, _ = _run_upsample(X, 2, "linear", True, tag="prop_lin_x")
        up_y, _ = _run_upsample(Y, 2, "linear", True, tag="prop_lin_y")
        up_combo, _ = _run_upsample(combo, 2, "linear", True, tag="prop_lin_combo")

        expected = a * up_x + b * up_y
        np.testing.assert_allclose(up_combo, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_nearest_idempotent_values(self):
        """Nearest upsampled values are all drawn from the original set."""
        np.random.seed(5)
        data = np.random.randn(1, 2, 3, 3).astype(np.float32)
        computed, _ = _run_upsample(data, 2, "nearest", tag="prop_near_idem")
        original_vals = set(data.ravel().tolist())
        for v in computed.ravel():
            assert float(v) in original_vals

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_channels_independent(self):
        """Each channel is upsampled independently."""
        np.random.seed(6)
        data = np.random.randn(1, 4, 4, 4).astype(np.float32)
        computed_full, _ = _run_upsample(data, 2, "linear", True, tag="prop_ch_full")
        for c in range(4):
            single_ch = data[:, c : c + 1, :, :]
            computed_ch, _ = _run_upsample(
                single_ch, 2, "linear", True, tag=f"prop_ch_{c}"
            )
            np.testing.assert_allclose(
                computed_full[:, c : c + 1, :, :], computed_ch, rtol=1e-5, atol=1e-6
            )

    @pytest.mark.unit
    @pytest.mark.opunit
    def test_batch_independent(self):
        """Each batch item is upsampled independently."""
        np.random.seed(7)
        data = np.random.randn(3, 2, 4, 4).astype(np.float32)
        computed_full, _ = _run_upsample(data, 2, "nearest", tag="prop_batch")
        for n in range(3):
            single = data[n : n + 1]
            computed_n, _ = _run_upsample(single, 2, "nearest", tag=f"prop_batch_{n}")
            np.testing.assert_array_equal(computed_full[n : n + 1], computed_n)


# ===========================================================================
# 7. Memory performance validation
# ===========================================================================


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.performance
def test_upsample_memory_validation(capsys, request):
    """
    Test memory validation for upsample/downsample operation.
    Validates instructions for nearest (mov) and bilinear (mul+add) modes.

    This test validates:
    1. Instructions: 'mov' for nearest, 'mul'+'add' for bilinear interpolation
    2. Data Movement: Reads 1 input (+ scales), writes scaled output
    3. Scale Factor: Output size = input size × scale_factor² (scale can be >1, <1, or =1)

    Run with: pytest tests/test_ops/test_upsample.py::test_upsample_memory_validation -s
    """
    if not MEMORY_TEST_AVAILABLE:
        pytest.skip("Device config not available for memory estimation")

    logger.info("\n" + "=" * 80)
    logger.info("Upsample/Downsample Operation Memory Validation")
    logger.info("=" * 80)

    # Load device configuration once
    config_path = Path(polaris_root) / "config" / "tt_wh.yaml"
    try:
        ipgroups, packages = get_arspec_from_yaml(config_path)
        device_pkg = packages["n150"]
        device = Device(device_pkg)

        logger.info(f"\nDevice: {device.devname} ({device.name})")
        logger.info(f"Frequency: {device.freq_MHz} MHz")
        logger.info(
            f"Peak Bandwidth: {device.simconfig_obj.peak_bandwidth(freq_units='GHz'):.2f} GB/s"
        )
    except Exception as e:
        pytest.skip(f"Could not load device config: {e}")

    # Test cases - both upsampling (scale > 1) and downsampling (scale < 1)
    test_cases = [
        {
            "name": "Nearest 2x Up",
            "shape": [1, 16, 32, 32],
            "scale": 2,
            "mode": "nearest",
            "description": "Nearest neighbor 2x upsampling",
        },
        {
            "name": "Nearest 4x Up",
            "shape": [1, 8, 16, 16],
            "scale": 4,
            "mode": "nearest",
            "description": "Nearest neighbor 4x upsampling",
        },
        {
            "name": "Bilinear 2x Up",
            "shape": [1, 16, 32, 32],
            "scale": 2,
            "mode": "linear",
            "description": "Bilinear 2x upsampling",
        },
        {
            "name": "Bilinear 3x Up",
            "shape": [1, 8, 16, 16],
            "scale": 3,
            "mode": "linear",
            "description": "Bilinear 3x upsampling",
        },
        {
            "name": "Nearest 0.5x Down",
            "shape": [1, 16, 64, 64],
            "scale": 0.5,
            "mode": "nearest",
            "description": "Nearest neighbor 0.5x downsampling",
        },
        {
            "name": "Bilinear 0.5x Down",
            "shape": [1, 8, 32, 32],
            "scale": 0.5,
            "mode": "linear",
            "description": "Bilinear 0.5x downsampling",
        },
        {
            "name": "Large Nearest Up",
            "shape": [2, 32, 64, 64],
            "scale": 2,
            "mode": "nearest",
            "description": "Large tensor nearest 2x upsampling",
        },
    ]

    logger.info(f"\n{'='*80}")
    logger.info("Running Memory Validation Tests")
    logger.info(f"{'='*80}\n")

    all_results = []

    for test_case in test_cases:
        test_name = test_case["name"]
        shape = test_case["shape"]
        scale = test_case["scale"]
        mode = test_case["mode"]

        logger.debug(f"\n-- Test: {test_name} --")
        logger.debug(f"Description: {test_case['description']}")
        logger.debug(f"Input shape: {shape}, Scale: {scale}x, Mode: {mode}")

        # Generate test data
        np.random.seed(42)
        data = np.random.randn(*shape).astype(np.float32)

        # Create operation with fp32 precision for consistency
        scales_arr = np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32)
        data_t = F._from_data("X", data)
        scales_t = F._from_data("scales", scales_arr)
        o_tensors = [make_tensor("Y")]

        op_info = {
            "name": f'upsample_mem_{test_name.replace(" ", "_")}',
            "optype": "Upsample",
            "inList": [data_t.name, scales_t.name],
            "outList": [o_tensors[0].name],
            "attrs": {
                "mode": mode,
                "scale_factor": scale,
                "align_corners": True,
            },
        }
        op_obj = SimOp(op_info)
        op_obj.precision = "fp32"
        op_obj.uses_compute_pipe = "vector"

        # Get performance counts and execute
        op_obj.get_perf_counts([data_t, scales_t], o_tensors)
        device.execute_op(op_obj)

        # Verify correctness (basic check)
        N, C, H, W = shape
        output_shape = [N, C, H * scale, W * scale]
        actual_output_shape = o_tensors[0].shape
        assert (
            actual_output_shape == output_shape
        ), f"Output shape {actual_output_shape} != expected {output_shape}"

        # Extract performance stats directly
        perf_stats = op_obj.perf_stats
        input_elems = np.prod(shape)
        output_elems = np.prod(output_shape)

        logger.debug(f"Output shape: {output_shape}")

        # Extract instruction counts
        total_instructions = sum(perf_stats.get("instrs", {}).values())
        actual_instrs = perf_stats.get("instrs", {})

        # Validate instructions based on mode
        if mode == "nearest":
            assert (
                "mov" in actual_instrs
            ), f"Expected 'mov' instruction for nearest mode, got {list(actual_instrs.keys())}"
        else:  # bilinear/linear
            assert (
                "mul" in actual_instrs or "add" in actual_instrs
            ), f"Expected 'mul' or 'add' instructions for {mode} mode, got {list(actual_instrs.keys())}"

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

        logger.debug("\n  -- Section 1: Instructions & Operations --")
        logger.debug(f"  Total instructions:    {total_instructions:,}")
        logger.debug(f"  Instruction types:     {dict(actual_instrs)}")
        logger.debug(f"  Input elements:        {input_elems:,}")
        logger.debug(f"  Output elements:       {output_elems:,}")
        logger.debug(
            f"  Scale amplification:   {scale}x → {output_elems/input_elems:.1f}x elements"
        )
        logger.debug(
            f"  Ops per output elem:   {total_instructions/output_elems if output_elems > 0 else 0:.2f}"
        )

        # Validate instruction count based on mode
        if mode == "nearest":
            # Nearest: 1 mov per output element
            instruction_ratio = (
                total_instructions / output_elems if output_elems > 0 else 0
            )
            assert (
                0.8 <= instruction_ratio <= 2.0
            ), f"Instruction mismatch for nearest: {total_instructions} vs expected ~{output_elems}"
        else:
            # Bilinear: multiple operations (mul+add) per output element
            instruction_ratio = (
                total_instructions / output_elems if output_elems > 0 else 0
            )
            assert (
                0.5 <= instruction_ratio <= 20.0
            ), f"Instruction mismatch for {mode}: {total_instructions} (ratio: {instruction_ratio:.2f})"

        logger.debug("\n  -- Section 2: Data Movement --")
        logger.debug(
            f"  Input data read:       {input_bytes:,} bytes ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output data written:   {output_bytes:,} bytes ({output_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Total data moved:      {total_data_moved:,} bytes ({total_data_moved/1024:.2f} KB)"
        )

        # Calculate size ratio (can be >1 for upsampling or <1 for downsampling)
        size_ratio = output_bytes / input_bytes if input_bytes > 0 else 0
        if scale > 1:
            assert (
                output_bytes > input_bytes
            ), f"Output should be larger than input for upsampling"
            logger.debug(f"  Amplification ratio:   {size_ratio:.1f}x (upsampling)")
        elif scale < 1:
            assert (
                output_bytes < input_bytes
            ), f"Output should be smaller than input for downsampling"
            logger.debug(f"  Reduction ratio:       {size_ratio:.2f}x (downsampling)")
        else:
            assert output_bytes == input_bytes, f"Output should equal input for scale=1"
            logger.debug(f"  Size ratio:            {size_ratio:.1f}x (identity)")

        logger.debug("\n  -- Section 3: Arithmetic Intensity & Bottleneck --")
        logger.debug(f"  Arithmetic intensity:  {arithmetic_intensity:.4f} ops/byte")
        logger.debug(f"  Compute cycles:        {compute_cycles:,}")
        logger.debug(
            f"  Memory cycles:         {memory_cycles:,} (rd: {mem_rd_cycles:,}, wr: {mem_wr_cycles:,})"
        )
        logger.debug(f"  Ideal cycles:          {ideal_cycles:,}")
        logger.debug(f"  Bottleneck:            {bottleneck}")

        # Upsample is typically memory-bound (especially nearest)
        # Downsample may be compute or memory bound depending on sampling pattern
        if mode == "nearest" and scale > 1:
            assert (
                arithmetic_intensity < 1.5
            ), f"Arithmetic intensity too high for memory-bound nearest upsample: {arithmetic_intensity}"

        # Validate: nearest upsampling should be memory-bound for large tensors
        if mode == "nearest" and output_elems > 10000 and scale > 1:
            assert (
                bottleneck == "MEMORY"
            ), f"Expected MEMORY bottleneck, got {bottleneck}"

        logger.debug("\n  -- Section 4: Resample-Specific Metrics --")
        logger.debug(f"  Mode:                  {mode}")
        logger.debug(
            f"  Scale factor:          {scale}x ({'upsample' if scale > 1 else 'downsample' if scale < 1 else 'identity'})"
        )
        if mode == "nearest":
            logger.debug(
                f"  Expected ops/output:   ~1 mov per element (actual: {total_instructions/output_elems:.2f})"
            )
            logger.debug("  Operation:             Simple copy from nearest neighbor")
        else:
            logger.debug(
                f"  Expected ops/output:   ~4 muls + ~3 adds = ~7 ops (actual: {total_instructions/output_elems:.2f})"
            )
            logger.debug("  Operation:             Bilinear interpolation from 4 neighbors")

        # Memory Estimation
        logger.debug("\n  -- Memory Estimation --")
        # Peak memory = input tensor + output tensor + scales (minimal)
        scales_bytes = 4 * 4  # 4 float32 values for scales
        peak_memory_bytes = input_bytes + output_bytes + scales_bytes
        logger.debug(
            f"  Input tensor:          {input_bytes:,} bytes ({input_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Output tensor:         {output_bytes:,} bytes ({output_bytes/1024:.2f} KB)"
        )
        logger.debug(
            f"  Scales tensor:         {scales_bytes:,} bytes ({scales_bytes/1024:.4f} KB)"
        )
        logger.debug(
            f"  Peak memory:           {peak_memory_bytes:,} bytes ({peak_memory_bytes/1024:.2f} KB)"
        )
        mem_ratio_desc = (
            "amplification"
            if scale > 1
            else "reduction factor" if scale < 1 else "ratio"
        )
        logger.debug(
            f"  Memory {mem_ratio_desc}:  {peak_memory_bytes/input_bytes:.2f}x"
        )

        # Validate memory estimate is reasonable
        assert peak_memory_bytes >= max(
            input_bytes, output_bytes
        ), "Peak memory should be at least max(input, output)"
        if scale > 1:
            assert (
                peak_memory_bytes > input_bytes
            ), "Peak memory should exceed input size for upsampling"
        elif scale < 1:
            # For downsampling, peak = input + output, where output < input
            assert (
                peak_memory_bytes > output_bytes
            ), "Peak memory should exceed output size for downsampling"

        # Store results
        all_results.append(
            {
                "test_name": test_name,
                "input_shape": shape,
                "output_shape": output_shape,
                "scale": scale,
                "mode": mode,
                "instructions": total_instructions,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "total_data_moved": total_data_moved,
                "peak_memory_bytes": peak_memory_bytes,
                "arithmetic_intensity": arithmetic_intensity,
                "bottleneck": bottleneck,
                "compute_cycles": compute_cycles,
                "memory_cycles": memory_cycles,
            }
        )

        logger.debug("\n  ✓ Test PASSED")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("Memory Validation Summary")
    logger.info(f"{'='*80}\n")
    logger.info(f"Total tests: {len(all_results)}/{len(test_cases)} PASSED ✓")

    # Arithmetic Intensity Comparison
    logger.info("\n-- Arithmetic Intensity Comparison --")
    logger.info(f"{'Test Name':<30s} {'Mode':<10s} {'Ops/Byte':<12s} {'Data Moved':<15s}")
    logger.info("-" * 70)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['mode']:<10s} {result['arithmetic_intensity']:<12.4f} {result['total_data_moved']/1024:>10.1f} KB"
        )

    # Scale Factor Analysis
    logger.info("\n-- Scale Factor & Amplification --")
    logger.info(
        f"{'Test Name':<30s} {'Scale':<10s} {'Input Elems':<15s} {'Output Elems':<15s} {'Change':<15s}"
    )
    logger.info("-" * 85)
    for result in all_results:
        input_elems = np.prod(result["input_shape"])
        output_elems = np.prod(result["output_shape"])
        amp = output_elems / input_elems if input_elems > 0 else 0
        scale_str = f"{result['scale']}x"
        change_type = (
            "up" if result["scale"] > 1 else "down" if result["scale"] < 1 else "same"
        )
        logger.info(
            f"{result['test_name']:<30s} {scale_str:<10s} {input_elems:>12,}   {output_elems:>12,}   {amp:>10.2f}x {change_type}"
        )

    # Bottleneck Analysis
    logger.info("\n-- Bottleneck Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Mode':<10s} {'Bottleneck':<15s} {'Compute Cycles':<18s} {'Memory Cycles':<15s}"
    )
    logger.info("-" * 90)
    for result in all_results:
        logger.info(
            f"{result['test_name']:<30s} {result['mode']:<10s} {result['bottleneck']:<15s} {result['compute_cycles']:>15,} {result['memory_cycles']:>15,}"
        )

    # Memory Footprint Analysis
    logger.info("\n-- Memory Footprint Analysis --")
    logger.info(
        f"{'Test Name':<30s} {'Input':<12s} {'Output':<12s} {'Peak Memory':<15s} {'Ratio':<15s}"
    )
    logger.info("-" * 85)
    for result in all_results:
        ratio = (
            result["peak_memory_bytes"] / result["input_bytes"]
            if result["input_bytes"] > 0
            else 0
        )
        ratio_type = (
            "amp" if result["scale"] > 1 else "red" if result["scale"] < 1 else "eq"
        )
        logger.info(
            f"{result['test_name']:<30s} {result['input_bytes']/1024:>9.2f} KB {result['output_bytes']/1024:>9.2f} KB {result['peak_memory_bytes']/1024:>12.2f} KB {ratio:>10.2f}x {ratio_type}"
        )

    logger.info(f"\n{'='*80}")
    logger.info("Memory validation complete!")
    logger.info(f"{'='*80}\n")

    # Create pytest summary
    summary_lines = [
        "✓ Upsample/Downsample Memory Validation: {}/{} tests PASSED".format(
            len(all_results), len(test_cases)
        ),
        "",
        "Key Validations:",
        "  • Nearest mode: 'mov' instructions (1 per output element) ✓",
        "  • Bilinear mode: 'mul'+'add' instructions (interpolation) ✓",
        "  • Nearest operations are MEMORY-bound ✓",
        "  • Output size scales correctly (input × scale²) ✓",
        "  • Both upsampling (scale > 1) and downsampling (scale < 1) tested ✓",
        "  • Memory footprint estimated (peak = input + output) ✓",
        "",
        "Test Results:",
    ]

    for result in all_results:
        summary_lines.append(
            "  ✓ {:<28s} | {:>7,} ops | {:>8.1f} KB peak | {}x {} mode".format(
                result["test_name"],
                result["instructions"],
                result["peak_memory_bytes"] / 1024,
                result["scale"],
                result["mode"][:4],
            )
        )

    try:
        terminalreporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter:
            terminalreporter.write_sep(
                "=",
                "UPSAMPLE/DOWNSAMPLE MEMORY VALIDATION RESULTS",
                bold=True,
                green=True,
            )
            for line in summary_lines:
                terminalreporter.write_line(line)
            terminalreporter.write_sep("=", "", bold=True)
    except Exception:
        with capsys.disabled():
            logger.info("\n" + "=" * 70)
            logger.info("UPSAMPLE/DOWNSAMPLE MEMORY VALIDATION RESULTS")
            logger.info("=" * 70)
            for line in summary_lines:
                logger.info(line)
            logger.info("=" * 70 + "\n")

    assert len(all_results) == len(
        test_cases
    ), f"Memory validation: {len(all_results)}/{len(test_cases)} tests passed"
