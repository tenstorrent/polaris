# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TopK roofline (TEN-4859) driven through the ttsim device cost path: the topk_large_indices law,
the generic factories, the gate regime, the Blackhole router, the DSA indexer shim and the
calibration set of the 2026-09-12 campaign (main tip 2dbd14bf632)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from ttsim.back.device import Device
from ttsim.perf import roofline_topk as rt
from ttsim.perf.roofline_topk import (TopkConfig, TopkCall, GenericConfig, GateConfig, IndexerConfig,
                                       LargeIndicesTerms, GenericTerms, GateTerms,
                                       predict_topk, predict_topk_call, predict_generic, predict_gate,
                                       predict_indexer, topk_perf_stats, snap_k, body_mode, route_topk,
                                       should_route_to_topk_large_indices, multi_core_split,
                                       column_split_config, insertion_steps, calibration_for)

GHZ = 1.35
MAIN = "2dbd14bf632"

# Measured device kernel durations (ns, tracy median) on the p100a, tt-metal main tip 2dbd14bf632,
# 2026-09-12 campaign (handoff/revamp/data/topk/results_*.csv), keyed by (N, K, rows).
LI_MAIN_NS = {
    (16384, 64, 1): 51_721, (65536, 512, 1): 278_461, (65536, 1024, 1): 237_006, (131072, 16, 1): 555_987,
    (65536, 2048, 1): 276_297, (16896, 512, 1): 72_516, (32768, 1024, 1): 119_304, (33792, 1024, 1): 124_867,
    (67584, 2048, 1): 288_873, (262144, 2048, 1): 1_099_784, (256, 16, 1): 3_215, (1024, 1024, 1): 7_125,
    (65536, 2048, 110): 279_940, (65536, 2048, 111): 550_908, (4096, 2048, 110): 25_153,
    (51200, 1536, 640): 1_284_662.5,        # nightly pin 1,286,400 ns
    (8192, 1024, 2048): 566_976, (8192, 2048, 2048): 670_090, (131072, 2048, 2048): 10_402_204,
    (440, 16, 640): 14_339, (8400, 304, 1): 28_190,
}
# indexer_score_dsa on the calibration checkout (data/topk/indexer_points.csv), 88 cores, Hi 64 D 128.
INDEXER_NS = {(2048, 8192): 11_347_128, (2048, 32768): 44_652_424.5, (512, 8192): 2_839_894.5}


def _ns(r):
    return r.device_cycles / GHZ


def _within(pred, meas, pct):
    return abs(pred / meas - 1) <= pct / 100.0


class _MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class _MockSimConfig:
    def __init__(self, freq_mhz=1350, devname="Blackhole", name=None):
        self._freq_mhz = freq_mhz
        self.devname = devname
        self.name = name or ("p100a" if devname == "Blackhole" else "test_device")
        self.ipgroups = [_MockIPGroup("compute"), _MockIPGroup("memory")]

    def frequency(self, pipe, units="MHz"):
        return self._freq_mhz

    def mem_frequency(self, units="MHz"):
        return self._freq_mhz

    def mem_size(self, units="GB"):
        return 32.0

    def peak_bandwidth(self, freq_units="GHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 10.0

    def peak_flops(self, pipe, instr, precision, mul_factor=1):
        return 100.0

    def peak_ipc(self, pipe, instr, precision):
        return 128.0


def _topk_op(name, perf_stats):
    return SimpleNamespace(
        name=name, optype="TopK", uses_compute_pipe="vector", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats=perf_stats,
    )


def _sinf_perf_stats(shape, k, axis=-1, valid_length=None, **attrs):
    """Run the real topk_sinf on functional SimTensors and return the op's perf_stats."""
    import ttsim.front.functional.op as F
    attrs = dict(k=k, axis=axis, largest=True, sorted=True, **attrs)
    if valid_length is not None:
        attrs["valid_length"] = valid_length
    h = F.topk(f"tk_{'x'.join(map(str, shape))}_{k}", **attrs)
    h(F._from_shape("tk_in", list(shape), np_dtype=np.float32))
    return h.sim_op.perf_stats


def _li_sinf(shape, k, **kw):
    """The DSA / MSA direct call: topk_large_indices tagged by the shim."""
    return _sinf_perf_stats(shape, k, topk_kernel="large_indices", **kw)


def _components_sum(ps):
    return int(round(sum(ps["topk_wall_components"].values())))


class _TtnnDevice:
    """Open a ttnn shim device as the default for the test body."""
    def __enter__(self):
        from ttsim.front.ttnn.device import open_device, set_default_device, get_default_device
        try:
            self.prev = get_default_device()
        except AssertionError:
            self.prev = None
        self.dev = open_device()
        set_default_device(self.dev)
        return self.dev

    def __exit__(self, *exc):
        from ttsim.front.ttnn.device import close_device, set_default_device
        close_device(self.dev)
        set_default_device(self.prev)


def _tt(dev, shape, dtype=None, layout=None, memory_config=None):
    from ttsim.front.ttnn.tensor import Tensor, DataType, Layout
    t = Tensor(shape=shape, dtype=dtype or DataType.BFLOAT16,
               layout=layout or Layout.TILE_LAYOUT, device=dev)
    if memory_config is not None:
        t._memory_config = memory_config
    return t


# ==============================================================================================
# Calibration sets and the kernel_rev knob
# ==============================================================================================

@pytest.mark.unit
def test_kernel_rev_defaults_to_main_tip_and_is_flagged():
    r = predict_topk(TopkConfig(N=16384, K=64))
    assert r.kernel_rev == MAIN and r.low_confidence
    assert r.low_confidence_reasons == [f"kernel_rev_defaulted_{MAIN}"]
    named = predict_topk(TopkConfig(N=16384, K=64, kernel_rev=MAIN))
    assert named.device_cycles == r.device_cycles and not named.low_confidence
    for alias in ("main", "main-tip", "2dbd14b"):
        assert calibration_for(alias)[0].kernel_rev == MAIN and calibration_for(alias)[1] is False
    assert calibration_for(None) == (rt.CALIBRATIONS[MAIN], True)
    for unknown in ("abcdef0", "73e0e25"):
        with pytest.raises(ValueError):
            calibration_for(unknown)
    # Every regime carries the revision into perf_stats and flags the default the same way.
    for res in (predict_generic(GenericConfig(rows=32, N=4096, k=32)),
                predict_gate(GateConfig("generalized_moe_gate", tokens=32)),
                predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88)),
                predict_topk_call(TopkCall(rows=32, N=128256, k=64))):
        ps = res.to_polaris_op_perf_stats()
        assert ps["topk_kernel_rev"] == MAIN and f"kernel_rev_defaulted_{MAIN}" in ps["topk_low_confidence_reasons"]


# ==============================================================================================
# Regime (a): topk_large_indices on main tip
# ==============================================================================================

@pytest.mark.unit
@pytest.mark.parametrize("N,K,mode,segments", [
    (16384, 64, "FusedEndToEnd", 1), (65536, 512, "Classic", 1), (131072, 16, "Classic", 1),
    (16896, 512, "Classic", 1), (65536, 1024, "FusedSegmented", 2), (32768, 1024, "FusedSegmented", 1),
    (33792, 1024, "FusedSegmented", 2), (65536, 2048, "FusedSegmented", 1), (67584, 2048, "FusedSegmented", 2),
    (262144, 2048, "FusedSegmented", 4),
])
def test_main_tip_row_law_fit_points(N, K, mode, segments):
    r = predict_topk(TopkConfig(N=N, K=K, kernel_rev=MAIN))
    assert (r.body_mode, r.segments, r.row_blocks) == (mode, segments, 1)
    assert _within(_ns(r), LI_MAIN_NS[(N, K, 1)], 0.5), f"pred {_ns(r):.0f} meas {LI_MAIN_NS[(N, K, 1)]}"
    assert not r.low_confidence
    if segments > 1:
        assert r.components["segment_merge"] == pytest.approx((segments - 1) * rt.LARGE_INDICES_TERMS_MAIN.c_seg[r.k_snap])


@pytest.mark.unit
def test_fused_bodies_are_cheaper_per_chunk_than_classic():
    t = rt.LARGE_INDICES_TERMS_MAIN
    e2e, classic = t.a_tot[(512, "FusedEndToEnd")], t.a_tot[(512, "Classic")]
    assert e2e < classic and t.a_tot[(1024, "FusedSegmented")] / 1024 < t.a_tot[(2048, "FusedSegmented")] / 2048
    # The mode step at K'=512: 32 chunks end to end, 33 chunks Classic (L2-057 measured 72516 ns).
    e = predict_topk(TopkConfig(N=16384, K=512, kernel_rev=MAIN))
    c = predict_topk(TopkConfig(N=16896, K=512, kernel_rev=MAIN))
    assert e.components["chunk_sort"] == pytest.approx(32 * e2e) and c.components["chunk_sort"] == pytest.approx(33 * classic)
    # Segmented K'=1024 beats Classic K'=512 at 65536 (measured 237 vs 278 us), so the bucket step
    # is not monotone in K any more; k in a bucket still prices alike.
    assert predict_topk(TopkConfig(N=65536, K=1024, kernel_rev=MAIN)).device_cycles < \
        predict_topk(TopkConfig(N=65536, K=512, kernel_rev=MAIN)).device_cycles
    assert predict_topk(TopkConfig(N=65536, K=513, kernel_rev=MAIN)).device_cycles == \
        predict_topk(TopkConfig(N=65536, K=1024, kernel_rev=MAIN)).device_cycles
    assert predict_topk(TopkConfig(N=65536, K=1025, kernel_rev=MAIN)).device_cycles == \
        predict_topk(TopkConfig(N=65536, K=2048, kernel_rev=MAIN)).device_cycles
    assert (snap_k(513), snap_k(1025)) == (1024, 2048)


@pytest.mark.unit
def test_k_is_flat_within_bucket_and_single_chunk_rows_carry_the_jitter_flag():
    base = predict_topk(TopkConfig(N=16384, K=512, kernel_rev=MAIN))
    for k in (16, 32, 64, 256):
        r = predict_topk(TopkConfig(N=16384, K=k, kernel_rev=MAIN))
        assert r.device_cycles == base.device_cycles and not r.low_confidence
    # Measured K=16..512 spread is 2 percent at N=256 against 0.02 to 0.5 percent at N >= 1024: a
    # single-chunk row is a 3 us kernel inside its own jitter, so the model says so.
    one = predict_topk(TopkConfig(N=256, K=16, kernel_rev=MAIN))
    assert one.chunks == 1 and "single_chunk_row_jitter_3pct" in one.low_confidence_reasons
    assert _within(_ns(one), LI_MAIN_NS[(256, 16, 1)], 3.0)
    assert one.components["op_fixed"] == pytest.approx(1968.5 + 192.5)
    two = predict_topk(TopkConfig(N=1024, K=16, kernel_rev=MAIN))
    assert two.chunks == 2 and not two.low_confidence
    # K'=1024 single chunk pinned by (1024, 1024); K'=2048 has no single-chunk cell and says so.
    assert _within(_ns(predict_topk(TopkConfig(N=1024, K=1024, kernel_rev=MAIN))), LI_MAIN_NS[(1024, 1024, 1)], 1.0)
    assert "single_chunk_row_unmeasured_at_this_k_bucket" in predict_topk(TopkConfig(N=2048, K=2048, kernel_rev=MAIN)).low_confidence_reasons


@pytest.mark.unit
def test_intercept_splits_into_per_op_and_per_row_parts():
    # Multi-wave cells (finding F1): every wave pays c0_row, the op pays c0_op once; the MSA block select
    # (640 x 440, k 16, 6 waves of a single-chunk row) is the cell that separates the two parts.
    msa = predict_topk(TopkConfig(N=440, K=16, rows=640, kernel_rev=MAIN))
    assert (msa.k_snap, msa.body_mode, msa.chunks, msa.row_blocks) == (512, "FusedEndToEnd", 1, 6)
    assert msa.components["row_fixed"] == 0.0                     # c0_row(512) clamped at zero
    assert msa.components["op_fixed"] == pytest.approx(1968.5 + 192.5)
    assert _within(_ns(msa), LI_MAIN_NS[(440, 16, 640)], 3.0)
    for key, tol in (((8192, 1024, 2048), 1.0), ((8192, 2048, 2048), 1.0), ((131072, 2048, 2048), 0.5)):
        r = predict_topk(TopkConfig(N=key[0], K=key[1], rows=key[2], kernel_rev=MAIN))
        assert r.row_blocks == 19
        assert r.components["row_fixed"] == pytest.approx(19 * rt.LARGE_INDICES_TERMS_MAIN.c0_row[r.k_snap])
        assert r.components["op_fixed"] == pytest.approx(rt.LARGE_INDICES_TERMS_MAIN.c0_op[r.k_snap])
        assert _within(_ns(r), LI_MAIN_NS[key], tol), key
    # The R=1 intercept is the sum of both parts, as fitted.
    t = rt.LARGE_INDICES_TERMS_MAIN
    assert t.c0_row[2048] + t.c0_op[2048] == pytest.approx(6300.2) and t.c0_row[1024] + t.c0_op[1024] == pytest.approx(4012.5)


@pytest.mark.unit
def test_start_up_grows_with_the_active_cores_and_shrinks_with_an_l1_input():
    t = rt.LARGE_INDICES_TERMS_MAIN
    assert t.c_launch(1) == 0.0 and t.c_launch(32) == pytest.approx(859.05) and t.c_launch(110) == pytest.approx(4862.6)
    assert 0.0 < t.c_launch(8) < t.c_launch(32) < t.c_launch(64) < t.c_launch(110) == t.c_launch(200)
    one = predict_topk(TopkConfig(N=65536, K=512, rows=1, kernel_rev=MAIN))
    full = predict_topk(TopkConfig(N=65536, K=512, rows=110, kernel_rev=MAIN))
    over = predict_topk(TopkConfig(N=65536, K=512, rows=111, kernel_rev=MAIN))
    assert full.components["launch"] == pytest.approx(4862.6) and one.components["launch"] == 0.0
    assert full.device_cycles - one.device_cycles == pytest.approx(4862.6, abs=1)
    assert over.device_cycles - full.device_cycles == pytest.approx(full.t_row_cycles, abs=1)   # the rows cliff
    # L3: R=C and R=C+1 at (65536, 2048), ratio 1.968 measured; the small-N R=C launch cell.
    c = predict_topk(TopkConfig(N=65536, K=2048, rows=110, kernel_rev=MAIN))
    c1 = predict_topk(TopkConfig(N=65536, K=2048, rows=111, kernel_rev=MAIN))
    assert _within(_ns(c), LI_MAIN_NS[(65536, 2048, 110)], 0.5) and _within(_ns(c1), LI_MAIN_NS[(65536, 2048, 111)], 0.5)
    assert 1.9 <= c1.device_cycles / c.device_cycles <= 2.1
    assert _within(_ns(predict_topk(TopkConfig(N=4096, K=2048, rows=110, kernel_rev=MAIN))), LI_MAIN_NS[(4096, 2048, 110)], 0.5)
    # L5: the same shapes with the input in L1 lose most of the start-up (DRAM first-chunk fill).
    l1 = predict_topk(TopkConfig(N=4096, K=2048, rows=110, input_memory="L1", kernel_rev=MAIN))
    assert l1.components["launch"] == pytest.approx(1788.0) and _within(_ns(l1), 22_910, 0.5)
    l1s = predict_topk(TopkConfig(N=4096, K=512, rows=110, input_memory="L1", kernel_rev=MAIN))
    assert l1s.components["launch"] == pytest.approx(281.0) and _within(_ns(l1s), 14_225, 0.5)
    assert "l1_input_start_up_interpolated_in_k_bucket" in \
        predict_topk(TopkConfig(N=4096, K=1024, rows=110, input_memory="L1", kernel_rev=MAIN)).low_confidence_reasons


@pytest.mark.unit
def test_production_anchors_and_valid_length():
    # Nightly pin (640, 51200, 1536): 1,284,662 ns measured against the test's 1,286,400.
    pin = predict_topk(TopkConfig(N=51200, K=1536, rows=640, kernel_rev=MAIN))
    assert (pin.k_snap, pin.chunks, pin.row_blocks) == (2048, 25, 6)
    assert _within(_ns(pin), 1_286_400, 0.5) and _within(_ns(pin), LI_MAIN_NS[(51200, 1536, 640)], 0.5)
    # valid_length trims the scanned chunks; the mode stays on the physical width.
    vl = predict_topk(TopkConfig(N=56320, K=1536, rows=2, N_phys=102400, kernel_rev=MAIN))
    assert vl.chunks == 28 and _within(_ns(vl), 241_709, 0.5)
    glm = predict_topk(TopkConfig(N=524288, K=2048, rows=160, N_phys=1048576, kernel_rev=MAIN))
    assert (glm.chunks, glm.segments, glm.row_blocks) == (256, 8, 2) and _within(_ns(glm), 4_394_243, 0.5)
    assert predict_topk(TopkConfig(N=65536, K=2048, rows=1, N_phys=131072, kernel_rev=MAIN)).device_cycles == \
        predict_topk(TopkConfig(N=65536, K=2048, rows=1, kernel_rev=MAIN)).device_cycles
    trimmed = predict_topk(TopkConfig(N=16384, K=512, N_phys=65536, kernel_rev=MAIN))
    assert (trimmed.body_mode, trimmed.chunks) == ("Classic", 32)
    # RT-DETR (1 x 8400, k 304): K'=512, 17 chunks end to end.
    rt_detr = predict_topk(TopkConfig(N=8400, K=304, kernel_rev=MAIN))
    assert (rt_detr.body_mode, rt_detr.chunks) == ("FusedEndToEnd", 17) and _within(_ns(rt_detr), LI_MAIN_NS[(8400, 304, 1)], 0.5)


@pytest.mark.unit
def test_body_mode_steps_follow_the_factory():
    # program_factory.cpp:130-143: snapped k >= 1024 -> FusedSegmented; else <= 32 chunks -> FusedEndToEnd.
    assert body_mode(512, 16384) == "FusedEndToEnd"       # 32 chunks
    assert body_mode(512, 16896) == "Classic"             # 33 chunks
    assert body_mode(1024, 1024) == "FusedSegmented"      # gate on K', even for one chunk
    assert predict_topk(TopkConfig(N=65536, K=528, kernel_rev=MAIN)).body_mode == "FusedSegmented"
    seg2 = predict_topk(TopkConfig(N=33792, K=1024, kernel_rev=MAIN))
    assert (seg2.body_mode, seg2.chunks, seg2.segments) == ("FusedSegmented", 33, 2)


@pytest.mark.unit
def test_predict_rejects_off_model_shapes():
    with pytest.raises(ValueError):
        predict_topk(TopkConfig(N=65536, K=4096))     # above the largest bucket
    with pytest.raises(ValueError):
        predict_topk(TopkConfig(N=256, K=512))        # N < K


@pytest.mark.unit
def test_column_split_law_is_off_by_default_and_follows_pr53457_when_on():
    row_law = predict_topk(TopkConfig(N=65536, K=2048, rows=1, kernel_rev=MAIN))
    assert "tree_merge" not in row_law.components and not row_law.low_confidence
    # Single row, 32 chunks: cost(P) = 2 ceil(32 / P) + ceil(log2 P) minimised at P=32 (7 units) vs 64.
    assert column_split_config(32, 1, (11, 10)) == (32, 7)
    split = predict_topk(TopkConfig(N=65536, K=2048, rows=1, column_split=True, kernel_rev=MAIN))
    u = rt.LARGE_INDICES_TERMS_MAIN.a_tot[(2048, "FusedSegmented")] / 2
    assert split.components["chunk_sort"] == pytest.approx(u * 2)
    assert split.components["tree_merge"] == pytest.approx(u * 5)
    assert split.device_cycles < row_law.device_cycles
    assert split.breakdown["column_split_cores"] == 32 and "column_split_uncalibrated" in split.low_confidence_reasons
    # Multi-row: every row needs its own rectangle, so more rows than half the grid falls back.
    assert column_split_config(32, 200, (11, 10)) is None
    many = predict_topk(TopkConfig(N=65536, K=2048, rows=200, column_split=True, kernel_rev=MAIN))
    assert many.device_cycles == predict_topk(TopkConfig(N=65536, K=2048, rows=200, kernel_rev=MAIN)).device_cycles


# ==============================================================================================
# Regime (b): generic ttnn.topk factories
# ==============================================================================================

@pytest.mark.unit
def test_insertion_steps_count():
    assert insertion_steps(4, 1) == 3            # Wt - 1 at Kt = 1
    assert insertion_steps(2004, 1) == 2003      # Sampling1D Llama width
    assert insertion_steps(16, 4) == 4 * 16 - 10  # Kt Wt - Kt (Kt + 1) / 2
    with pytest.raises(ValueError):
        insertion_steps(2, 4)


@pytest.mark.unit
def test_single_core_factory_law_class_a_and_the_row_cliff():
    t = rt.GENERIC_TERMS_MAIN
    r = predict_generic(GenericConfig(rows=32, N=4096, k=32, factory="single_core", kernel_rev=MAIN))
    wt, kt = 128, 1
    assert (r.breakdown["Wt"], r.breakdown["Kt"], r.breakdown["Ht"]) == (wt, kt, 1)
    assert r.components["insertion_sort"] == pytest.approx((wt - 1) * t.c_step["bf16/u16"])
    assert r.components["transpose_in"] == pytest.approx(wt * t.c_in["u16"])
    assert r.components["transpose_out"] == pytest.approx(2 * kt * t.c_out)
    assert r.components["launch"] == pytest.approx(t.c_launch) and r.components["slice"] == 0.0
    assert _within(_ns(r), 557_785, 0.5) and not r.low_confidence          # A-112
    # Kt slope: k=64 doubles the insertion steps minus the growing-phase correction (A-113 1,066,391 ns).
    r2 = predict_generic(GenericConfig(rows=32, N=4096, k=64, factory="single_core", kernel_rev=MAIN))
    assert r2.breakdown["n_sort"] == 2 * wt - 3 and _within(_ns(r2), 1_066_391, 0.5)
    assert _within(_ns(predict_generic(GenericConfig(rows=32, N=16384, k=2048, factory="single_core", kernel_rev=MAIN))), 123_694_586, 0.5)
    # Class C: tile rows over cores, Ht = C costs one block, Ht = C + 1 costs two (576,656 / 1,143,625 ns).
    c = predict_generic(GenericConfig(rows=32 * 110, N=4224, k=32, factory="single_core", kernel_rev=MAIN))
    c1 = predict_generic(GenericConfig(rows=32 * 111, N=4224, k=32, factory="single_core", kernel_rev=MAIN))
    assert _within(_ns(c), 576_656, 0.5) and _within(_ns(c1), 1_143_625, 0.5)
    assert 1.9 <= c1.device_cycles / c.device_cycles <= 2.1
    assert c1.device_cycles - c.device_cycles == pytest.approx(c.t_row_cycles, abs=1)


@pytest.mark.unit
def test_single_core_dtype_index_and_largest_false_terms():
    cells = [("bfloat16", 65504, 32, 8_840_236, "bf16/u16"), ("bfloat16", 65568, 32, 9_462_129, "bf16/u32"),
             ("bfloat16", 131072, 2048, 1_075_025_041, "bf16/u32"), ("bfloat8_b", 65536, 256, 68_616_352, "bfp8/u32"),
             ("bfloat8_b", 1024, 32, 143_440, "bfp8/u16"), ("float32", 65536, 256, 69_335_713, "fp32/fp32"),
             ("float32", 1024, 32, 159_992, "fp32/fp32")]
    for dtype, N, k, meas, key in cells:
        r = predict_generic(GenericConfig(rows=32, N=N, k=k, dtype=dtype, factory="single_core", kernel_rev=MAIN))
        assert r.breakdown["c_step_key"] == key and _within(_ns(r), meas, 0.5), (dtype, N, k, _ns(r), meas)
    t = rt.GENERIC_TERMS_MAIN
    assert t.c_step["bf16/u32"] / t.c_step["bf16/u16"] == pytest.approx(1.030, abs=0.002)
    assert t.c_step["fp32/fp32"] / t.c_step["bf16/u16"] == pytest.approx(1.035, abs=0.002)
    lo = predict_generic(GenericConfig(rows=32, N=4096, k=256, factory="single_core", largest=False, kernel_rev=MAIN))
    hi = predict_generic(GenericConfig(rows=32, N=4096, k=256, factory="single_core", kernel_rev=MAIN))
    assert lo.device_cycles - hi.device_cycles == pytest.approx(hi.breakdown["n_sort"] * t.largest_false_delta, abs=1)
    assert _within(_ns(lo), 4_086_092, 0.5)
    # An unknown dtype prices as bf16 and says so; the reader terms stay hidden behind compute.
    odd = predict_generic(GenericConfig(rows=32, N=4096, k=32, dtype="uint16", factory="single_core", kernel_rev=MAIN))
    assert "dtype_uint16_priced_as_bf16" in odd.low_confidence_reasons and odd.components["reader_wait"] == 0.0
    slow = GenericTerms(c_step={"bf16/u16": 6200.0}, c_in={"u16": 0.0}, c_out=0.0, c_row=0.0, c_launch=0.0,
                        c_read_tile=1000.0, c_idxgen=9000.0)
    rw = predict_generic(GenericConfig(rows=32, N=4096, k=32, factory="single_core", terms=slow, kernel_rev=MAIN))
    assert rw.components["reader_wait"] == pytest.approx(128 * 10000.0 - 127 * 6200.0)
    # A precomputed indices_tensor swaps index generation for a second tile read.
    idx = predict_generic(GenericConfig(rows=32, N=4096, k=32, factory="single_core", terms=slow, has_indices_tensor=True, kernel_rev=MAIN))
    assert idx.breakdown["t_reader_cycles"] == 128 * 2000.0


@pytest.mark.unit
def test_slices_follow_a_k_the_kernel_padded():
    # k=4 rounds to 32 for the kernel; two SliceDeviceOperations bring it back (M1 790 + 793 ns).
    m1 = predict_generic(GenericConfig(rows=32, N=128, k=4, kernel_rev=MAIN))
    assert m1.components["slice"] == pytest.approx(2 * (893.0 + 0.4631 * 32))
    assert _within((m1.device_cycles - m1.components["slice"]) / GHZ, 22_257, 1.0)
    assert _within(_ns(m1), 23_934, 3.0)
    assert predict_generic(GenericConfig(rows=32, N=256, k=32, kernel_rev=MAIN)).components["slice"] == 0.0
    # The composite slices only when k is not a multiple of 16.
    assert predict_topk_call(TopkCall(rows=32, N=128256, k=64, kernel_rev=MAIN)).components["slice"] == 0.0
    r2 = predict_topk_call(TopkCall(rows=32, N=128256, k=70, kernel_rev=MAIN))
    assert r2.breakdown["k_rounded"] == 80 and r2.components["slice"] > 0 and r2.k_snap == 512


@pytest.mark.unit
def test_multi_core_law_depends_on_kt():
    # Class E (finding F3): the local per-tile cost depends on Kt; at Kt=2 it is 1.31x the Kt=1 one and
    # Kt above 2 uses the Kt=2 cost.
    t = rt.GENERIC_TERMS_MAIN
    assert t.a_local[2] / t.a_local[1] == pytest.approx(1.31, abs=0.01)
    assert t.a_local_for(1) == t.a_local[1] and t.a_local_for(3) == t.a_local[2]
    for rows, N, k, meas, tol in ((32, 16384, 32, 132_787, 2.0), (32, 16384, 64, 211_404, 5.0), (32, 1024, 32, 37_460, 2.0),
                                  (32, 1024, 64, 54_354, 1.0), (32, 32768, 64, 304_696, 1.0), (32, 32768, 32, 171_996, 3.0),
                                  (64, 16384, 32, 201_625, 1.0), (128, 16384, 32, 349_124, 1.5)):
        r = predict_generic(GenericConfig(rows=rows, N=N, k=k, factory="multi_core", kernel_rev=MAIN))
        assert r.regime == "generic_multi_core" and _within(_ns(r), meas, tol), (rows, N, k, _ns(r), meas)
        assert (r.breakdown["Kt"] == 2) == (k == 64)
        assert all(v >= 0 for v in r.components.values())
        if rows > 32:
            assert "multi_core_later_rows_measured_at_one_width" in r.low_confidence_reasons
            assert r.components["row_pipeline"] == pytest.approx(t.row_pipe_frac * (r.components["local_sort"] + r.components["final_merge"]) * (r.breakdown["Ht"] - 1))
        else:
            assert not r.low_confidence and r.components["row_pipeline"] == 0.0
    assert "multi_core_kt_above_2_extrapolated" in \
        predict_generic(GenericConfig(rows=32, N=16384, k=96, factory="multi_core", kernel_rev=MAIN)).low_confidence_reasons


@pytest.mark.unit
def test_multi_core_split_matches_factory_worked_example():
    # TopK.md:335-345: W=16384 k=32 -> split 512, 32 local cores, scoring 7*16 + 2*32 = 176.
    assert multi_core_split(16384, 1, (11, 10)) == (32, 512)
    r = predict_generic(GenericConfig(rows=32, N=16384, k=32, factory="multi_core", kernel_rev=MAIN))
    assert (r.breakdown["n_local"], r.breakdown["Wt_local"], r.breakdown["Wt_final"]) == (32, 16, 32)
    assert r.components["local_sort"] == pytest.approx(rt.GENERIC_TERMS_MAIN.a_local[1] * 16)
    assert r.components["final_merge"] == pytest.approx(rt.GENERIC_TERMS_MAIN.a_final * 32)
    assert r.components["launch"] == pytest.approx(rt.GENERIC_TERMS_MAIN.c_mc0)
    # Observed splits on the full grid (ops CSV core counts minus the final core).
    for N, k, n in ((1024, 32, 8), (1024, 64, 8), (2048, 32, 16), (2048, 64, 8), (4096, 32, 16), (4096, 64, 16),
                    (8192, 32, 32), (8192, 64, 16), (16384, 64, 32), (32768, 32, 64), (32768, 64, 32)):
        assert multi_core_split(N, math.ceil(k / 32), (11, 10))[0] == n, (N, k)
    # No valid split on a tiny grid: fall back to the single core factory and flag it.
    small = predict_generic(GenericConfig(rows=32, N=16384, k=32, factory="multi_core", grid=(1, 1), kernel_rev=MAIN))
    assert small.regime == "generic_single_core" and "multi_core_split_unavailable" in small.low_confidence_reasons


@pytest.mark.unit
def test_generic_rejects_kt_beyond_the_l1_bound():
    with pytest.raises(ValueError):
        predict_generic(GenericConfig(rows=32, N=65536, k=4096, factory="single_core"))   # Kt 128 > 125
    predict_generic(GenericConfig(rows=32, N=65504, k=4000, factory="single_core"))       # Kt 125 fits


# ==============================================================================================
# Regime (c): MoE gates and moe_grouped_topk
# ==============================================================================================

@pytest.mark.unit
def test_gate_wrapper_ops_are_opt_in_and_track_the_token_sweep():
    """R1W: TTMoEGate.forward layout is linear in the tokens of the launch, and a bare ttnn gate call pays
    none of it. Six measured walls at 128 experts, ns of layout per call."""
    measured = {1: 10_759, 8: 12_181, 16: 12_568, 32: 13_724, 64: 14_588, 110: 16_394}
    for tokens, ns in measured.items():
        r = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=tokens, N=128, k=8, wrapper=True))
        assert r.components["wrapper_ops"] / 1.35 == pytest.approx(ns, rel=0.075), tokens
    # the slope is real: 110 tokens cost more layout than 1
    lo = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=1, N=128, k=8, wrapper=True))
    hi = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=110, N=128, k=8, wrapper=True))
    assert hi.components["wrapper_ops"] > lo.components["wrapper_ops"] * 1.3
    # opt-in: the bare ttnn op pays nothing
    bare = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=32, N=128, k=8))
    assert bare.components["wrapper_ops"] == 0.0
    # the expert axis was measured at 128 only
    assert "gate_wrapper_layout_above_128_experts_unmeasured" not in lo.low_confidence_reasons
    wide = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=32, N=512, k=8, wrapper=True))
    assert "gate_wrapper_layout_above_128_experts_unmeasured" in wide.low_confidence_reasons
    # the wrapper gives one core per token, so beyond the grid it does not run at all
    over = predict_gate(GateConfig(kernel="generalized_moe_gate", tokens=128, N=128, k=8,
                                   num_cores=110, wrapper=True))
    assert "gate_wrapper_above_one_token_per_core_does_not_run" in over.low_confidence_reasons


def test_gate_start_skew_grows_with_active_cores_then_whole_launches():
    # generalized_moe_gate, 256 experts, k=4 no softmax: 1322 / 1452 / 1474 ns at B 1 / 32 / 110
    # (finding F5: the duration spans the first core's start to the last core's end).
    for B, meas in ((1, 1_322), (32, 1_452), (110, 1_474)):
        r = predict_gate(GateConfig("generalized_moe_gate", tokens=B, k=4, kernel_rev=MAIN))
        assert _within(_ns(r), meas, 3.0), (B, _ns(r), meas)
        assert r.components["start_skew"] == pytest.approx(rt.GATE_TERMS_MAIN.skew(B))
    t = rt.GATE_TERMS_MAIN
    assert t.skew(1) == 0.0 and 0.0 < t.skew(8) < t.skew(32) < t.skew(110) == t.skew(200)
    full = predict_gate(GateConfig("generalized_moe_gate", tokens=110, k=4, kernel_rev=MAIN))
    over = predict_gate(GateConfig("generalized_moe_gate", tokens=111, k=4, kernel_rev=MAIN))
    assert over.row_blocks == 2 and over.device_cycles == pytest.approx(2 * full.device_cycles, abs=1)
    # k enters neither term: k=4 and k=8 price alike (measured within 3 percent).
    assert predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=8, kernel_rev=MAIN)).device_cycles == \
        predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=4, kernel_rev=MAIN)).device_cycles
    with pytest.raises(ValueError):
        predict_gate(GateConfig("topk_router_gpt", tokens=32))


@pytest.mark.unit
def test_gate_softmax_sigmoid_and_512_layout_terms():
    sm = predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=8, softmax=True, kernel_rev=MAIN))
    base = predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=8, kernel_rev=MAIN))
    assert sm.components["softmax"] == pytest.approx(50.4) and base.components["softmax"] == 0.0
    assert _within(_ns(sm), 1_510, 3.0) and _within(_ns(base), 1_473, 3.0)
    sig = predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=8, sigmoid=True, kernel_rev=MAIN))
    assert sig.device_cycles - base.device_cycles == pytest.approx(307.8, abs=1)     # 1712 vs 1484 ns
    big = predict_gate(GateConfig("generalized_moe_gate", tokens=32, k=8, N=512, kernel_rev=MAIN))
    assert big.components["combine_512"] == pytest.approx(6586.0) and _within(_ns(big), 6_370, 2.0)
    assert "gate_layout_above_512_experts_unmeasured" in predict_gate(GateConfig("generalized_moe_gate", tokens=32, N=1024, kernel_rev=MAIN)).low_confidence_reasons
    # deepseek_moe_gate runs its sigmoid front end by default: 1307 / 1413 / 1435 ns at B 1 / 32 / 110.
    for B, meas in ((1, 1_307), (32, 1_413), (110, 1_435)):
        d = predict_gate(GateConfig("deepseek_moe_gate", tokens=B, kernel_rev=MAIN))
        assert d.components["sigmoid"] > 0 and _within(_ns(d), meas, 1.0), (B, _ns(d), meas)
    # there is no "sampling" gate kernel: those call sites take the composite route, so asking for one raises
    with pytest.raises(ValueError, match="unknown gate kernel"):
        predict_gate(GateConfig("sampling", tokens=32, kernel_rev=MAIN))
    # an uncalibrated set is flagged through its terms
    placeholder = GateTerms(c_gate_token={"generalized_moe_gate": 580.0, "deepseek_moe_gate": 430.0}, calibrated=False)
    assert "gate_uncalibrated_on_this_kernel_rev" in predict_gate(GateConfig("generalized_moe_gate", tokens=32, terms=placeholder, kernel_rev=MAIN)).low_confidence_reasons


@pytest.mark.unit
def test_moe_grouped_topk_law_with_wave_gap_and_grouped_chain():
    # R2: N=128 k=4 at T 32 / 4096 (23,618 / 49,018 ns), N=384 T=4096 (134,427), grouped N=256.
    for T, N, groups, meas, tol in ((32, 128, 1, 23_618, 2.0), (1024, 128, 1, 24_263, 2.0), (4096, 128, 1, 49_018, 1.0),
                                    (4096, 384, 1, 134_427, 1.0), (32, 256, 8, 60_724, 2.0), (1024, 256, 8, 61_646, 1.0),
                                    (4096, 256, 8, 125_156, 2.0)):
        r = predict_gate(GateConfig("moe_grouped_topk", tokens=T, N=N, k=8, n_groups=groups, kernel_rev=MAIN))
        assert _within(_ns(r), meas, tol), (T, N, groups, _ns(r), meas)
        assert r.components["wave_gap"] == (1734.8 if T == 4096 else 0.0)
        assert (r.components["grouped_chain"] > 0) == (groups > 1)
        assert not r.low_confidence
    r = predict_gate(GateConfig("moe_grouped_topk", tokens=1024, N=128, k=4, kernel_rev=MAIN))
    assert r.breakdown["Wt"] == 4 and r.row_blocks == 1
    assert r.components["width_tiles"] == pytest.approx(4 * 7206.4) and r.components["row_fixed"] == pytest.approx(3394.2)


# ==============================================================================================
# Router: production cells from sweep_plan.md section 2.2
# ==============================================================================================

@pytest.mark.unit
@pytest.mark.parametrize("cell,call,route,detail", [
    ("S1 sampling Llama3", TopkCall(rows=32, N=128256, k=64), "composite", ("Classic", 251)),
    ("S1b sampling Qwen3", TopkCall(rows=32, N=151936, k=128), "composite", ("Classic", 297)),
    ("S1c sampling Gemma", TopkCall(rows=32, N=256000, k=128), "composite", ("Classic", 500)),
    ("S2 sampling chunk", TopkCall(rows=32, N=64128, k=32), "composite", ("Classic", 126)),
    ("S3 pow2 shard", TopkCall(rows=32, N=16384, k=32), "stock_multi_core", None),
    ("S3 non pow2 shard", TopkCall(rows=32, N=16032, k=32), "composite", ("FusedEndToEnd", 32)),
    ("S4 Sampling1D Llama", TopkCall(rows=32, N=64128, k=32, sub_core_grids=1), "stock_single_core", None),
    ("S4 Sampling1D Qwen", TopkCall(rows=32, N=75968, k=32, sub_core_grids=1), "stock_single_core", None),
    ("S5 log probs", TopkCall(rows=32, N=256, k=32, sub_core_grids=1), "stock_single_core", None),
    ("M1 GPT-OSS router", TopkCall(rows=32, N=128, k=4), "stock_single_core", None),
    ("M2 d_p gate prefill", TopkCall(rows=4096, N=128, k=4), "stock_single_core", None),
    ("M3 Gemma4 router", TopkCall(rows=1024, N=128, k=8), "stock_single_core", None),
    ("M4 TTMoEGate fallback", TopkCall(rows=32, N=512, k=10), "stock_single_core", None),
    ("M5 Mixtral prefill", TopkCall(rows=1024, N=64, k=32), "stock_single_core", None),
    # k > 64 skips the small-k width floor, so the Informer cell is routed at any width >= round16(k)
    ("M6 Informer k96", TopkCall(rows=256, N=96, k=96), "composite", ("FusedEndToEnd", 1)),
    ("M6 Informer k32", TopkCall(rows=256, N=96, k=32), "stock_single_core", None),
    ("E 1024 default", TopkCall(rows=32, N=1024, k=32), "stock_multi_core", None),
    ("E 4096 default", TopkCall(rows=32, N=4096, k=32), "composite", ("FusedEndToEnd", 8)),
    ("E 8192 default", TopkCall(rows=32, N=8192, k=64), "stock_multi_core", None),
    ("F 65536 k128", TopkCall(rows=32, N=65536, k=128), "composite", ("Classic", 128)),
])
def test_routing_of_production_cells(cell, call, route, detail):
    assert route_topk(call) == route, cell
    r = predict_topk_call(call)
    assert r.route == route
    if detail is not None:
        assert (r.body_mode, r.chunks) == detail, cell
    assert r.device_cycles == int(round(sum(r.components.values())))


@pytest.mark.unit
def test_sub_core_grids_keeps_the_kernel_factory_choice():
    # A passed grid disables the composite route only; the stock factory is chosen on that grid (multi core
    # needs a pow2 width in the gate range and a first range of at least 2 x 3). Class E passes the full grid.
    e = dict(rows=32, N=16384, k=32, kernel_rev=MAIN)
    assert route_topk(TopkCall(**e)) == "stock_multi_core"
    full = predict_topk_call(TopkCall(**e, sub_core_grids=110, sub_core_grid=(11, 10)))
    assert full.route == "stock_multi_core" and full.device_cycles == predict_topk_call(TopkCall(**e)).device_cycles
    for grid, route in (((3, 4), "stock_multi_core"), ((9, 10), "stock_multi_core"),
                        ((2, 3), "stock_single_core"), ((8, 1), "stock_single_core"), ((1, 1), "stock_single_core")):
        call = TopkCall(**e, sub_core_grids=grid[0] * grid[1], sub_core_grid=grid)
        assert route_topk(call) == route, grid
        assert predict_topk_call(call).route == route
    # Grid variation at (16384, 32): 3x4 gives n = 4 (690.1 us), 5x6 n = 16 (202.2 us, holdout),
    # 2x3 falls to one core (2,215.1 us, the class A cell).
    g34 = predict_topk_call(TopkCall(**e, sub_core_grids=12, sub_core_grid=(3, 4)))
    g56 = predict_topk_call(TopkCall(**e, sub_core_grids=30, sub_core_grid=(5, 6)))
    assert (g34.breakdown["n_local"], g56.breakdown["n_local"], full.breakdown["n_local"]) == (4, 16, 32)
    assert _within(_ns(g34), 690_082, 8.0) and _within(_ns(g56), 202_216, 8.0)
    assert _within(_ns(predict_topk_call(TopkCall(**e, sub_core_grids=6, sub_core_grid=(2, 3)))), 2_215_064, 1.0)
    # A count without extents stays on the single core factory; non pow2 widths do regardless.
    assert route_topk(TopkCall(**e, sub_core_grids=110)) == "stock_single_core"
    assert route_topk(TopkCall(rows=32, N=64128, k=32, sub_core_grids=110, sub_core_grid=(11, 10))) == "stock_single_core"


@pytest.mark.unit
def test_route_predicate_disablers():
    base = TopkCall(rows=32, N=128256, k=64)
    assert should_route_to_topk_large_indices(base)
    for change in (dict(largest=False), dict(stable=True), dict(has_indices_tensor=True),
                   dict(has_output_tensor=True), dict(sub_core_grids=110), dict(sharded=True),
                   dict(dim_is_last=False), dict(k=2049), dict(dtype="bfloat8_b"),
                   dict(layout="ROW_MAJOR"), dict(arch="Wormhole"), dict(N=1 << 20)):
        call = TopkCall(**{**base.__dict__, **change})
        assert not should_route_to_topk_large_indices(call), change
    # Small k: routed only when the padded width is >= 4096 and not multi-core eligible.
    assert not should_route_to_topk_large_indices(TopkCall(rows=32, N=2048, k=32))
    assert should_route_to_topk_large_indices(TopkCall(rows=32, N=4096, k=32))      # pow2 in [4096, 8192)
    assert not should_route_to_topk_large_indices(TopkCall(rows=32, N=8192, k=32))  # stock multi-core


@pytest.mark.unit
def test_composite_wrapper_ops_step_in_k_and_the_li_start_up_at_32_rows():
    t = rt.GENERIC_TERMS_MAIN
    # The finish op is flat up to k16 = 512 (it runs on k16 / 16 cores, 512 gathered elements each):
    # 21.4 to 25.0 us for every N at k 32 to 512; 85 / 94 / 105 us at k 2048 for N 8192 / 65536 / 262144.
    for N, k in ((4096, 32), (8192, 512), (128256, 64), (256000, 128)):
        assert predict_topk_call(TopkCall(rows=32, N=N, k=k, kernel_rev=MAIN)).components["route_finish"] == pytest.approx(t.c_fin_flat)
    for N, meas in ((8192, 84_973), (65536, 93_797), (262144, 104_763)):
        r = predict_topk_call(TopkCall(rows=32, N=N, k=2048, kernel_rev=MAIN))
        assert _within(r.components["route_finish"] / GHZ, meas, 3.5), (N, r.components["route_finish"] / GHZ)
        assert not r.low_confidence
    assert "finish_k_between_512_and_2048_unmeasured" in predict_topk_call(TopkCall(rows=32, N=65536, k=1024, kernel_rev=MAIN)).low_confidence_reasons
    assert "composite_wrapper_measured_at_32_rows" in predict_topk_call(TopkCall(rows=64, N=65536, k=128, kernel_rev=MAIN)).low_confidence_reasons
    # Prep is linear in the input elements (49.4 / 58.1 / 98.3 us on the three sampling cells).
    for N, k, meas in ((128256, 64, 49_435), (151936, 128, 58_075), (256000, 128, 98_303)):
        r = predict_topk_call(TopkCall(rows=32, N=N, k=k, kernel_rev=MAIN))
        assert r.components["route_prep"] == pytest.approx(t.c_prep0 + t.c_prep_elem * 32 * N)
        assert _within(r.components["route_prep"] / GHZ, meas, 4.0)
    # The composite's large_indices op runs 32 rows on 32 cores: the start-up is the 32-core value
    # (F-210 measured 14,109 ns against 14,050 at R=1; the full-grid value would add 25 percent).
    r = predict_topk_call(TopkCall(rows=32, N=4096, k=32, kernel_rev=MAIN))
    assert r.components["launch"] == pytest.approx(859.05)
    li = sum(v for key, v in r.components.items() if key not in ("route_prep", "route_finish", "slice"))
    assert _within(li / GHZ, 14_109, 5.0)


@pytest.mark.unit
@pytest.mark.parametrize("cell,call,measured_ns,tol", [
    # Composite totals over all ops per call (results_G.csv per_iter_sum_all_ops).
    ("S1 ttsampling llama3", TopkCall(rows=32, N=128256, k=64), 616_884, 1.0),
    ("S1b ttsampling qwen3", TopkCall(rows=32, N=151936, k=128), 725_411, 1.0),
    ("S1c ttsampling gemma", TopkCall(rows=32, N=256000, k=128), 1_205_713, 1.0),
    ("S3 shard non pow2", TopkCall(rows=32, N=16032, k=32), 82_762, 3.0),
    ("F 65536 k2048 (PR 53457 anchor)", TopkCall(rows=32, N=65536, k=2048), 396_999, 1.0),
    # Stock cells: the TopK op alone (slices excluded below) against the ops CSV.
    ("S3 shard pow2", TopkCall(rows=32, N=16384, k=32), 132_885, 2.0),
    ("S5 log probs", TopkCall(rows=32, N=256, k=32, sub_core_grids=1, sub_core_grid=(1, 1)), 39_625, 1.0),
    ("S4 sampling1d llama3", TopkCall(rows=32, N=64128, k=32, sub_core_grids=110, sub_core_grid=(11, 10)), 8_655_769, 1.0),
    ("S4 sampling1d qwen3 u32", TopkCall(rows=32, N=75968, k=32, sub_core_grids=110, sub_core_grid=(11, 10)), 10_966_780, 1.0),
    ("S1b stock contrast Kt 4", TopkCall(rows=32, N=151936, k=128, sub_core_grids=110, sub_core_grid=(11, 10)), 80_954_014, 1.0),
    ("M6 informer", TopkCall(rows=256, N=96, k=32), 18_189, 1.0),
])
def test_production_cells_track_the_campaign(cell, call, measured_ns, tol):
    call.kernel_rev = MAIN
    r = predict_topk_call(call)
    pred = r.device_cycles - (r.components.get("slice", 0.0) if r.route != "composite" else 0.0)
    assert _within(pred / GHZ, measured_ns, tol), f"{cell}: pred {pred / GHZ:.0f} meas {measured_ns}"


@pytest.mark.unit
@pytest.mark.parametrize("cell,call,topk_ns,total_ns", [
    ("M1 gpt-oss decode DRAM", TopkCall(rows=32, N=128, k=4), 22_257, 23_934),
    ("M1 gpt-oss decode L1", TopkCall(rows=32, N=128, k=4, input_memory="L1"), 21_781, 22_700),
    ("M2 deepseek d_p prefill", TopkCall(rows=4096, N=128, k=4), 39_541, 43_366),
    ("M4 qwen3.5 fallback", TopkCall(rows=32, N=512, k=10), 74_150, 75_747),
    ("M2 kimi ungrouped", TopkCall(rows=4096, N=384, k=8), 107_568, 111_804),
    ("M3 gemma4", TopkCall(rows=1024, N=128, k=8), 22_604, 24_882),
])
def test_router_cells_with_slices_track_the_campaign(cell, call, topk_ns, total_ns):
    call.kernel_rev = MAIN
    r = predict_topk_call(call)
    assert r.route == "stock_single_core" and r.components["slice"] > 0
    assert _within((r.device_cycles - r.components["slice"]) / GHZ, topk_ns, 6.0), cell     # M2 prefill is -5.1%
    assert _within(_ns(r), total_ns, 6.0), cell


# ==============================================================================================
# Components sum to the wall, in every regime and through the device path
# ==============================================================================================

@pytest.mark.unit
@pytest.mark.parametrize("result", [
    predict_topk(TopkConfig(N=51200, K=1536, rows=640, kernel_rev=MAIN)),
    predict_topk(TopkConfig(N=440, K=16, rows=640, kernel_rev=MAIN)),
    predict_topk(TopkConfig(N=65536, K=2048, rows=1, column_split=True, kernel_rev=MAIN)),
    predict_topk_call(TopkCall(rows=32, N=128256, k=64, kernel_rev=MAIN)),
    predict_topk_call(TopkCall(rows=32, N=65536, k=2048, kernel_rev=MAIN)),
    predict_topk_call(TopkCall(rows=128, N=16384, k=32, kernel_rev=MAIN)),
    predict_topk_call(TopkCall(rows=4096, N=128, k=4, kernel_rev=MAIN)),
    predict_gate(GateConfig("generalized_moe_gate", tokens=200, k=6, softmax=True, kernel_rev=MAIN)),
    predict_gate(GateConfig("moe_grouped_topk", tokens=4096, N=256, k=8, n_groups=8, kernel_rev=MAIN)),
    predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88, kernel_rev=MAIN)),
], ids=lambda r: f"{r.regime}:{r.route}:{r.kernel_rev}")
def test_components_sum_to_wall(result):
    ps = result.to_polaris_op_perf_stats()
    assert _components_sum(ps) == ps["fused_compute_cycles"] == result.device_cycles
    assert all(v >= 0 for v in ps["topk_wall_components"].values())
    assert ps["topk_regime"] == result.regime and ps["topk_route"] == result.route
    assert ps["topk_kernel_rev"] == result.kernel_rev


@pytest.mark.unit
def test_fused_cycles_honored_on_bh_and_dropped_off_bh():
    ps = {
        "inElems": 0, "outElems": 0, "inBytes": 4000, "outBytes": 4000,
        "instrs": {"mov": 65536},
        **topk_perf_stats(TopkConfig(N=65536, K=64, rows=130, kernel_rev=MAIN)),
    }
    bh = Device(_MockSimConfig())
    op = _topk_op("topk_bh", dict(ps))
    bh.execute_op(op)
    assert op.compute_cycles == int(math.ceil(ps["fused_compute_cycles"]))
    assert op.compute_cycles == _components_sum(ps)
    # Off-BH the arch gate drops the calibrated cost for the generic instr estimate.
    wh = Device(_MockSimConfig(devname="Wormhole"))
    op2 = _topk_op("topk_wh", dict(ps))
    wh.execute_op(op2)
    assert op2.compute_cycles != int(ps["fused_compute_cycles"])
    assert op2.compute_cycles == math.ceil(65536 / (128.0 * wh.DG_COMPUTE_UTIL_CONSTANT))
    # A composite (routed ttnn.topk) goes through the same path with its wrapper terms included.
    comp = predict_topk_call(TopkCall(rows=32, N=128256, k=64, kernel_rev=MAIN)).to_polaris_op_perf_stats()
    op3 = _topk_op("topk_composite", {"inBytes": 0, "outBytes": 0, "instrs": {"mov": 1}, **comp})
    bh.execute_op(op3)
    assert op3.compute_cycles == _components_sum(comp)


# ==============================================================================================
# topk_sinf: the sim op picks the regime and the calibration set from the shim attrs
# ==============================================================================================

@pytest.mark.unit
def test_topk_sinf_prices_direct_large_indices_call():
    ps = _li_sinf([1, 1, 130, 65536], k=64)
    expected = predict_topk(TopkConfig(N=65536, K=64, rows=130))
    assert ps["fused_compute_cycles"] == expected.device_cycles
    assert ps["topk_route"] == "large_indices_direct" and ps["topk_kernel_rev"] == MAIN
    assert f"kernel_rev_defaulted_{MAIN}" in ps["topk_low_confidence_reasons"]
    assert ps["sdpa_calibrated_arch"] == "Blackhole" and ps["sdpa_calibrated_sku"] == "p100a"
    assert ps["instrs"] == {"mov": 1 * 1 * 130 * 65536}   # device-agnostic fallback count
    assert ps["topk_cycle_breakdown"]["row_blocks"] == 2  # ceil(130/110)
    assert _components_sum(ps) == ps["fused_compute_cycles"]
    # Naming the revision clears the default flag.
    named = _li_sinf([1, 1, 130, 65536], k=64, kernel_rev=MAIN)
    assert named["topk_kernel_rev"] == MAIN and named["topk_low_confidence"] is False
    l1 = _li_sinf([1, 1, 130, 65536], k=64, kernel_rev=MAIN, input_memory="L1")
    assert l1["topk_wall_components"]["launch"] == pytest.approx(281.0)


@pytest.mark.unit
def test_topk_sinf_valid_length_bounds_the_scan():
    full = _li_sinf([1, 1, 8, 65536], k=64, kernel_rev=MAIN)
    trimmed = _li_sinf([1, 1, 8, 65536], k=64, valid_length=16384, kernel_rev=MAIN)
    assert trimmed["fused_compute_cycles"] == predict_topk(TopkConfig(N=16384, K=64, rows=8, N_phys=65536, kernel_rev=MAIN)).device_cycles
    assert trimmed["fused_compute_cycles"] < full["fused_compute_cycles"]
    assert trimmed["topk_body_mode"] == "Classic"           # mode from the physical width (128 chunks)
    unknown = _li_sinf([1, 1, 8, 65536], k=64, valid_length_unknown=1, kernel_rev=MAIN)
    assert unknown["fused_compute_cycles"] == full["fused_compute_cycles"]
    assert "valid_length_tensor_priced_at_full_width" in unknown["topk_low_confidence_reasons"]


@pytest.mark.unit
def test_topk_sinf_falls_back_generic_off_model():
    # topk_large_indices above its largest bucket, and any non-last-axis topk, keep the generic estimate.
    big_k = _li_sinf([1, 1, 8, 65536], k=4096)
    assert "fused_compute_cycles" not in big_k and big_k["instrs"] == {"mov": 0}
    off_axis = _sinf_perf_stats([4096, 8], k=16, axis=0)
    assert "fused_compute_cycles" not in off_axis and off_axis["instrs"] == {"mov": 0}
    # The generic kernel rejects Kt beyond the L1 bound too (k=4096 is not routed, k > 2048).
    generic_big = _sinf_perf_stats([1, 1, 32, 65536], k=4096)
    assert "fused_compute_cycles" not in generic_big
    # An unknown kernel revision has no calibration set: the generic estimate stays as well.
    unknown_rev = _sinf_perf_stats([1, 1, 32, 4224], k=32, kernel_rev="deadbeef")
    assert "fused_compute_cycles" not in unknown_rev


@pytest.mark.unit
def test_topk_sinf_functional_default_goes_to_the_stock_factory():
    # A functional float32 topk is not bf16, so it is never routed; a non pow2 width lands on the
    # stock single core factory (pow2 widths in the gate range would take the multi core factory).
    ps = _sinf_perf_stats([1, 1, 32, 4224], k=32, kernel_rev=MAIN)
    r = predict_generic(GenericConfig(rows=32, N=4224, k=32, dtype="float32", factory="single_core", kernel_rev=MAIN))
    assert ps["topk_route"] == "stock_single_core" and ps["fused_compute_cycles"] == r.device_cycles
    assert ps["topk_cycle_breakdown"]["index_dtype"] == "fp32" and ps["topk_cycle_breakdown"]["c_step_key"] == "fp32/fp32"


# ==============================================================================================
# ttnn shim: attrs captured, routes reached, DSA chain
# ==============================================================================================

@pytest.mark.unit
def test_ttnn_topk_large_indices_shim_routes_to_roofline():
    with _TtnnDevice() as dev:
        import ttsim.front.ttnn as ttnn
        from ttsim.front.ttnn.tensor import Layout
        x = _tt(dev, [1, 1, 2048, 8192], layout=Layout.ROW_MAJOR_LAYOUT)
        idx = ttnn.experimental.topk_large_indices(x, k=2048, valid_length=8192, kernel_rev="main")
        assert list(idx.shape) == [1, 1, 2048, 2048]
        op = list(dev.ops.values())[-1]
        assert op.optype == "TopK" and op.attrs["kernel_rev"] == "main" and op.attrs["input_memory"] == "DRAM"
        expected = predict_topk(TopkConfig(N=8192, K=2048, rows=2048, kernel_rev=MAIN))
        assert op.perf_stats["fused_compute_cycles"] == expected.device_cycles
        assert op.perf_stats["topk_route"] == "large_indices_direct" and op.perf_stats["topk_kernel_rev"] == MAIN
        assert op.perf_stats["topk_body_mode"] == "FusedSegmented" and op.perf_stats["topk_low_confidence"] is False
        assert op.perf_stats["outBytes"] == 2048 * 2048 * 4          # uint32 indices only
        # DSA default shape: 670 us measured on main tip.
        assert _within(op.perf_stats["fused_compute_cycles"] / GHZ, LI_MAIN_NS[(8192, 2048, 2048)], 1.0)
        ttnn.experimental.topk_large_indices(x, k=2048, valid_length=8192)
        op = list(dev.ops.values())[-1]
        assert "kernel_rev" not in op.attrs and f"kernel_rev_defaulted_{MAIN}" in op.perf_stats["topk_low_confidence_reasons"]


@pytest.mark.unit
def test_ttnn_topk_large_indices_shim_direct_cells():
    with _TtnnDevice() as dev:
        import ttsim.front.ttnn as ttnn
        from ttsim.front.ttnn.tensor import Layout, DataType
        # D2 MiniMax MSA block select: 640 x 440 block columns, k=16 -> one chunk, FusedEndToEnd, 6 waves.
        vl = _tt(dev, [1], dtype=DataType.UINT32, layout=Layout.ROW_MAJOR_LAYOUT)
        idx = ttnn.experimental.topk_large_indices(_tt(dev, [1, 1, 640, 440], layout=Layout.ROW_MAJOR_LAYOUT),
                                                   k=16, valid_length_tensor=vl, kernel_rev="main")
        op = list(dev.ops.values())[-1]
        assert list(idx.shape) == [1, 1, 640, 16]
        assert (op.perf_stats["topk_body_mode"], op.perf_stats["topk_cycle_breakdown"]["chunks"]) == ("FusedEndToEnd", 1)
        assert op.attrs["valid_length_unknown"] == 1 and "valid_length" not in op.attrs
        assert "valid_length_tensor_priced_at_full_width" in op.perf_stats["topk_low_confidence_reasons"]
        assert _within(op.perf_stats["fused_compute_cycles"] / GHZ, LI_MAIN_NS[(440, 16, 640)], 3.0)
        # D3 RT-DETR: 8400 anchors, k rounded to 304 -> K'=512, 17 chunks.
        ttnn.experimental.topk_large_indices(_tt(dev, [1, 1, 1, 8400], layout=Layout.ROW_MAJOR_LAYOUT), k=304, kernel_rev="main")
        op = list(dev.ops.values())[-1]
        assert (op.perf_stats["topk_body_mode"], op.perf_stats["topk_cycle_breakdown"]["chunks"]) == ("FusedEndToEnd", 17)
        assert _within(op.perf_stats["fused_compute_cycles"] / GHZ, LI_MAIN_NS[(8400, 304, 1)], 0.5)


@pytest.mark.unit
def test_ttnn_topk_shim_captures_routing_attrs():
    with _TtnnDevice() as dev:
        import ttsim.front.ttnn as ttnn
        from ttsim.front.ttnn.tensor import Layout, DataType
        # S1: sampling defaults route to the composite.
        vals, idxs = ttnn.topk(_tt(dev, [1, 1, 32, 128256]), k=64, dim=-1, kernel_rev="main")
        op = list(dev.ops.values())[-1]
        assert list(vals.shape) == [1, 1, 32, 64]
        assert op.attrs["input_layout"] == "TILE" and op.attrs["input_dtype"] == "bfloat16"
        assert op.attrs["stable"] == 0 and op.attrs["sharded"] == 0 and "sub_core_grid_cores" not in op.attrs
        assert op.perf_stats["topk_route"] == "composite" and op.perf_stats["topk_kernel_rev"] == MAIN
        assert _within(op.perf_stats["fused_compute_cycles"] / GHZ, 616_884, 1.0)
        # S4: Sampling1D passes sub_core_grids -> no route, single core factory on that grid.
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
        ttnn.topk(_tt(dev, [1, 1, 32, 64128]), k=32, dim=-1, sub_core_grids=one_core)
        op = list(dev.ops.values())[-1]
        assert (op.attrs["sub_core_grid_cores"], op.attrs["sub_core_grid_x"], op.attrs["sub_core_grid_y"]) == (1, 1, 1)
        assert op.perf_stats["topk_route"] == "stock_single_core"
        assert op.perf_stats["topk_cycle_breakdown"]["Wt"] == 2004
        # Class E: the full grid passed explicitly keeps the multi core factory for a pow2 width.
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))])
        ttnn.topk(_tt(dev, [1, 1, 32, 16384]), k=32, dim=-1, sub_core_grids=full_grid)
        op = list(dev.ops.values())[-1]
        assert (op.attrs["sub_core_grid_cores"], op.attrs["sub_core_grid_x"], op.attrs["sub_core_grid_y"]) == (110, 11, 10)
        assert op.perf_stats["topk_route"] == "stock_multi_core"
        # M1: routers hit the single core factory (Wt 4, Kt 1) and k is sliced back to 4; an L1
        # input is recorded for the start-up term.
        vals, _ = ttnn.topk(_tt(dev, [1, 1, 32, 128], memory_config=ttnn.L1_MEMORY_CONFIG), k=4, dim=-1)
        op = list(dev.ops.values())[-1]
        assert list(vals.shape) == [1, 1, 32, 4] and op.attrs["input_memory"] == "L1"
        assert op.perf_stats["topk_route"] == "stock_single_core"
        assert (op.perf_stats["topk_cycle_breakdown"]["Wt"], op.perf_stats["topk_cycle_breakdown"]["Kt"]) == (4, 1)
        assert op.perf_stats["topk_wall_components"]["slice"] > 0
        # stable / largest=False / bfp8 / indices_tensor each disable the route.
        ttnn.topk(_tt(dev, [1, 1, 32, 128256]), k=64, dim=-1, stable=True)
        assert list(dev.ops.values())[-1].perf_stats["topk_route"] != "composite"
        ttnn.topk(_tt(dev, [1, 1, 32, 128256]), k=64, dim=-1, largest=False)
        assert list(dev.ops.values())[-1].perf_stats["topk_route"] == "stock_single_core"
        ttnn.topk(_tt(dev, [1, 1, 32, 128256], dtype=DataType.BFLOAT8_B), k=64, dim=-1)
        op = list(dev.ops.values())[-1]
        assert op.attrs["input_dtype"] == "bfloat8_b" and op.perf_stats["topk_route"] == "stock_single_core"
        assert op.perf_stats["topk_cycle_breakdown"]["c_step_key"] == "bfp8/u32"
        idx_t = _tt(dev, [1, 1, 32, 128256], dtype=DataType.UINT16)
        ttnn.topk(_tt(dev, [1, 1, 32, 128256]), k=64, dim=-1, indices_tensor=idx_t)
        op = list(dev.ops.values())[-1]
        assert op.attrs["has_indices_tensor"] == 1 and len(op.inList) == 2
        assert op.perf_stats["topk_route"] == "stock_single_core"
        # S3 pow2 shard at default args stays on the stock multi core bitonic.
        ttnn.topk(_tt(dev, [1, 1, 32, 16384]), k=32, dim=-1)
        assert list(dev.ops.values())[-1].perf_stats["topk_route"] == "stock_multi_core"


# ==============================================================================================
# indexer_score_dsa
# ==============================================================================================

@pytest.mark.unit
def test_indexer_head_geometry_is_linear_in_hi_and_nearly_flat_in_d():
    """IXHD: six walls at Sq 512, T 8192. The wall is proportional to Hi and almost independent of D,
    because the op is not matmul bound; the old Hi x D form reads about 42 percent low at D 64."""
    measured = {(32, 64): 1_237_311, (32, 128): 1_419_104, (64, 64): 2_441_520,
                (64, 128): 2_840_402, (128, 64): 4_938_121, (128, 128): 5_657_622}
    for (hi, d), ns in measured.items():
        r = predict_indexer(IndexerConfig(Sq=512, T=8192, Hi=hi, D=d, num_cores=88))
        assert r.device_cycles / 1.35 == pytest.approx(ns, rel=0.02), (hi, d)
    # doubling Hi doubles the wall; halving D costs far less than half
    base = measured[(64, 128)]
    assert measured[(128, 128)] / base == pytest.approx(2.0, rel=0.02)
    assert measured[(64, 64)] / base == pytest.approx(0.86, rel=0.03)
    # the naive per-pair MAC scaling would have predicted 0.5 there
    naive = (64 * 64) / (64 * 128)
    assert abs(naive - measured[(64, 64)] / base) > 0.3
    # outside the swept axes the scale is an extrapolation and says so
    out = predict_indexer(IndexerConfig(Sq=512, T=8192, Hi=64, D=32, num_cores=88))
    assert any("indexer_d_outside" in x for x in out.low_confidence_reasons)


def test_indexer_model_is_linear_in_pairs_and_pins_the_three_points():
    for (sq, t), meas in INDEXER_NS.items():
        r = predict_indexer(IndexerConfig(Sq=sq, T=t, Hi=64, D=128, num_cores=88, kernel_rev=MAIN))
        assert _within(_ns(r), meas, 2.0), (sq, t, _ns(r), meas)
        assert not r.low_confidence
        assert r.components["pair_scan"] == pytest.approx(79.93 * sq * t / 88)
    r = predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88, kernel_rev=MAIN))
    # 0.67 ns per query-key pair on 88 cores; the HiFi2 matmul floor as the SDPA roofline counts it
    # (32 cycles per 32x32x32 tile-MAC) is a tenth of the wall, so the op is not matmul bound.
    assert _ns(r) / (2048 * 8192) == pytest.approx(0.673, abs=0.005)
    assert 0.09 < r.breakdown["fpu_floor_cycles"] / r.device_cycles < 0.11
    # Scaled by cores; the fidelity is recorded, not priced (the op is not matmul bound).
    r110 = predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=110, kernel_rev=MAIN))
    assert r110.device_cycles == pytest.approx(r.device_cycles * 88 / 110, rel=1e-6)
    hifi4 = predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88, fidelity="HiFi4", kernel_rev=MAIN))
    assert hifi4.device_cycles == r.device_cycles and hifi4.breakdown["fidelity"] == "HiFi4"
    # Outside Hi 64 D 128 the per-pair cost scales with the MACs per pair and the result is flagged.
    half = predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=32, D=128, num_cores=88, kernel_rev=MAIN))
    assert half.device_cycles == pytest.approx(r.device_cycles / 2, rel=1e-6)
    # Hi 32 is inside the swept axis now (IXHD), so the geometry no longer needs a flag
    assert not any("indexer_hi_outside" in x or "indexer_d_outside" in x
                   for x in half.low_confidence_reasons)
    assert predict_indexer(IndexerConfig(Sq=2048, T=8192, Hi=64, D=192, num_cores=88, kernel_rev=MAIN)).low_confidence
    with pytest.raises(ValueError):
        predict_indexer(IndexerConfig(Sq=0, T=8192, Hi=64, D=128))


@pytest.mark.unit
def test_indexer_shim_shape_inference_and_pricing():
    with _TtnnDevice() as dev:
        import ttsim.front.ttnn as ttnn
        q = _tt(dev, [1, 64, 2048, 128])
        k = _tt(dev, [1, 1, 8192, 128])
        w = _tt(dev, [1, 64, 2048, 1])
        grid = SimpleNamespace(compute_with_storage_grid_size=(8, 11))
        logits = ttnn.experimental.indexer_score_dsa(q, k, w, chunk_start_idx=6144, program_config=grid,
                                                     compute_kernel_config=ttnn.MathFidelity.HiFi2, kernel_rev="main")
        assert list(logits.shape) == [1, 1, 2048, 8192]
        op = list(dev.ops.values())[-1]
        assert op.optype == "IndexerScoreDSA" and op.inList == [q.name, k.name, w.name]
        assert op.attrs["num_cores"] == 88 and op.attrs["fidelity"] == "HiFi2" and op.attrs["chunk_start_idx"] == 6144
        ps = op.perf_stats
        assert ps["topk_low_confidence"] is False and ps["topk_kernel_rev"] == MAIN
        assert ps["fused_compute_cycles"] == predict_indexer(
            IndexerConfig(Sq=2048, T=8192, Hi=64, D=128, num_cores=88, kernel_rev=MAIN)).device_cycles
        assert _within(ps["fused_compute_cycles"] / GHZ, INDEXER_NS[(2048, 8192)], 2.0)
        assert _components_sum(ps) == ps["fused_compute_cycles"]
        assert ps["instrs"] == {"mac": 64 * 2048 * 8192 * 128}      # device-agnostic fallback
        assert ps["inBytes"] > 0 and ps["outBytes"] == 2048 * 8192 * 2
        # Without a program config the model uses the default grid, and without a revision it says so.
        ttnn.experimental.indexer_score_dsa(q, k, w)
        op = list(dev.ops.values())[-1]
        assert "num_cores" not in op.attrs and f"kernel_rev_defaulted_{MAIN}" in op.perf_stats["topk_low_confidence_reasons"]


@pytest.mark.unit
def test_dsa_chain_prices_end_to_end():
    # indexer_score_dsa -> topk_large_indices -> sparse_sdpa, every op carries a fused cost.
    with _TtnnDevice() as dev:
        import ttsim.front.ttnn as ttnn
        from ttsim.front.ttnn.tensor import Layout
        logits = ttnn.experimental.indexer_score_dsa(_tt(dev, [1, 64, 2048, 128]), _tt(dev, [1, 1, 8192, 128]),
                                                     _tt(dev, [1, 64, 2048, 1]), chunk_start_idx=6144, kernel_rev="main")
        idx = ttnn.experimental.topk_large_indices(ttnn.to_layout(logits, Layout.ROW_MAJOR_LAYOUT),
                                                   k=2048, valid_length=8192, kernel_rev="main")
        out = ttnn.transformer.sparse_sdpa(_tt(dev, [1, 16, 2048, 576]), _tt(dev, [1, 1, 8192, 576]), idx, 512)
        assert list(out.shape) == [1, 16, 2048, 576]
        priced = {op.optype: op.perf_stats.get("fused_compute_cycles") for op in dev.ops.values()}
        for optype in ("IndexerScoreDSA", "TopK", "ScaledDotProductAttention"):
            assert priced[optype] is not None and priced[optype] > 0, optype
        # The DSA chain proportions at this shape: the topk is a small share next to indexer and attention.
        assert priced["TopK"] < priced["IndexerScoreDSA"] and priced["TopK"] < priced["ScaledDotProductAttention"]
        assert _within(priced["TopK"] / GHZ, LI_MAIN_NS[(8192, 2048, 2048)], 1.0)
