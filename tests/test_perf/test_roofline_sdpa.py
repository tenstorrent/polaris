# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SDPA roofline (TEN-4716) driven through the ttsim device cost path."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ttsim.back.device import Device
from ttsim.perf.roofline_sdpa import SdpaConfig, predict, sdpa_perf_stats, ARCH_BH


# Measured per-core MATH cycles on Blackhole P100 (SDPA/verif_data sweeps).
MEASURED_MATH = {
    "baseline": {4096: 875_667, 8192: 3_442_080, 32768: 54_446_938},
    "wan":      {4096: 1_750_702, 32768: 112_393_664},
}

# Measured per-core MATH cycles for FlashMLA prefill (DeepSeek latent family: nh=16, d_qk=576,
# d_v=512, HiFi4, accurate exp, causal), from verif_data/mla_prefill_sweep.csv per-engine counters.
MEASURED_MATH_MLA = {1024: 189_265, 2048: 743_248, 4096: 2_945_197, 8192: 11_724_964}
MEASURED_FPU_MLA = {1024: 180_075, 2048: 707_329, 4096: 2_803_206, 8192: 11_160_368}

# Measured per-core FPU cycles for the head_dim=64 sweep (nh=16, bfp8, HiFi2, causal), from
# verif_data/sweep_hd64.csv (110 active cores). Anchors the head-dim overhead term.
MEASURED_FPU_HD64 = {4096: 195_137, 8192: 768_000, 32768: 12_137_416}

# Measured per-core FPU cycles for the q_chunk=64 sweep (nh=16, hd=128, bfp8, HiFi2, causal), from
# verif_data/sweep_q64.csv (110 active cores). Anchors the q-chunk overhead term.
MEASURED_FPU_Q64 = {1024: 25_456, 2048: 98_723, 4096: 388_692, 8192: 1_542_367}


def _mla_cfg(S):
    return SdpaConfig(S=S, head_dim=576, v_head_dim=512, q_chunk=32, k_chunk=128,
                      num_heads=16, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16",
                      accum_dtype="bfloat16", is_causal=True, exp_approx_mode=False,
                      num_cores=110, arch=ARCH_BH)


def _baseline_cfg(S):
    return SdpaConfig(S=S, num_cores=110, arch=ARCH_BH)


def _wan_cfg(S):
    return SdpaConfig(S=S, num_heads=40, q_chunk=256, k_chunk=256, is_causal=False,
                      input_dtype="bfloat16", accum_dtype="bfloat16",
                      exp_approx_mode=False, num_cores=110, arch=ARCH_BH)


class _MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class _MockSimConfig:
    def __init__(self, freq_mhz=1350, devname="Blackhole"):
        self._freq_mhz = freq_mhz
        self.devname = devname
        self.name = "bh_test_device"
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


class _RealBHSimConfig(_MockSimConfig):
    # BH p100a device params for the whole-pipeline correlation (real DRAM BW + mem clock).
    def mem_frequency(self, units="MHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 448.0        # 448 GB/s at 1 GHz mem clock -> bytes/cycle

    def ramp_penalty(self):
        return 100.0


def _sdpa_op(name, cfg):
    return SimpleNamespace(
        name=name, optype="SDPA", uses_compute_pipe="matrix", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats=sdpa_perf_stats(cfg),
    )


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 4096, 8192, 32768])
def test_fused_compute_cycles_override_is_honored(S):
    device = Device(_MockSimConfig())
    cfg = _baseline_cfg(S)
    op = _sdpa_op(f"sdpa_S{S}", cfg)
    device.execute_op(op)
    assert op.compute_cycles == int(math.ceil(predict(cfg).wall_clock_cycles))
    ipc_path = sum(math.ceil(c / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))
                   for c in op.perf_stats["instrs"].values())
    assert op.compute_cycles != ipc_path


@pytest.mark.unit
def test_arch_gate_falls_back_on_non_bh_device():
    # The roofline is BH-calibrated; off-BH execute_op drops it for the generic estimate, not crash.
    device = Device(_MockSimConfig(devname="Wormhole"))
    op = _sdpa_op("sdpa_wh", _baseline_cfg(4096))
    op.perf_stats["instrs"] = {"mov": 4096}   # sinf leaves a device-agnostic fallback count
    device.execute_op(op)
    assert op.compute_cycles != int(op.perf_stats["fused_compute_cycles"])


@pytest.mark.unit
def test_arch_gate_falls_back_off_bh():
    # Decode is BH-calibrated, so off-BH the gate must drop the roofline cost and use the generic
    # instr estimate (not crash, and not book the BH cost).
    from ttsim.perf.roofline_sdpa import decode_perf_stats
    device = Device(_MockSimConfig(devname="Wormhole"))
    ps = decode_perf_stats([1, 32, 1, 128], [1, 8, 4096, 128], attrs={"element_size": 2})
    ps["instrs"] = {"mov": 128}   # sinf leaves a device-agnostic fallback count
    op = SimpleNamespace(
        name="dec_wh", optype="SDPA", uses_compute_pipe="matrix", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={}, compute_cycles=0,
        compute_is_lower_bound=False,
        mem_rd_cycles=0, mem_wr_cycles=0, mem_rd_cycles_fractional=0.0,
        mem_wr_cycles_fractional=0.0, perf_stats=ps)
    device.execute_op(op)
    assert op.compute_cycles != int(ps["fused_compute_cycles"])   # generic path, not the BH cost


@pytest.mark.unit
def test_decode_v_head_dim_from_v_cache_shape():
    # FlashMLA decode: v_head_dim comes from the v_cache tensor, with attrs as the only fallback.
    from ttsim.perf.roofline_sdpa import decode_config_from_shapes
    cfg = decode_config_from_shapes([1, 32, 1, 576], [1, 1, 4096, 576],
                                    v_shape=[1, 1, 4096, 512], attrs={})
    assert cfg["v_head_dim"] == 512
    fallback = decode_config_from_shapes([1, 32, 1, 576], [1, 1, 4096, 576],
                                         attrs={"head_dim_v": 512})
    assert fallback["v_head_dim"] == 512


@pytest.mark.unit
def test_wall_factor_cross_only_when_kv_seq_differs():
    # kv_seq == S is plain self-attention, not cross, so it keeps the prefill wall factor.
    from ttsim.perf.roofline_sdpa import _wall_factor, WALL_FACTOR
    r = SimpleNamespace(regime="prefill", is_mla=False)
    same = SdpaConfig(S=4096, kv_seq=4096, is_causal=True, is_sparse=False)
    cross = SdpaConfig(S=4096, kv_seq=2048, is_causal=True, is_sparse=False)
    assert _wall_factor(same, r) == WALL_FACTOR["prefill_causal"]
    assert _wall_factor(cross, r) == WALL_FACTOR["cross"]


@pytest.mark.unit
def test_op_cost_is_wall_clock_with_floor_in_breakdown():
    # The op cost is now the full wall-clock estimate (not a floor); the compute floor stays in the
    # breakdown for reference.
    cfg = _baseline_cfg(4096)
    ps = sdpa_perf_stats(cfg)
    assert ps["sdpa_compute_is_floor"] is False and ps["sdpa_calibrated_arch"] == "Blackhole"
    assert ps["fused_compute_cycles"] == predict(cfg).wall_clock_cycles
    assert ps["sdpa_cycle_breakdown"]["compute_floor"] == predict(cfg).compute_latency_cycles
    assert ps["fused_compute_cycles"] > ps["sdpa_cycle_breakdown"]["compute_floor"]


@pytest.mark.unit
def test_floor_flag_is_accepted_by_opstats_schema():
    # Regression: execute_op writes compute_is_lower_bound into op.exec_stats, which hlmstats
    # merges into rows validated by TTSimHLWlDevRunOperatorPerfStats (extra='forbid'). The field
    # must be declared there or dump_stats raises ValidationError for every op.
    from ttsim.config.validators import TTSimHLWlDevRunOperatorPerfStats as Row
    assert "compute_is_lower_bound" in Row.model_fields
    row = dict(pipe="matrix", precision="bfp8", opnum=0, opname="sdpa", is_input_node=False,
               is_output_node=False, optype="ScaledDotProductAttention", op_rpt_count=1, attrs={},
               inList=[], outList=[], input_tensors="[]", output_tensors="[]", weight_tensors="[]",
               domain="ttnn", opclass="COMMON", removed=False, fused=False, fused_with_op="NA",
               inElems=0, outElems=0, inBytes=1, outBytes=1, instrs={}, inParamCount=0, inActCount=0,
               outActCount=0, instr_count=0, compute_cycles=1.0, mem_rd_cycles=0.0, mem_wr_cycles=0.0,
               ramp_penalty=100.0, rsrc_bnck="COMP", ideal_cycles=1.0, ideal_msecs=1.0, cycles=1.0,
               matrix_cycles=1.0, vector_cycles=0.0, msecs=1.0, matrix_pipe_util=0.5,
               vector_pipe_util=0.0, mem_rd_util=0.0, mem_wr_util=0.0, memory_traffic=0.0,
               mem_util=0.0, uses_perf_lookup=False)
    assert Row(**row, compute_is_lower_bound=True).compute_is_lower_bound is True


@pytest.mark.unit
def test_mla_config_is_flagged_low_confidence():
    # Asymmetric v_head_dim (FlashMLA) is emitted low-confidence: it is calibrated on a single
    # config family, so off-family MLA shapes should be treated as lower confidence.
    assert sdpa_perf_stats(_mla_cfg(4096))["sdpa_mla_low_confidence"] is True
    # Symmetric SDPA stays high-confidence.
    assert sdpa_perf_stats(_baseline_cfg(4096))["sdpa_mla_low_confidence"] is False


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 2048, 4096, 8192])
def test_mla_calibration_tracks_measured_math(S):
    # The MLA-specific constants (fpu_overhead 0.031, sfpu_overhead/iter 96.8, overlap 0) must
    # keep predicted MATH within ~5% of the measured per-engine counters.
    pred = predict(_mla_cfg(S)).math_active_cycles
    meas = MEASURED_MATH_MLA[S]
    rel = abs(pred - meas) / meas
    assert rel <= 0.05, f"MLA S={S}: pred={pred} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 2048, 4096, 8192])
def test_mla_calibration_tracks_measured_fpu(S):
    # FlashMLA per-engine counters exist now (mla_prefill_sweep.csv), so the FPU term is validated
    # against silicon, not just projected: keep predicted FPU within ~5% of measured.
    pred = predict(_mla_cfg(S)).fpu_cycles
    meas = MEASURED_FPU_MLA[S]
    rel = abs(pred - meas) / meas
    assert rel <= 0.05, f"MLA FPU S={S}: pred={pred} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
@pytest.mark.parametrize("S", [4096, 8192, 32768])
def test_head_dim_overhead_tracks_measured_hd64_fpu(S):
    # head_dim=64 was a cold -9% FPU corner; the head-dim overhead term brings it within ~3%.
    r = predict(SdpaConfig(S=S, head_dim=64, num_heads=16, num_cores=110, arch=ARCH_BH))
    meas = MEASURED_FPU_HD64[S]
    rel = abs(r.fpu_cycles - meas) / meas
    assert rel <= 0.03, f"hd64 FPU S={S}: pred={r.fpu_cycles} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 2048, 4096, 8192])
def test_q_chunk_overhead_tracks_measured_q64_fpu(S):
    # q_chunk=64 was a -10% FPU corner (overhead is a bigger share of a smaller chunk); the
    # q-chunk overhead term brings it within ~3%.
    r = predict(SdpaConfig(S=S, head_dim=128, num_heads=16, q_chunk=64, k_chunk=64,
                           num_cores=110, fidelity="HiFi2", is_causal=True,
                           input_dtype="bfp8_b", arch=ARCH_BH))
    meas = MEASURED_FPU_Q64[S]
    rel = abs(r.fpu_cycles - meas) / meas
    assert rel <= 0.03, f"q64 FPU S={S}: pred={r.fpu_cycles} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
def test_head_dim_overhead_leaves_128_unchanged_and_grows_for_small_head_dim():
    # The term is zero at the calibration head_dim (128) and larger for smaller head_dims.
    def frac(hd):
        r = predict(SdpaConfig(S=4096, head_dim=hd, num_heads=16, num_cores=110, arch=ARCH_BH))
        return r.fpu_overhead_cycles / r.fpu_matmul_cycles
    assert abs(frac(128) - ARCH_BH.fpu_overhead_frac) < 1e-3   # rounding only
    assert frac(64) > frac(128)


@pytest.mark.unit
def test_mla_uses_zero_overlap_and_low_fpu_overhead():
    # MLA path must not apply the standard-SDPA overlap/overhead (would over-predict ~13%).
    r = predict(_mla_cfg(4096))
    # overlap 0 -> math_active is the plain FPU+SFPU sum (no overlap subtraction).
    assert r.math_active_cycles == r.fpu_cycles + r.sfpu_cycles
    # near-zero FPU template overhead (< 5% of matmul), vs 13.2% for standard SDPA.
    assert r.fpu_overhead_cycles < 0.05 * r.fpu_matmul_cycles


@pytest.mark.unit
def test_generic_instrs_path_unchanged_without_override():
    device = Device(_MockSimConfig())
    op = SimpleNamespace(
        name="plain_mm", optype="MatMul", uses_compute_pipe="matrix", precision="fp32",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats={"inBytes": 4000, "outBytes": 4000, "instrs": {"mac": 4096}},
    )
    device.execute_op(op)
    assert op.compute_cycles == math.ceil(4096 / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))


@pytest.mark.unit
@pytest.mark.parametrize("shape,S", [("baseline", 4096), ("baseline", 8192),
                                     ("baseline", 32768), ("wan", 4096), ("wan", 32768)])
def test_roofline_tracks_measured_math(shape, S):
    cfg = _baseline_cfg(S) if shape == "baseline" else _wan_cfg(S)
    pred = predict(cfg).math_active_cycles
    meas = MEASURED_MATH[shape][S]
    rel = abs(pred - meas) / meas
    # q_chunk-dependent overlap tightened the q256 (wan) corner from ~6% to ~4%.
    assert rel <= 0.06, f"{shape} S={S}: pred={pred} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
def test_overlap_grows_with_q_chunk():
    # Overlap rises with q_chunk (more independent tiles to interleave) and saturates below 1;
    # non-causal overlaps more than causal; MLA does not overlap.
    from ttsim.perf.roofline_sdpa import _overlap_frac, OVERLAP_FRAC_MAX
    assert _overlap_frac(8, True, False) > _overlap_frac(4, True, False)     # q256 > q128
    assert _overlap_frac(4, False, False) > _overlap_frac(4, True, False)    # non-causal > causal
    assert _overlap_frac(64, True, False) <= OVERLAP_FRAC_MAX                 # capped below 1
    assert _overlap_frac(4, True, True) == 0.0                               # MLA


@pytest.mark.unit
def test_sdpa_config_from_shapes_gqa():
    from ttsim.perf.roofline_sdpa import sdpa_config_from_shapes
    # Llama GQA: q[1,32,4096,128], k/v[1,8,4096,128], causal
    cfg = sdpa_config_from_shapes([1, 32, 4096, 128], [1, 8, 4096, 128], [1, 8, 4096, 128],
                                  {"is_causal": True, "element_size": 1})
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.S) == (32, 8, 128, 4096)
    assert cfg.input_dtype == "bfp8_b" and cfg.is_causal
    ps = sdpa_perf_stats(cfg)
    assert ps["fused_compute_cycles"] > 0 and ps["inBytes"] > 0


@pytest.mark.unit
def test_sdpa_sinf_routes_prefill_and_decode_variants():
    from ttsim.ops.desc.ttsim_layout import sdpa_sinf
    def T(shape):
        return SimpleNamespace(shape=list(shape), dtype="bfloat16")
    # prefill: 3 inputs -> compute roofline (wall-clock cost)
    q, k, v = T([1, 32, 4096, 128]), T([1, 8, 4096, 128]), T([1, 8, 4096, 128])
    out = SimpleNamespace(shape=None, dtype=None)
    op = SimpleNamespace(attrs={"is_causal": True, "element_size": 1}, perf_stats=None)
    sdpa_sinf([q, k, v], [out], op)
    assert op.perf_stats.get("fused_compute_cycles", 0) > 0 and op.perf_stats["inBytes"] > 0
    assert op.perf_stats["sdpa_regime"] == "prefill"
    # decode: 4 inputs (q, k_cache, v_cache, cur_pos) -> MEMORY-bound KV-stream roofline
    op2 = SimpleNamespace(attrs={"element_size": 2}, perf_stats=None)
    sdpa_sinf([T([1, 32, 1, 128]), k, v, T([1])], [SimpleNamespace(shape=None, dtype=None)], op2)
    assert op2.perf_stats["sdpa_regime"] == "decode"
    assert op2.perf_stats["sdpa_cycle_breakdown"]["is_memory_bound"] is True
    assert op2.perf_stats["sdpa_compute_is_floor"] is False  # memory bound, not a compute floor
    # KV cache (8 heads x 4096 x 128 x2) dominates the tiny single-token compute:
    assert op2.perf_stats["inBytes"] > 10_000_000


@pytest.mark.unit
def test_decode_is_memory_bound_and_scales_with_kv_cache():
    from ttsim.perf.roofline_sdpa import predict_decode
    # KV-stream bytes must dominate the single-token compute, and grow linearly with cache_len.
    r1 = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    r2 = predict_decode(cache_len=8192, num_q_heads=32, num_kv_heads=8, head_dim=128)
    assert r1.is_memory_bound and r1.low_confidence and r1.regime == "decode"
    assert r2.dram_in_bytes == pytest.approx(2 * r1.dram_in_bytes, rel=0.02)  # 2x cache -> ~2x bytes
    # memory volume vastly exceeds the single-token compute (the whole point of P1)
    assert r1.dram_in_bytes > 100 * r1.math_active_cycles


@pytest.mark.unit
def test_decode_sliding_window_caps_kv_stream():
    from ttsim.perf.roofline_sdpa import predict_decode
    # Windowed decode (Gemma/Mistral SWA) attends only the last `window` keys -> KV stream (and
    # the memory bound) caps at the window, not the full cache.
    full = predict_decode(cache_len=32768, num_q_heads=32, num_kv_heads=8, head_dim=128)
    win = predict_decode(cache_len=32768, num_q_heads=32, num_kv_heads=8, head_dim=128,
                         sliding_window=4096)
    assert win.dram_in_bytes < full.dram_in_bytes
    # capped at the window: same bytes as a full cache of the window length
    capped = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    assert win.dram_in_bytes == capped.dram_in_bytes


@pytest.mark.unit
@pytest.mark.parametrize("cache,meas_us", [(1024, 113.7), (8192, 795.7), (32768, 3129.5)])
def test_decode_latency_includes_fixed_overhead(cache, meas_us):
    # Decode latency = fixed dispatch/ramp overhead + KV stream at the measured BW. Modelling the
    # fixed term brings small-context decode within ~8% (was ~17% low without it). b=8,nkv=8 config.
    from ttsim.perf.roofline_sdpa import predict_decode
    r = predict_decode(cache_len=cache, num_q_heads=32, num_kv_heads=8, head_dim=128, batch=8)
    us = r.compute_latency_cycles / 1350.0
    assert abs(us - meas_us) / meas_us <= 0.08, f"cache={cache}: {us:.1f}us vs {meas_us}us"


@pytest.mark.unit
def test_decode_kv_bytes_scale_with_kv_heads_not_query_heads():
    from ttsim.perf.roofline_sdpa import predict_decode
    # GQA: KV traffic depends on num_kv_heads, not num_q_heads (MQA reads the least).
    mha = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=32, head_dim=128)
    gqa = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=8, head_dim=128)
    mqa = predict_decode(cache_len=4096, num_q_heads=32, num_kv_heads=1, head_dim=128)
    assert mha.dram_in_bytes > gqa.dram_in_bytes > mqa.dram_in_bytes


@pytest.mark.unit
def test_mla_decode_reuses_k_as_v_cutting_dram():
    from ttsim.perf.roofline_sdpa import predict_decode
    # MLA decode reuses K as V (V is an L1 transpose), so its KV DRAM read excludes the V tensor.
    mla = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576,
                         v_head_dim=512, fidelity="HiFi4")
    assert mla.is_mla and mla.is_memory_bound
    # symmetric decode of the same K width would also read V tiles; MLA must read strictly less
    sym = predict_decode(cache_len=4096, num_q_heads=16, num_kv_heads=1, head_dim=576)
    assert mla.dram_in_bytes < sym.dram_in_bytes


@pytest.mark.unit
def test_sliding_window_saturates_not_unbounded():
    # A sliding window must SATURATE at W (not grow like the (K+1)/2 causal diagonal). This is the
    # Gemma/Mistral SWA overcount fix: at S=32k the window K_eff must be far below full-causal.
    full = predict(SdpaConfig(S=32768, num_cores=110, arch=ARCH_BH))
    win = predict(SdpaConfig(S=32768, sliding_window=4096, num_cores=110, arch=ARCH_BH))
    assert win.k_eff < 0.3 * full.k_eff
    assert win.fpu_matmul_cycles < full.fpu_matmul_cycles
    assert win.regime == "windowed" and win.low_confidence
    # Doubling S at fixed window leaves per-q-chunk K_eff ~unchanged (bounded by the window).
    win2 = predict(SdpaConfig(S=65536, sliding_window=4096, num_cores=110, arch=ARCH_BH))
    assert win2.k_eff == pytest.approx(win.k_eff, rel=0.05)


@pytest.mark.unit
def test_cross_attention_kv_seq_differs_from_q_seq():
    # Cross-attention (kv_seq != q_seq): K count and K/V DRAM follow kv_seq, Q/output follow S.
    cross = predict(SdpaConfig(S=1024, kv_seq=4096, num_heads=16, is_causal=False,
                               num_cores=110, arch=ARCH_BH))
    assert cross.k_chunks_per_q == 4096 // 128       # K from kv_seq, not S
    # same shapes but kv_seq==S would read less K/V DRAM
    square = predict(SdpaConfig(S=1024, num_heads=16, is_causal=False, num_cores=110, arch=ARCH_BH))
    assert cross.dram_in_bytes > square.dram_in_bytes


@pytest.mark.unit
@pytest.mark.parametrize("S,meas", [(8192, 1_668_319), (16384, 6_672_750)])
def test_masked_prefill_tracks_measured_math(S, meas):
    # Card-measured masked non-causal MATH (density-independent). The causal-equivalent model must
    # track it within ~2% (was +101% with the old compute-then-mask + add model).
    pred = predict(SdpaConfig(S=S, is_causal=False, has_attn_mask=True, num_heads=16, head_dim=128,
                              q_chunk=128, k_chunk=128, input_dtype="bfp8_b", accum_dtype="bfloat16",
                              fidelity="HiFi2", exp_approx_mode=True, num_cores=110, arch=ARCH_BH)).math_active_cycles
    assert abs(pred - meas) / meas <= 0.03, f"S={S}: pred={pred} meas={meas}"


@pytest.mark.unit
def test_masked_prefill_is_causal_equivalent_and_sink_adds_sfpu():
    # Card-measured: the dense-mask prefill path costs ~causal (density-independent), not
    # non-causal-full + a mask-add pass.
    base_nc = predict(SdpaConfig(S=4096, is_causal=False, num_cores=110, arch=ARCH_BH))
    base_c = predict(SdpaConfig(S=4096, is_causal=True, num_cores=110, arch=ARCH_BH))
    mask = predict(SdpaConfig(S=4096, is_causal=False, has_attn_mask=True, num_cores=110, arch=ARCH_BH))
    assert mask.math_active_cycles < base_nc.math_active_cycles   # masked does LESS, not more
    assert mask.k_eff == base_c.k_eff                            # causal-equivalent K_eff
    assert mask.fpu_matmul_cycles == base_c.fpu_matmul_cycles    # no mask-add pass
    assert mask.regime == "masked"
    sink = predict(SdpaConfig(S=4096, is_causal=False, attention_sink=True, num_cores=110, arch=ARCH_BH))
    assert sink.sfpu_extra_cycles > 0              # attention sink adds a softmax-normalization pass
    assert mask.low_confidence and sink.low_confidence


@pytest.mark.unit
def test_chunked_prefill_adds_dense_prefix_and_scatter_derate():
    # Chunked/paged prefill: this Q chunk attends the full DENSE prefix (rectangular, no causal
    # halving) plus the causal ramp within the chunk. K_eff = prefix/k_chunk + (q_nc+1)/2.
    ck = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128, k_chunk=128,
                            num_cores=110, arch=ARCH_BH))
    assert ck.k_eff == pytest.approx(4096 / 128 + (2048 // 128 + 1) / 2)  # 40.5
    assert ck.regime == "chunked" and ck.low_confidence
    # a plain first chunk (no prefix) visits far fewer K-chunks
    plain = predict(SdpaConfig(S=2048, q_chunk=128, k_chunk=128, num_cores=110, arch=ARCH_BH))
    assert ck.k_eff > plain.k_eff
    # paged scatter derate inflates effective K/V DRAM bytes vs a non-scattered read
    no_derate = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128,
                                   k_chunk=128, dram_scatter_derate=1.0, num_cores=110, arch=ARCH_BH))
    derated = predict(SdpaConfig(S=2048, kv_seq=6144, chunk_start_idx=4096, q_chunk=128,
                                 k_chunk=128, dram_scatter_derate=1.15, num_cores=110, arch=ARCH_BH))
    assert derated.dram_in_bytes > no_derate.dram_in_bytes


@pytest.mark.unit
def test_router_dispatches_chunked_prefill():
    from ttsim.ops.desc.ttsim_layout import _infer_sdpa_variant
    assert _infer_sdpa_variant([0, 1, 2], {}) == "prefill"
    assert _infer_sdpa_variant([0, 1, 2, 3], {}) == "decode"
    assert _infer_sdpa_variant([0, 1, 2, 3], {"chunk_start_idx": 4096}) == "chunked"
    assert _infer_sdpa_variant([0, 1, 2, 3], {"chunk_start_idx": 0}) == "decode"  # first chunk == decode-arity


@pytest.mark.unit
def test_batch_scales_prefill_work_and_dram():
    # Batch>1 prefill: per-core work and whole-op DRAM scale linearly with batch (card-validated
    # -3.6/-3.7/-4.5% at b=1/2/4). Was a silent under-count before the fix.
    b1 = predict(SdpaConfig(S=2048, num_heads=16, batch=1, num_cores=110, arch=ARCH_BH))
    b4 = predict(SdpaConfig(S=2048, num_heads=16, batch=4, num_cores=110, arch=ARCH_BH))
    assert b4.fpu_matmul_cycles == pytest.approx(4 * b1.fpu_matmul_cycles, rel=1e-3)  # ~ exact (rounding)
    assert b4.dram_in_bytes == 4 * b1.dram_in_bytes


@pytest.mark.unit
@pytest.mark.parametrize("hd,meas", [(64, 260_344), (96, 345_425), (160, 510_247), (256, 775_665)])
def test_head_dim_variety_tracks_measured(hd, meas):
    # Tile-aligned head_dims (vision/VAE use 96/160/256) must track measured MATH within ~7%.
    pred = predict(SdpaConfig(S=4096, head_dim=hd, num_heads=16, q_chunk=128, k_chunk=128,
                              is_causal=True, input_dtype="bfp8_b", accum_dtype="bfloat16",
                              fidelity="HiFi2", exp_approx_mode=True, num_cores=110, arch=ARCH_BH)).math_active_cycles
    assert abs(pred - meas) / meas <= 0.07, f"hd={hd}: pred={pred} meas={meas}"


@pytest.mark.unit
@pytest.mark.parametrize("topk,meas", [(512, 780_574), (1024, 1_556_380), (2048, 3_107_986)])
def test_sparse_mla_tracks_measured_math(topk, meas):
    # Sparse-MLA: each query attends TOPK selected latent KV (kv_seq=TOPK), asymmetric d (576/512),
    # shared latent, plus a calibrated scattered-KV gather uplift. Card-validated ~0% across TOPK.
    pred = predict(SdpaConfig(S=2048, kv_seq=topk, head_dim=576, v_head_dim=512, q_chunk=128,
                              k_chunk=128, num_heads=32, num_kv_heads=1, is_sparse=True, is_causal=False,
                              input_dtype="bfloat16", accum_dtype="bfloat16", fidelity="HiFi4",
                              exp_approx_mode=False, num_cores=110, arch=ARCH_BH))
    assert pred.regime == "sparse" and pred.low_confidence
    assert abs(pred.math_active_cycles - meas) / meas <= 0.05, f"TOPK={topk}"


def _polaris_device_us(perf_stats):
    # Projected device latency through the real Device.execute_op + get_exec_stats projection
    # (ideal = max(compute, mem) + ramp), with BH device params.
    device = Device(_RealBHSimConfig())
    op = _sdpa_op("corr", _baseline_cfg(4096))
    op.perf_stats = perf_stats
    device.execute_op(op)
    ideal = math.ceil(max(op.compute_cycles, op.mem_rd_cycles + op.mem_wr_cycles) + 100.0)
    return ideal / 1350.0


@pytest.mark.unit
@pytest.mark.parametrize("perf,meas", [
    (sdpa_perf_stats(SdpaConfig(S=4096, num_heads=32, num_kv_heads=8, head_dim=128, is_causal=True,
                                input_dtype="bfp8_b", fidelity="HiFi2", num_cores=110, arch=ARCH_BH)), 1740),
    (sdpa_perf_stats(SdpaConfig(S=1024, head_dim=576, v_head_dim=512, q_chunk=32, k_chunk=128,
                                num_heads=16, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16",
                                accum_dtype="bfloat16", is_causal=True, exp_approx_mode=False,
                                num_cores=110, arch=ARCH_BH)), 1619),
])
def test_whole_pipeline_correlation_vs_measured(perf, meas):
    # End-to-end: the projected device latency through the real Polaris pipeline tracks measured
    # device wall-clock within ~15% for each single-chip variant.
    us = _polaris_device_us(perf)
    assert abs(us - meas) / meas <= 0.15, f"projected {us:.0f}us vs measured {meas}us"


@pytest.mark.unit
def test_whole_pipeline_correlation_decode():
    from ttsim.perf.roofline_sdpa import predict_decode
    ps = predict_decode(cache_len=8192, num_q_heads=32, num_kv_heads=8, head_dim=128,
                        batch=8).to_polaris_op_perf_stats()
    assert abs(_polaris_device_us(ps) - 795.7) / 795.7 <= 0.15


@pytest.mark.unit
def test_wall_clock_per_regime_vs_measured():
    # Per-regime wall_factor projects device wall-clock across the harder regimes (measured on BH).
    mla = predict(SdpaConfig(S=1024, head_dim=576, v_head_dim=512, q_chunk=32, k_chunk=128,
                             num_heads=16, num_kv_heads=1, fidelity="HiFi4", input_dtype="bfloat16",
                             accum_dtype="bfloat16", is_causal=True, exp_approx_mode=False,
                             num_cores=110, arch=ARCH_BH))
    assert abs(mla.wall_clock_cycles / 1350.0 - 1619) / 1619 <= 0.15   # MLA ~11x floor
    sparse = predict(SdpaConfig(S=2048, kv_seq=1024, head_dim=576, v_head_dim=512, num_heads=32,
                                num_kv_heads=1, is_sparse=True, is_causal=False, input_dtype="bfloat16",
                                accum_dtype="bfloat16", fidelity="HiFi4", exp_approx_mode=False,
                                num_cores=110, arch=ARCH_BH))
    assert abs(sparse.wall_clock_cycles / 1350.0 - 5586) / 5586 <= 0.15


@pytest.mark.unit
@pytest.mark.parametrize("S,dev_us", [(2048, 402), (4096, 1740), (8192, 6380), (16384, 25900)])
def test_wall_clock_estimate_tracks_device_latency(S, dev_us):
    # The wall-clock estimate (compute floor x per-regime latency multiple) tracks measured DEVICE
    # wall-clock within ~15%; the floor stays a strict lower bound below it.
    r = predict(SdpaConfig(S=S, num_heads=32, num_kv_heads=8, head_dim=128, is_causal=True,
                           input_dtype="bfp8_b", fidelity="HiFi2", num_cores=110, arch=ARCH_BH))
    assert r.wall_clock_cycles > r.compute_latency_cycles     # wall-clock is above the floor
    us = r.wall_clock_cycles / 1350.0
    assert abs(us - dev_us) / dev_us <= 0.15, f"S={S}: wall={us:.0f}us dev={dev_us}us"


@pytest.mark.unit
def test_multichip_ring_is_out_of_scope_not_handled():
    # This roofline is single-chip only. A ring/multichip SDPA variant must NOT be given a
    # single-chip cost -- it has no front-end handler and falls through to passthrough.
    from ttsim.ops.desc.ttsim_layout import _SDPA_FRONTENDS
    assert "ring_distributed" not in _SDPA_FRONTENDS
    assert "ring_joint" not in _SDPA_FRONTENDS
    assert set(_SDPA_FRONTENDS) == {"prefill", "decode", "chunked"}


@pytest.mark.unit
def test_perf_stats_has_all_keys_downstream_reads():
    # hlmstats / device stats read these unconditionally; missing any is a KeyError at runtime.
    ps = sdpa_perf_stats(_baseline_cfg(4096))
    required = {"inBytes", "outBytes", "inElems", "outElems", "inParamCount",
                "inActCount", "outActCount", "instrs", "fused_compute_cycles"}
    assert required.issubset(ps), f"missing: {required - set(ps)}"
    assert ps["fused_compute_cycles"] > 0 and ps["inBytes"] > 0
    assert ps["inElems"] == 0  # sentinel so the bytes-per-element precision check is skipped


@pytest.mark.unit
def test_predict_rejects_bad_dtype_and_fidelity_but_pads_shapes():
    # dtype/fidelity must be known. Non-chunk-divisible S is PADDED up (matching the kernel), not
    # rejected -- so a 32-aligned-but-not-128-divisible prefill stays on the roofline instead of
    # silently falling to the passthrough mov-cost. head_dim need not be a multiple of 32 either.
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, input_dtype="fp8_e4m3", num_cores=110, arch=ARCH_BH))
    with pytest.raises(ValueError):
        predict(SdpaConfig(S=4096, fidelity="HiFi9", num_cores=110, arch=ARCH_BH))
    # S=5000 (not a multiple of q_chunk=128) is padded, not rejected:
    import math
    padded = predict(SdpaConfig(S=5000, num_cores=110, arch=ARCH_BH))
    assert padded.compute_latency_cycles > 0
    assert padded.q_chunks_per_core == (math.ceil(5000 / 128) * 32) / 110


@pytest.mark.unit
@pytest.mark.parametrize("hd", [18, 72, 96, 256])
def test_non_tile_aligned_head_dim_is_padded_not_rejected(hd):
    # Vision/VAE head_dims that are not multiples of 32 are rounded up to whole tiles (matching
    # the kernel's tile padding) instead of being rejected. Cost must scale with ceil(hd/32).
    import math
    r = predict(SdpaConfig(S=4096, head_dim=hd, num_heads=16, num_cores=110, arch=ARCH_BH))
    assert r.fpu_matmul_cycles > 0
    ref = predict(SdpaConfig(S=4096, head_dim=math.ceil(hd / 32) * 32, num_heads=16,
                             num_cores=110, arch=ARCH_BH))
    assert r.fpu_matmul_cycles == ref.fpu_matmul_cycles  # padded to the same tile count


@pytest.mark.unit
def test_v_head_dim_default_is_symmetric():
    # v_head_dim=0 must reproduce the symmetric head_dim result exactly.
    sym = predict(SdpaConfig(S=4096, head_dim=128, num_cores=110, arch=ARCH_BH))
    explicit = predict(SdpaConfig(S=4096, head_dim=128, v_head_dim=128, num_cores=110, arch=ARCH_BH))
    assert sym.fpu_matmul_cycles == explicit.fpu_matmul_cycles
    assert sym.compute_latency_cycles == explicit.compute_latency_cycles


@pytest.mark.unit
def test_asymmetric_v_head_dim_scales_pv_matmul():
    # MLA-style: v_head_dim > head_dim raises the P*V matmul, so total matmul cycles grow.
    base = predict(SdpaConfig(S=4096, head_dim=192, num_cores=110, arch=ARCH_BH))
    mla = predict(SdpaConfig(S=4096, head_dim=192, v_head_dim=512, num_cores=110, arch=ARCH_BH))
    assert mla.fpu_matmul_cycles > base.fpu_matmul_cycles
    # matmul cycles scale as (dct_qk + dct_v): (6+16) vs (6+6)
    assert mla.fpu_matmul_cycles == round(base.fpu_matmul_cycles * (6 + 16) / (6 + 6))
