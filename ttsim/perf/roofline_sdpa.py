# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compute-latency roofline for single-chip SDPA on Blackhole (TEN-4716).

Mechanistic matmul/SFPU/L1 terms plus a few hardware-calibrated constants.
predict(cfg) returns a RooflineResult; sdpa_perf_stats(cfg) emits the ttsim
perf_stats for a fused SDPA op. See the TEN-4716 Confluence page for the
derivation and validation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

TILE_HW = 32
BYTES_PER_TILE = {"bfp8_b": 1088, "bfloat16": 2048, "float32": 4096}

# FPU/SFPU concurrency, as a fraction of the smaller engine. Calibrated per regime from
# the measured MATH union; overlap is genuinely config-dependent (measured 0.45-0.86), so a
# single global value biased the causal case low. These two-way values leave a residual
# per-config bias (q_chunk=256 ~+6%, bf16 ~-4%) reported honestly in the validation.
OVERLAP_FRAC_CAUSAL = 0.579
OVERLAP_FRAC_NONCAUSAL = 0.755


@dataclass
class ArchConfig:
    """Per-Tensix-core hardware geometry. Defaults are BH P100."""
    name: str = "BH"
    fpu_m_per_cycle: int = 8       # llk_math_matmul.h
    fpu_k_per_cycle: int = 16
    fpu_n_per_cycle: int = 16
    cycles_per_tile_mac: Dict[str, float] = field(
        default_factory=lambda: {"LoFi": 16.0, "HiFi2": 32.0, "HiFi3": 48.0, "HiFi4": 64.0}
    )
    fpu_overhead_frac: float = 0.132          # LLK template cycles / matmul cycles (calibrated)
    sfpu_lanes: int = 32
    exp_tile_cycles: float = 88.2             # approx exp per tile (calibrated)
    exp_tile_cycles_accurate: float = 77.9    # accurate exp per tile (calibrated)
    exp_fc_cycles: float = 17.0               # first-column exp on k>0 rescale
    reduce_max_per_tile: float = 30.0
    recip_cycles: float = 17.0
    sfpu_overhead_per_inner_iter: float = 417.0
    unpacker_bw_bytes_per_cycle: float = 80.0
    packer_bw_bytes_per_cycle: float = 64.0
    idle_per_inner_iter: float = 0.0
    init_overhead_cycles: float = 30000.0
    clock_ghz: float = 1.35

    def cpt(self, fidelity: str) -> float:
        try:
            return self.cycles_per_tile_mac[fidelity]
        except KeyError:
            raise ValueError(
                f"unsupported fidelity {fidelity!r}; supported: {sorted(self.cycles_per_tile_mac)}"
            ) from None


@dataclass
class SdpaConfig:
    S: int
    head_dim: int = 128       # QK head dim (Q*K^T contraction)
    v_head_dim: int = 0       # P*V output dim; 0 means same as head_dim (symmetric SDPA)
    q_chunk: int = 128
    k_chunk: int = 128
    num_heads: int = 32       # query heads
    num_kv_heads: int = 0     # KV heads; 0 means same as num_heads (MHA). GQA/MQA read K/V fewer times.
    num_cores: int = 110
    fidelity: str = "HiFi2"
    input_dtype: str = "bfp8_b"
    accum_dtype: str = "bfloat16"
    is_causal: bool = True
    has_attn_mask: bool = False
    exp_approx_mode: bool = True
    arch: ArchConfig = field(default_factory=ArchConfig)


@dataclass
class RooflineResult:
    label: str = ""
    arch_name: str = ""
    q_chunks_per_core: float = 0.0
    k_chunks_per_q: float = 0.0
    k_eff: float = 0.0
    inner_iters: float = 0.0
    fpu_matmul_cycles: int = 0
    fpu_overhead_cycles: int = 0
    fpu_cycles: int = 0
    sfpu_exp_cycles: int = 0
    sfpu_reduce_cycles: int = 0
    sfpu_recip_cycles: int = 0
    sfpu_overhead_cycles: int = 0
    sfpu_cycles: int = 0
    math_active_cycles: int = 0        # FPU/SFPU execution union (calibrated overlap) = the compute estimate
    serial_sum_cycles: int = 0         # fpu+sfpu, the zero-overlap upper bracket (not the prediction)
    math_idle_cycles: int = 0
    init_overhead_cycles: int = 0
    compute_latency_cycles: int = 0
    unpack_bytes_total: int = 0        # per-core L1 unpacker bytes (drives the on-chip BW floor only)
    pack_bytes_total: int = 0
    unpack_min_cycles: int = 0
    pack_min_cycles: int = 0
    dram_in_bytes: int = 0             # whole-op DRAM read traffic (Q + K/V scaled by num_kv_heads)
    dram_out_bytes: int = 0            # whole-op DRAM write traffic (output)

    def to_polaris_op_perf_stats(self) -> Dict:
        """perf_stats for a fused SDPA op consumed by ttsim Device.execute_op.

        CONTRACT: fused_compute_cycles is the COMPUTE-LATENCY FLOOR (the FPU/SFPU math
        union), not full wall-clock. On BH it is ~44-65% of measured kernel wall-clock;
        the front-end/dispatch idle that separates the two is the deferred data-movement
        half of TEN-4716 and is NOT included here. Treat Polaris SDPA latency from this
        provider as a lower bound until that half lands.

        inBytes/outBytes are whole-op DRAM traffic (K/V counted per KV head, so GQA/MQA are
        not overcounted); the per-core L1 unpacker bytes stay internal to the on-chip BW
        floor. inElems/outElems are 0 so hlmstats skips the bytes-per-element check; SDPA
        carries no weights so param/act counts are 0. instrs is left empty: ttsim consumes
        it as instruction COUNTS (divided by IPC / summed for instr profiles), so cycle
        values must not live there. The per-engine cycle breakdown is exposed separately
        under sdpa_cycle_breakdown for reporting; fused_compute_cycles sets the op cost.
        """
        return {
            "inBytes": self.dram_in_bytes,
            "outBytes": self.dram_out_bytes,
            "inElems": 0,
            "outElems": 0,
            "inParamCount": 0,
            "inActCount": 0,
            "outActCount": 0,
            "instrs": {},
            "fused_compute_cycles": self.compute_latency_cycles,
            "sdpa_cycle_breakdown": {
                "fpu_matmul": self.fpu_matmul_cycles,
                "fpu_overhead": self.fpu_overhead_cycles,
                "sfpu_exp": self.sfpu_exp_cycles,
                "sfpu_reduce": self.sfpu_reduce_cycles,
                "sfpu_recip": self.sfpu_recip_cycles,
                "sfpu_overhead": self.sfpu_overhead_cycles,
                "math_active": self.math_active_cycles,
            },
        }


def sdpa_perf_stats(cfg: "SdpaConfig") -> Dict:
    return predict(cfg).to_polaris_op_perf_stats()


_ELEM_TO_DTYPE = {1: "bfp8_b", 2: "bfloat16", 4: "float32"}


def sdpa_config_from_shapes(q_shape, k_shape, v_shape, attrs=None, num_cores=110, arch=None):
    """Build an SdpaConfig from a prefill SDPA op's q/k/v tensor shapes + attrs.

    q: [..., num_heads, S, head_dim]; k: [..., num_kv_heads, S, head_dim];
    v: [..., num_kv_heads, S, v_head_dim]. Attrs may carry is_causal, q/k_chunk_size,
    element_size, exp_approx_mode. This is BH-calibrated (arch defaults to ARCH_BH).
    """
    attrs = attrs or {}
    q, k, v = [list(map(int, s)) for s in (q_shape, k_shape, v_shape)]
    nh, S, head_dim = q[-3], q[-2], q[-1]
    nkv, v_head_dim = k[-3], v[-1]
    dtype = _ELEM_TO_DTYPE.get(int(attrs.get("element_size", 2)), "bfloat16")
    qc = int(attrs.get("q_chunk_size") or 128)
    kc = int(attrs.get("k_chunk_size") or qc)
    return SdpaConfig(
        S=S, head_dim=head_dim, v_head_dim=(0 if v_head_dim == head_dim else v_head_dim),
        q_chunk=qc, k_chunk=kc, num_heads=nh, num_kv_heads=(0 if nkv == nh else nkv),
        num_cores=num_cores, is_causal=bool(attrs.get("is_causal", True)),
        input_dtype=dtype, accum_dtype="bfloat16",
        exp_approx_mode=bool(attrs.get("exp_approx_mode", True)),
        arch=arch or ARCH_BH,
    )


def predict(cfg: SdpaConfig) -> RooflineResult:
    a = cfg.arch
    r = RooflineResult(label=f"S={cfg.S}", arch_name=a.name)

    # The model assumes tile-aligned, chunk-divisible shapes; fail fast otherwise rather
    # than silently underestimating work from floor division. ValueError (not assert) so it
    # is not stripped under `python -O`.
    v_head_dim = cfg.v_head_dim or cfg.head_dim
    if not (cfg.q_chunk % TILE_HW == 0 and cfg.k_chunk % TILE_HW == 0
            and cfg.head_dim % TILE_HW == 0 and v_head_dim % TILE_HW == 0):
        raise ValueError(f"q_chunk/k_chunk/head_dim/v_head_dim must be multiples of {TILE_HW}")
    if not (cfg.S % cfg.q_chunk == 0 and cfg.S % cfg.k_chunk == 0):
        raise ValueError("S must be divisible by q_chunk and k_chunk")
    if cfg.input_dtype not in BYTES_PER_TILE or cfg.accum_dtype not in BYTES_PER_TILE:
        raise ValueError(f"unsupported dtype; supported: {sorted(BYTES_PER_TILE)}")

    q_chunks_total = (cfg.S // cfg.q_chunk) * cfg.num_heads
    Q = q_chunks_total / cfg.num_cores
    K = cfg.S // cfg.k_chunk
    K_eff = (K + 1) / 2.0 if cfg.is_causal else float(K)   # causal ends at the diagonal
    K_eff_gt0 = max(0.0, K_eff - 1.0)
    r.q_chunks_per_core, r.k_chunks_per_q, r.k_eff = Q, float(K), K_eff
    r.inner_iters = Q * K_eff

    qct = cfg.q_chunk // TILE_HW
    kct = cfg.k_chunk // TILE_HW
    dct_qk = cfg.head_dim // TILE_HW   # Q*K^T contracts over head_dim
    dct_v = v_head_dim // TILE_HW      # softmax*V produces v_head_dim (MLA: != head_dim)
    cpt = a.cpt(cfg.fidelity)

    # FPU: Q*K^T (over head_dim) + softmax*V (over v_head_dim), plus LLK template overhead.
    r.fpu_matmul_cycles = round(Q * K_eff * qct * kct * (dct_qk + dct_v) * cpt)
    r.fpu_overhead_cycles = round(r.fpu_matmul_cycles * a.fpu_overhead_frac)
    r.fpu_cycles = r.fpu_matmul_cycles + r.fpu_overhead_cycles

    # SFPU: exp, row-max reduce, recip, k>0 first-column rescale, per-iter overhead.
    exp_cyc = a.exp_tile_cycles if cfg.exp_approx_mode else a.exp_tile_cycles_accurate
    r.sfpu_exp_cycles = round(Q * K_eff * qct * kct * exp_cyc + Q * K_eff_gt0 * qct * a.exp_fc_cycles)
    r.sfpu_reduce_cycles = round(Q * K_eff * qct * a.reduce_max_per_tile)
    r.sfpu_recip_cycles = round(Q * qct * a.recip_cycles)
    r.sfpu_overhead_cycles = round(r.inner_iters * a.sfpu_overhead_per_inner_iter)
    r.sfpu_cycles = r.sfpu_exp_cycles + r.sfpu_reduce_cycles + r.sfpu_recip_cycles + r.sfpu_overhead_cycles

    # Per-core L1 unpacker bytes (drives the on-chip BW floor only). GQA/MQA share KV across
    # a query group, so K/V are read kv_frac as often as Q.
    ibpt = BYTES_PER_TILE[cfg.input_dtype]
    abpt = BYTES_PER_TILE[cfg.accum_dtype]
    nkv = cfg.num_kv_heads or cfg.num_heads
    kv_frac = nkv / cfg.num_heads
    unpack_q = Q * qct * dct_qk * ibpt
    unpack_k = Q * K_eff * kct * dct_qk * ibpt * kv_frac
    unpack_v = Q * K_eff * kct * dct_v * ibpt * kv_frac
    mask_chunks = (K_eff if (cfg.has_attn_mask and not cfg.is_causal) else (1.0 if cfg.is_causal else 0.0))
    unpack_mask = Q * mask_chunks * qct * kct * abpt if cfg.has_attn_mask or cfg.is_causal else 0.0
    r.unpack_bytes_total = round(unpack_q + unpack_k + unpack_v + unpack_mask)
    r.pack_bytes_total = round(Q * qct * dct_v * abpt)
    r.unpack_min_cycles = round(r.unpack_bytes_total / a.unpacker_bw_bytes_per_cycle)
    r.pack_min_cycles = round(r.pack_bytes_total / a.packer_bw_bytes_per_cycle)

    # Whole-op DRAM traffic for the Polaris memory path: each tensor read/written once, K/V
    # counted per KV head (not per query head), so GQA/MQA/MLA are not overcounted.
    st_tiles = cfg.S // TILE_HW
    r.dram_in_bytes = round(cfg.num_heads * st_tiles * dct_qk * ibpt      # Q
                            + nkv * st_tiles * dct_qk * ibpt              # K
                            + nkv * st_tiles * dct_v * ibpt)             # V
    r.dram_out_bytes = round(cfg.num_heads * st_tiles * dct_v * abpt)     # output

    # Compute-latency estimate = the FPU/SFPU execution union (calibrated, regime-aware
    # overlap), bounded below by the on-chip L1 BW floor, plus startup. This is a compute
    # FLOOR: measured wall-clock sits above it by the front-end/dispatch idle (next task).
    overlap_frac = OVERLAP_FRAC_CAUSAL if cfg.is_causal else OVERLAP_FRAC_NONCAUSAL
    r.math_active_cycles = round(r.fpu_cycles + r.sfpu_cycles - overlap_frac * min(r.fpu_cycles, r.sfpu_cycles))
    r.serial_sum_cycles = r.fpu_cycles + r.sfpu_cycles   # zero-overlap upper bracket, for reference
    l1_floor = max(r.unpack_min_cycles, r.pack_min_cycles)
    r.init_overhead_cycles = round(a.init_overhead_cycles)
    r.compute_latency_cycles = round(max(r.math_active_cycles, l1_floor) + r.init_overhead_cycles)
    r.math_idle_cycles = round(r.inner_iters * a.idle_per_inner_iter)
    return r


ARCH_BH = ArchConfig(name="BH")
