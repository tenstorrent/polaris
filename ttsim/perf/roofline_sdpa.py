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

OVERLAP_FRAC = 0.63  # FPU/SFPU concurrency, as a fraction of the smaller engine (calibrated)


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
    risc_producer_ipc: float = 1.0
    clock_ghz: float = 1.35

    def cpt(self, fidelity: str) -> float:
        return self.cycles_per_tile_mac[fidelity]


@dataclass
class SdpaConfig:
    S: int
    head_dim: int = 128       # QK head dim (Q*K^T contraction)
    v_head_dim: int = 0       # P*V output dim; 0 means same as head_dim (symmetric SDPA)
    q_chunk: int = 128
    k_chunk: int = 128
    num_heads: int = 32
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
    math_active_cycles: int = 0
    math_idle_cycles: int = 0
    init_overhead_cycles: int = 0
    compute_latency_cycles: int = 0
    unpack_bytes_total: int = 0
    pack_bytes_total: int = 0
    unpack_min_cycles: int = 0
    pack_min_cycles: int = 0

    def to_polaris_op_perf_stats(self) -> Dict:
        # perf_stats for a fused SDPA op. fused_compute_cycles is the calibrated per-op
        # latency Device.execute_op consumes directly; inBytes/outBytes drive the memory
        # path; instrs is kept for reporting. inElems/outElems are 0 so hlmstats skips the
        # bytes-per-element precision check (inBytes are tile bytes, not raw element bytes),
        # and the param/act counts are 0 (SDPA carries no weights).
        return {
            "inBytes": self.unpack_bytes_total,
            "outBytes": self.pack_bytes_total,
            "inElems": 0,
            "outElems": 0,
            "inParamCount": 0,
            "inActCount": 0,
            "outActCount": 0,
            "instrs": {
                "mac": self.fpu_matmul_cycles,
                "exp": self.sfpu_exp_cycles,
                "max": self.sfpu_reduce_cycles,
                "div": self.sfpu_recip_cycles,
            },
            "fused_compute_cycles": self.compute_latency_cycles,
        }


def sdpa_perf_stats(cfg: "SdpaConfig") -> Dict:
    return predict(cfg).to_polaris_op_perf_stats()


def predict(cfg: SdpaConfig) -> RooflineResult:
    a = cfg.arch
    r = RooflineResult(label=f"S={cfg.S}", arch_name=a.name)

    # The model assumes tile-aligned, chunk-divisible shapes; fail fast otherwise rather
    # than silently underestimating work from floor division.
    v_head_dim = cfg.v_head_dim or cfg.head_dim
    assert cfg.q_chunk % TILE_HW == 0 and cfg.k_chunk % TILE_HW == 0 \
        and cfg.head_dim % TILE_HW == 0 and v_head_dim % TILE_HW == 0, \
        f"q_chunk/k_chunk/head_dim/v_head_dim must be multiples of {TILE_HW}"
    assert cfg.S % cfg.q_chunk == 0 and cfg.S % cfg.k_chunk == 0, \
        "S must be divisible by q_chunk and k_chunk"

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

    # L1 bytes and bandwidth floor.
    ibpt = BYTES_PER_TILE[cfg.input_dtype]
    abpt = BYTES_PER_TILE[cfg.accum_dtype]
    unpack_q = Q * qct * dct_qk * ibpt
    unpack_k = Q * K_eff * kct * dct_qk * ibpt
    unpack_v = Q * K_eff * kct * dct_v * ibpt
    mask_chunks = (K_eff if (cfg.has_attn_mask and not cfg.is_causal) else (1.0 if cfg.is_causal else 0.0))
    unpack_mask = Q * mask_chunks * qct * kct * abpt if cfg.has_attn_mask or cfg.is_causal else 0.0
    r.unpack_bytes_total = round(unpack_q + unpack_k + unpack_v + unpack_mask)
    r.pack_bytes_total = round(Q * qct * dct_v * abpt)
    r.unpack_min_cycles = round(r.unpack_bytes_total / a.unpacker_bw_bytes_per_cycle)
    r.pack_min_cycles = round(r.pack_bytes_total / a.packer_bw_bytes_per_cycle)

    # Compute latency = max of the math union, the L1 BW floor, and the RISC-V issue floor,
    # plus startup. Measured wall-clock sits above this (front-end idle = the next task).
    overlap = OVERLAP_FRAC * min(r.fpu_cycles, r.sfpu_cycles)
    r.math_active_cycles = round(r.fpu_cycles + r.sfpu_cycles - overlap)
    l1_floor = max(r.unpack_min_cycles, r.pack_min_cycles)
    risc_floor = (r.fpu_cycles + r.sfpu_cycles) / a.risc_producer_ipc
    r.init_overhead_cycles = round(a.init_overhead_cycles)
    r.compute_latency_cycles = round(max(r.math_active_cycles, l1_floor, risc_floor) + r.init_overhead_cycles)
    r.math_idle_cycles = round(r.inner_iters * a.idle_per_inner_iter)
    return r


ARCH_BH = ArchConfig(name="BH")
