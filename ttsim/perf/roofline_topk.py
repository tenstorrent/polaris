# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-chip TopK roofline for Blackhole (TEN-4859): one router and three kernel regimes, (a)
topk_large_indices (SFPU bitonic sort-and-merge), (b) generic ttnn.topk (single and multi core factories),
(c) MoE gate kernels plus moe_grouped_topk. Every regime reports named terms whose rounded sum is the wall;
constants live in calibration sets keyed by tt-metal kernel revision (main tip 2dbd14bf632, 2026-09-12)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ttsim.perf.roofline_sdpa import (POLARIS_CALIBRATED_DEVNAME, POLARIS_CALIBRATED_SKU,
                                       ARCH_BH, _ceil_div)

TILE = 32
DEFAULT_NUM_CORES = 110
DEFAULT_GRID = (11, 10)          # p100a worker grid (compute_with_storage_grid_size)

ROUTE_COMPOSITE = "composite"
ROUTE_MULTI_CORE = "stock_multi_core"
ROUTE_SINGLE_CORE = "stock_single_core"
ROUTE_LARGE_INDICES = "large_indices_direct"

REGIME_LARGE_INDICES = "large_indices"
REGIME_GENERIC_SINGLE = "generic_single_core"
REGIME_GENERIC_MULTI = "generic_multi_core"
REGIME_GATE = "gate"
REGIME_INDEXER = "indexer_score_dsa"

# Kernel revisions with a calibration set. Main tip is the default; a caller that names none gets
# it with a low_confidence flag (a price without a stated kernel is a guess about the kernel).
KERNEL_REV_MAIN = "2dbd14bf632"
DEFAULT_KERNEL_REV = KERNEL_REV_MAIN
_KERNEL_REV_ALIASES = {"main": KERNEL_REV_MAIN, "main-tip": KERNEL_REV_MAIN, "2dbd14b": KERNEL_REV_MAIN,
                       KERNEL_REV_MAIN: KERNEL_REV_MAIN}


def _round_up(x: int, m: int) -> int:
    return m * _ceil_div(x, m)


def _is_pow2(w: int) -> bool:
    return w > 0 and (w & (w - 1)) == 0


def _piecewise(table: Dict[int, float], x: float) -> float:
    """Linear interpolation through the (x, y) points of a table, flat beyond the last point and
    zero below the first; the empty table is zero everywhere."""
    if not table:
        return 0.0
    xs = sorted(table)
    if x <= xs[0]:
        return table[xs[0]] if x == xs[0] else 0.0
    if x >= xs[-1]:
        return table[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= x <= hi:
            f = (x - lo) / (hi - lo)
            return table[lo] + f * (table[hi] - table[lo])
    return table[xs[-1]]


# ----------------------------------------------------------------------------------------------
# Result shared by all regimes
# ----------------------------------------------------------------------------------------------

@dataclass
class TopkResult:
    label: str = ""
    regime: str = ""
    route: str = ""
    body_mode: str = ""
    kernel_rev: str = ""
    k_snap: int = 0
    chunks: int = 0                # ceil(N_valid / K') per row (large_indices)
    segments: int = 0              # ceil(chunks / 32) (FusedSegmented)
    t_row_cycles: float = 0.0      # serial per-row (or per tile row) cost on one core
    row_blocks: int = 0            # ceil(rows / num_cores): rows-only work split
    device_cycles: int = 0         # round(sum(components))
    components: Dict[str, float] = field(default_factory=dict)
    breakdown: Dict[str, object] = field(default_factory=dict)
    low_confidence: bool = False
    low_confidence_reasons: List[str] = field(default_factory=list)

    def flag(self, reason: str):
        if reason not in self.low_confidence_reasons:
            self.low_confidence_reasons.append(reason)
        self.low_confidence = True

    def finish(self) -> "TopkResult":
        self.device_cycles = int(round(sum(self.components.values())))
        return self

    def to_polaris_op_perf_stats(self) -> Dict:
        """Fused-cycle perf_stats fields, merged over topk_sinf's element/byte counts. Same
        mechanism as the SDPA roofline: Device.execute_op keys the arch and SKU gate on the
        'sdpa_calibrated_*' keys and falls back to the generic instrs estimate elsewhere."""
        return {
            "fused_compute_cycles": self.device_cycles,
            "sdpa_calibrated_arch": POLARIS_CALIBRATED_DEVNAME,
            "sdpa_calibrated_sku": POLARIS_CALIBRATED_SKU,
            "topk_regime": self.regime,
            "topk_route": self.route,
            "topk_body_mode": self.body_mode,
            "topk_kernel_rev": self.kernel_rev,
            "topk_low_confidence": self.low_confidence,
            "topk_low_confidence_reasons": ",".join(self.low_confidence_reasons),
            "topk_wall_components": dict(self.components),
            "topk_cycle_breakdown": {
                "k_snap": self.k_snap,
                "chunks": self.chunks,
                "segments": self.segments,
                "t_row_cycles": self.t_row_cycles,
                "row_blocks": self.row_blocks,
                **self.breakdown,
            },
        }


# ----------------------------------------------------------------------------------------------
# Terms per regime
# ----------------------------------------------------------------------------------------------

K_BUCKETS = (512, 1024, 2048)
SEGMENT_CHUNKS = 32                                                   # compute.cpp:171-235 (main)
COLUMN_SPLIT_MAX_CORES = 128                                          # PR 53457 factory:351-455

BODY_FUSED_E2E = "FusedEndToEnd"
BODY_CLASSIC = "Classic"
BODY_FUSED_SEGMENTED = "FusedSegmented"


@dataclass(frozen=True)
class LargeIndicesTerms:
    """topk_large_indices terms, cycles at the AICLK. One row on one core costs
    c0_row + a_tot(K', mode) * chunks + c_seg * (segments - 1); the op adds c0_op once (the part of
    the R=1 intercept that is not paid again by later waves), c_single_chunk_op when a row is a
    single chunk, and c_launch(active cores) for the multi-core start-up."""
    c0_row: Dict[int, float]
    c0_op: Dict[int, float]
    a_tot: Dict[Tuple[int, str], float]           # (K', body mode) -> cycles per chunk
    c_seg: Dict[int, float]                       # per segment after the first (FusedSegmented)
    c_single_chunk_op: Dict[int, float] = field(default_factory=dict)
    c_launch_by_cores: Dict[int, float] = field(default_factory=dict)   # active cores -> cycles, DRAM input
    c_launch_l1_at_grid: Dict[int, float] = field(default_factory=dict)  # K' -> cycles at the full grid, L1 input
    c_ship: float = 0.0          # NoC ship per losing core under the column split (PR 53457, unmeasured)

    def a_tot_for(self, kp: int, mode: str) -> float:
        if (kp, mode) in self.a_tot:
            return self.a_tot[(kp, mode)]
        for (k, _m), v in self.a_tot.items():
            if k == kp:
                return v
        raise KeyError(f"no a_tot for K'={kp}")

    def c_launch(self, active_cores: int, kp: int = 2048, input_memory: str = "DRAM") -> float:
        """Multi-core start-up. With the input in DRAM it is the first-chunk fill of every active
        row (independent of K'); with the input in L1 it shrinks to a K'-dependent fraction,
        measured at the full grid and given the same shape over the active cores."""
        base = _piecewise(self.c_launch_by_cores, active_cores)
        if input_memory != "L1" or not self.c_launch_l1_at_grid or not self.c_launch_by_cores:
            return base
        full = self.c_launch_by_cores[max(self.c_launch_by_cores)]
        return base * (_piecewise(self.c_launch_l1_at_grid, kp) / full if full else 0.0)


@dataclass(frozen=True)
class GenericTerms:
    """Stock TopKDeviceOperation terms. Single core factory: per tile row c_row + Wt * c_in +
    n_sort * c_step(dtype/index) + 2 Kt * c_out, plus c_launch per op. Multi core factory: per tile
    row a_local(Kt) * Wt_local + a_final * n * Kt on the first row, later rows at row_pipe_frac of
    that (they overlap the final merge of the previous row), plus c_mc0 per op."""
    c_step: Dict[str, float]                      # "bf16/u16" style key -> cycles per insertion step
    c_in: Dict[str, float]                        # index dtype (or fp32) -> cycles per input tile
    c_out: float
    c_row: float
    c_launch: float
    largest_false_delta: float = 0.0              # extra cycles per insertion step
    c_read_tile: float = 0.0                      # reader terms: hidden behind compute, 0 (measured B twins)
    c_idxgen: float = 0.0
    a_local: Dict[int, float] = field(default_factory=dict)   # Kt (1 or >= 2) -> cycles per local tile
    a_final: float = 0.0                          # per gathered tile on the final core (B + G, collinear)
    c_mc0: float = 0.0                            # per op intercept at Ht = 1
    row_pipe_frac: float = 1.0                    # cost of a later tile row as a fraction of the first
    # Composite wrapper ops and the slice back to k
    c_prep0: float = 0.0
    c_prep_elem: float = 0.0                      # per input element (rows * N)
    c_fin_flat: float = 0.0                       # finish for k16 <= 512 (512 gathered elements per core)
    c_fin_hi0: float = 0.0                        # finish for k16 > 512: intercept
    c_fin_hi_elem: float = 0.0                    # and per input element (rows * N)
    c_slice0: float = 0.0                         # one SliceDeviceOperation
    c_slice_row: float = 0.0
    calibrated: bool = True

    def a_local_for(self, kt: int) -> float:
        if not self.a_local:
            return 0.0
        return self.a_local.get(kt, self.a_local[max(self.a_local)])


@dataclass(frozen=True)
class GateTerms:
    """Regime (c). Gates: one token per core, per token c_gate_token[kernel] (+ softmax, sigmoid,
    the 512-expert two-block combine) plus a start skew that grows with the active cores.
    moe_grouped_topk: per tile row c_row_g + Wt * c_wt_g (+ the grouped chain), a wave gap per
    extra wave."""
    c_gate_token: Dict[str, float]
    c_softmax: float = 0.0
    c_sigmoid: float = 0.0
    c_512_combine: float = 0.0
    c_skew_by_cores: Dict[int, float] = field(default_factory=dict)
    c_wrapper_ops: float = 0.0            # per launch, the TTMoEGate layout ops
    c_wrapper_ops_per_token: float = 0.0  # and their token slope
    c_launch: float = 0.0
    c_row_g: float = 0.0
    c_wt_g: float = 0.0
    c_grouped_chain: float = 0.0
    c_wave_gap: float = 0.0
    calibrated: bool = True

    def skew(self, active_cores: int) -> float:
        return _piecewise(self.c_skew_by_cores, active_cores)


@dataclass(frozen=True)
class IndexerTerms:
    """indexer_score_dsa: linear in query-key pairs, c_pair_core cycles per pair per core at the
    reference head geometry, scaled by the head geometry and divided by the active cores.

    The geometry scale is MEASURED (IXHD, six walls at Sq 512, T 8192): proportional to Hi, and almost
    flat in D. Halving D from 128 to 64 saves only 13 percent of the wall, not half, because the op is
    not matmul bound (tt-metal issue 56788), so the per-pair MAC count is the wrong basis. The earlier
    Hi * D form was ASSUMED and reads 42 percent low at D 64.
        scale = (Hi / ref_hi) * (d_flat + (1 - d_flat) * D / ref_d)
    """
    c_pair_core: float
    c_launch: float = 0.0
    ref_hi: int = 64
    ref_d: int = 128
    d_flat: float = 0.7362       # the D-independent share of a pair scan (IXHD; 1 - this scales with D)
    fidelity: str = "HiFi2"      # recorded only; the op runs at about 5 percent of the HiFi2 FPU peak


@dataclass(frozen=True)
class TopkCalibration:
    kernel_rev: str
    source: str
    large_indices: LargeIndicesTerms
    generic: GenericTerms
    gate: GateTerms
    indexer: IndexerTerms


# Main tip 2dbd14bf632 (tt-metal-fresh f3fbc8984f0, p100a, 2026-09-12; handoff/revamp/data/topk/
# topk_fits.csv and topk_campaign_results.md sections 2 to 6). Cycles = ns x 1.35.
LARGE_INDICES_TERMS_MAIN = LargeIndicesTerms(
    # The R=1 intercept c0 (1968.5 / 4012.5 / 6300.2) splits into a per-op and a per-row part
    # (finding F1); at K'=512 the per-op share is the whole intercept, the per-row part is clamped at 0.
    c0_row={512: 0.0, 1024: 644.2, 2048: 1157.8},
    c0_op={512: 1968.5, 1024: 3368.3, 2048: 5142.4},
    a_tot={(512, BODY_FUSED_E2E): 2122.7, (512, BODY_CLASSIC): 2921.0,
           (1024, BODY_FUSED_SEGMENTED): 4901.0, (2048, BODY_FUSED_SEGMENTED): 11436.0},
    c_seg={512: 0.0, 1024: 2393.3, 2048: 5677.0},
    # Single-chunk rows cost more than the slope extrapolates; the excess is per op (the 6-wave
    # MSA cell shows no per-row excess). K'=2048 has no single-chunk cell.
    c_single_chunk_op={512: 192.5, 1024: 705.3},
    # Multi-core start-up (finding F2): 636 ns at 32 active cores (composite li ops), 3602 ns at
    # the full grid (L3 R=C small-N cells), interpolated between, zero on one core.
    c_launch_by_cores={1: 0.0, 32: 859.05, 110: 4862.6},
    # L5 cells (L1-interleaved input, R=C) against their R=1 DRAM twins: 208 ns at K'=512 and
    # 1325 ns at K'=2048 in place of the 3602 ns DRAM start-up; K'=1024 is interpolated.
    c_launch_l1_at_grid={512: 281.0, 2048: 1788.0},
)

GENERIC_TERMS_MAIN = GenericTerms(
    c_step={"bf16/u16": 5414.7, "bf16/u32": 5578.4, "bfp8/u16": 5413.3, "bfp8/u32": 5572.2, "fp32/fp32": 5604.9},
    c_in={"u16": 415.35, "u32": 665.74, "fp32": 930.99},
    c_out=420.39,
    c_row=1694.7,
    c_launch=9776.3,
    largest_false_delta=88.705,
    # Class E Ht=1 cells fitted alone (a_local(Kt), B+G, intercept); the two Ht in {2, 4} cells at
    # (16384, 32) give the later-row fraction 0.566 (consecutive rows overlap their local and final stages).
    a_local={1: 6788.6, 2: 8877.9},
    a_final=1921.3,
    c_mc0=7396.7,
    row_pipe_frac=0.5663,
    # Composite wrapper ops (finding F4): prep linear in elements; the finish is flat up to k16 512
    # (k16 / 16 cores, 512 gathered elements each) with its own law at k16 2048.
    c_prep0=5793.3, c_prep_elem=0.014898,
    c_fin_flat=29973.0,
    c_fin_hi0=116297.0, c_fin_hi_elem=0.00312,
    # Two SliceDeviceOperations follow every k that is not a tile multiple (0.46 to 2.2 us each).
    c_slice0=893.0, c_slice_row=0.4631,
)

GATE_TERMS_MAIN = GateTerms(
    # generalized_moe_gate at B=1 without softmax (1356.5 ns), deepseek_moe_gate at B=1 with its
    # sigmoid front end (1307 ns).
    c_gate_token={"generalized_moe_gate": 1831.3, "deepseek_moe_gate": 1764.5 - 307.8},
    c_softmax=50.4,
    c_sigmoid=307.8,
    c_512_combine=6586.0,
    # Finding F5: the duration spans the first core's start to the last core's end, so it rises
    # with the active cores: +109 ns at 32 and +129 ns at 110 against B=1.
    c_skew_by_cores={1: 0.0, 32: 147.2, 110: 174.5},
    # TTMoEGate.forward layout ops around the gate op: a reshape, two sharded-to-interleaved, three slices
    # and one interleaved-to-sharded. Linear in the tokens of the launch over B 1 to 110 at 128 experts
    # (R1W, relative least squares on six walls, worst 7.0 percent at B 1): 11,466.9 + 48.528 per token in
    # ns. The router projection matmul is real work priced elsewhere and is not in this term. Charged only
    # when the caller asks for the wrapper; a bare ttnn gate op pays none of it.
    c_wrapper_ops=15480.3, c_wrapper_ops_per_token=65.513,
    c_row_g=3394.2, c_wt_g=7206.4, c_grouped_chain=21750.0, c_wave_gap=1734.8,
)

# indexer_score_dsa on the calibration checkout (kernels 72620d5; the main-tip op rejects the probe's
# weights layout): 0.6729 ns per pair on 88 cores = 79.93 cycles per pair per core, 3 points within 1.2 percent.
INDEXER_TERMS_SHARED = IndexerTerms(c_pair_core=79.93)

CALIBRATIONS: Dict[str, TopkCalibration] = {
    KERNEL_REV_MAIN: TopkCalibration(
        KERNEL_REV_MAIN, "p100a 2026-09-12, handoff/revamp/data/topk/topk_fits.csv",
        LARGE_INDICES_TERMS_MAIN, GENERIC_TERMS_MAIN, GATE_TERMS_MAIN, INDEXER_TERMS_SHARED),
}
LARGE_INDICES_TERMS_BH = LARGE_INDICES_TERMS_MAIN
GENERIC_TERMS_BH = GENERIC_TERMS_MAIN
GATE_TERMS_BH = GATE_TERMS_MAIN
INDEXER_TERMS_BH = INDEXER_TERMS_SHARED


def calibration_for(kernel_rev: Optional[str]) -> Tuple[TopkCalibration, bool]:
    """Calibration set for a kernel revision; (set, defaulted). None selects the default and
    reports it, so the caller can flag the result."""
    if kernel_rev is None:
        return CALIBRATIONS[DEFAULT_KERNEL_REV], True
    key = _KERNEL_REV_ALIASES.get(str(kernel_rev))
    if key is None:
        raise ValueError(f"unknown TopK kernel_rev {kernel_rev!r}; calibrated: {sorted(CALIBRATIONS)}")
    return CALIBRATIONS[key], False


def _apply_rev(r: TopkResult, cal: TopkCalibration, defaulted: bool) -> TopkResult:
    r.kernel_rev = cal.kernel_rev
    if defaulted:
        r.flag(f"kernel_rev_defaulted_{cal.kernel_rev}")
    return r


# ----------------------------------------------------------------------------------------------
# Regime (a): topk_large_indices
# ----------------------------------------------------------------------------------------------

def snap_k(k: int) -> int:
    """Smallest LLK sort bucket holding k (topk_large_indices_program_factory.cpp:33-41; the kernel
    supports k in [16, 2048], multiples of 16)."""
    for kp in K_BUCKETS:
        if k <= kp:
            return kp
    raise ValueError(f"TopK K={k} above the largest calibrated bucket ({K_BUCKETS[-1]})")


def body_mode(k_snap: int, n_phys: int) -> str:
    """compute_body_mode on origin/main (program_factory.cpp:130-143): gated on the snapped k and
    the physical width, not on valid_length."""
    if k_snap >= 1024:
        return BODY_FUSED_SEGMENTED
    return BODY_FUSED_E2E if _ceil_div(n_phys, k_snap) <= SEGMENT_CHUNKS else BODY_CLASSIC


@dataclass
class TopkConfig:
    N: int                 # row length scanned per row (valid_length when set)
    K: int                 # requested top-k
    rows: int = 1          # rows across all leading dims; one whole row per core
    num_cores: int = DEFAULT_NUM_CORES
    N_phys: Optional[int] = None      # physical row width when valid_length trims the scan
    grid: Tuple[int, int] = DEFAULT_GRID
    column_split: bool = False        # PR 53457 column-parallel engine, not on main; off by default
    input_memory: str = "DRAM"        # DRAM or L1 interleaved input (the start-up term differs)
    kernel_rev: Optional[str] = None
    terms: Optional[LargeIndicesTerms] = None


def column_split_config(chunks: int, rows: int, grid: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """PR 53457 compute_model_column_split_config: search a x b core rectangles, cost in merge
    units 2 * ceil(chunks / P) + ceil(log2 P) against the row law 2 * chunks. Returns (P, cost)
    when the split is enabled, else None."""
    gx, gy = grid
    cost_row = 2 * chunks
    best: Optional[Tuple[int, int]] = None
    for a in range(1, gx + 1):
        for b in range(1, gy + 1):
            p = a * b
            if p < 2 or p > min(chunks, COLUMN_SPLIT_MAX_CORES):
                continue
            if rows > 1 and (gx // a) * (gy // b) < rows:
                continue
            cost = 2 * _ceil_div(chunks, p) + math.ceil(math.log2(p))
            if best is None or cost < best[1] or (cost == best[1] and p < best[0]):
                best = (p, cost)
    if best is None:
        return None
    p, cost = best
    enabled = cost < cost_row if rows == 1 else cost + max(2, cost_row / 8) <= cost_row
    return (p, cost) if enabled else None


def predict_topk(cfg: TopkConfig) -> TopkResult:
    """topk_large_indices device cycles. Rows-only parallelism: each core runs whole rows serially,
    so device time steps at every num_cores rows (rows=num_cores+1 costs 2x rows=num_cores)."""
    if cfg.K < 1 or cfg.N < cfg.K or cfg.rows < 1:
        raise ValueError(f"TopK roofline needs 1 <= K <= N and rows >= 1; "
                         f"got N={cfg.N} K={cfg.K} rows={cfg.rows}")
    cal, defaulted = calibration_for(cfg.kernel_rev)
    terms = cfg.terms or cal.large_indices
    kp = snap_k(cfg.K)
    n_phys = cfg.N_phys if cfg.N_phys is not None else cfg.N
    r = TopkResult(label=f"N={cfg.N} K={cfg.K}", regime=REGIME_LARGE_INDICES,
                   route=ROUTE_LARGE_INDICES, k_snap=kp)
    _apply_rev(r, cal, defaulted)
    r.body_mode = body_mode(kp, n_phys)
    r.chunks = _ceil_div(cfg.N, kp)
    r.segments = _ceil_div(r.chunks, SEGMENT_CHUNKS) if r.body_mode == BODY_FUSED_SEGMENTED else 1
    a_tot = terms.a_tot_for(kp, r.body_mode)
    c_seg = terms.c_seg.get(kp, 0.0)
    r.t_row_cycles = terms.c0_row[kp] + a_tot * r.chunks + c_seg * (r.segments - 1)
    r.row_blocks = _ceil_div(cfg.rows, cfg.num_cores)
    active = min(cfg.rows, cfg.num_cores)
    launch = terms.c_launch(active, kp, cfg.input_memory)
    if cfg.input_memory == "L1" and active > 1 and kp not in terms.c_launch_l1_at_grid and terms.c_launch_l1_at_grid:
        r.flag("l1_input_start_up_interpolated_in_k_bucket")
    op_fixed = terms.c0_op[kp]
    if r.chunks == 1:
        if kp in terms.c_single_chunk_op:
            op_fixed += terms.c_single_chunk_op[kp]
        elif terms.c_single_chunk_op:
            r.flag("single_chunk_row_unmeasured_at_this_k_bucket")
        # A single-chunk row is a 3 us kernel whose call-to-call jitter is 1 to 3 percent, more
        # than the K dependence the model does not have (K=16..512 spread 2 percent at N=256).
        r.flag("single_chunk_row_jitter_3pct")
    r.breakdown.update({"active_cores": active})

    split = column_split_config(r.chunks, cfg.rows, cfg.grid) if cfg.column_split else None
    if split is not None:
        # Every row owns a rectangle of P cores: one wave, merge unit u = a_tot / 2.
        p, cost = split
        u = a_tot / 2.0
        r.row_blocks = 1
        r.t_row_cycles = terms.c0_row[kp] + u * cost + terms.c_ship * (p - 1)
        r.components = {
            "row_fixed": terms.c0_row[kp],
            "chunk_sort": u * 2 * _ceil_div(r.chunks, p),
            "tree_merge": u * math.ceil(math.log2(p)),
            "noc_ship": terms.c_ship * (p - 1),
            "segment_merge": 0.0,
            "op_fixed": op_fixed,
            "launch": terms.c_launch(min(cfg.rows * p, cfg.num_cores), kp, cfg.input_memory),
        }
        r.breakdown.update({"column_split_cores": p, "column_split_cost": cost})
        r.flag("column_split_uncalibrated")
        return r.finish()

    r.components = {
        "row_fixed": r.row_blocks * terms.c0_row[kp],
        "chunk_sort": r.row_blocks * a_tot * r.chunks,
        "segment_merge": r.row_blocks * c_seg * (r.segments - 1),
        "op_fixed": op_fixed,
        "launch": launch,
    }
    return r.finish()


def topk_perf_stats(cfg: TopkConfig) -> Dict:
    return predict_topk(cfg).to_polaris_op_perf_stats()


# ----------------------------------------------------------------------------------------------
# Regime (b): generic ttnn.topk (TILE), single core and multi core factories
# ----------------------------------------------------------------------------------------------

# L1 bound on Kt for the single core factory (topk_utils.cpp:297-339, BH worker L1 1572864 B),
# INFERRED in generic_topk_kernel.md section 1. Beyond this the op fails validation.
MAX_KT_SINGLE_CORE = {"u16": 125, "u32": 82, "fp32": 61}

FACTORY_AUTO = "auto"
FACTORY_SINGLE = "single_core"
FACTORY_MULTI = "multi_core"

_DTYPE_SHORT = {"bfloat16": "bf16", "bf16": "bf16", "bfloat8_b": "bfp8", "bfp8": "bfp8", "bfp8_b": "bfp8",
                "float32": "fp32", "fp32": "fp32"}


@dataclass
class GenericConfig:
    rows: int
    N: int
    k: int
    num_cores: int = DEFAULT_NUM_CORES
    grid: Tuple[int, int] = DEFAULT_GRID
    dtype: str = "bfloat16"
    largest: bool = True
    has_indices_tensor: bool = False
    factory: str = FACTORY_AUTO
    kernel_rev: Optional[str] = None
    terms: Optional[GenericTerms] = None


def generic_tile_geometry(rows: int, N: int, k: int) -> Tuple[int, int, int]:
    """(Ht, Wt, Kt): the host pads the width to >= 64 and to a tile multiple (topk.cpp:626-650),
    rounds k up to 32 (topk.cpp:44-46) and splits tile rows Ht = ceil(rows / 32)."""
    w_pad = max(64, _round_up(N, TILE))
    return _ceil_div(rows, TILE), w_pad // TILE, _ceil_div(k, TILE)


def index_dtype(w_pad: int, dtype: str) -> str:
    """UINT16 indices unless the padded width exceeds 65535 or the input is fp32 (topk_utils.cpp:341-345)."""
    if _DTYPE_SHORT.get(dtype, dtype) == "fp32":
        return "fp32"
    return "u32" if w_pad > 65535 else "u16"


def topk_multicore_structurally_eligible(w_pad: int, ht: int, k: int) -> bool:
    """topk_utils.cpp:27-44 with topk_constants.hpp:13-31: (W >= 8192 or (Ht <= 2 and W >= 1024))
    and W < 65535 and pow2(W) and k <= 64."""
    width_gate = w_pad >= 8192 or (ht <= 2 and w_pad >= 1024)
    return width_gate and w_pad < 65535 and _is_pow2(w_pad) and k <= 64


def insertion_steps(wt: int, kt: int) -> int:
    """64-element sorts per tile row on the single core path: exactly Wt - 1 at Kt = 1, and
    Kt * Wt - Kt * (Kt + 1) / 2 from the case logic in compute/topk.cpp:231-393 (confirmed by the
    0.3 percent fit of sweep class A)."""
    if kt < 1 or wt < kt:
        raise ValueError(f"insertion_steps needs 1 <= Kt <= Wt, got Wt={wt} Kt={kt}")
    return kt * wt - kt * (kt + 1) // 2


def multi_core_split(w_pad: int, kt: int, grid: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """find_topk_core_config_impl (topk_utils.cpp:73-223): pow2 splits from max(W / lp2(max_cores), 64) to
    W / 2, n = W / split local cores on a (grid_x - 1) x (grid_y - 2) rectangle, scored by 7 Wt_local +
    2 Wt_final. Returns (n_local, split) or None. The L1 footprint check is not reproduced (all production
    widths fit); the search matches the observed split on 25 of 25 campaign cells."""
    gx, gy = grid
    max_cores = (gx - 1) * (gy - 2)
    if gx < 2 or gy < 3 or max_cores < 2:
        return None
    lp2 = 1 << (max_cores.bit_length() - 1)
    split = max(w_pad // lp2, 64)
    split = 1 << (split - 1).bit_length()      # next pow2 at or above
    best: Optional[Tuple[int, int, int]] = None
    while split <= w_pad // 2:
        if w_pad % split == 0:
            n = w_pad // split
            if 1 < n <= max_cores:
                score = 7 * (split // TILE) + 2 * n * kt
                if best is None or score < best[2]:
                    best = (n, split, score)
        split *= 2
    return (best[0], best[1]) if best else None


def _slice_cycles(rows: int, terms: GenericTerms) -> float:
    """The two SliceDeviceOperations (values and indices) that follow a k the kernel padded."""
    return 2 * (terms.c_slice0 + terms.c_slice_row * rows)


def predict_generic(cfg: GenericConfig) -> TopkResult:
    """Stock TopKDeviceOperation. Single core factory: tile rows over cores, insertion sort per
    row. Multi core factory: width split over n local cores, final merge on one core, per tile
    row. Both add the slice back to k when k is not a tile multiple."""
    if cfg.k < 1 or cfg.N < cfg.k or cfg.rows < 1:
        raise ValueError(f"generic TopK roofline needs 1 <= k <= N and rows >= 1; "
                         f"got N={cfg.N} k={cfg.k} rows={cfg.rows}")
    cal, defaulted = calibration_for(cfg.kernel_rev)
    terms = cfg.terms or cal.generic
    ht, wt, kt = generic_tile_geometry(cfg.rows, cfg.N, cfg.k)
    w_pad = wt * TILE
    dt = _DTYPE_SHORT.get(cfg.dtype)
    idx = index_dtype(w_pad, cfg.dtype)
    r = TopkResult(label=f"rows={cfg.rows} N={cfg.N} k={cfg.k}")
    _apply_rev(r, cal, defaulted)
    r.breakdown.update({"Ht": ht, "Wt": wt, "Kt": kt, "index_dtype": idx})
    if dt is None:
        dt = "bf16"
        r.flag(f"dtype_{cfg.dtype}_priced_as_bf16")
    if not terms.calibrated:
        r.flag("generic_uncalibrated_on_this_kernel_rev")
    slice_cycles = _slice_cycles(cfg.rows, terms) if _round_up(cfg.k, TILE) != cfg.k else 0.0

    factory = cfg.factory
    if factory == FACTORY_AUTO:
        eligible = topk_multicore_structurally_eligible(w_pad, ht, _round_up(cfg.k, TILE))
        factory = FACTORY_MULTI if eligible else FACTORY_SINGLE
    if factory == FACTORY_MULTI:
        split = multi_core_split(w_pad, kt, cfg.grid)
        if split is None:
            r.flag("multi_core_split_unavailable")
            factory = FACTORY_SINGLE

    if factory == FACTORY_MULTI:
        assert split is not None      # the branch above falls back to the single core factory when it is
        n_local, split_w = split
        wt_local = split_w // TILE
        wt_final = n_local * kt
        local = terms.a_local_for(kt) * wt_local
        final = terms.a_final * wt_final
        later = terms.row_pipe_frac * (local + final) * (ht - 1)
        r.regime = REGIME_GENERIC_MULTI
        r.route = ROUTE_MULTI_CORE
        r.t_row_cycles = local + final
        r.row_blocks = ht
        r.components = {
            "local_sort": local,
            "final_merge": final,
            "row_pipeline": later,      # later tile rows overlap the final merge of the previous one
            "launch": terms.c_mc0,
            "slice": slice_cycles,
        }
        r.breakdown.update({"n_local": n_local, "Wt_local": wt_local, "Wt_final": wt_final})
        if ht > 1:
            r.flag("multi_core_later_rows_measured_at_one_width")
        if kt > 2:
            r.flag("multi_core_kt_above_2_extrapolated")
        return r.finish()

    if kt > MAX_KT_SINGLE_CORE[idx]:
        raise ValueError(f"generic TopK Kt={kt} exceeds the L1 bound for {idx} indices "
                         f"({MAX_KT_SINGLE_CORE[idx]} tiles)")
    n_sort = insertion_steps(wt, kt)
    step_key = f"{dt}/{idx}"
    c_step = terms.c_step.get(step_key, terms.c_step["bf16/u16"])
    if step_key not in terms.c_step:
        r.flag(f"c_step_{step_key}_priced_as_bf16_u16")
    c_step += 0.0 if cfg.largest else terms.largest_false_delta
    c_in = terms.c_in.get(idx, terms.c_in["u16"])
    t_row = terms.c_row + wt * c_in + n_sort * c_step + 2 * kt * terms.c_out
    idx_read = terms.c_read_tile if cfg.has_indices_tensor else terms.c_idxgen
    t_reader = wt * (terms.c_read_tile + idx_read)
    blocks = _ceil_div(ht, cfg.num_cores)
    r.regime = REGIME_GENERIC_SINGLE
    r.route = ROUTE_SINGLE_CORE
    r.t_row_cycles = max(t_row, t_reader)
    r.row_blocks = blocks
    r.components = {
        "row_fixed": blocks * terms.c_row,
        "transpose_in": blocks * wt * c_in,
        "insertion_sort": blocks * n_sort * c_step,
        "transpose_out": blocks * 2 * kt * terms.c_out,
        "reader_wait": blocks * max(0.0, t_reader - t_row),
        "launch": terms.c_launch,
        "slice": slice_cycles,
    }
    r.breakdown.update({"n_sort": n_sort, "t_reader_cycles": t_reader, "c_step_key": step_key})
    return r.finish()


# ----------------------------------------------------------------------------------------------
# Regime (c): MoE gate kernels
# ----------------------------------------------------------------------------------------------

# COUNTED SFPU issue slots per token (router_topk_kernels.md sections 5.1, 5.2), recorded in the
# breakdown for the thread-split work; the wall uses the measured per-token constants.
GATE_SLOTS_PER_TOKEN = {
    "deepseek_moe_gate": 215.0,
    "generalized_moe_gate": 290.0,
}
GATE_SOFTMAX_SLOTS = 60.0            # COUNTED: softmax branch in finalize_ungrouped (about 40 to 60)
# No "sampling" gate kernel: the sampling call sites take the composite route into topk_large_indices
# (verified on the device for 5 of 5 sampling cells), so a gate law for them would price nothing real.
GATE_KERNELS = ("deepseek_moe_gate", "generalized_moe_gate", "moe_grouped_topk")
GATE_SIGMOID_DEFAULT = {"deepseek_moe_gate": True, "generalized_moe_gate": False}


@dataclass
class GateConfig:
    kernel: str
    tokens: int
    N: int = 256
    k: int = 8
    softmax: bool = False
    sigmoid: Optional[bool] = None      # None: the kernel's own default (deepseek_moe_gate on)
    n_groups: int = 1
    num_cores: int = DEFAULT_NUM_CORES
    wrapper: bool = False               # price TTMoEGate.forward, not the bare ttnn gate op
    kernel_rev: Optional[str] = None
    terms: Optional[GateTerms] = None
    generic_terms: Optional[GenericTerms] = None


def predict_gate(cfg: GateConfig) -> TopkResult:
    """One token per core for the gates (a per-token constant plus a start skew that grows with
    the active cores, whole extra launches beyond the grid); moe_grouped_topk per tile row over
    cores with a wave gap per extra wave."""
    if cfg.kernel not in GATE_KERNELS:
        raise ValueError(f"unknown gate kernel {cfg.kernel!r}; known: {GATE_KERNELS}")
    if cfg.tokens < 1 or cfg.N < cfg.k or cfg.k < 1:
        raise ValueError(f"gate roofline needs tokens >= 1 and 1 <= k <= N; got {cfg}")
    cal, defaulted = calibration_for(cfg.kernel_rev)
    terms = cfg.terms or cal.gate
    r = TopkResult(label=f"{cfg.kernel} tokens={cfg.tokens} N={cfg.N} k={cfg.k}",
                   regime=REGIME_GATE, route=cfg.kernel)
    _apply_rev(r, cal, defaulted)
    if not terms.calibrated:
        r.flag("gate_uncalibrated_on_this_kernel_rev")
    if cfg.kernel == "moe_grouped_topk":
        ht, wt, kt = generic_tile_geometry(cfg.tokens, cfg.N, cfg.k)
        waves = _ceil_div(ht, cfg.num_cores)
        chain = terms.c_grouped_chain if cfg.n_groups > 1 else 0.0
        r.t_row_cycles = terms.c_row_g + wt * terms.c_wt_g + chain
        r.row_blocks = waves
        r.components = {
            "row_fixed": waves * terms.c_row_g,
            "width_tiles": waves * wt * terms.c_wt_g,
            "grouped_chain": waves * chain,
            "wave_gap": (waves - 1) * terms.c_wave_gap,
            "launch": terms.c_launch,
        }
        r.breakdown.update({"Ht": ht, "Wt": wt, "Kt": kt, "waves": waves})
        return r.finish()

    sigmoid = GATE_SIGMOID_DEFAULT[cfg.kernel] if cfg.sigmoid is None else cfg.sigmoid
    slots = GATE_SLOTS_PER_TOKEN[cfg.kernel] + (GATE_SOFTMAX_SLOTS if cfg.softmax else 0.0)
    launches = _ceil_div(cfg.tokens, cfg.num_cores)
    active = min(cfg.tokens, cfg.num_cores)
    combine = terms.c_512_combine if cfg.N > 256 else 0.0
    per_launch = terms.c_gate_token[cfg.kernel] + (terms.c_softmax if cfg.softmax else 0.0) + \
        (terms.c_sigmoid if sigmoid else 0.0) + combine
    r.t_row_cycles = per_launch
    r.row_blocks = launches
    r.components = {
        "token_compute": launches * terms.c_gate_token[cfg.kernel],
        "softmax": launches * (terms.c_softmax if cfg.softmax else 0.0),
        "sigmoid": launches * (terms.c_sigmoid if sigmoid else 0.0),
        "combine_512": launches * combine,
        "start_skew": launches * terms.skew(active),
        "launch": launches * terms.c_launch,
        "wrapper_ops": launches * ((terms.c_wrapper_ops + terms.c_wrapper_ops_per_token * active)
                                  if cfg.wrapper else 0.0),
    }
    r.breakdown.update({"launches": launches, "active_cores": active, "sfpu_slots_per_token": slots})
    if cfg.N > 512:
        r.flag("gate_layout_above_512_experts_unmeasured")
    if cfg.wrapper:
        # One measured point: 32 tokens, k 8, 128 experts. The wrapper cannot run above num_cores tokens
        # (one core per token), so there is no larger point to fit a slope against.
        if cfg.N > GATE_WRAPPER_MEASURED_EXPERTS:
            # +4.9 percent of layout at 512 experts against 128, one point, not modelled
            r.flag("gate_wrapper_layout_above_128_experts_unmeasured")
        if cfg.tokens > cfg.num_cores:
            r.flag("gate_wrapper_above_one_token_per_core_does_not_run")
    return r.finish()


# ----------------------------------------------------------------------------------------------
# Router: which kernel a ttnn.topk call runs on Blackhole
# ----------------------------------------------------------------------------------------------

GATE_WRAPPER_MEASURED_EXPERTS = 128       # the R1W token sweep ran at 128 experts
INDEXER_HI_MEASURED = (32, 128)           # IXHD swept Hi 32, 64, 128
INDEXER_D_MEASURED = (64, 128)            # and D 64, 128

SMALL_K_ROUTE_MIN_PADDED_WIDTH = 4096     # topk.cpp:254
LARGE_K_ROUTE_MAX_WIDTH = 1 << 19         # topk.cpp:264
ROUTE_MAX_K = 2048
FINISH_FLAT_MAX_K16 = 512                 # finding F4: 512 gathered elements per core up to here
FINISH_HI_MEASURED_K16 = 2048


@dataclass
class TopkCall:
    """What the shim captures from a ttnn.topk call; the fields the router reads."""
    rows: int
    N: int
    k: int
    dim_is_last: bool = True
    largest: bool = True
    sorted: bool = True
    stable: bool = False
    dtype: str = "bfloat16"
    layout: str = "TILE"                 # TILE or ROW_MAJOR
    sharded: bool = False                # input or output memory config sharded
    has_indices_tensor: bool = False
    has_output_tensor: bool = False      # preallocated outputs
    sub_core_grids: Optional[int] = None  # cores in the passed grid, None when absent
    sub_core_grid: Optional[Tuple[int, int]] = None   # (x, y) extents of its first range
    arch: str = "Blackhole"
    input_memory: str = "DRAM"           # DRAM or L1 interleaved input
    num_cores: int = DEFAULT_NUM_CORES
    grid: Tuple[int, int] = DEFAULT_GRID
    kernel_rev: Optional[str] = None


def should_route_to_topk_large_indices(call: TopkCall) -> bool:
    """Mirror of should_route_to_topk_large_indices (reduction/topk/topk.cpp:272-364, origin/main
    2dbd14bf632), evaluated in the same order."""
    if not call.largest or call.stable:
        return False
    if call.has_indices_tensor or call.has_output_tensor or call.sub_core_grids is not None:
        return False
    if call.sharded or not call.dim_is_last or call.k > ROUTE_MAX_K:
        return False
    w_pad = max(64, _round_up(call.N, TILE))
    if call.k <= 64:
        # Small k: route only wide, non multi-core-eligible widths (low-Ht relaxation disabled).
        if w_pad < SMALL_K_ROUTE_MIN_PADDED_WIDTH or topk_multicore_structurally_eligible(w_pad, 3, call.k):
            return False
    if call.dtype != "bfloat16" or call.layout != "TILE" or call.arch != "Blackhole":
        return False
    return _round_up(call.k, 16) <= call.N <= LARGE_K_ROUTE_MAX_WIDTH


def _factory_grid(call: TopkCall) -> Tuple[int, int]:
    """Grid the stock factory is chosen on: the first range of a passed sub_core_grids (a count
    without extents is taken as one core, which the kernel treats as too narrow), else the device."""
    if call.sub_core_grids is not None:
        return call.sub_core_grid or (1, 1)
    return call.grid


def route_topk(call: TopkCall) -> str:
    """Kernel that runs for a ttnn.topk call: the composite (prep + topk_large_indices + finish),
    the stock multi core factory, or the stock single core factory. A passed sub_core_grids only
    disables the composite route (topk.cpp:293); the factory choice still runs on that grid, so a
    pow2 width in the gate range on a range of at least 2 x 3 is multi core (topk_utils.cpp:89-93)."""
    if should_route_to_topk_large_indices(call):
        return ROUTE_COMPOSITE
    ht, wt, _ = generic_tile_geometry(call.rows, call.N, call.k)
    if topk_multicore_structurally_eligible(wt * TILE, ht, _round_up(call.k, TILE)) and \
            multi_core_split(wt * TILE, _ceil_div(call.k, TILE), _factory_grid(call)) is not None:
        return ROUTE_MULTI_CORE
    return ROUTE_SINGLE_CORE


def composite_wrapper_cycles(rows: int, N: int, k: int, terms: GenericTerms) -> Tuple[Dict[str, float], List[str]]:
    """topk_route_prep (untilize + clamp, linear in elements), topk_route_finish (gather + tilize:
    flat up to k16 512 where it runs on k16 / 16 cores, a separate law at 2048) and the slice back
    to k when k is not a multiple of 16 (topk.cpp:160-177, 370-391). Returns (terms, flags)."""
    k16 = _round_up(k, 16)
    elems = rows * N
    flags: List[str] = []
    if k16 <= FINISH_FLAT_MAX_K16:
        finish = terms.c_fin_flat
    else:
        finish = terms.c_fin_hi0 + terms.c_fin_hi_elem * elems
        if k16 != FINISH_HI_MEASURED_K16:
            flags.append("finish_k_between_512_and_2048_unmeasured")
    if rows != 32:
        flags.append("composite_wrapper_measured_at_32_rows")
    comps = {
        "route_prep": terms.c_prep0 + terms.c_prep_elem * elems,
        "route_finish": finish,
        "slice": _slice_cycles(rows, terms) if k16 != k else 0.0,
    }
    return comps, flags


def predict_topk_call(call: TopkCall) -> TopkResult:
    """Price a ttnn.topk call by the kernel that runs on Blackhole (route_topk)."""
    if not call.dim_is_last:
        raise ValueError("TopK roofline prices last-dim topk only (the host transposes other dims)")
    route = route_topk(call)
    if route == ROUTE_COMPOSITE:
        cal, _ = calibration_for(call.kernel_rev)
        k16 = _round_up(call.k, 16)
        r = predict_topk(TopkConfig(N=call.N, K=k16, rows=call.rows, num_cores=call.num_cores,
                                    grid=call.grid, input_memory=call.input_memory, kernel_rev=call.kernel_rev))
        r.route = ROUTE_COMPOSITE
        comps, flags = composite_wrapper_cycles(call.rows, call.N, call.k, cal.generic)
        r.components.update(comps)
        for f in flags:
            r.flag(f)
        if not cal.generic.calibrated:
            r.flag("composite_wrapper_uncalibrated_on_this_kernel_rev")
        r.breakdown["k_rounded"] = k16
        return r.finish()
    cores = call.sub_core_grids if call.sub_core_grids else call.num_cores
    factory = FACTORY_MULTI if route == ROUTE_MULTI_CORE else FACTORY_SINGLE
    return predict_generic(GenericConfig(rows=call.rows, N=call.N, k=call.k, num_cores=cores,
                                         grid=_factory_grid(call), dtype=call.dtype, largest=call.largest,
                                         has_indices_tensor=call.has_indices_tensor, factory=factory,
                                         kernel_rev=call.kernel_rev))


def topk_call_perf_stats(call: TopkCall) -> Dict:
    return predict_topk_call(call).to_polaris_op_perf_stats()


# ----------------------------------------------------------------------------------------------
# indexer_score_dsa: the op before topk_large_indices in the DSA chain
# ----------------------------------------------------------------------------------------------

@dataclass
class IndexerConfig:
    Sq: int
    T: int
    Hi: int
    D: int
    num_cores: int = DEFAULT_NUM_CORES
    fidelity: Optional[str] = None
    kernel_rev: Optional[str] = None
    terms: Optional[IndexerTerms] = None


def predict_indexer(cfg: IndexerConfig) -> TopkResult:
    """Lightning indexer logits [Sq, T]: linear in query-key pairs over the active cores. The
    causal saving from chunk_start_idx is not modelled (the probe scanned the full Sq x T)."""
    if min(cfg.Sq, cfg.T, cfg.Hi, cfg.D, cfg.num_cores) < 1:
        raise ValueError(f"indexer roofline needs positive Sq, T, Hi, D, num_cores; got {cfg}")
    cal, defaulted = calibration_for(cfg.kernel_rev)
    terms = cfg.terms or cal.indexer
    fidelity = cfg.fidelity or terms.fidelity
    cpt = ARCH_BH.cpt(fidelity)
    out_tiles = _ceil_div(cfg.Sq, TILE) * _ceil_div(cfg.T, TILE)
    tile_macs = cfg.Hi * out_tiles * _ceil_div(cfg.D, TILE)
    head_scale = (cfg.Hi / terms.ref_hi) * (terms.d_flat + (1.0 - terms.d_flat) * cfg.D / terms.ref_d)
    r = TopkResult(label=f"indexer Sq={cfg.Sq} T={cfg.T} Hi={cfg.Hi} D={cfg.D}",
                   regime=REGIME_INDEXER, route=REGIME_INDEXER)
    _apply_rev(r, cal, defaulted)
    r.components = {
        "pair_scan": terms.c_pair_core * head_scale * cfg.Sq * cfg.T / cfg.num_cores,
        "launch": terms.c_launch,
    }
    r.breakdown.update({"tile_macs": tile_macs, "out_tiles": out_tiles, "fidelity": fidelity,
                        "fpu_floor_cycles": tile_macs * cpt / cfg.num_cores, "head_scale": head_scale})
    if not INDEXER_D_MEASURED[0] <= cfg.D <= INDEXER_D_MEASURED[1]:
        # the D axis has two measured points, 64 and 128; outside that the flat share is an extrapolation
        r.flag(f"indexer_d_outside_{INDEXER_D_MEASURED[0]}_to_{INDEXER_D_MEASURED[1]}")
    if not INDEXER_HI_MEASURED[0] <= cfg.Hi <= INDEXER_HI_MEASURED[1]:
        r.flag(f"indexer_hi_outside_{INDEXER_HI_MEASURED[0]}_to_{INDEXER_HI_MEASURED[1]}")
    return r.finish()


def indexer_perf_stats(cfg: IndexerConfig) -> Dict:
    return predict_indexer(cfg).to_polaris_op_perf_stats()
