# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-chip SDPA roofline for Blackhole (TEN-4716).

A shared engine-cycle core (matmul + softmax) feeds per-variant front-ends (prefill/MLA/decode/
chunked/window/mask/cross/sparse). Compute-bound regimes export a compute cost; decode/paged export
DRAM KV-stream bytes; device.execute_op takes max(compute, memory).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict

TILE_HW = 32
BYTES_PER_TILE = {"bfp8_b": 1088, "bfloat16": 2048, "float32": 4096}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# Constants are calibrated for Blackhole p100a only; must match the ttsim device package name so
# Device.execute_op can refuse the cost on a non-BH device.
POLARIS_CALIBRATED_DEVNAME = "Blackhole"

# FPU/SFPU overlap saturates toward 1 as the q-chunk grows (more independent tiles to interleave):
# overlap = 1 - k/qct, k fit to measured q_chunk 128 & 256 per regime (measured range 0.45-0.86).
OVERLAP_K_CAUSAL = 1.81
OVERLAP_K_NONCAUSAL = 1.28
OVERLAP_FRAC_MAX = 0.95
# MLA is matmul-dominated (tiny softmax), so the engines do not overlap. MLA overhead/overlap are
# a separate calibration (DeepSeek latent family); off-family MLA is flagged low-confidence.
OVERLAP_FRAC_MLA = 0.0


def _overlap_frac(qct, causal_like, is_mla):
    """FPU/SFPU overlap fraction. Grows with q-chunk (qct = q_chunk/32), capped below 1."""
    if is_mla:
        return OVERLAP_FRAC_MLA
    k = OVERLAP_K_CAUSAL if causal_like else OVERLAP_K_NONCAUSAL
    return max(0.0, min(OVERLAP_FRAC_MAX, 1.0 - k / qct))

# Device wall-clock as a per-regime multiple of the compute floor (measured on BH; the kernel is
# latency-bound, ~90% idle for MLA). Used to project real op latency, not just the compute floor.
WALL_FACTOR = {
    "prefill_causal": 2.5, "prefill_noncausal": 1.6, "cross": 1.3,
    "windowed": 2.5, "masked": 2.1, "chunked": 2.2, "sparse": 4.75, "mla": 11.0,
}


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
    fpu_overhead_frac: float = 0.132          # LLK template cycles / matmul cycles at head_dim 128, q_chunk 128
    fpu_overhead_per_inv_dct: float = 0.88    # head-dim overhead scaling; fit across head_dim 64 & 128
    fpu_overhead_ref_dct_sum: float = 8.0     # dct_qk+dct_v at the calibration head_dim (128)
    fpu_overhead_per_inv_qct: float = 0.492   # q-chunk overhead scaling; fit across q_chunk 64/128/256
    fpu_overhead_ref_qct: float = 4.0         # qct at the calibration q_chunk (128)
    fpu_overhead_frac_mla: float = 0.031      # MLA is matmul-dominated -> near-zero template overhead
    sfpu_lanes: int = 32
    exp_tile_cycles: float = 88.2             # approx exp per tile
    exp_tile_cycles_accurate: float = 77.9    # accurate exp per tile
    exp_fc_cycles: float = 17.0               # first-column exp on k>0 rescale
    reduce_max_per_tile: float = 30.0
    recip_cycles: float = 17.0
    sfpu_overhead_per_inner_iter: float = 417.0
    sfpu_overhead_per_inner_iter_mla: float = 96.8   # MLA per-iter SFPU overhead
    sparse_gather_overhead_frac: float = 0.117       # sparse-MLA scattered-KV gather overhead
    unpacker_bw_bytes_per_cycle: float = 80.0
    packer_bw_bytes_per_cycle: float = 64.0
    idle_per_inner_iter: float = 0.0
    init_overhead_cycles: float = 30000.0
    # Decode (memory-bound): measured effective KV-stream BW + a fixed dispatch/ramp latency.
    dram_kv_stream_bw_gbps: float = 343.0
    decode_fixed_overhead_cycles: float = 22000.0   # ~16.3 us @ 1.35 GHz
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
    S: int                    # QUERY sequence length
    kv_seq: int = 0           # KV sequence length; 0 = same as S. Cross-attention sets kv_seq != S.
    batch: int = 1            # prefill work + DRAM scale linearly with batch
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
    has_attn_mask: bool = False    # dense additive mask (masked prefill path)
    sliding_window: int = 0        # 0 = full attention; >0 = local window width (tokens)
    attention_sink: bool = False   # extra softmax-normalization pass over learned sink logits
    chunk_start_idx: int = 0       # chunked/paged prefill: absolute start of this Q chunk (prefix len)
    dram_scatter_derate: float = 1.0  # paged KV gather BW penalty (>1 inflates effective bytes)
    is_sparse: bool = False        # sparse-MLA: each query attends TOPK selected latent KV (kv_seq=TOPK)
    exp_approx_mode: bool = True
    arch: ArchConfig = field(default_factory=ArchConfig)


@dataclass
class RooflineResult:
    label: str = ""
    arch_name: str = ""
    regime: str = "prefill"            # prefill / decode / chunked / windowed / masked / sparse
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
    sfpu_extra_cycles: int = 0         # mask-add + attention-sink + flash-decode combine passes
    sfpu_cycles: int = 0
    math_active_cycles: int = 0        # FPU/SFPU union (calibrated overlap) = the compute estimate
    serial_sum_cycles: int = 0         # fpu+sfpu, zero-overlap upper bracket (not the prediction)
    math_idle_cycles: int = 0
    init_overhead_cycles: int = 0
    compute_latency_cycles: int = 0    # compute FLOOR (FPU/SFPU busy + L1 floor + init)
    wall_clock_cycles: int = 0         # latency estimate = compute floor x per-regime wall factor (decode adds memory latency)
    unpack_bytes_total: int = 0        # per-core L1 unpacker bytes (drives the on-chip BW floor only)
    pack_bytes_total: int = 0
    unpack_min_cycles: int = 0
    pack_min_cycles: int = 0
    dram_in_bytes: int = 0             # whole-op DRAM read traffic (Q + K/V per KV head)
    dram_out_bytes: int = 0            # whole-op DRAM write traffic (output)
    is_mla: bool = False               # asymmetric v_head_dim != head_dim (FlashMLA)
    is_memory_bound: bool = False      # decode/paged: KV-stream DRAM bytes are the real bound
    low_confidence: bool = False       # variant/regime calibrated on a narrow config family

    def to_polaris_op_perf_stats(self) -> Dict:
        """perf_stats for a fused SDPA op consumed by ttsim Device.execute_op.

        fused_compute_cycles is the full device wall-clock estimate (compute floor x per-regime
        latency multiple; decode carries its memory latency). instrs is empty (ttsim reads it as
        instruction counts); the compute floor stays in the breakdown for reference.
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
            "fused_compute_cycles": self.wall_clock_cycles,
            # Device.execute_op rejects the cost off-BH and flags per-regime-calibrated variants.
            "sdpa_calibrated_arch": POLARIS_CALIBRATED_DEVNAME,
            "sdpa_compute_is_floor": False,
            "sdpa_mla_low_confidence": (self.is_mla or self.low_confidence),
            "sdpa_regime": self.regime,
            "sdpa_wall_clock_cycles": self.wall_clock_cycles,
            "sdpa_cycle_breakdown": {
                "compute_floor": self.compute_latency_cycles,
                "fpu_matmul": self.fpu_matmul_cycles,
                "fpu_overhead": self.fpu_overhead_cycles,
                "sfpu_exp": self.sfpu_exp_cycles,
                "sfpu_reduce": self.sfpu_reduce_cycles,
                "sfpu_recip": self.sfpu_recip_cycles,
                "sfpu_overhead": self.sfpu_overhead_cycles,
                "sfpu_extra": self.sfpu_extra_cycles,
                "math_active": self.math_active_cycles,
                "dram_in_bytes": self.dram_in_bytes,
                "is_memory_bound": self.is_memory_bound,
            },
        }


def sdpa_perf_stats(cfg: "SdpaConfig") -> Dict:
    return predict(cfg).to_polaris_op_perf_stats()


_ELEM_TO_DTYPE = {1: "bfp8_b", 2: "bfloat16", 4: "float32"}


def sdpa_config_from_shapes(q_shape, k_shape, v_shape, attrs=None, num_cores=110, arch=None):
    """Build an SdpaConfig from a prefill op's q/k/v shapes + attrs. BH-calibrated (arch=ARCH_BH)."""
    attrs = attrs or {}
    q, k, v = [list(map(int, s)) for s in (q_shape, k_shape, v_shape)]
    nh, S, head_dim = q[-3], q[-2], q[-1]
    batch = q[-4] if len(q) >= 4 else 1
    nkv, kv_seq, v_head_dim = k[-3], k[-2], v[-1]
    dtype = _ELEM_TO_DTYPE.get(int(attrs.get("element_size", 2)), "bfloat16")
    qc = int(attrs.get("q_chunk_size") or 128)
    kc = int(attrs.get("k_chunk_size") or qc)
    return SdpaConfig(
        S=S, kv_seq=(0 if kv_seq == S else kv_seq), batch=batch,
        head_dim=head_dim, v_head_dim=(0 if v_head_dim == head_dim else v_head_dim),
        q_chunk=qc, k_chunk=kc, num_heads=nh, num_kv_heads=(0 if nkv == nh else nkv),
        num_cores=num_cores, is_causal=bool(attrs.get("is_causal", True)),
        input_dtype=dtype, accum_dtype="bfloat16",
        has_attn_mask=bool(attrs.get("has_attn_mask", False)),
        sliding_window=int(attrs.get("sliding_window") or attrs.get("sliding_window_size") or 0),
        attention_sink=bool(attrs.get("attention_sink", False)),
        chunk_start_idx=int(attrs.get("chunk_start_idx") or 0),
        dram_scatter_derate=float(attrs.get("dram_scatter_derate")
                                  or (1.15 if attrs.get("chunk_start_idx") else 1.0)),
        exp_approx_mode=bool(attrs.get("exp_approx_mode", True)),
        arch=arch or ARCH_BH,
    )


def _windowed_k_eff(S, q_chunk, k_chunk, kv_seq, window, causal):
    """Average visited K-chunks per q-chunk for a sliding/local window. Saturates at window/k_chunk
    (unlike (K+1)/2 which grows with S) -- fixes SWA overcounting."""
    Sqt = q_chunk // TILE_HW
    Skt = k_chunk // TILE_HW
    nq = max(1, _ceil_div(S, q_chunk))
    win_tiles = _ceil_div(window, TILE_HW)
    kv_tiles = _ceil_div(kv_seq, TILE_HW)
    kv_chunks = _ceil_div(kv_tiles, Skt)
    total = 0.0
    for g in range(nq):
        q_lo_t, q_hi_t = g * Sqt, g * Sqt + Sqt
        if causal:
            k_hi = min(kv_chunks, _ceil_div(q_hi_t, Skt))   # cap at kv length (windowed cross-attn)
            k_lo = max(0, q_lo_t - win_tiles) // Skt
        else:
            k_hi = min(_ceil_div(kv_tiles, Skt), _ceil_div(q_hi_t + win_tiles // 2, Skt))
            k_lo = max(0, q_lo_t - win_tiles // 2) // Skt
        total += max(0.0, float(k_hi - k_lo))
    return total / nq


def _wall_factor(cfg, r):
    """Per-regime device-wall / compute-floor ratio (see WALL_FACTOR)."""
    if r.is_mla and not cfg.is_sparse:
        return WALL_FACTOR["mla"]
    if cfg.is_sparse:
        return WALL_FACTOR["sparse"]
    if r.regime in WALL_FACTOR:
        return WALL_FACTOR[r.regime]
    if cfg.kv_seq and cfg.kv_seq != cfg.S:
        return WALL_FACTOR["cross"]
    return WALL_FACTOR["prefill_causal"] if cfg.is_causal else WALL_FACTOR["prefill_noncausal"]


def _engine_cycles(r, *, Q, K_eff, K_eff_gt0, inner_iters, qct, kct, dct_qk, dct_v, cpt,
                   exp_approx, arch, fpu_overhead_frac, sfpu_overhead_per_inner_iter, overlap_frac,
                   mask_add_tiles_per_iter=0.0, sink_passes_per_q=0.0, combine_passes=0.0):
    """Shared core: FPU matmul + SFPU softmax + calibrated overlap, on resolved work quantities.
    Optional mask/sink/combine terms default to 0 (prefill/MLA bit-identical). Fills r's fpu/sfpu fields."""
    a = arch
    # FPU: Q*K^T (over head_dim) + softmax*V (over v_head_dim) + LLK overhead; dense mask adds one
    # FPU pass per visited K-tile.
    r.fpu_matmul_cycles = round(Q * K_eff * qct * kct * (dct_qk + dct_v) * cpt)
    mask_add_cycles = round(Q * K_eff * qct * kct * mask_add_tiles_per_iter * cpt)
    r.fpu_overhead_cycles = round(r.fpu_matmul_cycles * fpu_overhead_frac)
    r.fpu_cycles = r.fpu_matmul_cycles + r.fpu_overhead_cycles + mask_add_cycles

    # SFPU: exp, row-max reduce, recip, k>0 rescale, per-iter overhead, + optional sink/combine passes.
    exp_cyc = a.exp_tile_cycles if exp_approx else a.exp_tile_cycles_accurate
    r.sfpu_exp_cycles = round(Q * K_eff * qct * kct * exp_cyc + Q * K_eff_gt0 * qct * a.exp_fc_cycles)
    r.sfpu_reduce_cycles = round(Q * K_eff * qct * a.reduce_max_per_tile)
    r.sfpu_recip_cycles = round(Q * qct * a.recip_cycles)
    r.sfpu_overhead_cycles = round(inner_iters * sfpu_overhead_per_inner_iter)
    r.sfpu_extra_cycles = round(sink_passes_per_q * Q * qct * (a.reduce_max_per_tile + a.recip_cycles)
                                + combine_passes * qct * (a.reduce_max_per_tile + a.recip_cycles))
    r.sfpu_cycles = (r.sfpu_exp_cycles + r.sfpu_reduce_cycles + r.sfpu_recip_cycles
                     + r.sfpu_overhead_cycles + r.sfpu_extra_cycles)

    r.math_active_cycles = round(r.fpu_cycles + r.sfpu_cycles
                                 - overlap_frac * min(r.fpu_cycles, r.sfpu_cycles))
    r.serial_sum_cycles = r.fpu_cycles + r.sfpu_cycles


def predict(cfg: SdpaConfig) -> RooflineResult:
    """Prefill / MLA / cross / windowed / masked / sparse SDPA (compute-bound). cfg.S is the query
    length; decode/paged have their own front-end (predict_decode)."""
    a = cfg.arch
    r = RooflineResult(label=f"S={cfg.S}", arch_name=a.name, regime="prefill")

    # head_dim need not be a multiple of 32 (vision uses 72/96/256); it is rounded up to whole tiles.
    v_head_dim = cfg.v_head_dim or cfg.head_dim
    r.is_mla = v_head_dim != cfg.head_dim
    Skv = cfg.kv_seq or cfg.S
    if not (cfg.q_chunk % TILE_HW == 0 and cfg.k_chunk % TILE_HW == 0):
        raise ValueError(f"q_chunk/k_chunk must be multiples of {TILE_HW}")
    if cfg.input_dtype not in BYTES_PER_TILE or cfg.accum_dtype not in BYTES_PER_TILE:
        raise ValueError(f"unsupported dtype; supported: {sorted(BYTES_PER_TILE)}")

    # The kernel pads the sequence up to a whole chunk, so round chunk counts up (exact-divisible
    # S is bit-identical to ceil); avoids falling to the passthrough mov-cost on non-128 shapes.
    q_chunks_total = _ceil_div(cfg.S, cfg.q_chunk) * cfg.num_heads * cfg.batch
    Q = q_chunks_total / cfg.num_cores
    K = _ceil_div(Skv, cfg.k_chunk)
    if cfg.chunk_start_idx > 0:
        # Chunked/paged prefill: dense prefix (no causal halving) + causal ramp over local k-chunks.
        local_kc = _ceil_div(cfg.S, cfg.k_chunk)
        K_eff = cfg.chunk_start_idx / cfg.k_chunk + (local_kc + 1) / 2.0
    elif cfg.sliding_window > 0:
        # Sliding window: visited K-chunks saturate at the window, not (K+1)/2.
        K_eff = _windowed_k_eff(cfg.S, cfg.q_chunk, cfg.k_chunk, Skv, cfg.sliding_window, cfg.is_causal)
    elif cfg.has_attn_mask:
        # Measured: the masked prefill path costs ~causal (half the k-chunks), density-independent.
        K_eff = (K + 1) / 2.0
    else:
        K_eff = (K + 1) / 2.0 if cfg.is_causal else float(K)
    K_eff_gt0 = max(0.0, K_eff - 1.0)
    r.q_chunks_per_core, r.k_chunks_per_q, r.k_eff = Q, float(K), K_eff
    r.inner_iters = Q * K_eff

    qct = cfg.q_chunk // TILE_HW
    kct = cfg.k_chunk // TILE_HW
    dct_qk = _ceil_div(cfg.head_dim, TILE_HW)   # Q*K^T contraction (tile-padded)
    dct_v = _ceil_div(v_head_dim, TILE_HW)      # softmax*V output (MLA: != head_dim)
    cpt = a.cpt(cfg.fidelity)

    # MLA uses its own overhead/overlap; a masked op behaves causal-like (causal overlap).
    if r.is_mla:
        fpu_overhead_frac = a.fpu_overhead_frac_mla
    else:
        # LLK template overhead is a larger share of a smaller matmul, so scale it with both
        # 1/head_dim and 1/q_chunk (fewer tiles per call -> overhead is a bigger fraction).
        fpu_overhead_frac = (a.fpu_overhead_frac
            + a.fpu_overhead_per_inv_dct * (1.0 / (dct_qk + dct_v) - 1.0 / a.fpu_overhead_ref_dct_sum)
            + a.fpu_overhead_per_inv_qct * (1.0 / qct - 1.0 / a.fpu_overhead_ref_qct))
    sfpu_overhead_per_inner_iter = a.sfpu_overhead_per_inner_iter_mla if r.is_mla else a.sfpu_overhead_per_inner_iter
    causal_like = cfg.is_causal or cfg.has_attn_mask
    overlap_frac = _overlap_frac(qct, causal_like, r.is_mla)

    sink_passes = 1.0 if cfg.attention_sink else 0.0
    _engine_cycles(r, Q=Q, K_eff=K_eff, K_eff_gt0=K_eff_gt0, inner_iters=r.inner_iters,
                   qct=qct, kct=kct, dct_qk=dct_qk, dct_v=dct_v, cpt=cpt,
                   exp_approx=cfg.exp_approx_mode, arch=a, fpu_overhead_frac=fpu_overhead_frac,
                   sfpu_overhead_per_inner_iter=sfpu_overhead_per_inner_iter, overlap_frac=overlap_frac,
                   mask_add_tiles_per_iter=0.0, sink_passes_per_q=sink_passes)
    if cfg.is_sparse:
        # Sparse-MLA: kv_seq=TOPK gives K_eff=TOPK/k_chunk; add the calibrated scattered-gather uplift.
        r.math_active_cycles = round(r.math_active_cycles * (1.0 + a.sparse_gather_overhead_frac))
        r.regime = "sparse"
        r.low_confidence = True
    elif cfg.chunk_start_idx > 0:
        r.regime = "chunked"
        r.low_confidence = True
    elif cfg.sliding_window > 0 or cfg.has_attn_mask or cfg.attention_sink or cfg.kv_seq:
        r.regime = ("windowed" if cfg.sliding_window > 0 else
                    "masked" if cfg.has_attn_mask else "prefill")
        r.low_confidence = True

    # Per-core L1 unpacker bytes (on-chip BW floor only). GQA/MQA share KV across a query group.
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

    # Whole-op DRAM traffic: Q/output sized by S, K/V by Skv and per KV head (GQA/MQA/MLA correct).
    sq_tiles = _ceil_div(cfg.S, TILE_HW)
    skv_tiles = _ceil_div(Skv, TILE_HW)
    derate = cfg.dram_scatter_derate or 1.0   # paged gather BW penalty on K/V reads
    r.dram_in_bytes = round((cfg.num_heads * sq_tiles * dct_qk * ibpt                   # Q
                             + (nkv * skv_tiles * dct_qk * ibpt) * derate              # K
                             + (nkv * skv_tiles * dct_v * ibpt) * derate) * cfg.batch)  # V
    r.dram_out_bytes = round(cfg.num_heads * sq_tiles * dct_v * abpt * cfg.batch)    # output

    # Compute FLOOR = math union bounded below by the L1 BW floor, plus startup.
    l1_floor = max(r.unpack_min_cycles, r.pack_min_cycles)
    r.init_overhead_cycles = round(a.init_overhead_cycles)
    r.compute_latency_cycles = round(max(r.math_active_cycles, l1_floor) + r.init_overhead_cycles)
    # Wall-clock estimate = compute floor x the per-regime latency multiple (measured on BH).
    r.wall_clock_cycles = round(r.compute_latency_cycles * _wall_factor(cfg, r))
    r.math_idle_cycles = round(r.inner_iters * a.idle_per_inner_iter)
    return r


def predict_decode(cache_len, num_q_heads, num_kv_heads, head_dim, v_head_dim=0, k_chunk=128,
                   batch=1, cur_pos=None, sliding_window=0, fidelity="HiFi4", input_dtype="bfloat16",
                   accum_dtype="bfloat16", num_cores=110, num_cores_per_head=0, arch=None):
    """Single-token flash/paged decode. Memory-bound: one query token streams the whole KV cache up
    to cur_pos each step, so the binding roof is DRAM KV traffic. cache_len (or cur_pos+1) is the
    attended length; exact per-step latency needs the runtime cur_pos from the trace."""
    a = arch or ARCH_BH
    v_head_dim = v_head_dim or head_dim
    is_mla = v_head_dim != head_dim
    if input_dtype not in BYTES_PER_TILE or accum_dtype not in BYTES_PER_TILE:
        raise ValueError(f"unsupported dtype; supported: {sorted(BYTES_PER_TILE)}")
    attended = int(cur_pos + 1) if cur_pos is not None else int(cache_len)
    # Windowed decode (SWA) attends only the last `window` keys, so the KV stream caps at the window.
    if sliding_window and sliding_window > 0:
        attended = min(attended, int(sliding_window))
    r = RooflineResult(label=f"decode L={attended}", arch_name=a.name, regime="decode",
                       is_memory_bound=True, is_mla=is_mla, low_confidence=True)

    ibpt = BYTES_PER_TILE[input_dtype]
    abpt = BYTES_PER_TILE[accum_dtype]
    dct_qk = _ceil_div(head_dim, TILE_HW)
    dct_v = _ceil_div(v_head_dim, TILE_HW)
    kc_tiles = max(1, _ceil_div(k_chunk, TILE_HW))   # round the chunk up to whole tiles
    st_tiles = _ceil_div(attended, k_chunk) * kc_tiles

    # Whole KV cache re-streamed once per step. MLA reuses K as V (no separate DRAM V read).
    kv_v_tiles = 0 if is_mla else dct_v
    r.dram_in_bytes = round(batch * num_kv_heads * st_tiles * (dct_qk + kv_v_tiles) * ibpt
                            + batch * num_q_heads * dct_qk * ibpt)   # + Q (single token)
    r.dram_out_bytes = round(batch * num_q_heads * dct_v * abpt)     # single-token output

    # Compute (GEMV, not binding): grid = batch*num_kv_heads groups, KV sequence split across cores.
    groups = max(1, batch * num_kv_heads)
    ncph = num_cores_per_head or max(1, num_cores // groups)
    active_cores = max(1, min(num_cores, groups * ncph))
    r.q_chunks_per_core = groups / active_cores
    k_tiles_per_core = st_tiles / ncph
    r.k_eff = k_tiles_per_core
    r.inner_iters = r.q_chunks_per_core * k_tiles_per_core
    cpt = a.cpt(fidelity)
    combine_passes = 0.0
    if ncph > 1:
        import math as _m
        combine_passes = _m.ceil(_m.log2(ncph))   # flash-decode tree reduction
    _engine_cycles(r, Q=r.q_chunks_per_core, K_eff=k_tiles_per_core,
                   K_eff_gt0=max(0.0, k_tiles_per_core - 1.0), inner_iters=r.inner_iters,
                   qct=1, kct=1, dct_qk=dct_qk, dct_v=dct_v, cpt=cpt,
                   exp_approx=False, arch=a, fpu_overhead_frac=a.fpu_overhead_frac_mla,
                   sfpu_overhead_per_inner_iter=a.sfpu_overhead_per_inner_iter_mla,
                   overlap_frac=OVERLAP_FRAC_MLA, combine_passes=combine_passes)

    # Memory-bound latency = fixed dispatch/ramp + KV stream at the measured decode BW. Carried in
    # fused_compute_cycles (the byte-path can't add the fixed term); device.execute_op's max() keeps it.
    mem_cycles = r.dram_in_bytes * a.clock_ghz / a.dram_kv_stream_bw_gbps
    r.init_overhead_cycles = round(a.decode_fixed_overhead_cycles)
    r.compute_latency_cycles = round(max(r.math_active_cycles, mem_cycles) + a.decode_fixed_overhead_cycles)
    r.wall_clock_cycles = r.compute_latency_cycles
    return r


def decode_config_from_shapes(q_shape, k_shape, v_shape=None, attrs=None, num_cores=110, arch=None):
    """Build predict_decode args from a decode op's q + k_cache (+ v_cache) shapes + attrs.
    q: [batch, num_q_heads, seq_q, head_dim]; k_cache: [batch, num_kv_heads, cache_len, head_dim]."""
    attrs = attrs or {}
    q = list(map(int, q_shape))
    k = list(map(int, k_shape))
    batch = q[-4] if len(q) >= 4 else 1
    num_q_heads, head_dim = q[-3], q[-1]
    num_kv_heads, cache_len = k[-3], k[-2]
    dtype = _ELEM_TO_DTYPE.get(int(attrs.get("element_size", 2)), "bfloat16")
    # V head dim from the v_cache tensor (ground truth for FlashMLA), attr only as a fallback.
    v_head_dim = int(v_shape[-1]) if v_shape else int(attrs.get("head_dim_v") or 0)
    return dict(cache_len=cache_len, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, v_head_dim=v_head_dim, k_chunk=int(attrs.get("k_chunk_size") or 128),
                batch=batch, cur_pos=attrs.get("cur_pos"),
                sliding_window=int(attrs.get("sliding_window") or attrs.get("sliding_window_size") or 0),
                fidelity=str(attrs.get("fidelity") or "HiFi4"),
                input_dtype=dtype, accum_dtype="bfloat16", num_cores=num_cores, arch=arch or ARCH_BH)


def decode_perf_stats(q_shape, k_shape, v_shape=None, attrs=None, num_cores=110, arch=None) -> Dict:
    return predict_decode(**decode_config_from_shapes(q_shape, k_shape, v_shape, attrs, num_cores, arch)).to_polaris_op_perf_stats()


ARCH_BH = ArchConfig(name="BH")
