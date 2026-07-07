#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""First-order data-movement READ-latency model (O2O).

Predicts the latency (in NoC/fabric clock cycles, ``fclk``) of a Tensix core
issuing ``noc_async_read`` transactions from DRAM, followed by a barrier.

This implements the O2O "Read Latency Modeling" model
(Confluence page 2433646619). A single unified formula covers both regimes the
page describes:

* Layer 1 - single core, single channel, queue depth ``Q = 1``
  (``num_channels = 1``): the per-transaction primitive.
* Layer 3 - single core, multi-channel interleaved reads
  (``num_channels > 1``, ``Q >= 1``): reads spread round-robin across DRAM
  channels, throttled by the per-core NoC-inbound ceiling.

The latency is the larger of two arms (the binding constraint):

* issue-bound:     ``delta_issue * Q + Tdram + 2*hops*Chop + Tdetect + N/B_channel``
* transport-bound: ``min(Q, n*) * delta_issue + Tdram + 2*hops*Chop + Q*N/B_eff``

where ``B_eff = min(min(Q, num_channels) * B_channel, noc_inbound_bpc)`` and
``n* = ceil(noc_inbound_bpc / B_channel)`` is the number of channels needed to
saturate the NoC inbound link.

Default constants are taken from the O2O page's "Model Update" section, which is
    calibrated against the Blackhole ``DRAM Interleaved Page Read`` microbenchmark
(see ``tests/test_back/test_read_latency.py``). The model is intentionally
first-order: NoC congestion / VC contention, row-miss / refresh penalties, and
non-DRAM (L1) sources are out of scope.

For timing integration the model also exposes a component decomposition
(:class:`ReadLatencyBreakdown` / :func:`predict_read_latency_breakdown`) that splits
the binding arm into *exposed* latency (fill + drain + serial issue) versus *hideable*
streaming, so a caller can add only the exposed part on top of the existing
bytes/bandwidth read cost (the "additive exposed latency" / Option C integration).

This module has no Polaris dependencies on purpose, so the physics can be unit
tested in isolation before being wired into the Device timing path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Elements in one 32x32 tile. One tile is the default DRAM "page" / transaction.
TILE_HW = 32
TILE_ELEMS = TILE_HW * TILE_HW  # 1024


@dataclass(frozen=True, kw_only=True)
class ReadLatencyConfig:
    """Hardware constants for the read-latency model.

    All cycle counts are in the NoC/fabric clock domain (``fclk``); on Blackhole
    the NoC is tied to the AI clock, so ``fclk ~= matrix/vector pipe clock`` and
    no extra clock-domain conversion is needed for BH.

    Every field is an arch-calibrated constant with **no default** — the values
    are the single source of truth in the arch YAML (``config/tt_bh.yaml`` memory
    block's ``read_latency`` section, parsed by ``MemoryReadLatencyModel``) and
    must be supplied by the caller. ``Device`` injects them from the parsed arch
    spec; tests build the config from the same YAML. ``num_dram_channels`` is the
    sole exception: it comes from the memory IP's ``num_units`` (not the
    ``read_latency`` block).
    """

    # --- Interleaved (Layer 3) model constants ---
    #: DRAM access bucket (row-hit best case): row activate + CAS + burst.
    tdram_cyc: float
    #: Barrier-clear tail.
    tdetect_cyc: float
    #: Per-core NoC inbound ceiling (bytes / fclk cycle).
    noc_inbound_bpc: float
    #: Per-read issue cost (cyc/read) - slope of latency vs Q in the issue regime.
    delta_issue_cyc: float
    #: NoC cost per hop (cyc); round-trip contributes ``2 * hops * chop``.
    chop_cyc_per_hop: float
    #: Single-channel, single-stream effective rate (bytes / fclk cycle).
    b_channel_bpc: float
    #: Default gating-channel hop distance (h_gate from the O2O page).
    default_hops: int
    #: DRAM channels for interleaving (from memory IP num_units; p100a = 7).
    num_dram_channels: int

    # --- Layer 1 (unary) closed-form constants (O2O "Model Validation" table) ---
    #: Fixed latency to first data for the unary model (Tissue+Tnoc+Tdram+Tdetect).
    tfixed_unary_cyc: float
    #: NoC flit width (bytes).
    flit_bytes: int
    #: Single-request receive rate (cyc/flit), N <= 32 KB.
    recv_cyc_per_flit_lo: float
    #: Pipelined receive rate (cyc/flit), N > 32 KB.
    recv_cyc_per_flit_hi: float
    #: Flit count at which delivery transitions to the pipelined rate (32 KB / 64 B).
    recv_knee_flits: int


@dataclass(frozen=True, kw_only=True)
class ReadSourceDescriptor:
    """First-order description of a single op-input read, derived from its memory config.

    This is the regime-agnostic bridge between a Polaris op and the physics model.
    It captures *where* the data lives and *how* it fans in, so the same
    :func:`predict_read_latency` skeleton can be parameterized per memory config
    (DRAM-interleaved, DRAM-sharded, L1-sharded, ...) instead of special-casing each.

    Fields are populated by ``Device`` from each input tensor's ``MemoryConfig``
    (``buffer_type`` + ``memory_layout`` + ``shard_spec``); when no memory config is
    present (the TTSIM/ONNX path) the caller falls back to the default regime.
    """

    #: Canonical memory tag, e.g. ``"DRAM_INTERLEAVED"`` / ``"L1_BLOCK_SHARDED"``.
    #: Matches ``MemoryConfig.to_canonical_memory_tag()`` (and the LUT key).
    regime: str
    #: Total bytes read for this input.
    total_bytes: float
    #: Per-transaction (page/shard) size N in bytes.
    page_bytes: float
    #: Parallel memory endpoints supplying data: DRAM channels (interleaved) or
    #: producer-core count (sharded).
    n_sources: int
    #: Cores issuing reads before the barrier (participating grid).
    n_consumers: int
    #: NoC hop distance; ``None`` uses the regime default, ``0`` means no NoC.
    hops: int | None = None
    #: Shard already resident on the consuming core (L1) -> reads never hit the NoC.
    is_local: bool = False

    def queue_depth(self) -> int:
        """Reads issued per consuming core (Q), floored at 1."""
        if self.page_bytes <= 0:
            return 1
        total_pages = max(1, round(self.total_bytes / self.page_bytes))
        return max(1, total_pages // max(1, self.n_consumers))


@dataclass(frozen=True, kw_only=True)
class ReadLatencyBreakdown:
    """Component split of the binding arm of :func:`predict_read_latency`.

    The read-latency model is ``max(issue_bound, transport_bound)``; each arm is a sum
    of a fixed head/tail plus a size-dependent delivery term. For timing integration we
    need to separate the part that is *exposed* (on the critical path even when the op's
    body overlaps compute) from the part that is *hideable* (steady-state streaming,
    already captured by Polaris' bytes/bandwidth memory model). This dataclass reports
    the binding arm's terms so callers (Option C) can add only the exposed latency and
    drop ``delivery`` to avoid double-counting streaming.

    All values are ``fclk`` cycles. Invariant: ``base + tdetect + issue_cost + delivery
    == tlat`` for the binding arm.
    """

    #: max(issue_bound, transport_bound) — identical to predict_read_latency().
    tlat: float
    #: Which arm bound: ``'issue'`` or ``'transport'``.
    binding: str
    #: Fixed fill latency to first data (Tdram + NoC round-trip). Exposed.
    base: float
    #: Barrier-clear tail. Exposed. (0 in the transport arm, which omits it.)
    tdetect: float
    #: Serial per-read issue cost of the binding arm. Exposed.
    issue_cost: float
    #: Streaming/delivery term of the binding arm. Hideable (overlaps compute / the
    #: bandwidth model), so it is NOT added on top in the additive-latency timing path.
    delivery: float

    @property
    def exposed(self) -> float:
        """Non-overlappable latency: fill + drain + serial issue (fclk cycles)."""
        return self.base + self.tdetect + self.issue_cost

    @property
    def hideable(self) -> float:
        """Streaming delivery that overlaps compute / duplicates the bandwidth model."""
        return self.delivery


def read_latency_breakdown_for_source(
    desc: ReadSourceDescriptor,
    *,
    regimes: dict[str, ReadLatencyConfig],
    default_regime: str,
) -> ReadLatencyBreakdown:
    """Regime-aware :class:`ReadLatencyBreakdown` for one op input (see
    :func:`predict_read_latency_for_source` for regime/fallback semantics)."""
    if desc.total_bytes <= 0 or desc.page_bytes <= 0:
        return ReadLatencyBreakdown(
            tlat=0.0, binding='issue', base=0.0, tdetect=0.0, issue_cost=0.0, delivery=0.0
        )
    cfg = regimes.get(desc.regime) or regimes[default_regime]
    hops = 0 if desc.is_local else desc.hops
    return predict_read_latency_breakdown(
        desc.page_bytes,
        Q=desc.queue_depth(),
        hops=hops,
        num_channels=desc.n_sources,
        cfg=cfg,
    )


def predict_read_latency_for_source(
    desc: ReadSourceDescriptor,
    *,
    regimes: dict[str, ReadLatencyConfig],
    default_regime: str,
) -> float:
    """Regime-aware read latency (``fclk`` cycles) for one op input.

    Selects per-regime hardware constants from ``regimes`` (falling back to
    ``default_regime`` when the descriptor's regime is uncalibrated or absent, which
    also covers the TTSIM/ONNX path), derives ``Q`` from the descriptor, and evaluates
    the shared :func:`predict_read_latency` model. ``is_local`` reads collapse the NoC
    round-trip (``hops -> 0``); their regime params are expected to zero ``tdram_cyc``
    (SRAM has no DRAM access bucket), so the same formula degrades gracefully to an
    L1-resident cost without a separate code path.
    """
    return read_latency_breakdown_for_source(
        desc, regimes=regimes, default_regime=default_regime
    ).tlat


def predict_read_latency(
    N: float,
    Q: int = 1,
    *,
    hops: int | None = None,
    num_channels: int | None = None,
    cfg: ReadLatencyConfig,
) -> float:
    """Predict read latency in ``fclk`` cycles for ``Q`` reads of ``N`` bytes each.

    Args:
        N: transaction (page) size in bytes for a single ``noc_async_read``.
        Q: number of reads issued (per core) before the barrier. ``Q = 1`` and
            ``num_channels = 1`` gives the Layer-1 unary case.
        hops: gating-channel hop distance; defaults to ``cfg.default_hops``.
        num_channels: DRAM channels the reads interleave across; defaults to
            ``cfg.num_dram_channels``. Pass ``1`` for the single-channel case.
        cfg: hardware constants.

    Returns:
        Predicted latency in ``fclk`` cycles (float).
    """
    return predict_read_latency_breakdown(
        N, Q, hops=hops, num_channels=num_channels, cfg=cfg
    ).tlat


def predict_read_latency_breakdown(
    N: float,
    Q: int = 1,
    *,
    hops: int | None = None,
    num_channels: int | None = None,
    cfg: ReadLatencyConfig,
) -> ReadLatencyBreakdown:
    """Decomposed form of :func:`predict_read_latency`.

    Evaluates both arms, returns the binding one's component split (``base``,
    ``tdetect``, ``issue_cost``, ``delivery``). ``tlat`` equals
    :func:`predict_read_latency` exactly (same expressions, same ``max``). The
    transport arm intentionally omits ``tdetect`` (``tdetect=0``), matching the
    original single-line formula. See :class:`ReadLatencyBreakdown`.
    """
    if N <= 0:
        return ReadLatencyBreakdown(
            tlat=0.0, binding='issue', base=0.0, tdetect=0.0, issue_cost=0.0, delivery=0.0
        )
    hops = cfg.default_hops if hops is None else hops
    num_channels = cfg.num_dram_channels if num_channels is None else num_channels
    Q = max(1, int(Q))
    num_channels = max(1, int(num_channels))

    noc_roundtrip = 2.0 * hops * cfg.chop_cyc_per_hop
    base = cfg.tdram_cyc + noc_roundtrip

    # Issue-bound arm: serial per-read issue dominates (small N / large Q).
    issue_issue_cost = cfg.delta_issue_cyc * Q
    issue_delivery = N / cfg.b_channel_bpc
    issue_bound = issue_issue_cost + base + cfg.tdetect_cyc + issue_delivery

    # Transport-bound arm: NoC-inbound-limited streaming dominates (large N*Q).
    n_star = math.ceil(cfg.noc_inbound_bpc / cfg.b_channel_bpc)
    b_eff = min(min(Q, num_channels) * cfg.b_channel_bpc, cfg.noc_inbound_bpc)
    transport_issue_cost = min(Q, n_star) * cfg.delta_issue_cyc
    transport_delivery = (Q * N) / b_eff
    transport_bound = transport_issue_cost + base + transport_delivery

    # Tie -> issue arm, matching max(issue_bound, transport_bound).
    if issue_bound >= transport_bound:
        return ReadLatencyBreakdown(
            tlat=issue_bound, binding='issue', base=base,
            tdetect=cfg.tdetect_cyc, issue_cost=issue_issue_cost, delivery=issue_delivery,
        )
    return ReadLatencyBreakdown(
        tlat=transport_bound, binding='transport', base=base,
        tdetect=0.0, issue_cost=transport_issue_cost, delivery=transport_delivery,
    )


def predict_read_latency_unary(
    N: float,
    *,
    cfg: ReadLatencyConfig,
) -> float:
    """O2O Layer-1 (unary) closed form: single core, single channel, ``Q = 1``.

    Implements the conf-page "Model Validation" formula exactly::

        Tlat = Tfixed + Trecv
        Nflits = ceil(N / Wflit)
        N <= 32 KB:  Trecv = (Nflits - 1) * 2.8
        N >  32 KB:  Trecv = (512 - 1) * 2.8 + (Nflits - 512) * 1.83

    This is the reference for validating against the page's 64 B - 512 KB sweep
    (``test_bw_and_latency -m 1 -l -p <N> -i 1000``). Unlike
    :func:`predict_read_latency`, it models the >32 KB pipelined ("dual-rate")
    delivery regime, so it tracks the full single-read sweep. The two functions
    agree for small single reads but diverge above ~16 KB.
    """
    if N <= 0:
        return 0.0
    nflits = math.ceil(N / cfg.flit_bytes)
    knee = cfg.recv_knee_flits
    if nflits <= knee:
        trecv = (nflits - 1) * cfg.recv_cyc_per_flit_lo
    else:
        trecv = (
            (knee - 1) * cfg.recv_cyc_per_flit_lo
            + (nflits - knee) * cfg.recv_cyc_per_flit_hi
        )
    return cfg.tfixed_unary_cyc + trecv


def effective_bandwidth_bpc(
    N: float,
    Q: int = 1,
    *,
    hops: int | None = None,
    num_channels: int | None = None,
    cfg: ReadLatencyConfig,
) -> float:
    """Effective read bandwidth (bytes / fclk cycle) = ``Q * N / Tlat``."""
    tlat = predict_read_latency(
        N, Q, hops=hops, num_channels=num_channels, cfg=cfg
    )
    if tlat <= 0:
        return 0.0
    return (Q * N) / tlat
