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
    if N <= 0:
        return 0.0
    hops = cfg.default_hops if hops is None else hops
    num_channels = cfg.num_dram_channels if num_channels is None else num_channels
    Q = max(1, int(Q))
    num_channels = max(1, int(num_channels))

    noc_roundtrip = 2.0 * hops * cfg.chop_cyc_per_hop
    base = cfg.tdram_cyc + noc_roundtrip

    # Issue-bound arm: serial per-read issue dominates (small N / large Q).
    issue_bound = (
        cfg.delta_issue_cyc * Q + base + cfg.tdetect_cyc + N / cfg.b_channel_bpc
    )

    # Transport-bound arm: NoC-inbound-limited streaming dominates (large N*Q).
    n_star = math.ceil(cfg.noc_inbound_bpc / cfg.b_channel_bpc)
    b_eff = min(min(Q, num_channels) * cfg.b_channel_bpc, cfg.noc_inbound_bpc)
    head_issue = min(Q, n_star) * cfg.delta_issue_cyc
    transport_bound = head_issue + base + (Q * N) / b_eff

    return max(issue_bound, transport_bound)


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
