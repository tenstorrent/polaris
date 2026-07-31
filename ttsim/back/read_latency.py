#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""First-order analytical model for DRAM read latency.

Predicts the latency (in NoC/fabric clock cycles, fclk) of a Tensix core issuing
Q ``noc_async_read`` transactions of N bytes each from DRAM, followed by a barrier.

Small transactions are latency-bound rather than bandwidth-bound: the fixed DRAM
access bucket, the NoC round-trip and the per-read issue cost dominate, and a flat
bytes/bandwidth model underestimates their cost badly. Device takes the max of this
prediction and the bandwidth-limited cost, so large transactions are unaffected.

The latency is the larger of two arms (the binding constraint):

* issue-bound:     delta_issue * Q + Tdram + 2*hops*Chop + Tdetect + N/B_channel
* transport-bound: min(Q, n*) * delta_issue + Tdram + 2*hops*Chop + Q*N/B_eff

where B_eff = min(min(Q, num_channels) * B_channel, noc_inbound_bpc) and
n* = ceil(noc_inbound_bpc / B_channel) is the number of channels needed to
saturate the NoC inbound link.

Intentionally first-order: NoC congestion / VC contention, row-miss / refresh
penalties, and non-DRAM (L1-resident) sources are out of scope.

See doc/READ_LATENCY_MODEL.md for the calibration source, the rationale for the
max() integration, and what is deliberately not modeled.
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

    Every field is an arch-calibrated constant. The arch YAML is the single source
    of truth (config/tt_bh.yaml memory block's ``read_latency`` section, parsed by
    ``MemoryReadLatencyModel``) and must be supplied by the caller. Device injects
    them from the parsed arch spec; tests build the config from the same YAML.
    ``num_dram_channels`` is the sole exception: it comes from the memory IP's
    ``num_units`` (not the ``read_latency`` block).
    """

    #: DRAM access bucket (row-hit best case): row activate + CAS + burst.
    tdram_cyc: float
    #: Barrier-clear tail.
    tdetect_cyc: float
    #: Per-core NoC inbound ceiling (bytes / fclk cycle).
    noc_inbound_bpc: float
    #: Per-read issue cost (cyc/read) - slope of latency vs Q in the issue regime.
    delta_issue_cyc: float
    #: NoC cost per hop (cyc); round-trip contributes 2 * hops * chop.
    chop_cyc_per_hop: float
    #: Single-channel, single-stream effective rate (bytes / fclk cycle).
    b_channel_bpc: float
    #: Default gating-channel hop distance (h_gate from the O2O page).
    default_hops: int
    #: DRAM channels for interleaving (from memory IP num_units; p100a = 7).
    num_dram_channels: int


def predict_read_latency(
    N: float,
    Q: int = 1,
    *,
    hops: int | None = None,
    num_channels: int | None = None,
    cfg: ReadLatencyConfig,
) -> float:
    """Predict read latency in fclk cycles for Q reads of N bytes each.

    Args:
        N: transaction (page) size in bytes for a single noc_async_read.
        Q: number of reads issued (per core) before the barrier.
        hops: gating-channel hop distance; defaults to cfg.default_hops.
        num_channels: DRAM channels the reads interleave across; defaults to
            cfg.num_dram_channels. Pass 1 for the single-channel case.
        cfg: hardware constants.

    Returns:
        Predicted latency in fclk cycles (float).
    """
    if N <= 0:
        return 0.0
    hops = cfg.default_hops if hops is None else hops
    num_channels = max(1, int(num_channels if num_channels is not None else cfg.num_dram_channels))
    Q = max(1, int(Q))

    base = cfg.tdram_cyc + 2.0 * hops * cfg.chop_cyc_per_hop

    # Issue-bound arm: serial per-read issue dominates (small N / large Q).
    issue_bound = cfg.delta_issue_cyc * Q + base + cfg.tdetect_cyc + N / cfg.b_channel_bpc

    # Transport-bound arm: NoC-inbound-limited streaming dominates (large N*Q).
    n_star = math.ceil(cfg.noc_inbound_bpc / cfg.b_channel_bpc)
    # The min(Q, num_channels) factor is carried from the calibrated source formula but
    # cannot change the result under the shipped Blackhole constants: it only bites when
    # Q < num_channels and Q * b_channel < noc_inbound (i.e. Q <= 2), and at Q <= 2 this
    # arm reduces to issue_bound - tdetect and always loses the max(). Kept for fidelity
    # under a different calibration; do not expect a test to cover it.
    b_eff = min(min(Q, num_channels) * cfg.b_channel_bpc, cfg.noc_inbound_bpc)
    transport_bound = min(Q, n_star) * cfg.delta_issue_cyc + base + (Q * N) / b_eff

    return max(issue_bound, transport_bound)
