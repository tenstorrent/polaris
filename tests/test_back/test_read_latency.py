#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate the O2O read-latency model against the Blackhole DRAM-interleaved
read microbenchmark.

Ground-truth rows are taken verbatim from tt-metal's
``tests/.../data_movement/data/blackhole/DRAM Interleaved Page Read Numbers.csv``
(columns: transaction size N (bytes), number of transactions Q, measured
latency in cycles). Embedded here so the test is hermetic.
"""

import math

import pytest

from ttsim.back.read_latency import (
    TILE_ELEMS,
    ReadLatencyBreakdown,
    effective_bandwidth_bpc,
    predict_read_latency,
    predict_read_latency_breakdown,
)

# (N bytes, Q transactions, measured latency cycles) - Blackhole, riscv_1.
CSV_ROWS = [
    (64, 1, 472.0), (128, 1, 482.0), (256, 1, 553.0), (512, 1, 498.0),
    (1024, 1, 529.0), (2048, 1, 603.0), (4096, 1, 653.0), (8192, 1, 825.0),
    (16384, 1, 1169.0),
    (64, 4, 559.0), (128, 4, 567.0), (256, 4, 577.0), (512, 4, 584.0),
    (1024, 4, 616.0), (2048, 4, 681.0), (4096, 4, 806.0), (8192, 4, 1074.0),
    (16384, 4, 1601.0),
    (64, 16, 1055.0), (128, 16, 1052.0), (256, 16, 1165.0), (512, 16, 1074.0),
    (1024, 16, 1108.0), (2048, 16, 1190.0), (4096, 16, 1639.0),
    (8192, 16, 2676.0), (16384, 16, 4795.0),
    (64, 64, 2871.0), (128, 64, 2880.0), (256, 64, 2897.0), (512, 64, 2893.0),
    (1024, 64, 2930.0), (2048, 64, 3035.0), (4096, 64, 4802.0),
    (8192, 64, 9029.0), (16384, 64, 17436.0),
    (64, 256, 10209.0), (128, 256, 10160.0), (256, 256, 10180.0),
    (512, 256, 10186.0), (1024, 256, 10216.0), (2048, 256, 10527.0),
    (4096, 256, 17892.0), (8192, 256, 34917.0), (16384, 256, 67609.0),
]

# Interleaved microbenchmark uses all 7 p100a DRAM channels.
NUM_CHANNELS = 7
REL_TOL = 0.15


@pytest.mark.parametrize("N,Q,measured", CSV_ROWS)
def test_predicted_latency_matches_microbenchmark(N, Q, measured, bh_read_latency_cfg):
    predicted = predict_read_latency(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    rel_err = abs(predicted - measured) / measured
    assert rel_err <= REL_TOL, (
        f"N={N} Q={Q}: predicted={predicted:.1f} measured={measured} "
        f"rel_err={rel_err:.1%} > {REL_TOL:.0%}"
    )


def test_mean_error_is_small(bh_read_latency_cfg):
    errs = [
        abs(predict_read_latency(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg) - m) / m
        for N, Q, m in CSV_ROWS
    ]
    mean_err = sum(errs) / len(errs)
    assert mean_err <= 0.05, f"mean rel err {mean_err:.1%} too high"


def test_layer1_is_unary_special_case(bh_read_latency_cfg):
    # Layer 1 == single channel, Q=1. Q=1 interleaved touches one channel anyway,
    # so the two should agree for Q=1.
    for N, Q, _ in CSV_ROWS:
        if Q != 1:
            continue
        unary = predict_read_latency(N, 1, num_channels=1, cfg=bh_read_latency_cfg)
        interleaved_q1 = predict_read_latency(N, 1, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
        assert math.isclose(unary, interleaved_q1, rel_tol=1e-9)


def test_latency_monotonic_in_size_and_queue(bh_read_latency_cfg):
    # Larger transactions and deeper queues never reduce latency.
    for Q in (1, 4, 16, 64, 256):
        prev = 0.0
        for N in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            cur = predict_read_latency(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
            assert cur >= prev - 1e-9
            prev = cur
    for N in (64, 4096, 16384):
        prev = 0.0
        for Q in (1, 4, 16, 64, 256):
            cur = predict_read_latency(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
            assert cur >= prev - 1e-9
            prev = cur


def test_effective_bandwidth_saturates_near_noc_ceiling(bh_read_latency_cfg):
    cfg = bh_read_latency_cfg
    # Deep queue + large pages should approach the NoC inbound ceiling.
    bw = effective_bandwidth_bpc(16384, 256, num_channels=NUM_CHANNELS, cfg=cfg)
    assert 0.85 * cfg.noc_inbound_bpc <= bw <= cfg.noc_inbound_bpc + 1.0


def test_tile_page_size_default():
    assert TILE_ELEMS == 1024


def test_zero_bytes_is_zero_latency(bh_read_latency_cfg):
    assert predict_read_latency(0, 4, cfg=bh_read_latency_cfg) == 0.0


# --- ReadLatencyBreakdown (decomposition for timing integration) ------------

@pytest.mark.parametrize("N,Q,_measured", CSV_ROWS)
def test_breakdown_tlat_matches_scalar(N, Q, _measured, bh_read_latency_cfg):
    # The decomposed form must return the exact same total as the scalar function.
    scalar = predict_read_latency(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    bd = predict_read_latency_breakdown(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    assert isinstance(bd, ReadLatencyBreakdown)
    assert bd.tlat == scalar


@pytest.mark.parametrize("N,Q,_measured", CSV_ROWS)
def test_breakdown_components_sum_to_tlat(N, Q, _measured, bh_read_latency_cfg):
    # Invariant: base + tdetect + issue_cost + delivery == tlat (binding arm).
    bd = predict_read_latency_breakdown(N, Q, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    total = bd.base + bd.tdetect + bd.issue_cost + bd.delivery
    assert math.isclose(total, bd.tlat, rel_tol=1e-9)
    assert math.isclose(bd.exposed + bd.hideable, bd.tlat, rel_tol=1e-9)
    assert bd.binding in ("issue", "transport")


def test_breakdown_binding_regimes(bh_read_latency_cfg):
    # Small N with a deep queue is issue-bound; large N*Q streams -> transport-bound.
    small = predict_read_latency_breakdown(64, 256, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    assert small.binding == "issue"
    assert small.tdetect == bh_read_latency_cfg.tdetect_cyc
    large = predict_read_latency_breakdown(16384, 256, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    assert large.binding == "transport"
    # Transport arm omits the barrier tail.
    assert large.tdetect == 0.0


def test_breakdown_exposed_is_size_independent_when_issue_bound(bh_read_latency_cfg):
    # In the issue-bound arm, exposed latency (base+tdetect+delta*Q) does not depend
    # on N; only the hideable delivery (N/b_channel) grows with N.
    a = predict_read_latency_breakdown(64, 64, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    b = predict_read_latency_breakdown(256, 64, num_channels=NUM_CHANNELS, cfg=bh_read_latency_cfg)
    assert a.binding == "issue" and b.binding == "issue"
    assert math.isclose(a.exposed, b.exposed, rel_tol=1e-9)
    assert b.delivery > a.delivery


def test_breakdown_zero_bytes(bh_read_latency_cfg):
    bd = predict_read_latency_breakdown(0, 4, cfg=bh_read_latency_cfg)
    assert bd.tlat == 0.0
    assert bd.exposed == 0.0 and bd.hideable == 0.0
