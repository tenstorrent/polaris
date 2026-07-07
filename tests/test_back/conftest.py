#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for read-latency tests.

The read-latency HW constants (``tdram_cyc``, ``tdetect_cyc``, ``noc_inbound_bpc``)
live solely in the arch YAML (``config/tt_bh.yaml`` memory block's ``read_latency``
section). These fixtures build the model config from that YAML so the tests
validate the *shipped* calibration instead of duplicating the numbers.
"""

from pathlib import Path

import pytest

from ttsim.config import get_arspec_from_yaml
from ttsim.back.read_latency import ReadLatencyConfig

POLARIS_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def bh_read_latency_cfg() -> ReadLatencyConfig:
    """ReadLatencyConfig for Blackhole p100a DRAM_INTERLEAVED, sourced from config/tt_bh.yaml.

    The arch YAML now stores read-latency constants per memory-config regime; this
    fixture returns the calibrated ``DRAM_INTERLEAVED`` regime (the one validated
    against the microbenchmark) so the existing unit tests are unchanged.
    """
    _, packages = get_arspec_from_yaml(POLARIS_ROOT / "config" / "tt_bh.yaml")
    mem = packages["p100a"].get_ipgroup("memory")
    rl = mem.ipobj.read_latency
    assert rl is not None, "config/tt_bh.yaml gddr6_bh is missing a read_latency block"
    params = rl.regimes[rl.default_regime]
    return ReadLatencyConfig(num_dram_channels=mem.num_units, **params.model_dump())
