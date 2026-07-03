#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate the unary read-latency model against the O2O conf-page Layer 1 table.

Source: "O2O: Read Latency Modeling" (Confluence 2433646619), "Model Validation"
section - single core, single channel, queue depth Q=1, swept 64 B - 512 KB via
``test_bw_and_latency -m 1 -l -p <N> -i 1000``.

Rows are (N bytes, conf "Predicted" cyc, conf "Measured" cyc).
"""

import pytest

from ttsim.back.read_latency import predict_read_latency_unary

# (N, conf_predicted, conf_measured)
CONF_LAYER1 = [
    (64,        438.0,   443.0),
    (1024,      480.0,   500.0),
    (2048,      525.0,   541.0),
    (4096,      614.0,   625.0),
    (8192,      794.0,   833.0),
    (16384,    1152.0,  1153.0),
    (32768,    1869.0,  1836.0),
    (65536,    2805.0,  2771.0),
    # 128 KB: the conf "Predicted" cell reads 4714, but 438 + 511*2.8 + 1536*1.83
    # = 4679.68, which is what the formula (and the measured 4679) give. Treated
    # as a typo on the page; validated against the formula/measured value instead.
    (131072,   4680.0,  4679.0),
    (262144,   8428.0,  8347.0),
    (524288,  15923.0, 15866.0),
]

PRED_TOL_CYC = 1.0   # formula must reproduce the page's predicted column exactly
MEAS_REL_TOL = 0.06  # model vs hardware (page's own error is <= ~5%)


@pytest.mark.parametrize("N,conf_pred,conf_meas", CONF_LAYER1)
def test_unary_reproduces_conf_predicted(N, conf_pred, conf_meas, bh_read_latency_cfg):
    pred = predict_read_latency_unary(N, cfg=bh_read_latency_cfg)
    assert abs(pred - conf_pred) <= PRED_TOL_CYC, (
        f"N={N}: formula={pred:.2f} != conf predicted {conf_pred}"
    )


@pytest.mark.parametrize("N,conf_pred,conf_meas", CONF_LAYER1)
def test_unary_matches_measured(N, conf_pred, conf_meas, bh_read_latency_cfg):
    pred = predict_read_latency_unary(N, cfg=bh_read_latency_cfg)
    rel_err = abs(pred - conf_meas) / conf_meas
    assert rel_err <= MEAS_REL_TOL, (
        f"N={N}: pred={pred:.1f} measured={conf_meas} rel_err={rel_err:.1%}"
    )


def test_dual_rate_knee_at_32kb(bh_read_latency_cfg):
    cfg = bh_read_latency_cfg
    # The pipelined (1.83 cyc/flit) regime only kicks in above 32 KB.
    assert predict_read_latency_unary(32768, cfg=cfg) == pytest.approx(438 + 511 * 2.8)
    # First flit past the knee uses the high rate.
    step = predict_read_latency_unary(32768 + 64, cfg=cfg) - predict_read_latency_unary(32768, cfg=cfg)
    assert step == pytest.approx(1.83, abs=1e-6)
