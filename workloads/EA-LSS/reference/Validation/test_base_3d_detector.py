#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for Base3DDetector TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_base_3d_detector.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.base_3d_detector import Base3DDetector
from reference.Validation.ttsim_utils import print_header, print_test


def test_base_passthrough():
    print_header("Test 1: Base3DDetector identity passthrough")
    m   = Base3DDetector("base_v")
    x   = _from_shape("base_x", [1, 3, 64, 64])
    out = m(x)
    ok  = list(out.shape) == [1, 3, 64, 64]
    print_test("output shape == input shape [1,3,64,64]", f"got {list(out.shape)}", ok)
    return ok


def test_base_params():
    print_header("Test 2: Base3DDetector analytical_param_count == 0")
    m  = Base3DDetector("base_p")
    p  = m.analytical_param_count()
    ok = p == 0
    print_test("analytical_param_count == 0", f"got {p}", ok)
    return ok


def test_base_multiple_instances():
    print_header("Test 3: Multiple Base3DDetector instances (independent naming)")
    a = Base3DDetector("alpha")
    b = Base3DDetector("beta")
    ok = (a.name == "alpha") and (b.name == "beta")
    print_test("Names are independent", f"a={a.name}, b={b.name}", ok)
    return ok


if __name__ == "__main__":
    results = [test_base_passthrough(), test_base_params(), test_base_multiple_instances()]
    n_pass  = sum(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{len(results)} tests passed")
    if n_pass < len(results):
        sys.exit(1)
