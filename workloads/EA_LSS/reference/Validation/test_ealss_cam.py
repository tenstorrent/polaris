#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Validation script for EALSS_CAM TTSim module.

Run from polaris root:
    python3 workloads/EA-LSS/Reference/Validation/test_ealss_cam.py
"""

import os, sys

_ealss_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_polaris_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for p in [_polaris_root, _ealss_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ttsim.front.functional.op import _from_shape
from ttsim_modules.ealss_cam import EALSS_CAM
from reference.Validation.ttsim_utils import print_header, print_test

# EALSS_CAM defaults: imc=512, lc_fusion=False, camera_stream=True
# No LiDAR path by default → fewer heavy modules than full EALSS
EXPECTED_PARAMS_APPROX = 72_700_000   # allow ±1M tolerance


def test_ealss_cam_params():
    print_header("Test 1: EALSS_CAM analytical_param_count (approx 72.8M)")
    m  = EALSS_CAM("ecam_ptest")
    p  = m.analytical_param_count()
    ok = abs(p - EXPECTED_PARAMS_APPROX) < 1_000_000
    print_test(f"EALSS_CAM params ≈ 72.8M (±1M)",
               f"got {p:,}", ok)
    return ok


def test_ealss_cam_forward_shape():
    print_header("Test 2: EALSS_CAM forward – output is a prediction dict")
    m   = EALSS_CAM("ecam_fwd")
    img = _from_shape("ecam_img", [6, 3, 256, 704])
    out = m(img)
    ok  = isinstance(out, dict) and "heatmap" in out
    print_test("EALSS_CAM forward returns prediction dict with 'heatmap'",
               f"keys={sorted(out.keys())}", ok)
    return ok


def test_ealss_cam_imc_larger():
    print_header("Test 3: EALSS_CAM imc=512 gives more params than imc=256 variant")
    m_256 = EALSS_CAM("ecam_imc256", imc=256)
    m_512 = EALSS_CAM("ecam_imc512", imc=512)
    p_256 = m_256.analytical_param_count()
    p_512 = m_512.analytical_param_count()
    ok    = p_512 > p_256
    print_test("imc=512 has more params than imc=256",
               f"imc256={p_256:,}, imc512={p_512:,}", ok)
    return ok


def test_ealss_cam_lc_fusion():
    print_header("Test 4: EALSS_CAM lc_fusion=True adds LiDAR path + fusion params")
    m_nofus = EALSS_CAM("ecam_nofus", lc_fusion=False)
    m_fused = EALSS_CAM("ecam_fused", lc_fusion=True)
    p_nofus = m_nofus.analytical_param_count()
    p_fused = m_fused.analytical_param_count()
    ok      = p_fused > p_nofus
    print_test("lc_fusion=True adds LiDAR backbone+neck+fusion params",
               f"no_fus={p_nofus:,}, fused={p_fused:,}", ok)
    return ok


if __name__ == "__main__":
    results = [
        test_ealss_cam_params(),
        test_ealss_cam_forward_shape(),
        test_ealss_cam_imc_larger(),
        test_ealss_cam_lc_fusion(),
    ]
    n_pass = sum(results)
    print(f"\n{'='*60}")
    print(f"RESULTS: {n_pass}/{len(results)} tests passed")
    if n_pass < len(results):
        sys.exit(1)
