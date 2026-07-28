# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""SDPA roofline (TEN-4716) driven through the ttsim device cost path."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ttsim.back.device import Device
from ttsim.perf.roofline_sdpa import SdpaConfig, predict, sdpa_perf_stats, ARCH_BH


# Measured per-core MATH cycles on Blackhole P100 (SDPA/verif_data sweeps).
MEASURED_MATH = {
    "baseline": {4096: 875_667, 8192: 3_442_080, 32768: 54_446_938},
    "wan":      {4096: 1_750_702, 32768: 112_393_664},
}


def _baseline_cfg(S):
    return SdpaConfig(S=S, num_cores=110, arch=ARCH_BH)


def _wan_cfg(S):
    return SdpaConfig(S=S, num_heads=40, q_chunk=256, k_chunk=256, is_causal=False,
                      input_dtype="bfloat16", accum_dtype="bfloat16",
                      exp_approx_mode=False, num_cores=110, arch=ARCH_BH)


class _MockIPGroup:
    def __init__(self, iptype):
        self.iptype = iptype


class _MockSimConfig:
    def __init__(self, freq_mhz=1350):
        self._freq_mhz = freq_mhz
        self.devname = "BH-test"
        self.name = "bh_test_device"
        self.ipgroups = [_MockIPGroup("compute"), _MockIPGroup("memory")]

    def frequency(self, pipe, units="MHz"):
        return self._freq_mhz

    def mem_frequency(self, units="MHz"):
        return self._freq_mhz

    def mem_size(self, units="GB"):
        return 32.0

    def peak_bandwidth(self, freq_units="GHz"):
        return 1000.0

    def peak_bandwidth_per_cycle(self):
        return 10.0

    def peak_flops(self, pipe, instr, precision, mul_factor=1):
        return 100.0

    def peak_ipc(self, pipe, instr, precision):
        return 128.0


def _sdpa_op(name, cfg):
    return SimpleNamespace(
        name=name, optype="SDPA", uses_compute_pipe="matrix", precision="bfp8",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats=sdpa_perf_stats(cfg),
    )


@pytest.mark.unit
@pytest.mark.parametrize("S", [1024, 4096, 8192, 32768])
def test_fused_compute_cycles_override_is_honored(S):
    device = Device(_MockSimConfig())
    cfg = _baseline_cfg(S)
    op = _sdpa_op(f"sdpa_S{S}", cfg)
    device.execute_op(op)
    assert op.compute_cycles == int(math.ceil(predict(cfg).compute_latency_cycles))
    ipc_path = sum(math.ceil(c / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))
                   for c in op.perf_stats["instrs"].values())
    assert op.compute_cycles != ipc_path


@pytest.mark.unit
def test_generic_instrs_path_unchanged_without_override():
    device = Device(_MockSimConfig())
    op = SimpleNamespace(
        name="plain_mm", optype="MatMul", uses_compute_pipe="matrix", precision="fp32",
        repeat_count=1, removed_in_optimization=False, fused_in_optimization=False,
        fused_with_op=None, fused_op_cycles=None, exec_stats={},
        compute_cycles=0, mem_rd_cycles=0, mem_wr_cycles=0,
        mem_rd_cycles_fractional=0.0, mem_wr_cycles_fractional=0.0,
        perf_stats={"inBytes": 4000, "outBytes": 4000, "instrs": {"mac": 4096}},
    )
    device.execute_op(op)
    assert op.compute_cycles == math.ceil(4096 / (128.0 * device.DG_COMPUTE_UTIL_CONSTANT))


@pytest.mark.unit
@pytest.mark.parametrize("shape,S", [("baseline", 4096), ("baseline", 8192),
                                     ("baseline", 32768), ("wan", 4096), ("wan", 32768)])
def test_roofline_tracks_measured_math(shape, S):
    cfg = _baseline_cfg(S) if shape == "baseline" else _wan_cfg(S)
    pred = predict(cfg).math_active_cycles
    meas = MEASURED_MATH[shape][S]
    rel = abs(pred - meas) / meas
    assert rel <= 0.12, f"{shape} S={S}: pred={pred} meas={meas} rel={rel:.3f}"


@pytest.mark.unit
def test_perf_stats_has_all_keys_downstream_reads():
    # hlmstats / device stats read these unconditionally; missing any is a KeyError at runtime.
    ps = sdpa_perf_stats(_baseline_cfg(4096))
    required = {"inBytes", "outBytes", "inElems", "outElems", "inParamCount",
                "inActCount", "outActCount", "instrs", "fused_compute_cycles"}
    assert required.issubset(ps), f"missing: {required - set(ps)}"
    assert ps["fused_compute_cycles"] > 0 and ps["inBytes"] > 0
    assert ps["inElems"] == 0  # sentinel so the bytes-per-element precision check is skipped


@pytest.mark.unit
def test_predict_rejects_non_tile_aligned_shapes():
    with pytest.raises(AssertionError):
        predict(SdpaConfig(S=4096, head_dim=100, num_cores=110, arch=ARCH_BH))
    with pytest.raises(AssertionError):
        predict(SdpaConfig(S=5000, num_cores=110, arch=ARCH_BH))
    with pytest.raises(AssertionError):
        predict(SdpaConfig(S=4096, head_dim=128, v_head_dim=100, num_cores=110, arch=ARCH_BH))


@pytest.mark.unit
def test_v_head_dim_default_is_symmetric():
    # v_head_dim=0 must reproduce the symmetric head_dim result exactly.
    sym = predict(SdpaConfig(S=4096, head_dim=128, num_cores=110, arch=ARCH_BH))
    explicit = predict(SdpaConfig(S=4096, head_dim=128, v_head_dim=128, num_cores=110, arch=ARCH_BH))
    assert sym.fpu_matmul_cycles == explicit.fpu_matmul_cycles
    assert sym.compute_latency_cycles == explicit.compute_latency_cycles


@pytest.mark.unit
def test_asymmetric_v_head_dim_scales_pv_matmul():
    # MLA-style: v_head_dim > head_dim raises the P*V matmul, so total matmul cycles grow.
    base = predict(SdpaConfig(S=4096, head_dim=192, num_cores=110, arch=ARCH_BH))
    mla = predict(SdpaConfig(S=4096, head_dim=192, v_head_dim=512, num_cores=110, arch=ARCH_BH))
    assert mla.fpu_matmul_cycles > base.fpu_matmul_cycles
    # matmul cycles scale as (dct_qk + dct_v): (6+16) vs (6+6)
    assert mla.fpu_matmul_cycles == round(base.fpu_matmul_cycles * (6 + 16) / (6 + 6))
