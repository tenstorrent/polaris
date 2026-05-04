# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side device performance reporting.

Provides the same dict-shaped output as tt-metal's ``run_device_perf`` so
that HW and Polaris perf results can be compared side-by-side.  Unlike HW,
Polaris is deterministic — cycle estimates come from the analytical model
rather than Tracy profiling, so MIN/MAX/AVG are identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from ttsim.back.device import Device as BackendDevice
from ttsim.config import get_arspec_from_yaml, get_wlmapspec_from_yaml
from ttsim.config.wl2archmap import WL2ArchMap, WL2ArchTypeSpec


_DEFAULT_ARCHSPEC = "config/tt_wh.yaml"
_DEFAULT_WLMAPSPEC = "config/wl2archmapping.yaml"
_DEFAULT_DEVNAME = "n150"


def run_device_perf_polaris(
    test_fn: Callable,
    batch_size: int,
    cols: list[str],
    *,
    archspec: str = _DEFAULT_ARCHSPEC,
    wlmapspec: str = _DEFAULT_WLMAPSPEC,
    devname: str = _DEFAULT_DEVNAME,
) -> dict[str, Any]:
    """Run *test_fn* on Polaris and return perf metrics in the same shape as
    tt-metal's ``run_device_perf``.

    Parameters
    ----------
    test_fn:
        A callable ``(device, batch_size) -> device`` that builds the op graph
        on a Polaris front-end device and returns it.
    batch_size:
        Batch size forwarded to *test_fn* and used for samples/s calculation.
    cols:
        Column name prefixes (e.g. ``["DEVICE FW", "DEVICE KERNEL"]``).
        Used to construct the result dict keys, matching the HW format.
    archspec, wlmapspec, devname:
        Paths to Polaris config files and target device name.
    """
    from ttsim.front.ttnn.device import open_device, close_device

    fe_device = open_device()
    fe_device = test_fn(fe_device, batch_size=batch_size)
    wlgraph = fe_device.get_graph()
    close_device(fe_device)

    _, devspec = get_arspec_from_yaml(archspec)
    dev_obj = devspec[devname]
    be_device = BackendDevice(dev_obj)

    if WL2ArchTypeSpec.has_instance():
        wlmap = WL2ArchMap.from_yaml(wlmapspec)
    else:
        wlmap = get_wlmapspec_from_yaml(wlmapspec)
    be_device.execute_graph(wlgraph, wlmap)

    summary = be_device.get_exec_stats(wlgraph, batch_size)

    tot_duration_ns = summary["tot_msecs"] * 1e6
    samples_per_s = summary["perf_projection"]

    results: dict[str, Any] = {}
    for col in cols:
        d_col = f"{col} DURATION [ns]"
        s_col = f"{col} SAMPLES/S"
        results[f"AVG {d_col}"] = tot_duration_ns
        results[f"MIN {d_col}"] = tot_duration_ns
        results[f"MAX {d_col}"] = tot_duration_ns
        results[f"AVG {s_col}"] = samples_per_s
        results[f"MIN {s_col}"] = samples_per_s
        results[f"MAX {s_col}"] = samples_per_s

    results["_polaris_summary"] = summary

    logger.info(
        "\nPolaris analytical device perf"
        "\n{}\n",
        json.dumps(
            {k: v for k, v in results.items() if not k.startswith("_")},
            indent=4,
        ),
    )
    return results


def prep_device_perf_report_polaris(
    model_name: str,
    batch_size: int,
    post_processed_results: dict[str, Any],
    *,
    output_dir: str | Path = "__polaris_perf_reports",
    comments: str = "",
) -> Path:
    """Write a JSON report of Polaris perf results.

    Returns the path to the written report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "model_name": model_name,
        "batch_size": batch_size,
        "source": "polaris_analytical",
        "comments": comments,
        "results": {
            k: v
            for k, v in post_processed_results.items()
            if not k.startswith("_")
        },
    }

    summary = post_processed_results.get("_polaris_summary")
    if summary is not None:
        report["polaris_summary"] = {
            "tot_ideal_cycles": summary.get("tot_ideal_cycles"),
            "tot_ideal_msecs": summary.get("tot_ideal_msecs"),
            "tot_cycles": summary.get("tot_cycles"),
            "tot_msecs": summary.get("tot_msecs"),
            "ideal_throughput": summary.get("ideal_throughput"),
            "perf_projection": summary.get("perf_projection"),
            "fits_device": summary.get("fits_device"),
        }

    report_path = output_dir / f"{model_name}-polaris-device-perf.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, default=str)

    logger.info("Polaris device perf report written to {}", report_path)
    return report_path
