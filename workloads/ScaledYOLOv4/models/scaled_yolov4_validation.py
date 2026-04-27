#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ScaledYOLOv4Validation – Polaris Workload with Integrated Validation Suite

Extends ScaledYOLOv4 (Model) to additionally execute both run_all.py validation
scripts found under:
  - workloads/ScaledYOLOv4/reference/common/run_all.py  (18 common modules)
  - workloads/ScaledYOLOv4/reference/yolo/run_all.py    (Detect + Model)

the first time the workload is instantiated.

All captured stdout / stderr from both runners is merged into a single
timestamped Markdown report located in
workloads/ScaledYOLOv4/validation_output/. The absolute path to that report
is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the
plain ScaledYOLOv4 Model workload – it produces the same forward-graph and
the same Polaris JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl scaled_yolov4_validation \\
        -o ODDIR_scaled_yolov4_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from workloads.ScaledYOLOv4.models.yolo import Model  # noqa: E402
from workloads.validation_helpers import (  # noqa: E402
    run_subprocess,
    is_markdown_output,
    write_simple_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path of the polaris workspace root (three levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../..")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "validation_output")
)

# The two run_all scripts to execute, in order
_RUN_ALL_SCRIPTS: list[tuple[str, str]] = [
    (
        "Common Modules (Conv, Bottleneck, SPP, …)",
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "reference",
                "common",
                "run_all.py",
            )
        ),
    ),
    (
        "YOLO Modules (Detect, Model)",
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "reference",
                "yolo",
                "run_all.py",
            )
        ),
    ),
]

# ---------------------------------------------------------------------------
# Runner descriptions
# ---------------------------------------------------------------------------

_RUNNER_DESCRIPTIONS: dict[str, str] = {
    "Common Modules (Conv, Bottleneck, SPP, …)": (
        "Runs 18 common-module validations comparing TTSim vs PyTorch: "
        "Conv, DWConv, Bottleneck, BottleneckCSP, BottleneckCSP2, "
        "DWConvLayer, ConvLayer, CombConvLayer, Upsample, MaxPool, "
        "Concat, Flatten, Classify, Focus, SPP, SPPCSP, VoVCSP, and "
        "ConvSig. Each module is validated for shape and numerical "
        "agreement (rtol=1e-5, atol=1e-6)."
    ),
    "YOLO Modules (Detect, Model)": (
        "Runs 2 YOLO-level module validations comparing TTSim vs PyTorch: "
        "Detect (multi-scale detection head with per-layer output shape "
        "and numerical checks) and Model (full ScaledYOLOv4 model "
        "end-to-end, with weights transferred from a PyTorch checkpoint, "
        "layer-by-layer shape and value comparison, rtol=1e-4, atol=1e-4)."
    ),
}

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _run_one_script(label: str, fpath: str, timeout: int = 600) -> tuple[int, str]:
    """Execute a run_all.py script as a subprocess."""
    return run_subprocess(fpath, _POLARIS_ROOT, timeout=timeout)


def _write_report(md_path: str, results: list) -> None:
    """Write the validation-report Markdown file."""
    # results from run_once are [(item, rc, output)] where item = (label, fpath)
    flat = [(item[0], item[1], rc, output) for item, rc, output in results]

    def _extra_meta(label, rc, output):
        script_path = dict((lbl, fp) for lbl, fp in _RUN_ALL_SCRIPTS).get(label, "")
        rel = os.path.relpath(script_path, _POLARIS_ROOT) if script_path else label
        return [f"**Script:** `{rel}`  \n"]

    write_simple_markdown(
        md_path,
        "ScaledYOLOv4 Validation Report",
        [(lbl, rc, out) for lbl, _fp, rc, out in flat],
        _RUNNER_DESCRIPTIONS,
        default_description="Runs ScaledYOLOv4 module validations and reports pass/fail status.",
        label_key_fn=lambda x: x,
        extra_meta_fn=_extra_meta,
    )


def _run_validation_tests() -> str:
    """Run both run_all.py scripts and write a single Markdown report.

    Returns the **absolute path** to the generated Markdown file.
    Subsequent calls are no-ops; the cached path is returned instead.
    """
    global _VALIDATION_DONE, _VALIDATION_MD_PATH  # noqa: PLW0603

    if _VALIDATION_DONE and _VALIDATION_MD_PATH is not None:
        return _VALIDATION_MD_PATH

    import datetime

    os.makedirs(_OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_OUTPUT_BASE, f"validation_report_{timestamp}.md")

    print(f"\n{'=' * 72}")
    print(
        f"[ScaledYOLOv4Validation] Running {len(_RUN_ALL_SCRIPTS)} validation suite(s)…"
    )
    print(f"{'=' * 72}\n")

    results_flat: list[tuple[str, str, int, str]] = []
    for label, fpath in _RUN_ALL_SCRIPTS:
        print(f"  › Running: {label} …", end="", flush=True)
        rc, output = _run_one_script(label, fpath)
        status = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {status}")
        results_flat.append((label, fpath, rc, output))

    def _extra_meta(label, rc, output):
        script_path = dict(_RUN_ALL_SCRIPTS).get(label, "")
        rel = os.path.relpath(script_path, _POLARIS_ROOT) if script_path else label
        return [f"**Script:** `{rel}`  \n"]

    write_simple_markdown(
        md_path,
        "ScaledYOLOv4 Validation Report",
        [(lbl, rc, out) for lbl, _fp, rc, out in results_flat],
        _RUNNER_DESCRIPTIONS,
        default_description="Runs ScaledYOLOv4 module validations and reports pass/fail status.",
        label_key_fn=lambda x: x,
        extra_meta_fn=_extra_meta,
    )

    passed = sum(1 for _, _, rc, _ in results_flat if rc == 0)
    total = len(results_flat)

    print(f"\n{'=' * 72}")
    print(f"[ScaledYOLOv4Validation] {passed}/{total} suites passed.")
    print(f"[ScaledYOLOv4Validation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class ScaledYOLOv4Validation(Model):

    def create_input_tensors(self) -> None:
        # Run validation suites (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the ScaledYOLOv4 forward graph.
        super().create_input_tensors()
