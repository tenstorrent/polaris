#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FusionADValidation – Polaris Workload with Integrated Validation Suite

Extends FusionAD to additionally execute every run_all.py script found
recursively under workloads/FusionAD/reference/ the first time the workload
is instantiated.

Each run_all.py drives all test_*.py files in its own sub-directory and its
stdout / stderr is captured.  All results are written to a single timestamped
Markdown report located in workloads/FusionAD/validation_output/.  The
absolute path to that report is printed to the terminal once the run is
complete.

After running the validation suite the workload behaves identically to the
plain FusionAD workload – it produces the same forward-graph and the same
Polaris JSON projection output.

Suites discovered under reference/ (run in alphabetical order):
  mmdet_plugin/core/bbox            – bbox utility helpers
  mmdet_plugin/fusionad/dense_heads – task-head modules
  mmdet_plugin/fusionad/detectors   – end-to-end and tracking detectors
  mmdet_plugin/fusionad/modules     – transformer sub-modules
  mmdet_plugin/models/backbones     – VoVNet and SparseEncoderHD backbones
  mmdet_plugin/models/utils         – GridMask, bricks, functional helpers

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwl FusionAD_Validation \\
        -o ODDIR_fusionad_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import sys

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
)

from workloads.FusionAD.projects.mmdet_plugin.fusionad.detectors.fusionad_e2e import (  # noqa: E402
    FusionAD,
)
from workloads.validation_helpers import (  # noqa: E402
    run_suite,
    write_suite_markdown,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Polaris workspace root (six levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)

# Root of the reference test tree
_REFERENCE_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../reference")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../validation_output")
)

# ---------------------------------------------------------------------------
# Per-suite descriptions (shown in the Markdown report)
# ---------------------------------------------------------------------------

_SUITE_DESCRIPTIONS: dict[str, str] = {
    "mmdet_plugin/core/bbox": (
        "Validates bounding-box utility functions used throughout FusionAD: "
        "box encoding/decoding, coordinate normalization, and general bbox "
        "helper operations (`test_bbox_util.py`, `test_util.py`)."
    ),
    "mmdet_plugin/fusionad/dense_heads": (
        "Validates all FusionAD task-specific detection heads: the tracking "
        "head, panoptic segmentation head (PansegformerHead), motion "
        "forecasting head (MotionHead), occupancy prediction head (OccHead), "
        "and ego-motion planning head (PlanningHeadSingleMode)."
    ),
    "mmdet_plugin/fusionad/detectors": (
        "Validates the FusionAD detector pipeline: end-to-end multi-task "
        "model construction and forward pass (`test_fusionad_e2e.py`) and "
        "the tracking-specialised detector (`test_fusionad_track.py`), "
        "including shape verification and numerical accuracy checks."
    ),
    "mmdet_plugin/fusionad/modules": (
        "Validates FusionAD transformer sub-modules: custom base transformer "
        "layer, BEVFormer-style encoder, decoder, multi-scale deformable "
        "attention function, point-cloud cross-attention, spatial "
        "cross-attention, temporal self-attention, and the full transformer "
        "integration test."
    ),
    "mmdet_plugin/models/backbones": (
        "Validates the camera and LiDAR backbone architectures: VoVNet "
        "(`test_vovnet.py`) for image feature extraction and SparseEncoderHD "
        "(`test_sparse_encoder_hd.py`) for sparse-convolution-based LiDAR "
        "BEV feature extraction."
    ),
    "mmdet_plugin/models/utils": (
        "Validates shared utility modules: GridMask data augmentation "
        "(`test_grid_mask.py`), the `run_time` timing/profiling decorator "
        "(`test_bricks.py`), and general functional helpers "
        "(`test_functional.py`)."
    ),
}

_DEFAULT_SUITE_DESCRIPTION = (
    "Runs FusionAD-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _collect_suite_scripts() -> list[tuple[str, str]]:
    """Walk *_REFERENCE_DIR* and return all run_all.py scripts found.

    Returns
    -------
    list of (suite_name, abs_path_to_run_all.py)
        *suite_name* is the relative directory path from *_REFERENCE_DIR*
        (e.g. ``"mmdet_plugin/models/backbones"``).
        List is sorted alphabetically by *suite_name*.
    """
    if not os.path.isdir(_REFERENCE_DIR):
        return []
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(_REFERENCE_DIR):
        dirnames.sort()
        if "run_all.py" in filenames:
            abs_path = os.path.join(dirpath, "run_all.py")
            rel_dir = os.path.relpath(dirpath, _REFERENCE_DIR)
            entries.append((rel_dir, abs_path))
    return sorted(entries)


def _run_validation_tests() -> str:
    """Run all run_all.py suites and write a consolidated Markdown report.

    Returns the **absolute path** to the generated Markdown file.
    Subsequent calls are no-ops; the cached path is returned instead.
    """
    global _VALIDATION_DONE, _VALIDATION_MD_PATH  # noqa: PLW0603

    if _VALIDATION_DONE and _VALIDATION_MD_PATH is not None:
        return _VALIDATION_MD_PATH

    os.makedirs(_OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_OUTPUT_BASE, f"validation_report_{timestamp}.md")

    suites = _collect_suite_scripts()
    if not suites:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# FusionAD Validation Report\n\n")
            fh.write("**Result:** No run_all.py scripts found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(f"[FusionADValidation] Running {len(suites)} validation suite(s)…")
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for suite_name, run_all_path in suites:
        print(f"  › Running {suite_name}/run_all.py …", end="", flush=True)
        rc, output = run_suite(run_all_path, _POLARIS_ROOT)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((suite_name, rc, output))

    write_suite_markdown(
        md_path,
        "FusionAD Validation Report",
        results,
        _SUITE_DESCRIPTIONS,
        default_description=_DEFAULT_SUITE_DESCRIPTION,
    )

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[FusionADValidation] {passed}/{total} suites passed.")
    print(f"[FusionADValidation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class FusionADValidation(FusionAD):

    def create_input_tensors(self) -> None:
        # Run all reference validation suites (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the FusionAD forward graph.
        super().create_input_tensors()
