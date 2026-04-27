#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
UniAD_E2E_Validation – Polaris Workload with Integrated Validation Suite

Extends UniAD_E2E to additionally execute every test_*.py script found in
workloads/UniAD/reference/tests/ the first time the workload is instantiated.

All captured stdout / stderr is written to a timestamped Markdown report
located in workloads/UniAD/validation_output/. The absolute path to that
report is printed to the terminal once the run is complete.

After running the validation suite the workload behaves identically to the
plain UniAD_E2E workload – it produces the same forward-graph and the same
Polaris JSON projection output.

Usage
-----
    python polaris.py \\
        -w config/ip_workloads.yaml \\
        -a config/all_archs.yaml \\
        -m config/wl2archmapping.yaml \\
        --filterwlg ttsim \\
        --filterwl uniad_validation \\
        -o ODDIR_uniad_validation \\
        -s SIMPLE_RUN \\
        --outputformat json
"""

import datetime
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from workloads.UniAD.UniAD_E2E import UniAD_E2E  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute path to the tests folder that holds all test scripts
_VALIDATION_DIR: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "reference", "tests")
)

# Absolute path of the polaris workspace root (two levels above this file)
_POLARIS_ROOT: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# Directory into which the Markdown report is written
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "validation_output")
)

# Files inside _VALIDATION_DIR that are *not* test entry-points
_EXCLUDED_FILES = {"__init__.py"}

# ---------------------------------------------------------------------------
# Per-file descriptions (shown in the Markdown report)
# ---------------------------------------------------------------------------

_TEST_DESCRIPTIONS: dict[str, str] = {
    "test_UniAD_E2E.py": (
        "End-to-end integration tests for the UniAD_E2E ttsim workload: "
        "construction with minimal config, input tensor creation, forward "
        "graph retrieval, analytical parameter count, and call delegation "
        "across both image and BEV-feature inputs."
    ),
    "test_backbone.py": (
        "ResNet backbone TTSim tests: Bottleneck block shape validation "
        "(with and without downsample), multi-layer ResNet construction "
        "and forward-pass output shapes, and FPN feature-map extraction "
        "across all four resolution stages."
    ),
    "test_bev_modules.py": (
        "BEV encoder module tests: BEVFormerEncoder construction with "
        "small config, forward-pass output shape verification for the "
        "flattened BEV feature volume, and attention-mask handling across "
        "different camera and spatial-resolution configurations."
    ),
    "test_motion_head.py": (
        "MotionHead tests: module construction, forward pass with "
        "simulated track outputs, predicted-trajectory output shapes "
        "across varying predict_steps and num_anchor settings, and "
        "multi-mode motion prediction correctness."
    ),
    "test_neck.py": (
        "FPN neck TTSim tests: construction without error, forward-pass "
        "output shapes across four input feature levels (256/512/1024/2048 "
        "channels), and verification that output channel counts match the "
        "configured out_channels."
    ),
    "test_occ_head.py": (
        "OccHead (occupancy prediction head) tests: module construction, "
        "forward-pass output shape for the BEV occupancy grid "
        "(B × n_future × num_occ_classes × bev_h × bev_w), and "
        "configuration-variant checks."
    ),
    "test_planning_head.py": (
        "PlanningHead tests: module construction, forward pass with "
        "simulated motion head outputs, and trajectory prediction output "
        "shapes across varying planning_steps and decoder-layer counts."
    ),
    "test_plugin_modules.py": (
        "Plugin and utility module tests covering: MemoryBank, "
        "QueryInteractionModule, RuntimeTrackerBase, Instances "
        "(track_head_plugin); BevFeatureSlicer, MLP, SimpleConv2d, "
        "CVT_Decoder, UpsamplingAdd, Bottleneck (occ_head_plugin); "
        "IntentionInteraction, TrackAgentInteraction, MapInteraction, "
        "MotionDeformableAttention, CustomModeMultiheadAttention "
        "(motion_head_plugin); MultiScaleDeformableAttnFunction_fp32, "
        "CustomMSDeformableAttention, inverse_sigmoid, "
        "get_reference_points (modules); and "
        "calculate_birds_eye_view_parameters (utility)."
    ),
    "test_seg_head.py": (
        "PansegformerHead (segmentation head) tests: module construction, "
        "forward-pass output shapes for panoptic things and stuff "
        "segmentation masks, and multi-query decoder configuration "
        "variants."
    ),
    "test_track_head.py": (
        "BEVFormerTrackHead detection head tests: module construction, "
        "forward-pass output shapes for detection queries and bounding-box "
        "predictions, and decoder-layer configuration variants."
    ),
}

_DEFAULT_TEST_DESCRIPTION = (
    "Runs UniAD-related validation tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once across all workload instances
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: str | None = None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _collect_test_files() -> list[tuple[str, str]]:
    """Return sorted list of (filename, absolute_path) for every test_*.py."""
    if not os.path.isdir(_VALIDATION_DIR):
        return []
    entries: list[tuple[str, str]] = []
    for fname in sorted(os.listdir(_VALIDATION_DIR)):
        if (
            fname.startswith("test_")
            and fname.endswith(".py")
            and fname not in _EXCLUDED_FILES
        ):
            entries.append((fname, os.path.join(_VALIDATION_DIR, fname)))
    return entries


def _run_one_test(fpath: str, timeout: int = 300) -> tuple[int, str]:
    """Execute a single test script via pytest as a subprocess.

    UniAD tests use pytest-style ``@pytest.mark`` decorators and are
    discovered and run through ``python -m pytest``.

    Returns ``(returncode, combined_output_string)``.
    Negative values indicate a runner error.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", fpath, "-v"],
            cwd=_POLARIS_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        if proc.stderr and proc.stderr.strip():
            output += "\n--- stderr ---\n" + proc.stderr
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        return -1, f"(TIMEOUT: test exceeded {timeout} s)"
    except Exception as exc:  # noqa: BLE001
        return -2, f"(RUNNER ERROR: {exc})"


def _is_markdown_output(text: str) -> bool:
    """Return True if text contains Markdown formatting (headers, bold markers)."""
    for line in text.split("\n")[:40]:
        stripped = line.strip()
        if stripped.startswith("#") or (
            stripped.startswith("**") and len(stripped) > 4
        ):
            return True
    return False


def _write_markdown(
    md_path: str,
    results: list[tuple[str, int, str]],
) -> None:
    """Write the validation-report Markdown file."""
    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w", encoding="utf-8") as fh:
        # ------------------------------------------------------------------ #
        # Header
        # ------------------------------------------------------------------ #
        fh.write("# UniAD Validation Test Report\n\n")
        fh.write(f"**Generated:** {now}  \n")
        fh.write(f"**Result:** {passed}/{total} tests passed\n\n")

        # ------------------------------------------------------------------ #
        # Summary table
        # ------------------------------------------------------------------ #
        fh.write("## Summary\n\n")
        fh.write("| # | Test File | Status |\n")
        fh.write("|---|-----------|:------:|\n")
        for idx, (fname, rc, _) in enumerate(results, start=1):
            badge = "✅ PASSED" if rc == 0 else "❌ FAILED"
            fh.write(f"| {idx} | `{fname}` | {badge} |\n")
        fh.write("\n")

        # ------------------------------------------------------------------ #
        # Detailed per-test sections
        # ------------------------------------------------------------------ #
        fh.write("## Detailed Results\n\n")
        for idx, (fname, rc, output) in enumerate(results, start=1):
            status_label = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
            description = _TEST_DESCRIPTIONS.get(fname, _DEFAULT_TEST_DESCRIPTION)
            cleaned = output.strip() if output and output.strip() else "(no output)"

            fh.write(f"### {idx}. `{fname}`\n\n")
            fh.write(f"**Status:** `{status_label}`  \n")
            fh.write(f"**Description:** {description}\n\n")
            fh.write("**Output:**\n\n")

            if _is_markdown_output(cleaned):
                # Test output is already Markdown – embed directly in a
                # collapsible block so nested headers/fences render properly.
                fh.write("<details>\n<summary>View full output</summary>\n\n")
                fh.write(cleaned + "\n")
                fh.write("\n</details>\n\n")
            else:
                # Plain text (tracebacks, pytest output) – use a code fence.
                fh.write("```\n")
                fh.write(cleaned + "\n")
                fh.write("```\n\n")

            fh.write("---\n\n")


def _run_validation_tests() -> str:
    """Run all validation test scripts and write a Markdown report.

    Returns the **absolute path** to the generated Markdown file.
    Subsequent calls are no-ops; the cached path is returned instead.
    """
    global _VALIDATION_DONE, _VALIDATION_MD_PATH  # noqa: PLW0603

    if _VALIDATION_DONE and _VALIDATION_MD_PATH is not None:
        return _VALIDATION_MD_PATH

    os.makedirs(_OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_OUTPUT_BASE, f"validation_report_{timestamp}.md")

    test_files = _collect_test_files()
    if not test_files:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# UniAD Validation Test Report\n\n")
            fh.write("**Result:** No test files found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(f"[UniAD_E2E_Validation] Running {len(test_files)} validation test(s)…")
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for fname, fpath in test_files:
        print(f"  › Running {fname} …", end="", flush=True)
        rc, output = _run_one_test(fpath)
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((fname, rc, output))

    _write_markdown(md_path, results)

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[UniAD_E2E_Validation] {passed}/{total} tests passed.")
    print(f"[UniAD_E2E_Validation] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------


class UniAD_E2E_Validation(UniAD_E2E):

    def create_input_tensors(self) -> None:
        # Run validation tests (no-op on subsequent instances).
        _run_validation_tests()
        # Create the TTSim input tensors for the UniAD forward graph.
        super().create_input_tensors()
