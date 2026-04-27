#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
MapTrackerValidation – Integrated Validation Suite Runner (Maper location)

This is a copy of the top-level MapTracker validation runner but placed inside
the `plugin/models/maper` package so it can be referenced from workload YAML
entries that use `basedir: workloads/MapTracker/plugin/models/maper`.

The script locates the Polaris workspace root dynamically (searching for
`polaris.py` or `pyproject.toml`), discovers every `comparison_*` folder in
`workloads/MapTracker/reference/`, runs each folder's `run_all.py`, and writes
a consolidated Markdown report to `workloads/MapTracker/validation_output/`.
"""

import datetime
import os
import re
import subprocess
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Absolute directory that contains this file
_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))


def _find_workspace_root(start_dir: str) -> str:
    """Search upwards from `start_dir` for a workspace root indicator.

    Looks for `polaris.py` or `pyproject.toml`. If not found, falls back to a
    reasonable default (five levels above this file).
    """
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.exists(os.path.join(cur, "polaris.py")) or os.path.exists(
            os.path.join(cur, "pyproject.toml")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            # fallback: assume repository root is 5 levels up from this file
            return os.path.abspath(os.path.join(start_dir, "../../../../../"))
        cur = parent


# Polaris workspace root (detected)
_POLARIS_ROOT: str = os.path.normpath(_find_workspace_root(_THIS_DIR))

# Directory that holds all comparison_* sub-folders (explicitly MapTracker)
_REFERENCE_DIR: str = os.path.normpath(
    os.path.join(_POLARIS_ROOT, "workloads", "MapTracker", "reference")
)

# Output directory for the consolidated Markdown report
_OUTPUT_BASE: str = os.path.normpath(
    os.path.join(_POLARIS_ROOT, "workloads", "MapTracker", "validation_output")
)

# Patterns in stdout that indicate a test failure (mirrors run_all.py)
# Note: case-sensitive so "0 failed out of N" in run_all.py summary does NOT match.
_FAIL_PATTERNS = re.compile(r"\[FAIL\]|\[X\]|\bFAILED\b")

# ---------------------------------------------------------------------------
# Per-suite descriptions (shown in the consolidated Markdown report)
# ---------------------------------------------------------------------------

_SUITE_DESCRIPTIONS: dict[str, str] = {
    "comparison_backbones": (
        "Validates MapTracker backbone components: ResNet / MyResBlock building "
        "blocks, BEVFormer backbone & encoder integration, grid-mask augmentation, "
        "multi-scale deformable attention, spatial cross-attention, temporal "
        "self-attention, placeholder encoder, temporal-net, and the full BEVFormer "
        "transformer pipeline used inside MapTracker."
    ),
    "comparison_heads": (
        "Validates MapTracker detection and segmentation heads: MapDetectorHead "
        "(instance query, cross-attention, regression/classification sub-heads) and "
        "Map_Seg_Head (BEV segmentation branch with FPN-fused features)."
    ),
    "comparison_losses": (
        "Validates MapTracker loss functions: DETR-style set-prediction loss "
        "(Hungarian matching, focal classification loss, L1 + GIoU regression loss) "
        "and the segmentation loss (binary cross-entropy + dice over BEV masks)."
    ),
    "comparison_mapers": (
        "Validates the top-level MapTracker model stack: base mapper "
        "construction, positional-encoding 1-D module, upsample block, "
        "VectorInstanceMemory (temporal track management), and full MapTracker "
        "forward-pass shape propagation with TTSim compute functions."
    ),
    "comparison_necks": (
        "Validates the ConvGRU neck used for temporal BEV feature fusion: "
        "module construction, gated-recurrent update logic, multi-step "
        "propagation, and shape / parameter-count checks."
    ),
    "comparison_transformer_utils": (
        "Validates MapTracker transformer utilities: custom multi-scale "
        "deformable attention (MSDA), MapTransformer (encoder + decoder), "
        "and multi-head attention building blocks used throughout the model."
    ),
    "comparison_utils": (
        "Validates MapTracker utility modules: Embedder (positional / feature "
        "embedding layers) and MotionMLP (instance-level motion prediction "
        "head that refines track positions across frames)."
    ),
}

_DEFAULT_SUITE_DESCRIPTION = (
    "Runs MapTracker-related comparison tests and reports pass/fail status."
)

# ---------------------------------------------------------------------------
# Module-level guard – run validation only once per process
# ---------------------------------------------------------------------------

_VALIDATION_DONE: bool = False
_VALIDATION_MD_PATH: Optional[str] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_suites() -> list[tuple[str, str]]:
    """Return sorted list of ``(suite_name, run_all_path)`` for every comparison suite.

    Only directories named ``comparison_*`` that contain a ``run_all.py`` are included.
    """
    if not os.path.isdir(_REFERENCE_DIR):
        return []
    suites: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(_REFERENCE_DIR)):
        if not entry.startswith("comparison_"):
            continue
        suite_dir = os.path.join(_REFERENCE_DIR, entry)
        if not os.path.isdir(suite_dir):
            continue
        run_all = os.path.join(suite_dir, "run_all.py")
        if os.path.isfile(run_all):
            suites.append((entry, run_all))
    return suites


def _run_suite(run_all_path: str, timeout: int = 600) -> tuple[int, str]:
    """Execute a ``run_all.py`` script as a subprocess.

    Returns ``(returncode, combined_output_string)``.
    A negative return code indicates a runner-level error (timeout, OS error).
    """
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            [sys.executable, run_all_path],
            cwd=_POLARIS_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=sub_env,
        )
        output = proc.stdout or ""
        if proc.stderr and proc.stderr.strip():
            output += "\n--- stderr ---\n" + proc.stderr
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        return -1, f"(TIMEOUT: suite exceeded {timeout} s)"
    except Exception as exc:  # noqa: BLE001
        return -2, f"(RUNNER ERROR: {exc})"


def _count_tests_from_output(output: str) -> tuple[int, int]:
    """Parse the summary line emitted by run_all.py to extract (passed, total).

    Falls back to ``(-1, -1)`` when the line cannot be parsed.
    """
    # run_all.py emits: "Summary: N passed, M failed out of T test(s)"
    m = re.search(
        r"Summary:\s*(\d+)\s+passed,\s*(\d+)\s+failed\s+out\s+of\s*(\d+)",
        output,
        re.IGNORECASE,
    )
    if m:
        passed, failed, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return passed, total
    return -1, -1


def _write_markdown(
    md_path: str,
    results: list[tuple[str, int, str]],
) -> None:
    """Write the consolidated validation-report Markdown file."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tally overall pass/fail at the suite level
    suite_passed = sum(1 for _, rc, _ in results if rc == 0)
    suite_total = len(results)

    # Also tally individual test counts from each suite's output
    total_tests_passed = 0
    total_tests_run = 0
    for _, rc, output in results:
        p, t = _count_tests_from_output(output)
        if t > 0:
            total_tests_passed += p
            total_tests_run += t

    with open(md_path, "w", encoding="utf-8") as fh:
        # ------------------------------------------------------------------ #
        # Header
        # ------------------------------------------------------------------ #
        fh.write("# MapTracker Validation Report\n\n")
        fh.write(f"**Generated:** {now}  \n")
        fh.write(f"**Suites:** {suite_passed}/{suite_total} suites fully passed\n")
        if total_tests_run > 0:
            fh.write(
                f"**Individual tests:** {total_tests_passed}/{total_tests_run} "
                "tests passed across all suites\n"
            )
        fh.write("\n")

        # ------------------------------------------------------------------ #
        # Summary table
        # ------------------------------------------------------------------ #
        fh.write("## Summary\n\n")
        fh.write("| # | Comparison Suite | Individual Tests | Status |\n")
        fh.write("|---|-----------------|:----------------:|:------:|\n")
        for idx, (suite_name, rc, output) in enumerate(results, start=1):
            badge = "✅ PASSED" if rc == 0 else "❌ FAILED"
            p, t = _count_tests_from_output(output)
            test_cell = f"{p}/{t}" if t > 0 else "n/a"
            fh.write(f"| {idx} | `{suite_name}` | {test_cell} | {badge} |\n")
        fh.write("\n")

        # ------------------------------------------------------------------ #
        # Detailed per-suite sections
        # ------------------------------------------------------------------ #
        fh.write("## Detailed Results\n\n")
        for idx, (suite_name, rc, output) in enumerate(results, start=1):
            status_label = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
            description = _SUITE_DESCRIPTIONS.get(suite_name, _DEFAULT_SUITE_DESCRIPTION)

            fh.write(f"### {idx}. `{suite_name}`\n\n")
            fh.write(f"**Status:** `{status_label}`  \n")
            fh.write(f"**Description:** {description}\n\n")
            fh.write("**Output:**\n\n")
            fh.write("```\n")
            cleaned = output.strip() if output and output.strip() else "(no output)"
            fh.write(cleaned + "\n")
            fh.write("```\n\n")
            fh.write("---\n\n")


def _run_validation() -> str:
    """Run all comparison suites and write a consolidated Markdown report.

    Returns the absolute path to the generated report.
    Subsequent calls within the same process are no-ops; the cached path is
    returned instead.
    """
    global _VALIDATION_DONE, _VALIDATION_MD_PATH  # noqa: PLW0603

    if _VALIDATION_DONE and _VALIDATION_MD_PATH is not None:
        return _VALIDATION_MD_PATH

    os.makedirs(_OUTPUT_BASE, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(_OUTPUT_BASE, f"validation_report_{timestamp}.md")

    suites = _collect_suites()
    if not suites:
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# MapTracker Validation Report\n\n")
            fh.write("**Result:** No comparison suites found.\n")
        _VALIDATION_DONE = True
        _VALIDATION_MD_PATH = os.path.abspath(md_path)
        return _VALIDATION_MD_PATH

    print(f"\n{'=' * 72}")
    print(f"[MapTrackerValidation] Running {len(suites)} comparison suite(s)…")
    print(f"{'=' * 72}\n")

    results: list[tuple[str, int, str]] = []
    for suite_name, run_all_path in suites:
        print(f"  › Running {suite_name}/run_all.py …", end="", flush=True)
        rc, output = _run_suite(run_all_path)
        # Also check output for failure markers (mirrors run_all.py logic)
        output_has_failure = bool(_FAIL_PATTERNS.search(output))
        if rc == 0 and output_has_failure:
            rc = 1  # treat as failure
        label = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {label}")
        results.append((suite_name, rc, output))

    _write_markdown(md_path, results)

    suite_passed = sum(1 for _, rc, _ in results if rc == 0)
    suite_total = len(results)

    print(f"\n{'=' * 72}")
    print(
        f"[MapTrackerValidation] {suite_passed}/{suite_total} comparison suite(s) passed."
    )
    print("[MapTrackerValidation] Consolidated validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    _VALIDATION_DONE = True
    _VALIDATION_MD_PATH = os.path.abspath(md_path)
    return _VALIDATION_MD_PATH


# ---------------------------------------------------------------------------
# Polaris workload class
# ---------------------------------------------------------------------------

# Import MapTracker from the same package (same directory)
sys.path.insert(0, _THIS_DIR)
from MapTracker import MapTracker  # type: ignore  # noqa: E402


class MapTrackerValidation(MapTracker):
    """MapTracker workload with integrated validation suite.

    Runs every ``comparison_*/run_all.py`` script the first time the workload
    is instantiated, then behaves identically to the plain MapTracker workload.
    """

    def create_input_tensors(self) -> None:
        # Run validation suites (no-op on subsequent instances).
        _run_validation()
        # Create the TTSim input tensors for the MapTracker forward graph.
        super().create_input_tensors()


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report_path = _run_validation()
    sys.exit(0)
