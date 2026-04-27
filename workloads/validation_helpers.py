#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
validation_helpers.py – Shared utilities for Polaris validation workloads.

This module centralises all code that is common across the per-model validation
runners (ScaledYOLOv4, DeformableDETR, DiffusionDrive, MapTracker, BEVFormer).

Public API
----------
run_subprocess(fpath, polaris_root, timeout)
    Execute a single Python script as a subprocess, return (rc, output).

run_suite(run_all_path, polaris_root, timeout)
    Execute a run_all.py suite script; same signature as run_subprocess.

is_markdown_output(text)
    Return True when the text looks like already-formatted Markdown.

collect_test_files(validation_dir)
    Walk a directory tree and return sorted (rel_path, abs_path) pairs for
    every test_*.py file found.

collect_suite_scripts(reference_dir, prefix)
    Return sorted (suite_name, run_all_path) pairs for every <prefix>_*/
    sub-directory that contains a run_all.py.

write_simple_markdown(md_path, title, results, descriptions, default_description)
    Write a flat (non-sectioned) Markdown report where each result is one row.
    ``results`` must be a list of ``(label, returncode, output)`` tuples.

write_sectioned_markdown(md_path, title, results, section_titles, descriptions,
                         default_description)
    Write a sectioned Markdown report grouped by subsystem.
    ``results`` must be a list of ``(section_key, label, returncode, output)``
    tuples.

run_once(output_base, banner_tag, collect_fn, run_fn, write_fn, fail_message)
    Generic "run-once" driver: creates the output directory, collects items,
    runs each one, calls write_fn, prints a summary banner, and returns the
    absolute path to the generated report.
    Callers wrap this with a module-level guard variable.
"""

from __future__ import annotations

import datetime
import os
import re
import subprocess
import sys
from typing import Callable

__all__ = [
    "run_subprocess",
    "run_suite",
    "is_markdown_output",
    "collect_test_files",
    "collect_suite_scripts",
    "write_simple_markdown",
    "write_sectioned_markdown",
    "write_suite_markdown",
    "run_once",
]


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_subprocess(
    fpath: str,
    polaris_root: str,
    timeout: int = 300,
) -> tuple[int, str]:
    """Execute a Python script as a subprocess.

    Parameters
    ----------
    fpath:
        Absolute path to the script to run.
    polaris_root:
        Working directory passed to the subprocess (the Polaris repo root).
    timeout:
        Maximum wall-clock seconds to wait before killing the process.

    Returns
    -------
    (returncode, combined_output)
        *returncode* is 0 on success, positive for script errors, negative for
        runner-level errors (timeout → -1, OS/Python error → -2, not found → -3).
    """
    if not os.path.isfile(fpath):
        return -3, f"(SCRIPT NOT FOUND: {fpath})"
    try:
        proc = subprocess.run(
            [sys.executable, fpath],
            cwd=polaris_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout or ""
        if proc.stderr and proc.stderr.strip():
            output += "\n--- stderr ---\n" + proc.stderr
        return proc.returncode, output
    except subprocess.TimeoutExpired:
        return -1, f"(TIMEOUT: script exceeded {timeout} s)"
    except Exception as exc:  # noqa: BLE001
        return -2, f"(RUNNER ERROR: {exc})"


def run_suite(
    run_all_path: str,
    polaris_root: str,
    timeout: int = 600,
) -> tuple[int, str]:
    """Execute a ``run_all.py`` suite script as a subprocess.

    Identical in behaviour to :func:`run_subprocess` but sets
    ``PYTHONIOENCODING=utf-8`` in the subprocess environment and uses a longer
    default timeout suitable for multi-test suites.
    """
    if not os.path.isfile(run_all_path):
        return -3, f"(SCRIPT NOT FOUND: {run_all_path})"
    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [sys.executable, run_all_path],
            cwd=polaris_root,
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


# ---------------------------------------------------------------------------
# Markdown detection
# ---------------------------------------------------------------------------


def is_markdown_output(text: str) -> bool:
    """Return True if *text* appears to contain Markdown formatting.

    Checks only the first 40 lines for ``#``-headers or ``**bold**`` markers
    so the check is fast even for large outputs.
    """
    for line in text.split("\n")[:40]:
        stripped = line.strip()
        if stripped.startswith("#") or (
            stripped.startswith("**") and len(stripped) > 4
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# File / suite discovery
# ---------------------------------------------------------------------------


def collect_test_files(
    validation_dir: str,
) -> list[tuple[str, str]]:
    """Recursively find all ``test_*.py`` files under *validation_dir*.

    Returns
    -------
    list of (relative_path, absolute_path)
        Sorted deterministically (directories then filenames, both
        alphabetically).  Files named ``run_*.py`` or ``__init__.py`` are
        never included.
    """
    if not os.path.isdir(validation_dir):
        return []
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(validation_dir):
        dirnames.sort()
        for fname in sorted(filenames):
            if fname.startswith("test_") and fname.endswith(".py"):
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, validation_dir)
                entries.append((rel_path, abs_path))
    return entries


def collect_suite_scripts(
    reference_dir: str,
    prefix: str = "comparison_",
) -> list[tuple[str, str]]:
    """Find ``run_all.py`` scripts inside ``<prefix>*`` sub-directories.

    Parameters
    ----------
    reference_dir:
        Parent directory to search.
    prefix:
        Only sub-directories whose name starts with this prefix are included.

    Returns
    -------
    list of (suite_name, run_all_path)
        Sorted alphabetically by suite_name.
    """
    if not os.path.isdir(reference_dir):
        return []
    suites: list[tuple[str, str]] = []
    for entry in sorted(os.listdir(reference_dir)):
        if not entry.startswith(prefix):
            continue
        suite_dir = os.path.join(reference_dir, entry)
        if not os.path.isdir(suite_dir):
            continue
        run_all = os.path.join(suite_dir, "run_all.py")
        if os.path.isfile(run_all):
            suites.append((entry, run_all))
    return suites


# ---------------------------------------------------------------------------
# Markdown report writers
# ---------------------------------------------------------------------------


def _format_output_block(fh, cleaned: str) -> None:
    """Write *cleaned* output into the open file handle *fh*.

    If the output already looks like Markdown it is wrapped in a collapsible
    ``<details>`` block; otherwise it is fenced as plain text.
    """
    if is_markdown_output(cleaned):
        fh.write("<details>\n<summary>View full output</summary>\n\n")
        fh.write(cleaned + "\n")
        fh.write("\n</details>\n\n")
    else:
        fh.write("```\n")
        fh.write(cleaned + "\n")
        fh.write("```\n\n")


def write_simple_markdown(
    md_path: str,
    title: str,
    results: list[tuple[str, int, str]],
    descriptions: dict[str, str],
    default_description: str = "Runs validation tests and reports pass/fail status.",
    *,
    label_key_fn: Callable[[str], str] = os.path.basename,
    extra_meta_fn: Callable[[str, int, str], list[str]] | None = None,
) -> None:
    """Write a flat Markdown validation report.

    Parameters
    ----------
    md_path:
        Output file path.
    title:
        Report title (used in the ``# heading``).
    results:
        List of ``(label, returncode, output)`` tuples.
    descriptions:
        Mapping from label (or its basename) to description string.
    default_description:
        Fallback description when the label is not found in *descriptions*.
    label_key_fn:
        Callable applied to *label* to produce the dict-lookup key.
        Defaults to ``os.path.basename``.
    extra_meta_fn:
        Optional callable ``(label, rc, output) -> list[str]`` that returns
        extra ``fh.write(...)`` lines to insert before the output block.
    """
    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"**Generated:** {now}  \n")
        fh.write(f"**Result:** {passed}/{total} tests passed\n\n")

        # Summary table
        fh.write("## Summary\n\n")
        fh.write("| # | Test / Suite | Status |\n")
        fh.write("|---|-------------|:------:|\n")
        for idx, (label, rc, _) in enumerate(results, start=1):
            badge = "✅ PASSED" if rc == 0 else "❌ FAILED"
            fh.write(f"| {idx} | `{label}` | {badge} |\n")
        fh.write("\n")

        # Detailed sections
        fh.write("## Detailed Results\n\n")
        for idx, (label, rc, output) in enumerate(results, start=1):
            status_label = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
            key = label_key_fn(label)
            description = descriptions.get(key, default_description)
            cleaned = output.strip() if output and output.strip() else "(no output)"

            fh.write(f"### {idx}. `{label}`\n\n")
            fh.write(f"**Status:** `{status_label}`  \n")
            fh.write(f"**Description:** {description}\n\n")
            if extra_meta_fn:
                for line in extra_meta_fn(label, rc, output):
                    fh.write(line)
            fh.write("**Output:**\n\n")
            _format_output_block(fh, cleaned)
            fh.write("---\n\n")


def write_sectioned_markdown(
    md_path: str,
    title: str,
    results: list[tuple[str, str, int, str]],
    section_titles: dict[str, str],
    descriptions: dict[str, str],
    default_description: str = "Runs validation tests and reports pass/fail status.",
) -> None:
    """Write a sectioned Markdown validation report.

    Parameters
    ----------
    md_path:
        Output file path.
    title:
        Report title (used in the ``# heading``).
    results:
        List of ``(section_key, label, returncode, output)`` tuples.
    section_titles:
        Mapping from *section_key* to a human-readable section heading.
    descriptions:
        Mapping from *label* (filename) to description string.
    default_description:
        Fallback description when *label* is not in *descriptions*.
    """
    passed = sum(1 for _, _, rc, _ in results if rc == 0)
    total = len(results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"**Generated:** {now}  \n")
        fh.write(f"**Result:** {passed}/{total} tests passed\n\n")

        # Summary table
        fh.write("## Summary\n\n")
        fh.write("| # | Section | Test File | Status |\n")
        fh.write("|---|---------|-----------|:------:|\n")
        for idx, (section_key, label, rc, _) in enumerate(results, start=1):
            badge = "✅ PASSED" if rc == 0 else "❌ FAILED"
            section = section_titles.get(section_key, section_key)
            fh.write(f"| {idx} | {section} | `{label}` | {badge} |\n")
        fh.write("\n")

        # Detailed sections
        fh.write("## Detailed Results\n\n")
        current_section: str | None = None
        for idx, (section_key, label, rc, output) in enumerate(results, start=1):
            section = section_titles.get(section_key, section_key)
            if section_key != current_section:
                current_section = section_key
                fh.write(f"### {section}\n\n")

            status_label = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
            description = descriptions.get(label, default_description)
            cleaned = output.strip() if output and output.strip() else "(no output)"

            fh.write(f"#### {idx}. `{label}`\n\n")
            fh.write(f"**Status:** `{status_label}`  \n")
            fh.write(f"**Description:** {description}\n\n")
            fh.write("**Output:**\n\n")
            fh.write("```\n")
            fh.write(cleaned + "\n")
            fh.write("```\n\n")
            fh.write("---\n\n")


# ---------------------------------------------------------------------------
# Suite-level Markdown writer (for run_all.py-style suites)
# ---------------------------------------------------------------------------


def _parse_suite_counts(output: str) -> tuple[int, int]:
    """Parse a ``run_all.py`` summary line for (passed, total).

    Looks for: ``Summary: N passed, M failed out of T test(s)``

    Returns ``(-1, -1)`` when the pattern is not found.
    """
    m = re.search(
        r"Summary:\s*(\d+)\s+passed,\s*(\d+)\s+failed\s+out\s+of\s*(\d+)",
        output,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1)), int(m.group(3))
    return -1, -1


def write_suite_markdown(
    md_path: str,
    title: str,
    results: list[tuple[str, int, str]],
    descriptions: dict[str, str],
    default_description: str = "Runs validation suite and reports pass/fail status.",
    *,
    fail_patterns: re.Pattern | None = None,
) -> None:
    """Write a Markdown report for ``run_all.py``-style suite results.

    Parameters
    ----------
    md_path, title, descriptions, default_description:
        Same semantics as :func:`write_simple_markdown`.
    results:
        List of ``(suite_name, returncode, output)`` tuples.
    fail_patterns:
        Optional compiled regex.  When provided, any output matching this
        pattern causes the suite to be marked as failed regardless of its
        return code.
    """
    # Re-evaluate rc against fail_patterns (MapTracker-style re-check)
    if fail_patterns is not None:
        adjusted: list[tuple[str, int, str]] = []
        for suite_name, rc, output in results:
            if rc == 0 and fail_patterns.search(output):
                rc = 1
            adjusted.append((suite_name, rc, output))
        results = adjusted

    # Count suite-level passes
    suite_passed = sum(1 for _, rc, _ in results if rc == 0)
    suite_total = len(results)

    # Also tally individual test counts from run_all.py summary lines
    total_tests_passed = 0
    total_tests_run = 0
    for _, rc, output in results:
        p, t = _parse_suite_counts(output)
        if t > 0:
            total_tests_passed += p
            total_tests_run += t

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {title}\n\n")
        fh.write(f"**Generated:** {now}  \n")
        fh.write(
            f"**Suites:** {suite_passed}/{suite_total} suites fully passed\n"
        )
        if total_tests_run > 0:
            fh.write(
                f"**Individual tests:** {total_tests_passed}/{total_tests_run} "
                "tests passed across all suites\n"
            )
        fh.write("\n")

        # Summary table
        fh.write("## Summary\n\n")
        fh.write("| # | Suite | Individual Tests | Status |\n")
        fh.write("|---|-------|:----------------:|:------:|\n")
        for idx, (suite_name, rc, output) in enumerate(results, start=1):
            badge = "✅ PASSED" if rc == 0 else "❌ FAILED"
            p, t = _parse_suite_counts(output)
            test_cell = f"{p}/{t}" if t > 0 else "n/a"
            fh.write(f"| {idx} | `{suite_name}` | {test_cell} | {badge} |\n")
        fh.write("\n")

        # Detailed sections
        fh.write("## Detailed Results\n\n")
        for idx, (suite_name, rc, output) in enumerate(results, start=1):
            status_label = "PASSED" if rc == 0 else f"FAILED (exit code {rc})"
            description = descriptions.get(suite_name, default_description)
            cleaned = output.strip() if output and output.strip() else "(no output)"

            fh.write(f"### {idx}. `{suite_name}`\n\n")
            fh.write(f"**Status:** `{status_label}`  \n")
            fh.write(f"**Description:** {description}\n\n")
            fh.write("**Output:**\n\n")
            fh.write("```\n")
            fh.write(cleaned + "\n")
            fh.write("```\n\n")
            fh.write("---\n\n")


# ---------------------------------------------------------------------------
# Generic run-once driver
# ---------------------------------------------------------------------------


def run_once(
    output_base: str,
    banner_tag: str,
    collect_fn: Callable[[], list],
    run_fn: Callable[[str], tuple[int, str]],
    write_fn: Callable[[str, list], None],
    empty_report_fn: Callable[[str], None] | None = None,
    report_name_prefix: str = "validation_report",
) -> str:
    """Generic driver that runs a validation suite exactly once per process.

    This function does **not** manage the module-level guard variables –
    callers are responsible for that (see usage examples in each
    ``_run_validation_tests`` function).

    Parameters
    ----------
    output_base:
        Directory into which the Markdown report is written.  Created if it
        does not exist.
    banner_tag:
        Short identifier printed in the ``[...] Running N item(s)…`` banner,
        e.g. ``"ScaledYOLOv4Validation"``.
    collect_fn:
        Zero-argument callable that returns a list of items to process.  Each
        item must be something that *run_fn* can accept.  The list may be
        empty (see *empty_report_fn*).
    run_fn:
        Callable that accepts one item from the list returned by *collect_fn*
        and returns ``(returncode, output_string)``.
    write_fn:
        Callable ``(md_path, results_list) -> None`` that writes the Markdown
        report.  *results_list* contains the accumulated
        ``(item, rc, output)`` tuples.
    empty_report_fn:
        Optional callable ``(md_path) -> None`` called when *collect_fn*
        returns an empty list, to write a "nothing to run" placeholder report.
        If ``None``, a generic placeholder is written.
    report_name_prefix:
        Prefix for the timestamped report filename.

    Returns
    -------
    str
        Absolute path to the generated Markdown report.
    """
    os.makedirs(output_base, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_base, f"{report_name_prefix}_{timestamp}.md")

    items = collect_fn()

    if not items:
        if empty_report_fn is not None:
            empty_report_fn(md_path)
        else:
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write(f"# {banner_tag} Validation Report\n\n")
                fh.write("**Result:** No items found.\n")
        return os.path.abspath(md_path)

    print(f"\n{'=' * 72}")
    print(f"[{banner_tag}] Running {len(items)} item(s)…")
    print(f"{'=' * 72}\n")

    results = []
    for item in items:
        # items may be (label, path) tuples or plain paths
        if isinstance(item, (list, tuple)):
            label = item[0] if len(item) >= 1 else str(item)
            path = item[-1]
        else:
            label = os.path.basename(str(item))
            path = str(item)
        print(f"  › Running {label} …", end="", flush=True)
        rc, output = run_fn(path)
        status = "PASSED" if rc == 0 else f"FAILED (exit {rc})"
        print(f" {status}")
        results.append((item, rc, output))

    write_fn(md_path, results)

    passed = sum(1 for _, rc, _ in results if rc == 0)
    total = len(results)

    print(f"\n{'=' * 72}")
    print(f"[{banner_tag}] {passed}/{total} items passed.")
    print(f"[{banner_tag}] Validation report written to:")
    print(f"  {os.path.abspath(md_path)}")
    print(f"{'=' * 72}\n")

    return os.path.abspath(md_path)
