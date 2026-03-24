#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run every Python test script in comparison_bevformer/ and append
the output to a Markdown report file (comparison_results.md) in the same folder.

A test is marked FAIL if:
  - its exit code is non-zero, OR
  - its stdout contains any of the failure markers: [FAIL], [X], FAILED
"""

import os
import sys
import subprocess
import datetime
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FILE = os.path.join(SCRIPT_DIR, "comparison_results.md")

# Patterns in stdout that indicate a comparison failure
FAIL_PATTERNS = re.compile(r"\[FAIL\]|\[X\]|\bFAILED\b", re.IGNORECASE)

# Collect all test_*.py files in the folder, excluding this runner
this_file = os.path.basename(__file__)
scripts = sorted(
    f
    for f in os.listdir(SCRIPT_DIR)
    if f.endswith(".py") and f.startswith("test_") and f != this_file
)

if not scripts:
    print("No test scripts found in", SCRIPT_DIR)
    sys.exit(0)

python_exe = sys.executable

# Force UTF-8 for subprocess output so Unicode characters don't crash on Windows cp1252.
sub_env = os.environ.copy()
sub_env["PYTHONIOENCODING"] = "utf-8"

print(f"Found {len(scripts)} script(s) to run: {', '.join(scripts)}")
print(f"Report will be written to: {REPORT_FILE}\n")

num_pass = 0
num_fail = 0

with open(REPORT_FILE, "w", encoding="utf-8") as md:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md.write(f"# Comparison Results (bevformer) — {timestamp}\n\n")

    for script in scripts:
        script_path = os.path.join(SCRIPT_DIR, script)
        print(f"Running {script} ...")

        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            env=sub_env,
        )

        # Check for failures: non-zero exit code OR failure markers in output
        output_has_failure = bool(FAIL_PATTERNS.search(result.stdout or ""))
        failed = result.returncode != 0 or output_has_failure

        if failed:
            reasons = []
            if result.returncode != 0:
                reasons.append(f"exit code {result.returncode}")
            if output_has_failure:
                # Count how many failure markers
                fail_matches = FAIL_PATTERNS.findall(result.stdout or "")
                reasons.append(f"{len(fail_matches)} comparison failure(s) in output")
            status = f"FAIL ({'; '.join(reasons)})"
            num_fail += 1
        else:
            status = "PASS"
            num_pass += 1

        print(f"  {script}: {status}")

        md.write(f"## {script}  —  {status}\n\n")

        if result.stdout.strip():
            md.write("### stdout\n\n")
            md.write("```\n")
            md.write(result.stdout)
            if not result.stdout.endswith("\n"):
                md.write("\n")
            md.write("```\n\n")

        if result.stderr.strip():
            md.write("### stderr\n\n")
            md.write("```\n")
            md.write(result.stderr)
            if not result.stderr.endswith("\n"):
                md.write("\n")
            md.write("```\n\n")

        md.write("---\n\n")

    md.write("\n")

print(f"\nDone. Results written to {REPORT_FILE}")
print(f"Summary: {num_pass} passed, {num_fail} failed out of {len(scripts)} test(s)")

if num_fail > 0:
    sys.exit(1)
