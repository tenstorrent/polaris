#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run every Python test script in comparison detectors/ and append
the output to a Markdown report file (comparison_results.md) in the same folder.
"""

import os
import sys
import subprocess
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FILE = os.path.join(SCRIPT_DIR, "comparison_results.md")

# Collect all test_*.py files in the folder, excluding this runner
this_file = os.path.basename(__file__)
scripts = sorted(
    f for f in os.listdir(SCRIPT_DIR)
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

with open(REPORT_FILE, "w", encoding="utf-8") as md:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md.write(f"# Comparison Results (detectors) -- {timestamp}\n\n")

    for script in scripts:
        script_path = os.path.join(SCRIPT_DIR, script)
        print(f"Running {script} ...")

        result = subprocess.run(
            [python_exe, script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=SCRIPT_DIR,
            env=sub_env,
        )

        status = "PASS" if result.returncode == 0 else f"FAIL (exit code {result.returncode})"
        print(f"  {script}: {status}")

        md.write(f"## {script}  --  {status}\n\n")

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
