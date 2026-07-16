#!/bin/bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# setup-step-0-clean.sh
set -euo pipefail
#
# Description:
#   Full clean of a tt-metal + python_env + tt-npe workspace, so
#   setup-step-1-fresh-build.sh runs from scratch. Opposite of step-1.
#
# Usage:
#   cd /path/to/tt-metal
#   bash setup-step-0-clean.sh        # then: bash setup-step-1-fresh-build.sh
#
# Prerequisites:
#   - Run from the tt-metal repository root (build_metal.sh must exist).
#   - tt-npe is a sibling at ../tt-npe.

if [[ ! -f build_metal.sh ]]; then
    echo "ERROR: run from the tt-metal repository root (build_metal.sh not found)." >&2
    exit 1
fi

# venv must not be active while we delete it.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "ERROR: a virtualenv is active ($VIRTUAL_ENV). Run 'deactivate' first." >&2
    exit 1
fi

# 1. tt-metal C++/kernel artifacts + CPM/kernel caches (does NOT touch python_env).
./build_metal.sh --clean

# 2. Python venv (not covered by --clean).
rm -rf python_env

# 3. tt-npe build + install trees.
rm -rf ../tt-npe/build ../tt-npe/install

if [[ -n "${TT_METAL_CACHE:-}" ]]; then
    echo "NOTE: TT_METAL_CACHE is set ($TT_METAL_CACHE) — delete it manually to fully clean."
fi
echo "Clean complete. Next: bash setup-step-1-fresh-build.sh"
