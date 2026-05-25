#!/bin/bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# setup-step-2-new-login.sh
#
# Description:
#   Environment setup script to run after tt-metal is built.
#   Configures Python environment and TTNN settings for development and profiling.
#
# Usage:
#   cd /path/to/tt-metal
#   source setup-step-2-new-login.sh
#
#   Note: Must use 'source' or '.' to execute, not 'bash', so that
#   environment variables persist in your current shell.
#
# When to use:
#   - Every time you open a new terminal/shell session
#   - After running setup-step-1-fresh-build.sh for the first time
#   - Before running any tt-metal tests or profiling
#
# What this does:
#   1. Activates the Python virtual environment
#   2. Sets TT_METAL_HOME to current directory
#   3. Adds current directory to PYTHONPATH
#   4. Sources NPE profiling tools (required for run-ttnn-profiler.py)
#
# Notes:
#   - Profiling-specific configuration (TTNN_CONFIG_OVERRIDES, profiler flags)
#     is now handled automatically by run-ttnn-profiler.py
#   - For manual TTNN configuration or debug logging, see optional steps below
#
# Prerequisites:
#   - setup-step-1-fresh-build.sh must have completed successfully
#   - python_env/ directory must exist
#   - Run from the tt-metal repository root directory

# Sanity check: ensure this is sourced from the tt-metal repository root
if [[ ! -f python_env/bin/activate || ! -f build_metal.sh ]]; then
    echo "ERROR: setup-step-2-new-login.sh must be sourced from the tt-metal repository root." >&2
    echo "       Expected files not found: python_env/bin/activate, build_metal.sh" >&2
    return 1
fi

# Activate Python virtual environment
source python_env/bin/activate

# Set tt-metal home directory
export TT_METAL_HOME=$(pwd)

# Prepend tt-metal to Python path (preserving any existing entries)
if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$(pwd):${PYTHONPATH}"
else
    export PYTHONPATH="$(pwd)"
fi

# Enable NPE profiling tools (required for run-ttnn-profiler.py)
source ../tt-npe/ENV_SETUP

echo "Environment configured successfully!"
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "Python environment: $(which python)"
echo ""
echo "Note: run-ttnn-profiler.py will automatically configure profiling-specific"
echo "      settings (TTNN_CONFIG_OVERRIDES, TT_METAL_DEVICE_PROFILER, etc.)"

# ============================================================================
# Optional Manual Steps (uncomment as needed)
# ============================================================================

# Manually configure TTNN settings (if not using run-ttnn-profiler.py)
# export TTNN_CONFIG_OVERRIDES='{
#   "enable_fast_runtime_mode": false,
#   "enable_logging": true,
#   "report_name": "report",
#   "enable_graph_report": false,
#   "enable_detailed_buffer_report": true,
#   "enable_detailed_tensor_report": false,
#   "enable_comparison_mode": false
# }'

# Manually enable device profiler (if not using run-ttnn-profiler.py)
# export TT_METAL_DEVICE_PROFILER=1
# export TT_METAL_PROFILER_SYNC=1

# Enable debug-level operation logging
# export TT_LOGGER_TYPES=Op
# export TT_LOGGER_LEVEL=Debug
