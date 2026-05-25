#!/bin/bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# setup-step-1-fresh-build.sh
set -euo pipefail
#
# Description:
#   Initial setup script for a fresh tt-metal installation.
#   Builds tt-metal, creates a Python virtual environment,
#   and builds tt-npe (Neural Processing Engine) toolkit.
#
# Usage:
#   cd /path/to/tt-metal
#   bash setup-step-1-fresh-build.sh
#
# When to use:
#   - After fresh-cloning tt-metal repository
#   - After pulling major updates that require a rebuild
#   - When setting up tt-metal on a new machine
#
# What this does:
#   1. Builds tt-metal
#   2. Creates Python virtual environment in python_env/
#   3. Builds tt-npe with clang-20 compiler
#
# Next steps:
#   After this completes successfully, run setup-step-2-new-login.sh
#
# Prerequisites:
#   - Run from the tt-metal repository root directory
#   - build_metal.sh, create_venv.sh, and build-npe.sh must exist
#   - clang-20 and clang++-20 must be installed and in PATH
#   - Required build dependencies must be installed

# Build tt-metal (profiler support is always enabled)
./build_metal.sh

# Create Python virtual environment
./create_venv.sh

# Activate venv so tt-npe build picks up the right Python interpreter
source python_env/bin/activate

# Build tt-npe with clang-20 compiler
cd ../tt-npe
bash build-npe.sh
