# TT-Metal Helper Scripts

A collection of utility scripts for streamlining tt-metal development, setup, profiling, and performance analysis workflows.

## Overview

This repository contains helper scripts that codify common tt-metal development patterns and reduce manual setup steps when working with tt-metal, TTNN, and profiling tools.

## Installation & Setup

These scripts are part of the **Polaris** repository but are designed to work alongside **tt-metal** and **tt-npe** repositories. For use on IRD systems, the scripts need to be copied to the target machine.

### Recommended Directory Structure

```
workspace/
├── si_profiling_helpers/      # These helper scripts (from polaris/tools/si_profiling_helpers/)
│   ├── setup-step-1-fresh-build.sh
│   ├── setup-step-2-new-login.sh
│   ├── run-ttnn-profiler.py
│   └── README.md
├── tt-metal/                  # TT-Metal repository clone
│   ├── build_metal.sh
│   ├── create_venv.sh
│   ├── python_env/
│   └── ...
└── tt-npe/                    # TT-NPE repository clone
    ├── build-npe.sh
    ├── ENV_SETUP
    └── ...
```

### Copying Scripts to IRD Systems

```bash
# From your local machine with polaris repository
cd /path/to/polaris/tools/si_profiling_helpers

# Copy to IRD system
scp -r . username@ird-system:/path/to/workspace/si_profiling_helpers/

# Or copy individual files as needed
scp setup-step-1-fresh-build.sh setup-step-2-new-login.sh run-ttnn-profiler.py \
    username@ird-system:/path/to/workspace/si_profiling_helpers/
```

### Initial Setup on IRD System

```bash
# 1. SSH to IRD system
ssh username@ird-system

# 2. Create workspace directory and navigate to it
mkdir -p ~/workspace
cd ~/workspace

# 3. Clone required repositories (if not already present)
git clone <tt-metal-repo-url> tt-metal
git clone <tt-npe-repo-url> tt-npe

# 4. Run initial build (from tt-metal directory)
cd tt-metal
bash ../si_profiling_helpers/setup-step-1-fresh-build.sh

# 5. Configure environment (run in each new shell session)
source ../si_profiling_helpers/setup-step-2-new-login.sh
```

## Scripts Index

### Setup Scripts

#### [`setup-step-1-fresh-build.sh`](setup-step-1-fresh-build.sh)
Initial setup script for fresh tt-metal installation.
- Builds tt-metal (profiler support is always enabled)
- Creates Python virtual environment
- Builds tt-npe (Lightweight Network-on-Chip Performance Estimator) toolkit with clang-20
- **When to use:** After fresh-cloning tt-metal or pulling major updates
- **Usage:** `cd /path/to/tt-metal && bash setup-step-1-fresh-build.sh`
- **Prerequisites:** clang-20 and clang++-20 must be installed and in PATH

#### [`setup-step-2-new-login.sh`](setup-step-2-new-login.sh)
Environment configuration script for new shell sessions.
- Activates Python virtual environment
- Sets TT_METAL_HOME and PYTHONPATH
- Sources NPE profiling tools (required for run-ttnn-profiler.py)
- **When to use:** Every time you open a new terminal session
- **Usage:** `cd /path/to/tt-metal && source setup-step-2-new-login.sh`
- **Note:** Must use `source` or `.`, not `bash`
- **Note:** TTNN profiling configuration is automatically handled by run-ttnn-profiler.py

---

### Profiling and Analysis Tools

#### [`run-ttnn-profiler.py`](run-ttnn-profiler.py)
Comprehensive TTNN profiler runner that wraps tracy with multiple profiling modes.

**Features:**
- Supports three profiling modes: raw profiling, NOC traces, and performance counters
- Pytest integration support
- Optional cleanup of large generated directories (npe_viz, .logs)
- Dry-run capability for command preview
- Automatic TTNN environment configuration with profiling overrides
- Optional disable-logging flag for workloads with device sync issues

**Usage:**
```bash
# Basic profiling only
python run-ttnn-profiler.py \
    --command "test_ttnn_functional_vit.py" \
    --report-name vit_test \
    --output-dir ./profiler_output \
    --basic-only

# Full profiling with pytest and cleanup
python run-ttnn-profiler.py \
    --command "tests/ttnn/unit_tests/test_model.py::test_forward" \
    --pytest \
    --report-name model_test \
    --output-dir ./profiler_output \
    --cleanup

# With logging disabled (for workloads with device sync issues)
python run-ttnn-profiler.py \
    --command "test.py" \
    --report-name test \
    --output-dir ./output \
    --disable-logging
```

**Prerequisites:**
- NPE tools on PATH (automatically configured by setup-step-2-new-login.sh)
- loguru package installed
- **Board must be reset before profiling:** Use `tensix_reset.py` to automatically detect and reset the board
  - See "Board Reset" section below for details

#### [`tensix_reset.py`](tensix_reset.py)
Automatically detects and resets Tensix board(s) via `tt-smi`.

**Features:**
- Parses `tt-smi -ls` output to find resettable boards without manual index lookup
- Resets a single board by default; requires `--multiple-boards` when multiple boards are present

**Usage:**
```bash
# Reset the single available board
python tensix_reset.py

# Reset all boards on a multi-board system
python tensix_reset.py --multiple-boards
```

---

## Quick Start

### First-Time Setup
Assuming you've followed the recommended directory structure above:

```bash
# Navigate to tt-metal directory
cd ~/workspace/tt-metal

# 1. Build tt-metal (also builds tt-npe)
bash ../si_profiling_helpers/setup-step-1-fresh-build.sh

# 2. Configure environment (in new shell sessions)
source ../si_profiling_helpers/setup-step-2-new-login.sh
```

### Board Reset

**IMPORTANT:** Always reset the board before running profiling to ensure clean hardware state.

```bash
# Automatically detect and reset the board
python ../si_profiling_helpers/tensix_reset.py

# If multiple boards are present and all should be reset
python ../si_profiling_helpers/tensix_reset.py --multiple-boards
```

**Manual fallback** (if `tensix_reset.py` is not working):
```bash
# List devices to identify the board index
tt-smi -ls

# Reset by index (e.g., index 0)
tt-smi -r 0
```

**Why reset is required:**
- Hardware state from previous runs can interfere with profiling accuracy
- Ensures consistent starting conditions for each profiling session
- Clears any stale device state or hung operations

### Running Profiler
```bash
# From tt-metal directory:

# 1. Reset the board
python ../si_profiling_helpers/tensix_reset.py

# 2. Run profiler with all modes
python ../si_profiling_helpers/run-ttnn-profiler.py \
    --command "test_script.py" \
    --report-name my_test \
    --output-dir ./profiler_output

# Example: Run pytest test with profiler
python ../si_profiling_helpers/run-ttnn-profiler.py \
    --command "tests/ttnn/unit_tests/test_model.py::test_forward" \
    --pytest \
    --report-name pytest_test \
    --output-dir ./profiler_output_pytest
```

---

## Documentation

All scripts include comprehensive inline documentation:
- Detailed usage instructions and examples
- Required and optional parameters
- Prerequisites and environment requirements
- Input/output format specifications
- Error handling and exit codes

See individual script headers for complete documentation.

---

## Requirements

### Common Requirements
- Python 3.8 or later
- loguru package: `pip install loguru`

### Build Scripts
- clang-20 and clang++-20 installed and in PATH

### Profiling Scripts
- tt-metal repository
- tt-npe tools (for profiling operations)
- tracy profiler module

---

## Repository Structure

```
si_profiling_helpers/
├── README.md                      # This file
├── setup-step-1-fresh-build.sh    # Initial build setup
├── setup-step-2-new-login.sh      # Session environment setup
├── run-ttnn-profiler.py           # TTNN profiler runner
└── tensix_reset.py                # Automated board reset via tt-smi
```

---

## License

These scripts are utility tools for tt-metal development workflows.

---

## Contributing

When adding new scripts:
1. Include comprehensive header documentation
2. Add usage examples (multiple scenarios)
3. Document all parameters and prerequisites
4. Add function-level docstrings
5. Update this README.md index

---

**Last Updated:** April 8, 2026
