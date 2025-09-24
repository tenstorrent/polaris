# Tools

This directory contains various utility scripts and tools for the Polaris project.

## Scripts

## CI/CD Scripts

These scripts are located in the `ci/` subdirectory and are primarily used for continuous integration and development workflows.

### ci/check_behind_tailscale.sh
A cross-platform bash script that detects if the system is running behind Tailscale VPN.

**Usage:**
```bash
./tools/check_behind_tailscale.sh
```

**Exit Codes:**
- `0`: System is behind Tailscale (connected)
- `1`: System is NOT behind Tailscale (not connected)

**Platforms:** macOS, Linux

See [doc/tools/ci/README_check_behind_tailscale.md](../doc/tools/ci/README_check_behind_tailscale.md) for detailed documentation.

### ci/lfc_downloader.sh
Downloads models and files from the Large File Cache (LFCache) server.

**Requirements:**
- **wget** (automatically checked on macOS with helpful installation guidance)
- **Tailscale VPN connection** (for non-CI environments)
- CI environments automatically bypass Tailscale requirement

**Platform Support:**
- **Linux**: wget typically pre-installed
- **macOS**: Automatic wget detection with installation instructions for Homebrew/MacPorts

**Usage:**
```bash
./tools/ci/lfc_downloader.sh [-v|--verbose] [-n|--dryrun] [--type TYPE] [--extract] <server_path> [local_path]
```

See [doc/tools/ci/lfc_downloader_user_guide.md](../doc/tools/ci/lfc_downloader_user_guide.md) for complete documentation.

### ci/colorpicker.py
Python script for color selection based on values, conclusions, or exit codes.

**Usage:**
```bash
# Threshold-based mode
python3 ./tools/ci/colorpicker.py --value NUM --highcolor COLOR threshold1 color1 threshold2 color2 [...]

# Conclusion-based mode
python3 ./tools/ci/colorpicker.py --conclusion {success,failure,cancelled,skipped}

# Exit code-based mode
python3 ./tools/ci/colorpicker.py --exitcode NUM
```

**Requirements:**
- **bigpoldev conda environment** recommended

See [doc/tools/ci/README_colorpicker.md](../doc/tools/ci/README_colorpicker.md) for complete documentation.

### ci/makegist.py
Python script for creating and updating GitHub Gists with dynamic data.

**Usage:**
```bash
python3 ./tools/ci/makegist.py --gist-id ID --gist-filename FILE [key=value pairs...]
```

See [doc/tools/ci/README_makegist.md](../doc/tools/ci/README_makegist.md) for complete documentation.

### ci/rtl_scurve_badge.py
RTL S-curve test result processor that generates summary files, CSV exports, and dynamic badges from s-curve format test output.

**Usage:**
```bash
python3 ./tools/ci/rtl_scurve_badge.py --repo REPO --gistid GIST_ID --input FILE [--dryrun]
```

**Requirements:**
- **GIST_TOKEN environment variable** for GitHub gist creation
- **bigpoldev conda environment** recommended
- **Input files with s-curve section** (between `+ Test class s-curve:` and `+ Saving` markers)

**Key Features:**
- Specialized parser for pipe-delimited s-curve test result lines
- Geometric mean calculation for model/RTL cycle ratios
- Color-coded badges based on test pass rates and performance metrics

See [doc/tools/ci/README_rtl_scurve_badge.md](../doc/tools/ci/README_rtl_scurve_badge.md) for complete documentation.

## Other Tools

### compare_projections.py
Python script for comparing projection results.


### parse_nv_mlperf_results.py
Parser for NVIDIA MLPerf benchmark results.

### run_onnx_shape_inference.py
ONNX model shape inference utility.

### spdxchecker.py
SPDX license header checker.

See [doc/tools/ci/README_spdxchecker.md](../doc/tools/ci/README_spdxchecker.md) for complete documentation.

### statattr.py
Statistical attribute analysis tool.
