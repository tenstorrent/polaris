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
python3 ./tools/ci/rtl_scurve_badge.py --repo REPO --gistid GIST_ID --input FILE [--is-main-branch] [--dryrun]
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

### ci/cleanup_delme_files.py
Python script for cleaning up temporary DELETEME_ prefixed files from GitHub gists.

**Usage:**
```bash
python3 ./tools/ci/cleanup_delme_files.py
```

**Requirements:**
- **GIST_ID environment variable** for target gist
- **GIST_TOKEN environment variable** for GitHub API access

**Features:**
- Automatic detection of DELETEME_ prefixed files
- Safe deletion of temporary files
- Comprehensive logging and error handling

See [doc/tools/ci/README_gist_cleanup.md](../doc/tools/ci/README_gist_cleanup.md) for complete documentation.

## Correlation and Analysis Tools

### run_ttsi_corr.py
Metal-Tensix correlation analysis tool with modular architecture.

**Purpose**: Automated correlation between Polaris HLM projections and silicon performance measurements.

**Key Features**:
- Modular architecture via `ttsi_corr` package
- Automated Polaris simulation execution
- Multiple output formats (CSV, XLSX with S-curve charts, JSON)
- Geometric mean calculation for aggregate accuracy
- Support for HTML and Markdown data sources

**Usage**:
```bash
python tools/run_ttsi_corr.py \
    --tag 15oct25 \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml
```

**Modular Components** (in `tools/ttsi_corr/` package):
- `data_loader` - Load and validate reference metrics
- `workload_processor` - Process workload configurations
- `correlation` - Calculate correlation metrics
- `excel_writer` - Generate Excel reports with formatting
- `chart_builder` - Create S-curve visualization charts
- `simulator` - Orchestrate Polaris simulation

See [doc/tools/README_correlation.md](../doc/tools/README_correlation.md) for complete documentation.

### parse_ttsi_perf_results.py
Markdown and HTML metrics parser for extracting performance data.

**Purpose**: Extract and categorize performance metrics from TT-Metal documentation.

**Usage**:
```bash
python tools/parse_ttsi_perf_results.py \
    --tag 15oct25 \
    --input https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md \
    --output-dir data/metal/inf
```

See [doc/tools/README_correlation.md](../doc/tools/README_correlation.md) for complete workflow documentation.

### ttsi_corr Package
Modular Python package for correlation analysis (v0.1.0).

**Location**: `tools/ttsi_corr/`

**Public API**:
```python
from ttsi_corr import (
    load_metrics_from_sources,  # Data loading
    process_workload_configs,   # Workload processing
    compare_scores,             # Correlation calculation
    write_csv,                  # CSV export
    add_scurve_chart,          # Chart generation
)
```

**Architecture**: 6 specialized modules with single responsibility principle, full type annotations, and comprehensive documentation.

See [tools/ttsi_corr/README.md](ttsi_corr/README.md) for complete package documentation.

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
