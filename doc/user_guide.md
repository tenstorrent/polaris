# Polaris User Guide

## Overview
Polaris is a high-level simulator for performance analysis of AI architectures. It enables users to analyze and evaluate the performance characteristics of AI workloads on different hardware architectures through simulation.

## Key Features
- **Workload Analysis**: Input AI workloads are converted into DAG (Directed Acyclic Graph) representations
- **Architecture Simulation**: Simulate workloads on different hardware configurations
- **Performance Analysis**: Get detailed performance metrics and resource utilization data
- **Flexible Output Formats**: Support for YAML, JSON, and Pickle output formats

## Installation

### Prerequisites
- Python 3.13 or higher
- Miniforge package manager

### Download Polaris
Execute ```git clone https://github.com/tenstorrent/polaris.git```

### Setup Steps
1. Install Miniforge:
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   conda update -n base -c conda-forge conda
   source <your-conda-install-path>/etc/profile.d/conda.sh
   ```

2. Create and activate the Polaris environment:
   ```bash
   conda env create --file environment.yaml
   conda activate polaris
   ```

## Usage

### Basic Command Structure
```bash
python polaris.py [options] --archspec <arch_config> --wlspec <workload_spec> --wlmapspec <mapping_spec>
```

### Key Command Line Options
- `--archspec,    -a`: Path to architecture specification YAML file
- `--wlspec,      -w`: Path to workload specification YAML file
- `--wlmapspec,   -m`: Path to workload mapping specification YAML file
- `--study,       -s`: Name for the simulation study (default: "study")
- `--odir,        -o`: Output directory for results (default: ".")
- `--outputformat`: Output format for results (none/yaml/json/pickle)
- `--dump_stats_csv`: Enable CSV stats output
- `--dryrun,      -n`: Perform a dry run without actual simulation

#### Data and Output Options
- `--datatype,    -d`: Activation data type (fp64/fp32/tf32/fp16/bf16/fp8/int32/int8)
- `--dump_ttsim_onnx`: Dump ONNX graph for TTSIM workloads

#### Sweep Specifications
- `--frequency`: Frequency (MHz) range specification (start end step)
- `--batchsize`: Batch size range specification (start end step)

#### Profiling and Analysis Options
- `--enable_memalloc`: Enable memory allocation simulation
- `--instr_profile`: Enable instruction profiling
- `--enable_cprofile`: Enable Python cProfile for performance analysis

#### Logging Options
- `--log_level,    -l`: Set logging level (debug/info/warning/error/critical, default: info)

### Filtering Options
- `--filterarch`: Filter architecture configurations
- `--filterwlg`: Filter workload groups
- `--filterwl`: Filter specific workloads
- `--filterwli`: Filter workload instances

## Configuration Files

### Architecture Specification
The architecture specification file (`archspec`) defines the hardware configuration including:
- Device specifications
- Memory hierarchy
- Compute resources
- Clock frequencies

### Workload Specification
The workload specification file (`wlspec`) defines:
- AI model configurations
- Batch sizes
- Input/output specifications
- Operator configurations

### Workload Mapping Specification
The workload mapping specification file (`wlmapspec`) defines:
- Operator to datatype mappings
- Resource requirements
- Operator fusion rules
- Null operations

## Performance Correlation Workflow

### Overview
Polaris includes tools to correlate simulation results with actual hardware measurements from TT-Metal, enabling validation and calibration of simulation accuracy.

### Workflow Steps

#### 1. Extract Hardware Metrics
Extract performance metrics from TT-Metal documentation or measurement data:

```bash
# Parse TT-Metal README to extract metrics
python tools/parse_ttsi_perf_results.py \
    --input https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/README.md \
    --output-dir __tmp/data/metal/inf
```

This will create categorized YAML files in `__tmp/data/metal/inf/`:
- `tensix_md_perf_metrics_llm.yaml` - LLM models
- `tensix_md_perf_metrics_vision.yaml` - Vision models
- `tensix_md_perf_metrics_detection.yaml` - Object detection models
- `tensix_md_perf_metrics_nlp.yaml` - NLP models
- `tensix_md_perf_metrics_diffusion.yaml` - Diffusion models

#### 2. Configure Workloads for Correlation
Create a workload configuration file (e.g., `config/ttsi_correlation_workloads.yaml`) defining the models to correlate:

```yaml
workloads:
  - api: TTSIM
    name: resnet50
    basedir: workloads
    module: BasicResNet@basicresnet.py
    instances:
      corr:  # Special instance for correlation
        bs: 32
        layers: 50
        # Additional parameters...
```

#### 3. Run Correlation Analysis
Execute the correlation tool to compare Polaris simulation with hardware measurements:

```bash
python tools/run_ttsi_corr.py \
    --tag 15oct25 \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml \
    --output-dir __CORRELATION_OUTPUT
```

**Key Options:**
- `--tag`: Tag identifying the parsed metrics dataset (required)
- `--input-dir`: Base directory containing tagged metrics (default: `data/metal/inf`)
- `--workloads-config`: Workload configuration for Polaris simulation
- `--arch-config`: Architecture specification (e.g., Wormhole)
- `--workload-filter`: Filter specific workloads by name
- `--precision`: Override precision for all workloads (e.g., bf8, bf16, fp32)
- `--output-dir`: Directory for correlation results
- `--dry-run`: Preview actions without executing

**Note:** The data source format (html/md) is automatically detected from metadata created by `parse_ttsi_perf_results.py`.

#### 4. Analyze Results
The correlation tool generates:

**CSV Output** (`correlation_result.csv`):
- Side-by-side comparison of hardware vs. simulation
- Calculated ratios and differences
- Easy to import into spreadsheets

**XLSX Output** (`correlation_result.xlsx`):
- Formatted tables with color coding
- Excel formulas for ratios
- Conditional formatting for quick visual analysis
- Frozen panes for easier navigation

**JSON Output** (`correlation_geomean.json`):
- Geometric mean of correlation ratios
- Aggregate accuracy metrics
- Machine-readable format for automation

**Example Results Structure:**
```
__CORRELATION_OUTPUT/
├── correlation_result.csv         # CSV format
├── correlation_result.xlsx        # Excel format with formatting
├── correlation_geomean.json       # Geometric mean metrics
├── inputs/                        # Saved configuration
│   ├── runinfo.json
│   ├── tensix_workloads.yaml
│   └── tensix_runcfg.yaml
├── SIMPLE/                        # Polaris simulation results
│   ├── CONFIG/
│   ├── STATS/
│   └── SUMMARY/
```

### Understanding Correlation Metrics

**Key Metrics in Results:**
- **Reference Performance**: Measured hardware performance (from TT-Metal)
- **Actual Performance**: Polaris simulation performance
- **Ratio**: Simulation / Hardware (ideally close to 1.0)
- **Score-to-Ref**: Ratio expressed as percentage
- **Geometric Mean**: Aggregate accuracy across all workloads

**Interpretation:**
- Ratio ≈ 1.0: Excellent correlation
- Ratio > 1.0: Simulation overestimates performance
- Ratio < 1.0: Simulation underestimates performance
- Geometric Mean: Overall simulation accuracy

### Advanced Usage

#### Custom Metric Parsing
For custom hardware data formats:

```python
from tools.parsers.md_parser import extract_table_from_md_link, TensixMdPerfMetricModel

# Extract from custom source
metrics = extract_table_from_md_link(custom_url)

# Filter or transform as needed
filtered_metrics = [m for m in metrics if m.hardware == 'n150']

# Save for correlation
from tools.parsers.md_parser import save_md_metrics
from pathlib import Path
save_md_metrics(filtered_metrics, Path('data/custom'))
```

#### Selective Correlation
Run correlation for specific models:

```bash
# Only correlate specific workloads using workload filter
python tools/run_ttsi_corr.py \
    --tag 15oct25 \
    --workloads-config config/ttsi_correlation_workloads.yaml \
    --arch-config config/tt_wh.yaml \
    --workload-filter bert,llama,resnet50 \
    --output-dir __CORRELATION_OUTPUT
```

#### Programmatic Access
Use correlation tools in Python scripts:

```python
from tools.run_ttsi_corr import main

# Run correlation programmatically
result = main([
    'run_ttsi_corr',
    '--tag', '15oct25',
    '--workloads-config', 'config/ttsi_correlation_workloads.yaml',
    '--arch-config', 'config/tt_wh.yaml',
    '--output-dir', 'output'
])
```

### Troubleshooting Correlation

**Issue: No metrics extracted**
- Verify URL accessibility and format
- Check that tables have required columns (model, batch, hardware)
- Enable debug logging: Add at start of script:
  ```python
  from loguru import logger
  import sys
  logger.remove()
  logger.add(sys.stdout, level='DEBUG')
  ```

**Issue: Workload not found in configuration**
- Ensure workload name matches between hardware data and config file
- Check for `corr` instance in workload configuration
- Verify workload module exists and is importable

**Issue: Large correlation discrepancies**
- Review workload parameters (batch size, model size, etc.)
- Verify architecture configuration matches hardware
- Check data types and precision settings
- Consider hardware-specific optimizations not modeled

## Output and Analysis

### Output Formats
Polaris supports multiple output formats:
- YAML: Human-readable structured format
- JSON: Web-friendly format
- Pickle: Binary format for Python objects
- CSV: Tabular format for statistics

### Key Metrics
The simulation provides various performance metrics including:
- Architecture package name (`archname`) - Identifies the architecture family (e.g., Grendel, Wormhole)
- Device instance name (`devname`) - Specific device instance (e.g., Q1_A1, n150)
- Execution cycles and time
- Memory usage and requirements
- Resource bottlenecks
- Input/output parameter counts
- Resource utilization

### Output Directory Structure
```
output_dir/
├── study_name/
│   ├── SUMMARY/
│   │   └── study-summary.csv       # Includes archname and devname columns
│   └── stats/
│       ├── device-workload-stats.csv
│       └── device-workload-opstats.[yaml|json|pickle]
```

**Note**: The `archname` field distinguishes architecture packages (e.g., Wormhole, Grendel) from device instances (`devname` field, e.g., n150, n300). This enables easier grouping and filtering of results by architecture family.

## Best Practices
1. Start with a dry run using `--dryrun` to validate configurations
2. Use filtering options to focus on specific architectures or workloads
3. Enable CSV output for easy data analysis
4. Use appropriate output formats based on your needs:
   - YAML for human readability
   - JSON for web integration
   - Pickle for Python processing
5. Monitor memory requirements using `--enable_memalloc`

## Troubleshooting

### Common Issues
1. Memory Constraints
   - Use `--enable_memalloc` to check memory requirements
   - Verify device memory specifications
   
2. Performance Issues
   - Enable profiling with `--enable_cprofile`
   - Check resource bottlenecks in output statistics

3. Configuration Errors
   - Validate YAML syntax in specification files
   - Use `--dryrun` to check configurations
   - Verify file paths and permissions

## Support
For issues and questions:
- Check the project repository: https://github.com/tenstorrent/polaris
- Review existing issues or create new ones
- Consult the development team for advanced support
