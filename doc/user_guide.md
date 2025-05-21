# Polaris User Guide

## Overview
Polaris is a high-level simulator for performance analysis of AI architectures. It enables users to analyze and evaluate the performance characteristics of AI workloads on different hardware architectures through simulation.

## Key Features
- Workload Analysis: Input AI workloads are converted into DAG (Directed Acyclic Graph) representations
- Architecture Simulation: Simulate workloads on different hardware configurations
- Performance Analysis: Get detailed performance metrics and resource utilization data
- Flexible Output Formats: Support for YAML, JSON, and Pickle output formats

## Installation

### Prerequisites
- Python 3.13 or higher
- Miniforge package manager

### Setup Steps
1. Install Miniforge:
   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   conda update -n base -c conda-forge conda
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
- `-a, --archspec`: Path to architecture specification YAML file
- `-w, --wlspec`: Path to workload specification YAML file
- `-m, --wlmapspec`: Path to workload mapping specification YAML file
- `-s, --study`: Name for the simulation study (default: "study")
- `-o, --odir`: Output directory for results (default: ".")
- `--outputformat`: Output format for results (none/yaml/json/pickle)
- `--dump_stats_csv`: Enable CSV stats output
- `-n, --dryrun`: Perform a dry run without actual simulation
- `--enable_memalloc`: Enable memory allocation simulation
- `--instr_profile`: Enable instruction profiling
- `--enable_cprofile`: Enable Python cProfile for performance analysis

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

## Output and Analysis

### Output Formats
Polaris supports multiple output formats:
- YAML: Human-readable structured format
- JSON: Web-friendly format
- Pickle: Binary format for Python objects
- CSV: Tabular format for statistics

### Key Metrics
The simulation provides various performance metrics including:
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
│   │   └── study-summary.csv
│   └── stats/
│       ├── device-workload-stats.csv
│       └── device-workload-opstats.[yaml|json|pickle]
```

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
