# NeoSim User Guide

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Environment Setup](#environment-setup)
3. [Configuration Files](#configuration-files)
4. [Execution](#execution)
   - [Basic Execution](#basic-execution)
   - [Scripts Based Execution](#scripts-based-execution)
5. [Quick Start](#quick-start)
6. [Configuration Guide](#configuration-guide)
7. [Output and Analysis](#output-and-analysis)
8. [Troubleshooting](#troubleshooting)
9. [Support](#support)

## Overview

NeoSim is a cycle-accurate simulator for Tenstorrent's Tensix Neo architecture, part of the Polaris AI simulation framework. It provides detailed performance analysis of Low-Level Kernel (LLK) execution on Tensix cores, including pipeline modeling, memory hierarchy simulation, and instruction-level analysis.

### Key Features

- **Cycle-accurate simulation** of Tensix Neo cores with configurable pipeline stages
- **Multi-core simulation** supporting multiple Tensix cores with shared L1 memory
- **ELF binary execution** from compiled LLK kernels
- **Pipeline modeling** including unpacker, SFPU, matrix, and packer engines
- **Memory hierarchy simulation** with L1 cache, register files, and memory mapping
- **Performance analysis** with detailed instruction traces and Chrome trace output
- **Configurable architecture** supporting different Tensix variants (TTQS, TTWH, TTBH)

## Installation

The sections below describe steps to create appropriate conda environment. For more updated instructions for Miniforge installation please see section `Installation` from `polaris/README.md`. 

### Prerequisites

- **Operating System**: Linux, macOS, or Windows (with WSL)
- **Python**: 3.13.2 (recommended via miniforge)
- **Git**: For repository access
- **Storage**: ~2GB free space for environment and dependencies

### Environment Setup

#### 1. Install Miniforge

Install miniforge as described in https://github.com/conda-forge/miniforge:

```bash
# Download miniforge installer
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Run installer
bash Miniforge3-$(uname)-$(uname -m).sh

# Update conda
conda update -n base -c conda-forge conda
```

#### 2. Clone Repository

```bash
git clone https://github.com/tenstorrent/polaris.git
cd polaris
```

#### 3. Create Conda Environment

```bash
# Create environment from configuration
conda env create --file environment.yaml

# Activate environment
conda activate polaris

# Verify installation
python -c "import ttsim; print('NeoSim installation successful')"
```

### Development Setup (Optional)

For contributors and developers:

```bash
# Create development environment with additional tools
conda env create --file envdev.yaml --name polaris-dev
conda activate polaris-dev
```

## Configuration Files

NeoSim uses JSON configuration files located in `config/tensix_neo/` directory:

### Core Configuration Files

- **Input Configuration** (`ttqs_inputcfg_*.json`): Defines simulation parameters, ELF paths, and test configuration
- **Memory Map** (`ttqs_memory_map_*.json`): Specifies memory layout and address mappings
- **Architecture Configuration** (`ttqs_neo*.json`): Defines core architecture parameters
- **Default Configuration** (`defCfg.json`): Engine definitions and basic settings

### Configuration Versions

Available configuration sets by LLK version tag:

- `sep23`   : September 23rd version 
- `jul27`   : July 27th version 
- `jul1`    : July 1st version  
- `mar18`   : March 2018 version
- `feb19`   : February 2019 version

## Execution

### Basic Execution

```bash
Update th*Path in config/tensix_neo/ttqs_inputcfg_jul27.json. Refer to command-line arguments 
export PYTHONPATH=<polaris directory>
cd "<polaris directory>"

# Run single LLK
python ttsim/back/tensix_neo/tneosim.py \
    --cfg config/tensix_neo/ttqs_neo4_jul27.json \
    --inputcfg config/tensix_neo/ttqs_inputcfg_jul27.json

# Run single LLK with specific output directory
python ttsim/back/tensix_neo/tneosim.py \
    --cfg config/tensix_neo/ttqs_neo4_jul27.json \
    --inputcfg config/tensix_neo/ttqs_inputcfg_jul27.json \
    --odir __logs/llk_jul27

# Run single LLK with medium debug detail
python ttsim/back/tensix_neo/tneosim.py \
    --cfg config/tensix_neo/ttqs_neo4_jul27.json \
    --inputcfg config/tensix_neo/ttqs_inputcfg_jul27.json \
    --debug 15

```

Note: Output directories starting with '__' as specified in gitignore will not get accidentally included under version control. So --odir can point to base directories satisfying this condition

### Scripts Based Execution

```bash
# Run single LLK
cd "<polaris directory>"
python tests/standalone/execute_test.py \
    --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk \
    --tag jul27

# Run all LLK serial
python tests/standalone/execute_test.py \
    --tag jul27

# Run all LLK with two runs in parallel
python tests/standalone/execute_test.py \
    --tag jul27 \
    --parallel 2

```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--inputcfg` | string | Yes | Path to LLK configuration JSON file |
| `--debug` | int | No | Debug verbosity level (0-63, see Debug Levels) |
| `--cfg` | string | No | Override architecture configuration file |
| `--memoryMap` | string | No | Override memory map configuration file |
| `--defCfg` | string | No | Override default configuration file |
| `--risc.cpi` | float | No | RISC cycles per instruction override |
| `--odir` | string | No | Output directory (default: `__llk`) |
| `--exp` | string | No | Experiment prefix for log files (default: `neo`) |

### Debug Levels

Debug levels are bitwise flags that can be combined:

| Level | Component | Detail | Description |
|-------|-----------|---------|-------------|
| 1 | TRISC | Low | Basic RISC core activity |
| 2 | Tensix | Low | Basic Tensix core activity |
| 4 | TRISC | Medium | Detailed RISC instruction flow |
| 8 | Tensix | Medium | Detailed pipeline activity |
| 16 | TRISC | High | Full RISC debug output |
| 32 | Tensix | High | Full pipeline and memory debug |

**Examples:**

- `--debug 3`: TRISC + Tensix low detail (1 + 2)
- `--debug 15`: All components medium detail (1 + 2 + 4 + 8)
- `--debug 63`: Maximum debug output (all flags)

## Quick Start

- Follow the steps to install NeoSim at [Installation](#installation)
- Follow steps to run NeoSim under [Execution](#execution)
- Follow steps to view output at [Output and Analysis](#output-and-analysis)

## Configuration Guide

### Input Configuration Format

```json
{
  "llkVersionTag": "<llk tag>",
  "cfg": "<Architecture Configuration JSON>",
  "memoryMap": "<Memory Map JSON>",
  "debug": "<debug level>",
  "numTCores": "<Number of TRISC Cores>",
  "input": {
    "name": "test_simulation",
    "tc0": {
      "numThreads": "<Number of threads in TRISC Core 0>",
      "startFunction": "<Entry Function in LLK (assumes same in each thread)>",
      "th0Elf": "<ELF Filename of Thread0>",
      "th0Path": "</path/to/elf/directory>",
      "th1Elf": "<ELF Filename of Thread1>",
      "th1Path": "</path/to/elf/directory>",
      "th2Elf": "<ELF Filename of Thread2>",
      "th2Path": "</path/to/elf/directory>",
    }
  },
  "description": {"<Description of the LLK>"}
}
```

### Key Configuration Parameters

- **`llkVersionTag`**: Version identifier for LLK compatibility
- **`numTCores`**: Number of Tensix cores to simulate
- **`numThreads`**: Threads per core (1-4 supported)
- **`startFunction`**: Entry point function name
- **`th*Elf`**: ELF binary filename for each thread
- **`th*Path`**: Directory path containing ELF binaries

### Engine Configuration

Edit `Architecture Configuration JSON file` to customize instruction to engine mapping:

```json
{
  "engines": [
    {
      "engineName": "unpacker",
      "engineInstructions": ["unpacr"]
    },
    {
      "engineName": "sfpu", 
      "engineInstructions": ["dotpv"]
    },
    {
      "engineName": "matrix",
      "engineInstructions": ["elwadd", "elwsub", "elwmul", "mvmul"]
    },
    {
      "engineName": "packer",
      "engineInstructions": ["pacr"]
    }
  ]
}
```

## Output and Analysis

### Output Files

NeoSim generates several output files in the specified output directory:

- **`simreport_<exp>_tc<N>_<name>.json`**: Chrome trace format for visualization
- **Simulation logs**: Text logs with cycle-by-cycle execution details
- **Performance statistics**: Instruction counts, cycle counts, memory access stats

### Chrome Trace Visualization

1. Open Google Chrome browser
2. Navigate to `chrome://tracing/`
3. Load the generated JSON trace file
4. Analyze pipeline utilization, instruction timing, and memory access patterns

### Performance Metrics

Key metrics available in simulation output:

- **Total execution cycles**
- **Instructions per cycle (IPC)**
- **Pipeline utilization per engine**
- **Memory access latencies**
- **L1 cache hit/miss ratios**

## Troubleshooting

### Common Issues

1. **ELF file not found**
   - Verify `th*Path` points to correct directory
   - Ensure ELF files exist and are readable
   - Check file permissions

2. **Configuration validation errors**
   - Validate JSON syntax with `python -m json.tool config.json`
   - Ensure all required fields are present
   - Check file paths are absolute or relative to working directory

3. **Memory mapping errors**
   - Verify memory map configuration matches ELF layout
   - Ensure sufficient simulated memory is allocated
   - Check for address conflicts between threads

4. **Performance issues**
   - Reduce debug level for faster simulation
   - Use smaller test cases for development
   - Monitor system memory usage during simulation

## Support

- **Documentation**: See `doc/` directory for detailed technical documentation
- **Examples**: Check `tests/tensix_neo/` for test cases and examples
- **Issues**: Report bugs through the project's issue tracking system
- **Configuration**: Reference existing configurations in `config/tensix_neo/`

---

**Version**: Compatible with Polaris v2024.x and later  
**Last Updated**: September 2025
