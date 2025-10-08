# NeoSim User Guide

## Table of Contents

1. [Overview](#overview)
   - [Key Features](#key-features)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Environment Setup](#environment-setup)
     - [1. Install Miniforge](#1-install-miniforge)
     - [2. Clone Repository](#2-clone-repository)
     - [3. Create Conda Environment](#3-create-conda-environment)
   - [Development Setup (Optional)](#development-setup-optional)
4. [Configuration Files](#configuration-files)
   - [Core Configuration Files](#core-configuration-files)
   - [Configuration versions](#configuration-versions)
5. [Execution](#execution)
   - [Default execution mode](#default-execution-mode)
     - [Minimal input configuration file](#minimal-input-configuration-file)
     - [Execute a test](#execute-a-test)
     - [Test output](#test-output)
   - [Custom execution mode](#custom-execution-mode)
   - [Managed execution mode](#managed-execution-mode)
     - [Optional prerequisites](#optional-prerequisites)
   - [Additional command-line arguments](#additional-command-line-arguments)
     - [Batch execution mode](#batch-execution-mode)
6. [Configuration Guide](#configuration-guide)
   - [gitignore](#gitignore)
   - [Debug Levels](#debug-levels)
   - [Multi-core input configuration file](#multi-core-input-configuration-file)
   - [Engine Configuration](#engine-configuration)
7. [Output and Analysis](#output-and-analysis)
   - [Output Files](#output-files)
   - [Chrome Trace Visualization](#chrome-trace-visualization)
   - [Performance Metrics](#performance-metrics)
8. [Troubleshooting](#troubleshooting)
   - [Common Issues](#common-issues)
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

## Quick Start

- Follow the steps to install NeoSim at [Installation](#installation)
- Follow steps to run NeoSim under [Execution](#execution)
- Follow steps to view output at [Output and Analysis](#output-and-analysis)

## Installation

The sections below describe steps to create appropriate conda environment. For more updated instructions for Miniforge installation please see the [Installation section](https://github.com/tenstorrent/polaris?tab=readme-ov-file#installation) from the Polaris README.

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

NeoSim uses JSON configuration files located in `config/tensix_neo/`
directory:

### Core Configuration Files
- **Input Configuration** (`ttqs_inputcfg_*.json`): Defines simulation
parameters, ELF paths, and test configuration.
- **Memory Map** (`ttqs_memory_map_*.json`): Specifies memory layout and
address mappings.
- **Architecture Configuration** (`ttqs_neo*.json`): Defines core
architecture parameters
- **Default Configuration** (`defCfg.json`): Engine definitions and
basic settings
- **Instruction Set Architecture file**: Defines instruction set for
given architecture. These are YAML files and are located
in `ttsim/config/llk/instruction_sets/ttqs` for Quasar (`ttqs`)
architecture.

### Configuration versions

Available configuration sets by LLK version tag:

- `jul1`  : LLK tests from RTL snapshot taken on July 1, 2025
- `jul27` : LLK tests from RTL snapshot taken on July 27, 2025
- `sep23` : LLK tests from RTL snapshot taken on September 23, 2025

The NeoSim maintains 100% test pass rate for all the LLK tests in the
snapshots above.

RTL tags `feb19` and `mar18` have been deprecated.

## Execution

The NeoSim model can be called via Python script
`ttsim/back/tensix_neo/tneoSim.py`.

An RTL LLK test can be executed via NeoSim model using the same
binary files (ELF) generated during the execution of the test via RTL.
The ELF files and associated parameters need to be specified in an
input configuration file for execution via the model.

NeoSim supports 3 modes of execution. They are as follows.

| Mode        | Best for                                        | Configuration required       |
|-------------|-------------------------------------------------|------------------------------|
| **Default** | Quick testing, learning                         | Minimal inputcfg only        |
| **Custom**  | Custom architectures, design evaluation         | Full configuration control   |
| **Managed** | Enterprise and Tenstorrent users, batch testing | Test name + version tag only |

Each execution mode is described in the following section in further
details.

### Default execution mode

Assuming ELF files are available, an RTL test can be executed via
the NeoSim model with just an input script.

The following sections explain how to write the input configuration
file and how to execute a given test.

#### Minimal input configuration file

Input configuration file, referred to as `inputcfg` hereafter, is a JSON
file. A minimal inputcfg JSON file is as follows.
```json
{
  "llkVersionTag":     "<llk tag>",
  "numTCores":         "<Number of TRISC Cores>",
  "input": {
    "name":            "test_simulation",
    "tc0": {
      "numThreads":    "<Number of threads in TRISC Core 0>",
      "startFunction": "<Entry Function in LLK (assumes same in each thread)>",
      "th0Elf":        "<ELF Filename of Thread0>",
      "th0Path":       "</path/to/elf/directory>",
      "th1Elf":        "<ELF Filename of Thread1>",
      "th1Path":       "</path/to/elf/directory>",
      "th2Elf":        "<ELF Filename of Thread2>",
      "th2Path":       "</path/to/elf/directory>",
      "th3Elf":        "<>",
      "th3Path":       "<>",
    }
  },
  "description": "<Description of the LLK>"
}
```

The JSON keys seen above are required inputs, further explanation about
them is as follows.
- **`llkVersionTag`**: Version identifier for LLK compatibility.
  This is a mandatory argument. Accepted values are listed in
  section `Configuration Versions`.
- **`numTCores`**: Number of Tensix cores to simulate.
- **`numThreads`**: Number of threads per core (1-4 supported)
- **`startFunction`**: Entry point function name (e.g. `main`).
- **`th*Elf`**: ELF binary filename for each thread (e.g.
`thread_0.elf`, `thread_1.elf`, etc.)
- **`th*Path`**: Directory path containing ELF binaries (e.g. path of
`thread_0.elf`, etc.)

#### Execute a test

Once the input configuration file is prepared, the tests can be executed
with
```bash
PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json
```

If the `polaris` directory is already present in `PYTHONPATH`, the
initial `PYTHONPATH` can be omitted.
```bash
python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json
```

#### Test output

A successful test would print
1. Estimated number of cycles for the given test e.g.
`Total Cycles =  6841.0`.
2. Summary of the test. This includes
   1. RISCV and tensix instruction count.
   2. Tensix instruction histogram by engines, mnemonics.
   3. Configuration register writes.
   3. Number of L1 and register accesses.
3. A chrome trace file for further analysis of instruction timings,
access patterns, etc. Please see section `Chrome Trace Visualization`
for more details.


### Custom execution mode

The users can specify custom architecture configuration, memory map,
instruction set, instruction throughput, etc. either via command-line
options or via inputcfg. The command-line arguments supersede arguments
from inputcfg which in turn supersede defaults.

The command-line arguments are as follows:
| Argument          | Type   | Required | Description                                      |
|-------------------|--------|----------|--------------------------------------------------|
| `--inputcfg`      | string | Yes      | Path to LLK configuration JSON file              |
| `--cfg`           | string | No       | Override architecture configuration file         |
| `--debug`         | int    | No       | Debug verbosity level (0-63, see Debug Levels)   |
| `--defCfg`        | string | No       | Override default configuration file              |
| `--exp`           | string | No       | Experiment prefix for log files (default: `neo`) |
| `--memoryMap`     | string | No       | Override memory map configuration file           |
| `--odir`          | string | No       | Output directory (default: `__llk`)              |
| `--risc.cpi`      | float  | No       | Override RISCV number of cycles per instruction  |
| `--ttISAFileName` | string | No       | Override instruction set file.                   |

Please see section `Debug Levels` for more details on debug flag values.

The defaults for `cfg`, `memoryMap` and `ttISAFileName` are tied
to `llkVersionTag` tag in `inputcfg`. E.g. for `llkVersionTag` of `jul27`,
the defaults would be
1. `cfg`: `config/tensix_neo/ttqs_neo4_jul27.json`
2. `memoryMap`: `config/tensix_neo/ttqs_memory_map_jul27.json`
3. `ttISAFileName`: `ttsim/config/llk/instruction_sets/ttqs/assembly.jul27.yaml`

Custom architecture configuration file (`cfg`) allows customisation of
tensix instruction throughput, L1 memory latency, etc. This is a JSON file.
The file also contains stack pointer information, this would be
deprecated in future versions.

The memory map, specifically config register space can be customised
via the memory map file. This is a JSON file.

Please note that custom tensix instruction set is accepted as YAML file,
and NOT as a JSON file.

If a custom file is provided, the simulator completely disregards the
defaults. Hence, please ensure both the modified and unmodified fields
are included in the new file.

Arguments such as `risc.cpi` (number of cycles per instruction for
RV32 instructions) can also be specified either via `cfg` file or via
command-line. The command-line arguments overrides the value provided
via cfg file.

Arguments `debug`, `memoryMap`, `ttISAFileName` can also be specified
via `inputcfg`. As before, command-line argument overrides any values
provided via inputcfg file.

The following example explains the overrides better. Consider
following files.
```bash
$ cat cfg.json
...
"risc.cpi" : 10
...

$ cat inputcfg.json
...
"llkVersionTag" : "jul27",
"debug"         : 15
"memoryMap"     : "memoryMap.inputcfg.json"
"ttISAFileName" : "isa.inputcfg.yaml"
...

# execution
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json --debug 63
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json --memoryMap memoryMap.cli.json
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json --cfg cfg.json
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json --cfg cfg.json --risc.cpi 2
$ PYTHONPATH="." python ttsim/back/tensix_neo/tneoSim.py --inputcfg inputcfg.json --risc.cpi 2 --cfg cfg.json
```

The snapshot above shows custom `cfg` file called `cfg.json`
where `risc.cpi` is set to a rather large value of 10. It also shows an
`inputcfg` where debug flag, and paths for custom `memoryMap` and
`ttISAFileName` are specified. The effect of command-line and inputcfg
arguments on the simulator execution environment is summarised in the table below.

| Execution                                                            | inputcfg        | cfg        | memoryMap                 | ttISAFileName       | debug | risc.cpi                  |
|----------------------------------------------------------------------|-----------------|------------|---------------------------|---------------------|-------|---------------------------|
| `tneoSim.py --inputcfg inputcfg.json`                                | `inputcfg.json` | Default    | `memoryMap.inputcfg.json` | `isa.inputcfg.yaml` | 15    | 1 (from default cfg file) |
| `tneoSim.py --inputcfg inputcfg.json --debug 63`                     | `inputcfg.json` | Default    | `memoryMap.inputcfg.json` | `isa.inputcfg.yaml` | 63    | 1 (from default cfg file) |
| `tneoSim.py --inputcfg inputcfg.json --memoryMap memoryMap.cli.json` | `inputcfg.json` | Default    | `memoryMap.cli.json`      | `isa.inputcfg.yaml` | 15    | 1 (from default cfg file) |
| `tneoSim.py --inputcfg inputcfg.json --cfg cfg.json`                 | `inputcfg.json` | `cfg.json` | `memoryMap.inputcfg.json` | `isa.inputcfg.yaml` | 15    | 10 (from custom cfg file) |
| `tneoSim.py --inputcfg inputcfg.json --cfg cfg.json --risc.cpi 2`    | `inputcfg.json` | `cfg.json` | `memoryMap.inputcfg.json` | `isa.inputcfg.yaml` | 15    | 2 (from command-line)      |
| `tneoSim.py --inputcfg inputcfg.json --risc.cpi 2 --cfg cfg.json`    | `inputcfg.json` | `cfg.json` | `memoryMap.inputcfg.json` | `isa.inputcfg.yaml` | 15    | 2 (from command-line)      |

The inputcfg arguments take precedence over defaults. The command-line arguments
take precedence over both arguments from inputcfg and defaults. This
precedence is absolute and does not depend upon sequence in which the
arguments are provided.

The output of test execution is same that described in section
`Test output` in `Default execution mode`.

### Managed execution mode.

To further simplify the execution of RTL tests, enterprise and
Tenstorrent users may use `tools/run_rtl_neosim_correlation.py` script.

#### Optional prerequisites

The following steps are performed within the `run_rtl_neosim_correlation.py` script
(described next) as well, but can be carried out explicitly if required.
1. **Network connectivity check**: Check that you are connected to
  Tenstorrent Tailscale enabled network.
   ```bash
   cd polaris
   bash ./tools/ci/check_behind_tailscale.sh
   ```
   If successful the script will print `Running behind Tailscale`. The
   artifact download from the next step will succeed only if
   this check passes.
2. **RTL LLK test data download**: Download LLK test data for a
  particular `llkVersionTag` (`sep23` in the example below) with
   ```bash
   cd polaris
   bash ./tools/ci/lfc_downloader.sh --extract ext_rtl_test_data_set_sep23.tar.gz
   ```
   Please replace the tag with appropriate value of `llkVersionTag` to
   download artifacts from other snapshots. Please see
   `./doc/tools/ci/lfc_downloader_user_guide.md` for further details.

##### Why explicit download of RTL test data:
- To verify network connectivity before running multiple tests
- To pre-download test data for offline use
- To troubleshoot download issues

In managed mode, the users are not required provide `inputcfg`, instead
this script only requires LLK test name and `llkVersionTag`. The given
test can be executed as follows.
```bash
python tools/run_rtl_neosim_correlation.py --tag jul27 --test t6-quas-n1-ttx-Int32-upk-to-dest-llk
```
Here test `t6-quas-n1-ttx-Int32-upk-to-dest-llk` from RTL snapshot from
Jul 27, 2025 will be executed. The test uses default `cfg`, `memoryMap`
and `ttISA` files.

If not present, the test data (ELF files, instruction set) for a given
`llkVersionTag` is downloaded with help of `lfc_downloader.sh` as
explained above.

In addition to the test output specified earlier, the managed execution
mode also provides comparison between number of cycles from RTL test
execution and those estimated by the NeoSim model for given test.

The comparison is printed in tabular format to the `stdout` and stored
as a plot. The paths of the plots are printed to `stdout`.

The managed mode also allows for additional execution options.
1. `run_rtl_neosim_correlation.py --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk --tag jul27`
   executes a test associated with `llkVersionTag` `jul27`.
1. `run_rtl_neosim_correlation.py --tag jul27 --test t6-quas-n1-ttx-Int32-upk-to-dest-llk t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk`
   executes both the tests associated with `llkVersionTag` `jul27`.
1. `run_rtl_neosim_correlation.py --tag sep23` executes _all_ LLK tests associated
   with given `llkVersionTag` (`sep23` in this case). The execution of tests proceeds in serial manner, that is tests are executed in one after the other.
1. `run_rtl_neosim_correlation.py --tag sep23 --parallel 2` speeds up execution with
   two processes running in parallel.
1. `run_rtl_neosim_correlation.py --tag jul1 sep23 --parallel 2` multiple
   `llkVersionTag`s can be specified. This will execute all tests for
   each `llkVersionTag`.

#### Additional command-line arguments
Except for `inputcfg`, the script `run_rtl_neosim_correlation.py` also supports
all command line arguments listed in section `Custom execution mode`.

An example of these arguments is as follows.
```bash
python tools/run_rtl_neosim_correlation.py \
  --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk \
  --tag sep23 \
  --debug 63 \
  --cfg cfg.json \
  --memoryMap mm.json \
  --ttISAFileName ttisa.yaml \
  --risc.cpi 2.0 \
  --odir __elw_dir \
  --batch-name my_run
```
This command will execute test `t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk` from tag `sep23`.
The debug flag for this execution is set to 63. Configuration file used will be `cfg.json`.
Similarly memory map and tensix instruction set will be read from `mm.json` and `ttisa.yaml`
respectively. `risc.cpi` has been set to 2. The results of the run will be stored in `__elw_dir`.
The command line also accepts optional `batch-name` argument for the given run/execution.
The default for `batch-name` is "baseline".

The users are required to exercise caution when using `cfg`, `memoryMap` and `ttISAFileName`
arguments across multiple tags as command line arguments override defaults. E.g.
```bash
python tools/run_rtl_neosim_correlation.py \
  --parallel 10 \
  --tag jul1 sep23 \
  --debug 63 \
  --cfg cfg.json \
  --memoryMap mm.json \
  --ttISAFileName ttisa.yaml \
  --risc.cpi 2.0 \
  --odir __elw_dir
```
This will use the same `cfg.json`, `mm.json` and `ttisa.yaml` for tests
with tag `jul1` and for tests with tag `sep23`. This will most likely
lead to errors as the memory map of these two tags is not the same
(configuration registers are differ in both name and number).
Differences can also be seen in `assembly.jul1.yaml` and
`assembly.sep23.yaml`.

#### Batch execution mode
The `tools/run_rtl_neosim_correlation.py` script supports execution in
batch mode via the following argument.
```bash
python run_rtl_neosim_correlation.py --batch-file batch_file.yaml
```
An example batch file is as follows.
```yaml
batch:
  - name: run_sep23
    tag: sep23
    parallel: 10

  - name: cpi1
    test: t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk
    tag: sep23
    debug: 15

  - name: cpi2
    test:
    - t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk
    tag:
    - sep23
    risc.cpi: 2.0
    debug: 15
    cfg: cfg.json
    ttISAFileName: ttisa.yaml
    memoryMap: mm.json
    odir: abc
```
The batch file introduces two additional keys namely `batch` and `name`.
The `batch` is a mandatory key and has to be at the start of the file.
`name` is optional for a given run/batch. If not specified a script
generated name is assigned to the runs.

If two runs/batches have the same name, appropriate counters are
suffixed to the names of the repeat runs.

In the script above, run `run_sep23` is equivalent to
`run_rtl_neosim_correlation.py --tag sep23 --parallel 10`. This runs all the tests
with tag `sep23` in parallel mode with 10 number of processes.

Run `cpi1` is equivalent to
```bash
python tools/run_rtl_neosim_correlation.py \
  --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk \
  --tag sep23 \
  --debug 15 \
  --batch-name cpi1
```

Run `cpi2` is equivalent to
```bash
python tools/run_rtl_neosim_correlation.py \
  --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk \
  --tag sep23 \
  --debug 15 \
  --risc.cpi 2.0 \
  --cfg cfg.json \
  --ttISAFileName ttisa.yaml \
  --memoryMap mm.json \
  --odir abc \
  --batch-name cpi2
```
The results of runs `run_sep23` and `cpi1` will be stored at default
locations, whereas those for `cpi2` will be stored at `abc`.

#### Config file generation.

A config file is also written to file for a given run after execution
is complete.  E.g. A config file for
```bash
python tools/run_rtl_neosim_correlation.py \
  --test t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk \
  --tag sep23 \
  --debug 15 \
  --cfg cfg.json \
  --memoryMap mm.json \
  --ttISAFileName ttisa.yaml \
  --risc.cpi 2.0 \
  --odir abc
```
is written to `__config_files/baseline/batch_baseline.yaml`.
`baseline` is the name assigned to the run in this case. `batch`
is the prefix assigned to the file. The run config would be:
```yaml
batch:
- cfg: cfg.json
  debug: 15
  memoryMap: mm.json
  name: baseline
  odir: abc
  parallel: 1
  risc.cpi: 2.0
  tag:
  - sep23
  test:
  - t6-quas-n1-ttx-elwadd-broadcast-col0-fp16-llk
  ttISAFileName: ttisa.yaml
```

## Configuration Guide

### gitignore

All files and directories starting with `__` are not staged for commits.
Users may use this a gitignore character for files and directories that
do not need to be committed and pushed.

### Debug Levels

Debug levels are bitwise flags that can be combined:

| Level | Component | Detail | Description                    |
|-------|-----------|--------|--------------------------------|
| 1     | TRISC     | Low    | Basic RISC core activity       |
| 2     | Tensix    | Low    | Basic Tensix core activity     |
| 4     | TRISC     | Medium | Detailed RISC instruction flow |
| 8     | Tensix    | Medium | Detailed pipeline activity     |
| 16    | TRISC     | High   | Full RISC debug output         |
| 32    | Tensix    | High   | Full pipeline and memory debug |

**Examples:**
- `--debug 3`: TRISC + Tensix low detail (1 + 2)
- `--debug 15`: All components medium detail (1 + 2 + 4 + 8)
- `--debug 63`: Maximum debug output (all flags)

### Multi-core input configuration file
A multiple tensix cores and ELF files associated with them can be
specified in `inputcfg` file as follows.
```json
{
  "llkVersionTag": "jul27",
  "debug": 15,
  "ttISAFileName": "path/to/instruction_set/assembly.yaml",
  "numTCores": 4,
  "input": {
    "syn": 0,
    "name": "t6-quas-n4-ttx-matmul-l1-acc-multicore-2d-matmul-4-tiles-llk",
    "tc0": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "thread_0.elf",
      "th0Path": "path/to/elf/neo_0/thread_0/out",
      "th1Elf": "thread_1.elf",
      "th1Path": "path/to/elf/neo_0/thread_1/out",
      "th2Elf": "thread_2.elf",
      "th2Path": "path/to/elf/neo_0/thread_2/out",
      "th3Elf": "",
      "th3Path": ""
    },
    "tc1": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "thread_0.elf",
      "th0Path": "path/to/elf/neo_1/thread_0/out",
      "th1Elf": "thread_1.elf",
      "th1Path": "path/to/elf/neo_1/thread_1/out",
      "th2Elf": "thread_2.elf",
      "th2Path": "path/to/elf/neo_1/thread_2/out",
      "th3Elf": "",
      "th3Path": ""
    },
    "tc2": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "thread_0.elf",
      "th0Path": "path/to/elf/neo_2/thread_0/out",
      "th1Elf": "thread_1.elf",
      "th1Path": "path/to/elf/neo_2/thread_1/out",
      "th2Elf": "thread_2.elf",
      "th2Path": "path/to/elf/neo_2/thread_2/out",
      "th3Elf": "",
      "th3Path": ""
    },
    "tc3": {
      "numThreads": 4,
      "startFunction": "main",
      "th0Elf": "thread_0.elf",
      "th0Path": "path/to/elf/neo_3/thread_0/out",
      "th1Elf": "thread_1.elf",
      "th1Path": "path/to/elf/neo_3/thread_1/out",
      "th2Elf": "thread_2.elf",
      "th2Path": "path/to/elf/neo_3/thread_2/out",
      "th3Elf": "",
      "th3Path": ""
    }
  },
  "description": {}
}
```

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

This section further expands explanation of the output from tests
described earlier ([Test output](#test-output)).

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

5. **Could not find `tests/standalone/execute_test.py`**
   The `execute_test.py` script has been renamed to `run_rtl_neosim_correlation.py` and also has been
   moved from `polaris/tests/standalone` to `polaris/tools` directory. Please execute
   `python tools/run_rtl_neosim_correlation.py` with the same command-line arguments.

## Support

- **Documentation**: See `doc/` directory for detailed technical documentation (to be updated).
- **Issues**: Report bugs via Github (https://github.com/tenstorrent/polaris/issues).
- **Configuration**: Reference existing configurations in `config/tensix_neo/`.

---

**Version**: Compatible with Polaris v2024.x and later
**Last Updated**: September 2025