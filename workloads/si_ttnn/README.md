# SI TTNN Workloads

This directory contains TTNN (Tenstorrent Neural Network) workload implementations for silicon-specific models and operations.
Note that these are intended to be run on TT Silicon, and not on polaris. 
These models are hosted in the polaris repository, since these have been written for
the purpose of performance correlation with polaris.

## Overview

SI TTNN workloads are designed for direct execution on Tenstorrent hardware, providing high-fidelity performance simulation and analysis. These workloads use the TTNN API to implement neural network models with hardware-native operations.

## Directory Structure

- `vit/` - Vision Transformer (ViT) model implementations
  - `test_ttnn_functional_vit.py` - Unit tests for ViT components

## Usage

Example use: 
- Copy e.g. vit/test_ttnn_functional_vit.py to models/demos/vit/tests/pcc in tt-metal. 
- Then execute `pytest models/demos/vit/tests/pcc/test_ttnn_functional_vit.py::test_vit_attention`. 
- Change the file name and test name to suit your requirements
- To run TTNN profiler, execute `python tools/tracy/profile_this.py -n vitattention -c "pytest models/demos/vit/tests/pcc/test_ttnn_functional_vit.py::test_vit_attention" --collect-noc-traces` after building tt-metal.

