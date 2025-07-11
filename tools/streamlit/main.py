import streamlit as st
import os
import yaml
from utils import initApp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

st.set_page_config(
    page_title="TT Polaris Simulator",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cutive+Mono&display=swap');
    html, body, [class*="st-"], h1, h2, h3, h4, h5, h6 {
        font-family: 'Cutive Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize app
initApp()

# Main page content
st.markdown("""
<h1 style="text-align: center; color: #0D47A1; margin-bottom: 20px;">
    Tenstorrent Polaris Simulator
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 10px; border-radius: 5px; margin-bottom: 20px;">
Welcome to the Tenstorrent Polaris Simulator. Please use the navigation on the left to configure, run, and review your simulations.
</div>
""", unsafe_allow_html=True)

st.markdown("""
### <span style="color: #0277bd;">Overview</span>

*Polaris* is a high-level simulator for performance analysis of AI architectures. It takes an *AI Workload* and an *Architecture Configuration* as input to simulate and analyze performance metrics such as throughput, utilization, memory bandwidth, and power efficiency. This Streamlit application provides a user-friendly interface for interacting with Polaris.

### <span style="color: #0277bd;">Using the Streamlit Interface</span>

This application is designed to simplify the process of running Polaris simulations. Below is a guide to the different pages available in the app:

-   **<span style="color: #01579b;">Configure Architecture</span>**: This page allows you to create, view, and edit architecture specification files. You can load a predefined configuration, modify it to suit your needs, and save it for use in simulations. Key parameters include:
    - Compute resources (cores, units, ALUs)
    - Memory hierarchy (L1/L2/HBM capacities and bandwidths)
    - Network-on-chip specifications
    - Clock frequencies

-   **<span style="color: #01579b;">Configure Workload</span>**: On this page, you can define the workload and mapping specifications. This includes:
    - Setting up the AI model (operators, dimensions, dependencies)
    - Specifying batch sizes and input dimensions
    - Defining precision (FP32, FP16, INT8)
    - Configuring operator-specific parameters
    - Defining how operators are mapped to hardware resources and data types

-   **<span style="color: #01579b;">Run Simulation</span>**: Use this page to launch a simulation. You can:
    - Select your architecture, workload, and mapping configurations
    - Set trace levels (from 0-5) for debugging
    - Enable performance profiling
    - Specify additional command-line options
    - Monitor simulation progress in real-time

-   **<span style="color: #01579b;">View Results</span>**: Once a simulation is complete, this page provides tools for analyzing the results:
    - Interactive performance metrics dashboards
    - Layer-by-layer execution breakdown
    - Memory utilization graphs
    - Bottleneck identification
    - Comparative analysis between simulation runs

### <span style="color: #0277bd;">The Polaris Simulator</span>

The simulator can be run from the command line with the following structure:

```bash
python polaris.py [options] --archspec <arch_config> --wlspec <workload_spec> --wlmapspec <mapping_spec>
```

#### <span style="color: #01579b;">Key Inputs:</span>

*   **Architecture Specification (`--archspec`)**: A YAML file defining the hardware configuration (device specs, memory, compute resources).
    ```yaml
    # Example arch_config.yaml snippet
    device:
      name: "example_chip"
      compute_units: 256
      memory:
        l1_capacity_bytes: 2097152  # 2MB
        l2_capacity_bytes: 33554432  # 32MB
        hbm_bandwidth_gbps: 1024
    ```

*   **Workload Specification (`--wlspec`)**: A YAML file defining the AI model, batch sizes, and operator configurations.
    ```yaml
    # Example wlspec.yaml snippet
    model:
      name: "example_model"
      batch_size: 32
      operators:
        - name: "conv1"
          type: "conv2d"
          input_dimensions: [3, 224, 224]
          kernel_dimensions: [64, 3, 7, 7]
          stride: [2, 2]
    ```

*   **Workload Mapping Specification (`--wlmapspec`)**: A YAML file that defines how operators are mapped to datatypes and hardware resources.
    ```yaml
    # Example wlmapspec.yaml snippet
    mappings:
      - operator: "conv1"
        datatype: "fp16"
        compute_units: 128
        tiling_strategy: "spatial"
    ```

#### <span style="color: #01579b;">Common Options:</span>

*   `--study, -s`: Name for the simulation study (e.g., `--study resnet50_optimization`)
*   `--odir, -o`: Output directory for results (e.g., `--odir /path/to/results`)
*   `--dryrun, -n`: Validate configurations without running the full simulation
*   `--dump_stats_csv`: Enable CSV output for easier analysis and data export
*   `--trace_level`: Set verbosity level (0-5) for debugging (e.g., `--trace_level 3`)
*   `--profile`: Enable performance profiling of the simulator itself
*   `--threads`: Number of threads to use for parallel simulation (e.g., `--threads 8`)

#### <span style="color: #01579b;">Advanced Features:</span>

* **Power Modeling**: Enable with `--power_model` to estimate chip power consumption
* **Thermal Analysis**: Use `--thermal_model` to simulate thermal behavior under load
* **Batch Processing**: Run multiple configs with `--batch_config batch_file.json`
* **Visualization**: Generate SVG/PNG diagrams with `--gen_visuals`

To get started quickly, try one of our example configurations:
```bash
# Example command to run a ResNet50 simulation
python polaris.py --study resnet50 --archspec examples/arch/default_arch.yaml --wlspec examples/models/resnet50.yaml --wlmapspec examples/mappings/default_mapping.yaml --dump_stats_csv
```

""", unsafe_allow_html=True)