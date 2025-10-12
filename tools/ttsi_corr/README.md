# TTSI Correlation Package

Modular components for Metal-Tensix correlation analysis between Polaris HLM projections and silicon performance measurements.

## Overview

This package provides a clean, modular architecture for performing correlation analysis. It was created by refactoring `tools/run_ttsi_corr.py` into reusable, testable components.

## Package Structure

```
ttsi_corr/
├── __init__.py           # Package API and exports
├── data_loader.py        # Load and validate reference metrics
├── workload_processor.py # Process workload configurations
├── correlation.py        # Calculate correlation metrics
├── excel_writer.py       # Generate Excel reports
├── chart_builder.py      # Create visualization charts
├── simulator.py          # Orchestrate Polaris simulation
└── README.md            # This file
```

## Modules

### `data_loader`
Load and validate reference performance metrics from various sources (HTML tables, Markdown files).

**Key Functions:**
- `load_metrics_from_sources()` - Load metrics based on data source type
- `read_metadata()` - Read metadata from data directory
- `load_html_metrics()` - Load HTML format metrics
- `load_md_metrics()` - Load Markdown format metrics
- `normalize_md_metric()` - Normalize MD metrics to standard format

**Example:**
```python
from ttsi_corr.data_loader import load_metrics_from_sources, read_metadata

# Read metadata to determine data source
metadata = read_metadata(Path('data/metal/inf/15oct25'))
data_source = metadata.get('data_source')

# Load metrics
metrics = load_metrics_from_sources(
    tensix_perf_data_dir=Path('data/metal/inf/15oct25'),
    data_source=data_source
)
```

### `workload_processor`
Process and validate workload configurations, creating workload specifications for simulation.

**Key Functions:**
- `process_workload_configs()` - Process all configurations
- `process_single_config()` - Process a single configuration
- `find_workload_config()` - Find workload by name with fuzzy matching
- `get_workload_module_config()` - Get module configuration for workload

**Example:**
```python
from ttsi_corr.workload_processor import process_workload_configs

ttsim_wlspec, ref_scores, devices = process_workload_configs(
    all_configs=metrics,
    workloads_file=wl_file,
    workload_filter=None,
    default_precision='fp16',
    device_table={'n300': 'n300'},
    correlation_instance_name='corr'
)
```

### `correlation`
Compare and analyze correlation between reference and projected scores.

**Key Functions:**
- `compare_scores()` - Compare reference vs actual scores
- `read_scores()` - Read scores from simulation JSON
- `calculate_and_save_geomean()` - Calculate and save geometric mean

**Example:**
```python
from ttsi_corr.correlation import compare_scores, calculate_and_save_geomean

# Compare scores
comparison = compare_scores(
    ref_scores=silicon_scores,
    actual_scores=hlm_scores,
    include_override_precision=True
)

# Calculate geometric mean
geomean = calculate_and_save_geomean(
    comparison=comparison,
    output_path=Path('output/correlation_geomean.json')
)
```

### `excel_writer`
Generate Excel reports with formatted tables and charts.

**Key Classes:**
- `ExcelFormatter` - Utilities for Excel formatting and layout
  - `col_letter()` - Convert column index to Excel letter
  - `apply_number_formats()` - Apply number formatting
  - `apply_borders()` - Apply borders to cells
  - `apply_freeze_panes()` - Configure frozen panes

**Key Functions:**
- `write_csv()` - Write comparison results to CSV

**Example:**
```python
from ttsi_corr.excel_writer import ExcelFormatter, write_csv

# Use Excel formatting utilities
formatter = ExcelFormatter()
col_name = formatter.col_letter(1)  # Returns 'A'

# Write CSV
write_csv(
    comparison=correlation_data,
    output_path=Path('output/correlation_result.csv')
)
```

### `chart_builder`
Create visualization charts for correlation analysis.

**Key Classes:**
- `ScurveChartBuilder` - Builder for S-curve analysis charts (Builder pattern)
  - `prepare_data()` - Extract and sort ratio data
  - `calculate_statistics()` - Calculate min, max, median, geomean
  - `create_worksheet()` - Create worksheet with title and stats
  - `add_data_table()` - Add sorted data table
  - `create_chart()` - Create and configure line chart
  - `build()` - Build complete S-curve sheet

**Key Functions:**
- `add_scurve_chart()` - High-level facade for chart creation

**Example:**
```python
from ttsi_corr.chart_builder import add_scurve_chart
from openpyxl import Workbook

wb = Workbook()
add_scurve_chart(
    workbook=wb,
    comparison=correlation_data
)
wb.save('output/correlation_result.xlsx')
```

### `simulator`
Orchestrate Polaris simulation execution.

**Key Functions:**
- `validate_and_filter_configs()` - Validate and filter configurations
- `run_polaris_simulation()` - Execute Polaris simulation
- `validate_workload_filter()` - Validate workload filter

**Example:**
```python
from ttsi_corr.simulator import validate_and_filter_configs, run_polaris_simulation

# Validate configurations
valid_configs = validate_and_filter_configs(
    all_configs=metrics,
    workload_filter={'bert', 'resnet50'}
)

# Run simulation
ret = run_polaris_simulation(
    ttsim_wlspec=workload_specs,
    uniq_devs={'n300'},
    opath=Path('output'),
    args=args,
    dry_run=False
)
```

## Usage

### Basic Workflow

```python
from pathlib import Path
from ttsi_corr import (
    load_metrics_from_sources,
    read_metadata,
    process_workload_configs,
    validate_and_filter_configs,
    run_polaris_simulation,
    compare_scores,
    calculate_and_save_geomean,
    write_csv,
    add_scurve_chart
)

# 1. Load data
data_dir = Path('data/metal/inf/15oct25')
metadata = read_metadata(data_dir)
metrics = load_metrics_from_sources(data_dir, metadata['data_source'])

# 2. Validate and filter
valid_configs = validate_and_filter_configs(metrics, workload_filter=None)

# 3. Process workloads
ttsim_wlspec, ref_scores, devices = process_workload_configs(
    valid_configs, workloads_file, None, 'fp16', device_table, 'corr'
)

# 4. Run simulation
ret = run_polaris_simulation(ttsim_wlspec, devices, output_dir, args, False)

# 5. Generate correlation reports
actual_scores = read_scores(output_dir / 'study-summary.json', 'fp16')
comparison = compare_scores(ref_scores, actual_scores)
calculate_and_save_geomean(comparison, output_dir / 'geomean.json')
write_csv(comparison, output_dir / 'correlation_result.csv')
```

### Importing at Package Level

```python
# Import entire submodules
from ttsi_corr import data_loader, correlation, excel_writer

# Use functions from submodules
metrics = data_loader.load_metrics_from_sources(data_dir, 'html')
comparison = correlation.compare_scores(ref_scores, actual_scores)
excel_writer.write_csv(comparison, output_path)
```

### Importing Specific Functions

```python
# Import frequently-used functions directly
from ttsi_corr import (
    load_metrics_from_sources,
    compare_scores,
    write_csv,
    add_scurve_chart
)

metrics = load_metrics_from_sources(data_dir, 'html')
comparison = compare_scores(ref_scores, actual_scores)
write_csv(comparison, 'output.csv')
```

## Design Principles

### Modularity
Each module has a single, well-defined responsibility:
- **data_loader**: Loading and validating data
- **workload_processor**: Workload configuration
- **correlation**: Score comparison and statistics
- **excel_writer**: Report generation
- **chart_builder**: Visualization
- **simulator**: Simulation orchestration

### Reusability
All functions are designed to be:
- Standalone and composable
- Usable across multiple tools
- Well-documented with examples
- Type-annotated for IDE support

### Testability
- Functions are pure where possible
- Side effects are clearly documented
- Easy to mock external dependencies
- Comprehensive docstrings

### Maintainability
- Clear separation of concerns
- Consistent coding style
- Comprehensive documentation
- Type hints throughout

## Testing

Tests are located in `tests/test_ttsi_corr/`:
- `test_excel_writer.py` - Excel formatting and CSV writing tests
- `test_chart_builder.py` - Chart generation tests
- `test_data_loader.py` - Data loading tests (to be added)

Integration tests are in `tests/test_tools/test_run_ttsi_corr.py`.

## Migration Status

**Current Status:** ✅ Phase 8 Complete (78% → 89%)

**Completed:**
- ✅ Phase 1: Package Structure
- ✅ Phase 2: Excel Writer
- ✅ Phase 3: Chart Builder
- ✅ Phase 4: Data Loader
- ✅ Phase 5: Workload Processor
- ✅ Phase 6: Correlation Logic
- ✅ Phase 7: Simulator
- ✅ Phase 8: Main Entry Simplification

**Remaining:**
- 🔴 Phase 9: Documentation (in progress)

**Code Reduction:** 1407 lines → 573 lines (59.3% reduction)

## Contributing

When adding new functionality to this package:

1. **Follow the module structure** - Place functionality in the appropriate module
2. **Write comprehensive docstrings** - Include Args, Returns, Examples, and Notes
3. **Add type annotations** - Use modern Python typing (3.10+)
4. **Write tests** - Add unit tests for new functions
5. **Update this README** - Document new public API functions

## See Also

- `tools/run_ttsi_corr.py` - Main correlation script (uses this package)
- `tools/parse_ttsi_perf_results.py` - Data parsing tool
- `tests/test_ttsi_corr/` - Unit tests for this package
- `tests/test_tools/test_run_metal_correlation.py` - Integration tests
