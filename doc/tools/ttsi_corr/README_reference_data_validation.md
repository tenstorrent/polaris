# Reference Data Validation and Management Tools

## Overview

This document describes the tools and tests for validating, comparing, and managing TT-Metal reference performance data in Polaris.

## Components

### 1. Data Validation Tests (`tests/test_tools/test_reference_data_validation.py`)

Comprehensive test suite that validates the structure, integrity, and completeness of reference data.

**Test Classes:**

#### `TestReferenceDataStructure`
Validates the basic structure of reference data:
- Tag directories exist for all valid tags
- Metadata files are present and properly formatted
- YAML files are valid and parseable
- Metadata tag matches directory name

#### `TestBenchmarkEntries`
Validates individual benchmark entries:
- All required fields are present (`batch`, `gpu`, `hardware`, `id`, `model`, `precision`, `input_dtype`)
- At least one performance metric is non-null
- GPU field is `'Tensix'`
- Hardware field follows expected format
- Batch size is a positive integer

#### `TestReleaseFieldValidation`
Validates release field with warnings:
- Warns (non-failing) if entries have `null` release field
- Recommends adding release versions for reproducibility

#### `TestTagComparison`
Compares different tags:
- Validates expected differences between tags (e.g., `03nov25` vs `15oct25`)
- Ensures default tag is first in `TTSI_REF_VALID_TAGS`
- Documents changes (added/removed benchmarks, new hardware platforms)

#### `TestTagNamingStandard`
Enforces tag naming conventions:
- Format: `DDmmmYY` (e.g., `03nov25`, `3nov25`)
- Valid month abbreviations: `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec`
- Day range: 1-31

#### `TestDataProvenance`
Validates data provenance and metadata:
- Metadata has `source_url` field
- Metadata has `parsed_date` in ISO 8601 format
- Recommends including `source_commit` field for traceability

**Running Tests:**

```bash
# Run all reference data validation tests
pytest tests/test_tools/test_reference_data_validation.py -v

# Run specific test class
pytest tests/test_tools/test_reference_data_validation.py::TestBenchmarkEntries -v

# Run specific test
pytest tests/test_tools/test_reference_data_validation.py::TestReleaseFieldValidation::test_release_field_populated -v
```

---

### 2. Data Comparison Tool (`tools/ttsi_corr/compare_reference_data.py`)

Command-line tool to compare reference data between different tags.

**Features:**
- Side-by-side comparison of two tags
- Metadata comparison (source URL, parse date, commit hash)
- Summary statistics (total benchmarks, changes)
- Hardware platform changes (added/removed)
- Model changes (added/removed)
- Detailed entry-level differences (with `--verbose`)
- File-by-file breakdown
- Colorized terminal output

**Usage:**

```bash
# Compare default tag with previous tag
python tools/ttsi_corr/compare_reference_data.py

# Compare specific tags
python tools/ttsi_corr/compare_reference_data.py --old 15oct25 --new 03nov25

# Detailed comparison with all changes
python tools/ttsi_corr/compare_reference_data.py --old 15oct25 --new 03nov25 --verbose

# Custom data directory
python tools/ttsi_corr/compare_reference_data.py --old 15oct25 --new 03nov25 --data-dir /path/to/data/metal/inf
```

**Example Output:**

```
================================================================================
Reference Data Comparison: 15oct25 → 03nov25
================================================================================

Metadata:
  Old tag: 15oct25
    Parsed: 2025-10-23T12:29:58.742316
    Source: https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md

  New tag: 03nov25
    Parsed: 2025-11-25T13:52:42.092475
    Source: https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md
    Commit: abc123def456...

Summary:
  Total benchmarks: 73 → 79
  ✓ Added 6 new benchmarks

Hardware Platforms:
  ✓ Added: p150

File Breakdown:
  tensix_md_perf_metrics_nlp.yaml: 3 → 4 (+1)
  tensix_md_perf_metrics_vision.yaml: 9 → 11 (+2)
  ...
```

---

### 3. Metadata Provenance Updater (`tools/ttsi_corr/update_metadata_provenance.py`)

Tool to add or update git commit hash in metadata files for data provenance.

**Features:**
- Automatically fetches latest commit hash from GitHub
- Updates `source_commit` field in `_metadata.yaml`
- Supports manual commit hash specification
- Dry-run mode to preview changes
- Batch update all tags

**Usage:**

```bash
# Update specific tag (auto-fetch commit hash)
python tools/ttsi_corr/update_metadata_provenance.py --tag 03nov25

# Update with specific commit hash
python tools/ttsi_corr/update_metadata_provenance.py --tag 03nov25 --commit abc123def456

# Dry run to see changes without applying
python tools/ttsi_corr/update_metadata_provenance.py --tag 03nov25 --dry-run

# Update all tags
python tools/ttsi_corr/update_metadata_provenance.py --all
```

**Example Output:**

```
Fetching latest commit for tenstorrent/tt-metal/main...
Found commit: abc123de
✓ Updated data/metal/inf/03nov25/_metadata.yaml
```

---

### 4. Validation Utilities (`tools/ttsi_corr/ttsi_corr_utils.py`)

Shared validation functions used across tools and tests.

**Functions:**

#### `validate_tag_format(tag: str, strict: bool = False) -> tuple[bool, Optional[str]]`

Validates tag naming format.

**Parameters:**
- `tag`: Tag string to validate
- `strict`: If `True`, requires 2-digit day; if `False`, allows 1-digit day

**Returns:**
- Tuple of `(is_valid, error_message)`

**Example:**
```python
from tools.ttsi_corr.ttsi_corr_utils import validate_tag_format

is_valid, error = validate_tag_format('03nov25')
if not is_valid:
    print(f'Invalid tag: {error}')

# Valid tags: 03nov25, 3nov25, 15oct25
# Invalid tags: nov25, 3-nov-25, 32dec25, 3xyz25
```

#### `validate_release_field(entries: List[dict], warn: bool = True) -> List[dict]`

Validates and warns about null release fields in benchmark entries.

**Parameters:**
- `entries`: List of benchmark entry dictionaries
- `warn`: If `True`, prints warnings for null release fields

**Returns:**
- List of entries with missing release field

**Example:**
```python
from tools.ttsi_corr.ttsi_corr_utils import validate_release_field

entries = [
    {'model': 'BERT', 'release': 'v0.59.0', ...},
    {'model': 'ResNet', 'release': None, ...},
]

missing = validate_release_field(entries, warn=True)
# Prints: ⚠️  WARNING: 1 benchmark entries have null release field:
#    - ResNet on ...
```

---

### 5. Enhanced Parse Tool

The `tools/parse_ttsi_perf_results.py` tool now includes:

#### Automatic Commit Hash Capture
When parsing from GitHub URLs, automatically fetches and stores the git commit hash in metadata:

```yaml
tag: 03nov25
data_source: md
input_url: https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md
parsed_date: '2025-11-25T13:52:42.092475'
use_cache: false
source_commit: abc123def456789...  # ← Automatically added
```

#### Tag Format Validation
Validates tag format before processing:

```bash
# Valid tag
python tools/parse_ttsi_perf_results.py --tag 03nov25 --input ...

# Invalid tag - will error with helpful message
python tools/parse_ttsi_perf_results.py --tag nov25 --input ...
# Error: Invalid tag format: Tag "nov25" does not match format DDmmmYY
```

#### Release Field Warnings
Warns if parsed metrics have null release fields:

```
⚠️  WARNING: 3 benchmark entries have null release field:
   - ViT-base on p150 (Blackhole)
   - MobileNet-v2 on n150 (Wormhole)
   - ...
   Recommendation: Add release version for reproducibility
```

---

## Tag Naming Standard

**Format:** `DDmmmYY`
- `DD`: Day (1-2 digits, e.g., `3` or `03`)
- `mmm`: Month abbreviation (lowercase: `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec`)
- `YY`: Year (2 digits, e.g., `25` for 2025)

**Examples:**
- ✅ `03nov25` - November 3, 2025 (preferred for consistency)
- ✅ `3nov25` - November 3, 2025 (also valid)
- ✅ `15oct25` - October 15, 2025
- ✅ `01jan26` - January 1, 2026
- ❌ `nov25` - Missing day
- ❌ `3-nov-25` - Invalid separators
- ❌ `3NOV25` - Uppercase month
- ❌ `32dec25` - Invalid day (> 31)
- ❌ `3xyz25` - Invalid month

---

## Data Provenance

To ensure reproducibility and traceability, reference data should include:

### Required Fields (in `_metadata.yaml`):
- `tag`: Tag identifier
- `data_source`: Format (`md`, `html`)
- `input_url`: Source URL
- `parsed_date`: ISO 8601 timestamp

### Recommended Fields:
- `source_commit`: Git commit hash of source repository (enables exact reproduction)

### Example Complete Metadata:

```yaml
tag: 03nov25
data_source: md
input_url: https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md
parsed_date: '2025-11-25T13:52:42.092475'
use_cache: false
source_commit: abc123def456789abcdef0123456789abcdef012
```

---

## Workflow Examples

### Updating Reference Data with Full Validation

```bash
# 1. Parse new metrics with automatic commit hash capture
python tools/parse_ttsi_perf_results.py \
    --tag 10dec25 \
    --input https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/README.md \
    --output-dir data/metal/inf

# 2. Validate the new data
pytest tests/test_tools/test_reference_data_validation.py::TestBenchmarkEntries -v

# 3. Compare with previous tag
python tools/ttsi_corr/compare_reference_data.py --old 03nov25 --new 10dec25 --verbose

# 4. Update code to use new tag as default
# Edit tools/ttsi_corr/ttsi_corr_utils.py:
#   TTSI_REF_VALID_TAGS = ['10dec25', '03nov25', '15oct25']
#   TTSI_REF_DEFAULT_TAG = TTSI_REF_VALID_TAGS[0]

# 5. Run full validation suite
pytest tests/test_tools/test_reference_data_validation.py -v
```

### Adding Commit Hash to Existing Data

```bash
# Update specific tag
python tools/ttsi_corr/update_metadata_provenance.py --tag 03nov25

# Update all tags
python tools/ttsi_corr/update_metadata_provenance.py --all

# Dry run first to verify
python tools/ttsi_corr/update_metadata_provenance.py --tag 03nov25 --dry-run
```

### Comparing Two Historical Tags

```bash
# Basic comparison
python tools/ttsi_corr/compare_reference_data.py --old 15oct25 --new 03nov25

# Detailed comparison showing all changes
python tools/ttsi_corr/compare_reference_data.py --old 15oct25 --new 03nov25 --verbose
```

---

## Integration with CI/CD

These tools can be integrated into continuous integration workflows:

```yaml
# Example GitHub Actions workflow
name: Validate Reference Data

on: [pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Validate reference data
        run: |
          pytest tests/test_tools/test_reference_data_validation.py -v
      - name: Compare with baseline
        run: |
          python tools/ttsi_corr/compare_reference_data.py --verbose
```

---

## Best Practices

1. **Always validate new data**: Run the validation test suite after adding or updating reference data
2. **Document changes**: Use the comparison tool to generate a summary of changes
3. **Include commit hash**: Ensure `source_commit` is present in metadata for traceability
4. **Follow naming convention**: Use consistent tag naming (DDmmmYY format)
5. **Add release versions**: Include `release` field in benchmark entries when available
6. **Keep old tags**: Preserve previous tags for reproducibility and regression testing
7. **Update default carefully**: When changing `TTSI_REF_DEFAULT_TAG`, document the impact on correlation results

---

## Troubleshooting

### Tag Validation Fails

```
Error: Invalid tag format: Tag "nov25" does not match format DDmmmYY
```

**Solution:** Use proper format `DDmmmYY`, e.g., `03nov25` or `3nov25`

### Missing Source Commit Warning

```
⚠️  No source_commit in metadata (recommended for traceability)
```

**Solution:** Run `python tools/ttsi_corr/update_metadata_provenance.py --tag <TAG>` to add commit hash

### Release Field Warnings

```
⚠️  WARNING: 5 benchmark entries have null release field
```

**Solution:** This is informational. Update upstream data source to include release versions when possible.

### Comparison Shows Unexpected Changes

```
✗ Removed 3 benchmarks
```

**Solution:** Review the detailed comparison with `--verbose` to understand what changed. This may be intentional if benchmarks were deprecated upstream.

---

## Summary

The reference data validation and management tools provide:

- ✅ **Comprehensive validation** of data structure and integrity
- ✅ **Tag naming standards** enforcement
- ✅ **Data provenance tracking** with git commit hashes
- ✅ **Easy comparison** between data versions
- ✅ **Quality warnings** for missing metadata
- ✅ **Automated testing** integration
- ✅ **Clear documentation** and examples

These tools help maintain high-quality, traceable, and reproducible reference data for Polaris performance correlation.

