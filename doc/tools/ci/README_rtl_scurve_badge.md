# RTL S-Curve Badge Generator

**Script:** `tools/ci/rtl_scurve_badge.py`

## Overview

The RTL S-Curve Badge Generator is a specialized Python script that processes RTL test results in s-curve format and generates dynamic badges, summary statistics, and detailed CSV reports. This script is designed specifically for parsing test output that contains s-curve performance data between `+ Test class s-curve:` and `+ Saving` markers.

## Features

- **S-Curve Format Parsing**: Specialized parser for pipe-delimited test result lines
- **Dynamic Badge Generation**: Creates GitHub Gist badges for status and performance metrics
- **Statistical Analysis**: Calculates pass rates and geometric mean of performance ratios
- **Multiple Output Formats**: JSON summaries, CSV details, and dynamic badges
- **Failure Mode Support**: Generates failure badges when previous commands fail
- **Dry Run Mode**: Preview operations without executing them

## Installation & Setup

### Prerequisites

- Python 3.7+
- Access to `bigpoldev` conda environment
- `GIST_TOKEN` environment variable (GitHub Personal Access Token with gist permissions)
- `makegist.py` script (located in same directory)

### Environment Setup

```bash
# Activate the conda environment
conda activate bigpoldev

# Set GitHub token for gist operations
export GIST_TOKEN="your_github_token_here"
```

## Usage

### Basic Usage

```bash
python tools/ci/rtl_scurve_badge.py \
    --repo REPOSITORY_NAME \
    --gistid GIST_ID \
    --input INPUT_FILE \
    [--is-main-branch] \
    [--dryrun]
```

### Failure Mode

```bash
python tools/ci/rtl_scurve_badge.py \
    --runexitcode EXIT_CODE \
    --repo REPOSITORY_NAME \
    --gistid GIST_ID \
    --input INPUT_FILE \
    [--is-main-branch] \
    [--dryrun]
```

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--repo` | Yes | Repository name (used in badge filenames) |
| `--gistid` | Yes | GitHub Gist ID for storing badges |
| `--input` | Yes | Path to input file containing s-curve test results |
| `--runexitcode` | No | Exit code of previous command (triggers failure mode if non-zero) |
| `--is-main-branch` | No | Whether this is the main branch (affects filename prefix) |
| `--dryrun` | No | Show gist commands without executing them |

### Branch-Aware Filename Behavior

The script supports branch-aware filename generation for CI/CD environments:

- **Main Branch**: Standard filenames (e.g., `polaris_rtl_scurve_status.json`)
- **Non-Main Branch**: DELETEME_ prefix (e.g., `DELETEME_polaris_rtl_scurve_status.json`)

This allows for:
- **Testing**: Retry logic can be tested on all branches
- **Cleanup**: Temporary files can be easily identified and removed
- **Separation**: Clear distinction between permanent and temporary badges

## Input File Format

### Configuration Section

The script expects a configuration section at the beginning of the file with key-value pairs:

```
debug = 15
model_odir_prefix = __llk_out/llk_jul27
rtl_tags = ['jul27']
num_processes = 4
...
```

### S-Curve Section

The s-curve section must be bounded by these markers:

```
  + Test class s-curve:
    [ 1/96] test_name | TEST_CLASS | RTL: PASS | Cycles: 6750 | Model: PASS | Cycles: 9517 | Model/RTL: 1.41
    [ 2/96] test_name | TEST_CLASS | RTL: FAIL | Cycles:    - | Model: PASS | Cycles:  459 | Model/RTL:    -
    ...
  + Saving S-Curve plot to output.png
```

### Line Format

Each test result line follows this pattern:

```
    [serial/total] test_name | test_class | RTL: status | Cycles: value | Model: status | Cycles: value | Model/RTL: ratio
```

**Field Details:**
- `serial/total`: Sequential test number and total count (e.g., `[1/96]`)
- `test_name`: Test identifier string
- `test_class`: Test category (MATMUL, ELTW, SFPU, PCK, DATACOPY, UPK, REDUCE)
- `RTL/Model status`: Either `PASS` or `FAIL`
- `Cycles`: Integer cycle count or `-` for failed tests
- `Model/RTL ratio`: Float ratio or `-` for failed tests

## Output Files

### Summary JSON (`rtl_scurve_summary.json`)

Contains aggregated statistics:

```json
{
  "total_tests": 96,
  "model_passed_tests": 95,
  "rtl_passed_tests": 93,
  "model_2_rtl_ratio_geomean": 0.4575,
  "rtl_status_passed": 93,
  "rtl_status_failed": 3,
  "model_status_passed": 95,
  "model_status_failed": 1
}
```

### Details CSV (`rtl_scurve_details.csv`)

Contains per-test results with headers:
- `serial_number`, `total_tests`, `test_name`, `test_class`
- `rtl_status`, `rtl_cycles`, `model_status`, `model_cycles`, `model_rtl_ratio`

### Dynamic Badges

#### Status Badge (`REPO_rtl_scurve_status.json`)
- **Label**: "RTL Status"
- **Message**: "{model_passed_tests}/{total_tests}" (e.g., "95/96")
- **Colors**:
  - 🟢 **Bright Green**: 100% model tests passed
  - 🟠 **Orange**: ≥85% model tests passed
  - 🔴 **Red**: <85% model tests passed

#### Ratio Badge (`REPO_rtl_scurve_ratio_geomean.json`)
- **Label**: "RTL Ratio Geomean"
- **Message**: Geometric mean to 2 decimal places (e.g., "0.46")
- **Colors**:
  - 🟢 **Green**: Ratio within ±10% of 1.0 (0.90-1.10)
  - 🔴 **Red**: Ratio outside ±10% tolerance

## Statistics Calculation

### Pass Rates
- **Total Tests**: Count of all parsed test results
- **Model Passed Tests**: Count where `model_status == "PASS"` (regardless of RTL status)
- **RTL Passed Tests**: Count where `rtl_status == "PASS"` (regardless of model status)

### Performance Metrics
- **Geometric Mean**: Calculated from `model_rtl_ratio` values where both RTL and model passed
- **Status Counts**: Separate counts for RTL and model pass/fail regardless of the other status

## Examples

### Successful Processing

```bash
$ python tools/ci/rtl_scurve_badge.py \
    --repo polaris \
    --gistid abc123def456 \
    --input rtl_test_results.txt \
    --is-main-branch

Extracted 25 configuration items
Found s-curve section with 96 lines
Parsed 96 test results
Statistics: 95/96 model tests passed
Geometric mean ratio: 0.4575
Saved summary: __llk_out/rtl_scurve_summary.json
Saved CSV: __llk_out/rtl_scurve_details.csv
Created status badge: polaris_rtl_scurve_status.json
Created ratio badge: polaris_rtl_scurve_ratio_geomean.json
S-curve badge generation completed successfully!
```

### Dry Run Mode

```bash
$ python tools/ci/rtl_scurve_badge.py \
    --repo polaris \
    --gistid abc123def456 \
    --input rtl_test_results.txt \
    --is-main-branch \
    --dryrun

[DRYRUN] Would run: python3 makegist.py --gist-id abc123def456 --gist-filename polaris_rtl_scurve_status.json label=RTL Status message=95/96 color=orange
[DRYRUN] Would run: python3 makegist.py --gist-id abc123def456 --gist-filename polaris_rtl_scurve_ratio_geomean.json label=RTL Ratio Geomean message=0.46 color=red
```

### Failure Mode

```bash
$ python tools/ci/rtl_scurve_badge.py \
    --runexitcode 1 \
    --repo polaris \
    --gistid abc123def456 \
    --input dummy.txt \
    --is-main-branch \
    --dryrun

Previous command failed with exit code 1
Creating failure badges...
[DRYRUN] Would run: python3 makegist.py --gist-id abc123def456 --gist-filename polaris_rtl_scurve_status.json label=RTL Status message=Failed color=red
[DRYRUN] Would run: python3 makegist.py --gist-id abc123def456 --gist-filename polaris_rtl_scurve_ratio_geomean.json label=RTL Ratio Geomean message=Failed color=red
Failure badges created. Exiting.
```

## Error Handling

### Parsing Errors
The script provides detailed error messages for:
- Missing s-curve section markers
- Invalid line formats
- Inconsistent serial numbers
- Type conversion failures

### File Errors
- Input file not found
- Permission issues
- Invalid configuration format

### Gist Errors
- Missing `GIST_TOKEN` environment variable
- Network connectivity issues
- Invalid gist ID

## Troubleshooting

### Common Issues

**"S-curve start marker not found"**
- Ensure input file contains `+ Test class s-curve:` line
- Check file encoding and line endings

**"Invalid cycle count: value"**
- Verify cycle values are integers or `-` for failed tests
- Check for unexpected characters in cycle fields

**"Error creating badges"**
- Verify `GIST_TOKEN` is set and valid
- Check network connectivity
- Ensure gist ID exists and is accessible

**"Inconsistent total test count"**
- All test lines must have the same total count (e.g., `/96`)
- Check for malformed serial number fields

### Debug Mode

For detailed parsing information, you can modify the script to add debug prints or use the dry-run mode to see what commands would be executed.

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Generate RTL S-Curve Badges
  if: steps.run-rtl-tests.outcome == 'success' || steps.run-rtl-tests.outcome == 'failure'
  run: |
    EXIT_CODE=${{ steps.run-rtl-tests.outcome == 'success' && '0' || '1' }}
    python tools/ci/rtl_scurve_badge.py \
      --runexitcode $EXIT_CODE \
      --repo $REPO_NAME \
      --gistid $GIST_ID \
      --input rtl_test_results.txt \
      ${{ github.ref == 'refs/heads/main' && '--is-main-branch' || '' }}
```

## Related Documentation

- [Dynamic Badges Overview](README_dynamic_badges.md)
- [Large File Cache Usage](large_file_cache_usage.md)
- [Tools Overview](../README.md)

## License

This script is licensed under Apache-2.0. See the SPDX headers in the source file for details.
