# colorpicker.py - Threshold-Based Color Selection

## Overview

`colorpicker.py` is a Python script that selects colors based on numeric values and configurable thresholds, conclusion status, or exit codes. It's designed to simplify color logic in shell scripts and CI/CD workflows, eliminating the complexity of nested conditional expressions and inline quoting.

## Why Use This Script?

### Problem: Complex Inline Color Logic

Traditional approaches to threshold-based color selection involve complicated nested expressions with error-prone quoting:

```bash
# ❌ Complex and hard to read
COVERAGE_COLOR=$(expr $COVERAGE_PERCENTAGE "<" 50 > /dev/null && echo red || (expr $COVERAGE_PERCENTAGE "<" 85 > /dev/null && echo yellow) || echo brightgreen)

# ❌ Difficult to maintain and debug
TEST_COLOR=$([ $PASSED_TESTS -lt $TOTAL_TESTS ] && echo 'red' || echo 'brightgreen')

# ❌ Nested quoting nightmare
COLOR=$(expr $VALUE "<" $THRESHOLD1 > /dev/null && echo 'red' || expr $VALUE "<" $THRESHOLD2 > /dev/null && echo 'yellow' || echo 'green')
```

### Solution: Clean Threshold Configuration

`colorpicker.py` eliminates nested expression complexity with a clear, declarative syntax:

```bash
# ✅ Clean and readable - threshold-based
python3 tools/ci/colorpicker.py --value $COVERAGE_PERCENTAGE --highcolor brightgreen 50 red 85 yellow

# ✅ Simple conclusion-based coloring
python3 tools/ci/colorpicker.py --conclusion success

# ✅ Exit code-based coloring
python3 tools/ci/colorpicker.py --exitcode $?
```

## Usage

### Basic Syntax
```bash
# Threshold-based mode
python3 tools/ci/colorpicker.py --value NUMBER --highcolor COLOR threshold1 color1 threshold2 color2 [...]

# Conclusion-based mode
python3 tools/ci/colorpicker.py --conclusion {success,failure,cancelled,skipped}

# Exit code-based mode
python3 tools/ci/colorpicker.py --exitcode NUMBER
```

### Arguments

**Mutually Exclusive Modes** (exactly one required):
- `--value NUMBER`: Numeric value to evaluate against thresholds (supports decimals)
- `--conclusion {success,failure,cancelled,skipped}`: Conclusion status
- `--exitcode NUMBER`: Exit code value

**For --value mode only**:
- `--highcolor COLOR`: Color to return if value exceeds all thresholds (required)
- `threshold color` pairs: Even number of arguments defining thresholds and their colors (required)

**For --conclusion mode**:
- `success` → `brightgreen`
- `failure`, `cancelled`, `skipped` → `red`

**For --exitcode mode**:
- `0` → `brightgreen` 
- Any other value → `red`

### Logic
- **Value mode**: Returns the color corresponding to the **first threshold** that the value is **less than**. If the value exceeds all thresholds, returns the `--highcolor`
- **Conclusion mode**: Simple success/failure color mapping
- **Exit code mode**: Simple 0/non-zero color mapping

## Examples

### Basic Color Selection

#### Threshold-based Mode
```bash
# Value 75 with thresholds at 50 and 85
python3 tools/ci/colorpicker.py --value 75 --highcolor green 50 red 85 yellow
# Output: yellow (since 75 < 85)

# Value 30 with same thresholds  
python3 tools/ci/colorpicker.py --value 30 --highcolor green 50 red 85 yellow
# Output: red (since 30 < 50)

# Value 90 with same thresholds
python3 tools/ci/colorpicker.py --value 90 --highcolor green 50 red 85 yellow  
# Output: green (since 90 exceeds all thresholds)
```

#### Conclusion-based Mode
```bash
# Success conclusion
python3 tools/ci/colorpicker.py --conclusion success
# Output: brightgreen

# Failure conclusion
python3 tools/ci/colorpicker.py --conclusion failure
# Output: red

# Other conclusions
python3 tools/ci/colorpicker.py --conclusion cancelled
# Output: red
```

#### Exit Code-based Mode
```bash
# Success exit code
python3 tools/ci/colorpicker.py --exitcode 0
# Output: brightgreen

# Failure exit code
python3 tools/ci/colorpicker.py --exitcode 1
# Output: red

# Any non-zero exit code
python3 tools/ci/colorpicker.py --exitcode 42
# Output: red
```

### Coverage Badge Colors
```bash
# Coverage percentage: 0-50% = red, 50-85% = yellow, 85%+ = green
COVERAGE_COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE_PERCENTAGE --highcolor brightgreen 50 red 85 yellow)
echo "Coverage badge color: $COVERAGE_COLOR"

# Or simple success/failure based on coverage threshold
if [ "$COVERAGE_PERCENTAGE" -gt 80 ]; then
    COVERAGE_COLOR=$(python3 tools/ci/colorpicker.py --conclusion success)
else
    COVERAGE_COLOR=$(python3 tools/ci/colorpicker.py --conclusion failure)
fi
```

### Test Results Colors
```bash
# Test pass rate: 0-75% = red, 75-95% = yellow, 95%+ = green
PASS_RATE=$(expr $PASSED_TESTS \* 100 / $TOTAL_TESTS)
TEST_COLOR=$(python3 tools/ci/colorpicker.py --value $PASS_RATE --highcolor brightgreen 75 red 95 yellow)
echo "Test badge color: $TEST_COLOR"

# Simple pass/fail based on test results
if [ "$PASSED_TESTS" -eq "$TOTAL_TESTS" ]; then
    TEST_COLOR=$(python3 tools/ci/colorpicker.py --conclusion success)
else
    TEST_COLOR=$(python3 tools/ci/colorpicker.py --conclusion failure)
fi

# Exit code-based coloring from test command
./run_tests.sh
TEST_COLOR=$(python3 tools/ci/colorpicker.py --exitcode $?)
```

### Performance Metrics
```bash
# Response time thresholds: 0-100ms = green, 100-500ms = yellow, 500ms+ = red
RESPONSE_TIME=250
PERF_COLOR=$(python3 tools/ci/colorpicker.py --value $RESPONSE_TIME --highcolor red 100 green 500 yellow)
echo "Performance status: $PERF_COLOR"
```

### Decimal Values
```bash
# CPU usage with decimal precision
CPU_USAGE=67.5
CPU_COLOR=$(python3 tools/ci/colorpicker.py --value $CPU_USAGE --highcolor red 50.0 green 80.0 yellow)
echo "CPU status color: $CPU_COLOR"
```

## Features

### ✅ Multiple Operation Modes
- **Threshold-based**: Complex multi-threshold color selection
- **Conclusion-based**: Simple success/failure status mapping
- **Exit code-based**: Direct exit code to color mapping
- Mutually exclusive modes prevent argument confusion

### ✅ No Complex Nested Expressions
- Eliminates complicated `expr` chains and nested conditionals
- Simple threshold/color pair syntax for complex cases
- One-line solutions for simple cases
- Easy to read and understand logic

### ✅ Flexible Threshold Configuration
- Support for multiple thresholds (value mode)
- Decimal value support
- Configurable colors for each threshold
- Simple binary success/failure options

### ✅ Robust Input Validation
- Validates numeric values (integers and decimals)
- Ensures even number of threshold/color pairs
- Validates conclusion choices
- Clear error messages with problematic values
- Prevents incompatible argument combinations

### ✅ Clean Error Handling
- Error messages go to stderr
- Usage information goes to stdout
- Proper exit codes for automation
- Python-based robust argument parsing

### ✅ Pipeline Friendly
- Clean stdout output (just the color)
- Errors can be suppressed or redirected
- Works well in command substitution
- No external dependencies (bc, etc.)

## Error Handling

### Common Errors and Solutions

#### Multiple Mutually Exclusive Arguments
```bash
$ python3 tools/ci/colorpicker.py --value 75 --conclusion success
error: only one of --value, --conclusion, or --exitcode can be provided, got: --value, --conclusion
```

#### No Main Argument Provided
```bash
$ python3 tools/ci/colorpicker.py --highcolor green
error: exactly one of --value, --conclusion, or --exitcode must be provided
```

#### Invalid Conclusion Value
```bash
$ python3 tools/ci/colorpicker.py --conclusion invalid
error: argument --conclusion: invalid choice: 'invalid' (choose from 'success', 'failure', 'cancelled', 'skipped')
```

#### Incompatible Arguments with Conclusion/Exit Code
```bash
$ python3 tools/ci/colorpicker.py --conclusion success --highcolor green
error: --highcolor cannot be used with --conclusion or --exitcode

$ python3 tools/ci/colorpicker.py --exitcode 0 50 red
error: threshold/color pairs cannot be used with --conclusion or --exitcode
```

#### Value Mode Errors
```bash
# Missing required arguments for value mode
$ python3 tools/ci/colorpicker.py --value 75
error: --highcolor is required when using --value

# Invalid threshold
$ python3 tools/ci/colorpicker.py --value 75 --highcolor green xyz red 85 yellow
error: threshold 'xyz' must be a number

# Odd number of threshold/color pairs
$ python3 tools/ci/colorpicker.py --value 75 --highcolor green 50 red 85
error: threshold/color pairs must be even in number
```

## Integration Examples

### GitHub Actions Workflow
```yaml
- name: Generate Badge Colors
  run: |
    COVERAGE_PERCENTAGE=$(jq .totals.percent_covered coverage.json)
    PASS_PERCENTAGE=$(expr $PASSED_TESTS \* 100 / $TOTAL_TESTS)
    
    # Clean, readable threshold-based color selection
    COVERAGE_COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE_PERCENTAGE --highcolor brightgreen 50 red 85 yellow)
    TESTS_COLOR=$(python3 tools/ci/colorpicker.py --value $PASS_PERCENTAGE --highcolor brightgreen 75 red 95 yellow)
    
    # Simple success/failure colors
    ./run_tests.sh
    BUILD_COLOR=$(python3 tools/ci/colorpicker.py --exitcode $?)
    
    # Conclusion-based coloring
    if [ "$COVERAGE_PERCENTAGE" -gt 80 ] && [ "$PASS_PERCENTAGE" -gt 95 ]; then
        OVERALL_COLOR=$(python3 tools/ci/colorpicker.py --conclusion success)
    else
        OVERALL_COLOR=$(python3 tools/ci/colorpicker.py --conclusion failure)
    fi
    
    # Generate badge JSON
    python3 tools/ci/makegist.py --gist-id $GIST_ID --gist-filename coverage.json label="Coverage" message="$COVERAGE_PERCENTAGE%" color="$COVERAGE_COLOR"
    python3 tools/ci/makegist.py --gist-id $GIST_ID --gist-filename tests.json label="Tests" message="$PASSED_TESTS/$TOTAL_TESTS" color="$TESTS_COLOR"
```

### Shell Script Monitoring
```bash
#!/bin/bash
# System monitoring with color-coded alerts

# Memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
MEMORY_COLOR=$(python3 tools/ci/colorpicker.py --value $MEMORY_USAGE --highcolor red 70 green 90 yellow)

# Disk usage
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
DISK_COLOR=$(python3 tools/ci/colorpicker.py --value $DISK_USAGE --highcolor red 80 green 95 yellow)

echo "Memory: $MEMORY_USAGE% ($MEMORY_COLOR)"
echo "Disk: $DISK_USAGE% ($DISK_COLOR)"
```

### Build Pipeline Status
```bash
#!/bin/bash
# Build quality gates

BUILD_SCORE=85
COVERAGE=78
TEST_PASS_RATE=92

# Quality gate colors
SCORE_COLOR=$(python3 tools/ci/colorpicker.py --value $BUILD_SCORE --highcolor green 60 red 80 yellow)
COV_COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor green 70 red 85 yellow)
TEST_COLOR=$(python3 tools/ci/colorpicker.py --value $TEST_PASS_RATE --highcolor green 90 red 98 yellow)

echo "Build Score: $BUILD_SCORE ($SCORE_COLOR)"
echo "Coverage: $COVERAGE% ($COV_COLOR)"
echo "Tests: $TEST_PASS_RATE% ($TEST_COLOR)"
```

### Docker Health Checks
```bash
#!/bin/bash
# Container health monitoring

CPU_USAGE=$(docker stats --no-stream --format "{{.CPUPerc}}" myapp | sed 's/%//')
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" myapp | sed 's/%//')

CPU_STATUS=$(python3 tools/ci/colorpicker.py --value $CPU_USAGE --highcolor red 50 green 80 yellow)
MEM_STATUS=$(python3 tools/ci/colorpicker.py --value $MEMORY_USAGE --highcolor red 60 green 85 yellow)

echo "Container Health:"
echo "  CPU: $CPU_USAGE% ($CPU_STATUS)"
echo "  Memory: $MEMORY_USAGE% ($MEM_STATUS)"
```

## Best Practices

### 1. Order Thresholds from Low to High
```bash
# ✅ Good - ascending order
python3 tools/ci/colorpicker.py --value $VALUE --highcolor green 25 red 50 yellow 75 orange

# ❌ Confusing - mixed order
python3 tools/ci/colorpicker.py --value $VALUE --highcolor green 75 orange 25 red 50 yellow
```

### 2. Use Descriptive Color Names
```bash
# ✅ Good - clear intent
python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor excellent 70 poor 85 good

# ✅ Also good - standard colors
python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor brightgreen 50 red 85 yellow
```

### 3. Handle Errors in Scripts
```bash
if ! COLOR=$(python3 tools/ci/colorpicker.py --value $VALUE --highcolor green 50 red 2>/dev/null); then
    echo "Failed to determine color for value: $VALUE" >&2
    COLOR="gray"  # fallback
fi
```

### 4. Use Variables for Maintainability
```bash
# ✅ Good - configurable thresholds
COVERAGE_LOW=50
COVERAGE_HIGH=85
COVERAGE_COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor brightgreen $COVERAGE_LOW red $COVERAGE_HIGH yellow)
```

## Comparison with Traditional Approaches

| Aspect | `colorpicker.py` | Nested `expr` | Inline `if/then` |
|--------|------------------|---------------|------------------|
| **Readability** | High | Low | Medium |
| **Maintainability** | High | Low | Medium |
| **Error-prone** | Low | High | Medium |
| **Quoting complexity** | None | High | Medium |
| **Threshold flexibility** | High | Low | Medium |
| **Debugging ease** | High | Low | Medium |

### Before and After Comparison

#### Before (Complex nested expressions):
```bash
# ❌ Hard to read and maintain
COLOR=$(expr $COVERAGE "<" 50 > /dev/null && echo red || (expr $COVERAGE "<" 85 > /dev/null && echo yellow) || echo brightgreen)
```

#### After (Clean colorpicker.py):
```bash
# ✅ Clear and maintainable
COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor brightgreen 50 red 85 yellow)
```

## Advanced Usage

### Multiple Threshold Scenarios
```bash
# Fine-grained performance classification
RESPONSE_TIME=150
PERF_COLOR=$(python3 tools/ci/colorpicker.py --value $RESPONSE_TIME \
    --highcolor critical \
    50 excellent \
    100 good \
    200 acceptable \
    500 poor)
```

### Dynamic Threshold Configuration
```bash
# Load thresholds from environment or config
LOW_THRESHOLD=${COVERAGE_LOW:-50}
HIGH_THRESHOLD=${COVERAGE_HIGH:-85}

COLOR=$(python3 tools/ci/colorpicker.py --value $COVERAGE --highcolor green $LOW_THRESHOLD red $HIGH_THRESHOLD yellow)
```

## License

This script is part of the Tenstorrent AI ULC project and is licensed under Apache-2.0.

---

**See Also:**
- `makegist.py` - Companion script for JSON generation and GitHub Gist creation
- Project workflows in `.github/workflows/` for usage examples
- Badge generation examples in CI/CD pipelines

