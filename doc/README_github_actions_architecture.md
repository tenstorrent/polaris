# GitHub Actions Architecture: Reusable Actions and Workflows

## Overview

This document describes the project's architecture and best practices for structuring GitHub CI/CD using reusable actions and workflows. The project follows a modular approach that promotes code reuse, maintainability, and consistency across different CI/CD pipelines.

## Architecture Philosophy

### Core Principles

1. **DRY (Don't Repeat Yourself)**: Common setup tasks are abstracted into reusable actions
2. **Modularity**: Complex workflows are broken down into composable, testable units
3. **Consistency**: Standardized patterns across all workflows ensure predictable behavior
4. **Maintainability**: Centralized logic reduces duplication and simplifies updates
5. **Flexibility**: Parameterized actions allow customization while maintaining consistency

### Benefits of Reusable Actions

| Benefit | Description | Impact |
|---------|-------------|---------|
| **Code Reuse** | Common patterns abstracted into actions | Reduces duplication across workflows |
| **Centralized Updates** | Change logic once, affects all workflows | Easier maintenance and bug fixes |
| **Consistency** | Standardized setup procedures | Predictable CI/CD behavior |
| **Testing** | Actions can be tested independently | Higher reliability and confidence |
| **Documentation** | Clear interfaces and inputs | Better developer experience |

## Project Structure

### Directory Layout

```
.github/
├── actions/                    # Custom reusable actions
│   ├── generate-rtl-badges/   # RTL badge generation action
│   │   └── action.yml
│   ├── generate-status-badges/ # Status badge generation action
│   │   └── action.yml
│   ├── generate-test-badges/  # Test and coverage badge generation action
│   │   └── action.yml
│   ├── lfcdownload/           # Large File Cache download action
│   │   └── action.yml
│   ├── run-python-tests/      # Python unit and coverage tests action
│   │   └── action.yml
│   ├── run-rtl-tests/         # RTL test execution action
│   │   └── action.yml
│   ├── run-static-analysis/   # Static analysis action
│   │   └── action.yml
│   └── setup_mamba/           # Mamba environment setup action
│       └── action.yml
├── spdxchecker-config.yml     # SPDX license checker configuration
└── workflows/                 # CI/CD workflows using actions
    ├── checkin_tests.yml      # Pre-merge validation with integrated RTL testing
    └── nightly_tests.yml      # Comprehensive nightly testing
```

### Action vs Workflow Responsibility

| Component | Responsibility | Scope |
|-----------|---------------|-------|
| **Actions** | Reusable setup/utility tasks | Single responsibility, parameterized |
| **Workflows** | Complete CI/CD pipelines | Orchestrate actions and business logic |

## Reusable Actions Reference

### 1. run-rtl-tests Action

**Purpose**: Execute RTL tests with configurable parameters and handle LFC file downloads

**Location**: `.github/actions/run-rtl-tests/action.yml`

**Interface**:
```yaml
name: 'Run RTL Tests'
description: 'Execute RTL tests with configurable parameters and handle LFC file downloads'
inputs:
  tag:
    description: 'Test tag to run'
    required: true
  parallel:
    description: 'Number of parallel processes'
    required: false
    default: '4'
  lfc-files:
    description: 'Space-separated list of LFC files to download'
    required: true
  results-file:
    description: 'Path to RTL test results output file'
    required: true
outputs:
  outcome:
    description: 'Test execution outcome (success/failure)'
  exit-code:
    description: 'Test exit code for badge generation'
  results-file:
    description: 'Path to the generated RTL test results file'
```

**Usage Examples**:

#### Basic Usage
```yaml
- name: Run RTL Tests
  uses: ./.github/actions/run-rtl-tests
  with:
    tag: sep23
    lfc-files: 'ext_rtl_test_data_set_sep23.tar.gz'
    results-file: 'rtl_test_results.txt'
```

#### Custom Configuration
```yaml
- name: Run RTL Tests with Custom Settings
  uses: ./.github/actions/run-rtl-tests
  with:
    tag: sep23
    parallel: 8
    lfc-files: 'ext_rtl_test_data_set_sep23.tar.gz'
    results-file: 'custom_rtl_results.txt'
```

**Key Features**:
- **Automated LFC Downloads**: Handles test data downloads internally
- **Explicit Configuration**: Requires specification of test tag, data files, and results file path  
- **Configurable Parallelism**: Optional parallel execution parameter (defaults to 4)
- **Error Resilience**: Uses `continue-on-error` for non-blocking execution
- **Output Generation**: Provides outcomes and results file path for downstream badge generation

### 2. generate-rtl-badges Action

**Purpose**: Generate RTL test result badges and upload to GitHub Gists

**Location**: `.github/actions/generate-rtl-badges/action.yml`

**Interface**:
```yaml
name: 'Generate RTL Badges'
description: 'Generate RTL test result badges and upload to GitHub Gists'
inputs:
  gist-id:
    description: 'GitHub Gist ID for badge storage'
    required: true
  gist-token:
    description: 'GitHub token for gist access'
    required: true
  repo-name:
    description: 'Repository name for badge naming'
    required: true
  rtl-outcome:
    description: 'RTL test outcome (success/failure)'
    required: true
  rtl-exit-code:
    description: 'RTL test exit code'
    required: true
  results-file:
    description: 'Path to RTL test results file'
    required: true
```

**Usage Examples**:

#### With RTL Test Results
```yaml
- name: Generate RTL Badges
  uses: ./.github/actions/generate-rtl-badges
  with:
    gist-id: ${{ env.GIST_ID }}
    gist-token: ${{ env.GIST_TOKEN }}
    repo-name: ${{ env.REPO_NAME }}
    rtl-outcome: ${{ steps.run-rtl-tests.outputs.outcome }}
    rtl-exit-code: ${{ steps.run-rtl-tests.outputs.exit-code }}
    results-file: ${{ steps.run-rtl-tests.outputs.results-file }}
```

**Key Features**:
- **Badge Generation**: Creates dynamic RTL test status badges
- **Gist Integration**: Uploads badges to GitHub Gists for external access
- **Configurable Input**: Requires explicit results file path specification
- **Consistent API**: Follows same pattern as other badge generation actions

### 3. generate-status-badges Action

**Purpose**: Generate status badges based on workflow step outcomes

**Location**: `.github/actions/generate-status-badges/action.yml`

**Interface**:
```yaml
name: 'Generate Status Badges'
description: 'Creates status badges based on workflow step outcomes'
inputs:
  gist-id:
    description: 'GitHub Gist ID for badge storage'
    required: true
  gist-token:
    description: 'GitHub token for gist access'
    required: true
  repo-name:
    description: 'Repository name for badge naming'
    required: true
  static-tests-outcome:
    description: 'Outcome of static tests step (success, failure, skipped, cancelled)'
    required: true
  spdx-tests-outcome:
    description: 'Outcome of SPDX/license checks step (success, failure, skipped, cancelled)'
    required: true
  mypy-badge-label:
    description: 'Label for MyPy badge'
    required: false
    default: 'MyPy'
  spdx-badge-label:
    description: 'Label for SPDX badge'
    required: false
    default: 'SPDX'
```

**Usage Examples**:

#### With Test Outcomes
```yaml
- name: Generate Status Badges
  uses: ./.github/actions/generate-status-badges
  with:
    gist-id: ${{ env.GIST_ID }}
    gist-token: ${{ env.GIST_TOKEN }}
    repo-name: ${{ env.REPO_NAME }}
    static-tests-outcome: ${{ steps.run-static-analysis.outputs.static-tests-outcome }}
    spdx-tests-outcome: ${{ steps.run-static-analysis.outputs.license-checks-outcome }}
```

**Key Features**:
- **Multi-badge Generation**: Creates both MyPy and SPDX status badges
- **Outcome-based Colors**: Badge colors determined by test step outcomes
- **Explicit Configuration**: Requires explicit specification of test outcomes
- **Configurable Labels**: Optional customization of badge labels

### 4. generate-test-badges Action

**Purpose**: Generate test and coverage badges from test results and upload to GitHub Gists

**Location**: `.github/actions/generate-test-badges/action.yml`

**Interface**:
```yaml
name: 'Generate Test and Coverage Badges'
description: 'Creates dynamic badges from test results and uploads to GitHub Gists'
inputs:
  gist-id:
    description: 'GitHub Gist ID for badge storage'
    required: true
  gist-token:
    description: 'GitHub token for gist access'
    required: true
  repo-name:
    description: 'Repository name for badge naming'
    required: true
  results-dir:
    description: 'Directory containing test result files'
    required: false
    default: '__ci/json'
  coverage-yellow-threshold:
    description: 'Coverage percentage threshold for yellow color'
    required: true
  coverage-required-threshold:
    description: 'Coverage percentage threshold for red color'
    required: true
  test-yellow-threshold:
    description: 'Test pass rate threshold for yellow color'
    required: true
  test-required-threshold:
    description: 'Test pass rate threshold for red color'
    required: true
```

**Usage Examples**:

#### With Custom Thresholds
```yaml
- name: Generate Test and Coverage Badges
  uses: ./.github/actions/generate-test-badges
  with:
    gist-id: ${{ env.GIST_ID }}
    gist-token: ${{ env.GIST_TOKEN }}
    repo-name: ${{ env.REPO_NAME }}
    coverage-yellow-threshold: '75'
    coverage-required-threshold: '85'
    test-yellow-threshold: '75'
    test-required-threshold: '95'
```

**Key Features**:
- **Dual Badge Generation**: Creates both test pass rate and coverage percentage badges
- **Configurable Thresholds**: Requires explicit specification of all color thresholds
- **Automatic Metrics Extraction**: Processes test results and coverage data from JSON files
- **Color-coded Status**: Dynamic badge colors based on performance vs thresholds
- **Gist Integration**: Uploads badges to GitHub Gists for external access

### 5. lfcdownload Action

**Purpose**: Downloads required files from Large File Cache (LFC) for testing

**Location**: `.github/actions/lfcdownload/action.yml`

**Interface**:
```yaml
name: 'Download required files from LFC'
description: 'Download required files from LFC'
inputs:
  files:
    description: 'Files to download, space separated list'
    required: false
    default: 'ext_test_onnx_models.tar.gz ext_llk_elf_files.tar.gz ext_rtl_test_data_set_sep23.tar.gz'
```

**Implementation**:
```yaml
runs:
  using: "composite"
  steps:
    - name: Download files from LFC
      shell: bash
      run: |
         for file in ${{ inputs.files }}; do
            bash tools/ci/lfc_downloader.sh --extract $file
         done
```

**Usage Examples**:

#### Default Usage (All Files)
```yaml
- name: Download required files from LFC
  uses: ./.github/actions/lfcdownload
```

#### Custom File Selection
```yaml
- name: Download specific files from LFC
  uses: ./.github/actions/lfcdownload
  with:
    files: 'ext_test_onnx_models.tar.gz ext_llk_elf_files.tar.gz ext_rtl_test_data_set_sep23.tar.gz'
```

#### RTL Test Files Only
```yaml
- name: Download RTL test data
  uses: ./.github/actions/lfcdownload
  with:
    files: 'ext_rtl_test_data_set_jul27.tar.gz'
```

**Default Files Downloaded**:
- `ext_test_onnx_models.tar.gz` → `tests/__models/`
- `ext_llk_elf_files.tar.gz` → `tests/__data_files/llk_elf_files/`
- `ext_rtl_test_data_set_sep23.tar.gz` → RTL test data

### 2. setup_mamba Action

**Purpose**: Sets up Mamba/Micromamba environment with configurable parameters

**Location**: `.github/actions/setup_mamba/action.yml`

**Interface**:
```yaml
name: 'Setup mamba'
description: 'Setup mamba'
inputs:
  environment-file:
    description: 'Path to the environment file'
    required: false
    default: 'envdev.yaml'
  environment-name:
    description: 'Name of the conda environment'
    required: false
    default: 'polenvdev'
```

**Implementation**:
```yaml
runs:
  using: "composite"
  steps:
    - uses: mamba-org/setup-micromamba@v2
      id: setup-mamba
      with:
        environment-file: ${{ inputs.environment-file }}
        environment-name: ${{ inputs.environment-name }}
        post-cleanup: all

    - name: check-micromamba-config
      shell: bash
      run: |
        micromamba info
        micromamba list
```

**Usage Examples**:

#### Developer Environment
```yaml
- name: Setup mamba - developer
  uses: ./.github/actions/setup_mamba
  with:
    environment-file: envdev.yaml
    environment-name: poldevenv
```

#### User Environment
```yaml
- name: Setup mamba - user
  uses: ./.github/actions/setup_mamba
  with:
    environment-file: environment.yaml
    environment-name: poluserenv
```

#### Default Usage
```yaml
- name: Setup mamba
  uses: ./.github/actions/setup_mamba
  # Uses defaults: envdev.yaml and polenvdev
```

## Workflow Implementation Patterns

### Standard Workflow Structure

All workflows follow a consistent pattern:

```yaml
# 1. Header with SPDX license
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# 2. Workflow metadata
name: Workflow Name
on: [triggers]

# 3. Jobs with standard structure
jobs:
  job-name:
    runs-on: tt-ubuntu-2204-large-stable
    
    steps:
    # 4. Standard setup steps
    - name: Checkout
      uses: actions/checkout@v4
      
    - name: Download required files from LFC
      uses: ./.github/actions/lfcdownload
      # Optional: with:
      #   files: 'specific_file.tar.gz'
      
    - name: Setup mamba - [environment-type]
      uses: ./.github/actions/setup_mamba
      with:
        environment-file: [env-file]
        environment-name: [env-name]
    
    # 5. Workflow-specific business logic
    - name: [Business Logic Steps]
      run: |
        # Workflow-specific commands
```

### Workflow Comparison

| Workflow | Purpose | Environment | Key Features |
|----------|---------|-------------|--------------|
| **checkin_tests.yml** | Pre-merge validation with integrated RTL testing | Developer + User | Unit tests, coverage, static analysis, RTL tests, badge generation |
| **nightly_tests.yml** | Comprehensive testing | Developer | Full test suite including slow tests |

## Implementation Examples

### 1. checkin_tests.yml

**Purpose**: Pre-merge validation with multiple test environments

**Key Features**:
- **Two jobs**: Developer environment tests + User environment tests
- **Comprehensive validation**: Unit tests, coverage, static analysis, license checks
- **Parallel execution**: Different environments run concurrently

**Action Usage**:
```yaml
jobs:
  checkin-tests:
    steps:
    - uses: ./.github/actions/lfcdownload
      # Uses default files: all test data files
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: envdev.yaml
        environment-name: poldevenv

  userenv-tests:
    steps:
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: environment.yaml
        environment-name: poluserenv
```

### 2. nightly_tests.yml

**Purpose**: Comprehensive nightly testing

**Key Features**:
- **Full test suite**: Includes slow tests excluded from check-in
- **Single environment**: Developer environment only
- **Scheduled execution**: Runs nightly via cron

**Action Usage**:
```yaml
jobs:
  nightly-tests:
    steps:
    - uses: ./.github/actions/lfcdownload
      # Uses default files: all test data files
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: envdev.yaml
        environment-name: poldevenv
```

### 3. rtl_tests.yml

**Purpose**: RTL-specific nightly testing with specialized test data

**Key Features**:
- **RTL test execution**: Runs RTL tests with specific data sets
- **Scheduled execution**: Runs nightly at 19:00 UTC via cron
- **Error handling**: Continues on error with result capture
- **Artifact collection**: Captures test results and logs

**Action Usage**:
```yaml
jobs:
  nightly-rtl-tests:
    steps:
    - uses: ./.github/actions/lfcdownload
      with:
        files: 'ext_rtl_test_data_set_sep23.tar.gz'
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: envdev.yaml
        environment-name: poldevenv
```

**Special Features**:
- Uses parameterized lfcdownload action to download only RTL test data
- Runs tests with `--tag jul27 --parallel 4` for specific test suite
- Captures output with `tee` for debugging
- Uploads test results as artifacts


## Development Best Practices

### Creating New Reusable Actions

#### 1. Action Structure
```yaml
# .github/actions/my-action/action.yml
name: 'My Action'
description: 'Description of what this action does'
inputs:
  parameter-name:
    description: 'Parameter description'
    required: false
    default: 'default-value'
outputs:
  output-name:
    description: 'Output description'
    value: ${{ steps.step-id.outputs.output-name }}
runs:
  using: "composite"
  steps:
    - name: Action Step
      shell: bash
      run: |
        # Action implementation
```

#### 2. Design Principles

**Single Responsibility**:
```yaml
# ✅ Good - Single, focused responsibility
name: 'Setup Python Environment'
description: 'Install Python and dependencies'

# ❌ Bad - Multiple responsibilities
name: 'Setup Everything'
description: 'Setup Python, download files, run tests'
```

**Parameterization**:
```yaml
# ✅ Good - Configurable parameters
inputs:
  python-version:
    description: 'Python version to install'
    required: false
    default: '3.11'
  requirements-file:
    description: 'Requirements file path'
    required: false
    default: 'requirements.txt'

# ❌ Bad - Hard-coded values
steps:
  - name: Setup Python 3.11
    # Hard-coded version
```

**Error Handling**:
```yaml
# ✅ Good - Explicit error handling
- name: Validate inputs
  shell: bash
  run: |
    if [[ ! -f "${{ inputs.requirements-file }}" ]]; then
      echo "Error: Requirements file not found: ${{ inputs.requirements-file }}"
      exit 1
    fi
```

### Workflow Development Guidelines

#### 1. Action Integration
```yaml
# ✅ Good - Use custom actions for setup
- name: Download required files from LFC
  uses: ./.github/actions/lfcdownload

- name: Setup mamba - developer
  uses: ./.github/actions/setup_mamba
  with:
    environment-file: envdev.yaml
    environment-name: poldevenv

# ❌ Bad - Duplicate setup logic
- name: Setup mamba manually
  uses: mamba-org/setup-micromamba@v2
  with:
    environment-file: envdev.yaml
    environment-name: poldevenv
    post-cleanup: all
- name: Check config
  run: |
    micromamba info
    micromamba list
```

#### 2. Error Resilience
```yaml
# ✅ Good - Continue on errors when appropriate
- name: Run Static Tests
  if: always()
  run: python checkin_tests.py static

- name: Run License and Copyright Checks
  if: always()
  run: python tools/spdxchecker.py [options]
```

#### 3. Artifact Management
```yaml
# ✅ Good - Consistent artifact patterns
- name: Upload artifacts
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: descriptive-artifact-name
    path: |
       .coverage
       __RUN_TESTS
       !__RUN_TESTS/**/*.onnx
```

## Testing and Validation

### Action Testing Strategy

#### 1. Local Testing
```bash
# Test action components locally
cd .github/actions/setup_mamba
# Validate action.yml syntax
yq eval . action.yml

# Test shell scripts
bash -n script.sh  # Syntax check
```

#### 2. Integration Testing
```yaml
# Test action in isolation
name: Test Custom Actions
on: [workflow_dispatch]
jobs:
  test-actions:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: ./.github/actions/lfcdownload
    - uses: ./.github/actions/setup_mamba
```

#### 3. Workflow Validation
```bash
# Validate workflow syntax
yq eval .github/workflows/checkin_tests.yml

# Test workflows on feature branches
git push origin feature-branch
# Monitor workflow execution
```

## Migration Strategy

### Converting Inline Logic to Actions

#### Before: Inline Setup Logic
```yaml
# Duplicated across multiple workflows
- name: Setup micromamba
  uses: mamba-org/setup-micromamba@v2
  with:
    environment-file: envdev.yaml
    environment-name: poldevenv
    post-cleanup: all

- name: Check config
  run: |
    micromamba info
    micromamba list

- name: Download files
  run: |
    bash tools/ci/lfc_downloader.sh --extract ext_test_onnx_models.tar.gz
    bash tools/ci/lfc_downloader.sh --extract ext_llk_elf_files.tar.gz
    bash tools/ci/lfc_downloader.sh --extract ext_rtl_test_data_set_sep23.tar.gz
```

#### After: Reusable Actions
```yaml
# Clean, consistent usage with parameterization
- name: Download required files from LFC
  uses: ./.github/actions/lfcdownload
  # Uses default files, or specify custom:
  # with:
  #   files: 'ext_test_onnx_models.tar.gz ext_llk_elf_files.tar.gz ext_rtl_test_data_set_sep23.tar.gz'

- name: Setup mamba - developer
  uses: ./.github/actions/setup_mamba
  with:
    environment-file: envdev.yaml
    environment-name: poldevenv
```

### Migration Process

1. **Identify Common Patterns**: Look for repeated setup logic across workflows
2. **Extract to Actions**: Create parameterized actions for common patterns
3. **Test Actions**: Validate actions work across different scenarios
4. **Update Workflows**: Replace inline logic with action calls
5. **Cleanup**: Remove redundant setup code

## Maintenance and Evolution

### Version Management

#### Action Versioning
```yaml
# Option 1: Use specific commits (recommended for stability)
uses: ./.github/actions/setup_mamba@abc123

# Option 2: Use relative paths (current approach)
uses: ./.github/actions/setup_mamba

# Option 3: Use tags (for published actions)
uses: ./.github/actions/setup_mamba@v1.0
```

#### Backward Compatibility
```yaml
# Maintain backward compatibility with defaults
inputs:
  new-parameter:
    description: 'New optional parameter'
    required: false
    default: 'legacy-behavior'
```

### Documentation Requirements

#### Action Documentation
```yaml
# Each action should include:
name: 'Clear, descriptive name'
description: 'Detailed description of purpose and behavior'
inputs:
  parameter:
    description: 'Clear parameter description with examples'
    required: true/false
    default: 'default-value'
```

#### Workflow Documentation
```yaml
# Workflows should include:
# - Purpose and scope
# - Trigger conditions
# - Environment requirements
# - Expected artifacts
```

## Troubleshooting Guide

### Common Issues

#### Action Not Found
```yaml
# ❌ Problem
uses: ./.github/actions/nonexistent

# ✅ Solution - Verify path exists
uses: ./.github/actions/setup_mamba
```

#### Missing Inputs
```yaml
# ❌ Problem
uses: ./.github/actions/setup_mamba
# Missing required inputs

# ✅ Solution - Provide required inputs
uses: ./.github/actions/setup_mamba
with:
  environment-file: envdev.yaml
  environment-name: poldevenv
```

#### Shell Script Errors
```bash
# Debug action shell scripts
set -x  # Enable debug output
set -e  # Exit on errors
```

### Debugging Strategies

#### 1. Enable Debug Logging
```yaml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

#### 2. Add Debug Steps
```yaml
- name: Debug Environment
  run: |
    echo "Working directory: $(pwd)"
    echo "Environment file exists: $(test -f envdev.yaml && echo yes || echo no)"
    ls -la
```

#### 3. Test in Isolation
```yaml
# Create minimal test workflow
name: Debug Action
on: [workflow_dispatch]
jobs:
  debug:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: envdev.yaml
        environment-name: poldevenv
```

## Future Enhancements

### Potential New Actions

1. **Test Results Action**: Standardize test result processing and reporting
2. **Security Scan Action**: Centralize security scanning logic
3. **Documentation Action**: Automate documentation generation and updates
4. **Deployment Action**: Standardize deployment procedures

### Advanced Patterns

#### Conditional Actions
```yaml
- name: Setup Windows Environment
  if: runner.os == 'Windows'
  uses: ./.github/actions/setup_windows

- name: Setup Linux Environment
  if: runner.os == 'Linux'
  uses: ./.github/actions/setup_linux
```

#### Matrix Strategy with Actions
```yaml
strategy:
  matrix:
    environment: [envdev.yaml, environment.yaml]
    name: [poldevenv, poluserenv]

steps:
- uses: ./.github/actions/setup_mamba
  with:
    environment-file: ${{ matrix.environment }}
    environment-name: ${{ matrix.name }}
```

## License

This GitHub Actions architecture is part of the Tenstorrent AI ULC project and is licensed under Apache-2.0.

---

**See Also:**
- `.github/actions/` - Custom action implementations
- `.github/workflows/` - Workflow examples using these actions
- `doc/tools/ci/README_dynamic_badges.md` - Badge generation workflow documentation
- GitHub Actions Documentation: https://docs.github.com/en/actions
