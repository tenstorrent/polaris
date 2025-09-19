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
│   ├── lfcdownload/           # Large File Cache download action
│   │   └── action.yml
│   └── setup_mamba/           # Mamba environment setup action
│       └── action.yml
├── spdxchecker-ignore.yml     # SPDX license checker ignore rules
└── workflows/                 # CI/CD workflows using actions
    ├── checkin_tests.yml      # Pre-merge validation
    ├── rtl_tests.yml          # RTL-specific testing
    ├── nightly_tests.yml      # Comprehensive nightly testing
    └── post_mergepr.yml       # Post-merge status updates
```

### Action vs Workflow Responsibility

| Component | Responsibility | Scope |
|-----------|---------------|-------|
| **Actions** | Reusable setup/utility tasks | Single responsibility, parameterized |
| **Workflows** | Complete CI/CD pipelines | Orchestrate actions and business logic |

## Reusable Actions Reference

### 1. lfcdownload Action

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
    default: 'test_onnx_models.tar.gz llk_elf_files.tar.gz ext_rtl_test_data_set_feb19.tar.gz ext_rtl_test_data_set_jul1.tar.gz ext_rtl_test_data_set_jul27.tar.gz ext_rtl_test_data_set_mar18.tar.gz'
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
    files: 'test_onnx_models.tar.gz llk_elf_files.tar.gz'
```

#### RTL Test Files Only
```yaml
- name: Download RTL test data
  uses: ./.github/actions/lfcdownload
  with:
    files: 'ext_rtl_test_data_set_jul27.tar.gz'
```

**Default Files Downloaded**:
- `test_onnx_models.tar.gz` → `tests/__models/`
- `llk_elf_files.tar.gz` → `tests/__data_files/llk_elf_files/`
- `ext_rtl_test_data_set_feb19.tar.gz` → RTL test data
- `ext_rtl_test_data_set_jul1.tar.gz` → RTL test data
- `ext_rtl_test_data_set_jul27.tar.gz` → RTL test data
- `ext_rtl_test_data_set_mar18.tar.gz` → RTL test data

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
| **checkin_tests.yml** | Pre-merge validation | Developer + User | Unit tests, coverage, static analysis, license checks |
| **nightly_tests.yml** | Comprehensive testing | Developer | Full test suite including slow tests |
| **rtl_tests.yml** | RTL-specific testing | Developer | RTL test execution with specific data sets |
| **post_mergepr.yml** | Status updates | Developer | Badge generation, metrics collection |

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
        files: 'ext_rtl_test_data_set_jul27.tar.gz'
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

### 4. post_mergepr.yml

**Purpose**: Post-merge status updates and badge generation

**Key Features**:
- **Metrics collection**: Test results and coverage data
- **Badge generation**: Dynamic badges for README
- **Status reporting**: Upload to GitHub Gists

**Action Usage**:
```yaml
jobs:
  update-status-for-main-branch:
    steps:
    - uses: ./.github/actions/lfcdownload
    - uses: ./.github/actions/setup_mamba
      with:
        environment-file: envdev.yaml
        environment-name: poldevenv
```

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
    bash tools/ci/lfc_downloader.sh --extract test_onnx_models.tar.gz
    bash tools/ci/lfc_downloader.sh --extract llk_elf_files.tar.gz
    bash tools/ci/lfc_downloader.sh --extract ext_rtl_test_data_set_jul27.tar.gz
```

#### After: Reusable Actions
```yaml
# Clean, consistent usage with parameterization
- name: Download required files from LFC
  uses: ./.github/actions/lfcdownload
  # Uses default files, or specify custom:
  # with:
  #   files: 'test_onnx_models.tar.gz llk_elf_files.tar.gz'

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
