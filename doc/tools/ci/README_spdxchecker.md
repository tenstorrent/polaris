# spdxchecker.py - SPDX License and Copyright Validator

## Overview

`spdxchecker.py` is a Python script that validates SPDX (Software Package Data Exchange) headers in source code files across your repository. It ensures license and copyright compliance by checking that all source files contain proper SPDX headers, making it essential for **preventing CI/CD regressions** and maintaining legal compliance.

## Why Use This Script in CI/CD?

### Problem: License Compliance Regressions

Without automated license checking, projects can suffer from:

```bash
# ❌ Common CI/CD regression scenarios:
# - New files added without proper license headers
# - Incorrect copyright information in source files  
# - Missing SPDX identifiers causing legal compliance issues
# - Inconsistent licensing across the codebase
# - Manual license audits that are error-prone and time-consuming
```

### Solution: Automated SPDX Validation

`spdxchecker.py` prevents these regressions by automatically validating every source file:

```bash
# ✅ Automated CI/CD integration prevents:
python tools/spdxchecker.py --ignore .github/spdxchecker-ignore.yml \
    --allowed-licenses Apache-2.0 \
    --allowed-copyright "Tenstorrent AI ULC"
# Exit code 0 = all files compliant, Exit code 1 = violations found
```

## Usage

### Basic Syntax
```bash
python tools/spdxchecker.py [OPTIONS]
```

### Key Arguments
- `--allowed-licenses`: List of acceptable license identifiers (default: `Apache-2.0`)
- `--allowed-copyright`: Required copyright holder (default: `Tenstorrent AI ULC`)
- `--ignore/-i`: YAML file with ignore patterns for files to skip
- `--gitignore`: Respect .gitignore patterns (default: `true`)
- `--loglevel/-l`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `--dryrun/-n`: Test mode without making changes

## Supported File Types

The script validates SPDX headers in multiple programming languages:

| Language | Extensions | Comment Syntax |
|----------|------------|----------------|
| **Python** | `.py` | `# SPDX-License-Identifier: Apache-2.0` |
| **Shell** | `.sh` | `# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC` |
| **JavaScript** | `.js`, `.mjs`, `.cjs` | `// SPDX-License-Identifier: Apache-2.0` |
| **HTML** | `.html`, `.htm` | `<!-- SPDX-License-Identifier: Apache-2.0 -->` |
| **CSS** | `.css` | `/* SPDX-License-Identifier: Apache-2.0 */` |
| **YAML** | `.yaml`, `.yml` | `# SPDX-License-Identifier: Apache-2.0` |

### Required SPDX Headers

Each source file must contain both headers:

```python
#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Your code here...
```

```bash
#!/bin/bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC  
# SPDX-License-Identifier: Apache-2.0

# Your script here...
```

```javascript
// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Your JavaScript code here...
```

## CI/CD Integration Examples

### GitHub Actions Workflow
```yaml
# Prevent license compliance regressions
- name: Run License and Copyright Checks
  if: always()  # Run even if previous steps fail
  run: |
    python tools/spdxchecker.py \
      --ignore .github/spdxchecker-ignore.yml \
      --allowed-licenses Apache-2.0 \
      --allowed-copyright "Tenstorrent AI ULC"
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
repos:
- repo: local
  hooks:
    - id: spdx-checker
      name: Check SPDX License and Copyright
      language: python
      entry: tools/spdxchecker.py
      args: [
        --loglevel, error,
        --ignore, .github/spdxchecker-ignore.yml,
        --allowed-licenses, Apache-2.0,
        --allowed-copyright, "Tenstorrent AI ULC"
      ]
      additional_dependencies: [pyyaml, loguru, pydantic]
      pass_filenames: false
```

### Jenkins Pipeline
```groovy
pipeline {
    stages {
        stage('License Compliance') {
            steps {
                script {
                    def result = sh(
                        script: '''
                            python tools/spdxchecker.py \
                                --ignore .github/spdxchecker-ignore.yml \
                                --allowed-licenses Apache-2.0 \
                                --allowed-copyright "Tenstorrent AI ULC"
                        ''',
                        returnStatus: true
                    )
                    if (result != 0) {
                        error("License compliance check failed!")
                    }
                }
            }
        }
    }
}
```

### GitLab CI
```yaml
# .gitlab-ci.yml
license_check:
  stage: compliance
  script:
    - python tools/spdxchecker.py 
        --ignore .github/spdxchecker-ignore.yml
        --allowed-licenses Apache-2.0
        --allowed-copyright "Tenstorrent AI ULC"
  allow_failure: false  # Fail pipeline on license violations
```

## Configuration: Ignore Patterns

### Ignore File Format (`.github/spdxchecker-ignore.yml`)
```yaml
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

ignore:
  # Files to completely skip (no SPDX headers required)
  - '*.json'        # Data files
  - '*.onnx'        # Model files  
  - '*.csv'         # Data exports
  - '*.xlsx'        # Spreadsheets
  - '*.md'          # Documentation
  - '*.txt'         # Text files
  - '.gitignore'    # Git configuration
  - 'condarc'       # Conda configuration

warning:
  # Files that should have headers but won't fail CI if missing
  - '*.html'        # Generated HTML
  - '*.yaml'        # Configuration files
  - '*.yml'         # YAML configs
```

### Pattern Matching
- Uses **glob patterns** (`*.ext`, `path/*/file.ext`)
- **Case sensitive** matching
- **Recursive** directory scanning (respects `.gitignore`)

## Error Types and Resolution

### 1. Missing Headers
```bash
ERROR:tools/newscript.py: License: missing, Copyright: missing
```
**Solution:** Add both SPDX headers to the file:
```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
```

### 2. Incorrect License
```bash
ERROR:src/utils.js: License: incorrect, Copyright: ok
```
**Solution:** Use an allowed license identifier:
```javascript
// SPDX-License-Identifier: Apache-2.0  // ✅ Correct
// SPDX-License-Identifier: MIT         // ❌ Not in allowed list
```

### 3. Incorrect Copyright
```bash
ERROR:scripts/deploy.sh: License: ok, Copyright: incorrect
```
**Solution:** Use the exact copyright holder:
```bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC  # ✅ Correct
# SPDX-FileCopyrightText: (C) 2025 My Company         # ❌ Wrong holder
```

### 4. Ill-formed Headers
```bash
ERROR:config/settings.py: License: ok, Copyright: illformed
```
**Solution:** Fix the copyright format:
```python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC    # ✅ Correct
# SPDX-FileCopyrightText: Copyright 2025 Tenstorrent      # ❌ Wrong format
```

## CI/CD Regression Prevention Strategies

### 1. **Fail-Fast Approach**
```yaml
# Run license check early in pipeline
- name: License Compliance Check
  run: python tools/spdxchecker.py --loglevel error
  # Pipeline stops here if violations found
```

### 2. **Branch Protection Rules**
```yaml
# GitHub branch protection
# Require "License Check" status to pass before merge
required_status_checks:
  contexts: ["License Check"]
```

### 3. **Developer Workflow Integration**
```bash
# Add to developer setup scripts
echo "Setting up pre-commit hooks..."
pre-commit install

# Developers get immediate feedback on license violations
git commit -m "Add new feature"
# → Pre-commit hook runs spdxchecker.py
# → Commit blocked if headers missing
```

### 4. **Automated Remediation**
```bash
# Script to add missing headers
#!/bin/bash
# add_spdx_headers.sh

find . -name "*.py" -exec grep -L "SPDX-License-Identifier" {} \; | while read file; do
    echo "Adding SPDX headers to $file"
    sed -i '1i# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC\n# SPDX-License-Identifier: Apache-2.0\n' "$file"
done
```

## Advanced Usage

### Custom License Validation
```bash
# Allow multiple licenses
python tools/spdxchecker.py \
    --allowed-licenses Apache-2.0 MIT BSD-3-Clause \
    --allowed-copyright "Tenstorrent AI ULC"
```

### Debug Mode for Troubleshooting
```bash
# Detailed logging for debugging
python tools/spdxchecker.py \
    --loglevel DEBUG \
    --ignore .github/spdxchecker-ignore.yml
```

### Dry Run Mode
```bash
# Test configuration without failing
python tools/spdxchecker.py \
    --dryrun \
    --loglevel INFO
```

### Custom Ignore Patterns
```bash
# Use different ignore file
python tools/spdxchecker.py \
    --ignore custom-ignore-patterns.yml \
    --allowed-licenses Apache-2.0
```

## Integration with Other Tools

### Combined with Code Quality Checks
```yaml
# GitHub Actions - comprehensive quality gate
- name: Code Quality Gate
  run: |
    # License compliance (fail-fast)
    python tools/spdxchecker.py --loglevel error
    
    # Code formatting
    black --check .
    
    # Linting
    flake8 .
    
    # Type checking  
    mypy .
    
    # Security scanning
    bandit -r .
```

### Makefile Integration
```makefile
# Makefile
.PHONY: license-check
license-check:
	python tools/spdxchecker.py \
		--ignore .github/spdxchecker-ignore.yml \
		--allowed-licenses Apache-2.0 \
		--allowed-copyright "Tenstorrent AI ULC"

.PHONY: ci-checks
ci-checks: license-check lint test
	@echo "All CI checks passed!"
```

## Best Practices

### 1. **Early Pipeline Integration**
- Run license checks **before** expensive operations (builds, tests)
- Fail fast to save CI resources and developer time

### 2. **Comprehensive Ignore Patterns**
- Maintain clear ignore patterns for non-source files
- Document why files are ignored
- Regular review of ignore patterns

### 3. **Developer Education**
- Include SPDX header templates in project documentation
- Provide editor snippets/templates for common file types
- Clear error messages and remediation guidance

### 4. **Monitoring and Reporting**
- Track license compliance metrics over time
- Alert on new file types that need SPDX support
- Regular audits of ignored files

## Troubleshooting

### Common Issues

#### Git Integration Problems
```bash
# Error: No .git directory found
# Solution: Run from repository root
cd /path/to/repository/root
python tools/spdxchecker.py
```

#### Ignore File Not Found
```bash
# Error: FileNotFoundError: ignore file
# Solution: Create ignore file or use --no-gitignore
python tools/spdxchecker.py --no-gitignore
```

#### Permission Errors
```bash
# Error: Permission denied reading files
# Solution: Check file permissions
chmod +r problematic_file.py
```

## Exit Codes

- **0**: All files compliant, no violations found
- **1**: License or copyright violations detected
- **Non-zero**: Script error (file not found, permission issues, etc.)

## Dependencies

```bash
# Required Python packages
pip install pyyaml loguru pydantic
```

## License

This script is part of the Tenstorrent AI ULC project and is licensed under Apache-2.0.

---

**See Also:**
- `.github/spdxchecker-ignore.yml` - Ignore pattern configuration
- `.pre-commit-config.yaml` - Pre-commit hook setup
- Project workflows in `.github/workflows/` for CI/CD integration examples
