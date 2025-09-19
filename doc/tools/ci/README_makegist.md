# makegist.py - GitHub Gist Dictionary Creator

## Overview

`makegist.py` is a Python script that creates dictionaries from key=value pairs, uploads files directly to GitHub gists, or merges both approaches by combining JSON files with additional key=value pairs, using GitHub's gist API.

## Features

- **Key-Value Pair Parsing**: Accepts arguments in `key=value` format
- **File Upload**: Upload any file directly to a gist using `--input-file`
- **JSON Merging**: Combine a JSON file with additional key=value pairs
- **GitHub Gist Integration**: Updates existing GitHub gists via REST API
- **JSON Output**: Converts dictionary to properly formatted JSON
- **Comprehensive Logging**: Uses Python's loguru module with timestamps
- **Error Handling**: Detailed error messages for various failure scenarios
- **Flexible Input**: Supports key=value pairs, file upload, or both combined

## Prerequisites

- Python 3.13 or higher
- GitHub personal access token with gist permissions
- Existing GitHub gist ID
- Internet connection for API calls

## Installation

The script is included in the `tools/` directory and requires no additional installation beyond the Python standard library.

## Usage

### Basic Syntax

**Using key=value pairs:**
```bash
python3 tools/ci/makegist.py --gist-id <GIST_ID> --gist-filename <FILENAME> [key=value pairs...]
```

**Using file upload:**
```bash
python3 tools/ci/makegist.py --gist-id <GIST_ID> --gist-filename <FILENAME> --input-file <FILE_PATH>
```

**Using JSON file + key=value pairs (merging):**
```bash
python3 tools/ci/makegist.py --gist-id <GIST_ID> --gist-filename <FILENAME> --input-file <JSON_FILE> [key=value pairs...]
```

**With explicit token:**
```bash
python3 tools/ci/makegist.py --gist-token <TOKEN> --gist-id <GIST_ID> --gist-filename <FILENAME> [options...]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--gist-id` | The gist ID to update (found in the gist URL) |
| `--gist-filename` | The filename within the gist to update |

### Optional Arguments

| Argument | Description |
|----------|-------------|
| `--gist-token` | GitHub personal access token for gist access (defaults to `GIST_TOKEN` environment variable) |
| `--input-file` | Path to a file to upload to the gist. If used with key=value pairs, must be a JSON file containing a dictionary |
| `key=value pairs` | One or more key=value pairs to include in the dictionary. If used with --input-file, will be merged with the JSON dictionary |

## Examples

### Example 1: Basic Usage (using environment variable)

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id abc123def456 \
  --gist-filename config.json \
  name=John \
  age=30 \
  city=NYC
```

Or with explicit token:

```bash
python3 tools/ci/makegist.py \
  --gist-token ghp_xxxxxxxxxxxxxxxxxxxx \
  --gist-id abc123def456 \
  --gist-filename config.json \
  name=John \
  age=30 \
  city=NYC
```

This creates a JSON dictionary:
```json
{
  "name": "John",
  "age": "30",
  "city": "NYC"
}
```

### Example 2: Configuration Data (using environment variable)

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id def456ghi789 \
  --gist-filename settings.json \
  database_url=postgresql://localhost:5432/mydb \
  api_key=sk-xxxxxxxxxxxxxxxxxxxx \
  debug_mode=true \
  max_connections=100
```

### Example 3: File Upload

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id ghi789jkl012 \
  --gist-filename document.md \
  --input-file /path/to/README.md
```

This uploads the contents of `/path/to/README.md` directly to the gist.

### Example 4: JSON File + Key=Value Merging

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id jkl012mno345 \
  --gist-filename merged_config.json \
  --input-file base_config.json \
  environment=production \
  version=2.1.0 \
  debug=false
```

This reads `base_config.json` and merges it with the additional key=value pairs. If `base_config.json` contains:
```json
{
  "database_url": "postgresql://localhost:5432/mydb",
  "debug": true,
  "timeout": 30
}
```

The final result will be:
```json
{
  "database_url": "postgresql://localhost:5432/mydb",
  "debug": "false",
  "timeout": 30,
  "environment": "production",
  "version": "2.1.0"
}
```

Note: Key=value pairs override values from the JSON file if keys conflict.

### Example 5: Empty Dictionary

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id jkl012mno345 \
  --gist-filename empty.json
```

This creates an empty dictionary `{}`.

## GitHub Setup

### Creating a Personal Access Token

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Select the `gist` scope
4. Copy the generated token (it won't be shown again)

### Finding Gist ID

The gist ID is found in the gist URL:
```
https://gist.github.com/username/abc123def456
                              ^^^^^^^^^^^^^^^^
                              This is the gist ID
```

## Output

### Success

When successful, the script outputs:
- Confirmation message with gist ID and filename
- The content that was written to the gist (for key=value pairs) or confirmation of file upload

**Key=value pairs example:**
```
2025-09-11 12:54:46,370 - INFO - Successfully updated gist abc123def456 with file config.json
2025-09-11 12:54:46,371 - INFO - Dictionary content:
2025-09-11 12:54:46,371 - INFO - {
  "name": "John",
  "age": "30",
  "city": "NYC"
}
```

**File upload example:**
```
2025-09-11 12:54:46,370 - INFO - Read 1234 characters from file: /path/to/document.md
2025-09-11 12:54:46,371 - INFO - Successfully updated gist abc123def456 with file document.md
2025-09-11 12:54:46,372 - INFO - Uploaded content from file: /path/to/document.md
```

**JSON merging example:**
```
2025-09-11 12:54:46,370 - INFO - Merged JSON file (3 keys) with key=value pairs (2 keys)
2025-09-11 12:54:46,371 - INFO - Successfully updated gist abc123def456 with file config.json
2025-09-11 12:54:46,372 - INFO - Merged JSON file base.json with key=value pairs
2025-09-11 12:54:46,373 - INFO - Final content:
2025-09-11 12:54:46,374 - INFO - {
  "database_url": "postgresql://localhost:5432/mydb",
  "debug": "false",
  "timeout": 30,
  "environment": "production",
  "version": "2.1.0"
}
```

### Error Handling

The script provides detailed error messages for various scenarios:

#### Invalid Arguments
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json invalid_format
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - Error: Invalid format: 'invalid_format'. Expected key=value format.
```

#### Empty Key
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json =value
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - Error: Empty key in argument: '=value'
```

#### Missing Token Error
```bash
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json name=John
```
Output (when GIST_TOKEN is not set):
```
2025-09-11 12:54:46,370 - ERROR - Error: GitHub token must be provided via --gist-token argument or GIST_TOKEN environment variable
```

#### Invalid JSON File Error
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json --input-file data.txt name=John
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - Error: When using both --input-file and key=value pairs, the input file must be a .json file
```

#### Invalid JSON Content Error
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json --input-file invalid.json name=John
```
Output (when invalid.json contains invalid JSON):
```
2025-09-11 12:54:46,370 - ERROR - Error: Invalid JSON in input file: Expecting ',' delimiter: line 2 column 5 (char 15)
```

#### Non-Dictionary JSON Error
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json --input-file array.json name=John
```
Output (when array.json contains `[1,2,3]` instead of `{}`):
```
2025-09-11 12:54:46,370 - ERROR - Error: JSON file must contain a dictionary/object at the root level
```

#### File Not Found Error
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json --input-file nonexistent.txt
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - Error: Input file does not exist: nonexistent.txt
```

#### Authentication Error
```bash
export GIST_TOKEN="invalid_token"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json name=John
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - HTTP Error updating gist: 401 - Unauthorized
2025-09-11 12:54:46,371 - ERROR - Response: {"message":"Bad credentials","documentation_url":"https://docs.github.com/rest","status":"401"}
2025-09-11 12:54:46,371 - ERROR - Failed to update gist
```

#### Network Error
```bash
# When offline or GitHub is down
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py --gist-id id --gist-filename file.json name=John
```
Output:
```
2025-09-11 12:54:46,370 - ERROR - URL Error updating gist: [Errno 8] nodename nor servname provided, or not known
2025-09-11 12:54:46,371 - ERROR - Failed to update gist
```

## Logging

The script uses Python's loguru module with the following configuration:

- **Level**: INFO (shows info, warning, and error messages)
- **Format**: `YYYY-MM-DD HH:mm:ss - LEVEL - message`
- **Output**: stderr (separate from data output)

### Log Levels

- **INFO**: Successful operations and general information
- **WARNING**: Non-critical issues (e.g., empty dictionary)
- **ERROR**: Critical errors that prevent operation

## Advanced Usage

### Using with Environment Variables

**Recommended**: Store your token in the `GIST_TOKEN` environment variable for automatic detection:

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id abc123def456 \
  --gist-filename config.json \
  name=John \
  age=30
```

Alternatively, use a custom environment variable:

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-token "$GITHUB_TOKEN" \
  --gist-id abc123def456 \
  --gist-filename config.json \
  name=John \
  age=30
```

### Script Integration

The script can be integrated into larger workflows:

```bash
#!/bin/bash
# Example workflow script using key=value pairs

export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
GIST_ID="abc123def456"
FILENAME="build_info.json"

python3 tools/ci/makegist.py \
  --gist-id "$GIST_ID" \
  --gist-filename "$FILENAME" \
  build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  git_commit="$(git rev-parse HEAD)" \
  build_number="$BUILD_NUMBER" \
  environment="$ENVIRONMENT"
```

```bash
#!/bin/bash
# Example workflow script using file upload

export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
GIST_ID="def456ghi789"

# Upload a log file
python3 tools/ci/makegist.py \
  --gist-id "$GIST_ID" \
  --gist-filename "build.log" \
  --input-file "/tmp/build.log"

# Upload a generated report
python3 tools/ci/makegist.py \
  --gist-id "$GIST_ID" \
  --gist-filename "test-report.html" \
  --input-file "./reports/test-results.html"
```

```bash
#!/bin/bash
# Example workflow script using JSON merging

export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
GIST_ID="ghi789jkl012"

# Merge base configuration with build-specific data
python3 tools/ci/makegist.py \
  --gist-id "$GIST_ID" \
  --gist-filename "deployment_config.json" \
  --input-file "./config/base.json" \
  build_id="$BUILD_ID" \
  deployment_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  git_commit="$(git rev-parse HEAD)" \
  environment="$DEPLOY_ENV" \
  version="$APP_VERSION"

# Update monitoring dashboard with current metrics
python3 tools/ci/makegist.py \
  --gist-id "$GIST_ID" \
  --gist-filename "metrics.json" \
  --input-file "./monitoring/base_metrics.json" \
  last_updated="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  active_deployments="$(kubectl get deployments --no-headers | wc -l)" \
  cluster_status="healthy"
```

### Special Characters

The script handles special characters in values:

```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id id \
  --gist-filename file.json \
  description="This is a test with spaces and special chars: !@#$%^&*()" \
  json_data='{"nested": "value"}' \
  multiline="Line 1\nLine 2\nLine 3"
```

### File Upload Examples

Upload a text file:
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id abc123def456 \
  --gist-filename notes.txt \
  --input-file /path/to/notes.txt
```

Upload a configuration file:
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id def456ghi789 \
  --gist-filename config.yaml \
  --input-file ./config/production.yaml
```

### JSON Merging Examples

Merge base configuration with environment-specific values:
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id mno345pqr678 \
  --gist-filename app_config.json \
  --input-file ./config/base.json \
  environment=staging \
  log_level=debug \
  feature_flags='{"new_ui": true}'
```

Update build metadata by merging with existing data:
```bash
export GIST_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
python3 tools/ci/makegist.py \
  --gist-id pqr678stu901 \
  --gist-filename build_info.json \
  --input-file ./previous_build.json \
  build_number="$(git rev-list --count HEAD)" \
  commit_hash="$(git rev-parse HEAD)" \
  build_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

## Troubleshooting

### Common Issues

1. **Missing Token**: Ensure `GIST_TOKEN` environment variable is set or provide `--gist-token` argument
2. **Invalid JSON File**: When using both `--input-file` and key=value pairs, the input file must be a valid JSON file with a dictionary at the root
3. **File Extension**: When merging, the input file must have a `.json` extension
4. **File Not Found**: Ensure the file specified in `--input-file` exists and is readable
5. **401 Unauthorized**: Check your GitHub token and ensure it has gist permissions
6. **404 Not Found**: Verify the gist ID exists and you have access to it
7. **Network Errors**: Check your internet connection and GitHub's status
8. **Invalid Format**: Ensure all key=value pairs use the `=` separator

### Debug Mode

For more detailed logging, you can modify the script to use DEBUG level:

```python
from loguru import logger
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="DEBUG", format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")
```

### Testing with Dummy Data

Test the script's parsing without making API calls:

```bash
python3 -c "
import sys
sys.path.append('tools')
from makegist import parse_key_value_pairs
print(parse_key_value_pairs(['name=John', 'age=30', 'city=NYC']))
"
```

## Security Considerations

- **Token Security**: Never commit GitHub tokens to version control
- **Environment Variables**: The script automatically uses the `GIST_TOKEN` environment variable for security
- **Token Permissions**: Use tokens with minimal required permissions (only `gist` scope needed)
- **Token Rotation**: Regularly rotate your GitHub tokens
- **No Token Logging**: The script never prints or logs the token value for security

## Related Tools

- GitHub CLI (`gh`): Alternative way to manage gists from command line
- `jq`: Command-line JSON processor for manipulating JSON data

## License

SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC  
SPDX-License-Identifier: Apache-2.0

## Support

For issues or questions:
1. Check the error messages for specific guidance
2. Verify your GitHub token and gist permissions
3. Test with a simple example first
4. Check GitHub's API status if experiencing network issues