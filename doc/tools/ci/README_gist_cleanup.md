# Gist File Cleanup Process

## Overview

This document describes the cleanup process for temporary gist files created by the CI/CD badge generation system. Files with the `DELETEME_` prefix are temporary files created for non-main branches and should be periodically cleaned up.

## File Naming Convention

### Main Branch Files (Permanent)
Files created for the main branch use standard naming:
- `polaris_mypy_status.json`
- `polaris_spdx_status.json`
- `polaris_tests_badge.json`
- `polaris_coverage_badge.json`
- `polaris_unittestssummary.json`
- `polaris_unittestscoverage.json`
- `polaris_rtl_scurve_status.json`
- `polaris_rtl_scurve_ratio_geomean.json`

### Non-Main Branch Files (Temporary)
Files created for feature branches and other non-main branches use the `DELETEME_` prefix:
- `DELETEME_polaris_mypy_status.json`
- `DELETEME_polaris_spdx_status.json`
- `DELETEME_polaris_tests_badge.json`
- `DELETEME_polaris_coverage_badge.json`
- `DELETEME_polaris_unittestssummary.json`
- `DELETEME_polaris_unittestscoverage.json`
- `DELETEME_polaris_rtl_scurve_status.json`
- `DELETEME_polaris_rtl_scurve_ratio_geomean.json`

## Why DELETEME_ Prefix?

The `DELETEME_` prefix serves several purposes:

1. **Clear Identification**: Makes it obvious which files are temporary
2. **Easy Cleanup**: Simple to identify and remove files programmatically
3. **Retry Logic Testing**: Allows testing the retry functionality on all branches
4. **Gist Management**: Distinguishes between permanent and temporary files

## Cleanup Process

### Manual Cleanup

#### Using GitHub Web Interface

1. Navigate to the gist containing the files
2. Look for files with `DELETEME_` prefix
3. Delete these files manually

#### Using GitHub CLI

```bash
# List all files in the gist
gh gist view <GIST_ID>

# Delete specific files (replace with actual filenames)
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_mypy_status.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_spdx_status.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_tests_badge.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_coverage_badge.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_unittestssummary.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_unittestscoverage.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_rtl_scurve_status.json"
gh gist edit <GIST_ID> --delete-file "DELETEME_polaris_rtl_scurve_ratio_geomean.json"
```

### Automated Cleanup Script

Create a cleanup script to automatically remove DELETEME_ files:

```bash
#!/bin/bash
# cleanup_delme_files.sh

GIST_ID="your-gist-id-here"
GIST_TOKEN="your-github-token-here"

# List of DELETEME_ files to clean up
DELETEME_FILES=(
    "DELETEME_polaris_mypy_status.json"
    "DELETEME_polaris_spdx_status.json"
    "DELETEME_polaris_tests_badge.json"
    "DELETEME_polaris_coverage_badge.json"
    "DELETEME_polaris_unittestssummary.json"
    "DELETEME_polaris_unittestscoverage.json"
    "DELETEME_polaris_rtl_scurve_status.json"
    "DELETEME_polaris_rtl_scurve_ratio_geomean.json"
)

echo "Cleaning up DELETEME_ files from gist $GIST_ID..."

for file in "${DELETEME_FILES[@]}"; do
    echo "Attempting to delete: $file"
    gh gist edit "$GIST_ID" --delete-file "$file" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Deleted: $file"
    else
        echo "ℹ️  File not found or already deleted: $file"
    fi
done

echo "Cleanup completed."
```

### Python Cleanup Script

```python
#!/usr/bin/env python3
"""
Cleanup script for DELETEME_ prefixed files in GitHub gists.
"""

import os
import sys
import requests
import json
from typing import List, Dict, Any

def cleanup_delme_files(gist_id: str, token: str) -> None:
    """
    Clean up DELETEME_ prefixed files from a GitHub gist.
    
    Args:
        gist_id: The GitHub gist ID
        token: GitHub personal access token
    """
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get current gist data
    response = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers)
    if response.status_code != 200:
        print(f"❌ Error fetching gist: {response.status_code}")
        return
    
    gist_data = response.json()
    files = gist_data.get("files", {})
    
    # Find DELETEME_ files
    delme_files = [filename for filename in files.keys() if filename.startswith("DELETEME_")]
    
    if not delme_files:
        print("ℹ️  No DELETEME_ files found in gist")
        return
    
    print(f"Found {len(delme_files)} DELETEME_ files to clean up:")
    for filename in delme_files:
        print(f"  - {filename}")
    
    # Prepare update data (remove DELETEME_ files)
    update_data = {
        "files": {}
    }
    
    for filename in delme_files:
        update_data["files"][filename] = None  # Setting to None deletes the file
    
    # Update gist to remove files
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}",
        headers=headers,
        data=json.dumps(update_data)
    )
    
    if response.status_code == 200:
        print(f"✅ Successfully cleaned up {len(delme_files)} DELETEME_ files")
    else:
        print(f"❌ Error updating gist: {response.status_code}")
        print(f"Response: {response.text}")

def main():
    """Main function."""
    gist_id = os.getenv("GIST_ID")
    token = os.getenv("GIST_TOKEN")
    
    if not gist_id:
        print("❌ Error: GIST_ID environment variable not set")
        sys.exit(1)
    
    if not token:
        print("❌ Error: GIST_TOKEN environment variable not set")
        sys.exit(1)
    
    cleanup_delme_files(gist_id, token)

if __name__ == "__main__":
    main()
```

## Scheduled Cleanup

### GitHub Actions Workflow

Create a scheduled workflow to automatically clean up DELETEME_ files:

```yaml
name: Cleanup DELETEME Files

on:
  schedule:
    # Run daily at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - name: Cleanup DELETEME files
        env:
          GIST_ID: ${{ secrets.GIST_ID }}
          GIST_TOKEN: ${{ secrets.GIST_TOKEN }}
        run: |
          # Use the Python cleanup script
          python3 tools/ci/cleanup_delme_files.py
```

### Cron Job (Linux/macOS)

Add to crontab for daily cleanup:

```bash
# Add to crontab (crontab -e)
# Clean up DELETEME files daily at 2 AM
0 2 * * * /path/to/cleanup_delme_files.sh
```

## Best Practices

### 1. Regular Cleanup Schedule

- **Daily**: For active development environments
- **Weekly**: For less active projects
- **Before releases**: Clean up before major releases

### 2. Backup Important Data

Before cleanup, ensure any important data from DELETEME_ files is preserved:
- Check if any DELETEME_ files contain important test results
- Archive critical data before deletion

### 3. Monitoring

Monitor the cleanup process:
- Log cleanup activities
- Set up alerts for cleanup failures
- Track the number of files cleaned up

### 4. Testing

Test cleanup scripts in a safe environment:
- Use a test gist first
- Verify the script only deletes DELETEME_ files
- Test with various file scenarios

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure the GitHub token has gist write permissions
2. **File Not Found**: DELETEME_ files may have already been cleaned up
3. **Rate Limiting**: GitHub API rate limits may affect cleanup operations
4. **Network Issues**: Ensure stable internet connection for API calls

### Debug Mode

Enable debug logging in cleanup scripts:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Verification

After cleanup, verify files were removed:

```bash
# Check gist contents
gh gist view <GIST_ID>

# Or use GitHub web interface
# Navigate to the gist and verify DELETEME_ files are gone
```

## Security Considerations

- **Token Security**: Store GitHub tokens securely (use GitHub Secrets for Actions)
- **Access Control**: Limit cleanup scripts to authorized users
- **Audit Trail**: Log all cleanup activities for accountability
- **Backup**: Consider backing up important data before cleanup

## Related Documentation

- [makegist.py Documentation](README_makegist.md)
- [GitHub Actions Badge Generation](README_dynamic_badges.md)
- [GitHub Gist API Documentation](https://docs.github.com/en/rest/gists)

## License

SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC  
SPDX-License-Identifier: Apache-2.0
