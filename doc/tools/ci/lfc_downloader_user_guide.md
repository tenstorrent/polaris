# LFC Downloader User Guide

## Overview

The `ci/lfc_downloader.sh` script is a utility for downloading models and files from the Large File Cache (LFCache) server to your local machine. It supports both directory (recursive) and single file downloads with various options for customization.

## Synopsis

```bash
./tools/ci/lfc_downloader.sh [-v|--verbose] [-n|--dryrun] [--type TYPE] [--extract] <server_path> [local_path]
```

## Parameters

### Required Parameters

- **`server_path`** - Path on the LFCache server relative to `simulators-ai-perf` (required)

### Optional Parameters

- **`local_path`** - Local path to download files to (optional, defaults to same as `server_path`)

### Options

| Option | Long Form | Description | Default |
|--------|-----------|-------------|---------|
| `-c` | `--ci` | Enable CI mode (uses internal cluster URL) | `false` |
| `-v` | `--verbose` | Enable verbose output with detailed logging | `false` |
| `-n` | `--dryrun` | Dry run mode - shows commands without executing | `false` |
| `-h` | `--help` | Show usage information and exit | - |
| | `--type TYPE` | Download type: `dir` (directory) or `file` (single file) | `dir` (auto-detected) |
| | `--extract` | Extract `.tar.gz` files after download and remove archive | `false` |

## Download Types

### Directory Download (`--type dir`)
- Downloads entire directory structure recursively
- Mirrors the remote directory hierarchy
- Automatically cleans up `index.html` files created by wget
- Lists all downloaded files upon completion (in verbose mode)

### File Download (`--type file`)
- Downloads a single file
- Creates parent directories as needed
- Supports extraction of `.tar.gz` archives with `--extract` option

### Auto-Detection
The script automatically detects whether you want to download a file or directory based on the local path:
- If the local path has no directory component and contains a file extension, it's treated as a file
- Otherwise, it defaults to directory download
- You can override auto-detection using `--type`

## Server Modes

### Standard Mode (Default)
- Uses external LFCache server: `http://aus2-lfcache.aus2.tenstorrent.com`
- Suitable for general development use

### CI Mode (`--ci`)
- Uses internal cluster URL: `http://large-file-cache.large-file-cache.svc.cluster.local`
- Designed for Continuous Integration environments

## Examples

### Basic Usage

```bash
# Download a directory to the same local path
./tools/ci/lfc_downloader.sh tests/models/

# Download a directory with verbose output
./tools/ci/lfc_downloader.sh -v tests/models/

# Download to a custom local path
./tools/ci/lfc_downloader.sh tests/models/ /local/custom/path/
```

### File Downloads

```bash
# Download a single file (auto-detected as file type)
./tools/ci/lfc_downloader.sh models/resnet50.onnx

# Explicitly specify file type
./tools/ci/lfc_downloader.sh --type file models/large_model.tar.gz ./downloads/

# Download and extract a tar.gz file
./tools/ci/lfc_downloader.sh --type file --extract models/dataset.tar.gz
```

### Advanced Usage

```bash
# Dry run to see what would be downloaded
./tools/ci/lfc_downloader.sh -n -v tests/models/

# CI mode with verbose output
./lfc_downloader.sh --ci -v tests/models/

# Download file to specific directory and extract
./tools/ci/lfc_downloader.sh -v --type file --extract models/weights.tar.gz ./model_weights/weights.tar.gz
```

## Tailscale VPN Requirements

### Automatic Tailscale Detection
In non-CI environments, the script automatically:
- Checks if Tailscale VPN is running and connected
- Verifies access to internal resources
- Provides clear error messages if Tailscale is not available
- Uses the companion script `tools/ci/check_behind_tailscale.sh` for detection

### Setting Up Tailscale
If you encounter Tailscale-related errors:

1. **Install Tailscale**: Follow instructions at https://tailscale.com/download
2. **Connect to your tailnet**: Run `sudo tailscale up`
3. **Verify connection**: Run `tailscale status` to check connectivity
4. **Test the script**: Re-run the ci/lfc_downloader.sh command

### CI Environment Bypass
- CI environments (when `GITHUB_ACTIONS=true`) automatically bypass Tailscale checks
- No additional configuration needed for CI/CD pipelines

## Features

### Automatic Directory Creation
The script automatically creates necessary local directories:
- For file downloads: creates parent directories
- For directory downloads: creates the full target path

### Archive Extraction
When using `--extract` with `.tar.gz` files:
- Extracts the archive to the parent directory
- Removes the original archive file after extraction
- Only works with `--type file` downloads

### Intelligent Path Handling
- Automatically calculates `wget` cut-directories based on server path depth
- Handles trailing slashes in paths
- Provides warnings for potentially incorrect type selections

### Error Handling
- Validates command-line arguments
- Checks for required dependencies
- Provides clear error messages for common issues
- Graceful handling of download failures

## Troubleshooting

### Common Issues

1. **"Error: --extract can only be used with .tar.gz files"**
   - The `--extract` option only works with files ending in `.tar.gz`
   - Ensure your file path has the correct extension

2. **"Error: --extract can only be used with --type file"**
   - Archive extraction only works with single file downloads
   - Use `--type file` when downloading archives for extraction

3. **Download fails with connection errors**
   - Check if you need to use `--ci` mode for internal network access
   - Verify the server path exists on the LFCache server

4. **Warning about file path with directory type**
   - The script detected you might want to download a file but defaulted to directory
   - Use `--type file` to override the auto-detection

6. **"Warning: Tailscale check script not found"**
   - The companion script `tools/ci/check_behind_tailscale.sh` is missing
   - The script will proceed with caution but may fail if not on Tailscale
   - Ensure all repository files are present

### Best Practices

1. **Use verbose mode** (`-v`) when troubleshooting or when you want to see detailed progress
2. **Test with dry run** (`-n`) before downloading large datasets
3. **Use appropriate type** - let auto-detection work or explicitly specify `--type`
4. **Organize downloads** - use meaningful local paths to keep your workspace organized

## Dependencies

- `bash` (version 3.0 or higher)
- `wget` (for downloading files)
- `tar` (required only when using `--extract`)
- Basic Unix utilities: `mkdir`, `find`, `rm`

## Exit Codes

- `0` - Success
- `1` - Error (invalid arguments, download failure, etc.)

## Notes

- The script respects `.gitignore` patterns when present
- Large files are downloaded efficiently using `wget`'s mirroring capabilities
- The script is designed to be safe for CI/CD pipelines with non-interactive operation
