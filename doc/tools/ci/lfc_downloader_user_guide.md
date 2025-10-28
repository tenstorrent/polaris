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
- **Requires Tailscale connection** when not in CI mode

### CI Mode (Automatically Detected)
- Uses internal cluster URL: `http://large-file-cache.large-file-cache.svc.cluster.local`
- Automatically activated when `GITHUB_ACTIONS=true` environment variable is set
- Designed for Continuous Integration environments
- **No Tailscale requirement** in CI mode

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

# CI mode (automatically detected when GITHUB_ACTIONS=true)
# The following line sets GITHUB_ACTIONS manually for demonstration or local testing purposes.
# In actual GitHub Actions CI runs, this variable is set automatically.
GITHUB_ACTIONS=true ./tools/ci/lfc_downloader.sh -v tests/models/

# Download file to specific directory and extract
./tools/ci/lfc_downloader.sh -v --type file --extract models/weights.tar.gz ./model_weights/weights.tar.gz
```

## macOS Dependency Management

### Automatic wget Detection
On macOS systems, the script automatically checks for `wget` availability and provides helpful installation guidance if it's missing.

#### When wget is Available
```bash
$ ./tools/ci/lfc_downloader.sh -v tests/models/
Found wget at /opt/homebrew/bin/wget
Downloading from http://aus2-lfcache.aus2.tenstorrent.com/simulators-ai-perf/tests/models/ to tests/models/...
```

#### When wget is Missing (Homebrew Available)
```bash
$ ./tools/ci/lfc_downloader.sh tests/models/
Error: wget is not installed on this macOS system.

To install wget using Homebrew, run:
    brew install wget

Then retry running this script.
```

#### When wget is Missing (Homebrew Not Available)
```bash
$ ./tools/ci/lfc_downloader.sh tests/models/
Error: wget is not installed on this macOS system.

To install wget, you first need to install Homebrew (a package manager for macOS):

1. Install Homebrew by running:
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. After Homebrew installation completes, install wget:
    brew install wget

3. Then retry running this script.

Alternatively, you can install wget using other methods:
- MacPorts: sudo port install wget
- Direct download: https://www.gnu.org/software/wget/
```

### macOS Installation Options

#### Option 1: Homebrew (Recommended)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install wget
brew install wget
```

#### Option 2: MacPorts
```bash
# If you have MacPorts installed
sudo port install wget
```

#### Option 3: Manual Installation
Download wget directly from the [GNU wget website](https://www.gnu.org/software/wget/) and follow the installation instructions.

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

5. **"Error: Cannot find Tailscale checker script"**
   - The companion script `tools/ci/check_behind_tailscale.sh` is missing
   - Ensure all repository files are present and the script is in the same directory
   - This error occurs when the required dependency script is not found

### Best Practices

1. **Use verbose mode** (`-v`) when troubleshooting or when you want to see detailed progress
2. **Test with dry run** (`-n`) before downloading large datasets
3. **Use appropriate type** - let auto-detection work or explicitly specify `--type`
4. **Organize downloads** - use meaningful local paths to keep your workspace organized

## Dependencies

- `bash` (version 3.0 or higher)
- `wget` (for downloading files - automatically checked on macOS with installation help)
- `tar` (required only when using `--extract`)
- `check_behind_tailscale.sh` (must be in same directory as lfc_downloader.sh)
- **Tailscale VPN** (required when not in CI mode for accessing LFC server)
- Basic Unix utilities: `mkdir`, `find`, `rm`

## Exit Codes

- `0` - Success
- `1` - Error (invalid arguments, download failure, etc.)

## Notes

- The script respects `.gitignore` patterns when present
- Large files are downloaded efficiently using `wget`'s mirroring capabilities
- The script is designed to be safe for CI/CD pipelines with non-interactive operation
