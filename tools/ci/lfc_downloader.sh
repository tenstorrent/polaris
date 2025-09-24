#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# This script is used to download the models from the LFCache server to the local machine.
# Supports both Linux and macOS with automatic dependency checking and helpful error messages.
# Usage: lfc_downloader.sh [-v|--verbose] [-n|--dryrun] [--type TYPE] [--extract] <server_path> [local_path]
#   CI mode is automatically detected from GITHUB_ACTIONS environment variable
#   -v, --verbose    Enable verbose output (optional, default is false)
#   -n, --dryrun     Dry run mode (optional, default is false)
#   --type TYPE      Download type: 'dir' for directory (recursive) or 'file' for single file (optional, default is dir)
#   --extract        Extract .tar.gz files after download and remove archive (optional, only valid for .tar.gz files)
#   server_path      Path on the LFCache server relative to simulators-ai-perf (required)
#   local_path       Local path to download models to (optional, default is the same as server_path)
#
# macOS Support:
#   - Automatically detects wget availability
#   - Provides installation instructions for Homebrew/MacPorts if wget is missing
#   - Works in both CI and development environments

# Detect CI mode from environment
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    CI=true
else
    CI=false
fi

# Initialize variables
VERBOSE=false
DRYRUN=false
EXTRACT=false
TYPE="dir"
TYPE_EXPLICITLY_SET=false
CUT_DIR=""
SERVER_PATH=""
LOCAL_PATH=""

# Function to show usage
show_usage() {
    echo "Usage: $0 [-v|--verbose] [-n|--dryrun] [--type TYPE] [--extract] <server_path> [local_path]"
    echo "  CI mode is automatically detected from GITHUB_ACTIONS environment variable"
    echo "  -v, --verbose    Enable verbose output (optional, default is false)"
    echo "  -n, --dryrun     Dry run mode (optional, default is false)"
    echo "  --type TYPE      Download type: 'dir' for directory (recursive) or 'file' for single file (optional, default is dir)"
    echo "  --extract        Extract .tar.gz files after download and remove archive (optional, only valid for .tar.gz files)"
    echo "  server_path      Path on the LFCache server relative to simulators-ai-perf (required)"
    echo "  local_path       Local path to download models to (optional, default is the same as server_path)"
    echo ""
    echo "Examples:"
    echo "  $0 tests/models/                    # Downloads to tests/models/ (uses default local path)"
    echo "  $0 -v tests/models/ custom/path/    # Downloads to custom/path/ with verbose output"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--dryrun)
            DRYRUN=true
            shift
            ;;
        --type)
            if [[ -z "$2" ]] || ! [[ "$2" =~ ^(dir|file)$ ]]; then
                echo "Error: --type requires 'dir' or 'file' as argument" >&2
                show_usage
                exit 1
            fi
            TYPE="$2"
            TYPE_EXPLICITLY_SET=true
            shift 2
            ;;
        --extract)
            EXTRACT=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            show_usage
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "$SERVER_PATH" ]]; then
                SERVER_PATH="$1"
            elif [[ -z "$LOCAL_PATH" ]]; then
                LOCAL_PATH="$1"
            else
                echo "Error: Too many arguments" >&2
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$LOCAL_PATH" ]]; then
    LOCAL_PATH="$SERVER_PATH"
fi

# Auto-detect type if not explicitly set
if [[ "$TYPE" == "dir" && "$TYPE_EXPLICITLY_SET" == false ]]; then
    # Check if local path looks like a filename (no directory component and has extension)
    local_basename=$(basename "$LOCAL_PATH")
    local_dirname=$(dirname "$LOCAL_PATH")

    # If dirname is "." (no directory component) and basename has a non-empty extension
    if [[ "$local_dirname" == "." && "$local_basename" == *.* ]]; then
        # Extract extension and check if it's non-empty
        extension="${local_basename##*.}"
        # Skip if it's just a hidden file without real extension (e.g., .hiddenfile)
        # Allow hidden files with real extensions (e.g., .config.json)
        if [[ -n "$extension" && "$extension" != "$local_basename" && ! ("$local_basename" =~ "^\.[^.]+$" ) ]]; then
            TYPE="file"
            if [[ "$VERBOSE" == true ]]; then
                echo "Auto-detected file type based on local path: $LOCAL_PATH"
            fi
        fi
    fi
fi

# Issue warning if local_path has directory component and extension but type is dir
if [[ "$TYPE" == "dir" ]]; then
    local_basename=$(basename "$LOCAL_PATH")
    local_dirname=$(dirname "$LOCAL_PATH")

    # Check if has directory component and file extension
    if [[ "$local_dirname" != "." && "$local_basename" == *.* ]]; then
        # Extract extension and check if it's non-empty
        extension="${local_basename##*.}"
        # Skip warning for hidden files without real extensions
        if [[ -n "$extension" && "$extension" != "$local_basename" && ! ("$local_basename" =~ "^\.[^.]+$" ) ]]; then
            echo "Warning: Local path '$LOCAL_PATH' appears to be a file path (has directory and extension) but type is set to 'dir'. Consider using --type file." >&2
        fi
    fi
fi

# Auto-calculate cut-dir based on server_path
# Count the number of directory components in server_path
# Remove trailing slash if present, then count slashes and add 2
SERVER_PATH_CLEAN="${SERVER_PATH%/}"
if [[ -n "$SERVER_PATH_CLEAN" ]]; then
    # Count slashes in the path and add 2 (1 for simulators-ai-perf + 1 for the path itself)
    CUT_DIR=$(($(echo "$SERVER_PATH_CLEAN" | tr -cd '/' | wc -c) + 2))
else
    # If server_path is empty or just "/", use 1
    CUT_DIR=1
fi

# Check if required arguments are provided
if [[ -z "$SERVER_PATH" || -z "$LOCAL_PATH" ]]; then
    echo "Error: Missing required arguments" >&2
    show_usage
    exit 1
fi

# Validate --extract option
if [[ "$EXTRACT" == true ]]; then
    if [[ ! "$LOCAL_PATH" == *.tar.gz ]]; then
        echo "Error: --extract can only be used with .tar.gz files. Local path: $LOCAL_PATH" >&2
        show_usage
        exit 1
    fi
    if [[ "$TYPE" != "file" ]]; then
        echo "Error: --extract can only be used with --type file (or auto-detected file type)" >&2
        show_usage
        exit 1
    fi
fi

# Check if CI mode is enabled
if [[ "$CI" == true ]]; then
    SERVER_BASE_URL="http://large-file-cache.large-file-cache.svc.cluster.local"
else
    SERVER_BASE_URL="http://aus2-lfcache.aus2.tenstorrent.com"
fi
SERVER_URL="$SERVER_BASE_URL/simulators-ai-perf/$SERVER_PATH"

# Set wget verbosity based on verbose flag
WGET_VERBOSE_FLAG=""
if [[ "$VERBOSE" == true ]]; then
    WGET_VERBOSE_FLAG="-v"
    echo "Verbose mode enabled"
    echo "Server URL: $SERVER_URL"
    echo "Local path: $LOCAL_PATH"
else
    WGET_VERBOSE_FLAG="-nv"
fi

# Create local directory if it doesn't exist
if [[ "$TYPE" == "file" ]]; then
    # For file downloads, create the parent directory of the local path
    LOCAL_DIR=$(dirname "$LOCAL_PATH")
    if [[ "$VERBOSE" == true ]]; then
        echo "Creating directory: $LOCAL_DIR"
    fi
    mkdir -p "$LOCAL_DIR" || {
        echo "Error: Failed to create directory $LOCAL_DIR" >&2
        exit 1
    }
else
    # For directory downloads, create the full local path
    if [[ "$VERBOSE" == true ]]; then
        echo "Creating directory: $LOCAL_PATH"
    fi
    mkdir -p "$LOCAL_PATH" || {
        echo "Error: Failed to create directory $LOCAL_PATH" >&2
        exit 1
    }
fi

# Function to show macOS wget installation help
show_macos_wget_installation_help() {
    echo "Error: wget is not installed on this macOS system." >&2
    echo >&2
    
    if command -v brew >/dev/null 2>&1; then
        echo "To install wget using Homebrew, run:" >&2
        echo "    brew install wget" >&2
        echo >&2
        echo "Then retry running this script." >&2
    else
        echo "To install wget, you first need to install Homebrew (a package manager for macOS):" >&2
        echo >&2
        echo "1. Install Homebrew by running:" >&2
        echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
        echo >&2
        echo "2. After Homebrew installation completes, install wget:" >&2
        echo "    brew install wget" >&2
        echo >&2
        echo "3. Then retry running this script." >&2
        echo >&2
        echo "Alternatively, you can install wget using other methods:" >&2
        echo "- MacPorts: sudo port install wget" >&2
        echo "- Direct download: https://www.gnu.org/software/wget/" >&2
    fi
}

# Platform-specific dependency checks
if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v wget >/dev/null 2>&1; then
        show_macos_wget_installation_help
        exit 1
    elif [[ "$VERBOSE" == true ]]; then
        echo "Found wget at $(command -v wget)"
    fi
fi

# Download models from LFCache server
echo "Downloading from $SERVER_URL to $LOCAL_PATH..."
if [[ "$DRYRUN" == true ]]; then
    echo "Dry run mode enabled"
    if [[ "$TYPE" == "file" ]]; then
        echo "mkdir -p \"$(dirname "$LOCAL_PATH")\""
        echo "wget \"$WGET_VERBOSE_FLAG\" -P \"$(dirname "$LOCAL_PATH")\" \"$SERVER_URL\""
        if [[ "$EXTRACT" == true ]]; then
            echo "tar -xzf \"$LOCAL_PATH\" -C \"$(dirname "$LOCAL_PATH")\""
            echo "# Would report number of non-directory files extracted"
            echo "rm \"$LOCAL_PATH\""
        fi
    else
        echo "mkdir -p \"$LOCAL_PATH\""
        echo "wget -r -np -nH --mirror \"$WGET_VERBOSE_FLAG --cut-dir $CUT_DIR -P "$LOCAL_PATH" "$SERVER_URL""
    fi
    exit 0
fi

if [[ "$TYPE" == "file" ]]; then
    wget "$WGET_VERBOSE_FLAG" -P "$(dirname "$LOCAL_PATH")" "$SERVER_URL" || {
        echo "Error: File download failed" >&2
        exit 1
    }

    # Extract tar.gz file if --extract option is specified
    if [[ "$EXTRACT" == true ]]; then
        if [[ "$VERBOSE" == true ]]; then
            echo "Extracting $LOCAL_PATH..."
        fi

        # Count non-directory files in the archive before extraction
        FILE_COUNT=$(tar -tzf "$LOCAL_PATH" | grep -v '/$' | wc -l)

        tar -xzf "$LOCAL_PATH" -C "$(dirname "$LOCAL_PATH")" || {
            echo "Error: Failed to extract $LOCAL_PATH" >&2
            exit 1
        }

        # Report successful extraction with non-directory file count
        echo "Successfully extracted $FILE_COUNT non-directory files from $LOCAL_PATH"

        if [[ "$VERBOSE" == true ]]; then
            echo "Removing archive file $LOCAL_PATH..."
        fi
        rm "$LOCAL_PATH" || {
            echo "Warning: Failed to remove archive file $LOCAL_PATH" >&2
        }
    fi
else
    wget -r -np -nH --mirror "$WGET_VERBOSE_FLAG" --cut-dir "$CUT_DIR" -P "$LOCAL_PATH" "$SERVER_URL" || {
        echo "Error: Directory download failed" >&2
        exit 1
    }
fi

# Clean up index.html files (only for directory downloads)
if [[ "$TYPE" == "dir" ]]; then
    if [[ "$VERBOSE" == true ]]; then
        echo "Cleaning up index.html files..."
    fi
    find "$LOCAL_PATH" -name 'index.html*' -exec rm -f {} \; || { 
        echo "Warning: Failed to remove some index.html files" >&2
    }
fi

# List downloaded files (only for directory downloads)
if [[ "$TYPE" == "dir" ]]; then
    if [[ "$VERBOSE" == true ]]; then
        echo "Downloaded files:"
    fi
    find "$LOCAL_PATH"
fi
