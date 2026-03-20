#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# This script is used to download the models from the LFCache server to the local machine.
# Supports both Linux and macOS with automatic dependency checking and helpful error messages.
#
# Dependencies:
#   - wget (required, automatically checked with platform-specific installation help)
#   - tar (required for --extract option, automatically checked)
#   - check_behind_tailscale.sh (must be in same directory, used for Tailscale connectivity)
#
# Network Requirements:
#   - When not running in CI mode, the script first checks if the server is directly accessible
#   - If server is not accessible, Tailscale connection is required
#   - Uses check_behind_tailscale.sh script for Tailscale connectivity detection
#   - Provides installation and activation instructions if Tailscale is not available
#
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
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    CI=true
else
    CI=false
fi

# Detect OS type once for use throughout the script
OS_TYPE="unknown"
case "$(uname -s)" in
    Darwin*) OS_TYPE="macos" ;;
    Linux*)  OS_TYPE="linux" ;;
esac

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
    echo "Requirements:"
    echo "  - wget (required for all downloads, automatically checked)"
    echo "  - tar (required for --extract option, automatically checked)"
    echo "  - Network access to LFC server (direct or via Tailscale when not in CI mode)"
    echo "  - check_behind_tailscale.sh (for Tailscale connectivity check, must be in same directory)"
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
            if [[ $# -lt 2 ]] || [[ -z "${2:-}" ]] || ! [[ "${2:-}" =~ ^(dir|file)$ ]]; then
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

# Check if required arguments are provided
if [[ -z "$SERVER_PATH" ]]; then
    echo "Error: Missing required arguments" >&2
    show_usage
    exit 1
fi

# Validate SERVER_PATH for security (prevent path traversal)
if [[ "$SERVER_PATH" =~ (^|/)\.\.(/|$) ]]; then
    echo "Error: SERVER_PATH cannot contain '..' (path traversal not allowed)" >&2
    exit 1
fi

# Strip leading slash to prevent double-slash in URLs
SERVER_PATH="${SERVER_PATH#/}"

# Validate SERVER_PATH is not empty after stripping
if [[ -z "$SERVER_PATH" ]]; then
    echo "Error: SERVER_PATH cannot be empty or just '/'" >&2
    show_usage
    exit 1
fi

# Set LOCAL_PATH default from normalized SERVER_PATH
# When defaulted, LOCAL_PATH uses the normalized SERVER_PATH (always relative and safe)
# When explicitly provided, LOCAL_PATH is kept as-is (respects user intent, including absolute paths)
if [[ -z "$LOCAL_PATH" ]]; then
    # Default LOCAL_PATH from normalized SERVER_PATH (already stripped of leading slash)
    LOCAL_PATH="$SERVER_PATH"
fi

# Validate LOCAL_PATH doesn't contain path traversal sequences
if [[ "$LOCAL_PATH" =~ (^|/)\.\.(/|$) ]]; then
    echo "Error: LOCAL_PATH cannot contain '..' (path traversal not allowed)" >&2
    exit 1
fi

# Helper function to check if a filename has a real extension
has_real_extension() {
    local basename="$1"
    if [[ "$basename" != *.* ]]; then
        return 1  # No dot at all
    fi

    # Check for hidden files without real extensions (e.g., .hiddenfile, .bashrc)
    # These start with a dot and have no other dots
    if [[ "$basename" =~ ^\.[^.]+$ ]]; then
        return 1  # Hidden file without real extension
    fi

    local extension="${basename##*.}"
    # Check if extension is non-empty and not the whole basename
    if [[ -n "$extension" && "$extension" != "$basename" ]]; then
        return 0  # Has real extension
    fi
    return 1
}

# Auto-detect type and validate path (extract basename/dirname once)
local_basename=$(basename "$LOCAL_PATH")
local_dirname=$(dirname "$LOCAL_PATH")

if [[ "$TYPE" == "dir" && "$TYPE_EXPLICITLY_SET" == false ]]; then
    # If dirname is "." (no directory component) and has a real extension
    if [[ "$local_dirname" == "." ]] && has_real_extension "$local_basename"; then
        TYPE="file"
        if [[ "$VERBOSE" == true ]]; then
            echo "Auto-detected file type based on local path: $LOCAL_PATH"
        fi
    fi
fi

# Issue warning if local_path has directory component and extension but type is dir
if [[ "$TYPE" == "dir" && "$local_dirname" != "." ]] && has_real_extension "$local_basename"; then
    echo "Warning: Local path '$LOCAL_PATH' appears to be a file path (has directory and extension) but type is set to 'dir'. Consider using --type file." >&2
fi

# Auto-calculate cut-dir based on server_path
# Count the number of directory components in server_path
# Remove trailing slash if present, then count slashes and add 2
SERVER_PATH_CLEAN="${SERVER_PATH%/}"
if [[ -n "$SERVER_PATH_CLEAN" ]]; then
    # Count slashes using bash string manipulation (more efficient than tr+wc)
    slash_only="${SERVER_PATH_CLEAN//[^\/]/}"
    CUT_DIR=$((${#slash_only} + 2))
else
    # If server_path is empty or just "/", use 1
    CUT_DIR=1
fi

# Validate --extract option
if [[ "$EXTRACT" == true ]]; then
    if [[ "$LOCAL_PATH" != *.tar.gz ]]; then
        echo "Error: --extract can only be used with .tar.gz files. Local path: $LOCAL_PATH" >&2
        show_usage
        exit 1
    fi
    if [[ "$TYPE" != "file" ]]; then
        echo "Error: --extract can only be used with --type file (or auto-detected file type)" >&2
        show_usage
        exit 1
    fi
    # Check tar is available when extraction is requested
    if ! command -v tar >/dev/null 2>&1; then
        echo "Error: tar is not installed on this system." >&2
        echo "tar is required for the --extract option." >&2
        echo >&2
        if [[ "$OS_TYPE" == "macos" ]]; then
            echo "tar should be pre-installed on macOS. If missing, reinstall Xcode Command Line Tools:" >&2
            echo "    xcode-select --install" >&2
        elif [[ "$OS_TYPE" == "linux" ]]; then
            echo "To install tar on Linux:" >&2
            echo "  Ubuntu/Debian: sudo apt-get install tar" >&2
            echo "  RHEL/CentOS: sudo yum install tar" >&2
            echo "  Arch: sudo pacman -S tar" >&2
        fi
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
if [[ "$VERBOSE" == true ]]; then
    WGET_VERBOSE_FLAG="-v"
    echo "Verbose mode enabled"
    echo "Server URL: $SERVER_URL"
    echo "Local path: $LOCAL_PATH"
else
    WGET_VERBOSE_FLAG="-nv"
fi

# Function to show wget installation help
show_wget_installation_help() {
    local os_type="$1"
    echo "Error: wget is not installed on this system." >&2
    echo >&2
    echo "wget is required for downloading files from the LFC server." >&2
    echo >&2

    if [[ "$os_type" == "macos" ]]; then
        if command -v brew >/dev/null 2>&1; then
            echo "To install wget using Homebrew, run:" >&2
            echo "    brew install wget" >&2
        else
            echo "To install wget on macOS:" >&2
            echo "1. Install Homebrew:" >&2
            echo "    /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
            echo "2. Then install wget:" >&2
            echo "    brew install wget" >&2
            echo >&2
            echo "Alternatively:" >&2
            echo "- MacPorts: sudo port install wget" >&2
            echo "- Direct download: https://www.gnu.org/software/wget/" >&2
        fi
    elif [[ "$os_type" == "linux" ]]; then
        echo "To install wget on Linux, use your package manager:" >&2
        echo >&2
        echo "Ubuntu/Debian:" >&2
        echo "    sudo apt-get update && sudo apt-get install -y wget" >&2
        echo >&2
        echo "RHEL/CentOS/Fedora:" >&2
        echo "    sudo yum install -y wget" >&2
        echo "    # or on newer versions:" >&2
        echo "    sudo dnf install -y wget" >&2
        echo >&2
        echo "Arch Linux:" >&2
        echo "    sudo pacman -S wget" >&2
        echo >&2
        echo "Alpine:" >&2
        echo "    apk add wget" >&2
    else
        echo "Please install wget using your system's package manager." >&2
    fi

    echo >&2
    echo "After installing wget, retry running this script." >&2
}

# Function to show Tailscale installation instructions
show_tailscale_installation_help() {
    local os_type="$1"
    echo "Error: Tailscale is not installed on this system." >&2
    echo >&2
    echo "Tailscale is required to access the LFC server when not running in CI mode." >&2
    echo >&2

    if [[ "$os_type" == "macos" ]]; then
        if command -v brew >/dev/null 2>&1; then
            echo "To install Tailscale using Homebrew, run:" >&2
            echo "    brew install --cask tailscale" >&2
        else
            echo "To install Tailscale on macOS:" >&2
            echo "1. Download from: https://tailscale.com/download/mac" >&2
            echo "2. Or install Homebrew first and use: brew install --cask tailscale" >&2
        fi
    else
        echo "To install Tailscale on Linux:" >&2
        echo "1. Visit: https://tailscale.com/download/linux" >&2
        echo "2. Or use your package manager (e.g., apt, yum, pacman)" >&2
        echo "3. For Ubuntu/Debian: curl -fsSL https://tailscale.com/install.sh | sh" >&2
    fi

    echo >&2
    echo "After installation, you need to authenticate and connect to the Tailnet." >&2
}

# Function to show Tailscale activation instructions
show_tailscale_activation_help() {
    echo "Error: Tailscale is installed but not active/connected." >&2
    echo >&2
    echo "To activate Tailscale:" >&2
    echo "1. Start Tailscale:" >&2
    echo "   sudo tailscale up" >&2
    echo >&2
    echo "2. Follow the authentication link that appears" >&2
    echo "3. Complete the login process in your web browser" >&2
    echo >&2
    echo "You can check Tailscale status with:" >&2
    echo "   tailscale status" >&2
    echo >&2
    echo "After connecting to Tailscale, retry running this script." >&2
}

# Check for required dependencies (wget is required on all platforms)
if ! command -v wget >/dev/null 2>&1; then
    show_wget_installation_help "$OS_TYPE"
    exit 1
fi

if [[ "$VERBOSE" == true ]]; then
    echo "Found wget at $(command -v wget)"
fi

# Create local directory if it doesn't exist
if [[ "$TYPE" == "file" ]]; then
    # For file downloads, create the parent directory of the local path
    LOCAL_DIR=$(dirname "$LOCAL_PATH")
    if [[ "$VERBOSE" == true ]]; then
        echo "Creating directory: $LOCAL_DIR"
    fi
    mkdir -p "$LOCAL_DIR" || {
        echo "Error: Failed to create download directory: $LOCAL_DIR" >&2
        echo "Check permissions and disk space." >&2
        exit 1
    }
else
    # For directory downloads, create the full local path
    if [[ "$VERBOSE" == true ]]; then
        echo "Creating directory: $LOCAL_PATH"
    fi
    mkdir -p "$LOCAL_PATH" || {
        echo "Error: Failed to create download directory: $LOCAL_PATH" >&2
        echo "Check permissions and disk space." >&2
        exit 1
    }
fi

# Network connectivity check (only when not in CI mode)
if [[ "$CI" != true ]]; then
    if [[ "$VERBOSE" == true ]]; then
        echo "Checking connectivity to LFC server..."
    fi

    # Extract hostname from SERVER_BASE_URL
    SERVER_HOST="${SERVER_BASE_URL#http*://}"
    SERVER_HOST="${SERVER_HOST%%/*}"

    # Validate hostname extraction succeeded
    if [[ -z "$SERVER_HOST" ]]; then
        echo "Error: Failed to extract hostname from SERVER_BASE_URL: $SERVER_BASE_URL" >&2
        exit 1
    fi

    # Perform connectivity diagnostics - check both ICMP and HTTP
    # This provides better user feedback than optimizing for speed
    SERVER_ACCESSIBLE=false
    PING_SUCCESS=false
    HTTP_SUCCESS=false

    if [[ "$VERBOSE" == true ]]; then
        echo "Running connectivity diagnostics..."
    fi

    # Step 1: Check ICMP ping (network layer reachability)
    # Platform-specific ping flags
    if [[ "$OS_TYPE" == "macos" ]]; then
        # macOS: -W in milliseconds, -o exits after one reply
        PING_RESULT=$(ping -c 1 -W 2000 -o "$SERVER_HOST" >/dev/null 2>&1 && echo "success" || echo "fail")
    else
        # Linux: -W in seconds
        PING_RESULT=$(ping -c 1 -W 2 "$SERVER_HOST" >/dev/null 2>&1 && echo "success" || echo "fail")
    fi

    if [[ "$PING_RESULT" == "success" ]]; then
        PING_SUCCESS=true
        if [[ "$VERBOSE" == true ]]; then
            echo "  [PING] Network layer reachable (ICMP succeeded)"
        fi
    else
        if [[ "$VERBOSE" == true ]]; then
            echo "  [PING] Network layer unreachable or ICMP blocked"
        fi
    fi

    # Step 2: Check HTTP connectivity (what we actually need for downloads)
    # Note: wget is guaranteed to be available at this point (checked earlier)
    if [[ "$VERBOSE" == true ]]; then
        echo "  [HTTP] Testing HTTP connectivity..."
    fi

    # Capture wget exit status without triggering set -e
    if wget --spider --timeout=3 --tries=1 "$SERVER_BASE_URL" >/dev/null 2>&1; then
        WGET_STATUS=0
    else
        WGET_STATUS=$?
    fi

    # Exit codes: 0 = success, 8 = HTTP error (4xx/5xx) but server reachable
    if [[ $WGET_STATUS -eq 0 || $WGET_STATUS -eq 8 ]]; then
        HTTP_SUCCESS=true
        SERVER_ACCESSIBLE=true
        if [[ "$VERBOSE" == true ]]; then
            echo "  [HTTP] Application layer accessible (wget exit code: $WGET_STATUS)"
        fi
    else
        if [[ "$VERBOSE" == true ]]; then
            echo "  [HTTP] Application layer NOT accessible (wget exit code: $WGET_STATUS)"
        fi
    fi

    # Provide diagnostic summary and check Tailscale if needed
    if [[ "$SERVER_ACCESSIBLE" == true ]]; then
        if [[ "$VERBOSE" == true ]]; then
            echo "Connectivity check: Server is accessible via HTTP"
        fi
    else
        # Server not accessible - provide diagnostic info and check Tailscale
        if [[ "$PING_SUCCESS" == true && "$HTTP_SUCCESS" == false ]]; then
            echo "Warning: Server responds to ping but HTTP is not accessible."
            echo "This typically means HTTP traffic is blocked by firewall or routing policy."
        elif [[ "$VERBOSE" == true ]]; then
            echo "Server not accessible: Both ping and HTTP checks failed."
        fi

        echo "Checking Tailscale connectivity..."

        # Get the directory of the current script to find the tailscale checker
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        TAILSCALE_CHECKER="$SCRIPT_DIR/check_behind_tailscale.sh"

        # Check if the tailscale checker script exists
        if [[ ! -f "$TAILSCALE_CHECKER" ]]; then
            echo "Error: Cannot find Tailscale checker script at $TAILSCALE_CHECKER" >&2
            echo "Please ensure check_behind_tailscale.sh is in the same directory as this script." >&2
            exit 1
        fi

        # Run the tailscale checker script once, capturing both output and exit status
        # Use if-then-else to capture failures without triggering set -e
        if TAILSCALE_CHECKER_OUTPUT="$("$TAILSCALE_CHECKER" 2>&1)"; then
            TAILSCALE_CHECKER_STATUS=0
        else
            TAILSCALE_CHECKER_STATUS=$?
        fi

        if [[ $TAILSCALE_CHECKER_STATUS -eq 0 ]]; then
            # Tailscale check succeeded
            if [[ "$VERBOSE" == true ]]; then
                echo "Tailscale connectivity confirmed"
            fi
        else
            # Tailscale check failed
            if [[ "$VERBOSE" == true && -n "$TAILSCALE_CHECKER_OUTPUT" ]]; then
                echo "Tailscale checker error output:" >&2
                echo "$TAILSCALE_CHECKER_OUTPUT" >&2
            fi

            # Determine if it's installation or activation issue
            if ! command -v tailscale >/dev/null 2>&1; then
                # Tailscale not installed
                show_tailscale_installation_help "$OS_TYPE"
            else
                # Tailscale installed but not connected
                show_tailscale_activation_help
            fi
            exit 1
        fi
    fi
fi

# Download models from LFCache server
echo "Downloading from $SERVER_URL to $LOCAL_PATH..."
if [[ "$DRYRUN" == true ]]; then
    echo "Dry run mode enabled"
    if [[ "$TYPE" == "file" ]]; then
        echo "mkdir -p \"$(dirname "$LOCAL_PATH")\""
        echo "wget \"$WGET_VERBOSE_FLAG\" --timeout=30 --read-timeout=60 -O \"$LOCAL_PATH\" \"$SERVER_URL\""
        if [[ "$EXTRACT" == true ]]; then
            echo "tar -xzf \"$LOCAL_PATH\" -C \"$(dirname "$LOCAL_PATH")\""
            echo "# Would report number of non-directory files extracted"
            echo "rm \"$LOCAL_PATH\""
        fi
    else
        echo "mkdir -p \"$LOCAL_PATH\""
        echo "wget -np -nH --mirror \"$WGET_VERBOSE_FLAG\" --timeout=30 --read-timeout=60 --cut-dir \"$CUT_DIR\" -P \"$LOCAL_PATH\" \"$SERVER_URL\""
    fi
    exit 0
fi

if [[ "$TYPE" == "file" ]]; then
    wget "$WGET_VERBOSE_FLAG" --timeout=30 --read-timeout=60 -O "$LOCAL_PATH" "$SERVER_URL" || {
        echo "Error: Failed to download from $SERVER_URL" >&2
        echo "Check network connectivity and server availability." >&2
        exit 1
    }

    # Extract tar.gz file if --extract option is specified
    if [[ "$EXTRACT" == true ]]; then
        if [[ "$VERBOSE" == true ]]; then
            echo "Extracting $LOCAL_PATH..."
        fi

        # Count non-directory files in the archive before extraction
        # Use if-then-else to prevent set -e from exiting if tar listing fails
        # grep uses || true to prevent pipefail from exiting when no non-directory files exist
        # xargs trims whitespace from wc output
        if FILE_COUNT=$(tar -tzf "$LOCAL_PATH" 2>/dev/null | { grep -v '/$' || true; } | wc -l | xargs 2>/dev/null); then
            # tar listing succeeded - validate the count
            if [[ -z "$FILE_COUNT" || "$FILE_COUNT" -eq 0 ]]; then
                FILE_COUNT=0  # Valid empty archive or empty result
            fi
        else
            # tar listing failed (corrupt/partial archive)
            echo "Warning: Could not read archive for file count" >&2
            FILE_COUNT="unknown"
        fi

        tar -xzf "$LOCAL_PATH" -C "$(dirname "$LOCAL_PATH")" || {
            echo "Error: Failed to extract archive: $LOCAL_PATH" >&2
            echo "Archive may be corrupted or incomplete." >&2
            exit 1
        }

        # Report successful extraction
        if [[ "$FILE_COUNT" != "unknown" ]]; then
            echo "Successfully extracted $FILE_COUNT non-directory files from $LOCAL_PATH"
        else
            echo "Successfully extracted archive from $LOCAL_PATH"
        fi

        if [[ "$VERBOSE" == true ]]; then
            echo "Removing archive file $LOCAL_PATH..."
        fi
        rm "$LOCAL_PATH" || true  # Allow failure - not critical
    fi
else
    wget -np -nH --mirror "$WGET_VERBOSE_FLAG" --timeout=30 --read-timeout=60 --cut-dir "$CUT_DIR" -P "$LOCAL_PATH" "$SERVER_URL" || {
        echo "Error: Failed to download directory from $SERVER_URL" >&2
        echo "Check network connectivity and server availability." >&2
        exit 1
    }
fi

# Clean up index.html files (only for directory downloads)
if [[ "$TYPE" == "dir" ]]; then
    if [[ "$VERBOSE" == true ]]; then
        echo "Cleaning up index.html files..."
    fi
    find "$LOCAL_PATH" -name 'index.html*' -type f -delete 2>/dev/null || true
fi

# List downloaded files (only for directory downloads)
if [[ "$TYPE" == "dir" && "$VERBOSE" == true ]]; then
    echo "Downloaded files:"
    if ! find "$LOCAL_PATH" -type f 2>/dev/null; then
        echo "Warning: Could not list files in $LOCAL_PATH" >&2
    fi
fi
