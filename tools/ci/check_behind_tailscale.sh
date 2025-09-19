#!/bin/bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Script to check if running behind Tailscale
# Returns 0 if behind Tailscale, 1 otherwise
# Compatible with macOS and Linux

# Detect operating system
detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*)  echo "linux" ;;
        *)       echo "unknown" ;;
    esac
}

# Function to check if Tailscale is running
check_tailscale() {
    local os_type=$(detect_os)
    
    # Method 1: Check if tailscaled process is running
    if pgrep -x "tailscaled" > /dev/null 2>&1; then
        return 0
    fi
    
    # Method 2: Check if tailscale command is available and status shows connected
    if command -v tailscale > /dev/null 2>&1; then
        # Check if tailscale status shows we're connected
        if tailscale status > /dev/null 2>&1; then
            # Check if we have an active tailnet connection
            tailscale_status=$(tailscale status --json 2>/dev/null)
            if [ $? -eq 0 ] && echo "$tailscale_status" | grep -q '"BackendState":"Running"'; then
                return 0
            fi
        fi
    fi
    
    # Method 3: Check for tailscale network interface (OS-specific)
    if [ "$os_type" = "macos" ]; then
        # macOS uses ifconfig
        if ifconfig 2>/dev/null | grep -q "utun.*tailscale\|tailscale"; then
            return 0
        fi
    else
        # Linux uses ip command
        if ip addr show 2>/dev/null | grep -q "tailscale"; then
            return 0
        fi
    fi
    
    # Method 4: Check for typical tailscale IP range (100.x.x.x) - OS-specific
    if [ "$os_type" = "macos" ]; then
        # macOS uses ifconfig
        if ifconfig 2>/dev/null | grep -q "inet 100\."; then
            return 0
        fi
    else
        # Linux uses ip command
        if ip addr show 2>/dev/null | grep -q "inet 100\."; then
            return 0
        fi
    fi
    
    # Method 5: Check if we can resolve magic DNS names (tailscale feature)
    if nslookup _magicDNS 2>/dev/null | grep -q "100\.100\.100\.100"; then
        return 0
    fi
    
    # Method 6: Check for Tailscale's specific network interface patterns
    if [ "$os_type" = "macos" ]; then
        # On macOS, Tailscale typically uses utun interfaces
        if ifconfig 2>/dev/null | grep -A5 "^utun" | grep -q "inet 100\."; then
            return 0
        fi
    fi
    
    return 1
}

# Main execution
check_tailscale
exit_code=$?

# Optional: Print status message (comment out if you want silent operation)
if [ $exit_code -eq 0 ]; then
    echo "Running behind Tailscale" >&2
else
    echo "Not running behind Tailscale" >&2
fi

exit $exit_code
