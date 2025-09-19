# check_behind_tailscale.sh

A cross-platform bash script to detect if the system is running behind Tailscale VPN.

## Overview

This script provides a reliable way to programmatically determine if your system is connected to a Tailscale network. It uses multiple detection methods to ensure accuracy across different operating systems and Tailscale configurations.

## Compatibility

- ✅ **macOS** (Darwin)
- ✅ **Linux** (all distributions)
- ⚠️  **Other Unix-like systems** (may work with limited functionality)

## Installation

The script is located at `tools/ci/check_behind_tailscale.sh` and should already be executable. If not, make it executable:

```bash
chmod +x tools/ci/check_behind_tailscale.sh
```

## Usage

### Basic Usage

```bash
# Run the script
./tools/ci/check_behind_tailscale.sh

# Example output:
# Running behind Tailscale
# (or: Not running behind Tailscale)
```

### Silent Mode

To suppress output messages and only use exit codes:

```bash
# Comment out the echo statements in lines 48-52 of the script
# Or redirect stderr to /dev/null
./tools/ci/check_behind_tailscale.sh 2>/dev/null
```

### In Shell Scripts

```bash
#!/bin/bash

if ./tools/ci/check_behind_tailscale.sh 2>/dev/null; then
    echo "Tailscale is active - proceeding with secure operations"
    # Your Tailscale-dependent code here
else
    echo "Tailscale not detected - using alternative configuration"
    # Your fallback code here
fi
```

### Exit Codes

- **0**: System is running behind Tailscale (success)
- **1**: System is NOT running behind Tailscale (failure)

### In Conditional Logic

```bash
# Simple conditional
./tools/check_behind_tailscale.sh && echo "Connected!" || echo "Not connected"

# Store exit code
./tools/check_behind_tailscale.sh
tailscale_status=$?
if [ $tailscale_status -eq 0 ]; then
    echo "Tailscale detected"
fi

# One-liner with output suppression
if ./tools/ci/check_behind_tailscale.sh >/dev/null 2>&1; then
    echo "Behind Tailscale"
fi
```

## Detection Methods

The script uses multiple detection methods to ensure reliability:

### 1. Process Detection
- Checks for running `tailscaled` daemon process
- Works on all supported platforms

### 2. CLI Status Check
- Uses `tailscale status --json` to verify active connection
- Looks for `"BackendState":"Running"` in JSON output
- Most reliable method when Tailscale CLI is available

### 3. Network Interface Detection
- **macOS**: Uses `ifconfig` to find Tailscale interfaces
- **Linux**: Uses `ip addr show` to find interfaces named with "tailscale"

### 4. IP Range Detection
- Looks for IP addresses in the 100.x.x.x range (Tailscale's CGNAT range)
- **macOS**: Checks via `ifconfig`
- **Linux**: Checks via `ip addr show`

### 5. Magic DNS Detection
- Attempts to resolve `_magicDNS` and checks for Tailscale's magic DNS server (100.100.100.100)
- Works when Tailscale's DNS features are enabled

### 6. macOS-Specific utun Interface Check
- On macOS, specifically looks for `utun` interfaces with Tailscale IP ranges
- Addresses macOS-specific Tailscale networking implementation

## Examples

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Check Tailscale Status
  run: |
    if ./tools/ci/check_behind_tailscale.sh; then
      echo "tailscale_active=true" >> $GITHUB_ENV
    else
      echo "tailscale_active=false" >> $GITHUB_ENV
    fi

- name: Run Tailscale-dependent tests
  if: env.tailscale_active == 'true'
  run: |
    # Your Tailscale-dependent operations
```

### Docker Container Check

```bash
#!/bin/bash
# In a Docker container startup script

if ./tools/ci/check_behind_tailscale.sh 2>/dev/null; then
    echo "Container has Tailscale access"
    export TAILSCALE_ENABLED=true
else
    echo "Container running without Tailscale"
    export TAILSCALE_ENABLED=false
fi
```

### Development Environment Setup

```bash
#!/bin/bash
# Development environment setup

echo "Setting up development environment..."

if ./tools/ci/check_behind_tailscale.sh; then
    echo "✅ Tailscale detected - enabling secure development features"
    export DEV_MODE="secure"
    export API_ENDPOINT="https://internal.tailnet.example.com"
else
    echo "⚠️  Tailscale not detected - using public endpoints"
    export DEV_MODE="public"
    export API_ENDPOINT="https://api.example.com"
fi
```

### Monitoring Script

```bash
#!/bin/bash
# System monitoring script

check_interval=60  # Check every 60 seconds

while true; do
    if ./tools/ci/check_behind_tailscale.sh 2>/dev/null; then
        echo "$(date): Tailscale connection active"
    else
        echo "$(date): WARNING - Tailscale connection lost!"
        # Send alert, restart service, etc.
    fi
    sleep $check_interval
done
```

## Troubleshooting

### Common Issues

1. **Script returns false positive**
   - Check if you have other VPN software using 100.x.x.x range
   - Verify Tailscale is actually connected: `tailscale status`

2. **Script returns false negative**
   - Ensure Tailscale daemon is running: `sudo tailscale up`
   - Check if Tailscale CLI is in PATH: `which tailscale`
   - Verify network interfaces: `ifconfig` (macOS) or `ip addr` (Linux)

3. **Permission errors**
   - Make sure script is executable: `chmod +x tools/ci/check_behind_tailscale.sh`
   - Some detection methods may require elevated privileges

### Debug Mode

For debugging, you can modify the script to show which detection method succeeded:

```bash
# Add debug output by modifying the script
# Replace each `return 0` with:
echo "DEBUG: Detection method X succeeded" >&2
return 0
```

### Manual Verification

You can manually verify Tailscale status:

```bash
# Check Tailscale status
tailscale status

# Check for Tailscale processes
ps aux | grep tailscale

# Check network interfaces (macOS)
ifconfig | grep -A5 utun

# Check network interfaces (Linux)
ip addr show | grep tailscale
```

## Security Considerations

- The script only reads system information and doesn't modify anything
- No sensitive information is exposed in the output
- All network checks use standard system commands
- The script doesn't require elevated privileges for basic functionality

## Contributing

To improve the script:

1. Test on your specific OS/distribution
2. Add additional detection methods if needed
3. Report issues with specific Tailscale configurations
4. Suggest improvements for edge cases

## Related Tools

- `tailscale status` - Official Tailscale status command
- `tailscale netcheck` - Network connectivity diagnostics
- `tailscale ping <node>` - Test connectivity to Tailscale nodes
