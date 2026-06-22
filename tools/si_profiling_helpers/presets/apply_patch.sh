#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Idempotently apply a local-only source patch to the tt-metal checkout before a capture.
# Run from the tt-metal checkout root (as the presets are). Re-running is safe: an
# already-applied patch is detected and skipped, so a preset can be invoked repeatedly.
# These patches are deliberately NOT upstreamed to tt-metal (per-capture local tweaks).
#
# Usage (from a preset, run from the tt-metal root):
#   source "$(dirname "$0")/../apply_patch.sh"
#   apply_patch_if_needed "$(dirname "$0")/../patches/<name>.patch"
apply_patch_if_needed() {
    local patchfile="$1"
    if [[ ! -f "$patchfile" ]]; then
        echo "ERROR: patch file not found: $patchfile" >&2
        exit 1
    fi
    # Already applied? Reverse-applies cleanly => the change is present; skip.
    if patch -p1 -R --dry-run -f <"$patchfile" >/dev/null 2>&1; then
        echo "apply_patch: already applied, skipping $(basename "$patchfile")"
        return 0
    fi
    # Not applied yet but applies cleanly forward => apply it.
    if patch -p1 --dry-run -f <"$patchfile" >/dev/null 2>&1; then
        patch -p1 -f <"$patchfile" >/dev/null
        echo "apply_patch: applied $(basename "$patchfile")"
        return 0
    fi
    # Neither: partial state or tt-metal drift — refuse rather than half-apply.
    echo "ERROR: $(basename "$patchfile") neither already-applied nor cleanly applicable." >&2
    echo "       The tt-metal source may have drifted, or it is partially patched." >&2
    echo "       Re-create the patch against this checkout, or fix the source by hand." >&2
    exit 1
}
