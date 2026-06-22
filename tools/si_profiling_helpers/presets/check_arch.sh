#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Guard sourced by preset run-scripts: abort unless the board's arch matches the
# preset's target. Usage (from a preset):  require_arch wormhole_b0   # or: blackhole
require_arch() {
    local want="$1" family
    case "$want" in
        wormhole*|wh)  family=wormhole  ;;
        blackhole*|bh) family=blackhole ;;
        *) echo "check_arch: unknown expected arch '$want'" >&2; exit 2 ;;
    esac
    if [[ -z "${IRD_ARCH_NAME:-}" ]]; then
        echo "ERROR: IRD_ARCH_NAME is unset — not on a hardware node (would run the Polaris shim). Aborting." >&2
        exit 1
    fi
    shopt -s nocasematch
    if [[ "$IRD_ARCH_NAME" != *"$family"* ]]; then
        echo "ERROR: this preset targets '$family' but IRD_ARCH_NAME='$IRD_ARCH_NAME' — run it on a $family board. Aborting." >&2
        shopt -u nocasematch; exit 1
    fi
    shopt -u nocasematch
}

# Sanity-check that a preset's CMD targets the correct source tree.
#   require_ref_cmd  "$CMD"  — a reference (refrun) capture must run the HW-native
#                              model (e.g. models/demos/...); it must NOT touch the
#                              Polaris 'workloads/' tree.
#   require_dual_cmd "$CMD"  — a dual-mode capture must run the migrated Polaris
#                              workload under 'workloads/ttnn/'.
require_ref_cmd() {
    case "$1" in
        *workloads/*)
            echo "ERROR: refrun CMD references 'workloads/' — a reference run must use the HW-native model (models/...), not a Polaris workload: $1" >&2
            exit 1 ;;
    esac
}
require_dual_cmd() {
    case "$1" in
        *workloads/ttnn/*) : ;;
        *)
            echo "ERROR: dual-mode CMD must run a Polaris workload under 'workloads/ttnn/': $1" >&2
            exit 1 ;;
    esac
}
