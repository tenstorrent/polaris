#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared preset helper: record HW identity / provenance for a capture run.
# Sourced by preset run-scripts (alongside check_arch.sh). Two functions:
#
#   hw_id_board        echo the precise board_type from tt-smi (e.g. "p100a"). No device open.
#                      Intended for the RUNID / output-dir name. Empty string if tt-smi unavailable.
#
#   hw_id_write <dir>  write provenance into the capture's output dir:
#                        <dir>/hw_id.json          - parsed essentials (machine-readable)
#                        <dir>/tt_smi_snapshot.json - full tt-smi -s telemetry (clock etc.; full provenance)
#                      Best-effort: any field that can't be obtained is "unknown"; never fails the run.
#
# Note: this is RECORDING only (no board-level enforcement). Arch-level enforcement stays in
# check_arch.sh::require_arch.

hw_id_board() {
    # board_type may carry a revision suffix containing whitespace (e.g. WH reports "n150 L").
    # Sanitize to a filename/shell-safe token (trim, then collapse internal whitespace to '_')
    # so it is safe inside RUNID / --output dir names. Raw value is preserved in tt_smi_snapshot.json.
    tt-smi -s 2>/dev/null \
        | sed -nE 's/.*"board_type" *: *"([^"]+)".*/\1/p' | head -1 \
        | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/_/g'
}

hw_id_write() {
    local dir="$1"
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "hw_id_write: output dir '$dir' not found — skipping hw_id capture" >&2
        return 0
    fi

    # Full tt-smi snapshot: carries clock + all telemetry regardless of our parsing below.
    tt-smi -s > "$dir/tt_smi_snapshot.json" 2>/dev/null || true

    local board aiclk arch grid arch_grid sha dirty
    board=$(hw_id_board)
    # best-effort AICLK (MHz) from the snapshot; field name unverified on HW -> full snapshot has it regardless.
    aiclk=$(sed -nE 's/.*"aiclk" *: *"? *([0-9.]+) *"?.*/\1/p' "$dir/tt_smi_snapshot.json" 2>/dev/null | head -1)

    # arch + USABLE compute grid from ttnn (opens the device briefly; device is free post-profiler-run).
    # ttnn/loguru spam stdout (and loguru uses '|' as a field separator), so tag the result line with a
    # sentinel + TAB delimiter and extract ONLY that line — never parse raw stdout directly.
    local hwid_out
    # Use 'python' (the activated venv interpreter) not 'python3' — the system python3
    # may lack ttnn, which would make arch/grid resolve to "unknown".
    hwid_out=$(python - <<'PY' 2>/dev/null
import ttnn
arch = ttnn.get_arch_name()
grid = "unknown"
try:
    d = ttnn.open_device(device_id=0)
    try:
        g = d.compute_with_storage_grid_size()
        grid = f"{g.x}x{g.y}"
    finally:
        ttnn.close_device(d)
except Exception:
    pass
print(f"__HWID__\t{arch}\t{grid}")
PY
)
    arch_grid=$(printf '%s\n' "$hwid_out" | grep '^__HWID__' | tail -1)
    arch=$(printf '%s' "$arch_grid" | cut -f2)
    grid=$(printf '%s' "$arch_grid" | cut -f3)
    [ -z "$arch" ] && arch="unknown"
    [ -z "$grid" ] && grid="unknown"

    local tt_home="${TT_METAL_HOME:-$(pwd)}"
    sha=$(git -C "$tt_home" rev-parse --short HEAD 2>/dev/null || echo unknown)
    local ctime; ctime=$(git -C "$tt_home" show -s --format=%cI HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git -C "$tt_home" status --porcelain 2>/dev/null)" ]; then dirty=true; else dirty=false; fi

    cat > "$dir/hw_id.json" <<EOF
{
  "arch": "${arch:-unknown}",
  "board_type": "${board:-unknown}",
  "compute_grid_size": "${grid:-unknown}",
  "aiclk_mhz": "${aiclk:-unknown}",
  "tt_metal_git": "${sha}",
  "tt_metal_commit_time": "${ctime}",
  "tt_metal_dirty": ${dirty},
  "run_utc": "$(date -u +%FT%TZ)"
}
EOF
    echo "hw_id: wrote ${dir}/hw_id.json (board=${board:-?} grid=${grid:-?} git=${sha} dirty=${dirty})"
}
