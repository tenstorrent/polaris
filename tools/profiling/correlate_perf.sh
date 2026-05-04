#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# correlate_perf.sh — compare Polaris simulation output against HW profiler CSV
# for the optimized sharded ViT (Wormhole n150 or Blackhole p100a).
#
# Usage:
#   tools/profiling/correlate_perf.sh <arch> <hw_csv> [--regen] [--polaris-csv <path>]
#
#   arch               wh (Wormhole n150) or bh (Blackhole p100a)
#   hw_csv             path to merged_ops.csv from the HW profiler run (lfc:// paths auto-fetched)
#   --regen            re-run polproj before comparing
#   --polaris-csv PATH override the derived Polaris opstats CSV path
#
# Outputs (workspace root, gitignored):
#   correlate_{arch}.out    text report
#   correlate_{arch}.err    stderr
#   correlate_{arch}.xlsx   3-sheet workbook (Summary / By Layer Type / By Layer Signature)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
usage() { sed -n '6,17p' "$0"; exit "${1:-0}"; }

[[ $# -lt 2 ]] && usage 2

ARCH="$1"; shift
HW_CSV="$1"; shift

REGEN=0
POLARIS_CSV_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regen) REGEN=1 ;;
        --polaris-csv) POLARIS_CSV_OVERRIDE="$2"; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Arch-specific config
# ---------------------------------------------------------------------------
case "$ARCH" in
    wh) DEVNAME="n150";  BS=8;  RUNCFG="config/runcfg_vitoptim_wh.yaml" ;;
    bh) DEVNAME="p100a"; BS=10; RUNCFG="config/runcfg_vitoptim_bh.yaml" ;;
    *)  echo "ERROR: arch must be 'wh' or 'bh', got '$ARCH'" >&2; usage 2 ;;
esac

POLARIS_CSV="${POLARIS_CSV_OVERRIDE:-"__runvitoptim_${ARCH}/runvitoptim_${ARCH}/STATS/${DEVNAME}-TTNN-vitoptim_${ARCH}_perf_report-vitoptim_${ARCH}_b16_perf_report-b${BS}-opstats.csv"}"
OUT_TXT="$REPO_ROOT/correlate_${ARCH}.out"
OUT_ERR="$REPO_ROOT/correlate_${ARCH}.err"
OUT_XLSX="$REPO_ROOT/correlate_${ARCH}.xlsx"
TAG="[correlate_perf ${ARCH}]"

# ---------------------------------------------------------------------------
# Conda activation
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniforge3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate polarisdev

# ---------------------------------------------------------------------------
# Optionally regenerate Polaris CSV
# ---------------------------------------------------------------------------
if [[ "$REGEN" -eq 1 || ! -f "$POLARIS_CSV" ]]; then
    echo "$TAG Running polproj with $RUNCFG ..."
    python polproj.py --config "$RUNCFG"
fi

# ---------------------------------------------------------------------------
# Resolve lfc:// paths
# ---------------------------------------------------------------------------
if [[ "$HW_CSV" == lfc://* ]]; then
    echo "$TAG Fetching $HW_CSV from LFC ..."
    HW_CSV="$(PYTHONPATH=. python -c "from ttsim.utils.lfc import resolve_lfc_path; print(resolve_lfc_path('$HW_CSV'))")"
fi

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -f "$POLARIS_CSV" ]]; then
    echo "ERROR: Polaris CSV not found: $POLARIS_CSV" >&2; exit 1
fi
if [[ ! -f "$HW_CSV" ]]; then
    echo "ERROR: HW profiler CSV not found: $HW_CSV" >&2; exit 1
fi

echo "$TAG Polaris CSV : $POLARIS_CSV"
echo "$TAG HW    CSV   : $HW_CSV"
echo "$TAG Writing     : $OUT_TXT, $OUT_ERR, $OUT_XLSX"

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
PYTHONPATH=. python -m tools.profiling.compare_layers \
    "$POLARIS_CSV" "$HW_CSV" \
    --strip-singleton-dims --perf --summarize-by-signature \
    --xlsx "$OUT_XLSX" \
    >"$OUT_TXT" 2>"$OUT_ERR"

echo "$TAG === tail of summary ==="
tail -25 "$OUT_TXT"
echo "$TAG Done. XLSX: $OUT_XLSX"
