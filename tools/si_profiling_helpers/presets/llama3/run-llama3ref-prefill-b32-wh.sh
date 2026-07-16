#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
#
# ############################################################################################
# ⛔ TEMPORARY BLOCK — WH PREFILL CANNOT COMPLETE A FULL CAPTURE YET.
# The finalized reference command below fails NoC-trace capture on Wormhole in the *prefill*
# phase only (decode is fine on WH; both phases are fine on BH). This is a NoC-trace tooling
# issue that the user has raised and owns — it is NOT worked around by limiting the command
# (do NOT re-add --num_layers/--max_generated_tokens). Run this preset only after the NoC-trace
# issue is resolved; remove this block at that point. See presets/README.md capture matrix.
# ############################################################################################
# Enforce the block above until the NoC-trace issue is fixed: refuse to run by default so a
# known-failing long capture isn't started accidentally. Set ALLOW_WH_PREFILL=1 to deliberately
# override; remove this guard together with the block once the issue is resolved.
[ -n "${ALLOW_WH_PREFILL:-}" ] || { echo "⛔ WH prefill capture blocked (NoC-trace issue); set ALLOW_WH_PREFILL=1 to override." >&2; exit 2; }
source "$(dirname "$0")/../check_arch.sh"; require_arch wormhole_b0
source "$(dirname "$0")/../hw_id.sh"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=llama3-prefill-b32-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
# Finalized reference command: complete batch-32 run, prefill phase. All knobs other than the
# batch (from the -k marker) and the phase split (--mode) are left at the marked case's defaults.
# NOTE on quoting: the -k value must reach pytest as ONE token. tracy's report mode re-joins its
# argv with spaces and re-runs the command under `shell=True` (tt-metal tools/tracy/__main__.py),
# which would otherwise split the bare words of "performance and batch-32". The nested '"..."'
# keeps literal quotes on the token so it survives BOTH shlex.split here and tracy's shell re-parse.
CMD="models/tt_transformers/demo/simple_text_demo.py -k '\"performance and batch-32 and not log-probs\"' --mode prefill"
require_ref_cmd "$CMD"
export PYTEST_TIMEOUT=3600  # long device-perf capture+replay; avoid pytest-timeout abort
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --pytest --output-dir $RUNID --report-name $RUNID --disable-logging --merge-variant trace_replay "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
