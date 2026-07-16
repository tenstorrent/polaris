#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
source "$(dirname "$0")/../check_arch.sh"; require_arch wormhole_b0
source "$(dirname "$0")/../hw_id.sh"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=vitdual-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
CMD=workloads/ttnn/vit/wh/test_vit_device_perf_wh.py
require_dual_cmd "$CMD"
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --output-dir $RUNID --report-name $RUNID --merge-variant trace_replay "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
