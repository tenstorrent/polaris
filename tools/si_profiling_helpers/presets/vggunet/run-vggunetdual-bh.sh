#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
source "$(dirname "$0")/../check_arch.sh"; require_arch blackhole
source "$(dirname "$0")/../hw_id.sh"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=vggdual-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
CMD=workloads/ttnn/vgg_unet/bh/test_vgg_unet_device_perf_bh.py
require_dual_cmd "$CMD"
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --output-dir $RUNID --report-name $RUNID --disable-logging "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
