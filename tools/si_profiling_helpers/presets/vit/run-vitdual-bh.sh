#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
source "$(dirname "$0")/../check_arch.sh"; require_arch blackhole
source "$(dirname "$0")/../apply_patch.sh"
source "$(dirname "$0")/../hw_id.sh"
# Local-only tt-metal tweak (NOT upstreamed): the dual BH ViT HW path imports tt-metal's
# vit_test_infra, which the patch updates to thread device= into update_model_config (BH 110-core
# firmware grid workaround). Idempotent — safe to re-run. See firmware issue #38877.
apply_patch_if_needed "$(dirname "$0")/../patches/bh_vit_core_grid_fix.patch"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=vitdual-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
CMD=workloads/ttnn/vit/bh/test_vit_device_perf_bh.py
require_dual_cmd "$CMD"
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --output-dir $RUNID --report-name $RUNID "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
