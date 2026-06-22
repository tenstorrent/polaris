#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Run from the tt-metal checkout root. Prereqs: see ../si_profiling_helpers/presets/README.md
source "$(dirname "$0")/../check_arch.sh"; require_arch wormhole_b0
source "$(dirname "$0")/../apply_patch.sh"
source "$(dirname "$0")/../hw_id.sh"
# Local-only tt-metal tweak (NOT upstreamed): drop VGG e2e inference_iter_count 10 -> 3 so the
# NOC-trace profiler pass does not run out of resources. Idempotent — safe to re-run.
apply_patch_if_needed "$(dirname "$0")/../patches/vgg_unet_iter_count.patch"
BOARD=$(hw_id_board)
HEAD=$(git rev-parse --short=7 HEAD)
RUNID=vggref-${HEAD}-${IRD_ARCH_NAME}${BOARD:+-$BOARD}-$(date +%y%m%d)
CMD=models/demos/vision/segmentation/vgg_unet/wormhole/tests/perf/test_e2e_performant.py::test_vgg_unet_e2e
require_ref_cmd "$CMD"
python ../si_profiling_helpers/run-ttnn-profiler.py --command "$CMD" --pytest --output-dir $RUNID --report-name $RUNID --disable-logging "$@"
rc=$?
hw_id_write "$RUNID"
exit "$rc"
