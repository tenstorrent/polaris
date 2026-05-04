#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Correlate Polaris vs Wormhole n150 HW profiler for optimized sharded ViT.
# Pass --regen to re-run polproj first.
exec "$(dirname "${BASH_SOURCE[0]}")/correlate_perf.sh" wh \
    lfc://hlm-refrun/n150_vit_opt_sharded_wh_refrun_260415.csv "$@"
