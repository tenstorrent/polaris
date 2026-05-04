#!/usr/bin/env bash
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Correlate Polaris vs Blackhole p100a HW profiler for optimized sharded ViT.
# Pass --regen to re-run polproj first.
exec "$(dirname "${BASH_SOURCE[0]}")/correlate_perf.sh" bh \
    lfc://hlm-refrun/p100a_vit_opt_sharded_bh_refrun_260426.csv "$@"
