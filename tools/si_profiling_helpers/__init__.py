# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""SI profiling helpers: hardware-run preset scripts, the Tracy profiler wrapper,
and the three-CSV merge tool (co-located so a single rsync of this directory to a
hardware node carries everything needed to capture and merge a run on-device)."""
