#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import os
import workloads.basiclenet as basiclenet

def test_lenet(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    basiclenet.run_standalone(outdir=output_dir)
