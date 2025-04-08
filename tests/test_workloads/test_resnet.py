#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import os
import workloads.basicresnet as basicresnet

def test_resnet(session_temp_directory):
    output_dir = str(session_temp_directory)
    os.makedirs(output_dir, exist_ok=True)
    basicresnet.run_standalone(outdir=output_dir)