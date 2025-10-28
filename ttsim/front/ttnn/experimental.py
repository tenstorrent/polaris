#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .op import single_output_immediate_op

#TTNN EXPERIMENTAL
dropout = single_output_immediate_op('Dropout')
gather  = single_output_immediate_op('Gather')
