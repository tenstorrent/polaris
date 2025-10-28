#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from .op import SimOp
from .tensor import SimTensor
from .desc import initialize_op_desc

initialize_op_desc()
