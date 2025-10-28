#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .math    import register_math_ops
from .logical import register_logical_ops
from .nn      import register_nn_ops
from .tensor  import register_tensor_ops
from .custom  import register_custom_ops

first_time = True

def initialize_op_desc():
    global first_time
    if first_time:
        register_math_ops()
        register_logical_ops()
        register_nn_ops()
        register_tensor_ops()
        register_custom_ops()
        first_time = False

