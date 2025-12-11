#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .math       import register_math_ops
from .logical    import register_logical_ops
from .nn         import register_nn_ops
from .tensor     import register_tensor_ops
from .custom     import register_custom_ops
from .reduction  import register_reduction_ops
from .misc       import (
        register_quantization_ops,
        register_optional_ops,
        register_object_detection_ops,
        register_image_ops,
        register_text_ops,
)
from .sequence   import register_sequence_ops
from .einsum     import register_einsum_ops
from .attention  import register_attention_ops
from .qattention import register_qattention_ops

first_time = True

def initialize_op_desc():
    global first_time
    if first_time:
        register_math_ops()
        register_logical_ops()
        register_nn_ops()
        register_tensor_ops()
        register_custom_ops()
        register_reduction_ops()
        # Additional operator groups
        register_quantization_ops()
        register_optional_ops()
        register_object_detection_ops()
        register_image_ops()
        register_text_ops()
        register_sequence_ops()
        # Specific high-level ops
        register_einsum_ops()
        register_attention_ops()
        register_qattention_ops()
        first_time = False
