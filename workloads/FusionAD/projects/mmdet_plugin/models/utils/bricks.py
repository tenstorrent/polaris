
"""
TTSIM conversion of bricks.py - preserves all logic, inputs, and outputs.

CONVERSION NOTES:
-----------------
- Removed `import torch` (no longer needed)
- Removed `torch.cuda.synchronize()` calls (no CUDA in ttsim; ttsim execution is synchronous)
- All other logic, inputs, and outputs are preserved exactly as-is
"""
# =============================================================================
# PYTORCH
# =============================================================================
# import functools
# import logging
# import time
# from collections import defaultdict
# import torch
# time_maps = defaultdict(lambda :0.)
# count_maps = defaultdict(lambda :0.)
# def run_time(name):
#     def middle(fn):
#         def wrapper(*args, **kwargs):
#             torch.cuda.synchronize()
#             start = time.time()
#             res = fn(*args, **kwargs)
#             torch.cuda.synchronize()
#             time_maps['%s : %s'%(name, fn.__name__) ] += time.time()-start
#             count_maps['%s : %s'%(name, fn.__name__) ] +=1
#             logging.info("%s : %s takes up %f "% (name, fn.__name__,time_maps['%s : %s'%(name, fn.__name__) ] /count_maps['%s : %s'%(name, fn.__name__) ] ))
#             return res
#         return wrapper
#     return middle
#

# =============================================================================
# TTsim
# =============================================================================


#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import time
from collections import defaultdict

time_maps = defaultdict(lambda: 0.)
count_maps = defaultdict(lambda: 0.)


def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            # ttsim execution is synchronous — no cuda.synchronize() needed
            start = time.time()
            res = fn(*args, **kwargs)
            # ttsim execution is synchronous — no cuda.synchronize() needed
            time_maps['%s : %s' % (name, fn.__name__)] += time.time() - start
            count_maps['%s : %s' % (name, fn.__name__)] += 1
            logging.info(
                "%s : %s takes up %f " % (
                    name,
                    fn.__name__,
                    time_maps['%s : %s' % (name, fn.__name__)] / count_maps['%s : %s' % (name, fn.__name__)]
                )
            )
            return res
        return wrapper
    return middle
