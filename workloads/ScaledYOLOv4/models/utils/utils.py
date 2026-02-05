#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Torch utilities - TTSim compatible version"""

import time
import numpy as np
from loguru import logger


def init_seeds(seed=0):
    np.random.seed(seed)


def time_synchronized():
    return time.time()


def select_device(device=""):
    return "cpu"


def is_parallel(model):
    return False


def initialize_weights(model):
    pass


def model_info(model):
    pass


def fuse_conv_and_bn(conv_op, bn_op):
    return conv_op


def scale_img(img, ratio=1.0, same_shape=False):
    if ratio == 1.0:
        return img
    else:
        raise NotImplementedError(
            "Image scaling not implemented in ttsim. "
            "Please pre-scale images before passing to the model."
        )


def find_modules(model, mclass):
    return []


def sparsity(model):
    return 0.0


def prune(model, amount=0.3):
    raise NotImplementedError("Model pruning not implemented in ttsim")


def intersect_dicts(da, db, exclude=()):
    return {
        k: v
        for k, v in da.items()
        if k in db
        and not any(x in k for x in exclude)
        and (not hasattr(v, "shape") or v.shape == db[k].shape)
    }


def load_classifier(name="resnet101", n=2):
    raise NotImplementedError("Pretrained classifier loading not implemented in ttsim")


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        logger.debug("WARNING: ModelEMA not fully implemented in ttsim")
        self.ema = model
        self.updates = updates
        self.decay = lambda x: decay * (1 - np.exp(-x / 2000))

    def update(self, model):
        """Update EMA parameters (not implemented)"""
        self.updates += 1

    def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
        """Update EMA attributes"""
        copy_attr(self.ema, model, include, exclude)
