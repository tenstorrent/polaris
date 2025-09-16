#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Type

from .base import SchedulerBase
from .euler_discrete import EulerDiscreteScheduler
from .ddim import DDIMScheduler
from .heun_discrete import HeunDiscreteScheduler
from .lms_discrete import LMSDiscreteScheduler
from .dpm_multistep import DPMSolverMultistepScheduler


_NAME_TO_CLASS: Dict[str, Type[SchedulerBase]] = {
    # Short aliases
    "euler": EulerDiscreteScheduler,
    "ddim": DDIMScheduler,
    "heun": HeunDiscreteScheduler,
    "lms": LMSDiscreteScheduler,
    "dpmms": DPMSolverMultistepScheduler,
    "dpm_multistep": DPMSolverMultistepScheduler,
    "dpmmultistep": DPMSolverMultistepScheduler,
}


def create_scheduler(name: str, **config) -> SchedulerBase:
    """Factory to create a scheduler instance by name.

    Args:
        name: Scheduler name or class name (e.g., "euler", "DDIMScheduler").
        **config: Configuration parameters passed to the scheduler constructor.

    Returns:
        An instance of the requested scheduler.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """

    key = (name or "").strip()
    if not key:
        key = "euler"
    # Normalize: remove spaces/underscores/dashes and lowercase
    norm = key.replace("_", "").replace("-", "").lower()

    if norm in _NAME_TO_CLASS:
        return _NAME_TO_CLASS[norm](**config)

    # Try exact class name match ignoring case
    for cls in (EulerDiscreteScheduler, DDIMScheduler, HeunDiscreteScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler):
        if cls.__name__.replace("_", "").replace("-", "").lower() == norm:
            return cls(**config)

    raise ValueError(f"Unknown scheduler '{name}'. Available: euler, ddim, heun, lms, dpmms")


