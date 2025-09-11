#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for Polaris diffusion schedulers.

Tests scheduler implementations including Euler, DDIM, Heun, LMS, and DPM Multistep.
"""

import numpy as np

from workloads.diffusers.schedulers import (
    EulerDiscreteScheduler,
    DDIMScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    DPMSolverMultistepScheduler,
)


# def test_euler_ancestral_set_timesteps_and_scale():
#     sched = EulerAncestralDiscreteScheduler(num_train_timesteps=10)
#     sched.set_timesteps(num_inference_steps=4)
#     # Test commented out - EulerAncestralDiscreteScheduler not implemented yet

#     assert sched.num_inference_steps == 4
#     assert isinstance(sched.timesteps, np.ndarray)
#     assert len(sched.sigmas) == 4

#     t = sched.timesteps[0]
#     scale = sched.scale_model_input(sample=None, timestep=t)
#     assert isinstance(scale, float)
#     assert scale > 0.0


# def test_euler_ancestral_step_returns_params():
#     sched = EulerAncestralDiscreteScheduler(num_train_timesteps=10)
#     sched.set_timesteps(num_inference_steps=3)
#     t = sched.timesteps[0]
#     out = sched.step(model_output=None, timestep=t, sample=None)

#     assert isinstance(out, dict)
#     for key in ['sigma', 'sigma_next', 'gamma', 'prediction_type', 'is_ancestral', 'timestep']:
#         assert key in out
#     assert out['is_ancestral'] is True
#     # Test commented out - EulerAncestralDiscreteScheduler not implemented yet


def test_ddim_basic_step_params():
    sched = DDIMScheduler(num_train_timesteps=10, eta=0.0)
    sched.set_timesteps(num_inference_steps=4)
    t = int(sched.timesteps[0])
    out = sched.step(model_output=None, timestep=t, sample=None)

    for key in ['alpha_cum', 'alpha_cum_prev', 'eta', 'prediction_type', 'timestep']:
        assert key in out
    assert isinstance(out['alpha_cum'], float)
    assert isinstance(out['alpha_cum_prev'], float)


def test_heun_basic_step_params():
    sched = HeunDiscreteScheduler(num_train_timesteps=10)
    sched.set_timesteps(num_inference_steps=4)
    t = sched.timesteps[0]
    out = sched.step(model_output=None, timestep=float(t), sample=None)

    for key in ['sigma', 'sigma_next', 'gamma', 'prediction_type', 'is_heun', 'order', 'timestep']:
        assert key in out
    assert out['is_heun'] is True


def test_lms_basic_step_params():
    sched = LMSDiscreteScheduler(num_train_timesteps=10)
    sched.set_timesteps(num_inference_steps=4)
    t = sched.timesteps[0]
    out = sched.step(model_output=None, timestep=float(t), sample=None)

    for key in ['sigma', 'sigma_next', 'gamma', 'prediction_type', 'is_lms', 'order', 'timestep']:
        assert key in out
    assert out['is_lms'] is True


def test_dpmms_skeleton():
    sched = DPMSolverMultistepScheduler(num_train_timesteps=10, solver_order=2)
    sched.set_timesteps(num_inference_steps=4)
    t = sched.timesteps[0]
    scale = sched.scale_model_input(None, t)
    assert scale == 1.0
    out = sched.step(model_output=None, timestep=float(t), sample=None)
    assert out['solver_order'] == 2
    assert 'prediction_type' in out and 'timestep' in out


