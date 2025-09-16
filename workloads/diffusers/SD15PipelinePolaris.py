#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import sys
import importlib

import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.graph.wl_graph import WorkloadGraph

# Ensure local module imports resolve when loaded as a standalone file
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

try:
    create_scheduler = importlib.import_module('workloads.diffusers.schedulers.factory').create_scheduler  # type: ignore[attr-defined]
    AutoencoderKLPolaris = getattr(importlib.import_module('workloads.diffusers.AutoencoderKLPolaris'), 'AutoencoderKLPolaris')  # type: ignore[assignment]
    _te_mod = importlib.import_module('workloads.diffusers.TextEncodersPolaris')
    CLIPTextModelPolaris = getattr(_te_mod, 'CLIPTextModelPolaris')  # type: ignore[assignment]
    CLIPTokenizerHost = getattr(_te_mod, 'CLIPTokenizerHost')  # type: ignore[assignment]
    ClassifierFreeGuidance = getattr(importlib.import_module('workloads.diffusers.ClassifierFreeGuidancePolaris'), 'ClassifierFreeGuidance')  # type: ignore[assignment]
    _cond_mod = importlib.import_module('workloads.diffusers.ConditioningPolaris')
    ControlNetPolaris = getattr(_cond_mod, 'ControlNetPolaris')  # type: ignore[assignment]
    T2IAdapterPolaris = getattr(_cond_mod, 'T2IAdapterPolaris')  # type: ignore[assignment]
except Exception:
    create_scheduler = importlib.import_module('schedulers.factory').create_scheduler  # type: ignore[attr-defined]
    AutoencoderKLPolaris = getattr(importlib.import_module('AutoencoderKLPolaris'), 'AutoencoderKLPolaris')  # type: ignore[assignment]
    _te_mod = importlib.import_module('TextEncodersPolaris')
    CLIPTextModelPolaris = getattr(_te_mod, 'CLIPTextModelPolaris')  # type: ignore[assignment]
    CLIPTokenizerHost = getattr(_te_mod, 'CLIPTokenizerHost')  # type: ignore[assignment]
    ClassifierFreeGuidance = getattr(importlib.import_module('ClassifierFreeGuidancePolaris'), 'ClassifierFreeGuidance')  # type: ignore[assignment]
    _cond_mod = importlib.import_module('ConditioningPolaris')
    ControlNetPolaris = getattr(_cond_mod, 'ControlNetPolaris')  # type: ignore[assignment]
    T2IAdapterPolaris = getattr(_cond_mod, 'T2IAdapterPolaris')  # type: ignore[assignment]
# Use a local UNet functional stub to avoid package-relative import issues


class SD15PipelinePolarisWorkload(SimNN.Module):
    """
    Stable Diffusion 1.5 pipeline (text-to-image) skeleton integrated with Polaris.
    Uses single CLIP text encoder and UNet conditioned via cross-attention.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        super().__init__()
        self.name = name
        self.cfg = cfg

        # Pipeline-level config
        self.bs = cfg.get('bs', 1)
        self.num_inference_steps = cfg.get('num_inference_steps', 20)
        self.guidance_scale = cfg.get('guidance_scale', 7.5)
        self.height = cfg.get('height', 512)
        self.width = cfg.get('width', 512)
        self.mode = cfg.get('mode', 'txt2img')
        self.img2img_strength = cfg.get('img2img_strength', 0.8)
        self.inpaint_mask_ratio = cfg.get('inpaint_mask_ratio', 0.5)

        # Component configs
        self.vae_config = self._get_vae_config(cfg)
        self.text_encoder_config = self._get_text_encoder_config(cfg)
        self.unet_config = self._get_unet_config(cfg)
        self.scheduler_config = self._get_scheduler_config(cfg)

        # Initialize components
        self._initialize_components()

        # Graph
        self.workload_graph = WorkloadGraph(self.name)

        INFO(f"Initialized SD15 Pipeline Workload: {name}")
        INFO(f"Configuration: bs={self.bs}, steps={self.num_inference_steps}, guidance={self.guidance_scale}")
        INFO(f"Output size: {self.height}x{self.width}")

    def _get_vae_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'in_channels': cfg.get('vae_in_channels', 3),
            'latent_channels': cfg.get('vae_latent_channels', 4),
            'sample_size': cfg.get('vae_sample_size', 256),
            'block_out_channels': cfg.get('vae_block_out_channels', [128, 256, 512, 512]),
            'layers_per_block': cfg.get('vae_layers_per_block', 2),
            'scaling_factor': cfg.get('vae_scaling_factor', 0.18215),
            'norm_num_groups': cfg.get('vae_norm_num_groups', 32),
            'act_fn': cfg.get('vae_act_fn', 'silu'),
        }

    def _get_text_encoder_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'vocab_size': cfg.get('clip_vocab_size', 49408),
            'max_seq_length': cfg.get('clip_max_seq_length', 77),
            'hidden_size': cfg.get('clip_hidden_size', 768),
            'intermediate_size': cfg.get('clip_intermediate_size', 3072),
            'num_attention_heads': cfg.get('clip_num_attention_heads', 12),
            'num_hidden_layers': cfg.get('clip_num_hidden_layers', 6),
            'max_position_embeddings': cfg.get('clip_max_position_embeddings', 77),
        }

    def _get_unet_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'sample_size': cfg.get('unet_sample_size', 64),
            'in_channels': cfg.get('unet_in_channels', 4),
            'out_channels': cfg.get('unet_out_channels', 4),
            'cross_attention_dim': cfg.get('cross_attention_dim', 768),
            'block_out_channels': cfg.get('block_out_channels', [320, 640, 1280, 1280]),
            'down_block_types': cfg.get('down_block_types', [
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
            ]),
            'up_block_types': cfg.get('up_block_types', [
                "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
            ]),
            'transformer_layers_per_block': cfg.get('transformer_layers_per_block', [1, 2, 2, 2]),
            'attention_head_dim': cfg.get('attention_head_dim', [4, 8, 8, 8]),
            'layers_per_block': cfg.get('layers_per_block', 2),
            'act_fn': cfg.get('act_fn', 'silu'),
            'norm_num_groups': cfg.get('norm_num_groups', 32),
        }

    def _get_scheduler_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'num_train_timesteps': cfg.get('num_train_timesteps', 1000),
            'beta_start': cfg.get('beta_start', 0.0001),
            'beta_end': cfg.get('beta_end', 0.02),
            'beta_schedule': cfg.get('beta_schedule', 'scaled_linear'),
            'prediction_type': cfg.get('prediction_type', 'epsilon'),
            'timestep_spacing': cfg.get('timestep_spacing', 'leading'),
        }

    def _initialize_components(self) -> None:
        self.vae = AutoencoderKLPolaris("vae", self.vae_config)
        self.text_encoder = CLIPTextModelPolaris("text_encoder", self.text_encoder_config)
        self.tokenizer = CLIPTokenizerHost()
        # Reuse SDXL UNet functional stub for SD15 (dimension config differs via cfg)
        _UnetType: Any
        try:
            _UnetType = getattr(importlib.import_module('workloads.diffusers.SDXLPipelinePolaris'), 'UNet2DConditionModelPolaris')
        except Exception:
            _UnetType = getattr(importlib.import_module('SDXLPipelinePolaris'), 'UNet2DConditionModelPolaris')
        # Support both stub signatures: (name, cfg) and (onnx_path, **kwargs)
        try:
            self.unet = _UnetType("unet", self.unet_config)  # type: ignore[arg-type]
        except TypeError:
            self.unet = _UnetType("", **self.unet_config)

        scheduler_name = self.cfg.get('scheduler', 'euler')
        self.scheduler = create_scheduler(scheduler_name, **self.scheduler_config)
        self.guidance = ClassifierFreeGuidance(guidance_scale=self.guidance_scale)

        # Optional conditioning modules
        self.conditioning_type = str(self.cfg.get('conditioning', 'none')).lower()
        self.conditioning_cfg = self.cfg.get('conditioning_cfg', {'conditioning_strength': 1.0})
        self.conditioner: Optional[Any] = None
        if self.conditioning_type == 'controlnet':
            self.conditioner = ControlNetPolaris("controlnet", self.conditioning_cfg)
        elif self.conditioning_type in ('t2i_adapter', 't2i-adapter'):
            self.conditioner = T2IAdapterPolaris("t2i_adapter", self.conditioning_cfg)

    def generate_workload_graph(self, prompt: str = "a photo of a cat") -> WorkloadGraph:
        INFO(f"Generating SD15 workload graph for prompt: '{prompt}'")
        self.workload_graph = WorkloadGraph(self.name + "_graph")

        batch_size = self.bs
        latent_shape = [batch_size, 4, self.height // 8, self.width // 8]

        # 1) Text encoding
        text_inputs = self._add_text_encoding(prompt)

        # 2) Latent init (mode-dependent)
        mode = str(self.mode).lower()
        if mode == 'img2img':
            latents = self._add_img2img_latents(latent_shape, self.img2img_strength)
        elif mode == 'inpaint':
            latents = self._add_inpaint_latents(latent_shape, self.inpaint_mask_ratio)
        else:
            latents = self._add_latent_initialization(latent_shape)

        # 3) Denoising loop
        latents = self._add_denoising_loop(latents, text_inputs)

        # 4) Decode
        images = self._add_vae_decoding(latents)

        # 5) Output
        self._add_output_processing(images)
        return self.workload_graph

    def _add_text_encoding(self, prompt: str) -> Dict[str, SimTensor]:
        tokens = self.tokenizer(prompt, return_tensors="np")
        input_ids_tensor = F._from_data(f"{self.name}_input_ids", data=tokens['input_ids'], is_const=True)
        attention_mask_tensor = F._from_data(f"{self.name}_attention_mask", data=tokens['attention_mask'], is_const=True)
        input_ids_tensor.set_module(self)
        attention_mask_tensor.set_module(self)

        text_embeddings = self.text_encoder.add_call(
            self.workload_graph,
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )

        return {
            'text_embeddings': text_embeddings,
        }

    def _add_latent_initialization(self, latent_shape: List[int]) -> SimTensor:
        noise = F._from_data(f"{self.name}_noise", data=np.random.randn(*latent_shape).astype(np.float32))
        noise.set_module(self)
        init_sigma = F._from_data(f"{self.name}_init_sigma", data=np.array(1.0, dtype=np.float32), is_const=True)
        init_sigma.set_module(self)
        mul_op = F.Mul(f"{self.name}_init_mul")
        mul_op.set_module(self)
        return mul_op(noise, init_sigma)

    def _add_denoising_loop(self, latents: SimTensor, text_inputs: Dict[str, SimTensor]) -> SimTensor:
        self.scheduler.set_timesteps(self.num_inference_steps)
        for step, timestep in enumerate(self.scheduler.timesteps):
            # scale input
            scale_val = self.scheduler.scale_model_input(latents, timestep)
            scale_const = F._from_data(f"{self.name}_scale_{step}", data=np.array(scale_val, dtype=np.float32), is_const=True)
            scale_const.set_module(self)
            scale_op = F.Mul(f"{self.name}_mul_{step}")
            scale_op.set_module(self)
            scaled_latents = scale_op(latents, scale_const)

            # Optional conditioning
            cond_feat = None
            if self.conditioner is not None:
                cond_feat = self.conditioner.add_call(
                    self.workload_graph,
                    sample=scaled_latents,
                    encoder_hidden_states=text_inputs['text_embeddings']
                )

            # UNet (pass conditioning stub)
            noise_pred = self.unet.add_call(
                self.workload_graph,
                sample=scaled_latents,
                timestep=timestep,
                encoder_hidden_states=text_inputs['text_embeddings'],
                conditioning=cond_feat
            )

            # Guidance (placeholder - no-op in functional stub)

            # Scheduler step
            step_params = self.scheduler.step(noise_pred, timestep, latents)
            gamma_const = F._from_data(f"{self.name}_gamma_{step}", data=np.array(step_params['gamma'], dtype=np.float32), is_const=True)
            gamma_const.set_module(self)
            mul = F.Mul(f"{self.name}_upd_mul_{step}")
            mul.set_module(self)
            upd = mul(noise_pred, gamma_const)
            add = F.Add(f"{self.name}_add_{step}")
            add.set_module(self)
            latents = add(latents, upd)

        return latents

    def _add_vae_decoding(self, latents: SimTensor) -> SimTensor:
        scale = 1.0 / float(self.vae_config.get('scaling_factor', 0.18215))
        scaling_factor = F._from_data(f"{self.name}_vae_scale", data=np.array(scale, dtype=np.float32), is_const=True)
        scaling_factor.set_module(self)
        mul = F.Mul(f"{self.name}_vae_scale_mul")
        mul.set_module(self)
        scaled_latents = mul(latents, scaling_factor)
        return self.vae.add_decode_call(self.workload_graph, scaled_latents)

    def _add_img2img_latents(self, latent_shape: List[int], strength: float) -> SimTensor:
        init_latents = F._from_data(
            f"{self.name}_init_latents",
            data=(0.5 * np.random.randn(*latent_shape)).astype(np.float32),
        )
        init_latents.set_module(self)
        noise = F._from_data(f"{self.name}_img2img_noise", data=np.random.randn(*latent_shape).astype(np.float32))
        noise.set_module(self)
        s_const = F._from_data(f"{self.name}_img2img_strength", data=np.array(float(strength), dtype=np.float32), is_const=True)
        s_const.set_module(self)
        one_minus = F._from_data(f"{self.name}_img2img_1ms", data=np.array(1.0 - float(strength), dtype=np.float32), is_const=True)
        one_minus.set_module(self)
        mul_s = F.Mul(f"{self.name}_img2img_mul_s")
        mul_s.set_module(self)
        mul_1m = F.Mul(f"{self.name}_img2img_mul_1m")
        mul_1m.set_module(self)
        add = F.Add(f"{self.name}_img2img_add")
        add.set_module(self)
        return add(mul_s(noise, s_const), mul_1m(init_latents, one_minus))

    def _add_inpaint_latents(self, latent_shape: List[int], mask_ratio: float) -> SimTensor:
        init_latents = F._from_data(
            f"{self.name}_inpaint_init_latents",
            data=(0.5 * np.random.randn(*latent_shape)).astype(np.float32),
        )
        init_latents.set_module(self)
        noise = F._from_data(f"{self.name}_inpaint_noise", data=np.random.randn(*latent_shape).astype(np.float32))
        noise.set_module(self)
        mask_np = (np.random.rand(*latent_shape) < float(mask_ratio)).astype(np.float32)
        mask = F._from_data(f"{self.name}_inpaint_mask", data=mask_np, is_const=True)
        mask.set_module(self)
        one = F._from_data(f"{self.name}_inpaint_one", data=np.array(1.0, dtype=np.float32), is_const=True)
        one.set_module(self)
        inv = F.Sub(f"{self.name}_inpaint_sub")
        inv.set_module(self)
        inv_mask = inv(one, mask)
        mul_m = F.Mul(f"{self.name}_inpaint_mul_m")
        mul_m.set_module(self)
        mul_im = F.Mul(f"{self.name}_inpaint_mul_im")
        mul_im.set_module(self)
        add = F.Add(f"{self.name}_inpaint_add")
        add.set_module(self)
        return add(mul_m(noise, mask), mul_im(init_latents, inv_mask))

    def _add_output_processing(self, images: SimTensor) -> None:
        self.final_output = F.Sigmoid('output_activation')(images)

    def __call__(self):
        self.generate_workload_graph()
        return getattr(self, 'final_output', None)

    def analytical_param_count(self, lvl: int = 0) -> int:
        # Rough estimate
        return 0

    # TTSIM expected integration hooks
    def create_input_tensors(self) -> None:
        # No prebuilt tensors required; pipeline builds them during graph generation
        return None

    def get_forward_graph(self) -> WorkloadGraph:
        return self.workload_graph


def create_sd15_workload(name: str, cfg: Dict[str, Any]) -> SD15PipelinePolarisWorkload:
    return SD15PipelinePolarisWorkload(name, cfg)


# Logging
import logging
LOG = logging.getLogger(__name__)
INFO = LOG.info

# Alias expected by Polaris loader
SD15PipelinePolaris = SD15PipelinePolarisWorkload


