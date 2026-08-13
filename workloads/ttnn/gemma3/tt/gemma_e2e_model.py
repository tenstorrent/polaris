# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
from loguru import logger

import ttsim.front.ttnn as ttnn

from workloads.ttnn.gemma3.common.gemma_utils import is_blackhole
from workloads.ttnn.gemma3.tt.gemma_vision_model import TtGemmaTransformerVision
from workloads.ttnn.tt_transformers.generator import Generator
from workloads.ttnn.tt_transformers.model import Transformer


# ============================================================================
# Helper Functions
# ============================================================================
def _to_numpy(x):
    """Convert tensor to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _replicate_mapper(mesh_device):
    """Get ReplicateTensorToMesh mapper if available."""
    if hasattr(ttnn, "ReplicateTensorToMesh"):
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return None


def _as_ttnn_tensor(
    array,
    *,
    device=None,
    dtype,
    layout,
    memory_config,
    mesh_mapper=None,
):
    """
    Create ttnn tensor with fallback for API variations.

    Handles cases where mesh_mapper or device might not be supported
    in all ttsim versions.
    """
    kwargs = {
        "dtype": dtype,
        "layout": layout,
        "memory_config": memory_config,
    }
    if device is not None:
        kwargs["device"] = device
    if mesh_mapper is not None:
        kwargs["mesh_mapper"] = mesh_mapper

    try:
        return ttnn.as_tensor(array, **kwargs)
    except TypeError:
        # Fallback: remove unsupported kwargs
        kwargs.pop("mesh_mapper", None)
        tensor = ttnn.as_tensor(array, **kwargs)
        if device is not None and getattr(tensor, "device", None) is None:
            return ttnn.to_device(tensor, device, memory_config=memory_config)
        return tensor


def _softmax_np(x, axis=-1):
    """Numerically stable softmax in numpy."""
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _sample_top_p_numpy(probs, top_p):
    """
    Top-p (nucleus) sampling implementation in numpy.

    Args:
        probs: Probability distribution of shape [B, V] or [V]
        top_p: Cumulative probability threshold

    Returns:
        Sampled token indices of shape [B, 1]
    """
    if probs.ndim == 1:
        probs = np.expand_dims(probs, axis=0)

    batch_size = probs.shape[0]
    results = np.zeros((batch_size, 1), dtype=np.int64)

    for b in range(batch_size):
        prob = probs[b]

        # Sort probabilities in descending order
        sorted_indices = np.argsort(prob)[::-1]
        sorted_probs = prob[sorted_indices]

        # Compute cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs)

        # Find cutoff index where cumulative prob exceeds top_p
        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_probs))

        # Truncate and renormalize
        truncated_probs = sorted_probs[:cutoff_idx].copy()
        truncated_indices = sorted_indices[:cutoff_idx]

        denom = np.sum(truncated_probs)
        if denom == 0:
            truncated_probs = np.ones_like(truncated_probs) / len(truncated_probs)
        else:
            truncated_probs = truncated_probs / denom

        # Sample from truncated distribution
        sampled_idx = np.random.choice(len(truncated_probs), p=truncated_probs)
        results[b, 0] = truncated_indices[sampled_idx]

    return results


# ============================================================================
# TtGemmaModel
# ============================================================================
class TtGemmaModel(Transformer):
    """
    Gemma multimodal model with vision encoder support.

    Extends the base Transformer with vision processing capabilities
    for multimodal inputs (text + images).
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__(
            args,
            dtype,
            mesh_device,
            state_dict,
            weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )

        # Get tt_ccl from parent if available
        # tt_ccl = getattr(self, "tt_ccl", None)

        self.vision_model = TtGemmaTransformerVision(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="model.vision_tower.vision_model.",
            dtype=dtype,
            configuration=args,
            weight_cache_path=weight_cache_path,
        )

        self._configure_on_device_sampling_support()

    def _configure_on_device_sampling_support(self):
        """
        Configure on-device sampling support.

        For Polaris simulation, we disable on-device sampling and use
        host-based sampling instead.
        """
        self._supports_on_device_sampling = False
        self.sampling = None

    def encode_vision_embeddings_from_pixels(self, pixel_values):
        """
        Run only the vision tower and return host patch embeddings for image token positions.

        Args:
            pixel_values: Input image tensor(s)

        Returns:
            Vision embeddings as numpy array
        """
        vision_output = self.compute_vision_token(pixel_values)

        if is_blackhole():
            # BH: vision hidden dim is tensor-parallel sharded;
            # match embd readout (dim=-1) for multi-chip (e.g. P150x4).
            comp_vision_output = _to_numpy(
                ttnn.to_torch( # type: ignore[call-arg]
                    vision_output,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1), # type: ignore[attr-defined]
                )
            )
            comp_vision_output = comp_vision_output[: int(vision_output.shape[0])]
            if comp_vision_output.shape[-1] > self.args.dim:
                comp_vision_output = comp_vision_output[..., : self.args.dim]
        else:
            comp_vision_output = _to_numpy(
                ttnn.to_torch( # type: ignore[call-arg]
                    vision_output,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0), # type: ignore[attr-defined]
                )
            )[: vision_output.shape[0], :]

        return np.squeeze(comp_vision_output, axis=0)

    def _vision_embeddings_to_tensor(self, vision_embeddings, batch_rows: int):
        """
        Coalesce vision embeddings into a single tensor.

        Args:
            vision_embeddings: Single array or list of arrays
            batch_rows: Number of batch rows expected

        Returns:
            Coalesced numpy array or None
        """
        if vision_embeddings is None:
            return None

        if isinstance(vision_embeddings, np.ndarray):
            return vision_embeddings

        if isinstance(vision_embeddings, (list, tuple)):
            parts = [v for v in vision_embeddings if v is not None]
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]

            if batch_rows == 1:
                first = parts[0]
                if first.ndim == 3 and first.shape[0] == 1:
                    return np.concatenate(parts, axis=1)
                if first.ndim == 2:
                    return np.concatenate(parts, axis=0)
                return np.concatenate(parts, axis=0)

            if len(parts) == batch_rows:
                stacked = np.stack(parts, axis=0)
                if stacked.ndim == 4 and stacked.shape[1] == 1:
                    stacked = np.squeeze(stacked, axis=1)
                return stacked

            raise ValueError(
                f"vision_embeddings list length {len(parts)} does not match prompt batch rows {batch_rows}"
            )

        raise TypeError(f"vision_embeddings must be ndarray or sequence of ndarrays, got {type(vision_embeddings)}")

    def _fuse_vision_into_text_embeddings(self, pt_tokens, tokens_embd, image_features):
        """
        Fuse vision embeddings into text embeddings at image token positions.

        Args:
            pt_tokens: Token IDs (numpy array)
            tokens_embd: Text embeddings (numpy array)
            image_features: Vision embeddings (numpy array)

        Returns:
            Fused embeddings (numpy array)
        """
        result = np.array(tokens_embd, copy=True)
        mask = pt_tokens == self.args.image_token_index

        features = np.asarray(image_features)
        if features.ndim == 3 and features.shape[0] == 1:
            features = features[0]
        features = features.reshape(-1, features.shape[-1])

        num_slots = int(mask.sum())
        if num_slots == 0:
            return result

        result[mask] = features[:num_slots].astype(result.dtype, copy=False)
        return result

    def prepare_inputs_prefill(self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, **kwargs):
        """
        Prepare inputs for prefill. Returns ttnn tensors on device.

        For multimodal prompts, pass ``vision_embeddings`` (host tensor or list of tensors from
        :meth:`encode_vision_embeddings_from_pixels` / ``encode_vision_for_prefill``).
        If only ``pixel_values`` is set, embeddings are computed here.

        Args:
            pt_tokens: Token IDs as numpy array
            start_pos: Starting position for RoPE
            page_table: Optional page table for paged attention
            chunk_page_table: Optional chunk page table
            **kwargs: Additional arguments including vision_embeddings, pixel_values

        Returns:
            Tuple of (tokens_embd, tt_rot_mats_global, tt_rot_mats_local, tt_page_table, tt_chunk_page_table)
        """
        if not isinstance(pt_tokens, np.ndarray):
            pt_tokens = np.asarray(pt_tokens, dtype=np.int64)

        S = pt_tokens.shape[-1]
        batch_rows = pt_tokens.shape[0]

        # Create token tensor
        tokens = _as_ttnn_tensor(
            pt_tokens.reshape(1, 1, 1, -1).astype(np.uint32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

        # Get embeddings
        tokens_embd = self.embd(tokens)
        tokens_embd = _to_numpy(
            ttnn.to_torch( # type: ignore[call-arg]
                tokens_embd,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1), # type: ignore[attr-defined]
            )
        )

        # Handle vision embeddings
        vision_embeddings = kwargs.pop("vision_embeddings", None)
        pixel_values = kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("image_sizes", None)

        if vision_embeddings is None and pixel_values is not None:
            pvs = pixel_values if isinstance(pixel_values, (list, tuple)) else [pixel_values]
            vision_embeddings = [
                self.encode_vision_embeddings_from_pixels(pv) if pv is not None else None for pv in pvs
            ]

        if vision_embeddings is not None:
            vision_embeddings = self._vision_embeddings_to_tensor(vision_embeddings, batch_rows)
            if vision_embeddings is not None:
                tokens_embd = self._fuse_vision_into_text_embeddings(pt_tokens, tokens_embd, vision_embeddings)

        # Prepare residual tensor
        tokens_embd = self.args.prepare_residual_tensor_prefill(tokens_embd)

        # Convert back to ttnn tensor
        tokens_embd = _as_ttnn_tensor(
            np.asarray(tokens_embd, dtype=np.float32),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        # Slice rotation matrices for prefill sequence length
        assert (
            self.rope_setup.cos_matrix_prefill.shape[2] >= start_pos + S # type: ignore[attr-defined]
        ), f"Padded prefill end idx {start_pos + S} exceeds max seq len {self.rope_setup.cos_matrix_prefill.shape[2]}" # type: ignore[attr-defined]

        tt_rot_mats_prefill_global = [
            self.rope_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :], # type: ignore[attr-defined]
            self.rope_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :], # type: ignore[attr-defined]
        ]

        tt_rot_mats_prefill_local = [
            self.rope_local_setup.cos_matrix_prefill[:, :, start_pos : start_pos + S, :], # type: ignore[attr-defined]
            self.rope_local_setup.sin_matrix_prefill[:, :, start_pos : start_pos + S, :], # type: ignore[attr-defined]
        ]

        # Handle page tables
        if page_table is not None:
            if not isinstance(page_table, np.ndarray):
                page_table = np.asarray(page_table, dtype=np.int32)
            tt_page_table = _as_ttnn_tensor(
                page_table.astype(np.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            if not isinstance(chunk_page_table, np.ndarray):
                chunk_page_table = np.asarray(chunk_page_table, dtype=np.int32)
            tt_chunk_page_table = _as_ttnn_tensor(
                chunk_page_table.astype(np.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return (
            tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
        )

    def compute_vision_token(self, pixel_values, batch_size=3):
        """
        Process vision tokens in batches to avoid OOM for large number of images.

        Args:
            pixel_values: numpy array of shape (B, C, H, W) or list of such arrays
            batch_size: Number of images to process in one batch (max 3)

        Returns:
            Combined vision output tensor
        """
        assert 0 < batch_size <= 3, "Device runs OOM with batch size > 3"

        if not isinstance(pixel_values, list):
            pixel_values = [pixel_values]

        pixel_values_batches = []
        total_num_images = 0

        for image in pixel_values:
            if not isinstance(image, np.ndarray):
                image = np.asarray(image)

            num_images = image.shape[0]
            total_num_images += num_images

            if num_images < batch_size:
                pixel_values_batches.append(image)
            else:
                # Split large batches
                for i in range(0, num_images, batch_size):
                    end_idx = min(i + batch_size, num_images)
                    pixel_values_batches.append(image[i:end_idx])

        logger.info(f"Starting vision encoder for {total_num_images} image(s) in {len(pixel_values_batches)} batch(es)")

        vision_outputs = []
        for batch_idx, batch_pixel_values in enumerate(pixel_values_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(pixel_values_batches)}")
            batch_vision_output = self.vision_model(batch_pixel_values)
            vision_outputs.append(batch_vision_output)

        # Combine all vision outputs along the batch dimension
        combined_vision_output = ttnn.concat(vision_outputs, dim=1)
        logger.info("Vision encoder done")

        return combined_vision_output

    @staticmethod
    def sample_host(tt_input, temperature=0.6, top_p=0.08, on_host=True):
        """
        Sample from logits on host using numpy.

        Args:
            tt_input: Input logits tensor or numpy array
            temperature: Sampling temperature (0 for greedy)
            top_p: Top-p (nucleus) sampling threshold
            on_host: Whether to perform sampling on host (always True for simulation)

        Returns:
            Tuple of (None, output tokens as numpy array)
        """
        if isinstance(tt_input, np.ndarray):
            pt_input = tt_input
        else:
            pt_input = _to_numpy(ttnn.to_torch(tt_input))

        vocab_size = pt_input.shape[-1]
        pt_input = pt_input[..., :vocab_size]

        # [B, 1, V] -> [B, V] for correct softmax / top-p / argmax
        if pt_input.ndim == 3 and pt_input.shape[1] == 1:
            pt_input = np.squeeze(pt_input, axis=1)

        if temperature > 0:
            probs = _softmax_np(pt_input / temperature, axis=-1)
            pt_out = _sample_top_p_numpy(probs, top_p)
        else:
            pt_out = np.argmax(pt_input, axis=-1, keepdims=True)

        # Ensure consistent output shape
        if pt_out.ndim == 0:
            pt_out = np.expand_dims(pt_out, axis=0)
        elif pt_out.ndim == 1 and pt_input.ndim >= 2 and pt_input.shape[0] > 1:
            pass  # [B] next tokens - keep as is
        elif pt_out.ndim == 1:
            pt_out = np.expand_dims(pt_out, axis=0)

        return None, pt_out


# ============================================================================
# GemmaMultimodalGenerator
# ============================================================================
class GemmaMultimodalGenerator(Generator):
    """
    Generator for Gemma multimodal models.

    Extends the base Generator with support for vision inputs,
    handling both text-only and multimodal (text + image) prompts.
    """

    def encode_vision_for_prefill(self, pixel_values: list):
        """
        Encode vision inputs for prefill.

        Args:
            pixel_values: List of pixel value tensors/arrays

        Returns:
            List of vision embeddings (one per input image)
        """
        if not hasattr(self.model[0], "encode_vision_embeddings_from_pixels"):
            raise TypeError(
                "GemmaMultimodalGenerator requires TtGemmaModel (multimodal). "
                "text_demo uses tt_transformers.Generator with a plain Transformer."
            )
        return [
            self.model[0].encode_vision_embeddings_from_pixels(pv) if pv is not None else None
            for pv in pixel_values
        ]

    def _prepare_multimodal_prefill_kwargs(self, **kwargs):
        """
        Prepare kwargs for multimodal prefill.

        Converts pixel_values to vision_embeddings if needed.
        """
        if kwargs.get("vision_embeddings") is None and kwargs.get("pixel_values") is not None:
            kwargs = dict(kwargs)
            kwargs["vision_embeddings"] = self.encode_vision_for_prefill(kwargs["pixel_values"])
            kwargs.pop("pixel_values", None)
        return kwargs

    def prefill_forward_multimodal(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos=None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        """
        Forward pass for multimodal prefill.

        Prepares vision embeddings and delegates to parent's text prefill.
        """
        kwargs = self._prepare_multimodal_prefill_kwargs(**kwargs)
        return super().prefill_forward_text(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )

    def prefill_forward(
        self,
        vision_images,
        vision_masks,
        tokens,
        xattn_caches,
        total_lens,
        prompt_lens,
        page_table=None,
        kv_cache=None,
        cross_page_table=None,
        empty_slots=None,
        **kwargs,
    ):
        """
        Forward pass with vision images.

        This is the main entry point for multimodal prefill with images.
        """
        # Unused parameters
        del vision_masks, xattn_caches, total_lens, cross_page_table

        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            pixel_values=vision_images,
            **kwargs,
        )

    def prefill_forward_text(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos=None,
        return_hidden_states=False,
        warmup_prefill=True,
        **kwargs,
    ):
        """
        Forward pass for text-only prefill.

        Delegates to multimodal prefill (which handles the text-only case).
        """
        return self.prefill_forward_multimodal(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            enable_trace=enable_trace,
            model_id_warmup=model_id_warmup,
            sampling_params=sampling_params,
            start_pos=start_pos,
            return_hidden_states=return_hidden_states,
            warmup_prefill=warmup_prefill,
            **kwargs,
        )